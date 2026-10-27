//! The TP ranks' collectives: NCCL opened at run time, its unique id passed from rank 0 over the link. Every model
//! reduction is an all-gather of fp32 partials that each rank then adds in rank order (DESIGN.md section 3), so the
//! ranks hold the same bits without a broadcast.
const std = @import("std");
const cuda = @import("cuda");
const Link = @import("link.zig").Link;

const nccl = cuda.nccl;

/// TF_DS_TWO_HOP=1 (model.zig sets it, every rank alike; off by default): four nodes cabled as a ring, where node g
/// reaches only its column partner g ^ 2 and its row partner g ^ 1 (no cross links between the two pairs). A world-4
/// all-gather then goes as two hops of point-to-point exchanges between direct partners, never along NCCL's own ring:
/// the column partner's slice first, then both slices with the row partner. The same bytes in the same places.
pub var two_hop: bool = false;

/// One point-to-point step of a two-hop all-gather: the slots (rank indices) this rank sends to `peer`, then the
/// slots it receives from it, in posting order (NCCL matches a pair's sends and receives in order).
pub const Hop = struct { peer: u32, send: [2]u32, recv: [2]u32, n: usize };

/// Rank g's two hops on four ranks (g ^ 2, then g ^ 1).
pub fn hops(g: u32) [2]Hop {
    return .{
        .{ .peer = g ^ 2, .send = .{ g, 0 }, .recv = .{ g ^ 2, 0 }, .n = 1 },
        .{ .peer = g ^ 1, .send = .{ g, g ^ 2 }, .recv = .{ g ^ 1, g ^ 3 }, .n = 2 },
    };
}

fn size(t: nccl.DataType) usize {
    return switch (t) {
        .i8, .u8 => 1,
        .f16, .bf16 => 2,
        .i32, .u32 => 4,
        .i64, .u64, .f64 => 8,
        .f32 => 4,
    };
}

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        c.* = undefined;
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        if (two_hop and c.world == 4) return c.allGather2(send, recv, count, dtype, stream);
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle), "ncclAllGather");
    }

    /// allGather on a ring of four (two_hop): this rank's slice into its place, then the two hops.
    fn allGather2(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        const b = count * size(dtype);
        const own = recv + c.rank * b;
        if (send != own and b > 0) try stream.d.check(stream.d.api.cuMemcpyDtoDAsync_v2(own, send, b, stream.handle), "cuMemcpyDtoDAsync");
        for (hops(c.rank)) |h| {
            try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
            var rs: [4]nccl.Result = undefined;
            var k: usize = 0;
            for (h.send[0..h.n]) |q| {
                rs[k] = c.lib.api.ncclSend(recv + q * b, count, dtype, @intCast(h.peer), c.comm, stream.handle);
                k += 1;
            }
            for (h.recv[0..h.n]) |q| {
                rs[k] = c.lib.api.ncclRecv(recv + q * b, count, dtype, @intCast(h.peer), c.comm, stream.handle);
                k += 1;
            }
            try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
            for (rs[0..k]) |x| try c.lib.check(x, "ncclSend/ncclRecv");
        }
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        const s = c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle);
        const r = c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle);
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
    }
};

test "two hops on a ring of four: every rank ends with every slice in rank order, partners only" {
    // four ranks' recv buffers, slot q holding rank q's slice once it arrived (0: not yet)
    var have: [4][4]u8 = @splat(@splat(0));
    for (0..4) |g| have[g][g] = @intCast(g + 1);
    for (0..2) |step| {
        var next = have;
        for (0..4) |g| {
            const h = hops(@intCast(g))[step];
            try std.testing.expect(h.peer == g ^ 1 or h.peer == g ^ 2); // a direct partner, never g ^ 3
            const back = hops(h.peer)[step];
            try std.testing.expectEqual(@as(u32, @intCast(g)), back.peer);
            try std.testing.expectEqual(h.n, back.n);
            // what g posts to receive is, in order, what its peer posts to send, and it already holds them
            for (0..h.n) |i| {
                try std.testing.expectEqual(h.recv[i], back.send[i]);
                try std.testing.expect(have[h.peer][back.send[i]] != 0);
                next[g][h.recv[i]] = have[h.peer][back.send[i]];
            }
        }
        have = next;
    }
    for (0..4) |g| for (0..4) |q| try std.testing.expectEqual(@as(u8, @intCast(q + 1)), have[g][q]);
}
