//! The 2D split's decode-size exchanges over RDMA rings (DESIGN-zig-tp4.md section 4; the Python lane's Comm2D with
//! TF_DS_2D_INTO and TF_DS_2D_GROUPS): quarters over the four nodes' ring, cat_rank over the column pair's (node g and
//! g ^ 2, the same TP2 rank of each pair), the pair gather over the row pair's (g and g ^ 1, TP2 inside a pair), each
//! part written straight into its place in the caller's layout (no [4, ...] buffer, no copies after it). The rings
//! open on the fabric's verbs devices; their queue-pair infos go round the four nodes in one NCCL all-gather. Parts
//! travel as fp32 words (16-bit parts reinterpreted), so a part's row must be a whole number of 16 bytes. A part
//! larger than the rings take returns error.TooLarge, and the caller takes NCCL (prompt2d.zig), as the lane does.
const std = @import("std");
const cuda = @import("cuda");
const rdma = @import("rdma.zig");
const Comm = @import("comm.zig").Comm;

pub const Rings = struct {
    g: u32,
    r: u32,
    p: u32,
    all4: *rdma.Ring,
    col: *rdma.Ring,
    pair: *rdma.Ring,

    /// The three rings of node `g` (of 4): created, their infos exchanged over `comm` (world 4), connected and
    /// started on every node before any returns.
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, g: u32, s: rdma.Settings, comm: *const Comm, stream: cuda.Stream) !Rings {
        if (g >= 4 or comm.world != 4) return error.NotFourNodes;
        const r = g % 2;
        const p = g / 2;
        const all4 = try rdma.Ring.create(gpa, d, k, devices, g, 4, s);
        errdefer all4.destroy(gpa);
        const col = try rdma.Ring.create(gpa, d, k, devices, p, 2, s); // members (r, 0), (r, 1): ring rank = pair
        errdefer col.destroy(gpa);
        const pair = try rdma.Ring.create(gpa, d, k, devices, r, 2, s); // members (0, p), (1, p): ring rank = TP2 rank
        errdefer pair.destroy(gpa);
        // this node's infos: the four-node ring's toward each node, the column pair's and the row pair's toward each member
        const per = @sizeOf(rdma.Info);
        var mine: [8]rdma.Info = undefined;
        for (0..4) |q| mine[q] = all4.info(@intCast(q));
        for (0..2) |q| mine[4 + q] = col.info(@intCast(q));
        for (0..2) |q| mine[6 + q] = pair.info(@intCast(q));
        var dev = try cuda.DeviceBuffer.alloc(d, 5 * 8 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 8 * per, 8 * per, .u8, stream);
        try stream.synchronize();
        var all: [4][8]rdma.Info = undefined;
        try dev.download(8 * per, std.mem.sliceAsBytes(&all));
        var remote4: [4]rdma.Info = undefined;
        for (0..4) |q| remote4[q] = all[q][g];
        try all4.connect(&remote4);
        var remote_col: [2]rdma.Info = undefined;
        for (0..2) |pp| remote_col[pp] = all[r + 2 * pp][4 + p];
        try col.connect(&remote_col);
        var remote_pair: [2]rdma.Info = undefined;
        for (0..2) |rr| remote_pair[rr] = all[2 * p + rr][6 + r];
        try pair.connect(&remote_pair);
        // every queue pair ready on every node before any ring sends: one more all-gather as the barrier
        try comm.allGather(dev.ptr, dev.ptr + 8 * per, 1, .u8, stream);
        try stream.synchronize();
        try all4.start();
        try col.start();
        try pair.start();
        return .{ .g = g, .r = r, .p = p, .all4 = all4, .col = col, .pair = pair };
    }

    pub fn close(rs: *Rings, gpa: std.mem.Allocator) void {
        for ([_]*rdma.Ring{ rs.pair, rs.col, rs.all4 }) |x| {
            x.stop();
            x.destroy(gpa);
        }
        rs.* = undefined;
    }

    fn float4s(bytes: usize) !u32 {
        if (bytes % 16 != 0) return error.NotFloat4;
        return @intCast(bytes / 16);
    }

    /// quarters (Comm2D.quarters): `src` [rows, w[p]] (this node's columns of its TP2 rank's part) -> `dst`
    /// [2, rows, w0 + w1]: each TP2 rank's whole part, pair 0's columns then pair 1's.
    pub fn quarters(rs: *const Rings, stream: cuda.Stream, src: u64, dst: u64, rows: usize, w: [2]usize, esize: usize) !void {
        const row = (w[0] + w[1]) * esize;
        var dsts: [4]rdma.Ring.Dst = undefined;
        for (0..4) |q| dsts[q] = .{ .ptr = dst + (q % 2) * rows * row + (q / 2) * w[0] * esize, .row4 = try float4s(w[q / 2] * esize), .stride4 = try float4s(row) };
        try rs.all4.gatherInto(stream, src, @intCast(rows), try float4s(w[rs.p] * esize), &dsts);
    }

    /// cat_rank (Comm2D.cat_rank): `src` [rows, w[p]] -> `dst` [rows, w0 + w1], pair 0's part then pair 1's, from the
    /// column partner.
    pub fn catRank(rs: *const Rings, stream: cuda.Stream, src: u64, dst: u64, rows: usize, w: [2]usize, esize: usize) !void {
        const row = (w[0] + w[1]) * esize;
        var dsts: [2]rdma.Ring.Dst = undefined;
        for (0..2) |pp| dsts[pp] = .{ .ptr = dst + pp * w[0] * esize, .row4 = try float4s(w[pp] * esize), .stride4 = try float4s(row) };
        try rs.col.gatherInto(stream, src, @intCast(rows), try float4s(w[rs.p] * esize), &dsts);
    }

    /// The pair gather (Comm2D.gather: TP2 inside a pair): `src` [rows, width] -> `dst` [2, rows, width] in TP2 rank order.
    pub fn pairGather(rs: *const Rings, stream: cuda.Stream, src: u64, dst: u64, rows: usize, width: usize, esize: usize) !void {
        var dsts: [2]rdma.Ring.Dst = undefined;
        for (0..2) |rr| dsts[rr] = .{ .ptr = dst + rr * rows * width * esize, .row4 = try float4s(width * esize), .stride4 = try float4s(width * esize) };
        try rs.pair.gatherInto(stream, src, @intCast(rows), try float4s(width * esize), &dsts);
    }
};

/// Where node q's word j of part `kind` lands, for the checks: (TP2 rank, pair) of the node owning it.
pub fn owner(q: u32) [2]u32 {
    return .{ q % 2, q / 2 };
}

test "a 2D node's ring ranks and members" {
    // node 3 = TP2 rank 1 of pair 1: the column pair is nodes 1 and 3 (ring rank 1), the row pair nodes 2 and 3 (rank 1)
    try std.testing.expectEqual([2]u32{ 1, 1 }, owner(3));
    try std.testing.expectEqual([2]u32{ 0, 1 }, owner(2));
    try std.testing.expectError(error.NotFloat4, Rings.float4s(10));
    try std.testing.expectEqual(@as(u32, 640), try Rings.float4s(2560 * 4));
}
