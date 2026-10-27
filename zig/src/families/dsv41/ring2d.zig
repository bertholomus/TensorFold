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
const comm_mod = @import("comm.zig");
const Comm = comm_mod.Comm;

/// A verbs device and the /24 of its IPv4 GID (TF_DS_TWO_HOP: the subnet names the cable it is on).
pub const Dev = extern struct { name: [24]u8 = @splat(0), net: [3]u8 = .{ 0, 0, 0 }, ok: u8 = 0 };

/// The /24 of an IPv4-mapped GID as sysfs prints it ("0000:...:ffff:c000:1e07" -> 192.0.30), null for another GID.
pub fn gidNet(text: []const u8) ?[3]u8 {
    const t = std.mem.trim(u8, text, " \n");
    if (t.len != 39 or !std.mem.startsWith(u8, t, "0000:0000:0000:0000:0000:ffff:")) return null;
    const a = std.fmt.parseInt(u16, t[30..34], 16) catch return null;
    const b = std.fmt.parseInt(u16, t[35..39], 16) catch return null;
    return .{ @intCast(a >> 8), @intCast(a & 0xff), @intCast(b >> 8) };
}

/// This node's devices with their GID index `gi` subnets, from sysfs.
fn devs(io: std.Io, devices: []const []const u8, gi: u8) ![4]Dev {
    var out: [4]Dev = @splat(.{});
    if (devices.len > 4) return error.TooManyDevices;
    for (devices, 0..) |name, i| {
        if (name.len >= out[i].name.len) return error.DeviceName;
        @memcpy(out[i].name[0..name.len], name);
        var pb: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&pb, "/sys/class/infiniband/{s}/ports/1/gids/{d}", .{ name, gi });
        var tb: [64]u8 = undefined;
        const text = std.Io.Dir.cwd().readFile(io, path, &tb) catch continue;
        if (gidNet(text)) |n| {
            out[i].net = n;
            out[i].ok = 1;
        }
    }
    return out;
}

/// TF_DS_TWO_HOP: this node's devices toward one partner, those on a subnet one of the partner's devices is on, in
/// subnet order (so device d of the ring meets the partner's device d on the same cable).
pub fn pick(mine: [4]Dev, theirs: [4]Dev, out: *[4][]const u8, names: *[4][24]u8) usize {
    var idx: [4]usize = undefined;
    var n: usize = 0;
    for (mine, 0..) |m, i| {
        if (m.ok == 0) continue;
        for (theirs) |t| if (t.ok == 1 and std.mem.eql(u8, &m.net, &t.net)) {
            idx[n] = i;
            n += 1;
            break;
        };
    }
    std.mem.sort(usize, idx[0..n], mine, struct {
        fn lt(ms: [4]Dev, a: usize, b: usize) bool {
            return std.mem.order(u8, &ms[a].net, &ms[b].net) == .lt;
        }
    }.lt);
    for (idx[0..n], 0..) |i, k| {
        names[k] = mine[i].name;
        out[k] = std.mem.sliceTo(&names[k], 0);
    }
    return n;
}

pub const Rings = struct {
    g: u32,
    r: u32,
    p: u32,
    all4: ?*rdma.Ring, // null with TF_DS_TWO_HOP: the quarters go as two hops over col and pair
    col: *rdma.Ring,
    pair: *rdma.Ring,

    /// The three rings of node `g` (of 4): created, their infos exchanged over `comm` (world 4), connected and
    /// started on every node before any returns.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, g: u32, s: rdma.Settings, comm: *const Comm, stream: cuda.Stream) !Rings {
        if (g >= 4 or comm.world != 4) return error.NotFourNodes;
        const r = g % 2;
        const p = g / 2;
        // TF_DS_TWO_HOP: each partner's ring on the devices cabled to it (every node's subnets go round first)
        var col_devs: []const []const u8 = devices;
        var pair_devs: []const []const u8 = devices;
        var col_buf: [4][]const u8 = undefined;
        var pair_buf: [4][]const u8 = undefined;
        var col_names: [4][24]u8 = undefined;
        var pair_names: [4][24]u8 = undefined;
        if (comm_mod.two_hop) {
            const mine = try devs(io, devices, s.gid_index);
            const tb = @sizeOf([4]Dev);
            var tdev = try cuda.DeviceBuffer.alloc(d, 5 * tb);
            defer tdev.free();
            try tdev.upload(0, std.mem.asBytes(&mine));
            try comm.allGather(tdev.ptr, tdev.ptr + tb, tb, .u8, stream);
            try stream.synchronize();
            var tables: [4][4]Dev = undefined;
            try tdev.download(tb, std.mem.sliceAsBytes(&tables));
            col_devs = col_buf[0..pick(mine, tables[g ^ 2], &col_buf, &col_names)];
            pair_devs = pair_buf[0..pick(mine, tables[g ^ 1], &pair_buf, &pair_names)];
            if (col_devs.len == 0 or pair_devs.len == 0) return error.NoCableToPartner;
        }
        const all4: ?*rdma.Ring = if (comm_mod.two_hop) null else try rdma.Ring.create(gpa, d, k, devices, g, 4, s);
        errdefer if (all4) |x| x.destroy(gpa);
        const col = try rdma.Ring.create(gpa, d, k, col_devs, p, 2, s); // members (r, 0), (r, 1): ring rank = pair
        errdefer col.destroy(gpa);
        const pair = try rdma.Ring.create(gpa, d, k, pair_devs, r, 2, s); // members (0, p), (1, p): ring rank = TP2 rank
        errdefer pair.destroy(gpa);
        // this node's infos: the four-node ring's toward each node, the column pair's and the row pair's toward each member
        const per = @sizeOf(rdma.Info);
        var mine: [8]rdma.Info = undefined;
        for (0..4) |q| mine[q] = if (all4) |x| x.info(@intCast(q)) else std.mem.zeroes(rdma.Info);
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
        if (all4) |x| try x.connect(&remote4);
        var remote_col: [2]rdma.Info = undefined;
        for (0..2) |pp| remote_col[pp] = all[r + 2 * pp][4 + p];
        try col.connect(&remote_col);
        var remote_pair: [2]rdma.Info = undefined;
        for (0..2) |rr| remote_pair[rr] = all[2 * p + rr][6 + r];
        try pair.connect(&remote_pair);
        // every queue pair ready on every node before any ring sends: one more all-gather as the barrier
        try comm.allGather(dev.ptr, dev.ptr + 8 * per, 1, .u8, stream);
        try stream.synchronize();
        if (all4) |x| try x.start();
        try col.start();
        try pair.start();
        return .{ .g = g, .r = r, .p = p, .all4 = all4, .col = col, .pair = pair };
    }

    pub fn close(rs: *Rings, gpa: std.mem.Allocator) void {
        for ([_]?*rdma.Ring{ rs.pair, rs.col, rs.all4 }) |o| if (o) |x| {
            x.stop();
            x.destroy(gpa);
        };
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
        const a4 = rs.all4 orelse {
            // TF_DS_TWO_HOP: the column pair's parts make rank r's row of dst, then the row pair swaps whole rows
            const rw = (w[0] + w[1]) * esize;
            if (rows * rw > rs.pair.s.max_bytes) return error.TooLarge;
            const mine = dst + rs.r * rows * rw;
            try rs.catRank(stream, src, mine, rows, w, esize);
            return rs.pairGather(stream, mine, dst, rows, w[0] + w[1], esize);
        };
        try a4.gatherInto(stream, src, @intCast(rows), try float4s(w[rs.p] * esize), &dsts);
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

test "two hops: each partner's ring takes the devices cabled to it, in subnet order (the first node with the second and the third)" {
    try std.testing.expectEqual([3]u8{ 192, 0, 30 }, gidNet("0000:0000:0000:0000:0000:ffff:c000:1e07\n").?);
    try std.testing.expect(gidNet("fe80:0000:0000:0000:bd59:6cf0:c06e:0323") == null);
    const mk = struct {
        fn of(names: [4][]const u8, nets: [4]u8) [4]Dev {
            var o: [4]Dev = @splat(.{});
            for (names, nets, 0..) |nm, n, i| {
                @memcpy(o[i].name[0..nm.len], nm);
                o[i].net = .{ 192, 0, n };
                o[i].ok = 1;
            }
            return o;
        }
    };
    const names = [4][]const u8{ "rdma0", "rdma0f1", "rdma1", "rdma1f1" };
    const a7 = mk.of(names, .{ 30, 110, 40, 120 });
    const a8 = mk.of(names, .{ 30, 90, 40, 100 });
    const a5 = mk.of(names, .{ 110, 70, 120, 80 });
    var out: [4][]const u8 = undefined;
    var nb: [4][24]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 2), pick(a7, a8, &out, &nb));
    try std.testing.expectEqualStrings("rdma0", out[0]);
    try std.testing.expectEqualStrings("rdma1", out[1]);
    try std.testing.expectEqual(@as(usize, 2), pick(a7, a5, &out, &nb));
    try std.testing.expectEqualStrings("rdma0f1", out[0]);
    try std.testing.expectEqualStrings("rdma1f1", out[1]);
    var back: [4][]const u8 = undefined;
    var nb2: [4][24]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 2), pick(a5, a7, &back, &nb2));
    try std.testing.expectEqualStrings("rdma0", back[0]); // .110 meets the first node's .110
    try std.testing.expectEqualStrings("rdma1", back[1]);
    try std.testing.expectEqual(@as(usize, 0), pick(a8, a5, &out, &nb)); // the second and third: no cable
}
