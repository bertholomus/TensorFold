//! tf-dsv41-ring2d-check RANK MASTER_IP PORT FATBIN DEVICES [--rows 1,6,16,32,48] [--reps 200] [--gid N]: the 2D
//! split's decode-size exchanges over the RDMA rings (ring2d.zig) on the four nodes, each checked word for word against
//! the layout Comm2D gives and timed: quarters of the attention / MoE partials (fp32, 2,560 + 2,560), cat_rank of the
//! wo_a output (bf16, 2,048 + 2,048) and of the experts' intermediate (fp16, 640 + 512 over rows x 7 slots), the pair
//! gather (fp32, 5,120) and Engram's quarters (fp32, 12,800 + 12,800). One JSON line a rank, case and row count.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const rdma = dsv41.rdma;
const ring2d = dsv41.ring2d;

const usage = "usage: tf-dsv41-ring2d-check RANK MASTER_IP PORT FATBIN DEVICES [--rows 1,6,16,32,48] [--reps 200] [--gid N]\n";

const Kind = enum { quarters, cat_rank, pair };
const Case = struct { name: []const u8, kind: Kind, w: [2]usize, esize: usize, slots: usize = 1 };
const cases = [_]Case{
    .{ .name = "quarters fp32 2560+2560", .kind = .quarters, .w = .{ 2560, 2560 }, .esize = 4 },
    .{ .name = "cat_rank bf16 2048+2048", .kind = .cat_rank, .w = .{ 2048, 2048 }, .esize = 2 },
    .{ .name = "cat_rank fp16 640+512 x7", .kind = .cat_rank, .w = .{ 640, 512 }, .esize = 2, .slots = 7 },
    .{ .name = "pair fp32 5120", .kind = .pair, .w = .{ 5120, 5120 }, .esize = 4 },
    .{ .name = "quarters fp32 12800+12800", .kind = .quarters, .w = .{ 12800, 12800 }, .esize = 4 },
};

/// Node q's word i of a part: exact, and different on every node and word.
fn word(q: u32, i: usize) u32 {
    return (q << 28) | @as(u32, @intCast(i & 0x0FFF_FFFF));
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 6) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const g = try std.fmt.parseInt(u32, args[1], 10);
    const ip = try dsv41.link.parseIp(args[2]);
    const port = try std.fmt.parseInt(u16, args[3], 10);
    var rows_text: []const u8 = "1,6,16,32,48";
    var reps: usize = 200;
    var gid: u8 = 5;
    var i: usize = 6;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--rows")) rows_text = args[i + 1] else if (std.mem.eql(u8, args[i], "--reps")) reps = try std.fmt.parseInt(usize, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--gid")) gid = try std.fmt.parseInt(u8, args[i + 1], 10) else return error.BadArgument;
    }
    var devices: std.ArrayList([]const u8) = .empty;
    var dit = std.mem.splitScalar(u8, args[5], ',');
    while (dit.next()) |t| try devices.append(a, t);
    var rows_list: std.ArrayList(usize) = .empty;
    var rit = std.mem.splitScalar(u8, rows_text, ',');
    while (rit.next()) |t| try rows_list.append(a, try std.fmt.parseInt(usize, t, 10));

    var out_buf: [1 << 12]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w_out = &out.interface;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const image = try std.Io.Dir.cwd().readFileAlloc(io, args[4], a, .limited(1 << 26));
    var module = try cuda.Module.load(&driver, image);
    defer module.unload();

    var fan = try dsv41.prompt2d.Fan.open(io, ip, port, g, 4);
    defer fan.close();
    var comm = try fan.comm(g, 4);
    defer comm.deinit();
    var rings = try ring2d.Rings.open(gpa, &driver, try rdma.Kernels.load(module), devices.items, g, .{ .max_bytes = 4456448, .gid_index = gid }, &comm, stream);
    defer rings.close(gpa);

    var all_ok = true;
    for (rows_list.items) |rows| for (cases) |c| {
        const n = rows * c.slots; // the part's rows (the intermediate: rows x slots)
        const p = g / 2;
        const r = g % 2;
        const mine_w = if (c.kind == .pair) c.w[0] else c.w[p];
        const src_bytes = n * mine_w * c.esize;
        const dst_bytes = switch (c.kind) {
            .quarters => 2 * n * (c.w[0] + c.w[1]) * c.esize,
            .cat_rank => n * (c.w[0] + c.w[1]) * c.esize,
            .pair => 2 * n * c.w[0] * c.esize,
        };
        const host_src = try a.alloc(u32, src_bytes / 4);
        for (host_src, 0..) |*x, j| x.* = word(g, j);
        var src = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(host_src));
        defer src.free();
        var dst = try cuda.DeviceBuffer.alloc(&driver, dst_bytes);
        defer dst.free();
        const run = struct {
            fn once(rs: *const ring2d.Rings, s: cuda.Stream, cc: Case, sp: u64, dp: u64, nn: usize) !void {
                switch (cc.kind) {
                    .quarters => try rs.quarters(s, sp, dp, nn, cc.w, cc.esize),
                    .cat_rank => try rs.catRank(s, sp, dp, nn, cc.w, cc.esize),
                    .pair => try rs.pairGather(s, sp, dp, nn, cc.w[0], cc.esize),
                }
            }
        }.once;
        run(&rings, stream, c, src.ptr, dst.ptr, n) catch |e| {
            try w_out.print("{{\"rank\": {d}, \"case\": \"{s}\", \"rows\": {d}, \"error\": \"{s}\"}}\n", .{ g, c.name, rows, @errorName(e) });
            try w_out.flush();
            continue;
        };
        for (0..20) |_| try run(&rings, stream, c, src.ptr, dst.ptr, n);
        try stream.synchronize();
        const t0 = std.Io.Timestamp.now(io, .awake);
        for (0..reps) |_| try run(&rings, stream, c, src.ptr, dst.ptr, n);
        try stream.synchronize();
        const us = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e3 / @as(f64, @floatFromInt(reps));
        const got = try a.alloc(u32, dst_bytes / 4);
        try dst.download(0, std.mem.sliceAsBytes(got));
        // the layout Comm2D gives, word for word (a part's row is wq words: w x esize / 4)
        var bad: usize = 0;
        switch (c.kind) {
            .quarters => {
                const row = (c.w[0] + c.w[1]) * c.esize / 4;
                for (0..4) |q| {
                    const qr = q % 2;
                    const qp = q / 2;
                    const wq = c.w[qp] * c.esize / 4;
                    const col0 = qp * c.w[0] * c.esize / 4;
                    for (0..n) |row_i| for (0..wq) |k| {
                        if (got[qr * n * row + row_i * row + col0 + k] != word(@intCast(q), row_i * wq + k)) bad += 1;
                    };
                }
            },
            .cat_rank => {
                const row = (c.w[0] + c.w[1]) * c.esize / 4;
                for (0..2) |pp| {
                    const q: u32 = @intCast(r + 2 * pp);
                    const wq = c.w[pp] * c.esize / 4;
                    const col0 = pp * c.w[0] * c.esize / 4;
                    for (0..n) |row_i| for (0..wq) |k| {
                        if (got[row_i * row + col0 + k] != word(q, row_i * wq + k)) bad += 1;
                    };
                }
            },
            .pair => {
                const wq = c.w[0] * c.esize / 4;
                for (0..2) |rr| {
                    const q: u32 = @intCast(2 * p + rr);
                    for (0..n * wq) |k| if (got[rr * n * wq + k] != word(q, k)) {
                        bad += 1;
                    };
                }
            },
        }
        all_ok = all_ok and bad == 0;
        try w_out.print("{{\"rank\": {d}, \"case\": \"{s}\", \"rows\": {d}, \"bytes\": {d}, \"us\": {d:.2}, \"bad\": {d}}}\n", .{ g, c.name, rows, dst_bytes, us, bad });
        try w_out.flush();
    };
    try w_out.print("{{\"rank\": {d}, \"all_equal\": {}}}\n", .{ g, all_ok });
    try w_out.flush();
    return if (all_ok) 0 else 1;
}
