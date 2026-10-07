//! tf-dsv41-comm RANK MASTER_IP PORT: the two ranks' link and NCCL from Zig. Each rank all-gathers fp32 rows (hidden
//! 5120) with its own pattern, checks every rank's part, then times gathers of 1, 6, 24 and 2048 rows. JSON on stdout.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");

const hidden = 5120;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("usage: tf-dsv41-comm RANK MASTER_IP PORT\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const ip = try dsv41.link.parseIp(args[2]);
    const port = try std.fmt.parseInt(u16, args[3], 10);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var link = if (rank == 0) try dsv41.link.Link.listen(ip, port) else try dsv41.link.Link.connect(io, ip, port, 120);
    defer link.close();
    var comm = try dsv41.comm.Comm.init(link, rank, 2);
    defer comm.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const max_rows = 2048;
    var send = try cuda.DeviceBuffer.alloc(&driver, max_rows * hidden * 4);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(&driver, 2 * max_rows * hidden * 4);
    defer recv.free();
    const host = try a.alloc(f32, 2 * max_rows * hidden);
    for (host[0 .. max_rows * hidden], 0..) |*x, i| x.* = @floatFromInt((rank + 1) * 1000003 +% @as(u32, @intCast(i % 65521)));
    try send.upload(0, std.mem.sliceAsBytes(host[0 .. max_rows * hidden]));
    // correctness at 24 rows: rank r's slot holds rank r's pattern
    try comm.allGather(send.ptr, recv.ptr, 24 * hidden, .f32, stream);
    try stream.synchronize();
    try recv.download(0, std.mem.sliceAsBytes(host[0 .. 2 * 24 * hidden]));
    var ok = true;
    for (0..2) |r| for (0..24 * hidden) |i| {
        const want: f32 = @floatFromInt(@as(u32, @intCast(r + 1)) * 1000003 +% @as(u32, @intCast(i % 65521)));
        if (host[r * 24 * hidden + i] != want) ok = false;
    };
    var start = try cuda.Event.init(&driver, true);
    defer start.deinit();
    var end = try cuda.Event.init(&driver, true);
    defer end.deinit();
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    try out.interface.print("{{\"rank\": {d}, \"gather_ok\": {}", .{ rank, ok });
    for ([_]usize{ 1, 6, 24, 2048 }) |rows| {
        const reps: usize = if (rows > 64) 50 else 500;
        for (0..10) |_| try comm.allGather(send.ptr, recv.ptr, rows * hidden, .f32, stream);
        try start.record(stream);
        for (0..reps) |_| try comm.allGather(send.ptr, recv.ptr, rows * hidden, .f32, stream);
        try end.record(stream);
        try end.synchronize();
        const ms = try cuda.Event.elapsedMs(start, end);
        try out.interface.print(", \"us_{d}_rows\": {d:.1}", .{ rows, ms * 1000 / @as(f32, @floatFromInt(reps)) });
    }
    try out.interface.print("}}\n", .{});
    try out.interface.flush();
    return if (ok) 0 else 1;
}
