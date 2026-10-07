//! tf-dsv41-rdma-bench: the RDMA ring (rdma_ring.zig) across real nodes, one process a rank. Rank 0 listens on
//! `--port` for the other ranks' queue-pair infos and sends every rank the whole table; once all are connected, each
//! rank runs back-to-back gathers at each row count (50 a CUDA graph, replayed, as the Python ring's --gather-bench),
//! the plain gather and the strided gather_into, and checks every float it received. One JSON line a rank and row count.
//!   tf-dsv41-rdma-bench RANK WORLD MASTER_IP FATBIN DEVICES [--port P] [--rows 1,6,16,48] [--width 2560] [--reps 500]
//!                       [--pdl 0|1] [--gid N]
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const ring = dsv41.rdma_ring;
const posix = std.posix;

const usage = "usage: tf-dsv41-rdma-bench RANK WORLD MASTER_IP FATBIN DEVICES [--port P] [--rows 1,6,16,48] [--width 2560] [--reps 500] [--pdl 0|1] [--gid N]\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 6) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const world = try std.fmt.parseInt(u32, args[2], 10);
    const master = try ip4(args[3]);
    var port: u16 = 29_700;
    var rows_text: []const u8 = "1,6,16,48";
    var width: u32 = 2560;
    var reps: u32 = 500;
    var pdl = true;
    var gid: u8 = 5;
    var i: usize = 6;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--port")) port = try std.fmt.parseInt(u16, v, 10) else if (std.mem.eql(u8, k, "--rows")) rows_text = v else if (std.mem.eql(u8, k, "--width")) width = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--reps")) reps = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--pdl")) pdl = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--gid")) gid = try std.fmt.parseInt(u8, v, 10) else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    if (width % 4 != 0) return error.WidthNotFloat4;
    var rows: std.ArrayList(u32) = .empty;
    var it = std.mem.splitScalar(u8, rows_text, ',');
    while (it.next()) |t| try rows.append(a, try std.fmt.parseInt(u32, t, 10));
    var devices: std.ArrayList([]const u8) = .empty;
    var dit = std.mem.splitScalar(u8, args[5], ',');
    while (dit.next()) |t| try devices.append(a, t);
    const per_graph = 50;
    const loops = @max(1, reps / per_graph);
    var most: u32 = 1;
    for (rows.items) |n| most = @max(most, n);

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const image = try std.Io.Dir.cwd().readFileAlloc(io, args[4], a, .limited(1 << 26));
    var module = try cuda.Module.load(&driver, image);
    defer module.unload();
    const r = try ring.Ring.create(init.gpa, &driver, try ring.Kernels.load(module), devices.items, rank, world, .{ .max_bytes = std.mem.alignForward(usize, @as(usize, most) * width * 4, 64), .pdl = pdl, .gid_index = gid });
    defer r.destroy(init.gpa);

    // every rank's info(p) for every p, gathered by rank 0 and handed back whole; then a barrier once all are connected
    const row_bytes = @sizeOf(ring.Info) * world;
    const mine = try a.alloc(u8, row_bytes);
    for (0..world) |p| @memcpy(mine[p * @sizeOf(ring.Info) ..][0..@sizeOf(ring.Info)], std.mem.asBytes(&r.info(@intCast(p))));
    const table = try a.alloc(u8, row_bytes * world);
    try exchange(io, rank, world, master, port, mine, table);
    var remote: [ring.max_ranks]ring.Info = undefined;
    for (0..world) |q| remote[q] = std.mem.bytesToValue(ring.Info, table[q * row_bytes + rank * @sizeOf(ring.Info) ..][0..@sizeOf(ring.Info)]);
    try r.connect(remote[0..world]);
    var one = [_]u8{1};
    const all = try a.alloc(u8, world);
    try exchange(io, rank, world, master, port, &one, all);
    try r.start();

    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    for (rows.items) |n_rows| {
        const n: usize = @as(usize, n_rows) * width; // floats a rank
        const host = try a.alloc(f32, n * world);
        for (0..n) |j| host[j] = value(rank, j);
        var send = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(host[0..n]));
        defer send.free();
        var recv = try cuda.DeviceBuffer.alloc(&driver, n * world * 4);
        defer recv.free();
        var into = try cuda.DeviceBuffer.alloc(&driver, n * world * 4); // [rows][world x width]: rank q at columns q x width
        defer into.free();
        const n4: u32 = @intCast(n / 4);
        var dsts: [ring.max_ranks]ring.Ring.Dst = undefined;
        for (0..world) |q| dsts[q] = .{ .ptr = into.ptr + @as(u64, q) * width * 4, .row4 = width / 4, .stride4 = world * width / 4 };
        for (0..20) |_| try r.gather(stream, send.ptr, n4, recv.ptr);
        try stream.synchronize();
        // 50 gathers captured in one graph and replayed, as the Python ring's --gather-bench times them
        try cuda.graph.beginCapture(stream, .thread_local);
        for (0..per_graph) |_| try r.gather(stream, send.ptr, n4, recv.ptr);
        var g1 = try cuda.graph.endCapture(stream);
        defer g1.deinit();
        var e1 = try g1.instantiate();
        defer e1.deinit();
        try e1.launchOn(stream);
        try stream.synchronize();
        var t0 = std.Io.Timestamp.now(io, .awake);
        for (0..loops) |_| try e1.launchOn(stream);
        try stream.synchronize();
        const us_gather = micros(t0, io) / @as(f64, @floatFromInt(loops * per_graph));
        try recv.download(0, std.mem.sliceAsBytes(host));
        var bad: usize = 0;
        for (0..world) |q| for (0..n) |j| {
            if (host[q * n + j] != value(@intCast(q), j)) bad += 1;
        };
        for (0..20) |_| try r.gatherInto(stream, send.ptr, n_rows, width / 4, dsts[0..world]);
        try stream.synchronize();
        try cuda.graph.beginCapture(stream, .thread_local);
        for (0..per_graph) |_| try r.gatherInto(stream, send.ptr, n_rows, width / 4, dsts[0..world]);
        var g2 = try cuda.graph.endCapture(stream);
        defer g2.deinit();
        var e2 = try g2.instantiate();
        defer e2.deinit();
        try e2.launchOn(stream);
        try stream.synchronize();
        t0 = std.Io.Timestamp.now(io, .awake);
        for (0..loops) |_| try e2.launchOn(stream);
        try stream.synchronize();
        const us_into = micros(t0, io) / @as(f64, @floatFromInt(loops * per_graph));
        try into.download(0, std.mem.sliceAsBytes(host));
        var bad_into: usize = 0;
        for (0..n_rows) |row| for (0..world) |q| for (0..width) |c| {
            if (host[row * world * width + q * width + c] != value(@intCast(q), row * width + c)) bad_into += 1;
        };
        try out.interface.print("{{\"rank\": {d}, \"world\": {d}, \"rows\": {d}, \"bytes\": {d}, \"gathers\": {d}, \"pdl\": {d}, \"us_gather\": {d:.2}, \"us_into\": {d:.2}, \"bad\": {d}, \"bad_into\": {d}, \"failure\": \"{s}\"}}\n", .{ rank, world, n_rows, n * 4, loops * per_graph, @intFromBool(pdl), us_gather, us_into, bad, bad_into, r.failure() orelse "" });
        try out.interface.flush();
    }
    return 0;
}

/// The float rank `q` sends at index `j`: exact in f32 (below 2^24).
fn value(q: u32, j: usize) f32 {
    return @floatFromInt(@as(u32, q) * 1_000_000 + @as(u32, @intCast(j % 1_000_000)));
}

fn micros(t0: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e3;
}

fn ip4(text: []const u8) !u32 {
    var out: u32 = 0;
    var parts: u32 = 0;
    var it = std.mem.splitScalar(u8, text, '.');
    while (it.next()) |t| : (parts += 1) out = out << 8 | try std.fmt.parseInt(u8, t, 10);
    if (parts != 4) return error.BadAddress;
    return out;
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) !void {
    var sent: usize = 0;
    while (sent < bytes.len) {
        const rc = posix.system.write(fd, bytes[sent..].ptr, bytes.len - sent);
        switch (posix.errno(rc)) {
            .SUCCESS => sent += @intCast(rc),
            .INTR => {},
            else => return error.WriteFailed,
        }
    }
}

fn readAll(fd: posix.socket_t, buf: []u8) !void {
    var got: usize = 0;
    while (got < buf.len) {
        const rc = posix.system.read(fd, buf[got..].ptr, buf.len - got);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.PeerClosed;
                got += @intCast(rc);
            },
            .INTR => {},
            else => return error.ReadFailed,
        }
    }
}

fn socketTcp() !posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    return @intCast(rc);
}

/// Rank 0 collects every rank's `mine` (all the same length) into `table` in rank order and sends it to every rank; the
/// others send theirs and read the table back. Ranks other than 0 retry their connect for a minute.
fn exchange(io: std.Io, rank: u32, world: u32, master: u32, port: u16, mine: []const u8, table: []u8) !void {
    var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, if (rank == 0) 0 else master) };
    if (rank == 0) {
        const fd = try socketTcp();
        defer _ = posix.system.close(fd);
        const yes: c_int = 1;
        try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&yes));
        if (posix.errno(posix.system.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(fd, 16)) != .SUCCESS) return error.ListenFailed;
        @memcpy(table[0..mine.len], mine);
        var conns: [ring.max_ranks]posix.socket_t = undefined;
        for (1..world) |k| {
            const rc = posix.system.accept(fd, null, null);
            if (posix.errno(rc) != .SUCCESS) return error.AcceptFailed;
            conns[k] = @intCast(rc);
            var who: [4]u8 = undefined;
            try readAll(conns[k], &who);
            const q = std.mem.readInt(u32, &who, .little);
            if (q == 0 or q >= world) return error.BadRank;
            try readAll(conns[k], table[q * mine.len ..][0..mine.len]);
        }
        for (1..world) |k| {
            try writeAll(conns[k], table);
            _ = posix.system.close(conns[k]);
        }
        return;
    }
    var fd: posix.socket_t = undefined;
    var tries: u32 = 0;
    while (true) : (tries += 1) {
        const s = try socketTcp();
        if (posix.errno(posix.system.connect(s, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) {
            fd = s;
            break;
        }
        _ = posix.system.close(s);
        if (tries > 600) return error.ConnectFailed;
        std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    }
    defer _ = posix.system.close(fd);
    var who: [4]u8 = undefined;
    std.mem.writeInt(u32, &who, rank, .little);
    try writeAll(fd, &who);
    try writeAll(fd, mine);
    try readAll(fd, table);
}

test "an IPv4 address in dotted form" {
    try std.testing.expectEqual(@as(u32, 0xc0000201), try ip4("192.0.2.1"));
    try std.testing.expectError(error.BadAddress, ip4("10.200.10"));
}
