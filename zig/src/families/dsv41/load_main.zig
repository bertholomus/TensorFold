//! tf-dsv41-load MODEL_DIR RANK WORLD [--cache DIR] [--digests OUT.jsonl]: one rank's weights onto its GPU through the
//! Zig loader (from the lane's rank cache, else the checkpoint), timed, with each device tensor's sha256 read back.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");

const usage = "usage: tf-dsv41-load MODEL_DIR RANK WORLD [--cache DIR] [--digests OUT.jsonl] [--no-dspark]\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[2], 10);
    const world = try std.fmt.parseInt(u32, args[3], 10);
    var cache_dir: ?[]const u8 = null;
    var digests: ?[]const u8 = null;
    var dspark = true;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--cache") and i + 1 < args.len) {
            cache_dir = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--digests") and i + 1 < args.len) {
            digests = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--no-dspark")) dspark = false;
    }
    const cfg = try dsv41.Config.read(a, io, args[1]);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const t0 = std.Io.Timestamp.now(io, .awake);
    var w = if (cache_dir) |dir_path| blk: {
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        const name = try dsv41.rank_cache.find(a, io, dir, rank, world);
        var ix = try dsv41.rank_cache.open(init.gpa, io, dir, name);
        defer ix.deinit();
        var file = try dir.openFile(io, name, .{});
        defer file.close(io);
        break :blk try dsv41.weights.load(init.gpa, &driver, .{ .cache = .{ .file = file, .index = &ix, .io = io } }, cfg, .{ .rank = rank, .world = world }, dspark);
    } else blk: {
        var sh = try dsv41.checkpoint.Shards.open(init.gpa, io, args[1]);
        defer sh.close(io);
        break :blk try dsv41.weights.load(init.gpa, &driver, .{ .shards = &sh }, cfg, .{ .rank = rank, .world = world }, dspark);
    };
    defer w.deinit();
    const secs = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    try out.interface.print("{{\"rank\": {d}, \"tensors\": {d}, \"bytes\": {d}, \"seconds\": {d:.1}, \"source\": \"{s}\"}}\n", .{ rank, w.named.items.len, w.bytes, secs, if (cache_dir != null) "cache" else "checkpoint" });
    try out.interface.flush();
    if (digests) |path| {
        var f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        var fbuf: [1 << 16]u8 = undefined;
        var fw = f.writer(io, &fbuf);
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(init.gpa);
        for (w.named.items) |n| {
            try host.resize(init.gpa, n.len);
            try driver.check(driver.api.cuMemcpyDtoH_v2(host.items.ptr, n.ptr, n.len), "cuMemcpyDtoH");
            var h: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(host.items, &h, .{});
            try fw.interface.print("{{\"name\": \"{s}\", \"bytes\": {d}, \"sha256\": \"{x}\"}}\n", .{ n.name, n.len, &h });
        }
        try fw.interface.flush();
    }
    return 0;
}
