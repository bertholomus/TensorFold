//! tf-dsv41-check MODEL_DIR CACHE_DIR RANK WORLD [--no-dspark]: the TP split plan against the lane's rank cache index,
//! entry by entry (key, order, dtype, trellis tiles). Reads only the file's index; no GPU.
const std = @import("std");
const dsv41 = @import("dsv41");

const usage = "usage: tf-dsv41-check MODEL_DIR CACHE_DIR RANK WORLD [--no-dspark]\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[3], 10);
    const world = try std.fmt.parseInt(u32, args[4], 10);
    const dspark = !(args.len > 5 and std.mem.eql(u8, args[5], "--no-dspark"));
    const cfg = try dsv41.Config.read(a, io, args[1]);
    var dir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
    defer dir.close(io);
    const name = try dsv41.rank_cache.find(a, io, dir, rank, world);
    var ix = try dsv41.rank_cache.open(init.gpa, io, dir, name);
    defer ix.deinit();
    const want = try dsv41.plan.wants(a, cfg, .{ .rank = rank, .world = world }, dspark);
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w = &out.interface;
    const keys = ix.entries.keys();
    const vals = ix.entries.values();
    var bad: usize = 0;
    var bytes: u64 = 0;
    for (vals) |e| bytes += e.bytes;
    const n = @min(keys.len, want.len);
    for (0..n) |i| {
        const e = vals[i];
        const x = want[i];
        var why: ?[]const u8 = null;
        if (!std.mem.eql(u8, keys[i], x.key)) {
            why = "key";
        } else if (x.dtype) |d| {
            if (d != e.dtype) why = "dtype";
        }
        if (why == null) if (x.k16) |k| if (e.dim(0) != k) {
            why = "k16";
        };
        if (why == null) if (x.n16) |k| if (e.dim(1) != k) {
            why = "n16";
        };
        if (why) |y| {
            bad += 1;
            if (bad <= 10) try w.print("mismatch {d} ({s}): file {s} {t} {any} | plan {s}\n", .{ i, y, keys[i], e.dtype, e.shape[0..e.rank], x.key });
        }
    }
    try w.print("{{\"file\": \"{s}\", \"entries\": {d}, \"planned\": {d}, \"bytes\": {d}, \"index_at\": {d}, \"mismatches\": {d}, \"ok\": {}}}\n", .{ name, keys.len, want.len, bytes, ix.data_end, bad, bad == 0 and keys.len == want.len });
    try w.flush();
    return if (bad == 0 and keys.len == want.len) 0 else 1;
}
