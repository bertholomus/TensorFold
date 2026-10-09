//! tf-dsv41-check MODEL_DIR CACHE_DIR RANK WORLD [--no-dspark] [--parity] [--bytes EVERY]: the TP split plan against
//! the lane's rank cache, entry by entry: key, order, dtype and trellis tiles; the checkpoint reader's layout of every
//! entry; with --bytes, the bytes of every EVERY-th entry and of every entry outside the routed experts, read both ways.
//! --parity: the four-node split's balanced experts (TF_DS_2D_GU=parity). No GPU.
const std = @import("std");
const dsv41 = @import("dsv41");

const usage = "usage: tf-dsv41-check MODEL_DIR CACHE_DIR RANK WORLD [--no-dspark] [--parity] [--bytes EVERY]\n";

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
    var dspark = true;
    var parity = false;
    var every: usize = 0;
    var ai: usize = 5;
    while (ai < args.len) : (ai += 1) {
        if (std.mem.eql(u8, args[ai], "--no-dspark")) dspark = false;
        if (std.mem.eql(u8, args[ai], "--parity")) parity = true;
        if (std.mem.eql(u8, args[ai], "--bytes") and ai + 1 < args.len) {
            every = try std.fmt.parseInt(usize, args[ai + 1], 10);
            ai += 1;
        }
    }
    const cfg = try dsv41.Config.read(a, io, args[1]);
    var dir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
    defer dir.close(io);
    const name = try dsv41.rank_cache.find(a, io, dir, rank, world);
    var ix = try dsv41.rank_cache.open(init.gpa, io, dir, name);
    defer ix.deinit();
    // four ranks: the exact 2D split (node g = TP2 rank g % 2 of pair g / 2); the cache file is still rank g of 4
    const split: dsv41.plan.Split = if (world == 4) .{ .rank = rank % 2, .world = 2, .pair = rank / 2, .parity = parity } else .{ .rank = rank, .world = world };
    const want = try dsv41.plan.wants(a, cfg, split, dspark);
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
    // the checkpoint reader: every entry's layout, and with --bytes the bytes themselves against the cache file's
    var sh = try dsv41.checkpoint.Shards.open(init.gpa, io, args[1]);
    defer sh.close(io);
    var layout_bad: usize = 0;
    var compared: usize = 0;
    var byte_bad: usize = 0;
    var compared_bytes: u64 = 0;
    var cache_file = try dir.openFile(io, name, .{});
    defer cache_file.close(io);
    var a_buf: std.ArrayList(u8) = .empty;
    var b_buf: std.ArrayList(u8) = .empty;
    for (keys, vals, 0..) |k, e, i| {
        const s = dsv41.checkpoint.describe(&sh, k) catch |err| {
            layout_bad += 1;
            if (layout_bad <= 10) try w.print("layout {d}: {s}: {t}\n", .{ i, k, err });
            continue;
        };
        if (s.dtype != e.dtype or s.rank != e.rank or !std.mem.eql(usize, s.shape[0..s.rank], e.shape[0..e.rank]) or s.bytes != e.bytes) {
            layout_bad += 1;
            if (layout_bad <= 10) try w.print("layout {d}: {s}: reader {t} {any} {d} | cache {t} {any} {d}\n", .{ i, k, s.dtype, s.shape[0..s.rank], s.bytes, e.dtype, e.shape[0..e.rank], e.bytes });
            continue;
        }
        const expert = std.mem.indexOf(u8, k, ".ffn.experts.") != null;
        if (every == 0 or (expert and i % every != 0)) continue;
        try a_buf.resize(init.gpa, s.bytes);
        try b_buf.resize(init.gpa, s.bytes);
        try dsv41.checkpoint.read(&sh, k, a_buf.items);
        if (try cache_file.readPositionalAll(io, b_buf.items, e.offset) != s.bytes) return error.ShortRead;
        compared += 1;
        compared_bytes += s.bytes;
        if (!std.mem.eql(u8, a_buf.items, b_buf.items)) {
            byte_bad += 1;
            if (byte_bad <= 10) try w.print("bytes {d}: {s} differ\n", .{ i, k });
        }
    }
    a_buf.deinit(init.gpa);
    b_buf.deinit(init.gpa);
    const ok = bad == 0 and keys.len == want.len and layout_bad == 0 and byte_bad == 0;
    try w.print("{{\"file\": \"{s}\", \"entries\": {d}, \"planned\": {d}, \"bytes\": {d}, \"index_at\": {d}, \"mismatches\": {d}, \"layout_mismatches\": {d}, \"compared\": {d}, \"compared_bytes\": {d}, \"byte_mismatches\": {d}, \"ok\": {}}}\n", .{ name, keys.len, want.len, bytes, ix.data_end, bad, layout_bad, compared, compared_bytes, byte_bad, ok });
    try w.flush();
    return if (ok) 0 else 1;
}
