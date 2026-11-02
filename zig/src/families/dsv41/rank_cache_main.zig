//! tf-dsv41-rank-cache MODEL_DIR OUT_DIR RANK WORLD [--parity] [--no-dspark] [--index-only]: one rank's weight file
//! (the lane's rank cache, TF_DS_RANK_CACHE) from the checkpoint alone, no GPU and no Python. The entries are
//! plan.wants' (the Python loader's keys and order for the split: TP over 1 or 2 nodes, the exact 2D split over 4;
//! --parity: the balanced experts split, TF_DS_2D_GU=parity), each sliced by the checkpoint reader from its source
//! tensor read whole (checkpoint.readFrom: the bytes the loader keeps); then the index as weights.py RankCache writes it
//! (rank_cache.writeRow), its offset (u64 little-endian) and the magic. The file is OUT_DIR/rank{RANK}of{WORLD}-<the
//! index's sha256, 20 hex digits>.bin, written as a dot file and renamed when whole; OUT_DIR must not hold another file
//! of this rank and world (the lane refuses a folder with two). --index-only: print the index the file would hold to
//! stdout and write nothing.
const std = @import("std");
const dsv41 = @import("dsv41");

const usage = "usage: tf-dsv41-rank-cache MODEL_DIR OUT_DIR RANK WORLD [--parity] [--no-dspark] [--index-only]\n";

fn dim(s: dsv41.checkpoint.Slice, i: usize) usize {
    return if (i < s.rank) s.shape[i] else 1;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
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
    var index_only = false;
    for (args[5..]) |x| {
        if (std.mem.eql(u8, x, "--no-dspark")) {
            dspark = false;
        } else if (std.mem.eql(u8, x, "--parity")) {
            parity = true;
        } else if (std.mem.eql(u8, x, "--index-only")) {
            index_only = true;
        } else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    if (!(world == 1 or world == 2 or world == 4) or rank >= world or (parity and world != 4)) {
        std.debug.print("WORLD is 1, 2 or 4, RANK below it; --parity needs WORLD 4\n{s}", .{usage});
        return 2;
    }
    // four ranks: the exact 2D split (node g = TP2 rank g % 2 of pair g / 2); the file is still rank g of 4
    const split: dsv41.plan.Split = if (world == 4) .{ .rank = rank % 2, .world = 2, .pair = rank / 2, .parity = parity } else .{ .rank = rank, .world = world };
    const cfg = try dsv41.Config.read(a, io, args[1]);
    const want = try dsv41.plan.wants(a, cfg, split, dspark);
    var sh = try dsv41.checkpoint.Shards.open(gpa, io, args[1]);
    defer sh.close(io);

    // every entry's layout first, and from the layouts the index: a missing tensor or a layout the split does not
    // expect stops before a byte is written
    const slices = try a.alloc(dsv41.checkpoint.Slice, want.len);
    var index: std.Io.Writer.Allocating = .init(gpa);
    defer index.deinit();
    const iw = &index.writer;
    try iw.writeByte('[');
    var total: u64 = 0;
    for (want, slices, 0..) |x, *s, i| {
        s.* = dsv41.checkpoint.describe(&sh, x.key) catch |err| {
            std.debug.print("{s}: {t}\n", .{ x.key, err });
            return 1;
        };
        var bad = false;
        if (x.dtype) |d| bad = bad or d != s.dtype;
        if (x.k16) |k| bad = bad or dim(s.*, 0) != k;
        if (x.n16) |n| bad = bad or dim(s.*, 1) != n;
        if (bad) {
            std.debug.print("{s}: the checkpoint's {t} {any} is not the split's\n", .{ x.key, s.dtype, s.shape[0..s.rank] });
            return 1;
        }
        if (i > 0) try iw.writeAll(", ");
        try dsv41.rank_cache.writeRow(iw, x.key, s.dtype, s.shape[0..s.rank], total, s.bytes);
        total += s.bytes;
    }
    try iw.writeByte(']');
    const body = index.written();
    var out_buf: [1 << 16]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    if (index_only) {
        try out.interface.writeAll(body);
        try out.interface.flush();
        return 0;
    }

    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[2], .{ .open_options = .{ .iterate = true } });
    defer dir.close(io);
    if (dsv41.rank_cache.find(a, io, dir, rank, world)) |old| {
        std.debug.print("{s} already holds {s}: remove it or name an empty OUT_DIR\n", .{ args[2], old });
        return 1;
    } else |err| if (err != error.NoRankCache) return err;
    const partial = try std.fmt.allocPrint(a, ".rank{d}of{d}.partial", .{ rank, world });
    var file = try dir.createFile(io, partial, .{});
    var file_open = true;
    defer if (file_open) file.close(io);
    errdefer dir.deleteFile(io, partial) catch {};
    var wbuf: [1 << 20]u8 = undefined;
    var fw = file.writer(io, &wbuf);
    const w = &fw.interface;

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    var whole: std.ArrayList(u8) = .empty;
    defer whole.deinit(gpa);
    var last: ?dsv41.checkpoint.Span = null;
    var at: u64 = 0;
    var report: u64 = 8 << 30;
    for (want, slices) |x, s| {
        // each source tensor in one read (its parts' keys follow one another): slicing it through the map would read
        // the disk a row at a time
        const sp = try dsv41.checkpoint.span(&sh, x.key);
        if (last == null or last.?.file != sp.file or last.?.offset != sp.offset) {
            try whole.resize(gpa, sp.len);
            if (try sp.file.file.readPositionalAll(io, whole.items, sp.offset) != sp.len) return error.ShortRead;
            last = sp;
        }
        try buf.resize(gpa, s.bytes);
        try dsv41.checkpoint.readFrom(&sh, x.key, whole.items, buf.items);
        try w.writeAll(buf.items);
        at += s.bytes;
        if (at >= report) {
            std.debug.print("rank {d} of {d}: {d} of {d} MiB\n", .{ rank, world, at >> 20, total >> 20 });
            report += 8 << 30;
        }
    }
    try w.writeAll(body);
    var tail: [16]u8 = undefined;
    std.mem.writeInt(u64, tail[0..8], at, .little);
    @memcpy(tail[8..], dsv41.rank_cache.magic);
    try w.writeAll(&tail);
    try w.flush();
    try file.sync(io);
    file.close(io);
    file_open = false;

    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(body, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const name = try std.fmt.allocPrint(a, "rank{d}of{d}-{s}.bin", .{ rank, world, hex[0..20] });
    try dir.rename(partial, dir, name, io);
    try out.interface.print("{{\"file\": \"{s}\", \"entries\": {d}, \"data_bytes\": {d}, \"index_bytes\": {d}, \"rank\": {d}, \"world\": {d}, \"parity\": {}, \"dspark\": {}}}\n", .{ name, want.len, at, body.len, rank, world, parity, dspark });
    try out.interface.flush();
    return 0;
}
