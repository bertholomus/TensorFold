//! tf-dsv41-engram-check ENGRAM_DIR [--rows N] [--seed S] [--slots K]: the Engram reader's three paths on the real
//! tables: for every Engram layer, N random rows read by the pool (buffered preads, engram_io.cpp gather_rows2), by the
//! pool through O_DIRECT (gather_rows2_direct_at) and as AIO batches on the ring (aio_start / aio_wait), every byte
//! compared with the first; one JSON line a layer (bytes, whether the descriptors are direct, each path's time).
const std = @import("std");
const dsv41 = @import("dsv41");
const engram_io = dsv41.engram_io;
const engram_aio = dsv41.engram_aio;

const usage = "usage: tf-dsv41-engram-check ENGRAM_DIR [--rows N] [--seed S] [--slots K]\n";

fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var rows: usize = 4096;
    var seed: u64 = 1;
    var slots: usize = 2048;
    var ai: usize = 2;
    while (ai + 1 < args.len) : (ai += 2) {
        if (std.mem.eql(u8, args[ai], "--rows")) rows = try std.fmt.parseInt(usize, args[ai + 1], 10) else if (std.mem.eql(u8, args[ai], "--seed")) seed = try std.fmt.parseInt(u64, args[ai + 1], 10) else if (std.mem.eql(u8, args[ai], "--slots")) slots = try std.fmt.parseInt(usize, args[ai + 1], 10) else return error.BadArgument;
    }
    var out_buf: [1 << 12]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w = &out.interface;

    var tables = try engram_io.Tables.open(gpa, io, args[1]);
    defer tables.close();
    const pool = try engram_io.Pool.init(gpa, io, 32);
    defer pool.deinit(gpa);
    const aio = try engram_aio.Aio.init(gpa, io, slots);
    defer aio.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    var all_equal = true;
    var it = tables.layers.iterator();
    while (it.next()) |kv| {
        const t = kv.value_ptr.*;
        const ids = try a.alloc(i64, rows);
        for (ids) |*x| x.* = prng.random().intRangeLessThan(i64, 0, @intCast(t.rows));
        const want_w = try a.alloc(u8, rows * t.row_w);
        const want_s = try a.alloc(u8, rows * t.row_s);
        const got_w = try a.alloc(u8, rows * t.row_w);
        const got_s = try a.alloc(u8, rows * t.row_s);
        var t0 = std.Io.Timestamp.now(io, .awake);
        try pool.gather(t, ids, want_w, want_s);
        const buffered_ms = ms(io, t0);
        @memset(got_w, 0);
        @memset(got_s, 0);
        t0 = std.Io.Timestamp.now(io, .awake);
        try pool.gatherDirect(t, ids, got_w, got_s);
        const direct_ms = ms(io, t0);
        const direct_equal = std.mem.eql(u8, want_w, got_w) and std.mem.eql(u8, want_s, got_s);
        @memset(got_w, 0);
        @memset(got_s, 0);
        // AIO in batches a round could make (up to half the ring's slots: a weight and a scale read a row)
        const batch = @max(1, slots / 2);
        t0 = std.Io.Timestamp.now(io, .awake);
        var at: usize = 0;
        while (at < rows) : (at += batch) {
            const n = @min(batch, rows - at);
            const id = try aio.start(t, ids[at..][0..n], got_w[at * t.row_w ..][0 .. n * t.row_w], got_s[at * t.row_s ..][0 .. n * t.row_s]);
            try aio.wait(id);
        }
        const aio_ms = ms(io, t0);
        const aio_equal = std.mem.eql(u8, want_w, got_w) and std.mem.eql(u8, want_s, got_s);
        all_equal = all_equal and direct_equal and aio_equal;
        try w.print("{{\"layer\": {d}, \"rows\": {d}, \"row_w\": {d}, \"row_s\": {d}, \"direct_fds\": {}, \"direct_equal\": {}, \"aio_equal\": {}, \"buffered_ms\": {d:.2}, \"direct_ms\": {d:.2}, \"aio_ms\": {d:.2}}}\n", .{ kv.key_ptr.*, rows, t.row_w, t.row_s, t.fd_wd >= 0 and t.fd_sd >= 0, direct_equal, aio_equal, buffered_ms, direct_ms, aio_ms });
        try w.flush();
    }
    try w.print("{{\"layers\": {d}, \"all_equal\": {}}}\n", .{ tables.layers.count(), all_equal });
    try w.flush();
    return if (all_equal) 0 else 1;
}
