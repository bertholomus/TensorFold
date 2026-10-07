//! Decode rounds' Engram reads without a thread hand-off (the served lane's engram_io.cpp aio_init / aio_start /
//! aio_wait / aio_touch): Linux AIO on the tables' O_DIRECT descriptors. Each read takes its row's 4 KiB-aligned span
//! (one or two blocks) into a bounce slot of a ring; a batch's rows are copied out to their places as its reads
//! complete, reaped by the thread that waits (polling, no sleep): on GB10 a woken reader thread costs ~0.6 ms after an
//! idle round, an AIO submit and reap by the round's own thread ~0.4-0.8 ms for a 6-row window. A touch (a round's
//! first rows read only to wake the drive) goes through a thread of its own and copies nothing. The bytes are the
//! preads' (engram_io.Pool.gather's).
const std = @import("std");
const linux = std.os.linux;
const engram_io = @import("engram_io.zig");

pub const block = 4096;
/// A read's slot: a row's span is at most two blocks (rows are far smaller than a block).
pub const slot = 2 * block;

const aio_context_t = usize;
const cmd_pread: u16 = 0; // IOCB_CMD_PREAD

/// struct iocb (linux/aio_abi.h, little-endian).
const Iocb = extern struct {
    data: u64 = 0,
    key: u32 = 0,
    rw_flags: u32 = 0,
    lio_opcode: u16 = 0,
    reqprio: i16 = 0,
    fildes: u32 = 0,
    buf: u64 = 0,
    nbytes: u64 = 0,
    offset: i64 = 0,
    reserved2: u64 = 0,
    flags: u32 = 0,
    resfd: u32 = 0,
};

/// struct io_event.
const IoEvent = extern struct { data: u64, obj: u64, res: i64, res2: i64 };

fn ioSetup(n: u32, ctx: *aio_context_t) usize {
    return linux.syscall2(.io_setup, n, @intFromPtr(ctx));
}

fn ioDestroy(ctx: aio_context_t) usize {
    return linux.syscall1(.io_destroy, ctx);
}

fn ioSubmit(ctx: aio_context_t, n: usize, cbs: [*]*Iocb) isize {
    return @bitCast(linux.syscall3(.io_submit, ctx, n, @intFromPtr(cbs)));
}

fn ioGetevents(ctx: aio_context_t, min: usize, n: usize, ev: [*]IoEvent, ts: ?*linux.timespec) isize {
    return @bitCast(linux.syscall5(.io_getevents, ctx, min, n, @intFromPtr(ev), @intFromPtr(ts)));
}

const Batch = struct {
    first: u64, // reads [first, first + n) of the ring
    n: usize,
    left: usize, // still in flight
    dst: []?[*]u8, // per read: where its row goes (null: nothing copied, a touch)
    skew: []usize,
    bytes: []usize,
    failed: bool = false,
    touch: bool = false,
};

pub const Aio = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: aio_context_t = 0,
    ring: []align(block) u8,
    slots: usize,
    head: u64 = 0, // reads issued so far (slot = index % slots)
    next_id: u64 = 1,
    batches: std.AutoHashMapUnmanaged(u64, Batch) = .empty,
    live: std.ArrayList([2]u64) = .empty, // (batch id, first read) in issue order
    mutex: std.Io.Mutex = .init,

    /// The AIO context and a ring of `slots` bounce slots (8 KiB each); error.AioUnavailable when the kernel refuses.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, slots: usize) !*Aio {
        const a = try gpa.create(Aio);
        errdefer gpa.destroy(a);
        a.* = .{ .gpa = gpa, .io = io, .ring = try gpa.alignedAlloc(u8, .fromByteUnits(block), slots * slot), .slots = slots };
        errdefer gpa.free(a.ring);
        if (linux.errno(ioSetup(@intCast(@min(slots, 65536)), &a.ctx)) != .SUCCESS) return error.AioUnavailable;
        return a;
    }

    pub fn deinit(a: *Aio) void {
        // every read in flight lands before the ring goes
        a.mutex.lockUncancelable(a.io);
        while (a.inFlight()) a.reap();
        a.mutex.unlock(a.io);
        _ = ioDestroy(a.ctx);
        var it = a.batches.valueIterator();
        while (it.next()) |b| a.freeBatch(b.*);
        a.batches.deinit(a.gpa);
        a.live.deinit(a.gpa);
        a.gpa.free(a.ring);
        a.gpa.destroy(a);
    }

    fn inFlight(a: *Aio) bool {
        var it = a.batches.valueIterator();
        while (it.next()) |b| if (b.left > 0) return true;
        return false;
    }

    fn freeBatch(a: *Aio, b: Batch) void {
        a.gpa.free(b.dst);
        a.gpa.free(b.skew);
        a.gpa.free(b.bytes);
    }

    /// Whatever has completed (no wait): rows copied out, batches counted down. Holds `mutex`.
    fn reap(a: *Aio) void {
        var ev: [256]IoEvent = undefined;
        var zero: linux.timespec = .{ .sec = 0, .nsec = 0 };
        while (true) {
            const r = ioGetevents(a.ctx, 0, ev.len, &ev, &zero);
            if (r <= 0) return;
            for (ev[0..@intCast(r)]) |e| {
                const id = e.data >> 20;
                const j: usize = @intCast(e.data & 0xFFFFF);
                const b = a.batches.getPtr(id) orelse continue;
                if (e.res < @as(i64, @intCast(b.skew[j] + b.bytes[j]))) {
                    b.failed = true;
                } else if (b.dst[j]) |d| {
                    const at = ((b.first + j) % a.slots) * slot + b.skew[j];
                    @memcpy(d[0..b.bytes[j]], a.ring[at..][0..b.bytes[j]]);
                }
                b.left -= 1;
            }
        }
    }

    /// The submission (`mutex` held): rows `ids` of table `t`, a weight then a scale read each, copied into out_w /
    /// out_s as they complete when given (null: a touch). Returns the batch id.
    fn submitLocked(a: *Aio, t: engram_io.Table, ids: []const i64, out_w: ?[]u8, out_s: ?[]u8) !u64 {
        const reads = 2 * ids.len;
        if (reads > a.slots or reads >= 1 << 20) return error.TooManyRows;
        if (t.row_w + block > slot or t.row_s + block > slot) return error.RowTooWide;
        const fd_w = if (t.fd_wd >= 0) t.fd_wd else t.fd_w;
        const fd_s = if (t.fd_sd >= 0) t.fd_sd else t.fd_s;
        // the ring's slots [head, head + reads) must be free: reap until no live batch of the last lap holds them
        while (true) {
            var busy = false;
            for (a.live.items) |l| {
                const b = a.batches.get(l[0]) orelse continue;
                const lap_lo = a.head -| a.slots;
                const lap_hi = (a.head + reads) -| a.slots;
                if (b.left > 0 and l[1] < lap_hi and l[1] + b.n > lap_lo) {
                    busy = true;
                    break;
                }
            }
            if (!busy) break;
            a.reap();
        }
        const id = a.next_id;
        a.next_id += 1;
        var b: Batch = .{
            .first = a.head,
            .n = reads,
            .left = reads,
            .touch = out_w == null,
            .dst = try a.gpa.alloc(?[*]u8, reads),
            .skew = try a.gpa.alloc(usize, reads),
            .bytes = try a.gpa.alloc(usize, reads),
        };
        const cbs = try a.gpa.alloc(Iocb, reads);
        defer a.gpa.free(cbs);
        const ptrs = try a.gpa.alloc(*Iocb, reads);
        defer a.gpa.free(ptrs);
        for (0..reads) |j| {
            const i = j >> 1;
            const sc = j & 1 == 1;
            const rb = if (sc) t.row_s else t.row_w;
            const row: u64 = @intCast(ids[i]);
            if (row >= t.rows) return error.RowOutOfRange;
            const off = (if (sc) t.base_s else t.base_w) + row * rb;
            const lo = std.mem.alignBackward(u64, off, block);
            b.skew[j] = @intCast(off - lo);
            b.bytes[j] = rb;
            b.dst[j] = if (out_w) |w| (if (sc) out_s.?[i * t.row_s ..].ptr else w[i * t.row_w ..].ptr) else null;
            cbs[j] = .{
                .data = (id << 20) | j,
                .lio_opcode = cmd_pread,
                .fildes = @intCast(if (sc) fd_s else fd_w),
                .buf = @intFromPtr(a.ring[((a.head + j) % a.slots) * slot ..].ptr),
                .nbytes = std.mem.alignForward(u64, off + rb - lo, block),
                .offset = @intCast(lo),
            };
            ptrs[j] = &cbs[j];
        }
        a.head += reads;
        try a.batches.put(a.gpa, id, b);
        try a.live.append(a.gpa, .{ id, b.first });
        var sent: usize = 0;
        while (sent < reads) {
            const r = ioSubmit(a.ctx, reads - sent, ptrs[sent..].ptr);
            if (r <= 0) {
                const bp = a.batches.getPtr(id).?;
                bp.left -= reads - sent; // never submitted
                bp.failed = true;
                break;
            }
            sent += @intCast(r);
        }
        // finished touches leave the bookkeeping (a waited-for batch leaves in wait)
        var k: usize = 0;
        while (k < a.live.items.len) {
            const lid = a.live.items[k][0];
            const lb = a.batches.get(lid);
            if (lb == null or (lb.?.touch and lb.?.left == 0)) {
                if (a.batches.fetchRemove(lid)) |kv| a.freeBatch(kv.value);
                _ = a.live.orderedRemove(k);
            } else k += 1;
        }
        return id;
    }

    /// Submit rows `ids` of table `t` (weight and scale rows), landing in out_w ([ids.len, row_w]) and out_s as they
    /// complete (keep both alive until `wait` returns). Returns the batch id.
    pub fn start(a: *Aio, t: engram_io.Table, ids: []const i64, out_w: []u8, out_s: []u8) !u64 {
        if (out_w.len < ids.len * t.row_w or out_s.len < ids.len * t.row_s) return error.BadSize;
        a.mutex.lockUncancelable(a.io);
        defer a.mutex.unlock(a.io);
        return a.submitLocked(t, ids, out_w, out_s);
    }

    /// A touch of rows `ids`: their reads submitted, nothing copied, nobody waits (the lane hands these to a thread of
    /// their own so a slow submit never holds the round's thread; callers that want that run it on one).
    pub fn touch(a: *Aio, t: engram_io.Table, ids: []const i64) !void {
        a.mutex.lockUncancelable(a.io);
        defer a.mutex.unlock(a.io);
        _ = try a.submitLocked(t, ids, null, null);
    }

    /// Reap (polling) until batch `id`'s rows are all in; error.ReadFailed when one failed. A batch is waited for once.
    pub fn wait(a: *Aio, id: u64) !void {
        while (true) {
            a.mutex.lockUncancelable(a.io);
            defer a.mutex.unlock(a.io);
            const b = a.batches.getPtr(id) orelse return;
            a.reap();
            if (b.left == 0) {
                const failed = b.failed;
                if (a.batches.fetchRemove(id)) |kv| a.freeBatch(kv.value);
                for (a.live.items, 0..) |l, k| if (l[0] == id) {
                    _ = a.live.orderedRemove(k);
                    break;
                };
                if (failed) return error.ReadFailed;
                return;
            }
        }
    }
};

test "AIO rows equal the pool's, across a ring that wraps" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a [rows, 100] weight table and its [rows, 3] scales, rows straddling block boundaries
    const rows = 300;
    var json_buf: [256]u8 = undefined;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},100],\"data_offsets\":[0,{d}]}},\"layers.1.engram.embed.scale\":{{\"dtype\":\"U8\",\"shape\":[{d},3],\"data_offsets\":[{d},{d}]}}}}", .{ rows, rows * 100, rows, rows * 100, rows * 103 });
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    for (0..rows * 103) |i| try file.append(a, @intCast((i * 7 + 3) % 251));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var t = try engram_io.Tables.open(a, io, dir);
    defer t.close();
    const tab = t.layers.get(1).?;
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    const aio = Aio.init(a, io, 16) catch |e| switch (e) {
        error.AioUnavailable => return error.SkipZigTest,
        else => return e,
    };
    defer aio.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    for (0..12) |round| {
        var ids: [6]i64 = undefined;
        for (&ids) |*x| x.* = prng.random().intRangeLessThan(i64, 0, rows);
        var want_w: [6 * 100]u8 = undefined;
        var want_s: [6 * 3]u8 = undefined;
        try pool.gather(tab, &ids, &want_w, &want_s);
        var got_w: [6 * 100]u8 = @splat(0);
        var got_s: [6 * 3]u8 = @splat(0);
        if (round % 3 == 0) try aio.touch(tab, ids[0..2]);
        const id = try aio.start(tab, &ids, &got_w, &got_s);
        try aio.wait(id);
        try std.testing.expectEqualSlices(u8, &want_w, &got_w);
        try std.testing.expectEqualSlices(u8, &want_s, &got_s);
    }
    var big_w: [9 * 100]u8 = undefined;
    var big_s: [9 * 3]u8 = undefined;
    const nine: [9]i64 = @splat(0);
    try std.testing.expectError(error.TooManyRows, aio.start(tab, &nine, &big_w, &big_s));
}
