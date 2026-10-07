//! Engram rows from the original FP8 tables on local NVMe: a weight row and its scale row a row id, read by offset with
//! one pread each on a pool of threads (engram_io.cpp's gather_rows2), so a step's random rows cost about one latency.
//! gatherDirect reads them through O_DIRECT descriptors (gather_rows2_direct_at: each row's 4 KiB-aligned span into an
//! aligned bounce buffer, then the row copied out; the same bytes, the page cache skipped), and engram_aio.zig submits
//! a decode round's rows as kernel AIO on the same descriptors.
const std = @import("std");
const dio = @import("core").direct_io;

/// One layer's table: [rows, row_w] FP8 weights and [rows, row_s] E8M0 scales, by file and offset; fd_wd / fd_sd the same
/// files through O_DIRECT (-1 where the file system refuses it: the buffered ones serve).
pub const Table = struct { fd_w: std.c.fd_t, base_w: u64, row_w: usize, fd_s: std.c.fd_t, base_s: u64, row_s: usize, rows: u64, fd_wd: std.c.fd_t = -1, fd_sd: std.c.fd_t = -1 };

pub const Tables = struct {
    gpa: std.mem.Allocator,
    fds: std.ArrayList(std.c.fd_t) = .empty,
    /// By Engram layer id (layers.<id>.engram.embed.*).
    layers: std.AutoHashMapUnmanaged(u32, Table) = .empty,

    /// Every "*.safetensors" in `dir` whose header names a layer's engram.embed weight or scale.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Tables {
        var t: Tables = .{ .gpa = gpa };
        errdefer t.close();
        var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var halves: std.AutoHashMapUnmanaged(u32, [2]?struct { fd: std.c.fd_t, dfd: std.c.fd_t, base: u64, row: usize, rows: u64 }) = .empty;
        var it = d.iterate();
        while (try it.next(io)) |e| {
            if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
            const path = try std.fs.path.joinZ(a, &.{ dir, e.name });
            var file = try d.openFile(io, e.name, .{});
            defer file.close(io);
            var head: [8]u8 = undefined;
            if (try file.readPositionalAll(io, &head, 0) != 8) return error.BadSafetensors;
            const n: usize = @intCast(std.mem.readInt(u64, &head, .little));
            const json = try a.alloc(u8, n);
            if (try file.readPositionalAll(io, json, 8) != n) return error.BadSafetensors;
            // the header read here (not core.safetensors: the tables are F8_E4M3 values and F8_E8M0 scales)
            const h = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
            if (h != .object) return error.BadSafetensors;
            var opened: ?std.c.fd_t = null;
            var direct: std.c.fd_t = -1;
            var names = h.object.iterator();
            while (names.next()) |kv| {
                const name = kv.key_ptr.*;
                const which: usize = if (std.mem.endsWith(u8, name, "engram.embed.weight")) 0 else if (std.mem.endsWith(u8, name, "engram.embed.scale")) 1 else continue;
                const layer = try layerOf(name);
                const en = try entry(kv.value_ptr.*);
                if (opened == null) {
                    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
                    if (fd < 0) return error.FileNotFound;
                    try t.fds.append(gpa, fd);
                    opened = fd;
                    // and through O_DIRECT where the file system allows it
                    var df = dio.File.open(path) catch null;
                    if (df) |*f| {
                        if (f.direct) {
                            try t.fds.append(gpa, f.fd);
                            direct = f.fd;
                        } else f.close();
                    }
                }
                const gop = try halves.getOrPut(a, layer);
                if (!gop.found_existing) gop.value_ptr.* = .{ null, null };
                gop.value_ptr[which] = .{ .fd = opened.?, .dfd = direct, .base = 8 + n + en.begin, .row = en.cols * en.size, .rows = en.rows };
            }
        }
        var hv = halves.iterator();
        while (hv.next()) |kv| {
            const w = kv.value_ptr[0] orelse return error.MissingEngramTable;
            const s = kv.value_ptr[1] orelse return error.MissingEngramTable;
            if (w.rows != s.rows) return error.BadEngramTable;
            try t.layers.put(gpa, kv.key_ptr.*, .{ .fd_w = w.fd, .base_w = w.base, .row_w = w.row, .fd_s = s.fd, .base_s = s.base, .row_s = s.row, .rows = w.rows, .fd_wd = w.dfd, .fd_sd = s.dfd });
        }
        return t;
    }

    pub fn close(t: *Tables) void {
        for (t.fds.items) |fd| _ = std.c.close(fd);
        t.fds.deinit(t.gpa);
        t.layers.deinit(t.gpa);
        t.* = undefined;
    }
};

/// A 2-d header entry's rows, columns, element size and data begin (from the data region's start).
fn entry(v: std.json.Value) !struct { rows: u64, cols: usize, size: usize, begin: u64 } {
    if (v != .object) return error.BadSafetensors;
    const dt = v.object.get("dtype") orelse return error.BadSafetensors;
    const shape = v.object.get("shape") orelse return error.BadSafetensors;
    const offs = v.object.get("data_offsets") orelse return error.BadSafetensors;
    if (dt != .string or shape != .array or shape.array.items.len != 2 or offs != .array or offs.array.items.len != 2) return error.BadSafetensors;
    const one = [_][]const u8{ "F8_E4M3", "F8_E5M2", "F8_E8M0", "U8", "I8" };
    const two = [_][]const u8{ "BF16", "F16" };
    var size: usize = 0;
    for (one) |x| if (std.mem.eql(u8, dt.string, x)) {
        size = 1;
    };
    for (two) |x| if (std.mem.eql(u8, dt.string, x)) {
        size = 2;
    };
    if (std.mem.eql(u8, dt.string, "F32")) size = 4;
    if (size == 0) return error.UnsupportedDType;
    for ([_]std.json.Value{ shape.array.items[0], shape.array.items[1], offs.array.items[0] }) |x| if (x != .integer or x.integer < 0) return error.BadSafetensors;
    return .{ .rows = @intCast(shape.array.items[0].integer), .cols = @intCast(shape.array.items[1].integer), .size = size, .begin = @intCast(offs.array.items[0].integer) };
}

fn layerOf(name: []const u8) !u32 {
    // "layers.<id>.engram.embed.weight" (a "model." prefix allowed)
    const at = std.mem.indexOf(u8, name, "layers.") orelse return error.BadEngramTable;
    const rest = name[at + 7 ..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.BadEngramTable;
    return std.fmt.parseInt(u32, rest[0..dot], 10);
}

/// The largest row a direct read takes: its span is at most three 4 KiB blocks (read_row_direct's bounce buffer).
pub const direct_row_max = 2 * dio.alignment;

/// A row through an O_DIRECT descriptor (read_row_direct): its aligned span into `bounce`, then the row copied out.
fn readDirect(fd: std.c.fd_t, bounce: []align(dio.alignment) u8, out: []u8, at: u64) !void {
    const f: dio.File = .{ .fd = fd, .direct = true };
    @memcpy(out, try f.read(bounce, at, out.len));
}

fn preadAll(fd: std.c.fd_t, out: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < out.len) {
        const n = std.c.pread(fd, out.ptr + done, out.len - done, @intCast(at + done));
        if (n < 0) {
            if (std.c.errno(n) == .INTR) continue;
            return error.ReadFailed;
        }
        if (n == 0) return error.EndOfFile;
        done += @intCast(n);
    }
}

/// Reader threads that wait for a gather and share its rows by an atomic counter.
pub const Pool = struct {
    io: std.Io,
    threads: []std.Thread,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    done_cond: std.Io.Condition = .init,
    gen: u64 = 0,
    stop: bool = false,
    job: ?*Job = null,
    /// Worker threads inside work() on the current job: gather() returns (and its job goes) only once they are out.
    active: usize = 0,

    const Job = struct {
        t: Table,
        ids: []const i64,
        out_w: []u8,
        out_s: []u8,
        next: std.atomic.Value(usize) = .init(0),
        finished: std.atomic.Value(usize) = .init(0),
        failed: std.atomic.Value(bool) = .init(false),
        direct: bool = false,
    };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, n: usize) !*Pool {
        const p = try gpa.create(Pool);
        p.* = .{ .io = io, .threads = try gpa.alloc(std.Thread, n) };
        var started: usize = 0;
        errdefer {
            p.shutdown(started);
            gpa.free(p.threads);
            gpa.destroy(p);
        }
        for (p.threads) |*th| {
            th.* = try std.Thread.spawn(.{}, loop, .{p});
            started += 1;
        }
        return p;
    }

    fn shutdown(p: *Pool, n: usize) void {
        p.mutex.lockUncancelable(p.io);
        p.stop = true;
        p.cond.broadcast(p.io);
        p.mutex.unlock(p.io);
        for (p.threads[0..n]) |th| th.join();
    }

    pub fn deinit(p: *Pool, gpa: std.mem.Allocator) void {
        p.shutdown(p.threads.len);
        gpa.free(p.threads);
        gpa.destroy(p);
    }

    fn loop(p: *Pool) void {
        var seen: u64 = 0;
        while (true) {
            p.mutex.lockUncancelable(p.io);
            while (p.gen == seen and !p.stop) p.cond.waitUncancelable(p.io, &p.mutex);
            if (p.stop) {
                p.mutex.unlock(p.io);
                return;
            }
            seen = p.gen;
            const job = p.job;
            if (job != null) p.active += 1;
            p.mutex.unlock(p.io);
            if (job) |j| {
                p.work(j);
                p.mutex.lockUncancelable(p.io);
                p.active -= 1;
                p.done_cond.broadcast(p.io);
                p.mutex.unlock(p.io);
            }
        }
    }

    fn work(p: *Pool, j: *Job) void {
        var bounce: [3 * dio.alignment]u8 align(dio.alignment) = undefined;
        while (true) {
            const i = j.next.fetchAdd(1, .monotonic);
            if (i >= j.ids.len) return;
            const id: u64 = @intCast(j.ids[i]);
            const ok = id < j.t.rows;
            if (ok) {
                const w = j.out_w[i * j.t.row_w ..][0..j.t.row_w];
                const s = j.out_s[i * j.t.row_s ..][0..j.t.row_s];
                if (j.direct and j.t.fd_wd >= 0) {
                    readDirect(j.t.fd_wd, &bounce, w, j.t.base_w + id * j.t.row_w) catch j.failed.store(true, .monotonic);
                } else preadAll(j.t.fd_w, w, j.t.base_w + id * j.t.row_w) catch j.failed.store(true, .monotonic);
                if (j.direct and j.t.fd_sd >= 0) {
                    readDirect(j.t.fd_sd, &bounce, s, j.t.base_s + id * j.t.row_s) catch j.failed.store(true, .monotonic);
                } else preadAll(j.t.fd_s, s, j.t.base_s + id * j.t.row_s) catch j.failed.store(true, .monotonic);
            } else j.failed.store(true, .monotonic);
            if (j.finished.fetchAdd(1, .acq_rel) + 1 == j.ids.len) {
                p.mutex.lockUncancelable(p.io);
                p.done_cond.broadcast(p.io);
                p.mutex.unlock(p.io);
            }
        }
    }

    /// Rows `ids` of table `t` into out_w ([ids.len, row_w]) and out_s ([ids.len, row_s]); this thread reads too.
    pub fn gather(p: *Pool, t: Table, ids: []const i64, out_w: []u8, out_s: []u8) !void {
        return p.run(t, ids, out_w, out_s, false);
    }

    /// gather through the tables' O_DIRECT descriptors (gather_rows2_direct_at), the buffered ones where there are none.
    pub fn gatherDirect(p: *Pool, t: Table, ids: []const i64, out_w: []u8, out_s: []u8) !void {
        if (t.row_w > direct_row_max or t.row_s > direct_row_max) return error.RowTooWide;
        return p.run(t, ids, out_w, out_s, true);
    }

    fn run(p: *Pool, t: Table, ids: []const i64, out_w: []u8, out_s: []u8, direct: bool) !void {
        if (out_w.len != ids.len * t.row_w or out_s.len != ids.len * t.row_s) return error.BadSize;
        if (ids.len == 0) return;
        var job: Job = .{ .t = t, .ids = ids, .out_w = out_w, .out_s = out_s, .direct = direct };
        p.mutex.lockUncancelable(p.io);
        p.job = &job;
        p.gen += 1;
        p.cond.broadcast(p.io);
        p.mutex.unlock(p.io);
        p.work(&job);
        p.mutex.lockUncancelable(p.io);
        while (job.finished.load(.acquire) < ids.len or p.active > 0) p.done_cond.waitUncancelable(p.io, &p.mutex);
        p.job = null;
        p.mutex.unlock(p.io);
        if (job.failed.load(.monotonic)) return error.ReadFailed;
    }
};

test "rows of a toy table by id, on the pool" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a [6, 4] weight table and its [6, 1] scales for layer 14
    const json = "{\"layers.14.engram.embed.weight\":{\"dtype\":\"U8\",\"shape\":[6,4],\"data_offsets\":[0,24]}," ++
        "\"layers.14.engram.embed.scale\":{\"dtype\":\"U8\",\"shape\":[6,1],\"data_offsets\":[24,30]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    for (0..30) |i| try file.append(a, @intCast(i));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var t = try Tables.open(a, io, dir);
    defer t.close();
    const tab = t.layers.get(14).?;
    try std.testing.expectEqual(@as(usize, 4), tab.row_w);
    const pool = try Pool.init(a, io, 3);
    defer pool.deinit(a);
    var w: [3 * 4]u8 = undefined;
    var s: [3]u8 = undefined;
    try pool.gather(tab, &.{ 5, 0, 2 }, &w, &s);
    try std.testing.expectEqualSlices(u8, &.{ 20, 21, 22, 23, 0, 1, 2, 3, 8, 9, 10, 11 }, &w);
    try std.testing.expectEqualSlices(u8, &.{ 29, 24, 26 }, &s);
    // the O_DIRECT path (or the buffered one where the test's file system refuses it): the same bytes
    @memset(&w, 0);
    @memset(&s, 0);
    try pool.gatherDirect(tab, &.{ 5, 0, 2 }, &w, &s);
    try std.testing.expectEqualSlices(u8, &.{ 20, 21, 22, 23, 0, 1, 2, 3, 8, 9, 10, 11 }, &w);
    try std.testing.expectEqualSlices(u8, &.{ 29, 24, 26 }, &s);
    try std.testing.expectError(error.ReadFailed, pool.gather(tab, &.{6}, w[0..4], s[0..1]));
}
