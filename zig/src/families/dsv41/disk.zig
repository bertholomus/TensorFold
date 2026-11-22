//! DSV41 kept states on disk (--learn-conversations): the rows store serializes rank 0's image; rank 1
//! keeps its half under a per-rank file named by the state's disk key, loaded on restore when its copy went.
const std = @import("std");
const api = @import("engine_api");
const lanes_mod = @import("lanes.zig");
const cuda = @import("cuda");
const rows = api.prompt_cache.rows;
const Lanes = lanes_mod.Lanes;
const Snap = lanes_mod.Snap;
const Allocator = std.mem.Allocator;

const magic: u32 = 0x4453_5634; // "DSV4"
pub const Head = extern struct { magic: u32 = magic, version: u32 = 1, at: u64, disk_id: u64, bytes: u64, hash: u64 };

/// A rank's file for the disk state `key` (the rows state's name, low 64 bits) under `dir`.
pub fn path(buf: []u8, dir: []const u8, key: u64) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/ds41-{x:0>16}.snap", .{ dir, key }, 0);
}

/// The disk state's bytes at `at` (one image part: the packed prefix, `at` tokens of this rank's rows).
pub fn snapBytes(m: anytype, at: u32) u64 {
    return m.snapBytes(at);
}

/// Write `payload` (this rank's half, `at` tokens) as `key`'s file under `dir`; partial writes never appear.
pub fn writeFile(dir: []const u8, key: u64, at: u64, payload: []const u8) !void {
    var pbuf: [1200]u8 = undefined;
    const file = try path(&pbuf, dir, key);
    var tbuf: [1200]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tbuf, "{s}.part", .{file}, 0);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.SnapshotWrite;
    errdefer _ = std.c.unlink(tmp);
    {
        defer _ = std.c.close(fd);
        const head: Head = .{ .at = at, .disk_id = key, .bytes = payload.len, .hash = std.hash.Wyhash.hash(0x6473, payload) };
        try put(fd, std.mem.asBytes(&head));
        try put(fd, payload);
        if (std.c.fsync(fd) != 0) return error.SnapshotWrite;
    }
    if (std.c.rename(tmp, file) != 0) return error.SnapshotWrite;
}

/// `key`'s file under `dir` into `out` (exactly `head.bytes` bytes): its header, checksum and length must agree.
pub fn readFile(dir: []const u8, key: u64, out: []u8) !void {
    var pbuf: [1200]u8 = undefined;
    const file = try path(&pbuf, dir, key);
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.SnapshotRead;
    defer _ = std.c.close(fd);
    var head: Head = undefined;
    if (std.c.pread(fd, std.mem.asBytes(&head).ptr, @sizeOf(Head), 0) != @sizeOf(Head)) return error.SnapshotRead;
    if (head.magic != magic or head.version != 1 or head.disk_id != key or head.bytes != out.len) return error.SnapshotRead;
    try get(fd, out, @sizeOf(Head));
    if (std.hash.Wyhash.hash(0x6473, out) != head.hash) return error.SnapshotRead;
}

/// `key`'s file under `dir` removed (missing is fine).
pub fn unlinkFile(dir: []const u8, key: u64) void {
    var pbuf: [1200]u8 = undefined;
    const file = path(&pbuf, dir, key) catch return;
    _ = std.c.unlink(file);
}

/// This rank's spill files under `dir` trimmed to `cap` bytes, oldest modification first; `keep` stays.
pub fn trimDir(gpa: Allocator, dir: []const u8, cap: u64, keep: u64) void {
    const Entry = struct { name: [64]u8, len: usize, bytes: u64, mtime: i64 };
    var list: std.ArrayList(Entry) = .empty;
    defer list.deinit(gpa);
    var dbuf: [1100]u8 = undefined;
    const dz = std.fmt.bufPrintSentinel(&dbuf, "{s}", .{dir}, 0) catch return;
    const d = std.c.opendir(dz) orelse return;
    defer _ = std.c.closedir(d);
    var total: u64 = 0;
    while (std.c.readdir(d)) |ent| {
        if (ent.type != std.c.DT.REG) continue;
        const name = std.mem.sliceTo(&ent.name, 0);
        if (!std.mem.startsWith(u8, name, "ds41-") or !std.mem.endsWith(u8, name, ".snap")) continue;
        var fbuf: [1400]u8 = undefined;
        const file = std.fmt.bufPrintSentinel(&fbuf, "{s}/{s}", .{ dir, name }, 0) catch continue;
        var sx: std.os.linux.Statx = undefined;
        if (std.c.statx(std.posix.AT.FDCWD, file, 0, .{ .MTIME = true, .SIZE = true }, &sx) != 0) continue;
        var e: Entry = .{ .name = undefined, .len = name.len, .bytes = @intCast(@max(sx.size, 0)), .mtime = sx.mtime.sec };
        @memcpy(e.name[0..name.len], name);
        list.append(gpa, e) catch return;
        total += e.bytes;
    }
    if (total <= cap) return;
    std.mem.sort(Entry, list.items, {}, struct {
        fn lt(_: void, a: Entry, b: Entry) bool {
            return a.mtime < b.mtime;
        }
    }.lt);
    for (list.items) |e| {
        if (total <= cap) break;
        if (std.mem.eql(u8, e.name[0..e.len], "")) continue;
        var kbuf: [24]u8 = undefined;
        const ktext = e.name[5 .. 5 + 16];
        const key = std.fmt.parseInt(u64, ktext, 16) catch 0;
        if (key == keep) continue;
        var fbuf: [1400]u8 = undefined;
        const file = std.fmt.bufPrintSentinel(&fbuf, "{s}/{s}", .{ dir, e.name[0..e.len] }, 0) catch continue;
        _ = &kbuf;
        if (std.c.unlink(file) == 0) total -= e.bytes;
    }
}

fn of(ptr: *anyopaque) *Lanes {
    return @ptrCast(@alignCast(ptr));
}

fn savedOf(saved: api.prompt_cache.Saved) *Snap {
    return @ptrCast(@alignCast(saved));
}

/// The store's view of a kept state: a single part, this rank's packed prefix of `at` tokens.
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    try out.append(a, .{ .bytes = of(ptr).m.snapBytes(at) });
}

/// Ranges of a resident state out of its device buffer; the first gather after the state was stamped
/// publishes its disk key to the peer, which writes its own half's file in step.
fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const self = of(ptr);
    const sn = savedOf(saved);
    if (sn.disk_id != 0 and !sn.persisted) try self.publishPersist(sn);
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        defer o += len;
        try sn.buf.download(@intCast(g.off), out[o..][0..len]);
    }
}

/// A new kept state of `at` tokens for the store's disk reads: this rank's buffer registered, the peer's
/// half comes from its file on the restore that follows.
fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const self = of(ptr);
    const sn = try self.gpa.create(Snap);
    errdefer self.gpa.destroy(sn);
    sn.* = .{ .id = self.next_snap, .at = at, .base = 0, .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes(at)), .backed = true };
    errdefer sn.buf.free();
    try self.snaps.append(self.gpa, sn);
    self.next_snap += 1;
    return sn;
}

fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const sn = savedOf(saved);
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        try sn.buf.upload(@intCast(g.off), in[o..][0..len]);
        o += len;
    }
}

/// A state no longer resident (overwritten rows, dropped copy) cannot be read whole.
fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
    const sn = savedOf(saved);
    return std.mem.indexOfScalar(*Snap, of(ptr).snaps.items, sn) != null and (sn.backed or !sn.stale);
}

/// Every byte of `b` written, or a write error.
fn put(fd: c_int, b: []const u8) !void {
    var off: usize = 0;
    while (off < b.len) {
        const n = std.c.write(fd, b.ptr + off, b.len - off);
        if (n <= 0) return error.SnapshotWrite;
        off += @intCast(n);
    }
}

/// Every byte of `out` read at `pos`, or a read error.
fn get(fd: c_int, out: []u8, pos: usize) !void {
    var off: usize = 0;
    while (off < out.len) {
        const n = std.c.pread(fd, out.ptr + off, out.len - off, @intCast(pos + off));
        if (n <= 0) return error.SnapshotRead;
        off += @intCast(n);
    }
}
