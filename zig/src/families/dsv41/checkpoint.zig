//! The checkpoint's tensors as the split takes them, by the loader's keys (plan.zig): whole, by input rows or by
//! output columns, in the dtypes the Python loader keeps, so each entry's bytes equal the lane's rank cache entry.
const std = @import("std");
const core = @import("core");
const st = core.safetensors;
const DType = @import("rank_cache.zig").DType;

/// One planned entry's layout once read.
pub const Slice = struct {
    dtype: DType,
    rank: u8,
    shape: [st.max_rank]usize,
    bytes: usize,
};

/// Every shard of a checkpoint folder, mapped read-only, with its tensors by name.
pub const Shards = struct {
    gpa: std.mem.Allocator,
    files: []st.File,
    names: std.StringHashMapUnmanaged(u32) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Shards {
        var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var paths: std.ArrayList([]const u8) = .empty;
        defer {
            for (paths.items) |p| gpa.free(p);
            paths.deinit(gpa);
        }
        var it = d.iterate();
        while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) {
            try paths.append(gpa, try std.fs.path.join(gpa, &.{ dir, e.name }));
        };
        std.mem.sort([]const u8, paths.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        var sh: Shards = .{ .gpa = gpa, .files = try gpa.alloc(st.File, paths.items.len) };
        var n: usize = 0;
        errdefer {
            for (sh.files[0..n]) |*f| f.close(io);
            gpa.free(sh.files);
            sh.names.deinit(gpa);
        }
        for (paths.items) |p| {
            sh.files[n] = try st.File.open(gpa, io, p);
            n += 1;
            var names = sh.files[n - 1].names.iterator();
            while (names.next()) |kv| try sh.names.put(gpa, kv.key_ptr.*, @intCast(n - 1));
        }
        return sh;
    }

    pub fn close(sh: *Shards, io: std.Io) void {
        for (sh.files) |*f| f.close(io);
        sh.gpa.free(sh.files);
        sh.names.deinit(sh.gpa);
        sh.* = undefined;
    }

    pub fn get(sh: *const Shards, name: []const u8) ?st.Tensor {
        const i = sh.names.get(name) orelse return null;
        return sh.files[i].get(name);
    }
};

/// A loader key split into its fields: "name|dtype" (plain) or "prefix|cols|rows|part" (an EXL3 part).
const Key = struct {
    name: []const u8,
    dtype: ?DType = null,
    exl3: bool = false,
    cols: ?[2]usize = null,
    rows: ?[2]usize = null,
    part: []const u8 = "",

    fn parse(key: []const u8) !Key {
        var f: [4][]const u8 = undefined;
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, key, '|');
        while (it.next()) |x| : (n += 1) {
            if (n == 4) return error.BadKey;
            f[n] = x;
        }
        if (n == 2) {
            const d: ?DType = if (std.mem.eql(u8, f[1], "None")) null else if (std.mem.eql(u8, f[1], "torch.float32")) .f32 else if (std.mem.eql(u8, f[1], "torch.float16")) .f16 else return error.BadKey;
            return .{ .name = f[0], .dtype = d };
        }
        if (n != 4) return error.BadKey;
        return .{ .name = f[0], .exl3 = true, .cols = try range(f[1]), .rows = try range(f[2]), .part = f[3] };
    }

    /// "None" or Python's "(a, b)".
    fn range(text: []const u8) !?[2]usize {
        if (std.mem.eql(u8, text, "None")) return null;
        if (text.len < 5 or text[0] != '(' or text[text.len - 1] != ')') return error.BadKey;
        var it = std.mem.splitSequence(u8, text[1 .. text.len - 1], ", ");
        const a = try std.fmt.parseInt(usize, it.next() orelse return error.BadKey, 10);
        const b = try std.fmt.parseInt(usize, it.next() orelse return error.BadKey, 10);
        if (it.next() != null or b < a) return error.BadKey;
        return .{ a, b };
    }
};

fn dtypeOf(d: st.DType) !DType {
    return switch (d) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        .i16 => .i16,
        .i32 => .i32,
        .i64 => .i64,
        .u8 => .u8,
        else => error.UnsupportedDType,
    };
}

const Source = struct { t: st.Tensor, name: []const u8 };

fn source(sh: *const Shards, k: Key, buf: []u8) !Source {
    const name = if (!k.exl3) k.name else blk: {
        const tail = if (std.mem.eql(u8, k.part, "tr")) "trellis" else if (std.mem.eql(u8, k.part, "suh")) "suh" else if (std.mem.eql(u8, k.part, "svh")) "svh" else return error.BadKey;
        break :blk try std.fmt.bufPrint(buf, "{s}.{s}", .{ k.name, tail });
    };
    return .{ .t = sh.get(name) orelse return error.MissingTensor, .name = name };
}

/// The layout `key` reads into.
pub fn describe(sh: *const Shards, key: []const u8) !Slice {
    const k = try Key.parse(key);
    var buf: [256]u8 = undefined;
    const s = try source(sh, k, &buf);
    var out: Slice = .{ .dtype = try dtypeOf(s.t.dtype), .rank = s.t.rank, .shape = s.t.shape, .bytes = 0 };
    if (k.dtype) |d| out.dtype = d;
    if (k.exl3) {
        if (std.mem.eql(u8, k.part, "tr")) {
            if (k.rows) |r| out.shape[0] = (r[1] - r[0]) / 16;
            if (k.cols) |c| out.shape[1] = (c[1] - c[0]) / 16;
        } else if (std.mem.eql(u8, k.part, "suh")) {
            if (k.rows) |r| out.shape[0] = r[1] - r[0];
        } else if (k.cols) |c| out.shape[0] = c[1] - c[0];
    }
    var n: usize = out.dtype.size();
    for (out.shape[0..out.rank]) |d| n *= d;
    out.bytes = n;
    return out;
}

/// Where `key`'s source tensor lies in its shard (a reader that takes the whole tensor in one read, then readFrom).
pub const Span = struct { file: *const st.File, offset: usize, len: usize };

pub fn span(sh: *const Shards, key: []const u8) !Span {
    const k = try Key.parse(key);
    var buf: [256]u8 = undefined;
    const s = try source(sh, k, &buf);
    const f = &sh.files[sh.names.get(s.name).?];
    return .{ .file = f, .offset = @intFromPtr(s.t.bytes.ptr) - @intFromPtr(f.map.memory.ptr), .len = s.t.bytes.len };
}

/// `key`'s bytes into `out` (describe(key).bytes long).
pub fn read(sh: *const Shards, key: []const u8, out: []u8) !void {
    return readFrom(sh, key, null, out);
}

/// read, the source tensor's bytes taken from `bytes` (its span, read by the caller) instead of the mapped shard: a
/// column slice touches every row of its tensor, which through the map is a disk read a row.
pub fn readFrom(sh: *const Shards, key: []const u8, bytes: ?[]const u8, out: []u8) !void {
    const k = try Key.parse(key);
    var buf: [256]u8 = undefined;
    const s = try source(sh, k, &buf);
    var t = s.t;
    if (bytes) |b| {
        if (b.len != t.bytes.len) return error.BadSize;
        t.bytes = b;
    }
    const want = try describe(sh, key);
    if (out.len != want.bytes) return error.BadSize;
    if (!k.exl3) {
        const from = try dtypeOf(t.dtype);
        if (k.dtype == null or k.dtype.? == from) return @memcpy(out, t.bytes);
        if (from == .f16 and k.dtype.? == .f32) return widenHalf(t.bytes, out);
        return error.UnsupportedConversion;
    }
    const es = (try dtypeOf(t.dtype)).size();
    if (std.mem.eql(u8, k.part, "tr")) {
        const row = t.dim(1) * t.dim(2) * es; // one K tile row: every N tile's words
        const k0 = if (k.rows) |r| r[0] / 16 else 0;
        if (k.cols) |c| { // output columns, of the given input rows when both are set (the 2D split's wo_b, down, Engram)
            const tile = t.dim(2) * es;
            const lo = c[0] / 16 * tile;
            const w = (c[1] - c[0]) / 16 * tile;
            const kt = if (k.rows) |r| (r[1] - r[0]) / 16 else t.dim(0);
            for (0..kt) |i| @memcpy(out[i * w ..][0..w], t.bytes[(k0 + i) * row + lo ..][0..w]);
            return;
        }
        if (k.rows != null) return @memcpy(out, t.bytes[k0 * row ..][0..out.len]);
        return @memcpy(out, t.bytes);
    }
    const sel: ?[2]usize = if (std.mem.eql(u8, k.part, "suh")) k.rows else k.cols;
    if (sel) |r| return @memcpy(out, t.bytes[r[0] * es ..][0..out.len]);
    @memcpy(out, t.bytes);
}

/// f16 -> f32, exact (the loader's .to(torch.float32) of an f16 tensor).
fn widenHalf(src: []const u8, out: []u8) void {
    const n = src.len / 2;
    for (0..n) |i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, src[2 * i ..][0..2], .little));
        std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(@as(f32, h)), .little);
    }
}

test "loader keys parse as Python prints them" {
    const p = try Key.parse("layers.3.ffn.experts.17.w2|None|(1152, 2304)|tr");
    try std.testing.expect(p.exl3 and p.cols == null and p.rows.?[0] == 1152 and p.rows.?[1] == 2304);
    const q = try Key.parse("layers.0.ffn.gate.bias|torch.float32");
    try std.testing.expect(!q.exl3 and q.dtype.? == .f32);
    try std.testing.expectError(error.BadKey, Key.parse("a|b|c"));
    try std.testing.expectError(error.BadKey, Key.parse("a|(2, 1)|None|tr"));
}

test "column, row and whole slices of a toy trellis and its scales" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a trellis [2, 4, 2] i16 (K 32, N 64, two words a tile), its suh [32] f16 and svh [64] f16, a bias in f16
    var body: [16 * 2 + 32 * 2 + 64 * 2 + 4 * 2]u8 = undefined;
    for (&body, 0..) |*x, i| x.* = @truncate(i);
    const json = "{\"w.trellis\":{\"dtype\":\"I16\",\"shape\":[2,4,2],\"data_offsets\":[0,32]}," ++
        "\"w.suh\":{\"dtype\":\"F16\",\"shape\":[32],\"data_offsets\":[32,96]}," ++
        "\"w.svh\":{\"dtype\":\"F16\",\"shape\":[64],\"data_offsets\":[96,224]}," ++
        "\"g.bias\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[224,232]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    try file.appendSlice(a, &body);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var sh = try Shards.open(a, io, dir);
    defer sh.close(io);
    // output columns 32..64 of the trellis: tiles 2 and 3 of each K row (bytes 8..16 and 24..32)
    var cols: [16]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 16), (try describe(&sh, "w|(32, 64)|None|tr")).bytes);
    try read(&sh, "w|(32, 64)|None|tr", &cols);
    var expect: [16]u8 = undefined;
    @memcpy(expect[0..8], body[8..16]);
    @memcpy(expect[8..16], body[24..32]);
    try std.testing.expectEqualSlices(u8, &expect, &cols);
    var rows: [16]u8 = undefined;
    try read(&sh, "w|None|(16, 32)|tr", &rows);
    try std.testing.expectEqualSlices(u8, body[16..32], &rows);
    // both: K tile row 1, its tiles 2 and 3 (the 2D split's down and wo_b keys)
    var both: [8]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 8), (try describe(&sh, "w|(32, 64)|(16, 32)|tr")).bytes);
    try read(&sh, "w|(32, 64)|(16, 32)|tr", &both);
    try std.testing.expectEqualSlices(u8, body[24..32], &both);
    var suh: [32]u8 = undefined;
    try read(&sh, "w|None|(16, 32)|suh", &suh);
    try std.testing.expectEqualSlices(u8, body[32 + 32 .. 96], &suh);
    var svh: [64]u8 = undefined;
    try read(&sh, "w|(32, 64)|None|svh", &svh);
    try std.testing.expectEqualSlices(u8, body[96 + 64 .. 224], &svh);
    // the same from a copy of the tensor's span, as tf-dsv41-rank-cache reads it
    const sp = try span(&sh, "w|(32, 64)|(16, 32)|tr");
    try std.testing.expectEqual(@as(usize, 8 + json.len), sp.offset);
    try std.testing.expectEqual(@as(usize, 32), sp.len);
    var copy: [32]u8 = undefined;
    @memcpy(&copy, file.items[sp.offset..][0..sp.len]);
    var both2: [8]u8 = undefined;
    try readFrom(&sh, "w|(32, 64)|(16, 32)|tr", &copy, &both2);
    try std.testing.expectEqualSlices(u8, &both, &both2);
    try std.testing.expectError(error.BadSize, readFrom(&sh, "w|(32, 64)|(16, 32)|tr", copy[0..16], &both2));
    var wide: [16]u8 = undefined;
    try read(&sh, "g.bias|torch.float32", &wide);
    const h0: f16 = @bitCast(std.mem.readInt(u16, body[224..226], .little));
    try std.testing.expectEqual(@as(f32, h0), @as(f32, @bitCast(std.mem.readInt(u32, wide[0..4], .little))));
}
