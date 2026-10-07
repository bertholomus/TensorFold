//! The Python lane's per-rank weight file (TF_DS_RANK_CACHE, weights.py RankCache): this rank's slices of the checkpoint
//! in the loader's order and dtypes, a JSON index after them, then the index's offset and a magic word. Both engines load
//! from the one file.
const std = @import("std");
const Io = std.Io;

pub const magic = "TFDSRK01";
pub const max_rank = 4;

/// The dtypes the Python loader writes, by torch's names.
pub const DType = enum {
    bf16,
    f16,
    f32,
    i16,
    i32,
    i64,
    u8,
    f8e4m3,

    pub fn size(self: DType) usize {
        return switch (self) {
            .u8, .f8e4m3 => 1,
            .bf16, .f16, .i16 => 2,
            .f32, .i32 => 4,
            .i64 => 8,
        };
    }

    pub fn parse(text: []const u8) ?DType {
        const names = .{ .{ "bfloat16", .bf16 }, .{ "float16", .f16 }, .{ "float32", .f32 }, .{ "int16", .i16 }, .{ "int32", .i32 }, .{ "int64", .i64 }, .{ "uint8", .u8 }, .{ "float8_e4m3fn", .f8e4m3 } };
        inline for (names) |n| if (std.mem.eql(u8, text, n[0])) return n[1];
        return null;
    }
};

/// One tensor in the file: its bytes are [offset, offset + bytes).
pub const Entry = struct {
    dtype: DType,
    rank: u8,
    shape: [max_rank]usize,
    offset: u64,
    bytes: u64,

    pub fn dim(self: Entry, i: usize) usize {
        return if (i < self.rank) self.shape[i] else 1;
    }

    pub fn numel(self: Entry) usize {
        var n: usize = 1;
        for (self.shape[0..self.rank]) |d| n *= d;
        return n;
    }
};

/// The index: entries by the loader's key ("name|dtype" for a plain tensor, "prefix|cols|rows|part" for an EXL3 part),
/// in file order.
pub const Index = struct {
    arena: std.heap.ArenaAllocator,
    entries: std.StringArrayHashMapUnmanaged(Entry),
    data_end: u64,

    pub fn get(self: *const Index, key: []const u8) ?Entry {
        return self.entries.get(key);
    }

    pub fn deinit(self: *Index) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

fn uint(v: std.json.Value) ?u64 {
    return switch (v) {
        .integer => |i| std.math.cast(u64, i),
        .number_string => |s| std.fmt.parseInt(u64, s, 10) catch null,
        else => null,
    };
}

/// The index JSON (a list of [key, dtype, shape, offset, bytes]); every entry must lie inside [0, data_end), in order.
pub fn parseIndex(gpa: std.mem.Allocator, json: []const u8, data_end: u64) !Index {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
    if (v != .array) return error.BadRankCache;
    var out: std.StringArrayHashMapUnmanaged(Entry) = .empty;
    var at: u64 = 0;
    for (v.array.items) |row| {
        if (row != .array or row.array.items.len != 5) return error.BadRankCache;
        const f = row.array.items;
        if (f[0] != .string or f[1] != .string or f[2] != .array) return error.BadRankCache;
        const dtype = DType.parse(f[1].string) orelse return error.UnsupportedDType;
        const shape = f[2].array.items;
        if (shape.len > max_rank) return error.BadRankCache;
        var e: Entry = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .offset = uint(f[3]) orelse return error.BadRankCache, .bytes = uint(f[4]) orelse return error.BadRankCache };
        var n: u64 = dtype.size();
        for (shape, 0..) |d, i| {
            e.shape[i] = @intCast(uint(d) orelse return error.BadRankCache);
            n = std.math.mul(u64, n, e.shape[i]) catch return error.BadRankCache;
        }
        if (n != e.bytes or e.offset != at or e.offset + e.bytes > data_end) return error.BadRankCache;
        at = e.offset + e.bytes;
        const gop = try out.getOrPut(a, f[0].string);
        if (gop.found_existing) return error.BadRankCache;
        gop.value_ptr.* = e;
    }
    return .{ .arena = arena, .entries = out, .data_end = data_end };
}

/// The index of the file `name` in `dir`: its footer names where the JSON starts.
pub fn open(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, name: []const u8) !Index {
    var file = try dir.openFile(io, name, .{});
    defer file.close(io);
    const len = try file.length(io);
    if (len < 16) return error.BadRankCache;
    var tail: [16]u8 = undefined;
    if (try file.readPositionalAll(io, &tail, len - 16) != 16) return error.BadRankCache;
    if (!std.mem.eql(u8, tail[8..], magic)) return error.BadRankCache;
    const at = std.mem.readInt(u64, tail[0..8], .little);
    if (at > len - 16 or len - 16 - at > 1 << 30) return error.BadRankCache;
    const json = try gpa.alloc(u8, @intCast(len - 16 - at));
    defer gpa.free(json);
    if (try file.readPositionalAll(io, json, at) != json.len) return error.BadRankCache;
    return parseIndex(gpa, json, at);
}

/// The file in `dir` for rank `rank` of `world` (rank{r}of{w}-<key>.bin, the Python loader keeps one a split);
/// the name is `gpa`'s.
pub fn find(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, rank: u32, world: u32) ![]u8 {
    var prefix_buf: [32]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&prefix_buf, "rank{d}of{d}-", .{ rank, world });
    var found: ?[]u8 = null;
    errdefer if (found) |f| gpa.free(f);
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file or !std.mem.startsWith(u8, e.name, prefix) or !std.mem.endsWith(u8, e.name, ".bin")) continue;
        if (found != null) return error.SeveralRankCaches;
        found = try gpa.dupe(u8, e.name);
    }
    return found orelse error.NoRankCache;
}

/// A cache file as weights.py writes one, for the tests: each tensor's bytes, then the index, offset and magic.
fn writeFake(a: std.mem.Allocator, io: Io, dir: Io.Dir, name: []const u8, rows: []const struct { key: []const u8, dtype: []const u8, shape: []const usize, fill: u8 }) !void {
    var body: std.ArrayList(u8) = .empty;
    var index: std.ArrayList(u8) = .empty;
    try index.append(a, '[');
    for (rows, 0..) |r, i| {
        var n: usize = DType.parse(r.dtype).?.size();
        for (r.shape) |d| n *= d;
        const at = body.items.len;
        try body.appendNTimes(a, r.fill, n);
        if (i > 0) try index.appendSlice(a, ", ");
        try index.print(a, "[\"{s}\", \"{s}\", [", .{ r.key, r.dtype });
        for (r.shape, 0..) |d, k| try index.print(a, "{s}{d}", .{ if (k > 0) ", " else "", d });
        try index.print(a, "], {d}, {d}]", .{ at, n });
    }
    try index.append(a, ']');
    const at = body.items.len;
    try body.appendSlice(a, index.items);
    var word: [8]u8 = undefined;
    std.mem.writeInt(u64, &word, at, .little);
    try body.appendSlice(a, &word);
    try body.appendSlice(a, magic);
    try dir.writeFile(io, .{ .sub_path = name, .data = body.items });
}

test "a rank cache's index reads back in order, and its file is found by rank" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeFake(a, io, tmp.dir, "rank1of2-0123456789abcdef0123.bin", &.{
        .{ .key = "embed.weight|None", .dtype = "bfloat16", .shape = &.{ 4, 8 }, .fill = 1 },
        .{ .key = "layers.0.attn.wq_b|(0, 16384)|None|tr", .dtype = "int16", .shape = &.{ 2, 3, 48 }, .fill = 2 },
        .{ .key = "layers.0.attn.attn_sink|torch.float32", .dtype = "float32", .shape = &.{64}, .fill = 3 },
    });
    try writeFake(a, io, tmp.dir, "rank0of2-ffffffffffffffffffff.bin", &.{});
    const name = try find(a, io, tmp.dir, 1, 2);
    try std.testing.expectEqualStrings("rank1of2-0123456789abcdef0123.bin", name);
    try std.testing.expectError(error.NoRankCache, find(a, io, tmp.dir, 0, 4));
    var ix = try open(std.testing.allocator, io, tmp.dir, name);
    defer ix.deinit();
    try std.testing.expectEqual(@as(usize, 3), ix.entries.count());
    const tr = ix.get("layers.0.attn.wq_b|(0, 16384)|None|tr").?;
    try std.testing.expectEqual(DType.i16, tr.dtype);
    try std.testing.expectEqual(@as(u64, 64), tr.offset);
    try std.testing.expectEqual(@as(usize, 288), tr.numel());
    try std.testing.expectEqual(@as(u64, 64 + 576 + 256), ix.data_end);
    try std.testing.expectEqualStrings("embed.weight|None", ix.entries.keys()[0]);
}

test "an index whose entries overlap, leave gaps or disagree with their shapes is refused" {
    const bad = [_][]const u8{
        "[[\"a\", \"float32\", [2], 0, 8], [\"b\", \"float32\", [2], 4, 8]]",
        "[[\"a\", \"float32\", [2], 0, 8], [\"b\", \"float32\", [2], 12, 8]]",
        "[[\"a\", \"float32\", [3], 0, 8]]",
        "[[\"a\", \"float32\", [2], 0, 8], [\"a\", \"float32\", [2], 8, 8]]",
        "[[\"a\", \"float32\", [8], 0, 32]]",
        "{\"a\": 1}",
    };
    for (bad) |json| try std.testing.expectError(error.BadRankCache, parseIndex(std.testing.allocator, json, 16));
    try std.testing.expectError(error.UnsupportedDType, parseIndex(std.testing.allocator, "[[\"a\", \"complex64\", [1], 0, 8]]", 16));
}
