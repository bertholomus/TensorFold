//! Each logits column's token as the bytes it adds to a reply, so a grammar can test a token by its bytes.
const std = @import("std");
const json = @import("json.zig");
const Allocator = std.mem.Allocator;

/// How a token meets a grammar: text by its bytes, eos once the grammar is complete, special and none never.
pub const Kind = enum(u8) { none, text, eos, special };

pub const Vocab = struct {
    /// Token id i's bytes are bytes[offsets[i]..offsets[i + 1]] (empty unless its kind is text).
    offsets: []u32,
    bytes: []u8,
    kinds: []Kind,

    /// The table for logits `columns` wide; `source` is a tokenizer.Tokenizer (its `tokenBytes` and `added`).
    pub fn build(a: Allocator, columns: usize, source: anytype, eos: []const u32) !Vocab {
        const kinds = try a.alloc(Kind, columns);
        errdefer a.free(kinds);
        @memset(kinds, .text);
        // An added token is control markup decided by its id, though its text may decode to plain bytes.
        for (source.added) |t| {
            if (t.id < columns) kinds[t.id] = .special;
        }
        for (eos) |id| {
            if (id < columns) kinds[id] = .eos;
        }
        const offsets = try a.alloc(u32, columns + 1);
        errdefer a.free(offsets);
        var bytes: std.ArrayList(u8) = .empty;
        errdefer bytes.deinit(a);
        var scratch = std.heap.ArenaAllocator.init(a);
        defer scratch.deinit();
        for (kinds, 0..) |*k, id| {
            offsets[id] = @intCast(bytes.items.len);
            if (k.* != .text) continue;
            _ = scratch.reset(.retain_capacity);
            const got = try source.tokenBytes(scratch.allocator(), @intCast(id)) orelse "";
            if (got.len == 0) k.* = .none;
            try bytes.appendSlice(a, got);
        }
        offsets[columns] = @intCast(bytes.items.len);
        return .{ .offsets = offsets, .bytes = try bytes.toOwnedSlice(a), .kinds = kinds };
    }

    pub fn deinit(v: *Vocab, a: Allocator) void {
        a.free(v.offsets);
        a.free(v.bytes);
        a.free(v.kinds);
        v.* = undefined;
    }

    /// The logits' width the table covers.
    pub fn width(v: *const Vocab) usize {
        return v.kinds.len;
    }

    pub fn kind(v: *const Vocab, id: u32) Kind {
        return if (id < v.kinds.len) v.kinds[id] else .none;
    }

    pub fn piece(v: *const Vocab, id: u32) []const u8 {
        if (id >= v.kinds.len) return "";
        return v.bytes[v.offsets[id]..v.offsets[id + 1]];
    }

    /// Whether token `id` may come next after the reply that brought matcher `m` (or the one it points to) here.
    pub fn allows(v: *const Vocab, m: anytype, id: u32) bool {
        return switch (v.kind(id)) {
            .text => m.accepts(v.piece(id)),
            .eos => m.complete(),
            .special, .none => false,
        };
    }
};

/// A stand-in tokenizer: token i is `pieces[i]` (null: no token), plus the ids of its added tokens.
const Fake = struct {
    pieces: []const ?[]const u8,
    added: []const struct { id: u32 },

    fn tokenBytes(f: *const Fake, a: Allocator, id: u32) !?[]u8 {
        if (id >= f.pieces.len) return null;
        return try a.dupe(u8, f.pieces[id] orelse return null);
    }
};

test "a vocab table keeps each token's bytes and decides added tokens and eos by id" {
    const a = std.testing.allocator;
    const pieces = [_]?[]const u8{ "{", "}", "\"", "a", ":", " ", "1", "</think>", "", null, "\xe4\xb8", "\xad", "<eos>" };
    const fake = Fake{ .pieces = &pieces, .added = &.{ .{ .id = 7 }, .{ .id = 12 } } };
    var v = try Vocab.build(a, pieces.len + 3, &fake, &.{12});
    defer v.deinit(a);
    try std.testing.expectEqual(@as(usize, pieces.len + 3), v.width());
    try std.testing.expectEqualStrings("\xe4\xb8", v.piece(10));
    try std.testing.expectEqual(Kind.text, v.kind(10));
    try std.testing.expectEqual(Kind.special, v.kind(7));
    try std.testing.expectEqualStrings("", v.piece(7));
    try std.testing.expectEqual(Kind.none, v.kind(8));
    try std.testing.expectEqual(Kind.none, v.kind(9));
    try std.testing.expectEqual(Kind.eos, v.kind(12));
    try std.testing.expectEqual(Kind.none, v.kind(@intCast(pieces.len + 2)));
    try std.testing.expectEqual(Kind.none, v.kind(1 << 20));
    var m: json.Matcher = .{};
    try std.testing.expect(v.allows(m, 0) and !v.allows(m, 5) and !v.allows(m, 12));
    try std.testing.expect(m.advance("{\""));
    // "</think>" is valid string text by its bytes; only its id keeps it out.
    try std.testing.expect(m.accepts("</think>") and !v.allows(m, 7));
    try std.testing.expect(v.allows(m, 10) and !v.allows(m, 11));
    try std.testing.expect(m.advance("\xe4\xb8"));
    try std.testing.expect(v.allows(m, 11) and !v.allows(m, 2));
    try std.testing.expect(m.advance("\xad\":1"));
    try std.testing.expect(!v.allows(m, 12) and v.allows(m, 1));
    try std.testing.expect(m.advance("}"));
    for (0..pieces.len) |id| try std.testing.expectEqual(id == 12, v.allows(m, @intCast(id)));
}
