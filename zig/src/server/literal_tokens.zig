//! An added token a client quotes in a chat request's strings stays text: hidden before the template, encoded plain.
const std = @import("std");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The first private-use character; literal `i` is ``pua0 + i``.
pub const pua0: u21 = 0xE000;
const pua_room: usize = 0xF8FF - 0xE000 + 1;
pub const image = "<｜deepseek_image｜>";
const replacement = "\u{fffd}";

pub const Table = struct {
    /// The literals, longest first (a longer literal wins over a shorter one it contains).
    lits: []const []const u8 = &.{},
    /// Bytes a literal or a private-use character of the range may start with.
    starts: [256]bool = @splat(false),

    /// The added tokens a client could quote, in `a`: one-character tokens, placeholders and the image token left out.
    pub fn init(a: Allocator, contents: []const []const u8) Allocator.Error!Table {
        var out: std.ArrayList([]const u8) = .empty;
        for (contents) |c| {
            const n = std.unicode.utf8CountCodepoints(c) catch continue;
            if (n <= 1) continue;
            if (std.mem.indexOf(u8, c, "place_holder") != null or std.mem.indexOf(u8, c, "place▁holder") != null) continue;
            if (std.mem.eql(u8, c, image)) continue;
            var seen = false;
            for (out.items) |o| {
                if (std.mem.eql(u8, o, c)) {
                    seen = true;
                    break;
                }
            }
            if (!seen) try out.append(a, c);
        }
        std.mem.sort([]const u8, out.items, {}, longerFirst);
        if (out.items.len > pua_room) out.shrinkRetainingCapacity(pua_room);
        var t: Table = .{ .lits = try out.toOwnedSlice(a) };
        for (t.lits) |l| t.starts[l[0]] = true;
        if (t.lits.len > 0) {
            t.starts[0xEE] = true; // U+E000 .. U+EFFF
            t.starts[0xEF] = true; // U+F000 .. U+F8FF
        }
        return t;
    }

    fn longerFirst(_: void, x: []const u8, y: []const u8) bool {
        if (x.len != y.len) return x.len > y.len;
        return std.mem.lessThan(u8, x, y);
    }

    /// The literal a private-use character at `s[at..]` stands for (its index), else null.
    fn charAt(t: *const Table, s: []const u8, at: usize) ?usize {
        if (at + 3 > s.len) return null;
        const b0 = s[at];
        const b1 = s[at + 1];
        const b2 = s[at + 2];
        if ((b0 != 0xEE and b0 != 0xEF) or (b1 & 0xC0) != 0x80 or (b2 & 0xC0) != 0x80) return null;
        const cp: usize = (@as(usize, b0 & 0x0F) << 12) | (@as(usize, b1 & 0x3F) << 6) | @as(usize, b2 & 0x3F);
        if (cp < pua0 or cp >= pua0 + t.lits.len) return null;
        return cp - pua0;
    }

    /// The longest literal starting at `s[at..]` (its index), else null.
    fn literalAt(t: *const Table, s: []const u8, at: usize) ?usize {
        for (t.lits, 0..) |l, i| if (std.mem.startsWith(u8, s[at..], l)) return i;
        return null;
    }

    /// Whether ``hide`` would change `s`.
    fn touches(t: *const Table, s: []const u8) bool {
        for (s, 0..) |b, i| {
            if (!t.starts[b]) continue;
            if (t.charAt(s, i) != null or t.literalAt(s, i) != null) return true;
        }
        return false;
    }

    /// Whether ``hide`` would change any string in `v` (object keys are left alone, as Python's copy keeps them).
    pub fn needs(t: *const Table, v: Value) bool {
        if (t.lits.len == 0) return false;
        return switch (v) {
            .string => |s| t.touches(s),
            .array => |xs| for (xs) |x| {
                if (t.needs(x)) break true;
            } else false,
            .object => |o| for (o.values()) |x| {
                if (t.needs(x)) break true;
            } else false,
            else => false,
        };
    }

    /// `s` with every literal replaced by its private-use character and the range's own characters by U+FFFD.
    fn hideString(t: *const Table, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var i: usize = 0;
        while (i < s.len) {
            if (t.starts[s[i]]) {
                if (t.charAt(s, i) != null) {
                    try out.appendSlice(a, replacement);
                    i += 3;
                    continue;
                }
                if (t.literalAt(s, i)) |k| {
                    var buf: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(@intCast(pua0 + k), &buf) catch unreachable;
                    try out.appendSlice(a, buf[0..n]);
                    i += t.lits[k].len;
                    continue;
                }
            }
            try out.append(a, s[i]);
            i += 1;
        }
        return out.toOwnedSlice(a);
    }

    /// `v` with every literal in its strings hidden: `v` itself when nothing changes, else a copy of what changes.
    pub fn hide(t: *const Table, a: Allocator, v: Value) Allocator.Error!Value {
        if (!t.needs(v)) return v;
        return switch (v) {
            .string => |s| .{ .string = try t.hideString(a, s) },
            .array => |xs| blk: {
                const out = try a.alloc(Value, xs.len);
                for (out, xs) |*o, x| o.* = try t.hide(a, x);
                break :blk .{ .array = out };
            },
            .object => |o| blk: {
                const n = try json.newObject(a);
                var it = o.iterator();
                while (it.next()) |e| try n.put(a, e.key_ptr.*, try t.hide(a, e.value_ptr.*));
                break :blk .{ .object = n };
            },
            else => v,
        };
    }

    /// A rendered prompt's ids: `text` encoded around its private-use characters, each its literal's plain pieces.
    pub fn encode(t: *const Table, a: Allocator, text: []const u8, tx: model_text.Text) model_text.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var at: usize = 0;
        var i: usize = 0;
        var any = false;
        while (i < text.len) {
            if (text[i] == 0xEE or text[i] == 0xEF) {
                if (t.charAt(text, i)) |k| {
                    any = true;
                    if (i > at) try append(&out, a, tx, text[at..i]);
                    try t.plain(&out, a, tx, t.lits[k]);
                    i += 3;
                    at = i;
                    continue;
                }
            }
            i += 1;
        }
        if (!any) return tx.encode(a, text, false);
        if (at < text.len) try append(&out, a, tx, text[at..]);
        return out.toOwnedSlice(a);
    }

    /// `s` as ordinary pieces: its first character and its rest encoded apart, a literal inside the rest the same.
    fn plain(t: *const Table, out: *std.ArrayList(u32), a: Allocator, tx: model_text.Text, s: []const u8) model_text.Error!void {
        const first = std.unicode.utf8ByteSequenceLength(s[0]) catch 1;
        try append(out, a, tx, s[0..@min(first, s.len)]);
        var rest = s[@min(first, s.len)..];
        while (rest.len > 0) {
            var cut: ?usize = null;
            for (0..rest.len) |j| {
                if (t.starts[rest[j]] and t.literalAt(rest, j) != null) {
                    cut = j;
                    break;
                }
            }
            const j = cut orelse {
                try append(out, a, tx, rest);
                return;
            };
            if (j > 0) try append(out, a, tx, rest[0..j]);
            const lit = t.lits[t.literalAt(rest, j).?];
            try t.plain(out, a, tx, lit);
            rest = rest[j + lit.len ..];
        }
    }
};

fn append(out: *std.ArrayList(u32), a: Allocator, tx: model_text.Text, s: []const u8) model_text.Error!void {
    const ids = try tx.encode(a, s, false);
    try out.appendSlice(a, ids);
}
