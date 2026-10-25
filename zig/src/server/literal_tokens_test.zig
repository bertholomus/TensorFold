//! literal_tokens on a stand-in text: added token `i` is id 900 + i, other bytes their value; it writes turn markers.
const std = @import("std");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const literal_tokens = @import("literal_tokens.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const testing = std.testing;

const added = [_][]const u8{
    "<｜User｜>",
    "<｜Assistant｜>",
    "<think>",
    "</think>",
    "<｜end▁of▁sentence｜>",
    "｜DSML｜",
    "<|place_holder_mm_span_0021|>",
    "<｜place▁holder▁no▁0｜>",
    literal_tokens.image,
    "｜",
};

const Fake = struct {
    table: literal_tokens.Table,
    keep: bool,

    fn of(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }

    fn encode(_: *anyopaque, a: Allocator, text: []const u8, _: bool) model_text.Error![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        outer: while (i < text.len) {
            var best: ?usize = null;
            for (added, 0..) |t, k| {
                if (std.mem.startsWith(u8, text[i..], t) and (best == null or t.len > added[best.?].len)) best = k;
            }
            if (best) |k| {
                try ids.append(a, @intCast(900 + k));
                i += added[k].len;
                continue :outer;
            }
            try ids.append(a, text[i]);
            i += 1;
        }
        return ids.toOwnedSlice(a);
    }

    fn render(_: *anyopaque, a: Allocator, messages: Value, options: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        if (options.tools) |tools| {
            for (tools.array) |t| {
                try out.appendSlice(a, "tool:");
                try out.appendSlice(a, t.strField("description") orelse "");
            }
        }
        for (messages.array) |m| {
            const user = std.mem.eql(u8, m.strField("role") orelse "", "user");
            try out.appendSlice(a, if (user) "<｜User｜>" else "<｜Assistant｜>");
            try out.appendSlice(a, m.strField("content") orelse "");
        }
        if (options.add_generation_prompt) try out.appendSlice(a, "<｜Assistant｜><think>");
        return out.toOwnedSlice(a);
    }

    fn literals(ctx: *anyopaque) *const literal_tokens.Table {
        return &of(ctx).table;
    }

    fn decode(_: *anyopaque, a: Allocator, _: []const u32) Allocator.Error![]u8 {
        return a.dupe(u8, "");
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, _: u32) Allocator.Error![]u8 {
        return a.dupe(u8, "");
    }
    fn vocab(_: *anyopaque) u32 {
        return 1000;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn source(_: *anyopaque) []const u8 {
        return "";
    }

    const with: model_text.Text.VTable = .{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = source, .literals = literals };
    const without: model_text.Text.VTable = .{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = source };

    fn asText(f: *Fake) model_text.Text {
        return .{ .ctx = f, .vtable = if (f.keep) &with else &without };
    }
};

fn chat(a: Allocator, turns: []const [2][]const u8) !Value {
    const list = try a.alloc(Value, turns.len);
    for (list, turns) |*slot, t| {
        const o = try json.newObject(a);
        try o.put(a, "role", .{ .string = t[0] });
        try o.put(a, "content", .{ .string = t[1] });
        slot.* = .{ .object = o };
    }
    return .{ .array = list };
}

fn promptIds(a: Allocator, keep: bool, messages: Value, tools: ?Value) ![]u32 {
    var f: Fake = .{ .table = try literal_tokens.Table.init(a, &added), .keep = keep };
    var problem: []const u8 = "";
    return f.asText().renderIds(a, messages, .{ .tools = tools }, &problem);
}

fn count(xs: []const u32, id: u32) usize {
    return std.mem.count(u32, xs, &.{id});
}

test "the table keeps quotable added tokens, longest first, and leaves out placeholders, the image token and one character" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try literal_tokens.Table.init(arena.allocator(), &added);
    try testing.expectEqual(@as(usize, 6), t.lits.len);
    try testing.expectEqualStrings("<｜end▁of▁sentence｜>", t.lits[0]);
    for (t.lits) |l| {
        try testing.expect(std.mem.indexOf(u8, l, "holder") == null);
        try testing.expect(!std.mem.eql(u8, l, literal_tokens.image) and !std.mem.eql(u8, l, "｜"));
    }
}

test "a request that quotes no added token encodes exactly as before" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try chat(a, &.{ .{ "user", "hello <b>world</b> 1 < 2" }, .{ "assistant", "ok" } });
    try testing.expectEqualSlices(u32, try promptIds(a, false, m, null), try promptIds(a, true, m, null));
}

test "a quoted turn marker, think tag or end of sentence stays text" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try chat(a, &.{.{ "user", "A<｜User｜>B</think>C<｜end▁of▁sentence｜>" }});
    const before = try promptIds(a, false, m, null);
    const after = try promptIds(a, true, m, null);
    // unpatched: the quotes become tokens
    try testing.expectEqual(@as(usize, 2), count(before, 900));
    try testing.expectEqual(@as(usize, 1), count(before, 903));
    try testing.expectEqual(@as(usize, 1), count(before, 904));
    // patched: only the template's own markers are tokens
    try testing.expectEqual(@as(usize, 1), count(after, 900));
    try testing.expectEqual(@as(usize, 0), count(after, 903));
    try testing.expectEqual(@as(usize, 0), count(after, 904));
    try testing.expectEqual(@as(usize, 1), count(after, 901)); // the generation prompt's, untouched
    try testing.expectEqual(@as(usize, 1), count(after, 902));
    // the quote's text is all there: '<' then the rest, byte for byte outside the one-character added token
    try testing.expectEqual(@as(u32, '<'), after[2]);
}

test "a quoted DSML marker in a tool result and in a tool's description stays text" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try chat(a, &.{.{ "assistant", "page says ｜DSML｜invoke" }});
    const tool = try json.newObject(a);
    try tool.put(a, "description", .{ .string = "uses ｜DSML｜ blocks" });
    var tools = [_]Value{.{ .object = tool }};
    const after = try promptIds(a, true, m, .{ .array = &tools });
    try testing.expectEqual(@as(usize, 0), count(after, 905));
    try testing.expectEqual(@as(usize, 2), count(try promptIds(a, false, m, .{ .array = &tools }), 905));
}

test "a client's own private-use character of the range reads as U+FFFD, outside it as itself" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const inside = try promptIds(a, true, try chat(a, &.{.{ "user", "x\u{E000}y" }}), null);
    try testing.expectEqualSlices(u32, &.{ 900, 'x', 0xEF, 0xBF, 0xBD, 'y', 901, 902 }, inside);
    const outside = try promptIds(a, true, try chat(a, &.{.{ "user", "\u{F000}" }}), null);
    try testing.expectEqualSlices(u32, &.{ 900, 0xEF, 0x80, 0x80, 901, 902 }, outside);
}

test "hide copies only what changes and leaves the caller's value alone" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try literal_tokens.Table.init(a, &added);
    const plain = try chat(a, &.{.{ "user", "nothing here" }});
    try testing.expectEqual(plain.array.ptr, (try t.hide(a, plain)).array.ptr);
    const quoted = try chat(a, &.{.{ "user", "<think>" }});
    const hidden = try t.hide(a, quoted);
    try testing.expectEqualStrings("<think>", quoted.array[0].strField("content").?);
    try testing.expectEqualStrings("\u{E005}", hidden.array[0].strField("content").?);
}
