//! TENSORFOLD_PLAIN_SPECIALS=1: an added token's text that a message's sender wrote (user, tool and search-result
//! content, and tool-call arguments) is ordinary text in the prompt, so only the chat template's own markers are
//! control tokens; the image markers stay text in every message, as images.zig keeps them (#511). DeepInfer's served
//! prompts keep that rule (doc/inference_deployment_qa.md, 774ce244); the model maker's encoder, like ours with the
//! switch off, tokenizes the template's whole text, a written marker as its token. With TENSORFOLD_REPLY_IDS=1 too,
//! a reply that spells a marker in ordinary tokens keeps it as words, so an assistant turn's content and reasoning
//! are quoted as well: the reply a client sends back reads as the words it was.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const images = @import("images.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The roles whose content someone besides the operator writes: the user, tools and search results.
const senders = [_][]const u8{ "user", "tool", "function", "direct_search_results" };

pub const Shield = struct {
    /// The image markers (images.markerText's order), the added tokens' text, then the sentinel itself: a quoted
    /// marker's number for images.encodeShielded.
    markers: []const []const u8,
    /// markers[0..images] are the image markers, quoted in every message's text.
    images: usize,
    /// The added tokens' text, quoted where a sender wrote it.
    added: usize,
    /// The lane's image placeholder among the markers (a lane-vision engine's, Info.lane_vision), or null. Where a
    /// request's image parts put it (apply's `lane_images`) it stays the template's: normalizeImages refuses a
    /// sender's text that holds it, so each one there is an image part's.
    lane: ?u32,
    index: std.StringHashMapUnmanaged(u32),
    /// The markers' lengths, longest first, each once.
    lengths: []const usize,
    /// The bytes a marker starts with.
    starts: [256]bool,
    /// Config.reply_ids: an assistant turn's content and reasoning_content are quoted too, as a sender's text is.
    replies: bool = false,

    /// The text's added tokens and, with image input, its image markers; `lane_placeholder`: a lane-vision engine's
    /// image placeholder (one of the added tokens). All of it lives in `a`.
    pub fn init(a: Allocator, text: model_text.Text, vision: ?api.Vision, lane_placeholder: ?[]const u8) !Shield {
        var list: std.ArrayList([]const u8) = .empty;
        var index: std.StringHashMapUnmanaged(u32) = .empty;
        if (vision) |v| for (try images.markerText(text, a, v)) |m| try put(a, &list, &index, m);
        const n_images = list.items.len;
        for (text.addedText()) |m| try put(a, &list, &index, m);
        const n_added = list.items.len - n_images;
        const lane: ?u32 = if (lane_placeholder) |p| index.get(p) else null;
        try put(a, &list, &index, images.sentinel);
        var lengths: std.ArrayList(usize) = .empty;
        var starts: [256]bool = @splat(false);
        for (list.items) |m| {
            starts[m[0]] = true;
            if (std.mem.indexOfScalar(usize, lengths.items, m.len) == null) try lengths.append(a, m.len);
        }
        std.mem.sort(usize, lengths.items, {}, std.sort.desc(usize));
        return .{ .markers = list.items, .images = n_images, .added = n_added, .lane = lane, .index = index, .lengths = lengths.items, .starts = starts };
    }

    fn put(a: Allocator, list: *std.ArrayList([]const u8), index: *std.StringHashMapUnmanaged(u32), m: []const u8) !void {
        if (m.len == 0) return;
        const slot = try index.getOrPut(a, m);
        if (slot.found_existing) return; // an image marker that is also an added token keeps its image number
        slot.value_ptr.* = @intCast(list.items.len);
        try list.append(a, m);
    }

    /// `messages` with each marker a sender wrote as its sentinel: every marker in user, tool and search-result
    /// content and in tool calls (with `replies`, in assistant content and reasoning_content too), the image markers
    /// elsewhere; null when no message spells one, so the prompt is rendered and tokenized as with the switch off. A
    /// literal sentinel is quoted too, so it reads back as itself.
    /// `lane_images`: the request's image parts are in its text as the lane's placeholder, which stays as it is.
    pub fn apply(sh: *const Shield, a: Allocator, messages: Value, lane_images: bool) !?Value {
        const keep: ?u32 = if (lane_images) sh.lane else null;
        var quoted: usize = 0;
        const out: Value = if (messages == .array) blk: {
            const list = try a.alloc(Value, messages.array.len);
            for (messages.array, list) |m, *o| o.* = try sh.message(a, m, keep, &quoted);
            break :blk .{ .array = list };
        } else try sh.walk(a, messages, false, keep, &quoted);
        return if (quoted > 0) out else null;
    }

    fn message(sh: *const Shield, a: Allocator, m: Value, keep: ?u32, quoted: *usize) !Value {
        if (m != .object) return sh.walk(a, m, false, keep, quoted);
        const role = m.strField("role") orelse "";
        const sender = for (senders) |r| {
            if (std.mem.eql(u8, role, r)) break true;
        } else false;
        const reply = sh.replies and std.mem.eql(u8, role, "assistant");
        const out = try json.newObject(a);
        var it = m.object.iterator();
        while (it.next()) |kv| {
            const key = kv.key_ptr.*;
            const all = (sender and std.mem.eql(u8, key, "content")) or std.mem.eql(u8, key, "tool_calls") or
                (reply and (std.mem.eql(u8, key, "content") or std.mem.eql(u8, key, "reasoning_content")));
            try out.put(a, key, try sh.walk(a, kv.value_ptr.*, all, keep, quoted));
        }
        return .{ .object = out };
    }

    fn walk(sh: *const Shield, a: Allocator, v: Value, all: bool, keep: ?u32, quoted: *usize) !Value {
        return switch (v) {
            .string => |s| .{ .string = try sh.quote(a, s, all, keep, quoted) },
            .array => |items| blk: {
                const out = try a.alloc(Value, items.len);
                for (items, out) |x, *o| o.* = try sh.walk(a, x, all, keep, quoted);
                break :blk .{ .array = out };
            },
            .object => |o| blk: {
                const out = try json.newObject(a);
                var it = o.iterator();
                while (it.next()) |kv| try out.put(a, kv.key_ptr.*, try sh.walk(a, kv.value_ptr.*, all, keep, quoted));
                break :blk .{ .object = out };
            },
            else => v,
        };
    }

    /// `s` with each marker in reach as images.zig's sentinel and the marker's number, but `keep` as it is; `quoted`
    /// counts the markers.
    fn quote(sh: *const Shield, a: Allocator, s: []const u8, all: bool, keep: ?u32, quoted: *usize) ![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var done: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            const k = sh.at(s, i, all) orelse {
                i += 1;
                continue;
            };
            if (keep != null and k == keep.?) { // an image part's placeholder: the template's own
                i += sh.markers[k].len;
                continue;
            }
            try out.appendSlice(a, s[done..i]);
            try out.print(a, "{s}{u}", .{ images.sentinel, @as(u21, @intCast(0xF0001 + k)) });
            if (k + 1 < sh.markers.len) quoted.* += 1; // the sentinel alone changes nothing
            i += sh.markers[k].len;
            done = i;
        }
        if (done == 0) return s;
        try out.appendSlice(a, s[done..]);
        return out.items;
    }

    /// The marker `s[i..]` starts with, the longest (as the tokenizer splits added tokens), among the image
    /// markers, the sentinel and, when `all`, the added tokens' text; null: none.
    fn at(sh: *const Shield, s: []const u8, i: usize, all: bool) ?u32 {
        if (!sh.starts[s[i]]) return null;
        for (sh.lengths) |n| {
            if (n > s.len - i) continue;
            const k = sh.index.get(s[i..][0..n]) orelse continue;
            if (all or k < sh.images or k + 1 == sh.markers.len) return k;
        }
        return null;
    }
};

/// Added tokens "<|User|>", "<|Assistant|>", "<|Assistant|>!" and "</think>" (ids 1000 up) and the image marker
/// "<|image|>" (9); other bytes are their value plus 100. Renders "[role]" and the content per message, the user's
/// and the assistant's behind the template's own markers, and each tool call's arguments as JSON.
const Fake = struct {
    const added = [_][]const u8{ "<|User|>", "<|Assistant|>", "<|Assistant|>!", "</think>", "<|image|>" };

    fn text() model_text.Text {
        return .{ .ctx = undefined, .vtable = &.{ .encode = encode, .decode = undefined, .token_id = undefined, .token_string = tokenString, .vocab_size = undefined, .eos_ids = undefined, .render = render, .template_source = undefined, .encode_plain = plain, .added_text = addedText } };
    }

    fn id(k: usize) u32 {
        return if (std.mem.eql(u8, added[k], "<|image|>")) 9 else @intCast(1000 + k);
    }

    fn encode(_: *anyopaque, a: Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < s.len) {
            var best: ?usize = null;
            for (added, 0..) |m, k| if (std.mem.startsWith(u8, s[i..], m) and (best == null or m.len > added[best.?].len)) {
                best = k;
            };
            if (best) |k| {
                try out.append(a, id(k));
                i += added[k].len;
            } else {
                try out.append(a, @as(u32, s[i]) + 100);
                i += 1;
            }
        }
        return out.items;
    }

    fn plain(_: *anyopaque, a: Allocator, s: []const u8) model_text.Error![]u32 {
        const out = try a.alloc(u32, s.len);
        for (out, s) |*o, c| o.* = @as(u32, c) + 100;
        return out;
    }

    /// The text `ids` spell: added tokens as their text, other ids as bytes.
    fn decode(a: Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |t| {
            if (t == 9) try out.appendSlice(a, "<|image|>") else if (t >= 1000) try out.appendSlice(a, added[t - 1000]) else try out.append(a, @intCast(t - 100));
        }
        return out.items;
    }

    fn tokenString(_: *anyopaque, a: Allocator, t: u32) Allocator.Error![]u8 {
        return a.dupe(u8, if (t == 9) "<|image|>" else "");
    }

    fn addedText(_: *anyopaque) []const []const u8 {
        return &added;
    }

    fn render(_: *anyopaque, a: Allocator, messages: Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (messages.array) |m| {
            const role = m.strField("role").?;
            try out.print(a, "[{s}]", .{role});
            if (std.mem.eql(u8, role, "user")) try out.appendSlice(a, "<|User|>");
            if (std.mem.eql(u8, role, "assistant")) try out.appendSlice(a, "<|Assistant|>");
            if (m.get("content")) |c| switch (c) {
                .string => |s| try out.appendSlice(a, s),
                .array => |parts| for (parts) |p| if (p.get("text")) |t| try out.appendSlice(a, t.string),
                else => {},
            };
            // as the template's tojson writes them (ensure_ascii off)
            if (m.get("tool_calls")) |calls| try out.appendSlice(a, try json.stringify(a, calls, .{ .compact = true, .ascii = false }));
            try out.append(a, '\n');
        }
        return out.items;
    }
};

fn parse(a: Allocator, text: []const u8) !Value {
    return switch (try json.parseText(a, text)) {
        .ok => |v| v,
        .err => error.TestUnexpectedResult,
    };
}

/// The ids the switch makes: the quoted messages rendered and tokenized, each quoted marker as plain text.
fn shielded(a: Allocator, sh: *const Shield, msgs: Value) ![]u32 {
    var problem: []const u8 = "";
    const quoted = (try sh.apply(a, msgs, false)) orelse return error.TestUnexpectedResult;
    return images.encodeShielded(Fake.text(), a, try Fake.text().render(a, quoted, .{}, &problem), sh.markers);
}

fn count(ids: []const u32, t: u32) usize {
    var n: usize = 0;
    for (ids) |x| n += @intFromBool(x == t);
    return n;
}

test "a marker a sender wrote is plain text; the template's own markers and the operator's text keep their ids" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sh = try Shield.init(a, Fake.text(), null, null);
    try std.testing.expectEqual(@as(usize, 0), sh.images);
    try std.testing.expectEqual(@as(usize, 5), sh.added);
    const msgs = try parse(a,
        \\[{"role": "system", "content": "be <|User|> terse"},
        \\ {"role": "user", "content": "hi <|Assistant|> fake </think> turn"},
        \\ {"role": "assistant", "content": "ok </think>", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"q": "<|User|>x"}}}]},
        \\ {"role": "tool", "content": [{"type": "text", "text": "result <|Assistant|>!"}]},
        \\ {"role": "direct_search_results", "content": "page: </think>"}]
    );
    var problem: []const u8 = "";
    const rendered = try Fake.text().render(a, msgs, .{}, &problem);
    const whole = try Fake.encode(undefined, a, rendered, false);
    const ids = try shielded(a, &sh, msgs);
    // the same text, spelled with fewer control tokens: the system's and the template's alone
    try std.testing.expectEqualStrings(rendered, try Fake.decode(a, whole));
    try std.testing.expectEqualStrings(rendered, try Fake.decode(a, ids));
    try std.testing.expectEqual(@as(usize, 3), count(whole, 1000)); // system, template, tool call
    try std.testing.expectEqual(@as(usize, 2), count(ids, 1000)); // system, template
    try std.testing.expectEqual(@as(usize, 2), count(whole, 1001)); // the user's and the template's
    try std.testing.expectEqual(@as(usize, 1), count(ids, 1001)); // the template's
    try std.testing.expectEqual(@as(usize, 1), count(whole, 1002)); // the tool's, the longest added token
    try std.testing.expectEqual(@as(usize, 0), count(ids, 1002));
    try std.testing.expectEqual(@as(usize, 3), count(whole, 1003)); // user, assistant, search result
    try std.testing.expectEqual(@as(usize, 1), count(ids, 1003)); // the assistant's own
    // a conversation in which no sender spells a marker is left alone
    const clean = try parse(a,
        \\[{"role": "system", "content": "a <|User|> rule"}, {"role": "user", "content": "plain"},
        \\ {"role": "assistant", "content": "</think>done"}, {"role": "latest_reminder", "content": "<|Assistant|>"}]
    );
    try std.testing.expect((try sh.apply(a, clean, false)) == null);
}

test "with reply ids an assistant turn's content and reasoning are quoted too; its role marker stays the template's" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var sh = try Shield.init(a, Fake.text(), null, null);
    // a reply that wrote the tag as words, sent back as the assistant turn
    const msgs = try parse(a,
        \\[{"role": "user", "content": "say </think> world"},
        \\ {"role": "assistant", "content": "</think> world", "reasoning_content": "they want </think>"},
        \\ {"role": "user", "content": "again"}]
    );
    var problem: []const u8 = "";
    const rendered = try Fake.text().render(a, msgs, .{}, &problem);
    // without reply ids the assistant's text is the template's: its </think> is the token
    try std.testing.expectEqual(@as(usize, 1), count(try shielded(a, &sh, msgs), 1003));
    sh.replies = true;
    const ids = try shielded(a, &sh, msgs);
    try std.testing.expectEqualStrings(rendered, try Fake.decode(a, ids));
    try std.testing.expectEqual(@as(usize, 0), count(ids, 1003)); // the user's and the assistant's, words
    try std.testing.expectEqual(@as(usize, 1), count(ids, 1001)); // the template's <|Assistant|>
    try std.testing.expectEqual(@as(usize, 2), count(ids, 1000)); // the template's <|User|>, twice
    const quoted = (try sh.apply(a, msgs, false)).?;
    try std.testing.expect(std.mem.indexOf(u8, quoted.array[1].get("reasoning_content").?.string, images.sentinel) != null);
    // the operator's text stays the template's; an assistant turn alone is quoted only with reply ids
    const alone = try parse(a,
        \\[{"role": "system", "content": "a <|User|> rule"}, {"role": "assistant", "content": "</think>done"}]
    );
    const kept = (try sh.apply(a, alone, false)).?;
    try std.testing.expectEqualStrings("a <|User|> rule", kept.array[0].get("content").?.string);
    sh.replies = false;
    try std.testing.expect((try sh.apply(a, alone, false)) == null);
}

test "a literal sentinel reads back as itself once a marker is quoted, and alone changes nothing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sh = try Shield.init(a, Fake.text(), null, null);
    const lone = try parse(a, "[{\"role\": \"user\", \"content\": \"x\u{F0000}\u{F0002}y\"}]");
    try std.testing.expect((try sh.apply(a, lone, false)) == null);
    const msgs = try parse(a, "[{\"role\": \"system\", \"content\": \"s\u{F0000}\u{F0001}\"}, {\"role\": \"user\", \"content\": \"x\u{F0000}\u{F0002}y </think>\"}]");
    var problem: []const u8 = "";
    const rendered = try Fake.text().render(a, msgs, .{}, &problem);
    const ids = try shielded(a, &sh, msgs);
    try std.testing.expectEqualStrings(rendered, try Fake.decode(a, ids));
    try std.testing.expectEqual(@as(usize, 0), count(ids, 1003));
    try std.testing.expectEqual(@as(usize, 1), count(ids, 1000)); // the template's
}

test "with image input the image markers keep their numbers and stay text in every message" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const v: api.Vision = .{ .ctx = &dummy, .prepare = undefined, .image_token = 9, .markers = &.{9}, .image_tokens = 64, .max_images = 4 };
    const sh = try Shield.init(a, Fake.text(), v, null);
    try std.testing.expectEqual(@as(usize, 1), sh.images);
    try std.testing.expectEqualStrings("<|image|>", sh.markers[0]);
    try std.testing.expectEqual(@as(usize, 4), sh.added); // "<|image|>" is listed once, as the image marker
    try std.testing.expectEqual(sh.markers.len, 1 + 4 + 1);
    try std.testing.expectEqualStrings(images.sentinel, sh.markers[sh.markers.len - 1]);
    const msgs = try parse(a,
        \\[{"role": "system", "content": "show <|image|> and <|User|>"}, {"role": "user", "content": "what is <|image|>?"}]
    );
    const ids = try shielded(a, &sh, msgs);
    try std.testing.expectEqual(@as(usize, 0), count(ids, 9)); // both spelled out, as images.shield spells them
    try std.testing.expectEqual(@as(usize, 2), count(ids, 1000)); // the system's and the template's
    // the longest added token wins where two start: "<|Assistant|>!" whole, never "<|Assistant|>" and "!"
    const longest = try parse(a, "[{\"role\": \"user\", \"content\": \"<|Assistant|>!\"}]");
    const quoted = (try sh.apply(a, longest, false)).?;
    const s = quoted.array[0].get("content").?.string;
    try std.testing.expectEqual(@as(usize, 8), s.len);
    try std.testing.expectEqual(@as(u21, 0xF0001 + 1 + 2), try std.unicode.utf8Decode(s[4..8]));
}

test "a lane's image placeholder stays the template's where the request's image parts put it, and text elsewhere" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sh = try Shield.init(a, Fake.text(), null, "<|image|>");
    try std.testing.expectEqualStrings("<|image|>", sh.markers[sh.lane.?]);
    // an image part as normalizeImages writes it, and a marker the user wrote beside it
    const msgs = try parse(a,
        \\[{"role": "user", "content": "<|image|>\n\nwhat is </think>?"}]
    );
    var problem: []const u8 = "";
    const kept = (try sh.apply(a, msgs, true)).?;
    const ids = try images.encodeShielded(Fake.text(), a, try Fake.text().render(a, kept, .{}, &problem), sh.markers);
    try std.testing.expectEqual(@as(usize, 1), count(ids, 9)); // the image part's placeholder
    try std.testing.expectEqual(@as(usize, 0), count(ids, 1003)); // the user's </think>, plain text
    // in a request without image parts the same text is a sender's, quoted like any other marker
    try std.testing.expectEqual(@as(usize, 0), count(try shielded(a, &sh, msgs), 9));
    // an image part alone quotes nothing: the prompt is rendered and tokenized as with the switch off
    const alone = try parse(a, "[{\"role\": \"user\", \"content\": \"<|image|>\"}]");
    try std.testing.expect((try sh.apply(a, alone, true)) == null);
}
