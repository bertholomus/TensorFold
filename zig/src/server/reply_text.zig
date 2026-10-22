//! Reply text as it streams: think blocks, tool markup held back, Harmony channels, stop strings.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Markers = struct { open: []const u8, close: []const u8 };
/// Qwen's prompt opens the think block; Gemma 4's reply opens its thought channel.
pub const think_markers: Markers = .{ .open = "", .close = "</think>" };
/// The opener a reply writes itself when its template leaves the think block to it (Kolibri 1's does).
const think_open = "<think>";
pub const channel_markers: Markers = .{ .open = "<|channel>thought", .close = "<channel|>" };
/// (opener, closer) of a tool call's markup: Qwen's, Gemma 4's, DeepSeek-V4's DSML block, DeepSeek-V4.1's.
pub const calls = [_][2][]const u8{
    .{ "<tool_call>", "</tool_call>" },
    .{ "<|tool_call>", "<tool_call|>" },
    .{ "<\u{ff5c}DSML\u{ff5c}tool_calls>", "</\u{ff5c}DSML\u{ff5c}tool_calls>" },
    .{ "<\u{ff5c}DSML\u{ff5c} calls>", "</\u{ff5c}DSML\u{ff5c} calls>" },
};

/// ``text.find(tag, from)`` led by a vector scan for the tag's first byte: markup is rare in prose, so it is fast.
pub fn findTag(text: []const u8, from: usize, tag: []const u8) ?usize {
    if (tag.len == 0) return if (from <= text.len) from else null;
    var at = from;
    while (at < text.len) {
        const i = std.mem.indexOfScalarPos(u8, text, at, tag[0]) orelse return null;
        if (std.mem.startsWith(u8, text[i..], tag)) return i;
        at = i + 1;
    }
    return null;
}

fn find(text: []const u8, tag: []const u8) ?usize {
    return findTag(text, 0, tag);
}

/// Where a reply's marker tokens decoded (TENSORFOLD_REPLY_IDS): a marker token's text in a tag is markup only at the
/// offset its own token decoded to, so the same text spelled in ordinary tokens (a reply quoting a think end or a DSML
/// call) stays words. Null marks read every tag as its text does.
pub const Marks = struct {
    base: [*]const u8, // the decoded reply the offsets count in
    len: usize,
    at: []const Mark, // in reply order
    pieces: []const Piece, // the marker tokens (Server.reply_pieces)

    pub const Mark = struct { offset: usize, id: u32 };
    pub const Piece = struct { text: []const u8, id: u32 };

    /// Whether `tag` at `text[i..]` is markup: each marker token's text in it lies where that token decoded. Text
    /// outside the decoded reply (a copy) reads as its text does.
    pub fn real(m: *const Marks, text: []const u8, i: usize, tag: []const u8) bool {
        const p = @intFromPtr(text.ptr);
        const b = @intFromPtr(m.base);
        if (p < b or p > b + m.len) return true;
        const at = p - b + i;
        for (m.pieces) |piece| {
            var k: usize = 0;
            while (std.mem.indexOfPos(u8, tag, k, piece.text)) |j| : (k = j + piece.text.len) {
                if (!m.has(at + j, piece.id)) return false;
            }
        }
        return true;
    }

    fn has(m: *const Marks, offset: usize, id: u32) bool {
        for (m.at) |x| {
            if (x.offset == offset and x.id == id) return true;
            if (x.offset > offset) return false;
        }
        return false;
    }
};

/// ``findTag`` of a tag where it is markup (``Marks.real``); null marks: ``findTag``.
pub fn findMarkup(text: []const u8, from: usize, tag: []const u8, marks: ?*const Marks) ?usize {
    const m = marks orelse return findTag(text, from, tag);
    var at = from;
    while (findTag(text, at, tag)) |i| {
        if (m.real(text, i, tag)) return i;
        at = i + 1;
    }
    return null;
}

fn startsMarkup(text: []const u8, tag: []const u8, marks: ?*const Marks) bool {
    return std.mem.startsWith(u8, text, tag) and (marks == null or marks.?.real(text, 0, tag));
}

/// The first DeepSeek-V4.1 calls opener at or after `from` that opens a block, or null: markup, and not inside a
/// DeepSeek-V4 block that opened first (its argument's text). Such a block is read from its own tags: the parser, the
/// streamer and the prose hider find it first, and the other forms only in the text around it.
pub fn open41(text: []const u8, from: usize, marks: ?*const Marks) ?usize {
    const v41 = calls[3];
    const v4 = calls[2];
    var at = from;
    while (findMarkup(text, at, v41[0], marks)) |s| {
        const outer = findMarkup(text, at, v4[0], marks) orelse return s;
        if (outer > s) return s;
        at = (findTag(text, outer + v4[0].len, v4[1]) orelse return null) + v4[1].len; // unclosed, it runs to the end
    }
    return null;
}

/// Appends where `tokens[from..]`'s marker tokens decoded: each one's text ends the decode of the tokens through it
/// (`text`: the model's tokenizer; each decode is `scratch`'s and freed at once).
pub fn markTokens(a: Allocator, scratch: Allocator, text: anytype, pieces: []const Marks.Piece, tokens: []const u32, from: usize, out: *std.ArrayList(Marks.Mark)) Allocator.Error!void {
    for (tokens[from..], from..) |t, j| {
        const piece = for (pieces) |p| {
            if (p.id == t) break p;
        } else continue;
        const through = try text.decode(scratch, tokens[0 .. j + 1]);
        defer scratch.free(through);
        if (std.mem.endsWith(u8, through, piece.text)) try out.append(a, .{ .offset = through.len - piece.text.len, .id = t });
    }
}

pub fn isChannel(m: Markers) bool {
    return std.mem.eql(u8, m.close, channel_markers.close);
}

/// How many bytes at the end of ``text`` could begin ``tag``.
pub fn partialTag(text: []const u8, tag: []const u8) usize {
    if (tag.len == 0) return 0;
    var k = @min(tag.len - 1, text.len);
    while (k > 0) : (k -= 1) if (std.mem.endsWith(u8, text, tag[0..k])) return k;
    return 0;
}

fn trimNewlines(s: []const u8) []const u8 {
    return std.mem.trimStart(u8, s, "\n");
}

pub const Split = struct { reasoning: []const u8, answer: []const u8 };

/// (reasoning, answer) of a thinking reply; while it streams, a tail that could begin a marker is held back.
pub fn splitThinking(a: Allocator, full: []const u8, finished: bool, m: Markers) Allocator.Error!Split {
    return splitThinkingMarked(a, full, finished, m, null);
}

/// ``splitThinking`` with the reply's marks: a marker spelled in ordinary tokens neither opens nor closes the block.
pub fn splitThinkingMarked(a: Allocator, full: []const u8, finished: bool, m: Markers, marks: ?*const Marks) Allocator.Error!Split {
    var text = full;
    if (m.open.len > 0 and !startsMarkup(text, m.open, marks)) {
        if (findMarkup(text, 0, m.open, marks)) |start| if (start > 0) {
            var prefix = text[0..start];
            prefix = prefix[0 .. prefix.len - partialTag(prefix, m.open)]; // a stray partial opener
            const inner = try splitThinkingMarked(a, text[start..], finished, m, marks);
            return .{ .reasoning = inner.reasoning, .answer = try std.mem.concat(a, u8, &.{ prefix, inner.answer }) };
        };
        if (!finished) {
            var held = @max(partialTag(text, m.open), partialTag(text, m.close));
            for (calls) |c| held = @max(held, partialTag(text, c[0]));
            return .{ .reasoning = "", .answer = text[0 .. text.len - held] };
        }
        return .{ .reasoning = "", .answer = text };
    }
    if (m.open.len > 0) {
        text = trimNewlines(text[m.open.len..]);
    } else if (startsMarkup(text, think_open, marks)) {
        text = trimNewlines(text[think_open.len..]);
    } else if (!finished and std.mem.startsWith(u8, think_open, text)) {
        return .{ .reasoning = "", .answer = "" }; // held while the reply may still be writing that opener
    }
    if (findMarkup(text, 0, m.close, marks)) |end| return .{ .reasoning = text[0..end], .answer = trimNewlines(text[end + m.close.len ..]) };
    var call: ?usize = null;
    for (calls) |c| if (findMarkup(text, 0, c[0], marks)) |at| {
        if (call == null or at < call.?) call = at;
    };
    if (call) |at| {
        // a call written before the block closes: held while the block may still close, the answer if the reply ends
        if (!finished) return .{ .reasoning = text[0..at], .answer = "" };
        for (calls) |c| if (startsMarkup(text[at..], c[0], marks) and find(text[at..], c[1]) != null)
            return .{ .reasoning = text[0..at], .answer = text[at..] };
        return .{ .reasoning = text, .answer = "" };
    }
    var held: usize = 0;
    if (!finished) {
        held = partialTag(text, m.close);
        for (calls) |c| held = @max(held, partialTag(text, c[0]));
    }
    return .{ .reasoning = text[0 .. text.len - held], .answer = "" };
}

/// Hides tool-call blocks and partial opening tags while streaming, so calls arrive as deltas instead.
pub fn hideToolCalls(a: Allocator, text: []const u8, finished: bool) Allocator.Error![]const u8 {
    return hideToolCallsMarked(a, text, finished, null);
}

/// ``hideToolCalls`` with the reply's marks: a block whose opener is spelled in ordinary tokens is prose.
pub fn hideToolCallsMarked(a: Allocator, text: []const u8, finished: bool, marks: ?*const Marks) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    return hideIntoMarked(a, &out, text, finished, marks);
}

/// ``hideToolCalls`` into ``out`` (cleared first); text without a call block comes back as a slice of ``text``.
pub fn hideInto(a: Allocator, out: *std.ArrayList(u8), text: []const u8, finished: bool) Allocator.Error![]const u8 {
    return hideIntoMarked(a, out, text, finished, null);
}

/// ``hideInto`` with the reply's marks (``hideToolCallsMarked``). DeepSeek-V4.1's blocks are found first (``open41``),
/// and the other forms only in the text around them, as the parser reads a reply: a <tool_call> in the prose before a
/// block, or in its argument, hides nothing past the block's bounds.
pub fn hideIntoMarked(a: Allocator, out: *std.ArrayList(u8), text: []const u8, finished: bool, marks: ?*const Marks) Allocator.Error![]const u8 {
    out.clearRetainingCapacity();
    var gap: usize = 0;
    while (open41(text, gap, marks)) |s| {
        _ = try hideForms(a, out, text, gap, s, true, marks);
        gap = (findTag(text, s + calls[3][0].len, calls[3][1]) orelse return out.items) + calls[3][1].len;
    }
    return hideForms(a, out, text, gap, text.len, finished, marks);
}

/// ``hideIntoMarked``'s other forms in ``text[from..to]``, appended to ``out``. A ``to`` short of the end is a
/// DeepSeek-V4.1 block's start: an opener whose closer would lie past it is prose. ``text`` without a block comes back
/// as a slice of itself.
fn hideForms(a: Allocator, out: *std.ArrayList(u8), whole: []const u8, from: usize, to: usize, finished: bool, marks: ?*const Marks) Allocator.Error![]const u8 {
    const text = whole[0..to];
    var pos = from;
    var next: [calls.len]?usize = undefined; // each opener at or after `pos`, searched again only once `pos` passes it
    for (calls, &next) |c, *n| n.* = findMarkup(text, from, c[0], marks);
    while (true) {
        var start: ?usize = null;
        var which: usize = 0;
        for (calls, &next, 0..) |c, *n, i| {
            if (n.* != null and n.*.? < pos) n.* = findMarkup(text, pos, c[0], marks);
            if (n.*) |at| if (start == null or at < start.?) {
                start = at;
                which = i;
            };
        }
        if (start == null) {
            const tail = text[pos..];
            var held: usize = 0;
            if (!finished) for (calls) |c| {
                held = @max(held, partialTag(tail, c[0]));
            };
            if (pos == 0 and to == whole.len) return text[0 .. text.len - held];
            try out.appendSlice(a, tail[0 .. tail.len - held]);
            return out.items;
        }
        try out.appendSlice(a, text[pos..start.?]);
        const opener = calls[which][0];
        const end = findTag(text, start.? + opener.len, calls[which][1]) orelse {
            if (to < whole.len) try out.appendSlice(a, text[start.?..]);
            return out.items;
        };
        pos = end + calls[which][1].len;
    }
}

/// With tools, a stream holds trailing whitespace back until more text comes: a reply with a call does not show it.
pub fn heldBack(text: []const u8, tools: bool) []const u8 {
    return if (tools) text[0..pyRstrip(text).len] else text; // a prefix, so the stream's pointer checks still hold
}

/// ``hideToolCalls`` of a finished reply; once a call is hidden, the prose's trailing whitespace stays unsent too.
pub fn finishedProse(a: Allocator, text: []const u8) Allocator.Error![]const u8 {
    return finishedProseMarked(a, text, null);
}

/// ``finishedProse`` with the reply's marks (``hideToolCallsMarked``).
pub fn finishedProseMarked(a: Allocator, text: []const u8, marks: ?*const Marks) Allocator.Error![]const u8 {
    const prose = try hideToolCallsMarked(a, text, true, marks);
    return if (prose.len != text.len) prose[0..pyRstrip(prose).len] else prose;
}

const harmony_final = "<|channel|>final<|message|>";
const harmony_analysis = "<|channel|>analysis<|message|>";
const harmony_ends = [_][]const u8{ "<|return|>", "<|end|>", "<|call|>", "<|start|>" };

fn cutAt(s: []const u8, marker: []const u8) []const u8 {
    return if (find(s, marker)) |i| s[0..i] else s;
}

pub const Harmony = struct { content: []const u8, reasoning: ?[]const u8 };

/// Harmony's final and analysis channels; other text passes unchanged.
pub fn parseHarmony(text: []const u8) Harmony {
    if (find(text, "<|channel|>") == null) return .{ .content = text, .reasoning = null };
    var reasoning: ?[]const u8 = null;
    if (find(text, harmony_analysis)) |i| {
        var r = text[i + harmony_analysis.len ..];
        r = cutAt(r, harmony_final);
        for (harmony_ends) |e| r = cutAt(r, e);
        reasoning = r;
    }
    const f = find(text, harmony_final) orelse return .{ .content = "", .reasoning = reasoning };
    var content = text[f + harmony_final.len ..];
    for (harmony_ends) |e| content = cutAt(content, e);
    return .{ .content = content, .reasoning = reasoning };
}

/// The part of partly decoded output that should stream.
pub fn streamingVisible(text: []const u8) []const u8 {
    if (find(text, "<|channel|>") == null) return text;
    return parseHarmony(text).content;
}

pub fn stripTrailing(tokens: []const u32, stops: []const u32) []const u32 {
    var end = tokens.len;
    while (end > 0 and std.mem.indexOfScalar(u32, stops, tokens[end - 1]) != null) end -= 1;
    return tokens[0..end];
}

/// A thinking reply's reasoning tokens: through its close, else all of them.
pub fn reasoningCount(tokens: []const u32, think_end: ?u32) usize {
    const end = think_end orelse return 0;
    return if (std.mem.indexOfScalar(u32, tokens, end)) |i| i + 1 else tokens.len;
}

/// Python's ``s[n:]`` with ``n`` counted in characters.
pub fn afterChars(s: []const u8, n: usize) []const u8 {
    var seen: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] & 0xc0 == 0x80) continue;
        if (seen == n) return s[i..];
        seen += 1;
    }
    return "";
}

pub fn charCount(s: []const u8) usize {
    var n: usize = 0;
    for (s) |b| n += @intFromBool(b & 0xc0 != 0x80);
    return n;
}

/// Python's ``str.strip()`` for the whitespace replies hold.
pub fn pyStrip(s: []const u8) []const u8 {
    var start: usize = 0;
    var end = s.len;
    while (start < end) {
        const n = spaceAt(s, start);
        if (n == 0) break;
        start += n;
    }
    while (end > start) {
        var back: usize = 1;
        while (back < 4 and end - back > start and s[end - back] & 0xc0 == 0x80) back += 1;
        if (spaceAt(s[0..end], end - back) != back) break;
        end -= back;
    }
    return s[start..end];
}

/// ``str.lstrip()``: the leading whitespace off.
pub fn pyLstrip(s: []const u8) []const u8 {
    return s[@intFromPtr(pyStrip(s).ptr) - @intFromPtr(s.ptr) ..];
}

/// ``str.rstrip()``: the trailing whitespace off.
pub fn pyRstrip(s: []const u8) []const u8 {
    const stripped = pyStrip(s);
    if (stripped.len == 0) return "";
    return s[0 .. @intFromPtr(stripped.ptr) - @intFromPtr(s.ptr) + stripped.len];
}

/// Python's ``\s`` on one byte (ASCII whitespace and the separators below space).
pub fn isSpaceByte(ch: u8) bool {
    return ch == ' ' or (ch >= 0x09 and ch <= 0x0d) or (ch >= 0x1c and ch <= 0x1f);
}

fn spaceAt(s: []const u8, i: usize) usize {
    const c = s[i];
    if (c == ' ' or (c >= 0x09 and c <= 0x0d) or (c >= 0x1c and c <= 0x1f)) return 1;
    if (c < 0x80) return 0;
    const n = std.unicode.utf8ByteSequenceLength(c) catch return 0;
    if (i + n > s.len) return 0;
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return 0;
    return switch (cp) {
        0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => n,
        else => 0,
    };
}

/// Request stop strings: the visible text ends before the first, and a possible start is held while streaming.
pub const Stops = struct {
    strings: []const []const u8,

    pub fn visible(s: Stops, text: []const u8, partial: bool) []const u8 {
        var first: ?usize = null;
        for (s.strings) |stop| if (std.mem.indexOf(u8, text, stop)) |at| {
            if (first == null or at < first.?) first = at;
        };
        if (first) |at| return text[0..at];
        if (!partial) return text;
        var held: usize = 0;
        for (s.strings) |stop| {
            var size: usize = 1;
            while (size < stop.len and size <= text.len) : (size += 1) {
                if (std.mem.endsWith(u8, text, stop[0..size])) held = @max(held, size);
            }
        }
        return text[0 .. text.len - held];
    }

    /// The first stop the text holds, ties in request order (``matched_stop``).
    pub fn matched(s: Stops, text: []const u8) ?[]const u8 {
        var best: ?usize = null;
        var which: []const u8 = "";
        for (s.strings) |stop| if (std.mem.indexOf(u8, text, stop)) |at| {
            if (best == null or at < best.?) {
                best = at;
                which = stop;
            }
        };
        return if (best != null) which else null;
    }

    /// Tokens the engine's stop check decodes: the longest stop's length plus 8.
    pub fn tail(s: Stops) usize {
        var longest: usize = 0;
        for (s.strings) |stop| longest = @max(longest, charCount(stop));
        return longest + 8;
    }
};

/// The text a stream has sent: Python's ``streamed = visible``, grown by its delta while it only grows.
pub const Shown = struct {
    text: std.ArrayList(u8) = .empty,
    chars: usize = 0,
    at: ?[*]const u8 = null, // where the last shown text lay; the decode buffer only grows, so that prefix holds
    extends: bool = false,

    /// Python's ``now[len(shown):]``; ``stable``: ``now`` lies in the append-only decode buffer.
    pub fn after(s: *Shown, now: []const u8, stable: bool) []const u8 {
        const sent = s.text.items;
        s.extends = (stable and s.at == now.ptr and now.len >= sent.len) or std.mem.startsWith(u8, now, sent);
        if (!s.extends) return afterChars(now, s.chars);
        if (stable) s.at = now.ptr; // the buffer moved but kept its bytes: the next check is free again
        return now[sent.len..];
    }

    pub fn set(s: *Shown, a: Allocator, now: []const u8, delta: []const u8) Allocator.Error!void {
        if (s.extends) {
            try s.text.appendSlice(a, delta);
            s.chars += charCount(delta);
        } else {
            s.text = .empty;
            try s.text.appendSlice(a, now);
            s.chars = charCount(now);
        }
        s.at = now.ptr;
    }
};

/// Decodes only new tokens with the previous chunk as context, so characters split across tokens stay whole.
pub const Incremental = struct {
    tokens: std.ArrayList(u32) = .empty,
    text: std.ArrayList(u8) = .empty,
    prefix: usize = 0,
    read: usize = 0,

    pub fn extend(inc: *Incremental, a: Allocator, ctx: anytype, decode: anytype, more: []const u32) ![]const u8 {
        try inc.tokens.appendSlice(a, more);
        const before = try decode(ctx, a, inc.tokens.items[inc.prefix..inc.read]);
        const after = try decode(ctx, a, inc.tokens.items[inc.prefix..]);
        const nb = charCount(before);
        if (charCount(after) > nb and !std.mem.endsWith(u8, after, "\u{fffd}")) {
            try inc.text.appendSlice(a, afterChars(after, nb));
            inc.prefix = inc.read;
            inc.read = inc.tokens.items.len;
        }
        return inc.text.items;
    }

    /// A character is still split across tokens, so streaming waits for the rest.
    pub fn pending(inc: *const Incremental) bool {
        return inc.tokens.items.len > 0 and inc.read < inc.tokens.items.len;
    }
};

test "splits and holds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = try splitThinking(a, "plan</think>\n\nHi", true, think_markers);
    try std.testing.expectEqualStrings("plan", s.reasoning);
    try std.testing.expectEqualStrings("Hi", s.answer);
    const held = try splitThinking(a, "plan</thi", false, think_markers);
    try std.testing.expectEqualStrings("plan", held.reasoning);
    try std.testing.expectEqualStrings("a b", try hideToolCalls(a, "a <tool_call>x</tool_call>b<tool", false));
    const stops: Stops = .{ .strings = &.{"END"} };
    try std.testing.expectEqualStrings("ab", stops.visible("abEN", true));
    try std.testing.expectEqualStrings("x", pyStrip(" \u{3000}x\u{a0}\n"));
    try std.testing.expectEqualStrings("lo", afterChars("h\u{e9}lo", 2));
}

test "a reply that writes its own think opener keeps the tag out of its reasoning" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = "<think>\nPlan.\n</think>\n\nDone.";
    var sent: []const u8 = "";
    for (1..reply.len + 1) |n| {
        const s = try splitThinking(a, reply[0..n], false, think_markers);
        try std.testing.expect(std.mem.indexOfScalar(u8, s.reasoning, '<') == null and std.mem.startsWith(u8, s.reasoning, sent));
        sent = s.reasoning;
    }
    for ([_][]const u8{ reply, "Plan.\n</think>\n\nDone." }) |text| {
        const s = try splitThinking(a, text, true, think_markers);
        try std.testing.expectEqualStrings("Plan.\n", s.reasoning);
        try std.testing.expectEqualStrings("Done.", s.answer);
    }
}

test "with marks, a think end or a DSML opener spelled in ordinary tokens stays words" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pieces = [_]Marks.Piece{ .{ .text = "</think>", .id = 7 }, .{ .text = "\u{ff5c}DSML\u{ff5c}", .id = 9 } };
    // the reasoning quotes the tag in ordinary tokens; the block ends where the token decoded
    const reply = "a </think> b</think>\n\nanswer";
    const end = std.mem.lastIndexOf(u8, reply, "</think>").?;
    const marks: Marks = .{ .base = reply.ptr, .len = reply.len, .at = &.{.{ .offset = end, .id = 7 }}, .pieces = &pieces };
    const s = try splitThinkingMarked(a, reply, true, think_markers, &marks);
    try std.testing.expectEqualStrings("a </think> b", s.reasoning);
    try std.testing.expectEqualStrings("answer", s.answer);
    try std.testing.expectEqualStrings("a ", (try splitThinking(a, reply, true, think_markers)).reasoning); // by text
    // while it streams, the quoted tag is reasoning too, and nothing is answer before the token
    const part = reply[0 .. end - 1];
    const held = try splitThinkingMarked(a, part, false, think_markers, &marks);
    try std.testing.expectEqualStrings("a </think> ", held.reasoning);
    try std.testing.expectEqualStrings("", held.answer);
    // a DSML block spelled in ordinary tokens is prose; at its token's place it is a call, hidden
    const prose = "Use <\u{ff5c}DSML\u{ff5c} calls>x</\u{ff5c}DSML\u{ff5c} calls> like this.";
    const none: Marks = .{ .base = prose.ptr, .len = prose.len, .at = &.{}, .pieces = &pieces };
    try std.testing.expectEqualStrings(prose, try hideToolCallsMarked(a, prose, true, &none));
    try std.testing.expectEqualStrings(prose, try finishedProseMarked(a, prose, &none));
    try std.testing.expectEqualStrings("Use  like this.", try hideToolCalls(a, prose, true));
    const at = std.mem.indexOf(u8, prose, "\u{ff5c}DSML\u{ff5c}").?;
    const real: Marks = .{ .base = prose.ptr, .len = prose.len, .at = &.{.{ .offset = at, .id = 9 }}, .pieces = &pieces };
    try std.testing.expectEqualStrings("Use  like this.", try hideToolCallsMarked(a, prose, true, &real));
    // a copy outside the decoded reply reads as its text does
    const copy = try a.dupe(u8, prose);
    try std.testing.expectEqualStrings("Use  like this.", try hideToolCallsMarked(a, copy, true, &none));
}

test "a <tool_call> in the prose before a DeepSeek-V4.1 block, or in its argument, hides nothing past the block's bounds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const D = "\u{ff5c}DSML\u{ff5c}";
    const quoted = "a </tool_call> and <tool_call>{\"name\": \"weather\", \"arguments\": {}}</tool_call>";
    const block = "<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"write\">\n<" ++ D ++ " parameter name=\"content\" string=\"true\">" ++ quoted ++ "</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const reply = "I will not use <tool_call> here.\n\n" ++ block ++ "\nDone. <tool_call>{\"name\": \"weather\", \"arguments\": {}}</tool_call>";
    // the prose around the block shows, as the parser reads it; the block and the call after it are hidden
    try std.testing.expectEqualStrings("I will not use <tool_call> here.\n\n\nDone. ", try hideToolCalls(a, reply, true));
    try std.testing.expectEqualStrings("I will not use <tool_call> here.\n\n\nDone.", try finishedProse(a, reply));
    // while it streams, the prose shown only grows, to the whole reply's
    var shown: []const u8 = "";
    for (0..reply.len + 1) |n| {
        const now = try hideToolCalls(a, reply[0..n], false);
        try std.testing.expect(std.mem.startsWith(u8, now, shown));
        shown = now;
    }
    try std.testing.expectEqualStrings("I will not use <tool_call> here.\n\n\nDone. ", shown);
    // a block the reply has not closed hides to its end; the prose before it shows
    const open = "I will not use <tool_call> here.\n\n" ++ block[0 .. block.len - 20];
    try std.testing.expectEqualStrings("I will not use <tool_call> here.\n\n", try hideToolCalls(a, open, false));
}

test "a reply's marks lie where each marker token's text ends the decode through it" {
    const Fake = struct {
        pub fn decode(_: @This(), a: Allocator, ids: []const u32) Allocator.Error![]u8 {
            var out: std.ArrayList(u8) = .empty;
            for (ids) |t| try out.appendSlice(a, switch (t) {
                7 => "</think>",
                1 => "<",
                2 => "/think>",
                else => "x",
            });
            return out.toOwnedSlice(a);
        }
    };
    const gpa = std.testing.allocator;
    const pieces = [_]Marks.Piece{.{ .text = "</think>", .id = 7 }};
    var out: std.ArrayList(Marks.Mark) = .empty;
    defer out.deinit(gpa);
    // "x" "<" "/think>" "x" "</think>" "x": the spelled tag is no mark, the token at byte 10 is
    const tokens = [_]u32{ 3, 1, 2, 3, 7, 3 };
    try markTokens(gpa, gpa, Fake{}, &pieces, &tokens, 0, &out);
    try std.testing.expectEqual(@as(usize, 1), out.items.len);
    try std.testing.expectEqual(Marks.Mark{ .offset = 10, .id = 7 }, out.items[0]);
    // a stream scans only the tokens it has not scanned
    try markTokens(gpa, gpa, Fake{}, &pieces, &tokens, tokens.len, &out);
    try std.testing.expectEqual(@as(usize, 1), out.items.len);
}
