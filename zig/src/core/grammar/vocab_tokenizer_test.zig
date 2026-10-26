//! tokenizer.tokenBytes and grammar.Vocab over a real byte-level BPE tokenizer, as a server would build the table.
const std = @import("std");
const grammar = @import("grammar");
const tokenizer = @import("tokenizer");
const Vocab = grammar.Vocab;

const eos_text = "<｜end▁of▁sentence｜>";
const think_close = "</think>";
const dsml = "｜DSML｜";
/// Ids: bytes 0..255, the half character E4 B8 (256), `{"` (257), then the added eos, </think> and DSML markers.
const half = 256;
const brace_quote = 257;
const eos = 258;
const think = 259;
const marker = 260;
const width = 264;

/// GPT-2's byte-to-character map that ByteLevel uses.
fn byteChar(b: u8) u21 {
    if ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174) return b;
    var n: u21 = 0;
    for (0..b) |c| {
        if (!((c >= 33 and c <= 126) or (c >= 161 and c <= 172) or c >= 174)) n += 1;
    }
    return 256 + n;
}

fn appendToken(out: *std.ArrayList(u8), a: std.mem.Allocator, bytes: []const u8) !void {
    try out.append(a, '"');
    for (bytes) |b| {
        const c = byteChar(b);
        if (c == '"' or c == '\\') try out.append(a, '\\');
        var buf: [4]u8 = undefined;
        try out.appendSlice(a, buf[0..try std.unicode.utf8Encode(c, &buf)]);
    }
    try out.append(a, '"');
}

/// A ByteLevel BPE tokenizer: every byte, two merges, and three added tokens (the eos is special).
fn fixture(a: std.mem.Allocator) !tokenizer.Tokenizer {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try out.print(a,
        \\{{"added_tokens":[{{"id":{d},"content":"{s}","special":true,"lstrip":false,"rstrip":false,"normalized":false,"single_word":false}},
        \\ {{"id":{d},"content":"{s}","special":false,"lstrip":false,"rstrip":false,"normalized":false,"single_word":false}},
        \\ {{"id":{d},"content":"{s}","special":false,"lstrip":false,"rstrip":false,"normalized":true,"single_word":false}}],
        \\ "normalizer":null,
        \\ "pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},
        \\ "decoder":{{"type":"ByteLevel"}},
        \\ "model":{{"type":"BPE","dropout":null,"unk_token":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{
    , .{ eos, eos_text, think, think_close, marker, dsml });
    for (0..256) |b| {
        try appendToken(&out, a, &.{@intCast(b)});
        try out.print(a, ":{d},", .{b});
    }
    try appendToken(&out, a, "\xe4\xb8");
    try out.print(a, ":{d},", .{half});
    try appendToken(&out, a, "{\"");
    try out.print(a, ":{d}}},\"merges\":[", .{brace_quote});
    for ([_][2]u8{ .{ 0xe4, 0xb8 }, .{ '{', '"' } }, 0..) |m, i| {
        if (i > 0) try out.append(a, ',');
        try out.append(a, '[');
        try appendToken(&out, a, &.{m[0]});
        try out.append(a, ',');
        try appendToken(&out, a, &.{m[1]});
        try out.append(a, ']');
    }
    try out.appendSlice(a, "]}}");
    return tokenizer.parse(a, out.items);
}

fn bytesOf(t: *const tokenizer.Tokenizer, id: u32) !?[]u8 {
    return t.tokenBytes(std.testing.allocator, id);
}

test "tokenBytes: each id's bytes as the ByteLevel decoder joins them, half characters kept" {
    const a = std.testing.allocator;
    var t = try fixture(a);
    defer t.deinit();
    for (0..256) |b| {
        const got = (try bytesOf(&t, @intCast(b))).?;
        defer a.free(got);
        try std.testing.expectEqualSlices(u8, &.{@intCast(b)}, got);
    }
    const cases = [_]struct { u32, []const u8 }{
        .{ half, "\xe4\xb8" }, .{ brace_quote, "{\"" }, .{ eos, eos_text }, .{ think, think_close }, .{ marker, dsml },
    };
    for (cases) |c| {
        const got = (try bytesOf(&t, c[0])).?;
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
        const text = try t.decode(a, &.{c[0]}, false);
        defer a.free(text);
        if (c[0] != half) try std.testing.expectEqualStrings(text, got);
    }
    try std.testing.expectEqual(@as(?[]u8, null), try bytesOf(&t, width + 5));
    // The decoder repairs a half character to U+FFFD on its own, and joins it whole beside its other half.
    const alone = try t.decode(a, &.{half}, false);
    defer a.free(alone);
    try std.testing.expectEqualStrings("\u{FFFD}", alone);
    const joined = try t.decode(a, &.{ half, 0xad }, false);
    defer a.free(joined);
    try std.testing.expectEqualStrings("中", joined);
}

test "tokenBytes refuses a decoder whose pieces depend on their neighbours" {
    const json =
        \\{"added_tokens":[],"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
        \\ "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
        \\ "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
        \\ "model":{"type":"BPE","unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,
        \\  "vocab":{"<unk>":0,"<0xC3>":1,"<0xA9>":2,"h":3,"i":4,"▁":5,"hi":6,"▁h":7},"merges":["h i","▁ h"]}}
    ;
    var t = try tokenizer.parse(std.testing.allocator, json);
    defer t.deinit();
    try std.testing.expectError(error.UnsupportedDecoder, bytesOf(&t, 3));
}

test "a vocab table from the tokenizer: added tokens by id, eos, padded columns, and a reply walked token by token" {
    const a = std.testing.allocator;
    var t = try fixture(a);
    defer t.deinit();
    var v = try Vocab.build(a, width, &t, &.{eos});
    defer v.deinit(a);
    try std.testing.expectEqual(@as(usize, width), v.width());
    try std.testing.expectEqual(grammar.vocab.Kind.eos, v.kind(eos));
    try std.testing.expectEqual(grammar.vocab.Kind.special, v.kind(think));
    try std.testing.expectEqual(grammar.vocab.Kind.special, v.kind(marker));
    for (marker + 1..width) |id| try std.testing.expectEqual(grammar.vocab.Kind.none, v.kind(@intCast(id)));
    try std.testing.expectEqualStrings("\xe4\xb8", v.piece(half));
    const reply = "{\"中\":[1, \"x\"]}";
    const ids = try t.encode(a, reply);
    defer a.free(ids);
    try std.testing.expect(std.mem.indexOfScalar(u32, ids, half) != null and ids[0] == brace_quote);
    var m: grammar.Matcher = .{};
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    for (ids) |id| {
        try std.testing.expect(!v.allows(m, eos));
        try std.testing.expect(v.allows(m, id));
        try std.testing.expect(m.advance(v.piece(id)));
        try text.appendSlice(a, v.piece(id));
    }
    try std.testing.expectEqualStrings(reply, text.items);
    try std.testing.expect(m.complete() and v.allows(m, eos));
    // "</think>" inside a key is valid JSON by its bytes, but its token is markup and never allowed.
    const marked = try t.encode(a, "{\"</think>");
    defer a.free(marked);
    try std.testing.expectEqualSlices(u32, &.{ brace_quote, think }, marked);
    var k: grammar.Matcher = .{};
    try std.testing.expect(k.advance("{\"") and k.accepts(think_close) and !v.allows(k, think));
}
