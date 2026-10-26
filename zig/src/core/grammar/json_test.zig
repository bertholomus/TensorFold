//! The json_object matcher: accepted and refused replies, its limits, and random walks that std.json must parse.
const std = @import("std");
const json = @import("json.zig");
const Matcher = json.Matcher;

const Want = union(enum) { complete, prefix, reject: usize };

fn run(bytes: []const u8) Want {
    var m: Matcher = .{};
    for (bytes, 0..) |b, i| if (!m.step(b)) return .{ .reject = i };
    return if (m.complete()) .complete else .prefix;
}

fn expectRun(want: Want, bytes: []const u8) !void {
    const got = run(bytes);
    if (!std.meta.eql(want, got)) {
        std.debug.print("{f}: want {any}, got {any}\n", .{ std.ascii.hexEscape(bytes, .lower), want, got });
        return error.TestUnexpectedResult;
    }
}

/// `n` copies of `c`.
fn rep(comptime c: u8, comptime n: usize) [n]u8 {
    return @splat(c);
}

fn at(i: usize) Want {
    return .{ .reject = i };
}

test "the root is one object with no blank before or after it" {
    try expectRun(.complete, "{}");
    try expectRun(.complete, "{\"a\":1}");
    try expectRun(at(0), " {}");
    try expectRun(at(0), "\n{}");
    try expectRun(at(2), "{} ");
    try expectRun(at(2), "{}\n");
    try expectRun(at(2), "{}{}");
    try expectRun(at(0), "[1]");
    try expectRun(at(0), "\"a\"");
    try expectRun(at(0), "1");
    try expectRun(at(0), "null");
    try expectRun(.prefix, "");
    try expectRun(.prefix, "{\"a\":[1,{\"b\":2}]");
}

test "objects and arrays: keys are strings, commas separate, closers match" {
    try expectRun(.complete, "{\"a\":[],\"b\":{},\"c\":[1,\"x\",true,false,null,{\"d\":[[]]}]}");
    try expectRun(.complete, "{ \"a\" : 1 , \"b\" :\t2\r\n}");
    try expectRun(.complete, "{\"a\":[ ],\"b\":{ }}");
    try expectRun(at(1), "{a:1}");
    try expectRun(at(1), "{1:2}");
    try expectRun(at(1), "{,}");
    try expectRun(at(5), "{\"a\" 1}");
    try expectRun(at(7), "{\"a\":1,}");
    try expectRun(at(8), "{\"a\":[1,]}");
    try expectRun(at(7), "{\"a\":1 2}");
    try expectRun(at(7), "{\"a\":[1}");
    try expectRun(at(6), "{\"a\":{]}");
    try expectRun(at(6), "{\"a\":1]");
    try expectRun(at(7), "{\"a\":1}}");
    try expectRun(at(5), "{\"a\":}");
}

test "strings: escapes, \\u escapes as xgrammar takes them, no raw control byte" {
    try expectRun(.complete, "{\"a\":\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"}");
    try expectRun(.complete, "{\"a\":\"\\u00e9\\u00C9\\uFFFF\\u0000\"}");
    try expectRun(.complete, "{\"\\u0000\":1}");
    try expectRun(.complete, "{\"a\":\"\\uD83D\\uDE00\"}");
    // xgrammar's escape rule is any four hex digits, so a lone surrogate half is taken too.
    try expectRun(.complete, "{\"a\":\"\\ud800\"}");
    try expectRun(.complete, "{\"a\":\"\\uDC00x\"}");
    try expectRun(at(7), "{\"a\":\"\\x\"}");
    try expectRun(at(7), "{\"a\":\"\\U0041\"}");
    try expectRun(at(10), "{\"a\":\"\\u00G0\"}");
    try expectRun(at(10), "{\"a\":\"\\u12\"}");
    try expectRun(.complete, "{\"a\":\"\x7f ~\"}");
    for ([_]u8{ 0x00, 0x09, 0x0a, 0x0d, 0x1f }) |c| try expectRun(at(6), &[_]u8{ '{', '"', 'a', '"', ':', '"', c, '"', '}' });
    try expectRun(.prefix, "{\"a\":\"open");
}

test "strings are strict UTF-8: no overlong form, surrogate or code point past U+10FFFF" {
    try expectRun(.complete, "{\"a\":\"\xc2\x80\xdf\xbf\xe0\xa0\x80\xe4\xb8\xad\xed\x9f\xbf\xee\x80\x80\xef\xbf\xbf\"}");
    try expectRun(.complete, "{\"a\":\"\xf0\x90\x80\x80\xf0\x9f\x98\x80\xf3\xbf\xbf\xbf\xf4\x8f\xbf\xbf\"}");
    try expectRun(.complete, "{\"\xe4\xb8\xad\":1}");
    try expectRun(at(6), "{\"a\":\"\xc0\xaf\"}");
    try expectRun(at(6), "{\"a\":\"\xc1\xbf\"}");
    try expectRun(at(7), "{\"a\":\"\xe0\x80\xaf\"}");
    try expectRun(at(7), "{\"a\":\"\xe0\x9f\xbf\"}");
    try expectRun(at(7), "{\"a\":\"\xf0\x8f\xbf\xbf\"}");
    try expectRun(at(7), "{\"a\":\"\xf4\x90\x80\x80\"}");
    try expectRun(at(6), "{\"a\":\"\xf5\x80\x80\x80\"}");
    try expectRun(at(6), "{\"a\":\"\xff\"}");
    try expectRun(at(6), "{\"a\":\"\x80\"}");
    try expectRun(at(8), "{\"a\":\"\xe4\xb8\"}");
    try expectRun(at(7), "{\"a\":\"\xc3A\"}");
    // xgrammar 0.2.8 takes an encoded surrogate (ED A0..BF); the reply would decode to U+FFFD, so it is refused here.
    try expectRun(at(7), "{\"a\":\"\xed\xa0\x80\"}");
    try expectRun(at(7), "{\"a\":\"\xed\xbf\xbf\"}");
}

test "numbers as JSON writes them" {
    const good = [_][]const u8{ "0", "-0", "1", "-12", "0.5", "-0.5", "1e5", "1E-5", "1e+05", "1.5E+3", "0e0", "10.01e-1" };
    for (good) |n| {
        var buf: [32]u8 = undefined;
        try expectRun(.complete, try std.fmt.bufPrint(&buf, "{{\"a\":{s}}}", .{n}));
        try expectRun(.complete, try std.fmt.bufPrint(&buf, "{{\"a\":[{s},{s} ]}}", .{ n, n }));
    }
    try expectRun(at(6), "{\"a\":01}");
    try expectRun(at(7), "{\"a\":-01}");
    try expectRun(at(7), "{\"a\":1.}");
    try expectRun(at(5), "{\"a\":.5}");
    try expectRun(at(5), "{\"a\":+1}");
    try expectRun(at(6), "{\"a\":-}");
    try expectRun(at(7), "{\"a\":1e}");
    try expectRun(at(8), "{\"a\":1e+}");
    try expectRun(at(7), "{\"a\":1.e5}");
    try expectRun(at(8), "{\"a\":1.5.2}");
    try expectRun(at(6), "{\"a\":0x1}");
    try expectRun(at(6), "{\"a\":--1}");
}

test "literals" {
    try expectRun(.complete, "{\"a\":true,\"b\":false,\"c\":null}");
    try expectRun(at(8), "{\"a\":tru}");
    try expectRun(at(5), "{\"a\":True}");
    try expectRun(at(9), "{\"a\":nulll}");
    try expectRun(.prefix, "{\"a\":nul");
}

test "a run of exactly 32 blanks is taken in every place, the 33rd is not" {
    const places = [_][2][]const u8{
        .{ "{", "}" },           .{ "{", "\"a\":1}" },         .{ "{\"a\"", ":1}" },         .{ "{\"a\":", "1}" },
        .{ "{\"a\":1", "}" },    .{ "{\"a\":1", ",\"b\":2}" }, .{ "{\"a\":1,", "\"b\":2}" }, .{ "{\"a\":[", "]}" },
        .{ "{\"a\":[", "1]}" },  .{ "{\"a\":[1", "]}" },       .{ "{\"a\":[1,", "2]}" },     .{ "{\"a\":\"x\"", "}" },
        .{ "{\"a\":true", "}" }, .{ "{\"a\":{}", "}" },
    };
    const mix = " \t\r\n";
    for (places) |p| for ([_]usize{ 32, 33 }) |n| {
        var buf: [96]u8 = undefined;
        @memcpy(buf[0..p[0].len], p[0]);
        for (0..n) |i| buf[p[0].len + i] = mix[i % mix.len];
        @memcpy(buf[p[0].len + n ..][0..p[1].len], p[1]);
        const text = buf[0 .. p[0].len + n + p[1].len];
        try expectRun(if (n == 32) .complete else at(p[0].len + 32), text);
    };
    // Blanks inside a string are text, not a run.
    try expectRun(.complete, "{\"a\":\"" ++ rep(' ', 40) ++ "\"}");
}

test "containers nest at most 128 deep" {
    const deep = "{\"a\":" ++ rep('[', 127) ++ "1" ++ rep(']', 127) ++ "}";
    try expectRun(.complete, deep);
    try expectRun(at(5 + 127), "{\"a\":" ++ rep('[', 128));
    try expectRun(at(5 + 127), "{\"a\":" ++ rep('[', 127) ++ "{");
    var m: Matcher = .{};
    try std.testing.expect(m.advance("{\"a\":" ++ rep('[', 127)));
    try std.testing.expectEqual(@as(u8, json.max_depth), m.depth);
    try std.testing.expect(m.accepts("]") and m.accepts("1") and m.accepts("\"") and !m.accepts("[") and !m.accepts("{"));
    try std.testing.expect(m.advance("{}" ++ rep(']', 127) ++ "}") == false);
    try std.testing.expect(m.advance("{}") == false and m.advance("1" ++ rep(']', 127) ++ "}") and m.complete());
}

test "a refused byte leaves the matcher as it was; advance takes all or nothing; accepts takes none" {
    var m: Matcher = .{};
    try std.testing.expect(m.advance("{\"k\":[1"));
    const before = m;
    try std.testing.expect(!m.step('}'));
    try std.testing.expectEqual(before, m);
    try std.testing.expect(!m.advance(",2}"));
    try std.testing.expectEqual(before, m);
    try std.testing.expect(m.accepts(",2]}") and !m.complete());
    try std.testing.expectEqual(before, m);
    try std.testing.expect(m.advance(",2]}") and m.complete());
}

test "complete only once the object closes, and nothing after it" {
    const doc = "{\"a\":[1,{\"b\":\"c\"}],\"d\":-1.5e3}";
    var m: Matcher = .{};
    for (doc, 0..) |b, i| {
        try std.testing.expect(!m.complete());
        try std.testing.expect(m.step(b));
        try std.testing.expectEqual(i + 1 == doc.len, m.complete());
    }
    for (0..256) |b| try std.testing.expect(!m.accepts(&.{@intCast(b)}));
}

/// The bytes `m` takes next.
fn allowed(m: Matcher) [256]bool {
    var out: [256]bool = undefined;
    for (&out, 0..) |*o, b| o.* = m.accepts(&.{@intCast(b)});
    return out;
}

/// A walk's weight for byte `b`: closers and quotes win often, so most walks finish within the budget.
fn weight(m: Matcher, b: u8, len: usize) u32 {
    // std.json refuses a lone surrogate escape that xgrammar takes, so walks write none.
    if (m.lex == .hex and m.count == 0 and (b == 'd' or b == 'D')) return 0;
    return switch (b) {
        '}', ']' => 24,
        '"' => if (m.lex == .string) 6 else 10,
        ':' => 20,
        ',' => 5,
        '{', '[' => if (len > 300 or m.depth > 8) 0 else 4,
        '0'...'9' => 3,
        ' ', '\t', '\r', '\n' => 1,
        '\\' => 3,
        0x80...0xFF => 1,
        else => 1,
    };
}

fn walk(rng: std.Random, out: []u8) ?[]u8 {
    var m: Matcher = .{};
    var n: usize = 0;
    while (n < out.len) {
        if (m.complete()) return out[0..n];
        const ok = allowed(m);
        var total: u32 = 0;
        for (ok, 0..) |o, b| total += if (o) weight(m, @intCast(b), n) else 0;
        if (total == 0) return null;
        var pick = rng.uintLessThan(u32, total);
        const b: u8 = for (ok, 0..) |o, c| {
            const w = if (o) weight(m, @intCast(c), n) else 0;
            if (pick < w) break @intCast(c);
            pick -= w;
        } else unreachable;
        if (!m.step(b)) unreachable;
        out[n] = b;
        n += 1;
    }
    return null;
}

test "random walks over accepted bytes never stall, and every finished walk parses with std.json as an object" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7f4a7c15);
    var buf: [2048]u8 = undefined;
    var finished: usize = 0;
    const walks = 3000;
    for (0..walks) |_| {
        var m: Matcher = .{};
        const doc = walk(prng.random(), &buf) orelse continue;
        finished += 1;
        try std.testing.expect(m.advance(doc) and m.complete());
        const parsed = std.json.parseFromSlice(std.json.Value, a, doc, .{}) catch |e| {
            std.debug.print("std.json refused {f}: {t}\n", .{ std.ascii.hexEscape(doc, .lower), e });
            return e;
        };
        defer parsed.deinit();
        try std.testing.expect(parsed.value == .object);
    }
    try std.testing.expect(finished * 10 >= walks * 9);
}

test "no state the matcher reaches is a dead end: something is always allowed until the object closes" {
    var prng = std.Random.DefaultPrng.init(0x9e3779b9);
    const rng = prng.random();
    for (0..2000) |_| {
        var m: Matcher = .{};
        for (0..200) |_| {
            const ok = allowed(m);
            const count = std.mem.count(bool, &ok, &.{true});
            try std.testing.expectEqual(m.complete(), count == 0);
            if (count == 0) break;
            var pick = rng.uintLessThan(usize, count);
            for (ok, 0..) |o, b| if (o) {
                if (pick == 0) {
                    try std.testing.expect(m.step(@intCast(b)));
                    break;
                }
                pick -= 1;
            };
        }
    }
}
