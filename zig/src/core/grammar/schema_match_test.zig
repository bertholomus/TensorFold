//! The program matcher: hand cases, limits, trials against steps, and random walks a separate validator accepts.
const std = @import("std");
const schema = @import("schema.zig");
const keys = @import("schema_keys.zig");
const formats = @import("formats.zig");
const Matcher = @import("schema_match.zig").Matcher;
const Value = std.json.Value;
const Allocator = std.mem.Allocator;

const Want = union(enum) { complete, prefix, reject: usize };

fn run(p: *const schema.Program, bytes: []const u8) Want {
    var m: Matcher = .init(p);
    for (bytes, 0..) |b, i| if (!m.step(b)) return .{ .reject = i };
    return if (m.complete()) .complete else .prefix;
}

fn expectRun(p: *const schema.Program, want: Want, bytes: []const u8) !void {
    const got = run(p, bytes);
    if (!std.meta.eql(want, got)) {
        std.debug.print("{s}: want {any}, got {any}\n", .{ bytes, want, got });
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

fn compile(a: Allocator, text: []const u8) !schema.Program {
    var why: []const u8 = "";
    return schema.compile(a, text, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ text, why });
        return e;
    };
}

const signal =
    \\{"type": "object", "additionalProperties": false, "required": ["action", "confidence", "pair"],
    \\ "properties": {"action": {"type": "string", "enum": ["buy", "sell", "hold"]}, "confidence": {"type": "number"},
    \\ "pair": {"type": "string"}}}
;

test "a trade signal schema: its keys in order, an enum value, no other key, blanks only between tokens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try compile(arena.allocator(), signal);
    try expectRun(&p, .complete, "{\"action\":\"buy\",\"confidence\":0.72,\"pair\":\"BTC/USD\"}");
    try expectRun(&p, .complete, "{ \"action\" : \"hold\" ,\n\t\"confidence\": -1e-3, \"pair\": \"\\u00e9\" }");
    try expectRun(&p, at(0), " {\"action\":\"buy\"");
    try expectRun(&p, at(2), "{\"confidence\":1,\"action\":\"buy\",\"pair\":\"x\"}");
    try expectRun(&p, at(12), "{\"action\":\"short\"}");
    try expectRun(&p, at(16), "{\"action\":\"sell\"}");
    try expectRun(&p, at(42), "{\"action\":\"sell\",\"confidence\":1,\"pair\":\"x\",\"note\":1}");
    try expectRun(&p, at(43), "{\"action\":\"sell\",\"confidence\":1,\"pair\":\"x\"} ");
    try expectRun(&p, .prefix, "{\"action\":\"sell\",\"confidence\":1");
}

test "optional properties may be left out in order; other keys come last and are never a property's name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try compile(a,
        \\{"type": "object", "properties": {"a": {"type": "integer"}, "ab": {"type": "string"}, "b": {"type": "boolean"}},
        \\ "required": ["ab"], "additionalProperties": {"type": "null"}}
    );
    try expectRun(&p, .complete, "{\"ab\":\"x\"}");
    try expectRun(&p, .complete, "{\"a\":1,\"ab\":\"x\",\"b\":true}");
    try expectRun(&p, .complete, "{\"ab\":\"x\",\"abc\":null,\"\":null,\"\\u0061\":null}");
    try expectRun(&p, at(6), "{\"a\":1}"); // ab is required
    try expectRun(&p, at(2), "{\"b\":true,\"ab\":\"x\"}"); // b may not come before ab
    try expectRun(&p, at(2), "{\"z\":null,\"ab\":\"x\"}"); // other keys only once nothing is required
    try expectRun(&p, at(12), "{\"ab\":\"x\",\"a\":1}"); // a is a property's name, so never another key
    try expectRun(&p, at(22), "{\"ab\":\"x\",\"zz\":null,\"b\":true}"); // no property after another key
    const none = try compile(a, "{\"type\": \"object\", \"properties\": {\"k\": {\"type\": \"integer\"}}}");
    try expectRun(&none, .complete, "{}");
    try expectRun(&none, .complete, "{ \"k\" : 0 }");
    try expectRun(&none, at(2), "{\"j\":0}");
}

test "keys that start each other in any order, a required one between, other keys after it (xgrammar 0.2.8 verdicts)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try compile(arena.allocator(),
        \\{"type": "object", "properties": {"ab": {"type": "integer"}, "a": {"type": "integer"}, "abc": {"type": "integer"},
        \\ "b": {"type": "integer"}}, "required": ["abc"], "additionalProperties": true}
    );
    try expectRun(&p, .complete, "{\"abc\":1}");
    try expectRun(&p, .complete, "{\"a\":1,\"abc\":1}");
    try expectRun(&p, .complete, "{\"ab\":1,\"a\":1,\"abc\":1}");
    try expectRun(&p, at(10), "{\"a\":1,\"ab\":1,\"abc\":1}"); // ab comes before a
    try expectRun(&p, at(2), "{\"b\":1}"); // b lies past the required abc
    try expectRun(&p, .complete, "{\"abc\":1,\"zz\":1,\"\":1}");
    try expectRun(&p, at(12), "{\"abc\":1,\"ab\":1}"); // a passed name is no other key
    try expectRun(&p, at(11), "{\"abc\":1,\"a\":1}");
    try expectRun(&p, at(19), "{\"ab\":1,\"abc\":1,\"ab\":1}");
    try expectRun(&p, .complete, "{\"abc\":1,\"b\":1,\"abx\":1,\"abcd\":1}");
    try expectRun(&p, .prefix, "{\"ab");
}

test "an object of 30,000 properties takes each key in order; its key trie lists the properties under each prefix" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "{\"type\": \"object\", \"properties\": {");
    for (0..30_000) |i| try text.print(a, "{s}\"p{d}\": {{\"type\": \"integer\"}}", .{ if (i == 0) "" else ", ", i });
    try text.appendSlice(a, "}}");
    const p = try compile(a, text.items);
    try expectRun(&p, .complete, "{\"p29999\":1}");
    try expectRun(&p, .complete, "{\"p0\":1,\"p1\":1,\"p10\":1,\"p100\":1,\"p1000\":1,\"p10000\":1}");
    try expectRun(&p, at(11), "{\"p5\":1,\"p4\":1}");
    try expectRun(&p, .complete, "{\"p5\":1,\"p50\":1,\"p500\":1,\"p5000\":1}");
    try expectRun(&p, at(12), "{\"p50\":1,\"p5\":1}"); // p5 still starts p500 and on, so its closing quote is refused
    try expectRun(&p, at(7), "{\"p30000\":1}");
    // a key byte reads one trie node and searches its list: it never scans all 30,000 properties
    const o = p.objects[p.nodes[p.root].object];
    var node = o.trie;
    for ("p2") |b| node = p.child(node, b).?;
    const span = o.below[node - o.trie];
    const below = o.members[span.first..][0..span.len];
    try std.testing.expectEqual(@as(usize, 1 + 10 + 100 + 1000 + 10_000), below.len);
    try std.testing.expectEqual(@as(u16, 2), below[0]);
    for (below[1..], below[0 .. below.len - 1]) |x, prev| {
        try std.testing.expect(x > prev);
        try std.testing.expect(std.mem.startsWith(u8, o.props[x].name, "p2"));
    }
}

test "every key trie node lists exactly the properties whose keys start with its bytes, in order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(20261026);
    const r = prng.random();
    for (0..200) |_| {
        var names: std.ArrayList([]const u8) = .empty;
        for (0..r.uintLessThan(usize, 12)) |_| {
            const name = try a.alloc(u8, r.uintLessThan(usize, 5));
            for (name) |*c| c.* = "ab\"\\\x01\xc3\xa9"[r.uintLessThan(usize, 7)];
            if (!std.unicode.utf8ValidateSlice(name)) continue;
            for (names.items) |n| {
                if (std.mem.eql(u8, n, name)) break;
            } else try names.append(a, name);
        }
        var text: std.ArrayList(u8) = .empty;
        try text.appendSlice(a, "{\"type\": \"object\", \"properties\": {");
        for (names.items, 0..) |n, i| {
            if (i > 0) try text.appendSlice(a, ", ");
            try keys.writeString(a, &text, n);
            try text.appendSlice(a, ": {}");
        }
        try text.appendSlice(a, "}}");
        const p = try compile(a, text.items);
        const o = p.objects[p.nodes[p.root].object];
        try std.testing.expectEqual(names.items.len, o.props.len);
        // walk the trie by every prefix of every key, and compare its list with a scan
        for (o.props) |prop| for (0..prop.key.len) |n| {
            var node = o.trie;
            for (prop.key[1 .. n + 1]) |b| node = p.child(node, b).?;
            const span = o.below[node - o.trie];
            var want: std.ArrayList(u16) = .empty;
            for (o.props, 0..) |q, j| if (std.mem.startsWith(u8, q.key, prop.key[0 .. n + 1])) try want.append(a, @intCast(j));
            try std.testing.expectEqualSlices(u16, want.items, o.members[span.first..][0..span.len]);
            try std.testing.expectEqual(n + 1 == prop.key.len, p.tries[node].end);
        };
    }
}

test "arrays: prefix items, then items within minItems and maxItems; two blank runs in an empty prefix array" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try compile(a, "{\"type\": \"array\", \"prefixItems\": [{\"type\": \"integer\"}, {\"type\": \"string\"}], \"items\": {\"type\": \"null\"}, \"maxItems\": 3}");
    try expectRun(&p, .complete, "[]");
    try expectRun(&p, .complete, "[1]");
    try expectRun(&p, .complete, "[1,\"a\",null]");
    try expectRun(&p, at(11), "[1,\"a\",null,null]");
    try expectRun(&p, at(1), "[\"a\"]");
    try expectRun(&p, .complete, "[" ++ &rep(' ', 64) ++ "]");
    try expectRun(&p, at(65), "[" ++ &rep(' ', 65) ++ "]");
    try expectRun(&p, at(34), "[" ++ &rep(' ', 33) ++ "1]");
    const counted = try compile(a, "{\"type\": \"array\", \"items\": {\"type\": \"integer\"}, \"minItems\": 2, \"maxItems\": 3}");
    try expectRun(&counted, at(2), "[1]");
    try expectRun(&counted, .complete, "[1,2]");
    try expectRun(&counted, at(6), "[1,2,3,4]");
    try expectRun(&counted, at(1), "[]");
    try expectRun(&counted, at(33), "[" ++ &rep(' ', 33) ++ "1,2]");
}

test "strings: lengths count characters and take no escape, formats follow their pattern" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try compile(a, "{\"type\": \"string\", \"minLength\": 2, \"maxLength\": 3}");
    try expectRun(&text, .complete, "\"\u{e9}\u{1f600}\"");
    try expectRun(&text, .complete, "\"a\tb\"");
    try expectRun(&text, at(2), "\"a\"");
    try expectRun(&text, at(4), "\"abcd\"");
    try expectRun(&text, at(2), "\"a\\nb\"");
    try expectRun(&text, at(2), "\"\xed\xa0\x80\""); // xgrammar takes a surrogate, as for json_object
    const date = try compile(a, "{\"type\": \"string\", \"format\": \"date-time\"}");
    try expectRun(&date, .complete, "\"2026-10-26T10:15:00+01:00\"");
    try expectRun(&date, at(11), "\"2026-10-26 10:15:00Z\"");
}

test "numbers and literals at the root end with the reply; integers refuse a fraction and -0" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const int = try compile(a, "{\"type\": \"integer\"}");
    try expectRun(&int, .complete, "-12");
    try expectRun(&int, at(1), "-0");
    try expectRun(&int, at(1), "1.5");
    try expectRun(&int, at(1), "01");
    try expectRun(&int, at(2), "12 ");
    const choice = try schema.choice(a, &.{ "a", "ab", "yes" });
    try expectRun(&choice, .complete, "a");
    try expectRun(&choice, .complete, "ab");
    try expectRun(&choice, at(2), "abc");
    try expectRun(&choice, .prefix, "ye");
    try expectRun(&choice, at(0), " yes");
    const lits = try compile(a, "{\"enum\": [1, 12, \"x\", {\"k\": [true]}]}");
    try expectRun(&lits, .complete, "1");
    try expectRun(&lits, .complete, "12");
    try expectRun(&lits, .complete, "{\"k\":[true]}");
    try expectRun(&lits, at(5), "{\"k\": [true]}");
    try expectRun(&lits, at(2), "123");
}

test "a pydantic schema: confidence takes 0.0 to 1.0 as xgrammar 0.2.8 did, inside a whole object" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try compile(arena.allocator(), @embedFile("testdata/pydantic_bounds.json"));
    const head = "{\"label\":\"yes\",\"confidence\":";
    var buf: [128]u8 = undefined;
    for ([_][]const u8{ "0", "0.0", "0.85", "1", "1.0", "1.000000", "0.6" }) |x|
        try expectRun(&p, .complete, try std.fmt.bufPrint(&buf, "{s}{s},\"summary\":\"s\"}}", .{ head, x }));
    // the first refused byte, counted in the value
    for ([_]struct { []const u8, usize }{ .{ "1.01", 3 }, .{ "-0.1", 0 }, .{ "2", 0 }, .{ "1.0000000", 8 }, .{ "0.0000001", 8 }, .{ "-0", 0 }, .{ "1e0", 1 }, .{ ".5", 0 }, .{ "00.5", 1 } }) |c|
        try expectRun(&p, at(head.len + c[1]), try std.fmt.bufPrint(&buf, "{s}{s},\"summary\":\"s\"}}", .{ head, c[0] }));
    const full =
        \\{"label": "unsure", "confidence": 0.55, "summary": "x", "tags": ["a"], "note": null,
        \\ "extra": {"k": [1, "v"]}, "source": {"url": "u", "title": null}}
    ;
    try expectRun(&p, .complete, full);
    try expectRun(&p, at(36), "{\"label\": \"unsure\", \"confidence\": 1.5, \"summary\": \"x\"}");
    try expectRun(&p, .complete, "{\"label\":\"no\",\"confidence\":1,\"summary\":\"\"}");
    try expectRun(&p, at(31), "{\"label\":\"yes\",\"confidence\":0.5}");
    try expectRun(&p, at(57), "{\"label\":\"yes\",\"confidence\":0.5,\"summary\":\"s\",\"source\":{\"title\":\"t\"}}");
}

test "containers nest at most 128 deep; a token that opens more than the trial holds is tried on a copy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try compile(arena.allocator(), "{}");
    try expectRun(&p, .complete, &rep('[', 128) ++ &rep(']', 128));
    try expectRun(&p, at(128), &rep('[', 129)); // xgrammar has no limit
    var m: Matcher = .init(&p);
    try std.testing.expect(m.advance("[{\"a\":[[1,{}],"));
    for ([_][]const u8{ "[[[[[[[[[[[[1", "{\"b\":[[[[[[[[[[[{}]]]]]]]]]]]}", "1]]]],[[[[[[[[[[" }) |token| {
        var copy = m;
        var ok = true;
        for (token) |b| ok = ok and copy.step(b);
        try std.testing.expectEqual(ok, m.accepts(token));
    }
    try std.testing.expect(m.accepts("[[[[[[[[[[[[[[[[[[[[1"));
    try std.testing.expect(!m.accepts("[[[[[[[[[[[[[[[[[[[[}"));
    // a token that closes containers the trial did not open reads them in order
    var deep: Matcher = .init(&p);
    try std.testing.expect(deep.advance("[{\"a\":[{\"b\":[[1"));
    try std.testing.expect(deep.accepts("]]}]}]") and !deep.accepts("]]]}]}") and !deep.accepts("]]}}]"));
    try std.testing.expect(deep.advance("]]}]}]") and deep.complete());
}

const walked = [_][]const u8{
    signal,
    \\{"type": "object", "properties": {"steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
    \\ "final_answer": {"type": "string"}}, "$defs": {"step": {"type": "object", "properties": {"explanation":
    \\ {"type": "string"}, "output": {"type": "string"}}, "required": ["explanation", "output"],
    \\ "additionalProperties": false}}, "required": ["steps", "final_answer"], "additionalProperties": false}
    ,
    \\{"type": "object", "properties": {"type": {"type": "string", "enum": ["div", "button", "header"]}, "label":
    \\ {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#"}}, "attributes": {"type": "array",
    \\ "items": {"type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
    \\ "required": ["name", "value"], "additionalProperties": false}}}, "required": ["type", "label", "children",
    \\ "attributes"], "additionalProperties": false}
    ,
    \\{"type": "object", "properties": {"is_violating": {"type": "boolean"}, "category": {"type": ["string", "null"],
    \\ "enum": ["violence", "sexual", "self_harm", null]}, "explanation_if_violating": {"type": ["string", "null"]}},
    \\ "required": ["is_violating", "category", "explanation_if_violating"], "additionalProperties": false}
    ,
    \\{"type": "object", "properties": {"linked_list": {"$ref": "#/$defs/node"}}, "$defs": {"node": {"type": "object",
    \\ "properties": {"value": {"type": "number"}, "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}},
    \\ "additionalProperties": false, "required": ["next", "value"]}}, "required": ["linked_list"]}
    ,
    \\{"title": "Event", "type": "object", "properties": {"when": {"type": "string", "format": "date-time"},
    \\ "id": {"type": "string", "format": "uuid"}, "ip": {"type": "string", "format": "ipv4"},
    \\ "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}, "tags": {"items": {"type":
    \\ "string"}, "type": "array", "maxItems": 3}}, "required": ["when", "id", "tags"], "additionalProperties": false}
    ,
    \\{"type": "object", "properties": {"a": {"type": "integer"}, "ab": {"type": "string"}, "abc": {"type": "boolean"}},
    \\ "additionalProperties": {"type": "array", "prefixItems": [{"type": "integer"}, {"const": "x"}], "items":
    \\ {"type": "null"}}}
    ,
    \\{"type": "array", "items": {"anyOf": [{"type": "string", "format": "date"}, {"type": ["integer", "number"]},
    \\ {"enum": [true, false]}, {"type": "null"}, {"type": "object"}]}, "minItems": 1}
    ,
    \\{"type": ["object", "null"], "properties": {"x": {}, "y": {"type": "array"}}, "required": ["x"]}
    ,
    \\{"type": "object", "properties": {"ab": {"type": "integer"}, "a": {"type": "integer"}, "abc": {"type": "integer"},
    \\ "b": {"type": "integer"}}, "required": ["abc"], "additionalProperties": true}
    ,
    \\{"type": "object", "properties": {"confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "n": {"type":
    \\ "integer", "exclusiveMinimum": -5, "maximum": 1000}, "xs": {"type": "array", "items": {"anyOf": [{"type":
    \\ "integer", "minimum": -2, "exclusiveMaximum": 3}, {"type": "number", "minimum": -2.5, "maximum": -2.25}]}}},
    \\ "required": ["confidence"]}
    ,
};

/// A walk's weight for byte `b`: closers and quotes win often, so most walks finish within the budget.
fn weight(m: *const Matcher, b: u8, len: usize) u32 {
    const lx = m.lexer;
    // std.json refuses a lone surrogate escape that xgrammar takes, so walks write none
    if (lx.esc == .hex and lx.hex == 0 and (b == 'd' or b == 'D')) return 0;
    return switch (b) {
        '}', ']' => 24,
        '"' => if (lx.lex == .string) 8 else 10,
        ':' => 20,
        ',' => if (len > 200) 1 else 6,
        '{', '[' => if (len > 200 or lx.depth > 6) 0 else 4,
        '0'...'9' => 3,
        ' ', '\t', '\r', '\n' => 1,
        else => 1,
    };
}

fn walk(p: *const schema.Program, rng: std.Random, out: []u8) ?[]u8 {
    var m: Matcher = .init(p);
    var n: usize = 0;
    while (n < out.len) {
        var ok: [256]bool = undefined;
        var total: u32 = 0;
        for (&ok, 0..) |*o, b| {
            o.* = m.accepts(&.{@intCast(b)});
            total += if (o.*) weight(&m, @intCast(b), n) else 0;
        }
        if (m.complete() and (total == 0 or rng.uintLessThan(u32, 4) == 0)) return out[0..n];
        if (total == 0) return null;
        var pick = rng.uintLessThan(u32, total);
        const b: u8 = for (ok, 0..) |o, c| {
            const w = if (o) weight(&m, @intCast(c), n) else 0;
            if (pick < w) break @intCast(c);
            pick -= w;
        } else unreachable;
        if (!m.step(b)) unreachable;
        out[n] = b;
        n += 1;
    }
    return null;
}

/// A validator for the walks, written from the schema text and not from the program.
const Check = struct {
    a: Allocator,
    top: Value,

    fn resolve(c: Check, s: Value) Value {
        var cur = s;
        while (cur == .object and cur.object.get("$ref") != null) {
            const uri = cur.object.get("$ref").?.string;
            cur = c.top;
            var parts = std.mem.splitScalar(u8, uri[1..], '/');
            while (parts.next()) |part| if (part.len > 0) {
                cur = cur.object.get(part).?;
            };
        }
        return cur;
    }

    fn literal(c: Check, v: Value) []const u8 {
        var out: std.ArrayList(u8) = .empty;
        keys.write(c.a, &out, v) catch return "\x00"; // a reply number xgrammar would round: equal to no literal
        return out.items;
    }

    fn valid(c: Check, schema_: Value, v: Value) bool {
        const s = c.resolve(schema_);
        if (s == .bool) return s.bool;
        const o = s.object;
        if (o.get("const")) |k| return std.mem.eql(u8, c.literal(k), c.literal(v));
        if (o.get("enum")) |e| return for (e.array.items) |k| {
            if (std.mem.eql(u8, c.literal(k), c.literal(v))) break true;
        } else false;
        inline for (.{ "anyOf", "oneOf" }) |word| if (o.get(word)) |branches| return for (branches.array.items) |b| {
            if (c.valid(b, v)) break true;
        } else false;
        if (o.get("type")) |t| {
            if (t == .string) return c.typed(o, t.string, v);
            return for (t.array.items) |name| {
                if (c.typed(o, name.string, v)) break true;
            } else false;
        }
        if (o.contains("properties") or o.contains("additionalProperties")) return v == .object and c.object(o, v.object);
        if (o.contains("items") or o.contains("prefixItems")) return v == .array and c.array(o, v.array.items);
        return true;
    }

    fn typed(c: Check, o: std.json.ObjectMap, name: []const u8, v: Value) bool {
        const eql = std.mem.eql;
        if (eql(u8, name, "string")) {
            if (v != .string) return false;
            const f = o.get("format") orelse return true;
            const n = formats.nfa(formats.named(f.string).?);
            var state = formats.Nfa.start;
            for (v.string) |b| state = n.step(state, b);
            return n.accepting(state);
        }
        if (eql(u8, name, "integer")) return v == .number_string and std.mem.findAny(u8, v.number_string, ".eE") == null and !eql(u8, v.number_string, "-0") and inRange(o, v.number_string);
        if (eql(u8, name, "number")) return v == .number_string and inRange(o, v.number_string);
        if (eql(u8, name, "boolean")) return v == .bool;
        if (eql(u8, name, "null")) return v == .null;
        if (eql(u8, name, "object")) return v == .object and (keys.plain(o) or c.object(o, v.object));
        return v == .array and (keys.plain(o) or c.array(o, v.array.items));
    }

    /// In range, without exponent, at most 6 decimals; a double compares these walks' few-decimal bounds exactly.
    fn inRange(o: std.json.ObjectMap, text: []const u8) bool {
        const bound = for ([_][]const u8{ "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum" }) |k| {
            if (o.contains(k)) break true;
        } else false;
        if (!bound) return true;
        const dot = std.mem.indexOfScalar(u8, text, '.');
        if (std.mem.findAny(u8, text, "eE") != null or (dot != null and text.len - dot.? - 1 > 6)) return false;
        const x = std.fmt.parseFloat(f64, text) catch return false;
        const at_ = struct {
            fn get(m: std.json.ObjectMap, k: []const u8) ?f64 {
                return std.fmt.parseFloat(f64, (m.get(k) orelse return null).number_string) catch null;
            }
        }.get;
        if (at_(o, "minimum")) |b| if (x < b) return false;
        if (at_(o, "maximum")) |b| if (x > b) return false;
        if (at_(o, "exclusiveMinimum")) |b| if (x <= b) return false;
        if (at_(o, "exclusiveMaximum")) |b| if (x >= b) return false;
        return true;
    }

    fn object(c: Check, o: std.json.ObjectMap, v: std.json.ObjectMap) bool {
        const props = if (o.get("properties")) |p| p.object else std.json.ObjectMap.empty;
        const required = if (o.get("required")) |r| r.array.items else &.{};
        const extra: ?Value = if (o.get("additionalProperties")) |x| (if (x == .bool and !x.bool) null else x) else null;
        var cursor: usize = 0; // properties before this one are passed
        var other = false;
        for (v.keys(), v.values()) |k, val| {
            const j = props.getIndex(k);
            if (!other and j != null and j.? >= cursor and !c.requiredIn(props, required, cursor, j.?)) {
                if (!c.valid(props.values()[j.?], val)) return false;
                cursor = j.? + 1;
            } else if (extra != null and j == null and !c.requiredIn(props, required, cursor, props.count())) {
                other = true;
                if (!c.valid(extra.?, val)) return false;
            } else return false;
        }
        return !c.requiredIn(props, required, cursor, props.count());
    }

    fn requiredIn(_: Check, props: std.json.ObjectMap, required: []const Value, from: usize, to: usize) bool {
        for (required) |r| if (props.getIndex(r.string)) |i| if (i >= from and i < to) return true;
        return false;
    }

    fn array(c: Check, o: std.json.ObjectMap, items: []const Value) bool {
        const prefix = if (o.get("prefixItems")) |p| p.array.items else &.{};
        const rest = o.get("items");
        const min: usize = if (o.get("minItems")) |m| std.fmt.parseInt(usize, m.number_string, 10) catch 0 else 0;
        const max: ?usize = if (o.get("maxItems")) |m| std.fmt.parseInt(usize, m.number_string, 10) catch null else null;
        if (items.len < min or (max != null and items.len > max.?)) return false;
        for (items, 0..) |item, i| {
            const s = if (i < prefix.len) prefix[i] else rest orelse return false;
            if (!c.valid(s, item)) return false;
        }
        return true;
    }
};

test "random walks over accepted bytes finish, and std.json and a separate validator take every reply" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5eed_c1);
    var buf: [4096]u8 = undefined;
    for (walked) |text| {
        const p = try compile(a, text);
        const top = try std.json.parseFromSliceLeaky(Value, a, text, .{ .parse_numbers = false });
        const check: Check = .{ .a = a, .top = top };
        var finished: usize = 0;
        const walks = 400;
        for (0..walks) |_| {
            const doc = walk(&p, prng.random(), &buf) orelse continue;
            finished += 1;
            const v = std.json.parseFromSliceLeaky(Value, a, doc, .{ .parse_numbers = false, .duplicate_field_behavior = .use_first }) catch |e| {
                std.debug.print("std.json refused {s}: {t}\n", .{ doc, e });
                return e;
            };
            if (!check.valid(top, v)) {
                std.debug.print("the validator refused {s}\nunder {s}\n", .{ doc, text });
                return error.TestUnexpectedResult;
            }
        }
        try std.testing.expect(finished * 10 >= walks * 8);
    }
}

test "trials equal single steps, and no state a walk reaches is a dead end" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xd00d);
    const rng = prng.random();
    const more = [_][]const u8{ "{\"type\": \"string\", \"minLength\": 1, \"maxLength\": 4}", "{\"type\": \"string\", \"format\": \"email\"}", "{}" };
    for (walked ++ more) |text| {
        const p = try compile(a, text);
        for (0..200) |_| {
            var m: Matcher = .init(&p);
            for (0..300) |_| {
                var ok: [256]bool = undefined;
                for (&ok, 0..) |*o, b| o.* = m.accepts(&.{@intCast(b)});
                const count = std.mem.count(bool, &ok, &.{true});
                if (count == 0) {
                    try std.testing.expect(m.complete());
                    break;
                }
                // a short random run: as one trial, and step by step on a copy
                var run_: [6]u8 = undefined;
                for (&run_) |*r| r.* = if (rng.boolean()) "{}[]\",:0a \\e-tu"[rng.uintLessThan(usize, 16)] else rng.int(u8);
                const len = 1 + rng.uintLessThan(usize, run_.len);
                var copy = m;
                var stepped = true;
                for (run_[0..len]) |b| stepped = stepped and copy.step(b);
                try std.testing.expectEqual(stepped, m.accepts(run_[0..len]));
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
}
