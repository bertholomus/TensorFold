//! The schema compiler: what it refuses (and says why, before any token), and the programs it builds.
const std = @import("std");
const schema = @import("schema.zig");
const Program = schema.Program;

fn compile(a: std.mem.Allocator, text: []const u8) !schema.Program {
    var why: []const u8 = "";
    return schema.compile(a, text, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ text, why });
        return e;
    };
}

fn refusal(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var why: []const u8 = "";
    _ = schema.compile(a, text, &why) catch |e| switch (e) {
        error.Refused => return why,
        else => return e,
    };
    std.debug.print("compiled, but should not: {s}\n", .{text});
    return error.TestUnexpectedResult;
}

test "every keyword this engine cannot enforce is refused, with where and why" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "{\"type\": \"string\", \"pattern\": \"^a+$\"}", "#: pattern: it is not enforced by this engine" },
        .{ "{\"type\": \"object\", \"properties\": {\"n\": {\"type\": \"integer\", \"minimum\": 1, \"multipleOf\": 2}}}", "#/properties/n: multipleOf: it is not enforced by this engine" },
        .{ "{\"type\": \"array\", \"items\": {\"type\": \"integer\"}, \"uniqueItems\": true}", "#: uniqueItems: it is not enforced by this engine" },
        .{ "{\"type\": \"object\", \"properties\": {\"a\": {}}, \"patternProperties\": {\"^x\": {}}}", "#: patternProperties: it is not enforced by this engine" },
        .{ "{\"type\": \"string\", \"nullable\": true}", "#: nullable: nullable is not JSON Schema: use \"type\": [..., \"null\"]" },
        .{ "{\"not\": {\"type\": \"null\"}}", "#: not: it is not enforced by this engine" },
        .{ "{\"if\": {\"type\": \"null\"}, \"then\": {}}", "#: if: it is not enforced by this engine" },
        .{ "{\"allOf\": [{\"type\": \"integer\"}, {\"type\": \"number\"}]}", "#: allOf with other than one schema is not enforced by this engine (xgrammar takes any value)" },
        .{ "{\"anyOf\": [{\"type\": \"string\"}, {\"type\": \"string\", \"format\": \"date\"}]}", "#: two anyOf, oneOf or type branches can start with '\"': this engine needs branches that start with different bytes" },
        .{ "{\"anyOf\": [{\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}}}, {\"type\": \"object\", \"properties\": {\"number\": {\"type\": \"integer\"}}}]}", "#: two anyOf, oneOf or type branches can start with '{': this engine needs branches that start with different bytes" },
        .{ "{\"oneOf\": [{}, {\"type\": \"null\"}]}", "#: anyOf has a branch that takes any value beside others: this engine needs branches that start with different bytes" },
        .{ "{\"anyOf\": []}", "#: anyOf must be a non-empty list" },
        .{ "{\"type\": \"object\", \"properties\": {\"a\": {\"$ref\": \"https://example.com/a.json\"}}}", "#/properties/a: $ref https://example.com/a.json is not local: this engine follows only # and #/..." },
        .{ "{\"type\": \"object\", \"properties\": {\"a\": {\"$ref\": \"#/$defs/missing\"}}}", "#/properties/a: $ref #/$defs/missing does not resolve" },
        .{ "{\"$defs\": {\"a\": {\"$ref\": \"#/$defs/b\"}, \"b\": {\"$ref\": \"#/$defs/a\"}}, \"$ref\": \"#/$defs/a\"}", "#: a $ref refers only to itself" },
        .{ "{\"anyOf\": [{\"$ref\": \"#\"}, {\"type\": \"null\"}]}", "#: anyOf contains itself" },
        .{ "{\"type\": \"object\", \"properties\": {\"a\": {}}, \"required\": [\"b\"]}", "#: required names b, which properties does not define" },
        .{ "{\"type\": \"object\", \"properties\": {\"caf\u{e9}\": {}}, \"additionalProperties\": true}", "#: beside additionalProperties a property name must be printable ASCII without a quote or backslash: caf\u{e9}" },
        .{ "{\"enum\": [\"a\", 0.1]}", "#: enum and const take no number with more than 17 significant digits: xgrammar writes it rounded" },
        .{ "{\"const\": 1e400}", "#: enum and const take no number with more than 17 significant digits: xgrammar writes it rounded" },
        .{ "{\"enum\": []}", "#: enum must be a non-empty list" },
        .{ "{\"type\": \"string\", \"format\": \"ipv6\"}", "#: format ipv6 is not enforced by this engine" },
        .{ "{\"type\": \"string\", \"format\": \"date\", \"maxLength\": 10}", "#: format date with minLength or maxLength is not enforced: xgrammar drops the lengths" },
        .{ "{\"type\": \"string\", \"minLength\": 4, \"maxLength\": 3}", "#: minLength 4 is greater than maxLength 3" },
        .{ "{\"type\": \"string\", \"minLength\": 1.5}", "#: minLength must be an integer" },
        .{ "{\"minLength\": 3}", "#: minLength needs a type here: without one xgrammar takes any value" },
        .{ "{\"required\": [\"a\"]}", "#: required needs a type here: without one xgrammar takes any value" },
        .{ "{\"type\": \"array\", \"items\": {}, \"minItems\": 3, \"maxItems\": 1}", "#: minItems 3 is greater than maxItems 1" },
        .{ "{\"type\": \"array\", \"prefixItems\": [{}, {}], \"maxItems\": 1}", "#: maxItems 1 is less than the 2 prefixItems" },
        .{ "{\"type\": \"array\", \"minItems\": 1}", "#: minItems 1 is more than the 0 prefixItems, and no other item is allowed" },
        .{ "{\"type\": \"array\", \"items\": {}, \"maxItems\": 70000}", "#: minItems and maxItems above 65534 are too large for this engine" },
        .{ "{\"type\": \"array\", \"items\": [{}]}", "#: items must be a boolean or a schema object" },
        .{ "{\"type\": \"array\", \"prefixItems\": [true]}", "#: prefixItems must be a list of schema objects" },
        .{ "{\"type\": \"decimal\"}", "#: type decimal is not a JSON type" },
        .{ "{\"type\": 5}", "#: type must be a string or a list of strings" },
        .{ "{\"type\": \"object\", \"properties\": {\"a\": false}}", "#/properties/a: the schema false accepts no value" },
        .{ "[1]", "#: a schema must be an object or a boolean" },
        .{ "{\"a\": 1, \"a\": 2}", "#: the schema repeats a key" },
        .{ "{\"type\": \"object\"", "#: the schema is not JSON this engine reads (UnexpectedEndOfInput)" },
    };
    for (cases) |c| {
        const why = try refusal(a, c[0]);
        std.testing.expectEqualStrings(c[1], why) catch |e| {
            std.debug.print("for {s}\n", .{c[0]});
            return e;
        };
    }
}

test "a bound is refused where xgrammar ignores it or fails, and for a range no 6-decimal number lies in" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const needs = " needs type integer or number, without $ref, const, enum, anyOf, oneOf or allOf: xgrammar ignores it elsewhere";
    const cases = [_][2][]const u8{
        .{ "{\"minimum\": 1}", "#: minimum" ++ needs },
        .{ "{\"type\": \"string\", \"exclusiveMinimum\": 1}", "#: exclusiveMinimum" ++ needs },
        .{ "{\"type\": [], \"maximum\": 1}", "#: maximum" ++ needs },
        .{ "{\"type\": \"object\", \"properties\": {\"n\": {\"enum\": [1, 5], \"maximum\": 3}}}", "#/properties/n: maximum" ++ needs },
        .{ "{\"anyOf\": [{\"type\": \"integer\"}], \"minimum\": 0}", "#: minimum" ++ needs },
        .{ "{\"$ref\": \"#/$defs/n\", \"exclusiveMaximum\": 0, \"$defs\": {\"n\": {\"type\": \"number\"}}}", "#: exclusiveMaximum" ++ needs },
        .{ "{\"type\": \"integer\", \"minimum\": 1.5}", "#: minimum must be a whole number for type integer" },
        .{ "{\"type\": [\"integer\", \"number\"], \"minimum\": 0.5}", "#: minimum must be a whole number for type integer" },
        .{ "{\"type\": \"integer\", \"maximum\": 9223372036854775808}", "#: maximum must lie between -2^63 and 2^63 for type integer" },
        .{ "{\"type\": \"integer\", \"minimum\": -9.223372036854776e18}", "#: minimum must lie between -2^63 and 2^63 for type integer" },
        .{ "{\"type\": \"integer\", \"exclusiveMinimum\": 9223372036854775807}", "#: exclusiveMinimum leaves no int64" },
        .{ "{\"type\": \"integer\", \"exclusiveMaximum\": -9223372036854775808}", "#: exclusiveMaximum leaves no int64" },
        .{ "{\"type\": \"integer\", \"minimum\": 5, \"maximum\": 3}", "#: the range is empty: its lower bound is above its upper one" },
        .{ "{\"type\": \"integer\", \"exclusiveMinimum\": 2, \"exclusiveMaximum\": 3}", "#: the range is empty: its lower bound is above its upper one" },
        .{ "{\"type\": \"number\", \"minimum\": 5, \"exclusiveMaximum\": 5}", "#: the range is empty: its lower bound is not below its upper one" },
        .{ "{\"type\": \"number\", \"minimum\": 1e-7, \"maximum\": 2e-7}", "#: no number with at most 6 decimals lies in the range: xgrammar 0.2.8 takes an empty value there" },
        .{ "{\"type\": \"number\", \"exclusiveMinimum\": 0.5, \"maximum\": 0.5000001}", "#: no number with at most 6 decimals lies in the range: xgrammar 0.2.8 takes an empty value there" },
        .{ "{\"type\": \"number\", \"maximum\": 1e400}", "#: maximum is past the largest double" },
        .{ "{\"type\": \"number\", \"minimum\": \"0\"}", "#: minimum must be a number" },
        .{ "{\"type\": \"number\", \"exclusiveMinimum\": true}", "#: exclusiveMinimum must be a number" },
        .{ "{\"type\": \"number\", \"minimum\": 0, \"multipleOf\": 0.5}", "#: multipleOf: it is not enforced by this engine" },
    };
    for (cases) |c| {
        const why = try refusal(a, c[0]);
        std.testing.expectEqualStrings(c[1], why) catch |e| {
            std.debug.print("for {s}\n", .{c[0]});
            return e;
        };
    }
    // the same bounds compile where xgrammar applies them, also beside other types, which they leave alone
    for ([_][]const u8{
        "{\"type\": \"integer\", \"minimum\": 1.0, \"maximum\": 1e2}",
        "{\"type\": \"integer\", \"exclusiveMinimum\": -9223372036854775807, \"maximum\": 9223372036854775807}",
        "{\"type\": [\"number\", \"null\"], \"exclusiveMinimum\": -1, \"maximum\": 1}",
        "{\"type\": [\"integer\", \"string\"], \"minimum\": 3}",
        "{\"type\": \"number\", \"minimum\": -1e300, \"maximum\": 1e300}",
        "{\"type\": \"number\", \"maximum\": 1e-9}",
    }) |t| _ = try compile(a, t);
}

test "equal bounds share one automaton, and a schema whose bounds need too many states is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "{\"type\": \"object\", \"properties\": {");
    for (0..1000) |i| try text.print(a, "{s}\"p{d}\": {{\"type\": \"number\", \"minimum\": 0.0, \"maximum\": 1.0}}", .{ if (i == 0) "" else ", ", i });
    try text.appendSlice(a, "}}");
    const one = try compile(a, "{\"type\": \"number\", \"minimum\": 0.0, \"maximum\": 1.0}");
    try std.testing.expectEqual(one.states.len, (try compile(a, text.items)).states.len);
    // -1e306 to 1e306 needs ~800 states (whole parts of up to 307 digits), so 100 different such ranges pass the cap
    text.clearRetainingCapacity();
    try text.appendSlice(a, "{\"type\": \"object\", \"properties\": {");
    for (0..100) |i| try text.print(a, "{s}\"p{d}\": {{\"type\": \"number\", \"minimum\": -1.{d}e306, \"maximum\": 1.{d}e306}}", .{ if (i == 0) "" else ", ", i, i, i });
    try text.appendSlice(a, "}}");
    const why = try refusal(a, text.items);
    try std.testing.expect(std.mem.startsWith(u8, why, "#/properties/p") and std.mem.endsWith(u8, why, ": the numeric bounds need more than 65536 automaton states"));
}

test "a pydantic schema with a bounded confidence (ge=0.0, le=1.0) compiles, as xgrammar 0.2.8 took it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try compile(arena.allocator(), @embedFile("testdata/pydantic_bounds.json"));
    const o = p.objects[p.nodes[p.root].object];
    try std.testing.expectEqualStrings("\"confidence\"", o.props[1].key);
    try std.testing.expect(p.nodes[o.props[1].node] == .range);
    try std.testing.expectEqualSlices(u16, &.{ 0, 1, 2, 7, 7, 7, 7, 7 }, o.required_from);
}

test "a schema nested past 128 levels is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ schema.max_nesting - 1, schema.max_nesting }) |wraps| {
        var text: std.ArrayList(u8) = .empty;
        for (0..wraps) |_| try text.appendSlice(a, "{\"items\": ");
        try text.appendSlice(a, "{}"); // the 128th or the 129th schema
        for (0..wraps) |_| try text.append(a, '}');
        if (wraps < schema.max_nesting) {
            _ = try compile(a, text.items);
        } else try std.testing.expect(std.mem.endsWith(u8, try refusal(a, text.items), "the schema nests deeper than 128 levels"));
    }
}

test "a schema text past 1 MiB is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try a.alloc(u8, schema.max_text + 1);
    @memset(text, ' ');
    @memcpy(text[0..2], "{}");
    try std.testing.expectEqualStrings("#: the schema is longer than 1048576 bytes", try refusal(a, text));
    _ = try compile(a, text[0..schema.max_text]);
}

test "a trade signal schema: three required properties in order, an enum, no other key" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try compile(arena.allocator(),
        \\{"type": "object", "additionalProperties": false, "required": ["action", "confidence", "pair"],
        \\ "properties": {"action": {"type": "string", "enum": ["buy", "sell", "hold"]}, "confidence": {"type": "number"},
        \\ "pair": {"type": "string"}}}
    );
    const o = p.objects[p.nodes[p.root].object];
    try std.testing.expectEqual(@as(usize, 3), o.props.len);
    try std.testing.expectEqual(schema.none, o.additional);
    try std.testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3 }, o.required_from);
    try std.testing.expectEqualStrings("\"action\"", o.props[0].key);
    try std.testing.expect(p.nodes[o.props[0].node] == .literal);
    try std.testing.expect(p.nodes[o.props[1].node] == .number);
    try std.testing.expect(p.nodes[o.props[2].node] == .string);
}

test "plain object and array types take any keys and items; one more keyword makes them strict, as in xgrammar" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "{\"type\": \"object\"}", "{\"type\": \"object\", \"title\": \"t\", \"description\": \"d\"}" }) |t|
        try std.testing.expectEqual(Program.any_object, (try compile(a, t)).root);
    try std.testing.expectEqual(Program.any_array, (try compile(a, "{\"type\": \"array\", \"examples\": [[1]]}")).root);
    try std.testing.expectEqual(Program.any, (try compile(a, "{}")).root);
    try std.testing.expectEqual(Program.any, (try compile(a, "true")).root);
    for ([_][]const u8{ "{\"type\": \"object\", \"$id\": \"x\"}", "{\"type\": \"object\", \"x-vendor\": 1}", "{\"type\": \"object\", \"required\": []}" }) |t| {
        const p = try compile(a, t);
        const o = p.objects[p.nodes[p.root].object];
        try std.testing.expect(o.props.len == 0 and o.additional == schema.none);
    }
}

test "a union keeps one branch for each first byte: integer folds into number, enums and booleans merge" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try compile(arena.allocator(),
        \\{"anyOf": [{"type": ["integer", "number"]}, {"enum": ["a", "b"]}, {"const": "c"}, {"type": "boolean"},
        \\ {"type": "null"}, {"type": "array", "items": {"type": "string"}}, {"$ref": "#/$defs/o"}],
        \\ "$defs": {"o": {"type": "object", "properties": {"k": {"type": "integer"}}}}}
    );
    const u = p.unions[p.nodes[p.root].any_of].dispatch;
    for ("-0123456789") |b| try std.testing.expect(p.nodes[u[b]] == .number);
    try std.testing.expect(u['"'] == u['t'] and u['t'] == u['f'] and u['f'] == u['n'] and p.nodes[u['n']] == .literal);
    try std.testing.expect(p.nodes[u['[']] == .array and p.nodes[u['{']] == .object);
    try std.testing.expectEqual(schema.none, u['x']);
}

test "guided_choice is a trie of the raw strings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try schema.choice(arena.allocator(), &.{ "ab", "a", "a\"\n" });
    const root = p.nodes[p.root].literal;
    const t = p.child(root, 'a').?;
    try std.testing.expect(p.tries[t].end and p.child(t, 'b') != null and p.child(t, '"') != null);
    try std.testing.expect(p.child(root, 'b') == null);
}

test "a repeated type or branch adds no branch, as xgrammar takes it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        "{\"type\": [\"integer\", \"integer\"]}",
        "{\"type\": [\"string\", \"integer\", \"string\"], \"minLength\": 2}",
        "{\"type\": [\"object\", \"object\"], \"properties\": {\"a\": {\"type\": \"integer\"}}}",
        "{\"anyOf\": [{}, {}]}",
        "{\"anyOf\": [{\"type\": \"string\", \"format\": \"date\"}, {\"type\": \"string\", \"format\": \"date\"}]}",
        "{\"anyOf\": [{\"$ref\": \"#/$defs/p\"}, {\"$ref\": \"#/$defs/p\"}], \"$defs\": {\"p\": {\"type\": \"object\"}}}",
    }) |t| _ = try compile(a, t);
    const p = try compile(a, "{\"type\": [\"integer\", \"null\", \"integer\"]}");
    const u = p.unions[p.nodes[p.root].any_of].dispatch;
    try std.testing.expect(p.nodes[u['1']] == .integer and p.nodes[u['n']] == .literal and u['"'] == schema.none);
    // two objects stay two branches that start alike
    try std.testing.expectEqualStrings(
        "#: two anyOf, oneOf or type branches can start with '{': this engine needs branches that start with different bytes",
        try refusal(a, "{\"anyOf\": [{\"type\": \"object\", \"properties\": {\"a\": {}}}, {\"type\": \"object\", \"properties\": {\"a\": {}}}]}"),
    );
}

test "a union folds bounded numbers into number, unites them without it, and dispatches only the bytes they start" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const folded = try compile(a, "{\"anyOf\": [{\"type\": \"number\"}, {\"type\": \"integer\", \"minimum\": 3}]}");
    try std.testing.expect(folded.nodes[folded.unions[folded.nodes[folded.root].any_of].dispatch['-']] == .number);
    const united = try compile(a, "{\"type\": [\"integer\", \"number\"], \"minimum\": 0, \"maximum\": 2}");
    const u = united.unions[united.nodes[united.root].any_of].dispatch;
    try std.testing.expect(united.nodes[u['0']] == .range and u['1'] == u['0'] and u['2'] == u['0']);
    try std.testing.expect(u['3'] == schema.none and u['-'] == schema.none); // no reply starts so
    // a range that starts no '-' leaves it to an enum: a plain integer would not
    const lits = try compile(a, "{\"anyOf\": [{\"type\": \"integer\", \"minimum\": 0}, {\"enum\": [-1, \"x\"]}]}");
    const l = lits.unions[lits.nodes[lits.root].any_of].dispatch;
    try std.testing.expect(lits.nodes[l['-']] == .literal and lits.nodes[l['7']] == .range);
    try std.testing.expectEqualStrings(
        "#: two anyOf, oneOf or type branches can start with '-': this engine needs branches that start with different bytes",
        try refusal(a, "{\"anyOf\": [{\"type\": \"integer\"}, {\"enum\": [-1]}]}"),
    );
}
