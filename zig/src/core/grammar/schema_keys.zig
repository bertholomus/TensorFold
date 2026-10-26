//! A schema's keywords as this engine treats them, and the bytes xgrammar matches for a key, an enum value or a const.
const std = @import("std");
const Value = std.json.Value;
const Allocator = std.mem.Allocator;

/// Keys xgrammar 0.2.8 drops before it compares a schema with {"type": "object"} and the other plain types.
const annotations = [_][]const u8{ "title", "default", "description", "examples", "deprecated", "readOnly", "writeOnly", "$comment", "$schema" };

/// Keywords that constrain a reply and that this engine does not enforce (xgrammar enforces or ignores them): refused.
const refused = [_][]const u8{
    "pattern",           "patternProperties", "propertyNames",         "minimum",          "maximum",
    "exclusiveMinimum",  "exclusiveMaximum",  "multipleOf",            "minProperties",    "maxProperties",
    "dependentRequired", "dependentSchemas",  "dependencies",          "if",               "then",
    "else",              "not",               "unevaluatedProperties", "unevaluatedItems", "contains",
    "minContains",       "maxContains",       "uniqueItems",           "additionalItems",  "$dynamicRef",
    "$dynamicAnchor",    "$recursiveRef",     "$recursiveAnchor",
};

/// The first keyword of `schema` that this engine refuses, with why.
pub fn refusal(schema: std.json.ObjectMap) ?[]const u8 {
    for (schema.keys()) |k| {
        for (refused) |r| if (std.mem.eql(u8, k, r)) return k;
        if (std.mem.eql(u8, k, "nullable")) return k;
    }
    return null;
}

/// What to say about a refused keyword.
pub fn advice(keyword: []const u8) []const u8 {
    if (std.mem.eql(u8, keyword, "nullable")) return "nullable is not JSON Schema: use \"type\": [..., \"null\"]";
    return "it is not enforced by this engine";
}

/// Whether `key` is an annotation that xgrammar drops before it compares a schema with the plain types.
pub fn annotation(key: []const u8) bool {
    for (annotations) |n| if (std.mem.eql(u8, key, n)) return true;
    return false;
}

/// `schema` has only annotations and "type": xgrammar then takes any keys or items, not a strict object or array.
pub fn plain(schema: std.json.ObjectMap) bool {
    for (schema.keys()) |k| if (!annotation(k) and !std.mem.eql(u8, k, "type")) return false;
    return true;
}

/// A number past 17 significant digits, which xgrammar writes rounded ("%.17g"): enum and const do not take it.
pub const Unwritable = error{Unwritable};

/// The integer a number's text gives, as picojson parses it: integer digits that fit in 64 bits.
pub fn int64(v: Value) ?i64 {
    const text = switch (v) {
        .number_string => |t| t,
        .integer => |i| return i,
        else => return null,
    };
    if (std.mem.findAny(u8, text, ".eE") != null) return null;
    return std.fmt.parseInt(i64, text, 10) catch null;
}

/// Appends `v` as picojson writes an enum or const for xgrammar: compact, int64 in decimal, else "%.f" or "%.17g".
pub fn write(a: Allocator, out: *std.ArrayList(u8), v: Value) (Allocator.Error || Unwritable)!void {
    switch (v) {
        .null => try out.appendSlice(a, "null"),
        .bool => |b| try out.appendSlice(a, if (b) "true" else "false"),
        .integer, .float, .number_string => try writeNumber(a, out, v),
        .string => |s| try writeString(a, out, s),
        .array => |items| {
            try out.append(a, '[');
            for (items.items, 0..) |item, i| {
                if (i > 0) try out.append(a, ',');
                try write(a, out, item);
            }
            try out.append(a, ']');
        },
        .object => |o| {
            try out.append(a, '{');
            for (o.keys(), o.values(), 0..) |k, item, i| {
                if (i > 0) try out.append(a, ',');
                try writeString(a, out, k);
                try out.append(a, ':');
                try write(a, out, item);
            }
            try out.append(a, '}');
        },
    }
}

fn writeNumber(a: Allocator, out: *std.ArrayList(u8), v: Value) (Allocator.Error || Unwritable)!void {
    if (int64(v)) |i| return out.print(a, "{d}", .{i});
    const x: f64 = switch (v) {
        .float => |f| f,
        .number_string => |t| std.fmt.parseFloat(f64, t) catch return error.Unwritable,
        else => unreachable,
    };
    if (!std.math.isFinite(x)) return error.Unwritable;
    if (std.math.signbit(x)) try out.append(a, '-');
    var buf: [40]u8 = undefined;
    const digits, const exp = try decimal(&buf, @abs(x));
    if (exp < -4 or exp >= 17) {
        // %e: d.ddd, then the exponent with a sign and at least two digits
        try out.append(a, digits[0]);
        if (digits.len > 1) try out.print(a, ".{s}", .{digits[1..]});
        return out.print(a, "e{c}{d:0>2}", .{ @as(u8, if (exp < 0) '-' else '+'), @abs(exp) });
    }
    if (exp < 0) {
        try out.appendSlice(a, "0.");
        try out.appendNTimes(a, '0', @intCast(-exp - 1));
        return out.appendSlice(a, digits);
    }
    const whole: usize = @intCast(exp + 1);
    if (digits.len <= whole) {
        try out.appendSlice(a, digits);
        return out.appendNTimes(a, '0', whole - digits.len);
    }
    try out.print(a, "{s}.{s}", .{ digits[0..whole], digits[whole..] });
}

/// The exact digits of `x` (finite, positive or 0) without trailing zeros and the first one's power of ten; 17 at most.
fn decimal(buf: *[40]u8, x: f64) Unwritable!struct { []const u8, i32 } {
    if (x == 0) return .{ "0", 0 };
    const bits: u64 = @bitCast(x);
    const biased: i32 = @intCast(bits >> 52);
    var m: u64 = bits & ((1 << 52) - 1);
    var e: i32 = biased - 1075;
    if (biased == 0) e = -1074 else m |= 1 << 52;
    const zeros = @ctz(m);
    m >>= @intCast(zeros);
    e += @intCast(zeros);
    // x = m * 2^e = n * 10^point exactly, when n fits in 128 bits
    var n: u128 = m;
    var point: i32 = 0;
    if (e >= 0) {
        if (e >= 128 - 53) return error.Unwritable;
        n = @as(u128, m) << @intCast(e);
    } else {
        for (0..@intCast(-e)) |_| n = std.math.mul(u128, n, 5) catch return error.Unwritable;
        point = e;
    }
    while (n % 10 == 0) {
        n /= 10;
        point += 1;
    }
    const digits = std.fmt.bufPrint(buf, "{d}", .{n}) catch unreachable;
    if (digits.len > 17) return error.Unwritable;
    return .{ digits, point + @as(i32, @intCast(digits.len)) - 1 };
}

/// Appends `s` as picojson writes a string: \" \\ \b \f \n \r \t, \u00XX for other control bytes and DEL, else raw.
pub fn writeString(a: Allocator, out: *std.ArrayList(u8), s: []const u8) Allocator.Error!void {
    try out.append(a, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(a, "\\\""),
        '\\' => try out.appendSlice(a, "\\\\"),
        0x08 => try out.appendSlice(a, "\\b"),
        0x0C => try out.appendSlice(a, "\\f"),
        '\n' => try out.appendSlice(a, "\\n"),
        '\r' => try out.appendSlice(a, "\\r"),
        '\t' => try out.appendSlice(a, "\\t"),
        else => if (c < 0x20 or c == 0x7F) try out.print(a, "\\u{x:0>4}", .{c}) else try out.append(a, c),
    };
    try out.append(a, '"');
}

/// A name spelled the same raw and in JSON; xgrammar keeps other keys from names by raw bytes, so it needs this.
pub fn bare(key: []const u8) bool {
    for (key) |c| if (c < 0x20 or c > 0x7E or c == '"' or c == '\\') return false;
    return true;
}

fn written(v: []const u8) ![]u8 {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Value, a, v, .{ .parse_numbers = false });
    defer parsed.deinit();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try write(a, &out, parsed.value);
    return out.toOwnedSlice(a);
}

test "enum and const values are written as xgrammar 0.2.8 wrote them" {
    const a = std.testing.allocator;
    // from xgrammar's own grammar for each value (tools/zig/xgrammar_schema_corpus.py checks the same)
    const cases = [_][2][]const u8{
        .{ "{\"z\": 1, \"a\": [1, 2.50, true, null]}", "{\"z\":1,\"a\":[1,2.5,true,null]}" },
        .{ "\"a/b\\u007f\\b\\f\\r\\t\\u0001\\\"\\\\\"", "\"a/b\\u007f\\b\\f\\r\\t\\u0001\\\"\\\\\"" },
        .{ "\"\u{e9}\u{2028}\u{1f600}\"", "\"\u{e9}\u{2028}\u{1f600}\"" },
        .{ "9007199254740993", "9007199254740993" },
        .{ "-0", "0" },
        .{ "-0.0", "-0" },
        .{ "1.0", "1" },
        .{ "3.0e2", "300" },
        .{ "-9223372036854775808", "-9223372036854775808" },
        .{ "[0.5, -2.25, 0.0001220703125, 0.00006103515625, 1e20, 9007199254740992.0, 1e16, 123456.75]", "[0.5,-2.25,0.0001220703125,6.103515625e-05,1e+20,9007199254740992,10000000000000000,123456.75]" },
    };
    for (cases) |c| {
        const got = try written(c[0]);
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
    for ([_][]const u8{ "0.1", "1e100", "9223372036854775808", "1e-7", "[1, 0.3]", "1e400" }) |v|
        try std.testing.expectError(error.Unwritable, written(v));
}

test "keywords: refused ones, annotations and the plain-type rule" {
    const a = std.testing.allocator;
    const cases = [_]struct { []const u8, ?[]const u8, bool }{
        .{ "{\"type\": \"object\", \"title\": \"t\", \"description\": \"d\"}", null, true },
        .{ "{\"type\": \"object\", \"$id\": \"x\"}", null, false },
        .{ "{\"type\": \"string\", \"pattern\": \"a\"}", "pattern", false },
        .{ "{\"type\": \"string\", \"nullable\": true}", "nullable", false },
        .{ "{\"type\": \"array\", \"uniqueItems\": true}", "uniqueItems", false },
        .{ "{\"x-vendor\": 1}", null, false },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(Value, a, c[0], .{});
        defer parsed.deinit();
        const got = refusal(parsed.value.object);
        if (c[1]) |k| try std.testing.expectEqualStrings(k, got.?) else try std.testing.expectEqual(@as(?[]const u8, null), got);
        try std.testing.expectEqual(c[2], plain(parsed.value.object));
    }
    try std.testing.expect(bare("a_b-1 ~") and !bare("a\"b") and !bare("\u{e9}") and !bare("a\\b") and !bare("a\x7f"));
}
