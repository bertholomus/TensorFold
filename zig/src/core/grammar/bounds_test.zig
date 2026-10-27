//! Bounded numbers in the program matcher, against xgrammar 0.2.8's verdicts (more in the frozen corpus).
const std = @import("std");
const schema = @import("schema.zig");
const Matcher = @import("schema_match.zig").Matcher;

const Want = union(enum) { complete, prefix, reject: usize };

fn expectRun(p: *const schema.Program, want: Want, bytes: []const u8) !void {
    var m: Matcher = .init(p);
    const got: Want = for (bytes, 0..) |b, i| {
        if (!m.step(b)) break .{ .reject = i };
    } else if (m.complete()) .complete else .prefix;
    if (!std.meta.eql(want, got)) {
        std.debug.print("{s}: want {any}, got {any}\n", .{ bytes, want, got });
        return error.TestUnexpectedResult;
    }
}

fn at(i: usize) Want {
    return .{ .reject = i };
}

fn compile(a: std.mem.Allocator, text: []const u8) !schema.Program {
    var why: []const u8 = "";
    return schema.compile(a, text, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ text, why });
        return e;
    };
}

test "bounded numbers: canonical, at most 6 decimals, in range, no exponent (xgrammar 0.2.8 verdicts)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const unit = try compile(a, "{\"type\": \"number\", \"minimum\": 0, \"maximum\": 1}");
    for ([_][]const u8{ "0", "0.0", "0.85", "1", "1.0", "1.000000" }) |t| try expectRun(&unit, .complete, t);
    for ([_][]const u8{ "0.", "1." }) |t| try expectRun(&unit, .prefix, t);
    for ([_]struct { []const u8, usize }{ .{ "1.0000000", 8 }, .{ "1.000001", 7 }, .{ "1.01", 3 }, .{ "0.0000001", 8 }, .{ "-0", 0 }, .{ "-0.0", 0 }, .{ "00", 1 }, .{ "1e0", 1 }, .{ ".5", 0 }, .{ "2", 0 } }) |c|
        try expectRun(&unit, at(c[1]), c[0]);
    // a negative lower bound takes -0.0 but never -0; an exclusive bound wins a tie with an inclusive one
    const span = try compile(a, "{\"type\": \"number\", \"minimum\": -2.5, \"exclusiveMaximum\": 10}");
    for ([_][]const u8{ "-2.5", "-2.50", "-2.4", "-0.0", "-0.000000", "9.999999" }) |t| try expectRun(&span, .complete, t);
    try expectRun(&span, .prefix, "-0");
    try expectRun(&span, at(8), "-2.500001");
    try expectRun(&span, at(1), "10.0");
    try expectRun(&span, at(1), "-3");
    const tie = try compile(a, "{\"type\": \"number\", \"minimum\": 5, \"exclusiveMinimum\": 5}");
    try expectRun(&tie, .prefix, "5.0");
    try expectRun(&tie, .complete, "5.000001");
    // a bound off the 6-decimal grid moves inside it; past int64 xgrammar caps the whole part, so this takes no more
    const grid = try compile(a, "{\"type\": \"number\", \"minimum\": 0.0078125}");
    try expectRun(&grid, at(7), "0.007812");
    try expectRun(&grid, .complete, "0.007813");
    const huge = try compile(a, "{\"type\": \"number\", \"minimum\": 1e20}");
    try expectRun(&huge, .complete, "100000000000000000000.5");
    try expectRun(&huge, at(20), "100000000000000000001");
    try expectRun(&huge, at(0), "200000000000000000000");
    const capped = try compile(a, "{\"type\": \"number\", \"minimum\": 9.2e18, \"maximum\": 1e19}");
    try expectRun(&capped, .complete, "9223372036854775806");
    try expectRun(&capped, at(18), "9223372036854775807");
    try expectRun(&capped, .complete, "10000000000000000000.0");
    const ints = try compile(a, "{\"type\": \"integer\", \"minimum\": -9223372036854775808, \"maximum\": -9223372036854775000}");
    try expectRun(&ints, .complete, "-9223372036854775808");
    try expectRun(&ints, at(19), "-9223372036854775809");
    try expectRun(&ints, at(16), "-9223372036854774999");
    const percent = try compile(a, "{\"type\": \"integer\", \"minimum\": 0, \"maximum\": 100}");
    try expectRun(&percent, .complete, "100");
    for ([_]struct { []const u8, usize }{ .{ "101", 2 }, .{ "1000", 3 }, .{ "-0", 0 }, .{ "00", 1 }, .{ "1.0", 1 }, .{ "1e2", 1 } }) |c|
        try expectRun(&percent, at(c[1]), c[0]);
    // a union of ranges takes what any branch takes
    const either = try compile(a, "{\"anyOf\": [{\"type\": \"integer\", \"minimum\": 10}, {\"type\": \"number\", \"maximum\": -1.5}, {\"type\": \"null\"}]}");
    for ([_][]const u8{ "10", "-1.5", "-2", "null" }) |t| try expectRun(&either, .complete, t);
    for ([_][]const u8{ "9", "-1" }) |t| try expectRun(&either, .prefix, t);
    try expectRun(&either, at(3), "-1.4");
    try expectRun(&either, at(2), "10.5");
    const mixed = try compile(a, "{\"type\": [\"integer\", \"number\"], \"minimum\": 0, \"maximum\": 2}");
    try expectRun(&mixed, .complete, "1.5");
    try expectRun(&mixed, at(2), "2.5");
    try expectRun(&mixed, at(0), "-0.0");
}
