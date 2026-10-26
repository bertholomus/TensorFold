//! Schemas and choice lists against xgrammar's verdicts on a frozen corpus (tools/zig/xgrammar_schema_corpus.py).
const std = @import("std");
const schema = @import("schema.zig");
const Matcher = @import("schema_match.zig").Matcher;
const xgrammar_test = @import("xgrammar_test.zig");

const corpus = @embedFile("testdata/xgrammar_schema.txt");

fn verdict(buf: []u8, p: *const schema.Program, bytes: []const u8) []const u8 {
    var m: Matcher = .init(p);
    for (bytes, 0..) |b, i| if (!m.step(b)) return std.fmt.bufPrint(buf, "{d}", .{i}) catch unreachable;
    return if (m.complete()) "c" else "p";
}

/// The program a section header names: "= schema <json>" or "= choice <json list>".
fn program(a: std.mem.Allocator, header: []const u8) !schema.Program {
    var why: []const u8 = "";
    if (std.mem.startsWith(u8, header, "= schema ")) return schema.compile(a, header["= schema ".len..], &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ header, why });
        return e;
    };
    const list = try std.json.parseFromSliceLeaky([][]const u8, a, header["= choice ".len..], .{});
    return schema.choice(a, list);
}

test "every verdict equals xgrammar 0.2.8's on the frozen schema and choice corpus" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    var p: ?schema.Program = null;
    var sections: usize = 0;
    var cases: usize = 0;
    var kinds = [3]usize{ 0, 0, 0 };
    var bytes: [1024]u8 = undefined;
    var got: [16]u8 = undefined;
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        if (line[0] == '=') {
            p = try program(arena.allocator(), line);
            sections += 1;
            continue;
        }
        const space = std.mem.indexOfScalar(u8, line, ' ') orelse line.len;
        const want = line[0..space];
        const input = try xgrammar_test.decode(&bytes, if (space < line.len) line[space + 1 ..] else "");
        const mine = verdict(&got, &p.?, input);
        if (!std.mem.eql(u8, want, mine)) {
            std.debug.print("xgrammar {s}, matcher {s}: {f}\n", .{ want, mine, std.ascii.hexEscape(input, .lower) });
            return error.TestUnexpectedResult;
        }
        cases += 1;
        kinds[if (want[0] == 'c') 0 else if (want[0] == 'p') 1 else 2] += 1;
    }
    try std.testing.expectEqual(@as(usize, 54), sections);
    try std.testing.expect(cases > 4500);
    for (kinds) |k| try std.testing.expect(k >= 300);
}
