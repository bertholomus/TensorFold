//! The matcher against xgrammar's own verdicts on a frozen corpus (tools/zig/xgrammar_json_corpus.py writes it).
const std = @import("std");
const json = @import("json.zig");

const corpus = @embedFile("testdata/xgrammar_json_object.txt");

/// A corpus line's bytes (none when the line has no space): `_` is a space, %XX a byte, any other character itself.
fn decode(buf: []u8, text: []const u8) ![]u8 {
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len) : (n += 1) {
        if (n == buf.len) return error.LineTooLong;
        switch (text[i]) {
            '%' => {
                if (i + 3 > text.len) return error.BadEscape;
                buf[n] = try std.fmt.parseInt(u8, text[i + 1 .. i + 3], 16);
                i += 3;
            },
            '_' => {
                buf[n] = ' ';
                i += 1;
            },
            else => |c| {
                buf[n] = c;
                i += 1;
            },
        }
    }
    return buf[0..n];
}

/// xgrammar's verdict form: the first rejected byte's index, else c (complete) or p (an unfinished prefix).
fn verdict(buf: []u8, bytes: []const u8) []const u8 {
    var m: json.Matcher = .{};
    for (bytes, 0..) |b, i| if (!m.step(b)) return std.fmt.bufPrint(buf, "{d}", .{i}) catch unreachable;
    return if (m.complete()) "c" else "p";
}

test "every verdict equals xgrammar 0.2.8's on the frozen corpus" {
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    var cases: usize = 0;
    var kinds = [3]usize{ 0, 0, 0 };
    var bytes: [512]u8 = undefined;
    var got: [16]u8 = undefined;
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        const space = std.mem.indexOfScalar(u8, line, ' ') orelse line.len;
        const want = line[0..space];
        const input = try decode(&bytes, if (space < line.len) line[space + 1 ..] else "");
        const mine = verdict(&got, input);
        if (!std.mem.eql(u8, want, mine)) {
            std.debug.print("xgrammar {s}, matcher {s}: {f}\n", .{ want, mine, std.ascii.hexEscape(input, .lower) });
            return error.TestUnexpectedResult;
        }
        cases += 1;
        kinds[if (want[0] == 'c') 0 else if (want[0] == 'p') 1 else 2] += 1;
    }
    try std.testing.expectEqual(@as(usize, 3000), cases);
    for (kinds) |k| try std.testing.expect(k >= 100);
}
