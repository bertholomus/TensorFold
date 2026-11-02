//! DeepSeek-V4.1's text side on the 1.0 server (M7): the reasoning efforts as the numbers its chat template takes.
//! The checkpoint's template turns "low", "high" and "max" into 50, 75 and 100 and takes any integer 1-100 as it is
//! (the "Reasoning Effort: N" line before the first turn when thinking). The served Python lane (families/deepseek_v41/
//! cuda/app.py, EFFORT_VALUES) gives every OpenAI effort name a number of its own between those; the engine reports
//! these as its Info.efforts, so the server passes each name as asked and the template hears its number. A request
//! without one hears the template's default ("high": 75), the number the lane sends; "none" turns thinking off.
//! Reasoning split ("</think>" closes the block the generation prompt opens) and DSML tool calls are the server's own.

/// A name and its number, as engine_api.Effort takes them.
pub const Effort = struct { name: []const u8, value: i64 };

pub const efforts = [_]Effort{
    .{ .name = "minimal", .value = 25 },
    .{ .name = "low", .value = 50 },
    .{ .name = "medium", .value = 62 },
    .{ .name = "high", .value = 75 },
    .{ .name = "xhigh", .value = 88 },
    .{ .name = "max", .value = 100 },
};

test "the lane's effort numbers: rising, inside the template's 1-100, its own three where it names them" {
    const std = @import("std");
    var last: i64 = 0;
    for (efforts) |e| {
        try std.testing.expect(e.value > last and e.value <= 100);
        last = e.value;
    }
    for ([_]struct { []const u8, i64 }{ .{ "low", 50 }, .{ "high", 75 }, .{ "max", 100 } }) |t| {
        const hit = for (efforts) |e| {
            if (std.mem.eql(u8, e.name, t[0])) break e.value;
        } else 0;
        try std.testing.expectEqual(t[1], hit);
    }
}
