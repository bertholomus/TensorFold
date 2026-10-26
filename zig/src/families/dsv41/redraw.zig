//! Structured output's redraws on rank 0: a row drawn again among the tokens its grammar allows. Rank 1 never draws.
const std = @import("std");
const lanes = @import("lanes");
const sampling = @import("sampling.zig");
const model = @import("model.zig");
const Window = @import("round.zig").Window;

/// The rows the last prompt pass or verify left for a redraw, so a stale row is an error and never a wrong draw.
pub const Rows = struct {
    /// The stream whose prompt row Model.logits[0..V] holds, from its first draw until the next verify or prompt pass.
    prompt: ?*const lanes.Stream = null,
    /// The last verify's streams in window order (none after a prompt pass or a drafter pass).
    streams: [model.max_streams]*const lanes.Stream = undefined,
    count: usize = 0,
    /// Every row of that verify was greedy: its logits stayed on the device.
    greedy: bool = false,

    /// The round row a redraw reads (null: the prompt row); error.StaleRow when those logits are gone.
    pub fn at(rows: *const Rows, wins: []const Window, r: lanes.backend.Redraw) !?usize {
        const w = r.row orelse {
            if (rows.prompt != r.stream) return error.StaleRow;
            return null;
        };
        if (w.window >= rows.count or rows.streams[w.window] != r.stream or w.row >= wins[w.window].n) return error.StaleRow;
        return wins[w.window].row + w.row;
    }
};

/// The token a row draws among the tokens `allow` takes, as Lanes.draw on a copy with the others at -inf.
pub fn draw(row: []const f32, position: u64, s: ?lanes.Sampling, allow: lanes.backend.Allow) !?u32 {
    const sm = s orelse return if (sampling.argmaxAllowed(row, allow)) |i| @intCast(i) else null;
    const t = try sampling.sampleRowAllowed(row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p }, allow) orelse return null;
    return @intCast(t);
}

test "a redraw reads the prompt row or a row of the last verify, and nothing stale" {
    const gpa = std.testing.allocator;
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &.{1}, .max_new = 4 });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &.{1}, .max_new = 4 });
    defer b.deinit(gpa);
    const wins = [_]Window{ .{ .row = 0, .n = 3, .seq = &.{} }, .{ .row = 3, .n = 2, .seq = &.{} } };
    var rows: Rows = .{ .prompt = &a };
    try std.testing.expectEqual(@as(?usize, null), try rows.at(&wins, .{ .stream = &a, .position = 1 }));
    try std.testing.expectError(error.StaleRow, rows.at(&wins, .{ .stream = &b, .position = 1 }));
    try std.testing.expectError(error.StaleRow, rows.at(&wins, .{ .stream = &a, .row = .{ .window = 0, .row = 0 }, .position = 2 }));
    rows = .{ .count = 2 };
    rows.streams[0] = &a;
    rows.streams[1] = &b;
    try std.testing.expectEqual(@as(?usize, 4), try rows.at(&wins, .{ .stream = &b, .row = .{ .window = 1, .row = 1 }, .position = 3 }));
    try std.testing.expectError(error.StaleRow, rows.at(&wins, .{ .stream = &b, .row = .{ .window = 1, .row = 2 }, .position = 4 }));
    try std.testing.expectError(error.StaleRow, rows.at(&wins, .{ .stream = &a, .row = .{ .window = 1, .row = 0 }, .position = 2 }));
    try std.testing.expectError(error.StaleRow, rows.at(&wins, .{ .stream = &a, .position = 1 }));
}
