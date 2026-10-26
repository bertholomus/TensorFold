//! The masked draws equal the plain ones on a copy of the row with every refused token at -inf, bit for bit.
const std = @import("std");
const lanes = @import("lanes");
const sampling = @import("sampling.zig");
const redraw = @import("redraw.zig");
const draw = @import("lanes.zig").draw;

/// A random set of allowed ids: dense, sparse, or one alone.
const Allowed = struct {
    bits: []const bool,
    calls: *usize,

    pub fn allows(a: Allowed, id: u32) bool {
        a.calls.* += 1;
        return a.bits[id];
    }
};

fn masked(gpa: std.mem.Allocator, row: []const f32, bits: []const bool) ![]f32 {
    const out = try gpa.alloc(f32, row.len);
    for (out, row, bits) |*o, v, ok| o.* = if (ok) v else -std.math.inf(f32);
    return out;
}

test "masked argmax, candidates and keyed draws equal the plain ones on a -inf copy, over 10,000 rows" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    var calls: usize = 0;
    var scanned: usize = 0;
    for (0..10_000) |case| {
        const width = 1 + r.uintLessThan(usize, if (case % 10 == 0) 4096 else 300);
        const row = try gpa.alloc(f32, width);
        defer gpa.free(row);
        // a few levels make equal values, whose order falls to the lower id
        const levels: u32 = if (case % 3 == 0) 6 else 1 << 20;
        for (row) |*v| v.* = @as(f32, @floatFromInt(r.uintLessThan(u32, levels))) / @as(f32, @floatFromInt(levels)) * 24.0 - 12.0;
        const bits = try gpa.alloc(bool, width);
        defer gpa.free(bits);
        const density: u32 = switch (case % 4) {
            0 => 2,
            1 => 50,
            2 => 95,
            else => 100,
        };
        for (bits) |*b| b.* = r.uintLessThan(u32, 100) < density;
        bits[r.uintLessThan(usize, width)] = true;
        const copy = try masked(gpa, row, bits);
        defer gpa.free(copy);
        const dense = case % 4 >= 2 and width >= 1000;
        var row_calls: usize = 0;
        const allow: Allowed = .{ .bits = bits, .calls = if (dense) &calls else &row_calls };
        if (dense) scanned += width;

        const best = sampling.argmax(copy);
        try std.testing.expectEqual(@as(?usize, best), sampling.argmaxAllowed(row, allow));
        // greedy: the row's own argmax when the grammar allows it, else the masked one
        const plain = sampling.argmax(row);
        try std.testing.expectEqual(best, if (bits[plain]) plain else sampling.argmaxAllowed(row, allow).?);

        const count = @min(width, 1 + r.uintLessThan(usize, sampling.max_candidates));
        var v0: [sampling.max_candidates]f32 = undefined;
        var id0: [sampling.max_candidates]i64 = undefined;
        var v1: [sampling.max_candidates]f32 = undefined;
        var id1: [sampling.max_candidates]i64 = undefined;
        const n0 = sampling.topCandidates(copy, count, &v0, &id0);
        const n1 = sampling.topCandidatesAllowed(row, count, &v1, &id1, allow);
        try std.testing.expectEqual(n0, n1);
        try std.testing.expectEqualSlices(i64, id0[0..n0], id1[0..n1]);
        for (v0[0..n0], v1[0..n1]) |a, b| try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));

        const top_ks = [_]usize{ 1, 2, 5, 20, 40, 56 };
        const top_ps = [_]f64{ 1.0, 0.95, 0.8, 0.5, 0.0 };
        const s: sampling.Sampling = .{
            .seed = r.int(u64) >> 1,
            .temperature = if (case % 7 == 0) 0 else 0.2 + 1.3 * r.float(f64),
            .top_k = top_ks[r.uintLessThan(usize, top_ks.len)],
            .top_p = top_ps[r.uintLessThan(usize, top_ps.len)],
            .min_p = if (case % 5 == 0) 0.05 else 0.0,
        };
        const position = r.uintLessThan(u64, 1 << 20);
        try std.testing.expectEqual(@as(?i64, try sampling.sampleRow(copy, position, s)), try sampling.sampleRowAllowed(row, position, s, allow));
    }
    // where most tokens are allowed, the predicate runs for a few ids a row, not for the whole row
    try std.testing.expect(scanned > 0 and calls * 4 < scanned);
}

test "a row whose grammar allows no token draws nothing" {
    const row = [_]f32{ 1, 3, 2 };
    const none = [_]bool{ false, false, false };
    var calls: usize = 0;
    const allow: Allowed = .{ .bits = &none, .calls = &calls };
    try std.testing.expectEqual(@as(?usize, null), sampling.argmaxAllowed(&row, allow));
    try std.testing.expectEqual(@as(?i64, null), try sampling.sampleRowAllowed(&row, 7, .{ .seed = 1, .top_k = 2 }, allow));
}

/// An allowed-id set behind the lane core's Allow.
const Bits = struct {
    bits: []const bool,

    fn allow(b: *const Bits) lanes.backend.Allow {
        return .{ .ctx = b, .ok = ok };
    }

    fn ok(ctx: *const anyopaque, id: u32) bool {
        const b: *const Bits = @ptrCast(@alignCast(ctx));
        return b.bits[id];
    }
};

test "a redraw draws as the served draw does on the -inf copy, greedy and sampled" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xd5a1);
    const r = prng.random();
    for (0..2_000) |case| {
        const width = 1 + r.uintLessThan(usize, 2000);
        const row = try gpa.alloc(f32, width);
        defer gpa.free(row);
        for (row) |*v| v.* = r.floatNorm(f32) * 4;
        const bits = try gpa.alloc(bool, width);
        defer gpa.free(bits);
        for (bits) |*b| b.* = r.uintLessThan(u32, 100) < 30;
        bits[r.uintLessThan(usize, width)] = true;
        const copy = try masked(gpa, row, bits);
        defer gpa.free(copy);
        const set: Bits = .{ .bits = bits };
        const s: ?lanes.Sampling = switch (case % 3) {
            0 => null,
            1 => .{ .seed = 7, .temperature = 0 },
            else => .{ .seed = r.int(u64) >> 1, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        };
        const position = r.uintLessThan(u64, 1 << 20);
        try std.testing.expectEqual(@as(?u32, try draw(copy, position, s)), try redraw.draw(row, position, s, set.allow()));
    }
}
