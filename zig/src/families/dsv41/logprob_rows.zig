//! A logits row's logprob row on the host, as the CUDA kernel (kernels/cuda/logprob_rows.cuh) gives one from bf16
//! logits, here from the round's fp32 rows: the log normalizer over the finite logits, the pick's logit, the k best
//! (logit descending, ties by the lower id, NaN never taken), under the target's raw distribution.
const std = @import("std");
const Row = @import("lanes").logprob.Row;

fn better(v: f32, i: usize, bv: f32, bi: usize) bool {
    return v > bv or (v == bv and i < bi);
}

/// `pick`'s row in `logits` with its `k` best tokens (k at most Row's 20).
pub fn row(logits: []const f32, pick: u32, k: u8) Row {
    std.debug.assert(k <= @import("lanes").logprob.max_top);
    var m: f32 = -std.math.inf(f32);
    for (logits) |v| if (v > m) {
        m = v;
    };
    var sum: f64 = 0;
    if (m != -std.math.inf(f32)) for (logits) |v| if (v != -std.math.inf(f32) and !std.math.isNan(v)) {
        sum += @exp(@as(f64, v) - @as(f64, m));
    };
    const lse: f32 = if (m == -std.math.inf(f32)) m else @floatCast(@as(f64, m) + @log(sum));
    const pv: f32 = if (pick < logits.len) logits[pick] else std.math.nan(f32);
    var r: Row = .{ .token = pick, .logprob = pv - lse, .count = k };
    var vals: [@import("lanes").logprob.max_top]f32 = undefined;
    var n: usize = 0;
    if (k > 0) for (logits, 0..) |v, i| {
        if (std.math.isNan(v)) continue;
        if (n == k and !better(v, i, vals[n - 1], r.ids[n - 1])) continue;
        var j: usize = if (n < k) n else n - 1;
        if (n < k) n += 1;
        while (j > 0 and better(v, i, vals[j - 1], r.ids[j - 1])) : (j -= 1) {
            vals[j] = vals[j - 1];
            r.ids[j] = r.ids[j - 1];
        }
        vals[j] = v;
        r.ids[j] = @intCast(i);
    };
    r.count = @intCast(n);
    for (vals[0..n], r.logprobs[0..n]) |v, *lp| lp.* = v - lse;
    return r;
}

const t = std.testing;

test "a row's normalizer, pick and best tokens: logit descending, ties by the lower id" {
    const logits = [_]f32{ 1.0, 3.0, 2.0, 3.0, -std.math.inf(f32) };
    const lse: f32 = @floatCast(@log(@exp(1.0) + 2 * @exp(3.0) + @exp(2.0)));
    const r = row(&logits, 2, 3);
    try t.expectEqual(@as(u32, 2), r.token);
    try t.expectApproxEqAbs(2.0 - lse, r.logprob, 1e-6);
    try t.expectEqual(@as(u8, 3), r.count);
    try t.expectEqualSlices(u32, &.{ 1, 3, 2 }, r.ids[0..3]);
    try t.expectApproxEqAbs(3.0 - lse, r.logprobs[0], 1e-6);
    try t.expectApproxEqAbs(2.0 - lse, r.logprobs[2], 1e-6);
}

test "k of 0 gives the pick alone; a NaN is never among the best; -inf adds nothing to the normalizer" {
    const logits = [_]f32{ std.math.nan(f32), 0.0, -std.math.inf(f32), 0.0 };
    const none = row(&logits, 1, 0);
    try t.expectEqual(@as(u8, 0), none.count);
    try t.expectApproxEqAbs(-@log(2.0), none.logprob, 1e-6);
    const all = row(&logits, 3, 4);
    try t.expectEqual(@as(u8, 3), all.count); // the NaN left out
    try t.expectEqualSlices(u32, &.{ 1, 3, 2 }, all.ids[0..3]);
    try t.expect(all.logprobs[2] == -std.math.inf(f32));
}

test "every row of a random vocabulary: the best k are a sorted brute force's, the logprobs exp-sum to 1" {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var logits: [4096]f32 = undefined;
    for (0..20) |trial| {
        for (&logits) |*v| v.* = @floor(rnd.floatNorm(f32) * 8) / 4; // coarse: many ties
        const k: u8 = @intCast(trial % 21);
        const pick: u32 = rnd.uintLessThan(u32, logits.len);
        const r = row(&logits, pick, k);
        var order: [logits.len]u32 = undefined;
        for (&order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, &order, &logits, struct {
            fn lt(l: *const [4096]f32, a: u32, b: u32) bool {
                return better(l[a], a, l[b], b);
            }
        }.lt);
        try t.expectEqualSlices(u32, order[0..k], r.ids[0..k]);
        // one normalizer for the pick and the best: each is its logit less the same lse
        const lse = logits[pick] - r.logprob;
        for (r.ids[0..k], r.logprobs[0..k]) |id, lp| try t.expectEqual(logits[id] - lse, lp);
        if (trial == 0) {
            var total: f64 = 0;
            for (logits) |v| total += @exp(@as(f64, v - lse));
            try t.expectApproxEqAbs(@as(f64, 1), total, 1e-4);
        }
    }
}
