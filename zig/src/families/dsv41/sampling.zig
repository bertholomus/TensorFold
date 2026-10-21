//! The served target sampler of a decode or verify row (tensorfold.cuda.sampling.sample_rows with
//! exact_sampling.choose_rows): temperature 0 takes the row's argmax (the first of equal values). Otherwise, with top_k
//! > 0, the row's top_k + MARGIN largest logits (torch.topk's set; of equal values the lower ids) ordered by value, then
//! id (np.lexsort), the first top_k of them; each candidate's score is its logit over the temperature plus a Gumbel draw
//! keyed by (seed, position, token id) through a splitmix64 chain, so a verify row draws as the serial run at its
//! position. top_p sets the scores past the smallest prefix whose probability mass reaches top_p to -inf (the scaled
//! logits' exp over their sum: numpy's pairwise sum, then cumulative sums in order); min_p those under ln(min_p) of the
//! top. The token is the first best score. float64 as numpy computes it on the served image (numpy 2.1: its exp and log
//! are libm's, a row's sum of 20 its 8-accumulator pairwise sum; zrec np_libm.py). A top_k past max_candidates - margin
//! takes its candidates from the allocator by the same rule (numpy's sum of a longer row: below).
//! top_k 0 (top_k off: a request that sets only top_p and min_p) draws as cuda.sampling.nucleus_rows: each token's share
//! of the mass in fixed point, floor(exp(scaled - top) * 2^40), so a row's sums are exact integers; the nucleus is the
//! smallest prefix in (value, then id) order whose shares reach ceil(top_p * the row's total), min_p cuts that prefix
//! as above, and the keyed draw runs over what is left (the whole row when neither cuts). The shares' exp is libm's here
//! and CUDA's in the Python line: a last-bit difference moves a share by one unit, which changes a draw only when a
//! running sum lands within one unit of the need. sampling_cases.zig holds the Python line's draws for both
//! (tools/dsv41-zig/rec/zrec_sampling_cases.py).
const std = @import("std");

/// exact_sampling.Sampling (the served lane's defaults: temperature 1.0, top_k 20, top_p 0.95, min_p 0).
pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: usize = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,
};

/// exact_sampling.MARGIN: candidates beyond top_k read from the GPU.
pub const margin = 8;
/// The candidates a row takes on the stack (top_k + MARGIN); more come from the allocator.
pub const max_candidates = 64;
/// cuda.sampling.MASS: a token's share of the row's mass in fixed point (shares sum exactly).
pub const mass_one: f64 = 0x1p40;
/// cuda.sampling.NUCLEUS: the candidates a top_k-off row orders first; a nucleus that runs past them orders every
/// token with a share.
pub const nucleus_candidates = 1024;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

fn mix(x0: u64) u64 {
    var x = x0;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// exact_sampling.uniform: a (0, 1) double from a splitmix64 hash of (seed, position, token id).
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

/// numpy's sum of a contiguous float64 row (np.add.reduce): its 8,192-element buffers in order, each added to the total
/// as umath's pairwise_sum gives it.
pub fn pairwiseSum(a: []const f64) f64 {
    var res: f64 = 0.0;
    var i: usize = 0;
    while (i < a.len) : (i += 8192) res += pairwise(a[i..@min(a.len, i + 8192)]);
    return res;
}

/// umath's pairwise_sum: 8 accumulators up to its 128-element block, a longer run split in two at half its length less
/// that half's remainder of 8.
fn pairwise(a: []const f64) f64 {
    const n = a.len;
    if (n < 8) {
        var res: f64 = 0.0;
        for (a) |v| res += v;
        return res;
    }
    if (n > 128) {
        var n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise(a[0..n2]) + pairwise(a[n2..]);
    }
    var r: [8]f64 = a[0..8].*;
    var i: usize = 8;
    while (i < n - n % 8) : (i += 8) {
        for (0..8) |j| r[j] += a[i + j];
    }
    var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    while (i < n) : (i += 1) res += a[i];
    return res;
}

/// a before b in np.lexsort((ids, -values))'s order: the larger value, then the lower id.
fn before(va: f32, ia: i64, vb: f32, ib: i64) bool {
    return va > vb or (va == vb and ia < ib);
}

/// The row's first `count` (value, id) in that order: torch.topk's set (of equal values the lower ids) as np.lexsort
/// orders it.
pub fn topCandidates(row: []const f32, count: usize, values: []f32, ids: []i64) usize {
    const k = @min(count, row.len);
    var n: usize = 0;
    for (row, 0..) |v, i| {
        const id: i64 = @intCast(i);
        if (n == k and !before(v, id, values[k - 1], ids[k - 1])) continue;
        // insert in order, dropping the last when full
        var j: usize = if (n < k) n else k - 1;
        while (j > 0 and before(v, id, values[j - 1], ids[j - 1])) : (j -= 1) {
            values[j] = values[j - 1];
            ids[j] = ids[j - 1];
        }
        values[j] = v;
        ids[j] = id;
        if (n < k) n += 1;
    }
    return n;
}

/// The first index of the largest value (np.argmax / torch.argmax on a row without NaN).
pub fn argmax(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |v, i| {
        if (v > row[best]) best = i;
    }
    return best;
}

/// sample_rows for one row of fp32 logits at absolute position `position`: the token.
pub fn sampleRow(row: []const f32, position: u64, s: Sampling) !i64 {
    return sampleRowWith(std.heap.c_allocator, row, position, s);
}

/// sampleRow with the allocator a wide or top_k-off row takes its candidates from.
pub fn sampleRowWith(gpa: std.mem.Allocator, row: []const f32, position: u64, s: Sampling) !i64 {
    if (s.temperature <= 0) return @intCast(argmax(row));
    if (s.top_k == 0) return nucleusRow(gpa, row, position, s);
    const count = @min(row.len, s.top_k + margin);
    if (count > max_candidates) return wideRow(gpa, row, position, s, count);
    var values: [max_candidates]f32 = undefined;
    var ids: [max_candidates]i64 = undefined;
    const width = topCandidates(row, count, &values, &ids);
    var work: [3 * max_candidates]f64 = undefined;
    return chooseRow(values[0..width], ids[0..width], position, s, &work);
}

/// choose_rows on one row's candidates (already in lexsort's order): the token. `work` holds 3 * top_k float64 at least.
fn chooseRow(values: []const f32, ids: []const i64, position: u64, s: Sampling, work: []f64) i64 {
    const k = @max(1, @min(s.top_k, values.len));
    const t = @max(s.temperature, 1e-6);
    const scaled = work[0..k];
    const score = work[k..][0..k];
    for (0..k) |j| {
        scaled[j] = @as(f64, values[j]) / t;
        const u = uniform(s.seed, position, @bitCast(ids[j]));
        score[j] = scaled[j] - log(-log(u));
    }
    if (0.0 < s.top_p and s.top_p < 1.0) {
        var mx = scaled[0];
        for (scaled[1..k]) |v| mx = @max(mx, v);
        const probs = work[2 * k ..][0..k];
        for (0..k) |j| probs[j] = exp(scaled[j] - mx);
        const sum = pairwiseSum(probs);
        for (0..k) |j| probs[j] /= sum;
        var acc: f64 = 0.0;
        var below: usize = 0;
        for (0..k) |j| {
            acc += probs[j];
            if (acc < s.top_p) below += 1;
        }
        const keep = below + 1;
        for (keep..k) |j| score[j] = -std.math.inf(f64);
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + log(s.min_p);
        for (0..k) |j| {
            if (scaled[j] < floor) score[j] = -std.math.inf(f64);
        }
    }
    var best: usize = 0;
    for (0..k) |j| {
        if (score[j] > score[best]) best = j;
    }
    return ids[best];
}

/// sampleRow for a top_k whose candidates pass max_candidates: torch.topk's set (the whole row when it covers it),
/// then choose_rows.
fn wideRow(gpa: std.mem.Allocator, row: []const f32, position: u64, s: Sampling, count: usize) !i64 {
    const set = try gpa.alloc(Logit, @min(count, row.len));
    defer gpa.free(set);
    var n: usize = 0;
    if (set.len == row.len) {
        for (row, set, 0..) |v, *c, i| c.* = .{ .v = v, .id = @intCast(i) };
        n = row.len;
    } else {
        for (row, 0..) |v, i| n = Heap(Logit, logitBefore).offer(set, n, .{ .v = v, .id = @intCast(i) });
    }
    std.mem.sort(Logit, set[0..n], {}, logitBefore);
    const values = try gpa.alloc(f32, n);
    defer gpa.free(values);
    const ids = try gpa.alloc(i64, n);
    defer gpa.free(ids);
    for (set[0..n], values, ids) |c, *v, *id| {
        v.* = c.v;
        id.* = c.id;
    }
    const work = try gpa.alloc(f64, 3 * n);
    defer gpa.free(work);
    return chooseRow(values, ids, position, s, work);
}

/// A row's logit and token id, ordered as np.lexsort((ids, -values)) orders them.
const Logit = struct { v: f32, id: i64 };

fn logitBefore(_: void, a: Logit, b: Logit) bool {
    return before(a.v, a.id, b.v, b.id);
}

/// A token's scaled logit, id and share of the row's mass (cuda.sampling's float64 values and int64 mass).
const Share = struct { v: f64, id: i64, m: u64 = 0 };

fn shareBefore(_: void, a: Share, b: Share) bool {
    return a.v > b.v or (a.v == b.v and a.id < b.id);
}

/// The first `h.len` items offered, in `before_`'s order, kept as a heap whose root comes last (the first to drop).
fn Heap(comptime T: type, comptime before_: fn (void, T, T) bool) type {
    return struct {
        fn later(a: T, b: T) bool {
            return before_({}, b, a);
        }

        /// Offers `x` to the heap's first `n` items: the new count.
        fn offer(h: []T, n: usize, x: T) usize {
            if (n < h.len) {
                h[n] = x;
                var i = n;
                while (i > 0) {
                    const p = (i - 1) / 2;
                    if (!later(h[i], h[p])) break;
                    std.mem.swap(T, &h[i], &h[p]);
                    i = p;
                }
                return n + 1;
            }
            if (h.len == 0 or !before_({}, x, h[0])) return n;
            h[0] = x;
            var i: usize = 0;
            while (true) {
                const l = 2 * i + 1;
                if (l >= h.len) break;
                const c = if (l + 1 < h.len and later(h[l + 1], h[l])) l + 1 else l;
                if (!later(h[c], h[i])) break;
                std.mem.swap(T, &h[i], &h[c]);
                i = c;
            }
            return n;
        }
    };
}

/// A scaled logit's share of the mass: floor(exp(scaled - top) * 2^40); none more than 28 under the top (exp(-28) * 2^40
/// is under 0.63), at -inf or NaN.
fn share(scaled: f64, top: f64) u64 {
    const d = scaled - top;
    if (!(d >= -28.0)) return 0;
    const x = @floor(exp(d) * mass_one);
    return if (x >= 1.0) @intFromFloat(x) else 0;
}

/// ceil(p * n) for the double p's exact value (Python's math.ceil(Fraction(p) * n)), 0 < p < 1.
pub fn ceilTimes(p: f64, n: u64) u64 {
    const bits: u64 = @bitCast(p);
    const e: u64 = (bits >> 52) & 0x7ff;
    const frac = bits & ((@as(u64, 1) << 52) - 1);
    const mant: u64 = if (e == 0) frac else frac | (@as(u64, 1) << 52);
    const shift: u64 = if (e == 0) 1074 else 1075 - e; // p = mant / 2^shift
    const prod: u128 = @as(u128, mant) * n;
    if (shift >= 128) return @intFromBool(prod != 0);
    const sh: u7 = @intCast(shift);
    const rem = prod & ((@as(u128, 1) << sh) - 1);
    return @intCast((prod >> sh) + @intFromBool(rem != 0));
}

/// The nucleus's length in `order`: the tokens before the first whose running share reaches `need`, and that one; null
/// when the shares end first.
fn cutAt(order: []const Share, need: u64) ?usize {
    var acc: u64 = 0;
    for (order, 0..) |c, j| {
        acc += c.m;
        if (acc >= need) return j + 1;
    }
    return null;
}

/// cuda.sampling.nucleus_rows for one row on one rank (top_k off): the token.
fn nucleusRow(gpa: std.mem.Allocator, row: []const f32, position: u64, s: Sampling) !i64 {
    const t = @max(s.temperature, 1e-6);
    var top = -std.math.inf(f64);
    for (row) |x| top = @max(top, @as(f64, x) / t);
    // min_p's cut, ln(min_p) under the top (a min_p above 1 keeps the top token, as the top_k rule's all -inf scores do)
    const floor = if (s.min_p > 0.0) @min(top, top + log(s.min_p)) else -std.math.inf(f64);
    if (!(0.0 < s.top_p and s.top_p < 1.0)) {
        // no nucleus: the keyed draw over min_p's prefix, the whole row when min_p is off (the first best in order, as
        // nucleus_rows reads every token here)
        var best: ?Share = null;
        var best_score: f64 = 0;
        for (row, 0..) |x, i| {
            const c: Share = .{ .v = @as(f64, x) / t, .id = @intCast(i) };
            if (!(c.v >= floor)) continue;
            const score = c.v - log(-log(uniform(s.seed, position, i)));
            if (best == null or score > best_score or (score == best_score and shareBefore({}, c, best.?))) {
                best = c;
                best_score = score;
            }
        }
        return (best orelse return error.EmptyRow).id;
    }
    var total: u64 = 0;
    for (row) |x| total += share(@as(f64, x) / t, top);
    const need = ceilTimes(s.top_p, total);
    // the first nucleus_candidates in order; when the nucleus ends past them, every token with a share (one without
    // ends none: the need is at most the total)
    const first = try gpa.alloc(Share, @min(nucleus_candidates, row.len));
    defer gpa.free(first);
    var n: usize = 0;
    for (row, 0..) |x, i| n = Heap(Share, shareBefore).offer(first, n, .{ .v = @as(f64, x) / t, .id = @intCast(i) });
    std.mem.sort(Share, first[0..n], {}, shareBefore);
    for (first[0..n]) |*c| c.m = share(c.v, top);
    var order: []const Share = first[0..n];
    var keep = cutAt(order, need);
    var whole: std.ArrayList(Share) = .empty;
    defer whole.deinit(gpa);
    const covered = n == row.len or (keep != null and first[keep.? - 1].v > first[n - 1].v);
    if (!covered) {
        for (row, 0..) |x, i| {
            const c: Share = .{ .v = @as(f64, x) / t, .id = @intCast(i), .m = share(@as(f64, x) / t, top) };
            if (c.m > 0) try whole.append(gpa, c);
        }
        std.mem.sort(Share, whole.items, {}, shareBefore);
        order = whole.items;
        keep = cutAt(order, need);
    }
    var kept = keep orelse return error.EmptyRow;
    if (s.min_p > 0.0) {
        // min_p's prefix of the nucleus (the top token is always in it)
        var at: usize = 0;
        for (order[0..kept]) |c| at += @intFromBool(c.v >= floor);
        kept = @max(1, at);
    }
    var best: usize = 0;
    var best_score: f64 = 0;
    for (order[0..kept], 0..) |c, j| {
        const score = c.v - log(-log(uniform(s.seed, position, @bitCast(c.id))));
        if (j == 0 or score > best_score) {
            best = j;
            best_score = score;
        }
    }
    return order[best].id;
}

test "splitmix64 draws as exact_sampling.uniform makes them" {
    // (seed, position, id) -> the bits of exact_sampling.uniform's double
    try std.testing.expectEqual(@as(u64, 0x3fe8cd8867d6a69c), @as(u64, @bitCast(uniform(101, 2229, 2581))));
    try std.testing.expectEqual(@as(u64, 0x3fd08f785f86da1b), @as(u64, @bitCast(uniform(404, 59, 129279))));
    try std.testing.expectEqual(@as(u64, 0x3fc9ff45ea7ce2be), @as(u64, @bitCast(uniform(0, 0, 0))));
}

test "numpy's pairwise sum" {
    const a = [_]f64{ 1e16, 1.0, -1e16, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 0.5, 0.25 };
    // r = a[0..8] (no whole second block of 8); ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then the rest in order
    var want = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    for (a[8..]) |v| want += v;
    try std.testing.expectEqual(want, pairwiseSum(&a));
}

test "top candidates by value, then id" {
    const row = [_]f32{ 1, 5, 3, 5, 2, 5, 0 };
    var v: [4]f32 = undefined;
    var ids: [4]i64 = undefined;
    try std.testing.expectEqual(@as(usize, 4), topCandidates(&row, 4, &v, &ids));
    try std.testing.expectEqualSlices(i64, &.{ 1, 3, 5, 2 }, &ids);
    try std.testing.expectEqualSlices(f32, &.{ 5, 5, 5, 3 }, &v);
}

/// sampling_cases.zig's row `sh`, bit for bit as zrec_sampling_cases.py builds it.
fn caseRow(out: []f32, sh: cases.Shape) void {
    const scale: f64 = @bitCast(@as(u64, 1023 - @as(u64, sh.shift)) << 52); // 2^-shift
    const drop: u6 = @intCast(64 - @as(u7, sh.bits));
    for (out, 0..) |*v, i| {
        const q = mix(sh.gen +% @as(u64, i) *% 0x9E3779B97F4A7C15) >> drop;
        v.* = @floatCast(@as(f64, @floatFromInt(q)) * scale - sh.offset);
    }
    for (sh.peaks) |p| out[p] = @floatCast(@as(f64, out[p]) + sh.bump);
}

const cases = @import("sampling_cases.zig");

test "wide top_k and top_k off draw as the Python line's choose_rows and nucleus_rows" {
    const gpa = std.testing.allocator;
    for (cases.shapes, 0..) |sh, at| {
        const row = try gpa.alloc(f32, sh.vocab);
        defer gpa.free(row);
        caseRow(row, sh);
        for (cases.cases) |c| {
            if (c.shape != at) continue;
            const s: Sampling = .{ .seed = c.seed, .temperature = c.temperature, .top_k = c.top_k, .top_p = c.top_p, .min_p = c.min_p };
            const got = try sampleRowWith(gpa, row, c.position, s);
            if (got != c.want) {
                std.debug.print("shape {d} position {d} {any}: drew {d}, the Python line {d}\n", .{ at, c.position, s, got, c.want });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "numpy's sum past its 128-element block and its 8,192-element buffer" {
    const gpa = std.testing.allocator;
    for (cases.sums) |want| {
        const a = try gpa.alloc(f64, want.n);
        defer gpa.free(a);
        for (a, 0..) |*x, i| x.* = @as(f64, @floatFromInt(mix(99 +% @as(u64, i) *% 0x9E3779B97F4A7C15) >> 11)) * 0x1p-53;
        try std.testing.expectEqual(want.bits, @as(u64, @bitCast(pairwiseSum(a))));
    }
}

test "ceil(top_p * total) for the double's exact value" {
    for (cases.ceils) |c| try std.testing.expectEqual(c.want, ceilTimes(c.p, c.n));
}
