//! Engram's host side: the n-gram row ids of each position (DeepSeek's engram.py rule, as the served Engram.hashes
//! computes it with numpy) and the FP8 row decode. Rows themselves are read by the I/O module.
const std = @import("std");
const Config = @import("config.zig").Config;

pub const max_layers = 2;
pub const max_ngram = 4;
pub const max_heads = 8;
pub const max_cols = (max_ngram - 1) * max_heads;

/// Miller-Rabin with the bases ops._is_prime uses (deterministic far past the bucket sizes).
pub fn isPrime(n: u64) bool {
    if (n < 2) return false;
    const bases = [_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    for (bases) |p| if (n % p == 0) return n == p;
    var d = n - 1;
    var r: u32 = 0;
    while (d % 2 == 0) : (r += 1) d /= 2;
    outer: for (bases) |a| {
        var x = powMod(a, d, n);
        if (x == 1 or x == n - 1) continue;
        for (0..r - 1) |_| {
            x = mulMod(x, x, n);
            if (x == n - 1) continue :outer;
        }
        return false;
    }
    return true;
}

fn mulMod(a: u64, b: u64, m: u64) u64 {
    return @intCast(@as(u128, a) * b % m);
}

fn powMod(base: u64, e: u64, m: u64) u64 {
    var r: u64 = 1;
    var b = base % m;
    var k = e;
    while (k > 0) : (k >>= 1) {
        if (k & 1 == 1) r = mulMod(r, b, m);
        b = mulMod(b, b, m);
    }
    return r;
}

/// The bucket primes [layer][(ngram - 1) * heads] (ops.engram_primes): for every layer and order, `heads` primes drawn
/// upward from engram_vocab, none reused across the whole table.
pub fn primes(c: Config, out: *[max_layers][max_cols]i64) void {
    var used: [max_layers * max_cols]u64 = undefined;
    var n_used: usize = 0;
    for (0..c.engram_layers.len) |l| {
        for (0..c.engram_ngram - 1) |o| {
            var cur: u64 = c.engram_vocab - 1;
            for (0..c.engram_heads) |h| {
                cur += 1;
                while (!isPrime(cur) or std.mem.indexOfScalar(u64, used[0..n_used], cur) != null) cur += 1;
                used[n_used] = cur;
                n_used += 1;
                out[l][o * c.engram_heads + h] = @intCast(cur);
            }
        }
    }
}

/// Row ids of positions [start, start + n) of a sequence, as the served engine reads Engram tables.
pub const Hasher = struct {
    layers: usize,
    ngram: usize,
    heads: usize,
    primes: [max_layers][max_cols]i64 = undefined,
    offsets: [max_layers][max_cols]i64 = undefined,
    mult: [max_layers][max_ngram]i64,
    /// Token id -> compressed id (the tokenizer-normalized classes engram.py builds; the lane caches them as JSON).
    map: []const i32,
    pad: i64,

    /// `mult`: engram_multipliers (numpy's seeded generator: given, not recomputed); `map`: the compressed token map.
    pub fn init(c: Config, map: []const i32, mult: [max_layers][max_ngram]i64) Hasher {
        var h: Hasher = .{ .layers = c.engram_layers.len, .ngram = c.engram_ngram, .heads = c.engram_heads, .mult = mult, .map = map, .pad = map[c.engram_pad] };
        primes(c, &h.primes);
        for (0..h.layers) |l| {
            var acc: i64 = 0;
            for (0..h.cols()) |k| {
                h.offsets[l][k] = acc;
                acc += h.primes[l][k];
            }
        }
        return h;
    }

    pub fn cols(h: Hasher) usize {
        return (h.ngram - 1) * h.heads;
    }

    /// `ids`: the whole sequence so far (negative at image positions, which no n-gram reaches across); `out`:
    /// [n][layers][cols] row ids, each offset into its layer's table.
    pub fn hashes(h: Hasher, ids: []const i32, start: usize, n: usize, out: []i64) void {
        const nc = h.cols();
        for (0..n) |r| {
            const pos = start + r;
            var toks: [max_ngram]i64 = undefined;
            var blocked = false;
            for (0..h.ngram) |shift| {
                // numpy: src = comp[clip(pos - shift - lo, 0)], blocked |= pos < shift or src < 0 (cumulative)
                const src: i64 = if (pos >= shift) comp(h, ids[pos - shift]) else comp(h, ids[0]);
                blocked = blocked or pos < shift or src < 0;
                toks[shift] = if (blocked) h.pad else src;
            }
            for (0..h.layers) |l| {
                var rolling: i64 = toks[0] *% h.mult[l][0];
                for (1..h.ngram) |i| {
                    rolling ^= toks[i] *% h.mult[l][i];
                    for (0..h.heads) |k| {
                        const c = (i - 1) * h.heads + k;
                        out[(r * h.layers + l) * nc + c] = @mod(rolling, h.primes[l][c]) + h.offsets[l][c];
                    }
                }
            }
        }
    }

    fn comp(h: Hasher, id: i32) i64 {
        return if (id < 0) -1 else h.map[@intCast(id)];
    }
};

/// One row's FP8 E4M3 values times their E8M0 scale (one a 32) rounded to bf16, as Engram._decode does on the GPU:
/// the value exactly in fp32, times 2^(scale - 127), then round to nearest even.
pub fn decodeRow(w: []const u8, s: []const u8, out: []u16) void {
    for (w, 0..) |b, i| {
        const v = e4m3(b);
        const e: i32 = @as(i32, s[i / 32]) - 127;
        out[i] = bf16(v * pow2(e));
    }
}

/// 2^e as fp32 (subnormal below 2^-126, infinity from 2^128).
fn pow2(e: i32) f32 {
    if (e >= 128) return std.math.inf(f32);
    if (e >= -126) return @bitCast(@as(u32, @intCast(e + 127)) << 23);
    if (e >= -149) return @bitCast(@as(u32, 1) << @intCast(e + 149));
    return 0;
}

/// An FP8 E4M3 (fn: no infinities, 0x7f / 0xff are NaN) byte as fp32.
pub fn e4m3(b: u8) f32 {
    const sign: f32 = if (b & 0x80 != 0) -1 else 1;
    const exp: u32 = (b >> 3) & 0xf;
    const man: u32 = b & 7;
    if (exp == 15 and man == 7) return std.math.nan(f32);
    const mag: f32 = if (exp == 0)
        @as(f32, @floatFromInt(man)) * 0x1p-9
    else
        (1.0 + @as(f32, @floatFromInt(man)) / 8.0) * pow2(@as(i32, @intCast(exp)) - 7);
    return sign * mag;
}

/// fp32 -> bf16, round to nearest even; NaN becomes c10's canonical 0x7fc0.
pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7fc0;
    const round = 0x7fff + ((u >> 16) & 1);
    return @intCast((u +% round) >> 16);
}

test "primes are drawn upward from the vocabulary size and never repeat" {
    try std.testing.expect(isPrime(2) and isPrime(16000057) and !isPrime(16000000) and !isPrime(1));
    var why: @import("config.zig").Why = .{};
    const c = try @import("config.zig").parse(std.testing.allocator, @import("config.zig").test_config, &why);
    var p: [max_layers][max_cols]i64 = undefined;
    primes(c, &p);
    try std.testing.expect(p[0][0] >= 16000000 and isPrime(@intCast(p[0][0])));
    try std.testing.expect(p[0][1] > p[0][0] and p[0][8] > p[0][7]); // the next order skips every prime taken
    try std.testing.expect(p[1][0] > p[0][7]);
}

test "a hash row by hand: bigram to 4-gram over a two-layer toy table" {
    var why: @import("config.zig").Why = .{};
    var c = try @import("config.zig").parse(std.testing.allocator, @import("config.zig").test_config, &why);
    c.engram_vocab = 10;
    c.engram_heads = 1;
    c.engram_pad = 0;
    const map = [_]i32{ 7, 1, 2, 3, 4, 5 };
    const h = Hasher.init(c, &map, .{ .{ 3, 5, 7, 9 }, .{ 11, 13, 15, 17 } });
    // primes from 10 up, unique across the table: 11, 13, 17 (layer 0), 19, 23, 29 (layer 1)
    try std.testing.expectEqual(@as(i64, 11), h.primes[0][0]);
    try std.testing.expectEqual(@as(i64, 29), h.primes[1][2]);
    var out: [2 * 2 * 3]i64 = undefined;
    h.hashes(&.{ 1, 2, 3 }, 1, 2, &out);
    // position 1: toks [2, 1, pad 7, pad 7] (shift 2 runs past the start), layer 0: 2*3 ^ 1*5 = 3 -> % 11 = 3
    try std.testing.expectEqual(@as(i64, 3), out[0]);
    // layer 0, 3-gram: 3 ^ 7*7 = 50 -> % 13 = 11, + offset 11
    try std.testing.expectEqual(@as(i64, 11 + 11), out[1]);
    // position 2 with an image id in front: [3, -1 -> pad...] then every later shift is padded
    var dead: [2 * 3]i64 = undefined;
    h.hashes(&.{ 1, -1, 3 }, 2, 1, &dead);
    try std.testing.expectEqual(@as(i64, @mod(3 * 3 ^ 7 * 5, 11)), dead[0]);
}

test "FP8 rows decode as the GPU does: exact values, power-of-two scales, bf16 rounding" {
    try std.testing.expectEqual(@as(f32, 448), e4m3(0x7e));
    try std.testing.expectEqual(@as(f32, -0x1p-9), e4m3(0x81));
    try std.testing.expect(std.math.isNan(e4m3(0x7f)));
    var out: [32]u16 = undefined;
    const w: [32]u8 = @splat(0x38); // 1.0
    decodeRow(&w, &.{127 + 3}, &out);
    try std.testing.expectEqual(@as(u16, 0x4100), out[0]); // 8.0
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // a tie rounds to even
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000))));
}
