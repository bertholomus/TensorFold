//! The DSpark drafter on a 2D node (the served four-node lane's dspark.py and markov.py on Comm2D, with
//! TF_DS_2D_DRAFT_HEAD=1): its blocks run TP2 inside each pair (weights.load gives a 2D node the TP2 split of them), so
//! their sublayer gathers are the pair gather (prompt2d.Two.pairGather) and both pairs compute the same drafts; its
//! logits are this node's backbone head quarter, and the Markov loop scores that quarter, the four ranks' (value, index)
//! bests gathered each step over the four nodes into _pick with WORLD 4: the best of the union in argmax's total order,
//! TP2's pick. draft.zig takes it through thin hooks behind Engine.two.
const std = @import("std");
const Config = @import("config.zig").Config;
const tri = @import("tri.zig");
const tri_markov = @import("tri_markov.zig");
const prompt2d = @import("prompt2d.zig");

const Two = prompt2d.Two;

/// The columns of a Markov segment on a 2D node: the quarters (253 and 252 blocks of 128 a TP2 half) share no wider
/// unit, and the port's kernels (TP2's 6a6cf68) place segment s of a rank at (seg0 + s) * n_cols.
pub const seg_cols = 128;

/// This node's vocabulary quarter: its first column and width (rank 0's half first, each half pair 0's part first).
pub fn quarter(t: *const Two) [2]usize {
    const half = t.hw[0] + t.hw[1];
    return .{ t.r * half + (if (t.p == 0) 0 else t.hw[0]), t.hw[t.p] };
}

/// markov.Markov on this node's head quarter (TF_DS_2D_DRAFT_HEAD) with the port's kernels: 128-column segments, the
/// quarter's blocks (n_cols 128, segs its blocks, seg0 its first block), split, the bests gathered over the four nodes
/// (world 4); the head's logits stay [rows, W] (lg_w: the quarter's width). Every element's bias and score is the one
/// markov.py computes (its per-element arithmetic depends on no range, tile or row count), the staging and cached rows
/// hold the quarter's columns in vocabulary order, and _finish takes the best of the segments' partials in the same
/// total order: only the launch geometry is not the lane's. fill makes the cached rows as on TP2.
pub fn markov(c: Config, t: *const Two, head: u64, emb: u64, none: u64) !tri_markov.Markov {
    const q = quarter(t);
    if (q[0] % seg_cols != 0 or q[1] % seg_cols != 0 or seg_cols % tri_markov.bn != 0) return error.VocabSplit;
    const segs = q[1] / seg_cols;
    const s_tiles: usize = tri.cdiv(seg_cols, tri_markov.bc);
    return .{
        .head = head,
        .emb = emb,
        .none = none,
        .slot = none,
        .world = 4,
        .rank_dim = c.markov_rank,
        .split = true,
        .n_cols = seg_cols,
        .segs = segs,
        .seg0 = q[0] / seg_cols,
        .cols = segs * seg_cols,
        .b_tiles = tri.cdiv(seg_cols, tri_markov.bn * tri_markov.sub),
        .s_tiles = s_tiles,
        .n_part = s_tiles * segs,
        .bp = tri.pow2(s_tiles * segs),
        .lg_w = q[1],
    };
}

test "a 2D node's Markov quarter: 128-column segments in vocabulary order" {
    // DeepSeek-V4.1's vocabulary: 1,010 blocks of 128, a TP2 half 505 (pair 0's quarter 253 blocks, pair 1's 252)
    var t: Two = undefined;
    t.hw = .{ 253 * 128, 252 * 128 };
    const firsts = [4]usize{ 0, 505, 253, 758 }; // node g = 2 p + r: (r0 p0) (r1 p0) (r0 p1) (r1 p1)
    for (0..4) |g| {
        t.r = g % 2;
        t.p = g / 2;
        var cfg: Config = undefined;
        cfg.markov_rank = 256;
        const m = try markov(cfg, &t, 1, 2, 3);
        try std.testing.expectEqual(firsts[g], m.seg0);
        try std.testing.expectEqual(@as(usize, if (t.p == 0) 253 else 252), m.segs);
        try std.testing.expectEqual(m.segs * 128, m.cols);
        try std.testing.expectEqual(t.hw[t.p], m.lg_w);
        try std.testing.expectEqual(@as(usize, 4), m.world);
        try std.testing.expectEqual(@as(usize, 256), m.bp);
        try std.testing.expectEqual(m.segs, m.n_part);
    }
}
