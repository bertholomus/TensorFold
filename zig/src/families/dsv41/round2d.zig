//! The exact 2D-TP4 split on the port's decode rounds (round.zig; the served four-node lane's rounds.py with model.py
//! Comm2D, wo_2d with TF_DS_2D_WOB_FOLD, moe_2d's fused decode experts, esum and logits_full): a node computes its
//! columns of every split output as on the prompt path (prompt2d.zig), and every exchange only moves bytes, so after
//! each one the node holds its TP2 rank's bytes and the rest of the round is TP2's. round.zig takes it through thin
//! hooks behind Engine.two; a TP2 engine has none and runs as before.
//!
//! The exchanges of a round: the attention and MoE partials' and Engram's projection's quarters (TP2's [2, R, N] gather,
//! which the mixes and Engram's sum then take as on TP2), the head's vocabulary quarters (TP2's [2, R, half]), and two
//! cat_rank fills of a TP2 rank's whole input from the two pairs' parts: wo_b's rotated input rows (wo_a's epilogue
//! writes this node's half, wo_b's suh at that half) and the experts' intermediate (between gate / up and down). Here
//! they go over NCCL (prompt2d.Two's quarters and catRank: the bytes the lane's RDMA rings put there), as round.zig's TP2
//! gathers do; the rings (ring2d.zig) come with the graphs.
const std = @import("std");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const weights = @import("weights.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");

const Two = prompt2d.Two;
const Engine = prompt.Engine;

/// wo_a's epilogue rotation for its group at u's column `col` (Exl3Group.rotated's rot, wo_2d's fold): wo_b's rotated
/// input rows of this node's half, into t.own [R, uw[p]] fp16 with wo_b's suh at that half (the rank's suh from this
/// pair's first column).
pub fn woRot(t: *const Two, lay: weights.Layer, col: usize) exl3_linear.RotOut {
    const at: usize = if (t.p == 0) 0 else t.uw[0];
    return .{ .suh = lay.wo_b.suh + at * 2, .xh = t.own, .ldr = @intCast(t.uw[t.p]), .off = @intCast(col) };
}

/// wo_2d's exchange after wo_a (cat_rank_into): the TP2 rank's whole rotated input rows xb [R, uw0 + uw1] fp16 from the
/// two pairs' halves (this node's in t.own), for wo_b's glinear on them.
pub fn woExchange(t: *const Two, e: *const Engine, xb: u64, rows: usize) !void {
    try t.catRank(e, t.own, xb, rows, t.uw, 2);
}

/// moe_2d's experts on a decode window, after the routing (experts2d.gateup_fused, cat_rank, down_fused): this node's
/// gate / up blocks with decode_prep, the rank's whole intermediate assembled from the two pairs' blocks, down for this
/// node's output columns with the combine: out fp32 [R, dw[p]].
pub fn experts(t: *const Two, e: *const Engine, ex: weights.Experts, sc: exl3_experts.DecodeScratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, rows: usize, limit: f32) !void {
    if (t.ir != 0) {
        // TF_DS_2D_GU=parity (experts2d gateup_rest_fused, pair_parts, assemble_rest): the main blocks and the rest
        // columns of the experts this pair computes them for (one launch with Engine.par), then the packs exchanged
        // with the column partner and the half assembled
        if (e.par) |pk| try exl3_experts2d.decodeGateUpOne(e.ex, pk, e.s, ex, sc, t.rest_dec.?, x, x_stride, pick, wts, out, rows, limit) else {
            try exl3_experts2d.decodeGateUp(e.ex, e.s, ex, sc, x, x_stride, pick, wts, out, rows, limit);
            try exl3_experts2d.decodeGateUpRest(e.ex, e.ops, e.s, ex, sc, t.rest_dec.?, pick, rows, limit);
        }
        try t.assembleParity(e, ex, sc.xd, pick, rows * sc.slots);
    } else {
        try exl3_experts2d.decodeGateUp(e.ex, e.s, ex, sc, x, x_stride, pick, wts, out, rows, limit);
        try t.catRank(e, sc.xd, t.xd_full, rows * sc.slots, t.gu, 2);
    }
    try exl3_experts2d.decodeDown(e.ex, e.s, ex, sc, t.xd_full, pick, wts, out, rows);
}

test "wo_b's rotated input: each pair's half of the rank's suh" {
    var t: Two = undefined;
    t.uw = .{ 2048, 2048 };
    t.own = 0x7000;
    var lay: weights.Layer = undefined;
    lay.wo_b.suh = 0x10000;
    t.p = 0;
    try std.testing.expectEqual(exl3_linear.RotOut{ .suh = 0x10000, .xh = 0x7000, .ldr = 2048, .off = 1024 }, woRot(&t, lay, 1024));
    t.p = 1;
    try std.testing.expectEqual(exl3_linear.RotOut{ .suh = 0x10000 + 2048 * 2, .xh = 0x7000, .ldr = 2048, .off = 0 }, woRot(&t, lay, 0));
}
