//! kernels.py's row kernels around the layers: mHC pre / post of prompt chunks, the embedding, taps, the collapse
//! before the head, Engram's gate and the plain RMSNorm, each launched as its Python wrapper launches it.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const aot = cuda.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;

/// kernels.HC_BLOCKS: the mHC mixing dots' fixed K split.
pub const hc_blocks = 40;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

/// hc_pre: h [R, 4, D] bf16 -> out [R, D] = RMSNorm(sum_j pre_in[j] h_j); pre_out / post / comb from h's own mixes.
pub fn hcPre(t: Tri, h: u64, fn_w: u64, scale: u64, base: u64, pre_in: u64, norm_w: u64, eps: f32, hc_eps: f32, iters: usize, out: u64, pre_out: u64, post: u64, comb: u64, part: u64, rows: usize, d: usize) !void {
    try t.run("_hc_partial", .{ u(rows), hc_blocks, 1 }, &.{ p("X", "*bf16", h), p("FN", "*fp32", fn_w), p("PART", "*fp32", part) }, &.{ ci("WIDE", 4 * d), ci("NB", hc_blocks), ci("SUB", 128) });
    try t.run("_hc_finish", .{ u(rows), 1, 1 }, &.{
        p("X", "*bf16", h),          p("PART", "*fp32", part),       p("BASE", "*fp32", base), p("SCALE", "*fp32", scale),
        p("PRE_IN", "*fp32", pre_in), p("NW", "*bf16", norm_w),      p("OUT", "*bf16", out),   p("PRE_OUT", "*fp32", pre_out),
        p("POST", "*fp32", post),    p("COMB", "*fp32", comb),       aot.float("eps", eps),    aot.float("hc_eps", hc_eps),
    }, &.{ ci("D", d), ci("NB", hc_blocks), ci("ITERS", iters), ci("BLOCK", 1024) });
}

/// hc_post: gathered [world, R, D] fp32 partials, h [R, 4, D] bf16 -> out [R, 4, D] (may alias h).
pub fn hcPost(t: Tri, gathered: u64, h: u64, post: u64, comb: u64, out: u64, world: usize, rows: usize, d: usize) !void {
    try t.run("_hc_post", .{ u(rows), u(d / 1024), 1 }, &.{ p("G", "*fp32", gathered), aot.int("RS", @intCast(rows * d)), p("X", "*bf16", h), p("XOUT", "*bf16", out), p("POST", "*fp32", post), p("COMB", "*fp32", comb) }, &.{ ci("D", d), ci("WORLD", world), ci("BLOCK", 1024), t.pdlConst() });
}

/// rmsnorm: x [R, D] (row stride xs) bf16 -> out [R, D] (row stride os) bf16.
pub fn rmsnorm(t: Tri, x: u64, xs: usize, w: u64, out: u64, os: usize, eps: f32, rows: usize, d: usize) !void {
    try t.run("_rmsnorm", .{ u(rows), 1, 1 }, &.{ p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*bf16", w), p("OUT", "*bf16", out), aot.int("os_", @intCast(os)), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", tri.pow2(d)), t.pdlConst() });
}

/// embed_init: the streams h [R, hc, D] (every stream the token's embedding row) and pre [R, hc] = (1, 0, 0, 0).
pub fn embedInit(t: Tri, embed: u64, ids: u64, h: u64, pre: u64, rows: usize, d: usize, hc: usize) !void {
    try t.run("_embed_init", .{ u(rows), u(d / 1024), 1 }, &.{ p("EMB", "*bf16", embed), p("IDS", "*i64", ids), p("H", "*bf16", h), p("PRE", "*fp32", pre) }, &.{ ci("D", d), ci("HC", hc), ci("BLOCK", 1024), t.pdlConst() });
}

/// tap: out [R, D] (row stride os, a column block of the taps buffer) = h.to(fp32).mean(1).to(bf16), h [R, 4, D].
pub fn tap(t: Tri, h: u64, out: u64, os: usize, rows: usize, d: usize) !void {
    try t.run("_tap", .{ u(rows), u(d / 1024), 1 }, &.{ p("H", "*bf16", h), p("OUT", "*bf16", out), aot.int("os_", @intCast(os)) }, &.{ ci("D", d), ci("BLOCK", 1024), t.pdlConst() });
}

/// collapse_norm: out [R, D] = RMSNorm(sum_j pre[j] h_j) for the head.
pub fn collapseNorm(t: Tri, h: u64, pre: u64, w: u64, out: u64, eps: f32, rows: usize, d: usize) !void {
    try t.run("_collapse_norm", .{ u(rows), 1, 1 }, &.{ p("X", "*bf16", h), p("PRE", "*fp32", pre), p("NW", "*bf16", w), p("OUT", "*bf16", out), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", 1024), t.pdlConst() });
}

/// collapse: out [R, D] = sum_j pre[j] h_j (bf16), no norm.
pub fn collapse(t: Tri, h: u64, pre: u64, out: u64, rows: usize, d: usize) !void {
    try t.run("_collapse", .{ u(rows), u(d / 1024), 1 }, &.{ p("X", "*bf16", h), p("PRE", "*fp32", pre), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("BLOCK", 1024) });
}

/// engram_gate: h [R, 4, D] bf16, kv [R, 5 D] bf16 (4 keys, the value), qk [4, D] fp32 -> out [R, 4, D].
pub fn engramGate(t: Tri, h: u64, kv: u64, qk: u64, out: u64, eps: f32, rows: usize, d: usize) !void {
    try t.run("_engram_gate", .{ u(rows), 4, 1 }, &.{ p("H", "*bf16", h), p("KV", "*bf16", kv), p("QK", "*fp32", qk), p("OUT", "*bf16", out), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", 1024), t.pdlConst() });
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

test "hc_pre (prompt chunks) launches as recorded" {
    try tri.conform("_hc_partial", struct {
        fn call(t: Tri, c: Case) !void {
            const rows = c.dim("X", 0);
            const d = c.dim("X", 2);
            try hcPre(t, c.ptr("X"), c.ptr("FN"), 0x7e00_0000_0000, 0x7e00_0000_0100, 0x7e00_0000_0200, 0x7e00_0000_0300, 1e-20, 1e-6, 20, 0x7e00_0000_0400, 0x7e00_0000_0500, 0x7e00_0000_0600, 0x7e00_0000_0700, c.ptr("PART"), rows, d);
        }
    }.call);
    try tri.conform("_hc_finish", struct {
        fn call(t: Tri, c: Case) !void {
            try hcPre(t, c.ptr("X"), 0x7e00_0000_0000, c.ptr("SCALE"), c.ptr("BASE"), c.ptr("PRE_IN"), c.ptr("NW"), c.float("eps"), c.float("hc_eps"), @intCast(c.constInt("ITERS")), c.ptr("OUT"), c.ptr("PRE_OUT"), c.ptr("POST"), c.ptr("COMB"), c.ptr("PART"), c.dim("X", 0), c.dim("X", 2));
        }
    }.call);
}

test "hc_post launches as recorded" {
    try tri.conform("_hc_post", struct {
        fn call(t: Tri, c: Case) !void {
            try hcPost(t, c.ptr("G"), c.ptr("X"), c.ptr("POST"), c.ptr("COMB"), c.ptr("XOUT"), c.dim("G", 0), c.dim("G", 1), c.dim("G", 2));
        }
    }.call);
}

test "rmsnorm launches as recorded" {
    try tri.conform("_rmsnorm", struct {
        fn call(t: Tri, c: Case) !void {
            try rmsnorm(t, c.ptr("X"), @intCast(c.int("xs")), c.ptr("W"), c.ptr("OUT"), @intCast(c.int("os_")), c.float("eps"), c.dim("X", 0), c.dim("W", 0));
        }
    }.call);
}

test "embed_init, tap, collapse_norm, collapse and engram_gate launch as recorded" {
    try tri.conform("_embed_init", struct {
        fn call(t: Tri, c: Case) !void {
            try embedInit(t, c.ptr("EMB"), c.ptr("IDS"), c.ptr("H"), c.ptr("PRE"), c.dim("IDS", 0), c.dim("EMB", 1), c.dim("H", 1));
        }
    }.call);
    try tri.conform("_tap", struct {
        fn call(t: Tri, c: Case) !void {
            try tap(t, c.ptr("H"), c.ptr("OUT"), @intCast(c.int("os_")), c.dim("H", 0), c.dim("H", 2));
        }
    }.call);
    try tri.conform("_collapse_norm", struct {
        fn call(t: Tri, c: Case) !void {
            try collapseNorm(t, c.ptr("X"), c.ptr("PRE"), c.ptr("NW"), c.ptr("OUT"), c.float("eps"), c.dim("X", 0), c.dim("X", 2));
        }
    }.call);
    try tri.conform("_collapse", struct {
        fn call(t: Tri, c: Case) !void {
            try collapse(t, c.ptr("X"), c.ptr("PRE"), c.ptr("OUT"), c.dim("X", 0), c.dim("X", 2));
        }
    }.call);
    try tri.conform("_engram_gate", struct {
        fn call(t: Tri, c: Case) !void {
            try engramGate(t, c.ptr("H"), c.ptr("KV"), c.ptr("QK"), c.ptr("OUT"), c.float("eps"), c.dim("H", 0), c.dim("H", 2));
        }
    }.call);
}
