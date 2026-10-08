//! kernels.py's q and window-KV norms with their rotations and RoPE, RoPE on heads, the router's and the indexer's row
//! matmuls (rowmm2, rowmm_wts, rowmm_gate) and the MoE routing, each launched as its Python wrapper launches it.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const aot = cuda.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;

/// kernels.DECODE_ROWS (TF_DS_DECODE_ROWS unset in the served lane): windows up to this many rows take the row kernels.
pub const decode_rows = 16;
/// kernels.ROWMM2_ROWS: rowmm2's own kernel up to this many rows.
pub const rowmm2_rows = 2;
/// kernels.ROWMM_RB (TF_DS_ROWMM_RB unset in the served lane): _rowmm2r's rows a program at most.
pub const rowmm_rb = 16;
/// kernels.ROWMM_PARTS_MIN (TF_DS_ROWMM_PARTS_MIN unset in the served lane): rowmm_gate's chunk sums from this many rows.
pub const rowmm_parts_min = 1;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

/// A rotation of rmsnorm_rot / q_kv_norm (an EXL3 linear reading the normed rows): its suh [D] and input rows h [R, D], fp16.
pub const Rot = struct { suh: u64, h: u64 };

/// S0, H0, S1, H1 as the wrappers pass them: each rotation's (suh, h), then out (bf16) for each missing one.
fn rotArgs(rot: []const Rot, out: u64) ![4]aot.Arg {
    if (rot.len > 2) return error.TooManyRotations;
    const names = [4][]const u8{ "S0", "H0", "S1", "H1" };
    var a: [4]aot.Arg = undefined;
    for (0..2) |i| {
        a[2 * i] = if (i < rot.len) p(names[2 * i], "*fp16", rot[i].suh) else p(names[2 * i], "*bf16", out);
        a[2 * i + 1] = if (i < rot.len) p(names[2 * i + 1], "*fp16", rot[i].h) else p(names[2 * i + 1], "*bf16", out);
    }
    return a;
}

/// q_kv_norm: rmsnorm_rot(x, w, eps, rot) into out [R, D] (row stride D) and kv_norm_rope(y, wk, ...) into kout [R, DK], one launch.
pub fn qKvNorm(t: Tri, x: u64, xs: usize, w: u64, eps: f32, rot: []const Rot, y: u64, wk: u64, cos: u64, sin: u64, pos: u64, ring: u64, ring_rows: usize, slots: u64, quant: bool, rd: usize, out: u64, kout: u64, rows: usize, d: usize, dk: usize) !void {
    if (d > 2048) return error.RowTooWide;
    const st = try rotArgs(rot, out);
    // switch "qkv_split" (on in the served build): each rotation a program of its own beside the norm's
    const split = rot.len > 0;
    const progs = 2 + (if (split) rot.len else 0);
    try t.run("_q_kv_norm", .{ u(rows), u(progs), 1 }, &.{
        p("X", "*bf16", x),       aot.int("xs", @intCast(xs)), p("W", "*bf16", w),          p("OUT", "*bf16", out),                    aot.int("os_", @intCast(d)),
        aot.float("eps", eps),    st[0],                       st[1],                       st[2],                                     st[3],
        p("Y", "*bf16", y),       p("WK", "*bf16", wk),        p("COS", "*fp32", cos),      p("SIN", "*fp32", sin),                    p("POS", "*i64", pos),
        p("KOUT", "*bf16", kout), p("RING", "*bf16", ring),    p("SLOT_OF", "*i64", slots), aot.int("ring_size", @intCast(ring_rows)),
    }, &.{ cb("QUANT", quant), ci("D", d), ci("BLOCK", tri.pow2(d)), ci("NROT", rot.len), ci("DK", dk), ci("RD", rd), t.pdlConst(), cb("SPLIT", split) });
}

/// rmsnorm_rot: out [R, D] (row stride os) = RMSNorm(x) bf16 and, from its rows, each rotation's h (rot_many's bits).
pub fn rmsnormRot(t: Tri, x: u64, xs: usize, w: u64, eps: f32, rot: []const Rot, out: u64, os: usize, rows: usize, d: usize) !void {
    const st = try rotArgs(rot, out);
    try t.run("_rmsnorm_rot", .{ u(rows), 1, 1 }, &.{
        p("X", "*bf16", x),    aot.int("xs", @intCast(xs)), p("W", "*bf16", w), p("OUT", "*bf16", out), aot.int("os_", @intCast(os)),
        aot.float("eps", eps), st[0],                       st[1],              st[2],                  st[3],
    }, &.{ ci("D", d), ci("BLOCK", tri.pow2(d)), ci("NROT", rot.len), t.pdlConst() });
}

/// kv_norm_rope: the window KV out [R, D] = RMSNorm(y), RoPE on its last rd dims, FP8 quant-dequant, also into ring row slots[r].
pub fn kvNormRope(t: Tri, y: u64, w: u64, cos: u64, sin: u64, pos: u64, ring: u64, ring_rows: usize, slots: u64, eps: f32, quant: bool, rd: usize, out: u64, rows: usize, d: usize) !void {
    try t.run("_kv_norm_rope", .{ u(rows), 1, 1 }, &.{
        p("Y", "*bf16", y),     p("W", "*bf16", w),       p("COS", "*fp32", cos),      p("SIN", "*fp32", sin),                    p("POS", "*i64", pos),
        p("OUT", "*bf16", out), p("RING", "*bf16", ring), p("SLOT_OF", "*i64", slots), aot.int("ring_size", @intCast(ring_rows)), aot.float("eps", eps),
    }, &.{ cb("QUANT", quant), ci("D", d), ci("RD", rd), t.pdlConst() });
}

/// rope_heads: RoPE (inverse: the inverse rotation) on the last rd dims of every head of x [R, H, HD] bf16, in place.
pub fn ropeHeads(t: Tri, x: u64, cos: u64, sin: u64, pos: u64, rd: usize, inverse: bool, rows: usize, h: usize, hd: usize) !void {
    const hb = @min(h, 16);
    try t.run("_rope_heads", .{ u(rows), u(h / hb), 1 }, &.{ p("X", "*bf16", x), p("COS", "*fp32", cos), p("SIN", "*fp32", sin), p("POS", "*i64", pos) }, &.{ ci("H", h), ci("HD", hd), ci("RD", rd), cb("INV", inverse), ci("HB", hb), t.pdlConst() });
}

/// rowmm2: x [R, K] bf16 (row stride xs) @ w [N, K]^T fp16 -> out [R, N] fp32, rowmm's bits: _rowmm2, _rowmm2r past 2 rows, or rowmm.
pub fn rowmm2(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) !void {
    // a narrow layer: one output (one warp) a program
    const bn: usize = if (n > 64) 4 else 1;
    if (rows > rowmm2_rows and rowmm_rb > 0 and k % 256 == 0 and rows <= tri.decode_rows) {
        const rb = @min(rowmm_rb, rows);
        return t.run("_rowmm2r", .{ tri.cdiv(rows, rb), tri.cdiv(n, bn), 1 }, &.{
            p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("OUT", "*fp32", out), aot.float("scale", 1.0), aot.int("rows", @intCast(rows)),
        }, &.{ ci("K", k), ci("N", n), ci("BN", bn), ci("BK", 256), ci("RB", rb), t.pdlConst(), cb("WTS", false) });
    }
    if (rows > rowmm2_rows or k % 256 != 0) return rowmm(t, x, xs, w, out, rows, k, n);
    try t.run("_rowmm2", .{ u(rows), tri.cdiv(n, bn), 1 }, &.{
        p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("OUT", "*fp32", out), aot.float("scale", 1.0),
    }, &.{ ci("K", k), ci("N", n), ci("BN", bn), ci("BK", 256), t.pdlConst(), cb("WTS", false) });
}

/// rowmm: x [R, K] bf16 @ w [N, K]^T fp16 -> out [R, N] fp32, a row alone (rowmm2's fallback; no _rowmm launch is recorded).
pub fn rowmm(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) !void {
    try t.run("_rowmm", .{ u(rows), tri.cdiv(n, 8), 1 }, &.{ p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("OUT", "*fp32", out) }, &.{ ci("K", k), ci("N", n), ci("BN", 8), ci("BK", 256), t.pdlConst() });
}

/// rowmm_wts: out [R, N] bf16 = (rowmm(x, w).to(bf16) * scale).to(bf16) in one launch (the indexer's weights).
pub fn rowmmWts(t: Tri, x: u64, xs: usize, w: u64, scale: f32, out: u64, rows: usize, k: usize, n: usize) !void {
    const bn: usize = if (n > 64) 4 else 1;
    try t.run("_rowmm2", .{ u(rows), tri.cdiv(n, bn), 1 }, &.{
        p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("OUT", "*bf16", out), aot.float("scale", scale),
    }, &.{ ci("K", k), ci("N", n), ci("BN", bn), ci("BK", 256), t.pdlConst(), cb("WTS", true) });
}

/// rowmm_parts' tile: outputs a program, K chunks a program and warps (the variant's num_warps).
pub const RpTile = struct { bn: usize, cpg: usize, nw: usize };

/// _rp_tile: rowmm_parts' tile by rows (TF_DS_ROWMM_PARTS unset in the served lane).
pub fn rpTile(rows: usize) RpTile {
    return if (rows <= 6) .{ .bn = 16, .cpg = 1, .nw = 2 } else .{ .bn = 8, .cpg = 1, .nw = 2 };
}

/// rowmm_gate: the gate's logits as chunk sums out [R, K / 256, N] fp32 (decode windows) or rowmm2's [R, N]; returns route()'s kc (0: plain).
pub fn rowmmGate(t: Tri, x: u64, xs: usize, w: u64, out: u64, rows: usize, k: usize, n: usize) !usize {
    const tile = rpTile(rows);
    const kc = k / 256;
    // switch "rowmm_parts" (on in the served build); the wrapper's dtype check (x bf16, w fp16) holds by these pointer types
    if (!(rowmm_parts_min <= rows and rows <= tri.decode_rows and k % 256 == 0 and kc % tile.cpg == 0 and n >= 32)) {
        try rowmm2(t, x, xs, w, out, rows, k, n);
        return 0;
    }
    const args = [_]aot.Arg{ p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("W", "*fp16", w), p("S", "*fp32", out), aot.int("rows", @intCast(rows)) };
    const consts = [_]aot.Const{ ci("K", k), ci("N", n), ci("BN", tile.bn), ci("BK", 256), ci("CPG", tile.cpg), ci("RB", rows), t.pdlConst() };
    try t.run("_rowmm_parts", .{ tri.cdiv(n, tile.bn), u(kc / tile.cpg), 1 }, &args, &consts);
    return kc;
}

/// route: picks [R, SLOTS] i32 and weights [R, SLOTS] fp32 (shared expert last) from logits [R, NE] (kc 0) or chunk sums [R, kc, NE].
pub fn route(t: Tri, logits: u64, kc: usize, bias: u64, topk: usize, scale: f32, shared_id: usize, pick: u64, wts: u64, rows: usize, ne: usize, slots: usize) !void {
    if (kc > 0) return t.run("_route_parts", .{ u(rows), 1, 1 }, &.{
        p("S", "*fp32", logits), p("BIAS", "*fp32", bias), p("PICK", "*i32", pick), p("WTS", "*fp32", wts), aot.float("scale", scale), aot.int("shared_id", @intCast(shared_id)),
    }, &.{ ci("NE", ne), ci("NB", tri.pow2(ne)), ci("TOPK", topk), ci("SLOTS", slots), ci("SP", tri.pow2(slots)), ci("KC", kc), t.pdlConst() });
    // one warp a row (switch "rowmm", on in the served build): the variant's num_warps
    try t.run("_route", .{ u(rows), 1, 1 }, &.{
        p("L", "*fp32", logits), p("BIAS", "*fp32", bias), p("PICK", "*i32", pick), p("WTS", "*fp32", wts), aot.float("scale", scale), aot.int("shared_id", @intCast(shared_id)),
    }, &.{ ci("NE", ne), ci("NB", tri.pow2(ne)), ci("TOPK", topk), ci("SLOTS", slots), ci("SP", tri.pow2(slots)), t.pdlConst() });
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

/// A recorded tensor's dtype as torch names it ("bfloat16", "float32", ...).
fn dtype(c: Case, tensor: []const u8) []const u8 {
    return c.v.get("tensors").?.object.get(tensor).?.array.items[0].string;
}

/// The wrapper's rot list from the recorded S0 H0 S1 H1: a rotation's suh is a vector [D], a missing one's slot holds out [R, D].
fn rotOf(c: Case, buf: *[2]Rot) []const Rot {
    const pairs = [2][2][]const u8{ .{ "S0", "H0" }, .{ "S1", "H1" } };
    var n: usize = 0;
    for (pairs) |pr| {
        if (c.rank(pr[0]) != 1) break;
        buf[n] = .{ .suh = c.ptr(pr[0]), .h = c.ptr(pr[1]) };
        n += 1;
    }
    return buf[0..n];
}

test "q_kv_norm launches as recorded" {
    try tri.conform("_q_kv_norm", struct {
        fn call(t: Tri, c: Case) !void {
            var buf: [2]Rot = undefined;
            const quant = c.constInt("QUANT") != 0;
            const rd: usize = @intCast(c.constInt("RD"));
            try qKvNorm(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.float("eps"), rotOf(c, &buf), c.ptr("Y"), c.ptr("WK"), c.ptr("COS"), c.ptr("SIN"), c.ptr("POS"), c.ptr("RING"), c.dim("RING", 0), c.ptr("SLOT_OF"), quant, rd, c.ptr("OUT"), c.ptr("KOUT"), c.dim("X", 0), c.dim("X", 1), c.dim("Y", 1));
        }
    }.call);
}

test "rmsnorm_rot launches as recorded" {
    try tri.conform("_rmsnorm_rot", struct {
        fn call(t: Tri, c: Case) !void {
            var buf: [2]Rot = undefined;
            try rmsnormRot(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.float("eps"), rotOf(c, &buf), c.ptr("OUT"), c.stride("OUT", 0), c.dim("X", 0), c.dim("X", 1));
        }
    }.call);
}

test "kv_norm_rope launches as recorded" {
    try tri.conform("_kv_norm_rope", struct {
        fn call(t: Tri, c: Case) !void {
            const quant = c.constInt("QUANT") != 0;
            const rd: usize = @intCast(c.constInt("RD"));
            try kvNormRope(t, c.ptr("Y"), c.ptr("W"), c.ptr("COS"), c.ptr("SIN"), c.ptr("POS"), c.ptr("RING"), c.dim("RING", 0), c.ptr("SLOT_OF"), c.float("eps"), quant, rd, c.ptr("OUT"), c.dim("Y", 0), c.dim("Y", 1));
        }
    }.call);
}

test "rope_heads launches as recorded" {
    try tri.conform("_rope_heads", struct {
        fn call(t: Tri, c: Case) !void {
            try ropeHeads(t, c.ptr("X"), c.ptr("COS"), c.ptr("SIN"), c.ptr("POS"), @intCast(c.constInt("RD")), c.constInt("INV") != 0, c.dim("X", 0), c.dim("X", 1), c.dim("X", 2));
        }
    }.call);
}

test "rowmm2 (1-2 rows) and rowmm_wts launch _rowmm2 as recorded" {
    try tri.conform("_rowmm2", struct {
        fn call(t: Tri, c: Case) !void {
            // rowmm_wts writes the indexer's bf16 weights, rowmm2 fp32 rows
            if (std.mem.eql(u8, dtype(c, "OUT"), "bfloat16")) return rowmmWts(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.float("scale"), c.ptr("OUT"), c.dim("X", 0), c.dim("X", 1), c.dim("W", 0));
            try rowmm2(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.ptr("OUT"), c.dim("X", 0), c.dim("X", 1), c.dim("W", 0));
        }
    }.call);
}

test "rowmm2 (windows past 2 rows) launches _rowmm2r as recorded" {
    try tri.conform("_rowmm2r", struct {
        fn call(t: Tri, c: Case) !void {
            try rowmm2(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.ptr("OUT"), c.dim("X", 0), c.dim("X", 1), c.dim("W", 0));
        }
    }.call);
}

test "rowmm_gate launches _rowmm_parts as recorded" {
    try tri.conform("_rowmm_parts", struct {
        fn call(t: Tri, c: Case) !void {
            const kc = try rowmmGate(t, c.ptr("X"), c.stride("X", 0), c.ptr("W"), c.ptr("S"), c.dim("X", 0), c.dim("X", 1), c.dim("W", 0));
            try std.testing.expectEqual(c.dim("S", 1), kc);
        }
    }.call);
}

/// route() on a recorded case: rowmm_gate's chunk sums S [R, KC, NE] or plain logits L [R, NE].
fn callRoute(t: Tri, c: Case) !void {
    const parts = c.has("S");
    const lg = if (parts) "S" else "L";
    try route(t, c.ptr(lg), if (parts) c.dim("S", 1) else 0, c.ptr("BIAS"), @intCast(c.constInt("TOPK")), c.float("scale"), @intCast(c.int("shared_id")), c.ptr("PICK"), c.ptr("WTS"), c.dim(lg, 0), c.dim(lg, c.rank(lg) - 1), c.dim("PICK", 1));
}

test "route (logits) launches _route as recorded" {
    try tri.conform("_route", callRoute);
}

test "route (rowmm_gate's chunk sums) launches _route_parts as recorded" {
    try tri.conform("_route_parts", callRoute);
}
