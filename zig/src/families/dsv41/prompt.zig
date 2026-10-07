//! A prompt chunk's forward on one TP rank, the served build's Model._forward_k (model.py) step by step on the served
//! build's kernels: the streams of the chunk's tokens (embed_init), then a layer at a time the attention mixes (hc_pre, or
//! hc_pre_pf with the previous layer's gathered MoE partials posted first), attention (attention_k), the gather of its
//! partials, the FFN mixes with that post fused in (hc_pre_pf), the MoE and its gather, whose post goes into the next
//! layer's mixes (switch "hc_pf2"). So far window-only layers (compress ratio 0) without Engram; the others return
//! error.NotPortedYet.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const tri = @import("tri.zig");
const tri_basic = @import("tri_basic.zig");
const tri_hc = @import("tri_hc.zig");
const tri_norm = @import("tri_norm.zig");
const tri_attn = @import("tri_attn.zig");
const tri_markov = @import("tri_markov.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_experts = @import("exl3_experts.zig");
const ops = @import("ops.zig");
const cublas = @import("cublas.zig");
const comm = @import("comm.zig");

/// model.py's RING_EXTRA: window ring slots beyond the window (a verify window never clobbers a key it reads).
pub const ring_extra = 16;

/// Device memory carved from one allocation in 256-byte aligned pieces, freed together.
pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    pub fn init(d: *const cuda.Driver, bytes: usize) !Arena {
        return .{ .buf = try cuda.DeviceBuffer.alloc(d, bytes) };
    }

    pub fn deinit(a: *Arena) void {
        a.buf.free();
    }

    pub fn take(a: *Arena, bytes: usize) !u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        if (at + bytes > a.buf.len) return error.ArenaFull;
        a.used = at + bytes;
        return a.buf.ptr + at;
    }
};

/// A RoPE kind's tables on the device: cos, sin fp32 [rows, rope_dim / 2] (ops.RopeTables' rows).
pub const Rope = struct { cos: u64, sin: u64 };

/// What a chunk's steps run on: one rank's weights, kernels, stream and gather.
pub const Engine = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    t: tri.Tri,
    blas: *const cublas.Blas,
    comm: *const comm.Comm,
    pf: *const exl3_prefill.Kernels,
    ex: *const exl3_experts.Kernels,
    ops: *const ops.Ops,
    c: *const Config,
    w: *const weights.Weights,
    world: usize,
    plain: Rope,
    compressed: Rope,

    fn heads(e: *const Engine) usize {
        return e.c.heads / e.world;
    }

    fn slots(e: *const Engine) usize {
        return e.c.top_k + 1;
    }

    /// A sequence's window ring for one layer: bf16 [window + RING_EXTRA, head_dim].
    pub fn ringBytes(e: *const Engine) usize {
        return (e.c.window + ring_extra) * e.c.head_dim * 2;
    }
};

/// A prompt chunk's buffers for up to `cap` rows (model.py _forward_k's and its steps' torch.empty's).
pub const Chunk = struct {
    cap: usize,
    n: usize = 0,
    start: usize = 0,
    ids: u64, // int64 [cap]
    pos: u64, // int64 [cap]: start .. start + n
    h: u64, // bf16 [cap, hc, D]: the streams
    h_alt: u64, // their second buffer (hc_pre_pf writes the posted streams there)
    x: u64, // bf16 [cap, D]: a block's input rows
    part: u64, // fp32 [cap * HC_BLOCKS * 32]
    pre: u64, // fp32 [cap, hc]: the layer's input pre (the previous layer's FFN pre_out)
    pre_a: u64, // fp32 [cap, hc]
    pre_f: u64, // fp32 [cap, hc]
    post: u64, // fp32 [cap, hc]
    comb: u64, // fp32 [cap, hc, hc]
    // attention
    qa: u64, // bf16 [cap, q_lora]
    y: u64, // bf16 [cap, head_dim]
    qr: u64, // bf16 [cap, q_lora]
    q: u64, // bf16 [cap, heads, head_dim]
    wsrc: u64, // bf16 [window - 1 + cap, head_dim]
    wlo: u64, // int64 [1]
    o: u64, // bf16 [cap, heads, head_dim]
    u: u64, // bf16 [cap, groups * o_lora]
    pa: u64, // fp32 [cap, D]: the attention partial
    ga: u64, // fp32 [world, cap, D]: its gather
    neg: u64, // int64 [cap], -1: no ring slot (model.py _neg)
    ring_idx: u64, // int64 [ring]
    invalid: u64, // uint32 [1]
    // MoE
    xf: u64, // fp32 [cap, D]
    gate_f: u64, // fp32 [experts, D]
    logits: u64, // fp32 [cap, experts]
    pick: u64, // int32 [cap, slots]
    wts: u64, // fp32 [cap, slots]
    pm: u64, // fp32 [cap, D]: the MoE partial
    gm: u64, // fp32 [world, cap, D]: its gather (the pending post)
    xs: exl3_experts.Scratch,
    ws: exl3_prefill.Workspace,
    blas_ws: u64, // cuBLAS workspace
    pending: bool = false, // gm holds a MoE gather whose post is not in h yet

    /// Every buffer for `cap` rows from `a`; the scratch the served build zeros (or fills with -1) is set the same.
    pub fn init(e: *const Engine, a: *Arena, cap: usize) !Chunk {
        const c = e.c;
        const d = c.hidden;
        const hc = c.hc;
        const l0 = e.w.layers[0];
        const hl = e.heads();
        const hd = c.head_dim;
        const groups = l0.wo_a.len;
        const o_lora = l0.wo_a[0].n;
        const ex = l0.experts;
        const sl = e.slots();
        var ch: Chunk = undefined;
        ch.cap = cap;
        ch.n = 0;
        ch.start = 0;
        ch.pending = false;
        ch.ids = try a.take(cap * 8);
        ch.pos = try a.take(cap * 8);
        ch.h = try a.take(cap * hc * d * 2);
        ch.h_alt = try a.take(cap * hc * d * 2);
        ch.x = try a.take(cap * d * 2);
        ch.part = try a.take(cap * tri_basic.hc_blocks * 32 * 4);
        ch.pre = try a.take(cap * hc * 4);
        ch.pre_a = try a.take(cap * hc * 4);
        ch.pre_f = try a.take(cap * hc * 4);
        ch.post = try a.take(cap * hc * 4);
        ch.comb = try a.take(cap * hc * hc * 4);
        ch.qa = try a.take(cap * l0.wq_a.n * 2);
        ch.y = try a.take(cap * hd * 2);
        ch.qr = try a.take(cap * l0.wq_a.n * 2);
        ch.q = try a.take(cap * hl * hd * 2);
        ch.wsrc = try a.take((c.window - 1 + cap) * hd * 2);
        ch.wlo = try a.take(8);
        ch.o = try a.take(cap * hl * hd * 2);
        ch.u = try a.take(cap * groups * o_lora * 2);
        ch.pa = try a.take(cap * d * 4);
        ch.ga = try a.take(e.world * cap * d * 4);
        ch.neg = try a.take(cap * 8);
        ch.ring_idx = try a.take((c.window + ring_extra) * 8);
        ch.invalid = try a.take(4);
        ch.xf = try a.take(cap * d * 4);
        ch.gate_f = try a.take(c.experts * d * 4);
        ch.logits = try a.take(cap * c.experts * 4);
        ch.pick = try a.take(cap * sl * 4);
        ch.wts = try a.take(cap * sl * 4);
        ch.pm = try a.take(cap * d * 4);
        ch.gm = try a.take(e.world * cap * d * 4);
        const sz = exl3_experts.Scratch.sizes(cap, sl, ex.dims, ex.width, ex.count);
        var xs: exl3_experts.Scratch = undefined;
        xs.rows = cap;
        xs.slots = sl;
        inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
            @field(xs, f) = try a.take(sz[j]);
        }
        ch.xs = xs;
        // prefill.Workspace: the largest call's rotated input and W_q (fp16), and the Hadamard
        var max_xk: usize = 0;
        var max_kn: usize = 0;
        for ([_]weights.Linear{ l0.wq_a, l0.wkv, l0.wq_b, l0.wo_a[0], l0.wo_b }) |l| {
            max_xk = @max(max_xk, l.k);
            max_kn = @max(max_kn, @as(usize, l.k) * l.n);
        }
        ch.ws = .{ .xh = try a.take(cap * max_xk * 2), .w = try a.take(max_kn * 2), .h = try a.take(128 * 128 * 2) };
        ch.blas_ws = try a.take(cublas.Blas.workspace_bytes);
        // the scratch as experts.py makes it: zeros, the member lists -1
        for ([_]u64{ xs.xg, xs.xu, xs.xd, xs.z, xs.no_y, xs.ids, xs.count, xs.counts }, [_]usize{ sz[0], sz[1], sz[2], sz[3], sz[4], sz[5], sz[6], sz[7] }) |p, n| try e.d.check(e.d.api.cuMemsetD8_v2(p, 0, n), "cuMemsetD8");
        try e.d.check(e.d.api.cuMemsetD32_v2(xs.members, 0xffffffff, sz[8] / 4), "cuMemsetD32");
        try e.d.check(e.d.api.cuMemsetD8_v2(ch.neg, 0xff, cap * 8), "cuMemsetD8");
        var had: [128 * 128]u16 = undefined;
        exl3_prefill.hadamard(&had);
        try e.d.check(e.d.api.cuMemcpyHtoD_v2(ch.ws.h, &had, had.len * 2), "cuMemcpyHtoD");
        return ch;
    }
};

/// The chunk's rows: token ids at positions start .., their streams h [n, hc, D] (every stream the token's embedding
/// row) and pre [n, hc] = (1, 0, 0, 0): embed_init, the bytes of forward()'s embedding rows expanded over the streams.
pub fn begin(e: *const Engine, ch: *Chunk, ids: []const i64, start: usize, host_pos: []i64) !void {
    const n = ids.len;
    if (n > ch.cap or host_pos.len < n) return error.ChunkTooLong;
    ch.n = n;
    ch.start = start;
    ch.pending = false;
    for (host_pos[0..n], 0..) |*p, i| p.* = @intCast(start + i);
    try e.d.check(e.d.api.cuMemcpyHtoD_v2(ch.ids, ids.ptr, n * 8), "cuMemcpyHtoD");
    try e.d.check(e.d.api.cuMemcpyHtoD_v2(ch.pos, host_pos.ptr, n * 8), "cuMemcpyHtoD");
    try tri_basic.embedInit(e.t, e.w.embed, ch.ids, ch.h, ch.pre, n, e.c.hidden, e.c.hc);
}

/// The streams' second buffer becomes the streams (hc_pre_pf wrote the posted ones there) and the old one the spare.
fn swapStreams(ch: *Chunk, src: u64) void {
    std.debug.assert(src == ch.h_alt);
    ch.h_alt = ch.h;
    ch.h = src;
}

/// Layer `li`'s attention mixes: x = RMSNorm of the streams mixed by pre, pre_a / post / comb from their own mixes;
/// with a pending MoE gather, hc_pre_pf posts it into the streams first.
pub fn attnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    if (ch.pending) {
        const src = try tri_hc.hcPrePf(e.t, ch.h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_a, ch.post, ch.comb, ch.part, .{ .gathered = ch.gm, .world = e.world, .h_out = ch.h_alt }, ch.n, c.hidden);
        swapStreams(ch, src);
        ch.pending = false;
    } else {
        try tri_basic.hcPre(e.t, ch.h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_a, ch.post, ch.comb, ch.part, ch.n, c.hidden);
    }
}

/// mm(layer, x, out_dtype) of a prompt chunk (more than 128 rows): the prompt GEMM.
fn mm(e: *const Engine, ch: *const Chunk, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    if (ch.n <= 128) return error.NotPortedYet; // decode-sized rows take the grouped EXL3 linears
    try exl3_prefill.matmul(e.pf, e.t, ch.ws, l, x, .bf16, ldx, out, out_type, os, ch.n);
}

/// attention_k of a window-only layer: the partial pa [n, D] fp32 of this rank's heads; the chunk's keys go into the
/// layer's window ring (bf16 [window + RING_EXTRA, head_dim]). `floor`: no window key before this position.
pub fn attention(e: *const Engine, ch: *Chunk, li: usize, ring: u64, floor: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    if (lay.ratio != 0) return error.NotPortedYet;
    const n = ch.n;
    const start = ch.start;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const hl = e.heads();
    const rope = e.plain;
    const ring_rows = c.window + ring_extra;
    if (n <= ring_extra) return error.NotPortedYet; // ring mode (verify windows)
    // attn_in: wq_a and wkv of x, one prompt GEMM each
    try mm(e, ch, lay.wq_a, ch.x, c.hidden, ch.qa, .bf16, lay.wq_a.n);
    try mm(e, ch, lay.wkv, ch.x, c.hidden, ch.y, .bf16, hd);
    try tri_basic.rmsnorm(e.t, ch.qa, lay.wq_a.n, lay.q_norm, ch.qr, lay.wq_a.n, c.eps, n, lay.wq_a.n);
    try mm(e, ch, lay.wq_b, ch.qr, lay.wq_a.n, ch.q, .bf16, hl * hd);
    try tri_norm.ropeHeads(e.t, ch.q, rope.cos, rope.sin, ch.pos, rd, false, n, hl, hd);
    // the window keys: the ring's last window - 1 positions before start, then this chunk's rows
    const lo = @max(floor, start -| (c.window - 1));
    if (start > lo) return error.NotPortedYet; // a later chunk: the ring rows before it (row gather)
    const wsrc_rows = start - lo + n;
    var lo64: i64 = @intCast(lo);
    try e.d.check(e.d.api.cuMemcpyHtoD_v2(ch.wlo, &lo64, 8), "cuMemcpyHtoD");
    const kv = ch.wsrc + (start - lo) * hd * 2;
    // slots -1: the keys go to wsrc only (ring_mode is off); the ring is written after the attention
    try tri_norm.kvNormRope(e.t, ch.y, lay.kv_norm, rope.cos, rope.sin, ch.pos, ring, ring_rows, ch.neg, c.eps, true, rd, kv, n, hd);
    try tri_attn.sparseAttn(e.t, .{
        .q = ch.q,
        .out = ch.o,
        .rows = n,
        .h = hl,
        .hd = hd,
        .sink = lay.sink,
        .wsrc = ch.wsrc,
        .wsrc_rows = wsrc_rows,
        .wlo = ch.wlo,
        .ring = false,
        .pos = ch.pos,
        .scale = scale(hd),
        .window = c.window,
    });
    try tri_norm.ropeHeads(e.t, ch.o, rope.cos, rope.sin, ch.pos, rd, true, n, hl, hd);
    // ring[pos[-keep:] % R] = kv[-keep:]
    const keep = @min(n, ring_rows);
    if (keep != ring_rows) return error.NotPortedYet; // fewer rows than the ring: a scatter of some slots
    var idx: [256]i64 = undefined;
    for (0..keep) |j| {
        const p = start + n - keep + j;
        idx[p % ring_rows] = @intCast(n - keep + j);
    }
    try e.d.check(e.d.api.cuMemcpyHtoD_v2(ch.ring_idx, &idx, keep * 8), "cuMemcpyHtoD");
    try e.d.check(e.d.api.cuMemsetD8_v2(ch.invalid, 0, 4), "cuMemsetD8");
    try e.ops.gatherRows(e.s, kv, n, ch.ring_idx, ring, hd * 2, keep, ch.invalid);
    // wo_a: each group's column block of o read in place, written into its column block of u; then wo_b to fp32
    const groups = lay.wo_a.len;
    const gk = hl * hd / groups;
    const uw = groups * lay.wo_a[0].n;
    var col: usize = 0;
    for (lay.wo_a, 0..) |wo, g| {
        try mm(e, ch, wo, ch.o + g * gk * 2, hl * hd, ch.u + col * 2, .bf16, uw);
        col += wo.n;
    }
    try mm(e, ch, lay.wo_b, ch.u, uw, ch.pa, .fp32, c.hidden);
}

/// hd ** -0.5 as Triton passes the Python float: rounded to fp32.
pub fn scale(hd: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
}

/// Comm.gather: [world, n, D] fp32 of every rank's partial, in rank order.
pub fn gather(e: *const Engine, ch: *const Chunk, src: u64, dst: u64) !void {
    try e.comm.allGather(src, dst, ch.n * e.c.hidden, .f32, e.s);
}

/// Layer `li`'s FFN mixes: hc_pre_pf with the gathered attention partials posted into the streams first; x the MoE's
/// input rows, pre_f / post / comb the FFN's.
pub fn ffnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const src = try tri_hc.hcPrePf(e.t, ch.h, lay.hc_ffn[0], lay.hc_ffn[1], lay.hc_ffn[2], ch.pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_f, ch.post, ch.comb, ch.part, .{ .gathered = ch.ga, .world = e.world, .h_out = ch.h_alt }, ch.n, c.hidden);
    swapStreams(ch, src);
}

/// Model.moe of a prompt chunk: the gate's logits in fp32 (x.float() @ gate_w.float().t(), one cuBLAS GEMM as torch
/// issues it), the routing (the shared expert in every row's last slot) and the routed experts: the partial pm [n, D].
pub fn moe(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const d = c.hidden;
    const sl = e.slots();
    try e.ops.toF32(e.s, ch.x, ch.xf, n * d);
    try e.ops.f16ToF32(e.s, lay.gate_w, ch.gate_f, c.experts * d);
    try e.blas.xwT(ch.xf, ch.gate_f, ch.logits, n, d, c.experts);
    const shared_id = lay.experts.count - 1;
    try tri_norm.route(e.t, ch.logits, 0, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
    try exl3_experts.prompt(e.ex, e.s, lay.experts, ch.xs, ch.x, d, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
}

/// A layer's end: the MoE's gather is pending (the next layer's attention mixes post it) and the FFN's pre_out is the
/// next layer's pre (_forward_k's `pre, pre_f = pre_f, pre`).
pub fn endLayer(ch: *Chunk) void {
    ch.pending = true;
    std.mem.swap(u64, &ch.pre, &ch.pre_f);
}

test "hd ** -0.5 as the served build passes it" {
    try std.testing.expectEqual(@as(u32, 0x3d3504f3), @as(u32, @bitCast(scale(512))));
}
