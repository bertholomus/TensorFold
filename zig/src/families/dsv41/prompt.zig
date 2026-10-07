//! A prompt chunk's forward on one TP rank, the served build's Model._forward_k (model.py) step by step on the served
//! build's kernels: the streams of the chunk's tokens (embed_init), then a layer at a time the attention mixes (hc_pre, or
//! hc_pre_pf with the previous layer's gathered MoE partials posted first), attention (attention_k), the gather of its
//! partials, the FFN mixes with that post fused in (hc_pre_pf), the MoE and its gather, whose post goes into the next
//! layer's mixes (switch "hc_pf2"); before an Engram layer's mixes, its Engram (engram_apply). Compressed layers
//! (ratio 1, 2): a kv source's compressor and index keys into the caches (kv_source_update), the indexer's top-k
//! (prompt_keys: scored keys and the selection; from the candidate source on: fp32 scores and topk_indices), and the
//! sparse attention over the window (a later chunk's first keys from the ring) and the selected latents. Decode-sized
//! rows (1..128) take the grouped EXL3 linears. With replay (bounded replay prefill) the decoder's first layer cuts the
//! chunk to its rows at or past replay (replayCut); DSpark taps (tap) and the head on the last row (head) end it.
//! Not yet: ring mode (verify windows), a pool that leaves blocks out, a chunk starting inside a compressor group;
//! those return error.NotPortedYet.
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
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const exact = @import("exact.zig");
const tri_index = @import("tri_index.zig");
const exl3_linear = @import("exl3_linear.zig");
const prompt2d = @import("prompt2d.zig");

/// model.py's RING_EXTRA: window ring slots beyond the window (a verify window never clobbers a key it reads).
pub const ring_extra = 16;
/// model.py's RAW: per-position compressor inputs kept (a ratio-2 group's earlier row for the next chunk).
pub const raw_rows = 64;

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
    lin: *const exl3_linear.Kernels,
    ex: *const exl3_experts.Kernels,
    ops: *const ops.Ops,
    exact: *const exact.Exact,
    c: *const Config,
    w: *const weights.Weights,
    world: usize,
    plain: Rope,
    compressed: Rope,
    two: ?*const prompt2d.Two = null, // a node of the four-node 2D split (prompt2d.zig); `world` stays TP2's

    fn heads(e: *const Engine) usize {
        if (e.two) |t| return t.heads;
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
    // compressed layers
    kvc: u64 = 0, // fp32 [cap, head_dim]: the compressor's kv projection
    scc: u64 = 0, // fp32 [cap, head_dim]: its gate scores
    lat: u64 = 0, // bf16 [cap, head_dim]: the compressed latents
    lat2: u64 = 0, // bf16 [cap, head_dim]: their RoPE'd copy (the cache's)
    ik: u64 = 0, // bf16 [cap, index_head_dim]: the index keys
    groups: u64 = 0, // int64 [cap]: the new latents' groups
    gpos: u64 = 0, // int64 [cap]: their positions (group * ratio)
    raw_idx: u64 = 0, // int64 [RAW]
    iq: u64 = 0, // bf16 [cap, index_heads * index_head_dim]
    iq4: u64 = 0, // bf16 [cap, index_heads * index_head_dim]: fp4_qd's bytes
    wl: u64 = 0, // fp32 [cap, index_heads]
    iw: u64 = 0, // bf16 [cap, index_heads]: the heads' weights
    vis: u64 = 0, // int64 [cap]: compressed entries a row sees
    keys: u64 = 0, // int64 [cap, max_comp]
    score: u64 = 0, // fp32 [cap, max_comp]: the candidate pool's layers' index scores
    tmax: u64 = 0, // int64 [cap, cdiv(max_comp, 64)]
    cidx: u64 = 0, // int64 [cap, index_topk]: the selected latents
    max_comp: usize = 0,
    // MoE
    xf: u64, // fp32 [cap, D]
    gate_f: u64, // fp32 [experts, D]
    logits: u64, // fp32 [cap, experts]
    pick: u64, // int32 [cap, slots]
    wts: u64, // fp32 [cap, slots]
    pm: u64, // fp32 [cap, D]: the MoE partial
    gm: u64, // fp32 [world, cap, D]: its gather (the pending post)
    xs: exl3_experts.Scratch,
    // Engram (zero when no layer has it)
    eb: u64 = 0, // bf16 [cap, cols * engram_head_dim]: the rows read
    ek: u64 = 0, // fp32 [cap, hc * D + D]: their projection, this rank's columns
    ekg: u64 = 0, // fp32 [world, cap, hc * D + D]: its gather
    kv: u64 = 0, // bf16 [cap, hc * D + D]: the ranks' sum
    ws: exl3_prefill.Workspace,
    blas_ws: u64, // cuBLAS workspace
    // the grouped EXL3 linears of decode-sized rows (1..128): rotated rows, split-K partials, each layer's counters
    gxh: u64 = 0, // fp16 [128, the largest group's sum of K]
    gz: u64 = 0, // fp32 [the largest group's split-K partials at 128 rows]
    counters: [640]Counter = undefined,
    n_counters: usize = 0,
    ktop: u64 = 0, // int64 [cap, index_topk]: torch's top-k values for _topk_finish
    // the DSpark taps and the head
    taps: u64 = 0, // bf16 [taps, cap, D]: each tap layer's mean of the streams (Model.forward's taps)
    head_x: u64 = 0, // bf16 [1, D]: collapse_norm of the last row
    head_l: u64 = 0, // fp32 [1, this rank's vocabulary columns]: its logits
    head_g: u64 = 0, // fp32 [world, 1, columns]: their gather (the prompt's logits, rank after rank)
    pending: bool = false, // gm holds a MoE gather whose post is not in h yet

    /// Every buffer for `cap` rows from `a`, the compressed layers' for chunks ending by position `tokens`; the scratch
    /// the served build zeros (or fills with -1) is set the same.
    pub fn init(e: *const Engine, a: *Arena, cap: usize, tokens: usize) !Chunk {
        const c = e.c;
        const d = c.hidden;
        const hc = c.hc;
        const l0 = e.w.layers[0];
        const hl = e.heads();
        const hd = c.head_dim;
        const groups = l0.groups;
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
        // compressed layers: the smallest ratio sets the most compressed entries a row can see
        var min_ratio: usize = 0;
        for (e.w.layers) |lay| {
            if (lay.ratio != 0 and (min_ratio == 0 or lay.ratio < min_ratio)) min_ratio = lay.ratio;
        }
        if (min_ratio != 0) {
            const ih = c.index_heads * c.index_head_dim;
            ch.max_comp = tokens / min_ratio;
            ch.kvc = try a.take(cap * c.head_dim * 4);
            ch.scc = try a.take(cap * c.head_dim * 4);
            ch.lat = try a.take(cap * c.head_dim * 2);
            ch.lat2 = try a.take(cap * c.head_dim * 2);
            ch.ik = try a.take(cap * c.index_head_dim * 2);
            ch.groups = try a.take(cap * 8);
            ch.gpos = try a.take(cap * 8);
            ch.raw_idx = try a.take(raw_rows * 8);
            ch.iq = try a.take(cap * ih * 2);
            ch.iq4 = try a.take(cap * ih * 2);
            ch.wl = try a.take(cap * c.index_heads * 4);
            ch.iw = try a.take(cap * c.index_heads * 2);
            ch.vis = try a.take(cap * 8);
            ch.keys = try a.take(cap * ch.max_comp * 8);
            ch.score = try a.take(cap * ch.max_comp * 4);
            ch.tmax = try a.take(cap * ((ch.max_comp + tri_index.tile - 1) / tri_index.tile) * 8);
            ch.cidx = try a.take(cap * c.index_topk * 8);
            ch.ktop = try a.take(cap * c.index_topk * 8);
        }
        for (e.w.layers) |lay| {
            const ew = lay.engram_wkv orelse continue;
            const en = if (e.two) |t| t.engramWidth() else ew.n; // the summed projection's width (2D: both pairs')
            ch.eb = try a.take(cap * ew.k * 2);
            ch.ek = try a.take(cap * ew.n * 4);
            ch.ekg = try a.take(e.world * cap * en * 4);
            ch.kv = try a.take(cap * en * 2);
            break;
        }
        const sz = exl3_experts.Scratch.sizes(cap, sl, ex.dims, ex.width, ex.count);
        var xs: exl3_experts.Scratch = undefined;
        xs.rows = cap;
        xs.slots = sl;
        inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
            @field(xs, f) = try a.take(sz[j]);
        }
        ch.xs = xs;
        // prefill.Workspace: the largest prompt GEMM's rotated input and W_q (fp16) over every layer, and the Hadamard
        var max_xk: usize = 0;
        var max_kn: usize = 0;
        for (e.w.layers) |lay| {
            var ls: [16]?weights.Linear = @splat(null);
            ls[0] = lay.wq_a;
            ls[1] = lay.wkv;
            ls[2] = lay.wq_b;
            ls[3] = lay.wo_b;
            for (lay.wo_a[0..lay.groups], 0..) |wo, g| ls[4 + g] = wo;
            ls[8] = lay.comp_wkv;
            ls[9] = lay.comp_wgate;
            ls[10] = lay.idx_wq_b;
            ls[11] = lay.idx_wk;
            ls[12] = lay.engram_wkv;
            for (ls) |ol| {
                const l = ol orelse continue;
                max_xk = @max(max_xk, l.k);
                max_kn = @max(max_kn, @as(usize, l.k) * l.n);
            }
        }
        ch.ws = .{ .xh = try a.take(cap * max_xk * 2), .w = try a.take(max_kn * 2), .h = try a.take(128 * 128 * 2) };
        ch.blas_ws = try a.take(cublas.Blas.workspace_bytes);
        // grouped linears: every layer's counters (int32 [8 N / 128], zeros, left zero by each launch), the largest
        // group's rotated rows and partials at 128 rows
        var max_gk: usize = 0;
        var max_gz: usize = 0;
        ch.n_counters = 0;
        for (e.w.layers) |lay| {
            const in_group = [_]weights.Linear{ lay.wq_a, lay.wkv };
            max_gk = @max(max_gk, lay.wq_a.k + lay.wkv.k);
            max_gz = @max(max_gz, exl3_linear.zFloats(&in_group, 128));
            var wo_k: usize = 0;
            for (lay.wo_a[0..lay.groups]) |wo| wo_k += wo.k;
            max_gk = @max(max_gk, wo_k);
            max_gz = @max(max_gz, exl3_linear.zFloats(lay.wo_a[0..lay.groups], 128));
            var ls: [16]?weights.Linear = @splat(null);
            ls[0] = lay.wq_a;
            ls[1] = lay.wkv;
            ls[2] = lay.wq_b;
            ls[3] = lay.wo_b;
            for (lay.wo_a[0..lay.groups], 0..) |wo, g| ls[4 + g] = wo;
            ls[8] = lay.comp_wkv;
            ls[9] = lay.comp_wgate;
            ls[10] = lay.idx_wq_b;
            ls[11] = lay.idx_wk;
            ls[12] = lay.engram_wkv;
            for (ls) |ol| {
                const l = ol orelse continue;
                max_gk = @max(max_gk, l.k);
                max_gz = @max(max_gz, exl3_linear.zFloats(&[_]weights.Linear{l}, 128));
                if (ch.n_counters == ch.counters.len) return error.TooManyLinears;
                const bytes = 8 * (l.n / 128) * 4;
                const ptr = try a.take(bytes);
                try fill(e, ptr, 0, bytes);
                ch.counters[ch.n_counters] = .{ .words = l.words, .ptr = ptr };
                ch.n_counters += 1;
            }
        }
        {
            // the head: one row through its own group
            const l = e.w.head;
            max_gk = @max(max_gk, l.k);
            max_gz = @max(max_gz, exl3_linear.zFloats(&[_]weights.Linear{l}, 128));
            if (ch.n_counters == ch.counters.len) return error.TooManyLinears;
            const bytes = 8 * (l.n / 128) * 4;
            const ptr = try a.take(bytes);
            try fill(e, ptr, 0, bytes);
            ch.counters[ch.n_counters] = .{ .words = l.words, .ptr = ptr };
            ch.n_counters += 1;
            ch.head_x = try a.take(d * 2);
            ch.head_l = try a.take(l.n * 4);
            const hn = if (e.two) |t| t.hw[0] + t.hw[1] else l.n; // a rank's vocabulary half (2D: both pairs' parts)
            ch.head_g = try a.take(e.world * hn * 4);
        }
        ch.taps = try a.take(@max(c.dspark_taps.slice().len, 1) * cap * d * 2);
        ch.gxh = try a.take(128 * max_gk * 2);
        ch.gz = try a.take(@max(max_gz, 1) * 4);
        // the scratch as experts.py makes it: zeros, the member lists -1
        for ([_]u64{ xs.xg, xs.xu, xs.xd, xs.z, xs.no_y, xs.ids, xs.count, xs.counts }, [_]usize{ sz[0], sz[1], sz[2], sz[3], sz[4], sz[5], sz[6], sz[7] }) |p, n| try fill(e, p, 0, n);
        try fill32(e, xs.members, 0xffffffff, sz[8] / 4);
        try fill(e, ch.neg, 0xff, cap * 8);
        var had: [128 * 128]u16 = undefined;
        exl3_prefill.hadamard(&had);
        try upload(e, ch.ws.h, &had, had.len * 2);
        return ch;
    }
};

/// Host bytes to the device in the stream's order (the stream is non-blocking: the legacy stream's synchronous copies
/// would not wait for its kernels). Pageable host memory is staged before the call returns.
fn upload(e: *const Engine, dst: u64, src: *const anyopaque, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, src, bytes, e.s.handle), "cuMemcpyHtoDAsync");
}

/// cuMemsetD8 in the stream's order.
fn fill(e: *const Engine, dst: u64, value: u8, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemsetD8Async(dst, value, bytes, e.s.handle), "cuMemsetD8Async");
}

/// cuMemsetD32 in the stream's order.
fn fill32(e: *const Engine, dst: u64, value: u32, words: usize) !void {
    try e.d.check(e.d.api.cuMemsetD32Async(dst, value, words, e.s.handle), "cuMemsetD32Async");
}

/// A layer's split-K counters (Exl3Linear.counters) by its trellis words.
pub const Counter = struct { words: u64, ptr: u64 };

fn counterOf(ch: *const Chunk, l: weights.Linear) !u64 {
    for (ch.counters[0..ch.n_counters]) |c| if (c.words == l.words) return c.ptr;
    return error.NoCounters;
}

/// Exl3Group(ls) on m (1..128) rows (linear.py): one rot_many launch rotating every layer's input rows xs[i] (bf16,
/// row stride ldxs[i]) into the group's buffer, then a glinear launch for each (bits, warps) set, outputs into outs[i]
/// (row stride ldos[i], dtype types[i]); programmatic dependent launches, split-K partials dropped from L2 (the served
/// TF_EXL3_PDL and TF_EXL3_L2_DISCARD).
fn grouped(e: *const Engine, ch: *const Chunk, m: usize, ls: []const weights.Linear, xs: []const u64, ldxs: []const usize, outs: []const u64, ldos: []const usize, types: []const exl3_linear.DType) !void {
    if (m == 0 or m > 128 or ls.len > 8) return error.NotADecodeWindow;
    var calls: [8]exl3_linear.Call = undefined;
    var off: usize = 0;
    for (ls, 0..) |l, i| {
        calls[i] = .{ .layer = l, .x = xs[i], .ldx = @intCast(ldxs[i]), .x_dtype = .bf16, .xh = ch.gxh + off * 2, .y = outs[i], .ldy = @intCast(ldos[i]), .y_dtype = types[i], .counters = try counterOf(ch, l) };
        off += m * l.k;
    }
    try exl3_linear.rotMany(e.lin, e.s, calls[0..ls.len], m, true);
    try exl3_linear.glinear(e.lin, e.s, calls[0..ls.len], m, ch.gz, true, true);
}

/// One sequence's compressed-attention caches (model.py SeqCache: comp, index_k, comp_raw), by kv-source layer: the
/// compressed latents as packed FP4 (codes [rows, head_dim / 2], E8M0 scales [rows, head_dim / 16]), the index keys
/// (codes [rows, index_head_dim / 2], scales [rows, index_head_dim / 32], 127 when unwritten) and, at ratio 2, the
/// positional store of the compressor's raw kv and scores (fp32 [RAW, head_dim] each). rows = cap / ratio + 2.
pub const Caches = struct {
    comp_codes: [64]u64 = @splat(0),
    comp_scales: [64]u64 = @splat(0),
    idx_codes: [64]u64 = @splat(0),
    idx_scales: [64]u64 = @splat(0),
    raw_kv: [64]u64 = @splat(0),
    raw_score: [64]u64 = @splat(0),

    pub fn init(e: *const Engine, a: *Arena, cap: usize) !Caches {
        const c = e.c;
        var cs: Caches = .{};
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            const rows = cap / r + 2;
            cs.comp_codes[i] = try a.take(rows * c.head_dim / 2);
            cs.comp_scales[i] = try a.take(rows * c.head_dim / 16);
            try fill(e, cs.comp_codes[i], 0, rows * c.head_dim / 2);
            try fill(e, cs.comp_scales[i], 0, rows * c.head_dim / 16);
            if (lay.idx_wk != null) {
                cs.idx_codes[i] = try a.take(rows * c.index_head_dim / 2);
                cs.idx_scales[i] = try a.take(rows * c.index_head_dim / 32);
                try fill(e, cs.idx_codes[i], 0, rows * c.index_head_dim / 2);
                try fill(e, cs.idx_scales[i], 127, rows * c.index_head_dim / 32);
            }
            if (r > 1) {
                cs.raw_kv[i] = try a.take(raw_rows * c.head_dim * 4);
                cs.raw_score[i] = try a.take(raw_rows * c.head_dim * 4);
                try fill(e, cs.raw_kv[i], 0, raw_rows * c.head_dim * 4);
                try fill(e, cs.raw_score[i], 0, raw_rows * c.head_dim * 4);
            }
        }
        return cs;
    }
};

/// What one forward carries from layer to layer (model.py _forward_k's `shared`): the latest kv-source layer, and the
/// latest indexer's selection (in the chunk's cidx, kk a row).
pub const Shared = struct { kv_layer: ?usize = null, kk: ?usize = null };

/// The chunk's rows: token ids at positions start .., their streams h [n, hc, D] (every stream the token's embedding
/// row) and pre [n, hc] = (1, 0, 0, 0): embed_init, the bytes of forward()'s embedding rows expanded over the streams.
pub fn begin(e: *const Engine, ch: *Chunk, ids: []const i64, start: usize, host_pos: []i64) !void {
    const n = ids.len;
    if (n > ch.cap or host_pos.len < n) return error.ChunkTooLong;
    ch.n = n;
    ch.start = start;
    ch.pending = false;
    for (host_pos[0..n], 0..) |*p, i| p.* = @intCast(start + i);
    try upload(e, ch.ids, ids.ptr, n * 8);
    try upload(e, ch.pos, host_pos.ptr, n * 8);
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

/// mm(layer, x, out_dtype) of m rows: the prompt GEMM beyond 128 rows, the layer's own group (GROUPED) up to 128.
fn mmRows(e: *const Engine, ch: *const Chunk, m: usize, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    if (m > 128) return exl3_prefill.matmul(e.pf, e.t, ch.ws, l, x, .bf16, ldx, out, out_type, os, m);
    const dt: exl3_linear.DType = switch (out_type) {
        .bf16 => .bf16,
        .fp32 => .f32,
    };
    try grouped(e, ch, m, &.{l}, &.{x}, &.{ldx}, &.{out}, &.{os}, &.{dt});
}

/// mm of the chunk's rows.
pub fn mm(e: *const Engine, ch: *const Chunk, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    try mmRows(e, ch, ch.n, l, x, ldx, out, out_type, os);
}

/// attention_k: the partial pa [n, D] fp32 of this rank's heads; the chunk's keys go into the layer's window ring
/// (bf16 [window + RING_EXTRA, head_dim]). A compressed layer also attends to its kv source's latents the indexer picks
/// (a kv source first compresses the chunk into them). `floor`: no window key before this position.
pub fn attention(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, ring: u64, floor: usize, kv_done: bool) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const start = ch.start;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const hl = e.heads();
    // _cs: the layer's RoPE kind (compressed layers rotate by the compressed table) for every rotation it makes
    const rope = if (lay.ratio != 0) e.compressed else e.plain;
    const ring_rows = c.window + ring_extra;
    if (n <= ring_extra) return error.NotPortedYet; // ring mode (verify windows)
    // attn_in: wq_a and wkv of x, one prompt GEMM each, or one group up to 128 rows
    if (n > 128) {
        try mm(e, ch, lay.wq_a, ch.x, c.hidden, ch.qa, .bf16, lay.wq_a.n);
        try mm(e, ch, lay.wkv, ch.x, c.hidden, ch.y, .bf16, hd);
    } else {
        try grouped(e, ch, n, &.{ lay.wq_a, lay.wkv }, &.{ ch.x, ch.x }, &.{ c.hidden, c.hidden }, &.{ ch.qa, ch.y }, &.{ lay.wq_a.n, hd }, &.{ .bf16, .bf16 });
    }
    try tri_basic.rmsnorm(e.t, ch.qa, lay.wq_a.n, lay.q_norm, ch.qr, lay.wq_a.n, c.eps, n, lay.wq_a.n);
    try mm(e, ch, lay.wq_b, ch.qr, lay.wq_a.n, ch.q, .bf16, hl * hd);
    try tri_norm.ropeHeads(e.t, ch.q, rope.cos, rope.sin, ch.pos, rd, false, n, hl, hd);
    // the window keys: the ring's last window - 1 positions before start, then this chunk's rows
    const lo = @max(floor, start -| (c.window - 1));
    const wsrc_rows = start - lo + n;
    if (start > lo) {
        // wsrc[:start - lo] = ring[arange(lo, start) % R]
        var ridx: [256]i64 = undefined;
        if (start - lo > ridx.len) return error.WindowTooLong;
        for (0..start - lo) |j| ridx[j] = @intCast((lo + j) % ring_rows);
        try upload(e, ch.ring_idx, &ridx, (start - lo) * 8);
        try fill(e, ch.invalid, 0, 4);
        try e.ops.gatherRows(e.s, ring, ring_rows, ch.ring_idx, ch.wsrc, hd * 2, start - lo, ch.invalid);
    }
    var lo64: i64 = @intCast(lo);
    try upload(e, ch.wlo, &lo64, 8);
    const kv = ch.wsrc + (start - lo) * hd * 2;
    // slots -1: the keys go to wsrc only (ring_mode is off); the ring is written after the attention
    try tri_norm.kvNormRope(e.t, ch.y, lay.kv_norm, rope.cos, rope.sin, ch.pos, ring, ring_rows, ch.neg, c.eps, true, rd, kv, n, hd);
    var comp: tri_attn.Comp = .none;
    var n_idx: usize = 0;
    if (lay.ratio != 0) {
        const r: usize = lay.ratio;
        if (lay.comp_wkv != null and !kv_done) try kvSourceUpdate(e, ch, cs, sh, li);
        const src = sh.kv_layer orelse return error.NoKvSource;
        const n_comp_end = (start + n) / r;
        // vis = (pos + 1) // ratio: the compressed entries row i may see (host-made: the same integers)
        var vis_host: [4096]i64 = undefined;
        if (n > vis_host.len) return error.ChunkTooLong;
        for (0..n) |i| vis_host[i] = @intCast((start + i + 1) / r);
        try upload(e, ch.vis, &vis_host, n * 8);
        if (lay.idx_wq_b) |wqb| try indexer(e, ch, cs, sh, li, wqb, src, n_comp_end, rope);
        const kk = sh.kk orelse return error.NoIndexerSelection;
        comp = .{ .fp4 = .{ .codes = cs.comp_codes[src], .scales = cs.comp_scales[src] } };
        n_idx = kk;
        if (kk % 16 != 0) return error.NotPortedYet; // a prompt chunk's idx padded to whole 16-column tiles
    }
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
        .comp = comp,
        .idx = if (n_idx > 0) ch.cidx else 0,
        .n_idx = n_idx,
        .pos = ch.pos,
        .scale = scale(hd),
        .window = c.window,
    });
    try tri_norm.ropeHeads(e.t, ch.o, rope.cos, rope.sin, ch.pos, rd, true, n, hl, hd);
    // ring[pos[-keep:] % R] = kv[-keep:]
    const keep = @min(n, ring_rows);
    var idx: [256]i64 = undefined;
    for (0..keep) |j| idx[j] = @intCast((start + n - keep + j) % ring_rows);
    try upload(e, ch.ring_idx, &idx, keep * 8);
    try e.ops.scatterRows(e.s, kv + (n - keep) * hd * 2, hd * 2, ch.ring_idx, ring, hd * 2, hd * 2, keep);
    // wo_a: each group's column block of o read in place, written into its column block of u; then wo_b to fp32
    const groups = lay.groups;
    const gk = hl * hd / groups;
    const uw = groups * lay.wo_a[0].n;
    var col: usize = 0;
    if (n > 128) {
        for (lay.wo_a[0..groups], 0..) |wo, g| {
            try mm(e, ch, wo, ch.o + g * gk * 2, hl * hd, ch.u + col * 2, .bf16, uw);
            col += wo.n;
        }
    } else {
        // one group: the column blocks of o read in place, written into u's column blocks
        var xs: [4]u64 = undefined;
        var outs: [4]u64 = undefined;
        for (lay.wo_a[0..groups], 0..) |wo, g| {
            xs[g] = ch.o + g * gk * 2;
            outs[g] = ch.u + col * 2;
            col += wo.n;
        }
        const lds = [_]usize{ hl * hd, hl * hd, hl * hd, hl * hd };
        const oss = [_]usize{ uw, uw, uw, uw };
        const dts = [_]exl3_linear.DType{ .bf16, .bf16, .bf16, .bf16 };
        try grouped(e, ch, n, lay.wo_a[0..groups], xs[0..groups], lds[0..groups], outs[0..groups], oss[0..groups], dts[0..groups]);
    }
    if (e.two) |t| return t.woB(e, ch, lay); // 2D: the column partner's groups first, then this pair's columns
    try mm(e, ch, lay.wo_b, ch.u, uw, ch.pa, .fp32, c.hidden);
}

/// kv_source_update of a kv-source layer on the chunk (no earlier pending row: a chunk starting on a group boundary):
/// _compress's new latents (ratio 1: RMSNorm of the bf16 projection; ratio 2: of the pairs' softmax-weighted kv), its
/// positional store's last RAW rows, then the index keys (RMSNorm of idx_wk's projection, RoPE) and the RoPE'd latents
/// into the caches as FP4 (fp4_store).
fn kvSourceUpdate(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const start = ch.start;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const r: usize = lay.ratio;
    const cwkv = lay.comp_wkv.?;
    if (start % r != 0) return error.NotPortedYet; // the group's earlier rows from the positional store
    const full = n / r;
    if (r == 1) {
        try mm(e, ch, cwkv, ch.x, c.hidden, ch.lat2, .bf16, hd);
        try e.exact.rmsNorm(e.s, ch.lat2, hd, lay.comp_norm, ch.lat, hd, n, hd, c.eps);
    } else {
        if (r != 2) return error.NotPortedYet;
        try mm(e, ch, cwkv, ch.x, c.hidden, ch.kvc, .fp32, hd);
        try mm(e, ch, lay.comp_wgate.?, ch.x, c.hidden, ch.scc, .fp32, hd);
        // rk[pw] = kv[-keep:], rs[pw] = score[-keep:]: pw = positions start + n - keep .. % RAW
        const keep: usize = @min(n, raw_rows);
        var idx: [raw_rows]i64 = undefined;
        for (0..keep) |j| idx[j] = @intCast((start + n - keep + j) % raw_rows);
        try upload(e, ch.raw_idx, &idx, keep * 8);
        try e.ops.scatterRows(e.s, ch.kvc + (n - keep) * hd * 4, hd * 4, ch.raw_idx, cs.raw_kv[li], hd * 4, hd * 4, keep);
        try e.ops.scatterRows(e.s, ch.scc + (n - keep) * hd * 4, hd * 4, ch.raw_idx, cs.raw_score[li], hd * 4, hd * 4, keep);
        if (full == 0) return error.NotPortedYet;
        try e.exact.compress2(e.s, ch.kvc, ch.scc, ch.lat2, full, hd);
        try e.exact.rmsNorm(e.s, ch.lat2, hd, lay.comp_norm, ch.lat, hd, full, hd, c.eps);
    }
    sh.kv_layer = li;
    // groups first // r .. and their positions group * r (the rotations _rot reads)
    var g_host: [4096]i64 = undefined;
    var p_host: [4096]i64 = undefined;
    if (full > g_host.len) return error.ChunkTooLong;
    for (0..full) |j| {
        g_host[j] = @intCast(start / r + j);
        p_host[j] = @intCast((start / r + j) * r);
    }
    try upload(e, ch.groups, &g_host, full * 8);
    try upload(e, ch.gpos, &p_host, full * 8);
    const rot = e.compressed;
    if (lay.idx_wk) |iwk| {
        const id = c.index_head_dim;
        try mmRows(e, ch, full, iwk, ch.lat, hd, ch.lat2, .bf16, id);
        try e.exact.rmsNorm(e.s, ch.lat2, id, lay.idx_k_norm, ch.ik, id, full, id, c.eps);
        try e.exact.rope(e.s, ch.ik, id, id - rd, rot.cos, rot.sin, ch.gpos, full, rd / 2, false);
        // switch "comp": the packed cache's rows in one launch (fp4_store: store_rows' bytes)
        try tri_attn.fp4Store(e.t, ch.ik, id, cs.idx_codes[li], cs.idx_scales[li], ch.groups, full, id, 32, false);
    }
    try e.ops.copyRows(e.s, ch.lat, hd * 2, ch.lat2, hd * 2, hd * 2, full);
    try e.exact.rope(e.s, ch.lat2, hd, hd - rd, rot.cos, rot.sin, ch.gpos, full, rd / 2, false);
    try tri_attn.fp4Store(e.t, ch.lat2, hd, cs.comp_codes[li], cs.comp_scales[li], ch.groups, full, hd, 16, true);
}

/// The indexer of a layer with index queries (switch "prompt_keys", layers before the candidate source): the
/// queries (idx_wq_b of q's latent, RoPE, fp4_qd's bytes), the heads' weights (x.float() @ idx_proj.t() through cuBLAS,
/// to bf16, times idx_dim ** -0.5 * idx_heads ** -0.5), each row's int64 keys of the kv source's index keys and the
/// top-k selection into cidx.
fn indexer(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, wqb: weights.Linear, src: usize, n_comp_end: usize, rope: Rope) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const ih = c.index_heads;
    const id = c.index_head_dim;
    if (n_comp_end == 0) return error.NotPortedYet; // cidx [n, 0]
    if (n_comp_end > ch.max_comp) return error.ChunkTooLong;
    try mm(e, ch, wqb, ch.qr, lay.wq_a.n, ch.iq, .bf16, ih * id);
    try tri_norm.ropeHeads(e.t, ch.iq, rope.cos, rope.sin, ch.pos, c.rope_dim, false, n, ih, id);
    // switch "idx": fp4_qd's bytes in one launch
    try tri_attn.fp4QdP2(e.t, ch.iq, ch.iq4, n * ih * id);
    if (n <= tri_index.decode_rows) return error.NotPortedYet; // rowmm2 of the fp16 projection
    try e.ops.toF32(e.s, ch.x, ch.xf, n * c.hidden);
    try e.blas.xwT(ch.xf, lay.idx_proj, ch.wl, n, c.hidden, ih);
    try e.exact.bf16Scale(e.s, ch.wl, ch.iw, n * ih, exact.indexScale(id, ih));
    const kk = @min(c.index_topk, n_comp_end);
    // rows in blocks so the [rows, n_comp] keys stay bounded at long contexts
    const rb = @max(16, @min(n, (@as(usize, 1) << 26) / (4 * @max(n_comp_end, 1))));
    if (rb < n) return error.NotPortedYet; // more than one row block
    const keyed = li != c.candidate_source and !(c.candidate_source < li) and kk & (kk - 1) == 0;
    const k: tri_index.IndexK = .{ .fp4 = .{ .codes = cs.idx_codes[src], .scales = cs.idx_scales[src] } };
    if (!keyed) {
        // the candidate source and the layers after it: fp32 scores, the pool (_candidates at the source,
        // apply_candidates after it), topk_indices and -1 past vis. While every block of the scores fits in the pool
        // (cand_blocks of them) the pool keeps each block with a finite score (the newest pinned: it holds the row's
        // last visible key) and apply_candidates rewrites only scores already -inf: no launch changes a byte.
        const nb = (n_comp_end + c.candidate_block - 1) / c.candidate_block;
        if (nb > c.candidate_blocks) return error.NotPortedYet; // a pool that leaves blocks out
        try tri_index.indexScore(e.t, ch.iq4, k, ch.iw, ch.vis, n_comp_end, ch.score, null, false, null, false, n, ih, id);
        try e.ops.topkIndices(e.s, ch.score, n_comp_end, n, n_comp_end, kk, ch.vis, ch.cidx);
        sh.kk = kk;
        return;
    }
    try tri_index.indexScore(e.t, ch.iq4, k, ch.iw, ch.vis, n_comp_end, ch.keys, null, true, ch.tmax, false, n, ih, id);
    const torch_topk: tri_index.TopK = .{ .top = ch.ktop, .ctx = @ptrCast(@constCast(e)), .run = torchTopk };
    if (tri_index.prunes(n_comp_end, kk)) return error.NotPortedYet; // the pruned search's buffers
    try tri_index.topkSelectPruned(e.t, ch.keys, n_comp_end, ch.tmax, kk, ch.vis, ch.cidx, .{ .tpos = 0, .cand = 0, .every = 0 }, torch_topk, n, n_comp_end);
    sh.kk = kk;
}

/// topk_select's torch step, keys.topk(k).values: the top-k set of the unique int64 keys (ops.topkI64).
fn torchTopk(ctx: ?*anyopaque, _: tri.Tri, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) anyerror!void {
    const e: *const Engine = @ptrCast(@alignCast(ctx.?));
    try e.ops.topkI64(e.s, keys, ks, rows, n, k, top);
}

/// hd ** -0.5 as Triton passes the Python float: rounded to fp32.
pub fn scale(hd: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
}

/// Comm.gather: [world, n, D] fp32 of every rank's partial, in rank order.
pub fn gather(e: *const Engine, ch: *const Chunk, src: u64, dst: u64) !void {
    if (e.two) |t| return t.quarters(e, src, dst, ch.n, t.ow, 4); // 2D: the same [2, n, D] from the four quarters
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
    if (e.two) |t| return t.experts(e, ch, lay); // 2D: gate / up here, the intermediate's exchange, down here
    try exl3_experts.prompt(e.ex, e.s, lay.experts, ch.xs, ch.x, d, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
}

/// Engram on the host: the hasher (the compressed token map, the multipliers), this rank's hash columns [lo, hi) of
/// each Engram layer's, its tables and readers, and host buffers for a chunk's rows (model.py's Engram).
pub const EngramHost = struct {
    hasher: engram.Hasher,
    tables: *const engram_io.Tables,
    pool: *engram_io.Pool,
    lo: usize,
    hi: usize,
    hashes: []i64, // [cap][layers][cols]
    flat: []i64, // [cap * (hi - lo)]
    w: []u8, // FP8 rows
    s: []u8, // their E8M0 scales
    rows: []u16, // bf16 [cap, (hi - lo) * head_dim]

    pub fn init(gpa: std.mem.Allocator, c: *const Config, hasher: engram.Hasher, tables: *const engram_io.Tables, pool: *engram_io.Pool, rank: usize, world: usize, cap: usize) !EngramHost {
        const cols = hasher.cols();
        const lo = rank * cols / world;
        const hi = (rank + 1) * cols / world;
        const k = hi - lo;
        const t = tables.layers.get(@intCast(c.engram_layers.slice()[0])) orelse return error.MissingEngramTable;
        return .{
            .hasher = hasher,
            .tables = tables,
            .pool = pool,
            .lo = lo,
            .hi = hi,
            .hashes = try gpa.alloc(i64, cap * hasher.layers * cols),
            .flat = try gpa.alloc(i64, cap * k),
            .w = try gpa.alloc(u8, cap * k * t.row_w),
            .s = try gpa.alloc(u8, cap * k * t.row_s),
            .rows = try gpa.alloc(u16, cap * k * t.row_w),
        };
    }
};

/// Layer li's Engram on the chunk (engram_apply, no image span), as _forward_k runs it before the attention mixes: a
/// pending MoE post into the streams first (hc_post), then the rows of each position's n-grams (hashed on the host from
/// `seq`, the whole sequence's ids so far; read from the tables; decoded to bf16 as Engram._decode does), their
/// projection summed over the ranks in rank order, rounded to bf16, and the gate into the streams.
pub fn engramApply(e: *const Engine, ch: *Chunk, eh: *EngramHost, li: usize, seq: []const i32) !void {
    const lay = e.w.layers[li];
    const ew = lay.engram_wkv orelse return;
    const c = e.c;
    const n = ch.n;
    if (ch.pending) {
        try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, n, c.hidden);
        ch.pending = false;
    }
    const li32: u16 = @intCast(li);
    const l = std.mem.indexOfScalar(u16, c.engram_layers.slice(), li32) orelse return error.NotAnEngramLayer;
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    eh.hasher.hashes(seq, ch.start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
    for (0..n) |r| {
        for (0..k) |j| eh.flat[r * k + j] = eh.hashes[(r * eh.hasher.layers + l) * cols + eh.lo + j];
    }
    const m = n * k;
    try eh.pool.gather(t, eh.flat[0..m], eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s]);
    for (0..m) |i| engram.decodeRow(eh.w[i * t.row_w ..][0..t.row_w], eh.s[i * t.row_s ..][0..t.row_s], eh.rows[i * t.row_w ..][0..t.row_w]);
    if (m * t.row_w != n * ew.k) return error.EngramShape;
    try upload(e, ch.eb, eh.rows.ptr, m * t.row_w * 2);
    try mm(e, ch, ew, ch.eb, ew.k, ch.ek, .fp32, ew.n);
    // 2D (model.py esum): this node's columns of its rank's projection; the quarters give both ranks' whole ones
    const en = if (e.two) |two| two.engramWidth() else ew.n;
    if (e.two) |two| try two.quarters(e, ch.ek, ch.ekg, n, two.ew, 4) else try e.comm.allGather(ch.ek, ch.ekg, n * ew.n, .f32, e.s);
    if (e.world != 2) return error.NotPortedYet; // Comm.sum of more ranks: acc += g[r]
    try e.ops.add2Bf16(e.s, ch.ekg, ch.ekg + n * en * 4, ch.kv, n * en);
    try tri_basic.engramGate(e.t, ch.h, ch.kv, lay.engram_qk, ch.h_alt, c.eps, n, c.hidden);
    swapStreams(ch, ch.h_alt);
}

/// The decoder's bounded replay at its first layer (_forward_k's `replay`, CED's prefill): the layer's attention mixes
/// (the pending MoE post fused in) and its kv-source update over every row of the chunk, then the chunk cut to its rows
/// at positions >= replay, the streams and their pre copied down (h[first:], pre[first:]). The decoder layers then run
/// on those rows with no window key before replay and this layer's update done. False: an encoder-only chunk (no
/// row at or past replay), which ends here.
pub fn replayCut(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, replay: usize, host_pos: []i64) !bool {
    try attnMixes(e, ch, li);
    if (e.w.layers[li].comp_wkv == null) return error.NoKvSource;
    try kvSourceUpdate(e, ch, cs, sh, li);
    const first = @max(ch.start, replay) - ch.start;
    if (first >= ch.n) return false;
    if (first == 0) return true;
    const c = e.c;
    const n = ch.n - first;
    const row_h = c.hc * c.hidden * 2;
    try e.ops.copyRows(e.s, ch.h + first * row_h, row_h, ch.h_alt, row_h, row_h, n);
    std.mem.swap(u64, &ch.h, &ch.h_alt);
    const row_p = c.hc * 4;
    try e.ops.copyRows(e.s, ch.pre + first * row_p, row_p, ch.pre_f, row_p, row_p, n);
    std.mem.swap(u64, &ch.pre, &ch.pre_f);
    ch.start += first;
    ch.n = n;
    for (host_pos[0..n], 0..) |*q, i| q.* = @intCast(ch.start + i);
    try upload(e, ch.pos, host_pos.ptr, n * 8);
    return true;
}

/// DSpark tap j (_forward_k at a tap layer): a pending MoE gather posted into the streams first (hc_post, not fused),
/// then h.to(fp32).mean(1).to(bf16) into tap j's rows.
pub fn tap(e: *const Engine, ch: *Chunk, j: usize) !void {
    const c = e.c;
    if (ch.pending) {
        try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, ch.n, c.hidden);
        ch.pending = false;
    }
    try e.exact.hcMean4(e.s, ch.h, ch.taps + j * ch.cap * c.hidden * 2, c.hidden, ch.n, c.hidden);
}

/// Tap j's rows (bf16 [n, D]).
pub fn tapRows(e: *const Engine, ch: *const Chunk, j: usize) u64 {
    return ch.taps + j * ch.cap * e.c.hidden * 2;
}

/// The head on the chunk's last row (_forward_k without all_logits): the pending MoE post, collapse_norm of the row with
/// its pre, the head's columns (one row through the head's own group, fp32) and their gather: head_g holds the
/// prompt's logits, the ranks' columns in rank order.
pub fn head(e: *const Engine, ch: *Chunk) !void {
    const c = e.c;
    const n = ch.n;
    if (ch.pending) {
        try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, n, c.hidden);
        ch.pending = false;
    }
    const last_h = ch.h + (n - 1) * c.hc * c.hidden * 2;
    const last_pre = ch.pre + (n - 1) * c.hc * 4;
    try tri_basic.collapseNorm(e.t, last_h, last_pre, e.w.norm, ch.head_x, c.eps, 1, c.hidden);
    const hl = e.w.head;
    try mmRows(e, ch, 1, hl, ch.head_x, c.hidden, ch.head_l, .fp32, hl.n);
    if (e.two) |t| return t.quarters(e, ch.head_l, ch.head_g, 1, t.hw, 4); // 2D: the vocabulary quarters, TP2's layout
    try e.comm.allGather(ch.head_l, ch.head_g, hl.n, .f32, e.s);
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
