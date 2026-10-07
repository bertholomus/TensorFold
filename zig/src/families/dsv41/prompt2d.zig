//! The exact 2D-TP4 split on the prompt path (DESIGN-zig-tp4.md sections 3-4; model.py Comm2D, wo_2d, moe_2d and
//! exl3/experts2d.py on four nodes): node g = 2 p + r is TP2 rank r of pair p. A node computes every output column it
//! holds whole, over its TP2 slice's K in TP2's order, and every exchange only moves bytes, so after an exchange a
//! node holds TP2 rank r's bytes: `quarters` assembles TP2's [2, n, N] gather of the two ranks' partials (which the
//! mixes then add in rank order, as on TP2), `catRank` the rank's whole wo_a output or expert intermediate from the
//! two pairs' parts. Prompt chunks exchange over NCCL across the four (the RDMA rings drop prompt-size bursts on the
//! switch: slots P6, P7b), each as one all-gather of every node's part then copies into place: the bytes Comm2D's
//! NCCL paths (gather4, _p2p) put there. With RDMA rings (Two.rings, ring2d.zig) the exchanges the lane sends over
//! them take them, written straight into place: quarters of fewer than 64 rows (decode windows; TF_DS_2D_PROMPT_NCCL
//! keeps prompt chunks on NCCL) and every cat_rank its column pair's ring takes; the same bytes either way.
//!
//! prompt.zig takes it through thin hooks behind Engine.two; a TP2 engine has none and runs as before. The decode
//! rounds' part is round2d.zig's, on this node state.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ring2d = @import("ring2d.zig");
const round2d = @import("round2d.zig");
const round_rows = @import("round.zig").max_rows;
const Link = @import("link.zig").Link;
const Comm = @import("comm.zig").Comm;

const nccl = cuda.nccl;

/// The split of node `g` of `world` nodes: TP over 1 or 2, the 2D split over 4.
pub fn split(g: u32, world: u32) !plan.Split {
    return switch (world) {
        1, 2 => .{ .rank = g, .world = world },
        4 => .{ .rank = g % 2, .world = 2, .pair = g / 2 },
        else => error.UnsupportedWorld,
    };
}

/// The links of a run and its NCCL world: rank 0 listens on PORT + g - 1 for follower g, one follower at a time (each
/// follower retries its connect while rank 0 gets to it), so two ranks link exactly as before (rank 1 on PORT).
/// NCCL's unique id goes from rank 0 down every link. (The port's link to N followers replaces this.)
pub const Fan = struct {
    links: [3]?Link = .{ null, null, null },

    pub fn open(io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32) !Fan {
        var f: Fan = .{};
        errdefer f.close();
        if (world > 4) return error.UnsupportedWorld;
        if (world < 2) return f;
        if (rank == 0) {
            for (1..world) |g| f.links[g - 1] = try Link.listen(ip, port + @as(u16, @intCast(g - 1)));
        } else {
            f.links[0] = try Link.connect(io, ip, port + @as(u16, @intCast(rank - 1)), 300);
        }
        return f;
    }

    pub fn close(f: *Fan) void {
        for (&f.links) |*l| {
            if (l.*) |*x| x.close();
            l.* = null;
        }
    }

    /// Comm.init over every link: rank 0 makes the id and sends it to each follower. Needs the CUDA context current.
    pub fn comm(f: *const Fan, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (world > 1) {
            if (rank == 0) {
                try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
                for (f.links) |l| if (l) |x| try x.send(std.mem.asBytes(&id));
            } else {
                var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
                const got = try f.links[0].?.recv(&buf);
                if (got.len != buf.len) return error.BadUniqueId;
                id = std.mem.bytesToValue(nccl.UniqueId, got);
            }
        } else try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var c: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = c, .rank = rank, .world = world };
    }
};

fn len(r: ?[2]usize) usize {
    const x = r orelse return 0;
    return x[1] - x[0];
}

/// A 2D node's part of the prompt path and the decode rounds: the pairs' widths of each split output and the exchange
/// buffers.
pub const Two = struct {
    r: usize,
    p: usize,
    heads: usize, // the heads here (its pair's half of the rank's)
    uw: [2]usize, // wo_a's output a pair (its groups x o_lora): pair 0's, pair 1's
    ow: [2]usize, // wo_b's output columns a pair
    gu: [2]usize, // gate / up's columns a pair
    dw: [2]usize, // down's output columns a pair
    ew: [2]usize, // Engram wkv's output columns a pair (0 without an Engram layer)
    hw: [2]usize, // the head's vocabulary columns a pair (253 / 252 blocks of a TP2 half)
    parts: u64, // [4, rows, widest] where the all-gathers land
    pad: u64, // this node's part at the wider pair's width (when the pairs' widths differ)
    part_bytes: usize, // the largest part a node sends
    u_full: u64, // bf16 [cap, uw0 + uw1]: the rank's wo_a output, pair 0's groups then pair 1's
    xd_full: u64, // fp16 [cap * slots, gu0 + gu1]: the rank's expert intermediate, pair 0's blocks then pair 1's
    own: u64, // fp16 [round rows, uw[p]]: a round's half of wo_b's rotated input rows (wo_a's epilogue, round2d.woRot)
    rings: ?*const ring2d.Rings, // the RDMA rings (TF_DS_2D_INTO, TF_DS_2D_GROUPS), when the caller opened them

    /// The widths from the split (both pairs'), and the buffers for chunks of up to `cap` rows (and rounds) from `a`.
    pub fn init(c: *const Config, w: *const weights.Weights, a: *prompt.Arena, s: plan.Split, cap: usize) !Two {
        const pp = s.pair orelse return error.NotA2DSplit;
        const s0 = s.atPair(0);
        const s1 = s.atPair(1);
        const x0 = s0.expertParts(c.*);
        const x1 = s1.expertParts(c.*);
        var t: Two = undefined;
        t.r = s.rank;
        t.p = pp;
        t.heads = s.headSpan(c.*)[1];
        t.uw = .{ s0.woAGroups(c.*)[1] * c.o_lora, s1.woAGroups(c.*)[1] * c.o_lora };
        t.ow = .{ len(s0.woBCols(c.*)), len(s1.woBCols(c.*)) };
        t.gu = .{ len(x0.gu), len(x1.gu) };
        t.dw = .{ len(x0.dcols), len(x1.dcols) };
        var engram = false;
        for (w.layers) |lay| engram = engram or lay.engram_wkv != null;
        t.ew = if (engram) .{ len(s0.engramCols(c.*)), len(s1.engramCols(c.*)) } else .{ 0, 0 };
        t.hw = .{ len(s0.headCols(c.*)), len(s1.headCols(c.*)) };
        // the weights loaded here are this split's
        const l0 = w.layers[0];
        if (l0.wq_b.n != t.heads * c.head_dim or l0.groups * c.o_lora != t.uw[pp] or l0.wo_b.n != t.ow[pp] or
            l0.experts.width != t.gu[pp] or l0.experts.down_n != t.dw[pp]) return error.WeightsNotThisSplit;
        // prompt.gather assembles either sublayer's partials: both splits must give the same widths
        if (t.ow[0] != t.dw[0] or t.ow[1] != t.dw[1]) return error.UnequalPartialSplits;
        const slots = c.top_k + 1;
        const biggest = @max(@max(@max(cap * @max(t.ow[0], t.ow[1]) * 4, cap * @max(t.ew[0], t.ew[1]) * 4), @max(cap * @max(t.uw[0], t.uw[1]) * 2, cap * slots * @max(t.gu[0], t.gu[1]) * 2)), @max(t.hw[0], t.hw[1]) * 4);
        t.part_bytes = biggest;
        t.parts = try a.take(4 * biggest);
        t.pad = try a.take(biggest);
        t.u_full = try a.take(cap * (t.uw[0] + t.uw[1]) * 2);
        t.xd_full = try a.take(@max(cap, round_rows) * slots * (t.gu[0] + t.gu[1]) * 2);
        t.own = try a.take(round_rows * t.uw[pp] * 2);
        t.rings = null;
        return t;
    }

    /// Every node's part over NCCL into `parts` [4, n, wmax] (bytes), this node's `src` [n, w[p]] padded to wmax.
    fn allParts(t: *const Two, e: *const prompt.Engine, src: u64, n: usize, w: [2]usize, esize: usize) !usize {
        const wmax = @max(w[0], w[1]);
        if (n * wmax * esize > t.part_bytes) return error.PartTooBig;
        var send = src;
        if (w[t.p] != wmax) {
            try e.ops.copyRows(e.s, src, w[t.p] * esize, t.pad, wmax * esize, w[t.p] * esize, n);
            send = t.pad;
        }
        try e.comm.allGather(send, t.parts, n * wmax * esize, .u8, e.s);
        return wmax;
    }

    /// Whether a ring took the exchange: false when there is none or the part does not fit it (nothing sent then).
    fn ringed(r: anyerror!void) !bool {
        r catch |err| switch (err) {
            error.TooLarge, error.NotFloat4 => return false,
            else => return err,
        };
        return true;
    }

    /// quarters (Comm2D.quarters): `src` [n, w[p]] (this node's columns of its TP2 rank's partial) -> `dst`
    /// [2, n, w0 + w1]: rank r's partial, pair 0's columns then pair 1's, for every r.
    pub fn quarters(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (t.rings) |rs| if (n < 64 and try ringed(rs.quarters(e.s, src, dst, n, w, esize))) return;
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..4) |g| {
            const rr = g % 2;
            const pg = g / 2;
            try e.ops.copyRows(e.s, t.parts + g * n * wmax * esize, wmax * esize, dst + rr * n * row + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }

    /// cat_rank (Comm2D.cat_rank): `src` [n, w[p]] -> `dst` [n, w0 + w1]: this node's TP2 rank's whole width, pair 0's
    /// part then pair 1's (from its column partner, the same rank of the other pair).
    pub fn catRank(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.catRank(e.s, src, dst, n, w, esize))) return;
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..2) |pg| {
            const g = t.r + 2 * pg;
            try e.ops.copyRows(e.s, t.parts + g * n * wmax * esize, wmax * esize, dst + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }

    /// wo_2d after wo_a: the rank's four groups (this node's two in ch.u [n, uw[p]], the partner's two), then wo_b
    /// for this pair's output columns: ch.pa [n, ow[p]] fp32, the matching columns of TP2's partial.
    pub fn woB(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        try t.catRank(e, ch.u, t.u_full, ch.n, t.uw, 2);
        try prompt.mm(e, ch, lay.wo_b, t.u_full, t.uw[0] + t.uw[1], ch.pa, .fp32, lay.wo_b.n);
    }

    /// moe_2d's experts after the routing: this node's gate / up blocks, the rank's whole intermediate assembled, down
    /// for this node's output columns: ch.pm [n, dw[p]] fp32. A chunk of fewer than EXACT_ROWS rows takes the decode
    /// window's fused path and scratch (experts2d.fused_ok), as the rounds do.
    pub fn experts(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        const c = e.c;
        const n = ch.n;
        if (n < exl3_experts.exact_rows) return round2d.experts(t, e, lay.experts, ch.xsd, ch.x, c.hidden, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
        try exl3_experts2d.gateUp(e.ex, e.s, lay.experts, ch.xs, ch.x, c.hidden, ch.pick, n, c.swiglu_limit);
        try t.catRank(e, ch.xs.xd, t.xd_full, n * ch.xs.slots, t.gu, 2);
        try exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
    }

    /// Engram's projection summed over the ranks from its column quarters (model.py esum): the gathered width.
    pub fn engramWidth(t: *const Two) usize {
        return t.ew[0] + t.ew[1];
    }
};

test "a node's split: TP over two, the 2D split over four" {
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2 }, try split(1, 2));
    try std.testing.expectEqual(plan.Split{ .rank = 0, .world = 2, .pair = 1 }, try split(2, 4));
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2, .pair = 1 }, try split(3, 4));
    try std.testing.expectError(error.UnsupportedWorld, split(0, 3));
}
