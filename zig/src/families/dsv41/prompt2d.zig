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
const neighbor = @import("neighbor.zig");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ring2d = @import("ring2d.zig");
const rdma_mod = @import("rdma.zig");
const round2d = @import("round2d.zig");
const round_rows = @import("round.zig").max_rows;
const Link = @import("link.zig").Link;
const Comm = @import("comm.zig").Comm;

const nccl = cuda.nccl;

/// TF_DS_2D_YX=1 (model.zig sets it, every rank alike; off by default): prompt.gather exchanges the sublayer partials as
/// their bf16 sums (Two.ySums) instead of the four fp32 quarters.
pub var y_sums: bool = false;

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
        return f.commChecked(rank, world, null);
    }

    /// comm, with this rank's Settings (Model.open's): rank 0 sends its own after the id and waits for each follower's
    /// answer, a follower checks rank 0's against its own and answers. A difference ends both with SettingsDiffer, its
    /// words on stderr, before NCCL starts (a follower of fewer streams than rank 0 serves otherwise refuses the first
    /// step of a stream past its own and rank 0 waits for it in an exchange for good).
    pub fn commChecked(f: *const Fan, rank: u32, world: u32, mine: ?Settings) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (world > 1) {
            var why: [1024]u8 = undefined;
            const differ = if (rank == 0) blk: {
                try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
                break :blk try f.lead(std.mem.asBytes(&id), mine, &why);
            } else try f.follow(rank, std.mem.asBytes(&id), mine, &why);
            if (differ.len > 0) {
                std.debug.print("tf-dsv41: the ranks' settings differ: {s}\n", .{differ});
                return error.SettingsDiffer;
            }
        } else try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var c: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = c, .rank = rank, .world = world };
    }

    /// Rank 0: the id, and after it `mine` when given, down every link; then every follower's answer. "" when each one
    /// can follow, else the first one's words (in `why`).
    fn lead(f: *const Fan, id: []const u8, mine: ?Settings, why: []u8) ![]const u8 {
        var frame: [id_max + @sizeOf(Settings)]u8 = undefined;
        if (id.len > id_max) return error.BadUniqueId;
        @memcpy(frame[0..id.len], id);
        var n = id.len;
        if (mine) |s| {
            @memcpy(frame[n..][0..@sizeOf(Settings)], std.mem.asBytes(&s));
            n += @sizeOf(Settings);
        }
        for (f.links) |l| if (l) |x| try x.send(frame[0..n]);
        if (mine == null) return "";
        var first: []const u8 = "";
        var rest: [1024]u8 = undefined;
        for (f.links) |l| if (l) |x| {
            const got = try x.recv(if (first.len == 0) why else &rest);
            if (first.len == 0 and got.len > 0) first = got;
        };
        return first;
    }

    /// A follower: rank 0's id into `id`, and the Settings after it (an older rank 0 sends none: nothing to check)
    /// checked against `mine`, the answer sent back. "" when this rank can follow, else its words (in `why`).
    fn follow(f: *const Fan, rank: u32, id: []u8, mine: ?Settings, why: []u8) ![]const u8 {
        const l = f.links[0].?;
        var frame: [id_max + @sizeOf(Settings)]u8 = undefined;
        const got = try l.recv(&frame);
        if (got.len != id.len and got.len != id.len + @sizeOf(Settings)) return error.BadUniqueId;
        @memcpy(id, got[0..id.len]);
        if (got.len == id.len) return "";
        const theirs = std.mem.bytesToValue(Settings, got[id.len..]);
        if (theirs.magic != Settings.tag) return error.BadUniqueId;
        const text = if (mine) |s| s.differ(theirs, rank, why) else "";
        try l.send(text);
        return text;
    }

    const id_max = 256; // NCCL's id is 128 bytes
};

/// What the ranks of a run must agree on, checked at the start (Fan.commChecked) from Model.open's options.
pub const Settings = extern struct {
    magic: u32 = tag,
    world: u32,
    streams: u32, // TF_DS_STREAMS (a follower's --streams): a follower holds every stream rank 0 serves
    round_rows: u32, // TF_DS_ROUND_ROWS (--round-rows): the same (it picks the decode kernels)
    rails: u32, // TF_RDMA_DEVICES (--rdma): as many devices (0: NCCL)
    drafts: u32, // the drafter (--drafts): the same
    replay_floor: u32, // TF_DS_REPLAY_FLOOR: the same
    idx_split: u32 = 0, // TF_DS_IDX_SPLIT (prompt.split_from): the same (the keyed indexer layers all-gather inside)
    neighbor_rdma: u32 = 0, // TF_DS_NEIGHBOR_RDMA: the same exchange path on every rank
    reserved: u32 = 0,
    pool: u64, // rank 0's --context (a follower's --pool): a follower's holds rank 0's

    pub const tag: u32 = 0x31345344; // "DS41"

    /// What follower `rank` (its Settings `mine`) would have to change to follow rank 0's `lead`, in words ("": none).
    pub fn differ(mine: Settings, lead: Settings, rank: u32, buf: []u8) []const u8 {
        var w: std.Io.Writer = .fixed(buf);
        const sep = "; ";
        if (mine.world != lead.world)
            w.print("{s}rank {d} runs WORLD {d} and rank 0 {d}", .{ sepIf(&w, sep), rank, mine.world, lead.world }) catch {};
        if (mine.streams < lead.streams)
            w.print("{s}rank {d} holds {d} streams and rank 0 serves {d}: start rank {d} with --streams {d} (or TF_DS_STREAMS={d})", .{ sepIf(&w, sep), rank, mine.streams, lead.streams, rank, lead.streams, lead.streams }) catch {};
        if (mine.pool < lead.pool)
            w.print("{s}rank {d}'s pool holds {d} positions and rank 0's {d} (its --context, or TF_DS_WINDOW): start rank {d} with --pool {d}", .{ sepIf(&w, sep), rank, mine.pool, lead.pool, rank, lead.pool }) catch {};
        if (mine.round_rows != lead.round_rows)
            w.print("{s}rank {d} runs rounds of {d} rows and rank 0 {d}: start rank {d} with --round-rows {d} (or TF_DS_ROUND_ROWS={d})", .{ sepIf(&w, sep), rank, mine.round_rows, lead.round_rows, rank, lead.round_rows, lead.round_rows }) catch {};
        if (mine.rails != lead.rails)
            w.print("{s}rank {d} has {d} RDMA devices and rank 0 {d} (TF_RDMA_DEVICES): give rank {d} as many with --rdma", .{ sepIf(&w, sep), rank, mine.rails, lead.rails, rank }) catch {};
        if (mine.drafts != lead.drafts)
            w.print("{s}rank {d} {s} the drafter and rank 0 {s}: start rank {d} with --drafts {d}", .{ sepIf(&w, sep), rank, if (mine.drafts != 0) "loads" else "does not load", if (lead.drafts != 0) "does" else "does not", rank, lead.drafts }) catch {};
        if (mine.replay_floor != lead.replay_floor)
            w.print("{s}rank {d} has TF_DS_REPLAY_FLOOR={d} and rank 0 {d}: set the same on every rank", .{ sepIf(&w, sep), rank, mine.replay_floor, lead.replay_floor }) catch {};
        if (mine.idx_split != lead.idx_split)
            w.print("{s}rank {d} splits the keyed indexer rows from {d} columns and rank 0 from {d} (TF_DS_IDX_SPLIT, 0: unset): set the same on every rank", .{ sepIf(&w, sep), rank, mine.idx_split, lead.idx_split }) catch {};
        if (mine.neighbor_rdma != lead.neighbor_rdma)
            w.print("{s}rank {d} has TF_DS_NEIGHBOR_RDMA={d} and rank 0 {d}: set the same on every rank", .{ sepIf(&w, sep), rank, mine.neighbor_rdma, lead.neighbor_rdma }) catch {};
        return w.buffered();
    }

    fn sepIf(w: *const std.Io.Writer, sep: []const u8) []const u8 {
        return if (w.end > 0) sep else "";
    }
};

fn len(r: ?[2]usize) usize {
    const x = r orelse return 0;
    return x[1] - x[0];
}

/// A 2D node's part of the prompt path and the decode rounds: the pairs' widths of each split output and the exchange
/// buffers.
pub const Two = struct {
    neighbor_gather: ?*const neighbor.Gather,
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
    // TF_DS_2D_GU=parity: the rest block's width (0: "first"), the rest launches' scratch for a decode window and for a
    // prompt chunk, and the column partner's pack (its main rows, then its rest rows) as the exchange delivers it
    ir: usize = 0,
    rest_rows: usize = 0, // the prompt chunks' rows the rest scratch holds (init's cap)
    rest_dec: ?exl3_experts2d.RestScratch = null,
    rest_chunk: ?exl3_experts2d.RestScratch = null,
    peer: u64 = 0,

    /// The widths from the split (both pairs'), and the buffers for chunks of up to `cap` rows (and rounds) from `a`.
    pub fn init(c: *const Config, w: *const weights.Weights, a: *prompt.Arena, s: plan.Split, cap: usize) !Two {
        const pp = s.pair orelse return error.NotA2DSplit;
        const s0 = s.atPair(0);
        const s1 = s.atPair(1);
        const x0 = s0.expertParts(c.*);
        const x1 = s1.expertParts(c.*);
        var t: Two = undefined;
        t.neighbor_gather = null;
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
        // parity: the rest block after both pairs' main blocks in the assembled intermediate
        const ex0 = l0.experts;
        t.ir = if (ex0.rest) |r| r.width else 0;
        t.rest_rows = cap;
        t.rest_dec = null;
        t.rest_chunk = null;
        t.peer = 0;
        t.xd_full = try a.take(@max(cap, round_rows) * slots * (t.gu[0] + t.gu[1] + t.ir) * 2);
        if (t.ir != 0) {
            if (t.gu[0] != t.gu[1]) return error.UnequalParityParts;
            t.rest_dec = try restScratch(a, ex0, exl3_experts.exact_rows, slots, false);
            t.rest_chunk = try restScratch(a, ex0, cap, slots, true);
            t.peer = try a.take(@max(cap, round_rows) * slots * (t.gu[t.p] + t.ir) * 2);
        }
        t.own = try a.take(round_rows * t.uw[pp] * 2);
        t.rings = null;
        return t;
    }

    fn restScratch(a: *prompt.Arena, ex: weights.Experts, rows: usize, slots: usize, chunk: bool) !exl3_experts2d.RestScratch {
        const sz = try exl3_experts2d.RestScratch.sizes(rows, slots, ex, chunk);
        var rs: exl3_experts2d.RestScratch = undefined;
        inline for (.{ "ids", "count", "counts", "members", "cnt_gu", "z", "work_gu", "work_d", "rpick" }, 0..) |f, j| @field(rs, f) = try a.take(sz[j]);
        return rs;
    }

    /// The rest scratch's first use: members -1, the counters and partials zero (as the main scratch starts).
    pub fn clearRest(t: *const Two, e: *const prompt.Engine) !void {
        for ([_]?exl3_experts2d.RestScratch{ t.rest_dec, t.rest_chunk }, [_]bool{ false, true }) |o, chunk| {
            const rs = o orelse continue;
            const ex = e.w.layers[0].experts;
            const sz = try exl3_experts2d.RestScratch.sizes(if (chunk) t.rest_rows else exl3_experts.exact_rows, e.slots(), ex, chunk);
            try prompt.fill32(e, rs.members, 0xffffffff, sz[3] / 4);
            for ([_]u64{ rs.ids, rs.count, rs.counts, rs.cnt_gu, rs.z }, [_]usize{ sz[0], sz[1], sz[2], sz[4], sz[5] }) |ptr, n| try prompt.fill(e, ptr, 0, n);
        }
    }

    /// The column partner's pack (a parity split's main rows, then its rest rows: `bytes` each way) into t.peer: over
    /// the column ring when the caller opened the rings and it fits, else NCCL send / recv with the partner.
    pub fn packExchange(t: *const Two, e: *const prompt.Engine, mine: u64, rows: usize, width: usize) !void {
        if (t.rings) |rs| {
            var dsts: [2]rdma_mod.Ring.Dst = .{ .{}, .{} };
            dsts[1 - t.p] = .{ .ptr = t.peer, .row4 = @intCast(width * 2 / 16), .stride4 = width * 2 / 16 };
            if (rs.col.gatherInto(e.s, mine, @intCast(rows), @intCast(width * 2 / 16), &dsts)) |_| return else |err| switch (err) {
                error.TooLarge => {},
                else => return err,
            }
        }
        const count = try std.math.mul(usize, rows, width);
        if (t.neighbor_gather) |ng| {
            const bytes = try std.math.mul(usize, count, 2);
            if (neighbor.eligible(bytes)) {
                if (bytes > t.part_bytes) return error.PartTooBig;
                try ng.relay(e.s, mine, t.peer, bytes);
            } else {
                try neighbor.forward(e.comm, mine, t.peer, @splat(count), .f16, t.pad, t.part_bytes, e.s);
            }
            return;
        }
        const g: u32 = @intCast(t.r + 2 * t.p);
        try e.comm.exchange(mine, count, t.peer, count, .f16, g ^ 2, e.s);
    }

    /// moe_2d's parity experts after the main gate / up and the rest's: the packs exchanged with the column partner,
    /// the half's intermediate assembled into t.xd_full (pair 0's main, pair 1's main, each slot's rest from its
    /// owner), for down.
    pub fn assembleParity(t: *const Two, e: *const prompt.Engine, ex: weights.Experts, xd: u64, pick: u64, p: usize) !void {
        const rest = ex.rest orelse return error.NoRest;
        const ig = t.gu[t.p];
        try t.packExchange(e, xd, p, ig + t.ir);
        const part0 = if (t.p == 0) xd else t.peer;
        const part1 = if (t.p == 1) xd else t.peer;
        try e.ops.assembleRest(e.s, part0, part1, pick, rest.owner, t.xd_full, p, ig, t.ir);
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
        const bytes = n * wmax * esize;
        if (t.neighbor_gather) |ng| {
            if (neighbor.eligible(bytes)) {
                try ng.gather(e.s, send, t.parts, bytes);
                return wmax;
            }
        }
        try e.comm.allGather(send, t.parts, bytes, .u8, e.s);
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
        if (t.neighbor_gather) |ng| {
            const width = try std.math.mul(usize, w[0], esize);
            const bytes = try std.math.mul(usize, n, width);
            if (w[0] == w[1] and width % 16 == 0 and neighbor.eligible(bytes)) {
                try ng.quarters(e.s, src, dst, n, width);
                return;
            }
        }
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..4) |g| {
            const rr = g % 2;
            const pg = g / 2;
            try e.ops.copyRows(e.s, t.parts + g * n * wmax * esize, wmax * esize, dst + rr * n * row + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }

    /// TF_DS_2D_YX's prompt.gather, when y_sums is on and the chunk takes NCCL (no ring) with equal pair widths: the
    /// pair partners swap their fp32 partials of each other's half of the rows (src [n, w]), each sums its half in TP2
    /// rank order and rounds it to bf16 (add2Bf16: hc_mix_pf's and hc_post's y), the four nodes all-gather those halves
    /// (bf16 [half, w] each), and dst[0] gets them widened to fp32 [n, 2 w]. dst[1] (at cap * 2 w, in every buffer of
    /// `negs`) holds -0.0, written once (x + -0.0 == x for every x): the posting kernels' y is bit for bit the
    /// quarters'. Bytes a node receives: 5 n w against 12 n w. False: not taken (nothing sent).
    pub fn ySums(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, cap: usize, neg_done: *bool, negs: []const u64) !bool {
        if (!y_sums or n == 0 or t.ow[0] != t.ow[1]) return false;
        if (t.rings != null and n < 64) return false;
        const w = t.ow[0];
        if (2 * w != e.c.hidden) return false;
        const half = (n + 1) / 2;
        if (half * w * 6 > t.part_bytes) return error.PartTooBig;
        if (!neg_done.*) {
            for (negs) |g| try prompt.fill32(e, g + cap * 2 * w * 4, 0x80000000, cap * 2 * w);
            neg_done.* = true;
        }
        const mine = t.r * half;
        const mine_n = if (t.r == 0) half else n - half;
        const theirs = (1 - t.r) * half;
        const theirs_n = n - mine_n;
        const g: u32 = @intCast(t.r + 2 * t.p);
        const recv = t.pad;
        const own = t.pad + half * w * 4;
        try e.comm.exchange(src + theirs * w * 4, theirs_n * w, recv, mine_n * w, .f32, g ^ 1, e.s);
        const p0 = if (t.r == 0) src + mine * w * 4 else recv;
        const p1 = if (t.r == 0) recv else src + mine * w * 4;
        try e.ops.add2Bf16(e.s, p0, p1, own, mine_n * w);
        try e.comm.allGather(own, t.parts, half * w * 2, .u8, e.s);
        try e.ops.yWiden(e.s, t.parts, dst, n, half, w);
        return true;
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

    /// Comm2D.gather (TP2 inside a pair: the DSpark drafter's sublayer partials): `src` [n, width] -> `dst` [2, n, width],
    /// this pair's two ranks in TP2 rank order: over the row pair's ring, else the four-rank all-gather's pair half
    /// (Comm2D.gather4(x)[2p:2p + 2]).
    pub fn pairGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, width: usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.pairGather(e.s, src, dst, n, width, esize))) return;
        if (t.neighbor_gather) |ng| {
            const bytes = try std.math.mul(usize, n, try std.math.mul(usize, width, esize));
            if (neighbor.eligible(bytes)) {
                try ng.pair(e.s, src, dst, bytes);
                return;
            }
        }
        const wmax = try t.allParts(e, src, n, .{ width, width }, esize);
        try e.ops.copyRows(e.s, t.parts + 2 * t.p * n * wmax * esize, wmax * esize, dst, width * esize, width * esize, 2 * n);
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
        if (t.ir != 0) {
            // parity: the rest columns of the experts this pair computes them for, the packs exchanged and assembled
            try exl3_experts2d.gateUpRest(e.ex, e.ops, e.s, lay.experts, ch.xs, t.rest_chunk.?, ch.pick, n, c.swiglu_limit);
            try t.assembleParity(e, lay.experts, ch.xs.xd, ch.pick, n * ch.xs.slots);
            return exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
        }
        try t.catRank(e, ch.xs.xd, t.xd_full, n * ch.xs.slots, t.gu, 2);
        try exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
    }

    /// Engram's projection summed over the TP2 ranks and cast, kv [n, ew0 + ew1] bf16 = bf16(g0 + g1) (model.py esum
    /// then .to(bf16)), from this node's fp32 columns `src` [n, ew[p]] with less traffic than the quarters: the row
    /// partner's columns over NCCL send / recv, one fp32 add (two operands: the same bits in either order) cast to
    /// bf16, then the column partner's bf16 sums the same way. `tmp` holds n * (ew[p] * 6 + ew[1 - p] * 2) bytes.
    pub fn engramSum(t: *const Two, e: *const prompt.Engine, src: u64, tmp: u64, kv: u64, n: usize) !void {
        const w = t.ew;
        const mine = n * w[t.p];
        const other = tmp;
        const half = tmp + mine * 4;
        const their = half + mine * 2;
        const g: u32 = @intCast(t.r + 2 * t.p);
        try e.comm.exchange(src, mine, other, mine, .f32, g ^ 1, e.s);
        try e.ops.add2Bf16(e.s, src, other, half, mine);
        if (t.neighbor_gather != null) {
            try neighbor.forward(e.comm, half, their, .{ n * w[0], n * w[0], n * w[1], n * w[1] }, .bf16, t.pad, t.part_bytes, e.s);
        } else {
            try e.comm.exchange(half, mine, their, n * w[1 - t.p], .bf16, g ^ 2, e.s);
        }
        const row = (w[0] + w[1]) * 2;
        for (0..2) |pg| {
            const from = if (pg == t.p) half else their;
            try e.ops.copyRows(e.s, from, w[pg] * 2, kv + pg * w[0] * 2, row, w[pg] * 2, n);
        }
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

test "a follower holds rank 0's streams and pool and runs its round rows, or says what to change" {
    const served: Settings = .{ .world = 2, .streams = 8, .round_rows = 16, .rails = 2, .drafts = 1, .replay_floor = 1, .pool = 524288 };
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("", served.differ(served, 1, &buf));
    var more = served; // a follower may hold more streams and positions than rank 0 uses
    more.streams = 16;
    more.pool = 1 << 20;
    try std.testing.expectEqualStrings("", more.differ(served, 1, &buf));
    var four = served;
    four.streams = 4;
    try std.testing.expectEqualStrings("rank 1 holds 4 streams and rank 0 serves 8: start rank 1 with --streams 8 (or TF_DS_STREAMS=8)", four.differ(served, 1, &buf));
    var other = served;
    other.round_rows = 48;
    other.replay_floor = 0;
    other.pool = 262144;
    other.rails = 0;
    other.drafts = 0;
    const words = other.differ(served, 3, &buf);
    for ([_][]const u8{ "--pool 524288", "--round-rows 16", "--rdma", "--drafts 1", "TF_DS_REPLAY_FLOOR=0 and rank 0 1" }) |s|
        try std.testing.expect(std.mem.indexOf(u8, words, s) != null);
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, words, "; "));
}

test "a follower splits the keyed indexer rows as rank 0 does, or says to set the same" {
    const served: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 2, .drafts = 1, .replay_floor = 0, .pool = 1 << 21 };
    var buf: [1024]u8 = undefined;
    var lead = served;
    lead.idx_split = 32768;
    try std.testing.expectEqualStrings("", lead.differ(lead, 2, &buf));
    try std.testing.expectEqualStrings("rank 2 splits the keyed indexer rows from 0 columns and rank 0 from 32768 (TF_DS_IDX_SPLIT, 0: unset): set the same on every rank", served.differ(lead, 2, &buf));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Settings));
}

test "rank 0's settings go with NCCL's id: the follower answers, and an older rank 0's id alone is taken as before" {
    const io = std.testing.io;
    const served: Settings = .{ .world = 2, .streams = 8, .round_rows = 16, .rails = 0, .drafts = 1, .replay_floor = 1, .pool = 1 << 19 };
    var four = served;
    four.streams = 4;
    var neighbor_on = served;
    neighbor_on.neighbor_rdma = 1;
    const Case = struct { lead: ?Settings, follower: ?Settings, differs: bool };
    const cases = [_]Case{
        .{ .lead = served, .follower = served, .differs = false },
        .{ .lead = served, .follower = four, .differs = true },
        .{ .lead = served, .follower = neighbor_on, .differs = true },
        .{ .lead = neighbor_on, .follower = served, .differs = true },
        .{ .lead = neighbor_on, .follower = neighbor_on, .differs = false },
        .{ .lead = null, .follower = four, .differs = false }, // an older rank 0: the id alone, nothing to check
    };
    const base: u16 = 39500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    for (cases, 0..) |cs, k| {
        const Peer = struct {
            id: [128]u8 = undefined,
            why: [1024]u8 = undefined,
            said: []const u8 = "",
            err: ?anyerror = null,
            fn run(p: *@This(), port: u16, mine: ?Settings) void {
                var f: Fan = .{};
                f.links[0] = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, port, 10) catch |e| {
                    p.err = e;
                    return;
                };
                defer f.close();
                p.said = f.follow(1, &p.id, mine, &p.why) catch |e| {
                    p.err = e;
                    return;
                };
            }
        };
        const port = base + @as(u16, @intCast(k));
        var peer: Peer = .{};
        const t = try std.Thread.spawn(.{}, Peer.run, .{ &peer, port, cs.follower });
        var f: Fan = .{};
        f.links[0] = Link.listen(.{ 127, 0, 0, 1 }, port) catch |e| {
            t.join();
            return e;
        };
        var id: [128]u8 = undefined;
        for (&id, 0..) |*b, i| b.* = @intCast(i);
        var why: [1024]u8 = undefined;
        const heard = f.lead(&id, cs.lead, &why);
        t.join();
        f.close();
        try std.testing.expect(peer.err == null);
        try std.testing.expectEqualSlices(u8, &id, &peer.id);
        try std.testing.expectEqualStrings(peer.said, try heard);
        try std.testing.expectEqual(cs.differs, peer.said.len > 0);
        if (cs.differs) {
            const setting = if (cs.follower.?.streams != cs.lead.?.streams) "--streams 8" else "TF_DS_NEIGHBOR_RDMA";
            try std.testing.expect(std.mem.indexOf(u8, peer.said, setting) != null);
        }
    }
}

test "startup settings reject mixed neighbor modes before communicator creation" {
    const off: Settings = .{ .world = 4, .streams = 16, .round_rows = 48, .rails = 0, .drafts = 1, .replay_floor = 0, .pool = 1 << 21 };
    var on = off;
    on.neighbor_rdma = 1;
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("", off.differ(off, 1, &buf));
    try std.testing.expectEqualStrings("", on.differ(on, 1, &buf));
    try std.testing.expectEqualStrings("rank 1 has TF_DS_NEIGHBOR_RDMA=0 and rank 0 1: set the same on every rank", off.differ(on, 1, &buf));
    try std.testing.expectEqualStrings("rank 1 has TF_DS_NEIGHBOR_RDMA=1 and rank 0 0: set the same on every rank", on.differ(off, 1, &buf));
}
