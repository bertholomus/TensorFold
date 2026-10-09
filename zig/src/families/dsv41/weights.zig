//! One rank's weights on the GPU in the served engine's layouts (weights.py, exl3/linear.py, exl3/experts.py): EXL3
//! linears in strips, each layer's experts in one trellis buffer with pointer, width and scale tables, the rest as the
//! loader keeps it. Every device tensor is listed by the Python engine's name for digests against its weights.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const rank_cache = @import("rank_cache.zig");
const checkpoint = @import("checkpoint.zig");

/// An EXL3 linear in the strips layout: words int32 [N/128, K/16, 8, 8 * bits], fp16 suh [K] and svh [N].
/// plan_n: the N its decode plan is made for, when not n: a 2D node's column part of a TP2 slice keeps that slice's
/// plan (linear.py pinned plans), so its columns sum in the slice's K order.
pub const Linear = struct { words: u64, suh: u64, svh: u64, k: u32, n: u32, k2: u32, plan_n: u32 = 0 };

/// A layer's routed experts and its shared expert (the last entry), as exl3_experts.prepare lays them out. On a 2D node
/// (exl3/experts2d.py prepare) gate / up hold its intermediate blocks (width of the half's down_k) and down its
/// output columns (down_n of dims) over the half's whole intermediate.
pub const Experts = struct {
    trellis: u64, // gate trellises of every expert, then up, then down (pack_trellises)
    gate_ptr: u64, // int64 [E]
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64, // int32 [E]: half-bits a value
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]: down's input signs on the gate / up blocks here
    svh_d: u64, // fp16 [E, down_n]
    count: u32,
    dims: u32,
    width: u32,
    k2_gu: [2]u32,
    k2_d: [2]u32,
    down_k: u32, // down's K: the rank's whole intermediate (width but on a 2D node)
    down_n: u32, // down's output columns here (dims but on a 2D node)
    shared_gate: [2]u64 = .{ 0, 0 }, // the shared expert's gate trellis: its first byte and length (the paced L2 prefetch)
    shared_up: [2]u64 = .{ 0, 0 }, // its up and down trellises the same way (the served "moe" fork reads all three)
    shared_down: [2]u64 = .{ 0, 0 },
};

pub const Layer = struct {
    idx: u32,
    ratio: u8,
    hc_attn: [3]u64 = .{ 0, 0, 0 }, // fn [24, hc * d] f32, scale [3], base [24]
    hc_ffn: [3]u64 = .{ 0, 0, 0 },
    attn_norm: u64 = 0,
    ffn_norm: u64 = 0,
    wq_a: Linear = undefined,
    q_norm: u64 = 0,
    wq_b: Linear = undefined, // this rank's heads' columns
    wkv: Linear = undefined,
    kv_norm: u64 = 0,
    sink: u64 = 0, // this rank's heads, f32
    wo_a: [4]Linear = undefined, // this rank's groups
    groups: u32 = 4, // wo_a's entries loaded: the rank's groups, its pair's half of them on a 2D node
    wo_b: Linear = undefined, // this rank's groups' input rows (a 2D node: its pair's output columns of them)
    comp_wkv: ?Linear = null,
    comp_wgate: ?Linear = null,
    comp_norm: u64 = 0,
    idx_wq_b: ?Linear = null,
    idx_proj: u64 = 0, // [idx_heads, d] f32
    idx_proj_h: u64 = 0, // the same in fp16 (Model.__init__)
    idx_wk: ?Linear = null,
    idx_k_norm: u64 = 0,
    gate_w: u64 = 0, // [E, d] f16
    gate_b: u64 = 0, // [E] f32
    gate_b_vl: u64 = 0, // [E] f32: the routing bias of image-span tokens (the checkpoint's extra gate_bias_vl)
    experts: Experts = undefined,
    engram_wkv: ?Linear = null, // this rank's hash columns' input rows
    engram_qk: u64 = 0, // q_weight * k_weight [hc, d] f32
};

pub const DSpark = struct { blocks: []Layer, main_proj: Linear, main_norm: u64, norm: u64, markov_embed: u64, markov_head: u64, conf: u64 };

/// A device tensor by the Python engine's dotted name.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    embed: u64 = 0, // [V, d] bf16
    norm: u64 = 0,
    head: Linear = undefined, // this rank's vocabulary columns
    vocab_lo: usize = 0,
    vocab_hi: usize = 0,
    layers: []Layer = &.{},
    dspark: ?DSpark = null,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        for (w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);
        w.named.deinit(w.gpa);
        if (w.dspark) |d| w.gpa.free(d.blocks);
        w.gpa.free(w.layers);
        w.* = undefined;
    }
};

/// Where entries come from: the lane's rank cache file, or the checkpoint read by the split.
pub const Source = union(enum) {
    cache: struct { file: std.Io.File, index: *const rank_cache.Index, io: std.Io },
    shards: *const checkpoint.Shards,

    pub fn size(s: Source, key: []const u8) !usize {
        return switch (s) {
            .cache => |c| (c.index.get(key) orelse return error.MissingEntry).bytes,
            .shards => |sh| (try checkpoint.describe(sh, key)).bytes,
        };
    }

    pub fn read(s: Source, key: []const u8, out: []u8) !void {
        switch (s) {
            .cache => |c| {
                const e = c.index.get(key) orelse return error.MissingEntry;
                if (e.bytes != out.len) return error.BadSize;
                if (try c.file.readPositionalAll(c.io, out, e.offset) != out.len) return error.ShortRead;
            },
            .shards => |sh| try checkpoint.read(sh, key, out),
        }
    }
};

/// Trellis int16 [kt, nt, w16] as int32 words [nt / 8, kt, 8, w16 / 2] (linear.py strips): each 128-column block's
/// tiles in k order. `tile` is one tile's bytes (w16 * 2).
pub fn strips(src: []const u8, out: []u8, kt: usize, nt: usize, tile: usize) void {
    for (0..nt / 8) |j| for (0..kt) |i| {
        const from = (i * nt + j * 8) * tile;
        const to = (j * kt + i) * 8 * tile;
        @memcpy(out[to..][0 .. 8 * tile], src[from..][0 .. 8 * tile]);
    };
}

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    src: Source,
    w: *Weights,
    host: std.ArrayList(u8) = .empty,
    aux: std.ArrayList(u8) = .empty,

    fn upload(L: *Loader, name: []const u8, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.alloc(L.d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        try L.w.buffers.append(L.gpa, b);
        try L.w.named.append(L.gpa, .{ .name = try L.gpa.dupe(u8, name), .ptr = b.ptr, .len = bytes.len });
        L.w.bytes += bytes.len;
        return b.ptr;
    }

    fn entry(L: *Loader, key: []const u8) ![]u8 {
        try L.host.resize(L.gpa, try L.src.size(key));
        try L.src.read(key, L.host.items);
        return L.host.items;
    }

    /// plain(name, dtype).cuda(): the entry as it is.
    fn plain(L: *Loader, tensor: []const u8, dtype: ?rank_cache.DType) !u64 {
        var kb: [256]u8 = undefined;
        return L.upload(tensor, try L.entry(try plan.plainKey(&kb, tensor, dtype)));
    }

    /// linear(prefix, cols, rows): Exl3Linear.from_tensors in strips.
    fn linear(L: *Loader, prefix: []const u8, cols: ?[2]usize, rows: ?[2]usize) !Linear {
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        const tr = try L.entry(try plan.partKey(&kb, prefix, cols, rows, "tr"));
        const t = try L.shape(try plan.partKey(&kb, prefix, cols, rows, "tr"));
        const kt = t[0];
        const nt = t[1];
        const w16 = t[2];
        if (w16 % 8 != 0 or nt % 8 != 0) return error.UnexpectedTensor;
        try L.aux.resize(L.gpa, tr.len);
        strips(tr, L.aux.items, kt, nt, w16 * 2);
        const words = try L.upload(try std.fmt.bufPrint(&nb, "{s}.words", .{prefix}), L.aux.items);
        const suh = try L.upload(try std.fmt.bufPrint(&nb, "{s}.suh", .{prefix}), try L.entry(try plan.partKey(&kb, prefix, cols, rows, "suh")));
        const svh = try L.upload(try std.fmt.bufPrint(&nb, "{s}.svh", .{prefix}), try L.entry(try plan.partKey(&kb, prefix, cols, rows, "svh")));
        return .{ .words = words, .suh = suh, .svh = svh, .k = @intCast(kt * 16), .n = @intCast(nt * 16), .k2 = @intCast(w16 / 8) };
    }

    /// A trellis entry's [kt, nt, w16].
    fn shape(L: *Loader, key: []const u8) ![3]usize {
        return switch (L.src) {
            .cache => |c| blk: {
                const e = c.index.get(key) orelse return error.MissingEntry;
                if (e.rank != 3) return error.UnexpectedTensor;
                break :blk .{ e.shape[0], e.shape[1], e.shape[2] };
            },
            .shards => |sh| blk: {
                const s = try checkpoint.describe(sh, key);
                if (s.rank != 3) return error.UnexpectedTensor;
                break :blk .{ s.shape[0], s.shape[1], s.shape[2] };
            },
        };
    }

    /// load_block's experts: exl3_parts of w1 / w3 by output columns and w2 by input rows for every expert and the
    /// shared one, packed as pack_trellises packs them, with prepare's tables. `half`: the rank's intermediate; `gu`:
    /// gate / up's columns of it here (the half, or a 2D node's blocks); `dcols`: down's output columns (null: all).
    fn experts(L: *Loader, p: []const u8, n: usize, half: [2]usize, gu: [2]usize, dcols: ?[2]usize, d: usize) !Experts {
        const e_count = n + 1;
        const width = gu[1] - gu[0];
        const down_k = half[1] - half[0];
        const down_n = if (dcols) |dc| dc[1] - dc[0] else d;
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        const prefixes = try L.gpa.alloc([3][]u8, e_count);
        defer {
            for (prefixes) |pp| for (pp) |x| L.gpa.free(x);
            L.gpa.free(prefixes);
        }
        for (0..e_count) |e| {
            const ep = if (e < n) try std.fmt.bufPrint(&nb, "{s}.ffn.experts.{d}", .{ p, e }) else try std.fmt.bufPrint(&nb, "{s}.ffn.shared_experts", .{p});
            for ([_][]const u8{ "w1", "w3", "w2" }, 0..) |proj, j| prefixes[e][j] = try std.fmt.allocPrint(L.gpa, "{s}.{s}", .{ ep, proj });
        }
        // exl3_parts' (cols, rows) of projection j: gate / up by output columns, down by input rows (and on a 2D
        // node by its output columns too)
        const Parts = struct {
            gu: [2]usize,
            half: [2]usize,
            dcols: ?[2]usize,
            fn cols(q: @This(), j: usize) ?[2]usize {
                return if (j < 2) q.gu else q.dcols;
            }
            fn rows(q: @This(), j: usize) ?[2]usize {
                return if (j == 2) q.half else null;
            }
        };
        const pq: Parts = .{ .gu = gu, .half = half, .dcols = dcols };
        // sizes first: one buffer, gate trellises of every expert, then up, then down
        const sizes = try L.gpa.alloc(usize, 3 * e_count);
        defer L.gpa.free(sizes);
        var total: usize = 0;
        for (0..3) |j| for (0..e_count) |e| {
            const key = try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), "tr");
            sizes[j * e_count + e] = try L.src.size(key);
            total += sizes[j * e_count + e];
        };
        var buf = try cuda.DeviceBuffer.alloc(L.d, total);
        errdefer buf.free();
        const ptrs = try L.gpa.alloc(u64, 3 * e_count);
        defer L.gpa.free(ptrs);
        const k2s = try L.gpa.alloc(i32, 3 * e_count);
        defer L.gpa.free(k2s);
        var at: usize = 0;
        for (0..3) |j| for (0..e_count) |e| {
            const key = try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), "tr");
            const tr = try L.entry(key);
            try buf.upload(at, tr);
            const t = try L.shape(key);
            ptrs[j * e_count + e] = buf.ptr + at;
            k2s[j * e_count + e] = @intCast(t[2] / 8);
            at += sizes[j * e_count + e];
        };
        try L.w.buffers.append(L.gpa, buf);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "{s}.ffn.experts.trellis", .{p}), .ptr = buf.ptr, .len = total });
        L.w.bytes += total;
        var x: Experts = undefined;
        x.trellis = buf.ptr;
        x.shared_gate = .{ ptrs[n], sizes[n] }; // j = 0 (gate), e = n (the shared expert)
        x.shared_up = .{ ptrs[e_count + n], sizes[e_count + n] };
        x.shared_down = .{ ptrs[2 * e_count + n], sizes[2 * e_count + n] };
        x.count = @intCast(e_count);
        x.dims = @intCast(d);
        x.width = @intCast(width);
        x.down_k = @intCast(down_k);
        x.down_n = @intCast(down_n);
        const tables = [_]*u64{ &x.gate_ptr, &x.up_ptr, &x.down_ptr };
        const k2t = [_]*u64{ &x.gate_k2, &x.up_k2, &x.down_k2 };
        for (0..3) |j| {
            tables[j].* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.ptr{d}", .{ p, j }), std.mem.sliceAsBytes(ptrs[j * e_count ..][0..e_count]));
            k2t[j].* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.k2{d}", .{ p, j }), std.mem.sliceAsBytes(k2s[j * e_count ..][0..e_count]));
        }
        x.k2_gu = .{ std.math.maxInt(u32), 0 };
        x.k2_d = .{ std.math.maxInt(u32), 0 };
        for (k2s[0 .. 2 * e_count]) |v| x.k2_gu = .{ @min(x.k2_gu[0], @as(u32, @intCast(v))), @max(x.k2_gu[1], @as(u32, @intCast(v))) };
        for (k2s[2 * e_count ..]) |v| x.k2_d = .{ @min(x.k2_d[0], @as(u32, @intCast(v))), @max(x.k2_d[1], @as(u32, @intCast(v))) };
        // stack(mats, j, n): every expert's scale row in order; suh_d is down's input signs on gate / up's blocks
        // (experts2d.prepare: the half's [gu - half), all of it but on a 2D node)
        const off = gu[0] - half[0];
        const stacks = [_]struct { dst: *u64, proj: usize, part: []const u8, len: usize, src_len: usize, off: usize, name: []const u8 }{
            .{ .dst = &x.suh_g, .proj = 0, .part = "suh", .len = d, .src_len = d, .off = 0, .name = "suh_g" },
            .{ .dst = &x.suh_u, .proj = 1, .part = "suh", .len = d, .src_len = d, .off = 0, .name = "suh_u" },
            .{ .dst = &x.svh_g, .proj = 0, .part = "svh", .len = width, .src_len = width, .off = 0, .name = "svh_g" },
            .{ .dst = &x.svh_u, .proj = 1, .part = "svh", .len = width, .src_len = width, .off = 0, .name = "svh_u" },
            .{ .dst = &x.suh_d, .proj = 2, .part = "suh", .len = width, .src_len = down_k, .off = off, .name = "suh_d" },
            .{ .dst = &x.svh_d, .proj = 2, .part = "svh", .len = down_n, .src_len = down_n, .off = 0, .name = "svh_d" },
        };
        for (stacks) |s| {
            try L.aux.resize(L.gpa, e_count * s.len * 2);
            for (0..e_count) |e| {
                const j = s.proj;
                const row = try L.entry(try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), s.part));
                if (row.len != s.src_len * 2) return error.UnexpectedTensor;
                @memcpy(L.aux.items[e * s.len * 2 ..][0 .. s.len * 2], row[s.off * 2 ..][0 .. s.len * 2]);
            }
            s.dst.* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.{s}", .{ p, s.name }), L.aux.items);
        }
        return x;
    }

    fn block(L: *Loader, c: Config, s: plan.Split, p: []const u8, i: usize, n_experts: usize) !Layer {
        var nb: [256]u8 = undefined;
        const gl = s.groups(c);
        var lay: Layer = .{ .idx = @intCast(i), .ratio = c.ratios[i] };
        for ([_][]const u8{ "fn", "scale", "base" }, 0..) |part, k| {
            lay.hc_attn[k] = try L.plain(try std.fmt.bufPrint(&nb, "{s}.hc_attn_{s}", .{ p, part }), .f32);
            lay.hc_ffn[k] = try L.plain(try std.fmt.bufPrint(&nb, "{s}.hc_ffn_{s}", .{ p, part }), .f32);
        }
        lay.attn_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn_norm.weight", .{p}), null);
        lay.ffn_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn_norm.weight", .{p}), null);
        lay.wq_a = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wq_a", .{p}), null, null);
        lay.q_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.q_norm.weight", .{p}), null);
        lay.wq_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wq_b", .{p}), s.wqB(c), null);
        lay.wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wkv", .{p}), null, null);
        lay.kv_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.kv_norm.weight", .{p}), null);
        {
            var kb: [256]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{s}.attn.attn_sink", .{p});
            const all = try L.entry(try plan.plainKey(&kb, name, .f32));
            const hs = s.headSpan(c);
            lay.sink = try L.upload(name, all[hs[0] * 4 ..][0 .. hs[1] * 4]);
        }
        const ga = s.woAGroups(c);
        for (0..ga[1]) |g| lay.wo_a[g] = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wo_a.slice.{d}", .{ p, ga[0] + g }), null, null);
        lay.groups = @intCast(ga[1]);
        lay.wo_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wo_b", .{p}), s.woBCols(c), s.range(gl * c.o_lora));
        if (s.pair != null) {
            // the 2D column parts keep their TP2 slices' decode plans
            lay.wq_b.plan_n = @intCast(s.heads(c) * c.head_dim);
            lay.wo_b.plan_n = @intCast(c.hidden);
        }
        if (c.kv_sources.has(i)) {
            lay.comp_wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.wkv", .{p}), null, null);
            lay.comp_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.norm.weight", .{p}), null);
            if (c.ratios[i] > 1) lay.comp_wgate = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.wgate", .{p}), null, null);
        }
        if (c.index_sources.has(i)) {
            lay.idx_wq_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.wq_b", .{p}), null, null);
            const name = try std.fmt.bufPrint(&nb, "{s}.attn.indexer.weights_proj.weight", .{p});
            var kb: [256]u8 = undefined;
            const f32s = try L.entry(try plan.plainKey(&kb, name, .f32));
            lay.idx_proj = try L.upload(name, f32s);
            // Model.__init__: idx_proj.to(float16), the stored f16 values back (the f32 copy widened them exactly)
            try L.aux.resize(L.gpa, f32s.len / 2);
            for (0..f32s.len / 4) |k| {
                const v: f32 = @bitCast(std.mem.readInt(u32, f32s[4 * k ..][0..4], .little));
                std.mem.writeInt(u16, L.aux.items[2 * k ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little);
            }
            lay.idx_proj_h = try L.upload(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.weights_proj_h", .{p}), L.aux.items);
            if (c.kv_sources.has(i)) {
                lay.idx_wk = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.wk", .{p}), null, null);
                lay.idx_k_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.k_norm.weight", .{p}), null);
            }
        }
        lay.gate_w = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn.gate.weight", .{p}), .f16);
        lay.gate_b = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn.gate.bias", .{p}), .f32);
        const xp = s.expertParts(c);
        lay.experts = try L.experts(p, n_experts, xp.half, xp.gu, xp.dcols, c.hidden);
        if (c.engram_layers.has(i)) {
            lay.engram_wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.engram.wkv", .{p}), s.engramCols(c), s.range(s.engramRows(c)));
            if (s.pair != null) lay.engram_wkv.?.plan_n = @intCast((c.hc + 1) * c.hidden);
            var kb: [256]u8 = undefined;
            const q = try L.gpa.dupe(u8, try L.entry(try plan.plainKey(&kb, try std.fmt.bufPrint(&nb, "{s}.engram.q_weight", .{p}), .f32)));
            defer L.gpa.free(q);
            const k = try L.entry(try plan.plainKey(&kb, try std.fmt.bufPrint(&nb, "{s}.engram.k_weight", .{p}), .f32));
            if (q.len != k.len) return error.UnexpectedTensor;
            // q * k on the GPU in f32: one rounding a product, the same on the host
            for (0..q.len / 4) |j| {
                const a: f32 = @bitCast(std.mem.readInt(u32, q[4 * j ..][0..4], .little));
                const b: f32 = @bitCast(std.mem.readInt(u32, k[4 * j ..][0..4], .little));
                std.mem.writeInt(u32, q[4 * j ..][0..4], @bitCast(a * b), .little);
            }
            lay.engram_qk = try L.upload(try std.fmt.bufPrint(&nb, "{s}.engram.qk", .{p}), q);
        }
        return lay;
    }
};

/// Every tensor of rank `s` on device `d`, from `src` (weights.py _load, in its order).
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, src: Source, c: Config, s: plan.Split, dspark: bool) !Weights {
    var w: Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    var L: Loader = .{ .gpa = gpa, .d = d, .src = src, .w = &w };
    defer {
        L.host.deinit(gpa);
        L.aux.deinit(gpa);
    }
    w.embed = try L.plain("embed.weight", null);
    w.norm = try L.plain("norm.weight", null);
    const vr = s.headCols(c);
    w.head = try L.linear("head", vr, null);
    if (s.pair != null) w.head.plan_n = @intCast(s.vocab(c)); // the vocabulary half's plan
    w.vocab_lo = vr[0];
    w.vocab_hi = vr[1];
    w.layers = try gpa.alloc(Layer, c.layers);
    var nb: [64]u8 = undefined;
    for (0..c.layers) |i| w.layers[i] = try L.block(c, s, try std.fmt.bufPrint(&nb, "layers.{d}", .{i}), i, c.experts);
    if (dspark and c.dspark_block > 0) {
        const blocks = try gpa.alloc(Layer, c.draft_layers);
        errdefer gpa.free(blocks);
        // DSpark stays TP2 inside the pair on a 2D node (weights.py _load_2d)
        const tp2: plan.Split = .{ .rank = s.rank, .world = s.world };
        for (0..c.draft_layers) |j| blocks[j] = try L.block(c, tp2, try std.fmt.bufPrint(&nb, "mtp.{d}", .{j}), c.layers + j, c.draft_experts);
        var lb: [64]u8 = undefined;
        const last = try std.fmt.bufPrint(&lb, "mtp.{d}", .{c.draft_layers - 1});
        var tb: [128]u8 = undefined;
        w.dspark = .{
            .blocks = blocks,
            .main_proj = try L.linear("mtp.0.main_proj", null, null),
            .main_norm = try L.plain("mtp.0.main_norm.weight", null),
            .norm = try L.plain(try std.fmt.bufPrint(&tb, "{s}.norm.weight", .{last}), null),
            .markov_embed = try L.plain(try std.fmt.bufPrint(&tb, "{s}.markov_head.embed.weight", .{last}), null),
            .markov_head = try L.plain(try std.fmt.bufPrint(&tb, "{s}.markov_head.head.weight", .{last}), null),
            .conf = try L.plain(try std.fmt.bufPrint(&tb, "{s}.confidence_head.proj.weight", .{last}), null),
        };
    }
    return w;
}

test "strips: each 128-column block's tiles in k order" {
    // kt 2, nt 16 (two blocks of 8 tiles), one byte a tile
    var src: [32]u8 = undefined;
    for (&src, 0..) |*x, i| x.* = @intCast(i);
    var out: [32]u8 = undefined;
    strips(&src, &out, 2, 16, 1);
    // block 0: k row 0 tiles 0..7, k row 1 tiles 16..23; block 1: tiles 8..15, then 24..31
    const want = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 };
    try std.testing.expectEqualSlices(u8, &want, &out);
}
