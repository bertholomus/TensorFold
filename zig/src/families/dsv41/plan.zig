//! The TP split (DESIGN.md section 2) as the tensors a rank loads, in the Python loader's order and with its keys
//! (weights.py _load / load_block), so the Zig loader reads the lane's rank cache entry by entry.
//!
//! Heads 32 a rank (wq_b columns, wo_a groups, wo_b input rows, sinks), experts' intermediate 1152 a rank (gate and up
//! by output column, down by input row), the head's vocabulary halves, Engram wkv by hash column; the rest replicated.
//! On four nodes the exact 2D split (DESIGN-TP4.md 4.2, weights.py load_block_2d): a node is TP2 rank r inside pair p
//! and keeps pair p's whole 128-column blocks of each of its rank's split outputs; DSpark stays TP2 inside the pair.
const std = @import("std");
const Config = @import("config.zig").Config;
const DType = @import("rank_cache.zig").DType;

/// What a split means for one rank.
pub const Split = struct {
    rank: u32,
    world: u32,
    /// The 2D split's pair (node g = 2 p + r, `world` 2 inside the pair); null: plain TP over `world`.
    pair: ?u32 = null,
    /// TF_DS_2D_DOWN: pair 0's blocks of the experts' down outputs (hidden / 128 of them; 20 is even).
    down0: usize = 20,

    pub fn heads(s: Split, c: Config) usize {
        return c.heads / s.world;
    }
    pub fn groups(s: Split, c: Config) usize {
        return c.o_groups / s.world;
    }
    pub fn inter(s: Split, c: Config) usize {
        return c.expert_width / s.world;
    }
    pub fn vocab(s: Split, c: Config) usize {
        return c.vocab / s.world;
    }
    /// Engram wkv input rows a rank: its hash columns ((n-gram orders - 1) x heads over world) times the head width.
    pub fn engramRows(s: Split, c: Config) usize {
        return (c.engram_ngram - 1) * c.engram_heads / s.world * c.engram_head_dim;
    }
    pub fn range(s: Split, per: usize) [2]usize {
        return .{ s.rank * per, (s.rank + 1) * per };
    }
    /// The pair's part of `r` in whole 128-column blocks: the first `first` (default the larger half) are pair 0's.
    pub fn cut(s: Split, r: [2]usize, first: ?usize) [2]usize {
        const n = (r[1] - r[0]) / 128;
        const mid = r[0] + (first orelse (n + 1) / 2) * 128;
        return if (s.pair.? == 0) .{ r[0], mid } else .{ mid, r[1] };
    }
    /// The pair's part of `r` on a 2D split, else `r` (the plain TP slice).
    fn part(s: Split, r: [2]usize, first: ?usize) [2]usize {
        return if (s.pair != null) s.cut(r, first) else r;
    }
};

/// One tensor a rank loads: the loader's key, the dtype it is stored in (null: the checkpoint's own), and for an EXL3
/// trellis its K / 16 and N / 16 when the split fixes them.
pub const Want = struct {
    key: []const u8,
    dtype: ?DType = null,
    k16: ?usize = null,
    n16: ?usize = null,
};

/// plain(name, dtype)'s key: f"{name}|{dtype}", the dtype as torch prints it.
pub fn plainKey(buf: []u8, tensor: []const u8, dtype: ?DType) ![]const u8 {
    const t = if (dtype) |d| switch (d) {
        .f32 => "torch.float32",
        .f16 => "torch.float16",
        else => return error.UnsupportedDType,
    } else "None";
    return std.fmt.bufPrint(buf, "{s}|{s}", .{ tensor, t });
}

/// exl3_parts(prefix, cols, rows)'s key for one part ("tr", "suh" or "svh"); a range prints as Python's tuple.
pub fn partKey(buf: []u8, prefix: []const u8, cols: ?[2]usize, rows: ?[2]usize, part: []const u8) ![]const u8 {
    var cb: [48]u8 = undefined;
    var rb: [48]u8 = undefined;
    const cs = if (cols) |c| try std.fmt.bufPrint(&cb, "({d}, {d})", .{ c[0], c[1] }) else "None";
    const rs = if (rows) |r| try std.fmt.bufPrint(&rb, "({d}, {d})", .{ r[0], r[1] }) else "None";
    return std.fmt.bufPrint(buf, "{s}|{s}|{s}|{s}", .{ prefix, cs, rs, part });
}

const Builder = struct {
    a: std.mem.Allocator,
    out: std.ArrayList(Want) = .empty,

    fn plain(b: *Builder, tensor: []const u8, dtype: ?DType) !void {
        var buf: [256]u8 = undefined;
        try b.out.append(b.a, .{ .key = try b.a.dupe(u8, try plainKey(&buf, tensor, dtype)), .dtype = dtype });
    }

    fn exl3(b: *Builder, prefix: []const u8, cols: ?[2]usize, rows: ?[2]usize, k: ?usize, n: ?usize) !void {
        const k16: ?usize = if (rows) |r| (r[1] - r[0]) / 16 else if (k) |x| x / 16 else null;
        const n16: ?usize = if (cols) |c| (c[1] - c[0]) / 16 else if (n) |x| x / 16 else null;
        for ([_][]const u8{ "tr", "suh", "svh" }) |part| {
            var buf: [256]u8 = undefined;
            const key = try b.a.dupe(u8, try partKey(&buf, prefix, cols, rows, part));
            const is_tr = std.mem.eql(u8, part, "tr");
            try b.out.append(b.a, .{ .key = key, .dtype = if (is_tr) .i16 else .f16, .k16 = if (is_tr) k16 else null, .n16 = if (is_tr) n16 else null });
        }
    }

    fn fmt(b: *Builder, comptime f: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(b.a, f, args);
    }
};

/// One block, a target layer (`layers.i`) or a DSpark stage (`mtp.j`, i = layers + j), as load_block takes it.
fn block(b: *Builder, c: Config, s: Split, p: []const u8, i: usize, experts: usize) !void {
    const d = c.hidden;
    const hl = s.heads(c);
    const gl = s.groups(c);
    for ([_][]const u8{ "hc_attn", "hc_ffn" }) |hc| for ([_][]const u8{ "fn", "scale", "base" }) |part| {
        try b.plain(try b.fmt("{s}.{s}_{s}", .{ p, hc, part }), .f32);
    };
    try b.plain(try b.fmt("{s}.attn_norm.weight", .{p}), null);
    try b.plain(try b.fmt("{s}.ffn_norm.weight", .{p}), null);
    try b.exl3(try b.fmt("{s}.attn.wq_a", .{p}), null, null, d, c.q_lora);
    try b.plain(try b.fmt("{s}.attn.q_norm.weight", .{p}), null);
    try b.exl3(try b.fmt("{s}.attn.wq_b", .{p}), s.part(s.range(hl * c.head_dim), null), null, c.q_lora, null);
    try b.exl3(try b.fmt("{s}.attn.wkv", .{p}), null, null, d, c.head_dim);
    try b.plain(try b.fmt("{s}.attn.kv_norm.weight", .{p}), null);
    try b.plain(try b.fmt("{s}.attn.attn_sink", .{p}), .f32);
    // 2D: the pair's half of its rank's groups (split2d.py wo_a_groups)
    const g0 = s.rank * gl + if (s.pair) |pp| pp * (gl / 2) else 0;
    for (g0..g0 + if (s.pair != null) gl / 2 else gl) |g| {
        try b.exl3(try b.fmt("{s}.attn.wo_a.slice.{d}", .{ p, g }), null, null, c.heads / c.o_groups * c.head_dim, c.o_lora);
    }
    const wo_b_cols: ?[2]usize = if (s.pair != null) s.cut(.{ 0, d }, null) else null;
    try b.exl3(try b.fmt("{s}.attn.wo_b", .{p}), wo_b_cols, s.range(gl * c.o_lora), null, d);
    if (c.kv_sources.has(i)) {
        try b.exl3(try b.fmt("{s}.attn.compressor.wkv", .{p}), null, null, d, null);
        try b.plain(try b.fmt("{s}.attn.compressor.norm.weight", .{p}), null);
        if (c.ratios[i] > 1) try b.exl3(try b.fmt("{s}.attn.compressor.wgate", .{p}), null, null, d, null);
    }
    if (c.index_sources.has(i)) {
        try b.exl3(try b.fmt("{s}.attn.indexer.wq_b", .{p}), null, null, c.q_lora, null);
        try b.plain(try b.fmt("{s}.attn.indexer.weights_proj.weight", .{p}), .f32);
        if (c.kv_sources.has(i)) {
            try b.exl3(try b.fmt("{s}.attn.indexer.wk", .{p}), null, null, c.head_dim, c.index_head_dim);
            try b.plain(try b.fmt("{s}.attn.indexer.k_norm.weight", .{p}), null);
        }
    }
    try b.plain(try b.fmt("{s}.ffn.gate.weight", .{p}), .f16);
    try b.plain(try b.fmt("{s}.ffn.gate.bias", .{p}), .f32);
    const half = s.range(s.inter(c));
    const gate_up = s.part(half, null); // 2D: blocks 0-4 (pair 0) or 5-8 (pair 1) of the rank's 9 (TF_DS_2D_GU=first)
    const down_cols: ?[2]usize = if (s.pair != null) s.cut(.{ 0, d }, s.down0) else null;
    for (0..experts + 1) |e| {
        const ep = if (e < experts) try b.fmt("{s}.ffn.experts.{d}", .{ p, e }) else try b.fmt("{s}.ffn.shared_experts", .{p});
        try b.exl3(try b.fmt("{s}.w1", .{ep}), gate_up, null, d, null);
        try b.exl3(try b.fmt("{s}.w3", .{ep}), gate_up, null, d, null);
        try b.exl3(try b.fmt("{s}.w2", .{ep}), down_cols, half, null, d);
    }
    if (c.engram_layers.has(i)) {
        const engram_cols: ?[2]usize = if (s.pair != null) s.cut(.{ 0, 5 * d }, null) else null;
        try b.exl3(try b.fmt("{s}.engram.wkv", .{p}), engram_cols, s.range(s.engramRows(c)), null, null);
        try b.plain(try b.fmt("{s}.engram.q_weight", .{p}), .f32);
        try b.plain(try b.fmt("{s}.engram.k_weight", .{p}), .f32);
    }
}

/// Every tensor rank `s` loads, in load order (`dspark`: the DSpark stages and heads too). Keys live in `a`.
pub fn wants(a: std.mem.Allocator, c: Config, s: Split, dspark: bool) ![]Want {
    var b: Builder = .{ .a = a };
    try b.plain("embed.weight", null);
    try b.plain("norm.weight", null);
    try b.exl3("head", s.part(s.range(s.vocab(c)), null), null, c.hidden, null);
    for (0..c.layers) |i| try block(&b, c, s, try b.fmt("layers.{d}", .{i}), i, c.experts);
    if (dspark and c.dspark_block > 0) {
        // DSpark stays TP2 inside the pair on a 2D split (weights.py _load_2d: load_block at world 2)
        const tp2: Split = .{ .rank = s.rank, .world = s.world };
        for (0..c.draft_layers) |j| try block(&b, c, tp2, try b.fmt("mtp.{d}", .{j}), c.layers + j, c.draft_experts);
        const last = try b.fmt("mtp.{d}", .{c.draft_layers - 1});
        try b.exl3("mtp.0.main_proj", null, null, c.dspark_taps.len * c.hidden, c.hidden);
        try b.plain("mtp.0.main_norm.weight", null);
        try b.plain(try b.fmt("{s}.norm.weight", .{last}), null);
        try b.plain(try b.fmt("{s}.markov_head.embed.weight", .{last}), null);
        try b.plain(try b.fmt("{s}.markov_head.head.weight", .{last}), null);
        try b.plain(try b.fmt("{s}.confidence_head.proj.weight", .{last}), null);
    }
    return b.out.items;
}

test "a rank's tensors in load order: the split's ranges, keys as Python prints them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: @import("config.zig").Why = .{};
    const c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    const w = try wants(a, c, .{ .rank = 1, .world = 2 }, true);
    try std.testing.expectEqualStrings("embed.weight|None", w[0].key);
    try std.testing.expectEqualStrings("head|(64640, 129280)|None|tr", w[2].key);
    try std.testing.expectEqual(@as(?usize, 4040), w[2].n16);
    try std.testing.expectEqualStrings("layers.0.hc_attn_fn|torch.float32", w[5].key);
    var seen_wq_b = false;
    var seen_down = false;
    var seen_engram = false;
    var wo_a: usize = 0;
    for (w) |x| {
        if (std.mem.eql(u8, x.key, "layers.0.attn.wq_b|(16384, 32768)|None|tr")) seen_wq_b = x.n16.? == 1024 and x.k16.? == 80;
        if (std.mem.eql(u8, x.key, "layers.3.ffn.experts.383.w2|None|(1152, 2304)|tr")) seen_down = x.k16.? == 72 and x.n16.? == 320;
        if (std.mem.eql(u8, x.key, "layers.14.engram.wkv|None|(3072, 6144)|tr")) seen_engram = x.k16.? == 192;
        if (std.mem.startsWith(u8, x.key, "layers.0.attn.wo_a.slice.") and std.mem.endsWith(u8, x.key, "|tr")) wo_a += 1;
    }
    try std.testing.expect(seen_wq_b and seen_down and seen_engram);
    try std.testing.expectEqual(@as(usize, 4), wo_a);
    try std.testing.expectEqualStrings("mtp.2.confidence_head.proj.weight|None", w[w.len - 1].key);
}

test "the 2D split: node 2 (rank 0 of pair 1) keeps pair 1's blocks, DSpark stays TP2" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: @import("config.zig").Why = .{};
    const c = try @import("config.zig").parse(a, @import("config.zig").test_config, &why);
    const w = try wants(a, c, .{ .rank = 0, .world = 2, .pair = 1 }, true);
    const tp2 = try wants(a, c, .{ .rank = 0, .world = 2 }, true);
    try std.testing.expectEqual(tp2.len - 2 * 3 * c.layers, w.len); // two wo_a groups a layer, not four
    try std.testing.expectEqualStrings("head|(32384, 64640)|None|tr", w[2].key);
    const expect = [_][]const u8{
        "layers.0.attn.wq_b|(8192, 16384)|None|tr",
        "layers.0.attn.wo_a.slice.2|None|None|tr",
        "layers.0.attn.wo_a.slice.3|None|None|tr",
        "layers.0.attn.wo_b|(2560, 5120)|(0, 4096)|tr",
        "layers.0.ffn.experts.0.w1|(640, 1152)|None|tr",
        "layers.0.ffn.experts.0.w2|(2560, 5120)|(0, 1152)|tr",
        "layers.1.engram.wkv|(12800, 25600)|(0, 3072)|tr",
        "mtp.0.attn.wq_b|(0, 16384)|None|tr",
        "mtp.0.attn.wo_a.slice.3|None|None|tr",
    };
    for (expect) |k| {
        var found = false;
        for (w) |x| found = found or std.mem.eql(u8, x.key, k);
        if (!found) std.debug.print("missing {s}\n", .{k});
        try std.testing.expect(found);
    }
    for (w) |x| try std.testing.expect(!std.mem.eql(u8, x.key, "layers.0.attn.wo_a.slice.1|None|None|tr"));
    const p0 = try wants(a, c, .{ .rank = 1, .world = 2, .pair = 0 }, false);
    var down0 = false;
    for (p0) |x| down0 = down0 or std.mem.eql(u8, x.key, "layers.5.ffn.shared_experts.w2|(0, 2560)|(1152, 2304)|tr");
    try std.testing.expect(down0);
}
