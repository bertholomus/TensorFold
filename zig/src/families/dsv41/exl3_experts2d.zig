//! A 2D node's routed experts on a prompt chunk (exl3/experts2d.py gateup and down on its prompt path): this node's
//! gate / up blocks of every picked slot with their epilogue, then, once the caller's column exchange has assembled the
//! TP2 rank's whole intermediate, down over it for this node's output columns and the combine. The same eight launches
//! as exl3_experts.prompt, in two calls, with the kernel settings pinned from the TP2 half's shapes (Plan2D.of(D, I):
//! the half's I, not this node's narrower width) and only the launched widths narrowed, so every output column has the
//! bits of the TP2 rank's partial.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const X = @import("exl3_experts.zig");

fn launch(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
}

fn i32a(a: *cuda.Args, v: usize) void {
    a.add(@as(c_int, @intCast(v)));
}

/// The settings of a 2D node's prompt launches: TP2's for the half (D, I), checked as exl3_experts.prompt checks
/// them, and the widths launched here.
pub const Plan = struct {
    d: usize, // the model width (gate / up K)
    i: usize, // the TP2 half's intermediate (down K)
    ig: usize, // gate / up's output columns here
    dn: usize, // down's output columns here
    cgu: [4]usize,
    cd: [4]usize,

    pub fn of(ex: weights.Experts) !Plan {
        const d: usize = ex.dims;
        const i: usize = ex.down_k;
        const ig: usize = ex.width;
        const dn: usize = ex.down_n;
        if (i % 128 != 0 or d % 128 != 0 or ig % 128 != 0 or dn % 128 != 0 or ig > i or dn > d) return error.UnsupportedShape;
        const cgu = try X.config(d, i, true);
        const cd = try X.config(i, d, false);
        // the served fast path as on TP2: mma for gate / up (K chains a multiple of MMA_KB), grouped_rows for down
        const chain_gu = (d / 16) / (cgu[1] * cgu[2]);
        if ((d / 16) % (cgu[1] * cgu[2]) != 0 or chain_gu % 4 != 0) return error.UnsupportedShape;
        const chain_d = (i / 16) / (cd[1] * cd[2]);
        if ((i / 16) % (cd[1] * cd[2]) == 0 and chain_d % 4 == 0) return error.UnsupportedShape; // down would take mma
        if (cd[2] != 1) return error.UnsupportedShape;
        if (ig % X.mma_cols != 0) return error.UnsupportedShape;
        return .{ .d = d, .i = i, .ig = ig, .dn = dn, .cgu = cgu, .cd = cd };
    }
};

/// experts2d.gateup on a prompt chunk (R >= 64): the grouping (TP2's, by pick only), the rotated rows, gate and up for
/// this node's columns, the epilogue (TP2's values on them) into sc.xd [R * slots, ig] fp16. x bf16 [R, D] (row stride
/// x_stride elements), pick int32 [R, slots]; sc sized for this node's width.
pub fn gateUp(k: *const X.Kernels, s: cuda.Stream, ex: weights.Experts, sc: X.Scratch, x: u64, x_stride: usize, pick: u64, r: usize, limit: f32) !void {
    const pl = try Plan.of(ex);
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r < X.exact_rows or r > sc.rows) return error.NotAPromptChunk;
    if (slots > 32) return error.UnsupportedShape;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = X.memberStride(r);
    const n_gu = X.listLen(p, maxu, X.mma_rows_gu);
    const n_d = X.listLen(p, maxu, X.rows_rows_d);
    const mma = k.mma[X.range(ex.k2_gu[0], ex.k2_gu[1])] orelse return error.MissingKernel;

    var a: cuda.Args = .{};
    a.add(pick);
    a.add(sc.counts);
    i32a(&a, p);
    try launch(k.group_count, s, .{ e, 1, 1 }, X.place_threads, &a);

    a = .{};
    for ([_]u64{ pick, sc.counts, sc.ids, sc.count, sc.members }) |v| a.add(v);
    for ([_]usize{ p, slots, e, maxm }) |v| i32a(&a, v);
    try launch(k.group_place, s, .{ e, 1, 1 }, X.place_threads, &a);

    // both work lists here: down (after the exchange) uses the same grouping
    a = .{};
    for ([_]u64{ sc.counts, sc.ids, sc.count }) |v| a.add(v);
    i32a(&a, maxu);
    a.add(sc.work_gu);
    i32a(&a, X.mma_rows_gu);
    i32a(&a, n_gu);
    a.add(sc.work_d);
    i32a(&a, X.rows_rows_d);
    i32a(&a, n_d);
    a.add(@as(c_int, -1)); // skip_e: no shared-dense GEMM (none on 2D either)
    try launch(k.work_list, s, .{ 1, 1, 1 }, X.list_threads, &a);

    a = .{};
    a.add(x);
    i32a(&a, x_stride);
    for ([_]u64{ pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu }) |v| a.add(v);
    for ([_]usize{ pl.d, slots, e }) |v| i32a(&a, v);
    try launch(k.rot_in, s, .{ p, pl.d / 128, 2 }, 32, &a);

    // gate and up for this node's columns: grid (1, ig / 64, pairs x 2 matrices), TP2's K splits and warps
    a = .{};
    for ([_]u64{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ pl.d, pl.ig, p, pl.cgu[2], pl.cgu[1], maxm, slots, n_gu }) |v| i32a(&a, v);
    a.add(sc.work_gu);
    if (n_gu > 0) try launch(mma, s, .{ 1, pl.ig / X.mma_cols, n_gu * 2 }, 256, &a);

    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| a.add(v);
    for ([_]usize{ p, pl.ig, 1, e }) |v| i32a(&a, v);
    a.add(limit);
    a.add(@as(c_int, X.act_f32));
    try launch(k.gateup_epilogue, s, .{ p, pl.ig / 128, 1 }, 32, &a);
}

/// experts2d.down on a prompt chunk: down over the TP2 rank's whole intermediate xd_full [R * slots, I] fp16 for this
/// node's output columns, then the served combine (slots in order) into out fp32 [R, dn]. Uses the grouping and work
/// lists gateUp made in sc.
pub fn down(k: *const X.Kernels, s: cuda.Stream, ex: weights.Experts, sc: X.Scratch, xd_full: u64, pick: u64, wts: u64, out: u64, r: usize) !void {
    const pl = try Plan.of(ex);
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r < X.exact_rows or r > sc.rows) return error.NotAPromptChunk;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = X.memberStride(r);
    const n_d = X.listLen(p, maxu, X.rows_rows_d);
    const dfn = k.rows[X.range(ex.k2_d[0], ex.k2_d[1])] orelse return error.MissingKernel;

    // down: grid (1, dn / 128, pairs), 4 warps, two 16-row member tiles a program, one split (TP2's settings)
    var a: cuda.Args = .{};
    for ([_]u64{ xd_full, xd_full, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ pl.i, pl.dn, p, 1, maxm, slots, n_d }) |v| i32a(&a, v);
    a.add(sc.work_d);
    if (n_d > 0) try launch(dfn, s, .{ 1, pl.dn / 128, n_d }, 4 * 32, &a);

    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_d, sc.no_y, wts, 0, out }) |v| a.add(v);
    for ([_]usize{ p, pl.dn, 1, e, slots, 0 }) |v| i32a(&a, v);
    try launch(k.down_combine, s, .{ r, pl.dn / 128, 1 }, 32 * slots, &a);
}

test "a 2D node's prompt experts keep the TP2 half's settings" {
    var ex: weights.Experts = undefined;
    ex.dims = 5120;
    ex.down_k = 1152;
    ex.count = 385;
    // node of pair 0: blocks 0-4 of the half, down columns 0-2559
    ex.width = 640;
    ex.down_n = 2560;
    const p0 = try Plan.of(ex);
    try std.testing.expectEqual([4]usize{ 8, 4, 4, 1 }, p0.cgu);
    try std.testing.expectEqual([4]usize{ 8, 4, 1, 1 }, p0.cd);
    try std.testing.expectEqual(@as(usize, 1152), p0.i);
    // pair 1: blocks 5-8
    ex.width = 512;
    const p1 = try Plan.of(ex);
    try std.testing.expectEqual(p0.cgu, p1.cgu);
    try std.testing.expectEqual(@as(usize, 512), p1.ig);
    // TP2 itself: the whole half and every column
    ex.width = 1152;
    ex.down_n = 5120;
    const t = try Plan.of(ex);
    try std.testing.expectEqual(try X.config(5120, 1152, true), t.cgu);
    // a part that is not whole 128-blocks is refused
    ex.width = 576;
    try std.testing.expectError(error.UnsupportedShape, Plan.of(ex));
}
