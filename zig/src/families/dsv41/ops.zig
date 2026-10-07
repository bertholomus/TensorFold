//! The torch operations of the served forward that only move or widen bytes, on TensorFold's own torch-op kernels
//! (zig/kernels/cuda/torch_ops: pointwise.cu, movement.cu; ours in dsv41_ops.cu) with their development launchers'
//! geometry: `.float()` of bf16 and fp16 (exact), `.to(bf16)` of fp32 (round to nearest even), `.contiguous()` and row
//! copies, row gathers and scatters by index (`t[idx]`, `t[idx] = rows`), and the integer selections of the indexer
//! (torch's top-k of int64 keys, kernels.topk_indices). Their bytes are the same as torch's by construction; the layer
//! gate checks them.
const std = @import("std");
const cuda = @import("cuda");

/// launch_shape.h's tf_launch_blocks: a block a `threads` elements, at most 65535.
fn blocks(count: usize, threads: usize) u32 {
    return @intCast(@min((count + threads - 1) / threads, 65535));
}

pub const Ops = struct {
    pointwise: cuda.Module,
    movement: cuda.Module,
    family: cuda.Module,
    to_f32: cuda.Function,
    f16_to_f32: cuda.Function,
    add2_bf16: cuda.Function,
    topk_i64: cuda.Function,
    topk_indices: cuda.Function,
    scatter: cuda.Function,
    to_bf16: cuda.Function,
    strided: cuda.Function,
    gather: cuda.Function,

    /// The three images (cuda.kernels.torch_pointwise, torch_movement and dsv41_ops, or the fatbins' bytes).
    pub fn load(d: *const cuda.Driver, pointwise: []const u8, movement: []const u8, family: []const u8) !Ops {
        var pw = try cuda.Module.load(d, pointwise);
        errdefer pw.unload();
        var mv = try cuda.Module.load(d, movement);
        errdefer mv.unload();
        var fm = try cuda.Module.load(d, family);
        errdefer fm.unload();
        return .{
            .pointwise = pw,
            .movement = mv,
            .family = fm,
            .to_f32 = try pw.function("tf_bf16_to_f32_kernel"),
            .f16_to_f32 = try fm.function("tf_ds_f16_to_f32_kernel"),
            .add2_bf16 = try fm.function("tf_ds_add2_bf16_kernel"),
            .topk_i64 = try fm.function("tf_ds_topk_i64_kernel"),
            .topk_indices = try fm.function("tf_ds_topk_indices_kernel"),
            .scatter = try fm.function("tf_ds_scatter_rows_kernel"),
            .to_bf16 = try pw.function("tf_f32_to_bf16_kernel"),
            .strided = try mv.function("tf_strided_copy_kernel"),
            .gather = try mv.function("tf_gather_rows_kernel"),
        };
    }

    pub fn unload(o: *Ops) void {
        o.pointwise.unload();
        o.movement.unload();
        o.family.unload();
        o.* = undefined;
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: usize, block: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = block } }, s, args);
    }

    /// `.float()` of `count` bf16 values (exact).
    pub fn toF32(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.to_f32, s, blocks(count, 256), 256, &a);
    }

    /// `.float()` of `count` fp16 values (exact).
    pub fn f16ToF32(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.f16_to_f32, s, blocks(count, 256), 256, &a);
    }

    /// Comm.sum of two ranks' fp32 partials then `.to(bf16)`: out = bf16(g0 + g1), one fp32 add in rank order.
    pub fn add2Bf16(o: *const Ops, s: cuda.Stream, g0: u64, g1: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(g0);
        a.add(g1);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.add2_bf16, s, blocks(count, 256), 256, &a);
    }

    /// keys.topk(k, sorted=False).values of unique int64 keys [rows, n] (row stride ks) into top [rows, k], the set in no
    /// particular order (_topk_finish sorts the indices).
    pub fn topkI64(o: *const Ops, s: cuda.Stream, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) !void {
        if (rows == 0) return;
        if (k > n) return error.BadTopK;
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(c_longlong, @intCast(ks)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(top);
        try cuda.launch.launch(o.topk_i64, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// kernels.topk_indices(score, k) with torch.where(top < vis, top, -1): each row's k highest fp32 scores' columns
    /// ascending (ties to the lower column), a column at or past vis[r] as -1. score [rows, n] (row stride ss), vis
    /// [rows] int64, out [rows, k] int64.
    pub fn topkIndices(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, rows: usize, n: usize, k: usize, vis: u64, out: u64) !void {
        if (rows == 0) return;
        if (k > n or k > 1024) return error.BadTopK;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(vis);
        a.add(out);
        try cuda.launch.launch(o.topk_indices, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// dst row idx[i] = src row i (`t[idx] = rows`, distinct int64 indices): `rows` rows of `bytes` bytes, src rows
    /// `src_ld` bytes apart, dst rows `dst_ld`.
    pub fn scatterRows(o: *const Ops, s: cuda.Stream, src: u64, src_ld: usize, idx: u64, dst: u64, dst_ld: usize, bytes: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_longlong, @intCast(src_ld)));
        a.add(idx);
        a.add(dst);
        a.add(@as(c_longlong, @intCast(dst_ld)));
        a.add(@as(c_longlong, @intCast(bytes)));
        a.add(@as(c_int, @intCast(rows)));
        try go(o.scatter, s, rows, 256, &a);
    }

    /// `.to(torch.bfloat16)` of `count` fp32 values (round to nearest even).
    pub fn toBf16(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.to_bf16, s, blocks(count, 256), 256, &a);
    }

    /// `rows` rows of `bytes` bytes from src (rows `src_ld` bytes apart) to dst (rows `dst_ld` bytes apart).
    pub fn copyRows(o: *const Ops, s: cuda.Stream, src: u64, src_ld: usize, dst: u64, dst_ld: usize, bytes: usize, rows: usize) !void {
        if (rows == 0) return;
        if (src_ld < bytes or dst_ld < bytes) return error.Overlap;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        // tf_strided_copy (outer, middle, inner, source_outer, source_middle, destination_outer, destination_middle):
        // a row a block
        for ([_]u64{ rows, 1, bytes, src_ld, bytes, dst_ld, bytes }) |v| a.add(v);
        try go(o.strided, s, rows, 256, &a);
    }

    /// dst row i = src row idx[i] (int64 indices, rows of `bytes` bytes); an index outside [0, src_rows) sets the
    /// caller's zeroed uint32 flag `invalid` and leaves that row as it was.
    pub fn gatherRows(o: *const Ops, s: cuda.Stream, src: u64, src_rows: usize, idx: u64, dst: u64, bytes: usize, rows: usize, invalid: u64) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(idx);
        a.add(@as(u64, rows));
        a.add(@as(u64, src_rows));
        a.add(@as(u64, bytes));
        a.add(invalid);
        try go(o.gather, s, rows, 256, &a);
    }
};

test "launch geometry of the development launchers" {
    try std.testing.expectEqual(@as(u32, 1), blocks(1, 256));
    try std.testing.expectEqual(@as(u32, 2), blocks(257, 256));
    try std.testing.expectEqual(@as(u32, 65535), blocks(1 << 40, 256));
}
