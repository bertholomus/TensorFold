//! The served forward's torch arithmetic at torch's own rounding points (zig/kernels/cuda/dsv41_torch.cu), byte-equal to
//! the served build's torch ops on GB10 (tools/dsv41-zig/rec/zrec_torchlab.py): ops.rms_norm with its mean in torch's
//! reduction order, _compress's ratio-2 softmax and weighted sum, ops.rope_'s complex multiply, the indexer's bf16
//! head weights and the DSpark taps' mean over the streams.
const std = @import("std");
const cuda = @import("cuda");

/// launch_shape.h's tf_launch_blocks: a block a `threads` elements, at most 65535.
fn blocks(count: usize, threads: usize) u32 {
    return @intCast(@min((count + threads - 1) / threads, 65535));
}

fn lastPow2(n: usize) usize {
    return if (n == 0) 0 else @as(usize, 1) << @intCast(std.math.log2_int(usize, n));
}

/// Reduce.cuh's block width for the mean over a contiguous last dim of d floats (vectorized: d / 4 vector lanes) at
/// `rows` outputs, 512 threads a block at most (ReduceConfig.set_block_dimension).
pub fn rmsLanes(d: usize, rows: usize) usize {
    const dim0 = d / 4;
    const d0 = if (dim0 < 512) lastPow2(dim0) else 512;
    const d1 = if (rows < 512) lastPow2(rows) else 512;
    const bw: usize = @min(d0, 32);
    const bh: usize = @min(d1, 512 / bw);
    return @min(d0, 512 / bh);
}

pub const Exact = struct {
    module: cuda.Module,
    k_rms_norm: cuda.Function,
    k_compress2: cuda.Function,
    k_rope: cuda.Function,
    k_bf16_scale: cuda.Function,
    k_hc_mean4: cuda.Function,

    /// cuda.kernels.dsv41_torch (or the fatbin's bytes).
    pub fn load(d: *const cuda.Driver, image: []const u8) !Exact {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{
            .module = m,
            .k_rms_norm = try m.function("tf_ds_rms_norm_kernel"),
            .k_compress2 = try m.function("tf_ds_compress2_kernel"),
            .k_rope = try m.function("tf_ds_rope_kernel"),
            .k_bf16_scale = try m.function("tf_ds_bf16_scale_kernel"),
            .k_hc_mean4 = try m.function("tf_ds_hc_mean4_kernel"),
        };
    }

    pub fn unload(x: *Exact) void {
        x.module.unload();
        x.* = undefined;
    }

    /// ops.rms_norm(x, w, eps): out bf16 [rows, d] (row stride ldo) from x bf16 [rows, d] (row stride ldx), w bf16 [d];
    /// d a multiple of 4, at least 128 (torch's vectorized mean).
    pub fn rmsNorm(e: *const Exact, s: cuda.Stream, x: u64, ldx: usize, w: u64, out: u64, ldo: usize, rows: usize, d: usize, eps: f32) !void {
        if (rows == 0) return;
        if (d % 4 != 0 or d < 128) return error.UnsupportedShape;
        const lanes = rmsLanes(d, rows);
        const h: usize = @max(1, @min(512 / lanes, 64));
        // torch's mean factor: float(outputs) / numel
        const factor: f32 = @as(f32, @floatFromInt(rows)) / @as(f32, @floatFromInt(rows * d));
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(w);
        a.add(out);
        a.add(@as(c_longlong, @intCast(ldo)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(d)));
        a.add(factor);
        a.add(eps);
        const smem: u32 = if (lanes > 32) @intCast(lanes * h * 4) else 0;
        try cuda.launch.launch(e.k_rms_norm, .{ .grid = .{ .x = @intCast((rows + h - 1) / h) }, .block = .{ .x = @intCast(lanes), .y = @intCast(h) }, .shared = smem }, s, &a);
    }

    /// _compress at ratio 2: kv, score fp32 [g, 2, c] -> out bf16 [g, c] = bf16((kv * score.softmax(dim=1)).sum(1)).
    pub fn compress2(e: *const Exact, s: cuda.Stream, kv: u64, score: u64, out: u64, g: usize, c: usize) !void {
        if (g == 0) return;
        var a: cuda.Args = .{};
        a.add(kv);
        a.add(score);
        a.add(out);
        a.add(@as(c_int, @intCast(g)));
        a.add(@as(c_int, @intCast(c)));
        try cuda.launch.launch(e.k_compress2, .{ .grid = .{ .x = @min(4096, blocks(g * c, 256)) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// ops.rope_(x[..., off : off + 2 * half], complex(cos, sin)[pos]) of n rows of x bf16 (row stride ldx), in place;
    /// `inverse` rotates back (f.conj()).
    pub fn rope(e: *const Exact, s: cuda.Stream, x: u64, ldx: usize, off: usize, cos: u64, sin: u64, pos: u64, n: usize, half: usize, inverse: bool) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(@as(c_int, @intCast(off)));
        a.add(cos);
        a.add(sin);
        a.add(pos);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(half)));
        a.add(@as(c_int, @intFromBool(inverse)));
        try cuda.launch.launch(e.k_rope, .{ .grid = .{ .x = @min(4096, blocks(n * half, 256)) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// wl.to(bf16) * scale of `count` fp32 values (the indexer's head weights).
    pub fn bf16Scale(e: *const Exact, s: cuda.Stream, in: u64, out: u64, count: usize, scale: f32) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        a.add(scale);
        try cuda.launch.launch(e.k_bf16_scale, .{ .grid = .{ .x = blocks(count, 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// h.to(fp32).mean(1).to(bf16) of the streams h [rows, 4, d] bf16 into out [rows, d] (row stride os): a DSpark tap.
    pub fn hcMean4(e: *const Exact, s: cuda.Stream, h: u64, out: u64, os: usize, rows: usize, d: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(h);
        a.add(out);
        a.add(@as(c_longlong, @intCast(os)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(d)));
        try cuda.launch.launch(e.k_hc_mean4, .{ .grid = .{ .x = @min(4096, blocks(rows * d, 256)) }, .block = .{ .x = 256 } }, s, &a);
    }
};

/// The indexer's weight scale as Python computes it (idx_dim ** -0.5 * idx_heads ** -0.5) and torch passes it (fp32).
pub fn indexScale(idx_dim: usize, idx_heads: usize) f32 {
    const v = std.math.pow(f64, @floatFromInt(idx_dim), -0.5) * std.math.pow(f64, @floatFromInt(idx_heads), -0.5);
    return @floatCast(v);
}

test "torch's reduction block widths for the norms the forward takes" {
    // d 512: 32 lanes from 16 rows up, 64 at 8 rows, 128 at 4 or fewer (the recorded launches' blocks)
    try std.testing.expectEqual(@as(usize, 32), rmsLanes(512, 1024));
    try std.testing.expectEqual(@as(usize, 32), rmsLanes(512, 16));
    try std.testing.expectEqual(@as(usize, 64), rmsLanes(512, 8));
    try std.testing.expectEqual(@as(usize, 128), rmsLanes(512, 4));
    try std.testing.expectEqual(@as(usize, 128), rmsLanes(512, 1));
    // d 128: 32 vector lanes, always one warp
    try std.testing.expectEqual(@as(usize, 32), rmsLanes(128, 1));
    try std.testing.expectEqual(@as(usize, 32), rmsLanes(128, 2048));
}

test "the indexer's weight scale: 128 ** -0.5 * 32 ** -0.5 in fp32" {
    try std.testing.expectEqual(@as(u32, 0x3c800000), @as(u32, @bitCast(indexScale(128, 32))));
}
