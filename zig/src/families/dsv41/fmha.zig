//! The vision tower's attention as the served lane runs it: F.scaled_dot_product_attention on fp32 q, k, v [1, 16, n, 64]
//! picks PyTorch's memory-efficient kernel, fmha_cutlassF_f32_aligned_64x64_rf_sm80 (cutlass AttentionKernel<float,
//! Sm80, aligned, 64, 64, 64, single value iteration, supports dropout>). Its sm_120 SASS is the one GB10 runs; we launch
//! that same cubin (extracted from the served image's libtorch_cuda.so, BSD-3, see THIRD_PARTY_NOTICES) with the
//! Params torch fills (kernel_forward.h; the recorded launch's 272 bytes, field for field), so the output is torch's.
const std = @import("std");
const cuda = @import("cuda");

pub const kernel_name = "_Z39fmha_cutlassF_f32_aligned_64x64_rf_sm80N22PyTorchMemEffAttention15AttentionKernelIfN7cutlass4arch4Sm80ELb1ELi64ELi64ELi64ELb1ELb1EE6ParamsE";
/// The cubin's file in the kit (TF_DS_KIT/vision/).
pub const cubin_file = "torch_fmha_sm120.cubin";

/// AttentionKernel::Params, as the recording's launch lays it out (272 bytes).
pub const Params = extern struct {
    query: u64,
    key: u64,
    value: u64,
    attn_bias: u64 = 0,
    seqstart_q: u64 = 0,
    seqstart_k: u64 = 0,
    seqlen_k: u64 = 0,
    causal_diagonal_offset: u32 = 0,
    _pad0: u32 = 0,
    output: u64,
    output_accum: u64 = 0,
    logsumexp: u64 = 0,
    window_size: i32 = 0,
    scale: f32,
    head_dim: i32,
    head_dim_value: i32,
    num_queries: i32,
    num_keys: i32,
    num_keys_absolute: i32 = 0,
    custom_mask_type: u8 = 0,
    _pad1: [3]u8 = .{ 0, 0, 0 },
    q_stride_m: i32,
    k_stride_m: i32,
    v_stride_m: i32,
    bias_stride_m: i32 = 0,
    o_stride_m: i32,
    q_stride_h: i32,
    k_stride_h: i32,
    v_stride_h: i32,
    bias_stride_h: i64 = 0,
    q_stride_b: i64,
    k_stride_b: i64,
    v_stride_b: i64,
    bias_stride_b: i64 = 0,
    num_batches: i32,
    num_heads: i32,
    use_dropout: u8 = 0,
    _pad2: [7]u8 = @splat(0),
    dropout_batch_head_rng_offset: u64 = 0,
    dropout_prob: f32 = 0,
    _pad3: u32 = 0,
    philox_seed: u64 = 0,
    philox_offset: u64 = 0,
    philox_offset_intragraph: u32 = 0,
    philox_captured: u8 = 0,
    _pad4: [3]u8 = .{ 0, 0, 0 },
    _pad5: u64 = 0,
    extragraph_offset: u64 = 0,
    seed: u64 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Params) == 272);
    std.debug.assert(@offsetOf(Params, "output") == 64);
    std.debug.assert(@offsetOf(Params, "scale") == 92);
    std.debug.assert(@offsetOf(Params, "q_stride_m") == 120);
    std.debug.assert(@offsetOf(Params, "q_stride_b") == 160);
    std.debug.assert(@offsetOf(Params, "num_batches") == 192);
    std.debug.assert(@offsetOf(Params, "philox_seed") == 224);
}

pub const Fmha = struct {
    module: cuda.Module,
    f: cuda.Function,

    pub fn load(d: *const cuda.Driver, cubin: []const u8) !Fmha {
        var m = try cuda.Module.load(d, cubin);
        errdefer m.unload();
        return .{ .module = m, .f = try m.function(kernel_name) };
    }

    pub fn unload(x: *Fmha) void {
        x.module.unload();
    }

    /// Non-causal attention of one image: q, k, v fp32 [n, heads, 64] (row stride heads * 64), out fp32 [n, heads, 64];
    /// scale 1 / sqrt(64), torch's launch grid (ceil(n / 64), heads, 1), 128 threads, 36,352 bytes of shared memory.
    pub fn run(x: *const Fmha, s: cuda.Stream, q: u64, k: u64, v: u64, out: u64, n: usize, heads: usize) !void {
        const hd: i32 = 64;
        const row: i32 = @intCast(heads * 64);
        const p: Params = .{
            .query = q,
            .key = k,
            .value = v,
            .output = out,
            .scale = 0.125,
            .head_dim = hd,
            .head_dim_value = hd,
            .num_queries = @intCast(n),
            .num_keys = @intCast(n),
            .q_stride_m = row,
            .k_stride_m = row,
            .v_stride_m = row,
            .o_stride_m = row,
            .q_stride_h = hd,
            .k_stride_h = hd,
            .v_stride_h = hd,
            .q_stride_b = row,
            .k_stride_b = row,
            .v_stride_b = row,
            .num_batches = 1,
            .num_heads = @intCast(heads),
        };
        var a: cuda.Args = .{};
        a.add(p);
        try cuda.launch.launch(x.f, .{ .grid = .{ .x = @intCast((n + 63) / 64), .y = @intCast(heads) }, .block = .{ .x = 32, .y = 4 }, .shared = 36352 }, s, &a);
    }
};
