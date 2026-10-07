//! DeepSeek-V4.1-Flash on CUDA in Zig, tensor parallel over two GB10s: the served Python family's kernels, host first.

pub const Config = @import("config.zig").Config;
pub const native = @import("native.zig");
pub const rank_cache = @import("rank_cache.zig");
pub const plan = @import("plan.zig");
pub const engram = @import("engram.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const weights = @import("weights.zig");
pub const engram_io = @import("engram_io.zig");
pub const link = @import("link.zig");
pub const comm = @import("comm.zig");
pub const tri = @import("tri.zig");
pub const tri_basic = @import("tri_basic.zig");
pub const cublas = @import("cublas.zig");
pub const elf = @import("elf.zig");
pub const exl3_linear = @import("exl3_linear.zig");
pub const tri_norm = @import("tri_norm.zig");
pub const tri_index = @import("tri_index.zig");
pub const tri_attn = @import("tri_attn.zig");
pub const tri_markov = @import("tri_markov.zig");
pub const tri_hc = @import("tri_hc.zig");
pub const exl3_prefill = @import("exl3_prefill.zig");
pub const ops = @import("ops.zig");
pub const exl3_experts = @import("exl3_experts.zig");
pub const prompt = @import("prompt.zig");

test {
    // every declaration of every file, so functions no test calls are still compiled
    const refAll = @import("std").testing.refAllDecls;
    refAll(@import("config.zig"));
    refAll(native);
    refAll(rank_cache);
    refAll(plan);
    refAll(engram);
    refAll(checkpoint);
    refAll(weights);
    refAll(engram_io);
    refAll(link);
    refAll(comm);
    refAll(tri);
    refAll(tri_basic);
    refAll(cublas);
    refAll(elf);
    refAll(exl3_linear);
    refAll(tri_norm);
    refAll(tri_index);
    refAll(tri_attn);
    refAll(tri_markov);
    refAll(tri_hc);
    refAll(exl3_prefill);
    refAll(ops);
    refAll(exl3_experts);
    refAll(prompt);
}
