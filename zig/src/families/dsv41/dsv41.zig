//! DeepSeek-V4.1-Flash on CUDA in Zig, tensor parallel over two GB10s: the served Python family's kernels, host first.

pub const Config = @import("config.zig").Config;
pub const native = @import("native.zig");
pub const rank_cache = @import("rank_cache.zig");
pub const plan = @import("plan.zig");

test {
    _ = @import("config.zig");
    _ = native;
    _ = rank_cache;
    _ = plan;
}
