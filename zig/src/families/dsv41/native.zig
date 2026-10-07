//! DeepSeek-V4.1-Flash behind the native CUDA family interface. Port milestone M0: the checkpoint is read and refused.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Config = @import("config.zig").Config;

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3-mul1"};
/// The Python lane's default window; the server's --context moves it up to the model's 1,048,576.
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
/// The served build's prompt chunk (TF_DS_PREFILL_CHUNK).
pub const prompt_rows: u32 = 2048;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence.
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
};

/// Reads and checks config.json, then refuses: the engine is not ported yet.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    _ = .{ ctx, kernels, o };
    _ = try Config.read(gpa, io, dir);
    return error.NotPortedYet;
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NotPortedYet => "the native DeepSeek-V4.1 engine is still being ported; serve this checkpoint with the Python 0.6 build",
        else => null,
    };
}

test "the refusal names the Python build" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NotPortedYet).?, "Python 0.6") != null);
    try std.testing.expectEqual(@as(?[]const u8, null), explain(null, error.OutOfMemory));
}
