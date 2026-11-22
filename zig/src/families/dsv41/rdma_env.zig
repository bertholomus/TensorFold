//! The served RDMA ring's GID index from the environment (TF_RDMA_GID_INDEX), as on the deepseek-v41-zig line.
const std = @import("std");

/// The served ring has its own GID selection, independent of NCCL's.
pub fn parseGidIndex(text: ?[]const u8) error{InvalidRdmaGidIndex}!u8 {
    const value = text orelse return 5;
    if (value.len == 0) return error.InvalidRdmaGidIndex;
    for (value) |c| if (c < '0' or c > '9') return error.InvalidRdmaGidIndex;
    return std.fmt.parseInt(u8, value, 10) catch error.InvalidRdmaGidIndex;
}

pub fn servedGidIndex() error{InvalidRdmaGidIndex}!u8 {
    const text = if (std.c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null;
    return parseGidIndex(text) catch |err| {
        std.debug.print("TF_RDMA_GID_INDEX: expected a decimal GID index in 0..255\n", .{});
        return err;
    };
}

test "served GID defaults to 5 and accepts an explicit index, refusing malformed settings" {
    try std.testing.expectEqual(@as(u8, 5), try parseGidIndex(null));
    try std.testing.expectEqual(@as(u8, 3), try parseGidIndex("3"));
    try std.testing.expectEqual(@as(u8, 0), try parseGidIndex("0"));
    try std.testing.expectEqual(@as(u8, 255), try parseGidIndex("255"));
    for ([_][]const u8{ "", "-1", "+3", " 3", "3 ", "0x3", "3_0", "256", "invalid" }) |value|
        try std.testing.expectError(error.InvalidRdmaGidIndex, parseGidIndex(value));
}
