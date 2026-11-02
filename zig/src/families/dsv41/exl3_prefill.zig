//! The prompt path's EXL3 linear (exl3/prefill.py matmul, more than 128 rows): the input rows rotated into fp16
//! (rot_in: fp16(((x * suh) H) / sqrt(128))), the weight's trellis decoded into fp16 W_q [K, N] (unpack), then the
//! Triton GEMM (tri_markov.matmul) with the 128 x 128 Hadamard and svh. rot_in and unpack are the served build's
//! (linear.cu's cubin of tensorfold_exl3_linear_v7), launched as exl3_rot_in_cuda and exl3_unpack_cuda launch them.
const std = @import("std");
const cuda = @import("cuda");
const elf = @import("elf.zig");
const exl3_linear = @import("exl3_linear.zig");
const tri = @import("tri.zig");
const tri_markov = @import("tri_markov.zig");
const weights = @import("weights.zig");

pub const DType = exl3_linear.DType;

pub const Kernels = struct {
    module: cuda.Module,
    rot_in: cuda.Function,
    unpack: [17][3]?cuda.Function = @splat(@splat(null)),

    /// linear.cu's cubin (the one with unpack_kernel): kernels found by name, since the anonymous namespace's hash
    /// changes with every build.
    pub fn load(d: *const cuda.Driver, cubin: []const u8) !Kernels {
        var k: Kernels = .{ .module = try cuda.Module.load(d, cubin), .rot_in = undefined };
        errdefer k.module.unload();
        var names = elf.Symbols.init(cubin) orelse return error.BadCubin;
        var found = false;
        while (names.next()) |name| {
            var buf: [256]u8 = undefined;
            if (std.mem.indexOf(u8, name, "rot_in_kernel") != null) {
                k.rot_in = try k.module.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
                found = true;
                continue;
            }
            const at = std.mem.indexOf(u8, name, "unpack_kernelILi") orelse continue;
            const t = template2(name[at + "unpack_kernel".len ..]) orelse continue;
            if (t[0] > 16 or t[1] > 2) continue;
            k.unpack[t[0]][t[1]] = try k.module.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
        }
        if (!found) return error.MissingKernel;
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }
};

/// "ILi7ELi2EEEv..." -> {7, 2}
fn template2(s: []const u8) ?[2]u32 {
    var out: [2]u32 = undefined;
    var rest = s;
    for (&out) |*v| {
        const i = std.mem.indexOf(u8, rest, "Li") orelse return null;
        rest = rest[i + 2 ..];
        const e = std.mem.indexOfScalar(u8, rest, 'E') orelse return null;
        v.* = std.fmt.parseInt(u32, rest[0..e], 10) catch return null;
        rest = rest[e + 1 ..];
    }
    return out;
}

/// rot_in: xh [M, K] fp16 (contiguous) from x's M rows (row stride ldx elements, rows 16-byte aligned), K a multiple
/// of 128: a warp a 128-column block, four blocks a program, a row a grid row.
pub fn rotIn(k: *const Kernels, stream: cuda.Stream, x: u64, x_dtype: DType, ldx: usize, suh: u64, xh: u64, m: usize, kdim: usize) !void {
    if (kdim % 128 != 0 or x % 16 != 0 or (ldx * dtypeBytes(x_dtype)) % 16 != 0) return error.Unaligned;
    var args: cuda.Args = .{};
    args.add(x);
    args.add(@as(c_int, @backingInt(x_dtype)));
    args.add(@as(c_longlong, @intCast(ldx)));
    args.add(suh);
    args.add(xh);
    args.add(@as(c_int, @intCast(kdim)));
    try cuda.launch.launch(k.rot_in, .{ .grid = .{ .x = @intCast((kdim / 128 + 3) / 4), .y = @intCast(m) }, .block = .{ .x = 128 } }, stream, &args);
}

/// unpack: W_q [K, N] fp16 from the layer's trellis words, a warp a 16 x 16 tile.
pub fn unpack(k: *const Kernels, stream: cuda.Stream, l: weights.Linear, w: u64) !void {
    const f = k.unpack[l.k2][exl3_linear.codebook_mul1] orelse return error.MissingKernel;
    const s = exl3_linear.strides(l);
    var args: cuda.Args = .{};
    args.add(l.words);
    args.add(w);
    args.add(@as(c_int, @intCast(l.n)));
    args.add(@as(i64, @intCast(s[0])));
    args.add(@as(i64, @intCast(s[1])));
    try cuda.launch.launch(f, .{ .grid = .{ .x = l.n / 16, .y = l.k / 16 }, .block = .{ .x = 32 } }, stream, &args);
}

fn dtypeBytes(t: DType) usize {
    return switch (t) {
        .f32 => 4,
        else => 2,
    };
}

/// prefill.Workspace: one rotated input xh (fp16, at least M * K) and one W_q (fp16, at least K * N), each the
/// largest call's, reused by every call on the stream, and the bf16 Hadamard (hadamard()).
pub const Workspace = struct { xh: u64, w: u64, h: u64 };

/// prefill.matmul without a bias (no DeepSeek-V4.1 linear has one): out [M, N] (row stride os elements) =
/// x [M, K] (row stride ldx) @ W, the prompt path's arithmetic.
pub fn matmul(k: *const Kernels, t: tri.Tri, ws: Workspace, l: weights.Linear, x: u64, x_dtype: DType, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize, m: usize) !void {
    try rotIn(k, t.stream, x, x_dtype, ldx, l.suh, ws.xh, m, l.k);
    try unpack(k, t.stream, l, ws.w);
    try tri_markov.matmul(t, ws.xh, ws.w, ws.h, l.svh, null, out, out_type, os, m, l.k, l.n);
}

/// prefill.Workspace.hadamard: H [128, 128] bf16, 1 - 2 * parity(i & j) (the bits of +-1.0).
pub fn hadamard(out: *[128 * 128]u16) void {
    for (0..128) |i| {
        for (0..128) |j| out[i * 128 + j] = if (@popCount(i & j) & 1 == 1) 0xbf80 else 0x3f80;
    }
}

test "the prompt GEMM's Hadamard: +-1, symmetric, orthogonal rows" {
    var h: [128 * 128]u16 = undefined;
    hadamard(&h);
    for (0..128) |j| try std.testing.expectEqual(@as(u16, 0x3f80), h[j]);
    for (0..128) |i| {
        for (0..128) |j| {
            try std.testing.expectEqual(h[i * 128 + j], h[j * 128 + i]);
            var dot: i32 = 0;
            for (0..128) |c| dot += @as(i32, if (h[i * 128 + c] == 0x3f80) 1 else -1) * @as(i32, if (h[j * 128 + c] == 0x3f80) 1 else -1);
            try std.testing.expectEqual(@as(i32, if (i == j) 128 else 0), dot);
        }
    }
}

test "unpack kernels by template arguments" {
    try std.testing.expectEqualDeep([2]u32{ 7, 2 }, template2("ILi7ELi2EEEvPKjP6__halfill").?);
    try std.testing.expectEqualDeep([2]u32{ 16, 2 }, template2("ILi16ELi2EEEvPKjP6__halfill").?);
    try std.testing.expectEqual(@as(?[2]u32, null), template2("v"));
}
