//! EXL3 linears of 1 to 128 rows as the served build runs them (exl3/linear.py's Exl3Group, linear_grouped.cu): one
//! rot_many launch rotating every layer's input rows, one glinear launch for each (bits, codebook, warps) set, from
//! the served build's own cubin. Each output has its layer's own plan, K ranges and reduction order.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const elf = @import("elf.zig");

/// The layers one launch takes (linear_grouped.cu GMAX).
pub const gmax = 8;
/// mul1, the codebook of every tensor of our checkpoint (CODEBOOK_IDS).
pub const codebook_mul1 = 2;

pub const DType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

/// (K splits, warps a program) for a K x N layer (linear.py plan): the shape's alone, so a layer keeps one reduction
/// for every row count.
/// The N a layer's plan is made for: its own, or the TP2 slice's a 2D column part keeps (weights.Linear.plan_n).
fn planN(l: weights.Linear) usize {
    return if (l.plan_n != 0) l.plan_n else l.n;
}

pub fn plan(k: usize, n: usize) struct { sk: u32, wk: u32 } {
    const blocks = 192;
    const min_tiles = 8;
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ .sk = @intCast(sk), .wk = @intCast(wk) };
}

/// (words between k tiles, words between 128-column blocks) of a layer's words in the strips layout.
pub fn strides(l: weights.Linear) [2]i64 {
    const tw: i64 = 4 * @as(i64, l.k2);
    return .{ 8 * tw, @divExact(@as(i64, l.k), 16) * 8 * tw };
}

/// linear_grouped.cu's GLayer, field for field (C layout).
pub const GLayer = extern struct {
    xh: u64 = 0,
    T: u64 = 0,
    svh: u64 = 0,
    bias: u64 = 0,
    y: u64 = 0,
    Z: u64 = 0,
    counters: u64 = 0,
    ldx: i64 = 0,
    ldy: i64 = 0,
    stride_k: i64 = 0,
    stride_nb: i64 = 0,
    y_dtype: c_int = 0,
    K: c_int = 0,
    N: c_int = 0,
    SK: c_int = 0,
    first: c_int = 0,
    rsuh: u64 = 0,
    rxh: u64 = 0,
    ldr: i64 = 0,
    roff: c_int = 0,
    rcos: u64 = 0,
    rsin: u64 = 0,
    rpos: u64 = 0,
    rhd: c_int = 0,
    rrd: c_int = 0,
};

pub const GArgs = extern struct { l: [gmax]GLayer = @splat(.{}), n: c_int = 0, M: c_int = 0, discard: c_int = 0 };

pub const RLayer = extern struct { x: u64 = 0, suh: u64 = 0, xh: u64 = 0, ldx: i64 = 0, x_dtype: c_int = 0, K: c_int = 0, first: c_int = 0 };

pub const RArgs = extern struct { l: [gmax]RLayer = @splat(.{}), n: c_int = 0, M: c_int = 0 };

comptime {
    // the sizes the kernels were compiled with (pointers and long longs 8 bytes, ints 4, C padding)
    std.debug.assert(@sizeOf(GLayer) == 176 and @sizeOf(GArgs) == 1424);
    std.debug.assert(@sizeOf(RLayer) == 48 and @sizeOf(RArgs) == 392);
}

/// The cubin's kernels: glinear_kernel<K2, codebook, warps> by name, and rot_many_kernel.
pub const Kernels = struct {
    module: cuda.Module,
    rot_many: cuda.Function,
    glinear: [17][3][2]?cuda.Function = @splat(@splat(@splat(null))),

    /// The served build's linear_grouped cubin (cuobjdump of tensorfold_exl3_linear_v7's .so).
    pub fn load(d: *const cuda.Driver, cubin: []const u8) !Kernels {
        var k: Kernels = .{ .module = try cuda.Module.load(d, cubin), .rot_many = undefined };
        errdefer k.module.unload();
        var names = elf.Symbols.init(cubin) orelse return error.BadCubin;
        var found_rot = false;
        while (names.next()) |name| {
            var buf: [256]u8 = undefined;
            if (std.mem.indexOf(u8, name, "rot_many_kernel") != null) {
                k.rot_many = try k.module.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
                found_rot = true;
                continue;
            }
            const at = std.mem.indexOf(u8, name, "glinear_kernelILi") orelse continue;
            const t = template3(name[at + "glinear_kernel".len ..]) orelse continue;
            if (t[0] > 16 or t[1] > 2 or (t[2] != 4 and t[2] != 8)) continue;
            k.glinear[t[0]][t[1]][if (t[2] == 8) 1 else 0] = try k.module.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
        }
        if (!found_rot) return error.MissingKernel;
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }
};

/// "ILi7ELi2ELi8EE..." -> {7, 2, 8}
fn template3(s: []const u8) ?[3]u32 {
    var out: [3]u32 = undefined;
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

/// One layer's call: its rotated input rows, its output rows and dtype, its split-K counters (int32 [8 N / 128], left
/// zero by the kernel) and, for SK > 1, its partials' place in the launch's Z.
pub const Call = struct {
    layer: weights.Linear,
    x: u64,
    ldx: i64,
    x_dtype: DType,
    xh: u64,
    y: u64,
    ldy: i64,
    y_dtype: DType,
    counters: u64,
};

/// rot_many: every layer's input rows rotated into its xh rows [M, K] fp16, one launch.
pub fn rotMany(k: *const Kernels, stream: cuda.Stream, calls: []const Call, m: usize, pdl: bool) !void {
    var a: RArgs = .{ .n = @intCast(calls.len), .M = @intCast(m) };
    var warps: usize = 0;
    for (calls, 0..) |c, i| {
        a.l[i] = .{ .x = c.x, .suh = c.layer.suh, .xh = c.xh, .ldx = c.ldx, .x_dtype = @intFromEnum(c.x_dtype), .K = @intCast(c.layer.k), .first = @intCast(warps) };
        warps += m * (c.layer.k / 128);
    }
    var args: cuda.Args = .{};
    args.add(a);
    try cuda.launch.launch(k.rot_many, .{ .grid = .{ .x = @intCast((warps + 3) / 4) }, .block = .{ .x = 128 }, .pdl = pdl }, stream, &args);
}

/// The glinear launches of a group (Exl3Group.rotated, no folds): each (bits, codebook, warps) set in first-seen
/// order, its split layers' partials in z (fp32, SK * M * N floats each, in order).
pub fn glinear(k: *const Kernels, stream: cuda.Stream, calls: []const Call, m: usize, z: u64, pdl: bool, discard: bool) !void {
    var done: [gmax]bool = @splat(false);
    for (calls, 0..) |first, i| {
        if (done[i]) continue;
        const pf = plan(first.layer.k, planN(first.layer));
        var a: GArgs = .{ .M = @intCast(m) };
        var blocks: usize = 0;
        var zoff: u64 = 0;
        var any_z = false;
        for (calls[i..], i..) |c, j| {
            const pj = plan(c.layer.k, planN(c.layer));
            if (done[j] or c.layer.k2 != first.layer.k2 or pj.wk != pf.wk) continue;
            done[j] = true;
            const s = strides(c.layer);
            const n: usize = @intCast(a.n);
            a.l[n] = .{ .xh = c.xh, .ldx = @intCast(c.layer.k), .T = c.layer.words, .stride_k = s[0], .stride_nb = s[1], .svh = c.layer.svh, .y = c.y, .y_dtype = @intFromEnum(c.y_dtype), .ldy = c.ldy, .K = @intCast(c.layer.k), .N = @intCast(c.layer.n), .SK = @intCast(pj.sk), .counters = c.counters, .first = @intCast(blocks) };
            if (pj.sk > 1) {
                a.l[n].Z = z + zoff * 4;
                zoff += @as(u64, pj.sk) * m * c.layer.n;
                any_z = true;
            }
            blocks += (c.layer.n / 128) * pj.sk;
            a.n += 1;
        }
        a.discard = @intFromBool(discard and any_z);
        const f = k.glinear[first.layer.k2][codebook_mul1][if (pf.wk == 8) 1 else 0] orelse return error.MissingKernel;
        var args: cuda.Args = .{};
        args.add(a);
        const smem: u32 = @intCast(pf.wk * @min(m, 8) * 128 * 4);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = pf.wk * 32 }, .shared = smem, .pdl = pdl }, stream, &args);
    }
}

/// Floats of Z a group needs at m rows (every split layer's SK * M * N).
pub fn zFloats(layers: []const weights.Linear, m: usize) usize {
    var n: usize = 0;
    for (layers) |l| {
        const p = plan(l.k, planN(l));
        if (p.sk > 1) n += p.sk * m * l.n;
    }
    return n;
}

test "plans as linear.py makes them" {
    // narrow layers take K splits to fill the SMs; wide ones 8 warps; the K tile count bounds both
    try std.testing.expectEqual(@as(u32, 8), plan(1280, 32768).wk);
    try std.testing.expectEqual(@as(u32, 1), plan(1280, 32768).sk);
    const q = plan(5120, 1280);
    try std.testing.expectEqual(@as(u32, 4), q.wk);
    try std.testing.expect(q.sk > 1 and (5120 / 16) % (q.sk * q.wk) == 0);
    try std.testing.expectEqualDeep([3]u32{ 7, 2, 8 }, template3("ILi7ELi2ELi8EEEvNS_5GArgsE").?);
}

test "plans and strides equal every recorded glinear layer of the served build" {
    // (K, N, SK, WK, K2, stride_k, stride_nb) of each distinct layer the recording's glinear calls passed
    const seen = [_][7]u32{
        .{ 512, 128, 1, 4, 16, 512, 16384 },       .{ 1280, 4096, 2, 4, 10, 320, 25600 },     .{ 1280, 16384, 1, 8, 8, 256, 20480 },
        .{ 1280, 16384, 1, 8, 10, 320, 25600 },    .{ 3072, 25600, 1, 8, 8, 256, 49152 },     .{ 3072, 25600, 1, 8, 10, 320, 61440 },
        .{ 4096, 1024, 8, 4, 8, 256, 65536 },      .{ 4096, 1024, 8, 4, 10, 320, 81920 },     .{ 4096, 5120, 8, 4, 8, 256, 65536 },
        .{ 4096, 5120, 8, 4, 10, 320, 81920 },     .{ 5120, 512, 8, 4, 8, 256, 81920 },       .{ 5120, 512, 8, 4, 10, 320, 102400 },
        .{ 5120, 512, 8, 4, 12, 384, 122880 },     .{ 5120, 1280, 8, 4, 8, 256, 81920 },      .{ 5120, 1280, 8, 4, 10, 320, 102400 },
        .{ 5120, 1280, 8, 4, 12, 384, 122880 },    .{ 5120, 64640, 1, 8, 12, 384, 122880 },   .{ 15360, 5120, 8, 4, 8, 256, 245760 },
    };
    for (seen) |r| {
        const p = plan(r[0], r[1]);
        try std.testing.expectEqual(r[2], p.sk);
        try std.testing.expectEqual(r[3], p.wk);
        const s = strides(.{ .words = 0, .suh = 0, .svh = 0, .k = r[0], .n = r[1], .k2 = r[4] });
        try std.testing.expectEqual(@as(i64, r[5]), s[0]);
        try std.testing.expectEqual(@as(i64, r[6]), s[1]);
    }
}

test "a 2D node's column part keeps its TP2 slice's plan" {
    // wq_b of a pair: 8,192 of the rank's 16,384 columns, planned as the 16,384 the TP2 rank launches
    const part: weights.Linear = .{ .words = 0, .suh = 0, .svh = 0, .k = 1024, .n = 8192, .k2 = 2, .plan_n = 16384 };
    try std.testing.expectEqual(plan(1024, 16384), plan(part.k, planN(part)));
    const whole: weights.Linear = .{ .words = 0, .suh = 0, .svh = 0, .k = 1024, .n = 16384, .k2 = 2 };
    try std.testing.expectEqual(plan(1024, 16384), plan(whole.k, planN(whole)));
}
