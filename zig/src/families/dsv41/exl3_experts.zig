//! The routed experts of a prompt chunk (exl3/experts.py routed(), the fast path the served build takes at 64 rows and
//! more, with its switches at their served values: the "mma" prompt kernel, work lists, no shared-dense GEMM): eight
//! launches of the served build's kernels (tensorfold_exl3_experts_v19: experts.cu's cubin and the codebook's
//! experts_cb*.cu cubin), as experts.cpp / experts.cu / experts_grouped.cuh launch them:
//! group_count, group_place, work_list, rot_in, grouped_mma (gate and up, the K splits folded), gateup_epilogue,
//! grouped_rows (down) and down_combine. The map: pipeline/EXL3-EXPERTS-LAUNCHES.md.
const std = @import("std");
const cuda = @import("cuda");
const elf = @import("elf.zig");
const weights = @import("weights.zig");

/// experts.py's settings (GLM_GATEUP, GLM_DOWN by default_config; PROMPT_TILES; MMA_KB; EXACT_ROWS).
pub const exact_rows = 64;
pub const place_threads = 256; // experts.cu PLACE_THREADS
pub const list_threads = 1024; // experts.cu LIST_THREADS
pub const mma_cols = 64; // experts_grouped.cuh MMA_COLS
pub const mma_rows_gu = 64; // work list rows of the mma kernel
pub const rows_rows_d = 32; // work list rows of grouped_rows: 16 * PROMPT_TILES["down"][2]
pub const act_f32 = 1; // ACT_F32

/// A K2 range's template: (8, 8), (2, 10) or (2, 16), as every grouped launcher picks it.
pub fn range(lo: u32, hi: u32) usize {
    if (lo == 8 and hi == 8) return 0;
    if (lo >= 2 and hi <= 10) return 1;
    return 2;
}

/// default_config(K, N): (n tiles, warps, K splits, tiles in flight), GLM's where it divides.
pub fn config(k: usize, n: usize, gateup: bool) ![4]usize {
    const cands = [_][4]usize{ if (gateup) .{ 8, 4, 4, 1 } else .{ 8, 4, 1, 1 }, .{ 8, 4, 2, 1 }, .{ 8, 4, 1, 1 }, .{ 4, 4, 2, 2 }, .{ 4, 4, 1, 2 } };
    for (cands) |c| if (k % (16 * c[2] * c[1]) == 0 and n % (16 * c[0]) == 0) return c;
    return error.NoTileSetting;
}

/// _list_len: a work list's pairs for P member slots over at most maxu places in groups of `rows`.
pub fn listLen(p: usize, maxu: usize, rows: usize) usize {
    return (p + maxu * (rows - 1)) / rows;
}

/// The member lists' stride at R rows with work lists (busiest = R): whole 32-row tiles, at least one.
pub fn memberStride(r: usize) usize {
    const tile = 32;
    return @max(tile, (r + tile - 1) / tile * tile);
}

pub const Kernels = struct {
    base: cuda.Module,
    cb: cuda.Module,
    group_count: cuda.Function,
    group_place: cuda.Function,
    work_list: cuda.Function,
    rot_in: cuda.Function, // <__nv_bfloat16>
    gateup_epilogue: cuda.Function,
    down_combine: cuda.Function,
    mma: [3]?cuda.Function = @splat(null), // grouped_mma_kernel<CB, LO, HI, FOLD true, KB 4> by range()
    rows: [3]?cuda.Function = @splat(null), // grouped_rows_kernel<CB, 8, 4, 1, LO, HI, 2, FOLD false> by range()

    /// experts.cu's cubin and the codebook's (experts_cb<codebook>.cu); kernels by name (the anonymous namespace's
    /// hash changes with every build) and template arguments.
    pub fn load(d: *const cuda.Driver, base_cubin: []const u8, cb_cubin: []const u8, codebook: u32) !Kernels {
        var base = try cuda.Module.load(d, base_cubin);
        errdefer base.unload();
        var cb = try cuda.Module.load(d, cb_cubin);
        errdefer cb.unload();
        var k: Kernels = .{ .base = base, .cb = cb, .group_count = undefined, .group_place = undefined, .work_list = undefined, .rot_in = undefined, .gateup_epilogue = undefined, .down_combine = undefined };
        var found: u6 = 0;
        var names = elf.Symbols.init(base_cubin) orelse return error.BadCubin;
        while (names.next()) |name| {
            const plain = [_]struct { []const u8, u6 }{ .{ "18group_count_kernelE", 1 }, .{ "18group_place_kernelE", 2 }, .{ "16work_list_kernelE", 4 }, .{ "13rot_in_kernelI13__nv_bfloat16E", 8 }, .{ "22gateup_epilogue_kernelE", 16 }, .{ "19down_combine_kernelE", 32 } };
            for (plain) |e| {
                if (std.mem.indexOf(u8, name, e[0]) == null) continue;
                const f = try fnByName(base, name);
                switch (e[1]) {
                    1 => k.group_count = f,
                    2 => k.group_place = f,
                    4 => k.work_list = f,
                    8 => k.rot_in = f,
                    16 => k.gateup_epilogue = f,
                    else => k.down_combine = f,
                }
                found |= e[1];
            }
        }
        if (found != 63) return error.MissingKernel;
        names = elf.Symbols.init(cb_cubin) orelse return error.BadCubin;
        while (names.next()) |name| {
            if (std.mem.indexOf(u8, name, "18grouped_mma_kernelI")) |at| {
                var t: [5]i64 = undefined;
                if (!templateArgs(name[at + "18grouped_mma_kernel".len ..], &t)) continue;
                // <CB, LO, HI, FOLD, KB>
                if (t[0] != codebook or t[3] != 1 or t[4] != 4) continue;
                k.mma[range(@intCast(t[1]), @intCast(t[2]))] = try fnByName(cb, name);
            } else if (std.mem.indexOf(u8, name, "19grouped_rows_kernelI")) |at| {
                var t: [8]i64 = undefined;
                if (!templateArgs(name[at + "19grouped_rows_kernel".len ..], &t)) continue;
                // <CB, NT, W, PF, LO, HI, G, FOLD>
                if (t[0] != codebook or t[1] != 8 or t[2] != 4 or t[3] != 1 or t[6] != 2 or t[7] != 0) continue;
                k.rows[range(@intCast(t[4]), @intCast(t[5]))] = try fnByName(cb, name);
            }
        }
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.base.unload();
        k.cb.unload();
        k.* = undefined;
    }
};

fn fnByName(m: cuda.Module, name: []const u8) !cuda.Function {
    var buf: [512]u8 = undefined;
    return m.function(try std.fmt.bufPrintZ(&buf, "{s}", .{name}));
}

/// The integer and bool template arguments of a mangled kernel name's "I...E" list ("ILi2ELi8ELb1ELi4EE" -> 2, 8, 1,
/// 4); false unless there are exactly out.len of them.
pub fn templateArgs(s: []const u8, out: []i64) bool {
    if (s.len == 0 or s[0] != 'I') return false;
    var i: usize = 1;
    var n: usize = 0;
    while (i < s.len and s[i] == 'L') {
        if (i + 2 >= s.len) return false;
        const kind = s[i + 1];
        const e = std.mem.indexOfScalarPos(u8, s, i + 2, 'E') orelse return false;
        if (n == out.len) return false;
        out[n] = switch (kind) {
            'i', 'b' => std.fmt.parseInt(i64, s[i + 2 .. e], 10) catch return false,
            else => return false,
        };
        n += 1;
        i = e + 1;
    }
    return n == out.len and i < s.len and s[i] == 'E';
}

/// A prompt Scratch (experts.py Scratch(prompt=True)) for up to `rows` rows of `slots` slots, and a chunk's work lists:
/// device addresses the caller allocated with these sizes and initial contents.
pub const Scratch = struct {
    rows: usize,
    slots: usize,
    xg: u64, // fp16 [rows * slots, D], zeros
    xu: u64, // fp16 [rows * slots, D], zeros
    xd: u64, // fp16 [rows * slots, I], zeros
    z: u64, // fp32 [promptZ(D, I) * rows * slots], zeros
    no_y: u64, // fp32 [1], zeros
    ids: u64, // int32 [min(rows * slots, E)], zeros
    count: u64, // int32 [1], zeros
    counts: u64, // int32 [E], zeros
    members: u64, // int32 [min(rows * slots, E) * ceil(rows / 128) * 128], -1
    work_gu: u64, // int32 [listLen(rows * slots, maxu, 64) * 2]
    work_d: u64, // int32 [listLen(rows * slots, maxu, 32) * 2]

    pub fn promptZ(d: usize, i: usize) usize {
        return @max(2 * i + d / 2, d);
    }

    /// Bytes of each buffer, in the field order above (xg .. work_d).
    pub fn sizes(rows: usize, slots: usize, d: usize, i: usize, e: usize) [12]usize {
        const p = rows * slots;
        const maxu = @min(p, e);
        return .{ 2 * p * d, 2 * p * d, 2 * p * i, 4 * promptZ(d, i) * p, 4, 4 * maxu, 4, 4 * e, 4 * maxu * ((rows + 127) / 128 * 128), 8 * listLen(p, maxu, mma_rows_gu), 8 * listLen(p, maxu, rows_rows_d) };
    }
};

fn launch(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
}

fn i32a(a: *cuda.Args, v: usize) void {
    a.add(@as(c_int, @intCast(v)));
}

/// routed(x, pick, wts, ex, s, out, R, limit, ACT_F32) on a prompt chunk (R >= 64): out fp32 [R, D] = each row's slots'
/// expert outputs weighted by wts, summed in slot order. x bf16 [R, D] (row stride x_stride elements), pick int32 and
/// wts fp32 [R, slots] contiguous; every pick an expert id (the shared expert is the last).
pub fn prompt(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: Scratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) !void {
    const d: usize = ex.dims;
    const i: usize = ex.width;
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r < exact_rows or r > sc.rows) return error.NotAPromptChunk;
    if (slots > 32 or i % 128 != 0 or d % 128 != 0) return error.UnsupportedShape;
    const cgu = try config(d, i, true);
    const cd = try config(i, d, false);
    // the served fast path: mma for gate / up (K chains a multiple of MMA_KB), grouped_rows for down
    const chain_gu = (d / 16) / (cgu[1] * cgu[2]);
    if ((d / 16) % (cgu[1] * cgu[2]) != 0 or chain_gu % 4 != 0) return error.UnsupportedShape;
    const chain_d = (i / 16) / (cd[1] * cd[2]);
    if ((i / 16) % (cd[1] * cd[2]) == 0 and chain_d % 4 == 0) return error.UnsupportedShape; // down would take mma
    if (cd[2] != 1) return error.UnsupportedShape;
    const p = r * slots;
    const maxu = @min(p, e);
    const maxm = memberStride(r);
    const n_gu = listLen(p, maxu, mma_rows_gu);
    const n_d = listLen(p, maxu, rows_rows_d);
    const mma = k.mma[range(ex.k2_gu[0], ex.k2_gu[1])] orelse return error.MissingKernel;
    const down = k.rows[range(ex.k2_d[0], ex.k2_d[1])] orelse return error.MissingKernel;

    var a: cuda.Args = .{};
    a.add(pick);
    a.add(sc.counts);
    i32a(&a, p);
    try launch(k.group_count, s, .{ e, 1, 1 }, place_threads, &a);

    a = .{};
    for ([_]u64{ pick, sc.counts, sc.ids, sc.count, sc.members }) |v| a.add(v);
    for ([_]usize{ p, slots, e, maxm }) |v| i32a(&a, v);
    try launch(k.group_place, s, .{ e, 1, 1 }, place_threads, &a);

    a = .{};
    for ([_]u64{ sc.counts, sc.ids, sc.count }) |v| a.add(v);
    i32a(&a, maxu);
    a.add(sc.work_gu);
    i32a(&a, mma_rows_gu);
    i32a(&a, n_gu);
    a.add(sc.work_d);
    i32a(&a, rows_rows_d);
    i32a(&a, n_d);
    a.add(@as(c_int, -1)); // skip_e: no shared-dense GEMM
    try launch(k.work_list, s, .{ 1, 1, 1 }, list_threads, &a);

    a = .{};
    a.add(x);
    i32a(&a, x_stride);
    for ([_]u64{ pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu }) |v| a.add(v);
    for ([_]usize{ d, slots, e }) |v| i32a(&a, v);
    try launch(k.rot_in, s, .{ p, d / 128, 2 }, 32, &a);

    // gate and up: grid (1, N / 64, pairs x 2 matrices), the K splits summed in the program (FOLD)
    a = .{};
    for ([_]u64{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ d, i, p, cgu[2], cgu[1], maxm, slots, n_gu }) |v| i32a(&a, v);
    a.add(sc.work_gu);
    if (n_gu > 0) try launch(mma, s, .{ 1, i / mma_cols, n_gu * 2 }, 256, &a);

    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |v| a.add(v);
    for ([_]usize{ p, i, 1, e }) |v| i32a(&a, v);
    a.add(limit);
    a.add(@as(c_int, act_f32));
    try launch(k.gateup_epilogue, s, .{ p, i / 128, 1 }, 32, &a);

    // down: grid (1, D / 128, pairs), 4 warps, two 16-row member tiles a program, one split
    a = .{};
    for ([_]u64{ sc.xd, sc.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z }) |v| a.add(v);
    for ([_]usize{ i, d, p, 1, maxm, slots, n_d }) |v| i32a(&a, v);
    a.add(sc.work_d);
    if (n_d > 0) try launch(down, s, .{ 1, d / 128, n_d }, 4 * 32, &a);

    a = .{};
    for ([_]u64{ sc.z, pick, ex.svh_d, sc.no_y, wts, 0, out }) |v| a.add(v);
    for ([_]usize{ p, d, 1, e, slots, 0 }) |v| i32a(&a, v);
    try launch(k.down_combine, s, .{ r, d / 128, 1 }, 32 * slots, &a);
}

test "template arguments of the served experts kernels" {
    var t5: [5]i64 = undefined;
    try std.testing.expect(templateArgs("ILi2ELi2ELi10ELb1ELi4EEEvPK6__half", &t5));
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 10, 1, 4 }, &t5);
    var t8: [8]i64 = undefined;
    try std.testing.expect(templateArgs("ILi2ELi8ELi4ELi1ELi8ELi8ELi2ELb0EEEvPK6__half", &t8));
    try std.testing.expectEqualSlices(i64, &.{ 2, 8, 4, 1, 8, 8, 2, 0 }, &t8);
    try std.testing.expect(!templateArgs("ILi2ELi8EEEv", &t5));
    try std.testing.expect(!templateArgs("v", &t5));
}

test "a TP2 DeepSeek-V4.1 rank's prompt experts: settings, lists and ranges" {
    // D 5120, I 1152, 385 experts (the shared one last), 7 slots
    try std.testing.expectEqual([4]usize{ 8, 4, 4, 1 }, try config(5120, 1152, true));
    try std.testing.expectEqual([4]usize{ 8, 4, 1, 1 }, try config(1152, 5120, false));
    try std.testing.expectEqual(@as(usize, 2048), memberStride(2048));
    try std.testing.expectEqual(@as(usize, 192), memberStride(181));
    try std.testing.expectEqual(@as(usize, 64), memberStride(64));
    try std.testing.expectEqual(@as(usize, 32), memberStride(20));
    try std.testing.expectEqual(@as(usize, (14336 + 385 * 63) / 64), listLen(2048 * 7, 385, 64));
    try std.testing.expectEqual(@as(usize, 0), range(8, 8));
    try std.testing.expectEqual(@as(usize, 1), range(3, 10));
    try std.testing.expectEqual(@as(usize, 2), range(4, 12));
    try std.testing.expectEqual(@as(usize, 5120), Scratch.promptZ(5120, 1152));
}
