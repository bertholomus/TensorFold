//! The routed experts of a prompt chunk (exl3/experts.py routed(), the fast path the served build takes at 64 rows and
//! more, with its switches at their served values: the "mma" prompt kernel, work lists, no shared-dense GEMM): eight
//! launches of the served build's kernels (tensorfold_exl3_experts_v19: experts.cu's cubin and the codebook's
//! experts_cb*.cu cubin), as experts.cpp / experts.cu / experts_grouped.cuh launch them:
//! group_count, group_place, work_list, rot_in, grouped_mma (gate and up, the K splits folded), gateup_epilogue,
//! grouped_rows (down) and down_combine. And of a decode window (fewer rows: _routed_fused, the path routed() takes
//! with DECODE "fused", DECODE_PDL, DECODE_READY and the L2 discards at their served values): three launches,
//! decode_prep (the grouping and gate / up's rotated rows), grouped_cp_kernel for gate / up with their epilogue and
//! grouped_cp_kernel for down with the rows' combine, the last two programmatic dependents.
//! The map: pipeline/EXL3-EXPERTS-LAUNCHES.md.
const std = @import("std");
const cuda = @import("cuda");
const elf = @import("elf.zig");
const weights = @import("weights.zig");

/// experts.py's settings (GLM_GATEUP, GLM_DOWN by default_config; PROMPT_TILES; MMA_KB; EXACT_ROWS; DECODE_STAGES).
pub const exact_rows = 64;
pub const place_threads = 256; // experts.cu PLACE_THREADS
pub const list_threads = 1024; // experts.cu LIST_THREADS
pub const mma_cols = 64; // experts_grouped.cuh MMA_COLS
pub const mma_rows_gu = 64; // work list rows of the mma kernel
pub const rows_rows_d = 32; // work list rows of grouped_rows: 16 * PROMPT_TILES["down"][2]
pub const act_f32 = 1; // ACT_F32
pub const prep_threads = 256; // experts.cu PREP_THREADS
pub const decode_stages = 3; // DECODE_STAGES: each warp's ring of trellis copies in grouped_cp_kernel (its S)

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
    decode_prep: cuda.Function, // <__nv_bfloat16>
    mma: [3]?cuda.Function = @splat(null), // grouped_mma_kernel<CB, LO, HI, FOLD true, KB 4> by range()
    rows: [3]?cuda.Function = @splat(null), // grouped_rows_kernel<CB, 8, 4, 1, LO, HI, 2, FOLD false> by range()
    cp: [3][2]?cuda.Function = @splat(@splat(null)), // grouped_cp_kernel<CB, 8, 4, 3, LO, HI, EPI> by range(), EPI - 1

    /// experts.cu's cubin and the codebook's (experts_cb<codebook>.cu); kernels by name (the anonymous namespace's
    /// hash changes with every build) and template arguments.
    pub fn load(d: *const cuda.Driver, base_cubin: []const u8, cb_cubin: []const u8, codebook: u32) !Kernels {
        var base = try cuda.Module.load(d, base_cubin);
        errdefer base.unload();
        var cb = try cuda.Module.load(d, cb_cubin);
        errdefer cb.unload();
        var k: Kernels = .{ .base = base, .cb = cb, .group_count = undefined, .group_place = undefined, .work_list = undefined, .rot_in = undefined, .gateup_epilogue = undefined, .down_combine = undefined, .decode_prep = undefined };
        var found: u7 = 0;
        var names = elf.Symbols.init(base_cubin) orelse return error.BadCubin;
        while (names.next()) |name| {
            const plain = [_]struct { []const u8, u7 }{ .{ "18group_count_kernelE", 1 }, .{ "18group_place_kernelE", 2 }, .{ "16work_list_kernelE", 4 }, .{ "13rot_in_kernelI13__nv_bfloat16E", 8 }, .{ "22gateup_epilogue_kernelE", 16 }, .{ "19down_combine_kernelE", 32 }, .{ "18decode_prep_kernelI13__nv_bfloat16E", 64 } };
            for (plain) |e| {
                if (std.mem.indexOf(u8, name, e[0]) == null) continue;
                const f = try fnByName(base, name);
                switch (e[1]) {
                    1 => k.group_count = f,
                    2 => k.group_place = f,
                    4 => k.work_list = f,
                    8 => k.rot_in = f,
                    16 => k.gateup_epilogue = f,
                    32 => k.down_combine = f,
                    else => k.decode_prep = f,
                }
                found |= e[1];
            }
        }
        if (found != 127) return error.MissingKernel;
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
            } else if (std.mem.indexOf(u8, name, "17grouped_cp_kernelI")) |at| {
                var t: [7]i64 = undefined;
                if (!templateArgs(name[at + "17grouped_cp_kernel".len ..], &t)) continue;
                // <CB, NT, W, S, LO, HI, EPI>: grouped_cp_launch's one setting, the gate / up (1) and down (2) epilogues
                if (t[0] != codebook or t[1] != 8 or t[2] != 4 or t[3] != decode_stages or t[6] < 1 or t[6] > 2) continue;
                k.cp[range(@intCast(t[4]), @intCast(t[5]))][@intCast(t[6] - 1)] = try fnByName(cb, name);
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
    return m.function(try std.mem.printSentinel(&buf, "{s}", .{name}, 0));
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
    pub fn sizes(rows: usize, slots: usize, d: usize, i: usize, e: usize) [11]usize {
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

/// experts_grouped.cuh's DecodeEpi, field for field (C layout, its defaults; the C padding as zero fields, the bytes
/// every recorded launch passed): grouped_cp_kernel's by-value epilogue. The gate / up launch fills svh_g .. act_mode,
/// the down launch svh_d .. store_y, both pick, E, cnt, the readiness (ready, ready_cnt, epoch) and discard.
pub const DecodeEpi = extern struct {
    pick: u64 = 0,
    E: c_int = 0,
    pad0: c_int = 0,
    svh_g: u64 = 0,
    svh_u: u64 = 0,
    suh_d: u64 = 0,
    xd: u64 = 0,
    limit: f32 = 0,
    act_mode: c_int = 1,
    svh_d: u64 = 0,
    y: u64 = 0,
    wts: u64 = 0,
    add: u64 = 0,
    out: u64 = 0,
    store_y: c_int = 1,
    pad1: c_int = 0,
    cnt: u64 = 0,
    ready: u64 = 0,
    ready_cnt: u64 = 0,
    epoch: u64 = 0,
    discard: c_int = 0,
    pad2: c_int = 0,
};

comptime {
    // the layout the kernels were compiled with (pointers 8 bytes, ints and floats 4, C padding)
    std.debug.assert(@sizeOf(DecodeEpi) == 144 and @offsetOf(DecodeEpi, "svh_g") == 16 and @offsetOf(DecodeEpi, "cnt") == 104 and @offsetOf(DecodeEpi, "discard") == 136);
}

/// A decode Scratch (experts.py Scratch(prompt=False); the served engine's holds max(n, 64) rows) for windows of up to
/// `rows` rows of `slots` slots: device addresses the caller allocated with these sizes and initial contents. A call
/// writes ids, count and members (decode_prep), the routed slots' xg and xu rows (decode_prep), z and xd (gate / up)
/// and y (down) before it reads them. epoch is the call's number (decode_prep adds one) and ready[u] the number expert
/// place u's rows were last published to down under: both carry over from call to call. The counters cnt_gu, cnt_d and
/// ready_cnt start at zero and each launch's last program leaves them zero. Scratch's counts, no_y and no_work reach
/// no decode launch (experts.cu passes a null pointer for each tensor a launch does not take).
pub const DecodeScratch = struct {
    rows: usize,
    slots: usize,
    xg: u64, // fp16 [rows * slots, D], zeros
    xu: u64, // fp16 [rows * slots, D], zeros
    xd: u64, // fp16 [rows * slots, I], zeros
    z: u64, // fp32 [max(2 * gate/up splits * I, down splits * D) * rows * slots], zeros
    y: u64, // fp32 [rows * slots, D], zeros
    cnt_gu: u64, // int32 [maxu * max(1, I / 128)], zeros
    cnt_d: u64, // int32 [rows * max(1, D / 128)], zeros
    epoch: u64, // int32 [1], zeros
    ready: u64, // int32 [max(1, maxu)], zeros
    ready_cnt: u64, // int32 [max(1, maxu)], zeros
    ids: u64, // int32 [maxu], zeros
    count: u64, // int32 [1], zeros
    members: u64, // int32 [maxu * ceil(rows / 128) * 128], -1

    /// Bytes of each buffer, in the field order above (xg .. members); maxu = min(rows * slots, E).
    pub fn sizes(rows: usize, slots: usize, d: usize, i: usize, e: usize) ![13]usize {
        const p = rows * slots;
        const maxu = @min(p, e);
        const one: usize = 1;
        const zrow = @max(2 * (try config(d, i, true))[2] * i, (try config(i, d, false))[2] * d);
        return .{ 2 * p * d, 2 * p * d, 2 * p * i, 4 * zrow * p, 4 * p * d, 4 * maxu * @max(one, i / 128), 4 * rows * @max(one, d / 128), 4, 4 * @max(one, maxu), 4 * @max(one, maxu), 4 * maxu, 4, 4 * maxu * ((rows + 127) / 128 * 128) };
    }
};

/// One of decode's launches: its kernel (decode_prep, or grouped_cp_kernel with the gate / up or the down epilogue at
/// K2 range `range`), configuration and arguments.
pub const DecodeLaunch = struct {
    kernel: enum { prep, gate_up, down },
    range: usize = 0,
    cfg: cuda.Config,
    args: cuda.Args = .{},
};

/// routed(x, pick, wts, ex, s, None, R, limit, ACT_F32, before_down=None, shared_slot=slots - 1) on a decode window
/// (R < 64), the served Model.moe's one call a layer: out fp32 [R, D] = each row's slots' expert outputs weighted by
/// wts, summed in slot order. x bf16 [R, D] (row stride x_stride elements), pick int32 and wts fp32 [R, slots]
/// contiguous, every pick an expert id (the shared expert, the last, in every row's last slot); out need not be
/// initialized. shared_slot changes nothing here (only a prompt chunk's SHARED_DENSE path reads it), and without
/// before_down nothing waits before down. Three launches on `s`: decode_prep plain, gate / up and down as programmatic
/// dependents.
pub fn decode(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: DecodeScratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) !void {
    var ls = try decodeLaunches(ex, sc, x, x_stride, pick, wts, out, r, limit);
    var fs: [3]cuda.Function = undefined;
    for (&ls, &fs) |*l, *f| f.* = switch (l.kernel) {
        .prep => k.decode_prep,
        .gate_up => k.cp[l.range][0] orelse return error.MissingKernel,
        .down => k.cp[l.range][1] orelse return error.MissingKernel,
    };
    for (&ls, fs) |*l, f| try cuda.launch.launch(f, l.cfg, s, &l.args);
}

/// decode's three launches in order: experts.py _routed_fused, experts.cu exl3x_decode_prep_cuda and
/// exl3x_grouped_decode_cuda, experts_grouped.cuh grouped_cp_launch.
pub fn decodeLaunches(ex: weights.Experts, sc: DecodeScratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, r: usize, limit: f32) ![3]DecodeLaunch {
    const d: usize = ex.dims;
    const i: usize = ex.width;
    const e: usize = ex.count;
    const slots = sc.slots;
    if (r == 0 or r >= exact_rows or r > sc.rows) return error.NotADecodeWindow;
    // routed()'s gate of the fused path: GLM's tiles for gate / up (8 n tiles, 4 warps) and down (one split), 128-column
    // blocks, at most 32 slots; and every trellis 16-byte aligned (weights packs each at a multiple of 16 bytes)
    const cgu = try config(d, i, true);
    const cd = try config(i, d, false);
    if (cgu[0] != 8 or cgu[1] != 4 or cd[0] != 8 or cd[1] != 4 or cd[2] != 1) return error.UnsupportedShape;
    if (i % 128 != 0 or d % 128 != 0 or slots > 32) return error.UnsupportedShape;
    const p = r * slots;
    const maxu = @min(p, e); // the window's expert places (Scratch.window): ids [maxu], members [maxu, R]
    const mt = (r + 15) / 16; // a place's 16-row member tiles

    // decode_prep: block 0 adds one to epoch, groups the picks (ids, count, members) and combines the rows that route
    // no slot; the other blocks rotate gate / up's input rows, a warp a (member, 128 columns, matrix)
    var prep: cuda.Args = .{};
    prep.add(x);
    i32a(&prep, x_stride);
    for ([_]u64{ pick, ex.suh_g, ex.suh_u, sc.xg, sc.xu }) |v| prep.add(v);
    for ([_]usize{ d, slots, e, r }) |v| i32a(&prep, v);
    for ([_]u64{ sc.ids, sc.count, sc.members }) |v| prep.add(v);
    i32a(&prep, r); // maxm: the member lists' stride
    for ([_]u64{ wts, sc.y, 0, out }) |v| prep.add(v); // no add
    for ([_]usize{ d, 1 }) |v| i32a(&prep, v); // store_y: a slot that is not routed adds the y it finds
    prep.add(sc.epoch);
    const prep_warps = prep_threads / 32;
    const blocks = 1 + (p * (d / 128) * 2 + prep_warps - 1) / prep_warps;

    // gate and up: grid (places, I / 128, 2 matrices x K splits x member tiles), 4 warps copying trellis words through
    // 3-stage rings, the last place first; the program completing a (place, 128 columns) sums the splits into the
    // gate / up epilogue (xd) and the place's last such program publishes ready[place] = epoch; with discard the
    // summed partials (8 x splits <= 32 lanes) and the place's rotated rows leave L2 unwritten
    var gu: cuda.Args = .{};
    for ([_]u64{ sc.xg, sc.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, sc.ids, sc.count, sc.members, sc.z }) |v| gu.add(v);
    for ([_]usize{ d, i, p, cgu[2], r, slots }) |v| i32a(&gu, v);
    gu.add(DecodeEpi{ .pick = pick, .E = @intCast(e), .svh_g = ex.svh_g, .svh_u = ex.svh_u, .suh_d = ex.suh_d, .xd = sc.xd, .limit = limit, .act_mode = act_f32, .cnt = sc.cnt_gu, .ready = sc.ready, .ready_cnt = sc.ready_cnt, .epoch = sc.epoch, .discard = @intFromBool(8 * cgu[2] <= 32) });

    // down: grid (places, D / 128, member tiles), one split; a program copies its first trellis steps, waits for its
    // place's ready == epoch, writes its rows' slot outputs to y, and the program completing a (row, 128 columns)
    // combines the row's slots with wts into out, then drops their y from L2 (4 x slots <= 32 lanes); limit and
    // act_mode keep DecodeEpi's defaults
    var dn: cuda.Args = .{};
    for ([_]u64{ sc.xd, sc.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, sc.ids, sc.count, sc.members, sc.z }) |v| dn.add(v);
    for ([_]usize{ i, d, p, 1, r, slots }) |v| i32a(&dn, v);
    dn.add(DecodeEpi{ .pick = pick, .E = @intCast(e), .svh_d = ex.svh_d, .y = sc.y, .wts = wts, .out = out, .cnt = sc.cnt_d, .ready = sc.ready, .ready_cnt = sc.ready_cnt, .epoch = sc.epoch, .discard = @intFromBool(4 * slots <= 32) });

    return .{
        .{ .kernel = .prep, .cfg = .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = prep_threads }, .shared = @intCast(2 * p * 4) }, .args = prep },
        .{ .kernel = .gate_up, .range = range(ex.k2_gu[0], ex.k2_gu[1]), .cfg = .{ .grid = .{ .x = @intCast(maxu), .y = @intCast(i / 128), .z = @intCast(2 * cgu[2] * mt) }, .block = .{ .x = 4 * 32 }, .pdl = true }, .args = gu },
        .{ .kernel = .down, .range = range(ex.k2_d[0], ex.k2_d[1]), .cfg = .{ .grid = .{ .x = @intCast(maxu), .y = @intCast(d / 128), .z = @intCast(mt) }, .block = .{ .x = 4 * 32 }, .pdl = true }, .args = dn },
    };
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

test "a TP2 DeepSeek-V4.1 rank's decode scratch and windows" {
    // a decoder layer (D 5120, I 1152, 385 experts, 7 slots) and an MTP layer (129 experts, 4 slots), 64 rows
    try std.testing.expectEqual([13]usize{ 4587520, 4587520, 1032192, 16515072, 9175040, 13860, 10240, 4, 1540, 1540, 1540, 4, 197120 }, try DecodeScratch.sizes(64, 7, 5120, 1152, 385));
    try std.testing.expectEqual([13]usize{ 2621440, 2621440, 589824, 9437184, 5242880, 4644, 10240, 4, 516, 516, 516, 4, 66048 }, try DecodeScratch.sizes(64, 4, 5120, 1152, 129));
    // 1 to 63 rows (no more than the scratch holds); 64 and more are prompt chunks
    const ex: weights.Experts = .{ .trellis = 0, .gate_ptr = 0, .up_ptr = 0, .down_ptr = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = 385, .dims = 5120, .width = 1152, .down_k = 1152, .down_n = 5120, .k2_gu = .{ 6, 10 }, .k2_d = .{ 6, 10 } };
    const sc: DecodeScratch = .{ .rows = 64, .slots = 7, .xg = 0, .xu = 0, .xd = 0, .z = 0, .y = 0, .cnt_gu = 0, .cnt_d = 0, .epoch = 0, .ready = 0, .ready_cnt = 0, .ids = 0, .count = 0, .members = 0 };
    for ([_]usize{ 0, 64, 65 }) |r| try std.testing.expectError(error.NotADecodeWindow, decodeLaunches(ex, sc, 0, 5120, 0, 0, 0, r, 10));
    var small = sc;
    small.rows = 8;
    try std.testing.expectError(error.NotADecodeWindow, decodeLaunches(ex, small, 0, 5120, 0, 0, 0, 9, 10));
    _ = try decodeLaunches(ex, small, 0, 5120, 0, 0, 0, 8, 10);
}
