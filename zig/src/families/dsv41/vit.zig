//! The vision tower and aligner (DeepSeek's MIT reference vision.py; the served lane's Tower.encode / span_rows) on the
//! GPU, each step the op torch runs in the lane so every intermediate has its bytes: the patch embed and the bias
//! linears by torch's cuBLASLt calls (cublaslt.zig), the bias-free MLP linears by its cublasGemmEx (cublas.zig), RMSNorm
//! by the port's torch-order kernel (exact.zig), the attention by torch's own fp32 memory-efficient kernel (fmha.zig),
//! the rest (2D RoPE tables, rotary, silu * up, unfold, gelu, the span) in dsv41_ops.cu at torch's rounding points.
//! Rank 0 runs it; the span rows go into the prompt in place of the image tokens' embeddings.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const st = core.safetensors;
const cublas = @import("cublas.zig");
const cublaslt = @import("cublaslt.zig");
const exact_mod = @import("exact.zig");
const fmha_mod = @import("fmha.zig");
const picture = @import("picture.zig");

pub const dim = 1024;
pub const heads = 16;
pub const layers = 32;
pub const inter = 2816;
pub const patch_in = 3 * 14 * 14;
pub const model_dim = 5120;
pub const unfold_dim = dim * 9;
const eps: f32 = 1e-6;
const theta: f32 = 10000.0;

const Block = struct { norm1: u64, wqkv: u64, bqkv: u64, wo: u64, bo: u64, norm2: u64, w1: u64, w2: u64 };

/// One step's output a gate checks (null: no probe): patch embed, each block, the final norm, the aligner rows.
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, what: []const u8, index: usize, ptr: u64, bytes: usize) anyerror!void,

    fn call(p: ?Probe, what: []const u8, index: usize, ptr: u64, bytes: usize) !void {
        if (p) |x| try x.f(x.ctx, what, index, ptr, bytes);
    }
};

/// The checkpoint's shards that hold the vision tensors, by model.safetensors.index.json's weight_map.
const Sources = struct {
    gpa: std.mem.Allocator,
    files: std.ArrayList(st.File) = .empty,
    paths: std.ArrayList([]const u8) = .empty,
    arena: std.heap.ArenaAllocator, // the index's parse (192,452 entries: an arena, not the caller's allocator)

    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Sources {
        var sources: Sources = .{ .gpa = gpa, .arena = .init(std.heap.page_allocator) };
        errdefer sources.close(io);
        const a = sources.arena.allocator();
        const index_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(1 << 26));
        const map = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const wm = map.object.get("weight_map") orelse return error.NoWeightMap;
        var it = wm.object.iterator();
        while (it.next()) |kv| {
            const k = kv.key_ptr.*;
            if (!(std.mem.startsWith(u8, k, "vision.") or std.mem.startsWith(u8, k, "aligner.") or std.mem.startsWith(u8, k, "image_"))) continue;
            const f = kv.value_ptr.*.string;
            for (sources.paths.items) |p| {
                if (std.mem.endsWith(u8, p, f)) break;
            } else {
                const path = try std.fs.path.join(gpa, &.{ dir, f });
                try sources.paths.append(gpa, path);
                try sources.files.append(gpa, try st.File.open(gpa, io, path));
            }
        }
        return sources;
    }

    fn get(s: *const Sources, name: []const u8) ?st.Tensor {
        for (s.files.items) |*f| if (f.get(name)) |t| return t;
        return null;
    }

    fn close(s: *Sources, io: std.Io) void {
        for (s.files.items) |*f| f.close(io);
        s.files.deinit(s.gpa);
        for (s.paths.items) |p| s.gpa.free(p);
        s.paths.deinit(s.gpa);
        s.arena.deinit();
    }
};

pub const Tower = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    weights: cuda.DeviceBuffer,
    blocks: [layers]Block,
    patch_w: u64,
    patch_b: u64,
    norm: u64,
    al_w1: u64,
    al_b1: u64,
    al_w2: u64,
    al_b2: u64,
    image_start: u64,
    image_newline: u64,
    image_end: u64,
    module: cuda.Module,
    k_rope: cuda.Function,
    k_rot: cuda.Function,
    k_to_bf16: cuda.Function,
    k_add: cuda.Function,
    k_silu_mul: cuda.Function,
    k_unfold: cuda.Function,
    k_gelu: cuda.Function,
    k_span: cuda.Function,
    exact: *const exact_mod.Exact,
    blas: *const cublas.Blas,
    lt: cublaslt.Lt,
    fmha: fmha_mod.Fmha,
    // the workspace for the largest image so far
    work: ?cuda.DeviceBuffer = null,
    work_rows: usize = 0,

    const names = [_][]const u8{ "attn.wqkv.weight", "attn.wqkv.bias", "attn.wo.weight", "attn.wo.bias", "norm1.weight", "norm2.weight", "mlp.w1.weight", "mlp.w2.weight" };

    /// The tower's weights (bf16, the checkpoint's own) on the device, its kernels, and torch's attention cubin.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, model_dir: []const u8, fmha_cubin: []const u8, ops_image: []const u8, exact: *const exact_mod.Exact, blas: *const cublas.Blas, blas_workspace: u64) !Tower {
        // only the shards the index names for these tensors (the vision tensors sit in one of the 39)
        var sh = try Sources.open(gpa, io, model_dir);
        defer sh.close(io);
        // every tensor's bytes, laid end to end at 256-byte alignment (as torch's allocator places its parameters)
        var total: usize = 0;
        var keys: std.ArrayList([]const u8) = .empty;
        defer {
            for (keys.items) |k| gpa.free(k);
            keys.deinit(gpa);
        }
        for (0..layers) |i| for (names) |nm| try keys.append(gpa, try std.fmt.allocPrint(gpa, "vision.blocks.{d}.{s}", .{ i, nm }));
        for ([_][]const u8{ "vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", "vision.norm.weight", "aligner.w1.weight", "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias", "image_start", "image_newline", "image_end" }) |k| try keys.append(gpa, try gpa.dupe(u8, k));
        for (keys.items) |k| {
            const t = sh.get(k) orelse return error.MissingVisionTensor;
            total += std.mem.alignForward(usize, t.bytes.len, 256);
        }
        var buf = try cuda.DeviceBuffer.alloc(d, total);
        errdefer buf.free();
        var at: usize = 0;
        const ptrs = try gpa.alloc(u64, keys.items.len);
        defer gpa.free(ptrs);
        // each tensor read out of the mapping into plain memory first: a copy straight from the file mapping into the
        // device crawls on GB10 (minutes for these 970 MB), a copy from anonymous memory takes well under a second
        var tmp: std.ArrayList(u8) = .empty;
        defer tmp.deinit(std.heap.page_allocator);
        for (keys.items, ptrs) |k, *p| {
            const t = sh.get(k).?;
            try tmp.resize(std.heap.page_allocator, t.bytes.len);
            @memcpy(tmp.items, t.bytes);
            try buf.upload(at, tmp.items);
            p.* = buf.ptr + at;
            at += std.mem.alignForward(usize, t.bytes.len, 256);
        }
        var tw: Tower = undefined;
        tw.d = d;
        tw.stream = stream;
        tw.weights = buf;
        for (0..layers) |i| {
            const b = ptrs[i * names.len ..];
            tw.blocks[i] = .{ .wqkv = b[0], .bqkv = b[1], .wo = b[2], .bo = b[3], .norm1 = b[4], .norm2 = b[5], .w1 = b[6], .w2 = b[7] };
        }
        const r = ptrs[layers * names.len ..];
        tw.patch_w = r[0];
        tw.patch_b = r[1];
        tw.norm = r[2];
        tw.al_w1 = r[3];
        tw.al_b1 = r[4];
        tw.al_w2 = r[5];
        tw.al_b2 = r[6];
        tw.image_start = r[7];
        tw.image_newline = r[8];
        tw.image_end = r[9];
        tw.module = try cuda.Module.load(d, ops_image);
        errdefer tw.module.unload();
        tw.k_rope = try tw.module.function("tf_ds_vrope_kernel");
        tw.k_rot = try tw.module.function("tf_ds_vrot_kernel");
        tw.k_to_bf16 = try tw.module.function("tf_ds_f32_to_bf16_kernel");
        tw.k_add = try tw.module.function("tf_ds_add_bf16_kernel");
        tw.k_silu_mul = try tw.module.function("tf_ds_vsilu_mul_kernel");
        tw.k_unfold = try tw.module.function("tf_ds_vunfold3_kernel");
        tw.k_gelu = try tw.module.function("tf_ds_vgelu_kernel");
        tw.k_span = try tw.module.function("tf_ds_vspan_kernel");
        tw.exact = exact;
        tw.blas = blas;
        tw.lt = try cublaslt.Lt.open(blas, stream, blas_workspace);
        tw.fmha = try fmha_mod.Fmha.load(d, fmha_cubin);
        tw.work = null;
        tw.work_rows = 0;
        return tw;
    }

    pub fn close(tw: *Tower) void {
        if (tw.work) |*w| w.free();
        tw.fmha.unload();
        tw.lt.close();
        tw.module.unload();
        tw.weights.free();
    }

    /// The workspace's parts for n patches and l cells (each 256-byte aligned).
    const Work = struct {
        x: u64, // bf16 [n, 1024]: the residual stream
        h: u64, // bf16 [n, 1024]: a norm's output, then attention's bf16 output
        qkv: u64, // bf16 [n, 3072]
        q32: u64,
        k32: u64,
        v32: u64,
        o32: u64, // fp32 [n, 1024] each
        a: u64, // bf16 [n, 1024]: wo's and w2's output
        gu: u64, // bf16 [n, 5632]
        m: u64, // bf16 [n, 2816]
        cos: u64,
        sin: u64, // fp32 [n, 32]
        patches: u64, // bf16 [n, 588]
        unfold: u64, // bf16 [9216, l]
        a1: u64, // bf16 [l, 5120]
        rows: u64, // bf16 [l, 5120]: the aligner's output
        bytes: usize,

        fn plan(base: u64, n: usize, l: usize) Work {
            var at: usize = 0;
            const sizes = [_]usize{ n * dim * 2, n * dim * 2, n * 3 * dim * 2, n * dim * 4, n * dim * 4, n * dim * 4, n * dim * 4, n * dim * 2, n * 2 * inter * 2, n * inter * 2, n * 32 * 4, n * 32 * 4, n * patch_in * 2, unfold_dim * l * 2, l * model_dim * 2, l * model_dim * 2 };
            var p: [sizes.len]u64 = undefined;
            for (sizes, 0..) |s, i| {
                p[i] = base + at;
                at += std.mem.alignForward(usize, s, 256);
            }
            return .{ .x = p[0], .h = p[1], .qkv = p[2], .q32 = p[3], .k32 = p[4], .v32 = p[5], .o32 = p[6], .a = p[7], .gu = p[8], .m = p[9], .cos = p[10], .sin = p[11], .patches = p[12], .unfold = p[13], .a1 = p[14], .rows = p[15], .bytes = at };
        }
    };

    fn ensure(tw: *Tower, n: usize, l: usize) !Work {
        const need = Work.plan(0, n, l).bytes;
        if (tw.work == null or tw.work.?.len < need) {
            if (tw.work) |*w| w.free();
            tw.work = null;
            tw.work = try cuda.DeviceBuffer.alloc(tw.d, need);
        }
        return Work.plan(tw.work.?.ptr, n, l);
    }

    /// The workspace for the largest picture up front (n patches, l aligner rows), so serving allocates nothing.
    pub fn reserve(tw: *Tower, n: usize, l: usize) !void {
        _ = try tw.ensure(n, l);
    }

    fn blocksFor(count: usize, per: usize) u32 {
        return @intCast(@max(1, @min(65535, (count + per - 1) / per)));
    }

    /// Tower.encode of one picture (its patches on the host, bf16): the aligner's rows bf16 [n_llm_h * n_llm_w, 5120],
    /// left in the workspace (the returned address) until the next call.
    pub fn encode(tw: *Tower, g: picture.Grid, patches: []const u16, probe: ?Probe) !u64 {
        const n = g.patches();
        const l = g.n_llm_h * g.n_llm_w;
        const w = try tw.ensure(n, l);
        const s = tw.stream;
        try tw.upload(w.patches, std.mem.sliceAsBytes(patches));
        // the patch embed, then the 2D RoPE tables of this grid
        try tw.lt.linearBias(w.patches, false, tw.patch_w, tw.patch_b, w.x, n, patch_in, dim);
        try Probe.call(probe, "embed", 0, w.x, n * dim * 2);
        {
            var a: cuda.Args = .{};
            a.add(w.cos);
            a.add(w.sin);
            a.add(@as(c_int, @intCast(n)));
            a.add(@as(c_int, @intCast(g.n_vit_w)));
            a.add(theta);
            try cuda.launch.launch(tw.k_rope, .{ .grid = .{ .x = blocksFor(n * 32, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        try Probe.call(probe, "cos", 0, w.cos, n * 32 * 4);
        try Probe.call(probe, "sin", 0, w.sin, n * 32 * 4);
        for (tw.blocks, 0..) |b, i| {
            try tw.block(b, w, n);
            try Probe.call(probe, "block", i, w.x, n * dim * 2);
        }
        try tw.exact.rmsNorm(s, w.x, dim, tw.norm, w.h, dim, n, dim, eps);
        try Probe.call(probe, "norm", 0, w.h, n * dim * 2);
        // the aligner: 3x3 cells unfolded (zeros past the grid), w1 on the transposed unfold, gelu, w2
        {
            var a: cuda.Args = .{};
            a.add(w.h);
            a.add(w.unfold);
            a.add(@as(c_int, @intCast(g.n_vit_h)));
            a.add(@as(c_int, @intCast(g.n_vit_w)));
            a.add(@as(c_int, dim));
            a.add(@as(c_int, @intCast(g.n_llm_h)));
            a.add(@as(c_int, @intCast(g.n_llm_w)));
            try cuda.launch.launch(tw.k_unfold, .{ .grid = .{ .x = blocksFor(unfold_dim * l, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        try tw.lt.linearBias(w.unfold, true, tw.al_w1, tw.al_b1, w.a1, l, unfold_dim, model_dim);
        {
            var a: cuda.Args = .{};
            a.add(w.a1);
            a.add(@as(c_ulonglong, l * model_dim));
            try cuda.launch.launch(tw.k_gelu, .{ .grid = .{ .x = blocksFor(l * model_dim, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        try tw.lt.linearBias(w.a1, false, tw.al_w2, tw.al_b2, w.rows, l, model_dim, model_dim);
        try Probe.call(probe, "rows", 0, w.rows, l * model_dim * 2);
        return w.rows;
    }

    fn block(tw: *Tower, b: Block, w: Work, n: usize) !void {
        const s = tw.stream;
        const count = n * dim;
        try tw.exact.rmsNorm(s, w.x, dim, b.norm1, w.h, dim, n, dim, eps);
        try tw.lt.linearBias(w.h, false, b.wqkv, b.bqkv, w.qkv, n, dim, 3 * dim);
        {
            var a: cuda.Args = .{};
            a.add(w.qkv);
            a.add(w.cos);
            a.add(w.sin);
            a.add(w.q32);
            a.add(w.k32);
            a.add(w.v32);
            a.add(@as(c_int, @intCast(n)));
            try cuda.launch.launch(tw.k_rot, .{ .grid = .{ .x = blocksFor(n * heads * 32, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        try tw.fmha.run(s, w.q32, w.k32, w.v32, w.o32, n, heads);
        {
            var a: cuda.Args = .{};
            a.add(w.o32);
            a.add(w.h);
            a.add(@as(c_ulonglong, count));
            try cuda.launch.launch(tw.k_to_bf16, .{ .grid = .{ .x = blocksFor(count, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        try tw.lt.linearBias(w.h, false, b.wo, b.bo, w.a, n, dim, dim);
        try tw.add(w.x, w.a, count);
        try tw.exact.rmsNorm(s, w.x, dim, b.norm2, w.h, dim, n, dim, eps);
        try tw.blas.linearBf16(w.h, b.w1, w.gu, n, dim, 2 * inter);
        {
            var a: cuda.Args = .{};
            a.add(w.gu);
            a.add(w.m);
            a.add(@as(c_int, @intCast(n)));
            a.add(@as(c_int, inter));
            try cuda.launch.launch(tw.k_silu_mul, .{ .grid = .{ .x = blocksFor(n * inter, 256) }, .block = .{ .x = 256 } }, s, &a);
        }
        try tw.blas.linearBf16(w.m, b.w2, w.a, n, inter, dim);
        try tw.add(w.x, w.a, count);
    }

    /// cuMemcpyHtoDAsync on the tower's stream (a pageable source: the driver stages it before returning).
    fn upload(tw: *const Tower, dst: u64, bytes: []const u8) !void {
        const d = tw.d;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, tw.stream.handle), "cuMemcpyHtoDAsync");
        try tw.stream.synchronize(); // (the host copy may go once the call returns)
    }

    fn add(tw: *Tower, x: u64, y: u64, count: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(c_ulonglong, count));
        try cuda.launch.launch(tw.k_add, .{ .grid = .{ .x = blocksFor(count, 256) }, .block = .{ .x = 256 } }, tw.stream, &a);
    }

    /// Tower.span_rows: the image span's rows bf16 [span tokens, 5120] into `out` (device), the delimiters' learned
    /// embeddings around the aligner's rows (encode's output at `rows`).
    pub fn spanRows(tw: *Tower, g: picture.Grid, rows: u64, out: u64, scratch: u64) !void {
        // scratch: the span's types (u8) then each token's aligner row (i32), uploaded together
        const t = g.spanTokens();
        var types_buf: [4096]picture.Type = undefined;
        var row_of: [4096]i32 = undefined;
        if (t > types_buf.len) return error.SpanTooLong;
        picture.spanTypes(g, types_buf[0..t]);
        var r: i32 = 0;
        for (types_buf[0..t], 0..) |ty, i| {
            row_of[i] = if (ty == .image) r else 0;
            if (ty == .image) r += 1;
        }
        const types_at = scratch;
        const rows_at = scratch + std.mem.alignForward(usize, t, 256);
        try tw.upload(types_at, std.mem.sliceAsBytes(types_buf[0..t]));
        try tw.upload(rows_at, std.mem.sliceAsBytes(row_of[0..t]));
        var a: cuda.Args = .{};
        a.add(types_at);
        a.add(rows);
        a.add(tw.image_start);
        a.add(tw.image_newline);
        a.add(tw.image_end);
        a.add(out);
        a.add(@as(c_int, @intCast(t)));
        a.add(@as(c_int, model_dim));
        a.add(rows_at);
        try cuda.launch.launch(tw.k_span, .{ .grid = .{ .x = blocksFor(t * model_dim, 256) }, .block = .{ .x = 256 } }, tw.stream, &a);
    }
};
