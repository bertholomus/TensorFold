//! tf-dsv41-layer MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT REC_DIR [--layers N]: the Zig port's layer gate (M3).
//! Both ranks run the first 2,048-row prompt chunk the recording's layer fixtures hold through the port's prompt
//! forward (prompt.zig) on their GPUs and compare every point a layer exchanges with the served build's bytes: the
//! attention's input rows and partial, its gather, the MoE's input rows and partial, its gather. One JSON line a point;
//! the first difference stops the run (its first differing element and how many differ).
//! REC_DIR: aot/ (the recording's Triton set), cubins/ (linear.cubin, experts.cubin, experts_cb.cubin: the served
//! extension cubins), rope-plain-{cos,sin}.f32 and rope.json (zrec_fixtures.py), rank<R>/layers.jsonl and layers/.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const prompt = dsv41.prompt;

const usage = "usage: tf-dsv41-layer MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT REC_DIR [--layers N]\n";
const rope_rows = 8192; // the plain table's first rows (chunk 0's positions and more)
const chunk_rows = 2048;

/// One fixture line of layers.jsonl.
const Point = struct {
    call: u64,
    where: []const u8,
    layer: ?i64 = null,
    arg: []const u8,
    shape: []const i64,
    dtype: []const u8,
    sha256: ?[]const u8 = null,
    file: ?[]const u8 = null,
};

const Fixtures = struct {
    points: []Point,
    dir: []const u8,

    /// The first line after call `after` at `where` (and `layer`, when given) for argument `arg`.
    fn find(f: *const Fixtures, after: u64, where: []const u8, layer: ?i64, arg: []const u8) ?*const Point {
        for (f.points) |*p| {
            if (p.call <= after or !std.mem.eql(u8, p.where, where) or !std.mem.eql(u8, p.arg, arg)) continue;
            if (layer) |l| if (p.layer == null or p.layer.? != l) continue;
            return p;
        }
        return null;
    }
};

fn dtypeBytes(name: []const u8) usize {
    if (std.mem.eql(u8, name, "float32") or std.mem.eql(u8, name, "int32")) return 4;
    if (std.mem.eql(u8, name, "int64")) return 8;
    return 2;
}

fn numel(shape: []const i64) usize {
    var n: usize = 1;
    for (shape) |s| n *= @intCast(s);
    return n;
}

const Run = struct {
    a: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    s: cuda.Stream,
    fx: *const Fixtures,
    rank: u32,
    out: *std.Io.Writer,
    host: []u8,
    ok: bool = true,

    /// Our buffer `dev` against fixture point `p`: the same bytes (sha256), or where they first differ.
    fn check(r: *Run, label: []const u8, p: ?*const Point, dev: u64) !bool {
        const pt = p orelse {
            try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"error\": \"no fixture\"}}\n", .{ r.rank, label });
            try r.out.flush();
            r.ok = false;
            return false;
        };
        try r.s.synchronize();
        const bytes = numel(pt.shape) * dtypeBytes(pt.dtype);
        if (bytes > r.host.len) return error.PointTooBig;
        try r.d.check(r.d.api.cuMemcpyDtoH_v2(r.host.ptr, dev, bytes), "cuMemcpyDtoH");
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(r.host[0..bytes], &h, .{});
        var hex: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&hex, "{x}", .{&h}) catch unreachable;
        const same = pt.sha256 != null and std.mem.eql(u8, &hex, pt.sha256.?);
        try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"call\": {d}, \"bytes\": {d}, \"equal\": {}", .{ r.rank, label, pt.call, bytes, same });
        if (!same) {
            r.ok = false;
            if (pt.file) |name| try r.diff(pt, name, bytes);
        }
        try r.out.print("}}\n", .{});
        try r.out.flush();
        return same;
    }

    /// The first differing element against the fixture's saved tensor, and how many elements differ.
    fn diff(r: *Run, pt: *const Point, name: []const u8, bytes: usize) !void {
        const path = try std.fs.path.join(r.a, &.{ r.fx.dir, "layers", name });
        const want = std.Io.Dir.cwd().readFileAlloc(r.io, path, r.a, .limited(1 << 31)) catch |err| {
            try r.out.print(", \"diff\": \"{s} unreadable: {s}\"", .{ name, @errorName(err) });
            return;
        };
        defer r.a.free(want);
        if (want.len != bytes) {
            try r.out.print(", \"diff\": \"{d} bytes saved\"", .{want.len});
            return;
        }
        const esize = dtypeBytes(pt.dtype);
        var first: ?usize = null;
        var count: usize = 0;
        var i: usize = 0;
        while (i < bytes) : (i += esize) {
            if (!std.mem.eql(u8, r.host[i..][0..esize], want[i..][0..esize])) {
                if (first == null) first = i / esize;
                count += 1;
            }
        }
        if (first) |f| {
            const row_len = numel(pt.shape[1..]);
            try r.out.print(", \"first\": {d}, \"row\": {d}, \"col\": {d}, \"differ\": {d}, \"of\": {d}", .{ f, f / @max(row_len, 1), f % @max(row_len, 1), count, bytes / esize });
            try r.out.print(", \"ours\": \"{x}\", \"served\": \"{x}\"", .{ r.host[f * esize ..][0..esize], want[f * esize ..][0..esize] });
        }
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 8) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[3], 10);
    const world = try std.fmt.parseInt(u32, args[4], 10);
    const ip = try dsv41.link.parseIp(args[5]);
    const port = try std.fmt.parseInt(u16, args[6], 10);
    const rec = args[7];
    var layers: usize = 1;
    if (args.len >= 10 and std.mem.eql(u8, args[8], "--layers")) layers = try std.fmt.parseInt(usize, args[9], 10);

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w_out = &out.interface;

    const cfg = try dsv41.Config.read(a, io, args[1]);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();

    // the fixtures first (a bad REC_DIR fails before the weights load)
    const rank_dir = try std.fmt.allocPrint(a, "{s}/rank{d}", .{ rec, rank });
    const jl = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ rank_dir, "layers.jsonl" }), a, .limited(1 << 26));
    var points: std.ArrayList(Point) = .empty;
    var lines = std.mem.tokenizeScalar(u8, jl, '\n');
    while (lines.next()) |line| {
        const p = try std.json.parseFromSliceLeaky(Point, a, line, .{ .ignore_unknown_fields = true });
        try points.append(a, p);
    }
    const fx: Fixtures = .{ .points = points.items, .dir = rank_dir };
    // the first 2,048-row chunk: Model.forward's ids
    var start_call: u64 = 0;
    var ids_file: ?[]const u8 = null;
    for (fx.points) |p| {
        if (std.mem.eql(u8, p.where, "Model.forward") and std.mem.eql(u8, p.arg, "in2") and p.shape.len == 1 and p.shape[0] == chunk_rows) {
            start_call = p.call;
            ids_file = p.file;
            break;
        }
    }
    const ids_bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rank_dir, "layers", ids_file orelse return error.NoChunkFixture }), a, .limited(1 << 20), .@"8", null);
    const ids = std.mem.bytesAsSlice(i64, ids_bytes);

    // the weights (no DSpark blocks) from the lane's rank cache
    const t0 = std.Io.Timestamp.now(io, .awake);
    var cache_dir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
    defer cache_dir.close(io);
    const cache_name = try dsv41.rank_cache.find(a, io, cache_dir, rank, world);
    var ix = try dsv41.rank_cache.open(gpa, io, cache_dir, cache_name);
    defer ix.deinit();
    var cache_file = try cache_dir.openFile(io, cache_name, .{});
    defer cache_file.close(io);
    var w = try dsv41.weights.load(gpa, &driver, .{ .cache = .{ .file = cache_file, .index = &ix, .io = io } }, cfg, .{ .rank = rank, .world = world }, false);
    defer w.deinit();
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;

    var link = if (rank == 0) try dsv41.link.Link.listen(ip, port) else try dsv41.link.Link.connect(io, ip, port, 300);
    defer link.close();
    var comm = try dsv41.comm.Comm.init(link, rank, world);
    defer comm.deinit();

    // kernels: the recording's Triton set, the served extension cubins, the torch-op images
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, try std.fs.path.join(a, &.{ rec, "aot" }));
    defer set.deinit();
    const lin = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "linear.cubin" }), a, .limited(1 << 28), .@"16", null);
    var pf = try dsv41.exl3_prefill.Kernels.load(&driver, lin);
    defer pf.unload();
    const exb = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "experts.cubin" }), a, .limited(1 << 28), .@"16", null);
    const exc = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "experts_cb.cubin" }), a, .limited(1 << 28), .@"16", null);
    var exk = try dsv41.exl3_experts.Kernels.load(&driver, exb, exc, dsv41.exl3_linear.codebook_mul1);
    defer exk.unload();
    if (!cuda.kernels.available) return error.NoKernelImages;
    var ops = try dsv41.ops.Ops.load(&driver, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();

    // the plain RoPE table's first rows
    var rope_buf = try cuda.DeviceBuffer.alloc(&driver, 2 * rope_rows * (cfg.rope_dim / 2) * 4);
    defer rope_buf.free();
    for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32" }, 0..) |name, j| {
        var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ rec, name }), .{});
        defer f.close(io);
        const part = try a.alloc(u8, rope_rows * (cfg.rope_dim / 2) * 4);
        if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable;
        try rope_buf.upload(j * part.len, part);
    }
    const rope: prompt.Rope = .{ .cos = rope_buf.ptr, .sin = rope_buf.ptr + rope_rows * (cfg.rope_dim / 2) * 4 };

    var arena = try prompt.Arena.init(&driver, 3 << 30);
    defer arena.deinit();
    var blas_ws_ptr: u64 = undefined;
    var eng: prompt.Engine = .{ .d = &driver, .s = stream, .t = .{ .set = &set, .stream = stream }, .blas = undefined, .comm = &comm, .pf = &pf, .ex = &exk, .ops = &ops, .c = &cfg, .w = &w, .world = world, .plain = rope, .compressed = rope };
    var ch = try prompt.Chunk.init(&eng, &arena, chunk_rows);
    blas_ws_ptr = ch.blas_ws;
    var blas = try dsv41.cublas.Blas.open(stream, blas_ws_ptr);
    defer blas.close();
    eng.blas = &blas;
    const rings = try a.alloc(u64, cfg.layers);
    for (rings) |*r| {
        r.* = try arena.take(eng.ringBytes());
        try driver.check(driver.api.cuMemsetD8_v2(r.*, 0, eng.ringBytes()), "cuMemsetD8");
    }

    try w_out.print("{{\"rank\": {d}, \"loaded_s\": {d:.1}, \"weights_bytes\": {d}, \"chunk_call\": {d}, \"rows\": {d}, \"arena_bytes\": {d}}}\n", .{ rank, load_s, w.bytes, start_call, ids.len, arena.used });
    try w_out.flush();

    var run: Run = .{ .a = a, .io = io, .d = &driver, .s = stream, .fx = &fx, .rank = rank, .out = w_out, .host = try a.alloc(u8, 2 * chunk_rows * cfg.hidden * 4) };
    const host_pos = try a.alloc(i64, chunk_rows);
    try prompt.begin(&eng, &ch, ids, 0, host_pos);
    var after = start_call;
    for (0..layers) |li| {
        const l: i64 = @intCast(li);
        const att = fx.find(after, "Model.attention_k", l, "in2") orelse return error.NoLayerFixture;
        prompt.attnMixes(&eng, &ch, li) catch |err| return report(w_out, rank, li, "attention mixes", err);
        var label_buf: [64]u8 = undefined;
        if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention in", .{li}), att, ch.x)) break;
        prompt.attention(&eng, &ch, li, rings[li], 0) catch |err| return report(w_out, rank, li, "attention", err);
        if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention partial", .{li}), fx.find(after, "Model.attention_k", l, "out"), ch.pa)) break;
        try prompt.gather(&eng, &ch, ch.pa, ch.ga);
        if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention gather", .{li}), fx.find(att.call, "Comm.gather", null, "out"), ch.ga)) break;
        prompt.ffnMixes(&eng, &ch, li) catch |err| return report(w_out, rank, li, "ffn mixes", err);
        const mo = fx.find(att.call, "Model.moe", l, "in2");
        if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe in", .{li}), mo, ch.x)) break;
        prompt.moe(&eng, &ch, li) catch |err| return report(w_out, rank, li, "moe", err);
        if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe partial", .{li}), fx.find(att.call, "Model.moe", l, "out"), ch.pm)) break;
        try prompt.gather(&eng, &ch, ch.pm, ch.gm);
        if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe gather", .{li}), fx.find(mo.?.call, "Comm.gather", null, "out"), ch.gm)) break;
        prompt.endLayer(&ch);
        after = mo.?.call;
    }
    try w_out.print("{{\"rank\": {d}, \"layers\": {d}, \"all_equal\": {}}}\n", .{ rank, layers, run.ok });
    try w_out.flush();
    return if (run.ok) 0 else 1;
}

fn report(w: *std.Io.Writer, rank: u32, li: usize, step: []const u8, err: anyerror) !u8 {
    try w.print("{{\"rank\": {d}, \"layer\": {d}, \"step\": \"{s}\", \"error\": \"{s}\"}}\n", .{ rank, li, step, @errorName(err) });
    try w.flush();
    return 1;
}
