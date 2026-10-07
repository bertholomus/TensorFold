//! tf-dsv41-layer MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT REC_DIR [--layers N] [--chunks C] [--check-from F]
//!   [--engram DIR --token-map FILE] [--dump DIR]: the Zig port's layer gate (M3).
//! WORLD 4: the four nodes of the exact 2D split (prompt2d.zig), RANK the node, the links on PORT .. PORT + 2.
//! Both ranks run the recorded prompt's first C chunks (each Model.forward of the layer fixtures, in order; the
//! compressed caches and the window rings carried from chunk to chunk) through the port's prompt forward (prompt.zig)
//! on their GPUs, at most N layers a chunk, with the recording's bounded replay (replay = the prompt's length - the
//! window: the decoder's first layer cuts a chunk to its rows from there, an encoder-only chunk ends at it), and from
//! chunk F on compare every point a layer exchanges with the served build's bytes: the attention's input rows and
//! partial, its gather, the MoE's input rows and partial, its gather, on an Engram layer its rows' projection and their
//! gather; after a chunk's last layer the DSpark taps, the head's columns, their gather and the prompt's logits.
//! One JSON line a point; the first difference stops the run (its first differing element and how many differ).
//! REC_DIR: aot/ (the recording's Triton set), cubins/ (linear.cubin, linear_grouped.cubin, experts.cubin,
//! experts_cb.cubin: the served extension cubins), rope-{plain,compressed}-{cos,sin}.f32, rope.json and engram.json (zrec_fixtures.py), rank<R>/layers.jsonl and
//! layers/. Engram layers need the original Engram tables (DIR) and the compressed token map (the lane's JSON cache).
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const prompt = dsv41.prompt;

const usage = "usage: tf-dsv41-layer MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT REC_DIR [--layers N] [--chunks C] [--check-from F] [--engram DIR --token-map FILE] [--dump DIR]\n";
const rope_rows = 8192; // the plain table's first rows (chunk 0's positions and more)
const chunk_rows = 2048;
const cache_tokens = 4096; // the compressed caches' positions (the recorded prompt's)
const max_chunks = 8;

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
        if (bytes > r.host.len) r.host = try r.a.alloc(u8, bytes); // the arena: no free
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
    var chunks: usize = 1;
    var check_from: usize = 0;
    var engram_dir: ?[]const u8 = null;
    var token_map: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var ai: usize = 8;
    while (ai + 1 < args.len) : (ai += 2) {
        const key = args[ai];
        const val = args[ai + 1];
        if (std.mem.eql(u8, key, "--layers")) {
            layers = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--chunks")) {
            chunks = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--check-from")) {
            check_from = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--engram")) {
            engram_dir = val;
        } else if (std.mem.eql(u8, key, "--token-map")) {
            token_map = val;
        } else if (std.mem.eql(u8, key, "--dump")) {
            dump = val;
        } else return error.BadArgument;
    }

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
    // the prompt's chunks: each Model.forward's ids, and the layers the recording ran on it
    var chunk_call: [max_chunks + 1]u64 = undefined;
    var chunk_ids: [max_chunks][]const i64 = undefined;
    var n_chunks: usize = 0;
    for (fx.points) |p| {
        if (n_chunks == chunks or n_chunks == max_chunks) break;
        if (!std.mem.eql(u8, p.where, "Model.forward") or !std.mem.eql(u8, p.arg, "in2") or p.shape.len != 1) continue;
        const ids_bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rank_dir, "layers", p.file orelse return error.NoChunkFixture }), a, .limited(1 << 20), .@"8", null);
        chunk_call[n_chunks] = p.call;
        chunk_ids[n_chunks] = std.mem.bytesAsSlice(i64, ids_bytes);
        if (chunk_ids[n_chunks].len > chunk_rows) return error.ChunkTooLong;
        n_chunks += 1;
    }
    if (n_chunks < chunks) return error.NoChunkFixture;
    chunk_call[n_chunks] = std.math.maxInt(u64);
    var total: usize = 0;
    for (chunk_ids[0..n_chunks]) |ids| total += ids.len;
    // the recorded prompt's length (every chunk's rows): the bounded replay starts a window before its end
    var prompt_len: usize = 0;
    for (fx.points) |p| {
        if (std.mem.eql(u8, p.where, "Model.forward") and std.mem.eql(u8, p.arg, "in2") and p.shape.len == 1) prompt_len += @intCast(p.shape[0]);
    }
    const replay: usize = prompt_len -| cfg.window;

    // the weights (no DSpark blocks) from the lane's rank cache
    const t0 = std.Io.Timestamp.now(io, .awake);
    var cache_dir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
    defer cache_dir.close(io);
    const cache_name = try dsv41.rank_cache.find(a, io, cache_dir, rank, world);
    var ix = try dsv41.rank_cache.open(gpa, io, cache_dir, cache_name);
    defer ix.deinit();
    var cache_file = try cache_dir.openFile(io, cache_name, .{});
    defer cache_file.close(io);
    const sp = try dsv41.prompt2d.split(rank, world); // world 4: the 2D split (prompt2d.zig)
    var w = try dsv41.weights.load(gpa, &driver, .{ .cache = .{ .file = cache_file, .index = &ix, .io = io } }, cfg, sp, false);
    defer w.deinit();
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;

    var fan = try dsv41.prompt2d.Fan.open(io, ip, port, rank, world);
    defer fan.close();
    var comm = try fan.comm(rank, world);
    defer comm.deinit();

    // kernels: the recording's Triton set, the served extension cubins, the torch-op images
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, try std.fs.path.join(a, &.{ rec, "aot" }));
    defer set.deinit();
    const lin = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "linear.cubin" }), a, .limited(1 << 28), .@"16", null);
    var pf = try dsv41.exl3_prefill.Kernels.load(&driver, lin);
    defer pf.unload();
    const lgc = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "linear_grouped.cubin" }), a, .limited(1 << 28), .@"16", null);
    var lg = try dsv41.exl3_linear.Kernels.load(&driver, lgc);
    defer lg.unload();
    const exb = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "experts.cubin" }), a, .limited(1 << 28), .@"16", null);
    const exc = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "experts_cb.cubin" }), a, .limited(1 << 28), .@"16", null);
    var exk = try dsv41.exl3_experts.Kernels.load(&driver, exb, exc, dsv41.exl3_linear.codebook_mul1);
    defer exk.unload();
    if (!cuda.kernels.available) return error.NoKernelImages;
    var ops = try dsv41.ops.Ops.load(&driver, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();
    var ex = try dsv41.exact.Exact.load(&driver, cuda.kernels.dsv41_torch);
    defer ex.unload();

    // the plain RoPE table's first rows
    var rope_buf = try cuda.DeviceBuffer.alloc(&driver, 4 * rope_rows * (cfg.rope_dim / 2) * 4);
    defer rope_buf.free();
    for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
        var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ rec, name }), .{});
        defer f.close(io);
        const part = try a.alloc(u8, rope_rows * (cfg.rope_dim / 2) * 4);
        if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable;
        try rope_buf.upload(j * part.len, part);
    }
    const tbl = rope_rows * (cfg.rope_dim / 2) * 4;
    const rope: prompt.Rope = .{ .cos = rope_buf.ptr, .sin = rope_buf.ptr + tbl };
    const rope_c: prompt.Rope = .{ .cos = rope_buf.ptr + 2 * tbl, .sin = rope_buf.ptr + 3 * tbl };

    var arena = try prompt.Arena.init(&driver, 3 << 30);
    defer arena.deinit();
    var blas_ws_ptr: u64 = undefined;
    var two: ?dsv41.prompt2d.Two = if (sp.pair != null) try dsv41.prompt2d.Two.init(&cfg, &w, &arena, sp, chunk_rows) else null;
    var eng: prompt.Engine = .{ .d = &driver, .s = stream, .t = .{ .set = &set, .stream = stream }, .blas = undefined, .comm = &comm, .pf = &pf, .lin = &lg, .ex = &exk, .ops = &ops, .exact = &ex, .c = &cfg, .w = &w, .world = sp.world, .plain = rope, .compressed = rope_c, .two = if (two) |*t| t else null };
    var ch = try prompt.Chunk.init(&eng, &arena, chunk_rows, cache_tokens);
    const caches = try prompt.Caches.init(&eng, &arena, cache_tokens);
    blas_ws_ptr = ch.blas_ws;
    var blas = try dsv41.cublas.Blas.open(stream, blas_ws_ptr);
    defer blas.close();
    eng.blas = &blas;
    const rings = try a.alloc(u64, cfg.layers);
    for (rings) |*r| {
        r.* = try arena.take(eng.ringBytes());
        try driver.check(driver.api.cuMemsetD8Async(r.*, 0, eng.ringBytes(), stream.handle), "cuMemsetD8Async");
    }
    try stream.synchronize();
    try driver.check(driver.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too

    try w_out.print("{{\"rank\": {d}, \"loaded_s\": {d:.1}, \"weights_bytes\": {d}, \"chunks\": {d}, \"rows\": {d}, \"arena_bytes\": {d}}}\n", .{ rank, load_s, w.bytes, n_chunks, total, arena.used });
    try w_out.flush();

    // Engram on the host: the compressed token map, the recording's multipliers, the tables and 64 readers
    var eh: ?prompt.EngramHost = null;
    var tables: dsv41.engram_io.Tables = undefined;
    var pool: ?*dsv41.engram_io.Pool = null;
    defer if (pool) |p| p.deinit(gpa);
    defer if (eh != null) tables.close();
    if (engram_dir) |edir| {
        const map_text = try std.Io.Dir.cwd().readFileAlloc(io, token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
        const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
        const EngramJson = struct { multipliers: [][]i64 };
        const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ rec, "engram.json" }), a, .limited(1 << 26));
        const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
        var mult: [dsv41.engram.max_layers][dsv41.engram.max_ngram]i64 = @splat(@splat(0));
        for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
            mult[l][k] = v;
        };
        tables = try dsv41.engram_io.Tables.open(gpa, io, edir);
        pool = try dsv41.engram_io.Pool.init(gpa, io, 64);
        eh = try prompt.EngramHost.init(a, &cfg, dsv41.engram.Hasher.init(cfg, map, mult), &tables, pool.?, sp.rank, sp.world, chunk_rows);
    }
    // the whole sequence's ids (Engram hashes n-grams across chunk boundaries)
    const seq = try a.alloc(i32, total);
    {
        var at: usize = 0;
        for (chunk_ids[0..n_chunks]) |ids| for (ids) |id| {
            seq[at] = @intCast(id);
            at += 1;
        };
    }

    var run: Run = .{ .a = a, .io = io, .d = &driver, .s = stream, .fx = &fx, .rank = rank, .out = w_out, .host = try a.alloc(u8, 2 * chunk_rows * cfg.hidden * 4) };
    const host_pos = try a.alloc(i64, chunk_rows);
    var start: usize = 0;
    var ran: usize = 0;
    for (chunk_ids[0..n_chunks], 0..) |ids, ci| {
        // the layers the recording ran on this chunk (its attention points before the next chunk)
        var rec_layers: usize = 0;
        for (fx.points) |p| {
            if (p.call <= chunk_call[ci] or p.call >= chunk_call[ci + 1] or !std.mem.eql(u8, p.where, "Model.attention_k")) continue;
            if (p.layer) |l| rec_layers = @max(rec_layers, @as(usize, @intCast(l)) + 1);
        }
        const n_layers = @min(layers, cfg.layers);
        try w_out.print("{{\"rank\": {d}, \"chunk\": {d}, \"call\": {d}, \"start\": {d}, \"rows\": {d}, \"replay\": {d}, \"layers\": {d}, \"recorded_layers\": {d}, \"checked\": {}}}\n", .{ rank, ci, chunk_call[ci], start, ids.len, replay, n_layers, rec_layers, ci >= check_from });
        try w_out.flush();
        var shared: prompt.Shared = .{};
        try prompt.begin(&eng, &ch, ids, start, host_pos);
        if (!try runChunk(&run, &eng, &ch, &caches, &shared, rings, if (eh) |*x| x else null, seq[0 .. start + ids.len], chunk_call[ci], n_layers, ci >= check_from, dump, replay, host_pos)) {
            run.ok = false;
            break;
        }
        start += ids.len;
        ran += 1;
    }
    try w_out.print("{{\"rank\": {d}, \"chunks\": {d}, \"layers\": {d}, \"all_equal\": {}}}\n", .{ rank, ran, layers, run.ok });
    try w_out.flush();
    return if (run.ok) 0 else 1;
}

/// One chunk's layers through the prompt forward, each exchange checked against the fixtures after the chunk's call
/// when `checked` (else only run); at the decoder's first layer the replay cut (an encoder-only chunk ends there); a
/// chunk through every layer ends with its taps and the head, checked too. False on the first difference or a step's
/// error (reported).
fn runChunk(run: *Run, eng: *const prompt.Engine, ch: *prompt.Chunk, caches: *const prompt.Caches, shared: *prompt.Shared, rings: []const u64, eh: ?*prompt.EngramHost, seq: []const i32, call: u64, layers: usize, checked: bool, dump: ?[]const u8, replay: usize, host_pos: []i64) !bool {
    const fx = run.fx;
    const w_out = run.out;
    const rank = run.rank;
    const c = eng.c;
    var after = call;
    var floor: usize = 0;
    var kv_done: ?usize = null;
    var label_buf: [64]u8 = undefined;
    for (0..layers) |li| {
        const l: i64 = @intCast(li);
        if (li == c.layers / 2) {
            const go_on = prompt.replayCut(eng, ch, caches, shared, li, replay, host_pos) catch |err| return report(w_out, rank, li, "replay cut", err);
            try w_out.print("{{\"rank\": {d}, \"layer\": {d}, \"replay\": {d}, \"rows\": {d}, \"start\": {d}, \"encoder_only\": {}}}\n", .{ rank, li, replay, ch.n, ch.start, !go_on });
            try w_out.flush();
            if (!go_on) break;
            floor = replay;
            kv_done = li;
        }
        if (eng.w.layers[li].engram_wkv != null) {
            const e_h = eh orelse return report(w_out, rank, li, "engram", error.NoEngramTables);
            prompt.engramApply(eng, ch, e_h, li, seq) catch |err| return report(w_out, rank, li, "engram", err);
            if (dump) |dir| dumpEngram(run.a, run.io, dir, li, rank, e_h, ch.n) catch |err| {
                try w_out.print("{{\"rank\": {d}, \"dump\": \"{s}\"}}\n", .{ rank, @errorName(err) });
                try w_out.flush();
            };
            const eg = fx.find(after, "Comm.gather", null, "in1") orelse return report(w_out, rank, li, "engram fixture", error.NoLayerFixture);
            if (checked) {
                if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} engram projection", .{li}), eg, ch.ek)) return false;
                if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} engram gather", .{li}), fx.find(after, "Comm.gather", null, "out"), ch.ekg)) return false;
            }
            after = eg.call;
        }
        // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
        if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| {
            prompt.tap(eng, ch, j) catch |err| return report(w_out, rank, li, "tap", err);
        }
        const att = fx.find(after, "Model.attention_k", l, "in2") orelse return report(w_out, rank, li, "attention fixture", error.NoLayerFixture);
        prompt.attnMixes(eng, ch, li) catch |err| return report(w_out, rank, li, "attention mixes", err);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention in", .{li}), att, ch.x)) return false;
        prompt.attention(eng, ch, caches, shared, li, rings[li], floor, kv_done != null and kv_done.? == li) catch |err| return report(w_out, rank, li, "attention", err);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention partial", .{li}), fx.find(after, "Model.attention_k", l, "out"), ch.pa)) return false;
        try prompt.gather(eng, ch, ch.pa, ch.ga);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention gather", .{li}), fx.find(att.call, "Comm.gather", null, "out"), ch.ga)) return false;
        prompt.ffnMixes(eng, ch, li) catch |err| return report(w_out, rank, li, "ffn mixes", err);
        const mo = fx.find(att.call, "Model.moe", l, "in2") orelse return report(w_out, rank, li, "moe fixture", error.NoLayerFixture);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe in", .{li}), mo, ch.x)) return false;
        prompt.moe(eng, ch, li) catch |err| return report(w_out, rank, li, "moe", err);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe partial", .{li}), fx.find(att.call, "Model.moe", l, "out"), ch.pm)) return false;
        try prompt.gather(eng, ch, ch.pm, ch.gm);
        const mg = fx.find(mo.call, "Comm.gather", null, "out") orelse return report(w_out, rank, li, "moe gather fixture", error.NoLayerFixture);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe gather", .{li}), mg, ch.gm)) return false;
        prompt.endLayer(ch);
        after = mg.call;
        if (li + 1 < c.layers) continue;
        // the chunk's end: the taps (Model.forward's list), the head on the last row, the prompt's logits
        for (0..c.dspark_taps.slice().len) |j| {
            var arg_buf: [32]u8 = undefined;
            const arg = try std.fmt.bufPrint(&arg_buf, "post.in.taps.{d}", .{j});
            if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "tap {d}", .{j}), fx.find(call - 1, "Model.forward", null, arg), prompt.tapRows(eng, ch, j))) return false;
        }
        prompt.head(eng, ch) catch |err| return report(w_out, rank, li, "head", err);
        const hg = fx.find(after, "Comm.gather", null, "in1") orelse return report(w_out, rank, li, "head fixture", error.NoLayerFixture);
        if (checked) {
            if (!try run.check("head columns", hg, ch.head_l)) return false;
            if (!try run.check("head gather", fx.find(after, "Comm.gather", null, "out"), ch.head_g)) return false;
            if (!try run.check("prompt logits", fx.find(call - 1, "Model.forward", null, "out"), ch.head_g)) return false;
        }
    }
    try run.s.synchronize();
    return true;
}

/// The Engram rows' ids and their bf16 rows of layer li, for the served Engram.rows (zrec_engram_ref.py).
fn dumpEngram(a: std.mem.Allocator, io: std.Io, dir: []const u8, li: usize, rank: u32, eh: *const prompt.EngramHost, n: usize) !void {
    const m = n * (eh.hi - eh.lo);
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    for ([_][]const u8{ "ids.i64", "rows.bf16" }, [_][]const u8{ std.mem.sliceAsBytes(eh.flat[0..m]), std.mem.sliceAsBytes(eh.rows[0 .. m * t.row_w]) }) |ext, bytes| {
        const path = try std.fmt.allocPrint(a, "{s}/engram-L{d}-r{d}.{s}", .{ dir, li, rank, ext });
        var f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        try f.writePositionalAll(io, bytes, 0);
    }
}

fn report(w: *std.Io.Writer, rank: u32, li: usize, step: []const u8, err: anyerror) !bool {
    try w.print("{{\"rank\": {d}, \"layer\": {d}, \"step\": \"{s}\", \"error\": \"{s}\"}}\n", .{ rank, li, step, @errorName(err) });
    try w.flush();
    return false;
}
