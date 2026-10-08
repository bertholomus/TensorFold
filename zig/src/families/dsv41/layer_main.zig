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
//! With --rounds K the recorded decode rounds follow (eager rounds: RoundDecoder.run's rows, the stream in pool slot 0;
//! serial, or a drafting recording's windows of the pending token and its drafts): the first K of the prompt's request
//! through round.zig, each checked at every layer's exchanges (Engram's projection and gather, the attention's input
//! rows, partial and gather, the MoE's), the head's columns and gather, the logits and the taps, and the served
//! acceptance (the rows' greedy tokens against the drafts the next round kept and its first row). With --drafts B (a
//! drafting recording; B the request's max_tokens) the drafter runs as served around them (draft.zig): the prompt's taps
//! absorbed into its rings, then before each round the batched pass (every recorded drafter point: each stage's
//! attention rows, ring plane, partial and gather, the MoE's, the stages' streams, the head's columns, the Markov loop's
//! drafts) and the served depth from its confidences (_choose_k) against the round's window, after it the window's taps
//! absorbed.
//! With --light 1 (a token recording: TF_ZREC_ONLY=Model.forward,RoundDecoder.run) every recorded request instead: its
//! prompt's chunks through every layer unchecked, the prompt's logits checked, then its rounds (at most K with
//! --rounds K, all without), each round's logits checked and its greedy token against the recorded next one; the
//! caches and rings cleared between requests.
//! One JSON line a point; the first difference stops the run (its first differing element and how many differ).
//! REC_DIR: aot/ (the recording's Triton set), cubins/ (linear.cubin, linear_grouped.cubin, experts.cubin,
//! experts_cb.cubin: the served extension cubins), rope-{plain,compressed}-{cos,sin}.f32, rope.json and engram.json (zrec_fixtures.py), rank<R>/layers.jsonl and
//! layers/. Engram layers need the original Engram tables (DIR) and the compressed token map (the lane's JSON cache).
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const prompt = dsv41.prompt;
const round_mod = dsv41.round;
const draft_mod = dsv41.draft;
const sampling = dsv41.sampling;

const usage = "usage: tf-dsv41-layer MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT REC_DIR [--layers N] [--chunks C] [--check-from F] [--rounds K] [--drafts B] [--engram DIR --token-map FILE] [--dump DIR]\n";
const chunk_rows = 2048;
const cache_tokens = 4096; // the compressed caches' positions at least (the recorded prompt's)
/// The served pool's window (--context): the bucket rule's cap (graph.py bucket_for).
const pool_window = 1 << 20;
const max_chunks = 8;
/// --side 1: the rounds' mixes' side work on a stream of its own (round.Round.useSide).
var side_on = false;
/// --prefetch 1: the rounds' paced L2 prefetch (round.Round.usePrefetch).
var prefetch_on = false;

/// One fixture line of layers.jsonl.
const Point = struct {
    call: u64,
    where: []const u8,
    layer: ?i64 = null,
    arg: []const u8,
    shape: []const i64,
    dtype: []const u8,
    sha256: ?[]const u8 = null,
    head: ?[]const u8 = null, // the first 64 bytes, hex
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
    row: []f32 = &.{}, // a logits row on the host (argmax, the sampler)
    ok: bool = true,

    /// The host row buffer for n fp32 values (made once, grown when wider).
    fn rowBuf(r: *Run, n: usize) ![]f32 {
        if (r.row.len < n) r.row = try r.a.alloc(f32, n); // the arena: no free
        return r.row[0..n];
    }

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

    /// Our buffer `dev` against fixture point `p` (saved whole) on the listed rows only: a pool plane whose other rows
    /// the step does not read (the served pool's rows hold its warm-up's keys).
    fn checkRows(r: *Run, label: []const u8, p: ?*const Point, dev: u64, rows: []const usize) !bool {
        const pt = p orelse {
            try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"error\": \"no fixture\"}}\n", .{ r.rank, label });
            try r.out.flush();
            r.ok = false;
            return false;
        };
        const name = pt.file orelse return error.NotSavedWhole;
        try r.s.synchronize();
        const bytes = numel(pt.shape) * dtypeBytes(pt.dtype);
        const row_bytes = numel(pt.shape[1..]) * dtypeBytes(pt.dtype);
        if (bytes > r.host.len) r.host = try r.a.alloc(u8, bytes); // the arena: no free
        try r.d.check(r.d.api.cuMemcpyDtoH_v2(r.host.ptr, dev, bytes), "cuMemcpyDtoH");
        const want = try std.Io.Dir.cwd().readFileAlloc(r.io, try std.fs.path.join(r.a, &.{ r.fx.dir, "layers", name }), r.a, .limited(1 << 31));
        defer r.a.free(want);
        if (want.len != bytes) return error.SavedSizeDiffers;
        var differ: usize = 0;
        var first: ?usize = null;
        for (rows) |row| {
            const o = row * row_bytes;
            if (o + row_bytes > bytes) return error.RowOutOfRange;
            if (!std.mem.eql(u8, r.host[o..][0..row_bytes], want[o..][0..row_bytes])) {
                differ += 1;
                if (first == null) first = row;
            }
        }
        const same = differ == 0;
        try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"call\": {d}, \"rows\": {d}, \"equal\": {}", .{ r.rank, label, pt.call, rows.len, same });
        if (first) |f| try r.out.print(", \"first_row\": {d}, \"rows_differ\": {d}", .{ f, differ });
        try r.out.print("}}\n", .{});
        try r.out.flush();
        if (!same) r.ok = false;
        return same;
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
    var rounds: usize = 0;
    var budgets: []usize = &.{}; // --drafts: each drafting request's max_tokens, in admission order (none: no drafter)
    var seeds: []u64 = &.{}; // --seeds: each request's sampling seed, in admission order (none: greedy)
    var smp: sampling.Sampling = .{ .seed = 0 }; // --temperature, --top-k, --top-p: the requests' sampling
    var light = false;
    var use_graphs = false; // --graphs 1: a light run's rounds as captured stretches (round.Graphs)
    var engram_dir: ?[]const u8 = null;
    var token_map: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var rdma_devices: ?[]const u8 = null; // 2D: the decode-size exchanges over RDMA rings (ring2d.zig) on these devices
    var rdma_kernels: ?[]const u8 = null; // their kernels' image (rdma_gather.cu's fatbin)
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
        } else if (std.mem.eql(u8, key, "--rounds")) {
            rounds = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--drafts")) {
            var list: std.ArrayList(usize) = .empty;
            var it = std.mem.splitScalar(u8, val, ',');
            while (it.next()) |x| try list.append(a, try std.fmt.parseInt(usize, x, 10));
            budgets = list.items;
        } else if (std.mem.eql(u8, key, "--seeds")) {
            var list: std.ArrayList(u64) = .empty;
            var it = std.mem.splitScalar(u8, val, ',');
            while (it.next()) |x| try list.append(a, try std.fmt.parseInt(u64, x, 10));
            seeds = list.items;
        } else if (std.mem.eql(u8, key, "--temperature")) {
            smp.temperature = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, key, "--top-k")) {
            smp.top_k = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--top-p")) {
            smp.top_p = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, key, "--prefetch")) {
            prefetch_on = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--side")) {
            side_on = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--graphs")) {
            use_graphs = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--light")) {
            light = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--engram")) {
            engram_dir = val;
        } else if (std.mem.eql(u8, key, "--token-map")) {
            token_map = val;
        } else if (std.mem.eql(u8, key, "--dump")) {
            dump = val;
        } else if (std.mem.eql(u8, key, "--rdma")) {
            rdma_devices = val;
        } else if (std.mem.eql(u8, key, "--rdma-kernels")) {
            rdma_kernels = val;
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
    // the recorded prompt's length (its chunks' rows: every chunk before the first round, a recording of several
    // requests holding later prompts after it): the bounded replay starts a window before its end
    var prompt_len: usize = 0;
    for (fx.points) |p| {
        if (std.mem.eql(u8, p.where, "RoundDecoder.run")) break;
        if (std.mem.eql(u8, p.where, "Model.forward") and std.mem.eql(u8, p.arg, "in2") and p.shape.len == 1) prompt_len += @intCast(p.shape[0]);
    }
    const replay: usize = prompt_len -| cfg.window;

    // the weights (the DSpark blocks with --drafts) from the lane's rank cache
    const t0 = std.Io.Timestamp.now(io, .awake);
    var cache_dir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
    defer cache_dir.close(io);
    const cache_name = try dsv41.rank_cache.find(a, io, cache_dir, rank, world);
    var ix = try dsv41.rank_cache.open(gpa, io, cache_dir, cache_name);
    defer ix.deinit();
    var cache_file = try cache_dir.openFile(io, cache_name, .{});
    defer cache_file.close(io);
    const sp = try dsv41.prompt2d.split(rank, world); // world 4: the 2D split (prompt2d.zig)
    var w = try dsv41.weights.load(gpa, &driver, .{ .cache = .{ .file = cache_file, .index = &ix, .io = io } }, cfg, sp, budgets.len > 0);
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

    // the caches' positions: the deepest round's bucket and the widest extent the recording reaches (every row's: a
    // concurrent round's rows are several streams'); the deepest position any row or prompt reaches
    var tokens: usize = cache_tokens;
    var deepest: usize = 0;
    var filled: usize = 0; // the current prompt's positions so far (its chunks, until a round)
    for (fx.points) |*q| {
        if (std.mem.eql(u8, q.where, "Model.forward") and std.mem.eql(u8, q.arg, "in2") and q.shape.len == 1) {
            filled += @intCast(q.shape[0]);
            deepest = @max(deepest, filled);
            continue;
        }
        if (!std.mem.eql(u8, q.where, "RoundDecoder.run")) continue;
        filled = 0;
        const in2 = std.mem.eql(u8, q.arg, "in2");
        if (!in2 and !std.mem.eql(u8, q.arg, "in5")) continue;
        var most: i64 = headInt(q) catch 0;
        if (q.file) |name| {
            const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rank_dir, "layers", name }), a, .limited(1 << 20), .@"8", null);
            for (std.mem.bytesAsSlice(i64, bytes)) |v| most = @max(most, v);
        }
        const m: usize = @intCast(@max(most, 0));
        if (in2) deepest = @max(deepest, m + round_mod.max_rows);
        tokens = @max(tokens, if (in2) round_mod.bucketFor(m + 1, pool_window) else m);
    }
    // the RoPE tables' rows every position reaches (the fixtures hold the served tables' first rows)
    const rope_rows = @max(@min(tokens, deepest + 1), chunk_rows);
    var rope_buf = try cuda.DeviceBuffer.alloc(&driver, 4 * rope_rows * (cfg.rope_dim / 2) * 4);
    defer rope_buf.free();
    for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
        var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ rec, name }), .{});
        defer f.close(io);
        const part = try a.alloc(u8, rope_rows * (cfg.rope_dim / 2) * 4);
        if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // (a fixture of fewer rows)
        try rope_buf.upload(j * part.len, part);
        a.free(part);
    }
    const tbl = rope_rows * (cfg.rope_dim / 2) * 4;
    const rope: prompt.Rope = .{ .cos = rope_buf.ptr, .sin = rope_buf.ptr + tbl };
    const rope_c: prompt.Rope = .{ .cos = rope_buf.ptr + 2 * tbl, .sin = rope_buf.ptr + 3 * tbl };

    var arena = try prompt.Arena.init(&driver, 5 << 30); // (a 1M pool's caches, the candidate pool's buffers)
    defer arena.deinit();
    var blas_ws_ptr: u64 = undefined;
    var two: ?dsv41.prompt2d.Two = if (sp.pair != null) try dsv41.prompt2d.Two.init(&cfg, &w, &arena, sp, chunk_rows) else null;
    // 2D with --rdma: the rings (their infos go round over NCCL world 4)
    var rdma_mod: ?cuda.Module = null;
    defer if (rdma_mod) |*m| m.unload();
    var rdma_rings: ?dsv41.ring2d.Rings = null;
    defer if (rdma_rings) |*r| r.close(gpa);
    if (two != null and rdma_devices != null) {
        const image = try std.Io.Dir.cwd().readFileAllocOptions(io, rdma_kernels orelse return error.NoRdmaKernels, a, .limited(1 << 26), .@"16", null);
        rdma_mod = try cuda.Module.load(&driver, image);
        var devices: std.ArrayList([]const u8) = .empty;
        var dit = std.mem.splitScalar(u8, rdma_devices.?, ',');
        while (dit.next()) |d| try devices.append(a, d);
        rdma_rings = try dsv41.ring2d.Rings.open(gpa, &driver, try dsv41.rdma.Kernels.load(rdma_mod.?), devices.items, rank, .{ .max_bytes = 4456448, .gid_index = 5 }, &comm, stream);
        two.?.rings = &rdma_rings.?;
    }
    var eng: prompt.Engine = .{ .d = &driver, .s = stream, .t = .{ .set = &set, .stream = stream }, .blas = undefined, .comm = &comm, .pf = &pf, .lin = &lg, .ex = &exk, .ops = &ops, .exact = &ex, .c = &cfg, .w = &w, .world = sp.world, .plain = rope, .compressed = rope_c, .two = if (two) |*t| t else null };
    // TP2 with --rdma: the decode gathers over the RDMA ring (the served lane's), their bytes checked as NCCL's were
    var tp2_mod: ?cuda.Module = null;
    defer if (tp2_mod) |*mm| mm.unload();
    var tp2_ring: ?*dsv41.rdma.Ring = null;
    defer if (tp2_ring) |r| r.destroy(gpa);
    if (two == null and rdma_devices != null) {
        tp2_mod = try cuda.Module.load(&driver, cuda.kernels.dsv41_rdma);
        var devices: std.ArrayList([]const u8) = .empty;
        var dit = std.mem.splitScalar(u8, rdma_devices.?, ',');
        while (dit.next()) |dv| try devices.append(a, dv);
        tp2_ring = try dsv41.model.Model.openRing(gpa, &driver, try dsv41.rdma.Kernels.load(tp2_mod.?), devices.items, rank, &comm, stream);
        eng.ring = tp2_ring;
    }
    var ch = try prompt.Chunk.init(&eng, &arena, chunk_rows, tokens);
    // a drafting recording's pool: MultiDecoder's slots (each stream's window rings and positional stores)
    const slots: usize = if (budgets.len > 0) draft_mod.max_streams else 1;
    const caches = try prompt.Caches.init(&eng, &arena, tokens, slots);
    blas_ws_ptr = ch.blas_ws;
    var blas = try dsv41.cublas.Blas.open(stream, blas_ws_ptr);
    defer blas.close();
    eng.blas = &blas;
    const rings = try a.alloc(u64, cfg.layers);
    for (rings) |*r| {
        r.* = try arena.take(slots * eng.ringBytes());
        try driver.check(driver.api.cuMemsetD8Async(r.*, 0, slots * eng.ringBytes(), stream.handle), "cuMemsetD8Async");
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
    const seq = try a.alloc(i32, total + rounds * round_mod.max_rows);
    {
        var at: usize = 0;
        for (chunk_ids[0..n_chunks]) |ids| for (ids) |id| {
            seq[at] = @intCast(id);
            at += 1;
        };
    }

    var run: Run = .{ .a = a, .io = io, .d = &driver, .s = stream, .fx = &fx, .rank = rank, .out = w_out, .host = try a.alloc(u8, 2 * chunk_rows * cfg.hidden * 4) };
    if (light and budgets.len > 0) {
        const ok = try runDrafted(&run, &eng, &ch, &caches, rings, if (eh) |*x| x else null, a, &arena, budgets, seeds, smp, sp);
        try w_out.print("{{\"rank\": {d}, \"light\": true, \"drafted\": true, \"all_equal\": {}}}\n", .{ rank, ok });
        try w_out.flush();
        return if (ok) 0 else 1;
    }
    if (light) {
        var gs = round_mod.Graphs.init(gpa);
        defer gs.deinit();
        const ok = try runLight(&run, &eng, &ch, &caches, rings, if (eh) |*x| x else null, a, &arena, rounds, tokens, if (use_graphs) &gs else null);
        if (use_graphs) {
            try w_out.print("{{\"rank\": {d}, \"graphs\": {d}}}\n", .{ rank, gs.count() });
            try w_out.flush();
        }
        try w_out.print("{{\"rank\": {d}, \"light\": true, \"all_equal\": {}}}\n", .{ rank, ok });
        try w_out.flush();
        return if (ok) 0 else 1;
    }
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
    if (rounds > 0 and run.ok and ran == n_chunks) {
        if (n_chunks != chunks or start != prompt_len) return error.RoundsNeedTheWholePrompt;
        run.ok = try runRounds(&run, &eng, &ch, &caches, rings, if (eh) |*x| x else null, seq, start, rounds, a, &arena, if (budgets.len > 0) budgets[0] else 0, sp);
    }
    try w_out.print("{{\"rank\": {d}, \"chunks\": {d}, \"layers\": {d}, \"rounds\": {d}, \"all_equal\": {}}}\n", .{ rank, ran, layers, rounds, run.ok });
    try w_out.flush();
    return if (run.ok) 0 else 1;
}

/// The recorded rounds' checker: each point found after the last one matched, as the recorder numbered the calls.
const RoundCheck = struct {
    run: *Run,
    after: u64,
    att: u64 = 0,
    moe: u64 = 0,
    call: u64, // the round's RoundDecoder.run
    fwd: u64, // its RoundRunner.forward

    fn at(ctx: *anyopaque, what: round_mod.Point, layer: usize, dev: u64) anyerror!bool {
        const rc: *RoundCheck = @ptrCast(@alignCast(ctx));
        const fx = rc.run.fx;
        const l: i64 = @intCast(layer);
        var buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&buf, "R{d} L{d} {s}", .{ rc.call, layer, @tagName(what) });
        const pt: ?*const Point = switch (what) {
            .engram_proj => fx.find(rc.after, "Comm.gather", null, "in1"),
            .engram_gather => fx.find(rc.after, "Comm.gather", null, "out"),
            .attn_in => fx.find(rc.after, "RoundDecoder._attention2", l, "in2"),
            .attn_out => fx.find(rc.after, "RoundDecoder._attention2", l, "out"),
            .attn_gather => fx.find(rc.att, "Comm.gather", null, "out"),
            .moe_in => fx.find(rc.att, "Model.moe", l, "in2"),
            .moe_out => fx.find(rc.att, "Model.moe", l, "out"),
            .moe_gather => fx.find(rc.moe, "Comm.gather", null, "out"),
            .head_cols => fx.find(rc.after, "Comm.gather", null, "in1"),
            .head_gather => fx.find(rc.after, "Comm.gather", null, "out"),
            .logits => fx.find(rc.call - 1, "RoundDecoder.run", null, "out"),
            .taps => fx.find(rc.fwd - 1, "RoundRunner.forward", null, "out.1"),
        };
        const ok = try rc.run.check(label, pt, dev);
        if (pt) |q| switch (what) {
            .engram_gather => rc.after = q.call,
            .attn_in => rc.att = q.call,
            .moe_in => rc.moe = q.call,
            .moe_gather => rc.after = q.call,
            else => {},
        };
        return ok;
    }
};

/// A fixture tensor of int64 values, whole (a round's input rows).
fn ints(run: *Run, p: ?*const Point) ![]const i64 {
    const pt = p orelse return error.NoRoundFixture;
    const name = pt.file orelse return error.NoRoundFixture;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(run.io, try std.fs.path.join(run.a, &.{ run.fx.dir, "layers", name }), run.a, .limited(1 << 20), .@"8", null);
    return std.mem.bytesAsSlice(i64, bytes);
}

/// The logits' width: the ranks' vocabulary parts (a 2D rank's part is both pairs' quarters).
fn vocabOf(eng: *const prompt.Engine) usize {
    return eng.world * (if (eng.two) |t| t.hw[0] + t.hw[1] else eng.w.head.n);
}

/// The index of the first largest of n fp32 values on the device (torch.argmax's choice).
fn argmax(run: *Run, dev: u64, n: usize) !usize {
    try run.s.synchronize();
    const host = try run.rowBuf(n);
    try run.d.check(run.d.api.cuMemcpyDtoH_v2(host.ptr, dev, n * 4), "cuMemcpyDtoH");
    var best: usize = 0;
    for (host, 0..) |v, i| {
        if (v > host[best]) best = i;
    }
    return best;
}

/// The recorded drafter pass's checker: its points after the pass's BatchDraftGraph.run call, stage j's carrying layer
/// `layers` + j, each gather the first after its stage's attention or MoE call.
const DraftCheck = struct {
    run: *Run,
    call: u64, // the pass's BatchDraftGraph.run
    layers: usize,
    after: u64,
    vis: []const usize, // the plane rows the pass reads: each stream's window positions in its slot
    att: u64 = 0,
    moe: u64 = 0,

    fn at(ctx: *anyopaque, what: draft_mod.Point, stage: usize, dev: u64) anyerror!bool {
        const dc: *DraftCheck = @ptrCast(@alignCast(ctx));
        const fx = dc.run.fx;
        const l: i64 = @intCast(dc.layers + stage);
        var buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&buf, "D{d} S{d} {s}", .{ dc.call, stage, @tagName(what) });
        const pt: ?*const Point = switch (what) {
            .attn_in => fx.find(dc.after, "BatchDraftGraph._attention", l, "in2"),
            .attn_ring => fx.find(dc.after, "BatchDraftGraph._attention", l, "in3"),
            .attn_out => fx.find(dc.after, "BatchDraftGraph._attention", l, "out"),
            .attn_gather => fx.find(dc.att, "Comm.gather", null, "out"),
            .moe_in => fx.find(dc.att, "Model.moe", l, "in2"),
            .moe_out => fx.find(dc.att, "Model.moe", l, "out"),
            .moe_gather => fx.find(dc.moe, "Comm.gather", null, "out"),
            .stages_h => fx.find(dc.call, "BatchDraftGraph._stages_fused", null, "out.0"),
            .stages_pre => fx.find(dc.call, "BatchDraftGraph._stages_fused", null, "out.1"),
            .local => fx.find(dc.call, "Markov.steps", null, "in1"),
            .markov => fx.find(dc.call, "Markov.steps", null, "post.in2"),
        };
        const ok = if (what == .attn_ring) try dc.run.checkRows(label, pt, dev, dc.vis) else try dc.run.check(label, pt, dev);
        if (pt) |q| switch (what) {
            .attn_in => dc.att = q.call,
            .moe_in => dc.moe = q.call,
            .moe_gather => dc.after = q.call,
            else => {},
        };
        return ok;
    }
};

/// sample_rows of one row of n fp32 logits on the device at absolute position `position` (sampling.zig).
fn sampleDev(run: *Run, dev: u64, n: usize, position: usize, s: sampling.Sampling) !i64 {
    if (s.temperature <= 0) return @intCast(try argmax(run, dev, n));
    try run.s.synchronize();
    const host = try run.rowBuf(n);
    try run.d.check(run.d.api.cuMemcpyDtoH_v2(host.ptr, dev, n * 4), "cuMemcpyDtoH");
    return sampling.sampleRow(host, position, s);
}

/// The recorded decode rounds after the prompt: each RoundDecoder.run's rows (ids, positions, the stream's extent) through
/// round.zig with every point checked. A round's rows are its window: the pending token and the drafts it verifies (one
/// row a round when serial). The rows' greedy tokens (their targets) decide the next round as the served round keeps
/// them (multi.py _round): the drafts while each equals its target, then the target after them; the next round starts
/// at the kept position with that target, its rows overwriting the rejected ones (in the Engram sequence too). Only the
/// recorded prompt's request: a later prompt ends the rounds. `drafts` (the request's max_tokens; 0: none): the
/// drafter around each round, its pass checked at every recorded point, its drafts and served depth against the
/// round's window, the window's taps absorbed after it. False on the first difference.
fn runRounds(run: *Run, eng: *const prompt.Engine, ch: *prompt.Chunk, caches: *const prompt.Caches, rings: []const u64, eh: ?*prompt.EngramHost, seq: []i32, prompt_len: usize, rounds: usize, a: std.mem.Allocator, arena: *prompt.Arena, drafts: usize, sp: dsv41.plan.Split) !bool {
    const fx = run.fx;
    const w_out = run.out;
    const c = eng.c;
    const vocab = vocabOf(eng);
    const R_max = round_mod.max_rows;
    var rd = try round_mod.Round.init(eng, arena, a, ch.max_comp);
    if (side_on) try rd.useSide(eng);
    defer rd.dropSide(eng);
    if (prefetch_on) try rd.usePrefetch(eng);
    defer rd.dropPrefetch();
    var targets: [R_max]usize = undefined;
    targets[0] = try argmax(run, ch.head_g, vocab); // greedy from the prompt's logits
    var prev_ids: [R_max]i64 = undefined;
    var prev_n: usize = 1; // the prompt: its last row's target only
    var prev_pos: i64 = @as(i64, @intCast(prompt_len)) - 1;
    // from the prompt's last chunk (the latest Model.forward before the first round): a later one is another request's
    var after: u64 = 0;
    if (fx.find(0, "RoundDecoder.run", null, "in1")) |first| {
        for (fx.points) |*q| {
            if (q.call < first.call and isPoint(q, "Model.forward", "in2")) after = q.call;
        }
    }
    // the drafter: its pool (MultiDecoder's slots), the prompt's taps (the replayed rows, the tap layers' rows side by
    // side) absorbed into slot 0's rings from the taps' first position (Drafter.absorb after the prompt's last chunk)
    var pool: draft_mod.Pool = undefined;
    var dr: draft_mod.Drafter = undefined;
    if (drafts > 0) {
        pool = try draft_mod.Pool.init(eng, arena, draft_mod.max_streams);
        dr = try draft_mod.Drafter.init(eng, arena, sp);
        const d = c.hidden;
        for (0..c.dspark_taps.slice().len) |j| try eng.ops.copyRows(eng.s, prompt.tapRows(eng, ch, j), d * 2, dr.at + j * d * 2, dr.taps_w * 2, d * 2, ch.n);
        if (fx.find(after, "Drafter.absorb", null, "in3")) |ap| {
            if (!try run.check("prompt absorb taps", ap, dr.at)) return false;
        }
        dr.absorb(eng, ch, &pool, 0, dr.at, ch.n, ch.start) catch |err| {
            try w_out.print("{{\"rank\": {d}, \"absorb\": \"prompt\", \"error\": \"{s}\"}}\n", .{ run.rank, @errorName(err) });
            try w_out.flush();
            return false;
        };
    }
    for (0..rounds) |k| {
        const rp = fx.find(after, "RoundDecoder.run", null, "in1") orelse {
            try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"note\": \"no more recorded rounds\"}}\n", .{ run.rank, k });
            try w_out.flush();
            break;
        };
        if (fx.find(after, "Model.forward", null, "in2")) |mf| if (mf.call < rp.call) {
            try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"note\": \"a later request's prompt: its rounds are not this prompt's\"}}\n", .{ run.rank, k });
            try w_out.flush();
            break;
        };
        const ids = try ints(run, rp);
        const pos = try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in2"));
        const base = try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in4"));
        const end = try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in5"));
        if (ids.len == 0 or ids.len > R_max or pos.len != ids.len) return error.BadRound;
        // the previous round's acceptance: its kept drafts equal their targets, its first rejected one does not, and
        // this round's first row is the target after the kept ones
        const kept = pos[0] - prev_pos - 1;
        var accept_ok = kept >= 0 and kept < prev_n;
        if (accept_ok) {
            const ka: usize = @intCast(kept);
            for (0..ka) |j| {
                if (prev_ids[j + 1] != @as(i64, @intCast(targets[j]))) accept_ok = false;
            }
            if (ka + 1 < prev_n and prev_ids[ka + 1] == @as(i64, @intCast(targets[ka]))) accept_ok = false;
            if (ids[0] != @as(i64, @intCast(targets[ka]))) accept_ok = false;
        }
        try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"call\": {d}, \"rows\": {d}, \"pos\": {d}, \"id\": {d}, \"kept\": {d}, \"accept_equal\": {}}}\n", .{ run.rank, k, rp.call, ids.len, pos[0], ids[0], kept, accept_ok });
        try w_out.flush();
        if (!accept_ok) run.ok = false;
        if (drafts > 0) {
            // the round's drafter pass (its BatchDraftGraph.run, before the round's forward)
            const bp = fx.find(after, "BatchDraftGraph.run", null, "in1") orelse return error.NoDraftFixture;
            if (bp.call > rp.call) return error.NoDraftFixture;
            const tok = try ints(run, bp);
            const q0 = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, "in2"));
            const sl = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, "in3"));
            const most = 5; // the engine's drafts (--mtp-drafts 5)
            const steps = draft_mod.passSteps(tok.len, pool.slots, most, R_max, dr.n);
            // the plane rows the pass reads: each stream's window (its 128 positions before q0) in its slot
            var vis: [draft_mod.max_streams * 128]usize = undefined;
            var nv: usize = 0;
            for (q0, sl) |q, slot| {
                var p: i64 = @max(0, q - @as(i64, @intCast(c.window)));
                while (p < q) : (p += 1) {
                    vis[nv] = @as(usize, @intCast(slot)) * pool.ring + @as(usize, @intCast(p)) % pool.ring;
                    nv += 1;
                }
            }
            var dc: DraftCheck = .{ .run = run, .call = bp.call, .layers = c.layers, .after = bp.call, .vis = vis[0..nv] };
            dr.pass(eng, ch, &pool, tok, q0, sl, steps, .{ .ctx = &dc, .at = DraftCheck.at }) catch |err| {
                if (err == error.DraftMismatch) return false;
                try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"draft_call\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, k, bp.call, @errorName(err) });
                try w_out.flush();
                return false;
            };
            // its drafts against the recorded ones, the served depth from its confidences against the window's
            var arg_buf: [16]u8 = undefined;
            const want = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, try std.fmt.bufPrint(&arg_buf, "out.{d}", .{0})));
            var drafts_ok = want.len == steps;
            if (drafts_ok) for (0..steps) |j| {
                if (dr.drafts[0][j] != want[j]) drafts_ok = false;
            };
            const emitted: usize = @intCast(pos[0] - @as(i64, @intCast(prompt_len)) + 1);
            const cap_k = @min(draft_mod.depth(1, most, R_max), drafts -| emitted);
            const kk = draft_mod.chooseK(dr.confs[0][0..steps], @min(cap_k, steps), 1, most, R_max);
            var window_ok = kk + 1 == ids.len;
            if (window_ok) for (1..ids.len) |j| {
                if (ids[j] != dr.drafts[0][j - 1]) window_ok = false;
            };
            try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"draft_call\": {d}, \"steps\": {d}, \"drafts_equal\": {}, \"k\": {d}, \"conf\": [", .{ run.rank, k, bp.call, steps, drafts_ok, kk });
            for (dr.confs[0][0..steps], 0..) |cf, j| try w_out.print("{s}{e}", .{ if (j == 0) "" else ", ", cf });
            try w_out.print("], \"window_equal\": {}}}\n", .{window_ok});
            try w_out.flush();
            if (!drafts_ok or !window_ok) run.ok = false;
        }
        const p0: usize = @intCast(pos[0]);
        if (pos[0] < 0 or p0 + ids.len > seq.len) return error.RoundNotNext;
        for (ids, 0..) |id, i| seq[p0 + i] = @intCast(id);
        // the RoundRunner.forward around this run: the last one before it
        var fcall: u64 = 0;
        for (fx.points) |*q| {
            if (q.call < rp.call and std.mem.eql(u8, q.where, "RoundRunner.forward")) fcall = q.call;
        }
        var rc: RoundCheck = .{ .run = run, .after = rp.call, .call = rp.call, .fwd = fcall };
        const rows: round_mod.Rows = .{ .ids = ids, .pos = pos, .base = base[0], .end = end[0] };
        round_mod.forward(eng, &rd, ch, caches, rings, eh, seq[0 .. p0 + ids.len], rows, pool_window, .{ .ctx = &rc, .at = RoundCheck.at }, null) catch |err| {
            if (err == error.RoundMismatch) return false;
            try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, k, @errorName(err) });
            try w_out.flush();
            return false;
        };
        if (drafts > 0) {
            // the window's rows into slot 0's rings at once (TF_DS_EAGER_ABSORB: before the round's tokens are known)
            const items = [_]draft_mod.Item{.{ .slot = 0, .taps = rd.taps, .n = ids.len, .start = p0 }};
            try dr.absorbMany(eng, ch, &pool, &items);
            if (fx.find(rp.call, "Drafter.absorb_many", null, "in3.0.1")) |ap| {
                var label: [48]u8 = undefined;
                if (!try run.check(try std.fmt.bufPrint(&label, "R{d} absorb taps", .{rp.call}), ap, dr.at)) return false;
            }
        }
        for (0..ids.len) |j| targets[j] = try argmax(run, rd.logits + j * vocab * 4, vocab);
        @memcpy(prev_ids[0..ids.len], ids);
        prev_n = ids.len;
        prev_pos = pos[0];
        after = rp.call;
    }
    return run.ok;
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
    // the DSpark taps only when the recorded forward took them (a drafting engine: Model.forward's taps list)
    const want_taps = if (fx.find(call - 1, "Model.forward", null, "post.in.taps.0")) |q| q.call == call else false;
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
            if (checked) {
                const eg = fx.find(after, "Comm.gather", null, "in1") orelse return report(w_out, rank, li, "engram fixture", error.NoLayerFixture);
                // (2D: the projection here is a column part of the rank's; the gather below holds the ranks' whole ones)
                if (eng.two == null and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} engram projection", .{li}), eg, ch.ek)) return false;
                if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} engram gather", .{li}), fx.find(after, "Comm.gather", null, "out"), ch.ekg)) return false;
                after = eg.call;
            }
        }
        // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
        if (want_taps) {
            if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| {
                prompt.tap(eng, ch, j) catch |err| return report(w_out, rank, li, "tap", err);
            }
        }
        // the fixtures' points of this layer (a checked run only)
        const att: ?*const Point = if (checked) (fx.find(after, "Model.attention_k", l, "in2") orelse return report(w_out, rank, li, "attention fixture", error.NoLayerFixture)) else null;
        prompt.attnMixes(eng, ch, li) catch |err| return report(w_out, rank, li, "attention mixes", err);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention in", .{li}), att, ch.x)) return false;
        prompt.attention(eng, ch, caches, shared, li, rings[li], floor, kv_done != null and kv_done.? == li) catch |err| return report(w_out, rank, li, "attention", err);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention partial", .{li}), fx.find(after, "Model.attention_k", l, "out"), ch.pa)) return false;
        try prompt.gather(eng, ch, ch.pa, ch.ga);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention gather", .{li}), fx.find(att.?.call, "Comm.gather", null, "out"), ch.ga)) return false;
        prompt.ffnMixes(eng, ch, li) catch |err| return report(w_out, rank, li, "ffn mixes", err);
        const mo: ?*const Point = if (checked) (fx.find(att.?.call, "Model.moe", l, "in2") orelse return report(w_out, rank, li, "moe fixture", error.NoLayerFixture)) else null;
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe in", .{li}), mo, ch.x)) return false;
        prompt.moe(eng, ch, li) catch |err| return report(w_out, rank, li, "moe", err);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe partial", .{li}), fx.find(att.?.call, "Model.moe", l, "out"), ch.pm)) return false;
        try prompt.gather(eng, ch, ch.pm, ch.gm);
        if (checked) {
            const mg = fx.find(mo.?.call, "Comm.gather", null, "out") orelse return report(w_out, rank, li, "moe gather fixture", error.NoLayerFixture);
            if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe gather", .{li}), mg, ch.gm)) return false;
            after = mg.call;
        }
        prompt.endLayer(ch);
        if (li + 1 < c.layers) continue;
        // the chunk's end: the taps (Model.forward's list), the head on the last row, the prompt's logits
        for (0..if (want_taps) c.dspark_taps.slice().len else 0) |j| {
            var arg_buf: [32]u8 = undefined;
            const arg = try std.fmt.bufPrint(&arg_buf, "post.in.taps.{d}", .{j});
            if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "tap {d}", .{j}), fx.find(call - 1, "Model.forward", null, arg), prompt.tapRows(eng, ch, j))) return false;
        }
        prompt.head(eng, ch) catch |err| return report(w_out, rank, li, "head", err);
        if (checked) {
            const hg = fx.find(after, "Comm.gather", null, "in1") orelse return report(w_out, rank, li, "head fixture", error.NoLayerFixture);
            // (2D: the columns here are a quarter of the rank's; the gather holds both ranks' halves)
            if (eng.two == null and !try run.check("head columns", hg, ch.head_l)) return false;
            if (!try run.check("head gather", fx.find(after, "Comm.gather", null, "out"), ch.head_g)) return false;
            if (!try run.check("prompt logits", fx.find(call - 1, "Model.forward", null, "out"), ch.head_g)) return false;
        }
    }
    try run.s.synchronize();
    return true;
}

/// A fixture's one-element int64 (a round's id, position or extent) from its recorded first bytes.
fn headInt(p: *const Point) !i64 {
    const h = p.head orelse return error.NoHead;
    if (h.len < 16) return error.NoHead;
    var b: [8]u8 = undefined;
    _ = try std.fmt.hexToBytes(&b, h[0..16]);
    return std.mem.readInt(i64, &b, .little);
}

fn isPoint(p: *const Point, where: []const u8, arg: []const u8) bool {
    return std.mem.eql(u8, p.where, where) and std.mem.eql(u8, p.arg, arg);
}

/// A light run's round checker: the round's logits only (the recording holds no layer points).
const LightCheck = struct {
    run: *Run,
    call: u64,

    fn at(ctx: *anyopaque, what: round_mod.Point, layer: usize, dev: u64) anyerror!bool {
        _ = layer;
        const lc: *LightCheck = @ptrCast(@alignCast(ctx));
        if (what != .logits) return true;
        var buf: [48]u8 = undefined;
        return lc.run.check(try std.fmt.bufPrint(&buf, "R{d} logits", .{lc.call}), lc.run.fx.find(lc.call - 1, "RoundDecoder.run", null, "out"), dev);
    }
};

/// Every recorded request of a token recording: its prompt (the chunks' ids) through every layer, the prompt's logits
/// checked; then its rounds (at most max_rounds, 0: all), each round's logits checked and the greedy token against the
/// recorded next one; the caches and rings cleared between requests. A request whose first chunks the served engine
/// restored from a kept prompt (its recorded chunks start past 0: the first round's position says how many are missing)
/// takes them from an earlier request's prompt and computes them again: the state they leave is the restored one (the
/// same ids through the same encoder-only chunks). False on the first difference.
fn runLight(run: *Run, eng: *const prompt.Engine, ch: *prompt.Chunk, caches: *const prompt.Caches, rings: []const u64, eh: ?*prompt.EngramHost, a: std.mem.Allocator, arena: *prompt.Arena, max_rounds: usize, tokens: usize, graphs: ?*round_mod.Graphs) !bool {
    const fx = run.fx;
    const w_out = run.out;
    const c = eng.c;
    const pts = fx.points;
    const vocab = vocabOf(eng);
    var rd = try round_mod.Round.init(eng, arena, a, tokens);
    if (side_on) try rd.useSide(eng);
    defer rd.dropSide(eng);
    if (prefetch_on) try rd.usePrefetch(eng);
    defer rd.dropPrefetch();
    const seq = try a.alloc(i32, tokens + 16);
    const host_pos = try a.alloc(i64, chunk_rows);
    var prompts: std.ArrayList([]const i32) = .empty; // the earlier requests' prompts (kept prompts' prefixes)
    var i: usize = 0;
    var req: usize = 0;
    while (true) : (req += 1) {
        // the request: its chunks (Model.forward's ids), then its rounds (RoundDecoder.run's)
        while (i < pts.len and !(isPoint(&pts[i], "Model.forward", "in2") and pts[i].shape.len == 1)) i += 1;
        if (i == pts.len) break;
        var chunks: std.ArrayList(*const Point) = .empty;
        var rounds: std.ArrayList(*const Point) = .empty;
        while (i < pts.len and !isPoint(&pts[i], "RoundDecoder.run", "in1")) : (i += 1) {
            if (isPoint(&pts[i], "Model.forward", "in2") and pts[i].shape.len == 1) try chunks.append(a, &pts[i]);
        }
        while (i < pts.len and !(isPoint(&pts[i], "Model.forward", "in2") and pts[i].shape.len == 1)) : (i += 1) {
            if (isPoint(&pts[i], "RoundDecoder.run", "in1")) try rounds.append(a, &pts[i]);
        }
        // the prompt: every recorded chunk's ids, after the kept prefix the served engine restored (if any)
        var tail: usize = 0;
        for (chunks.items) |cp| tail += (try ints(run, cp)).len;
        const total: usize = if (rounds.items.len > 0) @intCast(try headInt(fx.find(rounds.items[0].call - 1, "RoundDecoder.run", null, "in2") orelse return error.NoRoundFixture)) else tail;
        if (total < tail or total > seq.len) return error.ChunkTooLong;
        const missing = total - tail;
        if (missing % chunk_rows != 0) return error.KeptPrefixNotChunked;
        if (missing > 0) {
            var found = false;
            for (prompts.items) |pp| {
                if (pp.len < missing) continue;
                @memcpy(seq[0..missing], pp[0..missing]);
                found = true;
                break;
            }
            if (!found) return error.NoKeptPrefix;
        }
        var len: usize = missing;
        for (chunks.items) |cp| {
            const ids = try ints(run, cp);
            for (ids, 0..) |id, j| seq[len + j] = @intCast(id);
            len += ids.len;
        }
        try prompts.append(a, try a.dupe(i32, seq[0..len]));
        const replay = len -| c.window;
        try caches.clear(eng);
        for (rings) |r| try eng.d.check(eng.d.api.cuMemsetD8Async(r, 0, eng.ringBytes(), eng.s.handle), "cuMemsetD8Async");
        var start: usize = 0;
        const ids64 = try a.alloc(i64, chunk_rows);
        while (start < len) {
            const n: usize = @min(chunk_rows, len - start);
            for (0..n) |j| ids64[j] = seq[start + j];
            var shared: prompt.Shared = .{};
            try prompt.begin(eng, ch, ids64[0..n], start, host_pos);
            if (!try runChunk(run, eng, ch, caches, &shared, rings, eh, seq[0 .. start + n], chunks.items[0].call, c.layers, false, null, replay, host_pos)) return false;
            start += n;
        }
        const last = chunks.items[chunks.items.len - 1];
        var buf: [48]u8 = undefined;
        const prompt_ok = try run.check(try std.fmt.bufPrint(&buf, "request {d} prompt logits", .{req}), fx.find(last.call - 1, "Model.forward", null, "out"), ch.head_g);
        var greedy = try argmax(run, ch.head_g, vocab);
        var have = len;
        var equal_tokens: usize = 0;
        var equal_logits: usize = 0;
        const n_rounds = if (max_rounds == 0) rounds.items.len else @min(max_rounds, rounds.items.len);
        for (rounds.items[0..n_rounds]) |rp| {
            const id = try headInt(rp);
            const pos = try headInt(fx.find(rp.call - 1, "RoundDecoder.run", null, "in2") orelse return error.NoRoundFixture);
            const base = try headInt(fx.find(rp.call - 1, "RoundDecoder.run", null, "in4") orelse return error.NoRoundFixture);
            const end = try headInt(fx.find(rp.call - 1, "RoundDecoder.run", null, "in5") orelse return error.NoRoundFixture);
            if (id == @as(i64, @intCast(greedy))) equal_tokens += 1;
            if (pos != @as(i64, @intCast(have)) or have >= seq.len) return error.RoundNotNext;
            seq[have] = @intCast(id);
            have += 1;
            var lc: LightCheck = .{ .run = run, .call = rp.call };
            const ids = [_]i64{id};
            const posv = [_]i64{pos};
            const rows: round_mod.Rows = .{ .ids = &ids, .pos = &posv, .base = 0, .end = end - base };
            const before = run.ok;
            round_mod.forward(eng, &rd, ch, caches, rings, eh, seq[0..have], rows, pool_window, .{ .ctx = &lc, .at = LightCheck.at }, graphs) catch |err| {
                if (err != error.RoundMismatch) {
                    try w_out.print("{{\"rank\": {d}, \"request\": {d}, \"round_call\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, req, rp.call, @errorName(err) });
                    try w_out.flush();
                    return false;
                }
            };
            if (run.ok and before) equal_logits += 1;
            greedy = try argmax(run, rd.logits, vocab);
        }
        try w_out.print("{{\"rank\": {d}, \"request\": {d}, \"prompt_tokens\": {d}, \"kept_prefix\": {d}, \"chunks\": {d}, \"prompt_logits_equal\": {}, \"rounds\": {d}, \"round_logits_equal\": {d}, \"greedy_equal\": {d}}}\n", .{ run.rank, req, len, missing, chunks.items.len, prompt_ok, n_rounds, equal_logits, equal_tokens });
        try w_out.flush();
        if (!run.ok) return false;
    }
    return run.ok;
}

/// A stream of the drafted light run (multi.py's Stream in MultiDecoder): its pool slot and extent, its ids (the prompt,
/// the kept tokens, the current window's rows), its kept length, pending token, tokens out and budget; its window.
const DStream = struct {
    req: usize,
    slot: usize,
    base: i64 = 0,
    end: i64 = 0,
    seq: []i32,
    len: usize = 0, // positions kept (sc.length)
    pending: i64 = 0,
    out: usize = 0, // tokens taken (len(s.out))
    budget: usize,
    done: bool = false,
    rounds: usize = 0,
    tokens: []i64, // the reply: the first token, then each round's new tokens
    smp: sampling.Sampling, // temperature 0: greedy
    // this round's window: its rows in the round, from row `row`
    want: usize = 0,
    win: [round_mod.max_rows]i64 = undefined,
    n: usize = 0,
    row: usize = 0,
};

/// The first point after call `after` at `where` with argument `arg` and call before `before` (a step between two).
fn findBefore(fx: *const Fixtures, after: u64, before: u64, where: []const u8, arg: []const u8) ?*const Point {
    const q = fx.find(after, where, null, arg) orelse return null;
    return if (q.call < before) q else null;
}

/// Every request of a drafting recording (the served concurrent decoder with drafts, greedy), as MultiDecoder serves
/// them: each request's prompt through every layer into its stream's slot (the lowest free one) and extent (the base
/// the recording's rounds show), its logits checked, its taps absorbed into the drafter's rings; then each recorded
/// round from the port's own scheduling: the live streams in admission order, the draft depth by their count, each
/// stream's want (its budget's rest), one batched drafter pass for the drafting streams (every recorded drafter point
/// checked), each stream's depth from its confidences (_choose_k), the windows [pending, drafts]; the rows against the
/// recorded round's, the round through every layer (every recorded point checked); each stream's targets (greedy), the
/// drafts kept while equal to them, the target after them, the end of its budget; every window's taps absorbed. A
/// round the recording holds and the scheduling does not make (or the other way) fails the run. `budgets`: each
/// request's max_tokens in admission order; `seeds` each request's sampling seed (with `smp`'s temperature, top_k and
/// top_p: the served sampler, sampling.zig, every target checked against the recorded DsEngine._sample's), none: greedy.
fn runDrafted(run: *Run, eng: *const prompt.Engine, ch: *prompt.Chunk, caches: *const prompt.Caches, rings: []const u64, eh: ?*prompt.EngramHost, a: std.mem.Allocator, arena: *prompt.Arena, budgets: []const usize, seeds: []const u64, smp: sampling.Sampling, sp: dsv41.plan.Split) !bool {
    const fx = run.fx;
    const w_out = run.out;
    const c = eng.c;
    const pts = fx.points;
    const vocab = vocabOf(eng);
    const R_max = round_mod.max_rows;
    const most = 5; // the engine's drafts (--mtp-drafts 5)
    var rd = try round_mod.Round.init(eng, arena, a, ch.max_comp);
    if (side_on) try rd.useSide(eng);
    defer rd.dropSide(eng);
    if (prefetch_on) try rd.usePrefetch(eng);
    defer rd.dropPrefetch();
    var pool = try draft_mod.Pool.init(eng, arena, draft_mod.max_streams);
    var dr = try draft_mod.Drafter.init(eng, arena, sp);
    var streams: std.ArrayList(DStream) = .empty;
    const host_pos = try a.alloc(i64, chunk_rows);
    const ids64 = try a.alloc(i64, chunk_rows);
    const ring_view = try a.alloc(u64, rings.len);
    var prev_round: u64 = 0; // the last round's RoundDecoder.run call
    var i: usize = 0;
    while (i < pts.len) : (i += 1) {
        const p = &pts[i];
        if (isPoint(p, "Model.forward", "in2") and p.shape.len == 1) {
            // a request's prompt: its chunks to the one that reaches the head (its logits)
            const req = streams.items.len;
            if (req >= budgets.len) return error.MoreRequestsThanBudgets;
            var chunks: std.ArrayList(*const Point) = .empty;
            var last: ?*const Point = null;
            var j = i;
            while (j < pts.len) : (j += 1) {
                const q = &pts[j];
                if (isPoint(q, "RoundDecoder.run", "in1")) break; // (a round inside a prompt's fill: not ported)
                if (!(isPoint(q, "Model.forward", "in2") and q.shape.len == 1)) continue;
                try chunks.append(a, q);
                if (fx.find(q.call - 1, "Model.forward", null, "out")) |o| if (o.call == q.call) {
                    last = q;
                    break;
                };
            }
            const lp = last orelse return error.PromptWithoutLogits;
            var len: usize = 0;
            for (chunks.items) |cp| len += (try ints(run, cp)).len;
            // its slot: the lowest one no live stream holds; its extent: the first round's rows in that slot
            var held: [draft_mod.max_streams]bool = @splat(false);
            for (streams.items) |*st| {
                if (!st.done) held[st.slot] = true;
            }
            const slot = std.mem.indexOfScalar(bool, &held, false) orelse return error.NoFreeSlot;
            var base: i64 = 0;
            var end: i64 = 0;
            var found = false;
            var k = j;
            while (k < pts.len and !found) : (k += 1) {
                const q = &pts[k];
                if (!isPoint(q, "RoundDecoder.run", "in3")) continue;
                const sl = try ints(run, q);
                for (sl, 0..) |v, r| {
                    if (v != @as(i64, @intCast(slot))) continue;
                    const rpos = try ints(run, fx.find(q.call - 1, "RoundDecoder.run", null, "in2"));
                    if (rpos[r] != @as(i64, @intCast(len))) return error.SlotNotThePrompts;
                    base = (try ints(run, fx.find(q.call - 1, "RoundDecoder.run", null, "in4")))[r];
                    end = (try ints(run, fx.find(q.call - 1, "RoundDecoder.run", null, "in5")))[r];
                    found = true;
                    break;
                }
            }
            const st = try streams.addOne(a);
            var ss = smp;
            if (seeds.len > 0) {
                if (req >= seeds.len) return error.MoreRequestsThanSeeds;
                ss.seed = seeds[req];
            } else ss.temperature = 0;
            st.* = .{ .req = req, .slot = slot, .base = base, .end = end, .seq = try a.alloc(i32, len + budgets[req] + R_max), .budget = budgets[req], .tokens = try a.alloc(i64, budgets[req]), .smp = ss };
            var at: usize = 0;
            for (chunks.items) |cp| for (try ints(run, cp)) |id| {
                st.seq[at] = @intCast(id);
                at += 1;
            };
            // the prompt into its slot's rings and positional stores, its extent's compressed rows
            const view = try caches.view(eng, slot, @intCast(base));
            for (rings, 0..) |r, li| ring_view[li] = r + slot * eng.ringBytes();
            const replay = len -| c.window;
            var start: usize = 0;
            while (start < len) {
                const n: usize = @min(chunk_rows, len - start);
                for (0..n) |jj| ids64[jj] = st.seq[start + jj];
                var shared: prompt.Shared = .{};
                try prompt.begin(eng, ch, ids64[0..n], start, host_pos);
                if (!try runChunk(run, eng, ch, &view, &shared, ring_view, eh, st.seq[0 .. start + n], lp.call, c.layers, false, null, replay, host_pos)) return false;
                start += n;
            }
            var buf: [64]u8 = undefined;
            const prompt_ok = try run.check(try std.fmt.bufPrint(&buf, "request {d} prompt logits", .{req}), fx.find(lp.call - 1, "Model.forward", null, "out"), ch.head_g);
            // its taps (the replayed rows, the tap layers side by side) into its slot's drafter rings
            const d = c.hidden;
            for (0..c.dspark_taps.slice().len) |jj| try eng.ops.copyRows(eng.s, prompt.tapRows(eng, ch, jj), d * 2, dr.at + jj * d * 2, dr.taps_w * 2, d * 2, ch.n);
            if (fx.find(lp.call, "Drafter.absorb", null, "in3")) |ap| {
                if (!try run.check(try std.fmt.bufPrint(&buf, "request {d} absorb taps", .{req}), ap, dr.at)) return false;
            }
            try dr.absorb(eng, ch, &pool, slot, dr.at, ch.n, ch.start);
            st.len = len;
            st.pending = try sampleDev(run, ch.head_g, vocab, len, st.smp);
            st.tokens[0] = st.pending;
            st.out = 1;
            if (st.pending == c.eos or st.out >= st.budget) st.done = true;
            try w_out.print("{{\"rank\": {d}, \"request\": {d}, \"prompt_tokens\": {d}, \"chunks\": {d}, \"slot\": {d}, \"base\": {d}, \"prompt_logits_equal\": {}, \"first\": {d}}}\n", .{ run.rank, req, len, chunks.items.len, slot, base, prompt_ok, st.pending });
            try w_out.flush();
            if (!found and !st.done) return error.NoRoundForThePrompt;
            i = j;
            continue;
        }
        if (!isPoint(p, "RoundDecoder.run", "in1")) continue;
        // a round: the live streams in admission order, each one's want and window
        const rp = p;
        var live: [draft_mod.max_streams]*DStream = undefined;
        var nl: usize = 0;
        for (streams.items) |*st| {
            if (st.done) continue;
            if (nl == live.len) return error.TooManyStreams;
            live[nl] = st;
            nl += 1;
        }
        if (nl == 0) return error.RoundWithoutStreams;
        const kdep = draft_mod.depth(nl, most, R_max);
        var tok: [draft_mod.max_streams]i64 = undefined;
        var q0: [draft_mod.max_streams]i64 = undefined;
        var sl: [draft_mod.max_streams]i64 = undefined;
        var nd: usize = 0;
        for (live[0..nl]) |st| {
            st.want = @min(kdep, st.budget -| st.out);
            st.win[0] = st.pending;
            st.n = 1;
            if (st.want == 0) continue;
            tok[nd] = st.pending;
            q0[nd] = @intCast(st.len);
            sl[nd] = @intCast(st.slot);
            nd += 1;
        }
        var label: [64]u8 = undefined;
        if (nd > 0) {
            // the drafting streams' pass: its recorded inputs, every recorded point, its drafts; each stream's depth
            const bp = findBefore(fx, prev_round, rp.call, "BatchDraftGraph.run", "in1") orelse return error.NoDraftFixture;
            const rtok = try ints(run, bp);
            const rq0 = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, "in2"));
            const rsl = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, "in3"));
            const inputs_ok = std.mem.eql(i64, rtok, tok[0..nd]) and std.mem.eql(i64, rq0, q0[0..nd]) and std.mem.eql(i64, rsl, sl[0..nd]);
            if (!inputs_ok) {
                try w_out.print("{{\"rank\": {d}, \"round_call\": {d}, \"draft_call\": {d}, \"draft_inputs_equal\": false}}\n", .{ run.rank, rp.call, bp.call });
                try w_out.flush();
                return false;
            }
            const steps = draft_mod.passSteps(nd, pool.slots, most, R_max, dr.n);
            var vis: [draft_mod.max_streams * 128]usize = undefined;
            var nv: usize = 0;
            for (q0[0..nd], sl[0..nd]) |q, slot| {
                var pp: i64 = @max(0, q - @as(i64, @intCast(c.window)));
                while (pp < q) : (pp += 1) {
                    vis[nv] = @as(usize, @intCast(slot)) * pool.ring + @as(usize, @intCast(pp)) % pool.ring;
                    nv += 1;
                }
            }
            var dc: DraftCheck = .{ .run = run, .call = bp.call, .layers = c.layers, .after = bp.call, .vis = vis[0..nv] };
            dr.pass(eng, ch, &pool, tok[0..nd], q0[0..nd], sl[0..nd], steps, .{ .ctx = &dc, .at = DraftCheck.at }) catch |err| {
                if (err == error.DraftMismatch) return false;
                try w_out.print("{{\"rank\": {d}, \"draft_call\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, bp.call, @errorName(err) });
                try w_out.flush();
                return false;
            };
            var di: usize = 0;
            for (live[0..nl]) |st| {
                if (st.want == 0) continue;
                const want = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, try std.fmt.bufPrint(&label, "out.{d}", .{di})));
                if (want.len != steps or !std.mem.eql(i64, want, dr.drafts[di][0..steps])) {
                    try w_out.print("{{\"rank\": {d}, \"draft_call\": {d}, \"stream\": {d}, \"drafts_equal\": false}}\n", .{ run.rank, bp.call, st.req });
                    try w_out.flush();
                    return false;
                }
                const kk = draft_mod.chooseK(dr.confs[di][0..steps], @min(st.want, steps), nl, most, R_max);
                for (0..kk) |jj| st.win[1 + jj] = dr.drafts[di][jj];
                st.n = 1 + kk;
                di += 1;
            }
        }
        // the rows: each live stream's window at its kept length, its slot and extent; its ids through the window
        var r_ids: [R_max]i64 = undefined;
        var r_pos: [R_max]i64 = undefined;
        var r_sl: [R_max]i64 = undefined;
        var r_base: [R_max]i64 = undefined;
        var r_end: [R_max]i64 = undefined;
        var wins: [draft_mod.max_streams]round_mod.Window = undefined;
        var R: usize = 0;
        for (live[0..nl], 0..) |st, wi| {
            if (R + st.n > R_max) return error.RoundTooWide;
            st.row = R;
            for (0..st.n) |jj| {
                r_ids[R + jj] = st.win[jj];
                r_pos[R + jj] = @intCast(st.len + jj);
                r_sl[R + jj] = @intCast(st.slot);
                r_base[R + jj] = st.base;
                r_end[R + jj] = st.end;
                st.seq[st.len + jj] = @intCast(st.win[jj]);
            }
            wins[wi] = .{ .row = R, .n = st.n, .seq = st.seq[0 .. st.len + st.n] };
            R += st.n;
        }
        const rows_ok = std.mem.eql(i64, try ints(run, rp), r_ids[0..R]) and
            std.mem.eql(i64, try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in2")), r_pos[0..R]) and
            std.mem.eql(i64, try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in3")), r_sl[0..R]) and
            std.mem.eql(i64, try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in4")), r_base[0..R]) and
            std.mem.eql(i64, try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in5")), r_end[0..R]);
        try w_out.print("{{\"rank\": {d}, \"round_call\": {d}, \"streams\": {d}, \"drafting\": {d}, \"rows\": {d}, \"rows_equal\": {}}}\n", .{ run.rank, rp.call, nl, nd, R, rows_ok });
        try w_out.flush();
        if (!rows_ok) return false;
        var fcall: u64 = 0;
        for (fx.points) |*q| {
            if (q.call < rp.call and std.mem.eql(u8, q.where, "RoundRunner.forward")) fcall = q.call;
        }
        var rc: RoundCheck = .{ .run = run, .after = rp.call, .call = rp.call, .fwd = fcall };
        const rows: round_mod.Rows = .{ .ids = r_ids[0..R], .pos = r_pos[0..R], .slots = r_sl[0..R], .bases = r_base[0..R], .ends = r_end[0..R], .windows = wins[0..nl] };
        round_mod.forward(eng, &rd, ch, caches, rings, eh, &.{}, rows, pool_window, .{ .ctx = &rc, .at = RoundCheck.at }, null) catch |err| {
            if (err == error.RoundMismatch) return false;
            try w_out.print("{{\"rank\": {d}, \"round_call\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, rp.call, @errorName(err) });
            try w_out.flush();
            return false;
        };
        // every window's taps into its slot's rings at once (the served eager absorb), then each stream's acceptance
        var items: [draft_mod.max_streams]draft_mod.Item = undefined;
        for (live[0..nl], 0..) |st, wi| items[wi] = .{ .slot = st.slot, .taps = rd.taps + st.row * dr.taps_w * 2, .n = st.n, .start = st.len };
        try dr.absorbMany(eng, ch, &pool, items[0..nl]); // (its rows: the round's taps, checked)
        var sample_after = rp.call;
        for (live[0..nl]) |st| {
            var targets: [R_max]i64 = undefined;
            for (0..st.n) |jj| targets[jj] = try sampleDev(run, rd.logits + (st.row + jj) * vocab * 4, vocab, st.len + 1 + jj, st.smp);
            // the served sampler's tokens for this window (a recording with sampler points)
            if (fx.find(sample_after, "DsEngine._sample", null, "out")) |sq| {
                const want = try ints(run, sq);
                if (!std.mem.eql(i64, want, targets[0..st.n])) {
                    try w_out.print("{{\"rank\": {d}, \"round_call\": {d}, \"stream\": {d}, \"sample_call\": {d}, \"samples_equal\": false}}\n", .{ run.rank, rp.call, st.req, sq.call });
                    try w_out.flush();
                    return false;
                }
                sample_after = sq.call;
            }
            var acc: usize = 0;
            while (acc + 1 < st.n and st.win[acc + 1] == targets[acc]) acc += 1;
            st.len += acc + 1;
            pool.absorbed[st.slot] = st.len;
            // the new tokens: the kept drafts and the target after them, to an end token or the budget
            var take: usize = acc + 1;
            for (0..acc + 1) |jj| {
                const t = if (jj < acc) st.win[jj + 1] else targets[acc];
                if (t == c.eos) {
                    take = jj + 1;
                    st.done = true;
                    break;
                }
            }
            take = @min(take, st.budget -| st.out);
            for (0..take) |jj| st.tokens[st.out + jj] = if (jj < acc) st.win[jj + 1] else targets[acc];
            st.out += take;
            if (st.out >= st.budget) st.done = true;
            st.pending = if (take == 0) st.pending else (if (take - 1 < acc) st.win[take] else targets[acc]);
            st.rounds += 1;
        }
        prev_round = rp.call;
    }
    // every stream's rounds ended where the recording's did
    var ok = run.ok;
    for (streams.items) |*st| {
        try w_out.print("{{\"rank\": {d}, \"request\": {d}, \"rounds\": {d}, \"tokens\": {d}, \"done\": {}, \"reply\": [", .{ run.rank, st.req, st.rounds, st.out, st.done });
        for (st.tokens[0..st.out], 0..) |t, jj| try w_out.print("{s}{d}", .{ if (jj == 0) "" else ", ", t });
        try w_out.print("]}}\n", .{});
        if (!st.done) ok = false;
    }
    try w_out.flush();
    return ok;
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
