//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const rank_cache = @import("rank_cache.zig");
const weights = @import("weights.zig");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const ops_mod = @import("ops.zig");
const exact = @import("exact.zig");
const cublas = @import("cublas.zig");
const comm_mod = @import("comm.zig");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const rdma = @import("rdma.zig");

/// The served prompt chunk (TF_DS_PREFILL_CHUNK): a prompt runs in chunks that start at its multiples.
pub const chunk_rows = 2048;
/// The pool's streams (--parallel 4).
pub const max_streams = draft.max_streams;
/// The pool's streams when unset (--parallel 4).
pub const max_streams_default = draft.default_streams;
/// The RDMA ring's slot (the served 4.25 MiB: a 16-row round's logits half fits).
pub const ring_bytes = 4456448;

pub const Options = struct {
    model_dir: []const u8, // config.json
    cache_dir: []const u8, // the lane's per-rank weight file (TF_DS_RANK_CACHE)
    kit_dir: []const u8, // a gate dir's layout: aot/, cubins/, rope-{plain,compressed}-{cos,sin}.f32, engram.json
    aot_dir: ?[]const u8 = null, // the recorded Triton set (aot.json and its cubins; null: kit_dir/aot)
    rank: u32,
    world: u32,
    master: [4]u8,
    port: u16,
    engram_dir: ?[]const u8 = null, // the Engram tables (none: a checkpoint without Engram layers)
    token_map: ?[]const u8 = null, // the compressed token map (the lane's JSON cache)
    pool: usize = 1 << 18, // positions of the streams' shared plane (their extents together)
    drafts: bool = true, // load the DSpark drafter
    rdma_devices: ?[]const u8 = null, // the decode gathers over RDMA on these devices (comma separated), else NCCL: TP2's
    // ring (rdma.zig), or at world 4 the 2D split's decode-size exchanges over its rings (ring2d.zig)
    rdma_kernels: ?[]const u8 = null, // world 4: the rings' kernels' image (rdma_gather.cu's fatbin; null: the embedded one)
    graphs: bool = false, // the rounds' stretches as CUDA graphs (round.Graphs)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    host: std.heap.ArenaAllocator, // allocations that live as long as the model
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    cfg: Config,
    rank: u32,
    world: u32,
    streams: usize, // the pool's streams (Options.streams)
    round_ms: []const f64, // the lane core's round costs by rows (Options.round_ms, else draft.round_ms)
    pool_cap: usize,
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: prompt2d.Fan,
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    caches: prompt.Caches,
    blas: cublas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    eh: ?prompt.EngramHost = null,
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    view: prompt.Caches = undefined,
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    timer: ?*round.PhaseTimer = null, // a profile's GPU time by phase (eager rounds)
    ptimer: ?*round.PhaseTimer = null, // the same for prompt chunks (fills), a chunk a 'round'
    t_enqueue: u64 = 0, // the host's time to issue the forward (the GPU idle before it)
    t_forward: u64 = 0,
    t_absorb: u64 = 0,
    // world 4: the 2D split's exchanges (prompt2d.zig, TP4's), with the RDMA rings when asked
    two: ?prompt2d.Two = null,
    rdma_rings: ?ring2d.Rings = null,

    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port), opens NCCL
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {
        if (o.world != 2 and o.world != 4) return error.NotPortedYet; // TP2, or the four-node 2D split (prompt2d.zig)
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.host = std.heap.ArenaAllocator.init(gpa);
        errdefer m.host.deinit();
        const a = m.host.allocator();
        m.io = io;
        m.ctx = ctx;
        m.rank = o.rank;
        m.world = o.world;
        if (o.streams == 0 or o.streams > max_streams) return error.BadStreams;
        m.streams = o.streams;
        m.round_ms = o.round_ms orelse &draft.round_ms;
        if (m.round_ms.len == 0) return error.BadRoundMs;
        m.pool_cap = o.pool;
        m.fill_slot = 0;
        m.prof = false;
        m.timer = null;
        m.ptimer = null;
        m.t_enqueue = 0;
        m.t_forward = 0;
        m.t_absorb = 0;
        m.dpool = null;
        m.dr = null;
        m.tables = null;
        m.epool = null;
        m.aio = null;
        m.eh = null;
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.two = null;
        m.rdma_rings = null;
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        const c = &m.cfg;
        m.stream = try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();

        // the weights (and the DSpark blocks) from the lane's rank cache
        m.cache_dir = try std.Io.Dir.cwd().openDir(io, o.cache_dir, .{ .iterate = true });
        errdefer m.cache_dir.close(io);
        const cache_name = try rank_cache.find(a, io, m.cache_dir, o.rank, o.world);
        m.ix = try rank_cache.open(gpa, io, m.cache_dir, cache_name);
        errdefer m.ix.deinit();
        m.cache_file = try m.cache_dir.openFile(io, cache_name, .{});
        errdefer m.cache_file.close(io);
        const sp = try prompt2d.split(o.rank, o.world);
        m.w = try weights.load(gpa, d, .{ .cache = .{ .file = m.cache_file, .index = &m.ix, .io = io } }, c.*, sp, o.drafts);
        errdefer m.w.deinit();
        if (o.drafts and m.w.dspark == null) return error.NoDrafter;

        // the ranks' link and NCCL (rank 0 listens: rank 1 retries its connect while rank 0 loads)
        m.fan = try prompt2d.Fan.open(io, o.master, o.port, o.rank, o.world);
        errdefer m.fan.close();
        m.comm = try m.fan.comm(o.rank, o.world);
        errdefer m.comm.deinit();
        // TP2: the decode gathers' RDMA ring (both ranks connect before either returns)
        if (sp.pair == null and o.rdma_devices != null) {
            const list = o.rdma_devices.?;
            if (!cuda.kernels.available) return error.NoKernelImages;
            m.rdma_mod = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
            var devices: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |dev| try devices.append(a, dev);
            m.ring = try openRing(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, &m.comm, m.stream);
        }
        errdefer if (m.ring) |r| r.destroy(gpa);
        errdefer if (m.rdma_mod) |*x| x.unload();

        // kernels: the recorded Triton set, the served extension cubins, the torch-op images
        m.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }));
        errdefer m.set.deinit();
        m.pf = try exl3_prefill.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear.cubin"));
        errdefer m.pf.unload();
        m.lg = try exl3_linear.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear_grouped.cubin"));
        errdefer m.lg.unload();
        m.exk = try exl3_experts.Kernels.load(d, try readKit(a, io, o.kit_dir, "experts.cubin"), try readKit(a, io, o.kit_dir, "experts_cb.cubin"), exl3_linear.codebook_mul1);
        errdefer m.exk.unload();
        if (!cuda.kernels.available) return error.NoKernelImages;
        m.ops = try ops_mod.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
        errdefer m.ops.unload();
        m.ex = try exact.Exact.load(d, cuda.kernels.dsv41_torch);
        errdefer m.ex.unload();

        // the RoPE tables (the served build's bits: torch CPU pow and polar), every position of the pool
        const half = c.rope_dim / 2;
        const rope_rows = o.pool + round.max_rows + 1;
        m.rope_buf = try cuda.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
        errdefer m.rope_buf.free();
        {
            const part = try gpa.alloc(u8, rope_rows * half * 4);
            defer gpa.free(part);
            for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
                var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.kit_dir, name }), .{});
                defer f.close(io);
                if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // a table of fewer rows than the pool
                try m.rope_buf.upload(j * part.len, part);
            }
        }
        const tbl = rope_rows * half * 4;
        const rope: prompt.Rope = .{ .cos = m.rope_buf.ptr, .sin = m.rope_buf.ptr + tbl };
        const rope_c: prompt.Rope = .{ .cos = m.rope_buf.ptr + 2 * tbl, .sin = m.rope_buf.ptr + 3 * tbl };

        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer m.arena.deinit();
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
        // round over NCCL world 4)
        errdefer if (m.rdma_rings) |*x| x.close(gpa);
        if (sp.pair != null) {
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, chunk_rows);
            if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (cuda.kernels.available) cuda.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try cuda.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, .{ .max_bytes = ring_bytes, .gid_index = 5 }, &m.comm, m.stream);
                m.two.?.rings = &m.rdma_rings.?;
            }
        }
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream }, .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (m.two) |*t| t else null };
        m.eng.round_rows = o.round_rows;
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, chunk_rows, o.pool);
        m.caches = try prompt.Caches.init(&m.eng, &m.arena, o.pool, m.streams);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try d.check(d.api.cuMemsetD8Async(r.*, 0, m.streams * m.eng.ringBytes(), m.stream.handle), "cuMemsetD8Async");
        }
        m.rd = try round.Round.init(&m.eng, &m.arena, a, o.pool);
        m.amax = try m.arena.take(round.max_rows * 4);
        if (o.graphs) m.graphs = round.Graphs.init(gpa);
        if (o.side) try m.rd.useSide(&m.eng);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (o.drafts) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa);
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        try m.stream.synchronize();
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too

        // Engram on the host: the compressed token map, the multipliers, the tables and 64 readers
        if (o.engram_dir) |edir| {
            const map_text = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
            const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
            const EngramJson = struct { multipliers: [][]i64 };
            const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "engram.json" }), a, .limited(1 << 26));
            const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
            var mult: [engram.max_layers][engram.max_ngram]i64 = @splat(@splat(0));
            for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
                mult[l][k] = v;
            };
            m.tables = try engram_io.Tables.open(gpa, io, edir);
            m.epool = try engram_io.Pool.init(gpa, io, 64);
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, chunk_rows);
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;

        m.host_pos = try a.alloc(i64, chunk_rows);
        m.ids64 = try a.alloc(i64, chunk_rows);
        m.vocab = sp.world * (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n); // 2D: both pairs' vocabulary parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        return m;
    }

    pub fn close(m: *Model) void {
        m.stream.synchronize() catch {};
        if (m.graphs) |*g| g.deinit();
        if (m.pgraphs) |*g| g.deinit();
        if (m.timer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        if (m.aio) |x| x.deinit();
        if (m.epool) |p| p.deinit(m.gpa);
        if (m.tables) |*t| t.close();
        if (m.rdma_rings) |*r| r.close(m.gpa);
        m.blas.close();
        m.arena.deinit();
        m.rope_buf.free();
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
        if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        }
        if (m.rdma_mod) |*x| x.unload();
        m.comm.deinit();
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
        m.stream.deinit();
        m.host.deinit();
        m.gpa.destroy(m);
    }

    /// TP2's RDMA ring for the decode gathers: created on both ranks, the queue-pair infos exchanged over NCCL,
    /// connected, and started once both are (an all-gather as the barrier).
    pub fn openRing(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream) !*rdma.Ring {
        if (comm.world != 2) return error.NotTwoRanks;
        const r = try rdma.Ring.create(gpa, d, k, devices, rank, 2, .{ .max_bytes = ring_bytes, .gid_index = 5 });
        errdefer r.destroy(gpa);
        const per = @sizeOf(rdma.Info);
        const mine = [2]rdma.Info{ r.info(0), r.info(1) };
        var dev = try cuda.DeviceBuffer.alloc(d, 3 * 2 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 2 * per, .u8, stream);
        try stream.synchronize();
        var all: [2][2]rdma.Info = undefined;
        try dev.download(2 * per, std.mem.sliceAsBytes(&all));
        try r.connect(&.{ all[0][rank], all[1][rank] });
        // every queue pair ready on both ranks before either sends
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 1, .u8, stream);
        try stream.synchronize();
        try r.start();
        return r;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// A stream's prompt is about to fill pool slot `slot`, its compressed rows in its extent from position `base`.
    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head.
    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        if (n == 0 or n > chunk_rows or seq.len != start + n) return error.BadChunk;
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        for (0..n) |j| m.ids64[j] = seq[start + j];
        var shared: prompt.Shared = .{};
        const pt = m.ptimer;
        if (pt) |t| try t.mark(m.stream, .start);
        try prompt.begin(e, ch, m.ids64[0..n], start, m.host_pos);
        if (pt) |t| try t.mark(m.stream, .embed);
        const taps = m.dr != null;
        var floor: usize = 0;
        var kv_done: ?usize = null;
        for (0..c.layers) |li| {
            if (li == c.layers / 2) {
                if (!try prompt.replayCut(e, ch, &m.view, &shared, li, replay, m.host_pos)) {
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profile alone waits
                    if (pt) |t| {
                        try m.stream.synchronize();
                        try t.finish();
                    }
                    return false;
                }
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                try prompt.engramApply(e, ch, if (m.eh) |*x| x else return error.NoEngramTables, li, seq);
                if (pt) |t| try t.mark(m.stream, .engram);
            }
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
            try prompt.attnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(m.stream, .attn);
            try prompt.gather(e, ch, ch.pa, ch.ga);
            if (pt) |t| try t.mark(m.stream, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .moe);
            try prompt.gather(e, ch, ch.pm, ch.gm);
            if (pt) |t| try t.mark(m.stream, .gather_m);
            prompt.endLayer(ch);
        }
        try prompt.head(e, ch);
        if (pt) |t| try t.mark(m.stream, .head);
        if (m.dr) |*dr| {
            // its taps (the tap layers side by side) into the slot's drafter rings
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.stream.synchronize();
            try t.finish();
        }
        return true;
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        const e = &m.eng;
        if (m.prof) try m.stream.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        const t0 = m.now();
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.pool_cap, null, if (m.graphs) |*g| g else null);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.stream.synchronize();
            m.t_forward = m.now() - t0;
            if (m.timer) |t| try t.finish();
        }
        const t1 = m.now();
        if (m.dr) |*dr| {
            const wins = rows.windows orelse return error.BadRound;
            var items: [max_streams]draft.Item = undefined;
            var n: usize = 0;
            for (wins, absorb) |win, yes| {
                if (!yes) continue;
                items[n] = .{ .slot = @intCast(rows.slots.?[win.row]), .taps = m.rd.taps + win.row * dr.taps_w * 2, .n = win.n, .start = @intCast(rows.pos[win.row]) };
                n += 1;
            }
            if (n > 0) try dr.absorbMany(e, &m.ch, &m.dpool.?, items[0..n]);
        }
        try m.stream.synchronize();
        if (m.prof) m.t_absorb = m.now() - t1;
    }

    /// The rounds' GPU time by phase (eager rounds; --profile 2): a timer on the round.
    pub fn usePhaseTimer(m: *Model) !void {
        if (m.timer != null) return;
        const t = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(t);
        t.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.timer = t;
        const pt = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(pt);
        pt.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.ptimer = pt;
        m.rd.timer = t;
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    }

    /// The round's logits, R rows of the vocabulary, on the host (after verify).
    /// The round's greedy tokens, R rows (the device's argmax of the round's logits, as sampling.argmax), on the host.
    pub fn roundArgmax(m: *Model, R: usize) ![]const u32 {
        const d = m.ctx.d;
        try m.ops.argmaxRows(m.stream, m.rd.logits, m.vocab, m.vocab, R, m.amax);
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.amax_host, m.amax, R * 4), "cuMemcpyDtoH");
        return m.amax_host[0..R];
    }

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.rd.logits, R * m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0 .. R * m.vocab];
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.ch.head_g, m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0..m.vocab];
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each (the drafter's drafts and confidences on the host).
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize) !void {
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);
        try dr.pass(&m.eng, &m.ch, pool, tokens, q0, slots, steps, null);
    }
};
