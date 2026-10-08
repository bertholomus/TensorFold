//! tf-dsv41-lanes MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT KIT_DIR --requests FILE [--parallel N] [--pool P]
//!   [--drafts 0|1] [--arena-gib G] [--engram DIR --token-map FILE]: the lane gate of the Zig port. TensorFold 1.0's
//! lane core (core/lanes: its round loop, depth rule, window planning, copies, acceptance) serves the requests through
//! the port's Backend (lanes.zig) on both ranks: rank 0 runs the core and admits up to N requests at a time (the host's
//! SuffixLookup proposer on each, min_match 4), rank 1 follows. Each reply's token_sha (sha256 of its comma-joined ids,
//! the served server's) against the served reply's: the drafted, concurrent reply must be the served one. One JSON
//! line a request, then the summary. FILE: zrec_lanereq.py's requests (prompt ids, max_tokens, sampling, expect_sha).
//! KIT_DIR: a gate dir's aot/, cubins/, RoPE tables (as many rows as the pool) and engram.json.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const lanes = @import("lanes");

const usage = "usage: tf-dsv41-lanes MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT KIT_DIR --requests FILE [--parallel N] [--pool P] [--drafts 0|1] [--arena-gib G] [--engram DIR --token-map FILE]\n";

const Request = struct {
    name: []const u8 = "",
    prompt: []const u32,
    max_tokens: u32,
    seed: ?u64 = null,
    temperature: f64 = 0,
    top_k: u32 = 20,
    top_p: f64 = 0.95,
    expect_sha: ?[]const u8 = null,
};

const Job = struct {
    req: *const Request,
    index: usize,
    proposer: lanes.SuffixLookup,
    stream: lanes.Stream,
    began: std.Io.Timestamp,
    done: bool = false,
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
    var o: dsv41.model.Options = .{
        .model_dir = args[1],
        .cache_dir = args[2],
        .rank = try std.fmt.parseInt(u32, args[3], 10),
        .world = try std.fmt.parseInt(u32, args[4], 10),
        .master = try dsv41.link.parseIp(args[5]),
        .port = try std.fmt.parseInt(u16, args[6], 10),
        .kit_dir = args[7],
    };
    var requests_file: ?[]const u8 = null;
    var parallel: usize = dsv41.model.max_streams;
    var arena_gib: ?usize = null;
    var profile = false;
    var serial = false; // --serial 1: every stream without drafts (one row a round)
    var ai: usize = 8;
    while (ai + 1 < args.len) : (ai += 2) {
        const key = args[ai];
        const val = args[ai + 1];
        if (std.mem.eql(u8, key, "--requests")) {
            requests_file = val;
        } else if (std.mem.eql(u8, key, "--parallel")) {
            parallel = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--pool")) {
            o.pool = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--drafts")) {
            o.drafts = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--arena-gib")) {
            arena_gib = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--aio")) {
            o.engram_aio = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--prefetch")) {
            o.prefetch = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--side")) {
            o.side = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--graphs")) {
            o.graphs = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--serial")) {
            serial = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--rdma")) {
            o.rdma_devices = val;
        } else if (std.mem.eql(u8, key, "--profile")) {
            profile = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--engram")) {
            o.engram_dir = val;
        } else if (std.mem.eql(u8, key, "--token-map")) {
            o.token_map = val;
        } else return error.BadArgument;
    }
    if (parallel == 0 or parallel > dsv41.model.max_streams) return error.BadArgument;
    o.arena_bytes = (arena_gib orelse dsv41.native.defaultArenaGib(o.pool)) << 30; // (rank 1 of a server: the server's)

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w = &out.interface;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const t0 = std.Io.Timestamp.now(io, .awake);
    const m = try dsv41.model.Model.open(gpa, io, &ctx, o);
    defer m.close();
    try w.print("{{\"rank\": {d}, \"loaded_s\": {d:.1}, \"pool\": {d}, \"drafts\": {}, \"arena_used\": {d}}}\n", .{ o.rank, seconds(t0, io), o.pool, m.drafting(), m.arena.used });
    try w.flush();

    if (o.rank != 0) {
        dsv41.lanes.follow(gpa, m) catch |err| {
            try w.print("{{\"rank\": {d}, \"follow\": \"{s}\"}}\n", .{ o.rank, @errorName(err) });
            try w.flush();
            return 1;
        };
        try w.print("{{\"rank\": {d}, \"followed\": true}}\n", .{o.rank});
        try w.flush();
        return 0;
    }

    // rank 0: the requests, the backend and the lane core
    const text = try std.Io.Dir.cwd().readFileAlloc(io, requests_file orelse return error.NoRequests, a, .limited(1 << 30));
    const Requests = struct { requests: []Request };
    const reqs = (try std.json.parseFromSliceLeaky(Requests, a, text, .{ .ignore_unknown_fields = true })).requests;
    var ln = dsv41.lanes.Lanes.init(gpa, m);
    defer ln.deinit();
    if (profile) {
        ln.prof = .{};
        m.prof = true;
        if (m.eh) |*eh| eh.io = io;
    }
    var step_ns: u64 = 0;
    const rows: u32 = dsv41.round.max_rows;
    var cfg = try lanes.Config.init(gpa, ln.facts(), rows, rows - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var core = lanes.Engine.init(gpa, &cfg, ln.backend(), clock.clock());
    defer core.deinit();

    const eos = [_]u32{@intCast(m.cfg.eos)};
    const jobs = try a.alloc(Job, reqs.len);
    var admitted: usize = 0;
    var finished: usize = 0;
    var all_equal = true;
    var tokens_out: usize = 0;
    const started = std.Io.Timestamp.now(io, .awake);
    while (finished < reqs.len) {
        // admit while a lane is free (the host's admission: one prompt pass at a time, before the rounds)
        while (admitted < reqs.len and core.activeCount() < parallel) {
            const r = &reqs[admitted];
            const j = &jobs[admitted];
            j.* = .{ .req = r, .index = admitted, .proposer = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 }), .stream = undefined, .began = std.Io.Timestamp.now(io, .awake) };
            j.stream = try lanes.Stream.init(gpa, .{
                .id = r.name,
                .prompt = r.prompt,
                .max_new = r.max_tokens,
                .eos = &eos,
                .sampling = if (r.temperature > 0) .{ .seed = r.seed orelse 0, .temperature = r.temperature, .top_k = r.top_k, .top_p = r.top_p } else null,
                .drafts = !serial,
                .proposer = j.proposer.proposer(),
            });
            admitted += 1;
            core.addStream(&j.stream) catch |err| {
                try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"error\": \"{s}\"}}\n", .{ j.index, r.name, @errorName(err) });
                try w.flush();
                return 1;
            };
            try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"prompt_tokens\": {d}, \"prefill_s\": {d:.2}}}\n", .{ j.index, r.name, r.prompt.len, seconds(j.began, io) });
            try w.flush();
        }
        const s0 = m.now();
        if (core.live.items.len > 0) core.step() catch |err| {
            try w.print("{{\"rank\": 0, \"step\": {d}, \"error\": \"{s}\"}}\n", .{ core.steps, @errorName(err) });
            try w.flush();
            return 1;
        };
        step_ns += m.now() - s0;
        // the finished streams' replies
        for (jobs[0..admitted]) |*j| {
            if (j.done or !j.stream.finished) continue;
            j.done = true;
            finished += 1;
            const reply = j.stream.emitted();
            tokens_out += reply.len;
            const sha = try tokenSha(a, reply);
            const equal = if (j.req.expect_sha) |e| std.mem.eql(u8, e, sha) else true;
            if (!equal) all_equal = false;
            try w.print("{{\"rank\": 0, \"request\": {d}, \"name\": \"{s}\", \"tokens\": {d}, \"reason\": \"{s}\", \"token_sha\": \"{s}\", \"expect_sha\": \"{s}\", \"equal\": {}, \"rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"min_rows\": {d}, \"seconds\": {d:.2}, \"reply\": [", .{ j.index, j.req.name, reply.len, j.stream.reason.name(), sha, j.req.expect_sha orelse "", equal, j.stream.rounds, j.stream.drafted, j.stream.accepted, j.stream.min_rows, seconds(j.began, io) });
            for (reply, 0..) |t, k| try w.print("{s}{d}", .{ if (k == 0) "" else ", ", t });
            try w.print("]}}\n", .{});
            try w.flush();
            j.stream.deinit(gpa);
            j.proposer.deinit();
        }
    }
    const total_s = seconds(started, io);
    if (ln.prof) |p| {
        const ms = struct {
            fn f(ns: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        try w.print("{{\"rank\": 0, \"profile\": {{\"rounds\": {d}, \"rows_a_round\": {d:.2}, \"step_ms\": {d:.2}, \"send_ms\": {d:.3}, \"enqueue_ms\": {d:.2}, \"forward_ms\": {d:.2}, \"absorb_ms\": {d:.2}, \"logits_ms\": {d:.2}, \"sample_ms\": {d:.2}, \"passes\": {d}, \"streams_a_pass\": {d:.2}, \"pass_ms\": {d:.2}, \"prefills\": {d}, \"prefill_ms\": {d:.1}}}}}\n", .{ p.rounds, ms(p.rows * 1000000, p.rounds), ms(step_ns, core.steps), ms(p.send, p.rounds), ms(p.enqueue, p.rounds), ms(p.forward, p.rounds), ms(p.absorb, p.rounds), ms(p.logits, p.rounds), ms(p.sample, p.rounds), p.passes, ms(p.pass_streams * 1000000, p.passes), ms(p.pass, p.passes), p.prefills, ms(p.prefill, p.prefills) });
        try w.flush();
        if (m.eh) |eh| {
            try w.print("{{\"rank\": 0, \"engram\": {{\"calls_a_round\": {d:.2}, \"hash_ms\": {d:.3}, \"read_ms\": {d:.3}, \"decode_ms\": {d:.3}, \"upload_ms\": {d:.3}}}}}\n", .{ ms(eh.calls * 1000000, p.rounds), ms(eh.t_hash, p.rounds), ms(eh.t_read, p.rounds), ms(eh.t_decode, p.rounds), ms(eh.t_upload, p.rounds) });
            try w.flush();
        }
    }
    try w.print("{{\"rank\": 0, \"requests\": {d}, \"tokens\": {d}, \"steps\": {d}, \"shared_rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"seconds\": {d:.2}, \"all_equal\": {}}}\n", .{ reqs.len, tokens_out, core.steps, core.shared_rounds, core.drafted, core.accepted, total_s, all_equal });
    try w.flush();
    return if (all_equal) 0 else 1;
}

fn seconds(t: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(t.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
}

/// The served server's token_sha: sha256 of the reply's ids joined by commas, its first 12 hex digits.
fn tokenSha(a: std.mem.Allocator, ids: []const u32) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    var buf: [16]u8 = undefined;
    for (ids, 0..) |t, k| {
        if (k > 0) try text.append(a, ',');
        try text.appendSlice(a, try std.fmt.bufPrint(&buf, "{d}", .{t}));
    }
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text.items, &h, .{});
    const hex = try std.fmt.allocPrint(a, "{x}", .{&h});
    return hex[0..12];
}
