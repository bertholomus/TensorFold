//! DeepSeek-V4.1-Flash behind TensorFold 1.0's lane core (core/lanes): the round loop picks each round's streams and
//! windows and keeps the accepted rows; this backend runs them on the port's model (model.zig) with the served lane's
//! arithmetic. Each stream has a pool slot (its window rings, positional stores and drafter rings) and an extent of
//! the shared compressed plane (first fit, chunk-aligned, as large as its prompt, its budget and a round's rows), and
//! its ids by position on both ranks (Engram's n-grams). verify is one round over every window (round.zig) with each
//! row drawn by the served sampler (sampling.zig) at its keyed position, and the drafting streams' window rows
//! absorbed into their drafter rings at once (the served eager absorb: the rows past the kept ones sit where the
//! drafter never reads, and the next window writes over them); draft is one batched drafter pass (draft.zig) for
//! every stream asked, the drafts held for the next round with their confidences, whose sigmoids are each draft's
//! chance of landing (probabilities: the round loop's allocation trims the windows by them). The caches are
//! positional, so keep only checks that a path is a prefix. Rank 0 runs the core; before each primitive it sends rank
//! 1 the primitive's inputs over the ranks' link, and rank 1 (follow) runs the same primitive: both ranks issue the
//! same kernels and collectives in the same order. The drafted reply is the serial one: every row is drawn with its
//! stream's key at its own position, whatever the round's composition.
const std = @import("std");
const lanes = @import("lanes");
const model = @import("model.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const sampling = @import("sampling.zig");
const link = @import("link.zig");

const be = lanes.backend;
const Model = model.Model;
const max_streams = model.max_streams;
const max_rows = round.max_rows;

/// A window of the pending row and the drafter's block: the widest one the gates checked row for row.
pub const window_rows = 6;

/// The primitives rank 0 sends rank 1 (each frame: kind u8, step u64, its fields little-endian).
pub const Kind = enum(u8) { fill_begin = 1, fill_chunk = 2, verify = 3, pass = 4, done = 9 };

/// A frame being written.
pub const Wire = struct {
    buf: std.ArrayList(u8) = .empty,

    pub fn begin(w: *Wire, gpa: std.mem.Allocator, kind: Kind, step: u64) !void {
        w.buf.clearRetainingCapacity();
        try w.int(gpa, u8, @backingInt(kind));
        try w.int(gpa, u64, step);
    }

    pub fn int(w: *Wire, gpa: std.mem.Allocator, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.buf.appendSlice(gpa, &b);
    }

    pub fn ids(w: *Wire, gpa: std.mem.Allocator, v: []const i32) !void {
        try w.int(gpa, u32, @intCast(v.len));
        try w.buf.ensureUnusedCapacity(gpa, v.len * 4);
        for (v) |x| try w.int(gpa, i32, x);
    }

    pub fn deinit(w: *Wire, gpa: std.mem.Allocator) void {
        w.buf.deinit(gpa);
    }
};

/// A frame being read.
pub const Reader = struct {
    b: []const u8,
    at: usize = 0,

    pub fn int(r: *Reader, comptime T: type) !T {
        const n = @sizeOf(T);
        if (r.at + n > r.b.len) return error.ShortFrame;
        const v = std.mem.readInt(T, r.b[r.at..][0..n], .little);
        r.at += n;
        return v;
    }

    /// A list of ids written by Wire.ids, into `out` (replacing it).
    pub fn ids(r: *Reader, gpa: std.mem.Allocator, out: *std.ArrayList(i32)) !void {
        const n = try r.int(u32);
        try out.resize(gpa, n);
        for (out.items) |*x| x.* = try r.int(i32);
    }
};

/// A window of a round as both ranks build it: its stream's slot and extent, its first position, whether its rows'
/// taps go into the drafter, its rows' ids.
pub const Win = struct { slot: usize, base: usize, end: usize, start: usize, absorb: bool, ids: []const i32 };

/// The rows of a round (round.Rows' slices point here).
pub const Built = struct {
    ids: [max_rows]i64 = undefined,
    pos: [max_rows]i64 = undefined,
    slots: [max_rows]i64 = undefined,
    bases: [max_rows]i64 = undefined,
    ends: [max_rows]i64 = undefined,
    wins: [max_streams]round.Window = undefined,
    absorb: [max_streams]bool = undefined,
    r: usize = 0,
    n: usize = 0,

    /// Each window's ids written at its positions in its slot's sequence (the rows past them dropped: the rejected
    /// rows of the last round), then the round's rows in window order.
    pub fn build(b: *Built, gpa: std.mem.Allocator, seqs: *[max_streams]std.ArrayList(i32), wins: []const Win) !round.Rows {
        if (wins.len == 0 or wins.len > max_streams) return error.BadRound;
        b.r = 0;
        b.n = wins.len;
        for (wins, 0..) |w, i| {
            if (w.slot >= max_streams or w.ids.len == 0) return error.BadRound;
            if (b.r + w.ids.len > max_rows) return error.WindowTooWide;
            const seq = &seqs[w.slot];
            if (seq.items.len < w.start) return error.SequenceGap;
            try seq.resize(gpa, w.start + w.ids.len);
            @memcpy(seq.items[w.start..], w.ids);
            for (w.ids, 0..) |id, j| {
                b.ids[b.r + j] = id;
                b.pos[b.r + j] = @intCast(w.start + j);
                b.slots[b.r + j] = @intCast(w.slot);
                b.bases[b.r + j] = @intCast(w.base);
                b.ends[b.r + j] = @intCast(w.end);
            }
            b.wins[i] = .{ .row = b.r, .n = w.ids.len, .seq = seq.items[0 .. w.start + w.ids.len] };
            b.absorb[i] = w.absorb;
            b.r += w.ids.len;
        }
        const R = b.r;
        return .{ .ids = b.ids[0..R], .pos = b.pos[0..R], .slots = b.slots[0..R], .bases = b.bases[0..R], .ends = b.ends[0..R], .windows = b.wins[0..b.n] };
    }
};

/// The first chunk-aligned extent of `size` positions in [0, cap) that overlaps none of `used`.
pub fn place(used: []const [2]usize, size: usize, cap: usize) ?usize {
    var base: usize = 0;
    while (base + size <= cap) {
        var clash: ?usize = null;
        for (used) |u| {
            if (base < u[1] and u[0] < base + size) clash = @max(clash orelse 0, u[1]);
        }
        const past = clash orelse return base;
        base = std.mem.alignForward(usize, past, model.chunk_rows);
    }
    return null;
}

/// The token a row's logits draw at absolute position `position`: greedy without sampling (or at temperature 0),
/// else the served sampler with the stream's key.
pub fn draw(row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
    const sm = s orelse return @intCast(sampling.argmax(row));
    if (sm.temperature <= 0) return @intCast(sampling.argmax(row));
    const t = try sampling.sampleRow(row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p });
    return @intCast(t);
}

/// Where rank 0's backend calls spend their time (ns, summed; --profile): a round's frame to rank 1, its forward
/// and absorb (synchronized apart), the logits' copy to the host and the draws; the drafter passes; the prefills.
pub const Profile = struct {
    send: u64 = 0,
    enqueue: u64 = 0,
    forward: u64 = 0,
    absorb: u64 = 0,
    logits: u64 = 0,
    sample: u64 = 0,
    pass: u64 = 0,
    prefill: u64 = 0,
    rounds: u64 = 0,
    rows: u64 = 0,
    passes: u64 = 0,
    pass_streams: u64 = 0,
    prefills: u64 = 0,
};

/// A stream's place in the pool and the drafts it holds.
const Lane = struct {
    slot: usize,
    base: usize,
    end: usize,
    first: u64 = 0, // the prompt's draw (a handle)
    held: [draft.max_block]u32 = undefined,
    confs: [draft.max_block]f32 = undefined,
    nheld: usize = 0,
};

extern "c" fn exp(x: f64) f64;

/// First tokens a handle names (each prompt's draw, read back right after its prefill).
const ring = 1024;

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    peer: ?link.Link, // rank 1 (null: one rank)
    step: u64 = 0,
    wire: Wire = .{},
    streams: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    used: [max_streams]bool = @splat(false),
    seqs: [max_streams]std.ArrayList(i32) = @splat(.empty),
    built: Built = .{},
    prof: ?Profile = null,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, m: *Model) Lanes {
        return .{ .gpa = gpa, .m = m, .peer = if (m.world > 1) m.peer() else null };
    }

    /// Tells rank 1 to stop, then frees the host state (the model is the caller's).
    pub fn deinit(self: *Lanes) void {
        if (self.peer != null) self.send(.done) catch {};
        self.streams.deinit(self.gpa);
        for (&self.seqs) |*q| q.deinit(self.gpa);
        self.wire.deinit(self.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .probabilities = probabilitiesFn,
            .release = releaseFn,
        } };
    }

    /// The facts the round loop reads: windows of the pending row and up to the drafter's block, several streams'
    /// windows in one forward of at most 16 rows (four streams), the drafter's chances, the served lane's round ms.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.m.drafting();
        return .{
            .exact_width = window_rows,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (self.m.dr) |dr| @intCast(dr.n) else 1,
            .window_costs = window_costs[0..],
            .mtp_step_ms = draft.draft_ms,
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = max_rows,
            .max_streams = max_streams,
            .shared_costs = shared_costs[0..],
            .draft_probabilities = drafting,
            .draft_streams = drafting,
        };
    }

    /// The served ROUND_MS table as the round loop's costs (it learns its own from the rounds it times).
    const window_costs = blk: {
        var out: [window_rows]lanes.config.Cost = undefined;
        for (&out, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = draft.round_ms[i] };
        break :blk out;
    };
    const shared_costs = blk: {
        var out: [max_rows]lanes.config.Cost = undefined;
        for (&out, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = draft.round_ms[i] };
        break :blk out;
    };

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The frame being built goes to rank 1 (after `begin` and the fields).
    fn flush(self: *Lanes) !void {
        if (self.peer) |p| try p.send(self.wire.buf.items);
    }

    fn begin(self: *Lanes, kind: Kind) !bool {
        if (self.peer == null) return false;
        try self.wire.begin(self.gpa, kind, self.step);
        self.step += 1;
        return true;
    }

    fn send(self: *Lanes, kind: Kind) !void {
        if (try self.begin(kind)) try self.flush();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// The lowest free slot, the first extent that fits, the prompt in chunks of the served size (at multiples of it:
    /// the chunks the gates checked; the host's chunk plan is not followed), the head's draw of the first token.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        const t0 = m.now();
        defer if (self.prof) |*p| {
            p.prefill += m.now() - t0;
            p.prefills += 1;
        };
        const ids = s.prompt();
        const len = ids.len;
        if (len == 0) return error.EmptyPrompt;
        const size = std.mem.alignForward(usize, len + s.max_new + max_rows + 2, model.chunk_rows);
        if (size > m.pool_cap) return error.PromptTooLong;
        if (self.streams.fetchRemove(s)) |old| self.used[old.value.slot] = false;
        const slot = std.mem.indexOfScalar(bool, &self.used, false) orelse return error.NoFreeSlot;
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        const base = place(taken[0..nt], size, m.pool_cap) orelse return error.ContextFull;
        try self.streams.put(gpa, s, .{ .slot = slot, .base = base, .end = base + size });
        self.used[slot] = true;
        errdefer {
            _ = self.streams.remove(s);
            self.used[slot] = false;
        }
        // a kept prompt state is not restored yet: the whole prompt runs
        if (s.reuse.saved != null) s.reuse_failed = true;
        s.cached = 0;
        const seq = &self.seqs[slot];
        try seq.resize(gpa, len);
        for (seq.items, ids) |*q, id| q.* = @intCast(id);
        if (try self.begin(.fill_begin)) {
            try self.wire.int(gpa, u32, @intCast(slot));
            try self.wire.int(gpa, u64, base);
            try self.wire.ids(gpa, seq.items);
            try self.flush();
        }
        try m.fillBegin(slot, base);
        const replay = len -| m.cfg.window;
        var start: usize = 0;
        var head = false;
        while (start < len) {
            if (start > 0 and s.isCancelled()) return error.Cancelled;
            const n = @min(model.chunk_rows, len - start);
            if (try self.begin(.fill_chunk)) {
                try self.wire.int(gpa, u64, start);
                try self.wire.int(gpa, u32, @intCast(n));
                try self.wire.int(gpa, u64, replay);
                try self.flush();
            }
            head = try m.fillChunk(seq.items[0 .. start + n], start, n, replay);
            start += n;
        }
        if (!head) return error.NoPromptLogits;
        const token = try draw(try m.promptLogitsHost(), len, s.sampling);
        self.streams.getPtr(s).?.first = self.take(token);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return l.first;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = .{ ptr, s, feed, position };
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Every window as one round at its stream's cache length: the pending token, the held drafts (or the host's), each
    /// row drawn at its keyed position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (windows.len > max_streams) return error.WindowTooWide;
        var ids: [max_streams][max_rows]i32 = undefined;
        var wins: [max_streams]Win = undefined;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = self.streams.getPtr(w.stream) orelse return error.UnknownStream;
            const rows = w.rows();
            if (rows > window_rows) return error.WindowTooWide;
            const start: usize = @intCast(w.stream.cache_len);
            for (w.positions, 0..) |p, r| if (p != start + 1 + r) return error.PositionMismatch;
            if (w.held > l.nheld) return error.HeldMismatch;
            ids[k][0] = @intCast(w.pending);
            for (l.held[0..w.held], 0..) |t, j| ids[k][1 + j] = @intCast(t);
            for (w.tokens, 0..) |t, j| ids[k][1 + w.held + j] = @intCast(t);
            wins[k] = .{ .slot = l.slot, .base = l.base, .end = l.end, .start = start, .absorb = m.drafting() and w.stream.drafts, .ids = ids[k][0..rows] };
        }
        const t0 = m.now();
        if (try self.begin(.verify)) {
            try self.wire.int(gpa, u32, @intCast(windows.len));
            for (wins[0..windows.len]) |w| {
                try self.wire.int(gpa, u32, @intCast(w.slot));
                try self.wire.int(gpa, u64, w.base);
                try self.wire.int(gpa, u64, w.end);
                try self.wire.int(gpa, u64, w.start);
                try self.wire.int(gpa, u8, @intFromBool(w.absorb));
                try self.wire.ids(gpa, w.ids);
            }
            try self.flush();
        }
        const t1 = m.now();
        const rows = try self.built.build(gpa, &self.seqs, wins[0..windows.len]);
        try m.verify(rows, self.built.absorb[0..windows.len]);
        const t2 = m.now();
        const logits = try m.roundLogits(rows.ids.len);
        const t3 = m.now();
        defer if (self.prof) |*p| {
            p.send += t1 - t0;
            p.enqueue += m.t_enqueue;
            p.forward += m.t_forward;
            p.absorb += m.t_absorb;
            p.logits += t3 - t2;
            p.sample += m.now() - t3;
            p.rounds += 1;
            p.rows += rows.ids.len;
        };
        const V = m.vocab;
        for (windows, out, 0..) |w, *o, k| {
            const row0 = self.built.wins[k].row;
            for (0..w.rows()) |r| o.sampled[r] = try draw(logits[(row0 + r) * V ..][0..V], w.positions[r], w.stream.sampling);
            const l = self.streams.getPtr(w.stream).?;
            @memcpy(o.drafts[0..w.held], l.held[0..w.held]);
            @memcpy(o.drafts[w.held..][0..w.tokens.len], w.tokens);
            l.nheld = 0;
        }
    }

    /// The caches are positional: the next window starts at the stream's new length and writes over the rows past it.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |_, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        }
    }

    /// One drafter pass for every stream asked: its pending token at its new length, `depth` drafts held.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (m.dr == null) return error.NoDraftHead;
        const block = m.dr.?.n;
        var tokens: [max_streams]i64 = undefined;
        var q0: [max_streams]i64 = undefined;
        var slots: [max_streams]i64 = undefined;
        var depths: [max_streams]usize = undefined;
        var asked: [max_streams]*Lane = undefined;
        var n: usize = 0;
        var steps: usize = 0;
        for (requests) |r| {
            if (r.lanes != null or r.early) return error.TreesNotBuilt;
            const l = self.streams.getPtr(r.stream) orelse return error.UnknownStream;
            l.nheld = 0;
            if (r.depth == 0) continue;
            if (n == max_streams) return error.TooManyStreams;
            if (r.position != r.stream.cache_len + 1) return error.PositionMismatch;
            const token: u32 = if (r.rows == null) self.value(r.first orelse return error.NoFirstToken) else (if (r.follow.len > 0) r.follow[r.follow.len - 1] else return error.NoFollowToken);
            tokens[n] = token;
            q0[n] = @intCast(r.position - 1);
            slots[n] = @intCast(l.slot);
            depths[n] = @min(r.depth, block);
            asked[n] = l;
            steps = @max(steps, depths[n]);
            n += 1;
        }
        if (n == 0) return;
        if (try self.begin(.pass)) {
            try self.wire.int(gpa, u32, @intCast(n));
            try self.wire.int(gpa, u32, @intCast(steps));
            for (0..n) |i| {
                try self.wire.int(gpa, i64, tokens[i]);
                try self.wire.int(gpa, i64, q0[i]);
                try self.wire.int(gpa, i64, slots[i]);
            }
            try self.flush();
        }
        const t0 = m.now();
        try m.pass(tokens[0..n], q0[0..n], slots[0..n], steps);
        if (self.prof) |*p| {
            p.pass += m.now() - t0;
            p.passes += 1;
            p.pass_streams += n;
        }
        const d = &m.dr.?;
        for (asked[0..n], depths[0..n], 0..) |l, k, i| {
            for (0..k) |j| {
                l.held[j] = @intCast(d.drafts[i][j]);
                l.confs[j] = d.confs[i][j];
            }
            l.nheld = k;
        }
    }

    /// The confidence head's chance that each held draft lands (sigmoid of its confidence: multi.py's conf rule
    /// counts them as the expected kept drafts).
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return false;
        if (out.len > l.nheld) return false;
        for (out, l.confs[0..out.len]) |*o, cf| o.* = 1.0 / (1.0 + exp(-@as(f64, cf)));
        return true;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.streams.fetchRemove(s) orelse return;
        self.used[kv.value.slot] = false;
    }
};

/// Rank 1: the primitives rank 0 sends, run in order until it says done.
pub fn follow(gpa: std.mem.Allocator, m: *Model) !void {
    const peer = m.peer();
    const buf = try gpa.alloc(u8, (m.pool_cap + 1) * 4 + (1 << 16));
    defer gpa.free(buf);
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    var win_ids: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&win_ids) |*q| q.deinit(gpa);
    var built: Built = .{};
    var step: u64 = 0;
    var slot: usize = 0;
    while (true) {
        var r: Reader = .{ .b = try peer.recv(buf) };
        const kind: Kind = switch (try r.int(u8)) {
            1 => .fill_begin,
            2 => .fill_chunk,
            3 => .verify,
            4 => .pass,
            9 => .done,
            else => return error.BadFrame,
        };
        if (try r.int(u64) != step) return error.OutOfStep;
        step += 1;
        switch (kind) {
            .fill_begin => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                try r.ids(gpa, &seqs[slot]);
                try m.fillBegin(slot, @intCast(base));
            },
            .fill_chunk => {
                const start: usize = @intCast(try r.int(u64));
                const n: usize = try r.int(u32);
                const replay: usize = @intCast(try r.int(u64));
                if (start + n > seqs[slot].items.len) return error.BadChunk;
                _ = try m.fillChunk(seqs[slot].items[0 .. start + n], start, n, replay);
            },
            .verify => {
                const nw = try r.int(u32);
                if (nw == 0 or nw > max_streams) return error.BadFrame;
                var wins: [max_streams]Win = undefined;
                for (0..nw) |k| {
                    const s: usize = try r.int(u32);
                    const base: usize = @intCast(try r.int(u64));
                    const end: usize = @intCast(try r.int(u64));
                    const start: usize = @intCast(try r.int(u64));
                    const absorb = try r.int(u8) != 0;
                    try r.ids(gpa, &win_ids[k]);
                    wins[k] = .{ .slot = s, .base = base, .end = end, .start = start, .absorb = absorb, .ids = win_ids[k].items };
                }
                const rows = try built.build(gpa, &seqs, wins[0..nw]);
                try m.verify(rows, built.absorb[0..nw]);
            },
            .pass => {
                const n = try r.int(u32);
                const steps = try r.int(u32);
                if (n == 0 or n > max_streams) return error.BadFrame;
                var tokens: [max_streams]i64 = undefined;
                var q0: [max_streams]i64 = undefined;
                var slots: [max_streams]i64 = undefined;
                for (0..n) |i| {
                    tokens[i] = try r.int(i64);
                    q0[i] = try r.int(i64);
                    slots[i] = try r.int(i64);
                }
                try m.pass(tokens[0..n], q0[0..n], slots[0..n], steps);
            },
            .done => return,
        }
    }
}

test "extents take the first chunk-aligned gap" {
    const used = [_][2]usize{ .{ 0, 4096 }, .{ 6144, 8192 } };
    try std.testing.expectEqual(@as(?usize, 4096), place(&used, 2048, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 8192), place(&used, 4096, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 0), place(&.{}, 2048, 2048));
    try std.testing.expectEqual(@as(?usize, null), place(&used, 4096, 10240));
}

test "a frame's fields read back as written" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .verify, 7);
    try w.int(gpa, u64, 1 << 40);
    try w.ids(gpa, &.{ 1, -2, 129279 });
    var r: Reader = .{ .b = w.buf.items };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.verify)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 7), try r.int(u64));
    try std.testing.expectEqual(@as(u64, 1 << 40), try r.int(u64));
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    try r.ids(gpa, &got);
    try std.testing.expectEqualSlices(i32, &.{ 1, -2, 129279 }, got.items);
    try std.testing.expectError(error.ShortFrame, r.int(u8));
}

test "a round's rows: each window at its positions, its ids written over the rejected rows" {
    const gpa = std.testing.allocator;
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    try seqs[0].appendSlice(gpa, &.{ 10, 11, 12, 13, 99, 98 }); // positions 4.. hold a rejected window
    try seqs[2].appendSlice(gpa, &.{ 20, 21 });
    var b: Built = .{};
    const rows = try b.build(gpa, &seqs, &.{
        .{ .slot = 0, .base = 0, .end = 4096, .start = 4, .absorb = true, .ids = &.{ 14, 15 } },
        .{ .slot = 2, .base = 4096, .end = 8192, .start = 2, .absorb = false, .ids = &.{22} },
    });
    try std.testing.expectEqualSlices(i64, &.{ 14, 15, 22 }, rows.ids);
    try std.testing.expectEqualSlices(i64, &.{ 4, 5, 2 }, rows.pos);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2 }, rows.slots.?);
    try std.testing.expectEqualSlices(i64, &.{ 4096, 4096, 8192 }, rows.ends.?);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 14, 15 }, rows.windows.?[0].seq);
    try std.testing.expectEqualSlices(i32, &.{ 20, 21, 22 }, rows.windows.?[1].seq);
    try std.testing.expectEqual(@as(usize, 2), rows.windows.?[1].row);
    try std.testing.expectError(error.SequenceGap, b.build(gpa, &seqs, &.{.{ .slot = 1, .base = 0, .end = 2048, .start = 3, .absorb = false, .ids = &.{1} }}));
}

test "greedy draws take the first largest logit" {
    const row = [_]f32{ 0.5, 2.0, -1.0, 2.0 };
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, null));
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, .{ .seed = 3, .temperature = 0 }));
}
