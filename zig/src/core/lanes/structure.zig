//! Structured output in the round loop: a stream's grammar at its committed tokens, and each window's path under it.
const std = @import("std");
const grammar = @import("grammar");
const accept = @import("accept.zig");
const be = @import("backend.zig");
const Config = @import("config.zig").Config;
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;

/// Why a stream ends when its grammar refuses a token it must commit (a forced token, or a row that allows none).
pub const refused = "the reply's grammar refused its next token";

/// A request's grammar: the token table, its program (null: json_object), and the think end it starts after.
pub const Spec = struct {
    vocab: *const grammar.Vocab,
    program: ?*const grammar.Program = null,
    after: ?u32 = null,
};

/// A reply's grammar at its committed tokens, a small value that callers copy to try tokens without taking them.
pub const State = struct {
    vocab: *const grammar.Vocab,
    matcher: grammar.Matcher,
    /// The token the grammar waits for before it starts; null once it has started.
    after: ?u32 = null,

    pub fn init(spec: Spec) State {
        return .{ .vocab = spec.vocab, .matcher = .init(spec.program), .after = spec.after };
    }

    pub fn active(st: *const State) bool {
        return st.after == null;
    }

    /// Whether `token` may come next; before the grammar starts, every token may.
    pub fn allows(st: *const State, token: u32) bool {
        return st.after != null or st.vocab.allows(&st.matcher, token);
    }

    /// Takes a committed token, or returns false and stays as it was when the grammar refuses it.
    pub fn take(st: *State, token: u32) bool {
        if (st.after) |end| {
            if (token == end) st.after = null;
            return true;
        }
        if (!st.vocab.allows(&st.matcher, token)) return false;
        if (st.vocab.kind(token) == .text) _ = st.matcher.advance(st.vocab.piece(token));
        return true;
    }

    fn allow(st: *const State) be.Allow {
        return .{ .ctx = st, .ok = ok };
    }

    fn ok(ctx: *const anyopaque, token: u32) bool {
        const st: *const State = @ptrCast(@alignCast(ctx));
        return st.vocab.allows(&st.matcher, token);
    }
};

/// Whether the engine keeps streams inside a grammar: its backend redraws rows, and no round runs a step ahead.
pub fn served(b: be.Backend, cfg: *const Config) bool {
    return b.vtable.redraw != null and !cfg.pipelined;
}

fn greedy(s: *const Stream) bool {
    const sm = s.sampling orelse return true;
    return sm.temperature <= 0;
}

/// The grammar's pick at a row drawn `drawn`: an allowed greedy pick stays (the allowed argmax), others draw again.
fn pick(e: *Engine, s: *Stream, st: *const State, at: be.Redraw, drawn: u32) !u32 {
    if (!st.active()) return drawn;
    if (greedy(s) and st.allows(drawn)) return drawn;
    const redraw = e.backend.vtable.redraw orelse return error.NoRedraw;
    // no allowed token: the draw stays, and the commit refuses it and ends only this stream
    return try redraw(e.backend.ptr, at, st.allow()) orelse drawn;
}

/// The first token: the prompt row's draw, drawn again when the grammar refuses it; null with no active grammar.
pub fn first(e: *Engine, s: *Stream, drawn: u64, position: u64) !?u32 {
    const st = if (s.structure) |*st| st else return null;
    if (!st.active()) return null;
    return try pick(e, s, st, .{ .stream = s, .position = position }, try e.backend.read(drawn));
}

/// A verified window's accepted path and the token after its last row (`last`); `redrawn`: a pick left its draw.
pub const Path = struct { rows: []u32, last: u32, redrawn: bool = false };

/// accept.acceptPath over the grammar's picks, up to an end token (nothing past it commits); without a grammar, itself.
pub fn path(e: *Engine, s: *Stream, window: u32, sampled: []const u32, tokens: []const u32, parents: []const i32, positions: []const u64) !Path {
    const a = e.arena.allocator();
    if (s.structure == null) {
        const rows = try accept.acceptPath(a, tokens, parents, sampled);
        return .{ .rows = rows, .last = sampled[rows[rows.len - 1]] };
    }
    var st = s.structure.?;
    var rows: std.ArrayList(u32) = .empty;
    try rows.append(a, 0);
    var redrawn = false;
    while (true) {
        const at = rows.items[rows.items.len - 1];
        const want = try pick(e, s, &st, .{ .stream = s, .row = .{ .window = window, .row = at }, .position = positions[at] }, sampled[at]);
        redrawn = redrawn or want != sampled[at];
        if (s.isEos(want) or !st.allows(want)) return .{ .rows = rows.items, .last = want, .redrawn = redrawn };
        const next = for (parents, 0..) |q, row| {
            if (q == @as(i32, @intCast(at)) and tokens[row] == want) break row;
        } else return .{ .rows = rows.items, .last = want, .redrawn = redrawn };
        _ = st.take(want);
        try rows.append(a, @intCast(next));
    }
}

/// The token after a forced window: its last row's draw, under the grammar the forced tokens leave.
pub fn tail(e: *Engine, s: *Stream, window: u32, forced: []const u32, sampled: []const u32, positions: []const u64) !u32 {
    const last: u32 = @intCast(sampled.len - 1);
    var st = s.structure orelse return sampled[last];
    for (forced) |t| if (!st.take(t)) return sampled[last]; // the commit refuses that forced token
    return pick(e, s, &st, .{ .stream = s, .row = .{ .window = window, .row = last }, .position = positions[last] }, sampled[last]);
}
