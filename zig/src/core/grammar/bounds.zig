//! Numeric bounds as xgrammar 0.2.8 compiles them: canonical numbers whose value lies in range, as an automaton.
const std = @import("std");
const keys = @import("schema_keys.zig");
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Allocator = std.mem.Allocator;

/// The bytes a bounded number reads: the ten digits, '.' and '-'.
pub const classes = 12;
const dot = 10;
const minus = 11;
/// No state: the byte cannot come next.
pub const dead: u32 = std.math.maxInt(u32);
/// The keywords this file enforces.
pub const words = [_][]const u8{ "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum" };
/// The most fraction digits a bounded number takes (xgrammar's precision).
const precision = 6;
/// xgrammar parses a range's whole parts into int64 and caps them there (ParseIntCapped).
const int64_max = "9223372036854775807";

/// An automaton state: the next state for each byte class, and whether the number may end here.
pub const State = struct { next: [classes]u32, accept: bool };

/// The most states a program's automata may take, so that a 1 MiB schema cannot ask for gigabytes.
pub const max_states = 1 << 16;
pub const too_many = std.fmt.comptimePrint("the numeric bounds need more than {d} automaton states", .{max_states});

/// A program's automata; equal bounds share one.
pub const Table = struct {
    states: std.ArrayList(State) = .empty,
    built: std.StringHashMapUnmanaged(u32) = .empty,
};

/// The class of a byte a bounded number reads, or null.
pub fn class(b: u8) ?u8 {
    return switch (b) {
        '0'...'9' => b - '0',
        '.' => dot,
        '-' => minus,
        else => null,
    };
}

/// The next state after `b` in `table` from `s`, or null.
pub fn step(table: []const State, s: u32, b: u8) ?u32 {
    const n = table[s].next[class(b) orelse return null];
    return if (n == dead) null else n;
}

pub const Error = error{ OutOfMemory, Refused };

/// The first bound keyword of `s`, if any.
pub fn given(s: ObjectMap) ?[]const u8 {
    for (s.keys()) |k| for (words) |w| if (std.mem.eql(u8, k, w)) return k;
    return null;
}

/// xgrammar applies bounds only where `type` names integer or number and no $ref, const, enum, anyOf, oneOf or allOf.
pub fn placed(s: ObjectMap) bool {
    for ([_][]const u8{ "$ref", "const", "enum", "anyOf", "oneOf", "allOf" }) |k| if (s.contains(k)) return false;
    const t = s.get("type") orelse return false;
    switch (t) {
        .string => |name| return numeric(name),
        .array => |names| for (names.items) |n| if (n == .string and numeric(n.string)) return true,
        else => {},
    }
    return false;
}

fn numeric(name: []const u8) bool {
    return std.mem.eql(u8, name, "integer") or std.mem.eql(u8, name, "number");
}

/// The automaton of `s`'s bounds for an integer or a number in `table`, and its start. Refused: `why` says why.
pub fn compile(a: Allocator, table: *Table, s: ObjectMap, whole: bool, why: *[]const u8) Error!u32 {
    var r: Reader = .{ .a = a, .why = why };
    const spec = if (whole) try r.integerSpec(s) else try r.numberSpec(s);
    const start = shared(a, table, spec) catch |e| return if (e == error.Refused) r.fail(too_many, .{}) else e;
    return start orelse r.fail("no number with at most 6 decimals lies in the range: xgrammar 0.2.8 takes an empty value there", .{});
}

/// The automaton of `spec`, built once per program.
fn shared(a: Allocator, table: *Table, spec: Spec) Error!?u32 {
    var key: std.ArrayList(u8) = .empty;
    try key.print(a, "{} {} {}", .{ spec.frac, spec.zero, spec.neg_zero });
    for ([_]?Side{ spec.neg, spec.pos }) |side| if (side) |d| {
        for ([_]?Bound{ d.lo, d.hi }) |bound| if (bound) |b| try key.print(a, " {s}.{s}{}", .{ b.int, b.frac, b.strict }) else try key.appendSlice(a, " -");
    } else try key.appendSlice(a, " none");
    const got = try table.built.getOrPut(a, key.items);
    if (!got.found_existing) got.value_ptr.* = try build(a, table, spec) orelse dead;
    return if (got.value_ptr.* == dead) null else got.value_ptr.*;
}

/// A decimal bound: whole digits (no leading 0 but "0"), up to 6 fraction digits without a trailing 0, strictness.
const Bound = struct { int: []const u8, frac: []const u8 = "", strict: bool = false };
/// Magnitudes from lo to hi (none: no end), never 0: xgrammar's PositiveRangeParts.
const Side = struct { lo: Bound, hi: ?Bound = null };
/// A bounded integer or number: magnitudes on each side, and which spellings of 0 it takes.
const Spec = struct {
    frac: bool,
    neg: ?Side = null,
    pos: ?Side = null,
    /// "0" (and for numbers "0.0" to "0.000000")
    zero: bool = false,
    /// "-0.0" to "-0.000000"; never "-0"
    neg_zero: bool = false,
};

/// Reads bounds as xgrammar's ParseInteger and ParseNumber read them.
const Reader = struct {
    a: Allocator,
    why: *[]const u8,

    fn fail(r: *Reader, comptime fmt: []const u8, args: anytype) Error {
        r.why.* = try std.fmt.allocPrint(r.a, fmt, args);
        return error.Refused;
    }

    fn float(r: *Reader, v: Value, word: []const u8) Error!f64 {
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            .number_string => |t| std.fmt.parseFloat(f64, t) catch return r.fail("{s} must be a number", .{word}),
            else => return r.fail("{s} must be a number", .{word}),
        };
    }

    /// An integer bound: an int64, or a double that is whole and inside int64 (else xgrammar fails).
    fn int(r: *Reader, s: ObjectMap, comptime word: []const u8) Error!?i64 {
        const v = s.get(word) orelse return null;
        const x = try r.float(v, word);
        if (keys.int64(v)) |i| return i;
        if (x != @floor(x)) return r.fail(word ++ " must be a whole number for type integer", .{});
        if (!(x > -0x1p63 and x < 0x1p63)) return r.fail(word ++ " must lie between -2^63 and 2^63 for type integer", .{});
        return @intFromFloat(x);
    }

    fn integerSpec(r: *Reader, s: ObjectMap) Error!Spec {
        const max = std.math.maxInt(i64);
        const min = std.math.minInt(i64);
        var lo = try r.int(s, "minimum");
        var hi = try r.int(s, "maximum");
        if (try r.int(s, "exclusiveMinimum")) |x| {
            if (x == max) return r.fail("exclusiveMinimum leaves no int64", .{});
            lo = if (lo) |l| @max(l, x + 1) else x + 1;
        }
        if (try r.int(s, "exclusiveMaximum")) |x| {
            if (x == min) return r.fail("exclusiveMaximum leaves no int64", .{});
            hi = if (hi) |h| @min(h, x - 1) else x - 1;
        }
        if (lo != null and hi != null and lo.? > hi.?) return r.fail("the range is empty: its lower bound is above its upper one", .{});
        var spec: Spec = .{ .frac = false };
        spec.zero = (lo == null or lo.? <= 0) and (hi == null or hi.? >= 0);
        if (lo == null or lo.? < 0) spec.neg = .{
            .lo = .{ .int = try r.digits(if (hi != null and hi.? < 0) @abs(hi.?) else 1) },
            .hi = if (lo) |l| .{ .int = try r.digits(@abs(l)) } else null,
        };
        if (hi == null or hi.? > 0) spec.pos = .{
            .lo = .{ .int = try r.digits(if (lo != null and lo.? > 0) @intCast(lo.?) else 1) },
            .hi = if (hi) |h| .{ .int = try r.digits(@intCast(h)) } else null,
        };
        return spec;
    }

    fn digits(r: *Reader, n: u64) Error![]const u8 {
        return std.fmt.allocPrint(r.a, "{d}", .{n});
    }

    fn num(r: *Reader, s: ObjectMap, word: []const u8) Error!?f64 {
        const x = try r.float(s.get(word) orelse return null, word);
        if (!std.math.isFinite(x)) return r.fail("{s} is past the largest double", .{word});
        return x;
    }

    fn numberSpec(r: *Reader, s: ObjectMap) Error!Spec {
        const min = try r.num(s, "minimum");
        const max = try r.num(s, "maximum");
        const xmin = try r.num(s, "exclusiveMinimum");
        const xmax = try r.num(s, "exclusiveMaximum");
        if ((min != null and max != null and min.? > max.?) or (min != null and xmax != null and min.? >= xmax.?) or
            (xmin != null and max != null and xmin.? >= max.?) or (xmin != null and xmax != null and xmin.? >= xmax.?))
            return r.fail("the range is empty: its lower bound is not below its upper one", .{});
        // GenerateNumber: the larger lower bound and the smaller upper one win, an exclusive one on a tie
        var start = min;
        var xs = false;
        if (xmin) |x| if (start == null or x >= start.?) {
            start = x;
            xs = true;
        };
        var end = max;
        var xe = false;
        if (xmax) |x| if (end == null or x <= end.?) {
            end = x;
            xe = true;
        };
        // FloatRangeRegex: negatives are the negated range's magnitudes, then 0, then positives
        var spec: Spec = .{ .frac = true };
        const negative = start == null or start.? < 0;
        spec.zero = (start == null or start.? < 0 or (start.? == 0 and !xs)) and (end == null or end.? > 0 or (end.? == 0 and !xe));
        spec.neg_zero = spec.zero and negative;
        if (negative) spec.neg = .{
            .lo = if (end != null and end.? < 0) try r.grid(-end.?, true, xe) else .{ .int = "0", .strict = true },
            .hi = if (start) |x| try r.grid(-x, false, xs) else null,
        };
        if (end == null or end.? > 0) spec.pos = .{
            .lo = if (start != null and start.? > 0) try r.grid(start.?, true, xs) else .{ .int = "0", .strict = true },
            .hi = if (end) |x| try r.grid(x, false, xe) else null,
        };
        return spec;
    }

    /// RoundBoundToGrid for `v` > 0: its 6-decimal rounding if that reads back as `v`, else the grid point inside.
    fn grid(r: *Reader, v: f64, lower: bool, strict: bool) Error!Bound {
        const b = try fixed(r.a, v);
        var text: std.ArrayList(u8) = .empty;
        try text.appendSlice(r.a, b.int);
        if (b.frac.len > 0) try text.print(r.a, ".{s}", .{b.frac});
        const back = std.fmt.parseFloat(f64, text.items) catch unreachable;
        if (back == v) return .{ .int = b.int, .frac = b.frac, .strict = strict };
        if (lower and back < v) return adjust(r.a, b, true);
        if (!lower and back > v) return adjust(r.a, b, false);
        return b;
    }
};

/// FormatFloat for `v` > 0: whole numbers below 2^63 exactly, else std::fixed to 6 places (exact, ties to even).
fn fixed(a: Allocator, v: f64) Allocator.Error!Bound {
    if (v < 0x1p63 and v == @floor(v)) return .{ .int = try std.fmt.allocPrint(a, "{d}", .{@as(u64, @intFromFloat(v))}) };
    const bits: u64 = @bitCast(v);
    const biased: u11 = @intCast(bits >> 52);
    var m: u64 = bits & ((1 << 52) - 1);
    var e: i32 = @as(i32, biased) - 1075;
    if (biased == 0) e = -1074 else m |= 1 << 52;
    if (e >= 0) return .{ .int = try shifted(a, m, @intCast(e)) };
    const s: u32 = @intCast(-e);
    var whole: u64 = if (s >= 64) 0 else m >> @intCast(s);
    const part: u128 = if (s >= 64) m else m - (whole << @intCast(s));
    const scaled = part * 1_000_000;
    var q: u64 = 0;
    if (s < 127) {
        q = @intCast(scaled >> @intCast(s));
        const left = scaled - (@as(u128, q) << @intCast(s));
        const half = @as(u128, 1) << @intCast(s - 1);
        if (left > half or (left == half and q % 2 == 1)) q += 1;
        if (q == 1_000_000) {
            whole += 1;
            q = 0;
        }
    }
    var frac = try std.fmt.allocPrint(a, "{d:0>6}", .{q});
    while (frac.len > 0 and frac[frac.len - 1] == '0') frac = frac[0 .. frac.len - 1];
    return .{ .int = try std.fmt.allocPrint(a, "{d}", .{whole}), .frac = frac };
}

/// The decimal digits of m * 2^e.
fn shifted(a: Allocator, m: u64, e: u32) Allocator.Error![]const u8 {
    var ds: std.ArrayList(u8) = .empty; // least significant first
    var n = m;
    while (n > 0) : (n /= 10) try ds.append(a, @intCast(n % 10));
    for (0..e) |_| {
        var carry: u8 = 0;
        for (ds.items) |*d| {
            const x = d.* * 2 + carry;
            d.* = x % 10;
            carry = x / 10;
        }
        if (carry > 0) try ds.append(a, carry);
    }
    const out = try a.alloc(u8, ds.items.len);
    for (out, 0..) |*o, i| o.* = '0' + ds.items[ds.items.len - 1 - i];
    return out;
}

/// AdjustGrid: one 10^-6 step up or down from `b`, canonical; the bound becomes inclusive.
fn adjust(a: Allocator, b: Bound, up: bool) Allocator.Error!Bound {
    var num: std.ArrayList(u8) = .empty;
    try num.appendSlice(a, b.int);
    try num.appendSlice(a, b.frac);
    try num.appendNTimes(a, '0', precision - b.frac.len);
    var i = num.items.len;
    const wrap: u8 = if (up) '9' else '0';
    while (i > 0 and num.items[i - 1] == wrap) : (i -= 1) num.items[i - 1] = if (up) '0' else '9';
    if (i == 0) {
        if (up) try num.insert(a, 0, '1') else @memset(num.items, '0');
    } else num.items[i - 1] = if (up) num.items[i - 1] + 1 else num.items[i - 1] - 1;
    while (num.items.len <= precision) try num.insert(a, 0, '0');
    const cut = num.items.len - precision;
    var int = num.items[0..cut];
    while (int.len > 1 and int[0] == '0') int = int[1..];
    var frac = num.items[cut..];
    while (frac.len > 0 and frac[frac.len - 1] == '0') frac = frac[0 .. frac.len - 1];
    return .{ .int = int, .frac = frac };
}

const Ord = enum(u2) { lt, eq, gt };

fn order(x: u8, y: u8) Ord {
    return if (x < y) .lt else if (x > y) .gt else .eq;
}

/// Two whole parts without leading zeros, by value.
fn below(x: []const u8, y: []const u8) bool {
    if (x.len != y.len) return x.len < y.len;
    return std.mem.order(u8, x, y) == .lt;
}

const Phase = enum(u3) { start, minus, int, zero, dot, frac };

/// What a prefix of a bounded number decides: the automaton's state before trimming.
const Key = struct {
    phase: Phase = .start,
    neg: bool = false,
    /// int, zero: whole digits so far (saturating), then their order against the side's lo, hi and int64 whole parts
    k: u16 = 0,
    ints: [3]Ord = @splat(.eq),
    /// dot, frac: the whole part is lo's, is hi's, lies strictly between them, or is a 0 that may stay 0
    low: bool = false,
    high: bool = false,
    mid: bool = false,
    zero: bool = false,
    /// frac: digits so far, their order against lo's and hi's fraction, and whether one was not 0
    j: u8 = 0,
    fracs: [2]Ord = @splat(.eq),
    nonzero: bool = false,
};

const Builder = struct {
    a: Allocator,
    spec: Spec,
    keys: std.ArrayList(Key) = .empty,
    ids: std.AutoHashMapUnmanaged(Key, u32) = .empty,
    rows: std.ArrayList([classes]u32) = .empty,
    /// States the table has room for: the build stops past them.
    room: usize,

    fn id(bd: *Builder, k: Key) Error!u32 {
        const got = try bd.ids.getOrPut(bd.a, k);
        if (!got.found_existing) {
            if (bd.keys.items.len == bd.room) return error.Refused;
            got.value_ptr.* = @intCast(bd.keys.items.len);
            try bd.keys.append(bd.a, k);
        }
        return got.value_ptr.*;
    }

    fn side(bd: *const Builder, neg: bool) ?Side {
        return if (neg) bd.spec.neg else bd.spec.pos;
    }

    /// The whole parts a number's whole part is ordered against: lo's, hi's, and int64's where xgrammar caps.
    fn wholes(bd: *const Builder, s: Side) [3][]const u8 {
        const capped = bd.spec.frac and s.hi != null;
        return .{ s.lo.int, if (s.hi) |h| h.int else "", if (capped) int64_max else "" };
    }

    fn next(bd: *const Builder, k: Key, c: u8) ?Key {
        const digit = c < 10;
        return switch (k.phase) {
            .start => if (c == minus) .{ .phase = .minus, .neg = true } else if (digit) bd.more(.{ .phase = if (c == 0) .zero else .int }, c) else null,
            .minus => if (digit) bd.more(.{ .phase = if (c == 0) .zero else .int, .neg = true }, c) else null,
            .int => if (digit) bd.more(k, c) else if (c == dot) bd.whole(k) else null,
            .zero => if (c == dot) bd.whole(k) else null,
            .dot, .frac => if (digit and k.j < precision) bd.fraction(k, c) else null,
        };
    }

    /// A whole digit: its order against each bound's digit at the same place, while the prefixes are equal.
    fn more(bd: *const Builder, k: Key, d: u8) ?Key {
        var n = k;
        const s = bd.side(k.neg) orelse return if (k.phase == .zero) n else null;
        const ws = bd.wholes(s);
        var limit: usize = 0;
        for (ws, &n.ints) |w, *o| {
            limit = @max(limit, w.len + 1);
            if (k.k < w.len) {
                if (o.* == .eq) o.* = order(d, w[k.k] - '0');
            } else o.* = .eq; // longer than this bound: its length decides
        }
        n.k = @intCast(@min(k.k + 1, limit));
        return n;
    }

    fn final(k: Key, w: []const u8, i: usize) Ord {
        if (k.k != w.len) return if (k.k < w.len) .lt else .gt;
        return k.ints[i];
    }

    const Whole = struct { low: bool, high: bool, mid: bool };

    /// The whole part is lo's, is hi's, or lies strictly between (below xgrammar's int64 cap, for numbers).
    fn classify(bd: *const Builder, s: Side, k: Key) Whole {
        const ws = bd.wholes(s);
        const lo = final(k, ws[0], 0);
        const hi: Ord = if (s.hi != null) final(k, ws[1], 1) else .lt;
        var mid = lo == .gt and hi == .lt;
        if (bd.spec.frac) mid = mid and below(s.lo.int, int64_max) and (s.hi == null or final(k, ws[2], 2) == .lt);
        return .{ .low = lo == .eq, .high = s.hi != null and hi == .eq, .mid = mid };
    }

    /// The whole part ends at '.': what the fraction must satisfy.
    fn whole(bd: *const Builder, k: Key) ?Key {
        if (!bd.spec.frac) return null;
        var n: Key = .{ .phase = .dot, .neg = k.neg };
        n.zero = k.phase == .zero and (if (k.neg) bd.spec.neg_zero else bd.spec.zero);
        if (bd.side(k.neg)) |s| {
            const w = bd.classify(s, k);
            n.low = w.low;
            n.high = w.high;
            n.mid = w.mid;
        }
        return if (n.low or n.high or n.mid or n.zero) n else null;
    }

    fn fraction(bd: *const Builder, k: Key, d: u8) Key {
        var n = k;
        n.phase = .frac;
        n.j = k.j + 1;
        n.nonzero = k.nonzero or d != 0;
        const s = bd.side(k.neg) orelse return n;
        const fs = [2][]const u8{ s.lo.frac, if (s.hi) |h| h.frac else "" };
        for (fs, &n.fracs, [2]bool{ k.low, k.high }) |f, *o, used| {
            if (used and o.* == .eq) o.* = order(d, if (k.j < f.len) f[k.j] - '0' else 0);
        }
        return n;
    }

    fn accept(bd: *const Builder, k: Key) bool {
        switch (k.phase) {
            .start, .minus, .dot => return false,
            .int, .zero => {
                if (k.phase == .zero and !k.neg and bd.spec.zero) return true;
                const s = bd.side(k.neg) orelse return false;
                const w = bd.classify(s, k);
                return within(s, w.low, w.high, w.mid, @splat(.eq), 0);
            },
            .frac => {
                if (k.zero and !k.nonzero) return true;
                const s = bd.side(k.neg) orelse return false;
                return within(s, k.low, k.high, k.mid, k.fracs, k.j);
            },
        }
    }
};

/// The order of a fraction of `j` digits against `f` from their digits' order: a shorter equal prefix lies below.
fn rest(o: Ord, j: u8, f: []const u8) Ord {
    return if (o == .eq and j < f.len) .lt else o;
}

/// A magnitude with this whole part and fraction lies in the side's range.
fn within(s: Side, low: bool, high: bool, mid: bool, fracs: [2]Ord, j: u8) bool {
    if (mid) return true;
    // PositiveRangeParts treats a lower bound of 0 as strict: 0 itself has its own spellings
    const zero = std.mem.eql(u8, s.lo.int, "0") and s.lo.frac.len == 0;
    const lo = rest(fracs[0], j, s.lo.frac);
    const ok_low = if (s.lo.strict or zero) lo == .gt else lo != .lt;
    var ok_high = true;
    if (s.hi) |h| {
        const hi = rest(fracs[1], j, h.frac);
        ok_high = if (h.strict) hi == .lt else hi != .gt;
    }
    return (low and ok_low and (!high or ok_high)) or (high and !low and ok_high);
}

/// The automaton of `spec` without dead states, appended to `table`; its start, or null when it takes no number.
fn build(a: Allocator, table: *Table, spec: Spec) Error!?u32 {
    const states = &table.states;
    var bd: Builder = .{ .a = a, .spec = spec, .room = max_states -| states.items.len };
    _ = try bd.id(.{});
    var i: usize = 0;
    while (i < bd.keys.items.len) : (i += 1) {
        var row: [classes]u32 = @splat(dead);
        const k = bd.keys.items[i];
        for (&row, 0..) |*to, c| if (bd.next(k, @intCast(c))) |n| {
            to.* = try bd.id(n);
        };
        try bd.rows.append(a, row);
    }
    // a state lives when an accepting state can follow it; states mostly point forward, so sweep backward
    const live = try a.alloc(bool, bd.keys.items.len);
    for (live, bd.keys.items) |*l, k| l.* = bd.accept(k);
    var changed = true;
    while (changed) {
        changed = false;
        var j = live.len;
        while (j > 0) {
            j -= 1;
            if (live[j]) continue;
            for (bd.rows.items[j]) |t| if (t != dead and live[t]) {
                live[j] = true;
                changed = true;
                break;
            };
        }
    }
    if (!live[0]) return null;
    const at = try a.alloc(u32, live.len);
    var n: u32 = @intCast(states.items.len);
    for (live, at) |l, *x| {
        x.* = if (l) n else dead;
        if (l) n += 1;
    }
    for (bd.rows.items, bd.keys.items, live) |row, k, l| if (l) {
        var s: State = .{ .next = undefined, .accept = bd.accept(k) };
        for (row, &s.next) |t, *o| o.* = if (t == dead) dead else at[t];
        try states.append(a, s);
    };
    return at[0];
}

/// One automaton for the numeric branches of a union (null: a plain integer): it takes what any of them takes.
pub fn unite(a: Allocator, table: *Table, branches: []const ?u32) Error!u32 {
    const plain: Side = .{ .lo = .{ .int = "1" } };
    const starts = try a.alloc(u32, branches.len);
    for (branches, starts) |b, *s| s.* = b orelse (try shared(a, table, .{ .frac = false, .neg = plain, .pos = plain, .zero = true })).?;
    const states = &table.states;
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    var tuples: std.ArrayList([]const u32) = .empty;
    const base: u32 = @intCast(states.items.len);
    try tuples.append(a, starts);
    try ids.put(a, std.mem.sliceAsBytes(starts), base);
    try states.append(a, undefined);
    var i: usize = 0;
    while (i < tuples.items.len) : (i += 1) {
        const from = tuples.items[i];
        var s: State = .{ .next = @splat(dead), .accept = false };
        for (from) |x| if (x != dead and states.items[x].accept) {
            s.accept = true;
        };
        for (&s.next, 0..) |*o, c| {
            const to = try a.alloc(u32, from.len);
            var any = false;
            for (from, to) |x, *y| {
                y.* = if (x == dead) dead else states.items[x].next[c];
                any = any or y.* != dead;
            }
            if (!any) continue;
            const got = try ids.getOrPut(a, std.mem.sliceAsBytes(to));
            if (!got.found_existing) {
                if (states.items.len >= max_states) return error.Refused;
                got.value_ptr.* = @intCast(states.items.len);
                try tuples.append(a, to);
                try states.append(a, undefined);
            }
            o.* = got.value_ptr.*;
        }
        states.items[base + i] = s;
    }
    return base;
}
