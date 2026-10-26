//! The string formats a schema may name and this engine enforces, from xgrammar 0.2.8's own patterns.
const std = @import("std");

pub const Format = enum(u8) { date_time, date, time, uuid, ipv4, email };

/// Formats xgrammar 0.2.8 enforces and this engine refuses; any other unknown format is a plain string, as in xgrammar.
pub const unsupported = [_][]const u8{ "duration", "ipv6", "hostname", "uri", "uri-reference", "uri-template", "json-pointer", "relative-json-pointer" };

const names = [_][]const u8{ "date-time", "date", "time", "uuid", "ipv4", "email" };

/// The format a schema's `format` names, when this engine enforces it.
pub fn named(text: []const u8) ?Format {
    for (names, 0..) |n, i| if (std.mem.eql(u8, n, text)) return @fromBackingInt(@intCast(i));
    return null;
}

pub fn name(f: Format) []const u8 {
    return names[@backingInt(f)];
}

// xgrammar 0.2.8 JSONFormatToRegexPattern without anchors, matched by a string's text; xgrammar's \w is [A-Za-z0-9_].
const date = "\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\\d|3[01])";
const time = "([01]\\d|2[0-3]):[0-5]\\d:([0-5]\\d|60)(\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)";
const decbyte = "(25[0-5]|2[0-4]\\d|[0-1]?\\d?\\d)";
const atext = "[\\w!#$%&'*+/=?^`{|}~-]";
const patterns = [_][]const u8{
    "(" ++ date ++ ")T" ++ time,
    date,
    time,
    "[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
    "(" ++ decbyte ++ "\\.){3}" ++ decbyte,
    "((" ++ atext ++ "+(\\." ++ atext ++ "+)*)|\\\\\"(\\\\[\\x20-\\x7E]|[\\x20\\x21\\x23-\\x5B\\x5D-\\x7E])*\\\\\")@" ++
        "([A-Za-z0-9]([\\-A-Za-z0-9]*[A-Za-z0-9])?)((\\.[A-Za-z0-9][\\-A-Za-z0-9]*[A-Za-z0-9])*)",
};

const tables = blk: {
    var all: [patterns.len]Nfa = undefined;
    for (patterns, &all) |p, *n| n.* = compile(p);
    break :blk all;
};

/// The automaton of format `f`.
pub fn nfa(f: Format) *const Nfa {
    return &tables[@backingInt(f)];
}

/// A pattern's automaton (Glushkov's): a state is the set of positions the text so far can end on.
pub const Nfa = struct {
    /// follow[p]: the positions that may come after position p; follow[63] holds the first positions.
    follow: [64]u64,
    /// by_byte[b]: the positions whose character class holds byte b.
    by_byte: [256]u64,
    /// The positions a whole match may end on, and `start` when the empty text matches.
    last: u64,

    /// The state before any byte.
    pub const start: u64 = 1 << 63;

    /// The state after byte `b`; 0 when no match can go on.
    pub fn step(n: *const Nfa, state: u64, b: u8) u64 {
        var next: u64 = 0;
        var it = state;
        while (it != 0) : (it &= it - 1) next |= n.follow[@ctz(it)];
        return next & n.by_byte[b];
    }

    /// The text so far matches the whole pattern.
    pub fn accepting(n: *const Nfa, state: u64) bool {
        return state & n.last != 0;
    }
};

const Frag = struct { first: u64 = 0, last: u64 = 0, nullable: bool = true };

/// A pattern of characters, classes, groups, `|`, `?`, `*`, `+` and `{n}` as an automaton, at compile time.
fn compile(comptime pattern: []const u8) Nfa {
    @setEvalBranchQuota(1_000_000);
    var c: Compiler = .{ .text = pattern };
    const whole = c.alt();
    if (c.i != pattern.len) @compileError("unexpected ')' in a format pattern");
    var n: Nfa = .{ .follow = c.follow, .by_byte = c.by_byte, .last = whole.last };
    n.follow[63] = whole.first;
    if (whole.nullable) n.last |= Nfa.start;
    return n;
}

const Compiler = struct {
    text: []const u8,
    i: usize = 0,
    count: u7 = 0,
    follow: [64]u64 = @splat(0),
    by_byte: [256]u64 = @splat(0),

    fn peek(c: *Compiler) u8 {
        return if (c.i < c.text.len) c.text[c.i] else 0;
    }

    fn next(c: *Compiler) u8 {
        const b = c.peek();
        c.i += 1;
        return b;
    }

    fn alt(c: *Compiler) Frag {
        var f = c.seq();
        while (c.peek() == '|') {
            c.i += 1;
            const g = c.seq();
            f = .{ .first = f.first | g.first, .last = f.last | g.last, .nullable = f.nullable or g.nullable };
        }
        return f;
    }

    fn seq(c: *Compiler) Frag {
        var f: Frag = .{};
        while (c.i < c.text.len and c.peek() != '|' and c.peek() != ')') f = c.concat(f, c.repeat());
        return f;
    }

    fn concat(c: *Compiler, f: Frag, g: Frag) Frag {
        c.link(f.last, g.first);
        return .{
            .first = f.first | (if (f.nullable) g.first else 0),
            .last = g.last | (if (g.nullable) f.last else 0),
            .nullable = f.nullable and g.nullable,
        };
    }

    fn link(c: *Compiler, from: u64, to: u64) void {
        var it = from;
        while (it != 0) : (it &= it - 1) c.follow[@ctz(it)] |= to;
    }

    fn repeat(c: *Compiler) Frag {
        const at = c.i;
        var f = c.atom();
        switch (c.peek()) {
            '?' => {
                c.i += 1;
                f.nullable = true;
            },
            '*', '+' => {
                f.nullable = f.nullable or c.next() == '*';
                c.link(f.last, f.first);
            },
            '{' => {
                c.i += 1;
                var times: usize = 0;
                while (std.ascii.isDigit(c.peek())) times = times * 10 + (c.next() - '0');
                if (c.next() != '}' or times == 0) @compileError("a format pattern repeats only as {n}");
                const after = c.i;
                for (1..times) |_| {
                    c.i = at; // each copy gets its own positions
                    f = c.concat(f, c.atom());
                }
                c.i = after;
            },
            else => {},
        }
        return f;
    }

    fn atom(c: *Compiler) Frag {
        var set: [256]bool = @splat(false);
        switch (c.next()) {
            '(' => {
                const f = c.alt();
                if (c.next() != ')') @compileError("a format pattern has an unclosed group");
                return f;
            },
            '[' => {
                if (c.peek() == '^') @compileError("format patterns have no negated class");
                while (c.peek() != ']') {
                    const lo = c.member(&set);
                    if (lo != null and c.peek() == '-' and c.text[c.i + 1] != ']') {
                        c.i += 1;
                        const hi = c.member(&set) orelse @compileError("a class range ends in a class");
                        for (lo.?..@as(usize, hi) + 1) |b| set[b] = true;
                    }
                }
                c.i += 1;
            },
            '\\' => _ = c.escape(&set),
            else => |b| set[b] = true,
        }
        if (c.count == 63) @compileError("a format pattern has more than 63 positions");
        const bit = @as(u64, 1) << @intCast(c.count);
        c.count += 1;
        for (set, 0..) |in, b| if (in) {
            c.by_byte[b] |= bit;
        };
        return .{ .first = bit, .last = bit, .nullable = false };
    }

    /// A class member: its byte (null for \d and \w, which add a set).
    fn member(c: *Compiler, set: *[256]bool) ?u8 {
        const b = c.next();
        if (b != '\\') {
            set[b] = true;
            return b;
        }
        return c.escape(set);
    }

    fn escape(c: *Compiler, set: *[256]bool) ?u8 {
        switch (c.next()) {
            'd' => for ('0'..'9' + 1) |b| {
                set[b] = true;
            },
            'w' => for (0..256) |b| {
                set[b] = std.ascii.isAlphanumeric(@intCast(b)) or b == '_' or set[b];
            },
            'x' => {
                const b = std.fmt.parseInt(u8, c.text[c.i .. c.i + 2], 16) catch @compileError("bad \\x escape");
                c.i += 2;
                set[b] = true;
                return b;
            },
            else => |b| {
                set[b] = true;
                return b;
            },
        }
        return null;
    }
};

fn matches(f: Format, text: []const u8) bool {
    const n = nfa(f);
    var s = Nfa.start;
    for (text) |b| {
        s = n.step(s, b);
        if (s == 0) return false;
    }
    return n.accepting(s);
}

test "each format takes its own shapes and refuses near misses" {
    const cases = [_]struct { Format, []const u8, bool }{
        .{ .date, "2026-10-26", true },
        .{ .date, "2026-13-01", false },
        .{ .date, "2026-02-31", true }, // the pattern does not know month lengths
        .{ .date, "2026-1-01", false },
        .{ .date, "02026-10-26", false },
        .{ .time, "23:59:60Z", true },
        .{ .time, "23:59:60.123+05:30", true },
        .{ .time, "24:00:00Z", false },
        .{ .time, "12:00:00", false },
        .{ .time, "12:00:00.Z", false },
        .{ .date_time, "2026-10-26T08:47:00Z", true },
        .{ .date_time, "2026-10-26T08:47:00.5-07:00", true },
        .{ .date_time, "2026-10-26 08:47:00Z", false },
        .{ .date_time, "2026-10-26t08:47:00Z", false },
        .{ .uuid, "123e4567-E89B-12d3-a456-426614174000", true },
        .{ .uuid, "123e4567-e89b-12d3-a456-42661417400", false },
        .{ .uuid, "123e4567e89b12d3a456426614174000", false },
        .{ .ipv4, "255.255.255.255", true },
        .{ .ipv4, "0.01.001.9", true },
        .{ .ipv4, "256.1.1.1", false },
        .{ .ipv4, "1.1.1", false },
        .{ .ipv4, "1.1.1.1.", false },
        .{ .email, "a.b+c@example.com", true },
        .{ .email, "a@b", true },
        .{ .email, "a@b.c", false }, // each label after a dot needs two characters
        .{ .email, "a@b.co", true },
        .{ .email, "\\\"x y\\\\z\\\"@host", true },
        .{ .email, "a..b@host", false },
        .{ .email, "@host", false },
        .{ .email, "a@-host", false },
        .{ .email, "a@host-", false },
    };
    for (cases) |c| {
        if (matches(c[0], c[1]) != c[2]) {
            std.debug.print("{s} {s}: want {}\n", .{ name(c[0]), c[1], c[2] });
            return error.TestUnexpectedResult;
        }
    }
}

test "no reachable format state may both end and go on with a quote, so a closing quote is never in doubt" {
    const a = std.testing.allocator;
    for (0..patterns.len) |i| {
        const n = nfa(@fromBackingInt(@intCast(i)));
        var seen: std.AutoArrayHashMapUnmanaged(u64, void) = .empty;
        defer seen.deinit(a);
        try seen.put(a, Nfa.start, {});
        var k: usize = 0;
        while (k < seen.count()) : (k += 1) {
            const s = seen.keys()[k];
            try std.testing.expect(!(n.accepting(s) and n.step(s, '"') != 0));
            for (0..256) |b| {
                const t = n.step(s, @intCast(b));
                if (t != 0) try seen.put(a, t, {});
            }
        }
        try std.testing.expect(seen.count() < 10_000);
    }
}

test "format names" {
    try std.testing.expectEqual(Format.date_time, named("date-time").?);
    try std.testing.expectEqual(@as(?Format, null), named("ipv6"));
    try std.testing.expectEqualStrings("email", name(.email));
}
