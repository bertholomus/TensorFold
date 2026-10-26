//! A byte matcher for a program (schema.zig): xgrammar 0.2.8's blanks, at most 32 a run, none around the root value.
const std = @import("std");
const schema = @import("schema.zig");
const formats = @import("formats.zig");
const json = @import("json.zig");
const Program = schema.Program;
const Id = schema.Id;
const none = schema.none;

/// Containers nest at most this deep, as in the json_object matcher; xgrammar has no limit.
pub const max_depth = json.max_depth;
pub const max_blanks = json.max_blanks;

/// An open container: its node, and the properties it passed (`tail`: other keys began) or the items it began.
pub const Frame = struct { node: Id, at: u16 };
const tail: u16 = std.math.maxInt(u16);
const no: u16 = std.math.maxInt(u16);
const gone: u32 = std.math.maxInt(u32);

const Lex = enum(u8) {
    root, // before the root value
    obj_open, // after '{': a key's quote or '}'
    obj_key, // inside a key
    obj_colon, // after a key: ':'
    obj_value, // after ':': the value
    obj_after, // after a value in an object: ',' or '}'
    obj_next, // after an object's ',': a key's quote
    arr_open, // after '[': an item or ']'
    arr_item, // after an array's ',': an item
    arr_after, // after an item: ',' or ']'
    string, // inside any string
    text, // inside a minLength/maxLength string
    format, // inside a format string
    literal, // inside an enum, const, choice, true, false or null
    number,
    done, // the root value is complete: only the end of the reply may follow
};
const Esc = enum(u2) { none, escape, hex };
const Num = enum(u8) { minus, zero, int, dot, frac, exp, exp_sign, exp_digits };
const Body = enum { more, close, bad };

/// The state after some bytes of a reply. Trials copy only this small part; the frames below `cur` stay put.
const Lexer = struct {
    lex: Lex = .root,
    depth: u8 = 0,
    blanks: u8 = 0,
    cur: Frame = .{ .node = 0, .at = 0 },
    /// obj_value: the value's node; string, text, format: the string's node.
    node: Id = 0,
    // a string's escape and UTF-8 state
    esc: Esc = .none,
    hex: u3 = 0,
    need: u2 = 0,
    lo: u8 = 0x80,
    hi: u8 = 0xBF,
    // a key: its node in the object's key trie (gone: it starts no key), and whether it can be another key
    trie: u32 = gone,
    free: bool = false,
    num: Num = .minus,
    integer: bool = false,
    /// text: characters taken; literal: the trie node.
    count: u32 = 0,
    nfa: u64 = 0,

    fn take(lx: *Lexer, p: *const Program, st: anytype, b: u8) bool {
        switch (lx.lex) {
            .root => return lx.begin(p, st, p.root, b),
            .obj_open, .obj_colon, .obj_value, .obj_after, .obj_next, .arr_open, .arr_item, .arr_after => {
                if (json.blank(b)) {
                    if (lx.blanks == lx.blankLimit(p)) return false;
                    lx.blanks += 1;
                    return true;
                }
                const run = lx.blanks;
                lx.blanks = 0;
                return lx.token(p, st, b, run);
            },
            .obj_key => return lx.key(p, b),
            .string => return switch (lx.body(b)) {
                .more => true,
                .close => lx.finish(p),
                .bad => false,
            },
            .text => return lx.textByte(p, b),
            .format => {
                const n = formats.nfa(p.nodes[lx.node].format);
                if (b == '"' and n.accepting(lx.nfa)) return lx.finish(p);
                lx.nfa = n.step(lx.nfa, b);
                return lx.nfa != 0;
            },
            .literal => {
                if (p.child(lx.count, b)) |t| {
                    lx.count = t;
                    return true;
                }
                return p.tries[lx.count].end and lx.end(p, st, b);
            },
            .number => return lx.numberByte(p, st, b),
            .done => return false,
        }
    }

    /// xgrammar writes an empty array with prefixItems and no minItems as '[' blanks blanks ']': two runs.
    fn blankLimit(lx: *const Lexer, p: *const Program) u8 {
        if (lx.lex != .arr_open) return max_blanks;
        const a = arr(p, lx.cur);
        return if (a.prefix.len > 0 and a.min == 0) 2 * max_blanks else max_blanks;
    }

    /// The first byte after the blanks between JSON tokens; `run` is how many blanks came before it.
    fn token(lx: *Lexer, p: *const Program, st: anytype, b: u8, run: u8) bool {
        switch (lx.lex) {
            .obj_open => return switch (b) {
                '"' => lx.keyStart(p),
                '}' => lx.closes(p) and lx.close(p, st),
                else => false,
            },
            .obj_next => return b == '"' and lx.keyStart(p),
            .obj_colon => {
                if (b != ':') return false;
                lx.lex = .obj_value;
                return true;
            },
            .obj_value => return lx.begin(p, st, lx.node, b),
            .obj_after => return switch (b) {
                ',' => lx.opens(p) and lx.to(.obj_next),
                '}' => lx.closes(p) and lx.close(p, st),
                else => false,
            },
            .arr_open => {
                const a = arr(p, lx.cur);
                if (b == ']') return a.min == 0 and lx.close(p, st);
                return run <= max_blanks and (a.prefix.len > 0 or (a.items != none and a.max != 0)) and lx.item(p, st, b);
            },
            .arr_item => return lx.item(p, st, b),
            .arr_after => {
                const a = arr(p, lx.cur);
                const n = lx.cur.at;
                return switch (b) {
                    ',' => (n < a.prefix.len or (a.items != none and (a.max == none or n < a.max))) and lx.to(.arr_item),
                    ']' => n >= a.min and lx.close(p, st),
                    else => false,
                };
            },
            else => unreachable,
        }
    }

    fn to(lx: *Lexer, lex: Lex) bool {
        lx.lex = lex;
        return true;
    }

    /// A value of `node` starts with byte `b`.
    fn begin(lx: *Lexer, p: *const Program, st: anytype, node: Id, b: u8) bool {
        switch (p.nodes[node]) {
            .any => return switch (b) {
                '{' => lx.push(st, Program.any_object, .obj_open),
                '[' => lx.push(st, Program.any_array, .arr_open),
                '"' => lx.open(.string, node),
                '-', '0'...'9' => lx.numberStart(b, false),
                else => lx.literalStart(p, p.words, b),
            },
            .object => return b == '{' and lx.push(st, node, .obj_open),
            .array => return b == '[' and lx.push(st, node, .arr_open),
            .string => return b == '"' and lx.open(.string, node),
            .text => return b == '"' and lx.open(.text, node),
            .format => return b == '"' and lx.open(.format, node),
            .integer => return lx.numberStart(b, true),
            .number => return lx.numberStart(b, false),
            .literal => |t| return lx.literalStart(p, t, b),
            .any_of => |u| {
                const branch = p.unions[u].dispatch[b];
                return branch != none and lx.begin(p, st, branch, b);
            },
            .alias => unreachable,
        }
    }

    fn open(lx: *Lexer, lex: Lex, node: Id) bool {
        lx.lex = lex;
        lx.node = node;
        lx.esc = .none;
        lx.need = 0;
        lx.count = 0;
        lx.nfa = formats.Nfa.start;
        return true;
    }

    fn push(lx: *Lexer, st: anytype, node: Id, lex: Lex) bool {
        if (lx.depth == max_depth) return false;
        if (lx.depth > 0 and !st.save(lx.depth - 1, lx.cur)) return false;
        lx.cur = .{ .node = node, .at = 0 };
        lx.depth += 1;
        lx.lex = lex;
        lx.blanks = 0;
        return true;
    }

    /// The innermost container closes, and so ends as a value.
    fn close(lx: *Lexer, p: *const Program, st: anytype) bool {
        lx.depth -= 1;
        if (lx.depth > 0) lx.cur = st.load(lx.depth - 1);
        return lx.finish(p);
    }

    /// A value has ended: its container's ',' or closer may follow, or nothing after the root value.
    fn finish(lx: *Lexer, p: *const Program) bool {
        lx.blanks = 0;
        lx.lex = if (lx.depth == 0) .done else if (p.nodes[lx.cur.node] == .object) .obj_after else .arr_after;
        return true;
    }

    /// A number or literal ends at the first byte that cannot extend it; that byte must then follow the value.
    fn end(lx: *Lexer, p: *const Program, st: anytype, b: u8) bool {
        _ = lx.finish(p);
        return lx.take(p, st, b);
    }

    fn item(lx: *Lexer, p: *const Program, st: anytype, b: u8) bool {
        const a = arr(p, lx.cur);
        const n = lx.cur.at;
        const node = if (n < a.prefix.len) a.prefix[n] else a.items;
        lx.cur.at = n +| 1;
        return node != none and lx.begin(p, st, node, b);
    }

    /// After a value in an object, another key may come: a property still ahead, or another key.
    fn opens(lx: *const Lexer, p: *const Program) bool {
        const o = obj(p, lx.cur);
        const c = lx.cur.at;
        if (c == tail) return true;
        return c < o.props.len or (o.additional != none and o.required_from[c] == o.props.len);
    }

    /// The object may close: no required property is still ahead.
    fn closes(lx: *const Lexer, p: *const Program) bool {
        const o = obj(p, lx.cur);
        const c = lx.cur.at;
        return c == tail or o.required_from[c] == o.props.len;
    }

    fn keyStart(lx: *Lexer, p: *const Program) bool {
        const o = obj(p, lx.cur);
        const c = lx.cur.at;
        const free = o.additional != none and (c == tail or o.required_from[c] == o.props.len);
        if ((c == tail or c >= o.props.len) and !free) return false;
        lx.trie = o.trie;
        lx.free = free;
        lx.esc = .none;
        lx.need = 0;
        lx.lex = .obj_key;
        return true;
    }

    /// A key's next byte: a property up to the next required one, or (if allowed) another key that is no name.
    fn key(lx: *Lexer, p: *const Program, b: u8) bool {
        const o = obj(p, lx.cur);
        const t: u32 = if (lx.trie == gone) gone else p.child(lx.trie, b) orelse gone;
        const cand = if (t == gone) no else ahead(o, t, lx.cur.at);
        // the key so far, closing quote included, is a property's key
        const whole = t != gone and p.tries[t].end;
        var free = lx.free;
        var other = false;
        if (free) {
            switch (lx.body(b)) {
                .bad => free = false,
                // xgrammar keeps every name, passed or ahead, from other keys; names are bare (schema_keys.bare)
                .close => if (whole) {
                    free = false;
                } else {
                    other = true;
                },
                .more => {},
            }
        }
        if (cand != no and whole) {
            lx.cur.at = cand + 1;
            lx.node = o.props[cand].node;
            lx.lex = .obj_colon;
            return true;
        }
        if (other) {
            lx.cur.at = tail;
            lx.node = o.additional;
            lx.lex = .obj_colon;
            return true;
        }
        if (cand == no and !free) return false;
        lx.trie = t;
        lx.free = free;
        return true;
    }

    /// A JSON string's next byte: no raw control byte, strict UTF-8 (as the json_object matcher).
    fn body(lx: *Lexer, b: u8) Body {
        if (lx.need > 0) return if (lx.continuation(b)) .more else .bad;
        switch (lx.esc) {
            .escape => {
                if (b == 'u') {
                    lx.esc = .hex;
                    lx.hex = 0;
                } else if (escaped(b)) lx.esc = .none else return .bad;
                return .more;
            },
            .hex => {
                if (!std.ascii.isHex(b)) return .bad;
                lx.hex += 1;
                if (lx.hex == 4) lx.esc = .none;
                return .more;
            },
            .none => {},
        }
        return switch (b) {
            '"' => .close,
            '\\' => blk: {
                lx.esc = .escape;
                break :blk .more;
            },
            0x20...0x21, 0x23...0x5B, 0x5D...0x7F => .more,
            else => if (lx.lead(b)) .more else .bad,
        };
    }

    /// A minLength/maxLength string's byte: xgrammar's [^"\\\r\n] by character, no escape, other control bytes taken.
    fn textByte(lx: *Lexer, p: *const Program, b: u8) bool {
        if (lx.need > 0) return lx.continuation(b);
        const t = p.nodes[lx.node].text;
        if (b == '"') return lx.count >= t.min and lx.finish(p);
        if (b == '\\' or b == '\r' or b == '\n' or lx.count == t.max) return false;
        if (b >= 0x80 and !lx.lead(b)) return false;
        lx.count += 1;
        return true;
    }

    fn continuation(lx: *Lexer, b: u8) bool {
        if (b < lx.lo or b > lx.hi) return false;
        lx.need -= 1;
        lx.lo = 0x80;
        lx.hi = 0xBF;
        return true;
    }

    /// A UTF-8 lead byte: no overlong form, surrogate, or code point past U+10FFFF.
    fn lead(lx: *Lexer, b: u8) bool {
        const n: struct { u2, u8, u8 } = switch (b) {
            0xC2...0xDF => .{ 1, 0x80, 0xBF },
            0xE0 => .{ 2, 0xA0, 0xBF },
            0xE1...0xEC, 0xEE, 0xEF => .{ 2, 0x80, 0xBF },
            0xED => .{ 2, 0x80, 0x9F },
            0xF0 => .{ 3, 0x90, 0xBF },
            0xF1...0xF3 => .{ 3, 0x80, 0xBF },
            0xF4 => .{ 3, 0x80, 0x8F },
            else => return false,
        };
        lx.need = n[0];
        lx.lo = n[1];
        lx.hi = n[2];
        return true;
    }

    fn literalStart(lx: *Lexer, p: *const Program, t: u32, b: u8) bool {
        lx.count = p.child(t, b) orelse return false;
        lx.lex = .literal;
        return true;
    }

    fn numberStart(lx: *Lexer, b: u8, integer: bool) bool {
        lx.num = switch (b) {
            '-' => .minus,
            '0' => .zero,
            '1'...'9' => .int,
            else => return false,
        };
        lx.integer = integer;
        lx.lex = .number;
        return true;
    }

    /// number: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?; integer: 0|-?[1-9][0-9]* (xgrammar refuses -0).
    fn numberByte(lx: *Lexer, p: *const Program, st: anytype, b: u8) bool {
        const d = std.ascii.isDigit(b);
        const dot = !lx.integer and b == '.';
        const e = !lx.integer and (b == 'e' or b == 'E');
        switch (lx.num) {
            .minus => lx.num = if (b == '0' and !lx.integer) .zero else if (b >= '1' and b <= '9') .int else return false,
            .zero => lx.num = if (dot) .dot else if (e) .exp else return lx.end(p, st, b),
            .int => lx.num = if (d) .int else if (dot) .dot else if (e) .exp else return lx.end(p, st, b),
            .dot => lx.num = if (d) .frac else return false,
            .frac => lx.num = if (d) .frac else if (e) .exp else return lx.end(p, st, b),
            .exp => lx.num = if (b == '+' or b == '-') .exp_sign else if (d) .exp_digits else return false,
            .exp_sign => lx.num = if (d) .exp_digits else return false,
            .exp_digits => if (!d) return lx.end(p, st, b),
        }
        return true;
    }

    fn complete(lx: *const Lexer, p: *const Program) bool {
        return switch (lx.lex) {
            .done => true,
            .number => lx.depth == 0 and switch (lx.num) {
                .zero, .int, .frac, .exp_digits => true,
                else => false,
            },
            .literal => lx.depth == 0 and p.tries[lx.count].end,
            else => false,
        };
    }
};

fn obj(p: *const Program, f: Frame) *const schema.Object {
    return &p.objects[p.nodes[f.node].object];
}

fn arr(p: *const Program, f: Frame) schema.Array {
    return p.arrays[p.nodes[f.node].array];
}

/// The first property from `c` up to the next required one whose key passes through key-trie node `t`, or `no`.
fn ahead(o: *const schema.Object, t: u32, c: u16) u16 {
    // after the last property every list ends before c, so only `tail` needs a test
    if (c == tail) return no;
    const span = o.below[t - o.trie];
    const list = o.members[span.first..][0..span.len];
    // the members are in property order: find the first at or after c
    var lo: usize = 0;
    var hi: usize = list.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (list[mid] < c) lo = mid + 1 else hi = mid;
    }
    // past a required property the key cannot be a later one
    return if (lo < list.len and list[lo] <= o.required_from[c]) list[lo] else no;
}

fn escaped(b: u8) bool {
    return switch (b) {
        '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => true,
        else => false,
    };
}

/// The frames below the innermost container, in the matcher.
const Own = struct {
    frames: *[max_depth]Frame,

    fn save(s: *Own, at: usize, f: Frame) bool {
        s.frames[at] = f;
        return true;
    }

    fn load(s: *Own, at: usize) Frame {
        return s.frames[at];
    }
};

/// The frames below the innermost container for a trial: the matcher's, read only, under a few new ones.
const Overlay = struct {
    base: *const [max_depth]Frame,
    /// The matcher's frames still below.
    keep: usize,
    local: [8]Frame = undefined,
    n: usize = 0,
    /// More containers opened than `local` holds: the trial must run on a whole copy.
    full: bool = false,

    fn save(s: *Overlay, at: usize, f: Frame) bool {
        std.debug.assert(at == s.keep + s.n);
        if (s.n == s.local.len) {
            s.full = true;
            return false;
        }
        s.local[s.n] = f;
        s.n += 1;
        return true;
    }

    fn load(s: *Overlay, at: usize) Frame {
        std.debug.assert(at + 1 == s.keep + s.n);
        if (s.n > 0) {
            s.n -= 1;
            return s.local[s.n];
        }
        s.keep -= 1;
        return s.base[s.keep];
    }
};

/// The state after some bytes of a reply under a program. `accepts` tries bytes without copying the frames.
pub const Matcher = struct {
    program: *const Program,
    lexer: Lexer = .{},
    frames: [max_depth]Frame = undefined,

    pub fn init(p: *const Program) Matcher {
        return .{ .program = p };
    }

    /// Takes `b` and returns true, or returns false and leaves the matcher as it was.
    pub fn step(m: *Matcher, b: u8) bool {
        var lx = m.lexer;
        // one byte opens or closes at most one container, so a refused byte leaves the frames below intact
        var own: Own = .{ .frames = &m.frames };
        if (!lx.take(m.program, &own, b)) return false;
        m.lexer = lx;
        return true;
    }

    /// Takes all of `bytes` and returns true, or takes none of them and returns false.
    pub fn advance(m: *Matcher, bytes: []const u8) bool {
        if (!m.accepts(bytes)) return false;
        for (bytes) |b| {
            const took = m.step(b);
            std.debug.assert(took);
        }
        return true;
    }

    /// Whether `bytes` may come next, without taking them.
    pub fn accepts(m: *const Matcher, bytes: []const u8) bool {
        var lx = m.lexer;
        var trial: Overlay = .{ .base = &m.frames, .keep = if (lx.depth > 0) lx.depth - 1 else 0 };
        for (bytes) |b| if (!lx.take(m.program, &trial, b)) {
            if (!trial.full) return false;
            var copy = m.*;
            for (bytes) |c| if (!copy.step(c)) return false;
            return true;
        };
        return true;
    }

    /// The root value is complete: the reply may end here.
    pub fn complete(m: *const Matcher) bool {
        return m.lexer.complete(m.program);
    }
};
