//! A byte matcher for xgrammar's json_object grammar: one object, no blank before or after it, at most 32 blanks a run.
const std = @import("std");

/// Containers nest at most this deep; xgrammar has no limit, and a deeper reply is refused here.
pub const max_depth = 128;
/// The most blanks (space, tab, CR, LF) in a row: xgrammar's max_whitespace_cnt as Python 0.6 set it.
pub const max_blanks = 32;

const words = [_][]const u8{ "true", "false", "null" };

const Lex = enum(u8) {
    start, // before the object: only its '{'
    first_key, // after '{': a key or '}'
    key, // after an object's ',': a key
    colon, // after a key: ':'
    value, // after ':' or an array's ',': a value
    first_item, // after '[': a value or ']'
    after, // after a value inside a container: ',' or the container's closer
    string, // inside a string
    escape, // after a string's backslash
    hex, // inside a \u escape's four hex digits
    minus, // a number's '-'
    zero, // a number's leading 0
    int, // a number's integer digits after a nonzero first one
    dot, // a number's '.'
    frac, // a number's fraction digits
    exp, // a number's e or E
    exp_sign, // the exponent's sign
    exp_digits, // the exponent's digits
    literal, // inside true, false or null
    done, // the object has closed: only the end of the reply may follow
};

/// The state after some bytes of a reply: a small value that callers copy to try bytes without taking them.
pub const Matcher = struct {
    lex: Lex = .start,
    /// The open string is an object's key.
    key: bool = false,
    /// Open containers; bit d of `arrays` is set when the container at depth d + 1 is an array.
    depth: u8 = 0,
    arrays: u128 = 0,
    /// Blanks in the current run.
    blanks: u6 = 0,
    /// UTF-8 continuation bytes still due in a string, and the range the next one must fall in.
    need: u2 = 0,
    lo: u8 = 0x80,
    hi: u8 = 0xBF,
    /// A \u escape's digits taken, or bytes of a literal taken.
    count: u3 = 0,
    word: u2 = 0,

    /// Takes `b` and returns true, or returns false and leaves the matcher as it was.
    pub fn step(m: *Matcher, b: u8) bool {
        var trial = m.*;
        if (!trial.take(b)) return false;
        m.* = trial;
        return true;
    }

    /// Takes all of `bytes` and returns true, or takes none of them and returns false.
    pub fn advance(m: *Matcher, bytes: []const u8) bool {
        var trial = m.*;
        for (bytes) |b| if (!trial.take(b)) return false;
        m.* = trial;
        return true;
    }

    /// Whether `bytes` may come next, without taking them.
    pub fn accepts(m: Matcher, bytes: []const u8) bool {
        var trial = m;
        for (bytes) |b| if (!trial.take(b)) return false;
        return true;
    }

    /// The object has closed: the reply may end here, and no byte may follow.
    pub fn complete(m: Matcher) bool {
        return m.lex == .done;
    }

    fn take(m: *Matcher, b: u8) bool {
        if (blankable(m.lex) and blank(b)) {
            if (m.blanks == max_blanks) return false;
            m.blanks += 1;
            return true;
        }
        m.blanks = 0;
        return switch (m.lex) {
            .start => b == '{' and m.open(false),
            .first_key => if (b == '}') m.close() else m.quote(b),
            .key => m.quote(b),
            .colon => m.to(b == ':', .value),
            .value => m.begin(b),
            .first_item => if (b == ']') m.close() else m.begin(b),
            .after => m.next(b),
            .string => m.char(b),
            .escape => if (b == 'u') m.hexStart() else m.to(escaped(b), .string),
            .hex => m.hexDigit(b),
            .minus, .zero, .int, .dot, .frac, .exp, .exp_sign, .exp_digits => m.number(b),
            .literal => m.letter(b),
            .done => false,
        };
    }

    fn to(m: *Matcher, ok: bool, lex: Lex) bool {
        if (ok) m.lex = lex;
        return ok;
    }

    /// The innermost open container is an array.
    fn inArray(m: Matcher) bool {
        return (m.arrays >> @intCast(m.depth - 1)) & 1 == 1;
    }

    fn open(m: *Matcher, array: bool) bool {
        if (m.depth == max_depth) return false;
        const bit = @as(u128, 1) << @intCast(m.depth);
        m.arrays = if (array) m.arrays | bit else m.arrays & ~bit;
        m.depth += 1;
        m.lex = if (array) .first_item else .first_key;
        return true;
    }

    fn close(m: *Matcher) bool {
        m.depth -= 1;
        m.lex = if (m.depth == 0) .done else .after;
        return true;
    }

    fn quote(m: *Matcher, b: u8) bool {
        if (b != '"') return false;
        m.key = true;
        m.lex = .string;
        return true;
    }

    /// A value's first byte.
    fn begin(m: *Matcher, b: u8) bool {
        switch (b) {
            '{' => return m.open(false),
            '[' => return m.open(true),
            '"' => {
                m.key = false;
                m.lex = .string;
            },
            '-' => m.lex = .minus,
            '0' => m.lex = .zero,
            '1'...'9' => m.lex = .int,
            't' => m.literal(0),
            'f' => m.literal(1),
            'n' => m.literal(2),
            else => return false,
        }
        return true;
    }

    /// After a value: a comma, or the innermost container's closer.
    fn next(m: *Matcher, b: u8) bool {
        const array = m.inArray();
        if (b == ',') return m.to(true, if (array) .value else .key);
        if (b == @as(u8, if (array) ']' else '}')) return m.close();
        return false;
    }

    /// A number's next byte, as JSON writes numbers: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?
    fn number(m: *Matcher, b: u8) bool {
        const d = digit(b);
        const e = b == 'e' or b == 'E';
        return switch (m.lex) {
            .minus => if (b == '0') m.to(true, .zero) else m.to(nonzero(b), .int),
            .zero => if (b == '.') m.to(true, .dot) else if (e) m.to(true, .exp) else m.end(b),
            .int => if (d) true else if (b == '.') m.to(true, .dot) else if (e) m.to(true, .exp) else m.end(b),
            .dot => m.to(d, .frac),
            .frac => if (d) true else if (e) m.to(true, .exp) else m.end(b),
            .exp => if (b == '+' or b == '-') m.to(true, .exp_sign) else m.to(d, .exp_digits),
            .exp_sign => m.to(d, .exp_digits),
            .exp_digits => d or m.end(b),
            else => unreachable,
        };
    }

    /// A number ends at the first byte that cannot extend it; that byte must then follow the value.
    fn end(m: *Matcher, b: u8) bool {
        m.lex = .after;
        return m.take(b);
    }

    fn literal(m: *Matcher, word: u2) void {
        m.word = word;
        m.count = 1;
        m.lex = .literal;
    }

    fn letter(m: *Matcher, b: u8) bool {
        const w = words[m.word];
        if (b != w[m.count]) return false;
        m.count += 1;
        if (m.count == w.len) {
            m.count = 0;
            m.lex = .after;
        }
        return true;
    }

    fn hexStart(m: *Matcher) bool {
        m.count = 0;
        m.lex = .hex;
        return true;
    }

    fn hexDigit(m: *Matcher, b: u8) bool {
        if (!std.ascii.isHex(b)) return false;
        m.count += 1;
        if (m.count == 4) {
            m.count = 0;
            m.lex = .string;
        }
        return true;
    }

    /// A string byte: no raw control byte, strict UTF-8 (no overlong form, surrogate, or code point past U+10FFFF).
    fn char(m: *Matcher, b: u8) bool {
        if (m.need > 0) {
            if (b < m.lo or b > m.hi) return false;
            m.need -= 1;
            m.lo = 0x80;
            m.hi = 0xBF;
            return true;
        }
        if (b == '"') return m.to(true, if (m.key) .colon else .after);
        if (b == '\\') return m.to(true, .escape);
        switch (b) {
            0x20...0x7F => {},
            0xC2...0xDF => m.lead(1, 0x80, 0xBF),
            0xE0 => m.lead(2, 0xA0, 0xBF),
            0xE1...0xEC, 0xEE, 0xEF => m.lead(2, 0x80, 0xBF),
            0xED => m.lead(2, 0x80, 0x9F),
            0xF0 => m.lead(3, 0x90, 0xBF),
            0xF1...0xF3 => m.lead(3, 0x80, 0xBF),
            0xF4 => m.lead(3, 0x80, 0x8F),
            else => return false,
        }
        return true;
    }

    fn lead(m: *Matcher, need: u2, lo: u8, hi: u8) void {
        m.need = need;
        m.lo = lo;
        m.hi = hi;
    }
};

fn blankable(lex: Lex) bool {
    return switch (lex) {
        .first_key, .key, .colon, .value, .first_item, .after => true,
        else => false,
    };
}

pub fn blank(b: u8) bool {
    return b == ' ' or b == '\n' or b == '\r' or b == '\t';
}

fn digit(b: u8) bool {
    return b >= '0' and b <= '9';
}

fn nonzero(b: u8) bool {
    return b >= '1' and b <= '9';
}

fn escaped(b: u8) bool {
    return switch (b) {
        '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => true,
        else => false,
    };
}
