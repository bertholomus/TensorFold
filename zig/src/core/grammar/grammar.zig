//! Structured output's grammars as byte matchers, and the token table that tests a logits column against one.
pub const json = @import("json.zig");
pub const vocab = @import("vocab.zig");
pub const Vocab = vocab.Vocab;
pub const schema = @import("schema.zig");
pub const Program = schema.Program;
pub const formats = @import("formats.zig");
pub const schema_match = @import("schema_match.zig");

/// A reply's grammar at its bytes so far: the json_object matcher, or a compiled program's (json_schema, choice).
pub const Matcher = union(enum) {
    object: json.Matcher,
    program: schema_match.Matcher,

    /// json_object without a program.
    pub fn init(program: ?*const Program) Matcher {
        return if (program) |p| .{ .program = .init(p) } else .{ .object = .{} };
    }

    /// Whether `bytes` may come next, without taking them.
    pub fn accepts(m: *const Matcher, bytes: []const u8) bool {
        return switch (m.*) {
            .object => |*o| o.accepts(bytes),
            .program => |*s| s.accepts(bytes),
        };
    }

    /// Takes all of `bytes` and returns true, or takes none of them and returns false.
    pub fn advance(m: *Matcher, bytes: []const u8) bool {
        return switch (m.*) {
            .object => |*o| o.advance(bytes),
            .program => |*s| s.advance(bytes),
        };
    }

    /// The reply may end here.
    pub fn complete(m: *const Matcher) bool {
        return switch (m.*) {
            .object => |*o| o.complete(),
            .program => |*s| s.complete(),
        };
    }
};

test {
    _ = json;
    _ = vocab;
    _ = formats;
    _ = @import("schema_keys.zig");
    _ = schema;
    _ = schema_match;
    _ = @import("schema_test.zig");
    _ = @import("schema_match_test.zig");
    _ = @import("json_test.zig");
    _ = @import("xgrammar_test.zig");
    _ = @import("xgrammar_schema_test.zig");
}
