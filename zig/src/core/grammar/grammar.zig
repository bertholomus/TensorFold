//! Structured output's grammars as byte matchers, and the token table that tests a logits column against one.
pub const json = @import("json.zig");
pub const Matcher = json.Matcher;
pub const vocab = @import("vocab.zig");
pub const Vocab = vocab.Vocab;

test {
    _ = json;
    _ = vocab;
    _ = @import("json_test.zig");
    _ = @import("xgrammar_test.zig");
}
