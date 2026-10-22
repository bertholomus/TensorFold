//! vLLM's ``cache_salt``: a salted request's prompt states are kept and matched in its salt's own namespace, so it
//! resumes only states an earlier request with the same salt kept, and a request without a salt resumes none of them.
//! The prompt cache matches a request by its key (``api.Request.cacheKey``: the prompt, a lane image's placeholders by
//! the image), so the salt changes the key alone: each of its first two words, a text token in every prompt the server
//! renders (the template's first tokens, or a completion's own ids), goes past every vocabulary's ids as
//! 0x8000_0000 | (the word xor a 31-bit word of the salt's SHA-256). Under one salt two prompts' keys differ wherever
//! their tokens do (the xor is one to one); two salts' keys for one prompt are the same at odds of 2^-62. The prompt,
//! its rows and the reply are the same with a salt or without one.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// How many of a key's first words the salt changes.
pub const words = 2;

/// ``prompt_key`` (the prompt as the cache matches it) in ``salt``'s namespace, allocated in ``a``: as long as it,
/// its first two words salted, the rest as they are.
pub fn key(a: Allocator, prompt_key: []const u32, salt: []const u8) Allocator.Error![]const u32 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(salt, &digest, .{});
    const out = try a.dupe(u32, prompt_key);
    for (out[0..@min(words, out.len)], 0..) |*w, i| {
        const s = std.mem.readInt(u32, digest[4 * i ..][0..4], .little);
        w.* = 0x8000_0000 | ((w.* ^ s) & 0x7fff_ffff);
    }
    return out;
}

test "a salted key is as long as the prompt, its first two words past every vocabulary's ids, the rest the prompt's" {
    const a = std.testing.allocator;
    const k = try key(a, &.{ 1, 2, 3, 4 }, "salt");
    defer a.free(k);
    try std.testing.expectEqual(@as(usize, 4), k.len);
    try std.testing.expect(k[0] >= 0x8000_0000 and k[1] >= 0x8000_0000);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, k[2..]);
}

test "one salt gives a prompt one key; another salt, another first token or no salt gives another" {
    const a = std.testing.allocator;
    const prompt = [_]u32{ 0, 104, 105, 255 };
    const k1 = try key(a, &prompt, "one");
    defer a.free(k1);
    const k2 = try key(a, &prompt, "one");
    defer a.free(k2);
    const other = try key(a, &prompt, "two");
    defer a.free(other);
    const first = try key(a, &.{ 1, 104, 105, 255 }, "one");
    defer a.free(first);
    try std.testing.expectEqualSlices(u32, k1, k2);
    try std.testing.expect(std.mem.indexOfDiff(u32, k1, other).? < words);
    try std.testing.expectEqual(@as(?usize, 0), std.mem.indexOfDiff(u32, k1, first));
    try std.testing.expectEqual(@as(?usize, 0), std.mem.indexOfDiff(u32, k1, &prompt));
}

test "a one-token prompt's one word is salted; an empty key stays empty" {
    const a = std.testing.allocator;
    const one = try key(a, &.{7}, "s");
    defer a.free(one);
    try std.testing.expect(one.len == 1 and one[0] >= 0x8000_0000);
    const none = try key(a, &.{}, "s");
    defer a.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}
