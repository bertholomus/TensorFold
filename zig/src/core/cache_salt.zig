//! ``cache_salt`` (vLLM's): a salted request's cache key spells every word through the salt, so its prompt states are
//! kept and resumed only for requests with the same salt. Prompt ids are below ``0x8000_0000``, a salted word above
//! it: no salt only matches no salt, and the prompt cache itself never sees the salt (every match keys on these words).
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The salt's identity: a 64-bit hash of its UTF-8 bytes.
pub fn of(salt: []const u8) u64 {
    return std.hash.Wyhash.hash(0x5a17_ca5e, salt);
}

/// Word `w` of a key under salt `h`: 31 bits hashed from both, in the word space image_key.zig's rows share.
pub fn word(h: u64, w: u32) u32 {
    return 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(h, std.mem.asBytes(&w)) & 0x7fff_ffff));
}

/// `key` (the prompt as the cache matches it) under `salt`; null: `key` itself.
pub fn key(a: Allocator, salt: ?[]const u8, k: []const u32) Allocator.Error![]const u32 {
    const s = salt orelse return k;
    const h = of(s);
    const out = try a.alloc(u32, k.len);
    for (k, out) |w, *o| o.* = word(h, w);
    return out;
}

test "a salted key keeps its length and prefixes, and no salted word is a prompt id" {
    const a = std.testing.allocator;
    const k = [_]u32{ 1, 2, 3, 0x8000_0001 };
    const x = try key(a, "s", &k);
    defer a.free(x);
    const y = try key(a, "s", k[0..2]);
    defer a.free(y);
    try std.testing.expectEqual(k.len, x.len);
    try std.testing.expectEqualSlices(u32, x[0..2], y);
    for (x) |w| try std.testing.expect(w >= 0x8000_0000);
    try std.testing.expectEqualSlices(u32, &k, try key(a, null, &k));
    const z = try key(a, "t", &k);
    defer a.free(z);
    try std.testing.expect(!std.mem.eql(u32, x, z));
}
