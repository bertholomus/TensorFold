//! The function symbols of a cubin (ELF64), so a kernel is found by its template arguments: the served build's names
//! carry a hash of the anonymous namespace that changes with every build.
const std = @import("std");

pub const Symbols = struct {
    bytes: []const u8,
    sym: usize = 0,
    end: usize = 0,
    entsize: usize = 24,
    strtab: usize = 0,

    /// The first SYMTAB section's function symbols; null when `bytes` is not a 64-bit little-endian ELF.
    pub fn init(bytes: []const u8) ?Symbols {
        if (bytes.len < 64 or !std.mem.eql(u8, bytes[0..4], "\x7fELF") or bytes[4] != 2 or bytes[5] != 1) return null;
        const shoff = std.mem.readInt(u64, bytes[0x28..][0..8], .little);
        const shentsize = std.mem.readInt(u16, bytes[0x3a..][0..2], .little);
        const shnum = std.mem.readInt(u16, bytes[0x3c..][0..2], .little);
        for (0..shnum) |i| {
            const h = shoff + i * shentsize;
            if (h + 64 > bytes.len) return null;
            const typ = std.mem.readInt(u32, bytes[h + 4 ..][0..4], .little);
            if (typ != 2) continue; // SHT_SYMTAB
            const off = std.mem.readInt(u64, bytes[h + 24 ..][0..8], .little);
            const size = std.mem.readInt(u64, bytes[h + 32 ..][0..8], .little);
            const link = std.mem.readInt(u32, bytes[h + 40 ..][0..4], .little);
            const ent = std.mem.readInt(u64, bytes[h + 56 ..][0..8], .little);
            const sh = shoff + link * shentsize;
            if (sh + 64 > bytes.len or ent == 0) return null;
            return .{ .bytes = bytes, .sym = off, .end = off + size, .entsize = ent, .strtab = std.mem.readInt(u64, bytes[sh + 24 ..][0..8], .little) };
        }
        return null;
    }

    /// The next function symbol's name.
    pub fn next(s: *Symbols) ?[]const u8 {
        while (s.sym + s.entsize <= s.end and s.sym + 24 <= s.bytes.len) {
            const e = s.bytes[s.sym..];
            s.sym += s.entsize;
            if (e[4] & 0xf != 2) continue; // STT_FUNC
            const at = s.strtab + std.mem.readInt(u32, e[0..4], .little);
            if (at >= s.bytes.len) return null;
            const z = std.mem.indexOfScalarPos(u8, s.bytes, at, 0) orelse return null;
            return s.bytes[at..z];
        }
        return null;
    }
};

test "a tiny ELF's function symbols" {
    // ELF header, a string table and a symbol table: names "", "kern_a", "data_b"; kern_a is a FUNC
    var b: [512]u8 = @splat(0);
    @memcpy(b[0..4], "\x7fELF");
    b[4] = 2;
    b[5] = 1;
    const strtab = 256;
    @memcpy(b[strtab..][0..15], "\x00kern_a\x00data_b\x00");
    const symtab = 300;
    std.mem.writeInt(u32, b[symtab + 24 ..][0..4], 1, .little);
    b[symtab + 24 + 4] = 0x12; // global FUNC
    std.mem.writeInt(u32, b[symtab + 48 ..][0..4], 8, .little);
    b[symtab + 48 + 4] = 0x11; // global OBJECT
    const shoff = 64;
    std.mem.writeInt(u64, b[0x28..][0..8], shoff, .little);
    std.mem.writeInt(u16, b[0x3a..][0..2], 64, .little);
    std.mem.writeInt(u16, b[0x3c..][0..2], 3, .little);
    // section 1: STRTAB, section 2: SYMTAB linked to 1
    std.mem.writeInt(u32, b[shoff + 64 + 4 ..][0..4], 3, .little);
    std.mem.writeInt(u64, b[shoff + 64 + 24 ..][0..8], strtab, .little);
    std.mem.writeInt(u32, b[shoff + 128 + 4 ..][0..4], 2, .little);
    std.mem.writeInt(u64, b[shoff + 128 + 24 ..][0..8], symtab, .little);
    std.mem.writeInt(u64, b[shoff + 128 + 32 ..][0..8], 72, .little);
    std.mem.writeInt(u32, b[shoff + 128 + 40 ..][0..4], 1, .little);
    std.mem.writeInt(u64, b[shoff + 128 + 56 ..][0..8], 24, .little);
    var s = Symbols.init(&b).?;
    try std.testing.expectEqualStrings("kern_a", s.next().?);
    try std.testing.expectEqual(@as(?[]const u8, null), s.next());
}
