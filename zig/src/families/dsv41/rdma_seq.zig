//! The RDMA ring's sequence bookkeeping (rdma_ring.zig's proxy thread), apart from the verbs so it can be tested: a
//! slot's parts over the devices, a peer's parts back to whole sequences, and when a slot's writes have all been sent.
const std = @import("std");

pub const max_ranks = 8;
pub const max_devices = 4;
pub const max_slots = 16;

/// Device `d`'s part of an `n`-byte slot over `nd` devices: whole 16-byte units each, the last device takes the rest.
pub fn part(n: u64, nd: u32, d: u32) struct { off: u64, len: u64 } {
    const unit = n / 16 / nd * 16;
    const off = unit * d;
    return .{ .off = off, .len = if (d == nd - 1) n - off else unit };
}

/// The immediate a write carries: its sequence's low 32 bits, in network order as the Python build's ring sends it.
pub fn immediate(q: u64) u32 {
    return std.mem.nativeToBig(u32, @truncate(q));
}

/// A receive completion's immediate back to a sequence. A device delivers a peer's parts in order, so it extends `last`,
/// the sequence that device delivered from that peer before.
pub fn extend(last: u64, imm: u32) u64 {
    var q = (last & ~@as(u64, 0xffff_ffff)) | std.mem.bigToNative(u32, imm);
    if (q <= last) q += 1 << 32;
    return q;
}

/// What each peer has delivered: the last sequence each device brought and the last one published in the flags.
pub const Arrivals = struct {
    nd: u32,
    got: [max_ranks][max_devices]u64 = @splat(@splat(0)),
    published: [max_ranks]u64 = @splat(0),

    /// A receive completion from `peer` on device `d`: the sequences `from`..`to` (inclusive; none when from > to) whose
    /// every part is now in, to publish in order (a peer may already be a gather ahead on one device).
    pub fn arrive(a: *Arrivals, peer: usize, d: usize, imm: u32) struct { from: u64, to: u64 } {
        a.got[peer][d] = extend(a.got[peer][d], imm);
        var in = a.got[peer][0];
        for (a.got[peer][1..a.nd]) |g| in = @min(in, g);
        const from = a.published[peer] + 1;
        if (in > a.published[peer]) a.published[peer] = in;
        return .{ .from = from, .to = in };
    }
};

/// Write completions a gather's slot still owes: one a peer and device.
pub const Sends = struct {
    need: u32,
    slots: u32,
    seq: [max_slots]u64 = @splat(0),
    done: [max_slots]u32 = @splat(0),

    /// One completed write of sequence `q`: q when it was the last its slot owed (the slot may be staged again), else null.
    pub fn complete(s: *Sends, q: u64) ?u64 {
        const slot: usize = @intCast(q % s.slots);
        if (s.seq[slot] != q) {
            s.seq[slot] = q;
            s.done[slot] = 0;
        }
        s.done[slot] += 1;
        return if (s.done[slot] == s.need) q else null;
    }
};

test "a slot's parts: whole 16-byte units a device, the rest on the last" {
    const a = part(120_000, 2, 0);
    const b = part(120_000, 2, 1);
    try std.testing.expectEqual(@as(u64, 0), a.off);
    try std.testing.expectEqual(@as(u64, 60_000), a.len);
    try std.testing.expectEqual(@as(u64, 60_000), b.off);
    try std.testing.expectEqual(@as(u64, 60_000), b.len);
    const c = part(48, 2, 1);
    try std.testing.expectEqual(@as(u64, 16), c.off);
    try std.testing.expectEqual(@as(u64, 32), c.len);
    try std.testing.expectEqual(@as(u64, 0), part(0, 2, 0).len);
    try std.testing.expectEqual(@as(u64, 0), part(0, 2, 1).len);
    try std.testing.expectEqual(@as(u64, 4096), part(4096, 1, 0).len);
}

test "immediates extend a device's last sequence, across the 32-bit wrap" {
    try std.testing.expectEqual(@as(u64, 1), extend(0, immediate(1)));
    try std.testing.expectEqual(@as(u64, 7), extend(6, immediate(7)));
    try std.testing.expectEqual(@as(u64, 1 << 32), extend(0xffff_ffff, immediate(1 << 32)));
    try std.testing.expectEqual(@as(u64, (1 << 32) + 3), extend((1 << 32) + 2, immediate((1 << 32) + 3)));
    // the Python build's htonl: the wire bytes are big-endian
    try std.testing.expectEqual(@as(u32, 0x0102_0304), std.mem.bigToNative(u32, immediate(0x0102_0304)));
}

test "a sequence is published once every device has delivered its part" {
    var a: Arrivals = .{ .nd = 2 };
    var r = a.arrive(1, 0, immediate(1));
    try std.testing.expect(r.from > r.to);
    r = a.arrive(1, 0, immediate(2));
    try std.testing.expect(r.from > r.to);
    r = a.arrive(1, 1, immediate(1));
    try std.testing.expectEqual(@as(u64, 1), r.from);
    try std.testing.expectEqual(@as(u64, 1), r.to);
    r = a.arrive(1, 1, immediate(2));
    try std.testing.expectEqual(@as(u64, 2), r.from);
    try std.testing.expectEqual(@as(u64, 2), r.to);
    // another peer is independent
    r = a.arrive(3, 1, immediate(1));
    try std.testing.expect(r.from > r.to);
    // one device: every completion publishes its sequence
    var b: Arrivals = .{ .nd = 1 };
    r = b.arrive(0, 0, immediate(1));
    try std.testing.expectEqual(@as(u64, 1), r.to);
}

test "a slot is sent after its last write: peers x devices completions" {
    var s: Sends = .{ .need = 6, .slots = 4 };
    for (0..5) |_| try std.testing.expectEqual(@as(?u64, null), s.complete(1));
    try std.testing.expectEqual(@as(?u64, 1), s.complete(1));
    // sequence 5 reuses slot 1 and starts its own count
    for (0..5) |_| try std.testing.expectEqual(@as(?u64, null), s.complete(5));
    try std.testing.expectEqual(@as(?u64, null), s.complete(2));
    try std.testing.expectEqual(@as(?u64, 5), s.complete(5));
}
