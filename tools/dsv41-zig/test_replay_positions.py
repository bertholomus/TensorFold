#!/usr/bin/env python3
"""CPU regression/reproducer for replayCut's position view; requires Zig from .zig-version."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PRELUDE = r'''
const std = @import("std");
const tri = struct { const decode_rows = 16; };
const Caches = struct {};
const Shared = struct {};
const Ops = struct {
    fn copyRows(_: Ops, _: u8, _: u64, _: usize, _: u64, _: usize, _: usize, _: usize) !void {}
};
const Layer = struct { comp_wkv: ?u8 = 1 };
const Engine = struct {
    c: struct { hc: usize = 4, hidden: usize = 8 } = .{},
    w: struct { layers: [1]Layer = .{.{}} } = .{},
    ops: Ops = .{}, s: u8 = 0,
};
const Chunk = struct {
    start: usize, n: usize, pos: u64, pos0: u64,
    h: u64 = 0, h_alt: u64 = 0, pre: u64 = 0, pre_f: u64 = 0,
};
fn attnMixes(_: *const Engine, _: *Chunk, _: usize) !void {}
fn kvSourceUpdate(_: *const Engine, _: *Chunk, _: *const Caches, _: *Shared, _: usize) !void {}
var uploads: usize = 0;
fn upload(_: *const Engine, dst: u64, src: *const anyopaque, bytes: usize) !void {
    const to: [*]u8 = @ptrFromInt(dst);
    const from: [*]const u8 = @ptrCast(src);
    @memcpy(to[0..bytes], from[0..bytes]);
    uploads += 1;
}
'''
TESTS = r'''
fn check(origin: usize, remaining: usize, noncontiguous: bool) !void {
    const e: Engine = .{};
    const cs: Caches = .{};
    var sh: Shared = .{};
    var host: [2048]i64 = undefined;
    var dev: [2048]i64 align(16) = undefined;
    for (0..2048) |i| {
        host[i] = @intCast(origin + i * @as(usize, if (noncontiguous) 3 else 1));
        dev[i] = host[i];
    }
    const expected = host;
    const base = @intFromPtr(&dev);
    var ch: Chunk = .{ .start = origin, .n = 2048, .pos = base, .pos0 = base };
    const first = 2048 - remaining;
    uploads = 0;
    try std.testing.expect(try replayCut(&e, &ch, &cs, &sh, 0, origin + first, &host));
    try std.testing.expectEqual(origin + first, ch.start);
    try std.testing.expectEqual(remaining, ch.n);
    // This branch's replayCut uploads every cut's surviving positions into pos0 (replay_positions.md).
    const fixed = first > 0;
    if (remaining <= 16) try std.testing.expectEqual(@as(u64, 0), ch.pos % 16);
    try std.testing.expectEqual(@as(usize, if (fixed) 1 else 0), uploads);
    try std.testing.expectEqual(if (fixed) base else base + first * 8, ch.pos);
    const positions: [*]const i64 = @ptrFromInt(ch.pos);
    try std.testing.expectEqualSlices(i64, expected[first..], positions[0..remaining]);
    try std.testing.expectEqualSlices(i64, &expected, &host);
}
test "reported lengths and adjacent replay boundaries" {
    const lengths = [_]usize{ 10353, 10363, 39039, 123003, 39038, 39040, 123002, 123004 };
    for (lengths) |length| {
        const replay = length - 128;
        const origin = replay / 2048 * 2048;
        try check(origin, 2048 - (replay - origin), false);
    }
}
test "all small odd cuts, even cuts, and larger views preserve exact host positions" {
    for ([_]usize{ 0, 8192, 36864, 120832 }) |origin| {
        for (1..34) |remaining| {
            try check(origin, remaining, false);
            try check(origin, remaining, true);
        }
    }
}
test "no cut and encoder-only chunks leave positions alone" {
    const e: Engine = .{};
    const cs: Caches = .{};
    var sh: Shared = .{};
    var host: [2048]i64 = undefined;
    var dev: [2048]i64 align(16) = undefined;
    const base = @intFromPtr(&dev);
    var ch: Chunk = .{ .start = 2048, .n = 2048, .pos = base, .pos0 = base };
    uploads = 0;
    try std.testing.expect(try replayCut(&e, &ch, &cs, &sh, 0, 0, &host));
    try std.testing.expect(try replayCut(&e, &ch, &cs, &sh, 0, 2048, &host));
    try std.testing.expect(!try replayCut(&e, &ch, &cs, &sh, 0, 4096, &host));
    try std.testing.expectEqual(@as(usize, 0), uploads);
    try std.testing.expectEqual(base, ch.pos);
    try std.testing.expectEqual(@as(usize, 2048), ch.start);
    try std.testing.expectEqual(@as(usize, 2048), ch.n);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", default=os.environ.get("ZIG", "zig"))
    parser.add_argument("--source", type=Path, default=ROOT / "zig/src/families/dsv41/prompt.zig")
    args = parser.parse_args()
    source = args.source.read_text()
    # Compile the actual function, not a Python translation; stub only GPU collaborators.
    start = source.index("pub fn replayCut(")
    end = source.index("\n}\n", start) + 3
    with tempfile.TemporaryDirectory(prefix="replay-positions-") as tmp:
        harness = Path(tmp) / "replay_positions.zig"
        harness.write_text(PRELUDE + source[start:end] + TESTS)
        subprocess.run([args.zig, "test", str(harness)], check=True)


if __name__ == "__main__":
    main()
