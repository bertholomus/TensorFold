//! The live status line (connections, decode and prefill tok/s), redrawn in a terminal, and /health's copy of it.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;

/// Python's ``round(x, 1)``.
fn round1(x: f64) f64 {
    var buf: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "{d:.1}", .{x}) catch return x;
    return std.fmt.parseFloat(f64, text) catch x;
}

/// /health's ``live``: open requests, how many wait, decode and prefill tokens a second, and the optional cache
/// account (absent when the engine cannot report one).
pub fn snapshot(a: Allocator, engine: api.Engine) Allocator.Error!json.Value {
    var status: api.Status = .{};
    engine.status(&status, &.{});
    const o = try json.newObject(a);
    try o.put(a, "connections", try json.intValue(a, status.running + status.waiting));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = round1(status.decode_tokens_per_second) });
    try o.put(a, "prefill_tokens_per_second", .{ .float = round1(status.prefill_tokens_per_second) });
    if (status.cache) |c| try o.put(a, "cache", try cacheValue(a, c));
    return .{ .object = o };
}

/// The additive ``live.cache`` group: the shared pool's disjoint categories, then the prompt-state store's storage and
/// counters. Missing groups stay absent; a disabled store says so instead of reporting healthy zeroes.
fn cacheValue(a: Allocator, c: api.CacheStatus) Allocator.Error!json.Value {
    const o = try json.newObject(a);
    try o.put(a, "schema_version", try json.intValue(a, c.schema_version));
    try o.put(a, "engine_epoch", try json.intValue(a, c.engine_epoch));
    try o.put(a, "observed_unix_ns", try json.intValue(a, c.observed_unix_ns));
    try o.put(a, "consistent", .{ .bool = c.consistent });
    try o.put(a, "context_limit_positions", try json.intValue(a, c.context_limit_positions));
    try o.put(a, "stream_capacity", try json.intValue(a, c.stream_capacity));
    if (c.pool) |p| {
        const po = try json.newObject(a);
        try po.put(a, "capacity_positions", try json.intValue(a, p.capacity_positions));
        try po.put(a, "active_reserved_positions", try json.intValue(a, p.active_reserved_positions));
        try po.put(a, "retained_prefix_positions", try json.intValue(a, p.retained_prefix_positions));
        try po.put(a, "retained_growth_positions", try json.intValue(a, p.retained_growth_positions));
        try po.put(a, "unclaimed_positions", try json.intValue(a, p.unclaimed_positions));
        try po.put(a, "valid_states", try json.intValue(a, p.valid_states));
        try po.put(a, "invalidations_total", try json.intValue(a, p.invalidations_total));
        if (p.placement_blocks_total) |v| try po.put(a, "placement_blocks_total", try json.intValue(a, v));
        try o.put(a, "pool", .{ .object = po });
    }
    if (c.prompt_state) |s| {
        const so = try json.newObject(a);
        try so.put(a, "enabled", .{ .bool = s.enabled });
        try so.put(a, "budget_bytes", try json.intValue(a, s.budget_bytes));
        try so.put(a, "held_bytes", try json.intValue(a, s.held_bytes));
        try so.put(a, "spare_bytes", try json.intValue(a, s.spare_bytes));
        try so.put(a, "entries", try json.intValue(a, s.entries));
        try so.put(a, "hits_total", try json.intValue(a, s.hits_total));
        try so.put(a, "misses_total", try json.intValue(a, s.misses_total));
        try so.put(a, "evictions_total", try json.intValue(a, s.evictions_total));
        try so.put(a, "refusals_total", try json.intValue(a, s.refusals_total));
        try so.put(a, "failures_total", try json.intValue(a, s.failures_total));
        try so.put(a, "failures_keep_total", try json.intValue(a, s.failures_keep_total));
        try so.put(a, "failures_restore_total", try json.intValue(a, s.failures_restore_total));
        try so.put(a, "failures_peer_forget_total", try json.intValue(a, s.failures_peer_forget_total));
        try o.put(a, "prompt_state", .{ .object = so });
    }
    return .{ .object = o };
}

/// ``[tensorfold] 3 connections (1 waiting) · decode 142 tok/s · prefill 1,210 tok/s``.
pub fn line(buf: []u8, status: api.Status) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    const open = status.running + status.waiting;
    w.print("[tensorfold] {d} connection{s}", .{ open, if (open == 1) "" else "s" }) catch {};
    if (status.waiting > 0) w.print(" ({d} waiting)", .{status.waiting}) catch {};
    var d: [32]u8 = undefined;
    var p: [32]u8 = undefined;
    w.print(" \u{b7} decode {s} tok/s \u{b7} prefill {s} tok/s", .{ grouped(&d, round1(status.decode_tokens_per_second)), grouped(&p, round1(status.prefill_tokens_per_second)) }) catch {};
    return w.buffered();
}

/// ``f"{x:,.0f}"``: rounded half to even, thousands separated by commas.
fn grouped(buf: []u8, x: f64) []const u8 {
    const n: u64 = @intFromFloat(@max(0, @round(x) - @as(f64, if (@abs(x - @trunc(x)) == 0.5 and @mod(@trunc(x), 2) == 0) 1 else 0)));
    var digits: [24]u8 = undefined;
    const text = std.fmt.bufPrint(&digits, "{d}", .{n}) catch return "0";
    var w: std.Io.Writer = .fixed(buf);
    for (text, 0..) |ch, i| {
        if (i > 0 and (text.len - i) % 3 == 0) w.writeByte(',') catch {};
        w.writeByte(ch) catch {};
    }
    return w.buffered();
}

/// Redraws the line every half second while the server runs in a terminal.
pub const Ticker = struct {
    engine: api.Engine,
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn start(t: *Ticker) void {
        t.thread = std.Thread.spawn(.{}, run, .{t}) catch null;
    }

    fn run(t: *Ticker) void {
        while (!t.stop.load(.acquire)) {
            std.Io.sleep(log.io(), .fromMilliseconds(500), .awake) catch {};
            if (t.stop.load(.acquire)) break;
            var status: api.Status = .{};
            t.engine.status(&status, &.{});
            var buf: [256]u8 = undefined;
            log.status(line(&buf, status), columns());
        }
    }

    pub fn finish(t: *Ticker) void {
        t.stop.store(true, .release);
        if (t.thread) |th| th.join();
        log.clearStatus();
    }
};

const Winsize = extern struct { row: u16, col: u16, xpixel: u16, ypixel: u16 };

/// The terminal's width (``COLUMNS``, then the tty), else 100.
pub var columns_env: ?[]const u8 = null;

fn columns() usize {
    if (columns_env) |c| if (std.fmt.parseInt(usize, c, 10)) |n| return n else |_| {};
    var ws: Winsize = .{ .row = 0, .col = 0, .xpixel = 0, .ypixel = 0 };
    const os = @import("builtin").os.tag;
    if (os == .linux) {
        if (std.os.linux.ioctl(1, std.os.linux.T.IOCGWINSZ, @intFromPtr(&ws)) == 0 and ws.col > 0) return ws.col;
    } else if (os.isDarwin()) {
        if (std.c.ioctl(1, @bitCast(@as(u32, 0x40087468)), &ws) == 0 and ws.col > 0) return ws.col;
    }
    return 100;
}

/// Whether to draw: stdout is a terminal and TENSORFOLD_NO_LIVE is not 1.
pub fn wanted(io: std.Io, no_live: bool) bool {
    if (no_live) return false;
    return std.Io.File.stdout().isTty(io) catch false;
}

test "status line" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("[tensorfold] 3 connections (1 waiting) \u{b7} decode 142 tok/s \u{b7} prefill 1,210 tok/s", line(&buf, .{ .running = 2, .waiting = 1, .decode_tokens_per_second = 142.2, .prefill_tokens_per_second = 1210.4 }));
}

test "the cache group is additive, keeps missing groups absent and names a disabled store" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try cacheValue(a, .{
        .consistent = true,
        .engine_epoch = 1793692800000000000,
        .observed_unix_ns = 1793692801000000000,
        .context_limit_positions = 1048576,
        .stream_capacity = 16,
        .pool = .{ .capacity_positions = 8388608, .active_reserved_positions = 0, .retained_prefix_positions = 0, .retained_growth_positions = 0, .unclaimed_positions = 8388608, .valid_states = 0, .invalidations_total = 0, .placement_blocks_total = null },
        .prompt_state = .{ .enabled = false },
    });
    const text = try json.stringify(a, v, .{ .compact = true });
    try std.testing.expect(std.mem.indexOf(u8, text, "\"capacity_positions\":8388608") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"unclaimed_positions\":8388608") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"enabled\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "placement_blocks_total") == null); // unknown stays absent
    try std.testing.expect(std.mem.indexOf(u8, text, "\"pool\"") != null);
}
