//! Deterministic ContextFull queue checks: no engine thread or wall-clock sleeps.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const ends = @import("lane_host_end.zig");
const LaneHost = api.LaneHost;
const warm = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
const cold = [_]u32{ 41, 42, 43, 44, 45, 46, 47, 48 };

/// Only the cold request cannot be placed beside an existing stream; the warm one fits.
const Gate = struct {
    refusals: usize = 0,

    fn prefill(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const gate: *Gate = @ptrCast(@alignCast(target.prefill_hook_ctx.?));
        if (stream.prompt()[0] == cold[0] and target.lanes.count() > 0) {
            gate.refusals += 1;
            return error.ContextFull;
        }
        try target.backend().prefill(stream);
    }

    const vtable: lanes.backend.Backend.VTable = blk: {
        var target: lanes.fake.Fake = undefined;
        var methods = target.backend().vtable.*;
        methods.prefill = prefill;
        break :blk methods;
    };

    fn backend(target: *lanes.fake.Fake) lanes.backend.Backend {
        return .{ .ptr = target, .vtable = &vtable };
    }
};

const Snaps = struct {
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return target.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        target.drop(saved);
    }
};

const Seen = struct {
    cold_cached: ?u32 = null,
    warm_cached: ?u32 = null,
    cancelled: usize = 0,
    other_finished: usize = 0,

    fn event(ptr: *anyopaque, id: api.Id, value: *const api.Event) void {
        const seen: *Seen = @ptrCast(@alignCast(ptr));
        switch (value.*) {
            .prefilled => |cached| switch (id) {
                2 => seen.cold_cached = cached,
                3 => seen.warm_cached = cached,
                else => {},
            },
            .finished => |done| {
                if (id == 2 and done.reason == .cancelled) {
                    seen.cancelled += 1;
                } else {
                    seen.other_finished += 1;
                }
            },
            else => {},
        }
    }
};

pub const Control = enum { progress, deadline, occupancy };

/// `admitOne` stays private to lane_host.zig; its tests pass it here to step the real scheduler synchronously.
pub fn exercise(comptime admitOne: anytype, comptime control: Control) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var gate: Gate = .{};
    target.prefill_hook_ctx = &gate;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, Gate.backend(&target), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_prompt = 0 }, 128);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    defer {
        ends.closeAll(&host);
        host.stop();
    }
    var seen: Seen = .{};
    const sink: api.Sink = .{ .ctx = &seen, .event = Seen.event };
    const driver = host.engine();
    const prior: api.Request = .{ .prompt = &.{ 61, 62, 63, 64 }, .max_tokens = 512, .drafts = false };
    const blocked: api.Request = .{ .prompt = &cold, .max_tokens = 512, .drafts = false };
    const fitting: api.Request = .{ .prompt = &warm, .max_tokens = 512, .drafts = false, .history_len = 4 };
    try driver.submit(1, &prior, sink);
    try std.testing.expect(admitOne(&host));
    if (control == .occupancy) {
        try driver.submit(4, &prior, sink);
        try std.testing.expect(admitOne(&host));
    }

    var seed = try lanes.Stream.init(gpa, .{ .id = "warm seed", .prompt = warm[0..4], .max_new = 1, .drafts = false });
    try target.backend().prefill(&seed);
    try std.testing.expect(store.keep(&warm, 4, &seed, &.{}, &.{}));
    target.backend().release(&seed);
    seed.deinit(gpa);
    host.cache = &store;

    try driver.submit(2, &blocked, sink);
    try driver.submit(3, &fitting, sink);
    try std.testing.expect(!admitOne(&host));
    try std.testing.expectEqual(@as(usize, 1), gate.refusals);
    if (control != .progress) {
        // Keep the deadline well ahead without sleeping; expire it explicitly, or free only one occupied lane.
        host.queued.items[0].full_retry = std.math.maxInt(i96);
        if (control == .deadline) {
            host.queued.items[0].full_retry = 0;
        } else {
            driver.cancel(4);
            ends.takeCancels(&host);
            seen.other_finished = 0;
        }
        try std.testing.expectEqual(@as(usize, 1), core.activeCount());
        try std.testing.expect(!admitOne(&host));
        try std.testing.expectEqual(@as(usize, 2), gate.refusals);
        try std.testing.expect(seen.warm_cached == null); // the now-eligible head kept its priority
        try std.testing.expectEqual(@as(api.Id, 2), host.queued.items[0].id);
    }
    // Admission order and bounded retries must not depend on how long CI pauses this fixture.
    host.queued.items[0].full_retry = std.math.maxInt(i96);
    _ = admitOne(&host);
    try std.testing.expectEqual(@as(?u32, 4), seen.warm_cached);
    try std.testing.expect(seen.cold_cached == null);
    try std.testing.expectEqual(@as(usize, 2), core.activeCount());
    try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
    try std.testing.expectEqual(@as(api.Id, 2), host.queued.items[0].id);

    // No tight retry loop while occupancy and the deadline are unchanged.
    for (0..8) |_| try std.testing.expect(!admitOne(&host));
    try std.testing.expectEqual(@as(usize, if (control == .progress) 1 else 2), gate.refusals);

    if (control != .progress) {
        driver.cancel(2);
        ends.takeCancels(&host);
        try std.testing.expectEqual(@as(usize, 1), seen.cancelled);
        try std.testing.expectEqual(@as(usize, 0), seen.other_finished);
        try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
        try std.testing.expectEqual(@as(usize, 2), core.activeCount());
        try std.testing.expectEqual(@as(api.Id, 1), host.admitted.items[0].id);
        try std.testing.expectEqual(@as(api.Id, 3), host.admitted.items[1].id);
        return;
    }

    driver.cancel(1);
    driver.cancel(3);
    ends.takeCancels(&host);
    try std.testing.expectEqual(@as(usize, 0), core.activeCount());
    try std.testing.expect(admitOne(&host));
    try std.testing.expectEqual(@as(?u32, 0), seen.cold_cached);
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
    try std.testing.expectEqual(@as(usize, 1), core.activeCount());
}
