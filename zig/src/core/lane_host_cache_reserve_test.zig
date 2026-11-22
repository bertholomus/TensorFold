//! Reservation waits for a successful placement only when snapshot drops reach every rank immediately.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const ends = @import("lane_host_end.zig");
const a = std.testing.allocator;

const Snaps = struct {
    target: *lanes.fake.Fake,
    drops: usize = 0,

    fn bytes(_: *anyopaque, _: u32) u64 {
        return 64;
    }
    fn save(ctx: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const s: *Snaps = @ptrCast(@alignCast(ctx));
        return s.target.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ctx: *anyopaque, saved: pc.Saved) void {
        const s: *Snaps = @ptrCast(@alignCast(ctx));
        s.drops += 1;
        s.target.drop(saved);
    }
    fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
};

fn seed(store: *pc.Store, target: *lanes.fake.Fake, prompt: []const u32) !void {
    var stream = try lanes.Stream.init(a, .{ .id = "seed", .prompt = prompt[0..4], .max_new = 1, .drafts = false });
    defer stream.deinit(a);
    defer target.backend().release(&stream);
    try target.backend().prefill(&stream);
    try std.testing.expect(store.keep(prompt, 4, &stream, &.{}, &.{}));
}

pub fn check(comptime admitOne: anytype, reject: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var snaps: Snaps = .{ .target = &target };
    var store = pc.Store.init(a, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_prompt = 0, .reserve_before_pass = false }, 128);
    defer store.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 3 });
    defer {
        ends.closeAll(&host);
        host.stop();
    }
    const sink: api.Sink = .{ .ctx = &snaps, .event = Snaps.event };
    const prior: api.Request = .{ .prompt = &.{ 61, 62, 63, 64 }, .max_tokens = 512, .drafts = false };
    try host.engine().submit(1, &prior, sink);
    try std.testing.expect(admitOne(&host));
    try std.testing.expectEqual(@as(usize, 1), core.activeCount());

    const warm_a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const warm_b = [_]u32{ 21, 22, 23, 24, 25, 26, 27, 28 };
    const cold = [_]u32{ 41, 42, 43, 44, 45, 46, 47, 48 };
    try seed(&store, &target, &warm_a);
    try seed(&store, &target, &warm_b);
    try std.testing.expectEqual(@as(u64, 128), store.held);
    host.cache = &store;

    const request: api.Request = .{ .prompt = &cold, .history_len = 4, .max_tokens = 512, .drafts = false };
    const prefills = target.prefill_count;
    if (reject) target.refuse_full = 1;
    try host.engine().submit(2, &request, sink);
    const admitted = admitOne(&host);
    if (reject) {
        try std.testing.expect(!admitted);
        try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
        try std.testing.expectEqual(@as(api.Id, 2), host.queued.items[0].id);
        try std.testing.expectEqual(@as(usize, 1), core.activeCount());
        try std.testing.expectEqual(prefills, target.prefill_count);
        try std.testing.expectEqual(@as(usize, 0), target.refuse_full);
        try std.testing.expectEqual(@as(u64, 0), store.counts.evicted);
        try std.testing.expectEqual(@as(usize, 0), snaps.drops);
        try std.testing.expect(store.find(&warm_a, &.{}, &.{}) != null);
        try std.testing.expect(store.find(&warm_b, &.{}, &.{}) != null);
    } else {
        try std.testing.expect(admitted);
        try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
        try std.testing.expectEqual(@as(usize, 2), core.activeCount());
        try std.testing.expectEqual(prefills + 1, target.prefill_count);
        try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
        try std.testing.expectEqual(@as(usize, 1), snaps.drops);
        try std.testing.expect(store.find(&warm_a, &.{}, &.{}) == null);
        try std.testing.expect(store.find(&warm_b, &.{}, &.{}) != null);
        try std.testing.expect(store.find(&cold, &.{}, &.{}) != null);
    }
    try std.testing.expectEqual(@as(u64, 128), store.held);
    try std.testing.expect(store.held <= store.budget);
}
