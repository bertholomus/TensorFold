//! The order a lane host fills waiting prompts in: foreground first, then the shortest (lane_host_fill.zig).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Engine = api.Engine;

test "a lane host fills the shortest waiting prompt first, so a short request waits on no long one" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Order = struct {
        mutex: std.Io.Mutex = .init,
        prefilled: [2]Id = .{ 0, 0 },
        count: usize = 0,
        done: usize = 0,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(std.testing.io);
            defer o.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => {
                    o.prefilled[o.count] = id;
                    o.count += 1;
                },
                .finished => o.done += 1,
                else => {},
            }
        }
    };
    var order: Order = .{};
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const short = [_]u32{ 2, 7, 1, 8 };
    const first: Request = .{ .prompt = &long, .max_tokens = 4 };
    const second: Request = .{ .prompt = &short, .max_tokens = 4 };
    const e = host.engine();
    // both wait before the engine thread starts: it admits them together and picks between them
    try e.submit(1, &first, .{ .ctx = &order, .event = Order.event });
    try e.submit(2, &second, .{ .ctx = &order, .event = Order.event });
    try host.start();
    defer host.stop();
    while (true) {
        order.mutex.lockUncancelable(std.testing.io);
        const d = order.done;
        order.mutex.unlock(std.testing.io);
        if (d == 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqualSlices(Id, &.{ 2, 1 }, &order.prefilled);
}

test "a foreground prompt arriving while a background one fills takes the next chunk" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Order = struct {
        mutex: std.Io.Mutex = .init,
        prefilled: [2]Id = .{ 0, 0 },
        count: usize = 0,
        done: usize = 0,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(std.testing.io);
            defer o.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => {
                    o.prefilled[o.count] = id;
                    o.count += 1;
                },
                .finished => o.done += 1,
                else => {},
            }
        }
    };
    var order: Order = .{};
    const short = [_]u32{ 2, 7, 1, 8 };
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const background: Request = .{ .prompt = &short, .max_tokens = 4, .background = true };
    const foreground: Request = .{ .prompt = &long, .max_tokens = 4 };
    const e = host.engine();
    // the background prompt's first chunk sends the foreground request: it is admitted while the other fills
    const Arrive = struct {
        engine: api.Engine,
        request: *const Request,
        order: *Order,
        sent: bool = false,
        fn call(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (a.sent) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.order, .event = Order.event }) catch {};
        }
    };
    var arrive: Arrive = .{ .engine = e, .request = &foreground, .order = &order };
    target.prefill_hook = Arrive.call;
    target.prefill_hook_ctx = &arrive;
    try e.submit(1, &background, .{ .ctx = &order, .event = Order.event });
    try host.start();
    defer host.stop();
    while (true) {
        order.mutex.lockUncancelable(std.testing.io);
        const d = order.done;
        order.mutex.unlock(std.testing.io);
        if (d == 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqualSlices(Id, &.{ 2, 1 }, &order.prefilled);
}

test "status reads a filling prompt's tokens while its pass runs, and none once it ends" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const Seen = struct {
        engine: Engine = undefined,
        lefts: std.ArrayList(u32) = .empty,
        done: std.atomic.Value(bool) = .init(false),

        fn hook(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            var status: api.Status = .{};
            seen.engine.status(&status, &.{}); // the pass runs without the host's lock: this can read
            if (status.fill_tokens_left) |left| seen.lefts.append(gpa, left) catch {};
        }
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            if (e.* == .finished) seen.done.store(true, .release);
        }
    };
    var seen: Seen = .{};
    seen.engine = host.engine();
    target.prefill_hook = Seen.hook;
    target.prefill_hook_ctx = &seen;
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
    while (!seen.done.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    defer seen.lefts.deinit(gpa);
    try std.testing.expect(seen.lefts.items.len >= 2); // each chunk of the pass sampled the status
    for (seen.lefts.items) |left| try std.testing.expect(left > 0 and left <= prompt.len);
    try std.testing.expect(seen.lefts.items[0] > seen.lefts.items[seen.lefts.items.len - 1]); // it counted down
    var status: api.Status = .{};
    seen.engine.status(&status, &.{});
    try std.testing.expect(status.fill_prompt_tokens == null and status.fill_tokens_left == null); // the pass is over
}

const Passes = struct {
    mutex: std.Io.Mutex = .init,
    prefilled: [2]Id = .{ 0, 0 },
    count: usize = 0,
    done: usize = 0,
    fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
        const o: *@This() = @ptrCast(@alignCast(ctx));
        o.mutex.lockUncancelable(std.testing.io);
        defer o.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => {
                o.prefilled[o.count] = id;
                o.count += 1;
            },
            .finished => o.done += 1,
            else => {},
        }
    }
    fn wait(o: *@This(), n: usize) void {
        while (true) {
            o.mutex.lockUncancelable(std.testing.io);
            const d = o.done;
            o.mutex.unlock(std.testing.io);
            if (d == n) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

test "a backend that holds one prompt pass at a time runs the pass it began to its end before the next comes in" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.steppedOne(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var order: Passes = .{};
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const short = [_]u32{ 2, 7, 1, 8 };
    const background: Request = .{ .prompt = &long, .max_tokens = 4, .background = true };
    const foreground: Request = .{ .prompt = &short, .max_tokens = 4 };
    const e = host.engine();
    // the long background prompt's first chunk sends a short foreground request, which a stepping backend of
    // several passes fills first (above): here it comes in once the long pass ends, and no two passes fill at once
    const Arrive = struct {
        engine: api.Engine,
        request: *const Request,
        order: *Passes,
        target: *lanes.fake.Fake,
        sent: bool = false,
        most: usize = 0,
        fn call(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            var n: usize = 0;
            var it = a.target.lanes.valueIterator();
            while (it.next()) |l| n += @intFromBool(l.filling);
            a.most = @max(a.most, n);
            if (a.sent) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.order, .event = Passes.event }) catch {};
        }
    };
    var arrive: Arrive = .{ .engine = e, .request = &foreground, .order = &order, .target = &target };
    target.prefill_hook = Arrive.call;
    target.prefill_hook_ctx = &arrive;
    try e.submit(1, &background, .{ .ctx = &order, .event = Passes.event });
    try host.start();
    defer host.stop();
    order.wait(2);
    try std.testing.expectEqualSlices(Id, &.{ 1, 2 }, &order.prefilled);
    try std.testing.expectEqual(@as(usize, 1), arrive.most);
}

test "a pass that began waits out fill_rounds of the live streams' rounds between two of its steps" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.steppedOne(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .fill_rounds = 3 });
    var order: Passes = .{};
    const first_prompt = [_]u32{ 2, 7, 1, 8 };
    const second_prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const first: Request = .{ .prompt = &first_prompt, .max_tokens = 2000 };
    const second: Request = .{ .prompt = &second_prompt, .max_tokens = 4 };
    const e = host.engine();
    // the first pass's last chunk sends the second request: the first stream decodes while the second fills
    const Seen = struct {
        engine: api.Engine,
        request: *const Request,
        order: *Passes,
        target: *lanes.fake.Fake,
        core: *lanes.Engine,
        sent: bool = false,
        at: [4]u64 = @splat(0), // the core's rounds run before each step of the second pass (a lone stream's
        // pipelined round makes no verify call, so the fake's own count of rounds would miss them)
        steps: usize = 0,
        fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len == second_prompt.len) {
                if (a.steps < a.at.len) a.at[a.steps] = a.core.steps;
                a.steps += 1;
                return;
            }
            if (chunk + 1 < a.target.prefill_chunks or a.sent) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.order, .event = Passes.event }) catch {};
        }
    };
    var seen: Seen = .{ .engine = e, .request = &second, .order = &order, .target = &target, .core = &core };
    target.prefill_hook = Seen.call;
    target.prefill_hook_ctx = &seen;
    try e.submit(1, &first, .{ .ctx = &order, .event = Passes.event });
    try host.start();
    defer host.stop();
    order.wait(2);
    try std.testing.expectEqual(@as(usize, 4), seen.steps);
    for (seen.at[0..3], seen.at[1..4]) |a, b| try std.testing.expect(b - a >= 3); // the first stream's rounds between
    try std.testing.expectEqualSlices(Id, &.{ 1, 2 }, &order.prefilled);
}

/// The replies a burst's requests finished (Passes keeps two prefill ids; a burst has three).
const Finished = struct {
    mutex: std.Io.Mutex = .init,
    done: usize = 0,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const o: *@This() = @ptrCast(@alignCast(ctx));
        o.mutex.lockUncancelable(std.testing.io);
        defer o.mutex.unlock(std.testing.io);
        switch (e.*) {
            .finished => o.done += 1,
            else => {},
        }
    }
    fn wait(o: *@This(), n: usize) void {
        while (true) {
            o.mutex.lockUncancelable(std.testing.io);
            const d = o.done;
            o.mutex.unlock(std.testing.io);
            if (d == n) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// The core's rounds run before each of three 4-token prompts' one-step passes, all waiting before the host starts.
fn burstSteps(one_pass: bool, fill_drain: u32) ![3]u64 {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (one_pass) target.steppedOne() else target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4, .fill_rounds = 4, .fill_drain = fill_drain });
    const Seen = struct {
        core: *lanes.Engine,
        at: [3]u64 = @splat(0),
        steps: usize = 0,
        fn call(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (a.steps < a.at.len) a.at[a.steps] = a.core.steps;
            a.steps += 1;
        }
    };
    var seen: Seen = .{ .core = &core };
    target.prefill_hook = Seen.call;
    target.prefill_hook_ctx = &seen;
    var order: Finished = .{};
    const prompts = [3][4]u32{ .{ 2, 7, 1, 8 }, .{ 3, 1, 4, 1 }, .{ 1, 6, 1, 8 } };
    var requests: [3]Request = undefined;
    const e = host.engine();
    for (&requests, &prompts, 1..) |*r, *p, id| {
        r.* = .{ .prompt = p, .max_tokens = 4 };
        try e.submit(@intCast(id), r, .{ .ctx = &order, .event = Finished.event });
    }
    try host.start();
    defer host.stop();
    order.wait(3);
    try std.testing.expectEqual(@as(usize, 3), seen.steps);
    return seen.at;
}

test "without fill_drain a burst of short prompts starts a round apart" {
    const at = try burstSteps(false, 0);
    try std.testing.expect(at[0] < at[1] and at[1] < at[2]);
}

test "fill_drain starts a burst of short prompts before the next round, as many rows of them as it allows" {
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0 }, &(try burstSteps(false, 64)));
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0 }, &(try burstSteps(true, 64))); // one pass at a time: admitted after
    const at = try burstSteps(false, 8); // two 4-token passes fit, the third waits for the next turn
    try std.testing.expect(at[0] == at[1] and at[2] > at[1]);
}
