//! Host memory reporting, isolated request refusal and the pool-full retry through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Memory = api.Memory;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "a lane host reports its backend's memory counts, and none without them" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().memory(false) == null);
    const Counts = struct {
        resets: u32 = 0,
        fn read(ctx: ?*anyopaque, reset_peak: bool) ?Memory {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            if (reset_peak) c.resets += 1;
            return .{ .active = 5, .peak = if (reset_peak) 5 else 9 };
        }
    };
    var counts: Counts = .{};
    host.memory = .{ .ctx = &counts, .read = Counts.read };
    try std.testing.expectEqual(@as(u64, 9), host.engine().memory(false).?.peak);
    try std.testing.expectEqual(@as(u64, 5), host.engine().memory(true).?.peak);
    try std.testing.expectEqual(@as(u32, 1), counts.resets);
}

test "a request the backend refuses fails alone, in the backend's words" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_sampled = true };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Words = struct {
        fn text(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return if (err == error.SamplingRefused) "send temperature 0" else null;
        }
    };
    host.explain = .{ .text = Words.text };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        message: []const u8 = "",
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                },
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var plain: Box = .{};
    var sampled: Box = .{};
    const greedy: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    const keyed: Request = .{ .prompt = &prompt, .max_tokens = 64, .sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 5 } };
    const e = host.engine();
    try e.submit(1, &greedy, .{ .ctx = &plain, .event = Box.event });
    try e.submit(2, &keyed, .{ .ctx = &sampled, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, sampled.wait());
    try std.testing.expectEqualStrings("send temperature 0", sampled.message);
    try std.testing.expectEqual(Reason.length, plain.wait());
    try std.testing.expectEqual(@as(usize, 64), plain.tokens);
}

test "a request refused for room while another stream runs is tried again within a second" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_full = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn state(b: *@This()) struct { ?Reason, usize } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ b.done, b.tokens };
        }
        // the reason the request ended with, or null when it has not ended within `ms`
        fn waitFor(b: *@This(), ms: u32) ?Reason {
            for (0..ms) |_| {
                if (b.state()[0]) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
            return b.state()[0];
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var long: Box = .{};
    var short: Box = .{};
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &long, .event = Box.event });
    while (long.state()[1] == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // the second request's prompt pass is refused once for room (error.ContextFull) while the first stream runs on:
    // it goes back to the queue and is tried again within a second, not only when that stream ends
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = &short, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, Reason.length), short.waitFor(5000));
    try std.testing.expectEqual(@as(usize, 8), short.state()[1]);
    try std.testing.expectEqual(@as(usize, 0), target.refuse_full);
    try std.testing.expect(long.state()[0] == null);
    e.cancel(1);
    try std.testing.expect(long.waitFor(5000) != null);
}

test "after a long prompt pass, a stream near its max_tokens ends before the next request's pass" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const Slow = struct {
        // each prompt pass takes 1.1 s, a long pass for the host (its whole pass runs inside the admission)
        fn hook(_: *anyopaque, _: *lanes.stream.Stream, _: usize) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(1100), .awake) catch {};
        }
    };
    var unused: u8 = 0;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_hook = Slow.hook, .prefill_hook_ctx = &unused };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Seq = struct {
        mutex: std.Io.Mutex = .init,
        n: u32 = 0,
        a_done: ?u32 = null,
        b_first: ?u32 = null,
        b_done: bool = false,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const q: *@This() = @ptrCast(@alignCast(ctx));
            q.mutex.lockUncancelable(std.testing.io);
            defer q.mutex.unlock(std.testing.io);
            q.n += 1;
            switch (e.*) {
                .tokens => if (id == 2 and q.b_first == null) {
                    q.b_first = q.n;
                },
                .finished => if (id == 1) {
                    q.a_done = q.n;
                } else {
                    q.b_done = true;
                },
                else => {},
            }
        }
        fn done(q: *@This()) bool {
            q.mutex.lockUncancelable(std.testing.io);
            defer q.mutex.unlock(std.testing.io);
            return q.a_done != null and q.b_done;
        }
    };
    var seq: Seq = .{};
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &seq, .event = Seq.event });
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &seq, .event = Seq.event });
    for (0..10000) |_| {
        if (seq.done()) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(seq.done());
    // the first request (64 tokens: within finish_left) ended before the second's first token
    try std.testing.expect(seq.a_done.? < seq.b_first.?);
}

test "a prompt pass in pieces: a decoding stream keeps its rounds between them, and the reply is the whole pass's" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const Watch = struct {
        mutex: std.Io.Mutex = .init,
        a: usize = 0, // the first request's tokens delivered so far
        seen: std.ArrayList(usize) = .empty, // that count at each chunk of a prompt pass
        b: std.ArrayList(u32) = .empty,
        a_done: bool = false,
        b_done: bool = false,
        fn hook(ctx: *anyopaque, _: *lanes.stream.Stream, _: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            w.mutex.lockUncancelable(std.testing.io);
            defer w.mutex.unlock(std.testing.io);
            w.seen.append(gpa, w.a) catch {};
        }
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            w.mutex.lockUncancelable(std.testing.io);
            defer w.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| if (id == 1) {
                    w.a += t.len;
                } else w.b.appendSlice(gpa, t) catch {},
                .finished => if (id == 1) {
                    w.a_done = true;
                } else {
                    w.b_done = true;
                },
                else => {},
            }
        }
        fn until(w: *@This(), comptime what: enum { a_decodes, b_done, a_done }) !void {
            for (0..20000) |_| {
                w.mutex.lockUncancelable(std.testing.io);
                const ok = switch (what) {
                    .a_decodes => w.a > 0,
                    .b_done => w.b_done,
                    .a_done => w.a_done,
                };
                w.mutex.unlock(std.testing.io);
                if (ok) return;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
            return error.Timeout;
        }
    };
    var w: Watch = .{};
    defer w.seen.deinit(gpa);
    defer w.b.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 6, .prefill_hook = Watch.hook, .prefill_hook_ctx = &w };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    core.fill_chunks = 1;
    core.fill_rounds = 2;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const e = host.engine();
    const pa = [_]u32{ 2, 7, 1, 8 };
    const pb = [_]u32{ 3, 1, 4, 1, 5 };
    try e.submit(1, &.{ .prompt = &pa, .max_tokens = 1_000_000 }, .{ .ctx = &w, .event = Watch.event });
    try w.until(.a_decodes);
    try e.submit(2, &.{ .prompt = &pb, .max_tokens = 8 }, .{ .ctx = &w, .event = Watch.event });
    try w.until(.b_done);
    e.cancel(1);
    try w.until(.a_done);
    // the first pass ran whole (nothing decoded yet); the second in six pieces, the first stream decoding between them
    try std.testing.expectEqual(@as(usize, 12), w.seen.items.len);
    for (w.seen.items[0..6]) |n| try std.testing.expectEqual(@as(usize, 0), n);
    for (w.seen.items[7..12], w.seen.items[6..11]) |now, before| try std.testing.expect(now > before);
    // the second reply is the fake's own continuation of its prompt
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &pb);
    try std.testing.expectEqual(@as(usize, 8), w.b.items.len);
    for (w.b.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

test "a prompt within the first piece runs whole: the decoding stream waits for it as before" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const Watch = struct {
        mutex: std.Io.Mutex = .init,
        a: usize = 0, // the first request's tokens delivered so far
        seen: std.ArrayList(usize) = .empty, // that count at each chunk of a prompt pass
        b: std.ArrayList(u32) = .empty,
        a_done: bool = false,
        b_done: bool = false,
        fn hook(ctx: *anyopaque, _: *lanes.stream.Stream, _: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            w.mutex.lockUncancelable(std.testing.io);
            defer w.mutex.unlock(std.testing.io);
            w.seen.append(gpa, w.a) catch {};
        }
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            w.mutex.lockUncancelable(std.testing.io);
            defer w.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| if (id == 1) {
                    w.a += t.len;
                } else w.b.appendSlice(gpa, t) catch {},
                .finished => if (id == 1) {
                    w.a_done = true;
                } else {
                    w.b_done = true;
                },
                else => {},
            }
        }
        fn until(w: *@This(), comptime what: enum { a_decodes, b_done, a_done }) !void {
            for (0..20000) |_| {
                w.mutex.lockUncancelable(std.testing.io);
                const ok = switch (what) {
                    .a_decodes => w.a > 0,
                    .b_done => w.b_done,
                    .a_done => w.a_done,
                };
                w.mutex.unlock(std.testing.io);
                if (ok) return;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
            return error.Timeout;
        }
    };
    var w: Watch = .{};
    defer w.seen.deinit(gpa);
    defer w.b.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 6, .prefill_hook = Watch.hook, .prefill_hook_ctx = &w };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    core.fill_chunks = 1;
    core.fill_rounds = 2;
    core.fill_first = 6; // the whole six-chunk prompt fits the first piece
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const e = host.engine();
    const pa = [_]u32{ 2, 7, 1, 8 };
    const pb = [_]u32{ 3, 1, 4, 1, 5 };
    try e.submit(1, &.{ .prompt = &pa, .max_tokens = 1_000_000 }, .{ .ctx = &w, .event = Watch.event });
    try w.until(.a_decodes);
    try e.submit(2, &.{ .prompt = &pb, .max_tokens = 8 }, .{ .ctx = &w, .event = Watch.event });
    try w.until(.b_done);
    e.cancel(1);
    try w.until(.a_done);
    // both passes ran whole: the first stream delivered nothing while the second's six chunks ran
    try std.testing.expectEqual(@as(usize, 12), w.seen.items.len);
    for (w.seen.items[0..6]) |n| try std.testing.expectEqual(@as(usize, 0), n);
    for (w.seen.items[7..12]) |n| try std.testing.expectEqual(w.seen.items[6], n);
    // the second reply is the fake's own continuation of its prompt
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &pb);
    try std.testing.expectEqual(@as(usize, 8), w.b.items.len);
    for (w.b.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

/// A long pass in pieces (cut_in set, nothing decoding), a request sent in its first chunk: chunk order, both replies.
const CutIn = struct {
    mutex: std.Io.Mutex = .init,
    seen: std.ArrayList([2]usize) = .empty,
    a: std.ArrayList(u32) = .empty,
    b: std.ArrayList(u32) = .empty,
    first: bool = false, // the long pass's first chunk is running
    sent: bool = false, // the second request is queued
    a_done: bool = false,
    b_done: bool = false,

    fn hook(ctx: *anyopaque, s: *lanes.stream.Stream, chunk: usize) void {
        const w: *CutIn = @ptrCast(@alignCast(ctx));
        w.mutex.lockUncancelable(std.testing.io);
        w.seen.append(std.testing.allocator, .{ s.prompt_len, chunk }) catch {};
        const wait = s.prompt_len == long.len and chunk == 0;
        if (wait) w.first = true;
        w.mutex.unlock(std.testing.io);
        // the long pass's first chunk holds the engine until the second request is in the queue
        if (wait) w.until(.sent) catch {};
    }

    fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
        const w: *CutIn = @ptrCast(@alignCast(ctx));
        w.mutex.lockUncancelable(std.testing.io);
        defer w.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| (if (id == 1) &w.a else &w.b).appendSlice(std.testing.allocator, t) catch {},
            .finished => if (id == 1) {
                w.a_done = true;
            } else {
                w.b_done = true;
            },
            else => {},
        }
    }

    fn until(w: *CutIn, comptime what: enum { first, sent, done }) !void {
        for (0..20000) |_| {
            w.mutex.lockUncancelable(std.testing.io);
            const ok = switch (what) {
                .first => w.first,
                .sent => w.sent,
                .done => w.a_done and w.b_done,
            };
            w.mutex.unlock(std.testing.io);
            if (ok) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    const long = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8 };

    /// The long request, then `second`: the chunk order (caller frees), each reply checked as the fake's.
    fn run(second: []const u32, opts: struct { cut_in: u32 = 4, refuse_full: usize = 0 }) ![][2]usize {
        const gpa = std.testing.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var w: CutIn = .{};
        defer w.a.deinit(gpa);
        defer w.b.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 6, .prefill_hook = hook, .prefill_hook_ctx = &w, .refuse_full = opts.refuse_full };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        core.fill_chunks = 1;
        core.fill_rounds = 2;
        core.cut_in = opts.cut_in;
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        try host.start();
        defer host.stop();
        const e = host.engine();
        try e.submit(1, &.{ .prompt = &long, .max_tokens = 6 }, .{ .ctx = &w, .event = event });
        try w.until(.first);
        try e.submit(2, &.{ .prompt = second, .max_tokens = 6 }, .{ .ctx = &w, .event = event });
        w.mutex.lockUncancelable(std.testing.io);
        w.sent = true;
        w.mutex.unlock(std.testing.io);
        try w.until(.done);
        try std.testing.expectEqual(@as(usize, 0), target.refuse_full);
        for ([_][]const u32{ &long, second }, [_][]const u32{ w.a.items, w.b.items }) |prompt, reply| {
            var history: std.ArrayList(u32) = .empty;
            defer history.deinit(gpa);
            try history.appendSlice(gpa, prompt);
            try std.testing.expectEqual(@as(usize, 6), reply.len);
            for (reply) |t| {
                try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
                try history.append(gpa, t);
            }
        }
        return w.seen.toOwnedSlice(gpa);
    }
};

test "a short prompt cuts in: its whole pass runs between a long pass's pieces, and both replies are unchanged" {
    const order = try CutIn.run(&.{ 3, 1, 4 }, .{});
    defer std.testing.allocator.free(order);
    try std.testing.expectEqual(@as(usize, 12), order.len);
    // the long pass's first chunk, the short pass whole, then the long pass's other five chunks
    try std.testing.expectEqual([2]usize{ CutIn.long.len, 0 }, order[0]);
    for (order[1..7], 0..) |o, i| try std.testing.expectEqual([2]usize{ 3, i }, o);
    for (order[7..12], 1..) |o, i| try std.testing.expectEqual([2]usize{ CutIn.long.len, i }, o);
}

test "a prompt longer than cut_in waits for the pass in pieces, as before" {
    const order = try CutIn.run(&.{ 3, 1, 4, 1, 5 }, .{});
    defer std.testing.allocator.free(order);
    try std.testing.expectEqual(@as(usize, 12), order.len);
    for (order[0..6], 0..) |o, i| try std.testing.expectEqual([2]usize{ CutIn.long.len, i }, o);
    for (order[6..12], 0..) |o, i| try std.testing.expectEqual([2]usize{ 5, i }, o);
}

test "a short prompt refused for room as it cuts in waits in the queue and then runs, as it does without cut_in" {
    // the fake refuses the short pass once (error.ContextFull) while the long pass holds its lane; its reply is whole
    for ([_]u32{ 4, 0 }) |cut_in| {
        const order = try CutIn.run(&.{ 3, 1, 4 }, .{ .cut_in = cut_in, .refuse_full = 1 });
        defer std.testing.allocator.free(order);
        try std.testing.expectEqual(@as(usize, 12), order.len);
    }
}
