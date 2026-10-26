//! CUDA graphs captured a key at a time; a capture that fails ran nothing, so its work runs eagerly instead.
const std = @import("std");
const cuda = @import("cuda");

/// A failed key's first hold in uses (each one eager), doubled at each further failure up to `max_hold`.
pub const first_hold: u32 = 1024;
pub const max_hold: u32 = first_hold << 6;

/// The uses a key runs eagerly after its `fails`-th failed capture (1 or more).
pub fn holdFor(fails: u32) u32 {
    const doublings: u5 = @intCast(@min(fails -| 1, std.math.log2_int(u32, max_hold / first_hold)));
    return first_hold << doublings;
}

/// The keys whose capture failed: how many more uses each runs eagerly, and how often its capture failed.
pub fn Holds(comptime Key: type) type {
    return struct {
        const Self = @This();
        pub const Hold = struct { left: u32, fails: u32 };

        map: std.AutoHashMapUnmanaged(Key, Hold) = .empty,

        pub fn deinit(h: *Self, gpa: std.mem.Allocator) void {
            h.map.deinit(gpa);
        }

        /// Whether `key` runs eagerly this time: each such use counts its hold down; at zero it is captured again.
        pub fn eager(h: *Self, key: Key) bool {
            const x = h.map.getPtr(key) orelse return false;
            if (x.left == 0) return false;
            x.left -= 1;
            return true;
        }

        /// Another failed capture of `key`: its new hold (fails 1 is its first failure, the one to log).
        pub fn fail(h: *Self, gpa: std.mem.Allocator, key: Key) std.mem.Allocator.Error!Hold {
            const got = try h.map.getOrPut(gpa, key);
            if (!got.found_existing) got.value_ptr.* = .{ .left = 0, .fails = 0 };
            got.value_ptr.fails +|= 1;
            got.value_ptr.left = holdFor(got.value_ptr.fails);
            return got.value_ptr.*;
        }

        /// `key` was captured: it is held no longer.
        pub fn clear(h: *Self, key: Key) void {
            _ = h.map.remove(key);
        }
    };
}

/// The graph calls a lazy capture makes, on CUDA (the host tests pass a fake with the same declarations).
pub const Cuda = struct {
    pub const Stream = cuda.Stream;
    pub const Graph = cuda.graph.Graph;
    pub const Exec = cuda.graph.Exec;

    pub fn begin(s: Stream) !void {
        return cuda.graph.beginCapture(s, .thread_local);
    }

    pub fn end(s: Stream) !Graph {
        return cuda.graph.endCapture(s);
    }

    pub fn instantiate(g: Graph) !Exec {
        return g.instantiate();
    }

    pub fn launch(x: Exec, s: Stream) !void {
        return x.launchOn(s);
    }

    pub fn dropGraph(g: *Graph) void {
        g.deinit();
    }

    pub fn dropExec(x: *Exec) void {
        x.deinit();
    }
};

/// A kind of work's graphs by key, each captured at its key's first use and kept (Value: the key's `exec` and more).
pub fn Lazy(comptime G: type, comptime Key: type, comptime Value: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        map: std.AutoHashMapUnmanaged(Key, Value) = .empty,
        holds: Holds(Key) = .{},
        failures: ?*std.atomic.Value(u64) = null, // the engine's failed captures (/metrics), when it counts them
        warned: u64 = 0, // the failures logged: each key's first

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        pub fn deinit(l: *Self) void {
            var it = l.map.valueIterator();
            while (it.next()) |v| G.dropExec(&v.exec);
            l.map.deinit(l.gpa);
            l.holds.deinit(l.gpa);
            l.* = undefined;
        }

        /// The captured keys.
        pub fn count(l: *const Self) usize {
            return l.map.count();
        }

        /// The key's kept graph, to replay.
        pub fn get(l: *Self, key: Key) ?*Value {
            return l.map.getPtr(key);
        }

        /// A key without a graph: the site's body() captured on `s`, kept as its value(exec), launched; eager if held.
        pub fn capture(l: *Self, s: G.Stream, key: Key, site: anytype) anyerror!void {
            if (l.holds.eager(key)) return site.body();
            // a failure up to here ran nothing on the device: the site's undo() puts its host state back
            var exec = record(s, site) catch |err| return l.fallback(key, site, err);
            const slot = l.map.getOrPut(l.gpa, key) catch |err| {
                G.dropExec(&exec);
                return l.fallback(key, site, err);
            };
            slot.value_ptr.* = site.value(exec);
            l.holds.clear(key);
            try G.launch(exec, s); // (a failed launch may have started work: it is returned, never run again)
        }

        /// The site's work captured and instantiated; a failure ends the capture first (its graph dropped).
        fn record(s: G.Stream, site: anytype) anyerror!G.Exec {
            try G.begin(s);
            site.body() catch |err| {
                if (G.end(s)) |got| {
                    var g = got;
                    G.dropGraph(&g);
                } else |_| {}
                return err;
            };
            var graph = try G.end(s);
            defer G.dropGraph(&graph);
            return G.instantiate(graph);
        }

        /// A failed capture of `key`: the site's host state back, the key held (its first failure logged), work eager.
        fn fallback(l: *Self, key: Key, site: anytype, err: anyerror) anyerror!void {
            site.undo();
            const hold = l.holds.fail(l.gpa, key) catch null;
            if (l.failures) |c| _ = c.fetchAdd(1, .monotonic);
            if (hold == null or hold.?.fails == 1) {
                l.warned += 1;
                std.log.warn("dsv41: a CUDA graph capture failed ({t}) at {any}: it runs eagerly for its next {d} uses, then is captured again (logged once a key)", .{ err, key, if (hold) |h| h.left else 0 });
            }
            return site.body();
        }
    };
}

/// A fake of the graph calls: each call's count, an error injected at a chosen call, the live graphs and execs.
const Fake = struct {
    pub const Stream = *Fake;
    pub const Graph = struct { f: *Fake };
    pub const Exec = struct { f: *Fake };
    pub const Call = enum { begin, end, instantiate, launch };

    fail_at: ?Call = null,
    fail_with: anyerror = error.CudaFailed,
    capturing: bool = false,
    calls: std.EnumArray(Call, u32) = .initFill(0),
    graphs: i32 = 0,
    execs: i32 = 0,

    fn step(f: *Fake, c: Call) !void {
        f.calls.getPtr(c).* += 1;
        if (f.fail_at == c) return f.fail_with;
    }

    pub fn begin(f: *Fake) !void {
        try f.step(.begin);
        f.capturing = true;
    }

    pub fn end(f: *Fake) !Graph {
        f.capturing = false; // (the driver ends a capture also when it reports it invalidated)
        try f.step(.end);
        f.graphs += 1;
        return .{ .f = f };
    }

    pub fn instantiate(g: Graph) !Exec {
        try g.f.step(.instantiate);
        g.f.execs += 1;
        return .{ .f = g.f };
    }

    pub fn launch(x: Exec, f: *Fake) !void {
        std.debug.assert(x.f == f);
        try f.step(.launch);
    }

    pub fn dropGraph(g: *Graph) void {
        g.f.graphs -= 1;
    }

    pub fn dropExec(x: *Exec) void {
        x.f.execs -= 1;
    }
};

const Kept = struct { exec: Fake.Exec };

/// A fake site: its host state (a counter its body bumps), its runs captured and eager, its undos, injected failures.
const FakeSite = struct {
    f: *Fake,
    state: u32 = 0,
    eager_runs: u32 = 0,
    undos: u32 = 0,
    body_fails: ?anyerror = null, // under capture
    eager_fails: ?anyerror = null,

    pub fn body(s: *FakeSite) !void {
        s.state += 1;
        if (s.f.capturing) {
            if (s.body_fails) |e| return e;
        } else {
            s.eager_runs += 1;
            if (s.eager_fails) |e| return e;
        }
    }

    pub fn undo(s: *FakeSite) void {
        s.undos += 1;
        s.state = 0;
    }

    pub fn value(_: *FakeSite, exec: Fake.Exec) Kept {
        return .{ .exec = exec };
    }
};

const TestKey = struct { rows: u32, bucket: u64 };
const TestLazy = Lazy(Fake, TestKey, Kept);

test "a capture that works is kept and launched once" {
    var f: Fake = .{};
    var l: TestLazy = .init(std.testing.allocator);
    defer l.deinit();
    var site: FakeSite = .{ .f = &f };
    const k: TestKey = .{ .rows = 4, .bucket = 1024 };
    try l.capture(&f, k, &site);
    try std.testing.expectEqual(@as(usize, 1), l.count());
    try std.testing.expect(l.get(k) != null);
    try std.testing.expectEqual(@as(u32, 1), f.calls.get(.launch));
    try std.testing.expectEqual(@as(u32, 0), site.eager_runs);
    try std.testing.expectEqual(@as(i32, 0), f.graphs); // the graph is dropped once instantiated
    try std.testing.expectEqual(@as(i32, 1), f.execs);
}

test "a failure in the captured work, at begin, at end capture or at instantiate runs the work eagerly from its start" {
    const Case = struct { at: ?Fake.Call, in_body: bool, with: anyerror };
    const cases = [_]Case{
        .{ .at = null, .in_body = true, .with = error.MissingTritonVariant }, // a launch the kit has no variant for
        .{ .at = null, .in_body = true, .with = error.CudaFailed }, // cuStreamWaitEvent refused inside the capture
        .{ .at = .end, .in_body = false, .with = error.CudaFailed }, // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901)
        .{ .at = .instantiate, .in_body = false, .with = error.CudaFailed }, // CUDA_ERROR_NOT_PERMITTED (800)
        .{ .at = .instantiate, .in_body = false, .with = error.OutOfDeviceMemory },
        .{ .at = .begin, .in_body = false, .with = error.CudaFailed },
    };
    for (cases) |c| {
        var f: Fake = .{ .fail_at = c.at, .fail_with = c.with };
        var l: TestLazy = .init(std.testing.allocator);
        defer l.deinit();
        var failures: std.atomic.Value(u64) = .init(0);
        l.failures = &failures;
        var site: FakeSite = .{ .f = &f, .body_fails = if (c.in_body) c.with else null };
        const k: TestKey = .{ .rows = 16, .bucket = 262144 };
        try l.capture(&f, k, &site);
        try std.testing.expectEqual(@as(u32, 1), site.undos);
        try std.testing.expectEqual(@as(u32, 1), site.eager_runs);
        try std.testing.expectEqual(@as(u32, 1), site.state); // the eager run started from the start state
        try std.testing.expect(!f.capturing); // a begun capture is always ended
        try std.testing.expectEqual(@as(u32, if (c.at == .begin) 0 else 1), f.calls.get(.end));
        try std.testing.expectEqual(@as(u32, 0), f.calls.get(.launch));
        try std.testing.expectEqual(@as(i32, 0), f.graphs);
        try std.testing.expectEqual(@as(i32, 0), f.execs);
        try std.testing.expectEqual(@as(usize, 0), l.count());
        try std.testing.expectEqual(@as(u64, 1), failures.load(.monotonic));
        try std.testing.expectEqual(first_hold, l.holds.map.get(k).?.left);
    }
}

test "a held key runs eagerly for its hold, then is captured again; a success ends the hold" {
    var f: Fake = .{ .fail_at = .instantiate };
    var l: TestLazy = .init(std.testing.allocator);
    defer l.deinit();
    var site: FakeSite = .{ .f = &f };
    const k: TestKey = .{ .rows = 2, .bucket = 8192 };
    try l.capture(&f, k, &site);
    try std.testing.expectEqual(@as(u32, 1), f.calls.get(.begin));
    for (0..first_hold) |_| try l.capture(&f, k, &site);
    try std.testing.expectEqual(@as(u32, 1), f.calls.get(.begin)); // no capture while held
    try std.testing.expectEqual(first_hold + 1, site.eager_runs);
    try std.testing.expectEqual(@as(u32, 1), site.undos); // a held use has nothing to undo
    f.fail_at = null;
    try l.capture(&f, .{ .rows = 3, .bucket = 8192 }, &site); // another key is not held
    try std.testing.expectEqual(@as(u32, 2), f.calls.get(.begin));
    try l.capture(&f, k, &site); // its hold is over: captured again, and kept
    try std.testing.expectEqual(@as(u32, 3), f.calls.get(.begin));
    try std.testing.expect(l.get(k) != null);
    try std.testing.expect(l.holds.map.get(k) == null);
    try std.testing.expectEqual(@as(u32, 2), f.calls.get(.launch));
}

test "each further failure of a key doubles its hold up to the most, and only its first is logged" {
    try std.testing.expectEqual(first_hold, holdFor(1));
    try std.testing.expectEqual(first_hold * 2, holdFor(2));
    try std.testing.expectEqual(max_hold, holdFor(7));
    try std.testing.expectEqual(max_hold, holdFor(std.math.maxInt(u32)));
    var f: Fake = .{ .fail_at = .end };
    var l: TestLazy = .init(std.testing.allocator);
    defer l.deinit();
    var site: FakeSite = .{ .f = &f };
    const k: TestKey = .{ .rows = 1, .bucket = 1024 };
    var want: u32 = first_hold;
    for (0..3) |i| {
        try l.capture(&f, k, &site);
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), l.holds.map.get(k).?.fails);
        try std.testing.expectEqual(want, l.holds.map.get(k).?.left);
        for (0..want) |_| try l.capture(&f, k, &site);
        want *= 2;
    }
    try std.testing.expectEqual(@as(u64, 1), l.warned);
    try l.capture(&f, .{ .rows = 1, .bucket = 2048 }, &site);
    try std.testing.expectEqual(@as(u64, 2), l.warned); // a new key logs its own first failure
}

test "an eager run after a failed capture returns its own error, and the key stays held" {
    var f: Fake = .{ .fail_at = .end };
    var l: TestLazy = .init(std.testing.allocator);
    defer l.deinit();
    var site: FakeSite = .{ .f = &f, .eager_fails = error.MissingTritonVariant };
    const k: TestKey = .{ .rows = 1, .bucket = 524288 };
    try std.testing.expectError(error.MissingTritonVariant, l.capture(&f, k, &site));
    try std.testing.expect(!f.capturing);
    try std.testing.expectEqual(first_hold, l.holds.map.get(k).?.left);
    try std.testing.expectError(error.MissingTritonVariant, l.capture(&f, k, &site)); // held: eager, no capture
    try std.testing.expectEqual(@as(u32, 1), f.calls.get(.begin));
}

test "a failed launch of a kept graph is returned: its work may have started, so it does not run again" {
    var f: Fake = .{ .fail_at = .launch };
    var l: TestLazy = .init(std.testing.allocator);
    defer l.deinit();
    var site: FakeSite = .{ .f = &f };
    const k: TestKey = .{ .rows = 4, .bucket = 1024 };
    try std.testing.expectError(error.CudaFailed, l.capture(&f, k, &site));
    try std.testing.expectEqual(@as(u32, 0), site.eager_runs);
    try std.testing.expectEqual(@as(u32, 0), site.undos);
    try std.testing.expect(l.get(k) != null); // kept and alive: the next use replays it
    try std.testing.expectEqual(@as(i32, 1), f.execs);
}
