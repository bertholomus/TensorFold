//! Rank 0 to rank 1 over TCP (the fabric addresses): length-prefixed frames for each round's step and NCCL's id, as
//! multi.py's Link sends them. One connection, TCP_NODELAY, SO_BUSY_POLL, blocking reads (recvSpin polls first).
const std = @import("std");
const c = std.c;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        tune(fd);
        return fd;
    }

    /// TCP_NODELAY, and SO_BUSY_POLL: a read polls the NIC's receive queue itself (up to 50 us) instead of waiting for
    /// its interrupt, whose core may be leaving a deep idle state (node to node, 64-byte frames: round trip 1.0 ms
    /// blocking, 0.72 ms polling the socket, 11 us with SO_BUSY_POLL).
    fn tune(fd: c.fd_t) void {
        const one: c_int = 1;
        const busy_us: c_int = 50;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.BUSY_POLL, std.mem.asBytes(&busy_us), @sizeOf(c_int));
    }

    fn addr(ip: [4]u8, port: u16) c.sockaddr.in {
        return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
    }

    /// Rank 0: listen on `ip`:`port` and take the one peer's connection.
    pub fn listen(ip: [4]u8, port: u16) !Link {
        const fd = try socket();
        errdefer _ = c.close(fd);
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
        var a = addr(ip, port);
        if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
        if (c.listen(fd, 1) != 0) return error.ListenFailed;
        const peer = c.accept(fd, null, null);
        if (peer < 0) return error.AcceptFailed;
        _ = c.close(fd);
        tune(peer);
        return .{ .fd = peer };
    }

    /// Rank 1: connect to rank 0, retrying for up to `seconds` while it starts.
    pub fn connect(io: std.Io, ip: [4]u8, port: u16, seconds: u32) !Link {
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            var a = addr(ip, port);
            if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) == 0) return .{ .fd = fd };
            _ = c.close(fd);
            if (tries >= seconds * 4) return error.ConnectFailed;
            std.Io.sleep(io, .fromMilliseconds(250), .awake) catch {};
        }
    }

    pub fn close(l: *Link) void {
        _ = c.close(l.fd);
        l.* = undefined;
    }

    fn writeAll(l: Link, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.send(l.fd, bytes.ptr + done, bytes.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    fn readAll(l: Link, out: []u8) !void {
        var done: usize = 0;
        while (done < out.len) {
            const n = c.recv(l.fd, out.ptr + done, out.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    /// One frame: its length (u32, little-endian), then its bytes.
    pub fn send(l: Link, bytes: []const u8) !void {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(bytes.len), .little);
        try l.writeAll(&len);
        try l.writeAll(bytes);
    }

    /// recv, polling the socket without blocking for up to `spin_ns` first: a frame that comes within it is read
    /// without the receiving thread's sleep and wake-up (a core's deepest idle state takes 433 us to leave on GB10; the
    /// follower's frames come a fraction of a millisecond after its GPU goes idle), and after it the read blocks.
    pub fn recvSpin(l: Link, buf: []u8, spin_ns: u64) ![]u8 {
        var len: [4]u8 = undefined;
        var got: usize = 0;
        const t0 = monotonicNs();
        var polls: u32 = 0;
        while (got < len.len) {
            const n = c.recv(l.fd, @as([*]u8, &len) + got, len.len - got, c.MSG.DONTWAIT);
            if (n > 0) {
                got += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN => {},
                .INTR => continue,
                else => return error.PeerDown,
            }
            polls +%= 1;
            if (polls % 64 == 0 and monotonicNs() -% t0 > spin_ns) break;
            std.atomic.spinLoopHint();
        }
        if (got < len.len) try l.readAll(len[got..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    fn monotonicNs() u64 {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// The next frame into `buf` (a frame longer than `buf` is refused); its bytes.
    pub fn recv(l: Link, buf: []u8) ![]u8 {
        var len: [4]u8 = undefined;
        try l.readAll(&len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }
};

/// Linux poll's POLLRDHUP (std's POLL leaves it out): the peer shut its end. Frames waiting to be read do not set it.
const POLLRDHUP: i16 = 0x2000;

/// A thread that ends this process when a peer rank goes. The ranks run every step together, so a rank that stops (an
/// error, a crash, a kill) leaves the others waiting inside their next exchange for good, their GPUs spinning on it.
/// When a watched link's peer shuts its end and nothing stops the watch within `grace_ms` (a follower's own end after
/// rank 0's done comes first), the watch says which rank went and exits 1. Stop it before this process ends its
/// links itself.
pub const Watch = struct {
    fds: [max_links]c.fd_t = undefined,
    ranks: [max_links]u32 = undefined,
    n: usize = 0,
    rank: u32 = 0, // this process's rank
    grace_ms: u32 = 5000,
    stopped: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// A test's stand-in for the exit: told the rank that went.
    on_gone: ?*const fn (w: *Watch, rank: u32) void = null,

    pub const max_links = 4;

    /// Watches `l`, the link to rank `peer_rank`.
    pub fn add(w: *Watch, l: Link, peer_rank: u32) !void {
        if (w.n == max_links) return error.TooManyLinks;
        w.fds[w.n] = l.fd;
        w.ranks[w.n] = peer_rank;
        w.n += 1;
    }

    pub fn start(w: *Watch) !void {
        if (w.n == 0 or w.thread != null) return;
        w.stopped.store(false, .release);
        w.thread = try std.Thread.spawn(.{}, run, .{w});
    }

    /// Ends the watch (nothing when it never started).
    pub fn stop(w: *Watch) void {
        w.stopped.store(true, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
    }

    fn run(w: *Watch) void {
        var p: [max_links]c.pollfd = undefined;
        for (p[0..w.n], w.fds[0..w.n]) |*q, fd| q.* = .{ .fd = fd, .events = POLLRDHUP, .revents = 0 };
        while (!w.stopped.load(.acquire)) {
            const r = c.poll(&p, @intCast(w.n), 200);
            if (r == 0) continue;
            if (r < 0) { // (interrupted: look again; else wait a while rather than spin on the error)
                _ = c.poll(&p, 0, 200);
                continue;
            }
            const went = for (p[0..w.n], 0..) |q, i| {
                if (q.revents & c.POLL.NVAL != 0) return; // (a link closed under the watch: this process is ending it)
                if (q.revents & (POLLRDHUP | c.POLL.HUP | c.POLL.ERR) != 0) break i;
            } else continue;
            var waited: u32 = 0;
            while (waited < w.grace_ms) : (waited += 100) {
                if (w.stopped.load(.acquire)) return;
                _ = c.poll(&p, 0, 100);
            }
            if (w.stopped.load(.acquire)) return;
            if (w.on_gone) |f| return f(w, w.ranks[went]);
            std.debug.print("dsv41: rank {d} stopped (its link to rank {d} closed), so rank {d} stops too: the ranks run every step together, and it would wait for rank {d} for good (rank {d}'s own output says why it stopped)\n", .{ w.ranks[went], w.rank, w.rank, w.ranks[went], w.ranks[went] });
            c._exit(1);
        }
    }
};

/// "192.0.2.7" as four bytes.
pub fn parseIp(text: []const u8) ![4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    for (&out) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    if (it.next() != null) return error.BadAddress;
    return out;
}

test "addresses parse as four bytes" {
    try std.testing.expectEqual([4]u8{ 192, 0, 2, 7 }, try parseIp("192.0.2.7"));
    try std.testing.expectError(error.BadAddress, parseIp("192.0.2"));
}

test "frames cross a loopback link in order" {
    const io = std.testing.io;
    const port: u16 = 29000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 500)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            l.send("step 1") catch return;
            l.send(&.{}) catch return;
            l.send("polled") catch return;
            l.send("late") catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("step 1", try l.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try l.recv(&buf)).len);
    try std.testing.expectEqualStrings("polled", try l.recvSpin(&buf, 1000 * std.time.ns_per_ms));
    try std.testing.expectEqualStrings("late", try l.recvSpin(&buf, 0));
    t.join();
}

/// What a test's watch saw: the rank that went (0: none yet; the tests' peers are ranks 1 and up).
var seen_gone: std.atomic.Value(u32) = .init(0);

fn sawGone(_: *Watch, rank: u32) void {
    seen_gone.store(rank, .release);
}

/// Waits up to `ms` for a test's watch to see a rank go.
fn awaitGone(ms: u32) u32 {
    var p: [1]c.pollfd = undefined;
    var waited: u32 = 0;
    while (waited < ms) : (waited += 20) {
        const r = seen_gone.load(.acquire);
        if (r != 0) return r;
        _ = c.poll(&p, 0, 20);
    }
    return seen_gone.load(.acquire);
}

test "a watch names the rank whose link closed, and frames on the link are no close" {
    const io = std.testing.io;
    const port: u16 = 29500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            l.send("a frame") catch {};
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch {}; // the test's go: then this rank stops
            l.close();
        }
    };
    seen_gone.store(0, .release);
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var w: Watch = .{ .rank = 0, .grace_ms = 100, .on_gone = sawGone };
    try w.add(l, 3);
    try w.start();
    defer w.stop();
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("a frame", try l.recv(&buf));
    try std.testing.expectEqual(@as(u32, 0), awaitGone(400)); // a frame, read or not, is no close
    try l.send("go");
    t.join();
    try std.testing.expectEqual(@as(u32, 3), awaitGone(5000));
}

test "a stopped watch lets its links close quietly" {
    const io = std.testing.io;
    const port: u16 = 29900 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 90)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            var buf: [8]u8 = undefined;
            _ = l.recv(&buf) catch {};
            l.close();
        }
    };
    seen_gone.store(0, .release);
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var w: Watch = .{ .rank = 1, .grace_ms = 100, .on_gone = sawGone };
    try w.add(l, 2);
    try w.start();
    w.stop(); // (rank 0's done, or a follower reading it)
    try l.send("go");
    t.join();
    try std.testing.expectEqual(@as(u32, 0), awaitGone(400));
}

test "a watch's grace lets this process end first" {
    const io = std.testing.io;
    const port: u16 = 30000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 400)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            l.close();
        }
    };
    seen_gone.store(0, .release);
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var w: Watch = .{ .rank = 2, .grace_ms = 3000, .on_gone = sawGone };
    try w.add(l, 1);
    try w.start();
    t.join();
    try std.testing.expectEqual(@as(u32, 0), awaitGone(300)); // closed, but within the grace
    w.stop(); // this process ends its links itself in time
    try std.testing.expectEqual(@as(u32, 0), awaitGone(100));
}
