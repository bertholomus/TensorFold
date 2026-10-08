//! Rank 0 to rank 1 over TCP (the fabric addresses): length-prefixed frames for each round's step and NCCL's id, as
//! multi.py's Link sends them. One connection, TCP_NODELAY, blocking reads.
const std = @import("std");
const c = std.c;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        return fd;
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
        _ = c.setsockopt(peer, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
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
