//! Pictures picture.zig does not decode itself (every format but the PNGs it reads: JPEG, WebP, GIF, BMP, TIFF, AVIF,
//! ICO, PPM, ...), decoded as the served lane decodes them: Pillow's Image.open(BytesIO(data)).convert("RGB") (the
//! DeepSeek reference's load_image), in a long-lived `python3 -I` child with the lane's pixel limit, one image at a
//! time. The pixels are Pillow's own, so they are the lane's when the runtime's Pillow is the lane's (the served
//! image's 12.3.0). Without python3 and Pillow the child is absent and such images are refused as Pillow refuses
//! data it cannot identify.
const std = @import("std");
const picture = @import("picture.zig");
const Allocator = std.mem.Allocator;

/// The child: a request is the image's length (u64 LE) and bytes; the answer a status byte (0: pixels, 1: Pillow's
/// DecompressionBombError, 2: another error), width, height and the length of what follows (u64 LE each), then the
/// RGB rows or the error's text. The lane refuses past 134,217,728 pixels: twice Image.MAX_IMAGE_PIXELS.
pub const script =
    \\import io, struct, sys, warnings
    \\from PIL import Image
    \\warnings.simplefilter("ignore")
    \\Image.MAX_IMAGE_PIXELS = 67108864
    \\inp, out = sys.stdin.buffer, sys.stdout.buffer
    \\def take(n):
    \\    b = inp.read(n)
    \\    if len(b) < n:
    \\        sys.exit(0)
    \\    return b
    \\out.write(b"PIL" + Image.__version__.encode().ljust(13))
    \\out.flush()
    \\while True:
    \\    data = take(struct.unpack("<Q", take(8))[0])
    \\    try:
    \\        with Image.open(io.BytesIO(data)) as source:
    \\            image = source.convert("RGB")
    \\        px = image.tobytes()
    \\        out.write(struct.pack("<BQQQ", 0, image.width, image.height, len(px)) + px)
    \\    except Image.DecompressionBombError as e:
    \\        m = str(e).encode()
    \\        out.write(struct.pack("<BQQQ", 1, 0, 0, len(m)) + m)
    \\    except Exception as e:
    \\        m = str(e).encode()
    \\        out.write(struct.pack("<BQQQ", 2, 0, 0, len(m)) + m)
    \\    out.flush()
;

pub const Pil = struct {
    io: std.Io,
    child: std.process.Child,
    mutex: std.Io.Mutex = .init,
    reader: std.Io.File.Reader,
    buf: [64 << 10]u8 = undefined,
    /// Pillow's version, as the child reports it on start.
    version: [13]u8 = undefined,

    /// The child started and its Pillow named; error when there is no python3 or no Pillow.
    pub fn start(gpa: Allocator, io: std.Io) !*Pil {
        const p = try gpa.create(Pil);
        errdefer gpa.destroy(p);
        p.* = .{ .io = io, .child = undefined, .reader = undefined };
        p.child = try std.process.spawn(io, .{ .argv = &.{ "python3", "-I", "-c", script }, .stdin = .pipe, .stdout = .pipe, .stderr = .inherit });
        errdefer p.child.kill(io);
        p.reader = p.child.stdout.?.readerStreaming(io, &p.buf);
        var hello: [16]u8 = undefined;
        p.reader.interface.readSliceAll(&hello) catch return error.NoPillow;
        if (!std.mem.eql(u8, hello[0..3], "PIL")) return error.NoPillow;
        @memcpy(&p.version, hello[3..16]);
        return p;
    }

    pub fn stop(p: *Pil, gpa: Allocator) void {
        if (p.child.stdin) |f| f.close(p.io);
        p.child.stdin = null;
        _ = p.child.wait(p.io) catch {};
        gpa.destroy(p);
    }

    pub fn versionText(p: *const Pil) []const u8 {
        return std.mem.trimEnd(u8, &p.version, " ");
    }

    pub const Decoded = union(enum) { rgb: picture.Rgb, refused: []const u8 };

    /// The image's RGB as Pillow's open(...).convert("RGB") gives it, or the lane's refusal of it (`a` holds either).
    pub fn decode(p: *Pil, a: Allocator, data: []const u8) !Decoded {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, data.len, .little);
        const in = p.child.stdin orelse return error.NoPillow;
        try in.writeStreamingAll(p.io, &len);
        try in.writeStreamingAll(p.io, data);
        var head: [25]u8 = undefined;
        try p.reader.interface.readSliceAll(&head);
        const w: usize = @intCast(std.mem.readInt(u64, head[1..9], .little));
        const h: usize = @intCast(std.mem.readInt(u64, head[9..17], .little));
        const n: usize = @intCast(std.mem.readInt(u64, head[17..25], .little));
        if (head[0] == 0 and n != w * h * 3) return error.BadPillowReply;
        const body = try a.alloc(u8, n);
        errdefer a.free(body);
        try p.reader.interface.readSliceAll(body);
        return switch (head[0]) {
            0 => .{ .rgb = .{ .w = w, .h = h, .px = body } },
            1 => .{ .refused = body },
            else => .{ .refused = try std.fmt.allocPrint(a, "image input: {s}", .{body}) },
        };
    }
};
