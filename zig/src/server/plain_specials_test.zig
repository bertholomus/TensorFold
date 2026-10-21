//! TENSORFOLD_PLAIN_SPECIALS with a lane-vision engine (the dsv41 lane's images): a chat's image parts keep their
//! placeholder, the image token the engine expands, while an added token's text the user wrote beside them is plain
//! text. The shield used to quote the placeholders too, so every image request was refused with "the prompt holds
//! fewer image placeholders than images".
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// The lane's image placeholder (id 300) and another added token (id 301).
const image = "<|img|>";
const marker = "<|User|>";
const added = [_][]const u8{ image, marker };

/// Bytes as tokens and the two added tokens as theirs; renders each message as its role, a colon and its content.
const Text = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .encode_plain = plain, .added_text = addedText } };
    }

    fn addedId(s: []const u8) ?u32 {
        for (added, 0..) |m, n| if (std.mem.startsWith(u8, s, m)) return @intCast(300 + n);
        return null;
    }

    fn encode(_: *anyopaque, a: Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < s.len) {
            if (addedId(s[i..])) |id| {
                try out.append(a, id);
                i += added[id - 300].len;
            } else {
                try out.append(a, s[i]);
                i += 1;
            }
        }
        return out.items;
    }

    fn plain(_: *anyopaque, a: Allocator, s: []const u8) model_text.Error![]u32 {
        const out = try a.alloc(u32, s.len);
        for (s, out) |c, *o| o.* = c;
        return out;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| {
            if (id < 256) try out.append(a, @intCast(id)) else if (id >= 300 and id - 300 < added.len) try out.appendSlice(a, added[id - 300]);
        }
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, s: []const u8) ?u32 {
        for (added, 0..) |m, n| if (std.mem.eql(u8, s, m)) return @intCast(300 + n);
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        if (id >= 300 and id - 300 < added.len) return a.dupe(u8, added[id - 300]);
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 302;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        if (messages != .array) return out.items;
        for (messages.array) |m| {
            try out.print(a, "{s}:", .{m.strField("role") orelse ""});
            if (m.get("content")) |c| if (c == .string) try out.appendSlice(a, c.string);
            try out.append(a, '\n');
        }
        return out.items;
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }

    fn addedText(_: *anyopaque) []const []const u8 {
        return &added;
    }
};

/// A lane-vision engine (each image three rows) that keeps the prompt and the images the server submits, then
/// finishes at once.
const Capture = struct {
    prompt: []u32 = &.{},
    images: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        return .{ .lane_vision = .{ .ctx = ctx, .placeholder = image, .token = 300, .prepare = prepare } };
    }

    fn prepare(_: *anyopaque, a: Allocator, bytes: []const u8, _: *[]const u8) api.LaneVision.PrepareError!api.LaneVision.Prepared {
        return .{ .tokens = 3, .payload = try a.dupe(u8, bytes) };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        std.testing.allocator.free(e.prompt);
        e.prompt = std.testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
        e.images = request.lane_images.len;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// A whole reply's status.
const Reply = struct {
    status: ?u16 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status: u16, _: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status;
    }
};

/// The status of a chat whose one user turn has the JSON `content`, with TENSORFOLD_PLAIN_SPECIALS on or off;
/// `backend` keeps what was submitted.
fn chat(backend: *Capture, plain_specials: bool, content: []const u8) !?u16 {
    var text: Text = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
        .plain_specials = plain_specials,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{{\"role\":\"user\",\"content\":{s}}}]}}", .{content}))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var reply: Reply = .{};
    openai.run(srv, a, reply.out(), .{ .conn = &conn }, true, raw);
    return reply.status;
}

fn count(ids: []const u32, t: u32) usize {
    var n: usize = 0;
    for (ids) |x| n += @intFromBool(x == t);
    return n;
}

const with_image =
    \\[{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}, {"type": "text", "text": "what is <|User|>?"}]
;

test "with plain specials on a chat's image part keeps its placeholder, and a marker the user wrote is plain text" {
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    try std.testing.expectEqual(@as(?u16, 200), try chat(&backend, true, with_image));
    try std.testing.expectEqual(@as(usize, 1), backend.images);
    try std.testing.expectEqual(@as(usize, 3), count(backend.prompt, 300)); // the image's three rows
    try std.testing.expectEqual(@as(usize, 0), count(backend.prompt, 301)); // the user's marker, as its bytes
    // off: the same image, and the user's marker is its token
    try std.testing.expectEqual(@as(?u16, 200), try chat(&backend, false, with_image));
    try std.testing.expectEqual(@as(usize, 1), backend.images);
    try std.testing.expectEqual(@as(usize, 3), count(backend.prompt, 300));
    try std.testing.expectEqual(@as(usize, 1), count(backend.prompt, 301));
}

test "with plain specials on the placeholder a user writes in a chat without images is plain text" {
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    try std.testing.expectEqual(@as(?u16, 200), try chat(&backend, true, "\"what is <|img|>?\""));
    try std.testing.expectEqual(@as(usize, 0), backend.images);
    try std.testing.expectEqual(@as(usize, 0), count(backend.prompt, 300));
}
