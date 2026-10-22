//! ``cache_salt`` on the completion and chat routes: the engine gets the request's key in the salt's namespace and
//! the prompt as without it; a request without a salt is submitted as before; a salt that is not a non-empty string
//! gets a 400 and reaches no engine.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const cache_salt = @import("cache_salt.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// Bytes as tokens (completion_prompt_test.zig's); every chat renders as "prompt".
const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Copies the prompt and the key field of each request the server submits, then finishes it at once.
const Capture = struct {
    prompt: []u32 = &.{},
    key: []u32 = &.{},
    submitted: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn deinit(e: *@This()) void {
        std.testing.allocator.free(e.prompt);
        std.testing.allocator.free(e.key);
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        e.deinit();
        e.prompt = std.testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
        e.key = std.testing.allocator.dupe(u32, request.cache_key) catch return error.Busy;
        e.submitted += 1;
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

/// The status of a completion of the ids 0, 'h', 'i', 255 (``chat`` false) or a one-message chat, with the JSON
/// fields ``extra`` (each after a comma) in its body.
fn ask(backend: *Capture, chat: bool, extra: []const u8) !?u16 {
    var text: ByteText = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = if (chat)
        try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]{s}}}", .{extra})
    else
        try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":2,\"prompt\":[0,104,105,255]{s}}}", .{extra});
    const raw = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var reply: Reply = .{};
    openai.run(srv, a, reply.out(), .{ .conn = &conn }, chat, raw);
    return reply.status;
}

test "without a cache_salt (or with null) a request's key is the prompt, as before" {
    var backend: Capture = .{};
    defer backend.deinit();
    for ([_]bool{ false, true }) |chat| for ([_][]const u8{ "", ",\"cache_salt\":null" }) |extra| {
        try std.testing.expectEqual(@as(?u16, 200), try ask(&backend, chat, extra));
        try std.testing.expectEqual(@as(usize, 0), backend.key.len);
    };
    try std.testing.expectEqual(@as(usize, 4), backend.submitted);
}

test "a completion's cache_salt keys it in the salt's namespace; its prompt is as without one" {
    var backend: Capture = .{};
    defer backend.deinit();
    const prompt = [_]u32{ 0, 'h', 'i', 255 };
    try std.testing.expectEqual(@as(?u16, 200), try ask(&backend, false, ",\"cache_salt\":\"abc\""));
    const want = try cache_salt.key(std.testing.allocator, &prompt, "abc");
    defer std.testing.allocator.free(want);
    try std.testing.expectEqualSlices(u32, want, backend.key);
    try std.testing.expectEqualSlices(u32, &prompt, backend.prompt);
    try std.testing.expectEqual(@as(?u16, 200), try ask(&backend, false, ",\"cache_salt\":\"abd\""));
    try std.testing.expect(std.mem.indexOfDiff(u32, want, backend.key).? < cache_salt.words);
    try std.testing.expectEqualSlices(u32, &prompt, backend.prompt);
}

test "a chat's cache_salt keys it in the salt's namespace too" {
    var backend: Capture = .{};
    defer backend.deinit();
    try std.testing.expectEqual(@as(?u16, 200), try ask(&backend, true, ",\"cache_salt\":\"abc\""));
    const want = try cache_salt.key(std.testing.allocator, backend.prompt, "abc");
    defer std.testing.allocator.free(want);
    try std.testing.expect(backend.prompt.len > cache_salt.words);
    try std.testing.expectEqualSlices(u32, want, backend.key);
}

test "a cache_salt that is not a non-empty string gets a 400 and reaches no engine" {
    var backend: Capture = .{};
    defer backend.deinit();
    for ([_]bool{ false, true }) |chat| for ([_][]const u8{ ",\"cache_salt\":\"\"", ",\"cache_salt\":5", ",\"cache_salt\":true", ",\"cache_salt\":[\"a\"]", ",\"cache_salt\":{}" }) |extra| {
        try std.testing.expectEqual(@as(?u16, 400), try ask(&backend, chat, extra));
    };
    try std.testing.expectEqual(@as(usize, 0), backend.submitted);
}
