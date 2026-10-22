//! cache_salt on /v1/completions: a fresh salt reports no cached tokens, its repeat and the unsalted repeat do.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// Bytes as tokens; decode skips ids past 255, as the checkpoint tokenizer skips unknown ids.
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

/// Restores the longest prefix of any earlier request's cache key, as a prompt cache keyed on it does.
const Prefixes = struct {
    keys: std.ArrayList([]u32) = .empty,

    fn deinit(e: *@This()) void {
        for (e.keys.items) |k| std.testing.allocator.free(k);
        e.keys.deinit(std.testing.allocator);
    }

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .prompt_cache = true };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Prefixes = @ptrCast(@alignCast(ctx));
        const k = request.cacheKey();
        var best: usize = 0;
        for (e.keys.items) |old| best = @max(best, std.mem.indexOfDiff(u32, old, k) orelse k.len);
        e.keys.append(std.testing.allocator, std.testing.allocator.dupe(u32, k) catch return error.Busy) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = @intCast(@min(best, k.len - 1)) });
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

/// A whole reply's status and usage.prompt_tokens_details.cached_tokens.
const Reply = struct {
    status: ?u16 = null,
    cached: ?i64 = null,

    fn out(r: *Reply) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status: u16, body: json.Value) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.status = status;
        const usage = body.get("usage") orelse return;
        const details = usage.get("prompt_tokens_details") orelse return;
        r.cached = (details.get("cached_tokens") orelse return).int64();
    }
};

/// One completion of a fixed prompt on `srv`, with `salt` as its JSON ``cache_salt`` (null: none).
fn complete(srv: *Server, salt: ?[]const u8) !Reply {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const extra = if (salt) |s| try std.fmt.allocPrint(a, ",\"cache_salt\":{s}", .{s}) else "";
    const raw = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":2,\"prompt\":\"a prompt the cache keeps\"{s}}}", .{extra}))).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var reply: Reply = .{};
    openai.run(srv, a, reply.out(), .{ .conn = &conn }, false, raw);
    return reply;
}

test "cache_salt: cached_tokens is 0 for a fresh salt, and the same salt or no salt resumes its own" {
    var backend: Prefixes = .{};
    defer backend.deinit();
    var text: ByteText = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const none = try complete(srv, null);
    try std.testing.expectEqual(@as(?u16, 200), none.status);
    try std.testing.expectEqual(@as(?i64, 0), none.cached);
    // a fresh salt: nothing cached, though the same prompt ran unsalted
    try std.testing.expectEqual(@as(?i64, 0), (try complete(srv, "\"r1\"")).cached);
    // the same salt again, and no salt again: their own states
    try std.testing.expect((try complete(srv, "\"r1\"")).cached.? > 0);
    try std.testing.expect((try complete(srv, null)).cached.? > 0);
    // another fresh salt
    try std.testing.expectEqual(@as(?i64, 0), (try complete(srv, "\"r2\"")).cached);
}

test "cache_salt must be a non-empty string" {
    var backend: Prefixes = .{};
    defer backend.deinit();
    var text: ByteText = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    for ([_][]const u8{ "\"\"", "7", "true", "[\"x\"]" }) |bad| try std.testing.expectEqual(@as(?u16, 400), (try complete(srv, bad)).status);
    try std.testing.expectEqual(@as(?u16, 200), (try complete(srv, "null")).status);
}
