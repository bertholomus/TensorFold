//! Prompt ids from a request, as ``prepare_prompt`` and ``render_prompt_ids`` make them.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const images = @import("images.zig");
const image_key = @import("image_key.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;

/// With image input, `ids` holds each image's tokens, `images` the images and `cache_key` the prompt the cache matches;
/// a lane-vision engine's images are `lane_images`, spans of its image token.
pub const Rendered = struct { ids: []const u32, history_len: usize = 0, rewind_len: usize = 0, images: []const api.Image = &.{}, cache_key: []const u32 = &.{}, lane_images: []const api.LaneImage = &.{} };

/// Where lessons decide to act on a reply: the row before the generation prompt's `<think>`, else before its last.
pub fn decideOf(srv: *const Server, ids: []const u32) u32 {
    if (srv.text.tokenId("<think>")) |open| {
        var i = ids.len;
        while (i > ids.len -| 8) {
            i -= 1;
            if (ids[i] == open and i > 0) return @intCast(i - 1);
        }
    }
    return @intCast(ids.len -| 2);
}

pub const isTitle = messages_mod.isTitleRequest;

/// ``render_prompt_ids``: messages normalized for this template, rendered, and a dangling ``<think>`` closed.
pub fn renderIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool) errors.Refused![]const u32 {
    return renderIdsOf(srv, cx, messages, tools, thinking, effort, generation, false);
}

/// renderIds; `lane_images`: the messages hold the request's image parts as the lane's placeholder (normalizeImages),
/// which TENSORFOLD_PLAIN_SPECIALS keeps as the template's.
fn renderIdsOf(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool, lane_images: bool) errors.Refused![]const u32 {
    const normalized = try messages_mod.toolArguments(cx, try messages_mod.normalizeWith(cx, messages, srv.late_system, srv.needs_user_after_tool, .of(srv.info.vision)));
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
        .reasoning_effort_value = if (thinking) srv.effortValue(effort) else null,
    };
    var ids = render(srv, cx.a, normalized, options, lane_images, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "{s}", .{problem}),
    };
    if (!thinking and generation and ids.len > 0) {
        const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
        if (std.mem.eql(u8, @import("reply_text.zig").pyStrip(last), "<think>")) {
            const close = srv.text.encode(cx.a, "</think>", false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => &[_]u32{},
            };
            if (close.len == 1) ids = try std.mem.concat(cx.a, u32, &.{ ids, close });
        }
    }
    return ids;
}

/// The template's ids for `messages`; with image input, a marker the text spells is ordinary text (images.zig), and
/// with TENSORFOLD_PLAIN_SPECIALS an added token's text a sender wrote (specials.zig; `lane_images`: the lane's
/// placeholders are the request's image parts, kept).
fn render(srv: *Server, a: std.mem.Allocator, messages: Value, options: model_text.RenderOptions, lane_images: bool, problem: *[]const u8) model_text.Error![]u32 {
    if (srv.shield) |sh| if (try sh.apply(a, messages, lane_images)) |quoted| {
        const text = try srv.text.render(a, quoted, options, problem);
        return images.encodeShielded(srv.text, a, text, sh.markers);
    };
    const v = srv.info.vision orelse return srv.text.renderIds(a, messages, options, problem);
    const markers = try images.markerText(srv.text, a, v);
    const text = try srv.text.render(a, try images.shield(a, messages, markers), options, problem);
    return images.encodeShielded(srv.text, a, text, markers);
}

/// The template's generation prompt (thinking on or off): what a probe conversation renders past its history.
fn generationPrompt(srv: *Server, cx: *Cx, thinking: bool) errors.Refused![]const u32 {
    const user = try json.newObject(cx.a);
    try user.put(cx.a, "role", .{ .string = "user" });
    try user.put(cx.a, "content", .{ .string = "x" });
    const list = try cx.a.alloc(Value, 1);
    list[0] = .{ .object = user };
    const with = try renderIds(srv, cx, .{ .array = list }, &.{}, thinking, null, true);
    const without = try renderIds(srv, cx, .{ .array = list }, &.{}, thinking, null, false);
    if (without.len >= with.len or !std.mem.eql(u32, with[0..without.len], without)) return &.{};
    return with[without.len..];
}

/// Earlier replies' decoded rows: from each turn's generation prompt end through its closing stop token.
pub fn replySpans(srv: *Server, cx: *Cx, prompt: []const u32) errors.Refused![]const [2]u32 {
    if (srv.eos.len == 0) return &.{};
    var probe: Cx = .{ .a = cx.a }; // a template that refuses the probe leaves the request without spans, not failed
    const openers = [2][]const u32{ generationPrompt(srv, &probe, false) catch &.{}, generationPrompt(srv, &probe, true) catch &.{} };
    var spans: std.ArrayList([2]u32) = .empty;
    var i: usize = 0;
    while (i < prompt.len) {
        var start: ?usize = null;
        var at_min: usize = prompt.len;
        for (openers) |o| if (o.len > 0) if (std.mem.indexOfPos(u32, prompt, i, o)) |at| if (at < at_min) {
            at_min = at;
            start = at + o.len;
        };
        const first = start orelse break;
        const end = std.mem.indexOfAnyPos(u32, prompt, first, srv.eos) orelse break;
        try spans.append(cx.a, .{ @intCast(first), @intCast(end + 1) });
        i = end + 1;
    }
    return spans.items;
}

/// A request's prompt: raw text or ids for a completion, else the chat template's, with its history length.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    if (input.prompt) |p| switch (p) {
        .text => |t| return .{ .ids = srv.text.encode(cx.a, t, true) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
        } },
        .ids => |ids| return .{ .ids = ids },
    };
    const prompt = try renderIdsOf(srv, cx, input.messages, input.tools, thinking, effort, true, input.lane_images.len > 0);
    const history = try renderIdsOf(srv, cx, input.messages, input.tools, thinking, effort, false, input.lane_images.len > 0);
    var history_len: usize = 0;
    if (history.len > 0 and history.len < prompt.len and std.mem.eql(u32, prompt[0..history.len], history)) history_len = history.len;
    if (history_len == 0 and prompt.len > 1 and std.mem.eql(u32, history, prompt)) history_len = prompt.len - 1; // a template with no generation suffix
    if (input.lane_images.len > 0) return laneSpans(srv, cx, prompt, history_len, input.lane_images); // no rewind kept, as --vision
    // --vision: placeholders expanded to image tokens before the window, chunks and cache see it; no rewind kept
    if (try images.expand(srv.info.vision, if (srv.config.vision_urls) images.Fetch.network(srv.io) else null, srv.text, cx, input.messages, prompt, history_len)) |x| return .{ .ids = x.ids, .history_len = x.history_len, .images = x.images, .cache_key = x.cache_key };
    const count = if (input.messages == .array) input.messages.array.len else 0;
    const rewind_len = if (srv.info.prompt_cache) rewindLen(srv, cx, input.messages, count, input.tools, prompt, thinking, effort) else 0; // one more render
    return .{ .ids = prompt, .history_len = history_len, .rewind_len = rewind_len };
}

/// Where the latest user message's text starts (of the first `real_count`): the prompt and a probe there agree to it.
pub fn rewindLen(srv: *Server, cx: *Cx, messages: Value, real_count: usize, tools: []const Value, prompt: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array or real_count > messages.array.len) return 0;
    const user_index = for (0..real_count) |n| {
        const i = real_count - 1 - n;
        const role = messages.array[i].strField("role") orelse continue;
        if (std.mem.eql(u8, role, "user")) break i;
    } else return 0;
    if (messages.array[user_index] != .object) return 0;
    const list = cx.a.dupe(Value, messages.array) catch return 0;
    const user = json.copyObject(cx.a, messages.array[user_index].object) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    list[user_index] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const probe = renderIds(srv, &scratch, .{ .array = list }, tools, thinking, effort, true) catch return 0;
    const i = std.mem.indexOfDiff(u32, prompt, probe) orelse return 0;
    return if (i < prompt.len) i else 0;
}

/// Each image placeholder of the rendered prompt as its image's span (the lane-vision family's prepare: the span's
/// length and the payload the engine reads), the images in prompt order.
fn laneSpans(srv: *Server, cx: *Cx, ids: []const u32, history_len: usize, lane_images: []const []const u8) errors.Refused!Rendered {
    const vision = srv.info.lane_vision orelse return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
    return placeSpans(cx, vision, ids, history_len, lane_images);
}

/// laneSpans with the engine's vision: the cache key spells each span by its payload's digest, and the history
/// (`history_len` ids) ends past its images' spans.
fn placeSpans(cx: *Cx, vision: api.LaneVision, ids: []const u32, history_len: usize, lane_images: []const []const u8) errors.Refused!Rendered {
    var out: std.ArrayList(u32) = .empty;
    var key: std.ArrayList(u32) = .empty;
    var placed: std.ArrayList(api.LaneImage) = .empty;
    var history_end: usize = 0;
    for (ids, 0..) |id, i| {
        if (i == history_len) history_end = out.items.len;
        if (id != vision.token) {
            try out.append(cx.a, id);
            try key.append(cx.a, id);
            continue;
        }
        const k = placed.items.len;
        if (k >= lane_images.len) return cx.refuse("image input: the prompt holds more image placeholders than images");
        var problem: []const u8 = "";
        const prep = vision.prepare(vision.ctx, cx.a, lane_images[k], &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.BadImage => return cx.refuse(problem),
        };
        try placed.append(cx.a, .{ .at = @intCast(out.items.len), .tokens = prep.tokens, .bytes = prep.payload });
        try out.appendNTimes(cx.a, vision.token, prep.tokens);
        try spellRows(cx.a, &key, prep.payload, prep.tokens);
    }
    if (placed.items.len != lane_images.len) return cx.refuse("image input: the prompt holds fewer image placeholders than images");
    return .{ .ids = out.items, .lane_images = placed.items, .cache_key = key.items, .history_len = if (history_len > 0) history_end else 0 };
}

/// A lane image's span as the cache key spells it, in image_key.zig's words: its first two rows carry 62 bits of its
/// payload's Blake3 digest whole (the payload is what the engine reads: one payload, one span of rows), the others a
/// hash of those bits and the row. (Blake3: a few milliseconds for the largest payload, and no crafted collisions.)
fn spellRows(a: std.mem.Allocator, key: *std.ArrayList(u32), payload: []const u8, tokens: u32) !void {
    var d: [32]u8 = undefined;
    std.crypto.hash.Blake3.hash(payload, &d, .{});
    const h = std.mem.readInt(u64, d[0..8], .little) & ((1 << 62) - 1);
    for (0..tokens) |j| try key.append(a, switch (j) {
        0 => image_key.open(h),
        1 => image_key.close(h),
        else => image_key.row(h, j),
    });
}

test "a lane image's key words name its payload and their rows, past every vocabulary's ids" {
    const a = std.testing.allocator;
    var one: std.ArrayList(u32) = .empty;
    defer one.deinit(a);
    var same: std.ArrayList(u32) = .empty;
    defer same.deinit(a);
    var other: std.ArrayList(u32) = .empty;
    defer other.deinit(a);
    try spellRows(a, &one, "the same pixels", 64);
    try spellRows(a, &same, "the same pixels", 64);
    try spellRows(a, &other, "other pixels", 64);
    try std.testing.expectEqual(@as(usize, 64), one.items.len);
    try std.testing.expectEqualSlices(u32, one.items, same.items);
    try std.testing.expect(one.items[0] != other.items[0] or one.items[1] != other.items[1]); // the digest's 62 bits whole
    try std.testing.expect(!std.mem.eql(u32, one.items[2..], other.items[2..]));
    for (one.items) |id| try std.testing.expect(id >= 0x8000_0000);
    try std.testing.expect(one.items[2] != one.items[3]);
}

test "lane images: each placeholder its span, the key one word a position, the history past its images" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Three = struct { // each image three rows, its payload its bytes
        fn prepare(_: *anyopaque, al: std.mem.Allocator, bytes: []const u8, _: *[]const u8) api.LaneVision.PrepareError!api.LaneVision.Prepared {
            return .{ .tokens = 3, .payload = try al.dupe(u8, bytes) };
        }
    };
    var dummy: u8 = 0;
    const v: api.LaneVision = .{ .ctx = &dummy, .placeholder = "<image>", .token = 7, .prepare = Three.prepare };
    var cx: Cx = .{ .a = a };
    // a history [1, 2, image A, 4] then this turn's [image B, 5]
    const ids = [_]u32{ 1, 2, 7, 4, 7, 5 };
    const x = try placeSpans(&cx, v, &ids, 4, &.{ "A", "B" });
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 7, 7, 7, 4, 7, 7, 7, 5 }, x.ids);
    try std.testing.expectEqual(@as(usize, 6), x.history_len);
    try std.testing.expectEqual(@as(usize, 2), x.lane_images.len);
    try std.testing.expectEqual(@as(u32, 2), x.lane_images[0].at);
    try std.testing.expectEqual(@as(u32, 6), x.lane_images[1].at);
    try std.testing.expectEqual(x.ids.len, x.cache_key.len);
    var rows_a: std.ArrayList(u32) = .empty;
    try spellRows(a, &rows_a, "A", 3);
    var rows_b: std.ArrayList(u32) = .empty;
    try spellRows(a, &rows_b, "B", 3);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, x.cache_key[0..2]);
    try std.testing.expectEqualSlices(u32, rows_a.items, x.cache_key[2..5]);
    try std.testing.expectEqual(@as(u32, 4), x.cache_key[5]);
    try std.testing.expectEqualSlices(u32, rows_b.items, x.cache_key[6..9]);
    try std.testing.expectEqual(@as(u32, 5), x.cache_key[9]);
    // another first image: the same ids, a key that parts at the image
    const y = try placeSpans(&cx, v, &ids, 4, &.{ "C", "B" });
    try std.testing.expectEqualSlices(u32, x.ids, y.ids);
    try std.testing.expectEqual(@as(?usize, 2), std.mem.indexOfDiff(u32, x.cache_key, y.cache_key));
    // no history: none kept past the images either
    try std.testing.expectEqual(@as(usize, 0), (try placeSpans(&cx, v, &ids, 0, &.{ "A", "B" })).history_len);
}

/// A reusable system prefix, found with a probe in place of the first user message; zero below 512 tokens.
pub fn systemPrefixLen(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array) return 0;
    const first_user = for (messages.array, 0..) |m, i| {
        const role = m.get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break i;
    } else return 0;
    const probe = cx.a.alloc(Value, first_user + 1) catch return 0;
    @memcpy(probe[0..first_user], messages.array[0..first_user]);
    const user = json.newObject(cx.a) catch return 0;
    user.put(cx.a, "role", .{ .string = "user" }) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    probe[first_user] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const other = renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, effort, true) catch return 0;
    var shared: usize = 0;
    while (shared < @min(prompt_ids.len, other.len) and prompt_ids[shared] == other[shared]) shared += 1;
    return if (shared >= 512) shared else 0;
}
