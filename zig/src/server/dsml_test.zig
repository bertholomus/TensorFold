//! The DSML call reader's tests (dsml.zig), the one set both the four-node and the two-node lines run: offered and
//! unoffered calls in either spelling, both kinds in one reply, call markup in the prose or in an argument, V4's strict
//! reading, and a streamed reply's calls equal to the whole reply's, through the streamer and through the server.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const dsml = @import("dsml.zig");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const tool_parse = @import("tool_parse.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Streamer = @import("tool_stream.zig").Streamer;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const parse = tool_parse.parse;

const D = "\u{ff5c}DSML\u{ff5c}";
const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"weather\"}},{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}},{\"type\":\"function\",\"function\":{\"name\":\"write\"}}]";
const spellings = [_]dsml.Tags{ dsml.v4, dsml.v41 };

fn isV41(tags: dsml.Tags) bool {
    return std.mem.eql(u8, tags.open, dsml.v41.open);
}

/// An invoke of ``name`` in ``tags``' spelling with string parameters (key, value).
fn invoke(a: Allocator, tags: dsml.Tags, name: []const u8, params: []const [2][]const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, tags.invoke);
    try out.appendSlice(a, name);
    try out.appendSlice(a, "\">\n");
    for (params) |p| {
        try out.appendSlice(a, tags.param);
        try out.appendSlice(a, p[0]);
        try out.appendSlice(a, "\" string=\"true\">");
        try out.appendSlice(a, p[1]);
        try out.appendSlice(a, tags.param_close);
        try out.append(a, '\n');
    }
    try out.appendSlice(a, tags.invoke_close);
    try out.append(a, '\n');
    return out.items;
}

/// A block in ``tags``' spelling around ``invokes``.
fn block(a: Allocator, tags: dsml.Tags, invokes: []const []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, tags.open);
    try out.append(a, '\n');
    for (invokes) |i| try out.appendSlice(a, i);
    try out.appendSlice(a, tags.close);
    return out.items;
}

/// A call as a client reads it: its name and its arguments' JSON text.
const Named = struct { name: []const u8, arguments: []const u8 };

fn named(a: Allocator, calls: ?[]const Value) Allocator.Error![]Named {
    const list: []const Value = calls orelse &.{};
    const out = try a.alloc(Named, list.len);
    for (list, out) |c, *o| {
        const f = c.get("function").?;
        o.* = .{ .name = f.strField("name").?, .arguments = f.strField("arguments").? };
    }
    return out;
}

fn expectCalls(want: []const Named, got: []const Named) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| {
        try std.testing.expectEqualStrings(w.name, g.name);
        try std.testing.expectEqualStrings(w.arguments, g.arguments);
    }
}

fn expectNames(want: []const []const u8, got: []const Named) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g.name);
}

test "an offered call, in either spelling, is a call under the name the request spelled" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, tools_json)).ok.array;
    for (spellings) |tags| {
        const reply = try std.mem.concat(a, u8, &.{ "Checking.\n\n", try block(a, tags, &.{try invoke(a, tags, "WEATHER", &.{.{ "city", "Oslo" }})}) });
        for ([_]?usize{ null, 1 }) |quota| {
            const r = try parse(a, reply, tools, quota);
            try expectCalls(&.{.{ .name = "weather", .arguments = "{\"city\":\"Oslo\"}" }}, try named(a, r.calls));
            try std.testing.expectEqualStrings("Checking.", r.content);
        }
    }
}

test "a call to a tool the request did not offer is a call under the model's own name; a one-call reply drops it, as before" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, tools_json)).ok.array;
    for (spellings) |tags| {
        const reply = try std.mem.concat(a, u8, &.{ "Launching.\n\n", try block(a, tags, &.{try invoke(a, tags, " launch ", &.{.{ "when", "now" }})}) });
        const r = try parse(a, reply, tools, null);
        try expectCalls(&.{.{ .name = "launch", .arguments = "{\"when\":\"now\"}" }}, try named(a, r.calls));
        try std.testing.expectEqualStrings("Launching.", r.content);
        const single = try parse(a, reply, tools, 1);
        try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings("Launching.", single.content);
    }
}

test "unoffered and offered calls in one reply keep the reply's order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, tools_json)).ok.array;
    for (spellings) |tags| {
        const reply = try block(a, tags, &.{ try invoke(a, tags, "launch", &.{.{ "when", "now" }}), try invoke(a, tags, "weather", &.{.{ "city", "Oslo" }}) });
        const r = try parse(a, reply, tools, null);
        try expectCalls(&.{ .{ .name = "launch", .arguments = "{\"when\":\"now\"}" }, .{ .name = "weather", .arguments = "{\"city\":\"Oslo\"}" } }, try named(a, r.calls));
        try std.testing.expectEqualStrings("", r.content);
        // one call a reply, as before: V4.1's invokes count one by one, V4's block whole (it names a tool not offered)
        const single = try parse(a, reply, tools, 1);
        if (isV41(tags)) try expectNames(&.{"weather"}, try named(a, single.calls)) else try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings("", single.content);
    }
    // across blocks and forms: a <tool_call> call, a V4.1 block, a V4 block
    const mixed = try std.mem.concat(a, u8, &.{
        "<tool_call>{\"name\": \"lookup\", \"arguments\": {\"q\": \"x\"}}</tool_call>\n",
        try block(a, dsml.v41, &.{try invoke(a, dsml.v41, "launch", &.{})}),
        "\n",
        try block(a, dsml.v4, &.{try invoke(a, dsml.v4, "weather", &.{})}),
    });
    try expectNames(&.{ "lookup", "launch", "weather" }, try named(a, (try parse(a, mixed, tools, null)).calls));
}

test "call markup in the prose or in an argument moves no V4.1 call, offered or not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, tools_json)).ok.array;
    const quoted = "a </tool_call> and <tool_call>{\"name\": \"weather\", \"arguments\": {}}</tool_call>";
    const b = try block(a, dsml.v41, &.{ try invoke(a, dsml.v41, "write", &.{.{ "content", quoted }}), try invoke(a, dsml.v41, "notes", &.{.{ "text", quoted }}) });
    // a <tool_call> in the prose before the block, and a </tool_call> and a whole call in its arguments
    const reply = try std.mem.concat(a, u8, &.{ "I will not use <tool_call> here.\n\n", b });
    for ([_]?usize{ null, 1 }) |quota| {
        const r = try parse(a, reply, tools, quota);
        try std.testing.expectEqualStrings("I will not use <tool_call> here.", r.content);
        const calls = try named(a, r.calls);
        try expectNames(if (quota == null) &.{ "write", "notes" } else &.{"write"}, calls);
        const keys = [_][]const u8{ "content", "text" };
        for (calls, keys[0..calls.len]) |c, key| {
            try std.testing.expectEqualStrings(quoted, (try json.parse(a, c.arguments)).ok.strField(key).?);
        }
    }
    try std.testing.expectEqualStrings("", try tool_parse.closeCall(a, reply, tools));
    // calls keep the reply's order: a <tool_call> call before the block comes first
    const both = try std.mem.concat(a, u8, &.{ "<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>\n", b });
    try expectNames(&.{ "weather", "write", "notes" }, try named(a, (try parse(a, both, tools, null)).calls));
}

test "DeepSeek-V4.1's DSML calls, read as the served lane reads them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"weather\"}},{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    // two invokes, string and JSON values, prose before the block
    const two = "Checking.\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"weather\">\n<" ++ D ++ " parameter name=\"city\" string=\"true\">Oslo</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"days\" string=\"false\">3</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n<" ++ D ++ " invoke name=\"LOOKUP\">\n<" ++ D ++ " parameter name=\"q\" string=\"false\">{\"a\": [1, 2.5, null]}</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const r = try parse(a, two, tools, null);
    try std.testing.expectEqualStrings("Checking.", r.content);
    try expectCalls(&.{ .{ .name = "weather", .arguments = "{\"city\":\"Oslo\",\"days\":3}" }, .{ .name = "lookup", .arguments = "{\"q\":{\"a\":[1,2.5,null]}}" } }, try named(a, r.calls));
    // one call a reply: the second invoke drops out
    try std.testing.expectEqual(@as(usize, 1), (try parse(a, two, tools, 1)).calls.?.len);
    // JSON that does not parse is the string; no string attribute is a string
    const loose = "<" ++ D ++ " calls><" ++ D ++ " invoke name=\"weather\"><" ++ D ++ " parameter name=\"x\" string=\"false\">not json</" ++ D ++ " parameter><" ++ D ++ " parameter name=\"y\">7</" ++ D ++ " parameter></" ++ D ++ " invoke></" ++ D ++ " calls>";
    try std.testing.expectEqualStrings("{\"x\":\"not json\",\"y\":\"7\"}", (try parse(a, loose, tools, null)).calls.?[0].get("function").?.get("arguments").?.string);
    // a tool the request did not offer: a call under the model's name, no call markup left in the text
    const other = "<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"launch\">\n<" ++ D ++ " parameter name=\"when\" string=\"true\">now</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const made = try parse(a, other, tools, null);
    try expectCalls(&.{.{ .name = "launch", .arguments = "{\"when\":\"now\"}" }}, try named(a, made.calls));
    try std.testing.expectEqualStrings("", made.content);
    try std.testing.expectEqualStrings("", (try parse(a, other, tools, 1)).content);
    // a block with no invoke goes; an unclosed block stays text
    try std.testing.expectEqualStrings("Hi.", (try parse(a, "Hi.<" ++ D ++ " calls>\n</" ++ D ++ " calls>", tools, null)).content);
    const open = "Hi.<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"weather\">";
    try std.testing.expectEqualStrings(open, (try parse(a, open, tools, null)).content);
    // a name is matched stripped, as the lane's parse of its rewritten call strips it; a blank one is no call
    const padded = "<" ++ D ++ " calls><" ++ D ++ " invoke name=\" weather \"><" ++ D ++ " parameter name=\"city\" string=\"true\">Oslo</" ++ D ++ " parameter></" ++ D ++ " invoke><" ++ D ++ " invoke name=\" \"></" ++ D ++ " invoke></" ++ D ++ " calls>";
    const one = try parse(a, padded, tools, null);
    try expectNames(&.{"weather"}, try named(a, one.calls));
    try std.testing.expectEqualStrings("", one.content);
}

test "a V4 block that is not well formed stays the reply's text, whatever it names" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, tools_json)).ok.array;
    const v4 = dsml.v4;
    for ([_][]const u8{ "weather", "launch" }) |name| {
        const bad = [_][]const u8{
            // text between invokes
            try block(a, v4, &.{ try invoke(a, v4, name, &.{.{ "q", "x" }}), "stray\n", try invoke(a, v4, "lookup", &.{}) }),
            // a parameter without its string attribute
            try std.mem.concat(a, u8, &.{ v4.open, v4.invoke, name, "\">", v4.param, "q\">x", v4.param_close, v4.invoke_close, v4.close }),
            // a blank name beside a good invoke
            try block(a, v4, &.{ try invoke(a, v4, name, &.{}), try invoke(a, v4, " ", &.{}) }),
            // no invoke
            try block(a, v4, &.{"just text\n"}),
        };
        for (bad) |reply| {
            const r = try parse(a, reply, tools, null);
            try std.testing.expect(r.calls == null);
            try std.testing.expectEqualStrings(reply, r.content);
            const single = try parse(a, reply, tools, 1);
            try std.testing.expect(single.calls == null);
            try std.testing.expectEqualStrings("", single.content);
        }
    }
}

/// Streamed call deltas joined as a client joins them: a call an index, its name from its first delta, its arguments in
/// order.
fn joined(a: Allocator, deltas: []const Value) ![]Named {
    var names: std.ArrayList([]const u8) = .empty;
    var args: std.ArrayList(std.ArrayList(u8)) = .empty;
    for (deltas) |d| for (d.get("tool_calls").?.array) |one| {
        const index: usize = @intCast(one.get("index").?.int64().?);
        while (names.items.len <= index) {
            try names.append(a, "");
            try args.append(a, .empty);
        }
        const f = one.get("function").?;
        if (one.get("id") != null) names.items[index] = f.strField("name").?;
        if (f.strField("arguments")) |t| try args.items[index].appendSlice(a, t);
    };
    const out = try a.alloc(Named, names.items.len);
    for (out, names.items, args.items) |*o, n, x| o.* = .{ .name = n, .arguments = x.items };
    return out;
}

/// The calls a client gets from a stream of ``text`` (openai.zig): what the streamer sent, once it sent a call, else the
/// whole reply's, which the stream's end sends. Null ``cut``: the streamer reads the text at each character's end, as the
/// server feeds it; else at ``cut`` and then whole.
fn streamCalls(a: Allocator, tools: []const Value, text: []const u8, cut: ?usize) ![]Named {
    var s = try Streamer.init(a, tools);
    var out: std.ArrayList(Value) = .empty;
    if (cut) |n| {
        try s.feed(text[0..n], &out);
        try s.feed(text, &out);
    } else {
        var i: usize = 0;
        while (i < text.len) {
            i = @min(text.len, i + @as(usize, std.unicode.utf8ByteSequenceLength(text[i]) catch 1));
            try s.feed(text[0..i], &out);
        }
    }
    if (!s.streamed) return named(a, (try parse(a, text, tools, null)).calls);
    return joined(a, out.items);
}

/// A reply and what its calls are named.
const Case = struct { text: []const u8, names: []const []const u8 };

/// Replies of DSML calls, for the streamed tests.
fn replies(a: Allocator) ![]const Case {
    const v4 = dsml.v4;
    const v41 = dsml.v41;
    const quoted = "a </tool_call> and <tool_call>{\"name\": \"weather\", \"arguments\": {}}</tool_call>";
    const xml = "<tool_call>\n<function=lookup>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>\n";
    const launch41 = try invoke(a, v41, "launch", &.{.{ "when", "now" }});
    const weather41 = try invoke(a, v41, "weather", &.{.{ "city", "Oslo" }});
    const launch4 = try invoke(a, v4, "launch", &.{.{ "when", "now" }});
    const weather4 = try invoke(a, v4, "weather", &.{.{ "city", "Oslo" }});
    const list = [_]Case{
        .{ .text = try std.mem.concat(a, u8, &.{ "Checking.\n\n", try block(a, v41, &.{weather41}) }), .names = &.{"weather"} },
        .{ .text = try block(a, v41, &.{launch41}), .names = &.{"launch"} },
        .{ .text = try block(a, v41, &.{ launch41, weather41 }), .names = &.{ "launch", "weather" } },
        // a call streams before the block: the block's calls stream once it closes
        .{ .text = try std.mem.concat(a, u8, &.{ xml, try block(a, v41, &.{ launch41, weather41 }) }), .names = &.{ "lookup", "launch", "weather" } },
        .{ .text = try std.mem.concat(a, u8, &.{ "I will not use <tool_call> here.\n\n", try block(a, v41, &.{ try invoke(a, v41, "write", &.{.{ "content", quoted }}), try invoke(a, v41, "notes", &.{.{ "text", quoted }}) }) }), .names = &.{ "write", "notes" } },
        // a block before a call: the reply's calls go at its end
        .{ .text = try std.mem.concat(a, u8, &.{ try block(a, v41, &.{launch41}), "\n", xml }), .names = &.{ "launch", "lookup" } },
        .{ .text = try block(a, v4, &.{ launch4, weather4 }), .names = &.{ "launch", "weather" } },
        .{ .text = try std.mem.concat(a, u8, &.{ xml, try block(a, v4, &.{launch4}) }), .names = &.{ "lookup", "launch" } },
    };
    return a.dupe(Case, &list);
}

test "a streamed reply's calls are the whole reply's, every character's end and every cut" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, tools_json)).ok.array;
    for (try replies(a)) |c| {
        const whole = try named(a, (try parse(a, c.text, tools, null)).calls);
        try expectNames(c.names, whole);
        try expectCalls(whole, try streamCalls(a, tools, c.text, null));
        for (0..c.text.len + 1) |n| {
            var run: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer run.deinit();
            try expectCalls(whole, try streamCalls(run.allocator(), tools, c.text, n));
        }
    }
}

/// Bytes as tokens, so a reply's characters arrive in pieces; decode replaces a character the ids cut with U+FFFD, as
/// model_text.Text says (Python's replace); every prompt renders as "prompt".
const Bytes = struct {
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
        var i: usize = 0;
        while (i < ids.len) {
            var char: [4]u8 = undefined;
            const n: usize = std.unicode.utf8ByteSequenceLength(@intCast(ids[i])) catch 0;
            var got: usize = 0;
            while (got < @max(n, 1) and i + got < ids.len and (got == 0 or ids[i + got] & 0xc0 == 0x80)) : (got += 1) char[got] = @intCast(ids[i + got]);
            if (n > 0 and got == n and std.unicode.utf8ValidateSlice(char[0..n])) try out.appendSlice(a, char[0..n]) else try out.appendSlice(a, "\u{fffd}");
            i += got;
        }
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

    fn render(_: *anyopaque, a: Allocator, _: Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine that writes ``reply`` a byte a step, then stops.
const Writer = struct {
    reply: []const u8 = "",

    fn engine(w: *@This()) api.Engine {
        return .{ .ctx = w, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const w: *Writer = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (w.reply) |byte| {
            const one = [_]u32{byte};
            sink.event(sink.ctx, id, &.{ .tokens = &one });
        }
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

/// What a chat client got: the reply's content, its calls and its finish reason, from the whole reply or joined from the
/// stream's chunks.
const Got = struct {
    a: Allocator,
    status: ?u16 = null,
    content: std.ArrayList(u8) = .empty,
    deltas: std.ArrayList(Value) = .empty,
    calls: []const Named = &.{},
    finish: []const u8 = "",

    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = 200;
    }

    fn event(ctx: *anyopaque, chunk: ?Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        const c = chunk orelse return;
        const choices = c.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const choice = choices.array[0];
        if (choice.get("delta")) |delta| {
            if (delta.strField("content")) |t| g.content.appendSlice(g.a, t) catch return error.Closed;
            if (delta.get("tool_calls") != null) g.deltas.append(g.a, delta) catch return error.Closed;
        }
        if (choice.strField("finish_reason")) |f| g.finish = f;
    }

    fn reply(ctx: *anyopaque, status: u16, body: Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status;
        const choices = body.get("choices") orelse return;
        const choice = choices.array[0];
        const message = choice.get("message").?;
        if (message.strField("content")) |t| g.content.appendSlice(g.a, t) catch {};
        if (message.get("tool_calls")) |c| g.calls = named(g.a, c.array) catch &.{};
        g.finish = choice.strField("finish_reason") orelse "";
    }
};

/// A chat completion offering ``tools_json``'s tools, whose reply the engine writes; one call a reply when ``single``.
fn chat(a: Allocator, srv: *Server, stream: bool, single: bool) !Got {
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"tools\":{s},\"max_tokens\":4096,\"stream\":{s}{s}}}", .{ tools_json, if (stream) "true" else "false", if (single) ",\"parallel_tool_calls\":false" else "" });
    const raw = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{ .a = a };
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, raw);
    if (stream) got.calls = try joined(a, got.deltas.items);
    return got;
}

test "through the server, a streamed reply's calls, content and finish reason are the whole reply's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var writer: Writer = .{};
    var text: Bytes = .{};
    const srv = try Server.init(std.testing.allocator, std.testing.io, writer.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const Chat = struct { text: []const u8, single: bool, names: []const []const u8 };
    var cases: std.ArrayList(Chat) = .empty;
    for (try replies(a)) |c| try cases.append(a, .{ .text = c.text, .single = false, .names = c.names });
    const v41 = dsml.v41;
    // one call a reply: only an offered tool's call, the first
    try cases.append(a, .{ .text = try block(a, v41, &.{ try invoke(a, v41, "launch", &.{}), try invoke(a, v41, "weather", &.{}) }), .single = true, .names = &.{"weather"} });
    try cases.append(a, .{ .text = try std.mem.concat(a, u8, &.{ "Launching.\n\n", try block(a, v41, &.{try invoke(a, v41, "launch", &.{})}) }), .single = true, .names = &.{} });
    // no call: a plain answer, and a <tool_call> whose text only starts with a DSML opener (d154)
    try cases.append(a, .{ .text = "Just text.", .single = false, .names = &.{} });
    try cases.append(a, .{ .text = "<tool_call><" ++ D ++ "tool_calls>x</tool_call>", .single = false, .names = &.{} });
    for (cases.items) |c| {
        writer.reply = c.text;
        const whole = try chat(a, srv, false, c.single);
        const streamed = try chat(a, srv, true, c.single);
        try std.testing.expectEqual(@as(?u16, 200), whole.status);
        try std.testing.expectEqual(@as(?u16, 200), streamed.status);
        try expectNames(c.names, whole.calls);
        try expectCalls(whole.calls, streamed.calls);
        try std.testing.expectEqualStrings(if (c.names.len > 0) "tool_calls" else "stop", whole.finish);
        try std.testing.expectEqualStrings(whole.finish, streamed.finish);
        try std.testing.expectEqualStrings(whole.content.items, std.mem.trim(u8, streamed.content.items, " \t\r\n"));
    }
}
