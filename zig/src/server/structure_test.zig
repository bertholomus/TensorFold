//! json_object, json_schema and choice reach the engine with the token table and a program; the rest get a 400.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const anthropic_translate = @import("anthropic_translate.zig");
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

const think_end: u32 = 300;
const eos: u32 = 301;

/// Bytes as tokens, </think> and an end token past them; every prompt renders as "prompt".
const ByteText = struct {
    table: bool = true,

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = if (t.table) &with_table else &without_table };
    }

    const with_table: model_text.Text.VTable = .{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .vocab = vocab };
    const without_table: model_text.Text.VTable = .{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource };

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.items;
    }

    fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
        return if (std.mem.eql(u8, piece, "</think>")) think_end else null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 302;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{eos};
    }

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }

    const Source = struct {
        added: []const struct { id: u32 } = &.{.{ .id = think_end }},

        pub fn tokenBytes(_: *const Source, a: Allocator, id: u32) !?[]u8 {
            if (id >= 256) return null;
            return try a.dupe(u8, &.{@intCast(id)});
        }
    };

    fn vocab(_: *anyopaque, a: Allocator, ends: []const u32) anyerror!api.grammar.Vocab {
        const source: Source = .{};
        return api.grammar.Vocab.build(a, 302, &source, ends);
    }
};

/// An engine that keeps what the server submitted and answers "{}" at once.
const Capture = struct {
    structures: bool = true,
    submitted: bool = false,
    structure: ?api.Structure = null,
    kinds: [3]api.grammar.vocab.Kind = @splat(.none), // the table's kinds of '{', </think> and the end token
    think_close: [8]u32 = undefined,
    think_close_len: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 64, .structures = e.structures };
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        e.submitted = true;
        e.structure = r.structure;
        if (r.structure) |st| if (st.vocab) |v| {
            e.kinds = .{ v.kind('{'), v.kind(think_end), v.kind(eos) };
        };
        e.think_close_len = r.think_close.len;
        @memcpy(e.think_close[0..r.think_close.len], r.think_close);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ '{', '}', eos } });
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

/// The whole reply's status and error message.
const Seen = struct {
    status: u16 = 0,
    message: []const u8 = "",
    opened: bool = false,

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.opened = true;
    }

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.status = status;
        const err = payload.get("error") orelse return;
        const message = err.get("message") orelse return;
        if (message == .string) c.message = message.string;
    }
};

fn send(text: *ByteText, engine: *Capture, a: Allocator, body: []const u8) !Seen {
    return sendValue(text, engine, a, (try json.parse(a, body)).ok);
}

fn sendValue(text: *ByteText, engine: *Capture, a: Allocator, raw: json.Value) !Seen {
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, engine.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{};
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, raw);
    seen.message = try a.dupe(u8, seen.message);
    return seen;
}

const ask = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":8,";

test "json_object reaches an engine that enforces structures, with the token table" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var text: ByteText = .{};
    var engine: Capture = .{};
    const seen = try send(&text, &engine, arena.allocator(), ask ++ "\"response_format\":{\"type\":\"json_object\"}}");
    try std.testing.expectEqual(@as(u16, 200), seen.status);
    const st = engine.structure.?;
    try std.testing.expect(st.kind == .json and st.after == null and st.vocab != null);
    try std.testing.expectEqualSlices(api.grammar.vocab.Kind, &.{ .text, .special, .eos }, &engine.kinds);
}

const signal = "{\"type\":\"object\",\"additionalProperties\":false,\"required\":[\"action\",\"confidence\",\"pair\"]," ++
    "\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"buy\",\"sell\",\"hold\"]},\"confidence\":{\"type\":\"number\"},\"pair\":{\"type\":\"string\"}}}";

/// The compiled trade signal schema: an object of three required properties, the first an enum.
fn expectGate(st: api.Structure) !void {
    try std.testing.expect(st.kind == .json_schema and st.vocab != null);
    const p = st.program.?;
    const o = p.objects[p.nodes[p.root].object];
    try std.testing.expectEqual(@as(usize, 3), o.props.len);
    try std.testing.expectEqualStrings("\"confidence\"", o.props[1].key);
    try std.testing.expect(p.nodes[o.props[0].node] == .literal);
}

test "json_schema and guided_choice reach the engine with their compiled program, from either field" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bodies = [_][]const u8{
        ask ++ "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"trade\",\"strict\":true,\"schema\":" ++ signal ++ "}}}",
        ask ++ "\"stream\":true,\"guided_json\":" ++ signal ++ "}",
        ask ++ "\"structured_outputs\":{\"json\":" ++ signal ++ "}}",
    };
    for (bodies) |body| {
        var text: ByteText = .{};
        var engine: Capture = .{};
        const seen = try send(&text, &engine, a, body);
        try std.testing.expect(seen.status == 200 or seen.opened);
        try expectGate(engine.structure.?);
    }
    var text: ByteText = .{};
    var engine: Capture = .{};
    const seen = try send(&text, &engine, a, ask ++ "\"guided_choice\":[\"yes\",\"no\"]}");
    try std.testing.expectEqual(@as(u16, 200), seen.status);
    const st = engine.structure.?;
    try std.testing.expect(st.kind == .choice and st.program != null);
    const p = st.program.?;
    const root = p.nodes[p.root].literal;
    try std.testing.expect(p.child(root, 'y') != null and p.child(root, 'n') != null and p.child(root, 'x') == null);
}

test "an Anthropic output_config.format reaches the engine as a compiled json_schema" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = (try json.parse(a, "{\"model\":\"m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," ++
        "\"output_config\":{\"format\":{\"type\":\"json_schema\",\"schema\":" ++ signal ++ "}}}")).ok;
    var cx: server.errors.Cx = .{ .a = a };
    const chat = try anthropic_translate.translate(&cx, body, false);
    var text: ByteText = .{};
    var engine: Capture = .{};
    const seen = try sendValue(&text, &engine, a, chat);
    try std.testing.expectEqual(@as(u16, 200), seen.status);
    try expectGate(engine.structure.?);
}

test "a pydantic response_format with confidence bounds reaches the engine compiled, held to 0.0 to 1.0" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var engine: Capture = .{};
    // a whole response_format (name, schema, description, strict null) with pydantic's bounds; it got HTTP 400 before
    const seen = try send(&text, &engine, a, ask ++ "\"response_format\":" ++ @embedFile("testdata/pydantic_bounds.response_format.json") ++ "}");
    try std.testing.expectEqual(@as(u16, 200), seen.status);
    const st = engine.structure.?;
    try std.testing.expect(st.kind == .json_schema and st.program != null);
    for ([_][]const u8{ "0", "0.0", "0.85", "1", "1.0" }) |x| {
        var m: api.grammar.Matcher = .init(st.program.?);
        const reply = try std.fmt.allocPrint(a, "{{\"label\":\"yes\",\"confidence\":{s},\"summary\":\"s\"}}", .{x});
        try std.testing.expect(m.advance(reply) and m.complete());
    }
    for ([_][]const u8{ "1.01", "-0.1", "2" }) |x| {
        var m: api.grammar.Matcher = .init(st.program.?);
        try std.testing.expect(m.advance("{\"label\":\"yes\",\"confidence\":"));
        try std.testing.expect(!m.accepts(x));
    }
}

test "a grammar this engine cannot enforce is refused before the engine sees it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { body: []const u8, said: []const u8, structures: bool = true, table: bool = true }{
        .{ .body = ask ++ "\"guided_regex\":\"a+\"}", .said = "guided_regex: this engine enforces json_object, json_schema and choice, not regex" },
        .{ .body = ask ++ "\"structured_outputs\":{\"grammar\":\"root ::= \\\"a\\\"\"}}", .said = "structured_outputs: this engine enforces json_object, json_schema and choice, not grammar" },
        .{ .body = ask ++ "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":{\"type\":\"object\",\"properties\":{\"p\":{\"type\":\"string\",\"pattern\":\"^a\"}}}}}}", .said = "response_format: the schema cannot be enforced: #/properties/p: pattern: it is not enforced by this engine" },
        .{ .body = ask ++ "\"stream\":true,\"guided_json\":{\"anyOf\":[{\"type\":\"string\"},{\"enum\":[\"a\"]}]}}", .said = "guided_json: the schema cannot be enforced: #: two anyOf, oneOf or type branches can start with '\"': this engine needs branches that start with different bytes" },
        .{ .body = ask ++ "\"guided_choice\":[\"a\",\"\\ud800\"]}", .said = "guided_choice: a choice is not valid Unicode" },
        .{ .body = ask ++ "\"response_format\":{\"type\":\"json_object\"},\"ignore_eos\":true}", .said = "response_format needs the model's end token: send it without ignore_eos" },
        .{ .body = ask ++ "\"response_format\":{\"type\":\"json_object\"}}", .said = "structured output needs a tokenizer with a ByteLevel decoder, and this one has none", .table = false },
        .{ .body = ask ++ "\"response_format\":{\"type\":\"json_object\"}}", .said = "structured output (response_format and the guided_* fields) is not supported by this engine yet", .structures = false },
    };
    for (cases) |c| {
        var text: ByteText = .{ .table = c.table };
        var engine: Capture = .{ .structures = c.structures };
        const seen = try send(&text, &engine, a, c.body);
        try std.testing.expectEqual(@as(u16, 400), seen.status);
        try std.testing.expectEqualStrings(c.said, seen.message);
        try std.testing.expect(!engine.submitted and !seen.opened);
    }
}

test "under a grammar the thinking budget's close ends at </think>; without one it keeps its blank line" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const thinking = ask ++ "\"chat_template_kwargs\":{\"enable_thinking\":true},\"thinking_budget\":4";
    for ([_]bool{ true, false }) |structured| {
        var text: ByteText = .{};
        var engine: Capture = .{};
        const body = if (structured) thinking ++ ",\"response_format\":{\"type\":\"json_object\"}}" else thinking ++ "}";
        const seen = try send(&text, &engine, a, body);
        try std.testing.expectEqual(@as(u16, 200), seen.status);
        const close = engine.think_close[0..engine.think_close_len];
        if (structured) {
            try std.testing.expectEqualSlices(u32, &.{ '\n', think_end }, close);
            try std.testing.expectEqual(@as(?u32, think_end), engine.structure.?.after);
        } else try std.testing.expectEqualSlices(u32, &.{ '\n', think_end, '\n', '\n' }, close);
    }
}
