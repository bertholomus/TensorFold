//! tool_choice "required" or a named function as a grammar: calls in the template's markup, arguments in their schema.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const prompt = @import("prompt.zig");
const tool_specs = @import("tool_specs.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// How the template writes a call: inside <tool_call>, Qwen3-Coder's XML (Nemotron, Qwen3.5) or a JSON object; or
/// DeepSeek's DSML, one block of invokes whose parameters are tags (V4's <｜DSML｜tool_calls>, V4.1's <｜DSML｜ calls>).
pub const Style = enum { xml, json, dsml, dsml41 };

/// A DSML spelling: the block's tags, an invoke's opener up to its name and its closer, a parameter's opener up to its
/// name, and xgrammar's json_schema style for the parameters. The layout is the template's (encoding.py's), as
/// xgrammar's own DeepSeek structural tags write it: "<block>\n", each invoke "<invoke name=...>\n", its parameters
/// one a line, "</invoke>\n", then "</block>".
const Dsml = struct { open: []const u8, close: []const u8, invoke: []const u8, invoke_end: []const u8, parameter: []const u8, params: []const u8 };
const marker = "\u{ff5c}DSML\u{ff5c}";
const dsml_v4: Dsml = .{ .open = "<" ++ marker ++ "tool_calls>", .close = "</" ++ marker ++ "tool_calls>", .invoke = "<" ++ marker ++ "invoke name=\"", .invoke_end = "</" ++ marker ++ "invoke>", .parameter = "<" ++ marker ++ "parameter name=\"", .params = "deepseek_xml" };
const dsml_v41: Dsml = .{ .open = "<" ++ marker ++ " calls>", .close = "</" ++ marker ++ " calls>", .invoke = "<" ++ marker ++ " invoke name=\"", .invoke_end = "</" ++ marker ++ " invoke>", .parameter = "<" ++ marker ++ " parameter name=\"", .params = "deepseek_v4_1_xml" };

fn dsmlOf(s: Style) ?Dsml {
    return switch (s) {
        .dsml => dsml_v4,
        .dsml41 => dsml_v41,
        .xml, .json => null,
    };
}

const probe = "tfprobe_fn";
const key = "tfprobe_k";

/// Blanks around and between calls: the template's newlines, never an endless run.
const blanks = "[ \\t\\r\\n]{0,8}";

/// The most blank characters between an argument's JSON tokens (grammar.py BLANKS).
const value_blanks = 32;

/// The style of `rendered`, a turn that calls `probe` with {key: "v"}; null: markup this grammar does not write.
pub fn style(rendered: []const u8) ?Style {
    inline for (.{ .{ Style.dsml41, dsml_v41 }, .{ Style.dsml, dsml_v4 } }) |c| if (std.mem.lastIndexOf(u8, rendered, c[1].open)) |at| {
        const text = rendered[at..];
        if (std.mem.indexOf(u8, text, c[1].invoke ++ probe ++ "\">") != null and std.mem.indexOf(u8, text, c[1].parameter ++ key ++ "\"") != null) return c[0];
    };
    const at = std.mem.lastIndexOf(u8, rendered, "<tool_call>") orelse return null;
    const text = rendered[at..];
    if (std.mem.indexOf(u8, text, "<function=" ++ probe ++ ">") != null and std.mem.indexOf(u8, text, "<parameter=" ++ key ++ ">") != null) return .xml;
    if (std.mem.indexOf(u8, text, "\"" ++ probe ++ "\"") != null and std.mem.indexOf(u8, text, "\"arguments\"") != null) return .json;
    return null;
}

/// The grammar for a reply that must call one of `tools`: one call, or (`parallel`) one or more, then the end.
pub fn structure(srv: *Server, cx: *Cx, tools: []const Value, parallel: bool, thinking: bool) errors.Refused!api.Structure {
    const probe_turn = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"" ++ probe ++ "\", \"arguments\": {\"" ++ key ++ "\": \"v\"}}}]}]")).ok;
    var scratch: Cx = .{ .a = cx.a };
    const rendered = if (prompt.renderIds(srv, &scratch, probe_turn, &.{}, false, null, false)) |ids| try srv.text.decode(cx.a, ids) else |_| "";
    const s = style(rendered) orelse return cx.refuse("tool_choice \"required\" or a named function needs a chat template that writes <tool_call> calls (XML parameters or a JSON object) or DeepSeek's DSML calls, and this one does not: send \"auto\"");
    const text = try build(cx.a, s, tools, parallel);
    return srv.compiledStructure(cx, .structural_tag, text, "tool_choice \"required\" or a named function", thinking);
}

/// The structural tag's JSON text (xgrammar's form).
pub fn build(a: Allocator, s: Style, tools: []const Value, parallel: bool) Allocator.Error![]const u8 {
    const calls = try a.alloc(Value, tools.len);
    for (tools, calls) |t, *c| c.* = try call(a, s, try tool_specs.toolName(a, t), try parameters(a, t));
    const one: Value = if (calls.len == 1) calls[0] else try node(a, "or", &.{.{ "elements", .{ .array = calls } }});
    const ws = try node(a, "regex", &.{.{ "pattern", .{ .string = blanks } }});
    var parts: std.ArrayList(Value) = .empty;
    if (dsmlOf(s)) |d| {
        // one block holds every call, the invokes back to back (each ends with the template's newline)
        try parts.appendSlice(a, &.{ ws, try constant(a, try std.mem.concat(a, u8, &.{ d.open, "\n" })), one });
        if (parallel) try parts.append(a, try node(a, "star", &.{.{ "content", one }}));
        try parts.appendSlice(a, &.{ try constant(a, d.close), ws });
    } else {
        try parts.appendSlice(a, &.{ ws, one });
        if (parallel) try parts.append(a, try node(a, "star", &.{.{ "content", try node(a, "sequence", &.{.{ "elements", try list(a, &.{ ws, one }) }}) }}));
        try parts.append(a, ws);
    }
    const format = try node(a, "sequence", &.{.{ "elements", .{ .array = parts.items } }});
    return json.stringify(a, try node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
}

/// One call to `name`: <tool_call>, the name and its arguments in the template's style, </tool_call>; for DSML, an
/// invoke of `name` with its parameters in their tags.
fn call(a: Allocator, s: Style, name: []const u8, schema: Value) Allocator.Error!Value {
    const ws = try node(a, "regex", &.{.{ "pattern", .{ .string = blanks } }});
    const elements = switch (s) {
        .xml => try list(a, &.{
            try constant(a, "<tool_call>"),                                        ws,
            try constant(a, try std.fmt.allocPrint(a, "<function={s}>", .{name})), try arguments(a, schema, "qwen_xml"),
            try constant(a, "</function>"),                                        ws,
            try constant(a, "</tool_call>"),
        }),
        .json => blk: {
            const props = try json.newObject(a);
            const fixed = try json.newObject(a);
            try fixed.put(a, "const", .{ .string = name });
            try props.put(a, "name", .{ .object = fixed });
            try props.put(a, "arguments", schema);
            const whole = try json.newObject(a);
            try whole.put(a, "type", .{ .string = "object" });
            try whole.put(a, "properties", .{ .object = props });
            try whole.put(a, "required", try list(a, &.{ .{ .string = "name" }, .{ .string = "arguments" } }));
            break :blk try list(a, &.{ try constant(a, "<tool_call>"), ws, try arguments(a, .{ .object = whole }, "json"), ws, try constant(a, "</tool_call>") });
        },
        .dsml, .dsml41 => blk: {
            const d = dsmlOf(s).?;
            break :blk try list(a, &.{
                try constant(a, try std.mem.concat(a, u8, &.{ d.invoke, name, "\">\n" })),
                try arguments(a, schema, d.params),
                try constant(a, try std.mem.concat(a, u8, &.{ d.invoke_end, "\n" })),
            });
        },
    };
    return node(a, "sequence", &.{.{ "elements", elements }});
}

/// The tool's parameter schema (OpenAI's function.parameters, a bare tool's parameters or input_schema).
fn parameters(a: Allocator, tool: Value) Allocator.Error!Value {
    const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
    for ([_][]const u8{ "parameters", "input_schema" }) |k| if (function.get(k)) |p| if (p == .object) return p;
    const empty = try json.newObject(a);
    try empty.put(a, "type", .{ .string = "object" });
    try empty.put(a, "properties", .{ .object = try json.newObject(a) });
    return .{ .object = empty };
}

fn arguments(a: Allocator, schema: Value, style_name: []const u8) Allocator.Error!Value {
    return node(a, "json_schema", &.{ .{ "json_schema", schema }, .{ "style", .{ .string = style_name } }, .{ "max_whitespace_cnt", try json.intValue(a, value_blanks) } });
}

fn constant(a: Allocator, text: []const u8) Allocator.Error!Value {
    return node(a, "const_string", &.{.{ "value", .{ .string = text } }});
}

fn list(a: Allocator, items: []const Value) Allocator.Error!Value {
    return .{ .array = try a.dupe(Value, items) };
}

/// {"type": kind, fields...}.
fn node(a: Allocator, kind: []const u8, entries: []const struct { []const u8, Value }) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = kind });
    for (entries) |e| try o.put(a, e[0], e[1]);
    return .{ .object = o };
}

test "the probe call's markup picks the style" {
    try std.testing.expectEqual(Style.xml, style("<think></think><tool_call>\n<function=tfprobe_fn>\n<parameter=tfprobe_k>\nv\n</parameter>\n</function>\n</tool_call>\n").?);
    try std.testing.expectEqual(Style.json, style("<tool_call>\n{\"name\": \"tfprobe_fn\", \"arguments\": {\"tfprobe_k\": \"v\"}}\n</tool_call>").?);
    try std.testing.expectEqual(@as(?Style, null), style("<|tool_call>call:tfprobe_fn{tfprobe_k:<|\"|>v<|\"|>}<tool_call|>"));
    const v41 = "</think>\n\n<" ++ marker ++ " calls>\n<" ++ marker ++ " invoke name=\"tfprobe_fn\">\n<" ++ marker ++ " parameter name=\"tfprobe_k\" string=\"true\">v</" ++ marker ++ " parameter>\n</" ++ marker ++ " invoke>\n</" ++ marker ++ " calls>";
    try std.testing.expectEqual(Style.dsml41, style(v41).?);
    const v4 = "\n\n<" ++ marker ++ "tool_calls>\n<" ++ marker ++ "invoke name=\"tfprobe_fn\">\n<" ++ marker ++ "parameter name=\"tfprobe_k\" string=\"true\">v</" ++ marker ++ "parameter>\n</" ++ marker ++ "invoke>\n</" ++ marker ++ "tool_calls>";
    try std.testing.expectEqual(Style.dsml, style(v4).?);
    // the block's tag alone, as a tools prompt shows it, is no call
    try std.testing.expectEqual(@as(?Style, null), style("write a \"<" ++ marker ++ " calls>\" block"));
}

test "a required call's grammar names every offered tool and its schema; one call without parallel" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"now\"}}]")).ok;
    const many = try build(a, .xml, tools.array, true);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<function=get_weather>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<function=now>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"style\":\"qwen_xml\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"required\":[\"city\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"type\":\"star\"") != null);
    const single = try build(a, .json, tools.array[0..1], false);
    try std.testing.expect(std.mem.indexOf(u8, single, "\"type\":\"star\"") == null and std.mem.indexOf(u8, single, "\"type\":\"or\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, single, "\"const\":\"get_weather\"") != null);
}

test "a DSML grammar holds every call in one block, each invoke's parameters in the template's tags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"now\"}}]")).ok;
    const many = try build(a, .dsml41, tools.array, true);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<" ++ marker ++ " calls>\\n\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"</" ++ marker ++ " calls>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<" ++ marker ++ " invoke name=\\\"get_weather\\\">\\n\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<" ++ marker ++ " invoke name=\\\"now\\\">\\n\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"</" ++ marker ++ " invoke>\\n\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"style\":\"deepseek_v4_1_xml\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"type\":\"star\"") != null and std.mem.indexOf(u8, many, "\"type\":\"or\"") != null);
    // the block opens once: a parallel reply's invokes repeat inside it, not the block
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, many, "<" ++ marker ++ " calls>"));
    const single = try build(a, .dsml41, tools.array[0..1], false);
    try std.testing.expect(std.mem.indexOf(u8, single, "\"type\":\"star\"") == null and std.mem.indexOf(u8, single, "\"type\":\"or\"") == null);
    const v4 = try build(a, .dsml, tools.array[0..1], false);
    try std.testing.expect(std.mem.indexOf(u8, v4, "\"value\":\"<" ++ marker ++ "tool_calls>\\n\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, v4, "\"style\":\"deepseek_xml\"") != null);
}

/// How far a reply's ids walk in a grammar: the ids it takes, and whether it then takes the stop and ends.
const Walk = struct { taken: usize, ended: bool };

fn walk(c: *api.grammar.Compiler, io: std.Io, grammar_text: []const u8, ids: []const u32, stop: u32) !Walk {
    var err: [512]u8 = undefined;
    const g = c.compile(io, .structural_tag, grammar_text, &err) catch |e| {
        std.debug.print("the grammar does not compile: {s}\n", .{std.mem.sliceTo(&err, 0)});
        return e;
    };
    defer g.free();
    const m = try g.matcher();
    defer m.free();
    const r = m.rules();
    for (ids, 0..) |t, i| if (!try r.vtable.accept(r.ptr, t)) return .{ .taken = i, .ended = false };
    return .{ .taken = ids.len, .ended = try r.vtable.accept(r.ptr, stop) and r.vtable.terminated(r.ptr) };
}

test "DSML grammars through libtfgrammar, a checkpoint's tokenizer and its template: the template's calls end at the stop; prose, an unoffered name, a missing or mistyped parameter and a second call when single do not" {
    // opt-in: TF_GRAMMAR_DIR=<a DeepSeek-V4.1 checkpoint folder: tokenizer.json, tokenizer_config.json, config.json>
    // TF_GRAMMAR_TEMPLATE=<its chat template> TF_GRAMMAR_VOCAB=<its logits' width> TF_GRAMMAR_STOP=<its eos id>, and
    // libtfgrammar where the loader finds it
    const dir = std.testing.environ.getPosix("TF_GRAMMAR_DIR") orelse return error.SkipZigTest;
    const template = std.testing.environ.getPosix("TF_GRAMMAR_TEMPLATE") orelse return error.SkipZigTest;
    const vocab = try std.fmt.parseInt(u32, std.testing.environ.getPosix("TF_GRAMMAR_VOCAB") orelse return error.SkipZigTest, 10);
    const stop = try std.fmt.parseInt(u32, std.testing.environ.getPosix("TF_GRAMMAR_STOP") orelse return error.SkipZigTest, 10);
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const hf = try @import("hf_text.zig").HfText.load(gpa, io, dir, template, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const tok = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "tokenizer.json" }), a, .limited(256 << 20));
    var err: [512]u8 = undefined;
    var c = try api.grammar.Compiler.open(io, a, tok, vocab, &.{stop}, &err);
    defer c.close();

    // the template's own probe turn picks the style, as structure() renders it
    const probe_turn = (try json.parse(a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"" ++ probe ++ "\", \"arguments\": {\"" ++ key ++ "\": \"v\"}}}]}]")).ok;
    const s = style(try text.render(a, probe_turn, .{ .add_generation_prompt = false }, &problem)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(Style.dsml41, s);

    const tools = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"city\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"now\"}}]")).ok.array;
    const any = try build(a, s, tools, true);
    const single = try build(a, s, tools, false);
    const weather_only = try build(a, s, tools[0..1], false);
    const now_only = try build(a, s, tools[1..2], false);

    // a reply as the template writes an assistant turn's calls: the block from its blank lines to its closer
    const Reply = struct {
        fn ids(t: @import("model_text.zig").Text, al: Allocator, calls: []const u8, prose: []const u8) ![]u32 {
            var p: []const u8 = "";
            const turn = try std.fmt.allocPrint(al, "[{{\"role\": \"user\", \"content\": \"x\"}}, {{\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": {s}}}]", .{calls});
            const whole = try t.render(al, (try json.parse(al, turn)).ok, .{ .add_generation_prompt = false }, &p);
            const at = std.mem.indexOf(u8, whole, "\n\n<" ++ marker ++ " calls>") orelse return error.NoCalls;
            const close = "</" ++ marker ++ " calls>";
            const end = (std.mem.indexOfPos(u8, whole, at, close) orelse return error.NoCalls) + close.len;
            return t.encode(al, try std.mem.concat(al, u8, &.{ prose, whole[at..end] }), false);
        }
    };
    const weather = "{\"type\": \"function\", \"function\": {\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": 3}}}";
    const now = "{\"type\": \"function\", \"function\": {\"name\": \"now\", \"arguments\": {}}}";
    const one = try Reply.ids(text, a, "[" ++ weather ++ "]", "");
    const two = try Reply.ids(text, a, "[" ++ weather ++ ", " ++ now ++ "]", "");
    const empty = try Reply.ids(text, a, "[" ++ now ++ "]", "");
    // the DSML marker is its token in each reply, as the model writes it
    try std.testing.expect(std.mem.indexOfScalar(u32, one, text.tokenId(marker).?) != null);

    const Case = struct { grammar: []const u8, ids: []const u32, ends: bool };
    const cases = [_]Case{
        .{ .grammar = any, .ids = one, .ends = true },
        .{ .grammar = any, .ids = two, .ends = true },
        .{ .grammar = single, .ids = one, .ends = true },
        .{ .grammar = single, .ids = two, .ends = false }, // a second call where one is asked for
        .{ .grammar = now_only, .ids = empty, .ends = true }, // a call with no arguments
        .{ .grammar = weather_only, .ids = empty, .ends = false }, // a named function: no other
        .{ .grammar = any, .ids = try Reply.ids(text, a, "[" ++ weather ++ "]", "Sure."), .ends = false }, // prose first
        .{ .grammar = any, .ids = try Reply.ids(text, a, "[{\"type\": \"function\", \"function\": {\"name\": \"forecast\", \"arguments\": {\"city\": \"Oslo\"}}}]", ""), .ends = false },
        .{ .grammar = any, .ids = try Reply.ids(text, a, "[{\"type\": \"function\", \"function\": {\"name\": \"weather\", \"arguments\": {\"days\": 3}}}]", ""), .ends = false }, // city missing
        .{ .grammar = any, .ids = try Reply.ids(text, a, "[{\"type\": \"function\", \"function\": {\"name\": \"weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": \"three\"}}}]", ""), .ends = false }, // days a string
        .{ .grammar = try build(a, .dsml, tools, true), .ids = one, .ends = false }, // V4's spelling is another block
    };
    for (cases, 0..) |case, i| {
        const w = try walk(&c, io, case.grammar, case.ids, stop);
        if (w.ended != case.ends or (case.ends and w.taken != case.ids.len)) {
            std.debug.print("case {d}: took {d} of {d} ids, ended {}: {s}\n", .{ i, w.taken, case.ids.len, w.ended, try text.decode(a, case.ids[0..@min(case.ids.len, w.taken + 1)]) });
            return error.TestUnexpectedResult;
        }
    }
}
