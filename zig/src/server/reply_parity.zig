//! Reads a reply-parity corpus through the server's think split and tool-call parse (reply_text.splitThinking at the
//! reply's end, tool_parse.parse as openai.zig applies it) and writes each result for a byte comparison with another
//! server's (TP4's tools/dsv41/tp4/m7_reply_corpus.py: the served DeepSeek-V4.1 lane's split_thinking and
//! parse_tool_calls). Corpus: {"tools": [[spec, ..], ..], "cases": [{"text", "tl" (index or null), "thinking", "max"}]}.
//! Results: [{"reasoning", "answer", "content", "calls": [[name, arguments], ..]}] in case order.
const std = @import("std");
const json = @import("json.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const Value = json.Value;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("usage: reply_parity <corpus.json> <results.json>\n", .{});
        return error.InvalidArguments;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 31));
    const corpus = switch (try json.parse(a, bytes)) {
        .ok => |v| v,
        .err => |m| {
            std.debug.print("corpus: {s}\n", .{m});
            return error.BadCorpus;
        },
    };
    const tool_sets = corpus.get("tools").?.array;
    const cases = corpus.get("cases").?.array;
    const out = try a.alloc(Value, cases.len);
    for (cases, out) |c, *slot| {
        const text = c.get("text").?.string;
        const tl = c.get("tl").?;
        const tools: []const Value = if (tl == .int) tool_sets[try std.fmt.parseInt(usize, tl.int, 10)].array else &.{};
        const max: ?usize = if (c.field("max")) |m| try std.fmt.parseInt(usize, m.int, 10) else null;
        var reasoning: Value = .null;
        var answer = text;
        if (c.get("thinking").?.bool) {
            const split = try reply_text.splitThinking(a, text, true, reply_text.think_markers);
            reasoning = .{ .string = split.reasoning };
            answer = split.answer;
        }
        const parsed = try tool_parse.parse(a, answer, tools, max);
        var calls: std.ArrayList(Value) = .empty;
        if (parsed.calls) |list| for (list) |call| {
            const f = call.get("function").?;
            const pair = try a.alloc(Value, 2);
            pair[0] = f.get("name").?;
            pair[1] = f.get("arguments").?;
            try calls.append(a, .{ .array = pair });
        };
        const row = try json.newObject(a);
        try row.put(a, "reasoning", reasoning);
        try row.put(a, "answer", .{ .string = answer });
        try row.put(a, "content", .{ .string = parsed.content });
        try row.put(a, "calls", .{ .array = calls.items });
        slot.* = .{ .object = row };
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = try json.stringify(a, .{ .array = out }, .{ .ascii = false }) });
    std.debug.print("{d} cases\n", .{cases.len});
}
