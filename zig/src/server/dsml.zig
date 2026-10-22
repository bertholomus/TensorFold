//! DeepSeek's DSML tool calls, V4's <｜DSML｜tool_calls> block and V4.1's <｜DSML｜ calls>: the one reader of a block's
//! calls that the four-node and the two-node lines both build, with its one test set (dsml_test.zig), so the lines read
//! a DSML call alike. tool_parse.zig finds the blocks in a reply (V4.1's first: reply_text.open41) and makes OpenAI calls
//! of what this reads; tool_stream.zig streams a block's calls once it closes, through tool_parse; reply_text.zig hides
//! the same blocks from streamed prose.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_params = @import("tool_params.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const strip = reply_text.pyStrip;

/// A spelling's tags: the block's (as reply_text.calls holds them), an invoke's opener up to its name and its closer, and
/// a parameter's likewise.
pub const Tags = struct {
    open: []const u8,
    close: []const u8,
    invoke: []const u8,
    invoke_close: []const u8,
    param: []const u8,
    param_close: []const u8,
};

/// DeepSeek-V4's spelling.
pub const v4: Tags = .{
    .open = reply_text.calls[2][0],
    .close = reply_text.calls[2][1],
    .invoke = "<\u{ff5c}DSML\u{ff5c}invoke name=\"",
    .invoke_close = "</\u{ff5c}DSML\u{ff5c}invoke>",
    .param = "<\u{ff5c}DSML\u{ff5c}parameter name=\"",
    .param_close = "</\u{ff5c}DSML\u{ff5c}parameter>",
};

/// DeepSeek-V4.1's (its chat template's, and the served Python lane's reply_text.py): a space after the marker, the
/// block named "calls".
pub const v41: Tags = .{
    .open = reply_text.calls[3][0],
    .close = reply_text.calls[3][1],
    .invoke = "<\u{ff5c}DSML\u{ff5c} invoke name=\"",
    .invoke_close = "</\u{ff5c}DSML\u{ff5c} invoke>",
    .param = "<\u{ff5c}DSML\u{ff5c} parameter name=\"",
    .param_close = "</\u{ff5c}DSML\u{ff5c} parameter>",
};

/// A call as the model wrote it: its name, stripped, and its arguments (an object).
pub const Call = struct { name: []const u8, arguments: Value };

/// The request's tools: each name lower-cased, to the spelling the request offered it in.
pub const Known = std.StringArrayHashMapUnmanaged([]const u8);

const Error = error{ Invalid, OutOfMemory };

fn find(hay: []const u8, needle: []const u8, from: usize) ?usize {
    if (from > hay.len) return null;
    return std.mem.indexOfPos(u8, hay, from, needle);
}

fn allSpace(s: []const u8) bool {
    return strip(s).len == 0;
}

/// ``block`` is one whole block in ``tags``' spelling, its opener to its closer. A <tool_call> whose text only starts
/// with an opener is not one: its inside, cut at the closer's length from its end, would start past its end.
pub fn whole(block: []const u8, tags: Tags) bool {
    return block.len >= tags.open.len + tags.close.len and std.mem.startsWith(u8, block, tags.open) and
        std.mem.endsWith(u8, block, tags.close);
}

/// The calls a whole block (``whole``) of either spelling makes, under the names the client gets: an offered tool's name
/// as the request spelled it, and the model's own name for a tool the request did not offer, so the client sees the
/// call and can answer that the tool is missing (as vLLM does). With a limit (one call a reply) only offered tools count,
/// as before: V4.1's invokes one by one, with finite arguments; V4's block whole, which drops out when it names a tool
/// not offered; up to the limit, past the ``made`` calls the reply already has. Null: a V4 block that is not well formed,
/// which stays the reply's text (and drops out of a one-call reply).
pub fn calls(a: Allocator, block: []const u8, known: *const Known, max_calls: ?usize, made: usize) Allocator.Error!?[]Call {
    var out: std.ArrayList(Call) = .empty;
    if (whole(block, v41)) {
        for (try read41(a, block[v41.open.len .. block.len - v41.close.len])) |c| {
            const offered = try offeredName(a, known, c.name);
            if (max_calls) |m| if (made + out.items.len >= m or offered == null or !tool_params.finite(c.arguments)) continue;
            try out.append(a, .{ .name = offered orelse c.name, .arguments = c.arguments });
        }
        return out.items;
    }
    if (!whole(block, v4)) return null;
    if (max_calls) |m| if (made >= m) return out.items;
    const read = read4(a, block[v4.open.len .. block.len - v4.close.len]) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Invalid => return null,
    };
    for (read) |c| {
        const offered = try offeredName(a, known, c.name);
        if (offered == null and max_calls != null) return null;
        if (max_calls == null or made + out.items.len < max_calls.?) try out.append(a, .{ .name = offered orelse c.name, .arguments = c.arguments });
    }
    return out.items;
}

/// The offered spelling of ``name``, or null when the request did not offer it.
fn offeredName(a: Allocator, known: *const Known, name: []const u8) Allocator.Error!?[]const u8 {
    return known.get(try std.ascii.allocLowerString(a, name));
}

/// A V4.1 block's inside as the served lane reads it (reply_text.py: dsml_to_tool_calls, then parse_tool_calls on each
/// invoke's ``<tool_call>{"name": .., "arguments": ..}</tool_call>``): every ``invoke name="N"`` up to its closer is a call
/// of its own, text between them is dropped; its ``parameter name="K"`` values are strings, or JSON with
/// ``string="false"`` (a string when that JSON does not parse). A name is stripped, as that parse strips it; a blank one
/// makes no call.
fn read41(a: Allocator, inside: []const u8) Allocator.Error![]Call {
    var out: std.ArrayList(Call) = .empty;
    var pos: usize = 0;
    while (find(inside, v41.invoke, pos)) |s| {
        pos = s + 1;
        const name_start = s + v41.invoke.len;
        const q = std.mem.indexOfScalarPos(u8, inside, name_start, '"') orelse continue;
        if (q == name_start or !std.mem.startsWith(u8, inside[q..], "\">")) continue;
        const e = find(inside, v41.invoke_close, q + 2) orelse continue;
        pos = e + v41.invoke_close.len;
        const name = strip(inside[name_start..q]);
        if (name.len == 0) continue;
        const body = inside[q + 2 .. e];
        const arguments = try json.newObject(a);
        var p: usize = 0;
        while (find(body, v41.param, p)) |ps| {
            p = ps + 1;
            const ns = ps + v41.param.len;
            const nq = std.mem.indexOfScalarPos(u8, body, ns, '"') orelse continue;
            if (nq == ns) continue;
            var v_start = nq + 1;
            var json_value = false;
            if (std.mem.startsWith(u8, body[v_start..], " string=\"true\"")) {
                v_start += " string=\"true\"".len;
            } else if (std.mem.startsWith(u8, body[v_start..], " string=\"false\"")) {
                v_start += " string=\"false\"".len;
                json_value = true;
            }
            if (v_start >= body.len or body[v_start] != '>') continue;
            v_start += 1;
            const pe = find(body, v41.param_close, v_start) orelse continue;
            p = pe + v41.param_close.len;
            const value = body[v_start..pe];
            const v: Value = if (json_value) switch (try json.parseText(a, value)) {
                .ok => |parsed| parsed,
                .err => .{ .string = value },
            } else .{ .string = value };
            try arguments.put(a, body[ns..nq], v);
        }
        try out.append(a, .{ .name = name, .arguments = .{ .object = arguments } });
    }
    return out.items;
}

/// Every invoke of a V4 block's inside, or Invalid when anything in it is not a well-formed invoke, a blank name
/// included.
fn read4(a: Allocator, inside: []const u8) Error![]Call {
    var out: std.ArrayList(Call) = .empty;
    var at: usize = 0;
    var pos: usize = 0;
    while (find(inside, v4.invoke, pos)) |s| {
        pos = s + 1;
        const name_start = s + v4.invoke.len;
        const q = std.mem.indexOfScalarPos(u8, inside, name_start, '"') orelse continue;
        if (!std.mem.startsWith(u8, inside[q..], "\">")) continue;
        const e = find(inside, v4.invoke_close, q + 2) orelse continue;
        if (!allSpace(inside[at..s])) return error.Invalid;
        at = e + v4.invoke_close.len;
        pos = at;
        const body = inside[q + 2 .. e];
        const arguments = try json.newObject(a);
        var p: usize = 0;
        var kept: usize = 0;
        var residue: std.ArrayList(u8) = .empty;
        while (find(body, v4.param, p)) |ps| {
            p = ps + 1;
            const ns = ps + v4.param.len;
            const nq = std.mem.indexOfScalarPos(u8, body, ns, '"') orelse continue;
            const flag = body[nq..];
            const is_string = std.mem.startsWith(u8, flag, "\" string=\"true\">");
            if (!is_string and !std.mem.startsWith(u8, flag, "\" string=\"false\">")) continue;
            const v_start = nq + (if (is_string) "\" string=\"true\">".len else "\" string=\"false\">".len);
            const pe = find(body, v4.param_close, v_start) orelse continue;
            const value = body[v_start..pe];
            try residue.appendSlice(a, body[kept..ps]);
            kept = pe + v4.param_close.len;
            p = kept;
            if (is_string) {
                try arguments.put(a, body[ns..nq], .{ .string = value });
            } else switch (try json.parseText(a, value)) {
                .ok => |v| try arguments.put(a, body[ns..nq], v),
                .err => return error.Invalid,
            }
        }
        try residue.appendSlice(a, body[kept..]);
        if (!allSpace(residue.items)) return error.Invalid;
        const name = strip(inside[name_start..q]);
        if (name.len == 0) return error.Invalid;
        try out.append(a, .{ .name = name, .arguments = .{ .object = arguments } });
    }
    if (out.items.len == 0 or !allSpace(inside[at..])) return error.Invalid;
    return out.items;
}
