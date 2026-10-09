//! Chat messages as the template sees them: leading instructions merged, text parts joined, call arguments as objects.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const Value = json.Value;
const Cx = errors.Cx;

const roles = [_][]const u8{ "system", "developer", "user", "assistant", "tool" };

fn isRole(role: ?Value) bool {
    const r = role orelse return false;
    if (r != .string) return false;
    for (roles) |name| if (std.mem.eql(u8, name, r.string)) return true;
    return false;
}

fn withField(cx: *Cx, o: *const json.Object, key: []const u8, value: Value) !*json.Object {
    const copy = try json.copyObject(cx.a, o);
    try copy.put(cx.a, key, value);
    return copy;
}

/// A request's images as normalize meets them when the server takes images (its engine's Info.vision): each image
/// part's bytes in prompt order, the part itself the placeholder in its message's text.
pub const Images = struct {
    placeholder: []const u8,
    list: std.ArrayList([]const u8) = .empty,
};

pub const image_text_refusal = "image input: message text must not contain the image placeholder token";

pub const image_role_refusal = "image input: images are accepted only in user messages and tool results";
const text_only_refusal = "this server accepts text parts only; image, audio and video inputs are unsupported";

fn isImagePart(part: Value) bool {
    if (part != .object) return false;
    const t = part.get("type") orelse return false;
    return t == .string and std.mem.eql(u8, t.string, "image_url");
}

/// Whether any message's content list holds an image part.
fn hasImages(list: Value) bool {
    for (list.array) |message| {
        if (message != .object) continue;
        const content = message.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |part| if (isImagePart(part)) return true;
    }
    return false;
}

/// An ``image_url`` part's bytes: ``{"url": "data:...,<base64>"}`` (other URLs are off), the base64 as Python's
/// b64decode reads it.
fn imageBytes(cx: *Cx, part: Value) errors.Refused![]const u8 {
    const r = part.get("image_url") orelse return cx.refuse(text_only_refusal);
    if (r != .object) return cx.refuse(text_only_refusal);
    const u = r.get("url") orelse return cx.refuse(text_only_refusal);
    if (u != .string) return cx.refuse(text_only_refusal);
    const url = u.string;
    if (!std.mem.startsWith(u8, url, "data:")) return cx.refuse("image input: image URLs are off on this server (send a base64 data URL)");
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse url.len;
    if (std.mem.indexOf(u8, url[0..comma], ";base64") == null) return cx.fail(.request, "image input: Unsupported data URL encoding: {s}", .{url[0..comma]});
    const data = if (comma < url.len) url[comma + 1 ..] else "";
    return switch (b64decode(cx.a, data)) {
        .ok => |b| b,
        .short => |n| cx.fail(.request, "image input: Invalid base64-encoded string: number of data characters ({d}) cannot be 1 more than a multiple of 4", .{n}),
        .padding => cx.refuse("image input: Incorrect padding"),
        .oom => error.OutOfMemory,
    };
}

const B64 = union(enum) { ok: []u8, short: usize, padding, oom };

/// binascii.a2b_base64 as base64.b64decode calls it (not strict): characters outside the alphabet skipped, the input
/// read until a complete padding, an incomplete last quad an error (one character over: "short", with the data
/// characters it counts; two or three: "padding").
fn b64decode(a: std.mem.Allocator, s: []const u8) B64 {
    const out = a.alloc(u8, s.len / 4 * 3 + 3) catch return .oom;
    var n: usize = 0;
    var quad: u8 = 0;
    var left: u8 = 0;
    var pads: usize = 0;
    for (s) |ch| {
        if (ch == '=') {
            if (quad >= 2) {
                pads += 1;
                if (quad + pads >= 4) return .{ .ok = out[0..n] };
            }
            continue;
        }
        const v: u8 = switch (ch) {
            'A'...'Z' => ch - 'A',
            'a'...'z' => ch - 'a' + 26,
            '0'...'9' => ch - '0' + 52,
            '+' => 62,
            '/' => 63,
            else => continue,
        };
        pads = 0;
        switch (quad) {
            0 => {
                quad = 1;
                left = v;
            },
            1 => {
                quad = 2;
                out[n] = (left << 2) | (v >> 4);
                n += 1;
                left = v & 0x0f;
            },
            2 => {
                quad = 3;
                out[n] = (left << 4) | (v >> 2);
                n += 1;
                left = v & 0x03;
            },
            else => {
                quad = 0;
                out[n] = (left << 6) | v;
                n += 1;
                left = 0;
            },
        }
    }
    if (quad == 1) return .{ .short = n / 3 * 4 + 1 };
    if (quad != 0) return .padding;
    return .{ .ok = out[0..n] };
}

test "base64 as Python's b64decode reads it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("hello", b64decode(a, "aGVsbG8=").ok);
    try std.testing.expectEqualStrings("hello", b64decode(a, "aGVs\nbG8=").ok); // a newline skipped
    try std.testing.expectEqualStrings("hello", b64decode(a, "aGVsbG8=ignored").ok); // stops at the padding
    try std.testing.expectEqual(@as(usize, 9), b64decode(a, "aGVsbG8hZ").short); // 9 data characters
    try std.testing.expect(b64decode(a, "aGVsbG8") == .padding);
    try std.testing.expectEqualStrings("", b64decode(a, "").ok);
}

/// ``normalize_messages`` (text only): leading system and developer text merged, later ones as ``late_system``; a template that needs a user query gains one user turn after a trailing tool run.
pub fn normalize(cx: *Cx, messages: ?Value, late_system: []const u8, needs_user_after_tool: bool) errors.Refused!Value {
    return normalizeImages(cx, messages, late_system, needs_user_after_tool, null);
}

/// normalize, and with `images` (a server that takes them) a request with an image part anywhere as the served lane
/// renders it (the reference's process_image_messages): every message's content list joined with blank lines, each
/// image part its placeholder (its bytes into `images`), no message text holding the placeholder.
pub fn normalizeImages(cx: *Cx, messages: ?Value, late_system: []const u8, needs_user_after_tool: bool, images: ?*Images) errors.Refused!Value {
    const list = messages orelse return cx.refuse("messages must be a non-empty list");
    if (list != .array or list.array.len == 0) return cx.refuse("messages must be a non-empty list");
    const with_images = images != null and hasImages(list);
    var out: std.ArrayList(Value) = .empty;
    var instructions: std.ArrayList(*json.Object) = .empty;
    for (list.array) |message| {
        if (message != .object) return cx.refuse("each message must be an object");
        const role = message.get("role");
        if (!isRole(role)) return cx.refuse("message role must be system, developer, user, assistant or tool");
        if (with_images) {
            const im = images.?;
            // media other than the content's images is refused as ever
            const rest = try json.copyObject(cx.a, message.object);
            _ = rest.orderedRemove("content");
            if (fields.hasMedia(.{ .object = rest })) return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
            for ([_][]const u8{ "content", "reasoning_content" }) |key| if (message.get(key)) |v| if (v == .string and std.mem.indexOf(u8, v.string, im.placeholder) != null) return cx.refuse(image_text_refusal);
            const content = message.get("content");
            var item: *json.Object = message.object;
            if (content != null and content.? == .array) {
                const r = role.?.string;
                if (!std.mem.eql(u8, r, "user") and !std.mem.eql(u8, r, "tool")) for (content.?.array) |part| if (isImagePart(part)) return cx.refuse(image_role_refusal);
                var text: std.ArrayList(u8) = .empty;
                for (content.?.array, 0..) |part, k| {
                    if (k > 0) try text.appendSlice(cx.a, "\n\n");
                    if (isImagePart(part)) {
                        try im.list.append(cx.a, try imageBytes(cx, part));
                        try text.appendSlice(cx.a, im.placeholder);
                        continue;
                    }
                    const typed = part == .object and part.get("type") != null and part.get("type").? == .string and std.mem.eql(u8, part.get("type").?.string, "text");
                    if (!typed or fields.hasMedia(part)) return cx.refuse(text_only_refusal);
                    const t: Value = part.get("text") orelse .null;
                    if (t != .string and t != .null) return cx.refuse("a text content part must contain a text string");
                    if (t == .string) {
                        if (std.mem.indexOf(u8, t.string, im.placeholder) != null) return cx.refuse(image_text_refusal);
                        try text.appendSlice(cx.a, t.string);
                    }
                }
                item = try withField(cx, message.object, "content", .{ .string = text.items });
            } else if (content == null or content.? == .null) {
                item = try withField(cx, message.object, "content", .{ .string = "" });
            } else if (content.? != .string) {
                return cx.refuse("message content must be text or an array of text parts");
            }
            try place(cx, &out, &instructions, item, role.?.string, late_system);
            continue;
        }
        if (fields.hasMedia(message)) return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
        const content = message.get("content");
        var item: *json.Object = message.object;
        if (content != null and content.? == .array) {
            var text: std.ArrayList(u8) = .empty;
            for (content.?.array) |part| {
                const typed = part == .object and part.get("type") != null and part.get("type").? == .string and std.mem.eql(u8, part.get("type").?.string, "text");
                if (!typed or fields.hasMedia(part)) return cx.refuse("this server accepts text parts only; image, audio and video inputs are unsupported");
                const t = part.get("text") orelse return cx.refuse("a text content part must contain a text string");
                if (t != .string) return cx.refuse("a text content part must contain a text string");
                try text.appendSlice(cx.a, t.string);
            }
            item = try withField(cx, message.object, "content", .{ .string = text.items });
        } else if (content == null or content.? == .null) {
            item = try withField(cx, message.object, "content", .{ .string = "" });
        } else if (content.? != .string) {
            return cx.refuse("message content must be text or an array of text parts");
        }
        try place(cx, &out, &instructions, item, role.?.string, late_system);
    }
    if (instructions.items.len > 0) {
        var joined: std.ArrayList(u8) = .empty;
        for (instructions.items, 0..) |m, i| {
            if (i > 0) try joined.appendSlice(cx.a, "\n\n");
            try joined.appendSlice(cx.a, m.get("content").?.string);
        }
        const first = try withField(cx, instructions.items[0], "role", .{ .string = "system" });
        try first.put(cx.a, "content", .{ .string = joined.items });
        try out.insert(cx.a, 0, .{ .object = first });
    }
    // A template that demands a user query (it raises "No user query found") refuses a conversation whose user turn became tool results; by the template's test a user turn wholly inside a <|im_start|> block is no query, so only a conversation without one gains a placeholder user turn after its last tool run.
    if (needs_user_after_tool) {
        var has_query = false;
        var last_tool: ?usize = null;
        for (out.items, 0..) |m, i| {
            const role = m.get("role");
            if (role == null or role.? != .string) continue;
            if (std.mem.eql(u8, role.?.string, "user")) {
                const content = m.get("content");
                const text = if (content != null and content.? == .string) content.?.string else "";
                const trimmed = std.mem.trim(u8, text, " \t\n\r\x0b\x0c");
                if (!(std.mem.startsWith(u8, trimmed, "<tool_response>") and std.mem.endsWith(u8, trimmed, "</tool_response>"))) has_query = true;
            } else if (std.mem.eql(u8, role.?.string, "tool")) {
                last_tool = i;
            }
        }
        if (!has_query) {
            const at = if (last_tool) |i| i + 1 else out.items.len;
            const turn = try json.newObject(cx.a);
            try turn.put(cx.a, "role", .{ .string = "user" });
            try turn.put(cx.a, "content", .{ .string = "(tool results above)" });
            try out.insert(cx.a, at, .{ .object = turn });
        }
    }
    return .{ .array = out.items };
}

/// A normalized message into the list: leading instructions held to be merged, a later one as ``late_system``.
fn place(cx: *Cx, out: *std.ArrayList(Value), instructions: *std.ArrayList(*json.Object), message: *json.Object, r: []const u8, late_system: []const u8) errors.Refused!void {
    var item = message;
    if (std.mem.eql(u8, r, "system") or std.mem.eql(u8, r, "developer")) {
        if (out.items.len == 0) {
            try instructions.append(cx.a, item);
            return;
        }
        if (!std.mem.eql(u8, r, late_system)) item = try withField(cx, item, "role", .{ .string = late_system });
    }
    try out.append(cx.a, .{ .object = item });
}

/// ``_normalize_tool_call_arguments``: call arguments as objects for templates; bad ones under ``_invalid_arguments``.
pub fn toolArguments(cx: *Cx, messages: Value) errors.Refused!Value {
    if (messages != .array) return messages;
    const out = try cx.a.alloc(Value, messages.array.len);
    for (messages.array, out) |message, *slot| {
        slot.* = message;
        const calls = message.get("tool_calls") orelse continue;
        if (calls != .array or calls.array.len == 0) continue;
        var touched = false;
        const new_calls = try cx.a.alloc(Value, calls.array.len);
        for (calls.array, new_calls) |call, *dst| {
            dst.* = call;
            const function = call.get("function") orelse continue;
            if (function != .object or !function.has("arguments")) continue;
            const args = function.get("arguments").?;
            if (args == .object) continue;
            var parsed: ?Value = args;
            if (args == .string) {
                parsed = switch (try json.parseText(cx.a, args.string)) {
                    .ok => |v| v,
                    .err => null,
                };
            }
            if (parsed == null or parsed.? != .object) {
                const wrapped = try json.newObject(cx.a);
                try wrapped.put(cx.a, "_invalid_arguments", args);
                parsed = .{ .object = wrapped };
            }
            const fn_copy = try withField(cx, function.object, "arguments", parsed.?);
            dst.* = .{ .object = try withField(cx, call.object, "function", .{ .object = fn_copy }) };
            touched = true;
        }
        if (touched) slot.* = .{ .object = try withField(cx, message.object, "tool_calls", .{ .array = new_calls }) };
    }
    return .{ .array = out };
}

/// ``{"type": "text", "text": text}``: a chat content part.
pub fn textPart(a: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error!Value {
    const p = try json.newObject(a);
    try p.put(a, "type", .{ .string = "text" });
    try p.put(a, "text", .{ .string = text });
    return .{ .object = p };
}

/// A short title request without tools (``is_title_request``): it yields to foreground turns.
pub fn isTitleRequest(messages: Value, has_tools: bool) bool {
    if (has_tools or messages != .array or messages.array.len == 0) return false;
    const first = messages.array[0];
    const role = first.get("role") orelse return false;
    if (role != .string or !std.mem.eql(u8, role.string, "system")) return false;
    var size: usize = 0;
    for (messages.array) |m| {
        const c = m.get("content");
        size += if (c != null and c.? == .string) std.unicode.utf8CountCodepoints(c.?.string) catch c.?.string.len else 4096;
    }
    const text = first.get("content") orelse return false;
    if (size >= 4096 or text != .string) return false;
    return std.ascii.findIgnoreCase(text.string, "title") != null;
}

test "a conversation with a user query stays byte-identical under the rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "user", "content": "run it"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "g", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t2", "content": "ok"}, {"role": "assistant", "content": "done"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", true);
    // byte-identical: the same messages, in order, nothing gained
    try std.testing.expectEqualStrings(try json.stringify(cx.a, messages, .{ .ascii = false }), try json.stringify(cx.a, out, .{ .ascii = false }));
}

test "a conversation with no user query gains one placeholder turn after its last tool run" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}, {"role": "tool", "tool_call_id": "t2", "content": "ok"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", true);
    try std.testing.expectEqual(@as(usize, 4), out.array.len);
    try std.testing.expectEqualStrings("tool", out.array[1].get("role").?.string);
    try std.testing.expectEqualStrings("42 rows", out.array[1].get("content").?.string); // the tool messages unchanged
    try std.testing.expectEqualStrings("tool", out.array[2].get("role").?.string);
    try std.testing.expectEqualStrings("ok", out.array[2].get("content").?.string);
    try std.testing.expectEqualStrings("user", out.array[3].get("role").?.string);
    try std.testing.expectEqualStrings("(tool results above)", out.array[3].get("content").?.string);
}

test "a user message whose content is wholly a tool block does not count as a query" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const user = try json.newObject(cx.a);
    try user.put(cx.a, "role", .{ .string = "user" });
    try user.put(cx.a, "content", .{ .string = "<tool_response>\ngot 42 rows\n</tool_response>" });
    const assistant = try json.newObject(cx.a);
    try assistant.put(cx.a, "role", .{ .string = "assistant" });
    try assistant.put(cx.a, "content", .{ .string = "" });
    const tool = try json.newObject(cx.a);
    try tool.put(cx.a, "role", .{ .string = "tool" });
    try tool.put(cx.a, "tool_call_id", .{ .string = "t1" });
    try tool.put(cx.a, "content", .{ .string = "done" });
    const messages: Value = .{ .array = try cx.a.dupe(Value, &.{ .{ .object = user }, .{ .object = assistant }, .{ .object = tool } }) };
    const out = try normalize(&cx, messages, "system", true);
    try std.testing.expectEqual(@as(usize, 4), out.array.len);
    try std.testing.expectEqualStrings("user", out.array[3].get("role").?.string);
    try std.testing.expectEqualStrings("(tool results above)", out.array[3].get("content").?.string);
}

test "a conversation keeps its shape when the template does not need the rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const messages = (try json.parse(cx.a,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}]}, {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}]
    )).ok;
    const out = try normalize(&cx, messages, "system", false);
    try std.testing.expectEqual(@as(usize, 2), out.array.len);
    try std.testing.expectEqualStrings("assistant", out.array[0].get("role").?.string);
    try std.testing.expectEqualStrings("tool", out.array[1].get("role").?.string);
}
