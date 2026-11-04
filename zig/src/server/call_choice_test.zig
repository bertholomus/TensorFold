//! Required and named choices reach the native call gate through a split-token DSML template.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const tool_specs = @import("tool_specs.zig");
const Server = @import("server.zig").Server;
const A = std.mem.Allocator;

const DsmlText = struct {
    const marker = "｜DSML｜";

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: A, input: []const u8, _: bool) model_text.Error![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        var at: usize = 0;
        while (at < input.len) {
            if (std.mem.startsWith(u8, input[at..], marker)) {
                try ids.append(a, 1000);
                at += marker.len;
            } else {
                try ids.append(a, input[at]);
                at += 1;
            }
        }
        return ids.toOwnedSlice(a);
    }

    fn decode(_: *anyopaque, a: A, ids: []const u32) A.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| {
            if (id == 1000) try out.appendSlice(a, marker) else try out.append(a, @intCast(id));
        }
        return out.toOwnedSlice(a);
    }

    fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
        return if (std.mem.eql(u8, piece, marker)) 1000 else null;
    }

    fn tokenString(ctx: *anyopaque, a: A, id: u32) A.Error![]u8 {
        return decode(ctx, a, &.{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 1001;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: A, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"tfprobe_fn\"></｜DSML｜ invoke></｜DSML｜ calls>");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

fn preflight(srv: *Server, cx: *errors.Cx, choice: []const u8) !api.Request {
    const tools = (try json.parse(cx.a, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}},{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}}]")).ok;
    const selected = (try json.parse(cx.a, choice)).ok;
    const active = try tool_specs.active(cx, tools, selected);
    const fields = try json.newObject(cx.a);
    try fields.put(cx.a, "tool_call_required", .{ .bool = try tool_specs.choiceRequiresCall(cx.a, selected) });
    var request: api.Request = .{ .prompt = &.{}, .max_tokens = 128 };
    try srv.checkFeatures(cx, .{ .object = fields }, active.len > 0, false, &.{}, active, &request);
    return request;
}

test "DSML required and named preflight accepts a split opener and the active names" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: DsmlText = .{};
    var srv: Server = .{
        .gpa = a,
        .io = std.testing.io,
        .engine = undefined,
        .info = .{ .call_gates = true },
        .text = text.text(),
        .config = .{ .served_name = "test", .model_ids = &.{"test"} },
        .keys = null,
        .metrics = .{ .gpa = a },
        .store = .{ .gpa = a },
        .arena = .init(a),
        .slide = .init(a, std.testing.io, null),
        .teacher = .init(a),
    };
    var cx: errors.Cx = .{ .a = a };
    const required = try preflight(&srv, &cx, "\"required\"");
    try std.testing.expectEqual(@as(u32, 1000), required.call.?.opener);
    try std.testing.expectEqual(@as(usize, 2), required.call.?.names.len);
    const named = try preflight(&srv, &cx, "{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}}");
    try std.testing.expectEqual(@as(usize, 1), named.call.?.names.len);
    try std.testing.expectEqualStrings("get_time", named.call.?.names[0]);
    try std.testing.expect((try preflight(&srv, &cx, "\"auto\"")).call == null);
    try std.testing.expect((try preflight(&srv, &cx, "\"none\"")).call == null);
    srv.info.call_gates = false;
    try std.testing.expectError(error.Refused, preflight(&srv, &cx, "\"required\""));
    try std.testing.expectError(error.Refused, preflight(&srv, &cx, "{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}}"));
    try std.testing.expect((try preflight(&srv, &cx, "\"auto\"")).call == null);
}
