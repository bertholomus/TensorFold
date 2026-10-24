const std = @import("std");
const cg = @import("call_gate.zig");
const expect = std.testing.expect;
const eq = std.testing.expectEqualSlices;

pub const Codec = struct {
    pub fn encode(_: *anyopaque, a: std.mem.Allocator, bytes: []const u8) ![]const u32 {
        const result = try a.alloc(u32, bytes.len);
        for (bytes, result) |c, *t| t.* = c;
        return result;
    }
    pub fn text(_: *anyopaque, a: std.mem.Allocator, token: u32) ![]const u8 {
        if (token == 1000) return "｜DSML｜";
        if (token == 1001 or token == 1002) return "think";
        if (token == 1003) return spec.lead ++ "get_x";
        if (token == 1004) return "get_time\">";
        if (token == 1005) return "get_time unexpected\"";
        if (token == 1006) return "< ";
        if (token == 1007) return " \t<";
        const result = try a.alloc(u8, 1);
        result[0] = @intCast(token);
        return result;
    }
    pub fn codec(ptr: *anyopaque) cg.Codec {
        return .{ .ptr = ptr, .encode = encode, .text = text };
    }
};

pub const spec: cg.Spec = .{
    .head = "<",
    .opener = 1000,
    .lead = " calls>\n<｜DSML｜ invoke name=\"",
    .tail = "\"",
    .names = &.{ "get_weather", "get_time" },
    .think_open = 1001,
    .think_end = 1002,
};

fn ids(a: std.mem.Allocator, text: []const u8) ![]const u32 {
    var dummy: u8 = 0;
    return Codec.encode(&dummy, a, text);
}
fn opening(a: std.mem.Allocator) ![]const u32 {
    return std.mem.concat(a, u32, &.{ &.{ '<', 1000 }, try ids(a, spec.lead) });
}
fn observe(g: *cg.Gate, tokens: []const u32) !void {
    const hook = g.hook();
    for (tokens) |t| try hook.observe(hook.ptr, t);
}

test "DSML head and opener are enforced without mutating proposed state" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    var g = cg.Gate.init(a, spec, Codec.codec(&dummy), &.{});
    const h = g.hook();
    const bad = try ids(a, "Sure");
    const first = (try h.cut(h.ptr, a, bad)).?;
    try expect(first.at == 0);
    try eq(u32, try opening(a), first.fix);
    const again = (try h.cut(h.ptr, a, bad)).?;
    try eq(u32, first.fix, again.fix);
    try std.testing.expectError(error.UnenforcedCallToken, h.observe(h.ptr, 'S'));
    const whitespace = (try h.cut(h.ptr, a, &.{ ' ', 'S' })).?;
    try expect(whitespace.at == 1);
    const split = (try h.cut(h.ptr, a, &.{ '<', 'x' })).?;
    try expect(split.at == 1);
    try eq(u32, (try opening(a))[1..], split.fix);
    const bare = (try h.cut(h.ptr, a, &.{1000})).?;
    try eq(u32, try opening(a), bare.fix);
    try observe(&g, first.fix);
    try observe(&g, try ids(a, "get_time\""));
    try expect(try h.cut(h.ptr, a, bad) == null);
}

test "required and named choices correct only the first divergent name token" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    for ([_][]const []const u8{ &.{ "get_weather", "get_time" }, &.{"get_time"} }) |names| {
        var s = spec;
        s.names = names;
        var g = cg.Gate.init(a, s, Codec.codec(&dummy), &.{});
        try observe(&g, try opening(a));
        const h = g.hook();
        const hit = (try h.cut(h.ptr, a, try ids(a, "get_x"))).?;
        try expect(hit.at == 4);
        try eq(u32, try ids(a, try std.mem.concat(a, u8, &.{ names[0][4..], "\"" })), hit.fix);
        try observe(&g, try ids(a, "get_"));
        try observe(&g, hit.fix);
        try expect(try h.cut(h.ptr, a, try ids(a, ">{arbitrary arguments}")) == null);
    }
}

test "thinking is unconstrained until closed and each request owns its state" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    var thinking = cg.Gate.init(a, spec, Codec.codec(&dummy), &.{1001});
    var answer = cg.Gate.init(a, spec, Codec.codec(&dummy), &.{ 1001, 1002 });
    const h = thinking.hook();
    const other = answer.hook();
    const prose = try ids(a, "Sure");
    try expect(try h.cut(h.ptr, a, prose) == null);
    try expect(try other.cut(other.ptr, a, prose) != null);
    const hit = (try h.cut(h.ptr, a, &.{ 'S', 1002, 'x' })).?;
    try expect(hit.at == 2);
    try observe(&thinking, &.{ 'S', 1002 });
    try observe(&thinking, hit.fix);
    try observe(&thinking, try ids(a, "get_weather\""));
}

test "single-token call opener retains its existing behavior" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    var g = cg.Gate.init(a, .{ .opener = 1000 }, Codec.codec(&dummy), &.{});
    const h = g.hook();
    const hit = (try h.cut(h.ptr, a, &.{'x'})).?;
    try eq(u32, &.{1000}, hit.fix);
    try observe(&g, hit.fix);
    try expect(try h.cut(h.ptr, a, &.{'x'}) == null);
}

test "a merged token spanning the lead and name corrects the whole uncommitted token" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    var g = cg.Gate.init(a, spec, Codec.codec(&dummy), &.{});
    try observe(&g, &.{ '<', 1000 });
    const h = g.hook();
    const hit = (try h.cut(h.ptr, a, &.{1003})).?;
    try expect(hit.at == 0);
    try eq(u32, try ids(a, spec.lead ++ "get_weather\""), hit.fix);
    try observe(&g, hit.fix);
    try expect(try h.cut(h.ptr, a, &.{'x'}) == null);

    var named = spec;
    named.names = &.{"get_time"};
    var other = cg.Gate.init(a, named, Codec.codec(&dummy), &.{});
    try observe(&other, try opening(a));
    try observe(&other, &.{1004});
    const named_hook = other.hook();
    try expect(try named_hook.cut(named_hook.ptr, a, &.{'x'}) == null);
}

test "quoted names require their closing quote across split and merged tokens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    var named = spec;
    named.names = &.{"get_time"};
    for ([_][]const []const u8{ spec.names, named.names }) |names| {
        var s = spec;
        s.names = names;
        for ([_]u8{ ' ', '\t', '\r', '\n' }) |space| {
            var g = cg.Gate.init(a, s, Codec.codec(&dummy), &.{});
            try observe(&g, try opening(a));
            try observe(&g, try ids(a, "get_time"));
            const h = g.hook();
            const hit = (try h.cut(h.ptr, a, &.{space})).?;
            try expect(hit.at == 0);
            try eq(u32, try ids(a, "\""), hit.fix);
            try std.testing.expectError(error.UnenforcedCallToken, h.observe(h.ptr, space));
            try observe(&g, hit.fix);
            try expect(try h.cut(h.ptr, a, try ids(a, ">{arguments}")) == null);
        }
        var g = cg.Gate.init(a, s, Codec.codec(&dummy), &.{});
        try observe(&g, try opening(a));
        const h = g.hook();
        const hit = (try h.cut(h.ptr, a, &.{1005})).?;
        try expect(hit.at == 0);
        try eq(u32, try ids(a, try std.mem.concat(a, u8, &.{ names[0], "\"" })), hit.fix);
        try observe(&g, hit.fix);
    }
}

test "DSML head accepts leading whitespace but rejects trailing whitespace" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    var g = cg.Gate.init(a, spec, Codec.codec(&dummy), &.{});
    const h = g.hook();
    const hit = (try h.cut(h.ptr, a, &.{ 1006, 1000 })).?;
    try expect(hit.at == 0);
    try eq(u32, try opening(a), hit.fix);
    try std.testing.expectError(error.UnenforcedCallToken, h.observe(h.ptr, 1006));
    try expect(try h.cut(h.ptr, a, &.{ 1007, 1000 }) == null);
    try observe(&g, &.{ 1007, 1000 });
    try observe(&g, try ids(a, spec.lead ++ "get_time\""));
}
