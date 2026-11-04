//! Required tool-call prefixes, ported from Elevatormusic/TensorFold's call_gate.py (Apache-2.0; see ATTRIBUTION.md).
const std = @import("std");
const A = std.mem.Allocator;

pub const Cut = struct { at: usize, fix: []const u32 };
pub const Hook = struct {
    ptr: *anyopaque,
    cut: *const fn (*anyopaque, A, []const u32) anyerror!?Cut,
    observe: *const fn (*anyopaque, u32) anyerror!void,

    pub fn feed(h: Hook, a: A, gpa: A, force: *std.ArrayList(u32), backend: anytype, input: anytype) !@TypeOf(input) {
        const token = switch (input) {
            .handle => |handle| try backend.read(handle),
            .value => |value| value,
        };
        return .{ .value = try h.enforce(a, gpa, force, token) };
    }

    pub fn enforce(h: Hook, a: A, gpa: A, force: *std.ArrayList(u32), token: u32) !u32 {
        if (try h.cut(h.ptr, a, &.{token})) |hit| {
            force.clearRetainingCapacity();
            try force.appendSlice(gpa, hit.fix[1..]);
            return hit.fix[0];
        }
        return token;
    }
};
pub const Spec = struct {
    opener: u32,
    head: []const u8 = "",
    lead: []const u8 = "",
    tail: []const u8 = "",
    names: []const []const u8 = &.{},
    think_open: ?u32 = null,
    think_end: ?u32 = null,
    filter: ?Hook = null,
};
pub const Codec = struct {
    ptr: *anyopaque,
    encode: *const fn (*anyopaque, A, []const u8) anyerror![]const u32,
    text: *const fn (*anyopaque, A, u32) anyerror![]const u8,
};
const Phase = enum { answer, open, lead, name, done };
const State = struct { armed: bool = true, phase: Phase = .answer, seen: []const u8 = "" };
const Step = struct { state: State, fix: ?[]const u32 = null };

pub const Gate = struct {
    a: A,
    spec: Spec,
    codec: Codec,
    state: State = .{},

    pub fn init(a: A, spec: Spec, codec: Codec, prompt: []const u32) Gate {
        var g: Gate = .{ .a = a, .spec = spec, .codec = codec };
        for (prompt) |t| {
            if (spec.think_open != null and t == spec.think_open.?) g.state.armed = false;
            if (spec.think_end != null and t == spec.think_end.?) g.state.armed = true;
        }
        return g;
    }

    pub fn hook(g: *Gate) Hook {
        return .{ .ptr = g, .cut = cutFn, .observe = observeFn };
    }

    fn encode(g: *const Gate, a: A, text: []const u8) ![]const u32 {
        return g.codec.encode(g.codec.ptr, a, text);
    }

    fn opening(g: *const Gate, a: A, head: bool) ![]const u32 {
        const h = if (head) try g.encode(a, g.spec.head) else &.{};
        const lead = if (g.spec.names.len > 0) try g.encode(a, g.spec.lead) else &.{};
        return std.mem.concat(a, u32, &.{ h, &.{g.spec.opener}, lead });
    }

    fn ends(g: *const Gate, c: u8) bool {
        if (g.spec.tail.len == 0 and std.ascii.isWhitespace(c)) return true;
        return std.mem.indexOfScalar(u8, if (g.spec.tail.len > 0) g.spec.tail else "<>{}\"'(),", c) != null;
    }

    fn step(g: *const Gate, a: A, before: State, token: u32) !Step {
        var s = before;
        if (s.phase == .done) return .{ .state = s };
        if (s.phase == .answer) {
            if (!s.armed) {
                s.armed = g.spec.think_end != null and token == g.spec.think_end.?;
                return .{ .state = s };
            }
            if (g.spec.think_open != null and token == g.spec.think_open.?) {
                s.armed = false;
                return .{ .state = s };
            }
            if (token == g.spec.opener and g.spec.head.len == 0) {
                s.phase = .lead;
                return .{ .state = s };
            }
            const text = try g.codec.text(g.codec.ptr, a, token);
            const trimmed = std.mem.trimStart(u8, text, " \t\r\n\x0b\x0c");
            if (trimmed.len == 0) return .{ .state = s };
            if (g.spec.head.len > 0 and std.mem.eql(u8, trimmed, g.spec.head)) {
                s.phase = .open;
                return .{ .state = s };
            }
            return .{ .state = s, .fix = try g.opening(a, true) };
        }
        if (s.phase == .open) {
            if (token != g.spec.opener) return .{ .state = s, .fix = try g.opening(a, false) };
            s.phase = .lead;
            return .{ .state = s };
        }
        if (g.spec.names.len == 0) return .{ .state = .{ .phase = .done } };
        const piece = try g.codec.text(g.codec.ptr, a, token);
        var written = try std.mem.concat(a, u8, &.{ s.seen, piece });
        var owed: []const u8 = "";
        if (s.phase == .lead) {
            if (std.mem.startsWith(u8, g.spec.lead, written)) {
                s.phase = if (written.len == g.spec.lead.len) .name else .lead;
                s.seen = if (s.phase == .name) "" else written;
                return .{ .state = s };
            }
            if (!std.mem.startsWith(u8, written, g.spec.lead))
                return .{ .state = s, .fix = try g.encode(a, g.spec.lead[s.seen.len..]) };
            owed = g.spec.lead[s.seen.len..];
            s.seen = "";
            written = written[g.spec.lead.len..];
        }
        var end: usize = 0;
        while (end < written.len and !g.ends(written[end])) : (end += 1) {}
        for (g.spec.names) |name| {
            if (end == written.len and std.mem.startsWith(u8, name, written))
                return .{ .state = .{ .armed = s.armed, .phase = .name, .seen = written } };
            if (end < written.len and std.mem.eql(u8, name, written[0..end]))
                return .{ .state = .{ .phase = .done } };
        }
        for (g.spec.names) |name| if (std.mem.startsWith(u8, name, s.seen)) {
            const rest = try std.mem.concat(a, u8, &.{ owed, name[s.seen.len..], g.spec.tail });
            return .{ .state = before, .fix = try g.encode(a, rest) };
        };
        return error.InvalidCallPrefix;
    }

    fn cutFn(ptr: *anyopaque, a: A, tokens: []const u32) !?Cut {
        const g: *Gate = @ptrCast(@alignCast(ptr));
        var s = g.state;
        for (tokens, 0..) |t, i| {
            const next = try g.step(a, s, t);
            if (next.fix) |fix| {
                if (fix.len == 0) return error.EmptyCallFix;
                return .{ .at = i, .fix = fix };
            }
            s = next.state;
        }
        return null;
    }

    fn observeFn(ptr: *anyopaque, token: u32) !void {
        const g: *Gate = @ptrCast(@alignCast(ptr));
        const next = try g.step(g.a, g.state, token);
        if (next.fix != null) return error.UnenforcedCallToken;
        g.state = next.state;
    }
};
