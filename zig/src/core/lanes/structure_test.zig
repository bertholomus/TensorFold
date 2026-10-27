//! Structured output on the fake target: masked replies equal drafted, plain and shared, and a refusal ends one stream.
const std = @import("std");
const grammar = @import("grammar");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const structure = @import("structure.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

const eos = fake.json_tokens.eos;
const think_end = fake.json_tokens.think_end;
const score = fake.json_tokens.score;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 60,
    sampling: ?Sampling = null,
    drafts: bool = true,
    structured: bool = true,
    program: ?*const grammar.Program = null,
    after: ?u32 = null,
    think_budget: u32 = 0,
    think_close: []const u32 = &.{ 90, think_end },
};

const Out = struct { emitted: []u32, reason: sm.Reason, message: []const u8, accepted: u64 };

const Run = struct {
    outs: []Out,
    redraws: usize,

    fn deinit(r: Run) void {
        for (r.outs) |o| gpa.free(o.emitted);
        gpa.free(r.outs);
    }
};

fn model(pipelined: bool) !Config {
    var costs: [8]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 6, .gpu_tokens = pipelined, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 16, .max_streams = 4, .draft_streams = true }, 6, 5);
}

/// Every case's stream admitted together and stepped until all finish, on a scored fake target.
fn run(vocab: *const grammar.Vocab, cases: []const Case) !Run {
    var cfg = try model(false);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .scored = .{ .score = score } };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        const spec: ?structure.Spec = if (c.structured) .{ .vocab = vocab, .program = c.program, .after = c.after } else null;
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{eos}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = c.think_close, .think_end = think_end, .structure = spec });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const outs = try gpa.alloc(Out, cases.len);
    for (outs, streams) |*o, *s| o.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .message = s.message, .accepted = s.accepted };
    return .{ .outs = outs, .redraws = target.redraws };
}

fn table() !grammar.Vocab {
    return fake.json_tokens.vocab(gpa);
}

/// Whether the grammar takes `tokens` from `m` on (an end token only once the object is complete).
fn walks(vocab: *const grammar.Vocab, tokens: []const u32, m: *grammar.Matcher) bool {
    for (tokens) |t| {
        if (t == eos) return m.complete();
        if (!vocab.allows(m, t)) return false;
        _ = m.advance(vocab.piece(t));
    }
    return true;
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };
const p3 = [_]u32{ 1, 4, 1, 4, 2, 1, 3, 5, 6 };
const sampled: Sampling = .{ .seed = 9, .temperature = 0.8, .top_k = 0, .top_p = 0.9 };

test "a json_object reply is the same drafted, plain and shared, greedy and sampled" {
    var vocab = try table();
    defer vocab.deinit(gpa);
    var completed: usize = 0;
    for ([_]?Sampling{ null, sampled }) |s| {
        for ([_][]const u32{ &p1, &p2, &p3 }) |p| {
            const drafted = try run(&vocab, &.{.{ .prompt = p, .sampling = s }});
            defer drafted.deinit();
            const plain = try run(&vocab, &.{.{ .prompt = p, .sampling = s, .drafts = false }});
            defer plain.deinit();
            const shared = try run(&vocab, &.{ .{ .prompt = &p2, .structured = false }, .{ .prompt = p, .sampling = s }, .{ .prompt = &p3, .sampling = sampled } });
            defer shared.deinit();
            try std.testing.expectEqualSlices(u32, plain.outs[0].emitted, drafted.outs[0].emitted);
            try std.testing.expectEqualSlices(u32, plain.outs[0].emitted, shared.outs[1].emitted);
            try std.testing.expect(drafted.redraws > 0 and drafted.outs[0].accepted > 0);
            var m: grammar.Matcher = .init(null);
            try std.testing.expect(walks(&vocab, drafted.outs[0].emitted, &m));
            const reply = drafted.outs[0].emitted;
            if (reply[reply.len - 1] == eos) {
                completed += 1;
                try std.testing.expectEqual(sm.Reason.stop, drafted.outs[0].reason);
                var bytes: std.ArrayList(u8) = .empty;
                defer bytes.deinit(gpa);
                for (reply[0 .. reply.len - 1]) |t| try bytes.appendSlice(gpa, vocab.piece(t));
                const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes.items, .{});
                defer parsed.deinit();
                try std.testing.expect(parsed.value == .object);
            } else try std.testing.expectEqual(sm.Reason.length, drafted.outs[0].reason);
            // the stream beside it, which has no grammar, commits what it commits alone
            const alone = try run(&vocab, &.{.{ .prompt = &p2, .structured = false }});
            defer alone.deinit();
            try std.testing.expectEqualSlices(u32, alone.outs[0].emitted, shared.outs[0].emitted);
        }
    }
    try std.testing.expect(completed >= 2);
}

/// A schema and a choice list the fake's tokens can spell: "{\"", "a", "\":", digits, ", \"", "b", true, false...
const object_schema =
    \\{"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}, "x": {"type": "array",
    \\ "items": {"enum": [null, 2]}, "maxItems": 2}}, "required": ["a"], "additionalProperties": false}
;
const choices = [_][]const u8{ "null", "ax", "json", "a" };

test "a json_schema reply and a choice reply are the same drafted, plain and shared, greedy and sampled" {
    var vocab = try table();
    defer vocab.deinit(gpa);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var why: []const u8 = "";
    const programs = [_]grammar.Program{
        try grammar.schema.compile(arena.allocator(), object_schema, &why),
        try grammar.schema.choice(arena.allocator(), &choices),
    };
    var completed = [_]usize{ 0, 0 };
    for (&programs, 0..) |*program, k| for ([_]?Sampling{ null, sampled }) |s| for ([_][]const u32{ &p1, &p2, &p3 }) |p| {
        const case: Case = .{ .prompt = p, .sampling = s, .program = program };
        var plain_case = case;
        plain_case.drafts = false;
        const drafted = try run(&vocab, &.{case});
        defer drafted.deinit();
        const plain = try run(&vocab, &.{plain_case});
        defer plain.deinit();
        const shared = try run(&vocab, &.{ .{ .prompt = &p2 }, case, .{ .prompt = &p3, .sampling = sampled, .structured = false } });
        defer shared.deinit();
        const reply = drafted.outs[0].emitted;
        try std.testing.expectEqualSlices(u32, plain.outs[0].emitted, reply);
        try std.testing.expectEqualSlices(u32, plain.outs[0].emitted, shared.outs[1].emitted);
        var m: grammar.Matcher = .init(program);
        try std.testing.expect(walks(&vocab, reply, &m));
        if (reply[reply.len - 1] != eos) continue;
        completed[k] += 1;
        try std.testing.expectEqual(sm.Reason.stop, drafted.outs[0].reason);
        if (k == 1) {
            var bytes: std.ArrayList(u8) = .empty;
            defer bytes.deinit(gpa);
            for (reply[0 .. reply.len - 1]) |t| try bytes.appendSlice(gpa, vocab.piece(t));
            try std.testing.expect(for (choices) |c| {
                if (std.mem.eql(u8, c, bytes.items)) break true;
            } else false);
        }
    };
    for (completed) |c| try std.testing.expect(c >= 2);
}

test "the first token is drawn again when the grammar refuses the prompt row's pick" {
    var vocab = try table();
    defer vocab.deinit(gpa);
    const sc: fake.Scored = .{ .score = score };
    for ([_][]const u32{ &p1, &p2, &p3 }) |p| {
        const r = try run(&vocab, &.{.{ .prompt = p }});
        defer r.deinit();
        const free = sc.best(p, null, p.len, null).?;
        const first = r.outs[0].emitted[0];
        try std.testing.expect(vocab.piece(first).len > 0 and vocab.piece(first)[0] == '{');
        if (free != first) try std.testing.expect(!vocab.allows(grammar.json.Matcher{}, free));
    }
}

test "under the thinking budget's close the grammar starts after </think>, its next token masked" {
    var vocab = try table();
    defer vocab.deinit(gpa);
    const sc: fake.Scored = .{ .score = score };
    var masked: usize = 0;
    for ([_][]const u32{ &p1, &p2, &p3 }) |p| {
        const case: Case = .{ .prompt = p, .after = think_end, .think_budget = 9 };
        const drafted = try run(&vocab, &.{case});
        defer drafted.deinit();
        var plain_case = case;
        plain_case.drafts = false;
        const plain = try run(&vocab, &.{plain_case});
        defer plain.deinit();
        const shared = try run(&vocab, &.{ .{ .prompt = &p3 }, case });
        defer shared.deinit();
        const reply = drafted.outs[0].emitted;
        try std.testing.expectEqualSlices(u32, plain.outs[0].emitted, reply);
        try std.testing.expectEqualSlices(u32, plain.outs[0].emitted, shared.outs[1].emitted);
        try std.testing.expectEqualSlices(u32, &.{ 90, think_end }, reply[8..10]);
        var m: grammar.Matcher = .init(null);
        try std.testing.expect(walks(&vocab, reply[10..], &m));
        try std.testing.expect(vocab.piece(reply[10])[0] == '{');
        // the forced window's last row drew this token: the target's own pick there is one the grammar refuses
        const history = try std.mem.concat(gpa, u32, &.{ p, reply[0..10] });
        defer gpa.free(history);
        const free = sc.best(history, null, history.len, null).?;
        if (free != reply[10]) masked += 1;
    }
    try std.testing.expect(masked > 0);
}

test "a forced token the grammar refuses fails only its stream" {
    var vocab = try table();
    defer vocab.deinit(gpa);
    // the close's trailing blank line before the object: the root admits no blank before it
    const bad: Case = .{ .prompt = &p1, .after = think_end, .think_budget = 9, .think_close = &.{ 90, think_end, 92 } };
    const together = try run(&vocab, &.{ bad, .{ .prompt = &p2 } });
    defer together.deinit();
    const alone = try run(&vocab, &.{.{ .prompt = &p2 }});
    defer alone.deinit();
    try std.testing.expectEqual(sm.Reason.@"error", together.outs[0].reason);
    try std.testing.expectEqualStrings(structure.refused, together.outs[0].message);
    try std.testing.expectEqual(@as(u32, think_end), together.outs[0].emitted[together.outs[0].emitted.len - 1]);
    try std.testing.expectEqualSlices(u32, alone.outs[0].emitted, together.outs[1].emitted);
}

test "an engine whose rounds run ahead, or whose backend cannot redraw, refuses a structured stream" {
    var vocab = try table();
    defer vocab.deinit(gpa);
    for ([_]bool{ true, false }) |pipelined| {
        var cfg = try model(pipelined);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa, .scored = if (pipelined) .{ .score = score } else null };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &p1, .max_new = 8, .structure = .{ .vocab = &vocab } });
        defer s.deinit(gpa);
        try std.testing.expect(!structure.served(target.backend(), &cfg));
        try std.testing.expectError(error.StructureUnsupported, engine.addStream(&s));
    }
}
