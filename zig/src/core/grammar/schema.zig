//! A JSON schema or guided_choice list as a program with xgrammar 0.2.8's strict semantics, as Python 0.6 used them.
const std = @import("std");
const formats = @import("formats.zig");
const bounds = @import("bounds.zig");
const bytetrie = @import("bytetrie.zig");
const keys = @import("schema_keys.zig");
const Value = std.json.Value;
const Allocator = std.mem.Allocator;

/// A node's index in Program.nodes.
pub const Id = u16;
/// No node: no other keys, no more items.
pub const none: Id = std.math.maxInt(Id);
/// The most properties, prefix items or items a count may name: the matcher keeps counts in 16 bits.
pub const max_count = none - 1;
/// The most schema levels the compiler follows.
pub const max_nesting = 128;
/// A string length with no bound.
pub const unbounded = std.math.maxInt(u32);
/// The longest schema text or choice list the compiler takes: a program lives on the server's heap with its request.
pub const max_text = 1 << 20;

pub const Node = union(enum) {
    /// Any JSON value.
    any,
    object: u32,
    array: u32,
    /// Any JSON string.
    string,
    /// A string of min to max characters, none of them a quote, a backslash, CR or LF.
    text: struct { min: u32, max: u32 },
    format: formats.Format,
    integer,
    number,
    /// An integer or number with bounds: its automaton's start in Program.states.
    range: u32,
    /// One of a set of byte strings: enum, const, true and false, null, guided_choice.
    literal: u32,
    any_of: u32,
    /// A $ref, until the compiler replaces it by its target.
    alias: Id,
};

/// An object: properties in their order, those not required may be left out, then other keys if allowed.
pub const Object = struct {
    props: []const Prop,
    /// required_from[c]: the first required property at or after property c, props.len for none.
    required_from: []const u16,
    /// The values of other keys; none: no other key.
    additional: Id,
    /// The trie of the keys after their opening quote, so a key's next byte costs no scan of the properties.
    trie: u32,
    /// below[t - trie]: the properties whose keys pass through trie node t, a slice of `members` in order.
    below: []const Span,
    members: []const u16,
};

pub const Span = struct { first: u32, len: u32 };

/// A property: its key as a JSON string, its raw name, and its value's node.
pub const Prop = struct { key: []const u8, name: []const u8, node: Id };

/// An array: prefix items in order (the array may end after any of them), then `items` if allowed.
pub const Array = struct { prefix: []const Id, items: Id, min: u16, max: u16 };

/// anyOf, oneOf or a list of types: the branch each first byte starts.
pub const Union = struct { dispatch: [256]Id };

pub const Trie = struct { first: u32, len: u32, end: bool };
pub const Edge = struct { byte: u8, to: u32 };

pub const Program = struct {
    nodes: []const Node,
    objects: []const Object,
    arrays: []const Array,
    unions: []const Union,
    tries: []const Trie,
    edges: []const Edge,
    /// The automata of bounded numbers.
    states: []const bounds.State,
    root: Id,
    /// The trie of true, false and null, for any value.
    words: u32,

    /// Nodes every program has.
    pub const any: Id = 0;
    pub const any_object: Id = 1;
    pub const any_array: Id = 2;
    const string: Id = 3;
    const integer: Id = 4;
    const number: Id = 5;
    const boolean: Id = 6;
    const @"null": Id = 7;

    /// Trie node `t`'s child by byte `b`. A node's edges are in byte order.
    pub fn child(p: *const Program, t: u32, b: u8) ?u32 {
        const n = p.tries[t];
        var lo = n.first;
        var hi = n.first + n.len;
        if (n.len <= 8) {
            for (p.edges[lo..hi]) |e| if (e.byte == b) return e.to;
            return null;
        }
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const e = p.edges[mid];
            if (e.byte == b) return e.to;
            if (e.byte < b) lo = mid + 1 else hi = mid;
        }
        return null;
    }
};

pub const Error = error{ OutOfMemory, Refused };

/// The program for a schema's text, in `a`; on error.Refused, `why` names what this engine cannot enforce, and where.
pub fn compile(a: Allocator, text: []const u8, why: *[]const u8) Error!Program {
    var c: Compiler = .{ .a = a, .why = why };
    try c.init();
    if (text.len > max_text) return c.fail("the schema is longer than {d} bytes", .{max_text});
    c.top = std.json.parseFromSliceLeaky(Value, a, text, .{ .parse_numbers = false }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.DuplicateField => return c.fail("the schema repeats a key", .{}),
        else => return c.fail("the schema is not JSON this engine reads ({s})", .{@errorName(e)}),
    };
    const root = try c.ref(.{ .string = "#" });
    return c.finish(root);
}

/// The program for guided_choice: exactly one of `options`, byte for byte, then the end of the reply.
pub fn choice(a: Allocator, options: []const []const u8) Allocator.Error!Program {
    var why: []const u8 = "";
    var c: Compiler = .{ .a = a, .why = &why };
    // a choice list has no keyword to refuse, and a few nodes
    return c.options(options) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Refused => unreachable,
    };
}

const Compiler = struct {
    a: Allocator,
    why: *[]const u8,
    top: Value = .null,
    nodes: std.ArrayList(Node) = .empty,
    objects: std.ArrayList(Object) = .empty,
    arrays: std.ArrayList(Array) = .empty,
    branches: std.ArrayList([]const Id) = .empty,
    tries: std.ArrayList(Trie) = .empty,
    edges: std.ArrayList(Edge) = .empty,
    numbers: bounds.Table = .{},
    /// Each literal node's byte strings by its trie, for anyOf to merge.
    sets: std.AutoHashMapUnmanaged(u32, []const []const u8) = .empty,
    refs: std.StringHashMapUnmanaged(Id) = .empty,
    path: std.ArrayList(u8) = .empty,
    depth: u32 = 0,
    words: u32 = 0,

    fn options(c: *Compiler, texts: []const []const u8) Error!Program {
        try c.init();
        return c.finish(try c.literal(texts));
    }

    fn fail(c: *Compiler, comptime fmt: []const u8, args: anytype) Error {
        c.why.* = try std.fmt.allocPrint(c.a, "{s}: " ++ fmt, .{c.path.items} ++ args);
        return error.Refused;
    }

    fn add(c: *Compiler, n: Node) Error!Id {
        if (c.nodes.items.len >= max_count) return c.fail("the schema needs more than {d} nodes", .{max_count});
        try c.nodes.append(c.a, n);
        return @intCast(c.nodes.items.len - 1);
    }

    fn init(c: *Compiler) Error!void {
        try c.path.appendSlice(c.a, "#");
        _ = try c.add(.any);
        _ = try c.add(.{ .object = try c.objectOf(&.{}, &.{0}, Program.any) });
        try c.arrays.append(c.a, .{ .prefix = &.{}, .items = Program.any, .min = 0, .max = none });
        _ = try c.add(.{ .array = 0 });
        _ = try c.add(.string);
        _ = try c.add(.integer);
        _ = try c.add(.number);
        _ = try c.literal(&.{ "true", "false" });
        _ = try c.literal(&.{"null"});
        c.words = try c.trie(&.{ "true", "false", "null" });
    }

    /// Walks into `segment` of the schema for messages; `back` returns.
    fn into(c: *Compiler, comptime fmt: []const u8, args: anytype) Error!usize {
        const mark = c.path.items.len;
        try c.path.print(c.a, fmt, args);
        return mark;
    }

    fn back(c: *Compiler, mark: usize) void {
        c.path.shrinkRetainingCapacity(mark);
    }

    fn sub(c: *Compiler, v: Value, comptime fmt: []const u8, args: anytype) Error!Id {
        const mark = try c.into(fmt, args);
        defer c.back(mark);
        return c.schema(v);
    }

    /// One schema in xgrammar's keyword order: $ref, const, enum, anyOf, oneOf, allOf, type, then untyped ones.
    fn schema(c: *Compiler, v: Value) Error!Id {
        c.depth += 1;
        defer c.depth -= 1;
        if (c.depth > max_nesting) return c.fail("the schema nests deeper than {d} levels", .{max_nesting});
        const s = switch (v) {
            .bool => |b| return if (b) Program.any else c.fail("the schema false accepts no value", .{}),
            .object => |o| o,
            else => return c.fail("a schema must be an object or a boolean", .{}),
        };
        if (keys.refusal(s)) |k| return c.fail("{s}: {s}", .{ k, keys.advice(k) });
        if (bounds.given(s)) |k| if (!bounds.placed(s))
            return c.fail("{s} needs type integer or number, without $ref, const, enum, anyOf, oneOf or allOf: xgrammar ignores it elsewhere", .{k});
        if (s.get("$ref")) |r| return c.ref(r);
        if (s.get("const")) |k| return c.values(&.{k});
        if (s.get("enum")) |e| {
            if (e != .array or e.array.items.len == 0) return c.fail("enum must be a non-empty list", .{});
            return c.values(e.array.items);
        }
        inline for (.{ "anyOf", "oneOf" }) |word| if (s.get(word)) |o| return c.anyOf(o, word);
        if (s.get("allOf")) |o| {
            if (o != .array or o.array.items.len != 1)
                return c.fail("allOf with other than one schema is not enforced by this engine (xgrammar takes any value)", .{});
            return c.sub(o.array.items[0], "/allOf/0", .{});
        }
        if (s.get("type")) |t| switch (t) {
            .string => |name| return c.typed(s, name),
            .array => |types| {
                if (types.items.len == 0) return c.untyped(s);
                var ids: std.ArrayList(Id) = .empty;
                // a repeated type name adds no branch; there are 7 names, so `seen` stays short
                var seen: std.ArrayList([]const u8) = .empty;
                for (types.items) |name| {
                    if (name != .string) return c.fail("type must be a string or a list of strings", .{});
                    const again = for (seen.items) |prev| {
                        if (std.mem.eql(u8, prev, name.string)) break true;
                    } else false;
                    if (again) continue;
                    try ids.append(c.a, try c.typed(s, name.string));
                    try seen.append(c.a, name.string);
                }
                return c.union_(ids.items);
            },
            else => return c.fail("type must be a string or a list of strings", .{}),
        };
        return c.untyped(s);
    }

    fn untyped(c: *Compiler, s: std.json.ObjectMap) Error!Id {
        if (s.contains("properties") or s.contains("additionalProperties")) return c.object(s);
        if (s.contains("items") or s.contains("prefixItems")) return c.array(s);
        // xgrammar takes any value here and drops these
        for ([_][]const u8{ "required", "minItems", "maxItems", "minLength", "maxLength", "format" }) |k|
            if (s.contains(k)) return c.fail("{s} needs a type here: without one xgrammar takes any value", .{k});
        return Program.any;
    }

    fn typed(c: *Compiler, s: std.json.ObjectMap, name: []const u8) Error!Id {
        const eql = std.mem.eql;
        if (eql(u8, name, "string")) return c.string(s);
        if (eql(u8, name, "integer")) return if (bounds.given(s) == null) Program.integer else c.bounded(s, true);
        if (eql(u8, name, "number")) return if (bounds.given(s) == null) Program.number else c.bounded(s, false);
        if (eql(u8, name, "boolean")) return Program.boolean;
        if (eql(u8, name, "null")) return Program.null;
        // a schema with only "type" and annotations is xgrammar's plain object or array: any keys, any items
        if (eql(u8, name, "object")) return if (keys.plain(s)) Program.any_object else c.object(s);
        if (eql(u8, name, "array")) return if (keys.plain(s)) Program.any_array else c.array(s);
        return c.fail("type {s} is not a JSON type", .{name});
    }

    fn bounded(c: *Compiler, s: std.json.ObjectMap, integer: bool) Error!Id {
        var why: []const u8 = "";
        const start = bounds.compile(c.a, &c.numbers, s, integer, &why) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return c.fail("{s}", .{why}),
        };
        return c.add(.{ .range = start });
    }

    /// A non-negative count, as xgrammar reads one: an integer that fits in 64 bits.
    fn count(c: *Compiler, s: std.json.ObjectMap, comptime word: []const u8) Error!?i64 {
        const v = s.get(word) orelse return null;
        return keys.int64(v) orelse return c.fail(word ++ " must be an integer", .{});
    }

    fn string(c: *Compiler, s: std.json.ObjectMap) Error!Id {
        const limit = std.math.maxInt(i32);
        var min: u32 = 0;
        var max: u32 = unbounded;
        if (try c.count(s, "minLength")) |n| {
            if (n < 0 or n > limit) return c.fail("minLength must be from 0 to {d}", .{limit});
            min = @intCast(n);
        }
        if (try c.count(s, "maxLength")) |n| {
            if (n < 0) return c.fail("maxLength must not be negative", .{});
            if (n <= limit) max = @intCast(n);
        }
        if (min > max) return c.fail("minLength {d} is greater than maxLength {d}", .{ min, max });
        if (s.get("format")) |f| {
            if (f != .string) return c.fail("format must be a string", .{});
            if (formats.named(f.string)) |known| {
                if (min != 0 or max != unbounded)
                    return c.fail("format {s} with minLength or maxLength is not enforced: xgrammar drops the lengths", .{f.string});
                return c.add(.{ .format = known });
            }
            for (formats.unsupported) |u| if (std.mem.eql(u8, u, f.string))
                return c.fail("format {s} is not enforced by this engine", .{f.string});
            // xgrammar takes an unknown format as a plain string
        }
        if (min == 0 and max == unbounded) return Program.string;
        return c.add(.{ .text = .{ .min = min, .max = max } });
    }

    fn object(c: *Compiler, s: std.json.ObjectMap) Error!Id {
        var props: std.ArrayList(Prop) = .empty;
        if (s.get("properties")) |pv| {
            if (pv != .object) return c.fail("properties must be an object", .{});
            if (pv.object.count() > max_count) return c.fail("an object has more than {d} properties", .{max_count});
            for (pv.object.keys(), pv.object.values()) |name, v| {
                var key: std.ArrayList(u8) = .empty;
                try keys.writeString(c.a, &key, name);
                if (key.items.len > 60_000) return c.fail("a property name is longer than this engine takes", .{});
                const node = try c.sub(v, "/properties/{s}", .{name});
                try props.append(c.a, .{ .key = key.items, .name = name, .node = node });
            }
        }
        const required = try c.a.alloc(bool, props.items.len);
        @memset(required, false);
        if (s.get("required")) |r| {
            if (r != .array) return c.fail("required must be a list of strings", .{});
            for (r.array.items) |name| {
                if (name != .string) return c.fail("required must be a list of strings", .{});
                const at = for (props.items, 0..) |p, i| {
                    if (std.mem.eql(u8, p.name, name.string)) break i;
                } else return c.fail("required names {s}, which properties does not define", .{name.string});
                required[at] = true;
            }
        }
        var additional = none;
        if (s.get("additionalProperties")) |ap| switch (ap) {
            .bool => |b| additional = if (b) Program.any else none,
            else => additional = try c.sub(ap, "/additionalProperties", .{}),
        };
        if (additional != none) for (props.items) |p| if (!keys.bare(p.name))
            return c.fail("beside additionalProperties a property name must be printable ASCII without a quote or backslash: {s}", .{p.name});
        const from = try c.a.alloc(u16, props.items.len + 1);
        from[props.items.len] = @intCast(props.items.len);
        var i = props.items.len;
        while (i > 0) {
            i -= 1;
            from[i] = if (required[i]) @intCast(i) else from[i + 1];
        }
        return c.add(.{ .object = try c.objectOf(props.items, from, additional) });
    }

    fn objectOf(c: *Compiler, props: []const Prop, from: []const u16, additional: Id) Error!u32 {
        const texts = try c.a.alloc([]const u8, props.len);
        for (props, texts) |p, *t| t.* = p.key[1..];
        var below: bytetrie.Below = .{};
        const root = try c.trieOf(texts, &below);
        try c.objects.append(c.a, .{
            .props = props,
            .required_from = from,
            .additional = additional,
            .trie = root,
            .below = below.spans.items,
            .members = below.members.items,
        });
        return @intCast(c.objects.items.len - 1);
    }

    fn array(c: *Compiler, s: std.json.ObjectMap) Error!Id {
        var prefix: std.ArrayList(Id) = .empty;
        if (s.get("prefixItems")) |pi| {
            if (pi != .array) return c.fail("prefixItems must be a list", .{});
            if (pi.array.items.len > max_count) return c.fail("prefixItems has more than {d} schemas", .{max_count});
            for (pi.array.items, 0..) |item, i| {
                if (item != .object) return c.fail("prefixItems must be a list of schema objects", .{});
                try prefix.append(c.a, try c.sub(item, "/prefixItems/{d}", .{i}));
            }
        }
        var items = none;
        if (s.get("items")) |it| switch (it) {
            .bool => |b| items = if (b) Program.any else none,
            .object => items = try c.sub(it, "/items", .{}),
            else => return c.fail("items must be a boolean or a schema object", .{}),
        };
        const min: i64 = @max(0, try c.count(s, "minItems") orelse 0);
        var max: i64 = -1;
        if (try c.count(s, "maxItems")) |n| {
            if (n < 0) return c.fail("maxItems must not be negative", .{});
            if (n <= std.math.maxInt(i32)) max = n;
        }
        const n: i64 = @intCast(prefix.items.len);
        if (max != -1 and min > max) return c.fail("minItems {d} is greater than maxItems {d}", .{ min, max });
        if (max != -1 and max < n) return c.fail("maxItems {d} is less than the {d} prefixItems", .{ max, n });
        if (items == none and n < min) return c.fail("minItems {d} is more than the {d} prefixItems, and no other item is allowed", .{ min, n });
        if (min > max_count or max > max_count) return c.fail("minItems and maxItems above {d} are too large for this engine", .{max_count});
        if (max == -1) max = none;
        try c.arrays.append(c.a, .{ .prefix = prefix.items, .items = items, .min = @intCast(min), .max = @intCast(max) });
        return c.add(.{ .array = @intCast(c.arrays.items.len - 1) });
    }

    /// enum and const: the values as xgrammar writes them.
    fn values(c: *Compiler, vs: []const Value) Error!Id {
        const texts = try c.a.alloc([]const u8, vs.len);
        for (vs, texts) |v, *t| {
            var out: std.ArrayList(u8) = .empty;
            keys.write(c.a, &out, v) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Unwritable => return c.fail("enum and const take no number with more than 17 significant digits: xgrammar writes it rounded", .{}),
            };
            t.* = out.items;
        }
        return c.literal(texts);
    }

    fn literal(c: *Compiler, texts: []const []const u8) Error!Id {
        const root = try c.trie(texts);
        try c.sets.put(c.a, root, texts);
        return c.add(.{ .literal = root });
    }

    /// A trie of `texts`, built without recursion (a value may be long).
    fn trie(c: *Compiler, texts: []const []const u8) Error!u32 {
        return c.trieOf(texts, null);
    }

    fn trieOf(c: *Compiler, texts: []const []const u8, below: ?*bytetrie.Below) Error!u32 {
        return bytetrie.build(.{ .a = c.a, .tries = &c.tries, .edges = &c.edges }, texts, below);
    }

    fn anyOf(c: *Compiler, o: Value, comptime word: []const u8) Error!Id {
        if (o != .array or o.array.items.len == 0) return c.fail(word ++ " must be a non-empty list", .{});
        const ids = try c.a.alloc(Id, o.array.items.len);
        for (o.array.items, ids, 0..) |item, *id, i| id.* = try c.sub(item, "/" ++ word ++ "/{d}", .{i});
        return c.union_(ids);
    }

    fn union_(c: *Compiler, ids: []const Id) Error!Id {
        try c.branches.append(c.a, ids);
        return c.add(.{ .any_of = @intCast(c.branches.items.len - 1) });
    }

    /// A local $ref ("#" or "#/a/b", parts split at '/' as xgrammar splits them), compiled once.
    fn ref(c: *Compiler, r: Value) Error!Id {
        if (r != .string) return c.fail("$ref must be a string", .{});
        const uri = r.string;
        if (c.refs.get(uri)) |id| return id;
        var target = c.top;
        if (!std.mem.eql(u8, uri, "#")) {
            if (!std.mem.startsWith(u8, uri, "#/")) return c.fail("$ref {s} is not local: this engine follows only # and #/...", .{uri});
            var parts = std.mem.splitScalar(u8, uri[2..], '/');
            while (parts.next()) |part| {
                if (part.len == 0) continue;
                if (target != .object or !target.object.contains(part)) return c.fail("$ref {s} does not resolve", .{uri});
                target = target.object.get(part).?;
            }
        }
        const id = try c.add(.{ .alias = none });
        try c.refs.put(c.a, uri, id);
        const saved = c.path;
        c.path = .empty;
        defer c.path = saved;
        try c.path.appendSlice(c.a, uri);
        const to = try c.schema(target); // before the store: compiling may move c.nodes
        c.nodes.items[id] = .{ .alias = to };
        return id;
    }

    fn finish(c: *Compiler, root: Id) Error!Program {
        c.path.clearRetainingCapacity();
        try c.path.appendSlice(c.a, "#");
        const nodes = c.nodes.items; // nothing adds a node until the unions below
        // each $ref takes its target's node, so the matcher never follows one
        const targets = try c.a.alloc(Id, nodes.len);
        for (nodes, 0..) |_, i| {
            var at: Id = @intCast(i);
            var steps: usize = 0;
            while (nodes[at] == .alias) : (steps += 1) {
                if (steps > nodes.len) return c.fail("a $ref refers only to itself", .{});
                at = nodes[at].alias;
            }
            targets[i] = at;
        }
        for (nodes, targets) |*n, t| n.* = nodes[t];
        const unions = try c.a.alloc(Union, c.branches.items.len);
        for (unions, 0..) |*u, i| u.* = try c.dispatch(@intCast(i));
        return .{
            .nodes = c.nodes.items, // a union may have added a merged literal
            .objects = c.objects.items,
            .arrays = c.arrays.items,
            .unions = unions,
            .tries = c.tries.items,
            .edges = c.edges.items,
            .states = c.numbers.states.items,
            .root = targets[root],
            .words = c.words,
        };
    }

    /// A union's branch for each first byte (nested unions opened, integer in number, literals merged), or a refusal.
    fn dispatch(c: *Compiler, u: u32) Error!Union {
        var all: std.ArrayList(Id) = .empty;
        try c.flatten(u, &all, 0);
        // branches with the same node take the same replies, so one of them stands for all
        var flat: std.ArrayList(Id) = .empty;
        var seen: std.AutoHashMapUnmanaged(Node, void) = .empty;
        for (all.items) |b| if (!(try seen.getOrPut(c.a, c.nodes.items[b])).found_existing) try flat.append(c.a, b);
        var number = false;
        var texts: std.ArrayList([]const u8) = .empty;
        for (flat.items) |b| switch (c.nodes.items[b]) {
            .number => number = true,
            .literal => |t| try texts.appendSlice(c.a, c.sets.get(t).?),
            .any => if (flat.items.len > 1) return c.fail("anyOf has a branch that takes any value beside others: this engine needs branches that start with different bytes", .{}),
            else => {},
        };
        var kept: std.ArrayList(Id) = .empty;
        // integer and bounded replies are numbers too; without number, two or more become one automaton
        var ints: std.ArrayList(Id) = .empty;
        for (flat.items) |b| switch (c.nodes.items[b]) {
            .integer, .range => if (!number) try ints.append(c.a, b),
            .literal => {},
            else => try kept.append(c.a, b),
        };
        if (ints.items.len == 1) try kept.append(c.a, ints.items[0]);
        if (ints.items.len > 1) try kept.append(c.a, try c.united(ints.items));
        if (texts.items.len > 0) try kept.append(c.a, try c.literal(texts.items));
        var out: Union = .{ .dispatch = @splat(none) };
        for (kept.items) |b| {
            const starts = c.firstBytes(b);
            for (starts, 0..) |yes, byte| if (yes) {
                if (out.dispatch[byte] != none)
                    return c.fail("two anyOf, oneOf or type branches can start with '{c}': this engine needs branches that start with different bytes", .{@as(u8, @intCast(byte))});
                out.dispatch[byte] = b;
            };
        }
        return out;
    }

    /// One automaton for integer and bounded branches: a reply any of them takes.
    fn united(c: *Compiler, ids: []const Id) Error!Id {
        const starts = try c.a.alloc(?u32, ids.len);
        for (ids, starts) |b, *s| s.* = if (c.nodes.items[b] == .range) c.nodes.items[b].range else null;
        const start = bounds.unite(c.a, &c.numbers, starts) catch |e| return if (e == error.Refused) c.fail("{s}", .{bounds.too_many}) else e;
        return c.add(.{ .range = start });
    }

    fn flatten(c: *Compiler, u: u32, out: *std.ArrayList(Id), depth: usize) Error!void {
        if (depth > c.branches.items.len) return c.fail("anyOf contains itself", .{});
        for (c.branches.items[u]) |b| switch (c.nodes.items[b]) {
            .any_of => |v| try c.flatten(v, out, depth + 1),
            else => try out.append(c.a, b),
        };
    }

    fn firstBytes(c: *Compiler, id: Id) [256]bool {
        var set: [256]bool = @splat(false);
        const number = "-0123456789";
        switch (c.nodes.items[id]) {
            .any => for ("{[\"tfn" ++ number) |b| {
                set[b] = true;
            },
            .object => set['{'] = true,
            .array => set['['] = true,
            .string, .text, .format => set['"'] = true,
            .integer, .number => for (number) |b| {
                set[b] = true;
            },
            .range => |r| for (number) |b| {
                set[b] = bounds.step(c.numbers.states.items, r, b) != null;
            },
            .literal => |t| {
                const n = c.tries.items[t];
                for (c.edges.items[n.first..][0..n.len]) |e| set[e.byte] = true;
            },
            .any_of, .alias => unreachable,
        }
        return set;
    }
};
