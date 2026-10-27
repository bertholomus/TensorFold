//! Byte tries for a schema program (its literals and each object's keys), built without recursion.
const std = @import("std");
const schema = @import("schema.zig");
const Allocator = std.mem.Allocator;

/// Each node's texts by index, in order, for a trie of keys.
pub const Below = struct { spans: std.ArrayList(schema.Span) = .empty, members: std.ArrayList(u16) = .empty };

/// The lists a program's tries live in.
pub const Lists = struct { a: Allocator, tries: *std.ArrayList(schema.Trie), edges: *std.ArrayList(schema.Edge) };

/// A trie of `texts` appended to the lists, and its root; with `below`, each node records the texts under it.
pub fn build(l: Lists, texts: []const []const u8, below: ?*Below) Allocator.Error!u32 {
    const order = try l.a.alloc(u32, texts.len);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, texts, struct {
        fn less(t: []const []const u8, x: u32, y: u32) bool {
            return std.mem.order(u8, t[x], t[y]) == .lt;
        }
    }.less);
    const Todo = struct { node: u32, lo: usize, hi: usize, depth: usize };
    var todo: std.ArrayList(Todo) = .empty;
    const root = try node(l, order[0..], below);
    try todo.append(l.a, .{ .node = root, .lo = 0, .hi = order.len, .depth = 0 });
    while (todo.pop()) |span| {
        var i = span.lo;
        // sorted, so the text that ends here (and its repeats) comes first
        while (i < span.hi and texts[order[i]].len == span.depth) : (i += 1) l.tries.items[span.node].end = true;
        const first: u32 = @intCast(l.edges.items.len);
        var j = i;
        while (j < span.hi) {
            const b = texts[order[j]][span.depth];
            var e = j;
            while (e < span.hi and texts[order[e]][span.depth] == b) e += 1;
            const to = try node(l, order[j..e], below);
            try l.edges.append(l.a, .{ .byte = b, .to = to });
            try todo.append(l.a, .{ .node = to, .lo = j, .hi = e, .depth = span.depth + 1 });
            j = e;
        }
        l.tries.items[span.node].first = first;
        l.tries.items[span.node].len = @intCast(l.edges.items.len - first);
    }
    return root;
}

/// A new trie node for the texts `order` names; with `below`, it records them in index order.
fn node(l: Lists, order: []const u32, below: ?*Below) Allocator.Error!u32 {
    const id: u32 = @intCast(l.tries.items.len);
    try l.tries.append(l.a, .{ .first = 0, .len = 0, .end = false });
    if (below) |w| {
        const first: u32 = @intCast(w.members.items.len);
        for (order) |i| try w.members.append(l.a, @intCast(i));
        std.mem.sort(u16, w.members.items[first..], {}, std.sort.asc(u16));
        try w.spans.append(l.a, .{ .first = first, .len = @intCast(order.len) });
    }
    return id;
}
