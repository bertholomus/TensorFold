//! The cache-observability contract: a family's shared position pool as four disjoint categories, and a prompt-state
//! store's storage and counters. Optional everywhere. An absent group means unsupported, a disabled store says so
//! instead of reporting healthy zeroes, and an inconsistent account is exported as such rather than clamped.
//!
//! The pool categories come from the allocator's own spans. They are not the per-stream `kv_cache_usage_ratio`
//! (each stream's tokens over that stream's context window), and they are not device or host memory.
const std = @import("std");

/// A half-open position interval of the pool's compressed plane.
pub const Iv = struct { lo: usize, hi: usize };

/// A kept state's allocator span: its rows [base, base + at) and the extent end it also holds back.
pub const KeptIv = struct { base: usize, at: usize, end: usize };

/// The shared pool's disjoint category sums. capacity = active + prefix + growth + unclaimed.
pub const Pool = struct {
    /// The effective allocator limit (the family's pool cap), not a configured or guessed capacity.
    capacity_positions: u64 = 0,
    /// Union of live reservations, including reserved output and alignment space.
    active_reserved_positions: u64 = 0,
    /// Valid, non-stale kept prefixes, excluding the live reservations they overlap.
    retained_prefix_positions: u64 = 0,
    /// Preserved growth tails of kept extents, excluding both categories above: reclaimable without staling.
    retained_growth_positions: u64 = 0,
    /// The remainder of the pool.
    unclaimed_positions: u64 = 0,
    /// Valid family-level kept states (not prompt-store entries, whose several marks may share a prefix).
    valid_states: u64 = 0,
    /// Valid -> stale transitions because a placement reused the rows; each counted once.
    invalidations_total: u64 = 0,
    /// Placement attempts refused for want of a fitting extent (attempts, not unique requests); null: not counted.
    placement_blocks_total: ?u64 = null,
};

/// A prompt-state store's storage and operation counters. `enabled` is false when the store is configured off: the
/// group is then present and empty, distinct from a backend that cannot report one at all (an absent group).
pub const Store = struct {
    enabled: bool = false,
    budget_bytes: u64 = 0,
    held_bytes: u64 = 0,
    spare_bytes: u64 = 0,
    entries: u64 = 0,
    hits_total: u64 = 0,
    misses_total: u64 = 0,
    evictions_total: u64 = 0,
    refusals_total: u64 = 0,
    /// The legacy aggregate (keep, restore and peer-forget failures together) and its classified stages.
    failures_total: u64 = 0,
    failures_keep_total: u64 = 0,
    failures_restore_total: u64 = 0,
    failures_peer_forget_total: u64 = 0,
};

/// /health.live.cache: an additive, optional group. Absent when the engine cannot account for its cache, so old
/// engines and other backends stay valid.
pub const Status = struct {
    schema_version: u16 = 1,
    /// Opaque instance identity of the counter domain: one value per engine start, never a request or a prompt.
    engine_epoch: u64 = 0,
    /// When this copy was produced (wall clock), so a reader can tell a fresh idle sample from a stale one.
    observed_unix_ns: u64 = 0,
    consistent: bool = true,
    /// The enforced per-request limit and the concurrency limit, not independent full-context windows.
    context_limit_positions: u64 = 0,
    stream_capacity: u32 = 0,
    pool: ?Pool = null,
    prompt_state: ?Store = null,
};

/// A backend's cache account for the lane host. Called on the engine thread at stable mutation boundaries, so the copy
/// it reads is coherent: it must not synchronize the GPU, scan the pool, or mutate anything.
pub const Observe = struct {
    ctx: ?*anyopaque = null,
    /// Fills `out` with the current account; false when the accounting is inconsistent and must be exported as such.
    read: *const fn (ctx: ?*anyopaque, out: *Status) bool,
};

fn lessIv(_: void, a: Iv, b: Iv) bool {
    return a.lo < b.lo or (a.lo == b.lo and a.hi < b.hi);
}

/// `list` sorted and merged into `out` (touching intervals joined; empty and inverted ones dropped).
fn mergeIvs(gpa: std.mem.Allocator, list: []Iv, out: *std.ArrayList(Iv)) !void {
    std.mem.sort(Iv, list, {}, lessIv);
    for (list) |iv| {
        if (iv.hi <= iv.lo) continue;
        if (out.items.len > 0) {
            const last = &out.items[out.items.len - 1];
            if (iv.lo <= last.hi) {
                if (iv.hi > last.hi) last.hi = iv.hi;
                continue;
            }
        }
        try out.append(gpa, iv);
    }
}

/// `src` minus `cut` (both sorted, merged) into `out`: the pieces of `src` the intervals in `cut` do not cover.
fn subtractIvs(gpa: std.mem.Allocator, src: []const Iv, cut: []const Iv, out: *std.ArrayList(Iv)) !void {
    var ci: usize = 0;
    for (src) |s| {
        var lo = s.lo;
        const hi = s.hi;
        while (ci < cut.len and cut[ci].hi <= lo) ci += 1;
        var k = ci;
        while (k < cut.len and cut[k].lo < hi) : (k += 1) {
            if (cut[k].lo > lo) try out.append(gpa, .{ .lo = lo, .hi = cut[k].lo });
            if (cut[k].hi > lo) lo = cut[k].hi;
            if (lo >= hi) break;
        }
        if (lo < hi) try out.append(gpa, .{ .lo = lo, .hi = hi });
    }
}

fn sumIvs(spans: []const Iv) u64 {
    var n: u64 = 0;
    for (spans) |iv| n += iv.hi - iv.lo;
    return n;
}

/// The four disjoint categories from the allocator's own spans (pure): active reservations win over kept prefixes,
/// which win over growth tails, and the rest is unclaimed. Out-of-capacity spans make it inconsistent.
pub fn accountIvs(gpa: std.mem.Allocator, cap: usize, live: []Iv, kept: []const KeptIv, out: *Pool) !void {
    var active: std.ArrayList(Iv) = .empty;
    defer active.deinit(gpa);
    var pref: std.ArrayList(Iv) = .empty;
    defer pref.deinit(gpa);
    var grow: std.ArrayList(Iv) = .empty;
    defer grow.deinit(gpa);
    var pref_kept: std.ArrayList(Iv) = .empty;
    defer pref_kept.deinit(gpa);
    var grow_kept: std.ArrayList(Iv) = .empty;
    defer grow_kept.deinit(gpa);
    var rest: std.ArrayList(Iv) = .empty;
    defer rest.deinit(gpa);
    for (live) |iv| if (iv.hi > cap) return error.Inconsistent;
    for (kept) |k| if (k.base + k.at > cap) return error.Inconsistent;
    try mergeIvs(gpa, live, &active);
    for (kept) |k| {
        try pref_kept.append(gpa, .{ .lo = k.base, .hi = k.base + k.at });
        if (k.end > k.base + k.at) try grow_kept.append(gpa, .{ .lo = k.base + k.at, .hi = @min(k.end, cap) });
    }
    try mergeIvs(gpa, pref_kept.items, &pref);
    try mergeIvs(gpa, grow_kept.items, &grow);
    try subtractIvs(gpa, pref.items, active.items, &rest);
    pref.clearRetainingCapacity();
    try mergeIvs(gpa, rest.items, &pref);
    rest.clearRetainingCapacity();
    try subtractIvs(gpa, grow.items, active.items, &rest);
    grow.clearRetainingCapacity();
    try subtractIvs(gpa, rest.items, pref.items, &grow);
    const a = sumIvs(active.items);
    const p = sumIvs(pref.items);
    const g = sumIvs(grow.items);
    out.capacity_positions = cap;
    out.active_reserved_positions = a;
    out.retained_prefix_positions = p;
    out.retained_growth_positions = g;
    if (a + p + g > cap) return error.Inconsistent;
    out.unclaimed_positions = cap - a - p - g;
}

/// The four categories as one assertion, for tests: they must sum to the capacity exactly.
pub fn expectCats(out: Pool, a: u64, p: u64, g: u64, u: u64) !void {
    try std.testing.expectEqual(a, out.active_reserved_positions);
    try std.testing.expectEqual(p, out.retained_prefix_positions);
    try std.testing.expectEqual(g, out.retained_growth_positions);
    try std.testing.expectEqual(u, out.unclaimed_positions);
    try std.testing.expectEqual(out.capacity_positions, a + p + g + u);
}

test "the pool's categories partition the allocator's spans without double counting overlapping state" {
    const gpa = std.testing.allocator;
    var out: Pool = .{};
    // an empty pool is all unclaimed, not unavailable
    try accountIvs(gpa, 1000, &.{}, &.{}, &out);
    try expectCats(out, 0, 0, 0, 1000);
    // a live stream resumed in place: its kept prefix is inside the reservation and counts once
    var live = [_]Iv{.{ .lo = 0, .hi = 300 }};
    var kept = [_]KeptIv{ .{ .base = 0, .at = 200, .end = 400 }, .{ .base = 500, .at = 100, .end = 700 } };
    try accountIvs(gpa, 1000, &live, &kept, &out);
    try expectCats(out, 300, 100, 200, 400);
    // two marks of one state share a base: the prefix is counted once, its preserved tail once
    var shared = [_]KeptIv{ .{ .base = 100, .at = 100, .end = 300 }, .{ .base = 100, .at = 50, .end = 300 } };
    try accountIvs(gpa, 1000, &.{}, &shared, &out);
    try expectCats(out, 0, 100, 100, 800);
    // growth behind a live reservation belongs to the reservation, not to growth
    var tail = [_]KeptIv{.{ .base = 200, .at = 100, .end = 600 }};
    var over = [_]Iv{.{ .lo = 300, .hi = 500 }};
    try accountIvs(gpa, 1000, &over, &tail, &out);
    try expectCats(out, 200, 100, 100, 600);
    // a state whose span runs past the allocator's limit is inconsistent, never a clamped healthy percentage
    var beyond = [_]Iv{.{ .lo = 900, .hi = 1200 }};
    try std.testing.expectError(error.Inconsistent, accountIvs(gpa, 1000, &beyond, &.{}, &out));
    var past = [_]KeptIv{.{ .base = 950, .at = 100, .end = 1000 }};
    try std.testing.expectError(error.Inconsistent, accountIvs(gpa, 1000, &.{}, &past, &out));
}
