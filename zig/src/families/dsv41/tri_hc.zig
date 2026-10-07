//! kernels.py's mHC steps of decode windows (hc_pre2, on the posted, deferred and split paths the served build takes)
//! and of prompt chunks (hc_mix_pf, hc_pre_pf), each launched as its Python wrapper launches it.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const aot = cuda.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;
const hc_blocks = @import("tri_basic.zig").hc_blocks;

/// kernels.HC_PF_RB: hc_mix_pf's rows a program (TF_DS_HC_PF_RB is unset in the served build).
pub const hc_pf_rb = 8;

/// kernels.HC_RB: _hc_mix_part's rows a program for windows of 1 to 8 rows.
const hc_rb = [_]usize{ 1, 2, 3, 2, 3, 2, 4, 4 };

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

/// _hc_rb: _hc_mix_part's rows a program (switch "hc_rb" on and TF_DS_HC_RB unset in the served build).
fn hcRb(rows: usize) usize {
    if (rows >= 1 and rows <= hc_rb.len) return hc_rb[rows - 1];
    return if (rows <= 12) 4 else 8;
}

/// hc_defer_on: whether hc_pre2 takes `sink` (switches "hc_defer" and "hc_split" on in the served build).
pub fn hcDeferOn(d: usize) bool {
    return d % 1024 == 0;
}

/// hc_dots_on: whether a posted hc_pre2 with `sink` takes its mixes off the main stream (switch "hc_dots" on, served).
pub fn hcDotsOn(d: usize) bool {
    return hcDeferOn(d);
}

/// hc_rot_on: whether hc_pre2 takes `rot`; never in the served build (switch "hc_rot" off), so hcPre2 takes none.
pub fn hcRotOn(d: usize) bool {
    _ = d;
    return false;
}

/// hc_pf2_ok: whether hc_mix_pf takes streams of width d (a K block within one stream, of whole 128-column tiles).
pub fn hcPf2Ok(d: usize) bool {
    const kb = 4 * d / hc_blocks;
    return (4 * d) % hc_blocks == 0 and hc_blocks % 4 == 0 and kb > 0 and d % kb == 0 and kb % 128 == 0;
}

/// The previous sublayer's post to fuse in (Python's gathered, h_out): partials [world, R, D] fp32, into h_out (not h).
pub const Posted = struct { gathered: u64, world: usize, h_out: u64 };

/// hc_pre2's `sink` (only if hcDeferOn(d)): a side stream and the event forking it; the caller joins it (rounds.py).
pub const Sink = struct { s: cuda.Stream, fork: cuda.Event };

/// sink.wait_stream(current stream): a Tri on the side stream after the work queued so far; no PDL (no _pdl() there).
fn sinkTri(t: Tri, sk: Sink) !Tri {
    if (t.log == null) {
        try sk.fork.record(t.stream);
        try sk.s.wait(sk.fork);
    }
    var st = t;
    st.stream = sk.s;
    st.pdl = false;
    return st;
}

/// t.run with `rows` appended as an i32 argument.
fn runRows(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const, rows: usize) !void {
    var a: [16]aot.Arg = undefined;
    @memcpy(a[0..args.len], args);
    a[args.len] = aot.int("rows", @intCast(rows));
    return t.run(name, grid, a[0 .. args.len + 1], consts);
}

/// The arguments _hc_finish, _hc_finish_u and _hc_finish_s share: the streams x [R, 4, D] bf16 to collapse and norm.
fn finishArgs(x: u64, part: u64, base: u64, scale: u64, pre_in: u64, norm_w: u64, out: u64, pre_out: u64, post: u64, comb: u64, eps: f32, hc_eps: f32) [12]aot.Arg {
    return .{
        p("X", "*bf16", x),           p("PART", "*fp32", part), p("BASE", "*fp32", base), p("SCALE", "*fp32", scale),
        p("PRE_IN", "*fp32", pre_in), p("NW", "*bf16", norm_w), p("OUT", "*bf16", out),   p("PRE_OUT", "*fp32", pre_out),
        p("POST", "*fp32", post),     p("COMB", "*fp32", comb), aot.float("eps", eps),    aot.float("hc_eps", hc_eps),
    };
}

/// _hc_mix_part: the mixes' partial dots of the streams x, or with `posted` of the post of x written to its h_out.
fn mixPart(t: Tri, x: u64, posted: ?Posted, post: u64, comb: u64, fn_w: u64, part: u64, rows: usize, d: usize) !void {
    const rb = hcRb(rows);
    const grid: [3]u32 = .{ tri.cdiv(rows, rb), hc_blocks, 3 };
    const consts = [_]aot.Const{ ci("WIDE", 4 * d), ci("D", d), ci("NB", hc_blocks), ci("SUB", 128), ci("MB", 8), ci("RB", rb), t.pdlConst() };
    // with the post, XPF: switch "hc_xpf" on in the served build
    if (posted) |ps| return runRows(t, "_hc_mix_part", grid, &.{
        p("X", "*bf16", x),       p("XO", "*bf16", ps.h_out), p("G", "*fp32", ps.gathered), aot.int("RS", @intCast(rows * d)),
        p("POST", "*fp32", post), p("COMB", "*fp32", comb),   p("FN", "*fp32", fn_w),       p("PART", "*fp32", part),
    }, &(consts ++ [_]aot.Const{ ci("WORLD", ps.world), ci("POSTED", 1), ci("XPF", 1) }), rows);
    try runRows(t, "_hc_mix_part", grid, &.{
        p("X", "*bf16", x),       p("XO", "*bf16", x),      p("G", "*bf16", x),     aot.int("RS", 0),
        p("POST", "*fp32", post), p("COMB", "*fp32", comb), p("FN", "*fp32", fn_w), p("PART", "*fp32", part),
    }, &(consts ++ [_]aot.Const{ ci("WORLD", 1), ci("POSTED", 0), ci("XPF", 0) }), rows);
}

/// _hc_finish_s (switch "hc_split" on, served): (r, 0) the Sinkhorn unless `deferred`, (r, 1) the collapse + RMSNorm.
fn finishSplit(t: Tri, x: u64, part: u64, base: u64, scale: u64, pre_in: u64, norm_w: u64, out: u64, pre_out: u64, post: u64, comb: u64, eps: f32, hc_eps: f32, iters: usize, deferred: bool, rows: usize, d: usize) !void {
    // no rotated rows (switch "hc_rot" off in the served build): S0..H3 all `out`, NROT 0, RSPLIT 1
    const o = out;
    try t.run("_hc_finish_s", .{ u(rows), if (deferred) 1 else 2, 1 }, &(finishArgs(x, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps, hc_eps) ++ [_]aot.Arg{
        p("S0", "*bf16", o), p("H0", "*bf16", o), p("S1", "*bf16", o), p("H1", "*bf16", o),
        p("S2", "*bf16", o), p("H2", "*bf16", o), p("S3", "*bf16", o), p("H3", "*bf16", o),
    }), &.{ ci("D", d), ci("NB", hc_blocks), ci("ITERS", iters), ci("BLOCK", 1024), ci("NROT", 0), ci("RSPLIT", 1), ci("SINK", @intFromBool(!deferred)), t.pdlConst() });
}

/// _hc_sinkhorn: the finish's Sinkhorn half (pre_out, post, comb from the mixes' partial sums), a program a row.
fn sinkhorn(t: Tri, part: u64, base: u64, scale: u64, pre_out: u64, post: u64, comb: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize) !void {
    try t.run("_hc_sinkhorn", .{ u(rows), 1, 1 }, &.{
        p("PART", "*fp32", part), p("BASE", "*fp32", base), p("SCALE", "*fp32", scale), p("PRE_OUT", "*fp32", pre_out),
        p("POST", "*fp32", post), p("COMB", "*fp32", comb), aot.float("eps", eps),      aot.float("hc_eps", hc_eps),
    }, &.{ ci("D", d), ci("NB", hc_blocks), ci("ITERS", iters) });
}

/// hc_pre2: hc_pre's outputs for a decode window (the same bits), the last post fused in if `posted`; returns the src.
pub fn hcPre2(t: Tri, h: u64, fn_w: u64, scale: u64, base: u64, pre_in: u64, norm_w: u64, eps: f32, hc_eps: f32, iters: usize, out: u64, pre_out: u64, post: u64, comb: u64, part: u64, posted: ?Posted, sink: ?Sink, rows: usize, d: usize) !u64 {
    if (posted) |ps| std.debug.assert(ps.h_out != h);
    if (sink != null and posted != null and hcDotsOn(d)) {
        // hc_dots: the post alone and the finish here, the mixes of the stored streams and the Sinkhorn on the side one
        const ps = posted.?;
        try t.run("_hc_post_only", .{ u(rows), hc_blocks, 1 }, &.{
            p("X", "*bf16", h),       p("XO", "*bf16", ps.h_out), p("G", "*fp32", ps.gathered), aot.int("RS", @intCast(rows * d)),
            p("POST", "*fp32", post), p("COMB", "*fp32", comb),
        }, &.{ ci("WIDE", 4 * d), ci("D", d), ci("KB", 4 * d / hc_blocks), ci("WORLD", ps.world), t.pdlConst() });
        try finishSplit(t, ps.h_out, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps, hc_eps, iters, true, rows, d);
        const st = try sinkTri(t, sink.?);
        try mixPart(st, ps.h_out, null, post, comb, fn_w, part, rows, d);
        try sinkhorn(st, part, base, scale, pre_out, post, comb, eps, hc_eps, iters, rows, d);
        return ps.h_out;
    }
    try mixPart(t, h, posted, post, comb, fn_w, part, rows, d);
    const src = if (posted) |ps| ps.h_out else h;
    if (d % 1024 == 0) {
        // the split finish (switch "hc_split" on in the served build); with `sink` its Sinkhorn half on the side stream
        if (sink) |sk| try sinkhorn(try sinkTri(t, sk), part, base, scale, pre_out, post, comb, eps, hc_eps, iters, rows, d);
        try finishSplit(t, src, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps, hc_eps, iters, sink != null, rows, d);
    } else {
        try t.run("_hc_finish_u", .{ u(rows), 1, 1 }, &finishArgs(src, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps, hc_eps), &.{ ci("D", d), ci("NB", hc_blocks), ci("ITERS", iters), ci("BLOCK", 1024), t.pdlConst() });
    }
    return src;
}

/// hc_mix_pf: a prompt chunk's mixes' partial dots of h, or of the post written to h_out if `posted`; returns the src.
pub fn hcMixPf(t: Tri, h: u64, fn_w: u64, part: u64, posted: ?Posted, post: u64, comb: u64, rows: usize, d: usize) !u64 {
    const rb = @max(1, @min(hc_pf_rb, rows));
    const grid: [3]u32 = .{ tri.cdiv(rows, rb) * hc_blocks, 1, 1 };
    const consts = [_]aot.Const{ ci("WIDE", 4 * d), ci("D", d), ci("NB", hc_blocks), ci("SUB", 128), ci("MB", 8), ci("RB", rb) };
    if (posted) |ps| {
        std.debug.assert(ps.h_out != h);
        try runRows(t, "_hc_mix_pf", grid, &.{
            p("X", "*bf16", h),       p("XO", "*bf16", ps.h_out), p("G", "*fp32", ps.gathered), aot.int("RS", @intCast(rows * d)),
            p("POST", "*fp32", post), p("COMB", "*fp32", comb),   p("FN", "*fp32", fn_w),       p("PART", "*fp32", part),
        }, &(consts ++ [_]aot.Const{ ci("WORLD", ps.world), ci("POSTED", 1) }), rows);
        return ps.h_out;
    }
    // no post: h stands in for XO and G, part for POST and COMB (none of them read)
    try runRows(t, "_hc_mix_pf", grid, &.{
        p("X", "*bf16", h),       p("XO", "*bf16", h),      p("G", "*bf16", h),     aot.int("RS", 0),
        p("POST", "*fp32", part), p("COMB", "*fp32", part), p("FN", "*fp32", fn_w), p("PART", "*fp32", part),
    }, &(consts ++ [_]aot.Const{ ci("WORLD", 1), ci("POSTED", 0) }), rows);
    return h;
}

/// hc_pre_pf: hc_pre's outputs for a prompt chunk (the same bits): hcMixPf, then hc_pre's finish of what it returns.
pub fn hcPrePf(t: Tri, h: u64, fn_w: u64, scale: u64, base: u64, pre_in: u64, norm_w: u64, eps: f32, hc_eps: f32, iters: usize, out: u64, pre_out: u64, post: u64, comb: u64, part: u64, posted: ?Posted, rows: usize, d: usize) !u64 {
    const src = try hcMixPf(t, h, fn_w, part, posted, post, comb, rows, d);
    try t.run("_hc_finish", .{ u(rows), 1, 1 }, &finishArgs(src, part, base, scale, pre_in, norm_w, out, pre_out, post, comb, eps, hc_eps), &.{ ci("D", d), ci("NB", hc_blocks), ci("ITERS", iters), ci("BLOCK", 1024) });
    return src;
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

/// A side stream for tests: never touched, as a logging Tri launches nothing.
const side: Sink = .{ .s = undefined, .fork = undefined };

/// Stand-ins for the inputs a checked launch does not show (16-aligned, apart from Case.ptr's addresses).
fn stand(i: u64) u64 {
    return 0x7e00_0000_0000 + 0x100 * i;
}

/// A recorded tensor's address, or stand-in `i` when the launch has no such tensor.
fn arg(c: Case, tensor: []const u8, i: u64) u64 {
    return if (c.has(tensor)) c.ptr(tensor) else stand(i);
}

/// The post a recorded mixes launch fuses in: G the gathered partials (fp32 [world, R, D]), XO h_out; null if G is h.
fn postedOf(c: Case) ?Posted {
    const g = c.v.get("tensors").?.object.get("G").?.array.items;
    if (!std.mem.eql(u8, g[0].string, "float32")) return null;
    return .{ .gathered = c.ptr("G"), .world = c.dim("G", 0), .h_out = c.ptr("XO") };
}

/// hcPre2 behind a recorded _hc_finish_s or _hc_sinkhorn launch, with a post or not (X is then h_out, or h).
fn finishCase(comptime posted: bool) *const fn (Tri, Case) anyerror!void {
    return struct {
        fn call(t: Tri, c: Case) !void {
            // the side stream shows only as SINK false (or the Sinkhorn kernel), the width only as D (no X)
            const sink: ?Sink = if (!c.hasConst("SINK") or c.constInt("SINK") == 0) side else null;
            const d: usize = if (c.has("X")) c.dim("X", 2) else @intCast(c.constInt("D"));
            const x = arg(c, "X", 0);
            const ps: ?Posted = if (posted) .{ .gathered = stand(1), .world = 2, .h_out = x } else null;
            const h = if (posted) stand(2) else x;
            _ = try hcPre2(t, h, stand(3), c.ptr("SCALE"), c.ptr("BASE"), arg(c, "PRE_IN", 4), arg(c, "NW", 5), c.float("eps"), c.float("hc_eps"), @intCast(c.constInt("ITERS")), arg(c, "OUT", 6), c.ptr("PRE_OUT"), c.ptr("POST"), c.ptr("COMB"), c.ptr("PART"), ps, sink, c.dim("PRE_OUT", 0), d);
        }
    }.call;
}

test "hc_pre2's mixes launch as recorded: plain, posted, and hc_dots' on the side stream" {
    try tri.conform("_hc_mix_part", struct {
        fn call(t: Tri, c: Case) !void {
            const rows = c.dim("X", 0);
            const d = c.dim("X", 2);
            if (postedOf(c)) |ps| {
                // posted mixes on this stream: hc_pre2 had no side stream (dspark), else hc_dots takes them onto it
                _ = try hcPre2(t, c.ptr("X"), c.ptr("FN"), stand(0), stand(1), stand(2), stand(3), 1e-20, 1e-6, 20, stand(4), stand(5), c.ptr("POST"), c.ptr("COMB"), c.ptr("PART"), ps, null, rows, d);
            } else if (t.pdl) {
                // the mixes of h (the same launch with a side stream or without)
                _ = try hcPre2(t, c.ptr("X"), c.ptr("FN"), stand(0), stand(1), stand(2), stand(3), 1e-20, 1e-6, 20, stand(4), stand(5), c.ptr("POST"), c.ptr("COMB"), c.ptr("PART"), null, side, rows, d);
            } else {
                // no PDL, though served with it on: hc_dots' mixes of the stored streams (X is h_out), on the side one
                var tm = t;
                tm.pdl = true;
                _ = try hcPre2(tm, stand(6), c.ptr("FN"), stand(0), stand(1), stand(2), stand(3), 1e-20, 1e-6, 20, stand(4), stand(5), c.ptr("POST"), c.ptr("COMB"), c.ptr("PART"), .{ .gathered = stand(7), .world = 2, .h_out = c.ptr("X") }, side, rows, d);
            }
        }
    }.call);
}

test "hc_pre2's split finish launches as recorded, with the post or without" {
    try tri.conform("_hc_finish_s", finishCase(false));
    try tri.conform("_hc_finish_s", finishCase(true));
}

test "hc_pre2's deferred Sinkhorn launches as recorded, with the post or without" {
    try tri.conform("_hc_sinkhorn", finishCase(false));
    try tri.conform("_hc_sinkhorn", finishCase(true));
}

test "hc_pre2's post alone (hc_dots) launches as recorded" {
    try tri.conform("_hc_post_only", struct {
        fn call(t: Tri, c: Case) !void {
            _ = try hcPre2(t, c.ptr("X"), stand(0), stand(1), stand(2), stand(3), stand(4), 1e-20, 1e-6, 20, stand(5), stand(6), c.ptr("POST"), c.ptr("COMB"), stand(7), postedOf(c).?, side, c.dim("X", 0), c.dim("X", 2));
        }
    }.call);
}

test "hc_mix_pf and hc_pre_pf launch as recorded" {
    try tri.conform("_hc_mix_pf", struct {
        fn call(t: Tri, c: Case) !void {
            try std.testing.expect(hcPf2Ok(c.dim("X", 2)));
            _ = try hcMixPf(t, c.ptr("X"), c.ptr("FN"), c.ptr("PART"), postedOf(c), c.ptr("POST"), c.ptr("COMB"), c.dim("X", 0), c.dim("X", 2));
        }
    }.call);
    try tri.conform("_hc_mix_pf", struct {
        fn call(t: Tri, c: Case) !void {
            _ = try hcPrePf(t, c.ptr("X"), c.ptr("FN"), stand(0), stand(1), stand(2), stand(3), 1e-20, 1e-6, 20, stand(4), stand(5), c.ptr("POST"), c.ptr("COMB"), c.ptr("PART"), postedOf(c), c.dim("X", 0), c.dim("X", 2));
        }
    }.call);
    try tri.conform("_hc_finish", struct {
        fn call(t: Tri, c: Case) !void {
            // served, hc_pre_pf always has a post (model._hc_pf2 runs hc_pre without one): its finish's X is h_out
            const ps: Posted = .{ .gathered = stand(2), .world = 2, .h_out = c.ptr("X") };
            _ = try hcPrePf(t, stand(0), stand(1), c.ptr("SCALE"), c.ptr("BASE"), c.ptr("PRE_IN"), c.ptr("NW"), c.float("eps"), c.float("hc_eps"), @intCast(c.constInt("ITERS")), c.ptr("OUT"), c.ptr("PRE_OUT"), c.ptr("POST"), c.ptr("COMB"), c.ptr("PART"), ps, c.dim("X", 0), c.dim("X", 2));
        }
    }.call);
}

/// One argument a logged launch must have: the launch's index, the argument's name, its address.
const Wire = struct { usize, []const u8, u64 };

/// Checks the logged launches are `names` in order, each wired as `wires` says, then empties the log.
fn expectLog(log: *tri.Log, names: []const []const u8, wires: []const Wire) !void {
    defer {
        log.deinit();
        log.items = .empty;
    }
    try std.testing.expectEqual(names.len, log.items.items.len);
    for (names, log.items.items) |n, k| try std.testing.expectEqualStrings(n, k.name);
    for (wires) |w| {
        const got = for (log.items.items[w[0]].args) |a| {
            if (std.mem.eql(u8, a.name, w[1])) break a.value.ptr.addr;
        } else return error.TestUnexpectedResult;
        try std.testing.expectEqual(w[2], got);
    }
}

test "hc_pre2 and hc_pre_pf launch in kernels.py's order, each kernel on the streams kernels.py gives it" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    const t: Tri = .{ .log = &log };
    const h = stand(1);
    const part = stand(2);
    const ps: Posted = .{ .gathered = stand(3), .world = 2, .h_out = stand(4) };
    const posted = [_]Wire{ .{ 0, "X", h }, .{ 0, "XO", ps.h_out }, .{ 0, "G", ps.gathered } };
    // hc_dots: the post and the finish of h_out here, then the mixes of h_out and the Sinkhorn on the side stream
    try std.testing.expectEqual(ps.h_out, try hcPre2(t, h, stand(5), stand(6), stand(7), stand(8), stand(9), 1e-20, 1e-6, 20, stand(10), stand(11), stand(12), stand(13), part, ps, side, 16, 5120));
    try expectLog(&log, &.{ "_hc_post_only", "_hc_finish_s", "_hc_mix_part", "_hc_sinkhorn" }, &(posted ++ [_]Wire{
        .{ 1, "X", ps.h_out }, .{ 2, "X", ps.h_out }, .{ 2, "XO", ps.h_out }, .{ 2, "G", ps.h_out }, .{ 3, "PART", part },
    }));
    // posted without a side stream (dspark): the posted mixes, then the finish of h_out with its Sinkhorn programs
    try std.testing.expectEqual(ps.h_out, try hcPre2(t, h, stand(5), stand(6), stand(7), stand(8), stand(9), 1e-20, 1e-6, 20, stand(10), stand(11), stand(12), stand(13), part, ps, null, 5, 5120));
    try expectLog(&log, &.{ "_hc_mix_part", "_hc_finish_s" }, &(posted ++ [_]Wire{.{ 1, "X", ps.h_out }}));
    // no post, a side stream: the mixes of h, the Sinkhorn on the side stream, then the finish of h
    try std.testing.expectEqual(h, try hcPre2(t, h, stand(5), stand(6), stand(7), stand(8), stand(9), 1e-20, 1e-6, 20, stand(10), stand(11), stand(12), stand(13), part, null, side, 16, 5120));
    try expectLog(&log, &.{ "_hc_mix_part", "_hc_sinkhorn", "_hc_finish_s" }, &.{ .{ 0, "X", h }, .{ 0, "XO", h }, .{ 0, "G", h }, .{ 1, "PART", part }, .{ 2, "X", h } });
    // a width off the 1024 grain: _hc_finish_u, the side stream unused
    try std.testing.expectEqual(h, try hcPre2(t, h, stand(5), stand(6), stand(7), stand(8), stand(9), 1e-20, 1e-6, 20, stand(10), stand(11), stand(12), stand(13), part, null, side, 16, 2560));
    try expectLog(&log, &.{ "_hc_mix_part", "_hc_finish_u" }, &.{.{ 1, "X", h }});
    // hc_pre_pf with the post: hc_mix_pf's posted dots, then hc_pre's finish of h_out
    try std.testing.expectEqual(ps.h_out, try hcPrePf(t, h, stand(5), stand(6), stand(7), stand(8), stand(9), 1e-20, 1e-6, 20, stand(10), stand(11), stand(12), stand(13), part, ps, 2048, 5120));
    try expectLog(&log, &.{ "_hc_mix_pf", "_hc_finish" }, &(posted ++ [_]Wire{.{ 1, "X", ps.h_out }}));
}
