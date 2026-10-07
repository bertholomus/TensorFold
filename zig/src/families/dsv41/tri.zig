//! Our Triton kernels from the captured set (aot.json + cubins), launched as the served build's wrappers launch them:
//! the same grid, constexprs and PDL. The set picks each launch's variant as Triton would (zig/src/cuda/aot.zig).
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

/// Where launches go: the captured set on a stream, or (tests) a log compared with the recorded launches.
pub const Tri = struct {
    set: ?*const aot.Set = null,
    stream: cuda.Stream = undefined,
    log: ?*Log = null,
    /// The "pdl" switch (TF_DS_TRITON_PDL, on in the served build): decode kernels as programmatic dependent launches.
    pdl: bool = true,

    pub fn run(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        if (t.log) |l| return l.add(name, grid, args, consts);
        try t.set.?.run(t.stream, name, grid, args, consts);
    }

    /// The PDL constexpr a wrapper passes when its kernel takes one (_pdl() in kernels.py).
    pub fn pdlConst(t: Tri) aot.Const {
        return aot.ci("PDL", @intFromBool(t.pdl));
    }
};

pub fn u(x: usize) u32 {
    return @intCast(x);
}

pub fn cdiv(a: usize, b: usize) u32 {
    return u((a + b - 1) / b);
}

pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(n, 1)) catch unreachable;
}

/// A test log of launches: each one's function, grid, arguments and constexprs, copied.
pub const Log = struct {
    gpa: std.mem.Allocator,
    items: std.ArrayList(Kept) = .empty,

    pub const Kept = struct { name: []const u8, grid: [3]u32, args: []aot.Arg, consts: []aot.Const };

    fn add(l: *Log, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        try l.items.append(l.gpa, .{ .name = name, .grid = grid, .args = try l.gpa.dupe(aot.Arg, args), .consts = try l.gpa.dupe(aot.Const, consts) });
    }

    pub fn deinit(l: *Log) void {
        for (l.items.items) |k| {
            l.gpa.free(k.args);
            l.gpa.free(k.consts);
        }
        l.items.deinit(l.gpa);
    }
};

// -- conformance with the recording (fixtures/launches.json: each Triton function's distinct recorded launches) -----

pub const fixtures = @embedFile("fixtures/launches.json");

/// One recorded launch: its tensors' dtype and shape (with 16-byte alignment), scalars and constexprs.
pub const Case = struct {
    v: std.json.ObjectMap,

    pub fn grid(c: Case) [3]u32 {
        const g = c.v.get("grid").?.array.items;
        return .{ @intCast(g[0].integer), @intCast(g[1].integer), @intCast(g[2].integer) };
    }

    /// A tensor argument's dimension `i`.
    pub fn dim(c: Case, tensor: []const u8, i: usize) usize {
        return @intCast(c.tensorV(tensor).array.items[1].array.items[i].integer);
    }

    /// A tensor argument's stride `i` (elements).
    pub fn stride(c: Case, tensor: []const u8, i: usize) usize {
        return @intCast(c.tensorV(tensor).array.items[2].array.items[i].integer);
    }

    pub fn rank(c: Case, tensor: []const u8) usize {
        return c.tensorV(tensor).array.items[1].array.items.len;
    }

    pub fn has(c: Case, tensor: []const u8) bool {
        return c.v.get("tensors").?.object.get(tensor) != null;
    }

    fn tensorV(c: Case, tensor: []const u8) std.json.Value {
        return c.v.get("tensors").?.object.get(tensor) orelse std.debug.panic("no tensor {s} in the case", .{tensor});
    }

    /// A fake device address for `tensor`: 16-aligned or not as recorded, distinct a name.
    pub fn ptr(c: Case, tensor: []const u8) u64 {
        const t = c.tensorV(tensor).array.items;
        const h: u64 = std.hash.Wyhash.hash(0, tensor) & 0xffff_ff00;
        return (0x7f00_0000_0000 + (h << 8)) + (if (t[3].bool) @as(u64, 0) else 2);
    }

    pub fn int(c: Case, name: []const u8) i64 {
        return c.v.get("scalars").?.object.get(name).?.integer;
    }

    pub fn float(c: Case, name: []const u8) f32 {
        const v = c.v.get("scalars").?.object.get(name).?;
        const bits = std.fmt.parseInt(u32, v.object.get("fp32_bits").?.string[2..], 16) catch unreachable;
        return @bitCast(bits);
    }

    pub fn constInt(c: Case, name: []const u8) i64 {
        const v = c.v.get("consts").?.object.get(name).?;
        return switch (v) {
            .integer => |i| i,
            .bool => |b| @intFromBool(b),
            else => std.debug.panic("constexpr {s} is not an integer", .{name}),
        };
    }

    pub fn hasConst(c: Case, name: []const u8) bool {
        return c.v.get("consts").?.object.get(name) != null;
    }
};

fn tritonType(dtype: []const u8) []const u8 {
    const names = .{ .{ "bfloat16", "*bf16" }, .{ "float16", "*fp16" }, .{ "float32", "*fp32" }, .{ "int64", "*i64" }, .{ "int32", "*i32" }, .{ "int16", "*i16" }, .{ "uint8", "*u8" }, .{ "int8", "*i8" }, .{ "bool", "*i1" }, .{ "float8_e4m3fn", "*fp8e4nv" } };
    inline for (names) |n| if (std.mem.eql(u8, dtype, n[0])) return n[1];
    return dtype;
}

/// Runs `call` on every recorded launch of `fn_name` and checks that it asks for that launch: grid, every scalar
/// (floats by their fp32 bits), every constexpr, and every tensor argument's pointer type and alignment.
pub fn conform(fn_name: []const u8, call: *const fn (Tri, Case) anyerror!void) !void {
    const a = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, a, fixtures, .{});
    defer parsed.deinit();
    const cases = (parsed.value.object.get(fn_name) orelse return error.NoFixtures).array.items;
    try std.testing.expect(cases.len > 0);
    for (cases, 0..) |cv, ci| {
        const c: Case = .{ .v = cv.object };
        var log: Log = .{ .gpa = a };
        defer log.deinit();
        try call(.{ .log = &log, .pdl = !c.hasConst("PDL") or c.constInt("PDL") != 0 }, c);
        const k = for (log.items.items) |k| {
            if (std.mem.eql(u8, k.name, fn_name)) break k;
        } else {
            std.debug.print("case {d}: the wrapper launched no {s}\n", .{ ci, fn_name });
            return error.TestUnexpectedResult;
        };
        errdefer std.debug.print("{s} case {d} (phase {s})\n", .{ fn_name, ci, c.v.get("phase").?.string });
        try std.testing.expectEqual(c.grid(), k.grid);
        var consts = c.v.get("consts").?.object.iterator();
        recorded: while (consts.next()) |e| {
            const got = for (k.consts) |x| {
                if (std.mem.eql(u8, x.name, e.key_ptr.*)) break x;
            } else {
                if (e.value_ptr.* == .integer and e.value_ptr.integer == 1 and folded(k.args, e.key_ptr.*)) continue :recorded;
                std.debug.print("constexpr {s} not passed\n", .{e.key_ptr.*});
                return error.TestUnexpectedResult;
            };
            switch (e.value_ptr.*) {
                .integer => |i| try std.testing.expectEqual(@as(?i64, i), got.int),
                .bool => |b| try std.testing.expectEqual(@as(?i64, @intFromBool(b)), got.int),
                .object => |o| try std.testing.expectEqual(std.fmt.parseInt(u32, o.get("fp32_bits").?.string[2..], 16) catch unreachable, @as(u32, @bitCast(got.f32.?))),
                else => return error.UnsupportedConstexpr,
            }
        }
        for (k.consts) |x| if (c.v.get("consts").?.object.get(x.name) == null) {
            std.debug.print("constexpr {s} passed but not in the variant\n", .{x.name});
            return error.TestUnexpectedResult;
        };
        var scalars = c.v.get("scalars").?.object.iterator();
        while (scalars.next()) |e| {
            const got = for (k.args) |x| {
                if (std.mem.eql(u8, x.name, e.key_ptr.*)) break x;
            } else {
                std.debug.print("scalar {s} not passed\n", .{e.key_ptr.*});
                return error.TestUnexpectedResult;
            };
            switch (e.value_ptr.*) {
                .integer => |i| switch (got.value) {
                    .i32 => |x| try std.testing.expectEqual(i, @as(i64, x)),
                    .u64 => |x| try std.testing.expectEqual(i, @as(i64, @intCast(x))),
                    else => return error.TestUnexpectedResult,
                },
                .object => |o| try std.testing.expectEqual(std.fmt.parseInt(u32, o.get("fp32_bits").?.string[2..], 16) catch unreachable, @as(u32, @bitCast(got.value.f32))),
                else => return error.UnsupportedScalar,
            }
        }
        var tensors = c.v.get("tensors").?.object.iterator();
        while (tensors.next()) |e| {
            const got = for (k.args) |x| {
                if (std.mem.eql(u8, x.name, e.key_ptr.*)) break x;
            } else {
                std.debug.print("tensor {s} not passed\n", .{e.key_ptr.*});
                return error.TestUnexpectedResult;
            };
            const t = e.value_ptr.array.items;
            try std.testing.expectEqualStrings(tritonType(t[0].string), got.value.ptr.ty);
            try std.testing.expectEqual(t[3].bool, got.value.ptr.addr % 16 == 0);
        }
        // and no argument the kernel does not take, which aot.Set.find would refuse
        for (k.args) |x| {
            if (c.v.get("scalars").?.object.get(x.name) != null or c.v.get("tensors").?.object.get(x.name) != null) continue;
            if (folded(k.args, x.name)) {
                if (c.v.get("consts").?.object.get(x.name)) |r| if (r == .integer and r.integer == 1) continue;
            }
            std.debug.print("argument {s} passed but not taken by this variant\n", .{x.name});
            return error.TestUnexpectedResult;
        }
    }
}

/// An int argument of 1, which Triton folds into a constexpr of the variant; aot.Set.find takes either form.
fn folded(args: []const aot.Arg, name: []const u8) bool {
    for (args) |a| if (std.mem.eql(u8, a.name, name)) return a.value == .i32 and a.value.i32 == 1;
    return false;
}
