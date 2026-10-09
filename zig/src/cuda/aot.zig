//! A captured set of Triton cubins (aot.json + cubins/): each launch picks the variant Triton itself would have picked.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const triton = @import("triton.zig");

const ParamJson = struct { name: []const u8, type: []const u8, div16: bool, nospec: bool };
const ConstJson = struct { int: ?i64 = null, f32: ?u32 = null };
const KernelJson = struct {
    @"fn": []const u8,
    hash: []const u8,
    name: []const u8,
    num_warps: u32,
    num_ctas: u32 = 1,
    shared: u32 = 0,
    global_scratch: u32 = 0,
    global_align: u32 = 1,
    profile_scratch: u32 = 0,
    pdl: bool = false,
    params: []ParamJson,
    consts: std.json.ArrayHashMap(ConstJson),
};
const SetJson = struct { kernels: []KernelJson };

/// One argument of a launch, by the kernel's parameter name.
pub const Arg = struct {
    name: []const u8,
    value: Value,

    pub const Value = union(enum) { ptr: struct { addr: u64, ty: []const u8 }, i32: i32, f32: f32, u64: u64 };
};

/// A constexpr the call site compiled the kernel with (Python's keyword arguments): ints, bools, fp32 bits.
pub const Const = struct { name: []const u8, int: ?i64 = null, f32: ?f32 = null };

pub fn ptr(name: []const u8, ty: []const u8, addr: u64) Arg {
    return .{ .name = name, .value = .{ .ptr = .{ .addr = addr, .ty = ty } } };
}
pub fn int(name: []const u8, v: i32) Arg {
    return .{ .name = name, .value = .{ .i32 = v } };
}
pub fn float(name: []const u8, v: f32) Arg {
    return .{ .name = name, .value = .{ .f32 = v } };
}
pub fn word(name: []const u8, v: u64) Arg {
    return .{ .name = name, .value = .{ .u64 = v } };
}
pub fn ci(name: []const u8, v: i64) Const {
    return .{ .name = name, .int = v };
}
pub fn cf(name: []const u8, v: f32) Const {
    return .{ .name = name, .f32 = v };
}

const Variant = struct {
    spec: KernelJson,
    kernel: triton.Kernel,
    /// Said once on stderr when a launch first takes this variant in place of its own specialization.
    told: std.atomic.Value(bool) = .init(false),
};

pub const Set = struct {
    parsed: std.json.Parsed(SetJson),
    variants: []Variant,
    gpa: std.mem.Allocator,
    /// TF_AOT_WEAKEST=1: every launch takes its most general fitting variant (a gate of the fallback's exactness).
    weakest: bool = false,

    /// Loads every cubin listed in `dir`/aot.json into its own module.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, device: abi.Device, dir: []const u8) !Set {
        const path = try std.fs.path.join(gpa, &.{ dir, "aot.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(SetJson, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        errdefer parsed.deinit();
        const variants = try gpa.alloc(Variant, parsed.value.kernels.len);
        var n: usize = 0;
        errdefer {
            for (variants[0..n]) |*v| v.kernel.unload();
            gpa.free(variants);
        }
        for (parsed.value.kernels) |k| {
            if (k.global_scratch != 0 or k.profile_scratch != 0) return error.ScratchUnsupported;
            const file = try std.fmt.allocPrint(gpa, "{s}/cubins/{s}.cubin", .{ dir, k.hash });
            defer gpa.free(file);
            const cubin = try std.Io.Dir.cwd().readFileAllocOptions(io, file, gpa, .limited(1 << 26), .@"16", null);
            defer gpa.free(cubin);
            const name_z = try gpa.dupeSentinel(u8, k.name, 0);
            defer gpa.free(name_z);
            const meta: triton.Meta = .{ .name = k.name, .num_warps = k.num_warps, .num_ctas = k.num_ctas, .shared = k.shared, .launch_pdl = k.pdl };
            variants[n] = .{ .spec = k, .kernel = try triton.Kernel.load(d, device, cubin, meta, name_z) };
            n += 1;
        }
        const weakest = if (std.c.getenv("TF_AOT_WEAKEST")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        return .{ .parsed = parsed, .variants = variants, .gpa = gpa, .weakest = weakest };
    }

    pub fn deinit(self: *Set) void {
        for (self.variants) |*v| v.kernel.unload();
        self.gpa.free(self.variants);
        self.parsed.deinit();
        self.* = undefined;
    }

    /// The variant of `function` compiled for these constexprs and these arguments' specialization; without one, the
    /// fitting variant that assumes least less (fits: the same constexprs, each assumption it makes held by the
    /// launch), named once on stderr.
    pub fn find(self: *const Set, function: []const u8, args: []const Arg, consts: []const Const) !*const Variant {
        if (!self.weakest) for (self.variants) |*v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            if (matches(v.spec, args, consts)) return v;
        };
        var best: ?*Variant = null;
        var best_held: usize = 0;
        for (self.variants) |*v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            const held = fits(v.spec, args, consts) orelse continue;
            const better = if (self.weakest) held < best_held else held > best_held;
            if (best == null or better) {
                best = v;
                best_held = held;
            }
        }
        if (best) |v| {
            if (!v.told.swap(true, .monotonic)) std.log.warn("{s}: a launch whose specialization no captured variant has runs variant {s} (assumes less)", .{ function, v.spec.hash[0..@min(12, v.spec.hash.len)] });
            return v;
        }
        std.log.err("no captured Triton variant of {s} for this launch:", .{function});
        for (args) |a| switch (a.value) {
            .ptr => |p| std.log.err("  {s}: {s} at {x} (16-aligned {})", .{ a.name, p.ty, p.addr, p.addr % 16 == 0 }),
            .i32 => |x| std.log.err("  {s}: i32 {d}", .{ a.name, x }),
            .f32 => |x| std.log.err("  {s}: fp32 {d}", .{ a.name, x }),
            .u64 => |x| std.log.err("  {s}: u64 {d}", .{ a.name, x }),
        };
        return error.MissingTritonVariant;
    }

    /// The smallest value of constexpr `name` at or above `at_least` among `function`'s variants.
    pub fn smallestConst(self: *const Set, function: []const u8, name: []const u8, at_least: i64) ?i64 {
        var best: ?i64 = null;
        for (self.variants) |v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            const c = (v.spec.consts.map.get(name) orelse continue).int orelse continue;
            if (c >= at_least and (best == null or c < best.?)) best = c;
        }
        return best;
    }

    /// Launches `function` on `grid` as Triton's launcher would: runtime arguments in the variant's order, scratch null.
    pub fn run(self: *const Set, stream: Stream, function: []const u8, grid: [3]u32, args: []const Arg, consts: []const Const) !void {
        const v = try self.find(function, args, consts);
        var packed_args: launch.Args = .{};
        for (v.spec.params) |p| {
            const a = lookup(args, p.name).?;
            switch (a.value) {
                .ptr => |x| packed_args.add(x.addr),
                .i32 => |x| packed_args.add(x),
                .f32 => |x| packed_args.add(x),
                .u64 => |x| packed_args.add(x),
            }
        }
        try v.kernel.launchOn(.{ .x = grid[0], .y = grid[1], .z = grid[2] }, stream, &packed_args, .{}, &.{});
    }
};

fn lookup(args: []const Arg, name: []const u8) ?Arg {
    for (args) |a| if (std.mem.eql(u8, a.name, name)) return a;
    return null;
}

/// Whether variant `k` may run this launch though compiled for a narrower one: the same constexprs, and every
/// assumption it makes held by the launch (an argument it takes as 16-divisible is; an int it folded to 1 is 1). Null
/// when it may not, else how many of the launch's own specializations it shares.
fn fits(k: KernelJson, args: []const Arg, consts: []const Const) ?usize {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return null;
        if (c.int) |x| if (got.int == null or got.int.? != x) return null;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return null;
    }
    var held: usize = 0;
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    const got = k.consts.map.get(a.name) orelse return null;
                    if (x != 1 or got.int == null or got.int.? != 1) return null;
                    held += 1;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return null;
                const d16 = !p.nospec and x != 1 and @mod(x, 16) == 0;
                if (p.div16 and !d16) return null;
                if (p.div16 == d16 and (p.nospec or x != 1)) held += 1;
            },
            .ptr => |x| {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, x.ty)) return null;
                const d16 = x.addr % 16 == 0;
                if (p.div16 and !d16) return null;
                if (p.div16 == d16) held += 1;
            },
            .f32 => {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, "fp32")) return null;
                held += 1;
            },
            .u64 => |x| {
                const p = param orelse return null;
                if (!std.mem.eql(u8, p.type, "u64")) return null;
                const d16 = x % 16 == 0;
                if (p.div16 and !d16) return null;
                if (p.div16 == d16) held += 1;
            },
        }
        runtime += 1;
    }
    if (runtime != k.params.len) return null;
    return held;
}

fn matches(k: KernelJson, args: []const Arg, consts: []const Const) bool {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return false;
        if (c.int) |x| if (got.int == null or got.int.? != x) return false;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return false;
    }
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    // an int Triton folded: only the value 1 is ever specialized to a constexpr
                    const got = k.consts.map.get(a.name) orelse return false;
                    if (x != 1 or got.int == null or got.int.? != 1) return false;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return false;
                if (!p.nospec and x == 1) return false;
                if (p.div16 != (!p.nospec and @mod(x, 16) == 0)) return false;
            },
            .ptr => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, x.ty) or p.div16 != (x.addr % 16 == 0)) return false;
            },
            .f32 => {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "fp32")) return false;
            },
            .u64 => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "u64") or p.div16 != (x % 16 == 0)) return false;
            },
        }
        runtime += 1;
    }
    return runtime == k.params.len;
}
