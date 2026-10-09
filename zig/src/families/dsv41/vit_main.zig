//! tf-dsv41-vit MODEL_DIR CUBIN MANIFEST: the vision tower gate (devchain dsv41 d93). For each picture the manifest
//! lists (py/vgate.py from a recording: its patches file, grid, and the served lane's sha256 at each step), the port's
//! tower runs on the recorded patches and each step's bytes are compared: the patch embed, the RoPE tables, every
//! block's output, the aligner's rows and the span rows. One JSON line a picture; exit 1 on any difference.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const vit = dsv41.vit;
const picture = dsv41.picture;

const Image = struct {
    patches: []const u8,
    n_vit_h: usize,
    n_vit_w: usize,
    n_llm_h: usize,
    n_llm_w: usize,
    embed: []const u8,
    cos: []const u8,
    sin: []const u8,
    blocks: []const []const u8,
    rows: []const u8,
    span: []const u8,
};

const Check = struct {
    a: std.mem.Allocator,
    stream: cuda.Stream,
    img: *const Image,
    host: []u8,
    equal: usize = 0,
    differ: usize = 0,
    first: ?[]const u8 = null,

    fn hex(bytes: []const u8) [64]u8 {
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
        return std.fmt.bytesToHex(d, .lower);
    }

    fn probe(ctx: *anyopaque, what: []const u8, index: usize, ptr: u64, bytes: usize) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        const want: []const u8 = if (std.mem.eql(u8, what, "embed")) c.img.embed else if (std.mem.eql(u8, what, "cos")) c.img.cos else if (std.mem.eql(u8, what, "sin")) c.img.sin else if (std.mem.eql(u8, what, "block")) c.img.blocks[index] else if (std.mem.eql(u8, what, "rows")) c.img.rows else return;
        try c.compare(what, index, ptr, bytes, want);
    }

    fn compare(c: *Check, what: []const u8, index: usize, ptr: u64, bytes: usize, want: []const u8) !void {
        try c.stream.synchronize();
        const d = c.stream.d;
        try d.check(d.api.cuMemcpyDtoH_v2(c.host.ptr, ptr, bytes), "cuMemcpyDtoH");
        const got = hex(c.host[0..bytes]);
        if (std.mem.eql(u8, &got, want)) {
            c.equal += 1;
        } else {
            c.differ += 1;
            if (c.first == null) c.first = try std.fmt.allocPrint(c.a, "{s}.{d}", .{ what, index });
        }
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) {
        std.debug.print("usage: tf-dsv41-vit MODEL_DIR CUBIN MANIFEST\n", .{});
        return 2;
    }
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w = &out.interface;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var ex = try dsv41.exact.Exact.load(&driver, cuda.kernels.dsv41_torch);
    var ws = try cuda.DeviceBuffer.alloc(&driver, dsv41.cublas.Blas.workspace_bytes);
    defer ws.free();
    var blas = try dsv41.cublas.Blas.open(stream, ws.ptr);
    defer blas.close();
    const cubin = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(64 << 20));
    var tower = try vit.Tower.open(gpa, io, &driver, stream, args[1], cubin, cuda.kernels.dsv41_ops, &ex, &blas, ws.ptr);
    defer tower.close();

    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[3], a, .limited(1 << 26));
    const images = try std.json.parseFromSliceLeaky([]Image, a, text, .{ .ignore_unknown_fields = true });
    var span_buf = try cuda.DeviceBuffer.alloc(&driver, 4096 * vit.model_dim * 2);
    defer span_buf.free();
    var scratch = try cuda.DeviceBuffer.alloc(&driver, 4096 * 8 + 4096);
    defer scratch.free();
    var host = try a.alloc(u8, 64 << 20);
    var bad: u8 = 0;
    for (images, 0..) |*img, k| {
        const patches_bytes = try std.Io.Dir.cwd().readFileAlloc(io, img.patches, a, .limited(1 << 28));
        const g: picture.Grid = .{ .best_h = img.n_vit_h * 14, .best_w = img.n_vit_w * 14, .n_vit_h = img.n_vit_h, .n_vit_w = img.n_vit_w, .n_llm_h = img.n_llm_h, .n_llm_w = img.n_llm_w };
        if (patches_bytes.len != g.patches() * vit.patch_in * 2) return error.PatchesSize;
        const patches = std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(patches_bytes)));
        if (host.len < g.patches() * vit.dim * 4) host = try a.alloc(u8, g.patches() * vit.dim * 4);
        var c: Check = .{ .a = a, .stream = stream, .img = img, .host = host };
        const rows = try tower.encode(g, patches, .{ .ctx = &c, .f = Check.probe });
        try tower.spanRows(g, rows, span_buf.ptr, scratch.ptr);
        try c.compare("span", 0, span_buf.ptr, g.spanTokens() * vit.model_dim * 2, img.span);
        try w.print("{{\"image\": {d}, \"patches\": {d}, \"equal\": {d}, \"differ\": {d}, \"first_difference\": \"{s}\"}}\n", .{ k, g.patches(), c.equal, c.differ, c.first orelse "" });
        try w.flush();
        if (c.differ > 0) bad = 1;
    }
    return bad;
}
