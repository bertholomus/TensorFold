//! Images on rank 0 for the served lane's vision path: an image's bytes into its picture on the request's thread
//! (lanes.stream.Vision.prepare: decoded, planned, padded and cut into the ViT's patches as the lane's `decode` makes
//! them, picture.zig), and the picture into its span's rows on the lane host's thread at the prompt pass (the tower's
//! encode and span rows, vit.zig), which lanes.zig then hands to the text model and to the other ranks.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const picture = @import("picture.zig");
const vit = @import("vit.zig");
const fmha = @import("fmha.zig");
const model = @import("model.zig");
const pil = @import("pil.zig");

/// A prepared image's payload: its grid (six u32, little-endian: best_h, best_w, n_vit_h, n_vit_w, n_llm_h, n_llm_w)
/// then its patches (bf16, picture.patchify's order), the patches 16-byte aligned.
const header_bytes = 32;

/// The span a picture can take at most (picture.Config.max_tokens aligner rows, a newline each row, the delimiters).
const max_span = 4096;

pub const Vision = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    cfg: picture.Config = .{},
    tower: vit.Tower,
    span: cuda.DeviceBuffer, // bf16 [max_span, 5120]: a picture's span rows
    scratch: cuda.DeviceBuffer, // the span's types and rows (Tower.spanRows)
    pil: ?*pil.Pil = null, // the formats picture.zig does not decode (null: no python3 with Pillow here)

    /// The tower on `m`'s stream (its exact kernels and cuBLAS handle, as the tower gate ran them), its weights from
    /// the checkpoint, torch's attention cubin from the kit's vision/, and its workspace for the largest picture.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, model_dir: []const u8, kit_dir: []const u8) !*Vision {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const cubin_path = try std.fmt.bufPrint(&path_buf, "{s}/vision/{s}", .{ kit_dir, fmha.cubin_file });
        const cubin = std.Io.Dir.cwd().readFileAlloc(io, cubin_path, gpa, .limited(64 << 20)) catch |err| return if (err == error.FileNotFound) error.NoVisionKit else err;
        defer gpa.free(cubin);
        if (m.w.layers.len == 0 or m.w.layers[0].gate_b_vl == 0) return error.NoVisionKit; // (model.zig loads the bias)
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        v.* = .{ .gpa = gpa, .m = m, .tower = undefined, .span = undefined, .scratch = undefined };
        const d = m.ctx.d;
        v.tower = try vit.Tower.open(gpa, io, d, m.stream, model_dir, cubin, cuda.kernels.dsv41_ops, &m.ex, &m.blas, m.ch.blas_ws);
        errdefer v.tower.close();
        try v.tower.reserve(v.cfg.max_tokens * v.cfg.ratio * v.cfg.ratio, v.cfg.max_tokens);
        v.span = try cuda.DeviceBuffer.alloc(d, max_span * vit.model_dim * 2);
        errdefer v.span.free();
        v.scratch = try cuda.DeviceBuffer.alloc(d, max_span * 8 + 4096);
        errdefer v.scratch.free();
        v.pil = pil.Pil.start(gpa, io) catch |err| blk: {
            std.debug.print("{{\"vision\": \"no Pillow ({s}): images other than PNG are refused\"}}\n", .{@errorName(err)});
            break :blk null;
        };
        if (v.pil) |p| std.debug.print("{{\"vision\": \"Pillow {s} decodes the formats other than PNG\"}}\n", .{p.versionText()});
        return v;
    }

    pub fn close(v: *Vision) void {
        if (v.pil) |p| p.stop(v.gpa);
        v.scratch.free();
        v.span.free();
        v.tower.close();
        v.gpa.destroy(v);
    }

    /// The server's view: the placeholder the template renders for each image and the picture it prepares.
    pub fn hook(v: *Vision) lanes.stream.Vision {
        return .{ .ctx = v, .placeholder = placeholder, .token = v.m.cfg.image_token.?, .prepare = prepareFn };
    }

    pub const placeholder = "<\u{ff5c}deepseek_image\u{ff5c}>";

    fn prepareFn(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, problem: *[]const u8) lanes.stream.Vision.PrepareError!lanes.stream.Vision.Prepared {
        const v: *Vision = @ptrCast(@alignCast(ctx));
        var rgb = try v.decode(a, bytes, problem);
        defer rgb.deinit(a);
        var pic = picture.fromRgb(a, rgb, v.cfg) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            problem.* = "image input: the image could not be prepared";
            return error.BadImage;
        };
        defer pic.deinit(a);
        const g = pic.grid;
        const payload = try a.alignedAlloc(u8, .@"16", header_bytes + pic.patches.len * 2);
        @memset(payload[0..header_bytes], 0);
        for ([_]usize{ g.best_h, g.best_w, g.n_vit_h, g.n_vit_w, g.n_llm_h, g.n_llm_w }, 0..) |x, i| std.mem.writeInt(u32, payload[i * 4 ..][0..4], @intCast(x), .little);
        @memcpy(payload[header_bytes..], std.mem.sliceAsBytes(pic.patches));
        return .{ .tokens = @intCast(g.spanTokens()), .payload = payload };
    }

    /// The image's RGB as the lane's Pillow decodes it: picture.zig's PNG, else Pillow's own (the refusal in the lane's
    /// words on BadImage).
    fn decode(v: *Vision, a: std.mem.Allocator, bytes: []const u8, problem: *[]const u8) lanes.stream.Vision.PrepareError!picture.Rgb {
        if (picture.decodePng(a, bytes)) |rgb| return rgb else |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TooLarge => {
                problem.* = try std.fmt.allocPrint(a, "Image size ({d} pixels) exceeds limit of {d} pixels, could be decompression bomb DOS attack.", .{ picture.pixels(bytes) orelse 0, picture.max_pixels });
                return error.BadImage;
            },
            else => {}, // not a PNG this decoder reads: Pillow's
        }
        const p = v.pil orelse {
            problem.* = "image input: cannot identify image file (no Pillow on this server for formats other than PNG)";
            return error.BadImage;
        };
        const got = p.decode(a, bytes) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            std.debug.print("{{\"vision\": \"Pillow's child failed ({s})\"}}\n", .{@errorName(err)});
            problem.* = "image input: the image decoder failed";
            return error.BadImage;
        };
        switch (got) {
            .rgb => |rgb| return rgb,
            .refused => |m| {
                problem.* = m;
                return error.BadImage;
            },
        }
    }

    /// A prepared picture's span rows bf16 [tokens, 5120] appended to `out` (the tower on the model's stream, then the
    /// rows to the host); its span's tokens.
    pub fn rows(v: *Vision, payload: []const u8, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !usize {
        if (payload.len < header_bytes) return error.BadImagePayload;
        var x: [6]usize = undefined;
        for (&x, 0..) |*f, i| f.* = std.mem.readInt(u32, payload[i * 4 ..][0..4], .little);
        const g: picture.Grid = .{ .best_h = x[0], .best_w = x[1], .n_vit_h = x[2], .n_vit_w = x[3], .n_llm_h = x[4], .n_llm_w = x[5] };
        const n = g.patches();
        if (payload.len != header_bytes + n * vit.patch_in * 2 or g.spanTokens() > max_span) return error.BadImagePayload;
        const body: []align(2) const u8 = @alignCast(payload[header_bytes..]);
        const patches = std.mem.bytesAsSlice(u16, body);
        const at = try v.tower.encode(g, patches, null);
        try v.tower.spanRows(g, at, v.span.ptr, v.scratch.ptr);
        const t = g.spanTokens();
        const bytes = t * vit.model_dim * 2;
        try v.tower.stream.synchronize();
        const from = out.items.len;
        try out.resize(gpa, from + bytes);
        const d = v.m.ctx.d;
        try d.check(d.api.cuMemcpyDtoH_v2(out.items[from..].ptr, v.span.ptr, bytes), "cuMemcpyDtoH");
        return t;
    }
};

test "a payload's grid reads back" {
    var buf: [header_bytes]u8 = undefined;
    for ([_]u32{ 462, 644, 33, 46, 11, 16 }, 0..) |x, i| std.mem.writeInt(u32, buf[i * 4 ..][0..4], x, .little);
    try std.testing.expectEqual(@as(u32, 11), std.mem.readInt(u32, buf[16..20], .little));
    const g: picture.Grid = .{ .best_h = 462, .best_w = 644, .n_vit_h = 33, .n_vit_w = 46, .n_llm_h = 11, .n_llm_w = 16 };
    try std.testing.expectEqual(@as(usize, 189), g.spanTokens());
}
