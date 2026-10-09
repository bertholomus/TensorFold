//! A picture for the vision tower, as DeepSeek's MIT reference (image_processor.py) and the served lane make it:
//! the image decoded to RGB as PIL's convert("RGB") gives it (PNG here: every colour type, bit depth and interlace),
//! its grid (plan_image_grid: the patch grid of the ViT and the token grid after the 3x3 aligner), ImageOps.pad to the
//! grid's pixel size (PIL's 8-bit BICUBIC resample of the image fitted inside, 127 grey around it), then each value
//! (v / 255 - 0.5) / 0.5 rounded to bf16, in patch order [patch][channel][row][column] (the ViT's linear patch embed
//! reads each patch flattened that way). Every step is integer or IEEE float math in the reference's order, so the
//! patches are the lane's byte for byte (gate: tf-dsv41-picture against the recorded `decode` points).
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The vision settings the grid depends on (config.json vision_config; DeepSeek-V4.1-Flash's values by default).
pub const Config = struct {
    patch: usize = 14,
    ratio: usize = 3, // the aligner's downsample
    max_tokens: usize = 1024, // an image span's tokens at most
    min_pixels: usize = 544 * 544,
};

/// Packed RGB rows, w * h * 3 bytes.
pub const Rgb = struct {
    w: usize,
    h: usize,
    px: []u8,

    pub fn deinit(r: *Rgb, a: Allocator) void {
        a.free(r.px);
        r.* = undefined;
    }
};

/// plan_image_grid's result: the padded pixel size (a whole number of patches each way), the ViT's patch grid and the
/// LLM's token grid.
pub const Grid = struct {
    best_h: usize,
    best_w: usize,
    n_vit_h: usize,
    n_vit_w: usize,
    n_llm_h: usize,
    n_llm_w: usize,

    /// num_image_tokens: [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END].
    pub fn spanTokens(g: Grid) usize {
        return g.n_llm_h * (g.n_llm_w + 1) + 2;
    }

    pub fn patches(g: Grid) usize {
        return g.n_vit_h * g.n_vit_w;
    }
};

/// The token types of an image span (image_token_types): IMAGE_START, then per row n_llm_w IMAGE and an
/// IMAGE_NEW_LINE, then IMAGE_END.
pub const Type = enum(u8) { start = 0, image = 1, newline = 2, end = 3 };

pub fn spanTypes(g: Grid, out: []Type) void {
    std.debug.assert(out.len == g.spanTokens());
    var i: usize = 0;
    out[i] = .start;
    i += 1;
    for (0..g.n_llm_h) |_| {
        for (0..g.n_llm_w) |_| {
            out[i] = .image;
            i += 1;
        }
        out[i] = .newline;
        i += 1;
    }
    out[i] = .end;
}

extern "c" fn pow(x: f64, y: f64) f64;
extern "c" fn sqrt(x: f64) f64;

/// Python's round() of a float: half to even.
fn pyRound(x: f64) i64 {
    const f = @floor(x);
    const d = x - f;
    var r = f;
    if (d > 0.5) r = f + 1 else if (d == 0.5) r = if (@mod(f, 2.0) == 0) f else f + 1;
    return @intFromFloat(r);
}

fn ceilDiv(a: f64, b: f64) usize {
    return @intFromFloat(@ceil(a / b));
}

/// llm_grid: the token grid the aligner makes of a patch grid of this pixel size.
fn llmGrid(best_h: usize, best_w: usize, p: usize, r: usize) [2]usize {
    const fh: f64 = @floatFromInt(best_h / p);
    const fw: f64 = @floatFromInt(best_w / p);
    const fr: f64 = @floatFromInt(r);
    return .{ @intFromFloat(@ceil(fh / fr)), @intFromFloat(@ceil(fw / fr)) };
}

fn spanOf(n_llm_h: usize, n_llm_w: usize) usize {
    return n_llm_h * (n_llm_w + 1) + 2;
}

/// solve_resize_ratio: the largest aspect-preserving pixel size whose token grid fits in max_tokens.
fn solveResize(height: f64, width: f64, p: usize, r: usize, max_n: usize) [2]usize {
    const ratio = height / width;
    const max_w = sqrt(@as(f64, @floatFromInt(max_n - 2)) / ratio + 0.25) - 0.5;
    const max_h = max_w * ratio;
    const cell = p * r;
    if (max_w < 1.0) return .{ (max_n - 2) / 2 * cell, cell };
    if (max_h < 1.0) return .{ cell, (max_n - 3) * cell };
    const fcell: f64 = @floatFromInt(cell);
    const beta = @min(@floor(max_w) * fcell / width, @floor(max_h) * fcell / height);
    const fp: f64 = @floatFromInt(p);
    return .{ @as(usize, @intFromFloat(@floor(height * beta / fp))) * p, @as(usize, @intFromFloat(@floor(width * beta / fp))) * p };
}

/// plan_image_grid for an image of width x height pixels (max_wh_ratio unset, as this model's config has it).
pub fn plan(width0: usize, height0: usize, cfg: Config) Grid {
    const p = cfg.patch;
    var width: f64 = @floatFromInt(width0);
    var height: f64 = @floatFromInt(height0);
    const area = width0 * height0;
    if (area > 0 and area < cfg.min_pixels) {
        // ratio = (min_pixels / (width * height)) ** 0.5; width = int(width * ratio); height = int(height * ratio)
        const ratio = pow(@as(f64, @floatFromInt(cfg.min_pixels)) / @as(f64, @floatFromInt(area)), 0.5);
        width = @trunc(width * ratio);
        height = @trunc(height * ratio);
    }
    const fp: f64 = @floatFromInt(p);
    var best_w = ceilDiv(width, fp) * p;
    var best_h = ceilDiv(height, fp) * p;
    var g = llmGrid(best_h, best_w, p, cfg.ratio);
    if (spanOf(g[0], g[1]) > cfg.max_tokens) {
        const hw = solveResize(height, width, p, cfg.ratio, cfg.max_tokens);
        best_h = hw[0];
        best_w = hw[1];
        g = llmGrid(best_h, best_w, p, cfg.ratio);
    }
    return .{ .best_h = best_h, .best_w = best_w, .n_vit_h = best_h / p, .n_vit_w = best_w / p, .n_llm_h = g[0], .n_llm_w = g[1] };
}

// ---------------------------------------------------------------------------------------------------- PNG

pub const Error = error{ NotPng, BadPng, UnsupportedPng, TooLarge, OutOfMemory };

const signature = "\x89PNG\r\n\x1a\n";

fn be32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .big);
}

/// The pixel count PIL decodes at most before it refuses (Image.MAX_IMAGE_PIXELS * 2 raises DecompressionBombError).
pub const max_pixels: usize = 2 * 89478485;

/// PNG bytes to RGB as PIL's open(...).convert("RGB") gives them: 8-bit samples as they are, 16-bit ones by their high
/// byte, grey of 1, 2 and 4 bits scaled to 0..255 (0 / 255, x 85, x 17), 16-bit grey clipped at 255 (PIL's I;16 to L),
/// palette indices through PLTE (missing entries black), alpha dropped. Chunk CRCs are checked, as PIL does.
pub fn decodePng(a: Allocator, data: []const u8) Error!Rgb {
    if (data.len < 8 or !std.mem.eql(u8, data[0..8], signature)) return error.NotPng;
    var at: usize = 8;
    var w: usize = 0;
    var h: usize = 0;
    var depth: u8 = 0;
    var ctype: u8 = 0;
    var interlace: u8 = 0;
    var palette: [256][3]u8 = @splat(.{ 0, 0, 0 });
    var have_ihdr = false;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(a);
    while (true) {
        if (at + 12 > data.len) return error.BadPng;
        const len = be32(data[at..]);
        const kind = data[at + 4 .. at + 8];
        if (len > data.len - at - 12) return error.BadPng;
        const body = data[at + 8 .. at + 8 + len];
        const crc = be32(data[at + 8 + len ..]);
        var hc = std.hash.Crc32.init();
        hc.update(kind);
        hc.update(body);
        if (hc.final() != crc) return error.BadPng;
        at += 12 + len;
        if (std.mem.eql(u8, kind, "IHDR")) {
            if (len != 13) return error.BadPng;
            w = be32(body[0..]);
            h = be32(body[4..]);
            depth = body[8];
            ctype = body[9];
            if (body[10] != 0 or body[11] != 0 or body[12] > 1) return error.UnsupportedPng;
            interlace = body[12];
            have_ihdr = true;
        } else if (std.mem.eql(u8, kind, "PLTE")) {
            if (len % 3 != 0 or len > 768) return error.BadPng;
            for (0..len / 3) |i| palette[i] = .{ body[3 * i], body[3 * i + 1], body[3 * i + 2] };
        } else if (std.mem.eql(u8, kind, "IDAT")) {
            try idat.appendSlice(a, body);
        } else if (std.mem.eql(u8, kind, "IEND")) {
            break;
        }
    }
    if (!have_ihdr or w == 0 or h == 0) return error.BadPng;
    if (w * h > max_pixels) return error.TooLarge;
    const channels: usize = switch (ctype) {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        else => return error.UnsupportedPng,
    };
    const ok_depth = switch (ctype) {
        0 => depth == 1 or depth == 2 or depth == 4 or depth == 8 or depth == 16,
        3 => depth == 1 or depth == 2 or depth == 4 or depth == 8,
        else => depth == 8 or depth == 16,
    };
    if (!ok_depth) return error.UnsupportedPng;
    const bits = channels * depth;
    const bpp = @max(1, bits / 8);

    // the passes' filtered rows, inflated
    var raw: std.Io.Writer.Allocating = .init(a);
    defer raw.deinit();
    {
        var in: std.Io.Reader = .fixed(idat.items);
        var z: std.compress.flate.Decompress = .init(&in, .zlib, &.{});
        _ = z.reader.streamRemaining(&raw.writer) catch return error.BadPng;
    }
    const zdata = raw.written();

    // samples per pixel at full depth for the whole image: [h][w][channels] as u16
    const samples = try a.alloc(u16, w * h * channels);
    defer a.free(samples);
    const passes: [7][4]usize = .{ .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 }, .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 } };
    const one: [1][4]usize = .{.{ 0, 0, 1, 1 }};
    const list: []const [4]usize = if (interlace == 1) &passes else &one;
    var off: usize = 0;
    var prev: std.ArrayList(u8) = .empty;
    defer prev.deinit(a);
    var cur: std.ArrayList(u8) = .empty;
    defer cur.deinit(a);
    for (list) |ps| {
        const x0 = ps[0];
        const y0 = ps[1];
        const dx = ps[2];
        const dy = ps[3];
        if (x0 >= w or y0 >= h) continue;
        const pw = (w - x0 + dx - 1) / dx;
        const ph = (h - y0 + dy - 1) / dy;
        const stride = (pw * bits + 7) / 8;
        try prev.resize(a, stride);
        @memset(prev.items, 0);
        try cur.resize(a, stride);
        for (0..ph) |row| {
            if (off + 1 + stride > zdata.len) return error.BadPng;
            const filter = zdata[off];
            const line = zdata[off + 1 .. off + 1 + stride];
            off += 1 + stride;
            const c = cur.items;
            const p = prev.items;
            for (0..stride) |i| {
                const left: u8 = if (i >= bpp) c[i - bpp] else 0;
                const up = p[i];
                const ul: u8 = if (i >= bpp) p[i - bpp] else 0;
                c[i] = switch (filter) {
                    0 => line[i],
                    1 => line[i] +% left,
                    2 => line[i] +% up,
                    3 => line[i] +% @as(u8, @intCast((@as(u16, left) + up) / 2)),
                    4 => line[i] +% paeth(left, up, ul),
                    else => return error.BadPng,
                };
            }
            // samples of this row's pixels
            const y = y0 + row * dy;
            for (0..pw) |col| {
                const x = x0 + col * dx;
                for (0..channels) |ch| {
                    const k = col * channels + ch;
                    samples[(y * w + x) * channels + ch] = sampleAt(c, k, depth);
                }
            }
            std.mem.swap(std.ArrayList(u8), &prev, &cur);
        }
    }

    const px = try a.alloc(u8, w * h * 3);
    errdefer a.free(px);
    for (0..w * h) |i| {
        const s = samples[i * channels ..][0..channels];
        const rgb: [3]u8 = switch (ctype) {
            0 => blk: {
                const v = grey(s[0], depth);
                break :blk .{ v, v, v };
            },
            2, 6 => .{ hi(s[0], depth), hi(s[1], depth), hi(s[2], depth) },
            4 => blk: {
                const v = hi(s[0], depth);
                break :blk .{ v, v, v };
            },
            3 => palette[@min(s[0], 255)],
            else => unreachable,
        };
        px[i * 3 ..][0..3].* = rgb;
    }
    return .{ .w = w, .h = h, .px = px };
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p: i16 = @as(i16, a) + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/// Sample k of a row at this bit depth (most significant bits first).
fn sampleAt(row: []const u8, k: usize, depth: u8) u16 {
    return switch (depth) {
        16 => (@as(u16, row[2 * k]) << 8) | row[2 * k + 1],
        8 => row[k],
        else => blk: {
            const per = 8 / depth;
            const byte = row[k / per];
            const shift: u3 = @intCast(8 - depth * (k % per + 1));
            const mask: u8 = @intCast((@as(u16, 1) << @intCast(depth)) - 1);
            break :blk (byte >> shift) & mask;
        },
    };
}

/// A colour sample to 8 bits: the high byte of a 16-bit one.
fn hi(v: u16, depth: u8) u8 {
    return if (depth == 16) @intCast(v >> 8) else @intCast(v);
}

/// A grey sample to 8 bits as PIL's modes give it: "1" (0 / 255), "L;2" (x 85), "L;4" (x 17), "L", and "I;16" whose
/// conversion to L clips at 255.
fn grey(v: u16, depth: u8) u8 {
    return switch (depth) {
        1 => if (v != 0) 255 else 0,
        2 => @intCast(v * 85),
        4 => @intCast(v * 17),
        8 => @intCast(v),
        else => @intCast(@min(v, 255)),
    };
}

// ---------------------------------------------------------------------------------------------- PIL resample

const precision_bits = 32 - 8 - 2;

fn bicubic(x0: f64) f64 {
    const a = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/// precompute_coeffs (Resample.c) with BICUBIC's support 2: each output pixel's first input pixel, its count and
/// fixed-point weights (normalize_coeffs_8bpc).
const Coeffs = struct {
    ksize: usize,
    bounds: []usize, // [out][2]: xmin, count
    k: []i32, // [out][ksize]

    fn init(a: Allocator, in_size: usize, in0: f64, in1: f64, out_size: usize) !Coeffs {
        const support0 = 2.0;
        const scale = (in1 - in0) / @as(f64, @floatFromInt(out_size));
        const filterscale = @max(scale, 1.0);
        const support = support0 * filterscale;
        const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const bounds = try a.alloc(usize, out_size * 2);
        errdefer a.free(bounds);
        const k = try a.alloc(i32, out_size * ksize);
        errdefer a.free(k);
        const kk = try a.alloc(f64, ksize);
        defer a.free(kk);
        for (0..out_size) |xx| {
            const center = in0 + (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
            const ss = 1.0 / filterscale;
            // (int)(center - support + 0.5), clamped: C truncation toward zero
            var xmin_i: i64 = @intFromFloat(@trunc(center - support + 0.5));
            if (xmin_i < 0) xmin_i = 0;
            var xmax_i: i64 = @intFromFloat(@trunc(center + support + 0.5));
            if (xmax_i > @as(i64, @intCast(in_size))) xmax_i = @intCast(in_size);
            xmax_i -= xmin_i;
            const xmin: usize = @intCast(xmin_i);
            const xmax: usize = @intCast(@max(xmax_i, 0));
            var ww: f64 = 0.0;
            for (0..xmax) |x| {
                const wv = bicubic((@as(f64, @floatFromInt(x + xmin)) - center + 0.5) * ss);
                kk[x] = wv;
                ww += wv;
            }
            for (0..xmax) |x| {
                if (ww != 0.0) kk[x] /= ww;
            }
            for (xmax..ksize) |x| kk[x] = 0;
            for (0..ksize) |x| {
                const v = kk[x] * @as(f64, @floatFromInt(@as(i64, 1) << precision_bits));
                // (int)(-0.5 + v) for negatives, (int)(0.5 + v) otherwise: truncation toward zero
                k[xx * ksize + x] = @intFromFloat(@trunc(if (kk[x] < 0) -0.5 + v else 0.5 + v));
            }
            bounds[2 * xx] = xmin;
            bounds[2 * xx + 1] = xmax;
        }
        return .{ .ksize = ksize, .bounds = bounds, .k = k };
    }

    fn deinit(c: *Coeffs, a: Allocator) void {
        a.free(c.bounds);
        a.free(c.k);
    }
};

/// clip8: the sum's integer part (an arithmetic shift) clamped to 0..255.
fn clip8(ss: i64) u8 {
    const v = ss >> precision_bits;
    return @intCast(std.math.clamp(v, 0, 255));
}

/// One BICUBIC pass over RGB rows: horizontal (input rows y0.. of the source, out w_out x h_out) or vertical.
fn passH(src: Rgb, y0: usize, out: *Rgb, c: Coeffs) void {
    for (0..out.h) |yy| {
        const row = src.px[(yy + y0) * src.w * 3 ..];
        for (0..out.w) |xx| {
            const xmin = c.bounds[2 * xx];
            const xmax = c.bounds[2 * xx + 1];
            const k = c.k[xx * c.ksize ..];
            var s: [3]i64 = @splat(@as(i64, 1) << (precision_bits - 1));
            for (0..xmax) |x| {
                for (0..3) |ch| s[ch] += @as(i64, row[(x + xmin) * 3 + ch]) * k[x];
            }
            for (0..3) |ch| out.px[(yy * out.w + xx) * 3 + ch] = clip8(s[ch]);
        }
    }
}

fn passV(src: Rgb, out: *Rgb, c: Coeffs) void {
    for (0..out.h) |yy| {
        const ymin = c.bounds[2 * yy];
        const ymax = c.bounds[2 * yy + 1];
        const k = c.k[yy * c.ksize ..];
        for (0..out.w) |xx| {
            var s: [3]i64 = @splat(@as(i64, 1) << (precision_bits - 1));
            for (0..ymax) |y| {
                for (0..3) |ch| s[ch] += @as(i64, src.px[((y + ymin) * src.w + xx) * 3 + ch]) * k[y];
            }
            for (0..3) |ch| out.px[(yy * out.w + xx) * 3 + ch] = clip8(s[ch]);
        }
    }
}

/// ImagingResampleInner with BICUBIC over the box (x0, y0, x1, y1) of the source: the horizontal pass first (on the
/// rows the vertical one reads), then the vertical one, each only when that side changes (Resample.c's need_horizontal
/// and need_vertical, compared as it compares them).
fn resampleBox(a: Allocator, src: Rgb, xsize: usize, ysize: usize, box: [4]f64) !Rgb {
    const need_h = xsize != src.w or box[0] != 0 or box[2] != @as(f64, @floatFromInt(xsize));
    const need_v = ysize != src.h or box[1] != 0 or box[3] != @as(f64, @floatFromInt(ysize));
    var ch = try Coeffs.init(a, src.w, box[0], box[2], xsize);
    defer ch.deinit(a);
    var cv = try Coeffs.init(a, src.h, box[1], box[3], ysize);
    defer cv.deinit(a);
    const ybox_first = cv.bounds[0];
    const ybox_last = cv.bounds[ysize * 2 - 2] + cv.bounds[ysize * 2 - 1];
    var cur = src;
    var owned: ?Rgb = null;
    errdefer if (owned) |*o| o.deinit(a);
    if (need_h) {
        for (0..ysize) |i| cv.bounds[i * 2] -= ybox_first;
        var tmp: Rgb = .{ .w = xsize, .h = ybox_last - ybox_first, .px = try a.alloc(u8, xsize * (ybox_last - ybox_first) * 3) };
        passH(src, ybox_first, &tmp, ch);
        owned = tmp;
        cur = tmp;
    }
    if (need_v) {
        var out: Rgb = .{ .w = cur.w, .h = ysize, .px = try a.alloc(u8, cur.w * ysize * 3) };
        passV(cur, &out, cv);
        if (owned) |*o| o.deinit(a);
        return out;
    }
    if (owned) |o| return o;
    return .{ .w = src.w, .h = src.h, .px = try a.dupe(u8, src.px) };
}

/// Image.resize(size, BICUBIC) of an RGB image (the whole source as the box), with PIL's two steps for a source more
/// than 100 times taller than wide being shortened.
pub fn resize(a: Allocator, src: Rgb, w: usize, h: usize) !Rgb {
    if (w == src.w and h == src.h) return .{ .w = w, .h = h, .px = try a.dupe(u8, src.px) };
    const fw: f64 = @floatFromInt(src.w);
    const fh: f64 = @floatFromInt(src.h);
    if (src.h > src.w * 100 and h < src.h) {
        var tall = try resampleBox(a, src, src.w, h, .{ 0, 0, fw, fh });
        defer tall.deinit(a);
        return resampleBox(a, tall, w, h, .{ 0, 0, fw, @floatFromInt(h) });
    }
    return resampleBox(a, src, w, h, .{ 0, 0, fw, fh });
}

/// ImageOps.pad(image, (w, h), color=(127, 127, 127)): contain() (the aspect kept, Python's round of the scaled side),
/// then pasted centred (Python's round of half the gap) on a grey canvas.
pub fn pad(a: Allocator, src: Rgb, w: usize, h: usize) !Rgb {
    const im_ratio = @as(f64, @floatFromInt(src.w)) / @as(f64, @floatFromInt(src.h));
    const dest_ratio = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    var cw = w;
    var chh = h;
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            const nh: usize = @intCast(pyRound(@as(f64, @floatFromInt(src.h)) / @as(f64, @floatFromInt(src.w)) * @as(f64, @floatFromInt(w))));
            if (nh != h) chh = nh;
        } else {
            const nw: usize = @intCast(pyRound(@as(f64, @floatFromInt(src.w)) / @as(f64, @floatFromInt(src.h)) * @as(f64, @floatFromInt(h))));
            if (nw != w) cw = nw;
        }
    }
    var fit = try resize(a, src, cw, chh);
    if (fit.w == w and fit.h == h) return fit;
    defer fit.deinit(a);
    const px = try a.alloc(u8, w * h * 3);
    @memset(px, 127);
    var x0: usize = 0;
    var y0: usize = 0;
    if (fit.w != w) {
        x0 = @intCast(pyRound(@as(f64, @floatFromInt(w - fit.w)) * 0.5));
    } else {
        y0 = @intCast(pyRound(@as(f64, @floatFromInt(h - fit.h)) * 0.5));
    }
    for (0..fit.h) |y| {
        @memcpy(px[((y + y0) * w + x0) * 3 ..][0 .. fit.w * 3], fit.px[y * fit.w * 3 ..][0 .. fit.w * 3]);
    }
    return .{ .w = w, .h = h, .px = px };
}

/// f32 to bf16, round to nearest even (torch's .to(torch.bfloat16)).
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7fff +% ((b >> 16) & 1)) >> 16);
}

/// The ViT's input of a padded image: n_vit_h * n_vit_w patches of [3][patch][patch] bf16, each value
/// (v / 255 - 0.5) / 0.5 in f32 (the reference's numpy / torch float32 steps).
pub fn patchify(a: Allocator, img: Rgb, p: usize) ![]u16 {
    const nh = img.h / p;
    const nw = img.w / p;
    const out = try a.alloc(u16, nh * nw * 3 * p * p);
    var lut: [256]u16 = undefined;
    for (0..256) |v| {
        const x = @as(f32, @floatFromInt(v)) / 255.0;
        lut[v] = bf16((x - 0.5) / 0.5);
    }
    for (0..nh) |ph| for (0..nw) |pw| {
        const base = (ph * nw + pw) * 3 * p * p;
        for (0..3) |c| for (0..p) |py| for (0..p) |px| {
            const y = ph * p + py;
            const x = pw * p + px;
            out[base + (c * p + py) * p + px] = lut[img.px[(y * img.w + x) * 3 + c]];
        };
    };
    return out;
}

/// A picture: its grid and its patches (owned).
pub const Picture = struct {
    grid: Grid,
    patches: []u16,

    pub fn deinit(p: *Picture, a: Allocator) void {
        a.free(p.patches);
        p.* = undefined;
    }
};

/// decode (load_image): bytes to the ViT's patches and the grids.
pub fn picture(a: Allocator, data: []const u8, cfg: Config) !Picture {
    var rgb = try decodePng(a, data);
    defer rgb.deinit(a);
    const g = plan(rgb.w, rgb.h, cfg);
    var padded = try pad(a, rgb, g.best_w, g.best_h);
    defer padded.deinit(a);
    return .{ .grid = g, .patches = try patchify(a, padded, cfg.patch) };
}

test "grids of the vision gate's images as the reference plans them" {
    const cfg: Config = .{};
    // (w, h) -> best_w, best_h, n_vit_h, n_vit_w, n_llm_h, n_llm_w (py/vfix.py on the reference)
    const cases = [_][8]usize{
        .{ 560, 400, 644, 462, 33, 46, 11, 16 },
        .{ 420, 520, 490, 616, 44, 35, 15, 12 },
        .{ 400, 300, 630, 476, 34, 45, 12, 15 },
        .{ 120, 2000, 140, 2226, 159, 10, 53, 4 },
        .{ 640, 240, 896, 336, 24, 64, 8, 22 },
        .{ 16, 16, 546, 546, 39, 39, 13, 13 },
        .{ 3000, 90, 3150, 98, 7, 225, 3, 75 },
    };
    for (cases) |c| {
        const g = plan(c[0], c[1], cfg);
        try std.testing.expectEqual(c[2], g.best_w);
        try std.testing.expectEqual(c[3], g.best_h);
        try std.testing.expectEqual(c[4], g.n_vit_h);
        try std.testing.expectEqual(c[5], g.n_vit_w);
        try std.testing.expectEqual(c[6], g.n_llm_h);
        try std.testing.expectEqual(c[7], g.n_llm_w);
    }
}

test "Python's round: half to even" {
    try std.testing.expectEqual(@as(i64, 2), pyRound(2.5));
    try std.testing.expectEqual(@as(i64, 4), pyRound(3.5));
    try std.testing.expectEqual(@as(i64, 3), pyRound(2.6));
    try std.testing.expectEqual(@as(i64, 0), pyRound(0.5));
}

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0xbf80), bf16(-1.0));
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16(@bitCast(@as(u32, 0x3f80c000)))); // above half: up
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // half, even: down
}

test "a span's token types" {
    const g: Grid = .{ .best_h = 0, .best_w = 0, .n_vit_h = 0, .n_vit_w = 0, .n_llm_h = 2, .n_llm_w = 3 };
    var t: [10]Type = undefined;
    spanTypes(g, &t);
    try std.testing.expectEqualSlices(Type, &.{ .start, .image, .image, .image, .newline, .image, .image, .image, .newline, .end }, &t);
}
