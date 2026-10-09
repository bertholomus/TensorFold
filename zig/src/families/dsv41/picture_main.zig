//! tf-dsv41-picture FILE...: the vision tower's input made of each image as the served lane makes it (picture.zig):
//! one JSON line a file with the decoded size, the grid, and sha256 of the RGB pixels, of the padded image and of the
//! bf16 patches (little-endian), for comparing with py/vfix.py's reference fixtures and the lane's `decode` points.
const std = @import("std");
const dsv41 = @import("dsv41");
const picture = dsv41.picture;

fn hex(d: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(d, .lower);
}

fn sha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return hex(d);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("usage: tf-dsv41-picture FILE...\n", .{});
        return 2;
    }
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w = &out.interface;
    var bad: u8 = 0;
    for (args[1..]) |path| {
        const data = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch |e| {
            try w.print("{{\"file\": \"{s}\", \"error\": \"{s}\"}}\n", .{ path, @errorName(e) });
            bad = 1;
            continue;
        };
        var rgb = picture.decodePng(a, data) catch |e| {
            try w.print("{{\"file\": \"{s}\", \"error\": \"{s}\"}}\n", .{ path, @errorName(e) });
            bad = 1;
            continue;
        };
        const g = picture.plan(rgb.w, rgb.h, .{});
        var padded = try picture.pad(a, rgb, g.best_w, g.best_h);
        const patches = try picture.patchify(a, padded, 14);
        try w.print("{{\"file\": \"{s}\", \"w\": {d}, \"h\": {d}, \"best_w\": {d}, \"best_h\": {d}, \"n_vit_h\": {d}, \"n_vit_w\": {d}, \"n_llm_h\": {d}, \"n_llm_w\": {d}, \"rgb_sha256\": \"{s}\", \"pad_sha256\": \"{s}\", \"patches_sha256\": \"{s}\"}}\n", .{
            path,        rgb.w,          rgb.h,                              g.best_w, g.best_h, g.n_vit_h, g.n_vit_w, g.n_llm_h, g.n_llm_w,
            sha(rgb.px), sha(padded.px), sha(std.mem.sliceAsBytes(patches)),
        });
        rgb.deinit(a);
        padded.deinit(a);
        a.free(patches);
        try w.flush();
    }
    try w.flush();
    return bad;
}
