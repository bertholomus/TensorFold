//! cuBLASLt by dlopen, for the bf16 linears with a bias the served build runs through torch (nn.Linear's addmm: torch's
//! gemm_and_bias): the vision tower's patch embed, wqkv and wo, the aligner's two linears. torch hands cuBLASLt its
//! cuBLAS handle as the Lt handle, a 1 MiB workspace, COMPUTE_32F with fp32 scale, TRANSA T (TRANSB T for an input that
//! is a transposed view), EPILOGUE_BIAS with the bias pointer, alignment preferences of each pointer, and takes the
//! first heuristic result (the recording's cuBLASLt log, level 5); the same calls on the same library pick the same
//! kernel, so the bytes are torch's.
const std = @import("std");
const cuda = @import("cuda");
const cublas = @import("cublas.zig");

pub const Status = c_int;
const Ptr = ?*anyopaque;
const S = Status;

pub const Error = error{ LibraryUnavailable, MissingSymbol, CublasLtFailed, NoAlgorithm };

const r_16bf: c_int = 14; // CUDA_R_16BF
const r_32f: c_int = 0; // CUDA_R_32F
const compute_32f: c_int = 68; // CUBLAS_COMPUTE_32F

const desc_transa: c_int = 3;
const desc_transb: c_int = 4;
const desc_epilogue: c_int = 7;
const desc_bias_pointer: c_int = 8;
const epilogue_bias: u32 = 4;

const pref_max_workspace: c_int = 1;
const pref_align_a: c_int = 5;
const pref_align_b: c_int = 6;
const pref_align_c: c_int = 7;
const pref_align_d: c_int = 8;

/// cublasLtMatmulHeuristicResult_t: the algorithm (64 bytes), its workspace, status, waves.
const Heuristic = extern struct { algo: [8]u64, workspace: usize, state: c_int, waves: f32, reserved: [4]c_int };

pub const Api = struct {
    cublasLtMatmulDescCreate: *const fn (*Ptr, c_int, c_int) callconv(.c) S,
    cublasLtMatmulDescDestroy: *const fn (Ptr) callconv(.c) S,
    cublasLtMatmulDescSetAttribute: *const fn (Ptr, c_int, *const anyopaque, usize) callconv(.c) S,
    cublasLtMatrixLayoutCreate: *const fn (*Ptr, c_int, u64, u64, i64) callconv(.c) S,
    cublasLtMatrixLayoutDestroy: *const fn (Ptr) callconv(.c) S,
    cublasLtMatmulPreferenceCreate: *const fn (*Ptr) callconv(.c) S,
    cublasLtMatmulPreferenceDestroy: *const fn (Ptr) callconv(.c) S,
    cublasLtMatmulPreferenceSetAttribute: *const fn (Ptr, c_int, *const anyopaque, usize) callconv(.c) S,
    cublasLtMatmulAlgoGetHeuristic: *const fn (Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, c_int, *Heuristic, *c_int) callconv(.c) S,
    cublasLtMatmul: *const fn (Ptr, Ptr, *const f32, u64, Ptr, u64, Ptr, *const f32, u64, Ptr, u64, Ptr, *const [8]u64, u64, usize, cuda.abi.Stream) callconv(.c) S,
};

/// torch's cuBLASLt workspace for these calls (the log's workSpaceSizeInBytes).
pub const workspace_bytes: usize = 1 << 20;

pub const Lt = struct {
    lib: std.DynLib,
    api: Api,
    handle: Ptr, // the cuBLAS handle, as torch passes it
    stream: cuda.Stream,
    workspace: u64,

    pub fn open(b: *const cublas.Blas, stream: cuda.Stream, workspace: u64) Error!Lt {
        var lib = std.DynLib.open("libcublasLt.so.13") catch std.DynLib.open("libcublasLt.so") catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        return .{ .lib = lib, .api = api, .handle = @ptrCast(b.handle), .stream = stream, .workspace = workspace };
    }

    pub fn close(l: *Lt) void {
        l.lib.close();
        l.* = undefined;
    }

    fn check(s: Status) Error!void {
        if (s != 0) {
            std.log.err("cuBLASLt status {d}", .{s});
            return error.CublasLtFailed;
        }
    }

    /// torch's _getAlignment: the largest power of two up to 256 dividing the address.
    fn alignment(p: u64) u32 {
        var a: u32 = 256;
        while (p % a != 0) a /= 2;
        return a;
    }

    /// nn.Linear of x bf16 and w bf16 [n, k] with bias b bf16 [n]: out bf16 [rows, n]. x is [rows, k] row-major
    /// (x_t false: B rows k, cols rows, ld k, TRANSB N) or a transposed view of a [k, rows] buffer (x_t true: B rows
    /// rows, cols k, ld rows, TRANSB T).
    pub fn linearBias(l: *const Lt, x: u64, x_t: bool, w: u64, b: u64, out: u64, rows: usize, k: usize, n: usize) Error!void {
        const a = &l.api;
        var pref: Ptr = null;
        try check(a.cublasLtMatmulPreferenceCreate(&pref));
        defer _ = a.cublasLtMatmulPreferenceDestroy(pref);
        var desc: Ptr = null;
        try check(a.cublasLtMatmulDescCreate(&desc, compute_32f, r_32f));
        defer _ = a.cublasLtMatmulDescDestroy(desc);
        const ta: c_int = 1;
        const tb: c_int = if (x_t) 1 else 0;
        try check(a.cublasLtMatmulDescSetAttribute(desc, desc_transa, &ta, 4));
        try check(a.cublasLtMatmulDescSetAttribute(desc, desc_transb, &tb, 4));
        try check(a.cublasLtMatmulDescSetAttribute(desc, desc_epilogue, &epilogue_bias, 4));
        try check(a.cublasLtMatmulDescSetAttribute(desc, desc_bias_pointer, &b, 8));
        var la: Ptr = null;
        try check(a.cublasLtMatrixLayoutCreate(&la, r_16bf, k, n, @intCast(k)));
        defer _ = a.cublasLtMatrixLayoutDestroy(la);
        var lb: Ptr = null;
        if (x_t) try check(a.cublasLtMatrixLayoutCreate(&lb, r_16bf, rows, k, @intCast(rows))) else try check(a.cublasLtMatrixLayoutCreate(&lb, r_16bf, k, rows, @intCast(k)));
        defer _ = a.cublasLtMatrixLayoutDestroy(lb);
        var lc: Ptr = null;
        try check(a.cublasLtMatrixLayoutCreate(&lc, r_16bf, n, rows, @intCast(n)));
        defer _ = a.cublasLtMatrixLayoutDestroy(lc);
        const ws: u64 = workspace_bytes;
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_max_workspace, &ws, 8));
        const aa = alignment(w);
        const ab = alignment(x);
        const ac = alignment(out);
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_align_a, &aa, 4));
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_align_b, &ab, 4));
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_align_c, &ac, 4));
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_align_d, &ac, 4));
        var h: Heuristic = undefined;
        var got: c_int = 0;
        try check(a.cublasLtMatmulAlgoGetHeuristic(l.handle, desc, la, lb, lc, lc, pref, 1, &h, &got));
        if (got == 0) return error.NoAlgorithm;
        const one: f32 = 1;
        const zero: f32 = 0;
        try check(a.cublasLtMatmul(l.handle, desc, &one, w, la, x, lb, &zero, out, lc, out, lc, &h.algo, l.workspace, workspace_bytes, l.stream.handle));
    }
};
