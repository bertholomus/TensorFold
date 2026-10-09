//! cuBLAS (not Lt) by dlopen, for the fp32 GEMMs the served build runs through torch.matmul: the MoE gate's logits and
//! the indexer's weights of prompt chunks, the DSpark confidence head. torch calls cublasSgemm_v2 on one handle with
//! CUBLAS_DEFAULT_MATH and a 32 MiB workspace (the recording's cuBLAS log); the same call on the same library picks the
//! same algorithm.
const std = @import("std");
const cuda = @import("cuda");

pub const Status = c_int;
pub const Handle = ?*opaque {};
pub const Op = enum(c_int) { n = 0, t = 1 };
pub const default_math: c_int = 0;

pub const Error = error{ LibraryUnavailable, MissingSymbol, CublasFailed };

const S = Status;
const D = u64;

pub const Api = struct {
    cublasCreate_v2: *const fn (*Handle) callconv(.c) S,
    cublasDestroy_v2: *const fn (Handle) callconv(.c) S,
    cublasSetStream_v2: *const fn (Handle, cuda.abi.Stream) callconv(.c) S,
    cublasSetWorkspace_v2: *const fn (Handle, D, usize) callconv(.c) S,
    cublasSetMathMode: *const fn (Handle, c_int) callconv(.c) S,
    cublasSgemm_v2: *const fn (Handle, Op, Op, c_int, c_int, c_int, *const f32, D, c_int, D, c_int, *const f32, D, c_int) callconv(.c) S,
    cublasGemmEx: *const fn (Handle, Op, Op, c_int, c_int, c_int, *const anyopaque, D, c_int, c_int, D, c_int, c_int, *const anyopaque, D, c_int, c_int, c_int, c_int) callconv(.c) S,
};

/// One handle bound to a stream, with torch's workspace size and math mode.
pub const Blas = struct {
    lib: std.DynLib,
    api: Api,
    handle: Handle = null,

    /// torch's default cuBLAS workspace on this GPU (the recorded cublasSetWorkspace_v2 size).
    pub const workspace_bytes: usize = 32 << 20;

    pub fn open(stream: cuda.Stream, workspace: u64) Error!Blas {
        var lib = std.DynLib.open("libcublas.so.13") catch std.DynLib.open("libcublas.so") catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        var b: Blas = .{ .lib = lib, .api = api };
        try b.check(api.cublasCreate_v2(&b.handle));
        try b.check(api.cublasSetStream_v2(b.handle, stream.handle));
        try b.check(api.cublasSetWorkspace_v2(b.handle, workspace, workspace_bytes));
        try b.check(api.cublasSetMathMode(b.handle, default_math));
        return b;
    }

    pub fn close(b: *Blas) void {
        _ = b.api.cublasDestroy_v2(b.handle);
        b.lib.close();
        b.* = undefined;
    }

    fn check(_: *const Blas, s: Status) Error!void {
        if (s != 0) {
            std.log.err("cuBLAS status {d}", .{s});
            return error.CublasFailed;
        }
    }

    /// nn.Linear without a bias of x bf16 [rows, k] and w bf16 [n, k]: out bf16 [rows, n], as torch issues it
    /// (cublasGemmEx: transa T, transb N, m = n, n = rows, lda = ldb = k, ldc = n, fp32 alpha 1 and beta 0, bf16
    /// operands, COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP; the recording's cuBLAS log for the vision MLP).
    pub fn linearBf16(b: *const Blas, x: u64, w: u64, out: u64, rows: usize, k: usize, n: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        const bf16_type: c_int = 14; // CUDA_R_16BF
        try b.check(b.api.cublasGemmEx(b.handle, .t, .n, @intCast(n), @intCast(rows), @intCast(k), &one, w, bf16_type, @intCast(k), x, bf16_type, @intCast(k), &zero, out, bf16_type, @intCast(n), 68, 99));
    }

    /// torch.matmul of x [rows, k] fp32 (row-major) and w [n, k] fp32 transposed: out [rows, n] fp32, as torch issues
    /// it (column-major C^T = W x^T: transa T, transb N, m = n, n = rows, lda = ldb = k, ldc = n, alpha 1, beta 0).
    pub fn xwT(b: *const Blas, x: u64, w: u64, out: u64, rows: usize, k: usize, n: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try b.check(b.api.cublasSgemm_v2(b.handle, .t, .n, @intCast(n), @intCast(rows), @intCast(k), &one, w, @intCast(k), x, @intCast(k), &zero, out, @intCast(n)));
    }

    /// torch.matmul of x [rows, k] fp32 (row-major) and v [1, k] fp32 transposed (the DSpark confidence head): out [rows]
    /// fp32. A [k, 1] column is contiguous by torch's rules (a size-1 dimension's stride is not checked), so torch
    /// issues C^T = v^T x^T untransposed: transa N, transb N, m = 1, n = rows, lda = 1, ldb = k, ldc = 1 (cuBLAS runs it
    /// as a batched dot product).
    pub fn xv(b: *const Blas, x: u64, v: u64, out: u64, rows: usize, k: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try b.check(b.api.cublasSgemm_v2(b.handle, .n, .n, 1, @intCast(rows), @intCast(k), &one, v, 1, x, @intCast(k), &zero, out, 1));
    }
};
