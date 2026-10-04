// Grouped EXL3 linears, 1-128 rows: up to GMAX independent layers in two launches, whatever their shapes and plans (the
// projections of one input, the slices of one attention output, or a single layer). rot_many_kernel rotates every
// layer's input rows (each its own suh; the input any of fp16 / bf16 / fp32 with any row stride) into fp16 rows, as
// rot_in does; glinear_kernel then runs every layer's blocks in one grid, each block exactly linear_kernel's block for
// its layer (its plan's K splits and warps, K ranges, mma order, warp-order and split-order sums), writing the layer's
// output rows with any row stride. So each output has the bits of that layer's own rot_in + linear launches, whatever
// rows and layers share the launch (mma keeps rows apart; nothing is summed across rows or layers).
//
// Both kernels are programmatic dependent launches where the GPU has them (sm_90+): each may start while the kernel
// before it finishes, and reads nothing but weights (glinear: its first trellis step) until that kernel is done.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstring>
#include <vector>

#include "linear_common.cuh"

using namespace tf_exl3;

namespace {

constexpr int GMAX = 8;                 // layers a launch

struct GLayer {
    const half* xh;                     // rotated input rows [M, K] fp16, row stride ldx elements
    const uint32_t* T;                  // trellis words
    const half* svh;                    // fp16 [N]
    const half* bias;                   // fp16 [N] or null
    void* y;                            // output rows [M, N], row stride ldy elements
    float* Z;                           // fp32 [SK, M, N] when SK > 1
    int* counters;                      // int32 [8 * N / 128], left zero
    long long ldx, ldy, stride_k, stride_nb;
    int y_dtype, K, N, SK, first;       // first: the layer's first block in the grid
    // the next layer's input rotation folded in (or rsuh null): its rotated rows rxh[row * ldr + roff + col] =
    // rot128_in(y as stored, rsuh[roff + col]), the bits rot_many makes of y
    const half* rsuh;
    half* rxh;
    long long ldr;
    int roff;
    // RoPE folded in (or rcos null): the outputs' last rrd columns of every rhd-column head rotated in adjacent pairs
    // by the row's position rpos[row], as rope_heads does to the stored bf16 values (its compiled fma order)
    const float* rcos;
    const float* rsin;
    const long long* rpos;
    int rhd, rrd;
};

struct GArgs {
    GLayer l[GMAX];
    int n, M;
};

struct RLayer {
    const void* x;                      // input rows [M, K], row stride ldx elements
    const half* suh;                    // fp16 [K]
    half* xh;                           // rotated rows [M, K] fp16, row stride K
    long long ldx;
    int x_dtype, K, first;              // first: the layer's first warp
};

struct RArgs {
    RLayer l[GMAX];
    int n, M;
};

// Programmatic dependent launch (sm_90+): wait() returns once the kernel before is done and its writes are visible;
// dependents() lets the next such launch start. Both are no-ops in a launch made without it.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// rot_in for every layer of a group: a warp a (layer, row, 128-wide block), rot_in's arithmetic (rot128_in).
__global__ void __launch_bounds__(128) rot_many_kernel(const __grid_constant__ RArgs args) {
    pdl_dependents();                                         // the group's linears may start: they read weights first
    pdl_wait();
    const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    int li = 0;
#pragma unroll
    for (int q = 1; q < GMAX; ++q)
        if (q < args.n && w >= args.l[q].first) li = q;
    const RLayer& L = args.l[li];
    const int nblk = L.K >> 7, local = w - L.first;
    if (local >= args.M * nblk) return;                       // past the last layer's rows (whole warps)
    const int row = local / nblk, k = (local % nblk) * 128 + 4 * lane;
    float v[4], s[4];
    load4(L.x, L.x_dtype, (size_t)row * L.ldx + k, v);
    load4(L.suh, F16, k, s);
    rot128_in(v, s, lane);
    store4(L.xh, F16, (size_t)row * L.K + k, v);
}

// The finished outputs of one row's 128 columns (4 a lane), as stored in y, rotated into the next layer's input rows:
// what rot_many computes from y (y's rounding, then rot128_in with the next layer's suh), from registers.
__device__ __forceinline__ void rot_out(const GLayer& L, const float (&v)[4], int y_dtype, int lane, size_t row,
                                        int col0) {
    float u[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        u[j] = y_dtype == BF16 ? __bfloat162float(__float2bfloat16_rn(v[j]))
             : y_dtype == F16 ? __half2float(__float2half_rn(v[j])) : v[j];
    const int c = L.roff + col0 + 4 * lane;
    load4(L.rsuh, F16, c, s);
    rot128_in(u, s, lane);
    store4(L.rxh, F16, row * L.ldr + c, u);
}

// rope_heads (forward) of one row's 128 stored columns (4 a lane) when they hold a head's rotary part: v rounded as
// stored (bf16), then re = fma(xe, cs, -(xo sn)), im = fma(xe, sn, xo cs) on each adjacent pair (rope_heads' order)
__device__ __forceinline__ void rope_out(const GLayer& L, float (&v)[4], int y_dtype, int lane, size_t row, int col0) {
    const int d0 = (col0 % L.rhd) + 4 * lane, lo = L.rhd - L.rrd;
    if (d0 < lo) return;
    const long long p = L.rpos[row];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int j = (d0 + 2 * q - lo) >> 1;
        const float cs = L.rcos[p * (L.rrd >> 1) + j], sn = L.rsin[p * (L.rrd >> 1) + j];
        const float xe = __bfloat162float(__float2bfloat16_rn(v[2 * q]));
        const float xo = __bfloat162float(__float2bfloat16_rn(v[2 * q + 1]));
        v[2 * q] = __fmaf_rn(xe, cs, -__fmul_rn(xo, sn));
        v[2 * q + 1] = __fmaf_rn(xe, sn, __fmul_rn(xo, cs));
    }
}

// linear_kernel for a block of any layer of the group (blocks of layer i: first_i .. first_i + N_i / 128 * SK_i - 1,
// block b of them linear_kernel's (b % (N_i / 128), b / (N_i / 128))).
// (4 warps: at most 170 registers, three programs an SM as linear_kernel gets, which nvcc otherwise gives up for the
// table's registers)
template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32, WK == 4 ? 3 : 1) glinear_kernel(const __grid_constant__ GArgs args) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int M = args.M;
    const int RH = min(M, 8);                                 // rows of red a warp

    int li = 0;                                               // this block's layer
#pragma unroll
    for (int q = 1; q < GMAX; ++q)
        if (q < args.n && (int)blockIdx.x >= args.l[q].first) li = q;
    const GLayer& L = args.l[li];
    const half* xh = L.xh;
    const half* svh = L.svh;
    const half* bias = L.bias;
    void* y = L.y;
    float* Z = L.Z;
    int* counters = L.counters;
    const long long ldx = L.ldx, ldy = L.ldy, stride_k = L.stride_k;
    const int y_dtype = L.y_dtype, K = L.K, N = L.N, SK = L.SK;
    const int NB = N >> 7;
    const int b = (int)blockIdx.x - L.first;
    const int nb = b % NB, split = b / NB;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    // the epilogue's tables (weights, read last): svh, the bias and the folded rotation's suh lines into L2 now (no
    // data moved, nothing computed)
    if (threadIdx.x < 2 && (SK == 1 || split == SK - 1)) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(svh + nb * 128 + threadIdx.x * 64));
        if (bias) asm volatile("prefetch.global.L2 [%0];" ::"l"(bias + nb * 128 + threadIdx.x * 64));
        if (L.rsuh) asm volatile("prefetch.global.L2 [%0];" ::"l"(L.rsuh + L.roff + nb * 128 + threadIdx.x * 64));
    }
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = L.T + nb * L.stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * ldx;
        const half* x1 = xh + (size_t)r1 * ldx;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
        if (m0 == 0) {
            pdl_wait();                              // the first weights are on their way; the rows are ready now
            pdl_dependents();
            if (L.rcos && threadIdx.x < M && threadIdx.x < 16) {   // the folded RoPE's rows of the tables, into L2
                const long long p = L.rpos[threadIdx.x];
                asm volatile("prefetch.global.L2 [%0];" ::"l"(L.rcos + p * (L.rrd >> 1)));
                asm volatile("prefetch.global.L2 [%0];" ::"l"(L.rsin + p * (L.rrd >> 1)));
            }
        }
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    if (L.rcos) rope_out(L, v, y_dtype, lane, (size_t)(m0 + rlo + r), col0);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * ldy + col0 + 4 * lane, v);
                    if (L.rsuh) rot_out(L, v, y_dtype, lane, (size_t)(m0 + rlo + r), col0);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    if (L.rcos) rope_out(L, v, y_dtype, lane, (size_t)m0 + r, col0);
                    store4(y, y_dtype, ((size_t)m0 + r) * ldy + col0 + 4 * lane, v);
                    if (L.rsuh) rot_out(L, v, y_dtype, lane, (size_t)m0 + r, col0);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

int gdtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

// A launch, as a programmatic dependent launch when asked and the GPU has it (sm_90+).
template <typename Kernel, typename Args>
void launch(Kernel kernel, unsigned blocks, unsigned threads, int smem, cudaStream_t stream, bool pdl, const Args& a) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = (size_t)smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (pdl) {
        int dev = 0, major = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
        if (major >= 9) {
            cfg.attrs = attr;
            cfg.numAttrs = 1;
        }
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, a));
}

}  // namespace

int exl3_glinear_max() { return GMAX; }

void exl3_rot_many_cuda(const std::vector<at::Tensor>& xs, const std::vector<at::Tensor>& suhs,
                        const std::vector<at::Tensor>& xhs, bool pdl) {
    RArgs a;
    std::memset(&a, 0, sizeof(a));
    a.n = (int)xs.size();
    a.M = (int)xs[0].size(0);
    int warps = 0;
    for (int i = 0; i < a.n; ++i) {
        RLayer& l = a.l[i];
        l.x = xs[i].data_ptr();
        l.x_dtype = gdtype_of(xs[i]);
        l.ldx = xs[i].stride(0);
        l.suh = reinterpret_cast<const half*>(suhs[i].data_ptr());
        l.xh = reinterpret_cast<half*>(xhs[i].data_ptr());
        l.K = (int)xs[i].size(1);
        l.first = warps;
        warps += a.M * (l.K / 128);
    }
    launch(rot_many_kernel, (unsigned)((warps + 3) / 4), 128, 0, at::cuda::getCurrentCUDAStream(), pdl, a);
}

void exl3_glinear_cuda(const std::vector<at::Tensor>& xhs, const std::vector<at::Tensor>& Ts,
                       const std::vector<int64_t>& stride_k, const std::vector<int64_t>& stride_nb,
                       const std::vector<at::Tensor>& svhs, const std::vector<c10::optional<at::Tensor>>& biases,
                       const std::vector<at::Tensor>& ys, const c10::optional<at::Tensor>& Z,
                       const std::vector<at::Tensor>& counters, const std::vector<int64_t>& SK, int64_t K2,
                       int64_t cb, int64_t WK, bool pdl, const std::vector<c10::optional<at::Tensor>>& rsuhs,
                       const std::vector<c10::optional<at::Tensor>>& rxhs, const std::vector<int64_t>& roffs,
                       const std::vector<int64_t>& ropes, const c10::optional<at::Tensor>& rcos,
                       const c10::optional<at::Tensor>& rsin, const c10::optional<at::Tensor>& rpos, int64_t rhd,
                       int64_t rrd) {
    const int n = (int)xhs.size();
    GArgs a;
    std::memset(&a, 0, sizeof(a));
    a.n = n;
    a.M = (int)xhs[0].size(0);
    float* zp = Z ? Z->data_ptr<float>() : nullptr;
    size_t zoff = 0;
    int blocks = 0;
    for (int i = 0; i < n; ++i) {
        GLayer& l = a.l[i];
        l.xh = reinterpret_cast<const half*>(xhs[i].data_ptr());
        l.ldx = xhs[i].stride(0);
        l.T = reinterpret_cast<const uint32_t*>(Ts[i].data_ptr());
        l.stride_k = stride_k[i];
        l.stride_nb = stride_nb[i];
        l.svh = reinterpret_cast<const half*>(svhs[i].data_ptr());
        l.bias = biases[i] ? reinterpret_cast<const half*>(biases[i]->data_ptr()) : nullptr;
        l.y = ys[i].data_ptr();
        l.y_dtype = gdtype_of(ys[i]);
        l.ldy = ys[i].stride(0);
        l.K = (int)xhs[i].size(1);
        l.N = (int)ys[i].size(1);
        l.SK = (int)SK[i];
        l.counters = counters[i].data_ptr<int>();
        l.Z = nullptr;
        if (SK[i] > 1) {
            l.Z = zp + zoff;
            zoff += (size_t)SK[i] * a.M * l.N;
        }
        l.first = blocks;
        blocks += (l.N / 128) * l.SK;
        if (i < (int)rsuhs.size() && rsuhs[i] && rxhs[i]) {
            l.rsuh = reinterpret_cast<const half*>(rsuhs[i]->data_ptr());
            l.rxh = reinterpret_cast<half*>(rxhs[i]->data_ptr());
            l.ldr = rxhs[i]->stride(0);
            l.roff = (int)roffs[i];
        }
        if (i < (int)ropes.size() && ropes[i] && rcos && rsin && rpos) {
            l.rcos = rcos->data_ptr<float>();
            l.rsin = rsin->data_ptr<float>();
            l.rpos = reinterpret_cast<const long long*>(rpos->data_ptr<int64_t>());
            l.rhd = (int)rhd;
            l.rrd = (int)rrd;
        }
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const int smem = (int)(WK * std::min(a.M, 8) * 128 * sizeof(float));    // at most 32 KiB: no opt-in
#define TF_LAUNCH(K2_, CB_)                                                                                        \
    if (K2 == K2_ && cb == CB_) {                                                                               \
        launch(WK == 4 ? glinear_kernel<K2_, CB_, 4> : glinear_kernel<K2_, CB_, 8>, (unsigned)blocks,          \
               (unsigned)(WK * 32), smem, stream, pdl, a);                                                      \
        return;                                                                                                 \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}
