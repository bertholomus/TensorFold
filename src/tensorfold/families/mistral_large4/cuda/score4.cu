// score8.cu's scores for a deep prompt's rounds in FP4, the MMA's own block scale over 16 values: twice its rate. A row of q or of a
// key is 80 bytes: 64 of codes (two a byte, the first in the low half), 8 ue4m3 scales of 16 values, at 72 a float: the row's scale (for
// q, w times it). Everything else is score8.cu's: score[r, t] = sum_h w[r, h] relu(q[r, h] . k[t]), -inf from vis[r] on, the largest of
// every eight scores for select_step's top8.
// A block holds 128 keys in registers as B fragments and takes 64 queries at a time, the heads going by four through two shared-memory
// stages; two blocks an SM, so that one's barriers and stores run beside the other's MMAs.

#ifndef SCORE4_ONLY
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int IH = 32, ID = 128;                                  // heads, dims
constexpr int ROW = 80, SCALES = 64, FACTOR = 72;                 // a row's bytes, where its block scales and its float are
constexpr int QB = 64, BN = 128, T = 128;                         // queries, keys, threads a block
constexpr int HS = 4, HEAD = QB * ROW, STAGE = HS * HEAD;         // a stage: four heads of a query tile
constexpr int MAX_TILES = 64;                                     // query tiles a block
constexpr int GROUP = 8;                                          // an MMA's columns
constexpr int KS = ID / 64;                                       // k steps of a head
constexpr int RING = 2 * STAGE > BN * ROW ? 2 * STAGE : BN * ROW; // the keys go through it first
constexpr int SMEM = RING + BN * 4;
__device__ constexpr float TRY[8] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f, 0.7f, 1.1f};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
    asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sb), "h"((uint16_t)0),
          "h"((uint16_t)0));
}
// Four values to two bytes of codes, the first in the low half.
__device__ __forceinline__ uint16_t narrow4(float a, float b, float c, float d) {
    uint16_t r;
    asm("{ .reg .b8 lo, hi; cvt.rn.satfinite.e2m1x2.f32 lo, %2, %1; cvt.rn.satfinite.e2m1x2.f32 hi, %4, %3; mov.b16 %0, {lo, hi}; }"
        : "=h"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
    return r;
}
// narrow4 back.
__device__ __forceinline__ void wide4(uint16_t two, float (&v)[4]) {
    uint32_t lo, hi;
    asm("{ .reg .b8 a, b; mov.b16 {a, b}, %2; cvt.rn.f16x2.e2m1x2 %0, a; cvt.rn.f16x2.e2m1x2 %1, b; }" : "=r"(lo), "=r"(hi) : "h"(two));
    const float2 x = __half22float2(*reinterpret_cast<__half2*>(&lo)), y = __half22float2(*reinterpret_cast<__half2*>(&hi));
    v[0] = x.x;
    v[1] = x.y;
    v[2] = y.x;
    v[3] = y.y;
}
// A value in [0, 448] as ue4m3, and what that code means.
__device__ __forceinline__ uint32_t scale_code(float x, float& back) {
    uint16_t two;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(two) : "f"(0.0f), "f"(x));
    uint32_t wide;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(wide) : "h"(two));
    back = __half2float(__low2half(*reinterpret_cast<__half2*>(&wide)));
    return two & 0xff;
}

// A warp a row of 128 fp16: its 80 bytes, the row's scale times W[row] when there is a W.
__global__ void __launch_bounds__(T) quant4_kernel(const __half* __restrict__ X, const float* __restrict__ W, uint8_t* __restrict__ OUT, int total) {
    const int row = blockIdx.x * (T / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= total) return;
    const uint2 v = *reinterpret_cast<const uint2*>(X + (int64_t)row * ID + lane * 4);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
    float block = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
    block = fmaxf(block, __shfl_xor_sync(~0u, block, 1));
    block = fmaxf(block, __shfl_xor_sync(~0u, block, 2));
    float amax = fmaxf(block, __shfl_xor_sync(~0u, block, 4));
    amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, 8));
    amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, 16));
    const float factor = amax / (6.0f * 448.0f), fit = amax > 0.0f ? block * (448.0f / amax) : 0.0f;
    float best = INFINITY;
    uint32_t code = 0;
    uint16_t codes = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {                                 // the block's scale with the least error: a little under its largest value, mostly
        float back;
        const uint32_t c = scale_code(fit * TRY[i], back);
        const float inv = back > 0.0f ? 1.0f / (back * factor) : 0.0f, at = back * factor;
        const uint16_t two = narrow4(a.x * inv, a.y * inv, b.x * inv, b.y * inv);
        float y[4];
        wide4(two, y);
        const float d0 = a.x - y[0] * at, d1 = a.y - y[1] * at, d2 = b.x - y[2] * at, d3 = b.y - y[3] * at;
        float e = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        e += __shfl_xor_sync(~0u, e, 1);
        e += __shfl_xor_sync(~0u, e, 2);
        if (e < best) {
            best = e;
            code = c;
            codes = two;
        }
    }
    uint8_t* out = OUT + (int64_t)row * ROW;
    *reinterpret_cast<uint16_t*>(out + lane * 2) = codes;
    if ((lane & 3) == 0) out[SCALES + (lane >> 2)] = code;
    if (lane == 0) *reinterpret_cast<float*>(out + FACTOR) = W ? W[row] * factor : factor;
}

__global__ void __launch_bounds__(T, 2) score4_kernel(const uint8_t* __restrict__ Q4, const uint8_t* __restrict__ K4, const int* __restrict__ VIS,
                                                      float* __restrict__ OUT, float* __restrict__ TOP, int n, int rows, int64_t ldo, int64_t ldt) {
    extern __shared__ __align__(16) uint8_t base[];
    uint8_t* ks = base;                                           // [BN][ROW], then the stages
    float* kscale = reinterpret_cast<float*>(base + RING);
    __shared__ int flags[MAX_TILES], active[MAX_TILES], count;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wq = warp >> 1, wk = warp & 1;
    const int g = lane >> 2, t2 = (lane & 3) * 2;
    const int t0 = blockIdx.x * BN;

    // the query tiles with a row that sees these keys; the others are not read
    const int mine = ((rows + QB - 1) / QB - blockIdx.y + gridDim.y - 1) / gridDim.y;
    for (int j = warp; j < mine; j += T / 32) {
        const int r = (blockIdx.y + j * gridDim.y) * QB;
        int v = r + lane < rows ? VIS[r + lane] : 0;
        if (r + 32 + lane < rows) v = max(v, VIS[r + 32 + lane]);
        const bool on = __any_sync(~0u, v > t0);
        if (lane == 0) flags[j] = on;
    }
    __syncthreads();
    if (warp == 0) {
        int at = 0;
        for (int j0 = 0; j0 < mine; j0 += 32) {
            const bool on = j0 + lane < mine && flags[j0 + lane];
            const uint32_t bits = __ballot_sync(~0u, on);
            if (on) active[at + __popc(bits & ((1u << lane) - 1))] = blockIdx.y + (j0 + lane) * gridDim.y;
            at += __popc(bits);
        }
        if (lane == 0) count = at;
    }
    __syncthreads();
    if (!count) return;

    for (int c = tid; c < BN * ROW / 16; c += T) {
        const int key = t0 + c / (ROW / 16);
        cp_async16(ks + c * 16, K4 + (int64_t)(key < n ? key : 0) * ROW + (c % (ROW / 16)) * 16, key < n);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    if (tid < BN) kscale[tid] = *reinterpret_cast<const float*>(ks + tid * ROW + FACTOR);

    uint32_t b[8][KS][2], sb[8][KS];                              // the warp's 64 keys x 128 dims, a lane 4g's the scales of key g of eight
    {
        const int row = (lane & 7) + ((lane >> 4) << 3), col = ((lane >> 3) & 1) << 4;
#pragma unroll
        for (int nn = 0; nn < 8; nn += 2)
#pragma unroll
            for (int k = 0; k < KS; k++)
                ldsm4(b[nn][k][0], b[nn][k][1], b[nn + 1][k][0], b[nn + 1][k][1], ks + (wk * 64 + nn * 8 + row) * ROW + k * 32 + col);
#pragma unroll
        for (int nn = 0; nn < 8; nn++)
#pragma unroll
            for (int k = 0; k < KS; k++) sb[nn][k] = *reinterpret_cast<const uint32_t*>(ks + (wk * 64 + nn * 8 + g) * ROW + SCALES + k * 4);
    }
    float acc[2][8][4];
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int nn = 0; nn < 8; nn++) acc[m][nn][0] = acc[m][nn][1] = acc[m][nn][2] = acc[m][nn][3] = 0.0f;

    __syncthreads();                                              // the keys are read: the stages may land on them
    const int stages = count * (IH / HS);
    auto stage = [&](int st) {
        const int r0 = active[st / (IH / HS)] * QB, h0 = st % (IH / HS) * HS;
        uint8_t* dst = base + (st & 1) * STAGE;
#pragma unroll
        for (int i = 0; i < HS * QB * ROW / 16 / T; i++) {
            const int c = tid + i * T, hh = c / (QB * ROW / 16), row = (c / (ROW / 16)) % QB, part = c % (ROW / 16);
            const bool ok = r0 + row < rows;
            cp_async16(dst + hh * HEAD + row * ROW + part * 16, Q4 + ((int64_t)(ok ? r0 + row : 0) * IH + h0 + hh) * ROW + part * 16, ok);
        }
        cp_async_commit();
    };
    stage(0);
    const int arow = wq * 32 + (lane & 7) + (((lane >> 3) & 1) << 3), acol = (lane >> 4) << 4;
    const int srow = wq * 32 + g + ((lane & 1) << 3);             // a lane 4g + 1 gives row g + 8's scales
    for (int st = 0; st < stages; st++) {
        cp_async_wait_all();
        __syncthreads();
        if (st + 1 < stages) stage(st + 1);
#pragma unroll 1
        for (int hh = 0; hh < HS; hh++) {
            const uint8_t* qh = base + (st & 1) * STAGE + hh * HEAD;
            uint32_t a[2][KS][4], sa[2][KS];
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int k = 0; k < KS; k++) {
                    ldsm4(a[m][k][0], a[m][k][1], a[m][k][2], a[m][k][3], qh + (arow + m * 16) * ROW + k * 32 + acol);
                    sa[m][k] = *reinterpret_cast<const uint32_t*>(qh + (srow + m * 16) * ROW + SCALES + k * 4);
                }
            float w[2][2];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                w[m][0] = *reinterpret_cast<const float*>(qh + (wq * 32 + m * 16 + g) * ROW + FACTOR);
                w[m][1] = *reinterpret_cast<const float*>(qh + (wq * 32 + m * 16 + g + 8) * ROW + FACTOR);
            }
#pragma unroll
            for (int nn = 0; nn < 8; nn += 2) {                   // four sums at a time: the mma of one wait on none of the others
                float s[2][2][4] = {};
#pragma unroll
                for (int k = 0; k < KS; k++)
#pragma unroll
                    for (int m = 0; m < 2; m++)
#pragma unroll
                        for (int j = 0; j < 2; j++) mma(s[m][j], a[m][k], b[nn + j][k][0], b[nn + j][k][1], sa[m][k], sb[nn + j][k]);
#pragma unroll
                for (int m = 0; m < 2; m++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        acc[m][nn + j][0] = fmaf(fmaxf(s[m][j][0], 0.0f), w[m][0], acc[m][nn + j][0]);
                        acc[m][nn + j][1] = fmaf(fmaxf(s[m][j][1], 0.0f), w[m][0], acc[m][nn + j][1]);
                        acc[m][nn + j][2] = fmaf(fmaxf(s[m][j][2], 0.0f), w[m][1], acc[m][nn + j][2]);
                        acc[m][nn + j][3] = fmaf(fmaxf(s[m][j][3], 0.0f), w[m][1], acc[m][nn + j][3]);
                    }
            }
        }
        if ((st + 1) % (IH / HS)) continue;

        const int r0 = active[st / (IH / HS)] * QB;
#pragma unroll
        for (int m = 0; m < 2; m++) {
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const int r = r0 + wq * 32 + m * 16 + g + half * 8;
                const int vis = r < rows ? VIS[r] : 0;
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    const int at = wk * 64 + nn * 8 + t2, key = t0 + at;
                    const float x = key < vis ? acc[m][nn][half * 2] * kscale[at] : -INFINITY;
                    const float y = key + 1 < vis ? acc[m][nn][half * 2 + 1] * kscale[at + 1] : -INFINITY;
                    float top = fmaxf(x, y);
                    top = fmaxf(top, __shfl_xor_sync(~0u, top, 1));
                    top = fmaxf(top, __shfl_xor_sync(~0u, top, 2));
                    if (r < rows) {
                        if (key < n) *reinterpret_cast<float2*>(OUT + r * ldo + key) = make_float2(x, y);
                        if ((lane & 3) == (nn & 3) && key < n) TOP[r * ldt + (t0 + wk * 64 + nn * 8) / GROUP] = top;
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
            for (int nn = 0; nn < 8; nn++) acc[m][nn][0] = acc[m][nn][1] = acc[m][nn][2] = acc[m][nn][3] = 0.0f;
    }
}

#ifndef SCORE4_ONLY
// x [R, 128] fp16 (R a multiple of nothing; rows of q go head by head), w [R] fp32 or empty -> out [R, 80] bytes
void quant4(const at::Tensor& x, const c10::optional<at::Tensor>& w, at::Tensor out) {
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && x.size(-1) == ID && out.size(-1) == ROW && (!w.has_value() || w->is_contiguous()));
    const int total = x.numel() / ID;
    if (!total) return;
    quant4_kernel<<<(total + T / 32 - 1) / (T / 32), T, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr()), w.has_value() ? w->data_ptr<float>() : nullptr, out.data_ptr<uint8_t>(), total);
}

// rows queries from query ``first`` (a multiple of 64) of q4 [Q, 32, 80]; keys [>= n, 80], vis [rows] -> out [rows, >= n] at an even stride,
// top [rows, >= n / 8]
void score4(const at::Tensor& q4, int64_t first, int64_t rows, const at::Tensor& keys, const at::Tensor& vis, int64_t n, at::Tensor out, at::Tensor top) {
    static bool ready = false;
    if (!ready) {
        cudaFuncSetAttribute(score4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        ready = true;
    }
    TORCH_CHECK(q4.size(1) == IH && q4.size(2) == ROW && q4.is_contiguous() && keys.size(1) == ROW && keys.stride(0) == ROW && first % QB == 0);
    TORCH_CHECK(out.stride(1) == 1 && out.stride(0) % 2 == 0 && top.stride(1) == 1);
    if (!rows || !n) return;
    const int tiles = (rows + QB - 1) / QB, split = (tiles + MAX_TILES - 1) / MAX_TILES;
    const int ktiles = (n + BN - 1) / BN;
    int gy = split;
    while (gy < tiles && (int64_t)ktiles * gy < 192) gy++;        // four blocks an SM to go round, for the tail
    score4_kernel<<<dim3(ktiles, gy), T, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        q4.data_ptr<uint8_t>() + first * IH * ROW, keys.data_ptr<uint8_t>(), vis.data_ptr<int>(), out.data_ptr<float>(), top.data_ptr<float>(),
        (int)n, (int)rows, out.stride(0), top.stride(0));
}
#else
void* use[] = {(void*)quant4_kernel, (void*)score4_kernel};
#endif

}  // namespace

#ifndef SCORE4_ONLY
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant4", &quant4);
    m.def("score4", &score4);
}
#endif
