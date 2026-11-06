// The indexer's scores of a prompt round in e4m3: score[r, t] = sum_h w[r, h] relu(q[r, h] . k[t]) (cp4_kernels._score_tile's job at
// twice its MMA rate), -inf from vis[r] on, and the largest of every eight scores for select_step's top8. q goes in with a scale a
// head, a key with a scale of its own, both by the row's largest value; the scales are positive, so they pass through the relu: the
// head's goes into w, the key's multiplies the sum at the end.
// A block holds 128 keys, narrowed once and kept in registers as B fragments, and takes 64 queries at a time, the heads going by
// two at a time through two shared-memory stages (loading A straight to registers cost 30 %: the warps read the same rows). Two blocks
// an SM: one's keys, barriers and stores run beside the other's MMAs. The sums stay in registers until the store.

#ifndef SCORE8_ONLY
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int IH = 32, ID = 128;                                  // heads, dims
constexpr int QB = 64, BN = 128, T = 128;                         // queries, keys, threads a block
constexpr int LD = ID + 16;                                       // bytes a staged row: ldmatrix rows fall in distinct banks
constexpr int HS = 2, HEAD = QB * LD + QB * 4, STAGE = HS * HEAD; // a stage: two heads of a query tile and their weights
constexpr int MAX_TILES = 64;                                     // query tiles a block
constexpr int GROUP = 8;                                          // an MMA's columns
constexpr int KS = ID / 32;                                       // k steps of a head
constexpr int RING = 2 * STAGE > BN * LD ? 2 * STAGE : BN * LD;   // the keys go through it first
constexpr int SMEM = RING + BN * 4;

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
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Four values to four e4m3 codes, the first in the low byte.
__device__ __forceinline__ uint32_t narrow4(float a, float b, float c, float d) {
    uint32_t r;
    asm("{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1; cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3; mov.b32 %0, {lo, hi}; }"
        : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
    return r;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, d));
    return v;
}
// A row of 128 fp16 across a warp: its four e4m3 codes in this lane, and the scale of the row.
__device__ __forceinline__ uint32_t narrow_row(uint2 v, float& scale) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
    const float amax = warp_max(fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
    const float inv = amax > 0.0f ? 448.0f / amax : 0.0f;
    scale = amax / 448.0f;
    return narrow4(a.x * inv, a.y * inv, b.x * inv, b.y * inv);
}

// A warp a head of a query: its values as e4m3, and w times the scale at QW[head, query].
__global__ void __launch_bounds__(T) quant_q_kernel(const __half* __restrict__ Q, const float* __restrict__ W, uint8_t* __restrict__ Q8,
                                                    float* __restrict__ QW, int total, int64_t ldw) {
    const int row = blockIdx.x * (T / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= total) return;
    float scale;
    const uint32_t code = narrow_row(*reinterpret_cast<const uint2*>(Q + (int64_t)row * ID + lane * 4), scale);
    *reinterpret_cast<uint32_t*>(Q8 + (int64_t)row * ID + lane * 4) = code;
    if (lane == 0) QW[(row % IH) * ldw + row / IH] = W[row] * scale;
}

__global__ void __launch_bounds__(T, 2) score8_kernel(const uint8_t* __restrict__ Q8, const float* __restrict__ QW, int64_t ldw,
                                                      const __half* __restrict__ K, const int* __restrict__ VIS, float* __restrict__ OUT,
                                                      float* __restrict__ TOP, int n, int rows, int64_t ldo, int64_t ldt) {
    extern __shared__ __align__(16) uint8_t base[];
    uint8_t* ks = base;                                           // [BN][LD], then the stages
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

#pragma unroll 4
    for (int i = 0; i < BN / (T / 32); i++) {                     // a warp a key at a time
        const int row = warp * (BN / (T / 32)) + i, key = t0 + row;
        const uint2 v = key < n ? *reinterpret_cast<const uint2*>(K + (int64_t)key * ID + lane * 4) : make_uint2(0, 0);
        float scale;
        const uint32_t code = narrow_row(v, scale);
        *reinterpret_cast<uint32_t*>(ks + row * LD + lane * 4) = code;
        if (lane == 0) kscale[row] = scale;
    }
    __syncthreads();

    uint32_t b[8][4][2];                                          // the warp's 64 keys x 128 dims
    {
        const int row = (lane & 7) + ((lane >> 4) << 3), col = ((lane >> 3) & 1) << 4;
#pragma unroll
        for (int nn = 0; nn < 8; nn += 2)
#pragma unroll
            for (int k = 0; k < 4; k++)
                ldsm4(b[nn][k][0], b[nn][k][1], b[nn + 1][k][0], b[nn + 1][k][1], ks + (wk * 64 + nn * 8 + row) * LD + k * 32 + col);
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
        for (int i = 0; i < HS * QB * ID / 16 / T; i++) {
            const int c = tid + i * T, hh = c / (QB * ID / 16), row = (c / (ID / 16)) % QB, part = c % (ID / 16);
            const bool ok = r0 + row < rows;
            cp_async16(dst + hh * HEAD + row * LD + part * 16, Q8 + ((int64_t)(ok ? r0 + row : 0) * IH + h0 + hh) * ID + part * 16, ok);
        }
        if (tid < HS * QB / 4) {
            const int hh = tid / (QB / 4), part = tid % (QB / 4);
            cp_async16(dst + hh * HEAD + QB * LD + part * 16, QW + (h0 + hh) * ldw + r0 + part * 4, r0 + part * 4 < rows);
        }
        cp_async_commit();
    };
    stage(0);
    const int arow = wq * 32 + (lane & 7) + (((lane >> 3) & 1) << 3), acol = (lane >> 4) << 4;
    for (int st = 0; st < stages; st++) {
        cp_async_wait_all();
        __syncthreads();
        if (st + 1 < stages) stage(st + 1);
        for (int hh = 0; hh < HS; hh++) {
            const uint8_t* qh = base + (st & 1) * STAGE + hh * HEAD;
            const float* wh = reinterpret_cast<const float*>(qh + QB * LD);
            uint32_t a[2][KS][4];
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int k = 0; k < KS; k++)
                    ldsm4(a[m][k][0], a[m][k][1], a[m][k][2], a[m][k][3], qh + (arow + m * 16) * LD + k * 32 + acol);
            float w[2][2];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                w[m][0] = wh[wq * 32 + m * 16 + g];
                w[m][1] = wh[wq * 32 + m * 16 + g + 8];
            }
#pragma unroll
            for (int nn = 0; nn < 8; nn += 2) {                   // four sums at a time: the mma of one wait on none of the others
                float s[2][2][4] = {};
#pragma unroll
                for (int k = 0; k < KS; k++)
#pragma unroll
                    for (int m = 0; m < 2; m++)
#pragma unroll
                        for (int j = 0; j < 2; j++) mma(s[m][j], a[m][k], b[nn + j][k][0], b[nn + j][k][1]);
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

#ifndef SCORE8_ONLY
// q [Q, 32, 128] fp16, w [Q, 32] fp32 -> q8 [Q, 32, 128] the same in e4m3 bytes, qw [32, >= Q rounded up to 4] w times a head's scale
void quant_q(const at::Tensor& q, const at::Tensor& w, at::Tensor q8, at::Tensor qw) {
    TORCH_CHECK(q.is_contiguous() && w.is_contiguous() && q8.is_contiguous() && qw.stride(1) == 1 && qw.stride(0) % 4 == 0);
    const int total = q.numel() / ID;
    if (!total) return;
    quant_q_kernel<<<(total + T / 32 - 1) / (T / 32), T, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr()), w.data_ptr<float>(), q8.data_ptr<uint8_t>(), qw.data_ptr<float>(), total, qw.stride(0));
}

// rows queries from query ``first`` (a multiple of 64) of q8, qw; keys [>= n, 128] fp16, vis [rows] -> out [rows, >= n] at an even
// stride, top [rows, >= n / 8]
void score8(const at::Tensor& q8, const at::Tensor& qw, int64_t first, int64_t rows, const at::Tensor& keys, const at::Tensor& vis, int64_t n,
            at::Tensor out, at::Tensor top) {
    static bool ready = false;
    if (!ready) {
        cudaFuncSetAttribute(score8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        ready = true;
    }
    TORCH_CHECK(q8.size(1) == IH && q8.size(2) == ID && q8.is_contiguous() && keys.size(1) == ID && keys.stride(0) == ID && first % QB == 0);
    TORCH_CHECK(out.stride(1) == 1 && out.stride(0) % 2 == 0 && top.stride(1) == 1 && qw.stride(1) == 1);
    if (!rows || !n) return;
    const int tiles = (rows + QB - 1) / QB, split = (tiles + MAX_TILES - 1) / MAX_TILES;
    const int ktiles = (n + BN - 1) / BN;
    int gy = split;
    while (gy < tiles && (int64_t)ktiles * gy < 192) gy++;        // four blocks an SM to go round, for the tail
    score8_kernel<<<dim3(ktiles, gy), T, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        q8.data_ptr<uint8_t>() + first * IH * ID, qw.data_ptr<float>() + first, qw.stride(0),
        reinterpret_cast<const __half*>(keys.data_ptr()), vis.data_ptr<int>(), out.data_ptr<float>(), top.data_ptr<float>(), (int)n, (int)rows,
        out.stride(0), top.stride(0));
}
#else
void* use[] = {(void*)quant_q_kernel, (void*)score8_kernel};
#endif

}  // namespace

#ifndef SCORE8_ONLY
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_q", &quant_q);
    m.def("score8", &score8);
}
#endif
