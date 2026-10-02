// Prompt chunks' MLA absorb (q_nope . W_UK) and expand (o_lat . W_UV) with the bits of latent._absorb_q / _expand_v.
//
// Those Triton kernels (num_warps 4, bf16 weights) sum each output in a fixed order set by their layouts:
//  absorb, a [256 k, 32 n] tile: thread (warp w, lane l) holds k = 32 j + 8 w + (l >> 2), j = 0..7, and columns
//    8 (l & 3) .. + 7; its partial is fma(W7, q7, .. fma(W2, q2, fma(W0, q0, W1 * q1))); lanes then add pairwise across
//    lane bits 4, 3, 2 (shfl xor 16, 8, 4), and the four warps' sums are added as (v0 + v2) + (v1 + v3).
//  expand, a [16 n, 512 k] tile: thread (w, l) holds n = (w >> 1) + 2 j, j = 0..7, and k = 256 (w & 1) + 8 l .. + 7;
//    the same chain over its 8 k, lanes added across bits 4, 3, 2, 1, 0, and the two k warps' sums added.
// Here the same chains and the same pairs give the same bits, without a barrier per row: a lane trades half its
// columns at each butterfly level (its partner sums the other half), and the warps' sums meet once for a block of rows.

#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

constexpr int ROWS = 16;    // rows a block of the row loop (one exchange of the warps' sums)

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

// 8 partials -> 1: at level s (xor 16, 8, 4) lane l keeps the half of its live values picked by its bit, sends the
// other half; the kept values add the partner's (each sum is the pair the full butterfly adds). The lane ends with the
// sum for value index 4 b4 + 2 b3 + b2 (bits 4, 3, 2 of l).
__device__ __forceinline__ float trade8(float (&v)[8], int lane) {
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float send = b4 ? v[i] : v[i + 4];
        const float got = __shfl_xor_sync(0xffffffffu, send, 16);
        h[i] = (b4 ? v[i + 4] : v[i]) + got;
    }
    float q[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float send = b3 ? h[i] : h[i + 2];
        const float got = __shfl_xor_sync(0xffffffffu, send, 8);
        q[i] = (b3 ? h[i + 2] : h[i]) + got;
    }
    const float send = b2 ? q[0] : q[1];
    const float got = __shfl_xor_sync(0xffffffffu, send, 4);
    return (b2 ? q[1] : q[0]) + got;
}

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all_but_one() { asm volatile("cp.async.wait_group 1;\n" ::); }

// Program (head h, 32 latent columns, a block of rows): out[r, h, n0 + n] = sum_k Q[r, h, k] WK[h, k, n0 + n]. The
// q rows come in blocks of ROWS through two shared buffers, the next block copied (cp.async) while this one computes.
__global__ void __launch_bounds__(128) absorb_kernel(const __nv_bfloat16* __restrict__ Q,
                                                     const __nv_bfloat16* __restrict__ WK,
                                                     __nv_bfloat16* __restrict__ OUT, int R, int H, int rows_per) {
    constexpr int D = 256, LW = 512, BN = 32;
    const int h = blockIdx.x, n0 = blockIdx.y * BN, r0 = blockIdx.z * rows_per;
    const int r1 = min(R, r0 + rows_per);
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, lr = lane >> 2, lc = lane & 3;
    __shared__ __align__(16) __nv_bfloat16 qs[2][ROWS][D];
    __shared__ float red[ROWS][BN][4];

    auto stage = [&](int buf, int rb) {               // block rb's q rows into qs[buf] (one cp.async group)
        const int nr = min(ROWS, r1 - rb);
        for (int i = tid; i < nr * (D / 8); i += 128) {
            const int r = i / (D / 8), c8 = i % (D / 8);
            cp16(&qs[buf][r][c8 * 8], Q + ((size_t)(rb + r) * H + h) * D + c8 * 8);
        }
        cp_commit();
    };
    if (r0 < r1) stage(0, r0);

    float wf[8][8];                                   // W[k_j, n0 + 8 lc + c], k_j = 32 j + 8 w + lr
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int k = 32 * j + 8 * w + lr;
        const uint4 raw = *reinterpret_cast<const uint4*>(WK + ((size_t)h * D + k) * LW + n0 + 8 * lc);
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int c = 0; c < 8; ++c) wf[j][c] = bf(v[c]);
    }

    int buf = 0;
    for (int rb = r0; rb < r1; rb += ROWS, buf ^= 1) {
        const int nr = min(ROWS, r1 - rb);
        if (rb + ROWS < r1) stage(buf ^ 1, rb + ROWS);  // the next block, while this one computes
        else cp_commit();                                // (an empty group keeps the count)
        cp_wait_all_but_one();
        __syncthreads();                                 // this block's rows are in; the last block's red is read
        for (int r = 0; r < nr; ++r) {
            float q[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) q[j] = bf(qs[buf][r][32 * j + 8 * w + lr]);
            float p[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float acc = __fmaf_rn(wf[0][c], q[0], __fmul_rn(wf[1][c], q[1]));
#pragma unroll
                for (int j = 2; j < 8; ++j) acc = __fmaf_rn(wf[j][c], q[j], acc);
                p[c] = acc;
            }
            red[r][8 * lc + lr][w] = trade8(p, lane);  // column 8 lc + 4 b4 + 2 b3 + b2 = 8 lc + lr
        }
        __syncthreads();
        for (int i = tid; i < nr * BN; i += 128) {
            const int r = i / BN, n = i % BN;
            const float s = (red[r][n][0] + red[r][n][2]) + (red[r][n][1] + red[r][n][3]);
            OUT[((size_t)(rb + r) * H + h) * LW + n0 + n] = __float2bfloat16_rn(s);
        }
    }
}

// Program (head h, 16 value columns, a block of rows): out[r, h, n0 + n] = sum_k OL[r, h, k] WV[h, n0 + n, k]. Each
// thread's eight o values of the next row load while this row computes.
__global__ void __launch_bounds__(128) expand_kernel(const __nv_bfloat16* __restrict__ OL,
                                                     const __nv_bfloat16* __restrict__ WV,
                                                     __nv_bfloat16* __restrict__ OUT, int R, int H, int DV,
                                                     int rows_per) {
    constexpr int LW = 512, BN = 16;
    const int h = blockIdx.x, n0 = blockIdx.y * BN, r0 = blockIdx.z * rows_per;
    const int r1 = min(R, r0 + rows_per);
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, wn = w >> 1, wk = w & 1;
    const int k0 = 256 * wk + 8 * lane;
    __shared__ float red[ROWS][BN][2];

    uint4 nxt = make_uint4(0, 0, 0, 0);
    if (r0 < r1) nxt = *reinterpret_cast<const uint4*>(OL + ((size_t)r0 * H + h) * LW + k0);
    float wf[8][8];                                   // W[n0 + wn + 2 j, k0 + i]
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint4 raw = *reinterpret_cast<const uint4*>(WV + ((size_t)h * DV + n0 + wn + 2 * j) * LW + k0);
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) wf[j][i] = bf(v[i]);
    }

    for (int rb = r0; rb < r1; rb += ROWS) {
        const int nr = min(ROWS, r1 - rb);
        __syncthreads();                              // the previous block's red is read
        for (int r = 0; r < nr; ++r) {
            const uint4 cur = nxt;
            if (rb + r + 1 < r1) nxt = *reinterpret_cast<const uint4*>(OL + ((size_t)(rb + r + 1) * H + h) * LW + k0);
            const __nv_bfloat16* ov = reinterpret_cast<const __nv_bfloat16*>(&cur);
            float o[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = bf(ov[i]);
            float p[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float acc = __fmaf_rn(wf[j][0], o[0], __fmul_rn(wf[j][1], o[1]));
#pragma unroll
                for (int i = 2; i < 8; ++i) acc = __fmaf_rn(wf[j][i], o[i], acc);
                p[j] = acc;
            }
            float s = trade8(p, lane);                // n index 4 b4 + 2 b3 + b2 of this lane
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            if ((lane & 3) == 0) red[r][wn + 2 * (lane >> 2)][wk] = s;
        }
        __syncthreads();
        for (int i = tid; i < nr * BN; i += 128) {
            const int r = i / BN, n = i % BN;
            OUT[((size_t)(rb + r) * H + h) * DV + n0 + n] = __float2bfloat16_rn(red[r][n][0] + red[r][n][1]);
        }
    }
}

void check(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
                " must be a contiguous CUDA bf16 tensor");
}

}  // namespace

void absorb(const at::Tensor& q, const at::Tensor& wk, at::Tensor out, int64_t rows_per) {
    check(q, "q");
    check(wk, "wk");
    check(out, "out");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 256 && wk.dim() == 3 && wk.size(1) == 256 && wk.size(2) == 512 &&
                    wk.size(0) == q.size(1) && out.size(2) == 512,
                "absorb: q [R, H, 256], wk [H, 256, 512], out [R, H, 512]");
    const int R = (int)q.size(0), H = (int)q.size(1);
    c10::cuda::CUDAGuard guard(q.device());
    dim3 grid((unsigned)H, 512 / 32, (unsigned)((R + rows_per - 1) / rows_per));
    absorb_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(wk.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), R, H, (int)rows_per);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void expand(const at::Tensor& ol, const at::Tensor& wv, at::Tensor out, int64_t rows_per) {
    check(ol, "o_lat");
    check(wv, "wv");
    check(out, "out");
    TORCH_CHECK(ol.dim() == 3 && ol.size(2) == 512 && wv.dim() == 3 && wv.size(2) == 512 && wv.size(1) % 16 == 0 &&
                    wv.size(0) == ol.size(1) && out.size(2) == wv.size(1),
                "expand: o_lat [R, H, 512], wv [H, DV, 512], out [R, H, DV]");
    const int R = (int)ol.size(0), H = (int)ol.size(1), DV = (int)wv.size(1);
    c10::cuda::CUDAGuard guard(ol.device());
    dim3 grid((unsigned)H, (unsigned)(DV / 16), (unsigned)((R + rows_per - 1) / rows_per));
    expand_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(ol.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(wv.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), R, H, DV, (int)rows_per);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("absorb", &absorb);
    m.def("expand", &expand);
}
