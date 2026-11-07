// A query's 64 heads over its selected pool rows on this rank, online softmax: the prompt rounds' attention (cp4_kernels.sparse_mla's
// job, same numbers up to the order of sums). One block a query, 8 warps: a warp holds 16 heads, half of the query's dims in registers
// (S of those dims; the two warps of a head group add their halves through shared memory) and half of the output columns. Rows reach
// shared memory by cp.async a tile ahead, in the pool's bytes; every thread widens its share of the next tile to fp16 once, for all
// heads, so the MMAs read fp16 through ldmatrix. Row: 512 e4m3 + 4 fp32 scales (+ 64 fp16 k_pe on rope layers).

#ifndef MLA_ONLY
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int H = 64;                                             // heads
constexpr int C = 512;                                            // latent values a row
constexpr int BN = 16;                                            // rows a tile
constexpr int T = 256;                                            // threads a block
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float STALE = 8.0f;                                     // log2: a max this far behind is kept, p stays in fp16's range

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
// Two e4m3 codes (low byte first) times a scale, rounded once to fp16, as the Triton kernel's fp32 product.
__device__ __forceinline__ uint32_t widen2(uint32_t codes, float s) {
    uint32_t h2;
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.rn.f16x2.e4m3x2 %0, lo; }" : "=r"(h2) : "r"(codes));
    const float2 f = __half22float2(*reinterpret_cast<__half2*>(&h2));
    return pack(f.x * s, f.y * s);
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ void pair_sync(int m) { asm volatile("bar.sync %0, 64;" ::"r"(m + 1) : "memory"); }

template <int RD>
struct Shape {
    static constexpr int LD = C + RD + 8;                         // halfs a widened row: ldmatrix rows fall in distinct banks
    static constexpr int ROW = C + 16 + RD * 2;                   // bytes a pool row
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int KS = (C / 2 + RD / 2) / 16;              // k steps of a warp's half of the dims
    static constexpr int TILE = BN * LD * 2;
    static constexpr int RAW = BN * ROW;
    static constexpr int SMEM = 2 * TILE + 2 * RAW + 2 * 4 * 2 * 32 * 16 + 2048 * 4;
};

template <int RD>
__global__ void __launch_bounds__(T, 1) mla_rows_kernel(const __half* __restrict__ QL, const __half* __restrict__ QP,
                                                        const int* __restrict__ ROWS, const int* __restrict__ CNT,
                                                        const uint8_t* __restrict__ PLANE, __half* __restrict__ OUT,
                                                        float* __restrict__ LSE, float sm, int64_t ks, int64_t qh, int64_t qr,
                                                        int64_t ph, int64_t pr, int64_t oh, int64_t orow, int64_t row_bytes) {
    using S = Shape<RD>;
    extern __shared__ __align__(16) uint8_t base[];
    __half* tile = reinterpret_cast<__half*>(base);                                   // [2][BN][LD]
    uint8_t* raw = base + 2 * S::TILE;                                                // [2][BN][ROW]
    float4* sx = reinterpret_cast<float4*>(raw + 2 * S::RAW);                         // [h][m][n tile][lane]
    int* ids = reinterpret_cast<int*>(sx + 2 * 4 * 2 * 32);                           // the query's rows

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, m = warp & 3, h = warp >> 2;
    const int g = lane >> 2, t2 = (lane & 3) * 2;
    const int64_t q = blockIdx.x;
    const int cnt = CNT[q];
    const int head0 = m * 16 + g, head1 = head0 + 8;

    if (cnt == 0) {
        for (int i = tid; i < H * C / 8; i += T) {
            const int hd = i / (C / 8), c = i % (C / 8);
            *reinterpret_cast<uint4*>(OUT + hd * oh + q * orow + c * 8) = make_uint4(0, 0, 0, 0);
        }
        if (tid < H) LSE[q * H + tid] = -INFINITY;
        return;
    }
    for (int i = tid; i < cnt; i += T) ids[i] = ROWS[q * ks + i];
    __syncthreads();

    const int tiles = (cnt + BN - 1) / BN;
    // A share of tile t's rows: one 16-byte piece a thread, ``piece`` of a tile's ceil(BN * CHUNKS / T).
    auto fetch_piece = [&](int t, int piece) {
        const int c = tid + piece * T;
        if (c < BN * S::CHUNKS) {
            const int r = c / S::CHUNKS, part = c % S::CHUNKS, i = t * BN + r;
            const bool ok = i < cnt;
            cp_async16(raw + (t & 1) * S::RAW + r * S::ROW + part * 16, PLANE + (ok ? (int64_t)ids[i] : 0) * row_bytes + part * 16, ok);
        }
    };
    constexpr int PIECES = (BN * S::CHUNKS + T - 1) / T;
    auto fetch = [&](int t) {
#pragma unroll
        for (int i = 0; i < PIECES; i++) fetch_piece(t, i);
        cp_async_commit();
    };
    fetch(0);
    if (tiles > 1) fetch(1);

    // The warp's half of the query: A fragments of k steps over 256 latent dims (and 32 rope dims).
    uint32_t qf[S::KS][4];
#pragma unroll
    for (int j = 0; j < S::KS; j++) {
        const bool lat = j < 16;
        const __half* b0 = lat ? QL + head0 * qh + q * qr + h * 256 + j * 16 : QP + head0 * ph + q * pr + h * (RD / 2) + (j - 16) * 16;
        const __half* b1 = lat ? QL + head1 * qh + q * qr + h * 256 + j * 16 : QP + head1 * ph + q * pr + h * (RD / 2) + (j - 16) * 16;
        qf[j][0] = *reinterpret_cast<const uint32_t*>(b0 + t2);
        qf[j][1] = *reinterpret_cast<const uint32_t*>(b1 + t2);
        qf[j][2] = *reinterpret_cast<const uint32_t*>(b0 + t2 + 8);
        qf[j][3] = *reinterpret_cast<const uint32_t*>(b1 + t2 + 8);
    }

    // Widen tile t: a thread takes 8 values of each 128-value group of one row, and a 16-byte piece of its k_pe. A group at a
    // time, between the MMAs of the tile before it, as the fetches: spread out, they cost the tensor cores nothing.
    const int wr = tid >> 4, wc = tid & 15;
    auto widen_group = [&](int t, int i) {
        const uint8_t* src = raw + (t & 1) * S::RAW + wr * S::ROW;
        __half* dst = tile + (t & 1) * (BN * S::LD) + wr * S::LD;
        const uint2 w = *reinterpret_cast<const uint2*>(src + i * 128 + wc * 8);
        const float s = *reinterpret_cast<const float*>(src + C + i * 4);
        uint4 o;
        o.x = widen2(w.x, s);
        o.y = widen2(w.x >> 16, s);
        o.z = widen2(w.y, s);
        o.w = widen2(w.y >> 16, s);
        *reinterpret_cast<uint4*>(dst + i * 128 + wc * 8) = o;
        if (RD && i == 3 && wc < RD / 8)
            *reinterpret_cast<uint4*>(dst + C + wc * 8) = *reinterpret_cast<const uint4*>(src + C + 16 + wc * 16);
    };

    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    const float scale = sm * LOG2E;

    cp_async_wait_all();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) widen_group(0, i);
    const int krow = (lane & 7) + ((lane >> 4) << 3), kcol = ((lane >> 3) & 1) << 3;
    const int vrow = lane & 15, vcol = (lane >> 4) << 3;
    float4* mine = sx + ((h * 4 + m) * 2) * 32 + lane;
    float4* other = sx + (((1 - h) * 4 + m) * 2) * 32 + lane;

    for (int t = 0; t < tiles; t++) {
        cp_async_wait_all();                                      // tile t + 1 has landed, this thread's part of it
        __syncthreads();                                          // ... and everyone's; tile t is wide, tile t - 1 is read
        const __half* tl = tile + (t & 1) * (BN * S::LD);

        float s[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        const bool more = t + 1 < tiles;
        auto kcol_of = [&](int j) { return j < 16 ? h * 256 + j * 16 : C + h * (RD / 2) + (j - 16) * 16; };
        uint32_t b[2][4];
        ldsm4(b[0], tl + (krow)*S::LD + kcol_of(0) + kcol);
#pragma unroll
        for (int j = 0; j < S::KS; j++) {
            if (j + 1 < S::KS) ldsm4(b[(j + 1) & 1], tl + (krow)*S::LD + kcol_of(j + 1) + kcol);
            mma(s[0], qf[j], b[j & 1][0], b[j & 1][1]);
            mma(s[1], qf[j], b[j & 1][2], b[j & 1][3]);
            if (t + 2 < tiles && (j & 3) == 1 && (j >> 2) < PIECES) fetch_piece(t + 2, j >> 2);
        }
        if (t + 2 < tiles) cp_async_commit();

        mine[0] = make_float4(s[0][0], s[0][1], s[0][2], s[0][3]);
        mine[32] = make_float4(s[1][0], s[1][1], s[1][2], s[1][3]);
        pair_sync(m);
        const float4 x0 = other[0], x1 = other[32];
        s[0][0] += x0.x, s[0][1] += x0.y, s[0][2] += x0.z, s[0][3] += x0.w;
        s[1][0] += x1.x, s[1][1] += x1.y, s[1][2] += x1.z, s[1][3] += x1.w;

        const int col = t * BN + t2;
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const bool ok = col + j * 8 + (c & 1) < cnt;
                s[j][c] = ok ? s[j][c] * scale : -INFINITY;
            }
        float a0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
        float a1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
        a0 = fmaxf(a0, __shfl_xor_sync(~0u, a0, 1));
        a1 = fmaxf(a1, __shfl_xor_sync(~0u, a1, 1));
        a0 = fmaxf(a0, __shfl_xor_sync(~0u, a0, 2));
        a1 = fmaxf(a1, __shfl_xor_sync(~0u, a1, 2));
        if (__any_sync(~0u, a0 > m0 + STALE || a1 > m1 + STALE)) {
            const float n0 = fmaxf(m0, a0), n1 = fmaxf(m1, a1);
            const float r0 = ex2(m0 - n0), r1 = ex2(m1 - n1);
            m0 = n0, m1 = n1;
            l0 *= r0, l1 *= r1;
#pragma unroll
            for (int j = 0; j < 32; j++) o[j][0] *= r0, o[j][1] *= r0, o[j][2] *= r1, o[j][3] *= r1;
        }
        uint32_t p[4];
        {
            const float p00 = ex2(s[0][0] - m0), p01 = ex2(s[0][1] - m0), p02 = ex2(s[0][2] - m1), p03 = ex2(s[0][3] - m1);
            const float p10 = ex2(s[1][0] - m0), p11 = ex2(s[1][1] - m0), p12 = ex2(s[1][2] - m1), p13 = ex2(s[1][3] - m1);
            l0 += (p00 + p01) + (p10 + p11);
            l1 += (p02 + p03) + (p12 + p13);
            p[0] = pack(p00, p01);
            p[1] = pack(p02, p03);
            p[2] = pack(p10, p11);
            p[3] = pack(p12, p13);
        }
        ldsm4t(b[0], tl + vrow * S::LD + h * 256 + vcol);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            if (j + 1 < 16) ldsm4t(b[(j + 1) & 1], tl + vrow * S::LD + h * 256 + (j + 1) * 16 + vcol);
            mma(o[2 * j], p, b[j & 1][0], b[j & 1][1]);
            mma(o[2 * j + 1], p, b[j & 1][2], b[j & 1][3]);
            if (more && (j & 3) == 1) widen_group(t + 1, j >> 2);
        }
    }

    l0 += __shfl_xor_sync(~0u, l0, 1);
    l1 += __shfl_xor_sync(~0u, l1, 1);
    l0 += __shfl_xor_sync(~0u, l0, 2);
    l1 += __shfl_xor_sync(~0u, l1, 2);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    __half* d0 = OUT + head0 * oh + q * orow + h * 256 + t2;
    __half* d1 = OUT + head1 * oh + q * orow + h * 256 + t2;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        *reinterpret_cast<uint32_t*>(d0 + j * 8) = pack(o[j][0] * i0, o[j][1] * i0);
        *reinterpret_cast<uint32_t*>(d1 + j * 8) = pack(o[j][2] * i1, o[j][3] * i1);
    }
    if (h == 0 && (lane & 3) == 0) {
        LSE[q * H + head0] = l0 > 0.0f ? (m0 + log2f(l0)) * LN2 : -INFINITY;
        LSE[q * H + head1] = l1 > 0.0f ? (m1 + log2f(l1)) * LN2 : -INFINITY;
    }
}

#ifndef MLA_ONLY
template <int RD>
void launch(const torch::Tensor& q_lat, const torch::Tensor& q_pe, const torch::Tensor& rows, const torch::Tensor& cnt,
            const torch::Tensor& plane, double sm, torch::Tensor& out, torch::Tensor& lse) {
    static bool ready = false;
    auto kernel = mla_rows_kernel<RD>;
    if (!ready) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape<RD>::SMEM);
        ready = true;
    }
    const auto* qp = RD ? reinterpret_cast<const __half*>(q_pe.data_ptr()) : nullptr;
    kernel<<<q_lat.size(0), T, Shape<RD>::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q_lat.data_ptr()), qp, rows.data_ptr<int>(), cnt.data_ptr<int>(),
        plane.data_ptr<uint8_t>(), reinterpret_cast<__half*>(out.data_ptr()), lse.data_ptr<float>(), (float)sm, rows.stride(0),
        q_lat.stride(1), q_lat.stride(0), RD ? q_pe.stride(1) : 0, RD ? q_pe.stride(0) : 0, out.stride(1), out.stride(0),
        plane.stride(0));
}

void mla_rows(torch::Tensor q_lat, c10::optional<torch::Tensor> q_pe, torch::Tensor rows, torch::Tensor cnt, torch::Tensor plane,
              double sm, torch::Tensor out, torch::Tensor lse) {
    TORCH_CHECK(q_lat.size(1) == H && q_lat.size(2) == C && q_lat.stride(2) == 1 && out.stride(2) == 1, "64 heads of 512, last dim dense");
    TORCH_CHECK(rows.size(1) <= 2048, "at most 2,048 rows a query");
    if (q_pe.has_value()) {
        TORCH_CHECK(q_pe->size(2) == 64 && q_pe->stride(2) == 1, "64 rope dims, last dim dense");
        TORCH_CHECK(plane.stride(0) == Shape<64>::ROW, "a rope layer's rows are 656 bytes");
        launch<64>(q_lat, *q_pe, rows, cnt, plane, sm, out, lse);
    } else {
        TORCH_CHECK(plane.stride(0) >= Shape<0>::ROW, "a row holds 528 bytes");
        launch<0>(q_lat, q_lat, rows, cnt, plane, sm, out, lse);
    }
}

#else
__global__ void keep(const __half* a, const int* b, const uint8_t* c, __half* d, float* e) {}
template __global__ void mla_rows_kernel<64>(const __half*, const __half*, const int*, const int*, const uint8_t*, __half*, float*, float,
                                             int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template __global__ void mla_rows_kernel<0>(const __half*, const __half*, const int*, const int*, const uint8_t*, __half*, float*, float,
                                            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
#endif

}  // namespace

#ifndef MLA_ONLY
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mla_rows", &mla_rows); }
#endif
