// Grouped EXL3 expert GEMV (after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): rows stay independent, K ranges fixed by shape, warps summed in order.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace tf_exl3x {

// Two codebook values (CB 0 3inst, 1 mcg, 2 mul1) as a half2, bit-identical to ExLlamaV3's decode_3inst_2<cb>.
template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

// K2 half-bits a value: a tile is 4 * K2 words; a lane's eight windows fall in NG runs of GV within two words.
template <int K2>
struct Fmt {
    static constexpr int TW = 4 * K2;
    static constexpr int LW = (TW + 31) / 32;
    // windows sharing one 64-bit merge (tests/cuda/test_exl3_experts.py checks every K2)
    static constexpr int GV = (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return (p >> 1) * K2 + ((p & 1) ? K2 : (K2 >> 1)); }
    // right shift of window j of a run (run starts at an even position) relative to the run's last window
    __host__ __device__ static constexpr int off(int j) { return end(GV - 1) - end(j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 128 * K2;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

// This lane's eight values of a tile as the B fragments of its two n8 halves.
template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        // 4 bits: lane L's windows are exactly words L-1 and L (the GLM kernel's decode)
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w[0] >> 16;
        st[4] = (w[0] >> 12) & 0xffffu;
        st[5] = (w[0] >> 8) & 0xffffu;
        st[6] = (w[0] >> 4) & 0xffffu;
        st[7] = w[0] & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = __ldg(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? __ldg(p + l * 32) : 0u;
    }
}

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc, PF tiles in flight.
template <int CB, int K2, int NT, int PF>
__device__ __forceinline__ void warp_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                           const half* x0, const half* x1, bool ok0, bool ok1, int lane,
                                           float (&acc)[NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                const int k = (kt0 + it) * 16;
                uint32_t a[4] = {load_pair(x0 + k, ok0), load_pair(x1 + k, ok1), load_pair(x0 + k + 8, ok0),
                                 load_pair(x1 + k + 8, ok1)};
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
}

// warp_tiles for G member tiles at once: each k tile's weights are decoded once and fed to every live tile's rows (off:
// a row's element offset in X, -1 for none); a row's mma sequence is warp_tiles' (same k tiles, same order, from zero).
template <int CB, int K2, int NT, int PF, int G>
__device__ __forceinline__ void warp_tiles_rows(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                                const half* __restrict__ X, const int (&off)[G][2], int live, int lane,
                                                float (&acc)[G][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    // the rows' A fragments a k tile ahead (the loads' latency behind this tile's decode and mma)
    const half* xr[G][2];
    bool ok[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            ok[g][j] = off[g][j] >= 0;
            xr[g][j] = X + (ok[g][j] ? off[g][j] : 0) + kt0 * 16;
        }
    uint32_t an[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g)
        if (g < live) {
            an[g][0] = load_pair(xr[g][0], ok[g][0]);
            an[g][1] = load_pair(xr[g][1], ok[g][1]);
            an[g][2] = load_pair(xr[g][0] + 8, ok[g][0]);
            an[g][3] = load_pair(xr[g][1] + 8, ok[g][1]);
        }

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                uint32_t a[G][4];
#pragma unroll
                for (int g = 0; g < G; ++g)
#pragma unroll
                    for (int c = 0; c < 4; ++c) a[g][c] = an[g][c];
                if (it + 1 < nkt) {
                    const int k = (it + 1) * 16;
#pragma unroll
                    for (int g = 0; g < G; ++g)
                        if (g < live) {
                            an[g][0] = load_pair(xr[g][0] + k, ok[g][0]);
                            an[g][1] = load_pair(xr[g][1] + k, ok[g][1]);
                            an[g][2] = load_pair(xr[g][0] + k + 8, ok[g][0]);
                            an[g][3] = load_pair(xr[g][1] + k + 8, ok[g][1]);
                        }
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        if (g < live) {
                            mma16816(acc[g][i][0], a[g], b0);
                            mma16816(acc[g][i][1], a[g], b1);
                        }
                    }
                }
            }
        }
    }
}

// The K2 values an instance covering [LO, HI] compiles (half-bits 2..16).
__host__ __device__ constexpr bool k2_supported(int k2) {
    return k2 >= 2 && k2 <= 16;
}

// Program (expert u, n block, split and member tile): up to 16 members times W_q over the split's K range; warps added in order.
template <int CB, int NT, int W, int PF, int LO, int HI>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles<CB, K2_, NT, PF>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);    \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

    // warps' partial sums through shared memory, added in warp order
    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
    }
}

// Prompt chunks: a program per (group of G member tiles, n block, expert u, split), the group's weights decoded once.
// Every row gets grouped_kernel's bits: the same K range a warp, the same mma sequence, warps added in order.
// Launch order runs an expert's member groups back to back (its weights from L2), then its column blocks (its rows).
// FOLD: one program runs every split in order and writes their sum from 0 (the gate/up epilogue's order: the epilogue
// then reads one split instead of SK), so the fp32 partials written and read drop SK times, the bits unchanged.
template <int CB, int NT, int W, int PF, int LO, int HI, int G, bool FOLD>
__global__ void __launch_bounds__(W * 32) grouped_rows_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, int nexp) {
    constexpr int GR = 16 * G;
    constexpr int TPT = 16 * NT * 16 / (W * 32);          // a thread's outputs of a member tile in the warp sum
    const int u = blockIdx.z % nexp;
    if (u >= ucount[0]) return;
    const int mg = blockIdx.x;
    const int splits = FOLD ? 1 : SK;                     // grid splits (FOLD runs them all in one program)
    const int split0 = FOLD ? 0 : (blockIdx.z / nexp) % SK;
    const int mat = blockIdx.z / nexp / splits;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[GR];
    for (int i = threadIdx.x; i < GR; i += W * 32) {
        const int m = mg * GR + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this group is empty
    int live = 0;                                         // its occupied tiles are a prefix
#pragma unroll
    for (int g = 0; g < G; ++g) live += rows_sh[g * 16] >= 0;
    int off[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int r0 = rows_sh[g * 16 + g8], r1 = rows_sh[g * 16 + g8 + 8];
        off[g][0] = r0 < 0 ? -1 : r0 * K + 2 * t;
        off[g][1] = r1 < 0 ? -1 : r1 * K + 2 * t;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int nt0 = blockIdx.y * NT;
    float tot[G][TPT];                                    // FOLD: the splits' sums so far, from 0
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < TPT; ++j) tot[g][j] = 0.f;
    __shared__ float red[W][16][NT * 16];

    for (int split = split0; split < (FOLD ? SK : split0 + 1); ++split) {
        const int kt0 = split * per_split + warp * per_warp;
        float acc[G][NT][2][4];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[g][i][h][c] = 0.f;

        switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles_rows<CB, K2_, NT, PF, G>(T, NTILES, kt0, per_warp, nt0, X, off, live, lane, acc);        \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
            TF_EXL3X_CASE(2)
            TF_EXL3X_CASE(3)
            TF_EXL3X_CASE(4)
            TF_EXL3X_CASE(5)
            TF_EXL3X_CASE(6)
            TF_EXL3X_CASE(7)
            TF_EXL3X_CASE(8)
            TF_EXL3X_CASE(9)
            TF_EXL3X_CASE(10)
            TF_EXL3X_CASE(11)
            TF_EXL3X_CASE(12)
            TF_EXL3X_CASE(13)
            TF_EXL3X_CASE(14)
            TF_EXL3X_CASE(15)
            TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
            default:
                __trap();
        }

        // warps' partial sums through shared memory, added in warp order, one member tile after another
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < live) {
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        red[warp][g8][col] = acc[g][i][h][0];
                        red[warp][g8][col + 1] = acc[g][i][h][1];
                        red[warp][g8 + 8][col] = acc[g][i][h][2];
                        red[warp][g8 + 8][col + 1] = acc[g][i][h][3];
                    }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < TPT; ++j) {
                    const int idx = threadIdx.x + j * W * 32;
                    const int row = idx / (NT * 16), col = idx % (NT * 16);
                    float s = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) s += red[w][row][col];
                    if constexpr (FOLD) {
                        tot[g][j] += s;
                    } else {
                        const int r = rows_sh[g * 16 + row];
                        if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
                    }
                }
                __syncthreads();
            }
        }
    }
    if constexpr (FOLD) {                                 // one split's worth: Z [mats, 1, P, N]
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < live) {
#pragma unroll
                for (int j = 0; j < TPT; ++j) {
                    const int idx = threadIdx.x + j * W * 32;
                    const int row = idx / (NT * 16), col = idx % (NT * 16);
                    const int r = rows_sh[g * 16 + row];
                    if (r >= 0) Z[((size_t)mat * P + r) * N + nt0 * 16 + col] = tot[g][j];
                }
            }
        }
    }
}

// Prompt chunks with the weights shared through shared memory: a program per (64 member rows, 64 columns, expert u,
// matrix), eight warps: warp w runs member rows 16 (w % 4) .. + 15 against columns 32 (w / 4) .. + 31. The k tiles go
// in steps of MMA_KB (4): during a step the warps decode the next step's sixteen 16 x 16 weight tiles (two each) into
// shared memory as mma B fragments, while the step's mma read the fragments decoded during the step before (two
// buffers, one barrier a step), so a weight is decoded once for up to 64 rows (grouped_rows: 32). The rows come through
// a ring of shared-memory stages (cp.async, MMA_STAGES - 1 steps ahead, ldmatrix), the trellis words two steps ahead.
// (Measured on GB10, 2,048 rows: 2 k tiles a step or 128 rows a program were slower; ncu showed the kernel latency- and
// issue-bound, now close to its weight and row traffic.)
// Every row keeps grouped_kernel's bits: the K range splits into SK x WK chains, each an mma chain from zero over the
// same k tiles in the same order; the WK chains of a split are added in order; with FOLD the splits are added in order
// from 0 (the gate/up epilogue's order: it then reads one split), without (SK 1) the split's sum is written as is.
constexpr int MMA_ROWS = 64;        // member rows a program
constexpr int MMA_COLS = 64;        // columns a program
constexpr int MMA_KB = 4;           // k tiles a step (it divides every chain: 24 for GLM's gate/up, 8 for its down)
constexpr int MMA_STAGES = 3;       // row stages in flight
constexpr int MMA_LD = MMA_KB * 16 + 8;     // halfs a staged row (padded: ldmatrix rows fall in distinct banks)
constexpr int MMA_DEC = MMA_KB / 2;         // weight tiles a warp decodes a step (8 warps, 4 n16 tiles a k tile)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

template <int CB, int K2, bool FOLD>
__device__ __forceinline__ void mma_slice(const uint32_t* __restrict__ T, int NTILES, int KT, int per_range, int WK,
                                          int nt0, const half* __restrict__ X, int K, const int* rows_sh, bool live,
                                          int warp, int lane, half (&As)[MMA_STAGES][MMA_ROWS][MMA_LD],
                                          uint4 (&bsh)[2][MMA_KB][4][32], float (&tot)[4][4]) {
    constexpr int LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * Fmt<K2>::TW;
    const int dk = (warp >> 2) * MMA_DEC, dn = warp & 3;      // the tiles this warp decodes: k tiles dk .. of a step, n16 dn
    const uint32_t* tp = T + ((size_t)dk * NTILES + nt0 + dn) * Fmt<K2>::TW + lane;
    const int rg = warp & 3, cg = warp >> 2;                  // its rows 16 rg .., its n16 tiles 2 cg, 2 cg + 1
    const int NS = KT / MMA_KB;
    // the staged chunks this thread copies: row tid / 4, 16 bytes (tid % 4) + 4 c of the step's MMA_KB * 16 columns
    constexpr int CPR = MMA_KB / 2;                           // chunks a thread a step
    const int crow = threadIdx.x >> 2, cpart = threadIdx.x & 3;
    const int cr = rows_sh[crow];
    const half* src = X + (size_t)(cr < 0 ? 0 : cr) * K + cpart * 8;
    // ldmatrix: lane l reads row (l & 7) + 8 ((l >> 3) & 1) of the warp's 16, columns 8 (l >> 4) of a k tile
    const int lrow = rg * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

#pragma unroll
    for (int st = 0; st < MMA_STAGES - 1; ++st) {
        if (st < NS)
#pragma unroll
            for (int c = 0; c < CPR; ++c)
                cp_async16(&As[st][crow][(cpart + 4 * c) * 8], src + st * MMA_KB * 16 + 32 * c, cr >= 0);
        cp_async_commit();
    }
    uint32_t words[MMA_DEC][LW];
#pragma unroll
    for (int d = 0; d < MMA_DEC; ++d) load_words<K2>(words[d], tp + (size_t)d * kstride, lane);
#pragma unroll
    for (int d = 0; d < MMA_DEC; ++d) {
        uint32_t b0[2], b1[2];
        decode_tile<CB, K2>(words[d], map, lane, b0, b1);
        bsh[0][dk + d][dn][lane] = make_uint4(b0[0], b0[1], b1[0], b1[1]);
    }
    if (NS > 1)
#pragma unroll
        for (int d = 0; d < MMA_DEC; ++d) load_words<K2>(words[d], tp + (size_t)(MMA_KB + d) * kstride, lane);
    cp_async_wait<MMA_STAGES - 2>();
    __syncthreads();

    float acc[4][4], sacc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][c] = sacc[i][c] = 0.f;
    int left = per_range, wk = 0;                             // k tiles left in this chain; its place in the split

    for (int st = 0; st < NS; ++st) {
        const int buf = st & 1, stage = st % MMA_STAGES;
        {                                                     // the rows of the step MMA_STAGES - 1 ahead
            const int ahead = st + MMA_STAGES - 1;
            if (ahead < NS)
#pragma unroll
                for (int c = 0; c < CPR; ++c)
                    cp_async16(&As[ahead % MMA_STAGES][crow][(cpart + 4 * c) * 8],
                               src + (size_t)ahead * MMA_KB * 16 + 32 * c, cr >= 0);
            cp_async_commit();
        }
        uint32_t fr[MMA_DEC][4];                              // the next step's tiles (their words came a step ago)
        if (st + 1 < NS) {
#pragma unroll
            for (int d = 0; d < MMA_DEC; ++d) {
                uint32_t b0[2], b1[2];
                decode_tile<CB, K2>(words[d], map, lane, b0, b1);
                fr[d][0] = b0[0]; fr[d][1] = b0[1]; fr[d][2] = b1[0]; fr[d][3] = b1[1];
            }
            if (st + 2 < NS)
#pragma unroll
                for (int d = 0; d < MMA_DEC; ++d)
                    load_words<K2>(words[d], tp + (size_t)((st + 2) * MMA_KB + d) * kstride, lane);
        }
        if (live) {
#pragma unroll
            for (int kk = 0; kk < MMA_KB; ++kk) {
                uint32_t a[4];
                ldmatrix_x4(a, &As[stage][lrow][kk * 16 + lcol]);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const uint4 b = bsh[buf][kk][2 * cg + j][lane];
                    const uint32_t bl[2] = {b.x, b.y}, bh[2] = {b.z, b.w};
                    mma16816(acc[2 * j], a, bl);
                    mma16816(acc[2 * j + 1], a, bh);
                }
            }
        }
        left -= MMA_KB;
        if (left == 0) {                                      // a chain ends: into its split's sum, in order
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sacc[i][c] = wk == 0 ? acc[i][c] : sacc[i][c] + acc[i][c];
                    acc[i][c] = 0.f;
                }
            left = per_range;
            if (++wk == WK) {                                 // a split ends
                wk = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c) tot[i][c] = FOLD ? tot[i][c] + sacc[i][c] : sacc[i][c];
            }
        }
        if (st + 1 < NS)
#pragma unroll
            for (int d = 0; d < MMA_DEC; ++d)
                bsh[buf ^ 1][dk + d][dn][lane] = make_uint4(fr[d][0], fr[d][1], fr[d][2], fr[d][3]);
        cp_async_wait<MMA_STAGES - 2>();                     // this thread's copies for the next step have landed
        __syncthreads();
    }
}

template <int CB, int LO, int HI, bool FOLD>
__global__ void __launch_bounds__(256) grouped_mma_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int WK, int maxm, int slots, int nexp) {
    const int u = blockIdx.z % nexp;
    if (u >= ucount[0]) return;
    const int mb = blockIdx.x;
    const int mat = blockIdx.z / nexp;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    const int per_range = KT / (SK * WK);

    __shared__ int rows_sh[MMA_ROWS];
    __shared__ __align__(16) half As[MMA_STAGES][MMA_ROWS][MMA_LD];
    __shared__ uint4 bsh[2][MMA_KB][4][32];                   // two steps' B fragments: [step][k tile][n16][lane]
    for (int i = threadIdx.x; i < MMA_ROWS; i += 256) {
        const int m = mb * MMA_ROWS + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                               // members come first, so this block is empty
    const int rg = warp & 3, cg = warp >> 2;
    const int r0 = rows_sh[rg * 16 + g8], r1 = rows_sh[rg * 16 + g8 + 8];
    const bool live = rows_sh[rg * 16] >= 0;                  // the warp has rows (they are a prefix)
    const int nt0 = blockIdx.y * (MMA_COLS / 16);

    float tot[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) tot[i][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            mma_slice<CB, K2_, FOLD>(T, NTILES, KT, per_range, WK, nt0, X, K, rows_sh, live, warp, lane, As, bsh, \
                                     tot);                                                                      \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }
    if (!live) return;
    // Z [mats, 1, P, N]: rows g8 and g8 + 8 of the warp's 16, columns 8 i + 2 t of its 32
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = nt0 * 16 + cg * 32 + i * 8 + 2 * t;
        if (r0 >= 0)
            *reinterpret_cast<float2*>(Z + ((size_t)mat * P + r0) * N + col) = make_float2(tot[i][0], tot[i][1]);
        if (r1 >= 0)
            *reinterpret_cast<float2*>(Z + ((size_t)mat * P + r1) * N + col) = make_float2(tot[i][2], tot[i][3]);
    }
}

// Prompt chunks with each warp decoding its own weight tiles (TF_EXL3_PROMPT_KERNEL=mma2; mma3 adds the input rotation
// and down's epilogue): a program per (64 member rows, 256 columns, expert u, matrix), eight warps, each warp all 64
// rows by its own 32 columns. A warp decodes its two 16 x 16 weight tiles a k tile straight into mma B fragments
// (registers: no shared-memory round trip, no barrier for them) and reads the rows' A fragments from shared memory
// (ldmatrix). Rows and trellis words come through a ring of S stages of KB k tiles by cp.async (one barrier a stage),
// so the weight stream from DRAM runs S - 1 stages ahead; 16-row tiles past the expert's last member skip their loads
// and mma. Each output is one fp32 mma chain over K in order: deterministic and independent of the window, but not
// grouped_kernel's split order (other bits than the decode windows'). Z [mats, P, N] holds the whole sum.
constexpr int M2_ROWS = 64;          // member rows a program
constexpr int M2_COLS = 256;         // columns a program (8 warps x 32)
constexpr int M2_TILES = M2_COLS / 16;          // n16 tiles a program

template <int S, int KB>
__host__ __device__ constexpr size_t m2_rows_bytes() { return (size_t)S * M2_ROWS * (KB * 16 + 8) * sizeof(half); }
template <int S, int KB>
__host__ __device__ constexpr size_t m2_words_bytes(int k2) { return (size_t)S * KB * M2_TILES * 4 * k2 * 4; }

template <int CB, int K2, int NW = 2>
__device__ __forceinline__ void m2_decode(const uint32_t* ts, int TW, const LaneMap<K2>& map, int lane,
                                          uint32_t (&b)[NW][4]) {
    constexpr int LW = Fmt<K2>::LW;
#pragma unroll
    for (int j = 0; j < NW; ++j) {
        uint32_t w[LW];
#pragma unroll
        for (int l = 0; l < LW; ++l) w[l] = (l * 32 + lane < Fmt<K2>::TW) ? ts[j * TW + l * 32 + lane] : 0u;
        uint32_t b0[2], b1[2];
        decode_tile<CB, K2>(w, map, lane, b0, b1);
        b[j][0] = b0[0]; b[j][1] = b0[1]; b[j][2] = b1[0]; b[j][3] = b1[1];
    }
}

template <int CB, int K2, int S, int KB>
__device__ __forceinline__ void m2_slice(const uint32_t* __restrict__ T, int NTILES, int KT, int nt0,
                                         const half* __restrict__ X, int K, const int* rows_sh, int live, int warp,
                                         int lane, half* As, uint32_t* Ts, float (&acc)[4][4][4]) {
    constexpr int TW = Fmt<K2>::TW;
    constexpr int LD = KB * 16 + 8;                           // halfs a staged row (padded)
    constexpr int CPR = KB * 2;                               // 16-byte chunks a row a stage
    constexpr int TCH = M2_TILES * TW / 4;                    // 16-byte chunks of a k tile's words for the program
    const LaneMap<K2> map(lane);
    const int NS = KT / KB;
    const uint32_t* tsrc = T + (size_t)nt0 * TW;
    // ldmatrix: lane l reads row (l & 7) + 8 ((l >> 3) & 1) of a 16-row tile, columns 8 (l >> 4) of a k tile
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

    auto load = [&](int st) {
        const int slot = st % S;
        half* as = As + (size_t)slot * M2_ROWS * LD;
        const int kb = st * KB * 16;
        for (int c = threadIdx.x; c < live * 16 * CPR; c += 256) {
            const int r = c / CPR, q = c % CPR;
            const int p = rows_sh[r];
            cp_async16(as + r * LD + q * 8, X + (size_t)(p < 0 ? 0 : p) * K + kb + q * 8, p >= 0);
        }
        uint32_t* ts = Ts + (size_t)slot * KB * M2_TILES * TW;
        for (int c = threadIdx.x; c < KB * TCH; c += 256) {
            const int kk = c / TCH, q = c % TCH;
            cp_async16(ts + kk * M2_TILES * TW + q * 4, tsrc + ((size_t)(st * KB + kk) * NTILES) * TW + q * 4, true);
        }
    };
#pragma unroll
    for (int st = 0; st < S - 1; ++st) {
        if (st < NS) load(st);
        cp_async_commit();
    }
    for (int st = 0; st < NS; ++st) {
        cp_async_wait<S - 2>();
        __syncthreads();                                      // stage st landed; every warp is done with st - 1
        if (st + S - 1 < NS) load(st + S - 1);
        cp_async_commit();
        const int slot = st % S;
        const half* as = As + (size_t)slot * M2_ROWS * LD;
        const uint32_t* ts = Ts + (size_t)slot * KB * M2_TILES * TW + (size_t)(2 * warp) * TW;
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            uint32_t bb[2][4];
            m2_decode<CB, K2>(ts + kk * M2_TILES * TW, TW, map, lane, bb);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < live) {
                    uint32_t a[4];
                    ldmatrix_x4(a, as + (size_t)(16 * i + lrow) * LD + kk * 16 + lcol);
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const uint32_t bl[2] = {bb[j][0], bb[j][1]}, bh[2] = {bb[j][2], bb[j][3]};
                        mma16816(acc[i][2 * j], a, bl);
                        mma16816(acc[i][2 * j + 1], a, bh);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
}

template <int CB, int LO, int HI>
__global__ void __launch_bounds__(256, 1) grouped_mma2_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int maxm, int slots, int nexp) {
    const int u = blockIdx.z % nexp;
    if (u >= ucount[0]) return;
    const int mb = blockIdx.x;
    const int mat = blockIdx.z / nexp;
    extern __shared__ __align__(16) unsigned char m2_smem[];
    __shared__ int rows_sh[M2_ROWS];
    __shared__ int live_sh;
    if (threadIdx.x < M2_ROWS) {
        const int m = mb * M2_ROWS + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                               // members come first, so this block is empty
    if (threadIdx.x == 0) {
        int n = 0;
        while (n < M2_ROWS && rows_sh[n] >= 0) ++n;
        live_sh = (n + 15) >> 4;
    }
    __syncthreads();
    const int live = live_sh;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int nt0 = blockIdx.y * M2_TILES;
    half* As = reinterpret_cast<half*>(m2_smem);
    uint32_t* Ts = reinterpret_cast<uint32_t*>(m2_smem + m2_rows_bytes<4, 4>());

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][q][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            m2_slice<CB, K2_, 4, 4>(T, N >> 4, K >> 4, nt0, X, K, rows_sh, live, warp, lane, As, Ts, acc);       \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }
    // Z [mats, P, N]: rows g8 and g8 + 8 of each 16-row tile, columns 8 q + 2 t of the warp's 32
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i >= live) break;
        const int pa = rows_sh[16 * i + g8], pb = rows_sh[16 * i + g8 + 8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int col = nt0 * 16 + warp * 32 + q * 8 + 2 * t;
            if (pa >= 0)
                *reinterpret_cast<float2*>(Z + ((size_t)mat * P + pa) * N + col) = make_float2(acc[i][q][0], acc[i][q][1]);
            if (pb >= 0)
                *reinterpret_cast<float2*>(Z + ((size_t)mat * P + pb) * N + col) = make_float2(acc[i][q][2], acc[i][q][3]);
        }
    }
}

// Prompt chunks' gate/up with the input rotation inside (TF_EXL3_PROMPT_KERNEL=mma3): grouped_mma2's programs, but the
// rows' A tiles are made in shared memory from the layer input x (bf16, one row a token, shared by every expert) as
// rot_in makes them: fp16((x * suh) @ H128 / sqrt(128)), a 128-column k block at a time, each warp eight of the 64
// rows. The 9 x 2 rotated copies a token (xg, xu: 16 x the input's bytes) are never written or read. The arithmetic of
// every A value and of every mma chain is rot_in's and grouped_mma2's (cfg 0), so Z holds the same bits. Two A buffers:
// a warp rotates k block b + 1 into one while every warp's mma reads block b from the other; the next block's x and
// suh values wait in registers, the trellis words in a ring of M3_S stages (one barrier a k block).
constexpr int M3_KB = 8;             // k tiles a block (the rotation's 128 columns)
constexpr int M3_LD = M3_KB * 16 + 8;           // halfs an A row (padded)

// NW n16 tiles a warp (2: 256 columns a program, as grouped_mma2; 4: 512, gate's or up's whole width on a TP4 rank, so
// a program rotates its rows once for all of them), S trellis word stages in flight
template <int NW> __host__ __device__ constexpr int m3_stages() { return NW == 2 ? 3 : 2; }
__host__ __device__ constexpr size_t m3_rows_bytes() { return (size_t)2 * M2_ROWS * M3_LD * sizeof(half); }
template <int NW>
__host__ __device__ constexpr size_t m3_words_bytes(int k2) { return (size_t)m3_stages<NW>() * M3_KB * 8 * NW * 4 * k2 * 4; }

// Walsh-Hadamard transform of 128 values, 4 a lane: experts.cu's fwht128 (the same butterflies in the same order).
__device__ __forceinline__ void m3_fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// This warp's eight rows of k block kb: x values (4 bf16 a lane a row) and suh (4 fp16 a lane) into registers.
__device__ __forceinline__ void m3_fetch(const __nv_bfloat16* __restrict__ x, int x_stride, const half* __restrict__ suh,
                                        const int* tok_sh, int warp, int lane, int kb, int nrows,
                                        uint2 (&xv)[8], uint2& sv) {
    const int col = kb * 128 + 4 * lane;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 8 * warp + i;
        const int tr = r < nrows ? tok_sh[r] : -1;
        xv[i] = tr >= 0 ? *reinterpret_cast<const uint2*>(x + (size_t)tr * x_stride + col) : make_uint2(0u, 0u);
    }
    sv = *reinterpret_cast<const uint2*>(suh + col);
}

// Rotate the fetched values into A rows (fp16, rot_in's arithmetic): rows past the members are left as they are.
__device__ __forceinline__ void m3_rotate(half* as, const uint2 (&xv)[8], uint2 sv, int warp, int lane, int nrows) {
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&sv);
    const half2* h2 = reinterpret_cast<const half2*>(&sv);
    const float su[4] = {__low2float(h2[0]), __high2float(h2[0]), __low2float(h2[1]), __high2float(h2[1])};
    (void)s2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 8 * warp + i;
        if (r >= nrows) break;
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv[i]);
        float v[4] = {__low2float(x2[0]) * su[0], __high2float(x2[0]) * su[1], __low2float(x2[1]) * su[2],
                      __high2float(x2[1]) * su[3]};
        m3_fwht128(v, lane);
        half2 o0 = __halves2half2(__float2half_rn(v[0] * 0.08838834764831845f), __float2half_rn(v[1] * 0.08838834764831845f));
        half2 o1 = __halves2half2(__float2half_rn(v[2] * 0.08838834764831845f), __float2half_rn(v[3] * 0.08838834764831845f));
        uint2 o;
        o.x = *reinterpret_cast<uint32_t*>(&o0);
        o.y = *reinterpret_cast<uint32_t*>(&o1);
        *reinterpret_cast<uint2*>(as + (size_t)r * M3_LD + 4 * lane) = o;
    }
}

template <int CB, int K2, int NW>
__device__ __forceinline__ void m3_slice(const uint32_t* __restrict__ T, int NTILES, int KT, int nt0,
                                         const __nv_bfloat16* __restrict__ x, int x_stride,
                                         const half* __restrict__ suh, const int* tok_sh, int nrows, int live,
                                         int warp, int lane, half* As, uint32_t* Ts, float (&acc)[4][2 * NW][4]) {
    constexpr int TW = Fmt<K2>::TW;
    constexpr int TILES = 8 * NW;                             // n16 tiles a program
    constexpr int M3_S = m3_stages<NW>();
    constexpr int TCH = TILES * TW / 4;                       // 16-byte chunks of a k tile's words for the program
    const LaneMap<K2> map(lane);
    const int NB = KT / M3_KB;
    const uint32_t* tsrc = T + (size_t)nt0 * TW;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

    auto load_words = [&](int kb) {
        uint32_t* ts = Ts + (size_t)(kb % M3_S) * M3_KB * TILES * TW;
        for (int c = threadIdx.x; c < M3_KB * TCH; c += 256) {
            const int kk = c / TCH, q = c % TCH;
            cp_async16(ts + kk * TILES * TW + q * 4, tsrc + ((size_t)(kb * M3_KB + kk) * NTILES) * TW + q * 4, true);
        }
    };
    uint2 xv[8], sv;
    m3_fetch(x, x_stride, suh, tok_sh, warp, lane, 0, nrows, xv, sv);
    m3_rotate(As, xv, sv, warp, lane, nrows);
    if (NB > 1) m3_fetch(x, x_stride, suh, tok_sh, warp, lane, 1, nrows, xv, sv);
#pragma unroll
    for (int st = 0; st < M3_S - 1; ++st) {
        if (st < NB) load_words(st);
        cp_async_commit();
    }
    for (int kb = 0; kb < NB; ++kb) {
        cp_async_wait<M3_S - 2>();
        __syncthreads();                    // block kb's words landed and its A rows rotated; every warp done with kb - 1
        if (kb + M3_S - 1 < NB) load_words(kb + M3_S - 1);
        cp_async_commit();
        // block kb + 1's A rows into the other buffer, block kb + 2's values fetched (all rows at once: rotating a row
        // between each k tile's mma measured slower, 5.6 against 5.3 ms a 2,048-row layer)
        if (kb + 1 < NB) {
            m3_rotate(As + (size_t)((kb + 1) & 1) * M2_ROWS * M3_LD, xv, sv, warp, lane, nrows);
            if (kb + 2 < NB) m3_fetch(x, x_stride, suh, tok_sh, warp, lane, kb + 2, nrows, xv, sv);
        }
        const half* as = As + (size_t)(kb & 1) * M2_ROWS * M3_LD;
        const uint32_t* ts = Ts + (size_t)(kb % M3_S) * M3_KB * TILES * TW + (size_t)(NW * warp) * TW;
#pragma unroll
        for (int kk = 0; kk < M3_KB; ++kk) {
            uint32_t b[NW][4];
            m2_decode<CB, K2, NW>(ts + kk * TILES * TW, TW, map, lane, b);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < live) {
                    uint32_t a[4];
                    ldmatrix_x4(a, as + (size_t)(16 * i + lrow) * M3_LD + kk * 16 + lcol);
#pragma unroll
                    for (int j = 0; j < NW; ++j) {
                        const uint32_t bl[2] = {b[j][0], b[j][1]}, bh[2] = {b[j][2], b[j][3]};
                        mma16816(acc[i][2 * j], a, bl);
                        mma16816(acc[i][2 * j + 1], a, bh);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
}

template <int CB, int LO, int HI, int NW>
__global__ void __launch_bounds__(256, 1) grouped_mma3_kernel(
    const __nv_bfloat16* __restrict__ x, int x_stride, const half* __restrict__ suh0, const half* __restrict__ suh1,
    const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1, const int* __restrict__ K2_0,
    const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int maxm, int slots, int nexp) {
    const int u = blockIdx.z % nexp;
    if (u >= ucount[0]) return;
    const int mb = blockIdx.x;
    const int mat = blockIdx.z / nexp;
    extern __shared__ __align__(16) unsigned char m2_smem[];
    __shared__ int rows_sh[M2_ROWS], tok_sh[M2_ROWS];
    __shared__ int n_sh;
    if (threadIdx.x < M2_ROWS) {
        const int m = mb * M2_ROWS + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
        tok_sh[threadIdx.x] = code >= 0 ? (code >> 5) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                               // members come first, so this block is empty
    if (threadIdx.x == 0) {
        int n = 0;
        while (n < M2_ROWS && rows_sh[n] >= 0) ++n;
        n_sh = n;
    }
    __syncthreads();
    const int nrows = n_sh, live = (nrows + 15) >> 4;
    const int e = uids[u];
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int nt0 = blockIdx.y * 8 * NW;
    half* As = reinterpret_cast<half*>(m2_smem);
    uint32_t* Ts = reinterpret_cast<uint32_t*>(m2_smem + m3_rows_bytes());

    float acc[4][2 * NW][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int q = 0; q < 2 * NW; ++q)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][q][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            m3_slice<CB, K2_, NW>(T, N >> 4, K >> 4, nt0, x, x_stride, suh, tok_sh, nrows, live, warp, lane, As,   \
                                  Ts, acc);                                                             \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i >= live) break;
        const int pa = rows_sh[16 * i + g8], pb = rows_sh[16 * i + g8 + 8];
#pragma unroll
        for (int q = 0; q < 2 * NW; ++q) {
            const int col = nt0 * 16 + warp * 16 * NW + q * 8 + 2 * t;
            if (pa >= 0)
                *reinterpret_cast<float2*>(Z + ((size_t)mat * P + pa) * N + col) = make_float2(acc[i][q][0], acc[i][q][1]);
            if (pb >= 0)
                *reinterpret_cast<float2*>(Z + ((size_t)mat * P + pb) * N + col) = make_float2(acc[i][q][2], acc[i][q][3]);
        }
    }
}


// Prompt chunks' down projection with its epilogue inside (mma3): grouped_mma2's program (setting 0) over xd, then the
// 64 x 256 sums go through shared memory and each row's two 128-column blocks get down_combine's per-slot arithmetic,
// fp32 (H128 butterflies, 1 / sqrt(128), svh_d), times the slot's routing weight, stored as bf16 Y [P, D]: half the
// bytes of the fp32 Z, and the combine kernel only adds a row's slots. The weighted, rounded slot outputs are summed by
// combine_y (other bits than down_combine's fp32 fma chain).
constexpr int M3_ELD = M2_COLS + 4;                   // floats an epilogue row (padded)

// Three stages and the epilogue in two halves of 32 rows (46 KB of shared memory at 3 bits) let two programs share an SM,
// one's epilogue beside the other's mma (2.9 against 3.6 ms a 2,048-row layer with four stages, one program an SM).
template <int S, bool HALVES>
__host__ __device__ constexpr size_t m3d_smem_bytes(int k2) {
    return m2_rows_bytes<S, 4>() + m2_words_bytes<S, 4>(k2) > (size_t)(HALVES ? 32 : 64) * M3_ELD * 4
               ? m2_rows_bytes<S, 4>() + m2_words_bytes<S, 4>(k2)
               : (size_t)(HALVES ? 32 : 64) * M3_ELD * 4;
}

template <int CB, int LO, int HI, int S, int MINB, bool HALVES>
__global__ void __launch_bounds__(256, MINB) grouped_down3_kernel(
    const half* __restrict__ Xd, const int64_t* __restrict__ TP, const int* __restrict__ K2s,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    const half* __restrict__ svh, const float* __restrict__ wts, __nv_bfloat16* __restrict__ Y, int K, int N, int P,
    int maxm, int slots, int nexp) {
    const int u = blockIdx.z;
    if (u >= ucount[0]) return;
    const int mb = blockIdx.x;
    extern __shared__ __align__(16) unsigned char m2_smem[];
    __shared__ int rows_sh[M2_ROWS];
    __shared__ int n_sh;
    if (threadIdx.x < M2_ROWS) {
        const int m = mb * M2_ROWS + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;
    if (threadIdx.x == 0) {
        int n = 0;
        while (n < M2_ROWS && rows_sh[n] >= 0) ++n;
        n_sh = n;
    }
    __syncthreads();
    const int nrows = n_sh, live = (nrows + 15) >> 4;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(TP[e]);
    const int k2 = K2s[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int nt0 = blockIdx.y * M2_TILES;
    half* As = reinterpret_cast<half*>(m2_smem);
    uint32_t* Ts = reinterpret_cast<uint32_t*>(m2_smem + m2_rows_bytes<S, 4>());

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][q][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            m2_slice<CB, K2_, S, 4>(T, N >> 4, K >> 4, nt0, Xd, K, rows_sh, live, warp, lane, As, Ts, acc);      \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }
    float* Es = reinterpret_cast<float*>(m2_smem);
    constexpr int ER = HALVES ? 32 : 64;                      // rows a pass of the epilogue holds
#pragma unroll
    for (int h = 0; h < 64 / ER; ++h) {
    if (h * ER >= nrows) break;
    __syncthreads();                                          // the main loop's (or the last pass's) buffers are free
#pragma unroll
    for (int i = h * ER / 16; i < (h + 1) * ER / 16; ++i) {
        if (i >= live) break;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int col = warp * 32 + q * 8 + 2 * t;
            const int er = 16 * i - h * ER + g8;
            *reinterpret_cast<float2*>(Es + er * M3_ELD + col) = make_float2(acc[i][q][0], acc[i][q][1]);
            *reinterpret_cast<float2*>(Es + (er + 8) * M3_ELD + col) = make_float2(acc[i][q][2], acc[i][q][3]);
        }
    }
    __syncthreads();
    const int nr = min(nrows - h * ER, ER);
    for (int it = warp; it < 2 * nr; it += 8) {
        const int r = h * ER + (it >> 1), blk = it & 1;
        const int p = rows_sh[r];
        const float4 z = *reinterpret_cast<const float4*>(Es + (r - h * ER) * M3_ELD + blk * 128 + 4 * lane);
        float v[4] = {z.x, z.y, z.z, z.w};
        m3_fwht128(v, lane);
        const int n = nt0 * 16 + blk * 128 + 4 * lane;
        const float w = wts[p];
        const half2* sv = reinterpret_cast<const half2*>(svh + (size_t)e * N + n);
        const float2 s0 = __half22float2(sv[0]), s1 = __half22float2(sv[1]);
        const float o0 = v[0] * 0.08838834764831845f * s0.x, o1 = v[1] * 0.08838834764831845f * s0.y;
        const float o2 = v[2] * 0.08838834764831845f * s1.x, o3 = v[3] * 0.08838834764831845f * s1.y;
        __nv_bfloat162 y0 = __floats2bfloat162_rn(w * o0, w * o1), y1 = __floats2bfloat162_rn(w * o2, w * o3);
        uint2 yy;
        yy.x = *reinterpret_cast<uint32_t*>(&y0);
        yy.y = *reinterpret_cast<uint32_t*>(&y1);
        *reinterpret_cast<uint2*>(Y + (size_t)p * N + n) = yy;
    }
    }
}


// W_q [K, N] fp16 of one matrix through the same lane decode (tests; not on the forward path).
template <int CB, int K2>
__global__ void dequant_kernel(const uint32_t* __restrict__ T, half* __restrict__ out, int K, int N) {
    const int kt = blockIdx.x, nt = blockIdx.y, lane = threadIdx.x;
    const int NTILES = N >> 4;
    uint32_t w[Fmt<K2>::LW];
    load_words<K2>(w, T + ((size_t)kt * NTILES + nt) * Fmt<K2>::TW + lane, lane);
    const LaneMap<K2> map(lane);
    uint32_t b0[2], b1[2];
    decode_tile<CB, K2>(w, map, lane, b0, b1);
    const int g = lane >> 2, t = lane & 3;
    uint32_t v[4] = {b0[0], b0[1], b1[0], b1[1]};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        half2 h = *reinterpret_cast<half2*>(&v[q]);
        const int col = nt * 16 + g + 8 * (q >> 1);
        const int row = kt * 16 + 2 * t + 8 * (q & 1);
        out[(size_t)row * N + col] = __low2half(h);
        out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}

// ---- decode windows (fewer than 64 rows): grouped_cp_kernel ------------------------------------------------------
// grouped_kernel's programs and arithmetic (every row's bits: the same chains over the same k tiles, the same decode,
// the same mma order, warps added in order), with three changes that leave the bits alone:
//  - each warp copies its k tiles' trellis words 16 bytes a lane (cp.async) S - 1 steps ahead into its own ring of S
//    shared-memory stages and reads them back one word a lane, as grouped_kernel's loads hand them to the decode (GB10,
//    a load-only probe of the same tiles from DRAM: 4-byte lane loads ~195 GB/s, 16-byte copies ~235); the rows' A
//    fragments come a k tile ahead; the ring is reused for the warps' sums when the k tiles are done;
//  - the warps' sums go through shared memory 8 rows at a time (rows 8-15 only when the tile has them), so with at
//    most 128 registers four programs fit an SM (two with the 16-row buffer: GB10 measured 1-8% faster at four);
//  - programs run expert by expert, the last expert (the shared one, the widest) first, each expert's programs back to
//    back (its column blocks read neighbouring trellis bytes together);
//  - EPI 1 (gate/up): the program that completes an (expert, 128-column block) runs the gate/up epilogue for the
//    expert's rows there (gateup_epilogue_kernel's arithmetic); EPI 2 (down, one split): each program writes its rows'
//    per-slot outputs (down_combine_kernel's per-slot arithmetic) and the program that completes a (row, 128-column
//    block) adds the row's slots in slot order (its combine); a counter per block, reset by its last program, and
//    fences order the partial sums before they are read. Who finishes last never changes what is added or in which
//    order, so a row's bits stay those of the separate launches, whatever shares the window.
// The fused launches chain as programmatic dependents (decode_prep -> gate/up -> down): a grid's programs start on the
// SMs its predecessor frees; down's programs copy their first trellis steps, then wait for their own expert's rows only
// (ready[u], published by the gate/up program that runs the expert's last epilogue), or for the whole gate/up grid.

constexpr float EPI_HAD = 0.08838834764831845f;      // 1 / sqrt(128)

// programmatic dependent launch (sm_90+): wait for the grid this one depends on (its writes visible), or let the next
// grid start; both return at once when the launch carries no programmatic dependency
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t load_pair_cg(const half* x, bool ok) {
    return ok ? __ldcg(reinterpret_cast<const unsigned int*>(x)) : 0u;
}

// experts.cu's fwht128 (the epilogues' Walsh-Hadamard transform), the same butterflies in the same order
__device__ __forceinline__ void epi_fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

__device__ __forceinline__ float epi_bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

struct DecodeEpi {
    const int* pick = nullptr;    // [P] the window's picks (rows x slots); picks < 0 or >= E are not routed
    int E = 0;
    // EPI 1: Xd [P, N] = fp16((SwiGLU(H (gate, up sums) * svh) * suh_d) H) of each member row
    const half* svh_g = nullptr;
    const half* svh_u = nullptr;
    const half* suh_d = nullptr;
    half* xd = nullptr;
    float limit = 0.f;
    int act_mode = 1;
    // EPI 2: y [P, N] = (H sum) * svh_d of each member row; with wts, out [rows, N] = the slots' wts-weighted y in slot
    // order (+ add, last); store_y: a non-routed slot adds its y (the caller's), else 0
    const half* svh_d = nullptr;
    float* y = nullptr;
    const float* wts = nullptr;
    const float* add = nullptr;
    float* out = nullptr;
    int store_y = 1;
    int* cnt = nullptr;           // zeroed: EPI 1 [expert places x N / 128], EPI 2 [rows x N / 128]
    // with ready: EPI 2 waits for its own expert's Xd (ready[u] == *epoch) instead of the whole gate/up grid; EPI 1's
    // program that runs an expert's last 128-column epilogue publishes it (ready_cnt counts them, reset by the last)
    int* ready = nullptr;
    int* ready_cnt = nullptr;
    const int* epoch = nullptr;   // this launch's number (decode_prep adds one)
    // the dead fp32 scratch leaves L2 unwritten (discard.global.L2): EPI 1 the gate/up partials Z an epilogue has
    // summed, EPI 2 (with wts) the per-slot outputs y a row's combine has added (nothing reads either afterwards)
    int discard = 0;
};

template <int CB, int K2, int NT, int S, int SW, bool WAIT>
__device__ __forceinline__ void warp_tiles_cp(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                              const half* x0, const half* x1, bool ok0, bool ok1, int lane,
                                              uint32_t* __restrict__ ring, float (&acc)[NT][2][4],
                                              const int* ready_u, int epoch_v) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    constexpr int NV = NT * TW / 4;                       // 16-byte chunks of a step's NT tiles (contiguous)
    constexpr int VPL = (NV + 31) / 32;
    static_assert(NT * TW <= SW, "a stage holds a step's tiles");
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW;
    auto issue = [&](int it) {
        uint32_t* dst = ring + (it % S) * SW;
        const uint32_t* src = tp + (size_t)it * kstride;
#pragma unroll
        for (int v = 0; v < VPL; ++v) {
            const int c = v * 32 + lane;
            if ((NV % 32) == 0 || c < NV) cp_async16(dst + 4 * c, src + 4 * c, true);
        }
    };
#pragma unroll
    for (int d = 0; d < S - 1; ++d) {
        if (d < nkt) issue(d);
        cp_async_commit();
    }
    if constexpr (WAIT) {
        // the trellis does not depend on the gate/up grid, the rows do
        if (ready_u != nullptr) {
            // this expert's rows are complete once its gate/up epilogues have published them (the rest of the gate/up
            // grid may still run); the rows are then read from L2
            if (lane == 0)
                while (ld_acquire(ready_u) != epoch_v) __nanosleep(128);
            __syncwarp();
        } else {
            pdl_wait();
        }
    }
    auto lp = [](const half* p, bool ok) { return WAIT ? load_pair_cg(p, ok) : load_pair(p, ok); };
    uint32_t an[4] = {lp(x0 + kt0 * 16, ok0), lp(x1 + kt0 * 16, ok1), lp(x0 + kt0 * 16 + 8, ok0),
                      lp(x1 + kt0 * 16 + 8, ok1)};
    for (int it = 0; it < nkt; ++it) {
        if (it + S - 1 < nkt) issue(it + S - 1);
        cp_async_commit();
        cp_async_wait<S - 1>();                           // this step's copies (this lane's) are done
        __syncwarp();                                     // ... and every lane's are visible
        const uint32_t* st = ring + (it % S) * SW + lane;
        uint32_t w[NT][LW];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int l = 0; l < LW; ++l) w[i][l] = ((TW % 32) == 0 || l * 32 + lane < TW) ? st[i * TW + l * 32] : 0u;
        __syncwarp();                                     // the stage is read before a later step refills it
        const uint32_t a[4] = {an[0], an[1], an[2], an[3]};
        if (it + 1 < nkt) {
            const int k = (kt0 + it + 1) * 16;
            an[0] = lp(x0 + k, ok0);
            an[1] = lp(x1 + k, ok1);
            an[2] = lp(x0 + k + 8, ok0);
            an[3] = lp(x1 + k + 8, ok1);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w[i], map, lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
    }
}

template <int CB, int NT, int W, int S, int LO, int HI, int EPI>
__global__ void __launch_bounds__(W * 32, 4) grouped_cp_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, const DecodeEpi ep) {
    static_assert(EPI == 0 || NT * 16 == 128, "the epilogues work on the 128-column blocks of a program");
    constexpr int SW = NT * 4 * HI;                       // words a stage: NT tiles of the launch's widest trellis
    constexpr int RING = W * S * SW;
    constexpr int RR = 8;                                 // rows a pass of the warps' reduction (two passes)
    constexpr int RED = W * RR * NT * 16;
    __shared__ __align__(16) uint32_t smem[RING > RED ? RING : RED];
    __shared__ int rows_sh[16];
    __shared__ int last_sh;
    // expert-major order, the last expert place (the largest id: the shared expert) first
    const int MT = (maxm + 15) / 16;
    const int NY = gridDim.y, NZ = gridDim.z;
    const int lin = blockIdx.x + gridDim.x * (blockIdx.y + NY * blockIdx.z);
    const int ur = lin / (NY * NZ), rem = lin % (NY * NZ);
    if constexpr (EPI == 1) {
        pdl_wait();                                       // the grouping and the rotated rows (decode_prep)
        pdl_launch();                                     // down's programs may take the SMs this grid frees
    }
    const int nu = ucount[0];
    if (ur >= nu) return;
    const int u = nu - 1 - ur, by = rem % NY, bz = rem / NY;
    const int mtile = bz % MT;
    const int split = (bz / MT) % SK;
    const int mat = bz / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    // EPI 1: the expert's member count (members are a prefix; maxm <= the block's threads)
    int cu = 0;
    if constexpr (EPI == 1) cu = __syncthreads_count(threadIdx.x < maxm && members[u * maxm + threadIdx.x] >= 0);
    else __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = by * NT;
    uint32_t* ring = smem + warp * S * SW;
    const int* ready_u = (EPI == 2 && ep.ready != nullptr) ? ep.ready + u : nullptr;
    const int epoch_v = (EPI == 2 && ep.ready != nullptr) ? *ep.epoch : 0;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles_cp<CB, K2_, NT, S, SW, EPI == 2>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0,  \
                                                        lane, ring, acc, ready_u, epoch_v);                     \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }
    cp_async_wait<0>();
    __syncthreads();                                      // every warp is done with its ring: it becomes red

    // warps' partial sums through shared memory, added in warp order (grouped_kernel's reduction): rows 0-7, then rows
    // 8-15 when the tile has them (half grouped_kernel's shared memory, the same sums)
    float (*red)[RR][NT * 16] = reinterpret_cast<float (*)[RR][NT * 16]>(smem);
#pragma unroll
    for (int hp = 0; hp < 16 / RR; ++hp) {
        if (hp > 0) {
            if (rows_sh[8] < 0) break;
            __syncthreads();                              // the first pass is read before red is overwritten
        }
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[i][h][2 * hp];         // row g (pass 0) or g + 8 (pass 1)
                red[warp][g][col + 1] = acc[i][h][2 * hp + 1];
            }
        __syncthreads();
        if constexpr (EPI != 2) {
            for (int idx = threadIdx.x; idx < RR * NT * 16; idx += W * 32) {
                const int row = idx / (NT * 16), col = idx % (NT * 16);
                const int r = rows_sh[row + RR * hp];
                if (r < 0) continue;
                float s = red[0][row][col];
#pragma unroll
                for (int w = 1; w < W; ++w) s += red[w][row][col];
                Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
            }
        }
        if constexpr (EPI == 2) {
            // each member row's per-slot output for this column block, then the row's combine by the program that
            // completes the row's routed slots here
            const int n = by * 128 + 4 * lane;
            for (int row = warp; row < RR; row += W) {
                const int p = rows_sh[row + RR * hp];
                if (p < 0) continue;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = 4 * lane + j;
                    float s = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) s += red[w][row][col];
                    float q = 0.f;                        // down_combine's split sum from 0 (one split)
                    q += s;
                    v[j] = q;
                }
                epi_fwht128(v, lane);
                float* yo = ep.y + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) yo[j] = v[j] * EPI_HAD * __half2float(ep.svh_d[(size_t)e * N + n + j]);
                if (ep.wts == nullptr) continue;
                __threadfence();
                __syncwarp();
                const int r = p / slots;
                int last = 0;
                if (lane == 0) {
                    int target = 0;
                    for (int q = 0; q < slots; ++q) {
                        const int pe = ep.pick[r * slots + q];
                        target += pe >= 0 && pe < ep.E;
                    }
                    int* c = ep.cnt + r * NY + by;
                    const int before = atomicAdd(c, 1);
                    last = before == target - 1;
                    if (last) *c = 0;
                }
                last = __shfl_sync(0xffffffffu, last, 0);
                if (!last) continue;
                __threadfence();
                float a4[4] = {0.f, 0.f, 0.f, 0.f};
                for (int q = 0; q < slots; ++q) {
                    const float w = ep.wts[r * slots + q];
                    const int pq = r * slots + q;
                    const int pe = ep.pick[pq];
                    float4 yv = make_float4(0.f, 0.f, 0.f, 0.f);
                    if ((pe >= 0 && pe < ep.E) || ep.store_y)
                        yv = __ldcg(reinterpret_cast<const float4*>(ep.y + (size_t)pq * N + n));
                    a4[0] = fmaf(w, yv.x, a4[0]);
                    a4[1] = fmaf(w, yv.y, a4[1]);
                    a4[2] = fmaf(w, yv.z, a4[2]);
                    a4[3] = fmaf(w, yv.w, a4[3]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    ep.out[(size_t)r * N + n + j] = ep.add ? __fadd_rn(a4[j], ep.add[(size_t)r * N + n + j]) : a4[j];
                if (ep.discard) {
                    // the row's slots' outputs of this column block (512 bytes each, 128-byte aligned: N a multiple
                    // of 128) are added and read by nothing else: drop their L2 lines (this warp read them all)
                    __syncwarp();
                    if (lane < 4 * slots) {
                        const float* yl = ep.y + (size_t)(r * slots + (lane >> 2)) * N + by * 128 + (lane & 3) * 32;
                        asm volatile("discard.global.L2 [%0], 128;" ::"l"(yl) : "memory");
                    }
                }
            }
        }
    }
    if constexpr (EPI == 1) {
        // the program completing this (expert, column block) runs the gate/up epilogue for the expert's rows there
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            const int target = (NZ / MT) * ((cu + 15) / 16);         // mats x splits programs a live member tile
            int* c = ep.cnt + u * NY + by;
            const int before = atomicAdd(c, 1);
            last_sh = before == target - 1;
            if (last_sh) *c = 0;
        }
        __syncthreads();
        if (!last_sh) return;
        __threadfence();
        const int n = by * 128 + 4 * lane;
        for (int m = warp; m < cu; m += W) {
            const int code = members[u * maxm + m];
            const int p = (code >> 5) * slots + (code & 31);
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float sg = 0.f, su = 0.f;
                for (int s = 0; s < SK; ++s) {
                    sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                    su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                }
                gv[j] = sg;
                uv[j] = su;
            }
            epi_fwht128(gv, lane);
            epi_fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float act;
                if (ep.act_mode == 0) {
                    float gg = fminf(epi_bf16r(gv[j] * EPI_HAD * __half2float(ep.svh_g[(size_t)e * N + n + j])),
                                     ep.limit);
                    float uu = fminf(fmaxf(epi_bf16r(uv[j] * EPI_HAD * __half2float(ep.svh_u[(size_t)e * N + n + j])),
                                           -ep.limit), ep.limit);
                    act = epi_bf16r(epi_bf16r(gg / (1.f + expf(-gg))) * uu);
                } else {
                    float gg = fminf(gv[j] * EPI_HAD * __half2float(ep.svh_g[(size_t)e * N + n + j]), ep.limit);
                    float uu = fminf(fmaxf(uv[j] * EPI_HAD * __half2float(ep.svh_u[(size_t)e * N + n + j]), -ep.limit),
                                     ep.limit);
                    act = gg / (1.f + expf(-gg)) * uu;
                }
                v[j] = act * __half2float(ep.suh_d[(size_t)e * N + n + j]);
            }
            epi_fwht128(v, lane);
            half* o = ep.xd + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * EPI_HAD);
            if (ep.discard) {
                // this member's gate/up partials of the column block (2 mats x SK splits x 512 bytes, 128-byte
                // aligned: N a multiple of 128), summed above by this warp alone, are dead: drop their L2 lines
                __syncwarp();
                if (lane < 8 * SK) {
                    const int ms = lane >> 2, c = (lane & 3) * 32;           // (mat, split) = ms / SK, ms % SK
                    const float* zl = Z + ((size_t)ms * P + p) * N + by * 128 + c;
                    asm volatile("discard.global.L2 [%0], 128;" ::"l"(zl) : "memory");
                }
            }
        }
        if (ep.ready != nullptr) {
            // the expert's last column block: publish its rows to down's programs (they wait on ready[u])
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last_sh = atomicAdd(ep.ready_cnt + u, 1) == NY - 1;
            __syncthreads();
            if (ep.discard && (K & 63) == 0) {
                // every gate/up program of this expert has read its members' rotated input rows (each arrived at its
                // block's counter after its k tiles; every block's epilogue has run): those fp16 rows (decode_prep's,
                // 128-byte aligned) belong to this expert alone and are dead, so their L2 lines go unwritten (before
                // the release below, so nothing after it can see them dropped)
                if (last_sh) {
                    const int lpm = K >> 6;                  // 128-byte lines of a row
                    for (int i = threadIdx.x; i < cu * 2 * lpm; i += W * 32) {
                        const int m = i / (2 * lpm), mat = (i / lpm) & 1, l = i % lpm;
                        const int code = members[u * maxm + m];
                        const int p = (code >> 5) * slots + (code & 31);
                        const half* xl = (mat ? X1 : X0) + (size_t)p * K + l * 64;
                        asm volatile("discard.global.L2 [%0], 128;" ::"l"(xl) : "memory");
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && last_sh) {
                ep.ready_cnt[u] = 0;
                __threadfence();
                st_release(ep.ready + u, *ep.epoch);
            }
        }
    }
}

struct GroupedArgs {
    const half* x0;
    const half* x1;
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int* uids;
    const int* ucount;
    const int* members;
    float* z;
    int K, N, P, SK, maxm, slots;
    int nexp_max;        // grid.x (upper bound of distinct experts)
    int mats, nt, warps, pf, lo, hi;
    int g = 1;           // grouped_rows: member tiles a program
    int fold = 0;        // grouped_rows / grouped_mma: one program runs every split (writes their sum: Z [mats, 1, P, N])
};

template <int CB>
void grouped_rows_launch(const GroupedArgs& a, cudaStream_t stream) {
    const int MG = (a.maxm + 16 * a.g - 1) / (16 * a.g);
    dim3 grid((unsigned)MG, (unsigned)(a.N / (16 * a.nt)), (unsigned)(a.nexp_max * a.mats * (a.fold ? 1 : a.SK)));
#define TF_LAUNCH_F(NT_, W_, PF_, G_, LO_, HI_, F_)                                                             \
    grouped_rows_kernel<CB, NT_, W_, PF_, LO_, HI_, G_, F_><<<grid, W_ * 32, 0, stream>>>(                      \
        a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK, a.maxm, \
        a.slots, a.nexp_max)
#define TF_LAUNCH(NT_, W_, PF_, G_, LO_, HI_)                                                                   \
    do {                                                                                                        \
        if (a.fold) TF_LAUNCH_F(NT_, W_, PF_, G_, LO_, HI_, true);                                              \
        else TF_LAUNCH_F(NT_, W_, PF_, G_, LO_, HI_, false);                                                    \
    } while (0)
#define TF_RANGES(NT_, W_, PF_, G_)                                                                             \
    if (a.lo == 8 && a.hi == 8) TF_LAUNCH(NT_, W_, PF_, G_, 8, 8);                                              \
    else if (a.lo >= 2 && a.hi <= 10) TF_LAUNCH(NT_, W_, PF_, G_, 2, 10);                                       \
    else TF_LAUNCH(NT_, W_, PF_, G_, 2, 16);
    if (a.nt == 8 && a.warps == 4 && a.pf == 1 && a.g == 1) { TF_RANGES(8, 4, 1, 1) }
    else if (a.nt == 8 && a.warps == 4 && a.pf == 2 && a.g == 1) { TF_RANGES(8, 4, 2, 1) }
    else if (a.nt == 8 && a.warps == 4 && a.pf == 1 && a.g == 2) { TF_RANGES(8, 4, 1, 2) }
    else if (a.nt == 4 && a.warps == 4 && a.pf == 2 && a.g == 2) { TF_RANGES(4, 4, 2, 2) }
    else TORCH_CHECK(false, "unsupported prompt tile setting nt=", a.nt, " warps=", a.warps, " pf=", a.pf, " g=", a.g);
#undef TF_RANGES
#undef TF_LAUNCH
#undef TF_LAUNCH_F
}

// grouped_mma_kernel: a program per (64 member rows, 64 columns, expert, matrix); warps = the window's K chains a split
template <int CB>
void grouped_mma_launch(const GroupedArgs& a, cudaStream_t stream) {
    const int MB = (a.maxm + MMA_ROWS - 1) / MMA_ROWS;
    dim3 grid((unsigned)MB, (unsigned)(a.N / MMA_COLS), (unsigned)(a.nexp_max * a.mats));
#define TF_LAUNCH_F(LO_, HI_, F_)                                                                               \
    grouped_mma_kernel<CB, LO_, HI_, F_><<<grid, 256, 0, stream>>>(                                             \
        a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK,        \
        a.warps, a.maxm, a.slots, a.nexp_max)
#define TF_LAUNCH(LO_, HI_)                                                                                     \
    do {                                                                                                        \
        if (a.fold) TF_LAUNCH_F(LO_, HI_, true);                                                                \
        else TF_LAUNCH_F(LO_, HI_, false);                                                                      \
    } while (0)
    TORCH_CHECK((a.K / 16) % (a.SK * a.warps * MMA_KB) == 0, "prompt mma: ", MMA_KB, " k tiles a step must divide a chain");
    if (a.lo == 8 && a.hi == 8) TF_LAUNCH(8, 8);
    else if (a.lo >= 2 && a.hi <= 10) TF_LAUNCH(2, 10);
    else TF_LAUNCH(2, 16);
#undef TF_LAUNCH
#undef TF_LAUNCH_F
}

// grouped_mma2_kernel: a program per (64 member rows, 256 columns, expert, matrix); shared memory sized to the widest
// trellis (HI) of the launch
template <int CB, int LO, int HI>
void m2_launch_one(const GroupedArgs& a, cudaStream_t stream) {
    const int MB = (a.maxm + M2_ROWS - 1) / M2_ROWS;
    dim3 grid((unsigned)MB, (unsigned)(a.N / M2_COLS), (unsigned)(a.nexp_max * a.mats));
    const size_t smem = m2_rows_bytes<4, 4>() + m2_words_bytes<4, 4>(HI);
    auto fn = grouped_mma2_kernel<CB, LO, HI>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    fn<<<grid, 256, smem, stream>>>(a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K,
                                    a.N, a.P, a.maxm, a.slots, a.nexp_max);
}

template <int CB>
void grouped_mma2_launch(const GroupedArgs& a, cudaStream_t stream) {
    TORCH_CHECK(a.N % M2_COLS == 0 && a.K % 64 == 0, "prompt mma2: N a multiple of ", M2_COLS, " and K of 64");
    if (a.lo == 6 && a.hi == 6) m2_launch_one<CB, 6, 6>(a, stream);
    else if (a.lo == 8 && a.hi == 8) m2_launch_one<CB, 8, 8>(a, stream);
    else if (a.lo >= 2 && a.hi <= 10) m2_launch_one<CB, 2, 10>(a, stream);
    else TORCH_CHECK(false, "prompt mma2: up to 5 bits a weight (shared memory), not ", a.hi / 2.0);
}

// bytes of shared memory grouped_mma2_launch takes for the widest trellis
inline size_t m2_smem_bytes(int hi) { return m2_rows_bytes<4, 4>() + m2_words_bytes<4, 4>(hi <= 6 ? 6 : hi <= 8 ? 8 : 10); }

struct Mma3Args {
    const __nv_bfloat16* x;
    int x_stride;
    const half* suh0;
    const half* suh1;
};

// grouped_mma3_kernel: gate and up (mats 2) from the layer input; Z [2, P, N]. a.g: 4 = 512 columns a program (one
// rotation of the rows for all of a mat's columns at GLM-5.3's TP4 width), 2 = 256 (grouped_mma2's tiles)
template <int CB, int NW, int LO, int HI>
void m3_launch_one(const GroupedArgs& a, const Mma3Args& m, cudaStream_t stream) {
    const int MB = (a.maxm + M2_ROWS - 1) / M2_ROWS;
    dim3 grid((unsigned)MB, (unsigned)(a.N / (128 * NW)), (unsigned)(a.nexp_max * a.mats));
    const size_t smem = m3_rows_bytes() + m3_words_bytes<NW>(HI);
    auto fn = grouped_mma3_kernel<CB, LO, HI, NW>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    fn<<<grid, 256, smem, stream>>>(m.x, m.x_stride, m.suh0, m.suh1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount,
                                    a.members, a.z, a.K, a.N, a.P, a.maxm, a.slots, a.nexp_max);
}

template <int CB>
void grouped_mma3_launch(const GroupedArgs& a, const Mma3Args& m, cudaStream_t stream) {
    TORCH_CHECK(a.N % M2_COLS == 0 && a.K % 128 == 0, "prompt mma3: N a multiple of ", M2_COLS, " and K of 128");
    const bool wide = a.g == 4 && a.N % 512 == 0;
    if (a.lo == 6 && a.hi == 6) {
        if (wide) m3_launch_one<CB, 4, 6, 6>(a, m, stream);
        else m3_launch_one<CB, 2, 6, 6>(a, m, stream);
    } else if (a.lo == 8 && a.hi == 8) {
        if (wide) m3_launch_one<CB, 4, 8, 8>(a, m, stream);
        else m3_launch_one<CB, 2, 8, 8>(a, m, stream);
    } else if (a.lo >= 2 && a.hi <= 8) {
        m3_launch_one<CB, 2, 2, 8>(a, m, stream);
    } else {
        TORCH_CHECK(false, "prompt mma3: up to 4 bits a weight (shared memory), not ", a.hi / 2.0);
    }
}

template <int CB>
void grouped_down3_launch(const GroupedArgs& a, const half* svh, const float* wts, __nv_bfloat16* Y,
                          cudaStream_t stream) {
    TORCH_CHECK(a.N % M2_COLS == 0 && a.K % 64 == 0, "prompt down3: N a multiple of ", M2_COLS, " and K of 64");
    const int MB = (a.maxm + M2_ROWS - 1) / M2_ROWS;
    dim3 grid((unsigned)MB, (unsigned)(a.N / M2_COLS), (unsigned)a.nexp_max);
#define TF_LAUNCH_C(LO_, HI_, S_, B_, H_)                                                                        \
    do {                                                                                                        \
        const size_t smem = m3d_smem_bytes<S_, H_>(HI_);                                                        \
        auto fn = grouped_down3_kernel<CB, LO_, HI_, S_, B_, H_>;                                               \
        C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));       \
        fn<<<grid, 256, smem, stream>>>(a.x0, a.tp0, a.k2_0, a.uids, a.ucount, a.members, svh, wts, Y, a.K, a.N, \
                                        a.P, a.maxm, a.slots, a.nexp_max);                                      \
    } while (0)
#define TF_LAUNCH(LO_, HI_) TF_LAUNCH_C(LO_, HI_, 3, 2, true)
    if (a.lo == 6 && a.hi == 6) TF_LAUNCH(6, 6);
    else if (a.lo == 8 && a.hi == 8) TF_LAUNCH(8, 8);
    else if (a.lo >= 2 && a.hi <= 8) TF_LAUNCH(2, 8);
    else TORCH_CHECK(false, "prompt down3: up to 4 bits a down weight (shared memory), not ", a.hi / 2.0);
#undef TF_LAUNCH
#undef TF_LAUNCH_C
}

template <int CB>
void grouped_launch(const GroupedArgs& a, cudaStream_t stream) {
    const int MT = (a.maxm + 15) / 16;
    dim3 grid((unsigned)a.nexp_max, (unsigned)(a.N / (16 * a.nt)), (unsigned)(a.mats * a.SK * MT));
#define TF_LAUNCH(NT_, W_, PF_, LO_, HI_)                                                                       \
    grouped_kernel<CB, NT_, W_, PF_, LO_, HI_><<<grid, W_ * 32, 0, stream>>>(                                   \
        a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK, a.maxm, \
        a.slots)
#define TF_RANGES(NT_, W_, PF_)                                                                                 \
    if (a.lo == 8 && a.hi == 8) TF_LAUNCH(NT_, W_, PF_, 8, 8);                                                  \
    else if (a.lo >= 2 && a.hi <= 10) TF_LAUNCH(NT_, W_, PF_, 2, 10);                                           \
    else TF_LAUNCH(NT_, W_, PF_, 2, 16);
    if (a.nt == 8 && a.warps == 4 && a.pf == 1) { TF_RANGES(8, 4, 1) }
    else if (a.nt == 8 && a.warps == 4 && a.pf == 2) { TF_RANGES(8, 4, 2) }
    else if (a.nt == 4 && a.warps == 4 && a.pf == 2) { TF_RANGES(4, 4, 2) }
    else TORCH_CHECK(false, "unsupported tile setting nt=", a.nt, " warps=", a.warps, " pf=", a.pf);
#undef TF_RANGES
#undef TF_LAUNCH
}

// grouped_cp_kernel: grouped_launch's grid and settings, pf = the stages of each warp's ring (3), epi as above, pdl:
// gate/up and down launched as programmatic dependents
template <int CB>
void grouped_cp_launch(const GroupedArgs& a, const DecodeEpi& ep, int epi, int pdl, cudaStream_t stream) {
    const int MT = (a.maxm + 15) / 16;
    dim3 grid((unsigned)a.nexp_max, (unsigned)(a.N / (16 * a.nt)), (unsigned)(a.mats * a.SK * MT));
    TORCH_CHECK(epi == 0 || (a.nt == 8 && a.maxm <= 32 * a.warps), "decode epilogues: 128-column blocks, maxm <= threads");
    TORCH_CHECK(epi != 2 || (a.SK == 1 && a.mats == 1), "the fused down combine takes one split");
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid;
    lc.blockDim = dim3(32 * a.warps);
    lc.dynamicSmemBytes = 0;
    lc.stream = stream;
    cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = la;
    lc.numAttrs = (pdl && epi != 0) ? 1 : 0;
#define TF_LAUNCH(NT_, W_, S_, LO_, HI_, EPI_)                                                                  \
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, grouped_cp_kernel<CB, NT_, W_, S_, LO_, HI_, EPI_>, a.x0, a.x1, a.tp0,  \
                                      a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P,    \
                                      a.SK, a.maxm, a.slots, ep))
#define TF_EPI(LO_, HI_)                                                                                        \
    if (epi == 1) TF_LAUNCH(8, 4, 3, LO_, HI_, 1);                                                              \
    else if (epi == 2) TF_LAUNCH(8, 4, 3, LO_, HI_, 2);                                                         \
    else TF_LAUNCH(8, 4, 3, LO_, HI_, 0);
    TORCH_CHECK(a.nt == 8 && a.warps == 4 && a.pf == 3, "unsupported decode tile setting nt=", a.nt, " warps=",
                a.warps, " stages=", a.pf);
    if (a.lo == 8 && a.hi == 8) { TF_EPI(8, 8) }
    else if (a.lo >= 2 && a.hi <= 10) { TF_EPI(2, 10) }
    else { TF_EPI(2, 16) }
#undef TF_EPI
#undef TF_LAUNCH
}

template <int CB>
void dequant_launch(const uint32_t* t, half* o, int K, int N, int k2, cudaStream_t stream) {
    dim3 grid((unsigned)(K / 16), (unsigned)(N / 16));
    switch (k2) {
#define TF_DQ(K2_) case K2_: dequant_kernel<CB, K2_><<<grid, 32, 0, stream>>>(t, o, K, N); break;
        TF_DQ(2) TF_DQ(3) TF_DQ(4) TF_DQ(5) TF_DQ(6) TF_DQ(7) TF_DQ(8) TF_DQ(9) TF_DQ(10) TF_DQ(11)
        TF_DQ(12) TF_DQ(13) TF_DQ(14) TF_DQ(15) TF_DQ(16)
#undef TF_DQ
        default: TORCH_CHECK(false, "unsupported K2 ", k2);
    }
}

}  // namespace tf_exl3x
