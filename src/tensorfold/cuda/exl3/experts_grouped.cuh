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
