// The four-node balanced experts split's gate / up on a decode window in one launch (TF_DS_2D_GU=parity): a pair's
// main 128-column blocks of every expert place and, past them, the rest columns (a TP2 half's ninth block) of the
// places whose expert's rest this pair computes, in one grid. The device code is the served build's grouped_cp_kernel
// with its gate / up epilogue (TensorFold's exl3 experts_grouped.cuh, after ExLlamaV3, MIT, Copyright (c) 2025
// Turboderp): its helpers below are copied as they are and the kernel's arithmetic is unchanged, so every program makes
// the products and sums the served launch makes for its block, and the rows are the bits of the served build's two
// launches (main, then rest). A rest program takes the rest's trellis, partials, counters, epilogue tables and rows. No
// readiness: the column exchange stands between gate / up and down.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tf_ds_par {

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

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

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

// The rest columns' tables (exl3_experts2d.RestTab): an expert's NR rest columns of gate and up, a 0 trellis pointer for
// an expert whose rest another pair computes.
struct RestTab {
    const int64_t* tp0;    // int64 [E]: gate's trellis [K / 16, NR / 16, *] of the rest columns, else 0
    const int64_t* tp1;    // up's
    const int* k2_0;       // int32 [E]
    const int* k2_1;
    float* z;              // fp32 [2, SK, P, NR]: the partials
    int* cnt;              // int32 [places x NR / 128]: the epilogue counters, zeros (each resets itself)
    const half* svh_g;     // fp16 [E, NR]
    const half* svh_u;
    const half* suh_d;     // fp16 [E, NR]: down's input signs on the rest columns
    half* xd;              // fp16 [P, NR]: the rest rows (the pack's, after the main rows)
    int N;                 // NR: whole 128-column blocks
};

// grouped_cp_kernel<CB, NT, W, S, LO, HI, 1> over N main columns (grid y: N / 128 main blocks, then NR / 128 rest
// blocks): a main program as the served main launch runs it, a rest program as the served rest launch runs its place
// (the same members, its tables). ep.ready unused.
template <int CB, int NT, int W, int S, int LO, int HI>
__device__ __forceinline__ void gate_up_par(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Zm, int K, int Nm, int P, int SK, int maxm, int slots, const DecodeEpi ep, const RestTab rt) {
    static_assert(NT * 16 == 128, "the epilogue works on the 128-column blocks of a program");
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
    pdl_wait();                                           // the grouping and the rotated rows (decode_prep)
    pdl_launch();                                         // the next grid's programs may take the SMs this grid frees
    const int nu = ucount[0];
    if (ur >= nu) return;
    const int u = nu - 1 - ur, by = rem % NY, bz = rem / NY;
    const int mtile = bz % MT;
    const int split = (bz / MT) % SK;
    const int mat = bz / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    // the blocks past the main ones are the rest's: its own trellis (0: another pair computes this expert's rest),
    // partials, counters, epilogue tables and rows
    const int NB = Nm >> 7;
    const bool rest = by >= NB;
    const int bk = rest ? by - NB : by;                   // the program's 128-column block in its part
    const int N = rest ? rt.N : Nm;
    const int64_t tw = rest ? (mat ? rt.tp1[e] : rt.tp0[e]) : (mat ? TP1[e] : TP0[e]);
    if (tw == 0) return;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(tw);
    const int k2 = rest ? (mat ? rt.k2_1[e] : rt.k2_0[e]) : (mat ? K2_1[e] : K2_0[e]);
    float* __restrict__ Z = rest ? rt.z : Zm;
    const half* svh_g = rest ? rt.svh_g : ep.svh_g;
    const half* svh_u = rest ? rt.svh_u : ep.svh_u;
    const half* suh_d = rest ? rt.suh_d : ep.suh_d;
    half* xd = rest ? rt.xd : ep.xd;
    int* cnt = rest ? rt.cnt + u * (rt.N >> 7) + bk : ep.cnt + u * NB + by;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    // the expert's member count (members are a prefix; maxm <= the block's threads)
    const int cu = __syncthreads_count(threadIdx.x < maxm && members[u * maxm + threadIdx.x] >= 0);
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = bk * NT;
    uint32_t* ring = smem + warp * S * SW;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (k2) {
#define TF_DS_PAR_CASE(K2_)                                                                                     \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles_cp<CB, K2_, NT, S, SW, false>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0,     \
                                                     lane, ring, acc, nullptr, 0);                              \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_DS_PAR_CASE(2)
        TF_DS_PAR_CASE(3)
        TF_DS_PAR_CASE(4)
        TF_DS_PAR_CASE(5)
        TF_DS_PAR_CASE(6)
        TF_DS_PAR_CASE(7)
        TF_DS_PAR_CASE(8)
        TF_DS_PAR_CASE(9)
        TF_DS_PAR_CASE(10)
        TF_DS_PAR_CASE(11)
        TF_DS_PAR_CASE(12)
        TF_DS_PAR_CASE(13)
        TF_DS_PAR_CASE(14)
        TF_DS_PAR_CASE(15)
        TF_DS_PAR_CASE(16)
#undef TF_DS_PAR_CASE
        default:
            __trap();
    }
    cp_async_wait<0>();
    __syncthreads();                                      // every warp is done with its ring: it becomes red

    // warps' partial sums through shared memory, added in warp order: rows 0-7, then rows 8-15 when the tile has them
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
    // the program completing this (expert, column block) runs the gate/up epilogue for the expert's rows there
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int target = (NZ / MT) * ((cu + 15) / 16);         // mats x splits programs a live member tile
        const int before = atomicAdd(cnt, 1);
        last_sh = before == target - 1;
        if (last_sh) *cnt = 0;
    }
    __syncthreads();
    if (!last_sh) return;
    __threadfence();
    const int n = bk * 128 + 4 * lane;
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
                float gg = fminf(epi_bf16r(gv[j] * EPI_HAD * __half2float(svh_g[(size_t)e * N + n + j])), ep.limit);
                float uu = fminf(fmaxf(epi_bf16r(uv[j] * EPI_HAD * __half2float(svh_u[(size_t)e * N + n + j])),
                                       -ep.limit), ep.limit);
                act = epi_bf16r(epi_bf16r(gg / (1.f + expf(-gg))) * uu);
            } else {
                float gg = fminf(gv[j] * EPI_HAD * __half2float(svh_g[(size_t)e * N + n + j]), ep.limit);
                float uu = fminf(fmaxf(uv[j] * EPI_HAD * __half2float(svh_u[(size_t)e * N + n + j]), -ep.limit),
                                 ep.limit);
                act = gg / (1.f + expf(-gg)) * uu;
            }
            v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
        }
        epi_fwht128(v, lane);
        half* o = xd + (size_t)p * N + n;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * EPI_HAD);
        if (ep.discard) {
            // this member's gate/up partials of the column block (2 mats x SK splits x 512 bytes, 128-byte aligned: N a
            // multiple of 128), summed above by this warp alone, are dead: drop their L2 lines
            __syncwarp();
            if (lane < 8 * SK) {
                const int ms = lane >> 2, c = (lane & 3) * 32;           // (mat, split) = ms / SK, ms % SK
                const float* zl = Z + ((size_t)ms * P + p) * N + bk * 128 + c;
                asm volatile("discard.global.L2 [%0], 128;" ::"l"(zl) : "memory");
            }
        }
    }
}

}  // namespace tf_ds_par

// The codebook's (mul1, CB 2) decode setting (NT 8, 4 warps, 3 stages) by K2 range, as grouped_cp_launch picks it:
// tf_ds_par_gu_<LO>_<HI>.
#define TF_DS_PAR_GU(NAME, LO_, HI_)                                                                                \
    extern "C" __global__ void __launch_bounds__(4 * 32, 4) NAME(                                                   \
        const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,                  \
        const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,                \
        const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,              \
        float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, const tf_ds_par::DecodeEpi ep,     \
        const tf_ds_par::RestTab rt) {                                                                              \
        tf_ds_par::gate_up_par<2, 8, 4, 3, LO_, HI_>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N,  \
                                                     P, SK, maxm, slots, ep, rt);                                   \
    }
TF_DS_PAR_GU(tf_ds_par_gu_8_8, 8, 8)
TF_DS_PAR_GU(tf_ds_par_gu_2_10, 2, 10)
TF_DS_PAR_GU(tf_ds_par_gu_2_16, 2, 16)
#undef TF_DS_PAR_GU
