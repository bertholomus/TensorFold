"""Where each decode EXL3 linear kind sits against a pure read stream of its own bytes at 1 row (rank 0's TP2 parts of
DeepSeek-V4.1, every layer's linear of that kind back to back in a CUDA graph, so nothing comes from L2):

  ldg      glinear's lane-word loads a k step ahead over glinear's exact grid and per-warp K chains, no decode
  dec      the same plus the decode and the mma (glinear's main loop without its epilogue)
  v16      the same bytes, the same blocks and chains, with 16-byte coalesced loads (a pure stream)
  glinear  the real launch (rot_many + glinear through Exl3Linear.grouped)

  python3 linear_floor.py [--model M]
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CUDA = r'''
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "linear_common.cuh"
using namespace tf_exl3;

// MODE 0: lane-word loads only, 1: + decode and mma, 2: 16-byte loads of the same chains
template <int K2, int WK, int MODE>
__global__ void __launch_bounds__(WK * 32, WK == 4 ? 3 : 1)
floor_kernel(const uint32_t* __restrict__ T, int K, int N, int SK, long long stride_k, long long stride_nb,
             float* out, int flag) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>(), NV = 8 * TW / 4;
    const int NB = N >> 7, nb = blockIdx.x % NB, split = blockIdx.x / NB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per_warp = (K >> 4) / SK / WK, kt0 = split * per_warp * WK + warp * per_warp;
    const uint32_t* tile = T + nb * stride_nb + (size_t)kt0 * stride_k;
    float acc[8][2][4] = {};
    const uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
    uint32_t x = 0;
    if constexpr (MODE == 2) {
        const uint4* p = reinterpret_cast<const uint4*>(tile);
        const int n16 = per_warp * NV;
        int i = lane;
        for (; i + 96 < n16; i += 128) {
            const uint4 v0 = __ldg(p + i), v1 = __ldg(p + i + 32), v2 = __ldg(p + i + 64), v3 = __ldg(p + i + 96);
            x ^= v0.x ^ v1.y ^ v2.z ^ v3.w;
        }
        for (; i < n16; i += 32) x ^= __ldg(p + i).x;
    } else {
        constexpr int SR = step_regs<K2>();
        uint32_t cur[SR], nxt[SR];
        load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                step_lane_words<K2>(cur, j, lane, false, w);
                if constexpr (MODE == 1) {
                    uint32_t b0[2], b1[2];
                    decode_lane<K2, 2>(w, lane, b0, b1);
                    mma16816(acc[j][0], a, b0);
                    mma16816(acc[j][1], a, b1);
                } else {
#pragma unroll
                    for (int q = 0; q < LW; ++q) x ^= w[q];
                }
            }
#pragma unroll
            for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
        }
    }
    float s = (float)x;
    for (int j = 0; j < 8; ++j)
        for (int h = 0; h < 2; ++h)
            for (int c = 0; c < 4; ++c) s += acc[j][h][c];
    if (flag) out[blockIdx.x * blockDim.x + threadIdx.x] = s;     // flag is 0: the sums only stay live
}

void run(torch::Tensor T, int64_t K, int64_t N, int64_t SK, int64_t WK, int64_t K2, int64_t sk, int64_t snb,
         int64_t mode, torch::Tensor out) {
    const int blocks = (int)(N / 128 * SK);
    auto st = at::cuda::getCurrentCUDAStream();
#define L(K2_, WK_, M_)                                                                                          \
    if (K2 == K2_ && WK == WK_ && mode == M_) {                                                                  \
        floor_kernel<K2_, WK_, M_><<<blocks, WK_ * 32, 0, st>>>((const uint32_t*)T.data_ptr(), (int)K, (int)N,  \
                                                                (int)SK, sk, snb, out.data_ptr<float>(), 0);     \
        return;                                                                                                  \
    }
#define LL(K2_) L(K2_, 4, 0) L(K2_, 4, 1) L(K2_, 4, 2) L(K2_, 8, 0) L(K2_, 8, 1) L(K2_, 8, 2)
    LL(8) LL(10) LL(12) LL(16)
    TORCH_CHECK(false, "no instance for this width / warps");
}
'''
CPP = ("void run(torch::Tensor T, int64_t K, int64_t N, int64_t SK, int64_t WK, int64_t K2, int64_t sk, int64_t snb, "
       "int64_t mode, torch::Tensor out);")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    from torch.utils.cpp_extension import load_inline

    import tensorfold.cuda.exl3 as exl3
    from tensorfold.cuda.build import arch_flags
    from linear_bench import load_attn
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import model as M
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, linear

    ext = load_inline("tf_linear_floor", cpp_sources=CPP, cuda_sources=CUDA, functions=["run"],
                      extra_include_paths=[os.path.dirname(os.path.abspath(exl3.__file__))],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", *arch_flags()], verbose=False)
    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    layers = [load_attn(sh, cfg, i, 0, 2) for i in range(cfg.n_layers)]
    head = linear(sh, "head", cols=(0, cfg.vocab // 2))
    out = torch.zeros(1 << 20, device="cuda")
    M.GROUPED = True
    kinds = {"wq_a": [L.wq_a for L in layers], "wkv": [L.wkv for L in layers], "wq_b": [L.wq_b for L in layers],
             "wo_a(4 slices)": [w for L in layers for w in L.wo_a], "wo_b": [L.wo_b for L in layers], "head": [head]}

    def graph(fn):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def timed(g, reps=5):
        ts = []
        for _ in range(7):
            g.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / reps)
        return statistics.median(ts)

    for kind, lins in kinds.items():
        l0, n = lins[0], len(lins)
        nbytes = sum(la.nbytes() for la in lins)
        row = {}
        for mode, nm in ((0, "ldg"), (1, "dec"), (2, "v16")):
            row[nm] = timed(graph(lambda mode=mode: [ext.run(la.words, la.k, la.n, la.split[0], la.split[1], la.k2,
                                                             *la.strides, mode, out) for la in lins]))
        x = torch.randn(1, l0.k, device="cuda").bfloat16()
        row["glinear"] = timed(graph(lambda: [la.grouped(x) for la in lins]))
        cols = "  ".join(f"{k} {v / n:7.2f} us ({nbytes / v / 1e3:5.1f} GB/s)" for k, v in row.items())
        print(f"{kind:15s} x{n:3d} K{l0.k} N{l0.n} bits{l0.bits} plan{l0.split} "
              f"blocks{l0.n // 128 * l0.split[0]}  {cols}", flush=True)


if __name__ == "__main__":
    main()
