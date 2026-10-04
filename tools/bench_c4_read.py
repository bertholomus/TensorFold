"""GB/s of plain reads (no decode) in the routed experts' access patterns on one GB10: how close an expert kernel could
get to the read ceiling with each layout and work split, before writing one.

Each warp sums the 32-bit words of its chunks (cp.async into a ring of shared-memory stages, as grouped_sm_kernel
streams them); experts are 3-bit 6,144 x 512 (gate/up) and 512 x 6,144 (down) matrices, 16 x 16 tiles of 96 bytes,
stored [K/16][N/16] (a k row of tiles contiguous); 68 random experts of 256 a layer, cold, many layers.

patterns: "blocks" (today's blocks: 4 warps = 4 K ranges of one 128-column block, 24 / 8 steps a warp);
"warpcols" (4 warps = 4 adjacent 128-column blocks of one split, each warp walking the split's 4 K ranges one after
another: 96 / 32 steps a warp); "seq" (each warp a contiguous run of the same bytes).

usage (one GPU): python3 tools/bench_c4_read.py [STAGES=4]
"""

from __future__ import annotations

import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ void cp16(void* s, const void* g) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait_n() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// mode 0 blocks: block (u, cb, z = mat * SK + split), warp w: steps it of K range (split * W + w)
// mode 1 warpcols: block (u, cbg, z), warp w: column block cbg * 4 + w, steps over the split's W K ranges in order
// mode 2 seq: warp walks a contiguous run of steps * 768 bytes from the expert's start + offset
template <int S>
__global__ void __launch_bounds__(128) readk(const int64_t* __restrict__ ptrs, const int* __restrict__ ids, int nexp,
                                             int K, int N, int SK, int mode, unsigned* __restrict__ out) {
    constexpr int TW = 24, NT = 8, SW = NT * TW, CH = SW / 4;
    extern __shared__ __align__(16) unsigned ring[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int KT = K / 16, NTILES = N / 16, NB = N / 128;
    int u, cb, z;
    if (mode == 1) { const int g = NB / 4; u = blockIdx.y; cb = (blockIdx.x % g) * 4 + warp; z = blockIdx.x / g; }
    else { u = blockIdx.y; cb = blockIdx.x % NB; z = blockIdx.x / NB; }
    if (u >= nexp) return;
    const int mat = z / SK, split = z % SK;
    const unsigned* T = reinterpret_cast<const unsigned*>(ptrs[ids[u] * 2 + mat]);
    const int per_split = KT / SK, per_warp = per_split / 4;
    const int steps = mode == 1 ? per_split : per_warp;
    auto src_of = [&](int it) -> const unsigned* {
        if (mode == 2) {
            const size_t base = ((size_t)(split * 4 + warp) * NB + cb) * (size_t)per_warp * SW;
            return T + base + (size_t)it * SW;
        }
        const int kt = mode == 1 ? split * per_split + it : split * per_split + warp * per_warp + it;
        return T + ((size_t)kt * NTILES + cb * NT) * TW;
    };
    unsigned* my = ring + warp * S * SW;
    auto issue = [&](int it) { const unsigned* s = src_of(it); unsigned* d = my + (it % S) * SW;
        for (int c = lane; c < CH; c += 32) cp16(d + 4 * c, s + 4 * c); };
    for (int d = 0; d < S - 1; ++d) { if (d < steps) issue(d); commit(); }
    unsigned acc = 0;
    for (int it = 0; it < steps; ++it) {
        if (it + S - 1 < steps) issue(it + S - 1);
        commit();
        wait_n<S - 1>();
        __syncwarp();
        const unsigned* st = my + (it % S) * SW;
#pragma unroll
        for (int i = 0; i < NT; ++i) acc ^= st[i * TW + (lane % TW)];
        __syncwarp();
    }
    if (acc == 0x12345678u) out[0] = acc;
}

void run(torch::Tensor ptrs, torch::Tensor ids, int64_t nexp, int64_t K, int64_t N, int64_t SK, int64_t mode,
         int64_t stages, torch::Tensor out) {
    const int NB = (int)(N / 128);
    dim3 grid(mode == 1 ? (unsigned)(NB / 4 * 2 * SK) : (unsigned)(NB * 2 * SK), (unsigned)nexp);
    if (K < N) grid.x = mode == 1 ? (unsigned)(NB / 4 * SK) : (unsigned)(NB * SK);     // down: one matrix
    const int smem = 4 * (int)stages * 8 * 24 * 4;
    auto st = at::cuda::getCurrentCUDAStream();
    if (stages == 4) readk<4><<<grid, 128, smem, st>>>(ptrs.data_ptr<int64_t>(), ids.data_ptr<int>(), (int)nexp, (int)K,
        (int)N, (int)SK, (int)mode, (unsigned*)out.data_ptr<int>());
    else readk<8><<<grid, 128, smem, st>>>(ptrs.data_ptr<int64_t>(), ids.data_ptr<int>(), (int)nexp, (int)K,
        (int)N, (int)SK, (int)mode, (unsigned*)out.data_ptr<int>());
}
"""

CPP = "void run(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor);"


def main() -> None:
    stages = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    major, minor = torch.cuda.get_device_capability()
    ext = load_inline("c4_read_v1", CPP, cuda_sources=SRC, functions=["run"], verbose=False,
                      extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"])
    E, D, I, L, NEXP = 256, 6144, 512, 16, 68
    gu_bytes, d_bytes = D * I * 3 // 8, I * D * 3 // 8
    layers = []
    for _ in range(L):
        gate = torch.randint(0, 2**31 - 1, (E, gu_bytes // 4), dtype=torch.int32, device="cuda")
        up = torch.randint(0, 2**31 - 1, (E, gu_bytes // 4), dtype=torch.int32, device="cuda")
        down = torch.randint(0, 2**31 - 1, (E, d_bytes // 4), dtype=torch.int32, device="cuda")
        gp = torch.stack([torch.tensor([gate[e].data_ptr(), up[e].data_ptr()]) for e in range(E)]).cuda()
        dp = torch.stack([torch.tensor([down[e].data_ptr(), down[e].data_ptr()]) for e in range(E)]).cuda()
        ids = torch.randperm(E)[:NEXP].to(torch.int32).cuda()
        layers.append((gate, up, down, gp, dp, ids))
    out = torch.zeros((1,), dtype=torch.int32, device="cuda")

    def timed(fn) -> float:
        fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / 5

    for name, mode in (("blocks", 0), ("warpcols", 1), ("seq", 2)):
        tg = timed(lambda: [ext.run(gp, ids, NEXP, D, I, 4, mode, stages, out) for _, _, _, gp, _, ids in layers])
        td = timed(lambda: [ext.run(dp, ids, NEXP, I, D, 1, mode, stages, out) for _, _, _, _, dp, ids in layers])
        print(f"{name:9s} {stages} stages: gate/up {L * NEXP * 2 * gu_bytes / tg / 1e9:6.1f} GB/s, "
              f"down {L * NEXP * d_bytes / td / 1e9:6.1f} GB/s", flush=True)


if __name__ == "__main__":
    main()
