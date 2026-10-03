"""GB10 ceilings for the prompt kernels, one GPU: cuBLAS fp16 / bf16 GEMM TFLOPS at prompt shapes, the raw
mma.sync m16n8k16 (fp16 in, fp32 accumulate) rate of the tensor cores, the FP8 mma rate, and DRAM copy GB/s.

usage (one GPU, tf container): python3 tools/bench_gb10_peak.py
"""

from __future__ import annotations

import time

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>

template <int CHAINS>
__global__ void mma_loop(float* out, int iters) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
    uint32_t b[2] = {threadIdx.x * 11u, threadIdx.x * 13u};
    float d[CHAINS][4];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) d[c][i] = 0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                         "{%0,%1,%2,%3};\n"
                         : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int CHAINS>
__global__ void mma_loop_f16acc(float* out, int iters) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
    uint32_t b[2] = {threadIdx.x * 11u, threadIdx.x * 13u};
    uint32_t d[CHAINS][2];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) d[c][0] = d[c][1] = 0u;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
                         "{%0,%1};\n"
                         : "+r"(d[c][0]), "+r"(d[c][1])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    uint32_t s = 0u;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) s ^= d[c][0] ^ d[c][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s;
}

template <int CHAINS>
__global__ void mma_loop_fp8(float* out, int iters) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
    uint32_t b[2] = {threadIdx.x * 11u, threadIdx.x * 13u};
    float d[CHAINS][4];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) d[c][i] = 0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c)
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                         "{%0,%1,%2,%3};\n"
                         : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

void run(torch::Tensor out, int64_t kind, int64_t blocks, int64_t threads, int64_t iters) {
    float* o = out.data_ptr<float>();
    if (kind == 0) mma_loop<8><<<blocks, threads>>>(o, iters);
    else if (kind == 1) mma_loop_f16acc<8><<<blocks, threads>>>(o, iters);
    else mma_loop_fp8<8><<<blocks, threads>>>(o, iters);
}
"""
CPP = "void run(torch::Tensor out, int64_t kind, int64_t blocks, int64_t threads, int64_t iters);"


def gemm(dtype, m, n, k, reps=10) -> float:
    a = torch.randn((m, k), device="cuda").to(dtype)
    b = torch.randn((k, n), device="cuda").to(dtype)
    c = a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    torch.cuda.synchronize()
    return 2 * m * n * k * reps / (time.perf_counter() - t) / 1e12


def main() -> None:
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}: {p.multi_processor_count} SMs, sm_{p.major}{p.minor}, L2 {p.L2_cache_size / 2**20:.0f} MiB",
          flush=True)
    for dt in (torch.float16, torch.bfloat16):
        for m, n, k in ((8192, 8192, 8192), (2048, 6144, 6144), (4096, 6144, 2048), (2048, 2048, 6144)):
            print(f"cuBLAS {str(dt)[6:]} {m}x{n}x{k}: {gemm(dt, m, n, k):.1f} TFLOPS", flush=True)
    ext = load_inline("tf_gb10_peak", cpp_sources=CPP, cuda_sources=SRC, functions=["run"],
                      extra_cuda_cflags=["-O3", "-arch=sm_121a" if p.major == 12 and p.minor == 1 else "-O3"])
    sms = p.multi_processor_count
    for kind, name, flops in ((0, "mma m16n8k16 f16 in, f32 acc", 4096), (1, "mma m16n8k16 f16 acc", 4096),
                              (2, "mma m16n8k32 e4m3, f32 acc", 8192)):
        for warps in (4, 8, 16):
            blocks, threads, iters = sms * 4, warps * 32, 20000
            out = torch.empty((blocks * threads,), device="cuda")
            ext.run(out, kind, blocks, threads, 100)
            torch.cuda.synchronize()
            t = time.perf_counter()
            ext.run(out, kind, blocks, threads, iters)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            total = blocks * warps * iters * 8 * flops
            print(f"{name}, {warps} warps a block x {blocks}: {total / dt / 1e12:.1f} TFLOPS", flush=True)
    x = torch.empty((1 << 28,), dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    y.copy_(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()
    print(f"DRAM copy: {2 * x.numel() * 10 / (time.perf_counter() - t) / 1e9:.0f} GB/s (read + write)", flush=True)


if __name__ == "__main__":
    main()
