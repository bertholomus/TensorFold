// DeepSeek-V4.1 (zig/src/families/dsv41): the torch casts its served forward makes that torch_ops/ does not have.
// Built like torch_ops (no fast math, --fmad=false --ftz=false); raw device pointers, the caller's stream.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// `.float()` of fp16 values (exact): the MoE gate's weights before torch's fp32 GEMM (model.py moe, prompt rows).
extern "C" __global__ void tf_ds_f16_to_f32_kernel(const __half* input, float* output, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) output[i] = __half2float(input[i]);
}

// Comm.sum of two ranks' fp32 partials, then `.to(bf16)`: acc = g0 + g1 (one fp32 add, rank order), rounded to nearest
// even into bf16 (model.py engram_apply: comm.sum(mm(engram_wkv, e, F32)).to(BF16)).
extern "C" __global__ void tf_ds_add2_bf16_kernel(const float* g0, const float* g1, __nv_bfloat16* output,
                                                  uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) output[i] = __float2bfloat16_rn(__fadd_rn(g0[i], g1[i]));
}

// torch's keys.topk(k, sorted=False).values for the indexer's int64 keys, a row a block (1024 threads): the k largest
// values are one multiset whatever the order (the keys are unique: score bits above, inverted index below), and
// _topk_finish sorts their indices. A radix select finds the k-th largest key (eight 8-bit digits from the top, signed
// order); every key above it is written to top [rows, k], then copies of it fill the rest (in no particular order).
// keys [rows, n] (row stride ks).
extern "C" __global__ void __launch_bounds__(1024) tf_ds_topk_i64_kernel(const long long* keys, long long ks, int n, int k,
                                                                           long long* top) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long prefix_s, mask_s;
    __shared__ int want_s, count_s;
    const long long* row = keys + (long long)blockIdx.x * ks;
    long long* out = top + (long long)blockIdx.x * k;
    if (threadIdx.x == 0) {
        prefix_s = 0;
        mask_s = 0;
        want_s = k;
        count_s = 0;
    }
    __syncthreads();
    for (int d = 7; d >= 0; --d) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const unsigned long long prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ull;
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> (8 * d)) & 0xff], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int want = want_s;
            int b = 255;
            for (; b > 0; --b) {
                if ((int)hist[b] >= want) break;
                want -= (int)hist[b];
            }
            want_s = want;
            prefix_s = prefix | ((unsigned long long)b << (8 * d));
            mask_s = mask | (0xffull << (8 * d));
        }
        __syncthreads();
    }
    const unsigned long long kth = prefix_s;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ull;
        if (u > kth) out[atomicAdd(&count_s, 1)] = row[i];   // fewer than k
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ull;
        if (u == kth) {
            const int at = atomicAdd(&count_s, 1);
            if (at < k) out[at] = row[i];
        }
    }
}
