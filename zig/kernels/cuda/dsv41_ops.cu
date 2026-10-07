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
