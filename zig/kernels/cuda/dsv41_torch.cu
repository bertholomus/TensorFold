// DeepSeek-V4.1 (zig/src/families/dsv41): the torch arithmetic of the served forward's compressed-attention layers, at
// torch's own rounding points, so each output has the served bytes: ops.rms_norm (its mean in torch's reduction order),
// _compress's softmax over a ratio-2 pair with the weighted sum, and ops.rope_ (a complex64 multiply). Every operation
// is an explicit round-to-nearest intrinsic (no contraction, no fast math); the RoPE multiply's fused multiply-adds are
// the ones torch's own build makes. Built like torch_ops (--fmad=false --ftz=false).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define TF_FULL_MASK 0xffffffffu

// ops.rms_norm(x, w, eps) of rows of D bf16 values (D a multiple of 4, at least 128): out = bf16(w * (x * r)),
// r = rsqrt(mean(x * x) + eps). The mean is torch's reduce_kernel's for a contiguous last dim (Reduce.cuh,
// vectorize_input): lane l of the row's L lanes (blockDim.x) sums the 4-wide vectors l, l + L, ... into four
// accumulators, combined in order; lanes above a warp fold in through shared memory (offsets L/2 .. 32), then a warp
// shuffle with decreasing offsets; times `factor` (torch's float(rows) / (rows * D)). Block (L, H): a row a y.
extern "C" __global__ void tf_ds_rms_norm_kernel(const __nv_bfloat16* x, long long ldx, const __nv_bfloat16* w,
                                                 __nv_bfloat16* out, long long ldo, int rows, int D, float factor,
                                                 float eps) {
    extern __shared__ float shared[];   // L * H floats when L > 32
    __shared__ float rs[64];            // each row's r
    const int L = blockDim.x;
    const int lane = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const bool live = row < rows;
    const __nv_bfloat16* xr = x + (long long)(live ? row : 0) * ldx;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    if (live) {
        for (int idx = lane; idx * 4 + 3 < D; idx += L) {
            const float a = __bfloat162float(xr[idx * 4 + 0]);
            const float b = __bfloat162float(xr[idx * 4 + 1]);
            const float c = __bfloat162float(xr[idx * 4 + 2]);
            const float d = __bfloat162float(xr[idx * 4 + 3]);
            acc0 = __fadd_rn(acc0, __fmul_rn(a, a));
            acc1 = __fadd_rn(acc1, __fmul_rn(b, b));
            acc2 = __fadd_rn(acc2, __fmul_rn(c, c));
            acc3 = __fadd_rn(acc3, __fmul_rn(d, d));
        }
    }
    float v = __fadd_rn(__fadd_rn(__fadd_rn(acc0, acc1), acc2), acc3);
    int dim_x = L;
    if (L > 32) {
        const int base = threadIdx.x + threadIdx.y * L;
        shared[base] = v;
        for (int offset = L / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (lane < offset && lane + offset < L) {
                v = __fadd_rn(v, shared[base + offset]);
                shared[base] = v;
            }
        }
        dim_x = 32;
    }
    __syncthreads();
    for (int offset = dim_x >> 1; offset > 0; offset >>= 1) v = __fadd_rn(v, __shfl_down_sync(TF_FULL_MASK, v, offset));
    if (lane == 0) rs[threadIdx.y] = rsqrtf(__fadd_rn(__fmul_rn(v, factor), eps));
    __syncthreads();
    if (!live) return;
    const float r = rs[threadIdx.y];
    __nv_bfloat16* o = out + (long long)row * ldo;
    for (int j = lane; j < D; j += L) {
        const float y = __fmul_rn(__bfloat162float(xr[j]), r);
        o[j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[j]), y));
    }
}

// _compress at ratio 2 after its two fp32 projections: kv, score [G, 2, C] -> out bf16 [G, C] =
// bf16((kv * score.softmax(dim=1)).sum(1)) at torch's rounding points: the spatial softmax over the pair (max, the
// sum of exp(s - max) from 0, then exp(s - max) / sum), the products, and the pair's sum as reduce_kernel's
// thread_reduce_impl adds it (four accumulators from 0, two of them used, combined in order).
extern "C" __global__ void tf_ds_compress2_kernel(const float* kv, const float* score, __nv_bfloat16* out, int G, int C) {
    const long long n = (long long)G * C;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long g = i / C, c = i % C;
        const float s0 = score[(g * 2) * C + c];
        const float s1 = score[(g * 2 + 1) * C + c];
        float m = -3.402823466e+38f;   // numeric_limits<float>::lowest()
        m = m < s0 ? s0 : m;
        m = m < s1 ? s1 : m;
        float sum = 0.f;
        sum = __fadd_rn(sum, expf(__fsub_rn(s0, m)));
        sum = __fadd_rn(sum, expf(__fsub_rn(s1, m)));
        const float p0 = __fdiv_rn(expf(__fsub_rn(s0, m)), sum);
        const float p1 = __fdiv_rn(expf(__fsub_rn(s1, m)), sum);
        const float y0 = __fmul_rn(kv[(g * 2) * C + c], p0);
        const float y1 = __fmul_rn(kv[(g * 2 + 1) * C + c], p1);
        float t = __fadd_rn(__fadd_rn(0.f, y0), __fadd_rn(0.f, y1));
        t = __fadd_rn(t, 0.f);
        t = __fadd_rn(t, 0.f);
        out[i] = __float2bfloat16_rn(t);
    }
}

// ops.rope_(x[..., off : off + 2 * half], f) of n rows (bf16, row stride ldx), in place: (a + b i)(c + d i), d negated
// for the inverse (f.conj()), each part rounded to bf16. f = complex(cos, sin) at table row pos[r] (fp32 [*, half]).
// torch's complex<float> multiply is a*c - b*d, a*d + b*c, and its CUDA build contracts each into one fused
// multiply-add over the other product: fma(a, c, -(b*d)) and fma(b, c, a*d) (the bytes of torch.mul on GB10, the
// four other roundings of the pair differ: zrec_torchlab.py).
extern "C" __global__ void tf_ds_rope_kernel(__nv_bfloat16* x, long long ldx, int off, const float* cos,
                                             const float* sin, const long long* pos, int n, int half, int inverse) {
    const long long total = (long long)n * half;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / half, k = i % half;
        __nv_bfloat16* p = x + r * ldx + off + 2 * k;
        const float a = __bfloat162float(p[0]);
        const float b = __bfloat162float(p[1]);
        const long long t = pos[r] * half + k;
        const float c = cos[t];
        const float d = inverse ? -sin[t] : sin[t];
        p[0] = __float2bfloat16_rn(__fmaf_rn(a, c, -__fmul_rn(b, d)));
        p[1] = __float2bfloat16_rn(__fmaf_rn(b, c, __fmul_rn(a, d)));
    }
}

// The indexer's head weights (model.py attention_k): wts = wl.to(bf16) * s, wl fp32: rounded to bf16, then times the
// Python scalar as torch's mul with a scalar computes it (fp32 opmath: float(bf16) * float(s)), rounded to bf16 again.
extern "C" __global__ void tf_ds_bf16_scale_kernel(const float* in, __nv_bfloat16* out, uint64_t count, float s) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += uint64_t(gridDim.x) * blockDim.x)
        out[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(__float2bfloat16_rn(in[i])), s));
}
