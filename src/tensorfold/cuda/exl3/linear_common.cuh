// What the EXL3 linear kernels share (linear.cu: a launch a layer; linear_grouped.cu: grouped launches): input and
// output conversions, the input rotation, the output epilogue and the trellis step loads. The rotation and the epilogue
// spell out the products nvcc used to contract on its own (fma), so every kernel built from them makes the same bits,
// today's included, whatever a compiler would contract in another context.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "decode.cuh"

namespace {

using namespace tf_exl3;

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// One row's 128 rotated inputs from x and suh (4 a lane): ((x * suh) @ H) / sqrt(128) in fp32. The first butterfly
// takes the products as fma(x0, s0, +-(x1 s1)) and fma(x2, s2, +-(x3 s3)), the contraction nvcc made of rot_in's plain
// code (v *= s, then fwht128), written out so that rot_in and rot_many make the same bits.
__device__ __forceinline__ void rot128_in(float (&v)[4], const float (&s)[4], int lane) {
    const float p1 = __fmul_rn(v[1], s[1]), p3 = __fmul_rn(v[3], s[3]);
    const float a = __fmaf_rn(v[0], s[0], p1), b = __fmaf_rn(v[0], s[0], -p1);
    const float c = __fmaf_rn(v[2], s[2], p3), d = __fmaf_rn(v[2], s[2], -p3);
    v[0] = __fadd_rn(a, c);
    v[1] = __fadd_rn(b, d);
    v[2] = __fsub_rn(a, c);
    v[3] = __fsub_rn(b, d);
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(v[j], HAD_SCALE);
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias (the
// last product and the bias as one fma, as nvcc contracted the plain code).
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float t = __fmul_rn(v[j], HAD_SCALE), sv = __half2float(__ldg(svh + col + j));
        // Speculative loads must have a valid address even when the split-K result has no bias.
        const half bv = __ldg((bias ? bias : svh) + col + j);
        v[j] = bias ? __fmaf_rn(t, sv, __half2float(bv)) : __fmul_rn(t, sv);
    }
}

// A lane's words of one k step: at 1 and 2 bits the warp loads the step together and each lane takes its words by shuffle.
template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

// Tile j's lane words from a loaded step; prev: the lane's first window starts in the word before its own.
template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;                  // lanes whose windows end in the same word
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;                      // a window inside one word never reads w[1]
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

}  // namespace

#define TF_EXL3_WIDTHS(X, CB) X(2, CB) X(4, CB) X(6, CB) X(8, CB) X(10, CB) X(12, CB) X(14, CB) X(16, CB)
#define TF_EXL3_ALL(X) TF_EXL3_WIDTHS(X, 0) TF_EXL3_WIDTHS(X, 1) TF_EXL3_WIDTHS(X, 2) X(3, 2) X(5, 2) X(7, 2)
