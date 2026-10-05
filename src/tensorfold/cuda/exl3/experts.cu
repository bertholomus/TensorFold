// EXL3 routed experts, any codebook and a width per expert: fixed-order splits, slots and butterflies, no atomics; 4-bit mcg matches GLM's kernel bit for bit.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "experts_grouped.cuh"

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ members,
                                                              int R, int slots, int E, int maxm) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;                                   // exclusive per warp
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

// The same grouping for any number of rows (prompt chunks; group_kernel stages every pick in one block's shared memory):
// a block per expert counts its members, then a block per used expert takes its place in id order and lists its
// members in row order, -1 after the last.
constexpr int PLACE_THREADS = 256;

__device__ __forceinline__ int block_sum(int v, int* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    int s = 0;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += sh[w];
    return s;
}

__global__ void __launch_bounds__(PLACE_THREADS) group_count_kernel(const int* __restrict__ pick,
                                                                    int* __restrict__ counts, int n) {
    __shared__ int sh[PLACE_THREADS / 32];
    const int e = blockIdx.x;
    int c = 0;
    for (int i = threadIdx.x; i < n; i += PLACE_THREADS) c += pick[i] == e;
    c = block_sum(c, sh);
    if (threadIdx.x == 0) counts[e] = c;
}

__global__ void __launch_bounds__(PLACE_THREADS) group_place_kernel(const int* __restrict__ pick,
                                                                    const int* __restrict__ counts,
                                                                    int* __restrict__ uids, int* __restrict__ ucount,
                                                                    int* __restrict__ members, int n, int slots, int E,
                                                                    int maxm) {
    __shared__ int sh[PLACE_THREADS / 32];
    __shared__ int warp_tot[PLACE_THREADS / 32];
    const int e = blockIdx.x;
    int before = 0;                                   // used experts with a smaller id: this one's place
    for (int i = threadIdx.x; i < e; i += PLACE_THREADS) before += counts[i] > 0;
    const int u = block_sum(before, sh);
    if (e == 0) {
        int used = 0;
        for (int i = threadIdx.x; i < E; i += PLACE_THREADS) used += counts[i] > 0;
        used = block_sum(used, sh);
        if (threadIdx.x == 0) ucount[0] = used;
    }
    if (counts[e] == 0) return;
    if (threadIdx.x == 0) uids[u] = e;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int base = 0;
    for (int start = 0; start < n; start += PLACE_THREADS) {
        const int i = start + threadIdx.x;
        const int hit = i < n && pick[i] == e;
        int inc = hit;                                // inclusive scan of hits in thread order
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += v;
        }
        __syncthreads();
        if (lane == 31) warp_tot[warp] = inc;
        __syncthreads();
        int prior = 0, total = 0;
        for (int w = 0; w < PLACE_THREADS / 32; ++w) {
            prior += w < warp ? warp_tot[w] : 0;
            total += warp_tot[w];
        }
        const int j = base + prior + inc - hit;
        if (hit && j < maxm) members[u * maxm + j] = (i / slots) * 32 + (i % slots);
        base += total;
    }
    for (int j = base + threadIdx.x; j < maxm; j += PLACE_THREADS) members[u * maxm + j] = -1;
}

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
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

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32.
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int E) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits). With
// store_y the routed slots' outputs go to y and a non-routed slot's comes from it (the caller's); without (prompt
// chunks: nothing reads them after the combine) y is neither written nor read and a non-routed slot adds w * 0. With
// add (a [rows, D] fp32 term, e.g. the shared expert's output) each combined row gets it added last, the add the
// caller would make after this launch.
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, const float* __restrict__ add,
                                    float* __restrict__ out, int P, int D, int SK, int E, int slots, int store_y) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            if (store_y) y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = store_y ? y[(size_t)p * D + n + j] : 0.f;
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        out[(size_t)r * D + n + j] = add ? __fadd_rn(acc[j], add[(size_t)r * D + n + j]) : acc[j];
}

// out[r][d] = the sum over slots in order of the routed slots' weighted bf16 outputs Y (grouped_down3), fp32, then add.
__global__ void combine_y_kernel(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ pick,
                                 const float* __restrict__ add, float* __restrict__ out, int D, int slots, int E) {
    const int r = blockIdx.x;
    const int d = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (d >= D) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < slots; ++k) {
        const int e = pick[r * slots + k];
        if (e < 0 || e >= E) continue;
        const uint2 v = *reinterpret_cast<const uint2*>(Y + ((size_t)r * slots + k) * D + d);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        acc[0] += a.x; acc[1] += a.y; acc[2] += b.x; acc[3] += b.y;
    }
    float4 o;
    if (add) {
        const float4 s = *reinterpret_cast<const float4*>(add + (size_t)r * D + d);
        o = make_float4(__fadd_rn(acc[0], s.x), __fadd_rn(acc[1], s.y), __fadd_rn(acc[2], s.z), __fadd_rn(acc[3], s.w));
    } else {
        o = make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + d) = o;
}

// Decode windows' first launch: block 0 groups the picks (group_kernel's output: distinct experts < E in id order,
// members row * 32 + slot in row order, -1 after the last), each pick in parallel: its expert's place is the number of
// distinct routed experts with a smaller id, its member position the number of earlier picks of the same expert; with
// wts it also combines the rows that route no slot (down_combine's arithmetic: their slots' y, or 0). The other blocks
// run rot_in_kernel's programs, a warp each (the same arithmetic).
constexpr int PREP_THREADS = 256;

template <typename TIN>
__global__ void __launch_bounds__(PREP_THREADS) decode_prep_kernel(
    const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick, const half* __restrict__ suh0,
    const half* __restrict__ suh1, half* __restrict__ out0, half* __restrict__ out1, int K, int slots, int E, int R,
    int* __restrict__ uids, int* __restrict__ ucount, int* __restrict__ members, int maxm,
    const float* __restrict__ wts, const float* __restrict__ y, const float* __restrict__ add, float* __restrict__ out,
    int D, int store_y, int* __restrict__ epoch) {
    const int n = R * slots;
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // gate/up (it waits for this grid's end)
    if (blockIdx.x == 0 && threadIdx.x == 0 && epoch != nullptr) epoch[0] += 1;     // this launch's number
    if (blockIdx.x == 0) {
        extern __shared__ int sh_prep[];
        int* sp = sh_prep;
        int* first = sh_prep + n;
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) sp[i] = pick[i];
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) {
            const int e = sp[i];
            int f = e >= 0 && e < E;
            for (int j = 0; f && j < i; ++j) f = sp[j] != e;
            first[i] = f;
        }
        __syncthreads();
        int used = 0;
        for (int i = threadIdx.x; i < n; i += PREP_THREADS) {
            const int e = sp[i];
            if (e < 0 || e >= E) continue;
            int place = 0, pos = 0, tot = 0;
            for (int j = 0; j < n; ++j) {
                const int ej = sp[j];
                place += first[j] && ej < e;
                tot += ej == e;
                pos += ej == e && j < i;
            }
            if (pos < maxm) members[place * maxm + pos] = (i / slots) * 32 + (i % slots);
            if (first[i]) {
                uids[place] = e;
                for (int k = tot; k < maxm; ++k) members[place * maxm + k] = -1;
                ++used;
            }
        }
        __shared__ int used_sh;
        if (threadIdx.x == 0) used_sh = 0;
        __syncthreads();
        if (used) atomicAdd(&used_sh, used);
        __syncthreads();
        if (threadIdx.x == 0) ucount[0] = used_sh;
        if (wts == nullptr) return;
        for (int r = 0; r < R; ++r) {
            int routed = 0;
            for (int q = 0; q < slots; ++q) routed += sp[r * slots + q] >= 0 && sp[r * slots + q] < E;
            if (routed) continue;
            for (int d = threadIdx.x * 4; d < D; d += PREP_THREADS * 4) {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                for (int q = 0; q < slots; ++q) {
                    const float w = wts[r * slots + q];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc[j] = fmaf(w, store_y ? y[((size_t)r * slots + q) * D + d + j] : 0.f, acc[j]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    out[(size_t)r * D + d + j] = add ? __fadd_rn(acc[j], add[(size_t)r * D + d + j]) : acc[j];
            }
        }
        return;
    }
    // rot_in_kernel's program (member row p, 128-block blk, matrix mat), one a warp
    const int nblk = K / 128;
    const int item = (blockIdx.x - 1) * (PREP_THREADS / 32) + (threadIdx.x >> 5);
    if (item >= n * nblk * 2) return;
    const int p = item / (nblk * 2), mat = (item / nblk) % 2, blk = item % nblk;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x & 31;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

namespace tf_exl3x {
extern template void grouped_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void grouped_cp_launch<0>(const GroupedArgs&, const DecodeEpi&, int, int, cudaStream_t);
extern template void grouped_cp_launch<1>(const GroupedArgs&, const DecodeEpi&, int, int, cudaStream_t);
extern template void grouped_cp_launch<2>(const GroupedArgs&, const DecodeEpi&, int, int, cudaStream_t);
extern template void grouped_rows_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_rows_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_rows_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void grouped_mma_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_mma_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_mma_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void grouped_mma2_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_mma2_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_mma2_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void grouped_mma3_launch<0>(const GroupedArgs&, const Mma3Args&, cudaStream_t);
extern template void grouped_mma3_launch<1>(const GroupedArgs&, const Mma3Args&, cudaStream_t);
extern template void grouped_mma3_launch<2>(const GroupedArgs&, const Mma3Args&, cudaStream_t);
extern template void grouped_down3_launch<0>(const GroupedArgs&, const half*, const float*, __nv_bfloat16*, cudaStream_t);
extern template void grouped_down3_launch<1>(const GroupedArgs&, const half*, const float*, __nv_bfloat16*, cudaStream_t);
extern template void grouped_down3_launch<2>(const GroupedArgs&, const half*, const float*, __nv_bfloat16*, cudaStream_t);
extern template void dequant_launch<0>(const uint32_t*, half*, int, int, int, cudaStream_t);
extern template void dequant_launch<1>(const uint32_t*, half*, int, int, int, cudaStream_t);
extern template void dequant_launch<2>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x

void exl3x_grouped_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                        const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                        const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                        int64_t SK, int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo,
                        int64_t hi, int64_t cp) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = (int)nt; a.warps = (int)warps; a.pf = (int)pf; a.lo = (int)lo; a.hi = (int)hi;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cp) {
        // the 16-byte copies need every trellis 16-byte aligned (checked once per layer by the caller)
        const tf_exl3x::DecodeEpi ep;
        if (cb == 0) tf_exl3x::grouped_cp_launch<0>(a, ep, 0, 0, stream);
        else if (cb == 1) tf_exl3x::grouped_cp_launch<1>(a, ep, 0, 0, stream);
        else if (cb == 2) tf_exl3x::grouped_cp_launch<2>(a, ep, 0, 0, stream);
        else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    } else {
        if (cb == 0) tf_exl3x::grouped_launch<0>(a, stream);
        else if (cb == 1) tf_exl3x::grouped_launch<1>(a, stream);
        else if (cb == 2) tf_exl3x::grouped_launch<2>(a, stream);
        else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_grouped_rows_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                             const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids,
                             const at::Tensor& ucount, const at::Tensor& members, at::Tensor& Z, int64_t mats,
                             int64_t K, int64_t N, int64_t P, int64_t SK, int64_t slots, int64_t cb, int64_t nt,
                             int64_t warps, int64_t pf, int64_t g, int64_t lo, int64_t hi, int64_t fold) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    TORCH_CHECK(P * K < (int64_t)1 << 31, "rows x K must fit 32-bit offsets");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = (int)nt; a.warps = (int)warps; a.pf = (int)pf; a.lo = (int)lo; a.hi = (int)hi;
    a.g = (int)g;
    a.fold = (int)fold;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_rows_launch<0>(a, stream);
    else if (cb == 1) tf_exl3x::grouped_rows_launch<1>(a, stream);
    else if (cb == 2) tf_exl3x::grouped_rows_launch<2>(a, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_grouped_mma_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                            const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids,
                            const at::Tensor& ucount, const at::Tensor& members, at::Tensor& Z, int64_t mats,
                            int64_t K, int64_t N, int64_t P, int64_t SK, int64_t slots, int64_t cb, int64_t warps,
                            int64_t lo, int64_t hi, int64_t fold) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % tf_exl3x::MMA_COLS == 0, "K and N must split evenly");
    TORCH_CHECK(fold || SK == 1, "an unfolded prompt mma launch writes one split");
    TORCH_CHECK(P * K < (int64_t)1 << 31, "rows x K must fit 32-bit offsets");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = tf_exl3x::MMA_COLS / 16; a.warps = (int)warps; a.pf = tf_exl3x::MMA_KB;
    a.lo = (int)lo; a.hi = (int)hi;
    a.fold = (int)fold;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_mma_launch<0>(a, stream);
    else if (cb == 1) tf_exl3x::grouped_mma_launch<1>(a, stream);
    else if (cb == 2) tf_exl3x::grouped_mma_launch<2>(a, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_grouped_mma2_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                             const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids,
                             const at::Tensor& ucount, const at::Tensor& members, at::Tensor& Z, int64_t mats,
                             int64_t K, int64_t N, int64_t P, int64_t slots, int64_t cb, int64_t lo, int64_t hi) {
    TORCH_CHECK(P * K < (int64_t)1 << 31, "rows x K must fit 32-bit offsets");
    const size_t smem = tf_exl3x::m2_smem_bytes((int)hi);
    int dev = 0, limit = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK(smem + 512 <= (size_t)limit, "prompt mma2 needs ", smem, " bytes of shared memory at ", hi / 2.0,
                " bits; this device allows ", limit);
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = 1; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = tf_exl3x::M2_TILES; a.warps = 8; a.pf = 0;
    a.lo = (int)lo; a.hi = (int)hi;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_mma2_launch<0>(a, stream);
    else if (cb == 1) tf_exl3x::grouped_mma2_launch<1>(a, stream);
    else if (cb == 2) tf_exl3x::grouped_mma2_launch<2>(a, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_grouped_mma3_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& suh0, const at::Tensor& suh1,
                             const at::Tensor& TP0, const at::Tensor& TP1, const at::Tensor& B0, const at::Tensor& B1,
                             const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members,
                             at::Tensor& Z, int64_t K, int64_t N, int64_t P, int64_t slots, int64_t cb, int64_t lo,
                             int64_t hi, int64_t nw) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "prompt mma3: bf16 rows");
    TORCH_CHECK(x_stride % 4 == 0, "prompt mma3: a row stride of whole 8-byte words");
    tf_exl3x::GroupedArgs a;
    a.x0 = nullptr;
    a.x1 = nullptr;
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = 1; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = 2; a.nt = tf_exl3x::M2_TILES; a.warps = 8; a.pf = 0;
    a.lo = (int)lo; a.hi = (int)hi; a.g = (int)nw;
    tf_exl3x::Mma3Args m;
    m.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    m.x_stride = (int)x_stride;
    m.suh0 = reinterpret_cast<const half*>(suh0.data_ptr());
    m.suh1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_mma3_launch<0>(a, m, stream);
    else if (cb == 1) tf_exl3x::grouped_mma3_launch<1>(a, m, stream);
    else if (cb == 2) tf_exl3x::grouped_mma3_launch<2>(a, m, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_grouped_down3_cuda(const at::Tensor& Xd, const at::Tensor& TP, const at::Tensor& B, const at::Tensor& uids,
                              const at::Tensor& ucount, const at::Tensor& members, const at::Tensor& svh,
                              const at::Tensor& wts, at::Tensor& Y, int64_t K, int64_t N, int64_t P, int64_t slots,
                              int64_t cb, int64_t lo, int64_t hi) {
    TORCH_CHECK(P * K < (int64_t)1 << 31, "rows x K must fit 32-bit offsets");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(Xd.data_ptr());
    a.x1 = a.x0;
    a.tp0 = TP.data_ptr<int64_t>();
    a.tp1 = a.tp0;
    a.k2_0 = B.data_ptr<int>();
    a.k2_1 = a.k2_0;
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = nullptr;
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = 1; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = 1; a.nt = tf_exl3x::M2_TILES; a.warps = 8; a.pf = 0;
    a.lo = (int)lo; a.hi = (int)hi;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto sv = reinterpret_cast<const half*>(svh.data_ptr());
    auto y = reinterpret_cast<__nv_bfloat16*>(Y.data_ptr());
    if (cb == 0) tf_exl3x::grouped_down3_launch<0>(a, sv, wts.data_ptr<float>(), y, stream);
    else if (cb == 1) tf_exl3x::grouped_down3_launch<1>(a, sv, wts.data_ptr<float>(), y, stream);
    else if (cb == 2) tf_exl3x::grouped_down3_launch<2>(a, sv, wts.data_ptr<float>(), y, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_combine_y_cuda(const at::Tensor& Y, const at::Tensor& pick, const at::Tensor& add, at::Tensor& out,
                          int64_t rows, int64_t D, int64_t slots, int64_t E, int64_t has_add) {
    TORCH_CHECK(D % 4 == 0, "combine_y: D a multiple of 4");
    dim3 grid((unsigned)rows, (unsigned)((D / 4 + 255) / 256));
    combine_y_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Y.data_ptr()), pick.data_ptr<int>(),
        has_add ? add.data_ptr<float>() : nullptr, out.data_ptr<float>(), (int)D, (int)slots, (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_dequant_cuda(const at::Tensor& T, at::Tensor& out, int64_t K, int64_t N, int64_t k2, int64_t cb) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto t = reinterpret_cast<const uint32_t*>(T.data_ptr());
    auto o = reinterpret_cast<half*>(out.data_ptr());
    if (cb == 0) tf_exl3x::dequant_launch<0>(t, o, (int)K, (int)N, (int)k2, stream);
    else if (cb == 1) tf_exl3x::dequant_launch<1>(t, o, (int)K, (int)N, (int)k2, stream);
    else tf_exl3x::dequant_launch<2>(t, o, (int)K, (int)N, (int)k2, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_group_cuda(const at::Tensor& pick, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members, int64_t R,
                      int64_t slots, int64_t E) {
    TORCH_CHECK(E <= GROUP_THREADS * GROUP_PER_THREAD, "too many experts for the grouping kernel");
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    // The picks are staged in dynamic shared memory: prompt chunks (2048 rows x 9 slots = 72 KiB) pass the
    // 48 KiB a launch gets by default, so opt in up to the device's per-block limit (GB10: 99 KiB) and refuse
    // clearly past it rather than failing the launch with "invalid argument".
    const size_t smem = (size_t)R * slots * sizeof(int);
    if (smem > 48 * 1024) {
        int dev = 0, limit = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        TORCH_CHECK(smem <= (size_t)limit, "grouping ", R, " rows x ", slots, " slots needs ", smem,
                    " bytes of shared memory; this device allows ", limit, " a block");
        C10_CUDA_CHECK(cudaFuncSetAttribute(group_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    }
    group_kernel<<<1, GROUP_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        pick.data_ptr<int>(), uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), (int)R,
        (int)slots, (int)E, (int)members.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_group_count_cuda(const at::Tensor& pick, at::Tensor& counts, int64_t R, int64_t slots, int64_t E) {
    group_count_kernel<<<(unsigned)E, PLACE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        pick.data_ptr<int>(), counts.data_ptr<int>(), (int)(R * slots));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_group_place_cuda(const at::Tensor& pick, const at::Tensor& counts, at::Tensor& uids, at::Tensor& ucount,
                            at::Tensor& members, int64_t R, int64_t slots, int64_t E) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    group_place_kernel<<<(unsigned)E, PLACE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        pick.data_ptr<int>(), counts.data_ptr<int>(), uids.data_ptr<int>(), ucount.data_ptr<int>(),
        members.data_ptr<int>(), (int)(R * slots), (int)slots, (int)E, (int)members.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_rot_in_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
                       const at::Tensor& suh1, at::Tensor& out0, at::Tensor& out1, int64_t rows, int64_t K,
                       int64_t slots, int64_t E) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(K / 128), 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
    auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto o0 = reinterpret_cast<half*>(out0.data_ptr());
    auto o1 = reinterpret_cast<half*>(out1.data_ptr());
    if (x.scalar_type() == at::kBFloat16)
        rot_in_kernel<__nv_bfloat16><<<grid, 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                                              (int)x_stride, pick.data_ptr<int>(), s0, s1, o0, o1,
                                                              (int)K, (int)slots, (int)E);
    else
        rot_in_kernel<half><<<grid, 32, 0, stream>>>(reinterpret_cast<const half*>(x.data_ptr()), (int)x_stride,
                                                     pick.data_ptr<int>(), s0, s1, o0, o1, (int)K, (int)slots,
                                                     (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_gateup_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g,
                                const at::Tensor& svh_u, const at::Tensor& suh_d, at::Tensor& xd, int64_t rows,
                                int64_t P, int64_t N, int64_t SK, int64_t slots, int64_t E, double limit,
                                int64_t act_mode) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(N / 128));
    gateup_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_g.data_ptr()),
        reinterpret_cast<const half*>(svh_u.data_ptr()), reinterpret_cast<const half*>(suh_d.data_ptr()),
        reinterpret_cast<half*>(xd.data_ptr()), (int)P, (int)N, (int)SK, (int)E, (float)limit, (int)act_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                              int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(D / 128));
    down_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
        y.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_combine_cuda(const at::Tensor& y, const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t D,
                        int64_t slots) {
    dim3 grid((unsigned)rows, (unsigned)((D + 255) / 256));
    combine_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<float>(), wts.data_ptr<float>(),
                                                                        out.data_ptr<float>(), (int)D, (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_combine_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                             const at::Tensor& wts, const at::Tensor& add, at::Tensor& out, int64_t rows, int64_t P,
                             int64_t D, int64_t SK, int64_t slots, int64_t E, int64_t store_y, int64_t has_add) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    down_combine_kernel<<<grid, (unsigned)(32 * slots), 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
        y.data_ptr<float>(), wts.data_ptr<float>(), has_add ? add.data_ptr<float>() : nullptr, out.data_ptr<float>(),
        (int)P, (int)D, (int)SK, (int)E, (int)slots, (int)store_y);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_decode_prep_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
                            const at::Tensor& suh1, at::Tensor& out0, at::Tensor& out1, at::Tensor& uids,
                            at::Tensor& ucount, at::Tensor& members, int64_t rows, int64_t K, int64_t slots, int64_t E,
                            const at::Tensor& wts, const at::Tensor& y, const at::Tensor& add, at::Tensor& out,
                            int64_t has_wts, int64_t has_add, int64_t store_y, at::Tensor& epoch, int64_t has_epoch) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    TORCH_CHECK(K % 128 == 0, "K a multiple of 128");
    const int n = (int)(rows * slots);
    const size_t smem = (size_t)2 * n * sizeof(int);
    TORCH_CHECK(smem <= 48 * 1024, "decode prep: ", n, " picks are too many for one grouping block");
    const int items = n * (int)(K / 128) * 2;
    const unsigned blocks = 1 + (unsigned)((items + PREP_THREADS / 32 - 1) / (PREP_THREADS / 32));
    const int D = has_wts ? (int)out.size(1) : 0;
    TORCH_CHECK(!has_wts || D % 4 == 0, "decode prep: D a multiple of 4");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
    auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto o0 = reinterpret_cast<half*>(out0.data_ptr());
    auto o1 = reinterpret_cast<half*>(out1.data_ptr());
    const float* w = has_wts ? wts.data_ptr<float>() : nullptr;
    const float* yp = has_wts ? y.data_ptr<float>() : nullptr;
    const float* ad = has_add ? add.data_ptr<float>() : nullptr;
    float* op = has_wts ? out.data_ptr<float>() : nullptr;
#define TF_PREP(T_)                                                                                             \
    decode_prep_kernel<T_><<<blocks, PREP_THREADS, smem, stream>>>(                                             \
        reinterpret_cast<const T_*>(x.data_ptr()), (int)x_stride, pick.data_ptr<int>(), s0, s1, o0, o1, (int)K,  \
        (int)slots, (int)E, (int)rows, uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(),    \
        (int)members.size(1), w, yp, ad, op, D, (int)store_y, has_epoch ? epoch.data_ptr<int>() : nullptr)
    if (x.scalar_type() == at::kBFloat16) TF_PREP(__nv_bfloat16);
    else TF_PREP(half);
#undef TF_PREP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// grouped_cp_kernel with a decode epilogue: epi 1 (gate/up: svh0 = svh_g, svh1 = svh_u, suh_d, xd), epi 2 (down, one
// split: svh0 = svh_d, y, and with has_wts the combine into out, + add with has_add); pdl: launched as a programmatic
// dependent; use_ready: down waits per expert (ready, ready_cnt, epoch: the scratch's, epoch counted by decode_prep)
void exl3x_grouped_decode_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0,
                               const at::Tensor& TP1, const at::Tensor& B0, const at::Tensor& B1,
                               const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members,
                               at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK, int64_t slots,
                               int64_t cb, int64_t stages, int64_t lo, int64_t hi, int64_t epi, const at::Tensor& pick,
                               int64_t E, const at::Tensor& svh0, const at::Tensor& svh1, const at::Tensor& suh_d,
                               at::Tensor& xd, double limit, int64_t act_mode, at::Tensor& y, const at::Tensor& wts,
                               const at::Tensor& add, at::Tensor& out, int64_t has_wts, int64_t has_add,
                               int64_t store_y, at::Tensor& cnt, int64_t pdl, at::Tensor& ready,
                               at::Tensor& ready_cnt, at::Tensor& epoch, int64_t use_ready, int64_t discard) {
    TORCH_CHECK(K % (16 * SK * 4) == 0 && N % 128 == 0, "K and N must split evenly");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = 8; a.warps = 4; a.pf = (int)stages; a.lo = (int)lo; a.hi = (int)hi;
    tf_exl3x::DecodeEpi ep;
    if (use_ready) {
        ep.ready = ready.data_ptr<int>();
        ep.ready_cnt = ready_cnt.data_ptr<int>();
        ep.epoch = epoch.data_ptr<int>();
        TORCH_CHECK(ready.numel() >= a.nexp_max && ready_cnt.numel() >= a.nexp_max, "ready flags too small");
    }
    ep.pick = pick.data_ptr<int>();
    ep.E = (int)E;
    ep.cnt = cnt.data_ptr<int>();
    // EPI 1: 2 * SK (mat, split) partial rows a member (<= 32 lanes); EPI 2: only with the combine (y dead after it)
    ep.discard = (int)(discard && ((epi == 1 && 8 * SK <= 32) || (epi == 2 && has_wts && 4 * slots <= 32)));
    if (epi == 1) {
        ep.svh_g = reinterpret_cast<const half*>(svh0.data_ptr());
        ep.svh_u = reinterpret_cast<const half*>(svh1.data_ptr());
        ep.suh_d = reinterpret_cast<const half*>(suh_d.data_ptr());
        ep.xd = reinterpret_cast<half*>(xd.data_ptr());
        ep.limit = (float)limit;
        ep.act_mode = (int)act_mode;
        TORCH_CHECK(cnt.numel() >= a.nexp_max * (N / 128), "decode counters too small");
    } else if (epi == 2) {
        ep.svh_d = reinterpret_cast<const half*>(svh0.data_ptr());
        ep.y = y.data_ptr<float>();
        ep.wts = has_wts ? wts.data_ptr<float>() : nullptr;
        ep.add = has_add ? add.data_ptr<float>() : nullptr;
        ep.out = has_wts ? out.data_ptr<float>() : nullptr;
        ep.store_y = (int)store_y;
        TORCH_CHECK(cnt.numel() >= (P / slots) * (N / 128), "decode counters too small");
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_cp_launch<0>(a, ep, (int)epi, (int)pdl, stream);
    else if (cb == 1) tf_exl3x::grouped_cp_launch<1>(a, ep, (int)epi, (int)pdl, stream);
    else if (cb == 2) tf_exl3x::grouped_cp_launch<2>(a, ep, (int)epi, (int)pdl, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
