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

// A profile's GPU clock (ns, %globaltimer) when the stream reaches it, appended with its tag: buf[0] the count (reset
// first when `reset`), then (clock, tag) pairs, at most `cap` (round phase timing; in a graph like any kernel).
extern "C" __global__ void tf_ds_clock_kernel(uint64_t* buf, uint32_t tag, uint32_t reset, uint32_t cap) {
    if (threadIdx.x == 0) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        const uint64_t i = reset ? 0 : buf[0];
        if (i < cap) {
            buf[1 + 2 * i] = t;
            buf[2 + 2 * i] = tag;
            buf[0] = i + 1;
        }
    }
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

// t[idx] = rows of whole rows (index_put_ with distinct indices: the window ring's and the positional store's slots):
// dst row idx[r] = src row r, `bytes` bytes a row (16-byte words when the rows are aligned so), a row a block.
extern "C" __global__ void tf_ds_scatter_rows_kernel(const uint8_t* src, long long src_ld, const long long* idx,
                                                     uint8_t* dst, long long dst_ld, long long bytes, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const uint8_t* s = src + (long long)r * src_ld;
    uint8_t* d = dst + idx[r] * dst_ld;
    if ((((uintptr_t)s | (uintptr_t)d | (uintptr_t)bytes) & 15) == 0) {
        const uint4* s4 = (const uint4*)s;
        uint4* d4 = (uint4*)d;
        for (long long j = threadIdx.x; j < bytes / 16; j += blockDim.x) d4[j] = s4[j];
    } else {
        for (long long j = threadIdx.x; j < bytes; j += blockDim.x) d[j] = s[j];
    }
}

// topk_indices' key of score x at column j as an unsigned integer in the keys' signed order: x's bits (-0 as +0, as
// score + 0.0 makes it), ordered as signed integers, above the inverted column.
__device__ __forceinline__ unsigned long long tf_ds_score_key(float x, int j) {
    unsigned int bits = __float_as_uint(x);
    if (bits == 0x80000000u) bits = 0u;                        // + 0.0
    const int sb = (int)bits;
    const int ordered = sb < 0 ? (sb ^ 0x7FFFFFFF) : sb;
    const unsigned long long key = ((unsigned long long)(long long)ordered << 32) | (unsigned long long)(0xFFFFFFFFu - (unsigned int)j);
    return key ^ 0x8000000000000000ull;                       // signed order as unsigned
}

// kernels.topk_indices(score, k), then torch.where(top < vis, top, -1) (model.py attention_k, the candidate pool's
// layers), a row a block (1024 threads): each score's int64 key (its bits, -0 as +0 (score + 0.0), ordered as signed
// integers, above the inverted column: a total order, ties to the lower column), the k largest keys by a radix select
// (eight 8-bit digits from the top), their columns ascending (a bitonic sort), a column at or past vis[r] as -1.
// score [rows, n] fp32 (row stride ss), vis [rows] int64, out [rows, k] int64; k <= 2048 and k <= n (the candidate
// pool's top blocks: k = 2048).
extern "C" __global__ void __launch_bounds__(1024) tf_ds_topk_indices_kernel(const float* score, long long ss, int n,
                                                                              int k, const long long* vis,
                                                                              long long* out) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long prefix_s, mask_s;
    __shared__ int want_s, count_s;
    __shared__ long long cols[2048];
    const float* row = score + (long long)blockIdx.x * ss;
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
            const unsigned long long u = tf_ds_score_key(row[i], i);
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
    const unsigned long long kth = prefix_s;   // the keys are unique: exactly k at or above it
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (tf_ds_score_key(row[i], i) >= kth) {
            const int at = atomicAdd(&count_s, 1);
            if (at < k) cols[at] = i;
        }
    }
    int p = 1;
    while (p < k) p <<= 1;
    for (int i = k + threadIdx.x; i < p; i += blockDim.x) cols[i] = 0x7FFFFFFFFFFFFFFFll;
    for (int size = 2; size <= p; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < p; i += blockDim.x) {   // (p up to 2048: two elements a thread)
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = (i & size) == 0;
                    const long long a = cols[i], b = cols[j];
                    if ((a > b) == up) {
                        cols[i] = b;
                        cols[j] = a;
                    }
                }
            }
        }
    }
    __syncthreads();
    const long long v = vis[blockIdx.x];
    long long* o = out + (long long)blockIdx.x * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) o[i] = cols[i] < v ? cols[i] : -1;
}

// The next kernels' weights touched into L2 ahead of them (the served build's paced l2_prefetch, as its trace shows it:
// a block of 128 threads on a stream of its own, up to 16 ranges a launch): every 128-byte line of each range, the
// ranges in order. It writes nothing.
struct TfDsPrefetch {
    const char* ptr[16];
    unsigned long long bytes[16];
    int n;
};
extern "C" __global__ void tf_ds_l2_prefetch_kernel(TfDsPrefetch a) {
    const unsigned long long step = (unsigned long long)blockDim.x * gridDim.x * 128ull;
    for (int t = 0; t < a.n; ++t) {
        const char* p = a.ptr[t];
        const unsigned long long n = a.bytes[t];
        for (unsigned long long off = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 128ull; off < n; off += step) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p + off));
        }
    }
}

// The indexer's selection when its top-k takes every key it scans (nb <= index_topk): kernels.topk_indices of all nb
// keys is 0 .. nb - 1 whatever the scores, then the visible mask: out[r, j] = j if j < vis[r] else -1. vis [rows] int64,
// out [rows, nb] int64; grid (blocks, rows).
extern "C" __global__ void tf_ds_iota_vis_kernel(const long long* vis, long long* out, int nb) {
    const int r = blockIdx.y;
    const long long v = vis[r];
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nb; j += gridDim.x * blockDim.x) {
        out[(long long)r * nb + j] = j < v ? (long long)j : -1;
    }
}

// rounds.py _candidates_fast where the pool takes every block (contexts up to nblocks * bsize keys): a row's block mask
// (torch's bool, as u8) = the block's maximum (amax: a NaN wins) above -inf, or the block is the row's newest,
// (vis - 1) // bsize (floor division). score [rows, nb * bsize] fp32 (row stride ss), vis [rows], out [rows, nb] (row
// stride os); grid (blocks, rows).
extern "C" __global__ void tf_ds_cand_fast_kernel(const float* score, long long ss, int nb, int bsize,
                                                  const long long* vis, uint8_t* out, long long os) {
    const int r = blockIdx.y;
    const long long t = vis[r] - 1;
    const long long last = t >= 0 ? t / bsize : -((-t + bsize - 1) / bsize);
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        const float* s = score + (long long)r * ss + (long long)b * bsize;
        bool nan = false, above = false;
        for (int j = 0; j < bsize; ++j) {
            const float x = s[j];
            if (x != x) nan = true;
            else if (x > -__int_as_float(0x7f800000)) above = true;
        }
        out[(long long)r * os + b] = (uint8_t)((!nan && above) || b == last);
    }
}

// model.py _candidates / rounds.py _candidate_blocks, their first step: a row's block maxima of the scores (amax over
// blocks of bsize, the last block padded with -inf, a NaN wins), its newest block, (vis - 1) // bsize (floor
// division), pinned to +inf. score [rows, width] fp32 (row stride ss), vis [rows] int64, out [rows, nb] fp32 (row
// stride os), nb = ceil(width / bsize); grid (blocks, rows).
extern "C" __global__ void tf_ds_block_max_kernel(const float* score, long long ss, int width, int bsize,
                                                  const long long* vis, float* out, long long os, int nb) {
    const int r = blockIdx.y;
    const long long t = vis[r] - 1;
    const long long last = t >= 0 ? t / bsize : -((-t + bsize - 1) / bsize);
    const float ninf = -__int_as_float(0x7f800000);
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        const float* s = score + (long long)r * ss + (long long)b * bsize;
        float m = ninf;
        bool nan = false;
        for (int j = 0; j < bsize && b * bsize + j < width; ++j) {
            const float x = s[j];
            if (x != x) nan = true;
            else if (x > m) m = x;
        }
        float v = nan ? __int_as_float(0x7fc00000) : m;
        if (b == last) v = __int_as_float(0x7f800000);
        out[(long long)r * os + b] = v;
    }
}

// The pool from a row's top blocks idx [rows, k] (kernels.topk_indices of the block maxima bmax [rows, nb], row stride
// bs): model.py _candidates' mask (u8 [rows, nb], row stride ms: zeros, then each picked block whose maximum is above
// -inf) and/or rounds.py _candidate_blocks' list (int32 [rows, k]: the block, or -1 when its maximum is not above -inf).
// A row a block of threads.
extern "C" __global__ void tf_ds_pool_pick_kernel(const float* bmax, long long bs, int nb, const long long* idx, int k,
                                                  uint8_t* mask, long long ms, int* cblk) {
    const int r = blockIdx.x;
    const float ninf = -__int_as_float(0x7f800000);
    if (mask != nullptr) {
        for (int b = threadIdx.x; b < nb; b += blockDim.x) mask[(long long)r * ms + b] = 0;
        __syncthreads();
    }
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        const long long c = idx[(long long)r * k + j];
        const bool ok = bmax[(long long)r * bs + c] > ninf;
        if (mask != nullptr && ok) mask[(long long)r * ms + c] = 1;
        if (cblk != nullptr) cblk[(long long)r * k + j] = ok ? (int)c : -1;
    }
}

// model.py apply_candidates: -inf at every position whose block the row's pool leaves out (mask u8 [rows, >= ceil(width
// / bsize)], row stride ms), in place. score [rows, width] fp32 (row stride ss); grid (blocks, rows).
extern "C" __global__ void tf_ds_apply_pool_kernel(float* score, long long ss, int width, int bsize, const uint8_t* mask,
                                                   long long ms) {
    const int r = blockIdx.y;
    const float ninf = -__int_as_float(0x7f800000);
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x) {
        if (!mask[(long long)r * ms + j / bsize]) score[(long long)r * ss + j] = ninf;
    }
}

// Rows of n16 16-byte words: dst row r = src row r (row strides in words). torch_ops' tf_strided_copy takes a block a
// row and a byte a thread; this one a word a thread over grid (ceil(n16 / blockDim), rows). Every pointer, stride and
// row length a multiple of 16 bytes (ops.copyRows checks).
extern "C" __global__ void tf_ds_copy_rows16_kernel(const uint4* src, unsigned long long src_ld, uint4* dst,
                                                    unsigned long long dst_ld, unsigned long long n16) {
    const unsigned long long r = blockIdx.y;
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) dst[r * dst_ld + i] = src[r * src_ld + i];
}

// A greedy row's token as sampling.argmax takes it on the host (best = 0; best = i where row[i] > row[best]): the
// first index of the largest value, a NaN never larger, and 0 when the row's first value is NaN. logits [rows, n] fp32
// (row stride ld), out [rows] u32; a row a block (1024 threads).
extern "C" __global__ void __launch_bounds__(1024) tf_ds_argmax_rows_kernel(const float* logits, unsigned long long ld,
                                                                             int n, unsigned* out) {
    const float* row = logits + (unsigned long long)blockIdx.x * ld;
    const float ninf = -__int_as_float(0x7f800000);
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v == v && (v > bv || (v == bv && i < bi))) {
            bv = v;
            bi = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    const int nw = (blockDim.x + 31) >> 5;
    bv = lane < nw ? sv[lane] : ninf;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) out[blockIdx.x] = row[0] != row[0] ? 0u : (unsigned)bi;
}
