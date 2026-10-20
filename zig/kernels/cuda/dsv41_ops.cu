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
// The served paced prefetch (exl3/l2_prefetch.cu, its decode settings): after delay_ns (a gather's staging kernel
// launched beside it first runs undisturbed), one block issues a wave of blockDim 8 KiB bulk L2 prefetches every
// wave_ns, the ranges in order, so DRAM's queue holds about a wave and the latency-bound kernels running beside keep
// most of their load latency. Each range starts on a chunk of its own; its bytes are cut to whole 16-byte units.
// wave_ns 0: every chunk at once from the grid's blocks. Writes nothing.
extern "C" __global__ void __launch_bounds__(128) tf_ds_l2_prefetch_kernel(TfDsPrefetch a, long long delay_ns, long long wave_ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const long long chunk = 8 << 10;
    if (delay_ns > 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        for (;;) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            if (t - t0 >= (unsigned long long)delay_ns) break;
            __nanosleep(500);
        }
    }
    long long end[16];
    long long total = 0;
    for (int t = 0; t < a.n; ++t) {
        total += ((long long)(a.bytes[t] / 16 * 16) + chunk - 1) / chunk;
        end[t] = total;
    }
    const long long stride = (long long)gridDim.x * blockDim.x;
    int r = 0;
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    long long wave = 0;
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c - threadIdx.x < total; c += stride) {
        if (wave_ns > 0) {
            __syncthreads();
            if (threadIdx.x == 0) {
                for (;;) {
                    unsigned long long t;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                    if (t - t0 >= (unsigned long long)(wave * wave_ns)) break;
                    __nanosleep(256);
                }
            }
            __syncthreads();
            ++wave;
        }
        if (c >= total) continue;
        while (c >= end[r]) ++r;
        const long long off = (c - (r ? end[r - 1] : 0)) * chunk;
        const long long left = (long long)(a.bytes[r] / 16 * 16) - off;
        const unsigned size = (unsigned)(left < chunk ? left : chunk);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a.ptr[r] + off), "r"(size) : "memory");
    }
#else
    const unsigned long long step = (unsigned long long)blockDim.x * gridDim.x * 128ull;
    for (int t = 0; t < a.n; ++t) {
        for (unsigned long long off = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) * 128ull; off < a.bytes[t]; off += step)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a.ptr[t] + off));
    }
#endif
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

// ---- the vision tower (vit.zig): DeepSeek's ViT and aligner as the served lane runs them on torch, each torch op its
// own rounding point (this file builds with --fmad=false, so a product and a sum stay two roundings).

// get_vision_cos_sin on the GPU: a patch at grid row h, column w (row-major over n_w columns) gets
// freqs[c] = float(c < 16 ? h : w) * inv_freq[c % 16], inv_freq[j] = 1 / powf(theta, float(2j) / 32), then cosf, sinf;
// out [n, 32] fp32 each.
extern "C" __global__ void tf_ds_vrope_kernel(float* cos_out, float* sin_out, int n, int nw, float theta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 32) return;
    const int r = i / 32, c = i % 32, j = c % 16;
    const float t = float(2 * j) / 32.0f;
    const float inv = 1.0f / powf(theta, t);
    const float pos = float(c < 16 ? r / nw : r % nw);
    const float f = pos * inv;
    cos_out[i] = cosf(f);
    sin_out[i] = sinf(f);
}

// The attention's inputs from wqkv's output qkv bf16 [n, 3 * 16 * 64]: q and k rotated (apply_rotary: x.float(), halves
// x1 = x[:32], x2 = x[32:], cat(x1 * cos - x2 * sin, x2 * cos + x1 * sin).to(bf16)) and every value then .float() for the
// fp32 attention: q32, k32, v32 fp32 [n, 16, 64]. One thread a (row, head, d < 32) pair.
extern "C" __global__ void tf_ds_vrot_kernel(const __nv_bfloat16* qkv, const float* cosb, const float* sinb, float* q32,
                                             float* k32, float* v32, int n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * 16 * 32) return;
    const long long r = i / 512;
    const int rem = int(i % 512), h = rem / 32, d = rem % 32;
    const float c = cosb[r * 32 + d], s = sinb[r * 32 + d];
    for (int which = 0; which < 2; ++which) {
        const __nv_bfloat16* x = qkv + r * 3072 + which * 1024 + h * 64;
        float* out = (which == 0 ? q32 : k32) + r * 1024 + h * 64;
        const float x1 = __bfloat162float(x[d]), x2 = __bfloat162float(x[d + 32]);
        const float o1 = x1 * c - x2 * s;
        const float o2 = x2 * c + x1 * s;
        out[d] = __bfloat162float(__float2bfloat16_rn(o1));
        out[d + 32] = __bfloat162float(__float2bfloat16_rn(o2));
    }
    const __nv_bfloat16* v = qkv + r * 3072 + 2048 + h * 64;
    v32[r * 1024 + h * 64 + d] = __bfloat162float(v[d]);
    v32[r * 1024 + h * 64 + d + 32] = __bfloat162float(v[d + 32]);
}

// `.to(bf16)` of count fp32 values (round to nearest even).
extern "C" __global__ void tf_ds_f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, unsigned long long count) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x)
        out[i] = __float2bfloat16_rn(in[i]);
}

// x + y of bf16 values in place on x (torch's add: fp32 sum, rounded once).
extern "C" __global__ void tf_ds_add_bf16_kernel(__nv_bfloat16* x, const __nv_bfloat16* y, unsigned long long count) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x)
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
}

// The MLP's F.silu(gate) * up of w1's output gu bf16 [n, 2 * inter]: silu in fp32 (x / (1 + expf(-x))) rounded to bf16,
// then the bf16 product rounded again; out bf16 [n, inter].
extern "C" __global__ void tf_ds_vsilu_mul_kernel(const __nv_bfloat16* gu, __nv_bfloat16* out, int n, int inter) {
    const long long total = (long long)n * inter;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / inter, c = i % inter;
        const float g = __bfloat162float(gu[r * 2 * inter + c]);
        const __nv_bfloat16 sg = __float2bfloat16_rn(g / (1.0f + expf(-g)));
        const float u = __bfloat162float(gu[r * 2 * inter + inter + c]);
        out[i] = __float2bfloat16_rn(__bfloat162float(sg) * u);
    }
}

// The aligner's unfold of the tower's output x bf16 [n_h * n_w, dim] (row-major patch grid) padded with zeros to whole
// 3x3 cells: out bf16 [dim * 9, cells] (F.unfold's [C * k * k, L] layout), feature f = c * 9 + ki * 3 + kj of cell
// l = lh * cells_w + lw taking x at grid (3 * lh + ki, 3 * lw + kj), 0 outside the grid.
extern "C" __global__ void tf_ds_vunfold3_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, int n_h, int n_w, int dim,
                                                 int cells_h, int cells_w) {
    const long long cells = (long long)cells_h * cells_w;
    const long long total = (long long)dim * 9 * cells;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long f = i / cells, l = i % cells;
        const int c = int(f / 9), k = int(f % 9), ki = k / 3, kj = k % 3;
        const int y = int(l / cells_w) * 3 + ki, xx = int(l % cells_w) * 3 + kj;
        out[i] = (y < n_h && xx < n_w) ? x[((long long)y * n_w + xx) * dim + c] : __float2bfloat16_rn(0.0f);
    }
}

// F.gelu (erf) of bf16 values in place: x * 0.5 * (1 + erf(x * M_SQRT1_2)) in fp32, rounded to bf16.
extern "C" __global__ void tf_ds_vgelu_kernel(__nv_bfloat16* x, unsigned long long count) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const float v = __bfloat162float(x[i]);
        x[i] = __float2bfloat16_rn(v * 0.5f * (1.0f + erff(v * float(M_SQRT1_2))));
    }
}

// An image span's rows bf16 [tokens, dim]: IMAGE_START / IMAGE_NEW_LINE / IMAGE_END take their learned embedding,
// IMAGE the aligner's rows in reading order. types: 0 start, 1 image, 2 newline, 3 end.
extern "C" __global__ void tf_ds_vspan_kernel(const unsigned char* types, const __nv_bfloat16* rows, const __nv_bfloat16* start,
                                              const __nv_bfloat16* newline, const __nv_bfloat16* end, __nv_bfloat16* out,
                                              int tokens, int dim, const int* row_of) {
    const long long total = (long long)tokens * dim;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int t = int(i / dim), c = int(i % dim);
        const unsigned char ty = types[t];
        out[i] = ty == 1 ? rows[(long long)row_of[t] * dim + c] : ty == 0 ? start[c] : ty == 2 ? newline[c] : end[c];
    }
}

#include "dsv41_candidates.cuh"
