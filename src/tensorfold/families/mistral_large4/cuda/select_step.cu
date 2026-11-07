// A decode step's selection in two launches, one block a row (tp_cuda.FusedTp._select; the sum is cp4.score_keys, kth_key and
// filter_selection): local_top keeps a rank's best k of its visible scores, in row order, as the int64 keys the ranks gather;
// pick_rows finds the k-th best key of all the ranks and keeps this rank's share. Both find a threshold by digits of the key
// (a histogram in shared memory, wide digits: three passes over 32 bits where torch.topk takes sixteen), then write the kept
// rows in row order with counts a warp, so no sort runs. Keys are unique: a score's ordered bits over 0xFFFFFFFF - position.
// A long prompt's rows go through top8 instead: the score kernel leaves the largest of every eight scores beside them, the k-th
// of those is a bound with k scores at or over it, and only the eights that reach it are read again. Same keys as local_top.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int T = 1024;                                           // threads a block
constexpr int WARPS = T / 32;
constexpr int BINS = 2048;                                        // the widest digit: 11 bits
constexpr int GROUP = 8;                                          // scores a largest stands for
constexpr int WIDE = 13, WIDE_BINS = 1 << WIDE;                   // the first digit of the largests
constexpr int CAP = 11264;                                        // scores over the bound a row holds in shared memory
constexpr int TOP8_SMEM = BINS * 4 + CAP * 8;
constexpr long long PAD_KEY = -9223372036854775807LL - 1;
constexpr int NO_ROW = 2147483647;

__device__ inline uint32_t ordered(float v) {
    int32_t b = __float_as_int(v);
    if (b == (int32_t)0x80000000) b = 0;                          // -0 is +0 (score_keys adds 0.0)
    return (uint32_t)(b >= 0 ? b : b ^ 0x7FFFFFFF) ^ 0x80000000u;
}

// The sum of v over this thread and every later one.
__device__ inline int suffix(int v, int* sums) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int d = 1; d < 32; d <<= 1) {
        const int o = __shfl_down_sync(~0u, v, d);
        if (lane + d < 32) v += o;
    }
    if (lane == 0) sums[warp] = v;
    __syncthreads();
    for (int w = warp + 1; w < WARPS; w++) v += sums[w];
    __syncthreads();
    return v;
}

// The bin of the want-th largest (want from 1): ``want`` becomes its rank inside the bin, ``in_bin`` the bin's count.
__device__ inline int pick(const int* hist, int& want, int& in_bin, int* sums, int* out) {
    const int t = threadIdx.x, a = hist[2 * t], b = hist[2 * t + 1];
    const int from = suffix(a + b, sums), above = from - a - b;
    if (above < want && want <= from) {
        const bool hi = above + b >= want;
        out[0] = 2 * t + hi;
        out[1] = want - above - (hi ? 0 : b);
        out[2] = hi ? b : a;
    }
    __syncthreads();
    const int digit = out[0];
    want = out[1];
    in_bin = out[2];
    __syncthreads();
    return digit;
}

// Rows kept before each warp's span: ``kept``, ``tied`` this warp's counts; returns the sums over earlier warps.
__device__ inline int2 before(int kept, int tied, int* a, int* b, int& all_kept) {
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        a[warp] = kept;
        b[warp] = tied;
    }
    __syncthreads();
    int2 r = make_int2(0, 0);
    int total = 0;
    for (int w = 0; w < WARPS; w++) {
        if (w < warp) {
            r.x += a[w];
            r.y += b[w];
        }
        total += a[w];
    }
    all_kept = total;
    __syncthreads();
    return r;
}

__device__ inline long long row_key(uint32_t m, int i, int world, int block, int rank) {
    const uint32_t pos = (uint32_t)((i / block) * (world * block) + rank * block + i % block);
    return (long long)(((uint64_t)(m ^ 0x80000000u) << 32) | (0xFFFFFFFFu - pos));
}

// A row's best k by digits, every pass over all its v scores.
__device__ inline void top_of_row(const float* s, int v, int k, int world, int block, int rank, long long* ko, int* ro, int* hist, int* sums,
                                  int* sums2, int* out) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const bool every = v <= k;
    uint32_t thresh = 0, mask = 0;
    int ties = 0;
    if (!every) {
        int want = k, in_bin = 0, shift = 32;
        const int widths[3] = {11, 11, 10};
#pragma unroll
        for (int p = 0; p < 3; p++) {
            shift -= widths[p];
            const uint32_t digits = (1u << widths[p]) - 1;
            for (int i = t; i < BINS; i += T) hist[i] = 0;
            __syncthreads();
            for (int i = t; i < v; i += T) {
                const uint32_t m = ordered(s[i]);
                if ((m & mask) == thresh) atomicAdd(&hist[(m >> shift) & digits], 1);
            }
            __syncthreads();
            thresh |= (uint32_t)pick(hist, want, in_bin, sums, out) << shift;
            mask |= digits << shift;
        }
        ties = want;
    }
    // a warp's span of rows, a multiple of 32: first its counts, then every row's place
    const int span = ((v + WARPS - 1) / WARPS + 31) / 32 * 32, lo = warp * span, hi = min(v, lo + span);
    int kept = 0, tied = 0;
    for (int b = lo; b < hi; b += 32) {
        const int i = b + lane;
        const uint32_t m = i < hi ? ordered(s[i]) : 0;
        kept += __popc(__ballot_sync(~0u, i < hi && (every || m > thresh)));
        tied += __popc(__ballot_sync(~0u, i < hi && !every && m == thresh));
    }
    int total = 0;
    const int2 sum = before(kept, tied, sums, sums2, total);
    int at = sum.x + min(sum.y, ties), seen = sum.y;
    const uint32_t lower = (1u << lane) - 1;
    for (int b = lo; b < hi; b += 32) {
        const int i = b + lane;
        const uint32_t m = i < hi ? ordered(s[i]) : 0;
        const bool tie = i < hi && !every && m == thresh;
        const uint32_t tie_bits = __ballot_sync(~0u, tie);
        const bool keep = (i < hi && (every || m > thresh)) || (tie && seen + __popc(tie_bits & lower) < ties);
        const uint32_t keep_bits = __ballot_sync(~0u, keep);
        if (keep) {
            const int o = at + __popc(keep_bits & lower);
            ko[o] = row_key(m, i, world, block, rank);
            ro[o] = i;
        }
        at += __popc(keep_bits);
        seen += __popc(tie_bits);
    }
    for (int i = (every ? v : k) + t; i < k; i += T) {
        ko[i] = PAD_KEY;
        ro[i] = NO_ROW;
    }
}

__global__ void __launch_bounds__(T) local_top_kernel(const float* score, int n, const int* vis, int k, int world, int block,
                                                      int rank, long long* keys, int* rows) {
    __shared__ int hist[BINS], sums[WARPS], sums2[WARPS], out[3];
    const int r = blockIdx.x;
    top_of_row(score + (int64_t)r * n, min(vis[r], n), k, world, block, rank, keys + (int64_t)r * k, rows + (int64_t)r * k, hist, sums, sums2,
               out);
}

// Every score of a row at or over the bound from its largests, in row order, into val (ordered bits) and idx: their count, or -1
// when they do not fit. There are at least k: every largest that reaches the bound is such a score.
__device__ inline int over_bound(const float* s, const float* g, int v, int k, uint32_t* val, int* idx, int* sums, int* out) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, ng = (v + GROUP - 1) / GROUP;
    int* wide = reinterpret_cast<int*>(val);                      // the list comes after the bound
    for (int i = t; i < WIDE_BINS; i += T) wide[i] = 0;
    __syncthreads();
    for (int i = t; i < ng; i += T) atomicAdd(&wide[ordered(g[i]) >> (32 - WIDE)], 1);
    __syncthreads();
    constexpr int EACH = WIDE_BINS / T;
    int mine = 0;
#pragma unroll
    for (int b = 0; b < EACH; b++) mine += wide[t * EACH + b];
    int above = suffix(mine, sums) - mine;
    if (above < k && k <= above + mine) {
#pragma unroll
        for (int b = EACH - 1; b >= 0; b--) {
            above += wide[t * EACH + b];
            if (above >= k) {
                out[0] = t * EACH + b;
                break;
            }
        }
    }
    __syncthreads();
    const uint32_t low = (uint32_t)out[0] << (32 - WIDE);
    __syncthreads();
    int c = 0;
    for (int b0 = 0; b0 < ng; b0 += T) {
        const int i = b0 + t;
        uint32_t m[GROUP];
        int n8 = 0;
        if (i < ng && ordered(g[i]) >= low) {
            const float4 x = reinterpret_cast<const float4*>(s)[i * 2], y = reinterpret_cast<const float4*>(s)[i * 2 + 1];
            const float f[GROUP] = {x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
#pragma unroll
            for (int j = 0; j < GROUP; j++) {
                m[j] = i * GROUP + j < v ? ordered(f[j]) : 0u;
                n8 += m[j] >= low;
            }
        }
        int run = n8;
        for (int d = 1; d < 32; d <<= 1) {
            const int o = __shfl_up_sync(~0u, run, d);
            if (lane >= d) run += o;
        }
        if (lane == 31) sums[warp] = run;
        __syncthreads();
        int at = c + run - n8, total = 0;
        for (int w = 0; w < WARPS; w++) {
            if (w < warp) at += sums[w];
            total += sums[w];
        }
        __syncthreads();
        if (c + total > CAP) return -1;
        if (n8) {
#pragma unroll
            for (int j = 0; j < GROUP; j++)
                if (m[j] >= low) {
                    val[at] = m[j];
                    idx[at] = i * GROUP + j;
                    at++;
                }
        }
        c += total;
    }
    __syncthreads();
    return c;
}

// score and largest [R, stride] (a row's scores and largests 32-byte aligned): local_top's keys and rows.
__global__ void __launch_bounds__(T) top8_kernel(const float* score, const float* largest, int64_t ls, int64_t lg, const int* vis, int k,
                                                 int world, int block, int rank, long long* keys, int* rows, int* slow) {
    extern __shared__ int pool[];
    int* hist = pool;
    uint32_t* val = reinterpret_cast<uint32_t*>(pool + BINS);
    int* idx = reinterpret_cast<int*>(val + CAP);
    __shared__ int sums[WARPS], sums2[WARPS], out[3];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, r = blockIdx.x, v = vis[r];
    const float* s = score + r * ls;
    long long* ko = keys + (int64_t)r * k;
    int* ro = rows + (int64_t)r * k;
    const int c = (v + GROUP - 1) / GROUP >= 2 * k ? over_bound(s, largest + r * lg, v, k, val, idx, sums, out) : -1;
    if (c < 0) {
        if (slow && t == 0 && v > k) atomicAdd(slow, 1);
        __syncthreads();
        top_of_row(s, v, k, world, block, rank, ko, ro, hist, sums, sums2, out);
        return;
    }
    int want = k, in_bin = 0, shift = 32;
    uint32_t thresh = 0, mask = 0;
    const int widths[3] = {11, 11, 10};
#pragma unroll
    for (int p = 0; p < 3; p++) {
        shift -= widths[p];
        const uint32_t digits = (1u << widths[p]) - 1;
        for (int i = t; i < BINS; i += T) hist[i] = 0;
        __syncthreads();
        for (int i = t; i < c; i += T) {
            const uint32_t m = val[i];
            if ((m & mask) == thresh) atomicAdd(&hist[(m >> shift) & digits], 1);
        }
        __syncthreads();
        thresh |= (uint32_t)pick(hist, want, in_bin, sums, out) << shift;
        mask |= digits << shift;
    }
    const int ties = want;
    const uint32_t lower = (1u << lane) - 1;
    int at = 0, seen = 0;
    for (int b0 = 0; b0 < c; b0 += T) {
        const int i = b0 + t;
        const uint32_t m = i < c ? val[i] : 0;
        const bool tie = i < c && m == thresh, over = i < c && m > thresh;
        const uint32_t tie_bits = __ballot_sync(~0u, tie);
        if (lane == 0) sums[warp] = __popc(tie_bits);
        __syncthreads();
        int tied = seen, all_tied = 0;
        for (int w = 0; w < WARPS; w++) {
            if (w < warp) tied += sums[w];
            all_tied += sums[w];
        }
        const bool keep = over || (tie && tied + __popc(tie_bits & lower) < ties);
        const uint32_t keep_bits = __ballot_sync(~0u, keep);
        if (lane == 0) sums2[warp] = __popc(keep_bits);
        __syncthreads();
        int o = at, all_kept = 0;
        for (int w = 0; w < WARPS; w++) {
            if (w < warp) o += sums2[w];
            all_kept += sums2[w];
        }
        if (keep) {
            o += __popc(keep_bits & lower);
            ko[o] = row_key(m, idx[i], world, block, rank);
            ro[o] = idx[i];
        }
        at += all_kept;
        seen += all_tied;
        __syncthreads();
    }
}

// A bound for the best k of ``world`` lists of k keys, ``stride`` keys apart, as an unsigned order: exactly they are at or over it.
__device__ inline uint64_t kth_key(const long long* lists, int64_t stride, int k, int world, int* hist, int* sums, int* out) {
    const int t = threadIdx.x;
    uint64_t thresh = 0, mask = 0;
    int want = k, in_bin = 0, shift = 64;
    const int widths[6] = {11, 11, 10, 11, 11, 10};
    for (int p = 0; p < 6 && want != in_bin; p++) {
        shift -= widths[p];
        const uint64_t digits = (1ull << widths[p]) - 1;
        for (int i = t; i < BINS; i += T) hist[i] = 0;
        __syncthreads();
        for (int w = 0; w < world; w++) {
            const long long* ks = lists + w * stride;
            for (int i = t; i < k; i += T) {
                const uint64_t u = (uint64_t)ks[i] ^ (1ull << 63);
                if ((u & mask) == thresh) atomicAdd(&hist[(u >> shift) & digits], 1);
            }
        }
        __syncthreads();
        thresh |= (uint64_t)pick(hist, want, in_bin, sums, out) << shift;
        mask |= digits << shift;
    }
    return thresh;                                               // the bin it stopped in holds exactly what is left of the budget
}

// A row's keys at or over ``thresh``, its rows (plus ``base``) in order, NO_ROW after them; returns how many.
__device__ inline int keep(const long long* mine, const int* ro, uint64_t thresh, int k, int base, int* oo, int* sums, int* sums2) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int span = ((k + WARPS - 1) / WARPS + 31) / 32 * 32, lo = warp * span, hi = min(k, lo + span);
    int kept = 0;
    for (int b = lo; b < hi; b += 32) {
        const int i = b + lane;
        const long long key = i < hi ? mine[i] : PAD_KEY;
        kept += __popc(__ballot_sync(~0u, key != PAD_KEY && ((uint64_t)key ^ (1ull << 63)) >= thresh));
    }
    int total = 0;
    int at = before(kept, 0, sums, sums2, total).x;
    const uint32_t lower = (1u << lane) - 1;
    for (int b = lo; b < hi; b += 32) {
        const int i = b + lane;
        const long long key = i < hi ? mine[i] : PAD_KEY;
        const bool on = key != PAD_KEY && ((uint64_t)key ^ (1ull << 63)) >= thresh;
        const uint32_t bits = __ballot_sync(~0u, on);
        if (on) oo[at + __popc(bits & lower)] = ro[i] + base;
        at += __popc(bits);
    }
    for (int i = total + t; i < k; i += T) oo[i] = NO_ROW;
    return total;
}

__global__ void __launch_bounds__(T) pick_rows_kernel(const long long* all_keys, const int* rows, int R, int k, int world, int rank,
                                                      int* out_rows, int* count) {
    __shared__ int hist[BINS], sums[WARPS], sums2[WARPS], out[3];
    const int r = blockIdx.x;
    const uint64_t thresh = kth_key(all_keys + (int64_t)r * k, (int64_t)R * k, k, world, hist, sums, out);
    const int total = keep(all_keys + ((int64_t)rank * R + r) * k, rows + (int64_t)r * k, thresh, k, 0, out_rows + (int64_t)r * k, sums, sums2);
    if (threadIdx.x == 0) count[r] = total;
}

// A round's two halves of pick_rows: a row's owner finds the threshold from every rank's list, every rank keeps its share.
__global__ void __launch_bounds__(T) kth_rows_kernel(const long long* all_keys, int R, int k, int world, long long* thresh) {
    __shared__ int hist[BINS], sums[WARPS], out[3];
    const int r = blockIdx.x;
    const uint64_t u = kth_key(all_keys + (int64_t)r * k, (int64_t)R * k, k, world, hist, sums, out);
    if (threadIdx.x == 0) thresh[r] = (long long)(u ^ (1ull << 63));
}

__global__ void __launch_bounds__(T) keep_rows_kernel(const long long* keys, const int* rows, const long long* thresh, int k, int base,
                                                      int* out_rows, int* count) {
    __shared__ int sums[WARPS], sums2[WARPS];
    const int r = blockIdx.x;
    const uint64_t u = (uint64_t)thresh[r] ^ (1ull << 63);
    const int total = keep(keys + (int64_t)r * k, rows + (int64_t)r * k, u, k, base, out_rows + (int64_t)r * k, sums, sums2);
    if (threadIdx.x == 0) count[r] = total;
}

void local_top(const at::Tensor& score, const at::Tensor& vis, int64_t k, int64_t world, int64_t block, int64_t rank, at::Tensor keys,
               at::Tensor rows) {
    TORCH_CHECK(score.is_contiguous() && score.scalar_type() == at::kFloat && vis.scalar_type() == at::kInt);
    TORCH_CHECK(keys.is_contiguous() && keys.scalar_type() == at::kLong && rows.is_contiguous() && rows.scalar_type() == at::kInt);
    local_top_kernel<<<(int)score.size(0), T, 0, at::cuda::getCurrentCUDAStream()>>>(
        score.data_ptr<float>(), (int)score.size(1), vis.data_ptr<int>(), (int)k, (int)world, (int)block, (int)rank,
        reinterpret_cast<long long*>(keys.data_ptr<int64_t>()), rows.data_ptr<int>());
}

void top8(const at::Tensor& score, const at::Tensor& largest, const at::Tensor& vis, int64_t k, int64_t world, int64_t block, int64_t rank,
          at::Tensor keys, at::Tensor rows, c10::optional<at::Tensor> slow) {
    TORCH_CHECK(score.scalar_type() == at::kFloat && largest.scalar_type() == at::kFloat && vis.scalar_type() == at::kInt);
    TORCH_CHECK(score.stride(1) == 1 && largest.stride(1) == 1 && score.stride(0) % GROUP == 0 && largest.size(1) * GROUP >= score.size(1));
    TORCH_CHECK(keys.is_contiguous() && keys.scalar_type() == at::kLong && rows.is_contiguous() && rows.scalar_type() == at::kInt);
    static bool ready = false;
    if (!ready) {
        cudaFuncSetAttribute(top8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TOP8_SMEM);
        ready = true;
    }
    if (score.size(0) == 0) return;
    top8_kernel<<<(int)score.size(0), T, TOP8_SMEM, at::cuda::getCurrentCUDAStream()>>>(
        score.data_ptr<float>(), largest.data_ptr<float>(), score.stride(0), largest.stride(0), vis.data_ptr<int>(), (int)k, (int)world,
        (int)block, (int)rank, reinterpret_cast<long long*>(keys.data_ptr<int64_t>()), rows.data_ptr<int>(),
        slow.has_value() ? slow->data_ptr<int>() : nullptr);
}

void pick_rows(const at::Tensor& every, const at::Tensor& rows, int64_t rank, at::Tensor out, at::Tensor count) {
    TORCH_CHECK(every.is_contiguous() && every.scalar_type() == at::kLong && every.dim() == 3);
    TORCH_CHECK(rows.is_contiguous() && out.is_contiguous() && count.scalar_type() == at::kInt);
    pick_rows_kernel<<<(int)every.size(1), T, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const long long*>(every.data_ptr<int64_t>()), rows.data_ptr<int>(), (int)every.size(1), (int)every.size(2),
        (int)every.size(0), (int)rank, out.data_ptr<int>(), count.data_ptr<int>());
}

void kth_rows(const at::Tensor& every, at::Tensor thresh) {
    TORCH_CHECK(every.is_contiguous() && every.scalar_type() == at::kLong && every.dim() == 3);
    if (every.size(1) == 0) return;
    kth_rows_kernel<<<(int)every.size(1), T, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const long long*>(every.data_ptr<int64_t>()), (int)every.size(1), (int)every.size(2), (int)every.size(0),
        reinterpret_cast<long long*>(thresh.data_ptr<int64_t>()));
}

void keep_rows(const at::Tensor& keys, const at::Tensor& rows, const at::Tensor& thresh, int64_t base, at::Tensor out, at::Tensor count) {
    TORCH_CHECK(keys.is_contiguous() && rows.is_contiguous() && out.is_contiguous() && thresh.scalar_type() == at::kLong);
    if (keys.size(0) == 0) return;
    keep_rows_kernel<<<(int)keys.size(0), T, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const long long*>(keys.data_ptr<int64_t>()), rows.data_ptr<int>(),
        reinterpret_cast<const long long*>(thresh.data_ptr<int64_t>()), (int)keys.size(1), (int)base, out.data_ptr<int>(),
        count.data_ptr<int>());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("local_top", &local_top);
    m.def("top8", &top8);
    m.def("pick_rows", &pick_rows);
    m.def("kth_rows", &kth_rows);
    m.def("keep_rows", &keep_rows);
}
