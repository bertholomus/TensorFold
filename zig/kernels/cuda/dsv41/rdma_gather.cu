// The device half of DeepSeek-V4.1's one-hop RDMA all-gather (the Python family's cuda/rdma_gather.cu v7). The staging
// kernel copies this rank's fp32 slice into a pinned host ring slot and rings a doorbell; the host's proxy thread
// (rdma.zig) RDMA-writes the slot to every peer with an immediate and publishes a peer's flag once all its parts
// are in; the collecting kernels wait for the flags and copy the slots out in rank order, or straight into the caller's
// views (gather_into). The kernel bodies are the served build's; only their names are the family's.
// v7: a nonzero `pdl` makes each kernel wait for the one before it and let the next one launch (programmatic dependent
// launches, sm_90+), so the pair does not break a chain of early launches.

#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr int STAGE_THREADS = 512;
constexpr int COLLECT_THREADS = 256;

__device__ __forceinline__ uint64_t load_sys(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void store_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void store_relaxed_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}


// v7 (TF_RDMA_PDL): wait for the kernel before this one (done, its memory visible), then let the next one launch
__device__ __forceinline__ void pdl_enter(int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}


}  // namespace

// Each block: wait until the slot's last writes are done, copy its share of the partial in; the last block to finish
// (a counter) publishes the size and the doorbell. (One block took ~6 us for a 6-row window's 120 KB and ~35 us for a
// 6-row head's 1.5 MB into the pinned ring on GB10; 8-16 blocks 2.5 and 9.) Every block reads the sequence before it
// counts itself in, and only the last one, after every block has, advances it.
extern "C" __global__ void __launch_bounds__(STAGE_THREADS) dsv41_rdma_stage(const float4* __restrict__ src, int n4,
        char* send_ring, uint64_t* sizes, uint64_t* doorbell, const uint64_t* sent, uint64_t* seq, int slots,
        uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, unsigned* staged, int fence_mode,
        float4* __restrict__ own_dst, int own_row4, long long own_stride4, int pdl) {
    __shared__ uint64_t s_seq;
    __shared__ int s_last;
    pdl_enter(pdl);                              // before anything reads the partial or this gather's sequence
    const uint64_t t0 = now_ns();
    if (threadIdx.x == 0) {
        const uint64_t q = *seq + 1;
        const int slot = (int)(q % slots);
        if (q > (uint64_t)slots)
            for (uint32_t i = 1; load_sys(sent + slot) + slots < q; ++i)
                if ((i & 1023) == 0 && load_sys(abort_word)) break;
        s_seq = q;
        // fence_mode 1: the size is written before this block's release below, so the publish needs no second fence
        if (fence_mode && blockIdx.x == 0) store_relaxed_sys(sizes + slot, (uint64_t)n4 * 16);
    }
    __syncthreads();
    const uint64_t q = s_seq;
    const int slot = (int)(q % slots);
    float4* dst = reinterpret_cast<float4*>(send_ring + (size_t)slot * max_bytes);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) dst[i] = src[i];
    if (fence_mode) {
        // the barrier orders every thread's stores before thread 0's system fence (a release for the whole block, as
        // NCCL publishes its proxy FIFO), then the counter
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
        }
    } else {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_last) {
        __threadfence_system();
        *staged = 0;
        *seq = q;
        if (fence_mode) {
            // the fence above acquired every block's release (data and size) and orders them before this strong store
            store_relaxed_sys(doorbell, q);
        } else {
            store_sys(sizes + slot, (uint64_t)n4 * 16);
            store_sys(doorbell, q);
        }
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
    if (own_dst) {
        // TF_RDMA_OWN_EARLY: this rank's own slice of the output, copied after the doorbell (during the RDMA writes and
        // the peer's) instead of by the collecting kernel after the wait; the same bytes from the same source
        if (s_last) __syncthreads();             // the publishing block lets its doorbell out first (s_last is uniform)
        if (own_stride4 == own_row4) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) own_dst[i] = src[i];
        } else {                                 // gather_into: rows of own_row4 float4, own_stride4 apart
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = src[i];
            }
        }
    }
}


// Wait for every peer's flag at this sequence, then copy the slots (and this rank's own source) out in rank order.
extern "C" __global__ void __launch_bounds__(COLLECT_THREADS) dsv41_rdma_collect(const float4* __restrict__ own,
        float4* __restrict__ dst, int n4, const char* recv_ring, const uint64_t* flags, const uint64_t* seq, int rank,
        int world, int slots, uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, int skip_own, int unroll,
        int pdl) {
    pdl_enter(pdl);                              // the staging kernel's sequence and own slice are written
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    if (threadIdx.x < world && threadIdx.x != rank)
        for (uint32_t i = 1; load_sys(flags + (size_t)slot * world + threadIdx.x) < q; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    __syncthreads();
    const uint64_t t2 = now_ns();
    for (int p = 0; p < world; ++p) {
        float4* out = dst + (size_t)p * n4;
        if (p == rank) {
            if (skip_own) continue;              // the staging kernel wrote it
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) out[i] = own[i];
        } else {
            const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
            if (unroll) {
                // TF_RDMA_COLLECT_UNROLL: up to four loads of the ring in flight a thread before its stores (the ring's
                // first reads after the flag wait out a DRAM queue when a prefetch runs beside)
                const int stride = gridDim.x * blockDim.x;
                for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n4; b += 4 * stride) {
                    float4 v[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (b + u * stride < n4) v[u] = __ldcv(in + b + u * stride);
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (b + u * stride < n4) out[b + u * stride] = v[u];
                }
            } else {
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
                    out[i] = __ldcv(in + i);
            }
        }
    }
    if (probe && blockIdx.x == 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            probe[1] += t2 - probe[4];
            probe[2] += now_ns() - t2;
            probe[3] += 1;
        }
    }
}


constexpr int MAXW = 8;

// gather_into's destinations: rank p's slice is rows x row4[p] float4 (packed in its ring slot), copied to rows
// stride4[p] float4 apart from ptr[p]; ptr[p] null: not copied out.
struct Dsts {
    float4* ptr[MAXW];
    long long stride4[MAXW];
    int row4[MAXW];
};

// Wait for every peer's flag at this sequence, then copy each peer's slot to its destination rows (this rank's own
// slice was written by the staging kernel).
extern "C" __global__ void __launch_bounds__(COLLECT_THREADS) dsv41_rdma_collect_into(Dsts d, int rows,
        const char* recv_ring, const uint64_t* flags, const uint64_t* seq, int rank, int world, int slots,
        uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, int pdl) {
    pdl_enter(pdl);                              // the staging kernel's sequence and own slice are written
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    if (threadIdx.x < world && threadIdx.x != rank)
        for (uint32_t i = 1; load_sys(flags + (size_t)slot * world + threadIdx.x) < q; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    __syncthreads();
    const uint64_t t2 = now_ns();
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < world; ++p) {
        if (p == rank || d.ptr[p] == nullptr) continue;
        const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
        float4* out = d.ptr[p];
        const int r4 = d.row4[p], n4 = rows * r4;
        const long long s4 = d.stride4[p];
        // up to four ring loads in flight a thread before its stores (as dsv41_rdma_collect's unrolled loop)
        for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n4; b += 4 * stride) {
            float4 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u)
                if (b + u * stride < n4) v[u] = __ldcv(in + b + u * stride);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = b + u * stride;
                if (i < n4) {
                    const int r = i / r4;
                    out[(size_t)r * s4 + (i - r * r4)] = v[u];
                }
            }
        }
    }
    if (probe && blockIdx.x == 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            probe[1] += t2 - probe[4];
            probe[2] += now_ns() - t2;
            probe[3] += 1;
        }
    }
}

