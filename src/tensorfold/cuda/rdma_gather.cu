// One-hop all-gather of small fp32 rank partials over RoCE (TP decode on GB10; tensorfold/cuda/rdma.py).
//
// A staging kernel copies this rank's partial into a slot of pinned host memory and rings a doorbell there. A CPU thread
// splits the slot over the node's RDMA devices (two 200G ports, each behind its own PCIe x4) and RDMA-writes the parts,
// with immediate, into every peer's receive slot for this rank. A peer's thread sees each write's receive completion
// (generated only after the data is placed) and, once every part of a peer's slot is in, publishes the sequence number
// in a flag; the collecting kernel waits for every peer's flag and copies the slots out in rank order. The bytes are
// NCCL's all-gather's; no host node or NCCL proxy is on the path, so a CUDA graph captures the kernel pair as it would
// any other.
//
// A peer that dies must not leave the others spinning: both waits also read an abort word in the same pinned memory
// (every 1024 spins, so a live gather's latency is unchanged). The proxy sets it when an RDMA completion fails (a write
// to a dead peer runs out of retries), and so does ``abort`` (a watchdog that lost a peer); after it every gather on this
// rank, captured ones included, returns at once with garbage and ``failure`` names why.

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int STAGE_THREADS = 512;
constexpr int COLLECT_THREADS = 256;
constexpr int RECV_DEPTH = 256;           // receive WRs kept posted on every QP (a write with immediate consumes one)

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

struct Nic {
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    ibv_mr* mr = nullptr;
    ibv_gid gid{};
    std::vector<ibv_qp*> qps;             // per peer
    std::vector<uint32_t> psns;
};

struct Peer {
    std::vector<uint32_t> qpn, psn, rkey;  // per device
    std::vector<std::string> gid;
    uint64_t recv = 0;
};

struct Gather {
    int rank = 0, world = 1, slots = 4, port = 1, gid_index = 5, cpu = -1;
    size_t max_bytes = 0;
    char* host = nullptr;
    size_t host_bytes = 0;
    char* send_ring = nullptr;            // [slots][max_bytes]
    char* recv_ring = nullptr;            // [slots][world][max_bytes]
    uint64_t* flags = nullptr;            // [slots][world]: the sequence whose data from a peer is in (slot, peer)
    uint64_t* doorbell = nullptr;         // the last sequence staged
    uint64_t* sizes = nullptr;            // [slots]: bytes staged in a slot
    uint64_t* sent = nullptr;             // [slots]: the last sequence whose writes from a slot completed
    uint64_t* abort_word = nullptr;       // nonzero: every wait gives up (a peer is gone)
    uint64_t* seq = nullptr;              // device: gathers so far
    unsigned* stage_done = nullptr;       // device: blocks of a multi-block stage that finished copying (0 between)
    uint64_t* probe = nullptr;            // device: summed ns (stage, doorbell -> flags, copy-out), count, doorbell time
    bool probing = false;
    std::vector<Nic> nics;
    std::vector<Peer> peers;
    std::thread proxy;
    std::atomic<bool> running{false};
    std::atomic<int> failed{0};
    char why[256] = {0};

    ~Gather() {
        running.store(false);
        if (proxy.joinable()) proxy.join();
    }
};

std::vector<std::unique_ptr<Gather>> gathers;

Gather& get(int64_t h) {
    TORCH_CHECK(h >= 0 && h < (int64_t)gathers.size() && gathers[h], "rdma gather: bad handle");
    return *gathers[h];
}

template <typename T>
T* dev(T* host_ptr) {
    T* d = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), host_ptr, 0));
    return d;
}

// One block: wait until the slot's last writes are done, copy the partial in, then publish its size and the doorbell.
__global__ void __launch_bounds__(STAGE_THREADS) stage_kernel(const float4* __restrict__ src, int n4, char* send_ring,
                                                               uint64_t* sizes, uint64_t* doorbell, const uint64_t* sent,
                                                               uint64_t* seq, int slots, uint64_t max_bytes,
                                                               uint64_t* probe, const uint64_t* abort_word,
                                                               uint64_t word = 0) {
    __shared__ uint64_t s_seq;
    const uint64_t t0 = now_ns();
    if (threadIdx.x == 0) {
        const uint64_t q = *seq + 1;
        const int slot = (int)(q % slots);
        if (q > (uint64_t)slots)
            for (uint32_t i = 1; load_sys(sent + slot) + slots < q; ++i)
                if ((i & 1023) == 0 && load_sys(abort_word)) break;
        s_seq = q;
    }
    __syncthreads();
    const uint64_t q = s_seq;
    const int slot = (int)(q % slots);
    float4* dst = reinterpret_cast<float4*>(send_ring + (size_t)slot * max_bytes);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dst[i] = src[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *seq = q;
        store_sys(sizes + slot, word ? word : (uint64_t)n4 * 16);
        store_sys(doorbell, q);
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
}

// stage_kernel's work over several blocks (a concurrent round's partials are 16 rows x 6,144 floats: one block copies
// them in ~10 us): every block waits for the slot and copies its share; the last block to finish (a device counter,
// reset by it) publishes the sequence, the size and the doorbell, as stage_kernel's thread 0 does. No block can read the
// sequence after it moves: the last block finishes after every block has started.
__global__ void __launch_bounds__(STAGE_THREADS) stage_multi_kernel(const float4* __restrict__ src, int n4,
                                                                     char* send_ring, uint64_t* sizes,
                                                                     uint64_t* doorbell, const uint64_t* sent,
                                                                     uint64_t* seq, unsigned* done, int slots,
                                                                     uint64_t max_bytes, uint64_t* probe,
                                                                     const uint64_t* abort_word, uint64_t word = 0) {
    __shared__ uint64_t s_seq;
    __shared__ bool s_last;
    const uint64_t t0 = now_ns();
    if (threadIdx.x == 0) {
        const uint64_t q = *seq + 1;
        const int slot = (int)(q % slots);
        if (q > (uint64_t)slots)
            for (uint32_t i = 1; load_sys(sent + slot) + slots < q; ++i)
                if ((i & 1023) == 0 && load_sys(abort_word)) break;
        s_seq = q;
    }
    __syncthreads();
    const uint64_t q = s_seq;
    const int slot = (int)(q % slots);
    float4* dst = reinterpret_cast<float4*>(send_ring + (size_t)slot * max_bytes);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) dst[i] = src[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) s_last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (s_last && threadIdx.x == 0) {
        __threadfence_system();
        *done = 0u;
        *seq = q;
        store_sys(sizes + slot, word ? word : (uint64_t)n4 * 16);
        store_sys(doorbell, q);
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
}

// residual_add's arithmetic (glm5_next glue._residual_add) for 4 values: the world partials added in rank order in fp32,
// rounded to bf16 (the branch), added to x and rounded to bf16.
template <typename Part>
__device__ __forceinline__ void residual4(Part part, int world, const __nv_bfloat16* __restrict__ x,
                                          __nv_bfloat16* __restrict__ xout, size_t i4) {
    float4 acc = part(0);
    for (int p = 1; p < world; ++p) {
        const float4 v = part(p);
        acc.x = __fadd_rn(acc.x, v.x);
        acc.y = __fadd_rn(acc.y, v.y);
        acc.z = __fadd_rn(acc.z, v.z);
        acc.w = __fadd_rn(acc.w, v.w);
    }
    const uint2 xr = *reinterpret_cast<const uint2*>(x + 4 * i4);
    const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xr.x));
    const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xr.y));
    const float b0 = __bfloat162float(__float2bfloat16_rn(acc.x)), b1 = __bfloat162float(__float2bfloat16_rn(acc.y));
    const float b2 = __bfloat162float(__float2bfloat16_rn(acc.z)), b3 = __bfloat162float(__float2bfloat16_rn(acc.w));
    __nv_bfloat162 o01 = __floats2bfloat162_rn(__fadd_rn(x01.x, b0), __fadd_rn(x01.y, b1));
    __nv_bfloat162 o23 = __floats2bfloat162_rn(__fadd_rn(x23.x, b2), __fadd_rn(x23.y, b3));
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o01);
    o.y = *reinterpret_cast<uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(xout + 4 * i4) = o;
}

// collect_kernel and residual_add in one launch: wait for every peer's flag at this sequence, then x + bf16(the partials
// in rank order) from the receive slots (and this rank's own partial), never writing the gathered [world, R, D] copy.
__global__ void __launch_bounds__(COLLECT_THREADS) collect_residual_kernel(
    const float4* __restrict__ own, int n4, const char* recv_ring, const uint64_t* flags, const uint64_t* seq,
    int rank, int world, int slots, uint64_t max_bytes, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ xout, uint64_t* probe, const uint64_t* abort_word) {
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    if (threadIdx.x < world && threadIdx.x != rank)
        for (uint32_t i = 1; load_sys(flags + (size_t)slot * world + threadIdx.x) < q; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    __syncthreads();
    const uint64_t t2 = now_ns();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        residual4([&](int p) -> float4 {
                      if (p == rank) return own[i];
                      return __ldcv(reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes) + i);
                  },
                  world, x, xout, (size_t)i);
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

// -- the two-hop reduce of a decode window's partials (bf16 branches: 2.7x fewer bytes than every fp32 partial to every
// rank), the same arithmetic as residual_add. Rank k owns rows [R k / world, R (k + 1) / world).
__device__ __forceinline__ int row_cut(int R, int k, int world) { return (int)(((long long)R * k) / world); }

__device__ __forceinline__ void wait_peers(const uint64_t* flags, int slot, int world, int rank, uint64_t q,
                                           const uint64_t* abort_word) {
    if (threadIdx.x < world && threadIdx.x != rank)
        for (uint32_t i = 1; load_sys(flags + (size_t)slot * world + threadIdx.x) < q; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    __syncthreads();
}

// After a scatter stage: every peer's slice of this rank's rows is at the start of its receive slot; out [rows, D] bf16 =
// bf16(the partials added in rank order) for this rank's rows (residual_add's branch).
__global__ void __launch_bounds__(COLLECT_THREADS) reduce_rows_kernel(
    const float4* __restrict__ own, const char* recv_ring, const uint64_t* flags, const uint64_t* seq, int rank,
    int world, int slots, uint64_t max_bytes, int R, int D, __nv_bfloat16* __restrict__ out, const uint64_t* abort_word) {
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    wait_peers(flags, slot, world, rank, q, abort_word);
    const int c0 = row_cut(R, rank, world), rows = row_cut(R, rank + 1, world) - c0, D4 = D / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * D4; i += gridDim.x * blockDim.x) {
        auto part = [&](int p) -> float4 {
            if (p == rank) return own[(size_t)c0 * D4 + i];
            return __ldcv(reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes) + i);
        };
        float4 acc = part(0);
        for (int p = 1; p < world; ++p) {
            const float4 v = part(p);
            acc.x = __fadd_rn(acc.x, v.x);
            acc.y = __fadd_rn(acc.y, v.y);
            acc.z = __fadd_rn(acc.z, v.z);
            acc.w = __fadd_rn(acc.w, v.w);
        }
        __nv_bfloat162 a = __floats2bfloat162_rn(acc.x, acc.y), b = __floats2bfloat162_rn(acc.z, acc.w);
        uint2 o;
        o.x = *reinterpret_cast<uint32_t*>(&a);
        o.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(out + 4 * (size_t)i) = o;
    }
}

// After the branch rows' all-gather: xout = bf16(x + branch) for all R rows, owner k's rows at the start of its receive
// slot (this rank's own from ``mine``).
__global__ void __launch_bounds__(COLLECT_THREADS) collect_rows_residual_kernel(
    const __nv_bfloat16* __restrict__ mine, const char* recv_ring, const uint64_t* flags, const uint64_t* seq, int rank,
    int world, int slots, uint64_t max_bytes, int R, int D, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ xout, const uint64_t* abort_word) {
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    wait_peers(flags, slot, world, rank, q, abort_word);
    const int D4 = D / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < R * D4; i += gridDim.x * blockDim.x) {
        const int r = i / D4, c4 = i - r * D4;
        int k = world - 1;
        while (k > 0 && row_cut(R, k, world) > r) --k;
        const int j = r - row_cut(R, k, world);
        const __nv_bfloat16* src = k == rank ? mine
            : reinterpret_cast<const __nv_bfloat16*>(recv_ring + ((size_t)slot * world + k) * max_bytes);
        uint2 bv;
        if (k == rank) bv = *reinterpret_cast<const uint2*>(src + (size_t)j * D + 4 * c4);
        else bv = __ldcv(reinterpret_cast<const uint2*>(src + (size_t)j * D + 4 * c4));
        const uint2 xr = *reinterpret_cast<const uint2*>(x + 4 * (size_t)i);
        const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xr.x));
        const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xr.y));
        const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bv.x));
        const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bv.y));
        __nv_bfloat162 o01 = __floats2bfloat162_rn(__fadd_rn(x01.x, b01.x), __fadd_rn(x01.y, b01.y));
        __nv_bfloat162 o23 = __floats2bfloat162_rn(__fadd_rn(x23.x, b23.x), __fadd_rn(x23.y, b23.y));
        uint2 o;
        o.x = *reinterpret_cast<uint32_t*>(&o01);
        o.y = *reinterpret_cast<uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xout + 4 * (size_t)i) = o;
    }
}

// residual4 over partials already in device memory ([world, n4] float4): the fused collect's arithmetic, for tests.
__global__ void rank_residual_kernel(const float4* __restrict__ parts, int n4, int world,
                                     const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ xout) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
        residual4([&](int p) -> float4 { return parts[(size_t)p * n4 + i]; }, world, x, xout, (size_t)i);
}

// Wait for every peer's flag at this sequence, then copy the slots (and this rank's own source) out in rank order.
__global__ void __launch_bounds__(COLLECT_THREADS) collect_kernel(const float4* __restrict__ own,
                                                                   float4* __restrict__ dst, int n4,
                                                                   const char* recv_ring, const uint64_t* flags,
                                                                   const uint64_t* seq, int rank, int world, int slots,
                                                                   uint64_t max_bytes, uint64_t* probe,
                                                                   const uint64_t* abort_word) {
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
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) out[i] = own[i];
        } else {
            const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
                out[i] = __ldcv(in + i);
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

void fail(Gather& g, const std::string& what) {
    if (!g.failed.exchange(1)) snprintf(g.why, sizeof(g.why), "%s", what.c_str());
    __atomic_store_n(g.abort_word, (uint64_t)1, __ATOMIC_RELEASE);      // the GPU's waits give up
}

bool post_recvs(ibv_qp* qp, int count) {
    for (int i = 0; i < count; ++i) {
        ibv_recv_wr wr{};
        ibv_recv_wr* bad = nullptr;
        if (ibv_post_recv(qp, &wr, &bad)) return false;
    }
    return true;
}

void proxy_loop(Gather* gp) {
    Gather& g = *gp;
    if (g.cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(g.cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    const int nd = (int)g.nics.size();
    uint64_t posted = 0;
    std::vector<uint64_t> done_seq(g.slots, 0);
    std::vector<int> done(g.slots, 0);
    std::vector<std::vector<uint64_t>> got(g.world, std::vector<uint64_t>(nd, 0));   // per peer: last part per device
    std::vector<uint64_t> published(g.world, 0);
    std::vector<std::pair<uint32_t, int>> owner;                                       // qp_num -> peer
    for (int d = 0; d < nd; ++d)
        for (int p = 0; p < g.world; ++p)
            if (g.nics[d].qps[p]) owner.push_back({g.nics[d].qps[p]->qp_num, p});
    ibv_wc wc[64];
    const volatile uint64_t* doorbell = g.doorbell;
    while (g.running.load(std::memory_order_relaxed)) {
        const uint64_t ring = *doorbell;
        while (posted < ring) {
            const uint64_t q = posted + 1;
            const int slot = (int)(q % g.slots);
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const uint64_t word = ((volatile uint64_t*)g.sizes)[slot];
            // a scatter stage (bit 63; rows in bits 0-23, a row's bytes in bits 24-47): peer p gets the rows it owns,
            // [R p / world, R (p + 1) / world), at the start of its slot for this rank; else every peer gets all n bytes
            const bool scatter = (word >> 63) != 0;
            const uint64_t rows = word & 0xffffffull, rowb = (word >> 24) & 0xffffffull;
            for (int d = 0; d < nd; ++d) {
                for (int p = 0; p < g.world; ++p) {
                    if (p == g.rank) continue;
                    const uint64_t base = scatter ? rows * p / g.world * rowb : 0;
                    const uint64_t n = scatter ? (rows * (p + 1) / g.world) * rowb - base : word;
                    const uint64_t unit = ((n / 16) / nd) * 16;   // whole 16-byte units a device, the last the rest
                    const uint64_t off = unit * d, len = d == nd - 1 ? n - off : unit;
                    const Peer& peer = g.peers[p];
                    ibv_sge sge{(uint64_t)(uintptr_t)(g.send_ring + (size_t)slot * g.max_bytes + base + off),
                                (uint32_t)len, g.nics[d].mr->lkey};
                    ibv_send_wr wr{};
                    wr.wr_id = q;
                    wr.sg_list = &sge;
                    wr.num_sge = len ? 1 : 0;
                    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.imm_data = htonl((uint32_t)(q & 0xffffffffu));
                    wr.wr.rdma.remote_addr = peer.recv + ((uint64_t)slot * g.world + g.rank) * g.max_bytes + off;
                    wr.wr.rdma.rkey = peer.rkey[d];
                    ibv_send_wr* bad = nullptr;
                    if (ibv_post_send(g.nics[d].qps[p], &wr, &bad)) {
                        fail(g, "ibv_post_send failed");
                        return;
                    }
                }
            }
            posted = q;
        }
        for (int d = 0; d < nd; ++d) {
            const int k = ibv_poll_cq(g.nics[d].cq, 64, wc);
            if (k < 0) {
                fail(g, "ibv_poll_cq failed");
                return;
            }
            for (int i = 0; i < k; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fail(g, std::string("RDMA completion: ") + ibv_wc_status_str(wc[i].status));
                    return;
                }
                if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                    int p = -1;
                    for (auto& o : owner)
                        if (o.first == wc[i].qp_num) p = o.second;
                    if (p < 0) {
                        fail(g, "receive completion on an unknown QP");
                        return;
                    }
                    // a device delivers a peer's parts in order: the 32-bit immediate extends its last sequence
                    uint64_t q = (got[p][d] & ~0xffffffffull) | ntohl(wc[i].imm_data);
                    if (q <= got[p][d]) q += 1ull << 32;
                    got[p][d] = q;
                    // a sequence is in once every device has delivered its part (a peer may already be a gather ahead)
                    uint64_t in = got[p][0];
                    for (int x = 1; x < nd; ++x) in = std::min(in, got[p][x]);
                    while (published[p] < in) {
                        const uint64_t r = ++published[p];
                        __atomic_store_n(g.flags + (size_t)(r % g.slots) * g.world + p, r, __ATOMIC_RELEASE);
                    }
                    if (!post_recvs(g.nics[d].qps[p], 1)) {
                        fail(g, "ibv_post_recv failed");
                        return;
                    }
                } else {
                    const uint64_t q = wc[i].wr_id;
                    const int slot = (int)(q % g.slots);
                    if (done_seq[slot] != q) {
                        done_seq[slot] = q;
                        done[slot] = 0;
                    }
                    if (++done[slot] == (g.world - 1) * nd) __atomic_store_n(g.sent + slot, q, __ATOMIC_RELEASE);
                }
            }
        }
    }
}

Nic open_nic(const std::string& name, Gather& g, int world, int rank) {
    int count = 0;
    ibv_device** list = ibv_get_device_list(&count);
    TORCH_CHECK(list && count > 0, "rdma gather: no RDMA devices");
    ibv_device* chosen = nullptr;
    for (int i = 0; i < count; ++i)
        if (name == ibv_get_device_name(list[i])) chosen = list[i];
    TORCH_CHECK(chosen, "rdma gather: no RDMA device named ", name);
    Nic nic;
    nic.ctx = ibv_open_device(chosen);
    ibv_free_device_list(list);
    TORCH_CHECK(nic.ctx, "rdma gather: ibv_open_device ", name, " failed");
    nic.pd = ibv_alloc_pd(nic.ctx);
    TORCH_CHECK(nic.pd, "rdma gather: ibv_alloc_pd failed");
    nic.cq = ibv_create_cq(nic.ctx, 4096, nullptr, nullptr, 0);
    TORCH_CHECK(nic.cq, "rdma gather: ibv_create_cq failed");
    nic.mr = ibv_reg_mr(nic.pd, g.host, g.host_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    TORCH_CHECK(nic.mr, "rdma gather: ibv_reg_mr of ", g.host_bytes, " pinned bytes on ", name, " failed");
    TORCH_CHECK(ibv_query_gid(nic.ctx, g.port, g.gid_index, &nic.gid) == 0, "rdma gather: ibv_query_gid failed");
    std::random_device rd;
    nic.qps.assign(world, nullptr);
    nic.psns.assign(world, 0);
    for (int p = 0; p < world; ++p) {
        if (p == rank) continue;
        ibv_qp_init_attr attr{};
        attr.send_cq = nic.cq;
        attr.recv_cq = nic.cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = 512;
        attr.cap.max_recv_wr = RECV_DEPTH;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.sq_sig_all = 0;
        nic.qps[p] = ibv_create_qp(nic.pd, &attr);
        TORCH_CHECK(nic.qps[p], "rdma gather: ibv_create_qp failed");
        ibv_qp_attr init{};
        init.qp_state = IBV_QPS_INIT;
        init.pkey_index = 0;
        init.port_num = g.port;
        init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        TORCH_CHECK(ibv_modify_qp(nic.qps[p], &init,
                                  IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                    "rdma gather: QP to INIT failed");
        TORCH_CHECK(post_recvs(nic.qps[p], RECV_DEPTH), "rdma gather: ibv_post_recv failed");
        nic.psns[p] = rd() & 0xffffff;
    }
    return nic;
}

int64_t create(int64_t rank, int64_t world, int64_t max_bytes, int64_t slots, const std::vector<std::string>& devices,
               int64_t gid_index, int64_t cpu) {
    TORCH_CHECK(world >= 2 && rank >= 0 && rank < world, "rdma gather: rank ", rank, " of ", world);
    TORCH_CHECK(max_bytes > 0 && max_bytes % 64 == 0, "rdma gather: max_bytes must be a positive multiple of 64");
    TORCH_CHECK(!devices.empty(), "rdma gather: name at least one RDMA device");
    auto g = std::make_unique<Gather>();
    g->rank = (int)rank;
    g->world = (int)world;
    g->slots = (int)slots;
    g->max_bytes = (size_t)max_bytes;
    g->gid_index = (int)gid_index;
    g->cpu = (int)cpu;
    const size_t send_b = (size_t)slots * max_bytes, recv_b = (size_t)slots * world * max_bytes;
    const size_t meta_b = ((size_t)slots * world + 1 + 2 * (size_t)slots + 1) * 8;
    g->host_bytes = send_b + recv_b + ((meta_b + 4095) / 4096) * 4096;
    C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&g->host), g->host_bytes,
                                 cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(g->host, 0, g->host_bytes);
    g->send_ring = g->host;
    g->recv_ring = g->host + send_b;
    g->flags = reinterpret_cast<uint64_t*>(g->host + send_b + recv_b);
    g->doorbell = g->flags + (size_t)slots * world;
    g->sizes = g->doorbell + 1;
    g->sent = g->sizes + slots;
    g->abort_word = g->sent + slots;
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->seq), 8));
    C10_CUDA_CHECK(cudaMemset(g->seq, 0, 8));
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->probe), 8 * 8));
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->stage_done), 4));
    C10_CUDA_CHECK(cudaMemset(g->stage_done, 0, 4));
    C10_CUDA_CHECK(cudaMemset(g->probe, 0, 8 * 8));
    for (const auto& name : devices) g->nics.push_back(open_nic(name, *g, (int)world, (int)rank));
    g->peers.assign(world, Peer{});
    gathers.push_back(std::move(g));
    return (int64_t)gathers.size() - 1;
}

// Per peer: what the other side's QPs toward this rank need: ([qpn], [psn], [gid], receive ring, [rkey]), a device each.
std::vector<py::tuple> local_info(int64_t h) {
    Gather& g = get(h);
    std::vector<py::tuple> out;
    for (int p = 0; p < g.world; ++p) {
        std::vector<uint32_t> qpn, psn, rkey;
        std::vector<py::bytes> gid;
        for (auto& nic : g.nics) {
            qpn.push_back(nic.qps[p] ? nic.qps[p]->qp_num : 0);
            psn.push_back(nic.psns[p]);
            rkey.push_back(nic.mr->rkey);
            gid.push_back(py::bytes(reinterpret_cast<const char*>(nic.gid.raw), 16));
        }
        out.push_back(py::make_tuple(qpn, psn, gid, (uint64_t)(uintptr_t)g.recv_ring, rkey));
    }
    return out;
}

void link_peers(int64_t h, const std::vector<py::tuple>& remote) {
    Gather& g = get(h);
    TORCH_CHECK((int)remote.size() == g.world, "rdma gather: one entry a rank");
    const int nd = (int)g.nics.size();
    for (int p = 0; p < g.world; ++p) {
        if (p == g.rank) continue;
        Peer& peer = g.peers[p];
        peer.qpn = remote[p][0].cast<std::vector<uint32_t>>();
        peer.psn = remote[p][1].cast<std::vector<uint32_t>>();
        peer.gid = remote[p][2].cast<std::vector<std::string>>();
        peer.recv = remote[p][3].cast<uint64_t>();
        peer.rkey = remote[p][4].cast<std::vector<uint32_t>>();
        TORCH_CHECK((int)peer.qpn.size() == nd, "rdma gather: every rank must open the same number of devices");
        for (int d = 0; d < nd; ++d) {
            TORCH_CHECK(peer.gid[d].size() == 16, "rdma gather: a GID is 16 bytes");
            ibv_qp_attr rtr{};
            rtr.qp_state = IBV_QPS_RTR;
            rtr.path_mtu = IBV_MTU_4096;
            rtr.dest_qp_num = peer.qpn[d];
            rtr.rq_psn = peer.psn[d];
            rtr.max_dest_rd_atomic = 1;
            rtr.min_rnr_timer = 12;
            rtr.ah_attr.is_global = 1;
            std::memcpy(rtr.ah_attr.grh.dgid.raw, peer.gid[d].data(), 16);
            rtr.ah_attr.grh.sgid_index = g.gid_index;
            rtr.ah_attr.grh.hop_limit = 64;
            rtr.ah_attr.port_num = g.port;
            TORCH_CHECK(ibv_modify_qp(g.nics[d].qps[p], &rtr,
                                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
                        "rdma gather: QP to RTR failed (peer ", p, ", device ", d, ")");
            ibv_qp_attr rts{};
            rts.qp_state = IBV_QPS_RTS;
            rts.timeout = 14;
            rts.retry_cnt = 7;
            rts.rnr_retry = 7;
            rts.sq_psn = g.nics[d].psns[p];
            rts.max_rd_atomic = 1;
            TORCH_CHECK(ibv_modify_qp(g.nics[d].qps[p], &rts,
                                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0,
                        "rdma gather: QP to RTS failed (peer ", p, ", device ", d, ")");
        }
    }
}

void start(int64_t h) {
    Gather& g = get(h);
    TORCH_CHECK(!g.running.load(), "rdma gather: already started");
    g.running.store(true);
    g.proxy = std::thread(proxy_loop, &g);
}

void stop(int64_t h) {
    Gather& g = get(h);
    if (g.running.exchange(false) && g.proxy.joinable()) g.proxy.join();
}

std::string failure(int64_t h) {
    Gather& g = get(h);
    return g.failed.load() ? std::string(g.why) : std::string();
}

// A watchdog lost a peer: every wait on this rank gives up (the gathers return garbage; ``failure`` says why).
void abort_all(int64_t h, const std::string& why) {
    Gather& g = get(h);
    fail(g, why);
}

void configure(int64_t h, bool probing) { get(h).probing = probing; }

// (stage, doorbell -> every flag, copy-out) mean ns over the probed gathers, and their count
std::vector<double> probes(int64_t h) {
    Gather& g = get(h);
    uint64_t host[8];
    C10_CUDA_CHECK(cudaMemcpy(host, g.probe, sizeof(host), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemset(g.probe, 0, sizeof(host)));
    const double n = host[3] ? (double)host[3] : 1.0;
    return {host[0] / n, host[1] / n, host[2] / n, (double)host[3]};
}

// recv [world * n] <- every rank's send [n] (fp32, contiguous, 16-byte aligned), on the current stream.
void gather(int64_t h, const at::Tensor& send, at::Tensor recv) {
    Gather& g = get(h);
    TORCH_CHECK(send.is_cuda() && recv.is_cuda() && send.scalar_type() == at::kFloat &&
                    recv.scalar_type() == at::kFloat && send.is_contiguous() && recv.is_contiguous(),
                "rdma gather: contiguous fp32 CUDA tensors");
    const int64_t n = send.numel();
    TORCH_CHECK(recv.numel() == n * g.world, "rdma gather: recv must hold world x send");
    TORCH_CHECK(n % 4 == 0 && (uint64_t)n * 4 <= g.max_bytes, "rdma gather: ", n, " floats (a multiple of 4, at most ",
                g.max_bytes / 4, ")");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(send.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(recv.data_ptr()) % 16 == 0, "rdma gather: 16-byte aligned tensors");
    if (g.failed.load()) TORCH_CHECK(false, "rdma gather: ", g.why);
    const int n4 = (int)(n / 4);
    uint64_t* probe = g.probing ? g.probe : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    stage_kernel<<<1, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const float4*>(send.data_ptr()), n4,
                                                  dev(g.send_ring), dev(g.sizes), dev(g.doorbell), dev(g.sent), g.seq,
                                                  g.slots, g.max_bytes, probe, dev(g.abort_word));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int blocks = std::max(1, std::min(16, (n4 * g.world + COLLECT_THREADS * 4 - 1) / (COLLECT_THREADS * 4)));
    collect_kernel<<<blocks, COLLECT_THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(send.data_ptr()), reinterpret_cast<float4*>(recv.data_ptr()), n4,
        dev(g.recv_ring), dev(g.flags), g.seq, g.rank, g.world, g.slots, g.max_bytes, probe, dev(g.abort_word));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The first half of gather: send [n] (fp32) staged and announced (blocks > 1: stage_multi_kernel); collect_residual
// finishes it. Nothing else may use this gather between the two.
void stage(int64_t h, const at::Tensor& send, int64_t blocks) {
    Gather& g = get(h);
    TORCH_CHECK(send.is_cuda() && send.scalar_type() == at::kFloat && send.is_contiguous(),
                "rdma stage: a contiguous fp32 CUDA tensor");
    const int64_t n = send.numel();
    TORCH_CHECK(n % 4 == 0 && (uint64_t)n * 4 <= g.max_bytes, "rdma stage: ", n, " floats (a multiple of 4, at most ",
                g.max_bytes / 4, ")");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(send.data_ptr()) % 16 == 0, "rdma stage: a 16-byte aligned tensor");
    if (g.failed.load()) TORCH_CHECK(false, "rdma gather: ", g.why);
    const int n4 = (int)(n / 4);
    uint64_t* probe = g.probing ? g.probe : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int nb = (int)std::max<int64_t>(1, std::min<int64_t>(blocks, (n4 + STAGE_THREADS - 1) / STAGE_THREADS));
    if (nb == 1)
        stage_kernel<<<1, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const float4*>(send.data_ptr()), n4,
                                                      dev(g.send_ring), dev(g.sizes), dev(g.doorbell), dev(g.sent),
                                                      g.seq, g.slots, g.max_bytes, probe, dev(g.abort_word));
    else
        stage_multi_kernel<<<nb, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const float4*>(send.data_ptr()), n4,
                                                             dev(g.send_ring), dev(g.sizes), dev(g.doorbell),
                                                             dev(g.sent), g.seq, g.stage_done, g.slots, g.max_bytes,
                                                             probe, dev(g.abort_word));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The second half: xout [R, D] bf16 = x + bf16(every rank's partial in rank order) (residual_add's arithmetic), from the
// receive slots of the gather the last stage started; own = this rank's partial (the tensor staged).
void collect_residual(int64_t h, const at::Tensor& own, const at::Tensor& x, at::Tensor xout) {
    Gather& g = get(h);
    TORCH_CHECK(own.is_cuda() && own.scalar_type() == at::kFloat && own.is_contiguous(), "rdma collect: fp32 own");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && xout.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
                    xout.is_contiguous() && x.numel() == own.numel() && xout.numel() == own.numel(),
                "rdma collect: x and xout bf16, contiguous, as many values as the partial");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(xout.data_ptr()) % 8 == 0,
                "rdma collect: 8-byte aligned x and xout");
    const int n4 = (int)(own.numel() / 4);
    uint64_t* probe = g.probing ? g.probe : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = std::max(1, std::min(32, (n4 + COLLECT_THREADS - 1) / COLLECT_THREADS));
    collect_residual_kernel<<<blocks, COLLECT_THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(own.data_ptr()), n4, dev(g.recv_ring), dev(g.flags), g.seq, g.rank, g.world,
        g.slots, g.max_bytes, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(xout.data_ptr()), probe, dev(g.abort_word));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The two-hop reduce of a decode window's partial send [R, D] fp32 into xout = bf16(x + bf16(the partials in rank order)):
// a scatter stage (each peer gets its rows), reduce_rows (this rank's rows' bf16 branch into ``mine`` [ceil(R / world),
// D]), a plain stage of ``mine`` (as fp32 words), and collect_rows_residual. ``part1``: only the first launch (the rest
// with reduce_finish), so the partial goes out as soon as it exists.
void reduce_stage(int64_t h, const at::Tensor& send, int64_t R, int64_t D, int64_t blocks) {
    Gather& g = get(h);
    TORCH_CHECK(send.is_cuda() && send.scalar_type() == at::kFloat && send.is_contiguous() && send.numel() == R * D,
                "rdma reduce: a contiguous fp32 [R, D] partial");
    TORCH_CHECK(D % 4 == 0 && R >= 1 && R < (1 << 24) && (uint64_t)R * D * 4 <= g.max_bytes && D * 4 < (1 << 24),
                "rdma reduce: shape");
    if (g.failed.load()) TORCH_CHECK(false, "rdma gather: ", g.why);
    const int n4 = (int)(R * D / 4);
    const uint64_t word = (1ull << 63) | ((uint64_t)(D * 4) << 24) | (uint64_t)R;
    uint64_t* probe = g.probing ? g.probe : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int nb = (int)std::max<int64_t>(1, std::min<int64_t>(blocks, (n4 + STAGE_THREADS - 1) / STAGE_THREADS));
    if (nb == 1)
        stage_kernel<<<1, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const float4*>(send.data_ptr()), n4,
                                                      dev(g.send_ring), dev(g.sizes), dev(g.doorbell), dev(g.sent),
                                                      g.seq, g.slots, g.max_bytes, probe, dev(g.abort_word), word);
    else
        stage_multi_kernel<<<nb, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const float4*>(send.data_ptr()), n4,
                                                             dev(g.send_ring), dev(g.sizes), dev(g.doorbell),
                                                             dev(g.sent), g.seq, g.stage_done, g.slots, g.max_bytes,
                                                             probe, dev(g.abort_word), word);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_finish(int64_t h, const at::Tensor& send, int64_t R, int64_t D, at::Tensor mine, const at::Tensor& x,
                   at::Tensor xout, int64_t blocks) {
    Gather& g = get(h);
    const int maxr = (int)((R + g.world - 1) / g.world);
    TORCH_CHECK(mine.scalar_type() == at::kBFloat16 && mine.is_contiguous() && mine.numel() >= (int64_t)maxr * D,
                "rdma reduce: mine bf16 [ceil(R / world), D]");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && xout.scalar_type() == at::kBFloat16 && x.is_contiguous() &&
                    xout.is_contiguous() && x.numel() == R * D && xout.numel() == R * D, "rdma reduce: x / xout");
    TORCH_CHECK((int64_t)maxr * D * 2 <= (int64_t)g.max_bytes && ((int64_t)maxr * D * 2) % 16 == 0, "rdma reduce: rows");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int D4 = (int)(D / 4);
    reduce_rows_kernel<<<std::max(1, std::min(16, (maxr * D4 + COLLECT_THREADS - 1) / COLLECT_THREADS)),
                         COLLECT_THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(send.data_ptr()), dev(g.recv_ring), dev(g.flags), g.seq, g.rank, g.world,
        g.slots, g.max_bytes, (int)R, (int)D, reinterpret_cast<__nv_bfloat16*>(mine.data_ptr()), dev(g.abort_word));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // the branch rows (padded to maxr rows) all-gathered as fp32 words
    const int n4 = (int)((int64_t)maxr * D * 2 / 16);
    uint64_t* probe = g.probing ? g.probe : nullptr;
    const int nb = (int)std::max<int64_t>(1, std::min<int64_t>(blocks, (n4 + STAGE_THREADS - 1) / STAGE_THREADS));
    if (nb == 1)
        stage_kernel<<<1, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const float4*>(mine.data_ptr()), n4,
                                                      dev(g.send_ring), dev(g.sizes), dev(g.doorbell), dev(g.sent),
                                                      g.seq, g.slots, g.max_bytes, probe, dev(g.abort_word));
    else
        stage_multi_kernel<<<nb, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const float4*>(mine.data_ptr()), n4,
                                                             dev(g.send_ring), dev(g.sizes), dev(g.doorbell),
                                                             dev(g.sent), g.seq, g.stage_done, g.slots, g.max_bytes,
                                                             probe, dev(g.abort_word));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    collect_rows_residual_kernel<<<std::max(1, std::min(32, (int)((R * D4 + COLLECT_THREADS - 1) / COLLECT_THREADS))),
                                   COLLECT_THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(mine.data_ptr()), dev(g.recv_ring), dev(g.flags), g.seq, g.rank,
        g.world, g.slots, g.max_bytes, (int)R, (int)D, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(xout.data_ptr()), dev(g.abort_word));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// collect_residual's arithmetic over partials in device memory ([world, ...] fp32): for tests against residual_add.
void rank_residual(const at::Tensor& parts, const at::Tensor& x, at::Tensor xout) {
    TORCH_CHECK(parts.is_cuda() && parts.scalar_type() == at::kFloat && parts.is_contiguous() && parts.dim() >= 2,
                "rank_residual: fp32 [world, ...] partials");
    const int world = (int)parts.size(0);
    const int n4 = (int)(parts.numel() / world / 4);
    TORCH_CHECK(x.numel() == n4 * 4 && xout.numel() == n4 * 4, "rank_residual: x / xout sizes");
    auto stream = at::cuda::getCurrentCUDAStream();
    rank_residual_kernel<<<std::max(1, std::min(64, (n4 + 255) / 256)), 256, 0, stream>>>(
        reinterpret_cast<const float4*>(parts.data_ptr()), n4, world,
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(xout.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stage", &stage);
    m.def("collect_residual", &collect_residual);
    m.def("rank_residual", &rank_residual);
    m.def("reduce_stage", &reduce_stage);
    m.def("reduce_finish", &reduce_finish);
    m.def("create", &create);
    m.def("local_info", &local_info);
    m.def("connect", &link_peers);
    m.def("start", &start);
    m.def("stop", &stop);
    m.def("failure", &failure);
    m.def("abort", &abort_all);
    m.def("configure", &configure);
    m.def("probes", &probes);
    m.def("gather", &gather);
}
