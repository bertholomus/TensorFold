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
//
// v5 (each part behind a switch; 0 restores c100's path, the bytes moved are the same in every mode):
// - TF_RDMA_HOST_REG (default 1): the rings and the metadata (flags, doorbell, sizes, sent, abort word) live in malloc'd
//   memory registered with cudaHostRegister (and mlock'd) instead of one cudaHostAlloc block. On GB10 the GPU reaches
//   cudaHostAlloc memory over an uncached path whose stores, loads and polls queue behind DRAM traffic (the L2 prefetch
//   running beside a gather), while registered malloc memory takes the coherent, L2-cached path of cudaMalloc memory (4 MB:
//   loads 6.3 us against 20.4, stores 6.2 against 20.7). The NIC registers both alike; GPUDirect RDMA and dma-buf
//   export of cudaMalloc memory are not available on GB10. TF_RDMA_META_REG=0 keeps only the metadata in cudaHostAlloc.
// - TF_RDMA_STAGE_FENCE (default 1): one system fence a staging block after a barrier (as NCCL publishes its proxy
//   FIFO) instead of one a thread, the size written before that release, and one fence before the doorbell, not three.
// - TF_RDMA_OWN_EARLY (default 1): the staging kernel copies this rank's own slice to the output after the doorbell,
//   while the writes are on the wire, so the collecting kernel copies only the peers' slices after its wait.
// - TF_RDMA_COLLECT_UNROLL (default 1): the collecting kernel keeps up to four ring loads in flight a thread.

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdlib>
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

__device__ __forceinline__ void store_relaxed_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

int env_int(const char* name, int dflt) {
    const char* e = std::getenv(name);
    return e && *e ? std::atoi(e) : dflt;
}

struct Nic {
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    ibv_mr* mr = nullptr;                 // the send ring's (the whole block's when the rings are cudaHostAlloc memory)
    ibv_mr* mr_recv = nullptr;            // the receive ring's (== mr in that case)
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
    char* host = nullptr;                 // the metadata's block (host_reg 0: rings + metadata in one cudaHostAlloc)
    size_t host_bytes = 0;
    int host_reg = 0;                     // TF_RDMA_HOST_REG at create (1: registered malloc memory)
    size_t send_bytes = 0, recv_bytes = 0;
    int fence_mode = 0;                   // TF_RDMA_STAGE_FENCE: 1 one fence a block and one publish fence, 0 as c100
    char* send_ring = nullptr;            // [slots][max_bytes]
    char* recv_ring = nullptr;            // [slots][world][max_bytes]
    uint64_t* flags = nullptr;            // [slots][world]: the sequence whose data from a peer is in (slot, peer)
    uint64_t* doorbell = nullptr;         // the last sequence staged
    uint64_t* sizes = nullptr;            // [slots]: bytes staged in a slot
    uint64_t* sent = nullptr;             // [slots]: the last sequence whose writes from a slot completed
    uint64_t* abort_word = nullptr;       // nonzero: every wait gives up (a peer is gone)
    uint64_t* seq = nullptr;              // device: gathers so far
    unsigned* staged = nullptr;           // device: blocks of the current staging launch done copying (reset by the last)
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

// Each block: wait until the slot's last writes are done, copy its share of the partial in; the last block to finish
// (a counter) publishes the size and the doorbell. (One block took ~6 us for a 6-row window's 120 KB and ~35 us for a
// 6-row head's 1.5 MB into the pinned ring on GB10; 8-16 blocks 2.5 and 9.) Every block reads the sequence before it
// counts itself in, and only the last one, after every block has, advances it.
__global__ void __launch_bounds__(STAGE_THREADS) stage_kernel(const float4* __restrict__ src, int n4, char* send_ring,
                                                               uint64_t* sizes, uint64_t* doorbell, const uint64_t* sent,
                                                               uint64_t* seq, int slots, uint64_t max_bytes,
                                                               uint64_t* probe, const uint64_t* abort_word,
                                                               unsigned* staged, int fence_mode,
                                                               float4* __restrict__ own_dst) {
    __shared__ uint64_t s_seq;
    __shared__ int s_last;
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
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) own_dst[i] = src[i];
    }
}

// Wait for every peer's flag at this sequence, then copy the slots (and this rank's own source) out in rank order.
__global__ void __launch_bounds__(COLLECT_THREADS) collect_kernel(const float4* __restrict__ own,
                                                                   float4* __restrict__ dst, int n4,
                                                                   const char* recv_ring, const uint64_t* flags,
                                                                   const uint64_t* seq, int rank, int world, int slots,
                                                                   uint64_t max_bytes, uint64_t* probe,
                                                                   const uint64_t* abort_word, int skip_own,
                                                                   int unroll) {
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
            const uint64_t n = ((volatile uint64_t*)g.sizes)[slot];
            const uint64_t unit = ((n / 16) / nd) * 16;   // whole 16-byte units a device, the last takes the rest
            for (int d = 0; d < nd; ++d) {
                const uint64_t off = unit * d, len = d == nd - 1 ? n - off : unit;
                for (int p = 0; p < g.world; ++p) {
                    if (p == g.rank) continue;
                    const Peer& peer = g.peers[p];
                    ibv_sge sge{(uint64_t)(uintptr_t)(g.send_ring + (size_t)slot * g.max_bytes + off), (uint32_t)len,
                                g.nics[d].mr->lkey};
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
    if (g.host_reg == 0) {
        nic.mr = ibv_reg_mr(nic.pd, g.host, g.host_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        TORCH_CHECK(nic.mr, "rdma gather: ibv_reg_mr of ", g.host_bytes, " pinned bytes on ", name, " failed");
        nic.mr_recv = nic.mr;
    } else {
        nic.mr = ibv_reg_mr(nic.pd, g.send_ring, g.send_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        TORCH_CHECK(nic.mr, "rdma gather: ibv_reg_mr of the ", g.send_bytes, "-byte send ring on ", name, " failed");
        nic.mr_recv = ibv_reg_mr(nic.pd, g.recv_ring, g.recv_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        TORCH_CHECK(nic.mr_recv, "rdma gather: ibv_reg_mr of the ", g.recv_bytes, "-byte receive ring on ", name,
                    " failed");
    }
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
    g->host_reg = env_int("TF_RDMA_HOST_REG", 1) ? 1 : 0;
    g->fence_mode = env_int("TF_RDMA_STAGE_FENCE", 1) ? 1 : 0;
    g->send_bytes = send_b;
    g->recv_bytes = recv_b;
    uint64_t* meta = nullptr;
    if (g->host_reg == 0) {
        g->host_bytes = send_b + recv_b + ((meta_b + 4095) / 4096) * 4096;
        C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&g->host), g->host_bytes,
                                     cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset(g->host, 0, g->host_bytes);
        g->send_ring = g->host;
        g->recv_ring = g->host + send_b;
        meta = reinterpret_cast<uint64_t*>(g->host + send_b + recv_b);
    } else {
        // malloc'd pages pinned by cudaHostRegister (mapped; on GB10 the device pointer is the host pointer)
        auto reg = [](size_t n) {
            void* p = nullptr;
            TORCH_CHECK(posix_memalign(&p, 1 << 16, n) == 0 && p, "rdma gather: no memory for ", n, " pinned bytes");
            std::memset(p, 0, n);
            // cudaHostRegister does not page-lock on GB10 (the GPU reaches the pages through the host page tables);
            // the NIC's registration pins the rings, and mlock keeps the metadata (flags, doorbell, abort word)
            // resident too. Best effort: a refusal (RLIMIT_MEMLOCK) leaves the old behaviour.
            (void)mlock(p, n);
            C10_CUDA_CHECK(cudaHostRegister(p, n, cudaHostRegisterMapped | cudaHostRegisterPortable));
            return reinterpret_cast<char*>(p);
        };
        g->send_ring = reg(send_b);
        g->recv_ring = reg(recv_b);
        g->host_bytes = ((meta_b + 4095) / 4096) * 4096;
        if (env_int("TF_RDMA_META_REG", 1)) {
            g->host = reg(g->host_bytes);
        } else {
            C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&g->host), g->host_bytes,
                                         cudaHostAllocMapped | cudaHostAllocPortable));
            std::memset(g->host, 0, g->host_bytes);
        }
        meta = reinterpret_cast<uint64_t*>(g->host);
    }
    g->flags = meta;
    g->doorbell = g->flags + (size_t)slots * world;
    g->sizes = g->doorbell + 1;
    g->sent = g->sizes + slots;
    g->abort_word = g->sent + slots;
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->seq), 8));
    C10_CUDA_CHECK(cudaMemset(g->seq, 0, 8));
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->staged), 8));
    C10_CUDA_CHECK(cudaMemset(g->staged, 0, 8));
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->probe), 8 * 8));
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
            rkey.push_back(nic.mr_recv->rkey);
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
    // TF_RDMA_STAGE_BLOCKS (default 16): the most staging blocks, one per 1024 float4 of the partial (1 = one block)
    static const int stage_max = [] {
        const char* e = std::getenv("TF_RDMA_STAGE_BLOCKS");
        return std::max(1, e && *e ? std::atoi(e) : 16);
    }();
    static const int collect_max = [] {
        const char* e = std::getenv("TF_RDMA_COLLECT_BLOCKS");
        return std::max(1, e && *e ? std::atoi(e) : 48);
    }();
    // TF_RDMA_OWN_EARLY (default 1): the staging kernel copies this rank's slice to the output after its doorbell
    static const bool own_early = env_int("TF_RDMA_OWN_EARLY", 1) != 0;
    // TF_RDMA_COLLECT_UNROLL (default 1): up to four ring loads in flight a collecting thread
    static const bool collect_unroll = env_int("TF_RDMA_COLLECT_UNROLL", 1) != 0;
    const int sblocks = std::max(1, std::min(stage_max, (n4 + 1023) / 1024));
    float4* own_dst = own_early ? reinterpret_cast<float4*>(recv.data_ptr()) + (size_t)g.rank * n4 : nullptr;
    stage_kernel<<<sblocks, STAGE_THREADS, 0, stream>>>(reinterpret_cast<const float4*>(send.data_ptr()), n4,
                                                        dev(g.send_ring), dev(g.sizes), dev(g.doorbell), dev(g.sent),
                                                        g.seq, g.slots, g.max_bytes, probe, dev(g.abort_word), g.staged,
                                                        g.fence_mode, own_dst);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // TF_RDMA_COLLECT_BLOCKS (default 48; 16 before): the most collecting blocks, one per 1024 float4 copied out
    const int blocks = std::max(1, std::min(collect_max, (n4 * g.world + COLLECT_THREADS * 4 - 1) / (COLLECT_THREADS * 4)));
    collect_kernel<<<blocks, COLLECT_THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(send.data_ptr()), reinterpret_cast<float4*>(recv.data_ptr()), n4,
        dev(g.recv_ring), dev(g.flags), g.seq, g.rank, g.world, g.slots, g.max_bytes, probe, dev(g.abort_word),
        own_early ? 1 : 0, collect_unroll ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
