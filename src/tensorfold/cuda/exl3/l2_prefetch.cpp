#include <torch/extension.h>
#include <vector>

int exl3_l2_prefetch_max();
void exl3_l2_prefetch_cuda(const std::vector<int64_t>&, const std::vector<int64_t>&, int64_t, int64_t, int64_t,
                           int64_t);

// Bulk L2 prefetches of the byte ranges (ptrs[i], sizes[i]) in order, chunk bytes each, from at most `blocks` blocks of
// 128 threads on the current stream, delay_ns after the kernel starts (wave_ns > 0: one block, a wave of 128 chunks
// every wave_ns); starts 16-byte aligned, sizes multiples of 16.
void l2_prefetch(const std::vector<int64_t>& ptrs, const std::vector<int64_t>& sizes, int64_t chunk, int64_t blocks,
                 int64_t delay_ns, int64_t wave_ns) {
    TORCH_CHECK(ptrs.size() == sizes.size() && (int)ptrs.size() <= exl3_l2_prefetch_max(), "at most ",
                exl3_l2_prefetch_max(), " ranges, a size for each");
    TORCH_CHECK(chunk >= 16 && chunk % 16 == 0 && chunk <= (1 << 20) && blocks >= 1, "chunk: 16 B .. 1 MiB, x16");
    for (size_t i = 0; i < ptrs.size(); ++i)
        TORCH_CHECK(ptrs[i] % 16 == 0 && sizes[i] % 16 == 0 && sizes[i] > 0, "ranges: 16-byte aligned, x16 bytes");
    exl3_l2_prefetch_cuda(ptrs, sizes, chunk, blocks, delay_ns, wave_ns);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l2_prefetch", &l2_prefetch);
    m.def("l2_prefetch_max", &exl3_l2_prefetch_max);
}
