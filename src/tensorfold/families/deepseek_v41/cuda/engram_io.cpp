// Parallel row reads for the Engram tables: one pread per row on a pool of threads, so a decode step's rows (dozens,
// random offsets in ~95 GiB files) cost one NVMe latency instead of one per row.
#include <torch/extension.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <atomic>

void gather_rows(int64_t fd, int64_t base, int64_t row_bytes, const at::Tensor& idx, at::Tensor out, int64_t threads) {
    TORCH_CHECK(idx.dtype() == at::kLong && idx.is_contiguous() && !idx.is_cuda(), "idx: contiguous int64 CPU");
    TORCH_CHECK(out.dtype() == at::kByte && out.is_contiguous() && !out.is_cuda(), "out: contiguous uint8 CPU");
    const int64_t n = idx.numel();
    TORCH_CHECK(out.numel() >= n * row_bytes, "out too small");
    const int64_t* ix = idx.data_ptr<int64_t>();
    uint8_t* dst = out.data_ptr<uint8_t>();
    std::atomic<int64_t> next(0);
    std::atomic<int> failed(0);
    auto work = [&]() {
        for (;;) {
            int64_t i = next.fetch_add(1);
            if (i >= n) break;
            int64_t off = base + ix[i] * row_bytes;
            int64_t got = 0;
            while (got < row_bytes) {
                ssize_t r = pread((int)fd, dst + i * row_bytes + got, row_bytes - got, off + got);
                if (r <= 0) { failed = 1; break; }
                got += r;
            }
        }
    };
    int64_t t = std::max<int64_t>(1, std::min<int64_t>(threads, n));
    if (t == 1) { work(); }
    else {
        std::vector<std::thread> pool;
        pool.reserve(t);
        for (int64_t k = 0; k < t; ++k) pool.emplace_back(work);
        for (auto& th : pool) th.join();
    }
    TORCH_CHECK(!failed, "Engram row read failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_rows", &gather_rows, "parallel preads of rows into a CPU buffer",
          py::call_guard<py::gil_scoped_release>());
}
