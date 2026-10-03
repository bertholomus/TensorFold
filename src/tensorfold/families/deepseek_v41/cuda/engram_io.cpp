// Parallel row reads for the Engram tables: one pread per row on a pool of threads, so a decode step's rows (dozens,
// random offsets in ~95 GiB files) cost one NVMe latency instead of one per row. The pool's threads start once and
// wait for work (a round's reads used to start and join up to 48 threads a call, four calls a round); a call reads a
// layer's weight rows and scale rows together.
#include <torch/extension.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace {

class Pool {
  public:
    explicit Pool(int threads) {
        for (int k = 0; k < threads; ++k) workers_.emplace_back([this] { loop(); });
    }

    // run job(0 .. n - 1) on the pool and this thread; returns when every index ran
    void run(int64_t n, const std::function<void(int64_t)>& job) {
        if (n <= 0) return;
        std::unique_lock<std::mutex> call(call_);          // one caller at a time
        {
            std::lock_guard<std::mutex> g(m_);
            job_ = &job;
            n_ = n;
            next_ = 0;
            finished_ = 0;
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> g(m_);
        done_.wait(g, [&] { return finished_ >= n_; });
        job_ = nullptr;
    }

  private:
    void work() {
        for (;;) {
            int64_t i = next_.fetch_add(1);
            if (i >= n_) break;
            (*job_)(i);
            if (finished_.fetch_add(1) + 1 >= n_) {
                std::lock_guard<std::mutex> g(m_);
                done_.notify_all();
            }
        }
    }

    void loop() {
        int64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> g(m_);
                cv_.wait(g, [&] { return gen_ != seen; });
                seen = gen_;
                if (job_ == nullptr) continue;
            }
            work();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_, call_;
    std::condition_variable cv_, done_;
    const std::function<void(int64_t)>* job_ = nullptr;
    std::atomic<int64_t> n_{0}, next_{0}, finished_{0};
    int64_t gen_ = 0;
};

Pool& pool(int threads) {
    static Pool* p = new Pool(std::max(1, threads - 1));   // the caller is one more reader (never destroyed)
    return *p;
}

bool read_row(int fd, uint8_t* dst, int64_t bytes, int64_t off) {
    int64_t got = 0;
    while (got < bytes) {
        ssize_t r = pread(fd, dst + got, bytes - got, off + got);
        if (r <= 0) return false;
        got += r;
    }
    return true;
}

}  // namespace

// A layer's rows: idx [n] row ids; weight rows (fd_w, base_w, row_w bytes) into out_w, scale rows into out_s.
void gather_rows2(int64_t fd_w, int64_t base_w, int64_t row_w, int64_t fd_s, int64_t base_s, int64_t row_s,
                  const at::Tensor& idx, at::Tensor out_w, at::Tensor out_s, int64_t threads) {
    TORCH_CHECK(idx.dtype() == at::kLong && idx.is_contiguous() && !idx.is_cuda(), "idx: contiguous int64 CPU");
    TORCH_CHECK(out_w.dtype() == at::kByte && out_w.is_contiguous() && !out_w.is_cuda(), "out_w: contiguous uint8 CPU");
    TORCH_CHECK(out_s.dtype() == at::kByte && out_s.is_contiguous() && !out_s.is_cuda(), "out_s: contiguous uint8 CPU");
    const int64_t n = idx.numel();
    TORCH_CHECK(out_w.numel() >= n * row_w && out_s.numel() >= n * row_s, "out too small");
    const int64_t* ix = idx.data_ptr<int64_t>();
    uint8_t* dw = out_w.data_ptr<uint8_t>();
    uint8_t* ds = out_s.data_ptr<uint8_t>();
    std::atomic<int> failed(0);
    std::function<void(int64_t)> job = [&](int64_t j) {
        const int64_t i = j >> 1;
        bool ok = (j & 1) ? read_row((int)fd_s, ds + i * row_s, row_s, base_s + ix[i] * row_s)
                          : read_row((int)fd_w, dw + i * row_w, row_w, base_w + ix[i] * row_w);
        if (!ok) failed = 1;
    };
    pool((int)threads).run(2 * n, job);
    TORCH_CHECK(!failed, "Engram row read failed");
}

// The single-table call (kept for tools): rows of one table.
void gather_rows(int64_t fd, int64_t base, int64_t row_bytes, const at::Tensor& idx, at::Tensor out, int64_t threads) {
    TORCH_CHECK(idx.dtype() == at::kLong && idx.is_contiguous() && !idx.is_cuda(), "idx: contiguous int64 CPU");
    TORCH_CHECK(out.dtype() == at::kByte && out.is_contiguous() && !out.is_cuda(), "out: contiguous uint8 CPU");
    const int64_t n = idx.numel();
    TORCH_CHECK(out.numel() >= n * row_bytes, "out too small");
    const int64_t* ix = idx.data_ptr<int64_t>();
    uint8_t* dst = out.data_ptr<uint8_t>();
    std::atomic<int> failed(0);
    std::function<void(int64_t)> job = [&](int64_t i) {
        if (!read_row((int)fd, dst + i * row_bytes, row_bytes, base + ix[i] * row_bytes)) failed = 1;
    };
    pool((int)threads).run(n, job);
    TORCH_CHECK(!failed, "Engram row read failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_rows", &gather_rows, "parallel preads of rows into a CPU buffer",
          py::call_guard<py::gil_scoped_release>());
    m.def("gather_rows2", &gather_rows2, "a layer's weight and scale rows in one parallel pass",
          py::call_guard<py::gil_scoped_release>());
}
