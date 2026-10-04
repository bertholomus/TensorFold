#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3_rot_in_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&);
void exl3_linear_cuda(const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
                      const c10::optional<at::Tensor>&, at::Tensor&, const c10::optional<at::Tensor>&, at::Tensor&,
                      int64_t, int64_t, int64_t, int64_t);
void exl3_unpack_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
int exl3_glinear_max();
void exl3_rot_many_cuda(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                        bool);
void exl3_glinear_cuda(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<int64_t>&,
                       const std::vector<int64_t>&, const std::vector<at::Tensor>&,
                       const std::vector<c10::optional<at::Tensor>>&, const std::vector<at::Tensor>&,
                       const c10::optional<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<int64_t>&,
                       int64_t, int64_t, int64_t, bool, const std::vector<c10::optional<at::Tensor>>&,
                       const std::vector<c10::optional<at::Tensor>>&, const std::vector<int64_t>&,
                       const std::vector<int64_t>&, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
                       const c10::optional<at::Tensor>&, int64_t, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

static void check_io(const at::Tensor& x, const char* name) {
    const auto t = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                    (t == at::kHalf || t == at::kBFloat16 || t == at::kFloat),
                name, ": expected a contiguous 2-d fp16, bf16 or fp32 CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, ": must be 16-byte aligned");
}

// xh [M, K] fp16 = fp16(((x * suh) @ H) / sqrt(128)); x [M, K] fp16, bf16 or fp32.
void rot_in(const at::Tensor& x, const at::Tensor& suh, at::Tensor xh) {
    check_io(x, "x");
    check(suh, at::kHalf, "suh");
    check(xh, at::kHalf, "xh");
    TORCH_CHECK(x.size(1) % 128 == 0 && suh.numel() == x.size(1) && xh.sizes() == x.sizes(),
                "x and xh must be [M, K], K a multiple of 128, suh [K]");
    c10::cuda::CUDAGuard guard(x.device());
    exl3_rot_in_cuda(x, suh, xh);
}

// y [M, N] = (xh @ W_q) @ H * svh + bias; Z [SK, M, N] fp32 when SK > 1; counters int32 [8 * N / 128], left zero.
void linear(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& svh,
            const c10::optional<at::Tensor>& bias, at::Tensor y, const c10::optional<at::Tensor>& Z,
            at::Tensor counters, int64_t K2, int64_t cb, int64_t SK, int64_t WK) {
    check(xh, at::kHalf, "xh");
    check_io(y, "y");
    check(svh, at::kHalf, "svh");
    check(T, at::kInt, "T");
    check(counters, at::kInt, "counters");
    const int64_t M = xh.size(0), K = xh.size(1), N = y.size(1);
    TORCH_CHECK(xh.dim() == 2 && y.size(0) == M && M >= 1 && M <= 128, "xh and y must have the same 1 to 128 rows");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
    TORCH_CHECK(svh.numel() == N, "svh must have N elements");
    TORCH_CHECK(T.numel() == K * N * K2 / 64, "T must hold K * N * bits / 32 words");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(T.data_ptr()) % 16 == 0, "T must be 16-byte aligned");
    TORCH_CHECK(counters.numel() >= 8 * (N / 128), "counters must hold 8 * N / 128 ints");
    if (bias) check(*bias, at::kHalf, "bias");
    if (SK > 1) {
        TORCH_CHECK(Z.has_value(), "Z is needed with more than one split");
        check(*Z, at::kFloat, "Z");
        TORCH_CHECK(Z->numel() >= SK * M * N, "Z too small");
    }
    c10::cuda::CUDAGuard guard(xh.device());
    exl3_linear_cuda(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, K2, cb, SK, WK);
}

// Rows of a grouped layer's input or output: 2-d, unit column stride, 16-byte aligned rows (row-strided views too).
static void check_rows(const at::Tensor& x, const char* name) {
    const auto t = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && (t == at::kHalf || t == at::kBFloat16 || t == at::kFloat), name,
                ": expected a 2-d fp16, bf16 or fp32 CUDA tensor");
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) >= x.size(1) && x.stride(0) % 4 == 0 &&
                    reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                name, ": rows must be unit-stride, 16-byte aligned, a multiple of 4 elements apart");
}

// xhs[i] [M, K_i] fp16 (contiguous) = rot_in(xs[i], suhs[i]) for up to glinear_max() layers in one launch; xs may be
// row-strided views. pdl: a programmatic dependent launch (sm_90+): it may start while the kernel before it finishes.
void rot_many(const std::vector<at::Tensor>& xs, const std::vector<at::Tensor>& suhs,
              const std::vector<at::Tensor>& xhs, bool pdl) {
    const size_t n = xs.size();
    TORCH_CHECK(n >= 1 && (int)n <= exl3_glinear_max() && suhs.size() == n && xhs.size() == n,
                "1 to ", exl3_glinear_max(), " layers, one entry a layer in every list");
    const int64_t M = xs[0].size(0);
    TORCH_CHECK(M >= 1 && M <= 128, "1 to 128 rows");
    for (size_t i = 0; i < n; ++i) {
        check_rows(xs[i], "x");
        check(suhs[i], at::kHalf, "suh");
        check(xhs[i], at::kHalf, "xh");
        TORCH_CHECK(xs[i].size(0) == M && xs[i].size(1) % 128 == 0 && suhs[i].numel() == xs[i].size(1) &&
                        xhs[i].sizes() == xs[i].sizes() && xs[i].device() == xs[0].device() &&
                        xhs[i].device() == xs[0].device(),
                    "x [M, K] (K a multiple of 128), suh [K], xh [M, K]: the same M and device");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(xhs[i].data_ptr()) % 16 == 0, "xh must be 16-byte aligned");
    }
    c10::cuda::CUDAGuard guard(xs[0].device());
    exl3_rot_many_cuda(xs, suhs, xhs, pdl);
}

// Up to glinear_max() layers of one K2 / codebook / WK in one launch: ys[i] [M, N_i] = (xhs[i] @ W_i) @ H * svh_i +
// bias_i, xhs the rotated rows (rot_many), each layer with its own SK as linear() runs it (the same bits); ys may be
// row-strided views. Z fp32 holds SK_i * M * N_i floats of every layer with SK_i > 1, in order; counters as for
// linear(), one per layer, distinct. pdl as for rot_many (it reads only its first weights before the kernel before it
// is done).
void glinear(const std::vector<at::Tensor>& xhs, const std::vector<at::Tensor>& Ts,
             const std::vector<int64_t>& stride_k, const std::vector<int64_t>& stride_nb,
             const std::vector<at::Tensor>& svhs,
             const std::vector<c10::optional<at::Tensor>>& biases, const std::vector<at::Tensor>& ys,
             const c10::optional<at::Tensor>& Z, const std::vector<at::Tensor>& counters,
             const std::vector<int64_t>& SK, int64_t K2, int64_t cb, int64_t WK, bool pdl,
             const std::vector<c10::optional<at::Tensor>>& rsuhs = {},
             const std::vector<c10::optional<at::Tensor>>& rxhs = {}, const std::vector<int64_t>& roffs = {},
             const std::vector<int64_t>& ropes = {}, const c10::optional<at::Tensor>& rcos = c10::nullopt,
             const c10::optional<at::Tensor>& rsin = c10::nullopt, const c10::optional<at::Tensor>& rpos = c10::nullopt,
             int64_t rhd = 0, int64_t rrd = 0) {
    const size_t n = xhs.size();
    TORCH_CHECK(n >= 1 && (int)n <= exl3_glinear_max(), "1 to ", exl3_glinear_max(), " layers a launch");
    TORCH_CHECK(Ts.size() == n && stride_k.size() == n && stride_nb.size() == n && svhs.size() == n &&
                    biases.size() == n && ys.size() == n && counters.size() == n && SK.size() == n,
                "one entry a layer in every list");
    TORCH_CHECK(WK == 4 || WK == 8, "WK must be 4 or 8");
    const int64_t M = xhs[0].size(0);
    TORCH_CHECK(M >= 1 && M <= 128, "1 to 128 rows");
    int64_t zneed = 0;
    for (size_t i = 0; i < n; ++i) {
        check_rows(xhs[i], "xh");
        TORCH_CHECK(xhs[i].scalar_type() == at::kHalf, "xh: rotated fp16 rows (rot_many)");
        check_rows(ys[i], "y");
        check(svhs[i], at::kHalf, "svh");
        check(Ts[i], at::kInt, "T");
        check(counters[i], at::kInt, "counters");
        TORCH_CHECK(ys[i].device() == xhs[0].device() && Ts[i].device() == xhs[0].device() &&
                        xhs[i].device() == xhs[0].device(), "one device");
        const int64_t K = xhs[i].size(1), N = ys[i].size(1);
        TORCH_CHECK(xhs[i].size(0) == M && ys[i].size(0) == M, "every input and output has the same rows");
        TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
        TORCH_CHECK(svhs[i].numel() == N, "svh [N]");
        TORCH_CHECK(Ts[i].numel() == K * N * K2 / 64, "T must hold K * N * bits / 32 words");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(Ts[i].data_ptr()) % 16 == 0, "T must be 16-byte aligned");
        TORCH_CHECK(counters[i].numel() >= 8 * (N / 128), "counters must hold 8 * N / 128 ints");
        TORCH_CHECK(SK[i] >= 1 && (K / 16) % (SK[i] * WK) == 0, "K / 16 must split evenly over SK * WK warps");
        if (biases[i]) {
            check(*biases[i], at::kHalf, "bias");
            TORCH_CHECK(biases[i]->numel() == N, "bias [N]");
        }
        if (SK[i] > 1) {
            zneed += SK[i] * M * N;
            for (size_t j = 0; j < i; ++j)
                TORCH_CHECK(SK[j] == 1 || counters[j].data_ptr() != counters[i].data_ptr(),
                            "layers of one launch need their own counters");
        }
    }
    if (zneed) {
        TORCH_CHECK(Z.has_value(), "Z is needed with more than one split");
        check(*Z, at::kFloat, "Z");
        TORCH_CHECK(Z->numel() >= zneed, "Z too small");
    }
    TORCH_CHECK(rsuhs.size() == rxhs.size() && rsuhs.size() == roffs.size() && (rsuhs.empty() || rsuhs.size() == n),
                "rotation folds: one (suh, rows, offset) a layer, or none");
    for (size_t i = 0; i < rsuhs.size(); ++i) {
        if (!rsuhs[i] || !rxhs[i]) continue;
        check(*rsuhs[i], at::kHalf, "rsuh");
        check_rows(*rxhs[i], "rxh");
        TORCH_CHECK(rxhs[i]->scalar_type() == at::kHalf && rxhs[i]->size(0) == M, "rxh: fp16 rows, the same M");
        TORCH_CHECK(roffs[i] >= 0 && roffs[i] % 128 == 0 && roffs[i] + ys[i].size(1) <= rxhs[i]->size(1) &&
                        roffs[i] + ys[i].size(1) <= rsuhs[i]->numel(),
                    "the folded rotation's columns: 128-aligned, inside rxh and rsuh");
    }
    TORCH_CHECK(ropes.empty() || ropes.size() == n, "RoPE flags: one a layer, or none");
    for (size_t i = 0; i < ropes.size(); ++i) {
        if (!ropes[i]) continue;
        TORCH_CHECK(rcos && rsin && rpos, "a folded RoPE needs its cos / sin tables and positions");
        check(*rcos, at::kFloat, "rcos");
        check(*rsin, at::kFloat, "rsin");
        check(*rpos, at::kLong, "rpos");
        TORCH_CHECK(ys[i].scalar_type() == at::kBFloat16, "a folded RoPE rotates bf16 outputs");
        TORCH_CHECK(rhd % 128 == 0 && rrd % 4 == 0 && rrd <= 128 && ys[i].size(1) % rhd == 0 && rpos->numel() >= M,
                    "a folded RoPE: heads of a multiple of 128 columns, its part inside their last 128");
    }
    c10::cuda::CUDAGuard guard(xhs[0].device());
    exl3_glinear_cuda(xhs, Ts, stride_k, stride_nb, svhs, biases, ys, Z, counters, SK, K2, cb, WK, pdl, rsuhs, rxhs,
                      roffs, ropes, rcos, rsin, rpos, rhd, rrd);
}

// W [K, N] fp16 = W_q, the trellis tiles decoded; tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb words.
void unpack(const at::Tensor& T, at::Tensor W, int64_t stride_k, int64_t stride_nb, int64_t K2, int64_t cb) {
    check(T, at::kInt, "T");
    check(W, at::kHalf, "W");
    TORCH_CHECK(W.dim() == 2 && W.size(0) % 128 == 0 && W.size(1) % 128 == 0, "W must be [K, N], multiples of 128");
    TORCH_CHECK(T.numel() == W.numel() * K2 / 64, "T must hold K * N * bits / 32 words");
    c10::cuda::CUDAGuard guard(T.device());
    exl3_unpack_cuda(T, W, stride_k, stride_nb, K2, cb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rot_in", &rot_in);
    m.def("linear", &linear);
    m.def("unpack", &unpack);
    m.def("glinear", &glinear, py::arg("xhs"), py::arg("Ts"), py::arg("stride_k"), py::arg("stride_nb"),
          py::arg("svhs"), py::arg("biases"), py::arg("ys"), py::arg("Z"), py::arg("counters"), py::arg("SK"),
          py::arg("K2"), py::arg("cb"), py::arg("WK"), py::arg("pdl"),
          py::arg("rsuhs") = std::vector<c10::optional<at::Tensor>>{},
          py::arg("rxhs") = std::vector<c10::optional<at::Tensor>>{}, py::arg("roffs") = std::vector<int64_t>{},
          py::arg("ropes") = std::vector<int64_t>{}, py::arg("rcos") = c10::nullopt, py::arg("rsin") = c10::nullopt,
          py::arg("rpos") = c10::nullopt, py::arg("rhd") = 0, py::arg("rrd") = 0);
    m.def("glinear_max", &exl3_glinear_max);
    m.def("rot_many", &rot_many);
}
