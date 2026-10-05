#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3x_grouped_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t);
void exl3x_grouped_rows_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3x_grouped_mma_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3x_grouped_mma2_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int64_t);
void exl3x_grouped_mma3_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int64_t, int64_t, int64_t);
void exl3x_grouped_down3_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3x_combine_y_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                          int64_t, int64_t, int64_t);
void exl3x_dequant_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_group_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_group_count_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_group_place_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t,
                            int64_t, int64_t);
void exl3x_rot_in_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_gateup_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                double, int64_t);
void exl3x_down_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t);
void exl3x_combine_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_down_combine_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
                             const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t);

void exl3x_decode_prep_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t,
                            int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t,
                            int64_t, int64_t, at::Tensor&, int64_t);
void exl3x_grouped_decode_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                               int64_t, int64_t, int64_t, int64_t, int64_t, const at::Tensor&, int64_t,
                               const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, double, int64_t,
                               at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                               int64_t, at::Tensor&, int64_t, at::Tensor&, at::Tensor&, at::Tensor&, int64_t,
                               int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
             const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
             const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK,
             int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo, int64_t hi,
             int64_t cp) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_grouped_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, SK, slots, cb, nt, warps,
                       pf, lo, hi, cp);
}

void grouped_rows(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                  const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                  const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK,
                  int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t g, int64_t lo,
                  int64_t hi, int64_t fold) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * (fold ? 1 : SK) * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_grouped_rows_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, SK, slots, cb, nt,
                            warps, pf, g, lo, hi, fold);
}

void grouped_mma(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                 const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                 const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK,
                 int64_t slots, int64_t cb, int64_t warps, int64_t lo, int64_t hi, int64_t fold) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_grouped_mma_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, SK, slots, cb, warps,
                           lo, hi, fold);
}

void grouped_mma2(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                  const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                  const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                  int64_t slots, int64_t cb, int64_t lo, int64_t hi) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_grouped_mma2_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, slots, cb, lo, hi);
}

void grouped_mma3(const at::Tensor& x, int64_t x_stride, const at::Tensor& suh0, const at::Tensor& suh1,
                  const at::Tensor& TP0, const at::Tensor& TP1, const at::Tensor& B0, const at::Tensor& B1,
                  const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members, at::Tensor Z,
                  int64_t K, int64_t N, int64_t P, int64_t slots, int64_t cb, int64_t lo, int64_t hi, int64_t nw) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, "x: CUDA bf16");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= 2 * P * N, "Z too small");
    c10::cuda::CUDAGuard guard(x.device());
    exl3x_grouped_mma3_cuda(x, x_stride, suh0, suh1, TP0, TP1, B0, B1, uids, ucount, members, Z, K, N, P, slots, cb,
                            lo, hi, nw);
}

void grouped_down3(const at::Tensor& Xd, const at::Tensor& TP, const at::Tensor& B, const at::Tensor& uids,
                   const at::Tensor& ucount, const at::Tensor& members, const at::Tensor& svh, const at::Tensor& wts,
                   at::Tensor Y, int64_t K, int64_t N, int64_t P, int64_t slots, int64_t cb, int64_t lo, int64_t hi) {
    check(Xd, at::kHalf, "Xd");
    check(TP, at::kLong, "TP");
    check(B, at::kInt, "B");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(svh, at::kHalf, "svh");
    check(wts, at::kFloat, "wts");
    TORCH_CHECK(Y.is_cuda() && Y.scalar_type() == at::kBFloat16 && Y.is_contiguous() && Y.numel() >= P * N,
                "Y: CUDA bf16 [P, N]");
    TORCH_CHECK(wts.numel() >= P, "wts: a weight a slot");
    c10::cuda::CUDAGuard guard(Xd.device());
    exl3x_grouped_down3_cuda(Xd, TP, B, uids, ucount, members, svh, wts, Y, K, N, P, slots, cb, lo, hi);
}

void combine_y(const at::Tensor& Y, const at::Tensor& pick, const at::Tensor& add, at::Tensor out, int64_t rows,
               int64_t D, int64_t slots, int64_t E, int64_t has_add) {
    TORCH_CHECK(Y.is_cuda() && Y.scalar_type() == at::kBFloat16, "Y: CUDA bf16");
    check(pick, at::kInt, "pick");
    check(out, at::kFloat, "out");
    if (has_add) check(add, at::kFloat, "add");
    c10::cuda::CUDAGuard guard(Y.device());
    exl3x_combine_y_cuda(Y, pick, add, out, rows, D, slots, E, has_add);
}

void dequant(const at::Tensor& T, at::Tensor out, int64_t k2, int64_t cb) {
    TORCH_CHECK(T.is_cuda() && T.scalar_type() == at::kShort && T.is_contiguous() && T.dim() == 3,
                "T: int16 [K/16, N/16, 8 * k2]");
    TORCH_CHECK(T.size(2) == 8 * k2, "trellis last dim must be 8 * k2");
    check(out, at::kHalf, "out");
    const int64_t K = T.size(0) * 16, N = T.size(1) * 16;
    TORCH_CHECK(out.numel() == K * N, "out must be [K, N]");
    c10::cuda::CUDAGuard guard(T.device());
    exl3x_dequant_cuda(T, out, K, N, k2, cb);
}

void group(const at::Tensor& pick, at::Tensor uids, at::Tensor ucount, at::Tensor members, int64_t R, int64_t slots,
           int64_t E) {
    check(pick, at::kInt, "pick");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    TORCH_CHECK(pick.numel() >= R * slots, "pick too small");
    TORCH_CHECK(uids.numel() >= std::min<int64_t>(R * slots, E), "uids too small");
    TORCH_CHECK(members.size(0) >= uids.numel() && members.size(1) >= 1, "members too small");
    c10::cuda::CUDAGuard guard(pick.device());
    exl3x_group_cuda(pick, uids, ucount, members, R, slots, E);
}

void group_count(const at::Tensor& pick, at::Tensor counts, int64_t R, int64_t slots, int64_t E) {
    check(pick, at::kInt, "pick");
    check(counts, at::kInt, "counts");
    TORCH_CHECK(pick.numel() >= R * slots && counts.numel() >= E, "pick or counts too small");
    c10::cuda::CUDAGuard guard(pick.device());
    exl3x_group_count_cuda(pick, counts, R, slots, E);
}

void group_place(const at::Tensor& pick, const at::Tensor& counts, at::Tensor uids, at::Tensor ucount,
                 at::Tensor members, int64_t R, int64_t slots, int64_t E) {
    check(pick, at::kInt, "pick");
    check(counts, at::kInt, "counts");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    TORCH_CHECK(pick.numel() >= R * slots && counts.numel() >= E, "pick or counts too small");
    TORCH_CHECK(uids.numel() >= std::min<int64_t>(R * slots, E), "uids too small");
    TORCH_CHECK(members.size(0) >= uids.numel() && members.size(1) >= 1, "members too small");
    c10::cuda::CUDAGuard guard(pick.device());
    exl3x_group_place_cuda(pick, counts, uids, ucount, members, R, slots, E);
}

void rot_in(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
            const at::Tensor& suh1, at::Tensor out0, at::Tensor out1, int64_t rows, int64_t K, int64_t slots,
            int64_t E) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf), "x: bf16/fp16 CUDA");
    check(pick, at::kInt, "pick");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(out0, at::kHalf, "out0");
    check(out1, at::kHalf, "out1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    c10::cuda::CUDAGuard guard(x.device());
    exl3x_rot_in_cuda(x, x_stride, pick, suh0, suh1, out0, out1, rows, K, slots, E);
}

void gateup_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g, const at::Tensor& svh_u,
                     const at::Tensor& suh_d, at::Tensor xd, int64_t rows, int64_t P, int64_t N, int64_t SK,
                     int64_t slots, int64_t E, double limit, int64_t act_mode) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_g, at::kHalf, "svh_g");
    check(svh_u, at::kHalf, "svh_u");
    check(suh_d, at::kHalf, "suh_d");
    check(xd, at::kHalf, "xd");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_gateup_epilogue_cuda(Z, pick, svh_g, svh_u, suh_d, xd, rows, P, N, SK, slots, E, limit, act_mode);
}

void down_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y, int64_t rows,
                   int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_epilogue_cuda(Z, pick, svh_d, y, rows, P, D, SK, slots, E);
}

void combine(const at::Tensor& y, const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t D, int64_t slots) {
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    c10::cuda::CUDAGuard guard(y.device());
    exl3x_combine_cuda(y, wts, out, rows, D, slots);
}

void down_combine(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y,
                  const at::Tensor& wts, const at::Tensor& add, at::Tensor out, int64_t rows, int64_t P, int64_t D,
                  int64_t SK, int64_t slots, int64_t E, int64_t store_y, int64_t has_add) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    if (has_add) {
        check(add, at::kFloat, "add");
        TORCH_CHECK(add.numel() >= rows * D, "add too small");
    }
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_combine_cuda(Z, pick, svh_d, y, wts, add, out, rows, P, D, SK, slots, E, store_y, has_add);
}

void decode_prep(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
                 const at::Tensor& suh1, at::Tensor out0, at::Tensor out1, at::Tensor uids, at::Tensor ucount,
                 at::Tensor members, int64_t rows, int64_t K, int64_t slots, int64_t E, const at::Tensor& wts,
                 const at::Tensor& y, const at::Tensor& add, at::Tensor out, int64_t has_wts, int64_t has_add,
                 int64_t store_y, at::Tensor epoch, int64_t has_epoch) {
    TORCH_CHECK(x.is_cuda() && x.stride(1) == 1, "x: rows of contiguous values");
    check(pick, at::kInt, "pick");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(out0, at::kHalf, "out0");
    check(out1, at::kHalf, "out1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    TORCH_CHECK(out0.numel() >= rows * slots * K && out1.numel() >= rows * slots * K, "rotated rows too small");
    TORCH_CHECK(pick.numel() >= rows * slots, "pick too small");
    TORCH_CHECK(uids.numel() >= std::min<int64_t>(rows * slots, E), "uids too small");
    TORCH_CHECK(members.size(1) >= 1, "members");
    if (has_wts) {
        check(wts, at::kFloat, "wts");
        check(y, at::kFloat, "y");
        check(out, at::kFloat, "out");
        TORCH_CHECK(out.dim() == 2 && out.size(0) >= rows, "out [rows, D]");
        TORCH_CHECK(y.numel() >= rows * slots * out.size(1), "y too small");
        if (has_add) check(add, at::kFloat, "add");
    }
    c10::cuda::CUDAGuard guard(x.device());
    if (has_epoch) check(epoch, at::kInt, "epoch");
    exl3x_decode_prep_cuda(x, x_stride, pick, suh0, suh1, out0, out1, uids, ucount, members, rows, K, slots, E, wts, y,
                           add, out, has_wts, has_add, store_y, epoch, has_epoch);
}

void grouped_decode(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                    const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                    const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                    int64_t SK, int64_t slots, int64_t cb, int64_t stages, int64_t lo, int64_t hi, int64_t epi,
                    const at::Tensor& pick, int64_t E, const at::Tensor& svh0, const at::Tensor& svh1,
                    const at::Tensor& suh_d, at::Tensor xd, double limit, int64_t act_mode, at::Tensor y,
                    const at::Tensor& wts, const at::Tensor& add, at::Tensor out, int64_t has_wts, int64_t has_add,
                    int64_t store_y, at::Tensor cnt, int64_t pdl, at::Tensor ready, at::Tensor ready_cnt,
                    at::Tensor epoch, int64_t use_ready, int64_t discard) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(cnt, at::kInt, "cnt");
    if (use_ready) {
        check(ready, at::kInt, "ready");
        check(ready_cnt, at::kInt, "ready_cnt");
        check(epoch, at::kInt, "epoch");
    }
    TORCH_CHECK(epi >= 0 && epi <= 2, "epi 0, 1 or 2");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    TORCH_CHECK(epi == 2 || Z.numel() >= mats * SK * P * N, "Z too small");
    TORCH_CHECK(pick.numel() >= P, "pick too small");
    TORCH_CHECK(P % slots == 0, "P = rows x slots");
    if (epi == 1) {
        TORCH_CHECK(mats == 2, "the gate/up epilogue takes gate and up");
        check(svh0, at::kHalf, "svh_g");
        check(svh1, at::kHalf, "svh_u");
        check(suh_d, at::kHalf, "suh_d");
        check(xd, at::kHalf, "xd");
        TORCH_CHECK(xd.numel() >= P * N, "xd too small");
    } else if (epi == 2) {
        check(svh0, at::kHalf, "svh_d");
        check(y, at::kFloat, "y");
        TORCH_CHECK(y.numel() >= P * N, "y too small");
        if (has_wts) {
            check(wts, at::kFloat, "wts");
            check(out, at::kFloat, "out");
            TORCH_CHECK(wts.numel() >= P && out.numel() >= (P / slots) * N, "wts / out too small");
            if (has_add) check(add, at::kFloat, "add");
        }
    }
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_grouped_decode_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, SK, slots, cb,
                              stages, lo, hi, epi, pick, E, svh0, svh1, suh_d, xd, limit, act_mode, y, wts, add, out,
                              has_wts, has_add, store_y, cnt, pdl, ready, ready_cnt, epoch, use_ready, discard);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped", &grouped, py::arg("X0"), py::arg("X1"), py::arg("TP0"), py::arg("TP1"), py::arg("B0"),
          py::arg("B1"), py::arg("uids"), py::arg("ucount"), py::arg("members"), py::arg("Z"), py::arg("mats"),
          py::arg("K"), py::arg("N"), py::arg("P"), py::arg("SK"), py::arg("slots"), py::arg("cb"), py::arg("nt"),
          py::arg("warps"), py::arg("pf"), py::arg("lo"), py::arg("hi"), py::arg("cp") = 0);
    m.def("grouped_rows", &grouped_rows);
    m.def("grouped_decode", &grouped_decode, py::arg("X0"), py::arg("X1"), py::arg("TP0"), py::arg("TP1"),
          py::arg("B0"), py::arg("B1"), py::arg("uids"), py::arg("ucount"), py::arg("members"), py::arg("Z"),
          py::arg("mats"), py::arg("K"), py::arg("N"), py::arg("P"), py::arg("SK"), py::arg("slots"), py::arg("cb"),
          py::arg("stages"), py::arg("lo"), py::arg("hi"), py::arg("epi"), py::arg("pick"), py::arg("E"),
          py::arg("svh0"), py::arg("svh1"), py::arg("suh_d"), py::arg("xd"), py::arg("limit"), py::arg("act_mode"),
          py::arg("y"), py::arg("wts"), py::arg("add"), py::arg("out"), py::arg("has_wts"), py::arg("has_add"),
          py::arg("store_y"), py::arg("cnt"), py::arg("pdl"), py::arg("ready"), py::arg("ready_cnt"),
          py::arg("epoch"), py::arg("use_ready"), py::arg("discard") = 0);
    m.def("decode_prep", &decode_prep);
    m.def("grouped_mma", &grouped_mma);
    m.def("grouped_mma2", &grouped_mma2);
    m.def("grouped_mma3", &grouped_mma3);
    m.def("grouped_down3", &grouped_down3);
    m.def("combine_y", &combine_y);
    m.def("dequant", &dequant);
    m.def("group", &group);
    m.def("group_count", &group_count);
    m.def("group_place", &group_place);
    m.def("rot_in", &rot_in);
    m.def("gateup_epilogue", &gateup_epilogue);
    m.def("down_epilogue", &down_epilogue);
    m.def("combine", &combine);
    m.def("down_combine", &down_combine);
}
