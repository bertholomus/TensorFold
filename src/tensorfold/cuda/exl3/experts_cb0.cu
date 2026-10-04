// Grouped EXL3 expert GEMV instances for codebook 0 (3inst).
#include "experts_grouped.cuh"

namespace tf_exl3x {
template void grouped_launch<0>(const GroupedArgs&, cudaStream_t);
template void grouped_sm_launch<0>(const GroupedArgs&, cudaStream_t);
template void grouped_rows_launch<0>(const GroupedArgs&, cudaStream_t);
template void grouped_mma_launch<0>(const GroupedArgs&, cudaStream_t);
template void grouped_mma2_launch<0>(const GroupedArgs&, cudaStream_t);
template void grouped_mma3_launch<0>(const GroupedArgs&, const Mma3Args&, cudaStream_t);
template void grouped_down3_launch<0>(const GroupedArgs&, const half*, const float*, __nv_bfloat16*,
                                       cudaStream_t);
template void dequant_launch<0>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x
