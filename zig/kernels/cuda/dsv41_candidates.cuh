// Exact GPU candidate selection for the existing host sampler.
struct TfDsCandidate { float value; unsigned int id; };

extern "C" __global__ void __launch_bounds__(256) tf_ds_candidates_kernel(
    const float* logits, const TfDsCandidate* partial, TfDsCandidate* out,
    unsigned long long ld, int n, int k, int tiles, int merge) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long prefix, mask, keys[64];
    __shared__ TfDsCandidate kept[64];
    __shared__ int want, count, bad;
    const int tile = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const int begin = merge ? 0 : tile * 4096;
    const int end = merge ? tiles * k : min(begin + 4096, n);
    if (t == 0) { prefix = mask = 0; want = min(k, end - begin); count = bad = 0; }
    if (t < 64) { keys[t] = 0; kept[t] = {0.0f, 0xffffffffu}; }
    __syncthreads();
    // Nonfinite rows use the original host sampler, including its NaN semantics.
    for (int i = begin + t; i < end; i += 256) {
        const TfDsCandidate c = merge ? partial[row * tiles * k + i] : TfDsCandidate{logits[row * ld + i], (unsigned int)i};
        if (c.id == 0xfffffffeu || (c.id < 0xfffffffeu && !isfinite(c.value))) atomicExch(&bad, 1);
    }
    __syncthreads();
    if (bad) {
        if (t < k) out[(row * (merge ? 1 : tiles) + tile) * k + t] = {0.0f, 0xfffffffeu};
        return;
    }
    // Unique keys: value descending, ties (including signed zero) to lower id.
    for (int digit = 7; digit >= 0; --digit) {
        hist[t] = 0;
        __syncthreads();
        const unsigned long long p = prefix, m = mask;
        for (int i = begin + t; i < end; i += 256) {
            const TfDsCandidate c = merge ? partial[row * tiles * k + i] : TfDsCandidate{logits[row * ld + i], (unsigned int)i};
            const unsigned long long key = c.id == 0xffffffffu ? 0 : tf_ds_score_key(c.value, c.id);
            if ((key & m) == p) atomicAdd(hist + ((key >> (digit * 8)) & 255), 1u);
        }
        __syncthreads();
        if (t == 0) {
            int b = 255;
            while (b > 0 && (int)hist[b] < want) want -= hist[b--];
            prefix |= (unsigned long long)b << (digit * 8);
            mask |= 255ull << (digit * 8);
        }
        __syncthreads();
    }
    for (int i = begin + t; i < end; i += 256) {
        const TfDsCandidate c = merge ? partial[row * tiles * k + i] : TfDsCandidate{logits[row * ld + i], (unsigned int)i};
        if (c.id != 0xffffffffu) {
            const unsigned long long key = tf_ds_score_key(c.value, c.id);
            if (key >= prefix) {
                const int at = atomicAdd(&count, 1);
                if (at < k) { keys[at] = key; kept[at] = c; }
            }
        }
    }
    __syncthreads();
    // Sort a fixed 64 entries. Padding stays below every finite value.
    for (int size = 2; size <= 64; size *= 2) {
        for (int stride = size / 2; stride; stride /= 2) {
            if (t < 64) {
                const int other = t ^ stride;
                if (other > t && ((keys[t] < keys[other]) == ((t & size) == 0))) {
                    const auto key = keys[t]; keys[t] = keys[other]; keys[other] = key;
                    const auto c = kept[t]; kept[t] = kept[other]; kept[other] = c;
                }
            }
            __syncthreads();
        }
    }
    if (t < k) out[(row * (merge ? 1 : tiles) + tile) * k + t] = kept[t];
}
