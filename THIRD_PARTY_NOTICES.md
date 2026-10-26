# Third-party notices

TensorFold uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
They are installed as dependencies.

## MLX and mlx-lm adaptations

The DeltaNet implementations in `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.

## Qwen Flash Next

The n-gram ID helpers in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py` translate
Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` into MLX and NumPy with renamed
identifiers. Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

## GLM-5.3-Flash on Apple Silicon

The MLX engine of `glm5_next` (`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits.

## CUDA

CUDA backends use [PyTorch](https://github.com/pytorch/pytorch), BSD-3-Clause, and
[Triton](https://github.com/triton-lang/triton), MIT, supplied by NVIDIA's container rather than bundled.
Dense Qwen implements mlx-lm's model math. CUDA DFlash2 implementations port z-lab's architecture under
the MIT License, Copyright © 2026 Z Lab.

Flash Next implements transformers' model math under the attribution above. Its new DeltaNet kernel
follows flash-linear-attention's numerics, MIT; its NCCL wrapper follows vLLM's stream convention,
Apache-2.0, without copying either implementation.

GLM's CUDA engine implements transformers' `models/glm5_next/modular_glm5_next.py` math, Apache-2.0,
without including that source. Its draft inputs and thinking-off rendering follow the public GLM recipe
from Mia-AiLab without including recipe code.

GLM EXL3 (`families/glm5_next/cuda/exl3.py`, `exl3.cu`, `exl3_mm.py`), the shared EXL3 module
(`src/tensorfold/cuda/exl3/`) and the EXL3 loaders of Qwen3.8-27B and Qwen3.8 Flash Next
(`families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py`) read
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format: its trellis layout and bitstream, its
"3inst", "mcg" and "mul1" codebooks, its half-integer bit widths and its tensor-core fragment order. Flash
Next's packs also carry ExLlamaV3's n-gram row codec, read as its `ngram_dequant` reads it. ExLlamaV3 uses the
MIT License, Copyright © 2025 Turboderp. TensorFold's decoders and kernels are separate implementations,
checked bit for bit against ExLlamaV3's dequantization.

Flash Next's optional int8 and int4 KV caches (`families/qwen4_exp/cuda/kvcache.py`) follow the cache quantization scheme of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) `-cq 8` and `-cq 4` (MIT License, Copyright (c) 2025 Turboderp, text below): groups of 32, one fp16 absmax scale per group, the group rotated by a 32-point Hadamard, midpoint-grid codes, `compand_a == 0`. 8-bit stores each code as a signed int8 (`q - 128`). 4-bit stores two unsigned codes per byte, low nibble first (the same bits as ExLlamaV3's little-endian packing, a uint8 tensor rather than their uint32 words). Their dequantizer folds another `1/sqrt(32)` into the scale and applies the unnormalized butterfly on the way out; this cache applies the normalized H32 to the query and to the merged output instead, and leaves the stored codes rotated. Scales match their quantizer bit for bit. Reconstructed values agree within fp16/bf16 rounding (under 0.01 on random groups), not bit for bit. The quantizer and the attention dequant are written for TensorFold and checked against an independent reference of that arithmetic.

GLM-5.3's optional quantized MLA caches (`TF_GLM_KV=q8` .. `q2`: `families/glm_moe_dsa/cuda/kvq.py`, read inside the
attention kernels of `mla_pe.py` and the indexer scorer of `select.py`) port the format and math of
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s `CacheLayer_MLA_quant` (`exllamav3/cache/mla.py`, its
quantizer in `exllamav3_ext/cache/q_cache_kernels.cuh` and its online dequantization in
`modules/attention_fn/triton_paged.py`; MIT License, Copyright (c) 2025 Turboderp, text below): a token's latent in
groups of 32 values rotated by the orthonormal 32-point Hadamard, one fp16 absmax scale a group, codes on the midpoint
grid packed in power-of-two bit planes, the rope key kept as it is, and readers that rotate the query once and the
attention's value sum back instead of rotating keys. TensorFold quantizes from bf16 rows in Triton (ExLlamaV3: fp16
rows in CUDA), rotates in fp32 and stores the bit planes in its own word order; the kernels are written for
TensorFold and checked against a float64 reference of that arithmetic. GLM-5.3's FP8 cache (`kv8.py`) is TensorFold's.

GLM-5.3's decode context parallelism (`TF_GLM_DCP=4`: `families/glm_moe_dsa/cuda/dcp.py` and the interleaved cache
writers) follows the design of TensorFold PR #159 by drowzeys (commit dc43ca6, "glm_moe_dsa: 1M-token context (decode
context parallelism 4)"; its `fused._attention_dcp`, `_merge_lse`, `_dcp_combine`, `dcp_gather` and selection):
positions interleaved over the ranks (p % 4), each rank's top-2048 merged into the same tie-free global choice on every
rank, all-gathered absorbed queries, every rank attending all heads over its own keys, and log-sum-exp partials merged
in rank order. It is restated for this fork's kernels and cache formats.

GLM-5.3 TP4's prefill pieces ported from TensorFold PR #159 by drowzeys (branch prefill159-e3290a0), restated
for this fork where noted:

- `cuda/exl3/prefill.py` `UnpackCache` (commits 1da2529 + 8b97845): a prompt chunk's second micro-batch reuses the
  first's decoded dense weights, LRU past `TF_GLM_UNPACK_CACHE_MB` (384 MiB), entries weakref keyed by layer id
  and deq mode (this fork dequantizes the unpack in place, so entries hold the post-dequant W).
- `families/glm_moe_dsa/cuda/select.py` `_scores` and `dcp.py` `_scores_dcp` (commit f0f382d): the indexer scoring
  tiles of 128 keys, 2 warps, 2 stages — bit-identical to the previous 64/4/3 by per-token independence.
- `families/glm_moe_dsa/cuda/engine.py` `prompt_chunk_rows` (commit f20913a): 4,096-row prompt chunks while a prompt
  is under 3 chunks; off by default here (`TF_GLM_PROMPT_ROWS_SHORT`), since without the chunk-invariant prompt path
  it moves prompt bits.

GLM-5.3 TP4's decode-context-parallelism prompt path ports from MiaAI-Lab's
[GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold)
(patches against TensorFold v0.6.0, Apache-2.0, head eb6414b), adapted from her `glm5_next` family to this fork's
`glm_moe_dsa` and restated for its kernels and cache formats:

- `families/glm_moe_dsa/cuda/dcp.py` `owner_rows`, `_threshold`, `_keep`, `owner_select` and `pipe_attention`
  (patch 0131, glm-full-cp-prompt-pipe): the prompt chunks' selection candidates go to the row owners (her
  `select_threshold` / `select_keep` radix select on our candidate keys, flipped to the uint64 order), and the
  query all-gather and the partials' owner exchange run in parts on the comm stream beside the attention (her
  `cp_prompt`, a ring of two buffers, first and last parts half as long). The same kernels over the same rows:
  the prompt chunks' bits are the lane's (`TF_GLM_DCP_PIPE`, on).
- `dcp.py` `PROMPT_TILES` (patch 0129, glm-full-cp-prompt-msa): the prompt chunks' DCP attention scores
  `mla_pe.PROMPT_KT` keys a tile instead of the decode windows' 32 (her prompt chunks through her faster
  sparse-attention kernel). Moves the prompt chunks' bits: `TF_GLM_DCP_PROMPT_TILES`, off by default.
- `dcp.py` `kvg_row_bytes`, `_gathered_like`, `_gather_views`, `kvg_buffers`, `kvg_gather`, `owner_lists`,
  `kvg_select`, `kvg_all_visible`, `kvg_attend` and the `engine.py` estimate term (patch 0141,
  glm-full-cp-kv-gather): prompt chunks gather every rank's latent and rope rows of the positions they see and
  attend their own heads over each row's whole top-k (her gathered planes, `owner_lists`, and
  `cp_kvg_all_visible`). Moves the prompt chunks' bits: `TF_GLM_DCP_KV_GATHER`, off by default.

Patches 0127 (overlapped MoE exchange) and 0135 (raw queries) are covered by this fork's existing prompt path
(micro-batch collectives on the comm stream; the row-share reduce) or superseded by 0141, and patch 0128's
expert prefetch is not ported yet.

The concurrency rule of TensorFold PR #159 by drowzeys (commit b1f9694, "chunks under 2048 rows fill whole while
others decode") is adapted to this fork's pacing gate: `families/glm_moe_dsa/cuda/multi.py` `_fill_choice`
returns a stream whose remaining prompt is under `TF_GLM_FILL_MIN_ROWS` (2048, his default) before the
`DECODE_SHARE` pacing. This fork fills whole chunks already (no layer steps), so only the wait is his.

## Ling-3.1-flash

The `bailing_v3` family implements the model math of inclusionAI's Ling-3.1-flash
(`modeling_bailing_moe_v3.py` of [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash),
revision 7b864daa), Apache License 2.0, Copyright 2025 Antgroup and The HuggingFace Inc. team. Its tests restate
parts of that file in torch as references: the grouped top-k router, the KDA recurrence and the rotary form.
`tests/fixtures/ling31/` holds the checkpoint's `chat_template.jinja` and `config.json`, part of its
`tokenizer_config.json` (the model card states MIT), and chat turns rendered with that template.

## Vendored code and weights

`src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified `dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.

`src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py` is the unmodified `encoding/encoding_dsv4.py` of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (revision 60d8d70), and
`tests/fixtures/deepseek_v4/` holds two of its test cases, MIT License, Copyright (c) 2023 DeepSeek.
The MTP layer TensorFold drafts with comes from that checkpoint's last shard (MIT), converted by
`families/deepseek_v4/convert.py`.

TensorFold ships no model weights. The `z-lab/Qwen3.8-27B-DFlash2` model card states Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives. Each checkpoint keeps its own license.

## MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Flash Next CUDA image integration

The multimodal rotary and image-feature integration is adapted from MiaAI-Lab's
[Flash Next vision patch 0008](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/a3aa89835022c55ca8e55008c37785954834e04f/patches/0008-flash-next-vision.patch),
MIT License, Copyright (c) 2026 MiaAI-Lab. The license is included in `LICENSES/MiaAI-Lab-MIT.txt`.
The port preserves the v0.5 CUDA execution APIs and adds an offline EXL3 vision adapter.
