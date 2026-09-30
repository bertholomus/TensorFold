# GLM-5.3-Flash TP4 per-rank weight math

Checkpoint: `/tank/models/llm/banked/GLM-5.3-BF16` (config.json).
Quant recipe: `convert.py -b 3.0 -hq` — routed experts average 3.0 bpw, attention 5 bpw,
dense MLP 4 bpw, output head 6 bpw, final average 3.04 bpw.
Target: 4x GB10 (121 GiB each, `ascent-01..04`), TP world = 4 (`TF_TP_WORLD=4`),
budget ~94 GiB/rank for weights + KV. Every number below is an exact integer byte count
divided by 4 ranks; GiB = bytes / 2^30. Re-derive with `python tools/verify_shape_math.py`.

## Dims from the real config.json

| dim | value |
|---|---|
| hidden_size (D) | 6144 |
| layers (L) | 78 (3 dense MLP + 75 MoE) |
| attention heads (H) | 64 → 16 per rank at world 4 |
| indexer heads | 32 (replicated, never split) |
| kv_lora_rank | 512 |
| qk head dim (192 nope + 64 rope) | 256 |
| v head dim | 256 |
| routed experts (E) | 256 (+ 1 shared expert, width 2048) |
| moe_intermediate (Ie) | 2048 |
| dense intermediate (Id) | 12288 |
| vocab (V) | 154,880 → 38,720 rows per rank at world 4 |
| MTP layers | 1 |

Split rules (`split.py`): gate/up rows and down cols for experts and dense MLP; q_b/kv_b
rows and o_proj cols for attention; lm_head by vocab rows; everything else replicated
(q_a, kv_a, indexer, norms, routers, hc, embed).

## Exact per-rank GiB table at TP4

| component | bytes/rank | GiB/rank |
|---|---:|---:|
| routed experts (256 x 75 layers, 3.0 bpw; gate/up rows, down cols /4) | 67,947,724,800 | 63.281 |
| shared expert (75 layers, 3.0 bpw, /4) | 265,420,800 | 0.247 |
| dense MLP (3 layers, 4 bpw, /4) | 84,934,656 | 0.079 |
| q_b_proj (heads/4, 5 bpw) | 408,944,640 | 0.381 |
| kv_b_proj (heads/4, 5 bpw) | 204,472,320 | 0.190 |
| o_proj (cols/4, 5 bpw) | 1,226,833,920 | 1.143 |
| q_a + kv_a (replicated, 5 bpw) | 785,940,480 | 0.732 |
| indexer wk + weights_proj + wq_b (replicated, BF16) | 1,461,977,088 | 1.362 |
| norms, routers, hc (replicated) | 922,196,808 | 0.859 |
| lm_head (6 bpw, vocab/4) | 178,421,760 | 0.166 |
| embed_tokens (replicated, BF16) | 475,791,360 | 0.443 |
| final norm | 12,288 | 0.000 |
| MTP layer (norms + eh_proj + DSA attn, MoE experts /4) | 1,047,957,504 | 0.976 |
| **TOTAL per rank** | **75,010,628,424** | **69.859** |

**Total: 69.859 GiB/rank — under the 94 GiB/rank budget with 24.14 GiB spare for
KV cache and prefill scratch.**

## The math, tensor by tensor

- Routed experts dominate: 75 MoE layers x 256 experts x (2 gate/up rows + 1 down col set)
  = 75 * 256 * 3 * 2048 * 6144 words at 3.0 bpw = 271,790,899,200 B full model,
  67,947,724,800 B/rank = 63.281 GiB.
- Dense MLP: 3 layers x 3 x 12288 x 6144 x 4 bpw = 339,738,624 B full, 0.079 GiB/rank.
- Attention per layer, 5 bpw: q_b [64*256, 2048] rows/4, kv_b [64*512, 512] rows/4,
  o [6144, 64*256] cols/4; replicated q_a [2048, 6144] + kv_a [576, 6144].
- Indexer is in the REP rule: all 32 indexer heads live on every rank (BF16 in the EXL3
  engine path): wk [128, 6144], weights_proj [32, 6144], wq_b [4096, 2048] per layer.
- lm_head [154880, 6144] at 6 bpw: each rank stores vocab/4 = 38,720 rows = 178,421,760 B.
- embed_tokens stays BF16 and replicated: 154880*6144*2 B, 0.443 GiB/rank.
- MTP: eh_proj [6144, 12288] 4 bpw + one plain DSA attention (5 bpw) + 256 experts /4.

## Fit conclusion

69.859 GiB weights leaves ~24 GiB of the 94 GiB/rank budget for KV (DSA latent cache is
512-wide per head, 16 heads/rank) and scratch. TP4 fits comfortably; the same recipe at
TP2 would need ~139 GiB/rank of weights alone and does not fit 121 GiB devices.
