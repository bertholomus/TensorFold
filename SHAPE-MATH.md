# GLM-5.3-Flash TP4 per-rank weight math

Checkpoint: `<BF16_DIR>` (config.json).
Quant recipe: `convert.py -b 3.0 -hq` — routed experts average 3.0 bpw, attention 5 bpw,
dense MLP 4 bpw, output head 6 bpw, final average 3.04 bpw.
Target: 4x GB10 (121 GiB each, `spark1..04`), TP world = 4 (`TF_TP_WORLD=4`),
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

## W=6: balanced uneven partition (4x 200G + 2x 200G members)

World 6 divides three dimensions unevenly; every split takes the balanced uneven share
(first n % 6 ranks one extra item, spans contiguous, shares differ by at most 1):

| dim | shares |
|---|---|
| q heads (64) | 11 / 11 / 11 / 11 / 10 / 10 |
| routed experts (256) | 43 / 43 / 43 / 43 / 42 / 42 |
| expert width rows (2048) | 342 / 342 / 341 / 341 / 341 / 341 |
| lm_head vocab rows (154,880) | 25,814 / 25,814 / 25,813 / 25,813 / 25,813 / 25,813 |
| dense intermediate (12,288) | 2048 each (divides) |

The EXL3 head keeps whole 128-column Hadamard blocks: 1,210 blocks split
202/202/202/202/201/201 (the padded width every rank computes with is ceil = 202 blocks,
tails masked with -inf), so every rank's head slice is a valid EXL3 linear of its own.
Indexer (32 heads), q_a/kv_a, embed_tokens, norms and routers stay replicated, immune
to the uneven split exactly as at TP4.

### Per-rank GiB table at TP6 (rank 0, the heavy 43-expert split; same recipe)

| component | bytes/rank | GiB/rank |
|---|---:|---:|
| routed experts (75 layers x 43 experts x 3 x 342 rows x 6144, 3.0 bpw) | 7,623,590,400 | 7.100 |
| shared expert (75 layers, 342 rows, 3.0 bpw) | 177,292,800 | 0.165 |
| dense MLP (3 layers, 4 bpw) | 56,623,104 | 0.053 |
| q_b_proj (11 heads, 5 bpw) | 281,149,440 | 0.262 |
| kv_b_proj (11 heads, 5 bpw) | 140,574,720 | 0.131 |
| o_proj (11 heads of cols, 5 bpw) | 843,448,320 | 0.786 |
| q_a + kv_a (replicated, 5 bpw) | 785,940,480 | 0.732 |
| indexer wk + weights_proj + wq_b (replicated, BF16) | 1,461,977,088 | 1.362 |
| norms, routers, hc (replicated) | 922,196,808 | 0.859 |
| lm_head (25,814 vocab rows, 6 bpw) | 118,950,912 | 0.111 |
| embed_tokens (replicated, BF16) | 1,903,165,440 | 1.772 |
| final norm | 12,288 | 0.000 |
| MTP layer (norms + eh_proj + DSA attn 5 bpw, 43 experts x 342 rows, 3.0 bpw) | 243,635,712 | 0.227 |
| **TOTAL per rank (rank 0)** | **14,440,362,312** | **13.449** |

Rank byte totals: 14,440,362,312 / 14,440,362,312 / 14,417,592,488 / 14,417,592,488 /
14,123,455,416 / 14,123,455,416 — max - min = 316,912,896 B (0.295 GiB), the balanced
guarantee in bytes. The table above amortizes nothing: embed_tokens is held in full on
every rank (the TP4 table divides it by 4; both conventions are per-rank accounting of
the same replicated tensor). At the TP4 table's amortized convention rank 0 carries
12,972,586,312 B = 12.082 GiB.

**Fit: 13.449 GiB weights/rank against the 94 GiB/rank budget — TP6 fits with ~80.5 GiB
spare for KV + scratch. The binding constraints at TP6 are KV per rank (10 heads vs 16 at
TP4 reduces it) and cross-rank latency of the 6-way all-gathers, not weight memory.**
