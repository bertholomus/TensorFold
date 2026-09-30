"""Exact per-rank TP4 weight bytes for GLM-5.3-Flash from the banked config.json.

Run: /tmp/tf-venv/bin/python tools/verify_shape_math.py
Recipe: convert.py -b 3.0 -hq  (experts 3.0, attn 5, dense 4, head 6 bpw)
"""

import json
from pathlib import Path

CFG = json.loads(Path("/tank/models/llm/banked/GLM-5.3-BF16/config.json").read_text())
D = CFG["hidden_size"]
L = CFG["num_hidden_layers"]
H = CFG["num_attention_heads"]
IH = CFG["index_n_heads"]
IDIM = CFG["index_head_dim"]
QL = CFG["q_lora_rank"]
KV = CFG["kv_lora_rank"]
QK = CFG["qk_nope_head_dim"] + CFG["qk_rope_head_dim"]
VD = CFG["v_head_dim"]
E = CFG["n_routed_experts"]
IE = CFG["moe_intermediate_size"]
ID_ = CFG["intermediate_size"]
DENSE = CFG["first_k_dense_replace"]
V = CFG["vocab_size"]
W = 4

GIB = 2 ** 30


def bytes_at(shape, bpw):
    return int.__mul__(shape[0], shape[1]) * bpw // 8


rows = []
add = rows.append

# routed experts: 75 moe layers, gate+up+down, 3.0 bpw; split rows/cols by 4
experts_full = 0
moe_layers = L - DENSE
experts_full = moe_layers * E * (2 * IE + IE) * D * 3 // 8      # 3*I_e rows total across gate/up/down -> (2*IE + IE)*D rows
add(("routed experts (256x75 layers, 3.0 bpw; gate/up rows, down cols /4)", experts_full // W if False else experts_full // 4))

# shared expert: 1 per moe layer, same widths, 3.0 bpw
shared_full = moe_layers * 3 * IE * D * 3 // 8
add(("shared expert (75 layers, 3.0 bpw, /4)", shared_full // 4))

# dense MLP: 3 layers, 4 bpw
dense_full = DENSE * 3 * ID_ * D * 4 // 8
add(("dense MLP (3 layers, 4 bpw, /4)", dense_full // 4))

# attention, 5 bpw. Split: q_b rows (H*QK -> /4), kv_b rows (H*(QK+VD) -> /4),
# o_proj cols (D, H*VD -> /4). Replicated: q_a, kv_a.
q_b_full = L * H * QK * QL * 5 // 8
kv_b_full = L * H * (QK + VD) * KV * 5 // 8
o_full = L * D * H * VD * 5 // 8
add(("q_b_proj (heads/4, 5 bpw)", q_b_full // 4))
add(("kv_b_proj (heads/4, 5 bpw)", kv_b_full // 4))
add(("o_proj (cols/4, 5 bpw)", o_full // 4))
add(("q_a + kv_a (replicated, 5 bpw)", L * (QL + KV + CFG["qk_rope_head_dim"]) * D * 5 // 8))

# indexer: wk [IDIM, D], weights_proj [IH, D], wq_b [IH*IDIM, QL] all replicated, BF16 (2 B)
indexer = L * ((IDIM * D) + (IH * D) + (IH * IDIM * QL)) * 2
add(("indexer wk + weights_proj + wq_b (replicated, BF16)", indexer))

# norms, routers, hc: small replicated BF16/FP32
misc = L * (D * 2 + 4 + 2) * 2 + L * (E * D + E) * (2 + 4)      # in/post norms, router weight+bias
hc = L * (24 * 4 * D + 24 * 4 + 3 * 4) * 2 * 2                  # attn+ffn hc fn/base/scale
add(("norms, routers, hc (replicated)", misc + hc))

# lm_head: 6 bpw, vocab rows /4
head_full = V * D * 6 // 8
add(("lm_head (6 bpw, vocab/4)", head_full // 4))

# embed: BF16 replicated
add(("embed_tokens (replicated, BF16)", V * D * 2 // 4))

# final norm
add(("final norm (replicated)", D * 2))

# MTP layer: enorm [D], hnorm [D], eh_proj [D, 2D] 4bpw, shared_head.norm [D], plus a plain
# DSA+MoE layer (no hc). Use same bpw recipe, replicated where the engine replicates.
mtp = (2 * D + D) * 2 + D * (2 * D) * 4 // 8
mtp_attn = (H * QK * QL + H * (QK + VD) * KV + D * H * VD + (QL + KV) * D) * 5 // 8
mtp_moe = E * 3 * IE * D * 3 // 8
add(("MTP layer (norms + eh_proj + DSA attn, MoE experts /4, recipe bpw)", mtp + mtp_attn + mtp_moe // 4))

total = sum(b for _, b in rows)
for name, b in rows:
    print(f"{name:70s} {b:>15,} B  {b / GIB:8.3f} GiB")
print("-" * 100)
print(f"{'TOTAL per rank at TP4':70s} {total:>15,} B  {total / GIB:8.3f} GiB")
print(f"budget 94 GiB/rank -> {'FITS' if total < 94 * GIB else 'OVER BUDGET'}, "
      f"{(94 * GIB - total) / GIB:.2f} GiB spare for KV + scratch")
