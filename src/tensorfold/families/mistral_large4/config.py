"""Mistral-Large-4 text config (``text_config.model_type == "mistral_large4_text"``), read from config.json.

The asserts mirror exllamav3 branch ml4-loader's ``MistralLarge4Config`` (architecture/mistral_large4.py:84-153,
read 2026-11-05 @ b6e82d1a) per notes D1 §12 and D3 §1: no q-LoRA, the four per-layer schedules with their length
and content checks, the KV-sharing source rule, plain softmax routing (no group masking, scale 1.0), silu, and the
YaRN 2048-over-512 table the LeRoPE alpha multiplies at load. A config that passes this ``Cfg`` is one the pack
tensor plan (D3 §2, cuda/weights.py) is defined for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: the pack's one vocabulary (D3 §1: a partial pack shares it with the full pack)
VOCAB = 204800


def kv_sharing_schedule(kv_source_layers: list[int], no_rope_layers: list[int], num_layers: int) -> set[int]:
    """Validate the KV-sharing layout against the reference semantics and return the set of layers that publish
    their latent rows to later readers. Port of exllamav3 ``mistral_large4.py:_kv_sharing_schedule``: a reader
    takes the rows last stored under its own rope type, and only layers named as a source by a later layer store,
    so a reader's named source must be the latest storing layer of the same rope type before it."""

    sources: set[int] = set()
    for idx in range(num_layers):
        src = kv_source_layers[idx]
        assert src <= idx and kv_source_layers[src] == src, \
            f"Layer {idx} must share KV with an earlier layer that owns its cache, got {src}"
        if src != idx:
            sources.add(src)
    for idx in range(num_layers):
        src = kv_source_layers[idx]
        if src == idx:
            continue
        assert no_rope_layers[src] == no_rope_layers[idx], \
            f"KV-sharing layer {idx} and its source {src} differ in rope type"
        later = [s for s in sources if src < s < idx and no_rope_layers[s] == no_rope_layers[idx]]
        assert not later, \
            f"KV-sharing layer {idx} names source {src}, but the reference would read layer {max(later)}"
    return sources


@dataclass
class Cfg:
    """One Mistral-Large-4 text stack. Field names follow the pack's HF names; per-layer lookups use the
    ``is_rope`` / ``is_reader`` / ``is_full_indexer`` / ``is_dense`` helpers below (D3 §2)."""

    vocab: int
    dim: int                      # hidden_size 6144
    n_layers: int
    n_heads: int                  # 64
    kv_lora: int                  # kv_lora_rank 512
    nope: int                     # qk_nope_head_dim 64
    rd: int                       # qk_rope_head_dim 64 (0-width rope slice on NoPE layers)
    v_head: int                   # v_head_dim 128
    qk_head: int                  # nope + rd, the whole 128-dim head (exllamav3 calls it qk_head_dim)
    inter: int                    # moe_intermediate_size 6144 (routed + shared experts)
    dense_inter: int              # intermediate_size 24576 (dense L0)
    n_routed: int                 # 128
    topk: int                     # 4
    n_shared: int                 # 1
    route_scale: float            # asserted 1.0 (plain softmax + renorm, no scale)
    eps: float                    # rms_norm_eps
    index_norm_eps: float         # the indexer's biased LayerNorm, 1e-6 in the reference
    idx_heads: int                # index_n_heads 32
    idx_dim: int                  # index_head_dim 128
    idx_topk: int                 # 2048
    softcap: float                # final_logit_softcapping (30 on the real pack)
    sm_scale: float               # qk_head ** -0.5, no YaRN mscale fold
    rope_flags: list[int]         # config's `no_rope_layers`: 1 = rotates, 0 = NoPE (SmolLM3 convention)
    kv_source_layers: list[int]   # layer -> its KV source layer (self = owns its cache)
    kv_sources: set[int]          # writers named by a later reader (publish rows; D1 §5)
    indexer_types: list[str]      # "full" | "shared"
    mlp_layer_types: list[str]    # "dense" | "sparse"
    rope_theta: float
    rope_type: str                # "yarn" (asserted)
    rope_factor: float            # 2048.0 (asserted; D1 §7)
    orig_len: int                 # original_max_position_embeddings, 512 (asserted)
    attention_factor: float       # 1.0: YaRN leaves cos/sin unscaled; LeRoPE alpha folds at load
    tie_word_embeddings: bool
    hidden_act: str               # "silu" (asserted)
    q_rank: None = None           # no q-LoRA (asserted None)

    @classmethod
    def read(cls, model_dir: str | Path) -> "Cfg":
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> "Cfg":
        t = raw.get("text_config", raw)
        q_lora_rank = t.get("q_lora_rank")
        assert q_lora_rank is None, \
            f"Mistral-Large-4 projects queries directly (q_proj), got q_lora_rank = {q_lora_rank}"
        kv_lora, nope, rd, v_head = (t["kv_lora_rank"], t["qk_nope_head_dim"], t["qk_rope_head_dim"],
                                     t["v_head_dim"])
        vocab = int(t["vocab_size"])
        assert vocab == VOCAB, f"ML4 vocab is {VOCAB}, got {vocab} (D3 §1: even partial packs keep it)"

        n = int(t["num_hidden_layers"])
        interval = int(t.get("no_rope_layer_interval", 4))
        rope_flags = [int(v) for v in (t.get("no_rope_layers")
                                       or [int((i + 1) % interval != 0) for i in range(n)])]
        kv_src = [int(v) for v in (t.get("kv_source_layers") or list(range(n)))]
        first_k_dense = int(t.get("first_k_dense_replace", 1))
        mlp_types = list(t.get("mlp_layer_types")
                         or ["dense"] * min(first_k_dense, n) + ["sparse"] * max(n - first_k_dense, 0))
        idx_types = list(t["indexer_types"])          # no default in the reference: KeyError is the error
        for name, sched in (("no_rope_layers", rope_flags), ("kv_source_layers", kv_src),
                            ("mlp_layer_types", mlp_types), ("indexer_types", idx_types)):
            assert len(sched) >= n, f"{name} has fewer than {n} entries"
        assert all(v in (0, 1) for v in rope_flags[:n]), f"no_rope_layers must be 0/1 flags, got {rope_flags[:n]}"
        assert all(s in ("full", "shared") for s in idx_types[:n]) and idx_types[0] == "full", \
            f"Unexpected indexer_types: {idx_types}"
        assert all(s in ("dense", "sparse") for s in mlp_types[:n]), f"Unexpected mlp_layer_types: {mlp_types}"
        layer_types = t.get("layer_types")
        assert layer_types is None or all(s == "indexed_attention" for s in layer_types[:n]), \
            f"Unexpected layer_types: {layer_types}"
        sources = kv_sharing_schedule(kv_src, rope_flags, n)

        assert str(t.get("hidden_act", "silu")) == "silu", f"hidden_act must be silu, got {t.get('hidden_act')}"
        n_group, topk_group = t.get("n_group", 1), t.get("topk_group", 1)
        assert n_group in (None, 1) and topk_group in (None, 1), \
            f"Group-limited expert routing (n_group = {n_group}, topk_group = {topk_group}) is not supported"
        route_scale = float(t.get("routed_scaling_factor", 1.0))
        assert route_scale == 1.0, \
            f"routed_scaling_factor {route_scale} != 1.0 is not supported with softmax routing"
        norm_type = t.get("indexer_norm_type")
        assert norm_type is None or str(norm_type).lower() in ("layernorm", "layer_norm"), \
            f"the indexer norms are a biased LayerNorm (q_input_norm), got {norm_type}"

        rs = t.get("rope_scaling") or t.get("rope_parameters") or {}
        rope_type = str(rs.get("rope_type") or rs.get("type") or "yarn")
        rope_factor = float(rs.get("factor", 2048))
        orig_len = int(rs.get("original_max_position_embeddings", 512))
        attention_factor = float(rs.get("attention_factor", 1.0))
        # D1 §7 / D3 §1: the pack's YaRN table is 2048x over a 512-token window with attention_factor 1.0; the
        # LeRoPE alpha (globals) multiplies these inverse frequencies at load, never a mscale fold into sm_scale.
        assert rope_type in ("yarn", "yarn_default"), f"ML4 rotates with YaRN, got rope type {rope_type!r}"
        assert rope_factor == 2048.0 and orig_len == 512, \
            f"ML4's YaRN table is factor 2048 over 512 positions, got {rope_factor} / {orig_len}"
        assert attention_factor == 1.0, f"YaRN attention_factor must be 1.0 (no mscale fold), got {attention_factor}"

        softcap = float(t.get("final_logit_softcapping", 0.0) or 0.0)
        assert softcap >= 0.0, f"final_logit_softcapping must be >= 0 (0 = off), got {softcap}"

        return cls(
            vocab=vocab, dim=int(t["hidden_size"]), n_layers=n, n_heads=int(t["num_attention_heads"]),
            kv_lora=kv_lora, nope=nope, rd=rd, v_head=v_head, qk_head=nope + rd,
            inter=int(t["moe_intermediate_size"]), dense_inter=int(t["intermediate_size"]),
            n_routed=int(t["n_routed_experts"]), topk=int(t["num_experts_per_tok"]),
            n_shared=int(t.get("n_shared_experts", 1)), route_scale=route_scale,
            eps=float(t["rms_norm_eps"]), index_norm_eps=1e-6,
            idx_heads=int(t["index_n_heads"]), idx_dim=int(t["index_head_dim"]), idx_topk=int(t["index_topk"]),
            softcap=softcap, sm_scale=float((nope + rd) ** -0.5),
            rope_flags=rope_flags[:n], kv_source_layers=kv_src[:n], kv_sources=sources,
            indexer_types=idx_types[:n], mlp_layer_types=mlp_types[:n],
            rope_theta=float(rs.get("rope_theta", t.get("rope_theta", 10000.0))), rope_type=rope_type,
            rope_factor=rope_factor, orig_len=orig_len, attention_factor=attention_factor,
            tie_word_embeddings=bool(t.get("tie_word_embeddings", False)), hidden_act="silu")

    # -- per-layer lookups (D3 §2: what a layer holds is decided only by these config lists) ------------------

    def is_rope(self, i: int) -> bool:
        """True when layer i rotates (the config list is named ``no_rope_layers`` but carries rope-enable flags)."""

        return bool(self.rope_flags[i])

    def is_reader(self, i: int) -> bool:
        """True when layer i has no kv_a / kv_a_layernorm and reads its source layer's latent rows."""

        return self.kv_source_layers[i] != i

    def is_full_indexer(self, i: int) -> bool:
        return self.indexer_types[i] == "full"

    def is_dense(self, i: int) -> bool:
        return self.mlp_layer_types[i] == "dense"

    @property
    def q_out(self) -> int:
        """q_proj output width: every head's whole qk_head (rope and NoPE alike)."""

        return self.n_heads * self.qk_head

    @property
    def o_in(self) -> int:
        """o_proj input width: the attention output, every head's v_head."""

        return self.n_heads * self.v_head

    def kv_a_rows(self, i: int) -> int:
        """kv_a_proj_with_mqa rows: [latent | rope key] on rope layers, latent only on NoPE layers."""

        return self.kv_lora + (self.rd if self.is_rope(i) else 0)

    def kv_b_rows(self, i: int) -> int:
        """kv_b_proj rows: per head [W_UK slice | W_UV slice]; the UK slice is the nope part on rope layers and
        the whole head on NoPE layers (exllamav3 MLAAttention.load_local)."""

        return self.n_heads * ((self.nope if self.is_rope(i) else self.qk_head) + self.v_head)
