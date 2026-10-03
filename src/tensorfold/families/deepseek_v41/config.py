"""DeepSeek-V4.1 text config, read from config.json (top level or text_config)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Cfg:
    vocab: int
    dim: int
    inter: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_dim: int
    q_rank: int
    o_rank: int
    o_groups: int
    window: int
    eps: float
    n_routed: int
    topk: int
    route_scale: float
    swiglu_limit: float
    compress_ratios: list[int]
    kv_sources: list[int]
    index_sources: list[int]
    idx_heads: int
    idx_dim: int
    idx_topk: int
    cand_source: int
    cand_blocks: int
    cand_block: int
    hc: int
    hc_iters: int
    hc_eps: float
    rope_theta: float
    compress_theta: float
    rope_factor: float
    orig_len: int
    beta_fast: float
    beta_slow: float
    engram_layers: list[int]
    engram_rows: list[int]
    engram_ngram: int
    engram_vocab: int
    engram_heads: int
    engram_dim: int
    engram_pad: int
    engram_cvocab: int
    dspark_block: int = 0
    dspark_noise: int = 0
    dspark_taps: list[int] = field(default_factory=list)
    dspark_rank: int = 0
    dspark_routed: int = 0
    dspark_topk: int = 0

    @classmethod
    def read(cls, model_dir: str | Path) -> "Cfg":
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> "Cfg":
        t = raw.get("text_config", raw)
        rs = t.get("rope_scaling") or {}
        return cls(
            vocab=t["vocab_size"], dim=t["hidden_size"], inter=t["moe_intermediate_size"],
            n_layers=t["num_hidden_layers"], n_heads=t["num_attention_heads"], head_dim=t["head_dim"],
            rope_dim=t["qk_rope_head_dim"], q_rank=t["q_lora_rank"], o_rank=t["o_lora_rank"], o_groups=t["o_groups"],
            window=t["sliding_window"], eps=t["rms_norm_eps"], n_routed=t["n_routed_experts"],
            topk=t["num_experts_per_tok"], route_scale=t["routed_scaling_factor"], swiglu_limit=t["swiglu_limit"],
            compress_ratios=list(t["compress_ratios"]), kv_sources=list(t["kv_source_layer_ids"]),
            index_sources=list(t["index_source_layer_ids"]), idx_heads=t["index_n_heads"],
            idx_dim=t["index_head_dim"], idx_topk=t["index_topk"], cand_source=t.get("candidate_source_layer_id", -1),
            cand_blocks=t.get("candidate_topk_blocks", 0), cand_block=t.get("candidate_block_size", 0),
            hc=t["hc_mult"], hc_iters=t["hc_sinkhorn_iters"], hc_eps=t["hc_eps"], rope_theta=t["rope_theta"],
            compress_theta=t["compress_rope_theta"], rope_factor=rs.get("factor", 1.0),
            orig_len=rs.get("original_max_position_embeddings", 0), beta_fast=rs.get("beta_fast", 32),
            beta_slow=rs.get("beta_slow", 1), engram_layers=list(t.get("engram_layer_ids", [])),
            engram_rows=list(t.get("engram_num_embeddings", [])), engram_ngram=t.get("engram_max_ngram_size", 1),
            engram_vocab=t.get("engram_vocab_size", 0), engram_heads=t.get("engram_n_heads", 0),
            engram_dim=t.get("engram_head_dim", 0), engram_pad=t.get("engram_pad_token_id", 2),
            engram_cvocab=t.get("engram_compressed_vocab_size", 0), dspark_block=t.get("dspark_block_size", 0),
            dspark_noise=t.get("dspark_noise_token_id", 0), dspark_taps=list(t.get("dspark_target_layer_ids", [])),
            dspark_rank=t.get("dspark_markov_rank", 0), dspark_routed=t.get("dspark_n_routed_experts", 0),
            dspark_topk=t.get("dspark_num_experts_per_tok", 0))
