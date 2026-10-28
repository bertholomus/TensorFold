"""Beam-1.0's architecture from config.json: the fields the CUDA engine reads, refused where it states a switch the
engine does not implement (Reflection's BeamConfig defaults fill what the file leaves out)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# BeamConfig's defaults for switches Beam-1.0's config.json leaves out, and the value the engine implements
IMPLEMENTED = {
    "ssmax": True, "ssmax_offset": 1, "ssmax_divide_by_log_offset": False, "ssmax_scale_mode": "root10",
    "use_paramfree_qk_norm": False, "attention_bias": False, "attention_output_gate": True,
    "attention_output_gate_bias": True, "attention_output_gate_floor": 0.0, "attention_output_gate_temperature": 1.0,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True, "enable_sandwich_norm": True,
    "self_attn_depth_scaled_residual": True, "mlp_depth_scaled_residual": True, "tie_word_embeddings": False,
    "hidden_act": "silu",
}


@dataclass(frozen=True)
class Config:
    vocab: int
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    eps: float
    window: int                   # keys a window layer's row sees, itself included
    full: tuple[int, ...]         # the full-attention layers (NoPE, SSMax-scaled queries)
    dense: tuple[int, ...]        # the dense SwiGLU layers (the rest are routed)
    intermediate: int             # the dense layers' width
    moe_width: int                # an expert's width
    experts: int
    top_k: int
    groups: int
    top_groups: int
    routed_scale: float
    softcap: float
    rope_theta: float
    max_positions: int
    eos: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], generation: dict[str, Any] | None = None) -> "Config":
        for key, want in IMPLEMENTED.items():
            got = raw.get(key, want)
            if got != want:
                raise ValueError(f"Beam's CUDA engine implements {key}={want!r}; this checkpoint states {got!r}")
        layers = int(raw["num_hidden_layers"])
        types = list(raw.get("layer_types") or [])
        if len(types) != layers or set(types) - {"sliding_attention", "full_attention"}:
            raise ValueError("layer_types must name sliding_attention or full_attention for every layer")
        full = tuple(i for i, t in enumerate(types) if t == "full_attention")
        nope = raw.get("nope_layer_ids")
        if nope is not None and tuple(sorted(nope)) != full:
            raise ValueError("the engine implements NoPE on exactly the full-attention layers")
        rope = raw.get("rope_parameters") or {}
        slide = rope.get("sliding_attention") or {"rope_type": "default", "rope_theta": raw.get("rope_theta", 1e4)}
        if slide.get("rope_type", "default") != "default" or rope.get("full_attention") is not None:
            raise ValueError(f"the engine implements plain RoPE on window layers and none on full ones, not {rope}")
        mlp = list(raw.get("mlp_layer_types") or [])
        if len(mlp) != layers or set(mlp) - {"dense", "sparse"}:
            raise ValueError("mlp_layer_types must name dense or sparse for every layer")
        eos = (generation or {}).get("eos_token_id", raw.get("eos_token_id"))
        eos = tuple(eos) if isinstance(eos, list) else (eos,) if eos is not None else ()
        cfg = cls(vocab=int(raw["vocab_size"]), hidden=int(raw["hidden_size"]), layers=layers,
                  heads=int(raw["num_attention_heads"]), kv_heads=int(raw["num_key_value_heads"]),
                  head_dim=int(raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]),
                  eps=float(raw["rms_norm_eps"]), window=int(raw["sliding_window"]), full=full,
                  dense=tuple(i for i, t in enumerate(mlp) if t == "dense"),
                  intermediate=int(raw["intermediate_size"]), moe_width=int(raw["moe_intermediate_size"]),
                  experts=int(raw["num_experts"]), top_k=int(raw["num_experts_per_tok"]), groups=int(raw["n_group"]),
                  top_groups=int(raw["topk_group"]), routed_scale=float(raw.get("routed_scaling_factor") or 1.0),
                  softcap=float(raw.get("final_logit_softcapping") or 0.0), rope_theta=float(slide["rope_theta"]),
                  max_positions=int(raw["max_position_embeddings"]), eos=eos)
        if cfg.head_dim != 128 or cfg.heads % cfg.kv_heads:
            raise ValueError("the engine's attention kernels take 128-wide heads in whole query groups")
        if cfg.experts % cfg.groups:
            raise ValueError("experts must split into whole routing groups")
        return cfg

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        root = Path(model_dir)
        gen = root / "generation_config.json"
        return cls.from_dict(json.loads((root / "config.json").read_text()),
                             json.loads(gen.read_text()) if gen.exists() else None)

    def sliding(self, layer: int) -> bool:
        return layer not in self.full

    def routed(self, layer: int) -> bool:
        return layer not in self.dense
