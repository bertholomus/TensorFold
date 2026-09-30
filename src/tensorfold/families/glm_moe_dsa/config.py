"""GLM-5.3's settings from the checkpoint's config.json, checked against what the engine implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# GLM-5.3 (exllamav3's glm_moe_dsa reference): DeepSeek-V3's MLA body with V3.2-style DSA on top of
# the latent cache, 64 query heads, a 32-head lightning indexer, and sigmoid noaux_tc MoE with 256
# routed experts plus one shared expert. Unlike GLM-5.3-Flash there are no KDA layers, no
# hyper-connections, and no k-pool compression: every layer scores raw token keys, and a "shared"
# indexer layer reuses the nearest preceding "full" layer's selection.


@dataclass(frozen=True)
class Config:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: list[str]          # per layer: "full" (scores and selects) or "shared" (reuses)
    mlp_layer_types: list[str]        # per layer: "dense" or "sparse"
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    intermediate_size: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    scoring_func: str
    topk_method: str
    num_nextn_predict_layers: int
    rope_theta: float
    eos_token_id: tuple[int, ...]

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def dense_layers(self) -> int:
        return sum(k == "dense" for k in self.mlp_layer_types)

    @property
    def full_indexer_layers(self) -> int:
        return sum(k == "full" for k in self.indexer_types)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)
        if int(t.get("n_group", 1)) != 1 or int(t.get("topk_group", 1)) != 1:
            raise ValueError("glm_moe_dsa: grouped expert selection (n_group > 1) is not implemented")
        if str(t.get("scoring_func", "sigmoid")) != "sigmoid" or str(t.get("topk_method", "noaux_tc")) != "noaux_tc":
            raise ValueError("glm_moe_dsa: only sigmoid scores with noaux_tc top-k are implemented")
        rope = t.get("rope_parameters") or {}
        theta = float(rope.get("rope_theta", t.get("rope_theta", 8000000.0)))
        if rope.get("rope_type", "default") not in ("default", None):
            raise ValueError(f"glm_moe_dsa: rope_type {rope.get('rope_type')!r} is not implemented (only plain rope)")
        for key in ("rope_interleave", "indexer_rope_interleave"):
            if t.get(key, True) is not True:
                raise ValueError(f"glm_moe_dsa: only interleaved (GPT-J) rope is implemented ({key} must be true)")
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        indexer = list(t["indexer_types"])
        if len(indexer) < int(t["num_hidden_layers"]) or indexer[0] != "full" \
                or any(k not in ("full", "shared") for k in indexer):
            raise ValueError("glm_moe_dsa: indexer_types must list full/shared per layer, starting with full")
        return cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=int(t["num_hidden_layers"]),
            vocab_size=int(t["vocab_size"]), rms_norm_eps=float(t["rms_norm_eps"]),
            num_attention_heads=int(t["num_attention_heads"]), q_lora_rank=int(t["q_lora_rank"]),
            kv_lora_rank=int(t["kv_lora_rank"]), qk_nope_head_dim=int(t["qk_nope_head_dim"]),
            qk_rope_head_dim=int(t.get("qk_rope_head_dim", 0)), v_head_dim=int(t["v_head_dim"]),
            index_n_heads=int(t["index_n_heads"]), index_head_dim=int(t["index_head_dim"]),
            index_topk=int(t["index_topk"]), indexer_types=indexer,
            mlp_layer_types=list(t.get("mlp_layer_types") or
                                 ["dense" if i < int(t.get("first_k_dense_replace", 3)) else "sparse"
                                  for i in range(int(t["num_hidden_layers"]))]),
            n_routed_experts=int(t["n_routed_experts"]), num_experts_per_tok=int(t["num_experts_per_tok"]),
            moe_intermediate_size=int(t["moe_intermediate_size"]), intermediate_size=int(t["intermediate_size"]),
            n_shared_experts=int(t.get("n_shared_experts") or 0),
            routed_scaling_factor=float(t["routed_scaling_factor"]), norm_topk_prob=bool(t.get("norm_topk_prob", True)),
            scoring_func=str(t.get("scoring_func", "sigmoid")), topk_method=str(t.get("topk_method", "noaux_tc")),
            num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)), rope_theta=theta,
            eos_token_id=tuple(int(e) for e in eos) if isinstance(eos, list)
            else ((int(eos),) if eos is not None else ()))

    def missing_tensors(self, config: dict[str, Any]) -> list[str]:
        """Tensor names the architecture needs that this checkpoint does not list (from its index or folder)."""

        import json
        import math
        from pathlib import Path

        where = Path(config.get("_model_dir") or ".")
        have = set()
        index = where / "model.safetensors.index.json"
        if index.is_file():
            have = set(json.loads(index.read_text())["weight_map"])
        else:
            import struct

            for file in sorted(where.glob("*.safetensors")):
                with file.open("rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    have |= set(json.loads(f.read(n)))
        need = [f"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
        for i in range(self.num_hidden_layers):
            p = f"model.layers.{i}"
            need += [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
                     f"{p}.self_attn.q_a_proj.weight", f"{p}.self_attn.q_a_layernorm.weight",
                     f"{p}.self_attn.q_b_proj.weight", f"{p}.self_attn.kv_a_proj_with_mqa.weight",
                     f"{p}.self_attn.kv_a_layernorm.weight", f"{p}.self_attn.kv_b_proj.weight",
                     f"{p}.self_attn.o_proj.weight"]
            if self.indexer_types[i] == "full":
                need += [f"{p}.self_attn.indexer.wk.weight", f"{p}.self_attn.indexer.k_norm.weight",
                         f"{p}.self_attn.indexer.k_norm.bias", f"{p}.self_attn.indexer.weights_proj.weight",
                         f"{p}.self_attn.indexer.wq_b.weight"]
            if self.mlp_layer_types[i] == "dense":
                need += [f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight", f"{p}.mlp.down_proj.weight"]
            else:
                need += [f"{p}.mlp.gate.weight", f"{p}.mlp.gate.e_score_correction_bias"]
                for e in range(self.n_routed_experts):
                    # EXL3 stores each routed projection as a trellis pack; a non-quantized checkpoint keeps
                    # plain weights: either name satisfies the architecture
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        need.append(f"{p}.mlp.experts.{e}.{proj}.trellis")
                if self.n_shared_experts:
                    need += [f"{p}.mlp.shared_experts.gate_proj.weight", f"{p}.mlp.shared_experts.up_proj.weight",
                             f"{p}.mlp.shared_experts.down_proj.weight"]
        if self.num_nextn_predict_layers:
            p = f"model.layers.{self.num_hidden_layers}"
            need += [f"{p}.enorm.weight", f"{p}.hnorm.weight", f"{p}.eh_proj.weight",
                     f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
                     f"{p}.self_attn.q_a_proj.weight", f"{p}.self_attn.q_b_proj.weight",
                     f"{p}.self_attn.kv_a_proj_with_mqa.weight", f"{p}.self_attn.kv_b_proj.weight",
                     f"{p}.self_attn.o_proj.weight", f"{p}.self_attn.indexer.wq_b.weight",
                     f"{p}.mlp.gate.weight", f"{p}.shared_head.norm.weight"]
        return [n for n in need if not any(alt in have for alt in _forms(n))]


def _forms(name: str) -> tuple[str, ...]:
    """A matrix's accepted names: its plain ``X.weight`` or its EXL3 group's ``X.trellis``, either way round."""

    if name.endswith(".trellis"):
        return (name, name[:-len(".trellis")] + ".weight")
    if name.endswith(".weight") and not name.endswith(("norm.weight", "gate.weight")) and ".norm." not in name:
        return (name, name[:-len(".weight")] + ".trellis")
    return (name,)
