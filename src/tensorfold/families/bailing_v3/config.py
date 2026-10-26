"""Ling-3.1-flash's settings (model_type ``bailing_hybrid``, architecture BailingMoeV3ForCausalLM) from the
checkpoint's config.json, checked against what the engine implements."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

# Ling-3.1-flash (the producer's modeling_bailing_moe_v3.py): Kimi Delta Attention layers with full-rank
# decay (f_proj) and output-gate (g_proj) projections, a 4-tap short convolution and fp32 beta, interleaved
# with RoPE MLA layers every layer_group_size (7, 15, ..., 71), a sigmoid noaux_tc MoE of 512 routed experts
# in 8 selection groups of 64 plus one shared expert, dense SwiGLU MLPs for the first four layers, and one
# MTP layer (index 72) whose MLA block and MoE draft through the shared head. The last layers' expert SwiGLUs
# carry a clamp the serving stacks apply and the reference file ignores: min(silu(g), L) times
# clip(u, -L, L), L = 10 on the routed experts of layers 65-71 and the shared expert of 64-71, never the MTP layer.


def _bf16_value(v: float) -> bool:
    """v is a bf16 number (so clamping a bf16 tensor at v rounds nothing, in any stack's arithmetic)."""

    try:
        bits = struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError:
        return False
    return bits & 0xFFFF == 0 and struct.unpack("<f", struct.pack("<I", bits))[0] == v


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
    lin_heads: int                  # the KDA heads (num_kv_heads_for_linear_attn 0: every attention head)
    lin_dim: int
    conv_kernel: int
    kda_lower_bound: float
    n_group: int
    topk_group: int
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    intermediate_size: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    rope_theta: float
    layer_kinds: list[str]          # per layer: "mla" or "kda" (modeling's attention_layer_type formula)
    mlp_layer_types: list[str]      # per layer: "dense" or "sparse"
    eos_token_id: tuple[int, ...]
    num_nextn_predict_layers: int
    shared_intermediate_size: int = 0   # moe_shared_expert_intermediate_size (0: moe_intermediate_size)
    # expert_swiglu_limit_list / share_expert_swiglu_limit_list per decoder layer, inf where the list says 0
    expert_swiglu_limits: tuple[float, ...] = ()
    shared_swiglu_limits: tuple[float, ...] = ()

    def swiglu_limit(self, layer: int, shared: bool = False) -> float:
        """The clamp L of a layer's routed (or shared) expert SwiGLU: inf where the list has 0 or no entry (the MTP
        layer, index num_hidden_layers, is past every list: from_dict refuses longer ones)."""

        limits = self.shared_swiglu_limits if shared else self.expert_swiglu_limits
        return limits[layer] if 0 <= layer < len(limits) else math.inf

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_b_per_head(self) -> int:
        """kv_b's rows a head: qk_nope key rows then v_head_dim value rows (modeling's key_shape split)."""
        return self.qk_nope_head_dim + self.v_head_dim

    @property
    def dense_layers(self) -> int:
        return sum(k == "dense" for k in self.mlp_layer_types)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)

        def refuse(why: str) -> None:
            raise ValueError(f"bailing_v3: {why} is not implemented (the engine reads the reference's own "
                             "switches only)")

        # the switches the reference implements and this engine does not
        if t.get("rope_scaling") is not None:
            refuse(f"rope_scaling {t['rope_scaling']!r} (rope runs at factor 1.0 on the 64-dim rope slice)")
        if t.get("rope_interleave", True) is not True:
            refuse("rope_interleave false (only interleaved (GPT-J) rope)")
        if t.get("no_kda_lora") is not True or t.get("use_kda_lora", False):
            refuse("no_kda_lora false / use_kda_lora true (the KDA f and g projections are full rank)")
        if t.get("gated_attention_proj_granularity_type") != "head_wise":
            refuse(f"gated_attention_proj_granularity_type {t.get('gated_attention_proj_granularity_type')!r} "
                   "(only the head-wise gate)")
        if t.get("mtp_use_kda", False):
            refuse("mtp_use_kda true (the MTP layer attends with MLA)")
        if int(t.get("num_kv_heads_for_linear_attn", 0)) != 0:
            refuse(f"num_kv_heads_for_linear_attn {t.get('num_kv_heads_for_linear_attn')!r} (0: every head is "
                   "a KDA head)")
        heads = int(t["num_attention_heads"])
        if int(t.get("num_key_value_heads", heads)) != heads:
            refuse(f"num_key_value_heads {t.get('num_key_value_heads')!r} != num_attention_heads {heads}")
        for key in ("use_qkv_bias", "use_bias"):
            if t.get(key, False):
                refuse(f"{key} true (the projections carry no bias)")
        for key in ("score_function", "scoring_func"):
            if key in t and str(t[key]) != "sigmoid":
                refuse(f"{key} {t[key]!r} (only sigmoid expert scores)")
        if str(t.get("topk_method", "noaux_tc")) != "noaux_tc":
            refuse(f"topk_method {t.get('topk_method')!r} (only noaux_tc group-limited top-k)")
        for key in ("value_norm", "up_proj_norm", "use_nGPT"):
            if t.get(key, False):
                refuse(f"{key} true")

        layers = int(t["num_hidden_layers"])
        group = int(t.get("layer_group_size", 8))
        # modeling 1004-1009: MLA when the layer closes a group or sits past the last full group
        kinds = ["mla" if (i + 1) % group == 0 or i >= layers // group * group else "kda"
                 for i in range(layers)]
        dense = int(t.get("first_k_dense_replace", 0))
        mlp = ["dense" if i < dense else "sparse" for i in range(layers)]

        def limits(key: str) -> tuple[float, ...]:
            """A per-layer SwiGLU clamp list: 0 (or None) is no clamp (inf); the stacks clamp the experts only."""

            raw = t.get(key)
            if raw is None:
                return ()
            if not isinstance(raw, (list, tuple)):
                refuse(f"{key} {raw!r} (a list of per-layer limits)")
            if len(raw) > layers:
                refuse(f"{key} with {len(raw)} entries for {layers} layers (an entry for the MTP layer, which "
                       "the stacks never clamp)")
            out = []
            for i, v in enumerate(raw):
                v = 0.0 if v is None else float(v)
                if not v:
                    out.append(math.inf)
                    continue
                if not (v > 0 and _bf16_value(v)):
                    refuse(f"{key}[{i}] = {v!r} (a limit that is not a positive bf16 value)")
                if mlp[i] == "dense":
                    refuse(f"{key}[{i}] = {v!r} on dense MLP layer {i} (only the experts' SwiGLU is clamped)")
                out.append(v)
            return tuple(out)
        eos = t.get("eos_token_id", config.get("eos_token_id"))
        return cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=layers, vocab_size=int(t["vocab_size"]),
            rms_norm_eps=float(t["rms_norm_eps"]), num_attention_heads=heads,
            q_lora_rank=int(t["q_lora_rank"]), kv_lora_rank=int(t["kv_lora_rank"]),
            qk_nope_head_dim=int(t["qk_nope_head_dim"]), qk_rope_head_dim=int(t["qk_rope_head_dim"]),
            v_head_dim=int(t["v_head_dim"]),
            lin_heads=int(t.get("num_kv_heads_for_linear_attn") or heads),
            lin_dim=int(t.get("head_dim", t["qk_nope_head_dim"])),
            conv_kernel=int(t.get("short_conv_kernel_size", 4)),
            kda_lower_bound=float(t.get("kda_lower_bound", -5.0)),
            n_group=int(t["n_group"]), topk_group=int(t["topk_group"]),
            n_routed_experts=int(t["num_experts"]), num_experts_per_tok=int(t["num_experts_per_tok"]),
            moe_intermediate_size=int(t["moe_intermediate_size"]), intermediate_size=int(t["intermediate_size"]),
            n_shared_experts=int(t.get("num_shared_experts") or 0),
            routed_scaling_factor=float(t["routed_scaling_factor"]),
            norm_topk_prob=bool(t.get("norm_topk_prob", True)),
            rope_theta=float(t.get("rope_theta", 6_000_000.0)),
            layer_kinds=kinds, mlp_layer_types=mlp,
            eos_token_id=tuple(int(e) for e in eos) if isinstance(eos, list) else ((int(eos),) if eos is not None else ()),
            num_nextn_predict_layers=int(t.get("num_nextn_predict_layers", 0)),
            shared_intermediate_size=int(t.get("moe_shared_expert_intermediate_size") or 0),
            expert_swiglu_limits=limits("expert_swiglu_limit_list"),
            shared_swiglu_limits=limits("share_expert_swiglu_limit_list"),
        )

    def missing_tensors(self, config: dict[str, Any]) -> list[str]:
        """Tensor names the architecture needs that this checkpoint does not list (from its index or folder)."""

        import json
        import struct
        from pathlib import Path

        where = Path(config.get("_model_dir") or ".")
        have = set()
        index = where / "model.safetensors.index.json"
        if index.is_file():
            have = set(json.loads(index.read_text())["weight_map"])
        else:
            for file in sorted(where.glob("*.safetensors")):
                with file.open("rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    have |= set(json.loads(f.read(n)))
        need = ["model.word_embeddings.weight", "lm_head.weight", "model.norm.weight"]
        for i in range(self.num_hidden_layers):
            need += self.layer_tensors(i)
        if self.num_nextn_predict_layers:
            need += [f"model.layers.{self.num_hidden_layers}.enorm.weight",
                     f"model.layers.{self.num_hidden_layers}.hnorm.weight",
                     f"model.layers.{self.num_hidden_layers}.eh_proj.weight",
                     f"model.layers.{self.num_hidden_layers}.final_layernorm.weight"]
            need += self.layer_tensors(self.num_hidden_layers, kind="mla", mlp="moe")
        return [n for n in need if not any(alt in have for alt in _forms(n))]

    def layer_tensors(self, i: int, kind: str | None = None, mlp: str | None = None) -> list[str]:
        """Layer ``i``'s tensor names; the MTP layer passes its own kind ("mla") and MLP ("moe")."""

        p = f"model.layers.{i}"
        kind = kind or (self.layer_kinds[i] if i < len(self.layer_kinds) else "mla")
        mlp = mlp or (self.mlp_layer_types[i] if i < len(self.mlp_layer_types) else "moe")
        need = [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"]
        if kind == "mla":
            need += [f"{p}.attention.q_a_proj.weight", f"{p}.attention.q_a_layernorm.weight",
                     f"{p}.attention.q_b_proj.weight", f"{p}.attention.kv_a_proj_with_mqa.weight",
                     f"{p}.attention.kv_a_layernorm.weight", f"{p}.attention.kv_b_proj.weight",
                     f"{p}.attention.g_proj.weight", f"{p}.attention.dense.weight"]
        else:
            need += [f"{p}.attention.{x}_proj.weight" for x in ("q", "k", "v", "f", "g")]
            need.append(f"{p}.attention.b_proj.weight")
            need += [f"{p}.attention.{x}_conv1d.weight" for x in "qkv"]
            need += [f"{p}.attention.A_log", f"{p}.attention.dt_bias", f"{p}.attention.o_norm.weight",
                     f"{p}.attention.o_proj.weight"]
        if mlp == "dense":
            need += [f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight", f"{p}.mlp.down_proj.weight"]
        else:
            need += [f"{p}.mlp.gate.weight", f"{p}.mlp.gate.expert_bias"]
            for e in range(self.n_routed_experts):
                # EXL3 stores each routed projection as a trellis pack; a non-quantized checkpoint keeps
                # plain weights: either name satisfies the architecture
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    need.append(f"{p}.mlp.experts.{e}.{proj}.trellis")
            if self.n_shared_experts:
                need += [f"{p}.mlp.shared_experts.gate_proj.weight", f"{p}.mlp.shared_experts.up_proj.weight",
                         f"{p}.mlp.shared_experts.down_proj.weight"]
        return need


def _forms(name: str) -> tuple[str, ...]:
    """A matrix's accepted names: its plain ``X.weight`` or its EXL3 group's ``X.trellis``, either way round; a KDA
    q / k / v projection or short conv also as the conversion's fused one (exllamav3 writes ``qkv_proj`` and
    ``conv1d``)."""

    head, sep, tail = name.rpartition(".attention.")
    if sep and tail in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
        return (name, name[:-len(".weight")] + ".trellis", head + sep + "qkv_proj.weight",
                head + sep + "qkv_proj.trellis")
    if sep and tail in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
        return (name, head + sep + "conv1d.weight")
    if name.endswith(".trellis"):
        return (name, name[:-len(".trellis")] + ".weight")
    if name.endswith(".weight") and not name.endswith(("norm.weight", "gate.weight")) and ".norm." not in name:
        return (name, name[:-len(".weight")] + ".trellis")
    return (name,)
