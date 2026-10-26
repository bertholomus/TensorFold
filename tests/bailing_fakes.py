"""A tiny BailingMoeV3-shaped checkpoint writer for the family's CPU tests: EXL3 trellis groups wherever the
conversion quantizes (routed and shared experts, the KDA and MLA projections, eh_proj, the head),
BF16/F32 elsewhere (kv_b, b_proj, the conv weights, A_log, dt_bias, the norms, the router)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# the real architecture at 1/64 scale with every projection a whole 128-column-block group: layers 0-1 KDA,
# layer 2 MLA (layer_group_size 3 closes the group there), the MTP layer 3 MLA + MoE; layer 0 dense MLP; the
# SwiGLU clamp where the real one sits relative to the end: the shared expert from layer 1, the routed
# experts on layer 2, the MTP layer never
CFG = {
    "architectures": ["BailingMoeV3ForCausalLM"],
    "model_type": "bailing_hybrid", "torch_dtype": "bfloat16",
    "hidden_size": 128, "num_hidden_layers": 3, "vocab_size": 256,
    "rms_norm_eps": 1e-6, "num_attention_heads": 8, "num_key_value_heads": 8,
    "num_kv_heads_for_linear_attn": 0, "head_dim": 128,
    "q_lora_rank": 128, "kv_lora_rank": 128, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64,
    "v_head_dim": 64, "qk_head_dim": 128,
    "layer_group_size": 3, "first_k_dense_replace": 1,
    "num_experts": 4, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
    "moe_intermediate_size": 256, "moe_shared_expert_intermediate_size": 256, "num_shared_experts": 1,
    "intermediate_size": 256, "routed_scaling_factor": 2.5, "norm_topk_prob": True,
    "score_function": "sigmoid", "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    "no_kda_lora": True, "use_kda_lora": False, "kda_lower_bound": -5.0, "kda_safe_gate": True,
    "short_conv_kernel_size": 4, "rope_interleave": True, "rope_scaling": None, "rope_theta": 6000000,
    "mtp_use_kda": False, "num_nextn_predict_layers": 1,
    "gated_attention_proj_granularity_type": "head_wise",
    "use_qkv_bias": False, "use_bias": False, "value_norm": False, "up_proj_norm": False, "use_nGPT": False,
    "max_position_embeddings": 4096,
    "eos_token_id": [5], "tie_word_embeddings": False,
    "expert_swiglu_limit_list": [0, 0, 4.0], "share_expert_swiglu_limit_list": [0, 4.0, 4.0],
}
QUANT = {"quant_method": "exl3", "bits": 4, "codebook": "mul1"}
# the same tiny model with TP4's uneven split in it: 1280-wide routed and shared experts (10 Hadamard blocks:
# 384/384/256/256 a rank at TP4, 640/640 at TP2, like Ling's), a 512-wide dense MLP and vocabulary (one block a rank
# at TP4); 8 KDA / MLA heads are 2 a rank
CFG_TP4 = {**CFG, "vocab_size": 512, "intermediate_size": 512, "moe_intermediate_size": 1280,
           "moe_shared_expert_intermediate_size": 1280}


def _bf16(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.bfloat16).view(torch.uint16).numpy().astype(np.uint16)


def _group(outs: int, ins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A random 4-bit EXL3 pack: trellis int16 [ins/16, outs/16, 64] (K-tiles then N-tiles), suh [ins], svh [outs]."""

    kt, nt = ins // 16, outs // 16
    trellis = torch.randint(-32768, 32767, (kt, nt, 64), dtype=torch.int16).numpy()
    suh = _bf16(torch.randn(ins))
    svh = _bf16(torch.randn(outs))
    return trellis, suh, svh


def _group_padded(outs: int, ins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same pack with the output padded to a whole 128-column block (kv_a's latent + rope slice)."""

    padded = -(-outs // 128) * 128
    trellis, suh, svh = _group(padded, ins)
    return trellis, suh, svh


def write_checkpoint(folder: Path, kv_b_trellis: bool = False, cfg: dict | None = None,
                     fused_qkv: bool = False) -> Path:
    """The tiny checkpoint as one safetensors file plus its config; tensor names in the reference layout.

    ``kv_b_trellis``: store the MLA layers' kv_b as an EXL3 group instead of the plain BF16 rows — the
    loader refuses that form (the absorb kernels read the rows per head), so tests can pin the refusal.
    ``cfg``: another config (CFG_TP4). ``fused_qkv``: the KDA layers in the EXL3 conversion's layout, q | k | v as
    one qkv_proj group and the three short convs as one conv1d (exllamav3's GatedDeltaNet writes them so).
    """

    CFG = cfg or globals()["CFG"]

    import struct

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, list[int], np.ndarray]] = {}

    def put(name: str, dtype: str, shape: list[int], data: np.ndarray) -> None:
        tensors[name] = (dtype, shape, data)

    def bf16(name: str, *shape: int) -> None:
        put(name, "BF16", list(shape), _bf16(torch.randn(*shape) * 0.05))

    def group(name: str, outs: int, ins: int, padded: bool = False) -> None:
        tr, suh, svh = _group_padded(outs, ins) if padded else _group(outs, ins)
        put(f"{name}.trellis", "I16", list(tr.shape), tr.astype(np.int16))
        put(f"{name}.suh", "BF16", [ins], suh)
        put(f"{name}.svh", "BF16", [svh.shape[0]], svh)
        put(f"{name}.mul1", "F32", [], np.zeros((), dtype=np.float32))

    d, v = CFG["hidden_size"], CFG["vocab_size"]
    H, nope, rope, vd, kv, ql = (CFG["num_attention_heads"], CFG["qk_nope_head_dim"], CFG["qk_rope_head_dim"],
                                 CFG["v_head_dim"], CFG["kv_lora_rank"], CFG["q_lora_rank"])
    ld = CFG["head_dim"]                              # the KDA head dim (lin_dim)
    bf16("model.word_embeddings.weight", v, d)
    group("lm_head", v, d)
    bf16("model.norm.weight", d)
    layers = CFG["num_hidden_layers"]
    for i in range(layers + 1):                       # the MTP layer last
        p = f"model.layers.{i}"
        bf16(f"{p}.input_layernorm.weight", d)
        bf16(f"{p}.post_attention_layernorm.weight", d)
        mla = (i + 1) % CFG["layer_group_size"] == 0 or i >= layers // CFG["layer_group_size"] * CFG["layer_group_size"]
        if mla:
            group(f"{p}.attention.q_a_proj", ql, d)
            bf16(f"{p}.attention.q_a_layernorm.weight", ql)
            group(f"{p}.attention.q_b_proj", H * (nope + rope), ql)
            group(f"{p}.attention.kv_a_proj_with_mqa", kv + rope, d, padded=True)
            bf16(f"{p}.attention.kv_a_layernorm.weight", kv)
            if kv_b_trellis:
                group(f"{p}.attention.kv_b_proj", H * (nope + vd), kv)
            else:
                bf16(f"{p}.attention.kv_b_proj.weight", H * (nope + vd), kv)
            bf16(f"{p}.attention.g_proj.weight", H, d)
            group(f"{p}.attention.dense", d, H * vd)
        else:
            for proj in ("f", "g") if fused_qkv else ("q", "k", "v", "f", "g"):
                group(f"{p}.attention.{proj}_proj", H * ld, d)
            if fused_qkv:
                group(f"{p}.attention.qkv_proj", 3 * H * ld, d)
            bf16(f"{p}.attention.b_proj.weight", H, d)
            for x in ("",) if fused_qkv else ("q_", "k_", "v_"):
                rows = (3 if fused_qkv else 1) * H * ld
                put(f"{p}.attention.{x}conv1d.weight", "BF16", [rows, 1, CFG["short_conv_kernel_size"]],
                    _bf16(torch.randn(rows, 1, CFG["short_conv_kernel_size"]) * 0.05))
            put(f"{p}.attention.A_log", "F32", [H], torch.randn(H).numpy().astype(np.float32))
            put(f"{p}.attention.dt_bias", "F32", [H * ld], torch.randn(H * ld).numpy().astype(np.float32))
            bf16(f"{p}.attention.o_norm.weight", ld)
            group(f"{p}.attention.o_proj", d, H * ld)
        if i < layers and i < CFG["first_k_dense_replace"]:
            group(f"{p}.mlp.gate_proj", CFG["intermediate_size"], d)
            group(f"{p}.mlp.up_proj", CFG["intermediate_size"], d)
            group(f"{p}.mlp.down_proj", d, CFG["intermediate_size"])
            continue
        bf16(f"{p}.mlp.gate.weight", CFG["num_experts"], d)
        put(f"{p}.mlp.gate.expert_bias", "F32", [CFG["num_experts"]],
            torch.randn(CFG["num_experts"]).numpy().astype(np.float32))
        for e in range(CFG["num_experts"]):
            for proj, outs, ins in (("gate_proj", CFG["moe_intermediate_size"], d),
                                    ("up_proj", CFG["moe_intermediate_size"], d),
                                    ("down_proj", d, CFG["moe_intermediate_size"])):
                group(f"{p}.mlp.experts.{e}.{proj}", outs, ins)
        if CFG["num_shared_experts"]:
            group(f"{p}.mlp.shared_experts.gate_proj", CFG["moe_shared_expert_intermediate_size"], d)
            group(f"{p}.mlp.shared_experts.up_proj", CFG["moe_shared_expert_intermediate_size"], d)
            group(f"{p}.mlp.shared_experts.down_proj", d, CFG["moe_shared_expert_intermediate_size"])
    if CFG["num_nextn_predict_layers"]:
        p = f"model.layers.{layers}"
        bf16(f"{p}.enorm.weight", d)
        bf16(f"{p}.hnorm.weight", d)
        group(f"{p}.eh_proj", d, 2 * d)
        bf16(f"{p}.final_layernorm.weight", d)

    header: dict = {}
    offset = 0
    payload = b""
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.nbytes]}
        payload += data.tobytes()
        offset += data.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with (folder / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(payload)
    config = {**CFG, "quantization_config": QUANT}
    (folder / "config.json").write_text(json.dumps(config))
    return folder
