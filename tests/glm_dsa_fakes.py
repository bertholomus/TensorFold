"""A tiny EXL3-shaped GLM-5.3 checkpoint writer for the family's CPU tests: 4-bit trellis routed experts, BF16 elsewhere."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# the real architecture at 1/64 scale: 1 full + 1 shared DSA layer pair, 1 dense + 1 MoE layer, MTP present
CFG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa", "dtype": "bfloat16",
    "hidden_size": 128, "num_hidden_layers": 3, "vocab_size": 256,
    "rms_norm_eps": 1e-5, "num_attention_heads": 4, "q_lora_rank": 32, "kv_lora_rank": 64,
    "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "index_n_heads": 2, "index_head_dim": 16, "index_topk": 8,
    "indexer_types": ["full", "shared", "full"],
    "mlp_layer_types": ["dense", "sparse", "sparse"],
    "first_k_dense_replace": 1, "intermediate_size": 128, "moe_intermediate_size": 32,
    "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
    "routed_scaling_factor": 2.5, "norm_topk_prob": True, "n_group": 1, "topk_group": 1,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    "num_nextn_predict_layers": 1,
    "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "rope_interleave": True, "indexer_rope_interleave": True,
    "eos_token_id": [5], "tie_word_embeddings": False,
}
QUANT = {"bits": 4, "codebook": "mcg", "quant_method": "exl3"}


def _bf16(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.bfloat16).view(torch.uint16).numpy().astype(np.uint16)


def _trellis(outs: int, ins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A random 4-bit EXL3 pack: trellis int16 [ins/16, outs/16, 64] (K-tiles then N-tiles, ExLlamaV3's layout),
    suh [ins] (the input-side scales of the rotated domain), svh [outs]."""

    kt, nt = ins // 16, outs // 16
    trellis = torch.randint(-32768, 32767, (kt, nt, 64), dtype=torch.int16).numpy()
    suh = _bf16(torch.randn(ins))
    svh = _bf16(torch.randn(outs))
    return trellis, suh, svh


def write_checkpoint(folder: Path) -> Path:
    """The tiny checkpoint as one safetensors file plus its config; tensor names in the reference (bare) layout."""

    import struct

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, list[int], np.ndarray]] = {}

    def put(name: str, dtype: str, shape: list[int], data: np.ndarray) -> None:
        tensors[name] = (dtype, shape, data)

    def bf16(name: str, *shape: int) -> None:
        put(name, "BF16", list(shape), _bf16(torch.randn(*shape) * 0.05))

    d, v = CFG["hidden_size"], CFG["vocab_size"]
    bf16("model.embed_tokens.weight", v, d)
    bf16("lm_head.weight", v, d)
    bf16("model.norm.weight", d)
    layers = CFG["num_hidden_layers"]
    H, nope, rope, vd, kv, ql = CFG["num_attention_heads"], CFG["qk_nope_head_dim"], CFG["qk_rope_head_dim"], \
        CFG["v_head_dim"], CFG["kv_lora_rank"], CFG["q_lora_rank"]
    for i in range(layers + 1):                     # the MTP layer last
        p = f"model.layers.{i}"
        bf16(f"{p}.input_layernorm.weight", d)
        bf16(f"{p}.post_attention_layernorm.weight", d)
        bf16(f"{p}.self_attn.q_a_proj.weight", ql, d)
        bf16(f"{p}.self_attn.q_a_layernorm.weight", ql)
        bf16(f"{p}.self_attn.q_b_proj.weight", H * (nope + rope), ql)
        bf16(f"{p}.self_attn.kv_a_proj_with_mqa.weight", kv + rope, d)
        bf16(f"{p}.self_attn.kv_a_layernorm.weight", kv)
        bf16(f"{p}.self_attn.kv_b_proj.weight", H * (nope + vd), kv)
        bf16(f"{p}.self_attn.o_proj.weight", d, H * vd)
        if i == layers or CFG["indexer_types"][i] == "full":
            ix = f"{p}.self_attn.indexer"
            bf16(f"{ix}.wq_b.weight", CFG["index_n_heads"] * CFG["index_head_dim"], ql)
            bf16(f"{ix}.wk.weight", CFG["index_head_dim"], d)
            bf16(f"{ix}.k_norm.weight", CFG["index_head_dim"])
            bf16(f"{ix}.k_norm.bias", CFG["index_head_dim"])
            bf16(f"{ix}.weights_proj.weight", CFG["index_n_heads"], d)
        if i < layers and CFG["mlp_layer_types"][i] == "dense":
            bf16(f"{p}.mlp.gate_proj.weight", CFG["intermediate_size"], d)
            bf16(f"{p}.mlp.up_proj.weight", CFG["intermediate_size"], d)
            bf16(f"{p}.mlp.down_proj.weight", d, CFG["intermediate_size"])
            continue
        bf16(f"{p}.mlp.gate.weight", CFG["n_routed_experts"], d)
        put(f"{p}.mlp.gate.e_score_correction_bias", "F32", [CFG["n_routed_experts"]],
            torch.randn(CFG["n_routed_experts"]).numpy().astype(np.float32))
        for e in range(CFG["n_routed_experts"]):
            for proj, outs, ins in (("gate_proj", CFG["moe_intermediate_size"], d),
                                    ("up_proj", CFG["moe_intermediate_size"], d),
                                    ("down_proj", d, CFG["moe_intermediate_size"])):
                tr, suh, svh = _trellis(outs, ins)
                put(f"{p}.mlp.experts.{e}.{proj}.trellis", "I16", list(tr.shape), tr.astype(np.int16))
                put(f"{p}.mlp.experts.{e}.{proj}.suh", "BF16", [ins], suh)
                put(f"{p}.mlp.experts.{e}.{proj}.svh", "BF16", [outs], svh)
        if CFG["n_shared_experts"]:
            bf16(f"{p}.mlp.shared_experts.gate_proj.weight", CFG["moe_intermediate_size"], d)
            bf16(f"{p}.mlp.shared_experts.up_proj.weight", CFG["moe_intermediate_size"], d)
            bf16(f"{p}.mlp.shared_experts.down_proj.weight", d, CFG["moe_intermediate_size"])
    if CFG["num_nextn_predict_layers"]:
        p = f"model.layers.{layers}"
        bf16(f"{p}.enorm.weight", d)
        bf16(f"{p}.hnorm.weight", d)
        bf16(f"{p}.eh_proj.weight", d, 2 * d)
        bf16(f"{p}.shared_head.norm.weight", d)

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
