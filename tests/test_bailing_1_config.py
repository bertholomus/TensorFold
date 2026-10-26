"""BailingMoeV3 config: the banked checkpoint's numbers, and every switch the engine refuses firing on a
mutated copy.

Run: python -m pytest tests/test_bailing_1_config.py -q
CPU only; reads the bank's config.json, never weights.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BANK = Path(os.environ.get("LING31_FP8_DIR", "/models/Ling-3.1-flash"))


def from_dict(raw: dict):
    from tensorfold.families.bailing_v3.config import Config

    return Config.from_dict(raw)


@pytest.mark.skipif(not BANK.is_dir(), reason="the banked Ling-3.1-flash checkpoint is not on this host")
def test_config_from_the_real_checkpoint():
    raw = json.loads((BANK / "config.json").read_text())
    cfg = from_dict(raw)
    assert cfg.hidden_size == 4096 and cfg.num_hidden_layers == 72 and cfg.vocab_size == 157_184
    assert cfg.rms_norm_eps == 1e-6 and cfg.num_attention_heads == 64
    assert cfg.q_lora_rank == 1536 and cfg.kv_lora_rank == 512
    assert cfg.qk_nope_head_dim == 128 and cfg.qk_rope_head_dim == 64
    assert cfg.qk_head_dim == 192 and cfg.v_head_dim == 128 and cfg.kv_b_per_head == 256
    assert cfg.lin_heads == 64 and cfg.lin_dim == 128 and cfg.conv_kernel == 4
    assert cfg.kda_lower_bound == -5.0
    assert cfg.n_routed_experts == 512 and cfg.num_experts_per_tok == 8
    assert cfg.n_group == 8 and cfg.topk_group == 4
    assert cfg.moe_intermediate_size == 1280 and cfg.intermediate_size == 10_240
    assert cfg.n_shared_experts == 1 and cfg.routed_scaling_factor == 2.5 and cfg.norm_topk_prob is True
    assert cfg.rope_theta == 6_000_000.0
    # MLA at layers 7, 15, ..., 71 (nine) and the MTP layer; KDA in the other 63; the dense MLPs are layers 0-3
    assert cfg.layer_kinds == ["kda" if (i + 1) % 8 else "mla" for i in range(72)]
    assert sum(k == "mla" for k in cfg.layer_kinds) == 9
    assert cfg.layer_kinds[0] == "kda" and cfg.layer_kinds[7] == "mla" and cfg.layer_kinds[71] == "mla"
    assert cfg.mlp_layer_types == ["dense"] * 4 + ["sparse"] * 68
    assert cfg.dense_layers == 4
    assert cfg.eos_token_id == (156895,)
    assert cfg.num_nextn_predict_layers == 1


@pytest.mark.skipif(not BANK.is_dir(), reason="the banked Ling-3.1-flash checkpoint is not on this host")
@pytest.mark.parametrize("key,value", [
    ("rope_scaling", {"rope_type": "linear", "factor": 4.0}),
    ("rope_interleave", False),
    ("no_kda_lora", False),
    ("use_kda_lora", True),
    ("gated_attention_proj_granularity_type", "element_wise"),
    ("gated_attention_proj_granularity_type", None),
    ("mtp_use_kda", True),
    ("num_kv_heads_for_linear_attn", 32),
    ("num_key_value_heads", 32),
    ("use_qkv_bias", True),
    ("use_bias", True),
    ("score_function", "softmax"),
    ("scoring_func", "softmax"),
    ("topk_method", "greedy"),
    ("value_norm", True),
    ("up_proj_norm", True),
    ("use_nGPT", True),
])
def test_each_refusal_fires_on_a_mutated_copy(key, value):
    """Every switch the reference implements and the engine does not refuses with the key named."""

    raw = json.loads((BANK / "config.json").read_text())
    raw[key] = value
    with pytest.raises(ValueError, match=key):
        from_dict(raw)


@pytest.mark.skipif(not BANK.is_dir(), reason="the banked Ling-3.1-flash checkpoint is not on this host")
def test_the_banked_config_passes_its_own_checks():
    """The unmutated bank parses cleanly (and a missing tensor list comes back empty against its index)."""

    raw = json.loads((BANK / "config.json").read_text())
    cfg = from_dict(raw)
    assert cfg.missing_tensors({"_model_dir": str(BANK)}) == []
