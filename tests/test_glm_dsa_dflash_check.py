"""A drafter brought with --drafter on GLM-5.3 (glm_moe_dsa.cuda.dflash.check): a DFlash2 checkpoint trained for the
target attaches; any other is refused with the reason before a weight loads."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

GLM = SimpleNamespace(hidden=6144, vocab=154880, layers=78)
GOOD = {
    "architectures": ["DFlash2DraftModel"], "hidden_size": 6144, "vocab_size": 154880, "head_dim": 128,
    "num_attention_heads": 48, "num_key_value_heads": 8, "intermediate_size": 12288, "sliding_window": 4096,
    "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 154856,
                      "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [16, 31, 45, 55, 65, 75]},
}


def _dir(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").write_bytes(b"")
    return tmp_path


def test_a_dflash2_drafter_for_glm53_attaches_on_four_ranks(tmp_path):
    from tensorfold.families.glm_moe_dsa.cuda import dflash

    t = dflash.check(_dir(tmp_path, GOOD), GLM, 4)
    assert dflash.need_rows(t) == 4096 + 8


@pytest.mark.parametrize("change,why", [
    ({"architectures": ["DFlashDraftModel"]}, "DFlash2 layout"),
    ({"hidden_size": 5120}, "hidden size"),
    ({"vocab_size": 151936}, "vocabulary"),
    ({"num_attention_heads": 46}, "num_attention_heads"),
    ({"num_key_value_heads": 6}, "num_key_value_heads"),
    ({"sliding_window": None}, "sliding window"),
    ({"dflash_config": {**GOOD["dflash_config"], "target_layer_ids": [5, 78]}}, "target layers"),
    ({"dflash_config": {**GOOD["dflash_config"], "block_size": 16}}, "block of 16"),
    ({"dflash_config": {**GOOD["dflash_config"], "conv_kernel_size": 3}}, "two-tap"),
])
def test_other_drafters_are_refused_with_the_reason(tmp_path, change, why):
    from tensorfold.families.glm_moe_dsa.cuda import dflash

    with pytest.raises(ValueError, match=why):
        dflash.check(_dir(tmp_path, {**GOOD, **change}), GLM, 4)


def test_a_directory_without_the_weights_is_refused(tmp_path):
    from tensorfold.families.glm_moe_dsa.cuda import dflash

    (tmp_path / "config.json").write_text(json.dumps(GOOD))
    with pytest.raises(ValueError, match="model.safetensors"):
        dflash.check(tmp_path, GLM, 4)


def test_a_requests_policy_sets_the_drafters_depth_and_chance():
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    code = GlmEngine._drafter_code
    e = SimpleNamespace(drafter=SimpleNamespace(block=8))
    assert code(e, "auto") == [0, 0] and code(e, "0") == [0, 0] and code(e, "") == [0, 0]
    assert code(e, "5") == [5, 0] and code(e, "5:0.3") == [5, 300000] and code(e, "9") == [7, 0]
    assert code(SimpleNamespace(drafter=None), "5:0.3") == [0, 0]       # MTP lanes: untouched
    assert code(e, "m3") == [-1, 0] and code(SimpleNamespace(drafter=None), "m3") == [0, 0]   # MTP beside the drafter
    with pytest.raises(ValueError):
        code(e, "5:1.5")
