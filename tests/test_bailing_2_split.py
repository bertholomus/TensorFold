"""BailingMoeV3 (Ling-3.1-flash) split rules: every tensor name of the banked FP8 checkpoint hits exactly one
rule, every rank's share matches the expected per-rank shapes, and the GLM-5.3-Flash and GLM-5.3 indexes rule
exactly as they did before the BailingMoeV3 patterns were added (tests/fixtures/bailing_split_golden.json,
recorded from the pre-Ling tree).

Run: python -m pytest tests/test_bailing_2_split.py -q
CPU only; tensor names and file headers are read, never weights.
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.path.insert(0, str(REPO / "tests"))

LING_DIR = Path(os.environ.get("LING31_FP8_DIR", "/models/Ling-3.1-flash"))
GLM_FLASH_INDEX = Path(os.environ.get("GLM53_FLASH_NVFP4_DIR", "/models/GLM-5.3-Flash-NVFP4"),
                       "model.safetensors.index.json")
GLM53_INDEX = Path(os.environ.get("GLM53_BF16_DIR", "/models/GLM-5.3-BF16"), "model.safetensors.index.json")
GOLDEN = REPO / "tests" / "fixtures" / "bailing_split_golden.json"

WORLD = 2


def _abstract(name: str) -> str:
    """A tensor name's rule-relevant shape: layer and expert indices blanked."""
    return re.sub(r"\d+", "#", name)


def _index_names(index: Path) -> list[str]:
    return list(json.loads(index.read_text())["weight_map"])


def _headers(model_dir: Path) -> dict[str, tuple[str, list[int]]]:
    """name -> (dtype, shape) for every tensor of a sharded checkpoint (headers only, no data)."""
    out: dict[str, tuple[str, list[int]]] = {}
    for file in sorted(model_dir.glob("model-*.safetensors")):
        with file.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name != "__metadata__":
                out[name] = (info["dtype"], list(info["shape"]))
    return out


def _rank_shape(shape: list[int], kind: str, group: int) -> list[int]:
    """The shape rank r holds at ``WORLD``: the balanced uneven share of the split axis (no data needed).

    Mirrors split_bytes' arithmetic: a row split cuts the leading dim in whole ``group``s, a col split the
    second dim, a rep split nothing. Every axis here divides by ``WORLD`` exactly (asserted separately), so
    share() and share_lo() agree for both ranks and one expected shape covers rank 0 and rank 1.
    """
    if kind == "rep":
        return list(shape)
    axis = 0 if kind == "row" else 1
    assert shape[axis] % WORLD == 0, (shape, kind)
    if kind == "row" and group:
        assert (shape[axis] // group) % WORLD == 0, (shape, kind)
    out = list(shape)
    out[axis] //= WORLD
    return out


@pytest.mark.skipif(not LING_DIR.is_dir(), reason="the banked Ling-3.1-flash checkpoint is not on this host")
def test_every_ling_tensor_name_hits_exactly_one_rule():
    """All 214,025 names of the FP8 bank (weights and their FP8 block scales) resolve to one of row/col/rep."""
    from tensorfold.families.glm5_next.cuda.split import rule

    names = _index_names(LING_DIR / "model.safetensors.index.json")
    assert len(names) > 200_000
    kinds: dict[str, str] = {}
    for name in names:
        kind = rule(name)
        assert kind in ("row", "col", "rep"), (name, kind)      # nothing Ling serves is dropped
        a = _abstract(name)
        assert kinds.get(a, kind) == kind, f"{a}: {kinds.get(a)} and {kind}"
        kinds[a] = kind
    # the family's shapes: KDA rows (q k v f g, conv, dt_bias) and MLA rows (q_b, kv_b, the head gate),
    # input columns (o_proj, dense, the down projections), and the replicated norms / low-rank inputs / router
    assert kinds["model.layers.#.attention.q_proj.weight"] == "row"
    assert kinds["model.layers.#.attention.f_proj.weight"] == "row"
    assert kinds["model.layers.#.attention.g_proj.weight"] == "row"
    assert kinds["model.layers.#.attention.b_proj.weight"] == "row"
    assert kinds["model.layers.#.attention.q_conv#d.weight"] == "row"
    assert kinds["model.layers.#.attention.A_log"] == "row"
    assert kinds["model.layers.#.attention.dt_bias"] == "row"
    assert kinds["model.layers.#.attention.o_proj.weight"] == "col"
    assert kinds["model.layers.#.attention.q_a_proj.weight"] == "rep"
    assert kinds["model.layers.#.attention.q_b_proj.weight"] == "row"
    assert kinds["model.layers.#.attention.kv_a_proj_with_mqa.weight"] == "rep"
    assert kinds["model.layers.#.attention.kv_b_proj.weight"] == "row"
    assert kinds["model.layers.#.attention.dense.weight"] == "col"
    assert kinds["model.layers.#.attention.o_norm.weight"] == "rep"
    assert kinds["model.word_embeddings.weight"] == "rep"
    assert kinds["model.norm.weight"] == "rep"
    assert kinds["lm_head.weight"] == "rep"
    assert kinds["model.layers.#.mlp.gate.weight"] == "rep"
    assert kinds["model.layers.#.mlp.gate.expert_bias"] == "rep"
    assert kinds["model.layers.#.mlp.gate_proj.weight"] == "row"
    assert kinds["model.layers.#.mlp.down_proj.weight"] == "col"
    assert kinds["model.layers.#.mlp.experts.#.gate_proj.weight"] == "row"
    assert kinds["model.layers.#.mlp.experts.#.down_proj.weight"] == "col"
    assert kinds["model.layers.#.mlp.shared_experts.gate_proj.weight"] == "row"
    assert kinds["model.layers.#.mlp.shared_experts.down_proj.weight"] == "col"
    assert kinds["model.layers.#.eh_proj.weight"] == "rep"
    assert kinds["model.layers.#.enorm.weight"] == "rep"
    assert kinds["model.layers.#.hnorm.weight"] == "rep"
    assert kinds["model.layers.#.final_layernorm.weight"] == "rep"
    # an FP8 block scale travels with its weight: the scale grid's rows are the weight's out-tile rows
    assert kinds["model.layers.#.attention.q_proj.weight_scale_inv"] == "row"
    assert kinds["model.layers.#.attention.o_proj.weight_scale_inv"] == "col"


@pytest.mark.skipif(not LING_DIR.is_dir(), reason="the banked Ling-3.1-flash checkpoint is not on this host")
def test_ling_per_rank_shapes_match_the_design_table():
    """At TP2 each rank's part of the checkpoint has the shapes this test's table states (all splits whole-head)."""
    from tensorfold.families.glm5_next.cuda.split import head_group, rule

    headers = _headers(LING_DIR)
    assert len(headers) > 200_000
    # one tensor per role, both ranks: (name suffix, kind, the rank shape both ranks hold)
    expected = {
        "attention.q_proj.weight": ("row", [4096, 4096]),          # 32 heads x 128 rows a rank
        "attention.k_proj.weight": ("row", [4096, 4096]),
        "attention.v_proj.weight": ("row", [4096, 4096]),
        "attention.f_proj.weight": ("row", [4096, 4096]),
        "attention.b_proj.weight": ("row", [32, 4096]),            # one beta logit a head
        "attention.q_conv#d.weight": ("row", [4096, 1, 4]),        # the rank's q|k|v channels
        "attention.A_log": ("row", [32]),
        "attention.dt_bias": ("row", [4096]),
        "attention.o_proj.weight": ("col", [4096, 4096]),          # 4096 input columns a rank
        "attention.q_a_proj.weight": ("rep", [1536, 4096]),
        "attention.q_a_layernorm.weight": ("rep", [1536]),
        "attention.q_b_proj.weight": ("row", [6144, 1536]),        # 32 heads x 192 (128 nope + 64 rope)
        "attention.kv_a_proj_with_mqa.weight": ("rep", [576, 4096]),
        "attention.kv_a_layernorm.weight": ("rep", [512]),
        "attention.kv_b_proj.weight": ("row", [8192, 512]),        # 32 heads x 256 (128 nope + 128 v)
        "attention.dense.weight": ("col", [4096, 4096]),
        "attention.o_norm.weight": ("rep", [128]),
        "mlp.gate_proj.weight": ("row", [5120, 4096]),             # the dense layers' SwiGLU
        "mlp.up_proj.weight": ("row", [5120, 4096]),
        "mlp.down_proj.weight": ("col", [4096, 5120]),
        "mlp.gate.weight": ("rep", [512, 4096]),                   # the router (fp32 math)
        "mlp.gate.expert_bias": ("rep", [512]),
        "mlp.experts.#.gate_proj.weight": ("row", [640, 4096]),    # 640 = 5 x 128 intermediate a rank
        "mlp.experts.#.up_proj.weight": ("row", [640, 4096]),
        "mlp.experts.#.down_proj.weight": ("col", [4096, 640]),
        "mlp.shared_experts.gate_proj.weight": ("row", [640, 4096]),
        "mlp.shared_experts.down_proj.weight": ("col", [4096, 640]),
        "word_embeddings.weight": ("rep", [157184, 4096]),
        # the head is stored whole like GLM's (the shared rule is name-only and GLM's lm_head is the same
        # name) and sliced to its 78,592-row served span at load: tests/test_bailing_loader.py pins that span
        "lm_head.weight": ("rep", [157184, 4096]),
        "eh_proj.weight": ("rep", [4096, 8192]),
        "enorm.weight": ("rep", [4096]),
        "hnorm.weight": ("rep", [4096]),
        "final_layernorm.weight": ("rep", [4096]),
    }
    cfg = {"qk_dim": 192, "qk_nope": 128, "v_dim": 128}            # the config's own head-major group sizes
    assert head_group("model.layers.7.attention.q_b_proj.weight", cfg) == 192
    assert head_group("model.layers.7.attention.kv_b_proj.weight", cfg) == 256
    found = 0
    for name, (dtype, shape) in headers.items():
        kind = rule(name)
        a = _abstract(name)
        want = next((spec for suffix, spec in expected.items() if a == suffix or a.endswith("." + suffix)), None)
        if want is None:
            continue
        found += 1
        assert want[0] == kind, (name, kind)
        assert _rank_shape(shape, kind, head_group(name, cfg)) == want[1], name
    assert found >= 40, found                                       # every role above was seen in the bank
    # both g_proj kinds split by head rows: the KDA output gate [8192, 4096] and the MLA head gate [64, 4096]
    for name, (dtype, shape) in headers.items():
        if _abstract(name).endswith("attention.g_proj.weight"):
            assert rule(name) == "row" and shape[0] in (64, 8192), name
            assert _rank_shape(shape, "row", 0)[0] in (32, 4096) and _rank_shape(shape, "row", 0)[0] == shape[0] // 2, name
    # the head-major rows travel whole: 12288 = 64 heads x 192, 16384 = 64 heads x 256 (config-driven groups)
    assert headers["model.layers.7.attention.q_b_proj.weight"][1] == [12288, 1536]
    assert headers["model.layers.7.attention.kv_b_proj.weight"][1] == [16384, 512]
    # the MTP layer's g_proj is FP8 (the nine main MLA layers' are BF16, both row-split by head)
    assert headers["model.layers.72.attention.g_proj.weight"][1] == [64, 4096]
    assert headers["model.layers.7.attention.g_proj.weight"][0] == "BF16"
    assert headers["model.layers.72.attention.g_proj.weight"][0] == "F8_E4M3"
    # the kv_a block scale's grid is [5, 32] (its last block row is half full): the dequant's partial block
    assert headers["model.layers.7.attention.kv_a_proj_with_mqa.weight_scale_inv"][1] == [5, 32]


@pytest.mark.skipif(not (GLM_FLASH_INDEX.is_file() and GLM53_INDEX.is_file()),
                    reason="the banked GLM checkpoints are not on this host")
def test_glm_indexes_rule_exactly_as_before_the_ling_patterns():
    """Every GLM-5.3-Flash and GLM-5.3 tensor name rules identically to the pre-Ling tree (the golden)."""
    from tensorfold.families.glm5_next.cuda.split import rule

    golden = json.loads(GOLDEN.read_text())
    for tag, index in (("glm5_flash", GLM_FLASH_INDEX), ("glm53", GLM53_INDEX)):
        names = _index_names(index)
        seen: dict[str, str] = {}
        for name in names:
            kind = rule(name)
            a = _abstract(name)
            assert seen.get(a, kind) == kind, f"{tag} {a}: {seen.get(a)} vs {kind} ({name})"
            seen[a] = kind
        assert seen == golden[tag], f"{tag}: the Ling patterns moved {len(golden[tag])} GLM rules"
