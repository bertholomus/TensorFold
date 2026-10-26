"""BailingMoeV3 (Ling-3.1-flash) at TP4: the split rules in whole 128-column Hadamard blocks, the conversion's fused
KDA names, the loader on four ranks, the header checker (tools/ling_split_check.py), and GLM unchanged at TP2 / TP4.

Ling's 1280-wide routed and shared experts are 10 Hadamard blocks: 640/640 a rank at TP2, 384/384/256/256 at TP4.
Every other Ling axis divides evenly at both worlds, so its rank shares are the plain split's.

Run: python -m pytest tests/test_bailing_10_tp4.py -q
CPU only.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "tests", REPO / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tensorfold.cuda.geometry import share, share_lo  # noqa: E402
from tensorfold.families.glm5_next.cuda import split  # noqa: E402

LING_DIR = Path(os.environ.get("LING31_FP8_DIR", "/models/Ling-3.1-flash"))
MODELS = Path(os.environ.get("TF_MODELS_DIR", "/models"))
GLM_EXL3 = sorted(MODELS.glob("GLM-5.3-Flash-EXL3-TR3-4bpw*")) if MODELS.is_dir() else []
HINT = {"qk_dim": 192, "qk_nope": 128, "v_dim": 128}            # Ling's head-major group sizes (config-driven)


def ling_config() -> dict:
    """Ling-3.1-flash's config.json numbers on the tiny fake's switches (the family's Config reads either)."""

    from bailing_fakes import CFG

    return {**CFG, "hidden_size": 4096, "num_hidden_layers": 72, "vocab_size": 157184,
            "num_attention_heads": 64, "num_key_value_heads": 64, "head_dim": 128,
            "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "v_head_dim": 128, "qk_head_dim": 192, "layer_group_size": 8, "first_k_dense_replace": 4,
            "num_experts": 512, "num_experts_per_tok": 8, "n_group": 8, "topk_group": 4,
            "moe_intermediate_size": 1280, "moe_shared_expert_intermediate_size": 1280, "intermediate_size": 10240,
            "expert_swiglu_limit_list": None, "share_expert_swiglu_limit_list": None}


def ling_exl3_names(cfg: dict, layers=(0, 4, 7, 72), experts: int = 3, fused: bool = True) -> dict:
    """The EXL3 conversion's headers at Ling's real dims for a few layers (KDA + dense, KDA + MoE, MLA + MoE, the
    MTP layer): exllamav3's names (fused qkv_proj and conv1d, the head gate and b_proj unquantized, kv_b raw)."""

    names: dict[str, tuple[str, list[int]]] = {}
    d, v, H, ld = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"], cfg["head_dim"]
    ql, kv, nope, rope, vd = (cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["qk_nope_head_dim"],
                              cfg["qk_rope_head_dim"], cfg["v_head_dim"])

    def plain(name: str, dtype: str, *shape: int) -> None:
        names[name] = (dtype, list(shape))

    def group(p: str, outs: int, ins: int, pad: bool = False) -> None:
        n = -(-outs // 128) * 128 if pad else outs
        names[p + ".trellis"] = ("I16", [ins // 16, n // 16, 48])
        names[p + ".suh"] = ("F16", [ins])
        names[p + ".svh"] = ("F16", [n])
        names[p + ".mcg"] = ("I32", [1])

    plain("model.word_embeddings.weight", "BF16", v, d)
    group("lm_head", v, d)
    plain("model.norm.weight", "BF16", d)
    L = cfg["num_hidden_layers"]
    for i in layers:
        p = f"model.layers.{i}"
        plain(f"{p}.input_layernorm.weight", "BF16", d)
        plain(f"{p}.post_attention_layernorm.weight", "BF16", d)
        if i >= L or (i + 1) % cfg["layer_group_size"] == 0:
            group(f"{p}.attention.q_a_proj", ql, d)
            plain(f"{p}.attention.q_a_layernorm.weight", "BF16", ql)
            group(f"{p}.attention.q_b_proj", H * (nope + rope), ql)
            group(f"{p}.attention.kv_a_proj_with_mqa", kv + rope, d, pad=True)
            plain(f"{p}.attention.kv_a_layernorm.weight", "BF16", kv)
            plain(f"{p}.attention.kv_b_proj.weight", "BF16", H * (nope + vd), kv)
            plain(f"{p}.attention.g_proj.weight", "BF16", H, d)
            group(f"{p}.attention.dense", d, H * vd)
        else:
            if fused:
                group(f"{p}.attention.qkv_proj", 3 * H * ld, d)
                plain(f"{p}.attention.conv1d.weight", "BF16", 3 * H * ld, 1, 4)
            else:
                for x in "qkv":
                    group(f"{p}.attention.{x}_proj", H * ld, d)
                    plain(f"{p}.attention.{x}_conv1d.weight", "BF16", H * ld, 1, 4)
            group(f"{p}.attention.f_proj", H * ld, d)
            group(f"{p}.attention.g_proj", H * ld, d)
            plain(f"{p}.attention.b_proj.weight", "BF16", H, d)
            plain(f"{p}.attention.A_log", "F32", H)
            plain(f"{p}.attention.dt_bias", "F32", H * ld)
            plain(f"{p}.attention.o_norm.weight", "BF16", ld)
            group(f"{p}.attention.o_proj", d, H * ld)
        if i < cfg["first_k_dense_replace"]:
            group(f"{p}.mlp.gate_proj", cfg["intermediate_size"], d)
            group(f"{p}.mlp.up_proj", cfg["intermediate_size"], d)
            group(f"{p}.mlp.down_proj", d, cfg["intermediate_size"])
            continue
        plain(f"{p}.mlp.gate.weight", "BF16", cfg["num_experts"], d)
        plain(f"{p}.mlp.gate.expert_bias", "F32", cfg["num_experts"])
        for e in range(experts):
            group(f"{p}.mlp.experts.{e}.gate_proj", cfg["moe_intermediate_size"], d)
            group(f"{p}.mlp.experts.{e}.up_proj", cfg["moe_intermediate_size"], d)
            group(f"{p}.mlp.experts.{e}.down_proj", d, cfg["moe_intermediate_size"])
        s = cfg["moe_shared_expert_intermediate_size"]
        group(f"{p}.mlp.shared_experts.gate_proj", s, d)
        group(f"{p}.mlp.shared_experts.up_proj", s, d)
        group(f"{p}.mlp.shared_experts.down_proj", d, s)
        if i >= L:
            plain(f"{p}.enorm.weight", "BF16", d)
            plain(f"{p}.hnorm.weight", "BF16", d)
            group(f"{p}.eh_proj", d, 2 * d)
            plain(f"{p}.final_layernorm.weight", "BF16", d)
    return names


# ---------------------------------------------------------------- rules and spans


def test_the_fused_kda_names_rule_and_split_in_three_segments():
    """exllamav3's fused q | k | v group and conv: row-split parts, three equal segments each split by head."""

    p = "model.layers.0.attention."
    assert split.rule(p + "qkv_proj.trellis") == "dim1"            # the output tiles
    assert split.rule(p + "qkv_proj.svh") == "row"
    assert split.rule(p + "qkv_proj.suh") == "rep"                 # the input side (hidden) stays whole
    assert split.rule(p + "qkv_proj.mcg") == "rep"
    assert split.rule(p + "conv1d.weight") == "row"
    assert split.segments(p + "qkv_proj.trellis") == split.segments(p + "conv1d.weight") == 3
    for other in ("q_proj.trellis", "f_proj.svh", "q_conv1d.weight", "o_proj.trellis", "dt_bias"):
        assert split.segments(p + other) == 1, other
    assert split.segments("model.layers.7.attention.q_b_proj.svh") == 1
    # the conversion keeps the MLA head gate and b_proj unquantized: plain row splits by head
    assert split.rule("model.layers.7.attention.g_proj.weight") == "row"
    assert split.rule(p + "b_proj.weight") == "row"


@pytest.mark.parametrize("world,want", [(2, [640, 640]), (4, [384, 384, 256, 256])])
def test_ling_expert_widths_move_in_whole_hadamard_blocks(world, want):
    """1280 = 10 blocks: TP2 halves them, TP4 gives the first two ranks three blocks and the last two two."""

    for prefix in ("model.layers.4.mlp.experts.0.", "model.layers.4.mlp.shared_experts."):
        tr = [256, 80, 48]                                          # gate / up: [K/16, N/16, words], N the width
        assert split.unit(prefix + "gate_proj.trellis", tr, "dim1", world) == 8
        got = [split.spans(80, world, r, 8) for r in range(world)]
        assert [t * 16 for (_, t), in got] == want
        assert [a * 16 for (a, _), in got] == [sum(want[:r]) for r in range(world)]
        assert split.unit(prefix + "gate_proj.svh", [1280], "row", world) == 128
        assert [split.spans(1280, world, r, 128)[0][1] for r in range(world)] == want
        down = [80, 256, 48]                                        # down: the width is its input tiles
        assert split.rule(prefix + "down_proj.trellis") == "row" and split.rule(prefix + "down_proj.suh") == "row"
        assert split.unit(prefix + "down_proj.trellis", down, "row", world) == 8
        assert split.unit(prefix + "down_proj.suh", [1280], "row", world) == 128
    # the plain share would cut a block at TP4 (320 = 2.5 blocks) and agree at TP2
    assert share(1280, world, 0) == (640 if world == 2 else 320)


@pytest.mark.parametrize("world", [2, 4])
def test_every_other_ling_axis_splits_exactly_as_the_plain_share(world):
    """Attention, dense MLP and head axes divide evenly: whole blocks and whole heads change no byte at TP2 / TP4."""

    cases = [  # (name, shape, kind): Ling's real shapes
        ("model.layers.0.attention.f_proj.trellis", [256, 512, 48], "dim1"),
        ("model.layers.0.attention.f_proj.svh", [8192], "row"),
        ("model.layers.0.attention.o_proj.trellis", [512, 256, 48], "row"),
        ("model.layers.0.attention.o_proj.suh", [8192], "row"),
        ("model.layers.0.mlp.gate_proj.trellis", [256, 640, 48], "dim1"),
        ("model.layers.0.mlp.down_proj.suh", [10240], "row"),
        ("model.layers.7.attention.q_b_proj.trellis", [96, 768, 48], "dim1"),
        ("model.layers.7.attention.q_b_proj.svh", [12288], "row"),
        ("model.layers.7.attention.dense.trellis", [512, 256, 48], "row"),
        ("model.layers.7.attention.kv_b_proj.weight", [16384, 512], "row"),
        ("model.layers.7.attention.g_proj.weight", [64, 4096], "row"),
        ("model.layers.0.attention.b_proj.weight", [64, 4096], "row"),
        ("model.layers.0.attention.dt_bias", [8192], "row"),
        ("model.layers.0.attention.A_log", [64], "row"),
    ]
    for name, shape, kind in cases:
        assert split.rule(name) == kind, name
        n = shape[0] if kind == "row" else shape[1]
        u = split.unit(name, shape, kind, world, HINT)
        for r in range(world):
            assert split.spans(n, world, r, u) == [(share_lo(n, world, r), share(n, world, r))], (name, r)
    # q_b: whole heads and whole blocks at once (192-row heads: 2 heads in 3 blocks), 16 heads a rank at TP4
    assert split.unit("model.layers.7.attention.q_b_proj.svh", [12288], "row", world, HINT) == 384
    assert split.unit("model.layers.7.attention.q_b_proj.trellis", [96, 768, 48], "dim1", world, HINT) == 24
    assert split.unit("model.layers.7.attention.kv_b_proj.weight", [16384, 512], "row", world, HINT) == 256


@pytest.mark.parametrize("world", [2, 4])
def test_the_fused_qkv_and_conv_hand_each_rank_its_heads_of_q_k_and_v(world):
    """A rank's share of qkv_proj / conv1d is [q_r | k_r | v_r]: its heads' rows of each third, the layout the KDA
    chain reads (separate q / k / v would give the same rows)."""

    C, HL = 8192, 64 // world                                       # one third, and the rank's heads
    for r in range(world):
        lo = share_lo(64, world, r) * 128
        assert split.spans(3 * C // 16, world, r, split.unit("x.attention.qkv_proj.trellis", [256, 1536, 48],
                                                                 "dim1", world), 3) == \
            [((s * C + lo) // 16, HL * 128 // 16) for s in range(3)]
        assert split.spans(3 * C, world, r, split.unit("x.attention.qkv_proj.svh", [3 * C], "row", world), 3) == \
            [(s * C + lo, HL * 128) for s in range(3)]
        assert split.spans(3 * C, world, r, split.unit("x.attention.conv1d.weight", [3 * C, 1, 4], "row", world),
                           3) == [(s * C + lo, HL * 128) for s in range(3)]


def test_scale_vectors_follow_the_trellis_tiles_below_one_block_a_rank():
    """Test-scale widths with fewer blocks than ranks keep the 16-column tile split, svh in the same tiles."""

    for world in (4, 6):
        tiles = [split.spans(2, world, r, split.unit("x.mlp.experts.0.gate_proj.trellis", [8, 2, 64], "dim1",
                                                     world)) for r in range(world)]
        svh = [split.spans(32, world, r, split.unit("x.mlp.experts.0.gate_proj.svh", [32], "row", world))
               for r in range(world)]
        assert [[(a * 16, t * 16) for a, t in rr] for rr in tiles] == svh
    assert [split.spans(32, 6, r, 16)[0][1] for r in range(6)] == [16, 16, 0, 0, 0, 0]


def test_glm53_at_w6_takes_whole_blocks_and_whole_heads():
    """GLM-5.3's 2048-wide experts at W=6 move in whole blocks (the TP6 lane's 384/384/384/384/256/256) and its
    EXL3 q_b in whole 256-row heads and blocks (11/11/11/11/10/10 heads), trellis and svh alike."""

    glm = GLM53
    u = split.unit("model.language_model.layers.5.mlp.experts.0.gate_proj.trellis", [384, 128, 48], "dim1", 6)
    assert [split.spans(128, 6, r, u)[0][1] * 16 for r in range(6)] == [384] * 4 + [256] * 2
    u = split.unit("model.layers.5.self_attn.q_b_proj.svh", [16384], "row", 6, glm)
    assert u == 256
    assert [split.spans(16384, 6, r, u)[0][1] // 256 for r in range(6)] == [11] * 4 + [10] * 2
    u = split.unit("model.layers.5.self_attn.q_b_proj.trellis", [128, 1024, 48], "dim1", 6, glm)
    assert [split.spans(1024, 6, r, u)[0][1] * 16 // 256 for r in range(6)] == [11] * 4 + [10] * 2


@pytest.mark.parametrize("kind", ["row", "dim1", "col"])
def test_split_bytes_and_split_device_reassemble_segments_and_groups(kind):
    """Every rank's runs, cut out of a byte pattern and put back where spans says, rebuild the tensor exactly."""

    torch = pytest.importorskip("torch")
    shape = {"row": [3 * 1280, 4], "dim1": [5, 3 * 80, 6], "col": [7, 3 * 1280]}[kind]
    n = shape[0] if kind == "row" else shape[1]
    group = 8 if kind == "dim1" else 128
    raw = np.random.default_rng(7).integers(0, 256, int(np.prod(shape)) * 2, dtype=np.uint8)   # itemsize 2
    full = raw.reshape(shape[0], -1) if kind == "row" else raw.reshape(shape[0], shape[1], -1)
    for world in (2, 4):
        rebuilt = np.zeros_like(full)
        for r in range(world):
            data, s = split.split_bytes(raw, shape, 2, kind, r, world=world, group=group, parts=3)
            dev, s2 = split.split_device(torch.from_numpy(raw.copy()), shape, 2, kind, r, world=world, group=group,
                                         parts=3)
            assert s == s2 and np.array_equal(dev.numpy(), data)
            runs = split.spans(n, world, r, group, 3)
            take = sum(t for _, t in runs)
            assert len(runs) == 3 and s[0 if kind == "row" else 1] == take
            part = data.reshape(take, -1) if kind == "row" else data.reshape(shape[0], take, -1)
            at = 0
            for a, t in runs:
                if kind == "row":
                    rebuilt[a:a + t] = part[at:at + t]
                else:
                    rebuilt[:, a:a + t] = part[:, at:at + t]
                at += t
        assert np.array_equal(rebuilt, full), world


# ---------------------------------------------------------------- the header checker


@pytest.mark.parametrize("fused", [True, False])
def test_the_checker_passes_ling_at_tp2_and_tp4_with_the_design_widths(fused):
    from ling_split_check import check

    cfg = ling_config()
    names = ling_exl3_names(cfg, fused=fused)
    for world, width in ((2, [640, 640]), (4, [384, 384, 256, 256])):
        problems, notes, table, total = check(names, world, cfg)
        assert problems == [] and notes == [], problems[:5]
        for li in (4, 7, 72):
            for who in ("routed experts", "shared expert"):
                for side in ("out", "in"):
                    assert table[f"L{li} {who} width ({side})"][3] == width
        assert table["L0 dense MLP width (out)"][3] == [10240 // world] * world
        assert table["L0 KDA heads"][3] == table["L4 KDA heads"][3] == [64 // world] * world
        assert table["L7 MLA heads"][3] == table["L72 MLA heads"][3] == [64 // world] * world
        if fused:
            kind, u, parts, shapes = table["KDA model.layers.#.attention.qkv_proj.trellis"]
            assert (kind, u, parts) == ("dim1", 8, 3) and shapes == [(256, 3 * 512 // world, 48)] * world
            assert table["KDA model.layers.#.attention.conv1d.weight"][3] == [(3 * 8192 // world, 1, 4)] * world
        assert table["MLA model.layers.#.attention.q_b_proj.svh"][1] == 384
        assert table["lm_head.trellis"][3] == [(256, 1228 // world * 8, 48)] * world   # 307 blocks a rank at TP4
    # TP4's only uneven bytes are the routed and shared experts' extra block on ranks 0-1: 128 more columns of
    # gate, up (each [4096 in, 128 out]) and down ([128 in, 4096 out]) at 3 bits, a block of each scale vector
    *_, total = check(names, 4, cfg)
    per_expert = 3 * (4096 * 128 * 48 // 256 * 2) + 3 * 128 * 2
    assert total[0] == total[1] and total[2] == total[3]
    assert total[0] - total[2] == 3 * (3 + 1) * per_expert            # 3 MoE layers x (3 experts + the shared)


def test_the_checker_reports_a_cut_block_a_split_group_and_an_unruled_name():
    from ling_split_check import check

    cfg = ling_config()
    names = ling_exl3_names(cfg, layers=(4,), experts=1)
    names["model.layers.4.attention.weird_proj.weight"] = ("BF16", [64, 4096])
    problems, *_ = check(names, 4, cfg)
    assert any("weird_proj" in p and "rule" in p for p in problems)
    # a 1312-wide expert (10.25 blocks: never a real width) cannot split in whole blocks: the tile split cuts one
    bad = ling_exl3_names({**cfg, "moe_intermediate_size": 1312}, layers=(4,), experts=1)
    problems, *_ = check(bad, 4, {**cfg, "moe_intermediate_size": 1312})
    assert any("cuts a 128-column Hadamard block" in p for p in problems)
    # an svh that disagrees with its trellis's output tiles: the group's two output-side parts split differently
    names = ling_exl3_names(cfg, layers=(4,), experts=1)
    names["model.layers.4.mlp.experts.0.gate_proj.svh"] = ("F16", [1024])
    problems, *_ = check(names, 4, cfg)
    assert any("gate_proj: its output-side parts" in p for p in problems)


def test_the_checker_cli_on_the_tiny_tp4_checkpoint(tmp_path, capsys):
    from bailing_fakes import CFG, CFG_TP4, write_checkpoint
    from ling_split_check import main

    folder = write_checkpoint(tmp_path / "tp4", cfg=CFG_TP4, fused_qkv=True)
    assert main([str(folder), "--world", "2", "--world", "4", "--table"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "routed experts width (out): [384, 384, 256, 256]" in out
    # the TP2 fake's 256-wide experts are two blocks: at TP4 the tile split would cut them
    folder = write_checkpoint(tmp_path / "tp2", cfg=CFG)
    assert main([str(folder), "--world", "4"]) == 1
    assert "cuts a 128-column Hadamard block" in capsys.readouterr().out


def test_the_checker_reports_a_name_two_shards_hold(tmp_path, capsys):
    """A name the conversion writes into more than one shard: byte-identical copies are a NOTE and the run
    still passes; differing copies are a PROBLEM and the exit is 1 (the dict kept the last copy silently)."""

    from bailing_fakes import CFG, write_checkpoint
    from ling_split_check import headers, main

    name = "model.layers.3.enorm.weight"                       # the MTP layer's enorm, a replicated tensor
    folder = write_checkpoint(tmp_path / "same", cfg=CFG)
    header, base = split.read_header(folder / "model.safetensors")
    a, b = header[name]["data_offsets"]
    raw = np.fromfile(folder / "model.safetensors", dtype=np.uint8)[base + a:base + b]
    split.write(str(folder / "model-00002.safetensors"),
                [(name, header[name]["dtype"], list(header[name]["shape"]), raw)], None)
    dups: list[tuple[str, str]] = []
    assert headers(folder, dups)[name] == (header[name]["dtype"], header[name]["shape"])
    assert [kind for kind, _ in dups] == ["note"] and name in dups[0][1]
    assert main([str(folder), "--world", "2"]) == 0
    out = capsys.readouterr().out
    assert "NOTE" in out and name in out and "byte-identical" in out

    folder = write_checkpoint(tmp_path / "diff", cfg=CFG)
    header, base = split.read_header(folder / "model.safetensors")
    a, b = header[name]["data_offsets"]
    wrong = np.fromfile(folder / "model.safetensors", dtype=np.uint8)[base + a:base + b].copy()
    wrong[0] ^= 0xFF
    split.write(str(folder / "model-00002.safetensors"),
                [(name, header[name]["dtype"], list(header[name]["shape"]), wrong)], None)
    dups = []
    headers(folder, dups)
    assert [kind for kind, _ in dups] == ["problem"] and name in dups[0][1]
    assert main([str(folder), "--world", "2"]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM" in out and name in out and "the copies differ" in out


@pytest.mark.skipif(not LING_DIR.is_dir(), reason="the banked Ling-3.1-flash checkpoint is not on this host")
def test_the_checker_on_the_fp8_bank_rules_every_name_and_splits_every_head():
    """The FP8 bank's names at TP4: every name rules, every head and width partitions; its block scales are notes
    (an FP8 rank split is not a serving path)."""

    from ling_split_check import check, headers

    names = headers(LING_DIR)
    config = json.loads((LING_DIR / "config.json").read_text())
    for world in (2, 4):
        problems, notes, table, total = check(names, world, config)
        assert problems == [], problems[:5]
        assert table["L7 MLA heads"][3] == [64 // world] * world
        assert table["L4 routed experts width (out)"][3] == [1280 // world] * world   # FP8: no Hadamard blocks
    assert any("experts.#.gate_proj.weight_scale_inv" in n for n in notes)            # [10, 32] grids at TP4


# ---------------------------------------------------------------- the loader and the rank reader on four ranks


def _full(folder: Path) -> dict[str, np.ndarray]:
    """Every tensor of the fake's one file as raw bytes, rows first ([rows, bytes a row])."""

    path = folder / "model.safetensors"
    header, base = split.read_header(path)
    blob = np.fromfile(path, dtype=np.uint8)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        out[name] = (blob[base + a:base + b], list(info["shape"]), split.DTYPE_BYTES[info["dtype"]])
    return out


@pytest.mark.parametrize("world", [2, 4])
def test_rank_reader_and_split_file_hand_out_the_same_bytes(tmp_path, monkeypatch, world):
    """Every tensor at every rank: RankReader on the full checkpoint == split_file's rank folder, and each equals the
    rank's runs of the full tensor (fused qkv / conv as [q_r | k_r | v_r], experts in whole blocks)."""

    torch = pytest.importorskip("torch")
    from bailing_fakes import CFG_TP4, write_checkpoint

    folder = write_checkpoint(tmp_path / "full", cfg=CFG_TP4, fused_qkv=True)
    monkeypatch.setenv("TF_TP_WORLD", str(world))
    hint = split.cfg_hint(folder)
    assert hint == {"qk_dim": 128, "qk_nope": 64, "v_dim": 64}
    full = _full(folder)
    for r in range(world):
        out = tmp_path / f"rank{r}"
        out.mkdir()
        split.split_file(folder / "model.safetensors", out, r, hint)
        (out / "config.json").write_text((folder / "config.json").read_text())
        whole, own = split.RankReader(folder, r, cfg_hint=hint), split.RankReader(out, r, cfg_hint=hint)
        try:
            for name, (raw, shape, size) in full.items():
                a, b = whole.get(name), own.get(name)
                assert a.dtype == b.dtype and a.shape == b.shape, name
                a = a.contiguous().reshape(-1).view(torch.uint8)
                assert torch.equal(a, b.contiguous().reshape(-1).view(torch.uint8)), name
                kind = split.rule(name)
                if kind == "rep":
                    continue
                u = split.unit(name, shape, kind, world, hint)
                runs = split.spans(shape[0] if kind == "row" else shape[1], world, r, u, split.segments(name))
                view = raw.reshape(shape[0], -1)
                if kind == "row":
                    want = np.concatenate([view[a0:a0 + t] for a0, t in runs])
                else:
                    inner = view.shape[1] // shape[1]
                    want = np.concatenate([view[:, a0 * inner:(a0 + t) * inner] for a0, t in runs], axis=1)
                assert np.array_equal(a.numpy(), want.reshape(-1)), name
        finally:
            whole.close()
            own.close()
    # a folder holding only another rank's share is refused for this rank (any world, not just 2)
    with pytest.raises(ValueError, match="another rank's share"):
        split.RankReader(tmp_path / f"rank{world - 1}", 0)


def _shard_with(folder: Path, name: str, data: np.ndarray) -> Path:
    """A second shard of ``folder`` that holds ``name`` again with the given raw bytes (the conversion
    writes a replicated name into more than one shard — the MTP layer's own pieces came in two)."""

    info = split.read_header(folder / "model.safetensors")[0][name]
    path = folder / "model-00002.safetensors"
    split.write(str(path), [(name, info["dtype"], list(info["shape"]), data)], None)
    return path


@pytest.mark.parametrize("world", [2, 4])
def test_split_file_writes_a_replicated_name_once(tmp_path, monkeypatch, world):
    """A name two shards hold (the real case: the MTP layer's own pieces in two shards, byte-identical) is
    written once: the copies are compared byte for byte, one is kept, and the rank folder loads."""

    torch = pytest.importorskip("torch")
    from bailing_fakes import CFG, write_checkpoint

    name = "model.layers.3.enorm.weight"                       # the MTP layer's enorm, a replicated tensor
    folder = write_checkpoint(tmp_path / "full", cfg=CFG)
    monkeypatch.setenv("TF_TP_WORLD", str(world))
    header, base = split.read_header(folder / "model.safetensors")
    a, b = header[name]["data_offsets"]
    _shard_with(folder, name, np.fromfile(folder / "model.safetensors", dtype=np.uint8)[base + a:base + b])
    out = tmp_path / "rank0"
    split.split_file(folder / "model.safetensors", out, 0, split.cfg_hint(folder))
    assert split.split_file(folder / "model-00002.safetensors", out, 0, split.cfg_hint(folder))["dup"] == 1
    names = [n for p in split.rank_files(out, 0) for n in split.read_header(p)[0] if n != "__metadata__"]
    assert names.count(name) == 1                              # one copy kept, not one a shard
    whole, own = split.RankReader(folder, 0), split.RankReader(out, 0)
    try:                                                       # the loader reads one copy, and the right one
        got = own.get(name).contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(got, whole.get(name).contiguous().reshape(-1).view(torch.uint8))
    finally:
        whole.close()
        own.close()


def test_split_file_refuses_differing_copies_of_a_replicated_name(tmp_path, monkeypatch):
    """Two shards with DIFFERENT bytes for one name: the split raises naming the tensor and both shards
    (the old skip kept the first of two differing copies silently)."""

    from bailing_fakes import CFG, write_checkpoint

    name = "model.layers.3.enorm.weight"
    folder = write_checkpoint(tmp_path / "full", cfg=CFG)
    monkeypatch.setenv("TF_TP_WORLD", "2")
    header, base = split.read_header(folder / "model.safetensors")
    a, b = header[name]["data_offsets"]
    wrong = np.fromfile(folder / "model.safetensors", dtype=np.uint8)[base + a:base + b].copy()
    wrong[0] ^= 0xFF
    _shard_with(folder, name, wrong)
    out = tmp_path / "rank0"
    split.split_file(folder / "model.safetensors", out, 0, split.cfg_hint(folder))
    with pytest.raises(ValueError, match=r"model\.layers\.3\.enorm\.weight: rank 0's copies differ"
                                        r" between model\.safetensors and model-00002\.safetensors"):
        split.split_file(folder / "model-00002.safetensors", out, 0, split.cfg_hint(folder))


@pytest.mark.parametrize("world,width", [(2, [640, 640]), (4, [384, 384, 256, 256])])
def test_loader_on_the_tiny_tp4_checkpoint_every_rank(tmp_path, monkeypatch, world, width):
    """weights.load at TF_TP_WORLD 2 and 4 on the fused-name fake: expert and shared widths in whole blocks, the
    fused q | k | v group and conv per rank, heads, the head's vocabulary span."""

    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU-only loader test")
    from bailing_fakes import CFG_TP4, write_checkpoint

    from tensorfold.families.bailing_v3 import check
    from tensorfold.families.bailing_v3.cuda import weights as W

    folder = write_checkpoint(tmp_path, cfg=CFG_TP4, fused_qkv=True)
    check(folder)                                   # the fused names satisfy the architecture's tensor list
    monkeypatch.setenv("TF_TP_WORLD", str(world))
    H, ld, d = CFG_TP4["num_attention_heads"], CFG_TP4["head_dim"], CFG_TP4["hidden_size"]
    LL = H // world
    got = []
    for r in range(world):
        w = W.load(folder, rank=r, device="cpu")
        k = w.layers[0].kda
        assert k.q is None and k.k is None and k.v is None and k.qkv is not None
        assert k.qkv.n == 3 * LL * ld and k.qkv.k == d
        assert k.conv.shape == (3 * LL * ld, CFG_TP4["short_conv_kernel_size"])
        assert k.f.n == k.g.n == LL * ld and k.o.k == LL * ld and k.heads == LL
        assert k.b.weight.shape == (LL, d) and k.a_log.shape == (LL,)
        a = w.layers[2].mla
        assert a.q_b.n == LL * CFG_TP4["qk_head_dim"] and a.kv_k.weight.shape[0] == LL * CFG_TP4["qk_nope_head_dim"]
        assert w.layers[0].mlp.width == CFG_TP4["intermediate_size"] // world
        widths = {(l.moe.experts.width, l.moe.shared.width) for l in w.layers[1:] + [w.mtp.layer] if l.moe}
        assert widths == {(width[r], width[r])}
        for l in w.layers[1:]:
            if l.moe is not None:
                dt, suh, svh = l.moe.experts.parts["down_proj"][0]
                assert dt.shape[0] * 16 == suh.shape[0] == width[r]
                gt, gsuh, gsvh = l.moe.experts.parts["gate_proj"][0]
                assert gt.shape[1] * 16 == gsvh.shape[0] == width[r]
        assert w.head.n == CFG_TP4["vocab_size"] // world and w.vocab_offset == r * CFG_TP4["vocab_size"] // world
        got.append(w)
    # the ranks' fused conv rows, put back by segment, are the checkpoint's whole conv
    full = _full(folder)["model.layers.0.attention.conv1d.weight"][0].view(np.uint16).reshape(3, H * ld, -1)
    parts = [g.layers[0].kda.conv.view(torch.int16).numpy().view(np.uint16).reshape(3, LL * ld, -1) for g in got]
    assert np.array_equal(np.concatenate(parts, axis=1), full)


# ---------------------------------------------------------------- GLM unchanged at TP2 and TP4


GLM53 = {"qk_dim": 256, "qk_nope": 192, "v_dim": 256}           # GLM-5.3's config: kv_b 448 rows a head
GLM_FLASH = {"qk_dim": 256, "qk_nope": 256, "v_dim": 256}       # GLM-5.3-Flash's (no rope slice): kv_b 512
GLM_SHAPES = [  # (name, shape, kind, head sizes): GLM-5.3 and GLM-5.3-Flash tensors at their real dims
    ("model.layers.5.mlp.experts.0.gate_proj.trellis", [384, 128, 48], "dim1", GLM53),
    ("model.layers.5.mlp.experts.0.gate_proj.svh", [2048], "row", GLM53),
    ("model.layers.5.mlp.experts.0.down_proj.trellis", [128, 384, 48], "row", GLM53),
    ("model.layers.5.mlp.experts.0.down_proj.suh", [2048], "row", GLM53),
    ("model.layers.5.mlp.shared_experts.gate_proj.trellis", [384, 128, 48], "dim1", GLM53),
    ("model.layers.5.mlp.shared_experts.down_proj.suh", [2048], "row", GLM53),
    ("model.layers.0.mlp.gate_proj.trellis", [384, 768, 48], "dim1", GLM53),
    ("model.layers.0.mlp.down_proj.trellis", [768, 384, 48], "row", GLM53),
    ("model.layers.5.self_attn.q_b_proj.trellis", [128, 1024, 48], "dim1", GLM53),
    ("model.layers.5.self_attn.q_b_proj.svh", [16384], "row", GLM53),
    ("model.layers.5.self_attn.kv_b_proj.weight", [28672, 512], "row", GLM53),
    ("model.layers.5.self_attn.o_proj.trellis", [1024, 384, 48], "row", GLM53),
    ("model.layers.5.self_attn.o_proj.suh", [16384], "row", GLM53),
    ("model.layers.1.self_attn.q_proj.trellis", [256, 256, 64], "dim1", GLM_FLASH),
    ("model.layers.1.self_attn.q_conv1d.weight", [4096, 1, 4], "row", GLM_FLASH),
    ("model.layers.1.self_attn.o_proj.trellis", [256, 256, 64], "row", GLM_FLASH),
    ("model.layers.1.mlp.experts.0.gate_proj.trellis", [256, 128, 64], "dim1", GLM_FLASH),
    ("model.layers.1.mlp.experts.0.down_proj.trellis", [128, 256, 64], "row", GLM_FLASH),
    ("model.layers.3.self_attn.q_b_proj.weight", [16384, 1536], "row", GLM_FLASH),
    ("model.layers.3.self_attn.kv_b_proj.weight", [32768, 512], "row", GLM_FLASH),
]


@pytest.mark.parametrize("world", [2, 4])
def test_glm_tensors_split_byte_for_byte_as_before(world):
    """Whole blocks and whole heads leave every GLM axis at TP2 and TP4 on the plain balanced share (the pre-Ling
    split): the same spans, so the same bytes."""

    for name, shape, kind, hint in GLM_SHAPES:
        assert split.rule(name) == kind, name
        assert split.segments(name) == 1
        n = shape[0] if kind == "row" else shape[1]
        u = split.unit(name, shape, kind, world, hint)
        for r in range(world):
            assert split.spans(n, world, r, u) == [(share_lo(n, world, r), share(n, world, r))], (name, r)
    assert split.unit("model.layers.5.self_attn.kv_b_proj.weight", [28672, 512], "row", world, GLM53) == 448


@pytest.mark.skipif(not GLM_EXL3, reason="the banked GLM-5.3-Flash EXL3 checkpoint is not on this host")
def test_the_glm_flash_exl3_bank_splits_as_before_at_tp2_and_tp4():
    """Every tensor of the banked GLM-5.3-Flash EXL3 conversion: the new spans are the plain share at TP2 and TP4."""

    from ling_split_check import headers

    folder = GLM_EXL3[0]
    names = headers(folder)
    glm = split.cfg_hint(folder)
    checked = 0
    for name, (dtype, shape) in names.items():
        kind = split.rule(name)
        if kind in ("rep", "drop") or not shape:
            continue
        n = shape[0] if kind == "row" else shape[1]
        for world in (2, 4):
            u = split.unit(name, shape, kind, world, glm)
            for r in range(world):
                assert split.spans(n, world, r, u, split.segments(name)) == \
                    [(share_lo(n, world, r), share(n, world, r))], (name, world, r)
        checked += 1
    assert checked > 1000


def test_capacity_estimate_prices_rank_zero_in_whole_blocks():
    """geometry.split_weights with split.unit: rank 0's Ling expert share at TP4 is 384 columns (3 blocks), not 320."""

    from tensorfold.cuda.geometry import split_weights

    est = split_weights(split.rule, 4, unit=split.unit)
    tr = {"dtype": "I16", "shape": [256, 80, 48]}
    assert est("model.layers.4.mlp.experts.0.gate_proj.trellis", tr)[0] == 256 * 24 * 48 * 2
    assert split_weights(split.rule, 4)("model.layers.4.mlp.experts.0.gate_proj.trellis", tr)[0] == 256 * 20 * 48 * 2
    assert est("model.layers.4.mlp.experts.0.gate_proj.svh", {"dtype": "F16", "shape": [1280]})[0] >= 384 * 2


def test_tp4_pricing_equals_rank_zeros_loaded_expert_bytes(tmp_path, monkeypatch):
    """The engine's pricing matches what rank 0 loads: split_weights(rule, 4, unit=unit) summed over the routed
    and shared expert tensors (trellis + suh + svh on both sides; the mul1 codebook markers on neither) equals
    rank 0's loaded expert bytes, and without unit= it prices 320-column experts - 64 columns a matrix short of
    rank 0's 384 (4 trellis tiles + 64 scale entries)."""

    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU-only loader test")
    from bailing_fakes import CFG_TP4, write_checkpoint

    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.bailing_v3 import check
    from tensorfold.families.bailing_v3.cuda import weights as W

    folder = write_checkpoint(tmp_path, cfg=CFG_TP4, fused_qkv=True)
    check(folder)
    monkeypatch.setenv("TF_TP_WORLD", "4")
    header, _ = split.read_header(folder / "model.safetensors")

    def expert_parts(name: str) -> bool:
        return (".mlp.experts." in name or ".mlp.shared_experts." in name) \
            and name.rsplit(".", 1)[-1] in ("trellis", "suh", "svh")

    priced = sum(split_weights(split.rule, 4, unit=split.unit)(n, header[n])[0]
                 for n in header if n != "__metadata__" and expert_parts(n))
    w = W.load(folder, rank=0, device="cpu")
    moes = [l.moe for l in w.layers if l.moe is not None] + [w.mtp.layer.moe]
    routed = sum(t.numel() * t.element_size() for m in moes for rows in m.experts.parts.values()
                 for triple in rows for t in triple)
    # the shared expert's raw parts live inside its X3 groups: measure them with the loader's own reader
    rd = split.RankReader(folder, 0, cfg_hint=split.cfg_hint(folder))
    try:
        shared = sum(rd.get(n).numel() * rd.get(n).element_size() for n in header
                     if ".mlp.shared_experts." in n and expert_parts(n))
    finally:
        rd.close()
    assert priced == routed + shared, (priced, routed, shared)
    # without unit= every one of the 45 expert matrices (3 projections x 4 routed + 1 shared on the 3 MoE
    # layers) prices 4 trellis tiles (64 columns) and 64 scale entries short
    plain = sum(split_weights(split.rule, 4)(n, header[n])[0]
                for n in header if n != "__metadata__" and expert_parts(n))
    matrices = 3 * (CFG_TP4["num_experts"] + CFG_TP4["num_shared_experts"]) * 3
    assert plain == priced - matrices * (4 * 8 * 64 * 2 + 64 * 2), (priced, plain)
    # one matrix, concretely: rank 0's gate trellis prices 24 tiles (384 columns) with unit=, 20 (320) without
    tr = header["model.layers.1.mlp.experts.0.gate_proj.trellis"]
    assert split_weights(split.rule, 4, unit=split.unit)(
        "model.layers.1.mlp.experts.0.gate_proj.trellis", tr)[0] == 8 * 24 * 64 * 2
    assert split_weights(split.rule, 4)("model.layers.1.mlp.experts.0.gate_proj.trellis", tr)[0] == 8 * 20 * 64 * 2
