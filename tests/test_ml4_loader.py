"""CPU tests for the Mistral-Large-4 EXL3 pack loader (note D3 §1-§4, milestone 1).

Run directly, no pytest and no numpy needed (the host-test venv is Python 3.9 + torch CPU only):

    $ML4_HOST_TESTS/.venv/bin/python tests/test_ml4_loader.py

pytest on a full dev box collects the same test_* functions. Everything here runs without a GPU and without the
real 315 GiB pack: the fake pack folders write real safetensors HEADERS (payloads are declared, never read —
only the 4-byte codebook markers carry bytes), and the expected-shape arithmetic is checked against the real
72-layer config dimensions and D3 §4's exact numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch                                                     # noqa: E402

from tensorfold.families import quant_method, read_config        # noqa: E402
from tensorfold.families.mistral_large4 import (MODEL_TYPES, PHASE1_MESSAGE, TITLE, check,  # noqa: E402
                                                cuda_engine)
from tensorfold.families.mistral_large4.config import VOCAB, Cfg, kv_sharing_schedule      # noqa: E402
from tensorfold.families.mistral_large4.cuda import weights as W                           # noqa: E402


# -- fixtures: the real 72-layer schedule (PACK-LAYOUT.md §Per-layer variants) -------------------------------------

def rope_flags(n: int) -> list[int]:
    return [int((i + 1) % 4 != 0) for i in range(n)]             # NoPE at 3, 7, ..., 71 (value 0)


def full_lists(n: int) -> dict:
    idx = ["full"] * n
    for l in range(17, n):                                       # shared pairs 17-18, 21-22, ..., 69-70
        if (l - 17) % 4 in (0, 1):
            idx[l] = "shared"
    kv = list(range(n))
    for l in range(39, n, 4):                                    # readers 39, 43, ..., 71 read layer 35
        kv[l] = 35
    return {"no_rope_layers": rope_flags(n)[:n], "kv_source_layers": kv, "indexer_types": idx,
            "mlp_layer_types": ["dense"] + ["sparse"] * (n - 1), "layer_types": ["indexed_attention"] * n}


QC_FULL = {"quant_method": "exl3", "version": 1, "bits": 2.5, "ml4_bits": dict(W.EXPECTED_BITS),
           "head_bits": 6, "calibration": dict(W.EXPECTED_CALIBRATION), "codebook": "mul1"}


def ml4_config(n: int = 72, **over) -> dict:
    tc = {"model_type": "mistral_large4_text", "vocab_size": VOCAB, "hidden_size": 6144,
          "num_attention_heads": 64, "kv_lora_rank": 512, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64,
          "v_head_dim": 128, "num_hidden_layers": n, "index_n_heads": 32, "index_head_dim": 128,
          "index_topk": 2048, "hidden_act": "silu", "intermediate_size": 24576, "moe_intermediate_size": 6144,
          "n_routed_experts": 128, "num_experts_per_tok": 4, "n_shared_experts": 1, "rms_norm_eps": 1e-6,
          "first_k_dense_replace": 1, "tie_word_embeddings": False, "final_logit_softcapping": 30.0,
          "rope_scaling": {"rope_type": "yarn", "factor": 2048.0, "original_max_position_embeddings": 512,
                           "attention_factor": 1.0, "rope_theta": 1000000.0}}
    tc.update(full_lists(n))
    for k, v in over.items():
        if v is None:
            tc.pop(k, None)
        else:
            tc[k] = v
    return {"model_type": "mistral_large4", "text_config": tc, "quantization_config": dict(QC_FULL)}


#: the parity pack shape (PACK-LAYOUT §Partial packs): lists cut to 8, reader 7 -> source 3 (both NoPE)
def ml4_config_partial() -> dict:
    l8 = full_lists(8)
    l8["kv_source_layers"][7] = 3
    return ml4_config(8, **l8)


#: a small but complete layer set for fake packs: dense+rope, sparse+rope, sparse+NoPE writer, reader+shared
def tiny_cfg_dict() -> dict:
    return ml4_config(4, hidden_size=128, num_attention_heads=2, kv_lora_rank=32, qk_nope_head_dim=16,
                      qk_rope_head_dim=16, v_head_dim=16, index_n_heads=2, index_head_dim=16, index_topk=8,
                      intermediate_size=48, moe_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2,
                      no_rope_layers=[1, 1, 0, 0], kv_source_layers=[0, 1, 2, 2],
                      indexer_types=["full", "full", "full", "shared"],
                      mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
                      final_logit_softcapping=0.0)


def expect_error(fn, what: str, exc=Exception, needle: str | None = None) -> None:
    try:
        fn()
    except exc as e:
        if needle is not None:
            assert needle in str(e), f"{what}: error must name {needle!r}, got: {e}"
        return
    raise AssertionError(f"{what}: expected {exc.__name__}")


# -- Cfg (D3 §1) -------------------------------------------------------------------------------------------------

def test_cfg_full_72():
    cfg = Cfg.from_dict(ml4_config())
    assert (cfg.n_layers, cfg.dim, cfg.n_heads, cfg.vocab) == (72, 6144, 64, 204800)
    assert (cfg.kv_lora, cfg.nope, cfg.rd, cfg.v_head, cfg.qk_head) == (512, 64, 64, 128, 128)
    assert cfg.q_out == 8192 and cfg.o_in == 8192 and cfg.sm_scale == 128 ** -0.5
    assert cfg.softcap == 30.0 and cfg.route_scale == 1.0 and cfg.q_rank is None
    assert (cfg.rope_factor, cfg.orig_len, cfg.attention_factor) == (2048.0, 512, 1.0)
    assert cfg.is_rope(0) and not cfg.is_rope(3) and not cfg.is_rope(71)
    assert [i for i in range(72) if cfg.is_reader(i)] == list(range(39, 72, 4))
    assert cfg.kv_sources == {35}
    assert cfg.kv_a_rows(0) == 576 and cfg.kv_a_rows(3) == 512 and cfg.kv_a_rows(39) == 512
    assert cfg.kv_b_rows(0) == 12288 and cfg.kv_b_rows(3) == 16384          # PACK-LAYOUT §Per-layer variants
    assert cfg.is_dense(0) and not cfg.is_dense(1)
    assert not cfg.is_full_indexer(17) and not cfg.is_full_indexer(18) and cfg.is_full_indexer(71)
    assert cfg.is_full_indexer(39)                                          # readers keep their own indexer
    assert sum(1 for i in range(72) if cfg.is_dense(i)) == 1


def test_cfg_partial_pack():
    cfg = Cfg.from_dict(ml4_config_partial())
    assert cfg.n_layers == 8 and cfg.kv_source_layers == [0, 1, 2, 3, 4, 5, 6, 3]
    assert cfg.kv_sources == {3} and cfg.is_reader(7) and not cfg.is_reader(3)
    assert len(cfg.indexer_types) == 8 and len(cfg.rope_flags) == 8
    assert kv_sharing_schedule([0, 1, 2, 3, 4, 5, 6, 3], rope_flags(8), 8) == {3}


def test_cfg_rejects():
    bad = [("q_lora_rank", 1536, "q_lora_rank"), ("vocab_size", 128256, "204800"), ("n_group", 8, "n_group"),
           ("routed_scaling_factor", 2.5, "routed_scaling_factor"), ("hidden_act", "gelu", "silu"),
           ("indexer_norm_type", "rmsnorm", "LayerNorm"), ("no_rope_layers", [2] * 72, "0/1"),
           ("layer_types", ["sliding_attention"] * 72, "layer_types")]
    for key, val, needle in bad:
        expect_error(lambda: Cfg.from_dict(ml4_config(**{key: val})), f"Cfg rejects {key}", needle=needle)
    idx = full_lists(72)["indexer_types"]
    idx[0] = "shared"
    expect_error(lambda: Cfg.from_dict(ml4_config(indexer_types=idx)), "indexer 0 must be full",
                 needle="indexer_types")
    expect_error(lambda: Cfg.from_dict(ml4_config(indexer_types=full_lists(72)["indexer_types"][:-1])),
                 "short schedule", needle="entries")
    kv = list(range(72))
    kv[0] = 5                                                        # forward source
    expect_error(lambda: Cfg.from_dict(ml4_config(kv_source_layers=kv)), "forward KV source",
                 needle="earlier layer")
    kv2 = full_lists(72)["kv_source_layers"]
    kv2[40] = 35                                                     # rope layer reading a NoPE source
    expect_error(lambda: Cfg.from_dict(ml4_config(kv_source_layers=kv2)), "rope-type mismatch",
                 needle="rope type")
    expect_error(lambda: Cfg.from_dict(ml4_config(rope_scaling={**ml4_config()["text_config"]["rope_scaling"],
                                                               "factor": 4.0})), "YaRN factor", needle="2048")
    expect_error(lambda: Cfg.from_dict(ml4_config(indexer_types=None)), "indexer_types required", exc=KeyError)


# -- EP4 split (D2 §2 / D3 §2) -------------------------------------------------------------------------------------

def test_local_ids():
    assert W.local_ids(0) == list(range(0, 32)) and W.local_ids(3) == list(range(96, 128))
    assert W.local_ids(2)[0] == 64 and W.local_ids(2)[-1] == 95              # default e // 32
    all_ids = [e for r in range(4) for e in W.local_ids(r)]
    assert sorted(all_ids) == list(range(128))                               # whole experts, each exactly once
    assert all(len(W.local_ids(r)) == 32 for r in range(4))
    assert W.local_ids(1, world=4, n_experts=128, rule="interleave") == list(range(1, 128, 4))
    assert W.local_ids(1, world=4, n_experts=8) == [2, 3]
    expect_error(lambda: W.local_ids(4), "rank out of world")
    expect_error(lambda: W.local_ids(0, world=3, n_experts=128), "experts must divide into ranks")
    expect_error(lambda: W.local_ids(0, rule="hash"), "unknown rule")


# -- name shim completeness (D3 §1, templates from PACK-LAYOUT.md) --------------------------------------------------

PACK_LAYER_GROUPS = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                     "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
                     "mlp.shared_experts.down_proj", "mlp.experts.{e}.gate_proj", "mlp.experts.{e}.up_proj",
                     "mlp.experts.{e}.down_proj"]
PACK_LAYER_NQ = ["input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight",
                 "self_attn.kv_b_proj.weight", "self_attn.softmax_scaling.scale", "self_attn.softmax_scaling.bias",
                 "self_attn.indexer.wq_b.weight", "self_attn.indexer.wq_b.weight_scale_inv",
                 "self_attn.indexer.wk.weight", "self_attn.indexer.weights_proj.weight",
                 "self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
                 "self_attn.indexer.q_input_norm.weight", "self_attn.indexer.q_input_norm.bias",
                 "self_attn.indexer.softmax_scaling.scale", "self_attn.indexer.softmax_scaling.bias",
                 "mlp.gate.weight"]


def test_name_shim_complete():
    want = {g: ("exl3",) for g in PACK_LAYER_GROUPS}
    want.update({n: () for n in PACK_LAYER_NQ})
    assert set(W.LAYER_SHIM) == set(want), (f"missing {sorted(set(want) - set(W.LAYER_SHIM))}, "
                                            f"extra {sorted(set(W.LAYER_SHIM) - set(want))}")
    assert set(W.GLOBALS_SHIM) == {f"{W.LM}.embed_tokens.weight", f"{W.LM}.rotary_emb.alpha", f"{W.LM}.norm.weight",
                                   "lm_head.weight"}
    for g in PACK_LAYER_GROUPS:
        assert W.LAYER_SHIM[g][0] == "exl3"
    casts = {W.LAYER_SHIM[n][0] for n in PACK_LAYER_NQ}
    assert casts == {"f32", "bf16", "fp8deq", "fp8scale"}, "only widening casts exist (D3 §2)"


def test_marker_constants_pinned():
    try:
        from tensorfold.cuda.exl3 import format as exl3_format
    except ImportError:                               # the host venv has no numpy; the literals still run below
        exl3_format = None
    if exl3_format is not None:
        assert exl3_format.MARKERS == W.MARKER_VALUE
    assert W.MARKER_VALUE["mul1"] == 0x83DCD12D      # D3 §4: the mul1 marker value
    assert W.PACK_VERSION == 1 and W.EXPECTED_BITS == {"experts": 2.5, "shared": 5.0, "attn": 5.0, "dense": 5.0}
    assert W.EXPECTED_CALIBRATION == {"rows": 128, "cols": 2048}


# -- expected tensor sets (D3 §2, shapes per D3 §4) ------------------------------------------------------------------

def test_expected_sets_real_dims():
    cfg = Cfg.from_dict(ml4_config())
    bits, cb = dict(W.EXPECTED_BITS), "mul1"
    q0 = W.expected_layer_quant(cfg, bits, cb, 0)
    p = W.LP.format(L=0)
    assert q0[p + "self_attn.q_proj.trellis"] == ("I16", (384, 512, 80))          # D3 §4 numbers, derived
    assert q0[p + "self_attn.o_proj.trellis"] == ("I16", (512, 384, 80))
    assert q0[p + "mlp.gate_proj.trellis"] == ("I16", (384, 1536, 80))            # dense L0
    assert q0[p + "mlp.down_proj.trellis"] == ("I16", (1536, 384, 80))
    assert q0[p + "mlp.gate_proj.suh"] == ("F16", (6144,)) and q0[p + "mlp.gate_proj.svh"] == ("F16", (24576,))
    assert q0[p + "mlp.gate_proj.mul1"] == ("I32", ())
    assert not any(".mcg" in n for n in q0)
    assert not any("mlp.experts." in n or "shared_experts" in n for n in q0)      # dense: no MoE groups
    q1 = W.expected_layer_quant(cfg, bits, cb, 1)
    p1 = W.LP.format(L=1)
    assert q1[p1 + "mlp.experts.127.down_proj.trellis"] == ("I16", (384, 384, 40))
    assert q1[p1 + "mlp.shared_experts.gate_proj.trellis"] == ("I16", (384, 384, 80))
    assert sum(1 for n in q1 if ".trellis" in n) == 2 + 3 + 3 * 128           # q, o, 3 shared, 384 routed
    assert not any("mlp.gate_proj.trellis" in n and "experts" not in n and "shared" not in n for n in q1)
    q3 = W.expected_layer_quant(cfg, dict(bits), "3inst", 3)
    assert not any(n.endswith(".mul1") or n.endswith(".mcg") for n in q3)         # 3inst carries no marker

    n0 = W.expected_layer_nq(cfg, 0)
    assert n0[p + "self_attn.kv_a_proj_with_mqa.weight"] == ("BF16", (576, 6144))
    assert n0[p + "self_attn.kv_b_proj.weight"] == ("BF16", (12288, 512))
    assert p + "mlp.gate.weight" not in n0                                        # dense layer: no router
    n3 = W.expected_layer_nq(cfg, 3)
    assert n3[W.LP.format(L=3) + "self_attn.kv_a_proj_with_mqa.weight"] == ("BF16", (512, 6144))
    assert n3[W.LP.format(L=3) + "self_attn.kv_b_proj.weight"] == ("BF16", (16384, 512))
    n17 = W.expected_layer_nq(cfg, 17)
    p17 = W.LP.format(L=17)
    assert not any(".indexer." in n for n in n17)                                 # shared indexer: none
    assert n17[p17 + "mlp.gate.weight"] == ("BF16", (128, 6144))
    n39 = W.expected_layer_nq(cfg, 39)
    p39 = W.LP.format(L=39)
    assert not any("kv_a" in n for n in n39)                                      # reader: no kv_a, no kv_a_norm
    assert n39[p39 + "self_attn.kv_b_proj.weight"] == ("BF16", (16384, 512))      # NoPE reader
    assert n39[p39 + "self_attn.indexer.wq_b.weight"] == ("F8_E4M3", (4096, 6144))
    assert n39[p39 + "self_attn.indexer.wq_b.weight_scale_inv"] == ("F32", (32, 48))
    assert W.expected_layer_nq(cfg, 71) is not None                               # 71: reader AND full indexer

    g = W.expected_globals(cfg, "quant")
    assert g[f"{W.LM}.embed_tokens.weight"] == ("BF16", (204800, 6144))
    assert g[f"{W.LM}.rotary_emb.alpha"] == ("BF16", (32,))
    assert "lm_head.weight" not in g
    assert W.expected_globals(cfg, "source")["lm_head.weight"] == ("BF16", (204800, 6144))
    assert W.expected_head(cfg, 6.0, cb)["lm_head.trellis"] == ("I16", (384, 12800, 96))


# -- fp8 block dequant (D3 §5.3) -------------------------------------------------------------------------------------

def _fp8(rows):
    return torch.tensor(rows, dtype=torch.float32).to(torch.float8_e4m3fn)


def test_dequant_fp8_block():
    out = W.dequant_fp8_block(_fp8([[1, 2], [3, 4]]), torch.tensor([[2.0]]), block=2)
    assert out.dtype == torch.float16 and out.shape == (2, 2)
    assert out.tolist() == [[2.0, 4.0], [6.0, 8.0]]                              # hand-computed
    out = W.dequant_fp8_block(_fp8([[1, 2], [3, 4], [5, 6]]), torch.tensor([[2.0], [0.5]]), block=2)
    assert out.tolist() == [[2.0, 4.0], [6.0, 8.0], [2.5, 3.0]]                  # per-row-block scale
    out = W.dequant_fp8_block(_fp8([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.tensor([[1.0, 2.0]]), block=2)
    assert out.tolist() == [[1.0, 2.0, 6.0, 8.0], [5.0, 6.0, 14.0, 16.0]]        # per-col-block scale
    expect_error(lambda: W.dequant_fp8_block(_fp8([[1, 2, 3, 4]]), torch.tensor([[1.0], [1.0]]), block=2),
                 "wrong grid size", needle="2x2")
    expect_error(lambda: W.dequant_fp8_block(torch.randn(2, 2), torch.tensor([[2.0]]), block=2),
                 "wrong source dtype", needle="F8_E4M3")
    expect_error(lambda: W.dequant_fp8_block(_fp8([[1, 2]]), torch.tensor([[2]], dtype=torch.float16), block=2),
                 "wrong scale dtype", needle="F32")


# -- fake pack builder (header-true files; only marker tensors carry bytes) -----------------------------------------

def fake_st(path: Path, tensors: list[tuple[str, str, tuple]], mul1: int = W.MARKER_VALUE["mul1"]) -> int:
    header, data, tensor_bytes = {}, b"", 0
    for name, dt, shape in tensors:
        n = W.DT_BYTES[dt] * math.prod(shape)
        if dt == "I32" and not shape:                       # a codebook marker: real bytes, packed together
            off, data = len(data), data + struct.pack("<I", mul1)   # unsigned: 0x83DCD12D is a signed int32 too
        else:                                               # payload declared, never written (not read here)
            off = 0
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        tensor_bytes += n
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)
    return tensor_bytes


def build_fake_pack(root: Path, cfg: Cfg, head_mode: str = "quant", partial: bool = False,
                    source: int | None = None) -> None:
    cfgd = tiny_cfg_dict()
    bits = dict(W.EXPECTED_BITS)
    files, total_tensors, total_bytes = {}, 0, 0
    entries = {f"L{i:02d}.safetensors": list(W.expected_layer_quant(cfg, bits, "mul1", i).items())
               for i in range(cfg.n_layers)}
    entries.update({f"L{i:02d}-nq.safetensors": list(W.expected_layer_nq(cfg, i).items())
                    for i in range(cfg.n_layers)})
    entries["globals.safetensors"] = list(W.expected_globals(cfg, head_mode).items())
    kind = {"globals": "globals"}
    if head_mode == "quant":
        entries["head.safetensors"] = list(W.expected_head(cfg, 6.0, "mul1").items())
        kind["head"] = "exl3-head"
    for fn in sorted(entries):
        tb = fake_st(root / fn, [(n, dt, sh) for n, (dt, sh) in entries[fn]])
        f = root / fn
        files[fn] = {"kind": kind.get(fn.split(".")[0], "nq-layer" if fn.endswith("-nq.safetensors")
                          else "exl3-layer"), "tensors": len(entries[fn]), "bytes": f.stat().st_size,
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        total_tensors += len(entries[fn])
        total_bytes += tb
    manifest = {"pack_version": 1, "tool": "quant/pack_ml4.py", "created_utc": "2026-11-05T00:00:00+00:00",
                "source": "/fake/source", "quant": "/fake/quant", "layers": [0, cfg.n_layers - 1],
                "partial": partial, "num_hidden_layers_source": source or cfg.n_layers, "head": head_mode,
                "bits": bits, "calibration": dict(W.EXPECTED_CALIBRATION), "codebook": "mul1",
                "tensors": total_tensors, "tensor_bytes": total_bytes, "skipped_source_tensors": [],
                "source_tensors_outside_pack": [], "files": files}
    (root / "ml4_pack.json").write_text(json.dumps(manifest, indent=1))
    cfgd["text_config"]["num_hidden_layers"] = cfg.n_layers
    for k in ("no_rope_layers", "kv_source_layers", "indexer_types", "mlp_layer_types", "layer_types"):
        cfgd["text_config"][k] = cfgd["text_config"][k][:cfg.n_layers]
    cfgd["quantization_config"] = dict(QC_FULL, head_bits=6 if head_mode == "quant" else 16)
    (root / "config.json").write_text(json.dumps(cfgd))
    lines = [f"{files[fn]['sha256']}  {fn}" for fn in sorted(files)]
    (root / "sha256.txt").write_text("\n".join(lines) + "\n")


def fake_pack(tmp: Path, **kw) -> tuple[Path, Cfg, dict]:
    root = tmp
    cfg = Cfg.from_dict(tiny_cfg_dict())
    build_fake_pack(root, cfg, **kw)
    return root, cfg, json.loads((root / "ml4_pack.json").read_text())


def patched_pack(tmp: Path, fn: str, mutate) -> Path:
    root, cfg, _ = fake_pack(tmp)
    if fn.endswith(".safetensors"):
        tensors = [(n, dt, sh) for n, (dt, sh) in
                   (W.expected_layer_quant(cfg, dict(W.EXPECTED_BITS), "mul1", int(fn[1:3])).items()
                    if not fn.endswith("-nq.safetensors")
                    else W.expected_layer_nq(cfg, int(fn[1:3])).items())]
        mutate(tensors)
        fake_st(root / fn, tensors)
    elif fn == "ml4_pack.json":
        m = json.loads((root / fn).read_text())
        mutate(m)
        (root / fn).write_text(json.dumps(m))
    elif fn == "config.json":
        c = json.loads((root / fn).read_text())
        mutate(c)
        (root / fn).write_text(json.dumps(c))
    return root


# -- integrity gates (D3 §4) -----------------------------------------------------------------------------------------

def test_fake_pack_passes_all_gates():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root, cfg, manifest = fake_pack(Path(td))
        index = W.PackIndex(root).load()
        W.verify_manifest(manifest, json.loads((root / "config.json").read_text()), index)
        for i in range(cfg.n_layers):
            W.check_layer_headers(index, cfg, dict(W.EXPECTED_BITS), "mul1", i)
        W.check_globals_head(index, cfg, "quant", 6.0, "mul1")
        W.check_exl3_markers(index, "mul1")                       # reads the real 4-byte markers
        assert W.verify_pack(root, log=lambda *a: None)["files"] == 4 * 2 + 2
        assert check(root) is None                              # the family check() reads the same config
        assert quant_method(read_config(root)) == "exl3"


def test_partial_pack_head_rules():
    """D3 §6 revised after the node smoke (2026-11-06): extent (full vs partial) and head mode are
    ORTHOGONAL. Full accepts [0, 71] w/ quant head; partial (pack_ml4.py --layers 0-K of the real
    quant) accepts head 'quant' (SAME recipe constants) or 'source'; the head pairing — quant <->
    head.safetensors present + head_bits 6, source <-> absent + head_bits 16 — holds both ways."""

    import tempfile

    def build(td, head_mode, partial=True, source=72):            # truncated pack: 4 of 72 layers
        root = Path(td)
        cfg = Cfg.from_dict(tiny_cfg_dict())
        build_fake_pack(root, cfg, head_mode=head_mode, partial=partial, source=source)
        return root, cfg, json.loads((root / "ml4_pack.json").read_text()), \
            json.loads((root / "config.json").read_text())

    with tempfile.TemporaryDirectory() as td:                     # partial + quant head: ACCEPTED now
        root, cfg, manifest, config = build(td, "quant")
        index = W.PackIndex(root).load()
        W.verify_manifest(manifest, config, index)
        W.check_globals_head(index, cfg, "quant", 6.0, "mul1")    # head.safetensors present as the quant head
        for i in range(cfg.n_layers):                             # expected sets follow the truncated lists
            W.check_layer_headers(index, cfg, dict(W.EXPECTED_BITS), "mul1", i)
        assert W.verify_pack(root, log=lambda *a: None)["files"] == 4 * 2 + 2
    with tempfile.TemporaryDirectory() as td:                     # partial + source head: still §6-legal
        root, cfg, manifest, config = build(td, "source", source=8)
        index = W.PackIndex(root).load()
        W.verify_manifest(manifest, config, index)
        W.check_globals_head(index, cfg, "source", 16.0, "mul1")  # lm_head.weight in globals, no head file
        expect_error(lambda: W.check_globals_head(index, cfg, "quant", 6.0, "mul1"),
                     "quant mode must reject the source-head pack", needle="head mode")
    with tempfile.TemporaryDirectory() as td:                     # pairing, way 1: head file present, mode source
        root, cfg, manifest, config = build(td, "source", source=8)
        fake_st(root / "head.safetensors", [("lm_head.trellis", "I16", (8, 200, 96))])
        expect_error(lambda: W.verify_manifest(manifest, config, W.PackIndex(root).load()),
                     "unexpected head file", needle="head.safetensors")
    with tempfile.TemporaryDirectory() as td:                     # pairing, way 2: mode quant, file absent
        root, cfg, manifest, config = build(td, "source", source=72)
        manifest["head"], config["quantization_config"]["head_bits"] = "quant", 6
        expect_error(lambda: W.verify_manifest(manifest, config, W.PackIndex(root).load()),
                     "quant mode requires the head file", needle="head.safetensors")


def test_partial_pack_tampering():
    """Manifest tampering on a legitimate partial pack is still refused: layer ordering/extent,
    K != truncated-config layers - 1, and any recipe-constant drift."""

    import tempfile

    def tamper(mut, what, needle):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_fake_pack(root, Cfg.from_dict(tiny_cfg_dict()), head_mode="quant", partial=True, source=72)
            m = json.loads((root / "ml4_pack.json").read_text())
            c = json.loads((root / "config.json").read_text())
            mut(m, c)
            index = W.PackIndex(root).load()
            expect_error(lambda: W.verify_manifest(m, c, index), what, needle=needle)

    tamper(lambda m, c: m.update(layers=[3, 0]), "layers reversed", "layers")
    tamper(lambda m, c: m.update(layers=[1, 3]), "layers not covering from 0", "layers")
    tamper(lambda m, c: c["text_config"].update(num_hidden_layers=8),
           "K (3) != num_hidden_layers - 1 (7) of the truncated config", "num_hidden_layers")
    tamper(lambda m, c: m.update(layers=[0, 7]), "manifest claims 8 layers, config has 4", "num_hidden_layers")
    tamper(lambda m, c: m["calibration"].update(rows=8, cols=128), "calibration mini (the tiny-pack reject)",
           "calibration")
    tamper(lambda m, c: m["bits"].update(attn=6.0), "attn bits drift", "bits")
    tamper(lambda m, c: m.update(codebook="mcg2"), "unknown codebook", "codebook")
    tamper(lambda m, c: c["quantization_config"].update(head_bits=16), "quant head keeps head_bits 6",
           "head_bits")
    tamper(lambda m, c: m.update(partial="yes"), "partial must be a bool", "partial")


def test_integrity_rejections():
    import tempfile

    cases = []

    def drop(tensors):
        tensors[:] = [t for t in tensors if "kv_a_proj" not in t[0]]

    def add(tensors):
        tensors.append((W.LP.format(L=3) + "self_attn.kv_a_proj_with_mqa.weight", "BF16", (32, 128)))

    def wrongdt(tensors):
        tensors[:] = [(n, "F32" if n.endswith("input_layernorm.weight") else dt, sh) for n, dt, sh in tensors]

    def wrongshape(tensors):
        tensors[:] = [(n, dt, (7, 128) if "kv_b_proj" in n else sh) for n, dt, sh in tensors]

    for label, fn, mut, needle in (
            ("missing tensor (writer's kv_a)", "L00-nq.safetensors", drop, "kv_a_proj"),
            ("unexpected tensor (reader carries kv_a)", "L03-nq.safetensors", add, "unexpected"),
            ("wrong dtype (norm as F32)", "L00-nq.safetensors", wrongdt, "input_layernorm"),
            ("wrong shape (kv_b rows)", "L00-nq.safetensors", wrongshape, "kv_b_proj"),
            ("shared indexer carries indexer.*", "L03-nq.safetensors",
             lambda t: t.append((W.LP.format(L=3) + "self_attn.indexer.wk.weight", "BF16", (16, 128))),
             "indexer.wk")):
        with tempfile.TemporaryDirectory() as td:
            root = patched_pack(Path(td), fn, mut)
            cfg = Cfg.from_dict(tiny_cfg_dict())
            index = W.PackIndex(root).load()
            i = int(fn[1:3])
            cases.append((label, lambda i=i, index=index, cfg=cfg, needle=needle: expect_error(
                lambda: W.check_layer_headers(index, cfg, dict(W.EXPECTED_BITS), "mul1", i), label, needle=needle)))
    for label, fn, mut, needle in (
            ("manifest pack_version 2", "ml4_pack.json", lambda m: m.update(pack_version=2), "pack_version"),
            ("manifest bits recipe", "ml4_pack.json", lambda m: m["bits"].update(experts=3.0), "bits"),
            ("manifest tensors count", "ml4_pack.json", lambda m: m.update(tensors=m["tensors"] - 1), "tensors"),
            ("manifest tensor_bytes", "ml4_pack.json", lambda m: m.update(tensor_bytes=m["tensor_bytes"] + 1),
             "tensor_bytes"),
            ("manifest missing sha256", "ml4_pack.json", lambda m: m["files"]["globals.safetensors"].pop("sha256"),
             "sha256"),
            ("manifest names a missing file", "ml4_pack.json",
             lambda m: m["files"].update({"L42.safetensors": {"kind": "exl3-layer", "tensors": 0, "bytes": 1,
                                                              "sha256": "0" * 64}}), "L42.safetensors"),
            ("config ml4_bits != manifest", "config.json", lambda c: c["quantization_config"].update(
                ml4_bits={"experts": 3.0}), "ml4_bits"),
            ("config head_bits mismatch", "config.json", lambda c: c["quantization_config"].update(head_bits=8),
             "head_bits"),
            ("file truncated", "TRUNCATE", None, "size")):
        with tempfile.TemporaryDirectory() as td:
            root = patched_pack(Path(td), fn, mut) if fn != "TRUNCATE" else fake_pack(Path(td))[0]
            if fn == "TRUNCATE":
                p = root / "L01.safetensors"
                p.write_bytes(p.read_bytes()[:-8])
            manifest = json.loads((root / "ml4_pack.json").read_text())
            config = json.loads((root / "config.json").read_text())
            index = W.PackIndex(root).load()
            cases.append((label, lambda m=manifest, c=config, ix=index, needle=needle: expect_error(
                lambda: W.verify_manifest(m, c, ix), label, needle=needle)))
    with tempfile.TemporaryDirectory() as td:                     # a marker carrying the wrong bit pattern
        root, cfg, _ = fake_pack(Path(td))
        tensors = [(n, dt, sh) for n, (dt, sh) in
                   W.expected_layer_quant(cfg, dict(W.EXPECTED_BITS), "mul1", 0).items()]
        fake_st(root / "L00.safetensors", tensors, mul1=0x12345678)
        index = W.PackIndex(root).load()
        expect_error(lambda: W.check_exl3_markers(index, "mul1"), "bad mul1 marker value", exc=ValueError,
                     needle="marker")
    for label, fn in cases:
        fn()


def test_verify_pack_sha():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root, _, _ = fake_pack(Path(td))
        assert W.verify_pack(root, log=lambda *a: None)["files"] == 10
        p = root / "L00.safetensors"
        p.write_bytes(p.read_bytes() + b"x")                    # a byte changed in transit
        expect_error(lambda: W.verify_pack(root, log=lambda *a: None), "tampered file", exc=ValueError,
                     needle="sha256 mismatch")


def test_verify_pack_file_sets():
    """pack_ml4.py hashes the WHOLE pack dir into sha256.txt (docstring) — config.json and
    model.safetensors.index.json included, which ml4_pack.json files[] never names (the real full
    pack's exact shape). verify_pack accepts exactly those two benign extras; every real drift —
    missing file, missing hash line, unnamed extra, stray .safetensors — still fails."""

    import hashlib
    import shutil
    import tempfile

    def add_lines(root):                                        # the packer's real output shape
        (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0},
                                                                       "weight_map": {}}))
        lines = (root / "sha256.txt").read_text().splitlines()
        for fn in ("config.json", "model.safetensors.index.json"):
            lines.append(f"{hashlib.sha256((root / fn).read_bytes()).hexdigest()}  {fn}")
        (root / "sha256.txt").write_text("\n".join(lines) + "\n")

    with tempfile.TemporaryDirectory() as td:                     # benign extras: PASS (what the real pack ships)
        root, _, _ = fake_pack(Path(td))
        add_lines(root)
        assert W.verify_pack(root, log=lambda *a: None)["files"] == 12
    with tempfile.TemporaryDirectory() as td:                     # a file the pack folder never got: FAIL
        (Path(td) / "full").mkdir()
        root, _, _ = fake_pack(Path(td) / "full")
        add_lines(root)
        partial_root = Path(td) / "short"
        partial_root.mkdir()                                      # copy-everything-except: creates, deletes nothing
        for src in sorted(root.iterdir()):
            if src.name != "L02.safetensors":
                shutil.copyfile(src, partial_root / src.name)
        expect_error(lambda: W.verify_pack(partial_root, log=lambda *a: None), "genuinely missing file",
                     needle="missing from the pack")
    with tempfile.TemporaryDirectory() as td:                     # hash line dropped from sha256.txt: FAIL
        root, _, _ = fake_pack(Path(td))
        add_lines(root)
        lines = (root / "sha256.txt").read_text().splitlines()
        (root / "sha256.txt").write_text("\n".join(l for l in lines if not l.endswith("  L02.safetensors")) + "\n")
        expect_error(lambda: W.verify_pack(root, log=lambda *a: None), "manifest file not hashed",
                     needle="sha256.txt does not hash")
    with tempfile.TemporaryDirectory() as td:                     # unnamed extra beyond the two loader inputs: FAIL
        root, _, _ = fake_pack(Path(td))
        (root / "EXTRA.md").write_text("stray")
        lines = (root / "sha256.txt").read_text().splitlines()
        lines.append(f"{hashlib.sha256((root / 'EXTRA.md').read_bytes()).hexdigest()}  EXTRA.md")
        (root / "sha256.txt").write_text("\n".join(lines) + "\n")
        expect_error(lambda: W.verify_pack(root, log=lambda *a: None), "unnamed listed file",
                     needle="does not name")
    with tempfile.TemporaryDirectory() as td:                     # stray .safetensors in the folder: FAIL
        root, _, _ = fake_pack(Path(td))
        (root / "L99.safetensors").write_bytes(b"stale layer file")
        expect_error(lambda: W.verify_pack(root, log=lambda *a: None), "stray safetensors",
                     needle="not named in ml4_pack.json")


# -- per-rank plan (D3 §2/§3/§5.4) -------------------------------------------------------------------------------------

def test_rank_plan_partitions():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root, cfg, _ = fake_pack(Path(td))
        index = W.PackIndex(root).load()
        plans = [W.rank_plan(cfg, "mul1", index, r, 4, "quant") for r in range(4)]
        for r, pl in enumerate(plans):
            assert pl["expert_ids"] == [r] and pl["read_bytes"] == pl["routed_bytes"] + pl["replicated_bytes"]
            assert pl["cache_bytes"] >= pl["routed_bytes"]      # f32 widening grows the kept bytes
        routed_all = sum(e[3] for n, e in index.entries.items() if ".experts." in n and not n.endswith(".mul1"))
        assert sum(pl["routed_bytes"] for pl in plans) == routed_all     # every expert byte held by exactly one
        assert len({pl["replicated_bytes"] for pl in plans}) == 1        # replicated is identical a rank
        # D3 §5.4: the estimate is not total/world — replicated bytes never divide by world
        total = sum(e[3] for e in index.entries.values())
        assert plans[0]["read_bytes"] > total / 4
        expect_error(lambda: W.rank_plan(cfg, "mul1", _with_junk(index), 0, 4, "quant"),
                     "unknown tensor aborts", needle="outside the rank plan")


def _with_junk(index: W.PackIndex) -> W.PackIndex:
    junk = W.PackIndex(index.root)
    junk.entries = dict(index.entries)
    junk.entries["model.language_model.rotary_emb.beta"] = ("globals.safetensors", "BF16", (8,), 16, 0)
    junk.file_sizes = index.file_sizes
    return junk


# -- family registration (phase-1 contract) ----------------------------------------------------------------------------

def test_family_stub():
    assert MODEL_TYPES == ("mistral_large4_text", "mistral_large4") and TITLE == "Mistral-Large-4"
    # phase 2b: cuda_engine is the family-owned rank dispatch (D1 §Extension points) — argument gates only,
    # no pack touched; the engine's own CPU build/serve gates are tests/test_ml4_engine.py
    expect_error(lambda: cuda_engine("irrelevant", tp=5), "tp gate at 4 ranks", needle="1 to 4 ranks")
    expect_error(lambda: cuda_engine("irrelevant", tp=4), "tp>1 needs master", needle="--master")
    expect_error(lambda: cuda_engine("irrelevant", drafter="dspark"), "no drafter", needle="no drafter")
    assert "loader" in PHASE1_MESSAGE
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        build_fake_pack(root, Cfg.from_dict(tiny_cfg_dict()))
        expect_error(lambda: check(root / "nope"), "missing dir", exc=Exception)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed")
    sys.exit(1 if fails else 0)
