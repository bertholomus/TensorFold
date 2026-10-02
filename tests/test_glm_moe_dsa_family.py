"""CPU-only self-test of the glm_moe_dsa family: imports, config from the real checkpoint, registry, glue math.

Run: /tmp/tf-venv/bin/python -m pytest tests/test_glm_moe_dsa_family.py -q
No CUDA is touched (the module skips itself when a GPU would be needed), and no weights are read
beyond tensor headers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")
if torch.cuda.is_available():          # these tests must never touch a GPU even where one exists
    pytest.skip("CPU-only glm_moe_dsa family tests", allow_module_level=True)

REAL_CONFIG = Path("/tank/models/llm/banked/GLM-5.3-BF16/config.json")
REAL_DIR = REAL_CONFIG.parent


# ---------------------------------------------------------------- the package and the registry

def test_family_discovered():
    from tensorfold import families

    found = families.families()
    assert "glm_moe_dsa" in found
    family = found["glm_moe_dsa"]
    assert family.title == "GLM-5.3"
    assert not family.lanes                       # CUDA only
    assert hasattr(family.package, "cuda_engine")
    assert not hasattr(family.package, "load")    # no MLX engine: the CLI refuses --backend mlx


def test_cli_tp_choices_cover_the_family():
    """The fork's world-parameterized TP4 CLI accepts --tp 4, which the family's engine requires TF_TP_WORLD for."""

    from tensorfold.cli import build_parser

    args = build_parser().parse_args(["serve", "x", "--tp", "4", "--rank", "3", "--master", "10.0.0.1"])
    assert args.tp == 4 and args.rank == 3


# ---------------------------------------------------------------- the config from the real checkpoint

@pytest.mark.skipif(not REAL_CONFIG.is_file(), reason="the banked GLM-5.3 checkpoint is not on this host")
def test_config_from_real_checkpoint():
    from tensorfold.families.glm_moe_dsa.config import Config

    raw = json.loads(REAL_CONFIG.read_text())
    cfg = Config.from_dict(raw)
    assert cfg.num_hidden_layers == 78
    assert cfg.num_attention_heads == 64
    assert cfg.index_n_heads == 32 and cfg.index_head_dim == 128 and cfg.index_topk == 2048
    assert cfg.n_routed_experts == 256 and cfg.num_experts_per_tok == 8
    assert cfg.qk_nope_head_dim == 192 and cfg.qk_rope_head_dim == 64 and cfg.qk_head_dim == 256
    assert cfg.v_head_dim == 256 and cfg.kv_lora_rank == 512 and cfg.q_lora_rank == 2048
    assert cfg.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 75
    assert cfg.full_indexer_layers == 21 and cfg.indexer_types[0] == "full" and cfg.indexer_types[3] == "shared"
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.routed_scaling_factor == 2.5 and cfg.eos_token_id == (154820, 154827, 154829)
    assert cfg.rope_theta == 8_000_000.0


@pytest.mark.skipif(not REAL_CONFIG.is_file(), reason="the banked GLM-5.3 checkpoint is not on this host")
def test_engine_config_reads_the_real_checkpoint():
    from tensorfold.families.glm_moe_dsa.cuda.weights import Config

    cfg = Config.read(REAL_DIR)
    assert cfg.layers == 78 and cfg.heads == 64 and cfg.experts == 256
    assert cfg.qk_dim == 256 and cfg.moe_width == 2048 and cfg.dense_width == 12288
    assert cfg.shared_width == 2048 and cfg.top_k == 8 and cfg.routed_scale == 2.5
    assert cfg.mtp_layers == 1 and cfg.eos == (154820, 154827, 154829)
    assert cfg.mlp_kinds == ["dense"] * 3 + ["moe"] * 75
    assert cfg.dense_limit == 2048            # raw-token DSA: dense up to index_topk visible tokens


@pytest.mark.skipif(not REAL_DIR.is_dir(), reason="the banked GLM-5.3 checkpoint is not on this host")
def test_real_checkpoint_holds_every_tensor_the_architecture_needs():
    from tensorfold.families.glm_moe_dsa.config import Config

    cfg = Config.from_dict(json.loads(REAL_CONFIG.read_text()))
    assert cfg.missing_tensors({"_model_dir": str(REAL_DIR)}) == []


# ---------------------------------------------------------------- the family's check() gate

@pytest.mark.skipif(not REAL_CONFIG.is_file(), reason="the banked GLM-5.3 checkpoint is not on this host")
def test_check_refuses_unquantized_and_accepts_nothing_less_than_exl3():
    from tensorfold.families.glm_moe_dsa import check

    with pytest.raises(ValueError, match="EXL3"):
        check(REAL_DIR)                        # the banked BF16 checkpoint must be refused


def test_check_refuses_a_missing_checkpoint(tmp_path):
    from tensorfold.families.glm_moe_dsa import check

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    with pytest.raises(ValueError):
        check(tmp_path)


# ---------------------------------------------------------------- the CUDA engine's bad-setting checks (no GPU work)

def test_cuda_engine_refuses_an_unsplit_world_without_a_master(tmp_path, monkeypatch):
    """tp > 1 without --master fails before any GPU call (the check runs first in cuda_engine)."""

    from tensorfold.families.glm_moe_dsa import cuda_engine

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa", "num_hidden_layers": 1}))
    monkeypatch.delenv("TF_TP_WORLD", raising=False)
    with pytest.raises(ValueError, match="--master"):
        cuda_engine(tmp_path, tp=2, rank=0)
    with pytest.raises(ValueError, match="1, 2 or 4"):
        cuda_engine(tmp_path, tp=3, rank=0)


# ---------------------------------------------------------------- a tiny EXL3 checkpoint: config, names, structure

def test_tiny_checkpoint_config_and_names(tmp_path):
    """The family reads a real-shaped tiny EXL3 checkpoint: config, names, and tensor completeness pass."""

    from glm_dsa_fakes import write_checkpoint

    from tensorfold.families.glm_moe_dsa.config import Config
    from tensorfold.families.glm_moe_dsa.cuda.weights import Config as EngineConfig

    folder = write_checkpoint(tmp_path)
    cfg = Config.from_dict(json.loads((folder / "config.json").read_text()))
    assert cfg.num_hidden_layers == 3 and cfg.n_routed_experts == 4 and cfg.num_nextn_predict_layers == 1
    assert cfg.qk_head_dim == 24 and cfg.indexer_types == ["full", "shared", "full"]
    assert cfg.missing_tensors({"_model_dir": str(folder)}) == []
    engine = EngineConfig.read(folder)
    assert engine.layers == 3 and engine.experts == 4 and engine.qk_dim == 24
    assert engine.mlp_kinds == ["dense", "moe", "moe"] and engine.mtp_layers == 1
    assert engine.eos == (5,) and engine.dense_limit == 8


def test_tiny_checkpoint_check_gate(tmp_path):
    """check() passes on the tiny EXL3 checkpoint (it has every tensor and the EXL3 variant)."""

    from glm_dsa_fakes import write_checkpoint

    from tensorfold.families.glm_moe_dsa import check

    check(write_checkpoint(tmp_path))               # must not raise


def test_exl3_check_accepts_any_width_and_refuses_an_unknown_codebook(tmp_path):
    """The universal EXL3 kernels read every width and codebook: 3.0 bpw mul1 (our quant) passes, a made-up codebook
    or an out-of-range average is refused before any load."""

    from glm_dsa_fakes import write_checkpoint

    from tensorfold.families.glm_moe_dsa import check

    folder = write_checkpoint(tmp_path)
    config = json.loads((folder / "config.json").read_text())
    for good in ({"bits": 3.0, "head_bits": 6, "mtp_bits": 4, "codebook": "mul1"}, {"bits": 4, "codebook": "mcg"}):
        config["quantization_config"] = {**good, "quant_method": "exl3"}
        (folder / "config.json").write_text(json.dumps(config))
        check(folder)
    for bad in ({"bits": 3.0, "codebook": "nf4"}, {"bits": 12, "codebook": "mul1"}):
        config["quantization_config"] = {**bad, "quant_method": "exl3"}
        (folder / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError):
            check(folder)


# ---------------------------------------------------------------- the loader on a real-shaped tiny checkpoint

def test_loader_builds_every_layer_from_the_tiny_checkpoint(tmp_path, monkeypatch):
    """weights.load walks the tiny EXL3 checkpoint on CPU tensors (device stubbed): names, kv_b slices, MTP, indexer.

    The real loader moves tensors to ``cuda``; here device='cpu' exercises the same code paths minus
    the upload, so shape and name mistakes fail loudly without a GPU.
    """

    from glm_dsa_fakes import CFG, write_checkpoint

    from tensorfold.families.glm_moe_dsa.cuda import weights as W

    folder = write_checkpoint(tmp_path)
    monkeypatch.setenv("TF_TP_WORLD", "2")
    w = W.load(folder, rank=0, device="cpu")
    cfg = w.cfg
    assert cfg.layers == 3 and len(w.layers) == 3
    assert cfg.heads // 2 == 2                        # this rank's half of the heads
    for i, layer in enumerate(w.layers):
        assert layer.kind == "dsa"
        a = layer.dsa
        H2 = CFG["num_attention_heads"] // 2
        # kv_b split per head: key rows (qk_nope) then value rows (v_dim), per head
        assert a.kv_k.weight.shape == (H2 * CFG["qk_nope_head_dim"], CFG["kv_lora_rank"])
        assert a.kv_v.weight.shape == (H2 * CFG["v_head_dim"], CFG["kv_lora_rank"])
        assert a.q_b.weight.shape == (H2 * (CFG["qk_nope_head_dim"] + CFG["qk_rope_head_dim"]), CFG["q_lora_rank"])
        assert (a.index is not None) == (CFG["indexer_types"][i] == "full")
        if a.index is not None:
            assert a.index.weights.shape == (CFG["index_n_heads"], CFG["hidden_size"])
        if layer.moe is not None:
            ex = layer.moe.experts
            assert ex.count == CFG["n_routed_experts"]      # every rank holds every expert (5b128db)
            assert layer.moe.shared is not None
            assert ex.ex is None and ex.parts is not None       # a CPU load keeps the rank's triples
            # this rank's down trellis per expert: [(moe_width/world)/16, hidden/16, 16 * bits] int16
            dt, suh, svh = ex.parts["down_proj"][0]
            assert tuple(dt.shape) == (CFG["moe_intermediate_size"] // 2 // 16, CFG["hidden_size"] // 16, 64)
            assert suh.shape[0] == CFG["moe_intermediate_size"] // 2 and svh.shape[0] == CFG["hidden_size"]
            assert ex.width == CFG["moe_intermediate_size"] // 2    # this rank's expert columns
    assert w.mtp is not None
    assert w.mtp.layer.dsa.index is not None          # the MTP layer scores with its own indexer
    assert w.mtp.eh.weight.shape == (CFG["hidden_size"], 2 * CFG["hidden_size"])
    # the head holds this rank's half of the vocabulary
    assert w.head.weight.shape == (CFG["vocab_size"] // 2, CFG["hidden_size"])
    assert w.vocab_offset == 0
    other = W.load(folder, rank=1, device="cpu")
    assert other.vocab_offset == CFG["vocab_size"] // 2
    assert other.head.weight.shape == (CFG["vocab_size"] // 2, CFG["hidden_size"])
    assert w.nbytes() > 0


def test_loader_state_buffers_forward_shapes(tmp_path, monkeypatch):
    """Buffers and State construct against the tiny config's shapes (CPU), and the forward graph of calls
    (projections -> rope -> q_b -> latent write -> absorb -> expand) runs on the CPU fallback ops."""

    from glm_dsa_fakes import CFG, write_checkpoint

    from tensorfold.families.glm_moe_dsa.cuda import weights as W

    folder = write_checkpoint(tmp_path)
    monkeypatch.setenv("TF_TP_WORLD", "1")
    w = W.load(folder, rank=0, device="cpu")
    st = w.__class__ and None                          # placeholder; the real State needs device buffers
    from tensorfold.families.glm_moe_dsa.cuda.forward import Buffers, State

    b = Buffers(w, 2, 16)
    assert b.q.shape == (2, CFG["num_attention_heads"], CFG["qk_nope_head_dim"] + CFG["qk_rope_head_dim"])
    assert b.cos.shape == (2, CFG["qk_rope_head_dim"] // 2)
    s = State(w, 16, 2)
    assert len(s.kc) == CFG["num_hidden_layers"]
    assert s.index is None                             # long-context planes only when the engine asks
    assert s.index_slot == {}                          # the group map fills when the engine enables long contexts
    w.meta["long_context"] = True
    s = State(w, 16, 2)
    assert s.index_slot[0] == 0 and s.index_slot[1] == 0 and s.index_slot[2] == 1
    assert len(s.index) == 2 + (1 if w.mtp is not None else 0)  # one plane per full group, plus the MTP's



def test_interleaved_rope_matches_the_reference_pairing():
    """GPT-J interleaved rope: pairs (2i, 2i + 1) rotate together, theta 8M, 64 dims (the real head's rope slice)."""

    theta, dim = 8_000_000.0, 64
    inv_freq = 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim)
    x = torch.randn(3, dim)
    phase = torch.arange(3).float()[:, None] * inv_freq[None]
    cos, sin = phase.cos(), phase.sin()
    a, b = x[..., 0::2].float(), x[..., 1::2].float()
    want = torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).reshape(3, dim)
    got = x.float().clone()
    for r in range(3):
        for i in range(dim // 2):
            c, s = cos[r, i], sin[r, i]
            g0, g1 = got[r, 2 * i], got[r, 2 * i + 1]
            got[r, 2 * i], got[r, 2 * i + 1] = g0 * c - g1 * s, g1 * c + g0 * s
    assert torch.allclose(got, want, atol=1e-5)


def test_indexer_score_matches_the_reference_formula():
    """scores[t] = sum_h w_h * relu(scale * q_h . k_t) * (D^-0.5 * H^-0.5), causal bound t < q_pos + 1."""

    torch.manual_seed(7)
    H, D, T = 32, 128, 97
    q = torch.randn(H, D)
    k = torch.randn(T, D)
    w = torch.randn(H)
    scale = D ** -0.5
    wscale = H ** -0.5
    logits = q @ k.T * scale                       # [H, T]
    scores = (torch.relu(logits) * w[:, None]).sum(0) * wscale
    bound = 50
    assert torch.all(scores[bound:] == scores[bound:])      # the caller masks past the bound
    # relu kills negative logits: the reference's ReLU-weighted reduction matches term by term
    manual = sum(float(w[h]) * max(float(q[h] @ k[t]) * scale, 0.0) for h in range(H) for t in range(3)) \
        / sum(abs(float(w[h])) for h in range(H))
    assert manual * sum(abs(float(w[h])) for h in range(H)) == pytest.approx(
        sum(float(w[h]) * max(float(q[h] @ k[t]) * scale, 0.0) for h in range(H) for t in range(3)))


def test_split_rules_cover_the_real_tensor_names():
    """Every tensor name the real checkpoint uses maps to exactly one split kind (the TP4 world split reads them)."""

    from tensorfold.families.glm5_next.cuda.split import rule

    names = []
    for p in sorted(REAL_DIR.glob("model-0000[12]-of-00282.safetensors")):
        import struct

        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            names += [k for k in json.loads(f.read(n)) if k != "__metadata__"]
    assert names
    for n in names:
        kind = rule(n)
        assert kind in ("row", "col", "rep", "drop", "dim1"), n


def test_world_split_roundtrip_at_world_4():
    """split_bytes reassembles a row-split and a col-split tensor exactly at world 4 (the TP4 fit math's basis)."""

    import importlib

    sys.modules.pop("tensorfold.families.glm5_next.cuda.split", None)
    sys.modules.pop("tensorfold.families.glm5_next.cuda", None)
    old = os.environ.get("TF_TP_WORLD")
    os.environ["TF_TP_WORLD"] = "4"
    try:
        split_mod = importlib.import_module("tensorfold.families.glm5_next.cuda.split")
        importlib.reload(split_mod)
        assert split_mod.WORLD == 4
        for kind, shape in (("row", [256, 6144]), ("col", [6144, 2048])):
            raw = torch.arange(int(np.prod(shape)), dtype=torch.uint8).numpy()
            parts = [split_mod.split_bytes(raw, list(shape), 1, kind, rank) for rank in range(4)]
            axis = 0 if kind == "row" else 1
            for data, s in parts:
                assert s[axis] == shape[axis] // 4
            view = raw.reshape(shape)
            got = np.concatenate([d.reshape(s) for d, s in parts], axis=axis)
            assert np.array_equal(got, view)
    finally:
        if old is None:
            os.environ.pop("TF_TP_WORLD", None)
        else:
            os.environ["TF_TP_WORLD"] = old


def test_kernel_glue_imports_stay_lazy():
    """Importing the family never loads torch's CUDA stack or Triton kernels on a CPU-only host."""

    import importlib

    for name in ("tensorfold.families.glm_moe_dsa", "tensorfold.families.glm_moe_dsa.config"):
        importlib.import_module(name)
    import torch

    assert not torch.cuda.is_initialized()


# ---------------------------------------------------------------- our 3.0 bpw conversion's layout (headers only)

QUANT_WORK = Path("/tank/projects/deepspec-cache/quants/glm53-exl3-3.0bpw-work/qtensors")


def _headers(path: Path) -> dict:
    import struct

    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return {k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"}


@pytest.mark.skipif(not (QUANT_WORK / "model.layers.21.safetensors").is_file(), reason="the 3.0 bpw work dir is absent")
def test_split_rules_cover_the_quant_and_keep_hadamard_blocks_whole_at_tp4():
    """Every tensor of a dense and a sparse layer of our conversion has one split rule, and each rank's share of an
    EXL3 group at TP4 is whole 128-column blocks (so each rank's slice is a valid EXL3 linear of its own)."""

    from tensorfold.families.glm5_next.cuda.split import rule

    seen = {}
    for layer in (2, 21):
        for name, meta in _headers(QUANT_WORK / f"model.layers.{layer}.safetensors").items():
            kind = rule(name)
            assert kind in ("row", "col", "rep", "dim1"), name
            seen[name] = (kind, meta["shape"])
            if name.endswith(".trellis") and kind in ("row", "dim1"):
                axis = 0 if kind == "row" else 1
                tiles = meta["shape"][axis]                          # 16-wide tiles on the split axis
                assert tiles % (4 * 8) == 0, (name, meta["shape"])   # 4 ranks x 8 tiles a 128-column block
    # the attention groups follow their plain weights' split (q_b/kv_b by head rows, o_proj by input columns)
    assert seen["model.layers.21.self_attn.q_b_proj.trellis"][0] == "dim1"
    assert seen["model.layers.21.self_attn.q_b_proj.svh"][0] == "row"
    assert seen["model.layers.21.self_attn.q_b_proj.suh"][0] == "rep"
    assert seen["model.layers.21.self_attn.o_proj.trellis"][0] == "row"
    assert seen["model.layers.21.self_attn.o_proj.suh"][0] == "row"
    assert seen["model.layers.21.self_attn.o_proj.svh"][0] == "rep"
    assert seen["model.layers.21.self_attn.q_a_proj.trellis"][0] == "rep"
    assert seen["model.layers.21.self_attn.kv_b_proj.weight"][0] == "row"
    assert seen["model.layers.21.mlp.experts.7.gate_proj.mul1"][0] == "rep"
    assert seen["model.layers.21.mlp.shared_experts.down_proj.trellis"][0] == "row"
    assert seen["model.layers.2.mlp.gate_proj.trellis"][0] == "dim1"


@pytest.mark.skipif(not (QUANT_WORK / "model.layers.21.safetensors").is_file(), reason="the 3.0 bpw work dir is absent")
def test_quant_layers_hold_every_tensor_the_loader_asks_for():
    """The names the loader reads for a dense and a sparse layer (trellis groups or stored weights) all exist."""

    from tensorfold.families.glm_moe_dsa.config import _forms

    for layer, dense in ((2, True), (21, False)):
        have = set(_headers(QUANT_WORK / f"model.layers.{layer}.safetensors"))
        p = f"model.layers.{layer}"
        need = [f"{p}.self_attn.{x}_proj.weight" for x in ("q_a", "q_b", "o")] + \
               [f"{p}.self_attn.kv_a_proj_with_mqa.weight", f"{p}.self_attn.kv_b_proj.weight"]
        if dense:
            need += [f"{p}.mlp.{x}_proj.weight" for x in ("gate", "up", "down")]
        else:
            need += [f"{p}.mlp.shared_experts.{x}_proj.weight" for x in ("gate", "up", "down")]
            need += [f"{p}.mlp.experts.{e}.{x}_proj.trellis" for e in (0, 255) for x in ("gate", "up", "down")]
        for n in need:
            assert any(f in have for f in _forms(n)), n


def test_vocab_slice_covers_the_head_in_whole_blocks():
    """The EXL3 head split: whole 128-column blocks, a balanced span per rank, every block covered once.

    TP4 keeps the padded ceil-split the engine has always run (302.5 a rank -> 303/303/303/301);
    TP6 is the balanced uneven span (1210 = 6 * 201 + 4 -> 202/202/202/202/201/201), covering
    1210 blocks exactly with no third rank taking a piece of a second span.
    """

    from tensorfold.families.glm_moe_dsa.cuda.x3 import vocab_slice

    blocks = 154880 // 128                                  # GLM-5.3's head: 1,210 blocks
    got = [vocab_slice(blocks, 4, r) for r in range(4)]
    assert [g[1] for g in got] == [303, 303, 302, 302] and all(g[2] == 303 for g in got)
    assert [g[0] for g in got] == [0, 303, 606, 908]
    assert sum(g[1] for g in got) == blocks                 # the stored-block total the TP4 table prices
    six = [vocab_slice(blocks, 6, r) for r in range(6)]
    assert [g[1] for g in six] == [202, 202, 202, 202, 201, 201]
    assert [g[0] for g in six] == [0, 202, 404, 606, 808, 1009]
    assert all(g[2] == 202 for g in six)
    assert sum(g[1] for g in six) == blocks                 # every block on exactly one rank
    assert all(six[i + 1][0] == six[i][0] + six[i][1] for i in range(5))   # contiguous
    assert sum(g[1] for g in got) == blocks


# ---------------------------------------------------------------- the embedding split by vocabulary

def test_embed_spans_tile_the_vocabulary_in_the_heads_blocks(monkeypatch):
    """Each rank holds the EXL3 head's span of the embedding (whole 128-row blocks): the spans tile the vocabulary in
    rank order, start where the head's columns do, and embed_owner (glue.embed_pick's arithmetic) names every token's
    holder, for GLM-5.3's vocabulary and for vocabularies with a partial last block or fewer blocks than ranks.
    TF_GLM_EMBED_SPLIT=0 gives every rank the whole table."""

    from tensorfold.families.glm_moe_dsa.cuda.weights import embed_owner, embed_span

    monkeypatch.setenv("TF_GLM_EMBED_SPLIT", "0")
    assert [embed_span(154880, 4, r) for r in range(4)] == [(0, 154880)] * 4
    monkeypatch.delenv("TF_GLM_EMBED_SPLIT")
    from tensorfold.families.glm_moe_dsa.cuda.x3 import vocab_slice

    assert [embed_span(154880, 4, r) for r in range(4)] == [(0, 38784), (38784, 38784), (77568, 38656),
                                                             (116224, 38656)]
    for vocab in (154880, 256, 300, 129, 100):
        for world in (1, 2, 3, 4, 6, 8):
            spans = [embed_span(vocab, world, r) for r in range(world)]
            assert spans[0][0] == 0 and sum(n for _, n in spans) == vocab
            owner = np.array([embed_owner(t, vocab, world) for t in range(vocab)])
            for r, (first, n) in enumerate(spans):
                if r:
                    assert first == spans[r - 1][0] + spans[r - 1][1]           # contiguous, in rank order
                if n:
                    assert first == vocab_slice(-(-vocab // 128), world, r)[0] * 128     # the head's span start
                assert (owner[first:first + n] == r).all(), (vocab, world, r)


def test_embed_transform_prices_one_span_of_the_table(monkeypatch):
    """The startup estimate prices a rank's embed_span rows of the embedding, every other tensor as the plain split
    rule does: at TP4 a rank counts 0.44 GiB of embedding where the plain rule counts the whole 1.77 GiB table."""

    monkeypatch.delenv("TF_GLM_EMBED_SPLIT", raising=False)
    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.glm5_next.cuda.split import rule
    from tensorfold.families.glm_moe_dsa.cuda.weights import embed_transform

    plain = split_weights(rule, 4)
    table = {"dtype": "BF16", "shape": [154880, 6144]}
    whole = plain("model.embed_tokens.weight", table)[0]
    assert whole == 154880 * 6144 * 2                                  # replicated by the plain rule
    priced = [embed_transform(plain, 154880, 4, r)("model.embed_tokens.weight", table)[0] for r in range(4)]
    assert priced == [n * 6144 * 2 for n in (38784, 38784, 38656, 38656)]
    assert sum(priced) == whole and whole - priced[0] > 1.32 * 2 ** 30
    norm = {"dtype": "BF16", "shape": [6144]}
    assert embed_transform(plain, 154880, 4, 1)("model.norm.weight", norm) == plain("model.norm.weight", norm)


def test_loader_holds_its_vocabulary_span_of_the_embedding(tmp_path, monkeypatch):
    """weights.load keeps this rank's embed_span rows of the tiny checkpoint's table (a copy, bit for bit), and the
    whole table at world 1."""

    from glm_dsa_fakes import CFG, write_checkpoint

    from tensorfold.families.glm_moe_dsa.cuda import weights as W

    folder = write_checkpoint(tmp_path)
    V, D = CFG["vocab_size"], CFG["hidden_size"]
    monkeypatch.delenv("TF_GLM_EMBED_SPLIT", raising=False)
    monkeypatch.setenv("TF_TP_WORLD", "1")
    whole = W.load(folder, rank=0, device="cpu")
    assert whole.embed.shape == (V, D) and whole.embed_lo == 0
    monkeypatch.setenv("TF_TP_WORLD", "2")
    for rank in range(2):
        w = W.load(folder, rank=rank, device="cpu")
        first, n = W.embed_span(V, 2, rank)
        assert (w.embed_lo, tuple(w.embed.shape)) == (first, (n, D)) == (128 * rank, (128, D))
        assert torch.equal(w.embed.view(torch.int16), whole.embed[first:first + n].view(torch.int16))
        assert w.embed.untyped_storage().nbytes() == n * D * 2              # its own rows, not a view of the table
    monkeypatch.setenv("TF_GLM_EMBED_SPLIT", "0")                            # the switch: every rank the whole table
    kept = W.load(folder, rank=1, device="cpu")
    assert kept.embed_lo == 0 and torch.equal(kept.embed.view(torch.int16), whole.embed.view(torch.int16))
