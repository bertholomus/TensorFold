"""The bailing_v3 loader on a tiny Ling-shaped checkpoint: every field's shape per rank, both ranks, on CPU.

The fake (tests/bailing_fakes.py) holds every tensor the engine loads, in the reference's
shapes at 1/64 scale — EXL3 trellis groups wherever the conversion quantizes, stored matrices elsewhere —
so the load walks the same names, split rules and group crops as the real engine without a GPU. It also
pins the head's served span: each rank scores its own vocabulary slice (78,592 rows a rank at the real
vocab) whether the head is stored plain or as an EXL3 group.

Run: python -m pytest tests/test_bailing_loader.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "tests"))

torch = pytest.importorskip("torch")
if torch.cuda.is_available():          # these tests must never touch a GPU even where one exists
    pytest.skip("CPU-only bailing_v3 loader tests", allow_module_level=True)


def test_loader_builds_every_layer_from_the_tiny_checkpoint(tmp_path, monkeypatch):
    from bailing_fakes import CFG, write_checkpoint

    from tensorfold.families.bailing_v3.cuda import weights as W

    folder = write_checkpoint(tmp_path)
    monkeypatch.setenv("TF_TP_WORLD", "2")
    w = W.load(folder, rank=0, device="cpu")
    cfg = w.cfg
    assert cfg.layers == 3 and len(w.layers) == 3
    assert cfg.heads // 2 == 4                        # this rank's half of the heads
    assert cfg.layer_kinds == ["kda", "kda", "mla"]
    H2 = CFG["num_attention_heads"] // 2
    for i, layer in enumerate(w.layers):
        assert layer.kind == ("mla" if i == 2 else "kda")
        if layer.kda is not None:
            a = layer.kda
            LL = CFG["num_attention_heads"] // 2
            for proj in (a.q, a.k, a.v, a.f, a.g):    # full-rank rows by head: [LL * lin_dim, hidden]
                assert proj.n == LL * CFG["head_dim"] and proj.k == CFG["hidden_size"]
            assert a.o.n == CFG["hidden_size"] and a.o.k == LL * CFG["head_dim"]
            assert a.b.weight.shape == (LL, CFG["hidden_size"])
            assert a.conv.shape == (3 * LL * CFG["head_dim"], CFG["short_conv_kernel_size"])
            assert a.a_log.shape == (LL,) and a.a_log.dtype == torch.float32
            assert a.dt_bias.shape == (LL * CFG["head_dim"],) and a.dt_bias.dtype == torch.float32
            assert a.norm.shape == (CFG["head_dim"],) and a.b_off == 3 * LL * CFG["head_dim"]
        if layer.mla is not None:
            a = layer.mla
            assert a.q_a.n == CFG["q_lora_rank"] and a.q_a.k == CFG["hidden_size"]
            assert a.q_b.n == H2 * CFG["qk_head_dim"] and a.q_b.k == CFG["q_lora_rank"]
            # kv_a's group is stored padded to a whole 128-column block; X3 crops to the model's 192 columns
            assert a.kv_a.n == CFG["kv_lora_rank"] + CFG["qk_rope_head_dim"]
            # kv_b's rows split per head by the config's qk_nope + v_dim: key rows then value rows
            assert a.kv_k.weight.shape == (H2 * CFG["qk_nope_head_dim"], CFG["kv_lora_rank"])
            assert a.kv_v.weight.shape == (H2 * CFG["v_head_dim"], CFG["kv_lora_rank"])
            # the forward absorbs q_nope alone (a power-of-two width): wk holds the key rows, no zero rope rows
            assert a.absorb.wk.shape == (H2, CFG["qk_nope_head_dim"], CFG["kv_lora_rank"])
            assert a.absorb.wv.shape == (H2, CFG["v_head_dim"], CFG["kv_lora_rank"])
            assert a.gate.weight.shape == (H2, CFG["hidden_size"])
            assert a.dense.n == CFG["hidden_size"] and a.dense.k == H2 * CFG["v_head_dim"]
            assert a.q_norm.shape == (CFG["q_lora_rank"],) and a.kv_norm.shape == (CFG["kv_lora_rank"],)
        if layer.moe is not None:
            ex = layer.moe.experts
            assert ex.count == CFG["num_experts"]     # every rank holds every expert (width-split, not id-split)
            assert layer.moe.shared is not None
            assert ex.ex is None and ex.parts is not None   # a CPU load keeps the rank's triples
            assert ex.width == CFG["moe_intermediate_size"] // 2
            dt, suh, svh = ex.parts["down_proj"][0]         # this rank's down trellis: K tiles cut, N tiles whole
            assert tuple(dt.shape) == (CFG["moe_intermediate_size"] // 2 // 16, CFG["hidden_size"] // 16, 64)
            assert suh.shape[0] == CFG["moe_intermediate_size"] // 2 and svh.shape[0] == CFG["hidden_size"]
            assert layer.moe.router.shape == (CFG["num_experts"], CFG["hidden_size"])
            assert layer.moe.bias.shape == (CFG["num_experts"],) and layer.moe.bias.dtype == torch.float32
        if layer.mlp is not None:                     # layer 0's dense SwiGLU
            assert layer.mlp.gu.gate.n == CFG["intermediate_size"] // 2   # this rank's gate rows
            assert layer.mlp.gu.up.n == CFG["intermediate_size"] // 2
            assert layer.mlp.down.n == CFG["hidden_size"] and layer.mlp.down.k == CFG["intermediate_size"] // 2
    assert w.embed.shape == (CFG["vocab_size"], CFG["hidden_size"])   # the whole table on every rank
    assert w.norm.shape == (CFG["hidden_size"],)
    assert w.mtp is not None
    assert w.mtp.layer.mla is not None and w.mtp.layer.moe is not None
    assert w.mtp.eh.n == CFG["hidden_size"] and w.mtp.eh.k == 2 * CFG["hidden_size"]
    assert w.mtp.enorm.shape == (CFG["hidden_size"],) and w.mtp.hnorm.shape == (CFG["hidden_size"],)
    assert w.mtp.norm.shape == (CFG["hidden_size"],)                  # final_layernorm: the chain norm
    # the head holds this rank's half of the vocabulary in whole 128-row blocks (the EXL3 group's span)
    assert w.head.n == CFG["vocab_size"] // 2 and w.vocab_offset == 0
    assert w.nbytes() > 0


def test_both_ranks_hold_disjoint_head_and_expert_spans(tmp_path, monkeypatch):
    from bailing_fakes import CFG, write_checkpoint

    from tensorfold.families.bailing_v3.cuda import weights as W

    folder = write_checkpoint(tmp_path)
    monkeypatch.setenv("TF_TP_WORLD", "2")
    w0 = W.load(folder, rank=0, device="cpu")
    w1 = W.load(folder, rank=1, device="cpu")
    assert w0.vocab_offset == 0 and w1.vocab_offset == CFG["vocab_size"] // 2
    assert w0.head.n == w1.head.n == CFG["vocab_size"] // 2
    for w in (w0, w1):
        a = w.layers[2].mla
        assert a.q_b.n == (CFG["num_attention_heads"] // 2) * CFG["qk_head_dim"]
        assert a.kv_k.weight.shape[0] == (CFG["num_attention_heads"] // 2) * CFG["qk_nope_head_dim"]
    assert w0.layers[0].kda.dt_bias.shape == w1.layers[0].kda.dt_bias.shape
    assert w0.layers[0].kda.a_log.shape == (CFG["num_attention_heads"] // 2,)


def test_loader_refuses_a_quantized_kv_b(tmp_path, monkeypatch):
    """kv_b feeds the absorb kernels per head: a trellis group there is refused with the reason."""

    from bailing_fakes import write_checkpoint

    from tensorfold.families.bailing_v3.cuda import weights as W

    folder = write_checkpoint(tmp_path, kv_b_trellis=True)
    monkeypatch.setenv("TF_TP_WORLD", "2")
    with pytest.raises(ValueError, match="kv_b"):
        W.load(folder, rank=0, device="cpu")
