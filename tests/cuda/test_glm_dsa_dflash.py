"""A DFlash2 drafter on GLM-5.3 (glm_moe_dsa, --drafter), the GPU pieces: the block attention with query rows that are
not a power of two (padded) against a plain reference; a context that skips a prompt's rows before the drafter's
window drafts exactly what the whole context drafts (4-bit and BF16 linears); the taps of a prompt chunk's
micro-batches are the chunk's rows after the layer. (Replies against serial run on the lane: tools/tf_same_lane.py.)"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import split  # noqa: E402

D, V, WINDOW = 256, 512, 48
# 6 query heads over 2 KV heads: 3 x 8 block rows = 24 query rows a KV head, padded to 32 in the block attention
DRAFT = {
    "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": D, "vocab_size": V,
    "head_dim": 128, "num_attention_heads": 6, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 2000000.0}, "sliding_window": WINDOW, "is_causal": True,
    "intermediate_size": 256, "num_hidden_layers": 2,
    "dflash_config": {"mask_token_id": V - 6, "conv_group_size": 16, "conv_kernel_size": 2, "block_size": 8,
                      "selector_rank": 16, "selector_top_k": 8, "target_layer_ids": [1, 3, 5]},
}


def write_drafter(path: Path) -> Path:
    rng = np.random.default_rng(7)
    tensors: list = []
    cfg = DRAFT
    H, KV, hd, inter = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"]
    taps = len(cfg["dflash_config"]["target_layer_ids"])
    rank = cfg["dflash_config"]["selector_rank"]

    def bf16(name: str, shape: list[int], scale: float = 0.05, offset: float = 0.0) -> None:
        x = torch.tensor(rng.standard_normal(shape) * scale + offset, dtype=torch.float32).to(torch.bfloat16)
        tensors.append((name, "BF16", shape, x.view(torch.uint16).numpy().view(np.uint8).reshape(-1)))

    bf16("fc.weight", [D, taps * D], 0.05)
    bf16("hidden_norm.weight", [D], 0.05, 1.0)
    bf16("norm.weight", [D], 0.05, 1.0)
    bf16("candidate_selector.hidden_projection.weight", [rank, D], 0.05)
    bf16("candidate_selector.predecessor_codebook", [V, rank], 0.3)
    bf16("candidate_selector.successor_codebook", [V, rank], 0.3)
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        bf16(p + "self_attn.q_proj.weight", [H * hd, D], 0.05)
        bf16(p + "self_attn.k_proj.weight", [KV * hd, D], 0.05)
        bf16(p + "self_attn.v_proj.weight", [KV * hd, D], 0.05)
        bf16(p + "self_attn.o_proj.weight", [D, H * hd], 0.05)
        bf16(p + "self_attn.q_norm.weight", [hd], 0.05, 1.0)
        bf16(p + "self_attn.k_norm.weight", [hd], 0.05, 1.0)
        bf16(p + "mlp.gate_proj.weight", [inter, D], 0.05)
        bf16(p + "mlp.up_proj.weight", [inter, D], 0.05)
        bf16(p + "mlp.down_proj.weight", [D, inter], 0.05)
        for conv in ("attention_conv", "mlp_conv"):
            bf16(p + conv + ".base_kernel", [2, 2, D], 0.1, 0.5)
            bf16(p + conv + ".kernel_projection.weight", [4 * D // 16, D], 0.02)
        bf16(p + "input_layernorm.weight", [D], 0.05, 1.0)
        bf16(p + "post_attention_layernorm.weight", [D], 0.05, 1.0)
    path.mkdir(parents=True, exist_ok=True)
    split.write(str(path / "model.safetensors"), tensors, {"format": "pt"})
    (path / "config.json").write_text(json.dumps(cfg))
    return path


def _target():
    """What the drafter reads of the target on one rank: its embedding table and head (a BF16 head here)."""

    from tensorfold.families.glm5_next.cuda import qmm

    g = torch.Generator(device="cuda").manual_seed(5)
    embed = (torch.randn((V, D), generator=g, device="cuda") * 0.05).bfloat16()
    head = qmm.make_b16((torch.randn((V, D), generator=g, device="cuda") * 0.03).bfloat16())
    return SimpleNamespace(device=torch.device("cuda"), rank=0, world=1, comm=None, embed=embed, head=head,
                           draft_head=None, vocab_offset=0)


@pytest.mark.parametrize("G,N", [(3, 8), (6, 8), (3, 4), (4, 8)])
def test_block_attention_with_padded_query_rows(G, N):
    import triton

    from tensorfold.families.glm5_next.cuda.dflash2 import _dattn_kernel

    gen = torch.Generator(device="cuda").manual_seed(G * 10 + N)
    KV, HD, window, cap = 2, 128, 40, 512
    H = G * KV
    s = 100
    kc = torch.randn((KV, cap, HD), generator=gen, device="cuda").bfloat16()
    vc = torch.randn((KV, cap, HD), generator=gen, device="cuda").bfloat16()
    q = torch.randn((H, N, HD), generator=gen, device="cuda").bfloat16()
    out = torch.empty((N, H * HD), dtype=torch.bfloat16, device="cuda")
    mp = triton.next_power_of_2(G * N)
    _dattn_kernel[(KV,)](q, kc, vc, out, torch.tensor([s], device="cuda"), window, HD ** -0.5, N=N, G=G, NH=H,
                         HD=HD, CAP=cap, BK=64, CAUSAL=False, RING=False, MP=mp, PAD=mp != G * N, num_warps=4)
    # reference: query row r (position s + r) sees context keys within the window and every block key
    pos = torch.arange(s + N, device="cuda")
    for h in range(H):
        kvh = h // G
        k, v = kc[kvh, :s + N].float(), vc[kvh, :s + N].float()
        sc = (q[h].float() @ k.T) * HD ** -0.5
        qpos = s + torch.arange(N, device="cuda")
        ok = ((pos[None, :] < s) & (qpos[:, None] - pos[None, :] <= window)) | (pos[None, :] >= s)
        ref = torch.softmax(sc.masked_fill(~ok, float("-inf")), dim=-1) @ v
        torch.testing.assert_close(out[:, h * HD:(h + 1) * HD].float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("bits", [4, 0], ids=["4bit", "bf16"])
@pytest.mark.parametrize("length", [5, 49, 300, 1000])
def test_a_context_that_skips_the_prompt_before_the_window_drafts_the_same(tmp_path, bits, length):
    """prefill's skip (dflash.prompt_from, Drafter.skip_to): the same candidates, values and selector rows as a drafter
    that took every prompt row, after the prompt and over decode-like rounds of kept rows (ring and flat context)."""

    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
    from tensorfold.families.glm_moe_dsa.cuda.dflash import prompt_from

    path = write_drafter(tmp_path / "d")
    w = _target()
    taps = len(DRAFT["dflash_config"]["target_layer_ids"])
    g = torch.Generator(device="cuda").manual_seed(length)
    rows = (torch.randn((length + 40, taps * D), generator=g, device="cuda") * 0.5).bfloat16()
    for ring in (True, False):
        full = Drafter(path, w, capacity=4096, ring=ring, bits=bits)
        cut = Drafter(path, w, capacity=4096, ring=ring, bits=bits)
        start = prompt_from(cut, length)
        assert start == max(0, length - WINDOW)
        if start:                                    # (then the same 64-row pieces as the skipped context's)
            full.add_taps(rows[:start])
        full.add_taps(rows[start:length])
        cut.skip_to(start)
        cut.add_taps(rows[start:length])
        assert full.context_end == cut.context_end == length
        pos, rng = length, np.random.default_rng(length)
        for keep in [0, 3, 1, 8, 2, 5]:
            if keep:
                full.add_taps(rows[pos:pos + keep])
                cut.add_taps(rows[pos:pos + keep])
                pos += keep
            pending = int(rng.integers(0, V - 8))
            a, b = full.candidates(pending, 7), cut.candidates(pending, 7)
            for x, y in zip(a, b):
                assert np.array_equal(x, y), (ring, pos)
            assert full.propose(pending, 7, None) == cut.propose(pending, 7, None)


def test_a_skip_never_goes_back_or_past_the_capacity(tmp_path):
    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

    d = Drafter(write_drafter(tmp_path / "d"), _target(), capacity=1024, ring=True)
    d.skip_to(100)
    with pytest.raises(ValueError):
        d.skip_to(50)
    with pytest.raises(ValueError):
        d.skip_to(2000)


@pytest.mark.parametrize("R,keep,h", [(600, 100, 304), (600, 400, 304), (90, 200, 48), (8, 8, 0)])
def test_taps_of_a_prompt_chunks_micro_batches_are_its_rows_after_the_layer(R, keep, h):
    """forward.tap: a chunk of R rows keeps its last ``keep`` rows; in two micro-batches (rows 0..h, h..R) each with
    its layer's branch still pending (prompt_layers), the taps are x + branch (residual's add), x left as it was."""

    from tensorfold.families.glm5_next.cuda import glue as fglue
    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd

    g = torch.Generator(device="cuda").manual_seed(R + keep)
    x = (torch.randn((R, D), generator=g, device="cuda")).bfloat16()
    branch = (torch.randn((R, D), generator=g, device="cuda") * 0.1).bfloat16()
    x0 = x.clone()
    want = torch.empty_like(x)
    fglue.residual_add(x, want, branch.view(1, R, D))
    first = R - min(R, keep)

    full = SimpleNamespace(x=x, tap_at={7: [0, 2], 9: [1]}, tap_rows=min(R, keep))
    full.taps = [torch.full((full.tap_rows, D), float("nan"), dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    parts = [(0, h), (h, R)] if h else [(0, R)]
    for lo, hi in parts:
        if hi <= lo:
            continue
        v = SimpleNamespace(x=x[lo:hi], tap_at=full.tap_at, full=full)
        fwd.tap(v, 7, hi - lo, fwd.Rows(branch[lo:hi].contiguous()), lo=lo, chunk=R)
    assert torch.equal(x, x0)                                   # x itself untouched
    for slot in (0, 2):
        assert torch.equal(full.taps[slot], want[first:R]), slot
    assert torch.isnan(full.taps[1].float()).all()              # layer 9 not tapped by layer 7's call
    # the whole chunk at once, nothing pending (the plain loop and decode windows): its last rows of x
    one = SimpleNamespace(x=x, tap_at={9: [0]}, tap_rows=min(R, keep))
    one.taps = [torch.empty((one.tap_rows, D), dtype=torch.bfloat16, device="cuda")]
    fwd.tap(one, 9, R)
    assert torch.equal(one.taps[0], x[first:R])
