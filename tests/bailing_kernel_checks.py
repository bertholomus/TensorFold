"""Ling-3.1-flash's own Triton kernels (bailing_v3/cuda/route.py, rope.py, act.py) against torch transcriptions of
the reference, the select kernel on the oracle's real router rows, and the latent attention's filled walk against
today's, run in a child process by tests/test_bailing_4_rope.py, test_bailing_6_router.py (route, router_rows),
test_bailing_9_swiglu.py and test_bailing_14_filled.py.

A child process because Triton's interpreter (TRITON_INTERPRET=1, the CPU path on hosts without CUDA) must
be chosen before triton loads: ``python tests/bailing_kernel_checks.py
rope|route|swiglu|router_rows|filled [cpu|cuda]``.
On a CUDA host the same checks run natively. Exit status 0 and "ok" on success; an AssertionError otherwise.

The interpreter converts fp32 to bf16 by truncation (Triton 3.8), so on the CPU the rope rotation is held to the
reference's arithmetic with that rounding (its structure); the GPU run holds it to the reference bit for bit.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

THETA, ROPE = 6_000_000.0, 64


# -- rope -----------------------------------------------------------------------------------------------------
def reference_cos_sin(positions: torch.Tensor, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """BailingMoeV3RotaryEmbedding (modeling 248-281): inv_freq made on the CPU, freqs = inv @ pos in fp32, emb =
    cat(freqs, freqs), cos / sin rounded to bf16, on ``device``. Returns [R, 64] bf16 each (on the CPU).

    The reference's freqs are a matmul with one term a sum: each element the fp32 product inv_i * pos, rounded once.
    On the CPU the matmul form is used verbatim; on a GPU the product is taken elementwise (the same values, and
    immune to TORCH_ALLOW_TF32_CUBLAS_OVERRIDE, which would round both factors to TF32 inside cuBLAS)."""

    inv = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, dtype=torch.int64).to(dtype=torch.float) / ROPE))
    if device == "cpu":
        freqs = (inv[None, :, None].float() @ positions[None, None, :].float()).transpose(1, 2)[0]   # [R, 32]
        assert torch.equal(freqs, positions.float()[:, None] * inv[None, :])
    else:
        freqs = positions.to(device).float()[:, None] * inv.to(device)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16).cpu(), emb.sin().to(torch.bfloat16).cpu()


def _round(x: torch.Tensor, mode: str) -> torch.Tensor:
    """fp32 -> bf16: "rne" (torch, and the kernel's .to(tl.bfloat16) on a GPU) or "trunc" (the low 16 bits dropped:
    what Triton 3.8's CPU interpreter does for that conversion, explicit rounding mode or not)."""

    if mode == "trunc":
        return (x.contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)
    return x.to(torch.bfloat16)


def _rotate_explicit(xd, rot, c, s, mode: str) -> torch.Tensor:
    """(xd * c) + (rot * s) of bf16 tensors written out: each product and the sum taken in fp32, each rounded."""

    ac = _round(xd.float() * c.float(), mode)
    rs = _round(rot.float() * s.float(), mode)
    return _round(ac.float() + rs.float(), mode)


def reference_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rounding: str = "rne") -> torch.Tensor:
    """apply_rotary_pos_emb_interleave (modeling 541-576) on rows x [R, ..., 64] bf16 (cos / sin [R, 64]), returned
    re-interleaved (pair i back at columns 2i, 2i + 1) so it compares with the engine's in-place layout.

    ``rounding`` "trunc": the same arithmetic with Triton's CPU interpreter's truncating bf16 conversion, so the
    interpreter run pins the kernel's structure (pairs, signs, which values are rounded) and a GPU run pins the
    rounding as well. The written-out form is first held equal to the reference's bf16 expression."""

    shape = x.shape
    d = shape[-1]
    c = cos.view(shape[0], *([1] * (x.dim() - 2)), d)
    s = sin.view(shape[0], *([1] * (x.dim() - 2)), d)
    xd = x.view(*shape[:-1], d // 2, 2).transpose(-1, -2).reshape(shape)                  # de-interleaved
    x1, x2 = xd[..., :d // 2], xd[..., d // 2:]
    rot = torch.cat((-x2, x1), dim=-1)                                                     # rotate_half
    out = (xd * c) + (rot * s)                                                             # bf16 ops
    assert torch.equal(out.view(torch.int16), _rotate_explicit(xd, rot, c, s, "rne").view(torch.int16))
    if rounding != "rne":
        out = _rotate_explicit(xd, rot, c, s, rounding)
    return out.view(*shape[:-1], 2, d // 2).transpose(-1, -2).reshape(shape)


def rope_check(device: str) -> None:
    from types import SimpleNamespace

    from tensorfold.families.bailing_v3.cuda import rope

    cfg = SimpleNamespace(qk_nope=128, qk_rope=ROPE, qk_dim=128 + ROPE, kv_lora=512)
    heads, R = 4, 6
    rounding = "trunc" if device == "cpu" else "rne"      # the interpreter truncates fp32 -> bf16
    # the table's inverse frequencies are the reference's, bit for bit
    assert torch.equal(rope.inv_freq(THETA, ROPE), 1.0 / (6000000 ** (torch.arange(0, ROPE, 2).float() / ROPE)))
    for base in (0, 1, 4095, 262143):
        g = torch.Generator().manual_seed(base + 1)
        t = rope.Table(R, THETA, ROPE, device)
        t.fill(torch.tensor([base], dtype=torch.int32, device=device), R)
        cos, sin = reference_cos_sin(torch.arange(base, base + R), device)
        assert torch.equal(t.cos.cpu(), cos[:, :ROPE // 2]) and torch.equal(t.sin.cpu(), sin[:, :ROPE // 2]), base
        # queries: q_b rows [R, heads * 192], each head's last 64 columns rotated; the nope columns untouched
        q = (torch.randn((R, heads * cfg.qk_dim), generator=g) * 3).to(torch.bfloat16)
        got = q.clone().to(device)                 # (to() on the CPU would alias q: rotated in place)
        rope.apply_q(got, t, R, cfg, heads)
        got = got.cpu().view(R, heads, cfg.qk_dim)
        want = q.view(R, heads, cfg.qk_dim).clone()
        want[..., cfg.qk_nope:] = reference_rotate(want[..., cfg.qk_nope:].contiguous(), cos, sin, rounding)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), f"q at {base}"
        # the shared rope key: kv_a's last 64 columns of a [R, 576] row (a strided view), the latent untouched
        lat = (torch.randn((R, cfg.kv_lora + ROPE), generator=g) * 3).to(torch.bfloat16)
        got = lat.clone().to(device)
        rope.apply_key(got[:, cfg.kv_lora:], t, R, cfg)
        want = lat.clone()
        want[:, cfg.kv_lora:] = reference_rotate(lat[:, cfg.kv_lora:].contiguous(), cos, sin, rounding)
        assert torch.equal(got.cpu().view(torch.int16), want.view(torch.int16)), f"key at {base}"
    # a window's row r has the bits of a one-row table at position base + r (rows never read each other)
    t = rope.Table(R, THETA, ROPE, device)
    t.fill(torch.tensor([4095], dtype=torch.int32, device=device), R)
    one = rope.Table(1, THETA, ROPE, device)
    for r in range(R):
        one.fill(torch.tensor([4095 + r], dtype=torch.int32, device=device), 1)
        assert torch.equal(one.cos[0], t.cos[r]) and torch.equal(one.sin[0], t.sin[r])


# -- router ---------------------------------------------------------------------------------------------------
E, N_GROUP, TOPK_GROUP, TOP_K, SCALE = 512, 8, 4, 8, 2.5


def reference_gate(scores: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BailingMoeV3Gate after the sigmoid (modeling 368-406), verbatim: (indices, weights) [R, 8]."""

    rows = scores.shape[0]
    sb = scores + bias
    group_scores = sb.view(rows, N_GROUP, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(rows, N_GROUP, E // N_GROUP).reshape(rows, -1)
    masked = sb.masked_fill(~score_mask.bool(), float("-inf"))
    _, top = torch.topk(masked, k=TOP_K, dim=-1)
    s = torch.gather(scores, 1, top)
    return top, s / (s.sum(dim=-1, keepdim=True) + 1e-20) * SCALE


def spec_gate(scores: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's rules in torch: stable descending sorts (the lower index on ties) and the weight sum taken in
    pick order, one fp32 add at a time, the quotient rounded once, then the scale."""

    rows = scores.shape[0]
    sb = scores + bias
    groups = sb.view(rows, N_GROUP, E // N_GROUP)
    top2 = torch.gather(groups, -1, torch.argsort(-groups, dim=-1, stable=True)[..., :2])
    gsum = top2[..., 0] + top2[..., 1]
    keep = torch.argsort(-gsum, dim=-1, stable=True)[..., :TOPK_GROUP]
    mask = torch.zeros_like(gsum, dtype=torch.bool).scatter_(1, keep, True)
    masked = torch.where(mask.repeat_interleave(E // N_GROUP, dim=1), sb, torch.full_like(sb, float("-inf")))
    top = torch.argsort(-masked, dim=-1, stable=True)[..., :TOP_K]
    s = torch.gather(scores, 1, top)
    total = torch.zeros(rows, dtype=torch.float32)
    for j in range(TOP_K):
        total = total + s[:, j]
    return top, s / (total[:, None] + 1e-20) * SCALE


def _run(scores: torch.Tensor, bias: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from tensorfold.families.bailing_v3.cuda import route

    rows = scores.shape[0]
    pick = torch.full((rows, TOP_K + 1), -7, dtype=torch.int32, device=device)
    wts = torch.full((rows, TOP_K + 1), -7.0, dtype=torch.float32, device=device)
    route.select(scores.to(device).contiguous(), bias.to(device), pick, wts, n_group=N_GROUP, topk_group=TOPK_GROUP,
                 top_k=TOP_K, scale=SCALE)
    return pick.cpu(), wts.cpu()


def route_check(device: str, rows: int = 768) -> None:
    from tensorfold.families.bailing_v3.cuda import route

    g = torch.Generator().manual_seed(17)
    logits = torch.randn((rows, E), generator=g) * 2
    bias = torch.randn((E,), generator=g) * 0.05
    scores = torch.sigmoid(logits)
    pick, wts = _run(scores, bias, device)
    # the shared expert's slot: id E, weight 1
    assert torch.equal(pick[:, TOP_K], torch.full((rows,), E, dtype=torch.int32))
    assert torch.equal(wts[:, TOP_K], torch.ones(rows))
    top, w = spec_gate(scores, bias)
    assert torch.equal(pick[:, :TOP_K].long(), top), "picks differ from the stable-sort spec"
    assert torch.equal(wts[:, :TOP_K], w), "weights differ from the pick-order fp32 transcription"
    ref_top, ref_w = reference_gate(scores, bias)
    assert torch.equal(torch.sort(ref_top, dim=-1).values, torch.sort(top, dim=-1).values), "expert sets differ"
    ref_by_pick = torch.gather(torch.zeros_like(scores).scatter(1, ref_top, ref_w), 1, top)
    # the reference sums the eight scores in torch's own order (it differs between CPU builds and devices: on one
    # GB10 the totals differed in their last bits), the kernel in pick order. Each sum of eight positive terms lies
    # within 7u of the exact one (u = eps / 2) and each side rounds its quotient and its scale once more: 14u + 4u =
    # 9 eps of the weight
    eps_w = torch.finfo(torch.float32).eps * w.abs()
    assert ((w - ref_by_pick).abs() <= 9 * eps_w).all(), "weights more than 9 eps from the reference's"
    # the sigmoid step: route() runs torch.sigmoid (the reference's op) on the device into the scores buffer, then
    # select: the same picks and weights as select on those scores
    pick2 = torch.empty((rows, TOP_K + 1), dtype=torch.int32, device=device)
    wts2 = torch.empty((rows, TOP_K + 1), dtype=torch.float32, device=device)
    sc = torch.empty((rows, E), dtype=torch.float32, device=device)
    route.route(logits.to(device), bias.to(device), sc, pick2, wts2, n_group=N_GROUP, topk_group=TOPK_GROUP,
                top_k=TOP_K, scale=SCALE)
    dev_scores = torch.sigmoid(logits.to(device)).cpu()
    assert torch.equal(sc.cpu(), dev_scores)
    p3, w3 = _run(dev_scores, bias, device)
    assert torch.equal(pick2.cpu(), p3) and torch.equal(wts2.cpu(), w3)
    # a row's bits do not depend on the window: each row alone
    for r in (0, 5, rows - 1):
        p1, w1 = _run(scores[r:r + 1], bias, device)
        assert torch.equal(p1[0], pick[r]) and torch.equal(w1[0], wts[r]), r
    # ties: all-equal scores keep index order; a group-score tie keeps the lower group
    flat = torch.full((2, E), 0.5)
    p, _ = _run(flat, torch.zeros(E), device)
    assert torch.equal(p[:, :TOP_K].long(), torch.arange(TOP_K).expand(2, -1))
    tie = torch.full((1, E), 0.1)
    for gi in range(N_GROUP):                  # every group's top-2 sum equal: 0.9 + 0.8
        tie[0, gi * 64 + 10], tie[0, gi * 64 + 20] = 0.9, 0.8
    p, _ = _run(tie, torch.zeros(E), device)
    kept = sorted(set((p[0, :TOP_K] // 64).tolist()))
    assert kept == [0, 1, 2, 3], kept          # the four lowest groups
    assert p[0, :TOP_K].tolist() == [10, 74, 138, 202, 20, 84, 148, 212]
    # one expert per group too few: the shape rules refuse what the kernel cannot pick
    try:
        route.check_shape(512, 8, 1, 65)
    except ValueError:
        pass
    else:
        raise AssertionError("check_shape accepted top-65 of one kept group of 64")


# -- the shared expert's clamped SwiGLU ------------------------------------------------------------------------
def _same_or_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """bf16 tensors: NaN in the same places (payloads may differ between the interpreter and torch), the same bits
    everywhere else."""

    na, nb = a.isnan(), b.isnan()
    return torch.equal(na, nb) and torch.equal(a[~na].view(torch.int16), b[~nb].view(torch.int16))


def swiglu_check(device: str) -> None:
    """act.silu_mul: at L = inf the plain kernel's (glm_moe_dsa glue.silu_mul) bits and 64-group sums; at a finite L
    the clamp after silu, bf16(min(s, L) * clip(u, -L, L)) with s the kernel's own bf16 silu (the plain kernel's
    output for up = 1: bf16(s * 1) is s); NaN gates and up values stay NaN; a row's bits do not depend on the
    window. The rounding model is the interpreter's truncation on the CPU, round-to-nearest-even on a GPU."""

    from tensorfold.families.bailing_v3.cuda import act
    from tensorfold.families.glm_moe_dsa.cuda import glue as dsa_glue

    rounding = "trunc" if device == "cpu" else "rne"
    rows, w = 6, 1280                                  # the shared expert's width
    g = torch.Generator().manual_seed(23)
    gu = (torch.randn((rows, 2 * w), generator=g) * 4).to(torch.bfloat16)
    gu[0, 5] = float("nan")                            # a NaN gate
    gu[1, w + 7] = float("nan")                        # a NaN up value

    def run(fn, x, *limit):
        n = x.shape[0]
        out = torch.full((n, w), -7.0, dtype=torch.bfloat16, device=device)
        xs = torch.full((n, w // 64), -7.0, dtype=torch.float32, device=device)
        fn(x.clone().to(device), out, xs, *limit)
        return out.cpu(), xs.cpu()

    def bits(t):
        return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)

    plain, plain_xs = run(dsa_glue.silu_mul, gu)
    for limit in ((), (math.inf,)):                    # the default, and inf given
        a, xs = run(act.silu_mul, gu, *limit)
        assert torch.equal(bits(a), bits(plain)) and torch.equal(bits(xs), bits(plain_xs)), limit
    ones = gu.clone()
    ones[:, w:] = 1.0
    s, _ = run(dsa_glue.silu_mul, ones)
    u = gu[:, w:].float()
    for L in (10.0, 2.5, 0.75):
        a, xs = run(act.silu_mul, gu, L)
        want = _round(s.float().clamp(max=L) * u.clamp(-L, L), rounding)
        assert _same_or_nan(a, want), L
        assert a[0, 5].isnan() and a[1, 7].isnan(), L
        sums = want.float().view(rows, w // 64, 64).sum(-1)
        assert torch.allclose(xs, sums, rtol=1e-5, atol=1e-4, equal_nan=True), L
        assert int((bits(a) != bits(plain)).sum()) > (rows * w) // (200 if L == 10.0 else 20), L    # it bites
        for r in (0, 3, rows - 1):
            one, one_xs = run(act.silu_mul, gu[r:r + 1], L)
            assert torch.equal(bits(one[0]), bits(a[r])) and torch.equal(bits(one_xs[0]), bits(xs[r])), (L, r)


# -- the oracle's real router rows -----------------------------------------------------------------------------
ORACLE_O2 = Path(os.environ.get("LING_ORACLE_O2", "/models/Ling-3.1-flash-oracle/o2"))
BANK = Path(os.environ.get("LING_BANK") or os.environ.get("LING31_FP8_DIR", "/models/Ling-3.1-flash"))
ROWS_LAYERS = (4, 68)


def _load_rows_dump(layer: int):
    """The oracle's dump of one layer's real router rows, or None: an .npz under the oracle's o2/ whose name contains
    router and the layer number (router_L4.npz, router_L68.npz), with the picks, the weights and the reference gate's
    own fp32 scores (or its logits, or the input rows); expert_bias is optional (the bank's is read when absent)."""

    import re

    import numpy as np

    if not ORACLE_O2.is_dir():
        return None
    for p in sorted(ORACLE_O2.glob("*.npz")):
        if "router" not in p.name or not re.search(rf"(?<!\d){layer}(?!\d)", p.name):
            continue
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        if any(k in d for k in ("picks", "indices")) or any(k in d for k in ("weights", "wts")):
            return d
    return None


def _bank_tensor(name: str) -> torch.Tensor:
    """One tensor of the banked FP8 checkpoint, read as stored (the index picks the shard)."""

    import json

    from safetensors import safe_open

    idx = json.loads((BANK / "model.safetensors.index.json").read_text())
    shard = idx["weight_map"][name]
    with safe_open(BANK / shard, framework="pt") as f:
        return f.get_tensor(name)


def _rows_scores(d: dict, layer: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    """(scores, bias, source): the reference gate's own fp32 sigmoid scores when the dump holds them (select then
    ranks exactly what the reference ranked), else sigmoid of its fp32 logits, else of F.linear(h, the bank's gate
    weight) on the input rows (another fp32 sum order: a near-tie can flip, so a dump should carry its scores); the
    bias the dump's fp32 expert_bias or the bank's (never rounded)."""

    def first(*names):
        return next(((n, torch.as_tensor(d[n])) for n in names if n in d), (None, None))

    _, bias = first("expert_bias", "bias")
    if bias is None:
        bias = _bank_tensor(f"model.layers.{layer}.mlp.gate.expert_bias").float()
    name, scores = first("scores", "router_scores")
    if scores is not None:
        return scores.float(), bias.float(), name
    name, logits = first("logits", "router_logits", "gate_logits")
    if logits is not None:
        return torch.sigmoid(logits.float()), bias.float(), f"sigmoid({name})"
    name, h = first("h", "hidden", "input", "router_input")
    assert h is not None, f"the layer-{layer} dump holds neither scores, logits nor input rows: {sorted(d)}"
    w = _bank_tensor(f"model.layers.{layer}.mlp.gate.weight")
    return torch.sigmoid(torch.nn.functional.linear(h.float(), w.float())), bias.float(), f"sigmoid(linear({name}))"


def router_rows_check(device: str) -> None:
    """route.select on the oracle's real layer-4 and layer-68 rows: per row the same expert set as the dump's picks
    (order free: torch.topk leaves ties unordered) and weights within 9 eps of the dump's matched by expert id (the
    sum-order bound of route_check). A failing row prints its nine largest sigmoid+bias values (ties show)."""

    # asserted: fp32 router arithmetic must not see TF32 or a reduced matmul precision
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") in ("", "0")     # as bailing_v3's guard
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"

    from tensorfold.families.bailing_v3.cuda import route

    dumps = {L: _load_rows_dump(L) for L in ROWS_LAYERS}
    assert any(d is not None for d in dumps.values()), f"no router dumps under {ORACLE_O2}"
    total = 0
    for L, d in dumps.items():
        if d is None:
            print(f"layer {L}: no dump (skipped)")
            continue
        picks = torch.as_tensor(d["picks"] if "picks" in d else d["indices"]).long()
        ref_w = torch.as_tensor(d["weights"] if "weights" in d else d["wts"]).float()
        scores, bias, source = _rows_scores(d, L)
        rows = scores.shape[0]
        assert picks.shape == (rows, TOP_K) and ref_w.shape == (rows, TOP_K), (L, picks.shape, ref_w.shape)
        pick = torch.full((rows, TOP_K + 1), -7, dtype=torch.int32, device=device)
        wts = torch.full((rows, TOP_K + 1), -7.0, dtype=torch.float32, device=device)
        route.select(scores.to(device).contiguous(), bias.to(device), pick, wts, n_group=N_GROUP,
                     topk_group=TOPK_GROUP, top_k=TOP_K, scale=SCALE)
        pick, wts = pick.cpu(), wts.cpu()
        got = pick[:, :TOP_K].long()
        bad_sets = (torch.sort(got, dim=-1).values != torch.sort(picks, dim=-1).values).any(dim=-1)
        ref_by_pick = torch.gather(torch.zeros_like(scores).scatter(1, picks, ref_w), 1, got)
        eps_w = torch.finfo(torch.float32).eps * ref_by_pick.abs()
        bad_w = ((wts[:, :TOP_K] - ref_by_pick).abs() > 9 * eps_w)
        bad = bad_sets | bad_w.any(dim=-1)
        total += rows
        print(f"layer {L}: {rows} rows from {source}, {int(bad.sum())} failing")
        for r in bad.nonzero()[:8].flatten().tolist():
            sb = (scores[r] + bias).float()
            nine = torch.topk(sb, 9).values.tolist()
            print(f"  row {r}: sets_ok={not bad_sets[r]} weights_ok={not bad_w[r].any()} " +
                  ", ".join(f"{v:.9g}" for v in nine))
        assert not bad.any(), f"layer {L}: {int(bad.sum())} of {rows} rows differ from the dump"
    assert total > 0, "no dump rows were checked"


# -- the latent attention's filled walk ------------------------------------------------------------------------
def filled_check(device: str) -> None:
    """mla_pe.attention with filled=True (TF_LING_FILLED): a call walking every chunk of the cache, as a decode graph
    captured for the window does, equals one walking the chunks its rows can see (eager's nch) bit for bit, and both
    equal today's walk (filled=False); partials past a row's last chunk are never read (NaN there changes nothing).
    Windows inside the first chunk, across a chunk boundary, at the cache's last slots, and a prompt chunk in row
    blocks across a boundary. On a GPU the output is also held to the attention it computes; Triton 3.8's interpreter
    multiplies bf16 tl.dot operands as their raw bits, so on the CPU only the walk is checked (which chunks merge, in
    which order: a skipped or extra chunk changes the bits or reads NaN)."""

    from tensorfold.families.glm5_next.cuda.latent import CHUNK, LatentScratch, chunks_for
    from tensorfold.families.glm_moe_dsa.cuda import mla_pe

    H, LW, PW, nch = 16, 512, 64, 5
    slots = nch * CHUNK
    g = torch.Generator().manual_seed(31)
    cache = torch.randn((slots, LW), generator=g).to(torch.bfloat16)
    pcache = torch.randn((slots, PW), generator=g).to(torch.bfloat16)
    scale = (128 + 64) ** -0.5

    for P, R, part in ((100, 8, None), (1020, 8, None), (slots - 8, 8, None), (1500, 64, 16)):
        qa = torch.randn((R, H, LW), generator=g).to(torch.bfloat16)
        qp = torch.randn((R, H, PW), generator=g).to(torch.bfloat16)

        def run(n: int, filled: bool) -> torch.Tensor:
            s = LatentScratch(R, H, nch, device, lw=LW, part_rows=part)
            if filled:
                for t in (s.po, s.pm, s.pl):
                    t.fill_(float("nan"))           # what a merge must never read
            out = torch.full((R, H, LW), -7.0, dtype=torch.bfloat16, device=device)
            pos = torch.tensor([P], dtype=torch.int32, device=device)
            mla_pe.attention(qa.to(device), qp.to(device), cache.to(device), pcache.to(device), pos, s, scale=scale,
                             nch=n, out=out, filled=filled)
            return out.cpu()

        eager = chunks_for(P + R)
        today, today_eager = run(nch, False), run(eager, False)
        graph, filled_eager = run(nch, True), run(eager, True)
        bits = [t.view(torch.int16) for t in (today, today_eager, graph, filled_eager)]
        assert all(torch.equal(bits[0], b) for b in bits[1:]), (P, R)
        assert not graph.isnan().any() and not (graph == -7.0).all(-1).any(), (P, R)     # every row written
        if device == "cpu":
            continue
        # and it is attention: each row over keys 0..P + r (scores of the latent and the rope key, scaled)
        for r in (0, R // 2, R - 1):
            n = P + r + 1
            sc = (qa[r].double() @ cache[:n].double().T + qp[r].double() @ pcache[:n].double().T) * scale
            want = torch.softmax(sc, dim=-1) @ cache[:n].double()
            err = (graph[r].double() - want).norm() / want.norm()
            assert err < 2e-2, (P, R, r, float(err))


if __name__ == "__main__":
    what = sys.argv[1]
    dev = sys.argv[2] if len(sys.argv) > 2 else ("cuda" if torch.cuda.is_available() else "cpu")
    {"rope": rope_check, "route": route_check, "swiglu": swiglu_check, "router_rows": router_rows_check,
     "filled": filled_check}[what](dev)
    print("ok")
