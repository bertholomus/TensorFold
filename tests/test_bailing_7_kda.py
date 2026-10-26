"""KDA: Ling's variant of glm5_next's chain (beta = sigmoid(fp32(b)) kept fp32, modeling 856)
against a float64 transcription of the reference recurrence, and the engine's exactness rules bit for bit.

The reference (modeling 787-917): q, k, v through ShortConvolution (4 causal taps, SiLU, bf16 out); fla's
fused_recurrent_kda with the gate in the kernel (decay = exp(lower * sigmoid(exp(A_log) * (f + dt_bias))) per key
channel), q and k L2-normalized in the kernel (eps 1e-6, q scaled by 128^-0.5), the delta rule
S <- S * decay; S <- S + beta k (v - k S); o = q S; then FusedRMSNormGated: w * rmsnorm(o) * sigmoid(g).

GPU only (the chain is a CUDA extension); runs on a GB10.

Run: python -m pytest tests/test_bailing_7_kda.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")


def _have_beta_switch() -> bool:
    """The Ling variant exists once the kda chain takes a beta_bf16 switch."""

    try:
        import inspect

        from tensorfold.families.glm5_next.cuda import kda

        return "beta_bf16" in inspect.signature(kda.chain).parameters
    except Exception:                                   # noqa: BLE001 - no toolchain here: skip
        return False


pytestmark = [pytest.mark.skipif(not _have_beta_switch(), reason="the kda chain has no beta_bf16 switch"),
              pytest.mark.skipif(not torch.cuda.is_available(), reason="the KDA chain is a CUDA extension")]

EPS, LOWER = 1e-6, -5.0                 # Ling's rms_norm_eps and kda_lower_bound


def _inputs(H: int, R: int, seed: int):
    """Random projection rows in the engine's layout: P = q | k | v | b (pre-conv), A = f, G = g, conv window."""

    from tensorfold.families.glm5_next.cuda import kda

    g = torch.Generator().manual_seed(seed)
    C = 3 * H * kda.DK
    b_off = C
    P = (torch.randn((R, C + H), generator=g) * 0.7).to(torch.bfloat16).cuda()
    P[:, b_off:] = (torch.randn((R, H), generator=g) * 2).to(torch.bfloat16).cuda()
    A = torch.randn((R, H * kda.DK), generator=g).to(torch.bfloat16).cuda()
    G = torch.randn((R, H * kda.DV), generator=g).to(torch.bfloat16).cuda()
    cs = (torch.randn((3, C), generator=g) * 0.7).to(torch.bfloat16).cuda()
    cw = (torch.randn((C, 4), generator=g) * 0.4).to(torch.bfloat16).cuda()
    st0 = (torch.randn((H, kda.DV, kda.DK), generator=g) * 0.05).cuda()
    a_log = (torch.rand((H,), generator=g) * 2 - 1).cuda()
    dt = (torch.randn((H * kda.DK,), generator=g) * 0.5).cuda()
    nw = (torch.rand((kda.DV,), generator=g) + 0.5).to(torch.bfloat16).cuda()
    return P, b_off, A, G, cs, cw, st0, a_log, dt, nw


def reference(P, b_off, A, G, cs, cw, st0, a_log, dt, nw, H: int):
    """The reference recurrence in float64 from the same bf16 inputs (conv and SiLU in the kernel's fp32 order, so
    the bf16 rounding of the conv output sees the same value; everything after in float64)."""

    from tensorfold.families.glm5_next.cuda import kda

    R = P.shape[0]
    C = 3 * H * kda.DK
    X = torch.cat([cs, P[:, :C]]).float()
    w = cw.float()
    S = st0.double().clone()                                    # [H, V, K]
    outs = []
    for r in range(R):
        acc = torch.zeros(C, dtype=torch.float32, device=P.device)
        for tap in range(4):                                    # the kernel's fp32 order (no FMA contraction)
            acc = acc + w[:, tap] * X[r + tap]
        act = (acc / (1.0 + torch.exp(-acc))).to(torch.bfloat16).double()      # SiLU, one bf16 rounding
        q, k, v = act.view(3, H, kda.DK)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * kda.DK ** -0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        f = A[r].double().view(H, kda.DK) + dt.double().view(H, kda.DK)
        decay = torch.exp(LOWER * torch.sigmoid(torch.exp(a_log.double())[:, None] * f))     # [H, K]
        beta = torch.sigmoid(P[r, b_off:b_off + H].double())                                # fp32 beta: no rounding
        S = S * decay[:, None, :]
        kv = (S * k[:, None, :]).sum(-1)                                                     # [H, V]
        S = S + ((v - kv) * beta[:, None])[:, :, None] * k[:, None, :]
        o = (S * q[:, None, :]).sum(-1)
        y = o.to(torch.bfloat16).double()                                                    # read-out in bf16
        yn = y / torch.sqrt((y * y).mean(-1, keepdim=True) + EPS)
        outs.append(nw.double() * yn * torch.sigmoid(G[r].double().view(H, kda.DV)))
    return torch.stack(outs), S


def _chain(P, b_off, A, G, cs, cw, st, a_log, dt, nw, R, *, beta_bf16=False, wide=None):
    from tensorfold.families.glm5_next.cuda import kda

    H = a_log.numel()
    sc = kda.KDAScratch(R, H, "cuda")
    out_state = torch.empty_like(st)
    out = kda.chain(P, b_off, A, G, cs, cw, st, a_log, dt, nw, EPS, LOWER, R, sc, out_state, wide=wide,
                    beta_bf16=beta_bf16).clone()
    return out, out_state, sc


@pytest.mark.parametrize("R", [48, 1024])
def test_ling_chain_against_the_reference_recurrence(R):
    """4 heads: outputs within one bf16 read-out step plus their own rounding of the float64 reference (3 * 2^-8
    relative), the fp32 state within 1e-4 of its largest entry (1,024 rows take the three-kernel prompt path, 48 the
    fused chain: WIDE_ROWS is 64)."""

    H = 4
    args = _inputs(H, R, 7 + R)
    out, state, _ = _chain(*args, R)
    ref_out, ref_state = reference(*args, H)
    out = out.double().view(R, H, -1)
    # the fp32 state agrees to ~1e-7, so now and then the kernel's fp32 read-out and the reference's float64 one
    # round to bf16 on either side of a boundary: one bf16 step, <= 2^-7 of the element, which the gated norm keeps
    # in relative size; then the output's own bf16 rounding, <= 2^-8. Slack 2^-7 of that for the fp32 norm and gate
    # (a GB10 run saw 1.05 * 2^-7 at 1,024 rows)
    bound = (2 ** -7 + 2 ** -8) * (1 + 2 ** -7) * ref_out.abs() + 1e-5
    assert ((out - ref_out).abs() <= bound).all(), ((out - ref_out).abs() / bound).max().item()
    err = (state.double() - ref_state).abs().max() / ref_state.abs().max()
    assert err <= 1e-4, err.item()


def test_beta_switch_keeps_fp32_for_ling_and_rounds_for_glm():
    """beta_bf16=False saves sigmoid(b) unrounded (Ling); True saves bf16(sigmoid(b)) (GLM-5.3-Flash's bits)."""

    H, R = 4, 16
    args = _inputs(H, R, 3)
    P, b_off = args[0], args[1]
    want = torch.sigmoid(P[:, b_off:b_off + H].float())
    _, _, ling = _chain(*args, R, beta_bf16=False)
    _, _, glm = _chain(*args, R, beta_bf16=True)
    assert (ling.b[:R] - want).abs().max() <= 2 * torch.finfo(torch.float32).eps
    assert (ling.b[:R].to(torch.bfloat16).float() != ling.b[:R]).float().mean() > 0.9     # not bf16 values
    assert torch.equal(glm.b[:R], glm.b[:R].to(torch.bfloat16).float())
    assert torch.equal(glm.b[:R], want.to(torch.bfloat16).float())


def test_ling_window_equals_serial_steps_and_replay_and_the_wide_path():
    """Bit for bit with beta in fp32: a window's rows equal one-row chains from the evolving state and conv window;
    replaying a kept prefix gives the serial state; the three-kernel prompt path gives the fused chain's bits."""

    from tensorfold.families.glm5_next.cuda import kda

    H, R = 4, 6
    P, b_off, A, G, cs0, cw, st0, a_log, dt, nw = _inputs(H, R, 11)
    C = 3 * H * kda.DK
    win, s_out, sc = _chain(P, b_off, A, G, cs0, cw, st0, a_log, dt, nw, R)
    state, cs = st0.clone(), cs0.clone()
    for r in range(R):
        one, nxt, _ = _chain(P[r:r + 1], b_off, A[r:r + 1], G[r:r + 1], cs, cw, state, a_log, dt, nw, 1)
        assert torch.equal(one[0], win[r]), r
        state = nxt
        cs = torch.cat([cs, P[r:r + 1, :C]])[1:].contiguous()
    assert torch.equal(state, s_out)
    for keep in (1, 3, 5):
        rep = torch.empty_like(st0)
        kda.replay(st0, sc, keep, rep)
        state, cs = st0.clone(), cs0.clone()
        for r in range(keep):
            _, state, _ = _chain(P[r:r + 1], b_off, A[r:r + 1], G[r:r + 1], cs, cw, state, a_log, dt, nw, 1)
            cs = torch.cat([cs, P[r:r + 1, :C]])[1:].contiguous()
        assert torch.equal(rep, state), keep
    # the prompt path (rows >= WIDE_ROWS run three kernels) against the fused chain, Ling's beta
    R = 300
    args = _inputs(H, R, 13)
    got = [_chain(*args, R, wide=wide) for wide in (False, True)]
    (o1, s1, k1), (o2, s2, k2) = got
    assert torch.equal(o1, o2) and torch.equal(s1, s2)
    for name in ("k", "v", "g", "b"):
        assert torch.equal(getattr(k1, name)[:R], getattr(k2, name)[:R]), name
