"""Checks of the Beam DFlash2 drafter's pieces against plain torch: the host walk's bf16 rounding (bit-equal to
torch's) and its picks (vs the same walk in torch on the GPU), the grouped block convolution with and without the
residual norm, and the windowed block attention over the ring (wrap-around included). Runs on one GPU without the
model.
usage: python tools/beam_drafter_check.py"""

import numpy as np
import torch

from tensorfold.families.beam.cuda import drafter as Dr


def words(t: torch.Tensor) -> np.ndarray:
    return t.bfloat16().view(torch.int16).numpy().view(np.uint16)


def check_bf16() -> None:
    torch.manual_seed(0)
    x = torch.cat([torch.randn(1 << 16) * 10, torch.randn(1 << 12) * 1e-30, torch.tensor([0.0, -0.0, 1.0, -3.5])])
    want = x.bfloat16().float().numpy()
    assert (Dr.bf16(x.numpy()).view(np.uint32) == want.view(np.uint32)).all()
    assert (Dr.f32(words(x)).view(np.uint32) == want.view(np.uint32)).all()
    print("bf16: rounding and widening bit-equal to torch")


def check_walk() -> None:
    torch.manual_seed(0)
    V, C, S, K, n = 4096, 256, 64, 3, 16
    pred, succ = torch.randn(V, C).bfloat16(), torch.randn(V, C).bfloat16()
    d = Dr.DFlash2.__new__(Dr.DFlash2)
    d.pred_book, d.succ_book = words(pred), words(succ)
    cand = torch.randint(0, V, (S, K, n))
    unary = torch.randn(S, K, n) * 4
    hp = torch.randn(S, K, C).bfloat16().float()
    anchors = torch.randint(0, V, (S,))
    got = d.walk(anchors.numpy(), cand.numpy(), unary.numpy(), hp.numpy(), [(0, 7, 0, K, None)] * S)
    pg, sg, cg, ug, hg = pred.cuda(), succ.cuda(), cand.cuda(), unary.cuda(), hp.cuda()
    prev, want = anchors.cuda(), []
    for t in range(K):
        wgt = (pg[prev].float() * hg[:, t]).bfloat16()
        score = ug[:, t] + torch.bmm(sg[cg[:, t]], wgt[:, :, None])[:, :, 0].float()
        prev = cg[:, t].gather(1, score.argmax(-1, keepdim=True))[:, 0]
        want.append(prev.cpu())
    same = (torch.from_numpy(got) == torch.stack(want, 1)).float().mean().item()
    assert same >= 0.97, same                    # a one-ulp tie may break the other way
    print(f"walk: host picks equal the torch walk's on {same:.1%} of {S * K}")


def ref_conv(x, dyn, base, side, B, gs, taps, res=None, nw=None, eps=1e-6):
    R, D = x.shape
    G = D // gs
    delta = dyn.float().view(R, 2, taps, G)[:, side].repeat_interleave(gs, -1)
    coef = base[side].float()[None] + delta
    xf = x.float()
    out = coef[:, 0] * xf
    p = torch.arange(R, device=x.device) % B
    for t in range(1, taps):
        sh = torch.zeros_like(xf)
        sh[t:] = xf[:-t]
        out += coef[:, t] * sh * (p >= t)[:, None]
    y = out.bfloat16()
    if res is None:
        return y, None
    r = (res.float() + y.float()).bfloat16()
    rf = r.float()
    return (rf * torch.rsqrt((rf * rf).mean(-1, keepdim=True) + eps) * nw.float()).bfloat16(), r


def check_conv() -> None:
    torch.manual_seed(0)
    D, gs, taps, B, R = 5120, 16, 2, 4, 12
    d = Dr.DFlash2.__new__(Dr.DFlash2)
    d.taps, d.group, d.rows, d.eps = taps, gs, B, 1e-6
    x = torch.randn(R, D, device="cuda").bfloat16()
    dyn = (0.1 * torch.randn(R, 2 * taps * D // gs, device="cuda")).bfloat16()
    base = torch.randn(2, taps, D, device="cuda").bfloat16()
    res = torch.randn(R, D, device="cuda").bfloat16()
    nw = torch.randn(D, device="cuda").bfloat16()
    for side in (0, 1):
        want, _ = ref_conv(x, dyn, base, side, B, gs, taps)
        got = d._conv(x, dyn, base, side)
        bad = (got.float() - want.float()).abs() > 1e-2 * want.float().abs().clamp(min=1e-3)
        assert bad.float().mean() < 1e-3, (side, bad.float().mean().item())
        want, r_want = ref_conv(x, dyn, base, side, B, gs, taps, res, nw)
        r = res.clone()
        got = d._conv(x, dyn, base, side, r, nw)
        assert (r.float() - r_want.float()).abs().max() <= 0.07, (r.float() - r_want.float()).abs().max()
        rel = ((got.float() - want.float()).abs() / want.float().abs().clamp(min=1e-2)).mean()
        assert rel < 1e-2, rel
    print("conv: matches torch (both sides, with and without the residual norm)")


def check_attention() -> None:
    torch.manual_seed(0)
    slots, ring, HKV, G, B, D, window = 2, 192, 4, 5, 4, 128, 100
    p0s, sl = [250, 37], [1, 0]
    S = len(p0s)
    ck = torch.randn(slots, ring, HKV, D, device="cuda").bfloat16()
    cv = torch.randn(slots, ring, HKV, D, device="cuda").bfloat16()
    qo = torch.randn(S * B, HKV * G, D, device="cuda").bfloat16()
    att = torch.empty(S * B, HKV * G * D, device="cuda", dtype=torch.bfloat16)
    p0 = torch.tensor(p0s, device="cuda")
    slot = torch.tensor(sl, device="cuda")
    scale = D ** -0.5
    Dr._dattn[(S, HKV)](qo, ck, cv, att, p0, slot, qo.stride(0), qo.stride(1), att.stride(0), ring, window, scale,
                        HKV=HKV, G=G, B=B, D=D, BM=32, BK=64, num_warps=4)
    worst = 0.0
    for s in range(S):
        for r in range(B):
            qpos = p0s[s] + r
            kp = torch.arange(max(0, qpos - window), p0s[s] + B, device="cuda")
            k = ck[sl[s], kp % ring].float()
            v = cv[sl[s], kp % ring].float()
            for h in range(HKV * G):
                q = qo[s * B + r, h].float()
                p = torch.softmax(k[:, h // G] @ q * scale, 0)
                want = p @ v[:, h // G]
                got = att[s * B + r, h * D:(h + 1) * D].float()
                worst = max(worst, (got - want).abs().max().item())
    assert worst < 2e-2, worst
    print(f"attention: matches torch over the ring window (max abs diff {worst:.4f})")


if __name__ == "__main__":
    check_bf16()
    check_walk()
    check_conv()
    check_attention()
    print("DRAFTER_CHECK_PASS")
