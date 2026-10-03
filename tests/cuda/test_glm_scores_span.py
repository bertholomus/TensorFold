"""TF_GLM_SCORES_SPAN: prompt chunks score their rows' lightning-indexer terms with _scores_span (a program's rows'
queries held across a span of key tiles). Against _scores (a program a 64-token tile, queries reloaded for each) the
scores agree to fp32 rounding for bf16, FP8 and Q8 keys, a row's scores do not depend on where its block starts, and
past each row's position they are -inf."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

H, D = 32, 128


def keys_of(n: int, kind: str, g: torch.Generator):
    from tensorfold.families.glm_moe_dsa.cuda import kv8, kvq

    k = (torch.randn((n, D), generator=g) * 0.5).to(torch.bfloat16).cuda()
    zero = torch.zeros((1,), dtype=torch.int32, device="cuda")
    if kind == "bf16":
        return k, (k, k, k), False, 0
    if kind == "fp8":
        p = kv8.Kv8(n, D, "cuda")
        kv8.write(k, p, zero)
        return p, (p.codes, p.scales, p.scales), True, 0
    p = kvq.KvQ(n, D, 8, "cuda")
    kvq.write(k, p, zero)
    return p, (p.codes, p.scales, p.h), False, 8


def run(kernel, qi, wts, keys, R, pos, np_max, kv8, qb, rb_ts=None):
    import triton

    from tensorfold.families.glm_moe_dsa.cuda import select

    ik, isc, hq = keys
    out = torch.full((R, np_max), 7.0, dtype=torch.float32, device="cuda")
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    if kernel == "span":
        select._scores_span[(triton.cdiv(R, select.SPAN_RB), triton.cdiv(np_max, select.SPAN_TS))](
            qi, wts, wts.stride(0), ik, isc, hq, out, pos_dev, R, np_max, D ** -0.5, H ** -0.5, H=H, HP=32, D=D,
            BT=64, RB=select.SPAN_RB, TS=select.SPAN_TS, KV8=kv8, QB=qb, num_warps=4)
    else:
        select._scores[(triton.cdiv(R, 16), triton.cdiv(np_max, 64))](
            qi, wts, wts.stride(0), ik, isc, hq, out, pos_dev, R, np_max, D ** -0.5, H ** -0.5, H=H, HP=32, D=D,
            BT=64, RB=16, KV8=kv8, QB=qb, num_warps=4)
    return out


@pytest.mark.parametrize("kind", ["bf16", "fp8", "q8"])
@pytest.mark.parametrize("R,pos", [(64, 5000), (130, 2900), (600, 12000)])
def test_span_scores_match_the_tile_scores(kind, R, pos):
    g = torch.Generator().manual_seed(R + pos + len(kind))
    n = pos + R
    _, keys, kv8, qb = keys_of(n, kind, g)
    qi = (torch.randn((R, H * D), generator=g) * 0.3).to(torch.bfloat16).cuda()
    wts = (torch.randn((R, H), generator=g)).to(torch.bfloat16).cuda()
    np_max = -(-n // 64) * 64 + 64                     # a tile past the last row's tokens
    a = run("span", qi, wts, keys, R, pos, np_max, kv8, qb)
    b = run("tile", qi, wts, keys, R, pos, np_max, kv8, qb)
    fin = torch.isfinite(b)
    assert torch.equal(fin, torch.isfinite(a))         # the same -inf past each row's position
    t = torch.arange(np_max, device="cuda")
    assert torch.equal(fin, t[None, :] <= (pos + torch.arange(R, device="cuda"))[:, None])
    err = float(((a[fin] - b[fin]).abs() / (b[fin].abs() + 1e-3)).max())
    assert err < 1e-4, err


def test_a_rows_span_scores_do_not_depend_on_its_block():
    g = torch.Generator().manual_seed(3)
    R, pos = 67, 4000
    n = pos + R
    _, keys, kv8, qb = keys_of(n, "fp8", g)
    qi = (torch.randn((R, H * D), generator=g) * 0.3).to(torch.bfloat16).cuda()
    wts = (torch.randn((R, H), generator=g)).to(torch.bfloat16).cuda()
    np_max = -(-n // 64) * 64
    whole = run("span", qi, wts, keys, R, pos, np_max, kv8, qb)
    for start in (1, 2, 3, 5):                        # the same rows at other places in their 4-row blocks
        part = run("span", qi[start:].contiguous(), wts[start:].contiguous(), keys, R - start, pos + start, np_max,
                   kv8, qb)
        assert torch.equal(part.view(torch.int32), whole[start:].view(torch.int32)), start
