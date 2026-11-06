"""CP4 kernels against the torch ops (DESIGN-CP4 §4 M6). On a node at real dims; on a CPU host with TRITON_INTERPRET=1
at sizes the interpreter finishes (it takes the same code path, so a wrong index shows there first).

    python tests/test_ml4_cp4_kernels.py            # needs a GPU, or TRITON_INTERPRET=1 and CP4_SMALL=1
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch                                                     # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1"
SMALL = os.environ.get("CP4_SMALL") == "1"
DEV = "cpu" if INTERP else "cuda"

import importlib.util                                            # noqa: E402

BASE = SRC / "tensorfold" / "families" / "mistral_large4" / "cuda"


def load(name):
    """By path: the package import pulls the EXL3 extension, which a CPU host does not have."""

    spec = importlib.util.spec_from_file_location(f"cp4_test_{name}", BASE / f"{name}.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kern = load("cp4_kernels")
fp8kv = load("fp8kv")


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.pow(2).mean().sqrt())


def test_scores(R, n, IH=32, ID=128):
    g = torch.Generator().manual_seed(R * 7 + n)
    q = (torch.randn(R, IH, ID, generator=g)).to(torch.float16).to(DEV)
    k = (torch.randn(n + 3, ID, generator=g)).to(torch.float16).to(DEV)
    w = (torch.randn(R, IH, generator=g) * 0.02).float().to(DEV)
    vis = torch.randint(0, n + 1, (R,), generator=g).to(torch.int32).to(DEV)
    vis[0] = n
    out = torch.empty((R, n), dtype=torch.float32, device=DEV)
    kern.index_scores(q, w, k, vis, n, out)
    ref = torch.einsum("qh,qht->qt", w.double(), torch.relu(torch.einsum("qhd,td->qht", q.double(), k[:n].double())))
    mask = torch.arange(n, device=DEV)[None, :] < vis[:, None].long()
    assert bool((out[~mask] == float("-inf")).all()), "a hidden key got a score"
    c, m = cos(out[mask], ref[mask]), rel(out[mask], ref[mask])
    print(f"  scores R {R} n {n}: cosine {c:.9f}, max-abs/rms {m:.2e}", flush=True)
    assert c >= 0.999999 and m <= 2e-3


def test_mla(Q, cap, K, rope, H=64):
    g = torch.Generator().manual_seed(Q * 13 + cap + rope)
    rd = 64 if rope else 0
    fmt = fp8kv.row_format("f32x128", 512, rd)
    lat = torch.randn(cap, 512, generator=g)
    kpe = torch.randn(cap, 64, generator=g) if rope else None
    plane = fp8kv.encode(lat, kpe, fmt).to(DEV)
    dl, dk = fp8kv.decode(plane.cpu(), fmt)
    q_lat = (torch.randn(H, Q, 512, generator=g) * 0.3).to(torch.float16).to(DEV).transpose(0, 1)
    q_pe = (torch.randn(Q, H, 64, generator=g) * 0.3).to(torch.float16).to(DEV) if rope else None
    cnt = torch.randint(0, K + 1, (Q,), generator=g).to(torch.int32).to(DEV)
    cnt[0], cnt[-1] = K, 0
    rows = torch.full((Q, K), 2 ** 31 - 1, dtype=torch.int32)
    for i in range(Q):
        c = int(cnt[i])
        rows[i, :c] = torch.randperm(cap, generator=g)[:c].sort().values.to(torch.int32)
    rows = rows.to(DEV)
    sm = 128 ** -0.5
    out = torch.empty((H, Q, 512), dtype=torch.float16, device=DEV).transpose(0, 1)
    lse = torch.empty((Q, H), dtype=torch.float32, device=DEV)
    kern.sparse_mla(q_lat, q_pe, rows, cnt, plane, sm, out, lse)
    worst_c, worst_m, worst_l = 1.0, 0.0, 0.0
    for i in range(Q):
        c = int(cnt[i])
        if not c:
            assert bool((lse[i] == float("-inf")).all()) and bool((out[i] == 0).all()), "an empty query kept something"
            continue
        r = rows[i, :c].long().cpu()
        # the kernel rounds the decoded latent to fp16 before the dot, so the reference does too
        L = dl[r].to(torch.float16).double()
        s = q_lat[i].double().cpu() @ L.t()
        if rope:
            s = s + q_pe[i].double().cpu() @ dk[r].to(torch.float16).double().t()
        s = s * sm
        want = torch.softmax(s, -1) @ L
        worst_c = min(worst_c, cos(out[i].cpu(), want))
        worst_m = max(worst_m, rel(out[i].cpu(), want))
        worst_l = max(worst_l, float((lse[i].double().cpu() - torch.logsumexp(s, -1)).abs().max()))
    print(f"  mla Q {Q} cap {cap} K {K} rope {rope}: cosine {worst_c:.7f}, max-abs/rms {worst_m:.2e}, lse {worst_l:.2e}",
          flush=True)
    assert worst_c >= 0.9999 and worst_l <= 1e-2


def test_fp8(m, n=4096, k=6144):
    """wq_b from its bytes against the fp16 weight the loader makes. Its output rounds to fp16: 2^-11 of the largest value."""

    g = torch.Generator().manual_seed(m)
    w = (torch.randn(n, k, generator=g) * 2).to(torch.float8_e4m3fn).to(DEV)
    s = (torch.rand(n // 128, k // 128, generator=g) * 1e-3 + 1e-4).float().to(DEV)
    x = torch.randn(m, k, generator=g).to(torch.float16).to(DEV)
    w16 = (w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float16)
    out = kern.fp8_linear(x, w, s)
    want = x.double() @ w16.double().t()
    c, r = cos(out, want), rel(out, want)
    said = f"  fp8 linear M {m}: cosine {c:.7f}, max-abs/rms {r:.2e}"
    if not INTERP:
        def timed(fn):
            fn()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t) / 20 * 1e3

        said += (f", {timed(lambda: kern.fp8_linear(x, w, s)):.3f} ms, fp16 F.linear "
                 f"{timed(lambda: torch.nn.functional.linear(x, w16)):.3f} ms")
    print(said, flush=True)
    assert c >= 0.99999 and r <= 4e-3


def main():
    t0 = time.time()
    if SMALL:
        test_fp8(3, 256, 256)
        test_fp8(40, 256, 256)
        test_scores(2, 200)
        test_scores(70, 300)
        test_mla(3, 200, 70, 1, H=16)
        test_mla(3, 200, 70, 0, H=16)
    else:
        for m in (1, 3, 16, 17, 1024):
            test_fp8(m)
        for R, n in [(1, 40000), (4, 300), (5, 3000), (1024, 4096), (4096, 20000)]:
            test_scores(R, n)
        for Q, cap, K, rope in [(1, 300000, 2048, 1), (1, 300000, 2048, 0), (64, 50000, 2048, 1), (257, 20000, 512, 0)]:
            test_mla(Q, cap, K, rope)
    print(f"kernels ok {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
