"""Does the routing keep its bits? (1) The router's fp32 logits at every tile setting (16- to 64-wide expert tiles, 1 to
8 warps, K steps of 32 to 256) against the first one (BT 16/32, BE 64, BK 64, 4 warps): each output is one FMA chain
over K in order whatever the tile. (2) The fused pick (ops._route_pick: torch's fp32 sigmoid, the bias, the group
top-2 sums and the top groups in one kernel, then torch's top_k) against torch's ops throughout: the same scores, ids
and weights, on 4M sigmoid values, random logits, one layer's real router on random inputs, and rows built with ties.
Times each at the decode row counts. Runs on one GPU; reads one layer's router and bias from the pack.
usage: python tools/beam_route_check.py [PACK_DIR]"""

import json
import sys
import time
from pathlib import Path

import torch
import triton
from safetensors import safe_open

from tensorfold.families.beam.cuda import ops

G, TG, TOPK = 8, 4, 8


def logits(x, w, be, bk, warps):
    T, K = x.shape
    E = w.shape[0]
    out = torch.empty(T, E, device="cuda", dtype=torch.float32)
    bt = 16 if T <= 16 else 32
    ops._router[(triton.cdiv(T, bt), triton.cdiv(E, be))](x, w, out, T, K, x.stride(0), E=E, BT=bt, BE=be, BK=bk,
                                                          num_warps=warps)
    return out


def both(lg, bias):
    ops.ROUTE_FUSED = False
    i0, w0 = ops.route(lg, bias, groups=G, top_groups=TG, top_k=TOPK)
    ops.ROUTE_FUSED = True
    i1, w1 = ops.route(lg, bias, groups=G, top_groups=TG, top_k=TOPK)
    return torch.equal(i0, i1) and torch.equal(w0.view(torch.int32), w1.view(torch.int32))


def timed(f, n=300):
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


if __name__ == "__main__":
    pack = Path(sys.argv[1] if len(sys.argv) > 1 else "/models/Beam-1.0-501B-A23B-EXL3")
    wm = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    p = "model.layers.10.mlp.gate."
    with safe_open(str(pack / wm[p + "weight"]), "pt", device="cuda") as f:
        W = f.get_tensor(p + "weight").float().contiguous()
        bias = f.get_tensor(p + "e_score_correction_bias").float().contiguous()
    E, K = W.shape
    torch.manual_seed(0)
    bad = []
    for T in (1, 4, 16, 32, 2048):
        x = torch.randn(T, K, device="cuda")
        want = logits(x, W, 64, 64, 4)
        for be, bk, warps in ((32, 64, 4), (16, 64, 4), (16, 128, 4), (16, 64, 8), (16, 32, 2), (16, 64, 1),
                              (16, 256, 8)):
            if not torch.equal(logits(x, W, be, bk, warps).view(torch.int32), want.view(torch.int32)):
                bad.append(f"router T={T} BE{be}/BK{bk}/w{warps}")
        if T <= 16:
            print(f"router {T} rows: first tiling {timed(lambda: logits(x, W, 64, 64, 4)):.0f} us, "
                  f"served {timed(lambda: ops.router_logits(x, W)):.0f} us", flush=True)
    x = torch.cat([torch.linspace(-40, 40, 2_000_000, device="cuda"), torch.randn(2_000_000, device="cuda") * 6])
    x = x.view(-1, E)
    s, c = torch.empty_like(x), torch.empty_like(x)
    ops._route_pick[(x.shape[0],)](x, torch.zeros(E, device="cuda"), s, c, E=E, G=G, TG=TG, num_warps=4)
    if not torch.equal(s.view(torch.int32), x.sigmoid().view(torch.int32)):
        bad.append("sigmoid")
    for b in (bias, torch.randn(E, device="cuda") * 0.05):
        for sigma in (0.5, 2.0, 5.0):
            if not both(torch.randn(65536, E, device="cuda") * sigma, b):
                bad.append(f"pick on random logits, sigma {sigma}")
    for T in (1, 4, 16, 2048, 32768):
        if not both(ops.router_logits(torch.randn(T, K, device="cuda"), W), bias):
            bad.append(f"pick on the real router, {T} rows")
    lg = torch.randn(4096, E, device="cuda")
    v = lg.view(4096, G, -1)
    v[:, :, 1] = v[:, :, 0]                     # a group's best twice
    v[:2048, 3, :] = v[:2048, 5, :]             # two groups with equal sums
    if not both(lg, torch.zeros(E, device="cuda")):
        bad.append("pick on tie rows")
    for T in (4, 16):
        lg = ops.router_logits(torch.randn(T, K, device="cuda"), W)
        ops.ROUTE_FUSED = False
        a = timed(lambda: ops.route(lg, bias, groups=G, top_groups=TG, top_k=TOPK))
        ops.ROUTE_FUSED = True
        print(f"route {T} rows: torch {a:.0f} us, fused pick "
              f"{timed(lambda: ops.route(lg, bias, groups=G, top_groups=TG, top_k=TOPK)):.0f} us", flush=True)
    print("ROUTE_SAME" if not bad else "ROUTE_DIFFERS: " + "; ".join(bad))
