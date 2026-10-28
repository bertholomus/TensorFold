"""The router's fp32 logits at prompt sizes: ops._router's tiles (rows a program, experts a program, K step, warps) at
2,048 and 4,096 rows on one layer's real router (fp32 [256, 5120]) and fp32 rows (the boundary's yf, as the prompt
path passes them), us a call, best first; every tiling's bits against today's ops.router_logits at 129 to 4,096 rows
(each output one FMA chain over K in order, whatever the tile: ROUTER_SAME / ROUTER_DIFFERS).
usage: python tools/beam_router_bench.py [PACK_DIR]"""

import json
import sys
import time
from itertools import product
from pathlib import Path

import torch
import triton
from safetensors import safe_open

from tensorfold.families.beam.cuda import ops


def logits(x, w, bt, be, bk, warps):
    T, K = x.shape
    out = torch.empty(T, w.shape[0], device="cuda", dtype=torch.float32)
    ops._router[(triton.cdiv(T, bt), triton.cdiv(w.shape[0], be))](x, w, out, T, K, x.stride(0), E=w.shape[0], BT=bt,
                                                                   BE=be, BK=bk, num_warps=warps)
    return out


def timed(f, n=20):
    for _ in range(3):
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
    p = "model.layers.10.mlp.gate.weight"
    with safe_open(str(pack / wm[p]), "pt", device="cuda") as f:
        W = f.get_tensor(p).float().contiguous()
    E, K = W.shape
    torch.manual_seed(0)
    xs = {T: torch.randn(T, K, device="cuda") for T in (129, 300, 1500, 2048, 4096)}
    want = {T: ops.router_logits(x, W).view(torch.int32).clone() for T, x in xs.items()}
    base = {T: timed(lambda: ops.router_logits(xs[T], W)) for T in (2048, 4096)}
    print(f"today: {base[2048]:.0f} us at 2048 rows, {base[4096]:.0f} us at 4096", flush=True)
    rows, bad = [], []
    for bt, be, bk, warps in product((32, 64, 128), (16, 32, 64, 128), (16, 32, 64), (2, 4, 8)):
        try:
            t2 = timed(lambda: logits(xs[2048], W, bt, be, bk, warps))
        except Exception as e:                       # noqa: BLE001  (registers or shared memory past the limit)
            print(f"BT{bt} BE{be} BK{bk} w{warps}: {type(e).__name__}", flush=True)
            continue
        same = all(torch.equal(logits(x, W, bt, be, bk, warps).view(torch.int32), want[T]) for T, x in xs.items())
        if not same:
            bad.append(f"BT{bt}/BE{be}/BK{bk}/w{warps}")
        rows.append((t2, timed(lambda: logits(xs[4096], W, bt, be, bk, warps)), bt, be, bk, warps, same))
    for t2, t4, bt, be, bk, warps, same in sorted(rows)[:12]:
        print(f"BT{bt:3d} BE{be:3d} BK{bk:2d} w{warps}: {t2:6.0f} us at 2048 ({(base[2048] - t2) / base[2048] * 100:+.0f} %), "
              f"{t4:6.0f} at 4096 ({(base[4096] - t4) / base[4096] * 100:+.0f} %) {'SAME' if same else 'DIFFERS'}")
    print("ROUTER_SAME" if not bad else "ROUTER_DIFFERS: " + "; ".join(bad))
