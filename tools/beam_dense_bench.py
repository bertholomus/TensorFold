"""Beam's dense prompt projections on one GB10: rank 0 of 2's columns / rows of one MoE layer's attention (q, k, v,
gate: hidden -> the rank's heads, bf16 out; o: the rank's heads -> hidden, an fp32 partial) and of one dense layer's
MLP (gate, up; down an fp32 partial), ms a 2,048-row call of exl3/prefill.matmul as serving calls it (the unpack
inside, no unpack cache), for each setting: the rotated path with the default tiles (today) and by shape
(TF_EXL3_PREFILL_TILES=1); the dequantized one on _gemm_deq fp16 (TF_EXL3_PREFILL_DEQ=fp16), on cuBLAS bf16 (bf16) and
each call's faster one (auto), cuBLAS plain and in fixed row blocks (the GLM prompt path's invariant.blocked). Each
setting's rows are checked against the same rows in calls of 1,500, 300 and 129 rows, and with 37 other rows first: a
row's bits must not depend on its chunk (INVARIANT / DEPENDS). Prints one line a projection and setting (its largest
difference from today's path, over the output's RMS), then each setting's total for a chunk over the model's layers.
usage: python tools/beam_dense_bench.py [PACK_DIR] [MOE_LAYER] [DENSE_LAYER]"""

import sys
import time
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import prefill as P
from tensorfold.families.beam.config import Config
from tensorfold.families.beam.cuda import weights as W
from tensorfold.families.glm_moe_dsa.cuda import invariant

M = 2048
CHECK = (1500, 300, 129)
SHIFT = 37
SETTINGS = (("rotated", False, "0", 0), ("rotated by shape", True, "0", 0), ("deq fp16", None, "fp16", 0),
            ("deq bf16", None, "bf16", 0), ("deq bf16 blocks 512", None, "bf16", 512),
            ("deq bf16 blocks 1024", None, "bf16", 1024), ("deq bf16 blocks 2048", None, "bf16", 2048),
            ("deq auto", None, "auto", 0), ("deq auto blocks 1024", None, "auto", 1024))


def timed(f, n=10):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def runner(m0):
    return (lambda fn, R, ins, outs: invariant.blocked(fn, R, ins, outs, M0=m0)) if m0 else None


if __name__ == "__main__":
    pack_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/models/Beam-1.0-501B-A23B-EXL3")
    LM = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    LD = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dev = torch.device("cuda")
    cfg = Config.read(pack_dir)
    pack = W.Pack(pack_dir)
    hd = cfg.head_dim
    q, kv, mlp = W.span(cfg.heads * hd, 2, 0, unit=hd), W.span(cfg.kv_heads * hd, 2, 0, unit=hd), \
        W.span(cfg.intermediate, 2, 0)
    a, d = f"{W.PREFIX}{LM}.self_attn.", f"{W.PREFIX}{LD}.mlp."
    projs = [("q", W.linear(pack, a + "q_proj", dev, cols=q), torch.bfloat16, cfg.layers),
             ("k", W.linear(pack, a + "k_proj", dev, cols=kv), torch.bfloat16, cfg.layers),
             ("v", W.linear(pack, a + "v_proj", dev, cols=kv), torch.bfloat16, cfg.layers),
             ("gate", W.linear(pack, a + "gate_proj", dev, cols=q), torch.bfloat16, cfg.layers),
             ("o", W.linear(pack, a + "o_proj", dev, rows=q), torch.float32, cfg.layers)]
    dense = sum(not cfg.routed(L) for L in range(cfg.layers))
    projs += [("mlp gate", W.linear(pack, d + "gate_proj", dev, cols=mlp), torch.bfloat16, dense),
              ("mlp up", W.linear(pack, d + "up_proj", dev, cols=mlp), torch.bfloat16, dense),
              ("mlp down", W.linear(pack, d + "down_proj", dev, rows=mlp), torch.float32, dense)]
    ws = P.Workspace()
    torch.manual_seed(0)
    total = {s[0]: 0.0 for s in SETTINGS}
    bad = []
    for name, lin, dt, count in projs:
        x = torch.randn((M, lin.k), device=dev).to(torch.bfloat16)
        x2 = torch.cat([torch.randn((SHIFT, lin.k), device=dev).to(torch.bfloat16), x])
        ref = None
        for label, by_shape, deq, m0 in SETTINGS:
            blocks = runner(m0)

            def call(rows, n=M):
                out = torch.empty((n, lin.n), dtype=dt, device=dev)
                return P.matmul(lin, rows, out, ws, by_shape=by_shape, deq=deq, blocks=blocks)

            full = call(x).clone()
            if ref is None:
                ref = full.float()
            same = all(torch.equal(bits(call(x[:R], R)), bits(full[:R])) for R in CHECK)
            same = same and all(torch.equal(bits(call(x2[:SHIFT + R], SHIFT + R)[SHIFT:]), bits(full[:R]))
                                for R in CHECK)
            err = float((full.float() - ref).abs().max() / ref.pow(2).mean().sqrt())
            ms = timed(lambda: call(x))
            total[label] += ms * count
            if not same and m0:
                bad.append(f"{name} {label}")
            print(f"{name:8s} {M}x{lin.k}->{lin.n} {str(dt)[6:]:8s} {label:22s} {ms:6.3f} ms  diff/rms {err:.2e}  "
                  f"{'INVARIANT' if same else 'DEPENDS'}", flush=True)
    base = total["rotated"]
    for label, *_ in SETTINGS:
        print(f"CHUNK {label:22s} {total[label]:7.1f} ms a {M}-row chunk ({(base - total[label]) / base * 100:+.1f} %)")
    print("BLOCKS_INVARIANT" if not bad else "BLOCKS_DEPEND: " + "; ".join(bad))
