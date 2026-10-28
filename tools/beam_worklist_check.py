"""Do the prompt expert kernels keep their bits with the work list and with 128-row mma3 programs? One layer's routed
experts as rank 0 of 2 serves them (gate/up columns and down rows [0, 1280) of 2,560), prompt chunks of 2,048 and 4,096
rows whose picks come from the layer's router on (1) random rows (spread), (2) 20 rows repeated (word soup: a few
experts take most rows) and (3) every row on the same 8 experts. Each chunk runs kernels mma and mma3 with the grid
(TF_EXL3_GROUP_LIST=0) and with the work list, and mma3 with 128-row gate/up programs (TF_EXL3_MMA3_ROWS=128, grid and
list), every buffer the kernels write filled with NaN first: the outputs must be the same bits (and mma's those of the
one-tile path, at 2,048 rows: its grouping holds a window's picks in shared memory). Times each launch setting (mean of
20 calls).
usage: python tools/beam_worklist_check.py [PACK_DIR] [LAYER]"""

import sys
import time
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import experts as X
from tensorfold.families.beam.config import Config
from tensorfold.families.beam.cuda import ops
from tensorfold.families.beam.cuda import weights as W


def call(y, idx, wt, ex, s, kernel, group_list, prompt=None, rows3=64):
    X.GROUP_LIST = group_list
    X.MMA3_ROWS = rows3
    for t in (s.z, s.xd, s.xg, s.xu):
        if t is not None:
            t.fill_(float("nan"))
    return X.routed(y, idx, wt, ex, s, None, y.shape[0], act_mode=X.ACT_F32, prompt=prompt, kernel=kernel,
                    y_unused=True)


def timed(f, n=20):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


if __name__ == "__main__":
    pack_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/models/Beam-1.0-501B-A23B-EXL3")
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dev = torch.device("cuda")
    cfg = Config.read(pack_dir)
    pack = W.Pack(pack_dir)
    ex, slabs = W.experts(pack, L, cfg, 0, 2, dev)
    p = f"{W.PREFIX}{L}.mlp.gate."
    router = pack.get(p + "weight", dev).contiguous()
    bias = pack.get(p + "e_score_correction_bias", dev).float().contiguous()
    torch.manual_seed(0)
    bad = []
    for R in (2048, 4096):
        s = X.Scratch(ex, R, cfg.top_k, device=dev, prompt=True)
        y = torch.randn(R, cfg.hidden, device=dev).to(torch.bfloat16)
        soup = torch.randn(20, cfg.hidden, device=dev).to(torch.bfloat16)[torch.arange(R, device=dev) % 20]
        cases = []
        for name, rows in (("spread", y), ("soup", soup)):
            idx, wt = ops.route(ops.router_logits(rows.float(), router), bias, groups=cfg.groups,
                                top_groups=cfg.top_groups, top_k=cfg.top_k)
            cases.append((name, rows, idx, wt * cfg.routed_scale))
        cases.append(("same8", y, cases[0][2][:1].expand(R, -1).contiguous(),
                      cases[0][3][:1].expand(R, -1).contiguous()))
        for name, rows, idx, wt in cases:
            counts = torch.bincount(idx.reshape(-1).long(), minlength=cfg.experts)[:cfg.experts]
            busiest, nwork = int(counts.max()), int(((counts + 63) // 64).sum())
            used = int((counts > 0).sum())
            line = f"{R} rows {name:6s} ({used} experts used, busiest {busiest}, {nwork} groups of 64 against a grid of " \
                   f"{-(-busiest // 64) * min(R * cfg.top_k, cfg.experts)})"
            print(line, flush=True)
            for kernel in ("mma", "mma3"):
                grid = call(rows, idx, wt, ex, s, kernel, False).clone()
                listed = call(rows, idx, wt, ex, s, kernel, True).clone()
                again = call(rows, idx, wt, ex, s, kernel, False).clone()
                same = (torch.equal(grid.view(torch.int32), listed.view(torch.int32)) and
                        torch.equal(grid.view(torch.int32), again.view(torch.int32)))
                if not same:
                    bad.append(f"{R} rows {name} {kernel}: list against grid")
                if bool(torch.isnan(listed).any()):
                    bad.append(f"{R} rows {name} {kernel}: NaN in the output")
                tg = timed(lambda: call(rows, idx, wt, ex, s, kernel, False))
                tl = timed(lambda: call(rows, idx, wt, ex, s, kernel, True))
                print(f"  {kernel:4s}: grid {tg:7.2f} ms, list {tl:7.2f} ms ({(tg - tl) / tg * 100:+.1f} %) "
                      f"{'SAME' if same else 'DIFFERS'}", flush=True)
                if kernel == "mma3":
                    wide = [call(rows, idx, wt, ex, s, kernel, gl, rows3=128).clone() for gl in (False, True)]
                    same = all(torch.equal(grid.view(torch.int32), w.view(torch.int32)) for w in wide)
                    if not same:
                        bad.append(f"{R} rows {name} mma3: 128-row programs against 64")
                    t1 = timed(lambda: call(rows, idx, wt, ex, s, kernel, True, rows3=128))
                    print(f"  mma3 128-row programs, list {t1:7.2f} ms ({(tl - t1) / tl * 100:+.1f} % vs 64) "
                          f"{'SAME' if same else 'DIFFERS'}", flush=True)
                if kernel == "mma" and R * cfg.top_k <= 16384:       # the one-tile grouping's shared memory
                    one = call(rows, idx, wt, ex, s, None, True, prompt=False).clone()
                    if not torch.equal(one.view(torch.int32), listed.view(torch.int32)):
                        bad.append(f"{R} rows {name} mma: list against the one-tile path")
                        print("  mma list against the one-tile path: DIFFERS", flush=True)
        del s
        torch.cuda.empty_cache()
    print("WORKLIST_SAME" if not bad else "WORKLIST_DIFFERS: " + "; ".join(bad))
