"""The decode indexer's per-row tile skip (switch "tile_skip": _index_score skips the dots of key tiles wholly at or
past a row's visible count and, for the pruned selection only, writes no keys there) against the kernels before it
(``--old``: a copy of kernels.py without the switch): the same fp32 scores, the same keys, the same selected
indices, bit for bit, on decode windows with per-row bases, mixed visible counts (1 short + long rows, rows with fewer
live tiles than k, vis 0 and 1), candidate masks; then the mixed-round times.

  python3 tools/dsv41/tile_skip_test.py --old OLD_KERNELS_PY [--quick] [--out F]     (one GPU, lane idle)
Exits 1 when a check fails.
"""

import argparse
import importlib.util
import json
import sys
import time

import torch


def load_old(path):
    spec = importlib.util.spec_from_file_location("kernels_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import kernels as Kn
    Ko = load_old(a.old)
    assert "tile_skip" in Kn.SMALL_SWITCHES and "tile_skip" not in Ko.SMALL_SWITCHES

    dev = "cuda"
    gen = torch.Generator().manual_seed(17)
    ih, idim, k = 32, 128, 512
    out: dict = {"checks": {}, "times_ms": {}}
    ok = out["checks"]
    for nb in ((8192, 131072) if a.quick else (8192, 65536, 131072, 1048576)):
        rows = 16
        codes = torch.randint(0, 256, (nb + 4096, idim // 2), generator=gen, dtype=torch.uint8).to(dev)
        scl = torch.randint(118, 132, (nb + 4096, idim // 32), generator=gen, dtype=torch.uint8).to(dev)
        q = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        w = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
        vis = torch.randint(1, nb + 1, (rows,), generator=gen)
        for i, v in enumerate((nb, 2000, 1, 0, 64, 65, 63, k * 64 - 1, k * 64 + 1, nb - 64, 300)):
            vis[i] = min(v, nb)
        vis = vis.to(dev)
        base = torch.randint(0, 4096, (rows,), generator=gen).to(dev)
        cand = torch.rand((rows, nb // 8), generator=gen).to(dev) < 0.3
        for with_base in (False, True):
            bs = base if with_base else None
            tag = f"{nb}/base{int(with_base)}"
            for sw in (True, False):
                Kn.set_switch("tile_skip", sw)
                ok[f"scores/{tag}/skip{int(sw)}"] = bool(torch.equal(
                    Kn.index_score(q, (codes, scl), w, vis, nb, base=bs),
                    Ko.index_score(q, (codes, scl), w, vis, nb, base=bs)))
                for cd in (None, cand):
                    ctag = f"{tag}/cand{int(cd is not None)}/skip{int(sw)}"
                    ko = Ko.index_keys(q, (codes, scl), w, vis, nb, base=bs, cand=cd, cand_block=8)
                    ok[f"keys/{ctag}"] = bool(torch.equal(
                        Kn.index_keys(q, (codes, scl), w, vis, nb, base=bs, cand=cd, cand_block=8), ko))
                    ref = Ko.topk_select(ko, k, vis)
                    kn, tm = Kn.index_keys(q, (codes, scl), w, vis, nb, base=bs, cand=cd, cand_block=8, tmax=True,
                                           pruned_k=k)
                    ok[f"pruned_select/{ctag}"] = bool(torch.equal(Kn.topk_select_pruned(kn, tm, k, vis), ref))
            Kn.set_switch("tile_skip", True)

    def timed(fn, reps=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t0))
        return round(sorted(ts)[len(ts) // 2], 3)

    # mixed rounds: a long row at its bucket's depth beside short rows (a 4K agent turn beside a deep stream), and a
    # solo stream at 60% of its bucket (the bucket doubles past the deepest row)
    for nb, viss in ((1048576, (1048576, 4096)), (1048576, (1048576,) * 3 + (4096,) * 3),
                     (1048576, (629146,) * 4), (131072, (131072, 4096)), (131072, (78643,) * 4)):
        rows = len(viss)
        codes = torch.randint(0, 256, (nb + 4096, idim // 2), generator=gen, dtype=torch.uint8).to(dev)
        scl = torch.randint(118, 132, (nb + 4096, idim // 32), generator=gen, dtype=torch.uint8).to(dev)
        q = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        w = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
        vis = torch.tensor(viss, dtype=torch.int64, device=dev)
        base = torch.zeros((rows,), dtype=torch.int64, device=dev)

        def sel(K, **kw):
            keys, tm = K.index_keys(q, (codes, scl), w, vis, nb, base=base, tmax=True, **kw)
            return K.topk_select_pruned(keys, tm, k, vis)

        res = {"keys+select old": timed(lambda: sel(Ko)), "keys+select new": timed(lambda: sel(Kn, pruned_k=k)),
               "scores old": timed(lambda: Ko.index_score(q, (codes, scl), w, vis, nb, base=base)),
               "scores new": timed(lambda: Kn.index_score(q, (codes, scl), w, vis, nb, base=base))}
        ok[f"timed_equal/{nb}/{viss}"] = bool(torch.equal(sel(Ko), sel(Kn, pruned_k=k)))
        out["times_ms"][f"{nb} vis {list(viss)}"] = res
    out["passed"] = all(ok.values())
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
