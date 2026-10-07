"""The indexer's exact pruned top-k (kernels.topk_select_pruned: the k tiles with the largest tile maxima, switch
"topk_prune") and the prompt path's keyed selection (switch "prompt_keys") against the paths they replace: the same int64 [R, k] indices, bit for bit, on adversarial scores (ties, -inf stretches,
-0 / +0, sizes that are not powers of two), then their times.

  python3 tools/dsv41/topk_select_test.py [--quick] [--out F]       (one GPU; ~1 GB)
  python3 tools/dsv41/topk_select_test.py --cpu                     (the algorithm in torch on the CPU, no Triton)
Exits 1 when a check fails.
"""

import argparse
import json
import sys
import time

import torch

PAD = -9223372036854775808


def keys_of(score: torch.Tensor) -> torch.Tensor:
    """topk_indices' keys (kernels.topk_indices): the score's bits in total order above, the inverted index below."""

    bits = (score + 0.0).view(torch.int32)
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    return (ordered << 32) | (0xFFFFFFFF - torch.arange(score.shape[-1], device=score.device, dtype=torch.int64))


def select_ref(keys: torch.Tensor, k: int, vis: torch.Tensor) -> torch.Tensor:
    """torch.where(topk_indices(...) < vis, ..., -1) from keys: the reference selection."""

    top = keys.topk(k, dim=-1, sorted=False).values
    idx = (0xFFFFFFFF - (top & 0xFFFFFFFF)).sort(dim=-1).values
    return torch.where(idx < vis[:, None], idx, -1)


def pruned_ref(keys: torch.Tensor, k: int, bn: int = 64) -> torch.Tensor:
    """The pruning in plain torch (topk_select_pruned's algorithm): the keys of the k tiles whose maxima are the k
    largest tile maxima (PAD elsewhere)."""

    rows, n = keys.shape
    nt = -(-n // bn)
    pad = torch.full((rows, nt * bn), PAD, dtype=torch.int64, device=keys.device)
    pad[:, :n] = keys
    tiles = pad.view(rows, nt, bn)
    best = tiles.max(-1).values.topk(min(k, nt), dim=-1).indices
    return torch.gather(tiles, 1, best[:, :, None].expand(-1, -1, bn)).reshape(rows, -1)


def scores(rows: int, n: int, kind: str, gen: torch.Generator, device: str) -> torch.Tensor:
    if kind == "random":
        s = torch.randn((rows, n), generator=gen, device="cpu")
    elif kind == "ties":                 # few distinct values: every tie broken by the lower index
        s = torch.randint(0, 7, (rows, n), generator=gen, device="cpu").float() * 0.5 - 1.0
    elif kind == "zeros":                # -0 and +0 (the same key) and some negatives
        s = torch.zeros((rows, n))
        s[:, ::3] = -0.0
        s[:, 1::5] = -1.0
    else:                                # "neginf": long -inf stretches, fewer finite scores than k in some rows
        s = torch.randn((rows, n), generator=gen, device="cpu")
        for r in range(rows):
            keep = int(torch.randint(1, max(2, n // (r + 2)), (1,), generator=gen))
            s[r, keep:] = float("-inf")
    return s.to(device)


def check_cpu() -> dict:
    gen = torch.Generator().manual_seed(1)
    ok, n_cases = True, 0
    for n in (4097, 5000, 12345, 65536, 300001):
        for kind in ("random", "ties", "zeros", "neginf"):
            sc = scores(3, n, kind, gen, "cpu")
            keys = keys_of(sc)
            vis = torch.tensor([n, n // 2, 1], dtype=torch.int64)
            ref = select_ref(keys, 512, vis)
            got = select_ref(pruned_ref(keys, 512), 512, vis)      # the selection over the k best tiles' keys
            ok &= bool(torch.equal(ref, got))
            n_cases += 1
    return {"cpu_pruned_equal": ok, "cases": n_cases}


def check_gpu(quick: bool) -> dict:
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    dev = "cuda"
    gen = torch.Generator().manual_seed(7)
    out: dict = {"checks": {}, "times_ms": {}}
    ok = out["checks"]
    ih, idim = 32, 128

    def tile_max_ref(keys, bn=64):
        rows, n = keys.shape
        nt = -(-n // bn)
        pad = torch.full((rows, nt * bn), PAD, dtype=torch.int64, device=keys.device)
        pad[:, :n] = keys
        return pad.view(rows, nt, bn).max(-1).values

    # 1. the prompt path: old (fp32 scores, topk_indices, mask) == keys + tile maxima + topk_select_pruned, on both
    #    score kernels (<= 16 rows: one a program; more: 8 a program); the one-row kernel's scores vs the 8-row one's
    for n_comp in ((5000, 70000, 262145) if quick else (5000, 70000, 262145, 524289, 1048577)):
        codes = torch.randint(0, 256, (n_comp + 2, idim // 2), generator=gen, dtype=torch.uint8).to(dev)
        scl = torch.randint(118, 132, (n_comp + 2, idim // 32), generator=gen, dtype=torch.uint8).to(dev)
        for rows in (16, 40):
            q = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
            w = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
            vis = torch.randint(n_comp // 2, n_comp + 1, (rows,), generator=gen).to(dev)
            vis[-1] = n_comp
            score = K.index_score(q, (codes, scl), w, vis, n_comp)
            top = K.topk_indices(score, 512)
            ref = torch.where(top < vis[:, None], top, -1)
            keys, tm = K.index_score(q, (codes, scl), w, vis, n_comp, keys=True, tmax=True)
            ok[f"prompt/keys/{n_comp}/{rows}"] = bool(torch.equal(keys, keys_of(score)))
            ok[f"prompt/tmax/{n_comp}/{rows}"] = bool(torch.equal(tm, tile_max_ref(keys)))
            ok[f"prompt/select_plain/{n_comp}/{rows}"] = bool(torch.equal(K.topk_select(keys, 512, vis), ref))
            ok[f"prompt/select_pruned/{n_comp}/{rows}"] = bool(torch.equal(K.topk_select_pruned(keys, tm, 512, vis),
                                                                          ref))
            if rows > 16:
                rw = K.index_score(q, (codes, scl), w, vis, n_comp, rowwise=True)
                out.setdefault("info", {})[f"rowwise_scores_equal_8row/{n_comp}"] = bool(torch.equal(rw, score))

    # 2. adversarial keys straight into the pruned selection (ties, -inf stretches, zeros): tile maxima by torch
    for n in ((5000, 65536, 300001) if quick else (5000, 65536, 300001, 1048590)):
        for kind in ("random", "ties", "zeros", "neginf"):
            rows = 6
            keys = keys_of(scores(rows, n, kind, gen, dev))
            vis = torch.randint(1, n + 1, (rows,), generator=gen).to(dev)
            vis[0] = n
            ok[f"pruned/{n}/{kind}"] = bool(torch.equal(K.topk_select_pruned(keys, tile_max_ref(keys), 512, vis),
                                                        select_ref(keys, 512, vis)))

    # 3. decode rounds: index_keys (per-row bases, candidate masks) with tile maxima + pruned == the plain path
    for nb in ((8192, 131072) if quick else (8192, 131072, 1048576)):
        rows = 16
        codes = torch.randint(0, 256, (nb + 4096, idim // 2), generator=gen, dtype=torch.uint8).to(dev)
        scl = torch.randint(118, 132, (nb + 4096, idim // 32), generator=gen, dtype=torch.uint8).to(dev)
        q = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        w = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
        vis = torch.randint(1, nb + 1, (rows,), generator=gen).to(dev)
        base = torch.randint(0, 4096, (rows,), generator=gen).to(dev)
        cand = torch.rand((rows, nb // 8), generator=gen).to(dev) < 0.3
        for with_cand in (False, True):
            cd = cand if with_cand else None
            keys = K.index_keys(q, (codes, scl), w, vis, nb, base=base, cand=cd, cand_block=8)
            ref = K.topk_select(keys, 512, vis)
            keys2, tm = K.index_keys(q, (codes, scl), w, vis, nb, base=base, cand=cd, cand_block=8, tmax=True)
            ok[f"decode/keys_same/{nb}/cand{int(with_cand)}"] = bool(torch.equal(keys, keys2))
            ok[f"decode/pruned/{nb}/cand{int(with_cand)}"] = bool(torch.equal(K.topk_select_pruned(keys2, tm, 512, vis),
                                                                              ref))
        # the candidate-source layer: fp32 scores (for the candidate pool), score_keys with tile maxima + pruned
        score = K.index_score(q, (codes, scl), w, vis, nb, base=base)
        ref = K.topk_select(K.score_keys(score), 512, vis)
        keys3, tm3 = K.score_keys(score, tmax=True)
        ok[f"decode/score_keys_same/{nb}"] = bool(torch.equal(keys3, K.score_keys(score)))
        ok[f"decode/score_keys_tmax/{nb}"] = bool(torch.equal(tm3, tile_max_ref(keys3)))
        ok[f"decode/cand_source_pruned/{nb}"] = bool(torch.equal(K.topk_select_pruned(keys3, tm3, 512, vis), ref))

    # 4. times (ms, median after 3 warm-ups)
    def timed(fn, reps=10):
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

    for rows, n_comp in ((32, 524289), (16, 1048577), (128, 131073)):
        codes = torch.randint(0, 256, (n_comp + 2, idim // 2), generator=gen, dtype=torch.uint8).to(dev)
        scl = torch.randint(118, 132, (n_comp + 2, idim // 32), generator=gen, dtype=torch.uint8).to(dev)
        q = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        w = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
        vis = torch.full((rows,), n_comp, dtype=torch.int64, device=dev)

        def old():
            score = K.index_score(q, (codes, scl), w, vis, n_comp)
            top = K.topk_indices(score, 512)
            return torch.where(top < vis[:, None], top, -1)

        def new(rowwise=False):
            keys, tm = K.index_score(q, (codes, scl), w, vis, n_comp, keys=True, tmax=True, rowwise=rowwise)
            return K.topk_select_pruned(keys, tm, 512, vis)

        out["times_ms"][f"prompt block {rows}x{n_comp}"] = {"old": timed(old), "new": timed(new),
                                                            "new_rowwise": timed(lambda: new(True))}
    for rows, nb in ((6, 131072), (16, 1048576)):
        codes = torch.randint(0, 256, (nb + 4096, idim // 2), generator=gen, dtype=torch.uint8).to(dev)
        scl = torch.randint(118, 132, (nb + 4096, idim // 32), generator=gen, dtype=torch.uint8).to(dev)
        q = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        w = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
        vis = torch.full((rows,), nb, dtype=torch.int64, device=dev)
        base = torch.zeros((rows,), dtype=torch.int64, device=dev)

        def dold():
            return K.topk_select(K.index_keys(q, (codes, scl), w, vis, nb, base=base), 512, vis)

        def dnew():
            keys, tm = K.index_keys(q, (codes, scl), w, vis, nb, base=base, tmax=True)
            return K.topk_select_pruned(keys, tm, 512, vis)

        out["times_ms"][f"decode {rows}x{nb}"] = {"old": timed(dold), "new": timed(dnew)}
    out["passed"] = all(ok.values())
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    res = check_cpu() if a.cpu else check_gpu(a.quick)
    passed = res["cpu_pruned_equal"] if a.cpu else res["passed"]
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
