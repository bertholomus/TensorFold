"""The reindex layers' candidate-only scoring (kernels.index_keys_cand, switch "cand_only"; rounds._candidate_blocks)
against the path it replaces (full-width scores masked to -inf outside the candidate pool, then the top-k): the same
int64 [R, k] indices, bit for bit, with full pools, pools that are not full (rows with fewer visible blocks than the
pool takes, fewer visible positions than k), per-row bases and ties; then their times.

  python3 tools/dsv41/cand_only_test.py [--quick] [--out F]       (one GPU, lane idle; ~2 GB)
  python3 tools/dsv41/cand_only_test.py --cpu                     (the algorithm in torch on the CPU, no Triton)
Exits 1 when a check fails.
"""

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

PAD = -9223372036854775808
NBLOCKS, BSIZE, K = 2048, 8, 512        # candidate_topk_blocks, candidate_block_size, index_topk (the checkpoint's)


def keys_of(score: torch.Tensor) -> torch.Tensor:
    """topk_indices' keys: the score's bits in total order above, the inverted index below."""

    bits = (score + 0.0).view(torch.int32)
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    return (ordered << 32) | (0xFFFFFFFF - torch.arange(score.shape[-1], device=score.device, dtype=torch.int64))


def select_ref(keys: torch.Tensor, k: int, vis: torch.Tensor) -> torch.Tensor:
    top = keys.topk(k, dim=-1, sorted=False).values
    idx = (0xFFFFFFFF - (top & 0xFFFFFFFF)).sort(dim=-1).values
    return torch.where(idx < vis[:, None], idx, -1)


def topk_indices(score: torch.Tensor, k: int) -> torch.Tensor:
    top = keys_of(score).topk(k, dim=-1, sorted=False).values
    return (0xFFFFFFFF - (top & 0xFFFFFFFF)).sort(dim=-1).values


def block_max(score: torch.Tensor, vis: torch.Tensor, bsize: int) -> torch.Tensor:
    width = score.shape[-1]
    s = F.pad(score, (0, -width % bsize), value=float("-inf")).unflatten(-1, (-1, bsize)).amax(-1)
    last = (vis - 1) // bsize
    return s.masked_fill(torch.arange(s.shape[-1], device=score.device)[None] == last[:, None], float("inf"))


def cand_mask(score, vis, nblocks, bsize):
    """model._candidates (the mask the full-width path applies)."""

    s = block_max(score, vis, bsize)
    idx = topk_indices(s, min(nblocks, s.shape[-1]))
    return torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, s.gather(-1, idx) > float("-inf"))


def cand_blocks(score, vis, nblocks, bsize):
    """rounds._candidate_blocks (the block ids the candidate-only path scores)."""

    s = block_max(score, vis, bsize)
    idx = topk_indices(s, min(nblocks, s.shape[-1]))
    return torch.where(s.gather(-1, idx) > float("-inf"), idx, -1).to(torch.int32)


def full_ref(score2, mask, vis, bsize, k):
    """The path replaced: the reindex layer's scores masked outside the pool (apply_candidates), keys, top-k."""

    keep = mask.repeat_interleave(bsize, dim=-1)[:, :score2.shape[-1]]
    return select_ref(keys_of(score2.masked_fill(~keep, float("-inf"))), k, vis)


def compact_keys(score2, cblk, bsize):
    """index_keys_cand in torch: each pool position's full-width key, PAD for -1 blocks."""

    rows, n = score2.shape
    j = torch.arange(cblk.shape[1] * bsize, device=score2.device)
    blk = cblk.long()[:, j // bsize]
    t = blk * bsize + j % bsize
    ok = (blk >= 0) & (t < n)
    full = keys_of(score2)
    return torch.where(ok, full.gather(1, t.clamp(0, n - 1)), PAD)


def test_scores(rows, n, vis, kind, gen):
    if kind == "ties":
        s = torch.randint(0, 5, (rows, n), generator=gen).float() * 0.25
    elif kind == "neg":                      # negative and zero scores (relu dots times negative head weights)
        s = -torch.rand((rows, n), generator=gen)
        s[:, ::7] = 0.0
        s[:, 1::11] = -0.0
    else:
        s = torch.randn((rows, n), generator=gen)
    return s.masked_fill(torch.arange(n)[None] >= vis[:, None], float("-inf"))


def visible_counts(rows, n, gen):
    """Every regime: a full pool (deep rows), a pool that takes every visible block (< 16384 visible), fewer visible
    positions than k, one visible position, the full bucket."""

    v = torch.randint(1, n + 1, (rows,), generator=gen)
    fixed = [n, n - 3, NBLOCKS * BSIZE + 5, NBLOCKS * BSIZE, NBLOCKS * BSIZE - 1, 5000, K, K - 1, 300, 9, 1]
    for i, x in enumerate(fixed[:rows]):
        v[i] = x
    return v


def check_cpu() -> dict:
    gen = torch.Generator().manual_seed(3)
    res, ok = {}, True
    for n in (32768, 65536, 131072):
        for kind in ("random", "ties", "neg"):
            rows = 14
            vis = visible_counts(rows, n, gen)
            s1 = test_scores(rows, n, vis, kind, gen)       # the candidate source's scores
            s2 = test_scores(rows, n, vis, kind, gen)       # a reindex layer's
            mask = cand_mask(s1, vis, NBLOCKS, BSIZE)
            cblk = cand_blocks(s1, vis, NBLOCKS, BSIZE)
            pool = torch.zeros_like(mask)
            r_i, c_i = (cblk >= 0).nonzero(as_tuple=True)
            pool[r_i, cblk[r_i, c_i].long()] = True
            ref = full_ref(s2, mask, vis, BSIZE, K)
            got = select_ref(compact_keys(s2, cblk, BSIZE), K, vis)
            res[f"{n}/{kind}/pool_same"] = bool(torch.equal(mask, pool))
            ok &= res[f"{n}/{kind}/pool_same"]
            res[f"{n}/{kind}/select_equal"] = bool(torch.equal(ref, got))
            ok &= res[f"{n}/{kind}/select_equal"]
    res["passed"] = ok
    return res


def check_gpu(quick: bool) -> dict:
    from tensorfold.families.deepseek_v41.cuda import kernels as Kn
    from tensorfold.families.deepseek_v41.cuda import model as M
    from tensorfold.families.deepseek_v41.cuda import rounds as Rd

    dev = "cuda"
    gen = torch.Generator().manual_seed(11)
    out: dict = {"checks": {}, "times_ms": {}}
    ok = out["checks"]
    ih, idim = 32, 128
    for nb in ((32768, 131072) if quick else (32768, 131072, 1048576)):
        rows = 16
        codes = torch.randint(0, 256, (nb + 4096, idim // 2), generator=gen, dtype=torch.uint8).to(dev)
        scl = torch.randint(118, 132, (nb + 4096, idim // 32), generator=gen, dtype=torch.uint8).to(dev)
        vis = visible_counts(rows, nb, gen).to(dev)
        for with_base in (False, True):
            base = torch.randint(0, 4096, (rows,), generator=gen).to(dev) if with_base else None
            q1 = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
            w1 = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
            q2 = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
            w2 = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
            tag = f"{nb}/base{int(with_base)}"
            # the candidate source: the mask (old) and the block ids (new) of the same scores
            s1 = Kn.index_score(q1, (codes, scl), w1, vis, nb, base=base)
            mask = M._candidates(s1, vis[:, None], NBLOCKS, BSIZE)
            cblk = Rd._candidate_blocks(s1, vis, NBLOCKS, BSIZE)
            pool = torch.zeros_like(mask)
            r_i, c_i = (cblk >= 0).nonzero(as_tuple=True)
            pool[r_i, cblk[r_i, c_i].long()] = True
            ok[f"pool_same/{tag}"] = bool(torch.equal(pool, mask))
            # a reindex layer: masked full width (plain and pruned selection, as rounds.py runs it) vs the pool only
            keys = Kn.index_keys(q2, (codes, scl), w2, vis, nb, base=base, cand=mask, cand_block=BSIZE)
            ref = Kn.topk_select(keys, K, vis)
            keys_t, tm = Kn.index_keys(q2, (codes, scl), w2, vis, nb, base=base, cand=mask, cand_block=BSIZE,
                                       tmax=True)
            ok[f"pruned_is_ref/{tag}"] = bool(torch.equal(Kn.topk_select_pruned(keys_t, tm, K, vis), ref))
            ck = Kn.index_keys_cand(q2, (codes, scl), w2, vis, nb, cblk, BSIZE, base=base)
            got = Kn.topk_select(ck, K, vis)
            ok[f"select_equal/{tag}"] = bool(torch.equal(got, ref))
            # bits: the pool's keys are the unmasked full-width keys at their positions
            nomask = Kn.index_keys(q2, (codes, scl), w2, vis, nb, base=base)
            j = torch.arange(ck.shape[1], device=dev)
            blk = cblk.long()[:, j // BSIZE]
            t = blk * BSIZE + j % BSIZE
            valid = (blk >= 0) & (t < nb)
            exp = torch.where(valid, nomask.gather(1, t.clamp(0, nb - 1)), PAD)
            ok[f"keys_bits/{tag}"] = bool(torch.equal(ck, exp))

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

    for rows, nb in ((1, 131072), (6, 131072), (16, 131072), (4, 1048576), (16, 1048576)):
        codes = torch.randint(0, 256, (nb + 4096, idim // 2), generator=gen, dtype=torch.uint8).to(dev)
        scl = torch.randint(118, 132, (nb + 4096, idim // 32), generator=gen, dtype=torch.uint8).to(dev)
        q = (torch.randn((rows, ih, idim), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        w = (torch.randn((rows, ih), generator=gen) * 0.1).to(torch.bfloat16).to(dev)
        vis = torch.full((rows,), nb, dtype=torch.int64, device=dev)
        base = torch.zeros((rows,), dtype=torch.int64, device=dev)
        s1 = Kn.index_score(q, (codes, scl), w, vis, nb, base=base)
        mask = M._candidates(s1, vis[:, None], NBLOCKS, BSIZE)
        cblk = Rd._candidate_blocks(s1, vis, NBLOCKS, BSIZE)

        def old():
            keys, tm = Kn.index_keys(q, (codes, scl), w, vis, nb, base=base, cand=mask, cand_block=BSIZE, tmax=True)
            return Kn.topk_select_pruned(keys, tm, K, vis)

        def new():
            return Kn.topk_select(Kn.index_keys_cand(q, (codes, scl), w, vis, nb, cblk, BSIZE, base=base), K, vis)

        out["times_ms"][f"reindex layer {rows}x{nb}"] = {"old": timed(old), "new": timed(new),
                                                          "pool_mask_old": timed(lambda: M._candidates(
                                                              s1, vis[:, None], NBLOCKS, BSIZE)),
                                                          "pool_ids_new": timed(lambda: Rd._candidate_blocks(
                                                              s1, vis, NBLOCKS, BSIZE))}
    out["passed"] = all(ok.values())
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    res = check_cpu() if a.cpu else check_gpu(a.quick)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
