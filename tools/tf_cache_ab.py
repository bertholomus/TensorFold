"""Cache A/B: the same lane's teacher-forced next-token distributions over its bf16 cache (arm A) and other cache
formats (arms B: TF_AB_ARMS, e.g. "idx8,fp8,q8,q6b,q4/q8"; kv8.parse).

A KLD of the weights says nothing about the cache, so this compares the lane with itself: every arm holds the same
weights and reads the same text; only the cache planes differ. Two probe arms measure the model's own sensitivity:
"bf16" (a second bf16 cache: must give KL 0) and "ulpN" (a bf16 cache whose latent rows, once a prompt chunk has
written them, have the last mantissa bit of N % of their values flipped: a perturbation far below any format's). Each prompt chunk runs through A's state, then each
B arm's (each its own cache, the same prompt kernels), and every row's logits over the whole vocabulary are compared
across the ranks' slices, at every position from TF_AB_FROM (2,048: where DSA's sparse selection starts) on:
  KL(P_A || P_B), whether the top-1 tokens agree, and each arm's NLL of the text's next token.
Then every arm decodes greedily from the prompt's end (TF_AB_DECODE tokens, eager) and the first token where an arm's
reply leaves A's is reported. After the first prompt the bf16 cache's real rows also measure the formats' row error
(relative RMS, in the latent's own domain): FP8 scale granularity (a power-of-two scale a row, an exact amax/448 scale a
row, power-of-two scales a 128- or 64-column tile) and the quantized formats at 8 / 6 / 5 / 4 bits.

Every rank runs the same command (tp4_run.sh, the lane stopped); rank 0 reads and tokenizes the texts, shares the ids
and writes OUT (JSON summary, after every prompt) and OUT.npz (per-position KL, top-1 agreement and NLLs).
usage: python3 tools/tf_cache_ab.py MODEL RANK MASTER PORT CONTEXT OUT SPEC [SPEC ...]
  SPEC = name:path:byte_offset:tokens   (a UTF-8 text in the container, e.g. /tfw/r3/ab/code.utf8)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL, RANK, MASTER, PORT = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
CONTEXT, OUT, SPECS = int(sys.argv[5]), sys.argv[6], sys.argv[7:]
ARMS = [a for a in (os.environ.get("TF_AB_ARMS") or "fp8").split(",") if a]
FROM = int(os.environ.get("TF_AB_FROM", "2048"))
DECODE = int(os.environ.get("TF_AB_DECODE", "256"))


def say(*args) -> None:
    if RANK == 0:
        print(*args, flush=True)


def gather(w, x: torch.Tensor) -> torch.Tensor:
    """Every rank's fp32 tensor x [n, k] -> [world, n, k] (rank order)."""

    x = x.contiguous().float()
    out = torch.empty((w.world * x.numel(),), dtype=torch.float32, device=x.device)
    w.comm.all_gather(x.view(-1), out)
    return out.view(w.world, *x.shape)


def log_partition(w, a: torch.Tensor):
    """Rows of this rank's logit slice -> (log Z over the whole vocabulary, this slice's max, its argmax id)."""

    m = a.max(1)
    s = torch.exp(a - m.values[:, None]).sum(1)
    g = gather(w, torch.stack([m.values, s], 1))
    M = g[:, :, 0].max(0).values
    return M + torch.log((g[:, :, 1] * torch.exp(g[:, :, 0] - M)).sum(0)), m.values, m.indices


def compare(w, a: torch.Tensor, za: torch.Tensor, b16: torch.Tensor, nxt: torch.Tensor):
    """Rows of both arms' logit slices (a fp32 with its log Z) -> per row KL(P_A || P_B), top-1 agreement and each
    arm's NLL of the next token (nxt, global ids), over the whole vocabulary."""

    b = b16.float()                                     # columns past the vocabulary are -inf in both
    zb, vb, ib = log_partition(w, b)
    pa = torch.exp(a - za[:, None])
    t = torch.where(pa > 0, pa * (a - b), torch.zeros_like(pa)).sum(1)
    va, ia = a.max(1)
    off = w.vocab_offset
    local = nxt.to(torch.int64) - off
    mine = (local >= 0) & (local < a.shape[1])
    li = local.clamp(0, a.shape[1] - 1)[:, None]
    la = torch.where(mine, a.gather(1, li)[:, 0], torch.zeros_like(va))
    lb = torch.where(mine, b.gather(1, li)[:, 0], torch.zeros_like(vb))
    h = gather(w, torch.stack([t, va, (ia + off).float(), vb, (ib + off).float(), la, lb], 1))   # [world, n, 7]
    kl = h[:, :, 0].sum(0) - za + zb
    rows = torch.arange(a.shape[0], device=a.device)
    top_a = h[:, :, 2][h[:, :, 1].max(0).indices, rows]        # the first rank with the max: the lowest id
    top_b = h[:, :, 4][h[:, :, 3].max(0).indices, rows]
    return kl, top_a == top_b, za - h[:, :, 5].sum(0), zb - h[:, :, 6].sum(0)


def flip_bits(st, at: int, R: int, share: float) -> None:
    """Flip the lowest mantissa bit of ``share`` of the latent values of rows at .. at + R (each plane, its own
    deterministic pattern): one bf16 ulp on those values."""

    for i, kc in enumerate(st.kc):
        rows = kc[at:at + R]
        g = torch.Generator(device=rows.device).manual_seed(at * 131 + i)
        mask = (torch.rand(rows.shape, generator=g, device=rows.device) < share).to(torch.int16)
        rows.view(torch.int16).bitwise_xor_(mask)


def row_errors(st, n: int) -> dict:
    """Relative RMS error of each format over the bf16 cache's real rows [0, n) of a few layers' latents and indexer
    keys, in the rows' own domain (FP8: e4m3 of x / scale, saturating at 448; Qn: kvq's reference round trip)."""

    from tensorfold.families.glm_moe_dsa.cuda import kv8, kvq

    def fp8(x: torch.Tensor, tile: int, exact: bool) -> torch.Tensor:
        R, W = x.shape
        t = x.view(R, W // tile, tile)
        amax = t.abs().amax(2, keepdim=True)
        if exact:
            scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
        else:
            e = torch.ceil(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - kv8.SHIFT
            scale = torch.where(amax > 0, torch.exp2(e), torch.ones_like(amax))
        return ((t / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale).view(R, W)

    formats = {"fp8 pow2 a row": lambda x: fp8(x, x.shape[1], False), "fp8 amax/448 a row": lambda x: fp8(x, x.shape[1], True),
               "fp8 pow2 a 128-tile": lambda x: fp8(x, 128, False), "fp8 pow2 a 64-tile": lambda x: fp8(x, 64, False)}
    for bits in (8, 6, 5, 4):
        formats[f"q{bits}"] = lambda x, b=bits: kvq.dequantize_reference(*kvq.quantize_reference(x, b), b, x.shape[1])
    out = {}
    picks = {"latent": [st.kc[i] for i in (0, 10, 20, 39, 58, 77)],
             "index": [st.index[i] for i in (0, 5, 10, 20)] if st.index else []}
    for kind, planes in picks.items():
        errs: dict[str, list[float]] = {}
        for plane in planes:
            for r0 in range(0, n, 8192):
                x = plane[r0:min(n, r0 + 8192)]
                xf = x.float()
                norm = float((xf * xf).sum())
                for name, fn in formats.items():
                    if kind == "index" and name.startswith("q"):
                        continue
                    d = fn(x if name.startswith("q") else xf) - xf
                    errs.setdefault(name, [0.0, 0.0])
                    errs[name][0] += float((d * d).sum())
                    errs[name][1] += norm
        out[kind] = {k: math.sqrt(v[0] / v[1]) for k, v in errs.items()}
    return out


@torch.no_grad()
def main() -> None:
    from tensorfold.families.glm5_next.cuda.decode import sample_rows
    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd, glue
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    t0 = time.time()
    eng = GlmEngine(MODEL, rank=RANK, master=MASTER, port=PORT, policy="0", context=CONTEXT, context_explicit=True)
    e, w = eng.e, eng.w
    c = w.cfg
    st_a = e.st
    if st_a.kv != "bf16":
        raise ValueError("arm A is the bf16 cache: run without TF_GLM_KV")
    states = {"A": st_a}
    for arm in ARMS:
        free0 = torch.cuda.mem_get_info()[0]
        states[arm] = fwd.State(w, st_a.capacity, e.rows, kv="bf16" if arm.startswith("ulp") else arm)
        say(f"   arm {arm}: {(free0 - torch.cuda.mem_get_info()[0]) / 2**30:.2f} GiB of cache")
    say(f"== loaded in {time.time() - t0:.0f} s: {st_a.capacity} slots; arms {', '.join(states)}; "
        f"{torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB free")
    app = GlmApp(eng, MODEL, "GLM") if RANK == 0 else None
    b = e.pbuf
    la = torch.empty((e.prefill_rows, w.head.n), dtype=torch.bfloat16, device="cuda")
    lb = torch.empty_like(la)
    results, arrays = [], {}
    errors = None
    for spec in SPECS:
        name, path, offset, count = spec.split(":")
        if RANK == 0:
            raw = Path(path).read_bytes()[int(offset):]
            text = raw.decode("utf-8", errors="ignore")
            ids = app.tok.encode(text[:int(count) * 8], add_special_tokens=False).ids[:int(count)]
        ids = eng._share(ids if RANK == 0 else None)
        L = len(ids)
        if L + DECODE + 8 > st_a.capacity:
            raise ValueError(f"{name}: {L} tokens + {DECODE} decoded past the {st_a.capacity} slots")
        say(f"== {name}: {L} tokens ({path} from byte {offset})")
        for st in states.values():
            st.reset()
        acc = {arm: ([], [], [], []) for arm in ARMS}
        secs = {arm: 0.0 for arm in states}
        ids_dev = torch.tensor(ids, dtype=torch.int64, device="cuda")
        last_rows = {}
        for start in range(0, L, e.prefill_rows):
            chunk = ids[start:start + e.prefill_rows]
            R = len(chunk)
            lo, hi = max(0, FROM - start), min(R, L - 1 - start)    # rows at positions >= FROM with a next token
            za = {}
            for arm, st in states.items():
                out = la if arm == "A" else lb
                torch.cuda.synchronize()
                t = time.perf_counter()
                fwd.stage(w, st, b, chunk)
                at = st.pos
                fwd.compute(w, st, b, R, logits=False, nch=fwd.chunks_for(st, R), host_pos=st.pos)
                if arm.startswith("ulp"):                               # the chunk's latent rows, last bit flipped
                    flip_bits(st, at, R, float(arm[3:]) / 100.0)
                glue.rmsnorm(b.x[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
                fwd.mm(b, b.fnormed[:R], w.head, b.fxs[:R], out[:R])
                fwd.commit(w, st, b, R, R)
                torch.cuda.synchronize()
                secs[arm] += time.perf_counter() - t
                if start + R >= L:
                    last_rows[arm] = out[R - 1:R].clone()               # the prompt's last row: the first reply token
                for r0 in range(lo, hi, 512):                           # 512 rows at a time: ~80 MB a fp32 slice
                    r1 = min(hi, r0 + 512)
                    if arm == "A":
                        za[r0] = log_partition(w, la[r0:r1].float())[0]
                        continue
                    kl, ag, na, nb = compare(w, la[r0:r1].float(), za[r0], lb[r0:r1],
                                             ids_dev[start + r0 + 1:start + r1 + 1])
                    for lst, v in zip(acc[arm], (kl, ag, na, nb)):
                        lst.append(v.cpu())
        if errors is None:
            errors = row_errors(st_a, L)
            say("   row error (relative RMS over real rows): " + json.dumps(
                {k: {f: round(v, 5) for f, v in d.items()} for k, d in errors.items()}))
        # greedy from the prompt's end: each arm's first token from its last prompt row, then eager one-row steps
        replies = {}
        for arm, st in states.items():
            tok = sample_rows(w, last_rows[arm], [L], None)[0]
            reply = [tok]
            for _ in range(DECODE - 1):
                fwd.stage(w, st, e.buf, [tok])
                logits = fwd.compute(w, st, e.buf, 1, nch=fwd.chunks_for(st, 1), host_pos=st.pos)
                tok = sample_rows(w, logits[:1], [st.pos + 1], None)[0]
                fwd.commit(w, st, e.buf, 1, 1)
                reply.append(tok)
            replies[arm] = reply
        for arm in ARMS:
            kl, ag, na, nb = (torch.cat(x).numpy() for x in acc[arm])
            first = next((i for i, (x, y) in enumerate(zip(replies["A"], replies[arm])) if x != y), None)
            row = {"prompt": name, "arm": arm, "tokens": L, "positions": int(kl.size),
                   "kl_mean": float(kl.mean()), "kl_p50": float(np.percentile(kl, 50)),
                   "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()), "top1": float(ag.mean()),
                   # (a handful of text tokens fall outside the head's vocabulary: their NLL is inf in both arms)
                   "ppl_a": float(np.exp(na[np.isfinite(na)].mean())), "ppl_b": float(np.exp(nb[np.isfinite(nb)].mean())),
                   "greedy_first_diff": first, "greedy_tokens": DECODE,
                   "prefill_s": {"A": round(secs["A"], 1), arm: round(secs[arm], 1)}}   # with the head over every row
            results.append(row)
            for key, v in (("kl", kl), ("top1", ag), ("nll_a", na), ("nll_b", nb)):
                arrays[f"{name}.{arm}.{key}"] = v
            say(f"   {arm:5s}: KL mean {row['kl_mean']:.5f} p50 {row['kl_p50']:.5f} p99 {row['kl_p99']:.4f} max "
                f"{row['kl_max']:.3f}; top-1 {100 * row['top1']:.2f}%; ppl A {row['ppl_a']:.4f} B {row['ppl_b']:.4f}; "
                f"greedy first differs at {first if first is not None else f'none of {DECODE}'}; prefill "
                f"{L / secs[arm]:.0f} tok/s (A {L / secs['A']:.0f})")
        if RANK == 0:
            write(results, arrays, errors)
    if RANK == 0 and results:
        for arm in ARMS:
            kl = np.concatenate([arrays[f"{r['prompt']}.{arm}.kl"] for r in results if r["arm"] == arm])
            ag = np.concatenate([arrays[f"{r['prompt']}.{arm}.top1"] for r in results if r["arm"] == arm])
            say(f"== {arm}: all {kl.size} positions: KL mean {kl.mean():.5f}, p99 {np.percentile(kl, 99):.4f}; "
                f"top-1 {100 * ag.mean():.2f}% (gate: mean KL <= 0.01 and top-1 >= 99 %)")
    eng.comm.barrier()
    say("== done")


def write(results, arrays, errors) -> None:
    summary = {"arm_a": "bf16", "from": FROM, "rows": results, "row_error": errors, "all": {}}
    for arm in ARMS:
        kls = [arrays[f"{r['prompt']}.{arm}.kl"] for r in results if r["arm"] == arm]
        if kls:
            kl = np.concatenate(kls)
            ag = np.concatenate([arrays[f"{r['prompt']}.{arm}.top1"] for r in results if r["arm"] == arm])
            summary["all"][arm] = {"positions": int(kl.size), "kl_mean": float(kl.mean()),
                                   "kl_p99": float(np.percentile(kl, 99)), "top1": float(ag.mean())}
    Path(OUT).write_text(json.dumps(summary, indent=1))
    np.savez_compressed(OUT + ".npz", **arrays)


if __name__ == "__main__":
    main()
