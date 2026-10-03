"""The lane's logits at every position of the KL reference panel's rows, for kld_ref_score.py (the scoring host's
pipeline/kld-ref): each row's input_ids as a fresh prompt through this engine's prompt path (the lane's settings: cache
format, prompt kernels), the final norm on every row, the head as decode steps run it (the row-invariant EXL3 linear),
the ranks' vocabulary slices gathered on rank 0. Rank 0 either scores each row against the reference in-process
(kld_ref_score's measures) or writes A_DIR/rowNNN.safetensors ("logits" bf16 [L, vocab], "input_ids") for the scorer.

usage (TP4, lane stopped; tp4_run.sh):
  python3 tools/tf_logits.py MODEL RANK MASTER PORT CONTEXT REF_DIR [A_DIR]
  REF_DIR: the reference (manifest.json, rowNNN.safetensors with input_ids and fp32 logits); its kld_ref_score.py is
  imported from TF_KL_SCORER (default REF_DIR/../kld_ref_score.py, else REF_DIR/kld_ref_score.py). With A_DIR the
  rows are written instead of scored. TF_KL_ROWS=N scores the first N rows only. TF_KL_AB=V1,V2,...: score the panel
  once per prompt-path variant in one load (each "+"-joined from deq0 / deqbf16 / deqfp16 / deqauto: TF_GLM_PROMPT_DEQ,
  and inv0 / inv1: TF_GLM_PROMPT_INVARIANT), each with its own summary.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


def scorer(ref: Path):
    for path in (os.environ.get("TF_KL_SCORER"), ref.parent / "kld_ref_score.py", ref / "kld_ref_score.py"):
        if path and Path(path).is_file():
            spec = importlib.util.spec_from_file_location("kld_ref_score", str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("kld_ref_score.py not found: set TF_KL_SCORER")


@torch.no_grad()
def main() -> None:
    from safetensors.torch import load_file, save_file

    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd, glue
    from tensorfold.families.glm_moe_dsa.cuda import x3 as x3mod
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    model, rank, master, port, context = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), \
        int(sys.argv[5])
    ref = Path(sys.argv[6])
    a_dir = Path(sys.argv[7]) if len(sys.argv) > 7 else None

    def say(*a):
        if rank == 0:
            print(*a, flush=True)

    eng = GlmEngine(model, rank=rank, master=master, port=port, policy="3", context=context, context_explicit=True)
    e, w = eng.e, eng.w
    c = w.cfg
    man = json.loads((ref / "manifest.json").read_text()) if rank == 0 else None
    rows = eng._share([man["rows"], man["length"], man["vocab_size"]] if rank == 0 else None)
    n_rows, length, vocab = rows
    n_rows = min(n_rows, int(os.environ.get("TF_KL_ROWS") or n_rows))
    blocks = -(-c.vocab // 128)
    spans = [x3mod.vocab_slice(blocks, w.world, r) for r in range(w.world)]
    width = w.head.n                                    # every rank's padded slice width
    kept = [max(0, min(mine * 128, c.vocab - lo * 128)) for lo, mine, _ in spans]
    mod = scorer(ref) if rank == 0 and a_dir is None else None
    if a_dir is not None and rank == 0:
        a_dir.mkdir(parents=True, exist_ok=True)
    what = f"writing {a_dir}" if a_dir else "scoring"
    say(f"== {n_rows} rows x {length} tokens, vocab {vocab}; settings TF_GLM_KV={os.environ.get('TF_GLM_KV', '')} "
        f"TF_GLM_PROMPT_DEQ={x3mod.PROMPT_DEQ}; {what}")

    def score(label: str) -> None:
        """The panel's rows through the prompt path as it is set now, scored (or written) row by row."""

        per_row, toks, lp, count = [], [], [0.0, 0.0], 0
        for i in range(n_rows):
            t0 = time.perf_counter()
            ids = (load_file(str(ref / f"row{i:03d}.safetensors"))["input_ids"].tolist() if rank == 0 else None)
            ids = eng._share(ids)
            e.reset()
            e.kept = None
            logits = [] if rank == 0 else None
            for start in range(0, len(ids), e.prefill_rows):
                chunk = ids[start:start + e.prefill_rows]
                R = fwd.stage(w, st, b, chunk)
                fwd.compute(w, st, b, R, logits=False, nch=fwd.chunks_for(st, R), host_pos=st.pos)
                glue.rmsnorm(b.x[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
                for r0 in range(0, R, 128):             # the head as decode steps run it, 128 rows a call
                    r1 = min(R, r0 + 128)
                    n = r1 - r0
                    part = torch.empty((n, width), dtype=torch.bfloat16, device=w.device)
                    w.head(b.fnormed[r0:r1], part, b.x3, prefill=False)
                    got = torch.empty((w.world * n * width // 2,), dtype=torch.float32, device=w.device)
                    w.comm.all_gather(part.view(torch.float32).view(-1), got)
                    if rank == 0:
                        g = got.view(torch.bfloat16).view(w.world, n, width)
                        logits.append(torch.cat([g[r, :, :kept[r]] for r in range(w.world)], dim=1).cpu())
                fwd.commit(w, st, b, R, R)
            torch.cuda.synchronize()
            if rank != 0:
                continue
            full = torch.cat(logits)                        # [L, vocab] bf16
            if a_dir is not None:
                save_file({"logits": full, "input_ids": torch.tensor(ids, dtype=torch.int64)},
                          str(a_dir / f"row{i:03d}.safetensors"))
                say(f"row {i:3d}: written ({time.perf_counter() - t0:.1f} s)")
                continue
            rb = load_file(str(ref / f"row{i:03d}.safetensors"))
            s = mod.score_row(full.float(), rb["logits"], rb["input_ids"], vocab)
            toks.append(s.pop("kl_ab_tokens"))
            lp[0] += s["logprob_a"]
            lp[1] += s["logprob_b"]
            count += s["count"]
            per_row.append(s)
            say(f"row {i:3d}: KL(A, B) {s['kl_ab']:.5f}  KL(B, A) {s['kl_ba']:.5f}  top-1 {s['top1_agree']:.4f} "
                f"({time.perf_counter() - t0:.1f} s)")
        if rank == 0 and per_row:
            allt = torch.cat(toks)
            n = len(per_row)
            say(f" -- KL divergence (A, B): {sum(r['kl_ab'] for r in per_row) / n:.8f}")
            say(f" -- KL divergence (B, A): {sum(r['kl_ba'] for r in per_row) / n:.8f}")
            say(f" -- KL (A, B) per token: median {allt.median().item():.8f}   p90 {allt.quantile(0.9).item():.8f}")
            say(f" -- top-1 agreement: {sum(r['top1_agree'] for r in per_row) / n:.4f}")
            say(f" -- perplexity A {math.exp(-lp[0] / count):.6f}   B {math.exp(-lp[1] / count):.6f}")

    st, b = e.st, e.pbuf
    variants = [v for v in (os.environ.get("TF_KL_AB") or "").split(",") if v] or [""]
    for label in variants:
        for setting in [x for x in label.split("+") if x]:
            if setting.startswith("deq"):
                x3mod.PROMPT_DEQ = setting[3:]
            elif setting in ("inv0", "inv1"):
                fwd.invariant.INVARIANT = setting == "inv1"
            else:
                raise SystemExit(f"TF_KL_AB: unknown setting {setting}")
        if label:
            say(f"== variant {label}: TF_GLM_PROMPT_DEQ={x3mod.PROMPT_DEQ} "
                f"TF_GLM_PROMPT_INVARIANT={int(fwd.invariant.INVARIANT)}")
        score(label)
    eng.comm.barrier()


if __name__ == "__main__":
    main()
