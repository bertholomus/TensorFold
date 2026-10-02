"""Whether GLM-5.3's prompt path gives the same bits at two prompt-chunk sizes, and where it first does not: every
layer's latent and rope-key cache rows (and the indexer planes) after prefill at ROWS_A vs ROWS_B, the first layer that
differs, which rows, by how much. One rank on its own node (tf_profile's Alone collectives), TF_TP_WORLD=4.

2026-10-02, 26,198-token prompt, 2,048 vs 4,096 rows: layer 0 identical; from layer 1 on, rows past the dense limit
(2,072 ..) differ by up to 2e-4 in the latent. The indexer's per-token head weights come from torch.mm (cuBLAS), whose
kernel for [M, 6144] @ [6144, 32] depends on M: a row's weights differ in the last bits between M <= 2,049 and
M >= 3,743, and with them the scores and, near ties, the selected tokens.

usage (tf container): python3 tools/check_prompt_chunking.py MODEL [ROWS_A=2048] [ROWS_B=4096] [TOKENS=24000]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

MODEL = Path(sys.argv[1])
ROWS_A = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
ROWS_B = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 24000


def main() -> None:
    from tf_profile import Alone

    from tensorfold.families.glm_moe_dsa.cuda import decode as dec
    from tensorfold.families.glm_moe_dsa.cuda import forward as fwd
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    fwd.PROMPT_REDUCE = "gather"
    # alone, no rank holds the others' embedding rows: keep the whole table (real rows, real routing)
    os.environ["TF_GLM_EMBED_SPLIT"] = "0"
    eng = GlmEngine(MODEL, rank=0, master="127.0.0.1", port=29999, policy="3", context=32768, context_explicit=True,
                    comm=Alone(0, int(os.environ.get("TF_TP_WORLD", "4"))), prefill_rows=max(ROWS_A, ROWS_B))
    e = eng.e
    st = e.st
    app = GlmApp(eng, MODEL, "GLM")
    filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(TOKENS / 13.6)))
    p = app._prepare({"messages": [{"role": "user", "content": filler + "\n\nSummarize."}], "max_tokens": 8,
                      "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
    n = len(p)

    def run(rows):
        e.prefill_rows = rows
        first = dec.prefill(e, p, None)
        torch.cuda.synchronize()
        snap = [kc[:n].clone() for kc in st.kc], [pc[:n].clone() for pc in st.pc], \
               [ix[:n].clone() for ix in (st.index or [])]
        return first, snap

    fa, a = run(ROWS_A)
    fb, b = run(ROWS_B)
    print(f"{n}-token prompt, chunks of at most {ROWS_A} vs {ROWS_B} rows: first token {fa} vs {fb}", flush=True)
    for name, xs, ys in (("latent", a[0], b[0]), ("rope key", a[1], b[1]), ("indexer", a[2], b[2])):
        for layer, (x, y) in enumerate(zip(xs, ys)):
            bad = (x.view(torch.int16) != y.view(torch.int16)).any(dim=1)
            if bad.any():
                rows = bad.nonzero().flatten()
                d = (x.float() - y.float()).abs().max().item()
                print(f"  {name} cache: first difference at slot {layer}: {int(rows.numel())} rows differ, first "
                      f"{rows[:8].tolist()} last {int(rows[-1])}; max |diff| {d:.4g}", flush=True)
                break
        else:
            print(f"  {name} cache: identical in all {len(xs)} slots", flush=True)


if __name__ == "__main__":
    main()
