"""Whether GLM-5.3's prompt path gives the same bits at two prompt-chunk sizes, and where it first does not: every
layer's latent and rope-key cache rows (and the indexer planes) after prefill at ROWS_A vs ROWS_B, the first layer that
differs, which rows, by how much. One rank on its own node (tf_profile's Alone collectives), TF_TP_WORLD=4.

2026-10-02, 26,198-token prompt, 2,048 vs 4,096 rows: layer 0 identical; from layer 1 on, rows past the dense limit
(2,072 ..) differ by up to 2e-4 in the latent. The indexer's per-token head weights come from torch.mm (cuBLAS), whose
kernel for [M, 6144] @ [6144, 32] depends on M: a row's weights differ in the last bits between M <= 2,049 and
M >= 3,743, and with them the scores and, near ties, the selected tokens.

usage (tf container): python3 tools/check_prompt_chunking.py MODEL [ROWS_A=2048] [ROWS_B=4096] [TOKENS=24000] [ROWS_C]
  Quantized or FP8 planes (TF_GLM_KV) compare their codes and scales. With TF_GLM_PROMPT_INVARIANT=1 every chunk size
  must give identical caches (ROWS_C: a third size, e.g. 3000, compared with ROWS_A too).
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
ROWS_C = int(sys.argv[5]) if len(sys.argv) > 5 else 0


def _rows(x, n: int) -> torch.Tensor:
    """A plane's first n slots as one int16 view: a bf16 tensor's rows, or an FP8 / quantized plane's codes and scales."""

    if isinstance(x, torch.Tensor):
        return x[:n].contiguous().view(torch.int16).view(n, -1)
    codes = x.codes[:n].contiguous().view(torch.uint8).view(n, -1)
    scales = x.scales[:n].contiguous().view(torch.uint8).view(n, -1)
    both = torch.cat([codes, scales], dim=1)
    if both.shape[1] % 2:
        both = torch.cat([both, torch.zeros_like(both[:, :1])], dim=1)
    return both.view(torch.int16)


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
                    comm=Alone(0, int(os.environ.get("TF_TP_WORLD", "4"))), prefill_rows=max(ROWS_A, ROWS_B, ROWS_C))
    e = eng.e
    st = e.st
    app = GlmApp(eng, MODEL, "GLM")
    filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(TOKENS / 13.6)))
    p = app._prepare({"messages": [{"role": "user", "content": filler + "\n\nSummarize."}], "max_tokens": 8,
                      "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
    n = len(p)

    def run(rows):
        e.prefill_rows = rows
        e.kept = None                                   # (a fresh prefill every time)
        first = dec.prefill(e, p, None)
        torch.cuda.synchronize()
        snap = [_rows(kc, n).clone() for kc in st.kc], [_rows(pc, n).clone() for pc in st.pc], \
               [_rows(ix, n).clone() for ix in (st.index or [])]
        return first, snap

    def compare(ra, rb, a, b, fa, fb):
        print(f"{n}-token prompt, chunks of at most {ra} vs {rb} rows: first token {fa} vs {fb}", flush=True)
        for name, xs, ys in (("latent", a[0], b[0]), ("rope key", a[1], b[1]), ("indexer", a[2], b[2])):
            for layer, (x, y) in enumerate(zip(xs, ys)):
                bad = (x != y).any(dim=1)
                if bad.any():
                    rows = bad.nonzero().flatten()
                    print(f"  {name} cache: first difference at slot {layer}: {int(rows.numel())} rows differ, first "
                          f"{rows[:8].tolist()} last {int(rows[-1])}", flush=True)
                    break
            else:
                print(f"  {name} cache: identical in all {len(xs)} slots", flush=True)

    fa, a = run(ROWS_A)
    fb, b = run(ROWS_B)
    compare(ROWS_A, ROWS_B, a, b, fa, fb)
    if ROWS_C:
        fc, cc = run(ROWS_C)
        compare(ROWS_A, ROWS_C, a, cc, fa, fc)


if __name__ == "__main__":
    main()
