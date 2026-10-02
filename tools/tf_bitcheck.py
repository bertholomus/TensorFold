"""A build's bits on a fixed long prompt, to compare two builds (or settings) of the same lane: the prompt's first
token, every cache plane's checksum, the last hidden row, then greedy decode (serial, then MTP-3) with each step's
token. Rank 0 writes OUT (JSON); compare two with ``python3 tools/tf_bitcheck.py --diff A.json B.json``.

usage (TP4, lane stopped; tp4_run.sh): python3 tools/tf_bitcheck.py MODEL RANK MASTER PORT CONTEXT OUT [TOKENS=26000]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def plane_sums(st, n: int) -> list[int]:
    import torch

    out = []
    planes = list(st.kc) + list(st.pc) + list(st.index or []) + [st.mtp_kc, st.mtp_pc]
    for x in planes:
        if hasattr(x, "checksum"):
            out.append(int(x.checksum(n)))
        else:
            out.append(int(x[:n].view(torch.int16).to(torch.int64).sum()))
    return out


def diff(a_path: str, b_path: str) -> None:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    for k in a:
        same = a[k] == b.get(k)
        detail = "" if same else (f"  ({sum(x != y for x, y in zip(a[k], b[k]))} of {len(a[k])} differ)"
                                  if isinstance(a[k], list) else f"  {a[k]} vs {b.get(k)}")
        print(f"{k}: {'same' if same else 'DIFFERENT'}{detail}")


def main() -> None:
    import torch

    torch.set_grad_enabled(False)
    from tensorfold.families.glm_moe_dsa.cuda import decode as dec
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    model, rank, master, port = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    context, out = int(sys.argv[5]), sys.argv[6]
    n = int(sys.argv[7]) if len(sys.argv) > 7 else 26000
    eng = GlmEngine(model, rank=rank, master=master, port=port, policy="3", context=context, context_explicit=True)
    e = eng.e
    app = GlmApp(eng, model, "GLM")
    filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(n / 13.6)))
    p = app._prepare({"messages": [{"role": "user", "content": filler + "\n\nSummarize the list in three sentences."}],
                      "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
                     True).prompt
    res = {"prompt_tokens": len(p)}
    first = dec.prefill(e, p, None)
    res["first_token"] = first
    res["cache_sums"] = plane_sums(e.st, len(p))
    res["mtp_len"] = e.st.mtp_len
    res["last_hidden"] = hashlib.sha256(e.last_hidden.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]
    s = dec.serial_decode(e, first, 64, None, stop_eos=False)
    res["serial_tokens"] = s.tokens
    first = dec.prefill(e, p, None)
    m = dec.mtp_decode(e, first, 64, None, policy=dec.DepthPolicy(3, fixed=True), stop_eos=False)
    res["mtp3_tokens"] = m.tokens
    res["mtp3_accepted"] = m.accepted
    if rank == 0:
        Path(out).write_text(json.dumps(res))
        print(f"== {len(p)}-token prompt: first token {first}, serial == MTP-3: {s.tokens == m.tokens}, "
              f"MTP-3 accepted {m.accepted} of {m.drafted}; wrote {out}", flush=True)
    eng.comm.barrier()


if __name__ == "__main__":
    if sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        main()
