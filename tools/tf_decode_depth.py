"""Decode speed at depth, no HTTP: real text cut to each depth, prefilled once, then serial decoding and MTP-3 decoding
under each MTP index-reuse setting (TF_GLM_MTP_REUSE 0 / 1 / 2, switched in-process), with tok/s, drafted/accepted and
whether every MTP reply equals the serial one. The cache format is this process's TF_GLM_KV.

usage (TP4, lane stopped; tp4_run.sh):
  python3 tools/tf_decode_depth.py MODEL RANK MASTER PORT CONTEXT TEXT DEPTHS [TOKENS=128] [REUSE=0,1,2]
  TEXT: a UTF-8 file (the prompt is its first DEPTH tokens and a question); DEPTHS: e.g. 32000,128000,240000
Each depth prefills once; every decode run starts from the prompt's state (positions rewound: decoding writes past it).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


@torch.no_grad()
def main() -> None:
    from tensorfold.families.glm_moe_dsa.cuda import decode as dec, mtp as mtp_mod
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    model, rank, master, port, context = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), \
        int(sys.argv[5])
    text_path, depths = sys.argv[6], [int(d) for d in sys.argv[7].split(",")]
    tokens = int(sys.argv[8]) if len(sys.argv) > 8 else 128
    modes = (sys.argv[9] if len(sys.argv) > 9 else "0,1,2").split(",")

    def say(*a):
        if rank == 0:
            print(*a, flush=True)

    t0 = time.time()
    eng = GlmEngine(model, rank=rank, master=master, port=port, policy="3", context=context, context_explicit=True)
    e = eng.e
    say(f"== loaded in {time.time() - t0:.0f} s: cache {e.st.kv}, {e.st.capacity} slots")
    app = GlmApp(eng, model, "GLM") if rank == 0 else None
    question = "\n\nWhat is the text above about? Answer in a few paragraphs."
    for depth in depths:
        if rank == 0:
            raw = Path(text_path).read_text(errors="ignore")
            body = app.tok.decode(app.tok.encode(raw[:depth * 8], add_special_tokens=False).ids[:depth])
            p = app._prepare({"messages": [{"role": "user", "content": body + question}], "max_tokens": tokens,
                              "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, True).prompt
        p = eng._share(p if rank == 0 else None)
        torch.cuda.synchronize()
        t = time.perf_counter()
        first = dec.prefill(e, p, None)
        torch.cuda.synchronize()
        pre = time.perf_counter() - t
        pos0, mtp0 = e.st.pos, e.st.mtp_len               # decoding writes past these only: each run starts here
        h0 = e.last_hidden.clone()                        # the prompt's last row: a chain's first draft reads it

        def rewind() -> None:
            e.st.set_pos(pos0)
            e.st.set_mtp_len(mtp0)
            e.st.mtp_drafted = 0
            e.last_hidden = h0.clone()

        s = dec.serial_decode(e, first, tokens, None, stop_eos=False)
        say(f"== {len(p)}-token prompt: prefill {pre:.1f} s ({len(p) / pre:.0f} tok/s); serial {s.tokens_per_second:.2f}"
            f" tok/s")
        for mode in modes:
            mtp_mod.MTP_REUSE = mode
            rewind()
            m = dec.mtp_decode(e, first, tokens, None, policy=dec.DepthPolicy(3, fixed=True), stop_eos=False)
            st = ", ".join(f"{k} {1e3 * v / max(1, m.rounds):.2f}" for k, v in m.stages.items())
            say(f"   MTP-3 reuse {mode}: {m.tokens_per_second:.2f} tok/s, accepted {m.accepted} of {m.drafted} "
                f"({m.accepted / max(1, m.drafted):.3f}), {m.rounds} rounds [ms/round: {st}]; == serial: "
                f"{m.tokens == s.tokens}")
    eng.comm.barrier()
    say("== done")


if __name__ == "__main__":
    main()
