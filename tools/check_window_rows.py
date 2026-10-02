"""Does a decode window's first row get the serial step's bits at depth? After a long prompt, the same token runs as a
1-row window and as the first row of an R-row window (eager, then the captured graphs); every layer's output row and
the selection of each full indexer layer are compared, so the first layer where they part is named.

usage (TP4, lane stopped; tp4_run.sh): python3 tools/check_window_rows.py MODEL RANK MASTER PORT CONTEXT [TOKENS] [R]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


@torch.no_grad()
def main() -> None:
    from tensorfold.families.glm_moe_dsa.cuda import decode as dec, forward as fwd
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    model, rank, master, port, context = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), \
        int(sys.argv[5])
    n = int(sys.argv[6]) if len(sys.argv) > 6 else 26000
    R = int(sys.argv[7]) if len(sys.argv) > 7 else 4

    def say(*a):
        if rank == 0:
            print(*a, flush=True)

    eng = GlmEngine(model, rank=rank, master=master, port=port, policy="3", context=context, context_explicit=True)
    e, w, st = eng.e, eng.w, eng.e.st
    app = GlmApp(eng, model, "GLM")
    filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(int(n / 13.6)))
    p = app._prepare({"messages": [{"role": "user", "content": filler + "\n\nSummarize the list in three sentences."}],
                      "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
                     True).prompt
    first = dec.prefill(e, p, None)
    pos = st.pos
    window = [first, 1140, 5610, 220, 16, 11, 24, 98965][:R]
    say(f"== {len(p)}-token prompt, window at {pos}: {window}")

    orig = fwd.layer_forward
    rec: list = []

    def recording(layer, w_, st_, b, R_, *a, **k):
        orig(layer, w_, st_, b, R_, *a, **k)
        rec.append((layer.index, b.x[:1].clone(), b.tokens[:1].clone(), b.counts[:1].clone(), b.iw[:1].clone(),
                    b.qi[:1].clone(), b.ik[:1].clone()))

    def run(rows: list[int]):
        rec.clear()
        st.set_pos(pos)
        fwd.layer_forward = recording
        try:
            fwd.stage(w, st, e.buf, rows)
            logits = fwd.compute(w, st, e.buf, len(rows), nch=fwd.chunks_for(st, len(rows)), host_pos=st.pos)
        finally:
            fwd.layer_forward = orig
        torch.cuda.synchronize()
        return logits[:1].clone(), list(rec)

    one, rec1 = run(window[:1])
    many, recR = run(window)
    say(f"   eager: row 0 logits equal (1 row vs {R}): {torch.equal(one.view(torch.int16), many.view(torch.int16))}")
    def eq(a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    first = None
    for (i, x1, t1, c1, w1, q1, k1), (_, xR, tR, cR, wR, qR, kR) in zip(rec1, recR):
        same_x = eq(x1, xR)
        same_t = torch.equal(t1, tR) and torch.equal(c1, cR)
        if w.layers[i].dsa.index is not None:
            say(f"   layer {i} (full indexer): head weights {eq(w1, wR)}, index query {eq(q1, qR)}, index key "
                f"{eq(k1, kR)}, selection {same_t}, hidden {same_x}")
        if first is None and not (same_x and same_t):
            first = i
            say(f"   first difference after layer {i}: hidden row equal {same_x}, selection equal {same_t} "
                f"(count {int(c1[0])} vs {int(cR[0])}, tokens differing "
                f"{len(set(t1[0].tolist()) ^ set(tR[0].tolist())) // 2})")
        if i >= 10:
            break
    if first is None:
        say("   every layer's row 0 equal")
    # the captured graphs, as serial_decode and mtp_decode run them
    if e.graphs is not None:
        from tensorfold.families.glm_moe_dsa.cuda.select import sparse_bucket

        outs = {}
        for rows in (window[:1], window):
            st.set_pos(pos)
            g = e.graphs.sparse.get((len(rows), 0, sparse_bucket(pos, len(rows))))
            fwd.stage(w, st, e.buf, rows)
            if g is None:
                say(f"   no graph for {len(rows)} rows")
                continue
            g.replay()
            torch.cuda.synchronize()
            outs[len(rows)] = e.buf.logits[:1].clone()
        if len(outs) == 2:
            a, b = outs.values()
            say(f"   graphs: row 0 logits equal (1 row vs {R}): {torch.equal(a.view(torch.int16), b.view(torch.int16))}"
                f"; graph 1 row == eager 1 row: {torch.equal(a.view(torch.int16), one.view(torch.int16))}")
    eng.comm.barrier()
    say("== done")


if __name__ == "__main__":
    main()
