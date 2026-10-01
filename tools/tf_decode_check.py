"""Decode consistency (TP1): prompt prefill then serial decode steps (CUDA graphs) vs one full prefill of the same rows.

usage: python3 tf_decode_check.py MODEL_DIR
"""
import os, sys
import torch

os.environ.setdefault("TF_TP_WORLD", "1")
MODEL = sys.argv[1]

from tensorfold.families.glm_moe_dsa.cuda import forward as fwd
from tensorfold.families.glm_moe_dsa.cuda.decode import Engine
from tensorfold.families.glm_moe_dsa.cuda.weights import load

w = load(MODEL, rank=0)
w.comm = None
ids = [785, 3974, 13867, 38627, 34041, 916, 279, 15666, 5562, 13, 758, 220, 16, 24, 21, 24, 11, 32864,
       64332, 6116, 279, 1156, 1697, 311, 4227, 389]
K = 8                                     # decode the last K rows one at a time
for graphs in (False, True):
    e = Engine(w, capacity=4096, max_rows=8, prefill_rows=256, graphs=graphs, graph_rows=(1, 2, 3, 4))
    with torch.no_grad():
        # reference: every row in one prefill chunk, logits for all rows through the decode head
        e.reset()
        R = fwd.stage(w, e.st, e.pbuf, ids)
        fwd.compute(w, e.st, e.pbuf, R, logits=False, nch=fwd.chunks_for(e.st, R), host_pos=e.st.pos)
        from tensorfold.families.glm_moe_dsa.cuda import glue

        glue.rmsnorm(e.pbuf.x[:R], w.norm, w.cfg.eps, e.pbuf.fnormed[:R], e.pbuf.fxs[:R])
        full = torch.stack([fwd.mm(e.buf, e.pbuf.fnormed[r:r + 1], w.head, e.pbuf.fxs[r:r + 1],
                                   e.buf.logits[:1]).float().clone()[0] for r in range(R)])
        # prefill the first len - K rows, then decode
        e.reset()
        R0 = fwd.stage(w, e.st, e.pbuf, ids[:-K])
        fwd.compute(w, e.st, e.pbuf, R0, nch=fwd.chunks_for(e.st, R0), host_pos=e.st.pos)
        fwd.commit(w, e.st, e.pbuf, R0, R0)
        errs, agree = [], 0
        for j in range(K):
            lg = e.forward([ids[len(ids) - K + j - 1 + 1 - 1 + 1 - 1]] if False else [ids[R0 + j]]).float()[0]
            fwd.commit(w, e.st, e.buf, 1, 1)
            ref = full[R0 + j]
            errs.append(((lg - ref).norm() / ref.norm()).item())
            agree += int(lg.argmax() == ref.argmax())
        print(f"graphs={graphs} replays={e.replays} decode-vs-prefill rel_err max {max(errs):.5f} "
              f"mean {sum(errs) / K:.5f} argmax agree {agree}/{K}", flush=True)
