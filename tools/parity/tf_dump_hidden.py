#!/usr/bin/env python3
"""tf_dump_hidden.py — TensorFold side of the step-3 parity (phase 2b4).

Loads pack-L00-03 (4 real layers, real dims — world=1) with the m2b2 loader, runs the fixed
parity fixture (tools/parity/ml4_fixture.py: 4 rows x 2048 tokens, seed 1105) and dumps the
residual stream — the embedding rows and the hidden states after every layer — to
``--out`` (default results/tf_hidden_L00_03.pt), layout ``h[L+1]`` = after layer L, ``h[0]`` =
embeddings, shape ``(1+n_layers, rows, seqlen, hidden)`` fp32 — the exact layout of the exl3 side
(parity_exl3_side.py), so compare_hidden.py can pair them elementwise.

Rows run one at a time through Ml4Model.forward (B == 1 is the model's contract); causal attention
makes a row's states batch-independent, so this equals the exl3 batch-4 stream row for row.

Node run (container; EXL3_INT8_GEMV per BRINGUP-PLAN; TORCH_EXTENSIONS_DIR caches the build):

    docker run --rm --name ml4-par-tf --gpus all --ipc=host --network host \
      -e PYTHONPATH=/w/src/tensorfold/src -e EXL3_INT8_GEMV=1 -e TORCH_EXTENSIONS_DIR=/w/torch-ext \
      -v "$ML4ROOT:/w" nvcr.io/nvidia/pytorch:26.07-py3 \
      python3 /w/src/tensorfold/tools/parity/tf_dump_hidden.py \
        --pack /w/pack-L00-03 --out /w/results/tf_hidden_L00_03.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # ml4_fixture (sibling file)


def dump_hidden(model, ids: "torch.Tensor", device: "torch.device"):
    """(1+nL, R, S, H) fp32 CPU: h[0] embedding rows, h[L+1] the residual stream after layer L.

    CPU-testable core (tests/test_ml4_parity.py drives it with the tiny fixture model)."""

    import torch
    from tensorfold.families.mistral_large4.cuda import model as M

    w = model.w
    embeds: list = []
    layers: list = [[] for _ in range(w.cfg.n_layers)]
    with torch.inference_mode():
        for r in range(ids.shape[0]):
            embeds.append(w.embed[ids[r:r + 1]].float().cpu())
            collect: list = []
            model.forward(ids[r:r + 1].to(device), M.new_cache(w.cfg), collect=collect)
            assert len(collect) == w.cfg.n_layers, f"collect {len(collect)} != {w.cfg.n_layers} layers"
            for L, st in enumerate(collect):
                layers[L].append(st.float().cpu())
    return torch.stack([torch.cat(embeds, 0)] + [torch.cat(ll, 0) for ll in layers], 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/w/pack-L00-03")
    ap.add_argument("--out", default="/w/results/tf_hidden_L00_03.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1105)
    args = ap.parse_args()

    import torch
    from ml4_fixture import fixture_ids
    from tensorfold.families.mistral_large4.cuda import weights as W
    from tensorfold.families.mistral_large4.cuda import model as M

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    ids = fixture_ids(args.pack, rows=args.rows, seqlen=args.seqlen, seed=args.seed)
    print(f"[tf-dump] fixture ids {tuple(ids.shape)} vocab from {args.pack}/config.json", flush=True)
    t0 = time.time()
    w = W.load(args.pack, rank=0, world=1)
    model = M.Ml4Model(w)
    print(f"[tf-dump] loaded {w.cfg.n_layers} layers in {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    h = dump_hidden(model, ids, dev)
    ok = bool(torch.isfinite(h).all())
    print(f"[tf-dump] forward {tuple(ids.shape)} in {time.time() - t0:.1f} s -> {tuple(h.shape)} "
          f"finite {ok}", flush=True)
    assert ok, "non-finite hidden state"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"h": h, "rows": list(range(args.rows)), "ids": ids, "source": "tf",
                "pack": str(args.pack)}, args.out)
    print(f"[tf-dump] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
