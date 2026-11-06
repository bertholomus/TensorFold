#!/usr/bin/env python3
"""parity_exl3_side.py — exllamav3 side of the step-3 parity (phase 2b4).

Adapted from loader-test/parity_L00_07.py's ``exl3`` phase (ML4-LOADER-1101 acceptance 3) for the
PARTIAL pack: the layer count comes from the pack's config.json (pack-L00-03: 4), the fixture is
the seeded ml4_fixture stream (4 rows x 2048, seed 1105 — replaces calib128x2048.npy rows
124..127), and the output mirrors tf_dump_hidden.py exactly so compare_hidden.py pairs them.

Dump: ``--out`` (default results/exl3_hidden_L00_03.pt), ``h[L+1]`` = residual stream after layer
L, ``h[0]`` = embeddings, ``(1+n_layers, rows, seqlen, hidden)`` fp32, keys ``h/rows/ids/source/pack``.

Run it wherever the loader phase ran its exl3 phase (the ml4-loader branch tree of exllamav3 + its
compiled extension). On a Spark,
mount the tree and keep EXL3_INT8_GEMV=1; tf32 stays OFF and fp32 matmul "highest" as in the
reference (the numeric regime the loader parity was gated in).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # ml4_fixture (sibling file)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/w/pack-L00-03")
    ap.add_argument("--out", default="/w/results/exl3_hidden_L00_03.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1105)
    ap.add_argument("--max-gib", type=float, default=44.0)
    args = ap.parse_args()

    import torch
    from ml4_fixture import fixture_ids
    from exllamav3.modules.transformer import TransformerBlock

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    tot = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gib * 2**30 / tot), dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    from exllamav3 import Config, Model

    ids = fixture_ids(args.pack, rows=args.rows, seqlen=args.seqlen, seed=args.seed).to(dev)
    print(f"[exl3] fixture ids {tuple(ids.shape)} vocab from {args.pack}/config.json", flush=True)
    t0 = time.time()
    config = Config.from_directory(args.pack)
    model = Model.from_config(config)
    model.load(dev)
    print(f"[exl3] loaded {config.num_hidden_layers} layers in {time.time() - t0:.1f} s, "
          f"{torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB", flush=True)
    params = {"attn_mode": "flash_attn_nc"}
    t1 = time.time()
    with torch.inference_mode():
        x = model.prepare_inputs(ids, params)
        outs = []
        for module, instance, idx in model.fwd_modules:
            params["layer_instance"] = instance
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if not outs:
                outs.append(x.float().cpu().clone())                 # embeddings
            if isinstance(module, TransformerBlock):
                outs.append(x.float().cpu().clone())
                if len(outs) == config.num_hidden_layers + 1:
                    break
        torch.cuda.synchronize()
    h = torch.stack(outs)                                            # (1+nL, R, S, H) fp32
    print(f"[exl3] forward {tuple(ids.shape)} in {time.time() - t1:.1f} s, "
          f"finite {bool(torch.isfinite(h).all())}, peak "
          f"{torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB", flush=True)
    assert bool(torch.isfinite(h).all()), "non-finite hidden state"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"h": h.cpu(), "rows": list(range(args.rows)), "ids": ids.cpu(), "source": "exl3",
                "pack": str(args.pack)}, args.out)
    print(f"[exl3] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
