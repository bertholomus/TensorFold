"""Preflight for zrec (no GPU needed): the recorder installed, every engine phase patched, an ATen op logged and dumped."""

import importlib
import json
import os
import sys

import torch

import zrec

bad = []
for mod, owner, attr, _ in zrec.PHASES:
    try:
        m = importlib.import_module(f"{zrec.FAMILY}.{mod}")
        if not getattr(getattr(getattr(m, owner) if owner else m, attr), "_zrec", False):
            bad.append(f"{mod}.{owner}.{attr}")
    except Exception as e:
        bad.append(f"{mod}.{owner}.{attr} ({e!r})")
f = zrec.scoped("check", lambda x: (x + 1).sum())
f(torch.ones(4))
zrec.dump()
ops = json.load(open(os.path.join(os.environ["TF_ZREC_DIR"], "aten.json")))["ops"]
seen = sorted({r["op"] for r in ops if r["phase"] == "check"})
print(json.dumps({"unpatched": bad, "check_ops": seen}))
sys.exit(1 if bad or not seen else 0)
