"""ml4_fixture.py — the step-3 parity fixture (phase 2b4).

Both dump sides (TF: tf_dump_hidden.py; exl3: parity_exl3_side.py) import THIS file, so the token
stream is identical by construction: ids come from a pure-hashlib byte stream keyed by
(seed, index) — no RNG library, so it cannot drift between the TF container's torch build and the
exllamav3 side's environment, and it is byte-stable forever (sha256). The shape is the loader-phase
parity shape — 4 rows x 2048 tokens (parity_L00_07.py used rows x 2048 of calib128x2048.npy; here
the rows are the seeded synthetic stream, seed 1105, the same constant every ML4 test uses). The
vocab comes from the SAME pack ``config.json`` both sides read. compare_hidden.py re-derives the
ids and asserts equality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

SEED = 1105
ROWS = 4
SEQLEN = 2048


def vocab_of(pack_dir: str | Path) -> int:
    """vocab_size from the pack's config.json (top level or text_config) — the one vocab both sides see."""

    cfg = json.loads((Path(pack_dir) / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    return int(t["vocab_size"])


def fixture_ids(pack_dir: str | Path, rows: int = ROWS, seqlen: int = SEQLEN,
                seed: int = SEED) -> torch.Tensor:
    """(rows, seqlen) int64 token ids in [0, vocab), deterministic from (seed, vocab, rows, seqlen)."""

    vocab = vocab_of(pack_dir)
    n = rows * seqlen
    stream = b""
    while len(stream) < n * 8:
        stream += hashlib.sha256(f"ml4-parity:{seed}:{len(stream) // 32}".encode()).digest()
    flat = [int.from_bytes(stream[i:i + 8], "big") % vocab for i in range(0, n * 8, 8)]
    return torch.tensor(flat, dtype=torch.int64).reshape(rows, seqlen)
