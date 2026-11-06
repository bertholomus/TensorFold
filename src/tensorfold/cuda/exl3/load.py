"""Safetensors checkpoint reading shared between CUDA families: the header-indexed shard reader, the rank cache,
and the EXL3 group readers.

Note D3 §1 lifts these to a family-neutral import point so a new family (``families/mistral_large4``) can reuse
them without importing another family's package. The implementations stay where they are — DeepSeek-V4.1's tree
stays as it is — and this module is the import point D3 prefers ("lift to
cuda/exl3/load.py"; importing from ``deepseek_v41`` is the accepted same-behavior variant). Nothing here changes
that family's behavior; it re-exports it.

``open_rank_cache`` is the one addition: note D3 §5.4 — the EP4 free-space estimate. A DS-V4.1 rank's cache file
holds its TP share of every tensor, so ``RankCache.open`` sizes it as ``total * 11 / (10 * world)``. An ML4 EP4
rank keeps whole experts (only ``1/world`` of them) and every replicated byte, so that estimate is wrong (it
undercounts the head, embedding, attention and indexer bytes and overcounts each expert). The caller passes the
plan from ``ml4_pack.json`` instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

from tensorfold.families.deepseek_v41.cuda.weights import (RankCache, Shards, _codebook, _kept, _trim, exl3_parts,
                                                           linear, pack_trellises)

__all__ = ["RankCache", "Shards", "exl3_parts", "linear", "pack_trellises", "open_rank_cache", "verify_free"]


def open_rank_cache(root: str | None, model_dir: Path, rank: int, world: int, extra: str, loader_file: Path,
                    need_bytes: int | None = None, prefix: str = "rank") -> RankCache | None:
    """``RankCache.open`` with the EP4 sizing (D3 §5.4): the file is keyed by ``loader_file``'s source (the family's
    own loader decides what bytes are kept) and, when ``need_bytes`` is given, sized from the pack plan
    (routed/world + replicated) with the same tenth of headroom ``RankCache.open`` allows, instead of
    ``total * 11 / (10 * world)``. Same file format and footer, so the returned object is an ordinary RankCache."""

    if not root:
        return None
    d = Path(root)
    key = hashlib.sha256()
    key.update(Path(loader_file).read_bytes())
    key.update(f"{rank}/{world}/{extra}".encode())
    for f in sorted(Path(model_dir).glob("*.safetensors")) + [Path(model_dir) / "config.json",
                                                              Path(model_dir) / "ml4_pack.json"]:
        if not f.exists():
            continue
        st = f.stat()
        key.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    pattern = f"{prefix}{rank}of{world}-*"
    path = d / f"{prefix}{rank}of{world}-{key.hexdigest()[:20]}.bin"
    try:
        with open(path, "rb") as f:
            f.seek(-16, os.SEEK_END)
            tail = f.read(16)
            if tail[8:] != RankCache.MAGIC:
                raise ValueError("no footer")
            at = struct.unpack("<Q", tail[:8])[0]
            f.seek(at)
            index = json.loads(f.read()[:-16])
        return RankCache(path, index)
    except (OSError, ValueError):
        pass
    try:
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob(pattern):                        # one file a rank split: older keys go
            old.unlink()
        if need_bytes is None:
            total = sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors"))
            need_bytes = total * 11 // (10 * world) if world else total
        else:
            need_bytes = need_bytes * 11 // 10
        if verify_free(d, need_bytes):
            return RankCache(path, None)
        print(f"[tensorfold] rank {rank}: rank cache off (needs ~{need_bytes / 2**30:.0f} GiB free in {d})",
              flush=True)
        return None
    except OSError:
        return None


def verify_free(d: Path, need_bytes: int) -> bool:
    """Whether ``d`` has ``need_bytes`` plus the 2 GiB floor free (the same floor RankCache.open uses)."""

    return shutil.disk_usage(d).free >= need_bytes + (2 << 30)
