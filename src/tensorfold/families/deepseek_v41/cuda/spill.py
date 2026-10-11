"""Kept prompts on disk (``TF_DS_SPILL_GIB``): a restart, or a kept prompt pushed out of the window, no longer costs a
conversation its prompt. Its next turn reads the rows back instead of filling them again.

- **What is written.** A kept prompt's rows (the compressed and indexer planes, token ids, host ids) as segments of
  at most SEG_MAX positions, and each kept boundary's snapshot (its slot's rings and compressor inputs, and the
  drafter rings when a snapshot holds them), then a manifest naming them. Rows are append-only by position, so a conversation's next turn
  writes only its new rows: its manifest names the earlier turn's segments up to where it continued. Every rank writes
  the same files to its own disk (the caches are replicated, not sharded).
- **When.** As a finished stream's prompt is kept (``Kept``), for prompts of TF_DS_SPILL_MIN tokens or more. It is
  write-through: nothing waits for an eviction or a clean shutdown, and a crash loses nothing already written.
- **Reading.** Rank 0 matches a new prompt's ids against the manifests' boundaries by prefix hash (no ids held in
  memory) and takes a disk boundary only past what the window's kept prompts offer. Every rank checks its own copy,
  and one small gather agrees, else the prompt fills fresh. The rows stream into a fresh extent segment by segment.
- **Recycling.** Files live in one directory per engine identity (the checkpoint, this package's code, the cache
  formats, the prompt mode); other identities' directories go at start. A manifest unused for TF_DS_SPILL_MAX_AGE_H
  goes; past TF_DS_SPILL_GIB the least recently used go first; then every segment and snapshot no manifest names.
  Half-written files (``.partial``) go too. It runs at start and after every write.
- Restored rows are the bytes the same engine identity wrote when it filled them, so a resumed reply is the reply a
  kept prompt in the window gives.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch

FORMAT = 1
SEG_MAX = 65536                  # positions a segment file at most (~57 MB of rows): bounded host memory either way


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


SPILL_GIB = _env_float("TF_DS_SPILL_GIB", 0.0)
SPILL_DIR = os.environ.get("TF_DS_SPILL_DIR") or str(Path.home() / ".cache" / "tensorfold" / "dsv41-spill")
SPILL_MAX_AGE_H = _env_float("TF_DS_SPILL_MAX_AGE_H", 168.0)
SPILL_MIN = int(_env_float("TF_DS_SPILL_MIN", 4096))
# the env settings that change what a cached row holds (TF_DS_SPILL_* and logging knobs do not)
IDENTITY_ENV = ("TF_DS_KV", "TF_DS_REPLAY", "TF_DS_REPLAY_FLOOR", "TF_DS_PREFILL_CHUNK", "TF_DS_KERNELS",
                "TF_DS_ENGRAM", "TF_DS_VISION_EXTRA")


_NAME = re.compile(r"[0-9A-Za-z_-]{1,64}")
_FILE = re.compile(r"(seg|snap)-[0-9A-Za-z_-]{1,64}-\d{1,9}\.st")


def _parts(x) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def _valid(m: dict) -> bool:
    """A manifest names only files of this directory's own shapes (a tampered one could name a path elsewhere)."""

    files = [seg.get("file", "") for seg in m.get("segments", [])] + [r.get("file", "") for r in m["snaps"].values()]
    return bool(_NAME.fullmatch(str(m.get("name", "")))) and all(_FILE.fullmatch(str(f)) for f in files)


def engine_id(model_dir, *, world: int, pool, dpool=None) -> str:
    """16 hex digits that change with anything that changes a cached row's bits: the checkpoint, this family's code,
    the tensorfold version, the cache planes' formats, the prompt mode and the ranks."""

    h = hashlib.sha256(f"format {FORMAT} world {int(world)}".encode())
    try:
        import tensorfold

        h.update(str(getattr(tensorfold, "__version__", "?")).encode())
    except Exception:                                    # noqa: BLE001
        pass
    here = Path(__file__).resolve().parent
    for p in sorted(list(here.glob("*.py")) + list(here.glob("*.cpp")) + list(here.glob("*.cu")) +
                    list(here.parent.glob("*.py"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    md = Path(model_dir)
    if (md / "config.json").is_file():
        h.update((md / "config.json").read_bytes())
    for p in sorted(md.glob("*.safetensors")):
        h.update(f"{p.name} {p.stat().st_size}".encode())
    for kind, planes in (("comp", pool.comp), ("index_k", pool.index_k)):
        for i, x in sorted(planes.items()):
            for j, t in enumerate(_parts(x)):
                h.update(f"{kind}.{i}.{j} {tuple(t.shape[1:])} {t.dtype}".encode())
    h.update(f"tokens {pool.tokens.dtype} ring {pool.ring_size} {len(pool.ring)}".encode())
    for t in pool.ring[:1]:
        h.update(f"{tuple(t.shape[1:])} {t.dtype}".encode())
    for i, pair in sorted(pool.comp_raw.items()):
        for t in pair:
            h.update(f"raw.{i} {tuple(t.shape[1:])} {t.dtype}".encode())
    if dpool is not None:
        for t in dpool.views[0].rings:
            h.update(f"draft {tuple(t.shape)} {t.dtype}".encode())
    h.update(json.dumps({k: os.environ.get(k) for k in IDENTITY_ENV}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _pieces(start: int, top: int) -> list[tuple[int, int]]:
    """[start, top) cut at SEG_MAX multiples."""

    out, a = [], int(start)
    while a < top:
        b = min(int(top), (a // SEG_MAX + 1) * SEG_MAX)
        out.append((a, b))
        a = b
    return out


class Spill:
    """One rank's kept prompts on its disk (module docstring). ``pool`` is the window (``Model.new_pool``), ``ratios``
    each kv-source layer's compress ratio."""

    def __init__(self, root, *, engine: str, rank: int, pool, ratios: dict, budget: int, max_age: float,
                 min_top: int = SPILL_MIN, log=None, device="cuda") -> None:
        self.root, self.engine, self.rank, self.device = Path(root), str(engine), int(rank), device
        self.dir = self.root / f"v{FORMAT}-{self.engine}"
        self.pool, self.ratios = pool, {int(i): int(r) for i, r in ratios.items()}
        self.budget, self.max_age, self.min_top = int(budget), float(max_age), int(min_top)
        self.log = log or (lambda msg: print(f"[tensorfold] {msg}", flush=True))
        self.manifests: dict[str, dict] = {}
        self.index: dict[str, list] = {}                 # rank 0: prefix hash -> [(name, boundary, full)]
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in (self.root, self.dir):                 # the files hold conversations (their ids): this user's only
            try:
                os.chmod(p, 0o700)
            except OSError:
                pass
        for p in self.root.iterdir():                   # other engine identities' files can never be read again
            if p.is_dir() and p != self.dir and p.name.startswith("v"):
                shutil.rmtree(p, ignore_errors=True)
        for p in self.dir.glob("*.partial"):
            p.unlink(missing_ok=True)
        for p in self.dir.glob("entry-*.json"):
            try:
                m = json.loads(p.read_text())
                if m.get("format") != FORMAT or m.get("engine") != self.engine:
                    raise ValueError("another format or engine")
                if not _valid(m) or p.name != f"entry-{m['name']}.json":
                    raise ValueError("a name or file outside this directory's shapes")
                self.manifests[m["name"]] = m
            except Exception:                            # noqa: BLE001  (unreadable: the recycler takes its files)
                p.unlink(missing_ok=True)
        self._reindex()

    # -- files ----------------------------------------------------------------------------------------------------
    def _path(self, fname: str) -> Path:
        return self.dir / fname

    def _entry(self, name: str) -> Path:
        return self._path(f"entry-{name}.json")

    def _mtime(self, name: str) -> float:
        try:
            return self._entry(name).stat().st_mtime
        except OSError:
            return 0.0

    def _save(self, fname: str, tensors: dict, meta: dict) -> None:
        from safetensors.torch import save_file

        path = self._path(fname)
        tmp = path.with_name(path.name + ".partial")
        save_file(tensors, str(tmp), metadata={k: str(v) for k, v in meta.items()})
        os.replace(tmp, path)

    def _write_json(self, path: Path, obj: dict) -> None:
        tmp = path.with_name(path.name + ".partial")
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, path)

    # -- writing ----------------------------------------------------------------------------------------------------
    def _covered_by(self, lineage: dict | None) -> tuple[list, int]:
        """The inherited segments and where they end, when every file they name is still here; else none."""

        if not lineage:
            return [], 0
        segs, end = lineage.get("segments") or [], 0
        for seg in segs:
            if seg["from"] != end or not self._path(seg["file"]).is_file():
                return [], 0
            end = seg["to"]
        return (list(segs), end) if end == lineage.get("cut", end) else ([], 0)

    def _write_segment(self, fname: str, base: int, p0: int, p1: int, host) -> None:
        tensors = {}
        for kind, planes in (("comp", self.pool.comp), ("index_k", self.pool.index_k)):
            for i, x in planes.items():
                r = self.ratios[int(i)]
                assert base % r == 0, (base, r)
                a, b = (base + p0) // r, (base + p1) // r
                for j, t in enumerate(_parts(x)):
                    tensors[f"{kind}.{i}.{j}"] = t[a:b].to("cpu", copy=True).contiguous()
        tensors["tokens"] = self.pool.tokens[base + p0:base + p1].to("cpu", copy=True).contiguous()
        tensors["host"] = torch.from_numpy(np.array(host[p0:p1], dtype=np.int32))
        self._save(fname, tensors, {"p0": p0, "p1": p1, "engine": self.engine})

    def _write_snap(self, fname: str, snap) -> None:
        tensors = {"ring": snap[0].to("cpu", copy=True).contiguous()}
        if snap[1] is not None:
            tensors["raw"] = snap[1].to("cpu", copy=True).contiguous()
        if len(snap) > 2 and snap[2] is not None:
            tensors["draft"] = snap[2].to("cpu", copy=True).contiguous()
        self._save(fname, tensors, {"n": len(snap), "engine": self.engine})

    @staticmethod
    def _hashes(keys, boundaries) -> dict:
        """Each boundary's prefix hash: sha256 of the prompt's ids (int64, image spans keyed by picture) to it."""

        k64 = np.ascontiguousarray(keys, dtype=np.int64)
        h, prev, out = hashlib.sha256(), 0, {}
        for b in sorted(int(x) for x in boundaries):
            if b > len(k64):
                break
            h.update(k64[prev:b].tobytes())
            prev = b
            out[str(b)] = h.copy().hexdigest()
        return out

    def persist(self, name: str, *, base: int, top: int, replay: bool, snaps: dict, host, lineage: dict | None,
                keys=None) -> dict | None:
        """Write a kept prompt at pool positions [base, base + top): the rows past what ``lineage`` (the kept or
        spilled prompt it continued, to where it continued) already holds on this disk, the snapshots not already
        there, then its manifest. Returns the manifest, None when the prompt is too short or a write failed (the
        prompt stays in the window either way)."""

        top = int(top)
        if top < self.min_top or not snaps:
            return None
        try:
            segs, start = self._covered_by(lineage)
            t0, wrote = time.perf_counter(), 0
            for p0, p1 in _pieces(start, top):
                fname = f"seg-{name}-{p0}.st"
                self._write_segment(fname, int(base), p0, p1, host)
                segs.append({"file": fname, "p0": p0, "from": p0, "to": p1})
                wrote += self._path(fname).stat().st_size
            inherited = (lineage or {}).get("snaps") or {}
            refs = {}
            for b, snap in sorted(snaps.items()):
                ref = inherited.get(str(b))
                if ref is not None and int(b) <= start and self._path(ref["file"]).is_file():
                    refs[str(b)] = ref
                    continue
                fname = f"snap-{name}-{b}.st"
                self._write_snap(fname, snap)
                refs[str(b)] = {"file": fname, "full": len(snap) > 2}
                wrote += self._path(fname).stat().st_size
            m = {"format": FORMAT, "engine": self.engine, "name": name, "top": top, "replay": bool(replay),
                 "segments": segs, "snaps": refs, "created": time.time()}
            if keys is not None:
                m["hashes"] = self._hashes(keys, [int(b) for b in refs])
            self._write_json(self._entry(name), m)
            self.manifests[name] = m
            self._index_add(m)
            if self.rank == 0:
                self.log(f"spilled kept prompt {name}: {top} positions, wrote rows {start}..{top} and "
                         f"{wrote / 2**20:.1f} MiB in {time.perf_counter() - t0:.2f}s")
            self._supersede(lineage, m)
            return m
        except Exception as exc:                         # noqa: BLE001  (a full disk must not fail a request)
            self.log(f"rank {self.rank}: could not spill kept prompt {name} ({type(exc).__name__}: {exc})")
            return None

    def _supersede(self, lineage: dict | None, m: dict) -> None:
        """The manifest ``m`` continued goes when ``m`` holds all of it: it continued at that manifest's top, so its
        rows are ``m``'s, and every boundary snapshot of it is ``m``'s too (full ones a newer turn replaces, as a
        covered kept prompt's in the window). Its files stay while ``m`` names them."""

        src = self.manifests.get((lineage or {}).get("src"))
        if src is None or src["name"] == m["name"] or int(lineage["cut"]) < int(src["top"]):
            return
        own = {b for b, ref in src["snaps"].items() if not ref["full"]}
        if own <= set(m["snaps"]):
            self.forget(src["name"])

    def forget(self, name: str) -> None:
        """A manifest a newer one covers (its files stay while another manifest names them)."""

        if self.manifests.pop(name, None) is not None:
            self._entry(name).unlink(missing_ok=True)
            self._reindex()

    # -- reading ----------------------------------------------------------------------------------------------------
    def _reindex(self) -> None:
        self.index = {}
        for m in self.manifests.values():
            self._index_add(m)

    def _index_add(self, m: dict) -> None:
        if self.rank != 0:
            return
        for b, hx in (m.get("hashes") or {}).items():
            ref = m["snaps"].get(b)
            if ref is not None:
                self.index.setdefault(hx, []).append((m["name"], int(b), bool(ref["full"])))

    def best(self, keys, plen: int, *, window: int, replay: bool, above: int = 0) -> tuple[str, int] | None:
        """Rank 0: (manifest, boundary) of the furthest spilled boundary this prompt shares, past ``above``: a
        boundary snapshot below the prompt's last row (with replay prefill, a window below), a full one anywhere below
        its last row; the most recently used of equals."""

        if not self.index:
            return None
        lim_b, lim_f = plen - (window if replay else 1), plen - 1
        cands = sorted({b for refs in self.index.values() for _, b, full in refs
                        if above < b <= (lim_f if full else lim_b)})
        if not cands:
            return None
        k64 = np.ascontiguousarray(keys, dtype=np.int64)
        h, prev, best = hashlib.sha256(), 0, None
        for b in cands:
            if b > len(k64):
                break
            h.update(k64[prev:b].tobytes())
            prev = b
            for name, bb, full in self.index.get(h.copy().hexdigest(), ()):
                m = self.manifests.get(name)
                if bb != b or m is None or bool(m.get("replay")) != bool(replay) or b > (lim_f if full else lim_b):
                    continue
                t = self._mtime(name)
                if best is None or (b, t) > (best[1], best[2]):
                    best = (name, b, t)
        return None if best is None else (best[0], best[1])

    def check(self, name: str, cut: int) -> bool:
        """Whether this rank holds manifest ``name`` with a snapshot at ``cut`` and every row before it."""

        m = self.manifests.get(name)
        if m is None or str(int(cut)) not in m["snaps"]:
            return False
        end = 0
        for seg in m["segments"]:
            if seg["from"] >= cut:
                break
            if seg["from"] != end or not self._path(seg["file"]).is_file():
                return False
            end = seg["to"]
        return end >= cut and all(self._path(ref["file"]).is_file() for b, ref in m["snaps"].items() if int(b) <= cut)

    def _read_snap(self, fname: str, device) -> tuple:
        from safetensors import safe_open

        with safe_open(str(self._path(fname)), framework="pt") as f:  # (copies: its tensors map the file until closed)
            meta, names = f.metadata() or {}, set(f.keys())
            ring = f.get_tensor("ring").to(device, copy=True)
            raw = f.get_tensor("raw").to(device, copy=True) if "raw" in names else None
            if int(meta.get("n", 2)) > 2:
                return (ring, raw, f.get_tensor("draft").to(device, copy=True) if "draft" in names else None)
            return (ring, raw)

    def load(self, name: str, cut: int, base: int, device=None) -> tuple[dict, np.ndarray, dict]:
        """Manifest ``name``'s rows before ``cut`` into pool positions [base, base + cut), segment by segment; returns
        its snapshots to ``cut`` (on ``device``), its host ids to ``cut`` and the lineage a stream continuing it keeps."""

        from safetensors import safe_open

        m, cut, base, device = self.manifests[name], int(cut), int(base), device or self.device
        host = []
        for seg in m["segments"]:
            lo, hi, p0 = seg["from"], min(seg["to"], cut), seg["p0"]
            if lo >= hi:
                continue
            with safe_open(str(self._path(seg["file"])), framework="pt") as f:
                for kind, planes in (("comp", self.pool.comp), ("index_k", self.pool.index_k)):
                    for i, x in planes.items():
                        r = self.ratios[int(i)]
                        a, b, dst = lo // r - p0 // r, hi // r - p0 // r, (base + lo) // r
                        for j, t in enumerate(_parts(x)):
                            t[dst:dst + (b - a)].copy_(f.get_slice(f"{kind}.{i}.{j}")[a:b])
                self.pool.tokens[base + lo:base + hi].copy_(f.get_slice("tokens")[lo - p0:hi - p0])
                host.append(f.get_slice("host")[lo - p0:hi - p0].numpy().copy())    # (not a view of the closed map)
        snaps = {int(b): self._read_snap(ref["file"], device) for b, ref in m["snaps"].items() if int(b) <= cut}
        try:
            os.utime(self._entry(name))                  # recently used (the recycler's order)
        except OSError:
            pass
        return snaps, (np.concatenate(host) if host else np.zeros(0, np.int32)), self.lineage(m, cut)

    @staticmethod
    def lineage(m: dict | None, cut: int) -> dict | None:
        """What a stream continuing manifest ``m`` at ``cut`` inherits: its segments to ``cut``, its snapshots there."""

        if not m:
            return None
        cut = int(cut)
        return {"cut": cut, "src": m["name"],
                "segments": [dict(seg, to=min(seg["to"], cut)) for seg in m["segments"] if seg["from"] < cut],
                "snaps": {b: ref for b, ref in m["snaps"].items() if int(b) <= cut}}

    # -- recycling --------------------------------------------------------------------------------------------------
    def _referenced(self) -> set[str]:
        return {seg["file"] for m in self.manifests.values() for seg in m["segments"]} | \
               {ref["file"] for m in self.manifests.values() for ref in m["snaps"].values()}

    def recycle(self, pinned=()) -> None:
        """Manifests past the age limit go, then the least recently used while the files they name pass the budget
        (``pinned``, the window's kept prompts, last), then every file no manifest names."""

        pinned, now, dropped = set(pinned), time.time(), 0
        for name in [n for n in self.manifests if now - self._mtime(n) > self.max_age]:
            self.manifests.pop(name, None)
            self._entry(name).unlink(missing_ok=True)
            dropped += 1
        sizes = {}
        for p in self.dir.iterdir():
            if p.suffix == ".st":
                try:
                    sizes[p.name] = p.stat().st_size
                except OSError:
                    pass

        def total() -> int:
            return sum(sizes.get(f, 0) for f in self._referenced())

        order = sorted(self.manifests, key=lambda n: (n in pinned, self._mtime(n)))
        while self.manifests and total() > self.budget and order:
            name = order.pop(0)
            self.manifests.pop(name, None)
            self._entry(name).unlink(missing_ok=True)
            dropped += 1
        keep = self._referenced()
        freed = 0
        for f, n in sizes.items():
            if f not in keep:
                self._path(f).unlink(missing_ok=True)
                freed += n
        for p in self.dir.glob("*.partial"):
            p.unlink(missing_ok=True)
        if dropped or freed:
            self._reindex()
            if self.rank == 0:
                self.log(f"spill recycler: {dropped} manifests and {freed / 2**30:.2f} GiB gone, "
                         f"{len(self.manifests)} kept prompts on disk ({total() / 2**30:.2f} GiB)")

    def stats(self) -> dict:
        files = self._referenced()
        size = sum(self._path(f).stat().st_size for f in files if self._path(f).is_file())
        return {"manifests": len(self.manifests), "bytes": size}
