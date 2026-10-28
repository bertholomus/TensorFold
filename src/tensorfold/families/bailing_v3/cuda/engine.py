"""Ling-3.1-flash on TF_TP_WORLD NCCL ranks (2 or 4); all sample by one keyed rule from the same gathered candidates,
so no broadcast. MTP-only drafting: the checkpoint's own layer-72 head (no DFlash2, no drafter model, no HC taps), so
auto resolves to adaptive MTP depth and every policy kind drafts with it or decodes serially."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_POLICY = "auto"
MAX_ROWS = 8                          # the widest verify window (a pending token and up to 7 drafts)
DENSE_CAPACITY = 2560                 # cache slots; Ling's MLA is dense at every length


def encode_policy(spec: str) -> list[int]:
    """Encode policy kind, maximum drafts, and two parameters in millionths as four integers.

    Kinds: 0 serial, 1 fixed depth N, 2 adaptive between a low and a high acceptance, 3 fixed depth with a
    cumulative-confidence floor, 4 auto (adaptive depth with exploration rounds)."""

    spec = str(spec).strip()
    bad = ValueError(f"draft policy {spec!r}: expected auto[:E:EVERY:MARGIN], 0, N, a[:LOW:HIGH], or cN:P")
    try:
        if spec == "auto" or spec.startswith("auto:"):
            parts = spec.split(":")
            if len(parts) not in (1, 4):
                raise bad
            explore, every, margin = (int(parts[1]), int(parts[2]), float(parts[3])) if len(parts) == 4 else (2, 8, 0.03)
            if explore < 1 or every < 0 or not 0 <= margin < 1:
                raise bad
            return [4, explore, every, int(round(margin * 1e6))]
        if spec.startswith("a"):
            parts = spec.split(":")
            if parts[0] != "a" or len(parts) not in (1, 3):
                raise bad
            low, high = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.8, 0.9)
            return [2, 3, int(round(low * 1e6)), int(round(high * 1e6))]
        if spec.startswith("c"):
            most_text, conf = spec[1:].split(":")
            most = int(most_text)
            if not 0 < most < MAX_ROWS:
                raise bad
            return [3, most, int(round(float(conf) * 1e6)), 0]
        most = int(spec)
    except ValueError:
        raise bad from None
    if not 0 <= most < MAX_ROWS:
        raise bad
    return [1 if most > 0 else 0, most, 0, 0]


def decode_policy(code: list[int]):
    """Decode a serial or MTP policy into a DepthPolicy (None: serial)."""

    from .decode import DepthPolicy

    kind, most, a, b = code
    if kind == 4:
        return ("auto", most, a, b / 1e6, False)
    if kind == 2:
        return DepthPolicy(min(most, MAX_ROWS - 1), low=a / 1e6, high=b / 1e6)
    if kind == 3:
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=a / 1e6)
    return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True) if kind == 1 else None


def _f64_ints(x: float) -> list[int]:
    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _ints_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


def _think(budget: int, think_end: int):
    """The run's ThinkStop from the header's two fields (budget 0: none)."""

    from tensorfold.engine.call_gate import ThinkStop

    return ThinkStop(budget, think_end) if budget > 0 else None


MTP_DEFAULT = "1"                     # TF_LING_MTP when unset: 1 (Ling has no other drafter)


def mtp_head(serial_only: bool, layers: int, value: str | None = None) -> bool:
    """TF_LING_MTP: 1 (the default) loads the MTP head, 0 never; without it every round decodes one token."""

    value = os.environ.get("TF_LING_MTP", "") if value is None else value
    value = value.strip().lower() or MTP_DEFAULT
    if value not in ("0", "1"):
        raise ValueError(f"TF_LING_MTP: 0 or 1, not {value!r}")
    return value == "1" and bool(layers)


def without_mtp(transform, layers: int):
    """A startup weight transform that leaves out the MTP layer's tensors (``layers.<num_hidden_layers>.``)."""

    prefix = f"model.layers.{layers}."
    return lambda name, info: (0, 0) if name.startswith(prefix) else transform(name, info)


def prompt_chunk_rows(value: str | None = None) -> int:
    """A prompt chunk's rows: TF_LING_PREFILL_ROWS (a multiple of 128 from 128 to 16,384), else geometry's 2,048.

    Fewer rows hold a smaller prompt Buffers and EXL3 prompt scratch (512: 0.84 GiB less a TP2 rank at the full
    window, memory.parts); a long prompt then takes more chunks, each reading the routed experts again. A prompt
    row's bits depend on its chunk's rows (kernels picked by the row count: the EXL3 prompt GEMM past 128 rows, the
    routed experts' prompt kernels, cuBLAS by M; glm_moe_dsa/cuda/invariant.py), so another value is a prompt path
    the quality gate has to pass on its own, not 2,048's bits (test_bailing_10_engine's chunk test)."""

    from tensorfold.cuda.geometry import PREFILL_ROWS

    value = os.environ.get("TF_LING_PREFILL_ROWS", "") if value is None else value
    if not value.strip():
        return PREFILL_ROWS
    rows = int(value) if value.strip().isdigit() else 0
    if not (128 <= rows <= 16384 and rows % 128 == 0):
        raise ValueError(f"TF_LING_PREFILL_ROWS: a multiple of 128 from 128 to 16,384, not {value!r}")
    return rows


def memory_line(torch, rank: int, what: str, meminfo: str = "/proc/meminfo") -> str:
    """One rank's memory after ``what``: torch's allocated and reserved bytes (the small pool's apart) and the node's
    MemAvailable, the only reading of a GB10's GPU memory a process has (its RSS leaves CUDA's out)."""

    s = torch.cuda.memory_stats()
    gib = float(1 << 30)
    line = (f"[tensorfold] rank {rank} {what}: torch allocated {s.get('allocated_bytes.all.current', 0) / gib:.2f} "
            f"GiB, reserved {s.get('reserved_bytes.all.current', 0) / gib:.2f} (small pool "
            f"{s.get('reserved_bytes.small_pool.current', 0) / gib:.2f})")
    try:
        with open(meminfo) as f:
            for row in f:
                if row.startswith("MemAvailable:"):
                    return line + f", MemAvailable {int(row.split()[1]) / float(1 << 20):.2f} GiB"
    except OSError:
        pass
    return line


def requested_window(context: int | None, explicit: bool) -> int | None:
    """The window admission is asked for: an explicit --context, else none (the native window, capped by what fits).

    Ling's MLA is dense at every length, so its dense limit (Config.dense_limit, 1 << 62) is no window to ask for;
    in the startup status row it overflowed the ranks' int32 all-gather past one rank.
    """

    return context if explicit else None


# Python, CUDA and the C++ host sources the extensions build from (exl3/linear.cpp, experts.cpp, kda.cpp, ...)
FINGERPRINT_SUFFIXES = (".py", ".cu", ".cuh", ".h", ".cpp", ".hpp", ".cc", ".c")


def source_fingerprint(root: Path | None = None) -> int:
    """This tree's TensorFold source as 28 bits (fits the ranks' int32 all-gather): sha256 over the package's
    Python, CUDA and C++ sources in relative-path order, __pycache__ left out (extensions build elsewhere)."""

    root = Path(__file__).resolve().parents[3] if root is None else root   # .../src/tensorfold, the running tree
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in FINGERPRINT_SUFFIXES
                       and "__pycache__" not in p.parts):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return int.from_bytes(digest.digest()[:4], "big") >> 4


class BailingEngine:
    """Ling-3.1-flash on TF_TP_WORLD ranks (this one ``rank``): weights, MTP drafting, per-request policies."""

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, policy: str = DEFAULT_POLICY,
                 context: int | None = None, context_explicit: bool | None = None, serial_only: bool = False, comm=None,
                 prefill_rows: int | None = None) -> None:
        """``comm``: a communicator with ``all_gather`` and ``barrier`` instead of NCCL between two machines (tests)."""

        import torch

        from tensorfold.cuda.comm import open_comm
        from .decode import Engine
        from .weights import Config, load
        from tensorfold.families.glm5_next.cuda.split import rule, unit
        from tensorfold.cuda.capacity import admit
        from tensorfold.cuda.geometry import split_weights
        from tensorfold.families.bailing_v3 import memory

        encode_policy(policy)                           # a bad default fails here, not in the first request
        torch.cuda.set_device(0)
        self.torch = torch
        self.rank = rank
        self.policy = "0" if serial_only else policy
        self.serial_only = serial_only
        world = int(os.environ.get("TF_TP_WORLD", "2"))
        self.world = world
        if comm is None:
            # decode.Engine settles every graph replay before its next uncaptured collective (Engine.settle), so NCCL
            # can skip its graph-mixing support, as glm_moe_dsa's engine does: an in-graph all-gather of a [1, 4096]
            # fp32 partial between two GB10s takes 30.9 us instead of 52.8 (eager 18.7)
            os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
        self.comm = comm if comm is not None else open_comm(rank, world, master, port)
        self.comm.barrier()
        self._check_source(source_fingerprint())    # a stale tree on one rank fails here, before admission
        cfg = Config.read(model_dir)
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        self.mtp_on = mtp_head(serial_only, cfg.mtp_layers)
        weights_estimate = split_weights(rule, self.world, unit=unit)
        if not self.mtp_on:
            weights_estimate = without_mtp(weights_estimate, cfg.layers)
        prefill_rows = prompt_chunk_rows() if prefill_rows is None else int(prefill_rows)
        # what this engine allocates (memory.py), not GLM's mla_geometry: Ling has 9 latent layers and the MTP
        # layer's, a rope key beside each latent, no indexer, and partials that grow with the window
        self.capacity_plan = admit(model_dir, requested_window(context, explicit), explicit, torch,
                                   lambda text: memory.geometry(text, self.world, rank, mtp=self.mtp_on, rows=MAX_ROWS,
                                                                prefill_rows=prefill_rows,
                                                                minimum_slots=DENSE_CAPACITY),
                                   weights_estimate, rank=rank, world=self.world, gather=self._gather_ints)
        self.limit = self.capacity_plan["context_window"]
        capacity = self.capacity_plan["cache_slots"]
        # both ranks must run the same calls: refuse to start when they were given different settings
        mine = [capacity, int(serial_only), prefill_rows, int(self.mtp_on)]
        # other conversations' kept prompts get what the window leaves, at most TF_LING_CACHE_GIB, the same on both ranks
        plan = self.capacity_plan
        wanted = int(float(os.environ.get("TF_LING_CACHE_GIB", "3")) * 2 ** 30)
        spare = max(0, min(wanted, plan["budget_bytes"] - plan["total_bytes_estimate"]))
        both = self._gather_ints(mine + [spare >> 20])
        if any(b[:-1] != both[0][:-1] for b in both):
            raise RuntimeError("the ranks were started with different settings (context, drafts, TF_LING_MTP): "
                               f"rank 0 {both[0][:-1]} vs {[b[:-1] for b in both[1:]]}")
        self.cache_bytes = min(b[-1] for b in both) << 20
        plan["kept_bytes"] = self.cache_bytes
        for key in ("serving_peak_bytes_estimate", "total_bytes_estimate"):
            plan[key] = plan[key] + self.cache_bytes
        if rank == 0 and self.cache_bytes < wanted:
            print(f"[tensorfold] other conversations' prompts are kept in {self.cache_bytes / 2 ** 30:.1f} GiB, what "
                  f"the {self.limit}-token window leaves (TF_LING_CACHE_GIB asks {wanted / 2 ** 30:.1f})", flush=True)
        if not self.mtp_on and not serial_only:
            print("[tensorfold] no MTP head loaded (TF_LING_MTP=0 or the checkpoint has none): every round decodes "
                  "one token", flush=True)
        w = load(model_dir, rank=rank)
        print(memory_line(torch, rank, "loaded"), flush=True)
        w.comm = self.comm
        self.comm.ready("loading")                   # a peer stuck loading is named, not waited on in NCCL
        self.comm.barrier()
        self.w = w
        self.e = Engine(w, capacity=capacity, max_rows=MAX_ROWS, prefill_rows=prefill_rows, graphs=True,
                        graph_rows=tuple(range(1, MAX_ROWS + 1)))
        print(memory_line(torch, rank, "built (buffers, states, graphs)"), flush=True)
        self.costs = self._calibrate()
        print(memory_line(torch, rank, "calibrated"), flush=True)
        if rank == 0:
            c = self.costs
            mtp = (f"; MTP draft {c['mtp']:.2f} (+{c['mtp_step']:.2f} a chained draft, +{c['mtp_row']:.2f} a row)"
                   if w.mtp is not None else "")
            print("[tensorfold] verify costs (ms): " + " ".join(f"{v:.1f}" for v in c["verify"]) + mtp, flush=True)
        self.eos = tuple(w.cfg.eos)
        self.model_dir = Path(model_dir)
        self.request = threading.local()    # the calling request's policy and stop-at-EOS (the app sets these)
        # kept conversations (decode.Snapshot, least recently used first) and the live caches' ids; states and saved rows stay within cache_bytes
        self.cache: list = []
        self.live: list[int] = []
        self.cache_entries = int(os.environ.get("TF_LING_CACHE_ENTRIES", "8"))

    def _calibrate(self) -> dict:
        """Per-piece ms: fastest of interleaved passes, equal on every rank."""

        import statistics

        import numpy as np

        from .decode import draft, prefill

        torch = self.torch
        e, st = self.e, self.e.st
        rng = np.random.default_rng(0)
        vocab = self.w.cfg.vocab

        def tokens(n: int) -> list[int]:
            return [int(t) for t in rng.integers(0, vocab, n)]

        prefill(e, tokens(64), None, mtp=True)
        hidden = e.pbuf.fnormed[:MAX_ROWS].clone()          # rows for timing the draft steps
        one, six = tokens(1), tokens(6)
        start = st.mtp_len

        def rewind() -> None:
            st.set_mtp_len(start)
            st.mtp_drafted = 0

        pieces: dict[str, tuple] = {f"v{r}": (lambda w=tokens(r): e.forward(w), None) for r in range(1, MAX_ROWS + 1)}
        if self.w.mtp is not None:
            pieces["m1"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 1, None), rewind)
            pieces["m3"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 3, None), rewind)
            pieces["m6"] = (lambda: draft(e, hidden[:6], six, st.pos + 1, 1, None), rewind)
        best = {name: float("inf") for name in pieces}
        for turn in range(9):
            for name, (fn, prep) in pieces.items():
                if prep is not None:
                    prep()
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                if turn >= 2:
                    best[name] = min(best[name], (time.perf_counter() - t) * 1e3)
            rewind()
        names = list(best)
        both = dict(zip(names, self._slowest([best[n] for n in names])))
        e.reset()
        rows = list(range(2, MAX_ROWS + 1))
        ys = [both[f"v{r}"] for r in rows]
        slope = statistics.median((ys[j] - ys[i]) / (rows[j] - rows[i]) for i in range(len(rows))
                                  for j in range(i + 1, len(rows)))
        base = statistics.median(y - slope * r for r, y in zip(rows, ys))
        verify = [both["v1"]] + [base + slope * r for r in rows]
        mtp = both.get("m1", 0.0)
        return {"verify": verify, "mtp": mtp, "mtp_step": max((both.get("m3", 0.0) - mtp) / 2, 0.0),
                "mtp_row": max((both.get("m6", 0.0) - mtp) / 5, 0.0),
                "timed": {k: round(v, 2) for k, v in both.items()}}

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        """Every rank's int list, in rank order."""

        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int32, device="cuda")
        got = torch.empty((self.world * len(values),), dtype=torch.int32, device="cuda")
        self.comm.all_gather(mine, got)
        return got.view(self.world, len(values)).tolist()

    def _check_source(self, fingerprint: int) -> None:
        """Every rank's fingerprint, in rank order; a mismatch raises before admission: a rank on a stale tree
        hangs instead of failing (K3's attempt 1 waited on a doorbell key the other commit never sets)."""

        seen = self._gather_ints([fingerprint])
        if any(row != seen[0] for row in seen):
            raise RuntimeError("the ranks run different TensorFold sources (fingerprint "
                               + " ".join(f"rank {rank} {row[0]:07x}" for rank, row in enumerate(seen))
                               + "): stage the same commit on every node")

    def _slowest(self, values: list[float]) -> list[float]:
        """Each value's largest over the ranks (the calibration's ms, equal on every rank)."""

        return [max(column) for column in zip(*self._gather_floats(values))]

    # the idle doorbell: ranks 1.. wait for each request on the rendezvous store (no store: no doorbell), not in NCCL.
    # One key a follower, deleted only by that follower: a shared key, deleted by the first follower to wake, can be
    # gone before a slower one (still finishing the last request) waits for it, and the request then hangs.
    def _store(self):
        return getattr(self.comm, "store", None)

    def _ring(self) -> None:
        store = self._store()
        if store is not None:
            self._bell = getattr(self, "_bell", 0) + 1
            for r in range(1, self.world):
                store.set(f"tf_ling_request_{self._bell}/{r}", b"1")

    def _await_bell(self) -> None:
        store = self._store()
        if store is None:
            return
        from datetime import timedelta

        key = f"tf_ling_request_{getattr(self, '_bell', 0) + 1}/{self.rank}"
        while True:
            try:
                store.wait([key], timedelta(hours=1))
                break
            except Exception as e:            # an idle hour: wait again (a lost rank 0 is a connection error instead)
                if "timeout" not in str(e).lower():
                    raise
        store.delete_key(key)
        self._bell = getattr(self, "_bell", 0) + 1

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (a length, then the values, through the all-gather)."""

        torch = self.torch
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32, device="cuda")
        got = torch.empty((self.world,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = (torch.tensor(values, dtype=torch.int32, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int32, device="cuda"))
        allv = torch.empty((self.world * count,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    def _resume(self, prompt: list[int], code: list[int]):
        """The longest snapshot of a strict prefix of ``prompt`` whose MTP caches fit the request's policy."""

        best = None
        for snap in self.cache:
            if len(snap.ids) < len(prompt) and prompt[:len(snap.ids)] == snap.ids and (
                    best is None or len(snap.ids) > len(best.ids)):
                best = snap
        return best

    def _drop(self, snap) -> None:
        """Forget a kept snapshot and free its saved rows now, even while a caller still holds the object."""
        snap.rows, snap.nbytes = None, 0
        self.cache.remove(snap)

    def _remember(self, snap, behind: bool = False) -> None:
        """Keep ``snap`` as the newest; ``behind``: as the first to go, for a gate's continuation (its prompt holds
        part of the reply), so the snapshot a later prompt resumes from stays (TF_LING_CACHE_GIB=0 keeps one)."""
        for c in [c for c in self.cache if c.ids == snap.ids and c is not snap]:
            self._drop(c)
        rest = [c for c in self.cache if c is not snap]
        self.cache[:] = [snap, *rest] if behind else [*rest, snap]   # a resumed prompt kept again moves last
        dropped = False
        while len(self.cache) > 1 and (len(self.cache) > self.cache_entries or self._held_bytes() > self.cache_bytes):
            self._drop(self.cache[0])
            dropped = True
        if dropped:
            import torch

            torch.cuda.empty_cache()

    def _take_over(self, keep: list[int]) -> None:
        """Save the rows of every kept snapshot the next prefill overwrites, dropping the oldest entries past the memory budget; both ranks decide alike."""
        from .decode import row_bytes, save_rows

        live = self.live
        dropped = False

        def resumes(c) -> bool:
            return len(c.ids) <= len(keep) and keep[:len(c.ids)] == c.ids

        for snap in list(self.cache):
            n = len(snap.ids)
            if snap not in self.cache or snap.rows is not None or resumes(snap):
                continue
            if live[:n] != snap.ids:                  # its rows are already gone: nothing to resume from
                self._drop(snap)
                continue
            need = row_bytes(self.e, snap)
            while self._held_bytes() + need > self.cache_bytes:
                old = next((c for c in self.cache if c is not snap and not resumes(c)), None)
                if old is None:
                    break
                self._drop(old)
                dropped = True
            if self._held_bytes() + need > self.cache_bytes:
                self._drop(snap)
                dropped = True
                continue
            save_rows(self.e, snap)
        if dropped:
            import torch

            torch.cuda.empty_cache()             # give the freed rows back rather than keep them in torch's pool

    def _held_bytes(self) -> int:
        from .decode import snapshot_bytes

        return sum(snapshot_bytes(c) for c in self.cache)

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable[[list[int]], Any],
             code: list[int], hit, draft: bool, constraint=None, think=None, continued: bool = False) -> dict[str, Any]:
        self.e.constraint, self.e.window = constraint, None       # both ranks walk and mask the same rows
        try:
            return self._run_once(prompt, max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, think, continued)
        finally:
            self.e.constraint = self.e.window = None

    def _run_once(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool,
                  on_tokens: Callable[[list[int]], Any], code: list[int], hit, draft: bool, think=None,
                  continued: bool = False) -> dict[str, Any]:
        from .decode import mtp_decode, prefill, serial_decode

        auto = code[0] == 4
        t0 = time.perf_counter()
        # a request writes the caches from its resume point: other conversations' rows are saved first, a saved resume point's restored
        from .decode import load_rows

        cut = len(hit.ids) if hit is not None else 0
        self._take_over(list(hit.ids) if hit is not None else [])
        if hit is not None and hit.rows is not None:
            load_rows(self.e, hit)
            hit.rows, hit.nbytes = None, 0            # live again
        self.live = list(prompt)
        use_mtp = draft and self.w.mtp is not None
        first = prefill(self.e, prompt, sampling, mtp=use_mtp, resume=hit,
                        keep_at=max(1, len(prompt) - 1) if draft else None,
                        keep=(lambda snap: self._remember(snap, behind=True)) if continued else self._remember)
        prefill_s = time.perf_counter() - t0
        stats: dict[str, Any] = {"prefill_s": prefill_s, "cached": cut}
        on_tokens([first])
        if max_tokens <= 1 or (stop_eos and first in self.eos) or (think is not None and think.hit([first])):
            return stats
        policy = decode_policy(code) if draft else None
        if policy is None:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens, think=think)
        else:
            if auto:                     # auto: adaptive MTP depth, greedy replies keep a confidence floor
                from .decode import DepthPolicy

                greedy = sampling is None or sampling.temperature <= 0
                policy = DepthPolicy(3, fixed=True, confidence=0.35) if greedy else DepthPolicy(3, low=0.6, high=0.85)
            res = mtp_decode(self.e, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                             on_tokens=on_tokens, think=think)
        # the caches now hold prompt and reply; only prompts are snapshotted, since a later prompt prefills the reply again
        self.live = list(prompt) + res.tokens[:self.e.st.pos - len(prompt)]
        stats.update(decode_s=res.seconds, rounds=res.rounds, min_rows=1 + min(res.depths, default=0),
                     tokens_per_round=round((len(res.tokens) - 1) / max(res.rounds, 1), 3),
                     sha256=hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        if policy is not None:                   # the drafts' counts, which /health, /metrics and the reply report
            stats.update(drafted=res.drafted, accepted=res.accepted)
        if res.stages:
            stats["stages_ms"] = {k: round(v * 1e3, 1) for k, v in res.stages.items()}
        return stats

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 constraint=None, think_stop: tuple[int, int] | None = None, continued: bool = False) -> dict[str, Any]:
        """Mirror one rank-0 request on every other rank; draft=False uses serial decoding and fresh prefill as the reference drafted replies must equal.
        ``think_stop`` (budget, think_end): every rank ends the run where the server's thinking budget cuts it;
        ``continued``: the prompt holds part of the reply (a gate's continuation), so its snapshot goes behind."""

        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        spec = "0" if not draft or self.serial_only else (getattr(self.request, "policy", None) or self.policy)
        code = encode_policy(spec)
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        hit = self._resume(list(prompt), code) if draft else None
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        header = [max_tokens, int(stop_eos), int(draft), len(hit.ids) if hit is not None else 0,
                  seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0), *_f64_ints(sampling.min_p if sampling else 0.0),
                  int(constraint is not None), *(think_stop or (0, -1)), int(continued)] + code
        from tensorfold.engine.grammar import pack

        self._ring()                                   # wakes ranks 1.., which idle on the store, not in the all-gather
        self._share(header)
        self._share(list(prompt))
        if constraint is not None:                     # the request's grammar: ranks 1.. compile the same
            self._share(pack(constraint))
        stats = self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, constraint,
                          _think(*(think_stop or (0, -1))), continued)
        stats.update(policy=spec, drafts=draft)
        return stats

    def score_labels(self, prompt_ids: list[int], label_ids: list[int]) -> tuple[list[float], float]:
        """Both ranks prefill the prompt and return its label logits plus the full-vocabulary logsumexp."""

        prompt = [int(token) for token in prompt_ids]
        labels = [int(token) for token in label_ids]
        if not prompt:
            raise ValueError("empty prompt")
        if not labels:
            raise ValueError("empty labels")
        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        self._ring()                    # ranks 1.. wait on the store before this header, as a chat request does
        self._share([0, len(labels)])   # max_tokens on a chat header is at least 1, so 0 is a score
        self._share(prompt)
        self._share(labels)
        return self._score_local(prompt, labels)

    def _score_local(self, prompt: list[int], labels: list[int]) -> tuple[list[float], float]:
        from tensorfold.families.glm5_next.cuda.score import prompt_logits
        from tensorfold.server.decisions import reduce_vocab_shards

        # The score prefill writes attention rows from position 0. Save or drop every snapshot those rows
        # still belong to, then stop naming them: the next chat must not resume the decision as that conversation.
        self._take_over([])
        self.live = []
        self.e.settle()                 # the score prefill's collectives are uncaptured
        local = prompt_logits(self.e, prompt)
        rows = [local] if self.comm is None else self._gather_floats(local)
        if self.rank != 0:
            return [], 0.0
        return reduce_vocab_shards(rows, labels, len(rows[0]))

    def _gather_floats(self, values: list[float]) -> list[list[float]]:
        """Every rank's float32 list, in rank order."""

        torch = self.torch
        mine = torch.tensor(values, dtype=torch.float32, device="cuda")
        got = torch.empty((self.world * len(values),), dtype=torch.float32, device="cuda")
        self.comm.all_gather(mine, got)
        return [[float(item) for item in row] for row in got.view(self.world, len(values)).tolist()]

    def follow(self) -> None:
        """Ranks 1..: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        while True:
            self._await_bell()
            header = self._share(None)
            if len(header) == 2 and header[0] == 0:     # a decision: both ranks prefill, neither samples
                prompt = self._share(None)
                labels = self._share(None)
                self._score_local(prompt, labels)
                continue
            (max_tokens, stop_eos, draft, cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi,
             shaped, think_n, think_end, continued, *code) = header
            prompt = self._share(None)
            packed = self._share(None) if shaped else []
            constraint = None
            if packed:                                  # compiled here as on rank 0
                from tensorfold.engine import grammar

                constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
            temperature = _ints_f64(t_lo, t_hi)
            seed = (s_top << 62) | (s_hi << 31) | s_lo
            sampling = (Sampling(seed, temperature, top_k, _ints_f64(p_lo, p_hi), _ints_f64(m_lo, m_hi))
                        if temperature > 0 else None)
            hit = None
            if cached:
                hit = next((c for c in self.cache if len(c.ids) == cached and prompt[:cached] == c.ids), None)
                if hit is None:
                    raise RuntimeError(f"rank {self.rank} has no snapshot of the {cached} tokens rank 0 resumes from")
            self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, code, hit, bool(draft),
                      constraint, _think(think_n, think_end), bool(continued))
