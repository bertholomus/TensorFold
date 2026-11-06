"""The Mistral-Large-4 serving engine: EP4 experts + DP4 attention over ``TF world`` Spark ranks.

Pattern: ``deepseek_v41/cuda/engine.py``'s ``DsEngine.__init__`` (comm → ``weights.load(model_dir,
rank, world)`` → ``Model`` → context limit → memory ceiling), reshaped for data-parallel attention
(note D2 §1): rank 0 serves and schedules; the followers run **their own rows** from each step
payload (``cuda/dp4.py``) and answer with their sampled tokens; the MoE all-to-all inside the
forward (``cuda/a2a.py``) is the lane's only per-layer sync. world=1 stays runnable — no comm, no
link, all experts local — which is the tiny-pack smoke and this repo's CPU test lane.

KV is the phase-2a naive cache, pool-**sized** like the engine plans it: fp16 82,688 B/token on the
real dims (D2 §6 — recomputed from ``Cfg``, not hardcoded), pool tokens from live ``MemAvailable``
minus floor and headroom (dsv41 engine.py:169-174's ceiling pattern). Extent-based paging and the
CUDA-graph buckets are phase B; the placement ledgers already speak that currency, so adopting the
paged planes does not move the scheduler.

Sampling runs where the rows are (DP4: logits never gather): each rank samples its own rows with
the stream's rule — rank 0's ``generate`` broadcasts the sampling on admission, followers unpack
the same rule. Greedy (``sampling=None``) is the exact path the CPU gates prove.

The fast path (DESIGN-CP4 §8, TF_ML4_FAST, on by default on CUDA) replaces the eager forward and its fp16 cache with
``cp4.Ml4Cp`` over one fp8 ``KvPool`` per rank, and adds the striped stream: a request past ``dp_limit`` tokens gets
an extent on every rank and runs as ``["cp", ...]`` rounds, one striped stream at a time, its rounds taking turns with
the DP rounds of whatever else is live. DP streams are the same forward as a group of one.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

PREFILL_CHUNK = int(os.environ.get("TF_ML4_PREFILL_CHUNK") or 512)      # D2 §4(a) chunk cap, tune in phase B
MEM_FLOOR_GIB = float(os.environ.get("TF_ML4_MEM_FLOOR_GIB") or 10.0)   # D2 §6 OS floor (warn under 8 — GB10 OOM)
KV_HEADROOM_GIB = float(os.environ.get("TF_ML4_KV_HEADROOM_GIB") or 7.0)  # D2 §6 fallback when no probe (m2b6)
HEADROOM_GIB = float(os.environ.get("TF_ML4_HEADROOM_GIB") or 0.0)     # >0 overrides the probe (m2b6; 0 = measured)
POOL_SAFETY_GIB = float(os.environ.get("TF_ML4_POOL_SAFETY_GIB") or 2.0)   # margin behind a forced TF_ML4_KV_TOKENS
DP_LIMIT = int(os.environ.get("TF_ML4_DP_LIMIT") or 131072)            # DESIGN-CP4 §8: past this a stream stripes
MAX_TEMP = float(os.environ.get("TF_ML4_CP_MAX_TEMP") or 0.0)          # °C; 0: a prompt's rounds never wait for a hot rank
TRIM_PROMPT = 32768                                                    # a prompt this long left its tokenizer's buffers in malloc's arenas
ROUND = int(os.environ.get("TF_ML4_CP_ROUND") or 0)                     # tokens of a round; 0: one block a rank
TP = os.environ.get("TF_ML4_TP", "1") == "1"                            # every stream striped, decode split over the ranks
COOL_ROWS = 4096                                                          # a round this wide waits for a hot rank
COPY_ROWS = int(os.environ.get("TF_ML4_CP_COPY", "163840"))             # a prompt up to here runs over a copy of its rows on every rank
TP_ROWS = int(os.environ.get("TF_ML4_TP_ROWS") or 8)                    # decode rows in one step
SLICE = int(os.environ.get("TF_ML4_CP_SLICE") or 12)                    # a prompt round's layers between two looks at the queue
STEPS = int(os.environ.get("TF_ML4_TP_STEPS") or 2)                     # decode steps between two slices
LOOK = 1.0                                                              # s between two looks at a waiting request's client


class NoRoom(RuntimeError):
    """No run of free rows for a new stream right now."""


def plan_pool_tokens(avail_bytes: int, kv_bpt: int, floor_gib: float, reserve_gib: float) -> int:
    """D2 §6's formula, one place (m2b6): pool = (MemAvailable - floor - reserve) / B-per-token."""

    budget = int(avail_bytes) - int((float(floor_gib) + float(reserve_gib)) * 2 ** 30)
    if budget <= 0:
        raise RuntimeError(f"no KV budget left: MemAvailable {int(avail_bytes) / 2 ** 30:.1f} GiB < floor "
                           f"{floor_gib} + reserve {reserve_gib} GiB (D2 §6)")
    return budget // int(kv_bpt)


def check_pool_request(tokens: int, avail_bytes: int, kv_bpt: int, floor_gib: float, reserve_gib: float,
                       safety_gib: float = POOL_SAFETY_GIB) -> int:
    """A forced TF_ML4_KV_TOKENS must still leave floor + measured reserve + a safety margin inside
    MemAvailable — refuse with the numbers in the message, never arrive at an OOM (m2b6 item 3)."""

    allowed = (int(avail_bytes) - int((float(floor_gib) + float(reserve_gib) + float(safety_gib)) * 2 ** 30)) \
        // int(kv_bpt)
    if int(tokens) > allowed:
        raise RuntimeError(f"TF_ML4_KV_TOKENS {tokens} cannot be backed: only {allowed} tokens fit behind "
                           f"MemAvailable {int(avail_bytes) / 2 ** 30:.1f} GiB - floor {floor_gib} - reserve "
                           f"{reserve_gib} - safety {safety_gib} GiB")
    return int(tokens)


def read_mem_avail() -> int | None:
    """Live MemAvailable in bytes, or None when there is no /proc (the macOS dev lane). One place so
    the ceiling and the pool cannot read two different snapshots (m2b10)."""

    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        return int(info["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        return None


def kv_bytes_per_token(cfg) -> int:
    """fp16 KV bytes per token on this config: 2 B x (kv_lora x writers + rd x rope writers +
    idx_dim x full indexers). The real dims give D2 §6's 82,688; the 9 reader layers and the
    NoPE layers' zero-width rope slices are why it is not ``72 x 1152``."""

    writers = [i for i, s in enumerate(cfg.kv_source_layers) if s == i]
    rope_w = [i for i in writers if cfg.rope_flags[i]]
    full = sum(1 for t in cfg.indexer_types if t == "full")
    return 2 * (cfg.kv_lora * len(writers) + cfg.rd * len(rope_w) + cfg.idx_dim * full)


def _pack(sampling) -> list | int:
    if sampling is None:
        return 0
    from tensorfold.engine.exact_sampling import Sampling   # numpy-backed: only when a lane samples

    assert isinstance(sampling, Sampling)
    return [int(sampling.seed), float(sampling.temperature), int(sampling.top_k), float(sampling.top_p),
            float(sampling.min_p)]


def _unpack(packed: Any):
    if packed == 0:
        return None
    from tensorfold.engine.exact_sampling import Sampling

    return Sampling(*packed)


class Ml4Engine:
    def __init__(self, model_dir: str | Path | None = None, *, rank: int = 0, world: int = 1,
                 master: str = "", port: int = 29551, context: int | None = None, weights=None,
                 comm=None, link=None, device: str = "cuda", pool_tokens: int | None = None,
                 log: Callable[[str], None] = print, fast: bool | None = None) -> None:
        from .model import Ml4Model
        from .weights import load as wload

        if device == "cuda":
            torch.cuda.set_device(0)
        self.rank, self.world, self._master, self.device = int(rank), int(world), master, device
        self._dev = torch.device(device)
        self.log = log
        self.fast = (device == "cuda" and os.environ.get("TF_ML4_FAST", "1") != "0") if fast is None else bool(fast)
        self._snap: tuple | None = None                       # m2b10: the ONE post-load mem snapshot
        self._reserve: tuple | None = None                    # (GiB, source) the pool was sized with
        self.w = weights if weights is not None else wload(model_dir, rank=int(rank), world=int(world), log=log)
        cfg = self.w.cfg
        self.comm = comm if int(world) > 1 else None
        if int(world) > 1 and self.comm is None and device == "cuda":
            from tensorfold.cuda.comm import open_comm      # honours TF_COMM_BACKEND: the RoCE a2a drop-in (D2 O2)

            from .tp_cuda import fast_comm

            self.comm = fast_comm(open_comm(int(rank), int(world), master, port), log)
        if self.comm is not None and not self.fast and os.environ.get("TF_ML4_TRACE", "1") == "1":
            from .dp4 import TraceComm

            # m2b11 (mandate 2): number EVERY collective this rank enters so the four node logs
            # diff on [ml4-trace] lines and name exactly where pairing stopped; TF_ML4_TRACE=0 off.
            # The fast path enters 448 collectives a token, so it stays untraced.
            self.comm = TraceComm(self.comm, log)
        self.model = None                                     # the fast path keeps no folds (DESIGN-CP4 §10)
        if not self.fast:
            self.model = Ml4Model(self.w)
            self.model.comm = self.comm
        self.kv_bpt = kv_bytes_per_token(cfg)
        if self.fast:
            from .cp4 import KvPool

            self.kv_bpt = KvPool.row_bytes(cfg)
            self._fast_ops()                                  # before the snapshot: it hands memory back
        if device == "cuda":
            # m2b10: the ceiling takes the ONE snapshot (post-load, pre-serving) that _pool_from_mem
            # then sizes from — and it must be set BEFORE any growth (probe, pool, first forward).
            self._memory_ceiling(log)
        self.pool_tokens = int(pool_tokens if pool_tokens is not None else self._pool_from_mem(log))
        self.ledgers = None                                   # rank 0 only (D2 §5 ledgers)
        self.placement = None
        self.rounds = None
        self.link = link
        self.seq, self.next_sid = 0, 0
        self.limit = int(context or 131072)
        self.dp_limit = min(self.limit, DP_LIMIT)
        self.fwd, self.seqs, self.extents, self._cp_turn = None, {}, None, False
        self.lane, self.graphs = None, None                   # tp.py's step: every stream striped, decode on all ranks
        self.flights: dict = {}                               # rounds between their layers, by stream
        self._steps = 0                                       # decode steps since the last slice
        if self.fast:
            self._build_forward(log)
        self.concurrent = self.lane is not None                # server.py: requests come in together, one round loop serves them
        self.eos = self._read_eos(model_dir)
        self.tp = int(world)                                  # server.py's background-yield check
        if int(rank) == 0:
            from .dp4 import Ledgers, Placement, RoundBuilder

            self.ledgers = Ledgers(self.pool_tokens, int(world))
            self.placement = Placement(self.ledgers)
            self.rounds = RoundBuilder(chunk=PREFILL_CHUNK)
            if self.fast:
                from .dp4 import Extents

                self.extents = [Extents(self.pool_tokens) for _ in range(int(world))]
        self._caches: dict[int, dict] = {}
        # Rank 0's broadcasts can come from DIFFERENT server threads (a concurrent /v1 request
        # admits while another thread's generate runs the rounds). Without this lock the socket
        # can carry admit2 before step1 while the comm sees step1's gather first — the digests
        # then pair across ops and the lane dies at its first model collective (load4,
        # 2026-11-06; t6's log: rank 1 "admit@2 mismatch, then step@1 mismatch"). The lock makes
        # rank 0's (send, gather) pairs a total order, which is what agree() and the followers'
        # FIFO digests assume. Followers have one follow thread; nothing to lock there.
        self._bcast_lock = threading.RLock()
        self._turn = threading.Lock()
        self._admitting = 0                                    # streams on their way in: run_round waits for them
        self._temp: float | None = None                        # the GPU's, once a long prompt has come
        self._work = threading.Event()                         # the lane's round loop has something to look at
        self._looping = False
        self._failed: Exception | None = None                  # what ended the round loop
        self._ended: list[int] = []                            # finished streams the other ranks still hold
        self._mirror: dict[int, Any] = {}                     # followers' admitted streams (sampling, stops)
        self._idle = None                                     # the empty-participation cache, built on demand
        if int(world) > 1 and link is None and device == "cuda" and int(rank) == 0 and \
                getattr(self.comm, "store", None) is not None:
            from .dp4 import Link

            self.link = Link(self.comm.store, rank=0, world=int(world), host=master)
        self._budget_check(log)        # m2b10: ceiling (early) must cover resident + pool + activations
        log(f"[ml4-engine] rank {rank} of {world}: KV {self.kv_bpt} B/token {'fp8' if self.fast else 'fp16'} (D2 §6), pool "
            f"{self.pool_tokens} tokens/rank, prefill chunk {PREFILL_CHUNK}, context {self.limit}, "
            f"device {device}" + (f", striped past {self.dp_limit}" if self.fast else ""))

    def _fast_ops(self) -> None:
        from . import cp4

        if self.device == "cuda":
            from .cp4_cuda import CudaOps

            self.ops = CudaOps(self._dev)
            self.ops.prepare(self.w)
            torch.cuda.empty_cache()
        else:
            self.ops = cp4.Ops()

    def _build_forward(self, log) -> None:
        """The fast path's pool and forward: fp8 rows on a GPU, fp32 rows on the CPU lane (TF_ML4_KV_KIND overrides)."""

        from . import cp4

        kind = os.environ.get("TF_ML4_KV_KIND") or ("fp8" if self.device == "cuda" else "raw")
        pool = cp4.KvPool(self.w.cfg, self.pool_tokens + 1, device=self._dev, kind=kind)   # the last row: tp.py's scratch
        if self.device == "cuda":
            from .cp4_fused import forward

            self.fwd = forward(self.w, self.comm, pool, self.ops)
            self.fwd.open_slots(self.pool_tokens * self.world)
        else:
            self.fwd = cp4.Ml4Cp(self.w, self.comm, pool, self.ops)
        self.cp_width = ROUND or self.world * cp4.BLOCK
        self.idle = cp4.Seq(cp4.Stripe(1, cp4.BLOCK))
        if self.world > 1 and TP:
            from . import tp

            if self.device == "cuda":
                from .tp_cuda import FusedTp

                self.lane = FusedTp(self.fwd)
                self.graphs = tp.Graphs(self.lane)
            else:
                self.lane = tp.Ml4Tp(self.fwd)
            self.lane.use(cp4.Stripe(self.world, cp4.BLOCK))
        log(f"[ml4-engine] rank {self.rank}: fast path, {kind} pool {pool.nbytes() / 2 ** 30:.2f} GiB, "
            f"MemAvailable now {(read_mem_avail() or 0) / 2 ** 30:.2f} GiB")

    @staticmethod
    def _read_eos(model_dir) -> tuple:
        if model_dir is None:
            return (0,)
        try:
            cfg = json.loads((Path(model_dir) / "config.json").read_text())
        except OSError:
            return (0,)
        e = cfg.get("eos_token_id", 2)
        return (int(e),) if isinstance(e, int) else tuple(int(x) for x in e)

    def _probe_device_audit(self) -> None:
        """m2b8 (mandate: fail loud, name the offender): every tensor the probe's forward reads must
        already live on the engine's device. Walks the forward's ACTUAL field list (model.py
        ``_layer``/``forward``), not the whole graph — a tensor forward never reads (a CPU-side
        ``alpha``) must not veto the probe. Offenders raise with ``name@device``; the probe's except
        turns that into the fallback line, so a bad audit never wedges the lane. In-function
        allocations (the m2b8 a2a bug: cuda values index_put into a deviceless cpu buffer) are not
        module state — device-pinned tests own those (tests/test_ml4_engine.py)."""

        dev = self._dev

        def same(t) -> bool:                                   # cuda == cuda:0; catches cuda vs cpu
            return t.type == dev.type and (t.index is None or dev.index is None or t.index == dev.index)

        off: list = []

        def see(name, v) -> None:
            d = getattr(v, "device", None)
            if d is not None and not same(d):
                off.append(f"{name}@{d}")

        def seeslot(name, v) -> None:                          # a plain tensor OR an Exl3 slot object
            if v is None:
                return
            if isinstance(v, torch.Tensor):
                see(name, v)
                return
            dd = getattr(v, "__dict__", None)
            if isinstance(dd, dict):
                for k, t in dd.items():
                    if isinstance(t, torch.Tensor):
                        see(f"{name}.{k}", t)

        see("model.inv_freq", self.model.inv_freq)
        for L, fold in enumerate(self.model.folds):
            for j, t in enumerate(fold):
                see(f"model.folds[{L}][{j}]", t)
        see("weights.embed", self.w.embed)
        see("weights.norm", self.w.norm)
        seeslot("weights.head", getattr(self.w, "head", None))
        seeslot("weights.head_bf16", getattr(self.w, "head_bf16", None))
        fields = ("attn_norm", "ffn_norm", "gate_w", "q", "o", "kv_a", "kv_a_norm",
                  "ssmax_s", "ssmax_b", "dense_gate", "dense_up", "dense_down",
                  "idx_wk", "idx_k_norm", "idx_k_norm_b", "idx_q_norm", "idx_q_norm_b",
                  "idx_wq_b", "idx_proj", "idx_ssmax_s", "idx_ssmax_b")
        for L, lay in enumerate(self.w.layers):
            for f in fields:
                seeslot(f"weights.layers[{L}].{f}", getattr(lay, f, None))
            seeslot(f"weights.layers[{L}].experts", lay.experts)
        if off:
            raise RuntimeError(f"probe refuses to run: tensors off {dev}: {', '.join(off[:8])}"
                               + (f" (+{len(off) - 8} more)" if len(off) > 8 else ""))

    def _probe_gate(self, refuse: str) -> str:
        """m2b11 (mandate 4): the all-or-nothing probe vote over the rendezvous store (TCP — a vote
        can never poison a lane the way an NCCL collective can). Publish my refusal reason (or
        "ok"), wait for the lane, and return the first refusal ANY rank published ("" = every rank
        cleared, so every rank may enter the probe's collectives together). The wait is generous
        (TF_ML4_PROBE_GATE_TIMEOUT, default 600 s) because arrival skew IS the weight load (rank 0
        reached the probe at 39 s on load5, followers at ~160 s); a rank that never arrives lands
        the whole lane on the same bounded skip, together — a half-probe never starts, which is
        exactly the wedge m2b8 only half-cured (its audit refusal still pulled one rank out of the
        others' in-flight collectives; on load5 all four ranks entered and all four hung)."""

        from datetime import timedelta

        store = self.comm.store
        store.set(f"tf_ml4_probe/{self.rank}", refuse or "ok")
        keys = [f"tf_ml4_probe/{r}" for r in range(self.world)]
        gate = float(os.environ.get("TF_ML4_PROBE_GATE_TIMEOUT") or 600.0)
        try:
            store.wait(keys, timedelta(seconds=gate))
        except Exception as exc:                                # store's timeout: a rank died pre-vote
            return f"lane vote incomplete after {gate:.0f} s (a rank never reached the probe gate: {exc})"
        bad = [f"rank {r}: {v.decode(errors='replace')}"        # store hands back bytes
               for r, v in enumerate(store.multi_get(keys)) if v != b"ok"]
        return "; ".join(bad)

    def _activation_probe(self, log):
        """m2b6 item 1: the peak CUDA-reserved growth of one PREFILL_CHUNK-token prefill, measured
        after the weight load and before pool sizing — the measured activation reserve replaces D2
        §6's fixed 7 GiB guess. Every rank probes at the same point of its own init, so the probe
        forward's lane collectives pair correctly (a2a's exchange order derives from the shared
        counts all_gather); a rank that arrives late just makes NCCL wait. TF_ML4_PROBE=0 skips.

        m2b8 hardening (load4: the probe raised cuda-vs-cpu on the followers and RANK 0 HUNG in it
        — the followers' exception pulled them out of the lane collectives while rank 0 waited in
        NCCL for ranks already gone): the body first device-audits, then runs on a daemon thread
        joined with a bounded wait (TF_ML4_PROBE_TIMEOUT, default 120 s) — ANY failure, including
        a stuck collective, lands on the 7 GiB fallback instead of wedging the load. Honest caveat
        for t6: a thread abandoned inside ncclAllGather leaves that comm suspect; the timeout turns
        a permanent hang into a bounded stall + a loud line, it does not un-poison a desynced
        comm — restart a lane whose probe timed out (or run TF_ML4_PROBE=0)."""

        if self.fast or self.device != "cuda" or os.environ.get("TF_ML4_PROBE", "1") != "1":
            return None
        timeout = float(os.environ.get("TF_ML4_PROBE_TIMEOUT") or 120.0)
        out: dict = {}

        # m2b11 (mandate 4): decide BEFORE the daemon thread enters any collective — audit runs here
        # (on-device reads only, no collectives), then the lane vote (_probe_gate, TCP store). A
        # refusal on ANY rank returns None on EVERY rank, so no rank is ever left alone in the
        # probe forward's all-gather. The thread body no longer audits (that was the wedge: an
        # exception on a follower pulled it out while rank 0 waited in NCCL for a rank already gone).
        refuse = ""
        try:
            self._probe_device_audit()
        except BaseException as exc:                            # noqa: BLE001  (refusal is data, not a crash)
            refuse = str(exc)
        if self.world > 1 and getattr(self.comm, "store", None) is not None:
            refuse = self._probe_gate(refuse)
        if refuse:
            log(f"[ml4-engine] rank {self.rank}: activation probe SKIPPED lane-wide ({refuse}); reserve "
                f"falls back to {KV_HEADROOM_GIB} GiB (D2 §6 O5); every rank skipped together")
            return None
        mark = getattr(self.comm, "trace", None)                # trace marker (no-op if tracing off)
        if mark is not None:
            mark("activation probe ENTER (lane vote cleared)")

        def work():
            try:
                from .model import new_cache

                torch.cuda.synchronize()
                base = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                ids = torch.zeros((1, PREFILL_CHUNK), dtype=torch.long, device=self._dev)
                from .dp4 import trace_tag
                with trace_tag(self.comm, "probe"):
                    with torch.inference_mode():
                        self.model.forward(ids, new_cache(self.w.cfg))
                torch.cuda.synchronize()
                out["gib"] = max(0.0, (torch.cuda.max_memory_reserved() - base) / 2 ** 30)
            except BaseException as exc:                        # noqa: BLE001  (thread edge: surface via out)
                out["err"] = exc

        th = threading.Thread(target=work, name="ml4-activation-probe", daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():                                       # m2b8: never let the probe wedge the lane
            log(f"[ml4-engine] rank {self.rank}: activation probe TIMED OUT after {timeout:.0f} s — reserve "
                f"falls back to {KV_HEADROOM_GIB} GiB (D2 §6 O5); restart this lane before trusting it "
                f"(a stuck collective leaves the comm suspect); TF_ML4_PROBE=0 skips the probe")
            return None
        if "err" in out:
            log(f"[ml4-engine] rank {self.rank}: activation probe failed ({out['err']}); reserve falls back "
                f"to {KV_HEADROOM_GIB} GiB (D2 §6 O5)")
            return None
        gib = out["gib"]
        if mark is not None:
            mark("activation probe EXIT (forward complete on every rank)")
        log(f"[ml4-engine] rank {self.rank}: activation probe ({PREFILL_CHUNK}-token prefill): peak "
            f"CUDA reserved +{gib:.2f} GiB — that is the KV reserve now")
        return gib

    def _pool_from_mem(self, log) -> int:
        """Pool tokens from the live host view after the weight load and the activation probe:
        MemAvailable - floor - reserve, reserve = TF_ML4_HEADROOM_GIB if set > measured probe > D2
        §6's 7 GiB (m2b6: measure, don't reserve blindly). A forced TF_ML4_KV_TOKENS is validated
        against the same numbers plus a 2 GiB safety and REFUSES loudly when unbackable.
        No /proc (macOS dev lane): a forced value passes unvalidated; otherwise refuse to guess."""

        forced = int(os.environ.get("TF_ML4_KV_TOKENS", "").strip() or 0)
        measured = self._activation_probe(log)
        if HEADROOM_GIB > 0:
            reserve, src = HEADROOM_GIB, "env"
        elif measured is not None:
            reserve, src = measured, "probe"
        else:
            reserve, src = KV_HEADROOM_GIB, "D2 §6 default"
        self._reserve = (reserve, src)                          # m2b10: _budget_check reads this back
        if MEM_FLOOR_GIB < 8.0:
            log(f"[ml4-engine] rank {self.rank}: WARNING floor {MEM_FLOOR_GIB} GiB < 8 — the GB10 OOM'd "
                f"below that in the dsv41 runs (D2 §6); run this low only with eyes on the lane")
        if self._snap is not None:
            avail = self._snap[0]                               # m2b10: the SAME snapshot the ceiling got
        else:
            avail = read_mem_avail()
        if avail is None:
            if forced:
                log(f"[ml4-engine] rank {self.rank}: TF_ML4_KV_TOKENS {forced} accepted unvalidated "
                    f"(no /proc/meminfo here)")
                return forced
            raise RuntimeError("pool sizing needs /proc/meminfo (Linux) or TF_ML4_KV_TOKENS set") from None
        if forced:
            pool = check_pool_request(forced, avail, self.kv_bpt, MEM_FLOOR_GIB, reserve)
            log(f"[ml4-engine] rank {self.rank}: KV pool {pool} tokens (forced) — backed by MemAvailable "
                f"{avail / 2 ** 30:.1f} GiB - floor {MEM_FLOOR_GIB} - reserve {reserve:.2f} ({src}) "
                f"- {POOL_SAFETY_GIB} GiB safety")
            return pool
        pool = plan_pool_tokens(avail, self.kv_bpt, MEM_FLOOR_GIB, reserve)
        # Every term on one line (m2b5's audit trail, m2b6's measured reserve): MemAvailable is read
        # AFTER the weight load, so on the unified-memory GB10 the weights are already gone from it.
        log(f"[ml4-engine] rank {self.rank}: KV pool {pool} tokens = {pool * self.kv_bpt / 2 ** 30:.2f} GiB "
            f"from MemAvailable {avail / 2 ** 30:.1f} GiB - floor {MEM_FLOOR_GIB} - reserve {reserve:.2f} "
            f"({src}) at {self.kv_bpt} B/token; resident now "
            f"{(torch.cuda.memory_allocated() if self.device == 'cuda' else 0) / 2 ** 30:.1f} GiB")
        return pool

    def _memory_ceiling(self, log) -> None:
        """The ONE post-load memory snapshot, taken before any serving growth, and the allocator
        ceiling derived from it. m2b9's load4 died at the token gate — cap 8.66 GiB with 93.78 GiB
        of weights already resident — from two faults, both fixed here (m2b10):

        * ORDER: the pool used to be sized first (one /proc read) and the ceiling set later (a
          second read); the whole budget now comes from THIS one snapshot, and ``self._snap`` hands
          the same numbers to _pool_from_mem so the two cannot disagree.
        * SEMANTICS: ``set_per_process_memory_fraction`` caps the process TOTAL. The old fraction
          was (available - floor)/total — the dsv41 pattern MINUS its ``reserved +`` term (dsv41
          engine.py:185) — so the cap landed UNDER the already-resident weights and every later
          allocation failed. Correct shape (item 3's growth cap): ceiling = resident + max(0,
          available - floor). The weights sit inside the cap; torch may GROW only into what the
          host can give while keeping TF_ML4_MEM_FLOOR_GIB free — a step over the cap fails as a
          CUDA OOM for that request instead of the kernel OOM killer taking the box (D2 §6, the
          dsv41 systemd-manager lesson).

        With the pool sized from the same snapshot the budget closes by construction:
        pool * kv_bpt + reserve = available - floor = the growth allowance; _budget_check asserts
        ceiling >= resident + pool + activations and fails with all four numbers."""

        self._snap = None
        if MEM_FLOOR_GIB <= 0:
            return
        avail = read_mem_avail()
        if avail is None:
            return
        torch.cuda.synchronize()
        total = torch.cuda.mem_get_info()[1]
        resident = torch.cuda.memory_reserved()
        growth = max(0, avail - int(MEM_FLOOR_GIB * 2 ** 30))
        ceiling = resident + growth
        self._snap = (avail, total, resident, growth)
        torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, ceiling / total)) if total else 1.0)
        log(f"[ml4-engine] rank {self.rank}: allocator ceiling {ceiling / 2 ** 30:.1f} GiB = resident "
            f"{resident / 2 ** 30:.1f} + growth {growth / 2 ** 30:.1f} (MemAvailable "
            f"{avail / 2 ** 30:.1f} GiB - floor {MEM_FLOOR_GIB}) — pool sizes from the same snapshot (m2b10)")

    def _budget_check(self, log) -> None:
        """m2b10's loud invariant: the ceiling must cover resident weights + pool + activations, or
        the first token gate OOMs cryptically (m2b9: '93.78 GiB in use'). By construction pool *
        kv_bpt + reserve == growth (one snapshot), so this only bites on an init-ordering
        regression or a caller passing ``pool_tokens`` straight to the constructor past the plan —
        and then it fails at INIT with all four numbers, not at the first request."""

        if self._snap is None or self.device != "cuda":
            return
        avail, total, resident, growth = self._snap
        reserve, src = self._reserve or (KV_HEADROOM_GIB, "D2 §6 default")
        pool_b = self.pool_tokens * self.kv_bpt
        need = pool_b + int(float(reserve) * 2 ** 30)
        if growth + (1 << 20) < need:
            raise RuntimeError(
                f"memory budget broken (rank {self.rank}): ceiling "
                f"{(resident + growth) / 2 ** 30:.2f} GiB < resident {resident / 2 ** 30:.2f} + pool "
                f"{pool_b / 2 ** 30:.2f} + activations {float(reserve):.2f} ({src}) = "
                f"{(resident + need) / 2 ** 30:.2f} needed; MemAvailable {avail / 2 ** 30:.2f} GiB, "
                f"floor {MEM_FLOOR_GIB} — lower TF_ML4_KV_TOKENS/TF_ML4_HEADROOM_GIB or raise "
                f"TF_ML4_MEM_FLOOR_GIB")
        log(f"[ml4-engine] rank {self.rank}: budget closed — ceiling "
            f"{(resident + growth) / 2 ** 30:.1f} GiB covers resident {resident / 2 ** 30:.1f} + pool "
            f"{pool_b / 2 ** 30:.2f} + activations {float(reserve):.2f} ({src})")

    # -- one step: this rank's rows (or an empty round to stay in the a2a collectives) --------------------------------

    def _pick(self, sid: int, logits: torch.Tensor, pos: int) -> int:
        """A token from the row's logits [V] by its stream's rule, position-keyed like every CUDA family."""

        stream = self.rounds.streams.get(sid) if self.rounds is not None else self._mirror.get(sid)
        sampling = stream.sampling if stream is not None else None
        if sampling is None:
            return int(logits.argmax())
        from tensorfold.cuda.sampling import sample_rows

        return int(sample_rows(logits[None], [pos], sampling)[0])

    def _open(self, sid: int, place: list) -> None:
        """A stream's rows on this rank, as rank 0 placed them."""

        from . import cp4

        world, block, bases, rows, copy = place
        if bases[self.rank] is not None:
            self.seqs[sid] = cp4.Seq(cp4.Stripe(world, block), bases[self.rank], rows, *((copy[0][self.rank], copy[1]) if copy else ()))

    def _step_fast(self, rows: list) -> list:
        """A DP round: this rank's row as a group of one; a rank without a row still serves the expert all-to-alls."""

        with torch.inference_mode():
            if not rows:
                self.fwd.round(self.idle, 0, 0, torch.zeros((0,), dtype=torch.long, device=self._dev))
                return []
            got: list = []
            for row in rows:
                sid, pos, toks = int(row[0]), int(row[2]), list(row[3])
                last = pos + len(toks) - 1
                logits = self.fwd.round(self.seqs[sid], pos, len(toks),
                                        torch.tensor(toks, dtype=torch.long, device=self._dev), last=last)
                got.append([sid, self._pick(sid, logits, last)])
            return got

    def _cool(self) -> None:
        """A hot rank pauses between two slices of a long prompt (one node hung past 82 °C, another reset at 85-89): the others wait in the
        slice's first collective, so the lane slows down instead of losing a node an hour into a prompt. One short pause a slice,
        and none for a short prompt: a request that arrives meanwhile goes next."""

        if not MAX_TEMP:
            return
        if self._temp is None:
            self._temp = 0.0
            threading.Thread(target=self._read_temp, daemon=True).start()
        if self._temp >= MAX_TEMP:
            self.log(f"[ml4-cp] rank {self.rank}: {self._temp:.0f} °C, pausing the round (limit {MAX_TEMP:.0f})")
            time.sleep(0.5)

    def _read_temp(self) -> None:
        """The GPU's temperature twice a second, off the round's thread: a read takes 20 ms, and one every two seconds let a rank run 4 °C past the limit."""

        while True:
            try:
                got = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10)
                self._temp = float(got.stdout.split()[0])
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                self._temp = 0.0
            time.sleep(0.5)

    def cp_step(self, op: list) -> list:
        """A striped round's next layers on this rank (``op`` without a count: all of them); when they are the last, the
        owner of ``last`` answers with a token."""

        _, _, sid, start, n, toks, last = op[:7]
        sid, c = int(sid), self.w.cfg
        seq = self.seqs[sid]
        f = self.flights.pop(sid, None)
        with torch.inference_mode():
            if n >= COOL_ROWS:
                self._cool()
            if f is None:
                p = torch.arange(start, start + n)
                mine = torch.tensor(toks, dtype=torch.long)[seq.stripe.rank(p) == self.rank].to(self._dev)
                f = self.fwd.begin(seq, start, n, mine, last)
            self.fwd.advance(f, op[7] if len(op) > 7 else c.n_layers)
            if f.at < c.n_layers:
                self.flights[sid] = f
                return []
            logits = self.fwd.finish(f)
        return [] if logits is None else [[sid, self._pick(sid, logits, last)]]

    def tp_step(self, op: list) -> list:
        """One decode step of ``op``'s rows on every rank: their next tokens, the same on all ranks unless sampled."""

        from . import tp

        _, _, rows, sampled = op
        k = self.w.cfg.idx_topk
        seqs = [self.seqs[int(r[0])] for r in rows]
        positions = [int(r[1]) for r in rows]
        width = max(tp.width(s.stripe, p + 1, self.world, k) for s, p in zip(seqs, positions))
        padded = 1 << (len(rows) - 1).bit_length()
        s = tp.make_rows(seqs, positions, self.rank, self._dev, n=width, k=k, scratch=self.pool_tokens, pad=padded)
        tokens = [int(r[2]) for r in rows] + [int(rows[0][2])] * (padded - len(rows))
        tokens = torch.tensor(tokens, dtype=torch.long, device=self._dev)
        with torch.inference_mode():
            if self.graphs is not None:
                ids, logits = self.graphs.run(tokens, s)
            else:
                logits = self.lane.step(tokens, s)
                ids = self.lane.best(logits)
            got = [[int(r[0]), int(t)] for r, t in zip(rows, ids.tolist())]
            if sampled:
                every = torch.empty((self.world, *logits.shape), dtype=logits.dtype, device=logits.device)
                self.comm.all_gather(logits.reshape(-1), every.view(-1))
                if self.rank == 0:
                    for i, r in enumerate(rows):
                        got[i][1] = self._pick(int(r[0]), every[:, i].reshape(-1), int(r[1]))
        return got

    def step(self, rows: list) -> list:
        from .model import new_cache

        if self.fast:
            return self._step_fast(rows)
        got: list = []
        if not rows:                                           # idle rank: the layer a2a is lane-wide (D2 §1)
            if self.comm is not None:
                self._idle = self._idle or new_cache(self.w.cfg)
                self.model.forward(torch.zeros((1, 0), dtype=torch.long, device=self._dev), self._idle)
            return got
        for row in rows:
            sid, kind, pos, toks = int(row[0]), int(row[1]), int(row[2]), list(row[3])
            stream = self.rounds.streams.get(sid) if self.rounds is not None else self._mirror.get(sid)
            cache = self._caches.setdefault(sid, new_cache(self.w.cfg))
            assert cache["past"] == pos, f"stream {sid}: step says pos {pos}, this rank is at {cache['past']}"
            logits = self.model.forward(torch.tensor([toks], dtype=torch.long, device=self._dev), cache)
            sampling = (stream.sampling if stream is not None else None)
            if sampling is None:                             # greedy: the numpy-free path (CPU lane + exact gates)
                got.append([sid, int(logits[-1].argmax())])
            else:                                            # position-keyed rule, shared with every CUDA family
                from tensorfold.cuda.sampling import sample_rows

                got.append([sid, int(sample_rows(logits[-1:], [pos + len(toks) - 1], sampling)[0])])
        return got

    # -- the follower lane -------------------------------------------------------------------------------------------

    def follow(self) -> None:
        """cli's rank>0 path: rank 0's admissions and steps, in order, forever (multi.py's shape)."""

        from .dp4 import Link, OutOfStep, rows_for

        link = self.link or Link(self.comm.store, rank=self.rank, world=self.world, host=self._master)
        self.link = link
        print(f"[ml4-engine] rank {self.rank} ready, following rank 0", flush=True)
        while True:
            op = link.receive()
            if op is None:
                raise RuntimeError("rank 0's step link closed")
            kind = op[0]
            try:
                if kind == "stop":
                    return
                if kind == "end":
                    from .dp4 import agree

                    agree(self.comm, "admission", op, self._dev)
                    self._let_go(op[1])
                elif kind == "admit":
                    from .dp4 import agree

                    _, sid, prompt, count, packed, owner = op[:6]
                    agree(self.comm, "admission", op, self._dev)
                    from .dp4 import Stream

                    self._mirror[int(sid)] = Stream(int(sid), list(prompt), int(count), int(owner),
                                                    sampling=_unpack(packed))
                    if len(op) > 6:
                        self._open(int(sid), op[6])
                elif kind == "cp":
                    from .dp4 import agree, continues

                    if not continues(op):
                        agree(self.comm, "step", op, self._dev)
                    link.send_back(["tokens", op[1], self.cp_step(op)])
                elif kind == "tp":
                    from .dp4 import agree

                    agree(self.comm, "step", op, self._dev)
                    self.tp_step(op)
                elif kind == "step":
                    from .dp4 import agree

                    agree(self.comm, "step", op, self._dev)
                    got = self.step(rows_for(op, self.rank))
                    link.send_back(["tokens", op[1], got])
            except OutOfStep as exc:
                print(f"[ml4-engine] rank {self.rank}: {exc}", flush=True)

    # -- rank 0: admission and the round loop --------------------------------------------------------------------------

    def admit(self, prompt: list[int], count: int, sampling=None, session: str = "",
              on_tokens: Callable | None = None, stop_eos: bool = True, box: Any = None):
        from .dp4 import Stream

        total = len(prompt) + int(count)
        if total > self.limit:
            raise ValueError(f"prompt {len(prompt)} + {count} tokens exceeds this engine's context {self.limit}")
        if self.fast:
            with self._turn:
                self._admitting += 1
            try:
                with self._bcast_lock:                         # the free rows and the streams are the round loop's too
                    owner, place = self._place(total, session, len(prompt))
                    s = Stream(self.next_sid, prompt, int(count), owner, sampling=sampling, on_tokens=on_tokens,
                               stop_eos=stop_eos)
                    s.place, s.cp, s.box = place, owner < 0, box
                    self.next_sid += 1
                    self._open(s.sid, place)
                    # followers only need the rows: the step ops carry the tokens, so a 1M prompt is not sent twice
                    self._broadcast(["admit", s.sid, [], int(count), _pack(sampling), owner, place])
                    self.rounds.add(s)
            finally:
                with self._turn:
                    self._admitting -= 1
            return s
        owner = self.placement.assign(total, session)          # D2 §5: least-loaded, sticky by session
        self.ledgers.reserve(owner, total)
        s = Stream(self.next_sid, prompt, int(count), owner, sampling=sampling, on_tokens=on_tokens,
                   stop_eos=stop_eos)
        self.next_sid += 1
        self.rounds.add(s)
        self._broadcast(["admit", s.sid, list(prompt), int(count), _pack(sampling), owner])
        return s

    def _place(self, total: int, session: str, prompt: int) -> tuple[int, list]:
        """Where a new stream's rows go: (owner, ``[world, block, bases by rank, rows, its prompt's copy]``), owner -1 for a stripe."""

        from . import cp4

        if self.world > 1 and (self.lane is not None or total > self.dp_limit):
            if self.lane is None and any(s.cp for s in self.rounds.live()):
                raise RuntimeError("one striped stream at a time: this lane already serves one")
            st = cp4.Stripe(self.world, cp4.BLOCK)
            rows = -(-total // (st.world * st.block)) * st.block
            bases = [e.take(rows) for e in self.extents]
            if None in bases:
                for e, b in zip(self.extents, bases):
                    if b is not None:
                        e.give(b, rows)
                raise NoRoom(f"a {total}-token stream needs {rows} free rows in a row on every rank: "
                             f"free runs {[e.runs for e in self.extents]}")
            for r in range(self.world):
                self.ledgers.reserve(r, rows)
            return -1, [self.world, st.block, bases, rows, self._take_copy(prompt)]
        owner = self.placement.assign(total, session)          # D2 §5: least-loaded, sticky by session
        base = self.extents[owner].take(total)
        if base is None:
            self.ledgers.fill(owner, total)
            raise RuntimeError(f"rank {owner}: no run of {total} free rows ({self.extents[owner].runs})")
        self.ledgers.reserve(owner, total)
        bases: list = [None] * self.world
        bases[owner] = base
        return owner, [1, cp4.BLOCK, bases, total, None]

    def _take_copy(self, rows: int) -> list | None:
        """A prompt's copy on every rank, ``[bases, rows]``: a stripe's rows are enough for it to run without, so None when
        it is too long or the rows are not there."""

        if not 0 < rows <= COPY_ROWS:
            return None
        bases = [e.take(rows) for e in self.extents]
        if None in bases:
            for e, b in zip(self.extents, bases):
                if b is not None:
                    e.give(b, rows)
            return None
        for r in range(self.world):
            self.ledgers.reserve(r, rows)
        return [bases, rows]

    def _give_copy(self, s) -> None:
        if s.place[4] is not None:
            bases, rows = s.place[4]
            for r, base in enumerate(bases):
                self.extents[r].give(base, rows)
                self.ledgers.release(r, rows)
            s.place[4] = None

    def _broadcast(self, op: list) -> None:
        from .dp4 import agree

        if self.world > 1:
            with self._bcast_lock:            # rank 0's collect total order == socket order (see __init__)
                if self.link is not None:
                    self.link.send(op)        # the op must reach followers BEFORE the digest
                agree(self.comm, "admission", op, self._dev)   # all_gather: every rank digests the same op

    def run_round(self) -> bool:
        """One lane round: rank 0 builds the step (per-rank row sets, D2 §1), the digest check runs
        before any rank enters the step's model collectives, then every rank runs exactly one row
        (or nothing) and answers.

        The whole round is under ``_bcast_lock`` (m2b5, load4 2026-11-06): builder state, seq, the
        socket send, the digest gather, the model step's a2a and the back channel must form ONE
        uninterrupted segment of rank 0's activity — a concurrent admission from another server
        thread landing between the send and the gather made the comm pair rank 0's step digest
        against a follower's admission digest (t6: "rank 1 agreed admit@2 mismatched, then
        step@1 mismatched"). One follower thread already gives that total order on its side."""

        from .dp4 import agree, pack_step

        while self._admitting:                                 # a lock is not fair: the round loop would take it again at once,
            time.sleep(0.001)                                  # and a new stream would wait for a long prompt's end
        with self._bcast_lock:
            if self.lane is not None:
                return self._run_tp()
            cp, rows = None, None
            if self.fast:                                      # a striped round and a DP round take turns
                self._cp_turn = not self._cp_turn
                cp = self.rounds.cp_round(self.cp_width) if self._cp_turn else None
            if cp is None:
                rows = self.rounds.round_rows(self.world)
            if self.fast and cp is None and rows is None:
                cp = self.rounds.cp_round(self.cp_width)
            if cp is None and rows is None:
                return False
            self.seq += 1
            t0 = time.time()
            if cp is not None:
                cp[1] = self.seq
                op = cp
            else:
                op = pack_step(self.seq, self.rounds.bucket_of(rows), rows)
            if self.world > 1:
                if self.link is not None:
                    self.link.send(op)                         # send first: followers agree on what they hold
                agree(self.comm, "step", op, self._dev)
            got = self.cp_step(op) if cp is not None else self.step(op[3][self.rank])
            for r in range(1, self.world):
                back = self.link.receive_back(r)
                if back is None or back[0] != "tokens" or int(back[1]) != self.seq:
                    raise RuntimeError(f"rank {r}: step {self.seq} back-channel desync ({back})")
                got += back[2]
            if cp is None:
                self._apply(rows, got)
            else:
                self._apply_cp(op, got, t0)
        return True

    def _run_tp(self) -> bool:
        """Prompts run in slices of a round and decode rows in steps: STEPS steps, then a slice, while both wait."""

        from .dp4 import agree, continues

        rows = self.rounds.tp_rows(TP_ROWS) if self._steps < STEPS else None
        cp = None
        if rows is None:
            cp = self.rounds.cp_slice(self.cp_width, SLICE, self.w.cfg.n_layers)
            self._steps = 0
            if cp is None:
                rows = self.rounds.tp_rows(TP_ROWS)
        if cp is None and rows is None:
            return False
        self.seq += 1
        t0 = time.time()
        if cp is not None:
            cp[1] = self.seq
            op = cp
        else:
            self._steps += 1
            op = ["tp", self.seq, rows, any(self.rounds.streams[int(r[0])].sampling is not None for r in rows)]
        if self.world > 1:
            if self.link is not None:
                self.link.send(op)
            if not continues(op):
                agree(self.comm, "step", op, self._dev)
        if cp is None:
            for sid, tok in self.tp_step(op):
                self._emit(self.rounds.streams[sid], tok)
            return True
        s = self.rounds.streams[int(cp[2])]
        s.started = s.started or t0
        got = self.cp_step(op)
        for r in range(1, self.world):
            back = self.link.receive_back(r)
            if back is None or back[0] != "tokens" or int(back[1]) != self.seq:
                raise RuntimeError(f"rank {r}: step {self.seq} back-channel desync ({back})")
            got += back[2]
        self._apply_cp(op, got, t0)
        return True

    def _apply_cp(self, op: list, got: list, t0: float) -> None:
        _, seq, sid, start, n, _, last = op[:7]
        s = self.rounds.streams[int(sid)]
        if s.left:
            return
        if n > 1:
            dt = time.time() - (s.started or t0)
            self.log(f"[ml4-cp] stream {sid} round {seq}: {start + n}/{len(s.prompt)} prompt tokens, {dt:.1f} s, "
                     f"{n / max(dt, 1e-9):.0f} tok/s")
        s.started = 0.0
        if last is not None:
            assert len(got) == 1, f"striped round {seq}: {len(got)} answers for one row"
            self._give_copy(s)
            self._emit(s, int(got[0][1]))

    def _apply(self, rows: list, got: list) -> None:
        by_sid = {int(s): t for s, t in got}
        for rows_r in rows:
            for row in rows_r:
                sid, kind, pos, toks = int(row[0]), int(row[1]), int(row[2]), row[3]
                if sid not in by_sid:
                    continue                                   # a mid-prompt chunk emits no token
                s = self.rounds.streams[sid]
                tok = int(by_sid[sid])
                if kind == 1:                                  # prefill row: only the last chunk emits
                    if s.filled < len(s.prompt):
                        continue
                    self.ledgers.fill(s.owner, len(s.prompt))
                self._emit(s, tok)

    def _emit(self, s, tok: int) -> None:
        """A stream's next token. ``on_tokens`` is the server's: True stops it (cuda/server.py's contract)."""

        s.out.append(tok)
        s.pending = tok
        if s.box is not None:
            s.box.put([tok])
            stop = s.halt
        else:
            stop = s.on_tokens is not None and s.on_tokens([tok])
        if stop or (s.stop_eos and tok in self.eos) or len(s.out) >= s.count:
            self._finish(s)

    def _finish(self, s) -> None:
        s.done, s.pending = True, None
        if s.place is None:
            self.ledgers.release(s.owner, len(s.prompt) + s.count)
            return
        self._give_copy(s)
        _, _, bases, rows, _ = s.place
        for r, base in enumerate(bases):
            if base is not None:
                self.extents[r].give(base, rows)
                self.ledgers.release(r, rows)
        self.seqs.pop(s.sid, None)
        self.rounds.streams.pop(s.sid, None)
        if s.box is not None:
            self._ended.append(s.sid)
            s.box.put(None)

    def _let_go(self, sids: list) -> None:
        """Every rank: what finished streams still hold, a round between its layers among it."""

        for sid in sids:
            f = self.flights.pop(int(sid), None)
            if f is not None:
                self.fwd.let_go(f)
            self.seqs.pop(int(sid), None)
            self._mirror.pop(int(sid), None)

    def _sweep(self) -> None:
        """Between two rounds: a stream whose client left stops here, mid-prompt too, and the other ranks let go of it."""

        with self._bcast_lock:
            for s in self.rounds.live():
                if s.halt:
                    self._finish(s)
            if self._ended:
                ended, self._ended = self._ended, []
                self._let_go(ended)
                self._broadcast(["end", ended])

    def _loop(self) -> None:
        """Rank 0's one round loop. A request's thread only waits for its tokens, so a slow client slows nobody else down."""

        try:
            while True:
                self._sweep()
                if not self.run_round():
                    self._work.wait(0.5)
                    self._work.clear()
        except Exception as exc:                               # noqa: BLE001  every waiting request gets it
            traceback.print_exc()
            with self._bcast_lock:
                self._failed = exc
                for s in self.rounds.live():
                    s.box.put(None)

    def _serve(self, prompt: list[int], count: int, sampling, on_tokens: Callable | None, session: str, stop_eos: bool) -> dict:
        """One request of many: in as soon as its rows are free, then its tokens as the round loop makes them."""

        if len(prompt) >= TRIM_PROMPT:
            from .cp4_cuda import trim_host

            trim_host()                                        # 2 ms, and 400 MB of a 1M prompt's back to the GPU
        with self._turn:
            if not self._looping:
                self._looping = True
                threading.Thread(target=self._loop, name="ml4-rounds", daemon=True).start()
        box: queue.SimpleQueue = queue.SimpleQueue()
        while True:
            if self._failed is not None:
                raise RuntimeError(f"the lane's round loop ended: {self._failed!r}")
            try:
                s = self.admit(prompt, count, sampling=sampling, session=session, stop_eos=stop_eos, box=box)
                break
            except NoRoom:
                if on_tokens is not None and on_tokens([]):
                    return {"cached": 0, "tokens": 0}
                time.sleep(LOOK)
        self._work.set()
        while True:
            try:
                new = box.get(timeout=LOOK)
            except queue.Empty:
                new = []                                       # a prompt's rounds show no tokens: its client may still have left
                if self._failed is not None:
                    break
            if new is None:
                break
            if on_tokens is not None and on_tokens(new):
                s.halt = True
        if self._failed is not None:
            raise RuntimeError(f"the lane's round loop ended: {self._failed!r}")
        return {"cached": 0, "tokens": len(s.out)}

    def generate(self, prompt: list[int], count: int, sampling=None, on_tokens: Callable | None = None,
                 *, session: str = "", stop_eos: bool = True, draft: bool = True, constraint=None,
                 background: bool = False, **_ignored: Any) -> dict:
        """One request, rank 0 (App's call shape). DP4 keeps the rows on the owner rank; other
        ranks stay in the lane with empty rounds (their experts serve the a2a). Drafts/constraints
        are not in this family yet — accepted-and-ignored / refused, so the server degrades loudly
        instead of answering wrongly."""

        assert self.rank == 0, "rank 0 serves; followers run follow()"
        if constraint is not None:
            raise NotImplementedError("Mistral-Large-4 phase 2b serves no structured-output constraint yet")
        if self.concurrent:
            return self._serve(list(prompt), count, sampling, on_tokens, session, stop_eos)
        s = self.admit(list(prompt), count, sampling=sampling, session=session, on_tokens=on_tokens,
                       stop_eos=stop_eos)
        while not s.done:
            if not self.run_round():
                break
        return {"cached": 0, "tokens": len(s.out)}
