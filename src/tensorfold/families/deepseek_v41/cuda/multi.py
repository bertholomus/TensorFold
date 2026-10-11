"""DeepSeek-V4.1's concurrent decoding (``--parallel N``): up to N requests on the lane at once, every live stream's
verify window in one forward a round, each reply exactly its solo run.

- **Pool.** One cache plane a layer holds every slot (``Model.new_pool``). A stream's prompt fills its slot through
  the single-stream path on the slot's views, so its prompt bits are the solo prompt's. (v1: the whole prompt fills at
  admission, between rounds.)
- **Rounds.** Every live stream's pending token and drafts go through one forward (``rounds.RoundRunner``): each row
  at its own position and slot. Every kernel is row-invariant, the cache kernels read each row's own stream and
  selection is a total order, so a row's logits are the solo run's at that position, and each stream samples its own
  rows by its own rule: its reply is its solo reply, whatever else runs beside it. A round holds 16 rows at most
  (the decode windows' kernels): each stream's depth comes from ``DEPTH_BY`` by how many streams decode.
- **Drafts.** Each stream's DSpark drafter runs on its slot's own drafter cache (a captured graph a slot). Drafts
  only propose: they change speed, never a reply.
- **Ranks.** Rank 0 schedules (``tensorfold.cuda.scheduler``) and sends each step to the followers over a TCP link
  (``Link``) before running it; every rank checks the step's digest with one small all-gather before its model
  collectives (``OutOfStep``: the step fails on every rank at the same point). ``Watchdog``: a lost rank breaks the
  lane on every rank instead of hanging it.
- **Kept prompts** (``TF_DS_KEEP``). A finished stream's prompt stays in the window (``Kept``): its extent's rows up
  to its last kept chunk boundary, and its slot's rings at its kept boundaries (the window keys and compressor inputs
  are a slot's, not an extent's). A later prompt that starts the same continues from the longest boundary they share:
  its rows are copied into its own extent (or it grows the kept extent in place), the rings come back, and its chunks
  from there are a fresh prefill's own chunks, so its reply is the fresh reply; usage's ``cached_tokens`` is that
  boundary. Only prompt rows are kept (a verify round's rows are not a prompt chunk's bits); with replay prefill the
  boundary stays a window below the prompt's end (the decoder layers' rows there are the prompt's own). Kept entries
  give their room back, oldest first, whenever an admission needs it.

Ported from our GLM-5.3 fork's ``glm_moe_dsa/cuda/multi.py`` (the same design, pipeline/CONCURRENCY-DESIGN.md).
Structured output and logprobs are not served with ``--parallel``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import torch

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream

from ..ops import HostIds
from .model import RAW

# the most drafts a stream verifies a round by how many streams decode ("5,5,3,3": up to the drafter's block of five
# alone or beside one other, three at three or four streams: 16 rows at four), capped by the engine's --mtp-drafts
DEPTH_BY = [int(x) for x in (os.environ.get("TF_DS_PARALLEL_DEPTH") or "5,5,3,3").split(",") if x.strip()]
# while prompts fill, decoding keeps this share of the time (a prompt chunk runs once the rounds since the last one
# took share / (1 - share) of its time); a prompt whose rest fits in QUICK_ROWS tokens fills before the next round
DECODE_SHARE = float(os.environ.get("TF_DS_DECODE_SHARE") or 0.5)
QUICK_ROWS = int(os.environ.get("TF_DS_QUICK_ROWS") or 1024)
# TF_DS_ROUND_STATS=1: each round's stages synchronized and timed (drafts, Engram rows + forward, sampling, absorb),
# their averages printed every 100 rounds by streams decoding (profiling only: the syncs cost a little)
ROUND_STATS = os.environ.get("TF_DS_ROUND_STATS", "0") == "1"
# TF_DS_HOST_TRACE=1: host timestamps through each round (no syncs added): every 100 rounds, per rank, the median ms of
# each step (the follower's wait for rank 0's op included), to find host gaps between and inside rounds
HOST_TRACE = os.environ.get("TF_DS_HOST_TRACE", "0") == "1"
# TF_DS_LINK_SPIN_MS (default 3): a follower polls its step link this long before a blocking receive, so the next round's
# step is picked up without a thread wake-up (~0.1-0.6 ms on GB10 after a short sleep); 0: block at once as before
LINK_SPIN = float(os.environ.get("TF_DS_LINK_SPIN_MS") or 3.0) / 1000.0
_HT: dict = {}


def _ht(tag: str) -> None:
    if HOST_TRACE:
        _HT.setdefault("_marks", []).append((tag, time.perf_counter()))


# TF_DS_DEPTH_POLICY=conf (default; "fixed": the table depth): each stream verifies the k (<= its table depth, up to the drafter's block) with the most
# expected tokens a millisecond: expected tokens from the confidence head's prefix survivals, the round's cost from a
# fixed table of forward ms by rows (TF_DS_ROUND_MS, a pure function of state every rank holds: no live timers)
# "even": while the drafting streams started together (their rounds within TF_DS_EVEN_COHORT), the stream with the
# most rounds left (its remaining tokens over its tokens a round lately) takes the k with the most expected tokens a
# millisecond, up to the drafter's block in the rows left, and every other stream the fewest drafts (<= its table
# depth) whose expected tokens still end it before that one (by TF_DS_EVEN_MARGIN): a burst's streams end together
# and its rounds stay short. Streams that did not start together get "conf".
DEPTH_POLICY = os.environ.get("TF_DS_DEPTH_POLICY", "conf")
EVEN_MARGIN = float(os.environ.get("TF_DS_EVEN_MARGIN") or 1.1)
EVEN_COHORT = int(os.environ.get("TF_DS_EVEN_COHORT") or 4)
ROUND_MS = [float(x) for x in (os.environ.get("TF_DS_ROUND_MS") or
                               "30.0,35.2,39.9,44.9,48.6,52.1,55.6,59.1,62.7,66.3,69.6,72.9,75.5,78.2,81.3,84.4,"
                               "87.5,90.6,93.7,96.8,99.9,103.0,106.1,109.2").split(",")]
# the drafter pass in the depth choice (ms): 3.5 = its measured cost on the pair since the vocabulary-split Markov loop
# (5.4 before; set-b code at 1 stream over test-server starts: 5.4 -> 99.6, 3.5 -> 100.1-100.7, 7.5 -> 98.9)
DRAFT_MS = float(os.environ.get("TF_DS_DRAFT_MS") or 3.5)
# TF_DS_EAGER_ABSORB=1 (default): every drafting stream's window rows go into its drafter rings right after the forward
# is launched, on a side stream behind the forward (its host launches overlap the forward, not the gap between rounds),
# instead of the kept rows after sampling. Rows past the kept ones sit at positions the drafter does not read (it reads
# up to the stream's length) until a later absorb writes them, so no draft changes; the main stream waits for the side
# stream at the round's end. 0: the kept rows after sampling, as before
EAGER_ABSORB = os.environ.get("TF_DS_EAGER_ABSORB", "1") == "1"
# TF_DS_CONF_LOG=1: per draft depth, drafts verified and kept, and kept by confidence (sigmoid tenths), printed every
# 200 rounds (the DSpark acceptance report)
CONF_LOG = os.environ.get("TF_DS_CONF_LOG", "0") == "1"
STEP_TIMEOUT = float(os.environ.get("TF_DS_STEP_TIMEOUT") or 900.0)
# TF_DS_KEEP=1 (default): finished prompts stay in the window for later prompts that start the same (``Kept``), at
# most KEEP_ENTRIES of them, each with its rings at KEEP_MARKS chunk boundaries at most (the doubling ones from the first
# chunk, the prompt's last two, the ones it continued from)
KEEP = os.environ.get("TF_DS_KEEP", "1") == "1"
KEEP_ENTRIES = int(os.environ.get("TF_DS_KEEP_ENTRIES") or 8)
KEEP_MARKS = int(os.environ.get("TF_DS_KEEP_MARKS") or 10)
# TF_DS_KEEP_SHRINK=1 (default): a request that needs room cuts the oldest kept prompt back to its largest kept boundary
# that makes the room (_shrink) before it forgets one whole; prompts also keep boundaries at 3/4 and 7/8 of their
# length (and every kept boundary in their upper half), so a long kept prompt shrinks a little, not to half. A 1M kept
# prompt left 10,240 free positions and any chat beside it (the lane's 32,768 default reply) forgot all of it.
# 0: forget whole kept prompts, doubling boundaries only, as before
KEEP_SHRINK = os.environ.get("TF_DS_KEEP_SHRINK", "1") == "1"
# prompts of at least this many tokens get the 3/4 and 7/8 boundaries (one ring snapshot each, ~8 MB): the ones whose
# fill is expensive to repeat; shorter prompts keep the boundaries they had
KEEP_SHRINK_MIN = int(os.environ.get("TF_DS_KEEP_SHRINK_MIN") or 131072)
# TF_DS_SPILL_GIB=N (default 0: off; with TF_DS_KEEP): every kept prompt of TF_DS_SPILL_MIN tokens or more also goes to
# this rank's disk (spill.py: TF_DS_SPILL_DIR, N GiB at most, unused ones gone after TF_DS_SPILL_MAX_AGE_H), so a
# prompt that continues one the window no longer holds (a restart, an eviction) reads its rows back instead of filling
# them: rank 0 offers the spilled boundary when it is past every kept prompt's, and every rank must hold it


ALIGN = 2048                     # extents start and end on multiples of this many positions


class Extents:
    """First-fit extents of a pool of ``total`` positions in ALIGN steps; a freed extent merges with its neighbours."""

    def __init__(self, total: int, align: int = ALIGN) -> None:
        self.align = int(align)
        self.total = int(total) // self.align * self.align
        self.gaps: list[tuple[int, int]] = [(0, self.total)] if self.total else []

    def size(self, rows: int) -> int:
        return -(-int(rows) // self.align) * self.align

    def take(self, rows: int) -> int | None:
        """The first gap that holds ``rows`` (aligned up): its start, else None."""

        n = self.size(rows)
        for i, (a, b) in enumerate(self.gaps):
            if b - a >= n:
                self.gaps[i:i + 1] = [(a + n, b)] if b - a > n else []
                return a
        return None

    def take_at(self, start: int, rows: int) -> bool:
        """``rows`` (aligned up) from ``start`` if they are free."""

        n = self.size(rows)
        for i, (a, b) in enumerate(self.gaps):
            if a <= start and start + n <= b:
                self.gaps[i:i + 1] = [g for g in ((a, start), (start + n, b)) if g[1] > g[0]]
                return True
        return False

    def give(self, start: int, rows: int) -> None:
        merged: list[tuple[int, int]] = []
        for a, b in sorted(self.gaps + [(start, start + self.size(rows))]):
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        self.gaps = merged


class OutOfStep(RuntimeError):
    """The ranks planned a different step: it fails on every rank before its collectives."""


class Link:
    """Rank 0's steps to every follower, in order, over one TCP connection each (port through the rendezvous store);
    a follower waits on its socket between steps, never inside a collective."""

    KEY = "tensorfold/dsv41/multi/port"

    def __init__(self, store, *, rank: int, world: int, host: str) -> None:
        import socket
        from datetime import timedelta

        self.rank, self.world, self.socks = rank, world, []
        if rank == 0:
            self.server = socket.create_server(("0.0.0.0", 0))
            store.set(self.KEY, str(self.server.getsockname()[1]))
        else:
            self.server = None
            store.wait([self.KEY], timedelta(hours=24))
            sock = socket.create_connection((host or "127.0.0.1", int(store.get(self.KEY).decode())))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(rank.to_bytes(4, "big"))
            self.socks = [sock]

    def _accept(self) -> None:
        import socket

        got = {}
        while len(got) < self.world - 1:
            sock, _ = self.server.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            got[int.from_bytes(self._read(sock, 4), "big")] = sock
        self.socks = [got[r] for r in sorted(got)]

    @staticmethod
    def _read(sock, n: int) -> bytes | None:
        out = b""
        while len(out) < n:
            chunk = sock.recv(n - len(out))
            if not chunk:
                return None
            out += chunk
        return out

    def send(self, op: list) -> None:
        if not self.socks:
            self._accept()
        data = json.dumps(op).encode()
        for sock in self.socks:
            sock.sendall(len(data).to_bytes(4, "big") + data)

    def receive(self) -> list | None:
        if LINK_SPIN > 0:                            # a follower: poll for rank 0's next step a while before blocking
            import select

            sock, until = self.socks[0], time.perf_counter() + LINK_SPIN
            while not select.select([sock], [], [], 0)[0] and time.perf_counter() < until:
                pass
        head = self._read(self.socks[0], 4)
        body = None if head is None else self._read(self.socks[0], int.from_bytes(head, "big"))
        return None if body is None else json.loads(body)


class Watchdog:
    """Liveness between rank 0 and each follower (heartbeats on their own TCP connections), NCCL's asynchronous error
    and a step that never ends: any of them breaks the lane, the RDMA gathers stop waiting, NCCL aborts, and the step
    in flight fails on every rank instead of hanging."""

    KEY = "tensorfold/dsv41/multi/watch"

    def __init__(self, comm, store, *, rank: int, world: int, host: str, timeout: float = 30.0,
                 every: float = 1.0) -> None:
        import socket
        import threading
        from datetime import timedelta

        self.comm, self.rank, self.world, self.timeout, self.every = comm, rank, world, timeout, every
        self.broken: str | None = None
        self.busy: float | None = None
        self.lock = threading.Lock()
        self.socks: list = []
        if rank == 0:
            server = socket.create_server(("0.0.0.0", 0))
            store.set(self.KEY, str(server.getsockname()[1]))

            def accept() -> None:
                for _ in range(world - 1):
                    sock, _ = server.accept()
                    peer = int.from_bytes(Link._read(sock, 4) or b"\xff\xff\xff\xff", "big")
                    with self.lock:
                        self.socks.append(sock)
                    threading.Thread(target=self._listen, args=(sock, peer), daemon=True).start()

            threading.Thread(target=accept, daemon=True).start()
        else:
            store.wait([self.KEY], timedelta(hours=24))
            sock = socket.create_connection((host or "127.0.0.1", int(store.get(self.KEY).decode())))
            sock.sendall(rank.to_bytes(4, "big"))
            self.socks = [sock]
            threading.Thread(target=self._listen, args=(sock, 0), daemon=True).start()
        threading.Thread(target=self._beat, daemon=True).start()
        threading.Thread(target=self._nccl, daemon=True).start()

    def _beat(self) -> None:
        while self.broken is None:
            with self.lock:
                socks = list(self.socks)
            for sock in socks:
                try:
                    sock.sendall(b"H")
                except OSError:
                    pass
            time.sleep(self.every)

    def _listen(self, sock, peer: int) -> None:
        sock.settimeout(self.timeout)
        while self.broken is None:
            try:
                got = sock.recv(64)
            except OSError:
                got = b""
            if not got:
                self.abort(f"rank {peer} stopped answering" if self.rank == 0 else "rank 0 stopped answering")
                return
            if b"A" in got:
                self.abort("rank 0 lost a rank")
                return

    def _nccl(self) -> None:
        nccl = getattr(self.comm, "nccl", self.comm)
        check = getattr(nccl, "async_error", None)
        while self.broken is None:
            time.sleep(self.every)
            code = check() if check is not None else 0
            if code > 0:
                self.abort(f"NCCL error {code}")
                return
            busy = self.busy
            if busy is not None and time.monotonic() - busy > STEP_TIMEOUT:
                self.abort(f"a step ran past {STEP_TIMEOUT:.0f} s on rank {self.rank}")
                return

    def abort(self, why: str) -> None:
        with self.lock:
            if self.broken is not None:
                return
            self.broken = why
            socks = list(self.socks)
        print(f"[tensorfold] rank {self.rank}: the lane is broken ({why}); every step fails from now on", flush=True)
        rdma = getattr(self.comm, "rdma", None)
        if rdma is not None:
            rdma.abort(why)
        nccl = getattr(self.comm, "nccl", self.comm)
        if hasattr(nccl, "abort"):
            nccl.abort()
        if self.rank == 0:
            for sock in socks:
                try:
                    sock.sendall(b"A")
                except OSError:
                    pass


def _pack(sampling) -> list | None:
    if sampling is None:
        return None
    return [int(sampling.seed), float(sampling.temperature), int(sampling.top_k), float(sampling.top_p),
            float(sampling.min_p)]


def _unpack(values):
    from tensorfold.engine.exact_sampling import Sampling

    return None if values is None else Sampling(values[0], values[1], values[2], values[3], values[4])


class Slot:
    """A stream's slot: its index (its window ring, compressor inputs and drafter rings) and, while a stream holds it,
    that stream's SeqCache over its extent of the pool."""

    def __init__(self, index: int, dc) -> None:
        self.index, self.sc, self.dc = index, None, dc


class Kept:
    """A finished stream's prompt kept in the window: positions [base, base + size) hold the compressed and indexer
    rows (and token ids) of its first ``top`` prompt tokens; ``snaps`` its slot's rings and compressor inputs at each
    kept chunk boundary; ``host`` the prompt's host ids to ``top`` (Engram's n-grams); ``keys`` (rank 0 only) the
    prompt's ids to ``top`` with each image span's positions keyed by its picture; ``tick`` when it was last used."""

    def __init__(self, eid: int, base: int, size: int, top: int, host: list, snaps: dict, replay: bool, keys,
                 tick: int) -> None:
        self.eid, self.base, self.size, self.top, self.host = eid, base, size, top, host
        self.snaps, self.replay, self.keys, self.tick, self.hits = snaps, replay, keys, tick, 0
        self.disk: dict | None = None                    # its spilled manifest (TF_DS_SPILL_GIB), when written


def _picture_key(pic) -> int:
    """A picture's 62-bit key (its patches and grids), kept on the picture."""

    key = getattr(pic, "_tf_key", None)
    if key is None:
        h = hashlib.sha256(repr((pic.n_vit_h, pic.n_vit_w, pic.n_llm_h, pic.n_llm_w)).encode())
        h.update(pic.patches.contiguous().view(torch.int16).numpy().tobytes())
        key = int.from_bytes(h.digest()[:8], "big") >> 2
        pic._tf_key = key
    return key


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``cap`` cache rows."""

    def __init__(self, engine, *, slots: int, cap: int) -> None:
        from .rounds import MAX_ROWS, RoundRunner

        self.e = engine
        m = engine.model
        self.m, self.cap = m, int(cap)
        self.pool = m.new_pool(slots, self.cap)
        d = engine.drafter
        self.dpool = None
        if d is not None:
            from .dspark import DraftPool

            self.dpool = DraftPool(d, slots)             # every slot's drafter rings in one plane a stage
        self.slots = [Slot(i, self.dpool.views[i] if d is not None else None) for i in range(slots)]
        self.extents = Extents(self.cap)              # every stream takes an extent of the one window
        self._table_view = m.pool_view(self.pool, 0, 0, self.extents.total)   # RoPE tables for any position
        self.drafters: dict[int, object] = {}            # batched drafter graphs by drafting streams
        self.free = list(range(slots))
        self.max_rows = MAX_ROWS
        if not engine.drafts + 1 <= MAX_ROWS < 64:     # a solo verify window must take the decode kernels, a round too
            raise ValueError(f"TF_DS_DECODE_ROWS={MAX_ROWS} must hold a verify window ({engine.drafts + 1} rows) "
                             "and stay under the 64-row prompt kernels")
        self.depth_most = engine.drafts
        graphs = engine.runner is not None
        self.runner = RoundRunner(m, self.pool, graphs=graphs,
                                  graph_pool=engine.runner.pool if graphs and engine.runner.pool is not None else None)
        self.eos = tuple(engine.eos)
        self.streams: dict[int, Stream] = {}           # decoding (and just ended) streams
        self.filling: list[Stream] = []                 # admitted, their prompts filling their slots (oldest first)
        self.chunk_s, self.since_fill = 0.0, 0.0        # the last prompt chunk's time; decoding since then
        self.next_id = 0
        self.link: Link | None = None
        self.watch: Watchdog | None = None
        self.rounds = 0
        self.round_log: list[tuple[int, int, float, int]] = []
        self.stage = {}                                 # ROUND_STATS: streams -> [rounds, rows, tokens, seconds a stage]
        self.conf_depth = [[0, 0] for _ in range(8)]     # CONF_LOG: (drafts reached, kept) by depth
        self.conf_bins = [[0, 0] for _ in range(10)]     # CONF_LOG: (drafts, kept) by sigmoid(confidence) tenths
        self.conf_rounds = 0
        self.k_hist: dict[int, list[int]] = {}           # CONF_LOG: drafts verified a stream a round, by streams live
        self.kept: dict[int, Kept] = {}                  # kept prompts by id
        self.next_kept, self.ticks = 0, 0
        self.keep_stats: dict[str, int] = {}             # admissions that continued a kept prompt, by placement
        self.spill = None                                # TF_DS_SPILL_GIB: kept prompts on this rank's disk
        self.spill_nonce = os.urandom(4).hex() if engine.rank == 0 else None   # names this run's spilled prompts
        if KEEP:
            from . import spill as sp

            if sp.SPILL_GIB > 0:
                try:
                    ratios = {i: m.cfg.compress_ratios[i] for i in set(self.pool.comp) | set(self.pool.index_k)}
                    ident = sp.engine_id(getattr(engine, "model_dir", "."), world=engine.world, pool=self.pool,
                                         dpool=self.dpool)
                    self.spill = sp.Spill(sp.SPILL_DIR, engine=ident, rank=engine.rank, pool=self.pool, ratios=ratios,
                                          budget=int(sp.SPILL_GIB * 2**30), max_age=sp.SPILL_MAX_AGE_H * 3600.0)
                    self.spill.recycle()
                    if engine.rank == 0:
                        st = self.spill.stats()
                        print(f"[tensorfold] spilled kept prompts: {self.spill.dir} ({st['manifests']} on disk, "
                              f"{st['bytes'] / 2**30:.2f} of {sp.SPILL_GIB:g} GiB)", flush=True)
                except Exception as exc:                 # noqa: BLE001  (no disk: kept prompts stay in the window only)
                    print(f"[tensorfold] rank {engine.rank}: spilled kept prompts off ({type(exc).__name__}: {exc})",
                          flush=True)
                    self.spill = None

    def warm(self, buckets=(1024, 2048, 4096, 8192)) -> None:
        """Before serving, on every rank in the same order: round graphs for 1 .. 16 rows at the small context
        buckets (synthetic rows on slot 0, which a request rewrites before it reads them) and each slot's drafter
        graph."""

        t0 = time.perf_counter()
        n = 0
        for b in buckets:
            if b > self.cap:
                break
            host = [1000 + (i * 7919) % 60000 for i in range(b)]
            for R in range(self.max_rows, 0, -1):
                if (R, self.runner.bucket(b)) in (self.runner.graphs or {}):
                    continue
                self.runner.forward([(0, 0, self.extents.total, b - R, host[b - R:b], host)], replay=False)
                n += 1
        if self.e.drafter is not None:
            for k in range(1, len(self.slots) + 1):
                self._drafts([0] * k, [1] * k, list(range(k)))
        torch.cuda.synchronize()
        if self.e.rank == 0:
            print(f"[tensorfold] --parallel warm-up: {n} round graphs, {len(self.drafters)} drafter graphs, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    def _drafts(self, tokens: list[int], q0: list[int], slots: list[int]) -> list[list[int]]:
        """Every drafting stream's drafts from one batched drafter pass (a graph a stream count)."""

        from .dspark import BatchDraftGraph

        N = len(tokens)
        g = self.drafters.get(N)
        if g is None:
            # as deep as any round with N drafting streams may verify ("even": the whole block)
            steps = None if DEPTH_POLICY == "even" else max(self._depth(L) for L in range(N, len(self.slots) + 1))
            g = BatchDraftGraph(self.e.drafter, self.m.pool_view(self.pool, 0, 0, self.extents.total), self.dpool, N,
                                steps=steps)
            g.tokens.copy_(torch.tensor(tokens, dtype=torch.long))
            g.q0.copy_(torch.tensor(q0, dtype=torch.long))
            g.slots.copy_(torch.tensor(slots, dtype=torch.long))
            if self.e.runner is not None:
                if self.e.runner.pool is None:
                    self.e.runner.pool = torch.cuda.graph_pool_handle()
                g.capture(self.e.runner.pool)
            self.drafters[N] = g
        _ht("drafter run")
        return g.run(tokens, q0, slots)

    # -- the scheduler's interface --------------------------------------------------------------------------------
    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def _send(self, op: list) -> None:
        self._alive()
        if self.link is not None:
            self.link.send(op)

    def _step(self, busy: bool) -> None:
        if self.watch is not None:
            self.watch.busy = time.monotonic() if busy else None

    def _broken(self, exc: BaseException) -> None:
        """A step that raised past its digest check (out of memory under the allocator ceiling, a bug) may have left
        another rank inside a collective: the lane breaks on every rank (the watchdog restarts it) instead of the next
        step's collectives pairing with the wrong ones. OutOfStep and NoRoom happen on every rank at the same point."""

        if self.watch is not None and not isinstance(exc, (OutOfStep, NoRoom)):
            self.watch.abort(f"a step failed on rank {self.e.rank} ({type(exc).__name__}: {str(exc)[:160]})")

    def _alive(self) -> None:
        if self.watch is not None and self.watch.broken is not None:
            raise RuntimeError(f"this lane lost a rank ({self.watch.broken}); restart it on every rank")

    def _agree(self, what: str, plan: list) -> None:
        """One small all-gather checks the step on every rank before any of its model collectives."""

        e = self.e
        if e.world < 2:
            return
        dgst = int.from_bytes(hashlib.sha256(json.dumps(plan).encode()).digest()[:8], "big")
        # 16-bit pieces as fp32 (exact): the model's gather, so a round's check rides the RDMA path (~0.1 ms, not ~0.9)
        mine = torch.tensor([(dgst >> b) & 0xFFFF for b in (0, 16, 32, 48)], dtype=torch.float32, device="cuda")
        every = self.m.comm.gather(mine).tolist()
        if any(row != every[0] for row in every):
            raise OutOfStep(f"the ranks planned different {what}s; its requests fail, serving goes on")

    def _all(self, ok: bool) -> bool:
        """Whether ``ok`` holds on every rank (one small gather: every rank calls it at the same point of a step)."""

        if self.e.world < 2:
            return bool(ok)
        mine = torch.tensor([1.0 if ok else 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        return all(row[0] == 1.0 for row in self.m.comm.gather(mine).tolist())

    def _shape(self) -> list:
        return [self.next_id, list(self.free), list(self.extents.gaps),
                [[s.sid, s.st.index, s.st.sc.length, len(s.out), bool(s.done)] for s in self.streams.values()],
                [[s.sid, s.st.index, s.filled] for s in self.filling],
                [[k.eid, k.base, k.size, k.top, k.tick, sorted(k.snaps)] for k in self.kept.values()]]

    # -- kept prompts ---------------------------------------------------------------------------------------------
    def _replay(self) -> bool:
        from . import engine as eng

        return bool(getattr(self.e, "replay_mode", eng.REPLAY))

    def _keys(self, s: Stream):
        """Rank 0: the prompt's ids, each image span's positions keyed by its picture instead (int64)."""

        import numpy as np

        keys = np.asarray(s.prompt, dtype=np.int64).copy()
        for start, pic in (s.vision.spans if s.vision is not None and getattr(s.vision, "spans", None) else []):
            keys[start:start + pic.tokens] = -2 - _picture_key(pic)
        return keys

    def _match(self, s: Stream, keys) -> list | None:
        """Rank 0: [kept id, boundary] for the kept prompt this one continues the furthest (the most recent of
        equals), else None. The boundary is one of its kept ones, within the prompts' shared start, and below the
        prompt's last row (with replay prefill, a window below)."""

        import numpy as np

        replay = self._replay()
        limit = len(s.prompt) - (self.m.cfg.window if replay else 1)
        best = None
        for k in self.kept.values():
            if k.replay != replay or k.keys is None:
                continue
            n = min(len(k.keys), len(keys))
            diff = np.flatnonzero(k.keys[:n] != keys[:n])
            common = int(diff[0]) if diff.size else n
            cut = max((b for b in k.snaps if b <= min(common, limit)), default=0)
            if cut and (best is None or (cut, k.tick) > (best[1], best[0].tick)):
                best = (k, cut)
        return None if best is None else [best[0].eid, best[1]]

    def _marks(self, n: int) -> set[int]:
        """The chunk boundaries a prompt of n tokens keeps its rings at: doubling from the first, and its last two."""

        from .engine import PREFILL_CHUNK as C

        marks = set(range(C, n + 1, C)[-2:])
        b = C
        while b <= n:
            marks.add(b)
            b *= 2
        if KEEP_SHRINK and n >= KEEP_SHRINK_MIN:     # cut points near a long prompt's end (_shrink)
            for num, den in ((3, 4), (7, 8)):
                b = n * num // den // C * C
                if b >= C:
                    marks.add(b)
        return marks

    def _snapshot(self, index: int):
        """Slot ``index``'s window rings (with replay prefill the encoder layers' only: a resumed prompt's decoder
        layers read no window key before its replay row) and compressor inputs, copied."""

        RS, n = self.pool.ring_size, len(self.pool.ring)
        layers = range(self.m.cfg.n_layers // 2) if self._replay() else range(n)
        ring = torch.stack([self.pool.ring[i][index * RS:(index + 1) * RS] for i in layers])
        raw = [torch.stack([x[index * RAW:(index + 1) * RAW] for x in pair])
               for _, pair in sorted(self.pool.comp_raw.items())]
        return ring, (torch.stack(raw) if raw else None)

    def _restore(self, index: int, snap) -> None:
        ring, raw = snap
        RS = self.pool.ring_size
        for i in range(ring.shape[0]):
            self.pool.ring[i][index * RS:(index + 1) * RS].copy_(ring[i])
        if raw is not None:
            for j, (_, pair) in enumerate(sorted(self.pool.comp_raw.items())):
                for h, x in enumerate(pair):
                    x[index * RAW:(index + 1) * RAW].copy_(raw[j, h])

    def _copy_rows(self, src: int, dst: int, n: int) -> None:
        """Positions [src, src + n) of the window's compressed, indexer and token rows to [dst, dst + n) (through a
        copy where the two overlap)."""

        if src == dst:
            return
        c = self.m.cfg
        overlap = src < dst + n and dst < src + n
        for planes in (self.pool.comp, self.pool.index_k):
            for i, x in planes.items():
                r = c.compress_ratios[i]
                a, b, k = src // r, dst // r, n // r
                for t in (x if isinstance(x, tuple) else (x,)):
                    t[b:b + k].copy_(t[a:a + k].clone() if overlap else t[a:a + k])
        t = self.pool.tokens
        t[dst:dst + n].copy_(t[src:src + n].clone() if overlap else t[src:src + n])

    def _forget(self, k: Kept, give: bool = True) -> None:
        self.kept.pop(k.eid, None)
        if give:
            self.extents.give(k.base, k.size)

    def _room(self, need: int) -> bool:
        """Rank 0: whether ``need`` positions fit once every kept prompt gave its room back."""

        ex = Extents(0, self.extents.align)
        ex.total, ex.gaps = self.extents.total, list(self.extents.gaps)
        for k in self.kept.values():
            ex.give(k.base, k.size)
        return any(b - a >= ex.size(need) for a, b in ex.gaps)

    def _shrink(self, k: Kept, need: int) -> bool:
        """Cut kept prompt ``k`` back to its largest kept boundary below its top whose freed tail makes room for
        ``need`` positions: its rows, snapshots, keys and host ids up to that boundary stay (a later prompt resumes
        there, as from any kept boundary), the rest of its extent goes back. False when no boundary makes the room.
        A function of state every rank holds (extents, boundaries), so every rank cuts the same."""

        for m in sorted((b for b in k.snaps if b < k.top), reverse=True):
            size = self.extents.size(m)
            if size >= k.size:
                continue
            trial = Extents(0, self.extents.align)
            trial.total, trial.gaps = self.extents.total, list(self.extents.gaps)
            trial.give(k.base + size, k.size - size)
            if any(b - a >= trial.size(need) for a, b in trial.gaps):
                self.extents.give(k.base + size, k.size - size)
                k.size, k.top = size, m
                k.snaps = {b: v for b, v in k.snaps.items() if b <= m}
                k.keys = None if k.keys is None else k.keys[:m].copy()
                k.host = k.host.copy(m)
                self.keep_stats["shrunk"] = self.keep_stats.get("shrunk", 0) + 1
                return True
        return False

    def _place(self, need: int, src: Kept | None) -> tuple[int | None, str]:
        """An extent for ``need`` positions (every rank the same): a free one (``src``'s rows get copied in, "copy"),
        else ``src``'s own extent grown in place ("here"), else after the oldest other kept prompts give theirs back;
        last, anywhere once ``src`` gave its own back too (its rows move, "move"). Without ``src``: "fresh"."""

        while True:
            base = self.extents.take(need)
            if base is not None:
                return base, "copy" if src is not None else "fresh"
            if src is not None:
                self.extents.give(src.base, src.size)
                if self.extents.take_at(src.base, need):
                    self._forget(src, give=False)
                    return src.base, "here"
                if not self.extents.take_at(src.base, src.size):
                    raise RuntimeError("a kept prompt's extent was not free to take back")
            others = sorted((k for k in self.kept.values() if k is not src), key=lambda k: k.tick)
            if others:
                if not (KEEP_SHRINK and self._shrink(others[0], need)):
                    self._forget(others[0])
                continue
            if src is None:
                return None, "fresh"
            self._forget(src)
            base = self.extents.take(need)
            return base, "move" if base is not None else "fresh"

    @staticmethod
    def _kept_marks(snaps: dict) -> list[int]:
        """The boundaries a finished stream keeps, KEEP_MARKS at most: the doubling ones, its last two, then the
        earliest."""

        from .engine import PREFILL_CHUNK as C

        marks = sorted(snaps)
        if len(marks) > KEEP_MARKS:
            must = {b for b in marks if (b // C) & (b // C - 1) == 0} | set(marks[-2:])
            if KEEP_SHRINK:
                must |= {b for b in marks if 2 * b > marks[-1]}       # the upper half's cut points (_shrink)
            rest = [b for b in marks if b not in must]
            marks = sorted(must | set(rest[:max(0, KEEP_MARKS - len(must))]))
        return marks

    def _keep(self, s: Stream) -> None:
        """A finished stream's prompt into ``kept``: its extent's rows to its last kept boundary stay, the rest goes
        back."""

        marks = self._kept_marks(s.snaps)
        top = marks[-1]
        size = self.extents.size(top)
        self.free.append(s.st.index)
        self.free.sort()
        if s.size > size:
            self.extents.give(s.base + size, s.size - size)
        keys = getattr(s, "keys", None)
        k = Kept(self.next_kept, s.base, size, top, s.st.sc.host.copy(top), {b: s.snaps[b] for b in marks},
                 self._replay(), None if keys is None else keys[:top], self.ticks)
        self.kept[k.eid] = k
        self.next_kept += 1
        if self.spill is not None:                       # write-through: its rows past what its lineage holds
            k.disk = self.spill.persist(f"{self.spill_nonce or 'run'}-{k.eid:06d}", base=k.base, top=k.top,
                                        replay=k.replay, snaps=k.snaps, host=k.host,
                                        lineage=getattr(s, "lineage", None), keys=k.keys)
        s.st.sc = None
        s.snaps = None

    def _covered(self, done: list[Stream]) -> list[int]:
        """Rank 0: the kept prompts the finishing streams' prompts will cover (the same ids to their top, every kept
        boundary of theirs kept again)."""

        import numpy as np

        drops: list[int] = []
        for s in done:
            snaps, keys = getattr(s, "snaps", None), getattr(s, "keys", None)
            if not snaps or keys is None or s.sid not in self.streams:
                continue
            marks = set(self._kept_marks(snaps))
            top = max(marks)
            for k in self.kept.values():
                if (k.eid not in drops and k.keys is not None and k.top <= top and k.replay == self._replay()
                        and set(k.snaps) <= marks and np.array_equal(k.keys, keys[:k.top])):
                    drops.append(k.eid)
        return drops

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """A request into the lowest free slot: its prompt fills the slot now, then it decodes in the rounds."""

        if s.constraint is not None or s.probabilities is not None:
            raise ValueError("structured output and logprobs are not served with --parallel on DeepSeek-V4.1")
        room = self.extents.total - len(s.prompt) - self.max_rows - 8 - 4
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.extents.total}-token "
                             "context")
        s.count = max(1, min(s.count, room))
        if not self.free:
            raise NoRoom("every stream slot is busy")
        if not self._room(self._need(s)):
            raise NoRoom("the window's free extents are too small for this request now")
        positions = s.vision.positions() if s.vision is not None and getattr(s.vision, "spans", None) else []
        index = self.free[0]
        s.keys = self._keys(s) if KEEP else None
        reuse = self._match(s, s.keys) if KEEP and self.kept else None
        if self.spill is not None and s.keys is not None:   # a spilled boundary past every kept prompt's
            far = self.spill.best(s.keys, len(s.prompt), window=self.m.cfg.window, replay=self._replay(),
                                  above=int(reuse[1]) if reuse else 0)
            if far is not None:
                reuse = ["disk", far[0], far[1]]
        self._send(["admit", list(s.prompt), s.count, _pack(s.sampling), bool(s.draft), bool(s.stop_eos), index,
                    positions, reuse])
        self._admit(s, index, positions, reuse)

    def _need(self, s: Stream) -> int:
        """A stream's positions: its prompt, its reply, a round's rows and its extent's scratch row (a group long)."""

        return len(s.prompt) + s.count + self.max_rows + 8 + 4

    def _admit(self, s: Stream, index: int, positions: list[int], reuse: list | None = None) -> None:
        """Its slot, its extent, its image rows (shared from rank 0) and its prompt's chunk steps; ``_fill`` runs
        them. ``reuse`` [kept id, boundary]: the prompt continues that kept prompt from the boundary; ["disk", name,
        boundary]: it continues a spilled one (TF_DS_SPILL_GIB), its rows read into a fresh extent, or it fills fresh
        when a rank cannot read them (every rank agrees first)."""

        e = self.e
        self._step(True)
        try:
            self._agree("admission", [self._shape(), list(s.prompt), s.count, _pack(s.sampling), bool(s.draft),
                                      bool(s.stop_eos), index, positions, reuse])
            self.ticks += 1
            need = self._need(s)
            disk = bool(reuse) and reuse[0] == "disk"
            if disk and not self._all(self.spill is not None and self.spill.check(str(reuse[1]), int(reuse[2]))):
                if e.rank == 0:
                    print(f"[tensorfold] spilled prompt {reuse[1]}: not on every rank, the prompt fills fresh",
                          flush=True)
                reuse, disk = None, False
            src = self.kept.get(int(reuse[0])) if reuse and not disk else None
            if reuse and not disk and src is None:
                raise OutOfStep("an admission continues a kept prompt this rank does not hold")
            base, how = self._place(need, src)
            if base is None:                             # (every rank at the same point: the same extents)
                raise NoRoom("the window's free extents are too small for this request now")
            slot = self.slots[index]
            self.free.remove(index)
            s.base, s.size = base, self.extents.size(need)
            slot.sc = self.m.pool_view(self.pool, index, base, s.size)
            s.sid, s.st = self.next_id, slot
            self.next_id += 1
            cut, s.snaps, s.lineage = 0, {}, None
            if disk:
                name, at, t0 = str(reuse[1]), int(reuse[2]), time.perf_counter()
                try:
                    snaps, host, lineage = self.spill.load(name, at, base)
                    loaded = at in snaps
                except Exception as exc:                 # noqa: BLE001  (every rank agrees on the outcome below)
                    print(f"[tensorfold] rank {e.rank}: spilled prompt {name} did not load ({type(exc).__name__}: "
                          f"{exc})", flush=True)
                    loaded = False
                if self._all(loaded):                    # else the rows written so far get filled over, fresh
                    cut, how = at, "disk"
                    self._restore(index, snaps[cut])
                    slot.sc.host = HostIds(host)
                    s.snaps, s.lineage = snaps, lineage
                    self.keep_stats[how] = self.keep_stats.get(how, 0) + 1
                    if e.rank == 0:
                        print(f"[tensorfold] spilled prompt {name}: continued at {cut} of {len(s.prompt)} (disk, "
                              f"{time.perf_counter() - t0:.2f}s)", flush=True)
                elif e.rank == 0:
                    print(f"[tensorfold] spilled prompt {name}: a rank could not read it, the prompt fills fresh",
                          flush=True)
            elif how != "fresh":
                cut = int(reuse[1])
                if how in ("copy", "move"):
                    self._copy_rows(src.base, base, cut)
                if how == "copy":
                    src.tick, src.hits = self.ticks, src.hits + 1
                self.keep_stats[how] = self.keep_stats.get(how, 0) + 1
                if e.rank == 0:
                    print(f"[tensorfold] kept prompt {src.eid}: continued at {cut} of {len(s.prompt)} ({how})",
                          flush=True)
                self._restore(index, src.snaps[cut])
                slot.sc.host = src.host.copy(cut)
                s.snaps = {b: v for b, v in src.snaps.items() if b <= cut}
                if self.spill is not None:               # its spilled rows to the boundary are this prompt's too
                    s.lineage = self.spill.lineage(src.disk, cut)
            image = None
            later = [p for p in positions if p >= cut]
            if later:
                rows = None
                if e.rank == 0:                          # the tower runs only for the spans past the boundary
                    rows = torch.cat([e.tower.span_rows(pic)[max(0, cut - start):] for start, pic in s.vision.spans
                                      if start + pic.tokens > cut])
                image = (later, e._share_rows(rows, len(later)))
            s.draft = bool(s.draft) and e.drafter is not None
            snap = None
            if KEEP:
                marks = self._marks(len(s.prompt))

                def snap(end: int) -> None:
                    if end in marks:
                        s.snaps[end] = self._snapshot(index)

            s.steps = e.prefill_steps(slot.sc, slot.dc if s.draft else None, list(s.prompt), image, start=cut,
                                      snap=snap)
            s.filled, s.prefill_s, s.cached = cut, 0.0, cut
            self.filling.append(s)
        except Exception as exc:
            self._broken(exc)
            raise
        finally:
            self._step(False)

    def _quick(self, s: Stream) -> bool:
        return len(s.prompt) - s.filled <= QUICK_ROWS

    def _fill(self, s: Stream) -> list[Stream]:
        """One chunk of a filling stream's prompt; at its last one its first token, and it decodes from the next
        round. Returns it if it ended there."""

        e = self.e
        self._step(True)
        try:
            self._agree("fill", [self._shape(), s.sid])
            t0 = time.perf_counter()
            try:
                s.filled = next(s.steps)
                last = None
            except StopIteration as end:
                last, s.filled = end.value, len(s.prompt)
            dt = time.perf_counter() - t0
            s.prefill_s += dt
            self.chunk_s, self.since_fill = dt, 0.0
            self.round_end = None                      # (ROUND_STATS: a fill is not host time between rounds)
            if last is None:
                return []
            self.filling.remove(s)
            s.steps = None
            first = e._sample(last, [len(s.prompt)], s.sampling)[0]
            s.started = time.perf_counter()
            s.pending = first
            self.streams[s.sid] = s
            s.take([first], self._ends(s))
            return [s] if s.done else []
        except Exception as exc:
            self._broken(exc)
            raise
        finally:
            self._step(False)

    def _absorb_eager(self, items: list):
        """absorb_many of ``items`` on the side stream once the forward (on the current stream) has run; returns the
        event that marks it done."""

        main = torch.cuda.current_stream()
        side = getattr(self, "_absorb_stream", None)
        if side is None:
            side = self._absorb_stream = torch.cuda.Stream()
        ready = torch.cuda.Event()
        ready.record(main)
        side.wait_event(ready)
        with torch.cuda.stream(side):
            self.e.drafter.absorb_many(self.dpool, self.m.pool_view(self.pool, 0, 0, self.extents.total) if
                                       self._table_view is None else self._table_view, items, eager=True)
            done = torch.cuda.Event()
            done.record(side)
        return done

    def _choose_k(self, conf: list[float], kmax: int, live: int) -> int:
        """The k in 0 .. kmax with the most expected tokens a millisecond: 1 + the prefix survivals' sum over the
        round's table cost (this stream's rows beside the other streams' at their table depth)."""

        import math

        others = (live - 1) * (self._depth(live) + 1)
        best, best_v, surv, exp_tokens = 0, -1.0, 1.0, 1.0
        for kk in range(0, kmax + 1):
            if kk > 0:
                surv *= 1.0 / (1.0 + math.exp(-float(conf[kk - 1])))
                exp_tokens += surv
            rows = min(len(ROUND_MS), others + kk + 1)
            v = exp_tokens * live / (ROUND_MS[rows - 1] + DRAFT_MS)
            if v > best_v:
                best, best_v = kk, v
        return best

    def _even_k(self, live: list[Stream], drafting: list[Stream], confs: list, k: int) -> dict:
        """TF_DS_DEPTH_POLICY=even: each drafting stream's k (a pure function of state every rank holds)."""

        import math

        def expected(cf: list[float], n: int) -> float:
            surv, e = 1.0, 1.0
            for j in range(n):
                surv *= 1.0 / (1.0 + math.exp(-float(cf[j])))
                e += surv
            return e

        left = [(s.count - len(s.out)) / max(1.0, getattr(s, "per_round", 2.0)) for s in drafting]
        c = max(range(len(drafting)), key=lambda i: (left[i], -i))
        ks, rows = {}, len(live)
        for i, (s, cf) in enumerate(zip(drafting, confs)):
            if i == c:
                continue
            need = EVEN_MARGIN * (s.count - len(s.out)) / max(1.0, left[c])
            kk, top = 0, min(k, s.count - len(s.out), len(cf))
            while kk < top and expected(cf, kk) < need:
                kk += 1
            ks[s.sid] = kk
            rows += kk
        s, cf = drafting[c], confs[c]
        top = max(0, min(self.depth_most, s.count - len(s.out), len(cf), self.max_rows - rows))
        best, best_v = 0, -1.0
        for kk in range(top + 1):
            v = expected(cf, kk) / (ROUND_MS[min(len(ROUND_MS), rows + kk) - 1] + DRAFT_MS)
            if v > best_v:
                best, best_v = kk, v
        ks[s.sid] = best
        return ks

    def _depth(self, live: int) -> int:
        k = DEPTH_BY[min(live, len(DEPTH_BY)) - 1] if DEPTH_BY else self.depth_most
        return max(0, min(k, self.depth_most, self.max_rows // max(live, 1) - 1))

    @torch.no_grad()
    def round(self, told: list | None = None) -> list[Stream]:
        """One round over every live stream; returns the streams that ended."""

        self._t_enter = time.perf_counter()
        live = [s for s in self.streams.values() if not s.done]
        if told is None and self.filling:
            s = next((f for f in self.filling if self._quick(f)), self.filling[0])
            owed = self.chunk_s * DECODE_SHARE / max(1e-6, 1.0 - DECODE_SHARE)
            if not live or self._quick(s) or self.since_fill >= owed:
                self._send(["fill", s.sid])
                return self._fill(s)
        if not live:
            return []
        k = self._depth(len(live))
        if told is None:
            self._send(["round", k])
        elif told[1] != k:
            raise OutOfStep("the ranks chose different draft depths")
        t0 = time.perf_counter()
        done = self._round(live, k)
        self.since_fill += time.perf_counter() - t0
        return done

    def _round(self, live: list[Stream], k: int) -> list[Stream]:
        e = self.e
        t0 = time.perf_counter()
        self._step(True)
        try:
            ta0 = time.perf_counter()
            _ht("round start")
            self._agree("round", [self._shape(), k])
            _ht("agreed")
            if ROUND_STATS:
                torch.cuda.synchronize()
            t_agree = time.perf_counter() - ta0
            between = ta0 - self.round_end if getattr(self, "round_end", None) else 0.0
            pre = ta0 - getattr(self, "_t_enter", ta0)                # inside round() before the step (op send)
            windows, kept = [], []
            marks = [time.perf_counter()]

            def mark():
                if ROUND_STATS:
                    torch.cuda.synchronize()
                    marks.append(time.perf_counter())

            want = {s.sid: (min(k, s.count - len(s.out)) if s.draft else 0) for s in live}
            drafting = [s for s in live if want[s.sid] > 0]
            if drafting:                                 # the rows known before the drafts: their Engram rows now
                self.runner.engram_touch([[*s.st.sc.host[max(0, s.st.sc.length - 8):s.st.sc.length].tolist(),
                                           s.pending] for s in live])
            proposed, confs = {}, {}
            _ht("touched")
            if drafting:
                rows = self._drafts([s.pending for s in drafting], [s.st.sc.length for s in drafting],
                                    [s.st.index for s in drafting])
                conf_rows = self.drafters[len(drafting)].last_conf
                even = None
                if DEPTH_POLICY == "even" and len(drafting) > 1 and \
                        max(s.rounds for s in drafting) - min(s.rounds for s in drafting) <= EVEN_COHORT:
                    even = self._even_k(live, drafting, conf_rows, k)
                for s, r, cf in zip(drafting, rows, conf_rows):
                    kk = want[s.sid]
                    if even is not None:
                        kk = even[s.sid]
                    elif DEPTH_POLICY in ("conf", "even"):
                        kk = self._choose_k(cf, min(kk, len(cf)), len(live))
                    if CONF_LOG:
                        self.k_hist.setdefault(len(live), [0] * 8)[kk] += 1
                    proposed[s.sid] = r[:kk]
                    confs[s.sid] = cf
            mark()
            for s in live:
                sc = s.st.sc
                P = sc.length
                drafts: list[int] = proposed.get(s.sid, [])
                window = [s.pending] + drafts
                sc.host.set(P, window)
                windows.append((s.st.index, s.base, s.size, P, window, sc.host.view()))
                kept.append((s, P, drafts))
            _ht("drafts back + plan")
            logits, taps = self.runner.forward(windows)
            _ht("forward launched")
            absorb_done = None
            if EAGER_ABSORB and taps is not None:
                eager, r = [], 0
                for (s, P, _), (_, _, _, _, window, _) in zip(kept, windows):
                    if s.draft:
                        eager.append((s.st.index, taps[r:r + len(window)], P))
                    r += len(window)
                if eager:
                    absorb_done = self._absorb_eager(eager)
            mark()
            done, r0, tokens = [], 0, 0
            t_absorb = 0.0
            absorbs = []
            for (s, P, drafts), (_, _, _, _, window, _) in zip(kept, windows):
                n = len(window)
                target = e._sample(logits[r0:r0 + n], [P + 1 + i for i in range(n)], s.sampling)
                a = 0
                while a < len(drafts) and drafts[a] == target[a]:
                    a += 1
                if CONF_LOG and drafts:
                    import math

                    cf = confs.get(s.sid, [])
                    for j in range(min(len(drafts), a + 1)):          # drafts reached: the first miss included
                        self.conf_depth[j][0] += 1
                        self.conf_depth[j][1] += int(j < a)
                        if j < len(cf):
                            b = min(9, int(10 / (1 + math.exp(-cf[j]))))
                            self.conf_bins[b][0] += 1
                            self.conf_bins[b][1] += int(j < a)
                new = drafts[:a] + [target[a]]
                s.st.sc.length = P + a + 1
                if s.draft and taps is not None:
                    if absorb_done is not None:
                        self.dpool.views[s.st.index].absorbed = P + a + 1
                    else:
                        absorbs.append((s.st.index, taps[r0:r0 + a + 1], P))
                s.counted(n)
                ends = self._ends(s)
                for i, t in enumerate(new):
                    if t in ends:
                        new = new[:i + 1]
                        break
                new = new[:s.count - len(s.out)]
                s.per_round = 0.75 * getattr(s, "per_round", 2.0) + 0.25 * len(new)
                s.pending = new[-1]
                s.take(new, ends)
                tokens += len(new)
                r0 += n
                if s.done:
                    done.append(s)
            if CONF_LOG:
                self.conf_rounds += 1
                if self.conf_rounds % 200 == 0 and e.rank == 0:
                    dep = [f"{j + 1}:{kept}/{n}" for j, (n, kept) in enumerate(self.conf_depth) if n]
                    bins = [f"{b / 10:.1f}:{kept}/{n}" for b, (n, kept) in enumerate(self.conf_bins) if n]
                    ks = [f"{n}:{','.join(str(x) for x in h[:self.depth_most + 1])}" for n, h in sorted(self.k_hist.items())]
                    print(f"[tensorfold] drafts kept by depth {' '.join(dep)}; by confidence {' '.join(bins)}; "
                          f"k chosen by streams {' '.join(ks)}", flush=True)
            _ht("sampled")
            if absorb_done is not None:                  # what runs on the main stream next sees the rings written
                torch.cuda.current_stream().wait_event(absorb_done)
            if absorbs:                                  # every drafting stream's kept rows into its rings, one pass
                ta = time.perf_counter()
                e.drafter.absorb_many(self.dpool, self.m.pool_view(self.pool, 0, 0, self.extents.total) if
                                      self._table_view is None else self._table_view, absorbs)
                t_absorb += time.perf_counter() - ta
            self.rounds += 1
            if HOST_TRACE:
                _ht("absorbed")
                marks = _HT.pop("_marks", [])
                prev = _HT.get("_end")
                if prev is not None and marks:
                    _HT.setdefault("between (op wait incl.)", []).append(marks[0][1] - prev)
                for (a, ta), (b, tb) in zip(marks, marks[1:]):
                    _HT.setdefault(f"{a} -> {b}", []).append(tb - ta)
                _HT["_end"] = time.perf_counter()
                _HT["_n"] = _HT.get("_n", 0) + 1
                if _HT["_n"] % 100 == 0:
                    import statistics
                    parts = [f"{k} {1000 * statistics.median(v[-100:]):.2f}" for k, v in _HT.items()
                             if not k.startswith("_")]
                    print(f"[tensorfold] rank {e.rank} host trace (median ms, {len(live)} streams): " + "; ".join(parts),
                          flush=True)
            self.round_log.append((len(live), r0, time.perf_counter() - t0, tokens))
            del self.round_log[:-4096]
            if ROUND_STATS:
                mark()
                st = self.stage.setdefault(len(live), [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                st[9] += pre if between < 1.0 else 0.0
                st[0] += 1
                st[1] += r0
                st[2] += tokens
                st[3] += marks[1] - marks[0]                       # drafts
                st[4] += marks[2] - marks[1]                       # Engram rows + the round's forward
                st[5] += marks[3] - marks[2] - t_absorb            # sampling, acceptance, emit
                st[6] += t_absorb
                st[7] += t_agree
                st[8] += between if between < 1.0 else 0.0         # host time from the last round's end (no fills)
                if st[0] % 100 == 0:
                    r = st[0]
                    print(f"[tensorfold] rank {e.rank} rounds at {len(live)} streams: {st[1] / r:.1f} rows, "
                          f"{st[2] / r:.2f} tokens a round; ms between {1000 * st[8] / r:.1f} (in round() "
                          f"{1000 * st[9] / r:.1f}) agree {1000 * st[7] / r:.1f} "
                          f"drafts {1000 * st[3] / r:.1f} forward {1000 * st[4] / r:.1f} sample {1000 * st[5] / r:.1f} "
                          f"absorb {1000 * st[6] / r:.1f}", flush=True)
            self.round_end = time.perf_counter()
            return done
        except Exception as exc:
            self._broken(exc)
            raise
        finally:
            self._step(False)

    def finish(self, done: list[Stream]) -> None:
        if ROUND_STATS and done:
            self.round_end = None                      # (a finish's link step is not counted as host time either)
        if not done:
            return
        sids = [s.sid for s in done if s.sid in self.streams]
        if not sids:
            return
        drops = self._covered(done) if KEEP else []
        self._send(["finish", sids, drops, self.spill_nonce])
        self._finish(sids, drops)

    def _release(self, s: Stream) -> None:
        self.free.append(s.st.index)
        self.free.sort()
        self.extents.give(s.base, s.size)
        s.st.sc = None

    def _finish(self, sids: list[int], drops: list[int] = ()) -> None:
        """Finished streams give their slots and extents back; with KEEP their prompts stay as kept prompts, the
        ``drops`` (kept prompts they cover) go, and so do the oldest past KEEP_ENTRIES. Spilled (TF_DS_SPILL_GIB):
        a dropped prompt's manifest goes once every new kept prompt that should have spilled did (the new manifests
        name its rows); prompts that only leave the window stay on disk."""

        kept_now: list[Kept] = []
        for sid in sids:
            s = self.streams.pop(sid, None)
            if s is None:
                continue
            if KEEP and getattr(s, "snaps", None):
                self._keep(s)
                kept_now.append(self.kept[self.next_kept - 1])
            else:
                self._release(s)
        sp = self.spill
        spilled = sp is not None and all(k.disk is not None for k in kept_now if k.top >= sp.min_top)
        for eid in drops:
            if int(eid) in self.kept:
                k = self.kept[int(eid)]
                if spilled and k.disk is not None:
                    sp.forget(k.disk["name"])
                self._forget(k)
        while len(self.kept) > KEEP_ENTRIES:
            self._forget(min(self.kept.values(), key=lambda k: k.tick))
        if sp is not None and kept_now:
            sp.recycle(pinned={k.disk["name"] for k in self.kept.values() if k.disk is not None})

    def drop(self) -> list[Stream]:
        """Every live stream fails (a step raised); their slots are free again."""

        out = list(self.streams.values()) + list(self.filling)
        try:
            self._send(["drop"])
        except Exception:                                # noqa: BLE001  (the link may be what failed)
            pass
        self._drop()
        return out

    def _drop(self) -> None:
        for s in list(self.streams.values()) + self.filling:
            self._release(s)
        self.streams.clear()
        self.filling.clear()
        for k in list(self.kept.values()):              # (the ranks may disagree about them now)
            self._forget(k)

    # -- followers ----------------------------------------------------------------------------------------------
    def follow(self, link: Link) -> None:
        """A follower rank: rank 0's steps, in order, forever (or until the link or the lane breaks)."""

        while True:
            op = link.receive()
            _ht("op received")
            if op is None:
                raise RuntimeError("rank 0's step link closed")
            kind = op[0]
            try:
                if kind == "admit":
                    _, prompt, count, sampling, draft, stop_eos, index, positions, reuse = op
                    s = Stream(list(prompt), int(count), _unpack(sampling), draft=bool(draft), stop_eos=bool(stop_eos))
                    s.emit = lambda new: None
                    s.keys = None
                    self._admit(s, int(index), list(positions), reuse)
                elif kind == "fill":
                    self._fill(next(f for f in self.filling if f.sid == int(op[1])))
                elif kind == "round":
                    self.round(told=op)
                elif kind == "finish":
                    if len(op) > 3 and op[3]:
                        self.spill_nonce = str(op[3])    # rank 0's name for this run's spilled prompts
                    self._finish([int(x) for x in op[1]], [int(x) for x in op[2]])
                elif kind == "drop":
                    self._drop()
            except OutOfStep as exc:
                print(f"[tensorfold] rank {self.e.rank}: {exc}", flush=True)
                self._drop()
