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

# drafts a stream verifies a round by how many streams decode ("3,3,3,3": three each at 1-4 streams, 16 rows at 4)
DEPTH_BY = [int(x) for x in (os.environ.get("TF_DS_PARALLEL_DEPTH") or "3,3,3,3").split(",") if x.strip()]
# while prompts fill, decoding keeps this share of the time (a prompt chunk runs once the rounds since the last one
# took share / (1 - share) of its time); a prompt whose rest fits in QUICK_ROWS tokens fills before the next round
DECODE_SHARE = float(os.environ.get("TF_DS_DECODE_SHARE") or 0.5)
QUICK_ROWS = int(os.environ.get("TF_DS_QUICK_ROWS") or 1024)
# TF_DS_ROUND_STATS=1: each round's stages synchronized and timed (drafts, Engram rows + forward, sampling, absorb),
# their averages printed every 100 rounds by streams decoding (profiling only: the syncs cost a little)
ROUND_STATS = os.environ.get("TF_DS_ROUND_STATS", "0") == "1"
STEP_TIMEOUT = float(os.environ.get("TF_DS_STEP_TIMEOUT") or 900.0)


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
    """A stream's cache slot: its index, its SeqCache (pool views) and its drafter cache (and graph)."""

    def __init__(self, index: int, sc, dc) -> None:
        self.index, self.sc, self.dc = index, sc, dc
        self.dg = None


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
        self.slots = [Slot(i, self.pool.views[i], self.dpool.views[i] if d is not None else None)
                      for i in range(slots)]
        self.drafters: dict[int, object] = {}            # batched drafter graphs by drafting streams
        self.free = list(range(slots))
        self.max_rows = MAX_ROWS
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
                self.runner.forward([(0, b - R, host[b - R:b], host)])
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
            g = BatchDraftGraph(self.e.drafter, self.pool.views[0], self.dpool, N)
            g.tokens.copy_(torch.tensor(tokens, dtype=torch.long))
            g.q0.copy_(torch.tensor(q0, dtype=torch.long))
            g.slots.copy_(torch.tensor(slots, dtype=torch.long))
            if self.e.runner is not None:
                if self.e.runner.pool is None:
                    self.e.runner.pool = torch.cuda.graph_pool_handle()
                g.capture(self.e.runner.pool)
            self.drafters[N] = g
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

    def _alive(self) -> None:
        if self.watch is not None and self.watch.broken is not None:
            raise RuntimeError(f"this lane lost a rank ({self.watch.broken}); restart it on every rank")

    def _agree(self, what: str, plan: list) -> None:
        """One small all-gather checks the step on every rank before any of its model collectives."""

        e = self.e
        if e.world < 2:
            return
        dgst = int.from_bytes(hashlib.sha256(json.dumps(plan).encode()).digest()[:8], "big")
        mine = torch.tensor([dgst & 0x7FFFFFFF, (dgst >> 31) & 0x7FFFFFFF, dgst >> 62], dtype=torch.int64,
                            device="cuda")
        got = torch.empty((e.world * 3,), dtype=torch.int64, device="cuda")
        e.nccl.all_gather(mine, got)
        every = got.view(e.world, 3).tolist()
        if any(row != every[0] for row in every):
            raise OutOfStep(f"the ranks planned different {what}s; its requests fail, serving goes on")

    def _shape(self) -> list:
        return [self.next_id, list(self.free),
                [[s.sid, s.st.index, s.st.sc.length, len(s.out), bool(s.done)] for s in self.streams.values()],
                [[s.sid, s.st.index, s.filled] for s in self.filling]]

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """A request into the lowest free slot: its prompt fills the slot now, then it decodes in the rounds."""

        if s.constraint is not None or s.probabilities is not None:
            raise ValueError("structured output and logprobs are not served with --parallel on DeepSeek-V4.1")
        room = self.cap - len(s.prompt) - self.max_rows - 8
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.cap}-token context")
        s.count = max(1, min(s.count, room))
        if not self.free:
            raise NoRoom("every stream slot is busy")
        positions = s.vision.positions() if s.vision is not None and getattr(s.vision, "spans", None) else []
        index = self.free[0]
        self._send(["admit", list(s.prompt), s.count, _pack(s.sampling), bool(s.draft), bool(s.stop_eos), index,
                    positions])
        self._admit(s, index, positions)

    def _admit(self, s: Stream, index: int, positions: list[int]) -> None:
        """Its slot, its image rows (shared from rank 0) and its prompt's chunk steps; ``_fill`` runs them."""

        e = self.e
        self._step(True)
        try:
            self._agree("admission", [self._shape(), list(s.prompt), s.count, _pack(s.sampling), bool(s.draft),
                                      bool(s.stop_eos), index, positions])
            slot = self.slots[index]
            self.free.remove(index)
            s.sid, s.st = self.next_id, slot
            self.next_id += 1
            image = None
            if positions:
                rows = None
                if e.rank == 0:
                    rows = torch.cat([e.tower.span_rows(pic) for _, pic in s.vision.spans])
                image = (positions, e._share_rows(rows, len(positions)))
            s.draft = bool(s.draft) and e.drafter is not None
            s.steps = e.prefill_steps(slot.sc, slot.dc if s.draft else None, list(s.prompt), image)
            s.filled, s.prefill_s, s.cached = 0, 0.0, 0
            self.filling.append(s)
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
        finally:
            self._step(False)

    def _depth(self, live: int) -> int:
        k = DEPTH_BY[min(live, len(DEPTH_BY)) - 1] if DEPTH_BY else self.depth_most
        return max(0, min(k, self.depth_most, self.max_rows // max(live, 1) - 1))

    @torch.no_grad()
    def round(self, told: list | None = None) -> list[Stream]:
        """One round over every live stream; returns the streams that ended."""

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
            self._agree("round", [self._shape(), k])
            if ROUND_STATS:
                torch.cuda.synchronize()
            t_agree = time.perf_counter() - ta0
            between = ta0 - self.round_end if getattr(self, "round_end", None) else 0.0
            windows, kept = [], []
            marks = [time.perf_counter()]

            def mark():
                if ROUND_STATS:
                    torch.cuda.synchronize()
                    marks.append(time.perf_counter())

            want = {s.sid: (min(k, s.count - len(s.out)) if s.draft else 0) for s in live}
            drafting = [s for s in live if want[s.sid] > 0]
            proposed = {}
            if drafting:
                rows = self._drafts([s.pending for s in drafting], [s.st.sc.length for s in drafting],
                                    [s.st.index for s in drafting])
                proposed = {s.sid: r[:want[s.sid]] for s, r in zip(drafting, rows)}
            mark()
            for s in live:
                sc = s.st.sc
                P = sc.length
                drafts: list[int] = proposed.get(s.sid, [])
                window = [s.pending] + drafts
                del sc.host[P:]
                sc.host.extend(int(t) for t in window)
                windows.append((s.st.index, P, window, sc.host))
                kept.append((s, P, drafts))
            logits, taps = self.runner.forward(windows)
            mark()
            done, r0, tokens = [], 0, 0
            t_absorb = 0.0
            for (s, P, drafts), (_, _, window, _) in zip(kept, windows):
                n = len(window)
                target = e._sample(logits[r0:r0 + n], [P + 1 + i for i in range(n)], s.sampling)
                a = 0
                while a < len(drafts) and drafts[a] == target[a]:
                    a += 1
                new = drafts[:a] + [target[a]]
                s.st.sc.length = P + a + 1
                if s.draft and taps is not None:
                    ta = time.perf_counter()
                    e.drafter.absorb(s.st.dc, s.st.sc, taps[r0:r0 + a + 1], P)
                    t_absorb += time.perf_counter() - ta
                s.counted(n)
                ends = self._ends(s)
                for i, t in enumerate(new):
                    if t in ends:
                        new = new[:i + 1]
                        break
                new = new[:s.count - len(s.out)]
                s.pending = new[-1]
                s.take(new, ends)
                tokens += len(new)
                r0 += n
                if s.done:
                    done.append(s)
            self.rounds += 1
            self.round_log.append((len(live), r0, time.perf_counter() - t0, tokens))
            del self.round_log[:-4096]
            if ROUND_STATS:
                mark()
                st = self.stage.setdefault(len(live), [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
                          f"{st[2] / r:.2f} tokens a round; ms between {1000 * st[8] / r:.1f} agree {1000 * st[7] / r:.1f} "
                          f"drafts {1000 * st[3] / r:.1f} forward {1000 * st[4] / r:.1f} sample {1000 * st[5] / r:.1f} "
                          f"absorb {1000 * st[6] / r:.1f}", flush=True)
            self.round_end = time.perf_counter()
            return done
        finally:
            self._step(False)

    def finish(self, done: list[Stream]) -> None:
        if not done:
            return
        sids = [s.sid for s in done if s.sid in self.streams]
        if not sids:
            return
        self._send(["finish", sids])
        self._finish(sids)

    def _finish(self, sids: list[int]) -> None:
        for sid in sids:
            s = self.streams.pop(sid, None)
            if s is not None:
                self.free.append(s.st.index)
                self.free.sort()

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
            self.free.append(s.st.index)
        self.free.sort()
        self.streams.clear()
        self.filling.clear()

    # -- followers ----------------------------------------------------------------------------------------------
    def follow(self, link: Link) -> None:
        """A follower rank: rank 0's steps, in order, forever (or until the link or the lane breaks)."""

        while True:
            op = link.receive()
            if op is None:
                raise RuntimeError("rank 0's step link closed")
            kind = op[0]
            try:
                if kind == "admit":
                    _, prompt, count, sampling, draft, stop_eos, index, positions = op
                    s = Stream(list(prompt), int(count), _unpack(sampling), draft=bool(draft), stop_eos=bool(stop_eos))
                    s.emit = lambda new: None
                    self._admit(s, int(index), list(positions))
                elif kind == "fill":
                    self._fill(next(f for f in self.filling if f.sid == int(op[1])))
                elif kind == "round":
                    self.round(told=op)
                elif kind == "finish":
                    self._finish([int(x) for x in op[1]])
                elif kind == "drop":
                    self._drop()
            except OutOfStep as exc:
                print(f"[tensorfold] rank {self.e.rank}: {exc}", flush=True)
                self._drop()
