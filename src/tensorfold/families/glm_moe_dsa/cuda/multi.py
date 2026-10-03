"""GLM-5.3's concurrent decoding (``--parallel N``): up to N requests on the lane at once, every live stream's verify
window in one forward a round, each reply exactly its solo run (pipeline/CONCURRENCY-DESIGN.md).

- **Pool.** One cache plane a layer holds every slot: rows [s * cap, (s + 1) * cap) are slot s. A stream's prompt
  fills its slot through the single-stream path on the slot's views (``forward.State`` over view planes), in the solo
  prompt's chunks, so its prompt bits are the solo prompt's. Chunks run between rounds: rank 0 runs one when nothing
  decodes, or once decoding has had ``DECODE_SHARE`` of the time since the last one, and tells the followers.
- **Rounds.** Every live stream's pending token and drafts go through one forward (``rows.Tables``: each row's
  position and slot). Every kernel is row-invariant and the cache kernels read each row's own stream, so a row's
  logits are the solo run's at that position, and each stream samples its own rows by its own rule: its reply is
  its solo reply, whatever else runs beside it. Rounds replay CUDA graphs captured per (rows, dense / sparse passes,
  bucket) on first use, reading the tables from device memory.
- **Drafts.** Each stream's MTP chain; step j of every drafting stream runs as one MTP forward over the same tables
  on the MTP planes. Drafts only propose: they change speed, never a reply.
- **Ranks.** Rank 0 schedules (``tensorfold.cuda.scheduler``) and sends each step to the followers over one TCP link
  each (``Link``) before running it; every rank checks the step's digest with one small all-gather before its model
  collectives (``OutOfStep``: the step fails on every rank at the same point, nobody waits on anybody).

Grammars (structured output), logprobs and DCP are not served with ``--parallel`` yet.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time

import torch

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream

from . import forward as fwd, rows as rows_mod

MAX_GRAPHS = 96                  # round graphs kept (rows x passes x bucket); past it a new shape runs eager
DECODE_SHARE = float(os.environ.get("TF_GLM_DECODE_SHARE") or 0.5)   # decoding's share of the time while prompts fill
# drafts a stream chains by how many streams decode (TF_GLM_PARALLEL_DEPTH="3,3,2,2": 3 alone or beside one other, 2
# at three or four): a round's rows grow with every stream's window, and deeper drafts pay off less as rows grow
DEPTH_BY = [int(x) for x in (os.environ.get("TF_GLM_PARALLEL_DEPTH") or "").split(",") if x.strip()]


class OutOfStep(RuntimeError):
    """The ranks planned a different step: it fails on every rank before its collectives."""


class Link:
    """Rank 0's steps to every follower, in order, over one TCP connection each (port through the rendezvous store);
    a follower waits on its socket between steps, never inside a collective."""

    KEY = "tensorfold/glm/multi/port"

    def __init__(self, store, *, rank: int, world: int, host: str) -> None:
        import socket
        from datetime import timedelta

        self.rank, self.world, self.socks = rank, world, []
        if rank == 0:
            self.server = socket.create_server((host or "0.0.0.0", 0))
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
            if not chunk:                               # the other side is gone
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
    """Liveness for a concurrent lane on several ranks: heartbeats between rank 0 and each follower on their own TCP
    connections (beside the step link), and NCCL's asynchronous error. A heartbeat missing for ``timeout`` seconds, a
    closed connection or an NCCL error breaks the lane: the RDMA gathers stop waiting (their abort word), NCCL aborts,
    rank 0 tells every follower to do the same, and the step in flight fails on every rank instead of hanging."""

    KEY = "tensorfold/glm/multi/watch"

    STEP_TIMEOUT = float(os.environ.get("TF_GLM_STEP_TIMEOUT") or 600.0)

    def __init__(self, comm, store, *, rank: int, world: int, host: str, timeout: float = 30.0,
                 every: float = 1.0) -> None:
        import socket
        import threading
        from datetime import timedelta

        self.comm, self.rank, self.world, self.timeout, self.every = comm, rank, world, timeout, every
        self.broken: str | None = None
        self.busy: float | None = None                   # when the step in flight started (a stuck step breaks the lane)
        self.lock = threading.Lock()
        self.socks: list = []
        if rank == 0:
            server = socket.create_server((host or "0.0.0.0", 0))
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
        import time

        while self.broken is None:
            with self.lock:
                socks = list(self.socks)
            for sock in socks:
                try:
                    sock.sendall(b"H")
                except OSError:
                    pass                                 # its listener sees the connection close
            time.sleep(self.every)

    def _listen(self, sock, peer: int) -> None:
        sock.settimeout(self.timeout)
        while self.broken is None:
            try:
                got = sock.recv(64)
            except OSError:                              # (a timeout included)
                got = b""
            if not got:
                self.abort(f"rank {peer} stopped answering" if self.rank == 0 else "rank 0 stopped answering")
                return
            if b"A" in got:
                self.abort("rank 0 lost a rank")
                return

    def _nccl(self) -> None:
        """NCCL's asynchronous errors, and a step that never ends (a rank that raised alone leaves the others in a
        collective it never joins)."""

        import time

        nccl = getattr(self.comm, "nccl", self.comm)
        check = getattr(nccl, "async_error", None)
        while self.broken is None:
            time.sleep(self.every)
            code = check() if check is not None else 0
            if code > 0:
                self.abort(f"NCCL error {code}")
                return
            busy = self.busy
            if busy is not None and time.monotonic() - busy > self.STEP_TIMEOUT:
                self.abort(f"a step ran past {self.STEP_TIMEOUT:.0f} s on rank {self.rank}")
                return

    def abort(self, why: str) -> None:
        """Break the lane on this rank (once) and, on rank 0, on every follower."""

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


def _rows_view(p, lo: int, hi: int):
    return p[lo:hi] if isinstance(p, torch.Tensor) else p.rows(lo, hi)


def slot_state(pool: fwd.State, lo: int, hi: int) -> fwd.State:
    """A single-stream State over pool rows [lo, hi): view planes, its own positions."""

    v = copy.copy(pool)
    dev = pool.pos_dev.device
    v.capacity = hi - lo
    v.pos, v.mtp_len, v.mtp_drafted = 0, 0, 0
    v.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
    v.mtp_pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
    v.kc = [_rows_view(x, lo, hi) for x in pool.kc]
    v.pc = [_rows_view(x, lo, hi) for x in pool.pc]
    v.vc = [None for _ in pool.kc]
    if pool.index is not None:
        v.index = [_rows_view(x, lo, hi) for x in pool.index]
    if hasattr(pool, "mtp_kc"):
        v.mtp_kc = _rows_view(pool.mtp_kc, lo, hi)
        v.mtp_pc = _rows_view(pool.mtp_pc, lo, hi)
    return v


class Slot:
    """A stream's cache slot: its first pool row and its single-stream State (views; positions are the stream's)."""

    def __init__(self, index: int, base: int, st: fwd.State) -> None:
        self.index, self.base, self.st = index, base, st


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``slot_cap`` cache rows."""

    def __init__(self, w, *, slots: int, slot_cap: int, depth: int, prefill_rows: int, graphs: bool = True) -> None:
        from .decode import Engine

        if w.meta.get("dcp", 1) > 1:
            raise ValueError("--parallel and decode context parallelism (TF_GLM_DCP) do not run together yet")
        dev = w.device
        self.w, self.depth, self.slot_cap, self.prefill_rows = w, int(depth), int(slot_cap), int(prefill_rows)
        self.eos = tuple(w.cfg.eos)
        width = self.depth + 1                           # a stream's window: its pending token and its drafts
        rows = slots * width
        self.pool = fwd.State(w, slots * slot_cap, width)
        self.slots = [Slot(i, i * slot_cap, slot_state(self.pool, i * slot_cap, (i + 1) * slot_cap))
                      for i in range(slots)]
        self.free = list(range(slots))
        self.buf = fwd.Buffers(w, rows, slot_cap)
        self.buf.rows_t, self.buf.stream_rows = rows_mod.Tables(rows, slots, dev), width
        self.mbuf = None
        if w.mtp is not None:
            self.mbuf = fwd.Buffers(w, rows, slot_cap)
            self.mbuf.rows_t, self.mbuf.stream_rows = rows_mod.Tables(rows, slots, dev), 1 + width
            self.mbuf.mhead = torch.empty((rows, w.cfg.hidden), dtype=torch.bfloat16, device=dev)
            self.mbuf.zero_first = False
        # a stream's prompt and its first draft chain: the single-stream engine on its slot's views
        self.one = Engine.__new__(Engine)
        one = self.one
        one.w, one.rows, one.prefill_rows = w, 8, self.prefill_rows
        one.buf = None
        one.pbuf = fwd.Buffers(w, self.prefill_rows, slot_cap, prefill=True)
        one.mbuf = fwd.Buffers(w, 8, slot_cap) if w.mtp is not None else None
        one.st, one.last_hidden, one.constraint, one.window = self.slots[0].st, None, None, None
        one.draft_n = (w.head if w.draft_head is None else w.draft_head).n
        one.graphs, one.serial_only = None, False
        one.replays = {"main": 0, "sparse": 0, "mtp": 0, "sparse_mtp": 0, "eager": 0}
        self.streams: dict[int, Stream] = {}             # decoding (and just ended) streams
        self.filling: list[Stream] = []                  # admitted, their prompts filling their slots (oldest first)
        self.next_id = 0
        self.chunk_s, self.since_fill = 0.0, 0.0         # rank 0's fill pacing: the last chunk's time, decoding since
        self.graphs = {} if graphs and dev.type == "cuda" else None
        self.graph_pool = torch.cuda.graph_pool_handle() if self.graphs is not None else None
        self.link: Link | None = None                    # rank 0 with followers: each step goes to them first
        self.watch: Watchdog | None = None               # several ranks: a lost rank breaks the lane on every rank
        self.rounds = {"graph": 0, "eager": 0, "captured": 0}
        self.round_log: list[tuple[int, int, float, int]] = []   # (streams decoding, rows, seconds, tokens) a round

    # -- the scheduler's interface ------------------------------------------------------------------------------------
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

        w = self.w
        if w.comm is None or w.world < 2:
            return
        d = int.from_bytes(hashlib.sha256(json.dumps(plan).encode()).digest()[:8], "big")
        mine = torch.tensor([d & 0x7FFFFFFF, (d >> 31) & 0x7FFFFFFF, d >> 62], dtype=torch.int32, device="cuda")
        got = torch.empty((w.world * 3,), dtype=torch.int32, device="cuda")
        w.comm.all_gather(mine, got)
        every = got.view(w.world, 3).tolist()
        if any(row != every[0] for row in every):
            raise OutOfStep(f"the ranks planned different {what}s; its requests fail, serving goes on")

    def _shape(self) -> list:
        return [self.next_id, list(self.free),
                [[s.sid, s.st.index, s.st.st.pos, s.st.st.mtp_len, len(s.out), list(s.drafts), bool(s.done)]
                 for s in self.streams.values()],
                [[s.sid, s.st.index, s.filled] for s in self.filling]]

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """A request into the lowest free slot; rounds fill its prompt (``_fill``), then it decodes."""

        if s.constraint is not None or s.probabilities is not None or s.vision is not None:
            raise ValueError("structured output, logprobs and images are not served with --parallel on GLM-5.3")
        room = self.slot_cap - len(s.prompt) - self.depth - 2
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.slot_cap}-token context")
        s.count = max(1, min(s.count, room))
        if not self.free:
            raise NoRoom("every stream slot is busy")
        self._send(["admit", list(s.prompt), s.count, _pack(s.sampling), bool(s.draft), bool(s.stop_eos)])
        self._agree("admission", [self._shape(), list(s.prompt), s.count, _pack(s.sampling), bool(s.draft),
                                  bool(s.stop_eos)])
        slot = self.slots[self.free.pop(0)]
        slot.st.reset()
        s.sid, s.st, s.filled = self.next_id, slot, 0
        self.next_id += 1
        self.filling.append(s)

    def _fill(self, s: Stream) -> list[Stream]:
        """One prompt chunk of ``s`` on its slot's views: decode.prefill's steps for that chunk, in its chunks. The last
        one samples the first token and makes the first draft chain; the stream then decodes. Returns it if it ended."""

        from .decode import draft
        from .mtp import mtp_forward

        e, w = self.one, self.w
        e.st = st = s.st.st
        b = e.pbuf
        t0 = time.perf_counter()
        start = s.filled
        chunk = list(s.prompt[start:start + self.prefill_rows])
        R = fwd.stage(w, st, b, chunk)
        last = fwd.compute(w, st, b, R, nch=fwd.chunks_for(st, R), host_pos=st.pos)
        end = start + R >= len(s.prompt)
        if end:                                          # before the MTP head reuses the buffers
            last = last.clone()
            e.last_hidden = b.fnormed[R - 1:R].clone()
        if w.mtp is not None:
            nxt = list(s.prompt[start + 1:start + R + 1])
            if nxt:
                mtp_forward(w, st, b, nxt, b.fnormed[:len(nxt)])
                st.set_mtp_len(st.mtp_len + len(nxt))
        fwd.commit(w, st, b, R, R)
        s.filled = start + R
        torch.cuda.synchronize()
        self._alive()
        s.prefill_s += time.perf_counter() - t0
        if not end:
            return []
        self.filling.remove(s)
        first = e.sample(last, [len(s.prompt)], s.sampling)[0]
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.streams[s.sid] = s
        mtp = s.draft and self.depth > 0 and w.mtp is not None
        s.drafts = (draft(e, e.last_hidden, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling)
                    if mtp and s.count > 1 else [])
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def _fill_choice(self) -> int | None:
        """Rank 0: the filling stream whose next chunk runs before this round, if any."""

        if not self.filling:
            return None
        decoding = any(not s.done for s in self.streams.values())
        if not decoding or self.since_fill * (1 - DECODE_SHARE) >= self.chunk_s * DECODE_SHARE:
            return self.filling[0].sid
        return None

    @torch.no_grad()
    def round(self, told: list | None = None) -> list[Stream]:
        """A prompt chunk when one is due (rank 0 decides), then one round over the decoding streams: their windows in
        one forward, each stream's rows sampled by its rule, the kept tokens committed, the next drafts chained.
        Returns the streams that ended. ``told``: a follower's copy of rank 0's (ended streams, chunk) for this step (a
        client that leaves ends a stream on rank 0 only)."""

        if told is not None:
            ended_ids, fill = told
            for sid in ended_ids:
                if sid in self.streams:
                    self.streams[sid].done = True
        else:
            fill = self._fill_choice()
        self._send(["round", [sorted(s.sid for s in self.streams.values() if s.done), fill]])
        self._step(True)
        try:
            return self._round(fill)
        finally:
            self._step(False)

    def _round(self, fill: int | None) -> list[Stream]:
        self._agree("round", [self._shape(), fill])
        ended: list[Stream] = []
        if fill is not None:
            t0 = time.perf_counter()
            ended += self._fill(next(s for s in self.filling if s.sid == fill))
            self.chunk_s, self.since_fill = time.perf_counter() - t0, 0.0
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return ended
        t_round = time.perf_counter()
        w, b = self.w, self.buf
        windows = [[s.out[-1]] + list(s.drafts) for s in live]
        t = b.rows_t
        R = t.fill([(s.st.base, s.st.st.pos, len(tok)) for s, tok in zip(live, windows)], w.cfg.dense_limit)
        b.staged.synchronize()
        b.ids_host[:R].numpy()[:] = [x for tok in windows for x in tok]
        b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
        b.staged.record()
        logits = self._forward(R)
        torch.cuda.synchronize()
        self._alive()                                    # a lost rank: the gathers gave up, their rows are garbage
        spans, a0 = [], 0
        for s, tok in zip(live, windows):
            spans.append((a0, len(tok), [s.st.st.pos + 1 + r for r in range(len(tok))], s.sampling))
            a0 += len(tok)
        picks = self._sample_spans(logits, spans)
        kept, a0 = [], 0
        for s, tok, sampled in zip(live, windows, picks):
            n = len(tok)
            st = s.st.st
            room = s.count - len(s.out)
            keep = 1
            for i, d in enumerate(tok[1:]):
                if keep >= room or sampled[i] != d or (s.stop_eos and sampled[i] in self.eos):
                    break
                keep += 1
            st.set_pos(st.pos + keep)
            s.committed.extend(tok[:keep])
            s.counted(n)
            new = sampled[:keep]
            last = len(s.out) + len(new) >= s.count or (s.stop_eos and new[-1] in self.eos)
            kept.append((s, a0, keep, new, last))
            a0 += n
        self._draft_all([(s, a0, keep, new) for s, a0, keep, new, last in kept if s.draft and not last])
        for s, _, _, new, _ in kept:
            s.take(new, self._ends(s))
        spent = time.perf_counter() - t_round
        self.since_fill += spent
        self.round_log.append((len(live), R, spent, sum(len(new) for _, _, _, new, _ in kept)))
        return ended + [s for s in live if s.done]

    def finish(self, done: list[Stream]) -> None:
        """Ended streams leave; their slots are free again."""

        try:
            self._send(["finish", [s.sid for s in done]])
        except (RuntimeError, OSError):              # a broken lane: the slots are freed here all the same
            pass
        for s in done:
            if self.streams.pop(s.sid, None) is not None or s in self.filling:
                if s in self.filling:
                    self.filling.remove(s)
                self.free.append(s.st.index)
        self.free.sort()

    def drop(self) -> list[Stream]:
        try:
            self._send(["drop"])
        except (RuntimeError, OSError):              # a broken lane: the followers are told by the watchdog
            pass
        live = list(self.streams.values()) + self.filling
        for s in live:
            self.free.append(s.st.index)
        self.streams.clear()
        self.filling = []
        self.free.sort()
        return live

    def warm(self) -> None:
        """A short greedy request through prefill, one round and its drafts, then forgotten (kernels compiled)."""

        s = Stream([1] * 24, 3)
        link, self.link = self.link, None
        try:
            self.admit(s)
            while not s.done:
                self.round()
            self.finish([s])
        finally:
            self.link = link

    # -- the round's parts ---------------------------------------------------------------------------------------------
    def _forward(self, R: int) -> torch.Tensor:
        """The round's logits: its captured graph (captured on first use), else eager."""

        b = self.buf
        if self.graphs is not None:
            key = b.rows_t.key()
            g = self.graphs.get(key)
            if g is None and len(self.graphs) < MAX_GRAPHS:
                g = self._capture(key, R)
            if g is not None:
                self.rounds["graph"] += 1
                g.replay()
                return b.logits[:R]
        self.rounds["eager"] += 1
        return fwd.compute_rows(self.w, self.pool, b, R)

    def _capture(self, key, R: int):
        """The round's forward as a CUDA graph. Its warm-up runs write the same cache rows the round writes (the same
        rows of the same inputs: the same bits), so capturing during a round changes nothing."""

        w, b = self.w, self.buf
        for _ in range(2):
            fwd.compute_rows(w, self.pool, b, R)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.graph_pool):
            fwd.compute_rows(w, self.pool, b, R)
        self.graphs[key] = g
        self.rounds["captured"] += 1
        return g

    def _sample(self, logits: torch.Tensor, positions: list[int], sampling, *, draft: bool = False) -> list[int]:
        from tensorfold.families.glm5_next.cuda.decode import sample_rows

        w = self.w
        offset = w.meta["draft_lo"] if draft and w.draft_head is not None else None
        return sample_rows(w, logits, positions, sampling, offset)

    def _sample_spans(self, logits: torch.Tensor, spans: list, *, draft: bool = False) -> list[list[int]]:
        """Each span's rows (first row, rows, positions, sampling) sampled by its own rule: every greedy span's rows in
        one call (a greedy row's token is its own argmax, whatever rows share the call), the others one call each."""

        out: list[list[int] | None] = [None] * len(spans)
        greedy = [i for i, sp in enumerate(spans) if sp[3] is None or sp[3].temperature <= 0]
        if greedy:
            rows = torch.cat([logits[spans[i][0]:spans[i][0] + spans[i][1]] for i in greedy])
            got = self._sample(rows, [p for i in greedy for p in spans[i][2]], None, draft=draft)
            r = 0
            for i in greedy:
                out[i] = got[r:r + spans[i][1]]
                r += spans[i][1]
        for i, (a0, n, positions, sampling) in enumerate(spans):
            if out[i] is None:
                out[i] = self._sample(logits[a0:a0 + n], positions, sampling, draft=draft)
        return out

    def _draft_all(self, todo: list) -> None:
        """Each drafting stream absorbs its kept rows and chains its drafts; every stream's step j in one MTP forward
        (the single-stream ``decode.draft`` without a confidence cut, stream by stream)."""

        for s, *_ in todo:
            s.drafts = []
        if self.mbuf is None:
            return
        depth = self.depth
        if DEPTH_BY:
            busy = sum(1 for s in self.streams.values() if not s.done)
            depth = min(depth, DEPTH_BY[min(busy, len(DEPTH_BY)) - 1])
        room = {s.sid: min(depth, s.count - len(s.out) - keep) for s, _, keep, _ in todo}
        todo = [x for x in todo if room[x[0].sid] > 0]
        if not todo:
            return
        w, mb, b = self.w, self.mbuf, self.buf
        for s, *_ in todo:
            st = s.st.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        # the absorb: each stream's kept main rows (its final-normed hidden rows) with the tokens that follow them
        n = mb.rows_t.fill([(s.st.base, s.st.st.mtp_len, keep) for s, _, keep, _ in todo], w.cfg.dense_limit)
        mb.hin[:n].copy_(torch.cat([b.fnormed[a0:a0 + keep] for _, a0, keep, _ in todo]))
        heads, r = [], 0
        for _, _, keep, _ in todo:
            r += keep
            heads.append(r - 1)
        logits = self._mtp(n, [x for _, _, _, new in todo for x in new], heads)
        for s, _, keep, _ in todo:
            s.st.st.set_mtp_len(s.st.st.mtp_len + keep)
        active = [s for s, *_ in todo]
        for j in range(self.depth):
            nxt = []
            picks = self._sample_spans(logits, [(i, 1, [s.st.st.pos + 1 + j], s.sampling) for i, s in enumerate(active)],
                                       draft=True)
            for i, s in enumerate(active):
                d = picks[i][0]
                s.drafts.append(d)
                if j + 1 < room[s.sid]:
                    nxt.append((i, s, d))
            if not nxt:
                return
            # the next step: one row a stream, its last draft and the head's own output row
            n = mb.rows_t.fill([(s.st.base, s.st.st.mtp_len, 1) for _, s, _ in nxt], w.cfg.dense_limit)
            picked = torch.tensor([i for i, _, _ in nxt], dtype=torch.int64, device=mb.hin.device)
            mb.hin[:n].copy_(torch.index_select(mb.fnormed, 0, picked))
            logits = self._mtp(n, [d for _, _, d in nxt], list(range(n)))
            for _, s, _ in nxt:
                s.st.st.set_mtp_len(s.st.st.mtp_len + 1)
                s.st.st.mtp_drafted += 1
            active = [s for _, s, _ in nxt]

    def _mtp(self, n: int, tokens: list[int], heads: list[int]) -> torch.Tensor:
        from .mtp import mtp_compute_rows

        mb = self.mbuf
        mb.staged.synchronize()
        mb.ids_host[:n].numpy()[:] = tokens
        mb.ids[:n].copy_(mb.ids_host[:n], non_blocking=True)
        mb.staged.record()
        out = mtp_compute_rows(self.w, self.pool, mb, n, torch.tensor(heads, dtype=torch.int64, device=mb.ids.device))
        torch.cuda.synchronize()
        self._alive()
        return out

    # -- followers ------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def follow(self, link: Link) -> None:
        """A follower rank: every step rank 0 sends, in order, until it stops."""

        while True:
            op = link.receive()
            if op is None or op[0] == "stop":
                return
            try:
                if op[0] == "admit":
                    prompt, count, smp, drafted, stop = op[1:]
                    self.admit(Stream(prompt, count, _unpack(smp), draft=drafted, stop_eos=stop))
                elif op[0] == "round":
                    self.round(told=op[1])
                elif op[0] == "finish":
                    known = {**{s.sid: s for s in self.filling}, **self.streams}
                    self.finish([known[sid] for sid in op[1] if sid in known])
                elif op[0] == "drop":
                    self.drop()
            except (OutOfStep, NoRoom, ValueError) as exc:     # rank 0 failed the same step at the same point
                print(f"[tensorfold] rank {self.w.rank}: a {op[0]} step failed: {exc}", flush=True)
