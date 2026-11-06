"""DP4 scheduler pieces (note D2 §1/§5): owner placement, the step message, the digest agreement,
and the rank-0/follower link — the ``multi.py`` pattern (rank 0 schedules, followers follow, one
small all-gather checks every step before its collectives) reshaped for data-parallel rows.

What DP4 changes about a step: under TP every rank runs every row; under DP4 **rank r's payload is
its own rows only** (``op[3][r]``), each stream living on exactly one owner rank with its KV,
selection state and caches (D2 §4). The MoE all-to-all inside the forward is what still binds the
lane together, so even a rank with no rows runs the round — an empty row set keeps that rank inside
the layer collectives (``Ml4Model._layer``'s zero-row path).

Placement (D2 §5): owner = least-loaded by ``free_pool_tokens`` primary, ``queued_prefill_tokens +
rows`` secondary — never round-robin; an ``X-Session-Id`` pins follow-ups to the same owner so the
per-rank kept-prompt cache hits (sticky). Bucket agreement (D2 O4) is deferred with graphs, but the
bucket field is already carried in the step and covered by its digest, so adoption needs no
protocol change later.

One row per rank per round in this phase: row-batched multi-stream forwards (dsv41's RoundRunner
shape) need per-row cache planes with row-invariant kernels — phase B. The round builder still
round-robins each rank's streams, so admission, ledgers and payload shapes do not change when
batching lands.
"""
from __future__ import annotations

import hashlib
import json
import socket
import time


class OutOfStep(RuntimeError):
    """The ranks disagree about this step: it fails on every rank at the same point (multi.py's)."""


# -- ledgers and placement (rank 0's view of the four pools, D2 §5) -------------------------------------------------

class Ledgers:
    """One ledger per rank: free KV tokens, queued prefill tokens, and live rows this round."""

    def __init__(self, pool_tokens: int, world: int) -> None:
        self.pool, self.world = int(pool_tokens), world
        self.free = [int(pool_tokens) for _ in range(world)]
        self.queued = [0] * world
        self.rows = [0] * world

    def reserve(self, rank: int, tokens: int) -> None:
        if self.free[rank] < tokens:
            raise RuntimeError(f"rank {rank}: {tokens} tokens do not fit its ledger ({self.free[rank]} free)")
        self.free[rank] -= tokens

    def fill(self, rank: int, tokens: int) -> None:
        self.queued[rank] = max(0, self.queued[rank] - tokens)

    def release(self, rank: int, tokens: int) -> None:
        self.free[rank] = min(self.pool, self.free[rank] + tokens)


class Placement:
    """Owner-rank assignment at admission (D2 §5): least-loaded first, sticky by session key."""

    def __init__(self, ledgers: Ledgers, *, sticky: bool = True) -> None:
        self.ledgers = ledgers
        self.sticky = sticky
        self.sessions: dict[str, int] = {}

    def assign(self, tokens: int, session: str = "") -> int:
        if self.sticky and session:
            r = self.sessions.get(session)
            if r is not None and self.ledgers.free[r] >= tokens:
                self.ledgers.queued[r] += tokens
                return r                                       # pinned: the kept-prompt cache lives there
        order = sorted(range(self.ledgers.world), key=lambda r: (-self.ledgers.free[r],
                                                                 self.ledgers.queued[r] + self.ledgers.rows[r]))
        for r in order:
            if self.ledgers.free[r] >= tokens:
                if self.sticky and session:
                    self.sessions[session] = r                 # re-pinned if the sticky rank ran dry
                self.ledgers.queued[r] += tokens
                return r
        raise RuntimeError(f"no rank holds {tokens} free KV tokens (max {self.ledgers.free[order[0]]})")


class Extents:
    """One rank's pool rows as free runs, first fit (rank 0 keeps one per rank and ships the bases): a stream's KV is one
    run of rows, so the attention kernels read a plain row number. Freed runs merge with their neighbours."""

    def __init__(self, rows: int) -> None:
        self.runs: list[list[int]] = [[0, int(rows)]] if rows else []

    def take(self, n: int) -> int | None:
        for i, run in enumerate(self.runs):
            if run[1] >= n:
                base = run[0]
                run[0] += n
                run[1] -= n
                if not run[1]:
                    del self.runs[i]
                return base
        return None

    def give(self, base: int, n: int) -> None:
        merged: list[list[int]] = []
        for b, c in sorted(self.runs + [[base, n]]):
            if merged and merged[-1][0] + merged[-1][1] == b:
                merged[-1][1] += c
            else:
                merged.append([b, c])
        self.runs = merged


# -- the step message and the digest agreement ----------------------------------------------------------------------

def pack_step(seq: int, bucket: int, rows_by_rank: list[list]) -> list:
    """``["step", seq, bucket, rows_by_rank]``; a row is ``[sid, kind, pos, tokens...]`` with kind
    0 = decode (one token) and 1 = prefill (a chunk). JSON like multi.py's ops: the digest is the
    sha256 of the encoding, exactly the ``_agree`` shape."""

    return ["step", int(seq), int(bucket), rows_by_rank]


def continues(op: list) -> bool:
    """A slice past a round's first: its tokens went with the first and the socket keeps the order, so nothing to agree on, and
    no reason to drain the GPU between two slices."""

    return op[0] == "cp" and len(op) > 7 and not op[5]


def step_digest(op: list) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(op).encode()).digest()[:8], "big")


_AGREE_SEEN: set = set()          # per-process memo for the one-line-per-step digest trace


def agree(comm, what: str, op: list, device="cpu") -> None:
    """One small all-gather checks the step (its row sets, its bucket — D2 O4) before any rank
    enters the step's model collectives (multi.py:486 ``_agree``; ``OutOfStep`` lands on all).

    DIGEST TRACE (m2b5) said what the m2b8 fix turns on: the load4 trace showed every rank
    digesting the SAME op (``admission digest=60624ddfd54c958a`` on 0/1/2/3) and the gather
    STILL mismatched — the bug was the READ, not the inputs. Two compound faults, both gone:
    the buffers were CPU tensors handed to a CUDA collective (host pointers into
    ``ncclAllGather`` — GB10's coherent CPU memory makes that "work" instead of crash, which
    hid it), and the CPU compared ``every`` with NO barrier while the async NCCL kernel was
    still filling it — a literal stale frame: identical ops declared different. Buffers now
    live on the collective's own device and the compare rides the stream behind the kernel:
    ``gather_done()`` (NCCL comm, = torch.cuda.synchronize) blocks until the kernel landed, and
    ``.item()`` is a stream-ordered D2H copy anyway. CPU lane (EmuComm) has neither, and needs
    neither: its fills are synchronous on the calling thread — which is why this race was
    CPU-invisible until the AsyncKernelComm repro (tests/test_ml4_engine.py, m2b8). The rank-0
    ``_bcast_lock`` (m2b5) stays: it makes the CALL ORDER match the socket order; this makes
    the READ see the write."""

    if comm is None or comm.world < 2:
        return
    import os
    import torch

    dgst = step_digest(op) & 0x7FFF_FFFF_FFFF_FFFF          # int64-safe (multi.py splits pieces; one word is enough)
    if os.environ.get("TF_ML4_AGREE_LOG", "1") == "1":
        key = (what, dgst)
        if key not in _AGREE_SEEN:
            _AGREE_SEEN.add(key)
            if len(_AGREE_SEEN) <= 64:                      # a trace, not a log bomb
                print(f"[ml4-agree] rank {comm.rank}: {what} seq={op[1] if what == 'step' else '-'} "
                      f"digest={dgst:016x}", flush=True)
    mine = torch.tensor([dgst], dtype=torch.int64, device=device)
    every = torch.empty((comm.world,), dtype=torch.int64, device=device)
    with trace_tag(comm, f"agree:{what}"):                    # m2b11: trace seqs name the op's gather
        comm.all_gather(mine, every)
    done = getattr(comm, "gather_done", None)               # NCCL: block until the kernel landed (m2b8)
    if done is not None:
        done()
    chk = getattr(comm, "check", None)
    if chk is not None:                                     # a transport failure must not read as agreement
        chk()
    if int((every != every[0]).sum().item()):               # .item(): stream-ordered read, never a raw one
        raise OutOfStep(f"the ranks planned different {what}s; its requests fail, serving goes on")


class TraceComm:
    """m2b11: stamp EVERY collective with a per-rank monotonic sequence
    number and log entry/exit, so the next node run's four logs diff line-for-line and name
    exactly WHERE the ranks stop pairing (load5: rank 0 sat in one layer's counts all-gather while
    a follower waited in the previous layer's MoE reply exchange — invisible without the numbers).
    Pure wrapper: rank/world/store forward, every Comm method delegates to ``base``, nothing here
    changes ordering or sizes. On by default for world>1 (``TF_ML4_TRACE=0`` off); full lines for
    the first ``TF_ML4_TRACE_MAX`` (4000) collectives per rank, then one every 4096th — the load5
    deadlocks all fired inside the first forward, well inside the cap.

    Line shape (greppable, one per event):
        [ml4-trace] r2 #137 exchange p0 out=[8 8 48] in=[0 0 0] a2a(0,2)reply
        [ml4-trace] r2 #137 exchange done
    The exchange line carries per-leg element counts, so a pairing mismatch — the load5 root cause
    — reads straight off the two lines a pair logs for the same collective."""

    def __init__(self, base, log=print) -> None:
        import os

        self.base, self.log = base, log
        self.rank, self.world = base.rank, base.world
        self.store = getattr(base, "store", None)
        self.tag = ""
        self._n = 0
        self._cap = int(os.environ.get("TF_ML4_TRACE_MAX") or 4000)

    def __getattr__(self, name):                               # gather_done, abort, async_error, ...
        return getattr(self.base, name)

    def _line(self, text: str) -> None:
        if self._n <= self._cap or self._n % 4096 == 0:
            self.log(f"[ml4-trace] r{self.rank} #{self._n} {text}")

    def trace(self, msg: str) -> None:                         # non-collective markers (combine, probe)
        self.log(f"[ml4-trace] r{self.rank} | {msg}")

    def _run(self, kind: str, detail: str, fn, *args) -> None:
        self._n += 1
        self._line(f"{kind} {detail}{' ' + self.tag if self.tag else ''}")
        try:
            fn(*args)
        except BaseException as exc:
            self._line(f"{kind} RAISED {type(exc).__name__}: {exc}")
            raise
        self._line(f"{kind} done")

    def all_gather(self, send, recv) -> None:
        self._run("all_gather", f"n={send.numel()}", self.base.all_gather, send, recv)

    def exchange(self, sends, recvs, peer) -> None:
        out = " ".join(str(t.numel()) for t in sends)
        inn = " ".join(str(t.numel()) for t in recvs)
        self._run("exchange", f"p{peer} out=[{out}] in=[{inn}]", self.base.exchange, sends, recvs, peer)

    def barrier(self) -> None:
        self._run("barrier", "", self.base.barrier)


class trace_tag:
    """Label the collectives entered inside this block (TraceComm output only; a no-op on an
    untraced comm, so call sites never need an env or type check). Walks the wrapper chain
    (AsyncKernelComm/Transport wrap TraceComm with their own __getattr__) to the tracer itself."""

    def __init__(self, comm, label: str) -> None:
        self.comm, self.label = comm, label

    def __enter__(self):
        c = self.comm
        while c is not None and not isinstance(c, TraceComm):
            c = getattr(c, "base", None)
        self.target = c
        if c is not None:
            self.prev, c.tag = c.tag, self.label
        return self

    def __exit__(self, *exc):
        if self.target is not None:
            self.target.tag = self.prev
        return False


def rows_for(op: list, rank: int) -> list:
    """Rank r's rows: ONLY its own (D2 §1). Rank 0 holds the other ranks' sets to ship them; a
    follower must never execute them."""

    return list(op[3][rank]) if 0 <= rank < len(op[3]) else []


# -- the link (multi.py's Link shape + a follower -> rank 0 back channel) ------------------------------------------

class Link:
    """Rank 0's steps to every follower, in order, one TCP socket each (port through the rendezvous
    store); followers answer sampled tokens on the same socket — DP4 needs the tokens back, TP
    mirrored them."""

    KEY = "tensorfold/ml4/multi/port"

    def __init__(self, store, *, rank: int, world: int, host: str) -> None:
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

    @staticmethod
    def _read(sock, n: int) -> bytes | None:
        out = b""
        while len(out) < n:
            chunk = sock.recv(n - len(out))
            if not chunk:
                return None
            out += chunk
        return out

    @staticmethod
    def _write(sock, op: list) -> None:
        data = json.dumps(op).encode()
        sock.sendall(len(data).to_bytes(4, "big") + data)

    def send(self, op: list) -> None:
        if not self.socks:                                     # rank 0: accept the followers on first use
            got = {}
            while len(got) < self.world - 1:
                sock, _ = self.server.accept()
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                got[int.from_bytes(self._read(sock, 4), "big")] = sock
            self.socks = [got[r] for r in sorted(got)]
        for sock in self.socks:
            self._write(sock, op)

    def receive(self) -> list | None:                          # a follower: rank 0's next step
        return self._recv(self.socks[0])

    def send_back(self, op: list) -> None:                     # a follower: its sampled tokens
        self._write(self.socks[0], op)

    def receive_back(self, peer: int) -> list | None:          # rank 0, in rank order: no interleaving
        return self._recv(self.socks[peer - 1])

    def _recv(self, sock) -> list | None:
        head = self._read(sock, 4)
        body = None if head is None else self._read(sock, int.from_bytes(head, "big"))
        return None if body is None else json.loads(body)


class EmuLink:
    """In-process link for the CPU lane tests: queues instead of sockets, same op protocol. One
    EmuLink is shared; ``rank`` picks whose queues are read/written."""

    def __init__(self, world: int, rank: int = 0) -> None:
        import queue

        self.world, self.rank = world, rank
        self.to = [queue.Queue() for _ in range(world)]         # rank 0 -> follower steps
        self.back = [queue.Queue() for _ in range(world)]       # follower -> rank 0 tokens

    def fork(self, rank: int) -> "EmuLink":
        twin = EmuLink.__new__(EmuLink)
        twin.world, twin.rank, twin.to, twin.back = self.world, rank, self.to, self.back
        return twin

    def send(self, op: list) -> None:
        for r in range(1, self.world):
            self.to[r].put(op)

    def send_back(self, op: list) -> None:
        self.back[self.rank].put(op)

    def receive(self) -> list:
        return self.to[self.rank].get()

    def receive_back(self, peer: int) -> list:
        return self.back[peer].get()


# -- the round builder (D2 §4 interleave rules) ----------------------------------------------------------------------

class Stream:
    """One served sequence on the lane: its prompt, owner rank, cache position bookkeeping and output."""

    def __init__(self, sid: int, prompt: list[int], count: int, owner: int, sampling=None,
                 on_tokens=None, stop_eos: bool = True) -> None:
        self.sid, self.prompt, self.count, self.owner = sid, list(prompt), int(count), owner
        self.sampling, self.on_tokens, self.stop_eos = sampling, on_tokens, stop_eos
        self.filled = 0                       # prompt tokens through the model
        self.pending: int | None = None       # token the next decode row feeds
        self.out: list[int] = []
        self.done = False
        self.place: list | None = None        # [world, block, bases by rank, rows]: its rows in the pools
        self.cp = False                       # striped over the group: its rounds are ["cp", ...] ops, owner -1
        self.flight: list | None = None       # [start, n, last] of a round between its layers
        self.left = 0                         # ... and the layers it has to go
        self.started = 0.0                    # when that round began
        self.box = None                       # served: its tokens go here, its request's thread takes them out
        self.halt = False                     # ... and says here that the client is gone or a stop string came

    @property
    def needs_fill(self) -> bool:
        return self.filled < len(self.prompt)


class RoundBuilder:
    """Rank 0's round policy (D2 §4): at most ONE big prefill chunk per round globally (a 2048-row
    prefill on one rank stalls the lane's a2a 10-30x for everyone else), decode rows keep flowing
    beside it; each rank gets at most one row this phase (row-batched rounds are phase B). The
    bucket is the max row length padded to 16 — graphs are later, but the field and its digest
    coverage exist now (D2 O4)."""

    def __init__(self, chunk: int = 512) -> None:
        self.chunk = int(chunk)
        self.streams: dict[int, Stream] = {}
        self.turn: dict[int, int] = {}        # rank -> next stream index among that rank's decoders
        self.next_row = 0                     # tp_rows' place in the ready streams

    def add(self, stream: Stream) -> None:
        self.streams[stream.sid] = stream

    def live(self) -> list[Stream]:
        return [s for s in self.streams.values() if not s.done]

    def tp_rows(self, cap: int) -> list | None:
        """Up to ``cap`` decode rows ``[sid, position, token]`` of striped streams, taking turns past the cap."""

        ready = [s for s in self.live() if s.cp and not s.needs_fill and s.pending is not None]
        if not ready:
            return None
        self.next_row = self.next_row % len(ready)
        take = (ready + ready)[self.next_row:self.next_row + min(cap, len(ready))]
        self.next_row += len(take)
        return [[s.sid, len(s.prompt) + len(s.out) - 1, s.pending] for s in take]

    def cp_slice(self, width: int, layers: int, total: int) -> list | None:
        """The next ``layers`` layers of a prompt's round, ``["cp", 0, sid, start, n, tokens, last, layers]``, the shortest
        prompt first: a short prompt overtakes a long one between two of its slices. A round's tokens ride its first slice."""

        filling = [s for s in self.live() if s.cp and (s.needs_fill or s.left)]
        if not filling:
            return None
        s = min(filling, key=lambda s: len(s.prompt) - s.filled + (s.flight[1] if s.left else 0))
        toks: list = []
        if not s.left:
            take = min(width - s.filled % width, len(s.prompt) - s.filled)
            if len(s.prompt) - s.filled - take < width // 8:     # a short last round would cost nearly a whole one: its experts' weights
                take = len(s.prompt) - s.filled
            start = s.filled
            s.filled += take
            s.flight = [start, take, start + take - 1 if s.filled == len(s.prompt) else None]
            s.left = total
            toks = s.prompt[start:start + take]
        k = min(max(layers, layers * width // s.flight[1]), s.left)   # a narrow round takes as long in more layers a slice
        s.left -= k
        return ["cp", 0, s.sid, s.flight[0], s.flight[1], toks, s.flight[2], k]

    def cp_round(self, width: int) -> list | None:
        """The next round of the striped stream, ``["cp", 0, sid, start, n, tokens, last]`` (``seq`` filled in by the
        caller): a prefill round runs to the next multiple of ``width``, so every rank holds at most its block of
        it; ``last`` is the position whose logits are wanted, None mid-prompt."""

        for s in self.live():
            if not s.cp:
                continue
            if s.needs_fill:
                take = min(width - s.filled % width, len(s.prompt) - s.filled)
                start = s.filled
                s.filled += take
                last = start + take - 1 if s.filled == len(s.prompt) else None
                return ["cp", 0, s.sid, start, take, s.prompt[start:start + take], last]
            if s.pending is not None:
                pos = len(s.prompt) + len(s.out) - 1
                return ["cp", 0, s.sid, pos, 1, [s.pending], pos]
        return None

    def round_rows(self, world: int) -> list[list] | None:
        rows: list[list] = [[] for _ in range(world)]
        live = [s for s in self.live() if not s.cp]
        if not live:
            return None
        filling = [s for s in live if s.needs_fill]
        if filling:                                          # one prefill chunk per round, lowest sid
            s = filling[0]
            take = min(self.chunk, len(s.prompt) - s.filled)
            rows[s.owner] = [[s.sid, 1, s.filled, list(s.prompt[s.filled:s.filled + take])]]
            s.filled += take
            return rows
        for r in range(world):                               # decode rows: round-robin per rank
            decoders = [s for s in live if s.owner == r and s.pending is not None]
            if not decoders:
                continue
            i = self.turn.get(r, 0) % len(decoders)
            self.turn[r] = i + 1
            s = decoders[i]
            rows[r] = [[s.sid, 0, len(s.prompt) + len(s.out) - 1, [s.pending]]]   # position of the fed token
        return rows if any(rows) else None

    @staticmethod
    def bucket_of(rows_by_rank: list[list]) -> int:
        n = max((len(row[3]) for rows in rows_by_rank for row in rows), default=0)   # rows of the round's forward
        return -(-n // 16) * 16
