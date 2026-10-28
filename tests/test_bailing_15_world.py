"""Ling-3.1-flash's engine past two ranks, and NCCL without graph-mixing support, on the CPU.

- The request-level exchanges run at any world: ``_share`` hands rank 0's list to every rank, ``_gather_ints`` and
  ``_gather_floats`` return every rank's list in rank order, ``_slowest`` (the calibration's) takes each value's
  largest over the ranks, and the doorbell wakes every follower, a late one too. The fake all-gather keeps
  comm.py's check, so a buffer sized for two ranks fails at four as NCCL's does.
- A request's header carries the answer reserve's stop (ThinkStop) and whether its prompt continues a reply, so every
  rank ends the run at the same round and keeps the same snapshots; the decode loops end with the round that holds
  the stop's token, every token up to it the unstopped run's.
- The engine opens NCCL with NCCL_GRAPH_MIXING_SUPPORT=0, which does not cover an uncaptured collective launched
  while a graph launch on the same communicator is outstanding, whatever the stream order. Through prefill, serial
  and MTP-drafted decode (some row counts captured, some eager), every uncaptured collective comes after a
  synchronize of every earlier replay; one rank (no communicator) gets no synchronize beyond the loops' own.

Run: python -m pytest tests/test_bailing_15_world.py -q
CPU only (torch for CPU tensors).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")

from tensorfold.families.bailing_v3.cuda import decode  # noqa: E402
from tensorfold.families.bailing_v3.cuda.engine import BailingEngine  # noqa: E402

# the engine's exchanges name device="cuda"; these tensors stay on the CPU
CPU = SimpleNamespace(int32=torch.int32, float32=torch.float32,
                      tensor=lambda data, dtype=None, device=None: torch.tensor(data, dtype=dtype),
                      empty=lambda shape, dtype=None, device=None: torch.empty(shape, dtype=dtype),
                      zeros=lambda shape, dtype=None, device=None: torch.zeros(shape, dtype=dtype))


# -- the request-level exchanges ------------------------------------------------------------------------------------
class Exchange:
    """An all-gather between threads, one a rank: every call returns when every rank has made it."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.barrier = threading.Barrier(world, timeout=10)
        self.slots: list = [None] * world


class RankComm:
    def __init__(self, ex: Exchange, rank: int) -> None:
        self.ex, self.rank, self.world = ex, rank, ex.world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:     # comm.py's check
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        self.ex.slots[self.rank] = send.reshape(-1).clone()
        self.ex.barrier.wait()
        recv.copy_(torch.cat(self.ex.slots))
        self.ex.barrier.wait()


def rank_engine(rank: int, world: int, comm) -> BailingEngine:
    e = object.__new__(BailingEngine)
    e.torch, e.rank, e.world, e.comm = CPU, rank, world, comm
    return e


def on_ranks(world: int, fn) -> list:
    """fn(engine) on every rank at once; each rank's result, in rank order."""

    ex = Exchange(world)
    results: list = [None] * world
    errors: list = []

    def run(r: int) -> None:
        try:
            results[r] = fn(rank_engine(r, world, RankComm(ex, r)))
        except BaseException as exc:          # noqa: BLE001  (reraised below)
            errors.append(exc)
            ex.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    if errors:
        raise errors[0]
    assert not any(t.is_alive() for t in threads), "a rank never returned"
    return results


@pytest.mark.parametrize("world", [2, 4])
def test_share_hands_rank_0s_list_to_every_rank(world):
    values = [5, 6, 7, 2 ** 31 - 1, 0]
    assert on_ranks(world, lambda e: e._share(values if e.rank == 0 else None)) == [values] * world


@pytest.mark.parametrize("world", [2, 4])
def test_the_gathers_return_every_rank_in_rank_order(world):
    ints = on_ranks(world, lambda e: e._gather_ints([e.rank, 10 + e.rank, -e.rank]))
    assert ints == [[[r, 10 + r, -r] for r in range(world)]] * world
    third = torch.tensor(1 / 3, dtype=torch.float32).item()
    floats = on_ranks(world, lambda e: e._gather_floats([e.rank + 0.5, 1 / 3]))
    assert floats == [[[r + 0.5, third] for r in range(world)]] * world


@pytest.mark.parametrize("world", [2, 4])
def test_the_calibration_takes_the_slowest_rank(world):
    assert on_ranks(world, lambda e: e._slowest([1.0 + e.rank, 9.0 - e.rank])) == [[float(world), 9.0]] * world


@pytest.mark.parametrize("world", [2, 4])
def test_equal_source_fingerprints_pass_at_every_world(world):
    on_ranks(world, lambda e: e._check_source(0x1A2B3C4))


@pytest.mark.parametrize("world", [2, 4])
def test_one_rank_on_a_stale_tree_raises_on_every_rank(world):
    fingerprints = [0x1A2B3C4, 0x0BADF00, 0x1A2B3C4, 0x1A2B3C4]          # rank 1 staged a different commit

    def attempt(e) -> str | None:
        try:
            e._check_source(fingerprints[e.rank])
        except RuntimeError as exc:
            return str(exc)
        return None

    messages = on_ranks(world, attempt)
    assert all(message is not None for message in messages), f"a rank did not raise: {messages}"
    for message in messages:                    # every rank names every rank's fingerprint
        assert f"rank 0 {0x1A2B3C4:07x}" in message and f"rank 1 {0x0BADF00:07x}" in message
        assert "stage the same commit on every node" in message


def test_the_source_fingerprint_fits_the_int32_gather_and_is_stable():
    from tensorfold.families.bailing_v3.cuda.engine import source_fingerprint

    assert 0 <= source_fingerprint() < 2 ** 28
    assert source_fingerprint() == source_fingerprint()


def test_the_source_fingerprint_sees_every_source_kind_and_nothing_a_run_writes(tmp_path):
    from tensorfold.families.bailing_v3.cuda.engine import source_fingerprint

    files = {"engine.py": "a", "kernels/k.cu": "b", "kernels/k.cuh": "c", "kernels/k.h": "d", "exl3/linear.cpp": "e"}
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    base = source_fingerprint(tmp_path)
    for name in files:                                     # an extension's C++ host source counts like the rest
        (tmp_path / name).write_text(files[name] + " changed")
        assert source_fingerprint(tmp_path) != base, name
        (tmp_path / name).write_text(files[name])
    assert source_fingerprint(tmp_path) == base
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "engine.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "__pycache__" / "stale.py").write_text("x")
    (tmp_path / "NOTES.md").write_text("x")
    assert source_fingerprint(tmp_path) == base


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("fits", [100_000, 10 ** 9])
def test_admission_without_a_context_gives_every_rank_the_window_one_rank_gets(world, fits, tmp_path, monkeypatch):
    """No --context asks admission for no window: every rank's status row fits the int32 all-gather, and every rank
    gets the window one rank alone got when it asked for the dense limit (the native one, capped by what fits). The
    dense limit (1 << 62) asked for past one rank overflows that gather (K3 on two GB10s, 2026-10-16)."""

    from tensorfold.cuda import capacity
    from tensorfold.families.bailing_v3.cuda.engine import requested_window
    from tensorfold.families.bailing_v3.cuda.weights import Config

    (tmp_path / "config.json").write_text((REPO / "tests" / "fixtures" / "ling31" / "config.json").read_text())
    dense = Config.read(tmp_path).dense_limit
    geometry = capacity.Geometry(lambda slots: slots * 2 ** 20, 64)
    weights = capacity.Weights(8 * capacity.GIB, 0)
    monkeypatch.setattr(capacity, "estimate_weights", lambda *args, **kwargs: weights)
    monkeypatch.setattr(capacity, "available_bytes", lambda torch: weights.resident + geometry.needed(fits))
    monkeypatch.setattr(capacity, "page_room", lambda torch: None)
    monkeypatch.setattr(capacity, "unified", lambda torch: True)

    def window(requested, e=None):
        kw = {"rank": e.rank, "world": e.world, "gather": e._gather_ints} if e is not None else {}
        return capacity.admit(tmp_path, requested, False, None, geometry, lambda name, info: (0, 0),
                              **kw)["context_window"]

    alone = window(dense)
    assert alone == min(fits, 262144)
    assert on_ranks(world, lambda e: window(requested_window(None, False), e)) == [alone] * world
    with pytest.raises(RuntimeError, match="overflow"):
        on_ranks(world, lambda e: window(dense, e))


class Store:
    """The rendezvous store as the doorbell uses it; a wait that times out raises, and the doorbell waits again."""

    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.cv = threading.Condition()

    def set(self, key: str, value: bytes) -> None:
        with self.cv:
            self.keys.add(key)
            self.cv.notify_all()

    def wait(self, keys: list[str], timeout) -> None:
        with self.cv:
            if not self.cv.wait_for(lambda: all(k in self.keys for k in keys), timeout=0.05):
                raise RuntimeError("Socket Timeout")

    def delete_key(self, key: str) -> None:
        with self.cv:
            self.keys.discard(key)


def test_the_doorbell_wakes_every_follower_a_late_one_too():
    world, store = 4, Store()
    engines = [rank_engine(r, world, SimpleNamespace(store=store)) for r in range(world)]
    woke: list[int] = []

    def follow(r: int) -> None:
        engines[r]._await_bell()
        woke.append(r)

    def wake(ranks, late=()) -> None:
        early = [threading.Thread(target=follow, args=(r,), daemon=True) for r in ranks]
        for t in early:
            t.start()
        engines[0]._ring()
        for t in early:
            t.join(5)
        for r in late:                  # still finishing the last request when the others woke
            t = threading.Thread(target=follow, args=(r,), daemon=True)
            t.start()
            t.join(2)

    wake((1, 2), late=(3,))
    assert sorted(woke) == [1, 2, 3], f"woke {sorted(woke)}: a follower missed the bell"
    wake((1, 2, 3))
    assert sorted(woke) == [1, 1, 2, 2, 3, 3]
    assert not store.keys               # each follower deleted its own key


class _Followed(Exception):
    pass


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("think_stop, continued", [((24, 90), False), ((7, 90), True), (None, True), (None, False)])
def test_every_rank_ends_the_run_at_the_cut_and_keeps_the_same_snapshot(world, think_stop, continued):
    def fn(e):
        e.limit, e.serial_only, e.cache, e.request, runs = 10_000, True, [], threading.local(), []

        def run(*args):
            runs.append(args[-2:])
            if e.rank:
                raise _Followed                 # out of follow's loop after one request
            return {}

        e._run = run
        if e.rank == 0:
            e.generate([5, 6, 7], 48, None, lambda new: None, think_stop=think_stop, continued=continued)
        else:
            with pytest.raises(_Followed):
                e.follow()
        (think, behind), = runs
        return (think.budget, think.think_end, think.seen, think.open) if think else None, behind

    want = ((*think_stop, 0, True) if think_stop else None, continued)
    assert on_ranks(world, fn) == [want] * world


# -- graph replays and uncaptured collectives --------------------------------------------------------------------------
V = 64              # this rank's vocabulary columns


class Log(list):
    def check(self) -> None:
        """No uncaptured collective while a replay is outstanding (not yet followed by a synchronize)."""

        outstanding = False
        for i, event in enumerate(self):
            if event == "replay":
                outstanding = True
            elif event == "sync":
                outstanding = False
            elif event == "collective":
                assert not outstanding, f"an uncaptured collective after an unsettled replay at event {i}: {self}"


class Graph:
    def __init__(self, log: Log) -> None:
        self.log = log

    def replay(self) -> None:
        self.log.append("replay")


class FakeState:
    def __init__(self) -> None:
        self.pos = self.parity = self.mtp_len = self.mtp_drafted = 0
        self.cur: list[int] = []

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n

    def set_pos(self, n: int) -> None:
        self.pos = n

    def reset(self) -> None:
        self.pos = self.mtp_len = self.mtp_drafted = 0


def fake_engine(monkeypatch, *, world: int, main_rows=(1, 2, 4), mtp_rows=(1,), mtp_step=8) -> tuple:
    """An Engine whose kernels log: captured shapes replay, the rest compute eagerly (an uncaptured collective)."""

    log = Log()
    gen = torch.Generator().manual_seed(0)

    def buffers(rows: int, prefill: bool = False):
        return SimpleNamespace(logits=torch.randn(rows, V, generator=gen), fnormed=torch.randn(rows, 16, generator=gen),
                               rows=rows, prefill=prefill)

    def all_gather(send, recv):
        log.append("collective")
        recv.copy_(send.repeat(world))  # every rank's shard alike: the same candidates

    def compute(w, st, b, R, nch=None, host_pos=None, cut=None):
        log.append("collective")
        return b.logits[:1] if b.prefill else b.logits[:R]

    def mtp_compute(w, st, b, n, nch=None, host_pos=None):
        log.append("collective")
        return b.logits[:1]

    def mtp_forward(w, st, b, next_tokens, hidden):
        log.append("collective")

    def commit(w, st, b, R, keep):
        st.pos += keep

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: log.append("sync"))
    monkeypatch.setattr(decode, "stage", lambda w, st, b, tokens: len(tokens))
    monkeypatch.setattr(decode, "mtp_stage", lambda w, st, b, next_tokens, hidden: len(next_tokens))
    monkeypatch.setattr(decode, "chunks_for", lambda st, R: 1)
    monkeypatch.setattr(decode, "compute", compute)
    monkeypatch.setattr(decode, "mtp_compute", mtp_compute)
    monkeypatch.setattr(decode, "mtp_forward", mtp_forward)
    monkeypatch.setattr(decode, "commit", commit)

    e = object.__new__(decode.Engine)
    comm = SimpleNamespace(all_gather=all_gather) if world > 1 else None
    e.w = SimpleNamespace(comm=comm, world=world, vocab_offset=0, mtp=object(),
                          cfg=SimpleNamespace(dense_limit=1 << 30, eos=()))
    e.st = FakeState()
    e.rows, e.prefill_rows = 8, 16
    e.buf, e.pbuf, e.mbuf = buffers(8), buffers(16, prefill=True), buffers(mtp_step)
    e.last_hidden, e.constraint, e.window, e.draft_n = None, None, None, V
    e.replays = {"main": 0, "mtp": 0, "eager": 0}
    e.replayed = False
    e.graphs = SimpleNamespace(main={(R, p): Graph(log) for R in main_rows for p in (0, 1)},
                               mtp={n: Graph(log) for n in mtp_rows})
    return e, log


@pytest.mark.parametrize("mtp_step", [8, 2])
def test_every_uncaptured_collective_follows_a_settled_replay(monkeypatch, mtp_step):
    e, log = fake_engine(monkeypatch, world=2, mtp_step=mtp_step)
    first = decode.prefill(e, list(range(1, 40)), None, mtp=True)
    res = decode.mtp_decode(e, first, 24, None, policy=decode.DepthPolicy(3, fixed=True))
    assert res.rounds and res.drafted
    serial = decode.serial_decode(e, res.tokens[-1], 6, None)
    assert serial.rounds == 5
    e.forward([1])                      # a replay, then the uncaptured paths straight after it
    e.forward([1, 2, 3, 4, 5])
    e.mtp([1], e.mbuf.fnormed[:1])
    decode.prefill(e, [7, 8, 9], None, mtp=True)
    assert e.replays["main"] and e.replays["mtp"] and e.replays["eager"]       # both paths ran
    log.check()


@pytest.mark.parametrize("drafted", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 5, 9, 17, 23, 24, 40])
def test_the_decode_loops_end_with_the_round_that_holds_the_cut(monkeypatch, budget, drafted):
    from tensorfold.engine.call_gate import ThinkStop

    def run(think=None):
        e, log = fake_engine(monkeypatch, world=2)
        first = decode.prefill(e, list(range(1, 40)), None, mtp=True)
        ends = [1]                                  # tokens after each round; the first token is a round of its own
        if think is not None and think.hit([first]):
            return [first], ends
        feed = lambda new: ends.append(ends[-1] + len(new))  # noqa: E731
        if drafted:
            res = decode.mtp_decode(e, first, 24, None, policy=decode.DepthPolicy(3, fixed=True), on_tokens=feed,
                                    think=think)
        else:
            res = decode.serial_decode(e, first, 24, None, on_tokens=feed, think=think)
        log.check()
        return res.tokens, ends

    full, ends = run()
    assert len(full) == 24
    for end in (-1, *full[:3]):
        closes = end in full[:budget - 1]         # the reply closes its own block before the cut: no stop
        at = next((n for n in ends if n >= budget), None)
        stopped, _ = run(ThinkStop(budget, end))
        assert stopped == (full if closes or at is None else full[:at]), (end, ends)


def test_one_rank_gets_no_extra_synchronize(monkeypatch):
    e, log = fake_engine(monkeypatch, world=1)
    first = decode.prefill(e, list(range(1, 40)), None, mtp=True)
    log.clear()
    res = decode.mtp_decode(e, first, 24, None, policy=decode.DepthPolicy(3, fixed=True))
    assert e.replays["mtp"] and res.rounds
    assert log.count("sync") == 2 + res.rounds          # the loop's own: start, end, one a verify round
    log.clear()
    serial = decode.serial_decode(e, res.tokens[-1], 6, None)
    assert log.count("sync") == 2 + serial.rounds


def test_the_engine_opens_nccl_without_graph_mixing_support_and_settles_the_score():
    source = (SRC / "tensorfold" / "families" / "bailing_v3" / "cuda" / "engine.py").read_text()
    at = source.index('os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")')
    assert at < source.index("open_comm(rank, world, master, port)")
    assert source.index("self.e.settle()") < source.index("local = prompt_logits(self.e, prompt)")
