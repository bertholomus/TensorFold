"""CPU gates for the ML4 phase-2b serving lane: the EP4 all-to-all, DP4 placement/steps, the digest
agreement and a full four-"rank" lane simulation. No CUDA, no GPU, no NCCL (note D2's rules for
this branch): the fake Comm below speaks the same all_gather/exchange protocol as
``tensorfold.cuda.comm.Comm``, so the collective shapes the real NCCL Comm will see are exercised.

Run directly, no pytest (host-test venv: Python 3.9 + torch CPU):

    $ML4_HOST_TESTS/.venv/bin/python tests/test_ml4_engine.py

Contracts proven here:
* bit-exactness — the CPU EP4 combine reproduces the world=1 all-experts-local MoE fp32 exactly
  (a token routed to experts on three ranks equals the local run, same slot add order);
* payload ownership — rank r's step payload carries only its own rows, and rows_for() reads only
  its own;
* placement — least-loaded by free_pool_tokens with queued+rows secondary, sticky by session;
* the digest — identical ops agree, a differing bucket field raises OutOfStep on every rank;
* the lane — a four-thread world=4 engine (rank 0 + three followers over fake comms) generates the
  same greedy tokens as a lone world=1 engine on the same weights.
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
SRC = HERE.parents[0] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import test_ml4_model as T2                                                    # noqa: E402
import torch                                                                   # noqa: E402
from refs import tiny_model as TM                                              # noqa: E402

from tensorfold.families.mistral_large4.cuda import a2a as A2A                 # noqa: E402
from tensorfold.families.mistral_large4.cuda import dp4 as DP                  # noqa: E402
from tensorfold.families.mistral_large4.cuda import engine as EN               # noqa: E402
from tensorfold.families.mistral_large4.cuda import model as M                 # noqa: E402
from tensorfold.families.mistral_large4.cuda.weights import Ml4Weights, local_ids  # noqa: E402

torch.manual_seed(1105)


def feed(into: list):
    """on_tokens feed with the App's contract (cuda/server.py): True stops the stream."""

    def _f(toks):
        into.extend(toks)
        return False

    return _f


# -- the fake transport: the tensorfold.cuda.comm.Comm protocol, in-process, threaded --------------------------------

class _Shared:
    def __init__(self):
        self.cond = threading.Condition()
        self.ag = {"n": 0, "m": 0, "gen": 0, "gen2": 0, "slots": {}}
        self.ex = {}


class EmuComm:
    """In-process Comm: all_gather/exchange with the same semantics as the NCCL Comm —
    two-phase barriers (arrive, copy, drain) so a fast rank can never overwrite a slot a slow
    rank is still reading. Tensors are CPU; the call pattern is what the real Comm will see."""

    def __init__(self, world, rank, shared):
        self.world, self.rank, self._s, self.store = world, rank, shared, None

    def all_gather(self, send, recv):
        n = send.numel()
        c = self._s.cond
        with c:
            st = self._s.ag
            g = st["gen"]
            st["slots"][self.rank] = send.detach().clone()
            st["n"] += 1
            if st["n"] == self.world:
                st["n"] = 0
                st["gen"] += 1
                c.notify_all()
            else:
                while st["gen"] == g:
                    c.wait()
        for r in range(self.world):
            recv[r * n:(r + 1) * n].copy_(st["slots"][r])
        with c:
            g = st["gen2"]
            st["m"] += 1
            if st["m"] == self.world:
                st["m"] = 0
                st["gen2"] += 1
                c.notify_all()
            else:
                while st["gen2"] == g:
                    c.wait()

    def exchange(self, sends, recvs, peer):
        pair = (min(self.rank, peer), max(self.rank, peer))
        c = self._s.cond
        with c:
            st = self._s.ex.setdefault(pair, {"n": 0, "m": 0, "gen": 0, "gen2": 0, "slots": {}})
            g = st["gen"]
            st["slots"][self.rank] = [t.detach().clone() for t in sends]
            st["n"] += 1
            if st["n"] == 2:
                st["n"] = 0
                st["gen"] += 1
                c.notify_all()
            else:
                while st["gen"] == g:
                    c.wait()
        for r, t in zip(recvs, st["slots"][peer]):
            r.copy_(t)
        with c:
            g = st["gen2"]
            st["m"] += 1
            if st["m"] == 2:
                st["m"] = 0
                st["gen2"] += 1
                c.notify_all()
            else:
                while st["gen2"] == g:
                    c.wait()

    def barrier(self):
        pass


def emu_comms(world):
    shared = _Shared()
    return [EmuComm(world, r, shared) for r in range(world)]


def slice_weights(w: Ml4Weights, rank: int, world: int) -> Ml4Weights:
    """A rank's view of a tiny world=1 weights object: whole experts per rank (the loader's
    local_ids split), everything else shared — the EP4 residency shape (D2 §2). Expert triples are
    shared memory (read-only use); only the layer's expert selection changes."""

    layers = []
    for lay in w.layers:
        if lay.experts is None:
            layers.append(lay)
            continue
        pos = {int(g): k for k, g in enumerate(lay.expert_ids)}
        ids = local_ids(rank, world, w.cfg.n_routed)
        trip = [lay.experts[pos[e]] for e in ids] + [lay.experts[-1]]   # shared stays last
        layers.append(type(lay)(**{**lay.__dict__, "experts": trip, "expert_ids": tuple(ids)}))
    return Ml4Weights(cfg=w.cfg, rank=rank, world=world, embed=w.embed, norm=w.norm, alpha=w.alpha,
                      head=w.head, head_bf16=w.head_bf16, vocab_lo=w.vocab_lo, vocab_hi=w.vocab_hi, layers=layers)


# -- test 1: bit-exact MoE across the wire ----------------------------------------------------------------------------

def test_a2a_bit_exact():
    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    m1 = M.Ml4Model(T2.tiny_weights(tc, W))
    lay = next(l for l in m1.w.layers if l.experts is not None)
    xr = torch.randn(6, tc.hidden, dtype=torch.float32)
    ref = m1._moe(lay, xr)

    world, out = 4, torch.zeros_like(ref)
    comms = emu_comms(world)
    models = []
    for r in range(world):
        wr = slice_weights(m1.w, r, world)
        mm = M.Ml4Model(wr)
        mm.comm = comms[r]
        models.append(mm)

    def run(r):
        rows = torch.arange(r, 6, world)
        out[rows] = models[r]._moe(models[r].w.layers[lay.idx], xr[rows])

    ts = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in ts), "the all-to-all deadlocked"
    assert torch.equal(out, ref), "EP4 combine is not bit-exact to all-experts-local fp32"


# -- test 2: a row whose picks land on ONE rank (multi-slot pair, no wire for it) ------------------------------------

def test_a2a_same_rank_picks():
    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    m1 = M.Ml4Model(T2.tiny_weights(tc, W))
    lay = next(l for l in m1.w.layers if l.experts is not None)
    xr = torch.randn(4, tc.hidden, dtype=torch.float32)
    # both picks of every row on experts 0 and 1 -> one rank owns the whole topk
    forced = torch.zeros_like(lay.gate_w)
    forced[0], forced[1] = 10.0 * xr.mean(0), 9.0 * xr.mean(0)
    ref = m1._moe(type(lay)(**{**lay.__dict__, "gate_w": forced}), xr)   # the ref runs the FORCED router
    lay2 = type(lay)(**{**lay.__dict__, "gate_w": forced})
    logits = m1._mat(lay2.gate_w, xr)
    _w, idx = M.route_softmax(logits, tc.top_k_experts)
    assert bool((idx == torch.tensor([0, 1])).all()), "fixture did not force both picks onto experts 0/1"
    world = 4
    comms = emu_comms(world)
    out = torch.zeros_like(ref)

    def run(r):
        wr = slice_weights(m1.w, r, world)
        mm = M.Ml4Model(wr)
        mm.comm = comms[r]
        rows = torch.arange(r, 4, world)
        lay_r = next(l for l in mm.w.layers if l.idx == lay.idx)
        out[rows] = mm._moe(type(lay_r)(**{**lay_r.__dict__, "gate_w": forced}), xr[rows])

    ts = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in ts), "the all-to-all deadlocked"
    assert torch.equal(out, ref), "same-rank multi-slot picks are not bit-exact"


# -- test 3: the full CPU lane — world=4 engines over fakes vs a lone world=1 engine ---------------------------------

def test_lane_simulation():
    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    w1 = T2.tiny_weights(tc, W)
    prompt = [5, 9, 23, 47, 61]
    count = 4
    solo = EN.Ml4Engine(None, rank=0, world=1, device="cpu", weights=w1, pool_tokens=4096, log=lambda _s: None)
    want = []
    solo.generate(prompt, count, on_tokens=feed(want))

    world = 4
    comms = emu_comms(world)
    link = DP.EmuLink(world, 0)
    engines = []
    for r in range(world):
        e = EN.Ml4Engine(None, rank=r, world=world, device="cpu", weights=slice_weights(w1, r, world),
                         comm=comms[r], link=link.fork(r), pool_tokens=4096, log=lambda _s: None)
        engines.append(e)
    got = []
    ts = [threading.Thread(target=engines[r].follow, name=f"rank{r}") for r in range(1, world)]
    for t in ts:
        t.start()
    try:
        engines[0].generate(prompt, count, on_tokens=feed(got))
    finally:
        engines[0].link.send(["stop"])
        for t in ts:
            t.join(timeout=120)
    assert not any(t.is_alive() for t in ts), "the emulated lane deadlocked"
    assert got == want, f"lane tokens {got} != world=1 tokens {want}"


# -- test 4: placement (D2 §5) -----------------------------------------------------------------------------------------

def test_placement():
    led = DP.Ledgers(1000, 4)
    pl = DP.Placement(led)
    r0 = pl.assign(100)
    assert r0 == 0 and led.free[0] == 1000, "assign must not eat free_pool before reserve"
    led.reserve(r0, 100)
    led.queued[r0] += 100
    assert pl.assign(100) == 1, "equal free pools: least queued/rows wins"
    led.reserve(1, 900)
    led.queued[1] += 900
    pinned = pl.assign(50, session="alpha")
    led.reserve(pinned, 50)
    led.queued[pinned] += 50
    led.free[pinned] -= 50
    again = pl.assign(40, session="alpha")
    assert again == pinned, "sticky sessions stay pinned while the owner fits"
    led.free[pinned] = 10                                   # the sticky rank runs dry
    moved = pl.assign(500, session="alpha")
    assert moved != pinned and led.free[moved] >= 500, "a dry sticky owner re-pins to the fittest rank"
    try:
        pl.assign(10 ** 9)
        raise AssertionError("over-capacity admission must raise")
    except RuntimeError:
        pass
    led.release(moved, 500)
    assert led.free[moved] == 1000


# -- test 5: step payloads, buckets and the digest ----------------------------------------------------------------------

def test_step_split_and_digest():
    rb = DP.RoundBuilder(chunk=3)
    a = DP.Stream(7, list(range(5)), 3, owner=2)            # 2 prefill chunks of 3/2
    b = DP.Stream(9, list(range(4)), 3, owner=3)
    rb.add(a)
    rb.add(b)
    rows = rb.round_rows(4)
    # one prefill chunk per round, on the owner rank only; other ranks hold empty sets
    assert rows[2][0] == [7, 1, 0, [0, 1, 2]] and rows[0] == [] and rows[1] == [] and rows[3] == []
    assert DP.RoundBuilder.bucket_of(rows) == 16
    rows = rb.round_rows(4)                                 # a's second chunk (2 rows)
    assert rows[2][0][3] == [3, 4] and DP.RoundBuilder.bucket_of(rows) == 16
    while a.needs_fill or b.needs_fill:                     # finish b's prefill
        rb.round_rows(4)
    a.out.append(42)                                        # the finished prefills' first tokens (as _apply)
    b.out.append(43)
    a.pending, b.pending = 42, 43
    for _ in range(2):                                      # decode rounds: each rank only its own sids
        rows = rb.round_rows(4)
        for r, rows_r in enumerate(rows):
            for row in rows_r:
                assert row[0] == {2: 7, 3: 9}.get(r), f"rank {r} holds a foreign row"
                assert row[1] == 0                          # decode rows
                assert row[2] == len({7: a, 9: b}[row[0]].prompt) + len({7: a, 9: b}[row[0]].out) - 1
        for row in rows[2] + rows[3]:                       # emulate _apply's token bookkeeping
            s = {7: a, 9: b}[row[0]]
            s.out.append(77)
            s.pending = 77
    op = DP.pack_step(9, 16, rows)
    assert DP.rows_for(op, 0) == op[3][0] and DP.rows_for(op, 2) == op[3][2]
    assert DP.step_digest(op) == DP.step_digest(list(op))
    assert DP.step_digest(op) != DP.step_digest([*op[:2], 32, op[3]]), "the bucket field must move the digest"

    # a digest disagreement raises OutOfStep on BOTH ranks
    comms = emu_comms(2)
    errs = []

    def ag(r, plan):
        try:
            DP.agree(comms[r], "step", plan)
        except DP.OutOfStep as exc:
            errs.append(str(exc))

    t0 = threading.Thread(target=ag, args=(0, op))
    t1 = threading.Thread(target=ag, args=(1, [*op[:2], 32, op[3]]))   # follower holds a different bucket
    t0.start()
    t1.start()
    t0.join(timeout=60)
    t1.join(timeout=60)
    assert len(errs) == 2, f"OutOfStep must land on every rank, got {errs}"


# -- test 6: the KV byte formula (D2 §6) --------------------------------------------------------------------------------

def test_kv_bytes_per_token():
    assert EN.kv_bytes_per_token(T2.tiny_cfg(TM.TinyConfig())) == 784      # 7*64 + 6*16 + 5*48 bytes
    kv = list(range(72))                                                    # the real shape: 63 writers,
    for reader in (7, 15, 23, 31, 39, 47, 55, 63, 71):                      # each fed from an earlier writer
        kv[reader] = reader - 4
    rope = [0 if i % 4 == 3 else 1 for i in range(72)]                      # 18 NoPE, 9 of them writers
    idx = ["shared" if i < 28 else "full" for i in range(72)]               # 44 full indexers (D2 §6 counts)
    cfg = T2.Cfg(vocab=1, dim=1, n_layers=72, n_heads=1, kv_lora=512, nope=64, rd=64, v_head=128, qk_head=128,
                 inter=1, dense_inter=1, n_routed=128, topk=4, n_shared=1, route_scale=1.0, eps=1e-6,
                 index_norm_eps=1e-6, idx_heads=32, idx_dim=128, idx_topk=2048, softcap=30.0, sm_scale=0.1,
                 rope_flags=rope, kv_source_layers=kv, kv_sources=set(), indexer_types=idx,
                 mlp_layer_types=["dense"] + ["sparse"] * 71, rope_theta=1e6, rope_type="yarn",
                 rope_factor=2048.0, orig_len=512, attention_factor=1.0, tie_word_embeddings=False,
                 hidden_act="silu", q_rank=None)
    assert EN.kv_bytes_per_token(cfg) == 82688, "D2 §6: fp16 82,688 B/token on the real dims"


# -- test 7: pool sizing override + engine plumbing ---------------------------------------------------------------------

def test_engine_pool_override():
    import os

    os.environ["TF_ML4_KV_TOKENS"] = "12345"
    try:
        tc = TM.TinyConfig()
        W = TM.random_weights(tc, 1105)
        e = EN.Ml4Engine(None, rank=0, world=1, device="cpu", weights=T2.tiny_weights(tc, W), log=lambda _s: None)
        assert e.pool_tokens == 12345 and e.tp == 1 and e.limit == 131072
        assert EN.kv_bytes_per_token(e.w.cfg) == 784
    finally:
        del os.environ["TF_ML4_KV_TOKENS"]


# -- test 8: world=1 generate runs the naive cache end to end (tiny smoke) ----------------------------------------------

def test_world1_generate():
    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    e = EN.Ml4Engine(None, rank=0, world=1, device="cpu", weights=T2.tiny_weights(tc, W),
                     pool_tokens=4096, log=lambda _s: None)
    got = []
    stats = e.generate([1, 2, 3, 4], 3, on_tokens=feed(got))
    assert len(got) == 3 and stats["tokens"] == 3
    again = []
    e2 = EN.Ml4Engine(None, rank=0, world=1, device="cpu", weights=T2.tiny_weights(tc, W),
                      pool_tokens=4096, log=lambda _s: None)
    e2.generate([1, 2, 3, 4], 3, on_tokens=feed(again))
    assert again == got, "greedy generation must be reproducible"


# -- test 9: A2A transport math on the raw API (counts matrix + pair order) ----------------------------------------------

def test_a2a_pair_order_and_counts():
    dest = torch.tensor([[3, 0], [1, 2], [0, 0]])
    assert A2A.route_dest(dest, 4, 8).tolist() == [[1, 0], [0, 1], [0, 0]]     # e // (8 // 4)
    assert A2A.row_pairs([3, 0, 3], 4) == [(0, [1]), (3, [0, 2])]              # dst ascending, slots ascending
    try:
        A2A.route_dest(dest, 3, 8)
        raise AssertionError("non-divisible expert split must raise")
    except ValueError:
        pass


# -- test 10: the load4 interleave — reproduced, and the rank-0 broadcast lock holding it ---------------------------

def _race_lane(world=2):
    """A tiny lane whose follower digests are gated on rank 0's gather ARRIVALS (deterministic
    skew, no sleep races): the follower always waits until rank 0 has entered the matching gather,
    and in repro mode waits for ``extra`` further rank-0 arrivals first — that hold is exactly the
    load4 window where a second server thread's gather jumps the follower's."""

    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    w1 = T2.tiny_weights(tc, W)
    comms = emu_comms(world)
    link = DP.EmuLink(world, 0)
    engines = [EN.Ml4Engine(None, rank=r, world=world, device="cpu",
                            weights=slice_weights(w1, r, world), comm=comms[r], link=link.fork(r),
                            pool_tokens=4096, log=lambda _s: None) for r in range(world)]
    order, excs = [], {}
    st = {"arrived": 0, "dig": 0, "extra": 0}                  # rank-0 gather arrivals vs follower digests

    base_ag = engines[0].comm.all_gather

    def counting_ag(send, recv):
        st["arrived"] += 1
        return base_ag(send, recv)

    engines[0].comm.all_gather = counting_ag

    orig = engines[1].link.receive

    def gated_receive():
        op = orig()
        order.append(op[0] if isinstance(op, list) else "none")
        deadline = time.monotonic() + 30
        while st["arrived"] <= st["dig"] + st["extra"] and time.monotonic() < deadline:
            time.sleep(0.005)                                  # the follower lags rank 0's threads
        st["dig"] += 1
        return op

    engines[1].link.receive = gated_receive
    t = threading.Thread(target=engines[1].follow, daemon=True)
    t.start()
    return engines, order, excs, st


def _race(engines, excs):
    """Two rank-0 server threads as the real server runs them: one drives the round, another
    admits a second request while that round's digest is still gathering."""

    def a():
        try:
            engines[0].run_round()
        except BaseException as exc:                           # noqa: BLE001  (the lane may die; we assert it)
            excs["round"] = exc

    def b():
        time.sleep(0.05)                                       # admit lands mid-step, like the concurrent client
        try:
            engines[0].admit([3, 1, 4, 1], 2)
        except BaseException as exc:                           # noqa: BLE001
            excs["admit"] = exc

    ts = [threading.Thread(target=a), threading.Thread(target=b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    return not any(t.is_alive() for t in ts)


def test_interleave_breaks_lane_without_the_lock():
    """Repro for load4's OutOfStep (mandate item 2): EmuComm pairs gathers by ARRIVAL like NCCL,
    so with rank 0's broadcast lock removed, thread B's admission digest lands in the same
    generation as thread A's step digest — the lane corrupts (both rank-0 threads raise, exactly
    the load4 "step mismatch here, admission mismatch there" trace). The bug is NOT a bad digest
    value (item 3: the digest inputs are byte-identical by construction — the follower digests
    rank 0's op object exactly as it ships); it is the GATHER ORDER."""

    engines, _order, excs, st = _race_lane()
    engines[0].admit([5, 9, 23, 47, 61], 3)                    # request 1 admitted and digested in the clear
    engines[0]._bcast_lock = contextlib.nullcontext()          # the pre-m2b5 engine
    st["extra"] = 1                                            # follower waits for B's gather to jump in
    _race(engines, excs)                                       # a corrupted lane may strand daemons
    assert excs, "the unserialized rank-0 broadcast did NOT corrupt the lane — the sim stopped " \
                 "modeling NCCL arrival pairing"


def test_interleave_serialized_by_bcast_lock():
    """With the lock (the shipped engine), the same race is a plain ordering: the admission waits
    for the round, the follower sees admit -> step -> admit, and no rank raises."""

    engines, order, excs, st = _race_lane()
    engines[0].admit([5, 9, 23, 47, 61], 3)
    assert _race(engines, excs), "race threads hung"
    assert not excs, f"locked lane still corrupted: {excs}"
    deadline = time.monotonic() + 15
    while len(order) < 3 and time.monotonic() < deadline:      # let the follower digest admit2
        time.sleep(0.01)
    assert order == ["admit", "step", "admit"], f"link order {order} != digest order rank 0 used"
    return {"order": order}


def test_pool_sizing_math():
    """m2b6 item 5: the pure sizing functions on synthetic MemAvailable values — the plan formula,
    the refuse when nothing is left, and the loud refusal of an unbackable TF_ML4_KV_TOKENS."""

    G = 2 ** 30
    bpt = 82688                                              # real-dims fp16 B/token (D2 §6)
    got = EN.plan_pool_tokens(int(19.25 * G), bpt, 10.0, 1.5)          # load4 host, measured reserve
    assert got == int((19.25 - 11.5) * G) // bpt and got > 0
    assert EN.plan_pool_tokens(int(19.25 * G), bpt, 6.0, 1.5) >= 131072, \
        "floor 6 + a ~1.5 GiB measured reserve must make ONE 128K stream possible (m2b6 table)"
    try:
        EN.plan_pool_tokens(int(16.0 * G), bpt, 10.0, 7.0)             # the §6 guess on a tighter host
        raise AssertionError("budget <= 0 must refuse")
    except RuntimeError as exc:
        assert "MemAvailable" in str(exc)
    assert EN.check_pool_request(100_000, int(19.25 * G), bpt, 6.0, 1.5) == 100_000
    try:
        EN.check_pool_request(1_000_000, int(19.25 * G), bpt, 6.0, 1.5)
        raise AssertionError("an unbackable forced pool must refuse, not OOM later")
    except RuntimeError as exc:
        assert "TF_ML4_KV_TOKENS" in str(exc) and "cannot be backed" in str(exc)
    return {"floor6_pool": EN.plan_pool_tokens(int(19.25 * G), bpt, 6.0, 1.5)}


# -- test 12: the m2b8 gather bug — digests identical, compare still raced (repro + fix) -----------------------------

class AsyncKernelComm:
    """Wraps a lane comm and gives it NCCL's ASYNC kernel semantics: all_gather computes the
    right bytes but returns with the caller's recv buffer NOT yet written (the "kernel" is in
    flight); `gather_done()` is the torch.cuda.synchronize stand-in that makes the bytes land —
    exactly the dp4.agree() barrier m2b8 adds. done=False models the pre-m2b8 reader (no barrier
    attribute at all); late=True models the drop-then-replay frame: each gather's bytes land one
    gather LATE, so the previous round's reply is what the next compare sees."""

    def __init__(self, base, *, done=True, late=False):
        self.base, self.done_flag, self.late = base, done, late
        self.rank, self.world = base.rank, base.world
        self.pending = []                                       # [(buffer, landed bytes)]
        self.stale_hits = 0
        if not done:
            self.gather_done = None        # shadow the method: the pre-m2b8 reader has no barrier

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _stale(self, recv):
        n = max(1, recv.numel() // self.world)                 # allocator leftovers differ per slot —
        for s in range(0, recv.numel(), n):                    #     a uniform buffer would pass self-consistent
            recv[s:s + n].fill_(1000 + self.rank * 10 + s)

    def all_gather(self, send, recv):
        tmp = torch.empty_like(recv)
        self.base.all_gather(send, tmp)                        # pairing happens (call order)
        if self.late:
            prev = getattr(self, "prev", None)
            recv.copy_(tmp)                                    # this round's bytes...
            if prev is not None:
                recv[-1] = prev[-1]                            # ...then LAST round's reply arrives
                self.stale_hits += 1                           #     late: one slot holds a stale digest
            else:
                self._stale(recv)                              # nothing of this gather has landed
            self.prev = tmp
        else:
            n = max(1, recv.numel() // self.world)
            recv.copy_(tmp)                                    # the slots the kernel already wrote...
            recv[(self.world - 1) * n:] = 1000 + self.rank * 10  # ...stop mid-flight: the last
            self.pending.append((recv, tmp))                   #     slots are still junk for a reader

    def gather_done(self):                                      # torch.cuda.synchronize equivalent
        while self.pending:
            dst, src = self.pending.pop(0)
            dst.copy_(src)


def _async_lane(world=2, **wrap):
    engines, order, excs, st = _race_lane(world)
    for e in engines:
        e.comm = AsyncKernelComm(e.comm, **wrap)
    return engines, order, excs, st


def test_gather_stale_frame_repro():
    """The load4 2026-11-06 (m2b5 re-run) failure on CPU: rank-0 lock HELD, digest inputs
    byte-identical (the gather itself paired correctly), yet OutOfStep — because agree() compared
    the recv buffer while the kernel was still filling it. The slots the kernel had not reached
    held stale bytes: every rank digests the same op and the compare still sees a mismatch."""

    engines, _order, _excs, _st = _async_lane(done=False)
    try:
        engines[0].admit([5, 9, 23, 47, 61], 3)
        raise AssertionError("a barrierless compare of an in-flight gather must race — the sim "
                             "stopped modelling the async kernel")
    except DP.OutOfStep as exc:
        assert "admission" in str(exc)


def test_gather_barrier_fixes_it():
    """With gather_done() + stream-ordered read (the shipped agree now), the SAME async kernel
    cannot fool the compare: admit and a full round pass, the follower mirrors the order."""

    engines, order, excs, _st = _async_lane(done=True)
    engines[0].admit([5, 9, 23, 47, 61], 3)
    assert engines[0].run_round() is True
    deadline = time.monotonic() + 15
    while len(order) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert order == ["admit", "step"] and not excs, f"fixed lane corrupted: {order} {excs}"


def test_gather_replay_poisons_and_recovers():
    """Drop-then-replay: when a gather's bytes land one gather LATE, a barrierless reader is
    poisoned TWICE (each compare sees the PREVIOUS round's digest — 'admit N-1's reply landing in
    gather N'), and the same lane on the fixed path recovers immediately."""

    engines, _order, _excs, _st = _async_lane(done=False, late=True)
    poisoned = 0
    for prompt in ([7, 1, 1, 2, 3], [8, 2, 2, 3, 5]):           # two admits: each compare lands one
        try:                                                    #     gather late — never its own bytes
            engines[0].admit(prompt, 2)
            break
        except DP.OutOfStep:
            poisoned += 1
    assert poisoned == 2, f"late replies must poison EVERY compare, got {poisoned}"
    for e in engines:                                           # the shipped comm/barrier
        e.comm = AsyncKernelComm(e.comm.base, done=True)
    engines[0].admit([7, 1, 1, 2, 3], 2)                        # same op, now clean


def test_gather_out_of_order_arrival_pairs_by_call_order():
    """Mandate item 2's arrival skew itself: rank 2 ENTERS the gather before rank 1 before rank 0.
    Pairing is by call order (NCCL semantics EmuComm keeps), so the landed data is exact — the
    hazard was never who arrives first; it was reading before the kernel lands (buffer still
    junk), which gather_done() closes."""

    comms = [AsyncKernelComm(c, done=True) for c in emu_comms(3)]
    recvs = {r: torch.zeros(3, dtype=torch.int64) for r in range(3)}
    ts = [threading.Thread(target=lambda r=r: comms[r].all_gather(torch.tensor([r + 1],
          dtype=torch.int64), recvs[r]), daemon=True) for r in (2, 1, 0)]
    for t in ts:
        t.start()
        time.sleep(0.03)                                       # arrival order 2, 1, 0
    for t in ts:
        t.join(timeout=30)
    for r in range(3):                                         # in flight: nothing readable yet
        assert recvs[r].tolist() == [1, 2, 1000 + r * 10], "kernel landed whole — sim stopped modelling async"
    for c in comms:
        c.gather_done()
    for r in range(3):
        assert recvs[r].tolist() == [1, 2, 3], f"rank {r} gather skewed: {recvs[r].tolist()}"


# -- m2b8: a2a pair-buffer device pin + probe audit + bounded probe timeout ---------------------------------------------

def test_a2a_pair_buffers_land_on_payload_device():
    """The m2b6 load4 crash in one line: `eids[slots] = idx[r, slots]` put an on-device value into a
    DEFAULT-device (cpu) buffer — on the node that is "found at least two devices, cuda:0 and cpu",
    in the first CUDA world>1 forward ever run (the activation probe). The meta device is the fake
    second device CPU has, so the pin is real: _pair_buffers must allocate ON the routing device."""

    topk = 4
    eids, wt = A2A._pair_buffers([0, 2], torch.full((topk,), 5, dtype=torch.int64, device="meta"),
                                 torch.full((topk,), 0.5, device="meta"), topk, torch.device("meta"))
    assert eids.device.type == "meta" and wt.device.type == "meta", \
        "pair buffers allocated on the default device: cuda values index_put into them crash the lane"
    eids, wt = A2A._pair_buffers([0, 2], torch.tensor([7, 3, 9, 4]),
                                 torch.tensor([0.5, 0.25, 0.1, 0.05]), topk, torch.device("cpu"))
    assert eids.tolist() == [7, -1, 9, -1]                      # semantics unchanged by the pin
    assert [round(x, 3) for x in wt.tolist()] == [0.5, 0.0, 0.1, 0.0]   # f32 payloads: compare rounded


def test_probe_audit_names_off_device_tensor():
    """Mandate: assert every module the probe touches is on the engine device, naming the offender.
    The CPU lane is clean; one planted meta tensor on a field forward reads must name that field —
    the same check that runs before the probe on the node (cuda dev, cpu leftover = offender)."""

    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    e = EN.Ml4Engine(None, rank=0, world=1, device="cpu", weights=T2.tiny_weights(tc, W),
                     pool_tokens=4096, log=lambda _s: None)
    e._probe_device_audit()                                     # the clean lane passes
    saved = e.w.layers[1].attn_norm
    e.w.layers[1].attn_norm = torch.zeros_like(saved, device="meta")
    try:
        try:
            e._probe_device_audit()
            raise AssertionError("audit must refuse an off-device tensor")
        except RuntimeError as exc:
            assert "layers[1].attn_norm" in str(exc) and "meta" in str(exc), str(exc)
    finally:
        e.w.layers[1].attn_norm = saved
    e._probe_device_audit()                                     # clean again


def test_probe_bounded_timeout_and_measured_path():
    """Mandate: never let the probe hang; ANY failure lands on the fallback. Device-parameterized
    fake (the mandate's cpu/cuda-string test): the engine CLAIMS cuda while tensors stay cpu and
    torch.cuda is stubbed — a forward that never returns times out into a TIMED OUT fallback line
    (bounded, deterministic: join(0.25) of a 5 s wait), and a fast forward yields the measured
    reserve. The lane must serve without the probe; it must never serve wedged."""

    import os

    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    e = EN.Ml4Engine(None, rank=0, world=1, device="cpu", weights=T2.tiny_weights(tc, W),
                     pool_tokens=4096, log=lambda _s: None)
    e.device = "cuda"                                           # fake-device parameterization
    e._dev = torch.device("cpu")
    real = (torch.cuda.synchronize, torch.cuda.memory_reserved,
            torch.cuda.reset_peak_memory_stats, torch.cuda.max_memory_reserved)
    done = threading.Event()
    try:
        torch.cuda.synchronize = lambda: None
        torch.cuda.reset_peak_memory_stats = lambda: None
        torch.cuda.memory_reserved = lambda: 100 * 2 ** 20
        torch.cuda.max_memory_reserved = lambda: 130 * 2 ** 20
        os.environ["TF_ML4_PROBE_TIMEOUT"] = "0.25"
        e.model.forward = lambda *a, **k: done.wait(5.0)        # a wedged probe forward
        lines: list = []
        assert e._activation_probe(lines.append) is None
        assert any("TIMED OUT" in s for s in lines), lines      # fallback, not a hang — the m2b8 rule
        done.set()                                              # free the daemon thread (capped at 5 s)

        e.model.forward = lambda *a, **k: None                  # a fast probe forward
        lines = []
        gib = e._activation_probe(lines.append)
        assert gib is not None and abs(gib - 30 * 2 ** 20 / 2 ** 30) < 1e-9, gib
        assert any("peak CUDA reserved" in s for s in lines), lines
    finally:
        done.set()
        (torch.cuda.synchronize, torch.cuda.memory_reserved,
         torch.cuda.reset_peak_memory_stats, torch.cuda.max_memory_reserved) = real
        os.environ.pop("TF_ML4_PROBE_TIMEOUT", None)


def test_memory_ceiling_caps_growth_not_total():
    """m2b10 (m2b9's load4 OOM): set_per_process_memory_fraction caps the process TOTAL, so the
    fraction must include the resident weights — the old (available - floor)/total landed the cap
    at ~9 GiB while 93.78 GiB sat inside the process, and the first token-gate allocation died.
    Device-parameterized (claims cuda, tensors stay cpu, torch.cuda stubbed): ONE fake MemAvailable
    must drive BOTH the ceiling and the pool, the ceiling must be set BEFORE pool sizing, the cap
    must sit at or above resident, and pool + activations must fit the growth exactly."""

    import os

    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    gib = 2 ** 30
    avail, total, reserved = 19 * gib, int(123.79 * gib), round(93.78 * gib)
    names = ("set_device", "synchronize", "mem_get_info", "memory_reserved", "memory_allocated",
             "set_per_process_memory_fraction")
    real_cuda, real_avail, fractions, lines = {n: getattr(torch.cuda, n) for n in names}, \
        EN.read_mem_avail, [], []
    try:
        torch.cuda.set_device = lambda _i: None
        torch.cuda.synchronize = lambda: None
        torch.cuda.mem_get_info = lambda: (30 * gib, total)
        torch.cuda.memory_reserved = lambda: reserved
        torch.cuda.memory_allocated = lambda: reserved
        torch.cuda.set_per_process_memory_fraction = fractions.append
        EN.read_mem_avail = lambda: avail
        os.environ["TF_ML4_PROBE"] = "0"
        e = EN.Ml4Engine(None, rank=0, world=1, device="cuda", fast=False, weights=T2.tiny_weights(tc, W),
                         log=lines.append)
    finally:
        for n, v in real_cuda.items():
            setattr(torch.cuda, n, v)
        EN.read_mem_avail = real_avail
        os.environ.pop("TF_ML4_PROBE", None)
    joined = "\n".join(lines)
    assert "allocator ceiling" in joined and "budget closed" in joined, joined
    assert joined.index("allocator ceiling") < joined.index("KV pool")      # m2b10 ordering
    assert joined.count(f"{avail / gib:.1f} GiB") >= 2                      # the ONE snapshot feeds both
    assert len(fractions) == 1 and fractions[0] * total >= reserved, fractions   # the m2b9 regression
    growth = avail - int(EN.MEM_FLOOR_GIB * gib)
    need = e.pool_tokens * e.kv_bpt + int(EN.KV_HEADROOM_GIB * gib)         # pool + activations
    assert e.pool_tokens > 0 and need <= growth and growth - need < gib, (growth, need)


def test_budget_check_fails_loudly_with_four_numbers():
    """The m2b10 invariant fails at INIT with all four numbers (ceiling, resident, pool,
    activations), never at the first token gate with a cryptic CUDA OOM: a constructor given
    pool_tokens past the growth allowance must raise."""

    import os

    tc = TM.TinyConfig()
    W = TM.random_weights(tc, 1105)
    gib = 2 ** 30
    avail, total, reserved = 19 * gib, int(123.79 * gib), round(93.78 * gib)
    names = ("set_device", "synchronize", "mem_get_info", "memory_reserved", "memory_allocated",
             "set_per_process_memory_fraction")
    real_cuda, real_avail = {n: getattr(torch.cuda, n) for n in names}, EN.read_mem_avail
    try:
        torch.cuda.set_device = lambda _i: None
        torch.cuda.synchronize = lambda: None
        torch.cuda.mem_get_info = lambda: (30 * gib, total)
        torch.cuda.memory_reserved = lambda: reserved
        torch.cuda.memory_allocated = lambda: reserved
        torch.cuda.set_per_process_memory_fraction = lambda _f: None
        EN.read_mem_avail = lambda: avail
        os.environ["TF_ML4_PROBE"] = "0"
        try:
            EN.Ml4Engine(None, rank=0, world=1, device="cuda", fast=False, weights=T2.tiny_weights(tc, W),
                         pool_tokens=10 ** 8, log=lambda _s: None)
            raise AssertionError("budget check accepted a pool past the ceiling")
        except RuntimeError as exc:
            msg = str(exc)
        assert "memory budget broken" in msg, msg
        for needle in ("ceiling", "resident", "pool", "activations", "MemAvailable"):
            assert needle in msg, msg
    finally:
        for n, v in real_cuda.items():
            setattr(torch.cuda, n, v)
        EN.read_mem_avail = real_avail
        os.environ.pop("TF_ML4_PROBE", None)


def test_trellis_bytes_lands_on_the_slot_device():
    """m2b10 (m2b9's probe refusal, 'trellis_bytes@cpu'): the per-expert stats table is the only
    field prepare() built without device= — the slot's audit offender. Pin: it lands on the slot's
    device (meta fake; default 'cuda' pinned by signature)."""

    import inspect
    import sys
    import types

    if "numpy" not in sys.modules:                                # the exl3 package's format.py imports numpy at
        try:                                                      # module level but only USES it in call paths;
            import numpy                                          # trellis_bytes itself is pure torch, so a
        except ModuleNotFoundError:                               # bare stub makes the import work in this venv
            sys.modules["numpy"] = types.ModuleType("numpy")
    from tensorfold.cuda.exl3.experts import trellis_bytes as tb_fn

    assert str(inspect.signature(tb_fn).parameters["device"].default) == "cuda"
    t = tb_fn(4096, 14336, [8, 8], [10, 3], [9, 7], device="meta")     # red on old (ignored device -> cpu)
    assert t.device.type == "meta" and t.shape == (2,) and t.dtype == torch.int64
    per = 4096 * 14336 // 256 * 16
    got = tb_fn(4096, 14336, [8, 8], [10, 3], [9, 7], device="cpu")     # the arithmetic, readable
    assert got.device.type == "cpu" and got.tolist() == [per * 27, per * 18], got.tolist()


# -- m2b11: the run_batch (CUDA regime) pairing battery — load5's deadlock class --------------------------------------

DIM, TOPK, NR = 8, 2, 8                     # tiny layer geometry; NR experts over `world` ranks


def _rb_run_batch(recv_x, eids, wts):       # deterministic own-block expert partials, like _ep4_batch:
    out = torch.zeros((recv_x.shape[0], recv_x.shape[-1]), dtype=torch.float32)   # one fp-accurate vec
    n = 0                                                          # per RECEIVED row, weight applied
    for p in range(recv_x.shape[0]):
        for s in range(eids.shape[1]):
            e = int(eids[p, s])
            if e >= 0:
                out[p] = out[p] + wts[p, s] * (recv_x[p] * (1.5 + 0.25 * e))
                n += 1
    return out, n


def _rb_expect(rows, picks, world):         # all-experts-local reference in the a2a's own add order:
    out = torch.zeros((len(rows), DIM), dtype=torch.float32)       # per row, partials in DEST-ASC order
    for r, x in enumerate(rows):
        by_dst: dict = {}
        for e, w in picks[r]:
            d = e // (NR // world)
            v = w * (x * (1.5 + 0.25 * e))
            by_dst[d] = v if d not in by_dst else by_dst[d] + v
        for d in sorted(by_dst):
            out[r] = out[r] + by_dst[d]
    return out


def _rb_all_to_all(world, picks_by_rank, wrap=None):
    """One MoE layer over `world` ranks in the run_batch regime — the CUDA regime every node forward
    uses and no CPU test ever ran until m2b11 (the CPU suites exercise run_local, whose reply sizes
    always paired). Completes with every rank's combine matching the all-experts reference, or says
    exactly which pairing failed."""

    torch.manual_seed(1105)
    rows = {r: [torch.randn(DIM, dtype=torch.float32) for _ in picks_by_rank[r]] for r in range(world)}
    comms = emu_comms(world)
    if wrap is not None:
        comms = [wrap(c) for c in comms]
    outs, errs = [None] * world, [None] * world

    def run(r):
        try:
            pk = picks_by_rank[r]
            if pk:
                x = torch.stack(rows[r])
                idx = torch.tensor([[e for e, _ in row] for row in pk], dtype=torch.int64)
                w = torch.tensor([[w for _, w in row] for row in pk], dtype=torch.float32)
            else:                                               # the idle rank's zero-row step path
                x = torch.zeros((0, DIM), dtype=torch.float32)
                idx = torch.zeros((0, TOPK), dtype=torch.int64)
                w = torch.zeros((0, TOPK), dtype=torch.float32)
            outs[r] = A2A.moe_all_to_all(comms[r], x, idx, w, n_routed=NR, run_batch=_rb_run_batch)
        except BaseException as exc:                              # noqa: BLE001  (surface via errs)
            errs[r] = exc

    ts = [threading.Thread(target=run, args=(r,), daemon=True) for r in range(world)]
    # daemon: a BROKEN pairing strands a peer at EmuComm's drain barrier forever — exactly the node
    # deadlock; the daemon flag lets the test report that failure and still let the process exit.
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in ts), "the run_batch all-to-all deadlocked — pairing never completes"
    dead = [r for r in range(world) if errs[r] is not None]
    assert not dead, "run_batch pairing raised: " + "; ".join(f"rank {r}: {errs[r]!r}" for r in dead)
    for r in range(world):
        ref = _rb_expect(rows[r], picks_by_rank[r], world)
        assert outs[r].shape == ref.shape and torch.equal(outs[r], ref), \
            f"rank {r} run_batch combine != all-experts reference"     # deterministic f32: exact


def test_a2a_run_batch_zero_row_follower():
    """MODE B reproduction, load5 (node run 2026-11-06, m2b10 build; t6 phaseB §40): rank 0 owns the
    round's only rows, so the followers' step runs the ZERO-ROW path — the engine round's real code
    asymmetry, at the a2a level. C[0,1] = 3 dispatched rows, C[1,0] = 0. The run_batch reply leg
    must size its RECV by rows dispatched OUT (len(out_list)), not rows received (n_in): pre-fix,
    rank 0's reply group was empty (NCCL skips zero-size legs) while rank 1 posted send+recv into a
    peer that had moved to the next layer's counts all_gather — silent P2P deadlock, node evidence
    "rank 0 GPU 96% / rank 1 CPU 97% / rounds_total 0 / no error ever". EmuComm turns the same
    mismatch into a loud copy_ size error, so this pins RED→GREEN deterministically."""

    _rb_all_to_all(2, {0: [[(0, 0.6), (4, 0.4)], [(1, 0.5), (5, 0.5)], [(0, 0.7), (4, 0.3)]],
                       1: []})                                    # rank 1 owns no rows: mode B


def test_a2a_run_batch_probe_shape_counts():
    """MODE A reproduction, load5 §40 correction: the probe did NOT have one rank refusing — ALL FOUR
    entered the probe forward (device audit passed on all: the m2b10 trellis fix worked) and ALL FOUR
    TIMED OUT. Symmetric ENTRY still deadlocks when counts are asymmetric, and identical-token
    routing still makes C[i,j] != C[j,i] (per-expert traffic differs per destination block). Four
    ranks, every rank with rows, uneven counts on every pair — the probe's shape at test scale."""

    _rb_all_to_all(4, {0: [[(0, 0.5), (1, 0.5)], [(3, 0.6), (6, 0.4)]],
                       1: [[(4, 0.6), (0, 0.4)], [(5, 0.5), (2, 0.5)]],
                       2: [[(6, 0.5), (0, 0.5)], [(7, 0.6), (1, 0.4)], [(4, 0.4), (5, 0.6)]],
                       3: [[(2, 0.6), (0, 0.4)]]})


def test_a2a_run_batch_async_counts_land():
    """The mode-B shape over m2b8's async-kernel sim: the counts all_gather returns with C NOT yet
    written (NCCL semantics). m2b11 pins a2a's gather_done() before C decides WHICH pairs exchange
    and at what size — agree()'s m2b8 landing lesson applied to the pairing decision itself; a rank
    reading stale C picks different pairs than its peer and the wire never matches."""

    _rb_all_to_all(2, {0: [[(0, 0.6), (4, 0.4)], [(1, 0.5), (5, 0.5)], [(0, 0.7), (4, 0.3)]],
                       1: []}, wrap=lambda c: AsyncKernelComm(c, done=True))


class _FakeStore:                             # TCPStore stand-in for the probe gate (set/wait/multi_get)
    def __init__(self):
        self.d, self.cv = {}, threading.Condition()

    def set(self, k, v):
        with self.cv:
            self.d[k] = v.encode()
            self.cv.notify_all()

    def multi_get(self, keys):
        with self.cv:
            return [self.d[k] for k in keys]

    def wait(self, keys, timeout):
        import time as _t

        end = _t.monotonic() + timeout.total_seconds()
        with self.cv:
            while not all(k in self.d for k in keys):
                left = end - _t.monotonic()
                if left <= 0:
                    raise RuntimeError("Wait timeout: store not with keys available")
                self.cv.wait(left)


def _gate_call(store, rank, world, refuse):
    import types as _types

    ns = _types.SimpleNamespace(rank=rank, world=world,
                                comm=_types.SimpleNamespace(store=store))
    return EN.Ml4Engine._probe_gate(ns, refuse)


def test_probe_gate_is_all_or_nothing():
    """m2b11 mandate 4 (structural probe fix): the refusal decision is computed on EVERY rank
    BEFORE anyone enters the probe's collectives, over the rendezvous store (TCP — a vote cannot
    poison a lane). One rank refuses → every rank gets a non-empty reason (all skip together, no
    collective touched); all clear → every rank gets "" (all enter together); a rank that never
    arrives → every rank lands on the same bounded skip. This is the wedge m2b8 only half-cured:
    its in-thread audit refusal pulled one rank out of collectives the others were already in."""

    import os

    assert hasattr(EN.Ml4Engine, "_probe_gate"), "mandate 4: _probe_gate must exist"
    store = _FakeStore()
    got = {}

    def run(r, refuse):
        got[r] = _gate_call(store, r, 2, refuse)

    ts = [threading.Thread(target=run, args=(0, "")), threading.Thread(target=run, args=(1, "evil@cpu"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in ts) and got[0] and got[1], got
    assert "rank 1: evil@cpu" in got[0] and "rank 1: evil@cpu" in got[1], got   # BOTH skip, same reason

    store = _FakeStore()
    got = {}
    ts = [threading.Thread(target=run, args=(0, "")), threading.Thread(target=run, args=(1, ""))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    assert got == {0: "", 1: ""}, f"an all-clear vote must let the whole lane probe: {got}"

    os.environ["TF_ML4_PROBE_GATE_TIMEOUT"] = "0.2"
    try:
        lonely = _gate_call(_FakeStore(), 0, 2, "")           # rank 1 never votes
        assert "never reached the probe gate" in lonely, lonely
    finally:
        os.environ.pop("TF_ML4_PROBE_GATE_TIMEOUT", None)


def test_trace_comm_numbers_pairing():
    """m2b11 mandate 2: TraceComm stamps every collective with a per-rank monotonic seq, byte
    counts on both sides of every exchange, and a walkable tag — so the four node logs diff exactly
    where pairing stops (load5: rank 0 parked in a counts all_gather while rank 1 sat in the PREVIOUS
    layer's reply exchange; without seqs both logs just said "NCCL waiting"). Also pins the engine
    default: world>1 engines wrap their comm unless TF_ML4_TRACE=0."""

    import re

    lines = {0: [], 1: []}
    comms = [DP.TraceComm(c, lines[r].append) for r, c in enumerate(emu_comms(2))]
    comms = [AsyncKernelComm(c, done=True) for c in comms]   # the tag must walk this wrapper chain
    op = DP.pack_step(1, 16, [[[1, 0, 0, 5]], [[1, 0, 0, 6]]])

    def run(r):
        with DP.trace_tag(comms[r], "agree:step"):
            DP.agree(comms[r], "step", op)
        a = torch.zeros((3,), dtype=torch.float32)
        b = torch.zeros((3,), dtype=torch.float32)
        comms[r].exchange([a], [b], 1 - r)                    # matched pair: out/in counts mirror
        getattr(comms[r], "trace", lambda _m: None)("a2a combine rows=1")

    ts = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in ts), "traced comms must not change pairing semantics"
    for r in (0, 1):
        text = "\n".join(lines[r])
        seqs = [int(s) for s in re.findall(r"#(\d+) ", text)]   # each collective stamps entry+exit, same #
        assert seqs == sorted(seqs) and seqs[:1] == [1], f"seqs not per-rank monotone from 1: {lines[r]}"
        assert len(set(seqs)) * 2 == len(seqs), f"each collective must log entry AND exit: {lines[r]}"
        assert "agree:step" in text, f"tag did not reach through AsyncKernelComm.base: {lines[r]}"
        assert "exchange p" in text and "out=[3] in=[3]" in text, f"exchange lacks its byte counts: {lines[r]}"
        assert text.count("done") >= 2, f"entry lines lack exits: {lines[r]}"
        assert "a2a combine" in text, f"combine marker missing: {lines[r]}"
    engines, _order, _excs, _st = _race_lane(2)
    assert isinstance(engines[0].comm, DP.TraceComm), "world>1 engines must trace by default (mandate 2)"


def _cp_prompt(rounds, sid, n):
    s = DP.Stream(sid, list(range(n)), 4, -1)
    s.cp = True
    rounds.add(s)
    return s


def test_cp_slice_shortest_first():
    rounds = DP.RoundBuilder()
    _cp_prompt(rounds, 0, 8192)
    _cp_prompt(rounds, 1, 2048)
    got = [rounds.cp_slice(1024, 4, 8)[2] for _ in range(5)]
    assert got == [1, 1, 1, 1, 0], got             # the short one's two rounds of two slices, then the long one
    narrow = _cp_prompt(rounds, 2, 256)
    op = rounds.cp_slice(1024, 4, 8)
    assert op[2] == narrow.sid and op[7] == 8 and not narrow.left, op      # overtakes between two slices; a narrow round takes more layers
    return {"slices": got}


def test_cp_slice_oldest_prompt_gets_its_turn():
    """A steady run of shorter prompts never lets the long one in by size alone; its turn every ``oldest`` slices does,
    every one of its rows goes through once and in order, and a short prompt still overtakes it between two of those turns."""

    def run(oldest, cap):
        rounds, width, total = DP.RoundBuilder(), 1024, 4
        long = _cp_prompt(rounds, 0, 8192)
        short, sid, mine, count = _cp_prompt(rounds, 1, 2048), 2, [], 0
        for count in range(1, cap + 1):
            op = rounds.cp_slice(width, 1, total, oldest)
            if op[2] == long.sid:
                mine.append(op)
                if not long.needs_fill and not long.left:
                    return count, mine
            if not short.needs_fill and not short.left:
                short, sid = _cp_prompt(rounds, sid, 2048), sid + 1
        return None, mine

    count, mine = run(0, 4000)
    assert count is None and not mine, (count, len(mine))
    count, mine = run(8, 4000)
    assert 8 * 8 * 3 < count <= 8 * 8 * 4, count       # 8 rounds of 4 layers, a layer every 8 slices until its tail is the shortest
    starts = [op[3] for op in mine if op[5]]
    assert starts == [i * 1024 for i in range(8)], starts
    assert all(op[4] == 1024 and op[7] == 1 for op in mine), mine[:2]
    assert sum(len(op[5]) for op in mine) == 8192
    return {"slices to finish": count}


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    import faulthandler

    faulthandler.dump_traceback_later(180, exit=True)       # a barrier deadlock dumps every thread's stack
    fails = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}", flush=True)
            traceback.print_exc()
    faulthandler.cancel_dump_traceback_later()
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed", flush=True)
    sys.exit(1 if fails else 0)
