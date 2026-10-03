"""GLM-5.3's concurrent decoding, the host side (CPU): rank 0's steps reach every follower in order over the TCP
links, a round's tables give each stream's rows its positions, slot base and segment, with the round's passes
and bucket, and rank 0 picks what fills before a round (short prompts batched, quick ones unpaced, long ones paced)."""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest


def test_link_sends_every_step_to_every_follower_in_order():
    from torch.distributed import TCPStore

    from tensorfold.families.glm_moe_dsa.cuda.multi import Link

    world = 4
    store = TCPStore("127.0.0.1", 0, None, True, timeout=timedelta(seconds=30), wait_for_workers=False)
    port = store.port
    leader = Link(store, rank=0, world=world, host="127.0.0.1")
    got = {r: [] for r in range(1, world)}

    def follow(rank: int) -> None:
        mine = TCPStore("127.0.0.1", port, None, False, timeout=timedelta(seconds=30))
        link = Link(mine, rank=rank, world=world, host="127.0.0.1")
        while True:
            op = link.receive()
            if op is None or op[0] == "stop":
                return
            got[rank].append(op)

    threads = [threading.Thread(target=follow, args=(r,)) for r in range(1, world)]
    for t in threads:
        t.start()
    ops = [["admit", [1, 2, 3], 8, None, True, True], ["round", []], ["round", [0]], ["finish", [0]], ["drop"]]
    for op in ops:
        leader.send(op)
    leader.send(["stop"])
    for t in threads:
        t.join(timeout=30)
    assert all(got[r] == ops for r in got), got


def test_tables_give_each_stream_its_rows():
    from tensorfold.families.glm_moe_dsa.cuda.rows import Tables

    t = Tables(16, 4, None)
    n = t.fill([(0, 5, 4), (8192, 2047, 2), (16384, 6000, 4)], 2048)
    assert n == 10 and t.R == 10
    assert t.pos_host[:n].tolist() == [5, 6, 7, 8, 2047, 2048, 6000, 6001, 6002, 6003]
    assert t.base_host[:n].tolist() == [0] * 4 + [8192] * 2 + [16384] * 4
    assert t.seg_host.tolist() == [[0, 4], [4, 2], [6, 4], [0, 0]]
    assert t.key() == (10, True, True, 8192)             # dense rows, sparse rows, the deepest row's bucket
    assert t.fill([(0, 100, 1)], 2048) == 1 and t.key() == (1, True, False, 0)
    assert t.fill([(0, 3000, 4), (8192, 2048, 1)], 2048) == 5 and t.key() == (5, False, True, 4096)
    with pytest.raises(ValueError):
        t.fill([(0, 0, 8), (0, 0, 8), (0, 0, 8)], 2048)      # more rows than the tables hold


def test_a_lost_rank_breaks_the_lane_on_every_rank():
    """A follower that goes away (its process killed: its sockets close) breaks the lane on rank 0 at once, and rank 0
    tells the other followers: every rank aborts its RDMA waits and NCCL, nobody keeps waiting."""

    import socket
    import time
    from types import SimpleNamespace

    from torch.distributed import TCPStore

    from tensorfold.families.glm_moe_dsa.cuda.multi import Watchdog

    world = 4
    store = TCPStore("127.0.0.1", 0, None, True, timeout=timedelta(seconds=30), wait_for_workers=False)
    calls = {r: [] for r in range(world)}

    def comm(rank: int):
        return SimpleNamespace(rdma=SimpleNamespace(abort=lambda why, r=rank: calls[r].append(("rdma", why))),
                               nccl=SimpleNamespace(abort=lambda r=rank: calls[r].append(("nccl",)),
                                                    async_error=lambda: 0))

    dogs = {0: Watchdog(comm(0), store, rank=0, world=world, host="127.0.0.1", timeout=30, every=0.2)}
    for r in range(1, world):
        client = TCPStore("127.0.0.1", store.port, None, False, timeout=timedelta(seconds=30))
        dogs[r] = Watchdog(comm(r), client, rank=r, world=world, host="127.0.0.1", timeout=30, every=0.2)
    time.sleep(1.0)
    assert all(d.broken is None for d in dogs.values())         # heartbeats keep an idle lane alive
    dead = dogs[2]
    dead.broken = "killed"                                      # its threads stop; its process "dies":
    for sock in dead.socks:
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
    deadline = time.time() + 5
    while time.time() < deadline and any(dogs[r].broken is None for r in (0, 1, 3)):
        time.sleep(0.05)
    for r in (0, 1, 3):
        assert dogs[r].broken is not None, r
        assert ("nccl",) in calls[r] and any(c[0] == "rdma" for c in calls[r]), calls[r]
    assert "rank 2" in dogs[0].broken and dogs[1].broken == "rank 0 lost a rank"


def _planner(filling, decoding=0, fill_rows=128, tiny=63, since=0.0, chunk=1.0):
    from types import SimpleNamespace

    from tensorfold.families.glm_moe_dsa.cuda import multi

    m = multi.MultiDecoder.__new__(multi.MultiDecoder)
    m.fbuf = object() if tiny else None
    m.tiny = tiny
    m.filling = [SimpleNamespace(sid=i, prompt=[1] * n, filled=f) for i, (n, f) in enumerate(filling)]
    m.streams = {100 + i: SimpleNamespace(done=False) for i in range(decoding)}
    m.since_fill, m.chunk_s = since, chunk
    return m


def test_short_prompts_fill_together_and_quick_ones_before_the_round(monkeypatch):
    from tensorfold.families.glm_moe_dsa.cuda import multi

    monkeypatch.setattr(multi, "FILL_ROWS", 128)
    monkeypatch.setattr(multi, "QUICK_ROWS", 1024)
    assert _planner([(20, 0)] * 4, decoding=2)._fill_choice() == [0, 1, 2, 3]
    assert _planner([(60, 0)] * 3)._fill_choice() == [0, 1]                    # a batched fill holds 128 rows
    assert _planner([(20, 0), (500, 0), (30, 0)], decoding=3)._fill_choice() == [0, 1, 2]
    assert _planner([(600, 0), (600, 0)], decoding=1)._fill_choice() == [0]     # quick ones up to 1,024 rows a step
    assert _planner([(2000, 0)], decoding=1)._fill_choice() is None             # past QUICK_ROWS: paced
    assert _planner([(9000, 8192)], decoding=1)._fill_choice() == [0]           # a long prompt's last chunk is quick
    assert _planner([(20, 0)], tiny=0)._fill_choice() == [0]                    # no batched fills: quick


def test_long_prompts_chunks_are_paced(monkeypatch):
    from tensorfold.families.glm_moe_dsa.cuda import multi

    monkeypatch.setattr(multi, "QUICK_ROWS", 1024)
    monkeypatch.setattr(multi, "DECODE_SHARE", 0.5)
    assert _planner([(20000, 0)])._fill_choice() == 0                           # nothing decodes: a chunk at once
    assert _planner([(20000, 0)], decoding=1, since=0.5, chunk=1.0)._fill_choice() is None
    assert _planner([(20000, 0)], decoding=1, since=1.0, chunk=1.0)._fill_choice() == 0
    assert _planner([], decoding=1)._fill_choice() is None


def test_a_prompt_resumes_from_the_warm_slot_sharing_the_most_chunks(monkeypatch):
    from types import SimpleNamespace

    from tensorfold.families.glm_moe_dsa.cuda import decode, invariant, multi

    monkeypatch.setattr(multi, "KEEP", True)
    monkeypatch.setattr(invariant, "INVARIANT", False)            # chunk-boundary cuts
    step = 1024
    base = list(range(5000))
    assert decode.resume_cut(base, base + [7, 8], step) == 4096           # chunk boundaries only
    assert decode.resume_cut(base, base, step) == 4096                    # a resend: a chunk is left to fill
    assert decode.resume_cut(base[:4096], base, step) == 3072             # a kept row short of the kept end
    assert decode.resume_cut(base, base[:2000] + [9] * 3000, step) == 1024
    assert decode.resume_cut(None, base, step) == 0
    m = multi.MultiDecoder.__new__(multi.MultiDecoder)
    m.prefill_rows, m.extents, m.short = step, None, 63
    m.slots = [SimpleNamespace(kept=None, used=0) for _ in range(4)]
    m.slots[1].kept, m.slots[1].used = base[:2500], 5
    m.slots[2].kept, m.slots[2].used = base[:4500], 3
    m.slots[3].kept, m.slots[3].used = [1, 2, 3], 1
    m.free = [0, 1, 2, 3]
    assert m._pick(base + [1]) == (2, 4096)                               # the longest shared chunks
    assert m._pick([5] * 3000) == (0, 0)                                  # none shared: a slot never kept
    m.free = [1, 2, 3]
    assert m._pick([5] * 3000) == (3, 0)                                  # else the one kept longest ago
    monkeypatch.setattr(multi, "KEEP", False)
    assert decode.resume_cut(base, base + [7], step) == 0
    m.free = [0, 1, 2, 3]
    assert m._pick(base + [1]) == (0, 0)


def test_extents_fit_first_and_merge_when_freed():
    from tensorfold.families.glm_moe_dsa.cuda.multi import Extents

    e = Extents(10 * 2048 + 100)                       # the tail past whole 2,048-row steps is not used
    assert e.total == 20480 and e.largest() == 20480
    a = e.take(3000)                                   # 2 steps
    b = e.take(2048)
    c = e.take(1)
    assert (a, b, c) == (0, 4096, 6144) and e.largest() == 12288
    e.give(b, 2048)
    assert e.take(4096) == 8192                        # the freed step is too small: first fit past it
    assert e.take(2048) == 4096                        # it fits there
    e.give(a, 3000)
    e.give(4096, 2048)
    e.give(c, 1)
    assert e.gaps[0] == (0, 8192)                      # neighbours merged
    assert e.take(20480) is None


def test_extent_admission_reuses_kept_extents_and_evicts_the_oldest(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from tensorfold.cuda.memory_gate import NoRoom
    from tensorfold.families.glm_moe_dsa.cuda import invariant, multi

    monkeypatch.setattr(multi, "KEEP", True)
    monkeypatch.setattr(invariant, "INVARIANT", False)
    monkeypatch.setattr(multi, "slot_state", lambda pool, lo, hi: SimpleNamespace(lo=lo, hi=hi))
    m = multi.MultiDecoder.__new__(multi.MultiDecoder)
    m.prefill_rows, m.pool, m.short = 1024, None, 63
    m.extents = multi.Extents(8 * 2048)
    m.slots = [multi.Slot(i, 0, None) for i in range(3)]
    m.free = [0, 1, 2]
    m._place(m.slots[0], 5000)                         # 3 steps
    m._place(m.slots[1], 6000)                         # 3 steps
    assert (m.slots[0].base, m.slots[0].size, m.slots[1].base) == (0, 6144, 6144)
    m.slots[0].kept, m.slots[0].used = list(range(4000)), 1
    m.slots[1].kept, m.slots[1].used = list(range(9, 3000)), 2
    assert m._pick(list(range(4500)), 5000) == (0, 3072)      # its kept prompt, and its extent holds the request
    assert m._pick(list(range(4500)), 7000)[1] == 0           # too small an extent: no resume from it
    m._place(m.slots[2], 6000)                         # 2 steps left: the oldest kept extent (slot 0) gives way
    assert m.slots[0].size == 0 and m.slots[0].kept is None and m.slots[2].base == 0
    m.free = [0, 1]
    with pytest.raises(NoRoom):
        m._place(m.slots[0], 16384)                    # slot 1 gives way too, and still no 8 free steps (2 is live)
    assert m.slots[1].size == 0


def test_invariant_blocks_cover_every_row_in_fixed_sizes():
    import torch

    from tensorfold.families.glm_moe_dsa.cuda import invariant

    assert invariant.ranges(2500, 1024) == [(0, 1024), (1024, 2048), (1476, 2500)]
    assert invariant.ranges(2048, 1024) == [(0, 1024), (1024, 2048)]
    assert invariant.ranges(300, 1024) == [(0, 1024)]
    buf = torch.arange(40 * 3, dtype=torch.float32).view(40, 3)
    view = buf[10:15]
    assert torch.equal(invariant.rows(view, 2, 8), buf[12:18])         # past the view, inside its storage
    assert invariant.rows(buf[30:], 0, 16) is None                    # past the storage
    seen = []

    def fn(x, y):
        seen.append(x.shape[0])
        y.copy_(x * 2)

    out = torch.zeros_like(buf)
    invariant.blocked(fn, 7, [buf[33:]], [out[33:]], M0=4)            # 7 rows from row 33: the last block is padded
    assert seen == [4, 4] and torch.equal(out[33:40], buf[33:40] * 2) and out[:33].abs().sum() == 0


def test_the_invariant_path_resumes_at_any_row(monkeypatch):
    from tensorfold.families.glm_moe_dsa.cuda import invariant, multi

    base = list(range(5000))
    monkeypatch.setattr(invariant, "INVARIANT", False)
    assert multi.resume_at(base, base + [7, 8], 1024) == 4096
    monkeypatch.setattr(invariant, "INVARIANT", True)
    assert multi.resume_at(base, base + [7, 8], 1024) == 4999          # every kept row but the last
    assert multi.resume_at(base, base[:2000] + [9] * 3000, 1024) == 1999
    assert multi.resume_at(base, base, 1024) == 4999                  # a resend: one row left to fill
    # a whole prompt of at most tiny_rows tokens fills with the decode kernels: no resume from or into one
    assert multi.resume_at(base[:40], base[:40] + [7] * 30, 1024, short=63) == 0
    assert multi.resume_at(base, base[:50], 1024, short=63) == 0
    assert multi.resume_at(base[:80], base[:80] + [7], 1024, short=63) == 79


def test_short_prompts_fill_with_the_decode_kernels_under_the_invariant_path(monkeypatch):
    from types import SimpleNamespace

    from tensorfold.families.glm_moe_dsa.cuda import invariant, multi

    monkeypatch.setattr(invariant, "INVARIANT", True)
    w = SimpleNamespace(cfg=SimpleNamespace(dense_limit=2048))
    assert multi.tiny_rows(w) > 0                                     # batched short fills stay on with it
    with invariant.suspended(True):
        assert invariant.INVARIANT is False
    assert invariant.INVARIANT is True
    with invariant.suspended(False):
        assert invariant.INVARIANT is True


def _drafter(probs, streams):
    """A MultiDecoder whose MTP steps and sampler are fakes: stream i (its position 100 i + 10) drafts token 100 + j at
    depth j with probability probs[i][j]."""

    from types import SimpleNamespace

    import torch

    from tensorfold.families.glm_moe_dsa.cuda import multi

    class St:
        def __init__(self, pos: int) -> None:
            self.pos, self.mtp_len, self.mtp_drafted = pos, pos, 0

        def set_mtp_len(self, n: int) -> None:
            self.mtp_len = n

    m = multi.MultiDecoder.__new__(multi.MultiDecoder)
    m.depth, m.accept = 3, {}
    m.w = SimpleNamespace(cfg=SimpleNamespace(dense_limit=2048))
    rows = SimpleNamespace(fill=lambda spans, limit: sum(n for *_, n in spans))
    m.mbuf = SimpleNamespace(rows_t=rows, hin=torch.zeros((16, 4)), fnormed=torch.zeros((16, 4)))
    m.buf = SimpleNamespace(fnormed=torch.zeros((16, 4)))
    ss = [SimpleNamespace(sid=i, done=False, count=100, out=[1], sampling=None, draft=True,
                          st=SimpleNamespace(base=0, st=St(100 * i + 10))) for i in range(streams)]
    m.streams = {s.sid: s for s in ss}
    m._mtp = lambda n, tokens, heads: torch.zeros((n, 8))

    def sample_spans(logits, spans, *, draft=False, probs=None):
        out = []
        for _, _, positions, _ in spans:
            sid, j = divmod(positions[0] - 11, 100)
            out.append([100 + j])
            if probs is not None:
                probs.append([probs_of[sid][j]])
        return out

    probs_of = probs
    m._sample_spans = sample_spans
    return m, ss


@pytest.mark.parametrize("streams,cut,want", [
    (2, 0.3, [[100, 101], [100, 101, 102]]),       # 0.9 x 0.5 = 0.45, x 0.5 = 0.225 < 0.3: stream 0 stops at two
    (2, 0.0, [[100, 101, 102], [100, 101, 102]]),  # no cut: the whole chain
    (1, 0.3, [[100, 101, 102]]),                  # one stream: no cut (its rows are cheap)
])
def test_the_draft_cut_ends_unlikely_chains_under_concurrency(monkeypatch, streams, cut, want):
    from tensorfold.families.glm_moe_dsa.cuda import multi

    monkeypatch.setattr(multi, "DRAFT_CUT", cut)
    monkeypatch.setattr(multi, "DRAFT_STATS", False)
    m, ss = _drafter({0: [0.9, 0.5, 0.5], 1: [0.95, 0.9, 0.9]}, streams)
    multi.MultiDecoder._draft_all(m, [(s, 0, 1, [7]) for s in ss])
    assert [s.drafts for s in ss] == want
    if cut and streams > 1:
        assert ss[0].draft_probs == [0.9, 0.45] and len(ss[1].draft_probs) == 3
        # the step that drafted the cut token ran (its MTP row goes back next round, as a rejected draft's does)
        assert ss[0].st.st.mtp_drafted == 2 and ss[1].st.st.mtp_drafted == 2
