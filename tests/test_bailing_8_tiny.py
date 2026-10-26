"""The tiny end-to-end model on tests/bailing_fakes.py's checkpoint (2 KDA + 1 MLA layers, 4
experts, the MTP layer; random EXL3 groups): the forward's exactness rules, bit for bit.

- window == serial: a decode window's rows equal one-row forwards, and keeping a prefix of it (commit's KDA replay
  and conv shift) leaves the serial state;
- the MTP layer: a window of draft rows equals one-row MTP steps (logits and final_layernorm rows);
- TP2: two ranks (one thread each on one GPU, host-side collectives between them) hold bit-identical hidden rows,
  latent caches and MTP caches after a prompt (in the engine's order: the chunk, the MTP layer on its rows, the
  commit) and decode steps;
- TP4: the same with four ranks on bailing_fakes.CFG_TP4 in the conversion's fused KDA layout, whose 1,280-wide
  experts split 384/384/256/256 as the real ones do. Past two ranks a prompt chunk's partials take glm_moe_dsa's
  two-hop "rows" reduce (grouped point-to-point transfers, uneven shares when the rows do not divide by four).

GPU only (EXL3, KDA and the latent kernels are CUDA); runs on a GB10. The random weights carry no meaning:
only bits are compared.

Run: python -m pytest tests/test_bailing_8_tiny.py -q
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "tests"))

torch = pytest.importorskip("torch")


def _forward_exists() -> bool:
    return (REPO / "src" / "tensorfold" / "families" / "bailing_v3" / "cuda" / "forward.py").is_file() \
        and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _forward_exists(), reason="the bailing_v3 forward is missing, "
                                                               "or no CUDA device is present")

PROMPT = [3, 17, 200, 45, 9, 101, 66, 250, 1, 77, 31, 8]
NEXT = [12, 99, 140, 7]
CAP = 64


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bit pattern (NaNs compare equal to themselves)."""

    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from bailing_fakes import write_checkpoint

    return write_checkpoint(tmp_path_factory.mktemp("ling_tiny"))


@pytest.fixture(scope="module")
def tiny_tp4(tmp_path_factory):
    from bailing_fakes import CFG_TP4, write_checkpoint

    return write_checkpoint(tmp_path_factory.mktemp("ling_tiny_tp4"), cfg=CFG_TP4, fused_qkv=True)


def _load(folder, world: int, rank: int, monkeypatch):
    from tensorfold.families.bailing_v3.cuda import weights as W

    monkeypatch.setenv("TF_TP_WORLD", str(world))
    return W.load(folder, rank=rank, device="cuda")


def _prefill(w, st, pb, tokens):
    from tensorfold.families.bailing_v3.cuda import forward as F

    F.forward(w, st, pb, tokens)
    F.commit(w, st, pb, len(tokens), len(tokens))


def test_window_equals_serial_and_commit_keeps_the_serial_state(tiny, monkeypatch):
    from tensorfold.families.bailing_v3.cuda import forward as F

    w = _load(tiny, 1, 0, monkeypatch)
    pb = F.Buffers(w, 32, CAP, prefill=True)
    db = F.Buffers(w, 8, CAP)
    st = F.State(w, CAP, 8)
    _prefill(w, st, pb, PROMPT)
    torch.cuda.synchronize()
    # serial: one row at a time, the state after each kept
    serial = st.clone()
    rows, after = [], []
    for t in NEXT:
        rows.append(F.forward(w, serial, db, [t]).clone()[0])
        F.commit(w, serial, db, 1, 1)
        after.append(serial.clone())
    # a window of the four rows: every row's logits are the serial step's
    win = st.clone()
    out = F.forward(w, win, db, NEXT).clone()
    for r in range(len(NEXT)):
        assert _same(out[r], rows[r]), r
    # keep two rows: the KDA states, conv windows and position are those after two serial steps
    F.commit(w, win, db, len(NEXT), 2)
    ref = after[1]
    assert win.pos == ref.pos
    # a commit flips cur once: the window committed once and the serial run twice, so their live
    # buffers sit at opposite parities; what those buffers hold must be the same bits
    assert win.cur == [1 - c for c in st.cur] and ref.cur == st.cur
    assert _same(win.rec[win.cur[0]], ref.rec[ref.cur[0]]) and _same(win.conv, ref.conv)
    for a, b in zip(win.kc + win.pc, ref.kc + ref.pc):
        assert _same(a[:win.pos], b[:ref.pos])
    # and the next step from there is the serial third step
    nxt = F.forward(w, win, db, [NEXT[2]]).clone()
    assert _same(nxt[0], rows[2])


def test_mtp_window_equals_one_row_steps(tiny, monkeypatch):
    from tensorfold.families.bailing_v3.cuda import forward as F
    from tensorfold.families.bailing_v3.cuda import mtp as M

    w = _load(tiny, 1, 0, monkeypatch)
    assert w.mtp is not None
    db = F.Buffers(w, 8, CAP)
    g = torch.Generator().manual_seed(5)
    hidden = torch.randn((4, w.cfg.hidden), generator=g).to(torch.bfloat16).cuda()
    toks = [4, 5, 6, 7]
    a = F.State(w, CAP, 8)
    logits = M.mtp_forward(w, a, db, toks, hidden, last_only=False).clone()
    final = db.fnormed[:4].clone()
    b = F.State(w, CAP, 8)
    for r in range(4):
        one = M.mtp_forward(w, b, db, toks[r:r + 1], hidden[r:r + 1], last_only=True).clone()
        assert _same(one[0], logits[r]), r
        assert _same(db.fnormed[0], final[r]), r
        b.set_mtp_len(b.mtp_len + 1)
    assert _same(a.mtp_kc[:4], b.mtp_kc[:4]) and _same(a.mtp_pc[:4], b.mtp_pc[:4])


class _Hub:
    """The ranks' collectives through host memory: the all-gather (each rank deposits its partial, every rank copies
    all in rank order) and grouped transfers (each rank posts its sends, every rank takes the ones addressed to it)."""

    def __init__(self, world: int, solo: int | None = None) -> None:
        self.world, self.solo = world, solo
        self.barrier = threading.Barrier(world)
        self.slots: list = [None] * world
        self.mail: list = [[[] for _ in range(world)] for _ in range(world)]     # [sender][receiver]: sends in order


class _Comm:
    def __init__(self, hub: _Hub, rank: int) -> None:
        self.hub, self.rank = hub, rank

    def all_gather(self, src: torch.Tensor, out: torch.Tensor) -> None:
        hub, n = self.hub, src.numel()
        torch.cuda.current_stream().synchronize()
        if hub.solo is not None:                 # warm-up: one rank alone, the other's partial zero
            out.zero_()
            out[self.rank * n:(self.rank + 1) * n].copy_(src)
            return
        hub.slots[self.rank] = src.detach().clone()
        torch.cuda.current_stream().synchronize()
        hub.barrier.wait()
        for k in range(hub.world):
            out[k * n:(k + 1) * n].copy_(hub.slots[k])
        torch.cuda.current_stream().synchronize()
        hub.barrier.wait()

    def grouped(self, sends, recvs) -> None:
        """comm.Comm.grouped: each (tensor, peer) of ``sends`` lands in that peer's receive from this rank, the i-th
        send to a peer in its i-th receive from this rank, as NCCL pairs them (empty ones skipped). NCCL moves
        data_ptr and numel, so every tensor must be contiguous and each receive the size and dtype of its send."""

        hub = self.hub
        for t, _ in [*sends, *recvs]:
            assert t.is_contiguous(), "grouped: a non-contiguous tensor (NCCL would move other bytes)"
        sends = [(t, p) for t, p in sends if t.numel()]
        recvs = [(t, p) for t, p in recvs if t.numel()]
        torch.cuda.current_stream().synchronize()
        if hub.solo is not None:                 # warm-up: one rank alone, what the others send zero
            mine = [t.detach().clone() for t, p in sends if p == self.rank]
            for t, p in recvs:
                if p == self.rank:
                    t.view(-1).copy_(mine.pop(0).reshape(-1))
                else:
                    t.zero_()
            return
        for peer in range(hub.world):
            hub.mail[self.rank][peer] = [t.detach().clone() for t, p in sends if p == peer]
        torch.cuda.current_stream().synchronize()
        hub.barrier.wait()
        got = [list(hub.mail[peer][self.rank]) for peer in range(hub.world)]
        for t, p in recvs:
            assert got[p], f"grouped: rank {self.rank} receives from rank {p}, which sent it nothing"
            src = got[p].pop(0)
            assert src.dtype == t.dtype and src.numel() == t.numel(), \
                f"grouped: rank {p} sent {src.numel()} {src.dtype} to rank {self.rank}, which receives " \
                f"{t.numel()} {t.dtype}"
            t.view(-1).copy_(src.reshape(-1))
        assert not any(got), f"grouped: rank {self.rank} never received a send addressed to it"
        torch.cuda.current_stream().synchronize()
        hub.barrier.wait()


def test_tp2_ranks_are_bit_identical(tiny, monkeypatch):
    _ranks_are_bit_identical(tiny, 2, monkeypatch)


def test_tp4_ranks_are_bit_identical(tiny_tp4, monkeypatch):
    _ranks_are_bit_identical(tiny_tp4, 4, monkeypatch)


def _ranks_are_bit_identical(folder, world: int, monkeypatch) -> None:
    from tensorfold.families.bailing_v3.cuda import forward as F
    from tensorfold.families.bailing_v3.cuda import mtp as M

    ws = [_load(folder, world, r, monkeypatch) for r in range(world)]
    assert all(w.mtp is not None for w in ws)
    run = {}

    def rank_job(r: int, hub: _Hub, out: dict) -> None:
        try:
            w = ws[r]
            w.comm = _Comm(hub, r)
            with torch.cuda.stream(torch.cuda.Stream()):
                pb = F.Buffers(w, 32, CAP, prefill=True)
                db = F.Buffers(w, 8, CAP)
                st = F.State(w, CAP, 8)
                n = len(PROMPT)
                # decode.prefill's order: the chunk, the MTP layer on its rows (_absorb_rows; 11 rows, so four ranks'
                # shares of a prompt reduce are uneven), then the commit
                F.forward(w, st, pb, PROMPT)
                taps = [("the prompt's final-normed rows", pb.fnormed[:n].clone())]
                M.mtp_forward(w, st, pb, PROMPT[1:], pb.fnormed[:n - 1])
                st.set_mtp_len(st.mtp_len + n - 1)
                taps.append(("the MTP layer's last prompt row", pb.fnormed[:1].clone()))
                F.commit(w, st, pb, n, n)
                for i, t in enumerate(NEXT[:2]):
                    F.forward(w, st, db, [t])
                    F.commit(w, st, db, 1, 1)
                    taps.append((f"decode step {i + 1}'s row", db.fnormed[:1].clone()))
                F.forward(w, st, db, NEXT[2:])
                taps.append(("the 2-row window's rows", db.fnormed[:2].clone()))
                taps.append(("the 2-row window's x", db.x[:2].clone()))
                taps += [(f"latent cache {i}", x.clone()) for i, x in enumerate(st.kc)]
                taps += [(f"rope-key cache {i}", x.clone()) for i, x in enumerate(st.pc)]
                taps += [("the MTP latent cache", st.mtp_kc[:st.mtp_len].clone()),
                         ("the MTP rope-key cache", st.mtp_pc[:st.mtp_len].clone())]
                torch.cuda.current_stream().synchronize()
                out[r] = taps
        except BaseException as e:                     # noqa: BLE001 - re-raised in the test thread
            out[r] = e
            if hub.solo is None and not hub.barrier.broken:
                hub.barrier.abort()

    # warm-up, one rank at a time (kernels and extensions compile here, never in two threads at once)
    for r in range(world):
        rank_job(r, _Hub(world, solo=r), run)
        assert not isinstance(run[r], BaseException), run[r]
    hub = _Hub(world)
    run.clear()
    threads = [threading.Thread(target=rank_job, args=(r, hub, run), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert not any(t.is_alive() for t in threads), "a rank never finished (a collective the others did not make?)"
    for r in range(world):
        assert not isinstance(run.get(r), BaseException), run.get(r)
    ref = run[0]
    for r in range(1, world):
        assert [name for name, _ in run[r]] == [name for name, _ in ref]
        for (name, a), (_, b) in zip(ref, run[r]):          # rows first, in the order they were computed
            assert _same(a, b), f"rank {r}: {name} differs from rank 0's"
