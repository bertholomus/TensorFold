"""The bailing_v3 engine glue on the tiny fake: drafted == serial (greedy, 64 tokens) and
CUDA-graph replay == eager, bit for bit, on one rank (TP2 rank bit-identity is test 8's).

- drafted == serial: mtp_decode's reply equals serial_decode's from the same prefill, token for token, with a fixed
  depth-3 policy under greedy sampling (drafts never change the reply);
- graph replay == eager: an Engine with captured graphs and one without produce the same logits for the same
  committed sequence, including MTP draft steps and windows of every captured row count;
- packed == per-expert: the routed trellises in one allocation a projection (weights.Slab) or one an expert give the
  same bytes behind every expert pointer and the same logits and states;
- latent row blocks and prompt chunks: a 600-token prompt committed with the latent attention's partials one row a
  block (TF_LING_LATENT_MB's floor) leaves the same states and caches as all rows a block, and the windows after it
  give the same logits; in chunks of 128 rows (TF_LING_PREFILL_ROWS) it commits every row and gives the same bits
  run twice, but not one chunk's bits (the kernels go by a chunk's row count: glm_moe_dsa/cuda/invariant.py).

GPU only (EXL3, KDA and the latent kernels are CUDA); runs on a GB10. The random weights carry no
meaning: only bits are compared.

Run: python -m pytest tests/test_bailing_10_engine.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "tests"))

torch = pytest.importorskip("torch")


def _forward_exists() -> bool:
    return (REPO / "src" / "tensorfold" / "families" / "bailing_v3" / "cuda" / "decode.py").is_file() \
        and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _forward_exists(), reason="the bailing_v3 engine has not landed, "
                                                             "or no CUDA device is present")

PROMPT = [3, 17, 200, 45, 9, 101, 66, 250, 1, 77, 31, 8]
N = 64
CAP = 256


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bit pattern (NaNs compare equal to themselves)."""

    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from bailing_fakes import write_checkpoint

    return write_checkpoint(tmp_path_factory.mktemp("ling_engine"))


def _load(folder, world: int = 1, pack: bool | None = None):
    from tensorfold.families.bailing_v3.cuda import weights as W

    old = os.environ.get("TF_TP_WORLD")
    os.environ["TF_TP_WORLD"] = str(world)
    try:
        return W.load(folder, rank=0, device="cuda", pack=pack)
    finally:
        if old is None:
            os.environ.pop("TF_TP_WORLD", None)
        else:
            os.environ["TF_TP_WORLD"] = old


def test_drafted_equals_serial_greedy(tiny):
    from tensorfold.families.bailing_v3.cuda.decode import DepthPolicy, Engine, mtp_decode, prefill, serial_decode

    serial_e = Engine(_load(tiny), capacity=CAP, graphs=False)
    draft_e = Engine(_load(tiny), capacity=CAP, graphs=False)
    first = prefill(serial_e, PROMPT, None)
    assert prefill(draft_e, PROMPT, None) == first
    ref = serial_decode(serial_e, first, N, None)
    got = mtp_decode(draft_e, first, N, None, policy=DepthPolicy(3, fixed=True))
    assert got.tokens == ref.tokens, (got.tokens[:16], ref.tokens[:16])
    assert got.drafted >= 1 and got.depths and max(got.depths) >= 1   # the run drafted (or this says nothing)


def test_graph_replay_equals_eager(tiny):
    from tensorfold.families.bailing_v3.cuda.decode import Engine, commit, prefill
    from tensorfold.families.bailing_v3.cuda.graphs import Graphs

    eager = Engine(_load(tiny), capacity=CAP, graphs=False)
    graf = Engine(_load(tiny), capacity=CAP, graphs=False)
    graf.graphs = Graphs(graf, main_rows=(1, 2, 3, 4, 5, 6, 7, 8), mtp_rows=(1, 2, 3))
    graf.reset()                                    # the capture's warm-ups wrote the caches
    assert graf.graphs.main and graf.graphs.mtp
    prefill(eager, PROMPT, None)
    prefill(graf, PROMPT, None)
    # decode windows of every captured row count: eager and replayed logits must be the same bits
    g = torch.Generator().manual_seed(11)
    for R in (1, 2, 3, 4, 5, 6, 7, 8):
        toks = [int(t) for t in torch.randint(0, eager.w.cfg.vocab, (R,), generator=g)]
        a = eager.forward(toks).clone()
        b = graf.forward(toks).clone()
        assert _same(a, b), f"window of {R} rows"
        commit(eager.w, eager.st, eager.buf, R, R)
        commit(graf.w, graf.st, graf.buf, R, R)
    assert graf.replays["main"] > 0                  # the windows actually replayed graphs
    # MTP draft steps: one more 3-row window gives both engines the same hidden rows to draft from
    toks = [int(t) for t in torch.randint(0, eager.w.cfg.vocab, (3,), generator=g)]
    a = eager.forward(toks).clone()
    b = graf.forward(toks).clone()
    assert _same(a, b)
    ha = eager.buf.fnormed[:3].clone()
    assert _same(ha, graf.buf.fnormed[:3])
    commit(eager.w, eager.st, eager.buf, 3, 3)
    commit(graf.w, graf.st, graf.buf, 3, 3)
    for n in (1, 2, 3):
        mts = [int(t) for t in torch.randint(0, eager.w.cfg.vocab, (n,), generator=g)]
        x = eager.mtp(mts, ha[:n]).clone()
        y = graf.mtp(mts, ha[:n]).clone()
        assert _same(x, y), f"mtp step of {n} rows"
    assert graf.replays["mtp"] > 0
    # and the states agree after it all
    assert _same(graf.st.rec, eager.st.rec) and _same(graf.st.conv, eager.st.conv)


def _routed(w):
    """The prepared routed layers (exl3.experts.Exl3RoutedExperts) of every MoE layer, the MTP layer's last."""

    found = [layer.moe.experts.ex for layer in w.layers if layer.moe is not None]
    if w.mtp is not None and w.mtp.layer.moe is not None:
        found.append(w.mtp.layer.moe.experts.ex)
    return found


def test_packed_experts_give_the_same_bits(tiny):
    from tensorfold.families.bailing_v3.cuda.decode import Engine, commit, prefill

    sep = Engine(_load(tiny, pack=False), capacity=CAP, graphs=False)
    pak = Engine(_load(tiny, pack=True), capacity=CAP, graphs=False)
    pairs = list(zip(_routed(sep.w), _routed(pak.w)))
    assert len(pairs) == 3                                        # layers 1 and 2, and the MTP layer's
    for a, b in pairs:
        assert len(a.keep) == len(b.keep) == 3 * a.count          # gate, up and down trellises, expert order
        for ta, tb in zip(a.keep, b.keep):
            assert _same(ta, tb)
        assert len({t.untyped_storage().data_ptr() for t in b.keep}) == 3      # one allocation a projection
        assert len({t.untyped_storage().data_ptr() for t in a.keep}) == len(a.keep)
        for f in ("suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d", "gate_k2", "up_k2", "down_k2"):
            assert _same(getattr(a, f), getattr(b, f)), f
        assert all(int(p) % 512 == 0 for p in b.gate_ptr.tolist() + b.up_ptr.tolist() + b.down_ptr.tolist())
    assert prefill(pak, PROMPT, None) == prefill(sep, PROMPT, None)
    assert _same(pak.last_hidden, sep.last_hidden)
    g = torch.Generator().manual_seed(17)
    for R in (1, 2, 3, 4, 5, 6, 7, 8):
        toks = [int(t) for t in torch.randint(0, sep.w.cfg.vocab, (R,), generator=g)]
        assert _same(pak.forward(toks).clone(), sep.forward(toks).clone()), f"window of {R} rows"
        commit(sep.w, sep.st, sep.buf, R, R)
        commit(pak.w, pak.st, pak.buf, R, R)
    toks = [int(t) for t in torch.randint(0, sep.w.cfg.vocab, (3,), generator=g)]
    assert _same(pak.forward(toks).clone(), sep.forward(toks).clone())
    hidden = sep.buf.fnormed[:3].clone()
    assert _same(pak.buf.fnormed[:3], hidden)
    commit(sep.w, sep.st, sep.buf, 3, 3)
    commit(pak.w, pak.st, pak.buf, 3, 3)
    for n in (1, 2, 3):
        mts = [int(t) for t in torch.randint(0, sep.w.cfg.vocab, (n,), generator=g)]
        assert _same(pak.mtp(mts, hidden[:n]).clone(), sep.mtp(mts, hidden[:n]).clone()), f"mtp step of {n} rows"
    assert _same(pak.st.rec, sep.st.rec) and _same(pak.st.conv, sep.st.conv)


def _caches(st):
    """A state's KDA states, conv windows and latent caches (the MTP layer's too), in a fixed order."""

    out = [st.rec, st.conv, *st.kc, *st.pc]
    if getattr(st, "mtp_kc", None) is not None:
        out += [st.mtp_kc, st.mtp_pc]
    return out


def test_latent_blocks_keep_the_bits_and_prompt_chunks_commit(tiny, monkeypatch):
    from tensorfold.families.bailing_v3 import memory
    from tensorfold.families.bailing_v3.cuda.decode import Engine, commit, prefill
    from tensorfold.families.bailing_v3.cuda.engine import prompt_chunk_rows

    assert prompt_chunk_rows("") == 2048 and prompt_chunk_rows("512") == 512
    for bad in ("0", "100", "-512", "512.0", "32768"):
        with pytest.raises(ValueError):
            prompt_chunk_rows(bad)
    g = torch.Generator().manual_seed(23)
    prompt = [int(t) for t in torch.randint(0, 256, (600,), generator=g)]
    cap = 1024
    ref = Engine(_load(tiny), capacity=cap, prefill_rows=1024, graphs=False)      # one chunk
    monkeypatch.setattr(memory, "LATENT_MB", 0)                # memory.latent_rows' floor: one row a block
    blocks = Engine(_load(tiny), capacity=cap, prefill_rows=1024, graphs=False)
    monkeypatch.undo()
    assert blocks.pbuf.lat_s.part_rows == 1 < ref.pbuf.lat_s.part_rows
    small = [Engine(_load(tiny), capacity=cap, prefill_rows=128, graphs=False) for _ in range(2)]   # five chunks
    first = prefill(ref, prompt, None)
    assert prefill(blocks, prompt, None) == first
    assert _same(blocks.last_hidden, ref.last_hidden)
    assert all(_same(a, b) for a, b in zip(_caches(blocks.st), _caches(ref.st)))
    # 128-row chunks take other kernels than one 600-row chunk (the EXL3 linear's prompt GEMM runs past 128 rows only):
    # other bits, so the chunked prompt is held to its own run's bits and to committing every row
    assert prefill(small[0], prompt, None) == prefill(small[1], prompt, None)
    assert small[0].st.pos == small[1].st.pos == ref.st.pos == len(prompt)
    assert _same(small[0].last_hidden, small[1].last_hidden)
    assert all(_same(a, b) for a, b in zip(_caches(small[0].st), _caches(small[1].st)))
    for R in (1, 3, 8):
        toks = [int(t) for t in torch.randint(0, 256, (R,), generator=g)]
        assert _same(blocks.forward(toks).clone(), ref.forward(toks).clone()), f"window of {R} rows"
        assert _same(small[0].forward(toks).clone(), small[1].forward(toks).clone()), f"window of {R} rows"
        for e in (ref, blocks, *small):
            commit(e.w, e.st, e.buf, R, R)
    assert all(_same(a, b) for a, b in zip(_caches(blocks.st), _caches(ref.st)))
