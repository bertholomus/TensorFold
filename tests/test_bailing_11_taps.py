"""Block-drafter taps on the bailing_v3 forward: every forward copies the residual stream after
the tap layers, and the copies change nothing else.

- a tap holds the residual stream right after its layer (HF's hidden_states[layer + 1]), row for row;
- taps leave the logits and the committed state bit-identical to an untapped engine's;
- graph replay fills the same tap bits as eager windows of every captured row count;
- the last layer is refused (HF's last hidden state is model.norm's output, not a tap).

GPU only (EXL3, KDA and the latent kernels are CUDA); runs on a GB10. Run:
python -m pytest tests/test_bailing_11_taps.py -q
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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the bailing_v3 forward needs a CUDA device")

PROMPT = [3, 17, 200, 45, 9, 101, 66, 250, 1, 77, 31, 8]
CAP = 256
TAPS = (0, 1)                 # the tiny fake's layers 0 and 1 (KDA); layer 2, the last, is refused


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from bailing_fakes import write_checkpoint

    return write_checkpoint(tmp_path_factory.mktemp("ling_taps"))


def _load(folder):
    from tensorfold.families.bailing_v3.cuda import weights as W

    old = os.environ.get("TF_TP_WORLD")
    os.environ["TF_TP_WORLD"] = "1"
    try:
        return W.load(folder, rank=0, device="cuda")
    finally:
        if old is None:
            os.environ.pop("TF_TP_WORLD", None)
        else:
            os.environ["TF_TP_WORLD"] = old


def test_a_tap_is_the_stream_after_its_layer(tiny):
    from tensorfold.families.bailing_v3.cuda import forward as F
    from tensorfold.families.bailing_v3.cuda.decode import Engine, prefill

    e = Engine(_load(tiny), capacity=CAP, taps=TAPS)
    prefill(e, PROMPT, None)
    toks = [5, 42, 7, 99]
    e.forward(toks)
    got = e.tap_rows(len(toks)).clone()
    D = e.w.cfg.hidden
    assert got.shape == (len(toks), len(TAPS) * D)
    # the same window again, layer by layer (nothing is committed between: the forward rewrites the same scratch)
    w, st, b = e.w, e.st, e.buf
    R = F.stage(w, st, b, toks)
    F.embed(w, b, b.ids[:R], b.x[:R])
    b.rope.fill(st.pos_dev, R)
    nch = F.chunks_for(st, R)
    for layer in w.layers:
        F.layer_forward(layer, w, st, b, R, nch, st.pos)
        for slot in b.tap_at.get(layer.index, ()):
            assert _same(b.x[:R], got[:, slot * D:(slot + 1) * D]), f"tap after layer {layer.index}"


def test_taps_change_no_logits_and_no_state(tiny):
    from tensorfold.families.bailing_v3.cuda.decode import DepthPolicy, Engine, mtp_decode, prefill

    plain = Engine(_load(tiny), capacity=CAP)
    tapped = Engine(_load(tiny), capacity=CAP, taps=TAPS)
    first = prefill(plain, PROMPT, None)
    assert prefill(tapped, PROMPT, None) == first
    a = mtp_decode(plain, first, 32, None, policy=DepthPolicy(3, fixed=True))
    b = mtp_decode(tapped, first, 32, None, policy=DepthPolicy(3, fixed=True))
    assert a.tokens == b.tokens
    assert _same(plain.st.rec, tapped.st.rec) and _same(plain.st.conv, tapped.st.conv)
    toks = [11, 12, 13]
    assert _same(plain.forward(toks).clone(), tapped.forward(toks).clone())


def test_graph_replay_fills_the_eager_taps(tiny):
    from tensorfold.families.bailing_v3.cuda.decode import Engine, commit, prefill
    from tensorfold.families.bailing_v3.cuda.graphs import Graphs

    eager = Engine(_load(tiny), capacity=CAP, taps=TAPS)
    graf = Engine(_load(tiny), capacity=CAP, taps=TAPS)
    graf.graphs = Graphs(graf, main_rows=(1, 2, 3, 4, 5, 6, 7, 8), mtp_rows=(1,))
    graf.reset()
    prefill(eager, PROMPT, None)
    prefill(graf, PROMPT, None)
    g = torch.Generator().manual_seed(5)
    for R in range(1, 9):
        toks = [int(t) for t in torch.randint(0, eager.w.cfg.vocab, (R,), generator=g)]
        eager.forward(toks)
        graf.forward(toks)
        assert _same(eager.tap_rows(R), graf.tap_rows(R)), f"taps of a {R}-row window"
        commit(eager.w, eager.st, eager.buf, R, R)
        commit(graf.w, graf.st, graf.buf, R, R)
    assert graf.replays["main"] > 0


def test_prompt_chunks_fill_taps_and_the_last_layer_is_refused(tiny):
    from tensorfold.families.bailing_v3.cuda.decode import Engine, prefill

    e = Engine(_load(tiny), capacity=CAP, taps=TAPS)
    prefill(e, PROMPT, None)
    rows = e.tap_rows(len(PROMPT), e.pbuf)
    assert rows.shape == (len(PROMPT), len(TAPS) * e.w.cfg.hidden) and bool(torch.isfinite(rows.float()).all())
    with pytest.raises(ValueError, match="tap follows"):
        e.buf.set_taps((e.w.cfg.layers - 1,))
