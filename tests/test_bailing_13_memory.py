"""bailing_v3.memory, the startup estimate: every State and Buffers byte the CPU-built engine pieces allocate on the
tiny fake equals the estimate's, the EXL3 scratch rules equal the code they mirror, and Ling's own numbers
(11,520 cache bytes a token with the MTP layer, the partials at 262K) come out of the real config. The 262K levers:
the shared call-local scratch (TF_LING_SHARED_SCRATCH) priced byte for byte as decode.Engine builds it, and the
vocabulary-split embedding (TF_LING_EMBED_SPLIT): each rank's span, the rows forward.embed picks it by, the loader's
rows and the startup estimate's.

CPU only: forward.State and forward.Buffers build on the CPU from the fake checkpoint (the experts' EXL3 scratch is
made on CUDA only, so it is held to exl3.experts.Scratch built directly).

Run: python -m pytest tests/test_bailing_13_memory.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "tests"))

torch = pytest.importorskip("torch")

from tensorfold.families.bailing_v3 import memory  # noqa: E402

LING = REPO / "tests" / "fixtures" / "ling31" / "config.json"


def nbytes(obj, skip: tuple[str, ...] = (), seen: set | None = None, depth: int = 0) -> int:
    """Bytes of every tensor storage reachable from ``obj`` (each storage once), leaving out attributes ``skip``."""

    seen = set() if seen is None else seen
    if isinstance(obj, torch.Tensor):
        s = obj.untyped_storage()
        key = (s.data_ptr(), s.nbytes())
        if key in seen or not s.nbytes():
            return 0
        seen.add(key)
        return s.nbytes()
    if depth > 4:
        return 0
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(x, skip, seen, depth + 1) for x in obj)
    if isinstance(obj, dict):
        return sum(nbytes(x, skip, seen, depth + 1) for x in obj.values())
    attrs = getattr(obj, "__dict__", None)
    if attrs is None:
        return 0
    return sum(nbytes(x, skip, seen, depth + 1) for k, x in attrs.items() if k not in skip and k != "w")


def _load(folder: Path, world: int, rank: int = 0, mtp: bool = True):
    from tensorfold.families.bailing_v3.cuda import weights as W

    old = os.environ.get("TF_TP_WORLD")
    os.environ["TF_TP_WORLD"] = str(world)
    try:
        return W.load(folder, rank=rank, device="cpu", mtp=mtp)
    finally:
        if old is None:
            os.environ.pop("TF_TP_WORLD", None)
        else:
            os.environ["TF_TP_WORLD"] = old


@pytest.fixture(scope="module")
def fakes(tmp_path_factory):
    from bailing_fakes import CFG_TP4, write_checkpoint

    return {"tiny": write_checkpoint(tmp_path_factory.mktemp("ling_mem"), fused_qkv=True),
            "tp4": write_checkpoint(tmp_path_factory.mktemp("ling_mem_tp4"), cfg=CFG_TP4, fused_qkv=True)}


def _text(folder: Path) -> dict:
    from tensorfold.cuda.capacity import config

    return config(folder)


CASES = [("tiny", 1, 0, True), ("tiny", 1, 0, False), ("tiny", 2, 0, True), ("tiny", 2, 1, True),
         ("tp4", 4, 0, True), ("tp4", 4, 3, True)]


@pytest.mark.parametrize("fake,world,rank,mtp", CASES)
def test_state_is_priced_byte_for_byte(fakes, fake, world, rank, mtp):
    from tensorfold.families.bailing_v3.cuda import forward as F

    w = _load(fakes[fake], world, rank, mtp)
    r = memory.Rank.of(_text(fakes[fake]), world, rank, mtp=mtp)
    for capacity in (256, 600, 1100, 2568):
        got = nbytes(F.State(w, capacity, 8))
        assert got == memory.state_fixed(r, 8) + memory.cache_per_slot(r) * capacity, (capacity, got)


@pytest.mark.parametrize("fake,world,rank,mtp", CASES)
def test_buffers_are_priced_byte_for_byte(fakes, fake, world, rank, mtp, monkeypatch):
    from tensorfold.families.bailing_v3.cuda import forward as F

    w = _load(fakes[fake], world, rank, mtp)
    r = memory.Rank.of(_text(fakes[fake]), world, rank, mtp=mtp)
    for latent_mb in (1024, 1):           # 1 MiB caps the prompt chunk's partial rows (the 262K case at real scale)
        monkeypatch.setattr(memory, "LATENT_MB", latent_mb)
        for capacity in (256, 1100, 5000):
            for rows, prefill in ((8, False), (64, True)):
                b = F.Buffers(w, rows, capacity, prefill=prefill)
                got = nbytes(b, skip=("x3", "moe"))
                want = memory.buffer_rows(r, rows, prefill) + memory.partials(r, rows, capacity)
                assert got == want, (latent_mb, capacity, rows, prefill, got, want)
                assert b.lat_s.part_rows == memory.latent_rows(rows, r.heads, memory.chunks_for(capacity), r.kv_lora)


@pytest.mark.parametrize("fake,world,rank,mtp", CASES)
def test_exl3_linear_scratch_mirrors_x3scratch(fakes, fake, world, rank, mtp):
    from tensorfold.families.glm_moe_dsa.cuda.x3 import X3Scratch

    w = _load(fakes[fake], world, rank, mtp)
    users = [(int(u.lin.k), int(u.lin.n)) for u in w.meta["x3"]]
    built = X3Scratch(w.meta["x3"], "cpu", prefill=False)
    assert nbytes(built) == memory.x3_scratch(users, prefill=False)
    # the loaded linears and the ones the estimate derives from the config give the same scratch
    r = memory.Rank.of(_text(fakes[fake]), world, rank, mtp=mtp)
    assert memory.x3_scratch(memory.x3_linears(r), prefill=False) == memory.x3_scratch(users, prefill=False)
    assert set(memory.x3_linears(r)) == set(users)


def test_plan_and_tiles_mirror_the_kernels():
    from tensorfold.cuda.exl3 import experts
    from tensorfold.cuda.exl3.linear import plan
    from tensorfold.families.glm5_next.cuda import qmm

    shapes = [(k, n) for k in (128, 640, 1280, 1536, 4096, 5120, 8192) for n in (128, 256, 384, 576, 640, 1536, 2560,
                                                                             4096, 5120, 6144, 12288, 39296, 78592)]
    for k, n in shapes:
        assert memory.x3_plan(k, n) == plan(k, n), (k, n)
        assert memory._b16_split_k(n, k) == qmm.b16_split_k(n, k), (k, n)
    for D, I in ((4096, 640), (4096, 384), (4096, 256), (128, 256), (128, 1280), (128, 384)):
        assert memory.expert_tiles(D, I, True) == experts.default_config(D, I, True)
        assert memory.expert_tiles(I, D, False) == experts.default_config(I, D, False)


@pytest.mark.parametrize("D,I,count,rows", [(4096, 640, 512, 8), (4096, 384, 512, 8), (4096, 256, 512, 130),
                                            (128, 256, 4, 64)])
@pytest.mark.parametrize("prompt", [False, True])
def test_expert_scratch_mirrors_experts_scratch(D, I, count, rows, prompt):
    from tensorfold.cuda.exl3 import experts

    ex = SimpleNamespace(dims=D, width=I, count=count)
    built = experts.Scratch(ex, rows, 9, device="cpu", prompt=prompt)
    assert nbytes(built) == memory.expert_scratch(D, I, count, rows, 9, prompt=prompt)


def _ling(world: int, rank: int = 0, mtp: bool = True) -> memory.Rank:
    text = json.loads(LING.read_text())
    text["_quantization"] = {"quant_method": "exl3"}
    return memory.Rank.of(text, world, rank, mtp=mtp)


def test_ling_shapes_and_cache():
    r2, r4, r4b = _ling(2), _ling(4), _ling(4, 2)
    assert (r2.kda_layers, r2.mla_layers, r2.mtp) == (63, 9, True)
    assert (r2.heads, r2.lin_heads, r2.dense_width, r2.shared_width, r2.expert_width, r2.vocab_rows) == \
        (32, 32, 5120, 640, 640, 78592)
    assert (r4.heads, r4.dense_width, r4.expert_width, r4.vocab_rows) == (16, 2560, 384, 39296)
    assert (r4b.expert_width, r4b.shared_width) == (256, 256)
    # 10 latent layers x (512 + 64) x 2 bytes, replicated on every rank
    assert memory.cache_per_slot(r2) == memory.cache_per_slot(r4) == 11520
    assert memory.cache_per_slot(_ling(2, mtp=False)) == 10368
    # two fp32 states a KDA layer: 264 MB at TP2, 132 MB at TP4
    assert 2 * 63 * 32 * 128 * 128 * 4 == 264_241_152
    assert memory.state_fixed(r2, 8) - memory.state_fixed(r4, 8) > 130_000_000


def test_ling_partials_at_262k():
    r2, r4 = _ling(2), _ling(4)
    slots = 262144 + 8
    assert memory.chunks_for(slots) == 513
    # decode windows: 8 rows within 1 GiB; the prompt chunk: 31 rows at TP2, 63 at TP4
    assert memory.latent_rows(2048, 32, 513, 512) == 31 and memory.latent_rows(2048, 16, 513, 512) == 63
    assert memory.partials(r2, 8, slots) == 513 * 8 * 32 * 514 * 4
    assert memory.partials(r2, 2048, slots) == 513 * 31 * 32 * 514 * 4
    assert memory.partials(r4, 2048, slots) == 513 * 63 * 16 * 514 * 4
    # 1M slots: even an 8-row window passes 1 GiB at TP2 (two row blocks a verify window)
    assert memory.latent_rows(8, 32, memory.chunks_for(1048576 + 8), 512) == 7


def test_ling_estimate_is_below_glms_and_engine_uses_it():
    from tensorfold.cuda.geometry import mla_geometry

    text = json.loads(LING.read_text())
    text["_quantization"] = {"quant_method": "exl3"}
    ours = memory.geometry(text, 2, 0, mtp=True, rows=8, minimum_slots=2560)
    glm = mla_geometry(text, 2, 8, minimum_slots=2560, latent=True, mtp=True)
    lo, hi = 32768 + 8, 262144 + 8
    slope = (ours.bytes_at(hi) - ours.bytes_at(lo)) / (hi - lo)
    glm_slope = (glm.bytes_at(hi) - glm.bytes_at(lo)) / (hi - lo)
    assert 11520 < slope < 11520 + 4 * 1024           # the cache plus the prompt partials' growth (capped rows)
    assert glm_slope > 2.5 * 11520                    # GLM's: 19 latent layers, an indexer, no rope key
    source = (SRC / "tensorfold" / "families" / "bailing_v3" / "cuda" / "engine.py").read_text()
    assert "memory.geometry(" in source and "mla_geometry(" not in source


@pytest.mark.parametrize("fake,world,rank,mtp", CASES)
def test_shared_scratch_is_priced_byte_for_byte(fakes, fake, world, rank, mtp, monkeypatch):
    from tensorfold.families.bailing_v3.cuda import decode as Dm

    w = _load(fakes[fake], world, rank, mtp)
    r = memory.Rank.of(_text(fakes[fake]), world, rank, mtp=mtp)
    users = [(int(u.lin.k), int(u.lin.n)) for u in w.meta["x3"]]
    monkeypatch.setattr(memory, "SHARED", True)
    for latent_mb in (1024, 1, 0):
        monkeypatch.setattr(memory, "LATENT_MB", latent_mb)
        for capacity in (256, 1100, 5000):
            e = Dm.Engine(w, capacity=capacity, max_rows=8, prefill_rows=64, graphs=False)
            bufs = [e.buf, e.pbuf] + ([e.mbuf] if e.mbuf is not None else [])
            assert len(bufs) == (2 if r.mtp else 1) + 1
            seen: set = set()
            got = sum(nbytes(b, skip=("x3", "moe"), seen=seen) for b in bufs)
            p = memory.parts(r, capacity, rows=8, prefill_rows=64, shared=True)
            assert got == p["decode buffers"] + p["prompt buffers"] + p["chunk partials"], (latent_mb, capacity)
            assert p["chunk partials"] == memory.partials(r, 64, capacity)
            seen = set()
            assert sum(nbytes(b.x3, skip=("prefill",), seen=seen) for b in bufs) == \
                memory.x3_scratch(users, prefill=False)
            for b in bufs:
                assert b.lat_s.part_rows == memory.latent_rows(b.rows, r.heads, memory.chunks_for(capacity),
                                                               r.kv_lora)
                assert b.lat_s.po.data_ptr() == bufs[0].lat_s.po.data_ptr()
                assert b.x3.xh.data_ptr() == bufs[0].x3.xh.data_ptr() and b.x3.z is bufs[0].x3.z


def test_ling_shared_scratch_at_262k(monkeypatch):
    monkeypatch.setattr(memory, "LATENT_MB", 270)              # 8 rows a call at 262K: the widest verify window
    monkeypatch.setattr(memory, "UNPACK_CACHE_MB", 0)
    r, slots = _ling(2), 262144 + 8
    assert memory.latent_rows(8, 32, 513, 512) == memory.latent_rows(2048, 32, 513, 512) == 8
    own = memory.parts(r, slots, shared=False)
    sh = memory.parts(r, slots, shared=True)
    assert own["chunk partials"] == 3 * sh["chunk partials"] == 3 * 513 * 8 * 32 * 514 * 4
    assert own["EXL3 scratch"] - sh["EXL3 scratch"] == 2 * memory.x3_scratch(memory.x3_linears(r), prefill=False)
    assert all(own[k] == sh[k] for k in ("latent cache", "KDA state", "decode buffers", "prompt buffers"))
    saved = sum(own.values()) - sum(sh.values())
    assert 0.65 * 2 ** 30 < saved < 0.67 * 2 ** 30           # 0.50 GiB of partials and 0.16 of EXL3 scratch a rank
    # MTP off: two Buffers (decode, prompt) share; the decode window's partials and X3 core are the ones saved
    r1 = _ling(2, mtp=False)
    assert memory.parts(r1, slots, shared=False)["chunk partials"] == \
        2 * memory.parts(r1, slots, shared=True)["chunk partials"]


def test_embed_span_is_the_heads_and_embed_picks():
    from tensorfold.families.bailing_v3.cuda.weights import embed_span
    from tensorfold.families.glm_moe_dsa.cuda.weights import embed_owner

    assert embed_span(157184, 2, 0, split=False) == (0, 157184)
    assert embed_span(157184, 1, 0, split=True) == (0, 157184)
    assert [embed_span(157184, 2, r, split=True) for r in (0, 1)] == [(0, 78592), (78592, 78592)]
    assert [embed_span(157184, 4, r, split=True)[1] for r in range(4)] == [39296] * 4
    assert _ling(2).vocab_rows == 78592 and _ling(4).vocab_rows == 39296   # the head's slice: the same span
    g = torch.Generator().manual_seed(7)
    for vocab, world in ((157184, 2), (157184, 4), (256, 2), (512, 4), (1000, 3), (130, 2), (300, 4)):
        spans = [embed_span(vocab, world, r, split=True) for r in range(world)]
        assert spans[0][0] == 0 and sum(n for _, n in spans) == vocab
        assert all(spans[i][0] + spans[i][1] == spans[i + 1][0] for i in range(world - 1))
        tokens = torch.randint(0, vocab, (500,), generator=g).tolist() + [0, vocab - 1]
        for t in tokens:                                       # glue.embed_pick's owner holds the row
            e0, en = spans[embed_owner(t, vocab, world)]
            assert e0 <= t < e0 + en, (vocab, world, t)


@pytest.mark.parametrize("fake,world", [("tiny", 2), ("tp4", 4)])
def test_split_embedding_loads_the_span_and_prices_it(fakes, fake, world, monkeypatch):
    from tensorfold.cuda.geometry import split_weights
    from tensorfold.families.bailing_v3.cuda import weights as W
    from tensorfold.families.glm5_next.cuda.split import rule, unit

    for rank in range(world):
        monkeypatch.setattr(W, "EMBED_SPLIT", False)
        whole = _load(fakes[fake], world, rank)
        monkeypatch.setattr(W, "EMBED_SPLIT", True)
        w = _load(fakes[fake], world, rank)
        vocab = whole.cfg.vocab
        e0, en = W.embed_span(vocab, world, rank)
        assert (w.embed_lo, w.embed.shape[0]) == (e0, en) and en < vocab
        assert torch.equal(w.embed.view(torch.int16), whole.embed[e0:e0 + en].view(torch.int16))
        assert whole.nbytes() - w.nbytes() == (vocab - en) * w.cfg.hidden * 2
        base = split_weights(rule, world, unit=unit)
        priced = W.embed_priced(base, vocab, world, rank)
        info = {"dtype": "BF16", "shape": [vocab, w.cfg.hidden]}
        name = "model.word_embeddings.weight"
        assert priced(name, info) == base(name, {**info, "shape": [en, w.cfg.hidden]})
        assert base(name, info)[0] > priced(name, info)[0]
        other = {"dtype": "BF16", "shape": [w.cfg.hidden]}
        assert priced("model.norm.weight", other) == base("model.norm.weight", other)


def test_engine_checks_and_prices_both_switches():
    source = (SRC / "tensorfold" / "families" / "bailing_v3" / "cuda" / "engine.py").read_text()
    assert "int(EMBED_SPLIT), int(memory.SHARED)]" in source       # both ranks must agree
    assert "embed_priced(weights_estimate" in source              # the admission prices the span
    decode = (SRC / "tensorfold" / "families" / "bailing_v3" / "cuda" / "decode.py").read_text()
    assert "Shared(w, max(max_rows, prefill_rows), capacity) if memory.SHARED" in decode
