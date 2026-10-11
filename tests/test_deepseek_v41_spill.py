"""DeepSeek-V4.1 spilled kept prompts (TF_DS_SPILL_GIB): rows and snapshots written by one run restore byte for byte
in the next, through the real MultiDecoder admission and finish paths, on CPU tensors (no GPU, kernels or checkpoint).
"""

import json
import os
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

RATIOS = {0: 4, 1: 128}          # two kv-source layers' compress ratios
RAW, RING, D, LAYERS, CHUNK = 4, 8, 16, 4, 2048


def _val(prefix, mul: int) -> int:
    return int((int(np.asarray(prefix, dtype=np.int64).sum()) * mul + len(prefix)) % 251)


def _pool(cap=131072, slots=2):
    comp = {i: (torch.zeros((cap // r + 2, D // 2), dtype=torch.uint8),
                torch.zeros((cap // r + 2, D // 16), dtype=torch.uint8)) for i, r in RATIOS.items()}
    index_k = {1: (torch.zeros((cap // 128 + 2, 8), dtype=torch.uint8),
                   torch.full((cap // 128 + 2, 1), 127, dtype=torch.uint8))}
    return SimpleNamespace(slots=slots, cap=cap, ring_size=RING, comp=comp, index_k=index_k,
                           ring=[torch.zeros((slots * RING, D), dtype=torch.bfloat16) for _ in range(LAYERS)],
                           comp_raw={1: (torch.zeros((slots * RAW, D)), torch.zeros((slots * RAW, D)))},
                           tokens=torch.zeros((cap,), dtype=torch.int64))


def _fill(pool, base: int, slot: int, prompt, p0: int, p1: int) -> None:
    """A fake prefill of positions [p0, p1): each compressed row a function of the ids to its group's end (a partial
    last group recomputed when the next chunk completes it), the slot's rings and compressor inputs of the ids so far."""

    for i, r in RATIOS.items():
        for q in range(p0 // r, -(-p1 // r)):
            v = _val(prompt[:min((q + 1) * r, p1)], 31 + i)
            for t in pool.comp[i]:
                t[base // r + q] = v
            if i in pool.index_k:
                for t in pool.index_k[i]:
                    t[base // r + q] = (v * 7) % 251
    pool.tokens[base + p0:base + p1] = torch.tensor(prompt[p0:p1], dtype=torch.int64)
    v = _val(prompt[:p1], 17)
    for layer, t in enumerate(pool.ring):
        t[slot * RING:(slot + 1) * RING] = float(v + layer)
    for pair in pool.comp_raw.values():
        for h, t in enumerate(pair):
            t[slot * RAW:(slot + 1) * RAW] = float(v + h)


@pytest.fixture
def mods(monkeypatch, tmp_path):
    """multi.py and spill.py with the CUDA model and engine modules stubbed (PREFILL_CHUNK, REPLAY, RAW)."""

    monkeypatch.setenv("TF_DS_KEEP", "1")
    model = ModuleType("tensorfold.families.deepseek_v41.cuda.model")
    model.RAW = RAW
    engine = ModuleType("tensorfold.families.deepseek_v41.cuda.engine")
    engine.PREFILL_CHUNK, engine.REPLAY = CHUNK, False
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    for name in ("tensorfold.families.deepseek_v41.cuda.multi", "tensorfold.families.deepseek_v41.cuda.spill"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import tensorfold.families.deepseek_v41.cuda.multi as multi
    import tensorfold.families.deepseek_v41.cuda.spill as spill

    return multi, spill, tmp_path / "spill"


def _decoder(multi, spill, root, *, budget=1 << 30, nonce="run1", min_top=4096):
    pool, calls = _pool(), []

    def pool_view(p, slot, base, size):
        return SimpleNamespace(base=base, size=size, slot=slot, length=0, host=multi.HostIds())

    def prefill_steps(sc, dc, prompt, image=None, start=0, snap=None):
        sc.length = start
        sc.host.truncate(start)
        first = min(len(prompt), (start // CHUNK + 1) * CHUNK)
        ends = [first] + list(range(first + CHUNK, len(prompt), CHUNK))
        if ends[-1] != len(prompt):
            ends.append(len(prompt))
        calls.append(start)
        for s, e in zip([start] + ends[:-1], ends):
            _fill(pool, sc.base, sc.slot, prompt, s, e)
            sc.host.set(s, prompt[s:e])
            if snap is not None and e % CHUNK == 0:
                snap(e)
            if e < len(prompt):
                yield e
        return "logits"

    e = SimpleNamespace(rank=0, world=1, drafter=None, eos=(1,), replay_mode=False, prefill_steps=prefill_steps,
                        _sample=lambda last, pos, sampling: [7])
    md = object.__new__(multi.MultiDecoder)
    md.e, md.pool, md.dpool, md.cap = e, pool, None, pool.cap
    md.m = SimpleNamespace(cfg=SimpleNamespace(window=128, compress_ratios=[4, 128, 1, 1], n_layers=LAYERS),
                           pool_view=pool_view, comm=None)
    md.slots = [multi.Slot(i, None) for i in range(pool.slots)]
    md.extents, md.free, md.max_rows, md.eos = multi.Extents(pool.cap), list(range(pool.slots)), 6, (1,)
    md.streams, md.filling, md.chunk_s, md.since_fill, md.next_id = {}, [], 0.0, 0.0, 0
    md.link = md.watch = md.round_end = None
    md.kept, md.next_kept, md.ticks, md.keep_stats = {}, 0, 0, {}
    md.spill = spill.Spill(root, engine="testengine", rank=0, pool=pool, ratios=RATIOS, budget=budget,
                           max_age=3600.0, min_top=min_top, device="cpu")
    md.spill_nonce = nonce
    return md, calls


def _serve(multi, md, prompt):
    """Admit, fill and finish one request; returns its stream."""

    s = multi.Stream(list(prompt), 1, None, draft=False, stop_eos=False)
    s.emit = lambda new: None
    md.admit(s)
    while s in md.filling:
        md._fill(s)
    md.finish([s])                                                   # rank 0's finish: covered kept prompts drop
    return s


def _rows(pool, base: int, n: int) -> list:
    out = [t[base // r:(base + n) // r].clone() for i, r in RATIOS.items() for t in pool.comp[i]]
    out += [t[base // 128:(base + n) // 128].clone() for t in pool.index_k[1]]
    return out + [pool.tokens[base:base + n].clone()]


def _prompt(n: int, seed: int) -> list[int]:
    return [int(x) for x in np.random.default_rng(seed).integers(1000, 60000, n)]


def test_a_restart_restores_a_spilled_prompt_byte_for_byte(mods):
    multi, spill, root = mods
    a = _prompt(20000, 1)
    md1, _ = _decoder(multi, spill, root)
    _serve(multi, md1, a)
    (k,) = md1.kept.values()
    assert k.disk is not None and k.top == 18432                     # its last chunk boundary
    want = _rows(md1.pool, k.base, k.top)

    md2, calls = _decoder(multi, spill, root, nonce="run2")          # a new run: an empty window, the same disk
    b = a + _prompt(3000, 2)
    s = multi.Stream(list(b), 1, None, draft=False, stop_eos=False)
    s.emit = lambda new: None
    md2.admit(s)
    assert s.cached == 18432 and md2.keep_stats == {"disk": 1}
    got = _rows(md2.pool, s.base, 18432)
    assert all(torch.equal(x, y) for x, y in zip(want, got))
    assert np.array_equal(s.st.sc.host.view()[:18432], np.asarray(a[:18432], dtype=np.int32))
    for bnd, snap in k.snaps.items():
        assert torch.equal(snap[0], s.snaps[bnd][0]) and torch.equal(snap[1], s.snaps[bnd][1])

    while s in md2.filling:                                          # its own rows from the boundary on
        md2._fill(s)
    assert calls == [18432]                                          # (the prompt's chunks start at the boundary)
    md2.finish([s])
    (k2,) = md2.kept.values()
    assert k2.top == 22528
    segs = k2.disk["segments"]
    assert segs[0]["file"] == k.disk["segments"][0]["file"] and segs[0]["to"] == 18432
    assert [g["from"] for g in segs] == [0, 18432]                   # only the new rows were written

    md3, calls3 = _decoder(multi, spill, root, nonce="run3")         # and again, across both runs' segments
    c = b + _prompt(500, 3)
    s3 = _serve(multi, md3, c)
    assert s3.cached == 22528 and calls3 == [22528]
    want2 = _rows(md2.pool, k2.base, 22528)
    md4, _ = _decoder(multi, spill, root, nonce="run4")
    s4 = multi.Stream(list(c), 1, None, draft=False, stop_eos=False)
    s4.emit = lambda new: None
    md4.admit(s4)
    assert s4.cached == 22528
    assert all(torch.equal(x, y) for x, y in zip(want2, _rows(md4.pool, s4.base, 22528)))


def test_a_fresh_fill_and_a_restored_one_hold_the_same_rows(mods):
    multi, spill, root = mods
    a, more = _prompt(20000, 4), _prompt(4000, 5)
    md1, _ = _decoder(multi, spill, root)
    _serve(multi, md1, a)
    md2, _ = _decoder(multi, spill, root, nonce="run2")
    s = _serve(multi, md2, a + more)                                 # restored at 18432, then filled
    fresh, _ = _decoder(multi, spill, root / "elsewhere", nonce="run3")
    f = _serve(multi, fresh, a + more)                               # filled from 0 (a different disk)
    assert s.cached == 18432 and f.cached == 0
    n = len(a + more) // CHUNK * CHUNK
    assert all(torch.equal(x, y) for x, y in zip(_rows(md2.pool, s.base, n), _rows(fresh.pool, f.base, n)))


def test_a_continued_prompt_in_the_window_writes_only_its_new_rows_and_drops_the_covered_manifest(mods):
    multi, spill, root = mods
    a = _prompt(20000, 6)
    md, _ = _decoder(multi, spill, root)
    _serve(multi, md, a)
    (k,) = md.kept.values()
    first = k.disk["name"]
    s = _serve(multi, md, a + _prompt(5000, 7))                      # continues it in the window ("here")
    assert s.cached == 18432
    (k2,) = md.kept.values()                                         # the first one is covered: dropped
    assert [g["from"] for g in k2.disk["segments"]] == [0, 18432]
    assert first not in md.spill.manifests and not md.spill._entry(first).exists()
    assert md.spill._path(k.disk["segments"][0]["file"]).exists()  # its rows stay: the new manifest names them


def test_segments_split_at_seg_max_and_restore_across_them(mods, monkeypatch):
    multi, spill, root = mods
    monkeypatch.setattr(spill, "SEG_MAX", 8192)
    a = _prompt(30000, 8)
    md1, _ = _decoder(multi, spill, root)
    _serve(multi, md1, a)
    (k,) = md1.kept.values()
    assert [g["from"] for g in k.disk["segments"]] == [0, 8192, 16384, 24576]
    md2, _ = _decoder(multi, spill, root, nonce="run2")
    s = _serve(multi, md2, a + [11, 12, 13])
    assert s.cached == k.top
    assert all(torch.equal(x, y) for x, y in zip(_rows(md1.pool, k.base, k.top), _rows(md2.pool, s.base, k.top)))


def test_a_missing_file_fills_fresh_instead(mods, capsys):
    multi, spill, root = mods
    a = _prompt(20000, 9)
    md1, _ = _decoder(multi, spill, root)
    _serve(multi, md1, a)
    (k,) = md1.kept.values()
    md1.spill._path(k.disk["segments"][0]["file"]).unlink()
    md2, calls = _decoder(multi, spill, root, nonce="run2")
    s = _serve(multi, md2, a + [5, 6, 7])
    assert s.cached == 0 and calls == [0]
    assert "not on every rank" in capsys.readouterr().out


def test_short_prompts_stay_in_the_window_only(mods):
    multi, spill, root = mods
    md, _ = _decoder(multi, spill, root)
    _serve(multi, md, _prompt(3000, 10))                             # its top (2048) is under TF_DS_SPILL_MIN
    (k,) = md.kept.values()
    assert k.disk is None and not md.spill.manifests


def test_the_recycler_keeps_disk_use_under_the_budget_least_recently_used_first(mods):
    multi, spill, root = mods
    md, _ = _decoder(multi, spill, root, budget=1 << 30)
    for seed in range(3):
        _serve(multi, md, _prompt(9000, 20 + seed))                  # three unrelated conversations
    names = sorted(md.spill.manifests, key=md.spill._mtime)
    assert len(names) == 3
    for age, name in zip((300, 200, 100), names):                     # the oldest was used longest ago
        t = time.time() - age
        os.utime(md.spill._entry(name), (t, t))
    one = md.spill.stats()["bytes"] / 3
    md.spill.budget = int(2.5 * one)
    md.spill.recycle()
    assert set(md.spill.manifests) == set(names[1:])
    md.spill.budget = 1 << 30
    md.spill.max_age = 150
    md.spill.recycle()                                                # unused past the age limit
    assert set(md.spill.manifests) == {names[2]}
    left = {p.name for p in md.spill.dir.iterdir()}
    assert left == md.spill._referenced() | {f"entry-{names[2]}.json"}   # no file nothing names


def test_pinned_prompts_go_last_and_stray_files_go_at_start(mods):
    multi, spill, root = mods
    md, _ = _decoder(multi, spill, root)
    for seed in range(2):
        _serve(multi, md, _prompt(9000, 30 + seed))
    old, new = sorted(md.spill.manifests, key=md.spill._mtime)
    t = time.time() - 100
    os.utime(md.spill._entry(old), (t, t))
    md.spill.budget = int(1.5 * md.spill.stats()["bytes"] / 2)
    md.spill.recycle(pinned={old})
    assert set(md.spill.manifests) == {old}                          # the window's prompt outlived a newer one
    (md.spill.dir / "seg-x-0.st.partial").write_bytes(b"half")
    (md.spill.dir / "snap-orphan-2048.st").write_bytes(b"nothing names me")
    (root / "v1-otherengine").mkdir()
    md2, _ = _decoder(multi, spill, root, nonce="run2")
    md2.spill.recycle()
    assert not (md.spill.dir / "seg-x-0.st.partial").exists()
    assert not (md.spill.dir / "snap-orphan-2048.st").exists()
    assert not (root / "v1-otherengine").exists() and old in md2.spill.manifests


def test_matching_takes_the_furthest_shared_boundary_past_the_window(mods):
    multi, spill, root = mods
    md, _ = _decoder(multi, spill, root)
    md.e.replay_mode = True                                           # replay prefill: boundaries a window below
    a = _prompt(20000, 40)
    _serve(multi, md, a)
    sp = md.spill
    keys = np.asarray(a + [1, 2, 3], dtype=np.int64)
    name = next(iter(sp.manifests))
    assert sp.best(keys, len(keys), window=128, replay=True) == (name, 18432)
    assert sp.best(keys, len(keys), window=128, replay=True, above=18432) is None
    assert sp.best(keys, len(keys), window=128, replay=False) is None             # another prompt mode's rows
    assert sp.best(keys[:17000], 17000, window=128, replay=True) == (name, 16384)
    assert sp.best(keys[:16600], 16600, window=128, replay=True) == (name, 16384)
    assert sp.best(keys[:16400], 16400, window=128, replay=True) == (name, 8192)   # 16384 is inside the window
    other = keys.copy()
    other[5000] += 1                                                  # differs before 8192: the 4096 boundary
    assert sp.best(other, len(other), window=128, replay=True) == (name, 4096)


def test_a_tampered_manifest_naming_files_elsewhere_is_dropped_untouched(mods, tmp_path):
    multi, spill, root = mods
    md, _ = _decoder(multi, spill, root)
    _serve(multi, md, _prompt(9000, 50))
    (k,) = md.kept.values()
    outside = tmp_path / "keep-me.st"
    outside.write_bytes(b"not the spill's")
    entry = md.spill._entry(k.disk["name"])
    m = json.loads(entry.read_text())
    m["segments"][0]["file"] = "../../keep-me.st"
    entry.write_text(json.dumps(m))
    md2, calls = _decoder(multi, spill, root, nonce="run2")
    assert not md2.spill.manifests and not entry.exists()             # dropped at load
    md2.spill.recycle()
    assert outside.read_bytes() == b"not the spill's"
    s = _serve(multi, md2, _prompt(9000, 50) + [1, 2])
    assert s.cached == 0 and calls == [0]


def test_a_continuation_at_the_spilled_top_supersedes_its_manifest_but_keeps_its_files(mods):
    multi, spill, root = mods
    a = _prompt(20000, 60)
    md1, _ = _decoder(multi, spill, root)
    _serve(multi, md1, a)
    (k,) = md1.kept.values()
    first, seg = k.disk["name"], k.disk["segments"][0]["file"]
    md2, _ = _decoder(multi, spill, root, nonce="run2")
    s = _serve(multi, md2, a + _prompt(3000, 61))                   # continued at the spilled prompt's top
    assert s.cached == k.top
    (k2,) = md2.kept.values()
    assert first not in md2.spill.manifests and not md2.spill._entry(first).exists()
    assert set(md2.spill.manifests) == {k2.disk["name"]}
    assert k2.disk["segments"][0]["file"] == seg and md2.spill._path(seg).exists()


def test_a_continuation_below_the_spilled_top_keeps_its_manifest(mods):
    multi, spill, root = mods
    a = _prompt(20000, 62)
    md1, _ = _decoder(multi, spill, root)
    _serve(multi, md1, a)
    (k,) = md1.kept.values()
    md2, _ = _decoder(multi, spill, root, nonce="run2")
    s = _serve(multi, md2, a[:17000] + _prompt(3000, 63))          # a branch: it shares only the first 17,000 ids
    assert s.cached == 16384 < k.top
    (k2,) = md2.kept.values()
    assert set(md2.spill.manifests) == {k.disk["name"], k2.disk["name"]}   # the first still holds 16384..18432
