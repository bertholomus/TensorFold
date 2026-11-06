"""CPU gates for CP4 (DESIGN-CP4 §4): the striped forward against single-rank eager over the full sequence.

Run directly, no pytest, no numpy (the host-test venv):

    $ML4_HOST_TESTS/.venv/bin/python tests/test_ml4_cp4.py

Gates are the ones the note fixed before this code existed: M1 attention merge, M2 selection, M3 layer streams,
M4 logits and greedy tokens (fp32 rows, so the merge alone is under test), M5 real-size attention layers. W = 4 through
the in-process ``EmuComm``; seeds 0-2. The reference is the eager model (one-shot prefill plus decode: the only regime
where eager is complete, see DESIGN-CP4 §1).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
HOST_TESTS = Path(os.environ.get("ML4_HOST_TESTS") or sys.exit("set ML4_HOST_TESTS to the reference suite"))
if str(HOST_TESTS) not in sys.path:
    sys.path.insert(0, str(HOST_TESTS))

import torch                                                     # noqa: E402
from refs import tiny_model as TM                                # noqa: E402
from test_ml4_engine import emu_comms, slice_weights             # noqa: E402
from test_ml4_model import tiny_weights                          # noqa: E402

from tensorfold.families.mistral_large4.cuda import cp4, fp8kv, tp   # noqa: E402
from tensorfold.families.mistral_large4.cuda import model as M   # noqa: E402

torch.set_num_threads(2)
W = 4


SPLIT = os.environ.get("ML4_SPLIT") == "1"                  # a quarter of every expert on each rank, not whole experts


def cos(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float((a @ b / (a.norm() * b.norm() + 1e-30)).item())


def rms(t):
    return float(t.double().pow(2).mean().sqrt())


def run_ranks(fn, world=W):
    """fn(rank, comm) on every rank at once; any failure re-raises here, naming the rank."""

    comms, out, errs = emu_comms(world), [None] * world, []

    def body(r):
        try:
            out[r] = fn(r, comms[r])
        except BaseException as exc:                             # noqa: BLE001
            errs.append((r, exc, traceback.format_exc()))
            print(f"rank {r} failed:\n{errs[-1][2]}", flush=True)
            os._exit(1)                                          # the other ranks are stuck in a collective now

    ts = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(world)]
    [t.start() for t in ts]
    [t.join(300) for t in ts]
    if errs:
        raise RuntimeError(f"rank {errs[0][0]}: {errs[0][2]}")
    assert not any(t.is_alive() for t in ts), "a rank hung"
    return out


def tiny(seed):
    tc = TM.TinyConfig()
    Wd = TM.random_weights(tc, seed)
    w = tiny_weights(tc, Wd)
    w.split = SPLIT
    return tc, w


def eager(w, tokens, steps):
    """Single-rank reference: one-shot prefill, then ``steps`` greedy decode steps. Returns streams and selections."""

    model = M.Ml4Model(w)
    cache = M.new_cache(w.cfg)
    collect, sels = [], []
    logits = model.forward(tokens[None], cache, collect=collect, sel_out=sels)
    last = [logits[-1]]
    fed = []
    dec = []
    for _ in range(steps):
        t = int(last[-1].argmax())
        fed.append(t)
        col = []
        lg = model.forward(torch.tensor([[t]]), cache, collect=col)
        dec.append(col)
        last.append(lg[-1])
    return collect, sels, last, fed, dec


def striped(w, tokens, steps, block, comm, rank, kind="raw", fmt="f32x128", fed=None, width=0):
    """The CP run on one rank: rounds of ``width`` rows (W*block by default), then decode rounds. ``fed`` forces the decode
    tokens."""

    st = cp4.Stripe(W, block)
    total = tokens.numel() + steps
    rows = st.rows_for(total + W * block, 0)
    pool = cp4.KvPool(w.cfg, rows, kind=kind, fmt=fmt)
    fwd = cp4.Ml4Cp(w, comm, pool)
    seq = cp4.Seq(st, 0, rows)
    collect, sels, logits = [], [], {}
    n = tokens.numel()
    width = width or W * block
    for start in range(0, n, width):
        m = min(width, n - start)
        pos = torch.arange(start, start + m)
        own = pos[st.rank(pos) == rank]
        lg = fwd.round(seq, start, m, tokens[own], last=n - 1 if start + m == n else None, collect=collect, sel_out=sels)
        if lg is not None:
            logits[n - 1] = lg
    return fwd, seq, collect, sels, logits


def decode_steps(fwd, seq, rank, comm, n, steps, first, fed):
    """Decode rounds: the owner of position p holds the row. ``fed`` (the eager tokens) keeps the two runs aligned, the
    tokens this run would pick are compared by the caller."""

    st = seq.stripe
    got, cols = {}, []
    for i in range(steps):
        p = n + i
        t = torch.tensor([fed[i]]) if int(st.rank(torch.tensor(p))) == rank else torch.zeros(0, dtype=torch.long)
        col = []
        lg = fwd.round(seq, p, 1, t, last=p, collect=col)
        cols.append(col)
        if lg is not None:
            got[p] = lg
    return got, cols


def check_stream(ref_layers, runs, n_total, what):
    """ref_layers: per layer [1, S, d]. runs: per rank list of (L, pos, x). Every layer's rows, reassembled by position."""

    worst_c, worst_m = 1.0, 0.0
    for L, ref in enumerate(ref_layers):
        ref = ref[0]
        got = torch.zeros_like(ref)
        seen = torch.zeros(ref.shape[0], dtype=torch.bool)
        for col in runs:
            for (l, pos, x) in col:
                if l == L:
                    got[pos - pos_base(ref_layers, pos)] = x
                    seen[pos - pos_base(ref_layers, pos)] = True
        assert seen.all(), f"{what}: layer {L} rows missing"
        c, m = cos(ref, got), float((ref - got).abs().max()) / rms(ref)
        worst_c, worst_m = min(worst_c, c), max(worst_m, m)
    return worst_c, worst_m


def pos_base(_ref, _pos):
    return 0


# -- M1 ---------------------------------------------------------------------------------------------------------------

def test_m1_attention_merge():
    """Softmax over a union from its parts, real dims: 64 heads, v 128, up to 2048 rows split unevenly over 4 ranks.
    The reference is fp64: fp32 single-rank attention itself misses the truth by up to 1.7e-5 x rms on these inputs,
    while the merge stays under 7e-6 (DESIGN-CP4 §4)."""

    g = torch.Generator().manual_seed(11)
    worst_c, worst_m = 1.0, 0.0
    for trial in range(6):
        H, V, n = 64, 128, 2048
        s = torch.randn(H, n, generator=g) * (1 + trial)
        v = torch.randn(n, V, generator=g)
        ref = torch.softmax(s.double(), -1) @ v.double()
        cut = torch.randint(0, W, (n,), generator=g)
        if trial == 5:
            cut[:] = 2                                            # all on one rank: the other three are empty
        o = torch.zeros(W, 1, H, V)
        lse = torch.full((W, 1, H), float("-inf"))
        for k in range(W):
            m = cut == k
            if m.any():
                o[k, 0] = torch.softmax(s[:, m], -1) @ v[m]
                lse[k, 0] = torch.logsumexp(s[:, m], -1)
        got = cp4.merge_partials(o, lse)[0]
        worst_c, worst_m = min(worst_c, cos(ref, got)), max(worst_m, float((ref - got).abs().max()) / rms(ref))
    print(f"  M1 merge: cosine {worst_c:.9f}, max-abs/rms {worst_m:.2e}")
    assert worst_c >= 0.9999999 and worst_m <= 1e-5


def test_stripe_map():
    st = cp4.Stripe(4, 8)
    p = torch.arange(0, 4096)
    r, row = st.rank(p), st.row(p)
    for k in range(4):
        rows = row[r == k]
        assert torch.equal(rows, torch.arange(rows.numel())), "local rows are 0,1,2.. in position order"
        assert torch.equal(st.pos(k, rows), p[r == k])
    for q in [0, 1, 7, 8, 9, 31, 32, 33, 100, 4095]:
        for k in range(4):
            assert int(st.vis(k, torch.tensor(q))) == int(((r == k) & (p <= q)).sum()), (q, k)
    assert st.rows_for(1 << 20, 3) == 1 << 18 and cp4.Stripe(4, 1024).rows_for(1 << 20, 0) == 1 << 18


def test_keys_order():
    s = torch.tensor([-1e30, -3.5, -0.0, 0.0, 1e-30, 2.5, 2.5, 1e30])
    p = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    k = cp4.score_keys(s, p)
    order = k.argsort().tolist()
    # -0 and +0 are one score, so the lower position wins; the tied 2.5s too
    assert order == [0, 1, 3, 2, 4, 6, 5, 7], order
    assert (k > cp4.PAD_KEY).all()


def test_fp8_row_round_trip():
    g = torch.Generator().manual_seed(5)
    lat = torch.randn(64, 512, generator=g) * torch.rand(64, 1, generator=g) * 3
    lat[0] = 0
    lat[1, :128] = 1e-5
    kpe = torch.randn(64, 64, generator=g)
    for name in fp8kv.SCALES:
        fmt = fp8kv.row_format(name)
        rows = fp8kv.encode(lat, kpe, fmt)
        assert rows.shape == (64, 656) and fmt.row_bytes == 656 and fmt.nope().row_bytes == 528, name
        got, gk = fp8kv.decode(rows, fmt)
        assert torch.isfinite(got).all()
        assert torch.equal(gk, kpe.half().float())
        assert torch.equal(got[0], lat[0])
        rel = (got - lat).norm() / lat.norm()
        print(f"  fp8 {name}: row error {float(rel):.4f}")
        assert rel < 0.04, name
        lat2, none = fp8kv.decode(fp8kv.encode(lat, None, fmt.nope()), fmt.nope())
        assert none is None and torch.equal(lat2, got)


# -- M2-M4 ------------------------------------------------------------------------------------------------------------

def compare_selection(sels_ref, sel_ranks, w, st, n):
    """Eager's selection against the union of the ranks' lists, every (full layer, query) of the prefill."""

    cfg = w.cfg
    swaps = checked = 0
    ref = {L: s for L, s in sels_ref if s is not None}
    for i, (L, qpos, _, _) in enumerate(sel_ranks[0]):
        if L not in ref:
            continue
        for qi in range(qpos.numel()):
            p = int(qpos[qi])
            want = set(int(v) for v in ref[L][0, p].tolist() if v >= 0)
            got = set()
            for k in range(W):
                _, _, rows, cnt = sel_ranks[k][i]
                got |= set(int(st.pos(k, r)) for r in rows[qi, :int(cnt[qi])].tolist())
            checked += 1
            if got != want:
                swaps += 1
    return checked, swaps


def m2_m4(seed, block, n=None, steps=16, width=0):
    tc, w = tiny(seed)
    cfg = w.cfg
    n = n or 9 * cfg.idx_topk * W
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, cfg.vocab, (n,), generator=g)
    ref_col, ref_sel, ref_last, fed, dec = eager(w, tokens, steps)
    stack = [c for c in ref_col]

    def work(rank, comm):
        fwd, seq, col, sels, lg = striped(w, tokens, steps, block, comm, rank, width=width)
        got, cols = decode_steps(fwd, seq, rank, comm, n, steps, None, fed)
        return col, sels, lg, got, cols

    out = run_ranks(work)
    st = cp4.Stripe(W, block)
    checked, swaps = compare_selection(ref_sel, [o[1] for o in out], w, st, n)
    c, m = check_stream(stack, [o[0] for o in out], n, "prefill")
    last = next(o[2][n - 1] for o in out if n - 1 in o[2])
    lc, lm = cos(ref_last[0], last), float((ref_last[0] - last).abs().max())
    worst_lc, worst_lm, same = lc, lm, int(ref_last[0].argmax()) == int(last.argmax())
    for i in range(steps):
        p = n + i
        lg = next(o[3][p] for o in out if p in o[3])
        worst_lc = min(worst_lc, cos(ref_last[i + 1], lg))
        worst_lm = max(worst_lm, float((ref_last[i + 1] - lg).abs().max()))
        same = same and int(ref_last[i + 1].argmax()) == int(lg.argmax())
        for L, col in enumerate(dec[i]):
            for o in out:
                for (l, pos, x) in o[4][i]:
                    if l == L and pos.numel():
                        c = min(c, cos(col[0], x))
                        m = max(m, float((col[0] - x).abs().max()) / rms(col[0]))
    print(f"  M2-M4 seed {seed} block {block} n {n}: selection {checked} checked, {swaps} differ; layer cos {c:.8f} "
          f"max/rms {m:.2e}; logits cos {worst_lc:.8f} max {worst_lm:.2e}; greedy same {same}")
    assert swaps == 0, f"{swaps} selections differ"
    assert c >= 0.999999 and m <= 1e-4
    assert worst_lc >= 0.999999 and worst_lm <= 1e-3 and same


def test_m2_m4_block2():
    for seed in (0, 1, 2):
        m2_m4(seed, 2)


def test_m2_m4_round_of_many_blocks():
    m2_m4(1, 2, width=24)


def test_m2_m4_block4():
    m2_m4(1, 4)


def test_m2_m4_block8_ragged():
    m2_m4(2, 8, n=301)                                            # the last round leaves ranks short or empty


def test_m2_m4_short_context():
    m2_m4(0, 2, n=5, steps=3)                                     # every row inside the budget: all visible rows


def test_decode_crosses_budget():
    """A short prompt whose decode crosses the budget. Eager cannot be the running reference here (its key plane starts
    when the budget binds, DESIGN-CP4 §1), so the reference is its one-shot prefill over the whole fed sequence."""

    tc, w = tiny(4)
    cfg = w.cfg
    n, steps, block = 5, 24, 2
    g = torch.Generator().manual_seed(4)
    full = torch.randint(0, cfg.vocab, (n + steps,), generator=g)
    logits = M.Ml4Model(w).forward(full[None], M.new_cache(cfg))

    def work(rank, comm):
        fwd, seq, _, _, lg = striped(w, full[:n], steps, block, comm, rank)
        got, _ = decode_steps(fwd, seq, rank, comm, n, steps - 1, None, full[n:].tolist())
        got.update(lg)
        return got

    out = run_ranks(work)
    worst = 1.0
    for p in range(n - 1, n + steps - 1):
        lg = next(o[p] for o in out if p in o)
        worst = min(worst, cos(logits[p], lg))
    print(f"  decode across the budget: logits cos {worst:.8f}")
    assert worst >= 0.999999


def test_dp_group_of_one():
    """The DP shape: world-1 stripe, the same forward, chunked prefill. The eager one-shot is still the reference."""

    tc, w = tiny(3)
    cfg = w.cfg
    n, steps = 70, 8
    g = torch.Generator().manual_seed(3)
    tokens = torch.randint(0, cfg.vocab, (n,), generator=g)
    ref_col, ref_sel, ref_last, fed, dec = eager(w, tokens, steps)
    st = cp4.Stripe(1, 16)
    rows = n + steps
    fwd = cp4.Ml4Cp(w, None, cp4.KvPool(cfg, rows, kind="raw"))
    seq = cp4.Seq(st, 0, rows)
    lg = None
    for start in range(0, n, 16):
        m = min(16, n - start)
        lg = fwd.round(seq, start, m, tokens[start:start + m], last=n - 1 if start + m == n else None)
    c = cos(ref_last[0], lg)
    for i in range(steps):
        lg = fwd.round(seq, n + i, 1, torch.tensor([fed[i]]), last=n + i)
        c = min(c, cos(ref_last[i + 1], lg))
    print(f"  DP group of one: logits cos {c:.8f}")
    assert c >= 0.999999


def test_tp_step_against_rounds():
    """tp.py's step against a decode round on the same rows: logits within fp32 rounding, through and past the budget."""

    for seed, block, n in ((0, 2, 24), (1, 4, 40), (2, 8, 3)):
        tc, w = tiny(seed)
        tokens = torch.randint(0, tc.vocab, (n,), generator=torch.Generator().manual_seed(seed))
        st, steps = cp4.Stripe(W, block), 10

        def body(rank, comm):
            wr = slice_weights(w, rank, W)
            fwd, seq, _, _, _ = striped(wr, tokens, steps, block, comm, rank)
            pool = cp4.KvPool(wr.cfg, seq.rows + 1, kind="raw")
            for mine, there in ((fwd.pool.lat, pool.lat), (fwd.pool.kpe, pool.kpe), (fwd.pool.keys, pool.keys)):
                for L in mine:
                    there[L][:seq.rows] = mine[L]
            lane = tp.Ml4Tp(cp4.Ml4Cp(wr, comm, pool))
            lane.use(st)
            step_seq, worst, gen = cp4.Seq(st, 0, seq.rows), 0.0, torch.Generator().manual_seed(seed + 5)
            for i in range(steps):
                p = n + i
                tok = torch.randint(0, tc.vocab, (1,), generator=gen)
                own = int(st.rank(torch.tensor(p))) == rank
                lg = fwd.round(seq, p, 1, tok if own else tok[:0], last=p)
                part = lane.step(tok, tp.make_rows([step_seq], [p], rank, "cpu", k=wr.cfg.idx_topk, scratch=seq.rows))
                best = lane.best(part)
                every = torch.empty((W, part.numel()))
                comm.all_gather(part.reshape(-1).contiguous(), every.view(-1))
                if own:
                    worst = max(worst, float((lg - every.view(-1)).abs().max()))
                    assert int(lg.argmax()) == int(best[0]), f"position {p}"
            return worst

        worst = max(run_ranks(body))
        assert worst < 2e-5, f"seed {seed}: logits differ by {worst}"


def kl(p_logits, q_logits):
    a = torch.log_softmax(p_logits.double(), -1)
    b = torch.log_softmax(q_logits.double(), -1)
    return float((a.exp() * (a - b)).sum())


def test_gate2_fp8_against_fp16():
    """Gate 2 on the tiny model, readers live: the same stream on fp16 rows and on fp8 rows (DESIGN-CP4 §6, §11). The
    fed tokens are shared, so every step compares the same context. Gates: KL <= 1e-3 nats on every step, the greedy
    token the same on at least 95 % of steps. Cosine is printed, not gated: this random model's logits sit near zero, so
    a small absolute change moves it a lot while the distribution stays put."""

    worst_c, worst_kl, same, steps_all = 1.0, 0.0, 0, 0
    for seed in (0, 1, 2):
        tc, w = tiny(seed)
        cfg = w.cfg
        assert any(s != i for i, s in enumerate(cfg.kv_source_layers)), "no reader in this config"
        n, steps = 96, 24
        tokens = torch.randint(0, cfg.vocab, (n,), generator=torch.Generator().manual_seed(seed))
        fed = torch.randint(0, cfg.vocab, (steps,), generator=torch.Generator().manual_seed(seed + 9)).tolist()

        def work(r, c, kind):
            fwd, seq, _, _, logits = striped(w, tokens, steps, 4, c, r, kind=kind)
            logits.update(decode_steps(fwd, seq, r, c, n, steps, None, fed)[0])
            return logits

        runs = {}
        for kind in ("fp16", "fp8"):
            runs[kind] = {}
            for o in run_ranks(lambda r, c, kind=kind: work(r, c, kind)):
                runs[kind].update(o)
        for p in sorted(runs["fp16"]):
            a, b = runs["fp16"][p], runs["fp8"][p]
            worst_c, worst_kl = min(worst_c, cos(a, b)), max(worst_kl, kl(a, b))
            same += int(a.argmax()) == int(b.argmax())
            steps_all += 1
    print(f"  gate 2 tiny: {steps_all} steps, logits cosine {worst_c:.6f}, worst KL {worst_kl:.2e}, "
          f"greedy same {same}/{steps_all}")
    assert worst_kl <= 1e-3 and same >= 0.95 * steps_all


def engine_lane(tp_on: bool):
    """The served path: four engines over the fake link. A long stream alone, a short one alone, then both at once, each
    with eager's tokens; the pools and ledgers are whole again at the end. Without the step the short one is a DP stream."""

    import test_ml4_engine as TE

    from tensorfold.families.mistral_large4.cuda import dp4 as DP
    from tensorfold.families.mistral_large4.cuda import engine as EN

    def quiet(_s):
        return None

    tc, w = tiny(5)
    g = torch.Generator().manual_seed(5)
    long_p = torch.randint(0, w.cfg.vocab, (41,), generator=g).tolist()
    short_p = torch.randint(0, w.cfg.vocab, (13,), generator=g).tolist()
    solo = EN.Ml4Engine(None, rank=0, world=1, device="cpu", weights=w, pool_tokens=4096, log=quiet)
    want_l, want_s = [], []
    solo.generate(long_p, 6, on_tokens=TE.feed(want_l))
    solo.generate(short_p, 4, on_tokens=TE.feed(want_s))

    old, cp4.BLOCK = cp4.BLOCK, 4
    was, EN.TP = EN.TP, tp_on
    slice_was, EN.SLICE = EN.SLICE, 2                         # a short prompt overtakes a long one inside its round
    comms, link = emu_comms(W), DP.EmuLink(W, 0)
    try:
        engines = [EN.Ml4Engine(None, rank=r, world=W, device="cpu", weights=TE.slice_weights(w, r, W), comm=comms[r],
                                link=link.fork(r), pool_tokens=256, log=quiet, fast=True) for r in range(W)]
    finally:
        cp4.BLOCK, EN.TP = old, was
    e = engines[0]
    e.dp_limit = 20
    ts = [threading.Thread(target=engines[r].follow, daemon=True) for r in range(1, W)]
    [t.start() for t in ts]
    old, cp4.BLOCK = cp4.BLOCK, 4
    served, e.concurrent = e.concurrent, False                # rounds by hand first
    try:
        got = []
        e.generate(long_p, 6, on_tokens=TE.feed(got))
        assert got == want_l, f"stripe alone: {got} != {want_l}"
        got = []
        e.generate(short_p, 4, on_tokens=TE.feed(got))
        assert got == want_s, f"DP stream alone: {got} != {want_s}"
        a, b = [], []
        s1 = e.admit(long_p, 6, on_tokens=TE.feed(a))
        if tp_on:
            assert e.run_round() and s1.left, "the long prompt's round stops after its first slice"
        s2 = e.admit(short_p, 4, on_tokens=TE.feed(b))
        assert s1.cp and s2.cp == tp_on
        if tp_on:
            while not s2.out:
                assert e.run_round(), "a live stream got no round"
            assert s1.left, "the short prompt answers while the long one is in its round"
        while not (s1.done and s2.done):
            assert e.run_round(), "a live stream got no round"
        assert a == want_l and b == want_s, f"both at once: {a} != {want_l} or {b} != {want_s}"
        stop = []
        e.generate(long_p, 6, on_tokens=lambda t: stop.extend(t) or True)
        assert len(stop) == 1, "True from on_tokens must stop the stream"
        if served:                                            # as served: a thread a request, one round loop
            e.concurrent = True
            a, b = [], []
            reqs = [threading.Thread(target=e.generate, args=(p, n), kwargs={"on_tokens": TE.feed(x)}, daemon=True)
                    for p, n, x in ((long_p, 6, a), (short_p, 4, b))]
            [t.start() for t in reqs]
            [t.join(120) for t in reqs]
            assert a == want_l and b == want_s, f"served together: {a} != {want_l} or {b} != {want_s}"
            look, EN.LOOK = EN.LOOK, 0.0
            stop = []
            e.generate(long_p, 6, on_tokens=lambda t: stop.extend(t) or True)
            EN.LOOK = look
            assert len(stop) <= 1, "a client that left gets no more tokens"
            for _ in range(2000):
                if not e._ended:
                    break
                time.sleep(0.01)
            assert not e._ended and not e.flights and e._failed is None, "a stopped stream left something behind"
    finally:
        cp4.BLOCK, EN.SLICE = old, slice_was
        e.link.send(["stop"])
        [t.join(60) for t in ts]
    assert not any(t.is_alive() for t in ts), "a follower hung"
    assert all(x.runs == [[0, 256]] for x in e.extents), f"extents leaked: {[x.runs for x in e.extents]}"
    assert e.ledgers.free == [256] * W, f"ledgers leaked: {e.ledgers.free}"
    print(f"  engine lane, {'all striped' if tp_on else 'stripe and DP'}: {want_l}, {want_s}")


def test_engine_lane():
    engine_lane(False)


def test_engine_lane_tp():
    engine_lane(True)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"ok   {t.__name__}")
        except BaseException:                                    # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    return bad


if __name__ == "__main__":
    sys.exit(main())
