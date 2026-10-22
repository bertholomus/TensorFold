"""response_format on DeepSeek-V4.1's engines (fake model and drafter, toy vocabulary, CPU): every short sequence of
admissions and rounds under ``--parallel``, on one rank and on two, gives each grammar's serial reply; drafts the
grammar rules out never reach a forward; a failed grammar ends its own request only. The serial engine's drafted and
serial replies equal the same reply, its follower walks the same grammar, and a failed grammar leaves it serving."""

from __future__ import annotations

import itertools
import json
import queue
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
xgr = pytest.importorskip("xgrammar")

pytestmark = pytest.mark.torch

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine import grammar  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import dspark, multi, rounds  # noqa: E402
from tensorfold.families.deepseek_v41.cuda.engine import DsEngine  # noqa: E402
from tensorfold.server.errors import RequestError  # noqa: E402

STOP = 0
VOCAB = ["", '{"line": "', '{"line":"', '"}', '"', "}", " ", "{", *"abcdefgh", "ab", "cd ", 'e"}', ":", ",", "line",
         "\n", *"0123456789"]
V = len(VOCAB)
LINE = {"type": "object", "properties": {"line": {"type": "string", "maxLength": 8}}, "required": ["line"],
        "additionalProperties": False}
SHORT = dict(LINE, properties={"line": {"type": "string", "maxLength": 7}})     # the grammar that fails (FAIL_AT)
TABLE = torch.randn(97, V, generator=torch.Generator().manual_seed(17))


def logits_at(ctx) -> torch.Tensor:
    """The fake model's logits after ``ctx``: a fixed row picked by its last two tokens and its length."""

    ctx = [int(t) for t in ctx]
    return TABLE[(ctx[-1] * 31 + (ctx[-2] if len(ctx) > 1 else 0) * 7 + len(ctx)) % 97]


def frozen(t: torch.Tensor) -> torch.Tensor:
    """``t`` as the engine returns logits: an inference tensor, which refuses an in-place write outside inference mode."""

    with torch.inference_mode():
        return t.clone()


class Failing(grammar.Constraint):
    """A grammar whose ``FAIL_AT``-th window or advance raises, as xgrammar failing mid-reply would."""

    def __init__(self, *args, fail_at: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls, self.fail_at = 0, fail_at

    def _tick(self) -> None:
        self.calls += 1
        if self.calls == self.fail_at:
            raise grammar.GrammarError("the reply's grammar failed: injected")

    def window(self, tokens, parents):
        self._tick()
        return super().window(tokens, parents)

    def advance(self, tokens) -> None:
        self._tick()
        super().advance(tokens)


class Grammars(grammar.Grammars):
    """The toy vocabulary's grammars; SHORT's constraints fail at ``fail_at`` (``follow`` too); with ``refuse`` SHORT
    does not compile (as a rank with another xgrammar might refuse a schema rank 0 took)."""

    fail_at = 0
    refuse = False

    def compile(self, spec):
        if self.refuse and spec.text == json.dumps(SHORT):
            raise RequestError("response_format: the grammar cannot be enforced: injected")
        return super().compile(spec)

    def constraint(self, compiled, *, think_end=None, spec=None):
        if spec is not None and spec.text == json.dumps(SHORT) and self.fail_at:
            c = Failing(self.xgr, compiled, self.vocab_size, think_end=think_end, fail_at=self.fail_at)
            c.spec = spec
            return c
        return super().constraint(compiled, think_end=think_end, spec=spec)


def grammars(fail_at: int = 0) -> Grammars:
    g = Grammars(xgr.TokenizerInfo(VOCAB, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[STOP]))
    g.fail_at = fail_at
    return g


def reference(prompt, count, schema=None) -> list[int]:
    """The serial greedy reply by hand: the fake model's best token, among the grammar's allowed ones."""

    m = None
    if schema is not None:
        m = xgr.GrammarMatcher(grammars().compile(grammar.Spec("json_schema", json.dumps(schema))))
    bits = xgr.allocate_token_bitmask(1, V)
    out, ctx = [], list(prompt)
    while len(out) < count:
        lg = logits_at(ctx).clone()
        if m is not None:
            m.fill_next_token_bitmask(bits, 0)
            allowed = torch.tensor([(int(bits[0, t // 32]) >> (t % 32)) & 1 for t in range(V)], dtype=torch.bool)
            lg[~allowed] = float("-inf")
        t = int(lg.argmax())
        out.append(t)
        ctx.append(t)
        if m is not None:
            assert m.accept_token(t)
        if t == STOP:
            break
    return out


def drafted(ref: list[int], at: int, k: int, how: str) -> list[int]:
    """``k`` drafts for the reply's token ``at`` on: the reply itself ("good"), one wrong in three ("wrong"), or a raw
    line break first, which the grammar rules out inside the line's string ("bad")."""

    out = list(ref[at:at + k]) + [6] * max(0, k - len(ref[at:at + k]))
    if how == "wrong":
        out = [(t + 1) % V if (at + j) % 3 == 1 else t for j, t in enumerate(out)]
    if how == "bad" and out:
        out[0] = VOCAB.index("\n")
    return out


# the streams a sequence admits: prompt, count, schema (None: free text), drafts
STREAMS = {
    "A": ([7, 21, 19], 48, LINE, "good"),
    "B": ([9, 9, 10, 11, 12, 13, 14, 15], 20, None, "wrong"),
    "C": ([8, 6, 6, 3, 1, 2, 4, 4], 48, LINE, "wrong"),
    "D": ([10, 4, 4], 48, SHORT, "bad"),
}
REFS = {name: reference(p, n, schema) for name, (p, n, schema, _) in STREAMS.items()}
BY_PROMPT = {tuple(p): name for name, (p, *_) in STREAMS.items()}


def legal(context, window) -> bool:
    """Whether a verified window holds only rows its stream's grammar allows after ``context`` (prompt and reply)."""

    name = next(n for n, (p, *_) in STREAMS.items() if list(context[:len(p)]) == p)
    prompt, _, schema, _ = STREAMS[name]
    if schema is None:
        return True
    m = xgr.GrammarMatcher(grammars().compile(grammar.Spec("json_schema", json.dumps(schema))))
    for t in [*context[len(prompt):], *window]:
        if m.is_terminated() or not m.accept_token(int(t)):
            return False
    return True


def cut_some(rows, name: str) -> bool:
    """Whether some window of ``name``'s ended short of the six rows its drafts filled, well before its reply's end."""

    n = len(STREAMS[name][0])
    return any(len(w) < 6 for ctx, w in rows if ctx[:n] == tuple(STREAMS[name][0]) and len(ctx) - n + 8 < len(REFS[name]))


# -- the --parallel lane --------------------------------------------------------------------------------------------
class Host:
    def __init__(self, cap: int) -> None:
        import numpy as np

        self.ids = np.zeros(cap, dtype=np.int64)

    def set(self, at: int, tokens) -> None:
        self.ids[at:at + len(tokens)] = tokens

    def view(self):
        return self.ids

    def __getitem__(self, index):
        return self.ids[index]


class Runner:
    """Each window row's logits from the slot's tokens up to it; ``rows`` records each forward window and its context."""

    def __init__(self, m, pool, graphs=False, graph_pool=None) -> None:
        self.graphs, self.rows = None, []

    def bucket(self, deepest: int) -> int:
        return deepest

    def engram_touch(self, tails) -> None:
        pass

    def forward(self, windows, replay=True):
        out = []
        for _, _, _, P, window, host in windows:
            self.rows.append((tuple(int(t) for t in host[:P]), [int(t) for t in window]))
            out += [logits_at(host[:P + i + 1]) for i in range(len(window))]
        return frozen(torch.stack(out)), None


def lane_engine(rank: int, comm, fail_at: int):
    m = SimpleNamespace(new_pool=lambda slots, cap: SimpleNamespace(),
                        pool_view=lambda pool, index, base, size: SimpleNamespace(length=0, host=Host(size), cap=size),
                        comm=comm, cfg=SimpleNamespace(window=128))

    def prefill_steps(sc, dc, prompt, image=None, start=0, snap=None):
        sc.host.set(0, prompt)
        sc.length = len(prompt)
        if len(prompt) > 6:                           # a second chunk: a round may run between the two
            yield 6
        return frozen(logits_at(prompt)[None])

    return SimpleNamespace(model=m, drafter=SimpleNamespace(), runner=None, drafts=5, eos=(STOP,), rank=rank,
                           world=2 if comm is not None else 1, prefill_steps=prefill_steps, model_dir=None,
                           grammars=grammars(fail_at),
                           _sample=lambda logits, positions, sampling: [int(x) for x in logits.argmax(-1).tolist()])


class Lane(multi.MultiDecoder):
    """The real decoder over the fakes: drafts from each stream's own reply, every finished stream recorded."""

    def __init__(self, engine) -> None:
        super().__init__(engine, slots=4, cap=4 * multi.ALIGN)
        self.record: dict[int, tuple[str, list[int], Exception | None]] = {}

    def _drafts(self, tokens, q0, slots):
        by_slot = {s.st.index: s for s in self.streams.values()}
        out = []
        for slot in slots:
            s = by_slot[slot]
            name = BY_PROMPT[tuple(s.prompt)]
            out.append(drafted(REFS[name], len(s.out), 5, STREAMS[name][3]))
        self.drafters[len(slots)] = SimpleNamespace(last_conf=[[0.0] * 5 for _ in slots])
        return out

    def _agree(self, what, plan):
        """The digest check on the fake ranks' gather (the real one builds its pieces on the GPU)."""

        import hashlib

        if self.e.world < 2:
            return
        dgst = int.from_bytes(hashlib.sha256(json.dumps(plan).encode()).digest()[:8], "big")
        every = self.m.comm.gather(torch.tensor([(dgst >> b) & 0xFFFF for b in (0, 16, 32, 48)],
                                                dtype=torch.float32)).tolist()
        if any(row != every[0] for row in every):
            raise multi.OutOfStep(f"the ranks planned different {what}s")

    def _all(self, ok):
        """The real check on the fake ranks' gather."""

        if self.e.world < 2:
            return bool(ok)
        return all(row[0] == 1.0 for row in self.m.comm.gather(torch.tensor([1.0 if ok else 0.0])).tolist())

    def _finish(self, sids, drops=()):
        for sid in sids:
            s = self.streams.get(sid)
            if s is not None:
                self.record[sid] = (BY_PROMPT[tuple(s.prompt)], list(s.out), s.error)
        super()._finish(sids, drops)


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    monkeypatch.setattr(rounds, "RoundRunner", Runner)
    monkeypatch.setattr(dspark, "DraftPool",
                        lambda d, slots: SimpleNamespace(views=[SimpleNamespace(absorbed=0) for _ in range(slots)]))
    monkeypatch.setattr(multi, "KEEP", False)
    monkeypatch.setattr(multi, "DEPTH_POLICY", "fixed")


def stream(name: str, g: Grammars, stop_eos: bool = True) -> Stream:
    """``stop_eos`` False (ignore_eos) for a grammar's stream: its grammar's end still ends it."""

    prompt, count, schema, _ = STREAMS[name]
    c = None
    if schema is not None:
        spec = grammar.Spec("json_schema", json.dumps(schema))
        c = g.constraint(g.compile(spec), spec=spec)
    s = Stream(list(prompt), count, None, constraint=c, stop_eos=stop_eos or c is None)
    s.emit = lambda new: None
    return s


def sequences(most: int):
    """Every sequence of at most ``most`` events: admit A, B, C or D (each once at most), or a round ("R")."""

    for n in range(1, most + 1):
        for seq in itertools.product("ABCDR", repeat=n):
            if all(seq.count(x) <= 1 for x in "ABCD") and any(x != "R" for x in seq):
                yield "".join(seq)


def drive(lane: Lane, seq: str, stop_eos: bool = True) -> None:
    """Rank 0's scheduler over ``seq``, then rounds until every stream ended."""

    g = lane.e.grammars
    for ev in seq:
        if ev != "R":
            lane.admit(stream(ev, g, stop_eos))
        lane.finish(lane.round() if ev == "R" else [])
    while lane.live():
        lane.finish(lane.round())


def check(record: dict, seq: str, fail_at: int) -> None:
    got = {name: (out, err) for name, out, err in record.values()}
    assert sorted(got) == sorted(x for x in seq if x != "R"), (seq, got)
    for name, (out, err) in got.items():
        if name == "D" and fail_at:
            assert isinstance(err, grammar.GrammarError), (seq, name, out)
            assert out == REFS[name][:len(out)], (seq, name, out)
        else:
            assert err is None and out == REFS[name], (seq, name, out, REFS[name])


def test_compiler_built_at_startup_is_the_servers(tmp_path):
    """The engine's startup compiler (built before its memory ceiling) is the one the server's requests compile with."""

    from tokenizers import Tokenizer, models

    Tokenizer(models.WordLevel({f"t{i}": i for i in range(40)}, unk_token="t0")).save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 40}))
    e = DsEngine.__new__(DsEngine)
    e.model_dir, e.eos, e.rank = tmp_path, (0,), 0
    e._grammar_compiler()
    assert grammar.compiler(SimpleNamespace(), tmp_path, e.eos) is e.grammars


def test_toy_replies_are_grammar_bound():
    """The references are what the grammar shapes: A, C and D end with the stop token on a {"line": ...} object."""

    for name in "ACD":
        text = "".join(VOCAB[t] for t in REFS[name][:-1])
        assert REFS[name][-1] == STOP and set(json.loads(text)) == {"line"}, (name, text)
    assert REFS["B"] != reference(STREAMS["B"][0], 20, LINE)


@pytest.mark.parametrize(("fail_at", "stop_eos"), [(0, True), (1, True), (2, True), (3, True), (4, True), (0, False)])
def test_every_sequence_one_rank(fail_at, stop_eos):
    """fail_at 1-2: D's grammar fails on its first token (mask, advance), 3-4: in its first round (window, advance).
    stop_eos False: the grammars' streams ignore the end token, and still end where their grammar does."""

    for seq in sequences(6 if fail_at == 0 else 5):
        lane = Lane(lane_engine(0, None, fail_at))
        drive(lane, seq, stop_eos)
        check(lane.record, seq, fail_at)
        assert lane.free == [0, 1, 2, 3] and lane.extents.gaps == [(0, lane.extents.total)], seq
        for ctx, window in lane.runner.rows:         # no row the grammar ruled out was verified
            assert legal(ctx, window), (seq, ctx, window)


def test_ruled_out_drafts_are_cut():
    """D's line-break drafts, which its grammar rules out inside the string, shorten its windows (C's wrong drafts are
    characters the string allows: the verify step rejects those)."""

    lane = Lane(lane_engine(0, None, 0))
    drive(lane, "D")
    assert cut_some(lane.runner.rows, "D")


class Link:
    """Rank 0's steps to rank 1 in order, through JSON as the TCP link sends them."""

    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()

    def send(self, op) -> None:
        self.q.put(None if op is None else json.dumps(op))

    def receive(self):
        got = self.q.get(timeout=20)
        return None if got is None else json.loads(got)


class Comm:
    """The two ranks' all-gather: a barrier on both sides, so a step one rank skips times out instead of passing."""

    def __init__(self) -> None:
        self.barrier, self.rows = threading.Barrier(2, timeout=20), [None, None]

    def side(self, rank: int):
        def gather(t):
            self.rows[rank] = t.clone()
            self.barrier.wait()
            out = torch.stack(self.rows)
            self.barrier.wait()
            return out

        return SimpleNamespace(gather=gather)


@pytest.mark.parametrize(("fails", "refuse"), [((0, 0), False), ((2, 2), False), ((3, 3), False),
                                               ((1, 0), False), ((2, 0), False), ((3, 0), False), ((4, 0), False),
                                               ((0, 1), False), ((0, 2), False), ((0, 3), False), ((0, 4), False),
                                               ((0, 0), True)])
def test_every_sequence_two_ranks(fails, refuse):
    """Both ranks compile each grammar, cut and mask the same rows, and end the same streams the same way. ``fails``:
    each rank's fail_at for D's grammar (one rank alone failing it); ``refuse``: rank 1 cannot compile D's grammar.
    Either way D fails on both ranks and they serve the rest."""

    for seq in sequences(5 if fails[0] == fails[1] and not refuse else 4):
        comm, link = Comm(), Link()
        lane0, lane1 = Lane(lane_engine(0, comm.side(0), fails[0])), Lane(lane_engine(1, comm.side(1), fails[1]))
        lane1.e.grammars.refuse = refuse
        lane0.link = link
        errors = []

        def follower():
            try:
                lane1.follow(link)
            except RuntimeError as exc:
                if "link closed" not in str(exc):
                    errors.append(exc)
            except Exception as exc:                 # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=follower)
        t.start()
        try:
            drive(lane0, seq)
        finally:
            link.send(None)
            t.join(30)
        assert not errors, (seq, errors)
        check(lane0.record, seq, max(fails) or int(refuse))
        same = fails[0] == fails[1] and not refuse      # (else the rank that failed first took fewer of D's tokens)
        assert {k: (n, o if same or n != "D" else None, type(e)) for k, (n, o, e) in lane1.record.items()} == \
               {k: (n, o if same or n != "D" else None, type(e)) for k, (n, o, e) in lane0.record.items()}, seq
        assert lane1.runner.rows == lane0.runner.rows, seq


# -- the serial engine (its windows go to the GPU as token ids, so these need one) ----------------------------------
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the serial engine builds its windows on the GPU")


class Cache(SimpleNamespace):
    pass


def serial_engine(fail_at: int, rank: int = 0, share=None, comm=None) -> DsEngine:
    e = DsEngine.__new__(DsEngine)
    seen = []

    def forward(sc, ids, P, all_logits=False, taps=None, host_ids=None, **_):
        window = [int(t) for t in ids.tolist()]
        seen.append((tuple(int(t) for t in sc.host[:P]), window))
        sc.host.set(P, window)
        sc.length = P + len(window)
        if taps is not None:
            taps.append(torch.zeros((len(window), 1)))
        return frozen(torch.stack([logits_at(sc.host[:P + i + 1]) for i in range(len(window))]))

    def draft(dc, sc, tok, P):
        prompt = e._prompt
        name = BY_PROMPT[tuple(prompt)]
        return drafted(REFS[name], P - len(prompt) + 1, 5, STREAMS[name][3]), [0.0] * 5

    def prefill(sc, dc, prompt, image=None):
        e._prompt = list(prompt)
        sc.host.set(0, prompt)
        sc.length = len(prompt)
        return frozen(logits_at(prompt)[None])

    e.model = SimpleNamespace(forward=forward, new_cache=lambda n: Cache(length=0, host=Host(n), cap=n))
    e.drafter = SimpleNamespace(draft=draft, absorb=lambda dc, sc, taps, P: None, new_cache=lambda: Cache())
    e.runner, e.drafts, e.limit, e.max_rows, e.eos = None, 5, 4096, 6, (STOP,)
    e.rank, e.world, e.quiet, e.multi, e.scheduler = rank, 2 if share else 1, True, None, None
    e.request, e.model_dir, e.grammars = threading.local(), None, grammars(fail_at)
    e.prefill, e.seen = prefill, seen
    e._sample = lambda logits, positions, sampling: [int(x) for x in logits.argmax(-1).tolist()]
    if share is not None:
        e._share = share
    if comm is not None:                              # the ranks' check, on the fake ranks' gather
        e._all = lambda ok: bool(comm.gather(torch.tensor([float(ok)])).amin())
    return e


def run_serial(e: DsEngine, name: str, draft: bool, stop_eos: bool = True):
    prompt, count, schema, _ = STREAMS[name]
    c = None
    if schema is not None:
        spec = grammar.Spec("json_schema", json.dumps(schema))
        c = e.grammars.constraint(e.grammars.compile(spec), spec=spec)
    out: list[int] = []
    e.request.stop_eos = stop_eos or c is None
    e.generate(list(prompt), count, None, out.extend, draft=draft, constraint=c)
    return out


@gpu
@pytest.mark.parametrize("name", list("ABCD"))
@pytest.mark.parametrize("draft", [True, False])
@pytest.mark.parametrize("stop_eos", [True, False])
def test_serial_engine(name, draft, stop_eos):
    """stop_eos False: a grammar's reply ignores the end token, and still ends where its grammar does."""

    e = serial_engine(0)
    assert run_serial(e, name, draft, stop_eos) == REFS[name]
    assert all(legal(ctx, w) for ctx, w in e.seen)
    if draft and name == "D":
        assert cut_some(e.seen, name)


@gpu
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("draft", [True, False])
def test_serial_engine_grammar_failure(fail_at, draft):
    e = serial_engine(fail_at)
    with pytest.raises(grammar.GrammarError):
        run_serial(e, "D", draft)
    assert run_serial(e, "A", draft) == REFS["A"]      # the engine goes on serving


def test_a_cut_window_replays_a_warm_graph(monkeypatch):
    """A verify window the grammar cut to its pending token (one row, taps on) replays a graph the warm-up captured:
    none is captured while serving, past the allocator ceiling."""

    from tensorfold.families.deepseek_v41.cuda import graph

    def capture(self, sc, n, b, taps, ids, start, e_rows):
        g = self.graphs[(n, b, taps)] = SimpleNamespace(run=lambda ids, start, e_rows: None, taps=None)
        return g

    monkeypatch.setattr(graph.GraphRunner, "_capture", capture)
    monkeypatch.setattr(graph.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(graph.torch.cuda, "memory_reserved", lambda: 0)
    r = graph.GraphRunner(SimpleNamespace(engram=None), max_rows=6)
    sc = SimpleNamespace(cap=4096, length=0)
    r.warm(sc, 4000)
    for start in (0, 300, 1500, 3990):
        for n, taps in ((1, False), (1, True), (2, True), (6, True)):
            r.forward(sc, [5] * n, start, taps)
    assert r.captures == 0
    alone = graph.GraphRunner(SimpleNamespace(engram=None), max_rows=1)     # no drafter: no verify windows
    alone.warm(sc, 4000)
    assert {taps for _, _, taps in alone.graphs} == {False}


def first_stop_prompt() -> list[int]:
    """A prompt whose next row prefers the stop token to "a": under the regex (a)? its reply is the stop token."""

    a = VOCAB.index("a")
    return next([x, y] for x in range(1, V) for y in range(1, V) if logits_at([x, y])[STOP] > logits_at([x, y])[a])


def test_a_grammar_that_ends_at_the_first_token_ends_the_lane_reply(monkeypatch):
    """A grammar that takes the empty reply, with ignore_eos: a first token that ends it ends the reply (past its end
    nothing is masked)."""

    prompt = first_stop_prompt()
    monkeypatch.setitem(BY_PROMPT, tuple(prompt), "Z")
    lane = Lane(lane_engine(0, None, 0))
    spec = grammar.Spec("regex", "(a)?")
    s = Stream(prompt, 6, None, constraint=lane.e.grammars.constraint(lane.e.grammars.compile(spec), spec=spec),
               stop_eos=False, draft=False)
    s.emit = lambda new: None
    lane.admit(s)
    while lane.live():
        lane.finish(lane.round())
    assert s.error is None and s.out == [STOP], s.out


@gpu
def test_a_grammar_that_ends_at_the_first_token_ends_the_serial_reply():
    e = serial_engine(0)
    spec = grammar.Spec("regex", "(a)?")
    out: list[int] = []
    e.request.stop_eos = False
    e.generate(first_stop_prompt(), 6, None, out.extend, draft=False,
               constraint=e.grammars.constraint(e.grammars.compile(spec), spec=spec))
    assert out == [STOP]


class Stop(Exception):
    pass


@gpu
@pytest.mark.parametrize(("fails", "refuse"), [((3, 3), False), ((1, 0), False), ((2, 0), False), ((3, 0), False),
                                               ((4, 0), False), ((0, 1), False), ((0, 2), False), ((0, 3), False),
                                               ((0, 4), False), ((0, 0), True)])
@pytest.mark.parametrize("draft", [True, False])
def test_serial_two_ranks_follow_the_grammar(fails, refuse, draft):
    """Rank 1 reads the grammar from the request header, walks the same rows, and outlives a failed grammar: one both
    ranks fail at the same step, one rank alone fails (``fails``: each rank's fail_at), or rank 1 cannot compile."""

    def lead(e0):
        out = {"A": run_serial(e0, "A", draft), "B": run_serial(e0, "B", draft)}
        with pytest.raises(grammar.GrammarError):
            run_serial(e0, "D", draft)
        out["C"] = run_serial(e0, "C", draft)
        return out

    out, e0, e1 = serial_pair(lead, fails, refuse)
    assert out == {k: REFS[k] for k in out}
    assert e1.seen == e0.seen


def serial_pair(lead, fails=(0, 0), refuse=False):
    """``lead(rank 0)``'s requests on two serial engines, rank 1 following over the header shares and the fake gather;
    returns what ``lead`` returned and both engines once rank 1 served them all without an error."""

    q: queue.Queue = queue.Queue()

    def share(values):
        q.put(list(values))
        return list(values)

    def follow(values):
        got = q.get(timeout=20)
        if got is None:
            raise Stop
        return got

    comm = Comm()
    e0, e1 = serial_engine(fails[0], 0, share, comm.side(0)), serial_engine(fails[1], 1, follow, comm.side(1))
    e1.grammars.refuse = refuse
    errors = []

    def follower():
        try:
            e1.follow()
        except Stop:
            pass
        except Exception as exc:                     # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=follower)
    t.start()
    try:
        out = lead(e0)
    finally:
        q.put(None)
        t.join(30)
    assert not errors, errors
    return out, e0, e1


@gpu
def test_serial_a_grammar_that_cannot_be_sent_fails_before_rank_1_reads_it():
    """A request whose grammar cannot be packed fails on rank 0 before any of it is shared: rank 1 serves the next."""

    def lead(e0):
        spec = grammar.Spec("json_schema", json.dumps(LINE))
        c = e0.grammars.constraint(e0.grammars.compile(spec), spec=spec)
        c.spec = grammar.Spec("regex", "\ud800")                 # no UTF-8 form
        with pytest.raises(UnicodeEncodeError):
            e0.generate([7, 21, 19], 8, None, lambda new: None, constraint=c)
        return run_serial(e0, "A", True)

    out, e0, e1 = serial_pair(lead)
    assert out == REFS["A"] and e1.seen == e0.seen
