"""Output logprobs on DeepSeek-V4.1's engines (the fakes of test_deepseek_v41_grammar, logits on the GPU): each kept
token's log probability and top alternatives come from the row that chose it, equal with drafts on or off, alone or
beside other streams in every short sequence, on the serial and the --parallel engine."""

from __future__ import annotations

import itertools
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xgrammar")
if not torch.cuda.is_available():
    pytest.skip("the log probability kernels run on the GPU", allow_module_level=True)

pytestmark = pytest.mark.torch

import test_deepseek_v41_grammar as g  # noqa: E402
from test_deepseek_v41_grammar import _fakes  # noqa: E402,F401  (the lane's fakes, autouse here too)

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.probabilities import Probabilities  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import rounds  # noqa: E402

TOP = 3
g.STREAMS["E"] = ([11, 12, 13, 6, 6], 24, None, "wrong")        # free text with logprobs, wrong drafts
g.STREAMS["F"] = ([13, 7, 21], 24, None, "good")
for _name in "EF":
    g.REFS[_name] = g.reference(g.STREAMS[_name][0], g.STREAMS[_name][1])
    g.BY_PROMPT[tuple(g.STREAMS[_name][0])] = _name


def expected(name: str) -> dict:
    """The rows by hand: log-softmax of the fake model's logits after each prefix, in float64."""

    prompt, out = g.STREAMS[name][0], g.REFS[name]
    rows = {}
    for i, t in enumerate(out):
        lp = torch.log_softmax(g.logits_at(prompt + out[:i]).double(), -1)
        top = torch.topk(lp, TOP)
        rows[len(prompt) + i] = (t, float(lp[t]), [int(x) for x in top.indices])
    return rows


def close(rows: dict, name: str) -> None:
    want = expected(name)
    assert sorted(rows) == sorted(want), name
    for pos, row in rows.items():
        t, lp, top = want[pos]
        assert row["id"] == t and abs(row["logprob"] - lp) < 1e-5, (name, pos, row, want[pos])
        assert [i for i, _ in row["top"]] == top, (name, pos)


def gpu(t: torch.Tensor) -> torch.Tensor:
    """``t`` on the GPU, still an inference tensor as the engine's logits are."""

    with torch.inference_mode():
        return t.cuda()


class CudaRunner(g.Runner):
    def forward(self, windows, replay=True):
        logits, taps = super().forward(windows, replay)
        return gpu(logits), taps


@pytest.fixture(autouse=True)
def _cuda_logits(monkeypatch, request):
    request.getfixturevalue("_fakes")                 # after the lane's fakes, so this runner replaces theirs
    monkeypatch.setattr(rounds, "RoundRunner", CudaRunner)


def cuda_engine(rank: int = 0, comm=None):
    e = g.lane_engine(rank, comm, 0)
    steps = e.prefill_steps

    def prefill_steps(*args, **kwargs):
        last = yield from steps(*args, **kwargs)
        return gpu(last)

    e.prefill_steps = prefill_steps
    return e


def stream(name: str, lane, draft: bool = True) -> Stream:
    if name not in "EF":
        return g.stream(name, lane.e.grammars)
    prompt, count, _, _ = g.STREAMS[name]
    s = Stream(list(prompt), count, None, draft=draft,
               probabilities=Probabilities(TOP, len(prompt), count) if name == "E" else None)
    s.emit = lambda new: None
    return s


def drive(lane, seq: str, draft: bool = True) -> dict:
    """Rank 0 over ``seq`` (as ``g.drive``); returns E's probability rows."""

    held = {}
    for ev in seq:
        if ev != "R":
            s = stream(ev, lane, draft)
            held[ev] = s
            lane.admit(s)
        lane.finish(lane.round() if ev == "R" else [])
    while lane.live():
        lane.finish(lane.round())
    for name, s in held.items():
        assert s.error is None and s.out == g.REFS[name], (seq, name)
    return held["E"].probabilities.rows if "E" in held else {}


def test_lane_rows_are_the_model_rows():
    lane = g.Lane(cuda_engine())
    close(drive(lane, "E"), "E")


def test_lane_rows_equal_with_and_without_drafts_beside_any_streams():
    """E's rows with drafts, beside A (grammar), B (free) and F (free, good drafts) in every order, equal its rows
    decoded one token a round alone, value for value."""

    solo = drive(g.Lane(cuda_engine()), "E", draft=False)
    for n in range(1, 6):
        for seq in itertools.product("ABEFR", repeat=n):
            if seq.count("E") == 1 and all(seq.count(x) <= 1 for x in "ABF"):
                assert drive(g.Lane(cuda_engine()), "".join(seq)) == solo, seq


def test_lane_two_ranks_rank0_reads_alone():
    comm, link = g.Comm(), g.Link()
    lane0, lane1 = g.Lane(cuda_engine(0, comm.side(0))), g.Lane(cuda_engine(1, comm.side(1)))
    lane0.link = link
    errors = []

    def follower():
        try:
            lane1.follow(link)
        except RuntimeError as exc:
            if "link closed" not in str(exc):
                errors.append(exc)

    t = threading.Thread(target=follower)
    t.start()
    try:
        rows = drive(lane0, "AERB")
    finally:
        link.send(None)
        t.join(30)
    assert not errors
    close(rows, "E")


def serial_rows(draft: bool) -> dict:
    e = g.serial_engine(0)
    forward, prefill = e.model.forward, e.prefill
    e.model.forward = lambda *a, **k: gpu(forward(*a, **k))
    e.prefill = lambda *a, **k: gpu(prefill(*a, **k))
    prompt, count, _, _ = g.STREAMS["E"]
    probs, out = Probabilities(TOP, len(prompt), count), []
    e.generate(list(prompt), count, None, out.extend, draft=draft, probabilities=probs)
    assert out == g.REFS["E"]
    assert [r["id"] for r in probs.emitted(out)] == out
    return probs.rows


def test_serial_rows_with_and_without_drafts():
    drafted = serial_rows(True)
    close(drafted, "E")
    assert drafted == serial_rows(False)


def test_serial_and_lane_rows_agree():
    assert serial_rows(True) == drive(g.Lane(cuda_engine()), "E")


class Refusing(Probabilities):
    """Probabilities that refuse their ``at``-th batch of rows, as a row that is not finite or a changed replay does."""

    def __init__(self, *args, at: int) -> None:
        super().__init__(*args)
        self.calls, self.at = 0, at

    def add(self, *args) -> None:
        self.calls += 1
        if self.calls == self.at:
            raise RuntimeError("target log probabilities are not finite")
        super().add(*args)


@pytest.mark.parametrize("at", [1, 2])
def test_lane_refused_rows_fail_their_request_only(at):
    """Rank 0 alone reads E's rows; when they are refused (at its first token, or in a round) E fails with the error,
    every other stream in every short sequence gets its reply, and rank 1 steps the same rows throughout."""

    for n in range(1, 5):
        for seq in itertools.product("ABEFR", repeat=n):
            if seq.count("E") != 1 or any(seq.count(x) > 1 for x in "ABF"):
                continue
            comm, link = g.Comm(), g.Link()
            lane0, lane1 = g.Lane(cuda_engine(0, comm.side(0))), g.Lane(cuda_engine(1, comm.side(1)))
            lane0.link = link
            errors = []

            def follower():
                try:
                    lane1.follow(link)
                except RuntimeError as exc:
                    if "link closed" not in str(exc):
                        errors.append(exc)

            t = threading.Thread(target=follower)
            t.start()
            try:
                for ev in seq:
                    if ev != "R":
                        s = stream(ev, lane0)
                        if ev == "E":
                            prompt, count, _, _ = g.STREAMS["E"]
                            s.probabilities = Refusing(TOP, len(prompt), count, at=at)
                        lane0.admit(s)
                    lane0.finish(lane0.round() if ev == "R" else [])
                while lane0.live():
                    lane0.finish(lane0.round())
            finally:
                link.send(None)
                t.join(30)
            assert not errors, (seq, errors)
            for name, out, err in lane0.record.values():
                if name == "E":
                    assert isinstance(err, RuntimeError) and "not finite" in str(err), (seq, err)
                else:
                    assert err is None and out == g.REFS[name], (seq, name)
            assert lane1.runner.rows == lane0.runner.rows, seq


@pytest.mark.parametrize("at", [1, 2])
@pytest.mark.parametrize("draft", [True, False])
def test_serial_refused_rows_fail_their_request_in_step(at, draft):
    """Rank 0's rows refused mid-reply: both ranks still decode the reply to its end in step, rank 0 raises the error,
    and they serve the next request."""

    def lead(e0):
        forward, prefill = e0.model.forward, e0.prefill
        e0.model.forward = lambda *a, **k: gpu(forward(*a, **k))
        e0.prefill = lambda *a, **k: gpu(prefill(*a, **k))
        prompt, count, _, _ = g.STREAMS["E"]
        with pytest.raises(RuntimeError, match="not finite"):
            e0.generate(list(prompt), count, None, lambda new: None, draft=draft,
                        probabilities=Refusing(TOP, len(prompt), count, at=at))
        return g.run_serial(e0, "A", draft)

    out, e0, e1 = g.serial_pair(lead)
    assert out == g.REFS["A"] and e1.seen == e0.seen


def test_logprobs_with_a_grammar_are_refused():
    e = g.serial_engine(0)
    spec = g.grammar.Spec("json_schema", g.json.dumps(g.LINE))
    c = e.grammars.constraint(e.grammars.compile(spec), spec=spec)
    with pytest.raises(ValueError, match="structured output"):
        e.generate([7, 21, 19], 8, None, lambda new: None, constraint=c, probabilities=Probabilities(TOP, 3, 8))
    assert e.supports_logprobs
