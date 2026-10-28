"""The answer reserve (server.answer_reserve): a thinking reply keeps part of max_tokens for its answer, the same in
every door; after the server's close an answer that would end empty starts from a lead, once; tools still come after
the close; nothing changes for a reply that closes its own thinking or with --answer-reserve 0. An engine whose ranks
decode in step ends the run at the cut (think_stop) with the same reply, and keeps the prompt's snapshot through the
cut's continuation (continued)."""

import json
import random
from types import SimpleNamespace

import pytest

from tensorfold.cuda.turns import Turns
from tensorfold.engine.call_gate import AnswerLead, ThinkBudget, ThinkStop
from tensorfold.engine.lane_engine import LaneStream
from tensorfold.server import answer_reserve
from tensorfold.server.answer_reserve import LEAD, reserve, settings, think_budget
from tests.test_cuda_admission import http_server
from tests.test_cuda_thinking_controls import app_for as thinking_app
from tests.test_cuda_tool_choice import CALL, CLOSE as CALL_END, END, EOS, OPEN, THINK, TOOLS, Tokens, events
from tests.test_responses_api import call, stream_events

NL = ord("\n")
SHUT = [NL, END, NL, NL]                       # the server's close: "\n</think>\n\n"
ON = (0.25, 1024)
LETTERS = [ord("a") + i % 26 for i in range(400)]


def text(ids):
    return Tokens().encode(ids).ids if isinstance(ids, str) else list(ids)


class Script:
    """A thinking model on a script, ``width`` tokens a round: from the generation prompt it reasons in letters (or
    closes by itself at ``closes_at`` and answers "Hi!"), after the server's close it writes ``after_close``, after
    the lead ``after_lead``; after a call's opener it writes the call. Records every run's prompt and length."""

    eos = (EOS,)

    def __init__(self, after_close=(EOS,), after_lead=" 42." , closes_at=None, width=3):
        self.after_close, self.after_lead, self.closes_at, self.width = after_close, after_lead, closes_at, width
        self.runs = []

    def reply(self, prompt):
        written = Tokens().decode(prompt)
        if written.endswith("<function="):
            return text(CALL[len("\n<function="):]) + [CALL_END, EOS]
        if prompt[-1] == OPEN:
            return text(CALL) + [CALL_END, EOS]
        if written.endswith(LEAD):
            return text(self.after_lead) + [EOS]
        if prompt[-len(SHUT):] == SHUT:
            return text(self.after_close)
        out = list(LETTERS)
        if self.closes_at is not None:
            out[self.closes_at:] = [END, NL, NL, *text("Hi!"), EOS]
        return out

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.runs.append((list(prompt), max_tokens))
        reply = self.reply(list(prompt))[:max_tokens]
        for at in range(0, len(reply), self.width):
            if on_tokens(reply[at:at + self.width]):
                break
        return {"rounds": 1}


def app_for(tmp_path, engine, reserve_=ON, max_tokens=48, budget=0, context=0, unset=False):
    app = thinking_app(tmp_path, engine, budget=budget)
    app.max_tokens = max_tokens
    app.native_context_window = app.context_window = context
    if not unset:                              # unset: an App from before the reserve (no attribute at all)
        app.answer_reserve = reserve_
    return app


def chat(app, stream=False, route="/v1/chat/completions", **fields):
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": stream, "return_token_ids": True, **fields}
    if route != "/v1/chat/completions":
        body = {**fields, **({"stream": True} if stream else {})}
    with http_server(app) as port:
        status, raw = call(port, "POST", route, body)
    assert status == 200, raw
    return raw


def reply_of(raw, stream):
    """(content, reasoning, finish, usage, tensorfold block) of a chat completion, streamed or not."""

    if not stream:
        body = json.loads(raw)
        choice = body["choices"][0]
        calls = [c["function"]["name"] for c in choice["message"].get("tool_calls") or []]
        return (choice["message"]["content"] or "", choice["message"].get("reasoning_content") or "",
                choice["finish_reason"], body["usage"], body["tensorfold"], calls or None)
    chunks = events(raw)
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    last = [c for c in chunks if c.get("choices")][-1]
    calls = [c["function"]["name"] for d in deltas for c in d.get("tool_calls") or []
             if (c.get("function") or {}).get("name")]
    return ("".join(d.get("content") or "" for d in deltas), "".join(d.get("reasoning_content") or "" for d in deltas),
            last["choices"][0]["finish_reason"], chunks[-1].get("usage") or last.get("usage"), last["tensorfold"],
            calls or None)


# -- the arithmetic ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("max_tokens, kept, budget", [
    (1, 0, 0), (2, 1, 1), (3, 1, 2), (24, 12, 12), (128, 64, 64), (1500, 750, 750), (2048, 1024, 1024),
    (4096, 1024, 3072), (6000, 1500, 4500), (65536, 16384, 49152)])
def test_the_reserve_is_a_quarter_at_least_1024_at_most_half(max_tokens, kept, budget):
    assert reserve(max_tokens, *ON) == kept
    assert think_budget(0, max_tokens, *ON) == budget
    assert reserve(max_tokens, 0.0, 1024) == 0 and think_budget(0, max_tokens, 0.0, 1024) == 0


@pytest.mark.parametrize("ratio, minimum, max_tokens, kept", [
    (0.1, 0, 30, 3), (0.1, 0, 31, 4), (0.5, 0, 9, 4), (0.25, 0, 4096, 1024), (0.25, 4000, 4096, 2048),
    (0.3, 1, 10, 3)])
def test_other_shares_and_floors(ratio, minimum, max_tokens, kept):
    assert reserve(max_tokens, ratio, minimum) == kept      # 0.1 * 30 is 3, not float noise above it


@pytest.mark.parametrize("asked, want", [(0, 3072), (-1, 3072), (1, 1), (100, 100), (3072, 3072), (3073, 3072),
                                         (10**6, 3072)])
def test_a_smaller_request_budget_wins_and_a_larger_one_is_held_to_the_cap(asked, want):
    assert think_budget(asked, 4096, *ON) == want


@pytest.mark.parametrize("asked", [-1, 0, 5, 10**6])
def test_without_a_reserve_the_budget_is_as_asked(asked):
    assert think_budget(asked, 4096, 0.0, 1024) == asked
    assert think_budget(asked, 1, *ON) == asked               # max_tokens 1 keeps nothing back


def test_settings_flags_family_and_refusals():
    assert settings(None, None, None) == (0.0, 1024)
    assert settings((0.25, 1024), None, None) == (0.25, 1024)
    assert settings((0.25, 1024), 0.0, None) == (0.0, 1024)
    assert settings(None, 0.3, 512) == (0.3, 512)
    for ratio, minimum in ((-0.1, None), (0.6, None), (None, -1)):
        with pytest.raises(ValueError):
            settings((0.25, 1024), ratio, minimum)
    from tensorfold.families import bailing_v3
    assert bailing_v3.ANSWER_RESERVE == (0.25, 1024)
    assert "3072" in answer_reserve.note(0.25, 1024, 4096) and answer_reserve.note(0.0, 1024, 4096) is None


def test_the_cli_takes_the_family_default_and_the_flags():
    from tensorfold.cli import _answer_reserve, build_parser

    parser = build_parser()
    ling = SimpleNamespace(package=SimpleNamespace(ANSWER_RESERVE=(0.25, 1024)))
    other = SimpleNamespace(package=SimpleNamespace())
    assert _answer_reserve(parser.parse_args(["serve", "x"]), ling) == (0.25, 1024)
    assert _answer_reserve(parser.parse_args(["serve", "x"]), other) == (0.0, 1024)
    assert _answer_reserve(parser.parse_args(["serve", "x", "--answer-reserve", "0"]), ling) == (0.0, 1024)
    assert _answer_reserve(parser.parse_args(["serve", "x", "--answer-reserve", "0.3",
                                              "--answer-reserve-min", "256"]), other) == (0.3, 256)


def test_the_reserve_runs_on_the_reply_length_left_near_the_context_end(tmp_path):
    app = app_for(tmp_path, Script(), max_tokens=4096)
    prompt = len(app.prepare({"messages": [{"role": "user", "content": "Hi"}]}, True).prompt)
    app.context_window = prompt + 1144                        # a request without max_tokens: 1144 replies left
    prepared = app.prepare({"messages": [{"role": "user", "content": "Hi"}]}, True)
    assert prepared.max_tokens == 1144 and prepared.think_budget == 1144 - 572
    off = app.prepare({"messages": [{"role": "user", "content": "Hi"}],
                       "chat_template_kwargs": {"enable_thinking": False}}, True)
    assert off.think_budget == 0                               # no thinking: no budget
    raw = app.prepare({"prompt": "Hi"}, False)
    assert raw.think_budget == 0                               # a raw completion never has one


# -- the cut, on the CUDA server --------------------------------------------------------------------------------

@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_a_reply_that_thinks_to_the_end_gets_its_answer(tmp_path, stream, width):
    engine = Script(after_close=text("Hi there.") + [EOS], width=width)
    content, reasoning, finish, usage, stats, _ = reply_of(chat(app_for(tmp_path, engine), stream), stream)
    assert [n for _, n in engine.runs] == [48, 48 - 27]       # max_tokens 48 keeps 24: the 24th token closes
    assert engine.runs[1][0][len(engine.runs[0][0]):] == LETTERS[:23] + SHUT
    assert content == "Hi there." and finish == "stop" and reasoning == Tokens().decode(LETTERS[:23]) + "\n"
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 25, "reasoning_cut": True}
    assert stats["think_cut"] == 23 and "answer_lead" not in stats


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("after_close, at", [((EOS,), 27), ((ord(" "), EOS), 28), ((NL, ord(" "), EOS), 29)])
def test_an_answer_that_would_end_empty_starts_from_the_lead(tmp_path, stream, after_close, at, capsys):
    engine = Script(after_close=after_close)
    content, _, finish, usage, stats, _ = reply_of(chat(app_for(tmp_path, engine), stream), stream)
    assert content.strip() == LEAD + " 42." and finish == "stop"
    assert len(engine.runs) == 3 and Tokens().decode(engine.runs[2][0]).endswith(LEAD)
    assert stats["think_cut"] == 23 and stats["answer_lead"] == at
    assert usage["completion_tokens_details"]["reasoning_cut"] is True
    done = [line for line in capsys.readouterr().out.splitlines() if "] done req-" in line]
    assert done and done[-1].endswith(f" think_cut=23 answer_lead={at}")


def test_the_lead_goes_in_once(tmp_path):
    engine = Script(after_close=(EOS,), after_lead="")      # the model ends right after the lead too
    content, _, finish, _, stats, _ = reply_of(chat(app_for(tmp_path, engine)), False)
    assert content == LEAD and finish == "stop" and len(engine.runs) == 3 and stats["answer_lead"] == 27


def test_the_lead_never_runs_past_max_tokens(tmp_path):
    engine = Script(after_close=(EOS,), after_lead=" a long answer that does not fit")
    content, _, finish, usage, _, _ = reply_of(chat(app_for(tmp_path, engine)), False)
    assert finish == "length" and usage["completion_tokens"] == 48 and content.startswith(LEAD)


def test_reserve_zero_is_the_budget_as_before(tmp_path):
    """--answer-reserve 0: no cut without a budget; with one, the old close and an empty answer, as before."""

    for reserve_ in ((0.0, 1024), None):
        engine = Script(after_close=(EOS,))
        app = app_for(tmp_path, engine, reserve_=reserve_, unset=reserve_ is None)
        content, _, finish, usage, stats, _ = reply_of(chat(app), False)
        assert len(engine.runs) == 1 and finish == "length" and content == "" and "think_cut" not in stats
        assert "reasoning_cut" not in usage["completion_tokens_details"]
        engine = Script(after_close=(EOS,))
        app = app_for(tmp_path, engine, reserve_=reserve_, unset=reserve_ is None)
        content, _, finish, usage, stats, _ = reply_of(chat(app, thinking_budget=8), False)
        assert len(engine.runs) == 2 and content == "" and finish == "stop" and stats["think_cut"] == 7


@pytest.mark.parametrize("closes_at", [0, 3, 10])
@pytest.mark.parametrize("width", [1, 4])
def test_a_reply_that_closes_its_own_thinking_is_unchanged(tmp_path, closes_at, width):
    replies = []
    for reserve_ in (ON, (0.0, 1024)):
        engine = Script(closes_at=closes_at, width=width)
        content, _, finish, usage, stats, _ = reply_of(chat(app_for(tmp_path, engine, reserve_=reserve_)), False)
        assert len(engine.runs) == 1 and content == "Hi!" and finish == "stop" and "think_cut" not in stats
        replies.append((stats["token_ids"], stats["token_sha"], usage))
    assert replies[0] == replies[1]


def test_a_closed_reply_that_ends_empty_is_left_alone(tmp_path):
    """The lead follows the server's close only: a reply that closes itself and stops is the model's own."""

    class Empty(Script):
        def reply(self, prompt):
            return [ord("a"), END, EOS] if prompt[-1] == THINK else super().reply(prompt)

    engine = Empty()
    content, _, finish, _, stats, _ = reply_of(chat(app_for(tmp_path, engine)), False)
    assert len(engine.runs) == 1 and content == "" and finish == "stop" and stats["token_ids"] == [ord("a"), END, EOS]


@pytest.mark.parametrize("asked, cut", [(5, 4), (12, 11), (30, 23), (-1, 23)])
def test_the_request_budget_under_the_reserve(tmp_path, asked, cut):
    engine = Script(after_close=text("ok") + [EOS])
    _, _, _, _, stats, _ = reply_of(chat(app_for(tmp_path, engine), thinking_budget=asked), False)
    assert stats["think_cut"] == cut


# -- an engine whose ranks decode in step ends the run at the cut -----------------------------------------------

class ToTheEnd(Script):
    """Two ranks decoding in step on the script: a run goes on to max_tokens or the end whatever on_tokens answers
    (bailing_v3 before think_stop); ``decoded`` counts each run's tokens."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.decoded = []

    def run(self, prompt, max_tokens, on_tokens, stop=None):
        self.runs.append((list(prompt), max_tokens))
        reply, n = self.reply(list(prompt))[:max_tokens], 0
        for at in range(0, len(reply), self.width):
            n += len(reply[at:at + self.width])
            on_tokens(reply[at:at + self.width])
            if stop is not None and stop.hit(reply[at:at + self.width]):
                break
        self.decoded.append(n)
        return {"rounds": 1}

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        return self.run(prompt, max_tokens, on_tokens)


class AtTheCut(ToTheEnd):
    """The same ranks taking think_stop (every rank ends the run where the server's budget cuts) and continued."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.asked, self.continued = [], []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, think_stop=None, continued=False):
        self.asked.append(think_stop)
        self.continued.append(continued)
        return self.run(prompt, max_tokens, on_tokens, ThinkStop(*think_stop) if think_stop else None)


def same_reply(raw, stream):
    content, reasoning, finish, usage, stats, calls = reply_of(raw, stream)
    return content, reasoning, finish, usage, stats["token_ids"], stats["token_sha"], stats.get("think_cut"), calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("width", [1, 3, 5])
def test_ranks_in_step_end_the_run_at_the_cut_with_the_same_reply(tmp_path, stream, width):
    on, stop = (engine(after_close=text("Hi there.") + [EOS], width=width) for engine in (ToTheEnd, AtTheCut))
    replies = [same_reply(chat(app_for(tmp_path, engine), stream), stream) for engine in (on, stop)]
    assert replies[0] == replies[1] and replies[0][0] == "Hi there." and replies[0][6] == 23
    assert [n for _, n in on.runs] == [n for _, n in stop.runs] == [48, 48 - 27]
    assert on.decoded == [48, 10]                              # on to max_tokens, then "Hi there." and the end
    assert stop.decoded == [min(48, -(-24 // width) * width), 10]  # the round that holds the 24th token
    assert stop.asked == [(24, END), None] and stop.continued == [False, True]


@pytest.mark.parametrize("closes_at", [0, 3, 10])
@pytest.mark.parametrize("width", [1, 4])
def test_ranks_in_step_run_a_reply_that_closes_its_own_thinking_as_before(tmp_path, closes_at, width):
    replies = []
    for engine in (ToTheEnd(closes_at=closes_at, width=width), AtTheCut(closes_at=closes_at, width=width)):
        replies.append(same_reply(chat(app_for(tmp_path, engine)), False))
        assert len(engine.runs) == 1 and engine.decoded == [closes_at + 7]    # </think>, a blank line, "Hi!", the end
    assert replies[0] == replies[1] and replies[0][0] == "Hi!" and replies[0][6] is None


def test_the_stop_is_asked_while_the_reply_thinks_and_continued_after_a_cut(tmp_path):
    for reserve_ in (ON, (0.0, 1024)):                         # a request's own budget, with the reserve or without
        engine = AtTheCut(after_close=text("ok") + [EOS])
        reply_of(chat(app_for(tmp_path, engine, reserve_=reserve_), thinking_budget=8), False)
        assert engine.asked == [(8, END), None] and engine.continued == [False, True]
    engine = AtTheCut(after_close=(EOS,))                      # the lead's run continues the reply too
    assert reply_of(chat(app_for(tmp_path, engine)), False)[0] == LEAD + " 42."
    assert engine.asked == [(24, END), None, None] and engine.continued == [False, True, True]
    for app_of, fields in ((lambda e: app_for(tmp_path, e), {"chat_template_kwargs": {"enable_thinking": False}}),
                           (lambda e: app_for(tmp_path, e, reserve_=(0.0, 1024)), {})):
        engine = AtTheCut()                                    # no thinking, or no reserve and no budget: no cut
        reply_of(chat(app_of(engine), **fields), False)
        assert engine.asked == [None] and engine.continued == [False]


class WantedOnce(Turns):
    """Turns where a foreground request arrives once, at the ``at``-th round a background reply asks."""

    def __init__(self, at):
        super().__init__()
        self.asked, self.at = 0, at

    def wanted(self):
        self.asked += 1
        return self.asked == self.at


@pytest.mark.parametrize("width", [1, 3])
def test_a_background_reply_that_yields_replays_to_the_same_cut(tmp_path, width):
    """A background reply yields after its second round and replays from its prompt: the replay's stop counts the
    tokens it decodes again, so both engines give the reply the request has without a yield."""

    replies = []
    for engine in (ToTheEnd(after_close=text("Hi there.") + [EOS], width=width),
                   AtTheCut(after_close=text("Hi there.") + [EOS], width=width)):
        app = app_for(tmp_path, engine)
        app.turns = WantedOnce(2)
        replies.append(same_reply(chat(app, priority="background"), False))
        assert [n for _, n in engine.runs] == [48, 48, 48 - 27]      # the replay runs the prompt again
        assert app.turns.asked > 2                                 # the replay asked again and went on
    assert replies[0] == replies[1] and replies[0][0] == "Hi there." and replies[0][6] == 23
    assert engine.asked == [(24, END), (24, END), None] and engine.continued == [False, False, True]


@pytest.mark.parametrize("seed", range(8))
def test_think_stop_ends_the_round_the_budget_cuts(seed):
    """ThinkStop fires in the round where ThinkBudget cuts, whatever the round widths, and never where it does not."""

    rng = random.Random(seed)
    for _ in range(300):
        tokens = [rng.choice((1, 2, 3)) for _ in range(rng.randint(1, 60))]
        if rng.random() < 0.6:
            tokens[rng.randrange(len(tokens))] = END
        budget, width = rng.randint(1, 70), rng.randint(1, 5)
        gate, stop = ThinkBudget(budget, SHUT, END), ThinkStop(budget, END)
        cut = ended = None
        for r, at in enumerate(range(0, len(tokens), width)):
            new = tokens[at:at + width]
            if cut is None and gate.cut(new) is not None:
                cut = r
            for token in new:
                gate.observe(token)
            if stop.hit(new):
                ended = r
                break
        assert ended == cut, (tokens, budget, width)


def test_a_continuation_keeps_the_prompts_snapshot():
    """bailing_v3 with room for one snapshot (TF_LING_CACHE_GIB=0): the prompt's stays through its cut's continuation,
    so the next request on that prompt resumes from it; with room for both, both stay; a later prompt is the newest."""

    from tensorfold.families.bailing_v3.cuda.engine import BailingEngine

    def snap(n):
        return SimpleNamespace(ids=list(range(n)), rows=None, nbytes=0)

    for room, kept in ((1, [100]), (2, [140, 100])):
        e = object.__new__(BailingEngine)
        e.cache, e.cache_entries, e.cache_bytes = [], 8, 10 * room
        e._held_bytes = lambda e=e: 10 * len(e.cache)
        e._remember(snap(100))                                 # the prompt
        e._remember(snap(140), behind=True)                    # its cut's continuation: the prompt, reply and close
        assert [len(c.ids) for c in e.cache] == kept
        assert e._resume(list(range(101)), []) is e.cache[-1]   # the prompt again resumes from its own snapshot
    e._remember(snap(180))
    assert [len(c.ids) for c in e.cache] == [100, 180]


# -- the doors ---------------------------------------------------------------------------------------------------

def test_every_door_closes_at_the_same_place(tmp_path):
    runs = {}
    for name, stream, route, fields in [
        ("chat", False, "/v1/chat/completions", {"max_tokens": 48}),
        ("chat-stream", True, "/v1/chat/completions", {"max_tokens": 48}),
        ("messages", False, "/v1/messages", {"model": "m", "max_tokens": 48, "messages": [{"role": "user",
                                                                                          "content": "Hi"}],
                                             "thinking": {"type": "enabled", "budget_tokens": 40}}),
        ("messages-adaptive", True, "/v1/messages", {"model": "m", "max_tokens": 48,
                                                     "messages": [{"role": "user", "content": "Hi"}],
                                                     "thinking": {"type": "adaptive"}}),
        ("responses", False, "/v1/responses", {"model": "m", "input": "Hi", "max_output_tokens": 48}),
        ("responses-stream", True, "/v1/responses", {"model": "m", "input": "Hi", "max_output_tokens": 48}),
    ]:
        engine = Script(after_close=(EOS,))
        app = app_for(tmp_path, engine, max_tokens=4096)
        raw = chat(app, stream, route, **fields)
        runs[name] = engine.runs
        if route == "/v1/messages":
            body = json.loads(raw) if not stream else [e for e in stream_events(raw) if e["type"] == "message_delta"][-1]
            details = body["usage"]["output_tokens_details"]
            assert details == {"thinking_tokens": 25, "thinking_cut": True}
            if not stream:
                assert [b["text"] for b in body["content"] if b["type"] == "text"] == [LEAD + " 42."]
        elif route == "/v1/responses":
            body = json.loads(raw) if not stream else [e for e in stream_events(raw)
                                                       if e["type"] == "response.completed"][-1]["response"]
            assert body["usage"]["output_tokens_details"] == {"reasoning_tokens": 25, "reasoning_cut": True}
            said = [c["text"] for item in body["output"] if item["type"] == "message" for c in item["content"]]
            assert said == [LEAD + " 42."]
        else:
            content, _, finish, usage, _, _ = reply_of(raw, stream)
            assert content == LEAD + " 42." and finish == "stop" and usage["completion_tokens_details"][
                "reasoning_cut"]
    first = runs["chat"]
    assert [n for _, n in first] == [48, 21, 14]
    assert all(r == first for r in runs.values())


def test_messages_budget_tokens_below_the_cap_wins(tmp_path):
    engine = Script(after_close=text("ok") + [EOS])
    app = app_for(tmp_path, engine, max_tokens=4096)
    chat(app, False, "/v1/messages", model="m", max_tokens=48, messages=[{"role": "user", "content": "Hi"}],
         thinking={"type": "enabled", "budget_tokens": 6})
    assert engine.runs[1][0][len(engine.runs[0][0]):] == LETTERS[:5] + SHUT


# -- tools after the cut -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("stream", [False, True])
def test_a_tool_call_after_the_cut(tmp_path, stream):
    engine = Script(after_close=[OPEN, *text(CALL), CALL_END, EOS])
    raw = chat(app_for(tmp_path, engine, max_tokens=256), stream, tools=TOOLS, tool_choice="auto")
    content, _, finish, usage, stats, calls = reply_of(raw, stream)
    assert finish == "tool_calls" and calls == ["get_weather"]
    assert stats["think_cut"] == 127 and "answer_lead" not in stats and not content.strip()
    assert usage["completion_tokens_details"]["reasoning_cut"] is True


def test_a_required_call_after_the_cut_opens_the_call_not_the_lead(tmp_path):
    engine = Script(after_close=(EOS,))
    raw = chat(app_for(tmp_path, engine, max_tokens=256), False, tools=TOOLS, tool_choice="required")
    content, _, finish, _, stats, calls = reply_of(raw, False)
    assert finish == "tool_calls" and calls == ["get_weather"]
    assert "answer_lead" not in stats and OPEN in engine.runs[2][0] and not Tokens().decode(
        engine.runs[2][0]).endswith(LEAD)


# -- the lane server ---------------------------------------------------------------------------------------------

class Stream(LaneStream):
    """A LaneStream driven as lane_family drives it: a round's committed tokens meet the budget's cut, then the lead."""

    def round(self, drawn):
        committed = list(self.force) + list(drawn)
        self.force = []
        cut = self.think_cut(committed)
        if cut is not None:
            committed = [*committed[:cut], self.start_close()]
        elif (hit := self.answer_fix(committed)) is not None:
            committed = [*committed[:hit[0]], hit[1][0]]
        return self.commit(committed)


def lane_stream(budget, eos=(EOS,), reserve_on=True):
    lead = AnswerLead(text(LEAD), eos, lambda t: not chr(t).strip() if t > 0 else False, END) if reserve_on else None
    return Stream(stream_id="s", prompt_ids=[THINK], max_new_tokens=24, eos_ids=frozenset(eos), think_budget=budget,
                  think_close=tuple(SHUT), think_end=END, think_open=True, answer_lead=lead)


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_the_lane_stream_leads_an_empty_answer_after_its_close(width):
    stream = lane_stream(12)
    script = LETTERS[:11] + [ord("z")]                                 # the 12th is cut
    tape = iter(script)
    while not stream.finished:
        if stream.think_open:
            drawn = [next(tape) for _ in range(min(width, 12 - len(stream.emitted)))]
        elif stream.force:
            drawn = []
        else:
            drawn = [EOS]                                                # the model would stop: no answer
        stream.round(drawn)
    out = stream.emitted
    assert out[:15] == LETTERS[:11] + SHUT and stream.closed_at == 11
    assert Tokens().decode(out[15:]) == LEAD and stream.answer_lead.at == 15 and stream.finish_reason == "stop"


def test_the_lane_stream_finds_the_end_beside_the_close_in_one_round():
    stream = lane_stream(3)
    stream.round([ord("a"), ord("b"), ord("c")])                       # the 3rd becomes the close's newline
    assert stream.emitted == [ord("a"), ord("b"), NL] and stream.force == [END, NL, NL]
    stream.round([EOS])                                                # forced END, NL, NL then a drawn end
    assert stream.emitted[3:6] == [END, NL, NL] and stream.emitted[6] == text(LEAD)[0]
    assert not stream.finished and stream.answer_lead.at == 6


def test_the_lane_stream_without_a_reserve_ends_empty():
    stream = lane_stream(3, reserve_on=False)
    stream.round([ord("a"), ord("b"), ord("c")])
    stream.round([EOS])
    assert stream.finished and stream.emitted == [ord("a"), ord("b"), NL, END, NL, NL, EOS]


def test_the_lane_app_holds_the_budget_to_the_reserve():
    from tests.test_ignore_eos import CommitEngine
    from tests.test_lane_server import FakeTokenizer, make_app
    from tests.http_fakes import post

    class ThinkTokenizer(FakeTokenizer):
        unk_token_id = None

        def convert_tokens_to_ids(self, token):
            return 90 if token == "</think>" else None

    for reserve_, asked, max_tokens, want in [(ON, None, 8, 4), (ON, 2, 8, 2), (ON, 7, 8, 4), (ON, None, 128, 64),
                                              ((0.0, 1024), None, 8, 0), ((0.0, 1024), 7, 8, 7)]:
        app = make_app(engine_factory=CommitEngine, tokenizer=ThinkTokenizer(), enable_thinking=True,
                       answer_reserve=reserve_)
        jobs = []
        submit = app.scheduler.submit
        app.scheduler.submit = lambda job, submit=submit: (jobs.append(job), submit(job))[1]
        app.accepts_cancellation = False
        try:
            fields = {} if asked is None else {"thinking_budget": asked}
            status, body = post(app, {"messages": [{"role": "user", "content": "x"}], "max_tokens": max_tokens,
                                      **fields}, "/v1/chat/completions")
            assert status == 200, body
            assert jobs[0].think_budget == want
            assert (jobs[0].answer_lead is not None) == (want > 0 and reserve_[0] > 0)
        finally:
            app.close()
