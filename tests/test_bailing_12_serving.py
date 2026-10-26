"""Ling-3.1-flash on the CUDA server, everything short of the engine: the checkpoint's template, its replies and
the routes.

- the five golden renders (Hugging Face's apply_chat_template over the checkpoint's template, tests/fixtures/ling31)
  render the same through the server's ChatTemplate, and an OpenAI-style history renders as HF's dict form;
- with the checkpoint's tokenizer (the bank or TF_LING31_MODEL, when present) every render encodes to the golden ids;
- Ling's replies split and parse as vLLM's ling3 parsers read them: reasoning up to ``</think>``, a finished call
  before it ends the reasoning, calls in the template's layout and the system prompt's, typed values, unknown tools
  left as text; a parsed call renders back into the history exactly as the model wrote it;
- reasoning efforts: none turns thinking off, every other word turns it on, and the template hears no effort;
- LingApp over HTTP, streamed and not: reasoning_content, content, tool_calls, the request's policy and stop-at-EOS
  on the engine, the context refusal; a required call's gate learns Ling's markup;
- cuda_engine's refusals and policies, and CUDA_APP.

Run: python -m pytest tests/test_bailing_12_serving.py -q
CPU only (no torch).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("tokenizers")

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.cuda.reply_text import parse_tool_calls
from tensorfold.server.errors import CONTEXT_LIMIT, RequestError
from tensorfold.server.text import split_thinking
from tests.test_cuda_admission import http_server, post
from tests.test_request_policy import stream_calls

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ling31"
GOLDEN = [json.loads(line) for line in (FIXTURES / "golden_renders.jsonl").read_text().splitlines()]
BANK = Path(os.environ.get("TF_LING31_MODEL") or os.environ.get("LING31_FP8_DIR", "/models/Ling-3.1-flash"))
on_bank = pytest.mark.skipif(not (BANK / "tokenizer.json").is_file(),
                             reason="Ling-3.1-flash's tokenizer is not on this host (set TF_LING31_MODEL)")

# the checkpoint's added tokens the template and the parsers use (tokenizer_config.json added_tokens_decoder)
TOKENS = {"<|startoftext|>": 156891, "<|endoftext|>": 156892, "<|role_end|>": 156895, "<tool_call>": 156896,
          "</tool_call>": 156897, "<tool_response>": 156898, "</tool_response>": 156899, "<think>": 156903,
          "</think>": 156904, "<arg_key>": 156905, "</arg_key>": 156906, "<arg_value>": 156907,
          "</arg_value>": 156908, "<role>": 157151, "</role>": 157152}
SPECIAL = {156891, 156892, 156895, 157151, 157152}      # special=true; the think and call marks are plain added tokens
EOS = 156895
SERVED = "Ling-3.1-flash-EXL3"

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}
BOOK_TOOL = {"type": "function", "function": {
    "name": "book_table", "description": "Book a restaurant table",
    "parameters": {"type": "object", "properties": {"guests": {"type": "integer"}, "outdoor": {"type": "boolean"},
                                                    "when": {"type": "object"},
                                                    "dishes": {"type": "array", "items": {"type": "string"}},
                                                    "note": {"type": "string"}},
                   "required": ["guests"]}}}
TOOLS = [WEATHER_TOOL, BOOK_TOOL]

# calls in the template's layout (a newline after the name and after each key, none between pairs)
WEATHER_CALL = ("<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Tokyo</arg_value>"
                "<arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n</tool_call>")
TOKYO = {"city": "Tokyo", "unit": "celsius"}
BOOK_CALL = ("<tool_call>book_table\n<arg_key>guests</arg_key>\n<arg_value>4</arg_value>"
             "<arg_key>outdoor</arg_key>\n<arg_value>true</arg_value>"
             '<arg_key>when</arg_key>\n<arg_value>{"day": "Friday", "hour": 19}</arg_value>'
             '<arg_key>dishes</arg_key>\n<arg_value>["paella", "flan"]</arg_value>'
             "<arg_key>note</arg_key>\n<arg_value>window seat\nnear the door</arg_value>\n</tool_call>")
BOOKED = {"guests": 4, "outdoor": True, "when": {"day": "Friday", "hour": 19}, "dishes": ["paella", "flan"],
          "note": "window seat\nnear the door"}
UNKNOWN_CALL = "<tool_call>launch_rocket\n<arg_key>when</arg_key>\n<arg_value>now</arg_value>\n</tool_call>"


class LingChars:
    """Ling's added tokens at their ids, every other character its code point (the tests' text stays far below
    156891), with the ``tokenizers.Tokenizer`` calls the server makes."""

    def encode(self, text, add_special_tokens=False):
        ids, at = [], 0
        while at < len(text):
            mark = next((m for m in TOKENS if text.startswith(m, at)), None)
            ids.append(TOKENS[mark] if mark else ord(text[at]))
            at += len(mark) if mark else 1
        return SimpleNamespace(ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        names = {i: m for m, i in TOKENS.items()}
        return "".join(names.get(i, chr(i)) for i in ids if not (skip_special_tokens and i in SPECIAL))

    def token_to_id(self, token):
        return TOKENS.get(token)

    def id_to_token(self, i):
        return {i: m for m, i in TOKENS.items()}.get(i, chr(i))

    def get_vocab_size(self, with_added_tokens=True):
        return 157_184


class Engine:
    """Writes a scripted reply one token a call (EOS last) and records what each request handed it."""

    eos = (EOS,)

    def __init__(self, limit=262_144, largest=None):
        self.tokens: list[int] = []
        self.calls: list[dict] = []
        self.request = threading.local()          # the app sets the request's policy and stop-at-EOS here
        self.limit = limit
        self.capacity_plan = {"largest_window": largest}

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        self.calls.append({"prompt": list(prompt), "policy": getattr(self.request, "policy", "unset"),
                           "stop_eos": getattr(self.request, "stop_eos", "unset")})
        for token in self.tokens[:max_tokens]:
            if on_tokens([token]):
                break
        return {"generated": len(self.tokens)}


def render(messages, *, tools=None, thinking=True):
    return ChatTemplate(FIXTURES).render(messages, tools=tools, enable_thinking=thinking)


# -- the template ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("record", GOLDEN, ids=[r["name"] for r in GOLDEN])
def test_the_server_renders_the_golden_text(record):
    tools = [WEATHER_TOOL] if record["name"] == "one-tool-call" else None
    assert render(record["messages"], tools=tools, thinking=record["name"] != "thinking-off") == record["rendered_text"]


def test_an_openai_history_renders_as_the_dict_form():
    # clients send arguments as a JSON string, null content beside calls, ids, and the reasoning back as
    # reasoning_content; the server's normalization renders that as Hugging Face's dict form
    record = next(r for r in GOLDEN if r["name"] == "one-tool-call")
    user, assistant, tool = record["messages"]
    call = assistant["tool_calls"][0]
    openai = [user,
              {"role": "assistant", "content": None, "tool_calls": [
                  {"id": "call_0", "type": "function",
                   "function": {"name": call["function"]["name"], "arguments": json.dumps(call["function"]["arguments"])}}]},
              {"role": "tool", "tool_call_id": "call_0", "content": tool["content"]}]
    assert render(openai, tools=[WEATHER_TOOL]) == record["rendered_text"]
    turns = next(r for r in GOLDEN if r["name"] == "multi-turn")
    folded = [dict(m) for m in turns["messages"]]
    folded[1] = {"role": "assistant",
                 "content": "<think>" + folded[1]["reasoning_content"] + "</think>" + folded[1]["content"]}
    assert render(folded) == turns["rendered_text"]                 # a think block left in content splits out


def test_the_fixture_is_the_checkpoint_template():
    if not (BANK / "chat_template.jinja").is_file():
        pytest.skip("the banked checkpoint is not on this host")
    assert (BANK / "chat_template.jinja").read_bytes() == (FIXTURES / "chat_template.jinja").read_bytes()


@on_bank
def test_the_checkpoint_tokenizer_encodes_the_golden_ids():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(BANK / "tokenizer.json"))
    for record in GOLDEN:
        ids = tok.encode(record["rendered_text"], add_special_tokens=False).ids
        assert ids == record["input_ids"], record["name"]
        assert tok.decode(ids, skip_special_tokens=False) == record["rendered_text"]
    assert {mark: tok.token_to_id(mark) for mark in TOKENS} == TOKENS


# -- replies ----------------------------------------------------------------------------------------------------------

REPLIES = [
    # (id, generated text, thinking, reasoning, content, calls)
    ("answer", "The user wants French.</think>Bonjour !", True, "The user wants French.", "Bonjour !", []),
    ("call-after-reasoning", "Tokyo needs the tool.</think>" + WEATHER_CALL, True, "Tokyo needs the tool.", "",
     [("get_weather", TOKYO)]),
    ("call-ends-reasoning", "Tokyo needs the tool." + WEATHER_CALL, True, "Tokyo needs the tool.", "",
     [("get_weather", TOKYO)]),
    ("prompt-layout", WEATHER_CALL.replace("</arg_value><arg_key>", "</arg_value>\n<arg_key>"), False, "", "",
     [("get_weather", TOKYO)]),
    ("two-calls", WEATHER_CALL + "\n" + WEATHER_CALL.replace("Tokyo", "Lima"), False, "", "",
     [("get_weather", TOKYO), ("get_weather", {"city": "Lima", "unit": "celsius"})]),
    ("text-then-call", "Checking now.\n" + WEATHER_CALL, False, "", "Checking now.", [("get_weather", TOKYO)]),
    ("typed-values", BOOK_CALL, False, "", "", [("book_table", BOOKED)]),
    ("unknown-tool", UNKNOWN_CALL, False, "", UNKNOWN_CALL, []),
]


def calls_of(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or []]


@pytest.mark.parametrize("text,thinking,reasoning,content,calls", [r[1:] for r in REPLIES], ids=[r[0] for r in REPLIES])
def test_a_reply_splits_and_parses_as_ling_writes_it(text, thinking, reasoning, content, calls):
    # the server's own two steps (cuda.server.App.run): the think split, then the calls in what follows it
    got_reasoning, answer = split_thinking(text, finished=True) if thinking else ("", text)
    got_content, got_calls = parse_tool_calls(answer, TOOLS)
    assert (got_reasoning, got_content, calls_of(got_calls)) == (reasoning, content, calls)
    if thinking:                                   # while it streams, nothing sent is taken back
        for end in range(len(text) + 1):
            part_reasoning, part_answer = split_thinking(text[:end], finished=False)
            assert reasoning.startswith(part_reasoning) and answer.startswith(part_answer), end


def test_a_parsed_call_renders_back_as_the_model_wrote_it():
    text = "Four guests on Friday." + "</think>" + BOOK_CALL
    reasoning, answer = split_thinking(text, finished=True)
    content, calls = parse_tool_calls(answer, TOOLS)
    history = [{"role": "user", "content": "Book a table for four."},
               {"role": "assistant", "content": content or None, "reasoning_content": reasoning, "tool_calls": calls},
               {"role": "tool", "tool_call_id": calls[0]["id"], "content": "booked"}]
    assert "<role>ASSISTANT</role>\n<think>" + text + "<|role_end|><role>OBSERVATION</role>" in render(
        history, tools=TOOLS)


# -- the app ----------------------------------------------------------------------------------------------------------

@pytest.fixture(params=["chars", "checkpoint"])
def make_app(request, tmp_path):
    """``make_app(default_thinking=True, **engine)``: LingApp over the fixture template, with LingChars or (when the
    bank is on this host) the checkpoint's tokenizer."""

    if request.param == "checkpoint" and not (BANK / "tokenizer.json").is_file():
        pytest.skip("Ling-3.1-flash's tokenizer is not on this host (set TF_LING31_MODEL)")
    from tokenizers import Tokenizer, models

    from tensorfold.families.bailing_v3.cuda.app import LingApp

    def make(default_thinking=True, **engine_args):
        model = tmp_path / SERVED
        model.mkdir(exist_ok=True)
        for name in ("chat_template.jinja", "tokenizer_config.json"):
            shutil.copy(FIXTURES / name, model / name)
        if request.param == "checkpoint":
            shutil.copy(BANK / "tokenizer.json", model / "tokenizer.json")
        else:
            Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(str(model / "tokenizer.json"))
        app = LingApp(Engine(**engine_args), model, SERVED, default_thinking=default_thinking,
                      sampling={"temperature": 0.0})
        if request.param == "chars":
            app.tok = LingChars()
        return app

    return make


def script(app, text):
    app.engine.tokens = list(app.tok.encode(text, add_special_tokens=False).ids) + [EOS]


def ask(app, body, stream):
    """(status, message, finish_reason) of a chat request, the streamed deltas joined into one message."""

    with http_server(app) as port:
        status, payload = post(port, {"model": SERVED, "temperature": 0, **body, "stream": stream}, True)
    if status != 200 or not stream:
        reply = json.loads(payload)
        return (status, reply, None) if status != 200 else (status, reply["choices"][0]["message"],
                                                            reply["choices"][0]["finish_reason"])
    message, finish = {"content": "", "reasoning_content": ""}, None
    for line in payload.splitlines():
        if line.startswith("data:") and line != "data: [DONE]":
            for choice in json.loads(line[5:]).get("choices", []):
                for key in ("content", "reasoning_content"):
                    message[key] += choice["delta"].get(key) or ""
                finish = choice.get("finish_reason") or finish
    calls = stream_calls(payload)
    if calls:
        message["tool_calls"] = [{"function": calls[i]} for i in sorted(calls)]
    return status, message, finish


USER = [{"role": "user", "content": "Say hello in French."}]
WEATHER_ASK = [{"role": "user", "content": "What's the weather in Tokyo?"}]


@pytest.mark.parametrize("stream", [False, True])
def test_a_thinking_reply_over_http(make_app, stream):
    app = make_app()
    script(app, "The user wants French.</think>Bonjour !")
    status, message, finish = ask(app, {"messages": USER}, stream)
    assert status == 200 and finish == "stop"
    assert (message["reasoning_content"], message["content"]) == ("The user wants French.", "Bonjour !")
    prompt = app.tok.decode(app.engine.calls[0]["prompt"], skip_special_tokens=False)
    assert prompt == render(USER)                  # the checkpoint's template, thinking on by default
    assert app.engine.calls[0]["stop_eos"] is True and app.engine.calls[0]["policy"] is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reply", ["Tokyo needs the tool.</think>" + WEATHER_CALL, "Tokyo needs the tool." + WEATHER_CALL],
                         ids=["after-reasoning", "ends-reasoning"])
def test_a_tool_call_over_http(make_app, stream, reply):
    app = make_app()
    script(app, reply)
    status, message, finish = ask(app, {"messages": WEATHER_ASK, "tools": TOOLS}, stream)
    assert status == 200 and finish == "tool_calls"
    assert message["reasoning_content"] == "Tokyo needs the tool." and not message["content"]
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in message["tool_calls"]] == [
        ("get_weather", TOKYO)]


@pytest.mark.parametrize("stream", [False, True])
def test_thinking_off_with_two_calls_over_http(make_app, stream):
    app = make_app()
    script(app, WEATHER_CALL + "\n" + BOOK_CALL)
    body = {"messages": WEATHER_ASK, "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}}
    status, message, finish = ask(app, body, stream)
    assert status == 200 and finish == "tool_calls" and not message.get("reasoning_content")
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in message["tool_calls"]] == [
        ("get_weather", TOKYO), ("book_table", BOOKED)]
    prompt = app.tok.decode(app.engine.calls[0]["prompt"], skip_special_tokens=False)
    assert prompt == render(WEATHER_ASK, tools=TOOLS, thinking=False)
    assert prompt.endswith("<role>ASSISTANT</role>\n<think></think>")


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
def test_every_effort_but_none_thinks_and_the_template_hears_none(make_app, effort):
    app = make_app(default_thinking=False)
    prepared = app._prepare({"messages": USER, "reasoning_effort": effort}, True)
    assert prepared.thinking
    assert app.tok.decode(prepared.prompt, skip_special_tokens=False) == render(USER)


@pytest.mark.parametrize("body", [{"reasoning_effort": "none"}, {"chat_template_kwargs": {"enable_thinking": False}},
                                  {"chat_template_kwargs": {"thinking": {"type": "disabled"}}}, {}],
                         ids=["effort-none", "enable_thinking", "thinking-switch", "server-default"])
def test_thinking_off(make_app, body):
    app = make_app(default_thinking=bool(body))        # the last case: a server started with --no-thinking
    prepared = app._prepare({"messages": USER, **body}, True)
    assert not prepared.thinking
    text = app.tok.decode(prepared.prompt, skip_special_tokens=False)
    assert text == render(USER, thinking=False) and "detailed thinking off" in text


def test_off_is_not_an_effort(make_app):
    app = make_app()
    with pytest.raises(RequestError, match="reasoning_effort must be none"):
        app._prepare({"messages": USER, "reasoning_effort": "off"}, True)


def test_the_request_policy_and_ignore_eos_reach_the_engine(make_app):
    app = make_app()
    script(app, "Short.</think>Oui.")
    ask(app, {"messages": USER, "model": SERVED + "@0"}, False)
    ask(app, {"messages": USER, "tf_policy": "a:0.7:0.9", "ignore_eos": True}, False)
    assert [(c["policy"], c["stop_eos"]) for c in app.engine.calls] == [("0", True), ("a:0.7:0.9", False)]


@pytest.mark.parametrize("stream", [False, True])
def test_a_prompt_past_the_window_is_refused_before_the_engine(make_app, stream):
    app = make_app(limit=40, largest=100_000)
    status, reply, _ = ask(app, {"messages": USER, "max_tokens": 64}, stream)
    assert status == 400 and app.engine.calls == []
    error = reply["error"]
    assert error["code"] == "context_length_exceeded" and error["message"].startswith(f"{CONTEXT_LIMIT} 40 tokens")
    assert "restart every rank with --context" in error["message"]


def test_a_required_call_opens_after_the_think_block(make_app):
    app = make_app()
    if not isinstance(app.tok, LingChars):
        pytest.skip("the gate's markup comes from the template; one tokenizer is enough")
    prompt = app._prepare({"messages": WEATHER_ASK, "tools": TOOLS, "tool_choice": "required"}, True).prompt
    gate = app._call_gate(prompt, TOOLS)
    assert (gate.lead, gate.tail, gate.names) == ("", "\n", ["get_weather", "book_table"])
    ids = app.tok.encode("Let me check.").ids
    assert gate.cut(ids) is None                   # the prompt left <think> open: reasoning runs free
    for token in [*ids, TOKENS["</think>"]]:
        gate.observe(token)
    assert gate.cut(app.tok.encode("Sure").ids) == (0, [TOKENS["<tool_call>"]])     # prose becomes the call's opener
    gate.observe(TOKENS["<tool_call>"])
    assert gate.cut(app.tok.encode("weather").ids) == (0, app.tok.encode("get_weather\n").ids)   # then a tool's name


# -- the engine hook ----------------------------------------------------------------------------------------------------

@pytest.fixture
def built(monkeypatch):
    from tensorfold.families.bailing_v3.cuda import engine as ling_engine

    made: list[dict] = []
    monkeypatch.setattr(ling_engine, "BailingEngine", lambda *a, **k: made.append(k) or SimpleNamespace(**k))
    monkeypatch.delenv("TF_TP_WORLD", raising=False)
    return made


def test_cuda_engine_builds_the_ranks_policies(built, tmp_path):
    from tensorfold.families import bailing_v3

    ranks = dict(tp=4, rank=3, master="10.0.0.1", master_port=29600)
    assert bailing_v3.cuda_engine(tmp_path, **ranks, context=131_072, context_explicit=True).policy == "auto"
    assert built[-1] == {"rank": 3, "master": "10.0.0.1", "port": 29600, "policy": "auto", "context": 131_072,
                         "context_explicit": True, "serial_only": False}
    assert os.environ["TF_TP_WORLD"] == "4"
    assert bailing_v3.cuda_engine(tmp_path, tp=2, master="m", mtp_drafts=0).policy == "0"
    assert bailing_v3.cuda_engine(tmp_path, tp=2, master="m", mtp_drafts=3).policy == "3"
    assert bailing_v3.cuda_engine(tmp_path, tp=2, master="m", no_drafts=True, drafter="").serial_only is True
    assert bailing_v3.cuda_engine(tmp_path, tp=2, master="m", no_drafts=True, drafter="/d").serial_only is True


@pytest.mark.parametrize("options,message", [
    ({"tp": 1}, "splits over 2 or 4 GPUs, not 1"),
    ({"tp": 6, "master": "m"}, "splits over 2 or 4 GPUs, not 6"),
    ({"tp": 2}, "--tp 2 needs --master"),
    ({"tp": 2, "master": "m", "drafter": "/drafts/dspark"}, "drafts with its own MTP layer"),
    ({"tp": 2, "master": "m", "parallel": 2}, "one request at a time"),
    ({"tp": 2, "master": "m", "mtp_drafts": 8}, "0 to 7 drafts"),
])
def test_cuda_engine_refuses_before_loading(built, tmp_path, options, message):
    from tensorfold.families import bailing_v3

    with pytest.raises(ValueError, match=message):
        bailing_v3.cuda_engine(tmp_path, **options)
    assert built == []


def test_the_serve_gates_and_the_app_class():
    import argparse

    from tensorfold import serve_options
    from tensorfold.families import bailing_v3
    from tensorfold.families.bailing_v3.cuda.app import LingApp

    assert bailing_v3.CUDA_APP is LingApp and bailing_v3.DRAFTER == "" and not hasattr(bailing_v3, "load")
    family = SimpleNamespace(title=bailing_v3.TITLE, model_type="bailing_hybrid", package=bailing_v3)
    for flags, message in (({"kv_dtype": "int8"}, "bf16 KV cache, not --kv-dtype int8"),
                           ({"mtp_confidence": 0.6}, "has no such rule"),
                           ({"decode_share": 0.25}, "--decode-share sets"),
                           ({"prefill_fp8": True}, "has none")):
        with pytest.raises(ValueError, match=message):
            serve_options.check(argparse.Namespace(**flags), family, "cuda")
