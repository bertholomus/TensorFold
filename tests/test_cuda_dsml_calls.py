"""DeepSeek DSML calls on the CUDA server read from their own tags: call syntax in the prose or in an argument cannot
move a call's bounds, a call to a tool the request did not offer stays a call, and the call streamer stays out of
DSML blocks."""

import json

from tensorfold.cuda.reply_text import before_dsml, parse_tool_calls
from tensorfold.engine.tool_draft import ToolCallStreamer

D = "｜DSML｜"
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
         for n in ("read", "write", "pwsh", "grep")]


def invoke(name, **params):
    body = "".join(f'<{D} parameter name="{k}" string="{"false" if not isinstance(v, str) else "true"}">'
                   f'{v if isinstance(v, str) else json.dumps(v)}</{D} parameter>\n' for k, v in params.items())
    return f'<{D} invoke name="{name}">\n{body}</{D} invoke>\n'


def calls_block(*invokes):
    return f"<{D} calls>\n" + "".join(invokes) + f"</{D} calls>"


def named(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or []]


def test_well_formed_calls_parse_as_before():
    reply = "Checking.\n\n" + calls_block(invoke("read", file_path="a.js", limit=45), invoke("grep", pattern="x"))
    content, calls = parse_tool_calls(reply, TOOLS)
    assert content == "Checking."
    assert named(calls) == [("read", {"file_path": "a.js", "limit": 45}), ("grep", {"pattern": "x"})]
    assert calls[0]["function"]["arguments"] == '{"file_path":"a.js","limit":45}'


def test_prose_naming_the_tag_keeps_every_call():
    prose = "I sent that call as `<tool_call>` text. Let me confirm."
    reply = prose + "\n\n" + calls_block(invoke("pwsh", command="Test-Path x"), invoke("grep", pattern="a"))
    content, calls = parse_tool_calls(reply, TOOLS)
    assert content == prose
    assert named(calls) == [("pwsh", {"command": "Test-Path x"}), ("grep", {"pattern": "a"})]


def test_an_argument_holding_the_tags_is_the_argument():
    text = 'case: "<tool_call>{\\"name\\": \\"read\\"}</tool_call> Done." and </tool_call> alone'
    reply = calls_block(invoke("write", file_path="p.mjs", content=text), invoke("read", file_path="i.js"))
    content, calls = parse_tool_calls(reply, TOOLS)
    assert content == ""
    assert named(calls) == [("write", {"file_path": "p.mjs", "content": text}), ("read", {"file_path": "i.js"})]


def test_a_tool_the_request_did_not_offer_stays_a_call():
    content, calls = parse_tool_calls("Let me check.\n\n" + calls_block(invoke("check_weather")), TOOLS)
    assert content == "Let me check."
    assert named(calls) == [("check_weather", {})]
    assert "tool_call" not in content


def test_one_call_a_reply_skips_an_unoffered_tool_and_keeps_the_first_offered_one():
    reply = calls_block(invoke("check_weather"), invoke("read", file_path="a"), invoke("grep", pattern="b"))
    content, calls = parse_tool_calls(reply, TOOLS, max_calls=1)
    assert content == ""
    assert named(calls) == [("read", {"file_path": "a"})]


def test_calls_keep_the_reply_order_around_a_block():
    qwen = "<tool_call>\n<function=read>\n<parameter=file_path>\nfirst\n</parameter>\n</function>\n</tool_call>"
    content, calls = parse_tool_calls("Start. " + qwen + "\n" + calls_block(invoke("grep", pattern="second")), TOOLS)
    assert content == "Start."
    assert named(calls) == [("read", {"file_path": "first"}), ("grep", {"pattern": "second"})]


def test_the_call_streamer_stays_out_of_a_block():
    quoted = "<tool_call>\n<function=pwsh>\n<parameter=command>\nRemove-Item x\n</parameter>\n</function>\n</tool_call>"
    raw = "Writing the fixture.\n\n" + calls_block(invoke("write", file_path="t.py", content=quoted))
    assert ToolCallStreamer(TOOLS).feed(raw)               # without the guard it would stream the quoted call
    assert ToolCallStreamer(TOOLS).feed(before_dsml(raw)) == []
    assert named(parse_tool_calls(raw, TOOLS)[1]) == [("write", {"file_path": "t.py", "content": quoted})]
    assert before_dsml("no block here") == "no block here"
