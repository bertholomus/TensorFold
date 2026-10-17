"""With TF_SSE_KEEPALIVE_S set, a streamed reply that goes quiet gets SSE comments, which clients skip; a busy one gets
none. Unset (the default) or 0, it is off."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda import http as cuda_http
from tests.test_cuda_server_disconnect import MESSAGES, WAIT, PacedEngine, app_for, serving

BODY = {"messages": MESSAGES, "max_tokens": 12, "stream": True}


def streamed(port, engine, quiet: float) -> list[bytes]:
    """The stream's blocks for BODY, the engine held ``quiet`` seconds before its first token (when it holds)."""

    data = json.dumps(BODY).encode()
    conn = socket.create_connection(("127.0.0.1", port), timeout=WAIT)
    conn.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                 + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    if engine.hold_at is not None:
        assert engine.held.wait(WAIT)
        time.sleep(quiet)
        engine.release.set()
    raw = b""
    while chunk := conn.recv(65536):
        raw += chunk
    conn.close()
    head, body = raw.split(b"\r\n\r\n", 1)
    assert b" 200 " in head.split(b"\r\n", 1)[0]
    return [block for block in body.split(b"\n\n") if block]


def first_content(blocks: list[bytes]) -> int:
    for i, block in enumerate(blocks):
        if block.startswith(b"data: {") and json.loads(block[6:])["choices"][0]["delta"].get("content"):
            return i
    raise AssertionError("no content delta")


def test_a_long_prompt_read_sends_comments_until_the_first_token(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_http, "KEEPALIVE", 0.1)
    engine = PacedEngine("hello", hold_at=0)             # held before its first token, as a long prompt read is
    with serving(app_for(tmp_path, engine)) as port:
        blocks = streamed(port, engine, quiet=0.55)
    comments = [i for i, block in enumerate(blocks) if block.startswith(b":")]
    assert len(comments) >= 3, blocks[:8]
    assert all(blocks[i] == b": keepalive" for i in comments)
    assert max(comments) < first_content(blocks)          # the tokens then come faster than the interval
    assert blocks[-1] == b"data: [DONE]"
    for block in blocks:                                  # every other block is still a whole JSON frame
        assert block.startswith(b":") or block == b"data: [DONE]" or json.loads(block[6:])


def test_a_busy_stream_gets_no_comments(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_http, "KEEPALIVE", 0.5)
    engine = PacedEngine("hello")                         # a token every 2 ms
    with serving(app_for(tmp_path, engine)) as port:
        blocks = streamed(port, engine, quiet=0)
    assert not [block for block in blocks if block.startswith(b":")], blocks
    assert blocks[-1] == b"data: [DONE]"


def test_zero_turns_it_off(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_http, "KEEPALIVE", 0.0)
    engine = PacedEngine("hello", hold_at=0)
    with serving(app_for(tmp_path, engine)) as port:
        blocks = streamed(port, engine, quiet=0.3)
    assert not [block for block in blocks if block.startswith(b":")], blocks
    assert blocks[-1] == b"data: [DONE]"


def test_off_unless_set():
    here = Path(__file__).parent              # the child imports this checkout's src, installed or not
    env = {k: v for k, v in os.environ.items() if k != "TF_SSE_KEEPALIVE_S"}
    env["PYTHONPATH"] = os.pathsep.join([str(here.parent / "src"), str(here)])
    out = subprocess.run([sys.executable, "-c", "from tensorfold.cuda import http; print(http.KEEPALIVE)"], env=env,
                         capture_output=True, text=True, timeout=120, check=True)
    assert float(out.stdout.split()[-1]) == 0.0
