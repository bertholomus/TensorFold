"""RigMark's prefill protocol on the CUDA server: /tokenize, token-ID completion prompts, cache_salt, cached_tokens."""

import http.client
import json
import os
import shutil
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

GLM = Path(os.environ.get("TF_GLM_TOKENIZER_DIR") or os.environ.get("GLM53_BF16_DIR", "/models/GLM-5.3-BF16"))
pytestmark = pytest.mark.skipif(not (GLM / "tokenizer.json").is_file(), reason="needs GLM-5.3's tokenizer files")
LIMIT = 4096


class FakeEngine:
    """Records what the server hands ``generate``; replies with two tokens and reports ``cached`` as told."""

    eos = (154820,)
    limit = LIMIT
    prompt_cache, salt_scoped = True, True

    def __init__(self):
        self.request = threading.local()
        self.calls = []
        self.cached = 0

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True, cache_salt=None, **_):
        self.calls.append({"prompt": list(prompt), "cache_salt": cache_salt, "sampling": sampling})
        on_tokens([11, 12])
        return {"cached": self.cached}


@pytest.fixture()
def model_dir(tmp_path):
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"):
        shutil.copy(GLM / name, tmp_path / name)
    return tmp_path


@pytest.fixture()
def app(model_dir):
    from tensorfold.families.glm_moe_dsa.cuda.app import GlmApp

    return GlmApp(FakeEngine(), model_dir, "glm-5.3", max_tokens=8)


@contextmanager
def serving(app):
    from tensorfold.cuda.server import make_handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def call(port, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        connection.request(method, path, json.dumps(body).encode() if body is not None else None,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def stream_usage(text):
    chunks = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    assert text.rstrip().endswith("data: [DONE]")
    return chunks[-1]["usage"]


def test_tokenize_matches_the_tokenizer_both_routes(app):
    text = "Unique cold-prefill nonce abc.\n"
    want = app.tok.encode(text, add_special_tokens=False).ids
    with serving(app) as port:
        for path in ("/tokenize", "/v1/tokenize"):
            status, out = call(port, "POST", path, {"model": "glm-5.3", "prompt": text, "add_special_tokens": False})
            assert status == 200, out
            got = json.loads(out)
            assert got["tokens"] == want and got["count"] == len(want) and got["max_model_len"] == LIMIT
        status, out = call(port, "POST", "/tokenize", {"prompt": text})
        assert json.loads(out)["tokens"] == app.tok.encode(text, add_special_tokens=True).ids
    assert app.engine.calls == []                   # tokenizing never reaches the engine


def test_tokenize_messages_is_the_chat_prompt(app):
    body = {"messages": [{"role": "user", "content": "Name three primes."}]}
    assert app.tokenize(dict(body))["tokens"] == list(app.prepare(dict(body), True).prompt)


@pytest.mark.parametrize("body", [{}, {"prompt": 5}, {"prompt": "x", "add_special_tokens": "no"}])
def test_tokenize_refuses_bad_bodies(app, body):
    with serving(app) as port:
        status, out = call(port, "POST", "/tokenize", body)
    assert status == 400 and json.loads(out)["error"]["type"] == "invalid_request_error"


def test_token_id_prompt_is_used_as_sent_and_counted_exactly(app):
    ids = app.tok.encode("hello world, " * 200, add_special_tokens=False).ids[:257]
    body = {"model": "glm-5.3", "prompt": ids, "add_special_tokens": False, "max_tokens": 8, "ignore_eos": True,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}, "cache_salt": "s1"}
    with serving(app) as port:
        status, out = call(port, "POST", "/v1/completions", body)
    assert status == 200, out
    usage = stream_usage(out)
    assert usage["prompt_tokens"] == 257 and usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert app.engine.calls[-1]["prompt"] == ids and app.engine.calls[-1]["cache_salt"] == "s1"


@pytest.mark.parametrize("prompt", [[], [1, "a"], [1, True], [[1, 2], [3]], [-1], [10**9]])  # [[ids]] is one prompt
def test_bad_token_id_prompts_are_refused(app, prompt):
    with serving(app) as port:
        status, out = call(port, "POST", "/v1/completions", {"prompt": prompt, "max_tokens": 4})
    assert status == 400, out
    assert app.engine.calls == []


@pytest.mark.parametrize("salt", ["", 7, ["a"], {"a": 1}, True])
def test_bad_cache_salt_is_refused_loudly(app, salt):
    with serving(app) as port:
        status, out = call(port, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4, "cache_salt": salt})
    assert status == 400 and "cache_salt" in json.loads(out)["error"]["message"]


def test_cache_salt_refused_where_the_engine_cannot_scope_its_cache(app):
    app.engine.salt_scoped = False          # e.g. --parallel slots kept with TF_GLM_KEEP_SLOTS=1
    with serving(app) as port:
        status, out = call(port, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4, "cache_salt": "x"})
        assert status == 400 and "cache_salt" in json.loads(out)["error"]["message"]
        status, _ = call(port, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4})
        assert status == 200
    app.engine.prompt_cache = False         # no rows kept across requests: every salt is trivially kept apart
    with serving(app) as port:
        status, out = call(port, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4, "cache_salt": "x"})
    assert status == 200, out


def test_unsalted_requests_reach_the_engine_as_none(app):
    with serving(app) as port:
        call(port, "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4})
    assert app.engine.calls[-1]["cache_salt"] is None


def test_cached_tokens_reports_the_engine_count(app):
    app.engine.cached = 128
    with serving(app) as port:
        status, out = call(port, "POST", "/v1/completions", {"prompt": "hi " * 300, "max_tokens": 4})
    assert json.loads(out)["usage"]["prompt_tokens_details"]["cached_tokens"] == 128


def test_models_reports_max_model_len(app):
    with serving(app) as port:
        status, out = call(port, "GET", "/v1/models")
    assert status == 200 and json.loads(out)["data"][0]["max_model_len"] == LIMIT


# -- the GLM engine: a new salt (or none after one) fills from row 0 on every rank -------------------------------------

def _engine(shared):
    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    eng = object.__new__(GlmEngine)
    eng.scheduler, eng.policy, eng.kept_salt, eng.drafter = None, "3", None, None
    eng.request = threading.local()
    eng._share = lambda values: shared.append(list(values)) or list(values)
    eng._run = lambda *a, fresh=False, **k: {"fresh": fresh}
    return eng


def test_glm_engine_fills_fresh_when_the_salt_changes():
    shared = []
    eng = _engine(shared)
    seen = []
    for salt in (None, None, "a", "a", "b", None, None):
        seen.append(eng.generate([1, 2, 3], 4, None, lambda new: None, cache_salt=salt)["fresh"])
    assert seen == [False, False, True, False, True, True, False]
    headers = shared[0::2]                  # the header the followers read, then the prompt
    assert [h[3] for h in headers] == [int(f) for f in seen]
