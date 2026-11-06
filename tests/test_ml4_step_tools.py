"""CPU gates for the m2b7 step kit: tools/ml4_step4_chat.py and tools/ml4_step6_conc.py run whole
against a tiny mock of the loopback API (no nodes, no CUDA): the judge verdicts, the determinism
re-run, the tool round-trip, and the concurrency sweep's refused-row accounting are all exercised
on synthetic answers. Run directly:

    $ML4_HOST_TESTS/.venv/bin/python tests/test_ml4_step_tools.py
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "tools"))

import ml4_step4_chat as S4                                          # noqa: E402
import ml4_step6_conc as S6                                          # noqa: E402

MOE_GOOD = ("A MoE layer routes each token to a few experts, so only a sparse subset of parameters "
            "activates; fewer FLOPs per token make it faster than a dense layer at equal total size.")
MOE_LOOP = "alpha beta gamma delta epsilon " * 5
MOCK = {"mode": "good"}


def chat_msg(content=None, calls=None):
    return {"message": {"content": content, **({"tool_calls": calls} if calls else {})}}


class MockAPI(BaseHTTPRequestHandler):

    def log_message(self, *_a):
        pass

    def _json(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):                                              # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        if self.path == "/v1/chat/completions":
            last = body["messages"][-1].get("content")
            last = last if isinstance(last, str) else ""
            blob = json.dumps(body["messages"])               # tool round 2 ends on a tool msg —
            bad = MOCK["mode"] == "bad"                        # the mock keys on the whole thread
            if "Mixture-of-Experts" in blob:
                out = {"choices": [chat_msg(MOE_LOOP if bad else MOE_GOOD)]}
            elif "distinguishes cause" in last:
                out = {"choices": [chat_msg("Run `ss -ltnp`: it shows the actual bind address, "
                                            "which tells cause 1 from cause 2.")]}
            elif "median of three" in last:
                out = {"choices": [chat_msg("```python\ndef f(a, b, c):\n    return a + b + c - "
                                            "min(a, b, c) - max(a, b, c)\n```")]}
            elif "weather in Tokyo" in last:
                unit = "f" if bad else "c"
                out = {"choices": [chat_msg(None, [{"id": "c0", "type": "function", "function": {
                    "name": "get_weather", "arguments": json.dumps({"city": "Tokyo", "unit": unit})}}])]}
            elif "fly from Oslo" in blob:
                if any(m.get("role") == "tool" for m in body["messages"]):
                    out = {"choices": [chat_msg(None, [{"id": "c1", "type": "function", "function": {
                        "name": "find_flights", "arguments": json.dumps(
                            {"from_city": "Oslo", "to_city": "Kyoto", "date": "2026-11-07"})}}])]}
                else:
                    out = {"choices": [chat_msg(None, [{"id": "c0", "type": "function", "function": {
                        "name": "get_time_zone", "arguments": json.dumps({"city": "Kyoto"})}}])]}
            elif last.startswith("Here is a text:"):
                out = {"choices": [chat_msg(f"It is seeded filler; the embedded code is "
                                            f"{S4.filler_8k(1105)[1]}.")]}
            else:
                out = {"choices": [chat_msg("")]}
            self._json(200, out)
            return
        if self.path == "/v1/completions":
            ctx = len(body["prompt"])
            if ctx > 100:
                if MOCK["mode"] == "error":
                    self._json(400, {"error": {"message": "boom: not a refusal"}})
                else:
                    self._json(500, {"error": {"message": f"no rank holds {ctx} free KV tokens (max 100)"}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(int(body.get("max_tokens", 4))):
                chunk = json.dumps({"choices": [{"text": "tok"}]})
                self.wfile.write(f"data: {chunk}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(404, {"error": "no route"})


PROMPTS = {"prompts": [
    {"id": "p1-chat", "kind": "chat",
     "messages": [{"role": "user", "content": "Explain in 3 sentences why a Mixture-of-Experts "
                                              "layer can be faster than a dense layer."}],
     "judge": "non-empty; mentions routing/sparsity; no repetition loop"},
    {"id": "p2-multiturn", "kind": "chat",
     "messages": [{"role": "user", "content": "My container listens on port 8080 but curl fails."},
                  {"role": "assistant", "content": "1) not listening 2) bound to 127.0.0.1 3) firewall."},
                  {"role": "user", "content": "Which single command distinguishes cause 1 from 2?"}],
     "judge": "ss/lsof/netstat; consistent with the list"},
    {"id": "p3-chat", "kind": "chat",
     "messages": [{"role": "user", "content": "Write a one-line Python function for the median of "
                                              "three numbers, no sort, no statistics module."}],
     "judge": "valid python; correct on (1,2,3),(3,2,1),(1,1,2)"},
    {"id": "p4-tool-single", "kind": "tool", "tools": [{"name": "get_weather"}],
     "messages": [{"role": "user", "content": "What's the weather in Tokyo right now? Use celsius."}],
     "judge": "one tool_call get_weather city=Tokyo unit=c"},
    {"id": "p5-tool-nested", "kind": "tool", "tools": [{"name": "find_flights"}, {"name": "get_time_zone"}],
     "messages": [{"role": "system", "content": "Today is 2026-11-05."},
                  {"role": "user", "content": "I want to fly from Oslo to Kyoto the day after "
                                              "tomorrow. First get_time_zone, then find_flights."}],
     "judge": "get_time_zone(Kyoto) then find_flights date=2026-11-07"},
    {"id": "p6-ctx8k", "kind": "chat",
     "messages": [{"role": "user", "content": "Here is a text: <seeded filler 1105>. Summarize and "
                                              "state the 4-digit code."}],
     "judge": "code matches the seed-derived code"},
]}


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _fixture(tmp: Path):
    (tmp / "prompts.json").write_text(json.dumps(PROMPTS))
    return str(tmp / "prompts.json")


def test_step4_all_pass(tmp="/tmp/ml4_step4_good"):
    """The whole judge flow on a well-behaved mock: 6 PASS lines, RC 0, raw outputs on disk."""

    Path(tmp).mkdir(parents=True, exist_ok=True)
    srv, url = serve()
    try:
        MOCK["mode"] = "good"
        rc = S4.main(["--base-url", url, "--model", "m", "--prompts", _fixture(Path(tmp)),
                      "--out", tmp, "--timeout", "15"])
        assert rc == 0, f"rc {rc}"
        lines = Path(tmp, "chat-summary.txt").read_text().splitlines()
        assert len(lines) == 7 and lines[-1] == "RC 04-chat 0"
        assert all(" PASS " in ln for ln in lines[:6]), lines
        assert Path(tmp, "chat", "p1-chat.run1.json").exists()
        assert Path(tmp, "chat", "p1-chat.run2.json").exists(), "determinism re-run must be saved"
    finally:
        srv.shutdown()


def test_step4_detects_bad_answers(tmp="/tmp/ml4_step4_bad"):
    """Looped p1 and a wrong tool arg must surface as FAIL lines with reasons, RC 1."""

    Path(tmp).mkdir(parents=True, exist_ok=True)
    srv, url = serve()
    try:
        MOCK["mode"] = "bad"
        rc = S4.main(["--base-url", url, "--model", "m", "--prompts", _fixture(Path(tmp)),
                      "--out", tmp, "--timeout", "15"])
        assert rc == 1
        lines = Path(tmp, "chat-summary.txt").read_text().splitlines()
        p1 = next(ln for ln in lines if ln.startswith("p1-chat"))
        p4 = next(ln for ln in lines if ln.startswith("p4-tool-single"))
        assert " FAIL " in p1 and "loop" in p1, p1
        assert " FAIL " in p4 and "mismatch" in p4, p4
        assert all(" PASS " in ln for ln in lines if ln.startswith(("p2", "p3", "p5", "p6")))
    finally:
        srv.shutdown()
    MOCK["mode"] = "good"


def test_step6_sweep_refusals(tmp="/tmp/ml4_step6"):
    """Mock refuses any ctx > 100: refused rows are SKIPPED (rc stays 0), accepted rows report
    ttft p50/p99 and prefill/decode rates, and the JSON table lands."""

    Path(tmp).mkdir(parents=True, exist_ok=True)
    srv, url = serve()
    try:
        MOCK["mode"] = "good"
        rc = S6.main(["--base-url", url, "--model", "m", "--counts", "1,2", "--ctxs", "8,512",
                      "--tokens", "4", "--timeout", "15", "--out", tmp])
        assert rc == 0, f"refusals must not fail the sweep, rc {rc}"
        rows = json.loads(Path(tmp, "conc-table.json").read_text())["rows"]
        by = {(r["c"], r["ctx"]): r for r in rows}
        assert by[(1, 8)]["accepted"] == 1 and by[(2, 8)]["accepted"] == 2
        assert by[(1, 512)]["refused"] == 1 and by[(1, 512)]["skipped"] == "refused"
        assert by[(2, 512)]["refused"] == 2 and by[(2, 512)]["skipped"] == "refused"
        r8 = by[(2, 8)]
        assert r8["ttft_p50_ms"] >= 0 and r8["ttft_p99_ms"] >= r8["ttft_p50_ms"]
        assert r8["prefill_tps"] > 0 and r8["decode_tps"] and r8["tokens_total"] == 8
    finally:
        srv.shutdown()


def test_step6_real_error_fails_rc(tmp="/tmp/ml4_step6_err"):
    """A NON-refusal 400 must fail the row's rc — silence is not success."""

    Path(tmp).mkdir(parents=True, exist_ok=True)
    srv, url = serve()
    try:
        MOCK["mode"] = "error"
        rc = S6.main(["--base-url", url, "--model", "m", "--counts", "1", "--ctxs", "512",
                      "--tokens", "4", "--timeout", "15", "--out", tmp])
        assert rc == 1
        rows = json.loads(Path(tmp, "conc-table.json").read_text())["rows"]
        assert rows[0]["errors"] and "boom" in rows[0]["errors"][0]
    finally:
        srv.shutdown()
    MOCK["mode"] = "good"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    fails = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed", flush=True)
    sys.exit(1 if fails else 0)
