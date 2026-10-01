"""Smoke-test a TensorFold OpenAI endpoint: greedy chat answers, timings. usage: tf_smoke.py URL [KEYFILE]"""
import json, sys, time, urllib.request

URL = sys.argv[1].rstrip("/")
KEY = open(sys.argv[2]).read().strip() if len(sys.argv) > 2 else ""


def call(body, timeout=900):
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read()[:400].decode(errors="replace")}, time.time() - t


tests = [
    ("math", "What is 17 * 23? Answer with just the number.", 256),
    ("capital", "What is the capital of Australia? One word.", 256),
    ("count", "Count from 1 to 20, separated by commas.", 200),
]
for name, q, mt in tests:
    b, dt = call({"messages": [{"role": "user", "content": q}], "max_tokens": mt, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}})
    if "choices" not in b:
        print(name, "ERROR", b, flush=True)
        continue
    m = b["choices"][0]["message"]
    u = b.get("usage", {})
    print(f"{name}: {dt:.1f}s usage={u} finish={b['choices'][0].get('finish_reason')}\n  content={repr((m.get('content') or '')[:200])}"
          f"\n  reasoning={repr((m.get('reasoning_content') or '')[:120])}", flush=True)
