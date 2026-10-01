"""Greedy outputs for fixed prompts (thinking off), saved for draft-vs-serial equality. usage: tf_greedy.py URL OUT"""
import json, sys, time, urllib.request

URL, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
P = ["Write a detailed explanation of how a hash table works, including collisions, load factor and resizing.",
     "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
     "List the planets of the solar system with one interesting fact about each.",
     "Translate to French: The weather is nice today, so we will go for a walk in the park after lunch."]
res = {}
for p in P:
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 300, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        b = json.loads(r.read())
    dt = time.time() - t
    n = b["usage"]["completion_tokens"]
    res[p] = b["choices"][0]["message"].get("content")
    print(f"{n:4d} tok {dt:6.1f}s {n / dt:6.2f} tok/s  {p[:40]!r}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
