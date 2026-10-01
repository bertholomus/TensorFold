"""Greedy token ids from a TensorFold endpoint for fixed prompts (for TP1 vs TPn equality). usage: tf_tokens.py URL OUT.json"""
import json, sys, urllib.request

URL, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
prompts = ["The capital of France is", "def fibonacci(n):", "1, 2, 3, 4, 5,"]
res = {}
for p in prompts:
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 24, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        b = json.loads(r.read())
    res[p] = b["choices"][0]["message"].get("content")
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
