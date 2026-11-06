"""The fixed greedy set: every speed change's gate. Greedy answers to the same prompts, compared with a saved run.

    python3 tools/ml4/cp4_greedy.py --port 8561 --out greedy-base.json
    python3 tools/ml4/cp4_greedy.py --port 8561 --out greedy-new.json --base greedy-base.json

Prints, per prompt, how many leading characters the two answers share and the end verdict: SAME, or the prompts that differ
and where. Host python, standard library only; answers come one at a time, so a run's rows are its own.
"""
from __future__ import annotations

import argparse
import http.client
import json
import time

PROMPTS = (
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
    "What is 17 times 23? Show the steps.",
    "Translate into French: The weather is lovely today, so we will walk to the market after lunch.",
    "List five prime numbers greater than 100 and say why each is prime.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "A train leaves at 9:15 and arrives at 13:40. How long is the trip?",
    "Write a haiku about a lighthouse in winter.",
    "What are the three laws of thermodynamics? One line each.",
    "Give a SQL query that returns the ten customers with the highest total order value.",
    "Why does a Mixture-of-Experts model read less memory per token than a dense model of the same size?",
    "Name the capital of Australia, Canada and Brazil.",
)


def ask(host: str, port: int, text: str, tokens: int, effort: str) -> dict:
    body = json.dumps({"model": "ml4", "messages": [{"role": "user", "content": text}], "max_tokens": tokens, "temperature": 0,
                       "reasoning_effort": effort})
    conn = http.client.HTTPConnection(host, port, timeout=None)
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    got = json.loads(r.read().decode("utf-8", "replace"))
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {got}")
    m = got["choices"][0]["message"]
    return {"text": (m.get("reasoning_content") or "") + (m.get("content") or ""), "seconds": round(time.time() - t0, 2),
            "tokens": got.get("usage", {}).get("completion_tokens", 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8561)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="")
    a = ap.parse_args()
    runs = []
    for i, p in enumerate(PROMPTS):
        runs.append(ask(a.host, a.port, p, a.tokens, a.effort))
        print(f"[greedy] {i}: {runs[-1]['tokens']} tokens {runs[-1]['seconds']} s {runs[-1]['text'][:70]!r}", flush=True)
    json.dump(runs, open(a.out, "w"), indent=0)
    total = sum(r["tokens"] for r in runs)
    print(f"[greedy] {total} tokens in {sum(r['seconds'] for r in runs):.1f} s", flush=True)
    if a.base:
        base, bad = json.load(open(a.base)), 0
        for i, (x, y) in enumerate(zip(base, runs)):
            if x["text"] == y["text"]:
                continue
            bad += 1
            at = next((k for k, (c, d) in enumerate(zip(x["text"], y["text"])) if c != d), min(len(x["text"]), len(y["text"])))
            print(f"[greedy] prompt {i} differs at character {at} of {len(x['text'])}: {x['text'][at:at + 40]!r} / "
                  f"{y['text'][at:at + 40]!r}", flush=True)
        print(f"[greedy] {'SAME' if not bad else f'{bad} of {len(runs)} differ'}", flush=True)


if __name__ == "__main__":
    main()
