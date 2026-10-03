"""reasoning_effort tiers over HTTP: for each tier (none, off, minimal, low, medium, high, xhigh, max, and no effort),
the effort line the prompt renders (/tokenize) and the reasoning tokens of greedy replies to a few problems (standard
library; the key is read from a file, never printed).

  python3 effort_check.py --base URL --model M [--key-file F] [--max-tokens 12000] [--parallel 2] --out F
"""

import argparse
import json
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

TIERS = [None, "none", "off", "minimal", "low", "medium", "high", "xhigh", "max"]
PROBLEMS = [
    "A train leaves at 09:40 and travels 237 km at 79 km/h, then waits 18 minutes, then travels 156 km at 104 km/h. "
    "At what time does it arrive? Give the time only.",
    "Five friends sit in a row. Ana is not at either end. Ben sits immediately right of Cy. Dee is at the left end. "
    "Eve is not next to Ana. Who sits in the middle seat?",
    "Write a Python function that returns the length of the longest strictly increasing contiguous run in a list "
    "of integers, and state its time complexity.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}

    def post(path, body):
        req = urllib.request.Request(a.base.rstrip("/") + path, json.dumps(body).encode(), h)
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"error": e.code, "body": e.read().decode()[:200]}

    rendered = {}
    for tier in TIERS:
        body = {"model": a.model, "messages": [{"role": "user", "content": "hi"}], "return_token_strs": True}
        if tier:
            body["reasoning_effort"] = tier
        r = post("/tokenize", body)
        if "tokens" not in r:
            rendered[str(tier)] = r
            continue
        text = "".join(r.get("token_strs") or [])
        m = re.search(r"Reasoning.Effort:.(\d+)", text)
        rendered[str(tier)] = {"effort_line": int(m.group(1)) if m else None, "thinking": "<think>" in text}
        print(json.dumps({str(tier): rendered[str(tier)]}), flush=True)

    jobs = [(tier, i) for tier in TIERS for i in range(len(PROBLEMS))]
    out, lock = {}, threading.Lock()

    def worker():
        while True:
            with lock:
                if not jobs:
                    return
                tier, i = jobs.pop(0)
            body = {"model": a.model, "messages": [{"role": "user", "content": PROBLEMS[i]}], "temperature": 0,
                    "max_tokens": a.max_tokens}
            if tier:
                body["reasoning_effort"] = tier
            t0 = time.time()
            r = post("/v1/chat/completions", body)
            u = r.get("usage") or {}
            row = {"reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   "completion_tokens": u.get("completion_tokens"), "s": round(time.time() - t0, 1),
                   "finish": (r.get("choices") or [{}])[0].get("finish_reason"), "error": r.get("error")}
            with lock:
                out[(str(tier), i)] = row
            print(json.dumps({"tier": str(tier), "problem": i, **row}), flush=True)

    th = [threading.Thread(target=worker) for _ in range(a.parallel)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    summary = {}
    for tier in TIERS:
        rr = [out[(str(tier), i)] for i in range(len(PROBLEMS)) if (str(tier), i) in out]
        toks = [r["reasoning_tokens"] or 0 for r in rr]
        summary[str(tier)] = {**(rendered.get(str(tier)) or {}), "reasoning_tokens": toks,
                              "median": statistics.median(toks) if toks else None,
                              "errors": [r["error"] for r in rr if r["error"]]}
    print(json.dumps({"summary": summary}), flush=True)
    json.dump({"rendered": rendered, "rows": {f"{k[0]}|{k[1]}": v for k, v in out.items()}, "summary": summary},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
