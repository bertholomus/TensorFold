"""The sampled recording's requests (Zig port M5b): four chat prompts with the served lane's sampling (temperature 1.0,
top_k 20, top_p 0.95) and a fixed seed each, first one at a time (solo), then all at once (concurrent: rounds of four
streams, then fewer as they end), max_tokens each.

  python3 zrec_sample.py --base URL --model NAME --key-file FILE --mode solo|conc --max-tokens N --out OUT.json
"""

import argparse
import json
import threading
import time
import urllib.request

from zrec_req import key

PROMPTS = [
    ("code", "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals given as a list "
             "of (start, end) pairs, with a docstring and three doctest examples. Then state its time complexity.", 101),
    ("prose", "Write a short, vivid story of about 300 words about a lighthouse keeper who finds a message in a bottle "
              "that was written by her grandmother.", 202),
    ("structured", "Return a JSON array of 8 objects describing fictional library books. Each object has the keys title, "
                   "author, year (between 1900 and 2020), genres (a list of strings) and isbn13. Output only the JSON.",
     303),
    ("networks", "Explain the difference between TCP and UDP in one short paragraph, then give two examples of protocols "
                 "built on each.", 404),
]


def ask(a, name, content, seed, out, slot):
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": a.max_tokens,
            "temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": seed, "stream": False}
    req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key(a.key_file)})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.loads(r.read())
    res["_seconds"] = round(time.time() - t0, 2)
    res["_name"] = name
    res["_seed"] = seed
    out[slot] = res
    print(json.dumps({"name": name, "seed": seed, "seconds": res["_seconds"], "usage": res.get("usage")}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--mode", choices=("solo", "conc"), required=True)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = [None] * len(PROMPTS)
    if a.mode == "solo":
        for i, (name, content, seed) in enumerate(PROMPTS):
            ask(a, name, content, seed, out, i)
    else:
        ts = [threading.Thread(target=ask, args=(a, name, content, seed, out, i)) for i, (name, content, seed) in
              enumerate(PROMPTS)]
        for t in ts:
            t.start()
            time.sleep(0.05)        # the same admission order on every run
        for t in ts:
            t.join()
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
