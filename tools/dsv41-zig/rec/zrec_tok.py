"""The token recording's requests (Zig port M4 Token gate): a fixed set of chat prompts, greedy, one at a time (so each
decode round is one stream's one row), to the served endpoint. Code, prose and structured asks with 256 tokens, and two
long word lists (about 3,000 and 8,000 tokens) with 128. The words come from zrec_req's fixed generator.

  python3 zrec_tok.py --base URL --model NAME --key-file FILE --out OUT.json
"""

import argparse
import json
import time
import urllib.request

from zrec_req import key, text

PROMPTS = [
    ("code", "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals given as a list "
             "of (start, end) pairs, with a docstring and three doctest examples. Then state its time complexity.", 256),
    ("prose", "Write a short, vivid story of about 300 words about a lighthouse keeper who finds a message in a bottle "
              "that was written by her grandmother.", 256),
    ("structured", "Return a JSON array of 8 objects describing fictional library books. Each object has the keys title, "
                   "author, year (between 1900 and 2020), genres (a list of strings) and isbn13. Output only the JSON.",
     256),
    ("words3k", None, 128),
    ("words8k", None, 128),
]
WORDS = {"words3k": 2300, "words8k": 6100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for name, content, max_tokens in PROMPTS:
        if content is None:
            content = text(WORDS[name])
        body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                "temperature": 0, "stream": False}
        req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + key(a.key_file)})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            res = json.loads(r.read())
        res["_seconds"] = round(time.time() - t0, 2)
        res["_name"] = name
        out.append(res)
        print(json.dumps({"name": name, "seconds": res["_seconds"], "usage": res.get("usage")}), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
