"""The layer fixtures' request: one fixed chat prompt of about 2,100 tokens (a full 2,048-row prompt chunk and a short
one), greedy, 8 tokens (or --max-tokens), to the served endpoint. The words come from a fixed generator, so every run
sends the same text.

  python3 zrec_req.py --base URL --model NAME --key-file FILE [--max-tokens N] --out OUT.json
"""

import argparse
import json
import time
import urllib.request

WORDS = ("river stone lantern copper meadow signal harbor quiet winter orchard ladder canvas thunder pocket velvet "
         "compass marble garden falcon ember silver needle candle bridge shadow window cedar planet violet anchor").split()


def text(n_words: int) -> str:
    x, out = 20261007, []
    for i in range(n_words):
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(WORDS[(x >> 33) % len(WORDS)])
        if i % 12 == 11:
            out[-1] += "."
    return "Read these words and then name the one that appears most often: " + " ".join(out)


def key(path: str) -> str:
    for line in open(path):
        if "=" in line:
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("no key")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--words", type=int, default=1900)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    body = {"model": a.model, "messages": [{"role": "user", "content": text(a.words)}], "max_tokens": a.max_tokens,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key(a.key_file)})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.loads(r.read())
    res["_seconds"] = round(time.time() - t0, 2)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"seconds": res["_seconds"], "usage": res.get("usage")}))


if __name__ == "__main__":
    main()
