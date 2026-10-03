"""A long prompt arriving while other streams decode: the decoding streams' longest gap between tokens (standard
library; the key is read from a file, never printed).

  python3 parallel_stall.py --base URL --model M [--key-file F] [--long 32768] --out F
"""

import argparse
import json
import random
import threading
import time
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file")
    p.add_argument("--long", type=int, default=32768)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}

    def stream(body, times):
        body = dict(body, model=a.model, stream=True)
        req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), h)
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    ch = json.loads(line[5:])
                    if any((c.get("delta") or {}).get("content") for c in ch.get("choices", [])):
                        times.append(time.time())

    rng = random.Random(3)
    words = " ".join(rng.choice(["alpha", "river", "stone", "cloud", "ember", "field", "glass", "harbor"])
                     for _ in range(int(a.long * 0.8)))
    decoders = [[], []]
    base = {"max_tokens": 600, "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    th = [threading.Thread(target=stream, args=({**base, "messages": [{"role": "user", "content": q}]}, t))
          for q, t in zip(["Write a long story about a lighthouse keeper.", "Explain how compilers work, in detail."],
                          decoders)]
    for t in th:
        t.start()
    time.sleep(4)
    long_times: list = []
    t_long = time.time()
    lt = threading.Thread(target=stream, args=({**base, "max_tokens": 16, "messages": [
        {"role": "user", "content": words + "\n\nSummarize the text above in one sentence."}]}, long_times))
    lt.start()
    for t in th + [lt]:
        t.join()
    gaps = [max((b - a_ for a_, b in zip(ts, ts[1:])), default=0.0) for ts in decoders]
    gaps_during = []
    for ts in decoders:
        inside = [x for x in ts if t_long <= x <= (long_times[0] if long_times else time.time())]
        gaps_during.append(max((b - a_ for a_, b in zip(inside, inside[1:])), default=0.0))
    out = {"long_prompt_ttft_s": round(long_times[0] - t_long, 1) if long_times else None,
           "decoder_max_gap_s": [round(g, 2) for g in gaps],
           "decoder_max_gap_while_long_fills_s": [round(g, 2) for g in gaps_during],
           "decoder_tokens_while_long_fills": [sum(1 for x in ts if t_long <= x <= (long_times[0] if long_times else 0))
                                               for ts in decoders]}
    print(json.dumps(out), flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
