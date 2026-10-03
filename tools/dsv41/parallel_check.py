"""Concurrent replies against solo replies (standard library; the key is read from a file, never printed): the oracle's
chat prompts as token ids, greedy and seeded-sampled, ignore_eos, first one at a time, then all at once and staggered;
every reply's token ids must equal its solo run's. Optionally against a reference file from another server.

  python3 parallel_check.py --base URL --model M --oracle O [--key-file F] [--tokens 128] [--reference REF.json]
                            --out F
"""

import argparse
import json
import threading
import time
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--key-file")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--reference")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    recs = [json.loads(line) for line in open(a.oracle)]
    recs = [r for r in recs if r["kind"] == "chat"][:a.prompts]
    cases = [(r["index"], "greedy", {"temperature": 0}) for r in recs] + \
            [(r["index"], "t0.6", {"temperature": 0.6, "top_p": 0.95, "seed": 1000 + r["index"]}) for r in recs[:4]]
    ids_of = {r["index"]: r["prompt_ids"] for r in recs}

    def run(case):
        idx, name, samp = case
        body = {"model": a.model, "prompt": ids_of[idx], "max_tokens": a.tokens, "ignore_eos": True,
                "return_token_ids": True, **samp}
        h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
        req = urllib.request.Request(a.base.rstrip("/") + "/v1/completions", json.dumps(body).encode(), h)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read())
        return (out.get("tensorfold") or {}).get("token_ids"), time.time() - t0

    res = {"solo": {}, "burst": {}, "staggered": {}}
    t0 = time.time()
    for c in cases:
        res["solo"][f"{c[0]}/{c[1]}"] = run(c)[0]
    res["solo_s"] = round(time.time() - t0, 1)

    def concurrent(label, delays):
        got, th = {}, []

        def worker(c, d):
            time.sleep(d)
            got[f"{c[0]}/{c[1]}"] = run(c)[0]

        t1 = time.time()
        for c, d in zip(cases, delays):
            th.append(threading.Thread(target=worker, args=(c, d)))
            th[-1].start()
        for t in th:
            t.join()
        res[label] = got
        res[label + "_s"] = round(time.time() - t1, 1)

    concurrent("burst", [0.0] * len(cases))
    concurrent("staggered", [0.4 * i for i in range(len(cases))])
    summary = {}
    for label in ("burst", "staggered"):
        same = sum(res[label][k] == res["solo"][k] for k in res["solo"])
        summary[label] = {"equal_to_solo": same, "of": len(res["solo"]),
                          "tokens_per_s": round(len(cases) * a.tokens / res[label + "_s"], 2)}
    summary["solo_tokens_per_s"] = round(len(cases) * a.tokens / res["solo_s"], 2)
    if a.reference:
        ref = json.load(open(a.reference))["solo"]
        summary["solo_equal_to_reference"] = {"equal": sum(ref.get(k) == v for k, v in res["solo"].items()),
                                              "of": len(res["solo"])}
    res["summary"] = summary
    print(json.dumps(summary, indent=1), flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
