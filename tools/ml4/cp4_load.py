"""Load on a served lane, every request streamed: time of first token, speed, errors.

    python3 tools/ml4/cp4_load.py --text-file prompts/needle120000.txt streams --lens 8000,12000,16000,20000,24000,28000,32000,32000
    python3 tools/ml4/cp4_load.py --text-file prompts/needle120000.txt ttft --long 120000 --short 1500
    python3 tools/ml4/cp4_load.py --text-file prompts/needle120000.txt soak --seconds 7200 --out soak.jsonl

``--text-file`` is the text prompts are cut from, at a random place each, so no two prompts share a start. Host python, standard
library only.
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import threading
import time

CHARS = 3.95        # characters a token of the prompt texts, near enough: usage gives the real count


def ask(a, text: str, max_tokens: int, tag: str = "", ignore_eos: bool = False) -> dict:
    body = json.dumps({"model": "ml4", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": "low", "stream": True, "stream_options": {"include_usage": True},
                       "ignore_eos": ignore_eos})
    r = {"tag": tag, "start": time.time(), "first": None, "times": [], "error": "", "finish": "", "prompt": 0, "out": 0, "text": ""}
    try:
        conn = http.client.HTTPConnection(a.host, a.port, timeout=None)
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        res = conn.getresponse()
        if res.status != 200:
            r["error"] = f"HTTP {res.status}: {res.read()[:200]!r}"
            return r
        for raw in res:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                r["prompt"], r["out"] = d["usage"].get("prompt_tokens", 0), d["usage"].get("completion_tokens", 0)
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece:
                    now = time.time()
                    r["first"] = r["first"] or now
                    r["times"].append(now)
                    r["text"] += piece
                if c.get("finish_reason"):
                    r["finish"] = c["finish_reason"]
        conn.close()
    except Exception as e:                                    # a dropped stream is what the soak counts
        r["error"] = repr(e)[:200]
    r["end"] = time.time()
    return r


def cut(text: str, tokens: int, rng: random.Random) -> str:
    n = min(int(tokens * CHARS), len(text))
    at = rng.randrange(0, len(text) - n + 1)
    return "Read this and say in one line what it is about.\n\n" + text[at:at + n]


def streams(a, text: str) -> None:
    lens = [int(x) for x in a.lens.split(",")]
    rng = random.Random(a.seed)
    out: list = [None] * len(lens)
    go = threading.Event()

    def one(i: int) -> None:
        p = cut(text, lens[i], rng)
        go.wait()
        out[i] = ask(a, p, a.tokens, f"s{i}", ignore_eos=True)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(lens))]
    for t in ts:
        t.start()
    time.sleep(0.2)
    t0 = time.time()
    go.set()
    for t in ts:
        t.join()
    bad = [r for r in out if r["error"] or r["first"] is None]
    for i, r in enumerate(out):
        print(f"[load] stream {i}: prompt {r['prompt']} tokens, first token {(r['first'] or 0) - t0:.1f} s, {len(r['times'])} pieces, "
              f"{r['finish']} {r['error']}", flush=True)
    if bad:
        print(f"[load] {len(bad)} streams failed", flush=True)
        return
    lo, hi = max(r["first"] for r in out), min(r["times"][-1] for r in out)
    if hi <= lo:
        print("[load] the streams never decoded together: more --tokens", flush=True)
        return
    rates = [sum(lo <= t <= hi for t in r["times"]) / (hi - lo) for r in out]
    print(f"[load] all {len(out)} decoding for {hi - lo:.1f} s: {sum(rates):.1f} tok/s in all, per stream "
          f"{min(rates):.1f} to {max(rates):.1f} ({' '.join(f'{x:.1f}' for x in rates)})", flush=True)


def ttft(a, text: str) -> None:
    rng = random.Random(a.seed)
    done = threading.Event()
    long = {}

    def bg() -> None:
        long.update(ask(a, cut(text, a.long, rng), 8, "long"))
        done.set()

    threading.Thread(target=bg, daemon=True).start()
    time.sleep(a.wait)
    got = []
    while not done.is_set():
        r = ask(a, cut(text, a.short, rng), 8, "short")
        if r["first"] is None:
            print(f"[load] short request failed: {r['error']}", flush=True)
            break
        got.append(r["first"] - r["start"])
        print(f"[load] short request, {r['prompt']} tokens: first token {got[-1]:.2f} s, whole {r['end'] - r['start']:.2f} s, "
              f"long done: {done.is_set()}", flush=True)
        time.sleep(a.gap)
    done.wait()
    if long.get("first"):
        print(f"[load] long request {long['prompt']} tokens: first token {long['first'] - long['start']:.1f} s, "
              f"{long['prompt'] / (long['first'] - long['start']):.0f} tok/s with the short ones beside it", flush=True)
    if got:
        print(f"[load] {len(got)} short requests: first token {min(got):.2f} to {max(got):.2f} s", flush=True)


def soak(a, text: str) -> None:
    end = time.time() + a.seconds
    log = open(a.out, "a") if a.out else None
    lock = threading.Lock()
    tally: dict = {}

    def note(kind: str, r: dict) -> None:
        with lock:
            t = tally.setdefault(kind, {"n": 0, "errors": 0, "first_max": 0.0, "prompt": 0, "out": 0})
            t["n"] += 1
            t["errors"] += bool(r["error"]) or r["first"] is None
            if r["first"]:
                t["first_max"] = max(t["first_max"], r["first"] - r["start"])
            t["prompt"] += r["prompt"]
            t["out"] += r["out"]
            if log:
                log.write(json.dumps({k: r[k] for k in ("tag", "start", "first", "end", "prompt", "out", "finish", "error") if k in r}) + "\n")
                log.flush()
            if r["error"]:
                print(f"[soak] {kind} error: {r['error']}", flush=True)

    def worker(kind: str, tokens: int, out: int, wait: float, seed: int) -> None:
        rng = random.Random(seed)
        while time.time() < end:
            note(kind, ask(a, cut(text, int(tokens * rng.uniform(0.5, 1.0)), rng), out, kind))
            time.sleep(wait)

    plan = [("chat", 1500, 200, 1.0, 1), ("chat", 800, 120, 3.0, 2), ("chat", 3000, 300, 5.0, 3), ("chat", 6000, 200, 8.0, 4),
            ("mid", 128000, 60, 5.0, 5), ("long", a.long, 60, 30.0, 6)]
    ts = [threading.Thread(target=worker, args=p, daemon=True) for p in plan]
    for t in ts:
        t.start()
    while time.time() < end:
        time.sleep(min(300, max(1, end - time.time())))
        with lock:
            print(f"[soak] {time.strftime('%H:%M:%S')} {int(end - time.time())} s left: {json.dumps(tally)}", flush=True)
    for t in ts:
        t.join(timeout=1800)
    print(f"[soak] done: {json.dumps(tally)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8561)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--seed", type=int, default=1)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("streams")
    s.add_argument("--lens", default="8000,12000,16000,20000,24000,28000,32000,32000")
    s.add_argument("--tokens", type=int, default=800)
    s = sub.add_parser("ttft")
    s.add_argument("--long", type=int, default=120000)
    s.add_argument("--short", type=int, default=1500)
    s.add_argument("--wait", type=float, default=15.0)
    s.add_argument("--gap", type=float, default=5.0)
    s = sub.add_parser("soak")
    s.add_argument("--seconds", type=int, default=7200)
    s.add_argument("--long", type=int, default=300000)
    s.add_argument("--out", default="")
    a = ap.parse_args()
    text = open(a.text_file).read()
    {"streams": streams, "ttft": ttft, "soak": soak}[a.cmd](a, text)


if __name__ == "__main__":
    main()
