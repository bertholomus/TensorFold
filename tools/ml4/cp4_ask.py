"""One request to a served lane, streamed, with the time of its first token and its speed after that.

    python3 tools/ml4/cp4_ask.py --port 8561 --text-file needle1040000.txt --max-tokens 512
    python3 tools/ml4/cp4_ask.py --port 8561 --text "What is the capital of France?"

No timeout: a 1M prompt answers after an hour. Host python, standard library only.
"""
from __future__ import annotations

import argparse
import http.client
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8561)
    ap.add_argument("--text", default="")
    ap.add_argument("--text-file", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    text = open(a.text_file).read() if a.text_file else a.text
    body = json.dumps({"model": "ml4", "messages": [{"role": "user", "content": text}], "max_tokens": a.max_tokens,
                       "temperature": 0, "reasoning_effort": a.effort, "stream": True,
                       "stream_options": {"include_usage": True}})
    t0 = time.time()
    print(f"[ask] {len(text)} characters, {len(body)} bytes sent at {time.strftime('%H:%M:%S')}", flush=True)
    conn = http.client.HTTPConnection(a.host, a.port, timeout=None)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    print(f"[ask] HTTP {r.status} after {time.time() - t0:.1f} s", flush=True)
    first, n, out = None, 0, []
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            if line and r.status != 200:
                print(line, flush=True)
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        d = json.loads(data)
        if d.get("usage"):
            print(f"[ask] usage {d['usage']}", flush=True)
        for c in d.get("choices") or []:
            delta = c.get("delta") or {}
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                if first is None:
                    first = time.time() - t0
                    print(f"[ask] first token after {first:.1f} s", flush=True)
                n += 1
                out.append(piece)
                print(piece, end="", flush=True)
            if c.get("finish_reason"):
                print(f"\n[ask] finish {c['finish_reason']}", flush=True)
    dt = time.time() - t0
    if first is not None and n > 1:
        print(f"[ask] {n} pieces in {dt - first:.1f} s after the first, {(n - 1) / max(dt - first, 1e-9):.2f} a second", flush=True)
    print(f"[ask] text: {''.join(out)!r}", flush=True)
    print(f"[ask] done {dt:.1f} s", flush=True)


if __name__ == "__main__":
    main()
