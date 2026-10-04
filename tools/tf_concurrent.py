"""Concurrent streaming requests against a lane, each logged with timestamps: when it started, its first token, how
many tokens arrived, when it ended and how (finished, or the error the server sent). For ``--parallel`` lanes: the
aggregate rate, and the kill test (stop one rank's container mid-round; every request must end with an error within
seconds, none may hang).

usage: tf_concurrent.py URL N [MAX_TOKENS=400] [TIMEOUT_S=600]
  TF_CONC_PROMPTS=code: one-line coding requests (tools/tf_multi.py's) instead of the chat ones
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request

URL = sys.argv[1].rstrip("/")
N = int(sys.argv[2])
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
TIMEOUT = float(sys.argv[4]) if len(sys.argv) > 4 else 600.0
TOPICS = ["how a hash table works", "the history of the printing press", "how vaccines train the immune system",
          "the rules of chess for a beginner", "how a CPU executes an instruction", "the water cycle",
          "how compilers optimize loops", "the causes of the French Revolution"]
CODING = ["Write a Python class implementing an LRU cache with O(1) get and put, with docstrings and unit tests.",
          "Write a C function that parses one line of CSV into fields, handling quoted fields and escaped quotes, with a "
          "small test in main().",
          "Implement Dijkstra's shortest path algorithm in Rust with a binary heap, and show it on a small graph.",
          "Write a Node.js Express server with CRUD endpoints for a todo list kept in memory, with input validation.",
          "Write a Python script that reads a web server access log, counts HTTP status codes per hour and prints a "
          "table.",
          "Implement a thread-safe bounded queue in Go with Put, Get and Close, and a test that uses several goroutines.",
          "Write a SQL schema for a small library system (books, members, loans) and five example queries.",
          "Write a Bash script that backs up a directory to a timestamped tar.gz and keeps only the last seven backups."]
CODE = __import__("os").environ.get("TF_CONC_PROMPTS") == "code"
T0 = time.time()


def stamp() -> str:
    return f"{time.time() - T0:7.2f}s"


def one(i: int, out: list) -> None:
    body = {"model": "GLM", "stream": True, "max_tokens": MAX_TOKENS, "temperature": 0,
            "messages": [{"role": "user", "content": CODING[i % len(CODING)] if CODE else
                          f"Write a detailed explanation of {TOPICS[i % len(TOPICS)]}."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec = {"i": i, "start": time.time() - T0, "first": None, "chunks": 0, "end": None, "how": None}
    out[i] = rec
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    rec["how"] = rec["how"] or "finished"
                    break
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in msg:
                    rec["how"] = f"error: {str(msg['error'])[:160]}"
                    break
                choice = (msg.get("choices") or [{}])[0]
                if (choice.get("delta") or {}).get("content"):
                    rec["chunks"] += 1
                    if rec["first"] is None:
                        rec["first"] = time.time() - T0
                if choice.get("finish_reason"):
                    rec["how"] = f"finished ({choice['finish_reason']})"
    except urllib.error.HTTPError as exc:
        rec["how"] = f"HTTP {exc.code}: {exc.read()[:160]!r}"
    except Exception as exc:                          # noqa: BLE001  (connection reset, timeout: logged as such)
        rec["how"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    rec["end"] = time.time() - T0
    print(f"{stamp()} request {i}: {rec['how']} after {rec['chunks']} chunks (first at "
          f"{rec['first'] if rec['first'] is None else round(rec['first'], 2)})", flush=True)


def main() -> None:
    out: list = [None] * N
    threads = [threading.Thread(target=one, args=(i, out)) for i in range(N)]
    print(f"{stamp()} starting {N} requests", flush=True)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    chunks = sum(r["chunks"] for r in out)
    first = min((r["first"] for r in out if r["first"] is not None), default=None)
    last = max(r["end"] for r in out)
    rate = chunks / (last - first) if first is not None and last > first else 0.0
    print(f"{stamp()} all {N} ended: {chunks} chunks, {rate:.2f} chunks/s from the first token to the last end; "
          + "; ".join(f"{r['i']}: {r['how']}" for r in out), flush=True)


if __name__ == "__main__":
    main()
