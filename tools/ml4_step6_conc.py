#!/usr/bin/env python3
"""Step 06-conc: concurrency x context sweep against a served lane (m2b7).

    python3 ml4_step6_conc.py --base-url http://127.0.0.1:8111 --model ml4 \
        [--counts 1,2,4,8,12,16] [--ctxs 1024,16384,131072] [--tokens 256] [--timeout 900] [--out results]

For every (c x ctx) row: fire c concurrent STREAMING /v1/completions requests — prompt = ctx copies
of token id 1000 (token-id prompts, server.py takes lists as given), max_tokens = --tokens — and
record per request: time-to-first-token, tokens received, and status. An admission refusal ("no
rank holds", "does not fit its ledger", "no room", CONTEXT_LIMIT) marks the request skipped=refused
— the row is NOT failed, that is the pool arithmetic answering. Everything else is an error and
fails the row's rc. Writes <out>/conc-table.json and prints the table; last line "RC 06-conc <0|1>".

Self-test knobs (also plain args): env CONC_COUNTS / CONC_CTXS / CONC_TOKENS / CONC_TIMEOUT shrink
the sweep so the mock-server test finishes in seconds. Stdlib only (py3.9+); no node commands.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

REFUSAL_RE = re.compile(r"does not fit|no rank holds|no room|CONTEXT_LIMIT|free KV", re.I)


def one_request(base: str, model: str, ctx: int, tokens: int, timeout: float):
    """One streaming completion: returns dict(status, ttft_s, toks, wall_s, note)."""
    body = {"model": model, "prompt": [1000] * ctx, "max_tokens": tokens, "temperature": 0.0,
            "stream": True}
    req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    toks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line_b in r:
                line = line_b.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                ch = (chunk.get("choices") or [{}])[0]
                piece = ch.get("text") if ch.get("text") is not None else \
                    (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    toks += 1
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        if exc.code >= 400 and REFUSAL_RE.search(detail):
            return {"status": "refused", "ttft_s": None, "toks": 0,
                    "wall_s": time.monotonic() - t0, "note": detail[:200]}
        return {"status": "error", "ttft_s": None, "toks": 0,
                "wall_s": time.monotonic() - t0, "note": f"HTTP {exc.code}: {detail[:200]}"}
    except Exception as exc:                                   # noqa: BLE001
        return {"status": "error", "ttft_s": None, "toks": 0,
                "wall_s": time.monotonic() - t0, "note": f"{type(exc).__name__}: {exc}"}
    if ttft is None or toks == 0:
        return {"status": "error", "ttft_s": None, "toks": 0, "wall_s": time.monotonic() - t0,
                "note": "stream carried no tokens"}
    return {"status": "ok", "ttft_s": ttft, "toks": toks, "wall_s": time.monotonic() - t0, "note": ""}


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def run_row(base, model, c, ctx, tokens, timeout):
    res = [None] * c
    wall0 = time.monotonic()

    def go(i):
        res[i] = one_request(base, model, ctx, tokens, timeout)

    ts = [threading.Thread(target=go, args=(i,)) for i in range(c)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=timeout + 30)
    wall = time.monotonic() - wall0
    ok = [r for r in res if r["status"] == "ok"]
    refused = [r for r in res if r["status"] == "refused"]
    errors = [r for r in res if r["status"] == "error"]
    row = {"c": c, "ctx": ctx, "accepted": len(ok), "refused": len(refused) or None,
           "skipped": "refused" if refused and not ok else None,
           "errors": [e["note"] for e in errors] or None,
           "wall_s": round(wall, 3)}
    if ok:
        ttfts = [r["ttft_s"] for r in ok]
        gen = sum(r["toks"] - 1 for r in ok)                   # tokens after the first (decode only)
        dec = sum(r["wall_s"] - r["ttft_s"] for r in ok)
        row.update({"ttft_p50_ms": round(pct(ttfts, 0.50) * 1000, 1),
                    "ttft_p99_ms": round(pct(ttfts, 0.99) * 1000, 1),
                    "prefill_tps": round(ctx * len(ok) / sum(ttfts), 1),
                    "decode_tps": round(gen / dec, 2) if dec > 0 else None,
                    "tokens_total": sum(r["toks"] for r in ok)})
    return row, not errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--counts", default=os.environ.get("CONC_COUNTS", "1,2,4,8,12,16"))
    ap.add_argument("--ctxs", default=os.environ.get("CONC_CTXS", "1024,16384,131072"))
    ap.add_argument("--tokens", type=int, default=int(os.environ.get("CONC_TOKENS", "256")))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("CONC_TIMEOUT", "900")))
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    from pathlib import Path

    counts = [int(x) for x in a.counts.split(",") if x.strip()]
    ctxs = [int(x) for x in a.ctxs.split(",") if x.strip()]
    rows, rc = [], 0
    hdr = f"{'c':>3} {'ctx':>7} {'ok':>3} {'refused':>7} {'ttft_p50':>9} {'ttft_p99':>9} {'prefill':>9} {'decode':>8}"
    print(hdr, flush=True)
    for c in counts:
        for ctx in ctxs:
            row, clean = run_row(a.base_url, a.model, c, ctx, a.tokens, a.timeout)
            rc |= 0 if clean else 1
            rows.append(row)
            print(f"{c:>3} {ctx:>7} {row.get('accepted', 0):>3} {row.get('refused') or 0:>7} "
                  f"{row.get('ttft_p50_ms', '-'):>9} {row.get('ttft_p99_ms', '-'):>9} "
                  f"{row.get('prefill_tps', '-'):>9} {row.get('decode_tps', '-'):>8}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "conc-table.json").write_text(json.dumps(
        {"meta": {"base_url": a.base_url, "model": a.model, "tokens": a.tokens}, "rows": rows}, indent=1))
    print(f"RC 06-conc {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
