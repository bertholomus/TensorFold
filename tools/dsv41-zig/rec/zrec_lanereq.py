#!/usr/bin/env python3
"""The lane gate's requests (tf-dsv41-lanes --requests) from recordings: each recorded prompt's ids (its chunks'
Model.forward ids; a prompt whose first chunks the served engine restored from an earlier prompt takes them from that
prompt: the first serial round after it says how long it is), with the served request's max_tokens, sampling and reply
token_sha (the server's tensorfold.token_sha: sha256 of the reply's comma-joined ids, 12 hex digits). Prompts are
matched to the request records (the clients' JSON outputs, in the order given) by prompt_tokens, ties in order; a
record with a seed is sampled (--temperature, --top-k, --top-p: the recording's), one without is greedy.

  python3 zrec_lanereq.py --out FILE --rec DIR RECORDS.json [RECORDS.json ...] [--rec DIR RECORDS.json ...]
      [--temperature T] [--top-k K] [--top-p P]

Standard library only (the nodes' host Python).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def is_chunk(p: dict) -> bool:
    return p["where"] == "Model.forward" and p["arg"] == "in2" and len(p["shape"]) == 1


def ints(rank_dir: Path, p: dict) -> list[int]:
    data = (rank_dir / "layers" / p["file"]).read_bytes()
    return list(struct.unpack("<%dq" % (len(data) // 8), data))


def head_int(p: dict) -> int:
    return struct.unpack("<q", bytes.fromhex(p["head"][:16]))[0]


def prompts(rank_dir: Path) -> list[list[int]]:
    """Every recorded prompt in order: its chunks to the one with logits, after its restored prefix."""

    pts = [json.loads(line) for line in open(rank_dir / "layers.jsonl")]
    out: list[list[int]] = []
    i, n = 0, len(pts)
    while i < n:
        if not is_chunk(pts[i]):
            i += 1
            continue
        chunks, j = [], i
        while j < n:
            q = pts[j]
            if q["where"] == "RoundDecoder.run":
                break
            if is_chunk(q):
                chunks.append(q)
            j += 1
            if q["where"] == "Model.forward" and q["arg"] == "out" and q["call"] == chunks[-1]["call"]:
                break
        ids = [x for c in chunks for x in ints(rank_dir, c)]
        # a serial round after it: its first position is the prompt's length (a restored prefix shows here)
        total = len(ids)
        for q in pts[j:]:
            if is_chunk(q):
                break
            if q["where"] == "RoundDecoder.run" and q["arg"] == "in2":
                if q["shape"] == [1]:
                    total = head_int(q)
                break
        missing = total - len(ids)
        if missing > 0:
            earlier = next((p for p in out if len(p) >= missing), None)
            if earlier is None:
                raise SystemExit(f"{rank_dir}: a prompt restores {missing} tokens no earlier prompt has")
            ids = earlier[:missing] + ids
        out.append(ids)
        i = j
    return out


def records(files: list[str]) -> list[dict]:
    out = []
    for f in files:
        j = json.loads(Path(f).read_text())
        out += j if isinstance(j, list) else [j]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rec", nargs="+", action="append", required=True, metavar=("DIR", "RECORDS"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()
    reqs = []
    for rec in a.rec:
        d, files = Path(rec[0]), rec[1:]
        ps = prompts(d / "rank0")
        rs = records(files)
        used = [False] * len(rs)
        for k, ids in enumerate(ps):
            r = next((x for x, u in zip(rs, used) if not u and x["usage"]["prompt_tokens"] == len(ids)), None)
            if r is None:
                raise SystemExit(f"{d}: no request record for prompt {k} ({len(ids)} tokens)")
            used[rs.index(r)] = True
            got = r["usage"]["completion_tokens"]
            finish = r["choices"][0]["finish_reason"]
            q = {"name": f"{d.name}:{r.get('_name') or k}", "prompt": ids,
                 "max_tokens": got if finish == "length" else got + 8,
                 "expect_sha": r["tensorfold"]["token_sha"], "expect_tokens": got, "finish": finish}
            if r.get("_seed") is not None:
                q.update(seed=r["_seed"], temperature=a.temperature, top_k=a.top_k, top_p=a.top_p)
            reqs.append(q)
        if not all(used):
            raise SystemExit(f"{d}: {used.count(False)} request records have no recorded prompt")
    Path(a.out).write_text(json.dumps({"requests": reqs}) + "\n")
    for q in reqs:
        print(json.dumps({k: (len(v) if k == "prompt" else v) for k, v in q.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
