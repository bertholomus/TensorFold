"""An agent trace over HTTP (streamed chat completions): a Hermes-style session, a long system prompt plus a tools block
of ~24 tool schemas, then turns that each append the model's reply and a tool's output (repo text); then a second
session with the same system prompt and tools. Per turn: prompt tokens, usage's cached_tokens, time to the first
streamed token, reply text digest. Run on a server with TF_DS_KEEP=0 and one with TF_DS_KEEP=1 (``--against`` the
first run's file: the replies must match and the first-token times compare). The key is read from a file.

  python3 agent_trace.py --base URL --model NAME [--key-file F] [--turns 8] [--against F] --out F
"""

import argparse
import hashlib
import json
import pathlib
import time
import urllib.request


def tools_block(n: int = 24) -> list[dict]:
    verbs = ["read", "write", "search", "list", "patch", "run", "fetch", "summarize", "diff", "grep", "move", "test"]
    nouns = ["file", "directory", "issue", "page", "notebook", "branch"]
    out = []
    for i in range(n):
        v, o = verbs[i % len(verbs)], nouns[i % len(nouns)]
        props = {f"{o}_{k}": {"type": t, "description": f"The {k.replace('_', ' ')} of the {o} this call acts on. "
                                                          f"{d} Leave it out to use the session default."}
                 for k, t, d in [("path", "string", "Absolute, or relative to the workspace root."),
                                 ("start_line", "integer", "1-indexed; lines before it are skipped."),
                                 ("end_line", "integer", "Inclusive; past the end means to the end."),
                                 ("pattern", "string", "A regular expression in Python syntax."),
                                 ("dry_run", "boolean", "Report what would change without changing it."),
                                 ("timeout_s", "number", "Give up after this many seconds and report partial output."),
                                 ("labels", "array", "Free-form labels attached to the result for later filtering.")]}
        props[f"{o}_labels"]["items"] = {"type": "string"}
        out.append({"type": "function", "function": {
            "name": f"{v}_{o}_{i}",
            "description": (f"{v.capitalize()} a {o} in the user's workspace and return the result as text. Use it when "
                            f"the task needs the {o}'s current contents or must change them; prefer the narrowest call "
                            "that answers the question, and never guess at contents you have not read. Results longer "
                            "than 20,000 characters are cut, with a note saying where; ask again with a narrower range."),
            "parameters": {"type": "object", "properties": props, "required": [f"{o}_path"]}}})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file")
    p.add_argument("--root", default="/w/TensorFold")
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--tokens", type=int, default=160)
    p.add_argument("--against")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    docs = sorted(q for q in pathlib.Path(a.root).rglob("*.md") if q.stat().st_size > 3000)
    notes = "\n\n".join(q.read_text(errors="ignore") for q in docs)
    srcs = sorted(q for q in pathlib.Path(a.root, "src").rglob("*.py") if q.stat().st_size > 4000)
    system = ("You are Hermes, a careful coding agent working in the user's repository through tools. Read before you "
              "change anything, keep changes small, explain what you did in one short paragraph, and stop when the task "
              "is done. Project notes follow; treat them as background.\n\n" + notes[:24000])
    tools = tools_block()
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}

    def turn(msgs):
        body = {"model": a.model, "messages": msgs, "tools": tools, "max_tokens": a.tokens, "temperature": 0,
                "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), h)
        t0 = time.time()
        first, text, calls, usage = None, [], {}, {}
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                usage = ev.get("usage") or usage
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                        first = time.time() - t0
                    text.append(d.get("content") or "")
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": tc.get("id") or f"call_{len(calls)}",
                                                                  "name": "", "args": ""})
                        f = tc.get("function") or {}
                        c["name"] += f.get("name") or ""
                        c["args"] += f.get("arguments") or ""
        return {"ttft": round(first or 0.0, 3), "total": round(time.time() - t0, 2), "text": "".join(text),
                "calls": [calls[k] for k in sorted(calls)], "usage": usage}

    rows = []
    for session, ask in enumerate(["Find where the server reports cached prompt tokens and explain the path.",
                                   "List the tests that cover streaming replies and say which ones are slow."]):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": ask}]
        for t in range(a.turns if session == 0 else 2):
            r = turn(msgs)
            u = r["usage"]
            row = {"session": session + 1, "turn": t + 1, "prompt": u.get("prompt_tokens"),
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "ttft": r["ttft"],
                   "total": r["total"],      # the reply's digest leaves out the server's random tool-call ids
                   "reply": hashlib.sha256((r["text"] + json.dumps([[c["name"], c["args"]] for c in r["calls"]])).encode()).hexdigest()[:16],
                   "calls": [c["name"] for c in r["calls"]]}
            rows.append(row)
            print(json.dumps(row), flush=True)
            src = srcs[(7 * t + 3 * session) % len(srcs)].read_text(errors="ignore")[:3500]
            if r["calls"]:
                msgs.append({"role": "assistant", "content": r["text"] or None,
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                                            for c in r["calls"]]})
                for c in r["calls"]:
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": src})
            else:
                msgs.append({"role": "assistant", "content": r["text"]})
                msgs.append({"role": "user", "content": "Tool output:\n" + src + "\nContinue."})
    one = [r for r in rows if r["session"] == 1]
    later = one[1:]
    res = {"rows": rows,
           "cached_share_turns_2plus": round(sum(r["cached"] or 0 for r in later) / max(1, sum(r["prompt"] for r in later)), 4),
           "cached_share_all": round(sum(r["cached"] or 0 for r in rows) / max(1, sum(r["prompt"] for r in rows)), 4),
           "ttft_mean_turns_2plus": round(sum(r["ttft"] for r in later) / max(1, len(later)), 3)}
    if a.against:
        ref = json.load(open(a.against))["rows"]
        res["replies_equal"] = f"{sum(x['reply'] == y['reply'] for x, y in zip(rows, ref))}/{len(rows)}"
        res["ttft_before_after"] = [[y["ttft"], x["ttft"]] for x, y in zip(rows, ref)]
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
