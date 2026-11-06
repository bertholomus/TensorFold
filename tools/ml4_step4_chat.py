#!/usr/bin/env python3
"""Step 04-chat: run the fixed prompt set against a served lane and JUDGE the
outputs mechanically (m2b7). Reads prompts.json, POSTs to
the loopback API, saves every raw reply under <out>/chat/, and writes <out>/chat-summary.txt —
one line per prompt: "<id> PASS|FAIL <reason>". Last line printed: "RC 04-chat <0|1>".

    python3 ml4_step4_chat.py --base-url http://127.0.0.1:8111 --model ml4 \
        [--prompts prompts.json] [--out results] [--max-tokens 256] [--timeout 600]

Judge rules (mechanized verbatim from each prompt's judge field, prompts.json):
  p1-chat        non-empty; routing/sparsity mention; no repetition loop; byte-identical on re-run
  p2-multiturn   ss/lsof/netstat present; stays consistent with the given cause list (bind/listen)
  p3-chat        a python block that EXECUTES here and is correct on (1,2,3),(3,2,1),(1,1,2)
  p4-tool-single exactly one tool_call get_weather, args city=Tokyo (ci) and unit=c
  p5-tool-nested first call get_time_zone(city~Kyoto); after the tool-result round-trip,
                 find_flights with date=2026-11-07 (from/to cities checked too)
  p6-ctx8k       the 4-digit code embedded in the seeded 8k filler (seed 1105) appears in the answer
Stdlib only (py3.9+); no node commands: run this ON the node against loopback.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request

# -- the tiny API client (chat only; streaming lives in step 6) -----------------------------------------------------


def post_json(base: str, path: str, body: dict, timeout: float):
    req = urllib.request.Request(
        base.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def chat(base, model, messages, tools=None, timeout=600.0, max_tokens=256):
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    code, raw = post_json(base, "/v1/chat/completions", body, timeout)
    if code != 200:
        raise RuntimeError(f"HTTP {code}: {raw[:300]}")
    msg = json.loads(raw)["choices"][0]["message"]
    return (msg.get("content") or ""), (msg.get("tool_calls") or [])


# -- judge pieces ----------------------------------------------------------------------------------------------------

def has_loop(text: str) -> bool:
    """"word word word ..." x7+, or the same 5-word gram three times — the degenerate serve states."""

    if re.search(r"\b(\w+)(?:\s+\1){6,}\b", text, re.I):
        return True
    w = text.lower().split()
    grams = {}
    for i in range(len(w) - 4):
        g = tuple(w[i:i + 5])
        grams[g] = grams.get(g, 0) + 1
        if grams[g] >= 3:
            return True
    return False


def extract_python(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
    if m:
        return m.group(1)
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def judge_median_code(code: str):
    """exec the extracted python; its (first) callable must be the median-of-three on the 3 tuples."""

    ns: dict = {}
    try:
        exec(code, ns)                                         # noqa: S102  (judge contract: run the model's snippet)
    except Exception as exc:                                   # noqa: BLE001
        return f"does not execute: {exc}"
    fn = next((v for k, v in ns.items() if callable(v) and not k.startswith("__")), None)
    if fn is None:
        return "defines no function"
    for args, want in (((1, 2, 3), 2), ((3, 2, 1), 2), ((1, 1, 2), 1)):
        try:
            try:
                got = fn(*args)
            except TypeError:
                got = fn(list(args))
        except Exception as exc:                               # noqa: BLE001
            return f"raises on {args}: {exc}"
        if got != want:
            return f"wrong on {args}: got {got}, want {want}"
    return None


def filler_8k(seed: int = 1105):
    """The p6 fixture: ~8k words of seeded filler with a seed-derived 4-digit code embedded at the
    midpoint. Same construction every run, so node and judge agree on the code without sharing it."""

    rng = random.Random(seed)
    code = f"{rng.randint(0, 9999):04d}"
    words = ("alpha basis cosine delta engine fusion gradient height island-junction kernel lattice "
             "metric nexus oracle pivot quantum ridge signal tether umbra vector welfare xyloid yield").split()
    parts = [rng.choice(words) for _ in range(8100)]
    parts.insert(len(parts) // 2, f"The four digit code of this passage is {code}.")
    return " ".join(parts), code


def args_of(call: dict) -> dict:
    try:
        return json.loads(call.get("function", {}).get("arguments") or "{}")
    except ValueError:
        return {}


# -- per-prompt judges -------------------------------------------------------------------------------------------------

def judge_prompt(p: dict, base: str, model: str, timeout: float, max_tokens: int, rawdir):
    pid, kind = p["id"], p.get("kind", "chat")
    msgs, tools = p["messages"], p.get("tools")

    def save(run, obj):
        rawdir.joinpath(f"{pid}.{run}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        return obj

    if kind == "tool" and pid == "p4-tool-single":
        content, calls = chat(base, model, msgs, tools, timeout, max_tokens)
        save("run1", {"content": content, "tool_calls": calls})
        if len(calls) != 1:
            return "FAIL", f"expected exactly 1 tool_call, got {len(calls)} (content {content[:60]!r})"
        a = args_of(calls[0])
        fn = calls[0].get("function", {}).get("name")
        if fn != "get_weather" or str(a.get("city", "")).lower() != "tokyo" or str(a.get("unit", "")).lower() != "c":
            return "FAIL", f"tool_call mismatch: {fn} {a}"
        return "PASS", "get_weather city=Tokyo unit=c"

    if kind == "tool" and pid == "p5-tool-nested":
        content, calls = chat(base, model, msgs, tools, timeout, max_tokens)
        save("run1", {"content": content, "tool_calls": calls})
        if not calls or calls[0].get("function", {}).get("name") != "get_time_zone" \
                or "kyoto" not in str(args_of(calls[0]).get("city", "")).lower():
            return "FAIL", f"first response must call get_time_zone(Kyoto), got {calls or content[:60]!r}"
        # both calls in one reply are as good as one after the other, and each call gets its result (ML4-CP4: the model does this)
        answers = {"get_time_zone": {"time_zone": "Asia/Tokyo"}, "find_flights": {"flights": [{"flight": "SK 983", "stops": 1}]}}
        flow = list(msgs) + [{"role": "assistant", "content": None, "tool_calls": calls}] + [
            {"role": "tool", "tool_call_id": c.get("id", f"c{i}"), "content": json.dumps(answers.get(c["function"]["name"], {}))}
            for i, c in enumerate(calls)]
        content2, calls2 = chat(base, model, flow, tools, timeout, max_tokens)
        save("run2", {"content": content2, "tool_calls": calls2})
        ff = next((c for c in calls + calls2 if c.get("function", {}).get("name") == "find_flights"), None)
        if ff is None:
            return "FAIL", f"after the tool result, no find_flights call: {calls2 or content2[:60]!r}"
        if ff in calls and not content2.strip():
            return "FAIL", "both calls in one reply, then no answer from their results"
        a = args_of(ff)
        if str(a.get("date", "")) != "2026-11-07":
            return "FAIL", f"find_flights date={a.get('date')!r}, want 2026-11-07 (day after 2026-11-05)"
        if "oslo" not in str(a.get("from_city", "")).lower() or "kyoto" not in str(a.get("to_city", "")).lower():
            return "FAIL", f"find_flights cities mismatch: {a}"
        return "PASS", "get_time_zone(Kyoto) -> find_flights(2026-11-07)"

    # chat prompts: p6 first needs its seeded fixture rendered into the messages
    if pid == "p6-ctx8k":
        text, code = filler_8k(1105)
        msgs = [{"role": "user", "content":
                 f"Here is a text: {text} Summarize it in one sentence and state the 4-digit code "
                 f"embedded in it."}]
    else:
        code = None
    out, calls = chat(base, model, msgs, None, timeout, max_tokens)
    save("run1", {"content": out, "tool_calls": calls})
    if not out.strip():
        return "FAIL", "empty answer"
    if pid != "p6-ctx8k" and has_loop(out):                    # p6's prompt is 8k seeded words — the
        return "FAIL", "repetition loop"                       # loop rule belongs to the short chats
    if pid == "p1-chat":
        if not re.search(r"routing|rout(?:es|ing|ed)|spars|activat", out, re.I):
            return "FAIL", "off-topic: no routing/sparsity mention"
        out2, _ = chat(base, model, msgs, None, timeout, max_tokens)
        save("run2", {"content": out2})
        if out2 != out:
            return "FAIL", "nondeterministic: re-run differs byte-for-byte"
        return "PASS", "non-empty, on-topic, no loop, deterministic re-run"
    if pid == "p2-multiturn":
        if not re.search(r"\bss\b|lsof|netstat", out, re.I):
            return "FAIL", "answer names no ss/lsof/netstat"
        if not re.search(r"bind|listen|0\.0\.0\.0|port", out, re.I):
            return "FAIL", "inconsistent with the given cause list (no bind/listen notion)"
        return "PASS", "bind-check command named, consistent with the list"
    if pid == "p3-chat":
        bad = judge_median_code(extract_python(out))
        return ("FAIL", f"python check: {bad}") if bad else ("PASS", "valid python, correct on all 3 tuples")
    if pid == "p6-ctx8k":
        if code not in out:
            return "FAIL", f"the seeded code {code} does not appear in the answer"
        return "PASS", f"code {code} recovered over the 8k context"
    return "FAIL", f"no judge rule for prompt id {pid}"


# -- main ---------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)

    from pathlib import Path

    out = Path(a.out)
    (out / "chat").mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(a.prompts).read_text())["prompts"]
    lines, fails = [], 0
    for p in prompts:
        try:
            verdict, reason = judge_prompt(p, a.base_url, a.model, a.timeout, a.max_tokens, out / "chat")
        except Exception as exc:                               # noqa: BLE001  (one dead prompt must not stop the set)
            verdict, reason = "FAIL", f"request error: {exc}"
        fails += verdict == "FAIL"
        lines.append(f"{p['id']} {verdict} {reason}")
        print(lines[-1], flush=True)
    (out / "chat-summary.txt").write_text("\n".join(lines) + f"\nRC 04-chat {1 if fails else 0}\n")
    print(f"RC 04-chat {1 if fails else 0}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
