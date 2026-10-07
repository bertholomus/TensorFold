"""Draft policies on one running lane (tf_policy per request: "0" serial, "N" at most N drafts a
round, "N:C" while their chance holds C), greedy, ~8k-token code and prose prompts (the first CHARS of each text), REPLY
tokens each. Prints each run's engine rate, rounds, drafted / accepted, checks every policy's reply equals the serial
one, and last a line "best POLICY" (highest geometric mean of the code and prose rates among the drafter's policies;
"mN" runs the MTP head's N drafts on the same lane, for comparison).
usage: tf_draft_sweep.py URL CODE_TEXT PROSE_TEXT POLICY [POLICY...]     (env CHARS=32000, REPLY=256)
"""

import json
import math
import os
import sys
import urllib.request

URL, CODE, PROSE = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
POLICIES = sys.argv[4:]
CHARS, REPLY = int(os.environ.get("CHARS", "32000")), int(os.environ.get("REPLY", "256"))
ASK = {"code": "Continue the code from where it stops.", "prose": "Continue the story from where the text stops."}


def ask(text: str, kind: str, policy: str) -> tuple[list, dict]:
    body = {"messages": [{"role": "user", "content": text[:CHARS] + "\n\n" + ASK[kind]}], "max_tokens": REPLY,
            "temperature": 0, "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False},
            "tf_policy": policy}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    tf = out.get("tensorfold") or {}
    return (tf.get("token_ids") or out["choices"][0]["message"].get("content")), {**tf, **(out.get("usage") or {})}


rates: dict = {}
same = total = 0
for kind, path in (("code", CODE), ("prose", PROSE)):
    text = open(path, encoding="utf-8", errors="replace").read()
    serial, s0 = ask(text, kind, "0")
    print(f"{kind:5s} serial   {s0.get('tokens_per_second', 0):6.2f} tok/s  prompt {s0.get('prompt_tokens')} tok", flush=True)
    for p in POLICIES:
        reply, st = ask(text, kind, p)
        ok = reply == serial
        same += ok
        total += 1
        n, rounds = st.get("completion_tokens") or 0, st.get("rounds") or 1
        rates.setdefault(p, {})[kind] = st.get("tokens_per_second", 0.0)
        print(f"{kind:5s} {p:8s} {st.get('tokens_per_second', 0):6.2f} tok/s  {n / rounds:4.2f} tok a round  drafted "
              f"{st.get('drafted')} accepted {st.get('accepted')}  {'== serial' if ok else 'DIFFERS FROM SERIAL'}",
              flush=True)
print(f"same {same} / {total}", flush=True)
drafting = [p for p in rates if not p.startswith("m")] or list(rates)      # ("mN": the MTP head, for comparison)
best = max(drafting, key=lambda p: math.sqrt(max(rates[p].get("code", 0), 1e-9) * max(rates[p].get("prose", 0), 1e-9)))
print(f"best {best}", flush=True)
