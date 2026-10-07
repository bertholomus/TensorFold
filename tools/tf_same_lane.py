"""On one running lane, the drafter's replies against the same lane's serial replies (tf_policy "0":
every round one token, no drafts), greedy and sampled (keyed seed), over short chats and prompts of a few thousand
tokens cut from the given texts (past the drafter's window and DSA's dense limit). Prints each pair's equality, the
engine's decode rate and rounds, and a final "same N / M" line.  usage: tf_same_lane.py URL TEXT [TEXT...]
"""

import hashlib
import json
import sys
import urllib.request

URL = sys.argv[1].rstrip("/")
TEXTS = [open(p, encoding="utf-8", errors="replace").read() for p in sys.argv[2:]]
REPLY = 256

PROMPTS = [("chat", "Explain in a few sentences why the sky looks blue, then give one everyday example."),
           ("list", "Write a numbered list of eight prime numbers above 100 and say how you checked each one.")]
for i, text in enumerate(TEXTS):
    for chars in (12000, 36000):            # ~3k and ~9k tokens
        PROMPTS.append((f"text{i}_{chars // 1000}k", text[:chars] + "\n\nContinue from where the text stops."))
MODES = [("greedy", {"temperature": 0}),
         ("sampled", {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "seed": 7})]


def ask(content: str, extra: dict, serial: bool) -> tuple[str, dict]:
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": REPLY, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    if serial:
        body["tf_policy"] = "0"
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"].get("content") or "", out.get("tensorfold") or {}


same = total = 0
for name, content in PROMPTS:
    for mode, extra in MODES:
        a, sa = ask(content, extra, serial=True)
        b, sb = ask(content, extra, serial=False)
        ok = a == b and sa.get("token_ids") == sb.get("token_ids")      # the reply's token ids, and its text
        same += ok
        total += 1
        print(f"{name:10s} {mode:8s} {'SAME' if ok else 'DIFFERENT'}  serial {sa.get('tokens_per_second', 0):6.2f} tok/s "
              f"({sa.get('rounds')} rounds)  drafter {sb.get('tokens_per_second', 0):6.2f} tok/s ({sb.get('rounds')} rounds)"
              f"  sha {hashlib.sha256(b.encode()).hexdigest()[:12]}", flush=True)
        if not ok:
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            print(f"   first difference at char {k}: serial {a[k:k + 60]!r} / drafter {b[k:k + 60]!r}", flush=True)
print(f"same {same} / {total}", flush=True)
