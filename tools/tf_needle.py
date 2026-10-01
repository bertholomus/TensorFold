"""Needle recall at a long prompt, with the engine's own prefill and decode speed (its "tensorfold" stats block).

A code word sits at DEPTH (a fraction) of numbered filler lines sized to ~TOKENS prompt tokens (the server's context
refusal for an oversized max_tokens reports the prompt's token count, so the size is found without a prefill); the
reply must open with the code word, then runs on (~REPLY tokens) so the decode rate is measured at that depth.
usage: tf_needle.py URL TOKENS [DEPTH] [REPLY]
"""

import json
import random
import re
import sys
import time
import urllib.error
import urllib.request

URL, TARGET = sys.argv[1].rstrip("/"), int(sys.argv[2])
DEPTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
REPLY = int(sys.argv[4]) if len(sys.argv) > 4 else 256
QUESTION = ("What is the secret code word? Reply with just the code word on the first line, then describe in about "
            "150 words how the document above is organized.")

rng = random.Random(TARGET * 1000 + int(DEPTH * 100))
word = f"{rng.choice(['PELICAN', 'HARBOR', 'QUASAR', 'MAPLE', 'TUNDRA', 'COBALT', 'SAFFRON'])}-{rng.randint(10, 99)}"


def body(lines: int, max_tokens: int) -> dict:
    items = [f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(lines)]
    cut = int(lines * DEPTH)
    doc = " ".join(items[:cut] + [f"The secret code word is {word}."] + items[cut:])
    return {"messages": [{"role": "user", "content": doc + "\n\n" + QUESTION}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


def post(b: dict, timeout: float):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def prompt_tokens(lines: int) -> int:
    """The prompt's token count from the server's context refusal (no prefill runs)."""

    try:
        post(body(lines, 10 ** 8), 600)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        m = re.search(r"\((\d+) prompt tokens", text)
        if m:
            return int(m.group(1))
        raise RuntimeError(f"unexpected refusal: {text[:300]}") from None
    raise RuntimeError("the server accepted a 10^8-token reply")


lines = max(1, TARGET // 15)
for _ in range(4):
    got = prompt_tokens(lines)
    if abs(got - TARGET) <= max(64, TARGET // 500):
        break
    lines = max(1, int(lines * (TARGET - 40) / got))
t = time.time()
b = post(body(lines, REPLY), 24 * 3600)
dt = time.time() - t
content = b["choices"][0]["message"].get("content") or ""
usage, tf = b.get("usage", {}), b.get("tensorfold", {})
prompt = usage.get("prompt_tokens", 0)
pre = tf.get("prefill_s") or 0.0
found = word in content.split("\n")[0]
print(f"prompt {prompt} tok, needle at {DEPTH:.0%}: {'FOUND' if found else 'MISSING'} {word}; prefill {pre:.1f} s "
      f"({prompt / max(pre, 1e-9):.0f} tok/s); decode {usage.get('completion_tokens')} tok at "
      f"{tf.get('tokens_per_second', 0):.2f} tok/s ({tf.get('rounds')} rounds); total {dt:.0f} s", flush=True)
print("   reply:", repr(content[:200]), flush=True)
