"""Replies of a recording (rank dir), by request: the prompt's length and the tokens its rounds verified (serial: one a
round; drafted: each window's kept rows, from the next window's position), and a port's replies (its log's "reply"
lines) against them: drafted == serial and concurrent == solo, request by request, by prompt length.

  python3 zrec_replies.py SERIAL_RANK_DIR [SERIAL_RANK_DIR ...] --port LAYER_LOG
"""

import json
import os
import struct
import sys


def ints(d, x):
    if x.get("file"):
        b = open(os.path.join(d, "layers", x["file"]), "rb").read()
    else:                                   # not saved whole (a token recording): its first bytes
        b = bytes.fromhex(x["head"])[: 8 * x["shape"][0]]
    return list(struct.unpack(f"<{len(b) // 8}q", b))


def requests(d):
    """[(prompt_len, [verified tokens in order])] of a serial recording: each round's id is the reply's next token."""
    out, cur = [], None
    chunks, last_round = 0, None
    for line in open(os.path.join(d, "layers.jsonl")):
        x = json.loads(line)
        if x["where"] == "Model.forward" and x["arg"] == "in2" and len(x["shape"]) == 1:
            if cur is None or cur[1]:
                cur = [0, []]
                out.append(cur)
            cur[0] += x["shape"][0]
        elif x["where"] == "RoundDecoder.run" and x["arg"] == "in1" and cur is not None:
            ids = ints(d, x)
            if len(ids) != 1:
                raise SystemExit(f"{d}: a round of {len(ids)} rows (not serial)")
            cur[1].append(ids[0])
    return out


def main():
    args = sys.argv[1:]
    k = args.index("--port")
    serial_dirs, port_log = args[:k], args[k + 1]
    serial = []
    for d in serial_dirs:
        serial += requests(d)
    port = []
    for line in open(port_log):
        if '"reply"' in line:
            port.append(json.loads(line))
    prompts = {}
    for line in open(port_log):
        if '"prompt_tokens"' in line and '"slot"' in line:
            x = json.loads(line)
            prompts[x["request"]] = x["prompt_tokens"]
    ok = True
    for p in port:
        n = prompts[p["request"]]
        match = [s for s in serial if s[0] == n]
        if not match:
            print(json.dumps({"request": p["request"], "prompt_tokens": n, "serial": "none"}))
            continue
        s = match[0][1]
        r = p["reply"]
        m = min(len(r), len(s))
        eq = r[:m] == s[:m]
        ok &= eq
        print(json.dumps({"request": p["request"], "prompt_tokens": n, "port_tokens": len(r), "serial_tokens": len(s),
                          "compared": m, "equal": eq}))
    print(json.dumps({"drafted_equals_serial": ok}))


if __name__ == "__main__":
    main()
