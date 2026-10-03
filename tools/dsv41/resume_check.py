"""Kept-prompt reuse against fresh prefill over HTTP: token-id prompts (/v1/completions) that share a primer's start
to various points, then an image chat continued by one more turn. Run once on a server started with TF_DS_KEEP=0
(``--save F``: the fresh replies) and once with TF_DS_KEEP=1 (``--against F``): every reply's token ids must equal the
fresh one's. Usage's cached_tokens and the wall time are reported per case. Text: the TensorFold tree's docs and
sources (two different files for the two token streams); the key is read from a file, never printed.

  python3 resume_check.py --base URL --model NAME --tokenizer DIR [--key-file F] [--tokens 48] \
      (--save F | --against F) --out F
"""

import argparse
import base64
import io
import json
import pathlib
import time
import urllib.request

C, W = 2048, 128                     # the lane's prefill chunk and the sliding window


def texts(root: str) -> tuple[str, str]:
    files = sorted(p for p in pathlib.Path(root).rglob("*") if p.suffix in (".md", ".py") and p.stat().st_size > 2000)
    docs = [p for p in files if p.suffix == ".md"]
    code = [p for p in files if p.suffix == ".py" and "deepseek" not in str(p)]
    return "\n\n".join(p.read_text(errors="ignore") for p in docs), "\n\n".join(p.read_text(errors="ignore") for p in code)


def png(colour, text: str) -> str:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (448, 224), colour)
    ImageDraw.Draw(im).text((20, 90), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--key-file")
    p.add_argument("--root", default="/w/TensorFold")
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--save")
    p.add_argument("--against")
    p.add_argument("--no-image", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from tokenizers import Tokenizer

    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    tok = Tokenizer.from_file(str(pathlib.Path(a.tokenizer) / "tokenizer.json"))
    ta, tb = texts(a.root)
    T = tok.encode(ta).ids[:24000]
    U = tok.encode(tb).ids[:8000]
    assert len(T) >= 24000 and len(U) >= 4000, (len(T), len(U))
    b = 5 * C                                      # 10240: the primer's last chunk boundary but one
    cases = [
        ("primer", T[:12000], {}),
        ("cut+5 past the window", T[:b + W + 5], {}),
        ("cut+130 (window from the kept ring)", T[:b + 130], {}),
        ("cut+127 (one short: replay falls back a chunk)", T[:b + W - 1], {}),
        ("primer again", T[:12000], {}),
        ("across the next chunk (+2100 past 12288)", T[:6 * C + 2100], {}),
        ("diverges at 9000", T[:9000] + U[:2000], {}),
        ("diverges at 3000", T[:3000] + U[:1500], {}),
        ("long continuation", T[:20000], {}),
        ("sampled t0.6", T[:13000], {"temperature": 0.6, "top_p": 0.95, "seed": 7}),
        ("diverges at 9000 again, longer", T[:9000] + U[:3500], {}),
    ]
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}

    def post(path, body):
        req = urllib.request.Request(a.base.rstrip("/") + path, json.dumps(body).encode(), h)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
        return out, time.time() - t0

    rows = []
    for name, ids, samp in cases:
        body = {"model": a.model, "prompt": ids, "max_tokens": a.tokens, "ignore_eos": True, "return_token_ids": True,
                "temperature": 0, **samp}
        out, dt = post("/v1/completions", body)
        u = out.get("usage") or {}
        rows.append({"case": name, "prompt": len(ids), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                     "s": round(dt, 2), "ids": (out.get("tensorfold") or {}).get("token_ids")})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "ids"}), flush=True)
    if not a.no_image:                             # an image early in a long chat, then the chat one turn on
        text = tok.decode(T[12000:17000])
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png((30, 90, 160), "KEPT 42")}},
                                             {"type": "text", "text": "Notes:\n" + text + "\nWhat does the picture say?"}]}]
        for turn in range(2):
            body = {"model": a.model, "messages": msgs, "max_tokens": a.tokens, "ignore_eos": True,
                    "return_token_ids": True, "temperature": 0, "reasoning_effort": "none"}
            out, dt = post("/v1/chat/completions", body)
            u = out.get("usage") or {}
            rows.append({"case": f"image chat turn {turn + 1}", "prompt": u.get("prompt_tokens"),
                         "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "s": round(dt, 2),
                         "ids": (out.get("tensorfold") or {}).get("token_ids")})
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "ids"}), flush=True)
            reply = out["choices"][0]["message"].get("content") or ""
            msgs = msgs + [{"role": "assistant", "content": reply},
                           {"role": "user", "content": "And the colour behind the text, in one word?"}]
    res = {"rows": rows}
    if a.save:
        json.dump(rows, open(a.save, "w"))
    if a.against:
        ref = {r["case"]: r for r in json.load(open(a.against))}
        same = [r["ids"] is not None and r["ids"] == ref.get(r["case"], {}).get("ids") for r in rows]
        res["equal"] = f"{sum(same)}/{len(same)}"
        res["differ"] = [r["case"] for r, s in zip(rows, same) if not s]
        res["resumed"] = sorted({r["cached"] for r in rows if r["cached"]})
        print(json.dumps({k: res[k] for k in ("equal", "differ", "resumed")}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
