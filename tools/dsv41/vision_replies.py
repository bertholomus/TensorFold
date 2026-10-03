"""Greedy replies to a fixed set of image prompts (standard library; key read from a file, never printed): run against
two servers (e.g. the tower's reference attention and the served kernel) and compare the files.

  python3 vision_replies.py --base URL --model M --images DIR [--key-file F] --out F [--compare OTHER.json]
"""
import argparse, base64, json, urllib.request
from pathlib import Path

QUESTIONS = ["Describe this image in one sentence.", "What is the main colour in this image? One word.",
             "Is there any text in this image? If so, quote it exactly."]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--model", required=True)
    p.add_argument("--images", required=True); p.add_argument("--key-file"); p.add_argument("--out", required=True)
    p.add_argument("--compare"); p.add_argument("--tokens", type=int, default=64)
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    rows = []
    for img in sorted(Path(a.images).glob("vision_*.png")):
        url = "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode()
        for q in QUESTIONS:
            body = {"model": a.model, "max_tokens": a.tokens, "temperature": 0, "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                              {"type": "text", "text": q}]}]}
            h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
            req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), h)
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            rows.append({"image": img.stem, "question": q, "reply": out["choices"][0]["message"].get("content") or ""})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    res = {"rows": rows}
    if a.compare:
        other = {(r["image"], r["question"]): r["reply"] for r in json.load(open(a.compare))["rows"]}
        same, first_diff = 0, []
        for r in rows:
            o = other.get((r["image"], r["question"]))
            if o is None:
                continue
            same += int(o == r["reply"])
            k = next((i for i, (x, y) in enumerate(zip(o, r["reply"])) if x != y), min(len(o), len(r["reply"])))
            first_diff.append(k)
        res["compare"] = {"identical_replies": same, "of": len(first_diff), "first_diff_chars": first_diff}
        print(json.dumps(res["compare"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
