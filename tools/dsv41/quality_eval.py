"""A small reproducible quality eval over HTTP (standard library only; the key is read from a file, never printed):
a fixed 200-question MMLU subset and a fixed 100-problem GSM8K subset, greedy, thinking off and on.

The subsets are drawn once with random.Random(20261004) from the MMLU "all" test split (14042 questions, cais/mmlu,
MIT) and the GSM8K test split (1319 problems, openai/grade-school-math, MIT); their indices and a digest of their text
are written with the results, so a rerun on the same files asks the same questions.

  python3 quality_eval.py --base URL --model M [--key-file F] --mmlu mmlu_all_test.jsonl --gsm8k gsm8k_test.jsonl \
      [--modes off,on] [--parallel 4] --out F
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request

SEED = 20261004
LETTERS = "ABCD"


def mmlu_prompt(r: dict, thinking: bool) -> str:
    subject = r["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}.", "", r["question"].strip()]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"])]
    lines += ["", "Think it through, then end your reply with a line 'Answer: X', where X is the letter of the correct "
                  "option." if thinking else "Answer with the letter of the correct option only."]
    return "\n".join(lines)


def gsm_prompt(r: dict) -> str:
    return r["question"].strip() + "\n\nSolve the problem. End your reply with a line 'Answer: N', where N is the final " \
                                   "number only."


def mmlu_pick(text: str, thinking: bool) -> str | None:
    m = re.findall(r"Answer\s*[:：]\s*\(?([ABCD])\b", text)
    if m:
        return m[-1]
    if not thinking:
        m = re.search(r"\b([ABCD])\b", text)
        return m.group(1) if m else None
    return None


def number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = m[-1]
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def gsm_pick(text: str) -> str | None:
    m = re.findall(r"Answer\s*[:：]\s*([^\n]+)", text)
    return number(m[-1]) if m else number(text[-200:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-file")
    p.add_argument("--mmlu", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--mmlu-n", type=int, default=200)
    p.add_argument("--gsm8k-n", type=int, default=100)
    p.add_argument("--modes", default="off,on")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--max-tokens-on", type=int, default=8192)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = None
    if a.key_file:
        line = [x for x in open(a.key_file).read().splitlines() if x.strip() and not x.startswith("#")][0]
        key = line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line.strip()
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}

    mmlu_all = [json.loads(x) for x in open(a.mmlu) if x.strip()]
    gsm_all = [json.loads(x) for x in open(a.gsm8k) if x.strip()]
    mi = sorted(random.Random(SEED).sample(range(len(mmlu_all)), a.mmlu_n))
    gi = sorted(random.Random(SEED + 1).sample(range(len(gsm_all)), a.gsm8k_n))
    digest = hashlib.sha256(json.dumps([[mmlu_all[i] for i in mi], [gsm_all[i]["question"] for i in gi]],
                                       sort_keys=True).encode()).hexdigest()[:16]

    def post(body):
        req = urllib.request.Request(a.base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), h)
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"error": e.code, "body": e.read().decode()[:200]}
        except Exception as e:  # noqa: BLE001 - a failed request is a wrong answer, recorded
            return {"error": type(e).__name__, "body": str(e)[:200]}

    jobs = []
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        for i in mi:
            jobs.append(("mmlu", mode, i))
        for i in gi:
            jobs.append(("gsm8k", mode, i))
    out, lock = {}, threading.Lock()

    def worker():
        while True:
            with lock:
                if not jobs:
                    return
                task, mode, i = jobs.pop(0)
            think = mode == "on"
            if task == "mmlu":
                r = mmlu_all[i]
                prompt, gold = mmlu_prompt(r, think), LETTERS[r["answer"]]
                cap = a.max_tokens_on if think else 16
            else:
                r = gsm_all[i]
                prompt, gold = gsm_prompt(r), number(r["answer"].split("####")[-1])
                cap = a.max_tokens_on if think else 2048
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                    "max_tokens": cap, "chat_template_kwargs": {"enable_thinking": think}}
            t0 = time.time()
            res = post(body)
            ch = (res.get("choices") or [{}])[0]
            text = (ch.get("message") or {}).get("content") or ""
            pred = mmlu_pick(text, think) if task == "mmlu" else gsm_pick(text)
            u = res.get("usage") or {}
            row = {"task": task, "mode": mode, "index": i, "gold": gold, "pred": pred, "ok": pred == gold,
                   "finish": ch.get("finish_reason"), "completion_tokens": u.get("completion_tokens"),
                   "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   "s": round(time.time() - t0, 1), "error": res.get("error")}
            with lock:
                out[(task, mode, i)] = row
                done = len(out)
            if done % 25 == 0:
                print(json.dumps({"done": done}), flush=True)

    t0 = time.time()
    th = [threading.Thread(target=worker) for _ in range(a.parallel)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    summary = {}
    for (task, mode, _), row in sorted(out.items()):
        s = summary.setdefault(f"{task} thinking {mode}", {"n": 0, "correct": 0, "unparsed": 0, "length_cut": 0,
                                                             "errors": 0})
        s["n"] += 1
        s["correct"] += int(row["ok"])
        s["unparsed"] += int(row["pred"] is None)
        s["length_cut"] += int(row["finish"] == "length")
        s["errors"] += int(bool(row["error"]))
    for s in summary.values():
        s["accuracy_pct"] = round(100.0 * s["correct"] / max(1, s["n"]), 1)
    print(json.dumps({"summary": summary, "subset_digest": digest, "wall_s": round(time.time() - t0, 1)}), flush=True)
    json.dump({"summary": summary, "seed": SEED, "subset_digest": digest, "mmlu_indices": mi, "gsm8k_indices": gi,
               "rows": [out[k] for k in sorted(out)], "wall_s": round(time.time() - t0, 1)}, open(a.out, "w"),
              indent=1)


if __name__ == "__main__":
    main()
