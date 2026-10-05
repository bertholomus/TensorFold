"""The token ranking behind the drafter's cached Markov rows (cuda/markov_tokens.py): token frequencies of a code sample
and an English sample, tokenized with the checkpoint's tokenizer, each sample's frequencies normalized, then averaged
(--english-weight); ranked, the first --top written. The samples: ExLlamaV3's standard calibration text (--cal: its
conversion/standard_cal_data folder, MIT; c4 and wiki as English, code as code) and the serving image's own files
(sorted paths, byte caps: Python, C/C++ headers, JS/TS; Markdown/RST docs). Every fourth 16 KB chunk is held out.
Also the cache's hit rate (tokens with a slot) at several K on the held-out chunks and on saved replies.

  python3 markov_tokens_gen.py --tokenizer M/tokenizer.json --cal DIR [--write cuda/markov_tokens.py] [--top 16384]
                               [--replies a.json,...] --out F
"""

import argparse
import collections
import glob
import json
import os
from pathlib import Path

CODE = [("/usr/lib/python3.12/**/*.py", 12 << 20), ("/usr/local/lib/python3.12/dist-packages/**/*.py", 8 << 20),
        ("/usr/include/**/*.h", 2 << 20), ("/usr/include/**/*.hpp", 1 << 20),
        ("/usr/local/lib/python3.12/dist-packages/**/*.js", 1 << 20),
        ("/usr/local/lib/python3.12/dist-packages/**/*.ts", 1 << 20)]
ENGLISH = [("/usr/local/lib/python3.12/dist-packages/**/*.md", 4 << 20),
           ("/usr/local/lib/python3.12/dist-packages/**/*.rst", 4 << 20),
           ("/usr/share/doc/**/README*", 1 << 20), ("/usr/share/common-licenses/*", 1 << 20)]


CHUNK = 16 << 10


def chunks(spec, extra=()):
    """16 KB text chunks of the given files and of each pattern's sorted paths up to its byte cap: (train, held out =
    every fourth chunk)."""

    paths = list(extra)
    for pat, cap in spec:
        got = 0
        for f in sorted(glob.glob(pat, recursive=True)):
            if not os.path.isfile(f) or "/test" in f:
                continue
            n = os.path.getsize(f)
            if n == 0 or n > (1 << 20):
                continue
            paths.append(f)
            got += n
            if got >= cap:
                break
    train, held = [], []
    i = 0
    for f in paths:
        text = Path(f).read_text(errors="ignore")
        for o in range(0, len(text), CHUNK):
            (held if i % 4 == 0 else train).append(text[o:o + CHUNK])
            i += 1
    return train, held


def counts(tok, texts):
    c = collections.Counter()
    for t in texts:
        c.update(tok.encode(t, add_special_tokens=False).ids)
    return c


def hit_rate(ranked, ids, ks):
    pos = {t: i for i, t in enumerate(ranked)}
    n = max(len(ids), 1)
    return {k: round(sum(1 for t in ids if pos.get(t, 1 << 30) < k) / n, 4) for k in ks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--cal", required=True, help="ExLlamaV3's conversion/standard_cal_data folder")
    p.add_argument("--english-weight", type=float, default=0.5)
    p.add_argument("--top", type=int, default=16384)
    p.add_argument("--write")
    p.add_argument("--replies", default="")
    p.add_argument("--out")
    a = p.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    cal = Path(a.cal)
    ctrain, cheld = chunks(CODE, [cal / "code.utf8"])
    etrain, eheld = chunks(ENGLISH, [cal / "c4.utf8", cal / "wiki.utf8"])
    cc, ec = counts(tok, ctrain), counts(tok, etrain)
    ct, et = sum(cc.values()), sum(ec.values())
    we = a.english_weight
    score = collections.Counter()
    for t, n in cc.items():
        score[t] += (1 - we) * n / ct
    for t, n in ec.items():
        score[t] += we * n / et
    full = [t for t, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
    ranked = full[:a.top]
    ks = [1024, 2048, 4096, 8192, 16384, 32768]
    res = {"code_chunks": len(ctrain), "code_tokens": ct, "english_chunks": len(etrain), "english_tokens": et,
           "english_weight": we, "ranked": len(ranked), "hit_rate": {}}
    hc = [t for x in cheld for t in tok.encode(x, add_special_tokens=False).ids]
    he = [t for x in eheld for t in tok.encode(x, add_special_tokens=False).ids]
    res["hit_rate"]["held_out_code"] = {"tokens": len(hc), **hit_rate(full, hc, ks)}
    res["hit_rate"]["held_out_english"] = {"tokens": len(he), **hit_rate(full, he, ks)}
    for f in filter(None, a.replies.split(",")):
        d = json.loads(Path(f).read_text())
        name = Path(f).parent.name + "/" + Path(f).name
        if "solo" in d:                                     # parallel_check: token id lists a reply
            for mode in ("greedy", "t0.6"):
                ids = [t for k, v in d["solo"].items() if k.endswith(mode) for t in v]
                res["hit_rate"][f"{name}:{mode}"] = {"tokens": len(ids), **hit_rate(full, ids, ks)}
        elif "result" in d:                                 # kit_bench decode: the first 400 characters a reply
            for r in d["result"]:
                ids = tok.encode(r.get("sample", ""), add_special_tokens=False).ids
                res["hit_rate"][f"{name}:{r.get('set')}/{r.get('prompt')}"] = {"tokens": len(ids),
                                                                               **hit_rate(full, ids, ks)}
    print(json.dumps(res, indent=1), flush=True)
    if a.write:
        lines = [", ".join(str(t) for t in ranked[i:i + 16]) for i in range(0, len(ranked), 16)]
        body = ",\n    ".join(lines)
        Path(a.write).write_text(
            '"""Token ids by frequency in a code + English sample, most frequent first (tools/dsv41/markov_tokens_gen.py:\n'
            f'the checkpoint\'s tokenizer; {ct} code and {et} English tokens from ExLlamaV3\'s standard calibration text,\n'
            'MIT, and the serving image\'s own files). The drafter caches the Markov bias rows of the first\n'
            'TF_DS_MARKOV_CACHE of them (markov.py); the list changes only which steps read a cached row, never a draft.\n'
            'Generated; do not edit."""\n\n'
            f"TOKENS = (\n    {body},\n)\n")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
