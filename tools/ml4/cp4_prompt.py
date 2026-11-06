"""Token files for tools/ml4/cp4_run.py: a chat prompt with the pack's own template, and the needle prompts.

    python3 tools/ml4/cp4_prompt.py PACK OUT_DIR smoke
    python3 tools/ml4/cp4_prompt.py PACK OUT_DIR needle 1048000      # fills the prompt to that many tokens

The needle text follows the phase-B builder's rule (fixed seed, three codes at 25/50/75 % depth), counted here with the
real tokenizer so the prompt lands on its target.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from jinja2 import Environment
from tokenizers import Tokenizer

SEED = 1105
WORDS = ("delta neutron harbor kettle lantern mosaic cactus umbrella vortex meadow compass gargoyle pebble saffron tundra "
         "whirl walnut zephyr basil falcon cipher dune ember fjord garnet hollow iris jade knoll lumen mica nectar opal "
         "prism quartz ridge sable thistle umber velvet willow xenon yarrow").split()
ASK = ("Here is a long text. Find the three registration codes and answer with only the three codes, in order of "
       "appearance.\n\n")


def chat(pack: Path, tok: Tokenizer, text: str) -> list[int]:
    cfg = json.loads((pack / "tokenizer_config.json").read_text())
    tmpl = Environment().from_string((pack / "chat_template.jinja").read_text())
    body = tmpl.render(messages=[{"role": "user", "content": text}], add_generation_prompt=True,
                       bos_token=cfg.get("bos_token") or "<s>", eos_token=cfg.get("eos_token") or "</s>",
                       tools=None, reasoning_effort="low")
    return tok.encode(body, add_special_tokens=False).ids


def needle(pack: Path, tok: Tokenizer, target: int, out: Path) -> None:
    rng = random.Random(SEED)
    per = len(tok.encode(" ".join(WORDS), add_special_tokens=False).ids) / len(WORDS)
    n = int(target / per)
    words = [rng.choice(WORDS) for _ in range(n)]
    codes = []
    for depth in (0.25, 0.50, 0.75):
        at = int(n * depth)
        code = f"NEEDLE{rng.randrange(1000, 9999)}"
        words[at:at + 8] = f"The registration code for this segment is {code}.".split()
        codes.append({"depth": depth, "code": code})
    text = ASK + " ".join(words)
    ids = chat(pack, tok, text)
    print(f"needle {target}: {len(ids)} tokens, codes {[c['code'] for c in codes]}", flush=True)
    (out / f"needle{target}.json").write_text(json.dumps(ids))
    (out / f"needle{target}.txt").write_text(text)
    (out / f"needle{target}.meta.json").write_text(json.dumps({"tokens": len(ids), "codes": codes}, indent=1))


def main() -> None:
    pack, out, what = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(pack / "tokenizer.json"))
    if what == "smoke":
        raw = tok.encode("The capital of France is", add_special_tokens=True).ids
        (out / "raw.json").write_text(json.dumps(raw))
        ids = chat(pack, tok, "What is the capital of France? Answer in one short sentence.")
        (out / "chat.json").write_text(json.dumps(ids))
        print("raw", raw, "chat", ids[:40], len(ids), repr(tok.decode(ids, skip_special_tokens=False))[:300], flush=True)
    else:
        needle(pack, tok, int(sys.argv[4]), out)


if __name__ == "__main__":
    main()
