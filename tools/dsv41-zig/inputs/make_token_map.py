#!/usr/bin/env python3
"""make_token_map.py MODEL_DIR OUT_JSON: Engram's compressed token map (TF_DS_TOKEN_MAP), standalone.

Token id -> compressed id, where tokens whose text normalizes alike share one, as DeepSeek's engram.py builds it (the
DeepSeek-V4.1 Python family's ops.compressed_token_map, branch deepseek-v41-tp2, copied here): NFKC, NFD, accents
stripped, lower case, whitespace runs to one space, stripped; a token whose decoded text holds U+FFFD keys by its own
token string. MODEL_DIR holds tokenizer.json and config.json (the EXL3 checkpoint); the number of compressed ids must
equal the config's engram compressed vocabulary. Needs the `tokenizers` package (in nvcr.io/nvidia/pytorch:26.07-py3).
The file is the JSON list the Python family writes (json.dumps of the list)."""
import json
import os
import sys

from tokenizers import Regex, Tokenizer, normalizers


def compressed_token_map(tokenizer_json):
    tok = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new = {}
    lookup = [0] * n
    for tid in range(n):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(tid)
        else:
            nt = norm.normalize_str(text)
            key = nt if nt else text
        new = key_to_new.get(key)
        if new is None:
            new = len(key_to_new)
            key_to_new[key] = new
        lookup[tid] = new
    return lookup, len(key_to_new)


def main():
    model, out = sys.argv[1:3]
    raw = json.load(open(os.path.join(model, "config.json")))
    t = raw.get("text_config", raw)
    lookup, n = compressed_token_map(os.path.join(model, "tokenizer.json"))
    want = t.get("engram_compressed_vocab_size")
    if want is not None and n != want:
        sys.exit(f"compressed ids {n} != the config's {want}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps(lookup))
    print(f"{out}: {len(lookup)} token ids -> {n} compressed ids")


if __name__ == "__main__":
    main()
