#!/usr/bin/env python3
"""pack_kit.py SRC_KIT OUT_KIT [--same-as SUMS] [--title TEXT]: the release copy of a lane's kernel kit, with its
MANIFEST and SHA256SUMS. Standard library only.

SRC_KIT is the kit folder a lane loads (TF_DS_KIT: aot/, cubins/, engram.json, the RoPE tables and rope.json, vision/).
Every file is copied to OUT_KIT (an empty or new folder) and listed in OUT_KIT/MANIFEST (sha256, bytes, origin, the
two-node flag, path) and OUT_KIT/SHA256SUMS (for `sha256sum -c`). Origin says how a cloner gets the file: `git`, kept in
the recipe repository; `rope`, made by make_rope.py from the checkpoint's config (the four 136 MB tables); `bias`, made
by make_bias_vl.py from DeepSeek's original checkpoint (model weights, never committed). Any other file over 50 MB stops
the pack: a file that big is generated, not committed. --same-as: a SHA256SUMS-style list of another kit (paths with or
without a leading `./` or `kit/`); a file with the same path and digest there is marked `tp2` in MANIFEST."""
import hashlib
import os
import shutil
import sys

GENERATED = {
    "rope-plain-cos.f32": "rope", "rope-plain-sin.f32": "rope",
    "rope-compressed-cos.f32": "rope", "rope-compressed-sin.f32": "rope",
    "vision/gate_bias_vl.safetensors": "bias",
}
LIMIT = 50 * 1000 * 1000
OWN = {"MANIFEST", "SHA256SUMS"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sums(path):
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        sha, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        for p in ("./", "kit/"):
            if name.startswith(p):
                name = name[len(p):]
        out[name] = sha
    return out


def main():
    args, opts, i = [], {}, 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a in ("--same-as", "--title"):
            opts[a] = sys.argv[i + 1]
            i += 2
            continue
        args.append(a)
        i += 1
    if len(args) != 2:
        sys.exit(__doc__)
    src, out = args
    same = sums(opts["--same-as"]) if "--same-as" in opts else {}
    files = sorted(os.path.relpath(os.path.join(d, f), src) for d, _, fs in os.walk(src) for f in fs)
    files = [f for f in files if f not in OWN]
    os.makedirs(out, exist_ok=True)
    if any(os.scandir(out)):
        sys.exit(f"{out} is not empty")
    rows, big = [], []
    for f in files:
        size = os.path.getsize(os.path.join(src, f))
        origin = GENERATED.get(f, "git")
        if origin == "git" and size > LIMIT:
            big.append(f"{f} ({size} bytes)")
        sha = digest(os.path.join(src, f))
        rows.append((sha, size, origin, "tp2" if same.get(f) == sha else "-", f))
    if big:
        sys.exit("files over 50 MB that no script makes: " + ", ".join(big))
    for sha, _, _, _, f in rows:
        dst = os.path.join(out, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(src, f), dst)
        if digest(dst) != sha:
            sys.exit(f"{f}: the copy differs")
    title = opts.get("--title", "the kernel kit")
    n = {o: sum(1 for r in rows if r[2] == o) for o in ("git", "rope", "bias")}
    with open(os.path.join(out, "MANIFEST"), "w") as m:
        m.write(f"# {title}\n")
        m.write(f"# {len(rows)} files, {sum(r[1] for r in rows)} bytes: every file the lane loads from its kit folder.\n")
        m.write("# Columns: sha256, bytes, origin, two-node, path. Origin: git = in the recipe repository "
                f"({n['git']} files); rope = made by tools/make_rope.py ({n['rope']}); bias = made by "
                f"tools/make_bias_vl.py from DeepSeek's original checkpoint ({n['bias']}). verify_kit.sh makes and checks "
                "them all.\n")
        m.write(f"# Two-node: tp2 = byte for byte the same file in our two-node kit v0.6.0 "
                f"({sum(1 for r in rows if r[3] == 'tp2')} files); - = this lane's own.\n")
        for sha, size, origin, tp2, f in rows:
            m.write(f"{sha}  {size:>10}  {origin:<4}  {tp2:<3}  {f}\n")
    with open(os.path.join(out, "SHA256SUMS"), "w") as s:
        for sha, _, _, _, f in rows:
            s.write(f"{sha}  {f}\n")
    print(f"{out}: {len(rows)} files, {sum(r[1] for r in rows)} bytes; git {n['git']} "
          f"({sum(r[1] for r in rows if r[2] == 'git')} bytes), rope {n['rope']}, bias {n['bias']}; "
          f"tp2 {sum(1 for r in rows if r[3] == 'tp2')}")


if __name__ == "__main__":
    main()
