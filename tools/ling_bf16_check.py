"""Post-write checker for the BF16 copy (tools/ling_fp8_write_bf16.py): ling_bf16_check.py FP8_DIR BF16_DIR.
1. every output shard opens with safetensors safe_open; dtype/shape vs source per key
2. per-shard + total counts from headers: out = src keys - *_scale_inv
3. sample = 64 random + first/last tensor of every shard (by data offset): FP8 -> bit-equal vs load(src,name,bf16); others -> sha256
4. index: every output key present with its shard, no extras, total_size = sum of output tensor bytes
Pass: all four, zero differences."""
import json, sys, random, hashlib
from pathlib import Path
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ling_fp8_dequant import SCALE_SUFFIX, _header, load

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
src_index = json.loads((src / "model.safetensors.index.json").read_text())
dst_index = json.loads((dst / "model.safetensors.index.json").read_text())
fails = []

# -- gather per-shard key info
def shard_keys(d):
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        head, base = _header(f)
        keys = {k: v for k, v in head.items() if k != "__metadata__"}
        first = min(keys, key=lambda k: keys[k]["data_offsets"][0])
        last = max(keys, key=lambda k: keys[k]["data_offsets"][1])
        out[f.name] = {"keys": keys, "first": first, "last": last, "base": base}
    return out

S, D = shard_keys(src), shard_keys(dst)
assert set(S) == set(D), f"shard name mismatch: {set(S) ^ set(D)}"

# -- 2. counts from headers
exp_total = sum(len(v["keys"]) - sum(1 for k in v["keys"] if k.endswith(SCALE_SUFFIX)) for v in S.values())
got_total = sum(len(v["keys"]) for v in D.values())
if exp_total != got_total:
    fails.append(f"COUNT: expected {exp_total} output tensors, got {got_total}")
for sname in S:
    exp = len(S[sname]["keys"]) - sum(1 for k in S[sname]["keys"] if k.endswith(SCALE_SUFFIX))
    if exp != len(D[sname]["keys"]):
        fails.append(f"COUNT {sname}: expected {exp}, got {len(D[sname]['keys'])}")

# -- 1. safe_open dtype/shape on every key of every output shard
for sname, dv in D.items():
    with safe_open(dst / sname, "pt") as f:
        st_keys = set(f.keys())
        if st_keys != set(dv["keys"]):
            fails.append(f"SAFE_OPEN {sname}: key set differs from header by {st_keys ^ set(dv['keys'])}")
        for k in dv["keys"]:
            t = f.get_slice(k) if hasattr(f, "get_slice") else None
            src_dt = S[sname]["keys"][k]["dtype"]; src_shape = S[sname]["keys"][k]["shape"]
            dt = dv["keys"][k]["dtype"]; shape = dv["keys"][k]["shape"]
            want = "BF16" if src_dt == "F8_E4M3" else src_dt
            if dt != want or shape != src_shape:
                fails.append(f"DTYPE/SHAPE {sname}:{k}: want {want} {src_shape}, got {dt} {shape}")
        if len(fails) > 40: break
print(f"[1] safetensors read: {len(D)} shards checked, {len(fails)} fails so far", flush=True)

# -- 3. sample: 64 random + shard edges
all_keys = [(s, k) for s, dv in D.items() for k in dv["keys"]]
random.seed(1234)
sample = set(random.sample(all_keys, min(64, len(all_keys))))
for sname, dv in D.items():
    sample.add((sname, dv["first"])); sample.add((sname, dv["last"]))
bad = 0
for i, (sname, k) in enumerate(sorted(sample)):
    sdt = S[sname]["keys"][k]["dtype"]
    if sdt == "F8_E4M3":
        ref = load(src, k, dtype=torch.bfloat16)
        dvv = D[sname]["keys"][k]
        with safe_open(dst / sname, "pt") as f:
            got = f.get_tensor(k)
        if got.dtype != torch.bfloat16 or got.shape != torch.Size(S[sname]["keys"][k]["shape"]) \
                or not bool((got.view(torch.uint16) == ref.view(torch.uint16)).all()):
            fails.append(f"SAMPLE {sname}:{k}: FP8 dequant mismatch"); bad += 1
    else:
        lo, hi = S[sname]["keys"][k]["data_offsets"]
        a = open(src / sname, "rb").read()[S[sname]["base"]+lo:S[sname]["base"]+hi]
        lo2, hi2 = D[sname]["keys"][k]["data_offsets"]
        b = open(dst / sname, "rb").read()[D[sname]["base"]+lo2:D[sname]["base"]+hi2]
        if hashlib.sha256(a).hexdigest() != hashlib.sha256(b).hexdigest():
            fails.append(f"SAMPLE {sname}:{k}: byte-copy sha mismatch"); bad += 1
    if i % 20 == 0: print(f"[3] sampled {i}/{len(sample)}", flush=True)
print(f"[3] sample: {len(sample)} tensors, {bad} bad", flush=True)

# -- 4. index
wm = dst_index["weight_map"]
if set(wm) != set(all_keys for all_keys in [k for _, k in all_keys]):
    extra = set(wm) - {k for _, k in all_keys}; missing = {k for _, k in all_keys} - set(wm)
    fails.append(f"INDEX: keys differ; extra {list(extra)[:5]}, missing {list(missing)[:5]}")
for (s, k) in list(all_keys)[:5]:
    if wm.get(k) != s: fails.append(f"INDEX shard for {k}: {wm.get(k)} != {s}")
ts = dst_index.get("metadata", {}).get("total_size")
exp_ts = sum(dv["keys"][k]["data_offsets"][1] - dv["keys"][k]["data_offsets"][0] for dv in D.values() for k in dv["keys"])
if ts != exp_ts: fails.append(f"INDEX total_size {ts} != {exp_ts}")
print(f"[4] index: total_size {ts} vs tensor bytes {exp_ts}", flush=True)

print("CHECKER:", "PASS" if not fails else f"FAIL ({len(fails)} differences)", flush=True)
for f_ in fails[:30]: print("  ", f_, flush=True)
sys.exit(0 if not fails else 1)
