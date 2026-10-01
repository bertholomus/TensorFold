"""Build a truncated GLM-5.3 EXL3 checkpoint (first N decoder layers, no MTP) for engine bring-up.

Keeps embed, layers 0..N-1, final norm, lm_head; rewrites config (num_hidden_layers=N, per-layer lists cut,
num_nextn_predict_layers=0). Tensors are copied byte-for-byte (TensorFold split.write, pure numpy).
usage: python3 make_trunc.py SRC OUT N
"""
import json, os, re, shutil, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ai/glm53-tf/TensorFold/src") if os.path.exists(os.path.expanduser("~/ai/glm53-tf/TensorFold/src")) else "/tf/src")
from tensorfold.families.glm5_next.cuda.split import read_header, write

src, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
lay = re.compile(r"^model\.layers\.(\d+)\.")


def keep(n):
    m = lay.match(n)
    return int(m.group(1)) < N if m else True


by_file = {}
for n, f in idx.items():
    if keep(n):
        by_file.setdefault(f, []).append(n)
wm = {}
for i, (f, names) in enumerate(sorted(by_file.items())):
    h, base = read_header(f"{src}/{f}")
    meta = h.pop("__metadata__", None)
    mm = np.memmap(f"{src}/{f}", dtype=np.uint8, mode="r")
    part = []
    for n in sorted(names, key=lambda k: h[k]["data_offsets"][0]):
        a, b = h[n]["data_offsets"]
        part.append((n, h[n]["dtype"], h[n]["shape"], np.array(mm[base + a:base + b])))
    name = f"model-{i + 1:05d}-of-{len(by_file):05d}.safetensors"
    write(f"{out}/{name}", part, meta)
    for n in names:
        wm[n] = name
    print(name, len(part), flush=True)
json.dump({"metadata": {}, "weight_map": wm}, open(f"{out}/model.safetensors.index.json", "w"), indent=1)
c = json.load(open(f"{src}/config.json"))
c["num_hidden_layers"] = N
for k in ("indexer_types", "mlp_layer_types"):
    if isinstance(c.get(k), list):
        c[k] = c[k][:N]
c["num_nextn_predict_layers"] = 0
json.dump(c, open(f"{out}/config.json", "w"), indent=1)
for n in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "quantization_config.json"):
    if os.path.exists(f"{src}/{n}"):
        shutil.copyfile(f"{src}/{n}", f"{out}/{n}")
print("DONE", len(wm), "tensors")
