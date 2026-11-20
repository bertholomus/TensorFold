# SM121 non-div-16 AOT variants

Two general AOT records for the DeepSeek-V4.1 kit, for launches whose tile count is not a multiple of 16. They close the `n=600000` / `nt=9375` gap in the current TP2 and TP4 kits.

## Records

- `_index_score` `052d6c1ceb0c0134932d80c70ab723b84d5d13784d79962cbb47a4560c82c02d`
  - cubin SHA256 `f11eef966275e317522a6ec3b441898acb1a7e102434f1215d5a333707a61edb`, 99,880 bytes
  - `HAS_BASE=1`, `KEYS=0`, `OUT=*fp32`; `n.div16=false`, `nt.div16=false`
  - 4 warps, 1 CTA, shared 10240, PDL on, no global/profile scratch
- `_score_keys` `9ab1ce45d139729e01bf8351bd87df01bed63459cd2c0d8072656fd178e6e88f`
  - cubin SHA256 `e9b6faf7cbb6b37736df75ddac645fdfb49ca52edc5c49ac55dc23e5027bd030`, 38,904 bytes
  - `HAS_TMAX=1`, `BN=1024`, `TB=64`; `n.div16=false`, `nt.div16=false`
  - 4 warps, 1 CTA, shared 256, PDL on, no global/profile scratch

All pointer arguments keep the pointer-alignment, dtype and constexpr constraints of the existing records. The argument names, types and constexprs match the current wrappers in `zig/src/families/dsv41/tri_index.zig` (`indexScore` and `scoreKeys`).

## Why these are needed

The current TP2 kit has 366 records. Its matching float-score `_index_score` record `34247d211cd2...` and `_score_keys` `HAS_TMAX=1` record `df85d0b39e2d...` both require `n.div16` and `nt.div16`. At `n=600000`, `nt=ceil(n/64)=9375`; 9375 is not divisible by 16, so neither record fits and `aot.Set.find` returns `MissingTritonVariant`. An eager retry does not help: the variant is absent, not merely uncaptured.

The same two SM121 cubins apply to the TP2 and TP4 GB10 kits; their `aot.json` record sets differ, so append the records rather than replacing a manifest.

## Provenance

- Kernel source: `src/tensorfold/families/deepseek_v41/cuda/kernels.py` at Git `8e0966e2b09250abe2f6d9cc51edc20630e6eb1a`, SHA256 `723c3af1c9c1608f7c6439eb4e05d1cadfc7e633727a3b71f10552a67dd4a530`.
- Compiler: Triton 3.7.1+gitf797708c.nv26.7, CUDA 13.3 ptxas, aarch64, SM121.
- Compile launches: `R=3,n=599999` and `R=6,n=600000`; the general records are the two with `n.div16=false` and `nt.div16=false`.
- The records were produced through the existing Recorder, `triton_aot_manifest.py` and `aot_pack.py`; only the two general records are included here.

The current Zig branch does not carry the Python engine source, so these cubins are the tested artifacts, not a byte-reproducible build from the current branch. A rebuild needs the pinned source above and the pinned compiler.

## Adding them to a kit

Merge into a copy of the kit, then repack:

```python3
import json, pathlib, shutil
kit = pathlib.Path("kit")                 # the existing kit
frag = json.loads(pathlib.Path("aot-fragment.json").read_text())
aot = json.loads((kit / "aot/aot.json").read_text())
have = {k["hash"]: k for k in aot["kernels"]}
for k in frag["kernels"]:
    if k["hash"] in have:
        raise SystemExit(f"{k['hash']}: already present")
    aot["kernels"].append(k)
    shutil.copyfile(pathlib.Path("cubins") / f"{k['hash']}.cubin",
                    kit / "aot/cubins" / f"{k['hash']}.cubin")
(kit / "aot/aot.json").write_text(json.dumps(aot, indent=1) + "\n")
```

Then run `pack_kit.py` on the merged kit to write a fresh `MANIFEST` and `SHA256SUMS`, and `verify_kit.sh` to check them.

## Test

```sh
python3 -B tools/dsv41-zig/kit/test_nondiv16.py
```

The test checks the two records' hashes, metadata, parameter divisibility flags and cubin bytes. It is a metadata guard, not a GPU test.

## Not covered

- Full model/inference qualification; the GPU evidence is synthetic kernel and dispatch testing.
- The initiating CUDA error or CUDA graph capture recovery (TensorFold PR #62).
- The integer-key case in MiaAI-Lab issue #65.
- Arbitrary caps or every base/candidate/key mode.
