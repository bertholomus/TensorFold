#!/bin/bash
# build_kit.sh OUT: the dsv41 family's kernel kit (the folder TF_DS_KIT and tf-dsv41-lanes read) built from source.
#
# Inputs, as environment variables:
#   IMAGE       the deepseek-v41-tp2 engine's container image (github.com/bertholomus/TensorFold, branch
#               deepseek-v41-tp2, installed in nvcr.io/nvidia/pytorch:26.07-py3), run on a GB10 (aarch64, sm_121)
#   EXT_DIR     that engine's torch-extension cache after its first start (TORCH_EXTENSIONS_DIR, holding
#               tensorfold_exl3_linear_v7/ and tensorfold_exl3_experts_v19/); tools/dsv41-zig/build_inputs.sh makes it
#   MODEL       the EXL3 checkpoint (Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
#   ORIGINAL    the original checkpoint (deepseek-ai/DeepSeek-V4.1-Flash), for the image routing bias
#   TOKEN_MAP   the Engram token map the engine wrote on its first start (build_inputs.sh)
#   RECORDINGS  space-separated recording dirs (each with rank0/ and rank1/: launches.json, triton/), made by running
#               the engine with tools/dsv41-zig/rec on PYTHONPATH and TF_ZREC_DIR=<dir>/rank<R> (see zrec.py); every
#               request shape you serve should be in one of them (a launch with no captured variant runs the
#               closest more general one, cuda/aot.zig)
#   MANIFEST    optional: a kit MANIFEST (sha256sum format) to check the result against
# Steps (no GPU except the recordings): 1 the EXL3 extension cubins and 2 PyTorch's attention cubin, extracted with
# cuobjdump; 3 Engram's constants and the RoPE tables (zrec_fixtures.py, CPU); 4 the image routing bias
# (make_bias_vl.py); 5 the Triton kernel set packed from the recordings (triton_aot_manifest.py + aot_pack.py).
# Byte-identical to a published kit: steps 2-4 always; step 1's linear cubins and step 5 given the same .so files and
# recordings. nvcc names anonymous-namespace symbols with a per-build hash, so experts cubins from a fresh extension
# build differ from the published ones in those bytes only.
set -eu
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
T=$(cd "$(dirname "$0")" && pwd)            # tools/dsv41-zig
ROOT=$(cd "$T/../.." && pwd)
: "${IMAGE:?}" "${EXT_DIR:?}" "${MODEL:?}" "${ORIGINAL:?}" "${TOKEN_MAP:?}" "${RECORDINGS:?}"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
run() { docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$W":/w -v "$OUT":/out \
  -v "$EXT_DIR":/ext:ro -v "$MODEL":/model:ro -v "$TOKEN_MAP":/token_map.json:ro -v "$ROOT":/tf:ro -w /w "$IMAGE" "$@"; }
mkdir -p "$OUT/cubins" "$OUT/vision" "$OUT/aot"

echo "1/5 EXL3 extension cubins"
run bash -c 'set -e; mkdir -p l e; cd l; cuobjdump -xelf all /ext/tensorfold_exl3_linear_v7/tensorfold_exl3_linear_v7.so >/dev/null
  cd ../e; cuobjdump -xelf all /ext/tensorfold_exl3_experts_v19/tensorfold_exl3_experts_v19.so >/dev/null; cd ..
  for f in l/*.sm_121.cubin; do
    if cuobjdump -symbols "$f" | grep -q glinear; then cp "$f" /out/cubins/linear_grouped.cubin; else cp "$f" /out/cubins/linear.cubin; fi
  done
  cp e/experts.sm_121.cubin /out/cubins/experts.cubin
  cp e/experts_cb2.sm_121.cubin /out/cubins/experts_cb.cubin'

echo "2/5 PyTorch's attention cubin (sm_120)"
run bash -c 'set -e; mkdir -p t; cd t; cuobjdump -xelf all "$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\", \"libtorch_cuda.so\"))")" >/dev/null
  K=_ZN39fmha_cutlassF_f32_aligned_64x64_rf_sm80N22PyTorchMemEffAttention15AttentionKernelIfN7cutlass4arch4Sm80ELb1ELi64ELi64ELi64ELb1ELb1EE6ParamsE
  f=$(grep -l "$K" *.sm_120.cubin | head -1); [ -n "$f" ]; cp "$f" /out/vision/torch_fmha_sm120.cubin'

echo "3/5 Engram constants and RoPE tables"
run bash -c 'set -e; python3 /tf/tools/dsv41-zig/rec/zrec_fixtures.py /model /token_map.json /w/fx >/dev/null
  cp /w/fx/engram.json /w/fx/rope.json /w/fx/rope-*.f32 /out/; cp /w/fx/jit.json /w/jit.json'

echo "4/5 image routing bias"
python3 "$T/kit/make_bias_vl.py" "$ORIGINAL" "$OUT/vision/gate_bias_vl.safetensors"

echo "5/5 Triton kernel set from the recordings"
args=()
for d in $RECORDINGS; do
  for r in 0 1; do
    python3 "$ROOT/tools/zig/triton_aot_manifest.py" --cache "$d/rank$r/triton" --launches "$d/rank$r/launches.json" \
      --mount "$d/rank$r/triton" --out "$W/manifest-$(basename "$d")-$r.json"
    args+=(--manifest "$W/manifest-$(basename "$d")-$r.json" --cache "$d/rank$r/triton")
  done
done
python3 "$ROOT/zig/tests/cuda/nemotron/aot_pack.py" "${args[@]}" --jit "$W/jit.json" --out "$OUT/aot"

if [ -n "${MANIFEST:-}" ]; then
  echo "checking against $MANIFEST"
  (cd "$OUT" && grep -E '  (aot/|cubins/|engram.json|rope|vision/)' "$MANIFEST" | sha256sum -c --quiet) && echo "kit equals MANIFEST"
fi
echo "kit in $OUT"
