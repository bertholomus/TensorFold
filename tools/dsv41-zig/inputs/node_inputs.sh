#!/bin/bash
# node_inputs.sh MODEL_DIR CACHE_DIR RANK [WORLD]: one node's run-time inputs for the DeepSeek-V4.1-Flash lane, built
# from the EXL3 checkpoint alone, no GPU, so a node can make them while its lane keeps serving:
#   CACHE_DIR/tf-ds-rank/rank<RANK>of<WORLD>-<hash>.bin  the node's weight file (TF_DS_RANK_CACHE=CACHE_DIR/tf-ds-rank):
#       its slices of the checkpoint in the loader's order, by tf-dsv41-rank-cache;
#   CACHE_DIR/dsv41_token_map.json  Engram's compressed token map (TF_DS_TOKEN_MAP), by make_token_map.py.
# They are the files build_inputs.sh has the Python engine write on the GPUs. WORLD is 2 unless given (RANK 0 on the
# head node, 1 on the other); this branch writes one or two nodes only (four nodes: branch deepseek-v41-zig-tp4).
# CACHE_DIR/tf-ds-rank must not hold another file of this rank. On two nodes the weight file is about 106 GB a node for
# a 2.9 bpw checkpoint; the checkpoint is read a tensor at a time.
#
# Environment:
#   RANK_CACHE_BIN  tf-dsv41-rank-cache; default: the one next to this script, else this checkout's zig-out/bin (zig
#                   build on Linux installs it; it needs no CUDA)
#   PYTHON          a Python with the `tokenizers` package (default python3); without one the token map is made with
#   IMAGE           docker in this image (default nvcr.io/nvidia/pytorch:26.07-py3, the serving container)
set -euo pipefail
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
[ $# -ge 3 ] && [ $# -le 4 ] || usage
RANK=$3; WORLD=${4:-2}
case $WORLD in
  1|2) ;;
  *) echo "node_inputs.sh: WORLD is 1 or 2 on this two-node branch; four nodes: use branch deepseek-v41-zig-tp4" >&2
     exit 2 ;;
esac
case $RANK in
  ''|*[!0-9]*) echo "node_inputs.sh: RANK is 0 .. WORLD - 1" >&2; exit 2 ;;
esac
[ "$RANK" -lt "$WORLD" ] || { echo "node_inputs.sh: RANK is 0 .. WORLD - 1" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=$(cd "$1" && pwd); CACHE_DIR=$2
BIN=${RANK_CACHE_BIN:-}
if [ -z "$BIN" ]; then
  BIN=$HERE/tf-dsv41-rank-cache
  [ -x "$BIN" ] || BIN=$HERE/../../../zig-out/bin/tf-dsv41-rank-cache
fi
PYTHON=${PYTHON:-python3}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
if [ ! -x "$BIN" ]; then
  echo "no tf-dsv41-rank-cache next to $0 or in zig-out/bin: zig build, or set RANK_CACHE_BIN" >&2; exit 1
fi
for f in config.json tokenizer.json; do
  [ -f "$MODEL_DIR/$f" ] || { echo "$MODEL_DIR has no $f: MODEL_DIR is the EXL3 checkpoint" >&2; exit 1; }
done
mkdir -p "$CACHE_DIR/tf-ds-rank"
CACHE_DIR=$(cd "$CACHE_DIR" && pwd)

echo "rank $RANK of $WORLD: the weight file into $CACHE_DIR/tf-ds-rank" >&2
"$BIN" "$MODEL_DIR" "$CACHE_DIR/tf-ds-rank" "$RANK" "$WORLD"

echo "the token map into $CACHE_DIR/dsv41_token_map.json" >&2
if "$PYTHON" -c 'import tokenizers' 2>/dev/null; then
  "$PYTHON" "$HERE/make_token_map.py" "$MODEL_DIR" "$CACHE_DIR/dsv41_token_map.json"
else
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$MODEL_DIR:/model:ro" \
    -v "$CACHE_DIR:/cache" -v "$HERE:/tools:ro" --entrypoint python3 "$IMAGE" \
    /tools/make_token_map.py /model /cache/dsv41_token_map.json
fi
