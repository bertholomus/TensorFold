#!/bin/bash
# node_inputs.sh MODEL_DIR CACHE_DIR RANK [WORLD]: one node's run-time inputs for the DeepSeek-V4.1-Flash lane, built
# from the EXL3 checkpoint alone (Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw), no GPU:
#   CACHE_DIR/rank-cache/rank<RANK>of<WORLD>-<hash>.bin  the node's weight file (TF_DS_RANK_CACHE): its slices of the
#       checkpoint in the loader's order, by tf-dsv41-rank-cache; on four nodes the split with the balanced experts
#       (TF_DS_2D_GU=parity) that the served launch runs;
#   CACHE_DIR/dsv41_token_map.json  Engram's compressed token map (TF_DS_TOKEN_MAP), by make_token_map.py.
# WORLD is 4 unless given (RANK 0-3, in the order of the recipe's NODES). CACHE_DIR/rank-cache must not hold another
# file of this rank. On four nodes the weight file is about 56 GB a node; the checkpoint is read a tensor at a time.
#
# Environment:
#   RANK_CACHE_BIN  tf-dsv41-rank-cache; default: the one next to this script, else this checkout's zig-out/bin (zig
#                   build on Linux installs it; it needs no CUDA)
#   PYTHON          a Python with the `tokenizers` package (default python3); without one the token map is made with
#   IMAGE           docker in this image (default nvcr.io/nvidia/pytorch:26.07-py3, the serving container)
set -euo pipefail
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
[ $# -ge 3 ] && [ $# -le 4 ] || usage
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=$(cd "$1" && pwd); CACHE_DIR=$2; RANK=$3; WORLD=${4:-4}
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
mkdir -p "$CACHE_DIR/rank-cache"
CACHE_DIR=$(cd "$CACHE_DIR" && pwd)
PARITY=; [ "$WORLD" = 4 ] && PARITY=--parity

echo "rank $RANK of $WORLD: the weight file into $CACHE_DIR/rank-cache" >&2
"$BIN" "$MODEL_DIR" "$CACHE_DIR/rank-cache" "$RANK" "$WORLD" $PARITY

echo "the token map into $CACHE_DIR/dsv41_token_map.json" >&2
if "$PYTHON" -c 'import tokenizers' 2>/dev/null; then
  "$PYTHON" "$HERE/make_token_map.py" "$MODEL_DIR" "$CACHE_DIR/dsv41_token_map.json"
else
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$MODEL_DIR:/model:ro" \
    -v "$CACHE_DIR:/cache" -v "$HERE:/tools:ro" --entrypoint python3 "$IMAGE" \
    /tools/make_token_map.py /model /cache/dsv41_token_map.json
fi
