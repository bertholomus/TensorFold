#!/bin/bash
# build_inputs.sh RANK HEAD_IP CACHE: one node's run-time inputs for the dsv41 family, made once per node. Run it on
# both nodes at the same time (rank 0 on the head first). Nothing it writes is committed anywhere: every file is
# generated from the public weights.
#
# The deepseek-v41-tp2 engine (github.com/bertholomus/TensorFold, branch deepseek-v41-tp2) writes these on its first
# start, and the Zig engine reads them:
#   CACHE/tf-ds-rank/          this rank's weights in one file (~106 GB a rank)   -> TF_DS_RANK_CACHE / CACHE_DIR
#   CACHE/dsv41_token_map.json the Engram token map (~0.9 MB)                     -> TF_DS_TOKEN_MAP / --token-map
#   CACHE/torch_extensions/    the EXL3 extensions build_kit.sh extracts its cubins from (EXT_DIR)
# This script starts that engine with CACHE as its cache, waits until rank 0 answers /v1/models, and stops it.
#
# Environment: IMAGE (the deepseek-v41-tp2 engine's image), MODEL (Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw),
# ENGRAM (a folder with deepseek-ai/DeepSeek-V4.1-Flash's shards 47 and 48), HCA (the RoCE ports, comma-separated),
# IFACE (the link's network interface), PORT (rank 0's HTTP port, default 18891).
set -eu
RANK=$1; HEAD=$2; CACHE=$(mkdir -p "$3" && cd "$3" && pwd)
: "${IMAGE:?}" "${MODEL:?}" "${ENGRAM:?}" "${HCA:?}" "${IFACE:?}"
PORT=${PORT:-18891}
NAME=dsv41-inputs-$RANK
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --device /dev/infiniband -v "$MODEL":/model:ro -v "$ENGRAM":/engram:ro -v "$CACHE":/root/.cache \
  -e TF_DS_REPLAY=1 -e TF_DS_PREFILL_CHUNK=2048 -e TF_DS_RANK_CACHE=/root/.cache/tf-ds-rank -e TF_DS_RANK_CACHE_READERS=32 \
  -e TF_DS_ENGRAM=/engram -e NCCL_IB_HCA="$HCA" -e NCCL_IB_GID_INDEX=5 -e NCCL_SOCKET_IFNAME="$IFACE" \
  "$IMAGE" tensorfold serve /model --tp 2 --rank "$RANK" --master "$HEAD" --host 127.0.0.1 --port "$PORT" \
  --context 1048576 --vision --parallel 4 --mtp-drafts 5 >/dev/null
echo "rank $RANK started; the first start writes the rank cache (this takes a while)"
if [ "$RANK" = 0 ]; then
  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models)" = 200 ]; do
    docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "rank 0 stopped: docker logs $NAME"; exit 1; }
    sleep 10
  done
  echo "rank 0 ready: inputs written"
else
  read -r -p "rank 1 is following; press Enter once rank 0 reports ready " _
fi
docker stop -t 30 $NAME >/dev/null; docker rm $NAME >/dev/null
ls -la "$CACHE" "$CACHE/tf-ds-rank"
