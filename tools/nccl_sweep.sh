#!/bin/bash
# In the tf container on every rank (tp4_run.sh): nccl_bench.py once per NCCL setting below, a port each.
# Entries are "ENV ASSIGNMENTS|MODE" (MODE allgather or p2p); "base" is the env tp4_start.sh gives.
# usage: nccl_sweep.sh RANK [FIRST_PORT]
RANK=$1
PORT=${2:-29700}
HERE=$(dirname "$0")
CONFIGS=(
  "base|allgather"
  "base|p2p"
  "NCCL_GRAPH_MIXING_SUPPORT=0|allgather"
  "NCCL_GRAPH_MIXING_SUPPORT=0|p2p"
  "NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_ALGO=PAT|allgather"
  "NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_P2P_LL_THRESHOLD=131072|p2p"
  "NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_P2P_LL_THRESHOLD=0|p2p"
  "NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_NCHANNELS_PER_NET_PEER=2|p2p"
)
for entry in "${CONFIGS[@]}"; do
  cfg=${entry%|*}
  mode=${entry#*|}
  vars=""
  [ "$cfg" != "base" ] && vars="$cfg"
  env $vars timeout 180 python3 "$HERE/nccl_bench.py" "$RANK" 4 10.200.10.1 "$PORT" "$cfg" "$mode" 2>&1 \
    | grep -E "^\[|Error|error|assert" | grep -v "NCCL WARN"
  PORT=$((PORT + 1))
  sleep 2
done
echo "sweep done"
