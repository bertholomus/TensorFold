#!/bin/bash
# In the tf container on every rank (tp4_run.sh): nccl_bench.py once per NCCL setting below, a port each.
# Entries are "ENV ASSIGNMENTS|MODE" (MODE allgather or p2p); "base" is the env tp4_start.sh gives; @DUAL is this
# node's two active RoCE devices (mlx5_0,mlx5_2 on nodes 1-2; rocep1s0f0,roceP2p1s0f0 on nodes 3-4).
# usage: nccl_sweep.sh RANK [FIRST_PORT]
RANK=$1
PORT=${2:-29700}
HERE=$(dirname "$0")
if [ -e /sys/class/infiniband/mlx5_2 ]; then DUAL=mlx5_0,mlx5_2; else DUAL=rocep1s0f0,roceP2p1s0f0; fi
CONFIGS=(
  "NCCL_GRAPH_MIXING_SUPPORT=0|p2p"
  "NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_IB_HCA=@DUAL|p2p"
  "NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_IB_HCA=@DUAL NCCL_NCHANNELS_PER_NET_PEER=2|p2p"
  "NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_IB_HCA=@DUAL NCCL_NCHANNELS_PER_NET_PEER=4|p2p"
  "NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_IB_HCA=@DUAL|allgather"
)
for entry in "${CONFIGS[@]}"; do
  cfg=${entry%|*}
  mode=${entry#*|}
  vars=""
  [ "$cfg" != "base" ] && vars="${cfg//@DUAL/$DUAL}"
  env $vars timeout 180 python3 "$HERE/nccl_bench.py" "$RANK" 4 10.0.0.1 "$PORT" "$cfg" "$mode" 2>&1 \
    | grep -E "^\[|Error|error|assert" | grep -v "NCCL WARN"
  PORT=$((PORT + 1))
  sleep 2
done
echo "sweep done"
