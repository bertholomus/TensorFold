#!/bin/bash
# Start full GLM-5.3 EXL3 3.0bpw on TensorFold TP4 across ascent-01..04 (throwaway containers, no restart policy).
# Rank 0 = A1 (HTTP on 127.0.0.1:18090, no auth: loopback only until a keyed door is decided), master 10.200.10.1.
# usage: tp4_start.sh [CONTEXT=32768] [EXTRA_FLAGS]
CTX=${1:-32768}
EXTRA=${2:---no-drafts}
M=/models/glm53-exl3-3.0bpw-full
declare -A HCA=([1]=mlx5_0 [2]=mlx5_0 [3]=rocep1s0f0 [4]=rocep1s0f0)
for n in 4 3 2 1; do
  r=$((n - 1))
  if [ $r -eq 0 ]; then
    cmd="tensorfold serve $M --tp 4 --rank 0 --master 10.200.10.1 --master-port 29661 $EXTRA --context $CTX --host 127.0.0.1 --port 18090 --name GLM-5.3-EXL3-3.0bpw --temperature 0"
  else
    cmd="tensorfold serve $M --tp 4 --rank $r --master 10.200.10.1 --master-port 29661 $EXTRA --context $CTX"
  fi
  ssh -o BatchMode=yes -n ascent-0$n "cd ~/ai/glm53-tf && docker rm -f tf-tp4 >/dev/null 2>&1; NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=${HCA[$n]} NCCL_NET_PLUGIN=spcx TF_NCCL_GATHER=${TF_NCCL_GATHER:-} NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-} TF_TP_WORLD=4 ./tfrun.sh tf-tp4 \"$cmd\" >/dev/null && echo \$(hostname) rank $r started" 2>&1 | grep -v local/bin/env
done
