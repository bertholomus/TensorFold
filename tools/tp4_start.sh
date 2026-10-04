#!/bin/bash
# Start full GLM-5.3 EXL3 3.0bpw on TensorFold TP4 across spark1..spark4 (throwaway containers, no restart policy).
# Rank 0 = spark1 (HTTP on 127.0.0.1:18090, no auth: loopback only until a keyed door is decided), master 10.0.0.1.
# usage: tp4_start.sh [CONTEXT=32768] [EXTRA_FLAGS]
CTX=${1:-32768}
EXTRA=${2:---no-drafts}
M=/models/glm53-exl3-3.0bpw-full
# RDMA device names: some OS images names every node's ports rocep1s0f0 / roceP2p1s0f0; DGX OS named
# the first two nodes' ports mlx5_0 / mlx5_2. Read them from each node so the script works on both OSes.
declare -A HCA
for n in 1 2 3 4; do
  HCA[$n]=$(ssh -o BatchMode=yes -n ascent-0$n 'd=$(ls /sys/class/infiniband); for p in rocep1s0f0 roceP2p1s0f0; do echo "$d" | grep -qx $p && printf "%s," $p; done; [ -z "$(echo "$d" | grep -x rocep1s0f0)" ] && printf "mlx5_0,mlx5_2,"' 2>/dev/null | sed 's/,$//')
done
# the engine's TF_GLM_* settings in this shell (e.g. TF_GLM_KV=fp8) go to every rank's container
GLMENV=$(env | grep -E '^TF_GLM_[A-Z0-9_]+=[A-Za-z0-9_.,:+-]*$' | tr '\n' ' ')
# stop every rank's old lane first and let the nodes take its ~100 GiB back: each rank's startup admission sizes the cache
# from free memory, and a just-stopped lane still holding some of it cost ~10k tokens of window (refused at 253952 once)
for n in 1 2 3 4; do ssh -o BatchMode=yes -n ascent-0$n 'docker rm -f tf-tp4 >/dev/null 2>&1' 2>&1 | grep -v local/bin/env; done
sleep 20
for n in 4 3 2 1; do
  r=$((n - 1))
  if [ $r -eq 0 ]; then
    cmd="tensorfold serve $M --tp 4 --rank 0 --master 10.0.0.1 --master-port 29661 $EXTRA --context $CTX --host 127.0.0.1 --port 18090 --name GLM-5.3-EXL3-3.0bpw --temperature 0"
  else
    cmd="tensorfold serve $M --tp 4 --rank $r --master 10.0.0.1 --master-port 29661 $EXTRA --context $CTX"
  fi
  ssh -o BatchMode=yes -n ascent-0$n "cd ~/ai/glm53-tf && docker rm -f tf-tp4 >/dev/null 2>&1; NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=${HCA[$n]} NCCL_NET_PLUGIN=spcx TF_NCCL_GATHER=${TF_NCCL_GATHER:-} NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-} $GLMENV TF_TP_WORLD=4 ./tfrun.sh tf-tp4 \"$cmd\" >/dev/null && echo \$(hostname) rank $r started" 2>&1 | grep -v local/bin/env
done
