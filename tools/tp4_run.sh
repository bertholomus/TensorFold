#!/bin/bash
# Run one command on every TP4 rank (spark1..spark4, rank 0 = spark1) in throwaway containers, e.g. tf_profile.py.
# "{rank}" in CMD becomes each node's rank. Stop the tf-tp4 lane first: each rank needs the GPU's memory.
# usage: tp4_run.sh NAME "CMD"     (logs: ssh spark1 docker logs NAME; NCCL_DEBUG=INFO tp4_run.sh ... for transport)
NAME=$1
CMD=$2
# RDMA device names: some OS images names every node's ports rocep1s0f0 / roceP2p1s0f0; DGX OS named
# the first two nodes' ports mlx5_0 / mlx5_2. Read them from each node as tp4_start.sh does.
declare -A HCA
for n in 1 2 3 4; do
  HCA[$n]=$(ssh -o BatchMode=yes -n spark$n 'd=$(ls /sys/class/infiniband); for p in rocep1s0f0 roceP2p1s0f0; do echo "$d" | grep -qx $p && printf "%s," $p; done; [ -z "$(echo "$d" | grep -x rocep1s0f0)" ] && printf "mlx5_0,mlx5_2,"' 2>/dev/null | sed 's/,$//')
done
# the engine's TF_GLM_* settings in this shell (e.g. TF_GLM_KV=fp8) go to every rank's container
GLMENV=$(env | grep -E '^TF_GLM_[A-Z0-9_]+=[A-Za-z0-9_.,:+-]*$' | tr '\n' ' ')
for n in 4 3 2 1; do
  r=$((n - 1))
  cmd=${CMD//\{rank\}/$r}
  ssh -o BatchMode=yes -n spark$n "cd ~/ai/glm53-tf && docker rm -f $NAME >/dev/null 2>&1; NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=${HCA[$n]} NCCL_NET_PLUGIN=spcx NCCL_DEBUG=${NCCL_DEBUG:-WARN} TF_NCCL_GATHER=${TF_NCCL_GATHER:-} NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-} $GLMENV TF_TP_WORLD=4 ./tfrun.sh $NAME \"$cmd\" >/dev/null && echo \$(hostname) rank $r started" 2>&1 | grep -v local/bin/env
done
