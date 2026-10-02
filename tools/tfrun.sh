#!/bin/bash
# Run a TensorFold command in a throwaway pytorch container on this node (TF fork mounted at /tfw).
# usage: tfrun.sh NAME CMD...   (detached, --rm; logs: docker logs NAME)
NAME=$1; shift
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --rm --name "$NAME" --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK $( [ -d /dev/infiniband ] && echo --device /dev/infiniband ) \
  -v $HOME/ai/glm53-tf:/tfw -v $HOME/ai/models:/models:ro -v $HOME/ai/glm53-tf/cache:/root/.cache \
  -e PYTHONUNBUFFERED=1 -e TF_TP_WORLD=${TF_TP_WORLD:-1} -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-} -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-5} -e NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-} -e TF_EXTRA=${TF_EXTRA:-} -w /tfw \
  $( [ -n "$TF_NCCL_GATHER" ] && echo "-e TF_NCCL_GATHER=$TF_NCCL_GATHER" ) \
  $( [ -n "$NCCL_GRAPH_MIXING_SUPPORT" ] && echo "-e NCCL_GRAPH_MIXING_SUPPORT=$NCCL_GRAPH_MIXING_SUPPORT" ) \
  $(env | grep -oE '^TF_GLM_[A-Z0-9_]+=' | tr -d = | sed 's/^/-e /') \
  nvcr.io/nvidia/pytorch:26.07-py3 bash -c "pip install --no-deps --no-build-isolation -q -e /tfw/TensorFold >/dev/null 2>&1; $*"
