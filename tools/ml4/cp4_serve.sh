#!/usr/bin/env bash
# One rank of the 1M lane, run on each of the four nodes: cp4_serve.sh <rank> <tag> <master-port> [tensorfold serve args].
# RoCE, the memory settings of DESIGN-CP4 section 10, a final rc line in the log. EXTRA is more docker arguments
# (-e TF_ML4_DP_LIMIT=2048 sends a short prompt down the striped path). The draft head is draft/mtp_dspark.safetensors
# in the pack; DRAFT=0 serves without it.
#   ML4_TREE     a tree holding this checkout at src/tensorfold, with logs/, and where the compiled extensions go (/w)
#   ML4_PACK     the EXL3 pack, the same path on every node
#   ML4_MASTER   rank 0's address on the fabric between the nodes
#   ML4_IFNAME   the fabric's network interface
#   ML4_HCA      the RDMA devices, comma separated
R="${ML4_TREE:?set ML4_TREE}"
PACK="${ML4_PACK:?set ML4_PACK}"
MASTER="${ML4_MASTER:?set ML4_MASTER}"
IFNAME="${ML4_IFNAME:?set ML4_IFNAME}"
HCA="${ML4_HCA:?set ML4_HCA}"
RANK="$1"; TAG="$2"; PORT="$3"; shift 3
LOG="$R/logs/$TAG-rank$RANK.log"
MODELS="$(dirname "$PACK")"
[ "${DRAFT:-1}" = 0 ] || EXTRA="-e TF_ML4_DRAFT=$PACK/draft/mtp_dspark.safetensors ${EXTRA:-}"
docker run --rm --name "ml4cp4-$TAG" --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --device /dev/infiniband \
  -e PYTHONPATH=/w/src/tensorfold/src -e TORCH_EXTENSIONS_DIR=/w/torch-ext -e TRITON_CACHE_DIR=/w/triton \
  -e EXL3_INT8_GEMV=1 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  -e NCCL_SOCKET_IFNAME=$IFNAME -e NCCL_IB_HCA=$HCA -e NCCL_IB_GID_INDEX=5 -e NCCL_DEBUG=WARN -e TF_RDMA_DEVICES=$HCA \
  -e NCCL_BUFFSIZE=1048576 -e NCCL_MAX_NCHANNELS=4 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TF_ML4_CP_BLOCK=64 -e TF_ML4_CP_ROUND=16384 -e TF_ML4_CP_PAIR=8192 -e TF_ML4_CP_SLICE=1 -e TF_ML4_CP_SCORE_MB=512 \
  -e TF_ML4_PROMPT_KERNEL=mma8 -e TF_ML4_RANK_CACHE=/w/rank-cache \
  -e TF_ML4_KV_TOKENS=262144 -e TF_ML4_MEM_FLOOR_GIB=5 -e TF_ML4_MEM_CEILING_GIB=103 -e TF_ML4_HEADROOM_GIB=3.0 \
  -e TF_ML4_POOL_SAFETY_GIB=0 -e TF_ML4_L2_MB=16,8,16 -e TF_ML4_CP_MAX_TEMP=85 ${EXTRA:-} \
  -v "$R:/w" -v "$MODELS:$MODELS:ro" -w /w/src/tensorfold \
  nvcr.io/nvidia/pytorch:26.07-py3 python3 -u -m tensorfold serve "$PACK" --tp 4 --rank "$RANK" --master "$MASTER" \
  --master-port "$PORT" --port 8561 --context 1048576 --max-tokens 8192 "$@" > "$LOG" 2>&1
echo "RC $TAG $?" >> "$LOG"
