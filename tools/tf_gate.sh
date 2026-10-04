#!/bin/bash
# Acceptance gate for the live TP4 lane, run on spark1: wait for it to serve, greedy equality against the serial
# reference (tf_greedy.py), then tf_bench.py, then the engine's decode lines. Output: opt/gate_LABEL.log.
# usage (on spark1): bash ~/ai/glm53-tf/TensorFold/tools/tf_gate.sh LABEL [REFERENCE_JSON]
LABEL=$1
REF=${2:-$HOME/ai/glm53-tf/opt/greedy_serial.ref.json}
cd ~/ai/glm53-tf || exit 1
mkdir -p opt
exec > >(tee "opt/gate_$LABEL.log") 2>&1
for i in $(seq 1 90); do
  docker logs tf-tp4 2>&1 | grep -q "serving GLM" && break
  if docker logs tf-tp4 2>&1 | grep -qE "Traceback|^tensorfold: "; then
    echo "lane failed to start"; docker logs tf-tp4 2>&1 | grep -v "NCCL INFO" | tail -25; exit 1
  fi
  sleep 10
done
docker logs tf-tp4 2>&1 | grep -E "startup estimate|serving GLM"
python3 TensorFold/tools/tf_greedy.py http://127.0.0.1:18090 "opt/greedy_$LABEL.json"
python3 - "$REF" "opt/greedy_$LABEL.json" <<'EOF'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same = [k for k in a if a[k] == b.get(k)]
print(f"equality {len(same)} / {len(a)}" + ("" if len(same) == len(a) else "  DIFFERENT: " + "; ".join(
    k[:40] for k in a if k not in same)))
EOF
python3 TensorFold/tools/tf_bench.py http://127.0.0.1:18090
docker logs tf-tp4 2>&1 | grep -E "decode [0-9]" | tail -12
