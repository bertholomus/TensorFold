#!/bin/bash
# Sweep MTP draft depth on TP4: restart with --mtp-drafts D, run tf_greedy, check equality vs serial, log speed.
for D in "$@"; do
  for n in 1 2 3 4; do ssh -o BatchMode=yes -n ascent-0$n 'docker rm -f tf-tp4 >/dev/null 2>&1' 2>&1 | grep -v local/bin/env; done
  bash /tmp/rq/tf/tp4_start.sh 32768 "--mtp-drafts $D" >/dev/null
  for i in $(seq 1 36); do sleep 15; s=$(ssh -o BatchMode=yes -n ascent-01 'docker logs tf-tp4 2>&1 | grep -cE "serving|Traceback|^tensorfold:"' 2>/dev/null); [ "${s:-0}" != "0" ] && break; done
  echo "=== depth $D"
  ssh -o BatchMode=yes -n ascent-01 "cd ~/ai/glm53-tf && timeout 400 python3 tf_greedy.py http://127.0.0.1:18090 /tmp/greedy_d$D.json 2>&1 | grep -v '^  \|^Trace' | tail -5; docker logs tf-tp4 2>&1 | grep -E 'decode [0-9]|request error' | tail -4; python3 -c \"
import json; a=json.load(open('/tmp/greedy_serial.json')); b=json.load(open('/tmp/greedy_d$D.json'))
print('equality', sum(a[k]==b[k] for k in a), '/', len(a))
\"" 2>&1 | grep -v local/bin/env
done
