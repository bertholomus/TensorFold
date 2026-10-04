#!/bin/bash
# Copy the full 3.0bpw EXL3 checkpoint from spark2 to spark1/spark3/spark4 over the 10.0.0.x fabric (tar | nc), in parallel.
set -u
F=glm53-exl3-3.0bpw-full
t0=$(date +%s)
port=29820
for n in 1 3 4; do
  (
    p=$((port + n))
    ssh -o BatchMode=yes -n ascent-0$n "mkdir -p ~/ai/models/$F && cd ~/ai/models/$F && timeout 7200 nc -l -s 10.0.0.$n -p $p | tar x && echo recv-ok-$n" 2>&1 | grep -v local/bin/env &
    sleep 6
    ssh -o BatchMode=yes -n spark2 "cd ~/ai/models/$F && tar c . | nc -q 1 10.0.0.$n $p && echo send-ok-$n" 2>&1 | grep -v local/bin/env
    wait
    echo "A$n leg $(( $(date +%s) - t0 ))s"
  ) &
done
wait
cd "${QUANT_DIR:?set QUANT_DIR}" && stat -c "%n %s" model-*.safetensors > /tmp/rq/full_sizes.txt
for n in 1 3 4; do
  scp -q /tmp/rq/full_sizes.txt ascent-0$n:/tmp/full_sizes.txt
  ssh -o BatchMode=yes -n ascent-0$n "cd ~/ai/models/$F && bad=0; while read f s; do r=\$(stat -c %s \$f 2>/dev/null); [ \"\$r\" = \"\$s\" ] || { echo MISMATCH \$f; bad=1; }; done < /tmp/full_sizes.txt; echo \$(hostname) files=\$(ls model-*.safetensors | wc -l) bad=\$bad \$(du -sh . | cut -f1) disk_left=\$(df -h ~ | awk 'NR==2{print \$4}')" 2>&1 | grep -v local/bin/env
done
echo "TOTAL $(( $(date +%s) - t0 ))s"
