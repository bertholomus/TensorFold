#!/bin/bash
# Push the local TensorFold fork working tree to ascent nodes (root-owned leftovers fixed via container).
# usage: tfpush.sh 2 [3 4 1]
cd /tank/projects/glm53-tensorfold && tar czf /tank/projects/glm53-tensorfold/tf-src.tgz --exclude=.git --exclude=__pycache__ --exclude=assets TensorFold || exit 1
for n in "$@"; do
  scp -q /tank/projects/glm53-tensorfold/tf-src.tgz /tank/projects/glm53-tensorfold/TensorFold/tools/tfrun.sh ascent-0$n:/tmp/ 2>&1 | grep -v local/bin/env
  ssh -o BatchMode=yes -n ascent-0$n 'mkdir -p ~/ai/glm53-tf/cache && docker run --rm -v $HOME/ai/glm53-tf:/tfw nvcr.io/nvidia/pytorch:26.07-py3 bash -c "find /tfw/TensorFold -user root -exec chown 1000:1000 {} + 2>/dev/null" >/dev/null 2>&1; cd ~/ai/glm53-tf && rm -rf TensorFold && tar xzf /tmp/tf-src.tgz && mv /tmp/tfrun.sh . && sed -i "s/--rm --name/--name/" tfrun.sh && chmod +x tfrun.sh && echo "$(hostname) pushed"' 2>&1 | grep -v local/bin/env
done
