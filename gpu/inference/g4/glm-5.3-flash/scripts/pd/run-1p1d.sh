#!/bin/bash
# Orchestrate 1P1D on the prefill node: wait for P and D, start proxy, smoke test, sweep.
# Usage: P=<P_IP>:8100 D=<D_IP>:8200 bash run-1p1d.sh
HERE=$(cd $(dirname $0) && pwd); W=$HOME/g4glm; mkdir -p $W
step(){ echo "=== STEP $* $(date -u +%FT%TZ)"; }
step wait-servers
for ep in $P $D; do
  for i in $(seq 1 120); do curl -sf http://$ep/v1/models >/dev/null && break; sleep 10; done
  curl -sf http://$ep/v1/models >/dev/null || { echo "SERVER TIMEOUT $ep"; exit 2; }
done
step proxy
PREFILL="$P" DECODE="$D" bash $HERE/proxy.sh
for i in $(seq 1 30); do curl -sf localhost:8000/healthcheck >/dev/null && break; sleep 2; done
step smoke
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm","messages":[{"role":"user","content":"用一句话介绍香港。"}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false}}' | tee $W/pd-smoke.json
echo
grep -q '"content":"' $W/pd-smoke.json || { echo "SMOKE FAILED"; sudo docker logs --tail 40 glm-proxy > $W/pd-proxy-fail.log 2>&1; exit 3; }
step bench
TAG=${TAG:-1p1d} bash $HERE/bench.sh
sudo docker logs glm-p > $W/pd-prefill.log 2>&1; sudo docker logs glm-proxy > $W/pd-proxy.log 2>&1
step ALL_DONE
