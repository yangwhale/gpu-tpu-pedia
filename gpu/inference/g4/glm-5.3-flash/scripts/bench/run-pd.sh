#!/bin/bash
# PD end-to-end on the proxy node: wait for all P and D instances, start the proxy, smoke test, concurrency sweep.
# Usage: TAG=2p2d-nvfp4 P='<A_IP>:8100 <A_IP>:8101' D='<B_IP>:8200 <B_IP>:8201' bash run-pd.sh
#        (P / D accept any number of host:port entries; results in ~/g4glm/pd-$TAG/)
HERE=$(cd $(dirname $0) && pwd); W=$HOME/g4glm; mkdir -p $W
TAG=${TAG:-pd}
step(){ echo "=== STEP $* $(date -u +%FT%TZ)"; }
step wait-servers
for ep in $P $D; do
  for i in $(seq 1 120); do curl -sf http://$ep/v1/models >/dev/null && break; sleep 10; done
  curl -sf http://$ep/v1/models >/dev/null || { echo "SERVER TIMEOUT $ep"; exit 2; }
done
step proxy
PREFILL="$P" DECODE="$D" bash $HERE/../serve/proxy.sh
for i in $(seq 1 30); do curl -sf localhost:8000/healthcheck >/dev/null && break; sleep 2; done
step smoke
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm","messages":[{"role":"user","content":"用一句话介绍香港。"}],"max_tokens":2000,"reasoning_effort":"low"}' | tee $W/pd-$TAG-smoke.json
echo
grep -q '"content":"' $W/pd-$TAG-smoke.json || { echo "SMOKE FAILED"; sudo docker logs --tail 40 glm-proxy > $W/pd-$TAG-proxy-fail.log 2>&1; exit 3; }
step bench
TAG=$TAG bash $HERE/concurrency.sh
sudo docker logs glm-p > $W/pd-$TAG-prefill.log 2>&1; sudo docker logs glm-proxy > $W/pd-$TAG-proxy.log 2>&1
step ALL_DONE
