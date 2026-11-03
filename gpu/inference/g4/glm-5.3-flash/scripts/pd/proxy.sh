#!/bin/bash
# Run the NIXL toy proxy (from vLLM tests/v1/kv_connector/nixl_integration at commit 76c6f3f2a)
# inside the vLLM image (it already has fastapi/httpx). Clients talk to :8000.
# Usage: PREFILL="10.0.0.1:8100 [10.0.0.3:8100]" DECODE="10.0.0.2:8200 [...]" bash proxy.sh
set -euo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57}
HERE=$(cd $(dirname $0) && pwd)
ph=(); pp=(); for x in $PREFILL; do ph+=(${x%:*}); pp+=(${x#*:}); done
dh=(); dp=(); for x in $DECODE;  do dh+=(${x%:*}); dp+=(${x#*:}); done
sudo docker rm -f glm-proxy >/dev/null 2>&1 || true
sudo docker run -d --name glm-proxy --network host -v $HERE:/pd --entrypoint python3 $IMAGE \
  /pd/toy_proxy_server.py --host 0.0.0.0 --port 8000 \
  --prefiller-hosts ${ph[@]} --prefiller-ports ${pp[@]} \
  --decoder-hosts ${dh[@]} --decoder-ports ${dp[@]}
echo "proxy on :8000 -> P[${PREFILL}] D[${DECODE}]"
