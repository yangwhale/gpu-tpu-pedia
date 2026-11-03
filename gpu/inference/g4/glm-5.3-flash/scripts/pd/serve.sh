#!/bin/bash
# Start one PD instance (TP8 on this node) with vLLM NixlConnector.
# Usage: ROLE=prefill|decode bash serve.sh
#   prefill -> container glm-p, port 8100, kv_producer
#   decode  -> container glm-d, port 8200, kv_consumer
# G4 has no RDMA: NIXL's UCX backend moves KV over TCP (UCX picks tcp + cuda_copy).
set -euo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57}
ROLE=${ROLE:?prefill or decode}
MY_IP=$(hostname -I | awk '{print $1}')
case $ROLE in
  prefill) NAME=glm-p; PORT=8100; KV_ROLE=kv_producer ;;
  decode)  NAME=glm-d; PORT=8200; KV_ROLE=kv_consumer ;;
esac
sudo docker rm -f $NAME >/dev/null 2>&1 || true
sudo docker run -d --name $NAME --gpus all --shm-size=64g --ipc=host --network host \
  -v /data/GLM-5.3-Flash:/model \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST=$MY_IP -e VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  -e UCX_NET_DEVICES=${UCX_NET_DEVICES:-all} -e UCX_TLS=${UCX_TLS:-all} \
  $IMAGE \
  --model /model --served-model-name glm --tensor-parallel-size 8 \
  --max-model-len 131072 --max-num-seqs ${MAX_NUM_SEQS:-256} \
  --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --port $PORT \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$KV_ROLE\",\"kv_load_failure_policy\":\"fail\"}" \
  ${EXTRA_ARGS:-}
echo "started $NAME on $MY_IP:$PORT"
