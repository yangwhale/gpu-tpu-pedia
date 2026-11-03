#!/bin/bash
# Start one PD instance (TP8 on this node) with vLLM NixlConnector.
# Usage: ROLE=prefill|decode [GPUS=0,1,2,3 PORT=8101 NAME=glm-p1 SIDE_PORT=5601] bash serve.sh
#   prefill -> container glm-p, port 8100, kv_producer
#   decode  -> container glm-d, port 8200, kv_consumer
# VLLM_SSM_CONV_STATE_LAYOUT=DS is required: NIXL's hybrid (KDA/Mamba) state transfer refuses to start
# with the default conv-state layout ("3-read Mamba conv transfer requires DS conv state layout").
# G4 has no RDMA: NIXL's UCX backend moves KV over TCP (UCX picks tcp + cuda_copy).
set -euo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57}
ROLE=${ROLE:?prefill or decode}
MODEL_DIR=${MODEL_DIR:-/data/GLM-5.3-Flash}   # NVFP4: /data/GLM-5.3-Flash-NVFP4 plus EXTRA_ARGS=--no-enable-flashinfer-autotune
MY_IP=$(hostname -I | awk '{print $1}')
case $ROLE in
  prefill) NAME=${NAME:-glm-p}; PORT=${PORT:-8100}; KV_ROLE=kv_producer ;;
  decode)  NAME=${NAME:-glm-d}; PORT=${PORT:-8200}; KV_ROLE=kv_consumer ;;
esac
# Several instances per node (e.g. two TP4 prefills): give each its own GPUS / PORT / NAME / SIDE_PORT
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; SIDE_PORT=${SIDE_PORT:-5600}
TP=$(echo $GPUS | tr ',' '\n' | wc -l)
sudo docker rm -f $NAME >/dev/null 2>&1 || true
sudo docker run -d --name $NAME --gpus "\"device=$GPUS\"" --shm-size=64g --ipc=host --network host \
  -v $MODEL_DIR:/model \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST=$MY_IP -e VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_PORT \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e UCX_NET_DEVICES=${UCX_NET_DEVICES:-all} -e UCX_TLS=${UCX_TLS:-all} \
  $IMAGE \
  --model /model --served-model-name glm --tensor-parallel-size $TP \
  --max-model-len 131072 --max-num-seqs ${MAX_NUM_SEQS:-256} \
  --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --port $PORT \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$KV_ROLE\",\"kv_load_failure_policy\":\"fail\"}" \
  ${EXTRA_ARGS:-}
echo "started $NAME (TP$TP, GPUs $GPUS) on $MY_IP:$PORT"
