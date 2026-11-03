#!/bin/bash
# Start one vLLM instance on a subset of GPUs (no PD). Used for capacity tests and TP4 layouts.
# Usage: NAME=glm-a GPUS=0,1,2,3 PORT=8001 MODEL_DIR=/data/GLM-5.3-Flash-NVFP4 bash serve-instance.sh
set -euo pipefail
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57}
NAME=${NAME:-glm}; GPUS=${GPUS:-0,1,2,3,4,5,6,7}; PORT=${PORT:-8000}; MODEL_DIR=${MODEL_DIR:-/data/GLM-5.3-Flash}
TP=$(echo $GPUS | tr ',' '\n' | wc -l)
sudo docker rm -f $NAME >/dev/null 2>&1 || true
sudo docker run -d --name $NAME --gpus "\"device=$GPUS\"" --shm-size=64g --ipc=host --network host \
  -v $MODEL_DIR:/model $IMAGE \
  --model /model --served-model-name glm --tensor-parallel-size $TP \
  --max-model-len 131072 --max-num-seqs ${MAX_NUM_SEQS:-256} \
  --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
  --tool-call-parser glm47 --reasoning-parser glm45 --port $PORT ${EXTRA_ARGS:-}
echo "started $NAME (TP$TP, GPUs $GPUS) on :$PORT"
