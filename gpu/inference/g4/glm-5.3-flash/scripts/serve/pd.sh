#!/bin/bash
# Start one PD instance with vLLM NixlConnector (P = kv_producer, D = kv_consumer).
# Usage (recommended NVFP4 TP4 instances, two per machine):
#   P: ROLE=prefill NAME=glm-p1 GPUS=0,1,2,3 PORT=8100 SIDE_PORT=5600 MAX_NUM_SEQS=64 GPU_UTIL=0.92 \
#      MODEL_DIR=/data/GLM-5.3-Flash-NVFP4 EXTRA_ARGS=--no-enable-flashinfer-autotune bash pd.sh
#   D: ROLE=decode NAME=glm-d1 GPUS=0,1,2,3 PORT=8200 SIDE_PORT=5600 MAX_NUM_SEQS=128 GPU_UTIL=0.92 ... bash pd.sh
#   (second instance on the same machine: GPUS=4,5,6,7, PORT=8101/8201, SIDE_PORT=5700)
# P and D MUST use the same TP size (KV block size depends on TP; mixed TP fails the NIXL handshake).
#   prefill -> container glm-p, port 8100, kv_producer
#   decode  -> container glm-d, port 8200, kv_consumer
# VLLM_SSM_CONV_STATE_LAYOUT=DS is required: NIXL's hybrid (KDA/Mamba) state transfer refuses to start
# with the default conv-state layout ("3-read Mamba conv transfer requires DS conv state layout").
# UCX_TCP_{TX,RX}_SEG_SIZE=1m: UCX's default TCP segment size makes GPU-memory transfers crawl
# (measured 5 Gbps per agent pair, ~220 MB/s per KV transfer inside vLLM); 1m gives ~27 Gbps per pair
# and 8 pairs saturate the NIC (176 Gbps). Pass UCX_TCP_TX_SEG_SIZE= UCX_TCP_RX_SEG_SIZE= (empty) is NOT the default;
# to reproduce the untuned behaviour remove these two -e lines.
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
  -e UCX_TCP_TX_SEG_SIZE=${UCX_TCP_TX_SEG_SIZE:-1m} -e UCX_TCP_RX_SEG_SIZE=${UCX_TCP_RX_SEG_SIZE:-1m} \
  $IMAGE \
  --model /model --served-model-name glm --tensor-parallel-size $TP \
  --max-model-len 131072 --max-num-seqs ${MAX_NUM_SEQS:-256} \
  --gpu-memory-utilization ${GPU_UTIL:-0.85} --kv-cache-dtype fp8 \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --port $PORT \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$KV_ROLE\",\"kv_load_failure_policy\":\"fail\"}" \
  ${EXTRA_ARGS:-}
echo "started $NAME (TP$TP, GPUs $GPUS) on $MY_IP:$PORT"
