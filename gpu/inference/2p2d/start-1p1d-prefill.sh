#!/bin/bash
#
# SGLang DeepSeek-V3 Prefill Node (1P+1D, single-node per role)
# Run on: b7 (10.8.0.12)
# Note: Single-node uses NVLink for MoE all-to-all, no DeepEP needed.
#

set -e

# CUDA Environment
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# SGLang Environment
export TORCH_CUDA_ARCH_LIST="10.0"
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=false
export SGLANG_LOCAL_IP_NIC=enp0s19
export GLOO_SOCKET_IFNAME=enp0s19
export NCCL_SOCKET_IFNAME=enp0s19
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export PYTHONUNBUFFERED=1

# Configuration
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V3}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/lssd/huggingface/hub}"

echo "============================================="
echo "SGLang 1P+1D Prefill Node (b7)"
echo "============================================="
echo "tp-size:         8"
echo "dp-size:         1"
echo "Transfer:        nixl"
echo "============================================="

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 8 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.82 \
    --watchdog-timeout 1000000 \
    --decode-log-interval 1 \
    "$@"
