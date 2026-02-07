#!/bin/bash
#
# SGLang DeepSeek-V3 Decode Node 0 (2P+2D with DeepEP)
# Run on: b9 (10.8.0.15)
#

set -e

# DeepEP Environment
export NVSHMEM_HOME=/opt/deepep/nvshmem
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"

# NVSHMEM IBGDA Configuration
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_HCA_PREFIX=mlx5
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1

# GPU to NIC mapping
export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0:1,1:mlx5_1:1,2:mlx5_2:1,3:mlx5_3:1,4:mlx5_4:1,5:mlx5_5:1,6:mlx5_6:1,7:mlx5_7:1

# DeepEP Python path
export PYTHONPATH=/opt/deepep/DeepEP:${PYTHONPATH:-}

# SGLang Environment
export TORCH_CUDA_ARCH_LIST="10.0"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768
export MC_TE_METRIC=true
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000
export SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=false
export SGLANG_LOCAL_IP_NIC=enp0s19
export GLOO_SOCKET_IFNAME=enp0s19
export NCCL_SOCKET_IFNAME=enp0s19
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export PYTHONUNBUFFERED=1

# Configuration - Decode uses its own master (b9)
DECODE_MASTER_IP="10.8.0.15"  # b9 is the decode master
DIST_INIT_ADDR="${DECODE_MASTER_IP}:5757"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V3}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/lssd/huggingface/hub}"
DEEPEP_CONFIG="${DEEPEP_CONFIG:-/lssd/huggingface/deepep_config.json}"
INIT_EXPERT_LOCATION="${INIT_EXPERT_LOCATION:-/lssd/huggingface/attachment_ep_statistics/decode_in2000out100.json}"

echo "============================================="
echo "SGLang 2P+2D Decode Node 0 (b9)"
echo "============================================="
echo "Dist Init Addr:  $DIST_INIT_ADDR"
echo "nnodes:          2"
echo "node-rank:       0"
echo "tp-size:         16"
echo "dp-size:         8"
echo "DeepEP:          enabled (low_latency)"
echo "============================================="

# Verify DeepEP
python3 -c "from deep_ep import Buffer, Config; print('DeepEP OK')" || exit 1

echo ""
echo "IMPORTANT: Wait for Prefill nodes to be fully started first!"
echo ""

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --nnodes 2 \
    --node-rank 0 \
    --tp-size 16 \
    --dp-size 8 \
    --enable-dp-attention \
    --host 0.0.0.0 \
    --port 30000 \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --disable-shared-experts-fusion \
    --attention-backend cutlass_mla \
    --disable-radix-cache \
    --disable-cuda-graph \
    --chunked-prefill-size 8192 \
    --max-running-requests 4096 \
    --context-length 2176 \
    --mem-fraction-static 0.78 \
    --num-reserved-decode-tokens 512 \
    --watchdog-timeout 1000000 \
    --decode-log-interval 1 \
    "$@"
