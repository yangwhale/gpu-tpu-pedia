#!/bin/bash
NODE_RANK=$1; DIST_ADDR=$2
source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_SPLIT_DATA_ON_QPS=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1 SGLANG_DG_CACHE_DIR=/mnt/ssd/dg-cache FLASHINFER_WORKSPACE_BASE=/mnt/ssd/fi-cache
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export NATS_SERVER=nats://dynamo-nats:4222 ETCD_ENDPOINTS=http://dynamo-etcd:2379 DYN_SYSTEM_PORT=8082
export SGLANG_RADIX_DISABLE_REUSE=1 SGLANG_DEFAULT_THINKING=1 SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096 SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1 SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=20
python3 -m dynamo.sglang --model-path /mnt/ssd/DeepSeek-V4-Pro --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 --watchdog-timeout 86400 \
  --tensor-parallel-size 8 --data-parallel-size 8 --expert-parallel-size 8 --pp-size 1 \
  --nnodes 2 --node-rank $NODE_RANK --dist-init-addr $DIST_ADDR \
  --enable-dp-attention --enable-dp-lm-head --moe-runner-backend deep_gemm --moe-a2a-backend megamoe --moe-dense-tp-size 1 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 30001 \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
  --mem-fraction-static 0.85 --swa-full-tokens-ratio ${SWA_RATIO:-0.15} --context-length 9216 \
  --max-running-requests 8192 --cuda-graph-max-bs 1280 --stream-interval 60 --disable-radix-cache --enable-metrics
