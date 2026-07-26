#!/bin/bash
# vLLM prefill (TP4, kv_producer) — DeepSeek-V4-Pro-DSpark on GB300
# 用法: bash vllm-prefill-tp4.sh <本 pod IP>
# ⚠️ 必须用 deepgemm 镜像，通用镜像会静默 fallback 到慢 kernel（见 runbook 文首）
SELF_IP=$1
# ── KV over NVLink 三件套之一：UCX 只留 NVLink 路径 ──
export UCX_TLS=cuda_copy,cuda_ipc,tcp        # 删掉 rdma/rc
export UCX_CUDA_IPC_ENABLE_MNNVL=y           # cuda_ipc 跨节点走多机 NVLink
export UCX_NET_DEVICES=all
export VLLM_USE_NCCL_SYMM_MEM=0 NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1
export VLLM_NIXL_SIDE_CHANNEL_PORT=5557 VLLM_NIXL_SIDE_CHANNEL_HOST=$SELF_IP
export PYTHONHASHSEED=0
# core dump / tmp 落到本地盘，别撑爆 ephemeral 触发 pod 驱逐
# ★ 默认 600s 不够：dep8 有 8 个 ApiServer，page cache 冷时权重加载+warmup+graph capture 会超时
#   报错 TimeoutError: Timed out waiting for engine core processes to start
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
export TMPDIR=/mnt/ssd/tmp && mkdir -p $TMPDIR
vllm serve /mnt/ssd/DeepSeek-V4-Pro-DSpark --served-model-name deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --trust-remote-code --enable-cumem-allocator --kv-cache-dtype fp8 --block-size 256 \
  --port 8001 --tensor-parallel-size 4 --enforce-eager \
  --max-num-seqs 16 --max-num-batched-tokens 16384 --no-disable-hybrid-kv-cache-manager \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
