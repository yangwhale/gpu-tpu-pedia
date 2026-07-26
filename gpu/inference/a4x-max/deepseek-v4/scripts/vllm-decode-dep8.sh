#!/bin/bash
# vLLM decode dep8 = TP1 + DP8-attention + EP8，跨 2 节点
# 用法: head   节点: bash vllm-decode-dep8.sh head   <head-ip> <本pod-ip>
#       worker 节点: bash vllm-decode-dep8.sh worker <head-ip> <本pod-ip>
# ⚠️ 第 3 个参数是【本 pod 自己的 IP】—— VLLM_NIXL_SIDE_CHANNEL_HOST 是 bind 地址不是对端
# 为什么 dep8 优于 TP4: MLA 的 KV 是所有头共享的 latent，TP 下不分片只复制
#   → TP4 把 KV 复制 4 份。DP-attention 每 rank 各存各请求，天然不复制；
#   EP8 把 384 expert 摊到每卡 48 个，省 HBM → 更大 batch。实测每卡效率 2.6×
ROLE=$1; HEAD_IP=$2; SELF_IP=$3
export UCX_TLS=cuda_copy,cuda_ipc,tcp
export UCX_CUDA_IPC_ENABLE_MNNVL=y
export UCX_NET_DEVICES=all
export VLLM_USE_NCCL_SYMM_MEM=0 NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1
export VLLM_NIXL_SIDE_CHANNEL_PORT=5558 VLLM_NIXL_SIDE_CHANNEL_HOST=$SELF_IP
export PYTHONHASHSEED=0
export TMPDIR=/mnt/ssd/tmp && mkdir -p $TMPDIR
DP_ARGS="--tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-size-local 4 \
  --data-parallel-address $HEAD_IP --data-parallel-rpc-port 13345 --enable-expert-parallel"
[ "$ROLE" = "worker" ] && DP_ARGS="$DP_ARGS --data-parallel-start-rank 4 --headless"
vllm serve /mnt/ssd/DeepSeek-V4-Pro-DSpark --served-model-name deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --trust-remote-code --enable-cumem-allocator --kv-cache-dtype fp8 --block-size 256 \
  --port 8002 $DP_ARGS \
  --max-num-seqs 1024 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 1024 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64,96,128,192,256,384,512,768,1024]}' \
  --gpu-memory-utilization 0.85 --no-disable-hybrid-kv-cache-manager \
  --attention_config.use_fp4_indexer_cache=True \
  --moe-backend deep_gemm_mega_moe --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
