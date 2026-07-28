#!/usr/bin/env bash
# Kimi K3 · TP8 无投机解码基线 —— 对应官方 111 tok/s/user
# 状态：[未实测]
set -euo pipefail
: "${HEAD_ADDR:?需要设置 HEAD_ADDR}"
NODE_RANK="${NODE_RANK:-0}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"

export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

exec vllm serve "$MODEL" \
  --enable-prefix-caching \
  --tensor-parallel-size 8 --nnodes 2 --node-rank "$NODE_RANK" \
  --moe-backend auto --trust-remote-code --load-format fastsafetensors \
  --max-num-seqs 512 --gpu-memory-utilization 0.9 --max-model-len auto \
  --max-cudagraph-capture-size 256 --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 \
  --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}'
