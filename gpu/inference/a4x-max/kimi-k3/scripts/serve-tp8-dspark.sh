#!/usr/bin/env bash
# Kimi K3 · TP8 + DSpark · GB300 (2 节点 × 4 GPU)
# 来源：vLLM day-0 博客 reproduce recipe，逐字复刻，仅参数化 HEAD_ADDR / NODE_RANK
# 状态：[未实测]
set -euo pipefail

: "${HEAD_ADDR:?需要设置 HEAD_ADDR（node-0 的 IP）}"
NODE_RANK="${NODE_RANK:-0}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
DRAFT="${DRAFT:-Inferact/Kimi-K3-DSpark}"

export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600        # 2.8T 加载 + graph capture 很久

exec vllm serve "$MODEL" \
  --enable-prefix-caching \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --moe-backend auto \
  --trust-remote-code \
  --load-format fastsafetensors \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.9 \
  --max-model-len auto \
  --max-cudagraph-capture-size 256 \
  --kv-cache-dtype fp8 \
  --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}' \
  --speculative-config "{\"model\":\"${DRAFT}\",\"method\":\"dspark\",\"num_speculative_tokens\":7,\"attention_backend\":\"FLASHINFER_MLA\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}"
