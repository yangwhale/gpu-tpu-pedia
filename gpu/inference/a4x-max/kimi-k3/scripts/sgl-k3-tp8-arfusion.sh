#!/usr/bin/env bash
# Kimi-K3 · SGLang · GB300 2x4 · Unified TP8 · 无投机基线
# 用法（在 pod 内）：  bash /tmp/serve.sh <node_rank>
# 镜像必须是 lmsysorg/sglang:kimi-k3-*（K3 不在 main 分支，普通 nightly 没有）
set -euo pipefail

RANK=${1:?需要 node_rank (0 或 1)}
ADDR=${ADDR:-k3sgl-0.k3sgl.default.svc.cluster.local:5000}
MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
PORT=${PORT:-30000}

# ★ 不显式设会被投机路径重置为 48；这里虽是 NOSPEC，仍然显式固定便于对照
MAX_RUNNING=${MAX_RUNNING:-256}
# ★ 官方 help 原文：ratio of mamba state memory to full kv cache memory
#   即 ratio ↑ = KDA 状态池变大 / MLA KV 池变小
MAMBA_RATIO=${MAMBA_RATIO:-0.86}
MEM_FRAC=${MEM_FRAC:-0.85}

export SGLANG_K3_AR_FUSION=1
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --tp-size 8 --nnodes 2 --node-rank "$RANK" --dist-init-addr "$ADDR" \
  --disable-custom-all-reduce \
  --mem-fraction-static "$MEM_FRAC" \
  --mamba-full-memory-ratio "$MAMBA_RATIO" \
  --max-running-requests "$MAX_RUNNING" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --host 0.0.0.0 --port "$PORT"
