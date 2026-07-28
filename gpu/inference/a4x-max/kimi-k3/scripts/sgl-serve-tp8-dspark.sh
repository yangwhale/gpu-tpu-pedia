#!/usr/bin/env bash
# SGLang · Kimi K3 · GB300 2x4 · Unified TP8 + DSPARK 投机解码
# 参数来源: SGLang K3 cookbook [官方标注 Not Verified]
# 启动纪律来源: 本仓库 deepseek-v4/SGLANG-V4PRO-RUNBOOK.md [本环境已验证]
set -euo pipefail

MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
DRAFT=${DRAFT:-RadixArk/Kimi-K3-DSpark}   # ⚠️ 不是 vLLM 用的 Inferact/...
PORT=${PORT:-30000}

# ★ 必须显式设置。开投机后未设置会被 SGLang 重置为 48,并发直接废掉
MAX_RUNNING=${MAX_RUNNING:-256}

# ★ KDA 状态池 vs MLA KV 池的划线 —— K3 版的 swa-full-tokens-ratio
#   调参判据: 哪个池先到 0.9+ 就给哪个加预算, 目标两边同时落在 0.88-0.93
MAMBA_RATIO=${MAMBA_RATIO:-0.86}

# 注意: 刻意不设 --moe-runner-backend 与三个 attention backend
#   Blackwell 上由 SGLang 自动解析 (FlashInfer MXFP4 / trtllm-gen SiTU);
#   设了任何一个 attention backend 会取消其余两个的自动解析。
#   但启动后必须从日志把实际选中的 backend 抄进 runbook §11 —— 依赖默认值是 V4 踩过的雷。

exec python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path "$MODEL" \
  --tp-size 8 \
  --disable-custom-all-reduce \
  --enable-symm-mem \
  --mem-fraction-static 0.85 \
  --mamba-full-memory-ratio "$MAMBA_RATIO" \
  --max-running-requests "$MAX_RUNNING" \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-dspark-block-size 7 \
  --enable-linear-replayssm-spec \
  --host 0.0.0.0 --port "$PORT"
