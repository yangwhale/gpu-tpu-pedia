#!/usr/bin/env bash
# SGLang · Kimi K3 · GB300 2x4 · Unified TP8 无投机基线
# 用途: DSPARK 崩了 (SGLang open bug #32569) 时的退路 + 加速比分母
set -euo pipefail

MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
PORT=${PORT:-30000}
MAX_RUNNING=${MAX_RUNNING:-256}
MAMBA_RATIO=${MAMBA_RATIO:-0.86}

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
  --host 0.0.0.0 --port "$PORT"
