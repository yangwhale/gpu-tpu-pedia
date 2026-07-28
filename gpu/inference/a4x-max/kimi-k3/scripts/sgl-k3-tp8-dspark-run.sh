#!/usr/bin/env bash
# Kimi-K3 · TP8 跨 2 节点 · A 配置(symm-mem 保留，已实测值 +35%) + DSPARK 投机解码
set -euo pipefail
RANK=${1:?node_rank}
ADDR=${ADDR:-k3sgl-0.k3sgl.default.svc.cluster.local:5000}
MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
DRAFT=${DRAFT:-/mnt/ssd/Kimi-K3-DSpark}
BLOCK=${BLOCK:-7}
MAX_RUNNING=${MAX_RUNNING:-256}
MAMBA_RATIO=${MAMBA_RATIO:-0.86}
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --tp-size 8 --nnodes 2 --node-rank "$RANK" --dist-init-addr "$ADDR" \
  --disable-custom-all-reduce --enable-symm-mem \
  --mem-fraction-static 0.85 \
  --mamba-full-memory-ratio "$MAMBA_RATIO" \
  --max-running-requests "$MAX_RUNNING" \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-dspark-block-size "$BLOCK" \
  --enable-linear-replayssm-spec \
  --host 0.0.0.0 --port 30000
