#!/usr/bin/env bash
# Kimi-K3 通用启动器：所有配置由环境变量驱动，便于 4 对节点并行跑不同实验
#   用法: VARIANT=<name> EXTRA="<额外参数>" bash serve.sh <node_rank> <dist_addr>
set -euo pipefail
RANK=${1:?node_rank}; ADDR=${2:?dist_addr}
MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
MAMBA_RATIO=${MAMBA_RATIO:-0.86}
MEM_FRAC=${MEM_FRAC:-0.85}
MAX_RUNNING=${MAX_RUNNING:-256}
EXTRA=${EXTRA:-}
echo "[serve] VARIANT=${VARIANT:-?} rank=$RANK addr=$ADDR ratio=$MAMBA_RATIO memfrac=$MEM_FRAC extra=$EXTRA"
# shellcheck disable=SC2086
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --tp-size 8 --nnodes 2 --node-rank "$RANK" --dist-init-addr "$ADDR" \
  --disable-custom-all-reduce \
  --mem-fraction-static "$MEM_FRAC" \
  --mamba-full-memory-ratio "$MAMBA_RATIO" \
  --max-running-requests "$MAX_RUNNING" \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --host 0.0.0.0 --port 30000 $EXTRA
