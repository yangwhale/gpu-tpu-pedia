#!/usr/bin/env bash
# K3 PD · prefill 侧：深度流水 PP8 x TP1（官方称 1.7x TEP8）
set -euo pipefail
RANK=${1:?rank}; ADDR=${2:?addr}
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK MC_FORCE_MNNVL=1
exec python3 -m sglang.launch_server \
  --model-path /mnt/ssd/Kimi-K3 --trust-remote-code \
  --pp-size 8 --tp-size 1 --nnodes 2 --node-rank "$RANK" --dist-init-addr "$ADDR" \
  --context-length ${CTX:-40960} \
  --mem-fraction-static 0.85 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --host 0.0.0.0 --port 30000
