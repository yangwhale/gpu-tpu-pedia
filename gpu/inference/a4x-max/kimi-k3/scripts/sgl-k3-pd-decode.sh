#!/usr/bin/env bash
# K3 PD · decode 侧：TP8 + DCP8（长上下文实测最优）
set -euo pipefail
RANK=${1:?rank}; ADDR=${2:?addr}
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK MC_FORCE_MNNVL=1
exec python3 -m sglang.launch_server \
  --model-path /mnt/ssd/Kimi-K3 --trust-remote-code \
  --tp-size 8 --dcp-size 8 --nnodes 2 --node-rank "$RANK" --dist-init-addr "$ADDR" \
  --context-length ${CTX:-40960} \
  --disable-custom-all-reduce \
  --mem-fraction-static 0.85 --mamba-full-memory-ratio 0.60 \
  --max-running-requests 256 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-decode-extra-slots 32 \
  --host 0.0.0.0 --port 30100
