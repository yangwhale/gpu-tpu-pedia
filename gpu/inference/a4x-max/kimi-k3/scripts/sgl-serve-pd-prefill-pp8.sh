#!/usr/bin/env bash
# SGLang · Kimi K3 · PD 分离 · Prefill 侧: 深度流水 PP8 x TP1
# 官方实测 PP8xTP1 约为 TEP8 上限的 1.7 倍, TTFT 更低
# ⚠️ 必须用满 8 stage。浅切 (PP4xTP2) 还要付 TP2 all-reduce, 打不过 TEP8
# ⚠️ DSPARK 要求 pp_size == 1, 与 Deep PP 互斥 —— 此脚本刻意不开投机
set -euo pipefail

MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
PORT=${PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
# ★ PD 两侧必须一致, 否则 Decode handshake failed
CTX_LEN=${CTX_LEN:-131072}

# ★ GKE 上 nixl 走 RoCE 调不通 (RoCE v2 over IPv6, netdev 名 gpuNipvlanM)
#   走 MNNVL: 成功标志是 decode 日志出现
#   "Using cross-node NVLink transport (MC_FORCE_MNNVL)"
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=${SGLANG_MOONCAKE_CUSTOM_MEM_POOL:-NVLINK}
export MC_FORCE_MNNVL=${MC_FORCE_MNNVL:-1}

exec python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path "$MODEL" \
  --pp-size 8 --tp-size 1 \
  --context-length "$CTX_LEN" \
  --mem-fraction-static 0.85 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --host 0.0.0.0 --port "$PORT"
