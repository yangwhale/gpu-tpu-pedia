#!/usr/bin/env bash
# SGLang · Kimi K3 · PD 分离 · Decode 侧
#   DCP=0 → TP8            (官方前沿 2,808 tok/s/GPU 用的就是这个)
#   DCP=1 → TP8 + DCP8     (逻辑 KV 1.5M→12.2M, agentic 48 并发 / 541 tok/s)
set -euo pipefail

MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
PORT=${PORT:-30100}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
CTX_LEN=${CTX_LEN:-131072}        # ★ 必须与 prefill 侧一致
DCP=${DCP:-0}
MAMBA_RATIO=${MAMBA_RATIO:-0.86}
MAX_RUNNING=${MAX_RUNNING:-256}
# ★ 不固定的话: <32 请求时默认两倍 batch, >32 时为零 —— 并发行为会很诡异
EXTRA_SLOTS=${EXTRA_SLOTS:-32}

export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=${SGLANG_MOONCAKE_CUSTOM_MEM_POOL:-NVLINK}
export MC_FORCE_MNNVL=${MC_FORCE_MNNVL:-1}

ARGS=(
  --trust-remote-code
  --model-path "$MODEL"
  --tp-size 8
  --context-length "$CTX_LEN"
  --mem-fraction-static 0.85
  --mamba-full-memory-ratio "$MAMBA_RATIO"
  --max-running-requests "$MAX_RUNNING"
  --disable-custom-all-reduce
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
  --disaggregation-mode decode
  --disaggregation-transfer-backend mooncake
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
  --disaggregation-decode-extra-slots "$EXTRA_SLOTS"
  --host 0.0.0.0 --port "$PORT"
)

if [[ "$DCP" == "1" ]]; then
  # ⚠️ DCP 下不能开 --enable-symm-mem (为 decode graph 正确性强制禁用)
  # ⚠️ --dcp-comm-backend 刻意不设, GB300 上自动选 fi_a2a
  ARGS+=( --dcp-size 8 )
else
  ARGS+=( --enable-symm-mem )
fi

exec python3 -m sglang.launch_server "${ARGS[@]}"
