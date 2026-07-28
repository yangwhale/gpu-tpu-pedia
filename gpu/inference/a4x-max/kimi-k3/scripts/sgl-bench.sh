#!/usr/bin/env bash
# SGLang · Kimi K3 压测
# ★ 纪律 (本环境已验证): 第一轮当 warmup 丢掉, 冷热差 6.5-7% 且高度可复现。
#   本脚本默认跑两轮, 只有第二轮的数字可以往 runbook 里填。
set -euo pipefail

HOST=${HOST:-localhost}
PORT=${PORT:-30000}
MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
ISL=${ISL:-4096}
OSL=${OSL:-1024}
CONC=${CONC:-64}          # ⚠️ 别用 conc=8 汇报数字, 只用到容量的个位数百分比
NUM=${NUM:-$((CONC * 4))}
OUT=${OUT:-/tmp/sgl-k3-bench}

mkdir -p "$OUT"

run() {
  local tag=$1
  echo "===== round: $tag  (conc=$CONC isl=$ISL osl=$OSL) ====="
  python3 -m sglang.bench_serving \
    --backend sglang-oai \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --max-concurrency "$CONC" \
    --num-prompts "$NUM" \
    2>&1 | tee "$OUT/${tag}.log"
}

run cold_DISCARD
sleep 5
run warm_REPORT

echo
echo "★ 只取 warm_REPORT 的数字。核对两轮 Benchmark duration 相近再做任何求和。"
