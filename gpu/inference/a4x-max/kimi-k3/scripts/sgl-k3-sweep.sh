#!/usr/bin/env bash
# Kimi-K3 · SGLang 压测扫描器（在 pod 内跑）
#   bash sweep.sh <tag> "<conc 列表>" [ISL] [OSL]
# 纪律（来自 deepseek-v4 runbook，已验证）：
#   - 开环 --request-rate inf（闭环会低估 13%）
#   - temperature 0（否则投机接受率崩，跨配置不可比）
#   - 每档跑两轮，只报第二轮（冷热差 6.5-7%，确定性成本）
set -uo pipefail
TAG=${1:?tag}; CONCS=${2:-"1 8 32 64 128 256"}; ISL=${3:-4096}; OSL=${4:-1024}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-30000}; MODEL=${MODEL:-/mnt/ssd/Kimi-K3}
OUT=/mnt/ssd/bench/$TAG; mkdir -p "$OUT"

run(){ # conc round
  local c=$1 r=$2 np
  np=$(( c*3 )); [ $np -lt 8 ] && np=8; [ $np -gt 768 ] && np=768
  python3 -m sglang.bench_serving --backend sglang-oai \
    --host "$HOST" --port "$PORT" --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 --temperature 0 \
    --request-rate inf --max-concurrency "$c" --num-prompts "$np" \
    --output-file "$OUT/c${c}_r${r}.jsonl" \
    > "$OUT/c${c}_r${r}.log" 2>&1
}

echo "[sweep] tag=$TAG isl=$ISL osl=$OSL concs=$CONCS  $(date -u +%H:%M:%S)"
for c in $CONCS; do
  echo "[sweep] conc=$c 冷轮(丢弃) $(date -u +%H:%M:%S)"; run "$c" cold
  echo "[sweep] conc=$c 热轮(采信) $(date -u +%H:%M:%S)"; run "$c" warm
  grep -hE "Output token throughput|Median TPOT|Median TTFT|Benchmark duration|Request throughput" \
    "$OUT/c${c}_rwarm.log" 2>/dev/null || grep -hE "Output token throughput|Median TPOT|Median TTFT|Benchmark duration" "$OUT/c${c}_rwarm.log" 2>/dev/null
done
echo "[sweep] DONE $(date -u +%H:%M:%S)" ; touch "$OUT/.done"
