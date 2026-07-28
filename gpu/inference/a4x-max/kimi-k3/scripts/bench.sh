#!/usr/bin/env bash
# Kimi K3 压测 —— 两条官方命令
#   ./bench.sh random   → 8K/1K bs=1，对标 TP8 111 / TP16 118 tok/s
#   ./bench.sh speed    → SPEED Bench bs=1，对标 TP8 331 / TP16 370 tok/s
# 状态：[未实测]
set -euo pipefail
: "${HEAD_ADDR:?需要设置 HEAD_ADDR}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
MODE="${1:-random}"

case "$MODE" in
  random)
    exec vllm-bench --backend openai --base-url "http://${HEAD_ADDR}:8000" \
      --model "$MODEL" --dataset-name random \
      --random-input-len 8192 --random-output-len 1024 --random-range-ratio 0.8 \
      --prompt-token-ids --ignore-eos \
      --sweep-max-concurrency 1 --sweep-num-prompts-factor 10 --seed 42 \
      --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,99" \
      --save-result ;;
  speed)
    # ⚠️ 官方用 temperature 1.0（DSpark draft 是 probabilistic 采样，与之匹配）
    #    别照搬 V4 的 temperature=0 —— 那条教训针对 greedy draft
    exec vllm-bench --backend openai --base-url "http://${HEAD_ADDR}:8000" \
      --model "$MODEL" --dataset-name speed-bench \
      --speed-bench-config throughput_16k --speed-bench-max-input-len 10240 \
      --speed-bench-category low_entropy --output-len 1536 \
      --num-prompts 10 --no-oversample --max-concurrency 1 \
      --temperature 1.0 --top-p 0.95 --save-result --save-detailed ;;
  *) echo "用法: $0 [random|speed]" >&2; exit 2 ;;
esac
