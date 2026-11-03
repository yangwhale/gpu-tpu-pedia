#!/bin/bash
# Concurrency sweep at input 8K / output 1K against a running vLLM server (container "glm", see RUNBOOK step 5).
# Output: $OUT/bench-in8192-out1024-c<N>.txt (+ .json with per-request metrics)
OUT=${OUT:-$HOME/g4glm/8k1k}; mkdir -p $OUT
for c in ${CONCURRENCY:-1 2 4 8 16 32 64 128}; do
  n=$(( c * 4 )); [ $n -lt 8 ] && n=8          # 4 waves per level, at least 8 requests
  echo "=== c=$c n=$n $(date -u +%FT%TZ)"
  sudo docker exec glm vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model glm --tokenizer /model \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --max-concurrency $c --num-prompts $n --ignore-eos --seed $c \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir /tmp/bench --result-filename in8192-out1024-c$c.json \
    > $OUT/bench-in8192-out1024-c$c.txt 2>&1
  sudo docker cp glm:/tmp/bench/in8192-out1024-c$c.json $OUT/ 2>/dev/null
  grep -E "Output token throughput|Mean TTFT|Mean TPOT" $OUT/bench-in8192-out1024-c$c.txt
done
echo "=== SWEEP_DONE $(date -u +%FT%TZ)"
