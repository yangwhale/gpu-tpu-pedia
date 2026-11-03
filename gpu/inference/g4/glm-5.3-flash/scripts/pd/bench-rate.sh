#!/bin/bash
# Fixed request-rate (Poisson arrivals) sweep through the PD proxy (:8000), 8K in / 1K out.
# Unlike the concurrency sweep, requests arrive spread out over time, so TTFT here is what a real
# client sees at that load instead of the queue of a burst. Run on the proxy node.
# Each rate uses its own seed: reusing one seed re-sends the same prompts and later rates hit the
# prefix cache, which makes TTFT look better as load rises. SEED_OFFSET separates parallel clients.
# Usage: TAG=2p1d-nvfp4 TOKENIZER=/data/GLM-5.3-Flash-NVFP4 RATES="0.5 1 1.5 2" bash bench-rate.sh
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57}
TAG=${TAG:?}; TOKENIZER=${TOKENIZER:-/data/GLM-5.3-Flash}; DUR=${DUR:-180}
OUT=${OUT:-$HOME/g4glm/rate-$TAG}; mkdir -p $OUT
for r in ${RATES:-0.5 1 1.5 2}; do
  n=$(python3 -c "print(max(60, int($r*$DUR)))")
  echo "=== rate=$r n=$n $(date -u +%FT%TZ)"
  sudo docker run --rm --network host -v $TOKENIZER:/tok:ro -v $OUT:/out --entrypoint vllm $IMAGE \
    bench serve --backend openai-chat --base-url ${BASE:-http://127.0.0.1:8000} --endpoint /v1/chat/completions \
    --model glm --tokenizer /tok --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --request-rate $r --max-concurrency 512 --num-prompts $n --ignore-eos --seed $(python3 -c "print(1000+int(float('$r')*100)+${SEED_OFFSET:-0})") \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir /out --result-filename rate-$r.json > $OUT/rate-$r.txt 2>&1
  grep -E "Successful|Failed requests|Output token throughput|Median TTFT|P99 TTFT|Mean TPOT|P99 ITL" $OUT/rate-$r.txt
done
echo "=== RATE_DONE $(date -u +%FT%TZ)"
