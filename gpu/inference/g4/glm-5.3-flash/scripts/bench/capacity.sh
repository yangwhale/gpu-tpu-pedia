#!/bin/bash
# Measure the two capacities that decide the P:D ratio, against one running instance (BASE).
#  (a) prefill capacity: 8K input, 1 output token -> input tok/s the instance can absorb
#      (uses /v1/completions: with chat + reasoning parser a 1-token reply can land in "reasoning" and the
#       benchmark then counts it as failed with "Never received a valid chunk")
#  (b) decode capacity:  8K context via a shared 8,184-token prefix (prefix-cache hit, so prefill is ~free),
#      1,024 output tokens -> output tok/s and TPOT vs concurrency, i.e. what a pure D instance can do
# Usage: BASE=http://127.0.0.1:8000 TAG=nvfp4-tp8 TOKENIZER=/data/GLM-5.3-Flash-NVFP4 bash bench/capacity.sh
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57}
BASE=${BASE:-http://127.0.0.1:8000}; TAG=${TAG:?}; TOKENIZER=${TOKENIZER:-/data/GLM-5.3-Flash}
OUT=${OUT:-$HOME/g4glm/cap-$TAG}; mkdir -p $OUT
bench(){ local name=$1; shift
  sudo docker run --rm --network host -v $TOKENIZER:/tok:ro -v $OUT:/out --entrypoint vllm $IMAGE \
    bench serve --backend openai --base-url $BASE --endpoint /v1/completions --model glm --tokenizer /tok \
    --dataset-name random --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir /out --result-filename $name.json "$@" > $OUT/$name.txt 2>&1
  echo "--- $name"; grep -E "Successful|Request throughput|Output token throughput|Total token throughput|Median TTFT|Mean TPOT" $OUT/$name.txt; }
echo "=== prefill $(date -u +%FT%TZ)"
for c in ${PREFILL_C:-4 16 64}; do
  bench prefill-c$c --random-input-len 8192 --random-output-len 1 --max-concurrency $c --num-prompts $((c*8)) --seed $((100+c))
done
echo "=== decode $(date -u +%FT%TZ)"
for c in ${DECODE_C:-32 64 128 256}; do
  bench decode-c$c --random-prefix-len 8184 --random-input-len 8 --random-output-len 1024 --max-concurrency $c --num-prompts $((c*2)) --seed $((200+c))
done
echo "=== CAP_DONE $(date -u +%FT%TZ)"
