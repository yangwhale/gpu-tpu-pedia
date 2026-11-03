#!/bin/bash
# 8K-in / 1K-out concurrency sweep against BASE (default http://127.0.0.1:8000: a single instance or the PD proxy).
# Several colocated instances: run one client per instance at the same time, BASE=http://127.0.0.1:<port>.
# Usage: BASE=... TAG=... TOKENIZER=/data/GLM-5.3-Flash-NVFP4 [CONCURRENCY='1 2 4 ... 256'] bash concurrency.sh
# Uses the decode-side tokenizer copy at /data/GLM-5.3-Flash via a throwaway container.
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57}
TAG=${TAG:-1p1d}; OUT=${OUT:-$HOME/g4glm/pd-$TAG}; mkdir -p $OUT
BASE=${BASE:-http://127.0.0.1:8000}; TOKENIZER=${TOKENIZER:-/data/GLM-5.3-Flash}
for c in ${CONCURRENCY:-1 2 4 8 16 32 64 128}; do
  n=$(( c * 4 )); [ $n -lt 8 ] && n=8
  echo "=== c=$c n=$n $(date -u +%FT%TZ)"
  sudo docker run --rm --network host -v $TOKENIZER:/model:ro -v $OUT:/out --entrypoint vllm $IMAGE \
    bench serve --backend openai-chat --base-url $BASE --endpoint /v1/chat/completions \
    --model glm --tokenizer /model \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --max-concurrency $c --num-prompts $n --ignore-eos --seed $c \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir /out --result-filename in8192-out1024-c$c.json \
    > $OUT/bench-in8192-out1024-c$c.txt 2>&1
  grep -E "Successful|Output token throughput|Median TTFT|Mean TPOT|P99 ITL" $OUT/bench-in8192-out1024-c$c.txt
done
echo "=== SWEEP_DONE $(date -u +%FT%TZ)"
