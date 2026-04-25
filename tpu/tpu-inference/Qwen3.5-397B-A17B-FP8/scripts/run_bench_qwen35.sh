#!/bin/bash
# Qwen3.5 throughput benchmark - warmup + record mode
# Each batch size runs 2 rounds: warmup (discarded) + record (kept)

MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8
URL=http://localhost:8000/v1/chat/completions
OUTDIR=/tmp/bench_qwen35
mkdir -p $OUTDIR
SUMMARY=$OUTDIR/summary.txt
> $SUMMARY

LEVELS=(1 4 16 64 256)

echo "=== Qwen3.5-397B-A17B-FP8 1K/1K Throughput Benchmark ===" | tee -a $SUMMARY
echo "start: $(date -u +%H:%M:%S)" | tee -a $SUMMARY
echo "" | tee -a $SUMMARY

for P in "${LEVELS[@]}"; do
  for ROUND in warmup record; do
    LABEL="p${P}_${ROUND}"
    LOG=$OUTDIR/${LABEL}.log
    echo "=== [$LABEL] start $(date -u +%H:%M:%S) ===" | tee -a $SUMMARY
    evalscope perf \
      --url $URL \
      --model $MODEL \
      --tokenizer-path $MODEL \
      --dataset random \
      --min-prompt-length 1024 --max-prompt-length 1024 \
      --max-tokens 1024 --min-tokens 1024 \
      --parallel $P --number $P \
      --api openai --stream \
      --read-timeout 1800 --connect-timeout 60 \
      --extra-args '{"chat_template_kwargs": {"enable_thinking": false}}' \
      > $LOG 2>&1
    if [ "$ROUND" = "record" ]; then
      echo "--- [$LABEL] METRICS (kept) ---" | tee -a $SUMMARY
    else
      echo "--- [$LABEL] (warmup, discarded) ---" | tee -a $SUMMARY
    fi
    grep -E 'Output token throughput|Average TTFT|Average TPOT|Average latency|Succeed requests|Failed requests|Total tokens|Output tokens' $LOG | tee -a $SUMMARY
    echo "" | tee -a $SUMMARY
  done
done

echo "=== END $(date -u +%H:%M:%S) ===" | tee -a $SUMMARY
