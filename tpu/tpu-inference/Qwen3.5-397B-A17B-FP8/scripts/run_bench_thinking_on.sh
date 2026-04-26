#!/bin/bash
# Thinking ON throughput: 1K input / 1K output (output 限 1K 避免 max_model_len overflow)
# 实际 generation 含 reasoning 会接近 1K，是真实 reasoning model 业务场景

MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8
URL=http://localhost:8000/v1/chat/completions
OUTDIR=/tmp/bench_qwen35_thinking_on
mkdir -p $OUTDIR
SUMMARY=$OUTDIR/summary.txt
> $SUMMARY

LEVELS=(1 16 64)

echo "=== Qwen3.5 1K/1K Thinking ON Benchmark ===" | tee -a $SUMMARY
echo "start: $(date -u +%H:%M:%S)" | tee -a $SUMMARY
echo "" | tee -a $SUMMARY

for P in "${LEVELS[@]}"; do
  for ROUND in warmup record; do
    LABEL="p${P}_${ROUND}"
    LOG=$OUTDIR/${LABEL}.log
    echo "=== [$LABEL] start $(date -u +%H:%M:%S) ===" | tee -a $SUMMARY
    # thinking ON: 不传 chat_template_kwargs (server-side 默认 ON)
    # 也不传 enable_thinking false
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
      > $LOG 2>&1
    if [ "$ROUND" = "record" ]; then
      echo "--- [$LABEL] METRICS (kept, thinking ON) ---" | tee -a $SUMMARY
    else
      echo "--- [$LABEL] (warmup, discarded) ---" | tee -a $SUMMARY
    fi
    grep -E '^\| Succeed|^\| Failed|^\| Average latency|^\| Output token throughput' $LOG | tee -a $SUMMARY
    echo "" | tee -a $SUMMARY
  done
done

echo "=== END $(date -u +%H:%M:%S) ===" | tee -a $SUMMARY
