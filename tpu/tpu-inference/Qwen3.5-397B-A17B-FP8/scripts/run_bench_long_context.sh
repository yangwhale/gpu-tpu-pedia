#!/bin/bash
# Long context benchmarks: 8K/1K and 1K/8K
# REQUIRES vLLM restarted with --max-model-len 16384

MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8
URL=http://localhost:8000/v1/chat/completions
OUTDIR=/tmp/bench_qwen35_long
mkdir -p $OUTDIR
SUMMARY=$OUTDIR/summary.txt
> $SUMMARY

LEVELS=(1 4 16 64)

echo "=== Qwen3.5 Long Context Benchmark ===" | tee -a $SUMMARY
echo "start: $(date -u +%H:%M:%S)" | tee -a $SUMMARY
echo "" | tee -a $SUMMARY

# Scenario 1: 8K input / 1K output (long prompt prefill)
echo "##### Scenario A: 8K input / 1K output (prefill heavy) #####" | tee -a $SUMMARY

for P in "${LEVELS[@]}"; do
  for ROUND in warmup record; do
    LABEL="8k_in_1k_out_p${P}_${ROUND}"
    LOG=$OUTDIR/${LABEL}.log
    echo "=== [$LABEL] start $(date -u +%H:%M:%S) ===" | tee -a $SUMMARY
    evalscope perf \
      --url $URL \
      --model $MODEL \
      --tokenizer-path $MODEL \
      --dataset random \
      --min-prompt-length 8192 --max-prompt-length 8192 \
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
    grep -E '^\| Succeed|^\| Failed|^\| Average latency|^\| Output token throughput' $LOG | tee -a $SUMMARY
    echo "" | tee -a $SUMMARY
  done
done

# Scenario 2: 1K input / 8K output (long generation decode)
echo "##### Scenario B: 1K input / 8K output (decode heavy) #####" | tee -a $SUMMARY

for P in "${LEVELS[@]}"; do
  for ROUND in warmup record; do
    LABEL="1k_in_8k_out_p${P}_${ROUND}"
    LOG=$OUTDIR/${LABEL}.log
    echo "=== [$LABEL] start $(date -u +%H:%M:%S) ===" | tee -a $SUMMARY
    evalscope perf \
      --url $URL \
      --model $MODEL \
      --tokenizer-path $MODEL \
      --dataset random \
      --min-prompt-length 1024 --max-prompt-length 1024 \
      --max-tokens 8192 --min-tokens 8192 \
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
    grep -E '^\| Succeed|^\| Failed|^\| Average latency|^\| Output token throughput' $LOG | tee -a $SUMMARY
    echo "" | tee -a $SUMMARY
  done
done

echo "=== END $(date -u +%H:%M:%S) ===" | tee -a $SUMMARY
