#!/bin/bash
# ============================================================================
# Qwen3.5-397B-A17B-FP8 吞吐量压测脚本
# ----------------------------------------------------------------------------
# 用 evalscope perf 打本地 OpenAI 兼容接口，扫描多个并发档位，
# 每档跑两轮 (warmup 丢弃 + record 保留)，最后汇总关键指标。
# 负载固定 1K 输入 / 1K 输出 (random dataset)。
# ============================================================================

# ---- 基本配置 ----
MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8           # 模型路径 (同时用作 tokenizer)
URL=http://localhost:8000/v1/chat/completions        # 推理服务 endpoint (vLLM/SGLang 等)
OUTDIR=/tmp/bench_qwen35                             # 日志 + 汇总输出目录
mkdir -p $OUTDIR
SUMMARY=$OUTDIR/summary.txt
> $SUMMARY                                           # 清空汇总文件

# 并发档位：从单请求一路扫到 256，用来画 throughput vs concurrency 曲线
LEVELS=(1 4 16 64 256)

echo "=== Qwen3.5-397B-A17B-FP8 1K/1K Throughput Benchmark ===" | tee -a $SUMMARY
echo "start: $(date -u +%H:%M:%S)" | tee -a $SUMMARY
echo "" | tee -a $SUMMARY

# ---- 主循环：每个并发档位跑 warmup + record 两轮 ----
for P in "${LEVELS[@]}"; do
  for ROUND in warmup record; do
    # warmup 用来预热 KV cache / JIT 编译 / 调度器稳态，结果不计入
    # record 才是真正记录的指标
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
    # 参数解释：
    #   --dataset random              随机生成 prompt，避免 cache 命中影响公平性
    #   --min/max-prompt-length 1024  固定输入长度 1K token
    #   --min/max-tokens 1024         固定输出长度 1K token
    #   --parallel $P --number $P     并发数 = 总请求数 (即每个 worker 各发 1 条)
    #   --stream                      流式，能拿到 TTFT/TPOT
    #   --read-timeout 1800           30 min 读超时，防大并发档被掐
    #   enable_thinking: false        关掉 Qwen3 的 thinking 模式 (省 token、稳延迟)

    if [ "$ROUND" = "record" ]; then
      echo "--- [$LABEL] METRICS (kept) ---" | tee -a $SUMMARY
    else
      echo "--- [$LABEL] (warmup, discarded) ---" | tee -a $SUMMARY
    fi
    # 从日志里抓核心指标进汇总：吞吐、首 token 延迟、每 token 延迟、端到端延迟、成功/失败数
    grep -E 'Output token throughput|Average TTFT|Average TPOT|Average latency|Succeed requests|Failed requests|Total tokens|Output tokens' $LOG | tee -a $SUMMARY
    echo "" | tee -a $SUMMARY
  done
done

echo "=== END $(date -u +%H:%M:%S) ===" | tee -a $SUMMARY
