#!/usr/bin/env bash
# 在一个 worker 上跑完一整个 benchmark config（6 个 cell）。
#
# 与 bodaborg 那套最大的不同：这里 pod 长期存活，没有 60 分钟上限，
# 所以不必再把 config 切成单 cell 一轮一个 —— 那套切分纯粹是为了绕开 pod 寿命。
#
# 用法: bash run-config.sh <config名> [模式]
#   模式 single (默认) = 强制单机视图，只用本机 8 device（路线 A）
#   模式 ray           = 保留 GKE 注入的多机 mesh env（路线 B，需先起 Ray）
set -uo pipefail

CFG_NAME="${1:?需要 config 名，如 qwen3.5-397b-fp8-tp8-dp1-ep}"
MODE="${2:-single}"
# 可选：只跑部分 cell。空则跑 config 原有的全部 6 个。
# 一次 server 启动要付 13 分钟固定成本，所以同一节点上的多个 cell 必须共用一次启动 ——
# 按 cell 切分再各自起 server 是上一套（bodaborg 60 分钟 pod 上限）逼出来的，这里不需要。
CELL_ISL_OSL="${3:-}"
CELL_CONC="${4:-}"
SRC=/work/vllm-torchtpu
MODEL_DIR=/work/models/qwen3.5-397b
OUT=/work/results/$CFG_NAME-$MODE

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
# 编译缓存落 tmpfs 并随 pod 长存。上周在 bodaborg 实测：冷编译 19 分钟（1141s），
# 有缓存后 0 个新图、16 次 pickle.load。这是单笔最大的启动时间节省。
export VLLM_CACHE_ROOT=/work/vllmcache
mkdir -p "$VLLM_CACHE_ROOT" "$OUT"

if [ "$MODE" = "single" ]; then
  # GKE 注入的是 2x2x4 全 mesh。降回单机视图，本机 8 device 独立成一个 slice。
  # 已实测：jax.devices() 返回 8，device_kind=TPU7x，matmul 正常。
  export TPU_WORKER_ID=0
  export TPU_PROCESS_ADDRESSES=localhost:8471
  export TPU_WORKER_HOSTNAMES=localhost
  export TPU_HOST_BOUNDS=1,1,1
  export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  export TPU_ACCELERATOR_TYPE=tpu7x-8
  unset TPU_MULTIHOST_BACKEND
else
  export TPU_MULTIHOST_BACKEND=ray
fi

ts(){ echo "@@ $1 $(date -u +%H:%M:%S)"; }
ts CFG_START

# 把 config 复制一份改 MODEL 指到本地权重，不动仓库里的原件。
# 结果 key 是 isl*_osl*_c*，与模型名无关，所以改 MODEL 不影响和 baseline 对齐。
CFG_SRC="$SRC/scripts/vllm/benchmarking/configs/$CFG_NAME.sh"
[ -f "$CFG_SRC" ] || { echo "RUN_FAILED: 找不到 config $CFG_SRC"; exit 1; }
# sed -i 是就地改。不留原件的话，第二次跑若不传 cell 参数会静默沿用上一次的限制，
# 结果少跑几个 cell 却看不出来 —— 这种「悄悄少测」比报错难发现得多。
[ -f "$CFG_SRC.orig" ] || cp "$CFG_SRC" "$CFG_SRC.orig"
cp "$CFG_SRC.orig" "$CFG_SRC"
sed -i "s#^MODEL=.*#MODEL=\"$MODEL_DIR\"#" "$CFG_SRC"
[ -n "$CELL_ISL_OSL" ] && sed -i "s#^ISL_OSL_CONFIGS=.*#ISL_OSL_CONFIGS=\"$CELL_ISL_OSL\"#" "$CFG_SRC"
[ -n "$CELL_CONC" ]    && sed -i "s#^CONCURRENCY_OPTIONS=.*#CONCURRENCY_OPTIONS=\"$CELL_CONC\"#" "$CFG_SRC"
grep -E '^(MODEL|TENSOR_PARALLELISM|DATA_PARALLELISM|MAX_NUM_SEQS|ISL_OSL_CONFIGS|CONCURRENCY_OPTIONS|BENCHMARK_TEMPERATURE)=' "$CFG_SRC"

# server.log 在容器内文件里，实时接到 stdout 才看得见启动失败的真因。
( while true; do
    L=$(ls -t $SRC/benchmark_runs/*/server.log 2>/dev/null | head -1)
    [ -n "$L" ] && { echo "=== tailing $L ==="; tail -F "$L" 2>/dev/null | sed 's/^/[srv] /'; }
    sleep 10
  done ) &
TAILER=$!

# 结果一产出就往持久处拷。pod 理论上长存，但 tmpfs 里的东西一旦 pod 没了就全丢。
#
# 只拷「本次调用产生的那个 run 目录」，不要 cp -r benchmark_runs/*。
# 拷全部会把该 pod 上所有历史 run 复制进每一个 config 的输出目录，导致
# 同一份结果在多处各存一份、且拷贝时间是新的 —— 按时间过滤也挡不住。
# 实测这个 bug 让 18 格的收集结果膨胀成 78 格。
RUN_DIR_MARK="$OUT/.started_at"
date +%s > "$RUN_DIR_MARK"
( while true; do
    sleep 60
    for d in $(find "$SRC/benchmark_runs" -maxdepth 1 -mindepth 1 -type d \
                 -newer "$RUN_DIR_MARK" 2>/dev/null); do
      cp -r "$d" "$OUT/" 2>/dev/null
    done
  done ) &
COPIER=$!

cd "$SRC"
bash ./scripts/vllm/benchmarking/run_benchmarks.sh --config "$CFG_NAME" 2>&1
RC=$?
ts CFG_DONE

kill $TAILER $COPIER 2>/dev/null
for d in $(find "$SRC/benchmark_runs" -maxdepth 1 -mindepth 1 -type d -newer "$RUN_DIR_MARK" 2>/dev/null); do
  cp -r "$d" "$OUT/" 2>/dev/null
done
echo "=== 结果文件 ==="
find "$OUT" -name '*.json' | head -20
echo "RUN_RC=$RC"
echo "CONFIG_DONE"
# 必须把失败码透出去。最后一行是 echo 的话，脚本永远退出 0 ——
# 内部 RUN_RC=1 明明失败，调用方（CI / 编排脚本 / watch agent）看到的却是成功。
# 实测这个 bug 让一个 51 秒就崩掉的 run 被上报成「A_RC=0 成功」，直接误导了结论。
exit $RC
