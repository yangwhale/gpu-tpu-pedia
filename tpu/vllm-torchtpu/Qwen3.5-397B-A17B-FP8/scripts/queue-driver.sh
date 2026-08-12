#!/usr/bin/env bash
# vllm-torchtpu 测试队列 driver。
# 每个任务一个独立 pod（60 分钟上限），跑完自动起下一个。
# 日志走容器 stdout（kubectl logs 可读），结果落本机 ~/ttpu-runs/results/。
set -uo pipefail

export KUBECONFIG=/tmp/kc-bodaborg.yaml
NS=priority-dev
BASE_YAML=/tmp/v7-r5.yaml
SRC_TGZ=/tmp/vtt.tgz
SETUP=~/gpu-tpu-pedia/tpu/vllm-torchtpu/Qwen3.5-397B-A17B-FP8/scripts/setup-in-pod.sh
OUT=~/ttpu-runs/results
QUEUE=~/ttpu-runs/queue.txt
STATE=~/ttpu-runs/state.txt
mkdir -p "$OUT"; touch "$STATE"

log(){ echo "[$(TZ=Asia/Hong_Kong date +%H:%M:%S)] $*"; }

run_one() {
  local TASK="$1" IDX="$2"
  local NAME="ttpu-q${IDX}"
  local POD_YAML=/tmp/${NAME}.yaml
  log "=== 任务 $IDX: $TASK ==="

  sed -e "s/chrisya-ttpu-r5/chrisya-${NAME}/" "$BASE_YAML" > "$POD_YAML"
  kubectl delete jobset "chrisya-${NAME}" -n $NS --wait=false >/dev/null 2>&1
  sleep 5
  local APPLY_OUT; APPLY_OUT=$(kubectl apply -f "$POD_YAML" 2>&1)
  log "apply: $APPLY_OUT"
  case "$APPLY_OUT" in *created*|*configured*|*unchanged*) ;; *) log "apply 失败，跳过"; return 1;; esac

  # 等 pod Running（最多 6 分钟；共享集群偶尔要等 NAP 扩容）
  local POD="" i
  for i in $(seq 1 72); do
    POD=$(kubectl get pods -n $NS -o name 2>/dev/null | grep "chrisya-${NAME}-slice" | head -1 | cut -d/ -f2)
    [ -n "$POD" ] && [ "$(kubectl get pod "$POD" -n $NS -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ] && break
    sleep 5
  done
  if [ -z "$POD" ]; then
    log "pod 没起来 —— 往上层查真因："
    log "  jobset: $(kubectl get jobset chrisya-${NAME} -n $NS -o jsonpath='{.status.conditions}' 2>/dev/null)"
    log "  jobs:   $(kubectl get jobs -n $NS 2>/dev/null | grep -c "chrisya-${NAME}") 个"
    log "  ctrl err: $(kubectl logs -n jobset-system -l control-plane=controller-manager --tail=200 2>/dev/null | grep -F "chrisya-${NAME}" | grep -o 'error\":.*' | tail -1 | cut -c1-260)"
    log "  workload: $(kubectl get workload -n $NS 2>/dev/null | grep "chrisya-${NAME}" | head -1)"
    return 1
  fi
  log "pod=$POD"

  kubectl exec -i "$POD" -n $NS -- bash -c 'mkdir -p /work && cd /work && tar xzf -' < "$SRC_TGZ" >/dev/null 2>&1
  kubectl cp "$SETUP" "$NS/$POD:/work/setup.sh" >/dev/null 2>&1

  local TOK; TOK=$(gcloud auth print-access-token 2>/dev/null)   # 每个任务重取，token 只活 1 小时

  # 任务脚本：unit 走 pytest，其余走 benchmark runner
  kubectl exec -i "$POD" -n $NS -- bash -c "cat > /work/task.sh" <<TASKEOF
set -x
export HF_HOME=/work/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
mkdir -p /work/hf
bash /work/setup.sh /work/vllm-torchtpu 2>&1 | tail -5
cd /work/vllm-torchtpu
date -u +"PHASE task_start %H:%M:%S"
if [ "$TASK" = "unit" ]; then
  python3 -m pip install --no-cache-dir pytest pytest-timeout 2>&1 | tail -1
  git config --global --add safe.directory /work/vllm-torchtpu
  python3 -m pytest tests -q --timeout=300 2>&1 | tail -60
else
  python3 -m pip install --no-cache-dir hf_transfer 2>&1 | tail -1
  bash ./scripts/vllm/benchmarking/run_benchmarks.sh --config "$TASK" 2>&1
fi
echo "TASK_RC=\$?"
date -u +"PHASE task_end %H:%M:%S"
echo "=== 结果文件 ==="; find /work/vllm-torchtpu/benchmark_runs -name "*.json" 2>/dev/null | head -10
for f in \$(find /work/vllm-torchtpu/benchmark_runs -name "*.json" 2>/dev/null | head -10); do echo "--- \$f ---"; cat "\$f"; done
echo "TASKDONE"
TASKEOF

  kubectl exec "$POD" -n $NS -- bash -c \
    "AR_TOKEN='$TOK' nohup bash -c 'bash /work/task.sh 2>&1 | tee /work/task.log > /proc/1/fd/1' >/dev/null 2>&1 &" >/dev/null 2>&1

  # 轮询到 TASKDONE 或 pod 结束（最多 58 分钟）
  local j
  for j in $(seq 1 116); do
    sleep 30
    local phase; phase=$(kubectl get pod "$POD" -n $NS -o jsonpath='{.status.phase}' 2>/dev/null)
    if kubectl logs "$POD" -n $NS --tail=5 2>/dev/null | grep -q TASKDONE; then log "任务完成"; break; fi
    [ "$phase" != "Running" ] && { log "pod 结束(phase=$phase)"; break; }
  done

  kubectl logs "$POD" -n $NS 2>/dev/null > "$OUT/${IDX}-${TASK}.log"
  log "日志已存 $OUT/${IDX}-${TASK}.log ($(wc -l < "$OUT/${IDX}-${TASK}.log") 行)"
  kubectl delete jobset "chrisya-${NAME}" -n $NS --wait=false >/dev/null 2>&1
  echo "$TASK" >> "$STATE"
}

IDX=0
while read -r TASK; do
  [ -z "$TASK" ] && continue
  case "$TASK" in \#*) continue;; esac
  IDX=$((IDX+1))
  grep -qxF "$TASK" "$STATE" && { log "跳过已完成 $TASK"; continue; }
  run_one "$TASK" "$IDX"
done < "$QUEUE"
log "=== 队列跑完 ==="
