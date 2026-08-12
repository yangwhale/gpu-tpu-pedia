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
  # 必须等旧的真正消失再 apply。之前 --wait=false + sleep 5 就提交，
  # 结果拿到上一轮残留、正在终止的同名 pod（名字 hash 一样），日志取到 0 字节。
  kubectl delete jobset "chrisya-${NAME}" -n $NS --wait=false >/dev/null 2>&1
  local w
  for w in $(seq 1 60); do
    kubectl get jobset "chrisya-${NAME}" -n $NS >/dev/null 2>&1 || break
    sleep 3
  done
  for w in $(seq 1 60); do
    [ -z "$(kubectl get pods -n $NS -o name 2>/dev/null | grep "chrisya-${NAME}-slice")" ] && break
    sleep 3
  done
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
# runner 默认等 server 90 分钟，比 pod 的 60 分钟寿命还长，导致它「超时后打印 server.log」
# 那段代码永远执行不到 —— 三轮都看不到 server 起不来的真因。压到 25 分钟。
export SERVER_READY_WAIT_MIN=25
mkdir -p /work/hf
# setup 失败必须立刻停：之前 GitHub 503 导致 vLLM 没装上，脚本却继续往下跑 benchmark，
# 最后报一句误导性的 "'vllm bench serve' not available"。
# 失败时也要打 TASKDONE，否则 driver 会一直轮询到 pod 60 分钟到期 —— 白等一个窗口
if ! bash /work/setup.sh /work/vllm-torchtpu 2>&1 | tail -40; then echo "SETUP_FAILED"; echo "TASKDONE"; exit 1; fi
python3 -c "import vllm" 2>/dev/null || { echo "SETUP_FAILED: vllm 不可 import"; echo "TASKDONE"; exit 1; }
cd /work/vllm-torchtpu
# 后台把 server.log 实时接到 stdout —— 它在容器内文件里，pod 一死就没了
( while true; do
    L=\$(ls -t /work/vllm-torchtpu/benchmark_runs/*/server.log 2>/dev/null | head -1)
    [ -n "\$L" ] && { echo "=== tailing \$L ==="; tail -F "\$L" 2>/dev/null | sed 's/^/[srv] /'; }
    sleep 10
  done ) &
date -u +"PHASE task_start %H:%M:%S"
if [ "$TASK" = "unit" ]; then
  python3 -m pip install --no-cache-dir pytest pytest-timeout 2>&1 | tail -1
  git config --global --add safe.directory /work/vllm-torchtpu
  # 不要用 | tail -N：pytest 全跑完才输出，pod 被杀就什么都拿不到
  python3 -m pytest tests -q --timeout=300 2>&1
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

  # 取日志要校验非空并重试：pod 正在终止时 kubectl logs 可能返回空，
  # 而日志是这一整个 60 分钟窗口唯一的产出，丢了就等于白跑。
  local F="$OUT/${IDX}-${TASK}.log" k
  for k in 1 2 3 4 5; do
    kubectl logs "$POD" -n $NS 2>/dev/null > "$F"
    [ -s "$F" ] && break
    log "第 $k 次取日志为空，3s 后重试"; sleep 3
  done
  if [ -s "$F" ]; then
    log "日志已存 $F ($(wc -l < "$F") 行)"
  else
    log "⚠ 日志取不到（pod=$POD 可能已被回收）—— 本任务无产出"
    kubectl get pod "$POD" -n $NS -o jsonpath='{.status.phase} {.status.reason} {.status.message}' >> "$F" 2>/dev/null
  fi
  kubectl delete jobset "chrisya-${NAME}" -n $NS --wait=false >/dev/null 2>&1
  echo "$TASK" >> "$STATE"
}

# ── 启动前自检：把 task.sh 按 heredoc 规则 render 出来做语法检查 ──
# 连着两轮死在生成物上（$L unbound、limits 重复键），都不是 driver 本身的问题。
# 生成的东西必须先验再用。用 python 做 render，避免 shell 嵌套引号自己打架。
preflight() {
  python3 - "$0" <<'PFPY'
import re,subprocess,sys,tempfile,os
src=open(sys.argv[1],encoding='utf-8').read()
m=re.search(r'<<TASKEOF\n(.*?)\nTASKEOF\n', src, re.S)
if not m: print("PREFLIGHT: 找不到 TASKEOF 块"); sys.exit(1)
body=m.group(1)
f=tempfile.NamedTemporaryFile('w',suffix='.body',delete=False,encoding='utf-8'); f.write(body); f.close()
r=subprocess.run(['bash','-c','TASK=__preflight__; eval "cat <<TASKEOF\n$(cat %s)\nTASKEOF"'%f.name],
                 capture_output=True,text=True)
os.unlink(f.name)
if r.returncode!=0: print("PREFLIGHT: render 失败", r.stderr[:300]); sys.exit(1)
rendered=r.stdout
g=tempfile.NamedTemporaryFile('w',suffix='.sh',delete=False,encoding='utf-8'); g.write(rendered); g.close()
c=subprocess.run(['bash','-n',g.name],capture_output=True,text=True); os.unlink(g.name)
if c.returncode!=0: print("PREFLIGHT: 生成的 task.sh 语法错\n"+c.stderr[:400]); sys.exit(1)
for need in ['SERVER_READY_WAIT_MIN=25','L=$(ls -t','SETUP_FAILED','pytest tests -q']:
    if need not in rendered: print("PREFLIGHT: 渲染结果缺少 "+need); sys.exit(1)
print("PREFLIGHT OK ({} 行)".format(len(rendered.split(chr(10)))))
PFPY
  [ $? -ne 0 ] && { log "启动前自检未通过，不提交任何任务"; exit 1; }
}
preflight

IDX=0
while read -r TASK; do
  [ -z "$TASK" ] && continue
  case "$TASK" in \#*) continue;; esac
  IDX=$((IDX+1))
  grep -qxF "$TASK" "$STATE" && { log "跳过已完成 $TASK"; continue; }
  run_one "$TASK" "$IDX"
done < "$QUEUE"
log "=== 队列跑完 ==="
