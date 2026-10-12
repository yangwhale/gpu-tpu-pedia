#!/usr/bin/env bash
# =============================================================================
# 07-fault-drill.sh —— §7.3：对一个在跑、开了 MTC 的多片 JobSet 注入一种故障，并把各层时间线录下来
#
# 三种故障与出处：
#   pod-kill      kubectl delete pod <worker-1 的 pod>。预期：backoffLimit 0 → Job BackoffLimitExceeded → JobSet RestartJobSet；切片不重切 [推断，07 §7.3 表]
#   slice-delete  kubectl delete slice <该任务的一个 Slice>。已实测：`Slice has been deleted, evicting workload` →
#                 全部切片重建 → 约 2 min 恢复准入
#   vm-stop       gcloud compute instances stop <worker-1 所在节点>。v6e sub-slice 上测过：
#                 被停节点所在 Slice → FAILED "Partition is in failed state"，其 Job BackoffLimitExceeded；节点恢复后 Slice
#                 仍 FAILED、需删了重建。
#                 tpu7x 多片 + Kueue 下：07 §7.3 那组 3 片实验（第三次演练）实测过——stop 后约 21 s Slice not operational →
#                 Kueue 驱逐 → 换 cube 重切，从 stop 起约 106 s 全 ACTIVE（07 §7.3 时序图）。那是手工 gcloud 打的，**不是用本脚本跑的**；
#                 本脚本的 vm-stop 路径本身 [需验证]。
#                 恢复：instances start；10 min 不回来用 MIG recreate-instances（abandon-instances 禁用）。
#                 注意：GKE 节点池 autoRepair=true 可能先一步修/重建这台 VM，时间线里会看到。
#                 公开文档里 GKE 对 TPU 的「手动触发维护」走 GKE 自己的流程（先 taint、驱逐、再请求 GCE）：
#                 https://docs.cloud.google.com/kubernetes-engine/docs/concepts/handle-disruption-gpu-tpu ——
#                 它模拟的是计划维护而不是突发宕机，本脚本不用它。
#                 `gcloud compute instances simulate-maintenance-event` 对 tpu7x 可用：第三次演练时手工打过。
#                 本脚本暂未接入这种模式（要做计划维护演练就按 07 §7.3 手工跑那条命令）。
#
# 时间线（§7.3 第 3 步要记的 5 个时刻）：后台 `kubectl get -w --output-watch-events` 录 workloads / slices / jobset / pods，
#   每条带 UTC 时间戳，全部套 `timeout`（录制自带死期，脚本被杀也不会留孤儿进程）。
#
# 用法：
#   ./07-fault-drill.sh --jobset <name> --mode pod-kill|slice-delete|vm-stop [--minutes 45] [--execute]
#   ./07-fault-drill.sh --restore-vm <node>  [--execute]      # vm-stop 之后把 VM 拉回来
# 默认 dry-run：打印选中的目标和将执行的命令，不注入、不录制。
# 前置：**任务需 ≥ 2 片**——目标 pod 取 job-index=1（第 2 片），1 片的任务选不到目标、脚本会打印「目标 pod=?」；
#       模板用 07-jobset-mtc.yaml（SLICES ≥ 2）。
#       任务已跑过至少一次本地检查点（LOCAL_CHECKPOINT_PERIOD 步）；有空闲健康 cube 可供重切（05-list-partitions.sh --free
#       ≥ 该任务 cube 数，否则 Kueue 只能等）；gcloud compute.instances.stop/start 权限；jq。
# 幂等：每次运行注入一次故障——重跑 = 再打一次，别在上一次恢复完之前重跑。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT REGION ZONE CLUSTER

JS=""; MODE=""; MIN=45; EXEC=0; RESTORE=""
while (($#)); do case "$1" in
  --jobset) JS="$2"; shift;; --mode) MODE="$2"; shift;; --minutes) MIN="$2"; shift;;
  --execute) EXEC=1;; --restore-vm) RESTORE="$2"; shift;; *) echo "未知参数 $1" >&2; exit 64;; esac; shift; done
run() { echo "+ $*"; if [[ $EXEC == 1 ]]; then "$@"; else echo "  (dry-run，未执行)"; fi; }
ts() { date -u +%FT%TZ; }

if [[ -n "$RESTORE" ]]; then
  run gcloud compute instances start "$RESTORE" --zone="$ZONE" --project="$PROJECT"
  mig=$(kubectl get node "$RESTORE" -o jsonpath='{.metadata.annotations.node\.gke\.io/created-by-mig}' 2>/dev/null | awk -F/ '{print $NF}')
  echo "10 min 内节点没回到 Ready：gcloud compute instance-groups managed recreate-instances ${mig:-<MIG>} --instances=$RESTORE --zone=$ZONE --project=$PROJECT"
  echo "验证：kubectl get node $RESTORE（Ready）；kubectl get node $RESTORE -L cloud.google.com/gke-tpu-partition-4x4x4-state（HEALTHY）；"
  echo "      替换节点回来后核对 gke-tpu-partition-* 标签（07 §7.5 第 4 条）；该 cube 的旧 Slice 若仍 FAILED，Kueue 路径会自己删，BYO 路径要手删重建"
  exit 0
fi
[[ -n "$JS" && "$MODE" =~ ^(pod-kill|slice-delete|vm-stop)$ ]] || { echo "用法见文件头" >&2; exit 64; }

# ---- 目标
wl=$(kubectl get workloads -n "$NS" -o json | jq -r --arg js "$JS" '.items[]|select(any(.metadata.ownerReferences[]?; .kind=="JobSet" and .name==$js))|.metadata.name' | head -1)
pod=$(kubectl get pods -n "$NS" -l "jobset.sigs.k8s.io/jobset-name=$JS,jobset.sigs.k8s.io/job-index=1,batch.kubernetes.io/job-completion-index=0" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
node=$( [[ -n "$pod" ]] && kubectl get pod -n "$NS" "$pod" -o jsonpath='{.spec.nodeName}' || true)
slice=$(kubectl get slices.accelerator.gke.io -o json | jq -r --arg wl "$wl" '[.items[]|select(.metadata.annotations["accelerator.gke.io/owner-workload-name"]==$wl)|.metadata.name]|sort|.[0] // ""')
echo "JobSet=$NS/$JS  Workload=${wl:-?}  目标 pod=${pod:-?}  节点=${node:-?}  目标 Slice=${slice:-?}"
last=$( [[ -n "$pod" ]] && kubectl logs -n "$NS" "$pod" -c jax-tpu --tail=2000 2>/dev/null | grep -o 'completed step: [0-9]*' | tail -1 || true)
echo "当前进度：${last:-读不到}（确认已过至少一次 LOCAL_CHECKPOINT_PERIOD）"
echo "可用备件 cube：$("$(dirname "$0")"/05-list-partitions.sh --free 2>&1 >/dev/null | tail -1)"

case $MODE in pod-kill) tgt=$pod;; slice-delete) tgt=$slice;; vm-stop) tgt=$node;; esac
[[ -n "$tgt" ]] || { echo "找不到 $MODE 的目标（JobSet 在跑吗？NS=$NS）" >&2; exit 1; }

# ---- 录制（先开录，再注入）
D="$STATE_DIR/drill-$JS-$MODE-$(date -u +%Y%m%dT%H%M%SZ)"; SEC=$(( MIN * 60 ))
w_workloads() { kubectl get workloads -n "$NS" -w --output-watch-events \
  -o 'custom-columns=NAME:.metadata.name,ADMITTED:.status.conditions[?(@.type=="Admitted")].status,EVICTED:.status.conditions[?(@.type=="Evicted")].reason'; }
w_slices()    { kubectl get slices.accelerator.gke.io -w --output-watch-events; }
w_jobset()    { kubectl get jobset -n "$NS" "$JS" -w --output-watch-events -o 'custom-columns=NAME:.metadata.name,RESTARTS:.status.restarts,SUSPEND:.spec.suspend'; }
w_pods()      { kubectl get pods -n "$NS" -l "jobset.sigs.k8s.io/jobset-name=$JS" -w --output-watch-events -o wide; }
w_events()    { kubectl get events -n "$NS" -w --output-watch-events; }
export -f w_workloads w_slices w_jobset w_pods w_events; export NS JS
if [[ $EXEC == 1 ]]; then
  mkdir -p "$D"
  for n in workloads slices jobset pods events; do
    ( timeout "$SEC" bash -c "w_$n" 2>&1 | while IFS= read -r l; do echo "$(ts) $l"; done > "$D/$n.log" ) &
  done
  sleep 5
else
  echo "将在后台录制 ${MIN} min（timeout 兜底）到 $D/：workloads slices jobset pods events"
fi

# ---- 注入
echo "$(ts) 注入 $MODE" | tee -a "${D}/inject.log" 2>/dev/null || true
case $MODE in
  pod-kill)     run kubectl delete pod -n "$NS" "$pod" --wait=false;;
  slice-delete) run kubectl delete slices.accelerator.gke.io "$slice" --wait=false;;
  vm-stop)      run gcloud compute instances stop "$node" --zone="$ZONE" --project="$PROJECT" --async
                echo "恢复：$0 --restore-vm $node --execute";;
esac

cat <<EOF

要看的（§7.3 第 3 步），全在 $D/*.log：
  故障时刻                    inject.log
  Workload Evicted / 重新准入  workloads.log  （EVICTED 列出现原因；ADMITTED False→True）
  新 Slice ACTIVE             slices.log     （AGE 归零 = 重切；STATE → ACTIVE）
  JobSet 重启次数              jobset.log     （RESTARTS +1）
  第一条 restore 日志           kubectl logs -n $NS <新 worker-0-0 pod> -c jax-tpu | grep -m1 -E "Found a restore directory|Restoring checkpoint"
  第一个新 step                 ... | grep -m1 "completed step:"（N 应 > 故障前最后一个本地检查点，07 §7.3 第 4 步）
  预期：pod-kill → RESTARTS+1、切片不重切 [推断]；slice-delete → 全部切片重建、约 2 min 重新准入；vm-stop → Slice FAILED → 驱逐 → 重切（第三次演练手工实测约 106 s 全 ACTIVE；本脚本路径未实测）
录制在 ${MIN} min 后自动停止；提前停：pkill -f "[k]ubectl get .* -w --output-watch-events"（只杀录制，不影响任务）。
EOF
