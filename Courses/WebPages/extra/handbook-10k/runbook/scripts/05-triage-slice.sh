#!/usr/bin/env bash
# =============================================================================
# 05-triage-slice.sh —— §5.6：一个切片出问题时，一次把三层的现场都抓下来（只读）
#
# 三层 = Slice CR / 事件（K8s）→ kueue-slice-controller（集群内，我们装的）→ GKE Slice Controller（托管，日志在 Cloud Logging）
# 抓什么：
#   - 三个「slice controller」各去哪看日志；kueue-slice-controller 日志命令
#   - GKE Slice Controller 日志过滤器 logName="projects/<P>/logs/slice-controller" AND textPayload=~"<slice>"
#   - GCE 侧 MIG detach 审计：resource.type="gce_instance_group_manager" protoPayload.methodName="v1.compute.instanceGroupManagers.patch"
#   - Workload 条件：05 §5.2（kubectl describe workload）
#
# 用法：./05-triage-slice.sh <slice> [--since 2h]
# 输出同时写 $STATE_DIR/triage-<slice>-<时间>.txt，升级时整份附上（08 §8.5）。只读，可随时重跑。
# 前置：kubectl 读权限；gcloud logging read 需要 roles/logging.viewer（审计日志还要 privateLogViewer）。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT REGION ZONE CLUSTER
SLICE="${1:?用法：$0 <slice> [--since 2h]}"; SINCE=2h; [[ "${2:-}" == --since ]] && SINCE="$3"
OUT="$STATE_DIR/triage-$SLICE-$(date -u +%Y%m%dT%H%M%SZ).txt"
exec > >(tee "$OUT") 2>&1
h() { echo; echo "===== $* ====="; }
x() { echo "+ $*"; "$@" 2>&1 || echo "  (命令失败，rc=$?)"; }

h "1. Slice CR（Ready.reason / message / finalizers / deletionTimestamp / partitionIds）"
x kubectl get slices.accelerator.gke.io "$SLICE" -o wide
x kubectl describe slices.accelerator.gke.io "$SLICE"
owner=$(kubectl get slices.accelerator.gke.io "$SLICE" -o jsonpath='{.metadata.annotations.accelerator\.gke\.io/owner-workload-name}' 2>/dev/null || true)
ons=$(kubectl get slices.accelerator.gke.io "$SLICE" -o jsonpath='{.metadata.annotations.accelerator\.gke\.io/owner-workload-namespace}' 2>/dev/null || true)

h "2. 事件（对象 = 这个 Slice；以及同名 Workload）"
x kubectl get events -A --field-selector "involvedObject.name=$SLICE" --sort-by=.lastTimestamp
if [[ -n "$owner" ]]; then
  h "3. Kueue Workload $ons/$owner（admissionChecks、Evicted 条件与原因）"
  x kubectl describe workload -n "$ons" "$owner"
  x kubectl get events -n "$ons" --field-selector "involvedObject.name=$owner" --sort-by=.lastTimestamp
fi

# partition id 的层级随形状：多 cube 片（4x4x4 及以上）用 4x4x4-id；sub-cube 片（2x2x1/2x2x2/2x2x4/2x4x4）的 id
# 就是同名层级的 id（实测：2x2x4 片按 4x4x4-id 查 "No resources found"，按 partition-2x2x4-id 查到 4 台）。
topo=$(kubectl get slices.accelerator.gke.io "$SLICE" -o jsonpath='{.spec.topology}' 2>/dev/null || true)
if [[ "$topo" =~ ^(2x2x1|2x2x2|2x2x4|2x4x4)$ ]]; then lvl="$topo"; else lvl=4x4x4; fi
h "4. 节点侧：这个切片的 partition 各自的状态标签（按 partition-${lvl}-id 查；期望 HEALTHY、4x4x4 级 16/16 Ready）"
for p in $(kubectl get slices.accelerator.gke.io "$SLICE" -o jsonpath='{.spec.partitionIds[*]}' 2>/dev/null); do
  echo "-- $p"
  kubectl get nodes -l "cloud.google.com/gke-tpu-partition-${lvl}-id=$p" \
    -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-partition-${lvl}-state,cloud.google.com/gke-tpu-slice --no-headers 2>&1 \
    | awk '{print "   "$0}'
done

h "5. 重叠审计：同一个 partition 出现在多个 Slice"
kubectl get slices.accelerator.gke.io -o json | jq -r '.items[] | .metadata.name as $n | .spec.partitionIds[] | "\(.)\t\($n)"' \
  | sort | awk -F'\t' '{c[$1]++; s[$1]=s[$1]" "$2} END{for(k in c) if(c[k]>1) print "  重叠 "k":"s[k]}' | grep . || echo "  无重叠"

h "6. kueue-slice-controller（集群内）日志，过滤本切片"
x bash -c "kubectl logs deployment/slice-controller-controller-manager -n slice-controller-system -c manager --since=$SINCE | grep -F '$SLICE' | tail -100"

h "7. GKE Slice Controller（托管）日志 [需验证：使用方项目是否可见]"
x gcloud logging read "logName=\"projects/$PROJECT/logs/slice-controller\" AND textPayload=~\"$SLICE\"" \
  --project="$PROJECT" --freshness="$SINCE" --limit=200 --format='value(timestamp,textPayload)'

h "8. K8s 审计：谁建/删/改了这个 Slice"
x gcloud logging read "protoPayload.resourceName:\"accelerator.gke.io/v1beta1/slices/$SLICE\"" \
  --project="$PROJECT" --freshness="$SINCE" --limit=50 --format='value(timestamp,protoPayload.methodName,protoPayload.authenticationInfo.principalEmail)'

h "9. GCE 读配额 429（Ready=Unknown / MMIGHealthProbeFailed 时看）"
x gcloud logging read "logName=\"projects/$PROJECT/logs/slice-controller\" AND (textPayload=~\"rate limit exceeded\" OR textPayload=~\"429\")" \
  --project="$PROJECT" --freshness="$SINCE" --limit=20 --format='value(timestamp,textPayload)'

echo; echo "已存：$OUT   下一步：对照 05 §5.6 分诊表；删除卡住 → 05-delete-slice-chain.sh $SLICE（先 dry-run）"
