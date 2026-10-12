#!/usr/bin/env bash
# =============================================================================
# 08-daily-check.sh —— 08 §8.4 每日容量报表（也可当 §8.1 部分告警的手动巡检）
#
# 出处：
#   - 健康 cube 判据：每个 4x4x4-id 下 16 节点 Ready 且 partition-4x4x4-state=HEALTHY（或约定的 +DEGRADED）
#   - 配额上限 = min(CUBES_MAX_JOB, 配额口径健康 cube − 备件 S) × 64，口径与 03-addons.sh quota 逐条一致：
#     HEALTHY（不认 DEGRADED）+ 16 节点全 Ready + allocatable google.com/tpu=4 + 无 PartOfBrokenSlice taint（05-isolate-cube.sh 打的）；CQ 名/资源名
#   - sub-block 字段 healthInfo.* / reservationSubBlockMaintenance.*
#   - reservation deleteAtTime，提前 30/14/3 天提醒
#   - GCE 实例状态 REPAIRING（https://cloud.google.com/compute/docs/instances/instance-lifecycle）；告警 #11 阈值 > 1 h
#   - 节点池非 RUNNING（告警 #16）；Ready 但 google.com/tpu allocatable ≠ 4（告警 #9）
#
# 用法：
#   ./08-daily-check.sh                    # 按 HEALTHY 口径
#   ./08-daily-check.sh --include-degraded # 按 HEALTHY+DEGRADED 口径（09 对齐清单约定的是哪种就用哪种）
#     注意：这个口径看的是 K8s 节点标签 partition-4x4x4-state。实测集群上（两千二百多个 TPU 节点）该标签只有
#     HEALTHY / UNHEALTHY（sub-cube 级另有 INCOMPLETE），没出现过 DEGRADED——GCE sub-block 报 DEGRADED(15/15) 的 cube
#     在 K8s 标签上是 UNHEALTHY，所以两种口径输出相同（实测）。GCE 侧的 DEGRADED 看本报表第 2 段
#     sub-block healthInfo。DEGRADED 在别的情况下会不会出现在 K8s 标签上没法从这次判断，判据先保留。
# 只读，可随时重跑；报表同时写 $STATE_DIR/daily-<日期>.txt，TSV 明细写 $STATE_DIR/cubes-<日期>.tsv。
#   ./08-daily-check.sh --probe            # 只跑第 1–2 段（纯 kubectl），给 cron 每 5–10 min 用；别拿完整报表当高频告警
#                                          #（完整版每次调 GCE list、拉 24 h Kueue 日志，还会因「reservation ≤30 天」连续几周每 5 min 退出 1）
# 退出码：健康 cube < CUBES_SLO → 2（page）；< CUBES_SLO+2、配额超过健康容量、有 Slice 非 ACTIVE ≥60 min、pod 总数 >100k、
#         reservation ≤30 天 → 1（通知）。8 段已在实测集群上跑通（约 5 min，08 §8.4）。
#         第 1–2 段的隔离 cube 排除与 min(128) [需验证]。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT REGION ZONE CLUSTER RESERVATION

OKSTATES='["HEALTHY"]'; PROBE=0
for a in "$@"; do case "$a" in --include-degraded) OKSTATES='["HEALTHY","DEGRADED"]';; --probe) PROBE=1;; esac; done
S="${SPARE_CUBES:-}"; S_NOTE=""; [[ -z "$S" ]] && { S=0; S_NOTE="（SPARE_CUBES 未约定，按 0 算——09 D04）"; }
TS=$(date -u +%Y%m%d-%H%MZ); RPT="$STATE_DIR/daily-$TS.txt"; RC=0   # 文件名用 UTC（08 §8.4 约定），表头另写本地时间（带时区）
exec > >(tee "$RPT")

echo "== $(date '+%F %T %Z')  $CLUSTER  口径=$OKSTATES"

# ---- 1. cube 视图（K8s 节点标签） -------------------------------------------
kubectl get nodes -l cloud.google.com/gke-tpu-partition-4x4x4-id -o json | jq -r '
  [.items[] | {
     id:    .metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-id"],
     state: .metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-state"],
     pool:  .metadata.labels["cloud.google.com/gke-nodepool"],
     sb:    (.metadata.labels["cloud.google.com/gce-topology-subblock"] // ""),
     ready: ([.status.conditions[]? | select(.type=="Ready" and .status=="True")] | length),
     tpu:   (.status.allocatable["google.com/tpu"] // "0"),
     unsch: (.spec.unschedulable // false),
     iso:   (([.spec.taints[]?.key] | index("PartOfBrokenSlice")) != null) }]
  | group_by(.id)[] | [ .[0].id, .[0].pool, ([.[].state] | unique | join("/")), length,
      ([.[] | select(.ready==1)] | length),
      ([.[] | select(.ready==1 and .tpu!="4")] | length),
      ([.[] | select(.unsch)] | length),
      ([.[] | select(.iso)] | length) ] | @tsv' > "$STATE_DIR/cubes-$TS.tsv"
# 列：id pool state 节点数 Ready数 Ready但TPU≠4 cordoned 带PartOfBrokenSlice-taint数
total=$(wc -l < "$STATE_DIR/cubes-$TS.tsv")
healthy=$(jq -Rn --argjson ok "$OKSTATES" '[inputs | split("\t") | select((.[2] as $s | $ok | index($s)) and .[4]=="16" and .[5]=="0")] | length' < "$STATE_DIR/cubes-$TS.tsv")
partial=$(awk -F'\t' '$5>0 && $5<16' "$STATE_DIR/cubes-$TS.tsv" | wc -l)
tpubad=$(awk -F'\t' '$6>0' "$STATE_DIR/cubes-$TS.tsv" | wc -l)
cordon=$(awk -F'\t' '$7>0' "$STATE_DIR/cubes-$TS.tsv" | wc -l)
isolated=$(awk -F'\t' '$8>0' "$STATE_DIR/cubes-$TS.tsv" | wc -l)
# 配额口径（= 03-addons.sh quota）：只认 HEALTHY、16 台 Ready、TPU=4、没有被隔离；--include-degraded 不影响这一项
healthy_q=$(jq -Rn '[inputs | split("\t") | select(.[2]=="HEALTHY" and .[4]=="16" and .[5]=="0" and .[7]=="0")] | length' < "$STATE_DIR/cubes-$TS.tsv")
echo "健康 cube：$healthy / $total（SLO 线 $CUBES_SLO，缓冲线 $((CUBES_SLO+2))）"
echo "  按 state：$(cut -f3 "$STATE_DIR/cubes-$TS.tsv" | sort | uniq -c | tr '\n' ' ')"
echo "  15/16 类（部分 Ready）：$partial   Ready 但 TPU≠4：$tpubad（告警 #9）   有 cordoned 节点：$cordon（告警 #10）   已隔离（PartOfBrokenSlice）：$isolated"
(( healthy >= CUBES_SLO + 2 )) || { echo "  !! 低于缓冲线（告警 #1）"; RC=1; }
(( healthy >= CUBES_SLO )) || { echo "  !!! 低于 SLO 线（告警 #1 page）"; RC=2; }
echo "剩余备件：$(( healthy - CUBES_SLO )) cube 高于 SLO 线；约定备件 S=$S $S_NOTE"

# ---- 2. Kueue 配额 vs 健康容量 ----------------------------------------------
quota=$(kubectl get clusterqueues -o json | jq '[.items[].spec.resourceGroups[]?.flavors[]?.resources[]? | select(.name=="google.com/tpu") | .nominalQuota | tonumber] | add // 0')
cq=$(( healthy_q - S )); (( cq < 0 )) && cq=0
(( cq > CUBES_MAX_JOB )) && cq=$CUBES_MAX_JOB
cap=$(( cq * 64 ))
echo "Kueue Σ nominalQuota(google.com/tpu) = $quota；min($CUBES_MAX_JOB, 配额口径健康 $healthy_q − S $S) × 64 = $cap（口径同 03-addons.sh quota：HEALTHY + 未隔离）"
(( quota <= cap )) || { echo "  !! 配额超过健康容量 → 按 03 §3.5 调小"; (( RC >= 1 )) || RC=1; }
# --probe：cron 5–10 min 用的快速探针，到这里为止只用了 kubectl（不碰 GCE 读配额 经验教训第 52 条、不拉 24 h 日志），按严重度退出
if (( PROBE )); then echo "probe 退出码 $RC（0 正常 / 1 通知 / 2 page）"; exit $RC; fi

# ---- 3. sub-block 视图（GCE） -----------------------------------------------
sbj=$(gcloud compute reservations sub-blocks list "$RESERVATION" --block-name "$BLOCK" --zone "$ZONE" --project "$PROJECT" --format=json)
echo "sub-block：$(jq -r '[.[] | "\(.healthInfo.healthStatus)(\(.healthInfo.healthyHostCount//0)/\(.inUseHostCount//0))"] | group_by(.) | map("\(length)×\(.[0])") | join("  ")' <<<"$sbj")"
echo "  维护 pending/ongoing 的 sub-block：$(jq -r '[.[] | select((.reservationSubBlockMaintenance.maintenancePendingCount//0)+(.reservationSubBlockMaintenance.maintenanceOngoingCount//0) > 0) | (.name|split("-")|last)] | join(",")' <<<"$sbj")"

# ---- 4. 修理中的 VM、非 RUNNING 的池 ----------------------------------------
echo "REPAIRING 的 VM：$(gcloud compute instances list --project "$PROJECT" --filter="zone:($ZONE) AND status=REPAIRING" --format='value(name)' | wc -l)（>1 h 即告警 #11；>12 h 按 08 §8.5 升级）"
echo "非 RUNNING 的节点池：$(gcloud container node-pools list --cluster "$CLUSTER" --location "$ZONE" --project "$PROJECT" --filter='status!=RUNNING' --format='value(name,status)' | tr '\n' ' ')"

# ---- 5. 卡住的 Slice、手工 Slice（只读） -------------------------
# 状态在 .status.conditions[type=Ready].reason，不在 .status.state（实测恒为空）
# 阈值 60 min = 实测集群 activation-timeout；告警 #4/#6
kubectl get slices.accelerator.gke.io -o json 2>/dev/null | jq -r '
  .items[] | .metadata.name as $n | (.metadata.annotations["accelerator.gke.io/owner-workload-name"] // "-") as $o
  | (.status.conditions[]? | select(.type=="Ready")) as $c
  | select($c.reason != "ACTIVE")
  | [$n, $c.reason, (((now - ($c.lastTransitionTime|fromdateiso8601))/60)|floor), $o] | @tsv' > "$STATE_DIR/slices-notactive-$TS.tsv" || true
stuck=$(awk -F'\t' '$3>=60' "$STATE_DIR/slices-notactive-$TS.tsv" | wc -l)
echo "非 ACTIVE ≥60 min 的 Slice：$stuck（明细 slices-notactive-$TS.tsv：名字 状态 分钟 owner；告警 #4/#6）"
(( stuck == 0 )) || { awk -F'\t' '$3>=60{printf "  %s %s %s min\n",$1,$2,$3}' "$STATE_DIR/slices-notactive-$TS.tsv"; (( RC >= 1 )) || RC=1; }
# 手工 Slice = 没有 owner-workload-name 注解；Kueue/TAS 看不见它占的 cube；告警 #28
manual=$(kubectl get slices.accelerator.gke.io -o json 2>/dev/null | jq -r '[.items[] | select(.metadata.annotations["accelerator.gke.io/owner-workload-name"]==null) | .metadata.name] | join(" ")' || true)
echo "手工 Slice（无 owner 注解，Kueue 看不见）：${manual:-无}（经验教训第 33 条：用完即删，或提交时加 05-avoid-manual-slices.py）"

# ---- 6. pod 总数（GKE 上限 200k，经验教训第 53 条；告警 #22） ------------------------
pods=$(kubectl get pods -A --no-headers -o custom-columns=P:.status.phase 2>/dev/null | sort | uniq -c | awk '{printf "%s=%s ",$2,$1; t+=$1} END{printf "total=%d", t}' || true)
ptotal=${pods##*total=}; ptotal=${ptotal:-0}
echo "pod：$pods（>100k 告警；完成态多说明 JobSet 没设 ttlSecondsAfterFinished）"
(( ptotal <= 100000 )) || { echo "  !! pod 总数 > 100k（经验教训第 53 条：删完成态 pod，补 TTL）"; (( RC >= 1 )) || RC=1; }

# ---- 7. 过去 24 h Kueue 控制器日志里的三类事（Event 可能被 RBAC 拒，只能看日志）--
klog=$(kubectl logs -n kueue-system -l app.kubernetes.io/name=kueue --since=24h --tail=-1 2>/dev/null) \
  || { echo "  !! 读 Kueue 日志失败——下面三个计数的 0 不代表没有"; klog=""; }
echo "Kueue 24 h：EvictedDueTo* $(grep -c 'EvictedDueTo' <<<"$klog")  conflicting rank $(grep -c 'conflicting rank' <<<"$klog")  events is forbidden $(grep -c 'events is forbidden' <<<"$klog")（后两个非 0 → 经验教训第 42 条、经验教训第 54 条）"

# ---- 8. reservation 到期 ----------------------------------------------------
del=$(gcloud compute reservations describe "$RESERVATION" --zone "$ZONE" --project "$PROJECT" --format='value(deleteAtTime)')
if [[ -n "$del" ]]; then left=$(( ( $(date -d "$del" +%s) - $(date +%s) ) / 86400 ))
  echo "reservation 到期：$del（剩 $left 天）"; (( left > 30 )) || { echo "  !! ≤30 天（30/14/3 天各提醒一次）"; (( RC >= 1 )) || RC=1; }
fi
echo "报表：$RPT   明细：$STATE_DIR/cubes-$TS.tsv"
exit $RC
