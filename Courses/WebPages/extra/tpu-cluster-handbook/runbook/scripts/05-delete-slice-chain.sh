#!/usr/bin/env bash
# =============================================================================
# 05-delete-slice-chain.sh —— §5.5：按「不留孤儿」的顺序删一个 Slice，卡住时按判据走 GCE 侧清理
#
# 默认 dry-run：只跑只读查询，改动类命令只打印。真删要 --execute；动 GCE / finalizer 的两步还要
# 额外开关 + --case <Google case 号>（§5.5 规定第 5–6 步只在 case 开着时做）。
#
# 出处：
#   - 强制清理命令原文（MMIG 成员查询 / PATCH multiMig null / 去 finalizer / 删 MMIG 与 workload policy）：按 Google 的排查步骤
#   - partition → MIG：节点注解 node.gke.io/created-by-mig
#   - finalizer 名 accelerator.gke.io/slice-finalizer
#
# ⚠ sub-slice ACTIVATE/DEACTIVATE 的动作是 configureAcceleratorTopologies（不是 getAvailableAcceleratorTopologies）。
#   本脚本不调这两个 API——ACTIVATE/DEACTIVATE 是 GKE 控制器的事，运维永远不要手发。
#
# 用法：
#   ./05-delete-slice-chain.sh <slice>                         # dry-run：看现状、打印将执行的命令
#   ./05-delete-slice-chain.sh <slice> --execute               # 删 Slice CR 并等它消失（第 3 步），然后核对（第 7 步）
#   ./05-delete-slice-chain.sh <slice> --execute --case 12345 --force-detach       # 卡住且恰好 1 个 MIG 挂着：PATCH 一次
#   ./05-delete-slice-chain.sh <slice> --execute --case 12345 --remove-finalizer   # 卡住且 0 个 MIG 挂着：去 finalizer+删 MMIG/WP
#   ./05-delete-slice-chain.sh --scan-orphans                  # 只读：找悬空 multiMig 引用 / 没有 Slice 的 MMIG 与 WP
# 选项：--timeout <秒>（默认 900，§5.1 DEACTIVATING/INCOMPLETE >10 min 算异常）；--allow-kueue（Kueue 建的 Slice 也照删）
#
# 幂等：Slice 已不存在时直接进入核对；PATCH / 去 finalizer / 删 MMIG 前都会重新读状态，已完成的步骤跳过。
# 回滚：没有。删 Slice 不可逆（Kueue 路径会自己重建；手动路径用 05-slice.yaml 重新 apply）。
# 前置：kubectl（集群管理员）、gcloud（含 alpha/beta 组件）、jq、curl；compute.instanceGroupManagers.update、
#       compute.resourcePolicies.delete 权限；第 1–2 步已做完：整池回收＝CQ 配额置 0 + 删 JobSet；部分回收＝先 05-isolate-cube.sh --execute 隔离 + 删 JobSet，不动配额（§5.5 WARN）。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT REGION ZONE CLUSTER

SLICE=""; EXEC=0; CASE=""; FORCE_DETACH=0; RM_FIN=0; SCAN=0; TIMEOUT=900; ALLOW_KUEUE=0
while (($#)); do case "$1" in
  --execute) EXEC=1;; --case) CASE="$2"; shift;; --force-detach) FORCE_DETACH=1;;
  --remove-finalizer) RM_FIN=1;; --scan-orphans) SCAN=1;; --timeout) TIMEOUT="$2"; shift;;
  --allow-kueue) ALLOW_KUEUE=1;; -*) echo "未知选项 $1" >&2; exit 64;; *) SLICE="$1";; esac; shift; done

LOG="$STATE_DIR/05-delete-slice-chain-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
run() { echo "+ $*"; if [[ $EXEC == 1 ]]; then "$@"; else echo "  (dry-run，未执行)"; fi; }
API=https://compute.googleapis.com/compute/alpha/projects/$PROJECT
gcurl() { curl -sS -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" "$@"; }

mig_state() {  # $1 = MIG 名 → "multiMig|isStable"（用 | 分隔：multiMig 为空时 tab 会被 read 吞掉）
  gcurl "$API/zones/$ZONE/instanceGroupManagers/$1" | jq -r '"\(.multiMig // "")|\(.status.isStable)"'
}
cube_migs() {  # 所有带 partition 标签节点的 MIG（去重）；可选 $@ = 只要这些 partition id
  kubectl get nodes -l cloud.google.com/gke-tpu-partition-4x4x4-id -o json | jq -r --args '
    ($ARGS.positional) as $want | .items[]
    | select(($want|length)==0 or (.metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-id"] as $p | $want | index($p)))
    | .metadata.annotations["node.gke.io/created-by-mig"] // empty | split("/") | last' "$@" | sort -u
}
cluster_hash() {  # 优先 CLUSTER_HASH；否则从现存 Slice 的 per-slice WP 名（<slice>-<hash8>）推出
  [[ -n "${CLUSTER_HASH:-}" ]] && { echo "$CLUSTER_HASH"; return; }
  local s; s=$(kubectl get slices.accelerator.gke.io -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | head -20)
  for n in $s; do
    local wp; wp=$(gcloud compute resource-policies list --project="$PROJECT" --regions="$REGION" \
      --filter="name~^${n}-[0-9a-f]{8}$" --format='value(name)' 2>/dev/null | head -1)
    [[ -n "$wp" ]] && { echo "${wp##*-}"; return; }
  done
  return 1
}

# ---------------------------------------------------------------- 孤儿扫描（只读）
if [[ $SCAN == 1 ]]; then
  say "扫描 cube MIG 的 multiMig 引用（每个 MIG 一次 GET；144 个约 1–2 min，注意 GCE 读配额 经验教训第 52 条）"
  live=$(kubectl get slices.accelerator.gke.io -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n')
  printf 'MIG\tMULTIMIG\tIS_STABLE\tVERDICT\n'
  for m in $(cube_migs); do
    IFS='|' read -r mm st < <(mig_state "$m") || true
    [[ -z "$mm" ]] && continue
    name=${mm##*/}; sl=${name%-*}; verdict=ok
    code=$(gcurl -o /dev/null -w '%{http_code}' "$API/regions/$REGION/multiMigs/$name")
    if [[ $code == 404 ]]; then verdict="DANGLING(MMIG 404) → 找 Google 开 MIG case"
    elif ! grep -qx "$sl" <<<"$live"; then verdict="ORPHAN(无对应 Slice CR)"; fi
    printf '%s\t%s\t%s\t%s\n' "$m" "$name" "$st" "$verdict"
  done
  if h=$(cluster_hash); then
    say "per-slice workload policy（名字以 -$h 结尾）里没有对应 Slice CR 的："
    gcloud compute resource-policies list --project="$PROJECT" --regions="$REGION" \
      --filter="name~-${h}$" --format='value(name)' | while read -r wp; do
        grep -qx "${wp%-*}" <<<"$live" || echo "  ORPHAN-WP $wp   → 删：gcloud compute resource-policies delete $wp --region=$REGION --project=$PROJECT"
      done
  else say "推不出集群 hash（集群里没有 Slice），设 CLUSTER_HASH=<8位> 再跑"; fi
  exit 0
fi

[[ -n "$SLICE" ]] || { echo "用法见文件头" >&2; exit 64; }
if [[ ( $FORCE_DETACH == 1 || $RM_FIN == 1 ) && -z "$CASE" ]]; then
  echo "--force-detach / --remove-finalizer 必须带 --case <Google case 号>（§5.5 第 5–6 步）" >&2; exit 64; fi

# ---------------------------------------------------------------- 0. 读现状
sj=$(kubectl get slices.accelerator.gke.io "$SLICE" -o json 2>/dev/null || true)
if [[ -n "$sj" ]]; then
  mapfile -t PIDS < <(jq -r '.spec.partitionIds[]' <<<"$sj")
  owner=$(jq -r '.metadata.annotations["accelerator.gke.io/owner-workload-name"] // ""' <<<"$sj")
  say "Slice $SLICE：$(jq -r '[.spec.topology, (.status.conditions[]?|select(.type=="Ready")|.reason), (.metadata.finalizers//[]|join(",")), (.metadata.deletionTimestamp//"-")]|join("  ")' <<<"$sj")"
  if [[ -n "$owner" && $ALLOW_KUEUE == 0 ]]; then
    echo "这个 Slice 是 Kueue 建的（owner-workload-name=$owner）。正路是删 JobSet/Workload，让 Kueue 删 Slice（§5.2 第 4 条）。"
    echo "确认要直接删：加 --allow-kueue。"; exit 2; fi
  printf '%s\n' "${PIDS[@]}" > "$STATE_DIR/slice-$SLICE-partitions.txt"
else
  say "Slice $SLICE 不存在，直接进入核对"
  [[ -f "$STATE_DIR/slice-$SLICE-partitions.txt" ]] && mapfile -t PIDS < "$STATE_DIR/slice-$SLICE-partitions.txt" || PIDS=()
fi
MIGS=(); (( ${#PIDS[@]} )) && mapfile -t MIGS < <(cube_migs "${PIDS[@]}")
say "涉及 MIG（${#MIGS[@]} 个）：${MIGS[*]:-无（partition 记录丢了：用 --scan-orphans）}"
HASH=$(cluster_hash || true); MMIG="${SLICE}-${HASH}"
if [[ -n "$HASH" ]]; then say "MMIG / per-slice WP 名推定：$MMIG"; else say "MMIG / per-slice WP 名：推不出集群 hash（设 CLUSTER_HASH=<8位>）"; fi

# ---------------------------------------------------------------- 3. 删 Slice 并等待
if [[ -n "$sj" && "$(jq -r '.metadata.deletionTimestamp // ""' <<<"$sj")" == "" ]]; then
  run kubectl delete slices.accelerator.gke.io "$SLICE" --wait=false
fi
if [[ -n "$sj" && $EXEC == 1 ]]; then
  t0=$(date +%s)
  while kubectl get slices.accelerator.gke.io "$SLICE" >/dev/null 2>&1; do
    (( $(date +%s) - t0 > TIMEOUT )) && break
    say "  等待：$(kubectl get slices.accelerator.gke.io "$SLICE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].reason}' 2>/dev/null)"
    sleep 30
  done
fi

# ---------------------------------------------------------------- 4–6. 卡住：看 GCE 侧，按判据分支
if kubectl get slices.accelerator.gke.io "$SLICE" >/dev/null 2>&1 && [[ $EXEC == 1 || -n "$sj" ]]; then
  say "Slice 仍在（超过 ${TIMEOUT}s 或 dry-run）。finalizers=$(kubectl get slices.accelerator.gke.io "$SLICE" -o jsonpath='{.metadata.finalizers}')"
  echo "+ gcloud alpha compute multi-migs describe $MMIG --region=$REGION --project=$PROJECT   # 看 appliedAcceleratorTopologies / membersCount"
  gcloud alpha compute multi-migs describe "$MMIG" --region="$REGION" --project="$PROJECT" 2>&1 | sed 's/^/    /' || true
  members=$(gcurl "$API/regions/$REGION/multiMigs/$MMIG/multiMigMembers" | jq -r '.items[]?.name // empty' 2>/dev/null || true)
  attached=(); for m in "${MIGS[@]}"; do
    IFS='|' read -r mm st < <(mig_state "$m") || true; [[ -n "$mm" ]] && attached+=("$m"); echo "    MIG $m multiMig=${mm:-<空>} isStable=$st"
  done
  say "MMIG 成员（API）：${members:-<空>}；仍挂着的 MIG：${#attached[@]} 个"
  if [[ $EXEC == 0 ]]; then say "dry-run：以上是现状。真删用 --execute；超时仍卡住时本脚本才按判据分支。"; exit 0; fi
  case ${#attached[@]} in
    0)
      if [[ $RM_FIN == 1 ]]; then
        run kubectl patch slices.accelerator.gke.io "$SLICE" --type merge -p '{"metadata":{"finalizers":[]}}'
        run gcloud alpha compute multi-migs delete "$MMIG" --region="$REGION" --project="$PROJECT" --quiet
        run gcloud compute resource-policies delete "$MMIG" --region="$REGION" --project="$PROJECT" --quiet
      else
        echo "判据：0 个 MIG 挂着 → 可以去 finalizer（case 开着时）。重跑加 --case <号> --remove-finalizer，或手动："
        echo "  kubectl patch slice $SLICE --type merge -p '{\"metadata\":{\"finalizers\":[]}}'"
        echo "  gcloud alpha compute multi-migs delete $MMIG --region=$REGION --project=$PROJECT"
        echo "  gcloud compute resource-policies delete $MMIG --region=$REGION --project=$PROJECT"; exit 2
      fi;;
    1)
      pool=$(kubectl get nodes -o json | jq -r --arg m "${attached[0]}" '.items[]|select((.metadata.annotations["node.gke.io/created-by-mig"]//"")|endswith("/"+$m))|.metadata.labels["cloud.google.com/gke-nodepool"]' | head -1)
      echo "判据：1 个 MIG 挂着 → cordon 该池 + 升级 Google 支持强制 detach。"
      run kubectl cordon -l "cloud.google.com/gke-nodepool=$pool"
      if [[ $FORCE_DETACH == 1 ]]; then   # 只 PATCH 一次：卡住或 412 就交给 MIG case，别反复重试
        echo "+ PATCH $API/zones/$ZONE/instanceGroupManagers/${attached[0]} {\"multiMig\": null}   # case $CASE"
        [[ $EXEC == 1 ]] && gcurl -X PATCH "$API/zones/$ZONE/instanceGroupManagers/${attached[0]}" --data '{"multiMig": null}' | jq -r '.name, .status, (.error // empty)'
        echo "  等 1–2 min 后重跑本脚本（不带 --force-detach）核对 multiMig 已空；然后再用 --remove-finalizer。"
      else
        echo "  等价 gcloud：gcloud beta compute instance-groups managed update ${attached[0]} --zone=$ZONE --project=$PROJECT --clear-multi-mig"
      fi; exit 2;;
    *)
      echo "判据：${#attached[@]} 个 MIG 挂着 → 直接升级 Google 支持（带上面的 describe 输出和本日志 $LOG），不要自己逐个 PATCH。"; exit 3;;
  esac
fi

# ---------------------------------------------------------------- 7. 核对
say "核对（原流程第 7 步）："
bad=0
for m in "${MIGS[@]}"; do
  IFS='|' read -r mm st < <(mig_state "$m") || true
  printf '    %-60s multiMig=%-20s isStable=%s\n' "$m" "${mm:-<空>}" "$st"
  [[ -z "$mm" && "$st" == true ]] || bad=1
done
left=$(kubectl get nodes -l "cloud.google.com/gke-tpu-slice=$SLICE" -o name | wc -l)
echo "    节点上残留 gke-tpu-slice=$SLICE 标签：$left 个（期望 0）"; (( left == 0 )) || bad=1
if [[ -n "$HASH" ]]; then
  code=$(gcurl -o /dev/null -w '%{http_code}' "$API/regions/$REGION/multiMigs/$MMIG")
  wp=$(gcloud compute resource-policies describe "$MMIG" --region="$REGION" --project="$PROJECT" --format='value(name)' 2>/dev/null || true)
  echo "    MMIG $MMIG：HTTP $code（期望 404）；WP：${wp:-不存在}（期望不存在）"
  [[ $code == 404 && -z "$wp" ]] || bad=1
fi
(( bad == 0 )) && say "OK：可以动节点池（§5.5 第 8 步）" || { say "未干净：先别删池，见 §5.6 / 经验教训第 40 条"; exit 1; }
