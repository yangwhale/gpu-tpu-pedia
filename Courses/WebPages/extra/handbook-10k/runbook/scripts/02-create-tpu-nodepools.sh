#!/usr/bin/env bash
# =============================================================================
# 02-create-tpu-nodepools.sh —— §2.4：一 cube 一池，按 sub-block 批量建 16 节点 TPU 池（并发 + 重试 + 健康检查）
#
# 用法：
#   ./02-create-tpu-nodepools.sh --dry-run                 # 只生成计划和命令，不动云
#   ./02-create-tpu-nodepools.sh                           # 建所有 HEALTHY 且 healthyHostCount=16 的 sub-block
#   ./02-create-tpu-nodepools.sh --parallel 10 --only 0016,0017     # 只建指定 sub-block（4 位编号）
#   ./02-create-tpu-nodepools.sh --include-degraded        # 连 DEGRADED sub-block 也建（预期起不全，见 §2.5）
#   ./02-create-tpu-nodepools.sh --watch-only              # 不建，只盯状态 + 跑健康检查
#
# 单池命令的 flag：
#   - 目标状态（双网卡池）字段 → flag：
#       additionalNodeNetworkConfigs ×2      → --additional-node-network network=…,subnetwork=…（×2）
#       config.labels gke-networking-dra-driver=true → --node-labels（公开文档
#                                              https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-multislice 「GKE DRANET」）
#       maxPodsConstraint 32 / podIpv4CidrSize 26 → --max-pods-per-node=32
#       bootDisk hyperdisk-balanced 100       → --disk-type --disk-size
#       gcfsConfig.enabled                    → --enable-image-streaming
#       shieldedInstanceConfig.enableSecureBoot → --shielded-secure-boot
#       workloadMetadataConfig GKE_METADATA   → --workload-metadata=GKE_METADATA
#       kubeletConfig.topologyManager / THP   → --system-config-from-file=02-tpu-node-system-config.yaml
#       upgradeSettings SURGE maxUnavailable 16 → --max-surge-upgrade=0 --max-unavailable-upgrade=16（GA 指南 update 命令的取值）
#     flag 名核对：https://cloud.google.com/sdk/gcloud/reference/container/node-pools/create
#     taint google.com/tpu=present:NoSchedule 由 GKE 对 TPU 池自动加（实测集群池 config.taints 有、但 GA 指南命令没传）[推断]。
#   - 并发：GKE 不串行化节点池操作（10 并发删除零 409，见手册）；
#     CTK -parallelism=30 建完上百个池约 20 min。默认 10。
#   - 超时：池一直 PROVISIONING 时 GKE 不区分「在开机」和「卡死」，给 45 min 宽限；
#     GCE 自己在 35 min 时把起不全的池置 ERROR（实测集群 <节点池>：「after 35m10s … Expected 16, running 15」）。
# [需验证] 这个循环脚本本身是本手册写的。
#
# 幂等：已存在的池一律跳过（不论状态）；ERROR 的池不会自动删——按 §2.5 对模式处理。重跑只补建缺的池。
# 重试：只在「create 调用失败且池没被建出来」时重试（配额/限流/并发操作冲突），指数退避 30s→60s→120s→240s，最多 5 次。
#       池已建出但 ERROR（缺货/15 of 16）不属于重试范围——重建池可能回不来。
# 回滚：./02-delete-tpu-nodepools.sh（带确认；容量事故期间不要删健康的池）
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
rb_env_check

PARALLEL=10; DRY=0; INCLUDE_DEGRADED=0; ONLY=""; WATCH_ONLY=0; TIMEOUT_MIN=45
while (( $# )); do case "$1" in
  --parallel) PARALLEL=$2; shift;; --dry-run) DRY=1;; --include-degraded) INCLUDE_DEGRADED=1;;
  --only) ONLY=$2; shift;; --watch-only) WATCH_ONLY=1;; --timeout-min) TIMEOUT_MIN=$2; shift;;
  *) echo "未知参数 $1" >&2; exit 2;; esac; shift; done

LOG="$STATE_DIR/nodepools-$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$LOG"
PLAN="$LOG/plan.tsv"; echo "日志目录：$LOG"

# ---- 1. 前置检查 ---------------------------------------------------------------
gcloud compute resource-policies describe "$POLICY_NAME" --region="$REGION" --project="$PROJECT" \
  --format="value(status)" | grep -qx READY || { echo "workload policy $POLICY_NAME 不是 READY，先做 §2.3" >&2; exit 1; }
for n in "$NET_DCN1:$SUBNET_DCN1" "$NET_DCN2:$SUBNET_DCN2"; do
  gcloud compute networks subnets describe "${n#*:}" --region="$REGION" --project="$PROJECT" >/dev/null \
    || { echo "附加子网 ${n#*:} 不存在，先做 §2.1（多网卡建池后不可改）" >&2; exit 1; }
done

# ---- 2. 生成计划：sub-block → 池名 ----------------------------------------------
gcloud compute reservations sub-blocks list "$RESERVATION" --block-name="$BLOCK" --zone="$ZONE" --project="$PROJECT" \
  --format="csv[no-heading](name,healthInfo.healthStatus,healthInfo.healthyHostCount,inUseHostCount)" > "$LOG/subblocks.csv"
gcloud container node-pools list --cluster="$CLUSTER" --location="$ZONE" --project="$PROJECT" \
  --format="value(name)" > "$LOG/existing-pools.txt"

: > "$PLAN"
while IFS=, read -r sb health healthy inuse; do
  num="${sb##*-subblock-}"                               # 0056
  pool="${POOL_PREFIX}$(printf '%03d' "$((10#$num))")"   # <节点池>（按 sub-block 序号编号）
  [[ -n "$ONLY" && ",$ONLY," != *",$num,"* ]] && continue
  if grep -qx "$pool" "$LOG/existing-pools.txt"; then echo -e "$pool\t$sb\tSKIP_EXISTS" >> "$PLAN"; continue; fi
  if [[ "$health" != HEALTHY || "$healthy" != 16 ]] && (( ! INCLUDE_DEGRADED )); then
    echo -e "$pool\t$sb\tSKIP_${health}_${healthy}of16" >> "$PLAN"; continue; fi
  if [[ "${inuse:-0}" != 0 ]]; then echo -e "$pool\t$sb\tSKIP_INUSE_${inuse}" >> "$PLAN"; continue; fi
  echo -e "$pool\t$sb\tCREATE" >> "$PLAN"
done < "$LOG/subblocks.csv"
echo "== 计划"; cut -f3 "$PLAN" | sed 's/_.*//' | sort | uniq -c
grep -v CREATE "$PLAN" | grep -v SKIP_EXISTS || true

# ---- 3. 并发建池 -----------------------------------------------------------------
create_one() {
  local pool=$1 sb=$2 attempt=1 delay=30
  local path="projects/${PROJECT}/reservations/${RESERVATION}/reservationBlocks/${BLOCK}/reservationSubBlocks/${sb}"
  local cmd=(gcloud container node-pools create "$pool"
    --project="$PROJECT" --cluster="$CLUSTER" --location="$ZONE" --node-locations="$ZONE"
    --machine-type="$MACHINE_TYPE" --num-nodes="$NODES_PER_POOL"
    --placement-policy="$POLICY_NAME"
    --reservation-affinity=specific --reservation="$path"
    --node-version="$NODE_VERSION" --image-type=COS_CONTAINERD
    --disk-type="$TPU_DISK_TYPE" --disk-size="$TPU_DISK_SIZE_GB"
    --max-pods-per-node="$TPU_MAX_PODS"
    --additional-node-network="network=${NET_DCN1},subnetwork=${SUBNET_DCN1}"
    --additional-node-network="network=${NET_DCN2},subnetwork=${SUBNET_DCN2}"
    --node-labels=cloud.google.com/gke-networking-dra-driver=true
    --system-config-from-file="$HERE/02-tpu-node-system-config.yaml"
    --enable-image-streaming --shielded-secure-boot --workload-metadata=GKE_METADATA
    --max-surge-upgrade=0 --max-unavailable-upgrade="$NODES_PER_POOL"
    --async --quiet)
  if (( DRY )); then printf '%q ' "${cmd[@]}" > "$LOG/$pool.cmd"; echo >> "$LOG/$pool.cmd"; return 0; fi
  while (( attempt <= 5 )); do
    if "${cmd[@]}" > "$LOG/$pool.out" 2>&1; then echo "[submit] $pool ($sb) attempt $attempt"; return 0; fi
    if grep -qi "already exists" "$LOG/$pool.out"; then echo "[exists] $pool"; return 0; fi
    # 调用失败但池其实建出来了（超时/断线）→ 不重试
    if gcloud container node-pools describe "$pool" --cluster="$CLUSTER" --location="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
      echo "[exists-after-error] $pool"; return 0; fi
    echo "[retry] $pool attempt $attempt 失败：$(tail -1 "$LOG/$pool.out")；${delay}s 后重试"
    sleep "$delay"; attempt=$((attempt+1)); delay=$((delay*2))
  done
  echo "[FAIL] $pool 重试 5 次仍失败，见 $LOG/$pool.out"; return 1
}
export -f create_one; export LOG DRY HERE

if (( ! WATCH_ONLY )); then
  awk -F'\t' '$3=="CREATE"{print $1, $2}' "$PLAN" \
    | xargs -r -P "$PARALLEL" -n 2 bash -c 'create_one "$0" "$1"' || echo "有池提交失败，见上面 [FAIL] 行"
  (( DRY )) && { echo "dry-run：命令写在 $LOG/*.cmd"; head -c 1500 "$(ls "$LOG"/*.cmd | head -1)" 2>/dev/null; exit 0; }
fi

# ---- 4. 盯状态直到全部落定或超时 --------------------------------------------------
deadline=$(( $(date +%s) + TIMEOUT_MIN*60 ))
while :; do
  gcloud container node-pools list --cluster="$CLUSTER" --location="$ZONE" --project="$PROJECT" \
    --filter="name~^${POOL_PREFIX}" --format="value(name,status)" > "$LOG/status.tsv"
  echo "$(date -u +%H:%M:%SZ) $(awk '{print $2}' "$LOG/status.tsv" | sort | uniq -c | tr '\n' ' ')"
  grep -qE "PROVISIONING|RECONCILING" "$LOG/status.tsv" || break
  (( $(date +%s) < deadline )) || { echo "超过 ${TIMEOUT_MIN} min 仍有池 PROVISIONING——按 §2.5 最后一行处理"; break; }
  sleep 60
done

# ---- 5. 健康检查 ------------------------------------------------------------------
"$HERE/02-check-nodepools.sh" | tee "$LOG/health.txt"
