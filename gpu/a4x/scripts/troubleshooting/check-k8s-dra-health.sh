#!/bin/bash
# check-k8s-dra-health.sh — 彻底检查 forrest GB200 k8s + DRA driver 8 类组件健康
#
# 检查项 (FAIL 时返回 non-zero, 末尾汇总):
#  1. control-plane Ready (master node)
#  2. Worker 节点 Ready (36 worker 全部)
#  3. Worker 节点 GPU=4 allocatable (无 Xid 故障)
#  4. NVIDIA device-plugin DS 全 Running (36 pod)
#  5. NVIDIA DRA driver controller Deployment 健康 (1 pod Running)
#  6. NVIDIA DRA kubelet-plugin DS 全 Running (36 pod)
#  7. **CD ResourceSlice publish 齐全** (每 worker 必须有 compute-domain.nvidia.com 一条 RS)
#  8. DRANET DS 全 Running (worker 36 pod, 排除 master)
#  9. GFD (gpu-feature-discovery) DS 全 Running + 全 worker 有 nvidia.com/gpu.clique label
# 10. 检查残留 stuck ResourceClaim / ComputeDomain (无活动 workload 时应该 0)
# 13. **Workload pod 调度健康** (stale orphan default ns pod + Pending pod 跟 reason)
# 11. 检查 worker 最近 24h dmesg Xid 137/145/94 (NVLink fabric / device-plugin 触发)
#
# 用法:
#   bash check-k8s-dra-health.sh                # 全部检查, summary
#   bash check-k8s-dra-health.sh --verbose      # 详细 list 每 node 状态
#   bash check-k8s-dra-health.sh --fix-rs       # 第 7 项 fail 时自动 kubectl delete 缺 RS 的 plugin pod
#   bash check-k8s-dra-health.sh --fix-race     # 第 12 项 fail 时 (controller race 致 DS DESIRED=0) 自动 Stage 1 label + Stage 2 kubelet restart
#
# 退出码: 0 = 全 PASS, 1 = 任意 FAIL
#
# 跑位置: 任意有 gx (gx k8n 走 master) 的本机

set -uo pipefail

VERBOSE=0
FIX_RS=0
FIX_RACE=0
FAILED=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose) VERBOSE=1; shift ;;
    --fix-rs) FIX_RS=1; shift ;;
    --fix-race) FIX_RACE=1; shift ;;
    --help|-h) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg $1"; exit 1 ;;
  esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }
ok()  { echo "  ✓ $*"; }
fail() { echo "  ✗ $*"; FAILED+=("$*"); }
warn() { echo "  ⚠ $*"; }

WORKER_NODES_RAW=$(gx k8n "kubectl get nodes -l 'feature.node.kubernetes.io/pci-10de.present=true' -o jsonpath='{.items[*].metadata.name}'" 2>&1)
read -ra WORKER_NODES <<< "$WORKER_NODES_RAW"
EXPECTED_WORKER_COUNT=${#WORKER_NODES[@]}

if [ "$EXPECTED_WORKER_COUNT" = "0" ]; then
  fail "no worker nodes detected (label feature.node.kubernetes.io/pci-10de.present=true)"
  exit 1
fi
log "detected $EXPECTED_WORKER_COUNT worker nodes"

# ============= Check 1: control-plane Ready =============
log ""
log "=== Check 1: control-plane Ready ==="
CP_STATUS=$(gx k8n "kubectl get nodes -l 'node-role.kubernetes.io/control-plane' --no-headers 2>&1 | awk '{print \$2}'")
if [ "$CP_STATUS" = "Ready" ]; then ok "control-plane Ready"; else fail "control-plane not Ready: $CP_STATUS"; fi

# ============= Check 2: Worker Ready =============
log ""
log "=== Check 2: $EXPECTED_WORKER_COUNT worker Ready ==="
NOTREADY=$(gx k8n "kubectl get nodes -l 'feature.node.kubernetes.io/pci-10de.present=true' --no-headers 2>&1 | awk '\$2!=\"Ready\" {print \$1}'")
if [ -z "$NOTREADY" ]; then
  ok "all $EXPECTED_WORKER_COUNT worker Ready"
else
  fail "worker NotReady: $(echo $NOTREADY | xargs)"
fi

# ============= Check 3: Worker GPU=4 =============
log ""
log "=== Check 3: $EXPECTED_WORKER_COUNT worker GPU=4 ==="
BAD_GPU=$(gx k8n "kubectl get nodes -l 'feature.node.kubernetes.io/pci-10de.present=true' -o json" 2>&1 | python3 -c "
import json, sys
d = json.load(sys.stdin)
for n in d['items']:
    name = n['metadata']['name']
    gpu = int(n.get('status', {}).get('allocatable', {}).get('nvidia.com/gpu', '0'))
    if gpu != 4:
        print(f'{name}:{gpu}')
")
if [ -z "$BAD_GPU" ]; then
  ok "all $EXPECTED_WORKER_COUNT worker GPU=4"
else
  fail "worker GPU<4: $(echo $BAD_GPU | xargs)"
  warn "  → 走 scripts/troubleshooting/xid-fault-auto-recover.sh --apply 修 (Xid 137/145 sticky fault)"
fi

# ============= Check 4: device-plugin =============
log ""
log "=== Check 4: nvidia-device-plugin DS Running ($EXPECTED_WORKER_COUNT pod) ==="
DP_RUNNING=$(gx k8n "kubectl get pods -n kube-system --no-headers 2>&1 | grep nvidia-device-plugin | grep -c Running")
DP_TOTAL=$(gx k8n "kubectl get pods -n kube-system --no-headers 2>&1 | grep -c nvidia-device-plugin")
if [ "$DP_RUNNING" = "$EXPECTED_WORKER_COUNT" ] && [ "$DP_TOTAL" = "$EXPECTED_WORKER_COUNT" ]; then
  ok "device-plugin $DP_RUNNING/$EXPECTED_WORKER_COUNT Running"
else
  fail "device-plugin $DP_RUNNING/$DP_TOTAL (expected $EXPECTED_WORKER_COUNT all Running)"
fi

# ============= Check 5: DRA controller =============
log ""
log "=== Check 5: NVIDIA DRA driver controller ==="
CTRL_STATE=$(gx k8n "kubectl get pods -n nvidia-dra-driver-gpu --no-headers 2>&1 | grep controller | awk '{print \$3}'")
if [ "$CTRL_STATE" = "Running" ]; then
  ok "DRA controller Running"
else
  fail "DRA controller not Running: $CTRL_STATE"
  warn "  → kubectl rollout restart deploy/nvidia-dra-driver-gpu-controller -n nvidia-dra-driver-gpu"
fi

# ============= Check 6: DRA kubelet-plugin DS =============
log ""
log "=== Check 6: DRA kubelet-plugin DS ($EXPECTED_WORKER_COUNT pod) ==="
KP_RUNNING=$(gx k8n "kubectl get pods -n nvidia-dra-driver-gpu --no-headers 2>&1 | grep kubelet-plugin | grep -c '2/2.*Running'")
if [ "$KP_RUNNING" = "$EXPECTED_WORKER_COUNT" ]; then
  ok "kubelet-plugin $KP_RUNNING/$EXPECTED_WORKER_COUNT (2/2 Running)"
else
  fail "kubelet-plugin $KP_RUNNING/$EXPECTED_WORKER_COUNT (2/2 Running)"
  warn "  → kubectl rollout restart daemonset/nvidia-dra-driver-gpu-kubelet-plugin -n nvidia-dra-driver-gpu"
fi

# ============= Check 7: CD ResourceSlice publish (KEY!) =============
log ""
log "=== Check 7: CD ResourceSlice publish 齐全 (每 worker 应有 1 条 compute-domain.nvidia.com) ==="
RS_NODES_RAW=$(gx k8n "kubectl get resourceslice --no-headers 2>&1 | grep compute-domain.nvidia.com" | grep -oE 'forrest-gb200-[0-9]+' | sort -u)
RS_COUNT=$(echo "$RS_NODES_RAW" | grep -c .)

if [ "$RS_COUNT" = "$EXPECTED_WORKER_COUNT" ]; then
  ok "CD ResourceSlice published on all $RS_COUNT/$EXPECTED_WORKER_COUNT worker"
else
  fail "CD ResourceSlice published on $RS_COUNT/$EXPECTED_WORKER_COUNT worker"
  # find missing nodes
  ALL_WORKERS=$(echo "${WORKER_NODES[@]}" | tr ' ' '\n' | sort -u)
  MISSING=$(comm -23 <(echo "$ALL_WORKERS") <(echo "$RS_NODES_RAW"))
  warn "  missing CD slice on: $(echo $MISSING | xargs)"
  warn "  → 修法 (auto with --fix-rs): kubectl delete pod plugin-pod-on-missing-node 让 DS 重 spawn"
  warn "  → memory: dra-plugin-resourceslice-missing-fix.md"

  if [ "$FIX_RS" = "1" ]; then
    log "  --fix-rs enabled, deleting stale plugin pods..."
    for n in $MISSING; do
      POD=$(gx k8n "kubectl get pods -n nvidia-dra-driver-gpu -o wide --no-headers 2>&1 | grep kubelet-plugin | grep $n | awk '{print \$1}'")
      if [ -n "$POD" ]; then
        gx k8n "kubectl delete pod -n nvidia-dra-driver-gpu $POD --grace-period=10" 2>&1 | tail -1
      fi
    done
    log "  wait 30s for DS respawn + RS publish..."
    sleep 30
    RS_RECHECK=$(gx k8n "kubectl get resourceslice --no-headers 2>&1 | grep compute-domain.nvidia.com" | grep -oE 'forrest-gb200-[0-9]+' | sort -u | wc -l)
    if [ "$RS_RECHECK" = "$EXPECTED_WORKER_COUNT" ]; then
      ok "  fix succeeded: now $RS_RECHECK/$EXPECTED_WORKER_COUNT publish"
    else
      warn "  fix partial: $RS_RECHECK/$EXPECTED_WORKER_COUNT (manual check)"
    fi
  fi
fi

# ============= Check 8: DRANET DS (worker only) =============
log ""
log "=== Check 8: DRANET DS Running on $EXPECTED_WORKER_COUNT worker (master pod 如果 crashloop 是已知残留) ==="
# 只数 worker 上 DRANET pod
DN_WORKER_RUNNING=$(gx k8n "kubectl get pods -n dranet-system -o wide --no-headers 2>&1 | grep dranet | grep gb200 | grep -c Running")
DN_MASTER_BAD=$(gx k8n "kubectl get pods -n dranet-system -o wide --no-headers 2>&1 | grep dranet | grep -v gb200 | awk '\$3!=\"Running\" {print \$1\":\"\$3}'")
if [ "$DN_WORKER_RUNNING" = "$EXPECTED_WORKER_COUNT" ]; then
  ok "DRANET $DN_WORKER_RUNNING/$EXPECTED_WORKER_COUNT worker Running"
  if [ -n "$DN_MASTER_BAD" ]; then warn "  master 残留 (无害): $DN_MASTER_BAD"; fi
else
  fail "DRANET $DN_WORKER_RUNNING/$EXPECTED_WORKER_COUNT worker Running"
fi

# ============= Check 9: clique label on worker (functional check) =============
log ""
log "=== Check 9: nvidia.com/gpu.clique label on $EXPECTED_WORKER_COUNT worker (functional check) ==="
WORKER_WITH_CLIQUE=$(gx k8n "kubectl get nodes -l 'feature.node.kubernetes.io/pci-10de.present=true' -L nvidia.com/gpu.clique --no-headers 2>&1 | awk '\$NF != \"\" && \$NF != \"<none>\" {print \$1}' | wc -l")
GFD_RUNNING=$(gx k8n "kubectl get pods -n kube-system --no-headers 2>&1 | grep -c '^gpu-feature-discovery\\|^nvidia-gpu-feature-discovery'")
if [ "$WORKER_WITH_CLIQUE" = "$EXPECTED_WORKER_COUNT" ]; then
  ok "clique label on $WORKER_WITH_CLIQUE/$EXPECTED_WORKER_COUNT worker (GFD pod 数 $GFD_RUNNING, 仅供参考)"
else
  fail "clique label on $WORKER_WITH_CLIQUE/$EXPECTED_WORKER_COUNT worker"
  warn "  → 升级或重装 gpu-feature-discovery chart (memory: gfd_official_clique_label.md)"
fi

# ============= Check 10: residual stuck RC/CD =============
log ""
log "=== Check 10: 残留 ResourceClaim / ComputeDomain ==="
RC_TOTAL=$(gx k8n "kubectl get resourceclaim -A --no-headers 2>&1" | grep -cE 'compute-domain-channel|rdma-nics')
CD_TOTAL=$(gx k8n "kubectl get computedomain -A --no-headers 2>&1" | grep -v 'No resources' | grep -c .)
RC_PENDING=$(gx k8n "kubectl get resourceclaim -A --no-headers 2>&1" | grep -c ' pending ')
# stale RC = state 含 "deleted,allocated" (delete-protection finalizer 卡住的 RC)
RC_STALE=$(gx k8n "kubectl get resourceclaim -A --no-headers 2>&1" | grep -c 'deleted,allocated')

if [ "$RC_TOTAL" = "0" ] && [ "$CD_TOTAL" = "0" ]; then
  ok "no residual: 0 RC, 0 CD"
elif [ "$RC_PENDING" -gt 0 ] 2>/dev/null; then
  fail "$RC_PENDING pending RC (workload stuck?), $RC_TOTAL total RC, $CD_TOTAL CD"
  warn "  → 如果 ResourceSlice 缺 publish, 跑本 script --fix-rs 或 memory: dra-plugin-resourceslice-missing-fix.md"
elif [ "$RC_STALE" -gt 0 ] 2>/dev/null; then
  fail "$RC_STALE stale RC (deleted,allocated finalizer 卡), $RC_TOTAL total RC, $CD_TOTAL CD"
  warn "  → 之前 workload yaml delete 后 RC 没 cascade GC, 强清: gx k8n \"kubectl get resourceclaim --no-headers | awk '{print \\\$1}' | xargs -I {} kubectl patch resourceclaim {} -p '{\\\"metadata\\\":{\\\"finalizers\\\":null}}' --type=merge\""
elif [ "$CD_TOTAL" = "0" ] && [ "$RC_TOTAL" -gt 0 ] 2>/dev/null; then
  fail "$RC_TOTAL RC 残留但 0 CD (workload yaml delete 不完整 — active workload 必有 CD), 强清同上"
else
  warn "active workload (OK if test running): $RC_TOTAL RC, $CD_TOTAL CD"
fi

# ============= Check 12: active CD daemon DS scale 健康度 =============
# 撞过的 stuck pattern: client pod schedule + RC allocated, 但 controller race
# "object has been modified" attempt 1 silent drop → daemon DS DESIRED 永远 0
# → daemon 不 spawn → channel-0 alloc 给 client 但 daemon 拿不到 → pod 永久 ContainerCreating.
# 修法: kubectl rollout restart deploy/nvidia-dra-driver-gpu-controller
log ""
log "=== Check 12: 每个 active CD 的 daemon DS DESIRED 健康度 (controller race detection) ==="
CD_LIST=$(gx k8n "kubectl get computedomain -A --no-headers 2>&1" | grep -v 'No resources' | awk '{print $2}' | grep -v '^$')
if [ -z "$CD_LIST" ]; then
  ok "no active CD, skip"
else
  CD_BAD=()
  for cd in $CD_LIST; do
    CD_UID=$(gx k8n "kubectl get computedomain $cd -o jsonpath='{.metadata.uid}'" 2>&1 | tr -d '\n')
    if [ -z "$CD_UID" ]; then continue; fi
    DS_LINE=$(gx k8n "kubectl get ds -n nvidia-dra-driver-gpu computedomain-daemon-$CD_UID --no-headers" 2>&1)
    DESIRED=$(echo "$DS_LINE" | awk '{print $2}')
    READY=$(echo "$DS_LINE" | awk '{print $4}')
    # 关联的 client channel RC allocated 数 — 用 yaml + grep 找 domainID match
    CLIENT_ALLOC=$(gx k8n "kubectl get resourceclaim -A -o yaml 2>&1" | grep -c "domainID: $CD_UID")
    CLIENT_ALLOC=${CLIENT_ALLOC:-0}
    if [ "$DESIRED" = "0" ] && [ "$CLIENT_ALLOC" -gt 0 ] 2>/dev/null; then
      CD_BAD+=("$cd:DESIRED=0,READY=0,client_RC_alloc=$CLIENT_ALLOC")
    elif [ -n "$DESIRED" ] && [ "$DESIRED" != "$READY" ]; then
      CD_BAD+=("$cd:DESIRED=$DESIRED,READY=$READY (partial)")
    fi
  done
  if [ "${#CD_BAD[@]}" = "0" ]; then
    ok "all active CD daemon DS healthy"
  else
    fail "controller race? daemon DS scale 异常: ${CD_BAD[@]}"
    warn "  → 修法 (auto with --fix-race): Stage 1 手动 label node CD UID + Stage 2 restart kubelet"
    warn "  → memory: dra-controller-race-2stage-workaround.md"

    if [ "$FIX_RACE" = "1" ]; then
      log "  --fix-race enabled, 触发 2-stage workaround..."
      for cd in $CD_LIST; do
        CD_UID=$(gx k8n "kubectl get computedomain $cd -o jsonpath='{.metadata.uid}'" 2>&1 | tr -d '\n')
        [ -z "$CD_UID" ] && continue
        # Stage 1: label nodes where client pod scheduled (用 channel RC pool 字段反查 node)
        log "  Stage 1: label nodes for CD $cd (UID $CD_UID)"
        PODS_NODES=$(gx k8n "kubectl get resourceclaim -A -o yaml 2>&1 | awk '/domainID: $CD_UID/{found=1} found && /pool:/{print \$2; found=0}' | sort -u")
        for n in $PODS_NODES; do
          gx k8n "kubectl label node $n resource.nvidia.com/computeDomain=$CD_UID --overwrite" 2>&1 | tail -1
        done
        sleep 30
        # check if daemon pod still ContainerCreating after 30s → Stage 2
        DS_READY=$(gx k8n "kubectl get ds -n nvidia-dra-driver-gpu computedomain-daemon-$CD_UID --no-headers" 2>&1 | awk '{print $4}')
        DS_DESIRED=$(gx k8n "kubectl get ds -n nvidia-dra-driver-gpu computedomain-daemon-$CD_UID --no-headers" 2>&1 | awk '{print $2}')
        if [ "$DS_READY" != "$DS_DESIRED" ] || [ "$DS_READY" = "0" ]; then
          log "  Stage 2: daemon still not ready ($DS_READY/$DS_DESIRED), restart kubelet on pod nodes"
          for n in $PODS_NODES; do
            gx k8n "ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null maxwellx@$n 'sudo systemctl restart kubelet'" 2>&1 | tail -1 &
          done
          wait
          log "  Stage 2 done: kubelet restart on $PODS_NODES"
        else
          log "  Stage 1 sufficient — daemon $DS_READY/$DS_DESIRED Ready"
        fi
      done
    fi
  fi
fi

# ============= Check 13: workload pod scheduling health (stale/orphan + Pending) =============
# 撞过 stuck pattern: STS delete 时 1 个 pod force delete failed (sigkill 没响应),
# 残留在 default ns 占着 GPU, 下次 workload 撞 "Insufficient nvidia.com/gpu".
# 也覆盖一般 Pending pod (调度器选不到 node, 通常 GPU 占满 / podAntiAffinity / RC pending).
log ""
log "=== Check 13: workload pod 调度健康 (stale orphan + Pending) ==="
# default ns 上所有 pod (排除已知 system ns)
DEFAULT_PODS=$(gx k8n "kubectl get pods -n default --no-headers 2>&1" | grep -v 'No resources' | grep -v '^$' || true)
ORPHAN_PODS=()
PENDING_PODS=()
if [ -n "$DEFAULT_PODS" ]; then
  # 找 orphan: pod 的 ownerReferences 指向不存在的 STS / Job / etc.
  while read -r line; do
    [ -z "$line" ] && continue
    pod=$(echo "$line" | awk '{print $1}')
    status=$(echo "$line" | awk '{print $3}')
    # 检 ownerReference
    OWNER=$(gx k8n "kubectl get pod -n default $pod -o jsonpath='{.metadata.ownerReferences[0].kind}/{.metadata.ownerReferences[0].name}' 2>&1" || true)
    OWNER_KIND=${OWNER%/*}
    OWNER_NAME=${OWNER#*/}
    if [ -n "$OWNER_KIND" ] && [ -n "$OWNER_NAME" ] && [ "$OWNER_KIND" != "/" ]; then
      # 查 owner 是否还在
      OWNER_EXISTS=$(gx k8n "kubectl get ${OWNER_KIND,,} -n default $OWNER_NAME -o name 2>/dev/null" || true)
      if [ -z "$OWNER_EXISTS" ]; then
        ORPHAN_PODS+=("$pod (owner $OWNER_KIND/$OWNER_NAME 已不存在)")
      fi
    fi
    # 检 Pending
    if [ "$status" = "Pending" ]; then
      REASON=$(gx k8n "kubectl get events -n default --field-selector involvedObject.name=$pod --no-headers 2>&1" | grep -oE 'Insufficient [^,]+|FailedScheduling|cannot allocate' | head -1)
      PENDING_PODS+=("$pod ($REASON)")
    fi
  done <<< "$DEFAULT_PODS"
fi
if [ "${#ORPHAN_PODS[@]}" -gt 0 ]; then
  fail "${#ORPHAN_PODS[@]} orphan pod (owner STS/Job 已删但 pod 残留, 占 GPU 阻塞新 workload)"
  for p in "${ORPHAN_PODS[@]}"; do warn "  - $p"; done
  warn "  → 强清: kubectl delete pod -n default <pod> --force --grace-period=0"
elif [ "${#PENDING_PODS[@]}" -gt 0 ]; then
  fail "${#PENDING_PODS[@]} Pending pod (调度失败, 通常 GPU 占满 / orphan 占位 / podAntiAffinity)"
  for p in "${PENDING_PODS[@]}"; do warn "  - $p"; done
  warn "  → kubectl describe pod -n default <pod> 看 Events, 通常先 cleanup orphan / stale workload"
else
  ok "no stale orphan pod, no Pending pod in default ns"
fi

# ============= Check 11: dmesg Xid 137/145/94 last 24h =============
log ""
log "=== Check 11: worker dmesg Xid 137/145/94 (24h) ==="
XID_BAD=()
for n in "${WORKER_NODES[@]}"; do
  XID=$(gx k8n "ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 maxwellx@$n 'sudo dmesg -T 2>&1 | grep -oE \"Xid \\(PCI:[^)]+\\): (137|145|94)\" | head -1'" 2>&1 | grep -oE "Xid \([^)]+\): (137|145|94)" | head -1)
  if [ -n "$XID" ]; then
    XID_BAD+=("$n:$XID")
    [ "$VERBOSE" = "1" ] && warn "$n: $XID"
  fi
done
if [ "${#XID_BAD[@]}" = "0" ]; then
  ok "no Xid 137/145/94 on any worker"
else
  fail "${#XID_BAD[@]} worker have Xid: ${XID_BAD[@]}"
  warn "  → scripts/troubleshooting/xid-fault-auto-recover.sh --apply"
fi

# ============= Summary =============
log ""
log "=== Summary ==="
if [ "${#FAILED[@]}" = "0" ]; then
  log "🎯 ALL PASS — k8s + DRA driver healthy"
  exit 0
else
  log "⚠ ${#FAILED[@]} CHECK FAILED:"
  for f in "${FAILED[@]}"; do log "  - $f"; done
  exit 1
fi
