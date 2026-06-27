#!/bin/bash
# xid-fault-auto-recover.sh — 自动检测 + 恢复 GB200 NVLink Xid fault 故障节点
#
# 检测: kubectl get nodes 找 nvidia.com/gpu allocatable < 4 节点
# 验证: ssh 节点 dmesg 看是否含 Xid 137/145 (NVLink fabric fault, sticky)
# 恢复: gcloud compute instances reset (host reboot) + force delete stale device-plugin pod
# 监控: 等节点 Ready + GPU=4
#
# 用法:
#   bash xid-fault-auto-recover.sh                          # detect + dry-run (列受影响节点 + 不动)
#   bash xid-fault-auto-recover.sh --apply                  # detect + serial fix (default, no cascade)
#   bash xid-fault-auto-recover.sh --apply --parallel       # parallel fix (faster but triggers IMEX storm cascade!)
#   bash xid-fault-auto-recover.sh --node forrest-gb200-XX --apply  # 单节点 fix
#
# ⚠️ SERIAL mode is default to avoid IMEX storm cascade:
#   12 nodes parallel reset → 10 NEW Xid faults on healthy nodes (45% trigger rate)
#   single reset → 0 cascade.
#   Serial is N × ~10 min (slower) but converges in 1 pass instead of cascade rounds.
#
# 前置: gx k8n 配通, gcloud project=gpu-launchpad-playground 默认
# 跑位置: 任意有 gx 的本机

set -uo pipefail

PROJECT="${GCP_PROJECT:-gpu-launchpad-playground}"
ZONE="${GCP_ZONE:-us-east1-d}"
EXPECTED_GPU=4
APPLY=0
SINGLE_NODE=""
SERIAL=1   # default serial (avoid IMEX storm cascade), --parallel to override

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --node) SINGLE_NODE="$2"; shift 2 ;;
    --parallel) SERIAL=0; shift ;;
    --help|-h)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *) echo "ERROR: unknown arg $1"; exit 1 ;;
  esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ------------- Step 1: detect bad nodes -------------
log "=== Step 1: detect bad nodes (nvidia.com/gpu allocatable < $EXPECTED_GPU) ==="

if [ -n "$SINGLE_NODE" ]; then
  BAD_NODE_NAMES="$SINGLE_NODE"
  log "single-node mode: $SINGLE_NODE"
else
  BAD_NODE_LINES=$(gx k8n "kubectl get nodes -o json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
bad = []
for n in d['items']:
    name = n['metadata']['name']
    if 'gb200' not in name:
        continue
    alloc = n.get('status', {}).get('allocatable', {})
    gpu = int(alloc.get('nvidia.com/gpu', '0'))
    if gpu < $EXPECTED_GPU:
        bad.append((name, gpu))
for name, gpu in sorted(bad):
    print(f'{name} {gpu}')
")
  BAD_NODE_NAMES=$(echo "$BAD_NODE_LINES" | awk '{print $1}')
fi

if [ -z "$BAD_NODE_NAMES" ]; then
  log "✓ no bad nodes detected"
  exit 0
fi

if [ -n "${BAD_NODE_LINES:-}" ]; then
  log "found bad nodes:"
  echo "$BAD_NODE_LINES" | while read line; do log "  - $line"; done
fi

# convert to bash array
declare -a BAD_NODES_ARR=()
for n in $BAD_NODE_NAMES; do
  BAD_NODES_ARR+=("$n")
done

# ------------- Step 2: verify Xid 137/145 -------------
log ""
log "=== Step 2: verify Xid 137/145 on each bad node (parallel SSH) ==="

declare -a VERIFIED_NODES_ARR=()
declare -A NODE_XIDS_MAP=()

# write parallel ssh outputs to per-node tmp files
TMPDIR=$(mktemp -d /tmp/xid-fault.XXXXXX)
trap "rm -rf $TMPDIR" EXIT

for NODE in "${BAD_NODES_ARR[@]}"; do
  (
    XIDS=$(gx k8n "ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 maxwellx@$NODE 'sudo dmesg 2>&1 | grep -oE \"Xid \\(PCI:[^)]+\\): [0-9]+\" | grep -oE \"[0-9]+\$\" | sort -nu | tr \"\\n\" \" \"'" 2>&1 | grep -oE '^[0-9 ]+$' | tail -1 | tr -s ' ')
    echo "$XIDS" > "$TMPDIR/$NODE"
  ) &
done
wait

# collect results
for NODE in "${BAD_NODES_ARR[@]}"; do
  XIDS=$(cat "$TMPDIR/$NODE" 2>/dev/null | tr -d '\n')
  if echo "$XIDS" | grep -qE "137|145"; then
    log "  ✓ $NODE: NVLink fault confirmed (Xid: $XIDS)"
    VERIFIED_NODES_ARR+=("$NODE")
    NODE_XIDS_MAP["$NODE"]="$XIDS"
  else
    log "  ⚠ $NODE: no NVLink fault Xid (Xid: $XIDS) — skip (manual investigation needed)"
  fi
done

if [ "${#VERIFIED_NODES_ARR[@]}" = "0" ]; then
  log "✗ no nodes confirmed NVLink fault — exit"
  exit 0
fi

log ""
log "=== verified ${#VERIFIED_NODES_ARR[@]} NVLink-fault nodes (will fix via host reset) ==="

if [ "$APPLY" = "0" ]; then
  log ""
  log "DRY RUN MODE — not applying fix. Add --apply to actually reset nodes."
  exit 0
fi

# ------------- Step 3 + 4: reset + wait Ready (SERIAL default) -------------
# WHY serial: parallel batch reset triggers IMEX storm cascade — N parallel
# nvidia-imex daemon stops cause cross-cluster MNNVL fabric session race,
# triggering NEW NVLink Xid faults on previously-healthy nodes.
# 实测: 12 parallel → 10 new fault; 10 parallel → 3 new; single → 0 new.
# Use --parallel to override (faster but causes cascade).

if [ "$SERIAL" = "1" ]; then
  log ""
  log "=== Step 3+4: serial reset (avoid IMEX cascade, ~10 min per node) ==="
  BOOT_TIMEOUT=900
  TOTAL_START=$(date +%s)

  for NODE in "${VERIFIED_NODES_ARR[@]}"; do
    log ""
    log "--- $NODE: issue reset + wait Ready ---"
    NODE_START=$(date +%s)
    gx k8n "gcloud compute instances reset $NODE --project=$PROJECT --zone=$ZONE --quiet" > /dev/null 2>&1
    log "  reset issued"

    while true; do
      STATUS=$(gx k8n "kubectl get node $NODE -o jsonpath='{.status.conditions[?(@.type==\"Ready\")].status}'" 2>/dev/null)
      if [ "$STATUS" = "True" ]; then
        log "  ✓ Ready=True ($((($(date +%s) - NODE_START)))s)"
        break
      fi
      ELAPSED=$(($(date +%s) - NODE_START))
      if [ "$ELAPSED" -gt "$BOOT_TIMEOUT" ]; then
        log "  ✗ TIMEOUT after ${BOOT_TIMEOUT}s"
        exit 2
      fi
      log "  [+${ELAPSED}s] still NotReady"
      sleep 30
    done
  done

  BOOT_START=$TOTAL_START
  log ""
  log "✓ all ${#VERIFIED_NODES_ARR[@]} nodes reset + Ready ($((($(date +%s) - TOTAL_START)))s total, serial)"
else
  log ""
  log "=== Step 3: gcloud compute instances reset (PARALLEL — may cause cascade) ==="
  RESET_START=$(date +%s)
  for NODE in "${VERIFIED_NODES_ARR[@]}"; do
    log "  issuing reset: $NODE"
    gx k8n "gcloud compute instances reset $NODE --project=$PROJECT --zone=$ZONE --quiet" > /dev/null 2>&1 &
  done
  wait
  log "✓ all reset issued ($((($(date +%s) - RESET_START)))s)"

  log ""
  log "=== Step 4: wait nodes recovery (~8-10 min per node) ==="
  BOOT_TIMEOUT=900
  BOOT_START=$(date +%s)

  while true; do
    PENDING=""
    for NODE in "${VERIFIED_NODES_ARR[@]}"; do
      STATUS=$(gx k8n "kubectl get node $NODE -o jsonpath='{.status.conditions[?(@.type==\"Ready\")].status}'" 2>/dev/null)
      if [ "$STATUS" != "True" ]; then
        PENDING="$PENDING $NODE"
      fi
    done

    if [ -z "$PENDING" ]; then
      log "✓ all nodes Ready=True ($((($(date +%s) - BOOT_START)))s after reset)"
      break
    fi

    ELAPSED=$(($(date +%s) - BOOT_START))
    if [ "$ELAPSED" -gt "$BOOT_TIMEOUT" ]; then
      log "✗ TIMEOUT: still not Ready after ${BOOT_TIMEOUT}s: $PENDING"
      exit 2
    fi

    log "  [+${ELAPSED}s] pending Ready: $PENDING"
    sleep 30
  done
fi

# ------------- Step 5: force delete stale device-plugin pod -------------
log ""
log "=== Step 5: force delete stale device-plugin pod (let DS spawn fresh) ==="

for NODE in "${VERIFIED_NODES_ARR[@]}"; do
  STALE=$(gx k8n "kubectl get pods -n kube-system -o wide" 2>&1 | grep nvidia-device-plugin | grep "$NODE " || echo "")
  STATE=$(echo "$STALE" | awk '{print $3}')
  POD=$(echo "$STALE" | awk '{print $1}' | head -1)
  if echo "$STATE" | grep -qE "Unknown|Pending|Terminating"; then
    log "  $NODE: force delete stale pod $POD ($STATE)"
    gx k8n "kubectl delete pod $POD -n kube-system --grace-period=0 --force" > /dev/null 2>&1
  else
    log "  $NODE: device-plugin OK (state=$STATE, no force delete needed)"
  fi
done

# ------------- Step 6: verify GPU=4 (poll up to 5 min) -------------
log ""
log "=== Step 6: poll + verify GPU allocatable=$EXPECTED_GPU (max 5 min) ==="

ALL_OK=0
for i in $(seq 1 20); do
  ALL_OK=1
  PENDING_GPU=""
  for NODE in "${VERIFIED_NODES_ARR[@]}"; do
    GPU=$(gx k8n "kubectl get node $NODE -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'" 2>/dev/null)
    if [ "$GPU" != "$EXPECTED_GPU" ]; then
      ALL_OK=0
      PENDING_GPU="$PENDING_GPU $NODE($GPU)"
    fi
  done
  if [ "$ALL_OK" = "1" ]; then
    log "  ✓ all nodes GPU=$EXPECTED_GPU (took ${i}×15s)"
    break
  fi
  log "  [poll $i/20] pending GPU=$EXPECTED_GPU:$PENDING_GPU"
  sleep 15
done

# final summary
for NODE in "${VERIFIED_NODES_ARR[@]}"; do
  GPU=$(gx k8n "kubectl get node $NODE -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'" 2>/dev/null)
  if [ "$GPU" = "$EXPECTED_GPU" ]; then
    log "  ✓ $NODE: GPU=$GPU"
  else
    log "  ✗ $NODE: GPU=$GPU (expected $EXPECTED_GPU) — needs manual check"
  fi
done

TOTAL=$(($(date +%s) - BOOT_START))
if [ "$ALL_OK" = "1" ]; then
  log ""
  log "🎯 SUCCESS: all ${#VERIFIED_NODES_ARR[@]} nodes recovered (total $((TOTAL/60))m $((TOTAL%60))s)"
  exit 0
else
  log ""
  log "⚠ PARTIAL: some nodes not fully recovered, manual check needed"
  exit 1
fi
