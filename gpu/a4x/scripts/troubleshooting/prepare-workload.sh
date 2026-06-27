#!/bin/bash
# prepare-workload.sh <yaml-path> [<master-pod-name>]
#
# Idempotent wrapper that bakes-in all the reproducible workflow steps for
# workload apply on forrest k8s 1.34. Solves:
#
#   1. stale Terminating CD daemons from previous workload (NVIDIA DRA v0.4 GC race)
#   2. DRA controller race: "object has been modified" silent drop → daemon DS DESIRED=0
#   3. kubelet PrepareResource cache stale after CD label switch
#   4. FailedPrepareDynamicResources DeadlineExceeded on freshly-labeled node
#
# Usage:
#   prepare-workload.sh <yaml-path>                  # apply + fix-race + wait (no Ready wait)
#   prepare-workload.sh <yaml-path> <master-pod>     # also wait until master pod Running
#   prepare-workload.sh -d <yaml-path>               # delete only (skip apply, run cleanup)
#
# yaml-path can be local file (will be pushed to master /tmp/) OR master /tmp/* path.
#
# Returns 0 if apply+wait successful, 1 otherwise.

set -uo pipefail

DELETE_ONLY=0
if [ "${1:-}" = "-d" ]; then
  DELETE_ONLY=1
  shift
fi

YAML="${1:-}"
MASTER_POD="${2:-}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

if [ -z "$YAML" ]; then
  sed -n '2,18p' "$0"
  exit 1
fi

log() { echo "[$(date +%H:%M:%S) prepare] $*"; }

# ============================================================
# Step 1: cleanup stale Terminating CD daemons (always)
# ============================================================
log "Step 1: clean stale Terminating CD daemons"
STALE=$(gx k8n "kubectl get pods -n nvidia-dra-driver-gpu --no-headers 2>&1 | awk '\$3==\"Terminating\" {print \$1}' | head -50")
if [ -n "$STALE" ]; then
  log "  force kill $(echo $STALE | wc -w) stale daemons"
  echo "$STALE" | xargs -r -I {} gx k8n "kubectl delete pod -n nvidia-dra-driver-gpu {} --force --grace-period=0 --wait=false" 2>&1 | tail -3
else
  log "  no stale Terminating daemons"
fi

if [ "$DELETE_ONLY" = "1" ]; then
  log "Step 2 (delete only): kubectl delete -f $YAML"
  if [[ "$YAML" == /tmp/* ]]; then
    gx k8n "kubectl delete -f $YAML --wait=false 2>&1 | tail -5"
  else
    gx k8n "cat > /tmp/prep-workload.yaml" < "$YAML"
    gx k8n "kubectl delete -f /tmp/prep-workload.yaml --wait=false 2>&1 | tail -5"
  fi
  exit 0
fi

# ============================================================
# Step 2: apply yaml
# ============================================================
log "Step 2: apply $YAML"
if [[ "$YAML" == /tmp/* ]]; then
  gx k8n "kubectl apply -f $YAML 2>&1" | tail -5
else
  REMOTE=/tmp/prep-workload-$(basename "$YAML")
  gx k8n "cat > $REMOTE" < "$YAML"
  gx k8n "kubectl apply -f $REMOTE 2>&1" | tail -5
fi

# ============================================================
# Step 3: wait pod scheduled (DRA controller has chance to allocate channel)
# ============================================================
log "Step 3: sleep 25s for pod schedule + DRA controller allocate"
sleep 25

# ============================================================
# Step 4: proactive fix-race (catch controller "object modified" silent drop)
# ============================================================
log "Step 4: fix-race (Stage 1 label + Stage 2 kubelet restart if needed)"
bash "$SCRIPT_DIR/check-k8s-dra-health.sh" --fix-race 2>&1 | grep -E '^\[|Stage|node/|FAIL|race' | tail -15

# ============================================================
# Step 5: wait master pod Running (if specified)
# ============================================================
if [ -n "$MASTER_POD" ]; then
  log "Step 5: wait $MASTER_POD Running (max 6min)"
  for i in $(seq 1 36); do
    PHASE=$(gx k8n "kubectl get pod $MASTER_POD -o jsonpath='{.status.phase}' 2>/dev/null" 2>&1 | tr -d '\n')
    INIT_STATUS=$(gx k8n "kubectl get pod $MASTER_POD -o jsonpath='{.status.initContainerStatuses[0].state}' 2>/dev/null" 2>&1)
    log "  $i/36 phase=$PHASE init=$(echo $INIT_STATUS | head -c 50)"
    if [ "$PHASE" = "Running" ]; then
      log "  ✓ $MASTER_POD Running"
      exit 0
    fi
    # second-chance fix-race if 3 min stuck Pending/Init
    if [ "$i" = "18" ] && { [ "$PHASE" = "Pending" ] || [ -z "$PHASE" ] || echo "$INIT_STATUS" | grep -q waiting; }; then
      log "  ⚠ stuck at $i*10s, retry fix-race"
      bash "$SCRIPT_DIR/check-k8s-dra-health.sh" --fix-race 2>&1 | tail -10
    fi
    sleep 10
  done
  log "  ✗ TIMEOUT waiting $MASTER_POD Running"
  exit 1
fi

log "Step 5: skipped (no master-pod arg, returning after fix-race)"
exit 0
