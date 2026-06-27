#!/bin/bash
# dra-cd-label-reconciler.sh
#
# Workaround for NVIDIA DRA driver v0.4.0 controller race:
#   "Operation cannot be fulfilled on computedomains... the object has been modified (attempt 1)"
#   → controller silent drops the work item, doesn't add `resource.nvidia.com/computeDomain=<CD-UID>` label
#   to client pod's node, daemon DS DESIRED stays 0.
#
# This reconciler periodically:
#   1. List all active ComputeDomains
#   2. For each CD, find all `compute-domain-channel` ResourceClaims allocated to it
#   3. For each RC, find owning Pod's nodeName
#   4. If node lacks `resource.nvidia.com/computeDomain=<CD-UID>` label, patch it
#
# Survives pod recreate / re-schedule / cross-CD scenarios that the original Monitor
# stage-1 single-fire fix doesn't cover.
#
# Usage:
#   bash dra-cd-label-reconciler.sh                # default INTERVAL=30s
#   INTERVAL=10 bash dra-cd-label-reconciler.sh    # faster reconcile
#   bash dra-cd-label-reconciler.sh --once         # single reconcile pass + exit (dry-run / test)
#   bash dra-cd-label-reconciler.sh --verbose      # print "no change" lines too
#
# Where to run: any host with `gx` (delegates to master via gx k8n).
#
# To run permanently in this session: bash dra-cd-label-reconciler.sh &
# For cluster-resident operator-style deployment, see Option B yaml in
# yamls/k8s134/dra-cd-label-reconciler-deployment.yaml.

set -uo pipefail

INTERVAL=${INTERVAL:-30}
ONCE=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) ONCE=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    --interval) INTERVAL=$2; shift 2 ;;
    --help|-h) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown arg $1"; exit 1 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

reconcile_once() {
  local change_count=0

  # list all active CD: namespace/name|uid per line
  local CD_LINES
  # NB: must use full name `computedomain` — server has no short-name `cd` registered, silent fail otherwise.
  CD_LINES=$(gx k8n "kubectl get computedomain -A -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}|{.metadata.uid}{\"\n\"}{end}'" 2>&1 | grep -v "No resources" | grep -v "^error" | grep -v "^$")

  if [ -z "$CD_LINES" ]; then
    [ "$VERBOSE" = "1" ] && log "no active CD"
    return 0
  fi

  while IFS='|' read -r CD_FULL CD_UID; do
    [ -z "$CD_FULL" ] || [ -z "$CD_UID" ] && continue
    local NS=${CD_FULL%/*}
    local CD_NAME=${CD_FULL#*/}

    # find all channel RC allocated to this CD UID; output ns/podName per line
    local POD_REFS
    POD_REFS=$(gx k8n "kubectl get resourceclaim -n $NS -o json" 2>&1 | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    pods = set()
    for rc in d.get('items', []):
        name = rc['metadata']['name']
        if 'compute-domain-channel' not in name: continue
        alloc = rc.get('status', {}).get('allocation')
        if not alloc: continue
        cfgs = alloc.get('devices', {}).get('config', []) or []
        for cfg in cfgs:
            params = cfg.get('opaque', {}).get('parameters', {})
            if params.get('domainID') == '$CD_UID':
                for owner in (rc.get('metadata', {}).get('ownerReferences') or []):
                    if owner.get('kind') == 'Pod':
                        pods.add(owner['name'])
                break
    for p in sorted(pods):
        print(p)
except Exception as e:
    pass
" 2>&1)

    [ -z "$POD_REFS" ] && continue

    # for each pod, find its nodeName, check if node has correct CD label
    while read -r pod; do
      [ -z "$pod" ] && continue
      local NODE
      NODE=$(gx k8n "kubectl get pod -n $NS $pod -o jsonpath='{.spec.nodeName}'" 2>&1 | tr -d '\n')
      if [ -z "$NODE" ]; then continue; fi
      local CURR
      CURR=$(gx k8n "kubectl get node $NODE -o jsonpath='{.metadata.labels.resource\.nvidia\.com/computeDomain}'" 2>&1 | tr -d '\n')
      if [ "$CURR" = "$CD_UID" ]; then
        [ "$VERBOSE" = "1" ] && log "  ✓ $NODE already labeled CD=$NS/$CD_NAME"
        continue
      fi
      log "  + label $NODE → CD $NS/$CD_NAME (UID=$CD_UID, was='$CURR', pod=$pod)"
      gx k8n "kubectl label node $NODE resource.nvidia.com/computeDomain=$CD_UID --overwrite" 2>&1 | tail -1
      change_count=$((change_count + 1))
    done <<< "$POD_REFS"
  done <<< "$CD_LINES"

  [ "$change_count" -gt 0 ] && log "reconcile pass: $change_count label change(s)"
  return 0
}

if [ "$ONCE" = "1" ]; then
  log "single reconcile pass (--once)"
  reconcile_once
  log "done"
  exit 0
fi

log "DRA CD label reconciler starting, INTERVAL=${INTERVAL}s (Ctrl-C to stop)"
log "watching: all CDs in all namespaces; labels: resource.nvidia.com/computeDomain=<CD-UID>"

while true; do
  reconcile_once
  sleep "$INTERVAL"
done
