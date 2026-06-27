#!/bin/bash
# verify-worker.sh — 10min 内确认 worker join 成功 (Ready + GPU=4)
# 在 master 本地跑 (ansible push 即可, 不需要从 cloudtop SSH)
# 用法: bash verify-worker.sh forrest-gb200-01
set -euo pipefail
# 脚本只用 NODE / 内置 KCTL / MAX_WAIT,无外部 env 依赖 (env.sh 在 host/, 不在 post-install/ 故不 source)

NODE="${1:?Usage: $0 <node-name>}"
KCTL="sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf"
MAX_WAIT=600
INTERVAL=15
ELAPSED=0

echo "Watching $NODE (max ${MAX_WAIT}s)..."
while [ $ELAPSED -lt $MAX_WAIT ]; do
  STATUS=$($KCTL get node $NODE --no-headers 2>/dev/null | awk '{print $2}')
  GPU=$($KCTL get node $NODE -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null)
  IP=$($KCTL get node $NODE -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
  echo "  [${ELAPSED}s] Status=$STATUS GPU=$GPU IP=$IP"
  [ "$STATUS" = "Ready" ] && [ "$GPU" = "4" ] && {
    echo
    echo "✅ $NODE Ready @ ${IP} (VPC IP) with $GPU GPUs"
    exit 0
  }
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

echo "❌ TIMEOUT after ${MAX_WAIT}s: Status=$STATUS GPU=$GPU"
echo "Debug 在节点上看: sudo tail -100 /var/log/forrest-worker-init.log"
echo "Debug 在 master 上看: $KCTL describe node $NODE | tail -30"
exit 1
