#!/bin/bash
# apply-rdma-deviceclass.sh — apply RDMA DeviceClass (DRA 路径 A)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# yaml 优先 deliverables/yamls/system/ (R3 reorg 后), 老 layout 保留 fallback
YAML=""
for p in \
  "$SCRIPT_DIR/../../yamls/system/k8s134-rdma-deviceclass.yaml" \
  "$SCRIPT_DIR/../yamls/k8s134-rdma-deviceclass.yaml" \
  "$SCRIPT_DIR/../../yamls/k8s134/k8s134-rdma-deviceclass.yaml"; do
  [ -f "$p" ] && YAML="$p" && break
done
[ -z "$YAML" ] && { echo "ERROR: yaml not found (tried ../../yamls/system/, ../yamls/, ../../yamls/k8s134/)"; exit 1; }
echo "Using yaml: $YAML"
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf apply -f "$YAML"
sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf get deviceclass
