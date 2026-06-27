#!/bin/bash
# install-dranet.sh — 一次性装 DRANET v1.3.0 (DRA 网络驱动)
# 在 master 本地跑 (ansible push 到 master 即可, 无需 cloudtop SSH)
# 前置: /etc/kubernetes/admin.conf 存在
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../host/env.sh"

sudo bash -c "
set -e
command -v helm >/dev/null 2>&1 || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

export KUBECONFIG=/etc/kubernetes/admin.conf

# nodeSelector: 不设 (chart default empty), 永远无 chicken-egg
# DRANET DS schedule 全节点 (含 cp), 无 RDMA 硬件节点 dranet 只 publish veth 不影响
# 之前用 nvidia.com/gpu.present=true 是 historic mistake (那是 gpu-operator label, 我们没装 operator)
# 之前用 nvidia.com/gpu.family=blackwell 也行但绑 GPU 架构, 加 H100/A100 时要改
helm upgrade --install dranet '"$DRANET_HELM"' \
  --version '"$DRANET_VERSION"' \
  --namespace dranet-system --create-namespace \
  --reset-values \
  --wait --timeout 300s

echo
echo '=== verify ==='
kubectl -n dranet-system get pods -o wide
kubectl get resourceslice -o wide 2>/dev/null | head -20 || echo \"resourceslice not yet populated, wait worker nodes\"
"
