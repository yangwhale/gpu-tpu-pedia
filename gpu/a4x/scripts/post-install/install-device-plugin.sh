#!/bin/bash
# install-device-plugin.sh — 一次性装 nvidia-device-plugin + gpu-feature-discovery via Helm
# 在 master 本地跑 (ansible push 到 master 即可, 无需 cloudtop SSH)
# 前置: /etc/kubernetes/admin.conf 存在 (kubeadm init 后自带)
#
# 装 2 个 chart:
# 1. nvidia-device-plugin — advertise nvidia.com/gpu 给 kubelet
# 2. gpu-feature-discovery (GFD) — 给 node 打官方 label (nvidia.com/gpu.clique 等)
#    包含 NFD (Node Feature Discovery) subchart
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../host/env.sh"

sudo bash -c "
set -e
command -v helm >/dev/null 2>&1 || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

export KUBECONFIG=/etc/kubernetes/admin.conf
helm repo add nvdp '"$NVDP_HELM_REPO"' 2>/dev/null || true
helm repo update

helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --version '"$DEVICE_PLUGIN_VERSION"' \
  --namespace kube-system \
  --set-string nodeSelector.\"feature\\.node\\.kubernetes\\.io/pci-10de\\.present\"=true \
  --wait --timeout 300s
# ⚠️ Phase 11 retro: device-plugin 也用 NFD 自动设的 feature.node.kubernetes.io/pci-10de.present
# 不用 nvidia.com/gpu.present (GFD 0.19+ 不再 set 这个 label, 是历史 label, 用了会卡死 chicken-egg)

helm upgrade --install gpu-feature-discovery nvdp/gpu-feature-discovery \
  --version '"$GFD_VERSION"' \
  --namespace kube-system \
  --set-string nodeSelector.\"feature\\.node\\.kubernetes\\.io/pci-10de\\.present\"=true \
  --wait --timeout 300s
# ⚠️ Phase 11 retro: GFD nodeSelector 用 NFD 自动设的 feature.node.kubernetes.io/pci-10de.present
# 而不是 nvidia.com/gpu.present (后者是 GFD 自己设的 → chicken-egg: 新 worker join 后 GFD 不 schedule)
# pci-10de.present 是 NFD worker detect PCI vendor 10de (NVIDIA) 后自动 set 的, 任何 GPU 节点 join 后秒级 set

echo
echo '=== verify ==='
kubectl get pods -n kube-system -l app.kubernetes.io/name=nvidia-device-plugin -o wide
kubectl get nodes -L nvidia.com/gpu.clique,nvidia.com/gpu.count
"
