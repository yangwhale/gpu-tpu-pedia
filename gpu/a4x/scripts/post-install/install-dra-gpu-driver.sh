#!/bin/bash
# install-dra-gpu-driver.sh — 装 NVIDIA DRA GPU Driver (提供 ComputeDomain CRD)
# 在 master 本地跑 (ansible push 到 master 即可, 无需 cloudtop SSH)
# 前置: /etc/kubernetes/admin.conf 存在
#
# nvidiaDriverRoot:
#   - TLinux 4 image 预装 driver 在系统标准路径 (/usr/lib64 + /usr/bin/nvidia-smi)
#   - 默认 "/" (host root) 工作 — Helm chart 自动在 host 上找
#   - 客户环境不同 image 时, 节点本地确认:
#       find / -name 'libnvidia-ml.so*' 2>/dev/null | head -3
#       ls -d /usr/lib64/nvidia/* /usr/lib/nvidia 2>/dev/null
#     再 NVIDIA_DRIVER_ROOT=... bash install-dra-gpu-driver.sh override
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../host/env.sh"

# 默认尝试 "/" (root 文件系统),让 chart 在 host 上找 driver
# 第 1 台 worker 起来后,实测正确路径再 update --set 重装
NVIDIA_DRIVER_ROOT="${NVIDIA_DRIVER_ROOT:-/}"
echo "Using nvidiaDriverRoot=$NVIDIA_DRIVER_ROOT (override with NVIDIA_DRIVER_ROOT env)"

sudo bash -c "
set -e
command -v helm >/dev/null 2>&1 || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

export KUBECONFIG=/etc/kubernetes/admin.conf

helm upgrade --install nvidia-dra-driver-gpu '"$DRA_GPU_DRIVER_HELM"' \
  --version '"$DRA_GPU_DRIVER_VERSION"' \
  --namespace nvidia-dra-driver-gpu --create-namespace \
  --set nameOverride=nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot='"$NVIDIA_DRIVER_ROOT"' \
  --set controller.affinity=null \
  --set controller.priorityClassName='' \
  --set kubeletPlugin.priorityClassName='' \
  --set gpuResourcesEnabledOverride=true \
  --set kubeletPlugin.tolerations[0].key=nvidia.com/gpu \
  --set kubeletPlugin.tolerations[0].operator=Exists \
  --set kubeletPlugin.tolerations[0].effect=NoSchedule \
  --set kubeletPlugin.tolerations[1].key=kubernetes.io/arch \
  --set kubeletPlugin.tolerations[1].operator=Equal \
  --set kubeletPlugin.tolerations[1].value=arm64 \
  --set kubeletPlugin.tolerations[1].effect=NoSchedule

# v0.4.0 CRD 手动 apply (helm 默认不带,numNodes 移除 required)
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/dra-driver-nvidia-gpu/v'"$DRA_GPU_DRIVER_VERSION"'/deployments/helm/dra-driver-nvidia-gpu/crds/resource.nvidia.com_computedomains.yaml

echo
echo '=== verify ==='
kubectl get crd computedomains.resource.nvidia.com -o jsonpath=\"{.spec.versions[0].schema.openAPIV3Schema.properties.spec.required}{\\n}\"
echo \"  expected: [channel] without numNodes (v0.4.0 CRD)\"
echo
kubectl -n nvidia-dra-driver-gpu get ds
kubectl api-resources | grep computedomain
"
