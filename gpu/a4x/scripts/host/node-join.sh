#!/bin/bash
# node-join.sh — 节点本地跑 kubeadm join (ansible 推送模型, 无需 master ssh 节点)
#
# 用法 (在节点上跑, 通常 ansible push):
#   JOIN_CMD="kubeadm join 10.10.0.18:6443 --token <T> --discovery-token-ca-cert-hash sha256:<H>" \
#     bash node-join.sh
#
# 或分散变量:
#   CP_IP=10.10.0.18 JOIN_TOKEN=abc.def JOIN_CACERT_HASH=sha256:xxx bash node-join.sh
#
# 前置: startup-forrest-gb200-k8s134.sh 已跑完 (kubelet/kubeadm 1.34.1 装好, /etc/.startup-done 在)
# 后续: kubelet 用 VPC IP 注册到 apiserver, label 由 device-plugin + IMEX Manager 自动设
set -euo pipefail

NODE_NAME="${NODE_NAME:-$(hostname -s)}"

if [ -z "${JOIN_CMD:-}" ]; then
  : "${CP_IP:?must set CP_IP (master VPC IP) or pass full JOIN_CMD}"
  : "${JOIN_TOKEN:?must set JOIN_TOKEN or pass full JOIN_CMD}"
  : "${JOIN_CACERT_HASH:?must set JOIN_CACERT_HASH (sha256:...) or pass full JOIN_CMD}"
  JOIN_CMD="kubeadm join ${CP_IP}:6443 --token ${JOIN_TOKEN} --discovery-token-ca-cert-hash ${JOIN_CACERT_HASH}"
fi

# Sanity: kubelet/kubeadm 必须就绪
command -v kubeadm >/dev/null 2>&1 || { echo "FATAL: kubeadm not installed (startup-*.sh 未跑完?)"; exit 1; }
[ -f /etc/.startup-done ] || echo "WARN: /etc/.startup-done missing (startup 可能未完成, 继续)"

# 已 join 过则 reset (idempotent)
if [ -f /etc/kubernetes/kubelet.conf ]; then
  echo "已 join 过 (/etc/kubernetes/kubelet.conf 在), kubeadm reset -f 后重 join..."
  sudo kubeadm reset -f 2>&1 | tail -5
  sudo rm -rf /etc/cni/net.d /var/lib/etcd
fi

echo "==========================================================="
echo "  node-join: ${NODE_NAME}"
echo "  cmd: sudo ${JOIN_CMD} --node-name ${NODE_NAME} --ignore-preflight-errors=Hostname"
echo "==========================================================="

sudo ${JOIN_CMD} \
  --node-name "${NODE_NAME}" \
  --ignore-preflight-errors=Hostname

echo
echo "✅ ${NODE_NAME} joined. 在能 kubectl 的地方 verify:"
echo "    kubectl get node ${NODE_NAME} -o wide"
echo
echo "label 全自动 (无需 ssh / label-node-after-join.sh):"
echo "  - nvidia.com/gpu.{count,present,product} ← device-plugin"
echo "  - nvidia.com/gpu.{cliqueid,clusteruuid}  ← IMEX Manager DaemonSet"
echo "  yaml 调度用 topologyKey: nvidia.com/gpu.cliqueid"
