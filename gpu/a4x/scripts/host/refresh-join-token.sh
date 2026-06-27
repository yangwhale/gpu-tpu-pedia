#!/bin/bash
# refresh-join-token.sh — 在 master 本地生成新 kubeadm join token + hash
# 用法: bash refresh-join-token.sh
# 输出:
#   - 完整 kubeadm join 命令 (一行)
#   - TOKEN / HASH 分量 (供 ansible variable 注入)
#   - node-join.sh 调用示例 (推荐: ansible push 到节点本地跑)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

JOIN_CMD=$(sudo kubeadm --kubeconfig=/etc/kubernetes/admin.conf token create --print-join-command)
TOKEN=$(echo "$JOIN_CMD" | grep -oP '(?<=--token )\S+')
HASH=$(echo "$JOIN_CMD" | grep -oP 'sha256:\S+')

echo "==========================================================="
echo "  k8s join command (valid 24h)"
echo "==========================================================="
echo "Full join command:"
echo "  $JOIN_CMD"
echo
echo "Components:"
echo "  CP_IP=$CP_VPC_IP"
echo "  JOIN_TOKEN=$TOKEN"
echo "  JOIN_CACERT_HASH=$HASH"
echo
echo "==========================================================="
echo "推荐: ansible 推送 node-join.sh 到节点, 节点本地跑"
echo "==========================================================="
echo "  ansible -i inv all -m copy -a 'src=scripts/k8s134/node-join.sh dest=/tmp/'"
echo "  ansible -i inv all -m shell -a \\"
echo "    'CP_IP=$CP_VPC_IP JOIN_TOKEN=$TOKEN JOIN_CACERT_HASH=$HASH bash /tmp/node-join.sh'"
echo
echo "或直接传完整 JOIN_CMD:"
echo "  ansible -i inv all -m shell -a 'JOIN_CMD=\"$JOIN_CMD\" bash /tmp/node-join.sh'"
echo
echo "VM startup-script metadata (新 VM init 时一并传, 节点上 startup script 末尾 NEXT 段提示):"
echo "  --metadata=cp-ip=$CP_VPC_IP[,tailscale-authkey=\$TS_AUTHKEY 仅 forrest POC]"
