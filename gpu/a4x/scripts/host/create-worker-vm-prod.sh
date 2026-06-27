#!/bin/bash
# create-worker-vm-prod.sh — 创建 1 台 GB200 worker VM (PROD: 无 tailscale, 自动 join)
#
# 跟 create-worker-vm.sh 差异:
#   ✗ 不传 tailscale-authkey metadata
#   + 自动从 master 拿 cp-ssh-pubkey + cp-join-cmd 传 metadata
#   + 用 startup-forrest-gb200-k8s134-prod.sh (Phase 9.5 auto-join)
#   + placement policy 自动选 (05-20 → SD/clique A, 21-36 → CD/clique B)
#
# 用法:
#   bash create-worker-vm-prod.sh forrest-gb200-05
#   bash create-worker-vm-prod.sh forrest-gb200-21 --policy=a4x-nvl72-policy
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

INSTANCE_NAME="${1:?Usage: $0 <instance-name> [--policy=<custom>]}"
POLICY_OVERRIDE=""
for arg in "${@:2}"; do
  case "$arg" in
    --policy=*) POLICY_OVERRIDE="${arg#--policy=}" ;;
  esac
done

# 自动选 placement policy: 01/02/05-20 → SD (clique A), 03/04/21-36 → CD (clique B)
if [ -n "$POLICY_OVERRIDE" ]; then
  POLICY="$POLICY_OVERRIDE"
else
  NUM=$(echo "$INSTANCE_NAME" | grep -oE '[0-9]+$')
  if [ -z "$NUM" ]; then
    echo "ERROR: cannot extract number from $INSTANCE_NAME; use --policy="; exit 1
  fi
  if [ "$NUM" -le 2 ] || { [ "$NUM" -ge 5 ] && [ "$NUM" -le 20 ]; }; then
    POLICY="$PLACEMENT_POLICY_SD"   # forrest-a4x-1x72-policy (clique A, 18 slots)
  elif { [ "$NUM" -ge 3 ] && [ "$NUM" -le 4 ]; } || { [ "$NUM" -ge 21 ] && [ "$NUM" -le 36 ]; }; then
    POLICY="$PLACEMENT_POLICY_CD"   # a4x-nvl72-policy (clique B, 18 slots)
  else
    echo "ERROR: instance num $NUM 不在 01-36 范围; use --policy="; exit 1
  fi
fi

STARTUP_SCRIPT="${SCRIPT_DIR}/startup-forrest-gb200-k8s134-prod.sh"
[ -f "$STARTUP_SCRIPT" ] || { echo "ERROR: $STARTUP_SCRIPT not found"; exit 1; }

# 从 master 拿 cp-ssh-pubkey + cp-join-cmd
echo "==========================================================="
echo "  Fetch cp-ssh-pubkey + cp-join-cmd from master (${CP_NAME})"
echo "==========================================================="
CP_SSH_PUBKEY=$(gx k8n cat /home/maxwellx/.ssh/google_compute_engine.pub 2>/dev/null | head -1)
[ -z "$CP_SSH_PUBKEY" ] && { echo "ERROR: 拿 cp-ssh-pubkey 失败"; exit 1; }

# kubeadm token create on master (token valid 24h, 多节点串行 create 没事)
CP_JOIN_CMD=$(gx k8n 'sudo kubeadm --kubeconfig=/etc/kubernetes/admin.conf token create --print-join-command' 2>/dev/null | grep '^kubeadm join' | head -1)
[ -z "$CP_JOIN_CMD" ] && { echo "ERROR: 拿 cp-join-cmd 失败"; exit 1; }

echo "  cp-ssh-pubkey: ${CP_SSH_PUBKEY:0:50}..."
echo "  cp-join-cmd  : ${CP_JOIN_CMD:0:60}..."

echo
echo "==========================================================="
echo "  Create VM: $INSTANCE_NAME (PROD mode, auto-join)"
echo "  zone     = $ZONE"
echo "  image    = $IMAGE ($IMAGE_PROJECT)"
echo "  policy   = $POLICY"
echo "  CP_IP    = $CP_VPC_IP"
echo "  startup  = $STARTUP_SCRIPT"
echo "==========================================================="

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image="$IMAGE" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type="$BOOT_DISK_TYPE" \
  --scopes=cloud-platform \
  --tags=allow-iap \
  --reservation-affinity=specific --reservation="$RESERVATION" \
  --provisioning-model=RESERVATION_BOUND \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --resource-policies="$POLICY" \
  --network-interface=nic-type=GVNIC,network="$GVNIC_NET",subnet="$GVNIC_SUB" \
  --network-interface=nic-type=GVNIC,network="$GVNIC_NET_1",subnet="$GVNIC_SUB_1",no-address \
  --network-interface=nic-type=MRDMA,network="$RDMA_NET",subnet="$RDMA_SUB_0",no-address \
  --network-interface=nic-type=MRDMA,network="$RDMA_NET",subnet="$RDMA_SUB_1",no-address \
  --network-interface=nic-type=MRDMA,network="$RDMA_NET",subnet="$RDMA_SUB_2",no-address \
  --network-interface=nic-type=MRDMA,network="$RDMA_NET",subnet="$RDMA_SUB_3",no-address \
  --metadata="cp-ip=${CP_VPC_IP},lustre-ip=${LUSTRE_IP},lustre-fs=${LUSTRE_FS},lustre-mount=${LUSTRE_MOUNT},cp-ssh-pubkey=${CP_SSH_PUBKEY},cp-join-cmd=${CP_JOIN_CMD}" \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT"

echo
echo "VM created. Wait ~7-10 min for IMEX reboot + k8s prereqs + auto-join."
echo "Track init log: gx k8n 'ssh maxwellx@${INSTANCE_NAME} \"sudo tail -50 /var/log/forrest-worker-init.log\"'"
echo "Or via IAP:     gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --tunnel-through-iap --command='sudo tail -50 /var/log/forrest-worker-init.log'"
echo "Verify join:    gx k8n k get node ${INSTANCE_NAME} -o wide"
