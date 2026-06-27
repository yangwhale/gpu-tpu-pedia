#!/bin/bash
# run-nccl-test.sh — NCCL 4 场景 wrapper (DRANET 路径)
# 在 master 本地跑 (ansible push 即可, 不需要从 cloudtop SSH)
# 用法:
#   bash run-nccl-test.sh single                    # 单节点 4 GPU
#   bash run-nccl-test.sh same-domain               # 同域 2 节点 (~836 GB/s)
#   bash run-nccl-test.sh cross-domain              # 跨域 2 节点 (~328 GB/s)
#   bash run-nccl-test.sh mixed                     # 4 节点 mixed (~690 GB/s)
#   bash run-nccl-test.sh <scope> all_gather 1M 8G  # 自定 collective / size range
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../host/env.sh"

scope="${1:-single}"
collective="${2:-all_reduce}"
begin="${3:-512M}"
end="${4:-8G}"

case "$scope" in
  single)        hosts=("nccl-single-node") ;;
  same-domain)   hosts=("nccl-sd-host-1" "nccl-sd-host-2") ;;
  cross-domain)  hosts=("nccl-cd-host-1" "nccl-cd-host-2") ;;
  mixed)         hosts=("nccl-mix-host-1" "nccl-mix-host-2" "nccl-mix-host-3" "nccl-mix-host-4") ;;
  *) echo "Usage: $0 <single|same-domain|cross-domain|mixed> [collective] [begin] [end]"; exit 1 ;;
esac

KCTL="sudo kubectl --kubeconfig=/etc/kubernetes/admin.conf"

# 1) 等所有 pod Ready
$KCTL wait --for=condition=Ready pod ${hosts[*]} --timeout=300s

# 2) 单节点直接跑;多节点要 SSH 互通 + MPI
if [ "$scope" = "single" ]; then
  $KCTL exec ${hosts[0]} -- bash -c '
    source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
    /usr/local/bin/'"${collective}"'_perf -b '"${begin}"' -e '"${end}"' -f 2 -g 4
  ' | tail -25
  exit 0
fi

# 3) 多节点: 交换 ed25519 SSH 密钥 (容器内必须 ed25519,RSA passphrase 会卡 /dev/tty)
echo "Exchanging ed25519 SSH keys across ${#hosts[@]} pods..."
for h in "${hosts[@]}"; do
  $KCTL exec $h -- bash -c '
    [ -f /root/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 -q
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
  '
done
for h_src in "${hosts[@]}"; do
  PUB=$($KCTL exec $h_src -- cat /root/.ssh/id_ed25519.pub)
  for h_dst in "${hosts[@]}"; do
    $KCTL exec $h_dst -- bash -c "echo '$PUB' >> /root/.ssh/authorized_keys"
  done
done

# 4) 拿 IP (DRANET 路径用 podIP)
HOST_IPS=()
for h in "${hosts[@]}"; do
  IP=$($KCTL get pod $h -o jsonpath='{.status.podIP}')
  HOST_IPS+=("${IP}:4")
done
HOST_LIST=$(IFS=','; echo "${HOST_IPS[*]}")
NP=$((${#hosts[@]} * 4))

# 5) MNNVL flag: same-domain/mixed=on(2), cross-domain=off(0)
case "$scope" in
  same-domain|mixed) MNNVL_FLAG="-x NCCL_MNNVL_ENABLE=2" ;;
  cross-domain)      MNNVL_FLAG="-x NCCL_MNNVL_ENABLE=0" ;;
esac

echo "Running mpirun -np $NP --host $HOST_LIST -t $collective -b $begin -e $end ..."
$KCTL exec ${hosts[0]} -- bash -c '
  source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
  /usr/local/mpi/bin/mpirun --allow-run-as-root \
    -np '"$NP"' -npernode 4 \
    --host '"$HOST_LIST"' \
    -x LD_LIBRARY_PATH '"$MNNVL_FLAG"' -x NCCL_CUMEM_ENABLE=1 \
    --mca plm_rsh_args "-p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no" \
    /tmp/all_reduce_perf_mpi -b '"$begin"' -e '"$end"' -f 2 -g 1
' | tail -25
