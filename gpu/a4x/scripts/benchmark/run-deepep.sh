#!/bin/bash
# Run DeepEP v2 test on N pods via dave-style outside launch (kubectl exec parallel)
# 跟 dave_doc_v2/scripts/run-deepep-{2node,128gpu}.sh 一致模式, 不用 master pod ssh barrier
#
# 前提: yaml 已 apply, pod 全 Running (sleep infinity), CD daemon Ready (race fixed)
# 检查: bash scripts/troubleshooting/check-k8s-dra-health.sh && --fix-race
#
# Usage: bash scripts/k8s134/run-deepep.sh <N> <D> [test_name]
#   N: total pod count (1, 2, 4, 18, 36)
#   D: ComputeDomain count (1 = same clique, 2 = cross clique)
#   test_name: test_ep (default) | test_internode | test_intranode | test_low_latency
set -eu

N=${1:?usage: $0 <N> <D> [test_name]}
D=${2:?usage: $0 <N> <D> [test_name]}
TEST=${3:-test_ep}
PER_DOM=$((N / D))
PREFIX="deepep-${N}n"

# pod list (group 1 first, then group 2 if D=2)
PODS=()
for d in $(seq 1 $D); do
  for i in $(seq 0 $((PER_DOM - 1))); do
    PODS+=("${PREFIX}-g${d}-${i}")
  done
done

MASTER_POD="${PREFIX}-g1-0"
MASTER_IP=$(gx k8n "kubectl get pod ${MASTER_POD} -o jsonpath='{.status.podIP}'")
MASTER_PORT=8377
WORLD_SIZE=$N

echo "=== ${PREFIX} dave-style launch ==="
echo "MASTER_POD=${MASTER_POD} MASTER_IP=${MASTER_IP} MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE} (= pod count)"
echo "TEST=${TEST}"
echo "PODS: ${PODS[*]}"
echo

# test cmd
case "$TEST" in
  test_ep)
    CMD="taskset -c 16-139 python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 6 --num-experts 256 --num-sms 64"
    ;;
  test_internode)
    CMD="taskset -c 16-139 python3 tests/legacy/test_internode.py --num-processes 4"
    ;;
  test_intranode)
    CMD="taskset -c 16-139 python3 tests/legacy/test_intranode.py --num-processes 4 --allow-mnnvl"
    ;;
  test_low_latency)
    CMD="taskset -c 16-139 python3 tests/legacy/test_low_latency.py --num-processes 4 --allow-mnnvl"
    ;;
  *)
    echo "unknown test: $TEST"; exit 1
    ;;
esac

# launch all pods in parallel (kubectl exec from master k8s node)
mkdir -p /tmp/deepep-logs
LOG_DIR="/tmp/deepep-logs/${PREFIX}-D${D}-${TEST}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
echo "log dir: $LOG_DIR"
echo

PIDS=()
for i in "${!PODS[@]}"; do
  POD="${PODS[$i]}"
  RANK=$i
  LOG="${LOG_DIR}/rank-${RANK}.log"
  echo "[rank ${RANK}] launching on ${POD} → ${LOG}"

  # nohup kubectl exec, env照 dave 同套 (没 EP_DISABLE_GIN, MNNVL=0)
  nohup gx k8n -t 7200 "kubectl exec ${POD} -- bash -c '
    export LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:/usr/local/gib/lib64:/usr/local/nvidia/lib64
    export EP_NCCL_ROOT_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nccl
    export EP_JIT_CACHE_DIR=/tmp/deepep-jit-cache
    export NCCL_NET=gIB
    export NCCL_PXN_C2C=1
    export NCCL_IB_ADAPTIVE_ROUTING=1
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=52
    export NCCL_IB_FIFO_TC=84
    export NCCL_NVLS_ENABLE=0
    export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-2}      # D=1 production 默认 2 (8x SU 性能), D=2 override 为 0
    export NCCL_CUMEM_ENABLE=1
    export NCCL_IB_GID_INDEX=3
    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    export NVSHMEM_BOOTSTRAP=MPI
    export NVSHMEM_BOOTSTRAP_MPI_PLUGIN=nvshmem_bootstrap_torch.so
    export MASTER_ADDR=${MASTER_IP}
    export MASTER_PORT=${MASTER_PORT}
    export WORLD_SIZE=${WORLD_SIZE}
    export RANK=${RANK}
    cd /opt/DeepEP
    ${CMD}
  '" > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo
echo "=== ${#PIDS[@]} kubectl exec launched (PIDs: ${PIDS[*]}) ==="
echo "tail rank 0: tail -f ${LOG_DIR}/rank-0.log"
echo "wait all: wait ${PIDS[*]}"
echo
echo "watching rank 0 + rank $((N-1)) until completion or fatal..."

# wait all + show rank 0 + last rank tail
wait "${PIDS[@]}" 2>/dev/null || true
echo
echo "=== all kubectl exec returned ==="
echo
echo "=== rank 0 tail 30 ==="
tail -30 "${LOG_DIR}/rank-0.log"
echo
echo "=== rank $((N-1)) tail 15 ==="
tail -15 "${LOG_DIR}/rank-$((N-1)).log"
echo
echo "=== summary: $(grep -c 'Testing with' ${LOG_DIR}/rank-0.log) test_case started / $(grep -c 'PASS\|GB/s.*combine' ${LOG_DIR}/rank-0.log) done markers ==="
echo "all logs: $LOG_DIR"
