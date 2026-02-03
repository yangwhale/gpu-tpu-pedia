#!/bin/bash
# 从本地机器远程并行运行 DeepEP Internode 测试
# 用法: ./run-internode-test-remote.sh <host1> <host2> [master_ip]
#
# 示例:
#   ./run-internode-test-remote.sh b7 b8
#   ./run-internode-test-remote.sh b7 b8 10.8.0.14

set -e

HOST1=${1:-""}
HOST2=${2:-""}
MASTER_IP=${3:-""}

if [ -z "$HOST1" ] || [ -z "$HOST2" ]; then
    echo "Usage: $0 <host1> <host2> [master_ip]"
    echo ""
    echo "Examples:"
    echo "  $0 b7 b8              # Auto-detect master IP from SSH config"
    echo "  $0 b7 b8 10.8.0.14    # Specify master IP"
    exit 1
fi

# 如果没有指定 master IP，从 SSH config 获取
if [ -z "$MASTER_IP" ]; then
    MASTER_IP=$(grep -A1 "Host $HOST1" ~/.ssh/config | grep HostName | awk '{print $2}')
    if [ -z "$MASTER_IP" ]; then
        echo "Error: Could not find HostName for $HOST1 in ~/.ssh/config"
        echo "Please specify master_ip manually"
        exit 1
    fi
fi

echo "=========================================="
echo "DeepEP Internode Test (Remote)"
echo "=========================================="
echo "Host 1 (Master): $HOST1"
echo "Host 2 (Worker): $HOST2"
echo "Master IP:       $MASTER_IP"
echo "=========================================="

# 测试脚本命令
TEST_CMD='source /opt/deepep/unified-env.sh && cd /opt/deepep/DeepEP/tests && python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8'

# 启动 worker (后台)
echo "Starting worker on $HOST2..."
ssh $HOST2 "export WORLD_SIZE=2 RANK=1 MASTER_ADDR=$MASTER_IP MASTER_PORT=29500 && $TEST_CMD" 2>&1 &
WORKER_PID=$!

# 等待 worker 启动
sleep 2

# 启动 master
echo "Starting master on $HOST1..."
ssh $HOST1 "export WORLD_SIZE=2 RANK=0 MASTER_ADDR=$MASTER_IP MASTER_PORT=29500 && $TEST_CMD" 2>&1

# 等待 worker 完成
wait $WORKER_PID

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
