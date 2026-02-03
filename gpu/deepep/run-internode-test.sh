#!/bin/bash
# DeepEP Internode 测试脚本
# 用法: ./run-internode-test.sh <master_ip> <rank> [world_size]
#
# 示例 (2节点测试):
#   节点1 (master): ./run-internode-test.sh 10.8.0.14 0 2
#   节点2 (worker): ./run-internode-test.sh 10.8.0.14 1 2
#
# 注意: 两个节点需要几乎同时启动

set -e

MASTER_ADDR=${1:-""}
RANK=${2:-""}
WORLD_SIZE=${3:-2}
MASTER_PORT=${MASTER_PORT:-29500}

# 测试参数
NUM_TOKENS=${NUM_TOKENS:-2048}
HIDDEN=${HIDDEN:-7168}
NUM_EXPERTS=${NUM_EXPERTS:-256}
NUM_TOPK=${NUM_TOPK:-8}

if [ -z "$MASTER_ADDR" ] || [ -z "$RANK" ]; then
    echo "Usage: $0 <master_ip> <rank> [world_size]"
    echo ""
    echo "Examples:"
    echo "  Node 1 (master): $0 10.8.0.14 0 2"
    echo "  Node 2 (worker): $0 10.8.0.14 1 2"
    echo ""
    echo "Environment variables:"
    echo "  MASTER_PORT  - Master port (default: 29500)"
    echo "  NUM_TOKENS   - Number of tokens (default: 2048)"
    echo "  HIDDEN       - Hidden size (default: 7168)"
    echo "  NUM_EXPERTS  - Number of experts (default: 256)"
    echo "  NUM_TOPK     - Top-k experts (default: 8)"
    exit 1
fi

echo "=========================================="
echo "DeepEP Internode Test"
echo "=========================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE:  $WORLD_SIZE"
echo "RANK:        $RANK"
echo "=========================================="
echo "Test parameters:"
echo "  num_tokens:  $NUM_TOKENS"
echo "  hidden:      $HIDDEN"
echo "  num_experts: $NUM_EXPERTS"
echo "  num_topk:    $NUM_TOPK"
echo "=========================================="

# 加载环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/unified-env.sh" ]; then
    source "$SCRIPT_DIR/unified-env.sh"
elif [ -f "/opt/deepep/unified-env.sh" ]; then
    source /opt/deepep/unified-env.sh
else
    echo "Error: unified-env.sh not found"
    exit 1
fi

# 验证环境
echo ""
echo "Checking environment..."
echo "  HCA devices: $(ls /sys/class/infiniband/ | tr '\n' ' ')"
echo "  PeerMappingOverride: $(grep PeerMappingOverride /proc/driver/nvidia/params 2>/dev/null || echo 'not set')"

# 验证 DeepEP
python3 -c "import deep_ep; print('  DeepEP:', deep_ep.__file__)"

# 设置分布式环境
export WORLD_SIZE=$WORLD_SIZE
export RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# 运行测试
echo ""
echo "Starting test..."
cd /opt/deepep/DeepEP/tests
python3 test_internode.py \
    --num-tokens $NUM_TOKENS \
    --hidden $HIDDEN \
    --num-experts $NUM_EXPERTS \
    --num-topk $NUM_TOPK
