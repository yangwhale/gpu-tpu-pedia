#!/bin/bash
# DeepEP Intranode 测试脚本 (单节点 8 GPU)
# 用法: ./run-intranode-test.sh

set -e

# 测试参数
NUM_TOKENS=${NUM_TOKENS:-2048}
HIDDEN=${HIDDEN:-7168}
NUM_EXPERTS=${NUM_EXPERTS:-256}
NUM_TOPK=${NUM_TOPK:-8}

echo "=========================================="
echo "DeepEP Intranode Test (Single Node)"
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
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

# 验证 DeepEP
python3 -c "import deep_ep; print('  DeepEP:', deep_ep.__file__)"

# 运行测试
echo ""
echo "Starting test..."
cd /opt/deepep/DeepEP/tests
python3 test_intranode.py \
    --num-tokens $NUM_TOKENS \
    --hidden $HIDDEN \
    --num-experts $NUM_EXPERTS \
    --num-topk $NUM_TOPK
