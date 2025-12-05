#!/bin/bash
# HunyuanVideo-1.5 三阶段分离生成脚本
# 依次运行三个阶段

set -e  # 遇到错误立即退出

echo "=============================================="
echo "HunyuanVideo-1.5 Staged Pipeline"
echo "=============================================="

# Stage 1
bash run_stage1.sh
if [ $? -ne 0 ]; then exit 1; fi

echo ""

# Stage 2
bash run_stage2.sh
if [ $? -ne 0 ]; then exit 1; fi

echo ""

# Stage 3
bash run_stage3.sh
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "=============================================="
echo "All stages completed successfully!"
echo "=============================================="