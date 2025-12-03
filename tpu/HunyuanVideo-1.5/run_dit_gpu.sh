#!/bin/bash

# HunyuanVideo-1.5 DiT GPU 性能测试脚本
# 使用方法:
#   单卡: bash run_dit_gpu.sh
#   8卡:  N_GPU=8 bash run_dit_gpu.sh

# 配置参数
N_GPU=${N_GPU:-1}  # GPU数量，默认1，可通过环境变量设置
FRAMES=${FRAMES:-121}  # 帧数，默认121
RESOLUTION=${RESOLUTION:-720p}  # 分辨率，默认720p
NUM_RUNS=${NUM_RUNS:-3}  # 测试次数，默认3
MODEL_PATH=${MODEL_PATH:-/dev/shm/HunyuanVideo-1.5/ckpts}  # 模型路径

echo "================================================"
echo "HunyuanVideo-1.5 DiT GPU 性能测试"
echo "================================================"
echo "GPU 数量: $N_GPU"
echo "帧数: $FRAMES"
echo "分辨率: $RESOLUTION"
echo "测试次数: $NUM_RUNS"
echo "模型路径: $MODEL_PATH"
echo "================================================"
echo ""

if [ "$N_GPU" -eq 1 ]; then
    # 单卡运行
    python dit_gpu.py \
        --frames $FRAMES \
        --resolution $RESOLUTION \
        --num_runs $NUM_RUNS \
        --model_path $MODEL_PATH
else
    # 多卡运行
    torchrun --nproc_per_node=$N_GPU dit_gpu.py \
        --frames $FRAMES \
        --resolution $RESOLUTION \
        --num_runs $NUM_RUNS \
        --model_path $MODEL_PATH
fi