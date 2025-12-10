#!/bin/bash
# HunyuanVideo-1.5 阶段3：VAE Decoder 测试脚本
# 使用多卡并行 tiling 进行 VAE 解码

# === Stage 3 参数 ===
FPS=24
OUTPUT_VIDEO=./stage_outputs/output_video.mp4
N_INFERENCE_GPU=8

# I/O 目录
INPUT_DIR=./stage_outputs

echo "=============================================="
echo "Stage 3: VAE Decoder (${N_INFERENCE_GPU} GPUs with tile parallelism)"
echo "=============================================="

# 使用 torchrun 启动以利用多卡并行 tiling
~/.local/bin/torchrun --nproc_per_node=$N_INFERENCE_GPU stage3_vae_decoder.py \
    --input_dir $INPUT_DIR \
    --output_video $OUTPUT_VIDEO \
    --fps $FPS

if [ $? -ne 0 ]; then
    echo "Stage 3 failed!"
    exit 1
fi

echo ""
echo "Stage 3 completed successfully!"
echo "Output video: $OUTPUT_VIDEO"