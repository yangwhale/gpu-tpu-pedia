#!/bin/bash
# HunyuanVideo-1.5 阶段2：Transformer (DiT) 测试脚本

# === Stage 2 参数 ===
ASPECT_RATIO=16:9
VIDEO_LENGTH=49
NUM_STEPS=50
SEED=42
GUIDANCE_SCALE=6.0
N_INFERENCE_GPU=8

# I/O 目录
INPUT_DIR=./stage_outputs
OUTPUT_DIR=./stage_outputs

echo "=============================================="
echo "Stage 2: Transformer (${N_INFERENCE_GPU} GPUs)"
echo "=============================================="

~/.local/bin/torchrun --nproc_per_node=$N_INFERENCE_GPU stage2_transformer.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --aspect_ratio $ASPECT_RATIO \
    --video_length $VIDEO_LENGTH \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --seed $SEED

if [ $? -ne 0 ]; then
    echo "Stage 2 failed!"
    exit 1
fi

echo ""
echo "Stage 2 completed successfully!"
echo "Output: $OUTPUT_DIR/stage2_latents.safetensors"