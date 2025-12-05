#!/bin/bash
# HunyuanVideo-1.5 阶段1：Text Encoder 测试脚本

# === Stage 1 参数 ===
PROMPT='A girl holding a paper with words "Hello, world!"'
NEGATIVE_PROMPT=""
RESOLUTION=720p
MODEL_PATH=/dev/shm/HunyuanVideo-1.5/ckpts
OUTPUT_DIR=./stage_outputs

echo "=============================================="
echo "Stage 1: Text Encoder"
echo "=============================================="

python stage1_text_encoder.py \
    --model_path $MODEL_PATH \
    --prompt "$PROMPT" \
    --negative_prompt "$NEGATIVE_PROMPT" \
    --resolution $RESOLUTION \
    --output_dir $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "Stage 1 failed!"
    exit 1
fi

echo ""
echo "Stage 1 completed successfully!"
echo "Output: $OUTPUT_DIR/stage1_embeddings.safetensors"