#!/bin/bash

# HunyuanVideo-1.5 Diffusers Generation Script
# 使用 Diffusers 库进行视频生成，相比官方实现更简单易用

# 提示词配置
PROMPT='A girl holding a paper with words "Hello, world!"'

# 模型配置
MODEL_ID=hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v  # 720p Text-to-Video 模型
# 其他可用模型：
# hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
# hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v
# hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v

# 生成参数
SEED=42
VIDEO_LENGTH=121  # 推荐值，生成约5秒的高质量视频（121帧 ÷ 24fps = 5.04秒）
NUM_STEPS=50  # 采样步数，推荐50步以获得最佳质量
GUIDANCE_SCALE=6.0  # CFG引导尺度，6.0为标准值，1.0表示关闭CFG
OUTPUT_PATH=./outputs/output.mp4
FPS=24  # 输出视频帧率

# 性能配置
DTYPE=bf16  # 数据类型：bf16（推荐）/ fp16 / fp32
DEVICE=cuda  # 计算设备：cuda / cpu

# 内存优化选项
ENABLE_CPU_OFFLOAD=false  # 启用CPU卸载以节省GPU显存（会降低速度）
#ENABLE_VAE_TILING=true  # 启用VAE分块，减少显存占用（推荐开启）
ENABLE_SEQUENTIAL_OFFLOAD=false  # 启用顺序CPU卸载（最省显存但最慢）

echo "================================================"
echo "HunyuanVideo-1.5 Diffusers 视频生成"
echo "================================================"
echo "模型: $MODEL_ID"
echo "提示词: $PROMPT"
echo "帧数: $VIDEO_LENGTH"
echo "采样步数: $NUM_STEPS"
echo "引导尺度: $GUIDANCE_SCALE"
echo "随机种子: $SEED"
echo "输出路径: $OUTPUT_PATH"
echo "数据类型: $DTYPE"
echo "设备: $DEVICE"
echo "CPU卸载: $ENABLE_CPU_OFFLOAD"
echo "VAE分块: $ENABLE_VAE_TILING"
echo "顺序卸载: $ENABLE_SEQUENTIAL_OFFLOAD"
echo "================================================"
echo ""

# 构建参数
ARGS="--prompt \"$PROMPT\" \
  --model_id $MODEL_ID \
  --num_frames $VIDEO_LENGTH \
  --num_inference_steps $NUM_STEPS \
  --guidance_scale $GUIDANCE_SCALE \
  --seed $SEED \
  --output_path $OUTPUT_PATH \
  --fps $FPS \
  --dtype $DTYPE \
  --device $DEVICE"

# 添加优化选项
if [ "$ENABLE_CPU_OFFLOAD" = "true" ]; then
    ARGS="$ARGS --enable_cpu_offload"
fi

if [ "$ENABLE_VAE_TILING" = "true" ]; then
    ARGS="$ARGS --enable_vae_tiling"
fi

if [ "$ENABLE_SEQUENTIAL_OFFLOAD" = "true" ]; then
    ARGS="$ARGS --enable_sequential_cpu_offload"
fi

# 运行生成
eval python generate_diffusers.py $ARGS

echo ""
echo "================================================"
echo "生成完成！视频已保存到: $OUTPUT_PATH"
echo "================================================"

# End of script