#!/bin/bash
# HunyuanVideo-1.5 阶段2：Transformer (DiT) TPU 版本测试脚本
#
# 用法:
#   ./run_stage2_tpu.sh                    # 默认参数运行
#   ./run_stage2_tpu.sh --video_length 121 # 指定视频帧数
#   ./run_stage2_tpu.sh --num_steps 30     # 指定推理步数

# === Stage 2 TPU 参数 ===
ASPECT_RATIO=16:9
VIDEO_LENGTH=49
NUM_STEPS=50
SEED=42
GUIDANCE_SCALE=6.0

# I/O 目录
INPUT_DIR=./stage_outputs
OUTPUT_DIR=./stage_outputs

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --video_length)
            VIDEO_LENGTH=$2
            shift 2
            ;;
        --num_steps)
            NUM_STEPS=$2
            shift 2
            ;;
        --seed)
            SEED=$2
            shift 2
            ;;
        --guidance_scale)
            GUIDANCE_SCALE=$2
            shift 2
            ;;
        --aspect_ratio)
            ASPECT_RATIO=$2
            shift 2
            ;;
        --input_dir)
            INPUT_DIR=$2
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "视频参数:"
            echo "  --video_length N    视频帧数 (默认: 49)"
            echo "  --num_steps N       推理步数 (默认: 50)"
            echo "  --seed N            随机种子 (默认: 42)"
            echo "  --guidance_scale N  CFG 强度 (默认: 6.0)"
            echo "  --aspect_ratio R    宽高比 (默认: 16:9)"
            echo ""
            echo "其他选项:"
            echo "  --input_dir DIR     输入目录 (默认: ./stage_outputs)"
            echo "  --output_dir DIR    输出目录 (默认: ./stage_outputs)"
            echo "  -h, --help          显示帮助"
            echo ""
            echo "注意: TPU 版本使用 Splash Attention，自动利用所有可用 TPU 核心"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检测 TPU 设备数量
TPU_CHIPS=$(python3 -c "import jax; print(jax.device_count())" 2>/dev/null || echo "未知")

echo "=============================================="
echo "Stage 2: Transformer (TPU + Splash Attention)"
echo "=============================================="
echo "配置:"
echo "  视频帧数: $VIDEO_LENGTH"
echo "  推理步数: $NUM_STEPS"
echo "  CFG 强度: $GUIDANCE_SCALE"
echo "  随机种子: $SEED"
echo "  宽高比: $ASPECT_RATIO"
echo "  TPU 核心数: $TPU_CHIPS"
echo "=============================================="

# 设置 JAX 缓存目录
export JAX_COMPILATION_CACHE_DIR=/dev/shm/jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

# 运行 Flax 版本的 Stage 2
python3 stage2_transformer_flax.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --aspect_ratio $ASPECT_RATIO \
    --video_length $VIDEO_LENGTH \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --seed $SEED

if [ $? -ne 0 ]; then
    echo "Stage 2 (TPU) failed!"
    exit 1
fi

echo ""
echo "Stage 2 (TPU) completed successfully!"
echo "Output: $OUTPUT_DIR/stage2_latents.safetensors"