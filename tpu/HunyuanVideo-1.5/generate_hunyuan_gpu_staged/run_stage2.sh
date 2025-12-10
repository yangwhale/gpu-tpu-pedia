#!/bin/bash
# HunyuanVideo-1.5 阶段2：Transformer (DiT) 测试脚本
#
# 用法:
#   ./run_stage2.sh                    # 默认 flash attention
#   ./run_stage2.sh --use_sageattn     # 使用 SageAttention (快但质量略差)
#   ./run_stage2.sh --sparse_attn      # 使用 Sparse Attention (仅 H100)
#   ./run_stage2.sh --attn_mode torch  # 使用 PyTorch SDPA
#   ./run_stage2.sh --enable_cache     # 启用 DeepCache 加速

# === Stage 2 参数 ===
ASPECT_RATIO=16:9
VIDEO_LENGTH=121
NUM_STEPS=50
SEED=42
GUIDANCE_SCALE=6.0
N_INFERENCE_GPU=8

# Attention 模式参数
# 可选: flash (默认), flash2, flash3, torch, sageattn, flex-block-attn
ATTN_MODE=flash

# Cache 加速参数
ENABLE_CACHE=false
CACHE_TYPE=deepcache
CACHE_START_STEP=11
CACHE_END_STEP=45
CACHE_STEP_INTERVAL=4

# I/O 目录
INPUT_DIR=./stage_outputs
OUTPUT_DIR=./stage_outputs

# 解析命令行参数
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_sageattn)
            EXTRA_ARGS="$EXTRA_ARGS --use_sageattn"
            echo "⚡ 启用 SageAttention (加速模式，质量可能略有下降)"
            shift
            ;;
        --sparse_attn)
            EXTRA_ARGS="$EXTRA_ARGS --sparse_attn"
            echo "⚡ 启用 Sparse Attention (需要 H100 GPU)"
            shift
            ;;
        --attn_mode)
            ATTN_MODE=$2
            echo "ℹ️ Attention 模式: $ATTN_MODE"
            shift 2
            ;;
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
        --gpus)
            N_INFERENCE_GPU=$2
            shift 2
            ;;
        --enable_cache)
            ENABLE_CACHE=true
            EXTRA_ARGS="$EXTRA_ARGS --enable_cache"
            echo "⚡ 启用 Cache 加速 (DeepCache)"
            shift
            ;;
        --cache_type)
            CACHE_TYPE=$2
            shift 2
            ;;
        --cache_start_step)
            CACHE_START_STEP=$2
            shift 2
            ;;
        --cache_end_step)
            CACHE_END_STEP=$2
            shift 2
            ;;
        --cache_step_interval)
            CACHE_STEP_INTERVAL=$2
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "Attention 模式选项 (互斥，选其一):"
            echo "  --attn_mode MODE    设置 attention 模式 (flash/flash2/flash3/torch/sageattn/flex-block-attn)"
            echo "  --use_sageattn      使用 SageAttention 加速 (~1.6x，质量略有下降)"
            echo "  --sparse_attn       使用 Sparse Attention (需要 H100 + distilled 模型)"
            echo ""
            echo "视频参数:"
            echo "  --video_length N    视频帧数 (默认: 121)"
            echo "  --num_steps N       推理步数 (默认: 50)"
            echo "  --seed N            随机种子 (默认: 42)"
            echo "  --guidance_scale N  CFG 强度 (默认: 6.0)"
            echo "  --aspect_ratio R    宽高比 (默认: 16:9)"
            echo ""
            echo "Cache 加速选项:"
            echo "  --enable_cache          启用 Cache 加速 (DeepCache/TeaCache)"
            echo "  --cache_type TYPE       Cache 类型 (deepcache/teacache, 默认: deepcache)"
            echo "  --cache_start_step N    开始使用 cache 的步数 (默认: 11)"
            echo "  --cache_end_step N      停止使用 cache 的步数 (默认: 45)"
            echo "  --cache_step_interval N Cache 步长间隔 (默认: 4)"
            echo ""
            echo "其他选项:"
            echo "  --input_dir DIR     输入目录 (默认: ./stage_outputs)"
            echo "  --output_dir DIR    输出目录 (默认: ./stage_outputs)"
            echo "  --gpus N            使用的 GPU 数量 (默认: 8)"
            echo "  -h, --help          显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Stage 2: Transformer (${N_INFERENCE_GPU} GPUs)"
echo "=============================================="
echo "配置:"
echo "  视频帧数: $VIDEO_LENGTH"
echo "  推理步数: $NUM_STEPS"
echo "  Attention 模式: $ATTN_MODE"
echo "  CFG 强度: $GUIDANCE_SCALE"
echo "  随机种子: $SEED"
if [ "$ENABLE_CACHE" = true ]; then
    echo "  Cache: $CACHE_TYPE (步骤 $CACHE_START_STEP-$CACHE_END_STEP, 间隔 $CACHE_STEP_INTERVAL)"
fi
echo "=============================================="

# 构建 cache 参数
CACHE_ARGS=""
if [ "$ENABLE_CACHE" = true ]; then
    CACHE_ARGS="--cache_type $CACHE_TYPE --cache_start_step $CACHE_START_STEP --cache_end_step $CACHE_END_STEP --cache_step_interval $CACHE_STEP_INTERVAL"
fi

~/.local/bin/torchrun --nproc_per_node=$N_INFERENCE_GPU stage2_transformer.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --aspect_ratio $ASPECT_RATIO \
    --video_length $VIDEO_LENGTH \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --seed $SEED \
    --attn_mode $ATTN_MODE \
    $CACHE_ARGS \
    $EXTRA_ARGS

if [ $? -ne 0 ]; then
    echo "Stage 2 failed!"
    exit 1
fi

echo ""
echo "Stage 2 completed successfully!"
echo "Output: $OUTPUT_DIR/stage2_latents.safetensors"