#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# Pai-Megatron-Patch Qwen3-Next MFU 测试脚本
# 
# 目标: 在 8 层模型配置下，通过调整参数找到 B200 最高 MFU
#
# 可调参数:
# - MBS: Micro Batch Size (每GPU每步样本数)
# - GBS: Global Batch Size (总批次大小)
# - SEQ_LEN: 序列长度
#
# 测试配置: (通过环境变量 MFU_TEST_CONFIG 选择)
# - config1: MBS=1, GBS=8, SEQ=1024 (基线，已测试 MFU=1.1%)
# - config2: MBS=2, GBS=16, SEQ=1024 (已测试 MFU=1.33%)
# - config3: MBS=4, GBS=32, SEQ=1024 (已测试 MFU=1.44%)
# - config4: MBS=2, GBS=16, SEQ=2048 (已测试 MFU=2.44%)
# - config5: MBS=4, GBS=32, SEQ=2048 (已测试 MFU=2.76%)
# - config6: MBS=1, GBS=8, SEQ=4096 (已测试 MFU=3.91%)
# - config_max: MBS=6, GBS=48, SEQ=4096 (已测试 MFU=5.26%)
#
# 高MFU配置 (目标 20%+):
# - config_24L: 24层, MBS=4, GBS=32, SEQ=4096 (增加计算密度)
# - config_48L: 48层, MBS=2, GBS=16, SEQ=4096 (最大化层数)
# - config_extreme: 8层, MBS=16, GBS=128, SEQ=8192 (极限批次)
# =============================================================================

set -e

echo "=============================================="
echo "Pai-Megatron Qwen3-Next MFU Test"
echo "=============================================="

# 选择测试配置
TEST_CONFIG=${MFU_TEST_CONFIG:-"config2"}

# 默认参数 (可被 case 语句覆盖)
NUM_LAYERS=8
TP_SIZE=1  # Tensor Parallel

case $TEST_CONFIG in
  "config1")
    MBS=1; GBS=8; SEQ_LEN=1024; PAD_LEN=1024
    echo "Config 1: MBS=1, GBS=8, SEQ=1024 (baseline)"
    ;;
  "config2")
    MBS=2; GBS=16; SEQ_LEN=1024; PAD_LEN=1024
    echo "Config 2: MBS=2, GBS=16, SEQ=1024 (larger batch)"
    ;;
  "config3")
    MBS=4; GBS=32; SEQ_LEN=1024; PAD_LEN=1024
    echo "Config 3: MBS=4, GBS=32, SEQ=1024 (even larger batch)"
    ;;
  "config4")
    MBS=2; GBS=16; SEQ_LEN=2048; PAD_LEN=2048
    echo "Config 4: MBS=2, GBS=16, SEQ=2048 (longer sequence)"
    ;;
  "config5")
    MBS=4; GBS=32; SEQ_LEN=2048; PAD_LEN=2048
    echo "Config 5: MBS=4, GBS=32, SEQ=2048 (large batch + long seq)"
    ;;
  "config6")
    MBS=1; GBS=8; SEQ_LEN=4096; PAD_LEN=4096
    echo "Config 6: MBS=1, GBS=8, SEQ=4096 (very long sequence)"
    ;;
  "config7")
    MBS=2; GBS=16; SEQ_LEN=4096; PAD_LEN=4096
    echo "Config 7: MBS=2, GBS=16, SEQ=4096 (long seq + larger batch)"
    ;;
  "config8")
    MBS=4; GBS=32; SEQ_LEN=4096; PAD_LEN=4096
    echo "Config 8: MBS=4, GBS=32, SEQ=4096 (long seq + large batch)"
    ;;
  "config9")
    MBS=8; GBS=64; SEQ_LEN=2048; PAD_LEN=2048
    echo "Config 9: MBS=8, GBS=64, SEQ=2048 (very large batch)"
    ;;
  "config_max")
    # 显存分析: B200=180GB, Config6用22GB (MBS=1,SEQ=4096)
    # 激活内存随 MBS 线性增长，模型参数固定约15GB
    # MBS=6: 估算 15GB + 7GB*6 ≈ 57GB，安全
    MBS=6; GBS=48; SEQ_LEN=4096; PAD_LEN=4096
    echo "Config MAX: MBS=6, GBS=48, SEQ=4096 (memory-optimized for B200 180GB)"
    ;;
  "config_24L")
    # 24层配置: 增加3倍层数，提升计算密度
    # 显存估算: 24.5GB × 3 ≈ 73GB (安全)
    MBS=4; GBS=32; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=24  # 覆盖默认8层
    echo "Config 24L: 24 layers, MBS=4, GBS=32, SEQ=4096"
    ;;
  "config_48L")
    # 48层配置: 接近满层数的一半
    # 显存估算: 24.5GB × 6 ≈ 147GB (接近极限)
    # 结果: 峰值 58.9 TFLOP/s, MFU=2.62% (MBS太小导致效率低)
    MBS=2; GBS=16; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=48
    echo "Config 48L: 48 layers, MBS=2, GBS=16, SEQ=4096"
    ;;
  "config_48L_v2")
    # 48层配置v2: 增加MBS和GBS提升效率
    # 目标: 保持高层数的同时提高并行效率
    MBS=4; GBS=32; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=48
    echo "Config 48L_v2: 48 layers, MBS=4, GBS=32, SEQ=4096"
    ;;
  "config_extreme")
    # 极限批次配置: 保持8层，最大化批次
    # 显存估算: 需要测试
    MBS=16; GBS=128; SEQ_LEN=8192; PAD_LEN=8192
    echo "Config EXTREME: MBS=16, GBS=128, SEQ=8192"
    ;;
  "config_tp8")
    # TP=8 配置: 8 GPU 张量并行, 大 batch, 高层数
    # TP=8 意味着模型切分到 8 个 GPU 上，减少每 GPU 显存但增加通信
    MBS=8; GBS=64; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=48
    TP_SIZE=8
    echo "Config TP8: TP=8, 48 layers, MBS=8, GBS=64, SEQ=4096"
    ;;
  "config_tp8_v2")
    # TP=8 配置v2: 更大 batch
    MBS=16; GBS=128; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=48
    TP_SIZE=8
    echo "Config TP8_v2: TP=8, 48 layers, MBS=16, GBS=128, SEQ=4096"
    ;;
  "config_full_mem")
    # 填满显存配置: 48层, 最大化 MBS 和 GBS
    # 当前 48L 用 95GB (MBS=4, GBS=32)
    # 目标: 使用 160-170GB
    # 策略: MBS=8, GBS=64 (约 2x 激活内存)
    MBS=8; GBS=64; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=48
    TP_SIZE=1
    echo "Config FULL_MEM: 48 layers, MBS=8, GBS=64, SEQ=4096 (target: 160GB+)"
    ;;
  "config_full_mem_v2")
    # 更激进的显存配置
    # MBS=12, GBS=96
    MBS=12; GBS=96; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=48
    TP_SIZE=1
    echo "Config FULL_MEM_v2: 48 layers, MBS=12, GBS=96, SEQ=4096 (aggressive)"
    ;;
  "config_full_mem_v3")
    # 极限显存配置
    # MBS=16, GBS=128
    MBS=16; GBS=128; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=48
    TP_SIZE=1
    echo "Config FULL_MEM_v3: 48 layers, MBS=16, GBS=128, SEQ=4096 (max batch)"
    ;;
  "config_seq8k")
    # 长序列配置: 增加序列长度到 8192
    # 更长序列 -> 更高计算密度
    MBS=1; GBS=8; SEQ_LEN=8192; PAD_LEN=8192
    NUM_LAYERS=48
    TP_SIZE=1
    echo "Config SEQ8K: 48 layers, MBS=1, GBS=8, SEQ=8192 (long sequence)"
    ;;
  "config_seq8k_v2")
    # 长序列 + 大批次
    # 结果: 稳定 85-88 TFLOP/s (MFU 3.78-3.91%), 峰值 106.1
    MBS=2; GBS=16; SEQ_LEN=8192; PAD_LEN=8192
    NUM_LAYERS=48
    TP_SIZE=1
    echo "Config SEQ8K_v2: 48 layers, MBS=2, GBS=16, SEQ=8192"
    ;;
  "config_seq8k_v3")
    # 长序列 + 更大批次
    # 目标: 使用更多显存提升 MFU
    MBS=4; GBS=32; SEQ_LEN=8192; PAD_LEN=8192
    NUM_LAYERS=48
    TP_SIZE=1
    echo "Config SEQ8K_v3: 48 layers, MBS=4, GBS=32, SEQ=8192"
    ;;
  "config_seq8k_v4")
    # 长序列 + 极限批次 (激进配置)
    MBS=8; GBS=64; SEQ_LEN=8192; PAD_LEN=8192
    NUM_LAYERS=48
    TP_SIZE=1
    echo "Config SEQ8K_v4: 48 layers, MBS=8, GBS=64, SEQ=8192 (aggressive)"
    ;;
  "config_96L")
    # 完整 96 层模型
    MBS=1; GBS=8; SEQ_LEN=4096; PAD_LEN=4096
    NUM_LAYERS=96
    TP_SIZE=1
    echo "Config 96L: 96 layers (full model), MBS=1, GBS=8, SEQ=4096"
    ;;
  *)
    echo "Unknown config: $TEST_CONFIG"
    echo "Valid configs: config1-config9, config_max, config_24L, config_48L, config_tp8, config_full_mem, config_seq8k, config_96L"
    exit 1
    ;;
esac

# =============================================================================
# 配置参数
# =============================================================================

PAI_WORKSPACE=${PAI_WORKSPACE_ROOT:-/mnt}
PATCH_DIR=${PAI_WORKSPACE}/Pai-Megatron-Patch
CKPT_DIR=${PAI_WORKSPACE}/ckpts
HF_CKPT_DIR=${CKPT_DIR}/huggingface
MCORE_CKPT_DIR=${CKPT_DIR}/mcore
DATASET_DIR=${PAI_WORKSPACE}/datasets

MODEL_NAME="Qwen3-Next-80B-A3B-Instruct"
MODEL_SIZE="A3B"

# JobSet 同步目录
JOB_NAME="${HOSTNAME_PREFIX%-workload-}"
SYNC_DIR="$PAI_WORKSPACE/sync_flags_${JOB_NAME}"

# 检查点精度
CKPT_PRECISION="bf16"

# 训练迭代数 (固定 100 步快速测试)
TRAIN_ITERS=100
WARMUP_ITERS=0

# =============================================================================
# Step 0: 跳过已完成的准备步骤
# =============================================================================
SKIP_CLONE_REPO=true
SKIP_DOWNLOAD_DATA=true
SKIP_CHECKPOINT_CONVERSION=true

echo ""
echo "Using pre-existing data and checkpoints..."
echo "  PATCH_DIR: $PATCH_DIR"
echo "  MCORE_CKPT_DIR: $MCORE_CKPT_DIR/${MODEL_NAME}-to-mcore-${CKPT_PRECISION}"

# =============================================================================
# Step 1: 训练
# =============================================================================
echo ""
echo "Starting MFU test training..."
echo "  NUM_LAYERS: $NUM_LAYERS"
echo "  MBS: $MBS, GBS: $GBS, SEQ_LEN: $SEQ_LEN"
echo "  TRAIN_ITERS: $TRAIN_ITERS"

cd ${PATCH_DIR}/examples/qwen3_next

# =========================================================================
# B200 兼容性 Patch (如果还没打过)
# =========================================================================
if ! grep -q 'MP_NUM_LAYERS' run_mcore_qwen3.sh; then
  sed -i 's/NUM_LAYERS=96/NUM_LAYERS=${MP_NUM_LAYERS:-96}/' run_mcore_qwen3.sh
  echo "  [Patch 1] Dynamic NUM_LAYERS enabled"
fi

if ! grep -q 'PATTERN_UNIT=' run_mcore_qwen3.sh; then
  sed -i '/HYBRID_TRANSFORMER_RATIO=/a \
PATTERN_UNIT="M-M-M-*-"\
PATTERN_REPEATS=$((NUM_LAYERS / 8))\
HYBRID_MAMBA_TRANSFORMER_PATTERN=$(printf "%s" $(yes "$PATTERN_UNIT" | head -n $PATTERN_REPEATS | tr -d "\\n"))' run_mcore_qwen3.sh
  echo "  [Patch 2] Dynamic Hybrid Pattern enabled"
fi

# =========================================================================
# 额外 Patch: 支持自定义 MBS, GBS, SEQ_LEN, TRAIN_ITERS
# =========================================================================
# Patch 3: 支持 MICRO_BATCH_SIZE 覆盖
if ! grep -q 'MP_MICRO_BATCH_SIZE' run_mcore_qwen3.sh; then
  sed -i 's/MICRO_BATCH_SIZE=$3/MICRO_BATCH_SIZE=${MP_MICRO_BATCH_SIZE:-$3}/' run_mcore_qwen3.sh
  echo "  [Patch 3] Dynamic MICRO_BATCH_SIZE enabled"
fi

# Patch 4: 支持 GLOBAL_BATCH_SIZE 覆盖
if ! grep -q 'MP_GLOBAL_BATCH_SIZE' run_mcore_qwen3.sh; then
  sed -i 's/GLOBAL_BATCH_SIZE=$4/GLOBAL_BATCH_SIZE=${MP_GLOBAL_BATCH_SIZE:-$4}/' run_mcore_qwen3.sh
  echo "  [Patch 4] Dynamic GLOBAL_BATCH_SIZE enabled"
fi

# Patch 5: 支持 SEQ_LEN 覆盖
if ! grep -q 'MP_SEQ_LEN' run_mcore_qwen3.sh; then
  sed -i 's/SEQ_LEN=$7/SEQ_LEN=${MP_SEQ_LEN:-$7}/' run_mcore_qwen3.sh
  echo "  [Patch 5] Dynamic SEQ_LEN enabled"
fi

# Patch 6: 支持 PAD_LEN 覆盖
if ! grep -q 'MP_PAD_LEN' run_mcore_qwen3.sh; then
  sed -i 's/PAD_LEN=$8/PAD_LEN=${MP_PAD_LEN:-$8}/' run_mcore_qwen3.sh
  echo "  [Patch 6] Dynamic PAD_LEN enabled"
fi

# Patch 7: 支持 TP_SIZE 覆盖
if ! grep -q 'MP_TP_SIZE' run_mcore_qwen3.sh; then
  sed -i 's/TP=$10/TP=${MP_TP_SIZE:-$10}/' run_mcore_qwen3.sh
  echo "  [Patch 7] Dynamic TP_SIZE enabled"
fi

# =========================================================================
# 配置输出目录
# =========================================================================
OUTPUT_DIR="${PAI_WORKSPACE}/logs/mfu_test_${TEST_CONFIG}"
mkdir -p $OUTPUT_DIR

MCORE_CKPT_PATH="${MCORE_CKPT_DIR}/${MODEL_NAME}-to-mcore-${CKPT_PRECISION}"
DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_text_document"
VALID_DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_text_document"

echo ""
echo "  Output: $OUTPUT_DIR"
echo "  Nodes: $NNODES, GPUs/Node: $GPUS_PER_NODE, Total: $((NNODES * GPUS_PER_NODE))"

# =========================================================================
# 计算 Token 数量 (基于迭代次数)
# TRAIN_TOKENS = TRAIN_ITERS * GBS * SEQ_LEN
# =========================================================================
TRAIN_TOKENS=$((TRAIN_ITERS * GBS * SEQ_LEN))
WARMUP_TOKENS=$((WARMUP_ITERS * GBS * SEQ_LEN))

echo "  TRAIN_TOKENS: $TRAIN_TOKENS"

# =========================================================================
# 运行训练
# =========================================================================
echo "  TP_SIZE: $TP_SIZE"

OMP_NUM_THREADS=12 \
NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
MP_NUM_LAYERS=$NUM_LAYERS \
MP_MICRO_BATCH_SIZE=$MBS \
MP_GLOBAL_BATCH_SIZE=$GBS \
MP_SEQ_LEN=$SEQ_LEN \
MP_PAD_LEN=$PAD_LEN \
MP_TP_SIZE=$TP_SIZE \
WORLD_SIZE=$NNODES \
RANK=$JOB_COMPLETION_INDEX \
KUBERNETES_CONTAINER_RESOURCE_GPU=$GPUS_PER_NODE \
bash run_mcore_qwen3.sh \
  dlc $MODEL_SIZE 1 8 1e-5 1e-6 1024 1024 ${CKPT_PRECISION} \
  1 1 1 1 8 true true true false none false 100000 \
  $DATASET_PATH $VALID_DATASET_PATH \
  $MCORE_CKPT_PATH \
  $TRAIN_TOKENS $WARMUP_TOKENS $OUTPUT_DIR

echo ""
echo "=============================================="
echo "MFU Test ($TEST_CONFIG) completed!"
echo "=============================================="
