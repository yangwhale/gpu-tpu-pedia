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
# Pai-Megatron-Patch Qwen3-Next 训练脚本
# 基于 PyTorch 25.12 镜像，支持 B200/SM100 GPU
#
# 模型: Qwen3-Next-80B-A3B-Instruct (Mamba + Transformer + MoE, 512 experts)
# 推荐配置: 2 节点 x 8 GPU = 16 GPU (EP=8, DP=2)
#
# 依赖说明:
# - NCCL: 由 GIB 在运行时提供优化版本
# - triton, mamba-ssm, causal-conv1d: 已在 Docker 镜像中预装
# - TransformerEngine: PyTorch 25.12 内置完整 B200 支持
#
# B200 调试成果 (2026-01-12):
# 1. Flash Attention 启用: Fused Attention 不支持 kv_channels=256，使用 FL=true
# 2. 动态层数支持: MP_NUM_LAYERS 环境变量控制层数 (默认 96，调试用 8)
# 3. Hybrid Pattern 动态生成: Pattern 单元 M-M-M-*- 按层数重复
# =============================================================================

set -e

echo "=============================================="
echo "Pai-Megatron-Patch Qwen3-Next Training Pipeline"
echo "=============================================="

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
TRAINING_MODE=${TRAINING_MODE:-"pretrain"}

# JobSet 同步目录
JOB_NAME="${HOSTNAME_PREFIX%-workload-}"
SYNC_DIR="$PAI_WORKSPACE/sync_flags_${JOB_NAME}"

# =============================================================================
# Step 0: 环境验证 (可选)
# =============================================================================
if [[ "${SKIP_ENV_CHECK}" != "true" ]]; then
  echo ""
  echo "Step 0: Verifying environment..."
  echo "  PyTorch 25.12 + B200/SM100 support"
  
  python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
  python -c "import triton; print(f'  triton: {triton.__version__}')"
  python -c "import mamba_ssm; print(f'  mamba-ssm: {mamba_ssm.__version__}')" 2>/dev/null || echo "  mamba-ssm: not found (will install)"
  python -c "import transformer_engine; print(f'  TransformerEngine: {transformer_engine.__version__}')"
  python -c "import huggingface_hub; print(f'  huggingface-hub: {huggingface_hub.__version__}')"
  python -c "import numpy as np; print(f'  numpy: {np.__version__}')"
fi

# =============================================================================
# Step 1: 克隆代码库 (仅主节点)
# =============================================================================
if [[ "${SKIP_CLONE_REPO}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 1: Cloning Pai-Megatron-Patch..."
  cd $PAI_WORKSPACE
  
  if [ -d "Pai-Megatron-Patch" ]; then
    echo "  Repository exists, updating..."
    cd Pai-Megatron-Patch && git pull && git submodule update --init --recursive && cd ..
  else
    git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
  fi
else
  echo "Step 1: Skipped (SKIP_CLONE_REPO=true or not rank 0)"
fi

# =============================================================================
# Step 2: 下载模型和数据集 (仅主节点)
# =============================================================================
if [[ "${SKIP_DOWNLOAD_DATA}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 2: Downloading model and datasets..."
  mkdir -p $HF_CKPT_DIR $DATASET_DIR
  
  # 下载模型
  echo "  Downloading $MODEL_NAME..."
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/$MODEL_NAME', local_dir='$HF_CKPT_DIR/$MODEL_NAME')"
  
  # 下载数据集
  echo "  Downloading datasets..."
  cd $DATASET_DIR
  wget -nc -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.bin || true
  wget -nc -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.idx || true
  wget -nc -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-train-general.json || true
  wget -nc -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-valid-general.json || true
  
  # 创建同步标志
  mkdir -p $SYNC_DIR && touch $SYNC_DIR/download_complete_flag
  echo "  Download completed!"
else
  echo "Step 2: Skipped (SKIP_DOWNLOAD_DATA=true or not rank 0)"
fi

# 等待主节点完成下载
if [[ "$JOB_COMPLETION_INDEX" -ne "0" ]]; then
  echo "  Waiting for node 0 to complete downloads..."
  while [[ ! -f $SYNC_DIR/download_complete_flag ]]; do sleep 5; done
fi

# =============================================================================
# Step 3: 检查点转换 (仅主节点)
#
# 为什么需要转换？
# - 这是**格式转换**，不是预分片
# - HuggingFace 和 MCore (Megatron-Core) 使用不同的权重命名规范：
#   HuggingFace: model.layers.0.self_attn.q_proj.weight
#   MCore:       decoder.layers.0.self_attention.query.weight
# - Megatron-Core 的 load_checkpoint() 只识别 MCore 格式
# - 转换后仍然是完整权重，不涉及 TP/PP/EP 分片
#
# TP/PP/EP 分片何时发生？
# - 分片发生在训练启动时，由 Megatron-Core 根据并行配置动态处理
# - 每个 rank 只加载自己需要的权重分片
# - 因此转换脚本不需要指定 TP/PP/EP 参数
# =============================================================================
if [[ "${SKIP_CHECKPOINT_CONVERSION}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 3: Converting checkpoints (HuggingFace → MCore format)..."
  
  # 精度配置 (可通过环境变量覆盖)
  CKPT_PRECISION=${CKPT_PRECISION:-"bf16"}
  
  # 输出目录包含精度信息，避免不同精度冲突
  MCORE_OUTPUT_DIR="${MCORE_CKPT_DIR}/${MODEL_NAME}-to-mcore-${CKPT_PRECISION}"
  
  # 检查是否需要转换 (支持 FORCE_CONVERSION=true 强制重新转换)
  if [[ "${FORCE_CONVERSION}" != "true" && -d "$MCORE_OUTPUT_DIR" && $(ls -A "$MCORE_OUTPUT_DIR" 2>/dev/null | grep -c "\.pt\|\.safetensors") -gt 0 ]]; then
    echo "  Checkpoint already converted (${CKPT_PRECISION}), skipping..."
    echo "  Set FORCE_CONVERSION=true to reconvert"
  else
    mkdir -p $MCORE_CKPT_DIR
    cd ${PATCH_DIR}/toolkits/distributed_checkpoints_convertor
    
    echo "  Converting to ${CKPT_PRECISION} format..."
    OMP_NUM_THREADS=12 WORLD_SIZE=1 RANK=0 \
    bash scripts/qwen3_next/run_8xH20.sh \
      $MODEL_SIZE \
      ${HF_CKPT_DIR}/${MODEL_NAME} \
      $MCORE_OUTPUT_DIR \
      false true ${CKPT_PRECISION}
  fi
  
  touch $SYNC_DIR/conversion_complete_flag
  echo "  Conversion completed!"
else
  echo "Step 3: Skipped (SKIP_CHECKPOINT_CONVERSION=true or not rank 0)"
fi

# 等待主节点完成转换
if [[ "$JOB_COMPLETION_INDEX" -ne "0" ]]; then
  echo "  Waiting for checkpoint conversion..."
  while [[ ! -f $SYNC_DIR/conversion_complete_flag ]]; do sleep 5; done
fi

# =============================================================================
# Step 4: 训练
# =============================================================================
if [[ "${SKIP_TRAINING}" != "true" ]]; then
  echo ""
  echo "Step 4: Starting training (mode: $TRAINING_MODE)..."
  cd ${PATCH_DIR}/examples/qwen3_next
  
  # =========================================================================
  # B200 兼容性 Patch: 修改 run_mcore_qwen3.sh 支持动态层数和 Pattern
  # =========================================================================
  echo "  Applying B200 compatibility patches..."
  
  # Patch 1: 动态层数支持 (NUM_LAYERS=${MP_NUM_LAYERS:-96})
  if ! grep -q 'MP_NUM_LAYERS' run_mcore_qwen3.sh; then
    sed -i 's/NUM_LAYERS=96/NUM_LAYERS=${MP_NUM_LAYERS:-96}/' run_mcore_qwen3.sh
    echo "    [Patch 1] Dynamic NUM_LAYERS enabled"
  fi
  
  # Patch 2: 动态 Hybrid Pattern 生成
  # Pattern 单元: M-M-M-*- (8字符 = 3M + 4- + 1*)
  # 按 NUM_LAYERS 重复，确保与 ratio 计算匹配
  if ! grep -q 'PATTERN_UNIT=' run_mcore_qwen3.sh; then
    # 在 HYBRID_TRANSFORMER_RATIO 行后插入动态 Pattern 生成逻辑
    sed -i '/HYBRID_TRANSFORMER_RATIO=/a \
# Dynamic Hybrid Pattern generation for B200 compatibility\
PATTERN_UNIT="M-M-M-*-"\
PATTERN_REPEATS=$((NUM_LAYERS / 8))\
HYBRID_MAMBA_TRANSFORMER_PATTERN=$(printf "%s" $(yes "$PATTERN_UNIT" | head -n $PATTERN_REPEATS | tr -d "\\n"))' run_mcore_qwen3.sh
    echo "    [Patch 2] Dynamic Hybrid Pattern enabled"
  fi
  
  # =========================================================================
  # 配置训练参数
  # =========================================================================
  TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
  GLOBAL_BATCH_SIZE=$TOTAL_GPUS
  
  # 层数配置 (可通过环境变量覆盖，调试时可用 MP_NUM_LAYERS=8)
  NUM_LAYERS=${MP_NUM_LAYERS:-96}
  
  # Token 配置 (可通过环境变量覆盖，快速测试用 MP_TRAIN_TOKENS=100000)
  TRAIN_TOKENS=${MP_TRAIN_TOKENS:-100000}
  WARMUP_TOKENS=${MP_WARMUP_TOKENS:-1000}
  
  if [ "$TRAINING_MODE" == "sft" ]; then
    OUTPUT_DIR="${PAI_WORKSPACE}/logs/output_mcore_qwen3_next_finetune"
    SFT_FLAG="true"
    DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_sft_text_document"
    VALID_DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_sft_text_document"
  else
    OUTPUT_DIR="${PAI_WORKSPACE}/logs/output_mcore_qwen3_next_pretrain"
    SFT_FLAG="false"
    DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_text_document"
    VALID_DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_text_document"
  fi
  mkdir -p $OUTPUT_DIR
  
  # 精度配置 (与检查点转换步骤一致)
  CKPT_PRECISION=${CKPT_PRECISION:-"bf16"}
  MCORE_CKPT_PATH="${MCORE_CKPT_DIR}/${MODEL_NAME}-to-mcore-${CKPT_PRECISION}"
  
  echo "  Nodes: $NNODES, GPUs/Node: $GPUS_PER_NODE, Total: $TOTAL_GPUS"
  echo "  GBS: $GLOBAL_BATCH_SIZE, EP: 8, TP: 1, PP: 1"
  echo "  NUM_LAYERS: $NUM_LAYERS (set MP_NUM_LAYERS to override)"
  echo "  Checkpoint: $MCORE_CKPT_PATH"
  
  # =========================================================================
  # 运行训练
  # =========================================================================
  # 并行配置: EP=8 (512 experts), TP=1, PP=1
  #
  # B200/SM100 Attention 配置:
  # - FL=true: 启用 Flash Attention (Fused Attention 不支持 kv_channels=256)
  # - NVTE_DEBUG: 调试 TransformerEngine attention backend
  #
  # 参数说明 (run_mcore_qwen3.sh):
  # $1=ENV, $2=MODEL_SIZE, $3=MICRO_BS, $4=GLOBAL_BS, $5=LR, $6=MIN_LR,
  # $7=SEQ_LEN, $8=PAD_LEN, $9=PRECISION, $10=TP, $11=PP, $12=CP, $13=VPP,
  # $14=EP, $15=SP, $16=DO, $17=FL, $18=SFT, $19=PR, $20=AC, $21=SAVE_INTERVAL,
  # $22=DATASET_PATH, $23=VALID_DATASET_PATH, $24=PRETRAIN_CKPT_PATH,
  # $25=TRAIN_TOKENS, $26=WARMUP_TOKENS, $27=OUTPUT_DIR
  #
  OMP_NUM_THREADS=12 \
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
  MP_NUM_LAYERS=$NUM_LAYERS \
  WORLD_SIZE=$NNODES \
  RANK=$JOB_COMPLETION_INDEX \
  KUBERNETES_CONTAINER_RESOURCE_GPU=$GPUS_PER_NODE \
  bash run_mcore_qwen3.sh \
    dlc $MODEL_SIZE 1 $GLOBAL_BATCH_SIZE 1e-5 1e-6 1024 1024 ${CKPT_PRECISION} \
    1 1 1 1 8 true true true $SFT_FLAG none false 100000 \
    $DATASET_PATH $VALID_DATASET_PATH \
    $MCORE_CKPT_PATH \
    $TRAIN_TOKENS $WARMUP_TOKENS $OUTPUT_DIR

  echo "  Training completed!"
else
  echo "Step 4: Skipped (SKIP_TRAINING=true)"
fi

echo ""
echo "=============================================="
echo "Pipeline completed!"
echo "=============================================="
