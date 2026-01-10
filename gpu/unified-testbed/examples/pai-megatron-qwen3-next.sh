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
# Pai-Megatron-Patch Qwen3-Next 训练示例
# 支持 Qwen3-Next-80B-A3B-Instruct 模型的预训练和指令微调
# 
# 参考: Pai-Megatron-Patch/examples/qwen3_next/README.md
# =============================================================================

echo "=============================================="
echo "Pai-Megatron-Patch Qwen3-Next Training Pipeline"
echo "Model: Qwen3-Next-80B-A3B-Instruct"
echo "=============================================="

# =============================================================================
# 配置参数
# =============================================================================

# 工作空间配置
PAI_WORKSPACE=${PAI_WORKSPACE_ROOT:-/mnt}
PATCH_DIR=${PAI_WORKSPACE}/Pai-Megatron-Patch

# 模型和数据路径
CKPT_DIR=${PAI_WORKSPACE}/ckpts
HF_CKPT_DIR=${CKPT_DIR}/huggingface
MCORE_CKPT_DIR=${CKPT_DIR}/mcore
DATASET_DIR=${PAI_WORKSPACE}/datasets

# 模型配置
MODEL_NAME="Qwen3-Next-80B-A3B-Instruct"
MODEL_SIZE="A3B"

# 训练模式: pretrain 或 sft
TRAINING_MODE=${TRAINING_MODE:-"pretrain"}

# =============================================================================
# 功能0: 环境准备 (仅主节点执行)
# =============================================================================
if [[ "${SKIP_ENV_SETUP}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 0: Setting up environment..."
  
  # 修复 NGC 镜像中的 triton 小错误
  TRITON_FILE="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py"
  if [ -f "$TRITON_FILE" ]; then
    sed -i 's|libs = subprocess.check_output(\["ldconfig"|libs = subprocess.check_output(["/sbin/ldconfig"|g' $TRITON_FILE
    echo "  Fixed triton ldconfig path"
  fi
  
  # 升级必要的依赖
  echo "  Installing/upgrading dependencies..."
  pip install --upgrade nvidia-nccl-cu12 -q
  pip install datasets==3.6.0 packaging==24.2 modelscope -q
  
  echo "  Environment setup completed!"
else
  echo ""
  echo "Step 0: Skipping environment setup (SKIP_ENV_SETUP=true or not rank 0)"
fi

# =============================================================================
# 功能1: 克隆 Pai-Megatron-Patch 代码库
# =============================================================================
if [[ "${SKIP_CLONE_REPO}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 1: Cloning Pai-Megatron-Patch repository..."
  cd $PAI_WORKSPACE
  
  if [ -d "Pai-Megatron-Patch" ]; then
    echo "  Repository already exists, pulling latest changes..."
    cd Pai-Megatron-Patch
    git pull
    git submodule update --init --recursive
    cd ..
  else
    git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
  fi
  echo "  Repository cloned successfully!"
else
  echo ""
  echo "Step 1: Skipping repository cloning (SKIP_CLONE_REPO=true or not rank 0)"
fi

# =============================================================================
# 功能2: 下载模型和数据集
# =============================================================================
if [[ "${SKIP_DOWNLOAD_DATA}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 2: Downloading models and datasets..."
  
  # 创建目录
  mkdir -p $HF_CKPT_DIR
  mkdir -p $DATASET_DIR
  
  # 下载 Qwen3-Next-80B-A3B-Instruct 模型
  echo "  Downloading $MODEL_NAME model from ModelScope..."
  cd $HF_CKPT_DIR
  if [ -d "$MODEL_NAME" ]; then
    echo "    Model already exists, skipping download..."
  else
    modelscope download --model Qwen/$MODEL_NAME --local_dir $MODEL_NAME
  fi
  
  # 下载预处理数据集
  echo "  Downloading preprocessed datasets..."
  cd $DATASET_DIR
  
  if [ ! -f "mmap_qwen3_datasets_text_document.bin" ]; then
    wget -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.bin
  fi
  
  if [ ! -f "mmap_qwen3_datasets_text_document.idx" ]; then
    wget -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.idx
  fi
  
  # 下载 SFT 数据集
  if [ ! -f "alpaca_zh-train-general.json" ]; then
    wget -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-train-general.json
  fi
  
  if [ ! -f "alpaca_zh-valid-general.json" ]; then
    wget -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-valid-general.json
  fi
  
  echo "  Download completed!"
else
  echo ""
  echo "Step 2: Skipping data download (SKIP_DOWNLOAD_DATA=true or not rank 0)"
fi

# =============================================================================
# 同步点：等待主节点完成下载
# =============================================================================
SYNC_DIR="$PAI_WORKSPACE/sync_flags_${JOB_ID}"

if [[ "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Creating sync flag to signal other nodes..."
  mkdir -p $SYNC_DIR
  touch $SYNC_DIR/download_complete_flag
fi

if [[ "$JOB_COMPLETION_INDEX" -ne "0" ]]; then
  echo ""
  echo "Node $JOB_COMPLETION_INDEX waiting for node 0 to complete downloads..."
  while [[ ! -f $SYNC_DIR/download_complete_flag ]]; do
    sleep 5
    echo "  Still waiting..."
  done
  echo "Download complete flag detected, proceeding..."
fi

# =============================================================================
# 功能3: 检查点格式转换 (HuggingFace -> MCore)
# =============================================================================
if [[ "${SKIP_CHECKPOINT_CONVERSION}" != "true" ]]; then
  echo ""
  echo "Step 3: Converting checkpoints (HuggingFace -> MCore)..."
  
  MCORE_OUTPUT_DIR="${MCORE_CKPT_DIR}/${MODEL_NAME}-to-mcore"
  
  if [ -d "$MCORE_OUTPUT_DIR" ]; then
    echo "  MCore checkpoint already exists, skipping conversion..."
  else
    mkdir -p $MCORE_CKPT_DIR
    cd ${PATCH_DIR}/toolkits/distributed_checkpoints_convertor
    
    # 使用分布式转换脚本
    # 参数: MODEL_SIZE LOAD_DIR SAVE_DIR MG2HF USE_CUDA PR
    OMP_NUM_THREADS=12 WORLD_SIZE=$NNODES RANK=$JOB_COMPLETION_INDEX \
    bash scripts/qwen3_next/run_8xH20.sh \
      $MODEL_SIZE \
      ${HF_CKPT_DIR}/${MODEL_NAME} \
      $MCORE_OUTPUT_DIR \
      false \
      true \
      bf16
    
    echo "  Checkpoint conversion completed!"
  fi
else
  echo ""
  echo "Step 3: Skipping checkpoint conversion (SKIP_CHECKPOINT_CONVERSION=true)"
fi

# =============================================================================
# 功能4: 运行训练
# =============================================================================
if [[ "${SKIP_TRAINING}" != "true" ]]; then
  echo ""
  echo "Step 4: Starting training..."
  echo "  Training mode: $TRAINING_MODE"
  
  cd ${PATCH_DIR}/examples/qwen3_next
  
  # 设置输出目录
  if [ "$TRAINING_MODE" == "sft" ]; then
    OUTPUT_DIR="${PAI_WORKSPACE}/logs/output_mcore_qwen3_next_finetune"
    SFT_FLAG="true"
    DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_sft_text_document"
    VALID_DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_sft_text_document"
    TRAIN_TOKENS="10000"
    WARMUP_TOKENS="100"
  else
    OUTPUT_DIR="${PAI_WORKSPACE}/logs/output_mcore_qwen3_next_pretrain"
    SFT_FLAG="false"
    DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_text_document"
    VALID_DATASET_PATH="${DATASET_DIR}/mmap_qwen3_datasets_text_document"
    TRAIN_TOKENS="1000000000"
    WARMUP_TOKENS="10000"
  fi
  
  mkdir -p $OUTPUT_DIR
  
  # Pai-Megatron-Patch 使用 WORLD_SIZE 作为节点数，RANK 作为节点 Rank
  # 参数说明参考 Pai-Megatron-Patch/examples/qwen3_next/README.md
  #
  # ENV=$1                          # dsw单机 或 dlc多机
  # MODEL_SIZE=$2                   # A3B
  # BATCH_SIZE=$3                   # per-GPU batch size
  # GLOBAL_BATCH_SIZE=$4            # 全局 batch size
  # LR=$5                           # 学习率
  # MIN_LR=$6                       # 最小学习率
  # SEQ_LEN=$7                      # 序列长度
  # PAD_LEN=$8                      # Padding长度
  # PR=$9                           # 精度: fp16, bf16, fp8
  # TP=${10}                        # 张量并行度
  # PP=${11}                        # 流水并行度
  # CP=${12}                        # 上下文并行度
  # ETP=${13}                       # 专家张量并行度
  # EP=${14}                        # 专家并行度
  # SP=${15}                        # 序列并行: true/false
  # DO=${16}                        # Megatron Zero-1优化器: true/false
  # FL=${17}                        # Flash Attention: true/false
  # SFT=${18}                       # 微调模式: true/false
  # AC=${19}                        # 激活检查点: sel/full/offload/none
  # OPTIMIZER_OFFLOAD=${20}         # Offload optimizer: false 或 0~1
  # SAVE_INTERVAL=${21}             # 保存间隔
  # DATASET_PATH=${22}              # 训练数据路径
  # VALID_DATASET_PATH=${23}        # 验证数据路径
  # PRETRAIN_CHECKPOINT_PATH=${24}  # 预训练模型路径
  # TRAIN_TOKENS_OR_ITERS=${25}     # 训练TOKEN数
  # WARMUP_TOKENS_OR_ITERS=${26}    # 预热TOKEN数
  # OUTPUT_BASEPATH=${27}           # 输出路径
  
  OMP_NUM_THREADS=12 \
  WORLD_SIZE=$NNODES \
  RANK=$JOB_COMPLETION_INDEX \
  KUBERNETES_CONTAINER_RESOURCE_GPU=$GPUS_PER_NODE \
  bash run_mcore_qwen3.sh \
    dlc \
    $MODEL_SIZE \
    1 \
    8 \
    1e-5 \
    1e-6 \
    1024 \
    1024 \
    bf16 \
    1 \
    1 \
    1 \
    1 \
    8 \
    true \
    true \
    false \
    $SFT_FLAG \
    none \
    false \
    100000 \
    $DATASET_PATH \
    $VALID_DATASET_PATH \
    ${MCORE_CKPT_DIR}/${MODEL_NAME}-to-mcore \
    $TRAIN_TOKENS \
    $WARMUP_TOKENS \
    $OUTPUT_DIR

  echo "  Training completed!"
else
  echo ""
  echo "Step 4: Skipping training (SKIP_TRAINING=true)"
fi

# =============================================================================
# 可选: 使用 JSON 格式数据进行 SFT
# =============================================================================
run_sft_with_json() {
  echo ""
  echo "Running SFT with JSON dataset..."
  
  cd ${PATCH_DIR}/examples/qwen3_next
  
  export MP_DATASET_TYPE="raw"
  
  OMP_NUM_THREADS=12 \
  WORLD_SIZE=$NNODES \
  RANK=$JOB_COMPLETION_INDEX \
  KUBERNETES_CONTAINER_RESOURCE_GPU=$GPUS_PER_NODE \
  bash run_mcore_qwen3.sh \
    dlc \
    $MODEL_SIZE \
    1 \
    8 \
    1e-5 \
    1e-6 \
    1024 \
    1024 \
    bf16 \
    1 \
    1 \
    1 \
    1 \
    8 \
    true \
    true \
    false \
    true \
    none \
    false \
    100000 \
    ${DATASET_DIR}/alpaca_zh-train-general.json \
    ${DATASET_DIR}/alpaca_zh-valid-general.json \
    ${MCORE_CKPT_DIR}/${MODEL_NAME}-to-mcore \
    10000 \
    100 \
    ${PAI_WORKSPACE}/logs/output_mcore_qwen3_next_sft_json
}

# 如果设置了 USE_JSON_SFT=true，则使用 JSON 格式 SFT
if [[ "${USE_JSON_SFT}" == "true" && "${SKIP_TRAINING}" != "true" ]]; then
  run_sft_with_json
fi

echo ""
echo "=============================================="
echo "Qwen3-Next Training Pipeline Completed!"
echo "=============================================="
