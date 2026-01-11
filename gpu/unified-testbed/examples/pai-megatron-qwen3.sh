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
# Pai-Megatron-Patch Qwen3 训练示例
# 完整的端到端训练流水线：环境准备 → 克隆 → 下载 → 转换 → 训练
# =============================================================================

echo "=============================================="
echo "Pai-Megatron-Patch Qwen3 Training Pipeline"
echo "Model: Qwen3-30B-A3B"
echo "=============================================="

# 工作空间配置
PAI_WORKSPACE=${PAI_WORKSPACE_ROOT:-/mnt}

# -----------------------------------------------------------------------------
# 功能0: 环境准备 (所有节点都需要安装依赖)
# 注意: huggingface_hub[cli], modelscope, triton 修复已在 default-launcher.sh 中完成
# 此处安装 Pai-Megatron-Patch 特定依赖（所有节点都需要 transformers）
# -----------------------------------------------------------------------------
if [[ "${SKIP_ENV_SETUP}" != "true" ]]; then
  echo ""
  echo "Step 0: Setting up Pai-Megatron-Patch specific dependencies on node $JOB_COMPLETION_INDEX..."
  
  # 升级必要的依赖（Pai-Megatron-Patch 特定，所有节点都需要 transformers）
  pip install --upgrade nvidia-nccl-cu12 datasets==3.6.0 packaging==24.2 transformers -q 2>/dev/null
  
  echo "  Environment setup completed on node $JOB_COMPLETION_INDEX!"
else
  echo ""
  echo "Step 0: Skipping environment setup (SKIP_ENV_SETUP=true)"
fi

# -----------------------------------------------------------------------------
# 功能1: 克隆 Pai-Megatron-Patch 代码库
# -----------------------------------------------------------------------------
if [[ "${SKIP_CLONE_REPO}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 1: Cloning Pai-Megatron-Patch repository..."
  cd $PAI_WORKSPACE
  
  if [ -d "Pai-Megatron-Patch" ]; then
    echo "  Repository already exists, pulling latest changes..."
    cd Pai-Megatron-Patch
    git pull
    cd ..
  else
    git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
  fi
  echo "  Repository cloned successfully!"
else
  echo ""
  echo "Step 1: Skipping repository cloning (SKIP_CLONE_REPO=true or not rank 0)"
fi

# -----------------------------------------------------------------------------
# 功能2: 下载模型和数据集
# -----------------------------------------------------------------------------
if [[ "${SKIP_DOWNLOAD_DATA}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 2: Downloading models and datasets..."
  cd $PAI_WORKSPACE
  
  # 下载 Qwen3-30B-A3B 模型
  mkdir -p qwen-ckpts
  echo "  Downloading Qwen3-30B-A3B model..."
  # 使用 Python 脚本方式调用，避免 huggingface-cli PATH 问题
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-30B-A3B', local_dir='$PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B')"
  
  # 下载预处理数据集（使用 -nc 避免重复下载）
  mkdir -p qwen-datasets
  cd qwen-datasets
  echo "  Downloading preprocessed datasets..."
  wget -nc -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.bin || true
  wget -nc -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.idx || true
  wget -nc -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-train-general.json || true
  wget -nc -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-valid-general.json || true
  
  echo "  Download completed!"
else
  echo ""
  echo "Step 2: Skipping data download (SKIP_DOWNLOAD_DATA=true or not rank 0)"
fi

# -----------------------------------------------------------------------------
# 同步点：等待主节点完成下载
# -----------------------------------------------------------------------------
# 使用 JobSet 名称作为唯一标识符（从 HOSTNAME_PREFIX 提取）
JOB_NAME="${HOSTNAME_PREFIX%-workload-}"
SYNC_DIR="$PAI_WORKSPACE/sync_flags_${JOB_NAME}"

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

# -----------------------------------------------------------------------------
# 功能3: 检查点转换（仅主节点执行，单节点即可完成转换）
# -----------------------------------------------------------------------------
if [[ "${SKIP_CHECKPOINT_CONVERSION}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 3: Converting checkpoints (single node mode)..."
  cd $PAI_WORKSPACE/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
  
  # 检查是否已转换过（跳过重复转换）
  if [[ -d "$PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B-to-mcore" && \
        $(ls -A "$PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B-to-mcore" 2>/dev/null | grep -c "\.pt\|\.safetensors") -gt 0 ]]; then
    echo "  Checkpoint already converted, skipping..."
  else
    # 单节点转换（WORLD_SIZE=1, RANK=0）
    OMP_NUM_THREADS=12 WORLD_SIZE=1 RANK=0 \
    bash scripts/qwen3/run_8xH20.sh \
      A3B \
      $PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B \
      $PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B-to-mcore \
      false \
      true \
      bf16
  fi
  
  echo "  Checkpoint conversion completed!"
  
  # 创建转换完成标志
  touch $SYNC_DIR/conversion_complete_flag
else
  echo ""
  echo "Step 3: Skipping checkpoint conversion (SKIP_CHECKPOINT_CONVERSION=true or not rank 0)"
fi

# -----------------------------------------------------------------------------
# 同步点：等待主节点完成检查点转换
# -----------------------------------------------------------------------------
if [[ "$JOB_COMPLETION_INDEX" -ne "0" ]]; then
  echo ""
  echo "Node $JOB_COMPLETION_INDEX waiting for checkpoint conversion to complete..."
  while [[ ! -f $SYNC_DIR/conversion_complete_flag ]]; do
    sleep 5
    echo "  Still waiting..."
  done
  echo "Checkpoint conversion complete, proceeding to training..."
fi

# -----------------------------------------------------------------------------
# 功能4: 运行训练
# -----------------------------------------------------------------------------
if [[ "${SKIP_TRAINING}" != "true" ]]; then
  echo ""
  echo "Step 4: Starting training..."
  cd $PAI_WORKSPACE/Pai-Megatron-Patch/examples/qwen3
  
  # Pai-Megatron-Patch 使用 WORLD_SIZE 作为节点数，RANK 作为节点 Rank
  OMP_NUM_THREADS=12 \
  WORLD_SIZE=$NNODES \
  RANK=$JOB_COMPLETION_INDEX \
  KUBERNETES_CONTAINER_RESOURCE_GPU=$GPUS_PER_NODE \
  sh run_mcore_qwen3.sh \
    dlc \
    A3B \
    1 \
    8 \
    1e-5 \
    1e-6 \
    128 \
    128 \
    bf16 \
    4 \
    2 \
    1 \
    1 \
    4 \
    true \
    true \
    true \
    false \
    sel \
    false \
    100000 \
    $PAI_WORKSPACE/qwen-datasets/mmap_qwen3_datasets_text_document \
    $PAI_WORKSPACE/qwen-datasets/mmap_qwen3_datasets_text_document \
    $PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B-to-mcore \
    10000 \
    100 \
    $PAI_WORKSPACE/logs/output_mcore_qwen3_pretrain

  echo "  Training completed!"
else
  echo ""
  echo "Step 4: Skipping training (SKIP_TRAINING=true)"
fi

echo ""
echo "=============================================="
echo "Qwen3 Training Pipeline Completed!"
echo "=============================================="
