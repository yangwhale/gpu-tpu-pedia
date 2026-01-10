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
# 功能0: 环境准备 (仅主节点执行)
# 注意: 由于使用 nvcr.io/nvidia/pytorch:25.06-py3 基础镜像，
#       需要修复一些兼容性问题并安装必要的依赖
# -----------------------------------------------------------------------------
if [[ "${SKIP_ENV_SETUP}" != "true" && "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo ""
  echo "Step 0: Setting up environment..."
  
  # 修复 NGC 镜像中的 triton ldconfig 路径问题
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
  huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir $PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B
  
  # 下载预处理数据集
  mkdir -p qwen-datasets
  cd qwen-datasets
  echo "  Downloading preprocessed datasets..."
  wget -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.bin
  wget -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.idx
  wget -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-train-general.json
  wget -q https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-valid-general.json
  
  echo "  Download completed!"
else
  echo ""
  echo "Step 2: Skipping data download (SKIP_DOWNLOAD_DATA=true or not rank 0)"
fi

# -----------------------------------------------------------------------------
# 同步点：等待主节点完成下载
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 功能3: 检查点转换
# -----------------------------------------------------------------------------
if [[ "${SKIP_CHECKPOINT_CONVERSION}" != "true" ]]; then
  echo ""
  echo "Step 3: Converting checkpoints..."
  cd $PAI_WORKSPACE/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
  
  OMP_NUM_THREADS=12 WORLD_SIZE=$NNODES RANK=$JOB_COMPLETION_INDEX \
  bash scripts/qwen3/run_8xH20.sh \
    A3B \
    $PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B \
    $PAI_WORKSPACE/qwen-ckpts/Qwen3-30B-A3B-to-mcore \
    false \
    true \
    bf16
  
  echo "  Checkpoint conversion completed!"
else
  echo ""
  echo "Step 3: Skipping checkpoint conversion (SKIP_CHECKPOINT_CONVERSION=true)"
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
