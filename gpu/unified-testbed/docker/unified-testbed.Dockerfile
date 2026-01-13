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
# A4 Unified Testbed Docker Image
# 基于 NVIDIA PyTorch 25.12 构建，用于 A4 GPU (B200/Blackwell) 训练和测试
#
# 版本选择说明:
# - B200 (Blackwell/SM100) GPU 需要最低 CUDA 12.8，推荐使用官方 PyTorch 容器
# - PyTorch 25.12: 包含 TransformerEngine、完整的 B200 attention backend 支持
# - 这是 NVIDIA 推荐的稳健型方案，适用于生产环境与大规模训练
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.12-py3

LABEL maintainer="Google Cloud AI Team"
LABEL description="Unified testbed for A4 GPU training and testing on GKE"
LABEL version="25.12"

WORKDIR /workspace

# =============================================================================
# 1. 系统工具和 Google Cloud 组件安装
# =============================================================================

# GCSfuse 和 Google Cloud CLI (用于 GCS 集成)
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    wget \
    vim \
    htop \
    nvtop \
    tmux \
    fio \
    net-tools \
    iputils-ping \
    dnsutils \
    pciutils \
    numactl \
    openssh-server \
    openssh-client \
    && echo "deb https://packages.cloud.google.com/apt gcsfuse-buster main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install --yes gcsfuse \
    && apt-get install --yes google-cloud-cli \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && mkdir -p /gcs /mnt /ssd

# =============================================================================
# 2. Python 基础依赖安装
# =============================================================================

# NVIDIA DLLogger 和常用工具
RUN pip install --no-cache-dir \
    git+https://github.com/NVIDIA/dllogger#egg=dllogger \
    tensorboard \
    wandb \
    hydra-core \
    omegaconf \
    huggingface_hub[cli] \
    modelscope

# =============================================================================
# 3. Pai-Megatron-Patch 通用依赖
# =============================================================================

# 通用 ML 依赖（所有训练脚本需要）
# 注意：
# - NCCL: 由 GIB 在运行时提供优化版本
# - huggingface-hub: 必须降级到 <1.0 以兼容 transformers
# - numpy: 必须降级到 <2.0 以兼容 Pai-Megatron-Patch (numpy.product)
RUN pip install --no-cache-dir \
    datasets \
    transformers \
    && pip install --no-cache-dir "huggingface-hub>=0.34.0,<1.0" \
    && pip install --no-cache-dir "numpy<2.0"

# =============================================================================
# 4. Qwen3-Next 特殊依赖（CUDA 扩展，需要源码编译）
# PyTorch 25.12 镜像已包含 TransformerEngine 和 B200 完整支持
# =============================================================================

# triton (PyTorch 25.12 应该已内置兼容版本，但确保最新)
RUN pip install --no-cache-dir --no-build-isolation --upgrade "triton>=3.5.0" || true

# 修复 triton ldconfig 编码问题 (NGC 镜像特有)
RUN for TRITON_PATH in /usr/local/lib/python3.*/dist-packages/triton/backends/nvidia/driver.py; do \
    if [ -f "$TRITON_PATH" ]; then \
    sed -i 's/\.decode()/.decode(errors="ignore")/g' "$TRITON_PATH" || true; \
    fi; \
    done

# mamba-ssm (Qwen3-Next 混合架构核心依赖)
RUN pip install --no-cache-dir --no-build-isolation mamba-ssm || \
    pip install --no-cache-dir --no-build-isolation --no-deps mamba-ssm || true

# causal-conv1d (mamba-ssm 依赖)
RUN pip install --no-cache-dir --no-build-isolation causal-conv1d || \
    pip install --no-cache-dir --no-build-isolation --no-deps causal-conv1d || true

# flash-linear-attention (Qwen3-Next 需要)
RUN pip install --no-cache-dir --no-build-isolation flash-linear-attention || true

# =============================================================================
# 最终版本锁定（防止后续依赖安装覆盖关键版本）
# =============================================================================
RUN pip install --no-cache-dir "huggingface-hub>=0.34.0,<1.0" "numpy<2.0"

# =============================================================================
# 5. SSH 服务配置 (用于多节点分布式训练通信)
# =============================================================================

# 配置 SSH 在端口 2222 上运行 (避免与主机 SSH 冲突)
RUN mkdir -p /var/run/sshd /root/.ssh \
    && cd /etc/ssh/ \
    && sed --in-place='.bak' 's/#Port 22/Port 2222/' sshd_config \
    && sed --in-place='.bak' 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' sshd_config \
    && sed --in-place='.bak' 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' sshd_config \
    && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config \
    && echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && echo "LogLevel ERROR" >> /etc/ssh/ssh_config

# 生成 SSH 密钥对
RUN ssh-keygen -t rsa -b 4096 -q -f /root/.ssh/id_rsa -N "" \
    && touch /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys \
    && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# =============================================================================
# 6. 环境变量配置
# =============================================================================

# CUDA 和 NCCL 环境变量
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_DEBUG=WARN

# Python 环境 (PyTorch 25.12 使用系统 Python)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# TransformerEngine 配置
# PyTorch 25.12 包含完整的 B200/SM100 支持，无需禁用 fused attention
# 如果遇到问题，可以设置 NVTE_FUSED_ATTN=0 回退到其他 backend
ENV NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# =============================================================================
# 7. 工作目录设置
# =============================================================================

WORKDIR /workspace

# 创建必要的目录结构
RUN mkdir -p /workspace/scripts \
    /workspace/configs \
    /workspace/logs \
    /workload/launcher \
    /workload/scripts \
    /workload/configs

# =============================================================================
# 8. 健康检查
# =============================================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nvidia-smi > /dev/null 2>&1 || exit 1

# =============================================================================
# 9. 默认命令
# =============================================================================

CMD ["/bin/bash"]
