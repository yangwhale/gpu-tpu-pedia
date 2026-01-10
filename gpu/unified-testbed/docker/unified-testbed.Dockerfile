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
# 基于 NVIDIA PyTorch 25.06 构建，用于 A4 GPU 训练和测试
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.06-py3

LABEL maintainer="Google Cloud AI Team"
LABEL description="Unified testbed for A4 GPU training and testing on GKE"
LABEL version="25.06"

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
# 2. Python 依赖安装
# =============================================================================

# NVIDIA DLLogger (训练指标记录)
RUN pip install --no-cache-dir \
    git+https://github.com/NVIDIA/dllogger#egg=dllogger \
    tensorboard \
    wandb \
    hydra-core \
    omegaconf

# =============================================================================
# 3. SSH 服务配置 (用于多节点分布式训练通信)
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
# 4. 环境变量配置
# =============================================================================

# CUDA 和 NCCL 环境变量
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_DEBUG=WARN

# Python 环境
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# 5. 工作目录设置
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
# 6. 健康检查
# =============================================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nvidia-smi > /dev/null 2>&1 || exit 1

# =============================================================================
# 7. 默认命令
# =============================================================================

CMD ["/bin/bash"]
