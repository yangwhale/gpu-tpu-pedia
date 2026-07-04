# DeepSeek V3 671B — GB200 NVL72 128 GPU HybridEP 训练复现指南

> forrest 集群实测, 2026-07-03 最佳稳态: **488 TFLOPs/GPU** (peak 518), PP=4 EP=32 HybridEP vs NVIDIA 256 GPU 参考 1,106 TFLOPs (PP=8 + VPP + CUDA graphs + overlap)

## 1. 硬件与集群

| 项目 | 规格 |
|---|---|
| GPU | 128× NVIDIA GB200 (184 GB HBM3e) |
| 节点 | 32× a4x-highgpu-4g (Grace + 4 Blackwell) |
| NVLink | 2 cliques × 16 nodes, NVL72 域内全互联 |
| RDMA | 4× MRDMA 400Gb/s per node (mlx5_0..3, RoCE), GIB v1.1.2 plugin |
| 网络 | GVNIC (管理) + MRDMA ×4 (GPU 通信) |
| 存储 | 4× 3TB NVMe LSSD (RAID0) + 70TB Lustre |
| k8s | v1.34.5 自管 kubeadm, Calico v3.32 |
| IMEX | Host nvidia-imex daemon (非 ComputeDomain), per-clique 独立 |

## 2. 软件栈

| 组件 | 版本 / commit |
|---|---|
| Base image | `nvcr.io/nvidia/pytorch:25.09-py3` (CUDA 13.0) |
| Megatron-LM | dev branch, commit `effebd81` (post PR #1917) |
| Transformer Engine | commit `7dd3914` (custom build, v2.9 based) |
| DeepEP (HybridEP) | commit `3f601f7`, branch `hybrid-ep` |
| cuDNN | v9.14 (`libcudnn9-cuda-13`) |
| NCCL | 2.30.4+cuda13.0 |
| GIB | v1.1.2 (nccl-plugin-gib-diagnostic-arm64) |

## 3. 镜像制作

### 3.1 Dockerfile

保存为 `Dockerfile.dsv3-perf`:

```dockerfile
# DeepSeek-V3 GB200 performance reproduction image
# Base: NGC pytorch:25.09-py3 (CUDA 13.0, PyTorch 2.9, ARM64)
# Build (on GB200 worker node, native ARM64):
#   buildah bud -f Dockerfile.dsv3-perf -t megatron-dsv3-perf:v1 .

FROM nvcr.io/nvidia/pytorch:25.09-py3 AS base

ENV SHELL=/bin/bash

# Stage 1: System packages
RUN apt-get update && \
    apt-get install -y sudo gdb bash-builtins git zsh tmux curl gettext libfabric-dev && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_arm64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Python packages (training + dev deps)
ENV CUDA_HOME=/usr/local/cuda
RUN unset PIP_CONSTRAINT && pip install --no-cache-dir \
    debugpy dm-tree einops wandb \
    sentencepiece tokenizers transformers torchvision ftfy datasets tqdm pydantic \
    tiktoken flask-restful \
    nltk wrapt pytest pytest-cov pytest_mock

# grouped_gemm needs CUDA + torch at build time, must skip PEP517 isolation
RUN CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="10.0" \
    pip install --no-build-isolation --no-cache-dir grouped_gemm

# Stage 3: cuDNN 9.14 (MXFP8 quantization + layernorm fusion)
RUN apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install libcudnn9-cuda-13 && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Stage 4: TransformerEngine (custom build with CPU + quantization optimizations)
ARG TE_COMMIT="7dd3914726abb79bc99ff5a5db1449458ed64151"
ARG TE_REPO="https://github.com/hxbai/TransformerEngine.git"
RUN pip install --no-cache-dir nvidia-mathdx==25.1.1 && \
    unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="100" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch \
    pip install --no-build-isolation --no-cache-dir "git+${TE_REPO}@${TE_COMMIT}"

# Stage 5: Megatron-LM dev branch (after PR 1917)
ARG MEGATRON_COMMIT="effebd81f410bc6566fffee6c320b6f8f762e06d"
RUN git clone https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM && \
    cd /opt/Megatron-LM && \
    git checkout ${MEGATRON_COMMIT} && \
    pip install --no-cache-dir -e .

# Stage 6: HybridEP from DeepEP hybrid-ep branch
ARG HYBRIDEP_COMMIT="3f601f7ac1c062c46502646ff04c535013bfca00"
RUN cd /opt && \
    git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git DeepEP-HybridEP && \
    cd DeepEP-HybridEP && \
    git checkout ${HYBRIDEP_COMMIT} && \
    TORCH_CUDA_ARCH_LIST="10.0" pip install --no-build-isolation --no-cache-dir .

# Stage 7: bindpcie (NUMA affinity, critical for GB200)
RUN wget https://raw.githubusercontent.com/NVIDIA/mlperf-common/refs/heads/main/client/bindpcie -O /usr/local/bin/bindpcie && \
    chmod 755 /usr/local/bin/bindpcie

# Stage 8: SSH server (for multi-node NCCL sshd barrier fallback)
RUN apt-get update && apt-get install -y openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Clean
RUN rm -rf /root/.cache /tmp/*

WORKDIR /opt/Megatron-LM
```

### 3.2 构建与推送

必须在 GB200 worker 节点上本地构建 (native ARM64, 无 qemu emulation):

```bash
# 1. 安装 buildah (idempotent)
sudo dnf install -y buildah

# 2. 构建 (约 30-60 min, TE CUDA 编译最慢)
sudo buildah bud --platform linux/arm64 \
  -t megatron-dsv3-perf:v1 \
  -f Dockerfile.dsv3-perf .

# 3. 推送到 Artifact Registry (节点 SA 自带 token)
AR_REPO="us-east1-docker.pkg.dev/<PROJECT>/forrest-repo-us-east1"

sudo gcloud auth print-access-token | \
  sudo buildah login -u oauth2accesstoken --password-stdin us-east1-docker.pkg.dev

sudo buildah push megatron-dsv3-perf:v1 \
  docker://${AR_REPO}/megatron-dsv3-perf:v1

# 注: SA token 1h 过期, build 耗时 > 1h 时 push 前需重新 auth
```

## 4. 模型配置

DeepSeek V3 全量 61 层, 671B total / 37B active:

```
--num-layers 61 --hidden-size 7168 --ffn-hidden-size 18432
--num-attention-heads 128 --kv-channels 128
--num-experts 256 --moe-router-topk 8 --moe-ffn-hidden-size 2048
--moe-shared-expert-intermediate-size 2048
--moe-layer-freq '([0]*3+[1]*58)'   # 3 dense + 58 MoE
--multi-latent-attention
--q-lora-rank 1536 --kv-lora-rank 512 --qk-head-dim 128
--qk-pos-emb-head-dim 64 --v-head-dim 128
```

## 5. 并行策略

| 维度 | 值 | 说明 |
|---|---|---|
| PP | 4 | 每 stage ~15 层, layout `Etttttttttttttttt\|ttttttttttttttt\|ttttttttttttttt\|tttttttttttttttL` |
| EP | 32 | 每 stage 的 32 GPU 组成 1 个 EP group, 完全在 NVLink 域内 |
| TP | 1 | |
| DP | 1 | 128 / (4×32×1) = 1 |
| GBS | 2048 | |
| MBS | 1 | MBS=2 OOM |

**关键架构**: EP 通信走域内 NVLink (HybridEP CUDA fabric memory), PP p2p 跨域走 RDMA (NCCL)。每个 PP stage 的 32 GPU 全在同一个 NVLink clique 内。

## 6. 核心环境变量

### NCCL (控制集合通信)

```bash
export NCCL_NET=gIB
export NCCL_MNNVL_ENABLE=0          # RDMA only, 避免跨域 MNNVL allreduce hang
export NCCL_CUMEM_ENABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=52
export NCCL_IB_FIFO_TC=84
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_PXN_C2C=1
export NCCL_NVLS_ENABLE=1            # NVLink SHARP, +3% throughput
export NCCL_GRAPH_REGISTER=1         # NCCL graph 注册, +1-2%
export NCCL_SET_STACK_SIZE=1
```

### HybridEP (独立于 NCCL)

```bash
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64  # 每域参与 EP 的 rank 数
export USE_MNNVL=1                                    # HybridEP 用 NVLink fabric
export NVLINK_DOMAIN_SIZE=72                          # NVL72 物理域大小
```

> **关键理解**: `NCCL_MNNVL_ENABLE` 和 `USE_MNNVL` 是两套独立开关。前者控制 NCCL transport, 后者控制 HybridEP 的 CUDA fabric memory。设 `NCCL_MNNVL_ENABLE=0` + `USE_MNNVL=1` 可以让 EP 走 NVLink 而 NCCL 走 RDMA, 绕开跨域 MNNVL allreduce hang (NCCL #2077)。

### TE / cuDNN

```bash
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
```

## 7. 参数实测对比

### Dispatcher 对比

| Dispatcher | NCCL_MNNVL | USE_MNNVL | TFLOPs/GPU | 提升 |
|---|---|---|---|---|
| alltoall (NCCL) | 0 | — | 300 | baseline |
| **HybridEP (flex)** | **0** | **1** | **488** | **+63%** |

### NCCL_ALGO 对比

| 配置 | avg TFLOPs | 峰值 | 最低 | 推荐 |
|---|---|---|---|---|
| NCCL_ALGO=Ring | **434** | 437 | 431 | 生产 (极稳定) |
| 无 NCCL_ALGO (auto) | **473** | 510 | 384 | benchmark |

### NVLS + GRAPH_REGISTER

| 配置 | avg TFLOPs | 峰值 | 说明 |
|---|---|---|---|
| NVLS=0, GRAPH_REG=0 | 473 | 510 | baseline |
| **NVLS=1, GRAPH_REG=1** | **488** | **518** | **+3%, 最低值从 384→466** |

### FP8 Recipe 对比

| Recipe | avg TFLOPs | 说明 |
|---|---|---|
| **blockwise** | **488** | 推荐 (稳定) |
| mxfp8 | 400 | PP=4 下持续衰退, 不适用 |

## 8. 不可行优化 (128 GPU PP=4 限制)

| 优化 | 结果 | 原因 |
|---|---|---|
| CUDA graphs | OOM | graph pool 59GB + HybridEP buffer 60GB + model 65GB > 184GB |
| VPP | 不兼容 | 61 层 / PP=4 不整除 |
| MBS=2 | OOM | activation 翻倍 |
| PP=2 EP=64 | OOM | 每 GPU ~30 层 + HybridEP buffer > 184GB |
| NCCL MNNVL=2 | hang | 跨域 64+ GPU allreduce hang (NCCL #2077) |

## 9. NCCL 跨域 MNNVL 问题

| GPU 数 | 域分布 | MNNVL | 结果 |
|---|---|---|---|
| 8 | 单域 | 2 | pass |
| 32 | 跨域 (4+4) | 2 | pass |
| 64 | 单域 | 2 | pass |
| 64 | 跨域 (8+8) | 2 | **hang** |
| 128 | 跨域 (16+16) | 2 | **hang** |

**根因**: 跨 2 IMEX domain 的 NCCL allreduce 在每域 32+ GPU 时 hang。参考: NCCL #2077, PyTorch #161116。

## 10. 最终推荐配置

### 生产 (稳定优先)

```bash
NCCL_ALGO=Ring  NCCL_NVLS_ENABLE=1  NCCL_GRAPH_REGISTER=1
--fp8-recipe blockwise  --moe-hybridep-num-sms 32
# → 434 TFLOPs/GPU (波动 ±3)
```

### Benchmark (最高吞吐)

```bash
# 不设 NCCL_ALGO
NCCL_NVLS_ENABLE=1  NCCL_GRAPH_REGISTER=1
--fp8-recipe blockwise  --moe-hybridep-num-sms 32
# → 488 TFLOPs/GPU avg, 518 peak
```

## 11. 进一步优化路径

| 需求 | 状态 | 路径 |
|---|---|---|
| PP=8 + VPP | 需 256 GPU | 解锁 overlap-moe + delay-wgrad |
| CUDA graphs | 需更大 HBM 或减小 HybridEP buffer | 代码修改 |
| NCCL MNNVL=2 | 需 NCCL 修复 | 跨域 64+ GPU hang, 等更新 |
| bindpcie | 需 bare-metal | k8s 不兼容 |

## 12. 部署参考

完整的 IMEX 配置、StatefulSet YAML、训练脚本和 Launcher 脚本见源文档。核心架构：
- 2 个 StatefulSet (per NVLink clique), nodeSelector 按 `nvidia.com/gpu.clique` label
- hostNetwork + hostPID, nvidia.com/gpu: 4, privileged: true
- GIB v1.1.2 init container (LD_PRELOAD libnccl.so.2)
- Per-clique IMEX daemon (非 ComputeDomain/DRA)
- Lustre 共享存储放训练脚本和日志
