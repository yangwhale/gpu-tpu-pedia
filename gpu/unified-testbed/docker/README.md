# Unified Testbed Docker Image for A4 GPU

基于 `nvcr.io/nvidia/pytorch:25.06-py3` 构建的统一测试环境镜像，用于在 Google A4 GPU 上进行训练和测试。

## 特性

- 基于 NVIDIA PyTorch 25.06 官方容器
- 为 A4 GPU (NVIDIA B200) 架构优化
- 包含 GCSfuse 组件，用于 Google Cloud Storage 集成
- 包含 Google Cloud CLI 工具，用于云资源管理
- 预配置了 2222 端口的 SSH，用于多节点通信
- 包含 dllogger、tensorboard、wandb 等训练监控工具
- 支持分布式训练的 SSH 密钥配置

## 文件说明

| 文件 | 说明 |
|------|------|
| [`unified-testbed.Dockerfile`](unified-testbed.Dockerfile) | 主要的 Dockerfile |
| [`cloudbuild.yml`](cloudbuild.yml) | Google Cloud Build 配置文件 |

## 构建镜像

### 使用 Google Cloud Build（推荐）

```bash
# 设置环境变量
export REGION=asia-southeast1
export ARTIFACT_REGISTRY=asia-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO

# 提交构建任务
cd gpu/unified-testbed/docker
gcloud builds submit --region=${REGION} \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet \
    --async
```

### 本地构建

```bash
docker build -t unified-testbed-pytorch:25.06-py3 -f unified-testbed.Dockerfile .
```

## 镜像标签

构建完成后，镜像将被标记为：

```
${ARTIFACT_REGISTRY}/unified-testbed-pytorch:25.06-py3
```

## 技术细节

### 基础镜像

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.06-py3
```

### 安装的组件

| 组件 | 用途 |
|------|------|
| GCSfuse | Google Cloud Storage 挂载 |
| Google Cloud CLI | 云资源管理 |
| dllogger | NVIDIA 训练指标记录 |
| tensorboard | 训练可视化 |
| wandb | 实验跟踪 |
| hydra-core | 配置管理 |
| OpenSSH | 多节点通信 |

### SSH 配置

- 端口：2222（避免与主机 SSH 冲突）
- 自动生成 RSA 密钥对
- 禁用 StrictHostKeyChecking
- 支持 root 登录

### 工作目录

| 路径 | 用途 |
|------|------|
| `/workspace` | 默认工作目录 |
| `/workload/launcher` | 启动脚本 |
| `/workload/scripts` | 用户脚本 |
| `/workload/configs` | 配置文件 |
| `/gcs` | GCS 挂载点 |
| `/mnt` | NFS 挂载点 |
| `/ssd` | 本地 SSD 挂载点 |

## GIB Sidecar 镜像

运行时需要配合 GIB (Google InfiniBand) sidecar 镜像使用：

```
us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic:latest
```

该 sidecar 提供：

- NCCL gIB 插件
- GPU Direct RDMA 支持
- A4 GPU 优化的 NCCL 配置

## 许可证

Copyright 2025 Google LLC

根据 Apache License 2.0 许可证授权。

## 故障排除

### 常见问题

1. **构建超时**
   - 增加 `--timeout` 参数值
   - 使用更高配置的构建机器

2. **权限问题**
   确保具有以下权限：
   - Cloud Build Editor
   - Artifact Registry Writer

3. **网络连接问题**
   - 确保可以访问 `nvcr.io`
   - 检查防火墙规则

### 调试命令

```bash
# 查看构建列表
gcloud builds list --limit=10

# 查看构建日志
gcloud builds log [BUILD_ID]

# 测试镜像
docker run --gpus all -it unified-testbed-pytorch:25.06-py3 nvidia-smi
```

## 相关资源

- [NVIDIA NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [Google Cloud Build 文档](https://cloud.google.com/build/docs)
- [Artifact Registry 文档](https://cloud.google.com/artifact-registry/docs)
- [GCSfuse 文档](https://cloud.google.com/storage/docs/gcs-fuse)
