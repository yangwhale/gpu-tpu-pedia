# A4 Unified Testbed

统一测试环境，用于在 Google A4 GPU 上进行训练和测试工作负载。融合了 `testbed` 的简洁性和 `alibaba-pai-megatron-patch` 的完整工作流能力。

## 概述

Unified Testbed 提供：

- 基于 NVIDIA PyTorch 25.06 的通用 Docker 镜像
- 灵活的 Helm Chart，支持多种部署模式
- 完整的分布式训练环境初始化
- 可选的 NFS 共享存储支持
- 多种示例脚本（NCCL 测试、DDP 测试、Pai-Megatron 训练）

## 目录结构

```
unified-testbed/
├── docker/                          # Docker 镜像构建
│   ├── unified-testbed.Dockerfile   # 主 Dockerfile
│   ├── cloudbuild.yml               # Cloud Build 配置
│   └── README.md                    # Docker 说明文档
├── gke-runtime/                     # GKE 运行时配置
│   ├── values.yaml                  # Helm Chart 默认值
│   ├── jobset/                      # Helm Chart
│   │   ├── Chart.yaml
│   │   └── templates/
│   │       ├── workload-job.yaml
│   │       ├── workload-launcher-configmap.yaml
│   │       ├── workload-svc.yaml
│   │       ├── workload-nfs.yaml
│   │       └── workload-lustre.yaml
│   └── launchers/                   # 启动器脚本
│       └── torchrun-startup.sh
├── examples/                        # 示例脚本
│   ├── nccl-test.sh                 # NCCL 性能测试
│   ├── torchrun-ddp-test.sh         # PyTorch DDP 测试
│   ├── pai-megatron-qwen3.sh        # Pai-Megatron Qwen3-30B-A3B 训练
│   └── pai-megatron-qwen3-next.sh   # Pai-Megatron Qwen3-Next-80B-A3B 训练
└── README.md                        # 本文档
```

## 测试环境要求

### GKE 集群要求

- [Regional standard cluster](https://cloud.google.com/kubernetes-engine/docs/concepts/configuration-overview) 版本: 1.31.7-gke.1265000 或更高
- GPU 节点池配置 [a4-highgpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-high-vms) (DENSE 部署类型)
- [Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) 已启用
- [Cloud Storage FUSE CSI driver for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver) 已启用
- [Kueue](https://kueue.sigs.k8s.io/docs/reference/kueue.v1beta1/) 和 [JobSet](https://jobset.sigs.k8s.io/docs/overview/) APIs 已安装

### 存储要求

- Google Cloud Storage (GCS) bucket 用于日志和工件
- Google Artifact Registry 用于 Docker 镜像
- **Filestore NFS (推荐)** 用于多节点共享数据、模型和检查点

## 存储架构说明

### 多节点数据共享机制

在多节点分布式训练中，**NFS 共享存储**是关键组件，用于存放模型、数据集和检查点。所有节点都挂载同一个 NFS 卷到 `/mnt` 路径，实现数据共享。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Google Filestore (NFS)                               │
│                              /mnt (共享挂载点)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  /mnt/                                                                       │
│  ├── Pai-Megatron-Patch/          # 训练框架代码 (Rank 0 克隆)              │
│  ├── ckpts/                                                                  │
│  │   ├── huggingface/             # HuggingFace 格式模型 (Rank 0 下载)      │
│  │   │   └── Qwen3-Next-80B-A3B-Instruct/                                   │
│  │   └── mcore/                   # MCore 格式检查点 (所有节点参与转换)      │
│  │       └── Qwen3-Next-80B-A3B-Instruct-to-mcore/                          │
│  ├── datasets/                    # 训练数据集 (Rank 0 下载)                 │
│  │   ├── mmap_qwen3_datasets_text_document.bin                              │
│  │   ├── mmap_qwen3_datasets_text_document.idx                              │
│  │   └── alpaca_zh-*.json                                                   │
│  ├── logs/                        # 训练日志和输出                           │
│  │   └── output_mcore_qwen3_next_pretrain/                                  │
│  └── sync_flags_${JOB_ID}/        # 节点同步标志                            │
│      └── download_complete_flag                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Node 0  │   │ Node 1  │   │ Node 2  │   │ Node 3  │
    │(Rank 0) │   │(Rank 1) │   │(Rank 2) │   │(Rank 3) │
    │  下载   │   │  等待   │   │  等待   │   │  等待   │
    │  转换   │   │  同步   │   │  同步   │   │  同步   │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

### 数据流程详解

1. **Rank 0 节点负责**：
   - 克隆代码仓库到 `/mnt/Pai-Megatron-Patch/`
   - 下载模型到 `/mnt/ckpts/huggingface/`
   - 下载数据集到 `/mnt/datasets/`
   - 创建同步标志 `/mnt/sync_flags_${JOB_ID}/download_complete_flag`

2. **其他节点等待**：

   ```bash
   while [[ ! -f $SYNC_DIR/download_complete_flag ]]; do
     sleep 5
   done
   ```

3. **检查点转换（分布式）**：
   - 所有节点参与转换，每个节点处理部分参数
   - 转换结果保存到 `/mnt/ckpts/mcore/`

4. **训练阶段**：
   - 所有节点从 NFS 读取相同的数据和检查点
   - 日志和输出保存到 `/mnt/logs/`

### 存储选项对比

| 存储类型 | 用途 | 性能 | 共享 | 适用场景 |
|---------|------|------|------|----------|
| **NFS (Filestore)** | 模型、数据、检查点 | 中等 | ✅ 多节点 | 训练工作流 |
| **Managed Lustre** | 高性能数据集 | 很高 | ✅ 多节点 | 大规模训练 |
| **GCS (gcsfuse)** | 日志、工件备份 | 低 | ✅ 多节点 | 日志收集 |
| **Local SSD** | 临时缓存 | 高 | ❌ 单节点 | 高速缓存 |

### NFS 配置示例

在 `values.yaml` 中配置 NFS：

```yaml
volumes:
  nfs:
    enabled: true                    # 启用 NFS
    ip: "10.x.x.x"                   # Filestore IP
    volume: "nfslarge"               # 共享卷名称
    region: "us-central1"            # 区域
    instance: "my-filestore"         # 实例名
    storage: "10Ti"                  # 存储容量
  
  pvcMounts:
  - claimName: nfs-pvc              # PVC 名称
    mountPath: "/mnt"                # 容器内挂载路径
```

### Managed Lustre 配置示例

本项目支持使用 [Managed Lustre CSI 驱动程序](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-lustre-csi-driver) 访问 GKE 上的 Managed Lustre 实例。

#### 前提条件

1. 启用 Managed Lustre CSI 驱动程序：

```bash
gcloud container clusters update ${CLUSTER_NAME} \
    --location=${LOCATION} \
    --enable-lustre-csi-driver \
    --enable-legacy-lustre-port
```

1. 确保 GKE 集群和 Lustre 实例在同一个 VPC 网络中。

#### 查找 Lustre 实例信息

```bash
gcloud lustre instances list \
    --project=${PROJECT_ID} \
    --location=${LOCATION}
```

输出示例：

```
capacityGib: '9000'
filesystem: lustrefs
mountPoint: 172.27.48.5@tcp:/lustrefs
name: projects/my-project/locations/us-central1-a/instances/my-lustre
state: ACTIVE
```

#### 配置 Lustre

在 `values.yaml` 中配置 Lustre：

```yaml
volumes:
  lustre:
    enabled: true                         # 启用 Lustre
    ip: "172.27.48.5"                     # Lustre 实例 IP
    filesystem: "lustrefs"                # 文件系统名称
    instanceName: "my-lustre"             # 实例名称
    projectId: "my-project"               # 项目 ID
    location: "us-central1-a"             # 可用区
    storage: "9000Gi"                     # 存储容量
    storageClassName: ""                  # 可留空
    mountPath: "/lustre"                  # 容器内挂载路径
```

#### Lustre 部署命令

```bash
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/pai-megatron-qwen3-next.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.lustre.enabled"=true \
    --set "volumes.lustre.ip"="172.27.48.5" \
    --set "volumes.lustre.filesystem"="lustrefs" \
    --set "volumes.lustre.instanceName"="my-lustre" \
    --set "volumes.lustre.projectId"="$PROJECT_ID" \
    --set "volumes.lustre.location"="us-central1-a" \
    --set "workload.envs[1].value"="false" \
    $USER-qwen3-lustre \
    $RECIPE_ROOT/gke-runtime/jobset
```

## 快速开始

### 1. 环境配置

```bash
# 设置环境变量
export PROJECT_ID=<YOUR_PROJECT_ID>
export REGION=us-central1
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export GCS_BUCKET=<YOUR_GCS_BUCKET>
export KUEUE_NAME=a4-high
export ARTIFACT_REGISTRY=$REGION-docker.pkg.dev/$PROJECT_ID/<YOUR_REPO>

# 设置默认项目
gcloud config set project $PROJECT_ID

# 获取集群凭证
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION
```

### 2. 克隆仓库

```bash
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-tpu-pedia
export REPO_ROOT=$(git rev-parse --show-toplevel)
export RECIPE_ROOT=$REPO_ROOT/gpu/unified-testbed
```

### 3. 构建 Docker 镜像

```bash
cd $RECIPE_ROOT/docker
gcloud builds submit --region=${REGION} \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet \
    --async
```

### 4. 部署测试环境

#### 基础模式（交互式调试）

```bash
export WORKLOAD_IMAGE=$ARTIFACT_REGISTRY/unified-testbed-pytorch:25.06-py3

helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/gke-runtime/launchers/torchrun-startup.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    $USER-testbed \
    $RECIPE_ROOT/gke-runtime/jobset
```

#### NCCL 测试模式

```bash
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/nccl-test.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "workload.envs[1].value"="false" \
    $USER-nccl-test \
    $RECIPE_ROOT/gke-runtime/jobset
```

#### DDP 测试模式

```bash
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/torchrun-ddp-test.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "workload.envs[1].value"="false" \
    $USER-ddp-test \
    $RECIPE_ROOT/gke-runtime/jobset
```

#### Pai-Megatron Qwen3 训练模式（需要 NFS）

Qwen3-30B-A3B 模型训练：

```bash
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/pai-megatron-qwen3.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "volumes.nfs.enabled"=true \
    --set "volumes.nfs.ip"=<FILESTORE_IP> \
    --set "workload.envs[1].value"="false" \
    $USER-qwen3-training \
    $RECIPE_ROOT/gke-runtime/jobset
```

#### Pai-Megatron Qwen3-Next 训练模式（需要 NFS）

Qwen3-Next-80B-A3B-Instruct 模型训练（支持预训练和 SFT 微调）：

```bash
# 预训练模式
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/pai-megatron-qwen3-next.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "volumes.nfs.enabled"=true \
    --set "volumes.nfs.ip"=<FILESTORE_IP> \
    --set "workload.envs[1].value"="false" \
    $USER-qwen3-next-training \
    $RECIPE_ROOT/gke-runtime/jobset

# SFT 微调模式
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/pai-megatron-qwen3-next.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "volumes.nfs.enabled"=true \
    --set "volumes.nfs.ip"=<FILESTORE_IP> \
    --set "workload.envs[1].value"="false" \
    --set "workload.envs[10].name"="TRAINING_MODE" \
    --set "workload.envs[10].value"="sft" \
    $USER-qwen3-next-sft \
    $RECIPE_ROOT/gke-runtime/jobset
```

## 配置参数

### 主要配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `workload.gpus` | 32 | 总 GPU 数量（必须是 8 的倍数） |
| `workload.image` | - | 容器镜像地址 |
| `workload.envs[0].value` (SSH_PUBLIC_KEY) | "" | SSH 公钥 |
| `workload.envs[1].value` (SLEEP_INFINITY) | "true" | 保持容器运行 |
| `volumes.nfs.enabled` | false | 是否启用 NFS |
| `volumes.nfs.ip` | "" | Filestore IP |
| `network.hostNetwork` | true | 使用主机网络 |
| `network.gibVersion` | latest | GIB 插件版本 |

### 工作流控制环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SKIP_ENV_SETUP` | "false" | 跳过环境准备（修复 triton、安装依赖） |
| `SKIP_CLONE_REPO` | "false" | 跳过代码克隆 |
| `SKIP_DOWNLOAD_DATA` | "false" | 跳过数据下载 |
| `SKIP_CHECKPOINT_CONVERSION` | "false" | 跳过检查点转换 |
| `SKIP_TRAINING` | "false" | 跳过训练 |
| `TRAINING_MODE` | "pretrain" | 训练模式：pretrain（预训练）或 sft（微调） |
| `USE_JSON_SFT` | "false" | 使用 JSON 格式数据进行 SFT（仅 Qwen3-Next） |

### 环境准备说明

由于 Unified Testbed 使用 `nvcr.io/nvidia/pytorch:25.06-py3` 基础镜像（而非原始的 pai-megatron-patch 镜像），需要在运行时进行环境准备：

```bash
# Step 0: 环境准备（自动执行）
# 1. 修复 triton ldconfig 路径问题
sed -i 's|libs = subprocess.check_output(\["ldconfig"|libs = subprocess.check_output(["/sbin/ldconfig"|g' \
    /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py

# 2. 升级 NCCL 库
pip install --upgrade nvidia-nccl-cu12 -q

# 3. 安装 Pai-Megatron-Patch 依赖
pip install datasets==3.6.0 packaging==24.2 modelscope -q
```

如果已在镜像中完成这些准备工作，可以设置 `SKIP_ENV_SETUP=true` 跳过此步骤。

## 监控和调试

### 查看 Pod 状态

```bash
kubectl get pods | grep $USER-testbed
```

### 查看日志

```bash
# 查看主节点日志
kubectl logs -f $USER-testbed-workload-0-0-xxxxx

# 进入容器调试
kubectl exec -it $USER-testbed-workload-0-0-xxxxx -- bash
```

### 查看 JobSet 状态

```bash
kubectl get jobset
kubectl describe jobset $USER-testbed
```

## GIB Sidecar 配置

本测试环境使用 GIB (Google InfiniBand) sidecar 提供 NCCL 优化：

```
us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic:latest
```

### NCCL 优化参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `NCCL_NET` | gIB | 使用 gIB 插件 |
| `NCCL_SOCKET_IFNAME` | eth0,eth1 | 网络接口 |
| `NCCL_TUNER_CONFIG_PATH` | tuner_config_a4.txtpb | A4 优化配置 |

## 清理资源

```bash
# 删除 Helm Release
helm uninstall $USER-testbed
helm uninstall $USER-nccl-test
helm uninstall $USER-ddp-test
helm uninstall $USER-qwen3-training
```

## 故障排除

### 常见问题

1. **Pod 启动失败**

   ```bash
   kubectl describe pod <POD_NAME>
   kubectl get events --sort-by=.metadata.creationTimestamp
   ```

2. **SSH 连接失败**
   - 检查 SSH 公钥是否正确配置
   - 确保 2222 端口可访问

3. **NCCL 通信失败**
   - 检查 GIB 插件是否正确安装
   - 验证网络接口配置

4. **NFS 挂载失败**
   - 确认 Filestore IP 正确
   - 检查 NFS 权限设置

### 调试命令

```bash
# 检查 GPU 状态
kubectl exec -it <POD_NAME> -- nvidia-smi

# 检查 NCCL 配置
kubectl exec -it <POD_NAME> -- cat /usr/local/gib/scripts/set_nccl_env.sh

# 手动运行 NCCL 测试
kubectl exec -it <POD_NAME> -- /third_party/nccl-tests/build/all_reduce_perf -b 1M -e 1M
```

## 许可证

Copyright 2025 Google LLC

根据 Apache License 2.0 许可证授权。

## 相关资源

- [NVIDIA NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [GKE JobSet 文档](https://cloud.google.com/kubernetes-engine/docs/how-to/jobset)
- [Kueue 文档](https://kueue.sigs.k8s.io/docs/)
