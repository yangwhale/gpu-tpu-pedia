# A4 GPU GKE 测试平台 - Docker 构建和 NCCL 测试

本指南提供了在 [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine) 上运行 A4 GPU 工作负载的综合测试平台。包括自动化 Docker 镜像构建、基于 Helm 的部署和 NCCL 性能测试功能。

## 概述

该测试平台旨在：
- 使用 Google Cloud Build 为 A4 GPU 工作负载构建自定义 Docker 镜像
- 使用 Helm Chart 和 JobSet 在 GKE 上部署分布式工作负载
- 运行 NCCL 性能测试以验证 GPU 通信
- 为开发和测试 A4 GPU 应用程序提供基础平台

## 编排和部署工具

本指南使用以下技术栈：

- **编排平台** - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- **容器构建** - [Google Cloud Build](https://cloud.google.com/build) 配合自定义 Dockerfile
- **作业管理** - 通过 Helm Chart 部署的 [Kubernetes JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
- **GPU 通信** - 配备 gIB 插件的 NCCL，针对 A4 性能优化

## 测试环境

本指南已在以下配置下进行优化和测试：

### GKE 集群要求
- [区域标准集群](https://cloud.google.com/kubernetes-engine/docs/concepts/configuration-overview)，版本：1.31.7-gke.1265000 或更高版本
- GPU 节点池：使用 DENSE 部署类型的 [a4-highgpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-high-vms) 节点
- 启用 [GKE 工作负载身份联合](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity)
- 启用 [GKE Cloud Storage FUSE CSI 驱动程序](https://cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver)
- 启用 [DCGM 指标](https://cloud.google.com/kubernetes-engine/docs/how-to/dcgm-metrics)
- 安装 [Kueue](https://kueue.sigs.k8s.io/docs/reference/kueue.v1beta1/) 和 [JobSet](https://jobset.sigs.k8s.io/docs/overview/) API
- 配置 Kueue 支持[拓扑感知调度](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/)

### 存储要求
- 区域性 Google Cloud Storage (GCS) 存储桶，用于存储日志和工件
- Google Artifact Registry，用于存储自定义 Docker 镜像

要准备所需环境，请参阅 [GKE 环境设置指南](../../../../docs/configuring-environment-gke-a4.md)。

## Docker 容器镜像

本指南构建针对 A4 GPU 工作负载优化的自定义 Docker 镜像：

**基础镜像**：`us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo25.04-gib1.0.6-A4`

镜像包含：
- NVIDIA NeMo 25.04 框架
- 针对 A4 GPU 优化的 NCCL gIB 插件 v1.0.6
- 分布式训练和测试所需的所有依赖项

## 主要特性

### 自动化 Docker 构建
- 使用 Google Cloud Build 进行可扩展的镜像构建
- 通过替换参数配置构建参数
- 高性能构建机器 (e2-highcpu-32)
- 异步构建过程，超时时间为 2 小时

### 分布式作业管理
- 基于 Helm 的部署，支持可定制的配置值
- 使用 JobSet 管理多节点工作负载
- 自动节点发现和 SSH 配置
- 物理拓扑感知调度

### NCCL 性能测试
- 多节点 NCCL all-reduce 性能测试
- 针对 A4 GPU 优化的 NCCL 设置
- 全面的通信参数调优
- 跨不同数据大小的性能验证

## 运行指南

在客户端工作站上完成以下步骤：

### 配置环境变量

设置环境变量以匹配您的环境：

```bash
export PROJECT_ID=<PROJECT_ID>
export REGION=<REGION>
export CLUSTER_NAME=<CLUSTER_NAME>
export GCS_BUCKET=<GCS_BUCKET>
export KUEUE_NAME=<KUEUE_NAME>
export ARTIFACT_REGISTRY=<ARTIFACT_REGISTRY>
```

替换以下值：
- `<PROJECT_ID>`：您的 Google Cloud 项目 ID
- `<REGION>`：集群所在的区域（例如：us-central1）
- `<CLUSTER_NAME>`：GKE 集群的名称
- `<GCS_BUCKET>`：Cloud Storage 存储桶的名称（不包含 `gs://` 前缀）
- `<KUEUE_NAME>`：Kueue 本地队列的名称（默认：`a4-high`）
- `<ARTIFACT_REGISTRY>`：您的 Artifact Registry URL

设置默认项目：

```bash
gcloud config set project $PROJECT_ID
```

### 获取代码仓库

克隆 `gpu-recipes` 仓库并设置指向配方文件夹的引用：

```bash
git clone -b a4-early-access https://github.com/yangwhale/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a4/testbed
cd $RECIPE_ROOT
```

### 构建自定义 Docker 镜像

使用 Google Cloud Build 构建自定义 Docker 镜像：

```bash
cd $REPO_ROOT/training/a4/testbed/docker
gcloud builds submit --region=${REGION} \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet \
    --async
```

**注意**：构建过程异步运行。您可以在 Google Cloud Console 中监控构建状态，或使用 `gcloud builds list` 检查进度。

### 获取集群凭据

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION
```

### 部署和运行测试平台

使用 Helm 部署测试平台工作负载：

```bash
cd $RECIPE_ROOT
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$REPO_ROOT/training/a4/testbed/gke-runtime/launchers/torchrun-stratup.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    $USER-testbed \
    $REPO_ROOT/training/a4/testbed/gke-runtime/jobset
```

其中 `$WORKLOAD_IMAGE` 应设置为您构建的镜像：
```bash
export WORKLOAD_IMAGE=us-central1-docker.pkg.dev/supercomputer-testing/chrisya-docker-repo-supercomputer-testing-uc1/testbed:nemo25.04-gib1.0.6-A4
```

## 测试和验证

测试平台自动运行以下测试：

### 1. 基本连通性测试
验证所有节点可以通过 MPI 通信：
```bash
mpirun --allow-run-as-root -np 16 -hostfile /etc/job-worker-services.txt \
--mca orte_keep_fqdn_hostnames 1 hostname
```

### 2. NCCL 性能测试
运行全面的 NCCL all-reduce 性能测试：
```bash
mpirun --allow-run-as-root \
--hostfile /etc/job-worker-services.txt \
-wdir /third_party/nccl-tests \
-mca plm_rsh_no_tree_spawn 1 \
--mca orte_keep_fqdn_hostnames 1 \
--map-by slot \
--mca plm_rsh_agent "ssh -q -o LogLevel=ERROR -o StrictHostKeyChecking=no" \
bash -c "source /tmp/export_init_env.sh && ./build/all_reduce_perf -b 2M -e 16G -f 2 -n 1 -g 1 -w 10"
```

## 监控作业

要检查作业中 Pod 的状态：

```bash
kubectl get pods | grep $USER-testbed
```

要获取特定 Pod 的日志：

```bash
kubectl logs POD_NAME
```

要跟踪主协调器 Pod 的日志：

```bash
kubectl logs -f $USER-testbed-workload-0-0-xxxxx
```

## 结果分析

### NCCL 性能指标

NCCL 测试输出性能指标，包括：
- **带宽**：数据传输速率（GB/s）
- **延迟**：通信延迟（微秒）
- **效率**：理论峰值性能的百分比

示例输出：
```
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
     2097152        524288     float     sum      -1    123.4   17.01   31.89      0    123.4   17.01   31.89      0
     4194304       1048576     float     sum      -1    234.5   17.89   33.54      0    234.5   17.89   33.54      0
```

### 日志收集

日志自动收集到配置的 GCS 存储桶中：

```
gs://${GCS_BUCKET}/testbed-logs/
├── nccl-test-results.txt
├── connectivity-test.log
└── pod-logs/
    ├── coordinator.log
    └── worker-*.log
```

## 故障排除

### 常见问题

1. **构建失败**
   - 检查 Cloud Build 日志：`gcloud builds list`
   - 验证 Artifact Registry 权限
   - 确保有足够的构建配额

2. **Pod 启动问题**
   - 检查 Pod 状态：`kubectl describe pod POD_NAME`
   - 验证镜像拉取权限
   - 检查节点资源可用性

3. **NCCL 通信失败**
   - 验证节点间网络连通性
   - 检查 gIB 插件安装
   - 查看 NCCL 调试日志

4. **SSH 连接问题**
   - 验证 SSH 公钥配置
   - 检查 Pod 中的 SSH 服务启动
   - 确保正确的网络策略

### 调试命令

```bash
# 检查 JobSet 状态
kubectl get jobset

# 检查 Pod 事件
kubectl get events --sort-by=.metadata.creationTimestamp

# 检查节点 GPU 状态
kubectl describe nodes -l cloud.google.com/gke-accelerator=nvidia-a4-high-8g

# 手动测试 NCCL
kubectl exec -it POD_NAME -- /third_party/nccl-tests/build/all_reduce_perf -b 1M -e 1M -i 1
```

## 自定义配置

### 环境变量

测试平台在 [`values.yaml`](training/a4/testbed/gke-runtime/values.yaml:1) 中支持以下环境变量：

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `SSH_PUBLIC_KEY` | 预配置 | 用于节点间通信的 SSH 公钥 |
| `SLEEP_INFINITY` | `true` | 保持容器运行以便调试 |
| `HF_TOKEN` | 预配置 | HuggingFace 模型访问令牌 |

### 扩展配置

要修改 GPU 或节点数量，请更新 [`values.yaml`](training/a4/testbed/gke-runtime/values.yaml:46)：

```yaml
workload:
  gpus: 32  # GPU 总数（必须是 8 的倍数）
```

### 自定义启动脚本

您可以通过修改 Helm 命令中的 `--set-file workload_launcher` 参数来提供自定义启动脚本。

## 卸载 Helm 发布

要清理测试平台资源：

```bash
helm uninstall $USER-testbed
```

## 后续步骤

该测试平台为以下用途提供基础：
- 开发自定义 A4 GPU 应用程序
- 性能基准测试和优化
- 分布式训练工作负载开发
- NCCL 通信模式分析

对于更高级的用例，请考虑：
- 与 MLOps 流水线集成
- 添加自定义性能指标收集
- 实施自动扩展策略
- 开发特定应用的测试套件