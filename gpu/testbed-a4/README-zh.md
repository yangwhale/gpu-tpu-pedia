# A4 GPU 测试平台 - GKE 上的 DeepEP 部署

在 [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine) 上部署和测试 [DeepEP](https://github.com/deepseek-ai/DeepEP)（DeepSeek Expert Parallelism），使用 A4 (B200) GPU。

## 包含内容

- **Docker 镜像** (`nemo-deepep:25.11`)：NeMo 25.11 + NVSHMEM v3.5.19-1 (IBGDA) + DeepEP GPU-NIC 映射
- **Cloud Build 配置**：一条命令构建镜像
- **Helm chart**：基于 JobSet 的多节点部署，支持拓扑感知调度
- **启动脚本**：自动 SSH 配置、节点发现、物理拓扑排序

## 技术栈

- **基础镜像**：`nvcr.io/nvidia/nemo:25.11`
- **NVSHMEM**：v3.5.19-1，启用 IBGDA（GPU 直接发起 RDMA）
- **DeepEP**：PR #466（GPU-NIC 显式映射），commit `8a07e7e`
- **GPU 架构**：SM 10.0 (B200)
- **编排**：GKE + Kubernetes JobSet + Helm

## 前提条件

- GKE 集群，A4 节点池（DENSE 部署类型）
- Artifact Registry（存储 Docker 镜像）
- GCS bucket（存储日志）
- 已安装 Kueue + JobSet API

## 快速开始

### 1. 构建镜像

```bash
cd gpu-tpu-pedia/gpu/testbed-a4/docker

export ARTIFACT_REGISTRY=<你的 registry URL>
# 例如 asia-docker.pkg.dev/your-project/your-repo

gcloud builds submit \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet --async
```

镜像 tag：`${ARTIFACT_REGISTRY}/nemo-deepep:25.11`

### 2. 部署到 GKE

```bash
export CLUSTER_NAME=<集群名>
export REGION=<区域>
export GCS_BUCKET=<bucket 名>
export WORKLOAD_IMAGE=${ARTIFACT_REGISTRY}/nemo-deepep:25.11

gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

cd gpu-tpu-pedia/gpu/testbed-a4
helm install -f gke-runtime/values.yaml \
    --set-file workload_launcher=gke-runtime/launchers/torchrun-stratup.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    $USER-deepep \
    gke-runtime/jobset
```

### 3. 运行 DeepEP 测试

进入协调器 Pod 执行：

```bash
# 加载 DeepEP 运行时环境
source /opt/deepep/unified-env.sh

# 运行 internode 测试
cd /opt/deepep/DeepEP
python3 tests/test_internode.py
```

预期结果：32/32 测试通过（BF16/FP8 × dispatch/combine × sync/async × with/without top-k）。

## 目录结构

```
testbed-a4/
├── docker/
│   ├── testbed.Dockerfile      # DeepEP 镜像：NeMo + NVSHMEM + DeepEP
│   ├── cloudbuild.yml          # Cloud Build 配置
│   └── README.md               # Docker 构建详情
└── gke-runtime/
    ├── values.yaml             # Helm 配置（GPU 数量、镜像、存储卷）
    ├── jobset/                 # Helm chart 模板
    └── launchers/
        └── torchrun-stratup.sh # 节点发现、SSH 配置、拓扑排序
```

## DeepEP 运行时关键环境变量

由 `/opt/deepep/unified-env.sh` 自动设置：

| 变量 | 值 | 用途 |
|------|-----|------|
| `NVSHMEM_REMOTE_TRANSPORT` | `ibgda` | GPU 直接发起 RDMA |
| `NVSHMEM_IBGDA_NIC_HANDLER` | `gpu` | GPU 直接操作网卡 |
| `DEEP_EP_DEVICE_TO_HCA_MAPPING` | `0:mlx5_0:1,...` | GPU-NIC 亲和性映射 |
| `NVSHMEM_DISABLE_CUDA_VMM` | `1` | IBGDA 必需 |
| `LD_PRELOAD` | `libnvshmem_host.so.3` | NVSHMEM 运行时 |

## 监控

```bash
kubectl get pods | grep deepep
kubectl logs -f <coordinator-pod>
kubectl get jobset
```

## 清理

```bash
helm uninstall $USER-deepep
```
