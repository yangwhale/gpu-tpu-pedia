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
- Kueue ClusterQueue 的 `nominalQuota` 需覆盖总 GPU 数（例如 2 节点需要 16）
- 如果 ResourceFlavor 设了 `topologyName`，所有节点必须在同一 topology block 内 — 或移除拓扑约束

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
    --set "workload.gpus"=16 \
    --set "queue=a4" \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "volumes.nfs.ip"=<filestore-ip> \
    --set "volumes.nfs.region"=<filestore-zone> \
    --set "volumes.nfs.instance"=<filestore-instance> \
    --set "volumes.nfs.volume"=<filestore-share> \
    --set "volumes.nfs.storage"=1024Gi \
    $USER-deepep \
    gke-runtime/jobset
```

关键参数：
- `workload.gpus`：总 GPU 数（必须是 8 的倍数，与可用节点数匹配）
- `queue`：Kueue LocalQueue 名称（必须已存在且 quota 足够）
- `volumes.nfs.*`：Filestore 实例信息（通过 `gcloud filestore instances list` 获取）

### 3. 运行 DeepEP 测试

**方式 A：自动化** — 部署前在 values.yaml 或 `--set` 中设置 `RUN_DEEPEP_TEST=true`，launcher 脚本会通过 MPI 自动运行测试。

**方式 B：手动** — exec 进协调器 Pod：

```bash
COORD_POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$USER-deepep \
    --sort-by=.metadata.name -o jsonpath='{.items[0].metadata.name}')

kubectl exec -it $COORD_POD -c workload -- bash
```

在 Pod 内执行：

```bash
source /opt/deepep/unified-env.sh
source /usr/local/gib/scripts/set_nccl_env.sh

# 生成每节点 1 进程的 hostfile（test_internode.py 自己 spawn 8 GPU 进程）
sed 's/slots=8/slots=1/g' /etc/job-worker-services.txt > /tmp/hostfile-1pernode.txt

mpirun --allow-run-as-root \
  --hostfile /tmp/hostfile-1pernode.txt \
  --mca orte_keep_fqdn_hostnames 1 \
  --mca plm_rsh_agent "ssh -q -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 222" \
  -np $NNODES \
  -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_IBGDA_NIC_HANDLER \
  -x DEEP_EP_DEVICE_TO_HCA_MAPPING -x NVSHMEM_DISABLE_CUDA_VMM \
  -x LD_PRELOAD -x LD_LIBRARY_PATH -x NVSHMEM_IB_GID_INDEX \
  -x NCCL_SOCKET_IFNAME=eth0,eth1 \
  -x NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs/tuner_config_a4.txtpb \
  bash -c "export WORLD_SIZE=$NNODES && export RANK=\$OMPI_COMM_WORLD_RANK && \
    export MASTER_ADDR=$MASTER_ADDR && export MASTER_PORT=29500 && \
    source /opt/deepep/unified-env.sh && source /usr/local/gib/scripts/set_nccl_env.sh && \
    cd /opt/deepep/DeepEP && python3 tests/test_internode.py"
```

预期结果：32/32 测试通过（BF16/FP8 × dispatch/combine × sync/async × with/without top-k）。

> **注意**：`test_internode.py` 使用 `torch.multiprocessing.spawn` 管理 GPU 进程，MPI 只需每节点启动 1 个进程（`slots=1`），不是 8 个。`WORLD_SIZE` 对 DeepEP 的 `init_dist()` 来说是节点数而非 GPU 总数。

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
