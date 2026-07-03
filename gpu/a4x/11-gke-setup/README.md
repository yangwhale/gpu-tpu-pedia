# A4X GB200 NVL72 on GKE (Google Kubernetes Engine)

GKE 原生方式部署 A4X GPU 集群，对比自建 K8s（01-03 章节）。

**实测环境**：`supercomputer-testing` 项目，europe-west4-b，2 台 `a4x-highgpu-4g`（GB200），K8s 1.36。

## GKE vs 自建 K8s

| 维度 | GKE 原生 | 自建 K8s (kubeadm) |
|---|---|---|
| 集群创建 | 一条 gcloud 命令 | 手动 kubeadm init/join |
| GPU 驱动 | GKE 自动安装 | 镜像预装 |
| GPU Stack (DRA/DRANET) | GKE 自动配置 | 手动部署 |
| RDMA 网络 | 预建 RDMA VPC + additionalNodeNetworkConfigs | 手动创建 MRDMA VPC + 子网 |
| ComputeDomain / IMEX | GKE 自动管理 | 手动创建 + 启动 IMEX |
| 节点扩缩 | Node Pool autoscaling | 手动增删 VM |
| 适合场景 | 生产环境、多租户 | 性能调优、开发测试 |

## 前提条件

- GCP 项目，已启用 GKE API + Compute API
- A4X Reservation（`specificReservationRequired: true`，`shareType: LOCAL`）
- 足够的 quota（GPU、CPU、IP）

> **Reservation 注意**：`shareType: LOCAL` 的 reservation 可以被同项目 GKE 直接消费。`shareType: SPECIFIC_PROJECTS` 的 reservation 可能需要额外权限配置。

## 架构概览

A4X GKE 每个节点需要 **6 个网卡**，对应 3 种 VPC：

| 网络类型 | 数量 | 用途 |
|---|---|---|
| 主 GVNIC (集群网络) | 1 | GKE 管理流量、节点/Pod/Service IP |
| 额外 GVNIC | 1 | 第二管理网卡 |
| RDMA (vpc-roce) | 4 | GPU 间高速 RDMA 通信 |

## Step 1: VPC 网络

需要创建 3 个 VPC。RDMA VPC 必须绑定 `vpc-roce` network profile，且每个 region 需要独立的 RDMA VPC（profile 是 zone 级别的，不能跨 region 共用）。

```bash
PROJECT=supercomputer-testing
REGION=europe-west4
ZONE=europe-west4-b

# === 主管理 VPC（可跨 region 复用）===
gcloud compute networks create chrisya-gke-mgmt \
  --subnet-mode=custom --mtu=8244 --project=$PROJECT

gcloud compute networks subnets create chrisya-gke-sub-${REGION} \
  --network=chrisya-gke-mgmt --region=$REGION \
  --range=10.51.0.0/16 \
  --secondary-range=pods=10.64.0.0/14,services=10.68.0.0/20 \
  --project=$PROJECT

# === 额外 GVNIC（可跨 region 复用）===
gcloud compute networks create chrisya-gke-net-1 \
  --subnet-mode=custom --mtu=8244 --project=$PROJECT

gcloud compute networks subnets create chrisya-gke-sub-1-${REGION} \
  --network=chrisya-gke-net-1 --region=$REGION \
  --range=10.61.0.0/18 \
  --project=$PROJECT

# === RDMA VPC（每 region 独立，绑 vpc-roce profile）===
gcloud compute networks create chrisya-gke-rdma-${REGION} \
  --subnet-mode=custom --mtu=8896 \
  --network-profile=projects/$PROJECT/global/networkProfiles/${ZONE}-vpc-roce \
  --project=$PROJECT

for i in 0 1 2 3; do
  BASE=$((192 + i * 16))
  gcloud compute networks subnets create chrisya-gke-rdma-${REGION}-sub-${i} \
    --network=chrisya-gke-rdma-${REGION} --region=$REGION \
    --range=192.168.${BASE}.0/20 \
    --project=$PROJECT
done

# === 防火墙 ===
gcloud compute firewall-rules create chrisya-gke-allow-internal \
  --network=chrisya-gke-mgmt --allow=tcp,udp,icmp \
  --source-ranges=10.50.0.0/8 --project=$PROJECT

gcloud compute firewall-rules create chrisya-gke-allow-ssh \
  --network=chrisya-gke-mgmt --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 --project=$PROJECT

for NET in chrisya-gke-net-1 chrisya-gke-rdma-${REGION}; do
  gcloud compute firewall-rules create ${NET}-allow-internal \
    --network=$NET --allow=tcp,udp,icmp \
    --source-ranges=0.0.0.0/0 --project=$PROJECT
done
```

### 网段规划参考

| 网络 | CIDR | 用途 |
|---|---|---|
| 主管理子网 | 10.51.0.0/16 | 节点 IP |
| Pod (secondary) | 10.64.0.0/14 | Pod CIDR |
| Service (secondary) | 10.68.0.0/20 | Service CIDR |
| 额外 GVNIC 子网 | 10.61.0.0/18 | 第二管理网卡 |
| RDMA sub-0~3 | 192.168.192-240.0/20 | RDMA NIC 0~3 |

> CIDR 需避开已有网络（为 VPC peering 预留空间）。

## Step 2: Placement Policy

A4X 必须使用 `workload-policy` 类型，指定 `acceleratorTopology=1x72`。

```bash
gcloud compute resource-policies create workload-policy chrisya-a4x-placement-${REGION} \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=1x72 \
  --region=$REGION \
  --project=$PROJECT
```

> `group-placement` 类型（即使加了 `--gpu-topology=1x72`）无法正确匹配 reservation 物理块，会报 GCE_STOCKOUT。

## Step 3: 创建 GKE 集群

三个关键 flag 必须在创建时指定（创建后不可更改）：

```bash
gcloud container clusters create chrisya-a4x-gke-${REGION} \
  --region=$REGION \
  --network=chrisya-gke-mgmt \
  --subnetwork=chrisya-gke-sub-${REGION} \
  --cluster-secondary-range-name=pods \
  --services-secondary-range-name=services \
  --release-channel=rapid \
  --enable-ip-alias \
  --enable-multi-networking \
  --enable-dataplane-v2 \
  --enable-private-nodes \
  --no-enable-private-endpoint \
  --master-authorized-networks=0.0.0.0/0 \
  --workload-pool=${PROJECT}.svc.id.goog \
  --addons=GcsFuseCsiDriver \
  --num-nodes=1 \
  --machine-type=e2-standard-16 \
  --project=$PROJECT
```

| Flag | 必须 | 说明 |
|---|---|---|
| --enable-multi-networking | 是 | A4X 多网卡前提 |
| --enable-dataplane-v2 | 是 | eBPF dataplane，多网卡必需 |
| --enable-private-nodes | 建议 | 节点无外网 IP |
| --no-enable-private-endpoint | 建议 | 允许外部 kubectl 访问 |
| --workload-pool | 是 | GCSFuse CSI 需要 Workload Identity |

## Step 4: 创建 A4X Node Pool

```bash
RESERVATION=nvidia-gb200-jwmrpsfbs8szi  # 替换为实际 reservation 名

gcloud container node-pools create a4x-pool \
  --cluster=chrisya-a4x-gke-${REGION} \
  --region=$REGION \
  --node-locations=$ZONE \
  --machine-type=a4x-highgpu-4g \
  --accelerator=type=nvidia-gb200,count=4,gpu-driver-version=LATEST \
  --num-nodes=2 \
  --reservation-affinity=specific \
  --reservation=projects/$PROJECT/reservations/$RESERVATION \
  --placement-policy=chrisya-a4x-placement-${REGION} \
  --additional-node-network=network=chrisya-gke-net-1,subnetwork=chrisya-gke-sub-1-${REGION} \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-0 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-1 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-2 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-3 \
  --ephemeral-storage-local-ssd=count=4 \
  --enable-gvnic \
  --project=$PROJECT
```

| 参数 | 值 | 说明 |
|---|---|---|
| machine-type | a4x-highgpu-4g | 每节点 4 GPU (GB200) |
| accelerator | nvidia-gb200,count=4 | GPU 类型和数量 |
| gpu-driver-version | LATEST | GKE 自动安装最新驱动 (580.x) |
| reservation-affinity | specific | 绑定特定 reservation |
| placement-policy | workload-policy | 1x72 NVL72 拓扑 |
| additional-node-network | 1 GVNIC + 4 RDMA | 6 网卡配置 |
| ephemeral-storage-local-ssd | 4 | 4x 3TB NVMe |

## Step 5: 验证

```bash
# 获取凭据
gcloud container clusters get-credentials chrisya-a4x-gke-${REGION} \
  --region=$REGION --project=$PROJECT

# 节点状态
kubectl get nodes -o wide

# GPU 验证
kubectl run nvidia-smi --rm -it --restart=Never \
  --image=nvidia/cuda:12.8.0-base-ubuntu22.04 \
  --overrides='{"spec":{"nodeSelector":{"cloud.google.com/gke-accelerator":"nvidia-gb200"},"containers":[{"name":"nvidia-smi","image":"nvidia/cuda:12.8.0-base-ubuntu22.04","command":["nvidia-smi"],"resources":{"limits":{"nvidia.com/gpu":"4"}}}]}}' \
  -- nvidia-smi
```

## 实测记录

### europe-west4-b Qwen3 30B 训练 (2026-07-03)

- 集群: `chrisya-a4x-gke-ew4`, K8s 1.36.0-gke.3302004
- Node pool: 2 台 `a4x-highgpu-4g` (8 GPU), RUNNING
- Reservation: `nvidia-gb200-jwmrpsfbs8szi`, inUseCount 0 → 2
- GPU 驱动: NVIDIA 580.126.20 (GKE 自动安装)
- RDMA VPC: `chrisya-gke-rdma-ew4` with `europe-west4-b-vpc-roce` profile

| 指标 | GKE (europe-west4) | 自建 K8s (A4X) | DGX-GB200 (官方) |
|---|---|---|---|
| Model TFLOP/s/GPU | **924.6** (peak) | 914 | 936 |
| Step Time | 6.52s | 6.60s | — |
| HBM Peak | 168.4 GiB | 184.7 GiB | — |
| 差距 vs 官方 | -1.2% | -2.3% | baseline |

GKE 比自建 K8s 快 1%，可能是因为 GKE 的 NCCL RDMA DaemonSet 自动优化了 GIB 配置

## 踩坑总结

经过 7 轮迭代才成功，关键教训：

### 1. Placement Policy 必须是 workload-policy 类型

| 方式 | 结果 |
|---|---|
| --placement-type=COMPACT | 报错: GPU placement policy must be provided |
| group-placement --collocation=COLLOCATED | 同上 |
| group-placement --gpu-topology=1x72 | GCE_STOCKOUT |
| workload-policy --type=HIGH_THROUGHPUT --accelerator-topology=1x72 | 成功 |

### 2. 集群级别 flag 创建后不可更改

`--enable-multi-networking` 和 `--enable-dataplane-v2` 必须在 `clusters create` 时指定。漏了只能删集群重建。

### 3. RDMA VPC 必须绑 vpc-roce network profile

普通 VPC 即使 MTU 设对了也不行，报错 "Network doesn't have a network profile and can't host a RDMA NIC"。且 profile 是 zone 级别的（如 `europe-west4-b-vpc-roce`），不能跨 region 共用同一个 RDMA VPC。

### 4. Reservation shareType 影响 GKE 消费

`shareType: LOCAL` 可以被同项目 GKE 正常消费。`shareType: SPECIFIC_PROJECTS` 在 us-central1-b 测试中无法消费（7 轮全部 GCE_STOCKOUT），可能需要额外的 IAM 或 service agent 配置。

### 5. GCSFuse CSI 需要 Workload Identity

创建集群时必须加 `--workload-pool=${PROJECT}.svc.id.goog`。

## 清理

```bash
# 删除 node pool
gcloud container node-pools delete a4x-pool \
  --cluster=chrisya-a4x-gke-${REGION} --region=$REGION \
  --project=$PROJECT --quiet

# 删除集群（保留 VPC 网络供下次使用）
gcloud container clusters delete chrisya-a4x-gke-${REGION} \
  --region=$REGION --project=$PROJECT --quiet
```
