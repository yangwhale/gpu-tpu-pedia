> 🌐 **中文** | [English](README.en.md)

# 0. 架构概述与核心概念 + 1. 环境准备

> 本章面向首次接触 GB200/A4X 的工程师，梳理硬件架构、GCP 机型差异、核心概念，以及 VPC/子网/防火墙/Placement Policy 的创建。

## 0. 架构概述与核心概念

**阅读建议**：本章面向首次接触 GB200/A4X 的工程师，梳理硬件架构、GCP 机型差异、以及部署过程中频繁出现的核心概念。已有经验的读者可跳至 [第 1 章](#1-环境准备)。

### 0.1 硬件架构对比

下表对比三代 GPU 平台的关键硬件差异——这些差异直接决定了部署方式和可达性能。

| 维度 | GB200 (Blackwell) | B200 (Blackwell) | H200 (Hopper) |
|------|-------------------|------------------|---------------|
| GPU 架构 | Blackwell (sm_100) | Blackwell (sm_100) | Hopper (sm_90) |
| NVLink 代际 | 第 5 代 NVLink | 第 5 代 NVLink | 第 4 代 NVLink |
| NVSwitch | 第 5 代 · 支持 MNNVL 跨节点 NVLink | 第 5 代 · 支持 MNNVL（受限） | 第 4 代 · 仅节点内 NVLink |
| NVLink 域大小 | **72 GPU**（NVL72：18 节点跨节点互联） | 16 GPU（NVL16：2 节点） | 8 GPU（仅节点内） |
| CPU | Grace ARM64 (aarch64) | x86_64 | x86_64 |
| 显存 | HBM3e | HBM3e | HBM3e |

**关键差异**：GB200 的 NVL72 域将 18 个节点的 72 块 GPU 通过 NVSwitch 互联，峰值双向带宽约 **840 GB/s**（每 GPU 对）。而跨域通信只能走 RDMA，峰值约 325 GB/s，差距约 2.5 倍。因此，将同一训练任务的所有节点放入同一 NVL72 域是性能优化的第一要务。

### 0.2 GCP 机型对比

GB200/B200/H200 在 GCP 上对应不同的 Accelerator-Optimized 机型系列，网络拓扑和部署方式存在显著差异。

| 维度 | A4X (GB200) | A4 (B200) | A3 Ultra (H200) |
|------|-------------|-----------|-----------------|
| 机型示例 | `a4x-highgpu-4g` | `a4-highgpu-8g` | `a3-ultragpu-8g` |
| GPU / 节点 | 4 (GB200) | 8 (B200) | 8 (H200) |
| CPU 架构 | ARM64 (Grace) | x86_64 | x86_64 |
| RDMA NIC | 4 × CX-7 400Gbps | 8 × CX-7 400Gbps | 8 × CX-7 400Gbps |
| NVLink 域 | **18 节点 / 72 GPU**（NVL72） | 2 节点 / 16 GPU（NVL16） | N/A（仅节点内 NVLink） |
| MNNVL 跨节点 NVLink | 支持（需 IMEX daemon） | 支持（受限，2 节点） | 不支持 |
| Placement Policy 拓扑 | `1x72` | `1x16` | N/A |
| 容器镜像注意事项 | 需 ARM64 / aarch64 镜像 | 标准 x86_64 镜像 | 标准 x86_64 镜像 |

**A4X ARM64 注意**：GB200 使用 Grace ARM64 CPU，所有容器镜像（包括 NCCL 插件、NVSHMEM 编译产物等）必须为 aarch64 架构。GIB NCCL 插件镜像名需带 `-arm64` 后缀。

### 0.3 核心概念速查

| 概念 | 说明 |
|------|------|
| **NVL72 Domain** | 18 个 A4X 节点（72 块 GB200 GPU）通过第 5 代 NVSwitch 组成的跨节点 NVLink 互联域。域内任意 GPU 对可通过 NVLink 直接通信，峰值双向带宽约 840 GB/s。域外通信只能走 RDMA（约 325 GB/s）。 |
| **MNNVL** (Multi-Node NVLink) | 跨节点 NVLink 通信能力。GB200 NVL72 的核心特性——使同一域内不同物理节点的 GPU 像节点内一样通过 NVLink 互联。需要 IMEX daemon 运行才能启用。 |
| **IMEX Daemon** (Inter-node Memory Exchange) | 管理跨节点 NVLink channel 的用户态守护进程 (`nvidia-imex`)。NVLS transport 的 `cuMulticastCreate` 依赖 IMEX 协调。若 IMEX 未运行，NCCL 会静默回退到 RDMA（带宽从 ~840 GB/s 降到 ~326 GB/s），且报 CUDA error 801。 |
| **ComputeDomain** | Kubernetes DRA CRD（由 DRA GPU Driver 提供），用于在 k8s 中声明式管理 IMEX daemon 生命周期。创建 ComputeDomain 后，域内每个节点自动启动 IMEX daemon pod，Pod 通过 ResourceClaimTemplate 引用 ComputeDomain 即可获得 MNNVL 能力。每个节点同一时间只能属于一个 ComputeDomain。 |
| **DRA** (Dynamic Resource Allocation) | Kubernetes 原生的硬件资源请求 API（k8s 1.33+ GA，API 版本 `resource.k8s.io/v1`）。Pod 通过 ResourceClaim 声明式申请 GPU、RDMA NIC 等设备，由 scheduler 和 DRA driver 协同分配。 |
| **DRANET** | 基于 DRA 的 RDMA NIC 分配驱动（v1.3.0）。将 CX-7 RDMA NIC 作为 DRA 设备发布到 ResourceSlice，Pod 通过 ResourceClaim 请求 RDMA 设备，实现 GPU-NIC PCIe 拓扑感知的精确分配。 |
| **GIB** (GPUDirect InfiniBand) | Google 提供的 NCCL 通信插件（v1.1.2），封装了 GPUDirect RDMA 优化。以 init container 方式注入到 Pod，挂载到 `/usr/local/gib`，通过 `set_nccl_env.sh` 自动配置 NCCL 环境变量。 |
| **Placement Policy** | GCP 资源策略，确保一组 VM 被分配到同一个物理 NVL72 域。A4X 使用 `--collocation=COLLOCATED --gpu-topology=1x72` 创建，每个 NVL72 域绑定一个 Placement Policy。生产环境中 N 个域需创建 N 个 Policy。 |

### 0.4 三层协调机制（Placement Policy → ComputeDomain → IMEX）

NVL72 跨节点 NVLink 通信需要三层协调，缺一不可：

```
Placement Policy    →  保证 VM 落在同一个 NVSwitch 物理域
       ↓
ComputeDomain CRD   →  按 gpu.clique label 发现同域节点，启动 IMEX daemon
       ↓
IMEX daemon         →  18 节点互相握手，建立 NVLink multicast channel
       ↓
NVLS transport 就绪 →  NCCL / NVSHMEM / DeepEP 走 NVSwitch 通信（~900 GB/s）
```

**如果缺了某一层会发生什么？**
- 缺 Placement Policy → VM 散落到不同物理域，NVSwitch 不通
- 缺 ComputeDomain → IMEX daemon 不启动，软件层不通
- 缺 IMEX → NCCL **不报错**，静默退化到 RDMA（900→325 GB/s）。DeepEP 吞吐差 8 倍

**MNNVL 环境变量控制**：`MNNVL_ENABLE=0` 强制走 RDMA；`MNNVL_ENABLE=2` 全开走 NVLink。

### 0.5 NVLink 带宽：节点内 = 跨节点

GB200 NVL72 的所有 GPU 通信**均通过 NVSwitch**，不区分节点内外：
- 节点内 4 块 GPU → 通过本节点 NVSwitch → 单向 900 GB/s
- 跨节点 GPU → 通过 NVSwitch 光缆互联 → 同样单向 900 GB/s

带宽完全对称。跨节点仅在延迟上比节点内多几微秒（光电转换），带宽无差异。

### 0.6 多 Team 共享一个 NVL72 域

**安全性**：一个 NVL72 域可以分给多个 Team 使用。GPU 间隔离由四层保证：
1. **k8s device isolation** — NVIDIA device plugin / DRA 只给 Pod 暴露分配的 GPU
2. **CUDA device visibility** — 进程只能 `cudaSetDevice` 到自己可见的 GPU
3. **Peer access 需显式开启** — 跨 GPU 内存访问需调 `cudaDeviceEnablePeerAccess`
4. **NCCL communicator 隔离** — 不同 Team 用不同 unique ID，communicator 天然隔离

**带宽不冲突**：NVSwitch 是全带宽无阻塞交换。Team A 的通信不占 Team B 的带宽。

**ComputeDomain 约束**：域内同时只能有一个 ComputeDomain。多 Team 共享域时需统一管理 ComputeDomain，不能各建各的。IMEX 是控制平面（协调 channel 建立），不是数据平面（不转发流量），连在一起不影响性能。

#### 概念关系图

```
┌─────────────────── NVL72 Domain (18 节点 / 72 GPU) ───────────────────┐
│                                                                        │
│  ┌─ Node 1 ──┐  ┌─ Node 2 ──┐       ┌─ Node 18 ─┐                    │
│  │ 4× GB200  │  │ 4× GB200  │  ...  │ 4× GB200  │                    │
│  │ 4× CX-7   │  │ 4× CX-7   │       │ 4× CX-7   │                    │
│  └─────┬─────┘  └─────┬─────┘       └─────┬─────┘                    │
│        │               │                    │                          │
│        └───── NVSwitch (MNNVL, ~840 GB/s) ──┘                         │
│                                                                        │
│  Placement Policy (1x72) ── 确保 VM 落在同一物理域                       │
│  ComputeDomain CRD ── 管理 IMEX daemon，启用 MNNVL                     │
│  DRANET ── 分配 CX-7 RDMA NIC 给 Pod                                  │
│  GIB ── NCCL 通信插件，优化 GPU RDMA                                    │
└────────────────────────────────────────────────────────────────────────┘
        │ 跨域通信 (RDMA only, ~325 GB/s)
┌───────┴──────── 另一个 NVL72 Domain ──────────────────────────────────┐
│  ...                                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 环境准备

### 全局变量（请根据实际环境修改）

**CIDR 规划说明**：以下网段适用于 **1800 GPU（450 A4X VM）** 规模的集群。主管理子网使用 /21（2046 IP），其余子网使用 /22（1022 IP），均可容纳 450+ A4X VM 并留有余量（Control Plane、普通 VM、未来扩展）。

**读者请根据内部网络团队分配的网段替换以下 CIDR 值**。仅需修改变量值，后续所有命令自动引用。

```bash
# ===== GCP 项目与区域 =====
PROJECT="your-gcp-project"                # ← 请替换为您的 GCP 项目 ID
REGION="us-east5"                          # ← 请替换为您的区域
ZONE="us-east5-a"                          # ← 请替换为您的可用区
RESERVATION="your-gb200-reservation"       # ← 请替换为您的 GB200 预留名称

# ===== 机器与镜像 =====
MACHINE_TYPE="a4x-highgpu-4g"
IMAGE="tlinux-server-4-gb200-v1"           # ← 请替换为您的 VM 镜像名称
IMAGE_PROJECT="$PROJECT"                   # ← 镜像所在项目（如与 VM 项目不同请修改）

# ===== 网络名称 =====
GVNIC_NET="a4x-gvnic-net-0"               # ← 主 GVNIC 管理网络（MTU 8896）
GVNIC_SUB="a4x-gvnic-sub-0"               # ← 主 GVNIC 子网
GVNIC_NET_1="a4x-gvnic-net-1"             # ← 辅助 GVNIC 网络
GVNIC_SUB_1="a4x-gvnic-sub-1"             # ← 辅助 GVNIC 子网
RDMA_NET="a4x-rdma-net"                   # ← RDMA HPC 网络（需 network profile）

# ===== 网络 CIDR（请根据内部网络团队分配的网段调整） =====
GVNIC_CIDR="10.0.0.0/21"                  # 主管理子网: /21 = 2046 IP
GVNIC_1_CIDR="10.0.8.0/22"                # 辅助管理子网: /22 = 1022 IP
RDMA_CIDR_0="10.0.16.0/22"                # RDMA 子网 0 (CX-7 NIC 0)
RDMA_CIDR_1="10.0.20.0/22"                # RDMA 子网 1 (CX-7 NIC 1)
RDMA_CIDR_2="10.0.24.0/22"                # RDMA 子网 2 (CX-7 NIC 2)
RDMA_CIDR_3="10.0.28.0/22"                # RDMA 子网 3 (CX-7 NIC 3)
SUBNET_SUPERNET="10.0.0.0/16"             # ← 所有子网的超网, 用于防火墙规则

# ===== NVL72 Domain 配置 =====
NUM_DOMAINS=2                              # ← 部署的 NVL72 Domain 数量（生产环境可设 25+）
NODES_PER_DOMAIN=18                        # A4X NVL72 固定值，请勿修改
PLACEMENT_PREFIX="a4x-nvl72-domain"        # Placement Policy 名称前缀

# ===== 集群命名 =====
CP_NAME="gb200-cp"                         # Control Plane 节点名称
WORKER_PREFIX="gb200"                      # Worker 节点名称前缀

# ===== Kubernetes =====
K8S_VERSION="1.34"
POD_CIDR="10.244.0.0/16"

# ===== 组件版本 =====
GIB_VERSION="v1.1.2"
DEVICE_PLUGIN_VERSION="v0.17.1"
DRANET_VERSION="v1.3.0"
DRA_GPU_DRIVER_VERSION="v25.12.0"
CALICO_VERSION="v3.29.3"
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:26.05-py3"
GIB_IMAGE="us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:${GIB_VERSION}"

# ===== 共享存储 =====
GCSFUSE_BUCKET="your-training-bucket"      # ← 请替换为您的 GCS bucket 名称
LOCAL_SSD_MOUNT="/mnt/stateful_partition"   # Local SSD RAID0 挂载点
```

### 1.1 前提条件

- GCP 项目已有 GB200 A4X 预留（DENSE 或 CALENDAR 类型）
- 本地已安装 `gcloud` CLI 并认证
- 已完成 1.2 节的 VPC/子网/防火墙创建

### 1.2 创建 VPC / 子网 / RDMA 网络

#### 1.2.1 主 GVNIC 管理网络（MTU 8896）

**MTU 必须设为 8896**（Jumbo Frames）。默认 MTU 1460/1500 会导致 Lustre 吞吐量下降约 10%。MRDMA 网络的 MTU 由 Network Profile 自动配置，无需手动设置。

```bash
# 创建主 GVNIC 网络（MTU 8896）
gcloud compute networks create $GVNIC_NET \
  --subnet-mode=custom --mtu=8896 --project=$PROJECT

gcloud compute networks subnets create $GVNIC_SUB \
  --network=$GVNIC_NET \
  --region=$REGION \
  --range=$GVNIC_CIDR \
  --project=$PROJECT
```

#### 1.2.2 辅助 GVNIC 网络

```bash
gcloud compute networks create $GVNIC_NET_1 \
  --subnet-mode=custom --project=$PROJECT

gcloud compute networks subnets create $GVNIC_SUB_1 \
  --network=$GVNIC_NET_1 \
  --region=$REGION \
  --range=$GVNIC_1_CIDR \
  --project=$PROJECT
```

#### 1.2.3 RDMA HPC 网络（需 Network Profile）

**Network Profile**：RDMA 网络需要使用 `--network-profile=${ZONE}-vpc-roce` 创建。该 Network Profile 由 GCP 自动配置最优 MTU 和 RDMA 参数。

```bash
# 创建 RDMA HPC 网络
gcloud beta compute networks create $RDMA_NET \
  --network-profile=${ZONE}-vpc-roce \
  --subnet-mode=custom \
  --project=$PROJECT

# 创建 4 个 RDMA 子网（每个对应一块 CX-7 NIC）
RDMA_CIDRS=($RDMA_CIDR_0 $RDMA_CIDR_1 $RDMA_CIDR_2 $RDMA_CIDR_3)
for n in 0 1 2 3; do
  gcloud compute networks subnets create ${RDMA_NET}-sub-${n} \
    --network=$RDMA_NET \
    --region=$REGION \
    --range="${RDMA_CIDRS[$n]}" \
    --project=$PROJECT
done
```

#### 1.2.4 防火墙规则

```bash
# SSH 防火墙（IAP 源 IP 范围）
gcloud compute firewall-rules create ${GVNIC_NET}-allow-iap-ssh \
  --network=$GVNIC_NET \
  --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --project=$PROJECT

# 主 GVNIC 内部通信（含 Pod CIDR — ComputeDomain daemon pods 需要跨节点通信）
gcloud compute firewall-rules create ${GVNIC_NET}-allow-internal \
  --network=$GVNIC_NET \
  --allow=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=${SUBNET_SUPERNET},${POD_CIDR} \
  --project=$PROJECT

# 辅助 GVNIC 内部通信
gcloud compute firewall-rules create ${GVNIC_NET_1}-allow-internal \
  --network=$GVNIC_NET_1 \
  --allow=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=${SUBNET_SUPERNET} \
  --project=$PROJECT
```

**防火墙注意**：某些 GCP 组织策略会自动删除新建的防火墙规则。如果 SSH 或 Pod 通信失败，请先检查防火墙规则是否存在。建议使用 IAP SSH（`--tunnel-through-iap`）以绕过防火墙限制。

#### 网络布局总览

| 网络 | 子网 | CIDR | 用途 |
|------|------|------|------|
| `$GVNIC_NET` | `$GVNIC_SUB` | `$GVNIC_CIDR` (/21, 2046 IP) | 主管理网络（SSH, k8s API, Pod CIDR） |
| `$GVNIC_NET_1` | `$GVNIC_SUB_1` | `$GVNIC_1_CIDR` (/22, 1022 IP) | 辅助管理网络 |
| `$RDMA_NET` | `${RDMA_NET}-sub-0..3` | `$RDMA_CIDR_0..3` (/22 each) | GPU RDMA 计算网络 (每 NIC 一个子网) |

### 1.3 查看预留状态

```bash
gcloud compute reservations describe $RESERVATION \
  --zone=$ZONE --project=$PROJECT \
  --format="table(specificReservation.count, specificReservation.inUseCount)"
```

### 1.4 Placement Policy（每个 Domain 需要）

**生产环境规模**：每个 NVL72 Domain（18 节点 / 72 GPU）需要独立的 Placement Policy。例如 1800 GPU = 25 个 Domain，需创建 25 个 Policy。Placement Policy 与 Domain 一一绑定，确保该 Policy 下的 VM 被分配到同一个 NVSwitch Domain。

```bash
# 为每个 Domain 创建 collocated placement policy
for d in $(seq 1 $NUM_DOMAINS); do
  gcloud beta compute resource-policies create group-placement \
    ${PLACEMENT_PREFIX}-${d} \
    --collocation=COLLOCATED \
    --gpu-topology=1x72 \
    --project=$PROJECT --region=$REGION
done
```

#### Placement Policy FAQ

**Q: `--gpu-topology=1x72` 只能写 72 吗？**
A: 该值跟机型对应。A4X (GB200) 用 `1x72`（18 节点 72 GPU），A4 (B200) 用 `1x16`（2 节点 16 GPU）。值定义的是最小拓扑约束——"给我至少这么大的一个全互联域"。

**Q: 必须一次创建 18 台 VM 才能用吗？**
A: 不需要。`1x72` 的域里只用 2 台、4 台完全可以。未使用的位置空着，将来可以加机器。ComputeDomain 会为域内全部 18 个物理位置生成 IMEX 配置，未部署的位置产生连接重试日志（无害，可忽略）。

**Q: 多个 Placement Policy 会分到同一个物理域吗？**
A: 可能会。GCP 调度器根据空位自动分配。如果两个 Policy 落在同一域，所有 VM 的 `gpu.clique` label 相同。此时 ComputeDomain 按 clique 选节点，域内只能有一个 ComputeDomain 实例——多个 Policy 的用户需要协调共用。

**Q: 已有其他人的 VM 在域里，我加机器会冲突吗？**
A: 不会。只要域内空位够，你的 VM 正常创建。GPU 通信走 NCCL communicator 隔离，NVSwitch 全带宽无阻塞，不影响彼此。但如果对方已创建 ComputeDomain，你需要复用它而不是另建一个。

### 1.5 创建额外防火墙规则（可选）

```bash
# SSH 防火墙（IAP 源 IP 范围）
gcloud compute firewall-rules create allow-ssh-iap-k8s1341 \
  --network=$GVNIC_NET \
  --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --project=$PROJECT

# 集群内部通信（含 Pod CIDR）
gcloud compute firewall-rules create allow-internal-k8s1341 \
  --network=$GVNIC_NET \
  --allow=tcp,udp,icmp \
  --source-ranges=${SUBNET_SUPERNET},${POD_CIDR} \
  --project=$PROJECT

# 注：Calico 使用 VXLAN 模式（UDP 4789），已被上方 allow-internal 规则覆盖
```

---

## 已知问题与注意事项

### TLinux 4 特性

| 问题 | 说明 |
|------|------|
| 启动盘设备名不固定 | NVMe 设备编号在不同启动间可能变化，需用 `findmnt` 动态查找 |
| 启动盘分区 | 50GB OS 镜像安装到 1TB 盘后不自动扩展。Startup script 使用 `sgdisk` 创建 4 分区布局 |
| 缺少基础组件 | sudo、gcloud CLI、perftest、CUDA Toolkit 均未预装，需手动安装 |
| Docker CE repo | 需硬编码 RHEL 9 baseurl，TLinux 4 不被 Docker 官方自动识别 |

### VPC 管理网卡 MTU

GVNIC 管理网络的 MTU 必须设置为 **8896**（Jumbo Frames）。默认 MTU 1460/1500 会导致 Lustre 吞吐量下降约 10%。

### 网卡绑定（Bond）不适用

GCP A4X 实例**不支持**传统的 Linux bonding（bond0/bond1 等），无需配置。

| 类型 | 设备名 | 数量 | 用途 | 冗余机制 |
|------|--------|------|------|----------|
| **GVNIC** | eth0, eth1 | 2 | 管理网络 | GCP Andromeda SDN 在虚拟化层提供 HA |
| **MRDMA** | mlx5_0..mlx5_3 | 4 | GPU RDMA 计算网络 | SR-IOV VF 由 GCP RDMA 基础设施管理，通过 DRANET DRA 分配 |

### nvidia-fabricmanager 不适用

GCP A4X 的 NVSwitch 由**虚拟化层（hypervisor）**管理，Guest VM 无需也不应运行 Fabric Manager 服务。NVSwitch 由 GCP hypervisor 层的 Service VM 管理。

### nvidia_peermem 不适用

GCP A4X 的 GPU Direct RDMA 通过 **GIB 插件**（`nccl-plugin-gib`）实现，不使用传统的 `nvidia_peermem` 内核模块。

### 混合云场景注意事项（自建 CP + GCP Worker）

如 Control Plane 部署在本地 IDC、Worker 在 GCP，需注意：

- **API Server 网络打通**：需 Cloud VPN 或 Cloud Interconnect
- **DRA 控制器跨网络认证**：kubeadm 默认 ClusterRole 缺少 DRA 权限（见 [03-gpu-stack](../03-gpu-stack/) 3.5 节）
- **自定义准入控制器兼容性**：确保不拦截 DRA 相关 API（`resource.k8s.io/v1`）
- **Calico VXLAN MTU**：跨专线需减去 50 字节 VXLAN overhead

---

## VM 交付验收对照表（附录 A）

### 系统与安全配置

| 适配项 | 客户标准 | GCP 适配状态 | GCP 说明 |
|--------|----------|-------------|----------|
| 操作系统 | TencentOS Server 4.0 ARM | 满足 | `tlinux-server-4-gb200-v1` 自定义镜像 |
| SELinux | Disabled | 满足 | startup script 设置 |
| Firewalld | Disabled | 满足 | startup script 禁用 |
| SSH 端口 | 仅允许 56000 | 满足 | startup script 修改 sshd_config |

### 网络与网卡配置

| 适配项 | 客户标准 | GCP 适配状态 | GCP 说明 |
|--------|----------|-------------|----------|
| 管理网卡 | bond1 | 不适用 | GVNIC 由 Andromeda SDN 管理 |
| RDMA 网卡 | bond2-bond5 | 替代方案 | 4 块 MRDMA NIC，通过 DRANET DRA 分配 |
| 网卡 MTU | >= 4200 | 满足 | GCP VPC MTU=8896 |
| 网卡驱动 | >= 5.8 | 满足 | mlx5_core 25.10-1.2.2 |

### GPU 驱动与服务

| 适配项 | 客户标准 | GCP 适配状态 | GCP 说明 |
|--------|----------|-------------|----------|
| GPU 驱动版本 | >= 535.247.01 | 满足 | NVIDIA 580.126.20（R580 Open） |
| nvidia_peermem | 已加载 | 不适用 | 使用 GIB 插件替代 |
| nvidia-fabricmanager | active | 不适用 | 虚拟化层 NVSwitch 管理 |
| NCCL/GIB | 包含 GIB | 满足 | GIB v1.1.2 通过 Pod initContainer 注入 |

### 存储与磁盘

| 适配项 | 客户标准 | GCP 适配状态 | GCP 说明 |
|--------|----------|-------------|----------|
| Boot disk | 1TB | 满足 | Hyperdisk Balanced 1TB |
| sda1 (/) | 21.5G | 满足 | startup script 分区 |
| sda2 (/boot/efi) | 512M | 满足 | TLinux 镜像默认 |
| sda3 (/usr/local) | 20G | 满足 | startup script 独立分区 |
| sda4 (/data) | ~950G | 满足 | startup script 剩余空间 |
| Local SSD | — | 满足 | NVMe RAID0 挂载到 /mnt/stateful_partition |
| Lustre 1PB | 随 GB200 交付 | 满足 | GCP Parallel Store (Lustre) CSI 驱动 |

### Startup Scripts（附录 B.5）

根据客户要求准备了两个版本的初始化脚本：

| 脚本 | 用途 | 说明 |
|------|------|------|
| `tlinux4-customer-init.sh` | 客户交付 | 不安装 docker/kubelet、SSH 56000、SELinux disabled、GDRCopy v2.6、boot disk 分区 |
| `tlinux4-internal-init.sh` | 内部测试 | 包含 containerd + NVIDIA Container Toolkit + kubeadm/kubelet/kubectl k8s 1.34 |

**注意**：VM 交付验收对照表中标注"不适用"的项目（bond、nvidia_peermem、nvidia-fabricmanager）是针对物理服务器 / 自建 IDC 环境的标准。在 GCP A4X 上，对应功能通过 GCP 原生机制实现（Andromeda SDN、GIB 插件、虚拟化层 NVSwitch 管理），功能等价但实现方式不同。

### 短主机名（附录 B.5 — Megatron 必需）

Megatron-LM 使用 Gloo 进行进程通信。如果 Pod hostname 过长，会触发 `File name too long` 错误。解决方案：在 Pod spec 中设置 `hostname` 字段为短名称（如 `mega-h1`）。
