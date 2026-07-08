# GB300 (A4X MAX) Deployment Guide

GB300 NVL72 (A4X MAX) 部署指南，基于 GB200 (A4X) 实战经验 + GCP 官方文档整理。

## GB300 vs GB200 硬件规格对比

| 维度 | GB200 (A4X) | GB300 (A4X MAX) | 影响 |
|------|-------------|-----------------|------|
| 机器类型 | a4x-highgpu-4g | a4x-maxgpu-4g-metal | 裸金属实例 |
| CPU | Grace ARM64, 140 vCPU | Grace ARM64, 144 vCPU | 架构相同 |
| 内存 | 884 GB | 960 GB | — |
| GPU | 4x B200 | 4x B300 Ultra | GPU 型号变更 |
| GPU 显存 | 744 GB (186 GB/GPU) | 1,116 GB (~279 GB/GPU, HBM3e) | +50% |
| 每域 GPU 显存 | 13.4 TB | 20 TB | — |
| 管理网卡 | GVNIC (Google Titanium) | IDPF (Intel) | 驱动/改名方式不同 |
| RDMA 网卡 | CX-7 VF, 挂在 CPU 上 | CX-8 SuperNIC PF, 直连 GPU (GPUDirect) | 延迟更低 |
| RDMA 网络接口数 | 4 (单端口) | 8 (CX-8 双端口, 8-way rail) | 子网翻倍 |
| 总网络接口数 | 6 | 10 (2 IDPF + 8 MRDMA) | VM 创建命令 |
| 网络带宽 | 2,000 Gbps | 3,200 Gbps | — |
| 网络栈 | IPv4 | IPv6-only | 重大变更，影响全栈 |
| Boot Disk | pd-balanced 或 hyperdisk-balanced | 仅 hyperdisk-balanced | — |
| NVL72 拓扑 | 18 节点/域, 1x72 | 18 节点/域, 1x72 | 不变 |
| Block 结构 | — | 25 sub-blocks = 450 VM = 1,800 GPU | 规模更大 |

## 三大核心差异

### 1. 网络架构升级

#### 1.1 管理网卡：GVNIC → IDPF

- GB200 使用 Google GVNIC，设备名 `enp0s3`、`enp192s2`
- GB300 使用 Intel IDPF，设备名需实际确认
- 裸金属实例不支持 gVNIC 或 VirtIO

> 来源: [裸金属实例](https://docs.google.com/compute/docs/instances/bare-metal-instances#differences-between-vm-instances-and-bare-metal-instances)

**行动项**: 拿到 GB300 VM 后 `ip -br link show` + `ethtool -i <dev>` 确认 IDPF 默认设备名。

#### 1.2 RDMA 网卡：CX-7 VF → CX-8 PF (GPUDirect)

这是 GB300 最重要的硬件架构升级。

| 维度 | GB200 | GB300 |
|------|-------|-------|
| 网卡型号 | CX-7 | CX-8 SuperNIC |
| 暴露方式 | SR-IOV VF | PF (物理功能) |
| 连接方式 | 挂在 CPU 上 | 直连 GPU (GPUDirect RDMA) |
| 每卡端口数 | 1 | 2 (800 Gbps = 2x400 Gbps) |
| MRDMA 接口数 | 4 | 8 |
| IB 设备 | mlx5_0~3 | 待确认 (可能 8 个 PF) |

> 来源: [GPU 网络带宽](https://cloud.google.com/compute/docs/gpus/gpu-network-bandwidth#a4x-max-and-a4x-machine-types)

**影响**: RDMA 子网 4→8，VM 创建命令 MRDMA 接口 4→8，ResourceClaimTemplate count 4→8。

#### 1.3 网络栈：IPv4 → IPv6-only

**最具影响力的变更。** GB300 所有 NIC 使用 `stack-type=IPV6_ONLY`。

| 组件 | IPv4 (GB200) | IPv6 (GB300) |
|------|-------------|-------------|
| 子网 | --range=10.x.x.x/22 | IPv6 子网配置 |
| 防火墙 | --source-ranges=10.x/22 | IPv6 规则 |
| kubeadm join | CP_IP:6443 | [fd20::x]:6443 |
| Calico CNI | IPv4 Pod CIDR | IPv6 或双栈 |
| NCCL/GIB | RoCEv2 over IPv4 | RoCEv2 over IPv6 |

> 来源: [A4X MAX 实例创建](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-a4xmax-instance#create-instance)

**关键风险**: IPv6-only 是最需要提前验证的变更。拿到 GB300 VM 后立即测试 metadata server 可达性、外网访问、kubeadm IPv6 行为。

### 2. VM 创建命令差异

```bash
# GB200
gcloud compute instances create ${NAME} \
  --machine-type=a4x-highgpu-4g \
  --network-interface=nic-type=GVNIC,network=$NET,subnet=$SUB \
  --network-interface=nic-type=MRDMA,...  # x4

# GB300
gcloud compute instances create ${NAME} \
  --machine-type=a4x-maxgpu-4g-metal \                           # 裸金属
  --network-interface=nic-type=IDPF,...,stack-type=IPV6_ONLY \   # GVNIC→IDPF
  --network-interface=nic-type=MRDMA,...,stack-type=IPV6_ONLY \  # x8 (翻倍)
```

变更点:
1. machine-type: `a4x-highgpu-4g` → `a4x-maxgpu-4g-metal`
2. nic-type: `GVNIC` → `IDPF`
3. 每个接口加 `stack-type=IPV6_ONLY`
4. MRDMA 从 4 条增加到 8 条

### 3. Kubernetes 部署差异

| 维度 | GB200 (A4X) | GB300 (A4X MAX) |
|------|-------------|-----------------|
| GKE 最低版本 | 1.32.8+ | 1.34.3-gke.1318000+ |
| GPU Driver | R580 分支 | R580.95.05+ |
| DRA Driver | v0.4.0 | v25.8.0 (推荐) |
| RDMA ResourceClaim count | 4 | 8 |
| RDMA 配置方式 | RDMA installer DaemonSet | asapd-lite DaemonSet |
| Hugepages | 未显式配置 | hugepage_size2m: 4096 |
| Pod Quota | 无特殊要求 | 2 x 节点数 + 1 |

> 来源: [A4X MAX GKE 部署](https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x-max#requirements)

## Placement Policy / Workload Policy

| 场景 | API | 参数 |
|------|-----|------|
| 自建 k8s (手动 VM) | gcloud beta compute resource-policies create group-placement | --collocation=COLLOCATED --gpu-topology=1x72 |
| GKE 托管集群 | gcloud beta compute resource-policies create workload-policy | --type HIGH_THROUGHPUT --accelerator-topology 1x72 |

两种 API 在 A4X MAX 上均适用。拓扑参数不变: 1x72 (18 节点/域)。

> 来源: [A4X MAX 实例创建](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-a4xmax-instance#create-compact-placement-policy)

## Reservation 差异

GB300 支持更细粒度的 reservation 指定:
- GB200: `--reservation=RESERVATION_NAME`
- GB300: `RESERVATION_NAME/reservationBlocks/BLOCK/reservationSubBlocks/SUBBLOCK`

## 镜像兼容性

| 驱动 | GB200 | GB300 | 状态 |
|------|-------|-------|------|
| NVIDIA GPU | 580.159.04 | >= R580.95.05 | 已满足 |
| GVE (GVNIC) | 需要 | 不需要 | 存在但不加载 |
| IDPF (Intel) | 不需要 | v1.0.11 oot | v5.3 镜像已包含 |
| MLNX OFED | 内核内置 | CX-8 需确认 | 需验证 CX-8 兼容性 |

架构均为 ARM64 (Grace CPU)，镜像架构无需变更。

## 行动计划

### P0: 拿到 GB300 VM 后立即确认

| # | 确认项 | 方法 |
|---|--------|------|
| 1 | IDPF 默认网卡名 | ip -br link show + ethtool -i |
| 2 | CX-8 IB 设备名和数量 | rdma link show |
| 3 | NM 配置文件路径 | ls /etc/NetworkManager/system-connections/ |
| 4 | IPv6 地址获取 | ip -6 addr show |
| 5 | Metadata Server IPv6 可达性 | curl -6 http://metadata.google.internal/... |
| 6 | CX-8 mlx5_core 驱动兼容性 | ibstat, rdma link show |
| 7 | 外网 IPv6 访问 | curl -6 https://download.docker.com/... |

### P1: 部署脚本修改

| # | 修改项 | 复杂度 |
|---|--------|--------|
| 1 | RDMA 子网 4→8 | 低 |
| 2 | VM 创建命令 (NIC 类型 + 数量 + stack-type) | 中 |
| 3 | Startup script 网卡改名 (设备名 + 数量) | 中 |
| 4 | ResourceClaimTemplate RDMA count 4→8 | 低 |
| 5 | DRA driver 版本升级 | 中 |

### P2: IPv6 全栈适配

| # | 修改项 | 复杂度 |
|---|--------|--------|
| 1 | IPv6 子网 + 防火墙 | 高 |
| 2 | kubeadm init/join IPv6 | 高 |
| 3 | Calico IPv6/双栈 | 高 |
| 4 | GIB/NCCL IPv6 RoCE | 高 |

## 可直接复用的部分

以下在 GB200/GB300 之间完全相同:

- Boot disk 分区逻辑
- Local SSD RAID0
- sshd 端口配置
- 拓扑标签获取
- k8s 组件安装 (containerd, kubeadm, kubelet)
- Calico VXLAN 模式
- ComputeDomain 配置
- DRA RBAC 修复
- Kueue TAS + JobSet
- MPI Operator
- GIB init container 模式 (镜像版本需确认)

## 通用 Startup Script v2

v2 startup script 自动适配 GB200/GB300，不再硬编码网卡名:

- 管理网卡: 扫描 `/sys/class/net/*/device` 自动检测，默认路由网卡→bond0
- RDMA: `rdma dev` 动态枚举，数量不限
- GB200: bond0-1 (2x GVNIC) + bond2-5 (4x CX-7) = 6 个
- GB300: bond0-1 (2x IDPF) + bond2-9 (8x CX-8) = 10 个

完整脚本见 [scripts/](scripts/) 目录 (Customer 版 + Internal 版)。

## 官方文档链接

| 主题 | URL |
|------|-----|
| A4X MAX GKE 部署 | https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x-max |
| A4X MAX 实例创建 | https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-a4xmax-instance |
| A4X GKE 部署 (对比) | https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x |
| 加速器机器类型规格 | https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines |
| GPU 网络带宽 | https://cloud.google.com/compute/docs/gpus/gpu-network-bandwidth |
| 裸金属实例 | https://docs.cloud.google.com/compute/docs/instances/bare-metal-instances |
