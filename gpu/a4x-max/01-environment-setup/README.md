# GB300 (A4X Max) 环境搭建

GB300 NVL72 Bare Metal 实例的 VPC 网络、VM 创建和初始化配置。

## 与 GB200 的核心差异

| 维度 | GB200 (A4X) | GB300 (A4X Max) |
|------|-------------|-----------------|
| 实例类型 | VM (虚拟机) | **Bare Metal** (裸金属) |
| 机器类型 | `a4x-highgpu-4g` | `a4x-maxgpu-4g-metal` |
| 管理网卡 | GVNIC (Google Titanium) | **IDPF** (Intel) |
| RDMA 网卡 | 4x CX-7 VF (挂在 CPU) | **8x CX-8 PF** (GPUDirect 直连 GPU) |
| 网络栈 | IPv4 | **IPv6-only** |
| Boot Disk | pd-balanced 或 hyperdisk-balanced | **仅 hyperdisk-balanced** |
| GPU | 4x B200 (186 GB/GPU) | 4x B300 Ultra (**278 GB/GPU**) |
| RDMA VPC | 手动创建 4 子网 | **RoCE profile 自动创建** 1 个共享子网 |

## VPC 网络创建

### 管理网络 (IDPF)

```bash
export PROJECT=tencent-gcp-taiji-poc
export REGION=us-central1
export ZONE=us-central1-b

# 管理 VPC (MTU 8896, 官方推荐)
gcloud compute networks create $PREFIX-idpf-net \
  --subnet-mode=custom --mtu=8896 \
  --enable-ula-internal-ipv6 --project=$PROJECT

# 子网 (IPv6-only)
for N in 0 1; do
  gcloud compute networks subnets create $PREFIX-idpf-sub-$N \
    --network=$PREFIX-idpf-net --region=$REGION \
    --stack-type=IPV6_ONLY --ipv6-access-type=INTERNAL \
    --project=$PROJECT
done

# 防火墙
gcloud compute firewall-rules create $PREFIX-idpf-internal \
  --network=$PREFIX-idpf-net --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=10.0.0.0/8 --project=$PROJECT
```

### RDMA 网络 (CX-8, RoCE profile)

```bash
# 查询可用的 network profile
gcloud compute network-profiles list --filter="location.name=$ZONE" --project=$PROJECT

# 创建 RoCE VPC (使用 vpc-roce-metal profile)
gcloud compute networks create $PREFIX-rdma-net \
  --network-profile=$ZONE-vpc-roce-metal \
  --subnet-mode=custom --mtu=8896 --project=$PROJECT

# RoCE VPC 会自动创建子网 default-subnet-1-$PREFIX-rdma-net
# 所有 8 个 CX-8 MRDMA 接口共享这一个子网
```

> **vs GB200**: GB200 需要手动创建 4 个 RDMA 子网（每个 CX-7 一个）。GB300 用 `vpc-roce-metal` profile 自动创建 1 个共享子网，8 个 CX-8 接口全挂上去。

## VM 创建

### Placement Policy

```bash
# 使用已有的 12 个 placement policy (gb300-central-nvl72-policy-0001~0012)
# 每个对应一个 NVL72 sub-block (18 节点, 72 GPU)
gcloud compute resource-policies list --filter="name~gb300" --project=$PROJECT
```

### 创建 GB300 Bare Metal 实例

```bash
gcloud compute instances create $WORKER_NAME \
  --machine-type=a4x-maxgpu-4g-metal \
  --zone=$ZONE --project=$PROJECT \
  --boot-disk-size=1000GB --boot-disk-type=hyperdisk-balanced \
  --network-interface=nic-type=IDPF,network=$PREFIX-idpf-net,subnet=$PREFIX-idpf-sub-0,stack-type=IPV6_ONLY \
  --network-interface=nic-type=IDPF,network=$PREFIX-idpf-net,subnet=$PREFIX-idpf-sub-1,stack-type=IPV6_ONLY,no-address \
  --network-interface=subnet=default-subnet-1-$PREFIX-rdma-net,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=default-subnet-1-$PREFIX-rdma-net,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=default-subnet-1-$PREFIX-rdma-net,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=default-subnet-1-$PREFIX-rdma-net,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=default-subnet-1-$PREFIX-rdma-net,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=default-subnet-1-$PREFIX-rdma-net,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=default-subnet-1-$PREFIX-rdma-net,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=default-subnet-1-$PREFIX-rdma-net,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --reservation-affinity=specific \
  --reservation=projects/$RESERVATION_PROJECT/reservations/$RESERVATION_NAME/reservationBlocks/$BLOCK/reservationSubBlocks/$SUBBLOCK \
  --provisioning-model=RESERVATION_BOUND \
  --maintenance-policy=TERMINATE --restart-on-failure \
  --resource-policies=$PLACEMENT_POLICY \
  --scopes=cloud-platform
```

**vs GB200 的变化**:
1. `--machine-type`: `a4x-highgpu-4g` → `a4x-maxgpu-4g-metal`
2. `nic-type`: `GVNIC` → `IDPF`
3. 每个 `--network-interface` 加 `stack-type=IPV6_ONLY`
4. MRDMA 从 4 条增加到 **8 条**
5. RDMA 子网从 4 个独立子网 → 1 个共享子网
6. Reservation 支持 block/subblock 级别指定

## Startup Script

使用 v2 通用版（自动适配 GB200/GB300），见 [scripts/](../scripts/) 目录。

## 首次登录验证

```bash
# GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 预期: NVIDIA GB300, 284208 MiB (x4)

# 网卡
ip -br link show
ethtool -i <IDPF设备名>  # 确认 IDPF 驱动

# RDMA
rdma link show  # 预期: 8 个 mlx5 设备

# NVLink
nvidia-smi topo -m  # 查看 GPU 互联拓扑
```

## GB200 Baseline

GB200 环境搭建参考: [a4x/01-environment-setup/](../../a4x/01-environment-setup/)

## GB300 实测记录

| 步骤 | 状态 | 备注 |
|------|------|------|
| VPC 创建 | — | |
| VM 创建 | — | |
| GPU 验证 | — | |
| 网卡验证 | — | |
| RDMA 验证 | — | |
