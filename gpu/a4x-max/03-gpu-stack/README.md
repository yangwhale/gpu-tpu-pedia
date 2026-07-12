# GB300 (A4X Max) GPU Stack 安装

GPU 驱动、IMEX、DRA、NCCL/GIB 等 GPU 基础设施组件安装。

## 与 GB200 的核心差异

| 组件 | GB200 (A4X) | GB300 (A4X Max) |
|------|-------------|-----------------|
| GPU Driver | R580 分支 | **R580.95.05+** (实测 580.159.04) |
| CUDA | 12.x | **13.0** |
| RDMA 配置 | `nccl-rdma-installer` DaemonSet | **`asapd-lite`** DaemonSet |
| DRA Driver | v0.4.0 / 25.12.0 | **25.8.0** (推荐) |
| DOCA OFED | 内核自带 mlnx | **doca-ofed-userspace** (CX-8 需要) |
| NCCL 版本 | 2.27.x | **2.28.9+** (需升级) |
| GIB | v1.1.2 (arm64) | 待确认版本 |
| IMEX | DRA 自动部署 | 手动部署 (无 DRA RBAC 权限时) |

## GPU Driver

GB300 COS 节点自带 NVIDIA driver，GKE 通过 `gpu-driver-version=latest` 自动安装。

```bash
# 验证
nvidia-smi
# 预期: Driver 580.159.04, CUDA 13.0, 4x NVIDIA GB300, 284208 MiB/GPU
```

## asapd-lite (RDMA 配置)

GB300 使用 `asapd-lite` 替代 GB200 的 `nccl-rdma-installer`:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/refs/heads/master/asapd-lite-installer/asapd-lite-installer-a4x-max-bm-cos.yaml

# 验证
kubectl get ds -n kube-system asapd-lite
# READY 数应等于 GB300 节点数
```

## DRA Driver (ComputeDomain + IMEX)

> **权限要求**: DRA helm install 需要 `container.admin` IAM 角色来创建 ClusterRole/ClusterRoleBinding。如果没有此权限，见下方"手动部署 IMEX"替代方案。

### 标准安装 (需要 container.admin)

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update

kubectl create ns nvidia-dra-driver-gpu

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --set controller.args.v=4 --set kubeletPlugin.args.v=4 \
  --version="25.8.0" \
  --create-namespace \
  --namespace nvidia-dra-driver-gpu \
  -f values-gb300.yaml
```

### 手动部署 IMEX (无 container.admin 权限时)

单节点 4 卡测试不需要 IMEX（节点内 NVLink 硬件直连）。多节点 MNNVL 需要 IMEX，可手动部署:

```bash
# TODO: IMEX DaemonSet YAML (从 NGC 拉 IMEX 镜像)
# 手动给节点打 ComputeDomain 标签
# 挂载 /dev/nvidia-caps-imex-channels 设备
```

## DOCA OFED + NCCL 升级

GB300 CX-8 需要 DOCA OFED 用户空间库。在训练容器镜像中安装:

```bash
apt update -y && apt install -y curl
export DOCA_URL="https://linux.mellanox.com/public/repo/doca/3.1.0/ubuntu22.04/arm64-sbsa/"
curl https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | \
  gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" \
  > /etc/apt/sources.list.d/doca.list
apt update
apt -y install doca-ofed-userspace
apt install --only-upgrade --allow-change-held-packages -y libnccl2 libnccl-dev
```

> NeMo 26.06 容器已内置这些依赖。自定义镜像需手动安装。

## GIB (Google InfiniBand)

```bash
# GIB 环境变量 (与 GB200 相同)
source /usr/local/gib/scripts/set_nccl_env.sh
```

## 验证清单

```bash
# 1. GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 2. RDMA
rdma link show  # 预期: 8 个设备

# 3. NCCL
python3 -c "import torch; print(torch.cuda.device_count())"  # 预期: 4

# 4. NVLink (单节点)
nvidia-smi topo -m

# 5. asapd-lite
kubectl get ds -n kube-system asapd-lite
```

## GB200 参考

GB200 GPU Stack 文档: [a4x/03-gpu-stack/](../../a4x/03-gpu-stack/)

## GB300 实测记录

| 组件 | 版本 | 状态 | 备注 |
|------|------|------|------|
| GPU Driver | 580.159.04 | ✅ 已验证 | nvidia-smi 正常 |
| CUDA | 13.0 | ✅ 已验证 | |
| asapd-lite | — | ✅ 已部署 | DaemonSet READY |
| DRA Driver | — | ❌ 需要 container.admin | |
| DOCA OFED | — | — | 待容器内验证 |
| NCCL | — | — | 待验证版本 |
| GIB | — | — | 待验证 |
| IMEX | — | — | 单节点不需要 |
