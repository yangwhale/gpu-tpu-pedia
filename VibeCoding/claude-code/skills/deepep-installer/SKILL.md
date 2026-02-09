---
name: deepep-installer
description: This skill should be used when users need to install, configure, or troubleshoot DeepEP (DeepSeek Expert Parallelism) on NVIDIA GPU systems. It covers the complete installation workflow including CUDA, DOCA-OFED, NVSHMEM with IBGDA support, and DeepEP itself. The skill is particularly useful for B200/H100/A100 GPUs with RoCE/InfiniBand networking, and includes comprehensive debugging capabilities for common installation failures.
---

# DeepEP Installer

## Overview

DeepEP 是 DeepSeek 的 Expert Parallelism 库，用于 MoE 模型的高性能 all-to-all 通信。使用 NVSHMEM IBGDA (InfiniBand GPUDirect Async) 实现 GPU 间 RDMA 通信。

**关键配置：GPU NIC Handler 模式（无需 GDRCopy 内核模块，但构建时需要 GDRCopy 库）**

**注意:** LSSD 存储请使用独立的 `lssd-mounter` skill。

## 版本配置

| 组件 | 版本 | 说明 |
|------|------|------|
| CUDA Toolkit | 12.9 | sm_100 架构支持 |
| DOCA-OFED | 3.2.1 | **必须安装**，将 HCA 从 rocep 转为 mlx5 |
| GDRCopy | latest | NVSHMEM 构建依赖（仅库，无需内核模块） |
| NVSHMEM | v3.5.19-1 | IBGDA 支持 |
| PyTorch | 2.9.1+cu129 | 与 SGLang 0.5.8 保持一致 |
| NumPy | latest | DeepEP 测试依赖 |
| DeepEP | HEAD + PR #466 | GPU-NIC 映射补丁 |

---

## 安装步骤 (已验证)

### 前提条件检查

```bash
# 检查 GPU
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 检查 CUDA
/usr/local/cuda/bin/nvcc --version
# 如果已安装: Cuda compilation tools, release 12.9

# 检查 HCA 设备
ls /sys/class/infiniband/
# 新安装: 显示 rocep* (需要安装 DOCA-OFED)
# 安装后: 显示 mlx5_0 mlx5_1 ... mlx5_7 (8个 mlx5 设备)

# 检查 PeerMappingOverride
grep PeerMappingOverride /proc/driver/nvidia/params
# 安装后预期: RegistryDwords: "PeerMappingOverride=0x1"
```

**根据检查结果执行相应的 Phase。如果 CUDA、DOCA-OFED 和 PeerMappingOverride 都已配置，可跳到 Phase 4。**

### Phase 1: CUDA Toolkit 12.9

```bash
# 检查是否已安装，如已安装则跳过
/usr/local/cuda/bin/nvcc --version 2>/dev/null && echo "CUDA 已安装，跳过 Phase 1"

# 如未安装，执行以下命令

sudo bash -c '
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -qq
apt-get install -y cuda-toolkit-12-9
ln -sf /usr/local/cuda-12.9 /usr/local/cuda
'
```

### Phase 2: DOCA-OFED 3.2.1

**仅当 `ls /sys/class/infiniband/` 显示 rocep* 时需要执行。**

```bash
# 检查是否已安装，如显示 mlx5 则跳过
ls /sys/class/infiniband/ | grep mlx5 && echo "DOCA-OFED 已安装，跳过 Phase 2"

# 如显示 rocep*，执行以下命令

sudo bash -c '
wget -qO - https://linux.mellanox.com/public/repo/doca/3.2.1/ubuntu24.04/x86_64/GPG-KEY-Mellanox.pub | apt-key add -
echo "deb https://linux.mellanox.com/public/repo/doca/3.2.1/ubuntu24.04/x86_64 ./" > /etc/apt/sources.list.d/doca.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y doca-ofed
'

# 此步骤需要较长时间 (编译内核模块)，完成后需要重启
```

### Phase 3: PeerMappingOverride + 重启

```bash
# 设置 PeerMappingOverride
sudo bash -c '
echo "options nvidia NVreg_RegistryDwords=PeerMappingOverride=0x1" > /etc/modprobe.d/nvidia-peermapping.conf
'

# 重启以应用 DOCA-OFED 和 PeerMappingOverride
sudo reboot
```

### Phase 4: 重启后验证

```bash
# 验证 HCA 设备 (必须是 mlx5_*)
ls /sys/class/infiniband/
# 预期: mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7
# 如果仍然是 rocep*，说明 DOCA-OFED 未正确安装

# 验证 PeerMappingOverride
grep PeerMappingOverride /proc/driver/nvidia/params
# 预期: RegistryDwords: "PeerMappingOverride=0x1"

# 验证 HCA 状态
ibv_devinfo -d mlx5_0 | head -20
# 预期: state: PORT_ACTIVE
```

### Phase 5: 构建依赖

```bash
# 安装构建工具
sudo apt-get install -y cmake ninja-build python3-pip python3-venv libibverbs-dev rdma-core ibverbs-utils

# 创建安装目录
sudo mkdir -p /opt/deepep
sudo chown $USER:$USER /opt/deepep
```

### Phase 6: GDRCopy 库 (NVSHMEM 构建依赖)

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

cd /opt/deepep
git clone --depth 1 https://github.com/NVIDIA/gdrcopy.git gdrcopy-src
cd gdrcopy-src

# 只构建库 (无需内核模块)
make -j$(nproc) prefix=/opt/deepep/gdrcopy lib lib_install

# 验证
ls /opt/deepep/gdrcopy/lib/libgdrapi.so*
ls /opt/deepep/gdrcopy/include/gdrapi.h
```

### Phase 7: NVSHMEM v3.5.19-1

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

cd /opt/deepep
git clone --depth 1 --branch v3.5.19-1 https://github.com/NVIDIA/nvshmem.git nvshmem-src
cd nvshmem-src
mkdir -p build && cd build

# 配置 (使用 GDRCopy 头文件)
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/opt/deepep/nvshmem \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DNVSHMEM_IBGDA_SUPPORT=ON \
    -DNVSHMEM_MPI_SUPPORT=OFF \
    -DNVSHMEM_SHMEM_SUPPORT=OFF \
    -DCUDA_HOME=$CUDA_HOME \
    -DGDRCOPY_INCLUDE=/opt/deepep/gdrcopy/include

# 构建 (可能需要 30+ 分钟)
make -j$(nproc)

# 手动安装 (跳过测试编译)
mkdir -p /opt/deepep/nvshmem/{lib,include,bin}
cp -a src/lib/*.so* /opt/deepep/nvshmem/lib/
cp -a src/lib/*.a /opt/deepep/nvshmem/lib/
cp -r ../src/include/* /opt/deepep/nvshmem/include/

# 验证
ls /opt/deepep/nvshmem/lib/libnvshmem_host.so*
ls /opt/deepep/nvshmem/lib/nvshmem_transport_ibgda.so*
```

### Phase 8: PyTorch 2.9.1+cu129 + NumPy

```bash
python3 -m pip install --break-system-packages \
    torch==2.9.1+cu129 \
    numpy \
    --index-url https://download.pytorch.org/whl/cu129

# 验证
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

**注意**: PyTorch 2.9.1 是 SGLang 0.5.8 的硬性依赖，使用此版本可以避免后续安装 SGLang 时重新编译 DeepEP。

### Phase 9: DeepEP + PR #466

```bash
cd /opt/deepep
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP

# 获取 PR #466 (GPU-NIC 映射)
git fetch origin pull/466/head:pr-466
git checkout pr-466

# 构建 (关键: TORCH_CUDA_ARCH_LIST 和 NVSHMEM_DIR)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=10.0    # B200 = sm_100, H100 = 9.0, A100 = 8.0
export NVSHMEM_DIR=/opt/deepep/nvshmem  # 使用自编译的 NVSHMEM (带 IBGDA)
python3 setup.py build_ext --inplace

# 验证编译配置
# 输出应显示: Arch list: 10.0, NVSHMEM path: /opt/deepep/nvshmem

# 验证
python3 -c "import deep_ep; print('DeepEP OK:', deep_ep.__file__)"
grep "_setup_device_hca_mapping" deep_ep/buffer.py && echo "PR #466 OK"
```

### Phase 10: 创建环境脚本

```bash
cat > /opt/deepep/unified-env.sh << 'EOF'
#!/bin/bash
export NVSHMEM_HOME=/opt/deepep/nvshmem
export CUDA_HOME=/usr/local/cuda

export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"

# NVSHMEM IBGDA 配置
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu      # GPU Handler 模式，无需 GDRCopy 内核模块
export NVSHMEM_HCA_PREFIX=mlx5
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1

# PR #466: GPU 到 NIC 显式映射 (8 GPU : 8 NIC)
export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0:1,1:mlx5_1:1,2:mlx5_2:1,3:mlx5_3:1,4:mlx5_4:1,5:mlx5_5:1,6:mlx5_6:1,7:mlx5_7:1

export PYTHONPATH=/opt/deepep/DeepEP:${PYTHONPATH:-}
EOF

chmod +x /opt/deepep/unified-env.sh
```

### 最终验证

```bash
source /opt/deepep/unified-env.sh

# 验证 DeepEP
python3 -c "import deep_ep; print('DeepEP OK')"

# 验证 HCA
ls /sys/class/infiniband/
# 预期: mlx5_0 ~ mlx5_7

# 验证 PeerMappingOverride
grep PeerMappingOverride /proc/driver/nvidia/params
# 预期: PeerMappingOverride=0x1
```

---

## 环境配置

环境脚本: `/opt/deepep/unified-env.sh`

### 关键配置说明

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `NVSHMEM_IBGDA_NIC_HANDLER` | `gpu` | GPU 直接处理 NIC doorbell，无需 GDRCopy 内核模块 |
| `NVSHMEM_HCA_PREFIX` | `mlx5` | DOCA-OFED 安装后的 HCA 前缀 |
| `NVSHMEM_ENABLE_NIC_PE_MAPPING` | `1` | 启用 NIC-PE 映射，跨节点通信必需 |
| `DEEP_EP_DEVICE_TO_HCA_MAPPING` | `0:mlx5_0:1,...` | PR #466: GPU 到 NIC 显式映射 |
| `LD_PRELOAD` | `libnvshmem_host.so.3` | 覆盖 PyTorch 自带的 NVSHMEM |

---

## 测试验证

### 1. 安装验证

```bash
source /opt/deepep/unified-env.sh

# 验证 DeepEP 模块
python3 -c "import deep_ep; print('DeepEP OK')"

# 验证 PR #466 补丁
grep "_setup_device_hca_mapping" /opt/deepep/DeepEP/deep_ep/buffer.py
# 应该有输出

# 验证 HCA 设备
ls /sys/class/infiniband/
# 预期: mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7

# 验证 PeerMappingOverride
grep PeerMappingOverride /proc/driver/nvidia/params
# 预期: PeerMappingOverride=0x1
```

### 2. Intranode 测试 (单节点)

```bash
source /opt/deepep/unified-env.sh
cd /opt/deepep/DeepEP/tests

python3 test_intranode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8
```

**性能基准 (B200 8-GPU):**

| 操作 | 数据类型 | NVLink 带宽 | 延迟 |
|------|----------|-------------|------|
| Dispatch | FP8 | 300-320 GB/s | ~250 µs |
| Dispatch | BF16 | 440-450 GB/s | ~350 µs |
| Combine | BF16 | 350-360 GB/s | ~450 µs |

### 3. Internode 测试 (跨节点)

**Node 1 (Master):**
```bash
source /opt/deepep/unified-env.sh
cd /opt/deepep/DeepEP/tests
export WORLD_SIZE=2 RANK=0 MASTER_ADDR=<node1_ip> MASTER_PORT=29500
python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8
```

**Node 2 (Worker):**
```bash
source /opt/deepep/unified-env.sh
cd /opt/deepep/DeepEP/tests
export WORLD_SIZE=2 RANK=1 MASTER_ADDR=<node1_ip> MASTER_PORT=29500
python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8
```

**性能基准 (B200 2节点 16-GPU):**

| 操作 | 数据类型 | RDMA 带宽 | NVLink 带宽 |
|------|----------|-----------|-------------|
| Dispatch | FP8 | 70-71 GB/s | 231-235 GB/s |
| Dispatch | BF16 | 81 GB/s | 265-271 GB/s |
| Combine | BF16 | 75 GB/s | 245-253 GB/s |

### 4. 4节点 Internode 测试

```bash
# 在 4 个节点上并行执行 (示例使用 b7-b10)
# Node 1 (Master): RANK=0
# Node 2-4 (Workers): RANK=1,2,3

source /opt/deepep/unified-env.sh
cd /opt/deepep/DeepEP/tests
export WORLD_SIZE=4 RANK=<0-3> MASTER_ADDR=<node1_ip> MASTER_PORT=29500
python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8
```

**性能基准 (B200 4节点 32-GPU):**

| 操作 | 数据类型 | RDMA 带宽 | NVLink 带宽 |
|------|----------|-----------|-------------|
| Dispatch | FP8 | 54 GB/s | 108-111 GB/s |
| Dispatch | BF16 | 57-58 GB/s | 114-118 GB/s |
| Combine | BF16 | 56-57 GB/s | 113-116 GB/s |

### 5. NIC 流量验证 (PR #466 效果)

```bash
# 检查 NIC 流量计数器
for i in 0 1 2 3 4 5 6 7; do
  echo -n "mlx5_$i: "
  cat /sys/class/infiniband/mlx5_$i/ports/1/counters/port_xmit_data
done
```

**PR #466 修复后:** 所有 8 个 NIC 都应有流量分布。

---

## 常见问题

### 问题 1: HCA 仍为 rocep* (未变成 mlx5_*)

**症状:** `ls /sys/class/infiniband/` 显示 rocep* 而非 mlx5_*

**根因:** DOCA-OFED 3.2.1 未正确安装或未重启

**修复:**
```bash
# 确认 DOCA-OFED 已安装
dpkg -l | grep doca-ofed

# 如果未安装，重新安装
sudo apt-get install -y doca-ofed

# 重启
sudo reboot
```

### 问题 2: NVSHMEM 构建失败 (GDRCopy 错误)

**症状:** cmake 报错 `GDRCOPY_INCLUDE not found`

**修复:** 构建 GDRCopy 库
```bash
cd /opt/deepep
git clone --depth 1 https://github.com/NVIDIA/gdrcopy.git gdrcopy-src
cd gdrcopy-src
make -j$(nproc) prefix=/opt/deepep/gdrcopy lib lib_install

# 重新运行 cmake 时添加
-DGDRCOPY_INCLUDE=/opt/deepep/gdrcopy/include
```

### 问题 3: RDMA 带宽过低 (~6.5 GB/s)

**症状:** Internode RDMA 带宽仅 ~6.5 GB/s

**根因:** PR #466 未生效，所有 GPU 使用默认 mlx5_0

**修复:**
1. 确认使用 PR #466 分支: `cd /opt/deepep/DeepEP && git branch`
2. 确认环境变量: `echo $DEEP_EP_DEVICE_TO_HCA_MAPPING`
3. 检查 NIC 流量分布

### 问题 4: Internode Dispatch 挂起

**症状:** CPU NIC Handler 模式下 internode dispatch 操作挂起

**修复:** 使用 GPU NIC Handler 模式
```bash
export NVSHMEM_IBGDA_NIC_HANDLER=gpu  # 不是 cpu
```

### 问题 5: PyTorch bundled NVSHMEM 冲突

**症状:** `NVSHMEM device library version does not match`

**修复:** 使用 LD_PRELOAD 强制加载自定义 NVSHMEM
```bash
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"
```

### 问题 6: 构建 DeepEP 时权限错误

**症状:** `Permission denied: /opt/deepep/DeepEP/build`

**修复:** 确保目录归属当前用户
```bash
sudo chown -R $USER:$USER /opt/deepep/DeepEP
```

### 问题 7: CUDA 符号找不到 (named symbol not found)

**症状:** 运行测试时报错 `CUDA error: 'named symbol not found'`

**根因:** DeepEP 编译时的 CUDA 架构与 GPU 不匹配

**修复:** 设置正确的 CUDA 架构重新编译
```bash
cd /opt/deepep/DeepEP
rm -rf build deep_ep/*.so
export TORCH_CUDA_ARCH_LIST=10.0  # B200=10.0, H100=9.0, A100=8.0
export NVSHMEM_DIR=/opt/deepep/nvshmem
python3 setup.py build_ext --inplace
```

### 问题 8: Internode dispatch 超时 (timeout dispatch CPU)

**症状:** `RuntimeError: DeepEP error: timeout (dispatch CPU)`，所有 `num_recv_tokens: -1`

**根因:** DeepEP 编译时使用了 PyTorch 自带的 NVSHMEM（无 IBGDA 支持）

**验证:** 编译时检查输出，应显示：
```
> NVSHMEM path: /opt/deepep/nvshmem  # 正确
# 而非
> NVSHMEM path: /home/.../site-packages/nvidia/nvshmem  # 错误
```

**修复:** 设置 `NVSHMEM_DIR` 重新编译
```bash
export NVSHMEM_DIR=/opt/deepep/nvshmem
python3 setup.py build_ext --inplace
```

---

## 技术原理

### GCP B200 RDMA 驱动演变

```
GCP 镜像预装 (基础 RDMA)     安装 DOCA-OFED 3.2.1 后
         │                            │
    rocep145s0                    mlx5_0
    rocep146s0      ─────>        mlx5_1
    rocep152s0                    mlx5_2
       ...                          ...
    rocep206s0                    mlx5_7
```

**关键:** DOCA-OFED 替换基础驱动，将 RoCE 设备名从 rocep* 改为 mlx5_*

### GPU-NIC 拓扑 (GCP B200)

```
GPU0-GPU1 ↔ PIX ↔ NIC0-NIC1 (mlx5_0, mlx5_1)
GPU2-GPU3 ↔ PIX ↔ NIC2-NIC3 (mlx5_2, mlx5_3)
GPU4-GPU5 ↔ PIX ↔ NIC4-NIC5 (mlx5_4, mlx5_5)
GPU6-GPU7 ↔ PIX ↔ NIC6-NIC7 (mlx5_6, mlx5_7)
```

每对 GPU 与对应的 NIC 共享同一 PCIe switch。PR #466 确保每个 GPU 使用距离最近的 NIC。

### NVSHMEM 架构

- PyTorch Distributed: 16 ranks (8 GPUs × 2 nodes)
- NVSHMEM: 2 RDMA ranks (1 per node)
- 每个节点的 8 个 GPU 共享同一个 NVSHMEM rdma_rank
- `rdma_rank = global_rank / NUM_MAX_NVL_PEERS` (NUM_MAX_NVL_PEERS = 8)

---

## GCP 磁盘镜像创建 (预装环境复用)

完成 DeepEP 安装后，可创建磁盘镜像用于快速扩容新机器。

### 创建流程

```bash
# 1. 停止实例 (如果在 MIG 中，先 abandon)
gcloud compute instance-groups managed abandon-instances <MIG_NAME> \
    --instances=<INSTANCE_NAME> --zone=<ZONE>
gcloud compute instances stop <INSTANCE_NAME> --zone=<ZONE>

# 2. 创建镜像
gcloud compute images create <IMAGE_NAME> \
    --source-disk=<INSTANCE_NAME> \
    --source-disk-zone=<ZONE> \
    --family=chrisya-b200-deepep \
    --storage-location=<REGION>

# 3. 基于镜像创建新的 Instance Template
# (需要包含 MRDMA 网络接口配置，见下方说明)

# 4. 更新 MIG 使用新 Template
gcloud compute instance-groups managed set-instance-template <MIG_NAME> \
    --template=<NEW_TEMPLATE> --zone=<ZONE>

# 5. 扩容
gcloud compute instance-groups managed resize <MIG_NAME> \
    --size=<N> --zone=<ZONE>
```

### 镜像包含内容

使用 DeepEP 预装镜像创建的新实例已包含：
- CUDA 12.9
- DOCA-OFED 3.2.1 (mlx5 HCA)
- PeerMappingOverride 配置
- GDRCopy 库
- NVSHMEM v3.5.19-1 (IBGDA)
- DeepEP + PR #466
- PyTorch 2.9.1+cu129
- unified-env.sh 环境脚本

### 新实例启动后验证

```bash
# 验证 DeepEP (需要先 source 环境)
source /opt/deepep/unified-env.sh
python3 -c "import deep_ep; print('DeepEP OK')"

# 验证 HCA
ls /sys/class/infiniband/
# 预期: mlx5_0 ~ mlx5_7

# 验证 PeerMappingOverride
grep PeerMappingOverride /proc/driver/nvidia/params
```

### 注意事项

- **MRDMA 网络接口:** Instance Template 中必须包含 8 个 MRDMA 接口，且 MRDMA 接口不能配置 external NAT（使用 `no-address`）
- **LSSD:** 磁盘镜像不包含 LSSD 数据，新实例需重新挂载 LSSD 并下载模型权重
- **SGLang:** 镜像不包含 SGLang，需在新实例上单独安装（使用 `/sglang-installer` skill）

---

## 版本历史

- **2026-02-08**: 磁盘镜像和 1P1D 实战经验
  - **NEW**: GCP 磁盘镜像创建流程（预装环境复用）
  - **NEW**: 镜像内容清单和新实例验证步骤
  - b1+b2 并行安装 → 镜像创建 → b3 扩容验证通过
  - 镜像: chrisya-b200-u2404-nv580-deepep-ase1-260208

- **2026-02-03**: 4 节点测试验证
  - b7-b10 4 节点 32-GPU 测试通过
  - RDMA 54-58 GB/s, NVLink 108-118 GB/s
  - 添加 4 节点测试文档

- **2026-02-03**: 2 节点验证通过
  - b9-b10 测试 RDMA 66-76 GB/s, NVLink 217-248 GB/s
  - 优化 Phase 1-3 检查逻辑（先检查后安装）

- **2026-02-03**: 移除 LSSD 安装步骤
  - LSSD 请使用独立的 `lssd-mounter` skill
  - Phase 编号从 11 减少到 10

- **2026-02-03**: 修复 DeepEP 编译关键问题
  - Phase 9 添加 `TORCH_CUDA_ARCH_LIST=10.0` (B200 GPU sm_100)
  - Phase 9 添加 `NVSHMEM_DIR=/opt/deepep/nvshmem` (使用自编译 NVSHMEM)
  - 新增问题 7: CUDA 符号找不到的解决方案
  - 新增问题 8: Internode dispatch 超时的解决方案
  - Internode 测试验证通过: RDMA 70-81 GB/s, NVLink 231-271 GB/s

- **2026-02-03**: 添加 NumPy 依赖
  - Phase 8 添加 numpy 安装（DeepEP 测试必需）

- **2026-02-03**: 完善安装文档
  - 明确 DOCA-OFED 3.2.1 必须安装（rocep → mlx5）
  - 添加 GDRCopy 库构建步骤（NVSHMEM 构建依赖）
  - 添加详细的分步安装指南
  - 修复权限问题文档

- **2026-02-03**: GPU NIC Handler 模式
  - 移除 GDRCopy 内核模块依赖
  - 添加 PR #466 GPU-NIC 映射
  - RDMA 带宽从 6.5 GB/s 提升到 50-80 GB/s

- **2026-02-01**: 初始版本
  - NVSHMEM v3.5.19-1 with IBGDA
  - CPU NIC Handler 模式 (需要 GDRCopy)
