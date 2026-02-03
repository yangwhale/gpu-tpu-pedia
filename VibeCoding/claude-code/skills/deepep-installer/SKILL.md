---
name: deepep-installer
description: This skill should be used when users need to install, configure, or troubleshoot DeepEP (DeepSeek Expert Parallelism) on NVIDIA GPU systems. It covers the complete installation workflow including CUDA, NVSHMEM with IBGDA support, and DeepEP itself. The skill is particularly useful for B200/H100/A100 GPUs with RoCE/InfiniBand networking, and includes comprehensive debugging capabilities for common installation failures.
---

# DeepEP Installer

## Overview

DeepEP 是 DeepSeek 的 Expert Parallelism 库，用于 MoE 模型的高性能 all-to-all 通信。使用 NVSHMEM IBGDA (InfiniBand GPUDirect Async) 实现 GPU 间 RDMA 通信。

**关键配置：GPU NIC Handler 模式（无需 GDRCopy）**

## 版本配置

| 组件 | 版本 | 说明 |
|------|------|------|
| CUDA Toolkit | 12.9 | sm_100 架构支持 |
| DOCA-OFED | 3.2.1 | 内核 6.14 兼容 |
| NVSHMEM | v3.5.19-1 | IBGDA 支持，无 GDRCopy |
| PyTorch | 2.10.0+cu129 | - |
| DeepEP | HEAD + PR #466 | GPU-NIC 映射补丁 |

## 快速安装

```bash
# 获取安装脚本
curl -O https://raw.githubusercontent.com/.../install-deepep.sh

# Phase 1: CUDA + DOCA + PeerMappingOverride
sudo bash install-deepep.sh
# 完成后重启
sudo reboot

# Phase 2: NVSHMEM + PyTorch + DeepEP
sudo bash install-deepep.sh

# 验证
source /opt/deepep/unified-env.sh
python3 -c "import deep_ep; print('DeepEP OK')"
```

## 环境配置

环境脚本: `/opt/deepep/unified-env.sh`

```bash
#!/bin/bash
export NVSHMEM_HOME=/opt/deepep/nvshmem
export CUDA_HOME=/usr/local/cuda

export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"

# NVSHMEM IBGDA 配置
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu      # GPU Handler 模式，无需 GDRCopy
export NVSHMEM_HCA_PREFIX=mlx5
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1

# PR #466: GPU 到 NIC 显式映射 (8 GPU : 8 NIC)
export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0:1,1:mlx5_1:1,2:mlx5_2:1,3:mlx5_3:1,4:mlx5_4:1,5:mlx5_5:1,6:mlx5_6:1,7:mlx5_7:1

export PYTHONPATH=/opt/deepep/python:${PYTHONPATH:-}
```

### 关键配置说明

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `NVSHMEM_IBGDA_NIC_HANDLER` | `gpu` | GPU 直接处理 NIC doorbell，无需 GDRCopy |
| `NVSHMEM_ENABLE_NIC_PE_MAPPING` | `1` | 启用 NIC-PE 映射，跨节点通信必需 |
| `DEEP_EP_DEVICE_TO_HCA_MAPPING` | `0:mlx5_0:1,...` | PR #466: GPU 到 NIC 显式映射 |

---

## 测试验证 (重点)

### 1. 安装验证

```bash
source /opt/deepep/unified-env.sh

# 验证 DeepEP 模块
python3 -c "import deep_ep; print('DeepEP OK')"

# 验证 NVSHMEM IBGDA (注意: 无 GDRCopy)
/opt/deepep/nvshmem/bin/nvshmem-info -a | grep -E "IBGDA|GDRCOPY"
# 预期输出:
# NVSHMEM_IBGDA_SUPPORT=ON
# NVSHMEM_GDRCOPY_SUPPORT=OFF

# 验证 PR #466 补丁
grep "_setup_device_hca_mapping" /opt/deepep/python/deep_ep/buffer.py
# 应该有输出

# 验证 HCA 设备
ls /sys/class/infiniband/
# 预期: mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7

# 验证 PeerMappingOverride
grep PeerMappingOverride /proc/driver/nvidia/params
# 预期: PeerMappingOverride: 1
```

### 2. Intranode 测试 (单节点)

测试单节点内 8 个 GPU 之间的 NVLink 通信。

```bash
source /opt/deepep/unified-env.sh
cd /opt/deepep/source/tests

# 运行测试
python3 test_intranode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8
```

**预期输出 (功能测试):**
```
[testing] Running with BF16, without top-k (async=False, previous=False) ... passed
[testing] Running with BF16, without top-k (async=True, previous=False) ... passed
[testing] Running with BF16, without top-k (async=False, previous=True) ... passed
...
[testing] Running with FP8, with top-k (async=True, previous=True) ... passed
```

**预期输出 (性能调优):**
```
[tuning] Best dispatch (FP8): SMs 24, NVL chunk 24: 316.85 GB/s (NVL), 255 us
[tuning] Best dispatch (BF16): SMs 24, NVL chunk 12: 446.09 GB/s (NVL), 352 us
[tuning] Best combine: SMs 24, NVL chunk 15: 355.45 GB/s (NVL), 442 us
```

**性能基准 (B200 8-GPU):**

| 操作 | 数据类型 | NVLink 带宽 | 延迟 |
|------|----------|-------------|------|
| Dispatch | FP8 | 300-320 GB/s | ~250 µs |
| Dispatch | BF16 | 440-450 GB/s | ~350 µs |
| Combine | BF16 | 350-360 GB/s | ~450 µs |

### 3. Internode 测试 (跨节点)

测试两个节点之间的 RDMA 通信。**这是最关键的测试，验证 PR #466 是否生效。**

**Node 1 (Master, 例如 b5=10.8.0.31):**
```bash
source /opt/deepep/unified-env.sh
cd /opt/deepep/source/tests
export WORLD_SIZE=2 RANK=0 MASTER_ADDR=10.8.0.31 MASTER_PORT=29500
python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8
```

**Node 2 (Worker, 例如 b6=10.8.0.32):**
```bash
source /opt/deepep/unified-env.sh
cd /opt/deepep/source/tests
export WORLD_SIZE=2 RANK=1 MASTER_ADDR=10.8.0.31 MASTER_PORT=29500
python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8
```

**预期输出 (功能测试):**
```
[testing] Running with BF16, without top-k (async=False, previous=False) ... passed
...
[testing] Running with FP8, with top-k (async=True, previous=True) ... passed
```

**预期输出 (性能调优):**
```
[tuning] Best dispatch (FP8): SMs 24, NVL 16, RDMA 24: 50.06 GB/s (RDMA), 167.07 GB/s (NVL)
[tuning] Best dispatch (BF16): SMs 24, NVL 16, RDMA 20: 80.47 GB/s (RDMA), 268.55 GB/s (NVL)
[tuning] Best combine: SMs 24, NVL 5, RDMA 20: 73.50 GB/s (RDMA), 245.29 GB/s (NVL)
```

**性能基准 (B200 2节点 16-GPU):**

| 操作 | 数据类型 | RDMA 带宽 | NVLink 带宽 |
|------|----------|-----------|-------------|
| Dispatch | FP8 | 36-50 GB/s | 120-170 GB/s |
| Dispatch | BF16 | 65-80 GB/s | 210-270 GB/s |
| Combine | BF16 | 70-75 GB/s | 230-250 GB/s |

### 4. NIC 流量验证 (PR #466 效果)

**关键检查: 确保所有 8 个 NIC 都有流量**

在 internode 测试期间或之后，检查 NIC 流量分布：

```bash
# 检查 NIC 流量计数器
for i in 0 1 2 3 4 5 6 7; do
  echo -n "mlx5_$i: "
  cat /sys/class/infiniband/mlx5_$i/ports/1/counters/port_xmit_data
done
```

**PR #466 修复前 (单 NIC 瓶颈):**
```
mlx5_0: 17100000000000  <- 所有流量集中在一个 NIC
mlx5_1: 0
mlx5_2: 0
mlx5_3: 0
mlx5_4: 0
mlx5_5: 0
mlx5_6: 0
mlx5_7: 0
```
RDMA 带宽: ~6.5 GB/s (单 NIC 限制)

**PR #466 修复后 (8 NIC 均衡):**
```
mlx5_0: 17100000000000
mlx5_1: 4370000000  <- 新增流量
mlx5_2: 4370000000  <- 新增流量
mlx5_3: 4370000000  <- 新增流量
mlx5_4: 4370000000  <- 新增流量
mlx5_5: 4370000000  <- 新增流量
mlx5_6: 4370000000  <- 新增流量
mlx5_7: 4370000000  <- 新增流量
```
RDMA 带宽: **50-80 GB/s** (8 NIC 并行)

### 5. 调试输出验证

如果需要验证 GPU-NIC 映射是否正确，可以临时启用调试输出：

```bash
# 在 buffer.py 的 _setup_device_hca_mapping 方法中添加 print 语句
# 测试时会输出每个 GPU 对应的 NIC
```

**预期调试输出:**
```
DEBUG_HCA_MAPPING: cuda_device=0, mapped_device=0, NVSHMEM_HCA_LIST=mlx5_0:1
DEBUG_HCA_MAPPING: cuda_device=1, mapped_device=1, NVSHMEM_HCA_LIST=mlx5_1:1
DEBUG_HCA_MAPPING: cuda_device=2, mapped_device=2, NVSHMEM_HCA_LIST=mlx5_2:1
...
DEBUG_HCA_MAPPING: cuda_device=7, mapped_device=7, NVSHMEM_HCA_LIST=mlx5_7:1
```

---

## 常见问题

### 问题 1: RDMA 带宽过低 (~6.5 GB/s)

**症状:** Internode RDMA 带宽仅 ~6.5 GB/s，远低于预期的 40+ GB/s

**根因:** PR #466 补丁未生效，所有 GPU 使用默认 mlx5_0

**验证:**
```bash
# 检查 NIC 流量分布
for i in 0 1 2 3 4 5 6 7; do
  echo -n "mlx5_$i: "
  cat /sys/class/infiniband/mlx5_$i/ports/1/counters/port_xmit_data
done
# 如果只有 mlx5_0 有流量，说明 PR #466 未生效
```

**修复:**
1. 确认 `_setup_device_hca_mapping` 存在于 `/opt/deepep/python/deep_ep/buffer.py`
2. 确认方法被调用（在 IBGDA 初始化之前）
3. 确认 `DEEP_EP_DEVICE_TO_HCA_MAPPING` 环境变量已设置

### 问题 2: Internode Dispatch 挂起

**症状:** CPU NIC Handler 模式下 internode dispatch 操作挂起

**根因:** CPU NIC Handler 在 GCP B200 环境下不兼容

**修复:** 切换到 GPU NIC Handler 模式
```bash
# 旧配置 (挂起)
export NVSHMEM_IBGDA_NIC_HANDLER=cpu

# 新配置 (正常)
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
```

### 问题 3: DOCA-OFED 内核模块编译失败

**症状:** DOCA 3.0.0 在内核 6.14.0+ 上编译失败

**修复:** 升级到 DOCA 3.2.1

### 问题 4: HCA 前缀错误

**症状:** `device mlx5_0 is not supported (expected HCA interface: rocep)`

**修复:**
```bash
# 检查实际 HCA 设备
ls /sys/class/infiniband/
# 根据输出设置 HCA_PREFIX
export NVSHMEM_HCA_PREFIX=mlx5  # 或 rocep
```

### 问题 5: PyTorch bundled NVSHMEM 冲突

**症状:** `NVSHMEM device library version does not match`

**修复:** 使用 LD_PRELOAD 强制加载自定义 NVSHMEM
```bash
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"
```

---

## 技术原理

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

## 版本历史

- **2026-02-03**: 重大简化 - GPU NIC Handler 模式
  - 移除 GDRCopy 依赖
  - 添加 PR #466 GPU-NIC 映射
  - RDMA 带宽从 6.5 GB/s 提升到 50-80 GB/s

- **2026-02-01**: 初始版本
  - NVSHMEM v3.5.19-1 with IBGDA
  - CPU NIC Handler 模式 (需要 GDRCopy)
