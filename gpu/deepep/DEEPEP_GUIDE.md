# DeepEP 安装与配置指南 (GCP B200)

## 概述

| 项目 | 详情 |
|------|------|
| **测试日期** | 2026-02-03 |
| **目标主机** | b7-b10 (4 节点集群) |
| **GPU 型号** | NVIDIA B200 (180GB HBM3e) × 8/节点 |
| **驱动版本** | 580.126.09 |
| **网络** | RoCE 400 Gb/s (mlx5) |
| **操作系统** | Ubuntu 24.04, Kernel 6.14.0-1021-gcp |

---

## 1. 安装组件

| 组件 | 版本 | 说明 |
|------|------|------|
| CUDA Toolkit | 12.9 | sm_100 架构支持 |
| DOCA-OFED | 3.2.1 | 内核 6.14 兼容版本 |
| GDRCopy | - | 跳过 (GPU Handler 模式不需要) |
| NVSHMEM | v3.5.19-1 | IBGDA 支持，无 GDRCopy |
| PyTorch | 2.10.0+cu129 | - |
| DeepEP | 1.2.1+29d31c0 | 含 PR #466 补丁 |

### 安装流程

```bash
# 推荐使用优化版安装脚本 (无 GDRCopy)
sudo bash install-deepep.sh
```

**Phase 1 (重启前):**
1. CUDA 12.9 Toolkit 安装
2. DOCA-OFED 3.2.1 安装 (含内核模块)
3. PeerMappingOverride 配置
4. 系统重启

**Phase 2 (重启后):**
1. nvidia_peermem 模块加载
2. NVSHMEM v3.5.19-1 编译 (IBGDA only)
3. PyTorch 安装
4. DeepEP 编译 (含 PR #466)
5. 环境脚本生成

---

## 2. 环境配置

环境脚本位置: `/opt/deepep/unified-env.sh`

### 关键配置项

```bash
#!/bin/bash
# DeepEP Environment (GPU NIC Handler Mode)

export NVSHMEM_HOME=/opt/deepep/nvshmem
export CUDA_HOME=/usr/local/cuda

# Library paths
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# Force our NVSHMEM over PyTorch bundled version
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"

# NVSHMEM IBGDA Configuration
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA=1

# GPU NIC Handler (不需要 GDRCopy)
export NVSHMEM_IBGDA_NIC_HANDLER=gpu

export NVSHMEM_HCA_PREFIX=mlx5
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1

# Performance settings
export NVSHMEM_IBGDA_NUM_RC_PER_PE=8
export NVSHMEM_IBGDA_NUM_DCI=4

# PR #466: GPU 到 NIC 的显式映射
# 格式: <CUDA_DEVICE_ID>:<HCA_NAME>:<PORT>,...
export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0:1,1:mlx5_1:1,2:mlx5_2:1,3:mlx5_3:1,4:mlx5_4:1,5:mlx5_5:1,6:mlx5_6:1,7:mlx5_7:1

export PYTHONPATH=/opt/deepep/python:${PYTHONPATH:-}
```

### 配置说明

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `NVSHMEM_IBGDA_NIC_HANDLER` | `gpu` | GPU 直接处理 NIC doorbell，无需 GDRCopy |
| `NVSHMEM_IB_GID_INDEX` | `3` | IPv4-mapped RoCE v2 (跨节点通信) |
| `NVSHMEM_ENABLE_NIC_PE_MAPPING` | `1` | 启用 NIC-PE 映射 |
| `DEEP_EP_DEVICE_TO_HCA_MAPPING` | `0:mlx5_0:1,...` | GPU 到 NIC 的显式映射 (PR #466) |

---

## 3. 测试结果

### Intranode 测试 (单节点 8× B200)

**功能测试:** 24/24 通过

| 操作 | 数据类型 | 最佳配置 | NVLink 带宽 | 延迟 |
|------|----------|----------|-------------|------|
| Dispatch | FP8 | SMs 24, chunk 24 | **316.85 GB/s** | 255 µs |
| Dispatch | BF16 | SMs 24, chunk 12 | **446.09 GB/s** | 352 µs |
| Combine | BF16 | SMs 24, chunk 15 | **355.45 GB/s** | 442 µs |

### Internode 测试 (2 节点 16× B200)

**功能测试:** 32/32 通过

| 操作 | 数据类型 | RDMA 带宽 | NVLink 带宽 |
|------|----------|-----------|-------------|
| Dispatch | FP8 | **70-71 GB/s** | 231-235 GB/s |
| Dispatch | BF16 | **81 GB/s** | 265-271 GB/s |
| Combine | BF16 | **75 GB/s** | 245-253 GB/s |

### Internode 测试 (4 节点 32× B200)

**功能测试:** 全部通过

| 操作 | 数据类型 | RDMA 带宽 | NVLink 带宽 |
|------|----------|-----------|-------------|
| Dispatch | FP8 | **54 GB/s** | 108-111 GB/s |
| Dispatch | BF16 | **57-58 GB/s** | 114-118 GB/s |
| Combine | BF16 | **56-57 GB/s** | 113-116 GB/s |

**注意:** 4 节点的 RDMA 带宽略低于 2 节点，这是因为跨节点通信路径更复杂（每个节点需要与 3 个远程节点通信）。

### 测试启动方式

```bash
# Intranode 测试
source /opt/deepep/unified-env.sh
cd /opt/deepep/DeepEP/tests
python3 test_intranode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8

# 2节点 Internode 测试
# Node 1 (Master):
export WORLD_SIZE=2 RANK=0 MASTER_ADDR=<node1_ip> MASTER_PORT=29500
python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8

# Node 2 (Worker):
export WORLD_SIZE=2 RANK=1 MASTER_ADDR=<node1_ip> MASTER_PORT=29500
python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8

# 4节点 Internode 测试 (WORLD_SIZE=4, RANK=0-3)
# 在每个节点上设置对应的 RANK (0=master, 1-3=workers)
export WORLD_SIZE=4 RANK=<0-3> MASTER_ADDR=<node1_ip> MASTER_PORT=29500
python3 test_internode.py --num-tokens 2048 --hidden 7168 --num-experts 256 --num-topk 8
```

---

## 4. 问题排查记录

### 问题 1: RDMA 带宽过低 (~6.5 GB/s)

**症状:** RDMA 带宽仅 ~6.5 GB/s，远低于预期的 40+ GB/s

**根因:** 安装脚本的 PR #466 补丁应用顺序错误
1. PR #466 被应用到 `/opt/deepep/source/deep_ep/buffer.py`
2. 但已安装的 Python 模块位于 `/opt/deepep/python/deep_ep/buffer.py`
3. 复制操作发生在补丁应用之前，导致已安装模块缺少 GPU-to-NIC 映射功能
4. 所有 GPU 进程都使用默认的 mlx5_0，形成单 NIC 瓶颈

**验证方法:**
```bash
# 检查 NIC 流量分布
for i in 0 1 2 3 4 5 6 7; do
  echo -n "mlx5_$i: "
  cat /sys/class/infiniband/mlx5_$i/ports/1/counters/port_xmit_data
done

# 修复前: 仅 mlx5_0 有流量
# 修复后: 所有 8 个 NIC 都有流量
```

**修复:**
- 在复制模块后，同时复制包含 PR #466 补丁的 Python 源代码
- 验证 `_setup_device_hca_mapping` 方法存在于已安装模块

**性能对比:**

| 指标 | 修复前 (单 NIC) | 修复后 (8 NIC) | 提升 |
|------|-----------------|----------------|------|
| RDMA 带宽 (FP8) | ~6.5 GB/s | **36-50 GB/s** | **5.5-7.7x** |
| RDMA 带宽 (BF16) | ~6.7 GB/s | **65-80 GB/s** | **9.7-12x** |
| NIC 利用率 | 1/8 | 8/8 | ✅ |

### 问题 2: Internode Dispatch 挂起

**症状:** CPU NIC Handler 模式下 internode dispatch 操作挂起，所有 `moe_recv_expert_counter` 值为 -1

**根因:** CPU NIC Handler 在 GCP B200 环境下不兼容

**修复:** 切换到 GPU NIC Handler 模式
```bash
# OLD (挂起)
export NVSHMEM_IBGDA_NIC_HANDLER=cpu
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3:${GDRCOPY_HOME}/lib/libgdrapi.so.2"

# NEW (正常)
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"
```

### 问题 3: DOCA-OFED 内核模块编译失败

**症状:** DOCA 3.0.0 在内核 6.14.0-1021-gcp 上编译失败

**错误:**
```
drivers/net/ethernet/mellanox/mlx5/core/en_accel/ipsec.c:1325:35: error:
initialization of 'int (*)(struct net_device *, struct xfrm_state *,
struct netlink_ext_ack *)' from incompatible pointer type
```

**修复:** 升级到 DOCA 3.2.1，包含内核 6.14 兼容补丁

---

## 5. 技术原理

### GPU-NIC 拓扑 (GCP B200)

```
GPU0-GPU1 ↔ PIX ↔ NIC0-NIC1 (mlx5_0, mlx5_1)
GPU2-GPU3 ↔ PIX ↔ NIC2-NIC3 (mlx5_2, mlx5_3)
GPU4-GPU5 ↔ PIX ↔ NIC4-NIC5 (mlx5_4, mlx5_5)
GPU6-GPU7 ↔ PIX ↔ NIC6-NIC7 (mlx5_6, mlx5_7)
```

每对 GPU 与对应的 NIC 共享同一 PCIe switch。PR #466 的作用是确保每个 GPU 使用距离最近的 NIC。

### NVSHMEM IBGDA NIC Handler 模式

| 模式 | 环境变量 | GDRCopy | 说明 |
|------|----------|---------|------|
| **CPU Handler** | `NVSHMEM_IBGDA_NIC_HANDLER=cpu` | 需要 | CPU 处理 NIC doorbell |
| **GPU Handler** | `NVSHMEM_IBGDA_NIC_HANDLER=gpu` | 不需要 | GPU 直接处理 NIC doorbell |

在 GCP B200 环境中，**必须使用 GPU Handler 模式**，CPU Handler 模式会导致 internode 通信挂起。

### GID Table 配置

每个 NIC 有 4 个 GID 条目：
- GID[0]: Link-local IPv6 (RoCE v1)
- GID[1]: Link-local IPv6 (RoCE v2)
- GID[2]: IPv4-mapped (RoCE v1)
- GID[3]: IPv4-mapped (RoCE v2) ← 用于跨节点通信

### NVSHMEM 架构

- PyTorch Distributed: 16 ranks (8 GPUs × 2 nodes)
- NVSHMEM: 2 RDMA ranks (1 per node)
- 每个节点的 8 个 GPU 共享同一个 NVSHMEM rdma_rank
- `rdma_rank = global_rank / NUM_MAX_NVL_PEERS` (NUM_MAX_NVL_PEERS = 8)

---

## 6. NVSHMEM 配置参考

### 关键配置项

| 配置项 | 默认值 | 推荐值 | 说明 |
|--------|--------|--------|------|
| `NVSHMEM_SYMMETRIC_SIZE` | 1GB | 2-4GB | 对称堆内存大小 |
| `NVSHMEM_REMOTE_TRANSPORT` | ibrc | ibgda | 传输层 |
| `NVSHMEM_IB_ENABLE_IBGDA` | false | true | GPU 直接访问 |
| `NVSHMEM_ENABLE_NIC_PE_MAPPING` | false | true | NIC-PE 映射 |
| `NVSHMEM_DISABLE_P2P` | false | false | P2P 通信 |
| `NVSHMEM_DISABLE_CUDA_VMM` | false | true | CUDA VMM (GCP 需禁用) |
| `NVSHMEM_IBGDA_NUM_RC_PER_PE` | - | 8 | 每 PE 的 RC QP 数量 |
| `NVSHMEM_IBGDA_NUM_DCI` | - | 4 | DCI 数量 |

### 调试配置

```bash
# 启用调试输出
export NVSHMEM_DEBUG=WARN,INFO
export NVSHMEM_DEBUG_SUBSYS=INIT,TRANSPORT,P2P

# 调试输出到文件
export NVSHMEM_DEBUG_FILE=/tmp/nvshmem_%h_%p.log

# 启用错误检查 (影响性能)
export NVSHMEM_ENABLE_ERROR_CHECKS=true

# NVTX 性能分析
export NVSHMEM_NVTX=common
nsys profile --trace=nvtx,cuda ./your_application
```

### 性能验证

```bash
# 验证 RDMA 连接 (单 NIC)
ib_write_bw -d mlx5_0 -x 3 -F  # Server
ib_write_bw -d mlx5_0 -x 3 -F <server_ip>  # Client
# 预期: ~45 GB/s (360+ Gb/s)

# 验证 GPU-NIC 拓扑
nvidia-smi topo -m

# 验证 IB 设备
ibv_devinfo
ls /sys/class/infiniband/
```

---

## 7. 验证命令

```bash
# 验证 DeepEP 模块
source /opt/deepep/unified-env.sh
python3 -c "import deep_ep; print('DeepEP OK')"

# 验证 NVSHMEM IBGDA
/opt/deepep/nvshmem/bin/nvshmem-info -a | grep -E "IBGDA|GDRCOPY"
# 输出: NVSHMEM_IBGDA_SUPPORT=ON, NVSHMEM_GDRCOPY_SUPPORT=OFF

# 验证 PR #466 补丁
grep "_setup_device_hca_mapping" /opt/deepep/python/deep_ep/buffer.py
# 应该有输出

# 验证 HCA 设备
ls /sys/class/infiniband/
# 输出: mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7
```

---

## 8. 最佳实践

### 安装

1. **使用优化版脚本**: `install-deepep.sh` (跳过 GDRCopy)
2. **DOCA 版本**: 必须使用 3.2.1+ (内核 6.14 兼容)
3. **验证 PR #466**: 确保 `_setup_device_hca_mapping` 存在于已安装模块

### 运行

1. **始终 source 环境脚本**: `source /opt/deepep/unified-env.sh`
2. **验证 NIC Handler**: `echo $NVSHMEM_IBGDA_NIC_HANDLER` 应为 `gpu`
3. **验证 NIC 映射**: 运行测试后检查所有 8 个 NIC 都有流量

### 故障排查

1. **RDMA 带宽低**: 检查 NIC 流量分布，确保 PR #466 生效
2. **Dispatch 挂起**: 确认使用 GPU NIC Handler 模式
3. **DKMS 编译失败**: 升级 DOCA 到 3.2.1+

---

## 附录: 安装脚本

```bash
sudo bash install-deepep.sh
```

此脚本使用 GPU NIC Handler 模式，无需 GDRCopy，适用于 GCP B200 环境。

---

**文档版本**: 2026-02-03
**测试环境**: GCP B200 (8 GPU × 4 nodes, 32 GPUs total)
**NVSHMEM 版本**: v3.5.19-1
**DeepEP 版本**: HEAD + PR #466
