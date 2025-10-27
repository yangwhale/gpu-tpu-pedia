# DeepEP on GKE A4 故障排查指南

## 问题总结

在 GKE A4 (NVIDIA B200 GPU) 集群上运行 DeepEP 时遇到的 NCCL/RDMA 配置问题。

---

## 根源问题

### 1. libibverbs 库版本冲突

**现象**：
```
libibverbs: Warning: couldn't load driver 'libmlx5-rdmav57.so'
NCCL WARN Failed to initialize any NET plugin
```

**根本原因**：
- `/usr/local/nvidia/lib64/` 包含 rdmav57 版本的 libibverbs（1.14.56.0）
- 系统 `/usr/lib/x86_64-linux-gnu/` 包含 rdmav34 版本（1.14.50.0）
- 两个版本的符号不兼容，导致库加载失败

**诊断命令**：
```bash
# 检查库版本
ls -la /usr/local/nvidia/lib64/libibverbs*
ls -la /usr/lib/x86_64-linux-gnu/libibverbs*

# 检查驱动加载
ibv_devinfo  # 需要正确的 LD_LIBRARY_PATH
```

---

### 2. RDMA 驱动文件路径问题

**现象**：
```
libibverbs: Warning: couldn't load driver 'libmlx5-rdmav57.so': 
libmlx5-rdmav57.so: cannot open shared object file: No such file or directory
```

**根本原因**：
- `libibverbs` 在以下路径查找驱动：
  1. `/usr/lib/x86_64-linux-gnu/libibverbs/` （系统默认）
  2. `$RDMAV_DRIVERS` 指定的路径
- 虽然在 `/usr/local/nvidia/lib64/libibverbs/` 创建了符号链接，但：
  - 子进程（multiprocessing spawn）可能不继承 `RDMAV_DRIVERS` 环境变量
  - 符号链接指向父目录，可能解析失败

**诊断命令**：
```bash
# 检查驱动搜索路径
env | grep RDMAV

# 检查驱动文件
ls -la /usr/lib/x86_64-linux-gnu/libibverbs/
ls -la /usr/local/nvidia/lib64/libibverbs/

# 测试驱动加载
ldd /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so
```

---

### 3. NCCL 配置不兼容

**现象**：
```
NCCL WARN NCCL/NET (shim) mismatch recommended: 
TORCH_NCCL_USE_COMM_NONBLOCKING=0 (expected unset)
```

**根本原因**：
- GIB (Google InfiniBand) shim 要求 `TORCH_NCCL_USE_COMM_NONBLOCKING` 未设置
- 容器镜像默认设置了此变量为 0

---

### 4. 环境变量路径错误

**现象**：
- `GDRCOPY_HOME=/opt/deepep/gdrcopy` 路径不存在
- `NVSHMEM_HOME=/opt/deepep/nvshmem` 路径不存在

**根本原因**：
- 安装脚本将库复制到 `/home/kubernetes/bin/nvidia/lib64/`
- 容器中挂载为 `/usr/local/nvidia/lib64/`
- 但环境变量指向了不存在的 `/opt/deepep/` 路径

---

## 解决方案

### 完整修复步骤

#### 1. 修复 RDMA 驱动路径

**在容器启动脚本中添加**：
```bash
# 复制驱动到系统默认路径（关键修复）
cp /usr/local/nvidia/lib64/libmlx5.so.1.25.56.0 \
   /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so
chmod +x /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so

# 也复制到 nvidia 路径（可选，用于一致性）
mkdir -p /usr/local/nvidia/lib64/libibverbs
cp /usr/local/nvidia/lib64/libmlx5.so.1.25.56.0 \
   /usr/local/nvidia/lib64/libibverbs/libmlx5-rdmav57.so
chmod +x /usr/local/nvidia/lib64/libibverbs/libmlx5-rdmav57.so
```

**为什么这样做**：
- 系统默认路径 `/usr/lib/x86_64-linux-gnu/libibverbs/` 会被所有进程搜索
- 避免依赖 `RDMAV_DRIVERS` 环境变量
- 确保子进程也能找到驱动

#### 2. 配置环境变量

**在 Kubernetes Job YAML 中**：
```yaml
env:
  # 库路径配置
  - name: LD_LIBRARY_PATH
    value: /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/gib/lib64
  
  - name: PYTHONPATH
    value: /usr/local/nvidia/deepep
  
  # RDMA 配置
  - name: RDMAV_DRIVERS
    value: /usr/local/nvidia/lib64/libibverbs
  
  # NCCL 配置
  - name: NCCL_NET
    value: gIB
  - name: NCCL_IB_TC
    value: "52"
  - name: NCCL_IB_ADAPTIVE_ROUTING
    value: "1"
  - name: NCCL_IB_QPS_PER_CONNECTION
    value: "4"
  - name: NCCL_CROSS_NIC
    value: "0"
  - name: NCCL_NET_GDR_LEVEL
    value: PIX
  - name: NCCL_P2P_NET_CHUNKSIZE
    value: "131072"
  - name: NCCL_NVLS_CHUNKSIZE
    value: "524288"
  - name: NCCL_IB_FIFO_TC
    value: "84"
  - name: NCCL_TUNER_CONFIG_PATH
    value: /usr/local/gib/configs/tuner_config_a4.txtpb
  
  # DeepEP/NVSHMEM 配置（修正路径）
  - name: GDRCOPY_HOME
    value: /usr/local/nvidia
  - name: NVSHMEM_HOME
    value: /usr/local/nvidia
  - name: NVSHMEM_IBGDA_SUPPORT
    value: "1"
  - name: NVSHMEM_USE_GDRCOPY
    value: "1"
  - name: USE_NVPEERMEM
    value: "1"
  
  # CUDA 配置
  - name: CUDA_HOME
    value: /usr/local/cuda
  - name: TORCH_CUDA_ARCH_LIST_B200
    value: "10.0"
```

#### 3. Unset 不兼容的环境变量

**在启动脚本中**：
```bash
# Unset TORCH_NCCL_USE_COMM_NONBLOCKING for GIB compatibility
unset TORCH_NCCL_USE_COMM_NONBLOCKING
```

---

## 验证步骤

### 1. 验证 RDMA 设备

```bash
# 使用正确的库路径
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ibv_devinfo
```

**期望输出**：
```
hca_id: mlx5_0
    port: 1
        state: PORT_ACTIVE (4)
        link_layer: Ethernet
...
（应显示 8 个 mlx5 设备）
```

### 2. 验证 NCCL 初始化

```bash
python3 -c "
import torch
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group(backend='nccl', rank=0, world_size=1)
print('✓ NCCL initialized successfully!')
dist.destroy_process_group()
"
```

### 3. 运行 DeepEP 测试

```bash
cd /tmp
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP

# Unset 不兼容的环境变量
unset TORCH_NCCL_USE_COMM_NONBLOCKING

# 运行测试
python3 tests/test_intranode.py
```

---

## 常见错误和解决方法

### 错误 1: "Failed to initialize any NET plugin"

**原因**: RDMA 驱动未正确加载

**解决**:
```bash
# 确认驱动文件存在且可执行
ls -la /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so

# 如果不存在，复制并设置权限
cp /usr/local/nvidia/lib64/libmlx5.so.1.25.56.0 \
   /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so
chmod +x /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so
```

### 错误 2: "version 'IBVERBS_PRIVATE_34' not found"

**原因**: 使用了错误版本的 libibverbs

**解决**:
```bash
# 确保系统库优先
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 错误 3: "NCCL/NET (shim) mismatch recommended"

**原因**: TORCH_NCCL_USE_COMM_NONBLOCKING 设置不兼容

**解决**:
```bash
unset TORCH_NCCL_USE_COMM_NONBLOCKING
```

---

## 关键配置文件

- Job 配置: [`gpu-runtime-wxg-job.yaml`](./gpu-runtime-wxg-job.yaml)
- 原始 Pod 配置: [`gpu-runtime-wxg.yaml`](./gpu-runtime-wxg.yaml)
- DeepEP distroless 示例: [`deepep-internode-distroless.yaml`](./deepep-internode-distroless.yaml)

---

## 参考资源

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [libibverbs Manual](https://linux.die.net/man/3/ibv_open_device)
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [GKE GPUDirect-TCPXO Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx)