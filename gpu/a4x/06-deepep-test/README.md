# 7. DeepEP 测试

**版本选择指南**：**推荐使用 v2**，因 v1 在 GCP 上无法支持跨节点 EP。

## 7.1 概述

DeepEP 是 DeepSeek 开源的 Expert Parallelism (EP) 通信库。本节提供两条并行部署路径供选择：

| 路径 | DeepEP 版本 | CUDA | Internode 后端 | NVSHMEM | NCCL |
|------|-------------|------|----------------|---------|------|
| **路径 A**（稳定/传统） | v1 | CUDA 12 | NVSHMEM IBGDA | **3.4.5**（必需） | 2.25+ |
| **路径 B**（推荐/新版） | v2 (main) | CUDA 13 | NCCL Gin | 可选/仅编译用 | **>= 2.30.4** |

**Internode 限制**：DeepEP v1 的 internode 通信依赖 NVSHMEM IBGDA，而 IBGDA 需要 GDRCopy。GKE/GCP 环境中 GDRCopy 无法使用（内核模块不可加载），因此 **v1 internode 在 GKE 上受限**。

**解决方案**：DeepEP v2 (Elastic EP) 使用 NCCL Gin 后端进行 internode 通信，完全消除 NVSHMEM/GDRCopy 依赖，已在 2 节点 x 4 GPU 环境验证通过（144/144 PASS）。**推荐使用路径 B**。

**CUDA 13 容器注意**：NGC 26.04 容器自带 NVSHMEM 3.6.5，与 DeepEP v1 不兼容。这是选择路径 B (v2) 的另一重要原因。

## 7.2 NVSHMEM 版本兼容矩阵

| NVSHMEM 版本 | Struct 布局 | ibgda RC QP 索引 | DMABuf 支持 | DeepEP 兼容 | GB200 状态 |
|-------------|------------|------------------|-------------|-------------|------------|
| 3.3.9 | v1 | PE-major | 无 ibv_reg_dmabuf_mr | 兼容 | 失败（无 nvidia_peermem，无 DMABuf FD） |
| **3.4.5** | **v1** | **PE-major** | **ibv_reg_dmabuf_mr** | **兼容** | **可用（GB200 推荐）** |
| 3.5.x+ | v2 | QP-ID-major | ibv_reg_dmabuf_mr | 不兼容 | 失败（struct/索引与 DeepEP 不匹配） |

| GPU 平台 | 推荐 NVSHMEM 版本 | 原因 |
|----------|------------------|------|
| **GB200 (A4X)** | **3.4.5** | GB200 不支持 POSIX FD handle，3.4.5 通过 `ibv_reg_dmabuf_mr` 绕过 |
| B200 / H200 | 3.3.9 | 支持 `nvidia_peermem`，无需 DMABuf |

## 7.3 路径 A：DeepEP v1 + CUDA 12（稳定版）

适用容器：`nvcr.io/nvidia/pytorch:25.04-py3`（CUDA 12，NVSHMEM 兼容）

```bash
# 部署 DeepEP 测试 Pod（ComputeDomain + DRANET）
kubectl apply -f yamls/k8s1341-deepep-test-dranet.yaml

# 等待 Pod 就绪
kubectl get pods -l name -w
kubectl logs deepep-h1 -f  # 确认 "Ready"

# 验证 IMEX channel
kubectl exec deepep-h1 -- ls /dev/nvidia-caps-imex-channels/
```

### Step 1：安装 NVSHMEM 3.4.5 并编译 DeepEP v1

```bash
kubectl exec deepep-h1 -- bash -c "
  apt-get update -qq && apt-get install -y -qq git 2>/dev/null
  pip install nvidia-nvshmem-cu12==3.4.5 2>&1 | tail -3  # 必须 3.4.5
  ln -sf /usr/local/gib/lib64/libnccl.so.2 /usr/local/gib/lib64/libnccl.so

  cd /tmp && git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git
  cd /tmp/DeepEP

  # 4-GPU 补丁（见 7.5 节详解）
  sed -i 's/#define NUM_MAX_NVL_PEERS 8/#define NUM_MAX_NVL_PEERS 4/' csrc/kernels/configs.cuh
  sed -i 's/NUM_MAX_NVL_PEERS == 8/NUM_MAX_NVL_PEERS == 4/g' csrc/deep_ep.hpp csrc/kernels/internode.cu
  sed -i 's/must be 8/must be 4/' csrc/deep_ep.hpp
  sed -i 's/num_ranks > NUM_MAX_NVL_PEERS/num_ranks >= NUM_MAX_NVL_PEERS/g' csrc/deep_ep.cpp csrc/kernels/layout.cu

  NCCL_DIR=/usr/local/gib TORCH_CUDA_ARCH_LIST='10.0' python setup.py build_ext --inplace 2>&1 | tail -5
  # 确认输出: deep_ep/_C.cpython-312-aarch64-linux-gnu.so
"
```

### Step 2：运行 Intranode 测试

```bash
kubectl exec deepep-h1 -- bash -c "
  export NCCL_DIR=/usr/local/gib
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  export PYTHONPATH=/tmp/DeepEP:\$PYTHONPATH
  cd /tmp/DeepEP/tests/legacy
  python test_intranode.py --num-processes 4 --num-experts 64 --num-topk 4
"
```

### Step 3：运行 Low-latency 测试

```bash
kubectl exec deepep-h1 -- bash -c "
  export NCCL_DIR=/usr/local/gib PYTHONPATH=/tmp/DeepEP:\$PYTHONPATH
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  cd /tmp/DeepEP/tests/legacy
  python test_low_latency.py --num-processes 4 --num-experts 64 --num-topk 4
"
```

| 测试 | 实测结果 |
|------|----------|
| Intranode BF16 dispatch | **24/24 PASS**，best 435.97 GB/s NVL (SMs=24, chunk=32) |
| Intranode FP8 dispatch | **24/24 PASS**，best 288.07 GB/s NVL (SMs=24, chunk=32) |
| Intranode Combine | **PASS**，best 321.35 GB/s NVL (SMs=24, chunk=16) |
| Low-latency dispatch | **168-188 GB/s**，延迟 ~20 us |
| Low-latency combine | **316-331 GB/s**，延迟 ~22 us |

## 7.4 路径 B：DeepEP v2 + CUDA 13（推荐）

适用容器：`nvcr.io/nvidia/pytorch:26.04-py3`（CUDA 13.2，NCCL 2.29.7）。DeepEP v2 (Elastic EP) 使用 NCCL Gin 后端，消除了 NVSHMEM internode 依赖。

**镜像说明**：使用 `nvcr.io/nvidia/pytorch:26.04-py3`，含 CUDA 13.2.1、PyTorch 2.12.0a0。init container 为 GIB NCCL plugin `v1.1.0` (ARM64)。

### Step 1：安装 NVSHMEM 3.4.5 + NCCL 2.30.4

容器自带 NVSHMEM 3.6.5（与 DeepEP 不兼容）和 NCCL 2.29.7（v2 需要 >= 2.30.4）。需额外安装：

```bash
# 在每个 CUDA 13 Pod 上执行：

# 安装 NVSHMEM 3.4.5（用于 legacy 代码编译，不影响 v2 运行时）
apt-get update && apt install -y cuda-keyring && apt update
apt install -y libnvshmem3-cuda-12=3.4.5-1 libnvshmem3-dev-cuda-12=3.4.5-1 libnvshmem3-static-cuda-12=3.4.5-1

# 修复 CUDA 13 CCCL 头文件路径（cuda/std/ 移至 cccl/cuda/std/）
ln -sf /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda

# 创建 NVSHMEM 3.4.5 目录结构
mkdir -p /scratch-data/nvshmem345
ln -sf /usr/include/nvshmem_12 /scratch-data/nvshmem345/include
ln -sf /usr/lib/aarch64-linux-gnu/nvshmem/12 /scratch-data/nvshmem345/lib

# 安装 NCCL 2.30.4（DeepEP v2 GIN 后端要求 >= 2.30.4）
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

### Step 2：克隆 DeepEP v2 并应用 4-GPU 补丁

```bash
cd /scratch-data
git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git DeepEP-v2
cd DeepEP-v2

# DeepEP v2 使用 LEGACY_ 前缀的 NVL peer 常量，需要相应补丁
sed -i "s/#define LEGACY_NUM_MAX_NVL_PEERS 8/#define LEGACY_NUM_MAX_NVL_PEERS 4/" csrc/kernels/legacy/compiled.cuh

find csrc/ -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.h" | \
  xargs sed -i "s/LEGACY_NUM_MAX_NVL_PEERS == 8/LEGACY_NUM_MAX_NVL_PEERS == 4/g"

sed -i "s/LEGACY_NUM_MAX_NVL_PEERS \* sizeof(bool) == sizeof(uint64_t)/LEGACY_NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint32_t)/" \
  csrc/kernels/legacy/internode.cu

sed -i "s/uint64_t cached_bitmask/uint32_t cached_bitmask/" csrc/kernels/legacy/internode.cu

sed -i "s/num_ranks > LEGACY_NUM_MAX_NVL_PEERS/num_ranks >= LEGACY_NUM_MAX_NVL_PEERS/" csrc/kernels/legacy/layout.cu
```

### Step 3：编译 DeepEP v2

```bash
export NVSHMEM_DIR=/scratch-data/nvshmem345
export EP_NCCL_ROOT_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nccl
export TORCH_CUDA_ARCH_LIST="10.0"
export EP_JIT_CACHE_DIR=/scratch-data/deepep-jit-cache
mkdir -p /scratch-data/deepep-jit-cache

pip install -e . --no-build-isolation 2>&1 | tail -10

# 验证
python3 -c "import deep_ep; print('DeepEP v2 import OK')"
```

**注意**：编译时勿将 GIB 加入 `LD_LIBRARY_PATH` — GIB 自带 NCCL 2.28.9，与容器 NCCL 2.29.7 冲突，会导致编译链接错误。

### Step 4：运行 Intranode 测试

```bash
cd /scratch-data/DeepEP-v2
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
python tests/test_intranode.py --num-processes 4
```

### Step 5：运行 Internode 测试（2 节点 x 4 GPU，NCCL Gin）

**注意**：不要使用 `tests/legacy/test_internode.py`，该测试会因 `LEGACY_NUM_MAX_NVL_PEERS=8` assertion 在 4-GPU A4X 上失败。请使用 v2 elastic 测试。

```bash
# 在两个 Pod 上创建启动脚本：
cat > /scratch-data/run_deepep_v2_test.sh << 'SCRIPT'
#!/bin/bash
export LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include
export NVSHMEM_DIR=/scratch-data/nvshmem345
export EP_NCCL_ROOT_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nccl
export EP_JIT_CACHE_DIR=/scratch-data/deepep-jit-cache
export NCCL_NET=gIB
export NCCL_PXN_C2C=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=52
export NCCL_IB_FIFO_TC=84
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT:-8363}
export WORLD_SIZE=${WORLD_SIZE:-2}
export RANK=${RANK:-0}

cd /scratch-data/DeepEP-v2
python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 6 --num-experts 256 "$@"
SCRIPT
chmod +x /scratch-data/run_deepep_v2_test.sh

# 获取 Pod 1 的 IP
HOST1_IP=$(kubectl get pod cuda13-dra-1 -o jsonpath='{.status.podIP}')

# 在 Pod 2 上启动（RANK=1）
kubectl exec cuda13-dra-2 -- bash -c "
  MASTER_ADDR=$HOST1_IP RANK=1 /scratch-data/run_deepep_v2_test.sh
" &

sleep 5

# 在 Pod 1 上启动（RANK=0）
kubectl exec cuda13-dra-1 -- bash -c "
  MASTER_ADDR=$HOST1_IP RANK=0 /scratch-data/run_deepep_v2_test.sh
"
```

**关键环境变量说明**：

- `LD_PRELOAD`：必须指定 NCCL 2.30.4，因为容器默认加载 NCCL 2.29.7，不含 Gin API
- `TRITON_PTXAS_PATH`：防止 Triton JIT 在子进程中编译失败
- `NCCL_NET=gIB`：启用 GPUDirect RDMA（GIB）进行跨节点通信

| 测试 | 实测结果 |
|------|----------|
| Intranode dispatch BF16 | **24/24 PASS**，best 460.87 GB/s NVL (SMs=24, chunk=32) |
| Intranode dispatch FP8 | **24/24 PASS**，best 308.11 GB/s NVL (SMs=24, chunk=32) |
| Intranode Combine | **PASS**，best 346.24 GB/s NVL (SMs=24, chunk=16) |
| Internode MNNVL Dispatch (SU) | **~580 GB/s**，延迟 ~58 us |
| Internode MNNVL Combine (SU) | **~660 GB/s**，延迟 ~99 us |
| 全部配置（Internode Elastic EP） | **144/144 PASS** |

**Internode v2 验证通过**：NCCL Gin 将所有 8 GPU（2 x 4）统一管理，NVLink 负责节点内通信，RDMA (GIB) 负责跨节点通信。

## 7.5 4-GPU 适配说明（A4X 补丁）

DeepEP 默认假设每节点 8 GPU（NVLink peers = 8），但 A4X 每节点仅 4 GPU。若不修改会导致 assertion 失败或 bitmask 越界。

| 版本 | 常量名 | 修改文件 | 改动 |
|------|--------|----------|------|
| v1 | `NUM_MAX_NVL_PEERS` | `csrc/kernels/configs.cuh` 等 | 8 → 4 |
| v2 | `LEGACY_NUM_MAX_NVL_PEERS` | `csrc/kernels/legacy/compiled.cuh` 等 | 8 → 4 |

**核心修改点：**

1. **常量定义**：`#define NUM_MAX_NVL_PEERS 8` 改为 `4`
2. **断言**：`static_assert(... == 8)` 改为 `== 4`
3. **边界检查**：`num_ranks > NUM_MAX_NVL_PEERS` 改为 `>= NUM_MAX_NVL_PEERS`（4 GPU 时 `> 4` 会跳过本应执行的分支）
4. **Bitmask 类型**：8 peers x `sizeof(bool)` = 8 bytes = `uint64_t`；4 peers x `sizeof(bool)` = 4 bytes = **`uint32_t`**

## 7.6 功能测试汇总

### 路径 A (v1 + CUDA 12) 测试命令

```bash
# Intranode（单节点 4 GPU）
cd /tmp/DeepEP/tests/legacy
python test_intranode.py --num-processes 4 --num-experts 64 --num-topk 4

# Low-latency（单节点 4 GPU，需 NVSHMEM 3.4.5）
python test_low_latency.py --num-processes 4 --num-experts 64 --num-topk 4

# Internode（需 NVSHMEM IBGDA，GKE 上受 GDRCopy 限制）
# 需设置：NVSHMEM_REMOTE_TRANSPORT=ibgda  NVSHMEM_DISABLE_CUDA_VMM=1
# 需设置：DEEP_EP_DEVICE_TO_HCA_MAPPING="0:mlx5_0,1:mlx5_1,2:mlx5_2,3:mlx5_3"
```

### 路径 B (v2 + CUDA 13) 测试命令

```bash
# Intranode（单节点 4 GPU）
cd /scratch-data/DeepEP-v2
python tests/test_intranode.py --num-processes 4

# Internode Elastic EP（2 节点 x 4 GPU，NCCL Gin 后端）
# 使用 /scratch-data/run_deepep_v2_test.sh 脚本（见 7.4 Step 5）
```

### CUDA 12 vs CUDA 13 性能对比

| 指标 | CUDA 12 (路径 A) | CUDA 13 (路径 B) |
|------|------------------|------------------|
| Intranode dispatch BF16 | ~436 GB/s | ~461 GB/s |
| Intranode dispatch FP8 | ~288 GB/s | ~308 GB/s |
| Intranode combine | ~321 GB/s | ~346 GB/s |
| Internode (v1 legacy) | 可用 | 阻塞（NVSHMEM 版本冲突） |
| Internode (v2 elastic) | N/A | **可用（NCCL Gin，144/144 PASS）** |

## DeepEP v1 vs v2 选择建议

| 维度 | DeepEP v1 (CUDA 12) | DeepEP v2 (CUDA 13) |
|------|---------------------|---------------------|
| Internode 后端 | NVSHMEM IBGDA — 需 GDRCopy（GCP 不可用） | NCCL Gin — 完全兼容 GCP |
| Intranode | 正常工作 | 正常工作 |
| NVSHMEM 依赖 | 严格要求 3.4.5（GB200）或 3.3.9（B200/H200） | 仅编译时需要，运行时不依赖 |
| CUDA 容器 | pytorch:25.04-py3 (CUDA 12) | pytorch:26.04-py3 / 26.05-py3 (CUDA 13) |
| GCP 推荐 | 仅 intranode 场景 | **推荐（internode 已验证 144/144 PASS）** |

**核心决策**：如需跨节点 Expert Parallelism（internode EP），必须使用 **DeepEP v2**。v1 的 internode 通信依赖 NVSHMEM IBGDA + GDRCopy，而 GDRCopy 在 GKE/GCP 环境中无法使用。v2 使用 NCCL Gin 后端完全消除此依赖。
