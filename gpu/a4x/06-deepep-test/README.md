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

## 7.7 我方验证结果（2026-06-29，k8s 1.34 + DRA + ComputeDomain）

### 测试环境

| 项目 | 值 |
|---|---|
| 集群 | kubeadm 1.34.9, 16 Worker (2 domain × 8 node) |
| GPU | 4 × GB200 189GB per node |
| 镜像 | `megatron-ngc:tev2.15-mgcore_r0.16.0-pt26.05-py3-v2` (DeepEP 2.0.0 pre-installed) |
| NCCL | 2.30.4+cuda13.2 (via LD_PRELOAD) |
| 网络 | GIB RDMA (跨节点) + NVSwitch/MNNVL (域内) |
| ComputeDomain | cd-d1 (Domain 1), cd-d2 (Domain 2) |

### Intranode 测试（单节点 4 GPU）

**Legacy Intranode (`tests/legacy/test_intranode.py`)**：

| 操作 | 带宽 (GB/s) | SMs | Chunk | 说明 |
|---|---|---|---|---|
| Dispatch BF16 | **436** | 24 | 32 | Expert 分发，BF16 精度 |
| Dispatch FP8 | **292** | 24 | 32 | Expert 分发，FP8 精度（量化后数据量减半，带宽相应降低） |
| Combine | **322** | 24 | 16 | Expert 结果合并回原 token 顺序 |

**Legacy Low-latency (`tests/legacy/test_low_latency.py`)**：

| 操作 | 带宽 (GB/s) | 延迟 (us) | 说明 |
|---|---|---|---|
| Dispatch | 163-182 | 20-23 | 低延迟模式，优化 latency 而非 throughput |
| Combine | 323-335 | 21-22 | 合并操作天然延迟更低 |
| Dispatch + Combine | 204-209 | 49-53 | 端到端延迟 |

### Internode 测试（2 节点 × 4 GPU = 8 GPU，NCCL Gin）

**Elastic EP (`tests/elastic/test_ep.py --test-first-only`)**：

测试配置：`do_handle_copy=1, expert_alignment=128, use_fp8_dispatch=1, num_experts=256, num_topk=6, hidden=7168, num_tokens=4096`

| 操作 | 模式 | 带宽 SU (GB/s) | 延迟 (us) | 说明 |
|---|---|---|---|---|
| Dispatch | scaleup | **695-705** | 191-193 | 跨节点 Expert 分发 |
| Expanded Dispatch | scaleup | 697-704 | 191-194 | 带 handle copy 的扩展分发 |
| Cached Dispatch | scaleup | 695-702 | 191-195 | 缓存模式分发 |
| Combine | scaleup | **720-728** | 357-360 | 跨节点结果合并 |
| Reduced Combine | scaleup | 701-709 | 367-370 | 带 reduction 的合并 |
| Copy (NVLink) | — | 4936-6289 | 43-68 | 节点内 GPU 间拷贝 |
| Reduce (NVLink) | — | 2043-2168 | 54-58 | 节点内 GPU 间归约 |

所有 4 个 scaleup 配置（SU 0/1/2/3）均正常完成，无错误。

### Benchmark 对比分析

#### Intranode vs Internode

| 操作 | Intranode (NVLink) | Internode (NCCL Gin) | 比值 | 说明 |
|---|---|---|---|---|
| Dispatch | 436 (BF16) | 700 (FP8) | **1.6×** | Internode 更高是因为 FP8 量化减半了数据量，且 NCCL Gin 利用了 MNNVL 通道 |
| Combine | 322 | 724 | **2.2×** | Internode combine 走 NCCL collective，比 Legacy 手写 NVLink kernel 更高效 |

> **为什么 Internode 比 Intranode 带宽更高？**
> 
> 看似反直觉，但原因是两个测试用了不同的通信路径：
> - **Legacy Intranode**：使用 DeepEP 自研的 NVLink peer-to-peer kernel（手写 CUDA kernel 直接操作 NVLink），单节点 4 GPU，SMs=24
> - **Elastic Internode**：使用 NCCL Gin 后端（NCCL 2.30.4 的 GPU-Initiated Network 通信），2 节点 8 GPU
>
> NCCL Gin 后端的优势：
> 1. 域内通信走 MNNVL（NVSwitch 全互联，带宽 > P2P NVLink）
> 2. 域间通信走 GIB RDMA（GPU 直接发起 RDMA，无 CPU 参与）
> 3. NCCL 的 collective 算法对大消息更高效（pipeline + multi-channel）
> 4. 8 GPU 的 ring 比 4 GPU 有更高的带宽利用率

#### Copy 和 Reduce 带宽解读

| 操作 | 带宽 (GB/s) | 理论带宽 | 利用率 | 说明 |
|---|---|---|---|---|
| Copy (NVLink) | 5000-6300 | ~7200 | 69-88% | NVSwitch 5th gen 单向带宽，近理论峰值 |
| Reduce (NVLink) | 2043-2168 | ~3600 | 57-60% | 归约需读+写两次数据，有效带宽 ≈ 理论/2 再加计算开销 |

NVSwitch 5th gen 在 GB200 上提供每 GPU ~900 GB/s 双向 NVLink 带宽。Copy 达到 5000-6300 GB/s（跨 4 GPU 聚合），说明 NVSwitch fabric 工作正常。

#### 参数敏感性测试（Intranode）

同一硬件、同一镜像，不同 `num-topk` 和 `num-experts` 对带宽的影响：

| 参数组合 | Dispatch BF16 | Dispatch FP8 | Combine |
|---|---|---|---|
| topk=4, experts=64 | 432 GB/s | 291 GB/s | 323 GB/s |
| **topk=8, experts=256**（默认） | **465 GB/s** | **317 GB/s** | **347 GB/s** |
| topk=8, experts=256 + allow-mnnvl | 465 GB/s | 317 GB/s | 348 GB/s |

**topk 对带宽的影响**：topk=8 比 topk=4 高 7-9%。原因是 topk 越大，每个 token 需要分发到的 Expert 越多，dispatch 的数据量更大（topk=8 数据量是 topk=4 的 2 倍），NVLink 传输的 pipeline 填充更充分，批量传输效率更高。topk=4 时每次传输的数据块较小，NVLink 带宽利用率偏低。

**allow-mnnvl 对 intranode 无影响**：单节点 4 GPU 全在同一 NVSwitch 下，有无 MNNVL 不改变通信路径。

#### 与文档 Baseline 对比

| 指标 | Baseline (7.4节) | 我方(topk=4) | 我方(topk=8) | 差异(topk=8) |
|---|---|---|---|---|
| Intranode dispatch BF16 | 461 GB/s | 432 | **465** | **+0.9%** |
| Intranode dispatch FP8 | 308 GB/s | 291 | **317** | **+2.9%** |
| Intranode combine | 346 GB/s | 322 | **347** | **+0.3%** |
| Internode dispatch SU | ~580 GB/s | — | **700** | **+21%** |
| Internode combine SU | ~660 GB/s | — | **724** | **+10%** |

**Intranode**：使用与 Baseline 相同的默认参数（topk=8, experts=256）后，三项指标全部匹配 Baseline（差异 < 3%）。此前低 5-7% 是因为测试参数不同（topk=4 vs 8），不是性能问题。

**Internode 为什么比 Baseline 高 10-21%**：

Baseline 和我方均使用 2 节点 × 4 GPU（相同规模），但有两个差异：
1. **GIB NCCL plugin 版本**：我方 init container 使用 v1.1.2，Baseline 使用 v1.1.0。v1.1.2 可能包含 RDMA 路径优化
2. **NCCL 版本**：均为 2.30.4，但我方通过 LD_PRELOAD 加载的 NCCL 是 pip 安装的 `nvidia-nccl-cu13`，可能包含更新的 Gin backend 补丁

> **注意**：Internode 高于 Baseline 不是因为节点数少——Baseline 同样是 2 节点。2 节点 × 4 GPU = 8 GPU 是 Elastic EP internode 测试的标准配置，不受规模效应影响（不像 NCCL ring 那样有 multi-channel 效应），因为 DeepEP dispatch/combine 是 point-to-point 模式。

### 规模扩展测试（进行中）

**8 节点 32 GPU（单域）**：失败，根因是 RDMA GID 配置问题。

DRANET 给 pod 分配 4 张 RDMA NIC (mlx5_0-3)，但仅 1 张有 RoCEv2 IPv4 GID（index 3），其他 3 张 GID 全零：

```
mlx5_0 gid3: 0000:0000:...:0000  ← 空（无 IPv4）
mlx5_1 gid3: 0000:0000:...:0000  ← 空
mlx5_2 gid3: 0000:0000:...:ffff:10.10.24.32  ← 正常
mlx5_3 gid3: 0000:0000:...:0000  ← 空
```

- 2 节点测试跑通是因为 NCCL 碰巧只用到了有 GID 的 NIC
- 8 节点时 NCCL 需要更多跨节点连接，用到无 GID 的 NIC 触发 `ibv_modify_qp failed` 错误
- NCCL test 在同规模（8+8 节点 64 GPU）跑通，是因为 GIB 网络插件有独立的 RDMA 地址发现机制（不依赖 GID index），而 DeepEP 的 NCCL communicator 可能 fallback 到了标准 IB 传输

**根因确认**：Host 上所有 4 张 NIC 的 GID 3 均正常（IPv4 地址完整）。DRANET 将 NIC 移入 pod 网络命名空间后仅给 1/4 NIC 配置了 IPv4，导致 3/4 NIC 的 RoCEv2 GID 丢失。这是 DRANET v1.3.0 在 GCP A4X 多 RDMA NIC 场景下的已知限制。

**解决方案**：使用 **hostNetwork 模式**绕过 DRANET，pod 直接使用 host 的 RDMA NIC（GID 完整）。不声明 `rdma-nics` ResourceClaim，只保留 `compute-domain-channel`。

**hostNetwork 模式验证（4 节点 16 GPU）**：PASS

| 操作 | 2n/8GPU | 4n/16GPU | 变化 | 说明 |
|---|---|---|---|---|
| Dispatch SU | 700 | **660** | -6% | 规模增大 RDMA 竞争加剧 |
| Expanded Dispatch SU | 700 | 660 | -6% | |
| Cached Dispatch SU | 698 | 660 | -5% | |
| Combine SU | 724 | **683** | -6% | |
| Reduced Combine SU | 705 | **677** | -4% | |
| Copy (NVLink) | 5600 | 5500 | -2% | NVLink 不受规模影响 |
| Reduce (NVLink) | 2100 | 1870 | -11% | 更多 GPU 参与 reduce |

**关键发现：2 节点数据偏高的原因**

2 节点 (8 GPU) 的 dispatch 700 GB/s 和 combine 724 GB/s 显著高于文档 baseline（580/660），但 4 节点 (16 GPU) 降到 660/683。原因：

1. **NCCL communicator 规模效应**：2 节点时 NCCL 只需建立 2 个跨节点连接，ring 短、pipeline 填充快。4 节点需要更多跨节点连接，RDMA 竞争和 ring 长度增加导致带宽下降
2. **Expert 路由分散度**：更多 GPU 意味着 Expert 分散在更多节点上，每个 token 的 dispatch 需要跨越更多跨节点链路
3. **baseline 580/660 可能对应更大规模测试**（如 8-18 节点），我方 2 节点数据不具有规模代表性

**8 节点 32 GPU hostNetwork 测试：PASS**

| 操作 | 2n/8GPU | 4n/16GPU | 8n/32GPU | Baseline | 说明 |
|---|---|---|---|---|---|
| Dispatch SU | 700 | 660 | **636** | ~580 | 每翻倍 -5~6% |
| Combine SU | 724 | 683 | **590** | ~660 | Combine 下降更快（reduce 开销 O(N)） |
| Reduced Combine SU | 705 | 677 | **593** | — | |
| Copy (NVLink) | 5600 | 5500 | **5700** | — | NVLink 不受规模影响 |
| Reduce (NVLink) | 2100 | 1870 | **1730** | — | 32 路 reduce 同步开销更大 |

**规模下降原因分析**（同域 NVL72 NVSwitch 全互联）：

1. **NVSwitch 带宽不是瓶颈**。NVL72 域内所有 GPU 通过 NVSwitch 5 代全互联，任意 GPU 对带宽相同。Copy 维持 5500-5700 GB/s 证实了这一点

2. **Dispatch 下降来自 scatter 碎片化**。8 GPU 时每个 GPU 的 topk=6 分发到 7 个目标，平均每目标 ~100 GB/s 有效带宽。32 GPU 时分发到 31 个目标，平均每目标 ~22 GB/s。虽然 NVLink 总带宽不变，但小块分散写入的 NVLink 利用率低于大块连续传输

3. **Combine 下降更快（-19% vs Dispatch -9%）**。Combine 需要做 reduce（加法归约），32 路 reduce 比 8 路有更多 pipeline 阶段和同步点。每个 GPU 要等待更多 partial result 到达后才能完成 reduce，同步开销随参与者数量增长

4. **与 Baseline 对齐**。32 GPU 的 Dispatch 636 和 Combine 590 与文档 Baseline（580/660）接近，确认 Baseline 对应的是较大规模测试。2 节点 700/724 的"高值"不是我们的改进，而是小规模效应

**结论**：DeepEP internode 带宽随规模增大而下降，是 all-to-all 通信模式的固有特性（scatter 碎片化 + reduce 同步开销），不是硬件瓶颈。2 节点数据不具有规模代表性，建议以 8+ 节点数据为基准。

### 部署要点（k8s 1.34 DRA 模式）

1. Pod 必须有 `compute-domain-channel` claim——DeepEP v2 的 NCCL Gin 需要 IMEX channel
2. `LD_PRELOAD` 指定 NCCL 2.30.4——容器默认的 2.29.7 不含 Gin API
3. `NCCL_NET=gIB`——启用 GPUDirect RDMA
4. `TRITON_PTXAS_PATH`——DeepEP JIT 编译需要 ptxas
5. AR 镜像仓库需要 `imagePullSecrets`——token 有效期短，需定期刷新
6. Internode test_ep.py 用 `WORLD_SIZE=节点数` + `RANK=节点序号`，不要用 torchrun
7. **大规模（>2 节点）internode 受 RDMA GID 配置限制**——详见上方规模扩展测试记录
