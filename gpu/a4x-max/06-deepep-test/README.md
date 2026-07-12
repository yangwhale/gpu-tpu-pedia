# 7. DeepEP 测试 (GB300 A4X MAX)

**版本选择指南**：**推荐使用 v2**，因 v1 在 GCP 上无法支持跨节点 EP。

## 7.1 概述

DeepEP 是 DeepSeek 开源的 Expert Parallelism (EP) 通信库。本节提供两条并行部署路径供选择：

| 路径 | DeepEP 版本 | CUDA | Internode 后端 | NVSHMEM | NCCL |
|------|-------------|------|----------------|---------|------|
| **路径 A**（稳定/传统） | v1 | CUDA 12 | NVSHMEM IBGDA | **3.4.5**（必需） | 2.25+ |
| **路径 B**（推荐/新版） | v2 (main) | CUDA 13 | NCCL Gin | 可选/仅编译用 | **>= 2.30.4** |

**Internode 限制**：DeepEP v1 的 internode 通信依赖 NVSHMEM IBGDA，而 IBGDA 需要 GDRCopy。GKE/GCP 环境中 GDRCopy 无法使用（内核模块不可加载），因此 **v1 internode 在 GKE 上受限**。

**解决方案**：DeepEP v2 (Elastic EP) 使用 NCCL Gin 后端进行 internode 通信，完全消除 NVSHMEM/GDRCopy 依赖。**推荐使用路径 B**。

**CUDA 13 容器注意**：NGC 26.04 容器自带 NVSHMEM 3.6.5，与 DeepEP v1 不兼容。这是选择路径 B (v2) 的另一重要原因。

### GB300 硬件优势

GB300 相比 GB200 在 DeepEP 场景下有三项关键升级：

| 维度 | GB200 (A4X) | GB300 (A4X MAX) | DeepEP 影响 |
|------|-------------|-----------------|-------------|
| GPU 显存 | 186 GB/GPU (HBM3e) | **278 GB/GPU (HBM3e)** | Expert 参数 + workspace buffer 容量 +50% |
| RDMA 网卡 | CX-7 VF, 挂在 CPU 上 | **CX-8 SuperNIC PF, 直连 GPU (GPUDirect)** | All-to-all dispatch 延迟更低 |
| RDMA 接口数 | 4, 2000 Gbps | **8, 3200 Gbps** | 跨节点带宽 +60%，更多 rail 并行 |

**CX-8 GPUDirect 对 DeepEP 的意义**：DeepEP 的 Expert dispatch/combine 本质是 GPU-initiated all-to-all 通信。GB200 上 RDMA 经过 CPU (CX-7 VF)，存在 PCIe hop 开销；GB300 上 CX-8 PF 直连 GPU，消除了 CPU 中转延迟。HybridEP dispatcher 和 Elastic EP 的 NCCL Gin 后端均可直接受益于 GPUDirect RDMA 路径，预期在 internode all-to-all 延迟上有显著改善。

## 7.2 NVSHMEM 版本兼容矩阵

| NVSHMEM 版本 | Struct 布局 | ibgda RC QP 索引 | DMABuf 支持 | DeepEP 兼容 | GB300 状态 |
|-------------|------------|------------------|-------------|-------------|------------|
| 3.3.9 | v1 | PE-major | 无 ibv_reg_dmabuf_mr | 兼容 | 失败（无 nvidia_peermem，无 DMABuf FD） |
| **3.4.5** | **v1** | **PE-major** | **ibv_reg_dmabuf_mr** | **兼容** | **可用（GB300 推荐）** |
| 3.5.x+ | v2 | QP-ID-major | ibv_reg_dmabuf_mr | 不兼容 | 失败（struct/索引与 DeepEP 不匹配） |

| GPU 平台 | 推荐 NVSHMEM 版本 | 原因 |
|----------|------------------|------|
| **GB300 (A4X MAX)** | **3.4.5** | GB300 不支持 POSIX FD handle，3.4.5 通过 `ibv_reg_dmabuf_mr` 绕过 |
| **GB200 (A4X)** | **3.4.5** | 同上 |
| B200 / H200 | 3.3.9 | 支持 `nvidia_peermem`，无需 DMABuf |

## 7.3 GB200 Baseline（对照组）

以下是 GB200 (A4X) 上 DeepEP v2 的实测结果，作为 GB300 测试的对照基准。

### GB200 测试环境

| 项目 | 值 |
|---|---|
| 机器类型 | a4x-highgpu-4g |
| GPU | 4 x GB200, 186 GB HBM3e/GPU |
| RDMA | CX-7 VF, 4 MRDMA 接口, 2000 Gbps |
| 镜像 | `megatron-ngc:tev2.15-mgcore_r0.16.0-pt26.05-py3-v2` (DeepEP 2.0.0 pre-installed) |
| NCCL | 2.30.4+cuda13.2 (via LD_PRELOAD) |
| 网络 | GIB RDMA (跨节点) + NVSwitch/MNNVL (域内) |

### GB200 Intranode 结果

| 操作 | 带宽 (GB/s) | SMs | Chunk | 说明 |
|---|---|---|---|---|
| Dispatch BF16 | **465** | 24 | 32 | topk=8, experts=256 |
| Dispatch FP8 | **317** | 24 | 32 | topk=8, experts=256 |
| Combine | **347** | 24 | 16 | topk=8, experts=256 |
| Low-latency Dispatch | 163-182 | — | — | 延迟 20-23 us |
| Low-latency Combine | 323-335 | — | — | 延迟 21-22 us |

### GB200 Internode 结果（Elastic EP, NCCL Gin）

| 操作 | 2n/8GPU | 4n/16GPU | 8n/32GPU | 说明 |
|---|---|---|---|---|
| Dispatch SU | 700 | 660 | **636** | 每翻倍 -5~6% |
| Combine SU | 724 | 683 | **590** | Combine 下降更快 |
| Reduced Combine SU | 705 | 677 | **593** | |
| Copy (NVLink) | 5600 | 5500 | **5700** | NVLink 不受规模影响 |
| Reduce (NVLink) | 2100 | 1870 | **1730** | 32 路 reduce 同步开销更大 |

### GB300 预期改善

基于 CX-8 GPUDirect + 8-way MRDMA 升级，预期在以下维度看到改善：

| 指标 | GB200 Baseline | GB300 预期 | 理由 |
|---|---|---|---|
| Internode Dispatch 延迟 | ~191 us | **降低** | CX-8 GPUDirect 消除 CPU hop，dispatch 延迟更低 |
| Internode Dispatch 带宽 | 636-700 GB/s | **提升** | 8-way MRDMA (3200 Gbps) vs 4-way (2000 Gbps) |
| Internode Combine 带宽 | 590-724 GB/s | **提升** | 更多 rail 并行，reduce 吞吐更高 |
| Intranode 带宽 | 465 GB/s (BF16) | **持平或小幅提升** | NVSwitch 拓扑相同 (1x72)，GPU SM 数可能微增 |
| 大规模下降幅度 | -5~6%/翻倍 | **更平缓** | CX-8 per-rail 带宽更高，scatter 碎片化影响更小 |

## 7.4 路径 A：DeepEP v1 + CUDA 12（稳定版）

适用容器：`nvcr.io/nvidia/pytorch:25.04-py3`（CUDA 12，NVSHMEM 兼容）

```bash
# 部署 DeepEP 测试 Pod（ComputeDomain + DRANET）
# 注意：GB300 的 ResourceClaimTemplate 需要 count=8（8 MRDMA 接口）
kubectl apply -f yamls/k8s1343-deepep-test-dranet.yaml

# 等待 Pod 就绪
kubectl get pods -l name -w
kubectl logs deepep-h1 -f  # 确认 "Ready"

# 验证 IMEX channel
kubectl exec deepep-h1 -- ls /dev/nvidia-caps-imex-channels/

# 验证 CX-8 RDMA 设备（应该看到 8 个 PF，而非 GB200 的 4 个 VF）
kubectl exec deepep-h1 -- rdma link show
```

### Step 1：安装 NVSHMEM 3.4.5 并编译 DeepEP v1

```bash
kubectl exec deepep-h1 -- bash -c "
  apt-get update -qq && apt-get install -y -qq git 2>/dev/null
  pip install nvidia-nvshmem-cu12==3.4.5 2>&1 | tail -3  # 必须 3.4.5
  ln -sf /usr/local/gib/lib64/libnccl.so.2 /usr/local/gib/lib64/libnccl.so

  cd /tmp && git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git
  cd /tmp/DeepEP

  # 4-GPU 补丁（见 7.6 节详解）
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

| 测试 | GB300 实测结果 |
|------|----------------|
| Intranode BF16 dispatch | 待测 |
| Intranode FP8 dispatch | 待测 |
| Intranode Combine | 待测 |
| Low-latency dispatch | 待测 |
| Low-latency combine | 待测 |

## 7.5 路径 B：DeepEP v2 + CUDA 13（推荐）

适用容器：`nvcr.io/nvidia/pytorch:26.04-py3`（CUDA 13.2，NCCL 2.29.7）。DeepEP v2 (Elastic EP) 使用 NCCL Gin 后端，消除了 NVSHMEM internode 依赖。

**镜像说明**：使用 `nvcr.io/nvidia/pytorch:26.04-py3`，含 CUDA 13.2.1、PyTorch 2.12.0a0。init container 为 GIB NCCL plugin `v1.1.0` (ARM64)。

**GB300 GKE 集群信息**：

| 项目 | 值 |
|---|---|
| 集群名 | chrisya-gb300-gke |
| 项目 | tencent-gcp-taiji-poc |
| Zone | us-central1-b |
| 机器类型 | a4x-maxgpu-4g-metal (bare metal) |
| GPU | 4 x GB300, 278 GB HBM3e/GPU |
| RDMA | CX-8 SuperNIC PF (GPUDirect), 8 MRDMA 接口, 3200 Gbps |
| GKE 最低版本 | 1.34.3-gke.1318000+ |
| DRA Driver | v25.8.0+ |

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

**注意**：不要使用 `tests/legacy/test_internode.py`，该测试会因 `LEGACY_NUM_MAX_NVL_PEERS=8` assertion 在 4-GPU A4X MAX 上失败。请使用 v2 elastic 测试。

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

# 获取 Pod 1 的 IP（注意 GB300 为 IPv6-only，需用 IPv6 地址）
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

**GB300 IPv6 注意事项**：GB300 网络栈为 IPv6-only，`MASTER_ADDR` 获取的将是 IPv6 地址（如 `fd20::xxxx`）。PyTorch 分布式和 NCCL 均支持 IPv6，但需确认：
1. `torch.distributed.init_process_group` 能正确解析 IPv6 地址
2. NCCL GIB plugin 的 RoCEv2 能走 IPv6（CX-8 应原生支持）

**GB300 CX-8 RDMA 设备映射**：GB300 有 8 个 MRDMA 接口（CX-8 PF），如需 v1 internode 的 `DEEP_EP_DEVICE_TO_HCA_MAPPING`：
```bash
# GB200 (4 NIC): "0:mlx5_0,1:mlx5_1,2:mlx5_2,3:mlx5_3"
# GB300 (8 NIC): 需实际 rdma link show 确认设备名，可能为：
# "0:mlx5_0,0:mlx5_1,1:mlx5_2,1:mlx5_3,2:mlx5_4,2:mlx5_5,3:mlx5_6,3:mlx5_7"
# 注意：4 GPU 对 8 NIC，每个 GPU 可能映射 2 个 NIC（双 rail）
```

## 7.6 4-GPU 适配说明（A4X MAX 补丁）

DeepEP 默认假设每节点 8 GPU（NVLink peers = 8），但 A4X MAX 每节点仅 4 GPU。若不修改会导致 assertion 失败或 bitmask 越界。

与 GB200 (A4X) 完全相同的补丁逻辑：

| 版本 | 常量名 | 修改文件 | 改动 |
|------|--------|----------|------|
| v1 | `NUM_MAX_NVL_PEERS` | `csrc/kernels/configs.cuh` 等 | 8 → 4 |
| v2 | `LEGACY_NUM_MAX_NVL_PEERS` | `csrc/kernels/legacy/compiled.cuh` 等 | 8 → 4 |

**核心修改点：**

1. **常量定义**：`#define NUM_MAX_NVL_PEERS 8` 改为 `4`
2. **断言**：`static_assert(... == 8)` 改为 `== 4`
3. **边界检查**：`num_ranks > NUM_MAX_NVL_PEERS` 改为 `>= NUM_MAX_NVL_PEERS`（4 GPU 时 `> 4` 会跳过本应执行的分支）
4. **Bitmask 类型**：8 peers x `sizeof(bool)` = 8 bytes = `uint64_t`；4 peers x `sizeof(bool)` = 4 bytes = **`uint32_t`**

## 7.7 功能测试汇总

### 路径 A (v1 + CUDA 12) 测试命令

```bash
# Intranode（单节点 4 GPU）
cd /tmp/DeepEP/tests/legacy
python test_intranode.py --num-processes 4 --num-experts 64 --num-topk 4

# Low-latency（单节点 4 GPU，需 NVSHMEM 3.4.5）
python test_low_latency.py --num-processes 4 --num-experts 64 --num-topk 4

# Internode（需 NVSHMEM IBGDA，GKE 上受 GDRCopy 限制）
# 需设置：NVSHMEM_REMOTE_TRANSPORT=ibgda  NVSHMEM_DISABLE_CUDA_VMM=1
# 需设置：DEEP_EP_DEVICE_TO_HCA_MAPPING（GB300 需映射 8 个 CX-8 PF）
```

### 路径 B (v2 + CUDA 13) 测试命令

```bash
# Intranode（单节点 4 GPU）
cd /scratch-data/DeepEP-v2
python tests/test_intranode.py --num-processes 4

# Internode Elastic EP（2 节点 x 4 GPU，NCCL Gin 后端）
# 使用 /scratch-data/run_deepep_v2_test.sh 脚本（见 7.5 Step 5）
```

### GKE 集群操作参考

```bash
# 设置 kubectl context
gcloud container clusters get-credentials chrisya-gb300-gke \
  --zone us-central1-b \
  --project tencent-gcp-taiji-poc

# 查看节点状态
kubectl get nodes -o wide

# 查看 GPU 资源
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

# 查看 RDMA 设备（应显示 8 个 CX-8 PF）
kubectl exec <pod-name> -- rdma link show

# 查看 CX-8 设备详情
kubectl exec <pod-name> -- ibstat
```

## 7.8 GB300 实测结果

### 测试环境

| 项目 | 值 |
|---|---|
| 集群 | chrisya-gb300-gke, tencent-gcp-taiji-poc, us-central1-b |
| 机器类型 | a4x-maxgpu-4g-metal (bare metal) |
| GPU | 4 x GB300, 278 GB HBM3e/GPU |
| RDMA | CX-8 SuperNIC PF (GPUDirect), 8 MRDMA, 3200 Gbps |
| 镜像 | 待定 |
| NCCL | 待定 (>= 2.30.4) |
| 网络 | GIB RDMA (跨节点, CX-8 GPUDirect) + NVSwitch/MNNVL (域内) |
| 网络栈 | IPv6-only |

### Intranode 测试（单节点 4 GPU）

**Legacy Intranode (`tests/legacy/test_intranode.py`)**：

| 操作 | 带宽 (GB/s) | SMs | Chunk | GB200 Baseline | vs GB200 | 说明 |
|---|---|---|---|---|---|---|
| Dispatch BF16 | 待测 | — | — | 465 | — | |
| Dispatch FP8 | 待测 | — | — | 317 | — | |
| Combine | 待测 | — | — | 347 | — | |

**Legacy Low-latency (`tests/legacy/test_low_latency.py`)**：

| 操作 | 带宽 (GB/s) | 延迟 (us) | GB200 Baseline | vs GB200 | 说明 |
|---|---|---|---|---|---|
| Dispatch | 待测 | 待测 | 163-182 / 20-23us | — | |
| Combine | 待测 | 待测 | 323-335 / 21-22us | — | |
| Dispatch + Combine | 待测 | 待测 | 204-209 / 49-53us | — | |

### Internode 测试（Elastic EP, NCCL Gin）

**Elastic EP (`tests/elastic/test_ep.py`)**：

| 操作 | 2n/8GPU | 4n/16GPU | 8n/32GPU | GB200 Baseline (8n) | vs GB200 | 说明 |
|---|---|---|---|---|---|---|
| Dispatch SU | 待测 | 待测 | 待测 | 636 | — | CX-8 GPUDirect 应降低延迟 |
| Combine SU | 待测 | 待测 | 待测 | 590 | — | |
| Reduced Combine SU | 待测 | 待测 | 待测 | 593 | — | |
| Copy (NVLink) | 待测 | 待测 | 待测 | 5700 | — | NVSwitch 拓扑相同 |
| Reduce (NVLink) | 待测 | 待测 | 待测 | 1730 | — | |

### 参数敏感性测试（Intranode）

| 参数组合 | Dispatch BF16 | Dispatch FP8 | Combine | GB200 Baseline |
|---|---|---|---|---|
| topk=4, experts=64 | 待测 | 待测 | 待测 | 432 / 291 / 323 |
| topk=8, experts=256（默认） | 待测 | 待测 | 待测 | 465 / 317 / 347 |

### CX-8 GPUDirect 延迟对比

重点关注 CX-8 GPUDirect 对 all-to-all 延迟的改善：

| 指标 | GB200 (CX-7 VF) | GB300 (CX-8 PF GPUDirect) | 改善 | 说明 |
|---|---|---|---|---|
| Internode Dispatch 延迟 | ~191 us | 待测 | — | CX-8 消除 CPU hop |
| Internode Combine 延迟 | ~358 us | 待测 | — | |
| HybridEP Dispatch 延迟 | — | 待测 | — | HybridEP 应充分利用 GPUDirect |
| HybridEP Combine 延迟 | — | 待测 | — | |

## DeepEP v1 vs v2 选择建议

| 维度 | DeepEP v1 (CUDA 12) | DeepEP v2 (CUDA 13) |
|------|---------------------|---------------------|
| Internode 后端 | NVSHMEM IBGDA — 需 GDRCopy（GCP 不可用） | NCCL Gin — 完全兼容 GCP |
| Intranode | 正常工作 | 正常工作 |
| NVSHMEM 依赖 | 严格要求 3.4.5（GB200/GB300）或 3.3.9（B200/H200） | 仅编译时需要，运行时不依赖 |
| CUDA 容器 | pytorch:25.04-py3 (CUDA 12) | pytorch:26.04-py3 / 26.05-py3 (CUDA 13) |
| GCP 推荐 | 仅 intranode 场景 | **推荐（internode 已验证 144/144 PASS on GB200）** |
| CX-8 GPUDirect | 受益有限（IBGDA 有独立 DMA 路径） | **充分受益**（NCCL Gin 直接走 GPUDirect RDMA） |

**核心决策**：如需跨节点 Expert Parallelism（internode EP），必须使用 **DeepEP v2**。v1 的 internode 通信依赖 NVSHMEM IBGDA + GDRCopy，而 GDRCopy 在 GKE/GCP 环境中无法使用。v2 使用 NCCL Gin 后端完全消除此依赖。

**GB300 额外优势**：DeepEP v2 的 NCCL Gin 后端在 GB300 上可直接利用 CX-8 GPUDirect RDMA 路径。相比 GB200 的 CX-7 VF（RDMA 经 CPU 中转），GB300 的 GPU-initiated 网络操作延迟更低，对 MoE 模型的 Expert dispatch/combine 性能至关重要。HybridEP dispatcher 混合使用 intranode NVLink + internode RDMA 时，CX-8 GPUDirect 让 internode 那条腿的延迟显著降低，整体 all-to-all 延迟更均匀。

## 7.9 部署要点（GB300 GKE DRA 模式）

1. Pod 必须有 `compute-domain-channel` claim — DeepEP v2 的 NCCL Gin 需要 IMEX channel
2. `LD_PRELOAD` 指定 NCCL 2.30.4 — 容器默认的 2.29.7 不含 Gin API
3. `NCCL_NET=gIB` — 启用 GPUDirect RDMA
4. `TRITON_PTXAS_PATH` — DeepEP JIT 编译需要 ptxas
5. AR 镜像仓库需要 `imagePullSecrets` — token 有效期短，需定期刷新
6. Internode test_ep.py 用 `WORLD_SIZE=节点数` + `RANK=节点序号`，不要用 torchrun
7. **RDMA ResourceClaimTemplate count = 8**（GB200 为 4）— GB300 有 8 个 CX-8 MRDMA 接口
8. **IPv6-only 网络**：`MASTER_ADDR` 使用 IPv6 地址，确认 PyTorch/NCCL IPv6 兼容
9. **DRA Driver >= v25.8.0** — GB300 GKE 要求更新版本的 DRA driver
10. **CX-8 设备验证**：部署后先 `rdma link show` 确认 8 个 PF 可见，再跑 DeepEP 测试

### GB200 vs GB300 部署差异速查

| 配置项 | GB200 (A4X) | GB300 (A4X MAX) |
|--------|-------------|-----------------|
| 机器类型 | a4x-highgpu-4g | a4x-maxgpu-4g-metal |
| GPU 显存 | 186 GB/GPU | 278 GB/GPU |
| RDMA NIC | CX-7 VF x4 | CX-8 PF x8 (GPUDirect) |
| MRDMA 接口 | 4 | 8 |
| 网络带宽 | 2000 Gbps | 3200 Gbps |
| 网络栈 | IPv4 | IPv6-only |
| ResourceClaim count | 4 | 8 |
| GKE 最低版本 | 1.32.8+ | 1.34.3-gke.1318000+ |
| DRA driver | v0.4.0 | v25.8.0+ |
| GKE 集群 | — | chrisya-gb300-gke |
| 项目 | — | tencent-gcp-taiji-poc |
| Zone | — | us-central1-b |
