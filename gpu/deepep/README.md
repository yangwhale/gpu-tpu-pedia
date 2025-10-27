# DeepEP 在 GKE B200 GPU 节点上的安装指南

本文档记录了在 Google Kubernetes Engine (GKE) 的 Ubuntu 节点上安装和测试 DeepEP（Deep Efficient Parallelism）的完整步骤。

## 目录

1. [前提条件](#前提条件)
2. [创建 GKE Ubuntu 节点池](#创建-gke-ubuntu-节点池)
3. [部署 DeepEP 安装程序](#部署-deepep-安装程序)
4. [运行测试](#运行测试)
5. [配置说明](#配置说明)

## 前提条件

- GKE 集群已创建
- 配置了必要的 VPC 网络和子网（用于 RDMA）
- kubectl 已配置并可访问集群
- gcloud CLI 已安装并配置

## 创建 GKE Ubuntu 节点池

使用以下命令创建配置了 NVIDIA B200 GPU 的 Ubuntu 节点池：

```bash
gcloud beta container --project "gpu-launchpad-playground" node-pools create "a4-highgpu-ubuntu-02" \
  --cluster "chrisya-gke-a4" \
  --region "asia-southeast1" \
  --node-version "1.33.5-gke.1162000" \
  --machine-type "a4-highgpu-8g" \
  --accelerator "type=nvidia-b200,count=8,gpu-driver-version=disabled" \
  --image-type "UBUNTU_CONTAINERD" \
  --disk-type "hyperdisk-balanced" \
  --disk-size "1000" \
  --ephemeral-storage-local-ssd count=32 \
  --metadata disable-legacy-endpoints=true \
  --scopes "https://www.googleapis.com/auth/cloud-platform" \
  --node-locations "asia-southeast1-b" \
  --spot \
  --num-nodes "2" \
  --enable-private-nodes \
  --enable-autoupgrade \
  --enable-autorepair \
  --max-surge-upgrade 1 \
  --max-unavailable-upgrade 0 \
  --shielded-integrity-monitoring \
  --no-shielded-secure-boot \
  --placement-type=COMPACT \
  --additional-node-network network=chrisya-gke-a4-net-1,subnetwork=chrisya-gke-a4-sub-1 \
  --additional-node-network network=chrisya-gke-a4-rdma-net,subnetwork=chrisya-gke-a4-rdma-sub-0 \
  --additional-node-network network=chrisya-gke-a4-rdma-net,subnetwork=chrisya-gke-a4-rdma-sub-1 \
  --additional-node-network network=chrisya-gke-a4-rdma-net,subnetwork=chrisya-gke-a4-rdma-sub-2 \
  --additional-node-network network=chrisya-gke-a4-rdma-net,subnetwork=chrisya-gke-a4-rdma-sub-3 \
  --additional-node-network network=chrisya-gke-a4-rdma-net,subnetwork=chrisya-gke-a4-rdma-sub-4 \
  --additional-node-network network=chrisya-gke-a4-rdma-net,subnetwork=chrisya-gke-a4-rdma-sub-5 \
  --additional-node-network network=chrisya-gke-a4-rdma-net,subnetwork=chrisya-gke-a4-rdma-sub-6 \
  --additional-node-network network=chrisya-gke-a4-rdma-net,subnetwork=chrisya-gke-a4-rdma-sub-7
```

**关键配置说明：**
- **机器类型**: `a4-highgpu-8g` - 配备 8 个 NVIDIA B200 GPU
- **GPU 驱动**: `gpu-driver-version=disabled` - 使用自定义驱动安装
- **镜像类型**: `UBUNTU_CONTAINERD` - Ubuntu 操作系统
- **存储**: 
  - 持久磁盘: 1000GB Hyperdisk Balanced
  - 本地 SSD: 32 个临时存储
- **网络**: 1 个主网络 + 8 个 RDMA 网络（用于 GPU 间高速通信）
- **节点放置**: `COMPACT` - 确保节点物理位置接近，降低延迟

## 部署 DeepEP 安装程序

### 1. 部署 DaemonSet

使用 [`deepep-installer.yaml`](deepep-installer.yaml) 部署安装程序：

```bash
kubectl apply -f deepep-installer.yaml
```

### 2. 验证部署状态

检查 DaemonSet 创建状态：

```bash
kubectl get daemonset deepep-installer-ubuntu -n kube-system
```

预期输出：
```
NAME                      DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
deepep-installer-ubuntu   2         2         0       2            0
```

### 3. 查看 Pod 状态

```bash
kubectl get pods -n kube-system -l k8s-app=deepep-installer-ubuntu -o wide
```

预期输出：
```
NAME                            READY   STATUS     RESTARTS   AGE
deepep-installer-ubuntu-xxxxx   0/1     Init:0/1   0          2m
```

### 4. 监控安装进度

查看实时安装日志：

```bash
# 获取 Pod 名称
POD_NAME=$(kubectl get pods -n kube-system -l k8s-app=deepep-installer-ubuntu -o jsonpath='{.items[0].metadata.name}')

# 查看安装日志
kubectl logs -n kube-system $POD_NAME -c deepep-installer -f
```

### 5. 等待安装完成

安装过程包括以下阶段（总计约 20-25 分钟）：

1. **DOCA OFED 安装** (3-5 分钟)
   - 下载和安装 Mellanox OFED 驱动
   
2. **GPU 驱动和 CUDA 安装** (5-8 分钟)
   - 安装 NVIDIA 开源驱动 575
   - 安装 CUDA Toolkit 12.9

3. **gdrcopy 编译安装** (2-3 分钟)
   - 从源码编译 gdrcopy v2.5.1
   
4. **NVSHMEM 编译安装** (3-5 分钟)
   - 从源码编译 NVSHMEM 3.2.5
   - **关键**: 指定 CUDA 架构 100 (sm_100) 支持 B200 GPU

5. **DeepEP 编译安装** (3-5 分钟)
   - 安装 PyTorch
   - 编译 DeepEP C++ 扩展

6. **二进制文件复制** (1-2 分钟)
   - 复制库到 GKE GPU 驱动路径

安装完成后，Pod 状态变为 `Running`：

```bash
kubectl get pods -n kube-system -l k8s-app=deepep-installer-ubuntu
```

预期输出：
```
NAME                            READY   STATUS    RESTARTS   AGE
deepep-installer-ubuntu-xxxxx   1/1     Running   0          25m
```

## 运行测试

### 部署测试 Pod

使用 [`deepep-intranode.yaml`](deepep-intranode.yaml) 部署测试 Pod：

```bash
kubectl apply -f deepep-intranode.yaml
```

### 进入 Pod 并运行测试

```bash
# 获取 Pod 名称
kubectl get pods

# 进入 Pod
kubectl exec -it privileged-sleeping-pod -- /bin/bash
```

### 配置环境变量

在 Pod 内执行：

```bash
export DEBIAN_FRONTEND=noninteractive
export PYTHONPATH=/usr/local/nvidia/deepep:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
```

### 安装 DeepEP（在 Pod 内）

```bash
# 更新软件包
apt-get update -y && apt install git python3-pip -y -qq 

# 安装构建依赖
apt install python3.12-dev python3.12 ninja-build cmake build-essential devscripts debhelper dkms -y -qq

# 安装 PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 --break-system-packages

# 克隆并测试 DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git && cd ./DeepEP
```

### 运行低延迟测试

```bash
python3 tests/test_low_latency.py
```

预期输出：
```
Allocating buffer size: 2146.961792 MB ...
[rank 0] Dispatch + combine bandwidth: 256.81 GB/s, avg_t=85.85 us, min_t=76.42 us, max_t=106.50 us
[rank 1] Dispatch + combine bandwidth: 257.06 GB/s, avg_t=85.77 us, min_t=74.02 us, max_t=99.62 us
[rank 2] Dispatch + combine bandwidth: 256.49 GB/s, avg_t=85.96 us, min_t=78.24 us, max_t=105.57 us
...
[rank 7] Dispatch send/recv time: 27.25 + 7.71 us | Combine send/recv time: 32.48 + 9.45 us
```

**性能指标说明：**
- **Dispatch + Combine 带宽**: ~256 GB/s
- **Dispatch 带宽**: ~236 GB/s  
- **Combine 带宽**: ~356 GB/s
- **延迟**: ~7-9 微秒

### 运行节点内测试（可选）

```bash
python3 tests/test_intranode.py
```

### 运行节点间测试

节点间测试用于验证跨节点的 GPU 通信性能。使用 [`deepep-internode.yaml`](deepep-internode.yaml) 部署多节点测试任务：

```bash
kubectl apply -f deepep-internode.yaml
```

**配置说明：**
- 部署一个 Kubernetes Job，包含 2 个并行 Pod
- 每个 Pod 运行在不同的 GKE 节点上
- 使用 `hostNetwork: true` 启用主机网络模式
- 通过 Headless Service 进行 Pod 间通信
- 自动配置分布式训练环境变量（RANK、WORLD_SIZE、MASTER_ADDR）

**查看 Pod 状态：**

```bash
kubectl get pods -l job-name=deepep-job -o wide
```

**查看日志：**

```bash
# 查看 rank 0 的日志
kubectl logs deepep-job-0-xxxxx

# 查看 rank 1 的日志
kubectl logs deepep-job-1-xxxxx
```

**进入 Pod 运行测试：**

```bash
# 进入任一 Pod
kubectl exec -it deepep-job-0-xxxxx -- /bin/bash

# 在 Pod 内配置环境并运行测试
export PYTHONPATH=/usr/local/nvidia/deepep:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

cd DeepEP
python3 tests/test_internode.py
```

**清理测试任务：**

```bash
kubectl delete -f deepep-internode.yaml
```

## 配置说明

### 安装程序环境变量

[`deepep-installer.yaml`](deepep-installer.yaml:62-76) 中的关键配置：

| 环境变量 | 值 | 说明 |
|---------|-----|------|
| `NVSHMEM_IBGDA_SUPPORT` | `1` | 启用 InfiniBand GPU Direct Async |
| `NVSHMEM_USE_GDRCOPY` | `1` | 启用 GPUDirect RDMA Copy |
| `GDRCOPY_HOME` | `/opt/deepep/gdrcopy` | gdrcopy 安装路径 |
| `NVSHMEM_HOME` | `/opt/deepep/nvshmem` | NVSHMEM 安装路径 |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA 安装路径 |
| `TORCH_CUDA_ARCH_LIST_B200` | `10.0` | B200 GPU CUDA 架构版本 |

### 软件版本

- **DOCA OFED**: v3.0.0-058000
- **NVIDIA 驱动**: 575 (开源版本)
- **CUDA Toolkit**: 12.9
- **gdrcopy**: v2.5.1
- **NVSHMEM**: 3.2.5 (源码编译，架构 sm_100)
- **PyTorch**: CUDA 12.9 版本
- **DeepEP**: 最新版本

### 关键技术点

1. **NVSHMEM 架构匹配**
   - B200 GPU 需要 CUDA compute capability 10.0 (sm_100)
   - 必须在编译时指定 `-DCMAKE_CUDA_ARCHITECTURES=100`
   - DeepEP 编译时使用 `TORCH_CUDA_ARCH_LIST=10.0`

2. **库路径配置**
   - NVSHMEM 库: `/opt/deepep/nvshmem/lib`
   - gdrcopy 库: `/opt/deepep/gdrcopy/lib`
   - 系统动态链接器配置: `/etc/ld.so.conf.d/nvshmem.conf`

3. **GKE GPU 驱动路径**
   - 所有库和驱动复制到: `/home/kubernetes/bin/nvidia/`
   - Pod 通过 hostPath 挂载使用

## 参考资源

- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [NVIDIA NVSHMEM 文档](https://docs.nvidia.com/nvshmem/)
- [gdrcopy GitHub](https://github.com/NVIDIA/gdrcopy)
- [GKE GPU 文档](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)

## 更新记录

- 2025-10-26: 初始版本，支持 B200 GPU 的 DeepEP 安装和测试
### 使用 distroless 镜像进行节点间测试

[`deepep-internode-distroless.yaml`](deepep-internode-distroless.yaml) 提供了一个简化的测试方案，使用轻量级 distroless 镜像自动运行跨节点测试。

**部署测试：**

```bash
kubectl apply -f deepep-internode-distroless.yaml
```

**查看 Pod 状态：**

```bash
kubectl get pods -l job-name=deepep-job -o wide
```

预期输出：
```
NAME                 READY   STATUS    RESTARTS   AGE   IP             NODE
deepep-job-0-xxxxx   1/1     Running   0          10s   192.168.0.42   gke-...-vbl5
deepep-job-1-xxxxx   1/1     Running   0          10s   192.168.0.43   gke-...-6slr
```

**查看测试日志：**

```bash
# 查看 rank 0 的日志
kubectl logs deepep-job-0-xxxxx

# 查看 rank 1 的日志  
kubectl logs deepep-job-1-xxxxx

# 实时跟踪日志
kubectl logs -f deepep-job-0-xxxxx
```

**测试结果示例：**

```
+ export RANK=0
+ RANK=0
+ export WORLD_SIZE=2
+ WORLD_SIZE=2
+ export MASTER_ADDR=deepep-job-0.deepep-service
+ MASTER_ADDR=deepep-job-0.deepep-service
+ '[' '!' -d /tmp/deepep_build ']'
+ cd /tmp/deepep_build
Starting test with RANK=0, WORLD_SIZE=2, MASTER_ADDR=deepep-job-0.deepep-service
+ echo 'Starting test with RANK=0, WORLD_SIZE=2, MASTER_ADDR=deepep-job-0.deepep-service'
+ python3 tests/test_internode.py
Testing with seed 0 ...
[config] num_tokens=4096, hidden=7168, num_topk_groups=2, num_topk=8
[layout] Kernel performance: 0.042 ms

[testing] Running with BF16, without top-k (async=False, previous=False) ... passed
[testing] Running with BF16, with top-k (async=False, previous=False) ... passed
[testing] Running with BF16, without top-k (async=False, previous=False) ... passed
[testing] Running with BF16, with top-k (async=False, previous=False) ... passed
[testing] Running with FP8, without top-k (async=False, previous=False) ... passed
[testing] Running with FP8, with top-k (async=False, previous=False) ... passed
[testing] Running with FP8, without top-k (async=False, previous=False) ... passed
[testing] Running with FP8, with top-k (async=False, previous=False) ... passed
[testing] Running with BF16, without top-k (async=True, previous=False) ... passed
[testing] Running with BF16, with top-k (async=True, previous=False) ... passed
[testing] Running with BF16, without top-k (async=True, previous=False) ... passed
[testing] Running with BF16, with top-k (async=True, previous=False) ... passed
[testing] Running with FP8, without top-k (async=True, previous=False) ... passed
[testing] Running with FP8, with top-k (async=True, previous=False) ... passed
[testing] Running with FP8, without top-k (async=True, previous=False) ... passed
[testing] Running with FP8, with top-k (async=True, previous=False) ... passed
[testing] Running with BF16, without top-k (async=False, previous=True) ... passed
[testing] Running with BF16, with top-k (async=False, previous=True) ... passed
[testing] Running with BF16, without top-k (async=False, previous=True) ... passed
[testing] Running with BF16, with top-k (async=False, previous=True) ... passed
[testing] Running with FP8, without top-k (async=False, previous=True) ... passed
[testing] Running with FP8, with top-k (async=False, previous=True) ... passed
[testing] Running with FP8, without top-k (async=False, previous=True) ... passed
[testing] Running with FP8, with top-k (async=False, previous=True) ... passed
[testing] Running with BF16, without top-k (async=True, previous=True) ... passed
[testing] Running with BF16, with top-k (async=True, previous=True) ... passed
[testing] Running with BF16, without top-k (async=True, previous=True) ... passed
[testing] Running with BF16, with top-k (async=True, previous=True) ... passed
[testing] Running with FP8, without top-k (async=True, previous=True) ... passed
[testing] Running with FP8, with top-k (async=True, previous=True) ... passed
[testing] Running with FP8, without top-k (async=True, previous=True) ... passed
[testing] Running with FP8, with top-k (async=True, previous=True) ... passed

[tuning] SMs 24, NVL chunk 4, RDMA chunk 4: 867 + 2008 us, 30.04 GB/s (RDMA), 98.10 GB/s (NVL) 
[tuning] SMs 24, NVL chunk 4, RDMA chunk 8: 134 + 1915 us, 31.50 GB/s (RDMA), 102.86 GB/s (NVL) 
[tuning] SMs 24, NVL chunk 4, RDMA chunk 12: 93 + 1924 us, 31.35 GB/s (RDMA), 102.38 GB/s (NVL) 
[tuning] SMs 24, NVL chunk 4, RDMA chunk 16: 134 + 1907 us, 31.63 GB/s (RDMA), 103.29 GB/s (NVL) 
[tuning] SMs 24, NVL chunk 4, RDMA chunk 20: 293 + 1944 us, 31.03 GB/s (RDMA), 101.33 GB/s (NVL) 
[tuning] SMs 24, NVL chunk 4, RDMA chunk 24: 130 + 2010 us, 30.01 GB/s (RDMA), 98.00 GB/s (NVL) 
[tuning] SMs 24, NVL chunk 4, RDMA chunk 28: 241 + 1963 us, 30.73 GB/s (RDMA), 100.35 GB/s (NVL) 
[tuning] SMs 24, NVL chunk 4, RDMA chunk 32: 159 + 1969 us, 30.63 GB/s (RDMA), 100.04 GB/s (NVL) 

... (中间省略性能调优过程) ...

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 40, RDMA chunk 16: 229 + 1391 us, 43.36 GB/s (RDMA), 141.61 GB/s (NVL)
[tuning] Best dispatch (BF16): SMs 24, NVL chunk 32, RDMA chunk 16: 135 + 2606 us, 44.89 GB/s (RDMA), 146.59 GB/s (NVL)
[tuning] Best combine: SMs 24, NVL chunk 4, RDMA chunk 20, 778.64 + 2681.00 us, 43.63 GB/s (RDMA), 142.49 GB/s (NVL)
```

**性能指标说明：**
- **FP8 最佳性能**: 43.36 GB/s (RDMA), 141.61 GB/s (NVL)
- **BF16 最佳性能**: 44.89 GB/s (RDMA), 146.59 GB/s (NVL)
- **组合最佳性能**: 43.63 GB/s (RDMA), 142.49 GB/s (NVL)
- **所有 32 个功能测试**: 全部通过 ✅

**清理测试任务：**

```bash
kubectl delete -f deepep-internode-distroless.yaml
```
