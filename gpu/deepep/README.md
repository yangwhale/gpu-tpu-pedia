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

### 使用 distroless 镜像进行节点内测试

[`deepep-intranode-distroless.yaml`](deepep-intranode-distroless.yaml) 提供了一个简化的单节点测试方案，使用轻量级 distroless 镜像自动运行节点内测试。

**部署测试：**

```bash
kubectl apply -f deepep-intranode-distroless.yaml
```

**查看 Pod 状态：**

```bash
kubectl get pod deepep-intranode-distroless
```

预期输出：
```
NAME                          READY   STATUS      RESTARTS   AGE
deepep-intranode-distroless   0/1     Completed   0          1m
```

**查看测试日志：**

```bash
kubectl logs deepep-intranode-distroless
```

**测试结果示例：**

```
+ '[' '!' -d /tmp/deepep_build ']'
+ cd /tmp/deepep_build
+ echo 'Starting intranode test'
Starting intranode test
+ python3 tests/test_intranode.py
[config] num_tokens=4096, hidden=7168, num_topk=8
[layout] Kernel performance: 0.049 ms

[testing] All 24 functional tests passed ✅
  - BF16 and FP8 precision tests
  - With/without top-k configurations
  - Async and sync modes
  - With/without previous parameter

[tuning] SMs 24, NVL chunk 4: 172.86 GB/s (NVL), 924.57 us
[tuning] SMs 24, NVL chunk 6: 194.47 GB/s (NVL), 821.85 us
[tuning] SMs 24, NVL chunk 8: 245.96 GB/s (NVL), 649.78 us
[tuning] SMs 24, NVL chunk 10: 284.17 GB/s (NVL), 562.43 us
[tuning] SMs 24, NVL chunk 12: 289.30 GB/s (NVL), 552.44 us
[tuning] SMs 24, NVL chunk 14: 310.92 GB/s (NVL), 514.03 us
[tuning] SMs 24, NVL chunk 16: 322.86 GB/s (NVL), 495.02 us
[tuning] SMs 24, NVL chunk 18: 330.79 GB/s (NVL), 483.16 us
[tuning] SMs 24, NVL chunk 20: 336.69 GB/s (NVL), 474.68 us
[tuning] SMs 24, NVL chunk 22: 341.20 GB/s (NVL), 468.42 us
[tuning] SMs 24, NVL chunk 24: 346.49 GB/s (NVL), 461.26 us
[tuning] SMs 24, NVL chunk 26: 347.37 GB/s (NVL), 460.09 us
[tuning] SMs 24, NVL chunk 28: 348.50 GB/s (NVL), 458.60 us
[tuning] SMs 24, NVL chunk 30: 352.43 GB/s (NVL), 453.49 us
[tuning] SMs 24, NVL chunk 32: 351.81 GB/s (NVL), 454.28 us
[tuning] SMs 24, NVL chunk default: 194.60 GB/s (NVL), 821.28 us
[tuning] Best dispatch (FP8): SMs 24, NVL chunk 30, 352.43 GB/s (NVL), t: 453.49 us

[tuning] SMs 24, NVL chunk 4: 306.74 GB/s (NVL), 1010.48 us
[tuning] SMs 24, NVL chunk 6: 357.41 GB/s (NVL), 867.22 us
[tuning] SMs 24, NVL chunk 8: 437.01 GB/s (NVL), 709.27 us
[tuning] SMs 24, NVL chunk 10: 451.98 GB/s (NVL), 685.78 us

... (中间省略性能调优过程) ...

[tuning] SMs 24, NVL chunk 14: 391.25 GB/s (NVL), 792.23 us
[tuning] SMs 24, NVL chunk 15: 399.63 GB/s (NVL), 775.61 us
[tuning] SMs 24, NVL chunk 16: 391.17 GB/s (NVL), 792.39 us
[tuning] SMs 24, NVL chunk default: 239.64 GB/s (NVL), 1293.42 us
[tuning] Best combine: SMs 24, NVL chunk 15: 399.63 GB/s (NVL), t: 775.61 us
```

**性能指标说明：**
- **FP8 最佳性能**: 352.43 GB/s (NVL), 延迟 453.49 us
- **BF16 最佳性能**: 483.86 GB/s (NVL), 延迟 640.59 us
- **Combine 最佳性能**: 399.63 GB/s (NVL), 延迟 775.61 us
- **所有 24 个功能测试**: 全部通过 ✅

**清理测试任务：**

```bash
kubectl delete pod deepep-intranode-distroless
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
[testing] All 32 functional tests passed ✅
  - BF16 and FP8 precision tests
  - With/without top-k configurations
  - Async and sync modes
  - With/without previous parameter
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
