# DeepEP 在 GKE B200 GPU 节点上的安装指南

本文档记录了在 Google Kubernetes Engine (GKE) 的 Ubuntu 节点上安装和测试 DeepEP（Deep Efficient Parallelism）的完整步骤。

## 目录

1. [前提条件](#前提条件)
2. [创建 GKE Ubuntu 节点池](#创建-gke-ubuntu-节点池)
3. [部署 DeepEP 安装程序](#部署-deepep-安装程序)
4. [运行测试](#运行测试)
5. [配置说明](#配置说明)
6. [故障排查](#故障排查)

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

## 故障排查

如果在部署或测试过程中遇到问题，请参阅 [**故障排查指南**](TROUBLESHOOTING.md)，其中包含：

- 常见问题的根本原因分析
- RDMA/NCCL 配置问题的解决方案
- libibverbs 驱动加载问题的修复步骤
- 完整的验证和调试流程

## 参考资源

- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [NVIDIA NVSHMEM 文档](https://docs.nvidia.com/nvshmem/)
- [gdrcopy GitHub](https://github.com/NVIDIA/gdrcopy)
- [GKE GPU 文档](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
- [故障排查指南](TROUBLESHOOTING.md) - DeepEP 在 GKE 上的常见问题和解决方案

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

### 使用 NGC 容器镜像进行节点内测试（完整端到端）

[`deepep-intranode-ngc-25.06.yaml`](deepep-intranode-ngc-25.06.yaml) 提供了一个基于 NGC PyTorch 容器的完整测试方案，使用 hostNetwork 模式进行单节点内 GPU 间通信测试。

#### 部署步骤

**1. 部署 Job：**

```bash
kubectl apply -f deepep-intranode-ngc-25.06.yaml
```

**2. 查看 Pod 状态：**

```bash
kubectl get pods -l job-name=gpu-runtime-wxg-job -o wide
```

预期输出（Pod 使用 hostNetwork，IP 为节点 IP）：
```
NAME                          READY   STATUS    RESTARTS   AGE   IP             NODE
gpu-runtime-wxg-job-0-xxxxx   2/2     Running   0          60s   192.168.0.42   gke-...-vbl5
gpu-runtime-wxg-job-1-xxxxx   2/2     Running   0          60s   192.168.0.43   gke-...-6slr
```

**3. 查看测试日志：**

```bash
# 查看 rank 0 的日志（主节点）
POD_NAME=$(kubectl get pods -l job-name=gpu-runtime-wxg-job -o jsonpath='{.items[0].metadata.name}')
kubectl logs $POD_NAME -c ngc-25-04 -f
```

#### 测试结果示例

```
Pod 0 started with RANK=0, WORLD_SIZE=2, MASTER_ADDR=gpu-runtime-wxg-job-0.gpu-runtime-wxg-service
Starting DeepEP intranode test...

Testing with seed 0 ...
[config] num_tokens=4096, hidden=7168, num_topk_groups=2, num_topk=8
[layout] Kernel performance: 0.048 ms

[testing] Running with BF16, without top-k (async=False, previous=False) ... passed
[testing] Running with BF16, with top-k (async=False, previous=False) ... passed
[testing] Running with FP8, without top-k (async=False, previous=False) ... passed
[testing] Running with FP8, with top-k (async=False, previous=False) ... passed

... (所有 32 个功能测试) ...

[testing] All 32 functional tests passed ✅
  - BF16 and FP8 precision tests
  - With/without top-k configurations
  - Async and sync modes
  - With/without previous parameter

[tuning] SMs 24, NVL chunk 4, RDMA chunk 4: 652 + 2113 us, 28.55 GB/s (RDMA), 93.22 GB/s (NVL)
[tuning] SMs 24, NVL chunk 8, RDMA chunk 8: 500 + 1616 us, 37.33 GB/s (RDMA), 121.89 GB/s (NVL)
[tuning] SMs 24, NVL chunk 12, RDMA chunk 16: 268 + 1601 us, 37.68 GB/s (RDMA), 123.04 GB/s (NVL)
[tuning] SMs 24, NVL chunk 28, RDMA chunk 16: 609 + 1458 us, 41.37 GB/s (RDMA), 135.10 GB/s (NVL)

... (中间省略性能调优过程) ...

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 28, RDMA chunk 16: 609 + 1458 us, 41.37 GB/s (RDMA), 135.10 GB/s (NVL)
[tuning] Best dispatch (BF16): SMs 24, NVL chunk 12, RDMA chunk 32: 330 + 2807 us, 41.68 GB/s (RDMA), 136.10 GB/s (NVL)
[tuning] Best combine: SMs 24, NVL chunk 6, RDMA chunk 20, 918.76 + 2763.00 us, 42.34 GB/s (RDMA), 138.26 GB/s (NVL)

Test completed successfully!
```

#### 性能指标说明

| 测试类型 | RDMA 带宽 | NVL 带宽 | 延迟 |
|---------|-----------|----------|------|
| **FP8 最佳** | 41.37 GB/s | 135.10 GB/s | 609 + 1458 us |
| **BF16 最佳** | 41.68 GB/s | 136.10 GB/s | 330 + 2807 us |
| **Combine 最佳** | 42.34 GB/s | 138.26 GB/s | 918 + 2763 us |

**关键特性：**
- ✅ **所有 32 个功能测试通过** - 涵盖 BF16 和 FP8 精度的各种配置
- ✅ **RDMA 通信正常** - 跨节点 GPU 间高速通信
- ✅ **NVL 通信正常** - 节点内 GPU 间 NVLink 通信
- ✅ **性能调优完成** - 自动找到最佳参数组合

#### 清理测试任务

```bash
kubectl delete -f deepep-intranode-ngc-25.06.yaml
```

#### 配置说明

该配置文件的关键特性：

1. **hostNetwork 模式**: 使用 `hostNetwork: true` 和 `hostPID: true`，直接使用节点网络，避免 Pod 间 SSH 配置复杂性

2. **RDMA 驱动修复**: 自动复制 `libmlx5-rdmav57.so` 到系统默认路径，解决 libibverbs 驱动加载问题

3. **环境变量配置**:
   - `NCCL_NET=gIB` - 使用 Google InfiniBand 插件
   - `GDRCOPY_HOME=/usr/local/nvidia` - GPUDirect RDMA Copy 路径
   - `NVSHMEM_HOME=/usr/local/nvidia` - NVSHMEM 库路径

4. **自动化测试**: 容器启动后自动克隆 DeepEP 仓库并运行 `test_intranode.py`

5. **Job 模式**: 使用 Kubernetes Job 的 Indexed completion 模式，支持多节点并行部署

详细的故障排查步骤请参阅 [故障排查指南](TROUBLESHOOTING.md)。

### NCCL 跨节点性能测试

[`nccl-test-internode-job.yaml`](nccl-test-internode-job.yaml) 提供了一个完整的 NCCL 跨节点性能测试方案，使用 SSH 进行节点间通信，测试 RDMA over InfiniBand 的性能。

#### 前置条件：创建 SSH Secret

NCCL 跨节点测试需要 SSH 互信，首先需要创建 SSH 密钥对并存储为 Kubernetes Secret。

**1. 生成 SSH 密钥对：**

```bash
# 在本地临时目录生成密钥对
mkdir -p /tmp/nccl_ssh_key
ssh-keygen -t rsa -b 4096 -f /tmp/nccl_ssh_key/id_rsa -N "" -C "nccl-test"

# 创建 authorized_keys
cat /tmp/nccl_ssh_key/id_rsa.pub > /tmp/nccl_ssh_key/authorized_keys

# 创建 SSH config
cat > /tmp/nccl_ssh_key/config <<EOF
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    Port 2222
EOF
```

**2. 创建 Kubernetes Secret：**

```bash
# 创建 Secret（包含私钥、公钥、authorized_keys 和 config）
kubectl create secret generic nccl-ssh-key \
  --from-file=id_rsa=/tmp/nccl_ssh_key/id_rsa \
  --from-file=id_rsa.pub=/tmp/nccl_ssh_key/id_rsa.pub \
  --from-file=authorized_keys=/tmp/nccl_ssh_key/authorized_keys \
  --from-file=config=/tmp/nccl_ssh_key/config

# 验证 Secret 创建成功
kubectl get secret nccl-ssh-key

# （可选）查看 Secret 内容
kubectl describe secret nccl-ssh-key
```

**3. （可选）清理本地密钥文件：**

```bash
# Secret 创建后，可以删除本地临时文件
rm -rf /tmp/nccl_ssh_key
```

#### 部署测试

**1. 部署 NCCL 测试 Job：**

```bash
kubectl apply -f nccl-test-internode-job.yaml
```

**2. 查看 Pod 状态：**

```bash
kubectl get pods -l job-name=nccl-test-job -o wide
```

预期输出：
```
NAME                    READY   STATUS    RESTARTS   AGE   IP             NODE
nccl-test-job-0-xxxxx   2/2     Running   0          60s   192.168.0.43   gke-...-6slr
nccl-test-job-1-xxxxx   2/2     Running   0          60s   192.168.0.42   gke-...-vbl5
```

**3. 监控测试进度：**

```bash
# 查看 rank 0（主节点）的实时日志
POD_NAME=$(kubectl get pods -l job-name=nccl-test-job -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $POD_NAME -c ngc-25-04
```

#### 测试结果示例

**SSH 连接验证：**
```
Testing SSH to both nodes on port 2222...
Warning: Permanently added '[nccl-test-job-0.nccl-test-service]:2222' (ED25519) to the list of known hosts.
gke-chrisya-gke-a4-a4-highgpu-ubuntu--752c886b-6slr
Warning: Permanently added '[nccl-test-job-1.nccl-test-service]:2222' (ED25519) to the list of known hosts.
gke-chrisya-gke-a4-a4-highgpu-ubuntu--752c886b-vbl5
```

**单节点 NCCL 性能测试（8 GPUs）：**
```
# nThread 1 nGpus 8 minBytes 8388608 maxBytes 17179869184 step: 2(factor)
#
# Using devices
#  Rank  0 Group  0 Pid 221599 device  0 [0000:8f:00] NVIDIA B200
#  Rank  1 Group  0 Pid 221599 device  1 [0000:90:00] NVIDIA B200
#  Rank  2 Group  0 Pid 221599 device  2 [0000:96:00] NVIDIA B200
#  Rank  3 Group  0 Pid 221599 device  3 [0000:97:00] NVIDIA B200
#  Rank  4 Group  0 Pid 221599 device  4 [0000:c4:00] NVIDIA B200
#  Rank  5 Group  0 Pid 221599 device  5 [0000:c5:00] NVIDIA B200
#  Rank  6 Group  0 Pid 221599 device  6 [0000:cb:00] NVIDIA B200
#  Rank  7 Group  0 Pid 221599 device  7 [0000:cc:00] NVIDIA B200
#
#       size         count      type   redop    root     time   algbw   busbw
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)
     8388608       2097152     float     sum      -1    80.32  104.44  182.77
    16777216       4194304     float     sum      -1    110.5  151.81  265.67
    33554432       8388608     float     sum      -1    172.6  194.36  340.13
    67108864      16777216     float     sum      -1    280.0  239.71  419.49
   134217728      33554432     float     sum      -1    396.9  338.18  591.81
   268435456      67108864     float     sum      -1    714.7  375.58  657.26
   536870912     134217728     float     sum      -1   1335.6  401.96  703.44
  1073741824     268435456     float     sum      -1   2576.7  416.72  729.25
  2147483648     536870912     float     sum      -1   5102.4  420.87  736.53
  4294967296    1073741824     float     sum      -1    10171  422.27  738.97
  8589934592    2147483648     float     sum      -1    17899  479.90  839.82
 17179869184    4294967296     float     sum      -1    35561  483.11  845.44
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 587.371 GB/s
```

**跨节点 NCCL 性能测试（16 GPUs，2 nodes）：**
```
# nThread 1 nGpus 1 minBytes 8388608 maxBytes 17179869184 step: 2(factor)
#
# Using devices (across 2 nodes)
#  Rank  0 Group  0 device  0 [0000:8f:00] NVIDIA B200 (node 0)
#  Rank  1 Group  0 device  1 [0000:90:00] NVIDIA B200 (node 0)
#  ...
#  Rank  8 Group  0 device  0 [0000:8f:00] NVIDIA B200 (node 1)
#  Rank  9 Group  0 device  1 [0000:90:00] NVIDIA B200 (node 1)
#  ...
#  Rank 15 Group  0 device  7 [0000:cc:00] NVIDIA B200 (node 1)
#
#       size         count      type   redop    root     time   algbw   busbw
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)
     8388608       2097152     float     sum      -1    158.8   52.83   99.06
    16777216       4194304     float     sum      -1    193.2   86.83  162.81
    33554432       8388608     float     sum      -1    291.0  115.32  216.22
    67108864      16777216     float     sum      -1    483.3  138.85  260.35
   134217728      33554432     float     sum      -1    714.1  187.94  352.39
   268435456      67108864     float     sum      -1   1114.8  240.79  451.48
   536870912     134217728     float     sum      -1   1808.9  296.79  556.48
  1073741824     268435456     float     sum      -1   3328.7  322.57  604.82
  2147483648     536870912     float     sum      -1   6171.1  347.99  652.48
  4294967296    1073741824     float     sum      -1    12219  351.51  659.09
  8589934592    2147483648     float     sum      -1    23842  360.29  675.54
 17179869184    4294967296     float     sum      -1    47062  365.05  684.46
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 448.13 GB/s
```

#### 性能指标对比

| 测试配置 | GPU 数量 | 平均带宽 | 最大带宽 | 特点 |
|---------|---------|---------|---------|------|
| **单节点** | 8 GPUs | **587.4 GB/s** | 845.4 GB/s | NVLink 节点内通信 |
| **跨节点** | 16 GPUs (2×8) | **448.1 GB/s** | 684.5 GB/s | RDMA over InfiniBand |

**关键特性：**
- ✅ **SSH 互信成功** - 通过 Kubernetes Secret 实现跨节点 SSH 通信
- ✅ **单节点高性能** - 587.4 GB/s 平均带宽（NVLink）
- ✅ **跨节点 RDMA** - 448.1 GB/s 平均带宽（InfiniBand）
- ✅ **16 GPU 全部识别** - 两个节点共 16 个 NVIDIA B200
- ✅ **MPI 协调成功** - 使用 Open MPI 进行多进程协调

#### 清理测试任务

```bash
# 删除 Job（会自动清理所有 Pod）
kubectl delete job nccl-test-job

# 删除 Service
kubectl delete service nccl-test-service

# （可选）删除 SSH Secret
kubectl delete secret nccl-ssh-key
```

#### 配置说明

该测试配置的关键特性：

1. **SSH 配置**:
   - 使用端口 2222（避免与主机 SSH 冲突）
   - Secret 挂载到 `/etc/ssh-keys`（只读）
   - 复制到 `/root/.ssh/`（可写，设置正确权限）

2. **MPI 配置**:
   - `--mca orte_keep_fqdn_hostnames 1` - 强制使用 FQDN
   - `--mca plm_rsh_args "-p 2222"` - SSH 端口 2222
   - Hostfile: `nccl-test-job-{0,1}.nccl-test-service`

3. **NCCL 环境变量**:
   - `NCCL_NET=gIB` - Google InfiniBand 插件
   - `NCCL_CROSS_NIC=0` - 禁用跨网卡通信
   - `NCCL_IB_HCA` - 指定 8 个 InfiniBand 适配器

4. **Job 模式**:
   - `completions: 2, parallelism: 2` - 两个并行节点
   - `completionMode: Indexed` - 提供唯一的 rank ID
   - `subdomain: nccl-test-service` - 稳定的 DNS 名称

详细的故障排查步骤请参阅 [故障排查指南](TROUBLESHOOTING.md)。

---

英文版本请参阅 [`readme_en.md`](readme_en.md)。
