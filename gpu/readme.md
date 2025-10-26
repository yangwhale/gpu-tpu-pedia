# DeepEP 在 GKE Ubuntu 节点上的安装指南

本文档记录了在 Google Kubernetes Engine (GKE) 的 Ubuntu 节点上安装 DeepEP（Deep Efficient Parallelism）的完整步骤。

## 目录

1. [前提条件](#前提条件)
2. [安装时间估算](#安装时间估算)
3. [安装步骤](#安装步骤)
4. [监控安装进度](#监控安装进度)
5. [验证安装](#验证安装)
6. [故障排查](#故障排查)
7. [配置说明](#配置说明)

## 前提条件

- GKE 集群已创建，包含 Ubuntu 节点
- 节点配置了 NVIDIA B200 或 H200 GPU
- 节点池名称：`a4-highgpu-ubuntu-02`（需要根据实际环境修改）
- kubectl 已配置并可访问集群
- 集群具有互联网访问权限（用于下载依赖包）
- **预留足够时间**：完整安装需要约 **20-25 分钟**

## 安装时间估算

完整的 DeepEP 安装过程需要约 **20-25 分钟**，包括以下主要阶段：

| 阶段 | 预计时间 | 说明 |
|-----|---------|------|
| DOCA OFED 下载和安装 | 3-5 分钟 | 下载约 580MB |
| GPU 驱动编译和安装 | 2-3 分钟 | 包括 DKMS 模块编译 |
| **节点重启** | 1-2 分钟 | ⚠️ **系统会自动重启以加载驱动** |
| CUDA Toolkit 12.8 安装 | 5-8 分钟 | 下载约 3.6GB |
| gdrcopy 编译安装 | 2-3 分钟 | 从源码编译 |
| NVSHMEM 编译安装 | 3-5 分钟 | 从源码编译 |
| PyTorch 和 DeepEP 安装 | 3-5 分钟 | 包括 Python 包和编译 |
| 二进制文件复制 | 1-2 分钟 | 复制到 GKE 路径 |

**重要提示：**
- 安装过程中**系统会自动重启一次**（在 GPU 驱动和 gdrcopy 安装后）
- 重启后安装会自动继续，无需人工干预
- Pod 会保持在 `Init:0/1` 状态直到所有组件安装完成
- 可以通过日志实时监控安装进度

## 安装步骤

### 1. 修改 YAML 配置（重要）

**在部署前，您必须修改 [`deepep-installer-standard.yaml`](deepep-installer-standard.yaml:34-37) 中的节点池名称以匹配您的实际环境：**

```yaml
- key: cloud.google.com/gke-nodepool
  operator: In
  values:
    - a4-highgpu-ubuntu-02  # ⚠️ 请修改为您实际的 nodepool 名称
```

**如何查找您的 nodepool 名称：**

```bash
# 方法1: 查看所有节点的标签
kubectl get nodes --show-labels | grep gke-nodepool

# 方法2: 查看特定节点的详细信息
kubectl describe node <your-node-name> | grep nodepool

# 方法3: 使用 gcloud 命令查看节点池列表
gcloud container node-pools list --cluster=<your-cluster-name>
```

### 2. 部署 DeepEP Installer DaemonSet

修改配置后，使用准备好的 YAML 配置文件部署安装程序：

```bash
kubectl apply -f deepep-installer-standard.yaml
```

**预期输出：**
```
daemonset.apps/deepep-installer-ubuntu created
```

### 2. 验证 DaemonSet 创建状态

检查 DaemonSet 是否成功创建：

```bash
kubectl get daemonset deepep-installer-ubuntu -n kube-system
```

**预期输出：**
```
NAME                      DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
deepep-installer-ubuntu   1         1         0       1            0           <none>          32s
```

**字段说明：**
- `DESIRED`: 期望的 Pod 数量（匹配节点数）
- `CURRENT`: 当前运行的 Pod 数量
- `READY`: 就绪的 Pod 数量
- `UP-TO-DATE`: 已更新的 Pod 数量
- `AVAILABLE`: 可用的 Pod 数量

### 3. 查看 Pod 状态

查看具体的 Pod 部署情况：

```bash
kubectl get pods -n kube-system -l k8s-app=deepep-installer-ubuntu -o wide
```

**预期输出：**
```
NAME                            READY   STATUS     RESTARTS   AGE   IP             NODE                                                  NOMINATED NODE   READINESS GATES
deepep-installer-ubuntu-b7s2q   0/1     Init:0/1   0          47s   192.168.0.28   gke-chrisya-gke-a4-a4-highgpu-ubuntu--752c886b-2b0z   <none>           <none>
```

**状态说明：**
- `Init:0/1`: 表示 initContainer 正在运行（0/1 表示 1 个 initContainer 中的第 0 个正在执行）
- 安装过程在 initContainer 中进行，完成后 Pod 才会变为 Ready 状态

## 监控安装进度

### 查看实时安装日志

使用以下命令查看安装日志（实时跟踪）：

```bash
kubectl logs -n kube-system deepep-installer-ubuntu-b7s2q -c deepep-installer -f
```

或查看最近的 100 行日志：

```bash
kubectl logs -n kube-system deepep-installer-ubuntu-b7s2q -c deepep-installer --tail=100 -f
```

**参数说明：**
- `-n kube-system`: 指定命名空间
- `-c deepep-installer`: 指定容器名称（initContainer）
- `--tail=100`: 显示最后 100 行日志
- `-f`: 实时跟踪日志输出

### 安装阶段说明

安装过程包含以下主要阶段：

1. **DOCA OFED 安装**
   - 下载 DOCA 主机包（约 580MB）
   - 安装 OFED（OpenFabrics Enterprise Distribution）
   - 验证安装：`ofed_info -s`

2. **GPU 驱动和 CUDA 安装**
   - 安装 NVIDIA 开源驱动
   - 安装 CUDA Toolkit 12.8
   - 验证：`nvidia-smi`

3. **gdrcopy 安装**（如果 `NVSHMEM_USE_GDRCOPY=1`）
   - 从 GitHub 克隆源代码
   - 编译并安装 gdrcopy v2.5
   - 安装到 `/opt/deepep/gdrcopy`

4. **系统重启**（⚠️ 自动进行）
   - **触发条件**：在以下任一条件满足时自动重启
     - NVIDIA 驱动模块未加载
     - gdrcopy 驱动模块未加载（如果 `NVSHMEM_USE_GDRCOPY=1`）
     - 驱动配置参数未生效
   - **重启过程**：
     - 系统执行 `reboot` 命令
     - Pod 会短暂断开连接（1-2分钟）
     - 节点重启后，Pod 自动继续执行后续安装步骤
   - **监控提示**：日志输出会显示 "Rebooting the node to load driver and modules..."
   - **无需干预**：重启完全自动化，无需人工操作

5. **NVSHMEM 安装**
   - 下载 NVSHMEM 源代码
   - 使用 CMake 和 Ninja 编译
   - 安装到 `/opt/deepep/nvshmem`
   - 启用 IBGDA 支持

6. **DeepEP 安装**
   - 安装 PyTorch（CUDA 12.8 版本）
   - 克隆 DeepEP 仓库
   - 编译 C++ 扩展
   - 验证 Python 导入

7. **二进制文件复制**
   - 复制所有必要的库到 GKE GPU 驱动路径
   - 目标路径：`/home/kubernetes/bin/nvidia/`

## 验证安装

### 1. 检查 Pod 最终状态

安装完成后，Pod 应该处于 Running 状态：

```bash
kubectl get pods -n kube-system -l k8s-app=deepep-installer-ubuntu
```

**预期输出（安装完成后）：**
```
NAME                            READY   STATUS    RESTARTS   AGE
deepep-installer-ubuntu-b7s2q   1/1     Running   0          15m
```

### 2. 验证安装日志

查看完整的安装日志，确认所有步骤都成功：

```bash
kubectl logs -n kube-system deepep-installer-ubuntu-b7s2q -c deepep-installer | grep -E "(complete|success|installed)"
```

### 3. 检查 NVSHMEM 配置

验证 NVSHMEM IBGDA 支持是否正确启用：

```bash
kubectl logs -n kube-system deepep-installer-ubuntu-b7s2q -c deepep-installer | grep "NVSHMEM_IBGDA_SUPPORT"
```

应该看到：
```
NVSHMEM_IBGDA_SUPPORT is enabled correctly.
```

### 4. 检查 DeepEP 导入

验证 DeepEP Python 模块是否可以正常导入：

```bash
kubectl logs -n kube-system deepep-installer-ubuntu-b7s2q -c deepep-installer | grep "DeepEP installation complete"
```

## 故障排查

### Pod 一直处于 Init 状态

**可能原因：**
1. 下载速度慢（DOCA OFED 包约 580MB，CUDA 约 3.6GB）
2. 编译时间长（NVSHMEM、gdrcopy 需要编译）
3. **节点正在重启**（这是正常现象，加载驱动需要重启）
4. PyTorch 和 DeepEP 编译需要时间

**解决方法：**
- 查看日志确定当前阶段：
  ```bash
  kubectl logs -n kube-system <pod-name> -c deepep-installer --tail=50
  ```
- 等待足够的时间（**完整安装需要 20-25 分钟**）
- 如果看到 "Rebooting the node to load driver and modules..."，这是正常的自动重启过程
- 重启后等待 1-2 分钟，日志会自动恢复

### 节点重启后日志中断

**现象：**
- 执行 `kubectl logs -f` 时连接断开
- Pod 状态仍为 `Init:0/1`

**原因：**
- 这是正常现象，节点重启会导致日志流中断

**解决方法：**
```bash
# 重新连接日志查看
kubectl logs -n kube-system <pod-name> -c deepep-installer --tail=100 -f
```

### 安装失败

**排查步骤：**

1. 查看详细错误日志：
```bash
kubectl logs -n kube-system deepep-installer-ubuntu-<pod-name> -c deepep-installer
```

2. 检查节点标签是否正确：
```bash
kubectl get nodes --show-labels | grep nvidia-b200
```

3. 验证节点是否有足够的资源：
```bash
kubectl describe node <node-name>
```

### 重启安装

如果需要重新安装，可以删除并重新创建 DaemonSet：

```bash
# 删除 DaemonSet
kubectl delete daemonset deepep-installer-ubuntu -n kube-system

# 重新部署
kubectl apply -f deepep-installer-standard.yaml
```

## 配置说明

### 环境变量

[`deepep-installer-standard.yaml`](deepep-installer-standard.yaml:62-77) 中配置的关键环境变量：

| 环境变量 | 值 | 说明 |
|---------|-----|------|
| `NVSHMEM_IBGDA_SUPPORT` | `1` | 启用 NVSHMEM InfiniBand GPU Direct Async 支持 |
| `NVSHMEM_USE_GDRCOPY` | `1` | 启用 GPUDirect RDMA Copy 支持 |
| `GDRCOPY_HOME` | `/opt/deepep/gdrcopy` | gdrcopy 安装路径 |
| `NVSHMEM_HOME` | `/opt/deepep/nvshmem` | NVSHMEM 安装路径 |
| `USE_NVPEERMEM` | `0` | 不使用 nvpeermem 模块 |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA 安装路径 |
| `TORCH_CUDA_ARCH_LIST_B200` | `10.0` | B200 GPU 的 CUDA 架构版本 |
| `TORCH_CUDA_ARCH_LIST_H200` | `9.0` | H200 GPU 的 CUDA 架构版本 |

### 节点选择器

DaemonSet 仅在满足以下条件的节点上运行：

- GPU 类型：`nvidia-b200`
- OS 分发：`ubuntu`
- 节点池：`a4-highgpu-ubuntu-02` ⚠️ **需要根据实际环境修改**

**重要提示：**
在部署前，您必须在 [`deepep-installer-standard.yaml`](deepep-installer-standard.yaml:34-37) 中修改 `cloud.google.com/gke-nodepool` 的值以匹配您的实际节点池名称。这是确保 DaemonSet 正确部署到目标节点的关键配置。

### 安装的软件包版本

- **DOCA OFED**: v3.0.0-058000 (Ubuntu 24.04)
- **CUDA Toolkit**: 12.8
- **gdrcopy**: v2.5
- **NVSHMEM**: CUDA 12 版本
- **PyTorch**: 最新版本（CUDA 12.8）
- **DeepEP**: 最新版本（从 GitHub）

### 安装路径

- **gdrcopy**: `/opt/deepep/gdrcopy`
- **NVSHMEM**: `/opt/deepep/nvshmem`
- **GKE GPU 驱动路径**: `/home/kubernetes/bin/nvidia/`
  - 库文件: `lib64/`
  - 可执行文件: `bin/`
  - DeepEP 模块: `deepep/`
  - 驱动模块: `drivers/`
  - 固件: `firmware/`

## 后续步骤

安装完成后，您可以：

1. 使用 DeepEP 进行多 GPU 训练
2. 配置应用程序 Pod 使用安装的库
3. 验证 NVSHMEM 和 GPUDirect 功能

## 参考资源

- [DeepEP GitHub 仓库](https://github.com/deepseek-ai/DeepEP)
- [NVIDIA NVSHMEM 文档](https://docs.nvidia.com/nvshmem/)
- [gdrcopy GitHub](https://github.com/NVIDIA/gdrcopy)
- [DOCA 下载页面](https://www.mellanox.com/downloads/DOCA/)

## 更新记录

- 2025-10-26: 初始版本，记录基于 [`deepep-installer-standard.yaml`](deepep-installer-standard.yaml) 的安装流程