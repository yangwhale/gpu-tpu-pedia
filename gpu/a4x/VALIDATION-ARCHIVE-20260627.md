> 🌐 **中文** | [English](VALIDATION-ARCHIVE-20260627.en.md)

# GB200 NVL72 部署验证 Archive — 2026-06-27

## 环境信息

| 项目 | 值 |
|---|---|
| GCP Project | gpu-launchpad-playground |
| Zone | us-east1-d |
| Reservation | nvidia-gb200-z4pzosg110ik8 |
| Machine Type | a4x-highgpu-4g (4 GPU per node) |
| Placement Policy | a4x-nvl72-policy (复用) |
| k8s Version | 1.34.9 |
| Network | chrisya-gvnic-net-0 (复用) |

## VM 清单

| VM | IP | 角色 | OS | 备注 |
|---|---|---|---|---|
| chrisya-a4x-cp | 10.14.0.3 | Control Plane | Rocky Linux 9.8 x86_64 | n4-standard-8 |
| chrisya-a4x-w0 | 10.14.0.4 | GPU Worker | Rocky Linux 9.8 aarch64 | a4x-highgpu-4g, 4x GB200 |
| chrisya-a4x-w1 | 10.14.0.6 | GPU Worker | Rocky Linux 9.8 aarch64 | a4x-highgpu-4g, 从自定义镜像 chrisya-a4x-worker-v1 创建 |

## 已验证的组件

| 章节 | 状态 | 关键结果 |
|---|---|---|
| 01 Environment Setup | PASS | 概念文档验证，三层协调、NVLink 对称带宽 |
| 02 k8s Cluster | PASS | kubeadm 1.34.9, Calico VXLANCrossSubnet |
| 03 GPU Stack | PASS | nvidia-device-plugin, DRA Driver 0.4.0, DRANET 1.3.0, ComputeDomain |
| 04 NCCL Test (同域 2n) | PASS | all_reduce 839.54, all_gather 683.83, reduce_scatter 693.07, alltoall 682.73 GB/s |
| 05 RDMA Test | PASS | 4x CX-7 NIC, 382.1-382.2 Gbps/NIC |
| 06 DeepEP Test | PENDING | Pod 就绪但未运行测试（机器回收） |
| 07 Megatron Training | NOT STARTED | |

## 发现并修复的关键 Bug

### 1. Calico IP 自动检测选错网卡 (02-k8s-cluster)
- **现象**: Pod DNS 完全瘫痪，calico-node on CP 永远 0/1
- **根因**: A4X Worker 6 个 NIC, Calico `firstFound: true` 选中 RDMA 网卡 (10.10.28.x) 而非管理 GVNIC (10.14.0.x)
- **修复**: `nodeAddressAutodetectionV4.cidrs: ["10.14.0.0/24"]`

### 2. kubeadm 1.34 Scheduler RBAC 几乎空白 (03-gpu-stack)
- **现象**: DRA ResourceClaims 永久 pending, Pods 无法调度
- **根因**: `system:kube-scheduler` ClusterRole 只有 events 权限, 缺 pods/nodes/services/DRA 等所有基础权限
- **修复**: 创建 `system:kube-scheduler:full` ClusterRole 补全所有 scheduler 所需权限

## GPU Stack 配置快照

```
Helm releases:
  nvidia-device-plugin 0.19.3 (kube-system)
  nvidia-dra-driver-gpu 0.4.0 (nvidia-dra-driver-gpu)
  dranet 1.3.0 (kube-system)

ComputeDomain: chrisya-compute-domain (UID: e189e7cd-fa34-427e-aa5c-5e391d69ca2c)
Node labels: resource.nvidia.com/computeDomain=<UID>, nvidia.com/gpu.clique=<UID>

Calico: v3.29.3 (Tigera Operator), VXLANCrossSubnet, CIDR autodetect 10.14.0.0/24
Scheduler RBAC: system:kube-scheduler:full (comprehensive)

Custom image: chrisya-a4x-worker-v1 (用于快速扩展 Worker)
Artifact Registry secret: ar-secret (GCE metadata token, 短期有效)
```

## 复现指南

重建此环境需要：
1. 创建 CP (n4-standard-8) + Worker (a4x-highgpu-4g) VMs，使用相同 Placement Policy
2. 按 02-k8s-cluster 文档搭建 k8s 1.34 集群
3. **务必**: Step 7 Calico Installation 加 `nodeAddressAutodetectionV4.cidrs`
4. 按 03-gpu-stack 安装 GPU Stack
5. **务必**: 3.5 Scheduler RBAC 使用完整版 (system:kube-scheduler:full)
6. 创建 ComputeDomain 并标记节点

或者使用自定义镜像 `chrisya-a4x-worker-v1` 快速扩展 Worker (需 `kubeadm reset -f` 后 re-join)。
