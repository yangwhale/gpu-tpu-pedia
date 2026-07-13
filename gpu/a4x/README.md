> 🌐 **中文** | [English](README.en.md)

# GB200 A4X 自建 Kubernetes 1.34.1 部署与验证指南

**平台**：NVIDIA GB200 (A4X) · ARM64 · NVL72 · TLinux 4

**Kubernetes**：1.34.1 (kubeadm) + Calico v3.29.3

**GPU 管理**：nvidia-device-plugin v0.17.1 + DRA GPU Driver v25.12.0 (ComputeDomain)

**RDMA 访问**：DRANET v1.3.0（SIG DRA 网络驱动）

**容器**：NGC pytorch:26.05-py3 · TransformerEngine v2.15 · Megatron-LM core_r0.16.0

**网络**：GIB v1.1.2 (NCCL 2.30.4+cuda13.0) · 4×CX-7 400Gbps MRDMA

---

## 架构说明

本文档采用 **DRANET + ComputeDomain** 单一路径——

- **RDMA 访问**：通过 DRANET v1.3.0 将 RDMA NIC 作为 DRA 设备分配给 Pod，无需 hostNetwork
- **IMEX 管理**：通过 ComputeDomain CRD（DRA GPU Driver 原生）管理 IMEX daemon 生命周期，确保 MNNVL 自动就绪
- **GPU 分配**：nvidia-device-plugin（`nvidia.com/gpu: 4`）

这是 NVIDIA 官方推荐的 k8s + DRA 原生路径，所有 MNNVL 资源（IMEX channel + RDMA NIC）均通过 ResourceClaim 声明式管理。

---

## 目录

| 章节 | 内容 | 链接 |
|------|------|------|
| **01-environment-setup** | 架构概述、核心概念、VPC/子网/防火墙、Placement Policy | [01-environment-setup/](01-environment-setup/) |
| **02-k8s-cluster** | Control Plane、Worker 节点创建、k8s join、节点标签 | [02-k8s-cluster/](02-k8s-cluster/) |
| **03-gpu-stack** | nvidia-device-plugin、DRA GPU Driver、DRANET、ComputeDomain、共享存储 | [03-gpu-stack/](03-gpu-stack/) |
| **04-nccl-test** | NCCL 通信测试：单节点、同域 MNNVL、跨域 RDMA、混合 4 节点 | [04-nccl-test/](04-nccl-test/) |
| **05-rdma-test** | RDMA 带宽测试 (ib_write_bw) | [05-rdma-test/](05-rdma-test/) |
| **06-deepep-test** | DeepEP v1/v2 测试、NVSHMEM 兼容矩阵、4-GPU 适配 | [06-deepep-test/](06-deepep-test/) |
| **07-megatron-training** | Megatron-LM 训练：单节点 + 多节点 MoE FP8 | [07-megatron-training/](07-megatron-training/) |
| **08-multi-domain** | 多 Domain 训练编排：JobSet + Kueue TAS、Per-Communicator MNNVL | [08-multi-domain/](08-multi-domain/) |
| **09-rl-training** | RL 训练：veRL/AReaL + MoE 模型在 NVL72 上的实践 | [09-rl-training/](09-rl-training/) |

---

## 测试结果汇总

| 类别 | 测试项 | 实际结果 |
|------|--------|----------|
| RDMA | ib_write_bw per NIC | **~381 Gbps** |
| | 4×NIC aggregate | **~1524 Gbps** |
| NCCL | 单节点 all_reduce 4 GPU @8G | **683.75 GB/s busbw** |
| | 同域 2 节点 MNNVL @8G | **834.95 GB/s busbw** |
| | 跨域 2 节点 RDMA @8G | **325.88 GB/s busbw** |
| | 混合 4 节点 16GPU @8G | **162.45 GB/s busbw** |
| DeepEP v2 | Elastic EP internode (144 configs) | **144/144 PASS** |
| Megatron-LM | 单节点 MoE EP=4 FP8 | **~356 TFLOP/s/GPU** |
| | 多节点 MoE EP=8 MNNVL FP8 | **~274 TFLOP/s/GPU** |

**环境**：k8s 1.34.1 + ComputeDomain (DRA GPU Driver v25.12.0) + DRANET v1.3.0 + GIB v1.1.2 + pytorch:26.05-py3

---

*文档基于 2026-06-19 验证 · GCP GPU Infrastructure Team*
