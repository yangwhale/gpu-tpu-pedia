> 🌐 [中文](README.md) | **English**

# GB200 A4X Self-Managed Kubernetes 1.34.1 Deployment & Validation Guide

**Platform**: NVIDIA GB200 (A4X) · ARM64 · NVL72 · TLinux 4

**Kubernetes**: 1.34.1 (kubeadm) + Calico v3.29.3

**GPU Management**: nvidia-device-plugin v0.17.1 + DRA GPU Driver v25.12.0 (ComputeDomain)

**RDMA Access**: DRANET v1.3.0 (SIG DRA network driver)

**Container**: NGC pytorch:26.05-py3 · TransformerEngine v2.15 · Megatron-LM core_r0.16.0

**Network**: GIB v1.1.2 (NCCL 2.30.4+cuda13.0) · 4×CX-7 400Gbps MRDMA

---

## Architecture

This guide adopts a single **DRANET + ComputeDomain** path:

- **RDMA access**: DRANET v1.3.0 assigns RDMA NICs to Pods as DRA devices — no hostNetwork required
- **IMEX management**: the ComputeDomain CRD (native to the DRA GPU Driver) manages the IMEX daemon lifecycle, ensuring MNNVL is automatically ready
- **GPU allocation**: nvidia-device-plugin (`nvidia.com/gpu: 4`)

This is NVIDIA's officially recommended k8s + DRA-native path; all MNNVL resources (IMEX channel + RDMA NIC) are managed declaratively via ResourceClaim.

---

## Table of Contents

| Chapter | Contents | Link |
|------|------|------|
| **01-environment-setup** | Architecture overview, core concepts, VPC/subnet/firewall, Placement Policy | [01-environment-setup/](01-environment-setup/README.en.md) |
| **02-k8s-cluster** | Control plane, worker node creation, k8s join, node labels | [02-k8s-cluster/](02-k8s-cluster/README.en.md) |
| **03-gpu-stack** | nvidia-device-plugin, DRA GPU Driver, DRANET, ComputeDomain, shared storage | [03-gpu-stack/](03-gpu-stack/README.en.md) |
| **04-nccl-test** | NCCL communication tests: single-node, intra-domain MNNVL, cross-domain RDMA, mixed 4-node | [04-nccl-test/](04-nccl-test/README.en.md) |
| **05-rdma-test** | RDMA bandwidth test (ib_write_bw) | [05-rdma-test/](05-rdma-test/README.en.md) |
| **06-deepep-test** | DeepEP v1/v2 tests, NVSHMEM compatibility matrix, 4-GPU adaptation | [06-deepep-test/](06-deepep-test/README.en.md) |
| **07-megatron-training** | Megatron-LM training: single-node + multi-node MoE FP8 | [07-megatron-training/](07-megatron-training/README.en.md) |
| **08-multi-domain** | Multi-domain training orchestration: JobSet + Kueue TAS, per-communicator MNNVL | [08-multi-domain/](08-multi-domain/README.en.md) |
| **09-rl-training** | RL training: veRL/AReaL + MoE models on NVL72 | [09-rl-training/](09-rl-training/README.en.md) |

---

## Test Results Summary

| Category | Test | Result |
|------|--------|----------|
| RDMA | ib_write_bw per NIC | **~381 Gbps** |
| | 4×NIC aggregate | **~1524 Gbps** |
| NCCL | Single-node all_reduce 4 GPU @8G | **683.75 GB/s busbw** |
| | Intra-domain 2-node MNNVL @8G | **834.95 GB/s busbw** |
| | Cross-domain 2-node RDMA @8G | **325.88 GB/s busbw** |
| | Mixed 4-node 16GPU @8G | **162.45 GB/s busbw** |
| DeepEP v2 | Elastic EP internode (144 configs) | **144/144 PASS** |
| Megatron-LM | Single-node MoE EP=4 FP8 | **~356 TFLOP/s/GPU** |
| | Multi-node MoE EP=8 MNNVL FP8 | **~274 TFLOP/s/GPU** |

**Environment**: k8s 1.34.1 + ComputeDomain (DRA GPU Driver v25.12.0) + DRANET v1.3.0 + GIB v1.1.2 + pytorch:26.05-py3

---

*Based on validation performed on 2026-06-19 · GCP GPU Infrastructure Team*
