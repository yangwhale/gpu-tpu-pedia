# DeepEP Installation Guide for GKE B200 GPU Nodes

This document provides a comprehensive guide for installing and testing DeepEP (Deep Efficient Parallelism) on Google Kubernetes Engine (GKE) Ubuntu nodes with NVIDIA B200 GPUs.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Creating GKE Ubuntu Node Pool](#creating-gke-ubuntu-node-pool)
3. [Deploying DeepEP Installer](#deploying-deepep-installer)
4. [Running Tests](#running-tests)
   - [Intranode Testing](#intranode-testing-with-distroless-image)
   - [Internode Testing](#internode-testing-with-distroless-image)
5. [Configuration Reference](#configuration-reference)
6. [Resources](#resources)

## Prerequisites

Before proceeding, ensure you have the following:

- GKE cluster already created
- VPC networks and subnets configured for RDMA
- `kubectl` configured with cluster access
- `gcloud` CLI installed and authenticated
- Appropriate IAM permissions for creating node pools

## Creating GKE Ubuntu Node Pool

Create a node pool configured with NVIDIA B200 GPUs:

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

### Key Configuration Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Machine Type** | `a4-highgpu-8g` | Equipped with 8× NVIDIA B200 GPUs |
| **GPU Driver** | `disabled` | Custom driver installation via DaemonSet |
| **Image Type** | `UBUNTU_CONTAINERD` | Ubuntu OS with containerd runtime |
| **Storage** | 1000GB Hyperdisk + 32× Local SSD | High-performance persistent and ephemeral storage |
| **Networking** | 1 primary + 8 RDMA networks | Enables high-speed GPU-to-GPU communication |
| **Placement** | `COMPACT` | Physical proximity for low latency |

## Deploying DeepEP Installer

### 1. Deploy the DaemonSet

Apply the installer configuration:

```bash
kubectl apply -f deepep-installer.yaml
```

### 2. Verify Deployment

Check DaemonSet status:

```bash
kubectl get daemonset deepep-installer-ubuntu -n kube-system
```

Expected output:
```
NAME                      DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
deepep-installer-ubuntu   2         2         0       2            0
```

### 3. Monitor Installation Progress

View real-time installation logs:

```bash
# Get pod name
POD_NAME=$(kubectl get pods -n kube-system -l k8s-app=deepep-installer-ubuntu -o jsonpath='{.items[0].metadata.name}')

# Follow installation logs
kubectl logs -n kube-system $POD_NAME -c deepep-installer -f
```

### 4. Installation Phases

The installation process takes approximately 20-25 minutes and includes:

1. **DOCA OFED Installation** (3-5 min)
   - Downloads and installs Mellanox OFED drivers

2. **GPU Driver and CUDA Installation** (5-8 min)
   - Installs NVIDIA open-source driver 575
   - Installs CUDA Toolkit 12.9

3. **gdrcopy Compilation** (2-3 min)
   - Compiles gdrcopy v2.5.1 from source

4. **NVSHMEM Compilation** (3-5 min)
   - Compiles NVSHMEM 3.2.5 from source
   - **Critical**: Specifies CUDA architecture 100 (sm_100) for B200 GPU support

5. **DeepEP Compilation** (3-5 min)
   - Installs PyTorch
   - Compiles DeepEP C++ extensions

6. **Binary Deployment** (1-2 min)
   - Copies libraries to GKE GPU driver path

### 5. Verify Installation

Once complete, pod status should show `Running`:

```bash
kubectl get pods -n kube-system -l k8s-app=deepep-installer-ubuntu
```

Expected output:
```
NAME                            READY   STATUS    RESTARTS   AGE
deepep-installer-ubuntu-xxxxx   1/1     Running   0          25m
```

## Running Tests

### Intranode Testing with Distroless Image

The [`deepep-intranode-distroless.yaml`](deepep-intranode-distroless.yaml) provides a lightweight single-node test using a distroless image.

**Deploy test:**

```bash
kubectl apply -f deepep-intranode-distroless.yaml
```

**Check status:**

```bash
kubectl get pod deepep-intranode-distroless
```

**View logs:**

```bash
kubectl logs deepep-intranode-distroless
```

**Expected results:**

```
Starting intranode test
[config] num_tokens=4096, hidden=7168, num_topk=8
[layout] Kernel performance: 0.049 ms

[testing] All 24 functional tests passed ✅
  - BF16 and FP8 precision tests
  - With/without top-k configurations
  - Async and sync modes
  - With/without previous parameter

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 30, 352.43 GB/s (NVL), t: 453.49 us
[tuning] Best dispatch (BF16): SMs 24, NVL chunk 13, 483.86 GB/s (NVL), t: 640.59 us
[tuning] Best combine: SMs 24, NVL chunk 15: 399.63 GB/s (NVL), t: 775.61 us
```

**Performance metrics:**
- FP8 best: 352.43 GB/s, 453.49 μs latency
- BF16 best: 483.86 GB/s, 640.59 μs latency
- All 24 functional tests passed ✅

**Cleanup:**

```bash
kubectl delete pod deepep-intranode-distroless
```

### Internode Testing with Distroless Image

The [`deepep-internode-distroless.yaml`](deepep-internode-distroless.yaml) provides a cross-node test using a distroless image.

**Deploy test:**

```bash
kubectl apply -f deepep-internode-distroless.yaml
```

**Check pod status:**

```bash
kubectl get pods -l job-name=deepep-job -o wide
```

Expected output:
```
NAME                 READY   STATUS    RESTARTS   AGE   IP             NODE
deepep-job-0-xxxxx   1/1     Running   0          10s   192.168.0.42   gke-...-vbl5
deepep-job-1-xxxxx   1/1     Running   0          10s   192.168.0.43   gke-...-6slr
```

**View logs:**

```bash
# Rank 0 logs
kubectl logs deepep-job-0-xxxxx

# Rank 1 logs
kubectl logs deepep-job-1-xxxxx

# Follow logs in real-time
kubectl logs -f deepep-job-0-xxxxx
```

**Expected results:**

```
[testing] All 32 functional tests passed ✅
  - BF16 and FP8 precision tests
  - With/without top-k configurations
  - Async and sync modes
  - With/without previous parameter

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 40, RDMA chunk 16: 229 + 1391 us, 43.36 GB/s (RDMA), 141.61 GB/s (NVL)
[tuning] Best dispatch (BF16): SMs 24, NVL chunk 32, RDMA chunk 16: 135 + 2606 us, 44.89 GB/s (RDMA), 146.59 GB/s (NVL)
[tuning] Best combine: SMs 24, NVL chunk 4, RDMA chunk 20, 778.64 + 2681.00 us, 43.63 GB/s (RDMA), 142.49 GB/s (NVL)
```

**Performance metrics:**
- FP8 best: 43.36 GB/s (RDMA), 141.61 GB/s (NVL)
- BF16 best: 44.89 GB/s (RDMA), 146.59 GB/s (NVL)
- All 32 functional tests passed ✅

**Cleanup:**

```bash
kubectl delete -f deepep-internode-distroless.yaml
```

## Configuration Reference

### Installer Environment Variables

Key environment variables in [`deepep-installer.yaml`](deepep-installer.yaml:62-76):

| Variable | Value | Description |
|----------|-------|-------------|
| `NVSHMEM_IBGDA_SUPPORT` | `1` | Enable InfiniBand GPU Direct Async |
| `NVSHMEM_USE_GDRCOPY` | `1` | Enable GPUDirect RDMA Copy |
| `GDRCOPY_HOME` | `/opt/deepep/gdrcopy` | gdrcopy installation path |
| `NVSHMEM_HOME` | `/opt/deepep/nvshmem` | NVSHMEM installation path |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA installation path |
| `TORCH_CUDA_ARCH_LIST_B200` | `10.0` | B200 GPU CUDA architecture version |

### Software Versions

- **DOCA OFED**: v3.0.0-058000
- **NVIDIA Driver**: 575 (open-source version)
- **CUDA Toolkit**: 12.9
- **gdrcopy**: v2.5.1
- **NVSHMEM**: 3.2.5 (compiled from source with sm_100 architecture)
- **PyTorch**: CUDA 12.9 compatible version
- **DeepEP**: Latest version from source

### Critical Technical Details

1. **NVSHMEM Architecture Matching**
   - B200 GPU requires CUDA compute capability 10.0 (sm_100)
   - Must specify `-DCMAKE_CUDA_ARCHITECTURES=100` during compilation
   - DeepEP compilation uses `TORCH_CUDA_ARCH_LIST=10.0`

2. **Library Path Configuration**
   - NVSHMEM libraries: `/opt/deepep/nvshmem/lib`
   - gdrcopy libraries: `/opt/deepep/gdrcopy/lib`
   - Dynamic linker config: `/etc/ld.so.conf.d/nvshmem.conf`

3. **GKE GPU Driver Path**
   - All libraries copied to: `/home/kubernetes/bin/nvidia/`
   - Pods mount via hostPath for shared access

## Resources

- [DeepEP GitHub Repository](https://github.com/deepseek-ai/DeepEP)
- [NVIDIA NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/)
- [gdrcopy GitHub](https://github.com/NVIDIA/gdrcopy)
- [GKE GPU Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)

## Changelog

- **2025-10-26**: Initial version supporting B200 GPU DeepEP installation and testing

---

For the Chinese version of this guide, see [`readme.md`](readme.md).