> 🌐 [中文](VALIDATION-ARCHIVE-20260627.md) | **English**

# GB200 NVL72 Deployment Validation Archive — 2026-06-27

## Environment Information

| Item | Value |
|---|---|
| GCP Project | gpu-launchpad-playground |
| Zone | us-east1-d |
| Reservation | nvidia-gb200-z4pzosg110ik8 |
| Machine Type | a4x-highgpu-4g (4 GPU per node) |
| Placement Policy | a4x-nvl72-policy (reused) |
| k8s Version | 1.34.9 |
| Network | chrisya-gvnic-net-0 (reused) |

## VM Inventory

| VM | IP | Role | OS | Notes |
|---|---|---|---|---|
| chrisya-a4x-cp | 10.14.0.3 | Control Plane | Rocky Linux 9.8 x86_64 | n4-standard-8 |
| chrisya-a4x-w0 | 10.14.0.4 | GPU Worker | Rocky Linux 9.8 aarch64 | a4x-highgpu-4g, 4x GB200 |
| chrisya-a4x-w1 | 10.14.0.6 | GPU Worker | Rocky Linux 9.8 aarch64 | a4x-highgpu-4g, created from custom image chrisya-a4x-worker-v1 |

## Validated Components

| Section | Status | Key Results |
|---|---|---|
| 01 Environment Setup | PASS | Conceptual doc validated; three-tier coordination, symmetric NVLink bandwidth |
| 02 k8s Cluster | PASS | kubeadm 1.34.9, Calico VXLANCrossSubnet |
| 03 GPU Stack | PASS | nvidia-device-plugin, DRA Driver 0.4.0, DRANET 1.3.0, ComputeDomain |
| 04 NCCL Test (intra-domain 2n) | PASS | all_reduce 839.54, all_gather 683.83, reduce_scatter 693.07, alltoall 682.73 GB/s |
| 05 RDMA Test | PASS | 4x CX-7 NIC, 382.1-382.2 Gbps/NIC |
| 06 DeepEP Test | PENDING | Pod ready but test not run (machine reclaimed) |
| 07 Megatron Training | NOT STARTED | |

## Key Bugs Found and Fixed

### 1. Calico IP auto-detection picked the wrong NIC (02-k8s-cluster)
- **Symptom**: Pod DNS completely down; calico-node on CP stuck at 0/1 forever
- **Root cause**: A4X Worker has 6 NICs; Calico `firstFound: true` selected the RDMA NIC (10.10.28.x) instead of the management GVNIC (10.14.0.x)
- **Fix**: `nodeAddressAutodetectionV4.cidrs: ["10.14.0.0/24"]`

### 2. kubeadm 1.34 Scheduler RBAC almost entirely blank (03-gpu-stack)
- **Symptom**: DRA ResourceClaims permanently pending; Pods cannot be scheduled
- **Root cause**: The `system:kube-scheduler` ClusterRole had only events permissions, missing all fundamental permissions such as pods/nodes/services/DRA
- **Fix**: Create a `system:kube-scheduler:full` ClusterRole to fill in all permissions required by the scheduler

## GPU Stack Configuration Snapshot

```
Helm releases:
  nvidia-device-plugin 0.19.3 (kube-system)
  nvidia-dra-driver-gpu 0.4.0 (nvidia-dra-driver-gpu)
  dranet 1.3.0 (kube-system)

ComputeDomain: chrisya-compute-domain (UID: e189e7cd-fa34-427e-aa5c-5e391d69ca2c)
Node labels: resource.nvidia.com/computeDomain=<UID>, nvidia.com/gpu.clique=<UID>

Calico: v3.29.3 (Tigera Operator), VXLANCrossSubnet, CIDR autodetect 10.14.0.0/24
Scheduler RBAC: system:kube-scheduler:full (comprehensive)

Custom image: chrisya-a4x-worker-v1 (used for rapidly scaling Workers)
Artifact Registry secret: ar-secret (GCE metadata token, short-lived)
```

## Reproduction Guide

To rebuild this environment:
1. Create CP (n4-standard-8) + Worker (a4x-highgpu-4g) VMs using the same Placement Policy
2. Build the k8s 1.34 cluster following the 02-k8s-cluster doc
3. **Be sure to**: In Step 7 Calico Installation, add `nodeAddressAutodetectionV4.cidrs`
4. Install the GPU Stack following the 03-gpu-stack doc
5. **Be sure to**: In 3.5 Scheduler RBAC, use the full version (system:kube-scheduler:full)
6. Create the ComputeDomain and label the nodes

Alternatively, use the custom image `chrisya-a4x-worker-v1` to rapidly scale Workers (requires `kubeadm reset -f` before re-joining).
