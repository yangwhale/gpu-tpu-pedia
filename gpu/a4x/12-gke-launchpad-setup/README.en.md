> 🌐 [中文](README.md) | **English**

# gpu-launchpad-playground GKE A4X Cluster Setup Guide

> Benchmarked against baker's (supercomputer-testing) GKE cluster, built in us-east1-d of the gpu-launchpad-playground project.
>
> Cluster name: `chrisya-a4x-gke-v2`, 16 GB200 machines (64 GPUs), single-domain NVL72.

## Prerequisites

- Project: `gpu-launchpad-playground`
- Zone: `us-east1-d`
- Reservation: `nvidia-gb200-z4pzosg110ik8` (36 a4x-highgpu-4g machines, 2 subblocks × 18)
- Placement Policy: `a4x-nvl72-policy` (domain 1) + `forrest-a4x-1x72-policy` (domain 2)
- VPCs already exist:
  - Primary management network: `chrisya-gvnic-net-0` / `chrisya-gvnic-sub-0` (10.14.0.0/16) — internal peering with CC-TW on the same VPC
  - Secondary gVNIC: `chrisya-gvnic-net-1` / `chrisya-gvnic-sub-1-ue1` (10.15.0.0/16)
  - RDMA: `chrisya-a4x-rdma-net` / 4 subnets (10.10.16-28.0/22)

## Step 1: Add GKE Secondary Ranges

GKE requires secondary IP ranges for Pods and Services:

```bash
gcloud compute networks subnets update chrisya-gvnic-sub-0 \
    --project=gpu-launchpad-playground --region=us-east1 \
    --add-secondary-ranges=gke-pods=10.28.0.0/14,gke-services=10.32.0.0/20
```

## Step 2: Create Cloud NAT

A private cluster needs NAT so that pods can pull public images:

```bash
gcloud compute routers create chrisya-gke-nat-router \
    --project=gpu-launchpad-playground --region=us-east1 \
    --network=chrisya-gvnic-net-0

gcloud compute routers nats create chrisya-gke-nat \
    --project=gpu-launchpad-playground --region=us-east1 \
    --router=chrisya-gke-nat-router \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges
```

## Step 3: Create the GKE Cluster

```bash
gcloud container clusters create chrisya-a4x-gke-v2 \
    --project=gpu-launchpad-playground \
    --zone=us-east1-d \
    --release-channel=rapid \
    --enable-ip-alias \
    --network=chrisya-gvnic-net-0 \
    --subnetwork=chrisya-gvnic-sub-0 \
    --cluster-secondary-range-name=gke-pods \
    --services-secondary-range-name=gke-services \
    --enable-dataplane-v2 \
    --enable-multi-networking \
    --enable-private-nodes \
    --master-ipv4-cidr=172.16.2.0/28 \
    --workload-pool=gpu-launchpad-playground.svc.id.goog \
    --num-nodes=1 \
    --machine-type=e2-standard-16 \
    --disk-type=pd-ssd --disk-size=200 \
    --addons=GcsFuseCsiDriver,LustreCsiDriver \
    --scopes=cloud-platform
```

Key parameter notes:
- `--enable-multi-networking`: required for RDMA multi-NIC
- `--enable-dataplane-v2`: matches baker's ADVANCED_DATAPATH
- `--enable-private-nodes`: security isolation
- `--addons=GcsFuseCsiDriver,LustreCsiDriver`: GCS and Lustre storage

## Step 4: Configure Master Authorized Networks

```bash
gcloud container clusters update chrisya-a4x-gke-v2 \
    --zone=us-east1-d --project=gpu-launchpad-playground \
    --enable-master-authorized-networks \
    --master-authorized-networks=<CC-TW-IP>/32,<GLINUX-IP>/32,10.14.0.0/16
```

## Step 5: Create the A4X GPU Node Pool

One node pool per domain, with the physical domain controlled by the placement policy:

```bash
# Domain 1 (a4x-nvl72-policy → subblock-0001)
gcloud container node-pools create a4x-domain-1 \
    --project=gpu-launchpad-playground \
    --cluster=chrisya-a4x-gke-v2 --zone=us-east1-d \
    --machine-type=a4x-highgpu-4g \
    --accelerator=type=nvidia-gb200,count=4,gpu-driver-version=latest \
    --num-nodes=8 \
    --disk-type=hyperdisk-balanced --disk-size=100 \
    --ephemeral-storage-local-ssd=count=4 \
    --reservation-affinity=specific --reservation=nvidia-gb200-z4pzosg110ik8 \
    --placement-type=COMPACT --placement-policy=a4x-nvl72-policy \
    --enable-gvnic \
    --additional-node-network=network=chrisya-gvnic-net-1,subnetwork=chrisya-gvnic-sub-1-ue1 \
    --additional-node-network=network=chrisya-a4x-rdma-net,subnetwork=chrisya-a4x-rdma-net-sub-0 \
    --additional-node-network=network=chrisya-a4x-rdma-net,subnetwork=chrisya-a4x-rdma-net-sub-1 \
    --additional-node-network=network=chrisya-a4x-rdma-net,subnetwork=chrisya-a4x-rdma-net-sub-2 \
    --additional-node-network=network=chrisya-a4x-rdma-net,subnetwork=chrisya-a4x-rdma-net-sub-3 \
    --scopes=cloud-platform

# Domain 2 (forrest-a4x-1x72-policy → subblock-0002)
# Same as above, but change --placement-policy=forrest-a4x-1x72-policy
```

### Key Notes on Placement Policy

**You must use a placement policy already bound to the reservation.** Each `1x72` COLLOCATED policy locks onto one physical NVL72 subblock. A newly created policy will report `ZONE_RESOURCE_POOL_EXHAUSTED` if both subblocks are already taken by existing policies.

Current reservation-to-subblock bindings:

| Placement Policy | Subblock | Notes |
|---|---|---|
| `a4x-nvl72-policy` | subblock-0001 | ivy 17 machines + tlinux 1 machine |
| `forrest-a4x-1x72-policy` | subblock-0002 | our 16 machines + tlinux 1 machine |

### `--ephemeral-storage-local-ssd=count=4`

**Must be specified.** The reservation's instance spec requires 4 × 3TB local SSDs. Omitting this parameter results in `ZONE_RESOURCE_POOL_EXHAUSTED` (reservation config mismatch).

## Step 6: Install GPU Stack Components

After the GKE cluster is created, the following components are installed automatically:
- GPU device plugin (nvidia-gpu-device-plugin-large-cos)
- Lustre CSI
- GCS Fuse CSI
- Networking DRA driver (gke-managed-networking-dra-driver)

The following must be installed manually:

### 6.1 LeaderWorkerSet Controller

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/latest/download/manifests.yaml
```

### 6.2 NCCL RDMA Installer (GIB)

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-rdma-installer-a4x.yaml
```

After installation, each node's `/home/kubernetes/bin/gib/` directory contains the GIB NCCL plugin (`libnccl-net.so`).

### 6.3 NVIDIA DRA GPU Driver (ComputeDomain)

> **Critical**: You must use **v25.12.0+** from the NVIDIA NGC Helm repo, not v0.4.0 from the open-source registry.k8s.io. The v25.3.x ComputeDomain daemon cannot initialize IMEX correctly (0/1 not ready); v25.12.0 fixes this issue.

```bash
# 1. ResourceQuota (the DRA daemon uses system-critical priority)
kubectl create ns nvidia-dra-driver-gpu
kubectl apply -n nvidia-dra-driver-gpu -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: nvidia-dra-driver-gpu-quota
spec:
  hard:
    pods: "37"
  scopeSelector:
    matchExpressions:
    - operator: In
      scopeName: PriorityClass
      values:
        - system-node-critical
        - system-cluster-critical
EOF

# 2. Install via NGC Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update

cat > /tmp/dra-values.yaml <<EOF
nvidiaDriverRoot: /home/kubernetes/bin/nvidia
resources:
  gpus:
    enabled: false
controller:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: "nvidia.com/gpu"
            operator: "DoesNotExist"
kubeletPlugin:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: cloud.google.com/gke-accelerator
                operator: In
                values:
                  - nvidia-gb200
              - key: kubernetes.io/arch
                operator: In
                values:
                  - arm64
  tolerations:
    - key: nvidia.com/gpu
      operator: Equal
      value: present
      effect: NoSchedule
    - key: kubernetes.io/arch
      operator: Equal
      value: arm64
      effect: NoSchedule
EOF

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
    --version="25.12.0" \
    --namespace nvidia-dra-driver-gpu \
    -f /tmp/dra-values.yaml
```

Verify after installation: `kubectl get pods -n nvidia-dra-driver-gpu` should show 1 controller + N kubelet-plugins, all Running.

**GKE COS key points**:
- `nvidiaDriverRoot` must be `/home/kubernetes/bin/nvidia` (the COS driver path)
- No need to manually apply NFD labels (the NGC chart uses `cloud.google.com/gke-accelerator` node affinity)
- No need to manually install the ComputeDomain CRD (bundled in the NGC chart)

### 6.4 Network Objects (RDMA)

```yaml
# Network + GKENetworkParamSet for gvnic-1 + rdma-0~3
# deviceMode for RDMA must be set to RDMA
apiVersion: networking.gke.io/v1
kind: GKENetworkParamSet
metadata:
  name: rdma-0
spec:
  vpc: chrisya-a4x-rdma-net
  vpcSubnet: chrisya-a4x-rdma-net-sub-0
  deviceMode: RDMA
```

## Step 7: Create the ComputeDomain and Label the Nodes

```bash
# Create the ComputeDomain
kubectl apply -f - <<EOF
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: my-compute-domain
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate:
      name: my-compute-domain-channel
EOF

# Get the UID and label the GPU nodes (triggers ComputeDomain daemon deployment)
CD_UID=$(kubectl get computedomain my-compute-domain -o jsonpath='{.metadata.uid}')
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-gb200 --no-headers | \
  awk '{print $1}' | xargs -I{} kubectl label node {} "resource.nvidia.com/computeDomain=$CD_UID" --overwrite

# Verify that all daemons are 1/1 Ready
kubectl get pods -n nvidia-dra-driver-gpu | grep computedomain
```

## Step 8: Verify

```bash
# GPU nodes
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-gb200

# DRA driver (1 controller + N kubelet-plugin)
kubectl get pods -n nvidia-dra-driver-gpu

# ComputeDomain daemon (should all be 1/1 Ready)
kubectl get pods -n nvidia-dra-driver-gpu | grep computedomain

# NCCL RDMA installer (N/N Running)
kubectl get daemonsets -n kube-system | grep nccl-rdma

# DeviceClasses
kubectl get deviceclasses  # compute-domain-*.nvidia.com + mrdma.google.com

# LWS
kubectl get pods -n lws-system
```

## Step 9: Deploy the Training Workload

The Pod requires the following configuration (refer to the official GKE docs):

```yaml
metadata:
  annotations:
    networking.gke.io/default-interface: 'eth0'
    networking.gke.io/interfaces: |
      [
        {"interfaceName":"eth0","network":"default"},
        {"interfaceName":"eth2","network":"rdma-0"},
        {"interfaceName":"eth3","network":"rdma-1"},
        {"interfaceName":"eth4","network":"rdma-2"},
        {"interfaceName":"eth5","network":"rdma-3"}
      ]
spec:
  resourceClaims:
  - name: compute-domain-channel
    resourceClaimTemplateName: my-compute-domain-channel
  volumes:
  - {name: nvidia, hostPath: {path: /home/kubernetes/bin/nvidia}}
  - {name: gib, hostPath: {path: /home/kubernetes/bin/gib}}
  containers:
  - resources:
      claims: [{name: compute-domain-channel}]
      limits: {nvidia.com/gpu: "4"}
    volumeMounts:
    - {name: nvidia, mountPath: /usr/local/nvidia}
    - {name: gib, mountPath: /usr/local/gib}
    env:
    - {name: LD_LIBRARY_PATH, value: "/usr/local/nvidia/lib64"}
```

### DSv3 16L Training Benchmark (2026-07-08)

Single-domain 16 nodes, 64 GPUs, NeMo Bridge `run_script.py -m deepseek -mr deepseek_v3 -c fp8_mx`:

| iter | step time | Notes |
|---|---|---|
| 6 | 3207ms | warmup |
| 7-10 | ~2540ms | **steady state** |
| 11 | 3159ms | VPP spike (normal) |
| 12-15 | ~2540ms | **steady state** |
| 16 | 3165ms | VPP spike |
| 17-20 | ~2545ms | **steady state** |

Steady state ~2.54s/step, estimated at **~1030 TFLOPs/GPU**. All 20 steps completed with zero errors.

**Note**: For a single domain, `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` must equal the EP degree (EP=32 → set 32, not 64).

## Troubleshooting Log

| Issue | Cause | Fix |
|---|---|---|
| `ZONE_RESOURCE_POOL_EXHAUSTED` when creating VMs | A new placement policy cannot allocate onto an already-occupied subblock | Use a placement policy already bound to the reservation |
| `ZONE_RESOURCE_POOL_EXHAUSTED` without a policy | The reservation requires 4 local SSDs, but the create command did not specify them | Add `--ephemeral-storage-local-ssd=count=4` |
| LWS image pull failure | Private cluster has no Cloud NAT | Create a Cloud Router + NAT |
| kubectl cannot reach the private cluster | Master authorized networks not configured | Add the IPs of CC-TW and gLinux |
| Lustre CSI not enabled | Forgot to add the addon at cluster creation | `--addons=LustreCsiDriver`, or a later `cluster update` |
| DRA v25.3.x CD daemon 0/1 not ready | IMEX initialization failure + 409 Conflict race | **Upgrade to v25.12.0** |
| CUDA 801 `operation not supported` | Training started before the ComputeDomain daemon was ready | Ensure all CD daemons are 1/1 Ready before starting training |
| `ranks 32 not divisible by ranks_per_node 64` | Single-domain EP=32 but `EP_RANKS_PER_DOMAIN=64` | Change to `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32` |
| DRA PreBind `nil request mappings` (v0.4.0) | The open-source DRA driver is incompatible with the GKE DRA scheduler | Switch to NGC v25.12.0 |
| NeMo image cannot be pulled | No permissions for cross-project AR | `crane copy` to this project's AR |
