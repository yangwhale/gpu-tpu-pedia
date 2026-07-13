> 🌐 [中文](README.md) | **English**

# A4X GB200 NVL72 on GKE (Google Kubernetes Engine)

Deploying an A4X GPU cluster the GKE-native way. Re-tested and confirmed reproducible.

**Measured results**:
- Qwen3 30B (8 GPU): **925 TFLOP/s/GPU** (official DGX-GB200: 936, 1.2% gap)
- Qwen3 235B (64 GPU): **686 TFLOP/s/GPU** (PP=2 EP=32 + MNNVL=2, +89% vs default V1)

**Reference docs**: [Create a custom AI-optimized GKE cluster which uses A4X](https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x)

## Prerequisites

- A GCP project with GKE API + Compute API enabled
- An A4X Reservation (`specificReservationRequired: true`, `shareType: LOCAL`)
- Sufficient quota (GPU, CPU, IP)

> **Reservation note**: `shareType: LOCAL` can be consumed directly by GKE in the same project. `shareType: SPECIFIC_PROJECTS` may require additional permission configuration (in practice, us-central1-b returned GCE_STOCKOUT on all 7 rounds).

## Deployment Steps

### Step 1: VPC Networking

A4X has 6 NICs per node and requires 3 types of VPC. The RDMA VPC must be bound to the `vpc-roce` network profile (zone-level, cannot be shared across regions).

```bash
PROJECT=supercomputer-testing
REGION=europe-west4
ZONE=europe-west4-b

# Primary management VPC (reusable across regions)
gcloud compute networks create chrisya-gke-mgmt \
  --subnet-mode=custom --mtu=8244 --project=$PROJECT

gcloud compute networks subnets create chrisya-gke-sub-${REGION} \
  --network=chrisya-gke-mgmt --region=$REGION \
  --range=10.51.0.0/16 \
  --secondary-range=pods=10.64.0.0/14,services=10.68.0.0/20 \
  --project=$PROJECT

# Additional GVNIC (reusable across regions)
gcloud compute networks create chrisya-gke-net-1 \
  --subnet-mode=custom --mtu=8244 --project=$PROJECT

gcloud compute networks subnets create chrisya-gke-sub-1-${REGION} \
  --network=chrisya-gke-net-1 --region=$REGION \
  --range=10.61.0.0/18 --project=$PROJECT

# RDMA VPC (independent per region, bound to the vpc-roce profile)
gcloud compute networks create chrisya-gke-rdma-${REGION} \
  --subnet-mode=custom --mtu=8896 \
  --network-profile=projects/$PROJECT/global/networkProfiles/${ZONE}-vpc-roce \
  --project=$PROJECT

for i in 0 1 2 3; do
  BASE=$((192 + i * 16))
  gcloud compute networks subnets create chrisya-gke-rdma-${REGION}-sub-${i} \
    --network=chrisya-gke-rdma-${REGION} --region=$REGION \
    --range=192.168.${BASE}.0/20 --project=$PROJECT
done

# Firewall
gcloud compute firewall-rules create chrisya-gke-allow-internal \
  --network=chrisya-gke-mgmt --allow=tcp,udp,icmp \
  --source-ranges=10.50.0.0/8 --project=$PROJECT

gcloud compute firewall-rules create chrisya-gke-allow-ssh \
  --network=chrisya-gke-mgmt --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 --project=$PROJECT

for NET in chrisya-gke-net-1 chrisya-gke-rdma-${REGION}; do
  gcloud compute firewall-rules create ${NET}-allow-internal \
    --network=$NET --allow=tcp,udp,icmp \
    --source-ranges=0.0.0.0/0 --project=$PROJECT
done
```

### Step 2: Placement Policy

```bash
gcloud compute resource-policies create workload-policy chrisya-a4x-placement-${REGION} \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=1x72 \
  --region=$REGION --project=$PROJECT
```

> You must use the `workload-policy` type. `group-placement` returns GCE_STOCKOUT even with `--gpu-topology=1x72` added.

### Step 3: Create the GKE Cluster

```bash
gcloud container clusters create chrisya-a4x-gke-${REGION} \
  --region=$REGION \
  --network=chrisya-gke-mgmt \
  --subnetwork=chrisya-gke-sub-${REGION} \
  --cluster-secondary-range-name=pods \
  --services-secondary-range-name=services \
  --release-channel=rapid \
  --enable-ip-alias \
  --enable-multi-networking \
  --enable-dataplane-v2 \
  --enable-private-nodes \
  --no-enable-private-endpoint \
  --master-authorized-networks=0.0.0.0/0 \
  --workload-pool=${PROJECT}.svc.id.goog \
  --addons=GcsFuseCsiDriver \
  --num-nodes=1 \
  --machine-type=e2-standard-16 \
  --project=$PROJECT
```

> `--enable-multi-networking` and `--enable-dataplane-v2` cannot be changed after creation. `--workload-pool` is a prerequisite for the GCSFuse CSI driver.

### Step 4: Cloud NAT (allows private nodes to reach the internet to pull images)

```bash
gcloud compute routers create chrisya-gke-router \
  --network=chrisya-gke-mgmt --region=$REGION --project=$PROJECT

gcloud compute routers nats create chrisya-gke-nat \
  --router=chrisya-gke-router --region=$REGION \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges --project=$PROJECT
```

### Step 5: Create the A4X Node Pool

```bash
RESERVATION=nvidia-gb200-jwmrpsfbs8szi  # replace with the actual reservation name

gcloud container node-pools create a4x-pool \
  --cluster=chrisya-a4x-gke-${REGION} \
  --region=$REGION \
  --node-locations=$ZONE \
  --machine-type=a4x-highgpu-4g \
  --accelerator=type=nvidia-gb200,count=4,gpu-driver-version=LATEST \
  --num-nodes=2 \
  --reservation-affinity=specific \
  --reservation=projects/$PROJECT/reservations/$RESERVATION \
  --placement-policy=chrisya-a4x-placement-${REGION} \
  --additional-node-network=network=chrisya-gke-net-1,subnetwork=chrisya-gke-sub-1-${REGION} \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-0 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-1 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-2 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-3 \
  --ephemeral-storage-local-ssd=count=4 \
  --enable-gvnic \
  --project=$PROJECT
```

### Step 6: Install the GPU Stack Components

After the node pool is created, install 3 components to enable MNNVL and RDMA:

```bash
# 6.1 NCCL RDMA DaemonSet
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-rdma-installer-a4x.yaml

# 6.2 GKE Network objects (GVNIC + RDMA mode)
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: GKENetworkParamSet
metadata:
  name: gvnic-1
spec:
  vpc: chrisya-gke-net-1
  vpcSubnet: chrisya-gke-sub-1-${REGION}
  deviceMode: NetDevice
---
apiVersion: networking.gke.io/v1
kind: Network
metadata:
  name: gvnic-1
spec:
  type: Device
  parametersRef:
    group: networking.gke.io
    kind: GKENetworkParamSet
    name: gvnic-1
EOF

for i in 0 1 2 3; do
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: GKENetworkParamSet
metadata:
  name: rdma-${i}
spec:
  vpc: chrisya-gke-rdma-${REGION}
  vpcSubnet: chrisya-gke-rdma-${REGION}-sub-${i}
  deviceMode: RDMA
---
apiVersion: networking.gke.io/v1
kind: Network
metadata:
  name: rdma-${i}
spec:
  type: Device
  parametersRef:
    group: networking.gke.io
    kind: GKENetworkParamSet
    name: rdma-${i}
EOF
done

# 6.3 NVIDIA DRA Driver (ComputeDomain + IMEX)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
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
      values: [system-node-critical, system-cluster-critical]
EOF

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.12.0" \
  --namespace nvidia-dra-driver-gpu \
  -f - <<EOF
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
          - key: nvidia.com/gpu
            operator: DoesNotExist
kubeletPlugin:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: cloud.google.com/gke-accelerator
                operator: In
                values: [nvidia-gb200]
              - key: kubernetes.io/arch
                operator: In
                values: [arm64]
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
```

> **DRA Driver version**: In K8s 1.36 the DRA API is `resource.k8s.io/v1` (GA). Version 25.3.x uses v1beta1 and will error out; you need **25.12.0+**.

### Step 7: Verify

```bash
# ComputeDomain CRD
kubectl api-resources | grep computedomain

# DRA pods
kubectl get pods -n nvidia-dra-driver-gpu

# NCCL RDMA DaemonSet
kubectl get ds -n kube-system nccl-rdma-installer

# GPU
kubectl exec <any-A4X-pod> -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Training Deployment

Training Pods require ComputeDomain + multi-NIC annotations + hostPath mounts for GIB/NVIDIA. See the full YAML at [nemo-gke-v3.yaml](yamls/nemo-gke-v3.yaml).

### Qwen3 30B (8 GPU, 2 nodes)

```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen -mr qwen3_30b_a3b --task pretrain \
    -g gb200 -c fp8_mx -ng 8 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_30b
```

Environment variables:
```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/gib/lib64:$LD_LIBRARY_PATH
source /usr/local/gib/scripts/set_nccl_env.sh
export NCCL_SOCKET_IFNAME=eth0,eth1
export NCCL_MNNVL_ENABLE=2
export NCCL_CUMEM_ENABLE=1
```

### Qwen3 235B (64 GPU, 16 nodes, optimized PP=2 EP=32)

```bash
torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen -mr qwen3_235b_a22b --task pretrain \
    -g gb200 -c fp8_mx -ng 64 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_235b \
    -cv v1 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 32 \
    --global_batch_size 512 \
    --micro_batch_size 1
```

Additional environment variable: `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32`

## Measured Results (re-tested and confirmed)

### Qwen3 30B (8 GPU)

| Metric | GKE | Self-built K8s | DGX-GB200 (official) |
|---|---|---|---|
| TFLOP/s/GPU | **925** | 914 | 936 |
| Step Time | 6.52s | 6.60s | — |
| Gap vs official | -1.2% | -2.3% | baseline |

### Qwen3 235B (64 GPU)

| Metric | V1 default (PP=8 EP=8) | PP=2 EP=32 (MNNVL=0) | PP=2 EP=32 (MNNVL=2) |
|---|---|---|---|
| TFLOP/s/GPU | 360 / 376 peak | 587 / 595 peak | **680 / 686 peak** |
| Step Time | ~27s | 8.2s | **7.1s** |
| NCCL transport | RDMA | RDMA | **NVLink** |
| Improvement | baseline | +63% | **+89%** |

> MNNVL=2 makes NCCL allreduce/PP p2p go over NVLink (900 GB/s) instead of RDMA (200 GB/s), and it does not hang within a single domain. The official V2 (256 GPU) figure of 1092 TFLOP/s requires VPP=3 + full_iteration CG, both of which are unavailable on 64 GPU.

## Full Log of 235B Optimization Iterations (64 GPU, Qwen3 235B-A22B MoE)

### Complete Parameter Comparison Table

| Round | PP | EP | TP | DP | MBS | GBS | CUDA Graph | MNNVL | recompute | TFLOP/s | step time | Status | What changed → why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 8 | 8 | 1 | 8 | 1 | 1024 | TE scoped | 2 | none | **360** | 27s | ✅ | V1 default recipe |
| R1c | 8 | 8 | 1 | 8 | 1 | 1024 | **full_iteration** | 2 | none | crash | — | ❌ | Wanted to enable full CG for speedup → HybridEP fabric memory incompatible with stream capture under PP>1 |
| R3 | 8 | 8 | 1 | 8 | 1 | 1024 | **full_iteration** | 2 | **48 layers** | crash | — | ❌ | Added recompute to free memory for CG → HBM 78, freed space but HybridEP still fails capture (not a memory issue) |
| R4 | 8 | 8 | 1 | 8 | 1 | 1024 | TE scoped | 2 | **24 layers** | crash | — | ❌ | Switched back to TE CG + recompute → assert: recompute only supports full_iteration CG |
| R5 | 8 | 8 | 1 | 8 | 1 | 1024 | TE scoped | 2 | none | crash | — | ❌ | Added NCCL_GRAPH_REGISTER=1 → assert conflict with expandable_segments |
| R7 | **1** | **64** | 1 | 64 | 4 | 512 | **full_iteration** | 2 | none | crash | — | ❌ | PP=1 to make full CG compatible with HybridEP → DOCA QP creation failed (EP=64 across 16 nodes) |
| R8 | **2** | **32** | 1 | 32 | 1 | 512 | **full_iteration** | 2 | none | crash (458 raw) | — | ❌ | PP=2 compromise: reduce bubble + try full CG → PP>1 HybridEP still incompatible, but raw perf 458 (27% higher than PP=8's 360) |
| R9 | **2** | **32** | 1 | 32 | 1 | 512 | TE scoped | **0** | none | **595** | 8.2s | ✅ | Kept PP=2 EP=32 + returned to TE CG → bubble from 30%→1.5%, EP group of 32 GPUs has high communication efficiency, **+63%** |
| **R10** | **2** | **32** | 1 | 32 | 1 | 512 | TE scoped | **2** | none | **686** | **7.1s** | ✅ | MNNVL=0→2: NCCL allreduce switched from RDMA to NVLink → single domain of 64 GPUs does not hang, **+15%** |

### Core Insights Per Round

**R1c-R3 (the full_iteration CG road)**: HybridEP's CUDA fabric memory operations get invalidated during stream capture across pipeline stages. This is not a memory issue (in R3, recompute dropped HBM from 130 down to 78 GiB and it still crashed); rather, it is a fundamental incompatibility between the HybridEP architecture and multi-stream CUDA Graphs. 30B PP=1 works because it is single-stream.

**R4-R5 (small optimizations hit a wall)**: recompute only supports full_iteration CG (a hard assert limit). NCCL_GRAPH_REGISTER=1 conflicts with expandable_segments=True (a hard assert limit). Both paths are dead ends.

**R7-R8 (changing the parallelism strategy)**: Shifted from "optimizing the existing PP=8 config" to "changing the parallelism strategy itself." The DOCA QP failure of PP=1 EP=64 exposed the RDMA limitation of EP spanning multiple nodes. The raw perf of 458 for PP=2 EP=32 proved the direction was correct.

**R9 (the breakthrough)**: PP=2 EP=32 + TE CG = 595. Two changes stacked: pipeline stages 8→2 eliminated the 30% bubble, and EP 8→32 means each card holds only 4 experts (vs 16), improving communication efficiency.

**R10 (NVLink boost)**: MNNVL=0 was a mistake, blindly copied from Xi's cross-domain workaround. Within a single domain, 64 GPUs setting MNNVL=2 is completely safe. NCCL allreduce switched from RDMA (200 GB/s) to NVLink (900 GB/s), +15%.

### Final Optimal Configuration

```
PP=2  EP=32  TP=1  DP=32  MBS=1  GBS=512
CUDA Graph: TE scoped (attn + moe_router + moe_preprocess)
NCCL_MNNVL_ENABLE=2  USE_MNNVL=1  NCCL_NVLS_ENABLE=1
→ 686 TFLOP/s/GPU, step time 7.1s
→ 89% improvement over the V1 default (360)
```

### Key Findings

1. **Parallelism strategy matters more than kernel optimization**: PP 8→2 alone yielded +63%, while CUDA Graph and NCCL tuning combined only added +15%
2. **full_iteration CG is incompatible with HybridEP when PP>1**: the root cause is fabric memory operations invalidating multi-stream capture, not a memory issue
3. **NCCL_MNNVL_ENABLE must be set according to the actual topology**: set 2 for a single domain (NVLink 900GB/s); only set 0 for cross-domain (RDMA fallback)
4. **recompute only supports full_iteration CG**: it is unavailable under TE scoped; this is a hard limit of Megatron Bridge
5. **NCCL_GRAPH_REGISTER=1 is mutually exclusive with expandable_segments**: must be set to 0

## Summary of GKE Setup Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| GPU placement policy must be provided | --placement-type=COMPACT is not enough | Use the workload-policy type |
| GCE_STOCKOUT (group-placement) | groupPlacementPolicy cannot match the reservation | Switch to workload-policy |
| Network can't host RDMA NIC | RDMA VPC missing the vpc-roce profile | --network-profile=...vpc-roce |
| GCE_STOCKOUT (SPECIFIC_PROJECTS) | reservation shareType restriction | Use a reservation with shareType: LOCAL |
| GCSFuse CSI requires Workload Identity | missing --workload-pool | Add it when creating the cluster |
| nvcr.io images cannot be pulled | private nodes have no internet access | Add Cloud NAT |
| DRA Driver CRD not found | 25.3.x uses v1beta1 vs K8s 1.36 v1 | Upgrade to 25.12.0 |
| GIB NCCL 2.28 vs NeMo 26.06 NCCL 2.30 | ncclWaitSignal symbol missing | Use the container's bundled NCCL; GIB only provides transport |

## Cleanup

```bash
# Delete the node pool (keep the cluster and VPC for next time)
gcloud container node-pools delete a4x-pool \
  --cluster=chrisya-a4x-gke-${REGION} --region=$REGION \
  --project=$PROJECT --quiet

# Full cleanup
gcloud container clusters delete chrisya-a4x-gke-${REGION} \
  --region=$REGION --project=$PROJECT --quiet
```
