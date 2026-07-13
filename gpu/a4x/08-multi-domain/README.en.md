> 🌐 [中文](README.md) | **English**

# 9. Multi-Domain Training Orchestration

When training scale exceeds a single NVL72 domain (72 GPUs), cross-domain scheduling is required. This chapter explains how to orchestrate multi-domain Megatron-LM training based on a **2-domain experimental environment**, and provides guidance for scaling to 4+ domains.

## 9.1 NVL72 Bandwidth Tiers

The communication bandwidth of A4X/GB200 decreases with each topology tier, so **the parallelism strategy must be aligned with the bandwidth tiers**:

| Tier | Scope | Interconnect | Bandwidth | Suitable Parallelism Dimension |
|------|------|----------|------|----------------|
| L1 Intra-node | 4 GPUs in the same node | NVLink 5th gen (NVSwitch) | ~900 GB/s | TP (Tensor Parallelism) |
| L2 Intra-domain, cross-node | 18 nodes in the same domain | MNNVL (cross-node NVLink) | ~840 GB/s | EP (Expert Parallelism) |
| L3 Cross-domain | Different domains | RDMA over CX-7 (4x200Gbps) | ~165 GB/s | DP (Data Parallelism) |

**Performance impact**: In measured MoE model (Qwen3-30B-A3B) training with pure EP=8, intra-domain MNNVL reached **~480 TFLOP/s/GPU**, while cross-domain RDMA dropped to **~282 TFLOP/s/GPU** (a 37.7% decrease).

## 9.2 Parallelism Strategy and Topology Mapping

Core principle: **map the parallelism dimensions with the highest communication volume onto the tiers with the highest bandwidth**.

| Parallelism Dimension | Communication Pattern | Communication Volume | Recommended Mapping Tier |
|----------|----------|--------|-------------|
| TP (Tensor Parallelism) | AllReduce, 2 times per layer | Very high | L1 Intra-node (<=4) |
| EP (Expert Parallelism) | AlltoAll, dispatch/combine per MoE layer | High | L2 Intra-domain (<=72) |
| DP (Data Parallelism) | AllReduce, gradient synchronization | Medium | L3 Cross-domain |
| PP (Pipeline Parallelism) | Point-to-point, per micro-batch | Low | L2 or L3 |

**Megatron-LM does not automatically detect topology**. The comment in `parallel_state.py` explicitly states *"the caller should make sure adjacent ranks are on the same DGX box"*. Therefore, **the caller must ensure that rank ordering aligns with the physical topology through NODE_RANK orchestration**.

### Rank Orchestration Rules

The rank computation formula for `torchrun`: `global_rank = NODE_RANK x nproc_per_node + LOCAL_RANK`. Megatron-LM partitions parallelism groups based on rank contiguity:

- **TP group**: `TP_SIZE` contiguous ranks (with TP=4: rank 0-3 = 1 TP group, corresponding to the 4 GPUs of 1 node)
- **EP group**: on top of the TP group, `EP_SIZE` contiguous ranks form an EP group
- **DP group**: ranks at corresponding positions across different EP groups form a DP group

Therefore, **nodes within the same domain must be assigned contiguous NODE_RANK values**:

```
# Domain 0: NODE_RANK = 0, 1, 2, ..., 17  (72 GPUs, rank 0-71)
# Domain 1: NODE_RANK = 18, 19, ..., 35   (72 GPUs, rank 72-143)
#
# TP=4: rank [0,1,2,3] = the 4 GPUs of node 0 (intra-node NVLink)
# EP=72: the corresponding EP positions of rank [0-71] = all GPUs of domain 0 (intra-domain MNNVL)
# DP=2: rank 0 and rank 72 form a DP pair (cross-domain RDMA)
```

## 9.3 Recommended Parallelism Configurations (MoE Models)

Using Qwen3-30B-A3B (128 experts, topk=8) as an example:

| Scale | Nodes | Total GPUs | TP | EP | DP | Experts per GPU | Cross-domain Communication |
|------|--------|---------|----|----|----|----|----------|
| 1 node | 1 | 4 | 1 | 4 | 1 | 32 | None |
| 1 domain (2 nodes) | 2 | 8 | 1 | 8 | 1 | 16 | None (MNNVL) |
| **2 domain** | **2x1** | **8** | **1** | **4** | **2** | **32** | **DP gradient sync** |
| 2 domain (full) | 2x18 | 144 | 4 | 16 | >=2 | 8 | DP gradient sync |
| 4 domain (full) | 4x18 | 288 | 4 | 32 | >=2 | 4 | DP gradient sync |

**EP value constraint**: `EP_SIZE` must evenly divide `num_experts` (128). Valid values: 1, 2, 4, 8, 16, 32, 64, 128.

## 9.4 Production Scheduling Solution: JobSet + Kueue TAS (Recommended)

JobSet (v0.12+) manages multiple groups of Jobs as a whole, and Kueue's (v0.18+) TAS (Topology Aware Scheduling) ensures that each group of Pods is scheduled onto the same topology domain.

| Feature | Manual Pod (Section 9.5) | JobSet + Kueue TAS (this section) |
|------|-------------------|---------------------------|
| Domain-level scheduling | `nodeSelector` manually specified | Kueue TAS `podset-required-topology` automatic |
| Node discovery | SSH + `kubectl get pod -o jsonpath` | DNS hostname (`enableDNSHostnames`) |
| NODE_RANK | Manual orchestration | `JOB_COMPLETION_INDEX + DOMAIN_OFFSET` |
| SSH key exchange | Must be exchanged manually | Not required (torchrun uses c10d/Gloo) |
| Fault recovery | Manual redeployment | Automatic Job restart |

### Step 1: Define the Topology CRD

```yaml
cat <<'EOF' | kubectl apply -f -
apiVersion: kueue.x-k8s.io/v1beta2
kind: Topology
metadata:
  name: gb200-nvl72-topology
spec:
  levels:
  - nodeLabel: "nvl72-domain"          # NVL72 domain level
  - nodeLabel: "kubernetes.io/hostname" # node level
EOF

# Verify node topology discovery
kubectl get topology gb200-nvl72-topology -o yaml
```

### Step 2: Create the Kueue Queue

```yaml
cat <<'EOF' | kubectl apply -f -
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: gb200-flavor
spec:
  nodeLabels:
    nvl72-domain: domain-1                 # ← Adjust to the actual domain label value
  topologyName: gb200-nvl72-topology
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  namespaceSelector: {}                    # ← Must be set explicitly in v0.18
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gb200-flavor
      resources:
      - name: "cpu"
        nominalQuota: 1000
      - name: "memory"
        nominalQuota: 4000Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 288              # ← Adjust to actual value
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: default-queue
  namespace: default
spec:
  clusterQueue: gpu-cluster-queue
EOF
```

**Kueue v0.18 notes**:

- All Kueue resources use the `kueue.x-k8s.io/v1beta2` API (v1beta1 is deprecated)
- When setting `topologyName` on a `ResourceFlavor`, **you must also provide `nodeLabels`**
- `ClusterQueue` requires `namespaceSelector: {}`
- TAS is **beta and enabled by default** in v0.18; no additional feature gate is needed

### Step 3: Submit the JobSet

One ReplicatedJob per domain; Kueue TAS guarantees that all Pods of the same ReplicatedJob land on the same NVL72 domain.

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: megatron-2domain
  labels:
    kueue.x-k8s.io/queue-name: default-queue
spec:
  network:
    enableDNSHostnames: true
    publishNotReadyAddresses: true
  replicatedJobs:
  - name: domain-0
    replicas: 1
    template:
      spec:
        completionMode: Indexed
        parallelism: 1                             # ← Change to 18 in production
        completions: 1
        template:
          metadata:
            annotations:
              kueue.x-k8s.io/podset-required-topology: "nvl72-domain"
          spec:
            containers:
            - name: pytorch
              image: nvcr.io/nvidia/pytorch:26.05-py3
              resources:
                limits:
                  nvidia.com/gpu: 4
                claims:
                - name: compute-domain-channel
                - name: rdma-nics
              env:
              - name: NNODES
                value: "2"
              - name: NPROC_PER_NODE
                value: "4"
              - name: DOMAIN_OFFSET
                value: "0"
              - name: MASTER_ADDR
                value: "megatron-2domain-domain-0-0-0"
              - name: MASTER_PORT
                value: "29500"
              - name: NCCL_MNNVL_ENABLE
                value: "0"                         # ← Must be disabled for cross-domain training
              - name: NCCL_CUMEM_ENABLE
                value: "0"
              - name: GLOO_SOCKET_IFNAME
                value: "eth0"
              - name: NCCL_SOCKET_IFNAME
                value: "eth0"
              command: ["/bin/bash", "-c"]
              args:
              - |
                source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null
                export LD_LIBRARY_PATH=/usr/local/gib/lib64:$LD_LIBRARY_PATH
                NODE_RANK=$(( DOMAIN_OFFSET + JOB_COMPLETION_INDEX ))
                cd /scratch-data/Megatron-LM
                torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES \
                  --node_rank=$NODE_RANK \
                  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
                  pretrain_gpt.py \
                  --use-mcore-models --transformer-impl transformer_engine \
                  --num-experts 128 --expert-model-parallel-size 4 \
                  --fp8-format hybrid --fp8-recipe delayed \
                  ...  # Remaining parameters are the same as in Chapter 8
            resourceClaims:
            - name: compute-domain-channel
              resourceClaimTemplateName: domain-0-compute-domain-channel
            - name: rdma-nics
              resourceClaimTemplateName: rdma-nics-domain-0
  - name: domain-1
    replicas: 1
    template:
      spec:
        completionMode: Indexed
        parallelism: 1
        completions: 1
        template:
          metadata:
            annotations:
              kueue.x-k8s.io/podset-required-topology: "nvl72-domain"
          spec:
            containers:
            - name: pytorch
              env:
              - name: DOMAIN_OFFSET
                value: "1"                         # ← Use 18 in production
              # Everything else is the same as domain-0
            resourceClaims:
            - name: compute-domain-channel
              resourceClaimTemplateName: domain-1-compute-domain-channel
            - name: rdma-nics
              resourceClaimTemplateName: rdma-nics-domain-1
```

### Kubeflow Trainer

The `TrainJob` CRD of Kubeflow Trainer (v2.2+) integrates Kueue + JobSet and provides a higher-level abstraction (automatically initializing torchrun, assigning ranks, mounting data volumes), which is suitable for teams that have already deployed Kubeflow.

## 9.5 Simple Verification Solution (Manual Pod + nodeSelector)

**Only suitable for small-scale experimental verification**. It requires manually managing SSH keys, Pod IPs, and NODE_RANK. For production environments, use JobSet + Kueue TAS from Section 9.4.

**Key parameter comparison (intra-domain vs cross-domain)**:

| Parameter | Intra-domain (Chapter 8) | Cross-domain (this chapter) | Notes |
|------|----------------|-------------|------|
| `NCCL_MNNVL_ENABLE` | 2 | **0** | Must be disabled for cross-domain |
| `NCCL_CUMEM_ENABLE` | 1 | **0** | cuMem is only used for MNNVL |
| `--expert-model-parallel-size` | 8 (across 2 nodes) | **4** (intra-node only) | EP is confined within the domain |
| DP | 1 | **2** | Computed automatically |
| ComputeDomain | Required | **domain-0 only** | No shared IMEX across domains |

## 9.6 Scaling to 4+ Domains

### NODE_RANK Orchestration (4 domain x 18 nodes = 72 nodes)

```
# Ensure that nodes in the same domain receive contiguous NODE_RANK values
#
# Domain 0: NODE_RANK = 0 .. 17   → rank 0-71   (72 GPUs)
# Domain 1: NODE_RANK = 18 .. 35  → rank 72-143  (72 GPUs)
# Domain 2: NODE_RANK = 36 .. 53  → rank 144-215 (72 GPUs)
# Domain 3: NODE_RANK = 54 .. 71  → rank 216-287 (72 GPUs)
#
# JobSet approach: add ReplicatedJobs (domain-2, domain-3),
# each setting DOMAIN_OFFSET = 36, 54; parallelism/completions = 18
```

### Parallelism Parameter Scaling

| Parameter | 2 domain (2 nodes) | 2 domain (36 nodes) | 4 domain (72 nodes) |
|------|--------------------|--------------------|---------------------|
| `--nnodes` | 2 | 36 | 72 |
| `--tensor-model-parallel-size` | 1 | 4 | 4 |
| `--expert-model-parallel-size` | 4 | 16 | 32 |
| DP (automatic) | 2 | >=2 | >=2 |
| Experts per GPU | 32 | 8 | 4 |
| `NCCL_MNNVL_ENABLE` | 0 | 0 | 0 |

### Parallelism Strategy for Dense Models

Dense models (such as Llama) have no EP dimension. Recommended mapping:

- **TP=4** (intra-node NVLink, high-frequency AllReduce communication)
- **PP=N** (cross-node pipeline, low point-to-point communication volume, can span domains)
- **DP=remaining** (gradient synchronization, can span domains)
- Prioritize placing the adjacent ranks of a PP stage within the same domain

## 9.7 Optimization Pending Verification: Per-Communicator MNNVL (Intra-domain MNNVL + Cross-domain RDMA)

**The content of this section is a research conclusion and has not yet been verified in the experimental environment.**

### Problem

Current cross-domain training must set `NCCL_MNNVL_ENABLE=0`, which forces all communication to be downgraded to RDMA. Intra-domain EP alltoall drops from ~840 GB/s to ~325 GB/s, a performance loss of about 60%.

### Research Findings

Analysis of the NCCL source code shows that MNNVL detection is actually **per-communicator**.

| NCCL Version Change | Impact |
|-------------|------|
| <= 2.26.x | Silently falls back to RDMA when MNNVL probing fails |
| **2.27.3+** | **Hard fail** (`ncclSystemError`) when MNNVL probing fails; no longer silently falls back |
| **2.28.3+** | Added `NCCL_MNNVL_CLIQUE_ID=-2`: automatically assigns cliques by chassis serial number |
| **2.30.3+** | Added `NCCL_MNNVL_CROSS_CLIQUE=1`: allows cross-clique P2P |

GIB v1.1.2 includes **NCCL 2.30.4**, so all of the above new features are available. In theory, `NCCL_MNNVL_ENABLE=2` (auto-detect) should be able to correctly handle mixed-domain scenarios.

### Verification Plan

```bash
# Experiment 1: Basic verification of MNNVL auto-detect
NCCL_MNNVL_ENABLE=2 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NVLS \
mpirun ... all_reduce_perf -b 1G -e 16G

# Experiment 2: Megatron EP=intra-domain + DP=cross-domain
NCCL_MNNVL_ENABLE=2 \
torchrun --nnodes=2 --nproc_per_node=4 \
  pretrain_gpt.py --expert-model-parallel-size 4 ...

# Experiment 3: Manual clique partitioning
NCCL_MNNVL_ENABLE=2 \
NCCL_MNNVL_CLIQUE_ID=-2 \
mpirun ... all_reduce_perf -b 1G -e 16G
```

**Expected benefit**: Intra-domain EP alltoall could improve from ~325 GB/s to ~840 GB/s (+158%), significantly boosting cross-domain training throughput for MoE models.

---

## 10. 64+8 Spare Capacity Management

**Strategy**: Out of 72 GPUs (18 nodes), use 64 GPUs (16 nodes) for pre-training and 8 GPUs (2 nodes) as spare capacity (~11% redundancy). Automatic capacity management is achieved through PriorityClass + Placeholder Pod.

### 10.1 PriorityClass Definition

```bash
kubectl apply -f yamls/spare-capacity-priority.yaml
kubectl get priorityclass
# gpu-training (1000)  gpu-placeholder (-1)
```

### 10.2 Placeholder Pod Deployment

```bash
kubectl apply -f yamls/spare-capacity-placeholder.yaml
kubectl get pods -l app=gpu-placeholder -o wide
```

### 10.3 Training Pod Preemption Test

```bash
kubectl apply -f yamls/spare-capacity-training-job.yaml
kubectl get pods -o wide
# training-preempt-test Running, one gpu-placeholder is evicted
```

### 10.4 Failover (Cordon/Drain/Uncordon)

```bash
# Step 1: Isolate the faulty node
kubectl cordon <NODE_NAME>
kubectl drain <NODE_NAME> --ignore-daemonsets --delete-emptydir-data --force

# Step 2: Verify Pod rescheduling
kubectl get pods -l app=gpu-placeholder -o wide

# Step 3: Restore after the node is repaired
kubectl uncordon <NODE_NAME>
```

---

## 11. Fault Reporting API (Report-Faulty)

### Single-Node Fault Reporting

```bash
gcloud compute instances report-host-as-faulty <INSTANCE_NAME> \
  --async \
  --disruption-schedule=IMMEDIATE \
  --fault-reasons=behavior=UNRECOVERABLE_GPU_ERROR,description="XID 79 detected" \
  --zone=$ZONE --project=$PROJECT
```

| fault-reasons behavior | Meaning | Scenario |
|----------------------|------|------|
| `PERFORMANCE` | GPU performance degradation | No Xid error but training speed drops |
| `UNRECOVERABLE_GPU_ERROR` | Unrecoverable GPU error | Xid error |
| `SILENT_DATA_CORRUPTION` | Silent data corruption | Abnormal training results |
| `CHIP_ERROR` | Chip error | Hardware chip-level fault |

### 11.2 Sub-block Domain-Level Fault Reporting

When domain-level components such as NVSwitch fail, the entire sub-block (rather than a single node) needs to be reported. First view the reserved topology structure (block / sub-block), then report the faulty sub-block.

```bash
# View the topology structure
gcloud alpha compute reservations blocks list $RESERVATION \
  --zone=$ZONE --project=$PROJECT

gcloud alpha compute reservations sub-blocks list $RESERVATION \
  --block-name=<BLOCK_NAME> \
  --zone=$ZONE --project=$PROJECT

# Sub-block fault reporting
gcloud alpha compute reservations sub-blocks report-subblock-as-faulty $RESERVATION \
  --block-name=<BLOCK_NAME> \
  --sub-block-name=<SUBBLOCK_NAME> \
  --disruption-schedule=IMMEDIATE \
  --fault-reasons=behavior=SWITCH_FAILURE,description="NVSwitch failure" \
  --failure-component=NVLINK_SWITCH \
  --zone=$ZONE --project=$PROJECT
```

| failure-component | Meaning | Scenario |
|-------------------|------|------|
| `NVLINK_SWITCH` | NVSwitch failure | Abnormal intra-domain NVLink communication |

### Python SDK

```python
from google.cloud import compute_v1

def report_host_as_faulty(project_id, zone, instance_name, fault_behavior, description):
    client = compute_v1.InstancesClient()
    request_body = compute_v1.InstancesReportHostAsFaultyRequest(
        disruption_schedule="IMMEDIATE",
        fault_reasons=[
            compute_v1.FaultReason(
                behavior=fault_behavior,
                description=description
            )
        ]
    )
    operation = client.report_host_as_faulty(
        project=project_id, zone=zone, instance=instance_name,
        instances_report_host_as_faulty_request_resource=request_body
    )
    print(f"Operation started: {operation.name}")
    return operation
```

---

## 12. GPU Monitoring (DCGM Exporter)

```bash
kubectl apply -f yamls/dcgm-exporter-daemonset.yaml

# Verify
kubectl get pods -n kube-system -l app=dcgm-exporter

# Test metrics
kubectl exec -n kube-system <DCGM_POD> -- wget -qO- http://localhost:9400/metrics | grep '^DCGM_FI' | head -20
```

### Key Monitoring Metrics

| Metric | Alert Threshold | Meaning |
|------|----------|------|
| `DCGM_FI_DEV_GPU_TEMP` | > 85 C | GPU overheating |
| `DCGM_FI_DEV_MEMORY_TEMP` | > 95 C | Memory overheating |
| `DCGM_FI_DEV_POWER_USAGE` | > 1000W | Abnormal power consumption |
| `DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS` | > 0 | Uncorrectable ECC error |
| `DCGM_FI_DEV_ROW_REMAP_FAILURE` | > 0 | Row remapping failure |
| `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | Continuously increasing | PCIe link quality issue |

**Prometheus + Grafana integration**: DCGM Exporter exposes `:9400/metrics` by default. Use the official NVIDIA Grafana Dashboard ID: 12239.

---

## Benchmark Standards (Appendix C)

### GEMM Performance

| Precision | GB200 Official Data (TFLOP/s) | GCP A4X Measured (TFLOP/s) |
|------|-------------------------|----------------------|
| FP4 | 6507 | 6845 |
| FP8 | 2805 | 3063 |
| FP16 | 1372 | 1492 |
| BF16 | 1471 | 1592 |
| TF32 | 675 | 733 |
| FP32 | 75 | 75 |

### NCCL Performance (Packet size = 16GB, busbw GB/s)

| Configuration | AllReduce | All2All | AllGather | ReduceScatter |
|------|-----------|---------|-----------|---------------|
| 1 domain (72 GPUs) | 845.01 | 660.61 | 687.94 | 707.22 |
| 2 domain (144 GPUs) | 749.56 | 65.47 | 749.56 | 689.05 |
| 4 domain (288 GPUs) | TBD | | | |

**Note on 2-domain All2All bandwidth**: Cross-domain All2All is only 65.47 GB/s, far below the intra-domain 660 GB/s. AlltoAll is the most sensitive to cross-domain bandwidth because each GPU needs to exchange data with all other GPUs.

### Training Performance

| Model | seq_len | GPUs | mbs | gbs | Parallelism Strategy | TFLOP/s/gpu | MFU |
|------|---------|--------|-----|-----|----------|-------------|-----|
| LLaMA2-7B | 4096 | 4 | 4 | 256 | tp1pp1 | 1,127 | 45.08% |
| LLaMA2-7B | 4096 | 8 | 4 | 256 | tp1pp1 | 1,086 | 43.44% |
| LLaMA2-13B | 4096 | 4 | 1 | 256 | tp1pp1 | 1,065 | 42.60% |
| LLaMA2-13B | 4096 | 8 | 2 | 256 | tp1pp1 | 1,065 | 42.60% |
| LLaMA3-70B | 4096 | 64 | 2 | 1024 | tp4pp2 | 971 | 38.83% |

**Environment**: TransformerEngine v2.15 / Megatron-LM core_r0.16.0 / pytorch:26.05-py3

---

## To Be Completed

The following are important production topics not yet covered in the source document; they need to be added later.

### Security Hardening Guide

<!-- TODO -->

Cluster security configuration, including Pod Security Standards (Restricted profile), NetworkPolicy network isolation rules, and container image signature verification (cosign / Binary Authorization).

### Prometheus GPU Health Alerting Rules

<!-- TODO -->

Prometheus AlertManager alerting rule definitions based on DCGM Exporter metrics, covering alert thresholds and notification policies for key scenarios such as GPU temperature, ECC errors, PCIe retransmissions, and row remapping failures.

### Training Checkpoint / Fault Recovery Strategy

<!-- TODO -->

Fault-tolerant checkpoint strategy for training jobs, including asynchronous checkpoint write configuration, automatic post-fault recovery workflow (resume from the most recent checkpoint + skip faulty nodes), and distributed checkpoint consistency guarantees for multi-domain training.

### Automated GPU Health Pipeline

<!-- TODO -->

An end-to-end automated pipeline from XID error detection to automatic node isolation: XID error detection (dmesg / DCGM) → node cordon/drain → automatic invocation of the report-host-as-faulty API → spare node replacement → automatic training job recovery.
