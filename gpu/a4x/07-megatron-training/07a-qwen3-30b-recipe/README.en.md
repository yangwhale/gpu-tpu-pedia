> 🌐 [中文](README.md) | **English**

# Qwen3 30B-A3B MoE Training on GB200 NVL72 (A4X)

Megatron Bridge + NeMo 26.06 container, Qwen3 30B-A3B MoE pretraining benchmark.

**Result**: 8 GPUs (2 nodes) reach **914 TFLOP/s/GPU**, vs. the official DGX-GB200 figure of 936 (a 2.3% gap). A 100% reproduction of the official recipe, with no recompute.

**Reference links**:
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) — official benchmark data
- [Megatron Bridge Performance Tuning Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html) — performance tuning guide
- [Qwen3 Workload Base Configs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_workload_base_configs.py) — recipe parallelism configuration
- [Qwen3 LLM Pretrain Config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_llm_pretrain.py) — recipe model configuration

## Prerequisites

- 2+ A4X workers in the same NVL72 domain (same Placement Policy)
- k8s 1.34+ cluster + GPU Stack (device-plugin + DRA + DRANET + ComputeDomain)
- Worker image: `chrisya-a4x-worker-v3` (kernel-locked + NVIDIA 580 + Lustre + IMEX)

## Step 1: Deploy the NeMo 26.06 training Pod

One Pod per worker, using the `nvcr.io/nvidia/nemo:26.06` container + GIB v1.1.2 NCCL plugin.

```yaml
containers:
- image: nvcr.io/nvidia/nemo:26.06
  resources:
    limits: {nvidia.com/gpu: 4}
    claims: [{name: cd}]   # ComputeDomain channel
  env:
  - {name: LD_PRELOAD, value: "/usr/local/gib/lib64/libnccl.so.2"}
  - {name: LD_LIBRARY_PATH, value: "/usr/local/gib/lib64:/usr/local/nvidia/lib64"}
  - {name: NCCL_MNNVL_ENABLE, value: "2"}
  - {name: NCCL_CUMEM_ENABLE, value: "1"}
initContainers:
- name: gib-installer
  image: us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2
  args:
  - |
    /scripts/container_entry.sh install --install-nccl
    cp -a /usr/local/gib/. /target/gib/
    cp -a /usr/lib/aarch64-linux-gnu/libibverbs.so* /target/gib/lib64/
    cp -a /usr/lib/aarch64-linux-gnu/libmlx5.so* /target/gib/lib64/
    cp -a /usr/lib/aarch64-linux-gnu/librdmacm.so* /target/gib/lib64/
    mkdir -p /target/gib/lib64/libibverbs
    cp -a /usr/lib/aarch64-linux-gnu/libibverbs/libmlx5-rdmav34.so /target/gib/lib64/libibverbs/ 2>/dev/null || true
resourceClaims:
- {name: cd, resourceClaimTemplateName: cd-chrisya-channel}
```

## Step 2: Environment variables

Megatron Bridge's Slurm launcher (`perf_plugins.py`) sets the following variables automatically. When running directly with torchrun you must set them manually:

```bash
source /usr/local/gib/scripts/set_nccl_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

# CuTeDSL fused grouped MLP
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8

# NVL72 domain configuration (required for hybridep)
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128

# GB200-specific
export NCCL_CTA_POLICY=1

# CUDA Graph memory management
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True

# LayerNorm SM margin
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

> **`TORCH_NCCL_AVOID_RECORD_STREAMS` must be 0**. Setting it to 1 under CUDA Graph mode causes the Graph to have no effect, dropping performance from 914 to 284. Use it together with `graph_capture_record_stream_reuse:True`.

## Step 3: Launch training

Use `run_script.py` (not `run_recipe.py`).

```bash
cd /opt/Megatron-Bridge/scripts/performance

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen \
    -mr qwen3_30b_a3b \
    --task pretrain \
    -g gb200 \
    -c fp8_mx \
    -ng 8 \
    --data mock \
    --max_steps 20 \
    --log_dir /tmp/nemo-results \
    -wde bench \
    -wdj qwen3_30b
```

### Configuration auto-loaded by the recipe

| Config | Value |
|---|---|
| EP | 8 |
| TP / PP | 1 / 1 |
| MBS / GBS | 4 / 512 |
| seq_length | 4096 |
| num_layers | 48 (full model) |
| cuda_graph_impl | full_iteration |
| moe_flex_dispatcher_backend | hybridep |
| moe_a2a_overlap | True |
| cutedsl_fused_grouped_mlp | True |
| fp8_dot_product_attention | True |

## GKE deployment (LeaderWorkerSet)

Deploy on a GKE cluster using LeaderWorkerSet + Kueue + ComputeDomain. Measured **926 TFLOP/s** reproduced on the a4x-baker cluster pool-7.

### Prerequisites

- GKE cluster with the following installed: NCCL RDMA DaemonSet, NVIDIA DRA Driver, LeaderWorkerSet controller, Kueue
- Network objects created: default, gvnic-1, rdma-0~3
- Kueue LocalQueue `tas-lq` pointing to ClusterQueue `tas-cq`

### Key differences vs self-managed K8s

| Dimension | Self-managed K8s | GKE LWS |
|---|---|---|
| IMEX | manual nvidia-imex daemon on the host | ComputeDomain + DRA Driver managed automatically |
| GPU allocation | device-plugin + hostPath | device-plugin + DRA ResourceClaim |
| GIB NCCL | installed via initContainer + LD_PRELOAD | injected automatically into /usr/local/nvidia/lib64 by the NCCL RDMA DaemonSet |
| NCCL version | GIB's bundled NCCL 2.28 takes priority | container's bundled NCCL 2.30 takes priority (LD_LIBRARY_PATH puts /usr/lib/aarch64-linux-gnu first) |
| Scheduling | manual assignment via nodeSelector | Kueue TAS topology-aware scheduling |

### YAML

```yaml
# 1. ComputeDomain (create first; LWS depends on the ResourceClaimTemplate it generates)
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: nemo-30b-domain
spec:
  numNodes: 2
  channel:
    resourceClaimTemplate:
      name: nemo-30b-channel
---
# 2. LeaderWorkerSet
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: nemo-30b
  labels:
    kueue.x-k8s.io/queue-name: tas-lq
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        annotations:
          networking.gke.io/default-interface: eth0
          networking.gke.io/interfaces: |
            [
              {"interfaceName":"eth0","network":"default"},
              {"interfaceName":"eth1","network":"gvnic-1"},
              {"interfaceName":"gpu0rdma0","network":"rdma-0"},
              {"interfaceName":"gpu1rdma0","network":"rdma-1"},
              {"interfaceName":"gpu2rdma0","network":"rdma-2"},
              {"interfaceName":"gpu3rdma0","network":"rdma-3"}
            ]
        labels:
          app: nemo-30b-pod
      spec:
        nodeSelector:
          cloud.google.com/gke-accelerator: nvidia-gb200
          cloud.google.com/gke-gpu: "true"
        resourceClaims:
        - name: compute-domain-channel
          resourceClaimTemplateName: nemo-30b-channel
        tolerations:
        - key: nvidia.com/gpu
          operator: Exists
        - {effect: NoSchedule, key: kubernetes.io/arch, operator: Equal, value: arm64}
        containers:
        - name: nemo
          image: us-central1-docker.pkg.dev/supercomputer-testing/nvcr/nemo:26.06.rc7
          command: ["/bin/bash", "-c", "sleep infinity"]
          resources:
            claims: [{name: compute-domain-channel}]
            limits: {nvidia.com/gpu: "4"}
            requests: {nvidia.com/gpu: "4"}
          securityContext: {privileged: true}
          volumeMounts:
          - {name: dshm, mountPath: /dev/shm}
          env:
          - {name: CUDA_DEVICE_MAX_CONNECTIONS, value: "1"}
          - {name: NVTE_CUTEDSL_FUSED_GROUPED_MLP, value: "1"}
          - {name: CUDNNFE_CLUSTER_OVERLAP_MARGIN, value: "8"}
          - {name: NVLINK_DOMAIN_SIZE, value: "72"}
          - {name: USE_MNNVL, value: "1"}
          - {name: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN, value: "8"}
          - {name: NUM_OF_TOKENS_PER_CHUNK_COMBINE_API, value: "128"}
          - {name: NCCL_CTA_POLICY, value: "1"}
          - {name: TORCH_NCCL_AVOID_RECORD_STREAMS, value: "0"}
          - {name: NCCL_GRAPH_REGISTER, value: "0"}
          - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True,graph_capture_record_stream_reuse:True"}
          - {name: NVTE_FWD_LAYERNORM_SM_MARGIN, value: "16"}
          - {name: NVTE_BWD_LAYERNORM_SM_MARGIN, value: "16"}
          - {name: GLOO_SOCKET_IFNAME, value: eth0}
          - {name: NCCL_SOCKET_IFNAME, value: eth0}
          - {name: NCCL_MNNVL_ENABLE, value: "2"}
          - {name: NCCL_CUMEM_ENABLE, value: "1"}
        volumes:
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 200Gi}
    workerTemplate:
      # Same as leaderTemplate (omitted; see repo for the full YAML)
```

### Launch training

Once the Pods are Running, execute the following on each of the two Pods:

```bash
# Leader (node_rank=0)
kubectl exec nemo-30b-0 -- bash -c "
  cd /opt/Megatron-Bridge/scripts/performance
  export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
  export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
  MASTER_IP=\$(hostname -i)
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=\$MASTER_IP --master_port=29600 \
    run_script.py -m qwen -mr qwen3_30b_a3b --task pretrain \
    -g gb200 -c fp8_mx -ng 8 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_30b
"

# Worker (node_rank=1, using the leader's IP)
kubectl exec nemo-30b-0-1 -- bash -c "
  cd /opt/Megatron-Bridge/scripts/performance
  export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
  export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<LEADER_IP> --master_port=29600 \
    run_script.py -m qwen -mr qwen3_30b_a3b --task pretrain \
    -g gb200 -c fp8_mx -ng 8 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_30b
"
```

### GKE gotchas log

| Problem | Cause | Fix |
|---|---|---|
| nvcr.io 403 Forbidden | NGC API key lacks permission | use the project AR image `us-central1-docker.pkg.dev/.../nvcr/nemo:26.06.rc7` |
| ncclWaitSignal undefined symbol | GIB NCCL 2.28 loaded with priority, conflicting with the container's NCCL 2.30 | put `/usr/lib/aarch64-linux-gnu` first in LD_LIBRARY_PATH |
| MNNVL available but not working | Pod has no IMEX channel (`/dev/nvidia-caps-imex-channels/` does not exist) | a bare Pod won't work; you must go through LWS + ComputeDomain + ResourceClaim |
| cudaIpcOpenMemHandle SIGABRT | same as above; HybridEP CUDA IPC requires IMEX | same as above |
| Gloo IPv6 Network unreachable | Gloo defaults to connecting over IPv6 | add `GLOO_SOCKET_IFNAME=eth0` to force IPv4 |
| ResourceClaimTemplate not found | ComputeDomain not yet created | ComputeDomain must be created before the LWS, and the channel name must match |

### GKE measured results

| Cluster | Node pool | TFLOP/s/GPU | Step Time |
|---|---|---|---|
| a4x-baker (us-central1) | pool-7 | **926** | 6.51s |
| chrisya-a4x-gke-ew4 (europe-west4) | a4x-pool | **925** | 6.52s |

## Performance results

| Metric | A4X (GCP) | DGX-GB200 (official) |
|---|---|---|
| **Model TFLOP/s/GPU** | **914** | **936** |
| Gap | -2.3% | baseline |
| Step Time | 6.60s | — |
| HBM Peak | 184.7 GiB | — |
| Alloc Retries | 0 | — |

## Optimization iteration summary

The key steps from 89 to 914:

| Stage | TFLOP/s | Key action |
|---|---|---|
| 1. Correct entry point | 89 | use `run_script.py`, not `run_recipe.py` |
| 2. + cutedsl | 284 | `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` |
| 3. + full CUDA Graph | **914** | `AVOID_RECORD_STREAMS=0` + `graph_capture_record_stream_reuse:True` + NVL72 env vars |

### Core lessons

1. **`run_recipe.py` vs `run_script.py`**: the former does not load GPU-specific optimization configs (CUDA Graph, hybridep, cutedsl, etc.); only the latter calls `get_perf_optimized_recipe()` for a full load
2. **Slurm environment variables**: `perf_plugins.py` automatically sets ~15 environment variables for the Slurm executor. Running torchrun bypasses Slurm, so all of them are missed. The most critical are `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` and the NVL72 domain variables
3. **CUDA Graph + AVOID_RECORD_STREAMS**: in non-CG mode, setting this to 1 saves memory; in CG mode it must be 0. Set the wrong way, CUDA Graph silently fails (no error, just slow)
4. **What CUDA Graph does for MoE**: Qwen3 30B has 128 experts × 48 layers, tens of thousands of CUDA kernel launches per step. CUDA Graph reduces host overhead from milliseconds to microseconds, cutting step time from 21s to 6.6s (3.2×)
