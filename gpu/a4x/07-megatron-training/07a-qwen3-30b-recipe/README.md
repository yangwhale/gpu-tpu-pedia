# Qwen3 30B-A3B MoE Training on GB200 NVL72 (A4X)

Megatron Bridge + NeMo 26.06 容器，Qwen3 30B-A3B MoE 预训练 benchmark。

**结果**：8 GPU (2 节点) 达到 **914 TFLOP/s/GPU**，官方 DGX-GB200 为 936（差 2.3%）。100% 复刻官方 recipe，无 recompute。

**参考链接**：
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) — 官方 benchmark 数据
- [Megatron Bridge Performance Tuning Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html) — 性能调优指南
- [Qwen3 Workload Base Configs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_workload_base_configs.py) — Recipe 并行度配置
- [Qwen3 LLM Pretrain Config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_llm_pretrain.py) — Recipe 模型配置

## 前提条件

- 2+ 台 A4X worker，同一 NVL72 域（同 Placement Policy）
- k8s 1.34+ 集群 + GPU Stack（device-plugin + DRA + DRANET + ComputeDomain）
- Worker 镜像：`chrisya-a4x-worker-v3`（kernel 锁定 + NVIDIA 580 + Lustre + IMEX）

## Step 1: 部署 NeMo 26.06 训练 Pod

每个 worker 一个 Pod，使用 `nvcr.io/nvidia/nemo:26.06` 容器 + GIB v1.1.2 NCCL 插件。

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

## Step 2: 环境变量

Megatron Bridge 的 Slurm launcher (`perf_plugins.py`) 自动设置以下变量。用 torchrun 直接跑必须手动设：

```bash
source /usr/local/gib/scripts/set_nccl_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

# CuTeDSL fused grouped MLP
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8

# NVL72 域配置（hybridep 必需）
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128

# GB200 特定
export NCCL_CTA_POLICY=1

# CUDA Graph 内存管理
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True

# LayerNorm SM margin
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

> **`TORCH_NCCL_AVOID_RECORD_STREAMS` 必须为 0**。CUDA Graph 模式下设 1 会导致 Graph 不生效，性能从 914 跌到 284。配合 `graph_capture_record_stream_reuse:True` 使用。

## Step 3: 启动训练

使用 `run_script.py`（不是 `run_recipe.py`）。

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

### Recipe 自动加载的配置

| 配置 | 值 |
|---|---|
| EP | 8 |
| TP / PP | 1 / 1 |
| MBS / GBS | 4 / 512 |
| seq_length | 4096 |
| num_layers | 48（完整模型） |
| cuda_graph_impl | full_iteration |
| moe_flex_dispatcher_backend | hybridep |
| moe_a2a_overlap | True |
| cutedsl_fused_grouped_mlp | True |
| fp8_dot_product_attention | True |

## GKE 部署方式（LeaderWorkerSet）

在 GKE 集群上使用 LeaderWorkerSet + Kueue + ComputeDomain 部署。实测在 a4x-baker 集群 pool-7 复现 **926 TFLOP/s**。

### 前提条件

- GKE 集群已安装：NCCL RDMA DaemonSet、NVIDIA DRA Driver、LeaderWorkerSet controller、Kueue
- Network 对象已创建：default、gvnic-1、rdma-0~3
- Kueue LocalQueue `tas-lq` 指向 ClusterQueue `tas-cq`

### 关键差异 vs 自建 K8s

| 维度 | 自建 K8s | GKE LWS |
|---|---|---|
| IMEX | host 上手动 nvidia-imex daemon | ComputeDomain + DRA Driver 自动管理 |
| GPU 分配 | device-plugin + hostPath | device-plugin + DRA ResourceClaim |
| GIB NCCL | initContainer 安装 + LD_PRELOAD | NCCL RDMA DaemonSet 自动注入到 /usr/local/nvidia/lib64 |
| NCCL 版本 | GIB 自带 NCCL 2.28 优先 | 容器自带 NCCL 2.30 优先（LD_LIBRARY_PATH 把 /usr/lib/aarch64-linux-gnu 放前面） |
| 调度 | nodeSelector 手动分配 | Kueue TAS 拓扑感知调度 |

### YAML

```yaml
# 1. ComputeDomain（先创建，LWS 依赖它生成的 ResourceClaimTemplate）
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
      # 与 leaderTemplate 相同（省略，完整 YAML 见仓库）
```

### 启动训练

Pod Running 后，两个 Pod 上分别执行：

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

# Worker (node_rank=1, 用 leader 的 IP)
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

### GKE 踩坑记录

| 问题 | 原因 | 修复 |
|---|---|---|
| nvcr.io 403 Forbidden | NGC API key 无权限 | 用项目 AR 镜像 `us-central1-docker.pkg.dev/.../nvcr/nemo:26.06.rc7` |
| ncclWaitSignal undefined symbol | GIB NCCL 2.28 被优先加载，与容器 NCCL 2.30 冲突 | LD_LIBRARY_PATH 把 `/usr/lib/aarch64-linux-gnu` 放最前 |
| MNNVL available but not working | Pod 没有 IMEX channel (`/dev/nvidia-caps-imex-channels/` 不存在) | 裸 Pod 不行，必须通过 LWS + ComputeDomain + ResourceClaim |
| cudaIpcOpenMemHandle SIGABRT | 同上，HybridEP CUDA IPC 需要 IMEX | 同上 |
| Gloo IPv6 Network unreachable | Gloo 默认用 IPv6 连接 | 加 `GLOO_SOCKET_IFNAME=eth0` 强制 IPv4 |
| ResourceClaimTemplate not found | ComputeDomain 还没创建 | ComputeDomain 必须先于 LWS 创建，channel name 必须对齐 |

### GKE 实测结果

| 集群 | 节点池 | TFLOP/s/GPU | Step Time |
|---|---|---|---|
| a4x-baker (us-central1) | pool-7 | **926** | 6.51s |
| chrisya-a4x-gke-ew4 (europe-west4) | a4x-pool | **925** | 6.52s |

## 性能结果

| 指标 | A4X (GCP) | DGX-GB200 (官方) |
|---|---|---|
| **Model TFLOP/s/GPU** | **914** | **936** |
| 差距 | -2.3% | baseline |
| Step Time | 6.60s | — |
| HBM Peak | 184.7 GiB | — |
| Alloc Retries | 0 | — |

## 优化迭代总结

从 89 到 914 的关键步骤：

| 阶段 | TFLOP/s | 关键操作 |
|---|---|---|
| 1. 正确入口 | 89 | 用 `run_script.py` 不是 `run_recipe.py` |
| 2. + cutedsl | 284 | `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` |
| 3. + full CUDA Graph | **914** | `AVOID_RECORD_STREAMS=0` + `graph_capture_record_stream_reuse:True` + NVL72 env vars |

### 核心教训

1. **`run_recipe.py` vs `run_script.py`**：前者不加载 GPU 特定优化配置（CUDA Graph、hybridep、cutedsl 等），只有后者调 `get_perf_optimized_recipe()` 完整加载
2. **Slurm 环境变量**：`perf_plugins.py` 为 Slurm executor 自动设 ~15 个环境变量。torchrun 跑不经过 Slurm，全部漏掉。最关键的是 `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` 和 NVL72 domain 变量
3. **CUDA Graph + AVOID_RECORD_STREAMS**：这个变量在非 CG 模式设 1 省内存，CG 模式必须设 0。设反了 CUDA Graph 静默失效（不报错，只是慢）
4. **CUDA Graph 对 MoE 的作用**：Qwen3 30B 有 128 expert × 48 层，每步上万个 CUDA kernel launch。CUDA Graph 把 host overhead 从毫秒级降到微秒级，step time 从 21s 降到 6.6s（3.2×）
