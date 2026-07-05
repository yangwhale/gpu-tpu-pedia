# Qwen3 235B-A22B MoE Training on GB200 NVL72 (A4X)

Megatron Bridge + NeMo 26.06 容器，Qwen3 235B-A22B MoE 预训练 benchmark。64 GPU（16 节点）。

**官方参考**：DGX-GB200 256 GPU MXFP8 → 7376 tok/s/GPU, **1092 TFLOP/s/GPU**（V2 config）。本文使用 64 GPU V1 config。

**参考链接**：
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) — 官方 benchmark 数据
- [Megatron Bridge Performance Tuning Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html) — 性能调优指南
- [Qwen3 Workload Base Configs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_workload_base_configs.py) — Recipe 并行度配置
- [Qwen3 LLM Pretrain Config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_llm_pretrain.py) — Recipe 模型配置
- [Performance Scripts README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/README.md) — 启动脚本用法

## 模型规格

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-235B-A22B |
| 总参数 | 235B |
| 每 token 激活参数 | 22B |
| Expert 数量 | 128 routed |
| TopK | 8 |
| 层数 | 94 |

## 前提条件

- 16 台 A4X worker（64 GPU），同一 NVL72 域（同 Placement Policy）
- k8s 1.34+ 集群 + GPU Stack（device-plugin + DRA + DRANET + ComputeDomain）
- Worker 镜像：`chrisya-a4x-worker-v3`

## 与 30B 的关键差异

| 维度 | 30B (07a) | 235B (本文) |
|---|---|---|
| GPU 数 | 8 (2 节点) | 64 (16 节点) |
| PP | 1 | **8** |
| EP | 8 | 8 |
| TP | 1 | 1 |
| MBS | 4 | 1 (默认) |
| GBS | 512 | 1024 |
| CUDA Graph | full_iteration | **transformer_engine** |
| cutedsl | Yes | No (V1 config) |

> **为什么 PP=8**：235B 模型 94 层 + 128 expert，单卡放不下。PP=8 把模型切成 8 个 pipeline stage，每个 stage ~12 层。
>
> **为什么 CUDA Graph 降级**：V1 用 TE CUDA graph（只 capture dense 部分），因为 PP>1 时 full_iteration CUDA graph 内存开销超过 10 GiB（官方文档原文）。V2（256 GPU）才用 full_iteration。

## Step 1: 创建 16 节点集群

```bash
# 用 v3 镜像创建 16 台 worker（同 Placement Policy 保证同域）
for i in $(seq 0 15); do
  gcloud compute instances create chrisya-a4x-d2-w${i} \
    --project=$PROJECT --zone=$ZONE \
    --machine-type=a4x-highgpu-4g \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific --reservation=$RESERVATION \
    --maintenance-policy=TERMINATE \
    --resource-policies=$PLACEMENT_POLICY \
    --image=chrisya-a4x-worker-v3 \
    --image-project=$PROJECT \
    --boot-disk-size=500GB --boot-disk-type=hyperdisk-balanced \
    --network-interface=nic-type=GVNIC,network=$GVNIC_NET,subnet=$GVNIC_SUB \
    --network-interface=nic-type=GVNIC,network=$GVNIC_NET_1,subnet=$GVNIC_SUB_1,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_0,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_1,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_2,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_3,no-address \
    --metadata="ssh-keys=$USER:$(cat ~/.ssh/google_compute_engine.pub)" \
    --scopes=cloud-platform &
done
wait
```

v3 镜像启动即用（~6.5 分钟），无需切内核。启动后 kubeadm join + GPU label + ComputeDomain label。

## Step 2: 部署 NeMo 26.06 训练 Pod

每个 worker 一个 Pod，共 16 个。YAML 结构同 30B recipe（见 [07a](../07a-qwen3-30b-recipe/)），改 nodeSelector 和 pod name。

## Step 3: 环境变量

与 30B 完全相同：

```bash
source /usr/local/gib/scripts/set_nccl_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NCCL_CTA_POLICY=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

## Step 4: 启动训练

```bash
cd /opt/Megatron-Bridge/scripts/performance

torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen \
    -mr qwen3_235b_a22b \
    --task pretrain \
    -g gb200 \
    -c fp8_mx \
    -ng 64 \
    --data mock \
    --max_steps 20 \
    --log_dir /tmp/nemo-results \
    -wde bench \
    -wdj qwen3_235b
```

### Recipe 自动加载的配置（V1, 64 GPU）

| 配置 | 值 | 说明 |
|---|---|---|
| PP | 8 | 94 层切 8 段 |
| EP | 8 | Expert Parallelism |
| TP | 1 | 无 Tensor 并行 |
| ETP | 1 | 无 Expert Tensor 并行 |
| GBS | 1024 | Global Batch Size |
| seq_length | 4096 | 序列长度 |
| cuda_graph_impl | transformer_engine | TE scoped CUDA Graph |
| cuda_graph_scope | attn, moe_router, moe_preprocess | Dense 模块 capture |
| moe_flex_dispatcher_backend | hybridep | NVL72 优化 |

### 并行度计算

- 总 GPU: 64
- PP=8: 模型切 8 段
- 每 PP stage 内 GPU 数: 64/8 = 8
- EP=8: 8 GPU 做 expert 分发
- DP: 64/(TP×PP) = 64/8 = 8, EP=8 → DP_effective=1
- GA: GBS/(MBS×DP_effective×PP_num_micro_batches) → 由 recipe 计算

## 性能参考（Megatron Bridge 官方）

### 官方 Benchmark（NeMo 26.06 Container）

| Config | GPU 数 | 精度 | GBS | MBS | TP | PP | CP | VP | EP | tok/s/GPU | TFLOP/s/GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V2 (官方发布) | 256 | MXFP8 | 8192 | 1 | 1 | 8 | 1 | 3 | 32 | 7376 | 1092 |
| V1 (64 GPU) | 64 | MXFP8 | 1024 | — | 1 | 8 | 1 | — | 8 | — | — |

> V2 使用 256 GPU，开了 VPP=3 + full_iteration CUDA Graph + cutedsl。V1（64 GPU）官方未发布性能数据。

### A4X 实测结果

| Config | GPU 数 | 拓扑 | PP | EP | MNNVL | TFLOP/s/GPU | Step Time | 备注 |
|---|---|---|---|---|---|---|---|---|
| V1 baseline | 64 | 单域 | 8 | 8 | 2 | 360 | 27s | 默认 recipe |
| PP=2 EP=32 | 64 | 单域 | 2 | 32 | 0 | 595 | 8.2s | RDMA only |
| PP=2 EP=32 | 64 | 单域 | 2 | 32 | 2 | **686** | 7.1s | NVLink 最优 |
| **PP=2 EP=32 跨域** | **64** | **双域 (8+8节点)** | **2** | **32** | **0** | **685** | **7.1s** | **baker pool-5+pool-7, USE_MNNVL=1** |

> **跨域结果惊喜**：MNNVL=0 + USE_MNNVL=1（奚老师方案）跨两个 NVL72 域跑出 685，几乎等于单域 MNNVL=2 的 686。原因：PP=2 跨域 p2p 通信量小（只传 activation），RDMA 200GB/s 不是瓶颈；EP=32 all-to-all 全在域内走 HybridEP NVLink，不受跨域影响。
>
> **NVLS_ENABLE=1 不可用**：235B 模型参数大，NVLS multicast buffer 分配 OOM。必须设 `NCCL_NVLS_ENABLE=0`。

## GKE 部署方式（LeaderWorkerSet）

在 GKE 集群上使用 LeaderWorkerSet + Kueue + ComputeDomain 部署 64 GPU 训练。

### YAML

```yaml
# 1. ComputeDomain（16 节点）
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: nemo-235b-domain
spec:
  numNodes: 16
  channel:
    resourceClaimTemplate:
      name: nemo-235b-channel
---
# 2. LeaderWorkerSet（size=16 = 1 leader + 15 workers）
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: nemo-235b
  labels:
    kueue.x-k8s.io/queue-name: tas-lq
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 16
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
          app: nemo-235b-pod
      spec:
        nodeSelector:
          cloud.google.com/gke-accelerator: nvidia-gb200
          cloud.google.com/gke-gpu: "true"
        resourceClaims:
        - name: compute-domain-channel
          resourceClaimTemplateName: nemo-235b-channel
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
          - {name: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN, value: "32"}
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
      # 与 leaderTemplate 相同（省略）
```

### 关键差异 vs 30B LWS

| 维度 | 30B LWS | 235B LWS |
|---|---|---|
| size | 2 | 16 |
| ComputeDomain numNodes | 2 | 16 |
| NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN | 8 | 32 |
| nodeSelector 节点池 | 按需指定 | 需 16 台空闲节点的池，或不指定让 Kueue TAS 自动选 |

### 启动训练（PP=2 EP=32 优化版）

```bash
# 所有 16 个 Pod 上分别执行（用脚本批量）
for i in $(seq 0 15); do
  POD="nemo-235b-0"
  [ $i -gt 0 ] && POD="nemo-235b-0-${i}"
  kubectl exec $POD -- bash -c "
    cd /opt/Megatron-Bridge/scripts/performance
    export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
    export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
    nohup torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$i \
      --master_addr=<LEADER_IP> --master_port=29600 \
      run_script.py -m qwen -mr qwen3_235b_a22b --task pretrain \
      -g gb200 -c fp8_mx -ng 64 --data mock \
      --max_steps 20 --log_dir /tmp/nemo-results \
      -wde bench -wdj qwen3_235b \
      -cv v1 \
      --pipeline_model_parallel_size 2 \
      --expert_model_parallel_size 32 \
      --global_batch_size 512 \
      --micro_batch_size 1 > /tmp/train.log 2>&1 &
  " &
done
wait
```

### 跨域部署（实测方案, baker pool-5 + pool-7）

跨两个 NVL72 域时，用两个 LWS + 两个 ComputeDomain，分别锁定不同的节点池：

```yaml
# Domain A (pool-5, PP stage 0, node_rank 0-7)
ComputeDomain: nemo-235b-domain-a (numNodes=8, channel=nemo-235b-channel-a)
LWS: nemo-235b-a (size=8, nodePool=pool-5)

# Domain B (pool-7, PP stage 1, node_rank 8-15)
ComputeDomain: nemo-235b-domain-b (numNodes=8, channel=nemo-235b-channel-b)
LWS: nemo-235b-b (size=8, nodePool=pool-7)
```

关键配置差异 vs 单域：
- `NCCL_MNNVL_ENABLE=0` — NCCL 走 RDMA 避免跨域 hang
- `USE_MNNVL=1` — HybridEP 域内 NVLink fabric
- `NCCL_NVLS_ENABLE=0` — 235B 模型太大，NVLS multicast buffer OOM

### GKE 踩坑记录

| 问题 | 原因 | 修复 |
|---|---|---|
| NVLS multicast OOM | 235B 参数量大，multicast buffer 分配超出 HBM | `NCCL_NVLS_ENABLE=0` |
| master_addr 为空 | shell 变量在 kubectl exec 单引号内不展开 | 直接写 IP，不用 `$VAR` |
| MNNVL available but not working | 裸 Pod 无 IMEX channel | 必须用 LWS + ComputeDomain + ResourceClaim |
| ncclWaitSignal undefined | GIB NCCL 2.28 vs 容器 NCCL 2.30 | LD_LIBRARY_PATH 把容器路径放前面 |
| Gloo IPv6 unreachable | Gloo 默认 IPv6 | `GLOO_SOCKET_IFNAME=eth0` |
| nvcr.io 403 | NGC API key 无权限 | 用项目 AR 镜像 |

## 注意事项

1. **16 节点必须在同一 NVL72 域**：用同一 Placement Policy 创建。A4X NVL72 域最多 18 节点，16 节点在范围内
2. **PP bubble**：PP=8 有 pipeline bubble 开销。V1 config 没有开 VPP（Virtual Pipeline Parallelism）。V2 开了 VP=3 减少 bubble
3. **内存**：235B 比 30B 大 8 倍，PP=8 分担后每卡 ~30B 参数。V1 用 TE CUDA graph（内存友好），不需要 recompute
4. **CUDA Graph 模式差异**：V1 用 `transformer_engine`（轻量），V2 用 `full_iteration`（需要更多内存但性能更好）
5. **config_variant**：`run_script.py` 默认加载 V2 config。如果 64 GPU 跑，需确认是否自动降级到 V1。可能需要 `--config_variant v1` 或 `-cv v1`
