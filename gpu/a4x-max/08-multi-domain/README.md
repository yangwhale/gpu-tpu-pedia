# 9. 多 Domain 训练编排

以 GB200 (A4X) 跨域实测数据为 baseline，规划 GB300 (A4X MAX) 的多 domain (跨 subblock) 训练。GB300 使用 CX-8 GPUDirect + 8-way rail (3,200 Gbps)，预期跨域通信性能显著优于 GB200 的 CX-7 (2,000 Gbps)。

## 9.1 NVL72 带宽层级 (GB300 vs GB200)

NVL72 拓扑结构不变 (18 节点/域)，但 L3 跨域带宽因 CX-8 GPUDirect 升级:

| 层级 | 范围 | 互联技术 | GB200 带宽 | GB300 带宽 | 提升 | 适合的并行维度 |
|------|------|----------|-----------|-----------|------|----------------|
| L1 节点内 | 同节点 4 GPU | NVLink 5th gen (NVSwitch) | ~900 GB/s | ~900 GB/s | 不变 | TP（张量并行） |
| L2 域内跨节点 | 同 domain 18 节点 | MNNVL（跨节点 NVLink） | ~840 GB/s | ~840 GB/s | 不变 | EP（专家并行） |
| L3 跨域 | 不同 domain | RDMA over CX-7/CX-8 | ~165 GB/s | **待测** | **预期 +60%** | DP（数据并行） |

**GB300 L3 预期提升来源**:
- CX-8 GPUDirect: 数据不经 CPU，减少 1 个 PCIe hop + CPU 内存拷贝延迟
- 8-way rail: 聚合带宽从 2,000 Gbps → 3,200 Gbps (+60%)
- PF vs VF: 无虚拟化开销

## 9.2 GB300 Block 拓扑结构

| 维度 | GB200 | GB300 |
|------|-------|-------|
| NVL72 domain | 18 节点, 72 GPU | 18 节点, 72 GPU (不变) |
| Block 结构 | — | **25 subblocks** = 450 VM = 1,800 GPU |
| 可用 Placement Policies | — | **12 个** (gb300-central-nvl72-policy-0001~0012) |
| Reservation 粒度 | reservation 级 | **subblock 级** (更细粒度) |

### Reservation 指定方式

```bash
# GB200: reservation 级
--reservation=RESERVATION_NAME

# GB300: 可精确到 subblock
--reservation=RESERVATION_NAME/reservationBlocks/BLOCK/reservationSubBlocks/SUBBLOCK
```

Subblock 级 reservation 支持:
- 精确指定跨域训练使用哪些 subblock
- 避免跨 block 调度导致物理距离过远
- 与 placement policy 配合实现拓扑感知调度

## 9.3 并行策略与拓扑映射

核心原则不变: **通信量大的并行维度映射到带宽高的层级**。

| 并行维度 | 通信模式 | 通信量 | 推荐映射层级 |
|----------|----------|--------|-------------|
| TP（张量并行） | AllReduce，每层 2 次 | 极高 | L1 节点内 (<=4) |
| EP（专家并行） | AlltoAll，MoE 层每次 dispatch/combine | 高 | L2 域内 (<=72) |
| DP（数据并行） | AllReduce，梯度同步 | 中 | L3 跨域 |
| PP（流水线并行） | 点对点，逐 micro-batch | 低 | L2 或 L3 |

Rank 编排规则同 GB200，`torchrun` 公式: `global_rank = NODE_RANK x nproc_per_node + LOCAL_RANK`。同一 domain 内的节点必须获得连续的 NODE_RANK。

```
# 2 Domain 示例:
# Domain 0 (subblock A): NODE_RANK = 0, 1, 2, ..., 17  （72 GPU, rank 0-71）
# Domain 1 (subblock B): NODE_RANK = 18, 19, ..., 35   （72 GPU, rank 72-143）
#
# TP=4: rank [0,1,2,3] = node 0 的 4 GPU (同节点 NVLink)
# EP=72: rank [0-71] 的对应 EP 位置 = domain 0 全部 GPU (域内 MNNVL)
# DP=2: rank 0 和 rank 72 组成 DP pair (跨域 CX-8 GPUDirect RDMA)
```

## 9.4 NCCL 环境变量 (GB300)

跨域训练关键参数与 GB200 一致:

| 参数 | 同域（域内） | 跨域（本章） | 说明 |
|------|-------------|-------------|------|
| `NCCL_MNNVL_ENABLE` | 2 | **0** | 跨域必须关闭 MNNVL |
| `NCCL_CUMEM_ENABLE` | 1 | **0** | cuMem 仅用于 MNNVL |
| `GLOO_SOCKET_IFNAME` | eth0 | eth0 | Gloo 通信接口 |
| `NCCL_SOCKET_IFNAME` | eth0 | eth0 | NCCL 控制通信接口 |

> **注意**: GB300 使用 IPv6-only 网络栈。NCCL/Gloo 在 IPv6 环境下的行为需拿到 VM 后验证。

## 9.5 GB200 Baseline 跨域结果

### NCCL 跨域性能 (GB200 实测)

| 配置 | AllReduce (GB/s) | All2All (GB/s) | AllGather (GB/s) | ReduceScatter (GB/s) |
|------|-----------|---------|-----------|---------------|
| 1 domain (72 GPU) | 845.01 | 660.61 | 687.94 | 707.22 |
| 2 domain (144 GPU) | 749.56 | 65.47 | 749.56 | 689.05 |
| 跨域 vs 同域 | -11.3% | **-90.1%** | +8.9% | -2.6% |

**关键发现**:
- AllReduce 跨域仅下降 11.3%（NCCL ring/tree 算法自适应）
- **All2All 跨域暴跌 90.1%**（65.47 vs 660.61 GB/s），对跨域带宽敏感度最高
- AllGather 跨域反而略升（NCCL 分层算法优化）

### 训练跨域性能 (GB200 实测)

MoE 模型（Qwen3-30B-A3B）纯 EP=8 训练:

| 配置 | TFLOP/s/GPU | 相对性能 |
|------|------------|---------|
| 同域 MNNVL | ~480 | 100% |
| 跨域 RDMA | ~282 | **59.2% (-40.8%)** |

### 训练推荐配置 (GB200, MoE)

| 规模 | 节点数 | GPU 总数 | TP | EP | DP | 每 GPU 专家数 | 跨域通信 |
|------|--------|---------|----|----|----|----|----------|
| 1 domain (2 nodes) | 2 | 8 | 1 | 8 | 1 | 16 | 无（MNNVL） |
| **2 domain** | **2x1** | **8** | **1** | **4** | **2** | **32** | **DP 梯度同步** |
| 2 domain (满配) | 2x18 | 144 | 4 | 16 | >=2 | 8 | DP 梯度同步 |
| 4 domain (满配) | 4x18 | 288 | 4 | 32 | >=2 | 4 | DP 梯度同步 |

## 9.6 生产调度方案：JobSet + Kueue TAS

与 GB200 方案一致，每个 domain 一个 ReplicatedJob，Kueue TAS 保证同一 ReplicatedJob 的所有 Pod 落在同一 NVL72 domain。

### 步骤 1: 定义 Topology CRD

```yaml
cat <<'EOF' | kubectl apply -f -
apiVersion: kueue.x-k8s.io/v1beta2
kind: Topology
metadata:
  name: gb300-nvl72-topology
spec:
  levels:
  - nodeLabel: "nvl72-domain"          # NVL72 domain 级别
  - nodeLabel: "kubernetes.io/hostname" # 节点级别
EOF

# 验证节点拓扑发现
kubectl get topology gb300-nvl72-topology -o yaml
```

### 步骤 2: 创建 Kueue 队列

```yaml
cat <<'EOF' | kubectl apply -f -
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: gb300-flavor
spec:
  nodeLabels:
    nvl72-domain: domain-1
  topologyName: gb300-nvl72-topology
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  namespaceSelector: {}
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gb300-flavor
      resources:
      - name: "cpu"
        nominalQuota: 1000
      - name: "memory"
        nominalQuota: 4000Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 288
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

### 步骤 3: 提交 JobSet (2 Domain)

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: megatron-2domain-gb300
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
        parallelism: 1                             # 生产改为 18
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
                value: "megatron-2domain-gb300-domain-0-0-0"
              - name: MASTER_PORT
                value: "29500"
              - name: NCCL_MNNVL_ENABLE
                value: "0"                         # 跨域训练必须关闭
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
                  ...  # 其余参数同第 8 章
            resourceClaims:
            - name: compute-domain-channel
              resourceClaimTemplateName: domain-0-compute-domain-channel
            - name: rdma-nics
              resourceClaimTemplateName: rdma-nics-domain-0      # count=8 (GB300)
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
                value: "1"                         # 生产用 18
              # 其余同 domain-0
            resourceClaims:
            - name: compute-domain-channel
              resourceClaimTemplateName: domain-1-compute-domain-channel
            - name: rdma-nics
              resourceClaimTemplateName: rdma-nics-domain-1      # count=8 (GB300)
```

**GB300 vs GB200 JobSet 差异**:

| 维度 | GB200 | GB300 |
|------|-------|-------|
| ResourceClaimTemplate RDMA count | 4 | **8** |
| 网络栈 | IPv4 | **IPv6-only** (需验证 MASTER_ADDR 格式) |
| RDMA 子网 | 4 个独立子网 | 1 个共享子网 |

## 9.7 GB300 实测结果

### NCCL 跨域性能

<!-- 拿到 GB300 VM 后填入实测数据 -->

| 配置 | AllReduce (GB/s) | All2All (GB/s) | AllGather (GB/s) | ReduceScatter (GB/s) |
|------|-----------|---------|-----------|---------------|
| 1 domain (72 GPU) | — | — | — | — |
| 2 domain (144 GPU) | — | — | — | — |
| 4 domain (288 GPU) | — | — | — | — |

**GB200 baseline 对比**:

| 指标 | GB200 2-domain | GB300 2-domain | 提升 |
|------|---------------|---------------|------|
| AllReduce | 749.56 GB/s | — | — |
| All2All | 65.47 GB/s | — | — |
| AllGather | 749.56 GB/s | — | — |
| ReduceScatter | 689.05 GB/s | — | — |

**预期**: CX-8 GPUDirect + 8-way rail 应使跨域带宽提升 ~60%。All2All 对带宽最敏感，预期从 65.47 → ~105 GB/s。

### 训练跨域性能

<!-- 拿到 GB300 VM 后填入实测数据 -->

| 模型 | 规模 | 并行策略 | GB200 TFLOP/s/GPU | GB300 TFLOP/s/GPU | 提升 |
|------|------|---------|------------------|------------------|------|
| Qwen3-30B-A3B (EP=4 DP=2) | 2x1 node (8 GPU) | 跨域 | ~282 | — | — |
| Qwen3-30B-A3B (EP=8 DP=1) | 2x1 node (8 GPU) | 同域 | ~480 | — | — |
| DSv3 16L | 2x18 nodes (144 GPU) | 跨域 | 待测 | — | — |

## 9.8 测试计划

### Phase 1: NCCL All-Reduce 跨域验证

```bash
# 环境
# 集群: chrisya-gb300-gke
# 项目: tencent-gcp-taiji-poc
# Zone: us-central1-b
# Placement policies: gb300-central-nvl72-policy-0001~0012

# Step 1: 同域 NCCL baseline (单 subblock)
# 使用 1 个 placement policy 分配 18 节点 (72 GPU)
NCCL_MNNVL_ENABLE=2 \
mpirun --np 72 all_reduce_perf -b 1G -e 16G

# Step 2: 跨域 NCCL (2 subblock)
# 使用 2 个 placement policy，确保 2 个 domain
NCCL_MNNVL_ENABLE=0 \
mpirun --np 144 all_reduce_perf -b 1G -e 16G

# Step 3: 跨域 All2All (最敏感指标)
NCCL_MNNVL_ENABLE=0 \
mpirun --np 144 all2all_perf -b 1G -e 16G
```

### Phase 2: 跨域训练验证

```bash
# Step 1: MoE 跨域训练 (2 domain x 1 node = 8 GPU)
# EP=4 (节点内) + DP=2 (跨域)
NCCL_MNNVL_ENABLE=0 \
torchrun --nnodes=2 --nproc_per_node=4 \
  pretrain_gpt.py \
  --num-experts 128 --expert-model-parallel-size 4 \
  ...

# Step 2: 满配跨域训练 (2 domain x 18 nodes = 144 GPU)
# TP=4 EP=16 DP=2
NCCL_MNNVL_ENABLE=0 \
torchrun --nnodes=36 --nproc_per_node=4 \
  pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --num-experts 128 --expert-model-parallel-size 16 \
  ...
```

### Phase 3: Per-Communicator MNNVL 验证

与 GB200 相同的验证计划。NCCL 2.30.4 (GIB v1.1.2 包含) 支持 `NCCL_MNNVL_CLIQUE_ID=-2` 按机箱序列号自动分 clique:

```bash
# 实验 1: MNNVL auto-detect (域内 MNNVL + 跨域自动 fallback RDMA)
NCCL_MNNVL_ENABLE=2 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NVLS \
mpirun --np 144 all_reduce_perf -b 1G -e 16G

# 实验 2: 手动 clique 分区
NCCL_MNNVL_ENABLE=2 \
NCCL_MNNVL_CLIQUE_ID=-2 \
mpirun --np 144 all_reduce_perf -b 1G -e 16G
```

**预期收益**: 域内 EP alltoall 从 ~325 GB/s 提升到 ~840 GB/s (+158%)，同时跨域 DP 仍使用 RDMA。

## 9.9 GB300 并行参数缩放指南

结合 GB300 显存优势 (288 GB vs 192 GB) 和 CX-8 GPUDirect 带宽优势:

| 参数 | GB200 2-domain (36 nodes) | GB300 2-domain (36 nodes) | 说明 |
|------|--------------------------|--------------------------|------|
| `--nnodes` | 36 | 36 | 不变 |
| `--tensor-model-parallel-size` | 4 | **2~4** | 显存大 → 可能降 TP |
| `--pipeline-model-parallel-size` | — | **减半** | 显存大 → PP 减半 |
| `--expert-model-parallel-size` | 16 | **16** | 域内 MNNVL 不变 |
| `--micro-batch-size` | 1 | **1~2** | 显存大 → MBS 翻倍 |
| DP (自动) | >=2 | >=2 | 跨域 RDMA |
| `NCCL_MNNVL_ENABLE` | 0 | 0 | 跨域必须关闭 |

### Dense 模型的并行策略 (GB300)

与 GB200 一致，但可利用显存优化:

- **TP=4** (节点内 NVLink，或 GB300 显存足够时降到 TP=2)
- **PP=N** (GB300 可减半 PP stage → 更少 bubble)
- **DP=剩余** (梯度同步跨域 RDMA)

## GKE 测试环境

- 集群: `chrisya-gb300-gke`
- 项目: `tencent-gcp-taiji-poc`
- Zone: `us-central1-b`
- 预留: 214 台, 856 GPU
- Placement policies: `gb300-central-nvl72-policy-0001~0012` (12 个)
- Block 结构: 25 subblocks, 支持 subblock 级 reservation 指定
