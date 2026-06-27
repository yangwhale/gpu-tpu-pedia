# 9. 多 Domain 训练编排

当训练规模超过单个 NVL72 domain（72 GPU），需要跨 domain 调度。本章基于 **2-domain 实验环境**说明如何编排多 domain Megatron-LM 训练，并给出扩展到 4+ domain 的指导。

## 9.1 NVL72 带宽层级

A4X/GB200 的通信带宽随拓扑层级递减，**并行策略必须与带宽层级对齐**：

| 层级 | 范围 | 互联技术 | 带宽 | 适合的并行维度 |
|------|------|----------|------|----------------|
| L1 节点内 | 同节点 4 GPU | NVLink 5th gen (NVSwitch) | ~900 GB/s | TP（张量并行） |
| L2 域内跨节点 | 同 domain 18 节点 | MNNVL（跨节点 NVLink） | ~840 GB/s | EP（专家并行） |
| L3 跨域 | 不同 domain | RDMA over CX-7（4x200Gbps） | ~165 GB/s | DP（数据并行） |

**性能影响**：实测 MoE 模型（Qwen3-30B-A3B）纯 EP=8 训练，同域 MNNVL 达 **~480 TFLOP/s/GPU**，跨域 RDMA 降至 **~282 TFLOP/s/GPU**（下降 37.7%）。

## 9.2 并行策略与拓扑映射

核心原则：**通信量大的并行维度映射到带宽高的层级**。

| 并行维度 | 通信模式 | 通信量 | 推荐映射层级 |
|----------|----------|--------|-------------|
| TP（张量并行） | AllReduce，每层 2 次 | 极高 | L1 节点内 (<=4) |
| EP（专家并行） | AlltoAll，MoE 层每次 dispatch/combine | 高 | L2 域内 (<=72) |
| DP（数据并行） | AllReduce，梯度同步 | 中 | L3 跨域 |
| PP（流水线并行） | 点对点，逐 micro-batch | 低 | L2 或 L3 |

**Megatron-LM 不自动感知拓扑**。`parallel_state.py` 注释明确写道 *"the caller should make sure adjacent ranks are on the same DGX box"*。因此，**必须由调用方通过 NODE_RANK 编排确保 rank 顺序与物理拓扑对齐**。

### Rank 编排规则

`torchrun` 的 rank 计算公式：`global_rank = NODE_RANK x nproc_per_node + LOCAL_RANK`。Megatron-LM 按 rank 的连续性来切分并行组：

- **TP group**：连续的 `TP_SIZE` 个 rank（TP=4 时：rank 0-3 = 1 个 TP group，对应 1 个节点的 4 GPU）
- **EP group**：在 TP group 之上，连续的 `EP_SIZE` 个 rank 组成 EP group
- **DP group**：不在同一 EP group 的对应位置 rank 组成 DP group

因此，**同一 domain 内的节点必须获得连续的 NODE_RANK**：

```
# Domain 0: NODE_RANK = 0, 1, 2, ..., 17  （72 GPU, rank 0-71）
# Domain 1: NODE_RANK = 18, 19, ..., 35   （72 GPU, rank 72-143）
#
# TP=4: rank [0,1,2,3] = node 0 的 4 GPU (同节点 NVLink)
# EP=72: rank [0-71] 的对应 EP 位置 = domain 0 全部 GPU (域内 MNNVL)
# DP=2: rank 0 和 rank 72 组成 DP pair (跨域 RDMA)
```

## 9.3 推荐并行配置（MoE 模型）

以 Qwen3-30B-A3B（128 experts, topk=8）为例：

| 规模 | 节点数 | GPU 总数 | TP | EP | DP | 每 GPU 专家数 | 跨域通信 |
|------|--------|---------|----|----|----|----|----------|
| 1 节点 | 1 | 4 | 1 | 4 | 1 | 32 | 无 |
| 1 domain (2 nodes) | 2 | 8 | 1 | 8 | 1 | 16 | 无（MNNVL） |
| **2 domain** | **2x1** | **8** | **1** | **4** | **2** | **32** | **DP 梯度同步** |
| 2 domain (满配) | 2x18 | 144 | 4 | 16 | >=2 | 8 | DP 梯度同步 |
| 4 domain (满配) | 4x18 | 288 | 4 | 32 | >=2 | 4 | DP 梯度同步 |

**EP 取值约束**：`EP_SIZE` 必须能整除 `num_experts`（128）。可用值：1, 2, 4, 8, 16, 32, 64, 128。

## 9.4 生产调度方案：JobSet + Kueue TAS（推荐）

JobSet（v0.12+）管理多组 Job 作为整体，Kueue（v0.18+）的 TAS（Topology Aware Scheduling）确保每组 Pod 调度到同一拓扑域。

| 特性 | 手动 Pod（9.5 节） | JobSet + Kueue TAS（本节） |
|------|-------------------|---------------------------|
| 域级调度 | `nodeSelector` 手动指定 | Kueue TAS `podset-required-topology` 自动 |
| 节点发现 | SSH + `kubectl get pod -o jsonpath` | DNS hostname（`enableDNSHostnames`） |
| NODE_RANK | 手动编排 | `JOB_COMPLETION_INDEX + DOMAIN_OFFSET` |
| SSH 密钥交换 | 必须手动交换 | 不需要（torchrun 使用 c10d/Gloo） |
| 故障恢复 | 手动重新部署 | Job 自动 restart |

### 步骤 1：定义 Topology CRD

```yaml
cat <<'EOF' | kubectl apply -f -
apiVersion: kueue.x-k8s.io/v1beta2
kind: Topology
metadata:
  name: gb200-nvl72-topology
spec:
  levels:
  - nodeLabel: "nvl72-domain"          # NVL72 domain 级别
  - nodeLabel: "kubernetes.io/hostname" # 节点级别
EOF

# 验证节点拓扑发现
kubectl get topology gb200-nvl72-topology -o yaml
```

### 步骤 2：创建 Kueue 队列

```yaml
cat <<'EOF' | kubectl apply -f -
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: gb200-flavor
spec:
  nodeLabels:
    nvl72-domain: domain-1                 # ← 按实际 domain 标签值调整
  topologyName: gb200-nvl72-topology
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  namespaceSelector: {}                    # ← v0.18 必须显式设置
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
        nominalQuota: 288              # ← 按实际调整
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

**Kueue v0.18 注意事项**：

- 所有 Kueue 资源使用 `kueue.x-k8s.io/v1beta2` API（v1beta1 已弃用）
- `ResourceFlavor` 设置 `topologyName` 时，**必须同时提供 `nodeLabels`**
- `ClusterQueue` 需要 `namespaceSelector: {}`
- TAS 在 v0.18 为 **beta 默认启用**，无需额外 feature gate

### 步骤 3：提交 JobSet

每个 domain 一个 ReplicatedJob，Kueue TAS 保证同一 ReplicatedJob 的所有 Pod 落在同一 NVL72 domain。

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
        parallelism: 1                             # ← 生产改为 18
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
                value: "0"                         # ← 跨域训练必须关闭
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
                value: "1"                         # ← 生产用 18
              # 其余同 domain-0
            resourceClaims:
            - name: compute-domain-channel
              resourceClaimTemplateName: domain-1-compute-domain-channel
            - name: rdma-nics
              resourceClaimTemplateName: rdma-nics-domain-1
```

### Kubeflow Trainer

Kubeflow Trainer（v2.2+）的 `TrainJob` CRD 集成了 Kueue + JobSet，提供更高级的抽象（自动初始化 torchrun、分配 rank、挂载数据卷），适合已部署 Kubeflow 的团队。

## 9.5 简易验证方案（手动 Pod + nodeSelector）

**仅适用于小规模实验验证**。需要手动管理 SSH 密钥、Pod IP、NODE_RANK。生产环境请使用 9.4 节 JobSet + Kueue TAS。

**关键参数对比（同域 vs 跨域）**：

| 参数 | 同域（第 8 章） | 跨域（本章） | 说明 |
|------|----------------|-------------|------|
| `NCCL_MNNVL_ENABLE` | 2 | **0** | 跨域必须关闭 |
| `NCCL_CUMEM_ENABLE` | 1 | **0** | cuMem 仅用于 MNNVL |
| `--expert-model-parallel-size` | 8（跨 2 节点） | **4**（仅节点内） | EP 限制在域内 |
| DP | 1 | **2** | 自动计算 |
| ComputeDomain | 需要 | **仅 domain-0** | 跨域无共享 IMEX |

## 9.6 扩展到 4+ Domain

### NODE_RANK 编排（4 domain x 18 nodes = 72 nodes）

```
# 确保同一 domain 的节点获得连续 NODE_RANK
#
# Domain 0: NODE_RANK = 0 .. 17   → rank 0-71   (72 GPU)
# Domain 1: NODE_RANK = 18 .. 35  → rank 72-143  (72 GPU)
# Domain 2: NODE_RANK = 36 .. 53  → rank 144-215 (72 GPU)
# Domain 3: NODE_RANK = 54 .. 71  → rank 216-287 (72 GPU)
#
# JobSet 方案：增加 ReplicatedJob（domain-2, domain-3），
# 每个设置 DOMAIN_OFFSET = 36, 54；parallelism/completions = 18
```

### 并行参数缩放

| 参数 | 2 domain (2 nodes) | 2 domain (36 nodes) | 4 domain (72 nodes) |
|------|--------------------|--------------------|---------------------|
| `--nnodes` | 2 | 36 | 72 |
| `--tensor-model-parallel-size` | 1 | 4 | 4 |
| `--expert-model-parallel-size` | 4 | 16 | 32 |
| DP (自动) | 2 | >=2 | >=2 |
| 每 GPU 专家数 | 32 | 8 | 4 |
| `NCCL_MNNVL_ENABLE` | 0 | 0 | 0 |

### Dense 模型的并行策略

Dense 模型（如 Llama）无 EP 维度，推荐映射：

- **TP=4**（节点内 NVLink，AllReduce 高频通信）
- **PP=N**（跨节点流水线，点对点通信量小，可跨域）
- **DP=剩余**（梯度同步，可跨域）
- 优先将 PP stage 的相邻 rank 安排在同一 domain 内

## 9.7 待验证优化：Per-Communicator MNNVL（域内 MNNVL + 跨域 RDMA）

**本节内容为调研结论，尚未在实验环境中验证。**

### 问题

当前跨域训练必须设置 `NCCL_MNNVL_ENABLE=0`，导致所有通信都降级为 RDMA。域内 EP alltoall 从 ~840 GB/s 降至 ~325 GB/s，性能损失约 60%。

### 调研发现

NCCL 源码分析表明，MNNVL 检测实际上是 **per-communicator** 的。

| NCCL 版本变化 | 影响 |
|-------------|------|
| <= 2.26.x | MNNVL 探测失败时静默回退到 RDMA |
| **2.27.3+** | MNNVL 探测失败时 **hard fail**（`ncclSystemError`），不再静默回退 |
| **2.28.3+** | 新增 `NCCL_MNNVL_CLIQUE_ID=-2`：按机箱序列号自动分 clique |
| **2.30.3+** | 新增 `NCCL_MNNVL_CROSS_CLIQUE=1`：允许跨 clique P2P |

GIB v1.1.2 包含 **NCCL 2.30.4**，上述新特性均可用。理论上 `NCCL_MNNVL_ENABLE=2`（auto-detect）应能正确处理混合域场景。

### 验证计划

```bash
# 实验 1：MNNVL auto-detect 基础验证
NCCL_MNNVL_ENABLE=2 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NVLS \
mpirun ... all_reduce_perf -b 1G -e 16G

# 实验 2：Megatron EP=域内 + DP=跨域
NCCL_MNNVL_ENABLE=2 \
torchrun --nnodes=2 --nproc_per_node=4 \
  pretrain_gpt.py --expert-model-parallel-size 4 ...

# 实验 3：手动 clique 分区
NCCL_MNNVL_ENABLE=2 \
NCCL_MNNVL_CLIQUE_ID=-2 \
mpirun ... all_reduce_perf -b 1G -e 16G
```

**预期收益**：域内 EP alltoall 可从 ~325 GB/s 提升到 ~840 GB/s（+158%），MoE 模型跨域训练吞吐显著提升。

---

## 10. 64+8 备用容量管理

**策略**：72 GPU (18 节点) 中，64 GPU (16 节点) 做 pre-training，8 GPU (2 节点) 作为备用容量（~11% 冗余）。通过 PriorityClass + Placeholder Pod 实现自动容量管理。

### 10.1 PriorityClass 定义

```bash
kubectl apply -f yamls/spare-capacity-priority.yaml
kubectl get priorityclass
# gpu-training (1000)  gpu-placeholder (-1)
```

### 10.2 Placeholder Pod 部署

```bash
kubectl apply -f yamls/spare-capacity-placeholder.yaml
kubectl get pods -l app=gpu-placeholder -o wide
```

### 10.3 训练 Pod 抢占测试

```bash
kubectl apply -f yamls/spare-capacity-training-job.yaml
kubectl get pods -o wide
# training-preempt-test Running, 某个 gpu-placeholder 被驱逐
```

### 10.4 故障切换 (Cordon/Drain/Uncordon)

```bash
# Step 1: 隔离故障节点
kubectl cordon <NODE_NAME>
kubectl drain <NODE_NAME> --ignore-daemonsets --delete-emptydir-data --force

# Step 2: 验证 Pod 重调度
kubectl get pods -l app=gpu-placeholder -o wide

# Step 3: 节点修复后恢复
kubectl uncordon <NODE_NAME>
```

---

## 11. 故障上报 API (Report-Faulty)

### 单节点故障上报

```bash
gcloud compute instances report-host-as-faulty <INSTANCE_NAME> \
  --async \
  --disruption-schedule=IMMEDIATE \
  --fault-reasons=behavior=UNRECOVERABLE_GPU_ERROR,description="XID 79 detected" \
  --zone=$ZONE --project=$PROJECT
```

| fault-reasons behavior | 含义 | 场景 |
|----------------------|------|------|
| `PERFORMANCE` | GPU 性能下降 | 无 Xid 错误但训练速度下降 |
| `UNRECOVERABLE_GPU_ERROR` | 不可恢复 GPU 错误 | Xid 错误 |
| `SILENT_DATA_CORRUPTION` | 静默数据损坏 | 训练结果异常 |
| `CHIP_ERROR` | 芯片错误 | 硬件芯片级故障 |

### 11.2 Sub-block 域级故障上报

当 NVSwitch 等域级组件故障时，需要上报整个 sub-block（而非单节点）。先查看预留的拓扑结构（block / sub-block），再针对故障 sub-block 上报。

```bash
# 查看拓扑结构
gcloud alpha compute reservations blocks list $RESERVATION \
  --zone=$ZONE --project=$PROJECT

gcloud alpha compute reservations sub-blocks list $RESERVATION \
  --block-name=<BLOCK_NAME> \
  --zone=$ZONE --project=$PROJECT

# Sub-block 故障上报
gcloud alpha compute reservations sub-blocks report-subblock-as-faulty $RESERVATION \
  --block-name=<BLOCK_NAME> \
  --sub-block-name=<SUBBLOCK_NAME> \
  --disruption-schedule=IMMEDIATE \
  --fault-reasons=behavior=SWITCH_FAILURE,description="NVSwitch failure" \
  --failure-component=NVLINK_SWITCH \
  --zone=$ZONE --project=$PROJECT
```

| failure-component | 含义 | 场景 |
|-------------------|------|------|
| `NVLINK_SWITCH` | NVSwitch 故障 | 域内 NVLink 通信异常 |

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

## 12. GPU 监控 (DCGM Exporter)

```bash
kubectl apply -f yamls/dcgm-exporter-daemonset.yaml

# 验证
kubectl get pods -n kube-system -l app=dcgm-exporter

# 测试指标
kubectl exec -n kube-system <DCGM_POD> -- wget -qO- http://localhost:9400/metrics | grep '^DCGM_FI' | head -20
```

### 关键监控指标

| 指标 | 告警阈值 | 含义 |
|------|----------|------|
| `DCGM_FI_DEV_GPU_TEMP` | > 85 C | GPU 过热 |
| `DCGM_FI_DEV_MEMORY_TEMP` | > 95 C | 显存过热 |
| `DCGM_FI_DEV_POWER_USAGE` | > 1000W | 功耗异常 |
| `DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS` | > 0 | 不可纠正 ECC 错误 |
| `DCGM_FI_DEV_ROW_REMAP_FAILURE` | > 0 | 行重映射失败 |
| `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | 持续增长 | PCIe 链路质量问题 |

**Prometheus + Grafana 集成**：DCGM Exporter 默认暴露 `:9400/metrics`，使用 NVIDIA 官方 Grafana Dashboard ID: 12239。

---

## 压测标准（附录 C）

### GEMM 性能

| 精度 | GB200 官方数据 (TFLOP/s) | GCP A4X 实测 (TFLOP/s) |
|------|-------------------------|----------------------|
| FP4 | 6507 | 6845 |
| FP8 | 2805 | 3063 |
| FP16 | 1372 | 1492 |
| BF16 | 1471 | 1592 |
| TF32 | 675 | 733 |
| FP32 | 75 | 75 |

### NCCL 性能（Packet size = 16GB，busbw GB/s）

| 配置 | AllReduce | All2All | AllGather | ReduceScatter |
|------|-----------|---------|-----------|---------------|
| 1 domain (72 GPU) | 845.01 | 660.61 | 687.94 | 707.22 |
| 2 domain (144 GPU) | 749.56 | 65.47 | 749.56 | 689.05 |
| 4 domain (288 GPU) | 待测 | | | |

**2-domain All2All 带宽说明**：跨 domain All2All 仅 65.47 GB/s，远低于同域 660 GB/s。AlltoAll 对跨域带宽敏感度最高，因为每个 GPU 需要与所有其他 GPU 交换数据。

### 训练性能

| 模型 | seq_len | GPU 数 | mbs | gbs | 并行策略 | TFLOP/s/gpu | MFU |
|------|---------|--------|-----|-----|----------|-------------|-----|
| LLaMA2-7B | 4096 | 4 | 4 | 256 | tp1pp1 | 1,127 | 45.08% |
| LLaMA2-7B | 4096 | 8 | 4 | 256 | tp1pp1 | 1,086 | 43.44% |
| LLaMA2-13B | 4096 | 4 | 1 | 256 | tp1pp1 | 1,065 | 42.60% |
| LLaMA2-13B | 4096 | 8 | 2 | 256 | tp1pp1 | 1,065 | 42.60% |
| LLaMA3-70B | 4096 | 64 | 2 | 1024 | tp4pp2 | 971 | 38.83% |

**环境**：TransformerEngine v2.15 / Megatron-LM core_r0.16.0 / pytorch:26.05-py3

---

## 待完善

以下为源文档中尚未覆盖的重要生产主题，需后续补充。

### 安全加固指南

<!-- TODO -->

集群安全配置，包括 Pod Security Standards（Restricted profile）、NetworkPolicy 网络隔离规则、以及容器镜像签名验证（cosign / Binary Authorization）。

### Prometheus GPU 健康告警规则

<!-- TODO -->

基于 DCGM Exporter 指标的 Prometheus AlertManager 告警规则定义，覆盖 GPU 温度、ECC 错误、PCIe 重传、行重映射失败等关键场景的告警阈值和通知策略。

### 训练 Checkpoint / 故障恢复策略

<!-- TODO -->

训练任务的容错 checkpoint 策略，包括异步 checkpoint 写入配置、故障后自动恢复流程（从最近 checkpoint 恢复 + 跳过故障节点）、以及多 domain 训练的分布式 checkpoint 一致性保障。

### 自动化 GPU 健康流水线

<!-- TODO -->

从 XID 错误检测到节点自动隔离的端到端自动化流水线：XID error 检测（dmesg / DCGM）→ 节点 cordon/drain → 自动调用 report-host-as-faulty API → 备用节点替换 → 训练任务自动恢复。
