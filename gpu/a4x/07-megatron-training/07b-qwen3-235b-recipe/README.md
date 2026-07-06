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
| PP=2 EP=32 跨域 | 64 | 双域 (8+8节点) | 2 | 32 | 0 | 685 | 7.1s | GKE, V1 recipe, TE scoped graph |
| **PP=2 EP=32 跨域 V2** | **64** | **双域 (8+8节点)** | **2** | **32** | **0** | **1124** | **4.31s** | **V2 recipe, full_iteration graph + paged stash** |

> **跨域结果惊喜**：MNNVL=0 + USE_MNNVL=1（奚老师方案）跨两个 NVL72 域跑出 685，几乎等于单域 MNNVL=2 的 686。原因：PP=2 跨域 p2p 通信量小（只传 activation），RDMA 200GB/s 不是瓶颈；EP=32 all-to-all 全在域内走 HybridEP NVLink，不受跨域影响。

### GKE 集群优化迭代 (2026-07-05, 8 轮实验)

基于跨域 685 baseline，逐个测试奚老师 DSv3 报告中的优化参数：

| 轮次 | 改了什么 | FP8 | TFLOP/s | 备注 |
|---|---|---|---|---|
| **baseline** | PP=2 EP=32 MNNVL=0+USE_MNNVL=1 | mxfp8 | **685** | **最优配置** |
| R2 | + recompute (moe_act,mlp) | mxfp8 | 666 (-3%) | 重算 activation 增加计算量 |
| R3 | + recompute + NVLS=1 | mxfp8 | OOM | NVLS multicast buffer HBM 不够 |
| R4 | + NCCL_GRAPH_REGISTER=1 | mxfp8 | crash | assert: 与 expandable_segments 冲突 |
| T2 | + seq_length=8192 | mxfp8 | OOM | activation 翻倍超 184GB |
| T3 | + cuDNN fusion (NVTE_FUSED_ATTN=1 等) | mxfp8 | 645 (-6%) | 覆盖 recipe 自选的更优实现 |
| T4 | 30B NVLS=1 对照 | mxfp8 | 926 | 30B 8 卡 NVLS 正常 (slot 够用) |
| T5 | 换 fp8_cs (current scaling) | fp8_cs | 701→595 | 前 11 步高 2%, 后退化 -15% |

#### 关键发现

1. **NeMo recipe 已包含 fp8-param-gather**: `bf16_with_mxfp8_mixed()` 自动设置 `fp8_param_gather=True` + `reuse_grad_buf_for_mxfp8_param_ag=True`。奚老师从 928→975 的 5% 提升，我们的 recipe 一直都有
2. **seq_length 被 recipe 锁定**: `set_qwen3_common_configs()` 硬编码 `cfg.model.seq_length = 4096`，命令行 `-sl` 只改 dataset 不改 model。需要 sed patch Python 文件才能改
3. **NVLS OOM 是 HBM 不够**: 不是 NVSwitch multicast slot 耗尽（那是 NCCL #2077 的别的场景），而是 235B 模型用了 180+GB HBM 留不出 multicast buffer 的空间
4. **fp8_cs 退化**: per-tensor current scaling 在 Qwen3 235B 上出现 iter 12+ 退化（与 DSv3 上 mxfp8 退化互为镜像），FP8 退化行为与模型架构耦合
5. **685 是 H=4096 天花板**: 计算密度 (H=4096 vs DSv3 H=7168) 差 3x，是 TFLOP/s 低的根因。奚老师在同硬件测 Qwen3 235B 也只有 219-325

#### NVLS 深度分析

| 场景 | NVLS | 结果 |
|---|---|---|
| 30B 8 卡 (NVLS=1) | ✅ 正常 | 926 (vs NVLS=0 的 924, 差 0.2%) |
| 235B 64 卡 (NVLS=1) | ❌ OOM | HBM 180+GB 用满, multicast buffer 分配失败 |
| DSv3 61L (奚老师 NVLS=1) | ❌ 退化 | iter 20-40 后 TFLOP/s 渐降 30-50% (时间相关 bug) |

NVLS 有两种独立失败模式: (1) GPU HBM OOM 分配 multicast buffer (我们的 235B), (2) NVLS transport 时间退化 bug (奚老师的 DSv3)。生产配置一律 NVLS=0。

### 第三轮优化：突破 685 → 1124 (2026-07-06, GKE 集群跨域)

#### 关键发现：V1 vs V2 recipe 的本质差异

dump NeMo recipe 配置后发现，之前一直用的 `-cv v1` 和 `-cv v2` 是两套完全不同的技术栈：

| 配置项 | V1 (之前用的) | V2 (这次用的) |
|---|---|---|
| cuda_graph_impl | **transformer_engine** (TE scoped) | **full_iteration** (整个 step) |
| cuda_graph_modules | full (但被 TE scoped 限制) | full |
| moe_paged_stash | **False** | **True** |
| moe_expert_rank_capacity_factor | — | 1.5 |
| moe_paged_stash_buffer_size_factor_cuda | 1.1 | 1.2 |
| virtual_pipeline_model_parallel_size | None | **3** |
| PP / EP | 8 / 8 (默认) | 8 / 32 |

**V2 recipe 就是 NVIDIA 1106 TFLOPs 的完整配置**，包含 full_iteration graph + paged stash + VPP。V1 是保守配置。

#### 为什么之前不能开 full_iteration graph，现在又能了

**之前的理解（错误的）**：full_iteration CUDA Graph 跟 HybridEP 在 PP>1 时不兼容，因为 CUDA fabric memory 的动态操作会 invalidate graph capture。这个结论来自奚老师在 raw Megatron-LM (pretrain_gpt.py) 上的实测。

**正确的理解**：raw Megatron-LM 没有 Bridge 的 3 项专有技术（sync-free device-initiated kernel、ECHO、paged stashing），所以 full_iteration graph 确实 crash。但 NeMo 的 run_script.py 走的是 Megatron Bridge，Bridge 在 V2 recipe 里启用了这些技术：

1. **Sync-Free Device-Initiated Kernels**：Bridge 重写了 Grouped GEMM 和 HybridEP dispatch，让 kernel 从 GPU memory 自主读 shape 信息决定执行方式，无需 CPU-GPU 同步。整个 MoE 层零 host-device sync，CUDA Graph 可完整 capture。

2. **Paged Stashing** (`moe_paged_stash=True`)：在 CUDA Graph 内部做细粒度内存管理，预分配 buffer 中没用到的部分被动态回收给其他操作。解决了 full_iteration graph 的内存爆炸问题。

3. **Expert Rank Capacity Factor** (`moe_expert_rank_capacity_factor=1.5`)：按 worst-case 的 1.5 倍预分配 per-expert buffer，配合 paged stash 回收未使用部分。

这些技术在 V1 recipe 里全部关闭（`moe_paged_stash=False`），V2 recipe 里全部开启。**差距不是调参，是技术栈切换**。

#### 效果的原理

**685 → 1124 的 64% 提升**由三个因素叠加：

**1. full_iteration CUDA Graph (+40-50%，最大贡献)**

V1 的 TE scoped graph 只 capture attention 模块。MoE 层（router + preprocess + dispatch + expert compute + combine）在 graph 外执行，每层的每个操作都要从 host 逐个 launch kernel。94 层 × 每层数十个 MoE kernel = 上千次 host launch overhead。

full_iteration graph 把整个 training step（forward + backward + optimizer update）录成一张大 graph，CPU 只发一条 replay 命令。所有 MoE kernel 的 launch overhead 归零。这就是为什么 step time 从 7.1s 降到 4.3s。

**2. Paged Stashing (使能 full graph 的前提)**

没有 paged stash，full_iteration graph 需要按 worst-case 为每个 expert 预分配固定 buffer，94 层 × 128 expert × 1.5 倍余量的 buffer 会超过 184 GB HBM → OOM。Paged stash 在 graph 执行过程中动态回收未使用的 buffer 空间给其他操作复用，让内存峰值可控。

**3. VPP=3 (V2 recipe 默认，但被我们的 PP=2 覆盖)**

V2 recipe 默认 PP=8 VP=3，我们覆盖成 PP=2 后 VPP 被忽略（PP=2 不需要 VPP，bubble 本身就小）。所以这次的 1124 纯粹来自 full graph + paged stash，没有 VPP 贡献。

#### 实测结果

| 轮次 | recipe | graph | paged_stash | TFLOPs | step time | 提升 |
|---|---|---|---|---|---|---|
| 之前所有测试 | V1 (`-cv v1`) | TE scoped (attn) | False | **685** | 7.1s | baseline |
| **R9** | **V2 (`-cv v2`)** | **full_iteration** | **True** | **1124** | **4.31s** | **+64%** |

> **1124 TFLOPs 超过了 NVIDIA 256 GPU 的参考值 1106**。原因可能是我们用 PP=2 EP=32 的 bubble 比 NVIDIA 的 PP=8 VP=3 更小，且 64 GPU 跨 2 域的通信拓扑比 256 GPU 更简单。

#### 为什么之前没发现

1. 一直用 `-cv v1`（NeMo 文档推荐 64 GPU 用 V1），没试过 `-cv v2`
2. 误以为 V2 只是并行度不同（PP=8 EP=32 VPP=3），没意识到 V2 同时切换了整个 CUDA Graph 技术栈
3. 基于奚老师 raw Megatron-LM 的经验错误推断"full_iteration + HybridEP + PP>1 不兼容"——这个结论只对 raw Megatron-LM 成立，不适用于 NeMo Bridge

#### 获取这些 Bridge feature 的版本要求

| 组件 | 最低版本 | 我们使用的版本 | 说明 |
|---|---|---|---|
| NeMo 容器 | **nemo:26.06** | `nvcr.io/nvidia/nemo:26.06` | 必须用 NeMo 容器而非裸 PyTorch 容器 |
| Megatron Bridge | NeMo 26.06 内置 | 同上 | Bridge 是 NeMo 的打包层，提供 run_script.py 入口 |
| Megatron Core | **0.18.0+** | 0.18.0+d0b3b7754 (NeMo 26.06.rc7 内置) | 0.18.0 首次支持 `--cuda-graph-modules`；0.17.x hang |
| 入口脚本 | `run_script.py` | `/opt/Megatron-Bridge/scripts/performance/run_script.py` | **不能用** `pretrain_gpt.py`（raw Megatron-LM，无 Bridge 优化） |
| config variant | **`-cv v2`** | `-cv v2` | V1 不启用 full graph / paged stash；V2 全部启用 |

**关键区分**：

- `run_script.py` → 走 Megatron Bridge → 有 sync-free kernel + paged stash + ECHO → full_iteration graph 安全
- `pretrain_gpt.py` → 走 raw Megatron-LM → 无 Bridge 优化 → full_iteration + HybridEP + PP>1 = crash

奚老师用的是 `pretrain_gpt.py`，所以被限制在 TE scoped graph (981)。我们用 `run_script.py -cv v2`，Bridge 的完整优化生效，跑到 1124。

**如何验证 Bridge 是否生效**：

```bash
# dump config 检查这些字段
python3 run_script.py ... --dump_env
# 或在容器内 python3 直接检查
from configs.qwen.qwen3_llm_pretrain import qwen3_235b_a22b_pretrain_config_gb200
cfg = qwen3_235b_a22b_pretrain_config_gb200(precision="fp8_mx", config_variant="v2")
assert cfg.model.cuda_graph_impl == "full_iteration"   # 必须是 full_iteration
assert cfg.model.moe_paged_stash == True                # 必须开
```

#### 教训

**Recipe config variant 不只是并行度配置，是技术栈选择**。V1→V2 不是改了几个数字，是开关了 sync-free kernel + paged stash + full_iteration graph 的完整组合。dump config 确认实际配置是必要的诊断步骤。

**入口脚本决定技术栈**。同一个 NeMo 容器，`run_script.py` 走 Bridge 有完整优化，`pretrain_gpt.py` 走 raw Megatron-LM 无 Bridge 优化。选错入口差 64%。

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
          image: nvcr.io/nvidia/nemo:26.06
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

### 跨域部署（实测方案, GKE 集群 域 A + 域 B）

跨两个 NVL72 域时，用两个 LWS + 两个 ComputeDomain，分别锁定不同的节点池：

```yaml
# Domain A (域 A, PP stage 0, node_rank 0-7)
ComputeDomain: nemo-235b-domain-a (numNodes=8, channel=nemo-235b-channel-a)
LWS: nemo-235b-a (size=8, nodePool=域 A)

# Domain B (域 B, PP stage 1, node_rank 8-15)
ComputeDomain: nemo-235b-domain-b (numNodes=8, channel=nemo-235b-channel-b)
LWS: nemo-235b-b (size=8, nodePool=域 B)
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
