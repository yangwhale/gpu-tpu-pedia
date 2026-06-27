# RL 训练指南：GB200 NVL72 上的强化学习

> **注意**：本章为**补充内容**，不来自原始 GB200 A4X 部署指南文档。内容基于 RL 训练最佳实践调研整理，结合 GB200 NVL72 硬件特性编写，供参考使用。

本章介绍在 GB200 NVL72 集群上进行大规模 RL（Reinforcement Learning）训练的实践指南，包括框架选择、并行策略、模型配置和调优建议。

## 硬件概述

| 参数 | 值 |
|------|-----|
| GPU 型号 | NVIDIA GB200 (Blackwell, sm_100) |
| 每节点 GPU | 4 |
| NVL72 Domain | 18 节点 / 72 GPU |
| 可用 GPU（扣除 holdback） | **64 GPU**（16 节点活跃，2 节点备用） |
| 域内互联 | NVLink 5th gen (MNNVL), ~840 GB/s |
| 跨域互联 | RDMA over CX-7, ~325 GB/s |
| 显存 | HBM3e per GPU |

**Holdback 策略**：72 GPU 中保留 8 GPU（2 节点）作为备用容量（详见 [08-multi-domain](../08-multi-domain/) 第 10 节），实际可用于训练的为 64 GPU。

## 框架选择

### veRL（推荐）

[veRL](https://github.com/volcengine/verl)（Volcano Engine Reinforcement Learning）是字节跳动开源的大模型 RL 训练框架，支持 GRPO、PPO、REINFORCE 等算法。

**推荐原因**：
- 原生支持 colocated 和 disaggregated 部署模式
- 与 Megatron-LM 和 vLLM 深度集成
- 支持 MoE 模型的 RL 训练
- 活跃的社区和文档

### AReaL

[AReaL](https://github.com/inclusiveai/AReaL)（Alibaba Reinforcement Learning）是阿里巴巴开源的 RL 训练框架，专注于大规模 reasoning 模型训练。

**适用场景**：
- 已有阿里云基础设施集成
- 需要 reasoning-oriented RL（如 math、code 任务）

## 模型选择

| 模型 | 参数量 | 类型 | Experts | Active Params | 推荐场景 |
|------|--------|------|---------|---------------|----------|
| **Qwen 3.5 397B** | 397B | MoE | 128 experts, topk=8 | ~50B | 单 domain (64 GPU) 内 RL |
| **DeepSeek V3 671B** | 671B | MoE | 256 experts, topk=8 | ~37B | 多 domain RL，需更大模型容量 |

**MoE 模型优势**：RL 训练中 rollout 阶段的计算量远大于 training 阶段，MoE 模型的稀疏激活特性（仅 topk experts 活跃）在 rollout 吞吐上有显著优势。

## 并行策略

### 推荐配置（64 GPU，单 domain）

```
TP=8   EP=8   DP=8   FSDP for training
```

| 并行维度 | 大小 | 映射 | 说明 |
|----------|------|------|------|
| **TP** (Tensor Parallel) | 8 | 2 节点 x 4 GPU/节点 | 跨 2 个相邻节点，使用 MNNVL (~840 GB/s) |
| **EP** (Expert Parallel) | 8 | 与 TP 共享同一组 8 GPU | EP 和 TP 共享物理 GPU，MoE 层用 EP，attention 层用 TP |
| **DP** (Data Parallel) | 8 | 64 / (TP x PP) = 8 | 跨不同 TP 组，梯度同步 |
| **FSDP** | 启用 | 在 DP 维度上 | ZeRO-3 级别的参数/梯度/优化器状态分片 |

**TP 和 EP 共享 GPU**：在 MoE 模型中，TP 和 EP 通常共享同一组物理 GPU。Attention/FFN 层使用 TP 切分，MoE 层使用 EP 切分。这避免了额外的 GPU 间数据搬运。

### 为什么不用 PP

Pipeline Parallel 在 RL 训练中引入额外的 bubble overhead，且 RL 的 rollout 和 training 交替执行模式与 PP 的流水线填充不兼容。对于 64 GPU 规模，TP+EP+DP+FSDP 组合已足够。

> **TP=8 域内放置警告**：TP=8 要求参与的 2 个节点（4 GPU/节点 x 2 = 8 GPU）必须位于**同一个 NVL72 domain** 内。域内节点间通过 MNNVL 互联，带宽约 **~840 GB/s**。如果 TP 组跨越不同 domain，通信将回退到 RDMA（约 **~325 GB/s**，带宽下降 61%），TP 的高频 AllReduce 通信（每层 2 次）会成为严重瓶颈，导致训练性能大幅下降。使用 JobSet + Kueue TAS 或手动 nodeSelector 确保 TP 组内的节点落在同一拓扑域。

## 部署模式

### Colocated（推荐，单 domain）

**Colocated 模式**将 rollout（生成）和 training（反向传播）部署在同一组 GPU 上，交替执行。

```
┌─────────── 64 GPU (单 domain) ───────────┐
│                                            │
│  Phase 1: Rollout (vLLM inference)         │
│  ├── 加载模型权重到所有 GPU                  │
│  ├── 生成 responses (batch 1024-2048)       │
│  └── 收集 trajectories                      │
│                                            │
│  Phase 2: Training (Megatron/FSDP)         │
│  ├── 计算 rewards                           │
│  ├── GRPO/PPO 更新                          │
│  └── 同步更新后的权重                        │
│                                            │
│  → 交替执行 Phase 1 和 Phase 2              │
└────────────────────────────────────────────┘
```

**优势**：
- 无需跨 GPU 组传输模型权重
- 所有 GPU 的 HBM 都用于同一阶段，利用率高
- 适合单 domain 规模（64-72 GPU）

**劣势**：
- Rollout 和 Training 不能同时进行，GPU 利用率受限于较慢的阶段

### Disaggregated（多 domain 扩展）

当训练规模需要超过 1 个 domain 时，可将 rollout 和 training 分配到不同的 GPU 组。

```
┌──── Domain 0: Training (64 GPU) ────┐   ┌──── Domain 1: Rollout (64 GPU) ────┐
│  接收 trajectories                    │   │  生成 responses                     │
│  GRPO/PPO 更新                        │←→│  发送 trajectories                   │
│  发送更新后权重                        │   │  接收更新后权重                      │
└──────────────────────────────────────┘   └──────────────────────────────────────┘
```

**适用场景**：rollout 计算量远大于 training 时（如 agentic RL），可用更多 GPU 做 rollout 加速整体吞吐。

## GRPO 配置建议

[GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) 是当前主流的 RL 算法，相比 PPO 省去了 critic model。

### 核心超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `batch_size` | 1024-2048 | 每个 RL iteration 的 prompt 数量 |
| `mini_batch_splits` | 2-4 | 将 batch 切分为 mini-batch 做 gradient accumulation |
| `num_iterations` | 1 | 每个 batch 的 policy 更新次数（GRPO 通常用 1） |
| `max_new_tokens` | 2048-8192 | rollout 生成的最大 token 数（agentic 任务需更长） |
| `temperature` | 0.7-1.0 | 采样温度，控制 exploration |
| `group_size` | 4-8 | 每个 prompt 生成的 response 数量（用于组内相对排序） |
| `kl_coeff` | 0.01-0.1 | KL 散度惩罚系数 |
| `clip_range` | 0.2 | PPO-style clipping |

### Rollout:Training 比例

| 任务类型 | 推荐比例 | 说明 |
|----------|----------|------|
| 标准 RL (math, code) | 2:1 到 3:1 | Rollout 生成序列较短，训练更新频繁 |
| **Agentic RL** (tool use, multi-turn) | **5:1 到 8:1** | Rollout 需要多轮工具调用，生成序列极长 |

**Agentic RL 注意**：agentic 任务（如 tool-use、multi-turn reasoning）的 rollout 涉及多轮工具调用，每次 rollout 的 token 生成量远大于标准 RL。需要更高的 rollout:training 比例来避免 training GPU 空等。

## veRL 配置示例

以下为在 64 GPU 上使用 veRL 进行 Qwen 3.5 397B GRPO 训练的参考配置：

```yaml
# verl_config.yaml
trainer:
  total_epochs: 3
  save_freq: 50
  test_freq: 10

rollout:
  name: vllm
  tp: 8                          # 与 training TP 一致
  gpu_memory_utilization: 0.85
  max_num_batched_tokens: 32768
  max_model_len: 16384
  temperature: 0.7
  top_p: 0.95
  max_new_tokens: 4096

actor:
  strategy: fsdp                  # FSDP for training
  tp: 8
  ep: 8
  optim:
    lr: 1e-6
    weight_decay: 0.01
    warmup_steps: 10

data:
  train_batch_size: 1024
  mini_batch_size: 256            # 1024 / 4 splits
  max_prompt_length: 4096

algorithm:
  name: grpo
  group_size: 4
  kl_coeff: 0.05
  clip_range: 0.2
  num_iterations: 1

resource:
  num_gpus: 64
  colocated: true                 # 单 domain 推荐 colocated
```

## 性能调优建议

### 1. Rollout 优化

- **vLLM + PagedAttention**：使用 vLLM 作为 rollout engine，PagedAttention 显著减少 KV cache 显存碎片
- **FP8 推理**：GB200 的 FP8 Tensor Core 吞吐约为 BF16 的 2 倍，rollout 阶段启用 FP8 可大幅加速
- **Continuous batching**：动态合并不同长度的生成请求，提升 GPU 利用率

### 2. Training 优化

- **FSDP + gradient checkpointing**：对于 397B+ MoE 模型，FSDP 分片 + 激活重算是必需的
- **Flash Attention**：使用 `--attention-backend fused` 启用融合注意力
- **TransformerEngine FP8**：training 阶段同样启用 FP8，配合 `--fp8-format hybrid --fp8-recipe delayed`

### 3. 通信优化

- **MNNVL**：确保 TP/EP 组在同一 NVL72 domain 内（~840 GB/s），避免跨域通信
- **GIB**：所有 RDMA 通信使用 GIB 插件，通过 `source /usr/local/gib/scripts/set_nccl_env.sh` 自动配置
- **NCCL 环境变量**：
  ```bash
  export NCCL_MNNVL_ENABLE=2      # 域内自动启用 MNNVL
  export NCCL_CUMEM_ENABLE=1      # 启用 cuMem for MNNVL
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

### 4. 数据流优化

- **Async rollout-training pipeline**：在 colocated 模式下，前一个 batch 的 training 完成后立即启动下一个 batch 的 rollout，减少等待时间
- **GCSFuse v2 预加载**：训练数据（prompts）通过 GCSFuse v2 并行下载到 Local SSD 缓存，避免 I/O 瓶颈
- **Checkpoint 异步写入**：使用 `torch.distributed.checkpoint` 的 async API，ckpt 写入不阻塞训练

## 已知限制

1. **MoE + RL 的显存压力**：RL 训练需要同时维护 policy model 和 reference model（用于 KL 计算），对于 397B MoE 模型，64 GPU + FSDP 的显存刚好够用，建议 `micro_batch_size=1`
2. **Rollout 长度限制**：NVL72 域内 64 GPU 的总 HBM 限制了最大 rollout 序列长度。对于 agentic RL 的超长序列（>16K tokens），可能需要 KV cache offloading 到 Local SSD
3. **跨域 RL**：如需超过 64 GPU 的 RL 训练，rollout 的 alltoall 通信跨域后带宽从 ~840 GB/s 降至 ~165 GB/s，建议使用 disaggregated 模式将 rollout 和 training 分开
