> 🌐 [中文](README.md) | **English**

# RL Training Guide: Reinforcement Learning on GB200 NVL72

> **Note**: This chapter is **supplementary content** and does not come from the original GB200 A4X deployment guide document. The content is compiled from research on RL training best practices, combined with the hardware characteristics of the GB200 NVL72, and is provided for reference only.

This chapter provides a practical guide to large-scale RL (Reinforcement Learning) training on a GB200 NVL72 cluster, covering framework selection, parallelism strategies, model configuration, and tuning recommendations.

## Hardware Overview

| Parameter | Value |
|------|-----|
| GPU model | NVIDIA GB200 (Blackwell, sm_100) |
| GPUs per node | 4 |
| NVL72 Domain | 18 nodes / 72 GPUs |
| Available GPUs (after holdback) | **64 GPUs** (16 active nodes, 2 spare nodes) |
| Intra-domain interconnect | NVLink 5th gen (MNNVL), ~840 GB/s |
| Cross-domain interconnect | RDMA over CX-7, ~325 GB/s |
| Memory | HBM3e per GPU |

**Holdback policy**: Of the 72 GPUs, 8 GPUs (2 nodes) are reserved as spare capacity (see [08-multi-domain](../08-multi-domain/) Section 10), leaving 64 GPUs actually available for training.

## Framework Selection

### veRL (Recommended)

[veRL](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning) is an open-source large-model RL training framework from ByteDance, supporting algorithms such as GRPO, PPO, and REINFORCE.

**Reasons for recommendation**:
- Native support for both colocated and disaggregated deployment modes
- Deep integration with Megatron-LM and vLLM
- Support for RL training of MoE models
- Active community and documentation

### AReaL

[AReaL](https://github.com/inclusiveai/AReaL) (Alibaba Reinforcement Learning) is an open-source RL training framework from Alibaba, focused on large-scale reasoning model training.

**Applicable scenarios**:
- Existing integration with Alibaba Cloud infrastructure
- Need for reasoning-oriented RL (e.g., math, code tasks)

## Model Selection

| Model | Parameters | Type | Experts | Active Params | Recommended Scenario |
|------|--------|------|---------|---------------|----------|
| **Qwen 3.5 397B** | 397B | MoE | 128 experts, topk=8 | ~50B | RL within a single domain (64 GPUs) |
| **DeepSeek V3 671B** | 671B | MoE | 256 experts, topk=8 | ~37B | Multi-domain RL, requiring larger model capacity |

**Advantages of MoE models**: In RL training, the computational load of the rollout phase is far greater than that of the training phase. The sparse activation characteristic of MoE models (only topk experts active) offers a significant advantage in rollout throughput.

## Parallelism Strategy

### Recommended Configuration (64 GPUs, single domain)

```
TP=8   EP=8   DP=8   FSDP for training
```

| Parallelism dimension | Size | Mapping | Description |
|----------|------|------|------|
| **TP** (Tensor Parallel) | 8 | 2 nodes x 4 GPU/node | Spans 2 adjacent nodes, using MNNVL (~840 GB/s) |
| **EP** (Expert Parallel) | 8 | Shares the same group of 8 GPUs with TP | EP and TP share physical GPUs; MoE layers use EP, attention layers use TP |
| **DP** (Data Parallel) | 8 | 64 / (TP x PP) = 8 | Across different TP groups, gradient synchronization |
| **FSDP** | Enabled | On the DP dimension | ZeRO-3 level sharding of parameters/gradients/optimizer states |

**TP and EP share GPUs**: In MoE models, TP and EP typically share the same group of physical GPUs. Attention/FFN layers are partitioned with TP, while MoE layers are partitioned with EP. This avoids extra inter-GPU data movement.

### Why Not Use PP

Pipeline Parallel introduces additional bubble overhead in RL training, and RL's alternating rollout-and-training execution pattern is incompatible with PP's pipeline filling. For the 64-GPU scale, the TP+EP+DP+FSDP combination is already sufficient.

> **TP=8 intra-domain placement warning**: TP=8 requires that the 2 participating nodes (4 GPU/node x 2 = 8 GPUs) reside within **the same NVL72 domain**. Nodes within a domain are interconnected via MNNVL, with a bandwidth of about **~840 GB/s**. If a TP group spans different domains, communication falls back to RDMA (about **~325 GB/s**, a 61% drop in bandwidth), and TP's high-frequency AllReduce communication (2 per layer) becomes a severe bottleneck, causing a substantial degradation in training performance. Use JobSet + Kueue TAS or a manual nodeSelector to ensure that the nodes within a TP group land in the same topology domain.

## Deployment Modes

### Colocated (Recommended, single domain)

**Colocated mode** deploys rollout (generation) and training (backpropagation) on the same group of GPUs, executing them alternately.

```
┌─────────── 64 GPU (single domain) ───────────┐
│                                            │
│  Phase 1: Rollout (vLLM inference)         │
│  ├── Load model weights onto all GPUs      │
│  ├── Generate responses (batch 1024-2048)  │
│  └── Collect trajectories                  │
│                                            │
│  Phase 2: Training (Megatron/FSDP)         │
│  ├── Compute rewards                       │
│  ├── GRPO/PPO update                       │
│  └── Sync updated weights                  │
│                                            │
│  → Alternate between Phase 1 and Phase 2   │
└────────────────────────────────────────────┘
```

**Advantages**:
- No need to transfer model weights across GPU groups
- The HBM of all GPUs is used for the same phase, yielding high utilization
- Suitable for single-domain scale (64-72 GPUs)

**Disadvantages**:
- Rollout and Training cannot run simultaneously; GPU utilization is limited by the slower phase

### Disaggregated (Multi-domain scaling)

When the training scale needs to exceed 1 domain, rollout and training can be allocated to different GPU groups.

```
┌──── Domain 0: Training (64 GPU) ────┐   ┌──── Domain 1: Rollout (64 GPU) ────┐
│  Receive trajectories                 │   │  Generate responses                 │
│  GRPO/PPO update                      │←→│  Send trajectories                   │
│  Send updated weights                 │   │  Receive updated weights             │
└──────────────────────────────────────┘   └──────────────────────────────────────┘
```

**Applicable scenarios**: When the rollout computation load is far greater than that of training (e.g., agentic RL), more GPUs can be used for rollout to accelerate overall throughput.

## GRPO Configuration Recommendations

[GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) is the current mainstream RL algorithm, which eliminates the critic model compared to PPO.

### Core Hyperparameters

| Parameter | Recommended Value | Description |
|------|--------|------|
| `batch_size` | 1024-2048 | Number of prompts per RL iteration |
| `mini_batch_splits` | 2-4 | Split the batch into mini-batches for gradient accumulation |
| `num_iterations` | 1 | Number of policy updates per batch (GRPO typically uses 1) |
| `max_new_tokens` | 2048-8192 | Maximum number of tokens generated during rollout (agentic tasks need longer) |
| `temperature` | 0.7-1.0 | Sampling temperature, controls exploration |
| `group_size` | 4-8 | Number of responses generated per prompt (used for within-group relative ranking) |
| `kl_coeff` | 0.01-0.1 | KL divergence penalty coefficient |
| `clip_range` | 0.2 | PPO-style clipping |

### Rollout:Training Ratio

| Task Type | Recommended Ratio | Description |
|----------|----------|------|
| Standard RL (math, code) | 2:1 to 3:1 | Rollout generates shorter sequences, training updates more frequently |
| **Agentic RL** (tool use, multi-turn) | **5:1 to 8:1** | Rollout requires multiple rounds of tool calls, generating extremely long sequences |

**Agentic RL note**: The rollout of agentic tasks (such as tool-use and multi-turn reasoning) involves multiple rounds of tool calls, and the token generation volume per rollout is far greater than in standard RL. A higher rollout:training ratio is needed to avoid the training GPUs idling.

## veRL Configuration Example

The following is a reference configuration for GRPO training of Qwen 3.5 397B on 64 GPUs using veRL:

```yaml
# verl_config.yaml
trainer:
  total_epochs: 3
  save_freq: 50
  test_freq: 10

rollout:
  name: vllm
  tp: 8                          # Same as training TP
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
  colocated: true                 # Colocated recommended for single domain
```

## Performance Tuning Recommendations

### 1. Rollout Optimization

- **vLLM + PagedAttention**: Use vLLM as the rollout engine; PagedAttention significantly reduces KV cache memory fragmentation
- **FP8 inference**: The GB200's FP8 Tensor Core throughput is about 2x that of BF16; enabling FP8 during the rollout phase can substantially accelerate it
- **Continuous batching**: Dynamically merge generation requests of different lengths to improve GPU utilization

### 2. Training Optimization

- **FSDP + gradient checkpointing**: For 397B+ MoE models, FSDP sharding + activation recomputation is required
- **Flash Attention**: Use `--attention-backend fused` to enable fused attention
- **TransformerEngine FP8**: Also enable FP8 during the training phase, together with `--fp8-format hybrid --fp8-recipe delayed`

### 3. Communication Optimization

- **MNNVL**: Ensure TP/EP groups are within the same NVL72 domain (~840 GB/s) to avoid cross-domain communication
- **GIB**: All RDMA communication uses the GIB plugin, automatically configured via `source /usr/local/gib/scripts/set_nccl_env.sh`
- **NCCL environment variables**:
  ```bash
  export NCCL_MNNVL_ENABLE=2      # Automatically enable MNNVL within a domain
  export NCCL_CUMEM_ENABLE=1      # Enable cuMem for MNNVL
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

### 4. Data Flow Optimization

- **Async rollout-training pipeline**: In colocated mode, immediately start the rollout of the next batch once the training of the previous batch completes, reducing wait time
- **GCSFuse v2 prefetching**: Training data (prompts) is downloaded in parallel to the Local SSD cache via GCSFuse v2, avoiding I/O bottlenecks
- **Async checkpoint writing**: Use the async API of `torch.distributed.checkpoint` so that ckpt writes do not block training

## Known Limitations

1. **Memory pressure of MoE + RL**: RL training requires maintaining both the policy model and the reference model (for KL computation) at the same time. For the 397B MoE model, 64 GPUs + FSDP provides just enough memory; `micro_batch_size=1` is recommended
2. **Rollout length limit**: The total HBM of 64 GPUs within an NVL72 domain limits the maximum rollout sequence length. For the ultra-long sequences of agentic RL (>16K tokens), KV cache offloading to Local SSD may be required
3. **Cross-domain RL**: If RL training beyond 64 GPUs is needed, the bandwidth of rollout's alltoall communication drops from ~840 GB/s to ~165 GB/s once it crosses domains; using disaggregated mode to separate rollout and training is recommended
