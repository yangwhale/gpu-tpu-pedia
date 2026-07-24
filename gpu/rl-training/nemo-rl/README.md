# NeMo RL 在 GB300 NVL72 上的端到端 RL 后训练（GRPO）复刻指南

> **本文性质**：基于 NVIDIA 官方开源框架 [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) 的 **GB300 NVL72 RL 后训练（GRPO）端到端复刻指南**。含理论分析 + 环境搭建 + **从小到大的规模递进 recipe** + 官方 baseline 目标 + 一次跑通的完整命令。所有配置/命令均来自该 repo 的实际 recipe 文件（`examples/configs/recipes/llm/performance/` + `infra/nrl_k8s/examples/`）。
>
> **为什么选 NeMo RL**：在主流 RL 后训练框架里（verl / slime / AReaL / OpenRLHF / NeMo RL），**只有 NeMo RL 提供了命名带 `gb300` 的官方 recipe YAML**（4 节点 Qwen3-30B、48/64 节点 Ultra），且是 NVIDIA 亲自维护、对 GB300 NVL72 拓扑（KAI 调度 + DRA ComputeDomain/RoCE）做了完整适配。slime 次之（docker 里有 GB300 sgl-kernel 处理，靠 SGLang 后端）；verl 是 Blackwell-ready 但无 GB300 专属 recipe；AReaL/OpenRLHF 代码里搜不到 GB300 适配。

---

## 目录

0. [TL;DR + 规模递进总览](#0-tldr)
1. [理论篇：RL 后训练 + GRPO + 异步架构](#1-理论篇)
2. [为什么 GB300 适合 RL 后训练](#2-为什么-gb300)
3. [环境搭建：k8s + KAI + DRA + Lustre + Ray](#3-环境搭建)
4. [⭐ 从小到大：五阶段 recipe 递进](#4-从小到大)
5. [一次跑通：GB300 4 节点主线完整命令](#5-一次跑通)
6. [官方 baseline 目标指标](#6-baseline)
7. [关键配置深度详解](#7-关键配置)
8. [排错 + 复现 checklist](#8-排错)

---

## 0. TL;DR

**RL 后训练 = 用「生成 rollout → 打分 → 更新策略」的闭环，把 SFT 后的模型进一步对齐/提升推理能力（GRPO 是 DeepSeek-R1 用的算法）。** GB300 上跑它，瓶颈和纯推理/纯训练都不同：**它同时需要高效的 rollout 生成（vLLM）+ 高效的策略训练（Megatron），两者要么共卡（colocated）要么分离（disaggregated async）。**

**规模递进总览（从小到大，每步都能独立验证）：**

| 阶段 | 模型 | 规模 | recipe | 目的 |
|---|---|---|---|---|
| 0 冒烟 | 任意小模型 | 1 GPU | `grpo_smoke.yaml` | 跑通流程 |
| 0 单机 | DeepSeek-R1-Distill-Qwen-1.5B | 1×8 H100/GB300 | `grpo-deepscaler-1.5b-8K.yaml` | **收敛 baseline**（reward 0.65@400步）|
| 1 双机 | Llama3.1-8B | 2 节点 | `grpo-llama3.1-8b-instruct-2n4g.yaml` | 多节点 + async 入门 |
| 2 ⭐ GB300 入门 | Qwen3-30B-A3B (MoE) | **4 节点 GB300** | `grpo-qwen3-30ba3b-4n4g-async-1off.yaml` + `qwen3_30b_math_4n_4gpu.gb300.infra.yaml` | **GB300 主线**（EP8 + async 1-off + gen/train 分离）|
| 3 中大 | Qwen3-32B / Qwen3-235B | 8–32 节点 | `grpo-qwen3-235b-32n4g-async-1off.yaml` | 扩展 + mxfp8 rollout |
| 4 满配 | DeepSeek-V3 671B / Nemotron3-Super | 48–64 节点 | `grpo-deepseek-v3-64n8g.yaml` + `ultra_48n/64n_pipeclean.gb300.infra.yaml` | 满规模 + FP8 |

**一次跑通（GB300 4 节点主线）：**
```bash
RECIPE=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml
INFRA=infra/nrl_k8s/examples/qwen3_30b_math_4n_4gpu.gb300.infra.yaml
nrl-k8s run $RECIPE --infra $INFRA --wait
```

---

## 1. 理论篇

### 1.1 RL 后训练全景

训练一个推理模型分三步：**预训练（PT）→ 监督微调（SFT）→ 强化学习（RL）**。RL 阶段让模型通过「自己生成答案 → 用可验证的奖励打分 → 朝高分方向更新」自我提升，这是 DeepSeek-R1、o1 这类推理模型的核心。RL 后训练的闭环有四个角色：

- **Policy（策略模型 / 被训练者）**：要优化的 LLM，用 Megatron/DTensor 做分布式训练。
- **Generation（rollout 生成）**：用当前 policy 权重，对一批 prompt 生成多条回答（rollout）。用 **vLLM/SGLang** 高速推理。
- **Environment（环境）**：接收回答，返回奖励。NeMo RL 里是 Ray Remote Actor（如 `MathEnvironment`：判断数学答案对不对给 0/1 奖励）。支持多环境并存。
- **Reward / Advantage**：把奖励转成「优势」（advantage），告诉训练该往哪个方向推。

### 1.2 GRPO 算法（Group Relative Policy Optimization）

GRPO 是 DeepSeek 提出、R1 使用的 RL 算法，相比 PPO **去掉了独立的 value/critic 网络**，用「同一 prompt 生成一组回答、组内相对比较」来估计优势，省一半显存。

关键机制（NeMo RL `grpo_math_1B.yaml` 实配）：
- **`num_prompts_per_step` × `num_generations_per_prompt`**：每步取 N 个 prompt，每个生成 M 条回答（如 1B: 32×16；30B: 64×32）。同一 prompt 的 M 条回答构成一个 **group**。
- **Leave-one-out baseline**（`use_leave_one_out_baseline: true`）：每条回答的 baseline = 同组**其他** M-1 条的平均奖励 → 优势 = 自己奖励 − 组内其他人均值。组内相对，无需 critic。
- **`normalize_rewards: true`**：奖励标准化稳定训练。
- **`adv_estimator: grpo`**（可选 `reinforce_plus_plus`）。
- **Loss**：带 clip 的策略梯度（PPO-style ratio clip），可选 KL 惩罚。

### 1.3 系统架构：同步 vs 异步 GRPO（最关键）

RL 后训练最大的系统挑战：**generation（rollout）和 training（policy update）如何共处**。两种模式：

**A. 同步 + 共卡（colocated，默认小规模）**：gen 和 train 用**同一批 GPU**，交替执行——先生成、再训练、再生成。简单，但 GPU 有一半时间在等（生成时训练卡空转，反之亦然）。

**B. 异步 + 分离（disaggregated async，GB300 主线）**：gen 和 train 跑在**不同 GPU** 上并发。Rollout worker 持续生成，training worker 持续更新，重叠掉等待时间。NeMo RL 的 async GRPO 配置（`async-grpo.md` + `grpo-qwen3-30ba3b-4n4g-async-1off.yaml`）：

```yaml
grpo:
  async_grpo:
    enabled: true
    max_trajectory_age_steps: 1      # 1-off policy: rollout 最多比训练落后 1 步
    in_flight_weight_updates: true   # 训练更新的权重实时推给 gen(不停 gen)
loss_fn:
  use_importance_sampling_correction: true  # 收敛必需
  force_on_policy_ratio: true
policy:
  generation:
    colocated:
      enabled: false                 # 分离: gen 和 train 不共卡
      resources: {num_nodes: 2, gpus_per_node: 4}   # gen 占 2 节点
```

- **1-off policy（`max_trajectory_age_steps: 1`）**：rollout 用的权重最多比当前训练权重旧 1 步。完全 on-policy 会让 gen 等 train（慢），完全 off-policy 会发散。1-off 是吞吐与收敛的平衡点。
- **Importance sampling correction（必需）**：因为 rollout 权重和训练权重有 1 步差，用重要性采样比值修正梯度，否则异步会发散。这是 async GRPO 能收敛的关键。
- **In-flight weight updates**：训练完一步立刻把新权重推给 gen worker，不停生成。
- **为什么这对 MoE 特别重要**：MoE 模型 gen（vLLM，EP 宽并行）和 train（Megatron，EP+TP）的最优并行策略不同，分离部署可各自最优。

---

## 2. 为什么 GB300

- **NVLink/MNNVL 大域**：NVL72 一个域 18 节点 72 GPU 全 NVLink 互联。RL 里 gen→train 的**权重同步**（in-flight weight updates 每步推整个模型）+ MoE 的 EP all-to-all 都吃 NVLink 带宽，大域直接受益。
- **288GB HBM/GPU**：RL 要同时放 policy 权重 + 优化器状态（Megatron）+（gen 侧）KV cache，比纯推理更吃显存。288GB 让大模型少切并行。
- **FP8 / MXFP8 rollout**：GB300 FP4/FP8 张量核。NeMo RL 支持 **FP8 生成 + BF16 训练**（`docs/fp8.md`）和 **mxfp8-rollout**（`*-mxfp8-rollout.yaml`）——rollout 用低精度加速，训练保持 BF16 精度，兼顾速度与收敛。
- **8× RoCE NIC/节点**：跨节点 NCCL（train 的 EP/DP all-reduce）走 RoCE，通过 DRA 挂载。

---

## 3. 环境搭建

NeMo RL 在 GB300 上走 **Kubernetes + Ray** 架构（`nrl-k8s` CLI 自动编排）。核心组件：

| 组件 | 作用 | 来源 |
|---|---|---|
| **镜像** | `nvcr.io/nvidian/nemo-rl:<tag>` | NGC，含 vLLM + Megatron + Ray |
| **KAI scheduler** | gang 调度 + `gb300-topology`（`nvidia.com/gpu.clique` 作为 placement key，把一个 RayCluster 的 GPU 排进同一 NVLink 域）| infra yaml |
| **DRA ComputeDomain channel** | 跨节点 NVLink/MNNVL（一个 ComputeDomain 跨 16 GPU）| `claims: compute-domain-channel` |
| **DRA RoCE channel** | 4× RoCE NIC/节点跨节点 NCCL | `claims: roce-channel` |
| **Lustre PVC (`rl-workspace`)** | 共享 RWX（FSx Lustre 2.4TiB）：代码 `/opt/nemo-rl`、数据集/checkpoint/HF cache `/mnt/rl-workspace` | infra yaml volumes |
| **Ray Cluster** | head（CPU 节点）+ N 个 GPU worker，Ray Job SDK 提交 | kuberay |

**GB300 worker 资源（infra yaml 实配）**：每 GB300 节点 allocatable ~140 CPU / ~924GiB / 4 GPU，claim 132 CPU / 900GiB / 4 GPU（留余量给 device-plugin）。`NCCL_MNNVL_ENABLE=1`。

**前置（onboarding，一次性）**：
1. `rl-workspace` PVC 绑定（共享 Lustre，RWX）。
2. `/mnt/rl-workspace/${user}/nemo-rl` 有 `git clone https://github.com/NVIDIA-NeMo/RL`（branch main）——**Lustre 上改代码不用重构镜像**。
3. secrets：`nvcr-secret`（拉镜像）、`${user}-secrets`（HF/wandb token）、`nemo-rl-endpoint-registry` SA。
4. KAI 拓扑 `gb300-topology` 已注册，advertise `nvidia.com/gpu.clique`。

---

## 4. 从小到大

> **原则：先在单机跑通收敛（验证正确性 baseline），再逐级扩到 GB300 多节点（验证性能）。** 每一级都能独立验证，出错好定位。

### Stage 0 — 单机冒烟 + 收敛 baseline（不需要 GB300 集群）

**0a. 冒烟（最快 sanity，1 GPU）：**
```bash
uv run examples/run_grpo.py --config examples/configs/grpo_smoke.yaml
```

**0b. 单机 1B（1 GPU，跑通完整 GRPO）：**
```bash
uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml
# cluster: {gpus_per_node: 1, num_nodes: 1}; 32 prompts × 16 gen; vLLM 生成 + DTensor/FSDP 训练
```

**0c. ⭐ DeepScaleR 收敛 baseline（1×8 H100/GB300，三阶段 8K→16K→24K 上下文）：**
```bash
# 复刻 DeepSeek-R1-Distill-Qwen-1.5B 在 DeepScaleR 数据集上的 GRPO（DeepScaleR 官方 recipe）
uv run examples/run_grpo.py --config examples/configs/recipes/llm/grpo-deepscaler-1.5b-8K.yaml       # 240 步
uv run examples/run_grpo.py --config examples/configs/recipes/llm/grpo-deepscaler-1.5b-16K.yaml policy.model_name=/path/to/8K/checkpoint/hf   # 290 步
uv run examples/run_grpo.py --config examples/configs/recipes/llm/grpo-deepscaler-1.5b-24K.yaml policy.model_name=/path/to/16K/checkpoint/hf  # 50 步
```
**这是「正确性 baseline」**（见 §6）：~400 步达到 average training reward **0.65**，AIME24 pass@1 随训练上升。

### Stage 1 — 双节点多机（Llama3.1-8B）

```bash
# 2 节点 × 4 GPU，先同步版跑通，再切 async
RECIPE=examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n4g.yaml           # 同步
RECIPE=examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n4g-async-1off.yaml # async 1-off
# 也有 2n8g / 2n8g-fp8-async-1off 变体
```
目的：验证多节点 Ray + NCCL + async gen/train 分离能跑通，为 GB300 铺路。

### Stage 2 — ⭐ GB300 入门主线（Qwen3-30B-A3B MoE，4 节点 16 GPU）

**这是 GB300 的第一个正式 recipe**，也是本文主线。拓扑：4 worker × 4 GPU = 16 GPU 一个 RayCluster，KAI 当一个 gang 调度，NVLink/MNNVL 跨全 16 GPU（ComputeDomain channel），NCCL 走 4× RoCE。

recipe 关键配置（`grpo-qwen3-30ba3b-4n4g-async-1off.yaml`，继承 `grpo-qwen3-30ba3b-4n4g.yaml` → `grpo_math_qwen30ba3b_megatron.yaml`）：
```yaml
grpo:
  async_grpo: {enabled: true, max_trajectory_age_steps: 1, in_flight_weight_updates: true}
loss_fn: {use_importance_sampling_correction: true, force_on_policy_ratio: true}
policy:
  train_global_batch_size: 2048
  megatron_cfg:            # 训练侧并行
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    expert_model_parallel_size: 8   # EP8 —— MoE 专家并行
  generation:             # rollout 侧
    colocated: {enabled: false, resources: {num_nodes: 2, gpus_per_node: 4}}  # gen 占 2 节点
    vllm_cfg: {async_engine: true, tensor_parallel_size: 1, gpu_memory_utilization: 0.8}
cluster: {gpus_per_node: 4, num_nodes: 4, segment_size: 2}
```
- **gen/train 分离**：4 节点里 2 节点跑 vLLM 生成、2 节点跑 Megatron 训练。
- **EP8**：MoE 专家分 8 路（与我们 vLLM dep8 推理的宽-EP 同思路）。
- **segment_size=2**：NVLink 域分段对齐（通常调到 = EP size）。
- **mxfp8-rollout 变体**：`grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout.yaml`（rollout 用 MXFP8 提速）。

### Stage 3 — 中大规模（Qwen3-32B 稠密 / Qwen3-235B MoE）

```bash
# Qwen3-32B 稠密, 4–8 节点
grpo-qwen3-32b-4n4g.yaml / grpo-qwen3-32b-8n4g-async-1off.yaml
# Qwen3-235B MoE, 16–32 节点 (+ mxfp8 rollout)
grpo-qwen3-235b-16n4g.yaml / grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout.yaml
# Nemotron3-Super-120B-A12B, 32 节点
grpo-nemotron3-super-120BA12B-32n4g-async-1off.yaml
```

### Stage 4 — 满配（DeepSeek-V3 671B，48–64 节点 Ultra）

```bash
# DeepSeek-V3 671B MoE, 64 节点
RECIPE=examples/configs/recipes/llm/performance/grpo-deepseek-v3-64n8g.yaml
RECIPE=examples/configs/recipes/llm/performance/grpo-deepseek-v3-64n8g-fp8-async-1off.yaml  # FP8
RECIPE=examples/configs/recipes/llm/performance/dapo-deepseek-v3-64n8g.v2.yaml              # DAPO 算法
# 配套 GB300 Ultra infra:
INFRA=infra/nrl_k8s/examples/ultra_48n_pipeclean.gb300.infra.yaml   # 48 节点
INFRA=infra/nrl_k8s/examples/ultra_64n_pipeclean.gb300.infra.yaml   # 64 节点
```

---

## 5. 一次跑通

GB300 4 节点主线的**完整一次跑通命令**（含 auto-teardown）：
```bash
RECIPE=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml
INFRA=infra/nrl_k8s/examples/qwen3_30b_math_4n_4gpu.gb300.infra.yaml

# 一次性跑 + 跑完自动拆集群:
nrl-k8s run $RECIPE --infra $INFRA --wait

# 或长驻集群迭代 (调参时):
nrl-k8s cluster up $RECIPE --infra $INFRA --role training --wait
nrl-k8s run $RECIPE --infra $INFRA --raycluster --wait
```

infra yaml 的 `entrypoint` 实际在 head pod 里跑（可改步数做冒烟）：
```bash
python -u examples/run_grpo.py \
  --config examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off.yaml \
  grpo.max_num_steps=5 grpo.val_period=5 checkpointing.save_period=5 \
  logger.wandb_enabled=true logger.tensorboard_enabled=true logger.monitor_gpus=true \
  logger.wandb.name=qwen3-30b-math-gb300
```
> `grpo.max_num_steps=5` 是冒烟（快速验证跑通）；正式训练去掉或调大。`logger.monitor_gpus=true` 会记录 GPU 利用率/MFU。

**关键**：CLI 会**自动创建并自动删除** ComputeDomain + RoCE ResourceClaimTemplate（根据 pod spec 的 `resourceClaims`），不用手动建 DRA 资源。

---

## 6. Baseline

RL 后训练的 baseline 分两类，**都要盯**：

**A. 正确性 / 收敛 baseline（模型是否学到东西）：**
- **DeepScaleR（1.5B）**：~400 步达到 average training reward **0.65**；AIME24 pass@1 随训练稳定上升（官方 `docs/guides/grpo-deepscaler.md` 有训练曲线图 `assets/deepscaler_training_progress.png`）。**这是复刻正确性的金标准——先在单机把这条曲线跑出来，证明 pipeline 对。**
- 大模型 recipe（Qwen3-30B/235B、DeepSeek-V3）：reward 曲线应单调上升、不发散（async 发散通常是 importance sampling correction 没开）。

**B. 性能 baseline（GB300 跑多快，这是我们端到端要追的目标）：**
- **NVIDIA 的 perf recipe 本身就是 baseline**：`examples/configs/recipes/llm/performance/` 下的配置是 NVIDIA 在 GB300 上**调优过**的目标配置。跑起来后从 **wandb / tensorboard** 读：
  - **step time**（每训练步秒数）——最直接的吞吐指标
  - **generation throughput**（rollout tok/s）+ **training throughput**（train tok/s）
  - **MFU**（`monitor_gpus=true` 记录）+ gen/train GPU 利用率
  - **gen:train 平衡**（async 下两者应重叠、都不闲——参考 SemiAnalysis《RL Systems Mind the Gap》讲的 trainer/generator throughput matching）
- **口径提醒**：RL 的吞吐 = rollout 生成 + 策略训练的**端到端每步时间**，不是单纯推理 tok/s。async 模式的目标是 gen 和 train 完全重叠、GPU 无空转。

> **建议**：先跑 Stage 0c DeepScaleR 拿到正确性 baseline（reward 0.65@400），再跑 Stage 2 GB300 4 节点，用 wandb 记录 step time / MFU 作为性能 baseline，之后逐级扩大对比。

---

## 7. 关键配置

| 配置 | 值（30B GB300）| 含义 |
|---|---|---|
| `async_grpo.enabled` | true | 异步：gen/train 并发 |
| `max_trajectory_age_steps` | 1 | 1-off policy（rollout 最多旧 1 步）|
| `in_flight_weight_updates` | true | 训练权重实时推 gen |
| `use_importance_sampling_correction` | true | **async 收敛必需**（不开会发散）|
| `colocated.enabled` | false | gen/train 分离部署 |
| `generation.resources` | 2 节点×4 GPU | gen 侧独占资源 |
| `megatron_cfg.expert_model_parallel_size` | 8 | 训练侧 EP8（MoE）|
| `vllm_cfg.async_engine` | true | vLLM 异步引擎（rollout）|
| `vllm_cfg.moe_backend` | triton | refit 兼容（权重热更新）|
| `cluster.segment_size` | 2（通常=EP size）| NVLink 域分段对齐 |
| `kuberay.segmentSize` | ≤18（NVL72）| 拓扑约束，通常调到 EP size |
| mxfp8-rollout 变体 | — | rollout MXFP8 提速，train 保 BF16 |

**gen/train 分离的资源账（4 节点 16 GPU）**：2 节点（8 GPU）跑 vLLM 生成，2 节点（8 GPU）跑 Megatron 训练。async 让两边并发，避免共卡时的一半空转。

---

## 8. 排错

| 症状 | 可能原因 | 排查 |
|---|---|---|
| async 训练发散（reward 崩）| `use_importance_sampling_correction` 没开 | 必开 + `force_on_policy_ratio` |
| pod 排不上 / 拓扑错 | KAI `gb300-topology` 未注册 / `nvidia.com/gpu.clique` 缺 | 查 KAI 调度器 + 节点标签 |
| 跨节点 NCCL 挂 | ComputeDomain / RoCE DRA claim 没分上 | 查 `claims: compute-domain-channel/roce-channel` + `NCCL_MNNVL_ENABLE=1` |
| HF 下载失败 | HF_HOME/token 未配 | `${user}-secrets` + `HF_HOME=/mnt/rl-workspace/${user}/hf-cache` |
| 权重同步慢 | in-flight 没开 / 域跨 rack | `in_flight_weight_updates: true` + `segmentSize` 对齐 EP |
| gen 或 train 一方空转 | gen:train 资源比失衡 | 调 `generation.resources` 节点数，追 gen/train throughput 平衡 |

**复现 checklist（GB300 4 节点主线）：**
1. ☐ 前置：Lustre PVC 绑定 + repo clone 到 `/mnt/rl-workspace/${user}/nemo-rl` + secrets + KAI 拓扑
2. ☐ 先单机 Stage 0c DeepScaleR 跑出 reward 0.65@400（验证 pipeline 正确）
3. ☐ `nrl-k8s run $RECIPE --infra $INFRA --wait` 起 GB300 4 节点，先 `max_num_steps=5` 冒烟
4. ☐ 冒烟过 → 去掉 step 限制正式训练，wandb 记 step time / MFU / reward
5. ☐ 逐级扩：4n → 8n → 16n(235B) → 64n(DeepSeek-V3)

---

## 参考

- NeMo RL repo：https://github.com/NVIDIA-NeMo/RL
- GRPO 指南：`docs/guides/grpo.md`；Async GRPO：`docs/guides/async-grpo.md`
- DeepScaleR 复刻：`docs/guides/grpo-deepscaler.md`（正确性 baseline）
- FP8：`docs/fp8.md`；集群搭建：`docs/cluster.md`
- GB300 infra：`infra/nrl_k8s/examples/*.gb300.infra.yaml`
- GB300 perf recipes：`examples/configs/recipes/llm/performance/`
- 跨框架 RL 对比：SemiAnalysis《RL Systems Mind the Gap》(2026-06)

> **状态**：本文为 recipe 复刻 + 理论整理，**尚未在本环境实跑**。下一步：按 §4 从 Stage 0c 单机收敛 baseline 起，逐级验证到 GB300 4 节点，把实测 step time/MFU/reward 曲线补进 §6。
