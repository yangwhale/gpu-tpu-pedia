# 腾讯混元 Hy3 (295B-A21B) 64 GPU 训练 — GB300 NVL72 (A4X Max) GKE

GB300 (A4X Max) GKE 集群上的 **Hy3（混元 3，295B 总参 / 21B 激活 / 80 层 + 1 MTP）** 16 节点 64 GPU 训练 benchmark 准备文档。

> **当前状态：实战进行中**（2026-07-25 22:30 HKT 开跑）。实战日志见 **§九**，配置设计见 §一~§三。
>
> 对标同级：DeepSeek V3 671B 见 [`07e-gb300-deepseekv3-671b-gke/`](../07e-gb300-deepseekv3-671b-gke/)（已跑通 ~1658 TFLOP/s，官方 99.3%），Qwen3 235B 见 [`07d-gb300-qwen3-235b-gke/`](../07d-gb300-qwen3-235b-gke/)。

**参考**：
- 模型卡：[huggingface.co/tencent/Hy3](https://huggingface.co/tencent/Hy3) · 许可证 Apache 2.0
- 官方训练文档：[`finetune/README.md`](https://huggingface.co/tencent/Hy3/blob/main/finetune/README.md)
- Megatron-Bridge 模型桥：[`models/hy_v3/hy_v3_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/hy_v3/hy_v3_bridge.py)
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)

---

## 结论先行：Hy3 训练支持现状

跑之前必须先搞清楚三件事，否则会照着 DSV3 的路子走进死胡同。

| 问题 | 答案 | 影响 |
|------|------|------|
| 腾讯官方有 Megatron recipe 吗？ | **没有**。官方只提供 DeepSpeed / LLaMA-Factory / ms-swift 三套 **SFT** 栈 | 不能指望官方脚本，Megatron 侧要自己写 |
| Megatron-Bridge 支持 Hy3 吗？ | **模型层支持，perf recipe 不支持**。`HYV3Bridge` 已注册（`HYV3ForCausalLM` → `GPTModel`），HF↔Megatron 双向权重映射（含 MTP 层）齐全；但 `recipes/` 下**没有** hunyuan/hy_v3 目录 | 权重转换开箱即用；并行/性能配置要自研 |
| 能照抄 DSV3 recipe 吗？ | **MoE 部分能，Attention 部分不能** | Hy3 MoE 是 DSV3 血统（见下），但 attention 是 GQA 不是 MLA |

### 为什么 MoE 能照抄 DSV3

Hy3 的 MoE 是 **DeepSeek V3 配方的一比一移植**，连 config 字段名都同源：

| DeepSeek V3 设计 | V3 字段 | Hy3 字段 | Qwen3 |
|---|---|---|---|
| sigmoid 路由 | `scoring_func: sigmoid` | `moe_router_use_sigmoid: true` | ❌ softmax |
| aux-loss-free 均衡（per-expert bias） | `topk_method: noaux_tc` | `moe_router_enable_expert_bias: true` | ❌ |
| shared expert | `n_shared_experts: 1` | `num_shared_experts: 1` | ❌ |
| MTP | `num_nextn_predict_layers: 1` | `num_nextn_predict_layers: 1` | ❌ |
| 前 k 层 dense | `first_k_dense_replace: 3` | `first_k_dense_replace: 1` | ❌ |
| routed scaling | `routed_scaling_factor: 2.5` | `router_scaling_factor: 2.826` | ❌ |

**实操含义**：DSV3 recipe 的 MoE 旋钮（hybridep 后端、`moe_a2a_overlap`、`cutedsl_fused_grouped_mlp`、`moe_paged_stash`、EP 甜点值）**可以直接迁移**；attention 相关（MLA 的 `q_lora_rank`/`kv_lora_rank`/`qk_rope_head_dim`）**全部作废**，换成 GQA 的 `num_query_groups=8` + `kv_channels=128` + `qk_layernorm=True`。

---

## 一、模型规格与 Megatron 参数映射

HF `config.json`（`tencent/Hy3`）→ Megatron `GPTModelProvider`。映射依据 `HYV3Bridge.provider_bridge()` 源码，**不是猜的**。

### 1.1 结构参数（直接映射）

| Megatron provider | 值 | HF config 来源 |
|---|---|---|
| `num_layers` | **80** | `num_hidden_layers` |
| `hidden_size` | **4096** | `hidden_size` |
| `ffn_hidden_size` | **13312** | `intermediate_size`（仅 dense 层用） |
| `num_attention_heads` | **64** | `num_attention_heads` |
| `num_query_groups` | **8** | `num_key_value_heads`（GQA） |
| `kv_channels` | **128** | `head_dim` |
| `seq_length` | 4096（benchmark）/ 262144（全长） | `max_position_embeddings` |
| `vocab_size` | **120832** | `vocab_size` |
| `rotary_base` | **11158840.0** | `rope_parameters.rope_theta` |
| `normalization` | `RMSNorm` | 固定 |
| `gated_linear_unit` | `True` | `hidden_act: silu` |
| `add_bias_linear` / `add_qkv_bias` | `False` / `False` | Hy3 无 QKV bias |
| `qk_layernorm` | **`True`** | `qk_norm: true` |
| `untie_embeddings_and_output_weights` | `True` | `tie_word_embeddings: false` |

### 1.2 MoE 参数（DSV3 血统）

| Megatron provider | 值 | HF config 来源 |
|---|---|---|
| `num_moe_experts` | **192** | `num_experts` |
| `moe_router_topk` | **8** | `num_experts_per_tok` |
| `moe_ffn_hidden_size` | **1536** | `moe_intermediate_size` |
| `moe_shared_expert_intermediate_size` | **1536** | `moe_intermediate_size × num_shared_experts` (1536×1) |
| `moe_layer_freq` | `[0]*1 + [1]*79` | `first_k_dense_replace=1` → 第 0 层 dense，1-79 层 MoE |
| `moe_router_score_function` | **`sigmoid`** | `moe_router_use_sigmoid` |
| `moe_router_enable_expert_bias` | **`True`** | `moe_router_enable_expert_bias` |
| `moe_router_pre_softmax` | `False` | 固定 |
| `moe_router_topk_scaling_factor` | **2.826** | `router_scaling_factor` |
| `moe_router_dtype` | `fp32` | 固定（路由数值稳定性） |
| `moe_grouped_gemm` | `True` | 固定 |
| `moe_permute_fusion` | `True` | 固定 |
| `mtp_num_layers` | **1** | `num_nextn_predict_layers` |

### 1.3 ⚠️ 训练必须覆盖的三个 bridge 默认值

`HYV3Bridge` 的默认值是给**权重转换 / 推理**用的，直接拿去 **from-scratch 预训练会导致专家负载失衡**：

| provider 字段 | Bridge 默认 | **训练应设** | 原因 |
|---|---|---|---|
| `moe_router_bias_update_rate` | `0` | **`1e-3`** | =0 则 expert bias 永不更新，aux-loss-free 均衡机制**形同虚设**（DeepSeek V3 论文用 0.001） |
| `moe_router_load_balancing_type` | `"none"` | `"none"`（保持）+ 上面的 bias update | aux-loss-free 路线就是 `none` + bias；**不要**改成 `aux_loss`，那会和 sigmoid+bias 打架 |
| `moe_aux_loss_coeff` | `0.0` | `0.0`（保持），可选 `seq_aux_loss_coeff=1e-4` 兜底 | V3 用极小的 sequence-level aux loss 做保险 |
| `moe_token_dispatcher_type` | `"alltoall"` | **`"flex"` + `moe_flex_dispatcher_backend="hybridep"`** | GB300 NVL72 上 hybridep 显著优于朴素 alltoall（见 07e） |

> **微调（SFT）场景例外**：加载官方权重做 SFT 时，专家路由已训好，`bias_update_rate` 保持 0 更稳（避免扰动已收敛的路由）。**只有 from-scratch / continued-pretrain 才需要开**。

### 1.4 参数量核对（验证配置对不对）

| 组成 | 计算 | 参数量 |
|---|---|---|
| 路由专家 | 79 层 × 192 expert × 3 × 4096 × 1536 | **286.2 B** |
| 共享专家 | 79 层 × 3 × 4096 × 1536 | 1.49 B |
| Attention | 80 层 × (Q 4096×8192 + K/V 2×4096×1024 + O 8192×4096) | 6.04 B |
| Dense FFN（第 0 层） | 3 × 4096 × 13312 | 0.16 B |
| Embedding + LM head | 2 × 120832 × 4096（untied） | 0.99 B |
| **合计** | | **≈ 294.9 B** ✓ 对上官方 295B |

> **97% 的参数在专家里** → **EP 是唯一有意义的显存旋钮**，TP 对这个模型几乎无用（attention 只占 2%）。

---

## 二、64 GPU（16 节点）并行策略设计

### 2.1 拓扑：单 NVLink Domain

GB300 一个 NVL72 域 ≤ 18 节点。**16 节点 = 64 GPU 正好装进一个 subblock**，比 07e 的 4-domain 256 GPU 简单得多：

```
64 GPU = 1 domain × 16 节点 × 4 GPU
┌──────────────────────────────────┐
│  subblock A  (ComputeDomain yw-cd-a) │
│  16 node × 4 GPU = 64 GPU            │
│  全部 NVLink (MNNVL) 互联            │
│  → EP 最大可开到 64（全域）          │
└──────────────────────────────────┘
```

**优势**：无跨域 RDMA，EP all-to-all 全走 NVLink；只需 1 个 ComputeDomain，规避 07e 里 4-CD clique 死结那一堆坑。

### 2.2 并行度约束

`world = TP × PP × DP`，且 **EP 必须整除 DP**（TP=1 时）：

| TP | PP | DP = 64/(TP×PP) | EP 可选 | 每 rank 专家数 (192/EP) |
|---|---|---|---|---|
| 1 | 2 | 32 | 8 / 16 / **32** | 6 |
| 1 | 4 | 16 | 8 / **16** | 12 |
| 1 | 8 | 8 | **8** | 24 |

> 07e 实测结论：**EP=32 是甜点**，EP=64 反而低 1.3%（all-to-all 跨度变大 + 每 rank 专家太少）。EP=8 每 rank 扛 24 个专家，显存吃紧。

### 2.3 起步配置（推荐 V1）

| 参数 | 值 | 理由 |
|---|---|---|
| TP | **1** | attention 仅占 2% 参数，TP 纯亏通信 |
| PP | **2** | 80 层 / 2 = 40 层/stage；PP 越小 bubble 越小，288GB 装得下 |
| VPP | **8** | 16 个 chunk，摊薄 bubble（对齐 DSV3 的 PP2×VPP8） |
| EP | **32** | 07e 验证的甜点；192/32 = 6 专家/rank |
| MBS | **1** | 首跑保守，跑通后试 2 |
| GBS | **2048** 起步 | 须被 `MBS × DP` = 1×32 整除；64 microbatch → bubble 仅 0.2%（见 §2.4） |
| 精度 | **BF16**（首跑） | 对齐官方口径；MoE 上 FP8 收益存疑，见 §三 |
| cuda_graph_impl | **`full_iteration`** | 07e 的核心成果，勿手动覆盖 |
| mtp_num_layers | **首跑设 0，跑通后再开 1** | 见下方 pp_layout 说明 |

### 2.4 BF16 显存测算 — 64 卡够用，MBS=1 安全

计算器：[`mem_calc.py`](mem_calc.py)（`python3 mem_calc.py` 扫网格 / `--detail N PP EP MBS` 看明细）。

#### 三条公式

```
每 GPU 专家参数   = 286.2B / (PP × EP)        专家被 PP 和 EP 双重切分
每 GPU 非专家参数 = 8.68B  / PP               EP 组内复制，只被 PP 切
优化器            = 294.9B × 12B / N          ← EP 会约掉，只跟总卡数有关
```

> **最反直觉的一条**：专家优化器 = `[286.2/(PP·EP)] × 12 / (DP/EP)` — 专家的 DP 组是 `DP/EP`，**EP 上下约掉**。
> 所以 **调 EP 只影响权重和梯度，压不动优化器**。优化器是硬地板，想降只能加卡：64 卡 51.5 GB、128 卡 25.7 GB。

#### 运行时开销标定（关键，别漏）

朴素的「权重+梯度+优化器+激活」会严重低估。用 07e 的实测锚点反标定：

| | DSV3 31L / 128 GPU / PP2 VPP8 EP64 / MBS1 / MXFP8 |
|---|---|
| 朴素四项合计 | 84.5 GB |
| **07e 实测 max reserved** | **113 GB** |
| **标定系数** | **1.34×** |

差的 28.5 GB = CUDA graph buffer + EP dispatch/combine buffer + NCCL buffer + 碎片。下表已乘 1.34。

> 单锚点标定，误差可能 ±15%。**Hy3 首跑后应重新标定**（记录 `torch.cuda.max_memory_reserved()`）。

#### 64 GPU 配置网格（BF16, seq=4096, VPP=8）

安全线取 288 × 0.90 = **259 GB**（OOM 通常发生在 graph capture 峰值，留 10%）。

| PP | EP | MBS | 权重 | 梯度 | 优化器 | 激活 | 朴素 | **实际** | 判定 |
|----|----|----|------|------|--------|------|------|--------|------|
| 2 | 32 | **1** | 16.4 | 32.8 | 51.5 | 45.2 | 145.9 | **195.5** | ✅ **推荐** |
| 2 | 32 | 2 | 16.4 | 32.8 | 51.5 | 90.3 | 191.0 | **256.0** | ✅ 贴线（259） |
| 2 | 16 | 1 | 24.7 | 49.5 | 51.5 | 45.2 | 170.9 | **229.0** | ✅ 安全 |
| 2 | 16 | 2 | 24.7 | 49.5 | 51.5 | 90.3 | 216.0 | **289.5** | ❌ OOM |
| 4 | 16 | 1 | 12.4 | 24.7 | 51.5 | 46.5 | 135.1 | **181.0** | ✅ 安全 |
| 4 | 16 | **2** | 12.4 | 24.7 | 51.5 | 93.0 | 181.6 | **243.3** | ✅ **MBS=2 最优选** |
| 4 | 8 | 2 | 20.7 | 41.4 | 51.5 | 93.0 | 206.6 | **276.8** | ⚠️ 超安全线 |
| 8 | 8 | 2 | 10.4 | 20.7 | 51.5 | 94.3 | 176.8 | **237.0** | ✅ 但 PP=8 bubble 大 |

#### 最少需要多少卡

| GPU | 节点 | 最省配置 | 优化器地板 | 实际 GB | 判定 |
|-----|------|---------|-----------|--------|------|
| 16 | 4 | PP=8 EP=2 | 206.0 | 481.2 | ❌ OOM |
| 24 | 6 | PP=8 EP=3 | 137.3 | 344.6 | ❌ OOM |
| **32** | 8 | PP=8 EP=4 | 103.0 | **276.3** | ⚠️ 塞得进 288 但无余量 |
| **40** | 10 | PP=8 EP=5 | 82.4 | **235.3** | ✅ **理论下限** |
| 48 | 12 | PP=8 EP=6 | 68.7 | 207.9 | ✅ 安全 |
| **64** | **16** | PP=2 EP=32 | 51.5 | **195.5** | ✅ **推荐（本次）** |
| 128 | 32 | PP=4 EP=32 | 25.7 | 129.8 | ✅ 很宽裕 |

> **结论：BF16 训 Hy3 理论最少 40 卡（10 节点），实操推荐 64 卡（16 节点）。**
> 32 卡虽然算下来 276 GB 能塞进 288，但零余量 + 需要 PP=8（bubble 大、48 专家/rank），不值得。
> 64 卡不只是"够"，是有 60+ GB 余量可以开 MBS=2 或加 MTP。

#### GBS 该开多大

约束：**GBS 必须被 `MBS × DP` 整除**。VPP=8 下 bubble = `(PP-1)/(microbatch数 × VPP)`。

推荐配置 **PP=2 EP=32 MBS=1（DP=32）**：

| GBS | microbatch/rank | pipeline bubble | tokens/step | 评价 |
|-----|----------------|-----------------|-------------|------|
| 1024 | 32 | 0.39% | 4.2 M | 偏小 |
| **2048** | 64 | **0.20%** | **8.4 M** | ✅ **首跑用这个** |
| 4096 | 128 | 0.10% | 16.8 M | 对标 DSV3 官方档 |
| 8192 | 256 | 0.05% | 33.6 M | 扫点上限 |

> **PP=2 + VPP=8 下 bubble 已经可以忽略**（GBS=2048 时才 0.2%），所以 07e 里"GBS 是最大旋钮"那条结论**在这个配置下不成立** —— 那是 DSV3 PP=2 但 256 卡 DP=128 的场景。
> 这里 GBS 往上加主要收益是**摊薄固定开销**（optimizer step、all-reduce、GC），不是消 bubble。首跑 2048 起，扫到 4096/8192 看是否还有增益。

### 2.5 pp_layout 推导（80 层 vs DSV3 61 层）

DSV3 61 层 PP2×VPP8 = 16 chunk 的布局是 `Et*4|(t*4|)*14tmL`：
- chunk 1: `E` + 4 层 → chunks 2-15: 14 × 4 层 = 56 → chunk 16: 1 层 + `m`(MTP) + `L`(loss)
- 合计 4 + 56 + 1 = **61** ✓，chunk 数 1 + 14 + 1 = **16** ✓

Hy3 80 层同样 16 chunk → 平均 5 层/chunk：

```
不带 MTP（首跑推荐）：  Et*5|(t*5|)*14t*5L      → 5 + 70 + 5 = 80 ✓
带 MTP：                Et*5|(t*5|)*14t*5mL     → 5 + 70 + 5 = 80 ✓（末 chunk 额外扛 m + L）
```

> ⚠️ **末 stage 负载不均风险**：DSV3 故意让末 chunk 只放 1 层，因为 MTP + loss head 很重。Hy3 均分 5 层后末 chunk 要扛 5 层 + MTP + loss，**可能 OOM 或成为 bubble 瓶颈**。
> **备选布局**（前重后轻）：`Et*6|(t*5|)*14t*4mL` → 6 + 70 + 4 = 80 ✓
> **这两个布局都是推导值，未实测**。首跑建议先 `mtp_num_layers=0` 用均分布局跑通，再加 MTP 并按显存实测调整。

### 2.6 备选配置（V1 跑不通时的回退）

| 场景 | 调整 | 说明 |
|---|---|---|
| OOM（capture 阶段） | PP 2 → **4**（EP 相应降到 16） | 每 stage 20 层，显存压力减半；代价是 bubble 变大 |
| OOM（仍不够） | 关 MTP + MBS=1 + `--recompute-granularity selective` | 见 07e 坑 D |
| hybridep 挂死 | 检查 `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` **是否 == EP** | 07e 重要坑 1，改 EP 必须两处一起改 |
| full graph capture 崩 | 确认 Part 3 的两个 full-graph env 都设了 | 07e 坑 C |

---

## 三、精度选择：官方推荐 BF16，FP8 对 MoE 未必有收益

### 3.1 Hy3 官方口径 — 全线 BF16，无任何 FP8 训练路径

逐个文件扒过官方仓库，结论一致：

| 来源 | 精度设置 |
|---|---|
| 模型卡「支持精度」 | **BF16** |
| `llama_factory_support/hy_v3_full_sft.yaml` | `bf16: true` |
| `ms_swift_support/hy_v3_full_sft.yaml` | `torch_dtype: bfloat16` / `bf16: true` / `# fp16: false` |
| `deepspeed_support/train.sh` | `--bf16` |
| `ds_zero3_*.json` | `bf16.enabled: auto`（由 `--bf16` 激活） |
| **Hy3-FP8** | **推理量化版**，AngelSlim PTQ 工具包产出 — **不是训练精度** |
| 预训练精度 | **官方未披露**（README 只提「预训练框架重建」，无精度说明） |

> **关键区分**：Hy3-FP8 是 post-training quantization 的推理产物，跟「用 FP8 训练」是两回事。DeepSeek V3 那种 FP8 预训练，腾讯**没有公开说自己做了**。

→ 我们在 Megatron 侧跑 FP8_MX 属于**自选路线，未经官方验证**。

### 3.2 FP8 对 MoE 到底有没有用 — 本仓 GB200 实测

直觉上 GB300 FP8 峰值 5400 vs BF16 2700 = 2×，但 **MoE 模型兑现不了**。本仓自己的实测：

| 来源 | 配置 | BF16 | FP8 / MXFP8 | 差异 |
|---|---|---|---|---|
| [a4x README A2 vs A3](../../a4x/07-megatron-training/README.md) | DSV3 12L, 8 GPU, seq 16384, EP8 MBS2 | **527** | 503 (FP8) | **FP8 慢 5%** |
| [a4x 07c §2.2 步骤 7→8](../../a4x/07-megatron-training/07c-deepseekv3-671b-recipe/README.md) | DSV3 32L, PP2 EP64, 已开 CUDA graph | 928 | 970（+mxfp8 +fp32 optim +fp8-param-gather） | **+4.5%** |

**原因**（a4x README 原文结论）：*"grouped GEMM 的 FP8 路径 overhead 抵消了 Tensor Core 加速，BF16 反而更快"*。MoE 的瓶颈在 all-to-all 通信和访存，不是纯 GEMM 算力。

### 3.3 ⚠️ 澄清：那些 1100–1658 的数字**全是 MXFP8，不是 BF16**

容易记混，列清楚：

| 平台 | 配置 | 精度 | TFLOP/s/GPU |
|---|---|---|---|
| GB200 | DSV3 16L 64GPU Bridge V2 (MNNVL=2) | MXFP8 | 1176 |
| GB200 | DSV3 16L 64GPU Bridge V2 | MXFP8 | 1124 |
| GB200 | DSV3 16L 64GPU (MNNVL=0) | MXFP8 | 1100 |
| GB200 | DSV3 32L 128GPU raw Megatron | MXFP8 | 992 |
| GB200 | DSV3 61L 256GPU（NVIDIA 参考） | MXFP8 | 1292 |
| GB300 | DSV3 61L 256GPU | MXFP8 | **1658** |
| GB200 | DSV3 61L PP4 EP32 baseline | **BF16** | **300**（alltoall 未优化） |

> **本仓没有「充分优化后的 BF16」DSV3 数据**。300 那条是 alltoall 未优化 baseline（后续靠 HybridEP/graph 优化到 900+，那些步骤与精度无关），**不能当 BF16 代表值**。
>
> 真正干净的 BF16↔FP8 同配置对照只有 §3.2 那两组：**-5% 和 +4.5%**。

### 3.4 修正后的 Hy3 预期区间

§七 的 1200–1500 是 **FP8_MX 口径**（锚点 GB300 DSV3 1658）。按 §3.2 的 FP8 实际收益推算：

| 精度 | Hy3 预期区间 | 依据 |
|---|---|---|
| FP8_MX | 1200 – 1500 | 从 DSV3 GB300 1658 按结构差异下调 |
| **BF16** | **1150 – 1450** | FP8 相对 BF16 仅 −5%~+5%，**不是减半** |

> **不要**按「峰值算力减半 → 吞吐减半」推 BF16。那个逻辑只对 dense 模型的纯 GEMM 负载成立。

**Hy3 特有的额外担忧**：`moe_ffn_hidden_size` 1536 比 DSV3 的 2048 更小，`hidden_size` 4096 比 7168 更窄 → **grouped GEMM 形状更小 → FP8 量化/反量化 overhead 占比更高**，收益可能比 DSV3 还差，甚至为负（对齐 a4x A3 那个 −5%）。

### 3.5 结论：首跑用 BF16

1. **对齐官方口径** — 腾讯自己就是 BF16 训的，少一个"和官方不一致"的变量。
2. **少一层风险** — FP8_MX 在 Hy3 上完全未验证（Bridge 无 hy3 recipe，TE 的 hy_v3 FP8 路径没人跑过）。
3. **收益本来就小** — MoE 上 FP8 只有 ±5%，不值得在 bring-up 阶段引入。
4. 跑通 BF16 拿到基线后，**再单独做一组 FP8_MX 同配置对照**（见 §七 待验证 #9），用实测决定要不要切。

---

## 四、三条落地路径

### 路径 A：Bridge AutoBridge + 自研 recipe（**推荐，benchmark 走这条**）

`HYV3Bridge` 已注册，能从 HF config 自动生成正确的 `GPTModelProvider`：

```python
from megatron.bridge import AutoBridge
bridge   = AutoBridge.from_hf_pretrained("tencent/Hy3")
provider = bridge.to_megatron_provider()     # MoE/router/MTP 参数自动填对

# 训练侧覆盖（见 §1.3）
provider.moe_router_bias_update_rate   = 1e-3
provider.moe_token_dispatcher_type     = "flex"
provider.moe_flex_dispatcher_backend   = "hybridep"
provider.moe_a2a_overlap               = True
# 并行度（见 §2.3）
provider.tensor_model_parallel_size    = 1
provider.pipeline_model_parallel_size  = 2
provider.virtual_pipeline_model_parallel_size = 8
provider.expert_model_parallel_size    = 32
```

完整脚本见同目录 [`hy3_provider.py`](hy3_provider.py)。

> **注意**：`run_script.py -m hy3 -mr ...` **不可用**（Bridge 无 hy3 perf config）。要么用自研 pretrain 脚本，要么往容器里补一个 `hy3_workload_base_configs.py`（结构照抄 `deepseek_workload_base_configs.py`）。

### 路径 B：mock 数据纯 benchmark（不下权重）

跑吞吐 benchmark 不需要真权重。用 §1.1/§1.2 的参数表手工构造 provider（`hy3_provider.py` 里的 `build_hy3_provider_from_scratch()`），`--data mock`，跟 07d/07e 同口径。**首跑建议走这条**，省掉 590GB 权重下载。

### 路径 C：官方 SFT 栈（精度对齐参照，不是性能路线）

腾讯官方三套：`finetune/deepspeed_support/`（HF Trainer + DeepSpeed）、`finetune/llama_factory_support/`、`finetune/ms_swift_support/`。

官方硬件要求（`max_seq_length=4096`，关 `make_moe_param_leaf_module` 和 zero3+offload）：
- LoRA 微调：≥ 1 机 8 卡（80GB+）
- 全量微调：≥ 4 机 32 卡（80GB+）

> **用途**：作为**精度基线**（跑几百步对比 loss 曲线，验证我们的 Megatron 配置没配错），**不要**用它跑 MFU——DeepSpeed ZeRO 在 GB300 上打不过 Megatron 的 PP+EP+CUDA graph。

### 权重准备（路径 A/C 需要）

官方 checkpoint **每个专家权重独立存储**，加载极慢。官方提供转换脚本融合成 3D tensor：

```sh
# 转换（在 finetune/tools/ 下）
python convert_ckpt_to_outer.py --input_dir <原始ckpt> --output_dir <输出> --workers 8
# 校验
python check_converted.py <输出目录> --spot-check 3
```

> 官方 README 里写的是 `train/tools` 目录，实际 HF 仓库路径是 **`finetune/tools/`**（文档笔误）。

### 训练数据格式（路径 A SFT / 路径 C）

Hy3 快慢思考双模式，数据里用 `reasoning_effort` 标注：

```json
{"reasoning_effort": "no_think", "messages": [{"role":"system","content":"..."},{"role":"user","content":"1+1=?"},{"role":"assistant","content":"1+1=2"}]}
{"reasoning_effort": "high",     "messages": [..., {"role":"assistant","content":"1+1=2","reasoning_content":"用户在问 1+1 ..."}]}
```

```python
tokenizer.apply_chat_template(messages, is_training=True)   # use_fast=False, trust_remote_code=True
```

> 慢思考样本的思维链放在 `reasoning_content` 字段，不是 `content`。混两种模式训练才能保住 `reasoning_effort` 开关的行为。

---

## 五、集群准备（16 节点单域）

### 可用 pool（bunny 2026-07-25 质检，5 个全 PASS，共 88 节点 / 352 GPU）

| Pool | 节点 | GPU | NCCL all_reduce | 说明 |
|---|---|---|---|---|
| **gb300-pool-0015** | 18 | 72 | 689.3 GB/s | ✅ **本文首选**（用 16 台，留 2 台热备） |
| gb300-pool-0016 | 18 | 72 | 688.3 GB/s | 备选 |
| gb300-pool-0017 | 18 | 72 | 689.6 GB/s | 备选 |
| gb300-pool-0014 | 16 | 64 | 688.9 GB/s | 刚好 16 台**零余量**；GKE STATUS 显示 ERROR 是抱怨没凑够 18 台，节点健康 |
| gb300-pool-0013 | 18 | 72 | 688.3 GB/s | 1 WARN：pqcm 节点 DRAM correctable ECC >1000（软错误，不影响正确性） |

**不要用**：`pool-0002`（yangwhale 负载在跑 + 坏 COS）、`pool-0006`（infer 团队 17 pod + 坏 COS）、`pool-0009`（空闲且 COS 好，但不在交付范围，要用先问 chris）。

多节点已验：MNNVL=ON **933 GB/s**，MNNVL=OFF 走 RDMA 379 GB/s，NVLink fabric 正常。

> **为什么选 18 节点池而不是刚好 16 台的 0014**：集群 **auto-repair 关闭**（刻意防误换机），节点坏了不自愈。18 节点池用 16 台可留 2 台同 subblock 热备，坏一台换标签即可，不用整池搬迁（07e 那种"整池征用"很痛）。

### ⚠️ 两个集群级硬约束

| # | 约束 | 后果 / 应对 |
|---|---|---|
| 1 | **节点绝不能滚到 node image `1.36.0-gke.4681000`（COS 224.80）** | 该版本 `nvidia.ko` 有 regression，`cuDevicePrimaryCtxRetain` 返回 `INVALID_VALUE` — `nvidia-smi` 看着正常但**所有 CUDA 负载全挂**。集群已设维护例外冻结升级，**2026-10-23 到期**，到期前需续期。可用 5 池 kubelet 均为 `1.36.0-gke.4447000`（好 image）。 |
| 2 | **auto-repair 全部关闭** | 节点坏了不会自愈，**必须自己监控**。训练中途掉节点 → 手动换标签到热备节点 → 删 pod 重调度。 |

### Step 0 — 给目标池打标签

```bash
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
POOL=gb300-pool-0015

# 挑 16 台 Ready 节点打标签（pool = subblock，天然同域，不用再挑 subblock）
N=$(kubectl --context $CTX get nodes -l cloud.google.com/gke-nodepool=$POOL --no-headers \
    | grep -w Ready | awk '{print $1}' | head -16)
kubectl --context $CTX label node $N team=yangwhale --overwrite

# 校验：应为 16，且 subblock 只有一个
kubectl --context $CTX get nodes -L cloud.google.com/reservation-subblocks -l team=yangwhale \
  --no-headers | awk '{print $NF}' | sort | uniq -c

# 顺带确认 node image 不是坏的 224.80
kubectl --context $CTX get nodes -l team=yangwhale \
  -o custom-columns=NODE:.metadata.name,KUBELET:.status.nodeInfo.kubeletVersion --no-headers \
  | awk '{print $2}' | sort | uniq -c    # 期望全是 v1.36.0-gke.4447000
```

> **必须同一 subblock，不能跨池拼节点**（07e 教训）。用整池内的节点天然满足。

### Step 1 — 拉起 pod 池

```bash
kubectl --context $CTX apply -f yw-pool-64.yaml
kubectl --context $CTX get pods -l job=yw --no-headers | grep -c Running   # 应为 16
kubectl --context $CTX get computedomains | grep yw                        # 1 个 Ready
```

### Step 2 — 分发启动脚本 + 起训练

```bash
B64=$(base64 -w0 run-hy3-yw.sh)
seq 0 15 | sed 's/^/yw-a-/' | xargs -P 16 -I {} kubectl --context $CTX exec {} -- bash -c \
  "echo $B64 | base64 -d > /tmp/run-hy3-yw.sh && chmod +x /tmp/run-hy3-yw.sh"

seq 0 15 | sed 's/^/yw-a-/' | xargs -P 16 -I {} kubectl --context $CTX exec {} -- bash -c \
  "nohup /tmp/run-hy3-yw.sh > /tmp/hy3-run.log 2>&1 &"
```

> 16 pod 规模下并行 `kubectl exec` 没问题（07e 的 konnectivity 限流是 64 pod 才触发）。

### Step 3 — 监控

```bash
kubectl --context $CTX exec yw-a-0 -- bash -c 'grep -E "Step Time|MODEL_TFLOP" /tmp/hy3-run.log | tail -6'
```

---

## 六、关键环境变量

与 07e 完全一致，**唯一差异是 `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` 要等于本 recipe 的 EP**。完整列表见 `run-hy3-yw.sh`，这里只列最容易漏的：

| env | 值 | 漏了会怎样 |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,graph_capture_record_stream_reuse:True` | full graph capture 崩 `StreamCaptureUnjoined` |
| `TORCH_NCCL_AVOID_RECORD_STREAMS` | `0` | 同上（base 默认是 1，必须显式改 0） |
| `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` | **`32`**（== EP） | hybridep all-to-all 挂死，collective timeout |
| `NVLINK_DOMAIN_SIZE` / `USE_MNNVL` | `72` / `1` | hybridep 不生效 |
| `NCCL_GRAPH_REGISTER` | `0` | GB300 GIB 下 =1 会 rendezvous 挂死 25min |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `32` | hybridep + sm100 性能掉 |
| `NVTE_FWD/BWD_LAYERNORM_SM_MARGIN` | `20` | hybridep 下 SM 争抢 |

---

## 七、待验证清单（开跑后逐项填）

| # | 验证项 | 判定标准 | 结果 |
|---|---|---|---|
| 1 | Provider 参数量核对 | log 打印总参 ≈ 295B（±1%） | ⬜ |
| 2 | 单节点 4 GPU smoke（缩层到 8 层） | 10 步不崩，loss 下降 | ⬜ |
| 3 | 64 GPU 起 + full graph capture | 越过 capture 进稳态 | ⬜ |
| 4 | 稳态吞吐 | 记录 TFLOP/s/GPU + step time | ⬜ |
| 5 | 专家负载均衡 | 开 `bias_update_rate=1e-3` 后 expert load CV 收敛 | ⬜ |
| 6 | MTP 开关对比 | mtp=0 vs mtp=1 的吞吐/显存差 | ⬜ |
| 7 | EP 扫点 | EP 16 / 32 对比（DSV3 结论能否复现） | ⬜ |
| 8 | GBS 扫点 | 2048 → 4096 → 8192，找 MFU 拐点 | ⬜ |
| 9 | **BF16 vs FP8_MX 同配置对照** | 各 30 步，验证 §3.2 的「MoE 上 FP8 仅 ±5%」在 Hy3 上是否成立 | ⬜ |
| 10 | **显存实测 vs 测算** | 记 `max_memory_reserved()`，对照 §2.4 预测 195.5 GB，重新标定 1.34× 系数 | ⬜ |

### 无对标基线，怎么判断"跑得好不好"

Hy3 **没有官方 Megatron benchmark**，NVIDIA perf summary 里也没有。判断标准只能横向类比：

| 模型 | 规模 | 精度 | GB300 实测 TFLOP/s/GPU | 备注 |
|---|---|---|---|---|
| DeepSeek V3 | 671B / 61L | MXFP8 | **~1658**（256 GPU） | 官方 1670 的 99.3% |
| DeepSeek V3 scale-in | 671B / 31L | MXFP8 | ~1550（128 GPU） | 层数减半 → 固定开销占比升 |
| **Hy3** | **295B / 80L** | **BF16 首跑** | **待测** | 见下方预期 |

**预期区间推理**（非实测，供判断用）：
- **利好**：层数多（80 vs 61）→ 每 stage 层数足，固定开销摊薄；专家更细（moe_ffn 1536 vs 2048）；GQA 比 MLA 计算简单。
- **不利**：hidden 4096 比 DSV3 的 7168 窄 → **GEMM 形状更小，Tensor Core 效率低**；总参小但层多 → 每层 activation 通信次数多。
- **粗判（BF16 首跑口径）**：稳态落在 **1150–1450 TFLOP/s/GPU** 属正常；低于 1000 说明配置有问题（优先查 EP / dispatcher / graph 是否真生效）。
- **FP8_MX 口径**：1200–1500。两者差距小，原因见 §三 —— MoE 上 FP8 相对 BF16 只有 −5%~+5%，**不是按峰值算力翻倍**。

---

## 八、预判踩坑（从 07e 迁移）

| # | 坑 | 预防 |
|---|---|---|
| 1 | 手动覆盖 `cuda_graph_impl` | **不要覆盖**，会绕过 `moe_paged_stash` 等 full graph 必需机制 |
| 2 | 改 EP 忘改 `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` | 两处必须一起改，否则 all-to-all 挂死 |
| 3 | 跨机型借参数（GB200 的 VPP） | 只用 GB300 原生数值 |
| 4 | torchrun 直跑不自动设 perf env | 全部手动 export（§6） |
| 5 | SSH 启动丢容器 ENV | `run-hy3-yw.sh` 已内置 `/proc/1/environ` 加载 |
| 6 | NCCL 崩溃刷爆磁盘 → DiskPressure 驱逐 | 崩后先清 Evicted pod 再重拉，别连环硬跑 |
| 7 | 重度 churn ComputeDomain 后立刻硬跑 | 静置让 DRA/IMEX/RDMA 收敛再跑（07e 最值钱教训） |
| 8 | **Hy3 专属**：bias_update_rate=0 导致专家失衡 | from-scratch 必须设 1e-3（§1.3） |
| 9 | **Hy3 专属**：末 stage 扛 5 层 + MTP + loss | 首跑关 MTP；OOM 就换前重后轻布局（§2.5） |
| 10 | **集群**：节点滚到 COS 224.80 → CUDA 全挂 | 维护例外冻结中（2026-10-23 到期）；启动前查 kubelet 版本 |
| 11 | **集群**：auto-repair 关闭，坏节点不自愈 | 自己监控；18 节点池留 2 台热备，坏了换标签 |

---

## 附：文件清单

| 文件 | 说明 |
|---|---|
| `yw-pool-64.yaml` | 单 ComputeDomain 16 节点 64 GPU sleep-infinity pod 池 |
| `run-hy3-yw.sh` | 单 pod 启动脚本（完整 env + rank 计算 + torchrun） |
| `hy3_provider.py` | Hy3 `GPTModelProvider` 构造（AutoBridge 路径 + from-scratch 路径 + 训练覆盖） |
| `mem_calc.py` | BF16 显存测算器（切分公式 + 1.34× 运行时开销标定 + GBS/最少卡数建议） |

---

*2026-07-25 · Recipe 设计完成，待 16 节点 64 GPU 实跑 · MoE 血统 = DeepSeek V3，Attention = GQA*

---

## 九、实战日志（2026-07-25 起）

> 边跑边记。每个已验证的步骤和每个踩的坑都在这里，避免信息丢失。

### 里程碑 M1 — 集群就位（22:26 HKT ✅）

| 步骤 | 结果 |
|---|---|
| 本机 kubectl 直连 gb300-gke-test | ✅ 可用，**不需要 ssh gLinux**（authorized-networks 已放行） |
| pool-0015 体检 | 18 台全 Ready，kubelet 全 `v1.36.0-gke.4447000`（好 COS），**同一 subblock** `10270c36...0632` |
| 打标签 16 台 | ✅ `hy3=true`，剩 2 台热备 |
| 拉起 pod 池 | ✅ 16/16 Running（首次拉镜像约 2 分钟），ComputeDomain `yw-cd-a` 创建成功 |

**坑 A：不能复用 `team=yangwhale` 标签。**
`pool-0002` 上已有 **17 台 `team=yangwhale`** 跑着别的负载。若沿用该标签，两边 nodeSelector 会互相串台。
→ 改用专属标签 **`hy3=true`**，YAML 已同步。**教训：进集群前先查标签占用，别假设标签是自己独占的。**

### 里程碑 M2 — 发现容器不支持 hy_v3，改走自研 recipe（22:35 HKT ✅）

**坑 B（阻断级）：容器内 Bridge r0.5.0 没有 `hy_v3` 模型桥。**

```
容器: Bridge 0.5.0+fcbb6031
models/ 有: bailing deepseek ernie gemma glm gpt_oss kimi llama mamba minimax
            ministral3 mistral nemotron olmoe qwen stepfun ...
models/ 无: hy_v3   ← §四 路径 A (AutoBridge) 直接作废
recipes/ 无: hunyuan / hy3
scripts/performance/configs/ 有: deepseek gpt_oss kimi llama nemotronh qwen qwen_vl wan
```

`HYV3Bridge` 只在 Megatron-Bridge **main 分支**，r0.5.0 容器里没有。

**坑 C：`run_script.py` 的 CLI 覆盖不够用。**
它只有 `--hidden_size` / `--num_layers` / `--first_k_dense_replace` / `--vocab_size` / `--pipeline_model_parallel_layout`，
**没有** `--num_moe_experts` / `--num_query_groups` / `--moe_ffn_hidden_size` —— 无法把 deepseek recipe 改造成 Hy3。

**解法：以 Qwen3-235B-A22B recipe 为骨架自建配置。**

选它的原因（实查 `qwen3_235b_a22b_pretrain_config()` 返回值）：
- 返回的就是**裸 `GPTModelProvider`**（不是 MLA 专用 provider），字段可以随便改
- 本身是 GQA MoE，且 **hidden_size 4096 / kv_channels 128 / qk_layernorm True / moe_ffn 1536 已经跟 Hy3 完全一致**
- optimizer / ddp / dataset / scheduler 都是调好的

只需改：`num_layers 94→80`、`ffn_hidden_size 12288→13312`、`num_moe_experts 128→192`、
`num_query_groups 4→8`、`vocab 151936→120832`，再叠加 DSV3 血统的 MoE 旋钮。

`GPTModelProvider` 有 **309 个字段**，实测我需要的 34 个字段只有 `untie_embeddings_and_output_weights` 不存在
（实际叫 **`share_embeddings_and_output_weights`**，取反）。

产物：**[`hy3_pretrain.py`](hy3_pretrain.py)**（取代 §四 路径 A 的 `hy3_provider.py`）。

### 里程碑 M3 — 高性能配方从 deepseek 源码原样移植（✅）

实读容器内 `scripts/performance/configs/deepseek/deepseek_llm_pretrain.py`，把两个函数原样搬过来：

`set_deepseek_v3_common_configs`：
```python
moe_router_fusion = True
recompute_granularity = "selective"
dist.enable_megatron_core_experimental = True
mixed_precision.grad_reduce_in_fp32 = False   # ← 梯度是 BF16
ddp.grad_reduce_in_fp32 = False
moe_router_force_load_balancing = True        # benchmark 专用
```

`set_full_iter_cg_configs`（**用户点名要的 full graph + paged stash**）：
```python
moe_pad_experts_for_cuda_graph_inference = True
moe_paged_stash = True                        # MCore PR #4247
moe_expert_rank_capacity_factor = 1.5
moe_paged_stash_buffer_size_factor_cuda = 1.2
moe_paged_stash_buffer_size_factor_cpu = 1.0
```
> 源码注释解释了机制：**dropless MoE 产生变长 per-expert tensor，CUDA graph 抓不住**；
> 先 pad 到固定容量（`pad_experts` + capacity factor），再用 paged stashing 把显存收回来。

**⭐ 修正 §2.4 显存测算**：`grad_reduce_in_fp32 = False` 意味着梯度是 **BF16 (2B) 不是 FP32 (4B)**。
64 GPU / PP2 / EP32 / MBS1 下梯度从 32.8 GB → **16.4 GB**，朴素合计 145.9 → **129.5 GB**，
×1.34 后 **173.6 GB**（原估 195.5 GB）。**余量比预想更足。**

### 里程碑 M4 — 配置构建通过（22:40 HKT ✅）

```
Hy3 295B | 64 GPU | TP1 PP2 VPP8 EP32 | MBS1 GBS2048 | bf16
  层数 80 (dense 1 + MoE 79)  hidden 4096  GQA 64Q/8KV x 128
  MoE 192 experts top-8 ffn 1536 shared 1536
  路由 sigmoid + expert_bias=True (rate 0.001) lb=none scale 2.826
  dispatcher flex/hybridep  graph full_iteration  paged_stash True
  MTP None  pp_layout Et*5|(t*5|)*14t*5L
  DP=32  microbatch/rank=64  专家参数 286.3B (每 rank 4.47B)
```

途中修掉 3 个 API 不匹配（都是 qwen3 recipe 与 deepseek recipe 的结构差异）：

| # | 报错 | 根因 | 修法 |
|---|---|---|---|
| 1 | `'str' object has no attribute 'grad_reduce_in_fp32'` | qwen3 recipe 的 `cfg.mixed_precision` 是**字符串** `'bf16_mixed'`，不是配置对象 | `if isinstance(cfg.mixed_precision, str): cfg.mixed_precision = bf16_mixed()` |
| 2 | `'NoneType' object has no attribute 'overlap_grad_reduce'` | qwen3 recipe 的 `cfg.comm_overlap` 默认 **None**（deepseek 配置里才构造） | 显式 `CommOverlapConfig(...)` |
| 3 | `CommOverlapConfig.__init__() missing 'tp_comm_overlap'` | 该字段是 **keyword-only 必填** | `CommOverlapConfig(tp_comm_overlap=False)`（TP=1 无 TP 重叠） |

> **通用教训**：跨 recipe 家族借骨架时，`ConfigContainer` 的**同名字段类型可能不同**（str vs 对象、None vs 实例）。
> 照抄 A 家族的赋值语句到 B 家族骨架上，必然踩这类空指针/类型错，逐个 dryrun 打掉即可。

### 里程碑 M5 — 4 GPU 缩层冒烟通过（15:02 HKT ✅）

配置：4 GPU / 8 层 / TP1 PP2 VPP2 EP2 / MBS1 GBS8 / seq4096 / BF16 / TE graph。

```
iteration 1/5 | elapsed 77498.7 ms | lm loss 1.251689E+01 | grad norm 6.013   <- 含 graph capture
iteration 2/5 | elapsed   572.3 ms | lm loss 1.251494E+01 | TFLOP/s/GPU 233.6
iteration 3/5 | elapsed   532.1 ms | lm loss 1.224570E+01 | TFLOP/s/GPU 251.3
iteration 4/5 | elapsed  4155.2 ms | lm loss 1.139484E+01 | <- 周期性 GC
iteration 5/5 | elapsed   455.9 ms | lm loss 1.086530E+01 | TFLOP/s/GPU 293.3
Rank 0: 16 graphs deleted with explicit reset
```

✅ **loss 12.52 → 10.87 单调下降，0 NaN、0 skipped，CUDA graph 正常创建/销毁。模型结构和前反向链路验证通过。**
每 rank 参数 8.12B（PP rank 1），总参 27.26B（8 层缩减版），dense+embedding 1.89B。

### 这一段踩的 4 个坑（都是"借骨架"的代价）

| # | 报错 | 根因 | 修法 |
|---|---|---|---|
| D | `moe_expert_rank_capacity_factor requires use_transformer_engine_op_fuser to be enabled` | paged stash 有**隐式依赖链**：`cutedsl_fused_grouped_mlp=True` → `use_transformer_engine_op_fuser=True` → 才允许 `moe_expert_rank_capacity_factor`（见 `scripts/performance/utils/overrides.py:238-239`）。我手写高性能配置时只搬了终端字段，漏了链条中间环节 | **改为直接复用官方 `WorkloadBaseConfig` + `set_workload_base_configs()`**，不再手工重实现映射 |
| E | `pipeline_model_parallel_layout cannot be set with other pipeline layout arguments` | qwen3 骨架默认 `account_for_embedding_in_pipeline_split=True` / `account_for_loss_in_pipeline_split=True`；而 pp_layout 字符串里已显式写了 `E` 和 `L`，两者互斥（deepseek 骨架默认 False 所以没这问题） | 设 pp_layout 时同步把两个 `account_for_*` 置 False |
| F | `Model vocab_size (120832) cannot be smaller than tokenizer's vocab_size (151669)` | 骨架带的是 **Qwen tokenizer**（词表 151669），比 Hy3 的 120832 大 | mock benchmark 改用 `NullTokenizer` + `vocab_size=120832`，并清空 `tokenizer_model` |
| G | 后台任务被静默杀掉，日志停在旧内容 | `kubectl exec ... "cmd &"` 的后台进程会随 exec 会话结束被 SIGTERM | 用 **`setsid nohup ... < /dev/null &`** 彻底脱离会话 |

> **贯穿性教训**：跨 recipe 家族借骨架，坑不在"字段值填错"，而在 **A 家族的隐式前提在 B 家族不成立**
> （默认 True/False 相反、类型是 str 而非对象、依赖链中间环节缺失）。
> **对策：能调官方 setter 就别手写赋值**——`set_workload_base_configs()` 一次性解决了 D 类问题。

### ⭐ 重大发现：官方 BF16 recipe **不用** full_iteration graph

实读 `deepseek_workload_base_configs.py`，NVIDIA 对 GB300 的两套配置差异很大：

| | `..._GB300_BF16_V1` | `..._GB300_FP8_MX_V1` |
|---|---|---|
| cuda_graph_impl | **`transformer_engine`** | **`full_iteration`** |
| cuda_graph_scope | `[attn, moe_router, moe_preprocess]` | （全迭代） |
| moe_a2a_overlap | **False** | True |
| cutedsl_fused_grouped_mlp | **未开** | True |
| fp8_dot_product_attention | — | True |
| recompute_modules | `["moe_act"]` | `[]` |
| PP / VPP / EP | 4 / 4 / 64 | 2 / 8 / 32 |

**即：`full_iteration` + `paged stash` 在官方 recipe 里是 FP8_MX 专属，BF16 走 TE graph。**
原因链：paged stash 需要 TE op fuser，op fuser 由 `cutedsl_fused_grouped_mlp` 打开，而官方只在 FP8 档开 cutedsl。

→ 策略调整：**先用官方 BF16 配置（TE graph）拿到稳定基线**，再单独试 `BF16 + cutedsl + full_iteration`
是否可行（`hy3_pretrain.py` 已留 `--cuda-graph full_iteration --cutedsl --a2a-overlap` 开关）。

### 里程碑 M6 — 64 GPU 全量跑通，BF16 基线 707 TFLOP/s/GPU（23:13 HKT ✅）

**V1 配置**：64 GPU / 80 层全量 / TP1 PP2 VPP8 EP32 / MBS1 GBS2048 / seq4096 / BF16 /
TE graph / hybridep / `recompute_modules=[moe_act]` + selective —— 即 NVIDIA 官方 GB300 BF16 recipe。

| 校验项 | 预期 | 实测 | 结论 |
|---|---|---|---|
| 总参数量 | 295 B（官方） | **294.97 B** | ✅ 完全吻合 |
| 显存峰值 | §2.4 预测 173.6 GB | **184 GB** | ✅ 回标后精确命中 |
| 稳态吞吐 | — | **707.2**（中位 n=20，706–710，**±0.3%**） | 基线 |
| Step Time | — | 25.3 s | |
| GPU 利用率 | — | 99–100%（全 16 pod） | ✅ |

**⭐ 显存模型回标**：`grad_reduce_in_fp32=False` → 梯度是 **BF16(2B)** 不是 FP32(4B)。
朴素四项 = 权重 16.4 + 梯度 16.4 + 优化器 51.5 + 激活 45.2 = **129.5 GB**，实测 184 GB → 系数 **1.42×**。
`mem_calc.py` 已更新，现在预测 183.9 GB vs 实测 184 GB。

> 首步 121.3 TFLOP/s 是 graph capture 开销。启动到首个稳态数约 **10 分钟**。

### 里程碑 M7 — 性能扫点：707 → 854 TFLOP/s（+20.8%）

| 版本 | 变更 | 稳态 TFLOP/s/GPU | Step | 显存 | vs V1 |
|---|---|---|---|---|---|
| **V1** | 官方 BF16 recipe | **707.2**（706–710） | 25.3s | 184 GB | 基线 |
| **V2** | 关全部 recompute | **744**（743.6–744.5） | 24.0s | 217 GB | **+5.2%** |
| **V3** | V2 + `cutedsl` + `a2a_overlap` | **827**（825.3–828.5） | 21.6s | **197 GB** | **+16.9%** |
| **V4** 🏆 | V3 + `full_iteration` + **paged stash** | **854.4**（854.2–854.8） | **20.97s** | 230 GB | **+20.8%** |
| V5 | V4 + MBS 2 | ❌ **OOM** | — | 撞 277 GiB 顶 | — |
| V6 | V4 + GBS 4096 | ❌ **卡死** | — | 283 GB（99.96%） | — |

**V2**：关 recompute +5.2%，代价 +33 GB。比 a4x 上 DSV3 的 selective recompute 代价（~9%）小 ——
Hy3 专家中间层 1536 比 DSV3 的 2048 窄，`moe_act` 要重算的 activation 更少。

**V3（收益最大）**：`cutedsl_fused_grouped_mlp` + `moe_a2a_overlap` 一次拿 **+11.2%**（744→827），
**且显存反从 217 降到 197 GB** —— 融合 grouped MLP 消掉了中间 tensor。

**V4**：cutedsl 打开 TE op fuser 后，`full_iteration` + paged stash 在 **BF16 上跑通**（官方 recipe 未覆盖此组合），
再拿 +3.3%。波动仅 **±0.04%**，四个版本里最稳。

#### ⭐⭐ 核心发现：BF16 能吃到 FP8 recipe 的全部高性能特性

官方把 `cutedsl` / `a2a_overlap` / `full_iteration` / paged stash **只放在 FP8_MX 档**，
实测这套组合在 **BF16 上全部可用，累计 +20.8%**。
上一节那句「官方 BF16 recipe 不用 full_iteration」描述的是官方**配置选择**，**不是技术限制**。

依赖链（缺一不可，手写配置必漏）：
```
cutedsl_fused_grouped_mlp=True
  └→ use_transformer_engine_op_fuser=True          (overrides.py:238)
       └→ 允许 moe_expert_rank_capacity_factor
            └→ 允许 moe_paged_stash
                 └→ full_iteration graph 才抓得住 dropless MoE 的变长 tensor
```

#### V5 / V6 负结果：full graph 之后显存成为硬约束

| 版本 | 现象 | 根因 |
|---|---|---|
| V5（MBS 2） | `OutOfMemoryError: Tried to allocate 48.00 MiB, GPU has 276.62 GiB total, 21.31 MiB free` | 激活 45→90 GB + full graph buffer 撞顶 |
| V6（GBS 4096） | 显存 283 GB / 276.5 GiB 可用，capture 阶段**日志冻结 10+ 分钟、GPU 100% 空转**，无 OOM 报错但永不出步 | **`full_iteration` 把整个 iteration 的全部 microbatch 抓进一张 graph**，GBS 2048→4096 让 microbatch 64→128，**graph 本身翻倍** |

> ⭐ **推翻 07e 的一条经验**：07e 在 DSV3 256 卡上结论是「GBS 是收益最大的旋钮」。
> **在 Hy3 64 卡 + full graph 下完全不成立**：(1) PP2×VPP8×DP32 时 GBS 2048 的 bubble 已仅 **0.2%**，无 bubble 可消；
> (2) full graph 让 GBS 直接换算成显存，加 GBS = 加 graph，**先撞显存墙**。
> **教训：旋钮收益取决于当前瓶颈在哪，照搬别的规模的调优排序会浪费机时。**

> ⭐ **标定系数与 graph 类型强相关**：TE graph 回标 1.42×（预测 183.9 / 实测 184，命中）；
> 同配置换 full_iteration 实测 230 GB，等效 **1.78×**。`mem_calc.py` 的 1.42 只适用 TE graph。

### 这一段踩的 3 个坑

#### 坑 H：`moe_paged_stash` 校验分支空指针（Megatron 侧 bug）
```
TypeError: 'NoneType' object is not iterable
  transformer_config.py:1691  {"expert_fc1","moe_act"} & set(self.offload_modules)
```
`offload_modules` 默认 `None`，而 paged stash 校验无条件对它 `set()`。**只在 full_iteration 路径触发**。
修：`m.offload_modules = []`。

#### 坑 I：`pkill -f <pattern>` 会杀掉自己所在的 `bash -c`
```bash
# ❌ pattern "hy3_pretrain" 匹配到本条 bash -c 命令行自身 -> 自杀，后面的写文件根本没跑
kubectl exec POD -- bash -c 'pkill -9 -f hy3_pretrain; echo "$B" | base64 -d > /tmp/hy3_pretrain.py'
```
症状极具迷惑性：**改了代码、重跑、报一模一样的错**，像是修复无效。
修：kill 与写文件拆成两次独立 exec，且 pattern 避开自身（改用 `torchrun`）。
**规矩：改完代码重跑前，先 `md5sum` 对比本地与远端。**

#### 坑 J：`pkill` 留下的僵尸 CUDA context 吃掉下一轮显存
V5 OOM 后用 `pkill -9 -f torchrun` 清场，V4 复跑**立刻 `CUDA error: out of memory`**：
```
39781, 230308 MiB   <- V5 的 worker，早已退出，CUDA context 未释放
49032,  52494 MiB   <- 新一轮只抢到剩下的
```
`pkill -f torchrun` 只杀 launcher，**multiproc worker 成孤儿，`ps` 看不见但驱动侧 context 不回收**。
清理法（不必重建 pod，且规避坑 I）：
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9
```
**每次扫点切换配置前必做**，否则会把残留误判成"新配置 OOM"。

### 🏆 当前最优配置（V4）

```bash
python hy3_pretrain.py \
  --num-gpus 64 --tp 1 --pp 2 --vpp 8 --ep 32 \
  --mbs 1 --gbs 2048 --seq-length 4096 \
  --cuda-graph full_iteration --cutedsl --a2a-overlap \
  --recompute-modules --recompute-granularity none \
  --mtp-layers 0 --max-steps 30
# env: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32（必须 == EP）
```

**854 TFLOP/s/GPU · 20.97 s/step · 230 GB/GPU · 16 节点单 NVLink 域**

**独立复现验证**（清空僵尸显存后重跑）：稳态中位 **854.5**，区间 854.2–856.6，**±0.14%**，
与首测 854.4 一致 —— 配置可复现，非偶然。

对标参照（均 MXFP8，非同精度，仅供量级）：DSV3 671B GB300 256 卡 1658 / GB200 256 卡 1292。
Hy3 hidden 4096 只有 DSV3 的 57%，GEMM 形状小、算力密度天然低，854 属合理区间。

### 待办
- [x] M6：64 GPU 跑通 → 707 基线
- [x] M7-1：关 recompute → 744（+5.2%）
- [x] M7-2：cutedsl + a2a_overlap → 827（+16.9%）
- [x] M7-3：full_iteration + paged stash → **854（+20.8%）**
- [x] M7-4：MBS 2 **OOM**；GBS 4096 **卡死**（full graph 下显存是硬约束）
- [ ] M8：开 MTP；EP 16 vs 32 扫点；BF16 vs FP8_MX 对照

## 十、消融实验超级大表格（2026-07-26）

**27 组配置全量扫点**，由 [`sweep.sh`](sweep.sh) 自动串行执行：
每组自动清僵尸 CUDA context（清不掉则重建 pod）→ 分发 → 16 pod 启动 → 等稳态 → 采集指标 → 写 [`results.csv`](results.csv)。

**采集口径**：TFLOP/s 取稳态末 5 步中位数（首步含 graph capture，已排除）；HBM 取全程 `nvidia-smi` 峰值；tok/s/GPU = `GBS × seq_len / step_time / 64`；
MFU = Model TFLOP/s ÷ 硬件峰值（**BF16 按 2,700，FP8 按 5,400**）。

> 基准 **A1 = 854.0 TFLOP/s**（BF16 冠军配置），`vs A1` 列为相对增减量。

### A 组 · 单开关消融

从 A1 冠军配置出发，每次只动一个开关，隔离单项贡献。

| 实验 | 状态 | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |
|---|---|---|---|---|---|---|---|
| **A1** V4_champion | ✅ | 854.0 | +0.0% | 31.6% | 225 GB | 6242 | 21.00s |
| **A2** no_cutedsl | ❌ CRASH | — | — | — | 212 GB | — | — |
| **A3** no_a2a_overlap | ✅ | 718.6 | **-15.9%** | 26.6% | 219 GB | 5253 | 24.95s |
| **A4** no_paged_stash | ❌ CRASH | — | — | — | 186 GB | — | — |
| **A5** no_router_fusion | ✅ | 843.1 | **-1.3%** | 31.2% | 226 GB | 6162 | 21.27s |
| **A6** no_permute_fusion | ✅ | 854.2 | +0.0% | 31.6% | 225 GB | 6244 | 20.99s |
| **A7** dispatcher_alltoall | ❌ CRASH | — | — | — | 0 GB | — | — |
| **A8** graph_none | ✅ | 572.5 | **-33.0%** | 21.2% | 187 GB | 4185 | 31.32s |
| **A9** graph_TE | ✅ | 827.7 | **-3.1%** | 30.7% | 193 GB | 6049 | 21.67s |
| **A10** no_force_lb | ⚠️ HANG | — | — | — | 219 GB | — | — |
| **A11** recompute_selective | ✅ | 853.5 | -0.1% | 31.6% | 225 GB | 6239 | 21.01s |

### B 组 · 并行度与批次

追查 854 与 Qwen3-235B 1360 的差距是否来自并行/批次配置。

| 实验 | 状态 | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |
|---|---|---|---|---|---|---|---|
| **B1** vpp2 | ✅ | 852.1 | -0.2% | 31.6% | 243 GB | 6230 | 21.04s |
| **B2** vpp4 | ✅ | 854.0 | +0.0% | 31.6% | 231 GB | 6242 | 21.00s |
| **B3** pp4_ep16 | ✅ | 855.7 | +0.2% | 31.7% | 230 GB | 6253 | 20.96s |
| **B4** ep16 | ✅ | 882.3 | **+3.3%** | 32.7% | 241 GB | 6450 | 20.32s |
| **B5** mbs2_TEgraph | ⚠️ HANG | — | — | — | 277 GB | — | — |
| **B6** mbs2_pp4 | ❌ OOM | — | — | — | 105 GB | — | — |
| **B7** gbs4096_TEgraph | ✅ | 834.2 | **-2.3%** | 30.9% | 194 GB | 6098 | 42.99s |
| **B8** gbs1024 | ✅ | 845.7 | -1.0% | 31.3% | 223 GB | 6183 | 10.60s |

### C 组 · 精度

BF16 vs FP8_MX，直接对标 Qwen3 的 MXFP8 口径。

| 实验 | 状态 | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |
|---|---|---|---|---|---|---|---|
| **C1** fp8_mx | ✅ | 1285.9 | **+50.6%** | 23.8% | 195 GB | 9396 | 13.95s |
| **C2** fp8_mx_mbs2 🏆 | ✅ | 1360.4 | **+59.3%** | 25.2% | 276 GB | 9945 | 26.36s |

### D 组 · 规模隔离

层数减半让权重减半，验证「权重挤占显存导致批次开不大」的假设。

| 实验 | 状态 | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |
|---|---|---|---|---|---|---|---|
| **D1** 40layer_bf16 | ✅ | 846.4 | -0.9% | 31.3% | 123 GB | 12114 | 10.82s |
| **D2** 40layer_bf16_mbs2 | ✅ | 892.7 | **+4.5%** | 33.1% | 201 GB | 12775 | 20.52s |
| **D3** 40layer_bf16_mbs4 | ❌ OOM | — | — | — | 276 GB | — | — |
| **D4** 40layer_fp8 | ✅ | 1272.3 | **+49.0%** | 23.6% | 105 GB | 18204 | 7.20s |
| **D5** fp8_ep16_mbs2 | ⚠️ HANG | — | — | — | 277 GB | — | — |
| **D6** fp8_mbs4 | ⏭ 跳过 | — | — | — | — | — | — |

### 单项影响力排名（按实测增益）

| 旋钮 | 增益 | 依据 | 备注 |
|---|---|---|---|
| **FP8_MX 精度** | **+50.6%** | A1 854.0 → C1 1285.9 | 最大杠杆，且省 30 GB 显存 |
| **CUDA graph** | **+44.6%** | A8 572.5 → A9 827.7 | 不开 graph 掉三分之一 |
| **a2a_overlap** | **+18.8%** | A3 718.6 → A1 854.0 | MoE 通信重叠，唯一显著的融合项 |
| **FP8 解锁的 MBS=2** | **+5.8%** | C1 1285.9 → C2 1360.4 | BF16 下此路不通 |
| 层数减半解锁的 MBS=2 | +5.5% | D1 846.4 → D2 892.7 | 与上条同源：腾显存换批次 |
| EP 32 → 16 | +3.3% | A1 854.0 → B4 882.3 | 需配 PP=2 才兑现 |
| full_iteration vs TE graph | +3.1% | A9 827.7 → A1 854.0 | 代价 +32 GB |
| router_fusion | +1.3% | A5 843.1 → A1 854.0 | 边际 |
| permute_fusion | **0%** | A6 854.2 ≈ A1 854.0 | 被 cutedsl 吸收，纯空转 |
| 并行度 PP/VPP | **~0%** | B1/B2/B3 全在 852–856 | **调并行不提性能** |

### full_iteration 的硬依赖（缺一即崩）

A2 / A4 / A7 三个 CRASH 共同勾勒出一条**强制依赖链**，不是「可选优化」：

```
cutedsl_fused_grouped_mlp   ← A2 关掉即 CRASH
  └→ use_transformer_engine_op_fuser=True      (overrides.py:238)
       └→ moe_expert_rank_capacity_factor      （固定专家容量）
            └→ moe_paged_stash                 ← A4 关掉即 CRASH
                 └→ full_iteration graph 才抓得住 dropless MoE 的变长 tensor
hybridep dispatcher         ← A7 换 alltoall 即 CRASH
```

### 显存墙：MBS=2 的四次尝试

| 尝试 | 层数 | 精度 | 手段 | 结果 |
|---|---|---|---|---|
| V5 | 80 | BF16 | full graph | ❌ OOM |
| B5 | 80 | BF16 | 退回 TE graph 省 32 GB | ⚠️ HANG @277 GB |
| B6 | 80 | BF16 | PP4 摊薄每 stage 激活 | ❌ OOM |
| **D2** | **40** | BF16 | **层数减半** | ✅ **892.7** |
| **C2** | 80 | **FP8** | **精度减半权重** | ✅ **1360.4** |

> 80 层 BF16 下 MBS=2 **无解**——换 graph、换并行都救不回来。
> 只有**减半权重**（减层 or 换 FP8）才能腾出激活空间。这是显存的物理约束，不是调参问题。

## 十一、为什么 full graph 之后只有 854？——完整归因

### 结论：**因为在跑 BF16。换 FP8_MX 就是 1360，与 Qwen3-235B 官方数字一模一样。**

| 配置 | 精度 | MBS | GPU | Model TFLOP/s | MFU | tok/s/GPU |
|---|---|---|---|---|---|---|
| A1 Hy3 冠军 | BF16 | 1 | 64 | 854.0 | **31.6%** | 6,242 |
| C1 Hy3 | **FP8_MX** | 1 | 64 | 1,285.9 | 23.8% | 9,396 |
| **C2 Hy3** | **FP8_MX** | **2** | **64** | **1,360.4** | **25.2%** | **9,945** |
| *Qwen3-235B（官方对标）* | *MXFP8* | *2* | ***256*** | *1,360* | *25.2%* | *—* |
| *DSV3 671B（官方对标）* | *MXFP8* | *1* | *256* | *1,658* | *30.7%* | *—* |

**C2 的 1360.4 / MFU 25.2% 与 Qwen3-235B 在 256 卡上的 1360 / 25.2% 完全重合，而我们只用了 64 卡。**

### 两个成因，且相互耦合

**① 精度口径（主因，+50.6%）**
Model TFLOP/s 的分子是模型的数学 FLOPs（与精度无关），分母是**该精度下的硬件峰值**。
GB300 BF16 峰值 2,700、FP8 峰值 5,400。跑 BF16 时算力天花板本来就只有一半。

**② FP8 省出的显存解锁了 MBS=2（+5.8%）**
这是最关键的一环，也印证了「权重太大开不了更大批次」的判断：

| 尝试 | 层数 | 精度 | 手段 | HBM | 结果 |
|---|---|---|---|---|---|
| V5 | 80 | BF16 | full graph | — | ❌ OOM |
| B5 | 80 | BF16 | 退 TE graph 省 32 GB | 277 GB | ⚠️ HANG |
| B6 | 80 | BF16 | PP4 摊薄每 stage | — | ❌ OOM |
| **D2** | **40** | BF16 | **层数减半** | 201 GB | ✅ 892.7（+5.5%） |
| **C2** | 80 | **FP8** | **精度减半权重** | 276 GB | ✅ 1,360.4（+5.8%） |

80 层 BF16 下 MBS=2 **三种打法全灭**。只有减半权重（减层 or 换 FP8）才腾得出激活空间。
**这不是调参问题，是显存的物理约束。**

因果链：
```
权重占显存 → 挤掉激活空间 → MBS 只能 = 1 → GEMM 形状小 → 单卡算力密度低
     ↑                                                            ↓
     └──── 换 FP8（权重减半）或减层，两条路都能解除 ────────────┘
```

### D 组的额外发现：TFLOP/s 与层数几乎无关

| 层数 | 精度 | TFLOP/s | HBM | tok/s/GPU |
|---|---|---|---|---|
| 80 | FP8 | 1,285.9 | 195 GB | 9,396 |
| 40 | FP8 | 1,272.3 | **105 GB** | **18,204** |
| 80 | BF16 | 854.0 | 225 GB | 6,242 |
| 40 | BF16 | 846.4 | **123 GB** | 12,114 |

**同精度下 80 层 vs 40 层的 TFLOP/s 差距仅 1.1%（FP8）/ 0.9%（BF16）。**
单卡算力密度由 **hidden_size × 精度 × 批次**决定，**与模型深度无关**。
层数只决定总工作量和显存，不影响效率——所以缩层调试是可靠的（07e 的 scale-in 方法论在此复现）。

而 BF16→FP8 的增益在两种层数下分别是 **+50.6%（80 层）和 +50.3%（40 层）**，
几乎完全一致 —— 说明**这 50% 是稳定的架构级收益，不是某个配置的偶然**。

### ⚠️ 纠正：我此前基于 GB200 数据得出的「MoE 上 FP8 只有 ±5%」是错的

第三节曾引用本仓 GB200 实测，断言「MoE 上 FP8 相对 BF16 只有 −5%~+5%，不值得在 bring-up 阶段引入」，
并据此把首跑定为 BF16。**该结论在 GB300 上不成立**：

| 平台 / 软件栈 | BF16 → FP8 |
|---|---|
| GB200 老栈（无 cutedsl） | **−5%**（DSV3 12L：527 → 503） |
| GB200 + CUDA graph | +4.5%（DSV3 32L：928 → 970） |
| **GB300 + cutedsl + full graph** | **+50.6%**（Hy3 80L：854 → 1,286） |

**关键差异是 `cutedsl_fused_grouped_mlp`**：老栈里 grouped GEMM 的 FP8 量化/反量化开销吃掉了
Tensor Core 红利，所以测不出收益；GB300 上融合核消除这层开销后，FP8 的算力优势才真正兑现。

> **教训**：性能结论**不能跨硬件代际和软件栈外推**。当时我用 GB200 的数据推 GB300 的选型，方向就错了。
> 正确做法是**先用小规模实跑验证再定基线**，而不是引用别的平台的数字。
> 这与「[判架构血缘要看 config 不看标签](#)」是同一类错误：**用二手结论替代一手实测**。

### 修正后的最优配置

```bash
python hy3_pretrain.py \
  --num-gpus 64 --tp 1 --pp 2 --vpp 8 --ep 32 \
  --mbs 2 --gbs 4096 --seq-length 4096 \
  --precision fp8_mx \
  --cuda-graph full_iteration --cutedsl --a2a-overlap \
  --recompute-modules --recompute-granularity none \
  --mtp-layers 0 --max-steps 30
# env: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32
```

**1,360.4 TFLOP/s/GPU · MFU 25.2% · 26.36 s/step · 276 GB/GPU · 9,945 tok/s/GPU**

> 显存 276 GB 已接近 288 GB 上限，**没有余量再叠加 EP16 或 MTP**（D5 实测三项叠加即卡死）。
> 若要同时开 MTP，需退回 MBS=1（C1 配置，1,285.9）或扩到 128 卡。

**BF16 场景下的最优**（若因精度对齐要求必须用 BF16）：**B4 = EP16，882.3 TFLOP/s / MFU 32.7%**。

---

## 十二、FP8 训练质量对齐验证（2026-07-26 05:35 ✅ 通过）

50% 的加速如果以训练质量为代价就毫无意义。本节用**受控对照实验**验证 FP8_MX 与 BF16 的等价性。

### 实验设计

由 [`loss_align.sh`](loss_align.sh) 执行。两轮**除精度外完全一致**：

| 变量 | 设置 |
|---|---|
| 并行 | TP1 / PP2 / VPP8 / **EP16** / MBS1 / GBS2048 / seq4096 |
| 性能开关 | full_iteration graph + cutedsl + a2a_overlap + paged stash，无 recompute |
| 数据 | mock（同一生成器，同 seed） |
| 步数 | 各 20 步 |
| **唯一差异** | `--precision bf16` vs `--precision fp8_mx` |

> **采集点：`yw-a-15`**。`lm loss` 只在**最后一个 pipeline stage** 打印，PP=2 / 16 pod 时即 15 号 pod。
> 在 rank 0 所在的 `yw-a-0` 上 grep 不到 loss —— 这点踩过一次，记下来。

### 逐步 loss 对照

| step | BF16 loss | FP8_MX loss | 绝对差 | 相对偏差 |
|---|---|---|---|---|
| 1 | 12.530730 | 12.529190 | -0.001540 | 0.0123% |
| 2 | 12.530780 | 12.529230 | -0.001550 | 0.0124% |
| 3 | 12.396570 | 12.403350 | +0.006780 | 0.0547% |
| 4 | 11.356320 | 11.359070 | +0.002750 | 0.0242% |
| 5 | 11.020130 | 10.998600 | -0.021530 | 0.1954% |
| 6 | 10.415350 | 10.413700 | -0.001650 | 0.0158% |
| 7 | 9.868304 | 9.863834 | -0.004470 | 0.0453% |
| 8 | 9.258566 | 9.255017 | -0.003549 | 0.0383% |
| 9 | 8.757781 | 8.755513 | -0.002268 | 0.0259% |
| 10 | 8.438774 | 8.435921 | -0.002853 | 0.0338% |
| 11 | 8.280344 | 8.276585 | -0.003759 | 0.0454% |
| 12 | 8.232012 | 8.223855 | -0.008157 | 0.0991% |
| 13 | 8.239738 | 8.231593 | -0.008145 | 0.0989% |
| 14 | 8.235600 | 8.229311 | -0.006289 | 0.0764% |
| 15 | 8.220032 | 8.218982 | -0.001050 | 0.0128% |
| 16 | 8.192741 | 8.194827 | +0.002086 | 0.0255% |
| 17 | 8.178925 | 8.184619 | +0.005694 | 0.0696% |
| 18 | 8.171684 | 8.175358 | +0.003674 | 0.0450% |
| 19 | 8.166067 | 8.167156 | +0.001089 | 0.0133% |
| 20 | 8.159149 | 8.157866 | -0.001283 | 0.0157% |

### 判定结果

| 判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| 最大相对偏差 | < 1% | **0.1954%** | ✅ 远优于阈值 |
| 平均相对偏差 | — | **0.0480%** | ✅ |
| 偏差是否随步数放大 | 不放大 | 前 10 步均值 0.0458% → 后 10 步 0.0502% | ✅ 未放大 |
| NaN iterations | 0 | **0** | ✅ |
| Skipped iterations | 0 | **0** | ✅ |
| 末步 loss | 应吻合 | BF16 **8.1591** vs FP8 **8.1579** | ✅ 差 0.0013 |
| 末步 grad norm | 应同量级 | FP8 **0.319**（0.315–0.322 平稳） | ✅ 无梯度异常 |

**偏差符号在正负之间随机振荡**（20 步里 6 次为正、14 次为负），**不是单向系统性漂移** ——
这是浮点数值噪声的典型特征，而非量化引入的偏置。若 FP8 真的损害训练，偏差应当同号且逐步累积放大。

最大偏差 0.1954% 出现在 **step 5**，正处于 loss 从 12.5 陡降到 8.2 的**最陡峭区段**；
该区段 loss 对任何微小扰动都最敏感，出现峰值偏差符合预期，且随后立刻回落到 0.02–0.09% 区间。

### 结论：**FP8_MX 可安全用于 Hy3 生产训练**

20 步窗口内 FP8 与 BF16 的训练轨迹在数值噪声级别上一致，**+50.6% 的吞吐提升是净收益，无质量代价**。
§十一 给出的 FP8 最优配置（1,360.4 TFLOP/s）**可直接采用**。

### 本验证的边界（不要过度解读）

1. **仅 20 步，只能证明短期数值稳定，不等于长程收敛等价。** FP8 的风险通常在训练中后期
   （梯度变小后量化相对误差上升）暴露。生产预训练建议至少跑 **1000+ 步**对照，并监控
   `grad norm` 与 loss spike 频率。
2. **mock 数据不是真实语料。** 真实数据的 token 分布更长尾，激活动态范围更大，对 FP8 更不利。
3. **本轮开着 `moe_router_force_load_balancing`**（benchmark 口径）。真实路由下专家负载倾斜，
   热专家的激活幅值更大，FP8 溢出风险高于此处测得的水平。
4. 若上生产，建议保留 **`first_last_layers_bf16`**（首末层用 BF16）这一常见保险，
   本轮用的是 Bridge `bf16_with_mxfp8_mixed()` 默认策略。
