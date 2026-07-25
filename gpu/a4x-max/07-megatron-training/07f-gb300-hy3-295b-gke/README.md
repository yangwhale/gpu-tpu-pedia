# 腾讯混元 Hy3 (295B-A21B) 64 GPU 训练 — GB300 NVL72 (A4X Max) GKE

GB300 (A4X Max) GKE 集群上的 **Hy3（混元 3，295B 总参 / 21B 激活 / 80 层 + 1 MTP）** 16 节点 64 GPU 训练 benchmark 准备文档。

> **当前状态：待跑（准备就绪）**。本文是开跑前的 recipe 设计 + 环境准备 + 预判踩坑，实测数据栏留空待填。
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
| GBS | **2048** 起步 | 须被 `MBS × DP` = 1×32 整除；跑通后按 07e 经验往上推（GBS 是收益最大的旋钮） |
| 精度 | **BF16**（首跑） | 对齐官方口径；MoE 上 FP8 收益存疑，见 §三 |
| cuda_graph_impl | **`full_iteration`** | 07e 的核心成果，勿手动覆盖 |
| mtp_num_layers | **首跑设 0，跑通后再开 1** | 见下方 pp_layout 说明 |

### 2.4 pp_layout 推导（80 层 vs DSV3 61 层）

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

### 2.5 备选配置（V1 跑不通时的回退）

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

### Step 0 — 确认单个 subblock 有 16 节点

```bash
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
kubectl --context $CTX get nodes -L cloud.google.com/reservation-subblocks -l team=yangwhale \
  --no-headers | awk '{print $NF}' | sort | uniq -c
```

不足 16 就给同 subblock 空闲节点补标签（**必须同一 subblock，不能跨池拼**，见 07e 教训）：

```bash
kubectl --context $CTX label node <NODE_NAME> team=yangwhale --overwrite
```

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
| 9 | **Hy3 专属**：末 stage 扛 5 层 + MTP + loss | 首跑关 MTP；OOM 就换前重后轻布局（§2.4） |

---

## 附：文件清单

| 文件 | 说明 |
|---|---|
| `yw-pool-64.yaml` | 单 ComputeDomain 16 节点 64 GPU sleep-infinity pod 池 |
| `run-hy3-yw.sh` | 单 pod 启动脚本（完整 env + rank 计算 + torchrun） |
| `hy3_provider.py` | Hy3 `GPTModelProvider` 构造（AutoBridge 路径 + from-scratch 路径 + 训练覆盖） |

---

*2026-07-25 · Recipe 设计完成，待 16 节点 64 GPU 实跑 · MoE 血统 = DeepSeek V3，Attention = GQA*
