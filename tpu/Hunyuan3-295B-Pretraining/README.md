# 腾讯混元 Hy3 (295B-A21B) TPU 预训练 — MaxText on v7 / v5p

在 TPU v7 (Ironwood) 和 v5p 上用 MaxText 预训练腾讯混元 3 的方案与基线。

**本文的定位**：GB300 侧已跑通并拿到完整基线（BF16 854 / FP8_MX 1360 TFLOP/s/GPU），
本文把那套配置搬到 TPU，先回答"能不能跑"，再回答"跑多快"。

> **模型架构的唯一真源（SSOT）是 GB300 那份文档**：
> [`gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/README.md`](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/README.md)。
> 本文的架构参数全部引自该文，不重新解读 HF config。
>
> MaxText 侧的操作范式参照
> [`tpu/DeepSeek-V3.2-Training`](../DeepSeek-V3.2-Training/README.md)。

---

## 结论先行：MaxText 支持现状

**扒过 MaxText 源码（`/tmp/mt-stable/maxtext`，42 个 model config）后的判断：
没有现成的 decoder block 能直接跑混元 3，两个候选各缺一半。**

混元 3 的结构是 **GQA attention + DeepSeek V3 血统的 MoE**。MaxText 里这两半分属两个 block：

| 混元 3 需要 | `decoder_block: "qwen3_moe"` | `decoder_block: "deepseek"` | 源码位置 |
|---|---|---|---|
| GQA attention | ✅ | ❌ **硬编码 MLA** | `layers/deepseek.py:72` 直接调 `attention_mla.mla_as_linen()` |
| sigmoid 路由 | ✅ | ✅ | `layers/moe.py:317` GateLogit 读 `routed_score_func`，两个 block 共用 |
| expert bias（aux-loss-free） | ✅ | ✅ | `layers/moe.py:316` 读 `routed_bias`，同上 |
| shared expert ×1 | ❌ **拿不到** | ✅ | `get_routed_moe()` (`moe.py:1864`) 硬编码返回 `RoutedMoE`；带 shared 的是 `RoutedAndSharedMoE` (`moe.py:1752`) |
| 第 0 层 dense（`first_k_dense_replace=1`） | ❌ **不支持** | ✅ | `first_num_dense_layers` 6 处消费点全在 `if cfg.decoder_block == DEEPSEEK` 分支内（`layers/decoders.py:670+`） |
| MTP ×1 | ✅ | ✅ | `layers/multi_token_prediction.py` 独立模块，与 block 无关 |

**好消息**：DSV3 的三件套（sigmoid / expert bias / routed_scaling）是 MaxText 的**全局 MoE 参数**，
定义在 `configs/base.yml` 而非 deepseek 专属，所以 GQA 侧也能用。缺的只是
**shared expert** 和 **dense 首层**两项。

### 三条落地路径

| | 做法 | 架构保真度 | 工作量 | 适合 |
|---|---|---|---|---|
| **A** | `qwen3_moe` + 近似（80 层全 MoE、无 shared expert） | 有偏差 | 只写 config | **先拿性能基线**，判断 TPU 值不值得投 |
| **B** | 新增 `hunyuan3` decoder block（GQA + `RoutedAndSharedMoE` + dense 首层） | 完全一致 | 一个 layer 类 + `decoders.py` 分支 + model config | 要跑真实预训练 |
| **C** | 改 `deepseek.py` 让 attention 可配 | 完全一致 | 改动面大，会碰现有 DSV3 路径 | 不推荐 |

**建议先 A 后 B**。路径 A 的架构偏差要算清楚，它是**双向**的：

| 变化 | 参数量 |
|---|---|
| 去掉 79 层的 shared expert | −1.49 B |
| 第 0 层 dense FFN 变成 MoE 层 | −0.16 B，+3.62 B |
| **净变化** | **+1.97 B（+0.67%）** |

注意方向是**变大**不是变小——第 0 层从 dense（0.16 B）换成 MoE（3.62 B）
的增量盖过了砍掉 shared expert 的减量。计算量随之同向变动，
所以 A 跑出来的 TFLOP/s 与真实架构有 <1% 的系统性偏差，**方向偏乐观**。

这个量级对"v7/v5p 上 MoE all-to-all 跑不跑得动"这个真正的未知数没有影响，
所以先用 A 拿基线是划算的；但正式对外报数字前必须走路径 B。

---

## 一、模型架构（引自 GB300 SSOT）

### 1.1 结构参数

| 项 | 值 |
|---|---|
| 层数 | **80** |
| hidden_size | **4096** |
| ffn_hidden_size（dense 层） | **13312** |
| attention heads | **64** |
| KV groups（GQA） | **8** |
| head_dim | **128** |
| vocab_size | **120832** |
| rope theta | **11158840.0** |
| normalization | RMSNorm |
| 激活 | SwiGLU (`silu` + `linear`) |
| QK LayerNorm | **是** |
| QKV bias | 无 |
| tie embeddings | 否（untied） |

### 1.2 MoE 参数（DSV3 血统）

| 项 | 值 |
|---|---|
| routed experts | **192** |
| top-k | **8** |
| moe_ffn_hidden_size | **1536** |
| shared experts | **1**（intermediate 1536） |
| dense 层分布 | 第 0 层 dense，1–79 层 MoE |
| 路由打分 | **sigmoid** |
| expert bias（aux-loss-free） | **启用** |
| routed scaling factor | **2.826** |
| MTP 层数 | **1** |

### 1.3 参数量分解 — 决定并行策略的关键

| 组成 | 参数量 | 占比 |
|---|---|---|
| 路由专家 | 286.2 B | **97.0%** |
| 共享专家 | 1.49 B | 0.5% |
| Attention | 6.04 B | 2.0% |
| Dense FFN（第 0 层） | 0.16 B | 0.1% |
| Embedding + LM head | 0.99 B | 0.3% |
| **合计** | **≈ 294.9 B** | |

> **97% 的参数在专家里** → **专家并行（EP）是唯一有意义的显存旋钮，
> TP 对这个模型几乎无用**（attention 只占 2%，切它纯亏通信）。
> 这条结论跨硬件成立，TPU 侧同样适用。

---

## 二、Megatron → MaxText 参数映射

GB300 侧用 Megatron `GPTModelProvider`，TPU 侧用 MaxText model config。逐项对照：

### 2.1 结构参数

| Megatron（GB300） | 值 | MaxText | 备注 |
|---|---|---|---|
| `num_layers` | 80 | `base_num_decoder_layers: 80` | |
| `hidden_size` | 4096 | `base_emb_dim: 4096` | |
| `ffn_hidden_size` | 13312 | `base_mlp_dim: 13312` | 仅 dense 层用 |
| `num_attention_heads` | 64 | `base_num_query_heads: 64` | |
| `num_query_groups` | 8 | `base_num_kv_heads: 8` | GQA |
| `kv_channels` | 128 | `head_dim: 128` | |
| `vocab_size` | 120832 | `vocab_size: 120832` | |
| `rotary_base` | 11158840.0 | `rope_max_timescale: 11158840` | |
| `qk_layernorm: True` | | `use_qk_norm: True` | |
| `gated_linear_unit` | True | `mlp_activations: ["silu","linear"]` | |
| `untie_embeddings_and_output_weights` | True | `logits_via_embedding: False` | |

### 2.2 MoE 参数

| Megatron（GB300） | 值 | MaxText | 路径 A 可用？ |
|---|---|---|---|
| `num_moe_experts` | 192 | `num_experts: 192` | ✅ |
| `moe_router_topk` | 8 | `num_experts_per_tok: 8` | ✅ |
| `moe_ffn_hidden_size` | 1536 | `base_moe_mlp_dim: 1536` | ✅ |
| `moe_router_score_function` | sigmoid | `routed_score_func: "sigmoid"` | ✅ |
| `moe_router_enable_expert_bias` | True | `routed_bias: True` | ✅ |
| `moe_router_topk_scaling_factor` | 2.826 | `routed_scaling_factor: 2.826` | ✅ |
| `moe_shared_expert_intermediate_size` | 1536 | `shared_experts: 1` | ❌ 见结论先行 |
| `first_k_dense_replace` | 1 | `first_num_dense_layers: 1` | ❌ 同上 |
| `mtp_num_layers` | 1 | `mtp_num_layers: 1` | ✅ |
| — | | `decoder_block: "qwen3_moe"` | 路径 A 用 |

### 2.3 结构最接近的现成范本

`configs/models/qwen3-235b-a22b.yml` 与混元 3 的骨架高度重合，可作为起草基础：

| 参数 | Qwen3-235B | 混元 3 | |
|---|---|---|---|
| `base_emb_dim` | 4096 | 4096 | 相同 |
| `base_num_query_heads` | 64 | 64 | 相同 |
| `head_dim` | 128 | 128 | 相同 |
| `base_moe_mlp_dim` | 1536 | 1536 | 相同 |
| `num_experts_per_tok` | 8 | 8 | 相同 |
| `use_qk_norm` | True | True | 相同 |
| `base_num_decoder_layers` | 94 | **80** | 要改 |
| `num_experts` | 128 | **192** | 要改 |
| `base_num_kv_heads` | 4 | **8** | 要改 |
| `vocab_size` | 151936 | **120832** | 要改 |
| 路由 | softmax + `load_balance_loss_weight` | **sigmoid + bias** | 要改 |

> Qwen3 走的是 softmax 路由 + 辅助损失均衡，混元 3 走 sigmoid + expert bias 的
> aux-loss-free 路线。**这两套不能混用**——套 Qwen3 模板时必须把
> `load_balance_loss_weight` / `norm_topk_prob` 去掉，换成 `routed_*` 三件套。

### 2.4 训练必须覆盖的默认值

GB300 文档 §1.3 指出，权重转换用的默认值直接拿去 from-scratch 预训练会
**导致专家负载失衡**。TPU 侧的等价项：

| 项 | 说明 |
|---|---|
| expert bias 更新率 | Megatron 侧要设 `moe_router_bias_update_rate=1e-3`（默认 0 则 bias 永不更新，aux-loss-free 机制形同虚设）。**MaxText 侧的对应参数待确认**，见待验证清单 |
| 辅助损失 | 保持关闭（aux-loss-free 路线），不要开 `load_balance_loss_weight` |
| SFT 场景例外 | 加载官方权重做 SFT 时 bias 更新率保持 0 更稳，只有 from-scratch / continued-pretrain 才需要开 |

---

## 三、GB300 基线（已实测，供 TPU 对标）

引自 GB300 文档 §10–§11。64 GPU（16 节点单 NVLink 域），TP=1 / PP=2 / VPP=8 / EP=32。

| 配置 | 精度 | MBS | GPU 数 | Model TFLOP/s | MFU | tok/s/GPU |
|---|---|---|---|---|---|---|
| A1 冠军 | BF16 | 1 | 64 | **854.0** | **31.6%** | 6,242 |
| C1 | FP8_MX | 1 | 64 | 1,285.9 | 23.8% | 9,396 |
| **C2 最快** | **FP8_MX** | **2** | **64** | **1,360.4** | **25.2%** | **9,945** |

MFU 分母：GB300 BF16 峰值 2,700 TFLOPS，FP8 峰值 5,400 TFLOPS。

### 三条可跨硬件迁移的结论

1. **TP 无用**：attention 只占 2% 参数，切它纯亏通信 → TPU 侧同样应以 EP/FSDP 为主
2. **BF16 是官方口径**：腾讯官方全线 BF16（LLaMA-Factory / ms-swift / DeepSpeed 配置一致），
   `Hy3-FP8` 是 **推理量化产物**不是训练精度。TPU 侧首跑也应用 BF16 对齐口径
3. **显存决定 MBS**：GB300 上 80 层 BF16 想开 MBS=2，full graph / 退 TE graph / PP4
   三种打法全部失败，只有减半权重（减层或换 FP8）才行。
   **这是显存的物理约束，不是调参问题** → TPU 侧要提前算显存，别指望调参绕过

---

## 四、TPU 侧配置设计

### 4.1 硬件对照

| | v5p-256 | v7 4x4x4 | GB300（参考） |
|---|---|---|---|
| 芯片数 | 128 | 64 | 64 GPU |
| JAX device 数 | **128**（1 dev/chip，MegaCore） | **128**（2 dev/chip） | — |
| HBM / chip | 95 GB HBM2e | 192 GB HBM3e | 288 GB |
| 总 HBM | 12.16 TB | 12.29 TB | 18.4 TB |
| BF16 TFLOPS / chip | 459 | 2,306 | 2,700 |
| 总 BF16 算力 | 58.8 PFLOPS | 147.6 PFLOPS | 172.8 PFLOPS |

> **v5p-256 与 v7 4x4x4 是正确的对照组**：device 数同为 128，总 HBM 接近
> （12.16 vs 12.29 TB），静态状态的每-device 分片相同。差异集中在单卡算力（5×）
> 和互联代际。单位换算见 [TPU-UNITS](https://github.com/yangwhale/tpu-recipes/blob/main/training/TPU-UNITS.md)。

### 4.2 显存测算（BF16，128 devices）

| 组成 | 计算 | 总量 | 每 device |
|---|---|---|---|
| 权重 BF16 | 294.9 B × 2 B | 590 GB | 4.6 GB |
| 梯度 BF16 | 294.9 B × 2 B | 590 GB | 4.6 GB |
| Adam 状态 + FP32 master | 294.9 B × 12 B | 3.54 TB | 27.7 GB |
| **静态小计** | | **4.72 TB** | **36.9 GB** |
| 可用 HBM / device | | | v5p 95 GB / v7 96 GB |
| **激活余量** | | | **≈ 58 GB** |

静态状态占 **38%**，两代都留出足够激活空间。**这个规模不需要 PP**，
纯 FSDP + EP 即可（对比 GB300 因单域只有 64 卡才需要 PP=2）。

### 4.3 起步配置（待验证）

| 参数 | v7 4x4x4 | v5p-256 | 理由 |
|---|---|---|---|
| `ici_fsdp_parallelism` | 待定 | 待定 | 与 EP 组合，见下 |
| `ici_expert_parallelism` | 待定 | 待定 | 97% 参数在专家里，EP 是主旋钮 |
| `ici_tensor_parallelism` | **1** | **1** | attention 仅 2%，TP 纯亏 |
| `per_device_batch_size` | 1（首跑） | 1（首跑） | 跑通后上探 |
| `max_target_length` | 4096 | 4096 | 对齐 GB300 benchmark 口径 |
| `dtype` / `weight_dtype` | bfloat16 | bfloat16 | 对齐官方精度 |
| `megablox` / `sparse_matmul` | True / True | True / True | Dense MoE 必 OOM，见 DSV3 文档踩坑 #4 |
| `attention` | 待定 | 待定 | v7 上 flash 有编译问题，见 DSV3 文档踩坑 #3 |
| `dataset_type` | synthetic | synthetic | 首轮只测吞吐 |

> EP / FSDP 的具体配比是**本轮要扫的第一个维度**。GB300 上 EP=32 是甜点
> （192 experts / 32 = 6 专家/rank），TPU 侧 device 数是 128，
> 候选 EP ∈ {8, 16, 32, 64}，需实测。

---

## 五、测试矩阵（**待填**）

> 以下所有性能数字**留空**，等实测后填入。空表本身就是测试计划。

### 5.1 v7 (Ironwood) 4x4x4 — 64 chips / 128 devices

| # | EP | FSDP | MBS | attention | step 时间 | TFLOP/s/device | MFU | HBM/device | tok/s/device | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 8 | 16 | 1 | dot_product | | | | | | ⬜ |
| V2 | 16 | 8 | 1 | dot_product | | | | | | ⬜ |
| V3 | 32 | 4 | 1 | dot_product | | | | | | ⬜ |
| V4 | 64 | 2 | 1 | dot_product | | | | | | ⬜ |
| V5 | 最优 | | 2 | dot_product | | | | | | ⬜ |
| V6 | 最优 | | 1 | flash | | | | | | ⬜ |

MFU 分母：**2,306** TFLOPS/chip；v7 是 2 device/chip，
**per-chip TFLOP/s = 日志值 × 2**（换算见 TPU-UNITS）。

### 5.2 v5p-256 — 128 chips / 128 devices

| # | EP | FSDP | MBS | attention | step 时间 | TFLOP/s/device | MFU | HBM/device | tok/s/device | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 8 | 16 | 1 | dot_product | | | | | | ⬜ |
| P2 | 16 | 8 | 1 | dot_product | | | | | | ⬜ |
| P3 | 32 | 4 | 1 | dot_product | | | | | | ⬜ |
| P4 | 64 | 2 | 1 | dot_product | | | | | | ⬜ |
| P5 | 最优 | | 2 | dot_product | | | | | | ⬜ |

MFU 分母：**459** TFLOPS/chip；v5p 是 MegaCore，**1 device = 1 chip，日志值不用乘 2**。

### 5.3 三方横向对比（**待填**）

| | GB300 64 GPU | v7 4x4x4 | v5p-256 |
|---|---|---|---|
| 计算单元数 | 64 GPU | 64 chips | 128 chips |
| BF16 峰值/单元 | 2,700 | 2,306 | 459 |
| **实测 TFLOP/s/单元** | **854.0** | ⬜ | ⬜ |
| **MFU** | **31.6%** | ⬜ | ⬜ |
| **tok/s/单元** | **6,242** | ⬜ | ⬜ |
| **整机吞吐 tok/s** | **399,488** | ⬜ | ⬜ |

> 对比时统一到 **per-chip** 口径。v7 日志是 per-device，需 ×2；v5p 不需要。
> 这是跨代际比较最容易出错的一步。

---

## 六、待验证清单

按依赖顺序排，前面不通后面免谈：

| # | 事项 | 为什么关键 |
|---|---|---|
| 1 | `decoder_block: "qwen3_moe"` + `routed_score_func: "sigmoid"` + `routed_bias: True` 能否正常构图 | 路径 A 的前提。源码上共用 `GateLogit` 应该可行，但没人这么配过 |
| 2 | 192 experts × 80 层能否在 128 devices 上编译出来 | DSV3 671B 在 v7 上曾出现 sparse matmul 编译 6 小时未完成 |
| 3 | MaxText 里 expert bias 的更新率参数叫什么 | Megatron 有 `moe_router_bias_update_rate`，MaxText 侧未找到对应项。若无，aux-loss-free 均衡可能失效 |
| 4 | EP / FSDP 最优配比 | 97% 参数在专家里，这是第一性能旋钮 |
| 5 | `attention=flash` 在 v7 上能否编译通过 | DSV3 上踩过坑（踩坑 #3，70+ 分钟未完成），需确认 GQA 是否同样受影响 |
| 6 | MTP 开启后的开销 | GB300 侧建议首跑设 0，跑通再开 |
| 7 | 路径 B 的工作量评估 | 若路径 A 基线可观，再投入写 `hunyuan3` block |

---

## 七、参考

| 来源 | 说明 |
|---|---|
| [GB300 混元 3 训练文档](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/README.md) | **架构 SSOT** + 基线 + 27 组消融 + 归因 |
| [GB300 混元 3 SFT 文档](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/SFT.md) | Bridge 移植、权重转换、评测闭环 |
| [DeepSeek V3.2 TPU 训练](../DeepSeek-V3.2-Training/README.md) | MaxText 操作范式 + v7 MoE 踩坑 |
| [ant-pretrain](../ant-pretrain/README.md) | MaxText fork 在 v7 上的训练实践 |
| `configs/models/deepseek3-671b.yml` | DSV3 MoE 参数命名范本 |
| `configs/models/qwen3-235b-a22b.yml` | GQA + MoE 结构范本（骨架最接近） |
| [tencent/Hy3](https://huggingface.co/tencent/Hy3) | 官方权重与 config |

---

## 当前状态

**规划阶段** — 架构映射与配置设计已完成，尚未上机。

**下一步**：按待验证清单第 1 项起步，先在小规模（如 v7 2x2x2）上验证
`qwen3_moe` + sigmoid 路由能否构图，再放大到 4x4x4 跑基线。
