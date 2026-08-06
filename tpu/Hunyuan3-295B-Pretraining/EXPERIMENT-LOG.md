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


## 结论先行

**MaxText 原生不支持腾讯混元 3。所需组件全在，已拼出
`decoder_block: "hunyuan3"`，并在 v5p（256 芯片）和 v7 Ironwood（64 芯片）
上跑通 80 层完整 295B-A21B。**

| | v5p 256 chips | v7 Ironwood 64 chips | GB300 64 GPU（参照） |
|---|---|---|---|
| 参数量（框架报） | 298.786 B | 298.786 B（逐位一致） | — |
| 稳态 step | 63.17 s | 20.43 s | — |
| **TFLOP/s / 计算单元** | **161.0** | **445.1** | 854.0 |
| **MFU** | **35.07%** | 19.29% | 31.6% |
| 整机 tok/s | 265,588 | 205,314 | 399,488 |
| 调优状态 | ✅ 已收敛 | 🔄 **进行中**，距 DSV3 水位还差 1.38× | 已调优 |

> v5p 这一列是**当前照 `run.sh` 能跑出来的水位**（2026-07-29 换项目 / 换 VPC /
> 换集群从零复现过，§7.3 战线三 b0″）。旧栈上曾拿到 **168.6 / 36.72%**，
> 但那套产物已随 §5.5 的合并删除，不再可复现 —— 本文凡是并列出现两个 v5p 数字的地方，
> **较低的那个才是你能拿到的**。

- **v5p 的 35.07% 已经超过 GB300 的 31.6%**，而单卡算力只有它的 1/5.9。
- **v7 首跑 17.54%，调到 19.29%**，目标是 DeepSeek V3 在同一硬件上的
  实测水位 612.7 TFLOP/s / 26.6%（§6.1 有完整参照表）。

### 三件最值得先看的事

1. **[MFU 从 2.45% 一路到 36.72%，涨了 15 倍](#41-我犯的错没有从官方配方出发)**
   —— 其中 **12.9 倍是"照抄官方 DeepSeek3 v5p 配方、只换模型名"这一步**换来的，
   剩下的才是调参。
2. **[TPU 上专家并行是负优化](#42-为什么-tpu-上-fsdp-打得过-ep)**
   —— 跟 GPU 结论相反。EP=64 不只是慢，是直接超显存 326 GB。
3. **[同一个 bug 模式在本项目出现 10 次](#八十二个-bug-与静态验证的边界)**
   —— MaxText 里每个"按模型家族名字列举"的分支都要单独补，漏了不报错。

### 交付物

| 交付项 | 内容 |
|---|---|
| **代码分支** | [`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3) —— 基于上游 main，三个 commit，按两个上游 PR 的边界拆开 |
| **一键复现（两个平台）** | [`prep.sh`](maxtext-hunyuan3/prep.sh) 拉分支 + 自检 + 传 GCS → [`run.sh`](maxtext-hunyuan3/run.sh) `PLATFORM=v5p\|v7`。流程见 §九 |
| 模型代码 | [`src/maxtext/models/hunyuan3.py`](https://github.com/yangwhale/maxtext/blob/hunyuan3/src/maxtext/models/hunyuan3.py) —— 约 160 行 nnx。**零新数学**：attention 继承 Qwen3，MoE 直接复用 DeepSeek 的 `RoutedAndSharedMoE`，本文件只做接线 |
| 模型配置 | [`hunyuan3-295b.yml`](https://github.com/yangwhale/maxtext/blob/hunyuan3/src/maxtext/configs/models/hunyuan3-295b.yml) / [`hunyuan3-smoke.yml`](https://github.com/yangwhale/maxtext/blob/hunyuan3/src/maxtext/configs/models/hunyuan3-smoke.yml) —— 在 `src/maxtext/configs/models/`，每个值对着 HF `config.json` 抄 |
| MaxText 侧改动 | 分支上的 commit，涉及**上游 12 个文件**。光靠 config 做不到，因为 MaxText 到处按模型家族名字分支（§八、[移植指南](MAXTEXT-PORTING-GUIDE.md)） |
| 未做 | 权重转换、真实数据集收敛验证（§十一 清单 #12/#13） |

### 怎么读这份文档

| 你想知道 | 看 |
|---|---|
| Hy3 是什么结构、跟 DSV3 差在哪 | §一 |
| 怎么在 MaxText 里把它拼出来 | §二 |
| 跑通过程和踩的坑 | §三（v5p）、§五（v7） |
| 性能怎么调上去的、每一项值多少 | §四（v5p）、§六（v7） |
| 所有轮次的实测数据 | §7.3（三条战线一张表） |
| 三个平台横向比 | §7.4 |
| 这个项目最大的教训 | §八 |
| **怎么把别的模型移植到 MaxText** | **[MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md)**（独立文档，写给外部团队） |
| **只想在 v5p 上把它跑起来** | **[QUICKSTART-v5p.md](QUICKSTART-v5p.md)**（不含历史过程，照着走就能复现） |
| 怎么部署、怎么跑测试 | §九 |
| 这些改动该 check in 到哪 | §十 |


## 一、模型架构：Hy3 到底是什么


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


### 1.2 MoE 参数（DeepSeek V3 血统）

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


### 1.4 但 Hy3 的 MoE 与 DeepSeek V3 并非完全相同

GB300 文档写的是"MoE 是 DSV3 配方的一比一移植"。读完两边代码，
**路由的六步数学确实逐步一致，但有一处机制 DSV3 有、Hy3 没有**：

| | DeepSeek V3 | Hunyuan 3 |
|---|---|---|
| sigmoid 打分 | ✅ | ✅ |
| expert bias 选择 / 原始值作权重 | ✅ | ✅ |
| 归一化 + routed_scaling | ✅ | ✅ |
| shared expert | ✅ | ✅ |
| **device-limited routing（专家分组）** | **✅ 有** | **❌ 没有** |
| 专家数 / top-k | 256 / 8 | 192 / 8 |

**device-limited routing**：DSV3 把 256 个专家分成 8 组，一个 token 先选 4 个组，
再在这 4 组内部选 top-8。目的是**限制单个 token 最多只跟 4 个设备通信**，
压缩 all-to-all 的扇出。MaxText 里对应 `n_routing_groups` / `topk_routing_group`，
`deepseek3-671b.yml` 的注释明确要求设成 8 / 4。

Hy3 **没有这套**——`HYV3TopKRouter` 直接在全部 192 个专家里做全局 top-8
（grep 官方代码，路由部分没有任何 group 相关字段）。代价是 all-to-all 扇出更大，
换来的是路由自由度更高。

> 本 config **没有设** `n_routing_groups`，MaxText 默认 `-1`（禁用），
> 正好匹配 Hy3。但这是**碰巧对上的**，本轮才实际核过——
> 如果哪天有人"参照 DSV3 配方"把这两项加进来，路由行为就变了。

另一处极小的差异：Hy3 归一化时写的是 `sum + 1e-20`，
DSV3 参考实现和 MaxText 都是裸 `sum`。只有 top-8 分数全部趋近 0 时才有区别，
实践中不会触发。


### 1.5 路由数学：官方原文与本轮修复

官方 `HYV3TopKRouter.forward()`：

```python
routing_weights   = torch.sigmoid(router_logits)              # 1. sigmoid
scores_for_choice = routing_weights + e_score_correction_bias # 2. 加 bias
_, top_k_index    = torch.topk(scores_for_choice, top_k)      # 3. 用「含 bias」的分数选
top_k_weights     = routing_weights.gather(1, top_k_index)    # 4. 取「不含 bias」的值
top_k_weights    /= top_k_weights.sum(-1, keepdim=True)       # 5. 归一化
top_k_weights    *= router_scaling_factor                     # 6. × 2.826
```

**第 3、4 步是 aux-loss-free 的精髓**：bias 只影响*选谁*，不影响*权重多少*。

MaxText 里这条路径由 `deepseek_routing()` 实现，逻辑完全一致
（`take_along_axis(pre_bias_logits, top_k_indices)`）。**但它被
`model_name.startswith("deepseek3")` 挡住了 —— 当时 `moe.py` 里共 4 处**
（分支跟到上游最新后是 5 处，且判断里多了 `deepseek4`）。
`hunyuan3-295b` 不匹配，于是：

- 不保存 `pre_bias_logits`（为 None）
- 退回 `jax.lax.top_k(gate_logits)` —— **gate_logits 是加过 bias 的**，
  于是权重里混进了 bias，第 4 步的语义丢失
- 相应的 sharding 约束也随之跳过

**形状全对、不报任何错、参数量不变**——只有权重数值是错的。
每一处都已把 `hunyuan3` 加进 `startswith` 的名单，
并加了验证脚本第 7 项（负向测试过：还原任意一处即报错）。

> **这是 MaxText 的一处设计债**：用模型名字符串前缀决定路由算法。
> 更健壮的做法是提一个 config flag（如 `use_pre_bias_routing_weights`）。
> 这里选择最小侵入，是为了不改动 deepseek2 / kimi-k2 的现有行为
> ——它们的 `decoder_block` 同为 `deepseek`，但 model_name 不以 deepseek3 开头，
> 改判断依据会波及它们。


## 二、在 MaxText 里实现 hunyuan3


### 2.1 为什么两个现成 block 都不行

**扒过 MaxText 源码（42 个 model config）后的判断：
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


### 2.2 曾经考虑过的三条路

| | 做法 | 架构保真度 | 结论 |
|---|---|---|---|
| A | `qwen3_moe` + 近似（80 层全 MoE、无 shared expert） | 偏差 +0.67% | **放弃**，见下 |
| **B** | **新增 `hunyuan3` block（GQA + `RoutedAndSharedMoE` + dense 首层）** | **完全一致** | **✅ 已实现** |
| C | 改 `deepseek.py` 让 attention 可配 | 完全一致 | 放弃，会碰现有 DSV3 路径 |

原本打算先用 A 拿基线。但真去算 A 的偏差时发现它是**双向**的，而且方向反直觉：

| 变化 | 参数量 |
|---|---|
| 去掉 79 层的 shared expert | −1.49 B |
| 第 0 层 dense FFN 变成 MoE 层 | −0.16 B，+3.62 B |
| **净变化** | **+1.97 B（+0.67%）** |

方向是**变大**不是变小——第 0 层从 dense（0.16 B）换成 MoE（3.62 B）的增量
盖过了砍掉 shared expert 的减量，计算量同向变动，**基线会偏乐观**。
既然路径 B 实际只要写一个接线文件，就没必要留一个方向已知偏乐观的近似值在文档里。

---


### 2.3 复用了什么，新写了什么

代码全部在 [`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3)
上，本仓不留副本（理由见 §10.2）：

| 文件 | 作用 |
|---|---|
| [`models/hunyuan3.py`](https://github.com/yangwhale/maxtext/blob/hunyuan3/src/maxtext/models/hunyuan3.py) | 两个 decoder layer 类，**只做接线** |
| [`configs/models/hunyuan3-295b.yml`](https://github.com/yangwhale/maxtext/blob/hunyuan3/src/maxtext/configs/models/hunyuan3-295b.yml) | model config，值全部来自 SSOT |
| 上游 12 个文件的改动 | 同在该分支的三个 commit 里 |

**新写的只有装配逻辑**，两半功能都是原样引入的：

| 组件 | 来源 | 是否修改 |
|---|---|---|
| GQA attention + QK-LayerNorm | `qwen3.self_attention_with_norm` | 原样调用 |
| sigmoid 路由 + expert bias + shared expert | `moe.get_routed_and_shared_moe` | 原样调用 |
| 收尾（metrics sow + scan 返回约定） | `deepseek.post_process` | 原样调用 |
| dense / MoE 分层扫描 | `decoders.py` 里 DeepSeek 那套 | 扩大条件，未改逻辑 |

两个类各自只有约 40 行有效代码：

- `Hunyuan3DenseLayer` — 层 0：qwen3 attention + 宽度 13312 的 SwiGLU
- `Hunyuan3MoELayer` — 层 1–79：qwen3 attention + DeepSeek MoE 块


### 2.4 注册处的三个要点

1. **返回两元素列表**。`get_decoder_layers()` 对 hunyuan3 返回
   `[Hunyuan3DenseLayer, Hunyuan3MoELayer]`，正好满足 `decoders.py` 里
   `assert len(RemattedBlockLayers) == 2` —— 于是 `first_num_dense_layers`
   那套为 DeepSeek 写的扫描机制**原封不动地驱动了 Hy3**。
2. **4 处条件判断从 `==` 改成 `in`**。原本写死
   `cfg.decoder_block == DecoderBlockType.DEEPSEEK` 的地方（分层扫描、
   pipeline stage 选择）改为 `in (DEEPSEEK, HUNYUAN3)`，不改逻辑只扩范围。
3. **必须用 `get_routed_and_shared_moe` 而不是 `get_routed_moe`**。
   后者硬编码返回裸 `RoutedMoE`，会**静默丢掉** Hy3 的共享专家——
   不报错，只是参数量少 1.49 B、精度对不上。这是本次最容易踩空的一处。


### 2.5 Megatron → MaxText 参数映射

GB300 侧用 Megatron `GPTModelProvider`，TPU 侧用 MaxText model config。逐项对照：

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

GB300 文档 §1.3 指出，权重转换用的默认值直接拿去 from-scratch 预训练会
**导致专家负载失衡**。TPU 侧的等价项：

| 项 | 说明 |
|---|---|
| expert bias 更新率 | Megatron 侧要设 `moe_router_bias_update_rate=1e-3`（默认 0 则 bias 永不更新，aux-loss-free 机制形同虚设）。**MaxText 侧的对应参数待确认**，见待验证清单 |
| 辅助损失 | 保持关闭（aux-loss-free 路线），不要开 `load_balance_loss_weight` |
| SFT 场景例外 | 加载官方权重做 SFT 时 bias 更新率保持 0 更稳，只有 from-scratch / continued-pretrain 才需要开 |

---


### 2.6 对照 HF config 原文的审计

上面的实现最初是照 GB300 文档（二手）写的。之后拉了
`tencent/Hy3` 的 `config.json` **原文**逐项核对，**抓到 2 个真错误**。

**错误 1：`rms_norm_eps` 差 10 倍。** HF 是 `1e-05`，我沿用了 qwen3/deepseek
惯用的 `1.0e-6`。二手文档没列这一项，我按"同类模型都这么写"填的默认值。

**错误 2（严重，静态验证抓不到）：路由权重会被 softmax 覆盖。**
`moe.py` 里选择路由权重算法的分支是按 block 类型硬判断的：

```python
if decoder_block == DEEPSEEK:
    top_k_weights = self.deepseek_scale_weights(...)   # 归一化 + × 2.826
elif decoder_block != LLAMA4:
    top_k_weights = softmax(top_k_weights)             # ← hunyuan3 掉进这里
```

`hunyuan3` 不等于 `DEEPSEEK`，于是 sigmoid 打出来的分数被 softmax 重新压一遍，
**`routed_scaling_factor=2.826` 完全不生效**。参数量一个字节都不变，
所以前五项自检全绿——这类错误只能靠读路由的实际数学来抓。
已把该分支改为 `in (DEEPSEEK, HUNYUAN3)`，并给验证脚本加了第 6 项专门盯它
（做过负向测试：把改动还原，脚本会失败）。

**另外差点犯的一个错**：HF 有 `route_norm: true`，看上去该映射到 MaxText 的
`norm_topk_prob`。**不能这么做**——`route_norm` 已经由
`deepseek_scale_weights()` 里那句「sigmoid 时先除以 top-k 之和」实现了。
再开 `norm_topk_prob` 会在乘完 scaling 之后**二次归一化**，把 2.826 除掉。
config 里显式写了 `norm_topk_prob: False` 并注明原因，防止后人"好心"补上。

补齐的三项：`max_position_embeddings: 262144`（MaxText 默认 163840）、
`rope_type: "default"`、`attention_bias: False`。

**config 层面 22 项逐项比对，0 不一致**（层数 / 维度 / 头数 / GQA / head_dim /
vocab / eps / 专家数 / top-k / moe 维度 / shared / dense 首层 / scaling /
MTP / max_pos / sigmoid / expert bias / qk_norm / untied / rope theta / rope type）。

**但仍不是 100% 一模一样**，剩下三项无法靠配置消除：

"配置消不掉"的意思是：**MaxText 根本没有暴露对应的配置项，改 yml 解决不了，
只能改 MaxText 源码或者接受差异。** 三项性质完全不同：

HF 要求所有权重按 std=0.006 的截断正态初始化。MaxText 把初始化写死成
`nd_dense_init(1.0, "fan_in", "truncated_normal")` —— 这是 **fan-in 缩放**
（std 随输入维度自动变），不是固定 0.006，且 yml 里没有任何开关能改。

**影响面**：只在 from-scratch 时决定初始权重分布，进而影响收敛曲线。
加载预训练权重则完全无关。跑吞吐基线不受影响。

这一项最初写的是"找不到对应参数，若确实没有则……"。**已查实，确实没有。**

```
grep -rn "bias_update_rate|update_expert_bias|expert_load" src/MaxText/   -> 0 匹配
routed_bias 的唯一去处: moe.py:316  use_bias=self.config.routed_bias
```

`routed_bias: True` 在 MaxText 里只是给 gate 的 `Dense` 加了一个
**普通可学习 bias，跟着 loss 梯度走**。而 DeepSeek V3 的 aux-loss-free 均衡是
另一回事：bias **不参与梯度**，每步统计各专家实际负载，超载的减 γ、欠载的加 γ
（论文 γ=1e-3）。MaxText 里**没有任何负载统计代码**。

所以：**开着 `routed_bias` 并不等于开了 aux-loss-free 负载均衡。**
Hy3 官方靠这套机制做均衡，MaxText 上这套机制目前是缺的。

**影响面**：这是三项里唯一会影响训练正确性的。而且——

> **修正一句我先前说得太满的话**：我之前写"只影响训练质量，不影响性能基线"。
> 这不够严谨。专家负载失衡会让 all-to-all 出现热点，少数卡排队、其余空等，
> **吞吐会被拖累**（GB300 文档 §1.3 和 §八 坑 #8 都点了"热门专家扎堆"这件事）。
> 只是短期 synthetic + 随机初始化的路由接近均匀，几十步内看不出来。
> **拿短期基线是安全的；长时间真实训练不安全。**

官方 `HYV3MoE.forward()`：该开关为 `false` 时走
`hidden = routed_output + self.shared_experts(hidden_states)`，直接相加不转 fp32。
MaxText `RoutedAndSharedMoE.__call__()` 就是 `routed_experts + shared_experts`。
**行为相同**，此项结案。

---


### 2.7 对照官方 modeling 代码的审计

上一轮只比了 config 数值。这一轮拉了 **transformers 主干里的官方实现**
（`models/hy_v3/modeling_hy_v3.py` 608 行 + `modular_hy_v3.py` 322 行，
Hy3 已进主干，不是 remote code）逐个组件对照算法。

`modular_hy_v3.py` 的继承链说明 Hy3 是从多个模型拼出来的：

```python
class HYV3Attention(ApertusAttention)              # 不是 Qwen3Attention
class HYV3TopKRouter(MixtralTopKRouter)
class HYV3Experts(Qwen3MoeExperts)
class HYV3MoE(MiniMaxM2SparseMoeBlock)
class HYV3DecoderLayer(DeepseekV3DecoderLayer)
```

> `HYV3Attention` 继承自 `ApertusAttention` 一度让我以为选错了组件。
> 读展开后的实现，它的数学是：GQA + `q_norm`/`k_norm`（RMSNorm on `head_dim`，
> 用 `rms_norm_eps`）**在 RoPE 之前**、`scaling = head_dim**-0.5`、四个 proj 全无 bias。
> **这与 Qwen3 的 attention 逐行等价**，继承谁只是 transformers 内部的代码复用选择。
> 用 `qwen3.self_attention_with_norm` 是对的。

| 组件 | 官方实现 | MaxText 侧 | 状态 |
|---|---|---|---|
| Attention | GQA + qk RMSNorm(head_dim) 在 RoPE 前 + scaling head_dim^-0.5 + 无 bias | `qwen3.self_attention_with_norm` | ✅ 等价 |
| 层布局 | `mlp_layer_types = ["dense"]*1 + ["sparse"]*79` | `first_num_dense_layers: 1` | ✅ 等价 |
| shared expert | `routed + shared`，无额外 gating | `RoutedAndSharedMoE` 同式 | ✅ 等价 |
| 路由数学 | sigmoid + 专家偏置，打分取偏置前的值（§1.5） | `deepseek_routing()`，需把 `hunyuan3` 加进名字门（§2.6） | ✅ **本轮修好** |
| router 精度 | **fp32 强制** | 跟 `cfg.dtype`（bf16） | ❌ **差异** |
| expert bias 更新 | buffer + 训练框架更新 | 可学习 Parameter | ❌ **差异** |
| 初始化 | std=0.006 | fan-in 缩放 | ❌ 差异 |


### 2.8 三件"生产级还差的事"，后来两件被上游补上了

> **先看结论，下面的长篇论证是当时（MaxText `3eb77db3`）的状况，已经过时。**
>
> | 缺口 | `3eb77db3`（v5p 用的旧版） | 新版（v7 用的 MaxText main） |
> |---|---|---|
> | ① 专家 bias 的无梯度更新规则 | ❌ 没有（下面五组关键词全仓搜过） | ✅ **已有 `routed_bias_update_rate`**，但**每个模型的 layer 必须自己把第三个返回值 `sow` 出去**，否则静默失效（见 §八 bug #10） |
> | ② router 强制 fp32 | ❌ 没有，我加了 `moe_router_dtype` | ✅ **已有 `float32_gate_logits`** |
> | ③ `initializer_range: 0.006` | ❌ 没暴露 | ❌ 仍缺（只影响 from-scratch） |
>
> 也就是说：**如果你用的是新版 MaxText，这一节只剩第 ③ 项。**
> 保留原文是因为里面那套"路由算法 / 权重加载 / 更新规则"的三层拆分
> 对理解 MoE 的 bias 机制仍然有用，而且第 ① 项的 **SFT 隐患（bias 会被梯度扰动，
> 正确做法是冻结）在两个版本上都还在**。

<details>
<summary>展开：当时的完整论证（含全仓搜索证据）</summary>

按优先级：

**① expert bias：要分清"路由算法"和"bias 更新规则"是两回事**

先把容易混淆的三层拆开——MaxText 对 DSV3 的支持，**前两层是完整的**：

| 层次 | MaxText 状况 |
|---|---|
| **路由算法**（sigmoid / bias 选择 / pre-bias 取值 / 归一化 / scaling） | ✅ **完全正确**，`deepseek_routing()` 与 Hy3 官方逐步一致 |
| **权重加载** | ✅ **完整**，`convert_deepseek_family_ckpt.py:161` 把 HF 的 `e_score_correction_bias` 映射到 `gate.bias` |
| **训练时的 bias 更新规则** | ❌ **没有**（下详） |

第三层的搜索证据（五组关键词，全仓 `src/MaxText/`）：

```
(expert|router|gate|routed).*bias.*(update|adjust|rate)   -> 0
e_score / correction_bias / aux-loss-free / violation      -> 仅 ckpt 转换脚本的映射表
expert_count / tokens_per_expert / expert_density          -> 0
jnp.sign / lax.sign                                        -> 0
base.yml 的 routed_*/moe_router_*                          -> 仅 4 项，无 update rate
```

MaxText 把 bias 建成 `nnx.Param`（跟主 loss 梯度走），DSV3 则是 `register_buffer`
（不参与梯度，每步按负载 ±γ）。

| 场景 | 需要什么 | 难度 |
|---|---|---|
| **加载官方权重做 SFT / continued-pretrain** | **不需要更新机制**——bias 已训好。反而需要**冻结**它 | 低 |
| **from-scratch 预训练** | 需要完整的负载驱动更新 | 高 |

GB300 文档 §1.3 已经写明这一点：「微调场景例外：加载官方权重做 SFT 时专家路由已训好，
`bias_update_rate` 保持 0 更稳，避免扰动已收敛的路由。**只有 from-scratch /
continued-pretrain 才需要开**。」

> **走 SFT 路线时反而有个反向隐患**：MaxText 的 bias 是可训练 `Param`，
> SFT 时**会被梯度更新**，而正确做法是保持不动。
> 本版 MaxText（`3eb77db3`）**没有 `trainable_parameters_mask`**（已查，不存在），
> 所以冻结也要自己加——但比实现负载更新简单一个量级。

**修正**：先前把这一项列为"阻塞真实预训练的最高优先级"。
按 SFT / 加载权重的路线，真正要做的是**冻结 bias**，不是实现更新规则。
下面的实现方案只在 from-scratch 场景才需要。

**from-scratch 才需要的负载驱动更新**

官方把 `e_score_correction_bias` 注册为 **buffer**（`register_buffer`，不参与梯度），
HF 的 modeling 里也没有更新它的代码——**那是训练框架的职责**。
Megatron 用 `moe_router_bias_update_rate=1e-3` 实现；MaxText 里
`routed_bias` 是普通可学习 `Parameter`，跟着 loss 梯度走，语义不同。

要做：把 bias 移出梯度路径 + 每步统计各专家 token 数 + 按
`bias -= γ·sign(load - mean_load)` 更新（DSV3 论文式）。

**② router 强制 fp32**（影响数值稳定性）

官方 `F.linear(hidden_states.float(), self.weight.float())` 显式转 fp32。
GB300 侧 Megatron 也设了 `moe_router_dtype: fp32`。
MaxText 的 `GateLogit` 用 `cfg.dtype`（训练时是 bf16），**没有 router dtype 开关**。
192 个专家的 sigmoid 打分在 bf16 下精度不足，会影响 top-k 选择的稳定性。

要做：给 `GateLogit` 加 dtype 覆盖，或在 config 里新增 `moe_router_dtype`。

**③ `initializer_range: 0.006`**（只影响 from-scratch）

MaxText 把初始化硬编码为 fan-in 缩放，没暴露 std。加载预训练权重则无关。

---

</details>

### 2.9 静态自检跑出来是什么样

> 下面这个 `verify_hunyuan3.py` 是当时的一次性校验脚本，**没有随仓发布**
> —— 它的职责现在由 `prep.sh` 的 8 项自检承担（§9.2）。
> 保留这段输出是因为它记录了「静态检查能验到什么、验不到什么」，
> 这个边界见 §八。

```
$ python3 verify_hunyuan3.py --root /path/to/maxtext
1) total params  294.97 B   (SSOT 294.9 B, delta 0.02%)
   activated     20.6 B    (official A21B)
   experts share 97.6%  -> EP is the only memory knob that matters
2) enum          DecoderBlockType.HUNYUAN3
3) layer classes ['Hunyuan3DenseLayer', 'Hunyuan3MoELayer']
4) dispatch      ['Hunyuan3DenseLayer', 'Hunyuan3MoELayer']
5) norm layer    rms_norm
ALL CHECKS PASSED
```

**参数量自检是最有力的一项**：294.97 B 对 SSOT 的 294.9 B 只差 0.02%，
且五个组成部分的占比逐项吻合。config 写错任何一个维度都会立刻暴露。

**但请注意这仍是静态验证。** 跑的时候 stub 掉了 `grain` / `tensorflow` /
`qwix`（数据管线与量化，均不参与构图），**没有做真实前向、没有做多卡分片、
没有做权重转换**。这些在待验证清单里。


## 三、v5p：从 4 芯片跑通到 256 芯片


### 3.1 环境与迭代方法

| 项 | 值 |
|---|---|
| 节点池 | `np-v5p-hy3-dev`，1 台 `ct5p-hightpu-4t`，拓扑 `2x2x1`，spot |
| device | 4（v5p 是 MegaCore，4 chips = 4 devices） |
| 镜像 | `maxtext-stable:oct`（MaxText `3eb77db3` + JAX 0.7.0） |
| 代码注入 | 本地改动 tar 后 `kubectl cp` 进 pod，解到 `/deps` 覆盖 |

**为什么先建 1 节点小池**：64 节点跑一次要等 6–7 分钟编译，改一行代码就得重来。
4 chips 上跑小模型只要几十秒，修 bug 的迭代速度差一个量级。

冒烟用的 `hunyuan3-smoke.yml` **结构与 295B 完全一致**（dense 首层 + MoE 层、
sigmoid + bias、shared expert、fp32 router），只把维度缩小——
目的是走遍每条代码路径，不是测性能。

当时套了一个一次性的迭代脚本：tar 本地改动 → `kubectl cp` 进 pod → 解包覆盖 →
跑 6 步 → grep `completed step` 判定，失败就打印首个错误。单轮约 40 秒，
这是能连续迭代二十轮的前提。

> **那个脚本没有进仓库，也不该进。** 它按文件名逐个注入改动，
> 正是 §9.2 后来明确否定的做法 —— 只注入部分文件，测的是
> 「我的改动 + 容器里的旧基座」，不是完整的那一份。
> 现在这件事由 `prep.sh`（打包整棵 `src/maxtext`）+ `run.sh`（pod 里整棵覆盖）承担，见 §九。
> 这里保留这段，只为说明**为什么要先建 1 节点小池**。


### 3.2 前三个 bug：K8s 调度 / 框架注册 / MXU 对齐

```
completed step: 3, seconds: 0.006, TFLOP/s/device: 21.951, loss: 10.602
completed step: 4, seconds: 0.006, TFLOP/s/device: 22.534, loss: 10.551
completed step: 5, seconds: 0.007, TFLOP/s/device: 20.018, loss: 10.522
```

**loss 单调下降**，前向和反向都通。运行时确认所有关键配置生效：

```
decoder_block    : DecoderBlockType.HUNYUAN3
routed_score_func: sigmoid | routed_bias: True
routed_scaling   : 2.826
shared_experts   : 1 | first_num_dense_layers: 1
moe_router_dtype : float32
norm_topk_prob   : False
```

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | pod 被 admission webhook 拒：`tpu-accelerator-topology-constraints cannot be bypassed` | 只写了 `gke-nodepool` selector。TPU pod **必须**同时带 `gke-tpu-accelerator` 和 `gke-tpu-topology` | nodeSelector 补两个标签 |
| 2 | `ValueError: Invalid model name was passed. Got hunyuan3-smoke` | 加了 model yml 还不够，`pyconfig.validate_model_name()` 里有一份**硬编码白名单** | 白名单加 `hunyuan3-295b` / `hunyuan3-smoke` |
| 3 | `Pallas TPU lowering requires the last two dimensions be divisible by 8 and 128`，block shape `(512, 192)` | smoke config 里 `base_moe_mlp_dim: 192`（1536÷8 随手算的）——**不是 128 的倍数** | 改 256 |

**第 3 个最值得记**：MXU 是 128×128 的脉动阵列，最后一维必须 128 对齐，
这是 TPU 上的基本约束。前面所有静态检查——参数量、enum、分派、路由分支——
**一个都查不出来**，因为它们只看配置和代码结构，不看维度能否落到硬件 tile 上。
只有真跑才会撞到。

> 三个 bug 分属三个层次：**K8s 调度**（1）、**框架注册**（2）、**硬件约束**（3）。
> 静态验证覆盖不到任何一层——它验的是"逻辑对不对"，
> 这三个问题都是"环境让不让你跑"。


### 3.3 渐进放大扫描 r4–r10

从跑通的最小配置出发，**每轮只动一个维度**，失败也继续下一轮：

| 轮次 | 变化 | 层 | 专家 | emb | moe_mlp | 结果 | step | TFLOP/s/dev | main_loss | mtp_loss |
|---|---|---|---|---|---|---|---|---|---|---|
| r4 | 开 MTP=1 | 4 | 8 | 512 | 256 | PASS | 0.012s | 11.33 | 10.419 | 1.070 |
| r5 | 层数 4→8 | 8 | 8 | 512 | 256 | PASS | 0.016s | 13.94 | 10.138 | 1.070 |
| r6 | 专家 8→32 | 8 | 32 | 512 | 256 | PASS | 0.021s | 10.47 | 10.206 | 1.071 |
| r7 | 维度翻倍 | 8 | 32 | 1024 | 512 | PASS | 0.040s | 10.86 | 9.549 | 1.045 |
| r8 | 同上 + EP=4 | 8 | 32 | 1024 | 512 | PASS | 0.031s | 13.92 | 9.549 | 1.045 |
| r9 | 接近真实维度 + EP4 | 8 | 64 | 2048 | 1024 | PASS | 0.071s | 11.94 | 8.054 | 0.984 |
| r10 | **真实宽度** 8 层 + EP4 | 8 | 192 | 4096 | 1536 | **OOM** | — | — | — | — |

三个可以直接读出来的结论：

**1. MTP 真的在跑。** 从 r4 起每一轮都单独打出 `mtp_loss`：

```
completed step: 7, seconds: 0.012, TFLOP/s/device: 11.330,
  loss: 11.489, main_model_loss: 10.419, mtp_loss: 1.070
```

`loss = main_model_loss + 0.1 × mtp_loss` 对得上（10.419 + 0.1×1.070 = 10.526，
剩下的差是 moe load-balance 项）。MTP 头不是挂着不动的死代码。

**2. r7 → r8 是一次干净的对照实验。** 两轮配置完全相同，唯一差别是
`ici_expert_parallelism` 从 1 改成 4（即从默认 FSDP 切成专家并行）：

| | step | TFLOP/s/dev | main_loss |
|---|---|---|---|
| r7 (FSDP=4) | 0.040s | 10.86 | 9.5490 |
| r8 (EP=4) | 0.031s | 13.92 | 9.5490 |

**吞吐 +28%，loss 一位不差。** loss 相同是重点——它证明 EP 只改变了权重
怎么摆在设备上，没有改变数学。这正是切分策略该有的性质，也顺带说明
前面那个 `deepseek_scale_weights` 分支修对了：如果路由数学被并行方式
影响，两轮的 loss 不可能逐位相同。

**3. r10 的 OOM 是容量，不是 bug。**

```
Ran out of memory in memory space hbm.
Used 120.49G of 95.74G hbm. Exceeded hbm capacity by 24.75G.
```

真实宽度下 7 个 MoE 层的专家权重 = 7 × 192 × 3 × 4096 × 1536 ≈ **25.4 B**，
EP=4 后每卡 6.3 B，Adam（fp32 参数 + m + v = 12 B/param）就要 76 GB，
加上激活和 dense 部分越过 95.74 GB。4 芯片本来就装不下真实宽度的 8 层——
这是**在预期之内的物理上限**，说明代码路径是通的，只是卡不够。


### 3.4 逐项换成真实值 r11–r16

r10 说明 4 芯片装不下真实宽度的 8 层。于是换个方向：**固定真实宽度、砍层数**，
然后把配置项一个一个换成 295B 的真实值，看每条代码路径在真实维度下能不能跑。

| 轮次 | 换成真实值的项 | 层 | 结果 | step | TFLOP/s/dev | main_loss | HBM/dev |
|---|---|---|---|---|---|---|---|
| r11 | emb 4096 / moe_mlp 1536 / 192 专家 | 4 | PASS | 0.163s | 6.43 | 7.431 | 61.2 G |
| r12 | 同上，加到 6 层 | 6 | PASS | 0.243s | 5.66 | 6.197 | **91.9 G** |
| r13 | + 64 query / 8 KV 头、top-8、dense 13312 | 4 | PASS | 0.223s | 78.27 | 4.556 | 64.6 G |
| r14 | + vocab 120832、位置窗口 262144 | 4 | PASS | 0.246s | 75.69 | 5.885 | 66.8 G |
| r15 | + `attention=flash` | 4 | PASS | 0.248s | 74.87 | 5.884 | 67.0 G |
| r16 | + 序列长度 512 → 2048 | 4 | PASS | 0.514s | **145.96** | 8.996 | 73.6 G |

跑完 r15 之后，`hunyuan3-smoke.yml` 和 `hunyuan3-295b.yml` 之间
**只剩一个字段不同**：

```
DIFF  295b[base_num_decoder_layers: 80]  smoke[base_num_decoder_layers: 4]
same  base_emb_dim: 4096          same  num_experts: 192
same  base_mlp_dim: 13312         same  num_experts_per_tok: 8
same  base_num_query_heads: 64    same  base_moe_mlp_dim: 1536
same  base_num_kv_heads: 8        same  shared_experts: 1
same  head_dim: 128               same  first_num_dense_layers: 1
same  vocab_size: 120832          same  routed_scaling_factor: 2.826
same  max_position_embeddings: 262144   same  rope_max_timescale: 11158840
same  mtp_num_layers: 1           same  moe_router_dtype: "float32"
```

也就是说，**除了深度，每一个架构维度都已经在真实硬件上跑过前向和反向**。

几个值得单独说的观察：

**r12 的 91.9 G 是 4 芯片的天花板。** 可用 95.74 G，6 层已经贴到边，
第 7 层必 OOM。这跟 r10 的结论一致，只是从另一头逼近。

**r13 的 TFLOP/s 从 5.66 跳到 78.27（13.8×）。** 这一轮同时换了三样：
query 头 8 → 64、top-k 2 → 8、dense 中间层 1664 → 13312。三样都是直接乘在
FLOP 上的，跳这么多是算术，不是优化。**注意这里 HBM 反而降了**
（91.9 → 64.6 G），因为层数从 6 退回 4——不要把这两个数放在一起读。

**r15 换 flash attention，loss 从 5.885 变成 5.884。** 差在第四位小数，
是累加顺序不同造成的浮点差异，不是数学变了。flash 在 v5p 上没有
[DSV3 文档踩坑 #3](../DeepSeek-V3.2-Training/README.md) 里 v7 的那个编译问题。

**r16 是唯一一个真正说明性能的数字。** 序列 512 → 2048，
TFLOP/s/device 从 74.87 涨到 145.96（+95%）——序列变长把固定开销摊薄了。
145.96 / 459 = **31.8% MFU**，跟 GB300 上 31.6% 处在同一水平。
但这只是 4 层小模型 4 芯片，**不能当基线**，真实基线见 §七。

---


### 3.5 r17–r20：长跑稳定性与 dropping 路径

| 轮次 | 内容 | 结果 | step | TFLOP/s/dev | main_loss |
|---|---|---|---|---|---|
| r17 | 真实配置 6 层 | PASS | 0.366s | 16.70 | 4.454 |
| r18 | 4 层跑 60 步 | PASS | 0.249s | 18.40 | **0.000** |
| r19 | 打开 xplane profiler | PASS | 0.249s | 18.40 | 2.319 |
| r20 | 换 dropping 路径（`capacity_factor=1.0`） | PASS | **0.157s** | **29.18** | 5.876 |

**r18 的 loss 收敛到 0 不是 bug。** synthetic 数据集反复喂同一批样本，
4 层小模型 60 步就把它背下来了。这一轮要看的不是 loss 值，
而是**跑 60 步没有 NaN、没有发散、step 时间没有漂移**——长跑稳定性过关。

**r20 值得单独记：dropping 比 dropless 快 37%**（0.157 vs 0.249 s）。
`megablox=True sparse_matmul=True` 是无丢弃路由（每个专家收多少算多少），
`capacity_factor=1.0` 是固定容量、超了就丢。后者的 GEMM 形状是静态的，
XLA 能编出更好的 kernel。**代价是丢 token 会影响收敛**，
预训练要不要用得单独评估——但至少在 TPU 上它不是"落后选项"。

---


### 3.6 必须先说清的命名坑：TensorCore / chip / device

本文里出现的 `np-v5p-256` 是**节点池名字**，不是 Google 的加速器类型名：

| | 芯片数 | JAX device 数 | 节点数 | 拓扑 |
|---|---|---|---|---|
| 节点池 `np-v5p-256` | **256** | 256 | 64 × `ct5p-hightpu-4t` | 4x8x8 |
| Google 命名法里的 `v5p-256` | **128** | 128 | 32 | — |

Google 的 `v5p-N` 里的 N 数的是 **TensorCore**，v5p 每芯片 2 个 TensorCore，
所以 `v5p-N` = N/2 芯片。**我们这个池按 Google 命名法应该叫 `v5p-512`。**

同时 v5p 是 MegaCore：两个 TensorCore 对 XLA 呈现为**一个** device，
所以 256 芯片 = 256 device（不是 512）。

> 三个数字（TensorCore / chip / JAX device）在 v5p 上的比例是 **2 : 1 : 1**，
> 在 v7 上是 **2 : 1 : 2**。跨代际对比时这是最容易算错的一步。
> 下文 §四、§七的表格已按**实际 256 芯片**修正。

---


### 3.7 80 层完整 295B 在 256 芯片上跑通

前面都是 4 芯片的小规模验证。这一节把真正的 80 层、192 专家、
298.8 B 参数放到 `np-v5p-256`（4x8x8，256 芯片）上。

#### 怎么把补丁发到 64 台机器

单节点靠 `kubectl cp` 就行，64 个 pod 不能这么干。改成 **GCS 中转**：

当时的做法是把改过的那几个文件打包传 GCS，pod 启动时各自拉下来解到 `/deps`。
比重新 build 镜像快得多（补丁 55 KB，改一行到重跑只要几十秒），
也不用把私有改动烤进镜像。

> ⚠️ **这个打法已经废弃，不要照做。** 只注入改动文件，测的是
> 「我的改动 + 容器里的旧基座」；§4.8 的问题 2/3 和 §八 的 bug #11/#12
> 全都是它埋的。现在由 `prep.sh` 打包**整棵 `src/maxtext`**、
> `run.sh` 在 pod 里整棵覆盖，见 §9.2。GCS 中转这个思路保留了下来，
> 变的是「传整棵树」而不是「传几个文件」。JobSet 用 `parallelism: 64 / completions: 64`
配 `exclusive-topology: gke-nodepool` 注解，拿整个 4x8x8 切片。

```
number parameters: 298.786 billion
Per train step: Total TFLOPs: 2800.97
  split as 97.64% learnable weight flops and 2.36% attention flops

completed step: 0, seconds: 74.031, TFLOP/s/device: 37.835,  loss: 13.424
completed step: 1, seconds:  0.323, TFLOP/s/device: 8671.953, loss: 13.424
completed step: 2, seconds: 106.792, TFLOP/s/device: 26.228, loss: 13.008
completed step: 3, seconds: 49.962, TFLOP/s/device: 56.062, main_model_loss: 11.550
completed step: 4, seconds: 49.994, TFLOP/s/device: 56.026, main_model_loss: 11.337
```

**80 层、192 专家、298.8 B 参数，全局 batch 1,048,576 token，loss 单调下降。**
稳态 **50.0 s/step**。

**坑一：step 1 的 8671 TFLOP/s 是假的。** v5p 峰值 459 TFLOPS/chip，
任何超过它的数字都不可能是真的。JAX 是异步派发的，step 1 的计时器量到的是
**入队时间**不是执行时间；被推迟的工作在 step 2 一起结算（106.8 s）。
**稳态只能取 step ≥ 3**，而且要取中位数——后面所有扫描都按这个口径。

**坑二：298.786 B ≠ SSOT 的 294.9 B。** 差 3.886 B，来自 **MTP 头**——
`mtp_num_layers: 1` 会实打实多出一整层（含 193 个专家）。
HF 的 295B 不含这一层，因为推理时它被丢掉。**不是参数量算错了**，
是训练态和发布态本来就不是同一个数。

**坑三：MegaCore 没切开。** 日志里反复出现：

```
TPU target is configured in megacore mode, but Mosaic failed to
partition the kernel across cores. Running on one core only.
```

v5p 每芯片两个 TensorCore 由 XLA 当一个 device 用，前提是 kernel 能沿某一维
对半切。切不开就只跑一个核——**等于白扔一半算力**。这是 v5p 上 MoE kernel
的常见情况，也是后面性能优化要重点看的一项。


### 3.8 bug #4：`fsdp_shard_on_exp` 与 EP 互斥

```
ValueError: fsdp_shard_on_exp requires ici_expert_parallelism = 1 and
            ici_tensor_parallelism/ici_tensor_transpose_parallelism = 1
```

这个开关是给**不用 EP** 的场景准备的（把专家维切到 FSDP 轴上）。
既然已经开了 EP=64，它就是多余的。顺带记一笔：新版 MaxText 里这个参数
叫 `shard_exp_on_fsdp`，本仓这版叫 `fsdp_shard_on_exp`，**名字是反的**。


### 3.9 bug #5：MFU 口径被虚高了约 5 倍

这个最隐蔽，因为它**不影响训练，只影响你以为自己跑得多快**。

`maxtext_utils.py` 里算解析 FLOP 的地方，又是一个按模型名列举的分支：

```python
if config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.LLAMA4):
    total_ffn_flops = calculate_routed_and_shared_ffn_tflops_per_device(config)
else:
    gate_flops = 2 * B * L * emb * num_experts
    total_ffn_flops = gate_flops + ffn_matmul(config, config.mlp_dim) * num_experts_per_tok
```

`hunyuan3` 不在名单里 → 走 `else`，于是专家 FFN 用 **`mlp_dim` = 13312**
（那是 dense 首层的宽度）而不是 **`moe_mlp_dim` = 1536**，
而且**完全不算 shared expert**，也不区分首层 dense。
13312 / 1536 = **8.7 倍**的单项偏差，摊到整模型后报出来的 TFLOP/s 虚高约 5 倍。

一共三处要改，都在 `maxtext_utils.py`：

| 位置 | 原本 | 问题 |
|---|---|---|
| MoE FFN 分支 | 只列 DEEPSEEK / LLAMA4 | 专家宽度用错，漏 shared expert |
| `get_dense_moe_layers()` | 只认 DEEPSEEK / LLAMA4，其余 `raise ValueError` | 拆不出 dense/MoE 层数 |
| 逐层汇总 | DEEPSEEK 单独一支 | helper 已按层累加，走 `else` 会再乘一次层数 |

> 这是本项目撞到的**第三个同类 bug**（前两个是 §2.6 的路由分支和
> §1.5 的 `model_name.startswith` 门）。模式完全一样：**MaxText 里凡是按模型家族
> 名字列举的分支，新模型都得逐个补进去；漏了不报错，只是安静地跑出
> 另一套语义。** 前两个改的是训练数学，这个改的只是报表——
> 但报表错了会让人拿着虚高 5 倍的 MFU 去做容量规划。

修完之后 §七 的 MFU 才有意义。**§3.1–§3.7 里所有 TFLOP/s 数字
都是修复前的口径，不要拿去对标 GB300**；对标数字见 §七。

验证：同一份代码、同一个 pod，只改这三处，`Total TFLOPs` 从
**2800.97 → 561.92**（4.985 倍），561.92 / 4096 token = **137.2 GFLOP/token**，
跟 GB300 侧的 136.8 GFLOP/token 对上了。step 时间和 loss 曲线**一字未变**——
确认改的只是报表。


## 四、v5p 性能调优：MFU 2.45% → 36.72%

**全部数字都是 FLOP 口径修正后的**（见 §3.9）。
稳态取 step ≥ 3 的中位数。MFU 分母 **459** TFLOPS/chip；
v5p 是 MegaCore，**1 device = 1 chip，日志值不用乘 2**。

| # | 配置 | step | TFLOP/s/dev | **MFU** | tok/s/dev | 整机 tok/s | HBM/dev |
|---|---|---|---|---|---|---|---|
| o1 | 自己攒的：EP64/FSDP4, pdbs=1, seq=4096, 3 个 XLA flag | 49.99 s | 11.24 | **2.45%** | 81.9 | 20,974 | 54.8 G |
| **o2** | **照抄官方 DSV3 recipe**：FSDP=256/无 EP, pdbs=4, seq=8192, 26 个 XLA flag | **34.67 s** | **144.87** | **31.56%** | **945.1** | **241,935** | — |

**o1 → o2 是 12.9 倍。** 这一节剩下的篇幅都在解释这 12.9 倍是怎么来的。


### 4.1 我犯的错：没有从官方配方出发

我先按自己对模型的理解攒了一套配置（o1），理由是
"97% 参数在专家里，EP 是主旋钮"。这个推理在 GPU 上成立，
在 TPU 上**结论是反的**。

同一份 tpu-recipes 文档里写着一句话，我当时读过：

> **从官方配方出发，只改被规模逼着改的那一个参数，不要顺手删减。**

我没照做。o2 就是把 [官方 DeepSeek3-671B 256chips 配方](https://github.com/yangwhale/tpu-recipes/tree/main/training/v5p/DeepSeek3-671B-MaxText-256chips)
原样搬过来，只换 `model_name` 和 tokenizer。

两套配置的差异：

| 项 | o1（我攒的） | o2（官方） | 差异含义 |
|---|---|---|---|
| `ici_fsdp_parallelism` | 4 | **-1（=256）** | 非专家权重从 4 路切成 256 路 |
| `ici_expert_parallelism` | 64 | **1（不用 EP）** | 靠 FSDP all-gather 而非 all-to-all |
| `per_device_batch_size` | 1 | **4** | |
| `max_target_length` | 4096 | **8192** | 合计每卡 token 数 **8 倍** |
| `use_custom_sort_vjp` | False | **True** | megablox 分组排序的自定义反向 |
| `sa_block_*` | 512（默认） | **2048** | flash attention 分块 |
| `tile_batch_seq/embed/mlp` | 未设 | **512/1024/1024** | MoE GEMM 的 tile 尺寸 |
| `out_proj` | 未设 | **offload** | 多卸一个张量到 host |
| XLA flags | 3 个 | **26 个** | 主要是 SparseCore 集合通信卸载 |


### 4.2 为什么 TPU 上 FSDP 打得过 EP

这是本轮最反直觉的一条，值得单独说。

- **GPU（GB300）**：专家并行靠 DeepEP 做 all-to-all，NVLink 域内带宽极高，
  EP 把专家权重摊开、只搬 token，是省显存又省带宽的打法。
- **TPU（v5p）**：ICI 是 3D torus，**64 路 all-to-all 要跨整个环面**，
  没有 NVLink 那种全连接域。而 FSDP 的 all-gather 是规则的近邻通信，
  还能整个卸载到 **SparseCore** 上跟 TensorCore 的矩阵乘重叠。

换句话说，**EP 在 TPU 上把通信换成了拓扑最不友好的那种形状**。
o2 里 `ici_expert_parallelism` 干脆是 1——192 个专家全靠
FSDP 沿 `embed` 维切开，all-gather 走 SparseCore。

> "EP 是主旋钮"那句话**对 GPU 成立，对 TPU 不成立**，
> 已在下方标注。这不是笔误，是把一代硬件的经验直接搬到另一代的典型翻车。


#### 原始设计里的推理长什么样

| 参数 | v7 4x4x4 | v5p-256 | 理由 |
|---|---|---|---|
| `ici_fsdp_parallelism` | 待定 | 待定 | 与 EP 组合，见下 |
| `ici_expert_parallelism` | 待定 | 待定 | 97% 参数在专家里，EP 是主旋钮 |
| `ici_tensor_parallelism` | **1** | **1** | attention 仅 2%，TP 纯亏 |
| `per_device_batch_size` | 1（首跑） | 1（首跑） | 跑通后上探 |
| `max_target_length` | 4096 | 4096 | 对齐 GB300 benchmark 口径 |
| `dtype` / `weight_dtype` | bfloat16 | bfloat16 | 对齐官方精度 |
| `megablox` / `sparse_matmul` | True / True | True / True | Dense MoE 必 OOM，见 DSV3 文档踩坑 #4 |
| `attention` | 待定 | 待定 | v7 上 flash 有编译问题，见 [DSV3 文档踩坑 #3](../DeepSeek-V3.2-Training/README.md) |
| `dataset_type` | synthetic | synthetic | 首轮只测吞吐 |

> ⚠️ **下面这段推理已被实测推翻，保留原文以便对照。**
>
> ~~EP / FSDP 的具体配比是本轮要扫的第一个维度。GB300 上 EP=32 是甜点
> （192 experts / 32 = 6 专家/rank），TPU 侧 device 数是 128，
> 候选 EP ∈ {8, 16, 32, 64}，需实测。~~
>
> 实测结论：**TPU 上最优是完全不用 EP**（`ici_expert_parallelism=1`、
> `ici_fsdp_parallelism=256`），MFU 从 2.45% 提到 31.56%。
> "97% 参数在专家里所以 EP 是主旋钮"这条推理**只对 GPU 成立**——
> TPU 的 3D torus 上 64 路 all-to-all 是最不友好的通信形状，
> 而 FSDP 的 all-gather 能整体卸载到 SparseCore。详见本节。

---


### 4.3 192 不是 2 的幂，这在 TPU 上是有代价的

`pyconfig.py:1073` 有一条硬约束（`sparse_matmul=True` 时）：

```python
if raw_keys["num_experts"] % expert_parallelism:
    raise ValueError(f"The expert dimension {num_experts} is not divisible by "
                     f"expert parallelism setting {expert_parallelism}")
```

192 = 2⁶ × 3。在 256 个 device 上，EP 必须同时整除 256 和 192 →
最大只能取 **64**，凑不出 128 或 256。DeepSeek V3 的 **256** 个专家
在 256 卡上是 1:1 完美对齐，Hy3 的 192 做不到。

同理 `fsdp_shard_on_exp=True`（把专家维切到 FSDP 轴上）要求
`num_experts % ici_fsdp_parallelism == 0`，192 % 256 ≠ 0，**这条路对 Hy3 直接封死**。

> 这是模型架构选择在硬件上留下的真实痕迹：
> **专家数取 2 的幂在 TPU 上不是审美问题，是能不能对齐分片的问题。**
> 好在 o2 证明了不用 EP 也能跑满，这个限制没有变成瓶颈。


### 4.4 消融：这 12.9 倍具体是谁贡献的

以 o2 为基准，每轮只动一项：

| # | 相对 o2 的改动 | step | TFLOP/s/dev | MFU | 整机 tok/s | HBM/dev | Δ MFU |
|---|---|---|---|---|---|---|---|
| **o12** | **pdbs 再上探到 8**（当前最优） | 59.60 s | **168.56** | **36.72%** | **281,488** | 96.9 G | **+5.16 pp** |
| o11 | pdbs=6 且 `out_proj=remat`（两项最优叠加） | 46.89 s | 160.70 | 35.01% | 268,363 | 80.9 G | +3.45 pp |
| o8 | pdbs 4 → 6 | 47.67 s | 158.04 | 34.43% | 263,920 | 80.3 G | +2.87 pp |
| o9 | `out_proj` 不 offload | 34.43 s | 145.86 | 31.78% | 243,591 | 62.4 G | +0.22 pp |
| **o2** | **（基准，官方配方）** | 34.67 s | 144.87 | 31.56% | 241,935 | — | — |
| o4 | 去掉 `tile_*` 三项 | 35.01 s | 143.45 | 31.25% | 239,555 | 61.9 G | −0.31 pp |
| o5 | 去掉 `use_custom_sort_vjp` | 43.03 s | 116.64 | 25.41% | 194,784 | 61.6 G | **−6.15 pp** |
| o10 | 换 dropping（`capacity_factor=1.0`） | 46.16 s | 108.81 | 23.71% | 181,710 | 79.6 G | **−7.85 pp** |
| o7 | seq 8192 → 4096 | 21.90 s | 102.64 | 22.36% | 191,538 | 50.2 G | **−9.20 pp** |
| o3 | 26 个 XLA flag → 只留 2 个 | 39.81 s | 126.18 | 27.49% | 210,716 | 66.8 G | **−4.07 pp** |
| o6 | 改回 EP=64 / FSDP=4 | — | — | **OOM** | — | — | 超 326.63 G |

按贡献排序，五条结论（**26 个 XLA flag 值 4.07 pp，13%**——
几乎全是把 all-gather / reduce-scatter 卸载到 SparseCore 上跟矩阵乘并行）：

**1. 序列长度是最大的单项（+9.2 pp）。** 8192 → 4096 掉到 22.36%。
80 层 MoE 每层都有一次 all-gather + 一次分组 GEMM，序列越短，
这些固定开销摊得越薄。注意 o7 的 step 只要 21.9 s，**看起来更快**——
但每步只算一半 token，单位吞吐反而低。**只看 step 时间会得出相反的结论。**

**2. `use_custom_sort_vjp` 值 6.15 pp。** 这是 megablox 做分组矩阵乘时
"把 token 按专家排序"那一步的自定义反向传播。默认 `False`，
不开就走 JAX 通用求导，慢 19.5%。**一个布尔量，快五分之一。**

**3. batch 还能再上探。** pdbs 4 → 6 拿到 34.43%，HBM 涨到 80.3 / 95.74 G。
当时判断 pdbs=8 大概率 OOM，**后来实测跑通了**——就是同表的 o12，36.72%。

**4. `out_proj=offload` 是负收益（−0.22 pp）。** 官方 DSV3 配方里有这一项，
搬到 Hy3 上反而略亏。原因大概是 Hy3 的 attention 只占 2.36% FLOP，
省下来的那点显存不值得多一趟 PCIe 往返。
**这是"照抄官方"唯一需要改回来的地方。**

`tile_*` 三项只值 0.31 pp——在 256 卡上几乎测不出来，
跟 tpu-recipes 文档里 DSV3 的记录不一致，可能是模型形状不同。

**o11 把两项正收益叠起来（pdbs=6 + `out_proj=remat`），拿到 35.01%；
o12 把 batch 再推到 8，拿到 36.72%——这是 v5p 侧的最终成绩。**
两项单独是 +2.87 和 +0.22，叠加后 +3.45——**略大于两者之和**，
说明少一趟 out_proj 的 PCIe 往返，在 batch 更大时价值更高一点。
o12 的完整配置如下。
**⚠️ 这是旧栈（linen）的参数集，不能直接照搬到现在的代码**：
`tile_batch_seq` 三项在新栈已拆成 18 个 `wi_/wo_tile_*`（§5.5 坑 2），
照抄会被 pydantic 直接拒。**当前可跑的那一份在 `run.sh` 的 v5p 分支里**，
这里保留原样是为了对照 o1–o12 的消融记录。

```bash
model_name=hunyuan3-295b
ici_fsdp_parallelism=-1        # 256 路，不用 EP
ici_tensor_parallelism=1
per_device_batch_size=8
max_target_length=8192
megablox=True sparse_matmul=True scan_layers=True
use_custom_sort_vjp=True        # 值 6.15 pp，默认 False 一定要打开
sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048
sa_block_q_dq=2048 sa_block_kv_dq=2048 sa_use_fused_bwd_kernel=False
remat_policy=custom decoder_layer_input=offload
out_proj=remat                  # 官方 DSV3 用 offload，Hy3 上要改回来
tile_batch_seq=512 tile_embed_dim=1024 tile_mlp_dim=1024
attention=flash dtype=bfloat16 weight_dtype=float32
```

配 26 个 XLA flag（官方 30 个里本镜像 libtpu 认的 23 个 + 3 个
SparseCore 运行模式）。**注意 libtpu 对不认识的 flag 是硬失败**——
`Unknown command line flag 'xla_tpu_bf16_emission_mode'` 直接让进程退出，
所以照抄别家配方前必须先在小池子上筛一遍。


### 4.5 同一件事在 4 芯片和 256 芯片上结论相反

256 卡池被 spot 抢空时，我在 4 芯片 dev pod 上跑了同一套消融当备用轨道。
**大部分方向一致，但有一项完全反过来：**

| 改动 | 4 芯片 | 256 芯片 |
|---|---|---|
| 去 26 个 XLA flag | 6.25% → 5.25%（−16%） | 31.56% → 27.49%（−13%，即 o3） |
| 去 `tile_*` | 6.25% → 6.25%（0） | 31.56% → 31.25%（−1%） |
| 去 `out_proj=offload` | 6.25% → 5.76%（**−8%**） | 31.56% → 31.78%（**+0.7%**） |
| `sa_block` 1024 → 512 | 6.25% → 6.21%（−0.6%） | — |
| **换 dropping** | 6.25% → **6.40%（+2.4%）** | 31.56% → **23.71%（−25%）** |

**dropping 在 4 芯片上是赚的，在 256 芯片上是亏的。**
小规模下 GEMM 形状静态化的收益占主导；到 256 卡，
固定容量意味着每张卡都要按最坏情况分配缓冲，
HBM 从 62.4 G 涨到 79.6 G，通信量也跟着涨——收益被淹没了。

> **教训：小规模消融能筛掉明显错误的方向，但不能用来选最优。**
> 4 芯片上 dropping 快 2.4%，照这个结论去 256 卡上配置，
> 会白丢 25% 的吞吐。备用轨道的价值是"别让夜里空转"，不是"替代真实规模"。


### 4.6 显存测算 vs 实测

原先按 **294.9 B / 128 device** 估的：

| 组成 | 计算 | 总量 | 每 device |
|---|---|---|---|
| 权重 BF16 | 294.9 B × 2 B | 590 GB | 4.6 GB |
| 梯度 BF16 | 294.9 B × 2 B | 590 GB | 4.6 GB |
| Adam 状态 + FP32 master | 294.9 B × 12 B | 3.54 TB | 27.7 GB |
| **静态小计** | | **4.72 TB** | **36.9 GB** |

实跑（256 device，`opt_type=adamw` + `mu_dtype=bfloat16` + `grad_dtype=bfloat16`）：

```
number parameters: 298.786 billion       <- 含 MTP 头，比 SSOT 多 3.886 B
Total memory size: 54.8 GB
  Output 10.9 GB | Temp 43.9 GB | Argument 10.9 GB | Host temp 2.5 GB
```

**每 device 54.8 GB / 95.74 GB，占 57%。** 比表里估的 36.9 GB 高，
原因是那张表只算了静态状态，没算 43.9 GB 的临时缓冲（激活、
all-to-all staging、megablox 的分组重排）。**MoE 的临时开销比静态权重还大**——
这是稠密模型的估算习惯套到 MoE 上会踩的坑。

结论不变：**这个规模不需要 PP**，纯 FSDP 就装得下
（对比 GB300 因单域只有 64 卡才需要 PP=2）。**也不需要 EP**——§4.2 实测 EP=64 直接超显存。


### 4.7 spot 抢占怎么应对

o2 跑到 step 5 时整个节点（`gke-tpu-34dda87d-cz54`）被回收，
节点池从 64 台掉到 **2 台**，一小时后才补回 64 台。这一夜被抢了两次。

跑批脚本因此改成**自愈队列**：

- **等节点**：轮询到 64 台 Ready 才提交，最多等 90 分钟
- **唯一命名**：每轮 JobSet 名字带 `HHMMSS` 后缀。
  之前用固定名字时，`kubectl logs job/hy3-o2-...` 读到了**上一轮同名 pod 的残留日志**，
  把正在正常运行的 o2 误判成 FAIL——这个假失败浪费了我一轮
- **提交后先睡 120 s** 再开始轮询，避开 pod 还没起来的窗口
- **拿到 2 个稳态点就收**，不等跑满 15 步。o2 的 step 3 和 step 4
  是 34.674 / 34.673 s，第三位小数才有差别——再多跑十步也是这个数

另外两次被同一个坑绊到：`pgrep -f hy3-sweep.sh` **会匹配到执行它自己的那个
shell**（命令行里含有这个字符串），导致 `while pgrep ...; do sleep; done`
永远退不出来，等待的下一批扫描根本没启动。
判断脚本是否在跑要用 `ps -eo pid,cmd | awk '$2=="bash" && $3=="/path"'` 这种精确匹配。


### 4.8 复现验证：用仓库里的产物从零跑一遍

文档写完之后做了一轮**复现审计**——不用我的工作目录，
而是把当时的 `register-hunyuan3.patch` 打到一棵
干净的 MaxText `3eb77db3c` 上，再用
当时的 `run-v5p-256.sh` 提交。（这两个产物已随 §5.5 的合并删除；
现在的复现入口是 [`run.sh`](maxtext-hunyuan3/run.sh) `PLATFORM=v5p`。）

结果**逐位对上**：

| | 原始 o12 | 复现 audit1 | 差 |
|---|---|---|---|
| 稳态 step | 59.601 s | 59.589 s | −0.02% |
| TFLOP/s/device | 168.555 | 168.593 | +0.02% |
| MFU | 36.72% | 36.73% | — |
| tok/s/device | 1,099.564 | 1,099.806 | +0.02% |
| 参数量 | 298.786 B | 298.786 B | 一致 |
| `total_weights` | 16,777,216 | 16,777,216 | 一致 |

稳态五步 59.589 / 59.584 / 59.593 / 59.597 / 59.600，抖动在毫秒级。

#### 审计抓到的五个问题

**这一轮的价值不在"跑出了同样的数"，而在跑之前、跑之后、以及修完再验的时候，各抓到一批错。**

| # | 问题 | 性质 |
|---|---|---|
| 1 | **文档里没有一条可执行的训练命令**（已修，见 §九） | 全文 4 个 bash 块：一个引用仓库里不存在的 `/tmp/hy3-iter.sh`，一个是带 `<改过的 8 个文件>` 占位符的伪代码，一个是参数清单。**照文档复现不出来** |
| 2 | **归档的 patch 少一个文件** | `register-hunyuan3.patch` 覆盖 5 个文件，实跑需要 6 个——漏的正是 `maxtext_utils.py`，也就是 §3.9 那个 FLOP 修复。**照旧 patch 打完，MFU 会虚高 5 倍**，正好是文档花一整节解释的那个 bug |
| 3 | **v7 的 `port.py` 少两个文件** | 只覆盖 4 个，实跑需要 6 个——漏 `configs/types.py` 和 `layers/nnx_decoders.py`，也就是 bug #6/#7/#8 对应的改动。因为 port.py 是在发现那三个 bug **之前**写的 |
| 4 | **结论先行的表里串了行** | 稳态 step 写的 47.67 s 是 o8 的数，同一行的 168.6 / 36.72% / 281,488 却是 o12 的。**四个数里三个对、一个来自另一次实验** |

**5. v7 的 `port.py` 静默漏改一处。** 修完问题 3 之后拿它在一棵全新的干净树上重跑，
发现 `utils/maxtext_utils.py` 只改到 2 处，应该是 3 处——
漏的是 FLOP 那个 MoE FFN 分支，因为上游把单行 tuple 拆成了多行，
我那条单行正则匹配不到。而 `edit()` 只断言"文件变了"，
另外两处改动成功就让整个函数通过了。

```python
# 改前：只要文件有任何变化就算过
assert s != s0, f"{rel}: 没有任何改动"

# 改后：按命中处数断言，漏一处当场炸
n = s.upper().count("HUNYUAN3")
assert n == expect, f"{rel}: 命中 {n} 处，期望 {expect} 处"
```

> **这是本项目第二次栽在"断言太弱"上。** 第一次是早期那个
> `verify_hunyuan3.py`（§2.9，未随仓发布）：它用固定行窗口去取 `if` 语句，
> 上游加了一行注释就取错位置。
> 两次的形态一样：**校验写得比它要保护的东西更宽松，
> 于是它在该报警的时候保持沉默。**
> 当时的修法是把 `port.py` 的期望值从一次已验证正确的运行里量出来再写死。
> **这两个脚本后来都删了**（§10.2），但这条纪律留了下来 ——
> 现在 `prep.sh` 的 8 项自检同样是按「命中几处」断言，不是按「有没有变」。

前四个里的前三个都是**"文档描述的东西和仓库里的东西不一致"**：
我在实跑中不断改代码，但归档是某个时间点拷过去的，之后没再同步。
第四个是**填表时从错误的行取数**——两次实验的数字混在一起，
单看每个数都合理，只有把整行放回原始 CSV 才看得出来。

> **归档产物必须从实跑的那棵树重新导出，不能凭记忆拷。**
> 现在的 patch 是 `git diff` 直接生成的，覆盖 6 个文件，
> 已验证能干净打到 `3eb77db3c` 上并通过 8 项静态自检。

---

## 五、v7 Ironwood：移植与跑通


### 5.1 v7 逼出了新版 MaxText —— 后来两个平台都收敛到它

> **本节是历史。** 结论已经变了：**新镜像两代硬件都能驱动，仓库里只留一份代码**（§5.5）。
> 保留这一节是因为「新旧两版 MaxText 差在哪」这张表本身仍然有用。

当初 v5p 侧用的 `maxtext-stable:oct` 镜像**驱动不了 Ironwood**：

```
libtpu build label: libtpu_lts_20250721_b_RC01
jaxlib._jax.XlaRuntimeError: INTERNAL: Failed to get global TPU topology.
```

2025 年 7 月的 libtpu 不认识 tpu7x。换成 `maxtext-latest:runner`
（ironwood 配方用的那个）之后，发现上游 MaxText 已经整体重构：

| | 旧（v5p 用的） | 新（v7 用的） |
|---|---|---|
| 包路径 | `src/MaxText/` | `src/maxtext/` |
| 神经网络框架 | flax **linen**（`nn.compact`） | flax **nnx** |
| 层代码位置 | `layers/deepseek.py` | `models/deepseek.py` |
| 类型定义 | `common_types.py` | `common/common_types.py` |
| 工具 | `maxtext_utils.py` | `utils/maxtext_utils.py` |
| 配置校验 | 手写 `validate_model_name()` | **pydantic** `Literal[...]` |
| 训练入口 | `src.MaxText.train` | `src.maxtext.trainers.pre_train.train` |

好消息是**新版把我记在 §2.8 的三个缺口补了两个**：

| 缺口（旧版） | 新版上游 |
|---|---|
| router 需要 fp32（我加了 `moe_router_dtype`） | 已有 `float32_gate_logits` |
| 专家 bias 的无梯度更新规则未实现 | 已有 `routed_bias_update_rate` |
| 初始化 `initializer_range` | 仍缺（不影响 SFT / 续训） |

所以移植后的 `hunyuan3.py` **比原来短**：Hy3 = Qwen3 的 attention +
DeepSeek 的 MoE，新版正好有 `AttentionWithNorm` 基类和 `RoutedAndSharedMoE`，
两个层类各自只跟 Qwen3 的对应类差**一行**。


### 5.2 v7 建池跟 v5p 完全不是一套流程

试了四次才成，每次的错都不一样，记下来省得别人再踩：

| 尝试 | 命令 | 报错 |
|---|---|---|
| 1 | `--placement-type=COMPACT`（v5p 的写法） | `tpu7x-standard-4t ... with placement policy is not supported. Use workload policy instead.` |
| 2 | 去掉 `--placement-type` | 同上——**GKE 在多机拓扑下会自动加 group placement** |
| 3 | 自建 workload policy（只给 `--type=HIGH_THROUGHPUT`） | `does not support TPU topology with group placement policy and workload policy at the same time` |
| 4 | workload policy 加 **`--accelerator-topology=4x4x4`** | 通过 |

当时的写法（来自 [tpu-recipes ironwood 配方](https://github.com/yangwhale/tpu-recipes/tree/main/training/ironwood)）：

> ✅ **2026-07-31 复核：gcloud 577.0.0 上 `resource-policies create workload-policy`
> 存在且可用**，下面两条命令直接照抄即可，不需要走 REST。
> （早前记的「577 上已删除」是错的，已更正。）
>
> 建池时**不要同时传 `--tpu-topology`** —— 会自动附加 group placement policy 跟
> workload policy 冲突。拓扑由 workload policy 的 `--accelerator-topology` 携带。

```bash
# v7 不会自动建 placement policy，必须先手工建，而且要带拓扑
gcloud compute resource-policies create workload-policy tpu7x-64chip \
  --region=us-central1 --type=HIGH_THROUGHPUT --accelerator-topology=4x4x4

gcloud container node-pools create np-v7x-64-hy3 \
  --cluster=... --region=us-central1 --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=4x4x4 --num-nodes=16 --spot \
  --placement-policy=tpu7x-64chip \
  --disk-type=hyperdisk-balanced --disk-size=200   # v7 必须 hyperdisk
```

> **第三次栽在同一件事上。** 路由分支、FLOP 公式、现在是建池命令——
> 每次都是"我按理解推了一个写法"而不是"先去找现成的配方"。
> 前面刚因为这个丢了 12.9 倍性能（§4.1），转头又来一遍。

容量方面：第一次 `Atomic resize failed with [GCE_STOCKOUT]`，
`us-central1-ai1a` 对本集群不可用，项目里也没有 tpu7x 预留。
改成 **64 → 32 → 16 递降重试**（spot 容量是波动的），
第二轮就在 `us-central1-c` 拿到了 64 芯片。多机 TPU 池是**全有全无**的：
4x4x4 要求物理连续立方体，16 台必须一次落位。


### 5.3 首跑结果

MFU 分母：**2,307** TFLOPS/chip；v7 是 2 device/chip，
**per-chip TFLOP/s = 日志值 × 2**（换算见 TPU-UNITS）。

| # | 配置 | step | TFLOP/s/dev | **per-chip** | **MFU** | tok/s/dev | 整机 tok/s |
|---|---|---|---|---|---|---|---|
| **V1** | FSDP=128 / 无 EP / pdbs=4 / seq=8192 / 只带 2 个 XLA flag | 25.11 s | 202.38 | **404.75** | **17.54%** | 1,305.1 | **167,059** |
| V2 | V1 + 补齐 XLA flag 集 + pdbs 上探 | | | | | | ⬜ |

```
number parameters: 298.786 billion          <- 与 v5p 侧逐位一致，移植没有走样
completed step: 5, seconds: 25.107, TFLOP/s/device: 202.375, loss: 12.815
completed step: 6, seconds: 25.107, TFLOP/s/device: 202.375, loss: 12.722
completed step: 7, seconds: 25.109, TFLOP/s/device: 202.358, loss: 12.645
completed step: 8, seconds: 25.111, TFLOP/s/device: 202.341, loss: 12.585
```

**V1 的 17.54% 只是起点，不是 v7 的水平。** v5p 侧从 2.45% 调到 36.72% 用了
26 个 XLA flag + pdbs=8；v7 这一轮只带了 2 个 flag、pdbs=4，
而且 `xla_tpu_enable_latency_hiding_layer_scheduler` 在 v7 上报
`requires sparse core collective aggregator to be enabled` 被迫摘掉。
按 v5p 的调优幅度推，v7 还有很大空间。


### 5.4 复现验证：v7 侧同样从仓库产物跑一遍

跟 §4.8 一样，不用工作目录：从镜像里拉一棵**干净的**新版 MaxText，
`cp hunyuan3.py` + `cp hunyuan3-295b.yml` + 跑
当时的 `port.py`（现已删除，改用分支），再用
[`run.sh`](maxtext-hunyuan3/run.sh)（`PLATFORM=v7`）提交。

| | 原始 c1 | 复现 v7audit | 差 |
|---|---|---|---|
| 稳态 step | 20.425 s | 20.437 s | +0.06% |
| TFLOP/s/device（日志值） | 222.56 | 222.46 | −0.04% |
| **TFLOP/s/chip** | **445.12** | **444.93** | −0.04% |
| **MFU** | **19.29%** | **19.29%** | — |
| tok/s/device | 1,604.4 | 1,603.3 | −0.07% |
| 整机 tok/s | 205,364 | 205,225 | −0.07% |
| 参数量 | 298.786 B | 298.786 B | 一致 |

> 每 device 每步固定 8 × 4096 = 32,768 token，所以 tok/s 完全由 step 时间决定。
> 本表用 20.425 / 20.437 s 精确反推；其他章节的 c1 整机 tok/s 记的是 **205,314**，
> 那是按 step 取两位小数（20.43 s）算的 —— **差 0.02%，是取整不是分歧**。

稳态四步 20.435 / 20.438 / 20.437 / 20.440，抖动毫秒级。
**两个平台都能从仓库里的东西复现出来。**

#### 顺带确认的一件运维事

复现前 v7 节点池被 spot 抢空，状态从 `RECONCILING` 变 `ERROR`：

```
canonicalCode: RESOURCE_EXHAUSTED
TPU: the nodes cannot be created now due to lack of capacity.
They will be created asynchronously once capacity is available.
You can either wait for the nodes to be up, or delete the node
pool and try re-creating it again later.
```

等了 1.5 小时（每 2 分钟轮询）**一台都没补回来**。
按报错自己的建议**删掉重建，5 分钟内 16 台全 Ready**。

> **`ERROR` + "will be created asynchronously" 这句话有误导性。**
> 它听起来像"排着队，等就行"，实测等 1.5 小时没动静，
> 删了重建立刻就有。多机 TPU 池是原子分配的，
> 一个卡住的申请单不会自己重排——**删了重建比等更快**。

---

### 5.5 反向验证：把 v5p 也搬到新栈（2026-07-28）

§5.1 说「v7 用的是另一套 MaxText，补丁得重写」，于是仓库里长期挂着**两套代码**。
但那是**镜像绑定的历史包袱，不是平台差异**：v5p 那个镜像（`maxtext-stable:oct`，
libtpu_lts_20250721）驱动不了 Ironwood，所以 v7 只能换新镜像，一换才发现上游整体重构了。

反过来问一句就没人问过：**新镜像能不能驱动 v5p？** 能的话两套就该合成一套。

#### go / no-go：能

在既有 `np-v5p-256` 上用 `maxtext-latest:runner` 起一个 4 层缩层冒烟（结构与 295B 完全一致，
只砍层数），**新 libtpu 完整识别 256 个 v5p device**，loss 13.42 → 11.68 单调下降，MTP 正常出数，零 NaN。
（**这是 256 芯片上的冒烟**，验的是 libtpu 认不认 v5p；§9.4 那条 13.45 → 10.35 是 4 芯片跑 8 步的，两回事。）

```
JAXDEV 256 TPU v5
completed step: 7, seconds: 0.750, loss: 12.878, main_model_loss: 11.676, mtp_loss: 1.202
```

#### 换栈踩的四个坑，全都不是代码逻辑问题，而是**版本契约变了**

| # | 症状 | 根因 | 处理 |
|---|---|---|---|
| 1 | `Unknown command line flag '2a886c8_chip_config_name'` | §4.4 那 26 个 XLA flag 是配**旧 libtpu** 的，新 libtpu 摘掉了这个 | 删掉，剩 25 个。**XLA flag 集绑 libtpu 版本，换镜像必须重过一遍** |
| 2 | `'tile_batch_seq' not in <一长串合法字段>` | 旧版 3 个 tile 参数，新版拆成 **18 个**：`{wi,wo}_tile_{fwd,dlhs,drhs}_{batch_seq,embed_dim,mlp_dim}` | 同值展开 18 个。**注意这只是"形式对齐"，不保证最优——新版允许六条通路各自不同** |
| 3 | `'Hunyuan3MoELayer' object has no attribute 'DeepSeekMoeBlock_0'` | `trainers/pre_train/train.py` 把 DeepSeek 的模块属性名**写死在无梯度 bias 更新路径里**，两处 | 我方属性命名为 `Hunyuan3MoeBlock_0`（诚实命名，不冒用 DeepSeek 名字），改 train.py 按 `decoder_block` 查表 |
| 4 | `FAILED_PRECONDITION: GetSliceInfo can only be invoked after a slice is built` | **我自己加的探针**：启动脚本里一行 `jax.device_count()` 会初始化 TPU 然后进程退出，单机没事，64 台一起跑就毒化真跑 | 删掉探针 |

> 坑 3 是「按家族名字写死」这个模式的**第 8、9 次**出现（§八 台账），而且这次**不在任何分派表里，在训练主循环**。
> 见 §八 bug #10 —— 它同时反向证明了 `moe_bias_updates` 那个修复真的生效：
> 崩溃点在 `if ... and moe_bias_updates is not None:` **通过之后**，Python 的 `and` 会短路，
> 所以「它崩了」本身就是「更新量确实传上去了」的证据。
>
> 坑 4 值得单独记：**诊断用的探针本身可以是故障源**。多机 TPU 上任何「初始化后就退出」的进程都会破坏切片。

#### 同参数移植后的性能：新栈慢 4.6%

完整 295B / 256 芯片 / §4.4 的 o12 参数原样搬过来（除上面两处被迫改动）：

| | 旧栈 `3eb77db3`（linen） | 新栈 latest（nnx） | 差 |
|---|---|---|---|
| step | 59.6 s | **63.193 s** | +6.0% |
| TFLOP/s/device | 168.6 | **160.81** | **−4.6%** |
| MFU | 36.72% | **35.03%** | −1.69 pp |
| 整机 tok/s | 281,488 | **265,472** | −5.7% |

新栈稳定性更好：9 步区间 63.178–63.204 s，**±0.02%**。

> ⚠️ **两版的 FLOP 统计口径不完全一致，跨版本比 MFU 要留意。**
> 反推每步 FLOP：旧栈 `168.6 × 59.6 = 10,048`，新栈 `160.81 × 63.193 = 10,162`，差 **1.1%**。
> 同模型同 batch 不该有这个差，说明分析式 FLOP 公式改过。
> 也就是说 −4.6% 里有约 1 pp 是**口径**，真实性能差约 −3.5%。

**这 4.6% 目前有两个未排除的不等价项**：被迫删掉的那个 XLA flag，以及 tile 参数的 18 路同值展开
（同值不等于最优）。消融结果见下节。

#### 扫这批新旋钮时踩的三个坑（都跟旋钮本身无关）

一开始我按「一次动一个」扫 16 组，Chris 一句「就不能一次性全放上看结果吗」把策略掀了——
**一把梭赢了就收工，崩了再二分是 log 级不是线性级**。这跟 §4.1「应该从官方全量配方出发」是同一条道理。
换成全开之后连崩三轮，三个坑全是方法论问题：

**坑 A：报错的第一条和最后一条不是同一回事。**
连续三轮都报 `FAILED_PRECONDITION: GetSliceInfo can only be invoked after a slice is built`，
我据此编了两个故事——先怪「删 JobSet 没等切片释放」，再怪「强删 pod 留了残留」，**两个都是错的**。
真凶在日志更早的位置：

```
MAXTEXT CONFIG ERROR: Value error, GMM v2 requires `use_tokamax_gmm=true`
```

pydantic 的配置校验发生在 **TPU 初始化之后**，所以 pod 先把 TPU 拉起来、再因配置非法秒退，
切片凑不齐 64 个成员 → 活着的那几个报 `GetSliceInfo` 失败。
**`GetSliceInfo` 是下游症状，配置错误才是病因。**
我的 grep 模式里没有 `MAXTEXT CONFIG ERROR` 和 pydantic 的 `Value error`，所以只看见了次生错误。

> **规矩：多机作业判错先看「最早的那条」和「有多少 pod 活着」，不要抓日志尾。**

**坑 B：判读结果前先数人头。**
其中一轮 JobSet 只创建了 **36/64** 个 pod。TPU 切片全有全无，缺一个都建不起来。
我当时直接看日志下结论，其实测的根本不是配置。
→ 现在流程固定为：提交 → sleep 75 → **先确认 `64 Running`** → 才开始计时等稳态。

**坑 C：TPU pod 不要 `--force --grace-period=0`。**
强删让 pod 卡在 Terminating 占着节点，下一批排不进去。要等优雅退出让驱动释放 `/dev/vfio`。
（不过这一条虽然真实，**并不是这三轮失败的原因**——见坑 A。把两件事分开记，避免下次又拿它当万能解释。）

**顺带查清一个隐式依赖**：`use_gmm_v2` 强制要求 `use_tokamax_gmm=True`，
而 §6.7 实测 `use_tokamax_gmm` 在 Hy3 上**会死锁**。这两个旋钮被绑死，其中一个是已知地雷，
所以 gmm_v2 在 Hy3 上暂时无法使用（除非先解决死锁，见待验证清单）。

#### 消融结果：新栈的新旋钮，在 Hy3 上一个都不是正的

| 配置 | step | TFLOP/s/device | MFU | vs 基准 |
|---|---|---|---|---|
| **基准**：旧栈 o12 参数原样移植 | 63.19 s | **160.81** | 35.03% | — |
| A 组：MoE / FSDP 六个 | 66.96 s | 151.75 | 33.06% | **−5.6%** |
| B 组：splash 五个 | 70.51 s | 144.11 | 31.40% | **−10.4%** |
| 全开（A + B，11 个） | 74.22 s | 136.90 | 29.83% | **−14.9%** |

A 组 = `prefuse_moe_weights` `fuse_expert_scales` `merge_gating_gmm` `use_ragged_sort`
`moe_fsdp_use_two_stage_all_gather` `dense_fsdp_use_two_stage_all_gather`
B 组 = `use_tokamax_splash` `use_splash_scheduler` `sa_fuse_reciprocal` `sa_use_base2_exp` `sa_use_fused_bwd_kernel`

> **两组的损失可以相乘**：`0.944 × 0.896 = 0.846`，预测 136.0，实测 136.9，**误差 0.67%**。
> 说明两组互不干涉，不用再测交叉项——**二分到这里就可以收手**。

B 组更狠这件事事后看很自然：`sa_use_fused_bwd_kernel` 在旧栈 §4.4 的消融里
**本来就是关掉才快**（o12 里显式设了 `False`）。我"全开"的时候等于把一个已知负项又打开了。
**「新版本提供的旋钮」不等于「该开的旋钮」**，旧版调出来的负结论不会因为换栈就失效。

#### 结论：这是一笔用吞吐换可维护性和能力的交易，不是性能升级

| 角度 | 说明 |
|---|---|
| ❌ 原以为的理由 | 新栈有新旋钮，说不定更快 → **实测证伪，全是负的** |
| ✅ 真正的理由 | **一份代码不是两份**；上游把 §2.8 那两个缺口补了（router fp32、**专家 bias 的无梯度更新**）。后者是**能力**不是性能——from-scratch 预训练没有它专家会塌缩 |
| 💰 代价 | 名义 −4.6%，扣掉约 1 pp 的 FLOP 口径差，**真实约 −3.5%** |

**决定：合并。** 3.5% 换一份代码 + 一个从零预训练必需的能力，值。
旧的 linen 补丁、旧镜像步骤、两个平台各自的 run 脚本已全部删除，
现在只剩一套：代码在分支上，入口是 [`run.sh`](maxtext-hunyuan3/run.sh) 加 `PLATFORM=v5p|v7`。

> **§4 那个 36.72% 是旧栈上测的，它的复现产物已经不在仓库里了。**
> 保留那个数字是历史记录（说明 2.45% → 36.72% 那条 15 倍的调优路径真实发生过）；
> **当前可复现的 v5p 水位是 35.07%**（§7.3 战线三 b0″），用 `run.sh PLATFORM=v5p` 得到。

---

## 六、v7 性能调优：目标与进展


### 6.1 目标该定在哪

先回答"为什么不是 900"。Ironwood 官方实测表（[tpu-recipes/training/ironwood](https://github.com/yangwhale/tpu-recipes/tree/main/training/ironwood)，
全部 bf16、synthetic、per-chip 口径）：

| 模型 | 类型 | chips | 序列 | **TFLOP/s/chip** | MFU |
|---|---|---|---|---|---|
| llama3.1-405b | **稠密** | 256 | 8192 | **1,261.4** | 54.7% |
| llama3.1-70b | **稠密** | 64 | 8192 | **1,207.1** | 52.3% |
| gemma4-31b | **稠密** | 64 | 8192 | **931.3** | 40.4% |
| gemma4-4b | 稠密 | 64 | 8192 | 1,002.5 | 43.5% |
| gemma4-26b | 稠密 | 64 | 4096 | 592.3 | 25.7% |
| **qwen3-235b-a22b** | **稀疏 MoE** | 256 | 4096 | **629.8** | 27.3% |
| **deepseek-v3 671B** | **稀疏 MoE** | 256 | 4096 | **612.7** | 26.6% |
| deepseek-v3 671B | 稀疏 MoE | 128 | 4096 | 607.5 | 26.3% |
| gpt-oss-120b | 稀疏 MoE | 256 | 8192 | 329.9 | 14.3% |
| **hunyuan3 295B（本项目首跑）** | **稀疏 MoE** | **64** | **8192** | **404.8** | **17.6%** |

**900 以上全是稠密模型。** 稀疏 MoE 在 Ironwood 上的实际水位是
**600–630 TFLOP/s/chip（26–27% MFU）**，最接近 Hy3 的两个参照——
qwen3-235b-a22b（629.8）和 deepseek-v3（612.7）——都在这条线上。

原因是结构性的，不是调参能翻越的：

- **稠密模型每个 token 走同一套权重**，GEMM 又大又规整，MXU 能吃满
- **MoE 每层要做一次路由、一次按专家分组重排、一次分组矩阵乘、一次还原**。
  分组矩阵乘的每个子块只有 `tokens_per_expert × emb × moe_mlp` 那么大，
  而且组大小随路由结果浮动，编译期拿不到静态形状
- 还要加上 all-gather / reduce-scatter 把 192 份专家权重摊开又收回

所以本项目 v7 侧的目标定为 **600–630 TFLOP/s/chip**，
对应 step 时间从 25.11 s 压到 **16–17 s**。当前 404.8，缺口 **1.5×**。


### 6.2 DeepSeek V3 在 v7 上的实测：BF16 与 FP8

数据源：[tpu-recipes/training/ironwood](https://github.com/yangwhale/tpu-recipes/tree/main/training/ironwood)
研发实测表。**跑的是满 61 层 671B，没有缩层代理**（manifest 里没有
`base_num_decoder_layers` 覆盖）。v7 峰值按 wiki 核实：
**BF16 2,307 / FP8 4,614 TFLOPS per chip**。

| 模型 | chips | 精度 | step | **TFLOP/s/chip** | MFU（对本精度峰值） | tok/s/chip |
|---|---|---|---|---|---|---|
| deepseek-v3 671B | 128 | **bf16** | 27.02 s | **607.53** | **26.3%** | 2,425.8 |
| deepseek-v3 671B | 128 | **fp8_full** | 22.47 s | **730.60** | **15.8%** | 2,917.2 |
| deepseek-v3 671B | 256 | **bf16** | 26.79 s | **612.66** | **26.6%** | 2,446.3 |
| deepseek-v3 671B | 256 | **fp8_full** | 22.08 s | **743.46** | **16.1%** | 2,968.5 |
| qwen3-235b-a22b | 256 | bf16 | 30.87 s | 629.79 | 27.3% | 4,245.9 |
| qwen3-235b-a22b | 256 | fp8_full | 27.67 s | 702.60 | 15.2% | 4,736.7 |
| llama3.1-405b（**稠密**） | 256 | bf16 | 98.34 s | 1,261.40 | **54.7%** | 499.8 |
| llama3.1-405b（**稠密**） | 256 | fp8_full | 64.37 s | 1,927.15 | **41.8%** | 763.5 |

recipe 文档自己的口径说明也印证了这一点：

> Against a v7 per-chip BF16 peak of 2,307 TFLOPS,
> 608 TFLOP/s/chip is roughly **26.4% MFU**.

**所以 DSV3 在 v7 上的 BF16 MFU 就是 26.3–26.6%，这是研发实测值，不是估算。**

FP8 峰值是 BF16 的两倍（4,614 vs 2,307），但 DSV3 的吞吐**只涨了 21%**
（612.66 → 743.46）。换算成 MFU：

| | BF16 | FP8 | 说明 |
|---|---|---|---|
| DSV3 TFLOP/s/chip | 612.66 | 743.46 | **+21.4%** |
| 对本精度峰值的 MFU | 26.6% | **16.1%** | 峰值翻倍但吞吐没翻，MFU 反而掉 |
| 若错误地对 BF16 峰值算 | 26.6% | 32.2% | **这么算会高估一倍** |
| 稠密 llama3.1-405b 同口径 | 54.7% | 41.8% | 稠密 FP8 涨 **+52.8%**，兑现度高得多 |

**MoE 兑现不了 FP8 的两倍峰值，稠密可以。** 原因跟 §6.1 开头那条一样：
MoE 的时间大量花在路由、分组重排、all-to-all 和小块 GEMM 上，
这些环节不吃 MXU 峰值，把精度从 16 位降到 8 位对它们几乎没帮助。
稠密模型的时间集中在大 GEMM 上，降精度直接兑现。

> **报 FP8 的 MFU 时一定要说明分母。** 同一个 743.46，
> 对 FP8 峰值算是 16.1%，对 BF16 峰值算是 32.2%——差一倍。
> 本文所有 FP8 的 MFU 都对 **FP8 峰值** 算。

| | TFLOP/s/chip | MFU | 相对 DSV3 |
|---|---|---|---|
| DSV3 671B bf16（研发实测） | 612.66 | 26.6% | 1.00× |
| qwen3-235b-a22b bf16 | 629.79 | 27.3% | 1.03× |
| **hunyuan3 295B bf16（本项目首跑）** | **404.75** | **17.54%** | **0.66×** |

差 **1.51 倍**。Hy3 的激活参数（21 B）比 DSV3（37 B）还少，
结构也更简单（GQA 而非 MLA、192 专家而非 256），
**没有理由跑不到同一水位**——差距来自配置，不是架构。


### 6.3 缺口从哪里补

首跑（V1）跟官方 Ironwood DeepSeek3 配方的差距：

| 项 | V1 首跑 | 官方 Ironwood | 预期作用 |
|---|---|---|---|
| XLA flags | **2 个** | **30 个** | v5p 上这一项值 4.07 pp（13%） |
| `use_tokamax_gmm` | 未设 | **True** | Tokamax 的分组矩阵乘 kernel，MoE 主计算 |
| `use_tokamax_splash` | 未设 | **True** | Tokamax splash attention |
| `sa_use_fused_bwd_kernel` | False | **True** | attention 反向融合 |
| `allow_split_physical_axes` | 未设 | **True** | 允许 mesh 轴跨物理维切分 |
| `opt_type` / `mu_dtype` / `grad_dtype` | 默认 | **adamw / bf16 / bf16** | 优化器状态减半 |
| `use_iota_embed` | 未设 | **True** | 省 embedding 显存 |
| `use_max_logit_estimate` | 未设 | **-1** | attention 数值路径 |
| `cost_estimate_flops_fwd/bwd` | 未设 | **5e12** | 给调度器的代价提示 |
| pdbs × 序列 | 4 × 8192 | **8 × 4096** | 每卡 token 数相同，但短序列的 attention 开销更低 |

`shard_exp_on_fsdp=True` 要求 `num_experts % ici_fsdp_parallelism == 0`。
128 个 device 全给 FSDP 时 192 % 128 = 64 ≠ 0，过不了校验——
我一开始据此把这条判了死刑，**这个判断是错的**：

| `ici_fsdp` × `ici_data` | 192 % fsdp | 能否开 |
|---|---|---|
| 128 × 1 | 64 | ❌ |
| **64 × 2** | **0** | ✅ |
| 32 × 4 | 0 | ✅ |

**只要不把 128 个 device 全给 FSDP，这个开关就能开。**
它把专家权重也切到 FSDP 轴上，省下来的显存能换更大的 batch，
是官方配方里唯一一个我们还没试过的切分手段。列为待测。


### 6.4 手上有哪些可调的

差距 1.51 倍，拆成五类。**A 类是 v5p 上根本不存在的东西，最可能是主因。**

**A. v7 专属内核**（v5p 没有对应物，所以 v5p 的调优经验完全没覆盖这一块）

| 开关 | 作用 | 状态 |
|---|---|---|
| `use_tokamax_gmm` | MoE 的分组矩阵乘内核。**MoE 的主计算就在这** | 🔄 x1 测试中 |
| `use_tokamax_splash` | splash attention 内核 | ⬜ x2 |
| `sa_use_fused_bwd_kernel` | attention 反向融合成一个 kernel | ⬜ x2 |

**B. XLA flag**（v5p 上这一项值 4.07 pp / 13%；v7 上一次全开会死锁，所以分组二分）

| 组 | 内容 | 状态 |
|---|---|---|
| SparseCore 卸载组 | 9 个 `*_sparse_core_collective_offload_*` | ⬜ x4 |
| 调度器组 | 4 个 `*_latency_hiding_layer_scheduler*` | ⬜ x5 |
| 杂项组 | 5 个（dvfs / bf16 emission / opt barrier 等） | ⬜ x6 |

**C. 切分**

| 手段 | 说明 | 状态 |
|---|---|---|
| `shard_exp_on_fsdp` + FSDP=64 × DP=2 | 专家权重也切到 FSDP 轴，省显存换 batch。**上面刚纠正过：这条是能开的** | ⬜ 待测 |

**D. Batch 与序列**

| 手段 | 依据 | 状态 |
|---|---|---|
| pdbs 4 → 8 | v5p 上 pdbs 上探值 +2.87 pp | ⬜ |
| seq 8192 → 4096 配 pdbs=8 | 官方 Ironwood 用的就是这个组合，短序列 attention 开销更低 | ⬜ |

**E. 优化器与显存**（本身不提速，但给 batch 腾空间）

`opt_type=adamw` + `mu_dtype=bfloat16` + `grad_dtype=bfloat16`（优化器状态减半）、
`use_iota_embed`、`allow_split_physical_axes`。

**F. 如果 A–E 扫完还差得远**

不再盲扫参数，**开 `profiler=xplane` 抓 trace**，
看时间到底花在路由、分组重排、all-to-all、offload 还是 GEMM 上。
现在所有推断都是从"跟官方配方的差异"倒推的，
trace 是唯一能直接回答"慢在哪"的东西。


### 6.5 调优轮次与结果

**w1 = 官方参数集一次性全开 → 卡死。**

```
Slow PjRt TPU operation detected: start_time=00:23:05 host_id=7
TpuDiagnosticCoordinator: Harvesting hardware telemetry for stalled chips: [7]
```

编译花了 17 分钟，step 0 本身 61 s，step 1 又隔了 7.5 分钟才出，之后一个芯片挂住。
这跟 v5p 的经验相反——v5p 上"照抄官方全套"一次就成（§4.1）。
差别在于 v7 的 SparseCore 卸载路径和 tokamax kernel 都是 v5p 上没有的，
**照抄的前提是两边的硬件路径一样**，v7 不满足。

改成从已知能跑的 V1 出发**增量叠加**：

| # | 相对前一轮的增量 | seq | pdbs | step | **TFLOP/s/chip** | **MFU** | 整机 tok/s |
|---|---|---|---|---|---|---|---|
| V1 | 基线：2 个 XLA flag | 8192 | 4 | 25.11 s | 404.75 | 17.54% | 167,059 |
| y1 | + `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | 8192 | 4 | 24.45 s | 415.16 | 18.00% | 171,356 |
| y2 | + adamw/bf16 优化器 + `iota_embed` + `split_physical_axes` | 8192 | 4 | 24.61 s | 412.88 | 17.90% | 170,413 |
| y3 | + SparseCore 卸载组（9 个 flag） | 8192 | 4 | 24.61 s | 412.56 | 17.88% | 170,281 |
| y4 | **+ 调度器组（4 个 flag）** | 8192 | 4 | 23.08 s | **440.30** | 19.09% | 181,732 |
| z1 | y1 + 换 batch/序列口径 | 4096 | 8 | 21.69 s | 418.89 | 18.16% | 193,214 |
| **c1** | **调度器组 × pdbs=8/seq=4096** | 4096 | 8 | **20.43 s** | **445.12** | **19.29%** | **205,314** |
| c2 | c1 + 杂项组（补齐 26 flag） | 4096 | 8 | 20.45 s | 444.63 | 19.27% | 205,089 |

失败的轮次同样有信息：

| # | 改动 | 结果 |
|---|---|---|
| w1 | 30 flag + tokamax 一次全开 | **HANG** stalled chips [7] |
| x1 | V1 + `use_tokamax_gmm` | **HANG** stalled chips [7] |
| z2 | pdbs 8 → 12（FSDP=128） | **OOM** 临时缓冲 95.17 G |
| z3 / c3 | `shard_exp_on_fsdp`（FSDP=64 × DP=2） | **OOM** 109.14 G / 94.74 G |
| c4 | c3 + pdbs=12 | **OOM** |

**当前最优 c1：445.12 TFLOP/s/chip，MFU 19.29%，整机 205,314 tok/s。**
相对首跑 +10.0%，距离 600 / 26% 的目标还差 **1.35 倍**。

各项贡献排序：

| 手段 | 贡献 |
|---|---|
| 调度器 flag 组（4 个） | **+6.6%** |
| pdbs=8 / seq=4096（batch 与序列口径） | **+12.8% 吞吐**（z1 对 y1；TFLOP/s 只 +0.9%，口径说明见 §4.4「序列长度」） |
| `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | +2.6% |
| 杂项 flag 组（5 个） | ±0 |
| SparseCore 卸载组（9 个） | ±0 |
| 优化器 / 显存组 | −0.5% |
| `use_tokamax_gmm` | **死锁** |
| `shard_exp_on_fsdp` | **OOM** |

### 6.6 三个负结果

**y2：优化器和显存那一组是 −0.5%。** `opt_type=adamw` + bf16 的动量梯度、
`use_iota_embed`、`allow_split_physical_axes` 加起来不但没提速，还略微掉。
合理——这几项**省的是显存不是时间**，只有拿省下的显存去加 batch 才兑现。

**y3：SparseCore 卸载那 9 个 flag，在 v7 上收益是 0。** 412.56 vs 412.88，
差在噪声里。而**同一组东西在 v5p 上值 4.07 pp（13%）**（§4.4 的 o3）。

这条否定结果直接改变了后面的方向：

> SparseCore 卸载是把 all-gather / reduce-scatter 从 TensorCore 挪走。
> 它在 v5p 上有效、在 v7 上无效，说明 **v7 上 Hy3 的瓶颈不在集合通信本身**。

**但我据此做的下一步判断是错的。** 我当时推理："既然不在通信，
剩下的 flag 组（调度器 4 个、杂项 5 个）期望收益也很低，跳过"，
然后去杀扫描进程。**结果 y4（调度器组）在我杀之前已经跑完了：440.30，
比 y3 高 6.6%，是当时的最优。**

| 轮次 | flag 组 | TFLOP/s/chip | Δ |
|---|---|---|---|
| y2 | 无 | 412.88 | — |
| y3 | + SparseCore 卸载（9 个） | 412.56 | ±0 |
| **y4** | **+ 调度器（4 个）** | **440.30** | **+6.6%** |

两组都是"通信相关"的 flag，一组 0 一组 +6.6%。区别在于：
**SparseCore 卸载改的是通信在哪执行，调度器改的是通信和计算怎么重叠。**
前者无效说明通信不是瓶颈；后者有效说明**通信没跟计算叠起来**——
不是通信太慢，是它没藏住。

> 教训：**"这一组没用"不能外推到"同类的下一组也没用"。**
> 我把"通信不是瓶颈"这个正确结论，错误地扩展成了"所有通信相关开关都没用"。
> 幸好那一轮已经在跑，否则这个 6.6% 就被我自己跳过去了。
> 消融的纪律是**每一组都要真跑**，不能靠上一组的结果替它下结论。

**z3：`shard_exp_on_fsdp` 能开，但对 Hy3 是净亏。**
§6.4 的待测清单里我纠正过一次：这个开关不是对 Hy3 封死，
只要 FSDP 不吃满 128 个 device 就能开。z3 真跑了 FSDP=64 × DP=2：

```
Ran out of memory in memory space hbm. Used 109.14G of 94.74G hbm.
```

**比不开还多用 14 G。** 原因是这笔交易两头都动：

| | FSDP=128 / 不开 | FSDP=64 × DP=2 / 开 |
|---|---|---|
| 专家权重 | 按 `embed` 维切 128 份 | 按专家维切（`shard_exp_on_fsdp` 的收益） |
| **非专家权重** | **切 128 份** | **只切 64 份，另外 2 份是 DP 复制** |
| 净效果 | — | **省的没有多花的多** |

Hy3 的非专家部分（attention 80 层 + embedding + dense 首层）约 7.2 B，
FSDP 从 128 降到 64，这部分每卡分片直接翻倍。
专家那头省下来的量抵不过。

> **"这个开关能开"和"这个开关有用"是两件事。** 我纠正了第一个判断
> （从"封死"改成"能开"），但没有立刻去验证第二个。
> 真跑一轮，结论是**对 Hy3 净亏**——192 不是 2 的幂，
> 代价不是"用不了这个开关"，而是"用它要拿一半 FSDP 宽度去换，不划算"。


### 6.7 `use_tokamax_gmm` 在 Hy3 上会死锁

两次挂死都带这个开关，两次通过都不带——**2 比 0**：

| 轮次 | `use_tokamax_gmm` | 结果 |
|---|---|---|
| V1 | 否 | ✅ 404.75 |
| y1 | 否 | ✅ 415.16 |
| x1 | **是** | ❌ stalled chips [7] |
| w1 | **是** | ❌ stalled chips [7] |

x1 是最干净的判别实验：在跑得好好的 V1 上**只加这一个开关**，
连 step 0 都没跑完就挂住。代码路径也对得上 —— `layers/moe.py` 里那个分支：

```python
if self.config.use_tokamax_gmm:
    ...
    output = mblx.gmm(..., use_tokamax_backend=self.config.use_tokamax_gmm, ...)
elif self.config.megablox:   # Older forked megablox  <- V1/y1 走这条
```

官方 Ironwood DSV3 配方里这个开关是 `True` 且能跑，说明 tokamax 后端
本身没问题，**是它跟 Hy3 的形状不合**——最可能又是 192 个专家：
DSV3 是 256，分组矩阵乘的组数正好是 2 的幂。这条待进一步确认。

> **我在这件事上翻过一次车，值得记。** x1 挂了之后我去 grep
> `moe.py` 里有没有 `tokamax`，返回空，于是我判断"这个开关对我们这条
> 路径是空操作，所以 x1 的挂不是它造成的"。
> 后来直接 `sed` 打印那段代码，`use_tokamax_gmm` 明明白白在第 1489 行。
> **那次 grep 是在一条复合 shell 命令里跑的，被引号吃掉了，静默返回空。**
>
> 教训不是"grep 会出错"，而是：**当实验证据（2/2 相关）和代码阅读
> 冲突时，先怀疑代码阅读。** 实验是黑箱但诚实，代码阅读依赖我没验证过的
> 工具链。我当时应该先重跑一遍 grep 确认它真的在工作。



> **"照抄官方配方"和"一次只动一个维度"不是互斥的，是分场景的。**
> v5p 上官方配方能整套照搬，因为硬件路径一致；
> v7 上整套照搬会死锁，就必须退回增量。
> 判断标准是**两边的执行路径是否相同**，不是"官方的就一定能用"。

---


## 七、性能总表：GB300 / v5p / v7 三方对比


### 7.1 硬件对照

实跑用的是节点池 `np-v5p-256`（4x8x8），**实际 256 芯片**——
不是 Google 命名法里的 `v5p-256`。命名坑见 §3.6。

| | `np-v5p-256`（**实跑**） | v5p-256（Google 命名） | v7 4x4x4 | GB300（参考） |
|---|---|---|---|---|
| Google 加速器类型 | `v5p-512` | `v5p-256` | — | — |
| 芯片数 | **256** | 128 | 64 | 64 GPU |
| JAX device 数 | **256**（1 dev/chip，MegaCore） | 128 | **128**（2 dev/chip） | — |
| HBM / chip | 95.74 GB HBM2e | 95.74 GB | 192 GB HBM3e | 288 GB |
| 总 HBM | **24.5 TB** | 12.25 TB | 12.29 TB | 18.4 TB |
| BF16 TFLOPS / chip | 459 | 459 | **2,307** | 2,700 |
| FP8 TFLOPS / chip | 459（无加速） | 459 | **4,614** | 5,400 |
| 总 BF16 算力 | **117.5 PFLOPS** | 58.8 PFLOPS | 147.6 PFLOPS | 172.8 PFLOPS |

> 原计划拿 **Google 命名的 v5p-256（128 芯片）**去对 v7 4x4x4（128 device），
> 因为两者 device 数相同、总 HBM 接近。手上实际能用的是 256 芯片的池，
> 所以下面的 v5p 数字是 **256 芯片**口径；跟 v7 对比时必须按
> **per-chip** 归一，不能比整机吞吐。
> 单位换算见 [TPU-UNITS](https://github.com/yangwhale/tpu-recipes/blob/main/training/TPU-UNITS.md)。


### 7.2 GB300 基线（已实测，供 TPU 对标）

引自 GB300 文档 §10–§11。64 GPU（16 节点单 NVLink 域），TP=1 / PP=2 / VPP=8 / EP=32。

| 配置 | 精度 | MBS | GPU 数 | Model TFLOP/s | MFU | tok/s/GPU |
|---|---|---|---|---|---|---|
| A1 冠军 | BF16 | 1 | 64 | **854.0** | **31.6%** | 6,242 |
| C1 | FP8_MX | 1 | 64 | 1,285.9 | 23.8% | 9,396 |
| **C2 最快** | **FP8_MX** | **2** | **64** | **1,360.4** | **25.2%** | **9,945** |

MFU 分母：GB300 BF16 峰值 2,700 TFLOPS，FP8 峰值 5,400 TFLOPS。

1. **TP 无用**：attention 只占 2% 参数，切它纯亏通信 → TPU 侧同样应以 EP/FSDP 为主
2. **BF16 是官方口径**：腾讯官方全线 BF16（LLaMA-Factory / ms-swift / DeepSpeed 配置一致），
   `Hy3-FP8` 是 **推理量化产物**不是训练精度。TPU 侧首跑也应用 BF16 对齐口径
3. **显存决定 MBS**：GB300 上 80 层 BF16 想开 MBS=2，full graph / 退 TE graph / PP4
   三种打法全部失败，只有减半权重（减层或换 FP8）才行。
   **这是显存的物理约束，不是调参问题** → TPU 侧要提前算显存，别指望调参绕过

---


### 7.3 全部实测记录（三条战线，一张表）

本项目一共跑了 **三轮独立的调优/验证战役**，下面把每一轮的每个数据点合到一起。
**同一战线内可以直接比，跨战线必须先看归一化口径**（§3.6 / §7.1）。

#### 战线一 · v5p 256 芯片 · 旧栈（linen，MaxText `3eb77db3`）

从官方 DSV3 v5p 配方（o2）出发的 12 轮消融。**这套产物已随 §5.5 合并删除，数字保留为历史记录。**

| # | 相对 o2 的改动 | step | TFLOP/s/dev | MFU | Δ MFU |
|---|---|---|---|---|---|
| **o12** | pdbs 上探到 8（该战线最优） | 59.60 s | **168.56** | **36.72%** | **+5.16 pp** |
| o11 | pdbs=6 + `out_proj=remat` | 46.89 s | 160.70 | 35.01% | +3.45 pp |
| o8 | pdbs 4 → 6 | 47.67 s | 158.04 | 34.43% | +2.87 pp |
| o9 | `out_proj` 不 offload | 34.43 s | 145.86 | 31.78% | +0.22 pp |
| **o2** | （基准，官方配方） | 34.67 s | 144.87 | 31.56% | — |
| o4 | 去掉 `tile_*` 三项 | 35.01 s | 143.45 | 31.25% | −0.31 pp |
| o3 | 26 个 XLA flag → 只留 2 个 | 39.81 s | 126.18 | 27.49% | **−4.07 pp** |
| o5 | 去掉 `use_custom_sort_vjp` | 43.03 s | 116.64 | 25.41% | **−6.15 pp** |
| o10 | 换 dropping（`capacity_factor=1.0`） | 46.16 s | 108.81 | 23.71% | **−7.85 pp** |
| o7 | seq 8192 → 4096 | 21.90 s | 102.64 | 22.36% | **−9.20 pp** |
| o6 | 改回 EP=64 / FSDP=4 | — | — | **OOM** | 超 326.63 G |

> 起点其实是 **2.45%**（我自己瞎调的配置），照抄官方配方直接跳到 31.56%。
> **12.9 倍里的绝大部分来自「从官方配方出发」，不是来自调参**（§4.1）。

#### 战线二 · v7 64 芯片 · 新栈（nnx）

| # | 相对前一轮的增量 | seq | pdbs | step | **TFLOP/s/chip** | **MFU** |
|---|---|---|---|---|---|---|
| **c1** | 调度器组 × pdbs=8/seq=4096（该战线最优） | 4096 | 8 | **20.43 s** | **445.12** | **19.29%** |
| c2 | c1 + 杂项组（补齐 26 flag） | 4096 | 8 | 20.45 s | 444.63 | 19.27% |
| y4 | + 调度器组（4 个 flag） | 8192 | 4 | 23.08 s | 440.30 | 19.09% |
| z1 | y1 + 换 batch/序列口径 | 4096 | 8 | 21.69 s | 418.89 | 18.16% |
| y1 | + `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | 8192 | 4 | 24.45 s | 415.16 | 18.00% |
| y2 | + adamw/bf16 优化器 + `iota_embed` | 8192 | 4 | 24.61 s | 412.88 | 17.90% |
| y3 | + SparseCore 卸载组（9 个 flag） | 8192 | 4 | 24.61 s | 412.56 | 17.88% |
| V1 | 基线：2 个 XLA flag | 8192 | 4 | 25.11 s | 404.75 | 17.54% |
| w1 | 30 flag + tokamax 一次全开 | — | — | **HANG** | stalled chips [7] | — |
| x1 | V1 + `use_tokamax_gmm` | — | — | **HANG** | stalled chips [7] | — |
| z2 | pdbs 8 → 12 | — | — | **OOM** | 临时缓冲 95.17 G | — |
| z3/c3 | `shard_exp_on_fsdp` | — | — | **OOM** | 109.14 / 94.74 G | — |

> **SparseCore 卸载组在 v5p 上值 4 pp，在 v7 上是 0**（y3 相对 y2 −0.02 pp）。
> 同一组 flag 跨硬件代际收益完全不同 —— 这是两个平台唯一实质性的差异来源。

#### 战线三 · v5p 256 芯片 · 新栈（2026-07-28 合并验证）

代码与战线二完全相同，只有 XLA flag 和几个调优项按平台分开。

| # | 配置 | step | TFLOP/s/dev | MFU | vs 基准 |
|---|---|---|---|---|---|
| **b0″** | **异地复现：换项目 / 换 VPC / 换集群（2026-07-29 晚）** | **63.170 s** | **160.98** | **35.07%** | **+0.05%，噪声内** |
| **b0′** | **12 文件全覆盖版（当前 v5p 可复现水位，2026-07-29）** | 63.199 s | **160.91** | **35.06%** | 见下方口径说明 |
| b0 | 7 文件版（同参数，改动前） | 63.193 s | 160.81 | 35.03% | — |
| a1 | + A 组：MoE / FSDP 六个新旋钮 | 66.96 s | 151.75 | 33.06% | **−5.6%** |
| b1 | + B 组：splash 五个新旋钮 | 70.51 s | 144.11 | 31.40% | **−10.4%** |
| m5 | + A + B 全开（11 个） | 74.22 s | 136.90 | 29.83% | **−14.9%** |
| — | smoke（4 层缩层，验证代码路径） | 0.82 s | — | — | loss 13.45→10.35 ✅ 三次逐位相同 |
| m1–m4 | `use_gmm_v2` / `shard_optimizer_over_data` | — | — | **FAIL** | 见 §5.5 三个坑 |

**A 组** = `prefuse_moe_weights` `fuse_expert_scales` `merge_gating_gmm` `use_ragged_sort`
`moe_fsdp_use_two_stage_all_gather` `dense_fsdp_use_two_stage_all_gather`
**B 组** = `use_tokamax_splash` `use_splash_scheduler` `sa_fuse_reciprocal` `sa_use_base2_exp` `sa_use_fused_bwd_kernel`

> **b0 → b0′ 的差是纯口径，不是性能。** 补完 12 处覆盖后重测：
> step **63.193 → 63.199 s（+0.009%）**，在 ±0.02% 的固有抖动内，等于没变；
> 而框架报的每步 FLOP 从 **10,162 → 10,169.38（+0.072%）**。
> 涨的这一点全部来自「最后一层 MoE 补算共享专家」那处修复。
>
> **量级自洽**：80 层 × (8 路由 + 1 共享) = 720 份专家计算，补 1 份 ≈ 0.14%；
> FFN 不是 FLOP 的全部，落到总量上打折后 **0.072% 正好在合理区间**。
> 如果这个数是 0.7% 或 7%，反而说明对该处改动的理解有误。**小、且小得符合预期，本身就是一重验证。**
>
> **两组损失可相乘**：`0.944 × 0.896 = 0.846` → 预测 136.0，实测 136.9，**误差 0.67%**。
> 互不干涉，交叉项不必再测。
>
> **新栈的新旋钮在 Hy3 上一个都不是正的。** 旧栈调出来的 o12 参数搬过来依然是最优附近。
> B 组更差的原因事后很自然：`sa_use_fused_bwd_kernel` 在战线一里**本来就是关掉才快**，
> 「全开」等于把一个已知负项又打开了 —— **「新版本提供的旋钮」不等于「该开的旋钮」**。

**b0″ 异地复现（2026-07-29 晚）.** 换到另一个全新的 GCP 项目：
自建 custom VPC + 全新集群 + 全新 `np-v5p-256` 节点池，
代码从 GitHub 分支现 clone 现打包（`prep.sh`），前置照 §9.0 补齐。

```
number parameters: 298.786 billion
completed step: 3, seconds: 63.174, TFLOP/s/device: 160.974, loss: 13.200
completed step: 4, seconds: 63.176, TFLOP/s/device: 160.969, loss: 13.129
completed step: 5, seconds: 63.174, TFLOP/s/device: 160.974, loss: 13.072
completed step: 6, seconds: 63.171, TFLOP/s/device: 160.981, loss: 13.029
completed step: 7, seconds: 63.170, TFLOP/s/device: 160.984, loss: 12.998
```

对 b0′ 差 **+0.05%**，参数量逐位一致，loss 八步单调下降。
**整条链路没有一样东西继承自旧环境**——这一轮才真正证明了「照文档从零能跑出同一个数」。
代价是暴露了 §9.0 那四条前置全都没写（其中 JobSet 缺失是静默失败，最难查）。

#### 三条战线的口径差异（读表前必看）

| | 战线一 v5p 旧栈 | 战线二 v7 | 战线三 v5p 新栈 |
|---|---|---|---|
| device : chip | 1 : 1 | **2 : 1** | 1 : 1 |
| MFU 分母 | 459 | **2,307** | 459 |
| XLA flag 数 | 26 | **15**（c1 用的；c2 补到 26 也能跑，收益 ±0） | 25（少一个被新 libtpu 摘除） |
| 每步 FLOP（反推） | 10,048 | — | **10,162（+1.1%）** |

> ⚠️ 最后一行很重要：**旧栈和新栈的分析式 FLOP 公式不一样**。
> 同模型同 batch，新栈报的每步 FLOP 多 1.1%。
> 所以战线三对战线一的 −4.6% 里，**约 1 pp 是口径差，真实性能差约 −3.5%**。
> 跨版本比 MFU 之前必须先做这个反推。

### 7.4 三方横向对比

全部按 **per-chip / per-GPU** 归一。三边跑的是同一个 295B-A21B、
同样 BF16、同样 synthetic 数据、同样不开 checkpoint。

| | GB300 64 GPU | v7 4x4x4（64 chips） | v5p 256 chips |
|---|---|---|---|
| 计算单元数 | 64 GPU | 64 chips | 256 chips |
| BF16 峰值/单元 | 2,700 | 2,307 | 459 |
| 序列长度 | 4,096 | 4,096（c1） | 8,192 |
| **实测 TFLOP/s/单元** | **854.0** | **445.1** | **168.6**（旧栈）/ **160.8**（新栈） |
| **MFU** | **31.6%** | **19.3%** | **36.7%**（旧栈）/ **35.0%**（新栈） |
| **tok/s/单元** | **6,242** | **3,208** | **1,100** |
| **整机吞吐 tok/s** | **399,488** | **205,314** | **281,488** |
| 调优程度 | 已调优 | **仍在调**：c1 只是参数扫描的阶段性结果，距 DSV3 的 26.6% 还差 1.38× | 已调优 |
| 当前可复现 | — | ✅ `run.sh PLATFORM=v7` | ✅ `run.sh PLATFORM=v5p` → **161.0 / 35.07%** |

读这张表的三个要点：

**1. v5p 的 36.7% 已经超过 GB300 的 31.6%。** 单卡算力差 5.9 倍，
但 MFU 反而更高——256 芯片的 3D torus + SparseCore 集合通信卸载，
把 MoE 那些碎通信藏得比 NVLink 域还干净。
代价是要 256 张卡才换来 GB300 64 卡七成的整机吞吐。

**2. v7 的 19.3% 离 v7 的上限还远。** 调优后从 17.54% 到 19.29%，
但 DSV3 在同硬件的官方水位是 **26.6%**，还差 1.38 倍。剩下的缺口见 §6.3。

**3. 别用整机吞吐横向比。** v5p 那一列是 256 芯片，
另外两列是 64 个单元。要比性价比得再乘上单价，
这张表只回答"每个计算单元能压出多少"。

> 对比时统一到 **per-chip** 口径。v7 日志是 per-device，需 ×2；v5p 不需要。
> 这是跨代际比较最容易出错的一步。


### 7.5 外部对照复核：一份 64 芯片 v5p 报告（2026-07-29）

收到另一个团队的一份 benchmark 报告（64 芯片 v5p spot，
用的是他们自己的 `maxtext_deepv4_fsdp` 镜像，不是本文档这套）。
它第 6 组标注 **"Aligned with yangwhale Recipe"**，报
**238.31 TFLOP/s/chip、51.92% MFU、19.838 s/step**，
是我们 160.91 / 35.06% 的 **1.48 倍**。逐条核过，**暂不采信**，理由如下。

| # | 不一致 | 它的值 | 应有的值 |
|---|---|---|---|
| 1 | `number parameters` | 20.117 B | 我们 **298.786 B**，差 14.85× |
| 2 | 第 3 组：FSDP=64 + fp32 优化器 + pdbs=2 | 峰值 66.0 GB，EXIT_CODE=0 | 光静态就要 **74.7 GB**（4.669 B/chip × 16 B） |
| 3 | 第 1 组：EP=64 / FSDP=1，params initialized | 3.58 GB | 每芯片约 165 亿参数 → fp32 约 **264 GB** |
| 4 | `abort_on_nan_loss` | **False** | 把最重要的健康检查关掉了 |
| 5 | mu/grad 降 bf16 的收益 | 声称省 26 GB/chip | 按真实参数量算是 **18.7 GB** |

**第 1 条的字段含义已从源码确认**：`metric_logger.py` 打这个数时调的是
`max_utils.calculate_num_params_from_pytree`，
`tree_map(jnp.size)` 后求和 —— 数的是**参数树全部叶子的总量**，不是激活参数量。
报告把 20.117 B 标成 "Active Parameters / Chip"，是读错了字段。

**第 2 条是最硬的反证**：它那一组用的还是 fp32 优化器，
真实 295B 在 FSDP=64 下每芯片 46.69 亿参数，
weights 4 + grads 4 + mu 4 + nu 4 = 16 B/param → 74.7 GB 静态，
一个 activation 都还没算就超了。它在 66.0 GB 跑成功，说明装进去的不是这个模型。

**它的 FLOP 数偏偏是按完整 config 算的**（`calculate_tflops_training_per_device`
读 config 不读参数树），所以分子按 295B 记、分母按小模型跑，
238 TFLOP/s 和 51.92% MFU 都建立在这个错配上。

#### 继承关系：方向是反的

- **FLOP 口径这个坑不是从我们这继承的 —— 是他们没继承到我们的修复。**
  §3.9 bug #5 就是这个坑（虚高约 5 倍），根因是 `HUNYUAN3` 不在
  `calculate_tflops_training_per_device` 的白名单 tuple 里，走通用分支时
  用 `mlp_dim` 去量专家、且漏掉 shared expert。我们把 `HUNYUAN3`
  加进白名单修掉了；他们那个镜像里没有这个改动。
- **真正从我们这抄过去的只有一条**：`mu_dtype=bfloat16 grad_dtype=bfloat16
  use_iota_embed=True`。这三个连在一起，与 `run.sh` 里 v7 分支的 `EXTRA` 一字不差 ——
  但那是 **v7 分支**的 EXTRA，我们 v5p 从没开过。
  等于抄了配方，抄的是另一半平台。

> **教训**：别人声称"对齐了你的配方"时，要问清对齐的是哪个平台分支、
> 用的是哪个镜像。配方是跟着代码走的，不是跟着字符串走的。

#### 自查：这些毛病我们自己有没有

| 项 | 我们 | 说明 |
|---|---|---|
| 参数量 | ✅ 干净 | 298.786 B，且核对过含 MTP 头 3.886 B（清单 #3） |
| FLOP 口径 | ✅ 已修 | §3.9，白名单已含 `HUNYUAN3` |
| 关 nan abort | ✅ 没关 | `run.sh` 未设 `abort_on_nan_loss`，保持默认 True |
| 合成数据 | ⚠️ **同样是** | `run.sh` 里写死 `dataset_type=synthetic`。已在 §7.4 和清单 #13 声明，**loss 不作为收敛证据** |
| 稳态样本 | ⚠️ **偏薄** | 07-29 两轮 256 芯片验证分别跑到 step 9 / step 7，loss 各留 1 个 / 5 个点。下轮补 |

#### 顺带结论：FSDP=64 × DP=4 为什么在真模型上不成立

按真实 298.786 B 算，每芯片 46.69 亿参数：

| 方案 | 静态 | pdbs=8 激活 | 合计 | vs HBM 95.74 GB |
|---|---|---|---|---|
| 现状 FSDP=256，fp32 | 18.7 GB | ~78 GB | **96.9 GB** | 贴顶（见下） |
| FSDP=64×DP4，fp32 | 74.7 GB | ~78 GB | **153 GB** | ❌ |
| FSDP=64×DP4，mu/grad 降 bf16 | 56 GB | ~78 GB | **134 GB** | ❌ |

> **96.9 GB 这个合计数超过了 95.74 GB，但 o12 确实跑通了 —— 所以它不是纯 HBM 占用。**
> `remat_policy=custom decoder_layer_input=offload` 会把一部分激活挪到 host，
> 落在 HBM 里的量必然 ≤ 95.74 GB，否则跟 r10 一样当场 OOM。
> **判断能不能装下，一律以 HBM 的 95.74 GB 为准**；上表的合计是「总需求」口径，
> 用来横向比三个方案的相对压力，不是 HBM 读数。

要装下必须把 pdbs 砍到 4 左右，而 §4.4 的 o8 / o12 实测 pdbs 4→8 值 **+5.2 pp MFU**。
**拓扑上 DP4 确实更贴合 4x8x8（DP 走长度 4 的轴、FSDP 铺 8×8 slab，集合通信更局部），
卡点纯粹是显存，且代价先亏 5 个点。**

---


## 八、十二个 bug 与静态验证的边界

移植过程中又撞了四次"按模型名列举的分支漏了 hunyuan3"，
**全部是运行时才报错，静态检查一个都抓不到**：

| # | 报错 | 位置 |
|---|---|---|
| 6 | `Input should be 'default', 'llama2-7b', ...` | `configs/types.py` 的 pydantic `Literal` 白名单 |
| 7 | `Loss-free load balancing is only supported for the DeepSeek decoder block` | `configs/types.py` 的 validator 把 `routed_bias_update_rate` 锁死在 DEEPSEEK |
| 8 | `Incorrect decoder_block name cfg.decoder_block.value='hunyuan3'` | `layers/nnx_decoders.py` **第三张**分派表（前两张在 `decoders.py`） |
| 9 | `Hunyuan3MoELayer.__init__() missing 1 required positional argument: 'quant'` | 两条构造路径签名不一致：nnx decoder 不传 `quant`，linen 路径传 |

把全文散落的实例汇到一处，**同一个模式在这个项目里出现了 10 次**：

| # | 分叉点 | 判断依据 | 记在哪 | 漏了会怎样 |
|---|---|---|---|---|
| 1 | `moe.py` 路由：`model_name.startswith("deepseek3")` | 模型名前缀 | §1.5 | 权重里混进 bias，**不报错** |
| 2 | 路由 softmax 被上游默认值覆盖 | 模型名前缀 | §2.6 | 打分函数走成 softmax，**不报错** |
| 3 | `maxtext_utils.py` FLOP 公式（3 处） | `decoder_block` 白名单 | §3.9 | MFU **虚高约 5 倍**，不报错 |
| 4 | `configs/types.py` 的 pydantic `Literal` 白名单 | 模型名 | §八 bug #6 | 配置解析被拒 |
| 5 | validator 把 `routed_bias_update_rate` 锁死在 DEEPSEEK | `decoder_block` | §八 bug #7 | 配置解析被拒 |
| 6 | `layers/nnx_decoders.py` 第三张分派表 | `decoder_block` | §八 bug #8 | `Incorrect decoder_block name` |
| 7 | 两条构造路径的 `quant` 签名不一致 | 走哪条 decoder | §八 bug #9 | 缺参数 `TypeError` |
| 8 | `train.py` 无梯度 bias 更新路径写死 `DeepSeekMoeBlock_0` | 模块属性名 | §5.4 坑 3 / §八 bug #10 | 首步 `AttributeError` |
| 9 | 同上，第二处（同一函数的另一分支） | 同上 | 同上 | 同上 |
| 10 | `moe.py` SwiGLU 激活截断白名单 | `decoder_block` | §八「第 10 处」（见下） | 调参开了 `mlp_activations_limit` 后**静默走错分支** |

**只有第 10 处是在事前拦住的**，其余 9 处都是运行时才炸、或者根本不炸只是安静地错。

> MaxText 里几乎每个"这个模型该走哪条路"的判断，都是一张按家族名字写死的表。
> 加新模型不是改一处，是**把所有这类表找齐**。漏掉的那张不会报错说"你漏了"，
> 它会报一个看起来完全无关的错——`quant` 参数缺失、pydantic 校验失败、
> 或者干脆不报错，安静地跑出另一套语义（路由分支和 FLOP 公式就是这种）。
>
> 找齐的办法只有一个：`grep -rn "DecoderBlockType.DEEPSEEK"` 和
> `grep -rn 'startswith(("deepseek'`，**每一处都问一遍"Hy3 该不该在这里"**。

#### 第 10 处：「我 config 里没这个字段」也不是理由

`layers/moe.py` 的 `apply_ffn_activation()` 里有一个 SwiGLU 激活截断分支，
白名单原本只有 `DEEPSEEK` / `DEEPSEEK4`，而且**整个分支还被
`mlp_activations_limit > 0.0` 二次把关**：

```python
elif (
    self.config.decoder_block in (DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4)
    and self.config.mlp_activations_limit > 0.0
):
```

我第一反应是判它「不适用」，理由是 **Hy3 的 config 里根本没有
`mlp_activations_limit` 这个字段**，所以那个分支恒假、加不加都一样。
这个理由看起来是可验证的事实，其实站不住：

> **`mlp_activations_limit` 是一个调优旋钮，不是模型属性。**
> 「我现在没设这个值」跟「我们没开那个开关」是同一种理由 ——
> 它成立到有人调参把它设上的那一刻，然后模型**静默走进不截断的那条路**，
> 不报错、不告警，只是数值行为跟预期不一样。
>
> 可接受的「不适用」只有一种：**这个行为在我的模型上永远不成立**。
> 比如 vLLM 那张权重映射表 —— DeepSeek 是 MLA、Hy3 是 GQA，
> 映射结构根本不同，套用一定错。那不是「暂时不需要」，是「用了就是 bug」。

所以把 `HUNYUAN3` 也加进了那个白名单（分支第三个 commit）。
**这一处没有造成任何故障，是这 10 次里唯一一次在事前拦住的** ——
代价只是多想了一步「这个条件为什么现在是假的」。

### bug #10：三元组只接了两个，无梯度 bias 更新静默失效（2026-07-28 补记）

前面 8 个都是「漏了一张分派表」，第 9 个是**另一种模式**，而且更隐蔽。

新版 MaxText 的 MoE block 返回的是**三元组** `(output, lb_loss, bias_updates)`。
第三项就是 §2.8 提到的、上游后来补上的 DSV3 无梯度 bias 更新量。它的完整链路是三段：

| 段 | 位置 | 做什么 |
|---|---|---|
| 算 | `layers/moe.py: calculate_load_balance_updates()` | `sign(平均负载 − 本专家负载) × γ`，正是 [arXiv 2408.15664](https://arxiv.org/abs/2408.15664) 的式子 |
| 传 | 各模型自己的 layer，`sow(nnx.Intermediate, "moe_bias_updates", ...)` | **这一段每个模型必须自己接** |
| 用 | `trainers/pre_train/train.py` | 优化器走完之后，`gate.bias += 更新量`——走的是梯度之外的旁路 |

我们的 `hunyuan3.py` 当时写的是：

```python
mlp_lnx, load_balance_loss, _ = self.moe_block(hidden_states)   # ← 第三项丢了
```

于是「传」这一段断了。训练循环按名字去中间量里找 `moe_bias_updates`，找不到，
就当没开——**而 `hunyuan3-295b.yml` 里 `routed_bias_update_rate: 0.001` 明明写着**。
配置说开、代码说关，全程零报错零日志。

> **为什么特别值得记**：bug #7 恰恰就是「validator 把 `routed_bias_update_rate` 锁死在 DEEPSEEK」——
> 也就是说我当时**已经知道这个功能存在，还专门为它改了校验**，然后忘了把 layer 里的线接上。
> 知道一个功能存在，和把它接通，是两件独立的事。
>
> 顺带修了同一处的第二个问题：`moe_lb_loss` 原本写成 `self.moe_lb_loss = nnx.Intermediate(x)`
> 直接赋值。训练循环读的是 `value[0]`，**只有 `sow` 才产生那个可下标的元组**，
> 直接赋值同样取不到。两处一并改成 `self.sow(...)`。

**怎么自查**：任何返回多元组的上游模块，都去 `grep` 官方模型（这里是 `models/deepseek.py`）
是怎么接的，逐项比对——**不要用 `_` 丢弃自己没看懂的返回值**。

### bug #11 / #12：改对了，但只改在临时树里（2026-07-29 补记）

前十个都是「上游/框架有个坑」，第十一、十二个不一样 —— **坑是我们自己的产物挖的**。

| # | 症状 | 真相 |
|---|---|---|
| 11 | `Input should be 'default', ..., 'hunyuan3-295b'`，传 `hunyuan3-smoke` 被拒 | 补丁脚本只注册了 `hunyuan3-295b`。而**文档恰恰让人先跑冒烟配置** |
| 12 | `'Hunyuan3MoELayer' object has no attribute 'Hunyuan3MoeBlock_0'` | 模型文件里属性叫 `moe_block`，训练循环的查找表写的是 `Hunyuan3MoeBlock_0`，两边对不上 |

**两处在我本地那棵临时调试树里都是对的。** 当天全部 v5p benchmark 都跑在那棵树上，
所以从头到尾零征兆 —— 直到从全新 clone 只跑脚本、再真跑一次，才同时暴露。

> **这两个 bug 是「删掉补丁脚本、改用分支」这个决定的直接理由**（§10.2）。
> 根因不是手滑，是**同一份改动存在于两个地方**：临时树一份、仓库产物一份。
> 只要存在两份，就一定会在某次修改后分叉，而且分叉的那一刻不会有任何报错。
>
> 配套纪律：**判定产物是否可用的唯一有效方法，是从全新 checkout 开始、
> 不做任何手工修补、然后真跑一次。** 「代码看着对」不算。

`verify_hunyuan3.py` 的 5 项检查全部通过，但**这 12 个实跑 bug 一个都没拦住**：

| bug | 层次 | 静态检查为什么看不见 |
|---|---|---|
| 1 pod 被 admission webhook 拒 | K8s 调度 | 不看集群 |
| 2 模型名不在白名单 | 框架注册 | 白名单是硬编码常量，不在被检查的代码路径上 |
| 3 `base_moe_mlp_dim=192` 不是 128 倍数 | 硬件约束 | 参数量算得出来，MXU tile 对不对齐算不出来 |
| 4 `fsdp_shard_on_exp`（新栈叫 `shard_exp_on_fsdp`）与 EP 互斥 | 配置组合 | 单项都合法，组合才非法 |
| 5 FLOP 公式漏 hunyuan3 | 报表 | 不影响任何被检查的对象 |
| 6 pydantic `Literal` 白名单 | 框架注册 | 同 2，换了实现方式 |
| 7 validator 锁死 `routed_bias_update_rate` | 配置校验 | 校验逻辑本身不在被检查的范围里 |
| 8 `nnx_decoders.py` 第三张分派表 | 框架注册 | 检查只看了 `decoders.py` 的两张 |
| 9 两条构造路径签名不一致 | 框架接口 | 需要真正实例化才暴露 |
| 10 三元组第三项被 `_` 丢弃 | 框架接口 | 语法完全合法，配置也合法，只有跑起来看 bias 有没有动才知道 |
| 11 白名单漏注册 `hunyuan3-smoke` | 产物自身 | 我们所有测试都跑在手工改过的临时树上，那棵树里是对的 |
| 12 模型属性名与训练循环查找表不一致 | 产物自身 | 同上；两处都只在「全新 checkout 只跑脚本」时才对不上 |

> 静态验证的价值是**证明逻辑对**（路由数学、参数量、分派结构），
> 这几项它确实抓到了两个真 bug（路由分支、`model_name` 门）。
> 但"能不能跑"是另一回事——**调度、注册、硬件对齐、接口签名，
> 只有真跑才知道。** 两类检查不可互相替代。

---


## 九、部署与测试流程（三步）

代码在 **[`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3)**，
基于上游 main，**这是代码的唯一真相**。
仓库里 [`maxtext-hunyuan3/`](maxtext-hunyuan3/) 只放跑测试用的两个脚本，不再放代码副本。

### 9.0 集群前置（换新项目 / 新集群时必做）

下面「三步」默认**集群、TPU 节点池、JobSet 都已就位**。
2026-07-29 从零搭一遍时，下面这些全都得先补，其中大部分是原文档没写的。

**① TPU 节点池（文档里一直只有规格，没有命令）**

```bash
# v5p：64 台 × 4 芯片 = 256 芯片
gcloud container node-pools create np-v5p-256 \
  --cluster=CLUSTER --project=PROJECT --region=us-central1 \
  --node-locations=us-central1-a \
  --machine-type=ct5p-hightpu-4t --tpu-topology=4x8x8 \
  --num-nodes=64 --spot --scopes=cloud-platform
```

冒烟另建一个 1 节点小池（4 芯片），改一行代码几十秒就能验一轮：

```bash
gcloud container node-pools create np-v5p-dev \
  --cluster=CLUSTER --project=PROJECT --region=us-central1 \
  --node-locations=us-central1-a \
  --machine-type=ct5p-hightpu-4t --tpu-topology=2x2x1 \
  --num-nodes=1 --spot --scopes=cloud-platform
```

256 芯片那个池实测 8 分 26 秒建完，64 台全 Ready。`--num-nodes` = 芯片数 ÷ 4，
且必须与 `--tpu-topology` 相乘一致，否则直接被拒。
**v5p 在 us-central1 只有 `-a` 有货**，集群 region 是 us-central1 就行，
节点池自己指定 zone；子网是区域级的，不用为此改集群。

> Spot 配额查不到不等于没有。区域配额里只列 v5e 那组
> （`PREEMPTIBLE_TPU_LITE_PODSLICE_V5`），v5p 不走这些老 metric，
> Cloud Quotas API 也返回空。**只能试**，报错会直接说是配额还是容量。

**② JobSet CRD —— 新集群没有，`run.sh` 会静默失败**

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/download/v0.11.1/manifests.yaml
kubectl wait --for=condition=Available deploy/jobset-controller-manager \
  -n jobset-system --timeout=180s
```

v0.11.1 自带证书，**不需要 cert-manager**。不装的话 `run.sh` 里的 `kubectl apply` 找不到 `jobset.x-k8s.io/v1alpha2`，
`set -e` 直接退出、什么都没起来。**当时那版脚本把这条错误也一起吞了**，
看上去像什么都没发生 —— 现在只吞 stdout，stderr 会打出来，但**装 JobSet 这一步仍然不能省**。

**③ 暂存桶 + 两处跨项目授权**

```bash
NODE_SA=<集群项目号>-compute@developer.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://YOUR-STAGE-BUCKET \
  --member="serviceAccount:$NODE_SA" --role=roles/storage.objectViewer
# 镜像在别的项目时，节点 SA 还要能拉
gcloud artifacts repositories add-iam-policy-binding gcr.io --location=us \
  --project=IMAGE_PROJECT --member="serviceAccount:$NODE_SA" \
  --role=roles/artifactregistry.reader
```

**④ 新项目里 default VPC 大概率建不出集群**

共享项目的 default VPC 是 auto 模式，`10.128.0.0/9` 全被各 region 子网占着，
`10.0.0.0/9` 又被几十个集群切碎，GKE 自动分配凑不出一整块 `/14`：

```
The network "default" does not have available private IP space in
10.0.0.0/9 to reserve a /14 block for pods
```

指定 `--cluster-ipv4-cidr` 也常撞（我第一次挑的 `10.160.0.0/16` 正好落在
auto 子网段里）。**直接建自己的 custom VPC 最省事**，顺带能把 MTU 开到 8896：

```bash
gcloud compute networks create NAME-vpc --subnet-mode=custom --mtu=8896
gcloud compute networks subnets create NAME-uc1 --network=NAME-vpc \
  --region=us-central1 --range=10.124.0.0/22 \
  --secondary-range=pods=10.125.0.0/16,services=10.124.16.0/20 \
  --enable-private-ip-google-access
```

三段全压进 `10.124.0.0/15`，将来做 VPC peering 只需告诉对方避开这一段。
选网段前先跑 `gcloud network-connectivity internal-ranges list` 拿权威占用表，
**不要只看 `clusters list` 的 `clusterIpv4Cidr`**——那漏掉子网自身占的地址。

**⑤ v7x（tpu7x）拿机器：跟 v5p 完全不是一回事**

v5p 直接 `--tpu-topology` + `--spot` 就行（§9.0 ①）。tpu7x 会被拒：

```
Creation of a managed instance group with tpu7x-standard-4t machine type
with placement policy is not supported. Use workload policy instead.
```

**必须先建一个 `workloadPolicy` 类型的 resource policy，再让节点池引用它。**
**2026-07-31 复核：gcloud 577.0.0 上 `gcloud compute resource-policies create
workload-policy` 存在且可用**，优先用它；下面的 REST 写法只在 gcloud 缺子命令时才需要：

```bash
TOK=$(gcloud auth application-default print-access-token)
curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
 "https://compute.googleapis.com/compute/v1/projects/$P/regions/us-central1/resourcePolicies" \
 -d '{"name":"NAME-wp-2x2x4","workloadPolicy":{"type":"HIGH_THROUGHPUT","acceleratorTopology":"2x2x4"}}'
# 然后节点池同时给 --tpu-topology 和 --placement-policy=NAME-wp-2x2x4
```

**DWS flex-start 的建池三要素**（缺一个就报错，而且报错各说一半）：

```bash
gcloud beta container node-pools create POOL ... \
  --tpu-topology=2x2x4 --placement-policy=NAME-wp-2x2x4 \
  --flex-start --num-nodes=0 \
  --enable-autoscaling --min-nodes=0 --max-nodes=4 --location-policy=ANY \
  --reservation-affinity=none
```

- 少 `--enable-autoscaling` → `Flex start node pools require autoscaling enabled.`
- `--num-nodes` 非 0 → `Flex start node pools require initial node count to be set to 0.`
- 上限用 `--total-max-nodes` 而不是 `--max-nodes` → 被判成 0，报
  `Maximum node count 0 is not a valid size of TPU pod slice with topology "2x2x4"`

**⑥ 空池（DWS）跑 JobSet 必须用 `NODEPOOL=`，否则 4 个 pod 只出 1 个**

`run.sh` 默认给 JobSet 打 `alpha.jobset.sigs.k8s.io/exclusive-topology` 注解。
该机制要求**先把 leader pod 调度上去**，follower 才能抄它的 `gke-nodepool` 选择器。
节点池是 0 节点时 leader 永远落不了地，webhook 就会拒建 follower：

```
admission webhook "vpod.kb.io" denied the request:
follower pod node selector for topology domain not found.
missing selector: cloud.google.com/gke-nodepool
```

后果很隐蔽：**只创建出 1 个 pod**，autoscaler 也只看得见 1 个 pending，
判断依据是残缺的。解法是把节点池写死进 `nodeSelector`，不依赖 leader 先行：

```bash
NODEPOOL=np-v7x-flex PLATFORM=v7 NODES=4 TOPO=2x2x4 bash run.sh myrun
```

设了 `NODEPOOL` 就走这条路径（去掉注解 + 加 `gke-nodepool` 选择器），
不设保持原样，v5p 固定节点池的流程不受影响。
**验证方法**：提交后数 pod 个数，应等于 `NODES`。

**⑦ 容量：2026-07-29/30 的实测时间线（us-central1-c，tpu7x）**

| 时间 (HKT) | 现象 |
|---|---|
| 07-29 21:57 | spot 8 台（32 芯片）**15 秒起满** |
| 07-29 23:18 | 切片**只活 2-3 分钟**就被抢，反复三轮 |
| 07-30 01:10 起 | 连 4 台都拿不到，6 轮探测全 0 |
| 07-30 07:35 | 别人的 FLEX_START 从 PENDING 转 RUNNING（局部恢复） |
| 07-30 08:17 | 我们 spot 8 台 / 4 台仍全 0 |
| 07-30 09:0x–09:5x | DWS flex-start 两次 `FailedScaleUp: Internal error`，45 分钟 0 台 |
| 07-30 10:05 | **换到另一个项目，同 zone 同机型同 spot，一样 0 台** |

> **最后一行是关键对照。** 两个项目、两套配额、两个集群，同一 zone 同时拿不到
> —— 说明这是 **zone 级物理容量**问题，**换项目 / 提配额都无效**。
> 配额和容量是两个独立的闸门：配额决定你**能不能申请**，容量决定你**能不能拿到**。
>
> **不是"没有货"，是"货全被占了"。** 同一时刻查那个项目：
> ```
> gcloud compute instances list --project=$OTHER_PROJECT \
>   --filter="machineType~tpu7x AND status=RUNNING" \
>   --format='value(zone,scheduling.provisioningModel)' | sort | uniq -c
>   152 us-central1-c  SPOT
>     1 us-central1-c  FLEX_START
> ```
> **152 台 spot（608 芯片）正在别人手里跑。** 抢占式没有排队，先到先得，
> 所以队列前面站满人时，新请求既拿不到也不会排上——只会一直 PROVISIONING 到超时。
> **判断"要不要继续等"就看这个数**，比反复建池探测便宜得多。
>
> 顺带：`PREEMPTIBLE-TPU7X-per-project-region` 在我们那个项目的上限是 **64 芯片**，
> 且是**全项目共享**。别人占 8 芯片时，4x4x4（64 芯片）这种原子切片就永远排不下
> ——TPU 拓扑没有 48/56 这种中间档，拿不满等于拿不到。自助提额到 128 提交后
> 12 小时仍是 `reconciling`，没批。

### 9.1 三步

```bash
cd maxtext-hunyuan3/                     # 两个脚本都在这个目录里
gcloud container clusters get-credentials CLUSTER --region=REGION --project=PROJECT
gcloud storage buckets create gs://your-bucket --location=US   # 已有就跳过

export GCS_STAGE=gs://your-bucket/hy3
export IMAGE=us-docker.pkg.dev/YOUR-PROJECT/gcr.io/YOUR-maxtext-latest:runner

# ① 准备代码（只有改了代码才要重跑；换 flag / 换参数不用）
bash prep.sh                    # clone hunyuan3 分支 → 8 项自检 → 打包传 GCS

# ② 起训练
PLATFORM=v5p bash run.sh myrun          # 或 PLATFORM=v7

# 4 层冒烟：必须显式缩规模，否则 run.sh 默认按 64 台 / 256 芯片起
NODES=1 TOPO=2x2x1 PLATFORM=v5p MODEL=hunyuan3-smoke STEPS=8 \
  bash run.sh smoke per_device_batch_size=1 max_target_length=2048

# ③ 看结果
kubectl logs -f job/hy3-myrun-slice-job-0 -c jax-tpu
```

### 9.2 三步各自在做什么

| 步 | 动作 | 为什么这样做 |
|---|---|---|
| ① `prep.sh` | clone 分支 → **8 项自检** → `tar src/maxtext` → 传 GCS | 自检挡的是「分支自己少东西」：三个文件在不在、白名单两个模型名全不全、枚举有没有 `HUNYUAN3`、`train.py` 补丁在不在、**`Hunyuan3MoeBlock_0` 在 model 和 train 两边对不对得上**。最后一项 2026-07-29 真踩过，不查的话要到 TPU 上跑起来才炸 |
| ② `run.sh` | 提交 JobSet，pod 里 **`rm -rf /deps/src/maxtext` 再解包** | **整棵覆盖，不是只注入改动文件**。只注入的话测的是「我的改动 + 容器里的旧基座」，不是分支本身——2026-07-28 夜那两个 bug 就是这么漏掉的 |
| ③ 读日志 | — | 见 §9.3 |

镜像两个平台共用，容器只提供 jax / libtpu / 依赖，MaxText 整个来自分支。

### 9.3 读结果前必看

1. **先确认 `N/N Running` 再看日志。** TPU 切片全有全无，人不齐时活着的 pod 会报
   `GetSliceInfo can only be invoked after a slice is built` —— 那是**症状不是病因**。
2. **判错看最早那条，不是日志尾。** 配置非法会**先起 TPU 再退**，真正的报错是
   `MAXTEXT CONFIG ERROR` / pydantic 的 `Value error`，位置在日志上方。
3. **step 0 含编译，step 1/2 是 JAX 异步派发的假读数**，稳态取 step ≥ 3。
4. 单位换算：v5p 1 device = 1 chip，MFU = TFLOP/s/device ÷ 459；
   v7 **2 device/chip**，per-chip = 日志值 × 2，MFU = per-chip ÷ 2307。

### 9.4 预期水位

| 平台 | 规模 | step | TFLOP/s（口径） | MFU |
|---|---|---|---|---|
| v5p | 256 chips | 63.2 s | 161.0 **per device**（= per chip） | 35.07% |
| v7 | 64 chips | 20.4 s | 445.1 **per chip**（日志值 ×2） | 19.29% |
| v5p smoke | 4 chips | 0.82 s | — | loss 13.45→10.35 / 8 步 |

---

## 十、代码放在哪、怎么跟上游

### 10.1 单一真相：fork 的分支

代码在 **[`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3)**，
基于上游 main。所有关于 Hy3 的改动 —— 功能、性能、bug 修复 —— **都以 commit 落在这个分支上**。

| 事情 | 怎么做 |
|---|---|
| 改代码 | 在分支上提 commit |
| 跟上游 | `git rebase upstream/main` |
| 跑测试 | `prep.sh` 从分支拉（§九） |
| 回馈上游 | 从分支上挑 commit 提 PR（见 10.3） |

### 10.2 为什么不再维护补丁脚本

早期这里放过一个 `port.py`：用字符串锚点去改上游文件，每处断言「正好命中 N 处」。
断言本身抓到过真问题（上游把单行 tuple 拆成多行导致静默漏改），**但整条路线是错的**：

| | 补丁脚本 | 分支 + rebase |
|---|---|---|
| 上游改了附近代码 | 锚点漂了，靠你**事先写对**断言才发现 | **明确报冲突**，必须处理 |
| 真相来源 | 脚本一份、分支一份 → **会分叉** | 只有分支一份 |
| 提 PR | 还要再转换一次 | 直接就是 commit |

> **两个真相来源真的咬过人。** 2026-07-28 夜同时维护「仓库里的模型文件 + 补丁脚本」和「分支」，
> 结果在一处改对、另一处没跟上，而所有 benchmark 恰好跑在改对的那份上——
> **整晚零征兆**，直到从全新 checkout 走一遍才暴露（§八 bug #11/#12）。

### 10.3 回馈上游：建议拆两个 PR

| PR | 内容 | 理由 |
|---|---|---|
| **①（先发）** | `trainers/pre_train/train.py`：把写死的 `DeepSeekMoeBlock_0` 改成按 `decoder_block` 解析 | **纯 bug 修复，跟 Hy3 无关**。任何非 DeepSeek 模型只要开 aux-loss-free 均衡都会撞（§八 bug #10）。3 行 + 一张查找表，独立成立，不需要先接受 Hy3 |
| **②（跟着发）** | 模型本体 + 其余 11 个文件的行为归类 | 就是「加一个模型」，跟 deepseek / qwen3 同级 |

分支上是三个 commit，**已经按这个边界拆好了**，可以直接 cherry-pick：

| commit | 归属 |
|---|---|
| `Resolve the loss-free-balancing bias path per decoder block` | PR ① |
| `Add Tencent Hunyuan 3 (295B-A21B)` | PR ② |
| `Let Hunyuan3 use the SwiGLU activation bound too` | PR ②（§八 台账第 10 处） |

### 10.4 发 PR 前要确认的三件事

1. **雇主 IP 归属**：`AI-Hypercomputer/maxtext` 是 Google 的 repo，走内部贡献流程还是个人账号提 PR，先跟内部确认。
2. **`initializer_range: 0.006` 仍缺**（§2.8 第 ③ 项）。PR ② 最好一并补上，否则 from-scratch 预训练的初始化对不齐官方。
3. **FLOP 公式那处改动要单独说明**：它只影响 MFU 报表不影响训练，但会让 Hy3 的 `Total TFLOPs` 从 2800.97 变成 561.92（§3.9）。review 的人看到数字大跳需要这个上下文。

---

## 十一、待验证清单

按依赖顺序排，✅ 是本轮闭环的，⬜ 是还没做的。

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| 0 | 写出 `hunyuan3` block 并通过静态自检 | ✅ | 5 项检查全过；但**实跑的 12 个 bug 一个都没抓到**，见下方复盘 |
| 1 | 小规模真实前向 | ✅ | 4 芯片 v5p，r1–r20 共 20 轮，见 §三 |
| 2 | 192 experts × 80 层能否编译 | ✅ | v5p 256 芯片和 v7 64 芯片都编译并跑出稳态 |
| 3 | 参数量与 SSOT 对齐 | ✅ | 框架报 298.786 B = SSOT 294.9 B + MTP 头 3.886 B，两个平台逐位一致 |
| 4 | `normalization_layer_epsilon` | ✅ | 曾填错 1.0e-6，HF 原文是 **1e-05**，已修 |
| 5 | router fp32 | ✅ | 旧版我加了 `moe_router_dtype`；**新版上游已有 `float32_gate_logits`** |
| 6 | 专家 bias 的无梯度更新 | ✅ | **新版上游已有 `routed_bias_update_rate`**（旧版确实没有）。我们的 layer 一度把 `bias_updates` 丢了导致静默失效，2026-07-28 修好，见 §八 bug #10 |
| 7 | EP / FSDP 最优配比 | ✅ | 结论与预期相反：**TPU 上不用 EP**，见 §4.2 |
| 8 | `attention=flash` 能否编译 | ✅ | v5p / v7 都通过；v7 上还能用 `use_tokamax_splash` |
| 9 | MTP 开销 | ✅ | 全程 `mtp_num_layers=1`，`mtp_loss` 单独打出，加了 3.886 B 参数 |
| 10 | v7 调优到 MoE 合理水位 | 🔄 进行中 | 已调到 445.1 TFLOP/s/chip / 19.29%（c1），目标 26.6% 见 §6.1 |
| 11 | SFT 路线：冻结 `gate.bias` | ⬜ | 上游有了更新规则，但 SFT 是要**冻结**它。本版无 `trainable_parameters_mask` |
| 12 | HF 权重 → MaxText Orbax 转换 | ⬜ | 只做吞吐基线可以不碰；要 SFT 必须做 |
| 13 | 真实数据集上的收敛验证 | ⬜ | 目前全是 synthetic，只证明"能算且不发散"，没证明"学得对" |
| 14 | `initializer_range` | ⬜ | from-scratch 才需要；加载权重或 SFT 不受影响。**发上游 PR 时一并补**（§10.4） |
| 15 | 上游 PR ①：train.py 的 bias 路径解耦 | ⬜ | 纯 bug 修复，独立成立，见 §10.3 |
| 16 | 上游 PR ②：hunyuan3 模型本体 + 6 处注册 | ⬜ | 见 §10.3；发之前先确认雇主 IP 归属（§10.4） |
| 17 | 分支从全新 clone 能跑 | ✅ | 2026-07-29 实测：clone 分支 → 整棵覆盖容器 → v5p 4 芯片 loss 13.45→10.35，且**补完 5 处休眠改动前后逐位相同** |
| 18 | vLLM 权重映射表（tunix） | ⬜ | Hy3 是 GQA，不能套 DeepSeek 的 MLA 映射，需单独写一份 |
| 19 | `scan(unroll=N)` 分组扫点 | ⬜ | 让 XLA 跨层重叠 MoE 通信。MaxText 未实现，改动约 10 行。机理与实验设计见 [移植指南 §5.2.1](MAXTEXT-PORTING-GUIDE.md)。**主线跑稳后再做** |
| 20 | v5p 上试 `mu_dtype=bfloat16 grad_dtype=bfloat16` | ⬜ | 只在 `run.sh` 的 v7 分支开着，v5p 从没试过。FSDP=256 下 16→12 B/param，静态 18.7→14.0 GB，**腾出 4.7 GB/chip**。我们现在 96.9 / 95.74 G 贴顶跑，这是白捡的余量。`nu_dtype` optax 不支持，恒随 `weight_dtype`（fp32），主权重也不动，是三份状态里最温和的一个 |
| 21 | 补一条完整 loss 曲线 | ⬜ | 07-29 两轮 256 芯片验证最多只跑到 step 9（§7.5 自查）。跑到 step 30+ 记全，作为"没发散"的证据。注意仍是 synthetic，不等于收敛（见 #13） |


## 十二、参考

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

**两代硬件都已跑通；v5p 调优收敛且异地复现通过，v7 调优进行中**（2026-07-30）。

| 平台 | 规模 | 状态 | 成绩 | 对标 |
|---|---|---|---|---|
| v5p | 4 chips（dev pool） | ✅ 20 轮迭代闭环 | — | 用于快速验证代码路径 |
| v5p | 256 chips | ✅ **当前可复现水位** | **160.98 TFLOP/s/chip · MFU 35.07%** | 超过 GB300 的 31.6% |
| v5p | 256 chips | 📜 历史最好（旧栈，产物已删） | 168.6 TFLOP/s/chip · MFU 36.72% · 281,488 tok/s | 15 倍调优路径的终点，见 §7.3 战线一 |
| v7 Ironwood | 64 chips | 🔄 13 轮，仍在调 | **445.1 TFLOP/s/chip · MFU 19.29% · 205,314 tok/s** | 目标 612.7 / 26.6%（DSV3 实测水位） |

> 07-29 晚在全新项目 / 全新 VPC / 全新集群上从零复现，得 160.98 / 35.07%，
> 对既有水位 **+0.05%**（§7.3 战线三 b0″）—— 这份文档是真能照着跑出来的。

**下一步**（按优先级）：

1. **抓 xplane trace** —— 参数扫到 445 之后收益明显变平（杂项 flag 组 ±0），
   继续盲扫期望很低。需要 trace 直接回答"时间花在路由 / 分组重排 /
   all-to-all / offload 还是 GEMM"。前一次开 profiler 的那轮因为
   profiler 自身开销没在窗口内跑出稳态，要单独给它更长的预算。
2. **查清 `use_tokamax_gmm` 的死锁根因**（§6.7）——
   官方 DSV3 用它且能跑，怀疑是 192 专家（非 2 的幂）导致分组矩阵乘的
   组划分出问题。这是唯一一个"官方有、我们用不了"的加速手段。
3. 权重转换（HF → Orbax）与真实数据集收敛验证（§十一 清单 #12/#13）。

代码：[`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3)（唯一真相，见 §10.1）；
跑测试的两个脚本在 [`maxtext-hunyuan3/`](maxtext-hunyuan3/)。
