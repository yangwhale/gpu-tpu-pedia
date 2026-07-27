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

**MaxText 原生不支持混元 3——但所需组件全都在，已用它们拼出
`decoder_block: "hunyuan3"`，代码在 [`maxtext-hunyuan3/`](maxtext-hunyuan3/)。**

| | |
|---|---|
| 新增代码 | 一个文件 176 行（`hunyuan3.py`），只做接线，不重写 attention 或 MoE |
| 改动 | `common_types.py` +1 行、`decoders.py` 6 处（见 patch） |
| 参数量自检 | **294.97 B**，对 SSOT 的 294.9 B 差 **0.02%**；激活 20.6 B 对上官方 A21B |
| 静态验证 | enum / 类 / 分派 / 归一化 5 项全过（`verify_hunyuan3.py`） |
| 未做 | 真实前向、多卡分片、权重转换（见[待验证清单](#六待验证清单)） |

下面先讲为什么必须新写一个 block，再讲怎么拼的。

### 为什么两个现成 block 都不行

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

### 曾经考虑过的三条路

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

## 一之二、实现：用现成组件拼出 hunyuan3 block

代码在 [`maxtext-hunyuan3/`](maxtext-hunyuan3/)：

| 文件 | 作用 |
|---|---|
| `hunyuan3.py` | 两个 decoder layer 类，**只做接线** |
| `hunyuan3-295b.yml` | model config，值全部来自 SSOT |
| `register-hunyuan3.patch` | `common_types.py` + `decoders.py` 的注册改动 |
| `verify_hunyuan3.py` | 静态自检脚本 |

### 复用了什么，新写了什么

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

### 注册处的三个要点

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

### 已验证 / 未验证

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

### 对照 HF config 原文的审计（2026-07-27）

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

#### 当前对标程度

**config 层面 22 项逐项比对，0 不一致**（层数 / 维度 / 头数 / GQA / head_dim /
vocab / eps / 专家数 / top-k / moe 维度 / shared / dense 首层 / scaling /
MTP / max_pos / sigmoid / expert bias / qk_norm / untied / rope theta / rope type）。

**但仍不是 100% 一模一样**，剩下三项无法靠配置消除：

"配置消不掉"的意思是：**MaxText 根本没有暴露对应的配置项，改 yml 解决不了，
只能改 MaxText 源码或者接受差异。** 三项性质完全不同：

#### ① `initializer_range: 0.006` — 参数没暴露

HF 要求所有权重按 std=0.006 的截断正态初始化。MaxText 把初始化写死成
`nd_dense_init(1.0, "fan_in", "truncated_normal")` —— 这是 **fan-in 缩放**
（std 随输入维度自动变），不是固定 0.006，且 yml 里没有任何开关能改。

**影响面**：只在 from-scratch 时决定初始权重分布，进而影响收敛曲线。
加载预训练权重则完全无关。跑吞吐基线不受影响。

#### ② expert bias 的更新机制 — **不是参数缺失，是机制没实现**

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
> **吞吐会被拖累**（GB300 文档 §3 就把"热门专家扎堆"列为毁掉高速互连的两个坑之一）。
> 只是短期 synthetic + 随机初始化的路由接近均匀，几十步内看不出来。
> **拿短期基线是安全的；长时间真实训练不安全。**

#### ③ `enable_moe_fp32_combine: false` — 已确认一致

官方 `HYV3MoE.forward()`：该开关为 `false` 时走
`hidden = routed_output + self.shared_experts(hidden_states)`，直接相加不转 fp32。
MaxText `RoutedAndSharedMoE.__call__()` 就是 `routed_experts + shared_experts`。
**行为相同**，此项结案。

---

## 一之三、对照官方 modeling 代码的深度审计（2026-07-27）

上一轮只比了 config 数值。这一轮拉了 **transformers 主干里的官方实现**
（`models/hy_v3/modeling_hy_v3.py` 608 行 + `modular_hy_v3.py` 322 行，
Hy3 已进主干，不是 remote code）逐个组件对照算法。

### 官方的真实血统

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

### 逐组件对照

| 组件 | 官方实现 | MaxText 侧 | 状态 |
|---|---|---|---|
| Attention | GQA + qk RMSNorm(head_dim) 在 RoPE 前 + scaling head_dim^-0.5 + 无 bias | `qwen3.self_attention_with_norm` | ✅ 等价 |
| 层布局 | `mlp_layer_types = ["dense"]*1 + ["sparse"]*79` | `first_num_dense_layers: 1` | ✅ 等价 |
| shared expert | `routed + shared`，无额外 gating | `RoutedAndSharedMoE` 同式 | ✅ 等价 |
| 路由数学 | 见下 | 见下 | ✅ **本轮修好** |
| router 精度 | **fp32 强制** | 跟 `cfg.dtype`（bf16） | ❌ **差异** |
| expert bias 更新 | buffer + 训练框架更新 | 可学习 Parameter | ❌ **差异** |
| 初始化 | std=0.006 | fan-in 缩放 | ❌ 差异 |

### 路由数学：官方原文与本轮修复

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
`model_name.startswith("deepseek3")` 挡住了——共 4 处**（`moe.py` L246 / L413 /
L889 / L1427）。`hunyuan3-295b` 不匹配，于是：

- L246 不保存 `pre_bias_logits`（为 None）
- L413 退回 `jax.lax.top_k(gate_logits)` —— **gate_logits 是加过 bias 的**，
  于是权重里混进了 bias，第 4 步的语义丢失
- L889 / L1427 的 sharding 约束也随之跳过

**形状全对、不报任何错、参数量不变**——只有权重数值是错的。
四处均已改为 `startswith(("deepseek3", "hunyuan3"))`，
并加了验证脚本第 7 项（负向测试过：还原任意一处即报错）。

> **这是 MaxText 的一处设计债**：用模型名字符串前缀决定路由算法。
> 更健壮的做法是提一个 config flag（如 `use_pre_bias_routing_weights`）。
> 这里选择最小侵入，是为了不改动 deepseek2 / kimi-k2 的现有行为
> ——它们的 `decoder_block` 同为 `deepseek`，但 model_name 不以 deepseek3 开头，
> 改判断依据会波及它们。

### 生产级落地还差三件事

按优先级：

**① 给 expert bias 补负载驱动更新**（阻塞真实预训练）

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

| # | 事项 | 状态 | 为什么关键 |
|---|---|---|---|
| 0 | 写出 `hunyuan3` block 并通过静态自检 | ✅ 已完成 | 见[实现](#一之二实现用现成组件拼出-hunyuan3-block) |
| 1 | **小规模真实前向**（如 4 层 / 8 experts / CPU 或 v7 2x2x2） | ⬜ | 静态验证只证明了接线，没证明能算。这是下一步 |
| 2 | 192 experts × 80 层能否在 128 devices 上编译出来 | ⬜ | DSV3 671B 在 v7 上曾出现 sparse matmul 编译 6 小时未完成 |
| 3 | **给 MaxText 补上 expert bias 的负载驱动更新** | ⬜ **最高优先** | 已查实 MaxText 只把 `routed_bias` 当普通可学习 bias，没有任何负载统计代码，DSV3 式 aux-loss-free 均衡**未实现**。要跑真实预训练必须补，否则专家会失衡（连带拖累 all-to-all 吞吐） |
| 4 | ~~`normalization_layer_epsilon` 与 HF `rms_norm_eps` 是否一致~~ | ✅ 已核 | 曾填错（1.0e-6），HF 原文是 **1e-05**，已修 |
| 4b | **给 MaxText 加 router fp32 开关** | ⬜ 高 | 官方 `F.linear(h.float(), w.float())` 强制 fp32，Megatron 也设 `moe_router_dtype: fp32`。MaxText 的 GateLogit 跟 `cfg.dtype`（bf16），192 专家的 sigmoid 打分精度不足 |
| 5 | EP / FSDP 最优配比 | ⬜ | 97% 参数在专家里，这是第一性能旋钮 |
| 6 | `attention=flash` 在 v7 上能否编译通过 | ⬜ | DSV3 上踩过坑（踩坑 #3，70+ 分钟未完成），需确认 GQA 是否同样受影响 |
| 7 | MTP 开启后的开销 | ⬜ | GB300 侧建议首跑设 0，跑通再开 |
| 8 | HF 权重 → MaxText Orbax 的转换 | ⬜ | 只做 from-scratch 基线可以先不碰；要 SFT 就必须做 |

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
