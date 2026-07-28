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

**MaxText 原生不支持混元 3。所需组件全在，已拼出 `decoder_block: "hunyuan3"`，
并在 v5p (256 芯片) 和 v7 Ironwood (64 芯片) 上跑通 80 层完整 295B。**

| | v5p 256 chips | v7 Ironwood 64 chips |
|---|---|---|
| 状态 | 已跑通并调优 | 已跑通，调优进行中 |
| 参数量（框架报） | 298.786 B | 298.786 B（逐位一致） |
| 稳态 step | 47.67 s | 25.11 s |
| **TFLOP/s / chip** | **168.6** | **404.8** |
| **MFU** | **36.72%** | 17.55%（首跑，2 个 XLA flag） |
| 整机 tok/s | 281,488 | 167,059 |
| loss | 单调下降 | 单调下降 |

对照 GB300 64 GPU 实测 **854.0 TFLOP/s/GPU、MFU 31.6%**：
**v5p 的 36.72% 已经超过 GB300 的 MFU**，v7 首跑未调优。

三件事最值得先看：

1. **[MFU 从 2.45% 到 36.72% 的 12.9 倍，来自"别自己攒配置"](#521-我犯的错没有从官方配方出发)** ——
   照抄官方 DeepSeek3 v5p 配方，只换模型名。
2. **[TPU 上专家并行是负优化](#522-为什么-tpu-上-fsdp-打得过-ep)** ——
   跟 GPU 结论相反，EP=64 不只是慢，是直接超显存 326 GB。
3. **[同一个 bug 模式在本项目出现 8 次](#512-第五到第八次踩同一个坑)** ——
   MaxText 里每个"按模型家族名字列举"的分支都要单独补，漏了不报错。

| | |
|---|---|
| 新增代码 | `hunyuan3.py`（v5p linen 版 176 行 / v7 nnx 版 159 行），只做接线 |
| 改动 | v5p 侧 6 个文件；v7 侧 6 个文件（上游重构后路径全变，见 §5.1.1） |
| 静态验证 | 8 项全过（`verify_hunyuan3.py`），但**一个实跑 bug 都没抓到** |
| 未做 | 权重转换、真实数据集收敛验证（见[待验证清单](#六待验证清单)） |

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

### Hy3 的 MoE 与 DeepSeek V3 并非完全相同

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

#### 但这一项的优先级取决于走哪条路线

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

## 一之四、实跑：在真实 v5p 上跑通（2026-07-28 凌晨）

前面几节全是静态验证。这一节是**真的把它跑起来**的记录。

### 结论：跑通了，三轮修掉三个 bug

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

### 环境

| 项 | 值 |
|---|---|
| 节点池 | `np-v5p-hy3-dev`，1 台 `ct5p-hightpu-4t`，拓扑 `2x2x1`，spot |
| device | 4（v5p 是 MegaCore，4 chips = 4 devices） |
| 镜像 | `chrisya-maxtext-stable:oct`（MaxText `3eb77db3` + JAX 0.7.0） |
| 代码注入 | 本地改动 tar 后 `kubectl cp` 进 pod，解到 `/deps` 覆盖 |

**为什么先建 1 节点小池**：64 节点跑一次要等 6–7 分钟编译，改一行代码就得重来。
4 chips 上跑小模型只要几十秒，修 bug 的迭代速度差一个量级。

冒烟用的 `hunyuan3-smoke.yml` **结构与 295B 完全一致**（dense 首层 + MoE 层、
sigmoid + bias、shared expert、fp32 router），只把维度缩小——
目的是走遍每条代码路径，不是测性能。

### 三个 bug

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

### 迭代方法

```bash
bash /tmp/hy3-iter.sh <round>   # 打包改动 -> cp 进 pod -> 跑 -> 判定
```

每轮自动：tar 本地改动 → `kubectl cp` → 解包覆盖 → 跑训练 →
grep `completed step` 判定成功，失败则打印首个错误。
单轮约 40 秒，这是能连续迭代十几轮的前提。

### 渐进放大扫描 r4–r10

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

### 逐项换成真实值 r11–r16

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
[DSV3 文档踩坑 #3](https://github.com/yangwhale/tpu-recipes) 里 v7 的那个编译问题。

**r16 是唯一一个真正说明性能的数字。** 序列 512 → 2048，
TFLOP/s/device 从 74.87 涨到 145.96（+95%）——序列变长把固定开销摊薄了。
145.96 / 459 = **31.8% MFU**，跟 GB300 上 31.6% 处在同一水平。
但这只是 4 层小模型 4 芯片，**不能当基线**，真实基线见 §五。

---

### 一个必须先说清的命名坑

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
> 下文 §四、§五的表格已按**实际 256 芯片**修正。

---

## 一之五、80 层完整 295B 在 256 芯片上跑通

### 怎么把补丁发到 64 台机器

单节点靠 `kubectl cp` 就行，64 个 pod 不能这么干。改成 **GCS 中转**：

```bash
tar czf hy3inject.tgz <改过的 8 个文件>
gsutil cp hy3inject.tgz gs://.../hy3/hy3inject.tgz
# 每个 pod 启动时：
gsutil -q cp gs://.../hy3/hy3inject.tgz /tmp/p.tgz && cd /deps && tar xzf /tmp/p.tgz
```

比重新 build 镜像快得多（补丁 55 KB，改一行到重跑只要几十秒），
也不用把私有改动烤进镜像。JobSet 用 `parallelism: 64 / completions: 64`
配 `exclusive-topology: gke-nodepool` 注解，拿整个 4x8x8 切片。

### 首跑结果

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

### 读这段日志要避开的三个坑

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

### bug #4：`fsdp_shard_on_exp` 和 EP 互斥

```
ValueError: fsdp_shard_on_exp requires ici_expert_parallelism = 1 and
            ici_tensor_parallelism/ici_tensor_transpose_parallelism = 1
```

这个开关是给**不用 EP** 的场景准备的（把专家维切到 FSDP 轴上）。
既然已经开了 EP=64，它就是多余的。顺带记一笔：新版 MaxText 里这个参数
叫 `shard_exp_on_fsdp`，本仓这版叫 `fsdp_shard_on_exp`，**名字是反的**。

### bug #5：MFU 口径被虚高了约 5 倍

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

| 行 | 原本 | 问题 |
|---|---|---|
| 461 | MoE FFN 分支只列 DEEPSEEK / LLAMA4 | 专家宽度用错，漏 shared expert |
| 308 | `get_dense_moe_layers()` 只认 DEEPSEEK / LLAMA4，其余 `raise ValueError` | 拆不出 dense/MoE 层数 |
| 524 | 逐层汇总时 DEEPSEEK 单独一支 | helper 已按层累加，走 `else` 会再乘一次层数 |

> 这是本项目撞到的**第三个同类 bug**（前两个是 §一之三的路由分支和
> `model_name.startswith` 门）。模式完全一样：**MaxText 里凡是按模型家族
> 名字列举的分支，新模型都得逐个补进去；漏了不报错，只是安静地跑出
> 另一套语义。** 前两个改的是训练数学，这个改的只是报表——
> 但报表错了会让人拿着虚高 5 倍的 MFU 去做容量规划。

修完之后 §五的 MFU 才有意义。**§一之四、§一之五里所有 TFLOP/s 数字
都是修复前的口径，不要拿去对标 GB300**；对标数字见 §五。

验证：同一份代码、同一个 pod，只改这三处，`Total TFLOPs` 从
**2800.97 → 561.92**（4.985 倍），561.92 / 4096 token = **137.2 GFLOP/token**，
跟 GB300 侧的 136.8 GFLOP/token 对上了。step 时间和 loss 曲线**一字未变**——
确认改的只是报表。

### r17–r20：稳定性与路径对照（4 芯片）

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

实跑用的是节点池 `np-v5p-256`（4x8x8），**实际 256 芯片**——
不是 Google 命名法里的 `v5p-256`。命名坑见 §一之四末尾。

| | `np-v5p-256`（**实跑**） | v5p-256（Google 命名） | v7 4x4x4 | GB300（参考） |
|---|---|---|---|---|
| Google 加速器类型 | `v5p-512` | `v5p-256` | — | — |
| 芯片数 | **256** | 128 | 64 | 64 GPU |
| JAX device 数 | **256**（1 dev/chip，MegaCore） | 128 | **128**（2 dev/chip） | — |
| HBM / chip | 95.74 GB HBM2e | 95.74 GB | 192 GB HBM3e | 288 GB |
| 总 HBM | **24.5 TB** | 12.25 TB | 12.29 TB | 18.4 TB |
| BF16 TFLOPS / chip | 459 | 459 | 2,306 | 2,700 |
| 总 BF16 算力 | **117.5 PFLOPS** | 58.8 PFLOPS | 147.6 PFLOPS | 172.8 PFLOPS |

> 原计划拿 **Google 命名的 v5p-256（128 芯片）**去对 v7 4x4x4（128 device），
> 因为两者 device 数相同、总 HBM 接近。手上实际能用的是 256 芯片的池，
> 所以下面的 v5p 数字是 **256 芯片**口径；跟 v7 对比时必须按
> **per-chip** 归一，不能比整机吞吐。
> 单位换算见 [TPU-UNITS](https://github.com/yangwhale/tpu-recipes/blob/main/training/TPU-UNITS.md)。

### 4.2 显存测算 vs 实测

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

结论不变：**这个规模不需要 PP**，纯 FSDP + EP 装得下
（对比 GB300 因单域只有 64 卡才需要 PP=2）。

### 4.3 起步配置（**原始设计，已被实测推翻**）

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
> 而 FSDP 的 all-gather 能整体卸载到 SparseCore。详见 §5.2.2。

---

## 五、测试矩阵与实测结果

### 5.1 v7 (Ironwood) 4x4x4 — 64 chips / 128 devices — 已跑通

MFU 分母：**2,306** TFLOPS/chip；v7 是 2 device/chip，
**per-chip TFLOP/s = 日志值 × 2**（换算见 TPU-UNITS）。

| # | 配置 | step | TFLOP/s/dev | **per-chip** | **MFU** | tok/s/dev | 整机 tok/s |
|---|---|---|---|---|---|---|---|
| **V1** | FSDP=128 / 无 EP / pdbs=4 / seq=8192 / 只带 2 个 XLA flag | 25.11 s | 202.38 | **404.75** | **17.55%** | 1,305.1 | **167,059** |
| V2 | V1 + 补齐 XLA flag 集 + pdbs 上探 | | | | | | ⬜ |

```
number parameters: 298.786 billion          <- 与 v5p 侧逐位一致，移植没有走样
completed step: 5, seconds: 25.107, TFLOP/s/device: 202.375, loss: 12.815
completed step: 6, seconds: 25.107, TFLOP/s/device: 202.375, loss: 12.722
completed step: 7, seconds: 25.109, TFLOP/s/device: 202.358, loss: 12.645
completed step: 8, seconds: 25.111, TFLOP/s/device: 202.341, loss: 12.585
```

**V1 的 17.55% 只是起点，不是 v7 的水平。** v5p 侧从 2.45% 调到 36.72% 用了
26 个 XLA flag + pdbs=8；v7 这一轮只带了 2 个 flag、pdbs=4，
而且 `xla_tpu_enable_latency_hiding_layer_scheduler` 在 v7 上报
`requires sparse core collective aggregator to be enabled` 被迫摘掉。
按 v5p 的调优幅度推，v7 还有很大空间。

### 5.1.1 v7 用的是另一套 MaxText，补丁得重写

v5p 侧用的 `chrisya-maxtext-stable:oct` 镜像**驱动不了 Ironwood**：

```
libtpu build label: libtpu_lts_20250721_b_RC01
jaxlib._jax.XlaRuntimeError: INTERNAL: Failed to get global TPU topology.
```

2025 年 7 月的 libtpu 不认识 tpu7x。换成 `chrisya-maxtext-latest:runner`
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

好消息是**新版把我记在 §一之三的三个缺口补了两个**：

| 缺口（旧版） | 新版上游 |
|---|---|
| router 需要 fp32（我加了 `moe_router_dtype`） | 已有 `float32_gate_logits` |
| 专家 bias 的无梯度更新规则未实现 | 已有 `routed_bias_update_rate` |
| 初始化 `initializer_range` | 仍缺（不影响 SFT / 续训） |

所以移植后的 `hunyuan3.py` **比原来短**：Hy3 = Qwen3 的 attention +
DeepSeek 的 MoE，新版正好有 `AttentionWithNorm` 基类和 `RoutedAndSharedMoE`，
两个层类各自只跟 Qwen3 的对应类差**一行**。

### 5.1.2 第五到第八次踩同一个坑

移植过程中又撞了四次"按模型名列举的分支漏了 hunyuan3"，
**全部是运行时才报错，静态检查一个都抓不到**：

| # | 报错 | 位置 |
|---|---|---|
| 5 | `Input should be 'default', 'llama2-7b', ...` | `configs/types.py` 的 pydantic `Literal` 白名单 |
| 6 | `Loss-free load balancing is only supported for the DeepSeek decoder block` | `configs/types.py` 的 validator 把 `routed_bias_update_rate` 锁死在 DEEPSEEK |
| 7 | `Incorrect decoder_block name cfg.decoder_block.value='hunyuan3'` | `layers/nnx_decoders.py` **第三张**分派表（前两张在 `decoders.py`） |
| 8 | `Hunyuan3MoELayer.__init__() missing 1 required positional argument: 'quant'` | 两条构造路径签名不一致：nnx decoder 不传 `quant`，linen 路径传 |

加上 §一之三的路由分支、`model_name` 门，和 §一之五的 FLOP 公式，
**同一个模式在这个项目里出现了 8 次**：

> MaxText 里几乎每个"这个模型该走哪条路"的判断，都是一张按家族名字写死的表。
> 加新模型不是改一处，是**把所有这类表找齐**。漏掉的那张不会报错说"你漏了"，
> 它会报一个看起来完全无关的错——`quant` 参数缺失、pydantic 校验失败、
> 或者干脆不报错，安静地跑出另一套语义（路由分支和 FLOP 公式就是这种）。
>
> 找齐的办法只有一个：`grep -rn "DecoderBlockType.DEEPSEEK"` 和
> `grep -rn 'startswith(("deepseek'`，**每一处都问一遍"Hy3 该不该在这里"**。

### 5.1.3 v7 建池跟 v5p 完全不是一套流程

试了四次才成，每次的错都不一样，记下来省得别人再踩：

| 尝试 | 命令 | 报错 |
|---|---|---|
| 1 | `--placement-type=COMPACT`（v5p 的写法） | `tpu7x-standard-4t ... with placement policy is not supported. Use workload policy instead.` |
| 2 | 去掉 `--placement-type` | 同上——**GKE 在多机拓扑下会自动加 group placement** |
| 3 | 自建 workload policy（只给 `--type=HIGH_THROUGHPUT`） | `does not support TPU topology with group placement policy and workload policy at the same time` |
| 4 | workload policy 加 **`--accelerator-topology=4x4x4`** | 通过 |

正确写法（来自 [tpu-recipes ironwood 配方](https://github.com/yangwhale/tpu-recipes/tree/main/training/ironwood)）：

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
> 前面刚因为这个丢了 12.9 倍性能（§5.2.1），转头又来一遍。

容量方面：第一次 `Atomic resize failed with [GCE_STOCKOUT]`，
`us-central1-ai1a` 对本集群不可用，项目里也没有 tpu7x 预留。
改成 **64 → 32 → 16 递降重试**（spot 容量是波动的），
第二轮就在 `us-central1-c` 拿到了 64 芯片。多机 TPU 池是**全有全无**的：
4x4x4 要求物理连续立方体，16 台必须一次落位。

### 5.2 `np-v5p-256`（4x8x8，**256 chips / 256 devices**）— 已实测

**全部数字都是 FLOP 口径修正后的**（见 §一之五 bug #5）。
稳态取 step ≥ 3 的中位数。MFU 分母 **459** TFLOPS/chip；
v5p 是 MegaCore，**1 device = 1 chip，日志值不用乘 2**。

| # | 配置 | step | TFLOP/s/dev | **MFU** | tok/s/dev | 整机 tok/s | HBM/dev |
|---|---|---|---|---|---|---|---|
| o1 | 自己攒的：EP64/FSDP4, pdbs=1, seq=4096, 3 个 XLA flag | 49.99 s | 11.24 | **2.45%** | 81.9 | 20,974 | 54.8 G |
| **o2** | **照抄官方 DSV3 recipe**：FSDP=256/无 EP, pdbs=4, seq=8192, 26 个 XLA flag | **34.67 s** | **144.87** | **31.56%** | **945.1** | **241,935** | — |

**o1 → o2 是 12.9 倍。** 这一节剩下的篇幅都在解释这 12.9 倍是怎么来的。

### 5.2.1 我犯的错：没有从官方配方出发

我先按自己对模型的理解攒了一套配置（o1），理由写在 §4.3 里：
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

### 5.2.2 为什么 TPU 上 FSDP 打得过 EP

这是本轮最反直觉的一条，值得单独说。

- **GPU（GB300）**：专家并行靠 DeepEP 做 all-to-all，NVLink 域内带宽极高，
  EP 把专家权重摊开、只搬 token，是省显存又省带宽的打法。
- **TPU（v5p）**：ICI 是 3D torus，**64 路 all-to-all 要跨整个环面**，
  没有 NVLink 那种全连接域。而 FSDP 的 all-gather 是规则的近邻通信，
  还能整个卸载到 **SparseCore** 上跟 TensorCore 的矩阵乘重叠。

换句话说，**EP 在 TPU 上把通信换成了拓扑最不友好的那种形状**。
o2 里 `ici_expert_parallelism` 干脆是 1——192 个专家全靠
FSDP 沿 `embed` 维切开，all-gather 走 SparseCore。

> §4.3 里"EP 是主旋钮"那句话**对 GPU 成立，对 TPU 不成立**，
> 已在下方标注。这不是笔误，是把一代硬件的经验直接搬到另一代的典型翻车。

### 5.2.3 192 不是 2 的幂，这在 TPU 上是有代价的

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

### 5.2.4 消融：这 12.9 倍具体是谁贡献的

以 o2 为基准，每轮只动一项：

| # | 相对 o2 的改动 | step | TFLOP/s/dev | MFU | 整机 tok/s | HBM/dev | Δ MFU |
|---|---|---|---|---|---|---|---|
| **o11** | **pdbs=6 且 `out_proj=remat`（两项最优叠加）** | 46.89 s | **160.70** | **35.01%** | **268,363** | 80.9 G | **+3.45 pp** |
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

**3. batch 还能再上探。** pdbs 4 → 6 拿到 34.43%，是全场最高，
HBM 涨到 80.3 / 95.74 G。pdbs=8 大概率 OOM，正在测。

**4. `out_proj=offload` 是负收益（−0.22 pp）。** 官方 DSV3 配方里有这一项，
搬到 Hy3 上反而略亏。原因大概是 Hy3 的 attention 只占 2.36% FLOP，
省下来的那点显存不值得多一趟 PCIe 往返。
**这是"照抄官方"唯一需要改回来的地方。**

`tile_*` 三项只值 0.31 pp——在 256 卡上几乎测不出来，
跟 tpu-recipes 文档里 DSV3 的记录不一致，可能是模型形状不同。

**o11 把两项正收益叠起来（pdbs=6 + `out_proj=remat`），拿到 35.01%。**
两项单独是 +2.87 和 +0.22，叠加后 +3.45——**略大于两者之和**，
说明少一趟 out_proj 的 PCIe 往返，在 batch 更大时价值更高一点。
这是目前的最优配置：

```bash
model_name=hunyuan3-295b
ici_fsdp_parallelism=-1        # 256 路，不用 EP
ici_tensor_parallelism=1
per_device_batch_size=6
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

### 5.2.5 同一件事在 4 芯片和 256 芯片上结论相反

256 卡池被 spot 抢空时，我在 4 芯片 dev pod 上跑了同一套消融当备用轨道。
**大部分方向一致，但有一项完全反过来：**

| 改动 | 4 芯片 | 256 芯片 |
|---|---|---|
| 去 26 个 XLA flag | 6.25% → 5.25%（−16%） | 待重跑 |
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

### 5.2.6 spot 抢占怎么应对

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

### 5.3 三方横向对比

全部按 **per-chip / per-GPU** 归一。三边跑的是同一个 295B-A21B、
同样 BF16、同样 synthetic 数据、同样不开 checkpoint。

| | GB300 64 GPU | v7 4x4x4（64 chips） | v5p 256 chips |
|---|---|---|---|
| 计算单元数 | 64 GPU | 64 chips | 256 chips |
| BF16 峰值/单元 | 2,700 | 2,306 | 459 |
| 序列长度 | 4,096 | 8,192 | 8,192 |
| **实测 TFLOP/s/单元** | **854.0** | **404.8** | **168.6** |
| **MFU** | **31.6%** | **17.6%** | **36.7%** |
| **tok/s/单元** | **6,242** | **2,610** | **1,100** |
| **整机吞吐 tok/s** | **399,488** | **167,059** | **281,488** |
| 调优程度 | 已调优 | **首跑，2 个 XLA flag** | 已调优，26 个 flag |

读这张表的三个要点：

**1. v5p 的 36.7% 已经超过 GB300 的 31.6%。** 单卡算力差 5.9 倍，
但 MFU 反而更高——256 芯片的 3D torus + SparseCore 集合通信卸载，
把 MoE 那些碎通信藏得比 NVLink 域还干净。
代价是要 256 张卡才换来 GB300 64 卡七成的整机吞吐。

**2. v7 的 17.6% 不是 v7 的水平，是"还没调"的水平。**
v5p 从 2.45% 起步调到 36.7%，v7 这一轮只带了 2 个 XLA flag、pdbs=4。
按同样比例，v7 单芯片有希望摸到 GB300 的量级。

**3. 别用整机吞吐横向比。** v5p 那一列是 256 芯片，
另外两列是 64 个单元。要比性价比得再乘上单价，
这张表只回答"每个计算单元能压出多少"。

> 对比时统一到 **per-chip** 口径。v7 日志是 per-device，需 ×2；v5p 不需要。
> 这是跨代际比较最容易出错的一步。

---

### 5.4 v7 调优：目标该定在哪

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

#### DeepSeek V3 在 v7 上的实测：BF16 与 FP8 分别是多少

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

#### FP8 这一栏比看上去复杂

FP8 峰值是 BF16 的两倍（4,614 vs 2,307），但 DSV3 的吞吐**只涨了 21%**
（612.66 → 743.46）。换算成 MFU：

| | BF16 | FP8 | 说明 |
|---|---|---|---|
| DSV3 TFLOP/s/chip | 612.66 | 743.46 | **+21.4%** |
| 对本精度峰值的 MFU | 26.6% | **16.1%** | 峰值翻倍但吞吐没翻，MFU 反而掉 |
| 若错误地对 BF16 峰值算 | 26.6% | 32.2% | **这么算会高估一倍** |
| 稠密 llama3.1-405b 同口径 | 54.7% | 41.8% | 稠密 FP8 涨 **+52.8%**，兑现度高得多 |

**MoE 兑现不了 FP8 的两倍峰值，稠密可以。** 原因跟 §5.4 开头那条一样：
MoE 的时间大量花在路由、分组重排、all-to-all 和小块 GEMM 上，
这些环节不吃 MXU 峰值，把精度从 16 位降到 8 位对它们几乎没帮助。
稠密模型的时间集中在大 GEMM 上，降精度直接兑现。

> **报 FP8 的 MFU 时一定要说明分母。** 同一个 743.46，
> 对 FP8 峰值算是 16.1%，对 BF16 峰值算是 32.2%——差一倍。
> 本文所有 FP8 的 MFU 都对 **FP8 峰值** 算。

#### 我们现在离 DSV3 有多远

| | TFLOP/s/chip | MFU | 相对 DSV3 |
|---|---|---|---|
| DSV3 671B bf16（研发实测） | 612.66 | 26.6% | 1.00× |
| qwen3-235b-a22b bf16 | 629.79 | 27.3% | 1.03× |
| **hunyuan3 295B bf16（本项目首跑）** | **404.75** | **17.54%** | **0.66×** |

差 **1.51 倍**。Hy3 的激活参数（21 B）比 DSV3（37 B）还少，
结构也更简单（GQA 而非 MLA、192 专家而非 256），
**没有理由跑不到同一水位**——差距来自配置，不是架构。

#### 缺口从哪里补

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

`shard_exp_on_fsdp=True` 是**唯一不能照抄的一项**：它要求
`num_experts % ici_fsdp_parallelism == 0`，192 % 128 = 64 ≠ 0。
又是 §5.2.3 那个"192 不是 2 的幂"的代价，这次卡在 v7 的 128 device 上。

#### 调优轮次（w1–w8，进行中）

**w1 = 官方参数集一次性全开 → 卡死。**

```
Slow PjRt TPU operation detected: start_time=00:23:05 host_id=7
PendingEventLogger: High-level software slow operation detected.
TpuDiagnosticCoordinator: Harvesting hardware telemetry for stalled chips: [7]
```

step 0 用了 61 s（含 17 分钟编译），step 1 隔了 7.5 分钟才出，
之后一个芯片直接挂住，XLA 的诊断协调器被触发。
**30 个 XLA flag + tokamax kernel 一次全开，在 Hy3 的形状上有死锁。**

这跟 v5p 的经验相反：v5p 上"照抄官方全套"一次就成（§5.2.1）。
差别在于 v7 的 SparseCore 卸载路径和 tokamax kernel 都是 v5p 上没有的，
**照抄的前提是两边的硬件路径一样**，v7 不满足。

改成**增量加**，从已知能跑的 V1（404.8 TFLOP/s/chip）出发，一组一组往上叠：

| # | 相对 V1 的增量 | TFLOP/s/chip | MFU | 状态 |
|---|---|---|---|---|
| V1 | 基线：2 个 XLA flag / pdbs=4 / seq=8192 | 404.8 | 17.55% | ✅ |
| x1 | + `use_tokamax_gmm` | | | 🔄 |
| x2 | + `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | | | ⬜ |
| x3 | + adamw / bf16 优化器状态 + `use_iota_embed` | | | ⬜ |
| x4 | + SparseCore 卸载组（9 个 flag） | | | ⬜ |
| x5 | + 调度器组（4 个 flag） | | | ⬜ |
| x6 | + 杂项组（5 个 flag） | | | ⬜ |
| w1 | （对照）30 个 flag 一次全开 | — | **HANG** | ❌ |

> **"照抄官方配方"和"一次只动一个维度"不是互斥的，是分场景的。**
> v5p 上官方配方能整套照搬，因为硬件路径一致；
> v7 上整套照搬会死锁，就必须退回增量。
> 判断标准是**两边的执行路径是否相同**，不是"官方的就一定能用"。

---

## 六、待验证清单

按依赖顺序排，✅ 是本轮闭环的，⬜ 是还没做的。

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| 0 | 写出 `hunyuan3` block 并通过静态自检 | ✅ | 8 项检查全过；但**实跑的 8 个 bug 一个都没抓到**，见下方复盘 |
| 1 | 小规模真实前向 | ✅ | 4 芯片 v5p，r1–r20 共 20 轮，见 §一之四 |
| 2 | 192 experts × 80 层能否编译 | ✅ | v5p 256 芯片和 v7 64 芯片都编译并跑出稳态 |
| 3 | 参数量与 SSOT 对齐 | ✅ | 框架报 298.786 B = SSOT 294.9 B + MTP 头 3.886 B，两个平台逐位一致 |
| 4 | `normalization_layer_epsilon` | ✅ | 曾填错 1.0e-6，HF 原文是 **1e-05**，已修 |
| 5 | router fp32 | ✅ | 旧版我加了 `moe_router_dtype`；**新版上游已有 `float32_gate_logits`** |
| 6 | 专家 bias 的无梯度更新 | ✅ | **新版上游已有 `routed_bias_update_rate`**（旧版确实没有） |
| 7 | EP / FSDP 最优配比 | ✅ | 结论与预期相反：**TPU 上不用 EP**，见 §5.2.2 |
| 8 | `attention=flash` 能否编译 | ✅ | v5p / v7 都通过；v7 上还能用 `use_tokamax_splash` |
| 9 | MTP 开销 | ✅ | 全程 `mtp_num_layers=1`，`mtp_loss` 单独打出，加了 3.886 B 参数 |
| 10 | v7 调优到 MoE 合理水位 | 🔄 进行中 | 首跑 404.8 TFLOP/s/chip，目标见 §5.4 |
| 11 | SFT 路线：冻结 `gate.bias` | ⬜ | 上游有了更新规则，但 SFT 是要**冻结**它。本版无 `trainable_parameters_mask` |
| 12 | HF 权重 → MaxText Orbax 转换 | ⬜ | 只做吞吐基线可以不碰；要 SFT 必须做 |
| 13 | 真实数据集上的收敛验证 | ⬜ | 目前全是 synthetic，只证明"能算且不发散"，没证明"学得对" |
| 14 | `initializer_range` | ⬜ | from-scratch 才需要；加载权重或 SFT 不受影响 |

### 静态验证抓到了什么，没抓到什么

`verify_hunyuan3.py` 的 8 项检查全部通过，但**这 8 个实跑 bug 一个都没拦住**：

| bug | 层次 | 静态检查为什么看不见 |
|---|---|---|
| 1 pod 被 admission webhook 拒 | K8s 调度 | 不看集群 |
| 2 模型名不在白名单 | 框架注册 | 白名单是硬编码常量，不在被检查的代码路径上 |
| 3 `base_moe_mlp_dim=192` 不是 128 倍数 | 硬件约束 | 参数量算得出来，MXU tile 对不对齐算不出来 |
| 4 `fsdp_shard_on_exp` 与 EP 互斥 | 配置组合 | 单项都合法，组合才非法 |
| 5 FLOP 公式漏 hunyuan3 | 报表 | 不影响任何被检查的对象 |
| 6 pydantic `Literal` 白名单 | 框架注册 | 同 2，换了实现方式 |
| 7 `nnx_decoders.py` 第三张分派表 | 框架注册 | 检查只看了 `decoders.py` 的两张 |
| 8 两条构造路径签名不一致 | 框架接口 | 需要真正实例化才暴露 |

> 静态验证的价值是**证明逻辑对**（路由数学、参数量、分派结构），
> 这几项它确实抓到了两个真 bug（路由分支、`model_name` 门）。
> 但"能不能跑"是另一回事——**调度、注册、硬件对齐、接口签名，
> 只有真跑才知道。** 两类检查不可互相替代。

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

**已跑通两代硬件，v7 调优进行中**（2026-07-28）。

| 平台 | 规模 | 状态 | 最好成绩 |
|---|---|---|---|
| v5p | 4 chips（dev） | ✅ 20 轮迭代闭环 | 用于快速验证代码路径 |
| v5p | 256 chips | ✅ 11 轮调优闭环 | **36.72% MFU / 281,488 tok/s** |
| v7 Ironwood | 64 chips | ✅ 跑通，🔄 调优中 | 404.8 TFLOP/s/chip / 17.55% MFU |

**下一步**：v7 侧照 Ironwood 官方 DeepSeek3 配方调优（§5.4），
之后是权重转换和真实数据集收敛验证。

代码：[`maxtext-hunyuan3/`](maxtext-hunyuan3/)（v5p / linen 版）、
[`maxtext-hunyuan3-v7/`](maxtext-hunyuan3-v7/)（v7 / nnx 版）。
