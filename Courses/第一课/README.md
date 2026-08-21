# 第一课 · 模型架构

> 🚧 **大纲阶段**，正文未动笔。建于 2026-08-21，同日改版。
>
> **改版说明**：初版把这一课定为「LLM 是什么」的轻量概览。
> 现调整为**完整的模型架构课** —— 凡是跟架构相关的内容都收进来，
> 一次讲透，后面的课不用再回头补。

---

## 这一课要回答的问题

**一句话**：一个大语言模型，从里到外是由什么零件、按什么方式装起来的？

这一课**不讲**硬件、不讲怎么训、不讲数据怎么来。只讲**模型本身长什么样**，
以及**为什么长成这样**。

学完之后应该能：

1. 说清楚「预测下一个词」这件事怎么长出看起来像智能的东西
2. 画出一个 Transformer 的完整数据流，说得出每一步在干什么
3. 解释 Attention 和 MLP 的分工，以及为什么这个分工决定了后面所有的性能问题
4. 看懂任意一份现代模型的 config，说得出它相对标准 Transformer 改了什么、为什么改
5. 手算一个模型的参数量，并说得出参数都堆在哪儿

---

## 八节

### 1 · 一台会接话的机器

- 撕掉 AI 回答的剧本 —— LLM 就是「给一段文字，猜下一个词」的函数
- 它给的不是一个词，是**所有可能词的概率分布**
- 聊天机器人只是在前面加了一段开场白
- 模型是确定的，回答每次不同 —— 因为是**采样**不是取最大
- 「大」在哪：参数即旋钮，没有任何人手动设过它们
- 为什么 2017 年之后是 Transformer：**它能并行**
  （这一条要点出来 —— 架构选择从一开始就被硬件形状塑造着）

> 素材：3b1b `mini-llm` 整篇

### 2 · 从文字到向量

- **Tokenization**：为什么不能按字、也不能按词 —— BPE 怎么在两者之间找平衡
- 词表大小是个权衡：大了嵌入矩阵撑爆，小了序列变长
- **Embedding**：把 token 变成高维空间里的一个点
- 关键直觉：**方向是有含义的**。king − man + woman、cats − cat 这个复数方向
- 点积衡量对齐程度 —— 后面 attention 全靠它
- **位置信息**：Transformer 天生看不见顺序，得额外告诉它
  - 绝对位置编码 → 学习式 → **RoPE**（旋转位置编码）
  - RoPE 为什么赢：它编码的是**相对**距离，且能外推

> 素材：3b1b `gpt`(Ch5) 的 Embedding / Direction / Dot Product 三节；
> CS336 `section2_tokenizer`（7 题）+ `section3_transformer` 的 RoPE 部分；
> wiki `concepts/rope`

### 3 · 零件之一：Attention

- 要解决什么：*bank* 在 *river* 旁边和在 *money* 旁边不是一个意思
- **Query / Key / Value**：一个「我在找什么」、一个「我是什么」、一个「那就给你什么」
- 为什么用点积算相关性、为什么要除以 $\sqrt{d_k}$
- **Causal mask**：训练时能一次算完整句，但不能偷看未来
- **多头**：一次问多个不同的问题
- 参数账：GPT-3 单头 4 个矩阵约 630 万参数，96 头 × 96 层 = **约 580 亿**

> 素材：3b1b `attention`(Ch6)；CS336 `section3_transformer` 的
> scaled dot-product attention 与 multihead attention 两题；
> wiki `concepts/flash-attention`

### 4 · 零件之二：MLP

- 跟 Attention 相反：向量之间**不通气**，各走各的
- 三步：升维（4×）→ 非线性 → 降维
- **事实存在这里**：Michael Jordan → basketball 那个例子
  升维矩阵的某一行在问「是不是这个人」，
  ReLU 把它变成一个是/否，降维矩阵的对应列把「篮球」这个方向加回去
  —— 本质上是一个 **AND 门**
- ⚠️ **必须紧接着讲 superposition**，否则上面那个例子会误导：
  真实模型里单个神经元几乎从不对应一个干净概念。
  Johnson-Lindenstrauss：允许「近似垂直」而非严格垂直后，
  能塞进去的方向数随维度**指数增长**（85° 容差 + 12,288 维 → 400 亿个以上）。
  **这同时解释了模型为什么难解释、以及为什么放大收益这么大。**
- 激活函数的演化：ReLU → GELU → **SwiGLU**（为什么要门控）
- 参数账：每层 12 亿 × 96 层 = **约 1160 亿，占全模型三分之二**

> 素材：3b1b `mlp`(Ch7) 整篇；CS336 `section3_transformer` 的 SwiGLU 题；
> wiki `concepts/swiglu`

### 5 · 把零件装起来：一个 Block

- **残差流是一条通信总线** —— 所有层从它读、往它写。
  这个比喻一旦立住，后面讲激活占多少显存、讲 recompute、讲流水线切在哪，全都顺
- 残差连接为什么是必需的（梯度、以及"默认什么都不做"这个初始状态）
- **归一化**：LayerNorm → RMSNorm（砍掉均值项，为什么不影响效果）
- **Pre-norm vs Post-norm**：为什么现代模型全站到了 pre-norm 这边
- 堆 N 层，最后 unembedding + softmax 出概率
- **温度**：softmax 里那个常数在做什么

> 素材：3b1b `gpt`(Ch5) 的 Unembedding / Softmax；
> CS336 `section3_transformer` 的 RMSNorm / TransformerBlock / TransformerLM 三题
> + `section7_experiments` 的两个消融（去掉 RMSNorm、post-norm vs pre-norm）

### 6 · 智商从哪来

**这一节是这门课区别于普通科普的地方，要写足。**

- 分工的一句话版本：**关联在 Attention，记忆在 MLP**
- **Induction heads**：「A B … A → B」的模式补全 ——
  上下文学习（in-context learning）的主要机制，
  也是「Attention 负责关联」最硬的证据
- **残差流视角**下重看整个模型：每一层都在往同一条总线上加东西
- **Superposition** 的更一般表述：特征是方向，不是神经元
- 可解释性能做到什么程度：circuit、attribution graph
- 由此能顺理成章解释的几件事：
  为什么模型「知道」很多却「推不动」几步；
  为什么扩参数量主要在扩 MLP；
  为什么 MoE 是在扩「记得多少」而不是「想得多深」

> 素材：`~/learning/transformer-circuits/`（Anthropic 8 篇论文笔记 + 术语表）

### 7 · 架构在演化：现代模型改了什么

标准 Transformer 是 2017 年的。今天的模型每一处都动过。这一节讲**动在哪、为什么动**。

**Attention 一族 —— 都在省同一样东西（KV cache）**

- MHA → **MQA** → **GQA**：K/V 头数一路砍
- **MLA**（DeepSeek）：把 KV 压成低秩隐向量
- **稀疏 / 滑窗**：SWA、attention sink
- **线性注意力**：DeltaNet、GDN、KDA —— 用状态代替全量 KV
- **混合架构**：几层线性配一层全注意力

**MLP 一族 —— MoE**

- 从 Dense 的最佳逼近推出 MoE 的几何直觉
- 路由、负载均衡：aux loss → loss-free → **Quantile Balancing**
- Shared expert / fine-grained expert
- **为什么稀疏化的是 MLP 不是 attention** ——
  回到第 4 节：参数三分之二在那儿，而且它天然是"一堆独立的问题"

**其它已经收敛的选择**

- RMSNorm、SwiGLU、RoPE、无 bias —— 现在几乎是标配，各自赢在哪
- MTP（多 token 预测）

**对照一遍现役模型**：DeepSeek V3 / Qwen3 / Hunyuan3 / Kimi K2 / GLM ——
同一张表看它们各自选了什么。

> 素材：`~/learning/moe-tour/`（苏剑林 9 篇 + 图）；
> `~/learning/DeepSeek/`（23 篇论文）；
> wiki `concepts/` 里的 mla / gated-mla / swa / linear-attention / deltanet /
> gdn / kda / hybrid-attention / moe / eplb / noaux-tc / latent-moe / mtp 等约 20 页

### 8 · 参数账：1750 亿是怎么摞出来的

**这一节是第一课的落点，也是通向第二课的桥。**

| 部件 | 算法 | 参数量 | 占比 |
|---|---|---|---|
| 嵌入矩阵 `W_E` | 50,257 × 12,288 | 617,558,016 | 0.35% |
| 反嵌入矩阵 `W_U` | 同上转置 | 617,558,016 | 0.35% |
| 注意力（全部） | 每头 ~6.3 M × 96 头 × 96 层 | **约 580 亿** | ~33% |
| MLP（全部） | 每层 12 亿 × 96 层 | **约 1160 亿** | ~66% |
| **合计** | | **约 1750 亿** | |

两个反直觉，各值一段：

1. **大头在 MLP 不在 attention** —— 「attention 得到了所有的注意力，
   但大多数参数在它旁边那些块里」
2. **那 1160 亿正是「事实」存放的地方** —— 第 4 节和第 8 节在这里合上

然后是那个锚点（3b1b 原话）：

> 假设你每秒能做 **10 亿次**加法和乘法，训练最大的那些语言模型要花你 **一亿年以上**。

最后留一个不回答的问题，交给第二课：

> 1750 亿参数，用 bfloat16 存，光权重就是 **350 GB**。
> 训练还要存优化器状态。而一块 TPU v7 的一个 device 只有约 **94.74 GB**。
>
> **光是把它放下就要几十个 device。那到底该怎么算这笔账？** → 第二课

---

## 练习（草案）

三档，从零门槛到需要一台笔记本：

1. **纸笔**：给定词表大小与嵌入维度，手算各部件参数量，加起来对上总数
2. **十行代码**：调一个现成小模型，打印下一个词的概率分布，
   把 temperature 从 0 调到 2，看分布怎么变
3. **手写零件**（对应 CS336 Assignment 1 Section 3）：
   自己实现 Linear / Embedding / RMSNorm / SwiGLU / RoPE / Attention，
   用 CS336 的测试套件判分 —— **跑得通测试就是学会了**
4. **观察题**：找一句有歧义的话（*bank*、*model*、*quill*），换上下文，
   看下一个词的分布怎么变 —— 亲眼看见 attention 在起作用

CS336 讲义已验证：参考实现在一台 M4 Max 上，MPS 不到 5 分钟、CPU 约 30 分钟
就能训出会讲儿童故事的小模型。**所以这一课的门槛是零，不需要任何加速器。**

---

## 参考资料

主线是 **3Blue1Brown** 的神经网络系列（讲得特别好，且官网有社区文字版），
配 **CS336** 做练习，**transformer-circuits** 挖深度，**moe-tour** 补 MoE 理论。

材料清单、出处与许可见 [`教学材料/README.md`](教学材料/README.md)。
图怎么自己渲见 [`RENDER.md`](RENDER.md)。

---

## 待办

- [ ] **抓 Michael Jordan 那一帧** —— 它在 `BasicMLPWalkThrough` 的动画中段，
      `-s` 只存最后一帧，要用 `-n <序号>` 取。第 4 节需要这张
- [ ] 渲 `attention.py`（19 个场景）和 `embedding.py`（19 个场景）—— 第 2、3 节要用
- [ ] 读 3b1b 2026-07 的 **Cross-Entropy / Compression is Intelligence**，
      判断「压缩即智能」要不要进第 1 节
- [ ] 找到 3b1b 第 7 章引的那篇 **Google DeepMind 2023-12** 事实存储论文原文
      （**不是** Geva et al. 的 key-value memory 那篇，别弄混）
- [ ] 第 7 节那张「现役模型对照表」—— 需要逐个扒 config，不能凭印象填
- [ ] 350 GB 那个数换成我们自己 AOT 的实测值
- [ ] 定体量 —— 八节一次读完太长，可能要拆成八个文件

---

## 许可

本目录随 `Courses/` 整体采用 **CC BY-NC-SA 4.0**（[全文](../LICENSE.txt)）——
与所引用的 3Blue1Brown 材料同一许可。

每处借用注明出处 + 原文链接；具体例子（Michael Jordan、撕掉的剧本、
每秒十亿次算一亿年）明确标明来自 3Blue1Brown；文字用自己的话重写，不做翻译搬运。
