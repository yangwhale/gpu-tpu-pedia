# -*- coding: utf-8 -*-
"""专题三 · 图 A —— Attention 编年史：从 RNN 到今天，以及各家的混合配比。

⭐ **为什么要有这一张。** 2026-09-07 现场要求：

    「你先画一个全景图，把 Attention 的编年史给它画出来：
      1. 从一开始的 RNN 开始，按时间出现了什么注意力
      2. 它的典型模型是哪一个
      3. 这个模型一个循环的配比是什么样的
      拿条形图画出来，不同的配比不一样的颜色。
      所有的信息都去搜最新，千问、蚂蚁的灵、Kimi、混元，主流开源模型都别漏。」

⭐⭐ **这张图的主脊（也是整个专题的主脊）：**

    Transformer 当初做的交易是「用平方的计算量，买来完全的并行度」。
    这段历史，是在把那笔交易**往回赎** —— 但不能把并行度还回去。

   线性注意力想赎回 O(N)，代价是重新引入一个串行的状态；
   于是 chunk 化又是为了把并行度找回来。**一个完整的圆。**

📌 **下半那张格子图是全图的重心。** 2026-09-07 现场要求重画：

    「太乱了。便宜的那一层就不用写了，你就写什么注意力加什么注意力。
      后面那一个循环里边，把它做成一个一个格子的这种条状图。就比如说这个
      MLA 占几格，然后这个 KDA 占几格的这种一目了然的比例的条状图。
      然后每一个类型该用自己的颜色就用自己的颜色。」

⭐⭐ **改成格子之后，多出来两条旧画法根本画不出来的信息：**

   ① **「有没有格子」本身就是一条信息。** 整条一色 ＝ 每层同构，
      花样在层**内部**（层内稀疏就长这样）；切成格子 ＝ 层与层不一样，
      花样在层**之间**。
      ⛔ 所以别再给层内稀疏单画一种虚线框 ——&nbsp;那是**用两套画法说一件事**，
        而它本来可以由同一套画法自己说出来。

   ② **一个类型一个颜色之后，「贵的那层是什么」第一次看得见了。**
      混合的十家里九家是「冷色 ＋ 橙黄」（便宜的层配一层全注意力），
      **只有 GLM-5.3-Flash 是「蓝 ＋ 红」——&nbsp;它配的那层「贵的」，
      本身已经是稀疏的了。** 这条结论在旧画法（一个配比一个颜色）下
      根本无从看起 ——&nbsp;那时候颜色编码的是配比，不是类型。

⭐⭐ **2026-09-07 第三轮，现场逐条追问逼出来的四条更正。**
   这一轮的价值不在于「补全」，而在于**前一版有三处是错的或漏的，而且都不报错**：

   1. **DeepSeek-V4 原先写成「只有 CSA 和 HCA 两种层」——&nbsp;错。**
      追问原话：「V4 好好调查一下，它是不是只有 CSA 跟 HCA？那 MLA 和 DSA 跑哪去了？」
      扒 `DeepSeek-V4-Flash/config.json` 里的 `compress_ratios`，它**就是层表**：
      `[0,0, 4,128,4,128,…,4, 0]`——&nbsp;0＝滑窗全注意力、4＝CSA、128＝HCA。
      **43 层 ＝ 2 层 SWA 引导 ＋ 21 层 CSA ＋ 20 层 HCA，是三种层不是两种。**
      · **MLA 去哪了：被换掉了。** HF 官方文档原话「replaces DeepSeek-V3's
        Multi-head Latent Attention (MLA) with a hybrid local + long-range design」；
        `num_key_value_heads: 1` ——&nbsp;底层是 **shared K=V 的 MQA**，
        同一个张量既当 key 又当 value。⭐ 绕一大圈回到了 2019 年的 MQA。
      · **DSA 去哪了：没消失，降级成了零件。** Lightning Indexer 还在，
        它是 **CSA 内部**那一步打分（`index_n_heads: 64, index_topk: 512`）。

   2. **Ling 2.6 的现场记忆是 KDA ——&nbsp;查下来不是。** 官方 base 模型卡写的是
      **Lightning Attention ＋ MLA，7:1**，而且它是**从 Ling-2.0 的 GQA 迁移改造**
      来的（Lightning conversion → linear warmup → MLA conversion → MLA warmup），
      不是从头训。KDA 是 **Ling-3.0** 那一代才换上的。⭐ 两代确实换了一支。

   3. **GLM 漏了 5.3。** 官方仓库列得很清楚：5 / 5.1 / 5.2 / 5.3 都是 744B-A40B，
      **5.3 旗舰跟 5.2 是同一个 base，纯后训练、架构一个字没动**；
      只有 **5.3-Flash（320B-A18B）是全新训练的 base**，才是那个线性＋稀疏混合的。
      所以「Pro 去哪了」的答案是：**Pro 就是 GLM-5.3 本身，但它架构上等于 5.2。**

   4. **缺基线。** 前四行补上 GPT-3（MHA）／PaLM（MQA）／Llama 2（GQA）／
      DeepSeek-V2（MLA），**每个机制配它首次出现时的那个模型**；
      稀疏那一支的起点 DeepSeek-V3.2-Exp 也补进去了。

⭐⭐⭐ **2026-09-07 第四轮，一条追问换来整张表的重心。** 原话：

    「MiniMax M2 全注意力这个你得深入研究一下。所谓退回全注意力不大可能，
      因为全注意力就不可能有长上下文，都这个年代了，它肯定是有什么 trade off
      折中方案之类的。而且压缩注意力说的也是全注意力的意思，
      它是不是用了压缩注意力的 MLA 这种？」

   扒 `MiniMax-M2/config.json`，**这个怀疑对了一半，也纠正了一半**：

   · **对的一半：它绝不是「什么都没做的全注意力」。**
     `num_attention_heads: 48` / `num_key_value_heads: 8` ——&nbsp;**GQA-8**，
     KV 直接是 MHA 的 1/6；`rotary_dim: 64` 配 `head_dim: 128` ——&nbsp;
     **partial RoPE，只转一半维度**。原先那一行标成笼统的「FULL」是偷懒。
   · **纠正的一半：它不是 MLA，也不是压缩注意力。**
     config 里没有 `kv_lora_rank`、没有 indexer、`sliding_window: null`、
     `attn_type_list` 62 层全是 1。

   ⭐⭐ **这一问逼出了整张图最重要的一条口径：
   「全注意力」是相对旋钮③（线性）说的，不是相对旋钮①（KV 存多少）说的。**
   M2 退回的是③，①上它一直压着 ——&nbsp;**三个旋钮正交，M2 就是活证据。**

   ⭐⭐⭐ **而「全注意力撑不起长上下文」这个直觉，被量化成了一整列。**
   于是加了「上下文」这一列，扫下来的结论硬得出乎意料：
   **做到 1M 以上的八家，无一例外都动了旋钮②或③；纯全注意力那一档最高只到 256K。**
   最硬的对照来自 MiniMax 自己：**01 用 7:1 线性做到 10M
   （config 写着 `10240000`），M2 退回纯全注意力只剩 192K（`196608`）——&nbsp;
   同一家、同一批人，改一处架构差 50 倍。**

   ⛔ 上下文那一列**只填核到一手出处的**（多数是直接读 config.json），
     核不到就留「—」。⭐ 空格不是「没有」，是「本轮没核到」——&nbsp;别把它读成 0。

⭐⭐⭐ **2026-09-07 第五轮：砍掉「配比」列，加上「KV cache」列。** 原话：

    「格子图画的就很清楚，一目了然几比几，所以配比那一列就不要了。
      ……最重要的信息就像我们上一个专题讲的那样，KV Cache 到底占多大，
      你一目了然每一个 header 到底压了多少……128K 的时候 KV Cache 到底占了
      多大的地方？用一个小长条表示，条里边写占了多大，占的越多的颜色越深。」

   · **砍「配比」的判据很干净：一个信息只该有一个出口。** 格子已经把配比说完了，
     再写一遍不是冗余，是在**跟格子抢注意力**。
   · **KV 那一列是算出来的，不是抄来的。** 公式和每家的参数都在本文件里，
     谁都可以复算。⛔ 别把结果硬编码成常数 ——&nbsp;那样改了层数它不会跟着动。
   · ⚠️ **条长是对数刻度**，因为跨了三个数量级：线性刻度下 576 GiB 会吃光整行，
     而 697 MiB 连一个像素都占不到。**这一条必须写在图上，不能只写在注释里** ——
     读图的人看不到注释，而对数条会让差距「看起来变小」。

⭐⭐ **这一列冒出来的终点结论**：从 GPT-3 的 **576 GiB** 到 DeepSeek-V4 的
   **697 MiB**，同一个 128K 长度下，六年 **846 倍**。而这 846 倍
   **不是一个旋钮拧出来的**：MHA→GQA 砍头数（576→40）、MLA 改压缩（40→8.4）
   是旋钮①；线性把大部分层的 KV **直接删成零**（8.4→1.0）是旋钮③；
   CSA／HCA 存压缩池（→0.7）是旋钮②。**三个旋钮各贡献了一段。**

   ⛔ GPT-3 那 576 GiB 是**假想值** ——&nbsp;它只有 2K 上下文，从没在 128K 上跑过。
     但正因为假想，它才是一把干净的尺：**同一个长度下，六年到底省了多少。**

⭐⭐⭐ **2026-09-07 第六轮：全表 KV 重算 ＋ 补齐缺口 ＋ 拿外部锚点验公式。** 原话：

    「KV cache 这部分的大小非常的重要，你即使已经写出来的值也去给我重新算一遍。
      ……那些没有查到值的话，你也根据模型架构去算……DSA 它肯定是基于 MLA 来的，
      所以 DSA 应该就是 MLA 对吧？」

   · **「DSA 就是 MLA」这个理解对，但有一个例外。** V3.2 / GLM-5 系列 / 混元 Hy4
     的 config 里都有 `kv_lora_rank: 512` ——&nbsp;**indexer 只决定算哪些，
     不改变存什么**，所以 DSA 的 KV cache 就是 MLA 的 KV cache。
     ⛔ **但 DeepSeek-V4 是例外**：它把 MLA 换掉了，改成 shared-KV MQA ＋ 压缩池。
     **这条规律到 V4 就断了。**

   ⭐⭐ **自己算出来的数，必须找外部锚点验一次** ——&nbsp;否则「算得很认真」
   和「算错了」在纸面上长得一模一样。这两个锚点都不是我们挑的，是人家自己报的：

     ① DeepSeek-V2 论文：MLA 的 KV「相当于只有 **2.25 组**的 GQA」。
        2×2.25×128×60 ＝ **34,560**；本式 60×576 ＝ **34,560**。**完全相等。**
     ② DeepSeek-V4 报告：KV 是 V3.2 的 **7%**。本式算出 **7.9%**。

   两个独立锚点都落回来了，这套公式才敢用。

   · **补齐的缺口**（都是这一轮现读的 config）：Qwen3-Next 12 层全注意力 GQA-2
     →&nbsp;3.0 GiB；Qwen3.5 15 层 GQA-2 →&nbsp;3.8 GiB；Ling 2.6 是 **MLA**
     （`kv_lora_rank: 512`，10 层）→&nbsp;1.4 GiB；
     **MiniMax M3 是 GQA-4 ＋ 稀疏，不是 MLA** →&nbsp;15 GiB，上下文 **1M**。
   · ⭐ **M3 那一行值得单独看**：它走稀疏，KV 反而**比走 MLA 的 GLM-5.2 还大**。
     **稀疏省 FLOPs，不省 KV** ——&nbsp;这是活证据，别把两件事混起来。
   · ⛔ **小米那两行仍是「未核到」**：config.json 里那段量化 `ignored_layers`
     极长，把关键字段挤出了可取回范围；`configuration_*.py` 里只有占位默认值。
     **宁可留空，也不按「同类模型大概长这样」去填** ——&nbsp;那种填法看起来最合理。
   · ⚠️ **PaLM 是全表唯一一格不是一手 config 的**：头维取 256（多方公开复现一致，
     注意 48×256 ≠ d_model 18432，这在 PaLM 里是有意的）。按 384 算是 22 GiB，
     量级与结论都不变。

⭐⭐ **2026-09-07 第七轮：同厂上底色 ＋ 补齐小米那两格。** 原话：

    「同属于一家的话，你应该给它标成一样的背景颜色，这样好区分。然后小米家的
      这个 MiMo 的模型，它也是开源，所以说你去拿那个模型的配置以及它的一些代码
      之类的，你把它的那个 KV Cache 的大小也算出来，别在那空着。」

   · **同厂上色的价值**：这张表是**按时间排**的，同一家的行天然被打散在各处。
     颜色一上，「某某家走了什么路」不用眼睛去找，**它自己浮出来** ——
     MiniMax 三个粉框、GLM 四个紫框、混元两个橙红框，跨时间线一眼连得起来。

   ⛔ **初版把整行都上了底色，被当场叫停**：「你做过度了，不是说整行都标上颜色，
     而是只是把模型名字那一列，用那个长条形的背景框给它标上颜色，
     把模型名字也框到那个小框框里。」

   ⭐⭐ **这条判据值得单独记：分组线索只需要落在「被分组的那个东西」上。**
     整行上色等于给二十几行全铺了一层底噪，格子和 KV 条的颜色反而被拉低了对比 ——
     **为了让一个维度更清楚，把另外两个维度弄糊了。**
     一个小色框就够，而且**更好认**：色块小、边界清楚、跟内容不争地方。
     ⭐ 副产品：色框自带标签，原先那行厂商图例也就成了冗余，一并删掉。

   ⭐⭐ **小米那两格的教训比数字本身更值钱。**
   上一轮我写「未核到」，理由是 `config.json` 被超长的量化 `ignored_layers`
   字段截断。**这个理由是成立的，但结论下早了** ——&nbsp;
   参数就在**官方模型卡的 Model Summary 表**里，而且比 config 还全。

   ⛔ **`config.json` 拿不到 ≠ 数据不公开。** 模型卡、技术报告、推理框架的
     recipe 页都可能有。**别在第一条路堵死之后就写「未核到」** ——&nbsp;
     那看起来像严谨，实际上是少查了两个地方。

   · MiMo-V2-Flash：48 层 ＝ 40 SWA ＋ 8 全注意力，窗口 128 →&nbsp;**5.0 GiB**。
     头配置按同代 MiMo-V2.5 推算（这一格是**推的**，不是直读）。
     ⭐ **锚点③**：模型卡自称「KV 省近 6×」，而 48 ÷ 8 ＝ 6，本式算出 **6.0×**。
   · MiMo-V2.5-Pro：70 层 ＝ 60 SWA ＋ 10 全注意力，128 头 / 8 KV 头，
     头维 QK 192 ／ V 128 →&nbsp;**6.3 GiB**，1M 上下文。全部直读自模型卡。
   ⚠️ 小米这一族的 **K 和 V 维度不一样**（QK 192 / V 128），
     所以公式里是 `qk + v` 相加，**不是像 GQA 那样乘 2**。照抄会算错。

⛔⛔ **2026-09-07 第八轮：一句质疑推翻了一格数，也定住了整列的口径。** 原话：

    「MiniMax-01（456B）10M 的上下文是什么鬼，真的会那么长吗，为啥，效果如何？」

   **查证结果：这一格我写错了。** 官方论文（arXiv 2501.08313）原话是
   「can reach up to **1 million tokens during training** and
   **extrapolate to 4 million tokens during inference**」——&nbsp;
   **训练 1M、推理外推 4M**。而 config 里那个 `max_position_embeddings: 10240000`
   是**位置编码的容量上限**，官方从没声称过 10M。已改成 4M。

   ⭐⭐⭐ **但比这一格更重要的是它暴露的东西：整列的口径都得重新交代。**
   这一列的每个数都读自各家 config 的同一个字段，而**那个字段各家含义并不一样** ——
   有的填的是验证过的上下文，有的填的是理论容量。**同一个字段名，不同的语义。**

   ⭐ 而「声明」和「能用」之间还隔着一整个 benchmark 的落差，
   最好的证据来自厂商自己：小米 V2.5-Pro 的模型卡直接写着
   **V2-Pro「到 1M 时塌到 0.00」**，V2.5-Pro 在 1M 也只有 0.37／0.62。

   📌 所以列头加了「⚠️ 声明值」，并且落点⑥ 专门讲这件事。
   **看到「支持 N 万上下文」，先问是谁、在什么任务上、测出多少分。**

⭐ **2026-09-07 第九轮：第一列统一规格，第二列整列删掉。** 原话：

    「第一列统一模型大小、激活、总参数和激活参数，再加上模型层数。
      第二列明显就不需要，因为第三列写的明明白白的。」

   · **删「这一层 ＋ 那一层」的判据，跟上一轮删「配比」是同一条：
     一个信息只该有一个出口。** 格子里已经印着 KDA／MLA／CSA 这些简写，
     旁边再用全名写一遍，既占地方又跟格子抢注意力。⛔ 别再加回来。
   · **第一列统一成「名字　总参/激活 · 层数」**，规格集中在一处，
     不再散落在备注里。稠密模型写「N 稠密」，不硬凑一个激活数。
   · 这一轮顺带补齐了五个原先缺的规格（都是现查的官方口径）：
     MiniMax-01 **456B/45.9B**（官方模型卡）· DeepSeek-V3.2-Exp **671B/37B**
     · DeepSeek-V4-Flash **284B/13B**（arXiv 2606.19348；⛔ 别跟 V4-Pro 的
     1.6T/49B 弄混）· MiniMax M3 **428B/23B** · Kimi K3 **2.8T/104B**。
     ⚠️ K3 那个激活数各处口径不一（有按 16/896 专家推出 ~50B 的说法），
     这里取 NVIDIA NIM 与多家推理平台一致的 104B。

⛔ **一条口径护栏：格子宽度固定，不按满宽等分。**
   否则 4 格的循环和 8 格的循环画出来一样长，「这个循环有多长」这条信息就没了。

⛔ **落点里的分类判据要求「恰好两种类型」。** 只写「含一个便宜的 ＋ 含一个贵的」
   会把 DeepSeek-V4 也算进层间混合（它有 SWA），而 V4 的 SWA 是引导层。
   ⭐ **分类判据比结论句更容易悄悄出错** ——&nbsp;它不报错，只是把一行放进错误的桶，
     然后结论跟着变，而你看不出来。

⭐⭐ **2026-09-07 补了混元和 GLM 之后，多出来一条原先看不见的线：**
   混元 Hy4 的 **IndexCache** 和 GLM-5.2 的 **IndexShare** 是同一个想法 ——&nbsp;
   两家的 `indexer_types` 都是 `full, shared, shared, shared` 四层一循环。
   **稀疏的第二阶段优化不再是「让每个 query 少看几块」，
   而是「别每层都重新算一遍该看谁」——&nbsp;索引本身变成了新的开销。**
   这是 2026 年才冒出来的一层，值得单独占时间轴上一个点。

📌 出处（全部 2026-09-07 现搜，公开）：
   MiniMax-01/M1 7:1 与 M2 退回全注意力（MiniMax 官方博客《Why Did M2 End Up as
   a Full Attention Model?》）· M3 的 MSA（arXiv 2606.13392，top-16 × 128-token 块）
   · Qwen3-Next / Qwen3.5 3:1（Qwen 官方博客与 HF 模型卡的层布局串）
   · Kimi Linear 3:1、K3 93 层＝69 KDA＋24 Gated MLA（arXiv 2510.26692 与多家 day-0 支持文）
   · Ling-3.0-flash 5:1＝35 KDA＋7 MLA（inclusionAI HF 模型卡）· Ling 2.6 Lightning:MLA 7:1
   · 小米 MiMo-V2-Flash 5:1／V2.5-Pro 6:1，窗口 128（小米 MiMo 官方博客与 HF 模型卡）
   · **混元 Hy3**（preview 2026-04-23／正式版 2026-07-06 Apache 2.0）——&nbsp;
     配比这一格不是查来的，是**读我们自己仓库里那份 config**：
     `tpu/Hunyuan3-295B-Pretraining/assets/hunyuan3-tokenizer/config.json`，
     `HYV3ForCausalLM`，80 层，`num_attention_heads: 64 / num_key_value_heads: 8`
     →&nbsp;**纯 GQA-8，没有线性、没有稀疏、没有混合**
   · **混元 Hy4-preview**（2026-08-28，HF `tencent/Hy4-preview` 模型卡＋config）——&nbsp;
     770B/49B，78 层 `layer_types` **全部** `deepseek_sparse_attention`，
     Gated DSA ＋ IndexCache，indexer 32 头×128 维、top-k 2048，1M 上下文
   · **GLM-5**（2026-02-12，z.ai 官方博客）355B–744B，MLA ＋ DSA
   · **GLM-5.2**（2026-06-16，z.ai 博客＋HF config）744B，`GlmMoeDsaForCausalLM`，
     `index_topk_freq: 4`，官方原话「uses the same indexer across every four
     sparse attention layers, reducing per-token FLOPs by 2.9× at a 1M context」
   · **GLM-5.3-Flash**（2026-08-26，z.ai 博客＋HF config＋vLLM recipe）321B/18B，
     `layer_types` 是 `linear×3 → deepseek_sparse_attention×1` 循环，
     45 层 ＝ **34 KDA ＋ 11 稀疏 MLA**（NoPE），GLM 家族**第一次线性和稀疏同锅**
"""
import io
import math

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
PU, CY, BR, PK = "#8430ce", "#00838f", "#7a5000", "#c5221f"
W = 1820
p = []


def wpx(s, size=11.5):
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


def t(x, y, s, cls="svgsm", fill=None, bold=False, size=None, anchor=None):
    st = ["font-size:%dpx" % size] if size else []
    p.append('<text class="%s" x="%d" y="%d"%s%s%s>%s</text>' % (
        cls, x, y, ' fill="%s"' % fill if fill else '',
        ' text-anchor="%s"' % anchor if anchor else '',
        ' style="%s"' % ';'.join(st) if st else '',
        '<tspan font-weight="700">%s</tspan>' % s if bold else s))


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=6, sw=1, dash=None):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" '
             'stroke="%s" stroke-width="%s"%s/>'
             % (x, y, w, h, r, fill, stroke, sw,
                ' stroke-dasharray="%s"' % dash if dash else ''))


# ⛔ 2026-09-07：原先这里把 viewBox 的高度写成字面量 1106，加五行模型就被裁掉底边。
# ⭐ 这跟当初把泳道行数写死是**同一类错**：一个由别的东西推出来的值，被抄成了常量。
#    改成占位符，最后按真实落点回填 —— 以后加行不用再手算高度。
_HDR = len(p)
p.append("")

t(0, 18, 'Attention 编年史 ——&#160;<tspan font-weight="700">'
         '从 RNN 的一个补丁，到今天各家的混合配比</tspan>',
  "svglbl", "#202124", size=15)
t(0, 39, '⭐ 一句话看懂整段历史：<tspan font-weight="700">Transformer 当初做的交易是'
         '「用平方的计算量，买来完全的并行度」——&#160;而这段历史，'
         '是在把那笔交易<tspan style="text-decoration:underline">往回赎</tspan>，但不能把并行度还回去。</tspan>')
t(0, 57, '线性注意力想赎回 O(N)，代价是重新引入一个<tspan font-weight="700">串行的状态</tspan>；'
         '于是 chunk 化又是为了把并行度找回来。'
         '<tspan font-weight="700">一个完整的圆。</tspan>'
         '　<tspan fill="%s">（信息截至 2026-09-07，全部现搜）</tspan>' % GY, fill=GY)

# ══════════ 上半：时间轴 ══════════════════════════════════════════════
TY = 72
_PANEL = len(p)
p.append("")   # 面板底框占位，高度算完再补
t(16, TY + 24, '一、编年史 ——&#160;四条支线，各修各的毛病', "svglbl", "#202124", size=13)

Y0, Y1 = 2014, 2026
AX0, AXW = 150, W - 150 - 30
def xf(y, frac=0.0):
    return AX0 + int((y - Y0 + frac) / (Y1 - Y0 + 1) * AXW)

# 年份刻度
AXY = TY + 46
p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
         % (AX0 - 8, AXY, W - 24, AXY, GY))
for y in range(Y0, Y1 + 1):
    x = xf(y, 0.5)
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
             % (x, AXY - 4, x, AXY + 4, GY))
    t(x, AXY - 9, "'%s" % str(y)[2:], fill=GY, anchor="middle", size=10)

# ⛔ 2026-09-07 重排。初版每个事件带一句说明，结果同年多事件时说明互相糊住
#    （2024 的 MLA 和「可并行 DeltaNet」直接叠在一起），2026 那几个还冲出右边界。
# ⭐ 判据：**时间轴的职责是「什么时候出现了什么」，不是解释机制。**
#    机制在下半那张表里全都有 —— 时间轴只留事件名，一个字说明都不留，
#    立刻就读得清了。⛔ 别再往这些点上加说明文字。
LANES = (
    ("前史 · 注意力是 RNN 的补丁", GY, "#f1f3f4", [
        (2014, "Bahdanau 注意力"),
        (2017, "⭐ Transformer / MHA"),
    ]),
    ("① 每个 token 存多少（KV 怎么小）", BL, "#e8f0fe", [
        (2019, "MQA"), (2023, "GQA"), (2024, "MLA"), (2026, "Gated MLA"),
    ]),
    ("② 每个 query 看多少（稀疏 · 压缩）", OR, "#fef7e0", [
        (2023, "SWA · sink"), (2025, "NSA · DSA"), (2026, "CSA＋HCA · MSA"),
        # ⭐ 2026 多出来的**第二阶段**：不是「少看几块」，是「别每层重算该看谁」。
        #    GLM-5.2 叫 IndexShare、混元 Hy4 叫 IndexCache，两家的 indexer_types
        #    都是 full,shared,shared,shared 四层一循环 —— 撞了同一个想法。
        (2026, "IndexShare · IndexCache"),
    ]),
    ("③ 换一套数学（线性注意力）", PU, "#f3e8fd", [
        (2020, "线性 Transformer"), (2021, "DeltaNet"),
        (2024, "可并行 DeltaNet · GDN"), (2025, "KDA · Lightning"),
        (2026, "Gated DeltaNet-2"),
    ]),
    ("④ 不改数学，只改怎么算", GR, "#e6f4ea", [
        (2022, "⭐ FlashAttention"),
    ]),
)
LY = AXY + 14
# ⛔ 第三轮。前两版分别栽在：①带说明文字互相糊；②只留名字但相邻年份仍撞；
#    ③改成上下两行交错之后，泳道③ 里 2024 和 2026 又落回同一行、又撞上了
#    （中间只隔一个 2025，而「可并行 DeltaNet · GDN」有 150px 宽）。
# ⭐⭐ 三次都是同一个错：**按位置的规律去排，而不是按实际占多宽去排。**
#    交错、奇偶、固定两行 —— 都是「看起来会错开」的规律，而真正决定撞不撞的
#    是标签的**渲染宽度**。现在改成**贪心装箱**：逐个事件量宽，放进第一条
#    还装得下的行；行数按需要长。以后再加事件也不会撞。
# ⛔ 别再改回固定行数。
def _rows(evs):
    """把事件按实际宽度贪心分行，返回 [(事件, 行号)] 和总行数。"""
    ends, out = [], []
    for (yr, lab) in evs:
        x = xf(yr, 0.5)
        w = wpx(lab)
        right = x + w + 10 > W - 30
        x0 = (x - 6 - w) if right else (x + 6)
        for r, e in enumerate(ends):
            if x0 > e + 10:
                ends[r] = x0 + w
                out.append((yr, lab, r, right))
                break
        else:
            ends.append(x0 + w)
            out.append((yr, lab, len(ends) - 1, right))
    return out, len(ends)


LANE_Y, ly = [], LY
for (name, col, fill, evs) in LANES:
    placed, nrow = _rows(evs)
    h = 18 + nrow * 17 + 6
    LANE_Y.append((ly, h, placed))
    ly += h + 4
for (name, col, fill, evs), (y, h, placed) in zip(LANES, LANE_Y):
    box(4, y, W - 28, h, fill, col, 5)
    t(14, y + 15, name, fill=col, bold=True)
    for (yr, lab, r, right) in placed:
        x = xf(yr, 0.5)
        ty = y + 28 + r * 17
        p.append('<circle cx="%d" cy="%d" r="4" fill="%s"/>' % (x, y + 4, col))
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.2"/>'
                 % (x, y + 4, x, ty - 9, col))
        t(x + (-6 if right else 6), ty, lab, fill=col, bold=True,
          anchor="end" if right else None)
TH = ly - TY + 8

# ══════════ 下半：配比条形图 ══════════════════════════════════════════
p[_PANEL] = ('<rect x="0" y="%d" width="%d" height="%d" rx="8" fill="#f8f9fa" '
             'stroke="#dadce0" stroke-width="1"/>' % (TY, W, TH))
BY = TY + TH + 14
# ⛔ BH 原先也是字面量 500 —— 跟 viewBox 那个 1106 一样，加行就会溢出白底。
#    同样改成占位符，等 ROWS 数完再回填。
_BPANEL = len(p)
p.append("")
t(16, BY + 24, '二、编年史续 · 一个循环里都有哪些层 ——&#160;'
               '<tspan font-weight="700">一格 ＝ 一层，格子里写的是那一层用什么注意力</tspan>',
  "svglbl", "#202124", size=13)
# ⛔ 2026-09-07 第三版（现场追问逐条）：
#    ①「再往前还缺一些传统注意力，得先作为 baseline 出现」→ 补 GPT-3／PaLM／
#      Llama 2／DeepSeek-V2／V3.2，**每个新机制都配它首次出现时的那个模型**。
#    ②「GLM-5 到底是五点几？GLM-5.3 你漏掉了」→ 5／5.1／5.2／5.3 全列，
#      并写明 5.3 旗舰跟 5.2 是同一个 base（纯 post-training，架构没动）。
#    ③「Ling 2.6 我记得用的是 KDA」→ 查了官方 base 模型卡：**是 Lightning ＋ MLA 7:1**，
#      KDA 是 Ling-3.0 那一代。⭐ 两代确实不同源，备注里写清楚。
#    ④「每一格里边用简写字母写上」→ 格子里印 KDA／MLA／CSA／HCA 这些。
#    ⑤「DeepSeek V4 好好调查，是不是只有 CSA 跟 HCA？MLA 和 DSA 跑哪去了」
#      → 见下面 V4 那一行的注释，答案比原先写的复杂得多，而且原先是错的。
t(16, BY + 43, '⭐ <tspan font-weight="700">整条一色</tspan>＝每一层都一样（花样在'
               '<tspan font-weight="700">层内部</tspan>）；'
               '<tspan font-weight="700">切成格子</tspan>＝层与层不一样（花样在'
               '<tspan font-weight="700">层之间</tspan>）。'
               '前四行是<tspan font-weight="700">基线</tspan>：'
               '每个新机制都配上它<tspan font-weight="700">首次出现时的那个模型</tspan>。',
  fill=RD)
# ⛔ 对数刻度这件事必须写在**图上**，不能只写在源码注释里 ——
#   读图的人看不到注释，而对数条会让差距"看起来变小"。
t(16, BY + 61, '⚠️ KV 那一列的<tspan font-weight="700">条长是对数刻度</tspan>'
               '——&#160;跨了三个数量级，线性刻度下 576 GiB 会吃光整行、'
               '而 697 MiB 连一个像素都占不到。<tspan font-weight="700">看数字，别看长度比。</tspan>'
               '　口径：<tspan font-weight="700">128K、BF16、batch 1、不含量化</tspan>；'
               '公式与每家的参数全在生成脚本里，可复算。', fill=GY)
# ⭐⭐ 自己算出来的数，必须找外部锚点验一次 —— 否则「算得很认真」和「算错了」
#    在纸面上长得一模一样。这两个锚点都不是我们选的，是人家自己报的口径。
t(16, BY + 78, '⭐ <tspan font-weight="700">这套公式验过两个外部锚点</tspan>：'
               '① DeepSeek-V2 论文说 MLA 的 KV「相当于只有 <tspan font-weight="700">2.25 组</tspan>的 GQA」'
               '——&#160;2×2.25×128×60 ＝ 34,560，本式 60×576 ＝ <tspan font-weight="700">34,560，完全相等</tspan>；'
               '② V4 报告说它的 KV 是 V3.2 的 <tspan font-weight="700">7%</tspan>'
               '——&#160;本式算出 <tspan font-weight="700">7.9%</tspan>。'
               '<tspan font-weight="700">两个独立锚点都落回来了，公式才敢用。</tspan>', fill="#0b6b30")

# ── 一个类型一个颜色。同族相近色相，异族拉开 ──────────────────────────
AMB, ORG, DKR = "#f9ab00", "#e8710a", "#a50e0e"
TYPE_COL = {
    # 全注意力一族 —— 黄／橙
    "MHA": AMB, "MQA": AMB, "GQA": AMB, "FULL": AMB, "gAT": AMB,
    "MLA": ORG, "gMLA": ORG,
    # 线性一族 —— 冷色
    "KDA": BL, "GDN": "#12b5cb", "LTN": PU,
    # 窗口 —— 青
    "SWA": CY,
    # 稀疏一族 —— 红
    "DSA": RD, "gDSA": RD, "MSA": RD, "CSA": RD, "HCA": DKR,
}
# ── 同一家用同一个底色 ────────────────────────────────────────────
# ⭐ 现场要求：「同属于一家的话，应该给它标成一样的背景颜色，这样好区分。」
#   ⭐⭐ 这条的价值在于：表是**按时间排**的，同一家的行天然被打散在各处。
#     底色一上，「某某家走了什么路」这条线不用眼睛去找，它自己浮出来。
# ⛔ 底色必须**很淡**——它是分组线索，不是内容。抢了格子的颜色就本末倒置了。
#   （按名字前缀匹配，第一个命中为准。加新行时若厂商没命中，会落到中性灰。）
VENDOR = [
    ("GPT-3",    "OpenAI",   "#5f6368", "#f1f3f4"),
    ("PaLM",     "Google",   "#1e8e3e", "#e6f4ea"),
    ("Llama",    "Meta",     "#546e7a", "#eceff1"),
    ("DeepSeek", "DeepSeek", "#00838f", "#e0f2f1"),
    ("MiniMax",  "MiniMax",  "#c2185b", "#fce4ec"),
    ("Qwen",     "阿里 千问", "#e8710a", "#fff3e0"),
    ("Kimi",     "月之暗面",  "#5e35b1", "#ede7f6"),
    ("小米",      "小米",     "#f9ab00", "#f9fbe7"),
    ("GLM",      "智谱",     "#8430ce", "#f3e5f5"),
    ("Ling",     "蚂蚁 百灵", "#0288d1", "#e1f5fe"),
    ("混元",      "腾讯 混元", "#d84315", "#fbe9e7"),
]


def vendor_of(name):
    for key, label, ink, bg in VENDOR:
        if key in name:
            return label, ink, bg
    return "", GY, "#fff"


SPARSE = {"DSA", "gDSA", "MSA", "CSA", "HCA"}
LINEAR = {"KDA", "GDN", "LTN"}
# 简写 → 全名（写在「这一层 ＋ 那一层」那一列）
FULLNAME = {
    "MHA": "MHA", "MQA": "MQA", "GQA": "GQA", "FULL": "全注意力",
    "gAT": "Gated Attention", "MLA": "MLA", "gMLA": "Gated MLA",
    "KDA": "KDA", "GDN": "Gated DeltaNet", "LTN": "Lightning",
    "SWA": "SWA", "DSA": "DSA", "gDSA": "Gated DSA", "MSA": "MSA",
    "CSA": "CSA", "HCA": "HCA",
}

# (时间, 模型, 一个循环的构成 [(简写, 几层)…], 备注)
# ⛔ 只有一项 ＝ 每层同构，画成整条一色。
# ══════════════════════════════════════════════════════════════════
# KV cache：**算出来的，不是抄来的。** 公式写在这儿，参数来自各家 config，
# 谁都可以拿去复算。⛔ 别把结果硬编码成常数 —— 那样改了层数它不会跟着动。
#
#   普通 MHA/GQA/MQA ：2（K 和 V 两份）× 层数 × 长度 × KV头数 × 头维 × 2 B
#   MLA 一族         ：       层数 × 长度 × (kv_lora_rank ＋ rope 维) × 2 B
#                       ——&nbsp;只存那个压缩过的潜向量，**所有头共用一份**
#   线性层           ：**一个字节都不占**（状态是固定大小，跟长度无关），
#                       所以「层数」这一项只数**有 KV 的那些层**
#   DeepSeek-V4      ：特判。它 shared K=V（只存一份不是两份），
#                       而且 CSA/HCA 存的是**压缩池**：长度 ÷ 压缩率
#
# ⚠️ 口径：长度取 128K、BF16、batch=1、不含任何量化。
#    这几条一改，数就全变 —— 报 KV cache 的时候必须连口径一起报。
SEQ, BPE = 131072, 2


def kv_gib(spec):
    """按 spec 算 128K 时的 KV cache（GiB）。spec 为 None 表示没核到 config。"""
    if spec is None:
        return None
    kind = spec[0]
    if kind == "gqa":
        _, L, H, D = spec
        return 2 * L * SEQ * H * D * BPE / 2 ** 30
    if kind == "mla":
        _, L, R = spec                 # L 只数带 KV 的层，线性层不算
        return L * SEQ * R * BPE / 2 ** 30
    if kind == "swahyb":
        # 小米那一族：n_full 层全注意力 ＋ n_swa 层滑窗（窗口 win，只存 win 个 token）
        # ⚠️ 它的 K 和 V 维度**不一样**（QK 192 / V 128），所以这里是 qk+v 相加，
        #    不是像 GQA 那样乘 2。⛔ 照抄 GQA 的 ×2 会算错。
        _, n_full, n_swa, kvh_f, kvh_s, qk, v, win = spec
        ent = n_full * SEQ * kvh_f + n_swa * min(SEQ, win) * kvh_s
        return ent * (qk + v) * BPE / 2 ** 30
    if kind == "v4":
        _, n_csa, n_hca, n_swa, D, m_csa, m_hca, win = spec
        # shared K=V → 每个条目只存一份；CSA/HCA 存压缩池；每层另挂一条滑窗支路
        ent = (n_csa * (SEQ // m_csa) + n_hca * (SEQ // m_hca)
               + (n_csa + n_hca + n_swa) * win)
        return ent * D * BPE / 2 ** 30
    raise ValueError(kind)


def kv_fmt(g):
    if g is None:
        return "—"
    if g >= 100:
        return "%d GiB" % round(g)
    if g >= 10:
        return "%.0f GiB" % g
    if g >= 1:
        return "%.1f GiB" % g
    return "%d MiB" % round(g * 1024)


# 占得越多颜色越深 —— 一眼扫下来就是一条从深到浅的坡
def kv_col(g):
    if g is None:
        return "#e8eaed"
    for lim, c in ((100, "#7f0000"), (30, "#b31412"), (10, "#d93025"),
                   (3, "#e8710a"), (1, "#f9ab00")):
        if g >= lim:
            return c
    return "#1e8e3e"                   # 不到 1 GiB —— 已经是另一个量级了


ROWS = [
    # (时间, 模型, 循环构成, 上下文, KV 规格, 备注)
    # ⛔ 上下文这一列**只填核到一手出处的**（多数是本轮直接读的 config.json）。
    #   核不到就留「—」——&nbsp;宁可缺一格，不猜一格。
    # ── 基线：每个机制配它首次出现的模型 ────────────────────────────
    ("2020-05", "GPT-3　175B 稠密 · 96 层", [("MHA", 1)], "2K", ("gqa", 96, 96, 128),
     "96 层全 MHA，96 头×128 维。<tspan font-weight=\"700\">基线：KV 按头数线性长，没有任何省法</tspan>"),
    # ⚠️ PaLM 的头维取 256（多方公开复现一致；注意 48×256 ≠ d_model 18432，
    #    这在 PaLM 里是有意的，Q 投影不与 d_model 对齐）。若按 384 算是 22 GiB，
    #    量级与结论都不变。⛔ 这一格是本表**唯一一个非一手 config** 的。
    ("2022-04", "PaLM　540B 稠密 · 118 层", [("MQA", 1)], "2K", ("gqa", 118, 1, 256),
     "118 层，48 头<tspan font-weight=\"700\">共用 1 组 KV</tspan> ——&#160;第一次大规模砍 KV（Shazeer 2019）"),
    ("2023-07", "Llama 2　70B 稠密 · 80 层", [("GQA", 1)], "4K", ("gqa", 80, 8, 128),
     "8 组 KV。MQA 砍太狠会掉质量，GQA 是折中（arXiv 2305.13245）"),
    ("2024-05", "DeepSeek-V2　236B/21B · 60 层", [("MLA", 1)], "128K", ("mla", 60, 576),
     "不砍头，改低秩压缩。<tspan font-weight=\"700\">KV cache 降 93.3%</tspan>；V3 原样沿用"),
    # ── 三个旋钮各自的第一次 ────────────────────────────────────────
    # ⭐⭐ 这一行和下面 M2 那一行**必须并排读**：同一家公司、同一批人，
    #    01 用 7:1 线性混合做到 10M，M2 退回纯全注意力只剩 192K。差 50 倍。
    # ⛔ 2026-09-07 现场质疑：「10M 的上下文是什么鬼，真的会那么长吗？」
    #   查证结果：**我写错了。** 官方论文（arXiv 2501.08313）原话是
    #   「can reach up to **1 million tokens during training** and
    #    **extrapolate to 4 million tokens during inference**」。
    #   config 里那个 `max_position_embeddings: 10240000` 是**位置编码的容量上限**，
    #   官方从没声称过 10M。⭐⭐ 教训见落点⑥：**这一列读的是声明，不是能力。**
    ("2025-01", "MiniMax-01　456B/45.9B · 80 层", [("LTN", 7), ("GQA", 1)], "4M", ("gqa", 10, 8, 128),
     "⭐ 线性第一次上旗舰规模。<tspan font-weight=\"700\">训练 1M，推理外推 4M</tspan>"
     "（config 那个 10,240,000 只是位置编码容量，别当能力读）"),
    ("2025-09", "DeepSeek-V3.2-Exp　671B/37B · 61 层", [("DSA", 1)], "160K", ("mla", 61, 576),
     "⭐ <tspan font-weight=\"700\">稀疏这一支的起点</tspan>：MLA ＋ Lightning Indexer，每 query 只留 top-k"),
    ("2025-09", "Qwen3-Next　80B/3B · 48 层", [("GDN", 3), ("gAT", 1)], "256K", ("gqa", 12, 2, 256),
     "48 层 ＝ 36 线性 ＋ 12 全注意力（GQA-2，头维 256）"),
    # ⭐⭐ 2026-09-07 现场追问：「所谓退回全注意力不大可能，全注意力就不可能有长上下文，
    #    它肯定有什么 trade off。它是不是用了压缩注意力的 MLA 这种？」
    #    去扒 MiniMax-M2/config.json，**对了一半，也纠正了一半**：
    #    · 对的一半：它绝不是「什么都没做」。`num_attention_heads: 48` /
    #      `num_key_value_heads: 8` →&nbsp;**GQA-8**；`rotary_dim: 64` 配
    #      `head_dim: 128` →&nbsp;**partial RoPE，只转一半维度**。
    #    · 纠正的一半：**它不是 MLA、也不是压缩注意力。** 没有 `kv_lora_rank`、
    #      没有 indexer、`sliding_window: null`、`attn_type_list` 62 层全是 1。
    #    ⭐⭐⭐ 这一问逼出了本图最重要的一条口径：
    #      **「全注意力」是相对旋钮③（线性）说的，不是相对旋钮①（KV 存多少）说的。**
    #      M2 退回的是③，①上它一直压着。三个旋钮正交，M2 就是活证据。
    #    ⭐ 而「全注意力撑不起长上下文」这个直觉，被上下文那一列量化了：192K。
    ("2025-10", "MiniMax M2　230B/10B · 62 层", [("GQA", 1)], "192K", ("gqa", 62, 8, 128),
     "⛔ <tspan font-weight=\"700\">「退回全注意力」不等于什么都没做</tspan>：它是 GQA-8 ＋ partial RoPE。"
     "但确实<tspan font-weight=\"700\">没有 MLA、没有稀疏</tspan>"),
    ("2025-10", "Kimi Linear　48B/3B · 27 层", [("KDA", 3), ("MLA", 1)], "1M", ("mla", 7, 576),
     "27 层 ＝ 20 KDA ＋ 7 MLA（<tspan font-weight=\"700\">末层强制 full，所以多一层</tspan>）。已用 NoPE"),
    # ⭐ 小米这两行原先是「未核到」——&nbsp;config.json 被超长的量化字段截断了。
    #   后来在**官方模型卡的 Model Summary 表**里拿到了全部参数，比 config 还全。
    #   ⛔ 教训：**config.json 拿不到不等于数据不公开** —— 模型卡、技术报告、
    #     推理框架的 recipe 页都可能有，别在第一条路堵死之后就写「未核到」。
    # 📌 V2-Flash：48 层 ＝ 8 个 hybrid block ×（5 SWA ＋ 1 GA）＝ 40 SWA ＋ 8 GA。
    #   头配置按**同代 MiMo-V2.5 那一列**（64 头 / GA 8 KV 头、SWA 4 KV 头 /
    #   头维 QK 192、V 128 / 窗口 128）——&nbsp;这一格是**推算**，不是直读。
    #   ⭐ 但有自洽锚点：模型卡自称「KV cache 省近 6×」，而 48 ÷ 8 ＝ 6，**对上了**。
    ("2026-01", "小米 MiMo-V2-Flash　309B/15B · 48 层", [("SWA", 5), ("FULL", 1)], "256K",
     ("swahyb", 8, 40, 8, 4, 192, 128, 128),
     "48 层 ＝ 40 SWA ＋ 8 全注意力，窗口 128。"
     "<tspan font-weight=\"700\">模型卡自称 KV 省近 6×，48÷8 正好是 6</tspan>"),
    ("2026-02", "GLM-5　744B/40B · 78 层", [("DSA", 1)], "198K", ("mla", 78, 576),
     "MLA ＋ DSA，78 层。<tspan font-weight=\"700\">GLM-5.1 是同一套架构</tspan>，只有后训练不同"),
    ("2026-03", "Qwen3.5　397B/17B · 60 层", [("GDN", 3), ("gAT", 1)], "256K", ("gqa", 15, 2, 256),
     "60 层 ＝ 45 线性 ＋ 15 全注意力，<tspan font-weight=\"700\">config 里 full_attention_interval: 4</tspan>"),
    # 📌 V2.5-Pro 的参数是官方模型卡 Model Summary 直给的，不是推的：
    #   70 层（10 全注意力 ＋ 60 SWA）、128 头 / 8 KV 头（GQA）、
    #   头维 QK 192 ／ V 128、窗口 128、1M 上下文。
    ("2026-04", "小米 MiMo-V2.5-Pro　1.02T/42B · 70 层", [("SWA", 6), ("FULL", 1)], "1M",
     ("swahyb", 10, 60, 8, 8, 192, 128, 128),
     "70 层 ＝ 60 SWA ＋ 10 全注意力，窗口还是 128 ——&#160;"
     "<tspan font-weight=\"700\">1M 上下文里最省的一档</tspan>"),
    ("2026-05", "DeepSeek-V4-Flash　284B/13B · 43 层",
     [("SWA", 2), ("CSA", 1), ("HCA", 1), ("CSA", 1), ("HCA", 1)], "1M",
     ("v4", 21, 20, 2, 512, 4, 128, 128),
     "⭐ 前 2 层 SWA 引导，之后 <tspan font-weight=\"700\">CSA／HCA 严格交替</tspan>（21＋20）。"
     "<tspan font-weight=\"700\">MLA 被换掉了</tspan>，底层是 shared-KV 的 MQA"),
    ("2026-06", "GLM-5.2　744B/40B · 78 层", [("DSA", 1)], "1M", ("mla", 78, 576),
     "⭐ ＋IndexShare：每四个稀疏层共用一个 indexer。<tspan font-weight=\"700\">198K → 1M 就是这一步</tspan>"),
    ("2026-06", "Ling 2.6-1T　1T/63B · 80 层", [("LTN", 7), ("MLA", 1)], "256K", ("mla", 10, 576),
     "⛔ 不是 KDA。而且是<tspan font-weight=\"700\">从 Ling-2.0 的 GQA 迁移改造</tspan>来的，不是从头训"),
    ("2026-06", "MiniMax M3　428B/23B · 60 层", [("MSA", 1)], "1M", ("gqa", 60, 4, 128),
     "⭐⭐ 60 层 GQA-4 ＋ 稀疏。<tspan font-weight=\"700\">它的 KV 比走 MLA 的 GLM-5.2 还大</tspan> ——&#160;<tspan font-weight=\"700\">稀疏省 FLOPs，不省 KV</tspan>"),
    ("2026-07", "Kimi K3　2.8T/104B · 93 层", [("KDA", 3), ("gMLA", 1)], "1M", ("mla", 24, 576),
     "93 层 ＝ 69 KDA ＋ 24 Gated MLA（<tspan font-weight=\"700\">末层 92、93 连着两层 full</tspan>）"),
    ("2026-07", "混元 Hy3　295B/21B · 80 层", [("GQA", 1)], "256K", ("gqa", 80, 8, 128),
     "⛔ 80 层全是 GQA-8 ——&#160;<tspan font-weight=\"700\">线性一层都没上</tspan>"),
    ("2026-07", "Ling-3.0-flash　124B/5.1B · 42 层", [("KDA", 5), ("gMLA", 1)], "256K", ("mla", 7, 576),
     "42 层 ＝ 35 KDA ＋ 7 MLA。<tspan font-weight=\"700\">跟 2.6 换了一支</tspan>；"
     "同代 tiny 是 3:1，旗舰尚未发布"),
    ("2026-08", "GLM-5.3　744B/40B · 78 层", [("DSA", 1)], "1M", ("mla", 78, 576),
     "⚠️ <tspan font-weight=\"700\">旗舰版跟 5.2 是同一个 base，纯后训练，架构一个字没动</tspan>"),
    ("2026-08", "混元 Hy4-preview　770B/49B · 78 层", [("gDSA", 1)], "1M", ("mla", 78, 576),
     "78 层全稀疏 ＋ IndexCache（每 4 层只有 1 层自己算索引）"),
    ("2026-08", "⭐ GLM-5.3-Flash　320B/18B · 45 层", [("KDA", 3), ("DSA", 1)], "1M", ("mla", 11, 576),
     "45 层 ＝ 34 KDA ＋ 11 稀疏 MLA（NoPE）——&#160;"
     "<tspan font-weight=\"700\">GLM 第一次线性和稀疏同锅</tspan>，全新的 base"),
]

# ── 图例（两行，按族分组）────────────────────────────────────────────
LX = 16
for li, (fam, keys) in enumerate((
        ("全注意力一族", ("MHA", "MQA", "GQA", "FULL", "gAT", "MLA", "gMLA")),
        ("线性", ("KDA", "GDN", "LTN")),
        ("窗口", ("SWA",)),
        ("稀疏一族", ("DSA", "gDSA", "MSA", "CSA", "HCA")))):
    pass
lx, ly2 = LX, BY + 102
for fam, keys in (("全注意力一族", ("MHA", "MQA", "GQA", "FULL", "gAT", "MLA", "gMLA")),
                  ("线性", ("KDA", "GDN", "LTN")),
                  ("窗口", ("SWA",)),
                  ("稀疏一族", ("DSA", "gDSA", "MSA", "CSA", "HCA"))):
    t(lx, ly2, fam + "：", fill="#202124", bold=True)
    lx += wpx(fam + "：") + 4
    for k in keys:
        c = TYPE_COL[k]
        box(lx, ly2 - 11, wpx(k, 9) + 12, 15, c, c, 3)
        t(lx + 6, ly2, k, fill="#fff", bold=True, size=9)
        lx += wpx(k, 9) + 12 + 6
    lx += 14

# ── 表头 ────────────────────────────────────────────────────────────
# ⛔ 2026-09-07：删掉「这一层 ＋ 那一层」那一列。现场原话：
#   「第二列明显就不需要，因为第三列写的明明白白的。」
# ⭐ 判据跟删「配比」那次是同一条：**一个信息只该有一个出口。**
#   格子里已经印着 KDA／MLA／CSA 这些简写了，旁边再用全名写一遍，
#   既占地方又跟格子抢注意力。⛔ 别再加回来。
# ⭐ 第一列同时统一成「名字　总参/激活 · 层数」——&nbsp;规格集中在一处，
#   不用再散落在备注里。
MDLX, BARX = LX + 62, 360
CELL, CGAP, MAXC = 42, 3, 8
BARW = MAXC * CELL + (MAXC - 1) * CGAP
# ⛔ 「配比」那一列删了 —— 现场原话：「格子图画的就很清楚，一目了然几比几，
#    所以配比那一列就不要了。」⭐ 判据很干净：**一个信息只该有一个出口。**
#    格子已经把配比说完了，再写一遍不是冗余，是在跟格子抢注意力。
CTXX = BARX + BARW + 16
KVX = CTXX + 62                       # KV cache 那一列，做得宽
KVW = 210                             # 条最长 210px
NOTEX = KVX + KVW + 76
HY = BY + 130
t(LX, HY, '时间', fill=GY, bold=True)
t(MDLX, HY, '模型', fill=GY, bold=True)
t(BARX, HY, '一个循环（一格 ＝ 一层）', fill=GY, bold=True)
t(CTXX, HY, '上下文', fill=GY, bold=True)
t(CTXX, HY - 13, '⚠️ 声明值', fill=RD, size=9)
t(KVX, HY, 'KV cache＠128K（BF16，batch 1）', fill=GY, bold=True)
t(NOTEX, HY, '备注', fill=GY, bold=True)
p.append('<line x1="16" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
         % (HY + 6, W - 16, HY + 6, "#dadce0"))

R0, RH = HY + 30, 30
for i, (tm, mdl, cyc, ctx, kvspec, note) in enumerate(ROWS):
    y = R0 + i * RH
    # ⛔ 2026-09-07：初版把**整行**都上了底色，被当场叫停：「你做过度了，
    #   不是说整行都标上颜色，而是只是把模型名字那一列，用那个长条形的
    #   背景框给它标上颜色，把模型名字也框到那个小框框里。」
    # ⭐ 判据：**分组线索只需要落在「被分组的那个东西」上。** 整行上色等于
    #   给二十几行全铺了一层底噪，格子和 KV 条的颜色反而被拉低了对比。
    #   一个小色框就够了 —— 它甚至更好认，因为色块小、边界清楚。
    t(LX, y, tm, fill=GY)
    # 模型名套一个厂商色的小框 —— 同一家一个颜色，扫一眼就连得起来
    _vlab, _vink, _vbg = vendor_of(mdl)
    box(MDLX - 7, y - 14, wpx(mdl) + 15, 20, _vbg, _vink, 5)
    t(MDLX, y, mdl, fill=_vink, bold=True)
    seen = []
    for ty, _ in cyc:
        if ty not in seen:
            seen.append(ty)
    if len(cyc) == 1:
        ty = cyc[0][0]
        c = TYPE_COL[ty]
        box(BARX, y - 13, BARW, 18, c, c, 4)
        t(BARX + BARW // 2, y, '%s ——&#160;每一层都是这个' % ty, fill="#fff",
          bold=True, anchor="middle", size=10)
    else:
        # ⛔ 格子宽度固定，不按满宽等分 ——&#160;否则 4 格的循环和 8 格的循环
        #   画出来一样长，「这个循环有多长」这条信息就没了。
        gx = BARX
        for ty, k in cyc:
            c = TYPE_COL[ty]
            for _ in range(k):
                box(gx, y - 13, CELL, 18, c, c, 3)
                t(gx + CELL // 2, y - 1, ty, fill="#fff", bold=True,
                  anchor="middle", size=9)
                gx += CELL + CGAP
    # 上下文：≥1M 的标红加粗 —— 那一档是这张表最想让人看见的分界
    if ctx == "—":
        t(CTXX, y, '—', fill="#bdc1c6")
    else:
        big = ctx.endswith("M")
        t(CTXX, y, ctx, fill=RD if big else GY, bold=big)
    # ── KV cache 条 ────────────────────────────────────────────────
    # ⛔ 条长用**对数**刻度。线性刻度下 GPT-3 那 576 GiB 会把整行吃光，
    #   而 DeepSeek-V4 那 0.7 GiB 连一个像素都占不到 —— 跨三个数量级的量
    #   本来就不该用线性条。⚠️ 但对数条会让差距"看起来变小"，
    #   所以条里必须写数值，而且图脚要注明是对数。
    g = kv_gib(kvspec)
    c = kv_col(g)
    if g is None:
        box(KVX, y - 13, 58, 18, "#f8f9fa", "#dadce0", 3)
        t(KVX + 29, y, '未核到', fill="#9aa0a6", anchor="middle", size=9)
    else:
        lo, hi = math.log10(0.3), math.log10(700.0)
        # ⛔ 最小宽度不能写死（原先写 46，"1008 MiB" 那种标签直接被条边裁掉）——
        #   ⭐ 同一个错这张图上已经犯过三次：**按位置定尺寸，而不是按内容实际多宽。**
        #     短条的下限必须由**标签自己的渲染宽度**决定。
        lab = kv_fmt(g)
        w = max(wpx(lab, 10) + 18,
                int((math.log10(g) - lo) / (hi - lo) * KVW))
        box(KVX, y - 13, w, 18, c, c, 3)
        t(KVX + 9, y, lab, fill="#fff", bold=True, size=10)
    if note:
        t(NOTEX, y, note, fill=GY)

# ── 落点 ────────────────────────────────────────────────────────────
LZ = R0 + len(ROWS) * RH + 6
BH = LZ - BY + 174 + 14
p[_BPANEL] = ('<rect x="0" y="%d" width="%d" height="%d" rx="8" fill="#fff" '
              'stroke="#dadce0" stroke-width="1"/>' % (BY, W, BH))
box(16, LZ, W - 32, 174, "#e8f0fe", BL, 6)
t(30, LZ + 20, '⭐ 这张格子图一眼能看出六件事', "svglbl", "#174ea6", size=12)
# ⛔ 「所有 X 都……」这种全称句会被后来加的行悄悄证伪，而且不报错。
#    所以按数据分类**先报数再下结论**。⭐ 分类判据写成代码，加行时自动跟着变。
_uni = [r for r in ROWS if len(r[2]) == 1]
_mix = [r for r in ROWS if len(r[2]) > 1]
# ⛔ 判据必须要求**恰好两种类型**。只写「含一个便宜的 ＋ 含一个贵的」会把
#    DeepSeek-V4 也算进来（它有 SWA，也有 CSA/HCA），而 V4 的 SWA 是**引导层**、
#    不是循环里那个便宜的层 —— 混进来 3:1～7:1 那条结论就假了。
# ⭐ 这是这张图第二次栽在同一个地方：**分类判据比结论句更容易悄悄出错**，
#   因为它不报错，只是把一行放进了错误的桶，然后结论跟着变、而你看不出来。
_hyb = [r for r in _mix
        if len({t_ for t_, _ in r[2]}) == 2
        and any(t_ in LINEAR or t_ == "SWA" for t_, _ in r[2])
        and any(t_ not in LINEAR and t_ != "SWA" for t_, _ in r[2])]
_oth = [r for r in _mix if r not in _hyb]
t(30, LZ + 40, '① 表里 %d 家：<tspan font-weight="700">%d 家是「便宜的层 ＋ 一层贵的」</tspan>，'
               '%d 家<tspan font-weight="700">每层同构</tspan>（整条一色），%d 家是别的混法。'
               '而那 %d 家混合的，<tspan font-weight="700">配比无一例外落在 3:1 ～ 7:1</tspan>'
               '——&#160;<tspan font-weight="700">没有人敢全用线性，也没有人只掺一两层。</tspan>'
               % (len(ROWS), len(_hyb), len(_uni), len(_oth), len(_hyb)), fill="#174ea6")
t(30, LZ + 58, '② <tspan font-weight="700">前四行是基线，也是一条完整的小史</tspan>：'
               'MHA →&#160;MQA（砍到 1 组）→&#160;GQA（折中）→&#160;MLA（改压缩）'
               '——&#160;<tspan font-weight="700">四步全都只在动「每个 token 存多少」这一个旋钮</tspan>，'
               '花了四年。', fill="#174ea6")
# ⭐⭐ 这一条是**改成按类型上色之后才冒出来的** —— 旧画法一个配比一个颜色，
#    根本看不见「贵的那层是什么」。
_warm = [r for r in _hyb if r[2][-1][0] not in SPARSE]
_cold = [r for r in _hyb if r[2][-1][0] in SPARSE]
t(30, LZ + 76, '③ ⭐ <tspan font-weight="700">扫一眼颜色搭配</tspan>：'
               '混合的那 %d 家里 <tspan font-weight="700">%d 家都是「冷色 ＋ 黄橙」</tspan>'
               '（便宜的层配一层全注意力）。'
               '<tspan font-weight="700">只有 %s 是「蓝 ＋ 红」：它配的那层「贵的」，'
               '本身已经是稀疏的了。</tspan>'
               % (len(_hyb), len(_warm),
                  "、".join(r[1].replace("⭐ ", "").split("（")[0] for r in _cold)),
  fill="#174ea6")
# ⭐⭐⭐ 2026-09-07 现场追问逼出来的一条，也是整张表最硬的一条：
#    「所谓退回全注意力不大可能，因为全注意力就不可能有长上下文。」
#    ——&nbsp;把上下文那一列竖着扫一遍，这个直觉被数据完全证实了，
#    而最硬的对照来自 MiniMax 自己：同一家、同一批人，改一处架构差 50 倍。
# ⛔ 空着的那几格是**没核到一手出处**，不是没有 —— 宁可缺不猜。
_1m = [r for r in ROWS if r[3].endswith("M") and r[3] != "—"]
_pure = [r for r in ROWS if len(r[2]) == 1 and r[2][0][0] not in SPARSE]
t(30, LZ + 94, '④ ⭐⭐ <tspan font-weight="700">把「上下文」那一列竖着扫一遍</tspan>：'
               '<tspan font-weight="700">做到 1M 以上的 %d 家，无一例外都动了旋钮②或③</tspan>；'
               '而纯全注意力那一档最高只到 256K。'
               '<tspan font-weight="700">最硬的对照来自 MiniMax 自己：'
               '01 用 7:1 线性外推到 4M，M2 退回纯全注意力只剩 192K ——&#160;'
               '同一家、同一批人，差二十倍。</tspan>' % len(_1m), fill="#174ea6")

# ⭐⭐⭐ 这一条是加了 KV 那一列才浮出来的，也是整张表的终点。
# ⛔ GPT-3 那 576 GiB 是**假想值**——它只有 2K 上下文，从来没在 128K 上跑过。
#   但正因为假想，它才是一把干净的尺：**同一个长度下，六年到底省了多少。**
_kv = [(r[1], kv_gib(r[4])) for r in ROWS if r[4] is not None]
_mx, _mn = max(_kv, key=lambda x: x[1]), min(_kv, key=lambda x: x[1])
t(30, LZ + 112, '⑤ ⭐⭐ <tspan font-weight="700">最后看 KV 那一列：从 %s 到 %s，'
                '整整 %d 倍。</tspan>'
                '而这 %d 倍<tspan font-weight="700">不是一个旋钮拧出来的</tspan> ——&#160;'
                'MHA→GQA 砍头数（576→40）、MLA 改压缩（40→8.4）是旋钮①；'
                '线性把大部分层的 KV <tspan font-weight="700">直接删成零</tspan>（8.4→1.0）是旋钮③；'
                'CSA／HCA 存压缩池（→0.7）是旋钮②。'
                '<tspan font-weight="700">三个旋钮各贡献了一段。</tspan>'
                % (kv_fmt(_mx[1]), kv_fmt(_mn[1]),
                   round(_mx[1] / _mn[1]), round(_mx[1] / _mn[1])),
  fill="#174ea6")

# ⭐⭐⭐ 2026-09-07 现场一句「10M 的上下文是什么鬼」逼出来的一条 ——
#    而且它比任何一个具体数字都重要，因为它管着**整列怎么读**。
# ⛔ 这一列的每个数都是从各家 config 的 `max_position_embeddings` 读的，
#    而那个字段各家含义并不一样：有的是验证过的上下文，有的是理论容量。
#    MiniMax-01 就是后者 —— config 写 10,240,000，官方只声称训练 1M、外推 4M。
# ⛔ 一行写不下就拆两行 —— 别指望缩字号，那是把「读不清」换成「看不见」。
t(30, LZ + 130, '⑥ ⛔⛔ <tspan font-weight="700">「上下文」这一列报的是'
                '<tspan style="text-decoration:underline">声明</tspan>，不是'
                '<tspan style="text-decoration:underline">能用</tspan>。</tspan>'
                '数来自各家 config 的 max_position_embeddings，而这个字段各家含义并不一样'
                '——&#160;MiniMax-01 那格 config 写着 <tspan font-weight="700">10,240,000</tspan>，'
                '而官方只声称<tspan font-weight="700">训练 1M、外推 4M</tspan>。',
  fill="#174ea6")
t(46, LZ + 148, '<tspan font-weight="700">声明和能用之间还隔着一整个 benchmark 的落差</tspan>：'
                '小米自己的模型卡就写着 V2-Pro「到 1M 时塌到 0.00」，而 V2.5-Pro 在 1M 也只有 0.37／0.62。'
                '⭐ <tspan font-weight="700">看到「支持 N 万上下文」，先问是谁、在什么任务上、测出多少分。</tspan>',
  fill="#174ea6")

# ══════════ 落点带 ══════════════════════════════════════════════════
FY, FH = BY + BH + 14, 226
box(0, FY, W, FH, "#fef7e0", OR)
t(16, FY + 24, '⭐ 全图落点：三家公司，各自把旋钮拧了一遍 ——&#160;'
               '而且拧的过程全写在公开的 config 里', "svglbl", BR, size=13)
# ⭐ 三条轨迹并列，才看得出「这不是某一家的偶然选择」。
#    左边留 118px 给公司名，三行对齐。
for i, (who, arc) in enumerate((
        ('MiniMax',
         '线性（M1）→&#160;<tspan font-weight="700">退回全注意力</tspan>（M2）'
         '→&#160;<tspan font-weight="700">稀疏</tspan>（M3）'
         '——&#160;三次转向，每次都公开写了为什么'),
        ('腾讯混元',
         'Hy3 <tspan font-weight="700">80 层纯 GQA-8</tspan>（连线性都没上）'
         '→&#160;Hy4 <tspan font-weight="700">78 层全 Gated DSA</tspan>'
         '——&#160;<tspan font-weight="700">跳过线性那一支，直接进稀疏</tspan>'),
        ('智谱 GLM',
         '5 上 DSA →&#160;5.2 加 <tspan font-weight="700">IndexShare</tspan> '
         '→&#160;5.3-Flash <tspan font-weight="700">第一次线性＋稀疏同锅</tspan>'
         '——&#160;半年走完三步'))):
    y = FY + 48 + i * 19
    t(16, y, who, fill=BR, bold=True)
    t(118, y, arc, fill=BR)

t(16, FY + 122, '⭐⭐ 再看一眼第二列：这次补完混元和 GLM，多出来一条原先看不见的线',
  "svglbl", "#174ea6", size=12)
t(16, FY + 142, '混元 Hy4 的 <tspan font-weight="700">IndexCache</tspan> 和 '
                'GLM-5.2 的 <tspan font-weight="700">IndexShare</tspan> 是同一个想法：'
                '两家的 <tspan font-weight="700">indexer_types</tspan> 都是 '
                '<tspan font-weight="700">full, shared, shared, shared</tspan> 四层一循环 '
                '——&#160;每 4 层只有 1 层自己算索引。', fill="#174ea6")
t(16, FY + 160, '⭐ 所以稀疏的<tspan font-weight="700">第二阶段</tspan>优化，'
                '已经不是「让每个 query 少看几块」，而是'
                '<tspan font-weight="700">「别每层都重新算一遍该看谁」</tspan>'
                '——&#160;<tspan font-weight="700">索引本身变成了新的开销。</tspan>'
                '这是 2026 年才冒出来的一层。', fill="#174ea6")

box(16, FY + 172, W - 32, 1, GY, GY, 0)
t(16, FY + 194, '⚠️ <tspan font-weight="700">一条必须带上的限定：稀疏那一档的账，'
                '纸面上拿不到</tspan>', "svglbl", RD, size=12)
t(16, FY + 212, 'MiniMax M3 的 GGUF 发布说明写着「MSA 不支持 →&#160;推理退回稠密」'
                '——&#160;<tspan font-weight="700">纸面省下的 FLOP，要 kernel 跟上了才算数</tspan>。'
                '这一栏的每一个「层内稀疏」，都该配一句「在哪个引擎上」。', fill=GY)

p.append('</svg>')
# ── 回填 svg 开标签：高度按真实落点算，不写死 ──────────────────────
p[_HDR] = ('<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="'
           'Attention 编年史：2014 年注意力作为 RNN 的补丁出现，2017 年 Transformer 把 RNN 拿掉，'
           '此后分成四支演化；下半是各家开源模型的混合配比条形图，'
           '含混元 Hy3／Hy4 与 GLM-5 系列">' % (W, FY + FH + 12))
io.open('fig3-chronicle.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-chronicle ok')
