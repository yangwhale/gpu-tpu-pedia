# -*- coding: utf-8 -*-
"""专题三 · 图 A —— Attention 编年史**时间轴**（只剩上半，下半的模型表已改成 HTML）。

⛔⛔ **2026-09-07 拆分。** 现场要求「表头你点哪个就按哪个排序」，而 **SVG 的文字是死的**：
   点不了、Ctrl+F 搜不到、复制不出来。39 行的参照表本来就不该是一张图。于是拆成两半：

     · 本文件            →&nbsp;**时间轴**，那是真·图，留 SVG
     · `topic03-table-models.py` →&nbsp;**模型表**，那是真·表，出可排序 HTML
     · `topic03_models.py`       →&nbsp;两边共用的**唯一一份数据**

⭐ 判据：**「图」和「表」是两种东西。** 图讲关系与形状，表供查与比。
   把表画成图，等于主动放弃搜索、复制、排序、无障碍这四样，只换来像素级排版 ——
   而排版对一张参照表本来就不是优点。


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

⭐⭐⭐ **2026-09-07 第十轮：补 DeepSeek-V3，顺带证出整张表最反直觉的一条。** 原话：

    「DeepSeek v2 它虽然模型很小，但是它的 KV cache 算出来跟 DeepSeek v3 差不多大。
      是不是因为它们的 MLA 的这个 lora 的 rank 都是一样的，然后层数还是一个 60
      一个 61，所以大差不差？……能不能把 DeepSeek v3 也加进去？
      因为毕竟我们第一讲是拿 DeepSeek v3 讲的，所以这个锚点它很重要。」

   · **推理完全正确，而且是从表上现推出来的。** V3 config 核实：61 层、
     `kv_lora_rank: 512`、`qk_rope_head_dim: 64`，跟 V2 只差一层。
     **参数 236B → 671B 差 2.8 倍，KV 只差 1.7%（8.4 → 8.6 GiB）。**
   · **加 V3 这一行还有独立价值**：专题一里提到它 21 次，是那一讲的锚点模型 ——&nbsp;
     跨专题共用同一个模型当参照，读者能把两讲的账对起来。

   ⭐⭐ **由此落点⑦：MLA 之后，KV cache 跟「模型多大」脱钩了。**
   MLA 的 KV 只跟 **层数 × (kv_lora_rank ＋ rope 维)** 走 ——&nbsp;
   跟专家多少、hidden 多宽、总参多大一点关系都没有。
   MHA 时代 KV 是**跟着模型一起长**的（看 GPT-3 那行 576 GiB），
   **这条链在 MLA 这里被剪断了。**

   ⭐ 顺带还看得见第三件事：**V3.2 在 V3 上加了 DSA，KV 一个字节没省**（都是 8.6）。
   跟 MiniMax M3 那行是同一条结论：**稀疏省 FLOPs，不省 KV。**

⭐⭐ **2026-09-07 第十一轮：把 GPT-3 那条折成上下两段。** 原话：

    「MHA 的 KV cache 太长了，影响后面的按比例对比，要不做成两折叠吧，上下两条那种。」

   ⭐⭐⭐ **这次改动的真正收益不是排版，是刻度。**
   折行之后满刻度可以从「一行 400px」改成「**两行 800px**」——&nbsp;
   GPT-3 正好占满两行，而**其余每一条都长了一倍**：
   Llama2 105→**211**、MiniMax M2 93→**186**、GLM-5 系 55→**110**、
   Kimi Linear 16→**33**、DeepSeek-V4 14→**27**。
   **现代模型之间的差距这才真正看得出来** ——&nbsp;之前它们全挤在 14–105 那一段。

   📌 实现上：两段各 8px ＋ 2px 间隔 ＝ 18px，**跟普通条一样高**，
   所以不用为它单独加行高；而「两条」这个形状本身就说明了它超长，不用额外标注。
   ⛔ 只有 GPT-3 会折（次大的 Llama2 才 211px），所以这条分支现在只走一次 ——&nbsp;
     但逻辑是通用的，以后再加更长的行也自动折。

⛔⛔ **2026-09-07 第十二轮：刻度从开方换回线性（真实比例），多段折行。** 原话：

    「你这两条一边长太不真实了，你就多搞几条也行，按照真实比例下的长度来。」

   **这个批评一针见血。** 上一版用开方刻度、满刻度＝两行，于是 GPT-3 **恰好**
   等于 2×满宽 ——&nbsp;两条一模一样长。那个「正好」不是数据长成那样，
   **是刻度凑出来的**。⭐ 教训：**当一个图形呈现出过于整齐的巧合，
   先怀疑是自己的刻度在说话，而不是数据在说话。**

   改成**线性 · 一整行 ＝ 60 GiB**：
   · GPT-3 折 **10 条**，末段 240px 是**自然残段**，一眼看得出没被凑
   · 那一行本身比别的行高出一截 ——&nbsp;**这不是排版事故，这就是结论**
   · 现代那批铺开到 5–267px（Llama2 267、M2 207、GLM-5 系 73、V4 5）

   ⚠️ **线性的代价要说清楚**：1 GiB 以下那几家（Kimi Linear 1008 MiB、
   DeepSeek-V4 697 MiB）在真实比例下就是几个像素，**互相之间没法用长度比** ——&nbsp;
   得读数字。但这本身也是一条结论：**它们相对 MHA 就是「几乎为零」。**

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
W = 1560
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
    # ⛔ 2026-09-07：这条泳道原先**漏了整条 Mamba/SSM 主干**，而它不是"另一支"，
    #   是这一支的**直系祖宗** ——&nbsp;Gated DeltaNet 那篇论文的标题就叫
    #   《Gated Delta Networks: **Improving Mamba2** with Delta Rule》
    #   （arXiv 2412.06464）。千问 Qwen3-Next／Qwen3.5 用的就是 GDN。
    # ⭐ 所以 Mamba 必须进这条泳道，而不是另开一条。
    ("③ 换一套数学（线性注意力 · SSM 同源）", PU, "#f3e8fd", [
        (2020, "线性 Transformer"), (2021, "DeltaNet"),
        (2023, "Mamba"),
        (2024, "Mamba-2（SSD）· 可并行 DeltaNet · GDN"),
        (2025, "KDA · Lightning"),
        (2026, "Mamba-3 · Gated DeltaNet-2"),
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

# 时间轴面板的底框（高度要等泳道排完才知道，所以回填）
p[_PANEL] = ('<rect x="0" y="%d" width="%d" height="%d" rx="8" fill="#f8f9fa" '
             'stroke="#dadce0" stroke-width="1"/>' % (TY, W, TH))
# ══════════ 落点带 ══════════════════════════════════════════════════
FY, FH = TY + TH + 16, 226
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
