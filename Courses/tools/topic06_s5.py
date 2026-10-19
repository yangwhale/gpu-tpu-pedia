# -*- coding: utf-8 -*-
"""专题六 · 第五节「单卡里的两块硬骨头：decode 注意力与 MoE」。图全在 topic06-fig-decode.py。"""

BODY = '''
  <p class="lead">座位坐满后，权重一趟已分给几百人，每一步本身却还慢。一张卡上的 decode 有两块最难啃：注意力读的是每个人自己的 KV（5.1、5.2），MoE 的每个专家只分到几行（5.2）。<b>两块都是搬运问题。</b>最后补一条小 AllReduce（把各卡的部分和加起来再发回每张卡，5.3），它卡的是延迟。</p>

  <h3 id="s5-1">5.1　注意力 kernel：攒 batch 救不了，只能把搬运做顺</h3>
  <p>长上下文除了占座位（第四节），每人每步要读的 KV 也更长，且各读各的，攒 batch 摊不薄。</p>
  <p>decode 注意力每读 1 字节只做 G 次运算（G ＝ 几个查询头共用一个 KV 头），推导见<a href="topic-03.html#s3-1">专题三 §3.1</a>。Llama-3.1-70B 64 个查询头配 8 个 KV 头，G＝8，v7 的屋脊点 313 是它的近 40 倍。</p>
__FIG_S5_AMORTIZE__
  <p>batch 小、序列长时，注意力的活只按 batch 和头分给计算单元，份数太少，单元大半闲着。
    Flash-Decoding（斯坦福 CRFM 博客，2023-10）把每条序列的 KV 切成几段同时读，最后合并（<a href="#s7-1">7.1</a> 有图）。</p>
  <p>名字借自 FlashAttention（0.1）。博客在 CodeLlama-34B 单请求上报，序列很长时 Flash-Decoding 最多是旧实现的 8 倍。</p>

  <p><b>一个 kernel 装下 prefill 和 decode。</b>chunked prefill 让一步里混着 prefill 和 decode。RPA（Ragged Paged Attention，2025 年进 vLLM 的 TPU 后端）把两者放进同一个注意力 kernel（arXiv 2604.15464 §3.4–3.5）。GPU 上做同一件事的是 FlashInfer（MLSys 2025），已是 SGLang、vLLM 的注意力后端之一。</p>
__FIG_S5_RPA__
  <p>RPA 在 TPU7x 上跑 Llama 3 8B，decode 最多用到峰值带宽的 86%，prefill 最多用到峰值算力的 73%，三招拆解见<a href="topic-03.html#s9-2">专题三 §9.2</a>。图右数据出自 Google 开发者博客的公开调优。</p>

  <p><b>KV 一页页怎么搬上芯片。</b>MLP 和专家那边，一批里的 token 不分是谁的，都是矩阵的一行，权重搬一次全员算完。注意力不行：每条序列的 KV 是它自己的私货，并发 512 就是 512 份，谁也替不了谁。所以注意力 kernel 能做的只剩一件：<b>每个字节只搬一次，搬的时候别让计算单元等。</b></p>
__FIG_S5_KVLOAD__
<figure class="fbox fwide" id="anim-kvload">
<video src="media/topic06-kvload.mp4" autoplay loop muted playsinline
       aria-label="动画：左边是显存里的 KV 页池，A、B、C 三条序列的页散在 16 个格子里，下面是三张页表。中间是片上两个槽，右边是四个查询头和一块 K、V，底部是一条两行时间轴：上行搬，下行算。开头按 A 的页表取两页，拼成一块飞进槽 1。之后每一步，一个槽在算、另一个槽同时在装下一块：算 A 块 1 时 A 块 2 飞进槽 2；算 A 块 2 时 A 还没算完，B 块 1 已经飞进槽 1；B 块 2 只有 2 个 token、C 只有 3 个 token，就只搬这么多。每算一块，四个查询头一起亮一下，共用这一块 K、V。时间轴上，搬的那一行总比算的那一行早一格。最后打出一句：每个字节只搬一次，算的时候下一块已在路上。"></video>
<figcaption>盯住两个槽：一个在算，另一个总在装下一块。
  <span class="sub">（30 秒无声循环，Manim 渲染。规则照 RPA 论文的流水线；页、块、长度是示意。）</span></figcaption></figure>
  <p>RPA 的做法（arXiv 2604.15464 §3.3、算法 1）拆开是五件事：</p>
  <ul>
    <li><b>按页表取。</b>一条序列的 KV 散在页池里不连号的页上。kernel 按页表把几页取来，在片上 VMEM 拼成连续的一块，再交给计算。</li>
    <li><b>只搬有效长度。</b>长短不齐不靠补齐：最后一页没写满、这一块只有 3 个 token，就只搬 3 个（§3.4）。</li>
    <li><b>边算边搬。</b>TPU 只有一条指令流，并行全靠异步 DMA。片上留两个槽，算这一块时下一块已在路上；一条序列算完，接着取下一条的第一块，交界处不停。</li>
    <li><b>一块喂一组头。</b>同组的 G 个查询头摞成 G 行，对同一块 K、V 算。KV 搬上来一次，用 G 次（§3.1.2）。</li>
    <li><b>新 KV 顺路写回。</b>本步新算的 K、V 拼进同一块参与计算，同时异步写回页池，不另起一趟（§3.3）。</li>
  </ul>
  <p>拿并发 512、每条 2K 到 8K 算一笔：平均 5K 的话，decode 一步每层要把约 256 万个 token 的 KV 过一遍（本课推导：512 × 5,000）。这个量躲不掉，能抠的是每趟搬运的效率。两个办法：页调大，查页表的次数就少，向量单元不再频频停下来等索引（5.1 图右，428 → 283 µs）；K 和 V 并进一趟搬（5.1，论文 §3.1.3）。</p>
  <p>prefill 是另一回事：一块查询有几百行，同一块 KV 被这么多行共用，每读一字节的运算跟着多出几百倍（本课推导：G × 查询行数），瓶颈回到算力，走的是 FlashAttention 的路数。所以 RPA 给 decode 和 prefill 各调一套块大小：decode 调到吃满带宽，prefill 调到矩阵单元不空（§3.5）。</p>
  <p><b>稀疏注意力的 prefill：挑出来的，反而按稠密算。</b>DSA 每个查询只挑 2048 个位置（<a href="#s4-1">4.1</a>）。decode 一次一行，照着挑的去 gather 正合适；prefill 几千行查询各挑各的，挑中的大量重叠，gather 会把同一行搬上千遍。一篇公开工程博客在 TPU v7 上量到：为取不到 2048 个不同行，物化出约 419 万行，稀疏路径 96.3% 的设备时间花在 gather 上。</p>
  <p>改法是不搬：把每行挑中的位置写成一张位图（在 SparseCore 上做），整块做稠密注意力，没挑中的掩掉，结果与稀疏等价。这正对上面那句：prefill 吃的是算力，矩阵单元最擅长整块算。4K 输入的首 token 时延 1338.81 → 729.04 ms。KV 太长时稠密不划算，按成本模型在设备上选路，原文的分界是 KV ≤ 9216。</p>
  <p><b>位图还能跨层共用。</b>挑位置不必每层都做：Hy4 的 78 层里只有 21 层算 top-k，其余层沿用前面算好的。位图跟着复用，高并发混合步里单这一项吞吐 +5.6%。</p>

  <p><b>布局：窄维被补成整块。</b>TPU 的显存按 tile（图里的「标准箱」）存：最内层（内存里连续排的那一维）和次内层都要补齐，fp8 的次内层补到 32 行。</p>
  <p>KV 头只有 1–2 个，若头这一维落在这两维上，就可能被补成整块，多出的全是空位。</p>
__FIG_S5_TILE__
  <p>RPA 的解法是加一个打包维、把 K 和 V 沿头维并起来，逼编译器选最小的 tile（论文 3.1 节）。</p>

  <h3 id="s5-2">5.2　V3 专用：MLA 一个阶段一套算法，MoE 全算比排序便宜</h3>
  <p><a href="topic-03.html#s5-2">专题三 §5.2</a> 推过「吸收」：把投影并进查询，所有头直接对同一条压缩后的潜向量算，K/V 不展开。那是 decode 用的，prefill 呢？</p>
__FIG_S5_MLA__
  <p>decode 缺带宽，用吸收；prefill 缺算力，要的是少算。每个头对每个历史 token，吸收要和 576 维的潜向量做点积、再按其中 512 维加权求和，共 1,088 次乘加；展开后每头 K 192 维、V 128 维，共 320 次，吸收是展开的 3.4 倍（V3 config 推导）。展开多读的字节，prefill 有富余带宽兜着。</p>

  <p><b>decode 的 MoE：全算比排序便宜。</b>1.2 算过，decode 时每个专家平均只分到几行；与其按专家把 token 排序分组，不如让所有 token 都过一遍，没选中的乘 0。</p>
__FIG_S5_DENSE__
  <p>搬行本身也能这样换：tpu-inference 2026 年 5 月合入的一项改动，在 batch 小时把排序和还原换成「one-hot 矩阵 × 隐状态」，一次规整的矩阵乘代替按下标取行。Qwen3.5 并发 64 实测总吞吐 +14.5%（1K 进 8K 出）、+7.3%（8K 进 1K 出）；合入后默认关，要用环境变量开（出处见第十六节台账）。</p>

  <h3 id="s5-3">5.3　小 AllReduce 是延迟问题</h3>
__FIG_S5_ALLREDUCE__
  <p>消息小，时间几乎全是固定延迟，加带宽救不回来（arXiv 2607.16100）。</p>

  <p class="landing">⭐ 从 2022 年的 FlashAttention 到 2025 年进 vLLM 的 RPA，单卡 kernel 一直在做一件事：<b>「少搬」从布局开始。</b>补齐的空位、展开的 K/V、搬来搬去的行都是没人要的字节，先去掉它们，再谈量化和加卡。留下的问题：一张卡上每个专家只分到几行，batch 攒得再大也喂不饱（第六节）。</p>
'''

FIGS = {
    "__FIG_S5_KVLOAD__": ("fig-s5-kvload", "fig6-s5-kvload.svg", "topic06-fig-decode.py",
        '<b>先看中间两个槽：一个在算，一个在装下一块；再看时间轴，搬总比算早一格。</b>'
        '<br><em>页、块、序列长度、G＝4 都是示意；规则照 RPA 论文。</em>'),
    "__FIG_S5_RPA__": ("fig-s5-rpa", "fig6-s5-rpa.svg", "topic06-fig-decode.py",
        '<b>左边看箭头：一批切成三段，各走一份版本；右边比两根条。</b>'
        '<br><em>段的切法和请求数是示意；页的格数只表示比例。</em>'),
    "__FIG_S5_AMORTIZE__": ("fig-s5-amortize", "fig6-s5-amortize.svg", "topic06-fig-decode.py",
        '<b>柱顶的数越降越慢：权重摊薄了，紫段摊不薄。</b>'
        '<br><em>只记读显存的字节，没算激活值和 logits（给词表里每个字打的分）。</em>'),
    "__FIG_S5_TILE__": ("fig-s5-tile", "fig6-s5-tile.svg", "topic06-fig-decode.py",
        '<b>同一份 KV，看三种摆法各占多大地方。</b>'
        '<br><em>按默认 tile 推导；编译器对极窄的数组有时改用更小的 tile，见出处。</em>'),
    "__FIG_S5_MLA__": ("fig-s5-mla", "fig6-s5-mla.svg", "topic06-fig-decode.py",
        '<b>上下两栏的头一样多，差别只在箭头从哪条数据出发。</b>'
        '<br><em>乘加按 2 次运算计，bf16；没算把查询吸收进去、把输出投影回来的那两次矩阵乘。</em>'),
    "__FIG_S5_DENSE__": ("fig-s5-dense", "fig6-s5-dense.svg", "topic06-fig-decode.py",
        '<b>比两根条：全算短于读权重，就藏得住。</b>'
        '<br><em>DeepSeek-V3 一个路由专家，TPU v7 单 device，按峰值带宽和峰值算力推导。</em>'),
    "__FIG_S5_ALLREDUCE__": ("fig-s5-allreduce", "fig6-s5-allreduce.svg", "topic06-fig-decode.py",
        '<b>比灰条和红条：就算做到硬件下限，带宽项也只是零头。</b>'
        '<br><em>4 张 GB200，小消息平均延迟（论文实测）；带宽项为本课推导，对数轴。</em>'),
}

NUM_LOCK = []
