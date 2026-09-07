# -*- coding: utf-8 -*-
"""专题三 · §零「起点：RNN」的三张图。

⭐ **为什么专题三要从 RNN 讲起。** 2026-09-07 现场定的顺序：

    「从头到尾一个一个讲那个 Attention 的变体，从头开始讲，讲完一个是一个深入的。
      从第一个朴素的 Attention 开始 ——&nbsp;不对，应该先讲 RNN，讲完 RNN 再讲。
      你先把 RNN 的图示给我弄出来，然后把它讲明白，让人一目了然就知道 RNN 是
      什么原理、怎么计算、怎么跟硬件结合、它为什么是串行的，然后它的痛点是啥。」

⭐⭐ **这三张图各自只回答一个问题，别混。**

     · `fig3-rnn-unroll` →&nbsp;**它是什么、怎么算**（折叠着看 ／ 展开着看）
     · `fig3-rnn-hw`     →&nbsp;**它为什么在硬件上快不起来**（权重被读 n 次）
     · `fig3-rnn-pain`   →&nbsp;**三个痛点各自通向后面的哪条路**

   ⛔ 别把「怎么算」和「为什么慢」画进同一张。第一张是**语义**、第二张是**性能**，
     两种坐标系。混在一起的后果不是难看，是读者说不清自己在看哪一层。

════════════════════════════════════════════════════════════════════
📌 出处（全部一手核过，2026-09-07）
════════════════════════════════════════════════════════════════════

· **Elman 1990**《Finding Structure in Time》(Cognitive Science 14, 179–211)
  —— 原文对 context units 的描述：「activations are copied from hidden layer
  to context layer **on a one-for-one basis, with fixed weight of 1.0**」，
  且「Dotted lines represent trainable connections」。⭐ 所以那条回抄的边
  **不是学出来的**，是硬接线。XOR 那组实验的规模也是原文里的：
  6 输入 / 20 隐藏 / 6 输出 / 20 context。

· **Bengio, Simard, Frasconi 1994**《Learning long-term dependencies with
  gradient descent is difficult》(IEEE TNN 5(2)) —— 梯度消失的出处。

· **Hochreiter & Schmidhuber 1997**《Long Short-Term Memory》—— 门控的出处。

· **Sutskever, Vinyals, Le 2014**《Sequence to Sequence Learning with Neural
  Networks》(arXiv 1409.3215) —— 摘要原话是把输入序列映到「a vector of a
  **fixed dimensionality**」。⛔ **「固定长度向量是瓶颈」这句话不是这篇说的**，
  是 Bahdanau 2014 反过来指出的 ——&nbsp;别把后人的批评安到原作者头上。

· **Vaswani et al. 2017**《Attention Is All You Need》(arXiv 1706.03762)
  —— ⭐⭐ 引言原话：「This **inherently sequential** nature precludes
  parallelization within training examples, which becomes critical at longer
  sequence lengths, as **memory constraints limit batching across examples**.」
  以及 Table 1（复杂度 / 串行步数 / 最长路径）三列，是本节最硬的锚点：
  **这笔账是 Transformer 作者自己算的，不是我们归纳的。**

· **NVIDIA《Recurrent Layers User's Guide》** —— 硬件侧的一手说法：
  ① 每个门每一相「is equivalent to a GEMM with **one dimension of one**」
     ——&nbsp;官方口径就是「一个维度是 1」，也就是矩阵乘向量；
  ② 「We can combine these GEMMs over the minibatch size, **but not over
     different sequence steps**」——&nbsp;⭐ 这一句就是「为什么串行」的硬件版；
  ③ LSTM 四个门可以拼成一个 GEMM，GRU 三个，ReLU/Tanh 一个。

· **Martin & Cundy 2018**《Parallelizing Linear Recurrent Neural Nets Over
  Sequence Length》(arXiv 1709.04057, ICLR'18) —— 「**只有线性**的循环依赖
  才能用 parallel scan 在序列长度上并行」。⭐ 这是后面 Mamba／线性注意力那
  一整支的地基，放在这里做伏笔。

════════════════════════════════════════════════════════════════════
⭐⭐ 第二张图那两个数是**算出来的**，推导链写在图上，别只写结论
════════════════════════════════════════════════════════════════════

   算术强度 ＝ FLOPs ÷ 从 HBM 搬进来的字节数。取 LSTM 的循环支路、
   隐藏维 d、batch B、BF16：

     · 权重块 [4d × d]，搬进来 4d·d·2 字节
     · 跟 [d × B] 相乘，2 · 4d · d · B FLOPs
     · 相除 →&nbsp;**算术强度 ＝ B**，⭐ **跟 d 没关系，只等于 batch size**

   而 Transformer 训练时同一块权重只搬一次，配的是 [d × (n·B)]，
   算术强度 ＝ **n·B**。⭐ **两者差了整整一个 n。**

   拐点（ridge point）用 TPU v7 算：官方每芯片 FP8 **4614 TFLOP/s**，
   BF16 取一半 ＝ **2307**；官方 HBM 带宽 **7.37 TB/s**。
   2307 ÷ 7.37 ＝ **313 FLOP/byte**。
   ⭐ 于是一句很扎人的结论：**要把 v7 喂饱，LSTM 的 batch 得开到 313 以上。**
   ⛔ 而 Vaswani 那句话说的正是这个的反面：**长序列时显存又不让你把 batch 开大。**

📌 **持久化 RNN（persistent RNN）**是当年绕开这条的办法：把权重钉在片上不搬。
   ⭐ 这跟后面 FlashAttention「不让中间结果落 HBM」是**同一个念头**，
   隔了七年在另一个算子上又出现了一次 ——&nbsp;正文里要把这条线连上。
"""
# ══════════════════════════════════════════════════════════════════
# ⭐⭐⭐ 2026-09-08 全部重画。现场：「怎么画得那么简陋呢？能不能画得有质感一点？」
#
# ⭐ 「质感」不是加阴影加渐变，是**对齐专题二那套已经理顺的视觉语言**。
#    照着专题二的图逐条拆，它的质感来自六样东西 ——&nbsp;下面全部做成了基元：
#
#      ① **图例条**：颜色一上来就有词典，读者不用猜哪个色是什么
#      ② **面板套面板**：外框 → 带标题栏的内面板 → 内容块，层次一眼分明
#      ③ **盒子两行字**：主标签 ＋ 一句说明。光秃秃一个词的盒子是廉价感的主因
#      ④ **矩阵块有纹理**：权重画成带网格的块，而不是一个空心矩形
#      ⑤ **高亮列 / 行**：用一条淡色竖带圈出「整张图的差别在这儿」
#      ⑥ **底部落点带 ＋ 出处行**：⭐ 绿 / ⚠️ 黄 / ⭐⭐ 蓝，再加一行 📌 灰字出处
#
# ⛔ 别再画「一个矩形 ＋ 居中一个词」——&nbsp;那正是上一版简陋的原因：
#    每个元素只承载一个信息，于是图上信息密度还不如一句话，
#    读者付出了「看图」的成本却没拿到「图才给得了」的东西。
# ⭐ 判据：**图里每一个盒子都该回答一个问题，而不是标一个名字。**
# ══════════════════════════════════════════════════════════════════

from topic03_draw import (Fig, wpx, _sz,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, LINE, BG2)


# ══════════════════════════════════════════════════════════════════
# 图一：它是什么、怎么算
# ══════════════════════════════════════════════════════════════════
def fig_unroll():
    W = 1400
    f = Fig(W, "RNN 的两种画法：左边是 Elman 1990 的折叠画法，隐藏层原样抄一份到 "
               "context 层；右边沿时间展开成一条链，每一步吃当前输入和上一步的状态")
    f.marks = set()
    y = f.header(
        'RNN ——&#160;把「之前发生了什么」塞进一个固定大小的盒子，一步一步往下传',
        '⭐ 全部设计就这一句。<tspan font-weight="700">后面所有的好处、'
        '所有的痛点，都是这一个决定的后果。</tspan>',
        [(GR, "输入"), (RD, "状态 h ——&#160;整个模型的记忆"), (BL, "输出"),
         (PU, "原样抄写，不训练")])

    # ── 左：折叠 ────────────────────────────────────────────────
    PW = 400
    PT = f.panel(0, y, PW, 300, "① 折叠着看", tint="#f1f3f4",
                 sub="Elman 1990 原始画法", tag="Cognitive Science 14")
    cx = 96
    f.cell(cx, PT + 16, 150, 42, "输出 y", "y = W_y · h", BL, "#e8f0fe")
    f.cell(cx, PT + 92, 150, 44, "隐藏 h", "[d] ——&#160;固定大小", RD, "#fce8e6")
    f.cell(14, PT + 186, 136, 42, "输入 x", "这一步的词", GR, "#e6f4ea")
    f.cell(186, PT + 186, 176, 42, "context 层", "上一步的 h", PU, "#f3e8fd")
    f.line(cx + 75, PT + 92, cx + 75, PT + 62, RD)
    f.line(82, PT + 186, cx + 44, PT + 140, GR)
    f.line(258, PT + 186, cx + 108, PT + 140, PU)
    f.path("M %d %d L %d %d L %d %d" % (cx + 150, PT + 112, 380, PT + 112,
                                        380, PT + 182), PU, 1.4, "4 3")
    f.t(252, PT + 76, "原样复制一份", PU, True, _sz(11))
    f.t(252, PT + 92, "固定权重 1.0，不训练", GY, size=_sz(11))
    f.t(14, PT + 250, '⭐ 那条<tspan font-weight="700">虚线不是学出来的</tspan>'
                      '——&#160;原文写死「one-for-one basis,', GY, size=_sz(11))
    f.t(14, PT + 266, 'with fixed weight of 1.0」。<tspan font-weight="700">'
                      '整个循环就靠这一条硬接线。</tspan>', GY, size=_sz(11))

    # ── 右：展开 ────────────────────────────────────────────────
    QX = 420
    QT = f.panel(QX, y, W - QX, 322, "② 展开着看",
                 sub="它其实就是一条链，每一格都得等前一格算完")
    n, step, x0 = 5, 182, QX + 40
    # ⛔ 上一版 t 标签画在 QT+24，而 y 盒子从 QT+12 起 —— **标签被盒子盖住了**。
    #   ⭐ 这类错在代码里完全看不出来，只有渲染出来量一眼。整块内容下移 22px。
    f.spot(x0 - 14, QT + 96, n * step - 26, 46, "#fce8e6")     # ⑤ 高亮那一行
    for i in range(n):
        x = x0 + i * step
        k = str(i + 1) if i < n - 1 else "n"
        f.t(x + 57, QT + 18, "t = %s" % k, GY2, size=_sz(11), anchor="middle")
        if i == n - 1:
            f.t(x - 36, QT + 122, "…", GY2, True, 17, "middle")
        f.cell(x, QT + 174, 114, 36, "x%s" % k, None, GR, "#e6f4ea")
        f.cell(x, QT + 100, 114, 38, "h%s" % k, None, RD, "#fce8e6")
        f.cell(x, QT + 34, 114, 36, "y%s" % k, None, BL, "#e8f0fe")
        f.line(x + 57, QT + 174, x + 57, QT + 144, GR)
        f.line(x + 57, QT + 100, x + 57, QT + 76, BL)
        # ⛔ 最后一格前面是跳过去的，画箭头会把「…」压在下面。
        if i and i < n - 1:
            f.line(x - step + 114, QT + 119, x - 6, QT + 119, RD, 2.1)
    f.t(x0 + 126, QT + 112, "上一步的 h", RD, True, _sz(11))
    f.t(QX + 16, QT + 236,
        '⛔ <tspan font-weight="700">红底那一行就是全部问题的根源</tspan>：'
        'h3 要用 h2，h2 要用 h1 ——&#160;一百万步就得排一百万轮。', GY, size=_sz(12))
    f.t(QX + 16, QT + 258,
        '⭐ 而<tspan font-weight="700">竖着的绿、蓝箭头彼此不相干</tspan>'
        '——&#160;能并行的方向一直都在，'
        '<tspan font-weight="700">被卡住的只有横着这一个。</tspan>', GY, size=_sz(12))

    yy = y + 322 + 16
    yy = f.band(yy, "info", "拆开一格看：里面就是两个矩阵乘、一个加法、一个非线性", [
        'h<tspan baseline-shift="sub" font-size="10">t</tspan> ＝ '
        'f( W<tspan baseline-shift="sub" font-size="10">h</tspan> · '
        'h<tspan baseline-shift="sub" font-size="10">t−1</tspan> ＋ '
        'W<tspan baseline-shift="sub" font-size="10">x</tspan> · '
        'x<tspan baseline-shift="sub" font-size="10">t</tspan> ＋ b )'
        '　　　y<tspan baseline-shift="sub" font-size="10">t</tspan> ＝ '
        'W<tspan baseline-shift="sub" font-size="10">y</tspan> · '
        'h<tspan baseline-shift="sub" font-size="10">t</tspan>'
        '　　<tspan fill="%s">形状：h 是 [d]，W'
        '<tspan baseline-shift="sub" font-size="10">h</tspan> 是 [d × d]</tspan>' % GY,
        '<tspan font-weight="700">d 就是它能记住的全部容量</tspan>'
        '——&#160;喂 10 个词还是 10 万个词，盒子一样大。',
        '⭐ LSTM／GRU 没改这个形状，只是把 f 换成几个门'
        '（<tspan font-weight="700">LSTM 三个门＋一个候选值 ＝ 四块矩阵；'
        'GRU 两个门＋一个候选 ＝ 三块</tspan>）。<tspan font-weight="700">链还是那条链。</tspan>'])
    yy = f.src(yy + 18, 'Elman 1990《Finding Structure in Time》'
                        '——&#160;context units「copied … on a one-for-one basis, '
                        'with fixed weight of 1.0」；LSTM: Hochreiter &amp; Schmidhuber 1997；'
                        'GRU: Cho et al. 2014')
    f.save("fig3-rnn-unroll.svg", yy + 6)


# ══════════════════════════════════════════════════════════════════
# 图二：为什么在硬件上快不起来
# ══════════════════════════════════════════════════════════════════
def fig_hw():
    W = 1400
    f = Fig(W, "RNN 与 Transformer 训练在硬件上的差别：RNN 每个时间步都要把同一块权重"
               "从 HBM 搬一次，算术强度等于 batch size；Transformer 训练时同一块权重"
               "只搬一次，算术强度是 batch size 乘以序列长度")
    f.marks = set()
    y = f.header(
        '为什么 RNN 在加速器上就是跑不快 ——&#160;'
        '<tspan font-weight="700">同一块权重被搬了 n 次</tspan>',
        '⭐ 「串行」在硬件上的具体形态就是这个：'
        '<tspan font-weight="700">权重每一步重搬一遍，而每次只配一条 batch 那么窄的向量。</tspan>',
        [("#9aa0a6", "带网格的块 ＝ 从 HBM 搬进来的权重"), (BL, "配给它的激活"),
         (RD, "串行：一步一格"), (GR, "并行：整段一次")])

    AH = 152
    AT = f.panel(0, y, W, AH, "Ⓐ RNN", RD, "#fce8e6",
                 sub="每一个时间步，都要把同一块权重 W 从 HBM 重新搬进片上一次",
                 tag="训练和推理都一样", tint="#fadad6")
    for i in range(6):
        x = 34 + i * 178
        # ⛔ 上一版这些箭头从 x-44 起 —— 那个位置**落在上一格的蓝条里面**。
        #   ⭐ 连线的起点要按「上一个元素的右边界」算，不能按「下一个元素往左退多少」算。
        f.cell(x, AT + 22, 92, 52, "W", "[k·d × d]", RD, "#fff", grid=True)
        f.t(x + 104, AT + 54, "·", GY2, True, 15, "middle")
        f.box(x + 116, AT + 22, 26, 52, "#e8f0fe", BL, 4)
        f.t(x + 129, AT + 92, "[d×B]", BL, size=_sz(11), anchor="middle")
        f.t(x + 46, AT + 92, "第 %s 步" % (i + 1 if i < 5 else "n"),
            GY2, size=_sz(11), anchor="middle")
        if i:
            px = x - 178 + 142        # 上一格蓝条的右边界
            if i == 5:
                f.t((px + x) / 2.0, AT + 56, "…", GY2, True, 17, "middle")
            else:
                f.line(px + 8, AT + 48, x - 8, AT + 48, RD, 1.2)
    f.t(14, AT + 114,
        '⛔ NVIDIA 官方文档的说法是「a GEMM with '
        '<tspan font-weight="700">one dimension of one</tspan>」'
        '——&#160;<tspan font-weight="700">名义上矩阵乘，实际是矩阵乘向量。</tspan>',
        "#a50e0e", size=_sz(12))

    BY = y + AH + 14
    BT = f.panel(0, BY, W, 130, "Ⓑ Transformer 训练", GR, "#e6f4ea",
                 sub="同一块权重只搬一次，n 个位置一起喂进去", tint="#d7ecdc")
    f.cell(34, BT + 22, 98, 52, "W", "[k·d × d]", GR, "#fff", grid=True)
    f.t(146, BT + 54, "·", GY2, True, 15, "middle")
    f.box(166, BT + 22, 900, 52, "#e8f0fe", BL, 6)
    f.t(616, BT + 54, "[ d × (n · B) ]　——　n 个位置全在这一块里", BL,
        True, 13.5, "middle")
    f.t(14, BT + 96, '⭐ 一次搬运换来 n 倍的活干。'
                     '<tspan font-weight="700">这就是「用平方的计算量买完全的并行度」'
                     '那笔交易的硬件形态。</tspan>', "#0d652d", size=_sz(12))

    CY = BY + 130 + 14
    CT = f.panel(0, CY, W, 190, "算术强度 ＝ 算了多少次 ÷ 搬了多少字节", OR, "#fef7e0",
                 sub="它决定你是在等算力，还是在等内存", tint="#fbeecb")
    COLS = ((16, "算的是谁"), (250, "FLOPs"), (470, "搬进来的字节"),
            (650, "算术强度"), (770, "于是"))
    for x, lab in COLS:
        f.colhead(x, CT + 22, lab)
    f.line(16, CT + 32, W - 16, CT + 32, "#e6c86a", 1, arrow=False)
    for i, (a, b, c, d_, e, col) in enumerate((
            ("RNN 每一步", "2 · k·d · d · B", "k·d · d · 2", "B",
             "⛔ <tspan font-weight=\"700\">跟 d 无关，就等于 batch size</tspan>", RD),
            ("Transformer 训练", "2 · k·d · d · (n·B)", "k·d · d · 2", "n · B",
             "⭐ <tspan font-weight=\"700\">整整多了一个 n</tspan>", GR))):
        yy = CT + 56 + i * 26
        f.t(16, yy, a, col, True, _sz(12))
        f.t(250, yy, b, INK, size=_sz(12), mono=True)
        f.t(470, yy, c, INK, size=_sz(12), mono=True)
        f.t(650, yy, d_, col, True, 13)
        f.t(770, yy, e, col, size=_sz(12))
    f.t(16, CT + 112, '⭐ 那个 k（LSTM 是 4、朴素 RNN 是 1）'
                      '<tspan font-weight="700">上下一约就没了</tspan>'
                      '——&#160;算术强度等于 batch，跟门数、跟隐藏维都无关。'
                      '（激活的搬运比权重小两个数量级，略去。）', BR, size=_sz(11))
    f.line(16, CT + 122, W - 16, CT + 122, "#e6c86a", 1, arrow=False)
    f.t(16, CT + 130 + 14, '拐点：TPU v7 官方每芯片 FP8 '
        '<tspan font-weight="700">4614 TFLOP/s</tspan>，BF16 取一半 ＝ '
        '<tspan font-weight="700">2307</tspan>；官方 HBM 带宽 '
        '<tspan font-weight="700">7.37 TB/s</tspan>。2307 ÷ 7.37 ＝ '
        '<tspan font-weight="700">313 FLOP/byte</tspan>', BR, size=_sz(12))

    yy = CY + 190 + 16
    yy = f.band(yy, "bad", "两头堵死", [
        '<tspan font-weight="700">要喂饱一块 v7，batch 得开到 313 以上</tspan>'
        '——&#160;而 batch 是 RNN 唯一的算术强度来源。',
        '而 Vaswani 引言那句原话讲的正是另一头：'
        '<tspan font-style="italic">「memory constraints limit batching across examples」</tspan>'
        '——&#160;<tspan font-weight="700">序列一长，显存就不让你把 batch 开大。</tspan>'])
    yy = f.src(yy + 18, 'NVIDIA《Recurrent Layers User\'s Guide》'
                        '——&#160;「a GEMM with one dimension of one」、'
                        '「can combine these GEMMs over the minibatch size, '
                        'but not over different sequence steps」；'
                        'Vaswani et al. 2017 (arXiv 1706.03762) 引言')
    f.save("fig3-rnn-hw.svg", yy + 6)


# ══════════════════════════════════════════════════════════════════
# 图三：解码的时候，Transformer 又变回了这个形状
# ══════════════════════════════════════════════════════════════════
def fig_decode():
    W = 1400
    f = Fig(W, "三条时间线的对比：RNN 每步搬一次权重且串行；Transformer 训练时权重"
               "只搬一次且并行；但 Transformer 解码时又变回一步一个 token，"
               "每步搬一次权重，而且还要额外搬一路越来越长的 KV cache")
    f.marks = set()
    y = f.header(
        '⭐⭐ 而生成文字的时候，'
        '<tspan font-weight="700">Transformer 又变回了 1990 年那条链的形状</tspan>',
        '⛔ <tspan font-weight="700">它没有治好 RNN 的病，'
        '它只是把病从「训练」挪到了「推理」</tspan>——&#160;而且挪过去之后更重。',
        [(RD, "串行"), (GR, "并行"), (PU, "串行 ＋ 一路变长的 KV")])

    ROWS = (
        ("Ⓐ", "RNN", "训练和推理都一样", RD, "#fce8e6", "#fadad6",
         "串行", "每步搬：W", "算术强度 ＝ B", "一步一格，权重每步重搬", "s"),
        ("Ⓑ", "Transformer 训练", "它当年赢下来的地方", GR, "#e6f4ea", "#d7ecdc",
         "并行", "整段只搬一次：W", "算术强度 ＝ n · B", "一次搬运换 n 倍的活干", "p"),
        ("Ⓒ", "Transformer 解码", "你现在用的每个大模型", PU, "#f3e8fd", "#e8d5f7",
         "串行", "每步搬：W ＋ 越来越长的 KV", "算术强度 ≈ B",
         "串行回来了，还多背一个变长的 KV", "g"),
    )
    ROWH = 128
    for i, (tag, name, note, col, fill, tint, ser, mov, ai, tail, mode) in enumerate(ROWS):
        yy = y + i * (ROWH + 12)
        top = f.panel(0, yy, W, ROWH, "%s %s" % (tag, name), col, fill,
                      sub=note, tint=tint,
                      tag="⛔ 病在这里" if mode == "g" else None)
        f.t(16, top + 22, ser, col, True, 13)
        f.t(16, top + 42, mov, GY, size=_sz(11))
        f.t(16, top + 62, ai, col, True, _sz(12))
        if mode == "p":
            f.box(300, top + 16, 760, 40, "#fff", col, 6)
            f.t(680, top + 41, "t1 … tn　全部一起算", col, True, 13.5, "middle")
            f.t(680, top + 74, "搬 W 一次", GY, size=_sz(11), anchor="middle")
        else:
            for k in range(6):
                x = 300 + k * 118
                lb = "t%d" % (k + 1) if k < 5 else "tn"
                f.cell(x, top + 16, 76, 36, lb, None, col, "#fff")
                if k:
                    if k == 5:
                        f.t(x - 22, top + 40, "…", col, True, 17, "middle")
                    else:
                        f.line(x - 42, top + 34, x - 8, top + 34, col, 1.4)
                if mode == "g":
                    bw = 14 + k * 12
                    f.box(x + 38 - bw / 2.0, top + 58, bw, 8, "#c9a0ea", col, 3)
                    f.t(x + 38, top + 82, "W ＋ KV", col, size=_sz(11),
                        anchor="middle")
                else:
                    f.t(x + 38, top + 74, "搬 W", GY, size=_sz(11), anchor="middle")
        f.t(1102, top + 44, tail, col, True, _sz(12))

    yy = y + 3 * (ROWH + 12) + 6
    yy = f.band(yy, "info", "把 Ⓐ 和 Ⓒ 摆在一起看 ——&#160;这就是整个专题三的舞台", [
        '两边都是「一步一个，每步把权重搬一遍」。'
        '<tspan font-weight="700">唯一的区别在「每步还得额外搬什么」：</tspan>',
        '<tspan fill="%s">· RNN：一个<tspan font-weight="700">固定大小</tspan>的状态 h'
        '——&#160;<tspan font-weight="700">跟上下文多长完全无关</tspan></tspan>' % RD,
        '<tspan fill="%s">· Transformer：一路<tspan font-weight="700">线性变长</tspan>的 '
        'KV cache ——&#160;<tspan font-weight="700">128K 上下文时它能比权重本身还大</tspan></tspan>' % PU,
        '⭐ 所以后面三个旋钮拧的全是同一件事：'
        '<tspan font-weight="700">让 Ⓒ 这一行每步要搬的东西变小</tspan>'
        '——&#160;存少点（①）、看少点（②）、或者换回一个固定大小的状态（③）。'])
    f.save("fig3-rnn-decode.svg", yy + 12)


# ══════════════════════════════════════════════════════════════════
# 图四：三个痛点各自通向哪条路
# ══════════════════════════════════════════════════════════════════
def fig_pain():
    W = 1400
    f = Fig(W, "RNN 的三个痛点各自通向后来的哪条技术路线：串行通向 Transformer，"
               "梯度消失通向门控与注意力的短路径，固定大小的状态通向注意力机制；"
               "而线性注意力与 Mamba 是把第一条反过来再走一遍")
    f.marks = set()
    y = f.header(
        'RNN 的三个痛点 ——&#160;'
        '<tspan font-weight="700">后面三十年的路线图，就是这三条各自的解药</tspan>',
        '⭐ 这一页是路标：<tspan font-weight="700">后面每一个变体，'
        '都能追回到这三条里的某一条。</tspan>')

    f.colhead(64, y + 4, "痛在哪")
    f.colhead(556, y + 4, "谁来解")
    f.colhead(880, y + 4, "解完之后欠下什么")
    y += 18

    ROWS = (
        ("①", "串行：算不快",
         "h<tspan baseline-shift=\"sub\" font-size=\"10\">t</tspan> 要等 "
         "h<tspan baseline-shift=\"sub\" font-size=\"10\">t−1</tspan>，"
         "序列多长就排多少轮；权重每轮重搬一次",
         "Transformer", "把循环整个拿掉",
         "<tspan font-weight=\"700\">注意力矩阵变成 O(N²)</tspan> ——&#160;"
         "本专题三个旋钮都是在还这笔账", RD, "#fce8e6"),
        ("②", "梯度消失：记不住",
         "反向传播要连乘 n 次，小于 1 就指数衰减 ——&#160;学不到远处的依赖",
         "门控 → 注意力", "LSTM 1997 / GRU 2014",
         "注意力更彻底：<tspan font-weight=\"700\">任意两个位置只隔一步</tspan>"
         "（表 1 最长路径 O(1) vs O(n)）", OR, "#fef7e0"),
        ("③", "状态是固定大小：装不下",
         "不管序列多长，全部历史压进一个 [d] 的向量 ——&#160;长句子必然丢信息",
         "Bahdanau 2014", "别只看最后那个向量",
         "让解码每一步回头看整段编码 ——&#160;⭐ "
         "<tspan font-weight=\"700\">注意力的出生证明：它一开始只是 RNN 的一个补丁</tspan>",
         GR, "#e6f4ea"),
    )
    for (num, name, hurt, cure, cure2, tail, col, fill) in ROWS:
        f.box(0, y, W, 96, fill, col, 9)
        f.box(14, y + 22, 40, 40, "#fff", col, 20)
        f.t(34, y + 49, num, col, True, 19, "middle")
        f.t(64, y + 34, name, col, True, 13.5, cls="svglbl")
        f.t(64, y + 60, hurt, GY, size=_sz(12))
        f.line(500, y + 46, 536, y + 46, col, 1.8)
        f.t(556, y + 34, cure, col, True, 13.5, cls="svglbl")
        f.t(556, y + 60, cure2, GY, size=_sz(12))
        f.t(880, y + 48, tail, GY, size=_sz(12))
        y += 104

    y = f.band(y + 4, "info", "而第 ① 条后来被反着又走了一遍 ——&#160;这就是专题三真正的主脊", [
        'Transformer 用「放弃状态」换来了并行度。'
        '<tspan font-weight="700">线性注意力和 Mamba 这一支，是想把状态请回来</tspan>'
        '——&#160;因为有状态，每步要搬的东西才不再变长。',
        '⛔ 但状态一回来，<tspan font-weight="700">串行也跟着回来</tspan>。'
        '于是又得把并行度找回来：<tspan font-weight="700">chunk 化、parallel scan</tspan>。',
        '⭐ 所以后面 DeltaNet / GDN / KDA 的公式为什么必须长成那样：'
        '<tspan font-weight="700">它们得线性到能被 scan，否则就退回 1990 年那条链。'
        '一个完整的圆。</tspan>'])
    y = f.src(y + 18,
              'Bengio, Simard, Frasconi 1994（梯度消失）；Hochreiter &amp; Schmidhuber 1997（LSTM）；'
              'Bahdanau et al. 2014 (arXiv 1409.0473)「a fixed-length vector is a bottleneck」',
              'Vaswani et al. 2017 (arXiv 1706.03762) 表 1；Martin &amp; Cundy 2018 '
              '(arXiv 1709.04057)：非线性依赖挡住并行，只有线性依赖能用 parallel scan 扫')
    f.save("fig3-rnn-pain.svg", y + 6)


fig_unroll()
fig_hw()
fig_decode()
fig_pain()
