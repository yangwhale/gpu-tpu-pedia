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
import io
import os
import xml.dom.minidom

HERE = os.path.dirname(os.path.abspath(__file__))

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
PU, CY, BR, INK = "#8430ce", "#00838f", "#7a5000", "#202124"
LGY, MID = "#f8f9fa", "#dadce0"


def wpx(s, size=11.5):
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


class Fig(object):
    """一张 SVG。⛔ 高度不写死 —— 收尾时按真实落点回填（跟编年史那张同一套）。"""

    def __init__(self, w, aria):
        self.w, self.aria, self.p = w, aria, []
        self._hdr = 0
        self.p.append("")

    def t(self, x, y, s, cls="svgsm", fill=None, bold=False, size=None, anchor=None):
        self.p.append('<text class="%s" x="%d" y="%d"%s%s%s>%s</text>' % (
            cls, x, y, ' fill="%s"' % fill if fill else '',
            ' text-anchor="%s"' % anchor if anchor else '',
            ' style="font-size:%dpx"' % size if size else '',
            '<tspan font-weight="700">%s</tspan>' % s if bold else s))

    def box(self, x, y, w, h, fill="#fff", stroke=MID, r=6, sw=1, dash=None):
        self.p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" '
                      'stroke="%s" stroke-width="%s"%s/>'
                      % (x, y, w, h, r, fill, stroke, sw,
                         ' stroke-dasharray="%s"' % dash if dash else ''))

    def line(self, x1, y1, x2, y2, col=GY, sw=1.2, dash=None, arrow=True):
        self.p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
                      'stroke-width="%s"%s%s/>'
                      % (x1, y1, x2, y2, col, sw,
                         ' stroke-dasharray="%s"' % dash if dash else '',
                         ' marker-end="url(#ah)"' if arrow else ''))

    def path(self, d, col=GY, sw=1.2, dash=None, arrow=True):
        self.p.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s"%s%s/>'
                      % (d, col, sw,
                         ' stroke-dasharray="%s"' % dash if dash else '',
                         ' marker-end="url(#ah)"' if arrow else ''))

    def cell(self, x, y, w, h, label, col, fill, size=11, r=5, sub=None):
        """一个带居中标签的小块。⛔ 宽度由调用方给，但**文字放不下就报错**，
        别让它悄悄溢出去 —— 这张图前身在这上面栽过三次。"""
        need = wpx(label, size) + 14
        assert w >= need, "「%s」要 %dpx，格子只有 %dpx" % (label, need, w)
        self.box(x, y, w, h, fill, col, r)
        self.t(x + w // 2, y + h // 2 + (0 if sub is None else -4) + 4,
               label, fill=col, bold=True, size=size, anchor="middle")
        if sub:
            self.t(x + w // 2, y + h // 2 + 14, sub, fill=GY, size=9, anchor="middle")

    def save(self, name, bottom):
        self.p.append('</svg>')
        self.p[self._hdr] = (
            '<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="%s">'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="%s"/></marker></defs>'
            % (self.w, bottom, self.aria, GY))
        s = "\n".join(self.p)
        # ⛔ 写盘前验良构：控制字符、没闭合的标签在浏览器里都是**静默**坏掉的。
        xml.dom.minidom.parseString(s.encode("utf-8"))
        io.open(os.path.join(HERE, name), "w", encoding="utf-8").write(s)
        print("ok  %s  %d×%d" % (name, self.w, bottom))


# ══════════════════════════════════════════════════════════════════
# 图一：它是什么、怎么算
# ══════════════════════════════════════════════════════════════════
def fig_unroll():
    W = 1400
    f = Fig(W, "RNN 的两种画法：左边是 Elman 1990 的折叠画法，隐藏层原样抄一份到 "
               "context 层；右边是沿时间展开成一条链，每一步吃当前输入和上一步的状态")
    f.t(0, 18, 'RNN ——&#160;<tspan font-weight="700">先把「之前发生了什么」'
               '塞进一个固定大小的盒子，再一步一步往下传</tspan>',
        "svglbl", INK, size=15)
    f.t(0, 39, '⭐ 一句话：<tspan font-weight="700">序列有先后，所以得有个东西把历史带下去。'
               'RNN 的答案是「带一个状态向量 h」——&#160;'
               '而这一个决定，后面所有的好处和所有的痛点都是它的后果。</tspan>', fill=GY)

    # ── 左：折叠 ────────────────────────────────────────────────
    PX, PW, PY = 0, 400, 62
    f.box(PX, PY, PW, 300, LGY, MID, 8)
    f.t(PX + 14, PY + 22, '① 折叠着看 ——&#160;Elman 1990 原始画法', "svglbl", INK, size=13)

    cx = PX + 96
    f.cell(cx, PY + 48, 130, 34, "输出 y", BL, "#e8f0fe")
    f.cell(cx, PY + 118, 130, 34, "隐藏 h", RD, "#fce8e6")
    f.cell(PX + 24, PY + 208, 120, 34, "输入 x", GR, "#e6f4ea")
    f.cell(PX + 190, PY + 208, 150, 34, "context 层", PU, "#f3e8fd")

    f.line(cx + 65, PY + 118, cx + 65, PY + 86, RD)                  # h → y
    f.line(PX + 84, PY + 208, cx + 40, PY + 156, GR)                 # x → h
    f.line(PX + 250, PY + 208, cx + 96, PY + 156, PU)                # ctx → h
    # h → context：原样抄，虚线 + 固定权重 1.0
    f.path("M %d %d L %d %d L %d %d" % (cx + 130, PY + 135, PX + 358, PY + 135,
                                        PX + 358, PY + 204), PU, 1.4, "4 3")
    f.t(PX + 236, PY + 108, '原样复制一份', fill=PU, bold=True, size=10)
    f.t(PX + 236, PY + 122, '固定权重 1.0，不训练', fill=GY, size=9)

    f.t(PX + 14, PY + 266, '⭐ 那条<tspan font-weight="700">虚线不是学出来的</tspan>'
                           '——&#160;原文写明是「one-for-one basis,', fill=GY, size=10)
    f.t(PX + 14, PY + 280, 'with fixed weight of 1.0」。'
                           '<tspan font-weight="700">整个循环结构就靠这一条硬接线。</tspan>',
        fill=GY, size=10)

    # ── 右：展开 ────────────────────────────────────────────────
    QX, QW = 420, W - 420
    f.box(QX, PY, QW - 8, 300, "#fff", MID, 8)
    f.t(QX + 14, PY + 22, '② 展开着看 ——&#160;它其实就是一条链，'
                          '<tspan font-weight="700">每一格都得等前一格算完</tspan>',
        "svglbl", INK, size=13)

    n, step, x0 = 5, 178, QX + 40
    for i in range(n):
        x = x0 + i * step
        # ⛔ 最后一列是「第 n 步」不是「第 5 步」—— 原先头上挂「…」、格子里却写 5，
        #    两种记号在说同一格，读者会以为链只有五步。
        k = str(i + 1) if i < n - 1 else "n"
        f.t(x + 55, PY + 46, "t=%s" % k, fill=GY, size=10, anchor="middle")
        if i == n - 1:
            f.t(x - 34, PY + 162, "…", fill=GY, bold=True, size=15, anchor="middle")
        f.cell(x, PY + 216, 110, 30, "x%s" % k, GR, "#e6f4ea", size=10)
        f.cell(x, PY + 140, 110, 34, "h%s" % k, RD, "#fce8e6", size=11)
        f.cell(x, PY + 58, 110, 30, "y%s" % k, BL, "#e8f0fe", size=10)
        f.line(x + 55, PY + 216, x + 55, PY + 178, GR)
        f.line(x + 55, PY + 140, x + 55, PY + 92, BL)
        if i:
            f.line(x - step + 110, PY + 157, x - 4, PY + 157, RD, 2.0)
    f.t(x0 + 110 + 12, PY + 150, '上一步的 h', fill=RD, bold=True, size=10)
    f.t(QX + 14, PY + 266, '⛔ <tspan font-weight="700">这条横箭头就是全部问题的根源</tspan>'
                           '：h3 要用 h2，h2 要用 h1 ——&#160;'
                           '五步就得排五轮，一百万步就得排一百万轮。', fill=GY, size=11)
    f.t(QX + 14, PY + 282, '⭐ 注意<tspan font-weight="700">竖着的那些箭头彼此不相干</tspan>'
                           '——&#160;能并行的方向一直都在，'
                           '<tspan font-weight="700">被卡住的只有横着这一个方向。</tspan>',
        fill=GY, size=11)

    # ── 底：一格里面到底在算什么 ─────────────────────────────────
    FY = PY + 316
    f.box(0, FY, W, 118, "#e8f0fe", BL, 8)
    f.t(16, FY + 24, '③ 拆开一格看：里面就是<tspan font-weight="700">两个矩阵乘、'
                     '一个加法、一个非线性</tspan>——&#160;'
                     '<tspan font-weight="700">RNN 本身一点都不复杂</tspan>',
        "svglbl", "#174ea6", size=13)
    f.t(16, FY + 50, 'h<tspan baseline-shift="sub" font-size="9">t</tspan> ＝ '
                     'tanh( W<tspan baseline-shift="sub" font-size="9">h</tspan> · '
                     'h<tspan baseline-shift="sub" font-size="9">t−1</tspan> ＋ '
                     'W<tspan baseline-shift="sub" font-size="9">x</tspan> · '
                     'x<tspan baseline-shift="sub" font-size="9">t</tspan> ＋ b )'
                     '　　　y<tspan baseline-shift="sub" font-size="9">t</tspan> ＝ '
                     'W<tspan baseline-shift="sub" font-size="9">y</tspan> · '
                     'h<tspan baseline-shift="sub" font-size="9">t</tspan>',
        fill="#174ea6", bold=True, size=14)
    f.t(16, FY + 76, '形状：h 是 [d]、x 是 [d<tspan baseline-shift="sub" font-size="9">x</tspan>]、'
                     'W<tspan baseline-shift="sub" font-size="9">h</tspan> 是 [d × d]。'
                     '<tspan font-weight="700">d 就是这个模型能记住的全部容量 ——'
                     '不管你喂它 10 个词还是 10 万个词，盒子就这么大。</tspan>',
        fill="#174ea6")
    f.t(16, FY + 98, '⭐ LSTM 和 GRU 没有改这个形状，只是把那个 tanh 换成了几个'
                     '<tspan font-weight="700">门</tspan>（LSTM 四个、GRU 三个），'
                     '让梯度有一条不被反复乘小数的通路。'
                     '<tspan font-weight="700">链还是那条链。</tspan>', fill=GY)
    f.save("fig3-rnn-unroll.svg", FY + 130)


# ══════════════════════════════════════════════════════════════════
# 图二：为什么在硬件上快不起来
# ══════════════════════════════════════════════════════════════════
def fig_hw():
    W = 1400
    f = Fig(W, "RNN 与 Transformer 在硬件上的差别：RNN 每个时间步都要把同一块权重"
               "从 HBM 搬一次，算术强度等于 batch size；Transformer 训练时同一块权重"
               "只搬一次，算术强度是 batch size 乘以序列长度")
    f.t(0, 18, '同样是矩阵乘，<tspan font-weight="700">为什么 RNN 在加速器上就是跑不快</tspan>'
               '——&#160;一句话：<tspan font-weight="700">权重被搬了 n 次</tspan>',
        "svglbl", INK, size=15)
    f.t(0, 39, '⭐ 先说清楚<tspan font-weight="700">它不是「算得多」，是「搬得多」</tspan>。'
               'RNN 的总计算量比同规模 Transformer 还小'
               '（Vaswani 表 1：O(n·d²) vs O(n²·d)，短序列时前者更少）——&#160;'
               '<tspan font-weight="700">慢的原因跟计算量无关。</tspan>', fill=GY)

    # ── 上排：RNN ────────────────────────────────────────────────
    AY = 62
    f.box(0, AY, W, 150, "#fce8e6", RD, 8)
    f.t(16, AY + 24, 'Ⓐ RNN：每一个时间步，都要把<tspan font-weight="700">同一块权重 W</tspan>'
                     '从 HBM 重新搬进片上一次', "svglbl", "#a50e0e", size=13)
    for i in range(6):
        x = 40 + i * 176
        f.cell(x, AY + 48, 92, 44, "W", RD, "#fff", size=13, sub="[4d × d]")
        f.cell(x + 100, AY + 54, 34, 32, "·", GY, "#f1f3f4", size=13)
        f.cell(x + 140, AY + 48, 26, 44, "", BL, "#e8f0fe", size=9)
        f.t(x + 153, AY + 106, "[d×B]", fill=BL, size=9, anchor="middle")
        f.t(x + 46, AY + 106, "第 %s 步" % (i + 1 if i < 5 else "n"),
            fill=GY, size=9, anchor="middle")
        if i:
            if i == 5:      # ⛔ 5 → n 之间是跳过去的，得画出来
                f.t(x - 20, AY + 76, "…", fill=GY, bold=True, size=15, anchor="middle")
            else:
                f.line(x - 34, AY + 70, x - 6, AY + 70, GY, 1.0)
    f.t(16, AY + 128, '⛔ 那块 W 每一步都得重新读一遍，而配给它的只有'
                      '<tspan font-weight="700">一个 batch 那么窄的一条</tspan>。'
                      'NVIDIA 官方文档的说法是「a GEMM with '
                      '<tspan font-weight="700">one dimension of one</tspan>」'
                      '——&#160;<tspan font-weight="700">名义上是矩阵乘，实际是矩阵乘向量。</tspan>',
        fill="#a50e0e")

    # ── 下排：Transformer ───────────────────────────────────────
    BY = AY + 166
    f.box(0, BY, W, 132, "#e6f4ea", GR, 8)
    f.t(16, BY + 24, 'Ⓑ Transformer 训练：同一块权重<tspan font-weight="700">只搬一次</tspan>，'
                     '后面 n 个位置<tspan font-weight="700">一起</tspan>喂进去',
        "svglbl", "#0d652d", size=13)
    f.cell(40, BY + 48, 92, 44, "W", GR, "#fff", size=13, sub="[4d × d]")
    f.cell(140, BY + 54, 34, 32, "·", GY, "#f1f3f4", size=13)
    f.box(182, BY + 48, 880, 44, "#e8f0fe", BL, 5)
    f.t(622, BY + 76, "[ d × (n · B) ]　——　n 个位置全在这一块里", fill=BL,
        bold=True, size=13, anchor="middle")
    f.t(16, BY + 112, '⭐ 一次搬运换来 n 倍的活干。'
                      '<tspan font-weight="700">这就是「用平方的计算量买完全的并行度」'
                      '那笔交易的硬件形态</tspan>——&#160;'
                      '多算的那些 FLOPs，换的是权重不用来回搬。', fill="#0d652d")

    # ── 账 ──────────────────────────────────────────────────────
    CY_ = BY + 148
    f.box(0, CY_, W, 172, "#fef7e0", OR, 8)
    f.t(16, CY_ + 24, '⭐⭐ 把账算出来 ——&#160;'
                      '<tspan font-weight="700">算术强度 ＝ 算了多少次 ÷ 搬了多少字节</tspan>，'
                      '它决定你是在等算力还是在等内存', "svglbl", BR, size=13)
    rows = (
        ("RNN 的循环支路", "2 · 4d · d · B", "4d · d · 2 字节", "B",
         "⛔ <tspan font-weight=\"700\">跟 d 无关，就等于 batch size</tspan>"),
        ("Transformer 训练", "2 · 4d · d · (n·B)", "4d · d · 2 字节", "n · B",
         "⭐ <tspan font-weight=\"700\">整整多了一个 n</tspan>"),
    )
    f.t(16, CY_ + 48, "算的是谁", fill=GY, bold=True, size=11)
    f.t(210, CY_ + 48, "FLOPs", fill=GY, bold=True, size=11)
    f.t(400, CY_ + 48, "从 HBM 搬进来的字节", fill=GY, bold=True, size=11)
    f.t(600, CY_ + 48, "算术强度", fill=GY, bold=True, size=11)
    f.t(710, CY_ + 48, "于是", fill=GY, bold=True, size=11)
    for i, (a, b, c, d_, e) in enumerate(rows):
        y = CY_ + 70 + i * 22
        col = RD if i == 0 else GR
        f.t(16, y, a, fill=col, bold=True)
        f.t(210, y, b, fill=INK)
        f.t(400, y, c, fill=INK)
        f.t(600, y, d_, fill=col, bold=True)
        f.t(710, y, e, fill=col)
    f.box(16, CY_ + 122, W - 32, 1, GY, GY, 0)
    f.t(16, CY_ + 144, '拐点在哪：TPU v7 官方每芯片 FP8 <tspan font-weight="700">4614 '
                       'TFLOP/s</tspan>，BF16 取一半 ＝ <tspan font-weight="700">2307</tspan>；'
                       '官方 HBM 带宽 <tspan font-weight="700">7.37 TB/s</tspan>。'
                       '2307 ÷ 7.37 ＝ <tspan font-weight="700">313 FLOP/byte</tspan>。',
        fill=BR)
    f.t(16, CY_ + 162, '⭐ 也就是说：<tspan font-weight="700">要把 v7 喂饱，LSTM 的 batch '
                       '得开到 313 以上</tspan>。而 Vaswani 那句原话讲的正是它的反面 ——&#160;'
                       '<tspan font-weight="700">「memory constraints limit batching '
                       'across examples」：序列一长，显存就不让你把 batch 开大。</tspan>',
        fill=BR)
    f.save("fig3-rnn-hw.svg", CY_ + 186)


# ══════════════════════════════════════════════════════════════════
# 图三：三个痛点各自通向哪条路
# ══════════════════════════════════════════════════════════════════
def fig_pain():
    W = 1400
    f = Fig(W, "RNN 的三个痛点各自通向后来的哪条技术路线：串行通向 Transformer，"
               "梯度消失通向门控与注意力的短路径，固定大小的状态通向注意力机制；"
               "而线性注意力与 Mamba 是把第一条反过来再走一遍")
    f.t(0, 18, 'RNN 的三个痛点 ——&#160;<tspan font-weight="700">'
               '后面三十年的路线图，其实就是这三条各自的解药</tspan>',
        "svglbl", INK, size=15)
    f.t(0, 39, '⭐ 这一页是整个专题的路标：<tspan font-weight="700">后面每一个变体，'
               '都能追回到这三条里的某一条。</tspan>', fill=GY)

    ROWS = (
        ("①", "串行：算不快",
         "h<tspan baseline-shift=\"sub\" font-size=\"9\">t</tspan> 要等 "
         "h<tspan baseline-shift=\"sub\" font-size=\"9\">t−1</tspan>，"
         "序列多长就得排多少轮；而<tspan font-weight=\"700\">权重每轮重搬一次</tspan>",
         "Transformer 把循环整个拿掉",
         "<tspan font-weight=\"700\">代价是注意力矩阵变成 O(N²)</tspan> ——&#160;"
         "本专题后面全部三个旋钮，都是在还这笔账", RD, "#fce8e6"),
        ("②", "梯度消失：记不住",
         "反向传播要连乘 n 次，小于 1 就指数衰减"
         "（Bengio 1994）——&#160;<tspan font-weight=\"700\">学不到远处的依赖</tspan>",
         "先是门控（LSTM 1997 / GRU 2014）",
         "后来注意力更彻底：<tspan font-weight=\"700\">任意两个位置之间只隔一步</tspan>"
         "（Vaswani 表 1 的最长路径 O(1) vs O(n)）", OR, "#fef7e0"),
        ("③", "状态是固定大小：装不下",
         "不管序列多长，全部历史压进一个 [d] 的向量 ——&#160;"
         "<tspan font-weight=\"700\">长句子必然丢信息</tspan>",
         "Bahdanau 2014：别只看最后那个向量",
         "让解码的每一步<tspan font-weight=\"700\">回头去看整段编码</tspan> ——&#160;"
         "⭐ <tspan font-weight=\"700\">这就是注意力的出生证明：它一开始只是 RNN 的一个补丁</tspan>",
         GR, "#e6f4ea"),
    )
    y = 62
    for (num, name, hurt, cure, tail, col, fill) in ROWS:
        f.box(0, y, W, 86, fill, col, 8)
        f.t(18, y + 34, num, fill=col, bold=True, size=26)
        f.t(52, y + 26, name, "svglbl", col, size=13, bold=True)
        f.t(52, y + 48, hurt, fill=GY)
        f.line(486, y + 42, 522, y + 42, col, 1.6)
        f.t(540, y + 26, cure, "svglbl", col, size=13, bold=True)
        f.t(540, y + 48, tail, fill=GY)
        f.t(52, y + 70, '', fill=GY)
        y += 94

    # ── 收口：圆是怎么合上的 ────────────────────────────────────
    f.box(0, y, W, 122, "#f3e8fd", PU, 8)
    f.t(16, y + 24, '⭐⭐ 而这三条里，<tspan font-weight="700">第 ① 条后来被反着又走了一遍</tspan>'
                    '——&#160;这就是专题三真正的主脊', "svglbl", PU, size=13)
    f.t(16, y + 50, 'Transformer 用「放弃状态」换来了并行度。'
                    '<tspan font-weight="700">线性注意力和 Mamba 这一支，是想把状态请回来</tspan>'
                    '——&#160;因为有状态才有 O(N)。', fill=PU)
    f.t(16, y + 72, '⛔ 但状态一回来，<tspan font-weight="700">串行也跟着回来了</tspan>。'
                    '于是又得想办法把并行度找回来：'
                    '<tspan font-weight="700">chunk 化、parallel scan</tspan> ——&#160;'
                    'Martin &amp; Cundy 2018 证明了'
                    '<tspan font-weight="700">只有「线性」的循环依赖才扫得动</tspan>。', fill=PU)
    f.t(16, y + 94, '⭐ 所以后面那些 DeltaNet / GDN / KDA 的公式为什么长成那个样子，'
                    '答案在这里：<tspan font-weight="700">它们必须线性到能被 scan，'
                    '否则就退回 1990 年的那条链。一个完整的圆。</tspan>',
        fill=PU, bold=False)
    f.save("fig3-rnn-pain.svg", y + 134)


fig_unroll()
fig_hw()
fig_pain()
