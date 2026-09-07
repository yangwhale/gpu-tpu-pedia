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
            # ⛔ 这里原先写死 size=9，**绕过了 MINSZ 那道地板** —— 地板只拦
            #   走 _sz() 的调用点，而这个默认参数没走。渲染后 11px，是全课最小的字。
            #   ⭐ 教训：**护栏只能拦住经过它的路径。** 默认参数是最容易绕过去的那条。
            self.t(x + w // 2, y + h // 2 + 15, sub, fill=GY, size=_sz(11),
                   anchor="middle")

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
# ⛔⛔ 2026-09-07 第二轮，现场要求「好好审计一下有没有不准确或者混乱的地方」
#     —— 审出**三处硬伤 ＋ 一处漏讲**，都记在这儿，别再犯。
#
#  1. **方向搞反了。** 上一版写「序列比维度短的时候 RNN 计算量反而更小」——
#     **正好说反了**。O(n·d²) < O(n²·d) ⟺ **d < n**，也就是**序列比维度长**的时候
#     RNN 才更省。而 Vaswani 原文写的是另一边：「self-attention layers are faster
#     than recurrent layers when the sequence length n is **smaller** than the
#     representation dimensionality d, which is most often the case」——
#     ⭐ **2017 年的常态下，自注意力连 FLOPs 都比 RNN 少。**
#     所以「RNN 慢不是因为算得多」这句话得换个说法（见图二），
#     不能靠一个反了的不等式撑着。
#     ⭐⭐ 形状：**比较类断言最容易在符号方向上翻车，而它读起来完全通顺。**
#        判据：任何 "A 比 B 大／小" 都要把不等式真的解一遍，别凭语感。
#
#  2. **门数说法不对。** 上一版写「LSTM 四个门、GRU 三个门」——&nbsp;那是
#     **拼进一个 GEMM 的矩阵块数**（NVIDIA 文档的口径），不是门数。
#     教科书口径：**LSTM 三个门（输入／遗忘／输出）＋ 一个候选值 ＝ 四块**；
#     **GRU 两个门（更新／重置）＋ 一个候选 ＝ 三块**。
#     ⛔ 把工程口径当成概念口径写出去，是这门课最该避免的一类错。
#
#  3. **Martin & Cundy 说过头了。** 上一版写「证明了只有线性的循环依赖才扫得动」。
#     人家摘要的原话是：非线性依赖 "prevent parallelizing training over sequence
#     length"，而 "RNNs with **only linear** sequential dependencies **can be**
#     parallelized ... using the parallel scan algorithm"。
#     ⭐ 是**给出做法**，不是**证明必要性**。改成他们自己的说法。
#
#  4. **漏了整个故事最狠的一格：Transformer 解码时又变回了 RNN 的形状。**
#     训练时它并行、权重只搬一次；**可生成的时候它一个 token 一个 token 往外吐，
#     每吐一个就要把全部权重再搬一遍** ——&nbsp;算术强度同样只等于 batch。
#     ⭐⭐ **Transformer 没有治好 RNN 的病，它把病从训练挪到了推理。**
#     而且更重：RNN 每步搬的是一个**固定大小**的状态，
#     Transformer 每步还要搬一路**变长**的 KV cache。
#     ⭐ 这一格把 §零 从「背景介绍」变成了「本专题的舞台说明」——
#       后面三个旋钮全都发生在这一行上。⛔ 别把这张图删了。
#
# 📌 另外：**字号地板抬到 11**。现场「图里面的字不要太小」。
#    实测这些图在版心里被放大 1.22 倍，专题二全部 42 张图的渲染后最小字号是
#    **12.2px**（＝ 10px × 1.22）；上一版这里有 9px 的小注，渲染后只有 11px，
#    **是全课最小的字**。⛔ 这个文件里不许再出现 size < 11 的文字。
# ══════════════════════════════════════════════════════════════════

MINSZ = 11          # ⛔ 字号地板。低于它 render 出来会比专题二任何一张图都小


def _sz(n):
    assert n >= MINSZ, "字号 %d 太小了（地板 %d）" % (n, MINSZ)
    return n


# ══════════════════════════════════════════════════════════════════
# 图一：它是什么、怎么算
# ══════════════════════════════════════════════════════════════════
def fig_unroll():
    W = 1400
    f = Fig(W, "RNN 的两种画法：左边是 Elman 1990 的折叠画法，隐藏层原样抄一份到 "
               "context 层；右边是沿时间展开成一条链，每一步吃当前输入和上一步的状态")
    f.t(0, 18, 'RNN ——&#160;<tspan font-weight="700">把「之前发生了什么」'
               '塞进一个固定大小的盒子，一步一步往下传</tspan>',
        "svglbl", INK, size=16)
    f.t(0, 40, '⭐ 全部设计就这一句。<tspan font-weight="700">后面所有的好处、'
               '所有的痛点，都是这一个决定的后果。</tspan>', fill=GY, size=_sz(12))

    PX, PW, PY = 0, 420, 64
    f.box(PX, PY, PW, 310, LGY, MID, 8)
    f.t(PX + 14, PY + 24, '① 折叠着看 ——&#160;Elman 1990 原始画法', "svglbl", INK, size=14)

    cx = PX + 100
    f.cell(cx, PY + 52, 140, 36, "输出 y", BL, "#e8f0fe", size=_sz(12))
    f.cell(cx, PY + 124, 140, 36, "隐藏 h", RD, "#fce8e6", size=_sz(12))
    f.cell(PX + 20, PY + 214, 130, 36, "输入 x", GR, "#e6f4ea", size=_sz(12))
    f.cell(PX + 196, PY + 214, 160, 36, "context 层", PU, "#f3e8fd", size=_sz(12))

    f.line(cx + 70, PY + 124, cx + 70, PY + 92, RD)
    f.line(PX + 85, PY + 214, cx + 44, PY + 164, GR)
    f.line(PX + 262, PY + 214, cx + 100, PY + 164, PU)
    f.path("M %d %d L %d %d L %d %d" % (cx + 140, PY + 142, PX + 376, PY + 142,
                                        PX + 376, PY + 210), PU, 1.4, "4 3")
    f.t(PX + 248, PY + 112, '原样复制一份', fill=PU, bold=True, size=_sz(11))
    f.t(PX + 248, PY + 128, '固定权重 1.0，不训练', fill=GY, size=_sz(11))
    f.t(PX + 14, PY + 274, '⭐ 那条<tspan font-weight="700">虚线不是学出来的</tspan>'
                           '——&#160;原文写死「one-for-one basis,', fill=GY, size=_sz(11))
    f.t(PX + 14, PY + 291, 'with fixed weight of 1.0」。'
                           '<tspan font-weight="700">整个循环就靠这一条硬接线。</tspan>',
        fill=GY, size=_sz(11))

    QX, QW = 440, W - 440
    f.box(QX, PY, QW - 8, 310, "#fff", MID, 8)
    f.t(QX + 14, PY + 24, '② 展开着看 ——&#160;它其实就是一条链，'
                          '<tspan font-weight="700">每一格都得等前一格算完</tspan>',
        "svglbl", INK, size=14)

    n, step, x0 = 5, 178, QX + 34
    for i in range(n):
        x = x0 + i * step
        k = str(i + 1) if i < n - 1 else "n"
        f.t(x + 55, PY + 50, "t=%s" % k, fill=GY, size=_sz(11), anchor="middle")
        if i == n - 1:
            f.t(x - 34, PY + 168, "…", fill=GY, bold=True, size=16, anchor="middle")
        f.cell(x, PY + 222, 110, 32, "x%s" % k, GR, "#e6f4ea", size=_sz(12))
        f.cell(x, PY + 146, 110, 36, "h%s" % k, RD, "#fce8e6", size=_sz(12))
        f.cell(x, PY + 62, 110, 32, "y%s" % k, BL, "#e8f0fe", size=_sz(12))
        f.line(x + 55, PY + 222, x + 55, PY + 186, GR)
        f.line(x + 55, PY + 146, x + 55, PY + 98, BL)
        if i:
            f.line(x - step + 110, PY + 164, x - 4, PY + 164, RD, 2.0)
    f.t(x0 + 122, PY + 156, '上一步的 h', fill=RD, bold=True, size=_sz(11))
    f.t(QX + 14, PY + 274, '⛔ <tspan font-weight="700">这条横箭头是全部问题的根源</tspan>'
                           '：h3 要用 h2，h2 要用 h1 ——&#160;'
                           '一百万步就得排一百万轮。', fill=GY, size=_sz(12))
    f.t(QX + 14, PY + 293, '⭐ 而<tspan font-weight="700">竖着的箭头彼此不相干</tspan>'
                           '——&#160;能并行的方向一直都在，'
                           '<tspan font-weight="700">被卡住的只有横着这一个。</tspan>',
        fill=GY, size=_sz(12))

    FY = PY + 326
    f.box(0, FY, W, 116, "#e8f0fe", BL, 8)
    f.t(16, FY + 26, '③ 拆开一格：<tspan font-weight="700">两个矩阵乘、一个加法、'
                     '一个非线性</tspan>——&#160;'
                     '<tspan font-weight="700">RNN 本身一点都不复杂</tspan>',
        "svglbl", "#174ea6", size=14)
    f.t(16, FY + 54, 'h<tspan baseline-shift="sub" font-size="10">t</tspan> ＝ '
                     'f( W<tspan baseline-shift="sub" font-size="10">h</tspan> · '
                     'h<tspan baseline-shift="sub" font-size="10">t−1</tspan> ＋ '
                     'W<tspan baseline-shift="sub" font-size="10">x</tspan> · '
                     'x<tspan baseline-shift="sub" font-size="10">t</tspan> ＋ b )'
                     '　　　y<tspan baseline-shift="sub" font-size="10">t</tspan> ＝ '
                     'W<tspan baseline-shift="sub" font-size="10">y</tspan> · '
                     'h<tspan baseline-shift="sub" font-size="10">t</tspan>',
        fill="#174ea6", bold=True, size=15)
    f.t(16, FY + 80, 'h 是 [d]，W<tspan baseline-shift="sub" font-size="10">h</tspan> 是 [d × d]。'
                     '<tspan font-weight="700">d 就是它能记住的全部容量 ——&#160;'
                     '喂 10 个词还是 10 万个词，盒子一样大。</tspan>',
        fill="#174ea6", size=_sz(12))
    f.t(16, FY + 102, '⭐ LSTM／GRU 没改这个形状，只是把 f 换成几个门'
                      '（<tspan font-weight="700">LSTM 三个门＋一个候选值，四块矩阵；'
                      'GRU 两个门＋一个候选，三块</tspan>）。'
                      '<tspan font-weight="700">链还是那条链。</tspan>',
        fill=GY, size=_sz(12))
    f.save("fig3-rnn-unroll.svg", FY + 128)


# ══════════════════════════════════════════════════════════════════
# 图二：为什么在硬件上快不起来
# ══════════════════════════════════════════════════════════════════
def fig_hw():
    W = 1400
    f = Fig(W, "RNN 与 Transformer 训练在硬件上的差别：RNN 每个时间步都要把同一块权重"
               "从 HBM 搬一次，算术强度等于 batch size；Transformer 训练时同一块权重"
               "只搬一次，算术强度是 batch size 乘以序列长度")
    f.t(0, 18, '为什么 RNN 在加速器上就是跑不快 ——&#160;'
               '<tspan font-weight="700">同一块权重被搬了 n 次</tspan>',
        "svglbl", INK, size=16)
    # ⛔ 这里**不要**复述正文那句「别去比总计算量」—— 那句连着 Vaswani 表 1，归正文。
    #   ⭐ 规矩：图上说过的正文别再说，正文说过的图上也别再说。
    #   （2026-09-07 审计发现同一句话在正文、图副标题、图注里出现了三遍。）
    f.t(0, 40, '⭐ <tspan font-weight="700">「串行」在硬件上的具体形态就是这个</tspan>'
               '——&#160;权重被搬 n 次，而每次只配一条 batch 那么窄的向量。',
        fill=GY, size=_sz(12))

    AY = 66
    f.box(0, AY, W, 156, "#fce8e6", RD, 8)
    f.t(16, AY + 26, 'Ⓐ RNN：每一个时间步，都要把<tspan font-weight="700">同一块权重 W</tspan>'
                     '从 HBM 重新搬进片上一次', "svglbl", "#a50e0e", size=14)
    for i in range(6):
        x = 40 + i * 176
        f.cell(x, AY + 52, 96, 46, "W", RD, "#fff", size=14, sub="[k·d × d]")
        f.cell(x + 104, AY + 58, 34, 34, "·", GY, "#f1f3f4", size=14)
        f.cell(x + 144, AY + 52, 26, 46, "", BL, "#e8f0fe")
        f.t(x + 157, AY + 114, "[d×B]", fill=BL, size=_sz(11), anchor="middle")
        f.t(x + 48, AY + 114, "第 %s 步" % (i + 1 if i < 5 else "n"),
            fill=GY, size=_sz(11), anchor="middle")
        if i:
            if i == 5:
                f.t(x - 20, AY + 80, "…", fill=GY, bold=True, size=16, anchor="middle")
            else:
                f.line(x - 40, AY + 75, x - 6, AY + 75, GY, 1.0)
    f.t(16, AY + 136, '⛔ W 每一步都要重读，而配给它的只有'
                      '<tspan font-weight="700">一个 batch 那么窄的一条</tspan>。'
                      'NVIDIA 官方文档的说法是「a GEMM with '
                      '<tspan font-weight="700">one dimension of one</tspan>」'
                      '——&#160;<tspan font-weight="700">名义上矩阵乘，实际是矩阵乘向量。</tspan>',
        fill="#a50e0e", size=_sz(12))

    BY = AY + 172
    f.box(0, BY, W, 136, "#e6f4ea", GR, 8)
    f.t(16, BY + 26, 'Ⓑ Transformer <tspan font-weight="700">训练</tspan>：同一块权重'
                     '<tspan font-weight="700">只搬一次</tspan>，'
                     'n 个位置<tspan font-weight="700">一起</tspan>喂进去',
        "svglbl", "#0d652d", size=14)
    f.cell(40, BY + 52, 96, 46, "W", GR, "#fff", size=14, sub="[k·d × d]")
    f.cell(144, BY + 58, 34, 34, "·", GY, "#f1f3f4", size=14)
    f.box(186, BY + 52, 880, 46, "#e8f0fe", BL, 5)
    f.t(626, BY + 81, "[ d × (n · B) ]　——　n 个位置全在这一块里", fill=BL,
        bold=True, size=14, anchor="middle")
    f.t(16, BY + 118, '⭐ 一次搬运换来 n 倍的活干。'
                      '<tspan font-weight="700">这就是「用平方的计算量买完全的并行度」'
                      '那笔交易的硬件形态。</tspan>', fill="#0d652d", size=_sz(12))

    CY_ = BY + 152
    f.box(0, CY_, W, 178, "#fef7e0", OR, 8)
    f.t(16, CY_ + 26, '⭐⭐ <tspan font-weight="700">算术强度 ＝ 算了多少次 ÷ 搬了多少字节</tspan>'
                      '——&#160;它决定你是在等算力，还是在等内存', "svglbl", BR, size=14)
    f.t(16, CY_ + 52, "算的是谁", fill=GY, bold=True, size=_sz(12))
    f.t(230, CY_ + 52, "FLOPs", fill=GY, bold=True, size=_sz(12))
    f.t(440, CY_ + 52, "搬进来的字节", fill=GY, bold=True, size=_sz(12))
    f.t(620, CY_ + 52, "算术强度", fill=GY, bold=True, size=_sz(12))
    f.t(740, CY_ + 52, "于是", fill=GY, bold=True, size=_sz(12))
    for i, (a, b, c, d_, e, col) in enumerate((
            ("RNN 每一步", "2 · k·d · d · B", "k·d · d · 2", "B",
             "⛔ <tspan font-weight=\"700\">跟 d 无关，就等于 batch size</tspan>", RD),
            ("Transformer 训练", "2 · k·d · d · (n·B)", "k·d · d · 2", "n · B",
             "⭐ <tspan font-weight=\"700\">整整多了一个 n</tspan>", GR))):
        y = CY_ + 76 + i * 24
        f.t(16, y, a, fill=col, bold=True, size=_sz(12))
        f.t(230, y, b, fill=INK, size=_sz(12))
        f.t(440, y, c, fill=INK, size=_sz(12))
        f.t(620, y, d_, fill=col, bold=True, size=_sz(12))
        f.t(740, y, e, fill=col, size=_sz(12))
    f.t(16, CY_ + 126, '⭐ 注意那个 k（LSTM 是 4、朴素 RNN 是 1）<tspan font-weight="700">'
                       '上下一约就没了</tspan>——&#160;'
                       '<tspan font-weight="700">算术强度等于 batch，跟门数、跟隐藏维都无关。</tspan>'
                       '（激活的搬运比权重小两个数量级，略去。）', fill=GY, size=_sz(11))
    f.box(16, CY_ + 136, W - 32, 1, GY, GY, 0)
    f.t(16, CY_ + 158, '拐点：TPU v7 官方每芯片 FP8 <tspan font-weight="700">4614 TFLOP/s</tspan>，'
                       'BF16 取一半 ＝ <tspan font-weight="700">2307</tspan>；官方 HBM 带宽 '
                       '<tspan font-weight="700">7.37 TB/s</tspan>。'
                       '2307 ÷ 7.37 ＝ <tspan font-weight="700">313 FLOP/byte</tspan>'
                       '　⭐ <tspan font-weight="700">要喂饱一块 v7，batch 得开到 313 以上</tspan>'
                       '——&#160;而长序列时显存<tspan font-weight="700">不让你开那么大</tspan>。',
        fill=BR, size=_sz(12))
    f.save("fig3-rnn-hw.svg", CY_ + 194)


# ══════════════════════════════════════════════════════════════════
# 图三（新增）：⭐⭐ 解码的时候，Transformer 又变回了 RNN 的形状
# ══════════════════════════════════════════════════════════════════
def fig_decode():
    W = 1400
    f = Fig(W, "三条时间线的对比：RNN 每步搬一次权重且串行；Transformer 训练时权重"
               "只搬一次且并行；但 Transformer 解码时又变回一步一个 token，"
               "每步搬一次权重，而且还要额外搬一路越来越长的 KV cache")
    f.t(0, 18, '⭐⭐ 而生成文字的时候，<tspan font-weight="700">'
               'Transformer 又变回了 1990 年那条链的形状</tspan>',
        "svglbl", INK, size=16)
    f.t(0, 40, '⛔ <tspan font-weight="700">Transformer 没有治好 RNN 的病，'
               '它把病从「训练」挪到了「推理」</tspan>——&#160;'
               '而且挪过去之后<tspan font-weight="700">更重</tspan>。', fill=GY, size=_sz(12))

    ROWS = (
        ("Ⓐ", "RNN（训练和推理都一样）", RD, "#fce8e6",
         "串行", "每步搬：W", "算术强度 ＝ B",
         "⛔ 一步一格，权重每步重搬", True),
        ("Ⓑ", "Transformer 训练", GR, "#e6f4ea",
         "并行", "整段只搬一次：W", "算术强度 ＝ n · B",
         "⭐ 这就是它当年赢下来的地方", False),
        ("Ⓒ", "Transformer 解码（你现在用的每个大模型）", PU, "#f3e8fd",
         "串行", "每步搬：W ＋ 越来越长的 KV", "算术强度 ≈ B",
         "⛔⛔ 串行回来了，而且多背一个变长的 KV", "grow"),
    )
    y = 64
    for (tag, name, col, fill, ser, mov, ai, note, serial) in ROWS:
        h = 126
        f.box(0, y, W, h, fill, col, 8)
        f.t(16, y + 26, '%s %s' % (tag, name), "svglbl", col, size=14)
        # 时间线
        if serial:
            for i in range(6):
                x = 300 + i * 116
                lb = "t%d" % (i + 1) if i < 5 else "tn"
                f.cell(x, y + 44, 74, 34, lb, col, "#fff", size=_sz(12))
                if i:
                    if i == 5:
                        f.t(x - 21, y + 66, "…", fill=col, bold=True, size=16,
                            anchor="middle")
                    else:
                        f.line(x - 42, y + 61, x - 6, y + 61, col, 1.6)
                # ⭐ Ⓒ 那一行「变长」不能只写在字里 —— 画出来才看得见：
                #   每步的 KV 条随 t 增长，⛔ 这是 Ⓐ 和 Ⓒ 唯一的区别所在。
                if serial == "grow":
                    bw = 12 + i * 12
                    f.box(x + 37 - bw // 2, y + 86, bw, 7, "#d0a3f0", PU, 2)
                    f.t(x + 37, y + 110, "W ＋ KV", fill=col, size=_sz(11),
                        anchor="middle")
                else:
                    f.t(x + 37, y + 100, "搬 W", fill=col, size=_sz(11), anchor="middle")
        else:
            f.box(300, y + 44, 764, 34, "#fff", col, 5)
            f.t(682, y + 66, "t1 … tn　全部一起算", fill=col, bold=True,
                size=14, anchor="middle")
            f.t(682, y + 100, "搬 W 一次", fill=col, size=_sz(11), anchor="middle")
        f.t(16, y + 50, ser, fill=col, bold=True, size=_sz(12))
        f.t(16, y + 70, mov, fill=GY, size=_sz(11))
        f.t(16, y + 88, ai, fill=col, bold=True, size=_sz(12))
        f.t(1088, y + 66, note, fill=col, bold=True, size=_sz(12))
        y += h + 8

    f.box(0, y, W, 140, "#e8f0fe", BL, 8)
    f.t(16, y + 26, '⭐⭐ 把 Ⓐ 和 Ⓒ 摆在一起看 ——&#160;'
                    '<tspan font-weight="700">这就是整个专题三的舞台</tspan>',
        "svglbl", "#174ea6", size=14)
    f.t(16, y + 54, '两边都是「一步一个，每步把权重搬一遍」。'
                    '<tspan font-weight="700">唯一的区别在「每步还得额外搬什么」：</tspan>',
        fill="#174ea6", size=_sz(12))
    f.t(40, y + 78, '· RNN：一个<tspan font-weight="700">固定大小</tspan>的状态 h ——&#160;'
                    '<tspan font-weight="700">跟上下文多长完全无关</tspan>',
        fill=RD, size=_sz(12))
    f.t(40, y + 98, '· Transformer：一路<tspan font-weight="700">线性变长</tspan>的 '
                    'KV cache ——&#160;'
                    '<tspan font-weight="700">128K 上下文时它能比权重本身还大</tspan>',
        fill=PU, size=_sz(12))
    f.t(16, y + 124, '⭐ 所以后面三个旋钮拧的全是同一件事：'
                     '<tspan font-weight="700">让 Ⓒ 这一行每步要搬的东西变小</tspan>'
                     '——&#160;存少点（①）、看少点（②）、'
                     '<tspan font-weight="700">或者干脆换回一个固定大小的状态（③）</tspan>。',
        fill="#174ea6", bold=False, size=_sz(12))
    f.save("fig3-rnn-decode.svg", y + 152)


# ══════════════════════════════════════════════════════════════════
# 图四：三个痛点各自通向哪条路
# ══════════════════════════════════════════════════════════════════
def fig_pain():
    W = 1400
    f = Fig(W, "RNN 的三个痛点各自通向后来的哪条技术路线：串行通向 Transformer，"
               "梯度消失通向门控与注意力的短路径，固定大小的状态通向注意力机制；"
               "而线性注意力与 Mamba 是把第一条反过来再走一遍")
    f.t(0, 18, 'RNN 的三个痛点 ——&#160;<tspan font-weight="700">'
               '后面三十年的路线图，就是这三条各自的解药</tspan>',
        "svglbl", INK, size=16)
    f.t(0, 40, '⭐ 这一页是路标：<tspan font-weight="700">后面每一个变体，'
               '都能追回到这三条里的某一条。</tspan>', fill=GY, size=_sz(12))

    ROWS = (
        ("①", "串行：算不快",
         "h<tspan baseline-shift=\"sub\" font-size=\"10\">t</tspan> 要等 "
         "h<tspan baseline-shift=\"sub\" font-size=\"10\">t−1</tspan>，"
         "序列多长就排多少轮；<tspan font-weight=\"700\">权重每轮重搬一次</tspan>",
         "Transformer 把循环整个拿掉",
         "<tspan font-weight=\"700\">代价是注意力矩阵变成 O(N²)</tspan> ——&#160;"
         "本专题三个旋钮都是在还这笔账", RD, "#fce8e6"),
        ("②", "梯度消失：记不住",
         "反向传播要连乘 n 次，小于 1 就指数衰减（Bengio 1994）"
         "——&#160;<tspan font-weight=\"700\">学不到远处的依赖</tspan>",
         "先是门控（LSTM 1997 / GRU 2014）",
         "后来注意力更彻底：<tspan font-weight=\"700\">任意两个位置只隔一步</tspan>"
         "（Vaswani 表 1 的最长路径 O(1) vs O(n)）", OR, "#fef7e0"),
        ("③", "状态是固定大小：装不下",
         "不管序列多长，全部历史压进一个 [d] 的向量 ——&#160;"
         "<tspan font-weight=\"700\">长句子必然丢信息</tspan>",
         "Bahdanau 2014：别只看最后那个向量",
         "让解码每一步<tspan font-weight=\"700\">回头看整段编码</tspan> ——&#160;"
         "⭐ <tspan font-weight=\"700\">注意力的出生证明：它一开始只是 RNN 的一个补丁</tspan>",
         GR, "#e6f4ea"),
    )
    y = 64
    for (num, name, hurt, cure, tail, col, fill) in ROWS:
        f.box(0, y, W, 92, fill, col, 8)
        f.t(18, y + 38, num, fill=col, bold=True, size=28)
        f.t(56, y + 28, name, "svglbl", col, size=14, bold=True)
        f.t(56, y + 54, hurt, fill=GY, size=_sz(12))
        f.line(500, y + 46, 536, y + 46, col, 1.6)
        f.t(554, y + 28, cure, "svglbl", col, size=14, bold=True)
        f.t(554, y + 54, tail, fill=GY, size=_sz(12))
        y += 100

    f.box(0, y, W, 130, "#f3e8fd", PU, 8)
    f.t(16, y + 26, '⭐⭐ 而第 ① 条后来<tspan font-weight="700">被反着又走了一遍</tspan>'
                    '——&#160;这就是专题三真正的主脊', "svglbl", PU, size=14)
    f.t(16, y + 54, 'Transformer 用「放弃状态」换来了并行度。'
                    '<tspan font-weight="700">线性注意力和 Mamba 这一支，'
                    '是想把状态请回来</tspan>——&#160;因为有状态，每步要搬的东西才不再变长。',
        fill=PU, size=_sz(12))
    f.t(16, y + 78, '⛔ 但状态一回来，<tspan font-weight="700">串行也跟着回来</tspan>。'
                    '于是又得把并行度找回来：<tspan font-weight="700">chunk 化、parallel scan</tspan>。'
                    'Martin &amp; Cundy 2018 的说法是：'
                    '<tspan font-weight="700">非线性的依赖挡住了并行，'
                    '只有线性依赖能用 parallel scan 扫</tspan>。', fill=PU, size=_sz(12))
    f.t(16, y + 106, '⭐ 所以后面 DeltaNet / GDN / KDA 的公式为什么必须长成那样，答案在这里：'
                     '<tspan font-weight="700">它们得线性到能被 scan，'
                     '否则就退回 1990 年那条链。一个完整的圆。</tspan>', fill=PU, size=_sz(12))
    f.save("fig3-rnn-pain.svg", y + 142)


fig_unroll()
fig_hw()
fig_decode()
fig_pain()
