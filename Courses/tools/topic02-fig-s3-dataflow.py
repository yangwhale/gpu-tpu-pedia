# -*- coding: utf-8 -*-
"""图 3-8 · 同一个矩阵乘，三种把它铺在硅上的方式。

⭐⭐ **这张图的来处，是全课到目前为止反应最大的一次现场提问。**
2026-09-06，Chris 顺着「为什么累加器是 M×N」一路追问：

    「左矩阵第一行跟右矩阵第一列乘加完，不就该放进累加器的一个格里吗？
      那我一共有这一行累加器不就够了？我干嘛要 M 乘 N 那么大的累加器？」

讲完之后他的原话是：**「这么核心、重要且 top1 的规则，怎么从来都没有人讲过？
他这个矩阵乘加单元居然是做了数学转换以后去算的，不是用朴素算法算的。」**

⭐ **卡点极其明确，而且是普遍的**：几乎所有人脑子里的矩阵乘都是
   **「先把一个点积算完」**（内积序）。而硬件跑的是
   **「k 放最外层、M×N 个格子同时往上涨」**（外积序）。
   两者算出的结果一模一样，**数据流完全不同** —— 这才是累加器必须二维的原因。

📌 三栏的分工，别改：
   ① 朴素（内积序）——&nbsp;起点，必须先画出来，否则读者不知道自己错在哪
   ② GPU（外积序）——&nbsp;M、N 摊在**空间**，K 走**时间**
   ③ TPU（脉动阵列）——&nbsp;K、N 摊在**空间**，**M 走时间**

⭐ **落点是那张「哪个维度在空间、哪个在时间」的对照表。**
   同一个数学，三种铺法；累加器的形状、复用率、乃至为什么 TPU 要 weight-stationary，
   全都是这张表的推论。

⛔ **两条准确性上的护栏，别在简化时丢掉：**
   1. GPU 那一栏画的是**算法级**的数据流。指令级上一条 MMA 已经在内部把
      atom_K（16／32／64）那一段规约掉了 ——&nbsp;**形状一样，只是成批做**。
      图上写明了，不要改成「每一步是一个秩一外积」。
   2. TPU 那一栏，K 方向的求和是**沿着阵列在空间上完成**的，
      每个 PE 做一次乘加把部分和往下传。**外部累加器只在跨 K 分块时才用到。**

📌 出处：矩阵乘分块与复用率是公开的经典结果；`tcgen05.mma` 跨 k 块累加
   （首块 ScaleOut::Zero、之后 ScaleOut::One）见 Colfax Research 的 Blackwell
   Tensor Memory 教程；MXU 是 256×256 weight-stationary 脉动阵列见
   Google Cloud TPU 架构文档。
"""
import io

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
PU, BR = "#8430ce", "#7a5000"
W = 1400
p = []


def wpx(s, size=11.5):
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


def t(x, y, s, cls="svgsm", fill=None, bold=False, size=None, anchor=None):
    st = []
    if size:
        st.append("font-size:%dpx" % size)
    p.append('<text class="%s" x="%d" y="%d"%s%s%s>%s</text>' % (
        cls, x, y, ' fill="%s"' % fill if fill else '',
        ' text-anchor="%s"' % anchor if anchor else '',
        ' style="%s"' % ';'.join(st) if st else '',
        '<tspan font-weight="700">%s</tspan>' % s if bold else s))


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=8, sw=1):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" '
             'stroke="%s" stroke-width="%s"/>' % (x, y, w, h, r, fill, stroke, sw))


def cellgrid(x, y, cols, rows, cw, labels=None, fill="#fff", stroke=GY,
             txtfill=None, sw=1, r=3, size=10):
    """画一个 cols×rows 的小格子阵。labels 按行优先给，None 表示空格。"""
    for rr in range(rows):
        for cc in range(cols):
            cx, cy = x + cc * cw, y + rr * cw
            f = fill(rr, cc) if callable(fill) else fill
            box(cx, cy, cw, cw, f, stroke, r, sw)
            if labels:
                s = labels[rr * cols + cc]
                if s:
                    t(cx + cw // 2, cy + cw // 2 + 4, s,
                      fill=txtfill or "#202124", anchor="middle", size=size)


def arrow(x1, y1, x2, y2, color=GY, dash=None, sw=1.6):
    p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"%s '
             'marker-end="url(#a-%s)"/>' % (x1, y1, x2, y2, color, sw,
                                            ' stroke-dasharray="%s"' % dash if dash else '',
                                            color.lstrip('#')))


p.append('<svg viewBox="0 0 %d 1268" width="100%%" role="img" aria-label="'
         '同一个矩阵乘的三种数据流：朴素内积序只要一个累加器但完全没有复用；'
         'GPU 把 M 和 N 摊在空间、K 走时间；TPU 把 K 和 N 摊在空间、M 走时间">' % W)
p.append('<defs>')
for c in (GY, OR, BL, GR, PU, RD):
    p.append('<marker id="a-%s" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % (c.lstrip('#'), c))
p.append('</defs>')

t(0, 17, '同一个矩阵乘，三种把它铺在硅上的方式 ——&#160;'
         '<tspan font-weight="700">结果一模一样，数据流完全不同</tspan>',
  "svglbl", "#202124", size=14)
t(0, 37, '⭐ <tspan font-weight="700">几乎所有人脑子里的矩阵乘都是「先把一个点积算完」'
         '——&#160;而硬件一个都不这么做。</tspan>'
         '这张图就是为了把那个差别摊开。全程用同一个小例子：'
         '<tspan font-weight="700">A 是 2×3，B 是 3×2，C 是 2×2</tspan>'
         '（M＝2，N＝2，K＝3）。')

CW = 34
ALAB = ['a₀₀', 'a₀₁', 'a₀₂', 'a₁₀', 'a₁₁', 'a₁₂']
BLAB = ['b₀₀', 'b₀₁', 'b₁₀', 'b₁₁', 'b₂₀', 'b₂₁']

# ══════════ 第一栏：朴素（内积序）════════════════════════════════════
Y1, H1 = 50, 220
box(0, Y1, W, H1, "#f8f9fa")
t(16, Y1 + 24, '① 朴素算法（内积序）——&#160;'
               '<tspan font-weight="700">你脑子里那个。一个累加器就够，但它几乎没有复用</tspan>',
  "svglbl", "#202124", size=13)
t(16, Y1 + 44, '写成代码就是三层循环，<tspan font-weight="700">k 在最里面</tspan>：'
               '先把 C 的一个格子彻底算完，再去下一个格子。', fill=GY)

# 设定图：C[0][0] = A 第 0 行 · B 第 0 列
GX, GY0 = 30, Y1 + 66
t(GX, GY0 - 6, 'A（2×3）', fill=BL, bold=True)
cellgrid(GX, GY0, 3, 2, CW, ALAB,
         fill=lambda r, c: "#d2e3fc" if r == 0 else "#fff", stroke=BL, txtfill="#174ea6")
t(GX + 3 * CW + 16, GY0 + CW, '×', "svglbl", "#202124", size=18)
BX = GX + 3 * CW + 36
t(BX, GY0 - 6, 'B（3×2）', fill=OR, bold=True)
cellgrid(BX, GY0, 2, 3, CW, BLAB,
         fill=lambda r, c: "#fce8b2" if c == 0 else "#fff", stroke=OR, txtfill=BR)
t(BX + 2 * CW + 16, GY0 + CW, '=', "svglbl", "#202124", size=18)
CX = BX + 2 * CW + 36
t(CX, GY0 - 6, 'C（2×2）', fill=PU, bold=True)
cellgrid(CX, GY0, 2, 2, CW, ['c₀₀', 'c₀₁', 'c₁₀', 'c₁₁'],
         fill=lambda r, c: "#efdcfb" if (r == 0 and c == 0) else "#fff",
         stroke=PU, txtfill="#5b1a9e")

TX = CX + 2 * CW + 40
t(TX, GY0 + 14, '<tspan font-weight="700">第 1 步</tspan>：拿 A 的<tspan fill="%s">'
                '<tspan font-weight="700">整个第 0 行</tspan></tspan>'
                '（3 个数）× B 的<tspan fill="%s"><tspan font-weight="700">整个第 0 列</tspan>'
                '</tspan>（3 个数），乘加完 ——&#160;得到 c₀₀ 一个格子，写出去。' % (BL, OR),
  fill="#202124")
t(TX, GY0 + 34, '<tspan font-weight="700">第 2 步</tspan>：还是 A 第 0 行，换 B 第 1 列，'
                '得到 c₀₁。'
                '⛔ <tspan fill="%s"><tspan font-weight="700">A 第 0 行那三个数，'
                '你又读了一遍。</tspan></tspan>' % RD, fill="#202124")
t(TX, GY0 + 54, '第 3、4 步同理，A 第 1 行也要读两遍。', fill="#202124")

box(TX, GY0 + 70, W - TX - 30, 88, "#fff", RD, 6)
t(TX + 14, GY0 + 92, '数一下这笔账', "svglbl", RD, size=12)
t(TX + 14, GY0 + 112, 'A 的 6 个数各读 2 遍 ＝ 12 次；B 的 6 个数各读 2 遍 ＝ 12 次。'
                      '<tspan font-weight="700">总共读 24 次，只换来 12 次乘加。</tspan>',
  fill="#202124")
t(TX + 14, GY0 + 132, '⛔ <tspan font-weight="700">复用率 0.5 ——&#160;读两次才干一次活。'
                      '所以真实硬件一个都不这么算。</tspan>'
                      '（累加器确实只要 1 个，代价是把操作数反复搬。）', fill=RD)
t(TX + 14, GY0 + 150, '⭐ 卡住很多人的正是这一步：'
                      '<tspan font-weight="700">「一个格子算完就退休」的画面太自然了，'
                      '自然到没人怀疑硬件不是这么干的。</tspan>', fill=GY)

# ══════════ 第二栏：GPU（外积序）═════════════════════════════════════
Y2, H2 = Y1 + H1 + 14, 320
box(0, Y2, W, H2, "#e8f0fe", BL)
t(16, Y2 + 24, '② GPU 的算法（外积序）——&#160;'
               '<tspan font-weight="700">把 k 挪到最外层，M×N 个格子同时往上涨</tspan>',
  "svglbl", "#174ea6", size=13)
t(16, Y2 + 44, '每走一步 k，只读 <tspan font-weight="700">A 的第 k 列（M 个数）'
               '＋ B 的第 k 行（N 个数）</tspan>，然后这些数<tspan font-weight="700">'
               '两两配对</tspan>，一次更新<tspan font-weight="700">全部 4 个格子</tspan>。',
  fill="#174ea6")

SHADE = ("#ede7f6", "#d7c4f0", "#b98de6")
for step in range(3):
    gx = 24 + step * 452
    gy = Y2 + 66
    t(gx, gy - 4, 'k ＝ %d' % step, "svglbl", "#174ea6", size=12)
    # B 的第 k 行（横条，放上方）
    cellgrid(gx + 44, gy + 8, 2, 1, CW, [BLAB[step * 2], BLAB[step * 2 + 1]],
             fill="#fce8b2", stroke=OR, txtfill=BR)
    t(gx + 44 + 2 * CW + 8, gy + 8 + 22, 'B 第 %d 行' % step, fill=OR, bold=True)
    # A 的第 k 列（竖条，放左侧）
    cellgrid(gx, gy + 50, 1, 2, CW, [ALAB[step], ALAB[3 + step]],
             fill="#d2e3fc", stroke=BL, txtfill="#174ea6")
    t(gx - 2, gy + 50 + 2 * CW + 16, 'A 第 %d 列' % step, fill=BL, bold=True)
    # C 累加器 2×2，填充深浅表示累加进度
    cellgrid(gx + 44, gy + 50, 2, 2, CW,
             ['＋1', '＋1', '＋1', '＋1'],
             fill=SHADE[step], stroke=PU, txtfill="#5b1a9e", sw=2)
    t(gx + 44 + 2 * CW + 8, gy + 50 + 22, '累加器', fill=PU, bold=True)
    t(gx + 44 + 2 * CW + 8, gy + 50 + 40, '4 格<tspan font-weight="700">全动</tspan>', fill=PU)
    t(gx, gy + 148, '读 <tspan font-weight="700">%d</tspan> 个数 →&#160;'
                    '做 <tspan font-weight="700">4</tspan> 次乘加 →&#160;'
                    '<tspan font-weight="700">4 个格子各涨 1／3</tspan>' % 4, fill="#174ea6")

box(24, Y2 + 232, W - 48, 74, "#fff", BL, 6)
t(38, Y2 + 254, '⭐ 三步走完，K 到头，四个格子<tspan font-weight="700">同时算完，一起倒出去</tspan>'
                '——&#160;<tspan font-weight="700">总共读 12 次，做 12 次乘加，复用率 1.0，'
                '整整好一倍</tspan>。', fill="#174ea6")
t(38, Y2 + 274, '⭐ <tspan font-weight="700">回答「每个格子被用到几次」：恰好 K 次，'
                '一次不多一次不少，左上和右下完全一样。</tspan>'
                '每走一步 k，所有格子同时被更新一次 ——&#160;'
                '累加器总更新次数 ＝ M×N×K ＝ 乘加总次数。', fill="#174ea6")
t(38, Y2 + 294, '⚠️ 这里画的是<tspan font-weight="700">算法级</tspan>的数据流。'
                '指令级上，一条 MMA 已经在内部把 K 方向的一小段（16／32／64 个）规约掉了'
                '——&#160;<tspan font-weight="700">形状完全一样，只是成批做</tspan>。', fill=GY)

# ══════════ 第三栏：TPU（脉动阵列）═══════════════════════════════════
Y3, H3 = Y2 + H2 + 14, 320
box(0, Y3, W, H3, "#e6f4ea", GR)
t(16, Y3 + 24, '③ TPU 的 MXU（脉动阵列）——&#160;'
               '<tspan font-weight="700">K 方向的求和不靠累加器，靠「沿着阵列往下流」</tspan>',
  "svglbl", "#0b6b30", size=13)
t(16, Y3 + 44, '<tspan font-weight="700">B 先装进阵列，装完就不动</tspan>'
               '（weight-stationary）。'
               '<tspan font-weight="700">A 一行一行地从左边流进去</tspan>，'
               '部分和一路向下累加，从底下出来时已经是算完的一行 C。', fill="#0b6b30")

AW = 46
AX, AY = 430, Y3 + 104
# A 的两行，从左边流入
t(96, AY - 14, 'A 的行（一次一行，走的是时间）', fill=BL, bold=True)
cellgrid(96, AY, 3, 2, 38, ALAB,
         fill="#d2e3fc", stroke=BL, txtfill="#174ea6", size=9)
t(96, AY + 2 * 38 + 18, 't ＝ 0 送第 0 行，t ＝ 1 送第 1 行', fill=BL)
for k in range(3):
    arrow(96 + 3 * 38 + 8, AY + 38, AX - 8, AY + k * AW + AW // 2, BL, dash="4 3", sw=1.4)

# 阵列本体：K（3 行）× N（2 列），每个 PE 里放 B 的一个元素
t(AX - 8, AY - 34, 'MXU 阵列 ——&#160;真机是 256×256', "svglbl", "#0b6b30", size=12)
t(AX - 8, AY - 14, 'K（3）＝ <tspan font-weight="700">行</tspan>，'
                   'N（2）＝ <tspan font-weight="700">列</tspan>'
                   '——&#160;<tspan font-weight="700">两个都是空间维</tspan>', fill="#0b6b30")
cellgrid(AX, AY, 2, 3, AW, BLAB, fill="#fce8b2", stroke=OR, txtfill=BR, sw=2, size=11)
# 部分和往下流
for j in range(2):
    for k in range(3):
        arrow(AX + j * AW + AW // 2, AY + k * AW + AW - 4,
              AX + j * AW + AW // 2, AY + (k + 1) * AW + 3, GR, sw=1.4)
    arrow(AX + j * AW + AW // 2, AY + 3 * AW + 3,
          AX + j * AW + AW // 2, AY + 3 * AW + 26, GR, sw=2)
t(AX + AW, AY + 3 * AW + 48, 'C 的一行，从底下出来', fill=PU, bold=True, anchor="middle")

# 右侧说明
EX = AX + 2 * AW + 210
box(EX, AY - 44, W - EX - 20, 190, "#fff", GR, 6)
t(EX + 14, AY - 22, '每个 PE 干的事', "svglbl", "#0b6b30", size=12)
t(EX + 14, AY - 2, '① 从左边接一个 a　② 乘上自己怀里那个 b　'
                    '③ <tspan font-weight="700">加上从上面流下来的部分和</tspan>',
  fill="#0b6b30")
t(EX + 14, AY + 16, '④ 把 a 往右传，把新的部分和往下传。'
                    '<tspan font-weight="700">就这四步，没有别的。</tspan>', fill="#0b6b30")
t(EX + 14, AY + 42, '⭐ <tspan font-weight="700">所以 K 方向那 3 项求和，是「空间上」完成的</tspan>'
                    '——&#160;三个 PE 各做一次乘加，', fill="#0b6b30")
t(EX + 14, AY + 60, '把部分和接力往下传。'
                    '<tspan font-weight="700">这 3 项不需要任何外部累加器。</tspan>',
  fill="#0b6b30")
t(EX + 14, AY + 86, '外部累加器只在<tspan font-weight="700">跨 K 分块</tspan>时才用到'
                     '——&#160;真机 K 常常几千，', fill="#0b6b30")
t(EX + 14, AY + 104, '阵列一次只吃得下 256，剩下的按块累加。', fill="#0b6b30")
t(EX + 14, AY + 128, '⭐ <tspan font-weight="700">M 走的是时间</tspan>：'
                     'A 的行一行一行进去，一行一行出来。', fill=PU)

t(16, Y3 + H3 - 14, '⭐ <tspan font-weight="700">复用的账两边其实一样</tspan>：'
                    'TPU 这边，<tspan font-weight="700">B 装一次被 M 行反复用，'
                    'A 的一个数进来一次被同一行的 N 个 PE 用</tspan>'
                    '——&#160;跟 GPU 那边靠二维累加器换来的复用，'
                    '<tspan font-weight="700">是同一笔账的两种物理实现</tspan>。', fill="#0b6b30")

# ══════════ 落点：对照表 ═════════════════════════════════════════════
Y4 = Y3 + H3 + 14
box(0, Y4, W, 258, "#fef7e0", OR)
t(16, Y4 + 24, '⭐ 落点：同一个数学，区别只在「哪个维度摊在空间上，哪个维度走时间」',
  "svglbl", BR, size=13)

COLS = (('', 300), ('① 朴素（内积序）', 300), ('② GPU（外积序）', 340), ('③ TPU（脉动阵列）', 420))
ROWS4 = (
    ('M（左矩阵的行）', '时间', '<tspan font-weight="700">空间</tspan>：累加器的行',
     '<tspan font-weight="700">时间</tspan>：一行一行流进去'),
    ('N（右矩阵的列）', '时间', '<tspan font-weight="700">空间</tspan>：累加器的列',
     '<tspan font-weight="700">空间</tspan>：阵列的列'),
    ('K（求和那一维）', '时间', '<tspan font-weight="700">时间</tspan>：累加多少步',
     '<tspan font-weight="700">空间</tspan>：阵列的行'),
    ('累加发生在哪', '一个寄存器里，反复加',
     'M×N 个格子里，<tspan font-weight="700">各被更新 K 次</tspan>',
     '<tspan font-weight="700">沿着阵列往下传</tspan>，边流边加'),
    ('要多大的累加器', '1 个', 'M×N 个（真机是<tspan font-weight="700">一个 tile</tspan>，'
                              '不是整个矩阵）', '只需跨 K 分块的那一份'),
    ('复用率', '<tspan fill="%s"><tspan font-weight="700">0.5</tspan></tspan>' % RD,
     'M×N ÷（M＋N）——&#160;128×256 的块约 <tspan font-weight="700">85</tspan>',
     'B 用 M 遍、A 用 N 遍，<tspan font-weight="700">账同上</tspan>'),
)
cx = [16]
for _, w_ in COLS[:-1]:
    cx.append(cx[-1] + w_)
ry = Y4 + 48
for i, (nm, w_) in enumerate(COLS):
    if nm:
        t(cx[i], ry, nm, fill=BR, bold=True)
p.append('<line x1="16" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
         % (ry + 8, W - 16, ry + 8, OR))
for r, row in enumerate(ROWS4):
    yy = ry + 28 + r * 22
    for i, v in enumerate(row):
        t(cx[i], yy, v, fill="#202124" if i else BR, bold=(i == 0))

t(16, Y4 + 216, '⭐ <tspan font-weight="700">一句话：GPU 把 M 和 N 摊在空间上、让 K 走时间；'
                'TPU 把 K 和 N 摊在空间上、让 M 走时间。</tspan>'
                '——&#160;这一个选择，决定了后面所有事：'
                '累加器要多大、tile 为什么是核心旋钮、'
                '以及 TPU 为什么非得让权重坐着不动。', fill=BR)
t(16, Y4 + 238, '⛔ <tspan font-weight="700">而三种铺法算出来的 C，一个数都不差。</tspan>'
                '<tspan fill="%s">「结果相同、数据流不同」正是这门课想让你养成的那种看法'
                '——&#160;规格表只写结果，快慢全在数据流里。</tspan>' % GY, fill=RD)

p.append('</svg>')
io.open('fig3-8.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-8 ok')
