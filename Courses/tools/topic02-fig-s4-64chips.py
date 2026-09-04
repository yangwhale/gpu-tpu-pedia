# -*- coding: utf-8 -*-
"""图 4-1 · 64 颗连在一起：两边连的方式不一样。

⭐ **为什么要有这一张。** 2026-09-05 Chris 讲完全课之后：

    「从头到尾都没有讲过 NVLink 和 ICI，这个是非常重要、恨不得一半重要的话题。
      你可以都拿 64 个芯片的互联来想一想。」

⭐ 用「64 颗」这个角度一想，缺口比「没讲规格」严重得多：
   **这门课的招牌实测就是 64 对 64，而在 64 这个点上，
     两边的互连结构根本不是一回事 ——&nbsp;课程从头到尾没说过。**
   · GB300 那 64 张卡，**整个装在一个 NVL72 域里**（域上限 72）→&nbsp;任意两点一跳
   · TPU v7 那 64 颗，是一个 **4×4×4 的三维环面** →&nbsp;最远 6 跳
   于是 §6.3 把 854 和 662.3 并排摆出来的时候，读者手上缺一块前置。

════════════════════════════════════════════════════════════════
⛔ 分工：这张图只画**硬件拓扑**，通信模式整个归专题五
════════════════════════════════════════════════════════════════
Chris 定的边界（2026-09-05）：「我们是不是有一个专题专门讲并行方案的？
DP、EP、PP、FSDP 那些。详情应该放到那个专题里去。
**专题二就是从硬点的角度把拓扑给它画出来就行。**」

所以这张图**只给硬件事实**：几条链路、每条多宽、连成什么形状、最远几跳。
⛔ **不画** all-reduce / all-to-all 谁吃亏 ——&nbsp;那是「哪种切法配哪种拓扑」，
   属于专题五（那边已经有「all-to-all 是全连接通信模式，对拓扑最挑剔」）。
   这里只留一句指针。

⛔ **也不画快慢。** torus 上的集合通信实测这门课一次都没跑过（L300 §4.5 自陈）。
   拓扑数学能推的（跳数、链路数）才画，性能一个字不编。

📌 图上每个数的来处：
   · ICI：6 条链路 × 每条双向 200 GB/s ＝ 1,200 GB/s／chip（官方规格）
   · NVLink 5：18 条链路 × 每条双向 100 GB/s ＝ 1,800 GB/s／GPU（官方规格）
   · NVL72 域上限 72 颗；ICI 最大 9,216 颗（两边官方）
   · 4×4×4 环面直径 ＝ 2＋2＋2 ＝ 6 跳（每维 4 个点，环绕后最远走 2 步）
   · 64 ≤ 72，所以 64 张 GB300 落在同一个 NVL72 域内 ——&nbsp;这是本图的题眼
"""
import io

BL, OR, GR, RD, GY, YL, PU = ("#1a73e8", "#e8710a", "#1e8e3e", "#d93025",
                              "#5f6368", "#f9ab00", "#8430ce")
GY2 = "#80868b"
W = 1400
p = []


def t(x, y, s, cls="svgsm", fill=None, size=None, anchor=None):
    p.append('<text class="%s" x="%s" y="%s"%s%s%s>%s</text>' % (
        cls, x, y, ' fill="%s"' % fill if fill else '',
        ' text-anchor="%s"' % anchor if anchor else '',
        ' style="font-size:%spx"' % size if size else '', s))


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=10, sw=1.0, dash=None):
    p.append('<rect x="%s" y="%s" width="%s" height="%s" rx="%s" fill="%s"'
             ' stroke="%s" stroke-width="%s"%s/>'
             % (x, y, w, h, r, fill, stroke, sw,
                ' stroke-dasharray="%s"' % dash if dash else ''))


def line(x1, y1, x2, y2, c=GY2, sw=1.2, dash=None):
    p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"%s/>'
             % (x1, y1, x2, y2, c, sw, ' stroke-dasharray="%s"' % dash if dash else ''))


TOP = 96
CH = 466
CW = 686
LX, RX = 0, 714
BY = TOP + CH + 26
H = BY + 132

p.append('<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="'
         '同样是 64 颗，GB300 的 64 张卡整个落在一个 NVL72 域内任意两点一跳，'
         'TPU v7 的 64 颗是 4×4×4 三维环面最远 6 跳">' % (W, H))

t(0, 16, '64 颗连在一起 ——&#160;<tspan font-weight="700">同样是 64，两边连的方式不是一回事</tspan>',
  "svglbl", "#202124", 14)
t(0, 36, '这门课那组招牌实测（§6.3）就是 64 对 64。摆那两个数之前，'
         '得先知道这 64 颗<tspan font-weight="700">各自是怎么连起来的</tspan>。')
t(W, 16, '⛔ 这张图只说结构，不说快慢', "svglbl", RD, 12, "end")
t(W, 36, '哪种通信模式吃亏、EP 为什么对拓扑最挑剔 →&#160;专题五', None, GY2, None, "end")

# ══════════ 左：NVL72 域 ══════════
box(LX, TOP, CW, CH, "#f7faff", BL, 12, 1.6)
t(LX + 20, TOP + 30, 'GB300 · NVLink 5 ＋ NVSwitch', "svglbl", BL, 16)
t(LX + 20, TOP + 52, '一个<tspan font-weight="700">交换式的域</tspan>：域内任意两点，'
                     '经过交换机<tspan font-weight="700">一跳可达</tspan>', None, "#174ea6")

# 交换层
box(LX + 60, TOP + 78, CW - 120, 40, "#e8f0fe", BL, 8, 1.4)
t(LX + CW / 2, TOP + 103, 'NVSwitch　·　域内非阻塞', "svglbl", BL, 14, "middle")

# 64 个方块：8 × 8
gx, gy, cell, gap = LX + 60, TOP + 142, 64, 8
for r in range(8):
    for c in range(8):
        x = gx + c * (cell + gap) / 1.0 * 0.98
        y = gy + r * 30
        box(x, y, 56, 22, "#e8f0fe", BL, 4, 0.9)
for c in range(8):
    line(gx + c * 70.6 + 28, TOP + 118, gx + c * 70.6 + 28, gy, BL, 0.8, "3 3")
t(LX + 20, gy + 254, '<tspan font-weight="700">64 张卡</tspan>　——&#160;'
                     '而域的上限是 <tspan font-weight="700">72</tspan>，'
                     '<tspan font-weight="700">64 整个装得下</tspan>', None, "#202124")
box(LX + 20, gy + 264, CW - 40, 46, "#e8f0fe", None, 8, 0)
t(LX + 36, gy + 285, '⭐ 所以在 64 这个规模上，GB300 这一侧', "svglbl", "#174ea6", 13)
t(LX + 36, gy + 303, '谁跟谁说话都是<tspan font-weight="700">一跳、同样的带宽</tspan>'
                     '——&#160;位置无关', "svglbl", "#174ea6", 14)

# ══════════ 右：4×4×4 环面 ══════════
box(RX, TOP, CW, CH, "#f5faf6", GR, 12, 1.6)
t(RX + 20, TOP + 30, 'TPU v7 · ICI 三维环面', "svglbl", GR, 16)
t(RX + 20, TOP + 52, '<tspan font-weight="700">没有交换机</tspan>：每颗只连自己的邻居，'
                     '远的要<tspan font-weight="700">一跳一跳走过去</tspan>', None, "#0d652d")

# 画 4×4×4 —— 用四层 4×4 网格并排，层间连线示意第三维
lay_x, lay_y, s = RX + 44, TOP + 92, 30
for L in range(4):
    ox = lay_x + L * 160
    t(ox + 1.5 * s, lay_y - 8, 'z ＝ %d' % L, None, GY2, None, "middle")
    for r in range(4):
        for c in range(4):
            box(ox + c * s, lay_y + r * s, s - 8, s - 8, "#e6f4ea", GR, 3, 0.9)
    # 层内 x/y 邻接
    for r in range(4):
        line(ox + 22, lay_y + r * s + 11, ox + 3 * s, lay_y + r * s + 11, GR, 0.7)
    for c in range(4):
        line(ox + c * s + 11, lay_y + 22, ox + c * s + 11, lay_y + 3 * s, GR, 0.7)
    # 环绕（wrap）用虚线示意
    p.append('<path d="M%d %d q 14 -18 0 -30" fill="none" stroke="%s" '
             'stroke-width="0.9" stroke-dasharray="3 3"/>'
             % (ox + 3 * s - 8, lay_y + 3 * s + 4, GR))
    if L < 3:
        line(ox + 3 * s + 4, lay_y + 1.5 * s, ox + 160, lay_y + 1.5 * s, GR, 1.1, "4 3")
t(RX + 20, lay_y + 148, '四层 4×4 并排画的是同一个 <tspan font-weight="700">4×4×4</tspan>；'
                        '虚线是<tspan font-weight="700">环绕连接</tspan>（首尾相接，所以叫环面）', None, GY2)

box(RX + 20, lay_y + 166, CW - 40, 76, "#fff", "#c3e2cc", 8)
t(RX + 36, lay_y + 190, '每一维 4 个点，环绕之后最远走 <tspan font-weight="700">2 步</tspan>', None, "#202124")
t(RX + 36, lay_y + 212, '三维加起来：', None, "#202124")
t(RX + 152, lay_y + 214, '2 ＋ 2 ＋ 2 ＝ 6 跳', "svglbl", GR, 17)
t(RX + 330, lay_y + 212, '——&#160;这是<tspan font-weight="700">最远</tspan>的一对；'
                         '近的邻居 1 跳', None, GY2)
t(RX + 36, lay_y + 234, '⭐ 也就是说：这一侧<tspan font-weight="700">谁跟谁说话，'
                        '贵不贵取决于离多远</tspan>', "svglbl", "#0d652d", 13)

box(RX + 20, lay_y + 250, CW - 40, 46, "#e6f4ea", None, 8, 0)
t(RX + 36, lay_y + 271, '⭐ 而同一套 ICI 一路铺到 <tspan font-weight="700">9,216 颗</tspan>，'
                        '中途不换协议', "svglbl", "#0d652d", 13)
t(RX + 36, lay_y + 289, '⚠️ 9,216 ÷ 72 ＝ 128 <tspan font-weight="700">只衡量能铺多远，'
                        '不衡量谁跑得快</tspan>', "svglbl", RD, 13)

# ══════════ 底带：链路账 ══════════
box(0, BY, W, 118, "#f8f9fa", "#dadce0", 10, 1.2)
t(20, BY + 26, '每颗对外的链路账 ——&#160;总量同一个量级，但拆开看结论是反的',
  "svglbl", "#202124", 14)
rows = [('TPU v7 · ICI', '6 条链路', '× 每条双向 200 GB/s', '＝ 1,200 GB/s / chip', GR),
        ('GB300 · NVLink 5', '18 条链路', '× 每条双向 100 GB/s', '＝ 1,800 GB/s / GPU', BL)]
for i, (nm, n, per, tot, c) in enumerate(rows):
    y = BY + 52 + i * 24
    t(20, y, nm, "svglbl", c, 13)
    t(210, y, n, "svglbl", "#202124", 13)
    t(320, y, per, None, GY2)
    t(560, y, tot, "svglbl", c, 13)
t(20, BY + 106, '⭐ 单条反而是 ICI <tspan font-weight="700">粗一倍</tspan>'
                '——&#160;NVIDIA 是靠<tspan font-weight="700">条数多三倍</tspan>把总量做到 1.5 倍的。'
                '　⚠️ 两个数<tspan font-weight="700">都是双向合计</tspan>，别当单链读。', None, "#202124")

p.append('</svg>')
io.open('fig4-1.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig4-1 ok  %d×%d' % (W, H))
