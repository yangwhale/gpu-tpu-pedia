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
CH = 580
CW = 686
LX, RX = 0, 714
BY = TOP + CH + 26
H = BY + 132


def path(d, c=GY2, sw=1.2, dash=None, fill="none"):
    p.append('<path d="%s" fill="%s" stroke="%s" stroke-width="%s"%s/>'
             % (d, fill, c, sw, ' stroke-dasharray="%s"' % dash if dash else ''))


p.append('<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="'
         'GB300 的 64 张卡各出 18 条 NVLink，一条接一台 NVSwitch，共 18 台交换机，'
         '所以任意两点经一台交换机一跳可达；TPU v7 的 64 颗排成 4×4×4 的三维立方体，'
         '每颗只连六个邻居，三个方向首尾相接成环面，最远 6 跳">' % (W, H))

t(0, 16, '64 颗连在一起 ——&#160;<tspan font-weight="700">同样是 64，两边连的方式不是一回事</tspan>',
  "svglbl", "#202124", 14)
# ⛔ 2026-09-05：这里原来写死「§6.3」。这张图**同时注入 L200 和 L300**，
#    而两份文档的节号不一样 ——&nbsp;硬编码节号必然有一份是错的。
t(0, 36, '这门课那组招牌实测就是 64 对 64。摆那两个数之前，'
         '得先知道这 64 颗<tspan font-weight="700">各自是怎么连起来的</tspan>。')
t(W, 16, '⛔ 这张图只说结构，不说快慢', "svglbl", RD, 12, "end")
t(W, 36, '哪种通信模式吃亏、EP 为什么对拓扑最挑剔 →&#160;专题五', None, GY2, None, "end")

# ══════════════════════════════════════════════════════════════════════
# 左：NVL72 —— 二部图，这才是这套拓扑的标准画法
#
# ⛔⛔ 2026-09-05 重画。上一版把交换层画成一根横条 ＋ 八根虚线，下面摆 64 个方块。
#    Chris：「这个画的太垃圾，网上都有非常经典的图，左边至少应该画出 18 条、配 18 台交换机。」
#    旧版**把这套拓扑最要紧的事实画丢了**：
#      · 每颗 GPU 有 **18 条 NVLink**，**一条接一台不同的 NVSwitch**
#      · 机架里一共 **18 颗 NVSwitch 芯片**（9 个 switch tray × 2）
#    出处：NVIDIA NVL72 参考架构文档原话 ——&nbsp;
#    "Each GPU has 18 NVLink Fifth-Generation links, **one per in-rack NVSwitch**"。
#    ⭐ 这么画之后，「任意两点一跳」不再是标语，是**图上看得见的事**。
# ⚠️ 64 × 18 ＝ 1,152 条全画是一团糊。标准做法：**一颗画满，其余淡出**。
box(LX, TOP, CW, CH, "#f7faff", BL, 12, 1.6)
t(LX + 20, TOP + 30, 'GB300 · NVLink 5 ＋ NVSwitch', "svglbl", BL, 16)
t(LX + 20, TOP + 52, '<tspan font-weight="700">每颗 GPU 18 条 NVLink，一条接一台交换机</tspan>',
  None, "#174ea6")
t(LX + CW - 20, TOP + 52, '机架里正好 18 台　·　域内非阻塞', None, GY2, None, "end")

SWY = TOP + 78
SW_N, SW_W, SW_G = 18, 28, 7
SW_X0 = LX + (CW - (SW_N * SW_W + (SW_N - 1) * SW_G)) / 2
# ⭐ 机架外框 —— 让这一排一眼看出是「一个机架里的东西」，不是飘着的十八个方块
box(SW_X0 - 14, SWY - 12, SW_N * SW_W + (SW_N - 1) * SW_G + 28, 56,
    "#eef4fe", BL, 8, 1.3)
for k in range(SW_N):
    x = SW_X0 + k * (SW_W + SW_G)
    box(x, SWY, SW_W, 32, "#d2e3fc", BL, 5, 1.1)
    t(x + SW_W / 2, SWY + 21, str(k + 1), "svglbl", "#174ea6", 11, "middle")

GY_ = TOP + 320                     # GPU 那两排
GN_, GW, GG = 32, 17, 3
GX0 = LX + (CW - (GN_ * GW + (GN_ - 1) * GG)) / 2
HI = 6
HX = GX0 + HI * (GW + GG) + GW / 2
# 淡的先画，浓的压上去
for hj in (HI - 4, HI + 6):
    hx = GX0 + hj * (GW + GG) + GW / 2
    for k in (0, 5, 11, 17):
        sx = SW_X0 + k * (SW_W + SW_G) + SW_W / 2
        p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" '
                 'stroke-width="0.8" opacity=".2"/>' % (hx, GY_, sx, SWY + 32, BL))
for k in range(SW_N):
    sx = SW_X0 + k * (SW_W + SW_G) + SW_W / 2
    line(HX, GY_, sx, SWY + 32, BL, 1.15)
for row in range(2):
    for k in range(GN_):
        x = GX0 + k * (GW + GG)
        on = (row == 0 and k == HI)
        box(x, GY_ + row * 26, GW, 20, "#1a73e8" if on else "#e8f0fe", BL, 3,
            1.4 if on else 0.8)

t(LX + 20, GY_ + 68, '蓝色那一颗的 18 条画满了；其余 63 颗<tspan font-weight="700">每颗都一样</tspan>'
                     '（全画出来是 1,152 条，看不清）', None, GY2)
t(LX + 20, GY_ + 92, '<tspan font-weight="700">64 张卡</tspan>　——&#160;'
                     '而域的上限是 <tspan font-weight="700">72</tspan>，'
                     '<tspan font-weight="700">64 整个装得下</tspan>', None, "#202124")
box(LX + 20, GY_ + 104, CW - 40, 62, "#e8f0fe", None, 8, 0)
t(LX + 36, GY_ + 126, '⭐ 「任意两点一跳」在这张图上是<tspan font-weight="700">看得见的</tspan>：'
                      '任何两颗 GPU 都挂在<tspan font-weight="700">同一批交换机</tspan>上，',
  "svglbl", "#174ea6", 13)
t(LX + 36, GY_ + 148, '中间只隔一台 ——&#160;<tspan font-weight="700">谁跟谁都一样，位置无关</tspan>。',
  "svglbl", "#174ea6", 14)

# ══════════════════════════════════════════════════════════════════════
# 右：4×4×4 —— 标准 2:1 等轴测立方体
#
# ⛔⛔ 上一版摊成四层 4×4 并排 ——&nbsp;那是四张二维图，不是一个三维体。
#    Chris：「右边这个它至少是 3D 的一个立方体。」
# ⚠️ 第一次重画时用了 i 和 k 都往右的斜投影，**结点大面积重叠**，糊成一片。
#    换成标准 2:1 等轴测：x 向右下、y 向左下、z 向上。
#    ⭐ 这套投影下不会重合，可以证：设 u=46 v=26 w=44，
#      两点同位需 13·Δ(i+j) ＝ 22·Δk，而 Δ(i+j) 必为偶数 ——&nbsp;只有全零解。
# ⚠️ 环绕链路**只画三条**（每个方向一条）——&nbsp;学术画法的通行做法，全画会糊死。
box(RX, TOP, CW, CH, "#f5faf6", GR, 12, 1.6)
t(RX + 20, TOP + 30, 'TPU v7 · ICI 三维环面', "svglbl", GR, 16)
t(RX + 20, TOP + 52, '<tspan font-weight="700">没有交换机</tspan>：每颗只连六个邻居'
                     '（±x ±y ±z）', None, "#0d652d")
t(RX + 20, TOP + 74, '<tspan fill="#d93025" font-weight="700">红色虚线 ＝ 环绕链路</tspan>'
                     '（每个方向画一条示意，实际三个方向都首尾相接）', None, GY2)

U, V, WZ = 46, 30, 44   # ⚠️ 同列最小间隔 |60m−44k| ＝ 12px > 直径 11
OX, OY = RX + CW / 2, TOP + 232
def proj(i, j, k):
    return OX + (i - j) * U, OY + (i + j) * V - k * WZ

# 先边后点，k 从低到高（低的在下方＝更近，后画压住）
# ⭐ 2026-09-05：加景深。等轴测点阵如果所有点一样深一样大，看着是平的 ——
#    远近靠**颜色深浅 ＋ 尺寸**来分，这是等轴测图能不能立起来的关键一步。
#    depth 定义为「离观察者多远」：i+j 越小、k 越大 ＝ 越靠后。
def depth(i, j, k):
    return ((6 - (i + j)) + k * 2) / 12.0      # 0 ＝ 最近，1 ＝ 最远

# 边：先远后近，远的淡
edges = []
for k in range(4):
    for j in range(4):
        for i in range(4):
            for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                if i + di < 4 and j + dj < 4 and k + dk < 4:
                    edges.append((depth(i, j, k), (i, j, k), (i + di, j + dj, k + dk)))
for d, aa, bb in sorted(edges, key=lambda e: -e[0]):
    x0, y0 = proj(*aa); x1, y1 = proj(*bb)
    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="%.2f" opacity="%.2f"/>'
             % (x0, y0, x1, y1, GR, 1.35 - 0.55 * d, 0.75 - 0.42 * d))

nodes = sorted(((depth(i, j, k), i, j, k)
                for k in range(4) for j in range(4) for i in range(4)),
               key=lambda n: -n[0])
for d, i, j, k in nodes:
    x0, y0 = proj(i, j, k)
    r = 7.4 - 2.0 * d
    p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2.4" '
             'fill="%s" stroke="%s" stroke-width="%.2f" opacity="%.2f"/>'
             % (x0 - r, y0 - r, 2 * r, 2 * r,
                "#ffffff" if d > .45 else "#d7efdf", GR, 1.5 - 0.5 * d, 1 - 0.28 * d))

# 三条环绕弧：x 方向、y 方向、z 方向各一条
def wrap(a, b, c0, c1):
    """两端各给一个控制点偏移，画一条绕到体外的环绕弧。"""
    (x0, y0), (x1, y1) = a, b
    path('M%.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f'
         % (x0, y0, x0 + c0[0], y0 + c0[1], x1 + c1[0], y1 + c1[1], x1, y1),
         RD, 1.7, "5 4")
wrap(proj(3, 0, 3), proj(0, 0, 3), (52, -46), (-52, -46))     # x 环绕（顶层后缘）
wrap(proj(0, 3, 3), proj(0, 0, 3), (-52, -34), (-52, 34))     # y 环绕
wrap(proj(3, 3, 3), proj(3, 3, 0), (66, -8), (66, 8))         # z 环绕
t(RX + CW - 24, TOP + 108, 'x 环绕', "svglbl", RD, 11, "end")
t(RX + 26, TOP + 238, 'y 环绕', "svglbl", RD, 11)
t(RX + CW - 24, TOP + 340, 'z 环绕', "svglbl", RD, 11, "end")

RBY = TOP + 424
box(RX + 20, RBY, CW - 40, 80, "#fff", "#c3e2cc", 8)
t(RX + 36, RBY + 24, '每一维 4 个点。<tspan font-weight="700">不接环绕</tspan>最远走 3 步，'
                     '<tspan font-weight="700">接上环绕</tspan>只要 2 步', None, "#202124")
t(RX + 36, RBY + 46, '三维加起来：', None, "#202124")
t(RX + 152, RBY + 48, '2 ＋ 2 ＋ 2 ＝ 6 跳', "svglbl", GR, 17)
t(RX + 330, RBY + 46, '——&#160;最远的一对；近邻 1 跳', None, GY2)
t(RX + 36, RBY + 68, '⭐ 这一侧<tspan font-weight="700">谁跟谁说话，贵不贵取决于离多远</tspan>',
  "svglbl", "#0d652d", 13)
box(RX + 20, RBY + 90, CW - 40, 52, "#e6f4ea", None, 8, 0)
t(RX + 36, RBY + 112, '⚠️ 那 6 跳全靠环绕撑着 ——&#160;'
                      '所以申请的是「一个 4×4×4」，不是「64 颗」', "svglbl", RD, 13)
t(RX + 36, RBY + 132, '⭐ 同一套 ICI 一路铺到 9,216 颗，中途不换协议',
  "svglbl", "#0d652d", 13)

# ══════════════════════════════════════════════════════════════════════
# 底带：两个「1 跳」不是一个单位
box(0, BY, W, 118, "#f8f9fa", "#dadce0", 10, 1.2)
t(20, BY + 26, '⚠️ 两边的「1 跳」不是一个单位 ——&#160;这是这张图最容易被误读的地方',
  "svglbl", RD, 14)
t(20, BY + 52, 'GB300 那一跳', "svglbl", BL, 13)
t(150, BY + 52, '是「卡 →&#160;交换机 →&#160;卡」：两段链路 ＋ 一次交换。'
                '<tspan font-weight="700">代价与位置无关</tspan>，但它不等于零。', None, "#202124")
t(20, BY + 76, 'TPU v7 那一跳', "svglbl", GR, 13)
t(150, BY + 76, '是<tspan font-weight="700">一条直连线</tspan>，没有交换机。'
                '所以 6 跳指的是「走 6 条线」，<tspan font-weight="700">不是「慢 6 倍」</tspan>。',
  None, "#202124")
t(20, BY + 102, '⛔ 因此 <tspan font-weight="700">1 对 6 只能读成结构差别，读不出延迟比</tspan>。'
                '　每颗几条链路、每条多宽、这个数的官方口径为什么对不上 ——&#160;'
                '<tspan font-weight="700">「从一颗到一个 pod」那张图专门算这笔账</tspan>。', None, "#202124")

p.append('</svg>')
io.open('fig4-1.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig4-1 ok  %d×%d' % (W, H))
