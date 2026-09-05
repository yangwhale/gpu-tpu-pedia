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
         'GB300 的 64 张卡各出 18 条 NVLink，一条接一台 NVSwitch，共 18 台交换机；'
         '图上点名四颗、各用一种颜色画满 18 条，并高亮其中一台交换机看到四色在它上面汇合；'
         '交换机彼此不互连，所以是 18 条互不相干的轨，任意两点经一台交换机一跳可达、有 18 条并行路；TPU v7 的 64 颗排成 4×4×4 的三维立方体，'
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
t(LX + 20, TOP + 52, '<tspan font-weight="700">每颗 GPU 18 条，一条接一台交换机 ＝ 18 条轨</tspan>',
  None, "#174ea6")
t(LX + CW - 20, TOP + 52, '9 tray × 2 ＝ 18 颗', None, GY2, None, "end")

# ⭐⭐ 2026-09-05 第三版：按**真实封装**画，不再是抽象的两排方块。
#    厂商 deck 画 NVL72 一律是「机架立面」——&nbsp;而机架里真实的分组是：
#      · 交换层：**9 个 switch tray，每个 2 颗 NVSwitch ＝ 18 颗**
#      · 计算层：**每个 compute tray 4 颗 GPU**（我们这 64 颗 ＝ 16 个 tray）
#    把这两层分组画出来，「18」和「64」就不再是两个抽象的数，
#    而是**你在机架上能数出来的东西**。
SWY = TOP + 78
TRAY_N, TRAY_W, TRAY_G = 9, 62, 8            # 9 个 switch tray
TR_X0 = LX + (CW - (TRAY_N * TRAY_W + (TRAY_N - 1) * TRAY_G)) / 2
box(TR_X0 - 14, SWY - 13, TRAY_N * TRAY_W + (TRAY_N - 1) * TRAY_G + 28, 58,
    "#eef4fe", BL, 8, 1.3)
CHIP = []                                     # 18 颗芯片的中心 x
for tr in range(TRAY_N):
    tx = TR_X0 + tr * (TRAY_W + TRAY_G)
    box(tx, SWY - 4, TRAY_W, 42, "#dce9fd", BL, 6, 1.0)
    for c in range(2):
        cx = tx + 6 + c * 26
        box(cx, SWY + 4, 24, 26, "#a8c7fa", BL, 4, 1.0)
        CHIP.append(cx + 12)

# ⛔⛔ 2026-09-05 第四版。第三版只把**一颗** GPU 的 18 条画满，Chris 当场否掉：
#    「你就一个节点连 18 条尾，谁能看出来你这 18 条尾又连了谁？
#      好歹画四个。颜色也得区分。轨与轨之间什么关系，这些都得表示明白。」
#    他说的是这张图**没有回答自己提出的问题**：
#    扇出去容易画，扇出去之后**到达了谁**才是拓扑的全部意义。
#    ⭐ 于是这一版补三样，缺一样这张图就白画：
#      ① **四颗**同时画满，四种颜色 ——&nbsp;让「每颗都连满 18 台」变成看得见的规律，
#         而不是「那一颗比较特殊」
#      ② **点名一台交换机**，把四种颜色在它上面汇合的那四条画粗 ——&nbsp;
#         这才是「连到谁」：任取一台，四颗（其实全部 64 颗）都在上面
#      ③ **说清轨与轨的关系**：交换机之间**没有链路**，所以 18 条轨是 18 个
#         互不相干的平面，每条各自完整连着全部 64 颗 → 任意两颗之间 18 条并行路
GY_ = TOP + 312
GT_N, GT_W, GT_G = 8, 72, 8                   # 每排 8 个 compute tray
GT_X0 = LX + (CW - (GT_N * GT_W + (GT_N - 1) * GT_G)) / 2
# 被点名的四颗：(排, tray, tray 内序号, 颜色)。
# ⚠️ 颜色不许用 TPU 绿 #1e8e3e ——&nbsp;全课把绿钉给了 TPU，在 GPU 这半张里
#    出现绿色会被当成跨平台线索。所以取 蓝／橙／紫／青 四个互不相邻的色相。
NAMED = [(0, 1, 1, "#1a73e8"), (0, 4, 2, "#e8710a"),
         (0, 7, 0, "#8430ce"), (1, 3, 2, "#00838f")]
FAN = []
for row in range(2):
    for tr in range(GT_N):
        tx = GT_X0 + tr * (GT_W + GT_G)
        ty = GY_ + row * 34
        box(tx, ty, GT_W, 26, "#eef4fe", BL, 5, 0.9)
        for g in range(4):
            gx = tx + 5 + g * 16
            col = next((c for r, t_, g_, c in NAMED
                        if r == row and t_ == tr and g_ == g), None)
            box(gx, ty + 5, 13, 16, col or "#cfe0fc", col or BL, 2,
                1.4 if col else 0.7)
            if col:
                FAN.append((gx + 6.5, ty + 5, col))
t(LX + CW - 20, GY_ - 10, '16 个 compute tray × 4 颗 ＝ 64', None, GY2, None, "end")

# ── 四把扇子。淡画全部 72 条，只把「汇合在同一台」的那四条画实 ──────────
RAIL = 9                                      # 点名第 10 台交换机（0 起）
for fx, fy, col in FAN:
    for c in range(18):
        if c == RAIL:
            continue
        p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" '
                 'stroke-width="0.75" opacity=".3"/>' % (fx, fy, CHIP[c], SWY + 30, col))
for fx, fy, col in FAN:
    line(fx, fy, CHIP[RAIL], SWY + 30, col, 2.0)
# 给那台交换机加个圈，否则「汇合」这件事只有画线的人知道
p.append('<circle cx="%s" cy="%s" r="20" fill="none" stroke="%s" '
         'stroke-width="2" stroke-dasharray="4 3"/>' % (CHIP[RAIL], SWY + 17, RD))
box(CHIP[RAIL] + 26, SWY + 40, 214, 38, "#ffffff", "#f6c7c3", 6, 1.0)
t(CHIP[RAIL] + 36, SWY + 58, '第 10 台交换机 ＝ 第 10 条轨', "svglbl", RD, 12)
t(CHIP[RAIL] + 36, SWY + 72, '四色都汇到这一台，其余 60 颗也在', None, RD, 11)

# ⚠️ 这一行 2026-09-05 缩过一次：原句冲出左半栏了。
#    **版面 lint 只查 SVG 外框，查不到「分栏内溢出」** ——&nbsp;两栏图里的长句要自己数。
t(LX + 20, GY_ + 84,
  '四色 ＝ 被点名的四颗，<tspan font-weight="700">每颗都连满 18 台</tspan>；'
  '其余 60 颗一样（1,152 条全画成一团）', None, GY2)

# ── 轨与轨之间是什么关系 ──────────────────────────────────────────────
box(LX + 20, GY_ + 96, CW - 40, 78, "#fff8e1", YL, 8, 1.2)
t(LX + 36, GY_ + 118, '一条轨长什么样，轨与轨之间又是什么关系', "svglbl", "#7a5000", 13)
t(LX + 36, GY_ + 138, '· <tspan font-weight="700">一条轨 ＝ 一台交换机 ＋ 它到全部 64 颗的链路</tspan>，'
                      '也就是一张<tspan font-weight="700">完整的星</tspan>', None, "#7a5000")
t(LX + 36, GY_ + 156, '· <tspan font-weight="700">交换机彼此之间没有链路</tspan>　→　'
                      '18 条轨互不相干，谁也不经过谁（单层非阻塞）', None, "#7a5000")

box(LX + 20, GY_ + 182, CW - 40, 74, "#e8f0fe", None, 8, 0)
t(LX + 36, GY_ + 204, '⭐ 于是「任意两点一跳」在图上是<tspan font-weight="700">数出来的</tspan>：'
                      '任取两颗，它们在<tspan font-weight="700">每一条轨上都碰头</tspan>，',
  "svglbl", "#174ea6", 13)
t(LX + 36, GY_ + 224, '所以中间永远只隔一台交换机，而且<tspan font-weight="700">有 18 条并行的路</tspan>'
                      '可选 ——&#160;<tspan font-weight="700">谁跟谁都一样，位置无关</tspan>。',
  "svglbl", "#174ea6", 13)
t(LX + 36, GY_ + 244, '<tspan font-weight="700">64 张卡</tspan>，而域的上限是 '
                      '<tspan font-weight="700">72</tspan> ——&#160;'
                      '<tspan font-weight="700">64 整个装得下同一个域</tspan>。', None, "#174ea6")

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

# ⛔⛔ 2026-09-05 第四版。第三版用 2:1 等轴测，Chris：「实在是太丑陋了。」
#    **病根不是画得不够细，是投影选错了。** 2:1 等轴测把三个轴画成互成 120°、
#    长度相同 ——&nbsp;三个方向视觉上完全对等，于是大脑把点阵读成**一张六边形网格**，
#    再怎么加景深也立不起来。
#    ⭐ 换成**斜二测（cabinet）**：x 纯水平、z 纯竖直、y 缩短并斜着往里。
#      三个轴视觉上**不对等**，这正是人手画盒子的画法，一眼就是立体。
#    再补三样让它真的像个体：**外框线框立方体** ＋ **三面淡色底** ＋ **坐标轴三脚架**。
#    ⭐ 不重合可证：两点同位需 70Δi＋34Δj＝0 且 27Δj＋70Δk＝0；
#      后式给 Δj ＝ −70Δk/27，|Δj|≤3 时只有 Δk＝0 → 全零解。
U, DX, DY, WZ = 62, 30, 24, 62     # x 步长 / y 的横竖分量（缩短＋倾斜）/ z 步长
OX0, OY0 = RX + 186, TOP + 358     # (0,0,0) 落点＝立方体左前下角


def proj(i, j, k):
    return OX0 + i * U + j * DX, OY0 - j * DY - k * WZ


def poly(pts, fill, op, stroke=None, sw=1.0):
    d = " ".join("%.1f,%.1f" % proj(*q) for q in pts)
    p.append('<polygon points="%s" fill="%s" fill-opacity="%.2f" stroke="%s" '
             'stroke-width="%.1f"/>' % (d, fill, op, stroke or "none", sw))


# ① 三个朝向观察者的面先铺一层淡底 —— 「体」的感觉九成来自这一步
poly([(0, 0, 0), (3, 0, 0), (3, 0, 3), (0, 0, 3)], GR, .07)   # 前面 j=0
poly([(0, 0, 3), (3, 0, 3), (3, 3, 3), (0, 3, 3)], GR, .13)   # 顶面 k=3
poly([(3, 0, 0), (3, 3, 0), (3, 3, 3), (3, 0, 3)], GR, .04)   # 右面 i=3

# ② 点阵的边：按深度 j 从远到近，远的细而淡
E = []
for k in range(4):
    for j in range(4):
        for i in range(4):
            for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                if i + di < 4 and j + dj < 4 and k + dk < 4:
                    E.append(((j + (j + dj)) / 6.0, (i, j, k), (i + di, j + dj, k + dk)))
for d, aa, bb in sorted(E, key=lambda e: -e[0]):
    x0, y0 = proj(*aa); x1, y1 = proj(*bb)
    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="%.2f" opacity="%.2f"/>'
             % (x0, y0, x1, y1, GR, 1.5 - 0.6 * d, 0.8 - 0.5 * d))

# ③ 外框线框：12 条棱加粗，立方体的轮廓一出来就不会被读成平面网格
C = [(0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0),
     (0, 0, 3), (3, 0, 3), (3, 3, 3), (0, 3, 3)]
for a_, b_ in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
    x0, y0 = proj(*C[a_]); x1, y1 = proj(*C[b_])
    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="1.6" opacity=".55"/>' % (x0, y0, x1, y1, "#0d652d"))

# ④ 结点：越靠后越小越淡
for d, i, j, k in sorted(((j / 3.0, i, j, k) for k in range(4)
                          for j in range(4) for i in range(4)), key=lambda n: -n[0]):
    x0, y0 = proj(i, j, k); r = 8.4 - 2.6 * d
    p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2.6" '
             'fill="%s" stroke="%s" stroke-width="%.2f" opacity="%.2f"/>'
             % (x0 - r, y0 - r, 2 * r, 2 * r,
                "#ffffff" if d > .5 else "#cfe9d7", GR, 1.6 - 0.6 * d, 1 - 0.25 * d))

# ⑤ 坐标轴三脚架 —— 告诉眼睛哪个方向是「往里」
AX, AY = RX + 62, TOP + 300
for dx, dy, lab, lx, ly in ((52, 0, 'x', 8, 5), (0, -52, 'z', -4, -8),
                            (30, -24, 'y（往里）', 6, -2)):
    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="1.6"/>' % (AX, AY, AX + dx, AY + dy, GY2))
    p.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"/>' % (AX + dx, AY + dy, GY2))
    t(AX + dx + lx, AY + dy + ly, lab, "svglbl", GY2, 11)

# ⑥ 高亮一颗，把它的 6 条邻居链路画出来。
#    ⭐ 这是跟左半张的**正面对照**：左边一颗出 18 条、条条通交换机；
#      右边一颗出 6 条、条条只到邻居。两个数在同一张图的同一个位置上，
#      「交换式的域」和「直连的网」就不用解释了。
#    ⚠️ 标签不再压在点阵上（上一版就是压着的），改成引线拉到体外。
HI3 = (1, 1, 1)
hx0, hy0 = proj(*HI3)
for di, dj, dk in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
    x1, y1 = proj(HI3[0]+di, HI3[1]+dj, HI3[2]+dk)
    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#0d652d" '
             'stroke-width="2.6"/>' % (hx0, hy0, x1, y1))
    p.append('<circle cx="%.1f" cy="%.1f" r="4.8" fill="#0d652d"/>' % (x1, y1))
p.append('<rect x="%.1f" y="%.1f" width="17" height="17" rx="3.5" fill="#0d652d"/>'
         % (hx0 - 8.5, hy0 - 8.5))
LBX, LBY = RX + CW - 150, TOP + 300
line(hx0 + 10, hy0 + 6, LBX - 6, LBY - 4, "#0d652d", 1.0)
t(LBX, LBY, '这一颗的 <tspan font-weight="700">6 条</tspan>', "svglbl", "#0d652d", 12)
t(LBX, LBY + 15, '全是直连邻居', None, "#0d652d", 11)

# ⑦ 三条环绕：贴着体外走，每个方向一条
for a_, b_, c0, c1, lab, lxy in (
        # ⚠️ 控制点必须**同号同向**，等于沿弦的法线整体外推 ——&nbsp;这样才是一道
        #    贴着体外的浅弓。上一版两端一正一负，弧先往外甩再拐回来，
        #    三条叠在一起成了两个大椭圆，看不出各自连的是哪两颗。
        ((3, 0, 0), (0, 0, 0), (0, 44), (0, 44), 'x 环绕', (OX0 + 74, OY0 + 52)),
        ((0, 3, 0), (0, 0, 0), (-26, -32), (-26, -32), 'y 环绕', (OX0 - 104, OY0 - 30)),
        ((3, 3, 3), (3, 3, 0), (58, 0), (58, 0), 'z 环绕', (OX0 + 272, OY0 - 150))):
    (x0, y0), (x1, y1) = proj(*a_), proj(*b_)
    path('M%.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f'
         % (x0, y0, x0 + c0[0], y0 + c0[1], x1 + c1[0], y1 + c1[1], x1, y1), RD, 1.8, "5 4")
    t(lxy[0], lxy[1], lab, "svglbl", RD, 11)

# ⑧ 小插图：一维上 4 个点接成环，最远 2 步 —— 「2＋2＋2」的分子在这儿
RCX, RCY, RR = RX + 92, TOP + 152, 34
p.append('<circle cx="%s" cy="%s" r="%s" fill="none" stroke="%s" stroke-width="1.4" '
         'stroke-dasharray="4 3" opacity=".5"/>' % (RCX, RCY, RR, RD))
RP = [(RCX, RCY - RR), (RCX + RR, RCY), (RCX, RCY + RR), (RCX - RR, RCY)]
for n, (qx, qy) in enumerate(RP):
    p.append('<rect x="%.1f" y="%.1f" width="13" height="13" rx="3" fill="%s" '
             'stroke="%s" stroke-width="1.4"/>'
             % (qx - 6.5, qy - 6.5, "#0d652d" if n in (0, 2) else "#cfe9d7", GR))
t(RCX, TOP + 100, '一维上的 4 个点', "svglbl", "#0d652d", 12, "middle")
t(RCX, TOP + 205, '首尾一接 ——&#160;<tspan font-weight="700">最远 2 步</tspan>', None, "#0d652d", 11, "middle")
t(RCX, TOP + 220, '（不接环绕要 3 步）', None, GY2, 10, "middle")

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
