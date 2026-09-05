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
# ⛔ 2026-09-05：这里原来写死「§6.3」。这张图**同时注入 L200 和 L300**，
#    而两份文档的节号不一样（L200 实测在 6.3，L300 在 7.2）——
#    一张图进两份文档，**硬编码节号必然有一份是错的**。
t(0, 36, '这门课那组招牌实测就是 64 对 64。摆那两个数之前，'
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

# 画 4×4×4 —— 用四层 4×4 网格并排。
#
# ⛔⛔ 2026-09-05 重画。旧版有两个真问题，而且是**同一个问题的两面**：
#   ① 层与层之间那条长线画成了**虚线**，而图注写着「虚线是环绕连接」——
#      可它是 z 方向的**相邻**，不是环绕。同一种笔画表示两个含义，图注还挑错了一个。
#   ② 真正的环绕**几乎没画**：每层只有右下角一个小弯钩，x/y 的其余环绕没有，
#      z 的环绕（z=3 接回 z=0）一根都没有。
#   ⭐ 后果很具体：**整张图的「6 跳」结论完全挂在环绕上**
#      （没有环绕，4 个点一维最远走 3 步，三维就是 9 跳）。
#      学员照着图数，会数出 9，数不出 6 —— 图自己不支持自己的结论。
#   → 现在：**实线 ＝ 相邻，虚线短桩 ＝ 环绕**，两种笔画各管一件事；
#     并且把 z=0 那一层的四条 y 向环绕完整画出来当范例，其余用短桩表示同理。
lay_x, lay_y, s_ = RX + 44, TOP + 96, 30
for L in range(4):
    ox = lay_x + L * 160
    t(ox + 1.5 * s_, lay_y - 10, 'z ＝ %d' % L, None, GY2, None, "middle")
    for r in range(4):
        for c in range(4):
            box(ox + c * s_, lay_y + r * s_, s_ - 8, s_ - 8, "#e6f4ea", GR, 3, 0.9)
    # 层内 x / y 相邻 —— 实线
    for r in range(4):
        line(ox + 22, lay_y + r * s_ + 11, ox + 3 * s_, lay_y + r * s_ + 11, GR, 0.8)
    for c in range(4):
        line(ox + c * s_ + 11, lay_y + 22, ox + c * s_ + 11, lay_y + 3 * s_, GR, 0.8)
    # 环绕短桩 —— 虚线，四个方向都有（表示「接到那一头去」）
    for r in range(4):
        cy = lay_y + r * s_ + 11
        line(ox - 10, cy, ox, cy, GR, 1.0, "2 2")
        line(ox + 3 * s_ + 22, cy, ox + 3 * s_ + 32, cy, GR, 1.0, "2 2")
    for c in range(4):
        cx = ox + c * s_ + 11
        line(cx, lay_y - 8, cx, lay_y, GR, 1.0, "2 2")
        line(cx, lay_y + 3 * s_ + 22, cx, lay_y + 3 * s_ + 30, GR, 1.0, "2 2")
    # 层与层之间 —— 第三维的相邻，实线（旧版这里是虚线，跟环绕混了）
    if L < 3:
        line(ox + 3 * s_ + 32, lay_y + 1.5 * s_, ox + 160 - 10, lay_y + 1.5 * s_, GR, 1.2)

# z 方向的环绕：z=3 接回 z=0（旧版完全没画，而 6 跳里有 2 跳靠它）
ZW = lay_y + 3 * s_ + 44
p.append('<path d="M%d %d V%d H%d V%d" fill="none" stroke="%s" stroke-width="1.2" '
         'stroke-dasharray="4 3"/>'
         % (lay_x + 3 * 160 + 3 * s_ + 32, lay_y + 1.5 * s_, ZW, lay_x - 10, lay_y + 1.5 * s_,
            GR))
t(lay_x + 200, ZW + 14, 'z 方向同样首尾相接：z ＝ 3 的邻居就是 z ＝ 0', None, GR)

t(RX + 20, lay_y + 180, '<tspan font-weight="700">实线 ＝ 相邻</tspan>；'
                        '<tspan font-weight="700">虚线短桩 ＝ 环绕链路</tspan>'
                        '（接到那一头去，首尾相接，所以叫环面）', None, GY2)

box(RX + 20, lay_y + 198, CW - 40, 96, "#fff", "#c3e2cc", 8)
t(RX + 36, lay_y + 222, '每一维 4 个点。<tspan font-weight="700">不接环绕</tspan>最远要走 3 步，'
                        '<tspan font-weight="700">接上环绕</tspan>就只要 2 步', None, "#202124")
t(RX + 36, lay_y + 244, '三维加起来：', None, "#202124")
t(RX + 152, lay_y + 246, '2 ＋ 2 ＋ 2 ＝ 6 跳', "svglbl", GR, 17)
t(RX + 330, lay_y + 244, '——&#160;这是<tspan font-weight="700">最远</tspan>的一对；'
                         '近的邻居 1 跳', None, GY2)
t(RX + 36, lay_y + 266, '⭐ 也就是说：这一侧<tspan font-weight="700">谁跟谁说话，'
                        '贵不贵取决于离多远</tspan>', "svglbl", "#0d652d", 13)
# ⚠️ 这行原来还带一句「形状本身是调度的一部分」——&nbsp;文字顶出白框了，
#    而且那句「从一颗到一个 pod」那张图已经完整讲过（切片必须是连续立方体）。删。
t(RX + 36, lay_y + 286, '⚠️ 那 6 跳全靠环绕撑着 ——&#160;'
                        '所以申请的是「一个 4×4×4」，不是「64 颗」', None, RD)

# ══════════ 底带：两个「1 跳」不是一个单位 + 链路账指到下一张 ══════════
#
# ⛔⛔ 2026-09-05：这条底带原来是一整笔链路账
#     （ICI 6 × 200 ＝ 1,200 对 NVLink 18 × 100 ＝ 1,800，
#      外加一句星标结论「单条反而是 ICI 粗一倍」）。撤掉，原因有两条：
#   ① **重复。** 紧挨着的下一张图（T-8）从头到尾就在算这笔账，
#      而且算得更细（每条链路、每轴、pod 上限）。两张图连着说同一件事。
#   ② **更要命的是没有口径警告。** T-8 有一整个红框写着
#      「1,200 GB/s 这个数官方自己写拧了」：官方正文写的是每轴双向 200，
#      三个轴 ＝ 600，对不上同一页表格里的 1,200；只有读成「每条链路 200」才自洽。
#      而这张图把它当**官方规格**印了出来，还在上面架了个星标结论。
#      **那句「单条粗一倍」完全建立在有争议的那个读法上** ——
#      换成官方原文那个读法，单条就是 100，跟 NVLink 一模一样，结论直接反过来。
#   ⭐ 形状：**推出来的东西被写成了「官方规格」。** 这是第一原则那条最典型的形状 ——
#     不是编了一个数，是把一个需要限定的读法说成了事实。
#   → 这一格改成只做本图**独有**的那件事：把两个「1 跳」不是一个单位说破。
# ⚠️ 指别的图一律**按名字**，不要写「下一张」——&nbsp;这些图 L200 和 L300 共用，
#    两边的前后顺序不一样（构建期有断言拦这类方位词）。
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
