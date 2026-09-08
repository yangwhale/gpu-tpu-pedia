# -*- coding: utf-8 -*-
r"""图 1-7 · **一次向量加，从头到尾（会动）** —— 让 0.17 变成看得见的东西。

════════════════════════════════════════════════════════════════════
⭐⭐ 为什么要有这一张（2026-09-08 现场定）
════════════════════════════════════════════════════════════════════

起因是看到 _How To Scale Your Model_ 里那张 `pointwise-product.gif`
（公开原文 jax-ml.github.io/scaling-book，MIT 许可）：
两个数组从 HBM 搬进 VMEM、切块进向量寄存器、向量单元做一次逐元素乘、
结果原路写回。**动起来之后，「搬了一大堆、只算了一下」是能看见的。**

⛔ **但那张 GIF 本身不进这门课**，三条理由（判断的是分量和风格，不是版权）：
   ① **7 MB 一张图**。整个 topic-02.html 才 1.3 MB —— 一张图把页面撑五倍。
   ② 橄榄黄的手绘风跟本课刚统一完的配色不是一套。
      ⭐ 引入第二套视觉语言的代价不是「这一张不好看」，
        是**从此这门课有两种画法**，后面每张图都要选边。
   ③ 本课已经有 **T-6「一个数走完全程」**，画的就是同样五站
      （HBM → VMEM → 向量寄存器 → MXU → 累加器），而且每站还多问一句
      「这一步是谁决定的」。**同一个硬件出现两套画法，比只有一套差。**

⭐ 那**该拿的是什么**：T-6 是**静态**的，而且走的是**矩阵乘**那条路
  （算力那头忙）。真正缺的恰恰是反过来那条 ——&nbsp;
  **带宽受限的算子动起来长什么样**。这一张只补这一件事。

════════════════════════════════════════════════════════════════════
⛔ 三条边界
════════════════════════════════════════════════════════════════════
① **站台词汇沿用 T-6**，一个字不新造：HBM / VMEM / 向量寄存器 / VPU。
   ⭐ 这一张跟 T-6 的关系是「同一条通路，换一个算子」，不是另起炉灶。
② **不画 MXU、不画累加器。** 向量加根本不经过它们 ——&nbsp;
   画上去只会让人以为矩阵单元也参与了。
③ **时间比例是示意，不是实测。** 图上明写这一句。
   ⛔⛔ 一旦动画看起来像在报时间，读者就会拿它去估耗时。
      而真实比例（搬 6 字节 vs 算 1 次）根本没法在同一屏里按比例演 ——&nbsp;
      **能演的是先后，不是快慢。** 这条必须写在图上，不能只写在这儿。

════════════════════════════════════════════════════════════════════
📌 图上的数（全部来自本课已有内容，没有新引进的量）
════════════════════════════════════════════════════════════════════
· 读 a 2 B ＋ 读 b 2 B ＋ 写 y 2 B ＝ **6 B**（bf16，每个元素 2 字节）
· `y = a + b` 是 **1 次加法 ＝ 1 个 FLOP**
  ⛔ 不是「1 次乘加」——&nbsp;这里一个乘法都没有（2026-09-05 已统一，见 1-6）
· 强度 ＝ 1 ÷ 6 ＝ **0.17**，对着 **312** 那条线，低 **约 1,800 倍**

════════════════════════════════════════════════════════════════════
🔧 技术：为什么用 CSS 动画而不是 SMIL / GIF
════════════════════════════════════════════════════════════════════
· **体积**：整张图约几十 KB，GIF 那条路是 7 MB
· **能暂停**：纯 CSS 复选框，`:checked ~` 把 `animation-play-state` 打成
  paused —— **不用一行 JS**
· **尊重系统设置**：`@media (prefers-reduced-motion: reduce)` 里直接
  `animation:none`，并把元素摆在「一轮结束」的静止姿态上 ——&nbsp;
  ⭐ 关掉动效的人看到的必须是**一张读得懂的静态图**，不是一堆叠在原点的方块。
· ⛔ SVG 内联 `<style>` 在 HTML 里是**文档级**的：类名和 `@keyframes`
  会跟整页共享命名空间。所以这里所有名字都带 `f17-` 前缀。
  忘了前缀不会报错，只会在某天跟别的规则撞上 ——&nbsp;又是一个静默失败。
"""
import io
import re

BL, OR, GR, RD, GY, PU = ("#1a73e8", "#e8710a", "#1e8e3e", "#d93025",
                          "#5f6368", "#9334e6")
INK, GY2, LINE = "#202124", "#80868b", "#dadce0"
# 900 深色档（写字用）——&nbsp;跟 topic-repalette.py 那份同源
I_BL, I_OR, I_GR, I_RD = "#174ea6", "#b06000", "#0d652d", "#a50e0e"

W = 1400
p = []


def wpx(s, size=11):
    """估宽。CJK 与全角按 1 个字宽，ASCII 按 0.56 —— 只求够用来当护栏。
    ⛔ 标签里的 tspan／实体先剥掉，否则会把标记也算进宽度。"""
    s = re.sub(r"<[^>]+>", "", s).replace("&#160;", " ")
    n = sum(1.0 if ord(c) > 0x2000 else 0.56 for c in s)
    return n * size


def t(x, y, s, fill=GY, size=11, bold=False, anchor=None, mono=False, w=None):
    # ⛔ 传了 w 就必须放得下。2026-09-08 第一版没有这道护栏，
    #   账本那行「合计：…强度 0.17」直接被右边缘裁掉了 0.17 ——&nbsp;
    #   **裁掉不报错、也不产生滚动条**，只有渲染出来盯着看才发现。
    if w is not None:
        need = wpx(s, size)
        assert need <= w, ("「%s」要 %.0fpx，只给了 %dpx —— 拆行或加宽"
                           % (re.sub(r"<[^>]+>", "", s)[:24], need, w))
    p.append('<text x="%s" y="%s" fill="%s" style="font:%s %spx %s"%s>%s</text>'
             % (x, y, fill, "700" if bold else "400", size,
                "\'Roboto Mono\',monospace" if mono
                else "\'Noto Sans CJK SC\',sans-serif",
                ' text-anchor="%s"' % anchor if anchor else '', s))


def box(x, y, w, h, fill="#fff", stroke=LINE, r=9, sw=1.0, cls=None, dash=None):
    p.append('<rect x="%s" y="%s" width="%s" height="%s" rx="%s" fill="%s"'
             ' stroke="%s" stroke-width="%s"%s%s/>'
             % (x, y, w, h, r, fill, stroke, sw,
                ' class="%s"' % cls if cls else '',
                ' stroke-dasharray="%s"' % dash if dash else ''))


def bar(x, y, h, col, r=2):
    """左侧 4px 彩条 —— 本课去彩底之后，面板靠它认身份。"""
    box(x, y, 4, h, col, col, r)
    box(x + 2, y, 3, h, "#fff", "#fff", 0)


# ════════════════════════════════════════════════════════════════
# 版面
# ════════════════════════════════════════════════════════════════
TOP = 96
# ⛔ 这几个数是**算出来的，不是试出来的**：先定账本要多宽（最长那行
#   「合计：搬 6 字节，算 1 次 → 强度 0.17」约 300px ＋ 左右留白），
#   剩下的宽度再四等分给站台。⭐ 反过来（先摆站台、剩多少算多少）
#   就是第一版那个「账本被裁掉」的来历。
SW, SH = 212, 132               # 站台卡
GAP = 56                        # 站台之间（留给箭头和飞行的方块）
X0 = 0
XS = [X0 + i * (SW + GAP) for i in range(4)]
LEDGER_X = XS[3] + SW + 36
LEDGER_W = W - LEDGER_X
assert LEDGER_W >= 330, "账本只有 %d px，放不下合计那行" % LEDGER_W

t(0, 16, '一次向量加，从头到尾 ——&#160;'
         '<tspan font-weight="700">全程只有一下是「算」，其余全是「搬」</tspan>',
  INK, 13, False)
t(0, 38, '⭐ 这就是 <tspan font-weight="700">0.17</tspan> 长的样子。'
         '同一条通路上跑矩阵乘是什么样，见「一个数走完全程」那张（图 T-6）——&#160;'
         '<tspan font-weight="700">通路没变，忙的人换了</tspan>。', GY, 11)

# 图例
_lg = [(OR, "a（2 字节）"), (GR, "b（2 字节）"), (BL, "y ＝ a＋b（2 字节）"),
       (RD, "唯一一次「算」")]
_x = 0
for col, lab in _lg:
    box(_x, 52, 10, 10, col, col, 2)
    t(_x + 15, 61, lab, GY, 10.5)
    _x += 16 + len(lab) * 11 + 18

# ── 四个站台 ─────────────────────────────────────────────────────
ST = (
    ("① HBM", "片外主存", I_RD, RD, [
        "a 和 b 原本待在这儿",
        "算完 y 也要写回这儿",
        "⭐ 这一层的兑换比是 312"]),
    ("② VMEM", "片上暂存（不是缓存）", I_OR, OR, [
        "编译器插一条 DMA 搬上来",
        "⛔ 搬多少、什么时候搬，",
        "　 编译期就写死了（见 T-6）"]),
    ("③ 向量寄存器", "VPU 的输入端", I_BL, BL, [
        "一条向量指令吃",
        "8 × 128 个元素",
        "a 一份、b 一份，各占一个"]),
    ("④ VPU", "向量单元", I_GR, GR, [
        "做那一次加法",
        "⭐ 整张图里，红一下的只有这里"]),
)
for i, (name, sub, ink, col, rows) in enumerate(ST):
    x = XS[i]
    box(x, TOP, SW, SH)
    bar(x, TOP, SH, col)
    t(x + 14, TOP + 24, name, ink, 12.5, True)
    t(x + 14, TOP + 42, sub, GY2, 10.5)
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1"/>' % (x + 13, TOP + 52, x + SW - 13, TOP + 52, LINE))
    for k, r in enumerate(rows):
        t(x + 14, TOP + 72 + k * 19, r, GY, 10.5, w=SW - 26)

# ── 传送带：飞行道**单独一层**，在面板下沿之外 ──────────────────
# ⛔⛔ 第一版把飞行道放在面板中线上（LANE_Y = 面板中间），于是方块飞到
#   站台时**正压在这个面板的正文上**。渲染出来才看见 ——&nbsp;
#   SVG 里两个元素重叠既不报错、也不影响布局，只是糊在一起。
# ⭐ 判据：**会动的东西要有自己的一层。** 让它跟静止的文字共用同一片
#   纵向空间，就等于每一帧都在赌它们不撞上。这条比「调一下 y」重要得多。
p.append('<defs><marker id="f17a" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="6" markerHeight="6" orient="auto">'
         '<path d="M0 0 L10 5 L0 10 z" fill="%s"/></marker>'
         '<marker id="f17b" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="6" markerHeight="6" orient="auto">'
         '<path d="M0 0 L10 5 L0 10 z" fill="%s"/></marker></defs>' % (BL, GY2))

TY = TOP + SH + 14               # 传送带顶
TH = 52                          # 传送带高
box(0, TY, XS[3] + SW, TH, "#fff", LINE, 8)
GO_Y = TY + 17                   # 去程中线
BK_Y = TY + 37                   # 回程中线
for i in range(4):               # 每个站台在带子上的停靠位
    cx = XS[i] + SW // 2
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1" stroke-dasharray="2 3"/>'
             % (cx, TY + 4, cx, TY + TH - 4, LINE))
    t(cx, TY - 4, "①②③④"[i], GY2, 10.5, anchor="middle")
for i in range(3):
    a = XS[i] + SW // 2 + 22
    b = XS[i + 1] + SW // 2 - 22
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.6" marker-end="url(#f17a)"/>' % (a, GO_Y, b, GO_Y, BL))
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
             'stroke-width="1.4" marker-end="url(#f17b)"/>' % (b, BK_Y, a, BK_Y, GY2))
_lab = ("DMA", "载入", "喂入")
for i in range(3):
    t((XS[i] + XS[i + 1]) // 2 + SW // 2, GO_Y - 7, _lab[i], GY2, 10.5,
      anchor="middle")

# ── 会动的三个方块 ───────────────────────────────────────────────
# ⭐ 每个方块只做一件事：沿传送带从一个停靠位滑到下一个。
#   ⛔ 不用 animateMotion（SMIL）：CSS transform 到处都稳，
#     而且能被 prefers-reduced-motion 一键关掉。
_S = 13
p.append('<rect class="f17-fly f17-a" x="%d" y="%d" width="%d" height="%d" '
         'rx="3" fill="%s" stroke="%s" stroke-width="1"/>'
         % (-_S - 3, GO_Y - _S // 2, _S, _S, OR, OR))
p.append('<rect class="f17-fly f17-b" x="%d" y="%d" width="%d" height="%d" '
         'rx="3" fill="%s" stroke="%s" stroke-width="1"/>'
         % (3, GO_Y - _S // 2, _S, _S, GR, GR))
p.append('<rect class="f17-fly f17-y" x="%d" y="%d" width="%d" height="%d" '
         'rx="3" fill="%s" stroke="%s" stroke-width="1"/>'
         % (-_S // 2, BK_Y - _S // 2, _S, _S, BL, BL))
# VPU 那一下
p.append('<rect class="f17-spark" x="%d" y="%d" width="%d" height="%d" rx="7" '
         'fill="none" stroke="%s" stroke-width="2.5"/>'
         % (XS[3] + 6, TOP + 6, SW - 12, SH - 12, RD))
t(XS[3] + SW // 2, TOP + SH - 16, "＋", I_RD, 20, True, anchor="middle")

t(0, TY + TH + 18, "灰色那条是回程：结果 → VMEM → HBM，同一套 DMA、同样是编译期排好的"
  "——&#160;<tspan font-weight=\"700\">回程也要占带宽，写 y 那 2 个字节一样算在 6 里面</tspan>",
  GY2, 10.5, w=W)

# ── 暂停开关 ───────────────────────────────────────────────────
# ⭐⭐ 开关**放在 SVG 内部**（foreignObject 里塞一个 HTML 复选框）。
#   ⛔ 原本想放在页面上、用兄弟选择器 `#f17-pause:checked ~ figure` 去打 ——&nbsp;
#     行不通：L200 的 fig() 只搬 figure 元素本身，**开关会被落在 L300**，
#     于是同一张图在两份文档里一个能暂停、一个不能，而且不报错。
#   ⭐ 判据：**能跟着图走的东西，就别留在页面上。** 图是产物、页面是宿主，
#     产物自带全部零件才不会在搬运中掉件。
# ⚠️ 因此选择器改用 `:has()`（现代浏览器都支持）。⭐ 万一某个浏览器不支持，
#   退化成「暂停键点了没反应」——&nbsp;动画照跑，图照读，不是坏掉。
p.append('<foreignObject x="%d" y="0" width="150" height="26">'
         '<div xmlns="http://www.w3.org/1999/xhtml" '
         'style="font:11px \'Noto Sans CJK SC\',sans-serif;color:#5f6368;'
         'text-align:right">'
         '<label style="cursor:pointer;user-select:none">'
         '<input type="checkbox" id="f17-pause" style="vertical-align:-1px"/>'
         ' 暂停动画</label></div></foreignObject>' % (W - 150))

# ── 账本 ─────────────────────────────────────────────────────────
LY = TOP
# ⭐ 账本**比站台卡高**：它一路盖到传送带下沿。
#   ⛔ 原来跟站台卡一样高（SH），于是「合计」那行的基线离最后一行只有 4px，
#     渲染出来两行叠在一起 ——&nbsp;几何探针查出来的，纸面上看是「有点挤」。
#   ⭐ 这也更讲得通：账是**整条通路**的账，不是某一站的账。
LH = SH + 14 + TH
box(LEDGER_X, LY, LEDGER_W, LH)
bar(LEDGER_X, LY, LH, PU)
t(LEDGER_X + 14, LY + 24, "这一轮的账", "#681da8", 12.5, True)
p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
         % (LEDGER_X + 13, LY + 34, LEDGER_X + LEDGER_W - 13, LY + 34, LINE))
_rows = (("f17-l1", "读 a", "2 字节", I_OR), ("f17-l2", "读 b", "2 字节", I_GR),
         ("f17-l3", "算　", "1 个 FLOP", I_RD), ("f17-l4", "写 y", "2 字节", I_BL))
for k, (cls, lab, val, ink) in enumerate(_rows):
    yy = LY + 54 + k * 20
    p.append('<g class="f17-led %s">' % cls)
    t(LEDGER_X + 14, yy, lab, GY, 11)
    t(LEDGER_X + 74, yy, val, ink, 11, True, mono=True)
    p.append('</g>')
t(LEDGER_X + 14, LY + 160, '合计：搬 <tspan font-weight="700">6</tspan> 字节，'
  '算 <tspan font-weight="700">1</tspan> 次 → 强度 '
  '<tspan font-weight="700" fill="%s">0.17</tspan>' % I_RD, GY, 11,
  w=LEDGER_W - 28)

# ── 落点 ─────────────────────────────────────────────────────────
FY = TY + TH + 34
FH = 78
box(0, FY, W, FH)
bar(0, FY, FH, GR)
t(14, FY + 24, "看这一轮里「算」占了多少", I_GR, 12.5, True)
t(14, FY + 46, '四个站台走一遍，<tspan font-weight="700">红框只闪一次</tspan>。'
  '搬 6 个字节换来 1 次加法 ——&#160;'
  '<tspan font-weight="700">这台机器每搬一个字节，本来能算 312 次</tspan>，'
  '现在只算了 <tspan font-weight="700">0.17</tspan> 次。', GY, 11)
t(14, FY + 65, '⭐ <tspan font-weight="700">所以「带宽受限」不是慢，是闲</tspan>'
  '——&#160;算力一直在等数据到位。'
  '把这样的算子一个个接起来还各写一趟 HBM，就是第 2 节要治的病。', GY, 11)

# ── 出处 / 免责 ──────────────────────────────────────────────────
SY = FY + FH + 16
t(0, SY, '⚠️ <tspan font-weight="700">示意图：时间比例不代表真实耗时，只表示先后。</tspan>'
  '真实比例（搬 6 字节 vs 算 1 次）没法在同一屏里按比例演 ——&#160;'
  '<tspan font-weight="700">能演的是次序，不是快慢。</tspan>', GY2, 10.5)
t(0, SY + 18, '图上每个数都能当场复核：bf16 每元素 2 字节 → 读 a 2 ＋ 读 b 2 ＋ 写 y 2 ＝ 6；'
  'y ＝ a＋b 是 1 次加法 ＝ 1 个 FLOP（⛔ 不是乘加）；1 ÷ 6 ＝ 0.167 ≈ 0.17；'
  '312 ÷ 0.167 ＝ 1,872 倍。', GY2, 10.5)
t(0, SY + 36, '⭐ 这张图的形式借自 How To Scale Your Model 的 pointwise-product 动图'
  '（jax-ml.github.io/scaling-book，MIT）——&#160;'
  '站台词汇与配色沿用本课 T-6，内容换成本节那个 0.17 的例子。', GY2, 10.5)

H = SY + 48

# ════════════════════════════════════════════════════════════════
# CSS 动画
# ════════════════════════════════════════════════════════════════
# 一轮 9 秒。⭐ 三个方块和四行账用**同一条时间轴**（同样的 9s、同样的
#   起点），靠各自 keyframes 里的百分比错开 ——&nbsp;
#   ⛔ 不要用 animation-delay 去排，那样暂停再播会错位。
DUR = 9
# ⭐ 四个停靠位 ＝ 四个站台在传送带上的**正中**。
#   ⛔ 这四个数必须跟上面画虚线停靠位的 `cx` **同一个式子算出来** ——&nbsp;
#     写成两份迟早对不齐，而对不齐的表现是「方块停在站台旁边一点」，
#     看着像手抖，不像 bug。
_x0, _x1, _x2, _x3 = [XS[i] + SW // 2 for i in range(4)]
CSS = """
/* ⛔ 所有名字带 f17- 前缀：内联 SVG 的样式表在 HTML 里是**文档级**的，
   类名和 @keyframes 跟整页共享命名空间。撞名不报错，只会某天突然生效。
   ⚠️ 这段注释里**不能写字面的尖括号标签名** —— 写了 XML 自检当场炸
   （2026-09-08 现场撞的：注释里提了两个标签名，解析器把它们当成真标签）。 */
.f17-fly{opacity:0;animation:f17-none %(d)ss linear infinite}
.f17-a{animation-name:f17-ka}
.f17-b{animation-name:f17-kb}
.f17-y{animation-name:f17-ky}
.f17-spark{opacity:0;animation:f17-kspark %(d)ss linear infinite}
.f17-led{opacity:.28;animation:f17-klit %(d)ss linear infinite}
.f17-l1{animation-name:f17-kl1}
.f17-l2{animation-name:f17-kl2}
.f17-l3{animation-name:f17-kl3}
.f17-l4{animation-name:f17-kl4}

/* a：0–22%% 走到 VMEM，22–36%% 走到寄存器，36–44%% 进 VPU，之后隐身 */
@keyframes f17-ka{
  0%%  {opacity:0;transform:translateX(%(x0)dpx)}
  4%%  {opacity:1;transform:translateX(%(x0)dpx)}
  22%% {opacity:1;transform:translateX(%(x1)dpx)}
  30%% {opacity:1;transform:translateX(%(x1)dpx)}
  40%% {opacity:1;transform:translateX(%(x2)dpx)}
  46%% {opacity:1;transform:translateX(%(x2)dpx)}
  54%% {opacity:1;transform:translateX(%(x3)dpx)}
  58%% {opacity:0;transform:translateX(%(x3)dpx)}
  100%%{opacity:0;transform:translateX(%(x3)dpx)}
}
/* b：比 a 晚一点点出发，两块并排飞 —— 读 a 和读 b 花的是同一份带宽 */
@keyframes f17-kb{
  0%%  {opacity:0;transform:translateX(%(x0)dpx)}
  8%%  {opacity:1;transform:translateX(%(x0)dpx)}
  26%% {opacity:1;transform:translateX(%(x1)dpx)}
  32%% {opacity:1;transform:translateX(%(x1)dpx)}
  42%% {opacity:1;transform:translateX(%(x2)dpx)}
  48%% {opacity:1;transform:translateX(%(x2)dpx)}
  54%% {opacity:1;transform:translateX(%(x3)dpx)}
  58%% {opacity:0;transform:translateX(%(x3)dpx)}
  100%%{opacity:0;transform:translateX(%(x3)dpx)}
}
/* y：算完之后原路回去 */
@keyframes f17-ky{
  0%%,62%%{opacity:0;transform:translateX(%(x3)dpx)}
  64%% {opacity:1;transform:translateX(%(x3)dpx)}
  74%% {opacity:1;transform:translateX(%(x2)dpx)}
  84%% {opacity:1;transform:translateX(%(x1)dpx)}
  96%% {opacity:1;transform:translateX(%(x0)dpx)}
  100%%{opacity:0;transform:translateX(%(x0)dpx)}
}
/* 唯一那一下 */
@keyframes f17-kspark{
  0%%,56%%{opacity:0} 58%%{opacity:1} 61%%{opacity:1} 63%%{opacity:0} 100%%{opacity:0}
}
@keyframes f17-kl1{0%%,20%%{opacity:.28} 24%%,100%%{opacity:1}}
@keyframes f17-kl2{0%%,26%%{opacity:.28} 30%%,100%%{opacity:1}}
@keyframes f17-kl3{0%%,56%%{opacity:.28} 60%%,100%%{opacity:1}}
@keyframes f17-kl4{0%%,94%%{opacity:.28} 98%%,100%%{opacity:1}}

/* 暂停：纯 CSS 复选框，不用一行 JS。开关就在这张图里（见上面 foreignObject），
   用 :has() 从 svg 根往下打 —— 图自带全部零件，搬到哪份文档都能用。 */
svg:has(#f17-pause:checked) .f17-fly,
svg:has(#f17-pause:checked) .f17-spark,
svg:has(#f17-pause:checked) .f17-led{animation-play-state:paused}

/* ⭐⭐ 关掉动效的人看到的必须是**一张读得懂的静态图**，
   不是三个方块叠在原点。所以这里不只是 animation:none ——
   还要把每个元素摆到「一轮跑完」的姿态上：三块各停在自己的站台，
   红框亮着，四行账全部点亮。 */
@media (prefers-reduced-motion: reduce){
  .f17-fly,.f17-spark,.f17-led{animation:none}
  .f17-a{opacity:1;transform:translateX(%(x1)dpx)}
  .f17-b{opacity:1;transform:translateX(%(x2)dpx)}
  .f17-y{opacity:1;transform:translateX(%(x0)dpx)}
  .f17-spark{opacity:1}
  .f17-led{opacity:1}
}
""" % {"d": DUR, "x0": _x0, "x1": _x1, "x2": _x2, "x3": _x3}

svg = ('<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="%s">'
       '<style>%s</style>%s</svg>'
       % (W, H,
          "一次向量加在 TPU 上从 HBM 到向量单元再写回的全过程动画："
          "读 a 两字节、读 b 两字节、向量单元做一次加法、写回 y 两字节，"
          "共搬六字节只算一次，算术强度 0.17",
          CSS, "\n".join(p)))

# ── 写盘前自检 ───────────────────────────────────────────────────
# ① XML 得能解析（属性里混进裸双引号是这门课栽过的坑）
import xml.etree.ElementTree as ET                          # noqa: E402
ET.fromstring(re.sub(r"&#160;", " ", svg))
# ② 所有类名和 keyframes 都必须带前缀 —— 内联样式表是文档级的。
#    ⛔⛔ 扫之前**先把 CSS 注释剥掉**。2026-09-08 第一版没剥，注释里那句
#      「类名和 @keyframes 跟整页共享命名空间」被正则当成了一条真规则，
#      于是断言报「@keyframes 跟整页共享命名空间 少了 f17- 前缀」。
#    ⭐ 今天第二次踩同一个形状（上一次是 head 残留断言把注释当残留）：
#      **护栏要查的是代码，不是碰巧长得像代码的散文。**
_css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
for nm in set(re.findall(r"@keyframes\s+([\w-]+)", _css)) | \
        set(re.findall(r"class=\"([^\"]+)\"", svg)):
    for one in nm.split():
        assert one.startswith("f17-"), "类名／动画名少了 f17- 前缀：%s" % one
# ③ 字号地板：本课宽图渲染时约 ×1.22，10.5px 是 12.8px，已是全书下限
for sz in re.findall(r"font:\d+ ([\d.]+)px", svg):
    assert float(sz) >= 10.5, "字号 %s 低于本课地板 10.5" % sz

io.open("fig1-7.svg", "w", encoding="utf-8").write(svg)
print("fig1-7 ok  %d×%d  %d 字符" % (W, H, len(svg)))
