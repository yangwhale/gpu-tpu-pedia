# -*- coding: utf-8 -*-
r"""专题四 · §1 插一格「derivative / partial derivative —— 名字本身就是解释」

⭐⭐⭐ 2026-09-22 现场（语音课）连提三问：
  1. 「这个导数在英文里不是叫 deviation 吗？翻译成导数不就失去语言上的意义了？」
     ——⛔ 词记错了：是 **derivative**，deviation 是「偏差」（standard deviation）。
       ⭐ 但他要的东西是对的：**这个词确实有语言上的意思，而且比「导数」还生动。**
  2. 「先把导数讲明白，再讲偏导数。」——&#160;所以 Ⓐ 在 Ⓑ 前面，这个顺序是他定的。
  3. ⭐⭐⭐ 最狠的一问：「一个数拧旋钮，对 loss 的影响要穿过的是**矩阵**不是整数；
     而且还取决于**同层横向那些参数**，彼此之间还有影响。太复杂，一点都不直观。
     你怎么才能让它直观？这就是 partial derivative 的神奇之处，画图讲明白。」

⭐ 三格各答一问，都用真几何：

  Ⓐ **「导」就是引流** ——&#160;拉丁语 dērīvāre ＝ de（从…）＋ rivus（河流），
     字面是「从一条河里引一股水出来」。
     ⭐⭐ 图上把这件事**画成可验证的关系**：左边曲线在某点的**切线斜率**，
       正好等于右边那条曲线在**同一个横坐标上的高度**。
       ——&#160;两条曲线之间那根虚线，就是「引出来」这个动作。
     ⛔ 不是并排摆两条曲线了事：不画那根「斜率 → 高度」的连线，
       这一格就退回成写字板。

  Ⓑ **partial ＝ 夹住其余的** ——&#160;一排旋钮，只有一个松着，其余全被夹子按住。
     ⭐ 下面那根一维数轴是这一格的落点：**N 维的问题当场塌成一维**，
       于是「偏导数」又变回了普通导数。

  Ⓒ **那把扇子，折进了一个数**（主图，答第 3 问）。
     左边把「一个权重影响 loss 的所有路径」真画出来 ——&#160;
     ⭐ 先让人看见**复杂是真的**，再给解法，否则解法没有分量。
     右边是同一件事：**来料 × 责任**，两个数相乘。
     中间那根粗箭头就是全部答案：**扇子没有被忽略，它已经折进右边那个数里了。**

⭐ 所有数脚本当场算，五条 assert 钉着。
📌 词源出处：Etymonline（derive / derivative）——&#160;
   Latin *derivare* "to lead or draw off (a stream of water) from its source",
   from *de rivo*（de "from" ＋ rivus "stream"）；数学义自 1670 年代。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400

# ══ Ⓐ 的数：f(u) = u²，f'(u) = 2u。在 u₀ 处「斜率 ＝ 右图高度」 ═══════
U0 = 0.60
SLOPE0 = 2 * U0                      # f'(0.6) = 1.2
assert abs(SLOPE0 - 1.2) < 1e-12, "Ⓐ 的斜率标注写死了 1.2，公式得对得上"

# ══ Ⓒ 的数：扇子画 3 跳 × 每层 6 个节点；真模型是 61 层 × 7168 宽 ══════
FAN_HOPS, FAN_W = 3, 6
FAN_PATHS = FAN_W ** FAN_HOPS                    # 图上真的画得出来的路径数
D_HID, N_AFTER = 7168, 60                        # V3：隐藏宽度 / 第一层之后还有几层
REAL_DIGITS = int(N_AFTER * math.log10(D_HID)) + 1
assert FAN_PATHS == 216, "扇子路径数跟画出来的线必须对得上：%d" % FAN_PATHS
assert 230 <= REAL_DIGITS <= 240, "真模型路径数的位数变了：%d" % REAL_DIGITS
# ⭐ 而反向传播不数路径 —— 它只做「每层两次矩阵乘」，总数跟层数成正比。
BP_MATMULS = 2 * (N_AFTER + 1)
assert BP_MATMULS == 122


def knob(f, cx, cy, r, col, clamped, ang):
    """一个旋钮：圆角方块 ＋ 一根指针。clamped 的额外压一道夹子。"""
    f.box(cx - r, cy - r, 2 * r, 2 * r, "#fff", col, r, sw=1.8 if not clamped else 1.2)
    f.line(cx, cy, cx + r * 0.72 * math.cos(ang), cy + r * 0.72 * math.sin(ang),
           col, 1.8 if not clamped else 1.2, arrow=False)
    if clamped:
        # 夹子：横跨旋钮的一道粗灰条 ＋ 两只脚
        f.box(cx - r - 4, cy - 4.5, 2 * r + 8, 9, "#fff", GY2, 3, sw=1.3)
        f.line(cx - r - 4, cy + 4.5, cx - r - 4, cy + r + 5, GY2, 1.3, arrow=False)
        f.line(cx + r + 4, cy + 4.5, cx + r + 4, cy + r + 5, GY2, 1.3, arrow=False)


def main():
    f = Fig(W, "三块。第一块讲导数这个词本身：derivative 来自拉丁语，"
               "字面意思是从一条河里引一股水出来。左边画一条原来的曲线，"
               "右边画从它引出来的那条。左边曲线上取一点画切线，"
               "那根切线的斜率是一点二；右边那条曲线在同一个横坐标上的高度，"
               "也正好是一点二。两点之间用一根虚线连起来 —— "
               "这根虚线就是引出来这个动作，也是整个导数的定义。"
               "第二块讲偏导数的偏字：一排九个旋钮，八个被夹子夹住不许动，"
               "只剩中间一个松着。下面配一根一维的数轴，"
               "意思是一个九维的问题当场塌成一维，偏导数又变回了普通导数。"
               "第三块是主图，回答一个权重的影响为什么看着那么复杂。"
               "左边把这个权重影响最终误差的所有路径都画出来："
               "同一层里别的权重先加进同一个出口，然后往下一层散开，"
               "一层比一层宽，三跳就已经两百一十六条路；"
               "真模型是六十层、每层七千多宽，路径条数写出来是两百多位的数字。"
               "右边是同一件事的另一种画法，只有两个格子相乘："
               "前向存下的来料，乘上反向传回来的责任。"
               "中间一根粗箭头连着两边，上面写着：那把扇子没有被忽略，"
               "它已经整个折进右边那个责任数里了。因为反向那一趟，"
               "是按层一次算完的，不是按参数两两去算的")

    # ══ Ⓐ 「导」就是引流 ═══════════════════════════════════════════
    # ⛔ 版面预算（踩过一次才写下来的）：这一格里**右边那根纵轴在 x=830**，
    #   而中间那句「同一个数」如果按两点中点居中，会正好压在它上面。
    #   ⭐ 所以中间的字一律**摆进两张图之间那段 530→830 的空白**，不按中点居中。
    PH, py = 282, 60
    f.panel(60, py, W - 120, PH,
            "Ⓐ 「导」就是引流 ——　dērīvāre ＝ de（从…）＋ rivus（河流）", BL,
            sub="左边那条切线的斜率，就是右边那条曲线在同一横坐标上的高度")

    LH, PW = 128, 380
    base = py + 222                       # 两个坐标系共用的底线
    LX0, RX0 = 150, 830
    MIDX = (LX0 + PW + RX0) / 2.0         # 两张图之间那段空白的中点

    for ox, lab, sub2, col, fn, norm in (
            (LX0, "原来这条河", "输入 → 输出", GY, lambda u: u * u, 1.0),
            (RX0, "引出来这条", "动一格 → 动几格", BL, lambda u: 2 * u, 2.0)):
        f.line(ox - 16, base, ox + PW + 30, base, INK, 2.0)          # 横轴
        f.line(ox, base + 14, ox, base - LH - 26, INK, 2.0)          # 纵轴
        f.t(ox - 16, base + 30, "输入", GY2, size=11)
        pts = [(ox + (k / 60.0) * PW, base - fn(k / 60.0) / norm * LH)
               for k in range(61)]
        f.path(pts, col, 2.6, arrow=False)
        # ⛔ 标签放曲线**末端的正上方并右对齐** —— 原来挂在曲线右侧，
        #   右图那条的副标题直接捅出了面板右沿（面板只到 x=1340）。
        f.t(ox + PW, base - LH - 32, lab, col, True, 13, "end")
        f.t(ox + PW, base - LH - 15, sub2, GY2, size=11, anchor="end")

    # 左：u₀ 处的点 ＋ 切线（斜率 = SLOPE0）
    lx, ly = LX0 + U0 * PW, base - (U0 * U0) * LH
    f.line(lx, base, lx, ly, GY2, 1.2, dash="3 3", arrow=False)
    dx = 84.0
    dy = SLOPE0 * (dx / PW) * LH          # 同一套像素比例下的切线落差
    f.line(lx - dx, ly + dy, lx + dx, ly - dy, OR, 2.8, arrow=False)
    f.box(lx - 24, ly - dy - 40, 132, 24, "#fff", OR, 6, sw=1.6)
    f.t(lx + 42, ly - dy - 23, "切线斜率 %.1f" % SLOPE0, OR, True, 12.5, "middle")

    # 右：同一个 u₀ 处的高度，正好 = SLOPE0
    rx, ry = RX0 + U0 * PW, base - (SLOPE0 / 2.0) * LH
    f.line(rx, base, rx, ry, GY2, 1.2, dash="3 3", arrow=False)
    f.box(rx - 56, ry - 40, 112, 24, "#fff", OR, 6, sw=1.6)
    f.t(rx, ry - 23, "高度 %.1f" % SLOPE0, OR, True, 12.5, "middle")

    # ⭐ 两点之间那根「引流」线 —— 这一格全部的信息量在这根线上
    f.line(lx + 110, ly - dy - 28, rx - 62, ry - 28, OR, 2.4, dash="6 4")
    f.t(MIDX, base - 74,
        "⭐ 同一个数", INK, True, 13.5, "middle")
    f.t(MIDX, base - 56, "左边是斜率，右边是高度", GY, size=12, anchor="middle")
    # ⛔ 这儿原来还有一句「没有换一条河，是从这条河里引出了第二条」——　删了。
    #   ① 它居中之后会压到右图那个「输入」轴标上（实测撞车）；
    #   ② ⭐ 更要紧的是**面板标题和落点带已经各说了一遍** ——　三个地方说同一句，
    #      正是「同一个问题好几个地方都讲」。删掉零损失。
    f._pan = None

    # ══ Ⓑ partial ＝ 夹住其余的 ════════════════════════════════════
    py2, PH2 = py + PH + 24, 214
    f.panel(60, py2, W - 120, PH2,
            "Ⓑ 「偏」不是偏差 ——　是只松开一个，其余全部夹住", PU,
            sub="partial ＝ 求导这件事只做了一部分：一次一个变量")

    N_KNOB, FREE = 9, 4
    kr, kx0, kgap = 26, 430, 68
    ky = py2 + 82
    for i in range(N_KNOB):
        cx = kx0 + i * kgap
        free = (i == FREE)
        knob(f, cx, ky, kr, PU if free else GY2, not free,
             -2.2 if free else -1.571)
        f.t(cx, ky + kr + 22, "w%d" % (i + 1), PU if free else GY2,
            free, 11.5, "middle")
    f.t(kx0 - 42, ky + 5, "9 个旋钮", INK, True, 13, "end")
    f.t(kx0 - 42, ky + 23, "其中 8 个被夹住", GY2, size=11, anchor="end")

    # 落点：塌成一维
    ax = kx0 + FREE * kgap
    f.line(ax, ky + kr + 34, ax, py2 + 156, PU, 2.0)
    f.line(ax - 150, py2 + 170, ax + 150, py2 + 170, INK, 2.0)
    for k in range(-3, 4):
        f.line(ax + k * 42, py2 + 165, ax + k * 42, py2 + 170, GY2, 1.2, arrow=False)
    f.box(ax - 7, py2 + 163, 14, 14, "#fff", PU, 4, sw=1.8)
    f.t(ax + 170, py2 + 174,
        "⭐⭐ 9 维的问题当场塌成<tspan font-weight=\"700\">一维</tspan>"
        "　——　偏导数又变回了<tspan font-weight=\"700\">普通导数</tspan>",
        INK, size=13)
    f._pan = None

    # ══ Ⓒ 那把扇子，折进了一个数（主图） ═══════════════════════════
    # ⛔⛔ 版面预算写在这儿，因为第一版就是在这块翻的车：
    #   扇子最外那列按 30+34×3 ＝ 132 半张开，而 cy0 只压在面板中线上 ——
    #   最上那个节点直接**顶穿了面板标题栏**。
    # ⭐⭐ 而且**图审计没抓到**：面板越界自检 `_note_ink` 只在 t() 里调，
    #   box() / line() / path() 画出界是**不报的**。
    #   ——　所以这一格的几何必须自己把预算算清楚：
    #     py3+52 / +70  两行说明
    #     py3+104 … +296 扇子（cy0 ＝ py3+200，最大半张开 96）
    #     py3+320       层号行
    #     py3+352       落点
    py3, PH3 = py2 + PH2 + 24, 380
    f.panel(60, py3, W - 120, PH3,
            "Ⓒ 那把扇子，折进了一个数 ——　这才是 partial derivative 的神奇之处", GR,
            sub="左边是「复杂是真的」，右边是「而它只要两个数」")

    cy0 = py3 + 200

    # ── 左半：扇子 ──────────────────────────────────────────────
    cx0 = 152
    f.box(cx0 - 32, cy0 - 17, 64, 34, "#fff", RD, 6, sw=2.2)
    f.t(cx0, cy0 + 5, "w", RD, True, 16, "middle")
    # ⛔ 这一句原来放 cy0+38 ——　正好压在下面那摞「同层其它权重」的第一个方块上。
    #   ⭐ 渲染出来才看见：光看代码只觉得「差 4 像素，应该还好」。
    f.t(cx0, cy0 + 96, "要算它的梯度", RD, True, 11.5, "middle")

    outx = cx0 + 126
    f.box(outx - 15, cy0 - 15, 30, 30, "#fff", INK, 15, sw=1.8)
    f.line(cx0 + 34, cy0, outx - 17, cy0, RD, 2.4)

    # 同层横向：别的权重也喂进同一个出口 —— ⭐ 给它们画出**实体来源**，
    # ⛔ 第一版只画了四根线、没有源头方块，看着像四根断线。
    for off in (-64, -34, 34, 64):
        f.box(cx0 - 22, cy0 + off - 10, 44, 20, "#fff", GY2, 4, sw=1.1)
        f.line(cx0 + 24, cy0 + off, outx - 16, cy0 - (3 if off < 0 else -3),
               GY2, 1.1)
    f.t(cx0, cy0 - 92, "同层横向的那些", GY, True, 11.5, "middle")
    f.t(cx0, cy0 - 77, "在这儿是「加」进来", GY2, size=11, anchor="middle")

    # 三跳扇形：6 → 6 → 6
    xs = [outx, outx + 128, outx + 256, outx + 384]

    def col_y(xi, n):
        span = 26 + 24 * xi           # 最外一列半张开 26+72 = 98 ≤ 预算 96+2
        return [cy0 - span + (2 * span) * (j / float(n - 1)) for j in range(n)]

    layers = [[cy0]] + [col_y(i + 1, FAN_W) for i in range(FAN_HOPS)]
    for hop in range(FAN_HOPS):
        for ya in layers[hop]:
            for yb in layers[hop + 1]:
                f.line(xs[hop] + 16, ya, xs[hop + 1] - 8, yb, LINE, 0.8, arrow=False)
    for hop in range(1, FAN_HOPS + 1):
        for yb in layers[hop]:
            f.box(xs[hop] - 8, yb - 8, 16, 16, "#fff", GY2, 8, sw=1.1)
        f.t(xs[hop], py3 + 320, "第 %d 层" % (hop + 1), GY2, size=11, anchor="middle")
    assert cy0 - (26 + 24 * FAN_HOPS) > py3 + 100, "扇子顶出了预算，会撞标题栏"

    f.t(outx + 192, py3 + 56, "⭐ 才走 %d 跳，已经 %d 条路" % (FAN_HOPS, FAN_PATHS),
        INK, True, 13.5, "middle")
    f.t(outx + 192, py3 + 76,
        "真模型 %d 层 × %s 宽 ——　路径数是个 %d 位数"
        % (N_AFTER, format(D_HID, ","), REAL_DIGITS), GY, size=12, anchor="middle")

    # ── 中间那根粗箭头 ─────────────────────────────────────────
    ax2 = xs[-1] + 40
    f.line(ax2, cy0, ax2 + 92, cy0, GR, 3.6)
    f.t(ax2 + 46, cy0 - 16, "全部折进去", GR, True, 12.5, "middle")

    # ── 右半：两个数相乘 ───────────────────────────────────────
    bx = ax2 + 112
    f.box(bx, cy0 - 78, 152, 54, "#fff", BL, 8, sw=2.0)
    f.t(bx + 76, cy0 - 56, "来料 x", BL, True, 14, "middle")
    f.t(bx + 76, cy0 - 38, "前向那一刻存下的", GY2, size=11, anchor="middle")
    f.t(bx + 76, cy0 - 6, "×", INK, True, 18, "middle")
    f.box(bx, cy0 + 12, 152, 54, "#fff", GR, 8, sw=2.0)
    f.t(bx + 76, cy0 + 34, "责任 δ", GR, True, 14, "middle")
    f.t(bx + 76, cy0 + 52, "反向传回来的", GY2, size=11, anchor="middle")
    f.line(bx + 76, cy0 + 70, bx + 76, cy0 + 82, INK, 1.6)
    f.t(bx + 76, cy0 + 100, "＝ w 的梯度", INK, True, 13.5, "middle")

    f.t(700, py3 + 348, "⭐⭐⭐ 那把扇子没有被忽略 ——　它已经在 δ 里面了",
        INK, True, 14.5, "middle")
    f.t(700, py3 + 368, "反向那一趟是按层一次算完的，不是按参数两两去算的",
        GY, size=12.5, anchor="middle")
    f._pan = None

    yb2 = f.band(py3 + PH3 + 20, "ok",
                 "复杂没有消失 ——　它被「只付一次」了",
                 ("⭐ 一个权重影响 loss 的路径多到写不下，可<tspan font-weight=\"700\">"
                  "算它的梯度只要两个数</tspan>：前向存下的那个输入、反向传来的那个责任。"
                  "——　中间所有的横向混合与下游分叉，"
                  "<tspan font-weight=\"700\">都已经在算 δ 的时候按层一次做掉了</tspan>。",
                  "⭐⭐ 代价也很清楚：<tspan font-weight=\"700\">参数两两之间</tspan>"
                  "的相互作用（二阶导）<tspan font-weight=\"700\">故意没算</tspan> ——　"
                  "每两个参数就有一个数，存不下。"
                  "<tspan font-weight=\"700\">而「故意不算」的代价，就是必须小步走</tspan>"
                  "——　这就是学习率存在的理由。"))

    yb2 = f.src(yb2 + 16,
                "📌 <tspan font-weight=\"700\">词源</tspan>：Etymonline（derive / derivative）"
                "——　Latin <tspan font-style=\"italic\">derivare</tspan> "
                "“to lead or draw off (a stream of water) from its source”，"
                "来自 <tspan font-style=\"italic\">de rivo</tspan>"
                "（de “from” ＋ rivus “stream”）；数学义自 1670 年代。"
                "⛔ 顺带纠一个常见的记混：<tspan font-weight=\"700\">deviation 是「偏差」</tspan>"
                "（standard deviation），跟 derivative 不是一个词。",
                "⚠️ Ⓒ 左边那把扇子<tspan font-weight=\"700\">是示意</tspan>："
                "画的是 %d 跳 × 每层 %d 个节点（%d 条路）。"
                "右上角那个「%d 位数」才是按真模型算的 ——　%d 层、每层 %s 宽。"
                % (FAN_HOPS, FAN_W, FAN_PATHS, REAL_DIGITS, N_AFTER, format(D_HID, ",")),
                "⭐ 对照一下就知道反向传播省在哪：路径条数是 <tspan font-weight=\"700\">"
                "指数级</tspan>，而反向真正做的矩阵乘只有 <tspan font-weight=\"700\">"
                "%d 次</tspan>（每层两次）——　<tspan font-weight=\"700\">跟层数成正比</tspan>。"
                % BP_MATMULS)

    f.save("fig4-derive.svg", yb2 + 14)


if __name__ == "__main__":
    main()
