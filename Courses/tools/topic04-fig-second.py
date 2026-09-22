# -*- coding: utf-8 -*-
r"""专题四 · §1.4b「二阶导到底是什么」——&#160;以及它跟 Adam 那个「二阶矩」的关系

⭐⭐⭐ 2026-09-22 现场：「你已经说了好几次二阶导了，那是个什么东西？
  能干嘛用？计算复杂度有多少？跟 Adam 优化器里边说的那个一阶动量和
  二阶动量有什么关系？」

⛔ 这一问照出一个真的洞：**「二阶导」在这一讲里承重了好几处**
  （1.4 的「故意不算它」、3.3b 的最优步长、Adam 那条线的动机），
  **却从来没有一个地方正面定义过它**。

⭐⭐ 而第四问是**最容易被中文坑到的一处**：
  **「二阶矩」和「二阶导」是两个完全不同的东西，只是中文都带「二阶」。**
    · 二阶**导**：对**参数**求两次导 →&#160;曲率 →&#160;**几何**，N² 个数
    · 二阶**矩**：对**梯度**取平方求平均 →&#160;这个参数梯度平常多大 →&#160;**统计**，N 个数
  ⛔ 同样，「一阶动量」也**不是**一阶导 ——&#160;它是梯度的滑动平均，
    那个「一阶」指的是**统计上的一阶矩（期望）**。

📌 数都是本脚本现算的（见下面的 assert），不是抄来的。
  ⚠️ 「每秒百亿亿次」那个机器是**随手设的换算基准**，只用来给量级，
    不是在说某台具体的机器。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2

W = 1400

N = 671e9                      # V3 参数量
BYTES = 2                      # 按 bf16 算
CARD = 80 * 2 ** 30            # 随手设的换算基准：一块 80 GiB
W_TIB = 1.22                   # 权重那一块（bf16）

HESS = N * N
HESS_B = HESS * BYTES
HESS_CARDS = HESS_B / CARD
W_CARDS = W_TIB * 1024 / 80
OPS = N ** 3
FLOPS = 1e18                   # 随手设：每秒百亿亿次
YEARS = OPS / FLOPS / 3.15576e7
UNIVERSE = 138e8               # 宇宙年龄约 138 亿年

assert 1e13 < HESS_CARDS < 1.1e13, "卡数量级变了，重新核一遍再改文案"
assert 15 < W_CARDS < 17, "权重那一块的卡数变了"
assert 9e9 < YEARS < 1e10, "年数量级变了"
assert YEARS < UNIVERSE, "这一格的包袱是「比宇宙年龄还短一点」，别抖反了"

# ⭐ Ⓐ：两条抛物线，**梯度一模一样**，可最优步长差 8 倍
GRAD = 2.0
CURV = (1.0, 8.0)              # 弯曲程度
STEPS = tuple(GRAD / h for h in CURV)    # 最优步长 ＝ 梯度 ÷ 二阶导
assert abs(STEPS[0] / STEPS[1] - CURV[1] / CURV[0]) < 1e-9, \
    "最优步长之比必须正好是曲率的反比 ——&#160;这一格的全部论点"


def main():
    f = Fig(W, "二阶导是坡度自己的变化率，也就是这儿弯得有多急。"
               "一阶导告诉你往哪走，二阶导告诉你这个坡还能保持多久。"
               "它的用处很硬：最优步长等于梯度除以二阶导，一步到底。"
               "可它的大小是参数量的平方，"
               "权重十六块卡就装下了，这张表要十万亿块卡；"
               "真要解它是参数量的三次方，百亿亿次的机器要跑九十六亿年。"
               "所以我们故意不算它。"
               "最后一件事：Adam 里那个二阶矩不是二阶导，"
               "一个是对参数求两次导属于几何，"
               "一个是对梯度取平方求平均属于统计，中文都带二阶纯属巧合")

    y0 = f.header(
        "二阶导到底是什么　——　<tspan font-weight=\"700\">以及它跟 Adam 那个"
        "「二阶矩」没有关系</tspan>",
        "⭐ 一阶导告诉你<tspan font-weight=\"700\">往哪走</tspan>；"
        "二阶导告诉你<tspan font-weight=\"700\">这个坡还能保持多久</tspan>",
        [(GR, "Ⓐ 它是什么"), (BL, "Ⓑ 能干嘛"),
         (RD, "Ⓒ 为什么不用"), (PU, "Ⓓ 跟二阶矩的区别")])

    # ══════════ Ⓐ 同样的梯度，不同的弯 ═════════════════════════════
    PH = 360
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 两个参数，<tspan font-weight=\"700\">梯度一模一样</tspan>"
                 "　——　可它们的处境完全不同", GR,
                 sub="⭐ 一阶导看它俩<tspan font-weight=\"700\">没有任何区别</tspan>；"
                     "分得开它俩的，只有二阶导")

    for k, h in enumerate(CURV):
        ox, oy = 240 + k * 420, py + 250        # 谷底
        SX, SY = 150.0, 34.0                    # 画布缩放
        col = (BL, OR)[k]
        # 抛物线 y = 0.5*h*x^2
        pts = []
        x = -1.6
        while x <= 1.601:
            yv = 0.5 * h * x * x
            if yv <= 5.6:
                pts.append((ox + x * SX, oy - yv * SY))
            x += 0.04
        f.path("M" + " L".join("%.1f,%.1f" % p for p in pts), col, 2.4, arrow=False)
        f.line(ox - 190, oy, ox + 190, oy, GY2, 1.0, arrow=False)

        x0 = GRAD / h                           # 此处梯度正好等于 GRAD
        y0p = 0.5 * h * x0 * x0
        px, pyy = ox + x0 * SX, oy - y0p * SY
        f.box(px - 6, pyy - 6, 12, 12, col, col, 6)
        # 切线：斜率相同 → 画同样的角度
        dxp = 68.0
        f.line(px - dxp, pyy + GRAD * dxp * SY / SX,
               px + dxp, pyy - GRAD * dxp * SY / SX, RD, 2.2, arrow=False)
        f.t(px + 84, pyy - 40, "坡度 <tspan font-weight=\"700\">%.0f</tspan>" % GRAD,
            RD, True, 14)
        # 到底的水平距离
        f.line(ox, oy + 22, px, oy + 22, col, 1.8, arrow=False)
        f.t((ox + px) / 2, oy + 44, "离底 <tspan font-weight=\"700\">%.2f</tspan>" % x0,
            col, True, 13.5, "middle")
        f.t(ox, py + 60, ("① 弯得缓", "② 弯得急")[k], col, True, 17, "middle")
        f.t(ox, py + 86, "二阶导 ＝ <tspan font-weight=\"700\">%.0f</tspan>" % h,
            INK, True, 14.5, "middle")

    f.t(700, py + 320,
        "⭐⭐⭐ 两个点的<tspan font-weight=\"700\">坡度完全一样</tspan>，"
        "可一个离底 %.2f、另一个离底 %.2f　——　<tspan font-weight=\"700\">差 %d 倍</tspan>。"
        "<tspan font-weight=\"700\">只看梯度，你分不出这两种处境。</tspan>"
        % (STEPS[0], STEPS[1], int(CURV[1] / CURV[0])),
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 能干嘛 ═══════════════════════════════════════════
    PH2 = 226
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 所以它的用处很硬：<tspan font-weight=\"700\">最优步长可以算出来</tspan>",
                  BL)
    f.box(70, py2 + 52, 600, 120, "#e8f0fe", BL, 8)
    f.t(370, py2 + 92, "最优步长 ＝ <tspan font-weight=\"700\">梯度 ÷ 二阶导</tspan>",
        INK, True, 19, "middle")
    f.t(370, py2 + 130, "在一条抛物线上，<tspan font-weight=\"700\">一步到底</tspan>"
        "　——　不用试，不用调。", INK, size=14.5, anchor="middle")
    f.t(370, py2 + 160, "上面那两个点：%.0f ÷ %.0f ＝ %.2f　·　%.0f ÷ %.0f ＝ %.2f"
        % (GRAD, CURV[0], STEPS[0], GRAD, CURV[1], STEPS[1]),
        GY, size=13.5, anchor="middle")

    f.box(710, py2 + 52, 640, 120, "#e6f4ea", GR, 8)
    f.t(1030, py2 + 90, "而且它正好治那个「拧反了」", GR, True, 16.5, "middle")
    f.t(1030, py2 + 124,
        "灵的旋钮该<tspan font-weight=\"700\">少</tspan>拧、钝的该<tspan "
        "font-weight=\"700\">多</tspan>拧　——　", INK, size=14.5, anchor="middle")
    f.t(1030, py2 + 152,
        "<tspan font-weight=\"700\">除以二阶导之后，不同参数才可比。</tspan>",
        INK, size=14.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 为什么不用 ═══════════════════════════════════════
    PH3 = 322
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ 那<tspan font-weight=\"700\">为什么不用</tspan>？"
                  "　——　因为它的大小是<tspan font-weight=\"700\">参数量的平方</tspan>", RD,
                  sub="⛔ 多参数时它不是一个数，是一张表："
                      "<tspan font-weight=\"700\">每两个参数之间一个数</tspan>")

    f.box(60, py3 + 58, 630, 200, "#fce8e6", RD, 8)
    f.t(375, py3 + 96, "存不下", RD, True, 18, "middle")
    f.t(375, py3 + 136,
        "权重本身 1.22 TiB　——　<tspan font-weight=\"700\">%d 块 80 GiB 的卡</tspan>"
        % round(W_CARDS), INK, size=15, anchor="middle")
    f.t(375, py3 + 172,
        "这张表　——　<tspan font-weight=\"700\">%.0f 万亿块</tspan>"
        % (HESS_CARDS / 1e12), INK, True, 17, "middle")
    f.t(375, py3 + 208, "不是十万块。是<tspan font-weight=\"700\">十万亿块</tspan>。",
        RD, True, 15, "middle")
    f.t(375, py3 + 240, "（%.1e 个数，按 bf16 算）" % HESS, GY2, size=12.5,
        anchor="middle")

    f.box(710, py3 + 58, 640, 200, "#fce8e6", RD, 8)
    f.t(1030, py3 + 96, "也算不动", RD, True, 18, "middle")
    f.t(1030, py3 + 136,
        "真要拿它去解，朴素做法是<tspan font-weight=\"700\">参数量的三次方</tspan>",
        INK, size=15, anchor="middle")
    f.t(1030, py3 + 174,
        "拿一台每秒百亿亿次的机器跑 ——　<tspan font-weight=\"700\">%.0f 亿年</tspan>"
        % (YEARS / 1e8), INK, True, 17, "middle")
    f.t(1030, py3 + 212,
        "⭐ 宇宙到现在才 <tspan font-weight=\"700\">%.0f 亿年</tspan>。"
        % (UNIVERSE / 1e8), RD, True, 15, "middle")
    f.t(1030, py3 + 244,
        "⚠️ 那台机器是随手设的换算基准，只给量级", GY2, size=12.5, anchor="middle")

    f.t(700, py3 + 288,
        "⭐⭐⭐ 所以「我们故意不算二阶导」<tspan font-weight=\"700\">不是偷懒</tspan>"
        "　——　<tspan font-weight=\"700\">它压根不在可能性范围内。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓓ 二阶矩 ≠ 二阶导 ═════════════════════════════════
    PH4 = 352
    py4 = f.panel(0, py3 + PH3 + 20, W, PH4,
                  "Ⓓ ⛔ 最后一件事：<tspan font-weight=\"700\">"
                  "Adam 里那个「二阶矩」不是「二阶导」</tspan>", PU,
                  sub="⭐ 它们<tspan font-weight=\"700\">中文都带「二阶」，纯属巧合</tspan>"
                      "　——　这是这一讲最容易被中文坑到的一处")

    ROWS = (("对谁做", "对<tspan font-weight=\"700\">参数</tspan>求<tspan "
             "font-weight=\"700\">两次导</tspan>",
             "对<tspan font-weight=\"700\">梯度</tspan>取<tspan "
             "font-weight=\"700\">平方</tspan>再求平均"),
            ("得到什么", "曲率　——　这儿<tspan font-weight=\"700\">弯得多急</tspan>",
             "这个参数的梯度<tspan font-weight=\"700\">平常多大</tspan>"),
            ("属于哪一路", "<tspan font-weight=\"700\">几何</tspan>",
             "<tspan font-weight=\"700\">统计</tspan>"),
            ("多少个数", "<tspan font-weight=\"700\">N 的平方</tspan>",
             "<tspan font-weight=\"700\">N</tspan>"))
    CX1, CX2, CX3 = 230, 640, 1060
    f.t(CX2, py4 + 62, "二阶<tspan font-weight=\"700\">导</tspan>（Hessian）",
        RD, True, 16, "middle")
    f.t(CX3, py4 + 62, "二阶<tspan font-weight=\"700\">矩</tspan>（Adam 的分母）",
        GR, True, 16, "middle")
    for k, (lab, a, b) in enumerate(ROWS):
        yy = py4 + 88 + k * 44
        f.box(60, yy, 1290, 38, "#fff" if k % 2 else "#f8f9fa", "#fff", 4)
        f.t(CX1, yy + 26, lab, GY, True, 13.5, "middle")
        f.t(CX2, yy + 26, a, INK, size=14, anchor="middle")
        f.t(CX3, yy + 26, b, INK, size=14, anchor="middle")

    f.t(700, py4 + 288,
        "⛔ 同样，「<tspan font-weight=\"700\">一阶动量</tspan>」也<tspan "
        "font-weight=\"700\">不是</tspan>一阶导　——　它是<tspan font-weight=\"700\">"
        "梯度的滑动平均</tspan>，那个「一阶」指的是<tspan font-weight=\"700\">"
        "统计上的一阶矩（期望）</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(700, py4 + 318,
        "⭐⭐ 所以 Adam <tspan font-weight=\"700\">不是「近似了 Hessian」</tspan>"
        "　——　它用 <tspan font-weight=\"700\">2 个 N</tspan> 个数，"
        "<tspan font-weight=\"700\">换了一条路</tspan>，而不是去逼近那张 N² 的表。",
        PU, True, 14.5, "middle")
    f._pan = None

    yb = f.band(py4 + PH4 + 18, "ok", "⭐ 两句话记住", [
        "<tspan font-weight=\"700\">二阶导 ＝ 坡度自己的变化率 ＝ 这儿弯得多急。</tspan>"
        "一阶导告诉你往哪走，它告诉你这个坡还能保持多久　——　"
        "<tspan font-weight=\"700\">最优步长 ＝ 梯度 ÷ 二阶导，一步到底</tspan>。"
        "⛔ 可它是 N² 个数，存不下也算不动，所以我们故意不算。",
        "<tspan font-weight=\"700\">而 Adam 的「二阶矩」跟它没有关系</tspan>　——　"
        "一个对<tspan font-weight=\"700\">参数</tspan>求两次导（几何），"
        "一个对<tspan font-weight=\"700\">梯度</tspan>取平方求平均（统计）。"
        "<tspan font-weight=\"700\">中文都带「二阶」，纯属巧合。</tspan>",
    ])

    yb = f.src(yb + 10,
               "数都是本脚本现算的：Hessian %.3e 个数（按 bf16 ＝ %.2e 字节）；"
               "换算基准一块 80 GiB ⇒ %.2e 块，而权重 1.22 TiB 只要 %d 块。"
               % (HESS, HESS_B, HESS_CARDS, round(W_CARDS)),
               "解它按 N³ ＝ %.2e 次运算，拿每秒 1e18 次的机器 ⇒ %.2e 年。"
               "⚠️ 那台机器是<tspan font-weight=\"700\">随手设的换算基准</tspan>，"
               "只用来给量级，不是在说某台具体的机器；宇宙年龄取 138 亿年。"
               % (OPS, YEARS))

    f.save("fig4-second.svg", yb + 14)


# ⭐ 2026-09-22：加这个守卫，是因为 **正文里那三个数要从这里现取**
#   （见 topic04-build.py 的「同一个数只能有一处来源」）。
#   ⛔ 没有守卫的话，一 import 就会把 SVG 又画一遍。
if __name__ == "__main__":
    main()
