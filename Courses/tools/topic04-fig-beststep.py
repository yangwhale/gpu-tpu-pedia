# -*- coding: utf-8 -*-
r"""专题四 · §3.3「那一步到底该迈多大」——&#160;最优步长是可以算出来的

⭐⭐⭐ 2026-09-17 新画。取法来自**李宏毅**（台大）讲 Gradient Descent 的那一段。
  ⛔ 我们 §3.3 原来讲到「Adam 把梯度的量纲除掉了，每步大约 0.2 × 学习率」就停了。
    那回答的是「除完之后是多少」，**没有回答「为什么要除」**。
  ⭐ 而他那一段给了这个问题最深的一个答案，而且**小到可以当场验**：

      最优的一步　＝　|一阶导| ÷ 二阶导

  在一条抛物线上，这个值**恰好等于「你离最低点还有多远」** ——&#160;
  一步就能到底。（脚本里当场算给你看。）

⭐⭐ 而它真正的杀伤力在 Ⓑ：**只看梯度大小是会看错的。**
  梯度大的那个点可能离底更近 ——&#160;
  跨参数比较必须**除以二阶导**，否则你比的是两个不可比的量。
  ⛔ 这一条跟本讲那条「口径不同不能并排比」是同一个形状，
    只不过这次两个量的分母藏在曲率里。

⭐ Ⓒ 收回到本讲：二阶导在几千亿参数上**算不起**，
  所以自适应优化器干的事是 ——&#160;**拿一阶导的历史去填那个分母的位置**。
  ⛔⛔ 2026-09-19 T10 修：原来这里写「**二阶导的替身**」——&#160;量纲对不上。
    二阶导是 loss/参数²，√v 是 loss/参数，差一个「参数」。
    ⭐ 而把这件事说清楚反而更值钱：正因为 m/√v **无量纲**，
      Adam 的学习率量纲是「参数」，跟 SGD 的「参数²/loss」不是一回事 ——
      **这才是 Adam 学习率好迁移的真正原因**，也跟本讲 3.3 对上了。
  Adagrad 用累加平方和，Adam 用滑动平均。
  **分母上那一坨，就是这么来的。**

📌 「Best step ＝ |First derivative| / Second derivative」与「Use first derivative
  to estimate second derivative」是李宏毅课程投影片上的原句（多份公开课堂笔记
  与二次分发的课件均可核）。⛔ 图是我们自己重画的，数是自己挑的。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ⭐ 两条抛物线 f = a·x²，各取一个点。挑的原则只有一条：
#   **左边那个点梯度更大，却离底更近** ——&#160;不这样，Ⓑ 就没有话说。
CASES = ((2.00, 0.5, RD, "#fce8e6", "陡"),        # 窄谷
         (0.25, 2.0, BL, "#e8f0fe", "平"))        # 宽谷
_d1 = [2 * a * x for a, x, *_ in CASES]
_d2 = [2 * a for a, x, *_ in CASES]
assert _d1[0] > _d1[1], "左边那个点的梯度必须更大"
assert CASES[0][1] < CASES[1][1], "而它必须离底更近 ——　这才是 Ⓑ 的全部意思"
for (a, x, *_), g, h in zip(CASES, _d1, _d2):
    assert abs(g / h - x) < 1e-12, "一阶÷二阶 应当正好等于到底的距离"


def main():
    f = Fig(W, "在一条抛物线上，一阶导除以二阶导，正好等于这一点离最低点的距离，"
               "所以最优的一步是可以算出来的。而只看梯度大小会看错："
               "陡谷上那个点梯度更大，却离底更近。"
               "跨参数比较必须除以二阶导。但二阶导在大模型上算不起，"
               "所以自适应优化器改用一阶导的历史去估它 —— "
               "Adam 分母上那一坨就是这么来的")

    y0 = f.header(
        "那一步该迈多大　——　<tspan font-weight=\"700\">"
        "它其实是有标准答案的</tspan>",
        "⛔ 前面只讲了「除完之后是多少」——&#160;"
        "<tspan font-weight=\"700\">这一格回答「为什么要除」</tspan>",
        [(PU, "最优步长 ＝ |一阶导| ÷ 二阶导")])

    # ══════════ Ⓐ 抛物线上一步到底 ═════════════════════════════════
    PH = 430
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 在一条抛物线上，<tspan font-weight=\"700\">"
                 "「一阶导 ÷ 二阶导」正好就是「离底还有多远」</tspan>", PU,
                 sub="⭐ 所以理论上<tspan font-weight=\"700\">一步就能到底</tspan>"
                     " ——&#160;不用试、不用调")

    # 画两条谷，并排
    for i, (a, x0, col, fill, tag) in enumerate(CASES):
        ox = 250 + i * 640                    # 谷底的 x 像素
        oy = py + 330                         # 谷底的 y 像素
        SX, SY = 100.0, 58.0                  # 每单位 x / y 多少像素
        # ⛔ 2026-09-17：第一版两条曲线都画到 t = ±2.6，结果**陡谷那条冲出了面板**
        #   （a=2 时 2.6² × 58 ≈ 780 px，比整个面板还高）。
        #   ⭐ 判据：**画函数图像，横轴范围要按各自的纵向上限反算，不能两条共用一个。**
        #     共用一个看起来「对称好看」，而它只对其中一条成立。
        HMAX = 196.0                          # 曲线允许的最大高度（像素）
        TMAX = min(2.6, (HMAX / (a * SY)) ** 0.5)
        assert TMAX >= x0 * 1.15, "画的范围要包得住那个点"
        d, t = "M %.1f %.1f" % (ox - TMAX * SX, oy - a * TMAX ** 2 * SY), -TMAX
        while t <= TMAX + 1e-9:
            d += " L %.1f %.1f" % (ox + t * SX, oy - a * t * t * SY)
            t += 0.04
        f.path(d, col, 1.8, arrow=False)
        f.line(ox - (TMAX + .3) * SX, oy, ox + (TMAX + .3) * SX, oy, GY2, 1, arrow=False)

        # 当前点
        px, pyy = ox + x0 * SX, oy - a * x0 * x0 * SY
        f.box(px - 6, pyy - 6, 12, 12, col, col, 6)
        f.t(px + 14, pyy - 12, "你在这儿", col, True, 13.5)
        # 到底的距离
        f.line(ox, oy + 22, px, oy + 22, col, 1.4, dash="4 3", arrow=False)
        f.t((ox + px) / 2.0, oy + 42, "还差 %g" % x0, col, True, 14, "middle")
        f.t(ox, oy + 68, "最低点", GY2, size=12, anchor="middle")

        g, h = 2 * a * x0, 2 * a
        f.t(ox, py + 52, "%s 谷" % tag, col, True, 20, "middle")
        f.t(ox, py + 82, "一阶导 %g　·　二阶导 %g" % (g, h), INK, True, 14.5, "middle")
        f.t(ox, py + 108, "%g ÷ %g ＝ <tspan font-weight=\"700\">%g</tspan>"
            % (g, h, g / h), col, True, 16, "middle")
        f.t(ox, py + 132, "⭐ 跟「还差 %g」<tspan font-weight=\"700\">一模一样</tspan>" % x0,
            GY, size=13, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 只看梯度会看错 ═══════════════════════════════════
    PH2 = 264
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐⭐ 而这就解释了一件反直觉的事："
                  "<tspan font-weight=\"700\">梯度大，不代表离得远</tspan>", RD,
                  sub="⛔ 所以<tspan font-weight=\"700\">只看梯度大小，"
                      "跨参数就会比错</tspan>")

    f.box(70, py2 + 38, 600, 180, "#fce8e6", RD, 8)
    f.t(370, py2 + 74, "只看一阶导（梯度）", RD, True, 18, "middle")
    f.t(370, py2 + 108, "陡谷 2.0　＞　平谷 1.0", INK, True, 16, "middle")
    f.t(370, py2 + 138, "→　「陡谷那个离得更远，该迈大步」", GY, size=14, anchor="middle")
    f.t(370, py2 + 176, "⛔ 错了。它其实离底<tspan font-weight=\"700\">更近</tspan>"
        "（0.5 对 2.0）", RD, True, 15, "middle")

    f.t(700, py2 + 124, "→", GY2, True, 22, "middle")

    f.box(730, py2 + 38, 600, 180, "#e6f4ea", GR, 8)
    f.t(1030, py2 + 74, "除以二阶导之后", GR, True, 18, "middle")
    f.t(1030, py2 + 108, "0.5　和　2.0", INK, True, 16, "middle")
    f.t(1030, py2 + 138, "→　<tspan font-weight=\"700\">两个都正好对</tspan>",
        GR, True, 15, "middle")
    f.t(1030, py2 + 176, "⭐ 除完之后，两个参数才<tspan font-weight=\"700\">可比</tspan>",
        GR, size=14, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 回到大模型 ════════════════════════════════════════
    PH3 = 250
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐ 那为什么没人真去算二阶导？"
                  "<tspan font-weight=\"700\">因为算不起</tspan>", OR,
                  sub="⭐ 这一格把上面那条漂亮的式子，"
                      "<tspan font-weight=\"700\">接回我们这一讲的现实</tspan>")

    STEPS = (
        (RD, "#fce8e6", "二阶导", "每两个参数之间都有一个",
         "几千亿参数　→　根本存不下，更别说算"),
        (OR, "#fef7e0", "那就估它", "用<tspan font-weight=\"700\">一阶导的历史</tspan>",
         "梯度一直很大 →　多半在陡的地方"),
        (GR, "#e6f4ea", "落到实现", "Adagrad：累加平方和<br>Adam：滑动平均",
         "⭐ 分母上那一坨，就是这么来的"),
    )
    for i, (col, fill, tag, what, note) in enumerate(STEPS):
        x = 40 + i * 442
        f.box(x, py3 + 36, 418, 170, fill, col, 8)
        f.box(x, py3 + 36, 418, 4, col, col, 2)
        f.t(x + 209, py3 + 74, tag, col, True, 19, "middle")
        for k, ln in enumerate(what.split("<br>")):
            f.t(x + 209, py3 + 108 + k * 24, ln, INK, True, 14.5, "middle")
        f.t(x + 209, py3 + 182, note, col, size=12.5, anchor="middle")
        if i:
            f.t(x - 14, py3 + 120, "→", GY2, True, 20, "middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "所以自适应优化器分母上那一项，不是「一个技巧」——　它在替二阶导<tspan text-decoration=\"underline\">占那个位置</tspan>",
                ("⛔ 但要说准：<tspan font-weight=\"700\">它不是二阶导，连量纲都不是</tspan>。"
                 "二阶导是 loss ÷ 参数²，而 √v 大致是梯度的均方根，量纲是 loss ÷ 参数 ——　"
                 "<tspan font-weight=\"700\">差一个「参数」</tspan>。"
                 "所以严格的牛顿步 g ÷ H 本身就已经是「参数」量纲、不用再乘学习率；"
                 "而 Adam 的 m ÷ √v <tspan font-weight=\"700\">是无量纲的</tspan>。",
                 "⭐⭐⭐ 而这个差别正好解释了本讲 3.3 那个结论："
                 "<tspan font-weight=\"700\">Adam 把梯度的尺度除干净了，"
                 "每步大约挪 0.2 × 学习率</tspan>　——　既然 m ÷ √v 无量纲，"
                 "<tspan font-weight=\"700\">Adam 的学习率量纲就是「参数」本身</tspan>，"
                 "跟 SGD 那个「参数² ÷ loss」<tspan text-decoration=\"underline\">不是同一个东西</tspan>。"
                 "这才是 Adam 的学习率好调、好迁移的真正原因。",
                 "⛔ 它替的只是<tspan font-weight=\"700\">位置</tspan>，不是那个量："
                 "「梯度一直大 ＝ 曲率大」是个经验假设，它不总成立 ——　"
                 "这正是为什么调参这件事至今还是手艺。"))

    yb = f.src(yb + 16,
               "📌 「最优步长 ＝ |一阶导| ÷ 二阶导」与「用一阶导去估二阶导」"
               "取自<tspan font-weight=\"700\">李宏毅</tspan>（台大）Gradient Descent "
               "课程投影片的讲法　——　<tspan font-weight=\"700\">图是我们自己重画的</tspan>。",
               "⭐ 图上四个数都能当场验：抛物线 f ＝ a·x² 上，"
               "一阶导 2ax ÷ 二阶导 2a ＝ x，<tspan font-weight=\"700\">"
               "正好是到底的距离</tspan>。脚本里带 assert。")

    f.save("fig4-beststep.svg", yb + 14)


if __name__ == "__main__":
    main()
