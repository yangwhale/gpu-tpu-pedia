# -*- coding: utf-8 -*-
r"""专题四 · §1.1「导数 / 偏导数 / 链式法则」——&#160;用一句话说完：兑换率

⭐⭐⭐ 2026-09-17 新画。现场原话：「反向传播、梯度下降、微积分、偏导数
  这些都给我讲得明明白白的，到底这个 loss 它怎么变成梯度，怎么更新 ……画图讲」。
  ⛔ 清点下来发现一个洞：**这一讲从没说过「导数是什么」。**
    §1.1 一句「求导是高中的事」就过去了，§1.4 讲的是
    「偏导数**为什么**让事情变容易」——&#160;那是另一个问题。
    ⭐ 对没修过微积分的人，这就是第一道坎，而且过不去后面全是空的。

⭐⭐ 这张图不讲极限、不讲 ε-δ，只讲一件事：**导数是一个兑换率。**
  · **导数** ——&#160;这个旋钮<tspan>拧一点点</tspan>，结果变多少。
    ⛔ 注意它**不是**「结果是多少」，是「你动一格，它动几格」。
  · **偏导数** ——&#160;其余三千亿个**按住不动**，只拧这一个时的兑换率。
    「偏」字的全部含义就是那三个字：**按住不动**。
  · **链式法则** ——&#160;中间隔着好几级，**每级一个兑换率，一路乘起来**。
    ⭐⭐⭐ 就是**换汇**：人民币→港币→美元，每步一个汇率，总汇率是乘出来的。

⭐⭐⭐ 「兑换率」这个说法不是为了好听，它**自带两个后续的钩子**：
  ① 接 §1.5 ——&#160;一串数相乘，**从哪一头开始乘**结果一样、代价差一万年。
  ② 接 §3.3 ——&#160;正文那个「折算系数 / 量纲」问题，
     用兑换率讲就是一句话：**梯度的单位是「loss 每参数」，
     而你要的是「参数」，所以中间必须再乘一个东西。**
  ⛔ 这两个钩子都是本讲自己的，不是从哪儿抄的。

⚠️ 图上那几个倍数（×2 / ×0.5 / ×3）是**编出来的示意数**，
  它们唯一的作用是让「乘起来」这件事看得见 ——&#160;脚本里 assert 了乘积。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ⭐ Ⓐ 的小例子：推子动 0.01，结果动 0.03 → 兑换率 3
D_KNOB, D_OUT = 0.01, 0.03
RATE_A = D_OUT / D_KNOB
assert abs(RATE_A - 3.0) < 1e-12

# ⭐ Ⓒ 的链条：三级，每级一个兑换率
CHAIN = (("推子", "中间量甲", 2.0),
         ("中间量甲", "中间量乙", 0.5),
         ("中间量乙", "难听程度", 3.0))
TOTAL = 1.0
for _, _, r in CHAIN:
    TOTAL *= r
assert abs(TOTAL - 3.0) < 1e-12, "总兑换率必须是三个乘起来，不是我写上去的"
# ⛔ 顺手挑一组「不都大于 1」的数 ——&#160;不然读者会以为链式法则只会放大
assert any(r < 1.0 for _, _, r in CHAIN), \
    "链条里要有一级是缩小的，否则「乘起来」会被读成「越乘越大」"


def main():
    f = Fig(W, "导数说白了就是一个兑换率：这个旋钮拧一点点，结果变多少。"
               "注意它不是结果是多少，是你动一格它动几格。"
               "偏导数就是把其余所有旋钮按住不动，只拧这一个时的兑换率，"
               "偏字的全部含义就是按住不动这三个字。"
               "而把三千亿个旋钮各自的兑换率排成一列，那一列就叫梯度。"
               "最后，旋钮和结果之间隔着很多级，每一级有自己的兑换率，"
               "总的兑换率就是把它们一路乘起来 —— 这就是链式法则，"
               "跟人民币换港币再换美元是同一回事")

    y0 = f.header(
        "导数、偏导数、链式法则　——　<tspan font-weight=\"700\">"
        "其实是同一个词：兑换率</tspan>",
        "⛔ 这一格<tspan font-weight=\"700\">不讲极限、不讲公式</tspan>"
        "　——　只讲这三个词到底在说什么事",
        [(BL, "导数：动一格，变几格"), (GR, "偏导数：其余按住"),
         (PU, "链式法则：一路乘")])

    # ══════════ Ⓐ 导数 ＝ 动一格，它动几格 ═════════════════════════
    PH = 348
    py = f.panel(0, y0, W, PH,
                 "Ⓐ <tspan font-weight=\"700\">导数</tspan>问的不是"
                 "「现在是多少」，是<tspan font-weight=\"700\">「你动一格，它动几格」</tspan>",
                 BL,
                 sub="⭐ 把参数想成一个旋钮，loss 是它右边那个读数")

    # 旋钮：一根竖槽 ＋ 一个把手
    KX, KY, KH = 220, py + 66, 190
    f.box(KX - 4, KY, 8, KH, "#f1f3f4", GY2, 4)
    f.box(KX - 34, KY + 118, 68, 22, "#fff", BL, 5, sw=1.8)
    f.t(KX, KY + KH + 30, "一个参数", INK, True, 15, "middle")
    f.t(KX, KY + KH + 52, "（三千亿个之一）", GY2, size=12, anchor="middle")
    # 往上推一点点
    f.line(KX + 54, KY + 124, KX + 54, KY + 96, BL, 2.2)
    f.t(KX + 66, KY + 116, "往上推一点点", BL, True, 13.5)
    f.t(KX + 66, KY + 138, "＋%.2f" % D_KNOB, GY, size=12.5)

    f.t(430, KY + 116, "→", GY2, True, 26, "middle")

    f.box(490, py + 66, 300, 190, "#fff", GY2, 8)
    f.t(640, py + 104, "读数（loss）", GY, True, 15, "middle")
    f.t(640, py + 152, "5.00", GY2, size=22, anchor="middle", mono=True)
    f.t(640, py + 182, "↓", GY2, True, 18, "middle")
    f.t(640, py + 214, "5.03", INK, True, 24, "middle", mono=True)
    f.t(640, py + 242, "变了 ＋%.2f" % D_OUT, BL, True, 13.5, "middle")

    f.box(850, py + 66, 500, 190, "#e8f0fe", BL, 8)
    f.t(1100, py + 104, "那这个旋钮的<tspan font-weight=\"700\">导数</tspan>就是",
        BL, True, 17, "middle")
    f.t(1100, py + 152, "%.2f ÷ %.2f ＝ <tspan font-weight=\"700\">%d</tspan>"
        % (D_OUT, D_KNOB, int(RATE_A)), INK, True, 22, "middle")
    f.t(1100, py + 196, "⭐ 读作：<tspan font-weight=\"700\">你动一格，它动三格</tspan>",
        BL, True, 15, "middle")
    f.t(1100, py + 230, "⛔ 它<tspan font-weight=\"700\">不是</tspan>「读数是 5.00」"
        "　——　那是<tspan font-weight=\"700\">值</tspan>，这是<tspan font-weight=\"700\">兑换率</tspan>",
        GY, size=13, anchor="middle")

    f.t(700, py + 306,
        "⭐⭐ 顺带记住它的<tspan font-weight=\"700\">单位</tspan>："
        "<tspan font-weight=\"700\">loss 每参数</tspan>　——　"
        "后面讲<tspan font-weight=\"700\">「那一步该迈多大」</tspan>那一节，整节都是被这个单位逼出来的。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 「偏」字 ＝ 其余全按住 ════════════════════════════
    PH2 = 308
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐⭐ 那<tspan font-weight=\"700\">「偏」</tspan>字是什么意思"
                  "　——　就三个字：<tspan font-weight=\"700\">按住不动</tspan>", GR,
                  sub="⭐ 一台三千亿个旋钮的调音台，一次只拧一个，其余全按住")

    NK, SX0, SGAP = 9, 150, 108
    for k in range(NK):
        x = SX0 + k * SGAP
        live = (k == 4)
        col = GR if live else GY2
        f.box(x - 3, py2 + 48, 6, 120, "#f1f3f4", GY2, 3)
        f.box(x - 26, py2 + (74 if live else 108), 52, 18,
              "#fff", col, 4, sw=1.8 if live else 1.0)
        if live:
            f.line(x + 42, py2 + 116, x + 42, py2 + 84, GR, 2.2)
            f.t(x, py2 + 196, "只动这一个", GR, True, 14, "middle")
        else:
            f.t(x, py2 + 196, "按住", GY2, size=12, anchor="middle")
    f.t(SX0 + NK * SGAP + 10, py2 + 120, "……　其余三千亿个，全按住",
        GY2, size=13)

    f.box(150, py2 + 224, 1200, 58, "#e6f4ea", GR, 8)
    f.t(750, py2 + 260,
        "⭐⭐⭐ <tspan font-weight=\"700\">偏导数 ＝ 其余全按住时，这一个旋钮的兑换率。</tspan>"
        "　而把三千亿个旋钮各自的那个数排成一列 ——&#160;"
        "<tspan font-weight=\"700\">那一列就叫「梯度」。</tspan>",
        INK, True, 15.5, "middle")
    f._pan = None

    # ══════════ Ⓒ 链式法则 ＝ 换汇 ═════════════════════════════════
    PH3 = 336
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐ 可旋钮<tspan font-weight=\"700\">不直接连到读数</tspan>"
                  "　——　中间隔着好几级", PU,
                  sub="⭐ 每一级有自己的兑换率，"
                      "<tspan font-weight=\"700\">总兑换率就是一路乘起来</tspan>")

    BX, BW2, BGAP = 80, 230, 84
    NODES = ("推子", "中间量甲", "中间量乙", "难听程度")
    for k, name in enumerate(NODES):
        x = BX + k * (BW2 + BGAP)
        col = PU if k in (0, len(NODES) - 1) else GY2
        f.box(x, py3 + 56, BW2, 74, "#fff", col, 8, sw=1.6)
        f.t(x + BW2 / 2, py3 + 100, name, INK if k else PU, True, 16, "middle")
        if k < len(NODES) - 1:
            mx = x + BW2 + BGAP / 2
            f.line(x + BW2 + 6, py3 + 93, x + BW2 + BGAP - 6, py3 + 93, PU, 2.0)
            f.t(mx, py3 + 46, "× %.1f" % CHAIN[k][2], PU, True, 17, "middle")

    f.box(80, py3 + 158, 620, 116, "#f3e8fd", PU, 8)
    f.t(390, py3 + 194, "总兑换率 ＝ 一路乘起来", PU, True, 17, "middle")
    f.t(390, py3 + 236, "%.1f × %.1f × %.1f ＝ <tspan font-weight=\"700\">%d</tspan>"
        % (CHAIN[0][2], CHAIN[1][2], CHAIN[2][2], int(TOTAL)),
        INK, True, 22, "middle")

    f.box(740, py3 + 158, 610, 116, "#e8f0fe", BL, 8)
    f.t(1045, py3 + 194, "⭐ 这就是<tspan font-weight=\"700\">换汇</tspan>",
        BL, True, 17, "middle")
    f.t(1045, py3 + 228, "人民币 → 港币 → 美元，每步一个汇率",
        INK, size=14.5, anchor="middle")
    f.t(1045, py3 + 254, "总汇率<tspan font-weight=\"700\">当然是乘出来的</tspan>"
        "　——　链式法则就这一件事", GY, size=13.5, anchor="middle")

    f.t(700, py3 + 306,
        "⛔ 而「<tspan font-weight=\"700\">这一串数从哪一头开始乘</tspan>」"
        "　——　结果完全一样，<tspan font-weight=\"700\">代价差一万年</tspan>。"
        "那是紧接着 `fig-reverse` 那一格的事。",
        INK, size=14.5, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "所以这一讲后面所有的东西，都只用到这三句话",
                ("✅ <tspan font-weight=\"700\">① 导数 ＝ 你动一格，它动几格（兑换率）。"
                 "② 偏导数 ＝ 其余全按住时的那个兑换率。"
                 "③ 链式法则 ＝ 中间每一级的兑换率，一路乘起来。</tspan>"
                 "——&#160;没有极限，没有 ε，没有要背的公式。",
                 "⛔ 唯一一个<tspan font-weight=\"700\">真的要小心</tspan>的点："
                 "兑换率<tspan font-weight=\"700\">不是「值」</tspan>。"
                 "读数是 5.00 跟「动一格变三格」是两件完全不同的事 ——&#160;"
                 "本讲后面每次说「梯度大」，说的都是<tspan font-weight=\"700\">后者</tspan>。",
                 "⭐⭐ 而 Ⓐ 末尾那个<tspan font-weight=\"700\">单位</tspan>是给 §3.3 埋的："
                 "梯度的单位是「loss 每参数」，可你要的是「参数该挪多少」——&#160;"
                 "<tspan font-weight=\"700\">两边对不上，中间必须再乘一个东西，那个东西就是学习率。</tspan>"))

    yb = f.src(yb + 16,
               "⚠️ Ⓐ 的 5.00 → 5.03、Ⓒ 的 ×2 / ×0.5 / ×3 都是"
               "<tspan font-weight=\"700\">编出来的示意数</tspan>，"
               "唯一的作用是让「相除」和「相乘」这两件事看得见。"
               "⭐ 脚本里 assert 了两条：总兑换率必须真的是三个乘积，"
               "而且<tspan font-weight=\"700\">链条里要有一级是缩小的</tspan> ——&#160;"
               "不然「乘起来」会被读成「越乘越大」，"
               "而那正是梯度消失/爆炸那一节要讲的反面。",
               "⭐ 这一格<tspan font-weight=\"700\">刻意不碰极限</tspan>："
               "严格地说导数是「动的那一点点趋于 0 时的极限」，"
               "而这张图画的是<tspan font-weight=\"700\">差商</tspan>。"
               "⛔ 对本讲够用 ——&#160;而且 §1.1 那个「笨办法」用的**正是差商**，"
               "所以这里不严格反而接得更顺。",
               "⭐⭐ 「兑换率」这个说法不是为了好听，它自带两个钩子，"
               "两个都是本讲自己的：<tspan font-weight=\"700\">一串数相乘，"
               "从哪头开始乘代价差一万年（§1.5）；"
               "而单位对不上所以必须再乘一个折算系数（§3.3）。</tspan>")

    f.save("fig4-slider.svg", yb + 14)


if __name__ == "__main__":
    main()
