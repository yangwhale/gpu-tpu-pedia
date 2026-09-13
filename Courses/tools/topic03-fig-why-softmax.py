# -*- coding: utf-8 -*-
r"""专题三 · §1.2b「为什么是 softmax、为什么 K 和 V 要分家」

⭐⭐⭐ 2026-09-14 **整张重画**。现场原话：
   「图是图，字是字。图的目的是让你把**原理画出来**，让人一目了然，
     而不是把本来应该写在文字部分的东西放在表格里 ——&nbsp;那些表格不是图。」

   上一版是**三栏纯文字**（一千一百多字、零个图形）：
   它把「softmax 满足三条性质」写成了三行字。⛔ 那是正文，不是图。

   这一版只画两件事，每件事配一个**能一眼看懂的画面**，字少、字大：

   ① **为什么是 softmax** ——&nbsp;让**同一组分数走三条归一化路线**，
      看柱子长什么样：
        · 直接当权重 →&nbsp;有**负柱**（「负着看一眼」没有意义）
        · 除以总和   →&nbsp;和为 1 了，**负柱还在**
        · softmax    →&nbsp;**全部朝上**
      ⭐ 三组柱子并排，结论不用写字。
   ② **为什么 K 和 V 要分家** ——&nbsp;画一张**图书馆卡片**：
      你按**书脊**找（K），取回的是**书里的内容**（V）。
      K＝V 就等于「卡片上写的就是全文」——&nbsp;
      **那你只能找到长得像你问题的东西。**

📌 口径：① 三条性质由 softmax 定义直接给出，柱子的数当场算并断言；
   「索引器换 ReLU，for throughput」见 §6.4b；② 的卡片是类比，不是论文原话。
"""
import math

from topic03_draw import (Fig, BL, GR, RD, GY, INK, GY2, LINE, LINE2, BG2)

W = 1400

# 一组分数 —— 负的那个是整张图的支点
S = [2.0, -1.0, 3.0, 1.0]
TOK = ["昨天", "苹果", "天气", "很好"]


def main():
    ex = [math.exp(v) for v in S]
    sm = [v / sum(ex) for v in ex]
    div = [v / sum(S) for v in S]                  # 「除以总和」那一路
    assert abs(sum(sm) - 1) < 1e-9 and min(sm) > 0
    assert min(div) < 0 and abs(sum(div) - 1) < 1e-9   # 和为 1，但有负

    f = Fig(W, "为什么是 softmax：同一组分数走三条归一化路线，直接用和除以总和"
               "都会留下负权重，只有 softmax 全部朝上；"
               "以及为什么 K 和 V 要分家：按书脊找，取回的是书里的内容")
    f.marks = set()
    y0 = f.header(
        "两个「为什么不那样做」",
        "把同一组分数<tspan font-weight=\"700\">走三条路</tspan>，"
        "柱子长什么样，一眼就知道",
        [(GR, "能用"), (RD, "不能用"), (BL, "检索是非对称的")])

    # ══════════ 上半：三条归一化路线 ══════════════════════════════
    PH = 338
    py = f.panel(0, y0, W, PH,
                 "① 注意力最后一步是「加权平均」——　所以权重只能长成一种样子",
                 GR, sub="同一组分数，三条路")

    bx, by = 40, py + 16
    CW, GAP = 96, 14
    f.t(bx, by + 26, "原始分数", GY, True, 16)
    for i, (tk, v) in enumerate(zip(TOK, S)):
        x = bx + 132 + i * (CW + GAP)
        f.box(x, by, CW, 44, BG2, LINE2, 6)
        f.t(x + CW / 2.0, by + 19, tk, GY, size=13, anchor="middle")
        f.t(x + CW / 2.0, by + 38, "%+.0f" % v, RD if v < 0 else INK,
            True, 18, "middle")
    f.t(bx + 132 + 4 * (CW + GAP) + 8, by + 30, "← 有一个是负的", RD, True, 16)

    ROUTES = [
        ("直接当权重", S, RD, "⛔ 负着看一眼？"),
        ("除以总和", div, RD, "⛔ 和为 1 了，负柱还在"),
        ("softmax", sm, GR, "✅ 全部朝上"),
    ]
    ry = by + 66
    ZERO = ry + 124
    UP, DN = 76, 32
    for j, (name, vals, col, verdict) in enumerate(ROUTES):
        gx = 40 + j * 452
        f.t(gx, ry + 20, name, col, True, 21)
        f.t(gx, ry + 42, verdict, col, True, 15)
        ax0 = gx + 4
        f.line(ax0, ZERO, ax0 + 382, ZERO, LINE, 1.2, arrow=False)
        f.t(ax0 + 388, ZERO + 5, "0", GY2, size=13)
        mx = max(abs(v) for v in vals)
        for i, v in enumerate(vals):
            h = (UP if v >= 0 else DN) * abs(v) / mx
            x = ax0 + 12 + i * 92
            f.box(x, ZERO - h if v >= 0 else ZERO, 60,
                  h if h > 1 else 1.5, col if v >= 0 else RD, "none", 3)
            f.t(x + 30, ZERO - h - 8 if v >= 0 else ZERO + h + 18,
                ("%.2f" % v) if abs(v) < 1 else ("%+.1f" % v),
                col if v >= 0 else RD, True, 15, "middle")
            f.t(x + 30, ZERO + 44, TOK[i], GY2, size=12, anchor="middle")

    # ══════════ 下半：K / V 分家 ═════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 268
    py2 = f.panel(0, y1, W, PH2,
                  "② K 和 V 为什么要分家 ——　因为检索天生是「按 A 找，取回 B」",
                  BL, sub="一张图书馆卡片")

    cy = py2 + 20

    def card(x, w, col, tint, top, l1, l2, l3):
        f.box(x, cy + 30, w, 140, "#fff", col, 10)
        f.box(x, cy + 30, w, 42, tint, "none", 10)
        f.t(x + 18, cy + 58, top, col, True, 18)
        f.t(x + 18, cy + 96, l1, GY, size=15)
        f.t(x + 18, cy + 126, l2, INK, True, 18)
        f.t(x + 18, cy + 156, l3, GY2, size=13)

    f.t(40, cy + 18, "K ≠ V（今天的做法）", BL, True, 20)
    card(40, 300, BL, "#e8f0fe", "书脊（K）：天气",
         "书里写的（V）：", "今天 25 度，多云", "按书脊找，取回内容")
    f.t(368, cy + 100, "→", GY2, True, 28)
    f.box(404, cy + 30, 312, 140, "#fff", GR, 10)
    f.t(422, cy + 62, "问：外面怎么样？", GR, True, 18)
    f.t(422, cy + 100, "跟「天气」像 ✅", GY, size=16)
    f.t(422, cy + 134, "取回：25 度，多云", GR, True, 19)
    f.t(422, cy + 160, "拿到的正是我缺的那块", GY2, size=13)

    f.t(752, cy + 18, "如果 K ＝ V", RD, True, 20)
    card(752, 300, RD, "#fce8e6", "卡片上就是全文",
         "书脊 ＝ 书里：", "今天 25 度，多云", "找它只能靠「像不像它」")
    f.t(1080, cy + 100, "→", GY2, True, 28)
    f.box(1116, cy + 30, 244, 140, "#fff", RD, 10)
    f.t(1134, cy + 62, "问：外面怎么样？", RD, True, 18)
    f.t(1134, cy + 100, "跟这句话不像 ⛔", GY, size=16)
    f.t(1134, cy + 136, "找不到", RD, True, 22)
    f.t(1134, cy + 160, "只找得到像我的", GY2, size=13)

    # ══════════ 一条落点带 ═══════════════════════════════════════
    yy = y1 + PH2 + 20
    yy = f.band(yy, "info", "⭐ 带走一条判据：先问这个分数拿去干什么", [
        "<tspan font-weight=\"700\">拿去加权平均</tspan> ——&#160;"
        "那就必须非负、和为 1、可导，<tspan font-weight=\"700\">softmax 一次全给了</tspan>。",
        "<tspan font-weight=\"700\">只拿去排序</tspan> ——&#160;那「和为 1」根本不必要。"
        "DSA 的索引器走的正是这一条：它<tspan font-weight=\"700\">换成了 ReLU</tspan>，"
        "论文给的理由就两个字 ——&#160;吞吐（见 §6.4b）。",
    ])

    yy = f.src(yy + 14,
               "三条性质由 softmax 的定义直接给出；柱子的数由本脚本当场算并断言",
               "「索引器用 ReLU，for throughput consideration」出自 "
               "DeepSeek-V3.2-Exp 技术报告 §2.1",
               "⚠️ 图书馆卡片是<tspan font-weight=\"700\">类比</tspan>，不是论文原话 "
               "——&#160;论文只给了 query / key / value 三个名字")
    f.save("fig3-why-softmax.svg", yy + 6)


main()
