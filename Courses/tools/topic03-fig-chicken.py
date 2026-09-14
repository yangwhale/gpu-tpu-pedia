# -*- coding: utf-8 -*-
r"""专题三 · §六「破鸡生蛋的三条路」

⭐⭐⭐ 2026-09-13 新画。调研发现本课只讲了**一种**破法（DSA 的师徒），
   而实际上有**三条真正不同的思路**，而且第三条正好是 V4 相对 V3.2 的新东西 ——
   我们现在把 CSA 讲成「压缩的两个方向」（记会议纪要），
   ⛔ **没把它跟鸡生蛋挂上钩**，可 CSA 最聪明的地方恰恰在这里。

鸡生蛋是什么：**要挑出重要的 key，得先算注意力；可算完了再挑就没意义了。**

  ① **师徒**（DSA）——&nbsp;真注意力当老师，训一个便宜的学生去学它的排序
  ② **一份算两用**（NSA）——&nbsp;不训第二个打分器。压缩分支的 softmax 分数
     本来就要算，**直接拿它当路由信号**
  ③ **降维打击**（CSA / V4）——&nbsp;先把序列压短 4 倍，
     **indexer 的搜索空间跟着缩 4 倍** ——&nbsp;鸡生蛋没破，但那只鸡小了 4 倍

📌 还有一条把 (a) 和「为什么必须整块取」串起来的链子（zhouyifan 的表述）：
   相似度算出来了再稀疏就没好处 →&nbsp;必须有个**便宜的近似打分器** →&nbsp;
   便宜的近似打分只能**按块**做 →&nbsp;所以整块取不是妥协，是 (a) 的直接推论。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    f = Fig(W, "破鸡生蛋的三条路：师徒让真注意力当老师教一个便宜的索引器；"
               "一份算两用不训第二个打分器，压缩分支的分数直接当路由；"
               "降维打击先把序列压短四倍，搜索空间跟着缩四倍")
    f.marks = set()
    y0 = f.header(
        "先说清这个死结：<tspan font-weight=\"700\">"
        "要挑出重要的，得先算注意力；可算完了再挑，就没意义了</tspan>",
        "⭐ 本课原来只讲了一种破法。其实有三条，而且是<tspan font-weight=\"700\">"
        "真正不同的三条</tspan>",
        [(RD, "死结"), (BL, "师徒"), (OR, "一份算两用"), (PU, "降维打击")])

    # ══════════ ① 死结本身 ══════════════════════════════════════
    PH = 244
    py = f.panel(0, y0, W, PH, "① 死结长什么样", RD, sub="先把它画出来")
    ay = py + 26
    f.box(96, ay + 20, 300, 92, "#fce8e6", RD, 10)
    f.t(246, ay + 56, "想只算重要的那几块", RD, True, 21, "middle")
    f.t(246, ay + 88, "→ 得先知道哪几块重要", GY, size=17, anchor="middle")
    f.elbow(400, ay + 66, 560, ay + 66, RD, 2.2, r=1)
    f.box(560, ay + 20, 300, 92, "#fce8e6", RD, 10)
    f.t(710, ay + 56, "想知道哪几块重要", RD, True, 21, "middle")
    f.t(710, ay + 88, "→ 得先把注意力算一遍", GY, size=17, anchor="middle")
    # 回环箭头：从右框底部绕回左框底部
    f.path("M 710 %d C 710 %d 246 %d 246 %d" % (ay + 118, ay + 168, ay + 168,
                                                ay + 118), RD, 2.2)
    f.t(478, ay + 178, "转回来了", RD, True, 19, "middle")
    f.box(900, ay + 20, 444, 92, "#fff", INK, 10)
    f.t(922, ay + 52, "⭐ 所以三条破法都在回答同一句话：", INK, True, 17)
    f.t(922, ay + 82, "「怎么在<tspan font-weight=\"700\">不算全</tspan>的前提下，"
        "知道该看谁」", GY, size=17)

    # ══════════ ② 三条路 ════════════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 396
    py2 = f.panel(0, y1, W, PH2, "② 三条破法 ——　它们是真正不同的三条", GR,
                  sub="不是同一招的三种说法")

    CW, GAP = 436, 24
    by = py2 + 24
    WAYS = [
        (BL, "① 师徒", "DSA", "teacher",
         "让真注意力当老师", "训一个便宜的学生去学它的排序",
         ["⚠️ 代价：要专门训一个索引器",
          "（V3.2 花了 943.7B token）"]),
        (OR, "② 一份算两用", "NSA", "dual",
         "不训第二个打分器", "压缩分支的分数本来就要算，直接拿它当路由",
         ["⭐ 白捡：top-k 在前向图上是个 no-op，",
          "只决定从显存搬哪些块"]),
        (PU, "③ 降维打击", "CSA · V4", "shrink",
         "先把序列压短 4 倍", "鸡生蛋没破 ——　但那只鸡小了 4 倍",
         ["⭐ 这是 V4 相对 V3.2", "真正的新东西"]),
    ]
    for i, (col, name, who, kind, a, b_, note) in enumerate(WAYS):
        x = 20 + i * (CW + GAP)
        f.box(x, by, CW, 300, "#fff", col, 10)
        f.box(x, by, CW, 5, col, col, 3)
        f.box(x, by + 3, CW, 5, "#fff", "#fff", 0)
        f.t(x + 22, by + 44, name, col, True, 25)
        f.t(x + CW - 22, by + 44, who, GY2, size=17, anchor="end")

        gx, gy = x + 22, by + 62
        if kind == "teacher":            # 老师看全部 → 学生抄一份名次表
            f.icon("person", gx, gy + 10, 44, 50, col, "#e8f0fe")
            f.t(gx + 22, gy + 76, "老师", col, True, 16, "middle")
            for k in range(6):           # 老师看全部
                f.box(gx + 62 + k * 16, gy + 14, 12, 42, "#e8f0fe", col, 2)
            f.elbow(gx + 168, gy + 35, gx + 214, gy + 35, col, 2.0, r=1)
            f.icon("person", gx + 224, gy + 10, 44, 50, GY2, "#f1f3f4")
            f.t(gx + 246, gy + 76, "学生", GY, True, 16, "middle")
            f.icon("paper", gx + 286, gy + 10, 40, 50, GY2)
            f.t(gx + 306, gy + 76, "只有名次表", GY2, size=15,
               anchor="middle")
        elif kind == "dual":             # 一份分数，两个出口
            f.box(gx, gy + 18, 96, 40, "#fef7e0", col, 6)
            f.t(gx + 48, gy + 44, "压缩分支", col, True, 16, "middle")
            f.t(gx + 48, gy + 78, "本来就要算", GY2, size=15, anchor="middle")
            f.elbow(gx + 100, gy + 32, gx + 170, gy + 14, col, 2.0, r=8, via="h")
            f.elbow(gx + 100, gy + 44, gx + 170, gy + 62, col, 2.0, r=8, via="h")
            f.t(gx + 178, gy + 20, "① 当输出用", GY, size=16)
            f.t(gx + 178, gy + 68, "② 当路由信号用", col, True, 16)
        else:                            # 序列压短 → 搜索空间跟着缩
            for k in range(12):
                f.box(gx + k * 15, gy + 16, 11, 26, BG2, LINE2, 2)
            f.t(gx, gy + 62, "12 格要挑", GY2, size=15)
            f.elbow(gx + 186, gy + 29, gx + 224, gy + 29, col, 2.0, r=1)
            for k in range(3):
                f.box(gx + 234 + k * 26, gy + 16, 22, 26, "#f3e8fd", col, 3)
            f.t(gx + 234, gy + 62, "只剩 3 格", col, True, 16)

        f.t(x + 22, by + 190, a, col, True, 20, w=CW - 44)
        f.t(x + 22, by + 218, b_, GY, size=17, w=CW - 44)
        f.box(x + 22, by + 236, CW - 44, 54, BG2, LINE2, 8)
        for k, ln in enumerate(note):
            f.t(x + 36, by + 262 + k * 22, ln, GY, size=16, w=CW - 72)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y1 + PH2 + 20
    yy = f.band(yy, "info", "⭐⭐ 顺着这条死结往下走一步，「为什么必须整块取」就不用单独讲了", [
        "把链子接起来：<tspan font-weight=\"700\">相似度一旦算出来，事后再稀疏就没好处</tspan>"
        " ——&#160;所以<tspan font-weight=\"700\">必须有一个便宜的近似打分器</tspan>；"
        "而便宜的近似打分<tspan font-weight=\"700\">只能按块做</tspan>（按 token 打分就等于把全表算了）。",
        "⭐ 于是<tspan font-weight=\"700\">「整块取」不是对硬件的妥协，是这个死结的直接推论</tspan>。"
        "本课原来把它们讲成两段互不相干的话 ——&#160;接上之后，两个难点变成一个。",
    ])
    yy = f.src(yy + 16,
               "① DSA 的师徒（两阶段训练、冻主模型 ＋ KL warmup）出自 "
               "DeepSeek-V3.2-Exp 技术报告 sec. 2；② NSA 的三支路出自 arXiv 2502.11089 sec. 3",
               "③ CSA 每 4 个 token 压成 1 个 entry、在压缩后的格上挑，"
               "出自 DeepSeek-V4 相关公开材料（见 CSA 那张图的出处）",
               "⚠️「相似度算出来再稀疏就没好处 →&#160;只能按块打分」这条链子的表述"
               "出自 zhouyifan.net 的 Log-linear Sparse Attention 一文；"
               "「师徒 / 一份算两用 / 降维打击」是<tspan font-weight=\"700\">本课的命名</tspan>")
    f.save("fig3-chicken.svg", yy + 6)


main()
