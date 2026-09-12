# -*- coding: utf-8 -*-
r"""专题三 · §一「为什么是 softmax，为什么 K 和 V 要分家」（2026-09-13 夜间 · R8）。

⭐⭐ 这张图回答两个「为什么不那样做」——&nbsp;它们都很少有人讲，
   而且各自能串起这一讲后面的一段。

  ① **为什么是 softmax** ——&nbsp;关键在于它要服务的是**加权平均**：
     权重必须**非负**（否则「负着看一眼」没有意义）、必须**和为 1**
     （否则输出的尺度随序列长度乱飘）、还必须**可导**（否则学不了）。
     指数还顺带放大了差距，让它像一个「软的 argmax」。

  ② ⭐⭐⭐ **但 softmax 不是唯一解 ——&nbsp;要看你拿它干什么。**
     §六 里 DSA 的那个索引器，用的就是 **ReLU 不是 softmax**，
     而且论文说得很直白：**为了吞吐**。
     为什么它敢换？——&nbsp;**因为索引器只需要排序，不需要加权平均。**
     不做加权平均，「和为 1」这条就不必要了。
     ⭐ 判据：**先问这个分数是拿去加权，还是只拿去排序 ——&nbsp;
     两者对归一化的要求根本不一样。**

  ③ **为什么 K 和 V 要分家** ——&nbsp;因为检索天然是**非对称**的：
     你**按 A 去找**，但要**取回 B**。
     K＝V 的话，只有「跟我像的」才会被取回来；
     可你要的经常不是「像我的」，而是「能补上我缺的那块的」。

📌 ① 的性质是 softmax 的定义直接给的；③ 是推导 ＋ 类比，图里标了口径。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "为什么是 softmax：因为它要服务加权平均，权重得非负、和为一、可导；"
               "但索引器只排序不加权，所以 DSA 敢换成 ReLU；"
               "以及为什么 K 和 V 要分家：检索是非对称的，按 A 找取回 B")
    f.marks = set()
    y0 = f.header(
        "两个「为什么不那样做」　——　softmax 与 K/V 分家",
        "⭐ 这两条都很少有人讲，"
        "而且<tspan font-weight=\"700\">各自能串起后面一整段</tspan>",
        [(GR, "softmax 必须满足的"), (OR, "换掉它的条件"),
         (BL, "检索的非对称"), (PU, "回扣后面的小节")])

    ph = 434

    # ══ ① softmax 要服务的是加权平均 ════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 为什么是 softmax", GR,
                 sub="先看它要服务什么")

    yy = py + 24
    f.box(x + 22, yy, pw - 44, 50, "#fff", INK, 8)
    f.t(x + 38, yy + 21, "注意力最后那一步是<tspan font-weight=\"700\">加权平均</tspan>", INK, True, 12.5)
    f.t(x + 38, yy + 40, "o ＝ Σ 权重 × value", GY, size=11.5, mono=True)
    yy += 64

    f.t(x + 22, yy, "于是权重被逼出三条硬要求：", GY, size=12)
    yy += 20
    for need, why, col in [
        ("非负", "「负着看一眼」没有意义", GR),
        ("加起来等于 1", "否则输出尺度随序列长度乱飘", GR),
        ("可导", "否则「该看哪儿」学不出来", GR),
    ]:
        f.box(x + 22, yy, pw - 44, 46, "#fff", col, 8)
        f.t(x + 38, yy + 20, need, col, True, 12)
        f.t(x + 38 + 96, yy + 20, why, GY, size=11.5, w=pw - 180)
        yy += 54

    yy += 4
    f.box(x + 22, yy, pw - 44, 78, "#fff", GR, 8)
    f.box(x + 22, yy, 4, 78, GR, GR, 2)
    f.box(x + 24, yy, 3, 78, "#fff", "#fff", 0)
    f.t(x + 40, yy + 24, "softmax 三条全中，还附送一条：", GR, True, 12.5)
    f.t(x + 40, yy + 46, "<tspan font-weight=\"700\">指数把差距放大</tspan> ——&#160;它像一个", GY, size=11.5)
    f.t(x + 40, yy + 66, "<tspan font-weight=\"700\">能求导的 argmax</tspan>：既能聚焦，又留着梯度", GY,
        size=11.5)
    fits(yy + 78, y0, ph, "①")

    # ══ ② 那为什么 DSA 敢换成 ReLU ══════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 那 DSA 为什么敢换成 ReLU", OR,
                 sub="⭐ 这一格串起 §六")

    yy = py + 24
    f.box(x + 22, yy, pw - 44, 56, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 56, OR, OR, 2)
    f.box(x + 24, yy, 3, 56, "#fff", "#fff", 0)
    f.t(x + 40, yy + 23, "§六 里 DSA 的索引器打分式子：", OR, True, 12.5)
    f.t(x + 40, yy + 43, "I(t,s) ＝ Σ w · <tspan font-weight=\"700\">ReLU</tspan>(q · k)", GY, size=12,
        mono=True)
    yy += 70

    f.t(x + 22, yy, "那它凭什么不用 softmax？", INK, True, 13,
        cls="svglbl")
    yy += 26
    for who, what, need, col in [
        ("主注意力", "要拿这些分数去<tspan font-weight=\"700\">加权平均</tspan>", "→ 三条全要", GR),
        ("索引器", "只拿这些分数去<tspan font-weight=\"700\">排个序</tspan>，挑 top-k", "→ 和为 1 不必要", OR),
    ]:
        f.box(x + 22, yy, pw - 44, 76, "#fff", col, 8)
        f.box(x + 22, yy, 4, 76, col, col, 2)
        f.box(x + 24, yy, 3, 76, "#fff", "#fff", 0)
        f.t(x + 40, yy + 24, who, col, True, 12.5)
        f.t(x + 40, yy + 46, what, GY, size=11.5, w=pw - 76)
        f.t(x + 40, yy + 66, need, col, True, 11.5)
        yy += 86

    yy += 2
    f.box(x + 22, yy, pw - 44, 96, "#fff", INK, 8)
    f.t(x + 38, yy + 25, "⭐⭐ 判据，记这一句就够：", INK, True, 13,
        cls="svglbl")
    f.t(x + 38, yy + 49, "<tspan font-weight=\"700\">这个分数是拿去加权，还是只拿去排序？</tspan>", INK,
        True, 12.5, w=pw - 76)
    f.t(x + 38, yy + 71, "两者对归一化的要求根本不一样 ——", GY, size=11.5)
    f.t(x + 38, yy + 89, "只排序的那个，可以换成便宜得多的东西。", GY,
        size=11.5)
    fits(yy + 96, y0, ph, "②")

    # ══ ③ 为什么 K 和 V 要分家 ══════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 为什么 K 和 V 要分家", BL,
                 sub="因为检索天生是非对称的")

    yy = py + 24
    f.box(x + 22, yy, pw - 44, 74, "#fff", BL, 8)
    f.box(x + 22, yy, 4, 74, BL, BL, 2)
    f.box(x + 24, yy, 3, 74, "#fff", "#fff", 0)
    f.t(x + 40, yy + 24, "k 是书脊上印的字，v 是书里的内容", BL, True, 12.5)
    f.t(x + 40, yy + 46, "你<tspan font-weight=\"700\">照着书脊找</tspan>，", GY, size=11.5)
    f.t(x + 40, yy + 66, "但<tspan font-weight=\"700\">拿走的是书里的东西</tspan>。", GY, size=11.5)
    yy += 88

    f.t(x + 22, yy, "那如果让 k ＝ v 会怎样？", INK, True, 13,
        cls="svglbl")
    yy += 26
    f.box(x + 22, yy, pw - 44, 96, "#fff", RD, 8)
    f.box(x + 22, yy, 4, 96, RD, RD, 2)
    f.box(x + 24, yy, 3, 96, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "那就只有「<tspan font-weight=\"700\">跟我像的</tspan>」会被取回来。", RD,
        True, 12.5)
    f.t(x + 40, yy + 48, "可你要的经常<tspan font-weight=\"700\">不是像你的那个</tspan>，", GY, size=11.5)
    f.t(x + 40, yy + 68, "是<tspan font-weight=\"700\">能补上你缺的那块的</tspan>那个 ——", GY, size=11.5)
    f.t(x + 40, yy + 88, "代词要找的是它指代的名词，不是别的代词。", GY2,
        size=11)
    yy += 110

    f.box(x + 22, yy, pw - 44, 76, "#fff", INK, 8)
    f.t(x + 38, yy + 25, "⭐ 一句话：注意力要的是", INK, True, 12.5)
    f.t(x + 38, yy + 48, "<tspan font-weight=\"700\">按 A 去找，取回 B</tspan> —— 一次非对称的检索。",
        INK, True, 12.5, w=pw - 76)
    f.t(x + 38, yy + 68, "📌 这一格是推导 ＋ 类比，不是论文原话", GY2,
        size=11)
    fits(yy + 76, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 优点和副作用，常常是同一条性质", [
        "softmax 那条「<tspan font-weight=\"700\">一行加起来等于 1</tspan>」，"
        "在这里是<tspan font-weight=\"700\">优点</tspan>（让加权平均的尺度稳定）；"
        "到了 §六，同一条性质就是 <tspan font-weight=\"700\">attention sink 的成因</tspan>"
        "——&#160;没什么想看的时候，多余的权重也得倒在某处。",
        "⭐ 所以那个 softmax-off-by-one 的补丁，本质就是<tspan font-weight=\"700\">"
        "把这一条放松掉</tspan>（分母 +1，允许一行加起来小于 1）。",
        "⛔ 判据：<tspan font-weight=\"700\">看到一个设计的副作用，先回头看它的优点是靠哪条性质换来的"
        "</tspan> ——&#160;十有八九是同一条。",
    ])

    yy = f.src(yy + 16,
               "① 三条要求是 softmax 与加权平均的定义直接给的；"
               "「指数放大差距」「软的 argmax」是通行说法",
               "② DSA 索引器用 ReLU 及其理由（为吞吐）出自 DeepSeek-V3.2-Exp 技术报告 §1；"
               "⚠️「因为只排序不加权所以敢换」是本课的解释，报告只给了「为吞吐」这个理由",
               "③ 为推导 ＋ 类比，非论文原话")
    f.save("fig3-why-softmax.svg", yy + 6)


main()
