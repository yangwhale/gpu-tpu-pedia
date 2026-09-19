# -*- coding: utf-8 -*-
r"""专题四 · §1.2「梯度是所有样本的诉求取平均」——&#160;众口难调

⭐⭐⭐ 2026-09-17 新画。现场原话：「挑那种讲的最亲民的、最形象的，
  所以可能跟实际生活中的事情结合起来的维度」。
  ⭐ 这一格是全讲**最口语的一张**：梯度这个词听着吓人，
    但它的本意就一句话 ——&#160;**一屋子人各提各的要求，最后取个平均。**

⭐⭐ 取法来自 **3Blue1Brown**《What is backpropagation really doing?》。
  四个装置都从官方讲义原文核过（⛔ 图是我们自己重画的，例子换成了本讲的下一个字）：
    · **nudge（推一下）** ——&#160;不能直接改输出，只能记下「希望它往哪边动」，
      而且 "the sizes of these nudges should be proportional to
      how far off each output value is from the target"
    · **三条路** ——&#160;想让一个神经元更亮，可以改偏置、改权重、
      或者改上一层的亮度
    · **按上游亮度成比例** ——&#160;"To get the most bang for your buck,
      adjust the weights in proportion to their associated activations"
      （他顺带提了赫布理论 "neurons that fire together wire together"，
      ⚠️ 原文自己也说这个类比并不严格，所以只放进出处不画上图）
    · ⭐⭐⭐ **众口难调** ——&#160;"It's impossible to perfectly satisfy all
      these competing desires"；而且 "If we only listened to what that
      image of a 2 wanted, the network would ultimately be incentivized
      to just classify all images as a 2"

⭐⭐⭐ 而这张图在**本讲**里的位置，是一口气把三处接上（三条都是本讲自己的合题）：
  · 接 §1.2 ——&#160;Ⓐ 那个「该上的推上去、该下的推下去，推多少看差多远」，
    **就是「预测 − 真值」那颗种子**，只是换成了人话。
  · 接 §1.6 ——&#160;Ⓑ「按上游亮度成比例」解释了**为什么反向必须用到前向的值**：
    要知道改哪根线最划算，得先知道那根线上游有多亮，
    **而那个亮度是前向算出来、被存下来的。** 激活扔不掉的根子就在这儿。
  · 接 §1.8 与专题五 ——&#160;Ⓒ 那个「取平均」不是比喻，
    **数据并行里的 all-reduce 干的就是这件事**：
    每张卡算自己那批样本的诉求，汇总起来平均。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400


def main():
    f = Fig(W, "梯度的本意是一屋子人各提各的要求，最后取平均。"
               "一条训练样本看到网络的输出之后，"
               "会希望正确那个字的分数推上去、其他字推下去，"
               "推多少跟差多远成正比 —— 这就是那颗种子，也就是预测减真值。"
               "而想让一个数变大有三条路：改偏置、改权重、让上一层更亮；"
               "改权重时按上游有多亮成比例最划算，"
               "这也正是反向必须用到前向存下来的值的原因。"
               "最后，如果只听一条样本的，网络会把所有输入都判成那一个答案，"
               "所以要把所有样本的诉求平均起来 —— 那个平均就是梯度，"
               "而多卡训练里的汇总，干的就是这个平均")

    y0 = f.header(
        "梯度是什么　——　<tspan font-weight=\"700\">"
        "一屋子人各提各的要求，最后取个平均</tspan>",
        "⭐ 这一格不讲公式，只讲<tspan font-weight=\"700\">这件事在干嘛</tspan>",
        [(BL, "① 一条样本的诉求"), (GR, "② 诉求怎么往回递"), (OR, "③ 众口难调，取平均")])

    # ══════════ Ⓐ 一条样本想要什么 ═══════════════════════════════════
    PH = 356
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一条样本看完输出，<tspan font-weight=\"700\">只会提一个要求</tspan>", BL,
                 sub="⭐ 正确答案是「的」，而网络给「的」只打了 0.3")

    CANDS = (("的", 0.30, 1.0, GR), ("了", 0.20, 0.0, RD),
             ("在", 0.15, 0.0, RD), ("和", 0.10, 0.0, RD))

    # ⛔ 第一版画成「灰条 ＋ 彩条」并排比高矮 ——&#160;可目标值是 0 的那三个
    #   只能画成 3 px 的小横杠，看着像渲染残留。
    #   ⭐⭐ 更要命的是：那种画法比的是**两根条谁高**，
    #     而这一格要讲的是**「差多远」决定「推多少」** ——&#160;
    #     差值才是主角，两根条都只是配角。
    #   ⭐ 判据：**图要把「要讲的那个量」画成最显眼的那个形状。**
    #     这里那个量是差值，所以让**箭头的长度**去承载它，柱子只当参照。
    BX, BY, BARW, BARH = 130, py + 58, 64, 150
    for k, (ch, pred, want, col) in enumerate(CANDS):
        x = BX + k * 150
        top = BY + BARH - BARH * pred
        tgt = BY + BARH - BARH * want
        f.line(x - 6, BY + BARH, x + BARW + 6, BY + BARH, GY2, 1.1, arrow=False)
        f.box(x, top, BARW, BARH * pred, "#f1f3f4", GY2, 4)
        # 目标线：虚线横杠，落在正确答案那个高度
        f.line(x - 10, tgt, x + BARW + 10, tgt, col, 1.4, dash="4 3", arrow=False)
        # ⭐ 箭头长度 ＝ 差距，这才是这一格的主角
        f.line(x + BARW / 2, top, x + BARW / 2, tgt, col, 2.6)
        gap = abs(want - pred)
        f.t(x + BARW + 16, (top + tgt) / 2.0 + 5, "差 %.2f" % gap,
            col, True, 12.5)
        f.t(x + BARW / 2, BY + BARH + 28, ch, INK, True, 19, "middle")
        f.t(x + BARW / 2, BY + BARH + 50, "现在 %.2f" % pred,
            GY2, size=11.5, anchor="middle")

    f.t(BX + 290, BY + BARH + 82,
        "灰柱 ＝ 网络现在给的　｜　虚线 ＝ 正确答案　｜　"
        "<tspan font-weight=\"700\">箭头长度 ＝ 要推多少</tspan>",
        GY2, size=12.5, anchor="middle")

    f.box(760, py + 56, 600, 240, "#e8f0fe", BL, 8)
    f.t(1060, py + 94, "它的要求就一句话", BL, True, 19, "middle")
    f.t(1060, py + 134, "<tspan font-weight=\"700\">「的」推上去，其他推下去</tspan>",
        INK, True, 17, "middle")
    f.t(1060, py + 170, "而且<tspan font-weight=\"700\">推多少，看差多远</tspan>",
        INK, True, 16, "middle")
    f.t(1060, py + 214, "⭐⭐ 这就是本讲说的那颗<tspan font-weight=\"700\">种子</tspan>",
        BL, True, 15.5, "middle")
    f.t(1060, py + 242, "<tspan font-weight=\"700\">「预测 −　真值」</tspan>"
        "　——　换成人话就是这一句", GY, size=13.5, anchor="middle")
    f.t(1060, py + 276, "⛔ 注意它只是「希望」——　没人能直接改输出",
        GY2, size=12.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 三条路，以及哪条最划算 ═══════════════════════════
    PH2 = 366
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 可输出<tspan font-weight=\"700\">改不了</tspan>"
                  "　——　想让一个数变大，只有三条路", GR,
                  sub="⭐ 而第三条走不通，于是它变成了"
                      "<tspan font-weight=\"700\">对上一层的新要求</tspan>")

    WAYS = (
        (GR, "① 改偏置", "直接给它加一点",
         "✅ 最省事，但能调的余地小"),
        (BL, "② 改权重", "把连过来的线加粗",
         "⭐ <tspan font-weight=\"700\">上游越亮的那根线，加粗越划算</tspan>"),
        (PU, "③ 让上一层更亮", "要求前一层把该亮的点亮起来",
         "⛔ 可上一层也改不了 ——　于是这条要求<tspan font-weight=\"700\">往回递一层</tspan>"),
    )
    for i, (col, name, how, verdict) in enumerate(WAYS):
        x = 40 + i * 440
        f.box(x, py2 + 42, 420, 186, "#fff", col, 8, sw=1.6)
        f.box(x, py2 + 42, 420, 4, col, col, 2)
        f.t(x + 210, py2 + 82, name, col, True, 18, "middle")
        f.t(x + 210, py2 + 118, how, INK, True, 15, "middle")
        f.t(x + 210, py2 + 166, verdict, GY, size=13.5, anchor="middle")

    f.t(700, py2 + 268,
        "⭐⭐⭐ 第 ② 条那句「上游越亮越划算」，正是"
        "<tspan font-weight=\"700\">反向为什么必须用到前向存下来的值</tspan>"
        "　——　要知道改哪根线回报最大，", INK, size=14.5, anchor="middle")
    f.t(700, py2 + 294,
        "得先知道那根线的<tspan font-weight=\"700\">上游当时有多亮</tspan>，"
        "而那个亮度是前向算出来的。<tspan font-weight=\"700\">"
        "激活扔不掉，根子就在这一句上。</tspan>", INK, size=14.5, anchor="middle")
    f.t(700, py2 + 322,
        "⭐ 而第 ③ 条「要求往回递一层」，就是"
        "<tspan font-weight=\"700\">反向传播</tspan>这个名字的全部含义。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 众口难调 ═════════════════════════════════════════
    PH3 = 342
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐⭐ 但<tspan font-weight=\"700\">不能只听一个人的</tspan>", OR,
                  sub="⛔ 只听那一条样本的，网络会学会"
                      "<tspan font-weight=\"700\">把什么都答成「的」</tspan>")

    SEATS = (("样本 1", "「的」推上去", GR), ("样本 2", "「了」推上去", BL),
             ("样本 3", "「在」推上去", PU), ("……", "各提各的", GY2))
    for k, (who, ask, col) in enumerate(SEATS):
        x = 60 + k * 230
        f.box(x, py3 + 48, 200, 96, "#fff", col, 8, sw=1.4)
        f.t(x + 100, py3 + 82, who, col, True, 15.5, "middle")
        f.t(x + 100, py3 + 114, ask, GY, size=13, anchor="middle")
        f.line(x + 100, py3 + 150, x + 100, py3 + 178, GY2, 1.2)

    f.t(1090, py3 + 96, "→", GY2, True, 26, "middle")
    f.box(1150, py3 + 48, 200, 96, "#fce8e6", OR, 8)
    f.t(1250, py3 + 82, "全都满足？", OR, True, 16, "middle")
    f.t(1250, py3 + 114, "⛔ 做不到", OR, True, 15, "middle")

    f.box(60, py3 + 186, 1290, 82, "#fef7e0", OR, 8)
    f.t(705, py3 + 222,
        "⭐⭐⭐ 于是<tspan font-weight=\"700\">把所有人的诉求加起来取平均</tspan>"
        "　——　<tspan font-weight=\"700\">那个平均，就是梯度。</tspan>",
        INK, True, 17, "middle")
    f.t(705, py3 + 252,
        "⭐ 而多卡训练里那一道「跨卡汇总」，干的<tspan font-weight=\"700\">就是这个平均</tspan>"
        "　——　每张卡先收自己那批人的意见，再凑到一起。",
        GY, size=13.5, anchor="middle")

    f.t(700, py3 + 298,
        "⚠️ 所以 batch 不是「为了跑得快」才有的　——　"
        "<tspan font-weight=\"700\">它首先是「别只听一个人的」。</tspan>",
        GY, size=13, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "整个反向传播，用人话讲完就是这三步",
                ("✅ <tspan font-weight=\"700\">一条样本提要求 →&#160;要求往回递一层又一层 →&#160;"
                 "所有样本的要求取平均。</tspan>"
                 "后面那些张量、链式法则、矩阵乘，都是这三句话的"
                 "<tspan font-weight=\"700\">算法实现</tspan>，不是另外一件事。",
                 "⛔ 而这一讲真正关心的账，全挂在第二步上："
                 "<tspan font-weight=\"700\">要求往回递的时候，必须回头看前向留下的东西</tspan>"
                 "　——　这就是激活扔不掉、显存下不来的全部原因。",
                 "⭐⭐ 顺带解掉一个常见误解：「batch 开大是为了把卡喂饱」只说对了一半。"
                 "<tspan font-weight=\"700\">它首先是统计上的需要</tspan> ——&#160;"
                 "只听一个人的，网络会被带偏（<tspan font-weight=\"700\">那张 batch 图</tspan>讲的是另一半：什么时候开大才白赚）。"))

    yb = f.src(yb + 16,
               "📌 「推一下（nudge），推多少跟差多远成正比」「想让一个神经元更亮有三条路」"
               "「改权重要按上游亮度成比例，回报最大」「众口难调，只能取平均；"
               "只听那张 2 的，网络会把所有图都判成 2」四个装置，取自 "
               "<tspan font-weight=\"700\">3Blue1Brown</tspan>"
               "《What is backpropagation really doing?》官方讲义 ——&#160;"
               "<tspan font-weight=\"700\">已逐条核过原文，图是我们自己重画的，"
               "例子换成了本讲一直在用的「下一个字」。</tspan>",
               "⚠️ 他在「按上游亮度成比例」那里顺带提了赫布理论"
               "（neurons that fire together wire together），"
               "但<tspan font-weight=\"700\">原文自己就说这个类比并不严格</tspan>"
               "（未训练的网络并没有在「想」那个答案），所以只记在这儿，不画上图。",
               "⭐ Ⓑ 与 §1.6、Ⓒ 与 §1.8 的那两处接头，"
               "<tspan font-weight=\"700\">原文都没有，是本讲自己的合题</tspan> ——&#160;"
               "3B1B 讲的是「反向传播在干嘛」，本讲要的是「它为什么这么费显存」。")

    f.save("fig4-vote.svg", yb + 14)


if __name__ == "__main__":
    main()
