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
TGT_N = 4095        # 一条长 4,096 的序列只有 4,095 个「下一个字」


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

    # ══════════ Ⓑ 上游越亮，加粗越划算 ═════════════════════════════
    # ⛔⛔ 2026-09-22 现场看着这一格说：「这一块内容感觉没啥用。」
    #   ⭐ 判断：**他是对的，但病因不是「这件事不重要」** ——&#160;
    #     原来这一格是**三块并排的写字板**（改偏置 / 改权重 / 让上一层更亮），
    #     三个圆角框里塞文字，**一点几何都没有**，读者只能读，不能看。
    #     而且其中两条后来被更好的图讲过了（往回递一层 → fig-reverse、
    #     权重怎么改 → fig-settle Ⓐ）。
    #   ⭐⭐ 但里面**有一句是承重的**：「上游越亮的那根线，加粗越划算」——&#160;
    #     整本显存账（激活为什么扔不掉）就挂在它上面。**所以是重画，不是删。**
    #   ⇒ 三块板子扔掉，只留这一件事，而且**画出来让人自己看**：
    #     同样粗一点点，接在亮的那根上回报大二十倍。
    # 📌 三条路的出处（3Blue1Brown）没丢，降成底下一行字。
    X_HI, X_LO = 4.0, 0.2          # 两条输入线各自的上游亮度
    DW = 0.1                       # 同样加粗这么多
    G_HI, G_LO = DW * X_HI, DW * X_LO
    assert abs(G_HI / G_LO - X_HI / X_LO) < 1e-9, \
        "回报之比必须等于亮度之比 ——&#160;这一格的全部论点就是这一条"

    PH2 = 372
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 一根权重的<tspan font-weight=\"700\">梯度</tspan>，"
                  "正比于<tspan font-weight=\"700\">上游有多亮</tspan>", GR,
                  sub="⭐ 两根线<tspan font-weight=\"700\">加粗同样多</tspan>，"
                      "回报差二十倍 ——　而「回报有多大」就是「这根权重的梯度有多大」")

    OY, CX, OX = py2 + 60, 190, 760
    for k, (xv, col, lab) in enumerate(((X_HI, OR, "很亮"), (X_LO, GY2, "很暗"))):
        cy = OY + 46 + k * 128
        r = 30
        f.box(CX - r, cy - r, 2 * r, 2 * r, "#fef7e0" if k == 0 else "#f1f3f4",
              col, r, sw=2.0)
        f.t(CX, cy + 6, "%.1f" % xv, col, True, 17, "middle")
        # ⛔ 这两个标签原来放在圆的**下面**，第二个圆的标签正好撞进
        #   底下那两行落点文字里（渲染出来才看见）。⭐ 挪到圆的左边。
        f.t(CX - r - 12, cy + 5, "上游<tspan font-weight=\"700\">%s</tspan>" % lab,
            col, size=13, anchor="end")
        # 线：两根一样细，各加粗同样多（虚线部分＝加粗的那一点）
        f.line(CX + r + 6, cy, OX - 34, OY + 110, col, 2.0, arrow=False)
        f.t((CX + OX) / 2 - 10, cy + (OY + 110 - cy) / 2 - 10 + (0 if k else -8),
            "＋%.1f 粗" % DW, col, True, 13, "middle")

    f.box(OX - 30, OY + 76, 60, 68, "#e8f0fe", BL, 8, sw=2.0)
    f.t(OX, OY + 116, "输出", BL, True, 14, "middle")

    for k, (g, col) in enumerate(((G_HI, OR), (G_LO, GY2))):
        yy = OY + 60 + k * 76
        f.line(OX + 40, yy + 14, OX + 74, yy + 14, col, 2.0)
        f.box(OX + 82, yy - 10, 250, 48, "#fef7e0" if k == 0 else "#f1f3f4",
              col, 6, sw=1.4)
        f.t(OX + 207, yy + 22, "输出涨 <tspan font-weight=\"700\">＋%.2f</tspan>" % g,
            col if k == 0 else GY, True, 17, "middle")

    f.box(OX + 360, OY + 46, 270, 104, "#e6f4ea", GR, 8)
    f.t(OX + 495, OY + 84, "同样的力气", GR, True, 15, "middle")
    f.t(OX + 495, OY + 118, "回报差 <tspan font-weight=\"700\">%d 倍</tspan>"
        % int(round(G_HI / G_LO)), INK, True, 20, "middle")
    f.t(OX + 495, OY + 146, "——　正好是亮度之比", GY, size=12.5, anchor="middle")

    # ⛔⛔ 2026-09-23 现场：「这个图说啥呢？真的有用吗？难道他说的是梯度吗？
    #   这个东西怎么跟上下文结合起来？」——&#160;⭐ **说的就是梯度**，
    #   可这一格从头到尾**一次都没提过「梯度」两个字**，它说的是「改哪根线最划算」。
    #   ⛔ 判据：**一格图讲的是某个术语时，就得把那个术语说出来。**
    #     换成大白话是为了好懂，可**连名字都换掉，读者就接不回上下文了** ——&#160;
    #     他不知道该把这一格挂在哪儿。
    #   ⚠️ 顺带删掉原来那条「三条路：改偏置 / 改权重 / 让上一层更亮，这格只讲 ②」——&#160;
    #     它又开了一根轴，而这一格本来就已经在解释一件不容易的事了。
    f.t(700, py2 + 262,
        "⭐ 说的就是<tspan font-weight=\"700\">梯度</tspan>："
        "<tspan font-weight=\"700\">一根权重的梯度 ＝ 上游那一刻的亮度 × 下游传来的责任。</tspan>"
        "　这一格画的是<tspan font-weight=\"700\">前面那个乘数</tspan>。",
        INK, size=15, anchor="middle")
    f.t(700, py2 + 296,
        "⭐⭐⭐ 所以「<tspan font-weight=\"700\">改哪根线最划算</tspan>」这个问题，"
        "答案<tspan font-weight=\"700\">不在反向里，在前向里</tspan>　——　"
        "它由那一刻的<tspan font-weight=\"700\">亮度</tspan>决定。",
        INK, size=15, anchor="middle")
    f.t(700, py2 + 326,
        "而那个亮度，是<tspan font-weight=\"700\">前向算出来、必须被存下来</tspan>的。"
        "<tspan font-weight=\"700\">整本显存账的根子，就在这一句上。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 四千多股力，加成一股 ═══════════════════════════
    # ⛔⛔ 2026-09-22 现场：「样本一、样本二 OK，但是小方块里带那个字，
    #   让人容易误解成 —— 其实那几个字是我们举那个单独 token 的例子时候用的……
    #   但是这个图并不是，这个图的意思是我要有不同的 token 的诉求，
    #   然后 4095 个 token 的诉求，再加上 batch 乘到一起的诉求，
    #   然后把它们对于权重调整的这个合力加到一起。」
    # ⭐ 完全对，而且这是**两个层级被画混了**：
    #     Ⓐ 讲的是「**一个位置内部**，词表上那一排」——&#160;那里才有「的 / 了 / 在」；
    #     Ⓒ 讲的是「**位置与位置之间**」——&#160;这里一格是一个位置，不是一个字。
    #   ⛔ 原图在 Ⓒ 的方块里写「『的』推上去」，等于把 Ⓐ 的例子搬到了 Ⓒ 的坐标系里。
    # ⭐⭐ 重画：每个位置的诉求画成**一个箭头**（方向和长度各不相同），
    #   把它们**真的加起来**，让台下自己看见「合力比各自短得多」——&#160;
    #   那就是「互相抵消」，也就是 batch 越大越稳的全部原因。
    import math as _m
    # 八股示意力（角度°，长度）。⛔ 写死，不用随机 ——&#160;构建要可复现。
    FORCES = ((18, 1.00), (74, 0.85), (-46, 0.95), (131, 0.70),
              (-100, 0.90), (36, 0.80), (168, 0.75), (-14, 0.88))
    _sx = sum(L * _m.cos(_m.radians(a)) for a, L in FORCES)
    _sy = sum(L * _m.sin(_m.radians(a)) for a, L in FORCES)
    _R = _m.hypot(_sx, _sy)
    _SUMLEN = sum(L for _, L in FORCES)
    assert _R < 0.5 * _SUMLEN, \
        "合力必须明显短于各分量长度之和 ——&#160;这一格的论点就是「互相抵消」"
    _ang = _m.degrees(_m.atan2(_sy, _sx))

    PH3 = 416
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ <tspan font-weight=\"700\">每一个位置都提自己的一份</tspan>"
                  "　——　几千股力，加成一股", OR,
                  sub="⛔ 注意这一格<tspan font-weight=\"700\">一格 ＝ 一个位置</tspan>，"
                      "不是一个字　——　字那一层在 Ⓐ，这一层是"
                      "<tspan font-weight=\"700\">位置与位置之间</tspan>")

    # 左：两条序列，各自一排箭头
    for r in range(2):
        cy = py3 + 96 + r * 116
        col = (BL, PU)[r]
        f.t(66, cy + 5, "序列 %d" % (r + 1), col, True, 14)
        for k, (a, L) in enumerate(FORCES):
            ox = 170 + k * 62
            aa = a + (0 if r == 0 else 24)          # 两条序列的诉求也不一样
            dx = 26 * L * _m.cos(_m.radians(aa))
            dy = -26 * L * _m.sin(_m.radians(aa))
            f.line(ox, cy, ox + dx, cy + dy, col, 2.0)
            if r == 0 and k < 2:
                f.t(ox, cy + 40, "位置 %d" % (k + 1), GY2, size=11, anchor="middle")
        f.t(170 + 8 * 62 + 4, cy + 5, "……　共 %s 个位置" % format(TGT_N, ","),
            GY2, size=12)

    f.t(430, py3 + 292,
        "⭐ 每一支箭头 ＝ <tspan font-weight=\"700\">一个位置对这块权重的诉求</tspan>"
        "（往哪改、改多少）", INK, size=14, anchor="middle")
    f.t(430, py3 + 318,
        "⛔ 它们<tspan font-weight=\"700\">方向各不相同</tspan>　——　这才是"
        "「众口难调」四个字的实际样子。", GY, size=13.5, anchor="middle")

    # 中：求和
    f.t(880, py3 + 150, "⊕", OR, True, 34, "middle")
    f.t(880, py3 + 190, "全部加起来", OR, True, 13, "middle")
    f.t(880, py3 + 214, "（batch＝2 就是 %s 支）" % format(TGT_N * 2, ","),
        GY2, size=11.5, anchor="middle")

    # 右：合力
    RX, RY = 1120, py3 + 150
    for a, L in FORCES:                       # 淡淡地把分量叠在原点，当参照
        f.line(RX, RY, RX + 26 * L * _m.cos(_m.radians(a)),
               RY - 26 * L * _m.sin(_m.radians(a)), "#dadce0", 1.2, arrow=False)
    f.line(RX, RY, RX + 26 * _R * _m.cos(_m.radians(_ang)),
           RY - 26 * _R * _m.sin(_m.radians(_ang)), OR, 3.4)
    f.t(RX + 130, RY - 46, "<tspan font-weight=\"700\">合力</tspan>", OR, True, 16,
        "middle")
    f.t(RX + 130, RY - 18,
        "各自加起来 %.1f，" % _SUMLEN, GY, size=13, anchor="middle")
    f.t(RX + 130, RY + 8,
        "合力只有 <tspan font-weight=\"700\">%.2f</tspan>" % _R,
        INK, True, 15, "middle")
    f.t(RX + 130, RY + 40,
        "⭐ <tspan font-weight=\"700\">大部分互相抵消了</tspan>", OR, True, 13.5,
        "middle")
    f.t(RX + 130, RY + 66,
        "剩下的那一点，才是真信号", GY2, size=12.5, anchor="middle")

    f.t(700, py3 + 356,
        "⭐⭐⭐ <tspan font-weight=\"700\">这股合力，就是这块权重这一步要挪的方向。</tspan>"
        "　而「抵消得多、剩得少」正是"
        "<tspan font-weight=\"700\">batch 越大越稳</tspan>的全部原因　——　",
        INK, size=14.5, anchor="middle")
    f.t(700, py3 + 384,
        "人越多，偏激的那几支越淹得住。"
        "⭐ 多卡训练里那道「跨卡汇总」，干的就是这个加法："
        "每张卡先加自己那批，再凑到一起。",
        GY, size=13.5, anchor="middle")
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
