# -*- coding: utf-8 -*-
r"""专题四 · §1.0「那一列 bit 到底在说什么」

⭐⭐⭐ 2026-09-23 现场（原话，一字不改）：
  「它里边的逻辑是这样的：我需要多少个比特能把这个 token **选出来**，不是记下来。
    就是为什么一开始困惑度高的时候，我需要更多的比特来把它选出来 ——&#160;
    因为它的候选就那么多。然后为什么后来我就需要更少的比特？这是有上下文的，
    不是它自己，是通过这个大模型的前向，最后把候选缩到 7 个，
    这个时候它才可以用 3 个比特把它表达出来。这个解释非常重要。」

⛔ 那一列原来只有一个换算系数（nat × 1.4427），**一个字都没说它是什么意思**。
  ⭐ 而现场自己把意思讲出来了，而且分成了三层 ——&#160;这张图就按那三层画：
    Ⓐ bit ＝ **选出来**要问几个是非问题（不是「存下来要几个字节」）
    Ⓑ 候选是**上下文**缩小的，不是这个词自己变简单了 ——&#160;
       那 14 个 bit 就是模型干的活
    Ⓒ 「等效」这个词要说准：词表还是 129,280 个，只是不均匀

⛔ 刻意没画的：熵的定义、编码定理。这一格要的是「这个数怎么读」，不是「它怎么证」。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2

W = 1400

VOCAB = 129280                      # DeepSeek-V3 词表
LOSS_HI = math.log(VOCAB)           # 刚初始化：均匀瞎猜
BIT_HI = math.log2(VOCAB)
LOSS_LO = 2.0                       # 现代大模型预训练末期的量级
PPL_LO = math.exp(LOSS_LO)
BIT_LO = math.log2(PPL_LO)
SAVED = BIT_HI - BIT_LO             # 模型省下来的那些 bit
RATIO = VOCAB / PPL_LO              # 候选缩小了多少倍

assert abs(LOSS_HI - 11.77) < 0.01 and abs(BIT_HI - 16.98) < 0.01
assert abs(BIT_LO - 2.89) < 0.01 and abs(SAVED - 14.09) < 0.01
# ⭐ 这一条是全图的落点，钉死：省下的 bit 数就是候选缩小倍数的对数
assert abs(2 ** SAVED - RATIO) < 1.0, (2 ** SAVED, RATIO)
assert 17000 < RATIO < 18000, RATIO


def main():
    f = Fig(W, "比特数回答的是「把这个词从候选里选出来，要问几个是非问题」，"
               "不是「把它存下来要几个字节」。每问一次，候选减半，"
               "所以问的次数就是候选个数的以二为底的对数。"
               "刚初始化的时候候选是整个词表十二万九千二百八十个，要问将近十七次；"
               "训练到后期只要不到三次。"
               "而候选变少不是因为这个词自己变简单了，是因为它前面那段上下文 —— "
               "大模型前向做的事，就是把十二万九千多个候选压到等效七个。"
               "省下来的那十四个比特，就是模型干的全部活。"
               "最后要说准：困惑度七不是说真的只剩七个候选，词表还是那么大，"
               "它说的是这份不均匀的分布跟在七个等概率候选里挑一样难")

    y0 = f.header(
        "那一列 <tspan font-weight=\"700\">bit</tspan> 在说什么"
        "　——　<tspan font-weight=\"700\">把它「选出来」要问几个是非问题</tspan>",
        "⛔ 不是「存下来要几个字节」——　"
        "<tspan font-weight=\"700\">是从候选里把它挑出来，要问几次「是 / 不是」</tspan>",
        [(BL, "每问一次，候选减半")])

    # ══════════ Ⓐ 一个 bit ＝ 一个是非问题 ═══════════════════════
    PH = 420
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一个 bit ＝ <tspan font-weight=\"700\">一个是非问题</tspan>"
                 "　——　每问一次，候选<tspan font-weight=\"700\">减半</tspan>", BL,
                 sub="⭐ 所以「要问几次」＝ 候选个数取以 2 为底的对数")

    def ladder(x0, ytop, n0, col, tag, rungs, tail, bits):
        f.t(x0, ytop - 16, tag, col, True, 15.5)
        wmax = 430.0
        yy = ytop + 10
        for i, (label, cnt) in enumerate(rungs):
            w = wmax / (2.0 ** i)
            f.box(x0, yy, w, 20, "#e8f0fe" if col is BL else "#fef7e0", col, 3)
            f.t(x0 + w + 12, yy + 15, "还剩 %s 个" % format(cnt, ","),
                GY, size=12)
            f.t(x0 - 8, yy + 15, label, GY2, size=11.5, anchor="end")
            yy += 28
        f.t(x0 + 8, yy + 16, tail, GY2, size=12.5)
        f.t(x0, yy + 48, bits, col, True, 16)

    ladder(150, py + 70, VOCAB, BL, "① 刚初始化：候选是<tspan font-weight=\"700\">"
           "整个词表</tspan>",
           [("开局", VOCAB), ("问 1 次", VOCAB // 2), ("问 2 次", VOCAB // 4),
            ("问 3 次", VOCAB // 8)],
           "…… 这样一直问下去，第 17 次才剩 1 个",
           "⇒ 要问 <tspan font-weight=\"700\">16.98</tspan> 次　"
           "（log₂ %s）" % format(VOCAB, ","))

    ladder(810, py + 70, 7, OR, "② 训练到后期：候选只剩<tspan font-weight=\"700\">"
           "等效 7 个</tspan>",
           [("开局", 7), ("问 1 次", 4), ("问 2 次", 2), ("问 3 次", 1)],
           "三次就到头了",
           "⇒ 要问 <tspan font-weight=\"700\">2.89</tspan> 次　（log₂ 7.39）")

    f.t(700, py + 348,
        "<tspan font-weight=\"700\">bit 数量的是「挑出来有多难」，不是「存起来有多大」。</tspan>"
        "　——　一个 token 的 id 拿 4 个字节就存下了，跟这一列没有关系。",
        INK, size=14.5, anchor="middle")
    f._pan = None

    yb = f.band(py + PH + 18, "ok", "⭐ 一句话记住这张图", [
        "<tspan font-weight=\"700\">bit ＝ 把它从候选里选出来要问几个是非问题</tspan>"
        "　——　候选越少，问得越少。",
        "<tspan font-weight=\"700\">而候选是上下文缩小的，不是这个词自己变简单了</tspan>"
        "　——　16.98 降到 2.89，那 14.09 个 bit 就是模型前向干的活。",
    ])

    yb = f.src(yb + 10,
               "换算里<tspan font-weight=\"700\">不含任何实测值</tspan>："
               "困惑度 ＝ e^loss，bit ＝ log₂(困惑度) ＝ loss ÷ ln2；"
               "词表 %s 取自 DeepSeek-V3 配置。" % format(VOCAB, ","),
               "⚠️ 「后期 loss 2.0 上下」是公开模型的大致量级，"
               "<tspan font-weight=\"700\">当参照看，别当某个具体模型的实测值</tspan>。")

    f.save("fig4-bits.svg", yb + 14)


# ══════════════════════════════════════════════════════════════════
# 第二张：候选是谁缩小的 ＋「等效」的意思。
# ⛔ 2026-09-23 现场：「这几个图前前后后都有类似的图，没有重要的意义，
#   全都给我折叠起来。」——&#160;⭐ 查了一遍：**这两格讲的东西，
#   课件在图的正上方已经用文字写全了**（「候选变少不是这个词自己变简单了，
#   是它前面那段上下文」「『等效 7 个』不是说真的只剩 7 个候选」）。
#   ⭐⭐ 判据：**新加一张图的时候，顺手看一眼正文是不是已经把它说完了。**
#     图和文哪个先有不重要 ——&#160;重要的是别让读者读两遍。
#   ⚠️ 没删：它把那段文字画成了画面，想看的人展开。
#     ⛔ 折叠这件事要求它先是一个独立产物，所以从主图里拆出来。
# ══════════════════════════════════════════════════════════════════
def more():
    g = Fig(W, "候选变少不是因为这个词自己变简单了，是因为它前面那段上下文："
               "大模型前向把十二万九千多个候选压到等效七个，"
               "十六点九八减二点八九等于十四点零九个比特，"
               "二的十四点零九次方正好是一万七千五百倍，那就是模型干的全部活。"
               "另外要说准：困惑度七不是说真的只剩七个候选，词表还是十二万九千多个，"
               "它说的是这份不均匀的分布跟在七个等概率候选里挑一样难")

    y0c = g.header(
        "接着上一张：<tspan font-weight=\"700\">候选是谁缩小的</tspan>，"
        "以及「等效」是什么意思",
        "⭐ 这两格的话，课件在图上面已经写过一遍 ——　"
        "<tspan font-weight=\"700\">这里是它的画面版</tspan>",
        [(OR, "Ⓐ 谁把候选缩小的"), (GR, "Ⓑ 「等效」的意思")])

    # ══════════ Ⓑ 候选是谁缩小的 ═════════════════════════════════
    PH2 = 396
    pa = g.panel(0, y0c, W, PH2,
                  "Ⓐ ⭐⭐⭐ 候选是<tspan font-weight=\"700\">上下文</tspan>缩小的"
                  "　——　<tspan font-weight=\"700\">不是这个词自己变简单了</tspan>", OR,
                  sub="⛔ 这是全图最要紧的一格：那 14 个 bit 是<tspan "
                      "font-weight=\"700\">模型省下来的</tspan>")

    BARW = 620
    for r, (lab, alive, col, note) in enumerate((
            ("没有上下文　（刚初始化）", 1.0, GY2,
             "整个词表都有份 ——　<tspan font-weight=\"700\">谁都可能</tspan>"),
            ("有上下文　「…… 今天天气真」", 0.02, OR,
             "绝大多数词的概率被压到几乎为 0　——　"
             "<tspan font-weight=\"700\">只剩一小撮还在场</tspan>"))):
        yy = pa + 76 + r * 110
        g.t(80, yy - 12, lab, INK, True, 15)
        g.box(80, yy, BARW, 40, "#f1f3f4", GY2, 4)
        if alive < 1.0:
            g.box(80, yy, BARW * alive, 40, "#fef7e0", OR, 4, sw=1.8)
        g.t(80 + BARW + 16, yy + 26, note, GY, size=13)
        g.t(80, yy + 66,
            ("候选 <tspan font-weight=\"700\">%s</tspan> 个　⇒　"
             "<tspan font-weight=\"700\">16.98 bit</tspan>" % format(VOCAB, ","))
            if r == 0 else
            ("等效候选 <tspan font-weight=\"700\">7</tspan> 个　⇒　"
             "<tspan font-weight=\"700\">2.89 bit</tspan>"),
            col if r else GY, True, 14.5)

    g.box(80, pa + 300, 1240, 62, "#fff8e1", OR, 8)
    g.t(104, pa + 326,
        "⭐⭐ <tspan font-weight=\"700\">中间那一步，就是大模型的前向。</tspan>"
        "　它把 %s 个候选压成<tspan font-weight=\"700\">等效 %.1f 个</tspan>"
        "　——　缩小了约 <tspan font-weight=\"700\">%s 倍</tspan>。"
        % (format(VOCAB, ","), PPL_LO, format(int(round(RATIO)), ",")),
        INK, size=14.5)
    g.t(104, pa + 350,
        "<tspan font-weight=\"700\">16.98 − 2.89 ＝ %.2f 个 bit</tspan>"
        "　——　这 %.0f 个 bit 就是模型干的全部活。"
        "<tspan fill=\"%s\">（而 2 的 %.2f 次方，正好就是那 %s 倍。）</tspan>"
        % (SAVED, SAVED, GY2, SAVED, format(int(round(RATIO)), ",")),
        INK, size=14.5)
    g._pan = None

    # ══════════ Ⓒ 「等效」这个词要说准 ═══════════════════════════
    PH3 = 356
    pb = g.panel(0, y0c + PH2 + 20, W, PH3,
                  "Ⓑ 「<tspan font-weight=\"700\">等效</tspan> 7 个」是什么意思"
                  "　——　<tspan font-weight=\"700\">词表并没有变小</tspan>", GR,
                  sub="⛔ 不是「真的只剩 7 个候选」——　是「跟 7 个等概率候选一样难猜」")

    REAL = (0.50, 0.18, 0.18, 0.07, 0.07)
    BASE = pb + 190                      # 柱子的基线
    bx = 150
    g.t(bx + 110, pb + 62, "真实分布　（%s 根柱子）" % format(VOCAB, ","),
        INK, True, 14.5, "middle")
    g.t(bx + 110, pb + 86, "高低极不均匀", GY, size=12.5, anchor="middle")
    for k, v in enumerate(REAL):
        h = 88 * v / 0.5
        g.box(bx + k * 48, BASE - h, 34, h, "#fef7e0" if k == 0 else "#f1f3f4",
              OR if k == 0 else GY2, 3)
    g.t(bx + 130, BASE + 26, "…… 其余 %s 个的概率几乎是 0"
        % format(VOCAB - len(REAL), ","), GY2, size=12)

    # ⛔ 这里原来用的是 ⇄ ——&#160;渲染环境缺这个字形，出来是一个空框。
    #   ⭐ 判据：**装饰性符号一律画出来，不要靠字体里有没有。**
    g.line(668, BASE - 48, 622, BASE - 48, GY, 2.2)
    g.line(732, BASE - 30, 778, BASE - 30, GY, 2.2)
    g.t(700, BASE + 2, "一样难猜", GY, True, 14, "middle")

    cx = 880
    g.t(cx + 155, pb + 62, "等概率的 7 个", INK, True, 14.5, "middle")
    g.t(cx + 155, pb + 86, "每个都是 1/7", GY, size=12.5, anchor="middle")
    for k in range(7):
        g.box(cx + k * 48, BASE - 56, 34, 56, "#e6f4ea", GR, 3)
    g.t(cx + 155, BASE + 26, "这就是「困惑度 ＝ 7」那句话的全部含义",
        GR, True, 13, "middle")

    g.t(700, pb + 268,
        "⭐ 所以困惑度<tspan font-weight=\"700\">不是候选的个数</tspan>，"
        "是「<tspan font-weight=\"700\">相当于在几个词之间犹豫</tspan>」。"
        "词表一直是 %s 个，变的只是概率有多集中。" % format(VOCAB, ","),
        INK, size=14.5, anchor="middle")
    g._pan = None


    yb = g.src(y0c + PH2 + 20 + PH3 + 26,
               "⭐ 这两格原本是上一张图的 Ⓑ Ⓒ，2026-09-23 拆出来"
               "——&#160;<tspan font-weight=\"700\">为的是让课件那边能把它折叠起来</tspan>。",
               "换算里<tspan font-weight=\"700\">不含任何实测值</tspan>："
               "困惑度 ＝ e^loss，bit ＝ log₂(困惑度)；词表 %s 取自 DeepSeek-V3 配置。"
               % format(VOCAB, ","))

    g.save("fig4-bits-more.svg", yb + 14)


main()
more()
