# -*- coding: utf-8 -*-
r"""专题三 · §2.6「幼儿版：KV cache 到底是在哪一步出生的」

⭐⭐⭐ 2026-09-16 新画。现场原话：
  「这个小结里画的那个图是 Scaling Book 里的**专家版**，信息量太大了，
    我怕同学们理解不了，咱们能不能画一个**幼儿版**的，简单点。」

⛔ 那张专家版（fig3-tx-base）**没有删，折起来了** ——&nbsp;它是全专题的主线图，
  后面好几章都要把它重画一遍、点亮被改动的那一处。删了后面接不上。
  ⭐ 判据（本页第三次用到）：**看不懂不等于没价值 ——&#160;收起来，不是删掉。**

⭐⭐ 这张图的取舍，就一条：**一个字母都不准出现。**
  没有 BTD、没有 BSKH、没有 reshape、没有 softmax、没有矩阵形状。
  ⛔ 理由不是「简化」，是**这一节根本不需要它们**：
    2.6 只要送出一句话 ——&nbsp;**这一层算完，只有 K 和 V 必须留到下一个字，
    而且每来一个字就多一格。** 其余全是噪音。

⭐ 三格的分工（刻意做成一句话一格，讲的时候一格一停）：
  Ⓐ 一个字进来，先变出三样东西，然后「拿问题去比钥匙、按比例拌内容」
  Ⓑ 算完之后桌上分三堆 ——&nbsp;**这一格是从专家版那里继承来的，它本来就是全图最好懂的一格**
  Ⓒ 于是那一堆越堆越高 ——&nbsp;KV cache 就是这么出生的

⛔⛔ 刻意没画的：
  ① **多头**。这一节不需要它，画了就得解释「头」是什么，立刻回到专家版。
  ② **N² 与 N³ 那笔账**。它在正文里写着（判据⑩：挨着图的那一处赢），
     图上只留「越堆越高」这个画面。
  ③ **残差 / norm / MLP**。它们一个字都不改这本书要讲的东西。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400


def main():
    f = Fig(W, "幼儿版：一个字进来先变出问题、钥匙、内容三样东西；"
               "拿自己的问题去跟所有人的钥匙比一比，谁像就多看谁，"
               "再按这个比例把大家的内容拌到一起。这一层算完，桌上分三堆："
               "权重是常驻设备，中间量是草稿纸算完就扔，"
               "只有钥匙和内容要锁进柜子留到下一个字 —— "
               "而每来一个字就多一格，这就是 KV cache 的出生")

    y0 = f.header(
        "<tspan font-weight=\"700\">KV cache 是在哪一步出生的</tspan>"
        "　——　一个字母都不用认的版本",
        "⭐ 这一格只送出一句话：<tspan font-weight=\"700\">"
        "这一层算完，只有两样东西必须留到下一个字</tspan>",
        [(BL, "问题"), (PU, "钥匙 · 内容"), (RD, "会一直变长")])

    # ══════════ Ⓐ 一个字进来，发生了什么 ══════════════════════════
    PH = 396
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一个字进来 ——　<tspan font-weight=\"700\">"
                 "先变出三样东西，再拿它们互相问一遍</tspan>", BL,
                 sub="⛔ 这一格一个记号都没有，看画就行")

    # 一个字 → 三样东西
    f.box(64, py + 78, 116, 64, "#f1f3f4", GY2, 8)
    f.t(122, py + 116, "一个字", INK, True, 18, "middle")
    f.t(122, py + 158, "比如「猫」", GY, size=13, anchor="middle")

    ROLES = (
        ("问题", "我想找什么", BL, "#e8f0fe"),
        ("钥匙", "我能回答什么", PU, "#f3e8fd"),
        ("内容", "找到我，我给什么", PU, "#faf5ff"),
    )
    for i, (nm, desc, col, fill) in enumerate(ROLES):
        y = py + 40 + i * 74
        f.line(184, py + 110, 246, y + 30, GY2, 1.4)
        f.box(246, y, 232, 60, fill, col, 8)
        f.t(362, y + 28, nm, col, True, 18, "middle")
        f.t(362, y + 48, desc, GY, size=12.5, anchor="middle")

    f.t(362, py + 290, "⭐ 三样都是<tspan font-weight=\"700\">同一个字变出来的</tspan>"
                       " ——　各管各的用途", GY, size=14, anchor="middle")

    # 右半：怎么用
    f.box(534, py + 30, 810, 300, "#fff", BL, 8)
    f.box(534, py + 30, 810, 4, BL, BL, 2)
    f.t(939, py + 62, "然后这一步只做两个动作", BL, True, 18, "middle")

    f.box(566, py + 88, 746, 92, "#e8f0fe", BL, 6)
    f.t(590, py + 120, "① 拿<tspan font-weight=\"700\">我的问题</tspan>，"
                       "去跟<tspan font-weight=\"700\">前面每一个字的钥匙</tspan>比一比",
        INK, True, 17)
    f.t(590, py + 152, "——　<tspan font-weight=\"700\">谁的钥匙跟我的问题越像，"
                       "我就越多看谁一眼</tspan>", GY, size=14.5)

    f.box(566, py + 196, 746, 92, "#f3e8fd", PU, 6)
    f.t(590, py + 228, "② 按刚才比出来的比例，"
                       "把<tspan font-weight=\"700\">大家的内容拌到一起</tspan>",
        INK, True, 17)
    f.t(590, py + 260, "——　拌出来的这一份，就是<tspan font-weight=\"700\">"
                       "这个字这一步的输出</tspan>", GY, size=14.5)

    f.t(939, py + 314, "⛔ 注意：<tspan font-weight=\"700\">①要用到前面每一个字的钥匙，"
                       "②要用到前面每一个字的内容</tspan>",
        RD, True, 14.5, "middle")
    f._pan = None

    # ══════════ Ⓑ 算完之后桌上分三堆 ═════════════════════════════
    PH2 = 296
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ 这一层算完，<tspan font-weight=\"700\">桌上的东西分三堆</tspan>"
                  " ——　只有一堆必须留着", BL,
                  sub="⭐ 判断标准只有一个：<tspan font-weight=\"700\">"
                      "下一个字还用不用得上它</tspan>")

    PILES = (
        (BL, "#f1f3f4", "常驻的设备", "模型的权重",
         ("所有字共用同一套", "不随对话变", "⭐ 留 ——　但它本来就在那儿")),
        (GR, "#e6f4ea", "草稿纸", "中间算出来的那些量",
         ("比出来的那些「像不像」", "拌完就没用了", "⭐ 扔 ——　算完就扔")),
        (RD, "#fce8e6", "锁进柜子的", "钥匙 和 内容",
         ("下一个字还要拿它们比、拿它们拌", "所以必须留着", "⛔ 留 ——　而且每来一个字就多一格")),
    )
    for i, (col, fill, nm, what, lines) in enumerate(PILES):
        x = 56 + i * 436
        f.box(x, py2 + 30, 412, 220, fill, col, 8)
        f.box(x, py2 + 30, 412, 4, col, col, 2)
        f.t(x + 206, py2 + 64, nm, col, True, 19, "middle")
        f.t(x + 206, py2 + 92, what, INK, True, 16, "middle")
        for k, ln in enumerate(lines[:2]):
            f.t(x + 206, py2 + 126 + k * 26, ln, GY, size=14, anchor="middle")
        f.t(x + 206, py2 + 214, lines[2], col, True, 15, "middle")
    f._pan = None

    # ══════════ Ⓒ 于是那一堆越堆越高 ═════════════════════════════
    PH3 = 320
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3,
                  "Ⓒ 于是那第三堆<tspan font-weight=\"700\">越堆越高</tspan>"
                  " ——　<tspan font-weight=\"700\">KV cache 就是这么出生的</tspan>", RD,
                  sub="⭐ 别的两堆都不随对话变，<tspan font-weight=\"700\">"
                      "只有这一堆一直长</tspan>")

    WORDS = ("今", "天", "天", "气", "真", "好")
    BX, BY = 120, py3 + 66
    for i, w in enumerate(WORDS):
        x = BX + i * 168
        f.t(x + 56, BY - 14, "第 %d 个字" % (i + 1), GY, size=12.5, anchor="middle")
        f.box(x, BY, 112, 40, "#f1f3f4", GY2, 6)
        f.t(x + 56, BY + 27, w, INK, True, 19, "middle")
        f.line(x + 56, BY + 44, x + 56, BY + 62, GY2, 1.4)
        f.box(x + 6, BY + 62, 48, 54, "#f3e8fd", PU, 5)
        f.t(x + 30, BY + 86, "钥", PU, True, 14, "middle")
        f.t(x + 30, BY + 106, "匙", PU, True, 14, "middle")
        f.box(x + 58, BY + 62, 48, 54, "#faf5ff", PU, 5)
        f.t(x + 82, BY + 95, "内容", PU, True, 13, "middle")

    f.box(BX - 16, BY + 56, 1028, 66, "none", RD, 8, 2.0)
    f.t(BX - 16, BY + 146, "⛔ 这一整条<tspan font-weight=\"700\">就是 KV cache</tspan>"
                           " ——　每来一个字，右边就多一格，<tspan font-weight=\"700\">"
                           "而且一格都不能扔</tspan>（下一个字还要用）", RD, True, 17)
    f.t(BX - 16, BY + 176, "⭐ 一万个字的对话，这里就是一万格；十万个字，就是十万格。"
                           "<tspan font-weight=\"700\">它跟着对话一直长下去。</tspan>",
        GY, size=15)
    f.t(BX - 16, BY + 206, "⭐⭐ 而第一堆（权重）从头到尾一样大 ——　"
                           "<tspan font-weight=\"700\">所以对话越长，账单里越是这一堆说了算。</tspan>",
        INK, True, 15.5)
    f._pan = None

    yy = f.band(py3 + PH3 + 22, "ok", "所以这一章的落点就一句话", [
        "<tspan font-weight=\"700\">1990 年那个盒子，换成了一条会一直变长的格子。</tspan>"
        "⭐ 盒子的好处是<tspan font-weight=\"700\">永远那么大</tspan>，坏处是<tspan font-weight=\"700\">只能一个一个来</tspan>；"
        "格子正好反过来 ——&#160;<tspan font-weight=\"700\">能一次算完，但会一直长。</tspan>",
        "⛔ <tspan font-weight=\"700\">后面七章，全是在想办法拿这条格子开刀</tspan> ——&#160;"
        "有人让每一格变小，有人让每步少看几格，"
        "还有人干脆想把它换回那个盒子。",
    ])

    yy = f.src(yy + 24,
               "⛔ <tspan font-weight=\"700\">这张图刻意不画多头、不画残差与 MLP、"
               "不写任何张量形状</tspan> ——&#160;"
               "完整的那张（沿用 How to Scale Your Model 的记号）就在本节里折着，"
               "点开即可；后面几章还会把它重画并点亮被改动的那一处",
               "⚠️ Ⓐ 里「问题 / 钥匙 / 内容」是 Q / K / V 的中文说法，"
               "本课从头到尾用这一套；⛔ <tspan font-weight=\"700\">「拌到一起」"
               "指的是按权重加权求和</tspan>，不是把内容搅混 ——&#160;"
               "这一点跟第一章那个「叠上去不是搅匀」是同一件事",
               "⚠️ Ⓒ 里那六个字只是<tspan font-weight=\"700\">画面</tspan>，"
               "真实的一格里装的是两条向量，不是两个汉字")
    f.save("fig3-kv-born.svg", yy + 6)


main()
