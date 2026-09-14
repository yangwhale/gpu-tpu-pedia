# -*- coding: utf-8 -*-
r"""专题三 · §2.3b「信息账的严格版本」

⭐⭐⭐ 2026-09-13 **整张重画**，换成一个人人都有过的画面：
   **一本越读越厚的书，和一个固定大小的笔记本。**

  ① **两个互信息，别混为一谈** ——&nbsp;用同一本书讲：
     · **隔得越远的两句话，关系越弱** ——&nbsp;但**弱得很慢**（幂律）。
       ⚠️ 如果只靠「记住上一句」（马尔可夫），忘得是**断崖式**的（指数）。
       ⭐ 两条线画在 log-log 上，**幂律是一条直线，指数一头栽下去** ——&nbsp;
       换成普通坐标，这个区别根本看不出来。
     · **把书从中间劈开，前半和后半之间的关联，随书变厚而增加** ——&nbsp;
       不矛盾：**单对越来越弱，可成对的数量越来越多。**
  ② **L2M 条件 ＝ 你的笔记本得跟着书一起变厚。**
     Transformer 的 KV 随长度线性长 →&nbsp;笔记本自动跟着厚（代价是平方计算）。
  ③ ⚠️ **三个旋钮在这把尺子下各自站在哪** ——&nbsp;这一格是**本课的推导**。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    f = Fig(W, "信息账的严格版本：一本越读越厚的书和一个固定大小的笔记本 —— "
               "隔得越远的两句话关系越弱但弱得很慢（幂律），只记上一句会断崖式忘掉"
               "（指数）；把书劈开两半的关联随书变厚而增加；笔记本得跟着一起变厚")
    f.marks = set()
    y0 = f.header(
        "「远处可以少看」这句直觉，严格版本长什么样",
        "用<tspan font-weight=\"700\">一本越读越厚的书</tspan>讲 ——&#160;"
        "以及<tspan font-weight=\"700\">一个固定大小的笔记本</tspan>",
        [(BL, "两句话之间"), (GR, "两半之间"), (PU, "笔记本得多大"),
         (OR, "三个旋钮站在哪")])

    # ══════════ ① 两句话之间：幂律 vs 指数 ══════════════════════
    PH = 340
    py = f.panel(0, y0, W, PH, "① 隔得越远的两句话，关系越弱 ——　但弱得有多快",
                 BL, sub="这是 log-log 坐标，换成普通坐标就看不出区别了")

    ay = py + 22
    X0, X1 = 120, 700
    Y0, Y1 = ay + 224, ay + 30
    VMIN = 1e-3

    def px(d):
        return X0 + (X1 - X0) * math.log10(d) / 3.0

    def py_(v):
        v = max(v, VMIN)
        return Y0 + (Y1 - Y0) * (math.log10(v) + 3.0) / 3.0

    f.line(X0, Y0, X1 + 20, Y0, LINE, 1.4, arrow=False)
    f.line(X0, Y0, X0, Y1 - 10, LINE, 1.4, arrow=False)
    f.t(X0 - 12, Y1 + 4, "强", GY2, size=15, anchor="end")
    f.t(X0 - 12, Y0 + 4, "弱", GY2, size=15, anchor="end")
    f.t(X1 + 26, Y0 + 6, "隔得越远 →", GY2, size=15)

    for lab, fn, col in (("真实的语言：慢慢变弱", lambda d: d ** -0.55, BL),
                         ("只记上一句：断崖式忘掉",
                          lambda d: math.exp(-(d - 1) / 40.0), RD)):
        pts = [(px(10 ** (3.0 * k / 60.0)), py_(fn(10 ** (3.0 * k / 60.0))))
               for k in range(61)]
        f.path(pts, col, 2.2)
    f.t(px(160), py_(160 ** -0.55) - 16, "真实的语言", BL, True, 19)
    f.t(px(160), py_(160 ** -0.55) + 8, "隔一百句还剩一点", BL, size=15)
    # ⭐ 标签钉在**两条线已经分开**的位置：d<100 时红线还在蓝线上面，
    #   标在那儿会让人以为「只记上一句」记得更牢。
    f.t(px(300), py_(math.exp(-299 / 40.0)) - 40, "只记上一句", RD, True, 19)
    f.t(px(300), py_(math.exp(-299 / 40.0)) - 16, "隔几十句就归零", RD, size=15)

    f.box(760, ay + 26, 600, 200, "#e8f0fe", BL, 10)
    f.t(784, ay + 66, "⭐⭐ 这就是语言不能用「只记上一句」近似的原因", BL,
        True, 17, w=552)
    f.t(784, ay + 108, "任何<tspan font-weight=\"700\">有限状态</tspan>的记忆方式，", GY,
        size=18)
    f.t(784, ay + 140, "衰减都是<tspan font-weight=\"700\">指数</tspan>的 ——&#160;"
        "说没就没；", GY, size=18)
    f.t(784, ay + 172, "而真实语言是<tspan font-weight=\"700\">幂律</tspan>的 ——&#160;"
        "一直有一点。", GY, size=17)
    f.t(784, ay + 210, "⚠️ 「有限状态」这个限定不能省", GY2, size=15)
    f.t(120, ay + 262, "⭐ 注意这是 <tspan font-weight=\"700\">log-log 坐标</tspan>"
        " ——&#160;换成普通坐标，两条线都长成「往下掉的一条线」，"
        "这张图要说的区别就看不见了。", GY, size=17)

    # ══════════ ② 把书劈成两半 ══════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 316
    py2 = f.panel(0, y1, W, PH2, "② 可是把书从中间劈开，两半之间的关联反而在变大",
                  GR, sub="不矛盾 ——　单对越来越弱，成对的数量越来越多")

    by = py2 + 24
    for i, (n, lab) in enumerate([(3, "薄书"), (6, "厚一点"), (12, "很厚")]):
        bx = 96 + i * 300
        f.t(bx, by + 24, lab, GR, True, 21)
        for k in range(n):
            f.box(bx + (k % 6) * 22, by + 36 + (k // 6) * 40, 16, 34,
                  "#e6f4ea", GR, 3)
        f.t(bx + 150, by + 60, "↔", GR, True, 26, "middle")
        for k in range(n):
            f.box(bx + 176 + (k % 6) * 22, by + 36 + (k // 6) * 40, 16, 34,
                  "#e6f4ea", GR, 3)
        f.t(bx, by + 140, "%d × %d ＝ %d 对" % (n, n, n * n), GY, size=17)
    f.box(1016, by + 20, 344, 140, "#e6f4ea", GR, 10)
    f.t(1040, by + 58, "⭐ 一个在减，一个在增", GR, True, 21)
    f.t(1040, by + 94, "单对的关联越来越弱，", GY, size=17)
    f.t(1040, by + 126, "可对数增长得更快。", GY, size=17)

    f.box(96, by + 178, 1264, 90, "#fff", PU, 10)
    f.t(120, by + 216, "⭐⭐ 于是 L2M 那条定理，说人话就是这一句：", PU, True, 17)
    f.t(120, by + 252, "<tspan font-weight=\"700\">你记笔记的那个本子，"
        "得跟着书一起变厚</tspan> ——&#160;本子大小固定，总有一本书是它兜不住的。",
        GY, size=17)

    # ══════════ ③ 三个旋钮站在哪 ════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 280
    py3 = f.panel(0, y2, W, PH3, "③ 那三个旋钮，各自站在哪一边",
                  OR, sub="⚠️ 这一格是本课按定义做的推导")

    ey = py3 + 22
    KN = [
        (GR, "旋钮①　每份更小", "本子还是跟着书一起变厚", "只是每页写得更省", "安全"),
        (GR, "旋钮②　挑着看", "本子照样跟着变厚", "只是不是每页都翻", "安全"),
        (RD, "旋钮③　固定状态", "⛔ 本子大小<tspan font-weight=\"700\">写死了</tspan>",
         "书再厚，本子不变", "总有一本兜不住"),
    ]
    for i, (col, name, a, b, verdict) in enumerate(KN):
        bx = 56 + i * 442
        f.box(bx, ey + 24, 400, 160, "#fff", col, 10)
        f.t(bx + 22, ey + 64, name, col, True, 22)
        f.t(bx + 22, ey + 102, a, GY, size=17, w=356)
        f.t(bx + 22, ey + 132, b, GY2, size=16, w=356)
        f.t(bx + 22, ey + 170, verdict, col, True, 20)
    f.t(56, ey + 218, "⚠️ 纯滑窗跟纯线性<tspan font-weight=\"700\">是同一类</tspan>"
        " ——&#160;它们动的都是<tspan font-weight=\"700\">阶</tspan>，不是常数。",
        RD, True, 17)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这一张真正的用处：它把一句直觉变成了一条能证伪的规律", [
        "「远处可以少看」<tspan font-weight=\"700\">不等于</tspan>「远处不重要」——&#160;"
        "两半之间的关联是<tspan font-weight=\"700\">随长度增长</tspan>的，只是增长得慢。",
        "⭐ 于是整件事很清楚：<tspan font-weight=\"700\">Transformer 的 KV 线性增长是"
        "「供给过量」</tspan>，而这条规律是「实际需求」。"
        "前两个旋钮就是在<tspan font-weight=\"700\">不掉到需求线以下的前提下，"
        "把那个过量的常数压小</tspan>。",
        "⭐⭐ 这条分类有一个白捡的验证：44 行表里<tspan font-weight=\"700\">"
        "滑窗混合落在 1:1～6:1，线性混合落在 3:1～7:1</tspan>，两族重叠在 3:1～6:1 ——&#160;"
        "同一条规律一口气解释了两类看起来毫不相干的混合。",
    ])

    yy = f.band(yy + 14, "warn", "口径四条，一条都不能省", [
        "⚠️ L2M 是<tspan font-weight=\"700\">必要条件，不是充分条件</tspan>："
        "本子够厚只是「有可能记住」，不等于「真的学会了」。",
        "⚠️ 它是<tspan font-weight=\"700\">渐近</tspan>命题（「总存在一本书」）——&#160;"
        "同一张表里 Mistral 7B 就是纯滑窗、真的上过两年生产。"
        "<tspan font-weight=\"700\">别说成「所有 SWA 模型都混」。</tspan>",
        "⛔⛔ 最容易讲过头的一条：<tspan font-weight=\"700\">定理要求的是"
        "「本子变厚」，不是「必须混着用」</tspan>。论文自己给了另一条出路 ——&#160;"
        "<tspan font-weight=\"700\">按书的厚度整个换一本更大的本子</tspan>"
        "（model series，Def 5.5 / Thm 5.6）。"
        "⭐ 混合是工程上选的那条，不是定理逼出来的那条。",
        "⚠️ 第三格是<tspan font-weight=\"700\">本课按定义做的推导</tspan>；"
        "尤其旋钮② 那行 ——&#160;论文<tspan font-weight=\"700\">明确把稀疏注意力排除</tspan>"
        "在那段分析之外（原文括注 excluding sparse attention），只说「可以照样分析」。",
    ], fold=True)

    yy = f.src(yy + 16,
               "① 出自 Lin ＆ Tegmark《Criticality in Formal Languages and "
               "Statistical Physics》（Entropy 2017, arXiv 1606.06737）",
               "⚠️ arXiv v1 的旧题名是《Critical Behavior in Physics and Probabilistic "
               "Formal Languages》，两者是同一篇 ——&#160;标了正式出处就用正式题名",
               "原文口径：互信息在任何<tspan font-weight=\"700\">概率正则文法</tspan>下"
               "指数衰减，而<tspan font-weight=\"700\">上下文无关文法</tspan>下可以是幂律",
               "② 出自 L2M（arXiv 2503.04725, ICML 2025）：双部互信息幂律 scaling、"
               "「状态维度必须至少同阶增长」的定理、以及 Transformer 自动满足的那段分析",
               "⚠️ 「书 / 笔记本」是<tspan font-weight=\"700\">本课的比喻</tspan>；"
               "③ 为本课推导，非论文结论")
    f.save("fig3-info-law.svg", yy + 6)


main()
