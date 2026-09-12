# -*- coding: utf-8 -*-
r"""专题三 · §二「信息账的严格版本」（2026-09-13 夜间 · R9）。

⭐⭐⭐ 这一张可能是整个专题三**理论上最重要**的一张：
   它把「远处的信息可以少看」这句直觉，换成了一条**能算、能证伪**的规律，
   而且这条规律**顺手回答了「为什么第三个旋钮必须混着用」**。

  ① **两个互信息，别混为一谈**
     · 传统的**两点互信息**：两个 token 之间的相关性，随距离**幂律衰减**。
       Lin & Tegmark 证明：任何马尔可夫/隐马尔可夫过程都是**指数**衰减，
       而实测自然语言是**幂律** ——&nbsp;这正是语言不能用马尔可夫近似的原因。
     · L2M 的**双部互信息**：把序列从中间切开，**两半之间**的互信息，
       随长度**幂律增长**。
     ⭐ 一个在衰减、一个在增长，并不矛盾：
       **单对的相关性越来越弱，但成对的数量越来越多。**

  ② **L2M 条件** ——&nbsp;模型能表达的互信息**受限于它历史状态的维度**，
     所以要有效建模长上下文，**状态维度必须至少以同样的幂律增长**。
     Transformer 的 KV 随长度线性增长 → **自动满足**（代价是平方计算）。

  ③ ⚠️ **三个旋钮在这把尺子下各自站在哪** ——&nbsp;这一格是**本课的推导**：
     旋钮①②动的是**常数**，阶还是 L，安全；
     旋钮③把阶降到 O(1)，**必然在某个长度上违反** ——&nbsp;
     所以它只能混着用。这就是 §八 存在的理由。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "信息账的严格版本：两点互信息随距离幂律衰减，双部互信息随长度"
               "幂律增长；L2M 条件要求模型的历史状态维度至少同阶增长；"
               "据此三个旋钮里只有第三个必然违反，所以它只能混着用")
    f.marks = set()
    y0 = f.header(
        "信息账的严格版本　——　「远处可以少看」这句话，能不能算出来",
        "⭐⭐ 这一张<tspan font-weight=\"700\">顺手回答了「为什么第三个旋钮必须混着用」</tspan>"
        "——&#160;那不是工程经验，是有理论下界的",
        [(BL, "两点互信息：衰减"), (GR, "双部互信息：增长"),
         (PU, "L2M 条件"), (OR, "本课的推导")])

    ph = 520

    # ══ ① 两个互信息 ════════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 两个互信息，别混为一谈", BL,
                 sub="一个在衰减，一个在增长")

    yy = py + 22
    # ⭐ 2026-09-13：新手读者在这张图上放弃了 —— 先给两句大白话，再上术语。
    f.box(x + 22, yy, pw - 44, 72, "#fff", INK, 8)
    f.t(x + 38, yy + 22, "先把词翻译一下：", INK, True, 12.5)
    f.t(x + 38, yy + 43, "「互信息」＝ 知道了 A 之后，对 B 的猜测能准多少。",
        GY, size=11.5, w=pw - 76)
    f.t(x + 38, yy + 62, "「幂律 / 指数」＝ 掉得慢 / 掉得快。", GY, size=11.5)
    yy += 86

    f.t(x + 22, yy, "两点互信息：两个 token 之间", BL, True, 12.5)
    yy += 12
    bx, bw, bh = x + 22, pw - 44, 72
    f.box(bx, yy, bw, bh, "#fff", LINE, 6)
    pts = []
    for i in range(40):
        d = 1 + i * 0.9
        v = d ** -0.55
        pts.append("%.1f,%.1f" % (bx + 10 + i * (bw - 20) / 39.0,
                                  yy + bh - 10 - v * (bh - 22)))
    f.path("M " + " L ".join(pts), BL, 1.8, arrow=False)
    f.t(bx + bw - 10, yy + bh - 14, "距离 →", GY2, size=11, anchor="end")
    yy += bh + 18
    f.t(x + 22, yy, "⭐ 随距离<tspan font-weight=\"700\">幂律衰减</tspan> —— 而任何马尔可夫过程", GY,
        size=11.5, w=pw - 44)
    f.t(x + 22, yy + 19, "都是<tspan font-weight=\"700\">指数</tspan>衰减。这正是语言不能用", GY,
        size=11.5, w=pw - 44)
    f.t(x + 22, yy + 38, "马尔可夫近似的原因。", GY, size=11.5)
    yy += 56

    f.line(x + 22, yy, x + pw - 22, yy, LINE, 1, arrow=False)
    yy += 20
    f.t(x + 22, yy, "双部互信息：把序列从中间切开", GR, True, 12.5)
    yy += 12
    f.box(bx, yy, bw, bh, "#fff", LINE, 6)
    f.box(bx + 10, yy + 10, (bw - 24) / 2.0, bh - 20, BG2, LINE2, 3)
    f.box(bx + 14 + (bw - 24) / 2.0, yy + 10, (bw - 24) / 2.0, bh - 20,
          BG2, LINE2, 3)
    f.t(bx + 10 + (bw - 24) / 4.0, yy + bh / 2.0 + 4, "前一半", GY,
        True, 12, "middle")
    f.t(bx + 14 + (bw - 24) * 0.75, yy + bh / 2.0 + 4, "后一半", GY,
        True, 12, "middle")
    f.t(bx + bw / 2.0, yy + bh / 2.0 + 4, "↔", GR, True, 15, "middle")
    yy += bh + 18
    f.t(x + 22, yy, "⭐ 这个量随长度<tspan font-weight=\"700\">幂律增长</tspan>。", GR, True, 12.5)
    f.t(x + 22, yy + 21, "不矛盾：N 个位置能配出 N² 对 ——", GY, size=11.5)
    f.t(x + 22, yy + 40, "<tspan font-weight=\"700\">单对越来越弱，对数越来越多。</tspan>", GY, size=11.5)
    fits(yy + 46, y0, ph, "①")

    # ══ ② L2M 条件 ══════════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② L2M 条件", PU,
                 sub="一条硬下界")

    yy = py + 26
    f.box(x + 22, yy, pw - 44, 78, "#fff", PU, 8)
    f.box(x + 22, yy, 4, 78, PU, PU, 2)
    f.box(x + 24, yy, 3, 78, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "模型能表达出来的互信息，", PU, True, 12.5)
    f.t(x + 40, yy + 47, "<tspan font-weight=\"700\">受限于它历史状态的维度</tspan>。", PU, True, 12.5)
    f.t(x + 40, yy + 68, "（缓存里装不下的东西，它就是不知道）", GY2,
        size=11)
    yy += 92

    f.box(x + 22, yy, pw - 44, 76, "#fff", INK, 8)
    f.t(x + 38, yy + 25, "⭐⭐ 于是有了一条必要条件：", INK, True, 13,
        cls="svglbl")
    f.t(x + 38, yy + 49, "<tspan font-weight=\"700\">状态维度必须至少以同样的幂律增长</tspan>", INK,
        True, 12.5, w=pw - 76)
    f.t(x + 38, yy + 68, "否则总存在一个长度，它兜不住", GY, size=11.5)
    yy += 90

    f.t(x + 22, yy, "那现有架构各自什么情况？", GY, True, 12.5)
    yy += 20
    for who, how, verdict, col in [
        ("Transformer", "KV 随长度<tspan font-weight=\"700\">线性</tspan>增长", "自动满足", GR),
        ("固定大小状态", "不随长度增长", "必然违反", RD),
    ]:
        f.box(x + 22, yy, pw - 44, 60, "#fff", col, 8)
        f.t(x + 38, yy + 24, who, col, True, 12.5)
        f.t(x + 38, yy + 45, how, GY, size=11.5, w=pw - 180)
        f.t(x + pw - 38, yy + 34, verdict, col, True, 12.5, "end")
        yy += 68

    f.t(x + 22, yy + 2, "⭐ 注意：Transformer 的「自动满足」是", GY,
        size=11.5)
    f.t(x + 22, yy + 21, "<tspan font-weight=\"700\">拿平方计算换来的</tspan> —— 它供给过量。", GY,
        size=11.5)
    fits(yy + 28, y0, ph, "②")

    # ══ ③ 三个旋钮在这把尺子下 ══════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 三个旋钮，各自站在哪", OR,
                 sub="⚠️ 这一格是本课的推导")

    yy = py + 24
    for knob, what, order, verdict, col in [
        ("旋钮① 每份更小", "MLA / GQA 压的是<tspan font-weight=\"700\">每个 token 存多少</tspan>",
         "阶还是 L，动的是常数", "安全", GR),
        ("旋钮② 动态挑选", "DSA / NSA / CSA：KV <tspan font-weight=\"700\">照样全存</tspan>，只是不读",
         "状态仍是 L", "安全", GR),
        ("旋钮② 固定窗口", "纯 SWA：窗口外的<tspan font-weight=\"700\">直接扔了</tspan>",
         "状态是 O(W) 常数", "同罪", RD),
        ("旋钮③ 固定状态", "把历史压成一个<tspan font-weight=\"700\">不随长度增长</tspan>的东西",
         "阶掉到 O(1)", "同罪", RD),
    ]:
        h = 82
        f.box(x + 22, yy, pw - 44, h, "#fff", col, 8)
        f.box(x + 22, yy, 4, h, col, col, 2)
        f.box(x + 24, yy, 3, h, "#fff", "#fff", 0)
        f.t(x + 40, yy + 25, knob, col, True, 12.5)
        f.t(x + pw - 38, yy + 25, verdict, col, True, 12.5, "end")
        f.t(x + 40, yy + 50, what, GY, size=11.5, w=pw - 76)
        f.t(x + 40, yy + 70, order, GY2, size=11, w=pw - 190)
        if col == RD:
            f.t(x + pw - 38, yy + 46, "只能混着用", RD, True, 11.5, "end")
        yy += h + 10
    fits(yy, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 「远处可以少看」这句直觉，现在有了严格版本", [
        "它<tspan font-weight=\"700\">不是</tspan>「远处不重要」——&#160;"
        "两半之间的互信息是<tspan font-weight=\"700\">随长度增长</tspan>的，"
        "只是<tspan font-weight=\"700\">增长得慢（次线性）</tspan>。",
        "⭐ 于是整件事变得很清楚："
        "<tspan font-weight=\"700\">Transformer 的 KV 线性增长是「供给过量」</tspan>，"
        "而这条幂律是「实际需求」。"
        "前两个旋钮就是在<tspan font-weight=\"700\">不掉到需求线以下的前提下，"
        "把那个过量的常数压小</tspan>。",
        "⛔ 而<tspan font-weight=\"700\">纯 SWA 和纯线性是同一类</tspan> ——&#160;"
        "它们动的都是<tspan font-weight=\"700\">阶</tspan>，不是常数。"
        "⭐⭐ 这条分类有一个白捡的验证：<tspan font-weight=\"700\">44 行表里所有 SWA 模型"
        "（Gemma 2/3/4、gpt-oss、MiMo 两代）也全是混着全注意力用的</tspan>，配比 1:1～6:1。",
        "⭐ 跟线性那批<tspan font-weight=\"700\">同一个区间</tspan> ——&#160;"
        "同一条理论，一口气解释了两类看起来无关的混合。",
    ])

    yy = f.band(yy + 14, "warn", "口径，三条", [
        "⚠️ L2M 是<tspan font-weight=\"700\">必要条件，不是充分条件</tspan>：状态够大只是"
        "「有可能记住」，不等于「真的学会了」。",
        "⚠️ L2M 是<tspan font-weight=\"700\">渐近</tspan>命题（「总存在一个长度」）——&#160;"
        "它说明<tspan font-weight=\"700\">纯线性/纯窗口不能无限外推</tspan>，"
        "<tspan font-weight=\"700\">不说明在 1M 这个具体尺度上非混不可</tspan>（那部分仍是消融出来的）。",
        "⚠️ 第三格（各旋钮站在哪）是<tspan font-weight=\"700\">本课按定义做的推导</tspan>，"
        "论文没有逐个旋钮分析 ——&#160;论文只分析了 Transformer 与状态空间模型两类。",
        "⚠️ 两点互信息与双部互信息<tspan font-weight=\"700\">是两个不同的量</tspan>，"
        "L2M 明确说它们独立地各自 scaling ——&#160;别拿一个去推另一个。",
    ])

    yy = f.src(yy + 16,
               "① 出自 Lin ＆ Tegmark《Critical Behavior in Physics and "
               "Probabilistic Formal Languages》（Entropy 2017, arXiv 1606.06737）："
               "马尔可夫/隐马尔可夫过程互信息指数衰减，实测自然语言近似幂律",
               "② 出自 L2M（arXiv 2503.04725, ICML 2025）：双部互信息幂律 scaling、"
               "「状态维度必须至少同阶增长」的定理、以及 Transformer 自动满足的那段分析",
               "③ 为本课推导，非论文结论")
    f.save("fig3-info-law.svg", yy + 6)


main()
