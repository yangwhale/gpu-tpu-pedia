# -*- coding: utf-8 -*-
r"""专题三 · §八「配比是一条轴，而两头都不好」（2026-09-13 夜间 · R14）。

⭐ §八 原来是**五张表**。表该留着（型号对配比，本来就是表），
   但它缺了一张**把结论摆出来的图**：

  ① 三种单用各自的致命短板 ——&nbsp;而其中一条现在有了**理论版本**：
     §二 那条 L2M 条件说得很死，**纯线性必然在某个长度上兜不住**。
     ⭐⭐ 于是「为什么必须混合」不再是工程经验，是有下界的。

  ② **配比是一条轴，而两头都不好。**
     把各家点在同一条轴上，建议区间 3:1 ～ 6:1 一眼就出来。
     ⭐ 而最值得讲的是左端那个反直觉结果：
     Kimi 的消融里 **0:1（纯全注意力）反而表现不好** ——&nbsp;
     **加线性层不只是省钱，它可能还带来了别的东西。**

  ③ **为什么全局层不用多**，以及 NoPE 那个意外红利。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "配比是一条轴而两头都不好：纯全注意力和纯线性各有致命短板，"
               "各家落在 3:1 到 7:1 之间；全局层不用多因为信息沿残差流传")
    f.marks = set()
    y0 = f.header(
        "混合　——　配比是一条轴，而<tspan font-weight=\"700\">两头都不好</tspan>",
        "⭐⭐ 「为什么必须混合」现在有理论版本了 ——&#160;"
        "§二 那条 L2M 条件说<tspan font-weight=\"700\">纯线性不能无限外推</tspan>——&#160;而它连滑窗一起判",
        [(RD, "单用的短板"), (GR, "各家实际落点"),
         (BL, "建议区间"), (PU, "意外红利")])

    ph = 430

    # ══ ① 三种单用的短板 ════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 单用任何一个都有致命短板", RD,
                 sub="第三条现在有理论版本了")

    yy = py + 26
    for who, bad, col in [
        ("只用滑窗", "跨不了长距离", RD),
        ("只用纯全注意力", "KV 和 FLOPs 都爆", RD),
        ("只用纯线性 / 纯滑窗", "精确检索塌 · 状态是常数", RD),
    ]:
        f.box(x + 22, yy, pw - 44, 50, "#fff", col, 8)
        f.t(x + 38, yy + 30, who, col, True, 12.5)
        f.t(x + pw - 38, yy + 30, bad, GY, size=12, anchor="end")
        yy += 58

    yy += 6
    f.box(x + 22, yy, pw - 44, 128, "#fff", PU, 8)
    f.box(x + 22, yy, 4, 128, PU, PU, 2)
    f.box(x + 24, yy, 3, 128, "#fff", "#fff", 0)
    f.t(x + 40, yy + 26, "⭐⭐ 第三条不只是「实测不行」", PU, True, 13,
        cls="svglbl")
    f.t(x + 40, yy + 50, "回到 §二 那条 L2M 条件：", GY, size=11.5)
    f.t(x + 40, yy + 70, "状态维度必须<tspan font-weight=\"700\">至少同阶增长</tspan>，", GY, size=11.5)
    f.t(x + 40, yy + 90, "而纯线性的状态<tspan font-weight=\"700\">根本不增长</tspan> ——", GY, size=11.5)
    f.t(x + 40, yy + 112, "<tspan font-weight=\"700\">总存在一个长度，它必然兜不住。</tspan>", PU,
        True, 12.5)
    yy += 142

    f.t(x + 22, yy, "⭐ 所以「纯线性不行」有一条<tspan font-weight=\"700\">渐近</tspan>必要条件撑着；", INK,
        True, 12.5, w=pw - 44)
    f.t(x + 22, yy + 21, "⚠️ 但<tspan font-weight=\"700\">「在 1M 这个尺度上非混不可」仍然是消融出来的</tspan>。",
        INK, size=11.5, w=pw - 44)
    fits(yy + 28, y0, ph, "①")

    # ══ ② 配比轴 ════════════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 配比是一条轴", GR,
                 sub="⚠️ 轴是「便宜层 : 贵层」")

    yy = py + 34
    ax0, ax1 = x + 40, x + pw - 40
    f.line(ax0, yy, ax1, yy, GY2, 1.4, arrow=False)
    f.t(ax0, yy - 14, "0 : 1", RD, True, 12)
    f.t(ax0, yy + 20, "全是全注意力", GY2, size=11)
    f.t(ax1, yy - 14, "1 : 0", RD, True, 12, "end")
    f.t(ax1, yy + 20, "全是便宜层", GY2, size=11, anchor="end")

    # 建议区间 3:1 ~ 6:1 的带
    def at(r):     # r 是「线性 : 全」的线性占比 0..1
        return ax0 + (ax1 - ax0) * r
    lo, hi = at(3 / 4.0), at(6 / 7.0)
    f.box(lo, yy - 8, hi - lo, 16, "#e6f4ea", "none", 4)
    f.t((lo + hi) / 2.0, yy - 16, "建议区间", GR, True, 11.5, "middle")

    for ratio, label, r in [
        ("3:1", "Kimi Linear · K3 · Qwen3.5 · GLM-5.3", 3 / 4.0),
        ("5:1", "Ling-3.0-flash · MiMo-V2-Flash", 5 / 6.0),
        ("6:1", "MiMo-V2.5-Pro", 6 / 7.0),
        ("7:1", "Ling 2.6 · MiniMax-01", 7 / 8.0),
    ]:
        px = at(r)
        f.box(px - 3, yy - 5, 6, 10, GR, "none", 2)
    yy += 44
    for ratio, label in [
        ("3 : 1", "Kimi Linear · K3 · Qwen3.5 · GLM-5.3-Flash"),
        ("5 : 1", "Ling-3.0-flash（线性）· MiMo-V2-Flash（滑窗）"),
        ("6 : 1", "MiMo-V2.5-Pro（滑窗）"),
        ("7 : 1", "Ling 2.6 · MiniMax-01（线性）"),
    ]:
        f.box(x + 22, yy, pw - 44, 40, "#fff", LINE, 6)
        f.t(x + 38, yy + 25, ratio, GR, True, 12.5)
        f.t(x + 96, yy + 25, label, GY, size=11, w=pw - 140)
        yy += 46

    yy += 4
    f.box(x + 22, yy, pw - 44, 76, "#fff", RD, 8)
    f.box(x + 22, yy, 4, 76, RD, RD, 2)
    f.box(x + 24, yy, 3, 76, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐ 左端那个结果最值得讲：", RD, True, 12.5)
    f.t(x + 40, yy + 47, "Kimi 的消融里 <tspan font-weight=\"700\">0:1（纯全注意力）</tspan>", GY, size=11.5)
    f.t(x + 40, yy + 67, "<tspan font-weight=\"700\">反而表现不好</tspan>。", RD, True, 12.5)
    fits(yy + 76, y0, ph, "②")

    # ══ ③ 为什么全局层不用多 ════════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 为什么全局层不用多", BL,
                 sub="＋ 一个意外红利")

    yy = py + 26
    f.box(x + 22, yy, pw - 44, 80, "#fff", BL, 8)
    f.box(x + 22, yy, 4, 80, BL, BL, 2)
    f.box(x + 24, yy, 3, 80, "#fff", "#fff", 0)
    f.t(x + 40, yy + 26, "只要有几层能做<tspan font-weight=\"700\">无损检索</tspan>，", BL, True, 12.5)
    f.t(x + 40, yy + 50, "信息就能<tspan font-weight=\"700\">沿着残差流</tspan>传给其余层用。", GY,
        size=11.5)
    f.t(x + 40, yy + 70, "——&#160;不是每层都得自己去查一遍", GY2, size=11)
    yy += 94

    f.t(x + 22, yy, "⭐⭐ 意外红利：K3 的全注意力层不加位置编码", PU,
        True, 13, cls="svglbl")
    yy += 24
    f.box(x + 22, yy, pw - 44, 74, "#fff", PU, 8)
    f.box(x + 22, yy, 4, 74, PU, PU, 2)
    f.box(x + 24, yy, 3, 74, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "因为夹在中间的线性层<tspan font-weight=\"700\">本身带时序</tspan>", PU,
        True, 12.5)
    f.t(x + 40, yy + 47, "（靠递归的衰减和门控编码顺序）", GY, size=11.5)
    f.t(x + 40, yy + 67, "⭐ <tspan font-weight=\"700\">位置由线性层给，全注意力只管检索</tspan>", GY,
        size=11.5)
    yy += 88

    for r in ["不用调 RoPE 外推 —— 直接到 1M",
              "MLA 层推理时可退化成纯 MQA（上投影能全吸收）",
              "KV 再降 75% —— ⚠️ 那是 3:1 配比的功劳，不是 NoPE 的"]:
        f.box(x + 22, yy, pw - 44, 38, "#fff", LINE, 6)
        f.t(x + 38, yy + 24, "→ " + r, GY, size=11.5, w=pw - 76)
        yy += 44
    fits(yy, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 「0:1 反而不好」这个结果，比「3:1 最好」有意思得多", [
        "如果混合只是「拿便宜的层换点钱」，那<tspan font-weight=\"700\">全用贵的应该最好才对</tspan>。"
        "可它不是 ——&#160;这说明<tspan font-weight=\"700\">加线性层不只是省钱，"
        "它可能还带来了别的东西</tspan>（比如一种天然的时序归纳偏置）。",
        "⭐ 讲课时这句比配比本身值钱：<tspan font-weight=\"700\">当一个「纯粹为了省钱」的改动"
        "反而把效果做好了，那它就不只是省钱 ——&#160;去找它顺带改变了什么。</tspan>",
    ])

    yy = f.band(yy + 14, "warn", "配比是超参，别背成常识", [
        "⚠️ <tspan font-weight=\"700\">这条轴上的「便宜层」不全是线性</tspan> ——&#160;"
        "小米那两家是<tspan font-weight=\"700\">滑窗</tspan>。⭐ 两类混合落在同一个区间，"
        "本身就是一条证据。<tspan font-weight=\"700\">3:1 是消融出来的，不是推出来的。</tspan>"
        "同一家不同规模就换配比（Ling 的 tiny 是 3:1、flash 是 5:1）。",
        "⭐ 两个口径别混：<tspan font-weight=\"700\">实测落点 3:1～7:1</tspan>（本课 44 行表），"
        "<tspan font-weight=\"700\">消融建议 3:1～6:1</tspan>（arXiv 2507.06457）——&#160;"
        "<tspan font-weight=\"700\">区间比点值可信</tspan>。",
    ])

    yy = f.src(yy + 16,
               "配比与型号见 §8.2 / §8.4 的表（每一行都标了出处，多数可在公开 config 里核）；"
               "系统性消融的建议区间出自 arXiv 2507.06457",
               "「纯线性违反 L2M 条件」一条接 §二 的 arXiv 2503.04725 ——&#160;"
               "⚠️ 论文分析的是状态空间模型这一类，套到「纯线性混合比为 1:0」是本课的推导",
               "NoPE 的三个后果与 Kimi Linear 的 75% / 6.3× 见 arXiv 2510.26692 与 K3 技术报告")
    f.save("fig3-hybrid.svg", yy + 6)


main()
