# -*- coding: utf-8 -*-
r"""专题三 · §八「混合：配比是一条轴，而两头都不好」

⭐⭐⭐ 2026-09-14 **整张重画**，换成一个带过团队的人都懂的画面：
   **一个组里，几个普通员工配一个资深。**

   · **便宜的层** ＝ 普通员工：只看手边那块记事板，反应快、便宜，
     但**查不了全部档案**。
   · **贵的层** ＝ 资深：**能翻全部历史**，什么都查得到，但慢、占地方。

   ① **两头都不好**：全是资深 → 贵得离谱，而且 ⭐ **消融里它居然还不是最好的**；
      全是普通员工 → 长了就兜不住（§二 那条 L2M 条件说的是**状态必须变大**）。
   ② **配比是一条轴**，各家落在哪，画在一根对数轴上 ——&nbsp;
      ⭐⭐ 最有意思的事实是：**两端各 1/8 的区间里，一个模型都没有。**
   ③ **为什么资深不用配很多** ——&nbsp;因为他查到的东西
      **会顺着流程传给后面所有人**，不用每个人自己去查一遍。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    f = Fig(W, "混合：把一层层想成一个组 —— 几个普通员工配一个资深。"
               "全是资深贵得离谱而且消融里还不是最好，全是普通员工长了兜不住；"
               "各家配比落在 1:1 到 7:1，两端各八分之一的区间里一个模型都没有")
    f.marks = set()
    y0 = f.header(
        "混合 ——　几个普通员工，配一个资深",
        "<tspan font-weight=\"700\">便宜的层</tspan> ＝ 只看手边记事板的普通员工；"
        "<tspan font-weight=\"700\">贵的层</tspan> ＝ 能翻全部档案的资深",
        [(GR, "普通员工"), (BL, "资深"), (RD, "两头都不好")])

    # ══════════ ① 两头都不好 ════════════════════════════════════
    PH = 312
    py = f.panel(0, y0, W, PH, "① 为什么不能只用一种人", RD,
                 sub="两头都试过，两头都不好")

    ay = py + 24

    def team(x, n_cheap, n_pro, col, title, good, bad):
        f.box(x, ay + 26, 420, 200, "#fff", col, 10)
        f.t(x + 24, ay + 64, title, col, True, 23)
        for i in range(n_cheap):
            f.box(x + 24 + i * 46, ay + 84, 36, 44, "#e6f4ea", GR, 6)
            f.t(x + 42 + i * 46, ay + 112, "普", GR, True, 17, "middle")
        for i in range(n_pro):
            f.box(x + 24 + (n_cheap + i) * 46, ay + 84, 36, 44, "#e8f0fe",
                  BL, 6)
            f.t(x + 42 + (n_cheap + i) * 46, ay + 112, "资", BL, True, 17,
                "middle")
        f.t(x + 24, ay + 162, good, GY, size=17, w=376)
        f.t(x + 24, ay + 196, bad, RD, True, 18, w=376)

    team(56, 0, 8, RD, "全是资深", "什么都查得到",
         "⛔ 贵得离谱 ——　而且消融里它还不是最好的")
    team(500, 8, 0, RD, "全是普通员工", "又快又省",
         "⛔ 一长就兜不住：板子大小是固定的")
    team(944, 6, 2, GR, "混着用", "大部分人快，少数几个能查全部",
         "✅ 今天所有人的选择")

    f.t(56, ay + 258, "⭐⭐ 左边那个结果最反直觉：Kimi 的消融里 "
        "<tspan font-weight=\"700\">0:1（全是资深）反而表现不好</tspan> ——&#160;"
        "⚠️ 原文只有这一句定性描述，没公开数值", INK, True, 19)
    f.t(56, ay + 286, "如果混合只是「拿便宜的换点钱」，那全用贵的应该最好才对。"
        "它不是 ——&#160;说明<tspan font-weight=\"700\">加便宜层不只是省钱</tspan>。",
        GY, size=17)

    # ══════════ ② 配比轴 ════════════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 262
    py2 = f.panel(0, y1, W, PH2, "② 那到底几个配一个 ——　各家都落在哪",
                  GR, sub="横轴：便宜层 : 贵层（对数刻度）")

    by = py2 + 46
    ax0, ax1 = 120, W - 120

    def at(k):
        return ax0 + (ax1 - ax0) * math.log(k) / math.log(8.0)

    f.line(ax0, by, ax1, by, GY2, 1.6, arrow=False)
    lo, hi = at(3), at(6)
    f.box(lo, by - 11, hi - lo, 22, "#e6f4ea", "none", 5)
    f.t((lo + hi) / 2.0, by - 22, "消融建议区间 3:1 ～ 6:1", GR, True, 17,
        "middle")
    f.t(ax0, by + 44, "1 : 1", GY2, size=15)
    f.t(ax1, by + 44, "8 : 1", GY2, size=15, anchor="end")

    ROWS = [
        (1, "1 : 1", "Gemma 2 · gpt-oss-120b", OR),
        (3, "3 : 1", "Kimi Linear · K3 · Qwen3.5 · GLM-5.3F · Llama 4 Scout", GR),
        (5, "5 : 1", "Ling-3.0-flash · MiMo-V2-Flash · Gemma 3/4", GR),
        (6, "6 : 1", "MiMo-V2.5-Pro", OR),
        (7, "7 : 1", "Ling 2.6 · MiniMax-01 · Jamba", BL),
    ]
    for n_, (k, lab, who, col) in enumerate(ROWS):
        px = at(k)
        f.box(px - 5, by - 11, 10, 22, col, "none", 3)
        f.t(px, by - 44 if n_ % 2 else by + 78, lab, col, True, 20, "middle")
    ly = by + 104
    for k, lab, who, col in ROWS:
        f.t(120, ly, lab, col, True, 17)
        f.t(220, ly, who, GY, size=16)
        ly += 26

    # ══════════ ③ 资深不用配很多 ════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 268
    py3 = f.panel(0, y2, W, PH3, "③ 为什么资深不用配很多", BL,
                  sub="他查到的东西会往下传")

    ey = py3 + 24
    for i in range(8):
        x = 96 + i * 158
        pro = (i == 2)
        f.box(x, ey + 40, 110, 70, "#e8f0fe" if pro else "#e6f4ea",
              BL if pro else GR, 8)
        f.t(x + 55, ey + 74, "资深" if pro else "普通", BL if pro else GR,
            True, 19, "middle")
        f.t(x + 55, ey + 100, "第 %d 层" % (i + 1), GY2, size=13,
            anchor="middle")
        if i < 7:
            f.line(x + 112, ey + 75, x + 152, ey + 75, GY2, 1.4)
    f.line(206, ey + 34, 1310, ey + 34, BL, 2.0)
    f.t(700, ey + 22, "他查到的结果，顺着残差流传给后面每一层", BL, True, 19,
        "middle")

    f.box(96, ey + 136, 600, 104, "#e8f0fe", BL, 10)
    f.t(120, ey + 174, "⭐ 只要有几层能「查全部档案」", BL, True, 21)
    f.t(120, ey + 210, "后面的人直接用他的结论就行 ——　不用每层自己查一遍", GY,
        size=17)

    f.box(736, ey + 136, 574, 104, "#f3e8fd", PU, 10)
    f.t(760, ey + 174, "⭐⭐ 意外红利：全注意力层可以不加位置编码", PU, True, 20)
    f.t(760, ey + 210, "因为夹在中间的便宜层<tspan font-weight=\"700\">本身带时序</tspan>",
        GY, size=17)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这条轴上最值得说的，是「大家都在哪」", [
        "<tspan font-weight=\"700\">两端各 1/8 的区间里，一个模型都没有</tspan> ——&#160;"
        "没有人只掺一两层，也没有人敢全用便宜的。",
        "⭐ 拆成两族看更有意思：<tspan font-weight=\"700\">线性混合 3:1 / 5:1 / 7:1</tspan>、"
        "<tspan font-weight=\"700\">滑窗混合 1:1 / 3:1 / 5:1 / 6:1</tspan>，"
        "两族<tspan font-weight=\"700\">重叠在 3:1 ～ 6:1</tspan>。"
        "两类看起来毫不相干的混合落进同一段区间，<tspan font-weight=\"700\">"
        "这件事本身就是一条证据</tspan>。",
    ])

    yy = f.band(yy + 14, "warn", "配比是超参，别背成常识", [
        "⚠️ 这条轴上的「便宜层」<tspan font-weight=\"700\">不全是线性</tspan> ——&#160;"
        "小米那两家是<tspan font-weight=\"700\">滑窗</tspan>。"
        "<tspan font-weight=\"700\">3:1 是消融出来的，不是推出来的</tspan>；"
        "同一家不同规模就换配比（Ling 的 tiny 是 3:1、flash 是 5:1）。",
        "⚠️ 「纯线性一定兜不住」要说准：§二 那条 L2M 条件要求的是"
        "<tspan font-weight=\"700\">状态必须随长度变大</tspan> ——&#160;"
        "而「变大」的办法<tspan font-weight=\"700\">不止混合一种</tspan>"
        "（论文自己给的另一条是按长度整个放大模型）。"
        "⭐ <tspan font-weight=\"700\">混合是工程上选的那条，不是定理逼出来的那条。</tspan>",
        "⚠️ 而且它是<tspan font-weight=\"700\">渐近</tspan>命题 ——&#160;"
        "同一张表里 Mistral 7B 就是<tspan font-weight=\"700\">纯滑窗</tspan>、"
        "一层全注意力都没有，还真的上过生产用了两年。",
    ])

    yy = f.src(yy + 16,
               "配比与型号见 §8.2 / §8.4 的表（每一行都标了出处，多数可在公开 "
               "config 里核）；系统性消融的建议区间出自 arXiv 2507.06457",
               "「0:1 反而表现不好」出自 Kimi Linear arXiv 2510.26692 ——&#160;"
               "⚠️ 原文只有一句定性描述，<tspan font-weight=\"700\">没有公开数值</tspan>",
               "NoPE 见 <tspan font-weight=\"700\">Kimi Linear</tspan> 同文（对所有全注意力层"
               "用 NoPE）——&#160;⚠️ <tspan font-weight=\"700\">是它先做的，K3 是沿用</tspan>",
               "⚠️ 「普通员工 / 资深」是<tspan font-weight=\"700\">本课的比喻</tspan>")
    f.save("fig3-hybrid.svg", yy + 6)


main()
