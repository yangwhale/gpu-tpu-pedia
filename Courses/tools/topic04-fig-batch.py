# -*- coding: utf-8 -*-
r"""专题四 · §1.8「batch 开大到底快在哪」——&#160;而那句「大 batch 更快」有个前提

⭐⭐⭐ 2026-09-17 新画。取法来自**李宏毅**（台大）讲 batch 的那一段。
  他的原话大意是：**在有平行运算的情况下，小 batch 和大 batch
  跑一次的时间并没有太大差距**，除非大到非常大；
  所以「更新一次」小 batch 快，而「跑完一个 epoch」反过来 ——&#160;大 batch 快。

⛔⛔ 但这一讲**不能照抄这个结论** ——&#160;它成立的前提是「并行度还没吃满」。
  ⭐ 而我们整讲谈的是几百亿到几千亿参数、几百张卡的场景，
    那基本就在**吃满之后**那一段：这时候一步的时间**正比于 batch**，
    「大 batch 一个 epoch 更快」这句话直接失效。
  ⭐⭐⭐ 所以这张图画的不是那个结论，是**那个结论的适用区间** ——&#160;
    而这正是本讲反复在说的那条判据：
    **小规模上验过的结论，到目标规模必须重验。**
  ⛔ 判据（给自己的）：**借别人的讲法时，连同它的前提一起借。**
    只搬结论、把前提留在原处，是这一讲已经踩过的坑（§3.5 那篇 warmup 论文）。

⭐ 而 Ⓒ 把它跟 §1.8 已有的那条接上：梯度通信量**跟 batch 完全无关**，
  所以在**两段里**它都是「batch 越大摊得越薄」——&#160;
  通信和计算在这件事上的行为不一样，这一点值得单说。

📌 「有平行运算时大小 batch 单次时间差别不大 / 一个 epoch 反过来」
  出自李宏毅 2021《类神经网络训练不起来怎么办（二）：批次与动量》。
  ⛔ 图是我们自己重画的；图上不含任何实测数字，只画两段的**形状**。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400


def main():
    f = Fig(W, "横轴是 batch 大小，纵轴是一次更新要花的时间。左半段并行度还没吃满，"
               "时间几乎不随 batch 变化，所以 batch 开大是白赚的；"
               "右半段吃满之后，时间正比于 batch，再开大就不白赚了。"
               "教程里常说的「大 batch 一个 epoch 更快」，"
               "成立的前提是左半段；而大模型训练基本都在右半段。"
               "另外梯度通信量跟 batch 完全无关，所以它在两段里都是摊得越薄越好")

    y0 = f.header(
        "「batch 开大就更快」　——　<tspan font-weight=\"700\">"
        "这句话有个前提，而它通常不写</tspan>",
        "⭐ 这一格画的不是结论，是<tspan font-weight=\"700\">结论的适用区间</tspan>",
        [(GR, "左段：没吃满"), (RD, "右段：吃满了"), (BL, "通信：两段都无关")])

    # ══════════ Ⓐ 一次更新的时间，分两段 ═══════════════════════════
    PH = 400
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 横轴 batch、纵轴<tspan font-weight=\"700\">一次更新要多久</tspan>"
                 "　——　这条线<tspan font-weight=\"700\">是折的，不是直的</tspan>", RD,
                 sub="⛔ 大多数讲法只画了左半段")

    OX, OY = 190, py + 320            # 原点
    AW, AH = 900, 230                 # 轴长
    KNEE = 0.42                       # 拐点位置（占横轴比例）
    F0 = 0.20                         # 左段的高度（占纵轴比例）

    f.line(OX, OY, OX + AW + 40, OY, GY2, 1.2)
    f.line(OX, OY, OX, OY - AH - 40, GY2, 1.2)
    f.t(OX + AW + 46, OY + 18, "batch →", GY2, size=12)
    f.t(OX - 8, OY - AH - 48, "一次更新的时间", GY2, size=12, anchor="end")

    kx = OX + AW * KNEE
    ky = OY - AH * F0
    # 左段：几乎平
    f.path("M %.1f %.1f L %.1f %.1f" % (OX, OY - AH * F0 * 0.86, kx, ky),
           GR, 2.4, arrow=False)
    # 右段：线性上升
    f.path("M %.1f %.1f L %.1f %.1f" % (kx, ky, OX + AW, OY - AH * 0.92),
           RD, 2.4, arrow=False)
    f.line(kx, OY, kx, ky - 8, GY2, 1, dash="4 3", arrow=False)
    f.box(kx - 7, ky - 7, 14, 14, INK, INK, 7)
    f.t(kx, ky - 24, "并行度在这儿吃满", INK, True, 13.5, "middle")

    f.t(OX + AW * KNEE * 0.5, OY - AH * F0 - 42, "几乎是平的", GR, True, 17, "middle")
    f.t(OX + AW * KNEE * 0.5, OY - AH * F0 - 18,
        "batch 翻倍，时间几乎不变", GY, size=13, anchor="middle")
    f.t(OX + AW * 0.74, OY - AH * 0.62, "正比于 batch", RD, True, 17, "middle")
    f.t(OX + AW * 0.74, OY - AH * 0.62 + 24,
        "batch 翻倍，时间也翻倍", GY, size=13, anchor="middle")

    f.t(OX + AW * KNEE * 0.5, OY + 30, "⭐ 卡还没喂饱", GR, True, 14, "middle")
    f.t(OX + AW * 0.74, OY + 30, "⛔ 卡已经满了", RD, True, 14, "middle")

    f.t(700, py + 372, "⭐⭐⭐ <tspan font-weight=\"700\">左段 batch 开大是白赚的</tspan>"
        "（步数变少，每步没变贵）；"
        "<tspan font-weight=\"700\">右段就不白赚了</tspan>（步数变少，每步同比变贵）。",
        INK, size=14.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 两段各自的结论 ═══════════════════════════════════
    PH2 = 268
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐ 同一句话，在两段里"
                  "<tspan font-weight=\"700\">一句成立、一句不成立</tspan>", OR,
                  sub="⛔ 而教程里那句「大 batch 更快」，"
                      "说的<tspan font-weight=\"700\">几乎都是左段</tspan>")

    SIDES = (
        (GR, "#e6f4ea", "左段　并行度没吃满",
         "「跑完固定的数据量，大 batch 更快」",
         "✅ 成立　——　步数少了，而每步没变贵",
         "⭐ 小模型、单卡、教学例子，多半在这儿"),
        (RD, "#fce8e6", "右段　并行度吃满了",
         "同一句话",
         "⛔ 不成立　——　每步变贵的倍数，正好抵掉步数少的倍数",
         "⭐ 几百亿参数、几百张卡的训练，基本在这儿"),
    )
    for i, (col, fill, tag, claim, verdict, who) in enumerate(SIDES):
        x = 60 + i * 650
        f.box(x, py2 + 36, 630, 190, fill, col, 8)
        f.box(x, py2 + 36, 630, 4, col, col, 2)
        f.t(x + 315, py2 + 74, tag, col, True, 18, "middle")
        f.t(x + 315, py2 + 110, claim, INK, True, 15.5, "middle")
        f.t(x + 315, py2 + 152, verdict, col, True, 14.5, "middle")
        f.t(x + 315, py2 + 202, who, GY, size=13, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 通信是另一回事 ═══════════════════════════════════
    PH3 = 214
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐ 但<tspan font-weight=\"700\">通信那一笔不分左右段</tspan>"
                  "　——　它跟 batch 完全无关", BL,
                  sub="⭐ 所以「batch 越大摊得越薄」这句话，"
                      "<tspan font-weight=\"700\">对通信永远成立</tspan>")

    f.box(70, py3 + 38, 600, 142, "#e8f0fe", BL, 8)
    f.t(370, py3 + 74, "每步要汇总的梯度", BL, True, 18, "middle")
    f.t(370, py3 + 108, "＝ 参数量 × 每参数字节数", INK, True, 16, "middle")
    f.t(370, py3 + 142, "喂一条序列和喂一千条，传的一样多", GY, size=13.5, anchor="middle")

    f.t(700, py3 + 106, "→", GY2, True, 22, "middle")

    f.box(730, py3 + 38, 600, 142, "#e6f4ea", GR, 8)
    f.t(1030, py3 + 74, "所以它<tspan font-weight=\"700\">只会被摊薄</tspan>",
        GR, True, 18, "middle")
    f.t(1030, py3 + 110, "batch 翻倍 →　每 token 分摊的通信减半", INK, True, 15, "middle")
    f.t(1030, py3 + 146, "⭐ 这一条在左段右段都成立", GR, size=13.5, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "warn",
                "这一格真正要留下的，不是「大 batch 快不快」，是那个「看情况」怎么看",
                ("⭐⭐⭐ 计算和通信在这件事上<tspan font-weight=\"700\">行为不一样</tspan>："
                 "计算<tspan font-weight=\"700\">分两段</tspan>（没吃满时白赚，吃满后不赚），"
                 "通信<tspan font-weight=\"700\">不分段</tspan>（永远摊得越薄越好）。"
                 "——　所以「batch 该开多大」不是一个数，是<tspan font-weight=\"700\">"
                 "看你现在卡在哪一栏</tspan>。",
                 "⛔ 判据（给自己的）：<tspan font-weight=\"700\">"
                 "借别人的讲法时，连同它的前提一起借。</tspan>"
                 "　只搬结论、把前提留在原处，是本讲已经踩过的坑。"))

    yb = f.src(yb + 16,
               "📌 「有平行运算时，大小 batch 跑一次的时间差别不大；"
               "而一个 epoch 反过来」出自<tspan font-weight=\"700\">李宏毅</tspan>"
               "2021 年《类神经网络训练不起来怎么办（二）：批次与动量》。"
               "⛔ 图是我们自己重画的。",
               "⚠️ 图上<tspan font-weight=\"700\">不含任何实测数字</tspan>　——　"
               "只画两段的形状。拐点落在哪，取决于模型、卡、并行配置，"
               "<tspan font-weight=\"700\">得自己在目标配置上量</tspan>。")

    f.save("fig4-batch.svg", yb + 14)


if __name__ == "__main__":
    main()
