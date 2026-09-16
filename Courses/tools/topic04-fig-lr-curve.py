# -*- coding: utf-8 -*-
r"""专题四 · §3.5「两种学习率曲线，终点却撞在同一个数上」

⭐⭐⭐ 2026-09-16 新画。现场原话：
  「学习率这个是个非常高深的手艺，该怎么调？有哪些方法？
    下降的速度是多少？一开始是多少？**这些都得讲。**」

⭐⭐ 这张图的取舍：**纵轴画「占峰值的百分比」，不画绝对学习率。**
  ⛔ 画绝对值的话，GPT-3 的 0.6e-4 和 V3 的 2.2e-4 差 3.7 倍，
    两条曲线在图上就没法叠，而**它们真正要比的是形状不是高度**。
  ⭐ 判据：**要比形状就归一化；要比大小才用绝对值。**
    （峰值的绝对值另有一张表，写在正文里 ——&#160;判据⑩：挨着图的那一处赢。）

⭐ 图上每一段都是论文里的原话换算来的：
  · GPT-3：头 3.75 亿 token 线性 warmup；2,600 亿 token 内余弦降到 10%；
    之后一直保持 10%。总量 3,000 亿 token。arXiv 2005.14165 §2.3
  · V3：2,000 步线性升到 2.2e-4；恒定到 10T token；再用 4.3T token
    余弦降到 2.2e-5；最后 500B 里前 333B 保持 2.2e-5，后 167B 换 7.3e-6。
    总量 14.8T token。arXiv 2412.19437 §4.2

⛔⛔ 刻意没画的：
  ① **warmup 那一小段的真实宽度。** GPT-3 的 warmup 只占 0.125%，
     按真实比例画就是一条竖线 ——&#160;所以这里**刻意画宽了**，图上注明了。
  ② **绝对学习率的数值轴。** 见上。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

# ── GPT-3（arXiv 2005.14165 §2.3）
G3_TOTAL, G3_WARM, G3_DECAY_END, G3_FLOOR = 300.0, 0.375, 260.0, 10.0
# ── DeepSeek-V3（arXiv 2412.19437 §4.2），单位 T token
V3_TOTAL, V3_STABLE_END, V3_DECAY_END = 14.8, 10.0, 14.3
V3_PEAK, V3_MID, V3_LAST = 2.2e-4, 2.2e-5, 7.3e-6
V3_FLOOR = V3_MID / V3_PEAK * 100
V3_TAIL = V3_LAST / V3_PEAK * 100
V3_CONST_END = V3_DECAY_END + 0.333

assert abs(V3_FLOOR - 10.0) < 1e-6          # ⭐ 也正好是十分之一
assert 3.3 < V3_TAIL < 3.4
assert abs(G3_WARM / G3_TOTAL * 100 - 0.125) < 1e-9


def main():
    f = Fig(W, "两条学习率曲线叠在一起比形状。"
               "GPT-3 是经典余弦：很短的 warmup 之后一路下滑，"
               "在总量的百分之八十七处降到峰值的十分之一，之后保持不变。"
               "DeepSeek-V3 是 warmup 加长平台加末段衰减："
               "升到峰值后一直恒定到总量的三分之二，才开始降。"
               "两条曲线的落点都在峰值的十分之一附近")

    y0 = f.header(
        "两种学习率曲线　——　<tspan font-weight=\"700\">"
        "形状差很多，落点却撞在同一个数上</tspan>",
        "⭐ 纵轴是<tspan font-weight=\"700\">占峰值的百分比</tspan>，不是绝对学习率"
        "　·　⚠️ warmup 那一小段<tspan font-weight=\"700\">按真实比例画就是一条竖线</tspan>，"
        "图上放宽了",
        [(BL, "GPT-3 · 余弦"), (RD, "V3 · 平台＋衰减"), (GY2, "峰值的 10%")])

    PH = 470
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 横轴是<tspan font-weight=\"700\">训练进度</tspan>（占总 token 数的比例）",
                 BL,
                 sub="⭐ 两条都归一化了，所以可以直接叠在一起比")

    X0, X1 = 150, 1320
    BOT, TOP = py + 336, py + 44

    def X(frac):
        return X0 + frac * (X1 - X0)

    def Y(pct):
        return BOT - pct / 105.0 * (BOT - TOP)

    # 轴
    f.line(X0, BOT, X1 + 10, BOT, GY2, 2.0, arrow=False)
    f.line(X0, BOT, X0, TOP - 10, GY2, 2.0, arrow=False)
    for fr in (0, 0.25, 0.5, 0.75, 1.0):
        f.line(X(fr), BOT, X(fr), BOT + 6, GY2, 1.2, arrow=False)
        f.t(X(fr), BOT + 26, "%d%%" % (fr * 100), GY2, size=12.5, anchor="middle")
    f.t((X0 + X1) / 2.0, BOT + 54, "训练进度（已消耗的 token 占总量的比例）",
        GY, True, 14, "middle")
    for pct in (100, 50, 10):
        f.t(X0 - 14, Y(pct) + 5, "%d%%" % pct, GY2, size=12, anchor="end")
    f.t(X0 - 14, TOP - 22, "占峰值", GY, True, 13, "end")

    # ⭐ 那条「十分之一」——&#160;全图唯一的水平参考线
    f.line(X0, Y(10), X1, Y(10), GY2, 1.6, dash="6 5", arrow=False)
    f.t(X1 - 6, Y(10) - 12, "峰值的 10%", GY2, True, 13.5, "end")

    WARM_DRAW = 0.035          # ⛔ 画宽了，真实是 0.125%（GPT-3）

    # ── GPT-3：warmup → 余弦到 10% → 恒定
    import math
    pts = ["M %.1f %.1f" % (X(0), Y(0)),
           "L %.1f %.1f" % (X(WARM_DRAW), Y(100))]
    a, b = WARM_DRAW, G3_DECAY_END / G3_TOTAL
    for i in range(1, 41):
        fr = a + (b - a) * i / 40.0
        u = (fr - a) / (b - a)
        pct = 10 + 90 * 0.5 * (1 + math.cos(math.pi * u))
        pts.append("L %.1f %.1f" % (X(fr), Y(pct)))
    pts.append("L %.1f %.1f" % (X(1.0), Y(10)))
    f.path(" ".join(pts), BL, 3.0, arrow=False)
    f.t(X(0.44), Y(46) - 14, "GPT-3　余弦", BL, True, 16, "middle")
    f.line(X(b), Y(10), X(b), BOT, BL, 1.2, dash="4 4", arrow=False)
    f.t(X(b), BOT + 76, "2,600 亿 token 处降到 10%", BL, size=12.5, anchor="middle")

    # ── V3：warmup → 恒定 → 余弦到 10% → 恒定 → 末段 3.3%
    s1 = V3_STABLE_END / V3_TOTAL
    s2 = V3_DECAY_END / V3_TOTAL
    s3 = V3_CONST_END / V3_TOTAL
    pts = ["M %.1f %.1f" % (X(0), Y(0)),
           "L %.1f %.1f" % (X(WARM_DRAW), Y(100)),
           "L %.1f %.1f" % (X(s1), Y(100))]
    for i in range(1, 25):
        fr = s1 + (s2 - s1) * i / 24.0
        u = (fr - s1) / (s2 - s1)
        pct = V3_FLOOR + (100 - V3_FLOOR) * 0.5 * (1 + math.cos(math.pi * u))
        pts.append("L %.1f %.1f" % (X(fr), Y(pct)))
    pts.append("L %.1f %.1f" % (X(s3), Y(V3_FLOOR)))
    pts.append("L %.1f %.1f" % (X(s3), Y(V3_TAIL)))
    pts.append("L %.1f %.1f" % (X(1.0), Y(V3_TAIL)))
    f.path(" ".join(pts), RD, 3.0, arrow=False)
    f.t(X(0.34), Y(100) - 16, "DeepSeek-V3　长平台", RD, True, 16, "middle")
    f.line(X(s1), Y(100), X(s1), BOT, RD, 1.2, dash="4 4", arrow=False)
    f.t(X(s1), BOT + 100, "10T token 才开始降", RD, size=12.5, anchor="middle")

    # 峰值那一小段的说明
    bx, by = X(0.26), Y(30)
    f.box(bx, by, 268, 56, "#fff", GY2, 8)
    f.t(bx + 16, by + 24, "⚠️ warmup 画宽了", GY, True, 14)
    f.t(bx + 16, by + 46, "GPT-3 真实只占 0.125%", GY2, size=12.5)
    f.line(bx, by + 18, X(WARM_DRAW) + 6, Y(58), GY2, 1.2, dash="3 3")
    f._pan = None

    # ══════════ Ⓑ 为什么会有第二种 ═══════════════════════════════
    PH2 = 282
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ ⭐ WSD 这两年流行起来的理由特别实在", RD,
                  sub="⛔ 不是「它收敛更好」——　"
                      "<tspan font-weight=\"700\">是它把一个决定往后推了</tspan>")

    COLS = (
        (BL, "#e8f0fe", "余弦的硬约束",
         "曲线形状<tspan font-weight=\"700\">依赖终点</tspan>",
         "⛔ 所以你必须**一开始就知道总共训多少步**",
         "中途想多训一段，整条曲线都得重来"),
        (RD, "#fce8e6", "平台段的自由",
         "恒定段<tspan font-weight=\"700\">可以随时截断</tspan>",
         "⭐ 接一小段快速衰减，就能出一个能用的 checkpoint",
         "想加数据接着训，从平台段续上就行"),
    )
    for i, (col, fill, nm, a, b, c) in enumerate(COLS):
        x = 88 + i * 632
        f.box(x, py2 + 34, 592, 176, fill, col, 8)
        f.box(x, py2 + 34, 592, 4, col, col, 2)
        f.t(x + 296, py2 + 70, nm, col, True, 19, "middle")
        f.t(x + 296, py2 + 104, a, INK, True, 15.5, "middle")
        f.t(x + 296, py2 + 142, b.replace("**", ""), GY, size=13.5, anchor="middle")
        f.t(x + 296, py2 + 176, c, GY, size=13.5, anchor="middle")
    f.t(700, py2 + 240, "⭐⭐ 一句话：<tspan font-weight=\"700\">"
                        "它把「训多久」这个决定，从开局推迟到了随时</tspan>",
        RD, True, 16, "middle")
    f._pan = None

    yy = f.band(py2 + PH2 + 22, "ok", "两个落点，一个巧合", [
        "⭐ <tspan font-weight=\"700\">warmup 的量级是「总量的千分之几」</tspan>，"
        "不是需要精调的东西 ——&#160;GPT-3 占 0.125%，V3 是头 2,000 步。"
        "⛔ 而且有论文实测：<tspan font-weight=\"700\">目标学习率固定的话，"
        "warmup 拉长基本没收益</tspan>，决定效果的是峰值本身。",
        "⭐⭐ <tspan font-weight=\"700\">两条曲线的落点都在峰值的十分之一附近</tspan>"
        " ——&#160;GPT-3 明写「降到 10%」；V3 是 2.2e-4 →&#160;2.2e-5，也正好十分之一"
        "（最后一小段再往下到 7.3e-6，约 3.3%）。"
        "⚠️ 两个样本不构成定律，<tspan font-weight=\"700\">"
        "但足以说明「降到零」不是默认做法</tspan>。",
    ], keep=True)

    yy = f.src(yy + 24,
               "GPT-3：<tspan font-weight=\"700\">arXiv 2005.14165</tspan> §2.3 ——&#160;"
               "「头 3.75 亿 token 线性 warmup」「2,600 亿 token 内余弦降到 10%，"
               "之后继续以 10% 训练」；总量 3,000 亿 token",
               "DeepSeek-V3：<tspan font-weight=\"700\">arXiv 2412.19437</tspan> §4.2 ——&#160;"
               "2,000 步线性升到 2.2e-4 →&#160;恒定到 10T →&#160;4.3T 内余弦降到 2.2e-5 "
               "→&#160;末 500B 里前 333B 保持、后 167B 换 7.3e-6；梯度裁剪范数 1.0",
               "⭐ warmup 的真实机制（<tspan font-weight=\"700\">不是「样本太少估不准」</tspan>）："
               "arXiv <tspan font-weight=\"700\">2406.09405</tspan>（NeurIPS 2024）——&#160;"
               "主要好处是让网络能承受更大的目标学习率。"
               "这个形状的系统分析：arXiv <tspan font-weight=\"700\">2410.05192</tspan>",
               "⛔ 本图<tspan font-weight=\"700\">不含绝对学习率</tspan>（两者峰值差 3.7 倍，"
               "叠在一起就看不出形状了）——&#160;七档规模的峰值对照表写在正文里")
    f.save("fig4-lr-curve.svg", yy + 6)


main()
