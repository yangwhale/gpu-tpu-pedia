# -*- coding: utf-8 -*-
r"""专题四 · §2.4「两条斜率不同的线，必然相交」

⭐⭐⭐ 2026-09-16 新画。现场原话：「那张交叉点图我觉得最值钱」——&#160;确实。
  «两条斜率不同的线必然相交» 这句话，**用图讲一秒就懂，用字讲要一整段**。

⭐⭐ 这张图是全专题第一张**真正的函数图**（前面都是框和箭头）。
  ⛔ 所以坐标必须是**双对数**，不是为了好看，是因为：
    ① 纵轴要同时容下 512 和 163,840 ——&#160;跨 320 倍；
    ② 只有在 log-log 上，「常数」才是水平线、「正比于 S」才是直线。
    **换成线性轴，这张图想讲的那个「必然相交」当场就看不见了。**

⭐ 图上每一个数都能当场验：
  · 线性层每字节代价 ＝ 它的输入宽度（§2.3 的闭式解，S 和 n 全约掉）
  · attention 每字节代价 ＝ 1.25 · S（V3 头维度 192/128，causal 已折半）
  · 交点 1.25·S = 7168 →&#160;S ≈ 5,734
  · GPT-3 那一竖（S=2048）落在 2,560 ——&#160;**比所有线性层都低**
  · V3 那一竖（S=131072）落在 163,840 ——&#160;**比最贵的线性层还高 23 倍**

⛔⛔ 刻意没画的：
  ① **具体省多少 GiB。** 那是 §2.3 那张名次表的事，这张图只讲「谁比谁贵」。
  ② **FlashAttention。** 它让这条斜线在现代框架里根本不进候选名单 ——&#160;
     那句话写在正文里（判据⑩：挨着图的那一处赢）。
"""
import math
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

# ⭐ 全部常数集中在这儿。改图先改这里，别在文字里手写第二遍。
HEAD_QK, HEAD_V = 192, 128          # V3 的头维度
ATT_SLOPE = (HEAD_QK + HEAD_V) / (HEAD_V * 2.0)   # causal 已折半 → 1.25
S_GPT3, S_V3 = 2048, 131072

# （输入宽度, 名字, 颜色）——&#160;每字节代价就等于输入宽度本身
# 第四项是标签靠哪边：1 右，-1 左（1,536 与 2,048 在对数轴上只差 15px，必须分开）
LINEARS = (
    (512,  "K/V 解压（输入宽 512）",        GR, 1),
    (1536, "Q 展开（输入宽 1,536）",        GR, -1),
    (2048, "专家输出（输入宽 2,048）",      OR, 1),
    (7168, "gate / up / 路由（输入宽 7,168）", RD, 1),
)
CROSS = 7168 / ATT_SLOPE            # 跟最贵那条线性层的交点

assert abs(ATT_SLOPE - 1.25) < 1e-9
assert abs(CROSS - 5734.4) < 0.1
assert abs(ATT_SLOPE * S_GPT3 - 2560) < 1e-6
assert abs(ATT_SLOPE * S_V3 - 163840) < 1e-6

# 画布内的坐标系（双对数）
X0, X1 = 150, 1320
LX0, LX1 = 10.0, 18.0               # log2(序列长度)：1K → 256K
LY0, LY1 = 2.6, 5.6                 # log10(每字节 FLOPs)


def main():
    f = Fig(W, "双对数坐标上，线性层的每字节重算代价是一条水平线（等于它的输入宽度），"
               "而 attention 的是一条随序列长度上升的斜线。两条斜率不同，"
               "所以必然相交：对 V3 交点在约 5,734。"
               "GPT-3 的 2,048 落在交点左边，所以 2022 年那篇论文选 attention 是对的；"
               "V3 的 131,072 落在右边很远，所以 attention 变成最不该重算的那一个")

    y0 = f.header(
        "为什么同一条判据会给出<tspan font-weight=\"700\">相反</tspan>的答案"
        "　——　<tspan font-weight=\"700\">两条斜率不同的线，必然相交</tspan>",
        "⭐ 纵轴：<tspan font-weight=\"700\">扔掉它、再算回来 ——&#160;一个字节的代价</tspan>"
        "　·　⚠️ 双对数坐标 ——&#160;<tspan font-weight=\"700\">"
        "常数在这里是水平线，正比于 S 的是直线</tspan>",
        [(GR, "便宜 → 该扔"), (RD, "贵 → 该留"), (PU, "attention")])

    PH = 520
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 横轴是序列长度 ——　"
                 "<tspan font-weight=\"700\">整张图只有这一个变量在动</tspan>", BL,
                 sub="⭐ 线性层那几条是<tspan font-weight=\"700\">水平的</tspan>："
                     "它们的代价跟序列长度<tspan font-weight=\"700\">无关</tspan>")

    TOP, BOT = py + 44, py + 44 + 372

    def X(s):
        return X0 + (math.log(s, 2) - LX0) / (LX1 - LX0) * (X1 - X0)

    def Y(v):
        return BOT - (math.log10(v) - LY0) / (LY1 - LY0) * (BOT - TOP)

    # ── 坐标轴
    f.line(X0, BOT, X1 + 10, BOT, GY2, 2.0, arrow=False)
    f.line(X0, BOT, X0, TOP - 10, GY2, 2.0, arrow=False)
    for s in (1024, 4096, 16384, 65536, 262144):
        x = X(s)
        f.line(x, BOT, x, BOT + 6, GY2, 1.2, arrow=False)
        f.t(x, BOT + 26, "%dK" % (s // 1024), GY2, size=12.5, anchor="middle")
    f.t((X0 + X1) / 2.0, BOT + 54, "序列长度（token）", GY, True, 14, "middle")
    for v in (1000, 10000, 100000):
        y = Y(v)
        f.line(X0 - 6, y, X1, y, LINE2, 0.8, dash="3 6", arrow=False)
        f.t(X0 - 14, y + 5, "%dK" % (v // 1000), GY2, size=12, anchor="end")
    f.t(X0 - 14, TOP - 22, "FLOPs／字节", GY, True, 13, "end")

    # ── 线性层：四条水平线（每字节代价 ＝ 输入宽度）
    for kk, nm, col, sd in LINEARS:
        y = Y(kk)
        f.line(X0, y, X1, y, col, 2.0, arrow=False)
        f.t(X1 - 8 if sd > 0 else X0 + 10, y - 10 if sd > 0 else y + 20,
            nm, col, True, 13.5, "end" if sd > 0 else "start")

    # ── attention：一条斜线 v = 1.25·S
    f.path("M %.1f %.1f L %.1f %.1f" % (X(1024), Y(ATT_SLOPE * 1024),
                                        X(262144), Y(ATT_SLOPE * 262144)),
           PU, 3.2, arrow=False)
    f.t(X(26000), Y(ATT_SLOPE * 26000) - 18,
        "attention　＝　1.25 × 序列长度", PU, True, 15, "middle")

    # ── 交点
    cx, cy = X(CROSS), Y(7168)
    f.box(cx - 7, cy - 7, 14, 14, "#fff", INK, 7, 2.4)
    f.line(cx, cy + 10, cx, BOT, INK, 1.4, dash="4 4", arrow=False)
    f.box(cx - 112, cy - 84, 224, 62, "#fff", INK, 8, 1.6)
    f.t(cx, cy - 60, "⭐⭐ 交点　S ≈ 5,734", INK, True, 16, "middle")
    f.t(cx, cy - 38, "左边 attention 最便宜，右边最贵", GY, size=12.5, anchor="middle")

    # ── 两个真实模型的竖线
    for s, nm, col, side in ((S_GPT3, "GPT-3　2,048", OR, -1),
                             (S_V3, "V3　131,072", RD, -1)):
        x = X(s)
        v = ATT_SLOPE * s
        f.line(x, BOT, x, TOP - 4, col, 1.8, dash="6 4", arrow=False)
        f.box(x - 6, Y(v) - 6, 12, 12, col, col, 6)
        f.t(x + side * 10, TOP + 14, nm, col, True, 14.5,
            "start" if side > 0 else "end")
        f.t(x + side * 10, TOP + 34, "→ %s FLOPs／字节" % format(int(v), ","),
            col, size=12.5, anchor="start" if side > 0 else "end")
    f._pan = None

    yy = f.band(py + PH + 22, "bad", "同一条判据，结论翻转 ——　变的只有那一根竖线", [
        "⭐ 2022 年那篇（<tspan font-weight=\"700\">arXiv 2205.05198</tspan>）说："
        "挑「占显存不少、但重算起来不贵」的扔。"
        "<tspan font-weight=\"700\">它选中了 attention</tspan> ——&#160;"
        "因为在 2,048 上，attention 只要 2,560，"
        "<tspan font-weight=\"700\">比图上每一条线性层都低</tspan>。论文没错。",
        "⛔ 到 131,072，同一条斜线爬到 <tspan font-weight=\"700\">163,840</tspan> ——&#160;"
        "比最贵的那条线性层还高 <tspan font-weight=\"700\">23 倍</tspan>。"
        "于是<tspan font-weight=\"700\">它从最该扔的变成最该留的</tspan>。"
        "⭐⭐ <tspan font-weight=\"700\">判据一个字没改，翻转的是前提。</tspan>",
    ], keep=True)

    yy = f.src(yy + 24,
               "⭐ 线性层那条闭式解：<tspan font-weight=\"700\">一个 [S,k]×[k,n] 的矩阵乘，"
               "每字节代价 ＝ k</tspan>（重算 2·S·k·n FLOPs ÷ 产出 S·n·2 字节，"
               "S 和 n 全约掉）——&#160;所以它在图上必然是水平线",
               "⚠️ attention 那条斜率 1.25 依赖 <tspan font-weight=\"700\">V3 的头维度"
               "（qk 192 / v 128）与 causal 折半的口径</tspan>；"
               "换个模型斜率会变，<tspan font-weight=\"700\">但「它是斜的」不会变</tspan>"
               " ——&#160;这才是要记的东西",
               "⛔ 交点 5,734 <tspan font-weight=\"700\">不是工程阈值</tspan>，"
               "它还取决于你拿哪条线性层做对照（图上四条给的交点各不相同）",
               "📌 2022 那篇的收益数字（GPT-3 省 70% 付 2.7%）用的是"
               "<tspan font-weight=\"700\">未折半的 Megatron 口径</tspan>；"
               "换成本图统一的 causal 口径，GPT-3 的 attention 占比是 1.37%")
    f.save("fig4-per-byte.svg", yy + 6)


main()
