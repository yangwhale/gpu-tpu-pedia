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
  · GPT-3 那一竖（S=2048）落在 **2,048**（它的斜率是 1.00，不是 V3 的 1.25）
    ——&#160;**比所有线性层都低**
    ⛔ 这一行早先写的是 2,560，那正是本文件下面记的那个 bug 的产物；
      **改了代码没改文件头**，于是错的那个数在注释里又活了几天。
  · V3 预训练那一竖（S=4096）落在 5,120 ——&#160;**比最宽的线性层 7,168 还低**
  · V3 扩训那一竖（S=131072）落在 163,840 ——&#160;**比它高 23 倍**
  ⭐⭐ 2026-09-19 T03 改：工作点从 131,072 挪到 4,096（现场定的基准），
    128K 降为虚线对比点。**翻转因此变成同一个模型内部的事**，比跨模型比更硬。
    ⛔ 别把话说过头：4K 上 attention 的比值 5.50 仍高于正文那条「小于 3」的切线，
      所以它是从「绝不」变成「边际」，**不是变成「该重算」**。

⛔⛔ 刻意没画的：
  ① **具体省多少 GiB。** 那是 §2.3 那张名次表的事，这张图只讲「谁比谁贵」。
  ② **FlashAttention。** 它让这条斜线在现代框架里根本不进候选名单 ——&#160;
     那句话写在正文里（判据⑩：挨着图的那一处赢）。
"""
import math
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

# ⭐ 全部常数集中在这儿。改图先改这里，别在文字里手写第二遍。
# ⛔⛔ 2026-09-17 红队抓到：原来只有一个 ATT_SLOPE = 1.25，那是 **V3 的**头维度算的，
#   却被拿去标 GPT-3 —— 而 GPT-3 是 qk = v = 128，斜率是 1.00，在 2,048 处是 2,048 不是 2,560。
#   ⭐ 而且对照的线性层也不该共用：V3 最宽 7,168，GPT-3 最宽 12,288。
#   ⭐⭐ 判据：**同一张图里出现两个模型时，每个模型的每一条线都要用它自己的常数。**
#     修完结论反而更强 —— 翻转靠的是斜率不同，不靠某一组具体数值。
V3_QK, V3_V = 192, 128              # DeepSeek-V3（MLA）
G3_QK, G3_V = 128, 128              # GPT-3 175B（标准 MHA，d_head = 128）
V3_SLOPE = (V3_QK + V3_V) / (V3_V * 2.0)     # causal 已折半 → 1.25
G3_SLOPE = (G3_QK + G3_V) / (G3_V * 2.0)     #              → 1.00
ATT_SLOPE = V3_SLOPE
S_GPT3 = 2048
S_V3_BASE = 4096        # ⭐ V3 预训练的真实长度 ——&#160;本讲基准，落在交点**左边**
S_V3_LONG = 131072      # 长上下文扩训 ——&#160;落在交点**右边**很远
S_V3 = S_V3_BASE        # 图上的「工作点」
V3_WIDEST, G3_WIDEST = 7168, 12288           # 各自最宽的线性层（＝ d_model）

# （输入宽度, 名字, 颜色）——&#160;每字节代价就等于输入宽度本身
# 第四项是标签靠哪边：1 右，-1 左（1,536 与 2,048 在对数轴上只差 15px，必须分开）
LINEARS = (
    (512,  "K/V 解压（输入宽 512）",        GR, 1),
    (1536, "Q 展开（输入宽 1,536）",        GR, -1),
    (2048, "专家输出（输入宽 2,048）",      OR, 1),
    (7168, "gate / up / 路由（输入宽 7,168）", RD, 1),
)
CROSS = V3_WIDEST / V3_SLOPE
G3_CROSS = G3_WIDEST / G3_SLOPE

assert abs(V3_SLOPE - 1.25) < 1e-9 and abs(G3_SLOPE - 1.00) < 1e-9
assert abs(CROSS - 5734.4) < 0.1 and abs(G3_CROSS - 12288) < 1e-6
assert abs(G3_SLOPE * S_GPT3 - 2048) < 1e-6      # ⛔ 不是 2560
assert abs(V3_SLOPE * S_V3_LONG - 163840) < 1e-6
assert abs(V3_SLOPE * S_V3_BASE - 5120) < 1e-6
assert S_GPT3 < G3_CROSS                         # GPT-3 落在自己交点左边
# ⭐⭐⭐ 这张图现在讲的是**同一个模型内部**的翻转 ——&#160;比跨模型比更有说服力：
#   V3 预训练在 4,096（交点左边），扩训推到 131,072（交点右边）。
assert S_V3_BASE < CROSS < S_V3_LONG, "V3 的两档必须夹住自己的交点，否则这张图就没得讲了"

# 画布内的坐标系（双对数）
X0, X1 = 150, 1320
LX0, LX1 = 10.0, 18.0               # log2(序列长度)：1K → 256K
LY0, LY1 = 2.6, 5.6                 # log10(每字节 FLOPs)


def main():
    f = Fig(W, "双对数坐标上，线性层的每字节重算代价是一条水平线（等于它的输入宽度），"
               "而 attention 的是一条随序列长度上升的斜线。两条斜率不同，"
               "所以必然相交：对 V3 交点在约 5,734。"
               "V3 预训练用的 4,096 落在交点左边，attention 只是普通候选；"
               "长上下文扩训推到 131,072 落到右边很远，同一个 attention 就变成最不该重算的那一个。"
               "GPT-3 的 2,048 也在它自己交点的左边，所以 2022 年那篇论文选 attention 是对的")

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
    f.t(X(26000), Y(V3_SLOPE * 26000) - 18,
        "V3 的 attention　＝　1.25 × 序列长度", PU, True, 15, "middle")

    f.path("M %.1f %.1f L %.1f %.1f" % (X(1024), Y(G3_SLOPE * 1024),
                                        X(262144), Y(G3_SLOPE * 262144)),
           OR, 2.2, arrow=False, dash="7 5")
    f.t(X(4600), Y(G3_SLOPE * 4600) + 24,
        "GPT-3 的 attention　＝　1.00 × 序列长度", OR, True, 14, "middle")
    f.line(X0, Y(G3_WIDEST), X1, Y(G3_WIDEST), OR, 1.6, dash="4 6", arrow=False)
    f.t(X0 + 10, Y(G3_WIDEST) - 10, "GPT-3 最宽的线性层（12,288）", OR, True, 13)
    gx, gy = X(G3_CROSS), Y(G3_WIDEST)
    f.box(gx - 6, gy - 6, 12, 12, "#fff", OR, 6, 2.0)
    f.t(gx, gy + 28, "GPT-3 自己的交点　12,288", OR, True, 13.5, "middle")

    # ── 交点
    cx, cy = X(CROSS), Y(7168)
    f.box(cx - 7, cy - 7, 14, 14, "#fff", INK, 7, 2.4)
    f.line(cx, cy + 10, cx, BOT, INK, 1.4, dash="4 4", arrow=False)
    f.box(cx - 112, cy - 84, 224, 62, "#fff", INK, 8, 1.6)
    f.t(cx, cy - 60, "⭐⭐ 交点　S ≈ 5,734", INK, True, 16, "middle")
    f.t(cx, cy - 38, "左边 attention 最便宜，右边最贵", GY, size=12.5, anchor="middle")

    # ── 两个真实模型的竖线
    # ⭐ 三条竖线：GPT-3、V3 预训练（实线 ＝ 本讲基准）、V3 扩训（虚线 ＝ 对比点）
    for s, nm, col, side, slope, solid in (
            (S_GPT3,     "GPT-3　2,048",           OR, -1, G3_SLOPE, False),
            (S_V3_BASE,  "V3 预训练　4,096",        RD,  1, V3_SLOPE, True),
            (S_V3_LONG,  "V3 扩训　131,072",        PU, -1, V3_SLOPE, False)):
        x = X(s)
        v = slope * s
        f.line(x, BOT, x, TOP - 4, col, 2.4 if solid else 1.8,
               dash=None if solid else "6 4", arrow=False)
        f.box(x - 6, Y(v) - 6, 12, 12, col, col, 6)
        f.t(x + side * 10, TOP + 14, nm, col, True, 14.5,
            "start" if side > 0 else "end")
        f.t(x + side * 10, TOP + 34, "→ %s FLOPs／字节" % format(int(v), ","),
            col, size=12.5, anchor="start" if side > 0 else "end")
    # ⭐ 把两档之间那段路画出来 ——&#160;翻转是沿着这根箭头发生的
    ax0, ax1 = X(S_V3_BASE), X(S_V3_LONG)
    ay = TOP + 62
    f.line(ax0 + 4, ay, ax1 - 4, ay, PU, 2.0)
    f.t((ax0 + ax1) / 2.0, ay - 10,
        "长上下文扩训：同一个模型，往右走 32 倍", PU, True, 13, "middle")
    f._pan = None

    yy = f.band(py + PH + 22, "bad", "同一条判据，结论翻转 ——　而两个模型的线<tspan text-decoration=\"underline\">都不一样</tspan>", [
        "⭐ 2022 年那篇（<tspan font-weight=\"700\">arXiv 2205.05198</tspan>）说："
        "挑「占显存不少、但重算起来不贵」的扔，<tspan font-weight=\"700\">它选中了 attention</tspan>。"
        "⛔ 看橙色那一组：GPT-3 在 2,048 处只要 <tspan font-weight=\"700\">2,048</tspan>，"
        "而它自己的交点在 <tspan font-weight=\"700\">12,288</tspan> ——&#160;"
        "<tspan font-weight=\"700\">远在左边，attention 确实最便宜。论文没错。</tspan>",
        "⭐⭐⭐ 而真正值得记的是<tspan font-weight=\"700\">红色那一组自己走的这段路</tspan>："
        "V3 <tspan font-weight=\"700\">预训练在 4,096</tspan>，每字节 "
        "<tspan font-weight=\"700\">5,120</tspan> ——&#160;"
        "<tspan font-weight=\"700\">比 gate/up 那条线还低，attention 只是个普通候选</tspan>。"
        "扩训把它推到 <tspan font-weight=\"700\">131,072</tspan>，同一个 attention 变成 "
        "<tspan font-weight=\"700\">163,840</tspan>，越过交点 5,734 <tspan font-weight=\"700\">很远</tspan>，"
        "成了最该留的那一个。<tspan font-weight=\"700\">同一个模型、同一条判据、"
        "只有一个旋钮在动 ——&#160;结论换了边。</tspan>"
        "⭐⭐ 判据一个字没改 ——&#160;<tspan font-weight=\"700\">"
        "而且注意：两个模型的斜率和对照宽度<tspan text-decoration=\"underline\">都不一样</tspan>，"
        "翻转靠的是「斜率不同必然相交」这件事本身，不靠任何一组具体数值。</tspan>",
    ], keep=True)

    yy = f.src(yy + 24,
               "⭐ 线性层那条闭式解：<tspan font-weight=\"700\">一个 [S,k]×[k,n] 的矩阵乘，"
               "每字节代价 ＝ k</tspan>（重算 2·S·k·n FLOPs ÷ 产出 S·n·2 字节，"
               "S 和 n 全约掉）——&#160;所以它在图上必然是水平线",
               "⚠️ 两条斜率分别是 <tspan font-weight=\"700\">V3 的 1.25（qk 192 / v 128）"
               "与 GPT-3 的 1.00（qk ＝ v ＝ 128）</tspan>；口径都是 causal 折半。"
               "⛔ <tspan font-weight=\"700\">换个模型就得重画它自己那两条线"
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
