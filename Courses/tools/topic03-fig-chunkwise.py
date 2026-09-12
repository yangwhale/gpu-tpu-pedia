# -*- coding: utf-8 -*-
r"""专题三 · §7.4「串行的状态怎么榨出并行度」（2026-09-13 夜间 · R13）。

⭐⭐ 这一节原来是**纯散文**，而它讲的是一件**彻底的图形化的事**：
   把一条长序列切成块，**块内并行、块间串行**。
   画出来之后，「并行度从 1 变成 C、串行步数从 L 变成 L/C」这句话
   就不需要解释了 ——&nbsp;它在图上。

⭐ 这一张还承担一个落点：**C 怎么选是一个纯硬件问题**，两头都被夹住 ——
   小了算力吃不满，大了块内中间结果放不进片上内存。
   **跟专题一 splash attention 的块大小是同一类问题**，
   而那一讲已经证明过：块大小看的是比例，不是绝对值。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "串行的状态怎么榨出并行度：逐 token 串行跑一百万步不可能；"
               "chunkwise 把序列切块，块内一次矩阵乘算完、块间只传状态；"
               "块大小两头被夹，被片上内存顶死")
    f.marks = set()
    y0 = f.header(
        "串行的状态，怎么榨出并行度　——　块内并行，块间串行",
        "⭐ 这一段是<tspan font-weight=\"700\">我们的角度</tspan>，"
        "别人的课不会这么讲：纸上是 O(L)，硬件上是一场 kernel 战争",
        [(RD, "串行：加速器最怕的"), (GR, "块内：一次矩阵乘"),
         (BL, "块间：只传一个状态"), (OR, "两头夹住块大小")])

    ph = 380

    # ══ ① 问题 ══════════════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 问题：它天生是串行的", RD,
                 sub="加速器最不喜欢的形状")

    yy = py + 30
    # 一串小方块 + 首尾相连的箭头
    for i in range(11):
        bx = x + 26 + i * 34
        f.box(bx, yy, 26, 26, "#fff", RD if i < 3 else LINE, 4)
        if i < 10:
            f.line(bx + 27, yy + 13, bx + 32, yy + 13, GY2, 1.1)
    f.t(x + 26, yy + 48, "S 依赖上一个 S，一个 token 一步", RD, True, 12.5,
        w=pw - 52)
    yy += 70

    f.box(x + 22, yy, pw - 44, 78, "#fff", RD, 8)
    f.box(x + 22, yy, 4, 78, RD, RD, 2)
    f.box(x + 24, yy, 3, 78, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "训一条一百万 token 的序列", RD, True, 12.5)
    f.t(x + 40, yy + 48, "＝ 串行跑一百万步 ——", GY, size=11.5)
    f.t(x + 40, yy + 68, "<tspan font-weight=\"700\">直接不用想</tspan>", RD, True, 12.5)
    yy += 92

    f.box(x + 22, yy, pw - 44, 96, "#fff", LINE, 8)
    f.t(x + 38, yy + 25, "⭐ 这里有个反讽值得点一句", INK, True, 12.5)
    f.t(x + 38, yy + 48, "第三个旋钮本来是为了<tspan font-weight=\"700\">省</tspan>", GY, size=11.5)
    f.t(x + 38, yy + 68, "——&#160;把平方降成线性；", GY, size=11.5)
    f.t(x + 38, yy + 88, "结果<tspan font-weight=\"700\">先丢掉的是并行度</tspan>。", INK, True, 12.5)
    fits(yy + 96, y0, ph, "①")

    # ══ ② chunkwise ═════════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 解法：块内并行，块间串行", GR,
                 sub="chunkwise parallel")

    yy = py + 28
    # 三个块，每块内部一排方块，块之间一条状态箭头
    C = 4
    bw = 4 * 26 + 3 * 5
    for b in range(3):
        bx = x + 26 + b * (bw + 40)
        f.box(bx - 6, yy - 6, bw + 12, 44, "#fff", GR, 6)
        for i in range(C):
            f.box(bx + i * 31, yy, 26, 26, "#e6f4ea", "none", 4)
        f.t(bx + bw / 2.0 - 6, yy + 52, "块 %d" % (b + 1), GR, True, 11.5,
            "middle")
        if b < 2:
            f.line(bx + bw + 8, yy + 14, bx + bw + 30, yy + 14, BL, 1.6)
    f.t(x + 26, yy + 76, "块内：一次矩阵乘算完，<tspan font-weight=\"700\">全并行</tspan>", GR, True, 12,
        w=pw - 52)
    f.t(x + 26, yy + 96, "块间：只把<tspan font-weight=\"700\">一个状态</tspan>传下去", BL, True, 12,
        w=pw - 52)
    yy += 116

    for lab, before, after in [
        ("并行度", "1", "C（块长）"),
        ("串行步数", "L", "L / C"),
    ]:
        f.box(x + 22, yy, pw - 44, 46, "#fff", LINE, 8)
        f.t(x + 38, yy + 28, lab, INK, True, 12)
        f.t(x + 150, yy + 28, before, GY2, size=12, mono=True)
        f.line(x + 196, yy + 24, x + 228, yy + 24, GY2, 1.2)
        f.t(x + 240, yy + 28, after, GR, True, 12.5, mono=True)
        yy += 54

    yy += 2
    f.t(x + 22, yy, "⭐ 注意它<tspan font-weight=\"700\">没有改数学</tspan> —— 算出来的结果", GY, size=11.5,
        w=pw - 44)
    f.t(x + 22, yy + 20, "跟逐 token 串行<tspan font-weight=\"700\">一模一样</tspan>，只是换了个算法。",
        GY, size=11.5, w=pw - 44)
    fits(yy + 26, y0, ph, "②")

    # ══ ③ C 怎么选 ══════════════════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 块长 C 怎么选：两头都被夹住", OR,
                 sub="⭐ 这是一个纯硬件问题")

    yy = py + 26
    for side, why1, why2, col in [
        ("C 太小", "串行步数多；而且每步那个矩阵乘<tspan font-weight=\"700\">太瘦</tspan>",
         "算力吃不满", RD),
        ("C 太大", "块内那个中间矩阵<tspan font-weight=\"700\">放不进片上内存</tspan>",
         "只好往 HBM 上倒，白搭", RD),
    ]:
        f.box(x + 22, yy, pw - 44, 76, "#fff", col, 8)
        f.box(x + 22, yy, 4, 76, col, col, 2)
        f.box(x + 24, yy, 3, 76, "#fff", "#fff", 0)
        f.t(x + 40, yy + 25, side, col, True, 13)
        f.t(x + 40, yy + 48, why1, GY, size=11.5, w=pw - 76)
        f.t(x + 40, yy + 68, why2, GY2, size=11)
        yy += 86

    f.box(x + 22, yy, pw - 44, 78, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 78, OR, OR, 2)
    f.box(x + 24, yy, 3, 78, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐ 所以块大小是<tspan font-weight=\"700\">被片上内存顶死的</tspan>", OR,
        True, 12.5)
    f.t(x + 40, yy + 48, "跟专题一 splash attention 的块大小", GY, size=11.5)
    f.t(x + 40, yy + 68, "是同一类问题 ——&#160;那边已经证过一次", GY2,
        size=11)
    yy += 92

    f.t(x + 22, yy, "⚠️ 而且那边的结论也能搬过来：", GY, size=11.5)
    f.t(x + 22, yy + 20, "<tspan font-weight=\"700\">块大小看的是比例，不是绝对值。</tspan>", INK, True,
        12.5, w=pw - 44)
    fits(yy + 26, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 这条路有多难走，看两个信号就够了", [
        "① Kimi 为了 KDA <tspan font-weight=\"700\">专门写了一个 CUTLASS kernel</tspan>"
        "（FlashKDA）——&#160;因为块内并行和块间串行"
        "<tspan font-weight=\"700\">交替进行的时候 SM 会空转</tspan>。",
        "② 长序列训练还得有 <tspan font-weight=\"700\">KDA 专用的上下文并行</tspan>："
        "标准做法是把各段的局部结果直接相加，"
        "<tspan font-weight=\"700\">但这对 delta rule 不成立</tspan> ——&#160;"
        "它的状态更新是 token 相关的矩阵连乘，前一段的影响不是简单的加法。",
        "⭐ 一句话收：<tspan font-weight=\"700\">线性注意力在纸上是 O(L)，"
        "在硬件上是一场 kernel 战争。</tspan>",
    ])

    yy = f.src(yy + 16,
               "chunkwise parallel 的形式见 DeltaNet 并行化那篇 arXiv 2406.06484 §3；"
               "FlashKDA 与 KDA 上下文并行见 Kimi Linear arXiv 2510.26692",
               "「块大小被片上内存顶死」「看比例不看绝对值」两条，"
               "与专题一 splash attention 那一节同源")
    f.save("fig3-chunkwise.svg", yy + 6)


main()
