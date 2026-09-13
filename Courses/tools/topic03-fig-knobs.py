# -*- coding: utf-8 -*-
r"""专题三 · §四「三个旋钮」骨架图

⭐⭐⭐ 2026-09-14 **整张重画** ——&nbsp;既然叫「旋钮」，那就**真的画三个旋钮**。

   这张图要做的**不是罗列三个旋钮**，是**证明为什么恰好是三个**：
   一个 query 从头到尾只做三件事，**每件事对应一个旋钮**，
   ⭐ 看完应该得到的是「**没有第四个位置可以拧**」这个封闭感。

   画面：**一台机器的控制面板，上面只有三个旋钮。**
   每个旋钮下面画出「拧过去之后，那一格变成什么样」——&nbsp;
   ①「每份变小」② 「读的变少」③ 「整块换掉」。

📌 FlashAttention 画成面板外面的一个**灰色开关**：
   ⛔ 它不改算什么，只改怎么算 ——&nbsp;**所以它根本不在这块面板上。**
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    f = Fig(W, "三个旋钮：一个 query 从头到尾只做三件事 —— 取出一份 KV、"
               "跟哪些位置算、用什么数学；每件事对应一个旋钮，"
               "所以没有第四个位置可以拧；FlashAttention 不在这块面板上")
    f.marks = set()
    y0 = f.header(
        "三个旋钮 ——　这块面板上就这么三个",
        "⭐ 不是「有三类方法」，是<tspan font-weight=\"700\">"
        "一个 query 只做三件事，所以只有三个地方能拧</tspan>",
        [(BL, "存什么"), (OR, "读哪些"), (PU, "用什么数学"),
         (GY, "不在面板上")])

    # ══════════ ① 一个 query 只做三件事 ═════════════════════════
    PH = 236
    py = f.panel(0, y0, W, PH, "① 先看一个 query 从头到尾做了什么",
                 INK, sub="只有三步 ——　这就是「只有三个旋钮」的全部理由")

    ay = py + 30
    STEPS = [
        (BL, "第一步", "从每个位置各取一份", "K 和 V"),
        (OR, "第二步", "跟其中哪些位置算", "然后加权求和"),
        (PU, "第三步", "用哪一套数学", "算这个加权求和"),
    ]
    for i, (col, no, a, b) in enumerate(STEPS):
        bx = 96 + i * 420
        f.box(bx, ay + 18, 356, 118, "#fff", col, 10)
        f.t(bx + 24, ay + 56, no, GY2, True, 16)
        f.t(bx + 24, ay + 90, a, col, True, 22)
        f.t(bx + 24, ay + 122, b, GY, size=17)
        if i < 2:
            f.line(bx + 362, ay + 78, bx + 412, ay + 78, GY2, 1.8)
    f.t(96, ay + 176, "⛔ 数一数 ——　<tspan font-weight=\"700\">就这三步，没有第四步</tspan>。"
        "所以下面那块面板上，也只可能有三个旋钮。", INK, True, 21)

    # ══════════ ② 三个旋钮 ══════════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 388
    py2 = f.panel(0, y1, W, PH2, "② 于是面板上就这三个旋钮", GR,
                  sub="每个旋钮下面，画的是「拧过去之后那一格变成什么样」")

    by = py2 + 24
    KNOBS = [
        (BL, "旋钮 ①", "每个 token 存多少",
         "MQA → GQA → MLA → Gated MLA", "每份变小", "small"),
        (OR, "旋钮 ②", "每个 query 读多少",
         "SWA · NSA · DSA · CSA / HCA", "读的变少", "few"),
        (PU, "旋钮 ③", "换一套数学",
         "线性注意力：DeltaNet → GDN → KDA", "整块换掉", "swap"),
    ]
    for i, (col, no, what, who, effect, kind) in enumerate(KNOBS):
        bx = 96 + i * 420
        f.box(bx, by + 20, 356, 304, "#fff", col, 12)
        # 旋钮本体：一个圆 ＋ 一根指针
        cx, cy2, r = bx + 178, by + 92, 42
        f.spot(cx - r, cy2 - r, 2 * r, 2 * r, "#f1f3f4")
        f.box(cx - r, cy2 - r, 2 * r, 2 * r, "none", col, r, 2.4)
        ang = (-0.9 + i * 0.9)
        f.line(cx, cy2, cx + r * 0.72 * math.sin(ang),
               cy2 - r * 0.72 * math.cos(ang), col, 4.0, arrow=False)
        f.t(cx, cy2 + r + 30, no, col, True, 24, "middle")
        f.t(cx, cy2 + r + 58, what, INK, True, 20, "middle")

        # 拧过去之后那一格长什么样
        gy_ = by + 212
        f.t(bx + 24, gy_ - 6, "拧过去之后：" + effect, col, True, 18)
        if kind == "small":
            for k in range(6):
                f.box(bx + 28 + k * 54, gy_ + 8, 44, 44, BG2, LINE2, 4)
                f.box(bx + 42 + k * 54, gy_ + 22, 16, 16, col, "none", 3)
            f.t(bx + 24, gy_ + 78, "格子还是那么多，每格里的东西变小", GY,
                size=16)
        elif kind == "few":
            for k in range(6):
                on = k in (1, 4)
                f.box(bx + 28 + k * 54, gy_ + 8, 44, 44,
                      col if on else BG2, "none" if on else LINE2, 4)
            f.t(bx + 24, gy_ + 78, "每格还是那么大，这一步只读其中几个", GY,
                size=16)
        else:
            f.box(bx + 28, gy_ + 8, 152, 44, BG2, LINE2, 4)
            f.t(bx + 104, gy_ + 36, "一长排", GY2, size=16, anchor="middle")
            f.line(bx + 188, gy_ + 30, bx + 214, gy_ + 30, col, 2.0)
            f.box(bx + 222, gy_ + 8, 106, 44, "#f3e8fd", col, 4)
            f.t(bx + 275, gy_ + 36, "一块板子", col, True, 17, "middle")
            f.t(bx + 24, gy_ + 78, "整排换成一块固定大小的板子", GY, size=16)

    # ══════════ ③ 不在面板上的那个 ══════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 188
    py3 = f.panel(0, y2, W, PH3, "③ 那 FlashAttention 呢 ——　它不在这块面板上",
                  GY, sub="它改的不是算什么，是怎么算")

    ey = py3 + 22
    f.box(56, ey + 20, 620, 116, "#f1f3f4", GY2, 10)
    f.t(80, ey + 60, "它是机器侧面的一个开关", GY, True, 22)
    f.t(80, ey + 96, "打开：同样的结果，少搬很多次；关上：一样算得出来", GY,
        size=17)
    f.t(80, ey + 128, "⛔ 它一个字节的 KV 都不省", RD, True, 19)

    f.box(708, ey + 20, 652, 116, "#fff", INK, 10)
    f.t(732, ey + 60, "⭐ 判据：一个在所有分支上取值都一样的变量", INK, True, 21)
    f.t(732, ey + 96, "对这一讲<tspan font-weight=\"700\">没有解释力</tspan> ——&#160;"
        "三个旋钮怎么拧，它都在那儿，而且都一样有用。", GY, size=17)
    f.t(732, ey + 128, "所以它不是第四个旋钮，它是另一层的事", GY2, size=16)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这张图要留下的是「封闭感」，不是「有三类方法」", [
        "<tspan font-weight=\"700\">一个 query 只做三件事 → 只有三个地方能拧。</tspan>"
        "拿到任何一个新名字，先把它放进某一格 ——&#160;"
        "放得进去的，它的优点和代价<tspan font-weight=\"700\">你已经知道了</tspan>；"
        "<tspan font-weight=\"700\">放不进去的，才值得你花时间。</tspan>",
        "⚠️ 一定会被问的那一条：<tspan font-weight=\"700\">KV 量化算不算第四个旋钮？</tspan>"
        "——&#160;不算。三个旋钮管的是<tspan font-weight=\"700\">存几个数、读几个数</tspan>，"
        "量化管的是<tspan font-weight=\"700\">每个数用几个 bit</tspan>。"
        "两者正交，可以任意组合（V4 就是稀疏 ＋ KV 混合精度一起上）。精度整个归专题八。",
    ])

    yy = f.src(yy + 16,
               "三步的拆法与三个旋钮的对应关系是<tspan font-weight=\"700\">本课的骨架</tspan>，"
               "不是某一篇论文的分类；每个旋钮下面的代表方法见 §五 / §六 / §七 各自的出处",
               "⚠️ 「控制面板 / 旋钮」是本课的比喻 ——&#160;"
               "它承担的是「封闭性」这个论证，不只是一个好记的名字")
    f.save("fig3-knobs.svg", yy + 6)


main()
