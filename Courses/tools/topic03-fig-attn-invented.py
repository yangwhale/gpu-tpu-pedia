# -*- coding: utf-8 -*-
r"""专题三 · §1.0「注意力是怎么被发明出来的」

⭐⭐⭐ 2026-09-13 **整张重画**（现场：「图是图，字是字 ——&nbsp;把原理画出来」）。
   上一版是三栏文字，一千三百多字。这一版**三格三个画面**，
   每格只回答一句「上一步哪儿不对」，字大、字少。

   ① **2014 ——&nbsp;一个向量装不下。**
      画面：一整句话被挤进**一个小方块**，再由它生成整段译文。
      ⭐ 关键那一步是「**软**」：硬挑一个词不可导，
      softmax 加权平均可导 ——&nbsp;于是对齐能**跟翻译模型一起学出来**。
   ② **2017 第一刀 ——&nbsp;打分函数太慢。**
      画面：左边**每一对都要过一个小网络**（加性），
      右边**整张表一次矩阵乘**（点积）。
      ⭐⭐ 论文原话：两者理论复杂度相仿，但点积**快得多、省内存**，
      因为**能用高度优化的矩阵乘实现** ——&nbsp;
      **选点积不是因为它更准，是因为它能变成矩阵乘。**
      这是本课「硬件反过来决定公式」的第一个例子，而且是论文原话。
   ③ **2017 第二刀 ——&nbsp;既然能直连，还要 RNN 干什么。**
      画面：一条**要走 n 步**的链，对上一张**任意两点一步可达**的表。
      ⚠️ 同一句话里论文就认了代价：**加权平均降低了有效分辨率**，
      「an effect we counteract with Multi-Head Attention」——&nbsp;
      **多头是来补偿这个代价的**，不是「多个视角」这种营销词。
"""
from topic03_draw import (Fig, BL, GR, RD, GY, PU, INK, GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]
SRC = ["我", "今天", "很", "开心"]


def main():
    f = Fig(W, "注意力是被三次「这儿不对」推出来的：2014 年一个向量装不下整句，"
               "于是软对齐；2017 年加性打分每一对都要过小网络太慢，换成点积因为"
               "能写成矩阵乘；再去掉循环，多头用来补偿加权平均的分辨率损失")
    f.marks = set()
    y0 = f.header(
        "注意力是怎么被发明出来的",
        "三步 ——&#160;<tspan font-weight=\"700\">每一步都在修上一步的一个具体毛病</tspan>",
        [(RD, "上一步哪儿不对"), (GR, "这一步怎么修"),
         (PU, "被硬件推着走的那一刀")])

    PH = 408

    # ══════════ ① 2014：一个向量装不下 ═══════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, PH, "① 2014　一个向量装不下", BL,
                 sub="Bahdanau / Cho / Bengio")

    yy = py + 24
    # 上：旧做法 —— 整句挤进一个小方块
    for i, w in enumerate(SRC):
        f.box(x + 22 + i * 74, yy, 66, 38, BG2, LINE2, 6)
        f.t(x + 55 + i * 74, yy + 25, w, GY, True, 16, "middle")
    for i in range(4):
        f.line(x + 55 + i * 74, yy + 42, x + 202, yy + 62, GY2, 1.1)
    f.box(x + 168, yy + 66, 68, 40, "#fce8e6", RD, 8)
    f.t(x + 202, yy + 92, "一个", RD, True, 17, "middle")
    f.t(x + 202, yy + 124, "向量", RD, True, 17, "middle")
    f.line(x + 202, yy + 132, x + 202, yy + 156, GY2, 1.4)
    f.box(x + 100, yy + 160, 204, 40, "#fff", GY, 8)
    f.t(x + 202, yy + 186, "整段译文都从这儿出", GY, True, 16, "middle")
    f.t(x + 22, yy + 228, "⛔ 句子越长，挤得越狠", RD, True, 18)
    # ⛔⛔ 2026-09-14 R4 核对一手论文时抓到的：原文是
    #   「**we conjecture that** the use of a fixed-length vector is a bottleneck」
    #   ——&#160;这是作者自己下的**推测**，不是已证的结论。
    #   ⭐ 判据（CLAUDE.md 第一原则那条的引申）：**引用要连确定性一起引。**
    #     把 "we conjecture" 吃掉，等于替作者把话说满了。
    f.t(x + 22, yy + 254, "原文：we conjecture that … a fixed-", GY2, size=13)
    f.t(x + 22, yy + 274, "length vector is a bottleneck（作者自己的推测）",
        GY2, size=13)

    f.box(x + 22, yy + 290, pw - 44, 64, "#fff", GR, 8)
    f.t(x + 38, yy + 316, "⭐ 修法的关键是那个「软」字", GR, True, 17)
    f.t(x + 38, yy + 342, "硬挑一个词不可导；加权平均可导", GY, size=15)

    # ══════════ ② 2017 第一刀：打分函数 ══════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, PH, "② 2017 第一刀　打分太慢", GR,
                 sub="加性 → 点积")

    yy = py + 24
    f.t(x + 22, yy + 18, "加性（旧）", RD, True, 19)
    f.t(x + 22, yy + 42, "每一对，过一个小网络", GY, size=15)
    for r in range(3):
        for c in range(3):
            cx, cy2 = x + 34 + c * 62, yy + 56 + r * 46
            f.box(cx, cy2, 50, 34, "#fff", RD, 5)
            f.t(cx + 25, cy2 + 22, "net", RD, size=13, anchor="middle")
    f.t(x + 232, yy + 122, "n² 次", RD, True, 20)
    f.t(x + 232, yy + 148, "小网络", RD, True, 20)

    f.line(x + 22, yy + 208, x + pw - 44, yy + 208, LINE, 1, arrow=False)

    f.t(x + 22, yy + 238, "点积（今天）", GR, True, 19)
    f.t(x + 22, yy + 262, "整张表，一次矩阵乘", GY, size=15)
    f.box(x + 34, yy + 276, 152, 76, "#e6f4ea", GR, 6)
    f.t(x + 110, yy + 322, "Q · Kᵀ", GR, True, 24, "middle")
    f.t(x + 206, yy + 322, "一次", GR, True, 20)

    # ══════════ ③ 2017 第二刀：去掉循环 ══════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, PH, "③ 2017 第二刀　那还要 RNN 干什么", PU,
                 sub="两点之间要走几步")

    yy = py + 24
    f.t(x + 22, yy + 18, "循环：一步一步传", RD, True, 19)
    for i in range(5):
        f.box(x + 26 + i * 82, yy + 36, 62, 40, "#fff", RD, 6)
        f.t(x + 57 + i * 82, yy + 62, str(i + 1), RD, True, 18, "middle")
        if i < 4:
            f.line(x + 90 + i * 82, yy + 56, x + 104 + i * 82, yy + 56, RD, 1.4)
    f.t(x + 22, yy + 102, "①→⑤ 要走 4 步", RD, True, 18)

    f.line(x + 22, yy + 124, x + pw - 44, yy + 124, LINE, 1, arrow=False)

    f.t(x + 22, yy + 154, "自注意力：直接连", GR, True, 19)
    for i in range(5):
        f.spot(x + 26 + i * 82, yy + 172, 62, 40, "#e6f4ea")
        f.t(x + 57 + i * 82, yy + 198, str(i + 1), GR, True, 18, "middle")
    f.path([(x + 57, yy + 170), (x + 220, yy + 138), (x + 385, yy + 170)],
           GR, 1.8)
    f.t(x + 22, yy + 238, "①→⑤ 一步", GR, True, 18)

    f.box(x + 22, yy + 262, pw - 44, 92, "#fff", PU, 8)
    f.t(x + 38, yy + 288, "⚠️ 同一句话里，论文认了代价", PU, True, 17)
    f.t(x + 38, yy + 314, "加权平均<tspan font-weight=\"700\">降低了有效分辨率</tspan>", GY,
        size=15)
    f.t(x + 38, yy + 340, "多头就是拿来补这个的", PU, True, 17)

    # ══════════ 落点带 ═══════════════════════════════════════════
    yy = y0 + PH + 22
    yy = f.band(yy, "info", "⭐⭐ 第二格那一刀，是这门课的主线第一次出现", [
        "论文自己写的理由是：两者<tspan font-weight=\"700\">理论复杂度相仿</tspan>，"
        "但点积<tspan font-weight=\"700\">快得多、省内存，因为它能用高度优化的矩阵乘实现</tspan>。",
        "⭐ 换句话说 ——&#160;<tspan font-weight=\"700\">选点积不是因为它更准，"
        "是因为它能变成矩阵乘。</tspan>"
        "这一讲后面每一个变体，几乎都能追到同一句话上。",
    ])

    yy = f.src(yy + 14,
               "① Bahdanau 等 arXiv 1409.0473；②③ Vaswani 等 arXiv 1706.03762 "
               "§3.2.1（点积 vs 加性）与 §3.2.2（多头补偿分辨率）",
               "⚠️ 画面里的句子、方框数量都是<tspan font-weight=\"700\">示意</tspan>，"
               "不对应任何一次真实实验")
    f.save("fig3-attn-invented.svg", yy + 6)


main()
