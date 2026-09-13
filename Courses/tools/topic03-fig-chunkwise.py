# -*- coding: utf-8 -*-
r"""专题三 · §7.4「串行的状态怎么榨出并行度」

⭐⭐⭐ 2026-09-14 **整张重画**，换成一个人人排过队的画面：
   **一百万个人要办事，只有一个窗口 vs 分成一批一批地办。**

   ① **逐 token 串行 ＝ 一个窗口、一个一个来** ——&nbsp;一百万步，没法并行。
   ② **chunkwise ＝ 分批** ——&nbsp;一批人**同时办**（块内并行），
      办完这一批，**只把一张交接单传给下一批**（块间串行）。
      ⭐ 画出来就不用解释：**并行度 1 → C，串行步数 L → L/C。**
   ③ **一批该放多少人，是个纯硬件问题** ——&nbsp;两头都被夹住：
      人少了柜台空着（算力吃不满），人多了大厅站不下（片上内存放不下）。
      ⭐ 跟专题一 splash attention 的块大小是同一类问题，
      而那一讲已经证过：**块大小看的是比例，不是绝对值。**
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    L, C = 24, 6                       # 示意：24 个 token，切成每块 6 个
    assert L % C == 0

    f = Fig(W, "串行的状态怎么榨出并行度：一个窗口一个一个办要一百万步；"
               "分批办就是块内并行、块间只传一张交接单；"
               "一批放多少人是纯硬件问题，人少了柜台空着，人多了大厅站不下")
    f.marks = set()
    y0 = f.header(
        "串行的状态，怎么榨出并行度",
        "把它想成<tspan font-weight=\"700\">排队办事</tspan>："
        "一个窗口一个一个来，还是<tspan font-weight=\"700\">分批办</tspan>",
        [(RD, "一个一个来"), (GR, "分批办"), (OR, "一批放多少人")])

    # ══════════ ① 一个窗口 ══════════════════════════════════════
    PH = 226
    py = f.panel(0, y0, W, PH, "① 逐 token 跑 ——　一个窗口，一个一个来",
                 RD, sub="这就是递推本身")

    ay = py + 26
    for i in range(L):
        x = 56 + i * 54
        f.box(x, ay + 30, 40, 40, "#fce8e6", RD, 5)
        f.t(x + 20, ay + 56, str(i + 1), RD, True, 15, "middle")
        if i < L - 1:
            f.line(x + 42, ay + 50, x + 52, ay + 50, RD, 1.2)
    f.t(56, ay + 20, "每一步都要等上一步的结果", RD, True, 20)
    f.t(56, ay + 108, "⛔ 一百万个 token ＝ <tspan font-weight=\"700\">一百万步，"
        "一步都不能并</tspan>", RD, True, 21)
    f.t(56, ay + 142, "加速器最怕这个 ——&#160;几千个算力单元，一次只喂得上一个", GY,
        size=17)

    # ══════════ ② 分批办 ════════════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 336
    py2 = f.panel(0, y1, W, PH2, "② chunkwise ——　分批办：一批同时办，只把交接单传下去",
                  GR, sub="块内并行，块间串行")

    by = py2 + 24
    for b in range(L // C):
        bx = 56 + b * 330
        f.box(bx, by + 26, 296, 118, "#e6f4ea", GR, 10)
        f.t(bx + 18, by + 58, "第 %d 批" % (b + 1), GR, True, 21)
        for i in range(C):
            f.box(bx + 18 + i * 46, by + 72, 36, 50, "#fff", GR, 5)
            f.t(bx + 36 + i * 46, by + 102, str(b * C + i + 1), GR, True, 15,
                "middle")
        f.t(bx + 18, by + 166, "这一批<tspan font-weight=\"700\">同时办</tspan>", GR,
            size=17)
        if b < L // C - 1:
            f.line(bx + 300, by + 84, bx + 326, by + 84, INK, 2.0)
            f.t(bx + 313, by + 64, "交接单", INK, True, 14, "middle")

    f.box(56, by + 196, 1304, 100, "#fff", INK, 10)
    f.t(80, by + 234, "⭐ 交接单上只有一样东西：<tspan font-weight=\"700\">"
        "那块板子现在的样子</tspan>（状态 S）", INK, True, 21)
    f.t(80, by + 272, "所以并行度从 <tspan font-weight=\"700\">1 变成 %d</tspan>，"
        "串行步数从 <tspan font-weight=\"700\">%d 变成 %d</tspan> ——&#160;"
        "数学一个字没改，改的是算的顺序。" % (C, L, L // C), GY, size=18)

    # ══════════ ③ 一批放多少人 ══════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 258
    py3 = f.panel(0, y2, W, PH3, "③ 一批该放多少人 ——　这是个纯硬件问题",
                  OR, sub="两头都被夹住")

    ey = py3 + 24
    for i, (col, title, pic, why) in enumerate([
        (RD, "太少", "柜台空着", "一批 2 个人，几千个算力单元只用上几个"),
        (GR, "刚好", "柜台坐满，大厅站得下", "这就是要找的那个 C"),
        (RD, "太多", "大厅站不下", "块内的中间结果<tspan font-weight=\"700\">"
         "塞不进片上内存</tspan>，被迫往外倒"),
    ]):
        bx = 56 + i * 442
        f.box(bx, ey + 24, 400, 148, "#fff", col, 10)
        f.t(bx + 22, ey + 62, title, col, True, 24)
        f.t(bx + 22, ey + 96, pic, GY, True, 19)
        f.t(bx + 22, ey + 130, why, GY, size=16, w=356)
        n = (2, 6, 14)[i]
        for k in range(n):
            f.box(bx + 22 + (k % 7) * 26, ey + 146 + (k // 7) * 14, 20, 10,
                  col, "none", 2)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这一张真正的落点：它不是「更快的注意力」，是一次改算法顺序", [
        "chunkwise <tspan font-weight=\"700\">数学上跟逐 token 递推等价</tspan>"
        "（⚠️ 数值上不完全等价 ——&#160;求和顺序变了，舍入就变了）。"
        "它改的只有一件事：<tspan font-weight=\"700\">什么时候算什么</tspan>。",
        "⭐ 而这恰好是这门课的主线又一次出现："
        "<tspan font-weight=\"700\">一个数学上无所谓的选择，在硬件上决定生死</tspan> ——&#160;"
        "不分块，线性注意力根本喂不饱加速器，再省显存也没用。",
    ])

    yy = f.band(yy + 14, "warn", "块大小这件事，专题一已经证过一次", [
        "⚠️ <tspan font-weight=\"700\">C 被片上内存顶死</tspan>，"
        "跟专题一 splash attention 的块大小是同一类问题 ——&#160;"
        "而那一讲已经证过：<tspan font-weight=\"700\">块大小看的是比例，不是绝对值</tspan>，"
        "换一代硬件就得重调。",
        "⛔ 所以别去记「C 取多少」这个数 ——&#160;记<tspan font-weight=\"700\">"
        "「它被什么夹住」</tspan>：下面是算力吃不满，上面是片上内存放不下。",
    ])

    yy = f.src(yy + 16,
               "chunkwise 的形式与并行度 / 串行步数的改变出自 DeltaNet 并行化那篇 "
               "Yang 等 arXiv 2406.06484（WY 表示 ＋ 分块）",
               "⚠️ 图里的 %d 个 token / 每块 %d 个是<tspan font-weight=\"700\">示意</tspan>，"
               "真实实现的块长在几十到几百之间，随硬件变" % (L, C),
               "⚠️ 「排队办事 / 交接单」是<tspan font-weight=\"700\">本课的比喻</tspan>")
    f.save("fig3-chunkwise.svg", yy + 6)


main()
