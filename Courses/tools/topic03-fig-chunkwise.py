# -*- coding: utf-8 -*-
r"""专题三 · §7.4「串行的状态怎么榨出并行度」

⭐⭐⭐ 2026-09-13 **整张重画**，换成一个人人排过队的画面：
   **一百万个人要办事，只有一个窗口 vs 分成一批一批地办。**

   ① **逐 token 串行 ＝ 一个窗口、一个一个来** ——&nbsp;一百万步，没法并行。
   ② **chunkwise ＝ 分批** ——&nbsp;一批人**同时办**（块内并行），
      办完这一批，**只把一张交接单传给下一批**（块间串行）。
      ⭐ 画出来就不用解释：**并行度 1 → C，串行步数 L → L/C。**
   ③ **C 是一条轴，两端都是已经认识的东西** ——&nbsp;2026-09-14 R40 新增。
      C＝1 就是①那个一个窗口，C＝L 一张交接单都不用。
      ⛔ 右端是**线性注意力的并行形式**，不是 softmax 注意力 ——
      整条轴上从头到尾都没有 softmax。
   ④ **一批该放多少人，是个纯硬件问题** ——&nbsp;两头都被夹住：
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
    f.t(56, ay + 20, "每一步都要等上一步的结果", RD, True, 17)
    f.t(56, ay + 108, "⛔ 一百万个 token ＝ <tspan font-weight=\"700\">一百万步，"
        "一步都不能并</tspan>", RD, True, 17)
    f.t(56, ay + 142, "加速器最怕这个 ——&#160;几千个算力单元，一次只喂得上一个", GY,
        size=17)

    # ══════════ ② 分批办 ════════════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 360   # ⚠️ R40 +24：交接单那块塞进 S 方块后长到 124 高
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
            # ⛔⛔ 2026-09-15 R7：这里原来是「一根箭头 ＋ 头顶三个字『交接单』」。
            #   可两个批次框之间只有 34 px，而三个 14 px 汉字要 42 px ——&nbsp;
            #   **三处「交接单」全被左右两个绿框切掉了半个字**，而它正是这一格的核心词。
            # ⭐ 改法不是把字缩小（缩到能塞下就看不清了），是**把它画成东西**：
            #   缝里放一张小纸，字只写一次、写在框底下那行空地上。
            #   ⭐ 判据：**一个词在图上出现 N 次却每次都放不下，说明它该变成图形。**
            f.icon("paper", bx + 302, by + 62, 22, 28, INK, "#fff")
            f.line(bx + 300, by + 106, bx + 326, by + 106, INK, 1.8)
            if b == 0:
                f.t(bx + 313, by + 166, "交接单", INK, True, 14, "middle")

    # ⭐⭐ 2026-09-14 R40 补的一条（全网调研出来的第二个真空）：
    #   画 S 是「固定小方块」的人（刀刀宁、Jia-Bin Huang、MiniMax-01 Fig 5）
    #   画的都是 attention-vs-linear 的对照；而画 chunkwise 的人
    #   （Songlin Yang、GLA Fig 1、snowchord、Lightning-2），
    #   **S 一律是个没有标注的方块**。唯一的交叉点 TFLA Fig 3 标了轴，
    #   却把所有盒子画成一样大 ——&#160;等于标了跟没标一样。
    # ⛔ 结果是：读者看完 chunkwise 那张图，**仍然不知道那张交接单有多大** ——
    #   而这恰恰是「为什么要这么切」的全部答案。所以这里把它画出来。
    f.box(56, by + 196, 1304, 124, "#fff", INK, 10)
    f.t(80, by + 234, "⭐ 交接单上只有一样东西：<tspan font-weight=\"700\">"
        "那块板子现在的样子</tspan>（状态 S）", INK, True, 17)
    f.t(80, by + 272, "所以并行度从 <tspan font-weight=\"700\">1 变成 %d</tspan>，"
        "串行步数从 <tspan font-weight=\"700\">%d 变成 %d</tspan> ——&#160;"
        "数学一个字没改，改的是算的顺序。" % (C, L, L // C), GY, size=17)
    f.t(80, by + 302, "⭐⭐ 而且<tspan font-weight=\"700\">每一张交接单都一样大</tspan>"
        " ——&#160;跟这一批有几个人、整句话有多长，<tspan font-weight=\"700\">"
        "都没关系</tspan>。这才是它敢这么切的全部理由。", GY, size=17)
    f.box(1146, by + 214, 86, 86, "#e8f0fe", BL, 8, 1.6)
    f.t(1189, by + 250, "S", BL, True, 26, "middle")
    f.t(1189, by + 274, "d_k × d_v", BL, size=14, anchor="middle")
    f.t(1254, by + 250, "← 这个尺寸", BL, True, 17)
    f.t(1254, by + 274, "从头到尾不变", BL, size=16)

    # ══════════ ③ C 是一条轴，两端都退化 ════════════════════════
    # ⭐⭐⭐ 2026-09-14 R40 新增。这是全网调研出来的**主真空**：
    #   所有人都知道 C=1 退化成递推、C=L 退化成并行形式，
    #   所有人都**用文字说一遍** ——&#160;
    #   但「画出来」的只有 TFLA arXiv 2503.14376 Figure 19 一张，
    #   而那是埋在附录里的 **FLOPs 曲线**：它画的是「代价」高低，
    #   不是「计算本身变成了什么」。
    # ⭐ 而这张图有个别人没有的便宜可占：**面板① 画的就是 C=1 那一端。**
    #   所以这里不用另起炉灶，把①②接到一条轴上就行。
    # ⛔⛔ 一个必须写死的口径：右端是**线性注意力的并行形式**，
    #   **不是 softmax 注意力**。「chunk 开满就变回 Transformer」是错的 ——
    #   整条轴上从头到尾都没有 softmax。调研报告里那句
    #   「C=n 退化成并行注意力」在中文里有歧义，这里说清楚。
    y2 = y1 + PH2 + 18
    PHX = 382   # ⚠️ 两条落点原来压在最后一行 C＝24 的条上，实测才看见
    pyx = f.panel(0, y2, W, PHX,
                  "③ 那 C 到底是什么 ——&#160;<tspan font-weight=\"700\">"
                  "它是一条轴，而两头都是你已经认识的东西</tspan>", PU,
                  sub="批越宽，交接单越少 ——　两端各少掉一样")
    xy = pyx + 22
    SPEC = [(1, "就是①那个一个窗口"), (3, ""), (C, "就是②画的那一行"),
            (L, "一张交接单都不用")]
    assert SPEC[0][0] == 1 and SPEC[-1][0] == L, "两端必须是退化端，否则不成其为轴"
    assert all(L % c == 0 for c, _ in SPEC), "示意图里每种 C 都要整除 L"
    # ⚠️ 这三个数是量出来的，别凭感觉动：TOTW 从 1000 收到 850，是因为最右边那句
    #   「24 批 · 23 张交接单　←　就是①那个一个窗口」实测约 330 px，1000 会顶穿右边框。
    X0, TOTW, GAPB, RH = 120, 850.0, 9.0, 30
    for r, (c, tail) in enumerate(SPEC):
        nb = L // c
        bw = (TOTW - (nb - 1) * GAPB) / nb
        yb = xy + 30 + r * 66
        col = RD if c == 1 else (BL if c == L else GR)
        f.t(X0 - 14, yb + 21, "C＝%d" % c, col, True, 18, "end")
        for b in range(nb):
            bx = X0 + b * (bw + GAPB)
            f.box(bx, yb, bw, RH, "#fce8e6" if c == 1 else
                  ("#e8f0fe" if c == L else "#e6f4ea"), col, 4, 1.2)
            if b < nb - 1:                      # 交接单：批与批之间那一竖
                f.line(bx + bw + 1, yb + RH / 2, bx + bw + GAPB - 1,
                       yb + RH / 2, INK, 1.6, arrow=False)
        f.t(X0 + TOTW + 16, yb + 21,
            "%d 批 ·&#160;<tspan font-weight=\"700\">%d 张交接单</tspan>%s"
            % (nb, nb - 1, ("　←　" + tail) if tail else ""), GY, size=16)
    # ⭐ 一条贯穿四行的箭头 ——&#160;没有它这就是「四个例子」而不是「一条轴」
    # ⚠️ 标注放在箭头**上方**而不是旁边：放旁边（anchor=end）会被左边框切掉半个字。
    f.t(46, xy + 22, "C 越大", PU, True, 15, "middle")
    # ⚠️ 收到最后一行的**中线**就停，不要画到行底 ——&#160;画到行底时箭头尖
    #   正好戳在下面那句 ⭐⭐ 的星星上。
    f.line(46, xy + 34, 46, xy + 30 + 3 * 66 + RH / 2, PU, 2.0)
    f.t(30, xy + 284,
        "⭐⭐ <tspan font-weight=\"700\">所以 chunkwise 不是第三种算法，"
        "是连接那两端的一个旋钮</tspan> ——&#160;"
        "往上拧回 ①（一步一个，喂不饱算力），往下拧到底就是一整张矩阵算完。",
        INK, size=17)
    # ⚠️ 这一行**不能再长了**：渲染后几何体检量到它顶出图框，才砍成现在这样
    #   （原句多「一整张」「上」「都」「。」，量准字宽后是 1386px，图才 1400 宽）。
    f.t(30, xy + 316,
        "⛔ 右端是<tspan font-weight=\"700\">线性注意力的并行形式</tspan>"
        "（L×L 下三角一次算完），<tspan font-weight=\"700\">不是 softmax "
        "注意力</tspan> ——&#160;整条轴从头到尾没有 softmax，"
        "「chunk 开满就变回 Transformer」是错的。", RD, size=17)

    # ══════════ ④ 一批放多少人 ══════════════════════════════════
    y2 = y2 + PHX + 18
    PH3 = 258
    py3 = f.panel(0, y2, W, PH3, "④ 一批该放多少人 ——　这是个纯硬件问题",
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
