# -*- coding: utf-8 -*-
r"""专题三 · §4.2b「事后压 vs 从头按压缩训」

⭐⭐⭐ 2026-09-13 **整张重画**，换成一个装修过房子的人都懂的画面：
   **老房改造 vs 一开始就这么设计。**

   · **事后压 ＝ 房子已经盖好了再改造** ——&nbsp;快、便宜、不用搬家，
     但**承重墙动不了**，有些地方就是别扭。
   · **native ＝ 图纸阶段就按这个户型画** ——&nbsp;要重盖（重训），
     但**哪儿都合适**。

   这条对立在这一讲里**出现四次**，每次在不同的分支上 ——&nbsp;
   ⚠️ 但四次**全落在旋钮①② 上，旋钮③ 一格都没有**（为什么留空见 §4.2b 正文）。

⚠️ 一条必须讲清的口径（否则这张图会教出一个错的心智模型）：
   **「事后 / native」说的是这个方法<b>这一次被怎么用</b>，不是方法本身的属性。**
   GQA 最初是 uptraining 出来的（改造），但今天 Llama 那一系
   **从第一天就是 GQA**（新建）。同一个机制，两种用法。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    f = Fig(W, "事后压 vs 从头按压缩训：就像老房改造和一开始就按这个户型设计 —— "
               "改造快、便宜、不用搬家，但承重墙动不了；"
               "新建要重盖，但哪儿都合适。这条对立在四个分支上各出现一次")
    f.marks = set()
    y0 = f.header(
        "事后压，还是从头按压缩训",
        "就像<tspan font-weight=\"700\">老房改造</tspan> 和 "
        "<tspan font-weight=\"700\">图纸阶段就按这个户型画</tspan>",
        [(OR, "改造（事后）"), (GR, "新建（native）"),
         (GY, "同一个机制的两种用法")])

    # ══════════ ① 两种做法 ══════════════════════════════════════
    PH = 262
    py = f.panel(0, y0, W, PH, "① 两种做法 ——　各有各的好，不是谁淘汰谁",
                 INK, sub="装修过的人都懂")

    ay = py + 24
    f.box(56, ay + 24, 636, 188, "#fff", OR, 10)
    f.t(80, ay + 64, "改造 ——　房子已经盖好了", OR, True, 25)
    for i, ln in enumerate([
        "✅ 快、便宜，不用搬家（不用重训）",
        "✅ 想撤随时撤（很多是推理期开关）",
        "⛔ 承重墙动不了 ——　有些地方就是别扭",
    ]):
        f.t(104, ay + 108 + i * 36, ln, GY, size=17)

    f.box(724, ay + 24, 636, 188, "#e6f4ea", GR, 10)
    f.t(748, ay + 64, "新建 ——　图纸阶段就这么画", GR, True, 25)
    for i, ln in enumerate([
        "⛔ 要重盖（重训），贵、慢",
        "⛔ 盖完了就改不回去",
        "✅ 哪儿都合适 ——　省得更狠，掉点更小",
    ]):
        f.t(772, ay + 108 + i * 36, ln, GY, size=17)

    # ══════════ ② 四次出现：画成四栋房子 ══════════════════════
    # ⭐⭐⭐ 2026-09-13 重画。原来是一张三列四行的表 ——&nbsp;
    #   而这一格要说的是「**同一处地方，两种盖法**」，那就该画房子。
    # ⭐ 旋钮③ 那一栏**空着** ——&nbsp;空着这件事本身就是信息，
    #   表格里看不出来（就是少一行），画出来一眼就是「这儿没有」。
    y1 = y0 + PH + 18
    PH2 = 470
    py2 = f.panel(0, y1, W, PH2, "② 这条对立，在这一讲里出现四次",
                  GR, sub="四次都在不同的分支上 ——　所以它不是巧合")

    COLS = [
        ("第 1 次", "旋钮① 低秩分解",
         "Eigen / Palu / LoRC", "省 40% / 50%", "MLA", "省 56.9×"),
        ("第 2 次", "旋钮① 砍头",
         "GQA uptraining", "接近 MHA", "GQA from scratch", "不用两段"),
        ("第 3 次", "旋钮② 挑着看",
         "H2O", "随时可开关", "DSA", "掉点小得多"),
        ("第 4 次", "旋钮② 挑着看",
         "ClusterKV / Quest", "⛔ 三家三种毛病", "NSA", "持平或超过"),
        ("—", "旋钮③ 换数学", None, None, None, None),   # ⭐ 故意留空
    ]
    CW2, CG = 258, 20
    by = py2 + 24
    for i, (no, where, a1, a2, b1, b2) in enumerate(COLS):
        cx = 20 + i * (CW2 + CG)
        empty = (a1 is None)
        f.t(cx, by + 20, no, GY2, size=16)
        f.t(cx, by + 46, where, INK, True, 19, w=CW2)

        # ── 上：改造（房子已经盖好，正在砸墙，承重墙打叉）──────────
        hy = by + 62
        if empty:
            f.box(cx, hy, CW2, 152, "#fafafa", LINE2, 10, 1, "6,5")
            f.t(cx + CW2 / 2.0, hy + 72, "这一栏空着", GY2, True, 20, "middle")
            f.t(cx + CW2 / 2.0, hy + 100, "本讲没有核过的例子", GY2, size=16,
                anchor="middle")
        else:
            f.box(cx, hy, CW2, 152, "#fff7ed", OR, 10)
            f.t(cx + 14, hy + 28, "改造", OR, True, 18)
            # 房子：屋顶 ＋ 墙
            hx = cx + 84
            f.poly([(hx, hy + 58), (hx + 40, hy + 30), (hx + 80, hy + 58)], OR)
            f.box(hx + 8, hy + 58, 64, 48, "#fff", OR, 3)
            f.line(hx + 40, hy + 58, hx + 40, hy + 106, OR, 2.0, arrow=False)
            for dx, dy in ((-11, -11), (-11, 11)):     # 承重墙上打叉
                f.line(hx + 40 - dx, hy + 82 - dy, hx + 40 + dx, hy + 82 + dy,
                       RD, 2.6, arrow=False)
            f.t(hx + 96, hy + 76, "承重墙", RD, True, 16)
            f.t(hx + 96, hy + 98, "动不了", RD, True, 16)
            f.t(cx + 14, hy + 128, a1, OR, True, 16, w=CW2 - 28)
            f.t(cx + 14, hy + 148, a2, GY, size=15, w=CW2 - 28)

        # ── 下：新建（还在图纸上，随便画）───────────────────────
        ny = hy + 166
        if empty:
            f.box(cx, ny, CW2, 128, "#fafafa", LINE2, 10, 1, "6,5")
            f.t(cx + CW2 / 2.0, ny + 58, "也空着", GY2, True, 20, "middle")
            f.t(cx + CW2 / 2.0, ny + 86, "为什么留空见正文", GY2,
                size=15, anchor="middle")
        else:
            f.box(cx, ny, CW2, 128, "#e6f4ea", GR, 10)
            f.t(cx + 14, ny + 26, "新建", GR, True, 18)
            gx2 = cx + 84
            f.box(gx2, ny + 34, 96, 50, "#fff", GR, 3, 1, "4,3")   # 图纸
            for k in range(3):
                f.line(gx2, ny + 46 + k * 13, gx2 + 96, ny + 46 + k * 13,
                       "#b7e1c1", 1, arrow=False)
            f.t(cx + 14, ny + 52, "还在", GR, True, 16)
            f.t(cx + 14, ny + 74, "图纸上", GR, True, 16)
            f.t(cx + 14, ny + 102, b1, GR, True, 16, w=CW2 - 28)
            f.t(cx + 14, ny + 122, b2, GY, size=15, w=CW2 - 28)

    f.t(20, by + 386, "⚠️ 四次<tspan font-weight=\"700\">全落在旋钮①② 上</tspan>"
        " ——　旋钮③ 那一栏是空的。"
        "<tspan font-weight=\"700\">空着比硬凑一栏好</tspan>："
        "空栏可以被后来的人填上，凑出来的会被当成事实背下去。", RD, True, 17,
        w=1360)

    # ══════════ 落点 ════════════════════════════════════════════
    # ⛔ 这里原来写死 y1 + PH2 + 20，而面板②最后那行提示是画在**面板外面**的
    #   （by 已经跑到 PH2 以下）。落点带一长高就压上去了。
    # ⭐ 判据：落点带的起点要跟**真实画到哪儿**走，不能跟声明的面板高走。
    yy = max(y1 + PH2, by + 40) + 20
    yy = f.band(yy, "warn", "⚠️ 一条不讲会教出错误心智模型的口径", [
        "<tspan font-weight=\"700\">「改造 / 新建」说的是这个方法"
        "<tspan font-style=\"italic\">这一次被怎么用</tspan>，"
        "不是方法本身的属性。</tspan>",
        "最好的例子就是 GQA：它最初是 <tspan font-weight=\"700\">uptraining</tspan> 出来的"
        "（改造），可今天 Llama 那一系<tspan font-weight=\"700\">从第一天就是 GQA</tspan>"
        "（新建）。⭐ <tspan font-weight=\"700\">同一个机制，两栏都待过。</tspan>",
    ])

    yy = f.band(yy + 14, "info", "⭐ 带走的用法：拿到一个新名字，先把它放进某一格", [
        "放得进去的，<tspan font-weight=\"700\">它的优点和代价你已经知道了</tspan> ——&#160;"
        "改造类的就去问「承重墙在哪」（哪里改不动）；"
        "新建类的就去问「重训要多少钱」。",
        "⭐ <tspan font-weight=\"700\">放不进去的，才值得你花时间</tspan> ——&#160;那才是真正的新东西。",
    ])

    yy = f.src(yy + 16,
               "各家的出处见 §五 / §六 对应小节；"
               "GQA 的 mean pooling ＋ α=5% 续训出自 arXiv 2305.13245 §2.2",
               "⚠️ 「老房改造 / 图纸阶段」是<tspan font-weight=\"700\">本课的比喻</tspan>；"
               "⚠️ 「四次」只数到旋钮①②，旋钮③ 那一格本讲没有核过数，故留空")
    f.save("fig3-when-axis.svg", yy + 6)


main()
