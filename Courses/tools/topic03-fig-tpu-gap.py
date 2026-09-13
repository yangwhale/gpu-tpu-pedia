# -*- coding: utf-8 -*-
r"""专题三 · §10.2「这些注意力落到 TPU 上，难在哪」

⭐⭐⭐ 2026-09-13 **整张重画**，换成两个生活画面：
   **中央厨房 vs 点单现做**，以及**仓库取货**。

  ① **TPU 像中央厨房**：菜单提前定死，所有东西按批预制 ——&nbsp;
     出餐极快，代价是**临时改单很贵**。
     **GPU 像点单现做**：来什么做什么，灵活，但每道菜都要现开火。
     ⚠️ 这不是谁好谁坏 ——&nbsp;**中央厨房快，正是因为它不接临时改单。**
  ② **而现代注意力偏偏全是临时改单**：每桌人数不一样（ragged）、
     食材散在仓库各处（分页 KV）、**今天做哪几道菜要开工了才知道**（运行时 top-k）。
  ③ **对上之后最疼的一处画成仓库取货**：
     连号货架 →&nbsp;一趟推车拉走；散落各处 →&nbsp;跑很多趟。
     ⭐ 这就是 RPA 原文那句「让 DMA 调度变得困难」。

⛔ 最该念的一句出自 Google 自己那篇 RPA 论文（arXiv 2604.15464，2026-04）：
   现有 LLM 推理 kernel **基本都是 GPU 中心的**，
   **还没有一套成熟的办法**把 LLM 负载高效地映射到 TPU 架构上。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    f = Fig(W, "注意力落到 TPU 上难在哪：TPU 像中央厨房，菜单提前定死、按批预制，"
               "出餐快但临时改单贵；而现代注意力全是临时改单 —— 每桌人数不同、"
               "食材散在仓库各处、今天做哪几道要开工才知道")
    f.marks = set()
    y0 = f.header(
        "落到 TPU 上 ——　先看清楚是哪两件事对不上",
        "<tspan font-weight=\"700\">中央厨房</tspan> 碰上 "
        "<tspan font-weight=\"700\">全是临时改单的客人</tspan>",
        [(BL, "中央厨房 ＝ TPU"), (OR, "临时改单 ＝ 现代注意力"),
         (RD, "最疼的一处"), (GR, "已经有的解法")])

    # ══════════ ① 两种厨房：真的画出来 ══════════════════════════
    # ⭐⭐⭐ 2026-09-13 重画。审图原话：「**中央厨房 vs 点单现做**
    #   这个全课最好的比喻，一笔都没画 ——&nbsp;①② 是五张要点卡。」
    # ⛔ 比喻只写在文字里，等于没有比喻：读者看到的还是三条 bullet。
    PH = 344
    py = f.panel(0, y0, W, PH, "① 两种厨房 ——　这不是谁好谁坏",
                 BL, sub="中央厨房快，正是因为它不接临时改单")

    ay = py + 24
    # ── 左：中央厨房 ────────────────────────────────────────────
    f.box(56, ay + 22, 636, 254, "#e8f0fe", BL, 10)
    f.t(80, ay + 60, "TPU ＝ 中央厨房", BL, True, 26)
    # 墙上钉死的菜单
    f.box(80, ay + 78, 150, 108, "#fff", BL, 6)
    f.t(155, ay + 104, "今日菜单", BL, True, 17, "middle")
    for k in range(4):
        f.line(96, ay + 122 + k * 16, 214, ay + 122 + k * 16, "#aecbfa", 1.6,
               arrow=False)
    f.t(155, ay + 204, "钉在墙上，改不了", GY2, size=15, anchor="middle")
    # 一排蒸屉，整批出餐
    for k in range(4):
        f.box(256 + k * 62, ay + 84, 52, 40, "#fff", BL, 5)
        f.box(256 + k * 62, ay + 128, 52, 40, "#fff", BL, 5)
    f.t(256, ay + 192, "一排蒸屉，整批上", GY2, size=15)
    # 连号货架
    for k in range(5):
        f.box(524 + k * 30, ay + 96, 24, 56, "#fff", BL, 4)
        f.t(536 + k * 30, ay + 172, str(k + 1), BL, size=15, anchor="middle")
    f.line(524, ay + 186, 668, ay + 186, BL, 2.0)
    f.t(524, ay + 210, "连号货架，一趟拉走", GY2, size=15)
    f.t(80, ay + 246, "⭐ 出餐极快 ——　代价是临时改单很贵", BL, True, 20)

    # ── 右：点单现做 ────────────────────────────────────────────
    f.box(724, ay + 22, 636, 254, "#fff7ed", OR, 10)
    f.t(748, ay + 60, "GPU ＝ 点单现做", OR, True, 26)
    # 一张随手写的改单
    f.box(748, ay + 78, 150, 108, "#fff", OR, 6, 1, "5,4")
    f.t(823, ay + 104, "临时改单", OR, True, 17, "middle")
    f.t(823, ay + 130, "3 号桌", GY, size=16, anchor="middle")
    f.t(823, ay + 152, "少辣、加一份", GY, size=16, anchor="middle")
    f.t(823, ay + 204, "来什么做什么", GY2, size=15, anchor="middle")
    # 一个灶台，一次一份
    f.box(936, ay + 96, 112, 72, "#fff", OR, 6)
    f.t(992, ay + 124, "一个灶台", OR, True, 18, "middle")
    f.t(992, ay + 150, "一次一份", GY, size=16, anchor="middle")
    f.t(936, ay + 192, "一份也做", GY2, size=15)
    # 满仓库跑腿：散落的货架
    SC = ((1100, 92), (1188, 132), (1260, 96), (1150, 176), (1264, 168))
    for k, (gx, gy) in enumerate(SC):
        f.box(gx, gy + ay - 60, 34, 30, "#fff", OR, 4)
    for k in range(len(SC) - 1):
        f.line(SC[k][0] + 17, SC[k][1] + ay - 45, SC[k + 1][0] + 17,
               SC[k + 1][1] + ay - 45, OR, 1.2, arrow=False)
    f.t(1100, ay + 210, "满仓库跑腿，散落取货也认了", GY2, size=15)
    f.t(748, ay + 246, "⭐ 灵活 ——　代价是每道菜都要现开火", OR, True, 20)

    # ══════════ ② 客人全是临时改单 ══════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 284
    py2 = f.panel(0, y1, W, PH2, "② 而现代注意力，偏偏全是临时改单",
                  OR, sub="三样，全是最近五年长出来的")

    by = py2 + 24
    for i, (t, what, where) in enumerate([
        ("每桌人数都不一样", "一个 batch 里各请求长度不同", "vLLM 那套调度带来的"),
        ("食材散在仓库各处", "一条序列的 KV 散在不连续的页上", "PagedAttention 带来的"),
        ("今天做哪几道菜，开工了才知道", "这一步到底读哪 2048 条",
         "DSA / NSA 这一支带来的"),
    ]):
        bx = 56 + i * 442
        f.box(bx, by + 24, 400, 168, "#fff", OR, 10)
        f.t(bx + 22, by + 66, t, OR, True, 21, w=356)
        f.t(bx + 22, by + 108, what, GY, size=17, w=356)
        f.t(bx + 22, by + 172, "来源：" + where, GY2, size=15)

    f.t(56, by + 220, "⛔ 这三样<tspan font-weight=\"700\">全是在 GPU 上先长出来的</tspan>"
        " ——&#160;它们默认了一台「随手跑腿不太贵」的机器。", RD, True, 21)

    # ══════════ ③ 仓库取货 ══════════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 308
    py3 = f.panel(0, y2, W, PH3, "③ 对上之后，最疼的是哪一处 ——　仓库取货",
                  RD, sub="RPA 论文 §2 逐条点了名")

    ey = py3 + 22
    f.t(56, ey + 24, "连号货架：一趟推车拉走", GR, True, 22)
    for i in range(20):
        on = 4 <= i < 12
        f.box(56 + i * 32, ey + 40, 26, 56, "#e6f4ea" if on else BG2,
              GR if on else LINE2, 4)
    f.line(180, ey + 112, 430, ey + 112, GR, 2.2)
    f.t(56, ey + 142, "✅ 一次搬运，地址连着", GR, True, 19)

    f.t(760, ey + 24, "散落各处：跑很多趟", RD, True, 22)
    for i in range(20):
        on = i in (1, 4, 9, 13, 14, 18)
        f.box(760 + i * 32, ey + 40, 26, 56, "#fce8e6" if on else BG2,
              RD if on else LINE2, 4)
    for i in (1, 4, 9, 13, 14, 18):
        f.line(773 + i * 32, ey + 104, 773 + i * 32, ey + 118, RD, 1.4)
    f.t(760, ey + 142, "⛔ 六次搬运，地址还是跑起来才算出来的", RD, True, 19)

    f.box(56, ey + 168, 1304, 104, "#fce8e6", RD, 10)
    f.t(80, ey + 208, "⭐ 论文原话：这让<tspan font-weight=\"700\">"
        "DMA 调度</tspan>变得困难", RD, True, 22)
    f.t(80, ey + 246, "——&#160;DMA 就是那台推车：它最擅长「一趟拉一整排」，"
        "最怕「这一趟拉哪几个，得先算一下」。", GY, size=18)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "bad", "⛔ 这句话值得原样念一遍 —— 它出自 2026 年 4 月，不是五年前", [
        "「现有 LLM 推理 kernel 和服务系统<tspan font-weight=\"700\">基本都是 GPU 中心的"
        "</tspan>，而且<tspan font-weight=\"700\">还没有一套成熟的办法</tspan>"
        "把 LLM 负载高效地映射到 TPU 架构上。」",
        "——&#160;Google 自己那篇 Ragged Paged Attention 论文的摘要（arXiv 2604.15464）。"
        "⭐ 所以这一节讲的不是「怎么调参」，是<tspan font-weight=\"700\">一个还在打开的工程战场</tspan>。",
    ])

    yy = f.band(yy + 14, "info", "⭐ 一条能带走的判据：看一个机制默认了什么样的机器", [
        "临时改单、散落取货、开工才知道做什么 ——&#160;"
        "<tspan font-weight=\"700\">这三样都默认「随手跑腿不太贵」</tspan>。"
        "在一家为「按批预制」优化的厨房里，这个假设<tspan font-weight=\"700\">不成立</tspan>。",
        "⭐ 所以移植的活儿不是「翻译代码」，是"
        "<tspan font-weight=\"700\">把那个隐含的硬件假设找出来，再换一个等价但规整的做法</tspan>。"
        "下一张讲的就是这个「换法」。",
    ])

    yy = f.src(yy + 16,
               "三处疼与那句摘要出自 Ragged Paged Attention（Jiang 等，"
               "arXiv 2604.15464，2026-04）§1 与 §2.4；TPU 的三条约束亦见该文 §1",
               "⚠️ 「中央厨房 / 点单现做 / 仓库取货」是"
               "<tspan font-weight=\"700\">本课的比喻</tspan> ——&#160;"
               "论文那侧的说法是 static-first 编译、tiled 粗粒度布局、"
               "以及「从动态算出来的不连续地址 gather」")
    f.save("fig3-tpu-gap.svg", yy + 6)


main()
