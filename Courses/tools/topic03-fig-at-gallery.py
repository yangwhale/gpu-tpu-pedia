# -*- coding: utf-8 -*-
r"""专题三 · §七「A_t 的形状决定了一切」（2026-09-13 夜间 · R12）。

⭐⭐⭐ §7.2 和 §7.2b 原来是**三张表 ＋ 大段散文**，而它们讲的其实是
   **一个矩阵长什么样** ——&nbsp;这是全讲最该画、却一直没画的地方。

   所有这些方法都长成同一个递推：

        S_t ＝ S_{t-1} · A_t ＋ v_t k_tᵀ

   **区别只有一处：允许 A_t 长什么样。**
   把八种 A_t 的形状并排画出来，谱系、亲缘、代价，一眼全在。

⭐⭐ 这张图要立住的判据是那句：
   **表达力和可算性是一起设计的，不是先设计再优化。**
   A 越自由，表达力越强，但能不能分块并行**当场就决定了它能不能活下来** ——
   Mamba-2 甚至**故意退回**最简的 a·I，就为了证得出跟线性注意力的对偶。
"""
from topic03_draw import (Fig, wpx, sub, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400


def main():
    # (名字, 年份, A_t 形式, 画法, 修了上一步的什么, 并行代价, 色)
    ROWS = [
        ("朴素线性注意力", "2020", "I", "eye",
         "—— 起点：只加不减", "最容易", GY),
        # ⛔ 2026-09-14 二轮学生审稿：原来 RetNet 和 GLA 挤在同一格，
        #   而那一格的标签（「逐行衰减」「不依赖输入」）只符合 RetNet。
        #   GLA 摘要原话是 data-dependent gates，**2023 年就是逐通道 + 依赖输入**。
        # ⭐ 判据：**把一个东西摆错位置，会让整条演进线断掉一节** ——
        #   「固定衰减 → 依赖输入 → 逐通道」这条线原来缺了中间那段，
        #   于是 KDA 的「门从标量升成逐通道」看起来比实际更新。
        ("RetNet", "2023-07", "γ·I（固定标量）", "diag",
         "让陈年旧事按固定速度淡出", "最容易", GR),
        ("GLA", "2023-12", "Diag(α_t)，依赖输入", "diag-chan",
         "⭐ 淡出速度由内容定，且逐通道", "最容易", GR),
        ("Mamba-2", "2024-05", "a·I（标量×单位阵）", "scalar",
         "⭐ 故意退回最简，换来 SSD 对偶", "最容易", BL),
        ("DeltaNet", "2021 / 2024 并行化", "I − β k kᵀ", "rank1",
         "先擦掉旧的再写新的", "要专门的技巧", OR),
        ("GDN", "2024-12", "α(I − β k kᵀ)", "rank1-decay",
         "门控 ＋ 擦除，合到一起", "要专门的技巧", OR),
        ("KDA", "2025-10", "Diag(α)(I − β k kᵀ)", "rank1-chan",
         "⭐ 逐通道门接到擦除上", "要特化的 DPLR 算法", PU),
        ("（若允许全矩阵）", "——", "任意 A", "full",
         "表达力最强", "⛔ 每步 combine 是 d×d 矩阵乘", RD),
    ]

    f = Fig(W, "A_t 的形状决定了一切：八种结构并排，从单位阵到对角、到单位阵减秩一、"
               "到逐通道门控；A 越自由表达力越强，但能不能分块并行当场决定它能不能活")
    f.marks = set()
    # ⛔ 2026-09-14 二轮学生审稿两条，都在这个标题上：
    #   ① `S_{t-1}` / `ℝ^(d_v×d_k)` 是**没渲染的 LaTeX 源码** —— 见 topic03_draw.sub()。
    #   ② 写侧漏了系数：fig-notepad 写的是 `S(I−βkkᵀ) + **β** v kᵀ`，GDN / KDA 也都有 β。
    #      按这个「统一递推」，后三格根本套不进去 —— 学生把两张图并排一看，
    #      第一个问题就是「那个 β 去哪了」。⭐ 补一个 b_t，并说明它管什么。
    EQ = ("%s ＝ %s · %s ＋ %s · %s %sᵀ"
          % (sub("S", "t"), sub("S", "t−1"), sub("A", "t"), sub("b", "t"),
             sub("v", "t"), sub("k", "t")))
    y0 = f.header(
        "所有这些方法长成同一个递推　" + EQ,
        "⭐⭐ <tspan font-weight=\"700\">区别只有一处：允许 "
        + sub("A", "t") + " 长什么样。</tspan>"
        "（" + sub("A", "t") + " 管<tspan font-weight=\"700\">擦</tspan>，"
        + sub("b", "t") + " 管<tspan font-weight=\"700\">写多重</tspan> ——&#160;"
        "这一讲只比前者，后者各家都是一个标量或门；S 是 d_v × d_k 的矩阵，"
        + sub("A", "t") + " <tspan font-weight=\"700\">右乘</tspan>）",
        [(GR, "对角类：最好并行"), (OR, "单位阵减秩一"),
         (PU, "逐通道"), (RD, "算不动的那一档")])

    # ⛔⛔ 麻瓜视角：这八张缩略图的**读法**原来只写在最后一行 📌 灰字出处里
    #   （y≈902，图高 930）。读者盯了十几秒也看不出八张图差在哪，
    #   因为他不知道竖条和斜纹各代表什么。⭐ 读图的钥匙必须在图之前。
    f.t(0, y0 + 6, "⚠️ 先说怎么读这八张小图："
        "<tspan font-weight=\"700\">对角线上那根竖条的高度 ＝ 衰减强度</tspan>"
        "（越矮擦得越狠）；", INK, size=18, w=1396)
    f.t(0, y0 + 32, "<tspan font-weight=\"700\">盖在上面的斜纹 ＝ 这一整块都被动过</tspan>"
        "（也就是那个 −βkkᵀ）。八张图的差别全在这两样上。", INK, size=18, w=1396)
    y0 += 52

    # ── 八个矩阵缩略图，两行四列 ────────────────────────────────
    N = 7                     # 缩略矩阵画 7×7
    CELL = 12
    CW, CH = 344, 214
    yy = y0 + 6
    for idx, (name, year, form, kind, fix, par, col) in enumerate(ROWS):
        cx = (idx % 4) * (CW + 8)
        cy = yy + (idx // 4) * (CH + 10)
        f.box(cx, cy, CW, CH, "#fff", LINE, 8)
        f.box(cx, cy, 4, CH, col, col, 2)
        f.box(cx + 2, cy, 3, CH, "#fff", "#fff", 0)
        f.t(cx + 18, cy + 24, name, col, True, 16, w=CW - 90)
        f.t(cx + CW - 16, cy + 24, year, GY2, size=14, anchor="end")

        # ⛔⛔ 2026-09-14 二轮学生量了 WCAG 对比度：这八格原来靠**深浅**区分，
        #   DeltaNet 底色 vs GDN 底色 **1.05 : 1**，I vs a·I 1.64 : 1 ——
        #   在 2× DPI 的 PNG 上都分不出来，投影仪上就是八个一样的方块。
        # ⭐ 判据：**别用深浅编码要「一眼看出来」的差别，用结构。**
        #   现在：对角格里画一根**高度可变的竖条**（＝衰减强度），
        #   秩一那一块另外盖一层**斜纹**（＝那个 −βkkᵀ）。
        #   深浅只做辅助，拿掉颜色这张图照样读得出来。
        mx = cx + 18
        my = cy + 38
        RANK1 = kind.startswith("rank1")
        for r in range(N):
            for c in range(N):
                f.box(mx + c * CELL, my + r * CELL, CELL - 2, CELL - 2,
                      "#fff", LINE2, 2)
        if RANK1 or kind == "full":       # 整块底：秩一 / 全矩阵
            f.box(mx, my, N * CELL - 2, N * CELL - 2,
                  "#fce8e6" if kind == "full" else "#f1f3f4", "none", 3)
            for k in range(-N, N):        # 斜纹，表示「一整块都被动了」
                f.line(mx + k * CELL, my, mx + (k + N) * CELL,
                       my + N * CELL - 2, LINE2, 0.8)
        for r in range(N):                # 对角线上的竖条：高度 = 衰减强度
            v = {"eye": 1.0,
                 "diag": 1.0 - r * 0.105,
                 "diag-chan": 0.35 + ((r * 3) % 5) * 0.16,
                 "scalar": 0.62,
                 "rank1": 1.0,
                 "rank1-decay": 0.62,
                 "rank1-chan": 0.3 + ((r * 2) % 4) * 0.23,
                 "full": 0.55}[kind]
            h = max(2.0, (CELL - 2) * v)
            bar = "#d93025" if col == RD else col
            f.box(mx + r * CELL, my + r * CELL + (CELL - 2 - h), CELL - 2, h,
                  bar, "none", 2)

        f.t(mx + N * CELL + 16, my + 18, form, INK, True, 15,
            w=CW - N * CELL - 44)
        f.t(mx + N * CELL + 16, my + 44, "修了什么", GY2, size=14)
        f.t(mx + N * CELL + 16, my + 62, fix, GY, size=14,
            w=CW - N * CELL - 44)

        # ⭐ 2026-09-13：投影上「深浅」分不开，补一行结构标签 ——
        #   让差别靠**读得出来的词**传达，不靠像素亮度。
        SHAPE = {"eye": "对角线，竖条等高（全同）",
                 "diag": "对角线，竖条逐行变矮（固定 γ）",
                 "diag-chan": "对角线，竖条高低不齐（逐通道）",
                 "scalar": "对角线，竖条等高但矮一截（×a）",
                 "rank1": "斜纹整块 ＋ 对角竖条等高",
                 "rank1-decay": "斜纹整块 ＋ 对角竖条整体变矮",
                 "rank1-chan": "斜纹整块 ＋ 对角竖条高低不齐",
                 "full": "整块红底 —— 没有结构可利用"}
        f.t(mx, my + N * CELL + 16, SHAPE[kind], col if col != GY else GY2,
            True, 14, w=CW - 36)

        f.t(cx + 18, cy + CH - 18, "并行", GY2, size=14)
        f.t(cx + 58, cy + CH - 18, par, col if col != GY else GY, True, 14.5,
            w=CW - 90)
    yy = yy + 2 * (CH + 10) + 6

    yy = f.band(yy, "info", "⭐⭐ 这张图真正的结论：表达力和可算性是一起设计的", [
        "从上到下，A 越来越自由 ——&#160;<tspan font-weight=\"700\">但这不是一个"
        "「表达力越来越强」的单调故事</tspan>。"
        "每一步都在「A 能多复杂」和「还算不算得动」之间<tspan font-weight=\"700\">重新划一次线</tspan>。",
        "⭐ 最能说明问题的是 <tspan font-weight=\"700\">Mamba-2 主动往回退</tspan>："
        "它把 A 退成最简的 a·I ——&#160;正因为退了，才证得出它跟线性注意力是"
        "<tspan font-weight=\"700\">对偶</tspan>的（SSD），才能把递推写成矩阵乘、吃上 Tensor Core。",
        # ⛔ 2026-09-14：这里原来写「没有高效的并行扫描 —— 并行度直接归零」。
        #   **错的**：矩阵乘可结合，Blelloch 扫描照样能做，深度 O(log L)。
        #   真正的障碍是**成本**：每次 combine 是一个 d×d 矩阵乘，总量 O(L·d³)，
        #   而且退化不成能吃 Tensor Core 的分块 matmul。
        # ⭐ 判据：**把一个成本论证说成存在性论证，等于给懂行的人递了一个把柄。**
        "⛔ 最后一格是反例，但要说准：<tspan font-weight=\"700\">并行扫描是存在的</tspan>"
        "（矩阵乘可结合，深度 O(log L)）——&#160;"
        "<tspan font-weight=\"700\">不在的是成本</tspan>：每次 combine 都是一个 d×d 矩阵乘，"
        "总量 O(L·d³)，而且拼不出能吃 Tensor Core 的分块 matmul。于是它不在候选名单里。",
    ])

    yy = f.band(yy + 14, "ok", "⭐ 顺带回答「那 Mamba 呢」—— 它不是另一支，是这一支的祖宗", [
        "不用推理，有一条字面证据：Gated DeltaNet 那篇论文的标题就是"
        "《<tspan font-weight=\"700\">Gated Delta Networks: Improving Mamba2 with Delta Rule"
        "</tspan>》，而 Qwen3-Next / Qwen3.5 用的就是 GDN。",
        "⭐ 再加上 Mamba-2 那篇自己的标题 ——&#160;"
        "《<tspan font-weight=\"700\">Transformers are SSMs</tspan>》。"
        "<tspan font-weight=\"700\">SSM 和线性注意力从来就不是两支，是同一件事的两种写法。</tspan>",
    ])

    yy = f.src(yy + 16,
               "递推式与各家 A_t 的结构见：线性注意力 arXiv 2006.16236、"
               "DeltaNet arXiv 2102.11174（2021）与并行化 arXiv 2406.06484、"
               "Mamba-2 / SSD arXiv 2405.21060、GDN arXiv 2412.06464、"
               "KDA（Kimi Linear）arXiv 2510.26692",
               "⚠️ 矩阵缩略图是<tspan font-weight=\"700\">示意</tspan>：只画结构 ——&#160;"
               "<tspan font-weight=\"700\">对角竖条的高度</tspan>代表衰减强度、"
               "<tspan font-weight=\"700\">斜纹</tspan>代表「一整块都被动了」（那个 −βkkᵀ），"
               "<tspan font-weight=\"700\">不代表真实数值</tspan>")
    f.save("fig3-at-gallery.svg", yy + 6)


main()
