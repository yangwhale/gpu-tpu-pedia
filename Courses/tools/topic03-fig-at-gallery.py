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
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400


def main():
    # (名字, 年份, A_t 形式, 画法, 修了上一步的什么, 并行代价, 色)
    ROWS = [
        ("朴素线性注意力", "2020", "I", "eye",
         "—— 起点：只加不减", "最容易", GY),
        ("RetNet / GLA", "2023", "γ 或 Diag(γ)", "diag",
         "让陈年旧事自己淡出", "最容易", GR),
        ("Mamba", "2023", "对角，且依赖输入", "diag-sel",
         "衰减速度由内容决定", "容易", GR),
        ("Mamba-2", "2024-05", "a·I（标量×单位阵）", "scalar",
         "⭐ 故意退回最简，换来 SSD 对偶", "最容易", BL),
        ("DeltaNet", "2021 / 2024 并行化", "I − β k kᵀ", "rank1",
         "先擦掉旧的再写新的", "要专门的技巧", OR),
        ("GDN", "2024-12", "α(I − β k kᵀ)", "rank1-decay",
         "门控 ＋ 擦除，合到一起", "要专门的技巧", OR),
        ("KDA", "2025-10", "Diag(α)(I − β k kᵀ)", "rank1-chan",
         "门从标量升成逐通道", "要特化的 DPLR 算法", PU),
        ("（若允许全矩阵）", "——", "任意 A", "full",
         "表达力最强", "⛔ 并行度归零", RD),
    ]

    f = Fig(W, "A_t 的形状决定了一切：八种结构并排，从单位阵到对角、到单位阵减秩一、"
               "到逐通道门控；A 越自由表达力越强，但能不能分块并行当场决定它能不能活")
    f.marks = set()
    y0 = f.header(
        "所有这些方法长成同一个递推　S_t ＝ S_{t-1} · A_t ＋ v_t k_tᵀ",
        "⭐⭐ <tspan font-weight=\"700\">区别只有一处：允许 A_t 长什么样。</tspan>"
        "（S ∈ ℝ^(d_v×d_k)，A_t <tspan font-weight=\"700\">右乘</tspan>）"
        "把八种形状并排画出来，谱系、亲缘、代价一眼全在",
        [(GR, "对角类：最好并行"), (OR, "单位阵减秩一"),
         (PU, "逐通道"), (RD, "算不动的那一档")])

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
        f.t(cx + 18, cy + 24, name, col, True, 12.5, w=CW - 90)
        f.t(cx + CW - 16, cy + 24, year, GY2, size=11, anchor="end")

        # 矩阵
        mx = cx + 18
        my = cy + 38
        for r in range(N):
            for c in range(N):
                on, shade = False, 1.0
                if kind == "eye":
                    on = (r == c)
                elif kind == "diag":
                    on, shade = (r == c), 1.0 - r * 0.11
                elif kind == "diag-sel":
                    on, shade = (r == c), 0.45 + ((r * 3) % 5) * 0.13
                elif kind == "scalar":
                    on, shade = (r == c), 0.72
                elif kind == "rank1":
                    on = True
                    shade = 1.0 if r == c else 0.22
                elif kind == "rank1-decay":
                    on = True
                    shade = 0.7 if r == c else 0.18
                elif kind == "rank1-chan":
                    on = True
                    shade = (0.4 + ((r * 2) % 4) * 0.2) if r == c else 0.18
                elif kind == "full":
                    on, shade = True, 0.55
                if on:
                    g = int(255 - shade * 150)
                    fill = "#%02x%02x%02x" % (g, min(255, g + 18), 255 - int(shade * 60))
                    if col == RD:
                        fill = "#%02x%02x%02x" % (255, g, g)
                    f.box(mx + c * CELL, my + r * CELL, CELL - 2, CELL - 2,
                          fill, "none", 2)
                else:
                    f.box(mx + c * CELL, my + r * CELL, CELL - 2, CELL - 2,
                          "#fff", LINE2, 2)
        f.t(mx + N * CELL + 16, my + 18, form, INK, True, 12,
            w=CW - N * CELL - 44)
        f.t(mx + N * CELL + 16, my + 44, "修了什么", GY2, size=11)
        f.t(mx + N * CELL + 16, my + 62, fix, GY, size=11,
            w=CW - N * CELL - 44)

        # ⭐ 2026-09-13：投影上「深浅」分不开，补一行结构标签 ——
        #   让差别靠**读得出来的词**传达，不靠像素亮度。
        SHAPE = {"eye": "只有对角线，全同",
                 "diag": "对角线，逐行衰减",
                 "diag-sel": "对角线，衰减由输入决定",
                 "scalar": "对角线，整块同一个标量",
                 "rank1": "对角线 ＋ 一整块秩一",
                 "rank1-decay": "秩一块 ＋ 对角线整体变淡",
                 "rank1-chan": "秩一块 ＋ 对角线逐通道不同",
                 "full": "全满 —— 没有结构可利用"}
        f.t(mx, my + N * CELL + 16, SHAPE[kind], col if col != GY else GY2,
            True, 11, w=CW - 36)

        f.t(cx + 18, cy + CH - 18, "并行", GY2, size=11)
        f.t(cx + 58, cy + CH - 18, par, col if col != GY else GY, True, 11.5,
            w=CW - 90)
    yy = yy + 2 * (CH + 10) + 6

    yy = f.band(yy, "info", "⭐⭐ 这张图真正的结论：表达力和可算性是一起设计的", [
        "从上到下，A 越来越自由 ——&#160;<tspan font-weight=\"700\">但这不是一个"
        "「表达力越来越强」的单调故事</tspan>。"
        "每一步都在「A 能多复杂」和「还算不算得动」之间<tspan font-weight=\"700\">重新划一次线</tspan>。",
        "⭐ 最能说明问题的是 <tspan font-weight=\"700\">Mamba-2 主动往回退</tspan>："
        "它把 A 退成最简的 a·I ——&#160;正因为退了，才证得出它跟线性注意力是"
        "<tspan font-weight=\"700\">对偶</tspan>的（SSD），才能把递推写成矩阵乘、吃上 Tensor Core。",
        "⛔ 最后一格是反例：<tspan font-weight=\"700\">全矩阵 A 表达力最强，"
        "但没有高效的并行扫描 ——&#160;并行度直接归零，于是它根本不在候选名单里。</tspan>",
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
               "⚠️ 矩阵缩略图是<tspan font-weight=\"700\">示意</tspan>：只画结构（哪里非零、"
               "深浅代表是否逐通道不同），不代表真实数值")
    f.save("fig3-at-gallery.svg", yy + 6)


main()
