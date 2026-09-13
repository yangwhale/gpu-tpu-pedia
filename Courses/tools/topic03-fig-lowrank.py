# -*- coding: utf-8 -*-
r"""专题三 · §5「白送的那段 vs 赌的那段」—— 让读者自己核数字

⭐⭐⭐ 2026-09-13 新画。调研发现：planetbanatt（planetbanatt.net/articles/mla.html）
   用一段 Manim 动画把「秩够的时候，低秩分解是无损的」画成
   **一个写满真整数的矩阵拆开再复原，数字一模一样** ——&nbsp;
   「无损」是让你自己核，不是被断言的。

⛔ 但**他只画了左边那栏**。而本课的比喻里「白送的那段 vs 赌的那段」是**两段**：
   · 白送 ＝ 一摞复印件换成一张原稿（数学恒等，零损失）
   · 赌   ＝ 把原稿缩印到 512 维（秩不够了，有损）
   ⭐ 右边那栏**全网没人画**。这张图两栏都画，都用真数字。

📌 数字当场算，脚本里断言：
   左栏取一个**真·秩 2** 的 4×4（用两个向量外积造出来），rank=2 分解 → 误差为 0；
   右栏取一个**真·秩 3** 的 4×4，硬压到 rank=1 → 误差标红。
   ⚠️ 用 numpy 的 SVD 取最优低秩逼近（Eckart–Young），
   所以右栏那个误差是**理论下界**，不是「我们没调好」。
"""
import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def best_rank(M, r):
    U, S, Vt = np.linalg.svd(M)
    return (U[:, :r] * S[:r]) @ Vt[:r], S


def main():
    # ── 左栏：真·秩 2 的矩阵，用 rank=2 去拆 → 无损 ───────────────
    a1, b1 = np.array([2, 1, 3, 1]), np.array([1, 2, 1, 0])
    a2, b2 = np.array([1, 0, 1, 2]), np.array([0, 1, 2, 1])
    A = np.outer(a1, b1) + np.outer(a2, b2)
    assert np.linalg.matrix_rank(A) == 2
    A2, _ = best_rank(A, 2)
    assert np.abs(A - A2).max() < 1e-9            # 白送：误差恒等于 0

    # ── 右栏：真·秩 3 的矩阵，硬压到 rank=1 → 有损 ────────────────
    B = (np.outer([3, 1, 2, 1], [1, 2, 0, 1])
         + np.outer([0, 2, 1, 1], [2, 0, 1, 1])
         + np.outer([1, 1, 0, 2], [0, 1, 2, 0]))
    assert np.linalg.matrix_rank(B) == 3
    B1, SV = best_rank(B, 1)
    err = np.abs(B - B1)
    assert err.max() > 1.0                        # 赌：确实差了一截

    f = Fig(W, "低秩分解的两段：秩够的时候拆开再复原，数字一模一样，这一段是白送的；"
               "秩不够硬压，复原出来的数字就差了一截，这一段是在赌")
    f.marks = set()
    y0 = f.header(
        "MLA 省的那 56.9 倍，<tspan font-weight=\"700\">一半是白送的，"
        "一半是在赌</tspan>",
        "⭐ 两栏都是真数字，<tspan font-weight=\"700\">你可以自己核</tspan> ——&#160;"
        "不用相信我们",
        [(GR, "白送：秩够，恒等"), (RD, "赌：秩不够，有损")])

    U = 46

    def grid(x, y, M, ref=None, col=INK, tint="#fff"):
        n, m = M.shape
        for i in range(n):
            for j in range(m):
                v = M[i, j]
                bad = ref is not None and abs(v - ref[i, j]) > 1e-6
                f.box(x + j * U, y + i * U, U - 3, U - 3,
                      "#fce8e6" if bad else tint, RD if bad else LINE2, 3)
                f.t(x + j * U + (U - 3) / 2.0, y + i * U + 30,
                    ("%.1f" % v) if abs(v - round(v)) > 1e-6 else "%d" % round(v),
                    RD if bad else col, bad, 17, "middle")
        return x + m * U

    def column(x0, title, sub, col, M, R, r, verdict, note):
        f.t(x0, y0 + 30, title, col, True, 24)
        f.t(x0, y0 + 58, sub, GY, size=17)
        gy = y0 + 82
        f.t(x0, gy - 10, "原来的矩阵", GY2, size=16)
        grid(x0, gy, M)
        ax = x0 + 4 * U + 26
        f.t(ax, gy + 2 * U, "拆成两块", col, True, 18)
        f.t(ax, gy + 2 * U + 24, "（秩 %d）" % r, GY2, size=16)
        # ⛔ 第一版这两块按原尺寸画（4U 高／宽），横的那块直接压到右边网格上。
        #   ⭐ 因子块只是**示意形状**（高瘦 × 矮宽），按中间那段空隙缩放就行。
        U2 = 20
        f.box(ax, gy + 2 * U + 44, 13 * r, 4 * U2, col, col, 3)
        f.t(ax + 13 * r + 9, gy + 2 * U + 44 + 2 * U2 + 6, "×", GY, True, 18)
        f.box(ax + 13 * r + 22, gy + 2 * U + 44, 4 * U2, 13 * r, col, col, 3)
        bx = ax + 190
        f.t(bx, gy - 10, "再乘回来", GY2, size=16)
        grid(bx, gy, R, ref=M, col=col)
        f.box(x0, gy + 4 * U + 22, bx + 4 * U - x0, 86, "#fff", col, 10)
        f.t(x0 + 18, gy + 4 * U + 50, verdict, col, True, 21)
        for k, ln in enumerate(note):
            f.t(x0 + 18, gy + 4 * U + 76 + k * 24, ln, GY, size=17,
                w=bx + 4 * U - x0 - 36)

    column(40, "① 白送的那一段", "一摞复印件 →　只留一张原稿", GR,
           A, A2, 2, "✅ 每一个数都一模一样",
           ["秩够的时候，低秩分解是<tspan font-weight=\"700\">恒等变换</tspan>",
            "——　零损失，不用做实验。"])
    column(740, "② 赌的那一段", "把原稿缩印 ——　秩不够了", RD,
           B, B1, 1, "⛔ 红格子就是差出来的",
           ["这已经是<tspan font-weight=\"700\">最优</tspan>的 rank-1 逼近",
            "（SVD 的理论下界）——　不是没调好，是压不下去。"])

    yy = y0 + 82 + 4 * U + 126
    yy = f.band(yy, "info", "⭐⭐ 所以看任何一个压缩方案，都先把它拆成这两段", [
        "<tspan font-weight=\"700\">哪一段是「表示冗余」白送的？</tspan>"
        "——&#160;那一段不用看掉点，它是恒等变换。"
        "MLA 把 K、V 合进一个共享隐向量，这一步就属于白送（本课算出来是 4.571×）。",
        "<tspan font-weight=\"700\">哪一段是在赌「它低秩／稀疏／可近似」？</tspan>"
        "——&#160;那一段<tspan font-weight=\"700\">必须看掉点</tspan>。"
        "MLA 把它压到 512 维，这一步是赌（12.4×）。两段相乘才是 56.9×。",
        "⭐ 这条判据对 KV、权重、激活、梯度<tspan font-weight=\"700\">全都适用</tspan>"
        " ——&#160;它不是 MLA 专属的。",
    ])
    yy = f.src(yy + 16,
               "「拆开再复原、数字一模一样」这个装置偷自 planetbanatt.net/articles/mla.html "
               "的 Manim 动画；⭐ 但他只画了左栏，右栏是本课补的",
               "两栏的矩阵都由脚本当场构造并断言：左栏真秩 2、右栏真秩 3；"
               "低秩逼近用 numpy SVD（Eckart–Young 最优解）",
               "⚠️ 4×4 是<tspan font-weight=\"700\">示意尺寸</tspan>；"
               "MLA 真实是 32,768 → 576")
    f.save("fig3-lowrank.svg", yy + 6)


main()
