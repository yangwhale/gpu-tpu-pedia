# -*- coding: utf-8 -*-
r"""专题四 · §3.3b「Muon 从零讲」第四张：那五轮迭代到底在干什么

2026-11-08 补。「Newton–Schulz 五轮」原来在讲稿里写着「超纲，今天不讲」。
现场讲通的说法：奇数次矩阵多项式对每个频率的强度**单独**套同一个多项式，模式不动
（XXᵀX ＝ U Σ³ Vᵀ）。于是整件事退化成一个一维问题：一个数过五遍多项式，变成多少。

三格：
  Ⓐ 同一个多项式，五遍之后长什么样（横轴对数）：经典 1.5x−0.5x³ 太慢，0.001 五遍后还是 0.0076；
     Muon 的系数让小数每遍放大三倍多，从 0.001 起五遍就到 0.47；超过约 1.26 会炸 → 所以要先除以总大小。
  Ⓑ 三个频率（强度 10／1／0.01，缩放后 0.995／0.0995／0.001）逐轮的样子：
     在 0.7–1.1 之间来回晃，五轮后 0.70／0.71／0.47。
  Ⓒ 能量占比：99%／1%／0% → 41%／41%／18%。

数全部来自 topic04_muon_nums.py（现算并断言）。
刻意没画：系数是怎么调出来的（作者用梯度法在约束下最大化 0 附近斜率，见出处），这一讲不要求。
"""
import math

import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)
import topic04_muon_nums as N

W = 1400
P5_5 = lambda x: N.iterate(N.p5, np.array([x]))[-1][0]
P3_5 = lambda x: N.iterate(N.p3, np.array([x]))[-1][0]
assert abs(P3_5(0.001) - 0.0076) < 0.0001
assert abs(P5_5(0.001) - 0.4705) < 0.001
assert abs(float(np.sqrt((N.RAW ** 2).sum())) - 10.05) < 0.005     # 图上写「除以总大小 10.05」


def main():
    f = Fig(W, "五轮迭代在干什么。矩阵多项式对每个频率的强度单独套同一个多项式，方向不动。"
               "第一格：横轴是进来的强度（对数），纵轴是五轮之后的强度；经典三次多项式五轮后 0.001 只到 0.0076，"
               "Muon 的系数让 0.001 五轮到 0.47，大于约 0.05 的都落在 0.68 到 1.2 之间；超过约 1.26 会炸，所以先除以总大小。"
               "第二格：三个频率 0.995、0.0995、0.001 逐轮来回晃，五轮后 0.70、0.71、0.47。"
               "第三格：能量占比从 99、1、0 变成 41、41、18")
    y0 = f.header("那五轮在干什么　——　<tspan font-weight=\"700\">每个强度各过五遍同一个多项式，方向一根不动</tspan>",
                  "矩阵乘出来的奇数次多项式，只作用在强度上：XXᵀX ＝ U Σ³ Vᵀ（U、V 原样，强度 σ 变成 σ³）。"
                  "所以只要看「一个数过五遍，变成多少」",
                  [(GY2, "经典 1.5x − 0.5x³"), (GR, "Muon 的系数 (3.4445, −4.7750, 2.0315)"),
                   (BL, "频率 ①"), (OR, "频率 ②"), (PU, "频率 ③")])

    # ══ Ⓐ ═══════════════════════════════════════════════════════
    PH = 400
    py = f.panel(0, y0, W, PH, "Ⓐ 进来多大，五轮后多大", GR, sub="横轴是对数：每一格差十倍")
    X0, X1, YB, YT = 110, 860, py + 320, py + 50
    LMIN, LMAX = -3.2, math.log10(1.35)
    YMAX = 1.3

    def xp(v):
        return X0 + (math.log10(v) - LMIN) / (LMAX - LMIN) * (X1 - X0)

    def yp(v):
        return YB - v / YMAX * (YB - YT)

    # 稳定带
    for v in (0.68, 1.2):
        f.line(X0, yp(v), X1, yp(v), GR, 1.1, dash="5 4", arrow=False)
    f.t(X0 + 100, yp(0.6), "两条虚线之间（0.68 ～ 1.2）：「差不多是 1」", GR, True, 13)
    # 炸区
    f.box(xp(N.BLOW), YT, X1 - xp(N.BLOW) + 10, YB - YT, "#fce8e6", "none", 0)
    f.t(xp(N.BLOW) - 6, YT - 6, "超过 %.2f：越推越大" % N.BLOW, RD, True, 13, "end")
    # 轴
    f.line(X0, YB, X1 + 14, YB, GY2, 1.2, arrow=False)
    f.line(X0, YB, X0, YT - 10, GY2, 1.2, arrow=False)
    for e in (-3, -2, -1, 0):
        v = 10 ** e
        f.line(xp(v), YB, xp(v), YB + 6, GY2, 1, arrow=False)
        f.t(xp(v), YB + 22, ("%g" % v), GY, size=12.5, anchor="middle")
    for v in (0.5, 1.0):
        f.line(X0 - 6, yp(v), X0, yp(v), GY2, 1, arrow=False)
        f.t(X0 - 10, yp(v) + 4, "%g" % v, GY, size=12.5, anchor="end")
    f.t((X0 + X1) / 2, YB + 46, "进来的强度（已除以总大小）", GY, size=13, anchor="middle")
    f.t(X0 - 4, YT - 18, "五轮后的强度", GY, size=13)
    # 两条曲线
    for fn, col, sw in ((P3_5, GY2, 2.0), (P5_5, GR, 2.8)):
        pts = []
        L = LMIN
        while L <= math.log10(1.25):
            v = 10 ** L
            pts.append("%.1f %.1f" % (xp(v), yp(min(fn(v), YMAX))))
            L += 0.01
        f.path("M " + " L ".join(pts), col, sw, arrow=False)
    # 三个例子点
    for v, out, col in zip(N.SCALED, N.FINAL, (BL, OR, PU)):
        f.box(xp(v) - 6, yp(out) - 6, 12, 12, col, "#fff", 6, 1.5)
    f.t(xp(N.SCALED[2]) + 12, yp(N.FINAL[2]) + 22, "0.001 → %.2f" % N.FINAL[2], PU, True, 13)
    f.t(xp(N.SCALED[2]) + 12, yp(P3_5(0.001)) - 10, "经典的：0.001 → %.4f" % P3_5(0.001), GY, True, 13)
    # 右侧三句
    tx = 900
    f.t(tx, py + 70, "① 小数每轮放大 %.2f 倍" % N.A, GR, True, 15)
    f.t(tx, py + 94, "（0 附近的斜率就是第一个系数）", GY, size=12.5)
    f.t(tx, py + 136, "② 大的不会冲出去：落在 0.68～1.2", GR, True, 15)
    f.t(tx, py + 160, "不追求正好等于 1，「差不多」就够用", GY, size=12.5)
    f.t(tx, py + 202, "③ 先除以总大小，才能保证 ≤ 1", RD, True, 15)
    f.t(tx, py + 226, "总大小 ＝ 所有强度平方和开根号，", GY, size=12.5)
    f.t(tx, py + 248, "比哪一个强度都大，一除全部 ≤ 1", GY, size=12.5)
    f.t(tx, py + 290, "经典的 1.5x − 0.5x³ 稳，但太慢：", INK, True, 14)
    f.t(tx, py + 314, "0.001 五轮后还不到 0.01", INK, True, 14)
    f._pan = None

    # ══ Ⓑ ═══════════════════════════════════════════════════════
    PH2 = 330
    py2 = f.panel(0, py + PH + 22, W, PH2, "Ⓑ 三个频率，一轮一轮看", BL,
                  sub="进来时强度 10 ／ 1 ／ 0.01，先除以总大小 10.05")
    X0, X1, YB, YT = 110, 860, py2 + 260, py2 + 50

    def xr(k):
        return X0 + k / 5.0 * (X1 - X0)

    def yr(v):
        return YB - v / 1.2 * (YB - YT)

    for v in (0.7, 1.1):
        f.line(X0, yr(v), X1, yr(v), GR, 1.1, dash="5 4", arrow=False)
    f.line(X0, YB, X1 + 10, YB, GY2, 1.2, arrow=False)
    f.line(X0, YB, X0, YT - 6, GY2, 1.2, arrow=False)
    for k in range(6):
        f.t(xr(k), YB + 22, "第 %d 轮" % k if k else "缩放后", GY, size=12.5, anchor="middle")
    for v in (0.5, 1.0):
        f.t(X0 - 10, yr(v) + 4, "%g" % v, GY, size=12.5, anchor="end")
    for j, col in enumerate((BL, OR, PU)):
        pts = ["%.1f %.1f" % (xr(k), yr(N.TRACE[k][j])) for k in range(6)]
        f.path("M " + " L ".join(pts), col, 2.4, arrow=False)
        for k in range(6):
            f.box(xr(k) - 4, yr(N.TRACE[k][j]) - 4, 8, 8, col, col, 4)
    for j, (col, dy) in enumerate(((BL, -10), (OR, 18), (PU, 4))):
        f.t(xr(5) + 12, yr(N.FINAL[j]) + dy, "%.2f" % N.FINAL[j], col, True, 14)
    tx = 960
    f.t(tx, py2 + 76, "进来：0.995 ／ 0.0995 ／ 0.001", INK, True, 15)
    f.t(tx, py2 + 104, "五轮：%.2f ／ %.2f ／ %.2f" % tuple(N.FINAL), GR, True, 15)
    f.t(tx, py2 + 146, "大的那个：在 0.7 和 1.1 之间来回晃", GY, size=13.5)
    f.t(tx, py2 + 172, "小的那两个：每轮放大三倍多，追上来", GY, size=13.5)
    f.t(tx, py2 + 214, "差一千倍 → 差不到两倍", INK, True, 15)
    f._pan = None

    # ══ Ⓒ ═══════════════════════════════════════════════════════
    PH3 = 190
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3, "Ⓒ 能量（强度²）怎么分", OR,
                  sub="一根条 ＝ 100%，三段 ＝ 三个频率各占多少")
    BX, BWD = 220, 900
    for r, (lab, share) in enumerate((("五轮之前", N.E_BEFORE), ("五轮之后", N.E_AFTER))):
        yy = py3 + 36 + r * 66
        f.t(BX - 16, yy + 30, lab, INK, True, 15, "end")
        x = BX
        for j, col in enumerate((BL, OR, PU)):
            w = share[j] * BWD
            if w > 0.5:
                f.box(x, yy + 8, w, 36, col, "#fff", 2)
            if w > 44:
                f.t(x + w / 2, yy + 32, "%.0f%%" % (share[j] * 100), "#fff", True, 14, "middle")
            x += w
        if r == 0:
            f.t(BX + BWD + 12, yy + 32, "② 1%　③ 0%", GY, True, 13)
    f._pan = None

    yb = f.band(py3 + PH3 + 22, "ok", "带走一句：五轮迭代 ＝ 把每根推子都推到「差不多一样高」", [
        "推的是强度，方向一根没碰；之所以是迭代而不是直接拆开算，是因为几次矩阵乘在显卡上快，还能用 BF16 算。",
    ])
    yb = f.src(yb + 10,
               "出处：系数 (3.4445, −4.7750, 2.0315)、5 轮、先除以 Frobenius 范数：Keller Jordan 博客与公开实现 muon.py。"
               "博客写明：系数按「0 附近斜率尽量大」来调，代价是只收敛到 1 附近而不是正好 1，实测不伤 loss；"
               "Newton–Schulz 能在 bfloat16 里稳定地跑，而另一种（耦合牛顿迭代）至少要 float32。",
               "出处：「奇数次多项式与奇异值分解可交换」：Bernstein《Deriving Muon》（jeremybernste.in/writing/deriving-muon），"
               "经典三次 1.5x − 0.5x³ 出自 Kovarik 1970，也是该文举的最简单例子。",
               "本课推导（topic04_muon_nums.py 现算并断言）：5 轮后起点 ≥ 0.001 的落在 0.47～1.20；"
               "多跑几十轮在 0.68～1.13 之间晃；p(x) ＝ x 的正根约 1.264，超过它就被越推越大（1.25 五轮还在，1.3 五轮就溢出）。",
               "注：公开实现的说明里写的是「输出强度大致在 0.5～1.5」，比这里的单个多项式现算宽一些；"
               "我们只报自己算出来的数。")
    f.save("fig4-ns.svg", yb + 14)


main()
