# -*- coding: utf-8 -*-
r"""专题六 · fig6-ppl.svg（2026-10-20 加，两档共用；当前 L200 用）。

回答：同样压到 4 bit，掉多少由什么决定？
  · 数字：AWQ（arXiv 2306.00978v5）Llama-2-7B、WikiText-2 困惑度：FP16 5.47、AWQ（INT4 g128）5.60、
    直接四舍五入取整（RTN）5.73（L300 12.1 正文与台账）。
  · 画法：一把困惑度刻度尺，三个点落在尺上，越往右越差；两段括弧标「比 16 位多掉的」，不写差值。
"""
from topic03_draw import Fig, wpx, GR, RD, GY, INK, GY2

W = 1400
PTS = [(5.47, "16 位", "不压", GY2), (5.60, "AWQ 4 bit", "挑过缩放", GR), (5.73, "四舍五入 4 bit", "直接取整", RD)]
LO, HI = 5.40, 5.80


def fig_ppl():
    f = Fig(W, "一把困惑度刻度尺，从 5.40 到 5.80，越往右越差。AWQ 那组测试里 Llama-2-7B 在 WikiText-2 上："
               "16 位 5.47；AWQ 挑过缩放再压到 4 bit 是 5.60；直接四舍五入压到 4 bit 是 5.73，比 AWQ 离 16 位远得多")
    y0 = f.header("同样压到 4 bit　——　<tspan font-weight=\"700\">挑缩放的方法决定掉多少</tspan>",
                  "AWQ 那组测试 · Llama-2-7B · WikiText-2 困惑度（越往右越差）",
                  [(GY2, "16 位"), (GR, "AWQ"), (RD, "直接取整")])
    PH = 272
    py = f.panel(0, y0, W, PH, "困惑度刻度尺", INK)
    X0, X1 = 140, W - 140
    def X(v):
        return X0 + (v - LO) / (HI - LO) * (X1 - X0)
    ya = py + 130
    f.line(X0, ya, X1, ya, INK, 1.6, arrow=False)
    for i in range(9):
        v = LO + i * 0.05
        f.line(X(v), ya, X(v), ya + 8, GY2, 1.2, arrow=False)
        f.t(X(v), ya + 28, "%.2f" % v, GY, size=13, anchor="middle")
    for v, lab, sub, col in PTS:
        x = X(v)
        f.box(x - 9, ya - 9, 18, 18, col, col, 9)
        f.t(x, ya - 46, lab, col, True, 17, "middle")
        f.t(x, ya - 24, "%.2f" % v, col, True, 15, "middle")
        f.t(x, ya + 56, sub, GY, size=14, anchor="middle")
    for v, col, dy in ((5.60, GR, 76), (5.73, RD, 96)):
        yb_ = ya + dy
        f.line(X(5.47), yb_, X(v), yb_, col, 2.0, arrow=False)
        f.line(X(5.47), yb_ - 5, X(5.47), yb_ + 5, col, 2.0, arrow=False)
        f.line(X(v), yb_ - 5, X(v), yb_ + 5, col, 2.0, arrow=False)
    f.t(X(5.73) + 16, ya + 100, "比 16 位多掉的", GY, size=14)
    f._pan = None
    yb = f.src(y0 + PH + 6,
               "AWQ：arXiv 2306.00978v5，Llama-2-7B、WikiText-2 困惑度 FP16 5.47、RTN 5.73、AWQ 5.60（INT4，每 128 个数一组缩放）。只这一组模型和数据集。")
    f.save("fig6-ppl.svg", yb + 14)


fig_ppl()
