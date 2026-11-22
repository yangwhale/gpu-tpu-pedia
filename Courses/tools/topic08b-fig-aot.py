# -*- coding: utf-8 -*-
r"""专题九 · 9.6「编译时间：从 25 分钟到两分半」的阶梯图（2026-11-22 补）。

fig8b-aot   回答：离线编译一格为什么要 25 分钟，三刀之后为什么只要两分半？—— 每一步对上一步，半层和 128 片两根条：
            647 / 1522 → 361 / 898（两进程）→ 277 / 643（牛顿迭代改 JAX 循环）→ 268 / 614（同形参数批量）→ 154 / 153（视觉塔 scan）秒；
            右边是优化前程序的指令数。

⛔ 数据全部 import 自 topic08b_numbers.py（AOT_*），断言在那边。不写编译器开关名、机器名、桶路径。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def fig_aot():
    st = N.AOT_STEPS
    f = Fig(W, "离线编译一格（正常步、回放步两个程序）的墙钟时间，独占一台编译机。每一行叠在上一行之上："
               + "；".join("%s，半层 %d 秒、128 片 %d 秒，优化前程序 %.1f 千条 / %.1f 千条指令" % (n, a, b, k[0], k[1]) for _i, n, a, b, k in st)
               + "。最后 128 片和半层一样快：两个程序已经一样大。")
    y0 = f.header("编译一格：从 25 分钟到两分半",
                  "每行两根条：上 ＝ 半层，下 ＝ 128 片；条长 ＝ 编一格的秒数（独占一台编译机）；右边 ＝ 优化前程序的指令数（千条）",
                  [(BL, "半层"), (OR, "128 片"), (GR, "最后一刀以后")])
    RH = 86
    PH = 60 + RH * len(st)
    py = f.panel(0, y0, W, PH, "离线编译实测（每一步叠在上一步之上）", BL)
    X0, SC = 470, 0.38
    for i, (k, name, half, full, ins) in enumerate(st):
        yy = py + 40 + i * RH
        f.t(30, yy + 22, ("第 %d 刀" % (k - 1)) if k >= 2 else ("流程" if k == 1 else "原样"), GY, True, 13)
        f.t(110, yy + 22, name, INK, True, 15, w=X0 - 130)
        for j, (v, col) in enumerate(((half, BL), (full, OR))):
            y2 = yy + 4 + j * 30
            c = GR if k == st[-1][0] else col
            f.box(X0, y2, v * SC, 24, c, c, 3)
            f.t(X0 + v * SC + 10, y2 + 18, "%d 秒（%.1f 分钟）" % (v, v / 60.0), c, True, 14)
        f.t(1300, yy + 22, "%.1f k" % ins[0], BL, True, 14)
        f.t(1300, yy + 52, "%.1f k" % ins[1], OR, True, 14)
    f.t(1300, py + 22, "指令数", GY, True, 13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "编译时间跟着程序大小走：把展开的 Python 循环换成 JAX 循环，程序缩了一大半", [])
    yb = f.src(yb + 10,
               "实测：独占一台 192 vCPU 的编译机；第 1 步起两个程序分两个进程同时编。原来在编译机上十几格挤着编，128 片一格要 %d–%d 分钟。" % N.AOT_CROWD_MIN,
               "实测：半层一格 %d%% 的时间在 XLA 编译里；循环外最大的那个计算优化前有 %.1f 千条（半层）、%.1f 千条（128 片）指令，其中优化器约占 %d%%。" % ((N.AOT_XLA_PCT,) + N.AOT_ENTRY_K + (N.AOT_OPT_PCT,)),
               "诊断格（故意改程序，只为看清时间花在哪，不是方案）：%s %d%%；%s %d%%。" % (N.AOT_DIAG[0][0], N.AOT_DIAG[0][1], N.AOT_DIAG[1][0], N.AOT_DIAG[1][1]),
               "两进程并行编：正常步的产物逐字节相同；回放步的程序逐字节相同，可执行文件只差 3 处模块编号。三刀改的是程序本身，数值见正文。")
    f.save("fig8b-aot.svg", yb + 14)


if __name__ == "__main__":
    fig_aot()
