# -*- coding: utf-8 -*-
r"""专题九 · 第二节「先把数量准」的三张图。

fig8b-numerator  回答：静态公式数的分子里，哪些不是模型真正需要的？—— 一根 608.55 的条，按四项减减加加到 325.18。
                 数字：topic08b_numbers（截层，第 5–9 步；百分比四舍五入，断言差在 1 TFLOP 以内）。
fig8b-docmask    回答：为什么跨文档的注意力能占四成？—— 一行拼着很多篇文档，注意力只在每篇里面做。
                 左：整行因果的大三角；右：同一行按文档切成小三角。示意：8 篇文档，长度是编的，比例现算；真实数据的比例写在旁边。
fig8b-ruler      回答：「快了 2.7%」算不算数？—— 同一分配里同一份程序跑五次，最大差 0.002 秒；跨作业的对照最大差 0.018 秒。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def fig_numerator():
    f = Fig(W, "一根横条从静态公式数的分子 608.55 TFLOP 开始，减掉跨文档的注意力 41.7%、减掉填充 8.2%，"
               "加上静态公式漏计的 MTP 输出头 2.8%、视觉塔按实际图像块 0.6%，得到有效分子 325.18 TFLOP，只有原来的 53%。")
    y0 = f.header("静态公式数的分子，一半不是模型真正需要的",
                  "条长 ＝ 每设备每步的运算量（TFLOP）；红 ＝ 扣掉的，绿 ＝ 补上的，蓝 ＝ 剩下的有效分子",
                  [(GY2, "静态公式"), (RD, "扣掉"), (GR, "补上"), (BL, "有效分子")])
    rows = [("静态公式", N.NUM_STATIC_SH, None)]
    cur = N.NUM_STATIC_SH
    for name, p in N.NUM_ITEMS:
        d = N.NUM_STATIC_SH * p / 100
        rows.append((name, d, cur))
        cur += d
    rows.append(("有效分子", N.NUM_EFF_SH, None))
    PH = 60 + 52 * len(rows)
    py = f.panel(0, y0, W, PH, "20 层底座，第 5–9 步，每设备每步（实测 ＋ 逐项复算）", BL)
    X0, XW = 330, 1000

    def X(v):
        return X0 + v / 650.0 * XW
    for i, (name, v, start) in enumerate(rows):
        yy = py + 34 + i * 52
        f.t(X0 - 16, yy + 22, name, INK, True, 15, "end")
        if start is None:
            col = GY2 if i == 0 else BL
            f.box(X0, yy, X(v) - X0, 32, col, col, 3)
            f.t(X(v) + 12, yy + 22, "%.2f TFLOP" % v, INK, True, 15)
        else:
            a, b = (start + v, start) if v < 0 else (start, start + v)
            col = RD if v < 0 else GR
            f.box(X(a), yy, max(X(b) - X(a), 3), 32, col, col, 3)
            f.t(X(b) + 12, yy + 22, "%+.1f%%（%+.0f）" % (v / N.NUM_STATIC_SH * 100, v), col, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "有效分子只有静态分子的 %.0f%%：同一次运行，两种口径的 MFU 也差这么多" % (N.NUM_EFF_FRAC_SH * 100), [])
    yb = f.src(yb + 10,
               "📌 实测与复算：静态分子 608.55 TFLOP 是训练日志里按模型形状记的每设备每步运算量；有效分子按同一步的文档分段逐行重算（跨文档配对不计、填充 token 不计、视觉塔按实际图像块计），在线读数与离线重放逐位一致。",
               "📌 四项占静态分子的比例是第 5–9 步的读数，四舍五入到 0.1%；四项相加后与 325.18 差不到 1 TFLOP（断言）。",
               "📌 不计重算：重算是实现的选择，不是模型要的计算。",
               "⚠️ 同一个有效分子的口径，换一格（第一节那一格，半层底座、滤掉长文档）是静态 1142.7、有效 673.5，占 %.0f%%：数据不同，比例不同。" % (N.NUM_EFF_FRAC_HALF * 100))
    f.save("fig8b-numerator.svg", yb + 14)


# 示意用的 8 篇文档长度（编的，加起来 32768）
DOCS = [9200, 6100, 4800, 4200, 3300, 2600, 1500, 1068]
assert sum(DOCS) == 32768
S = sum(DOCS)
FULL = S * (S + 1) / 2
WITHIN = sum(L * (L + 1) / 2 for L in DOCS)
DEMO_FRAC = WITHIN / FULL


def fig_docmask():
    f = Fig(W, "左边是一行 32768 个 token 当成一整篇文档时的因果注意力：一个大三角，每个 token 看它前面所有 token。"
               "右边是同一行实际拼着 8 篇文档（示意），注意力只在每篇里面做：只剩对角线上 8 个小三角，面积占大三角的 %.0f%%。"
               "真实数据一行平均约 28 篇文档，文档内配对只占整行因果的 8.4%% 到 11.2%%。" % (DEMO_FRAC * 100))
    y0 = f.header("一行拼着很多篇文档，注意力只在每篇里面做",
                  "方块里每个点 ＝ 一对 (看的 token, 被看的 token)；涂色的才是要算的；示意用 8 篇文档，长度是编的",
                  [(GY2, "当成一整篇时要算的"), (BL, "按文档切开后要算的")])
    PH = 500
    py = f.panel(0, y0, W, PH, "同一行 32768 个 token（示意）", BL)
    SZ = 380

    def tri(x0, y0_, x1, col, fill):
        f.poly([(x0, y0_), (x0, y0_ + (x1 - x0)), (x1, y0_ + (x1 - x0))], fill, col, 1.2)
    # 左：大三角
    lx, ly = 120, py + 50
    f.box(lx, ly, SZ, SZ, "none", LINE, 2)
    tri(lx, ly, lx + SZ, GY2, "#dadce0")
    f.t(lx + SZ / 2, ly + SZ + 32, "当成一整篇：%.1f 亿对" % (FULL / 1e8), INK, True, 15, "middle")
    # 右：小三角
    rx = 760
    f.box(rx, ly, SZ, SZ, "none", LINE, 2)
    pos = 0
    for L in DOCS:
        a = rx + pos / S * SZ
        b = rx + (pos + L) / S * SZ
        f.poly([(a, ly + (a - rx)), (a, ly + (b - rx)), (b, ly + (b - rx))], "#d2e3fc", BL, 1.2)
        pos += L
    f.t(rx + SZ / 2, ly + SZ + 32, "按文档切开：只剩 %.0f%%" % (DEMO_FRAC * 100), BL, True, 15, "middle")
    f.t(rx + SZ + 20, ly + 120, "真实数据：", INK, True, 15)
    f.t(rx + SZ + 20, ly + 150, "一行约 %d 篇" % N.DOCS_PER_ROW, INK, size=15)
    f.t(rx + SZ + 20, ly + 180, "文档内只占 %.1f%%–%.1f%%" % (N.PAIR_FRAC_OLD * 100, N.PAIR_FRAC_PK * 100), BL, True, 15)
    f.line(lx + SZ + 40, ly + SZ / 2, rx - 40, ly + SZ / 2, GY2, 1.6)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "静态公式按大三角数，模型要的只是对角线上那几个小三角", [])
    yb = f.src(yb + 10,
               "⚠️ 示意：8 篇文档的长度是编的（加起来 32768），小三角占大三角的 %.1f%% 由脚本现算。" % (DEMO_FRAC * 100),
               "📌 实测：真实图文数据一行平均约 28 篇文档；文档内配对占整行因果的比例，在线打包的数据是 8.4%，离线预打包的数据是 11.2%（第 5–9 步）。",
               "📌 实际算的比要的多：注意力核按 1024 × 1024 的块算，块里只要有一个点要算，整块都算。预打包数据某一步平均每个设备保留满因果的 24.2% 的块，要的配对只有 11.2%（本课推导：两者不是同一个分母的精确比，只说明一半左右的块是半掩的）。")
    f.save("fig8b-docmask.svg", yb + 14)


def fig_ruler():
    f = Fig(W, "两行散点，横轴是步时（秒）。上面一行是同一个作业、同一批机器上跑的五格：对照和四格编出来的程序跟对照逐条相同的开关，"
               "步时 5.309 到 5.311 秒，最大差 0.002 秒。下面一行是不同作业里的同一个对照配置，十次，5.300 到 5.318 秒，最大差 0.018 秒。"
               "两条真正生效的开关在同一个作业里是 5.166 和 5.093 秒，远在带外。")
    y0 = f.header("性能的尺子：同一份程序重跑，步时差多少",
                  "横轴 ＝ 步时（秒）；灰带 ＝ 同一行里两两之差的最大值；红点 ＝ 同一个作业里两条真正生效的开关",
                  [(BL, "同一个作业里的五格"), (GY2, "不同作业里的对照"), (RD, "生效的开关")])
    PH = 330
    py = f.panel(0, y0, W, PH, "半层底座、滤掉长文档的数据（实测）", BL)
    LO, HI = 5.08, 5.33
    X0, XW = 230, 1100

    def X(v):
        return X0 + (v - LO) / (HI - LO) * XW
    y1, y2 = py + 80, py + 180
    f.t(X0 - 16, y1 + 6, "同一个作业", INK, True, 15, "end")
    f.t(X0 - 16, y2 + 6, "不同作业", INK, True, 15, "end")
    vs = [v for _n, v in N.SAME_ALLOC]
    f.box(X(min(vs)), y1 - 18, max(X(max(vs)) - X(min(vs)), 3), 36, LINE2, GY2, 3)
    for v in vs:
        f.box(X(v) - 6, y1 - 6, 12, 12, BL, BL, 6)
    f.t(X(max(vs)), y1 - 28, "五格最大差 %.3f 秒（%.2f%%）" % (N.RULER_SAME, N.RULER_SAME_PCT), BL, True, 14, "end")
    f.box(X(min(N.CROSS_JOB)), y2 - 18, X(max(N.CROSS_JOB)) - X(min(N.CROSS_JOB)), 36, LINE2, GY2, 3)
    for v in N.CROSS_JOB:
        f.box(X(v) - 5, y2 - 5, 10, 10, GY2, GY2, 5)
    f.t(X(max(N.CROSS_JOB)), y2 + 40, "十次最大差 %.3f 秒（%.2f%%）" % (N.RULER_CROSS, N.RULER_CROSS_PCT), GY, True, 14, "end")
    for v, lab in ((N.LEGALIZER_S, "生效开关甲 %.3f：快 %.3f 秒 ＝ 尺子 × %d" % (N.LEGALIZER_S, 5.311 - N.LEGALIZER_S, int(N.LEG_X))),
                   (N.SML_S, "生效开关乙 %.3f" % N.SML_S)):
        f.box(X(v) - 7, y1 - 7, 14, 14, RD, RD, 7)
        f.t(X(v), y1 + 34, lab, RD, True, 14, "middle")
    yax = py + PH - 56
    f.line(X0, yax, X0 + XW, yax, GY2, 1.2, arrow=False)
    for i in range(6):
        v = 5.08 + i * 0.05
        f.line(X(v), yax - 5, X(v), yax + 5, GY2, 1.1, arrow=False)
        f.t(X(v), yax + 24, "%.2f 秒" % v, GY, size=13, anchor="middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "同一个作业里比，尺子是 0.002 秒；跨作业比，尺子放宽到 0.018 秒", [])
    yb = f.src(yb + 10,
               "📌 实测：同一个作业里，对照 5.311 秒（第 13–20 步），四格共开了五条开关（有一格同时开两条）、编出来的程序跟对照只差步数常量（离线逐条比过），5.309、5.310、5.309、5.310 秒（第 5–9 步）。",
               "📌 实测：不同作业里同一个对照配置 10 次：5.300、5.311、5.313、5.311、5.313、5.308、5.310、5.314、5.315、5.318 秒。窗口不全相同（作业第一格取第 13–20 步或第 12–19 步，其余取第 5–9 步）。",
               "⚠️ 本课推导：尺子取同一行两两之差的最大值，跟专题八的噪声尺子同一个造法；生效开关甲比对照快 0.145 秒，是同作业尺子的 %.1f 倍。" % N.LEG_X,
               "⚠️ 跨作业为什么更散，素材里没有拆（推断：机器、时段不同）。")
    f.save("fig8b-ruler.svg", yb + 14)


if __name__ == "__main__":
    fig_numerator()
    fig_docmask()
    fig_ruler()
