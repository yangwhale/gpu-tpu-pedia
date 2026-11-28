# -*- coding: utf-8 -*-
r"""专题八 · §10.3「同一份程序复跑也要逐位」的原理图（2026-11-28 夜补）。

fig8-race  回答：同一份程序、同样的输入，复跑为什么会偶尔差一位？
           左：内核往同一个累加位置连发两笔「带加法的写」，中间只等第一笔「读完源」就发第二笔 —— 第二笔可能先落地，两笔加法颠倒；
           右：每块之后发一次小拷贝、等它「写完」（写栅栏），之前发出的写都已落地，永远先 1 后 2。
           底下用 1 ＋ 2²⁴ − 2²⁴ 这组数把「颠倒」变成看得见的 0 和 1（就是对齐记录单卡微基准挑的那组数）。
数据：对齐记录混元 4 案例 19（第 99–138 行机理与单卡微基准，第 164–205 行修法与代价；实测）。
⚠️ 刻意没画的：时间轴没有刻度 —— 每笔写要多久没有测过，画上刻度就是编；块数只画 2 块（真实内核按查询块一块一块发）。
"""
import numpy as np

import topic08_figfont  # noqa: F401  专题八局部字号地板 13.1（须在 topic03_draw 的 wpx 之前）
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2
Fig.BAND_KEEP_LINES = 0
import topic08_numbers as N

W = 1400

# ① 那组数在 FP32 里真算：起点 1，块 1 加 A ＝ 2²⁴，块 2 加 B ＝ −2²⁴
ONE, A, B = np.float32(1), np.float32(2 ** 24), np.float32(-2 ** 24)
IN_ORDER = float((ONE + A) + B)          # 1 ＋ 2²⁴ 舍成 2²⁴（平局取偶），再减得 0
SWAPPED = float((ONE + B) + A)           # 1 − 2²⁴ 精确，再加得 1
assert (IN_ORDER, SWAPPED) == (0.0, 1.0), (IN_ORDER, SWAPPED)
STALE_N, STALE_EL, FENCED_EL = N.RACE_STALE


def lane(f, x, y, w_read, w_write, col, fill, lab, lab_x=None):
    """一笔带加法的写：先「读源」（浅），再「写进累加器」（深，斜纹表示还在路上）。返回写落地的 x。"""
    f.t((x if lab_x is None else lab_x) - 14, y + 22, lab, col, True, 14, "end")
    f.box(x, y, w_read, 34, fill, col, 5, sw=1.4)
    f.t(x + w_read / 2, y + 22, "读源", col, True, 13, "middle")
    f.box(x + w_read, y, w_write, 34, "#fff", col, 5, sw=1.4, dash="5 3")
    f.t(x + w_read + w_write / 2, y + 22, "写进累加器", col, True, 13, "middle")
    return x + w_read + w_write


def fig_race():
    f = Fig(W, "左右两块，横轴都是时间。两块都画了两笔带加法的写，块甲和块乙都要往累加器里同一个位置加一笔，每笔先读源、再写进累加器。"
               "左块只等读：块甲的源一读完，wait 就返回，内核马上发块乙；块乙的写比块甲先落地，两笔加法颠倒。"
               "右块每块之后发一次小拷贝并等它写完，这一等把之前发出的写都围住，块乙永远在块甲落地之后才开始。"
               "底下用累加器原值 1、块甲加 2 的 24 次方、块乙加负 2 的 24 次方这组数：按顺序在 FP32 里得 0，颠倒得 1。FP32 只有 24 位有效位，1 碰上 2 的 24 次方会被吞掉。"
               "单卡微基准：不加栅栏 75 次里 3732 个元素读到旧值，加栅栏 0 个；修法每步约多 3 到 6 毫秒。")
    y0 = f.header("复跑偶尔差一位：内核只等「读完」，没等「写完」",
                  "两笔加法都落在累加器的同一个位置上，先后一换，FP32 舍入就不同；横轴是时间，没有刻度（每笔写要多久没测过）",
                  [(BL, "块甲的加法"), (OR, "块乙的加法"), (GR, "写栅栏：等一次写完")])
    PH, PW, GAP = 388, 680, 40
    X0 = 96                                   # 两块里时间线的起点（相对面板左边）

    # ── Ⓐ 只等读 ─────────────────────────────────────────
    py = f.panel(0, y0, PW, PH, "Ⓐ 只等读（原来的写法）", RD)
    l1 = lane(f, X0, py + 46, 110, 330, BL, "#e8f0fe", "块甲")
    wx = X0 + 110
    f.line(wx, py + 36, wx, py + 150, RD, 1.6, dash="6 4", arrow=False)
    f.t(wx + 6, py + 40, "wait 返回：只说明源读完了", RD, True, 12.5)
    l2 = lane(f, wx + 16, py + 98, 90, 150, OR, "#fef7e0", "块乙", lab_x=X0)
    assert l2 < l1                              # 画面上块 2 确实先落地
    ay = py + 196
    f.line(X0 - 6, ay, PW - 30, ay, GY2, 1.6, arrow=False)
    f.t(X0 - 14, ay + 5, "累加器", INK, True, 13.5, "end")
    f.line(l1, py + 80, l1, ay - 14, BL, 1.4, dash="2 3", arrow=False)
    f.line(l2, py + 132, l2, ay - 14, OR, 1.4, dash="2 3", arrow=False)
    for x, col, n in ((l2, OR, "乙"), (l1, BL, "甲")):
        f.box(x - 13, ay - 13, 26, 26, col, col, 13)
        f.t(x, ay + 5, n, "#fff", True, 13.5, "middle")
    f.t((l1 + l2) / 2, ay + 40, "先乙后甲：颠倒了", RD, True, 14, "middle")
    f.t(X0 - 66, py + 274, "累加器原值 1，块甲加 2²⁴，块乙加 −2²⁴", INK, True, 14)
    f.t(X0 - 66, py + 302, "按甲、乙的顺序 → %d　　颠倒 → %d" % (IN_ORDER, SWAPPED), RD, True, 14)
    f.t(X0 - 66, py + 330, "FP32 只有 24 位有效位，1 碰上 2²⁴ 会被吞掉（第七节的吞小项）", INK, size=13)
    f.t(X0 - 66, py + 354, "大多数时候写来得及落地，偶尔来不及：%d 次复跑 %d 次分开" % (N.RACE_RUNS, N.RACE_SPLIT), GY, size=13)
    f._pan = None

    # ── Ⓑ 等写完 ─────────────────────────────────────────
    RX = PW + GAP
    py2 = f.panel(RX, y0, W - RX, PH, "Ⓑ 每块之后等一次「写完」（写栅栏）", GR)
    b1 = lane(f, RX + X0, py2 + 46, 100, 180, BL, "#e8f0fe", "块甲")
    fx = b1 + 6
    f.box(fx, py2 + 72, 60, 30, "#e6f4ea", GR, 5, sw=1.4)
    f.t(fx + 30, py2 + 92, "栅栏", GR, True, 13, "middle")
    f.t(fx, py2 + 62, "发一次小拷贝，等它写完", GR, True, 12.5)
    b2 = lane(f, fx + 70, py2 + 98, 60, 100, OR, "#fef7e0", "块乙", lab_x=RX + X0)
    assert b2 < W - 40
    assert b2 > b1
    ay2 = py2 + 196
    f.line(RX + X0 - 6, ay2, W - 30, ay2, GY2, 1.6, arrow=False)
    f.t(RX + X0 - 14, ay2 + 5, "累加器", INK, True, 13.5, "end")
    f.line(b1, py2 + 80, b1, ay2 - 14, BL, 1.4, dash="2 3", arrow=False)
    f.line(b2, py2 + 132, b2, ay2 - 14, OR, 1.4, dash="2 3", arrow=False)
    for x, col, n in ((b1, BL, "甲"), (b2, OR, "乙")):
        f.box(x - 13, ay2 - 13, 26, 26, col, col, 13)
        f.t(x, ay2 + 5, n, "#fff", True, 13.5, "middle")
    f.t((b1 + b2) / 2, ay2 + 40, "永远先甲后乙", GR, True, 14, "middle")
    f.t(RX + X0 - 66, py2 + 274, "累加器原值 1，块甲加 2²⁴，块乙加 −2²⁴", INK, True, 14)
    f.t(RX + X0 - 66, py2 + 302, "永远按甲、乙的顺序 → %d" % IN_ORDER, GR, True, 14)
    f.t(RX + X0 - 66, py2 + 330, "不改算术，只多等一下：每步约 ＋%d–%d ms" % N.RACE_FENCE_MS, GY, size=13)
    f.t(RX + X0 - 66, py2 + 354, N.RACE_FENCE_PCT_T, GY, size=13)
    f._pan = None

    yb = f.band(y0 + PH + 20, "ok", "wait 返回不等于写已落地；多等一次「写完」，之前发出的写就都到了", [
        "",
    ])
    yb = f.src(yb + 10,
               "📌 实测（对齐记录混元 4 案例 19，混元 4 的一个内部配置，开稀疏注意力，TPU v7x 128 个设备）：稀疏注意力反向把 dKV 的部分行用「带加法的间接 DMA」"
               "散射回 SparseCore 共享内存里的累加器；JAX 的 SparseCore 下沉代码自己注释写着散射没有写栅栏、wait() 只等读。"
               "单卡微基准每行置 1，加 2²⁴ 只等读，再加 −2²⁴：按序得 0、颠倒得 1、其他值是丢写；收尾形态（只等读就读回）不加栅栏 %d 次里 %d 个元素读到旧值、加栅栏 %d 个；"
               "块间颠倒在单卡孤立时序下一次也没碰上（约 1e10 个元素），整模型里靠正对照坐实。整模型那 %d 次分开推断是两笔颠倒，也可能是收尾读到旧值；写栅栏两种都盖住。" % (STALE_N, STALE_EL, FENCED_EL, N.RACE_SPLIT),
               "📌 修法：每个查询块的加法排空后，往一段永远不读回的丢弃行发一次 (8, 128) FP32（4 KiB）本地拷贝并等它写完；SparseCore 上这种写等待会把同一个子核之前发出的流一起围住。"
               "不改算术；256K 整模型每步约 ＋%d–%d ms（%s）。" % (N.RACE_FENCE_MS + (N.RACE_FENCE_PCT_T,)),
               "⚠️ 图里只画 2 块、时间轴不带刻度：真实内核按查询块一块一块发，每笔写多久没有量过。")
    f.save("fig8-race.svg", yb + 14)


fig_race()
