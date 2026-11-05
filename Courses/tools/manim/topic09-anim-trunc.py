# -*- coding: utf-8 -*-
r"""专题九「GPU→TPU 训练精度对齐」§5.2 的动画：GPU 张量核每 16 个一组抹零，对照就近舍入（TruncVsRound）。
挂在静态图 fig9-truncate 下面。画面和本文件里都不写案例答案。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · 论点是「越加越偏」：抹零每一组只丢一点点，偏差是一组一组攒出来的；就近舍入每组有进有舍，攒不起来。
    静态图只能给几个 K 下的终点值，画不出「同一次求和，抹零那条一路往下走、就近舍入那条一直在 0 附近晃」——
    差别就在「往下走」和「来回晃」这两种运动上，只有放在时间里才看得见。
  · 第一步的「对齐」也是动作：几行尾数从左对齐滑到按指数对齐的位置，窗口外的位才露出来、才被剪掉。

⛔ 数据从哪来（全部在本文件里现算，numpy）
  · 张量核模型：照实测的 B200 经验模型（topic09_numbers.TC_GROUP ＝ 16、TC_KEEP ＝ 25，不是厂商规格）：
    每 16 个 BF16 乘积连同累加器一起对齐到最大指数 emax（emax ＝ ⌊log2|x|⌋；乘积的指数取两个输入指数之和，不按乘积规格化），
    低于 2^(emax−25) 的位向零截掉，精确相加，写回 FP32 也向零截断。
  · 对照：FP32 逐项顺序累加，每加一项就近舍入一次（RNE）。画面里只叫「就近舍入（对照）」，不说它是哪家硬件。
  · 输入：合成正数，A、B 都是 [0, 1) 均匀随机数舍到 BF16（跟实测 TC_BIAS_K 用的合成输入同一种分布），固定种子。
  · 第一步那几行位：取自第二步那条求和里真实的一组（第 SHOW_G 组），16 个乘积里按筛子挑 4 个
    （都有位落在窗口外、就近舍入下既有「多」也有「少」），按原顺序取前几个满足的。
  · 第二步那两条线：同一条点积（K ＝ 65,536）的运行中相对误差；这一条从 256 条里按筛子挑：
    就近舍入那条穿过 0 的次数不少于中位数、终点离 0 不比中位数远，取第一条满足的。
  · 与实测核对（断言钉住）：256 条平均的「朝零相对偏差」在 K ＝ 65,536 时落在 TC_BIAS_K（3.1e-4）的 0.5–2 倍之内。
    K ＝ 256 / 4,096 时累加器模型本身只有 3e-7 / 1.1e-5，比 TC_BIAS_K（2.2e-5 / 7.0e-5）小得多 ——
    实测脚本里 BF16 矩阵乘的输出是 BF16，量到的是「累加器截断 ＋ 输出舍到 BF16」；本文件把模型结果再 RNE 舍到 BF16 后，
    256 / 4,096 两处都回到 TC_BIAS_K 的 0.5–2 倍之内（断言）。所以短求和那两个实测点主要是输出舍入，不是累加器截断（推断，判据就是这组断言）。

⛔ 刻意没画
  · 输出再舍到 BF16 那一步、cuBLAS 的分块与 split-K、负数输入（同样朝零）、TF32。
  · 画面不写任何案例里的具体张量和长度；纵轴只到 4e-4，不标「实测」。
  · 不说 TPU 是就近舍入 —— 那是推断，画面里只叫「就近舍入（对照）」，并且用中性白、不用蓝（全课蓝＝TPU）。
    抹零红（偏差来源），GPU 的位格子灰。终点数字旁小字「模拟，口径与实测曲线不同」：这里是单条点积的运行中相对误差，
    实测曲线是许多条点积平均的朝零偏差，短求和处还含输出舍到 BF16（见上「与实测核对」）。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic09-anim-trunc.py TruncVsRound WebPages/media/topic09-trunc.mp4
   每一步落地的停顿时刻写进 steps/TruncVsRound.json（课件挂 data-pauses）。
"""
import json
import os
import sys

import numpy as np

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, VMobject, Rectangle, Line, DashedLine, Text, ValueTracker, always_redraw,
                   FadeIn, FadeOut, Create, Write, Brace,
                   WHITE, GREY, GREY_B, BLUE, RED, GREEN, YELLOW,
                   UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic09_numbers as NUM  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")
G, KEEP = NUM.TC_GROUP, NUM.TC_KEEP
assert (G, KEEP) == (16, 25)


# ════════════════════════════════════════════════════════════════
# ① 模型：numpy 现算
# ════════════════════════════════════════════════════════════════
def bf16(x):
    """float32 → BF16（RNE），返回 float64。"""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def flog2(x):
    """⌊log2|x|⌋；0 给一个很小的数。"""
    _, e = np.frexp(x)
    return np.where(x == 0, -100000, e - 1).astype(np.int64)


def rz_fp32(s):
    """FP64 → FP32 向零截断。"""
    f = s.astype(np.float32)
    over = np.abs(f.astype(np.float64)) > np.abs(s)
    f[over] = np.nextafter(f[over], np.float32(0))
    return f.astype(np.float64)


def tc_model(A, B, keep_hist=False):
    """张量核模型，逐组累加。A、B：[M, K]（BF16 值，float64 存）。返回终值或每组之后的累加器 [K/G, M]。"""
    M, K = A.shape
    ea, eb = flog2(A), flog2(B)
    acc = np.zeros(M)
    hist = []
    for g0 in range(0, K, G):
        p = A[:, g0:g0 + G] * B[:, g0:g0 + G]                       # BF16 × BF16：16 位有效数字，FP64 里精确
        ep = ea[:, g0:g0 + G] + eb[:, g0:g0 + G]                     # 乘积指数 ＝ 输入指数之和
        emax = np.maximum(ep.max(1), flog2(acc))
        unit = np.ldexp(1.0, emax - KEEP)[:, None]
        terms = np.concatenate([acc[:, None], p], axis=1)
        acc = rz_fp32(np.sum(np.trunc(terms / unit) * unit, axis=1))   # 截掉、精确相加、向零写回 FP32
        if keep_hist:
            hist.append(acc)
    return np.array(hist) if keep_hist else acc


def fp32_rne(A, B):
    """FP32 逐项顺序累加、每项 RNE。返回每组之后的累加器 [K/G, M]。"""
    P = (A * B).astype(np.float32)                                    # 精确
    acc = np.zeros(A.shape[0], np.float32)
    hist = []
    for k in range(A.shape[1]):
        acc = acc + P[:, k]
        if (k + 1) % G == 0:
            hist.append(acc.astype(np.float64))
    return np.array(hist)


def toward_zero(y, ref):
    """实测脚本同一个指标：Σ e·sign(ref) / Σ|ref|，负 ＝ 朝零。"""
    return float(((y - ref) * np.sign(ref)).sum() / np.abs(ref).sum())


RNG = np.random.default_rng(20261105)


def synth(M, K):
    return (bf16(RNG.random((M, K), dtype=np.float32)), bf16(RNG.random((M, K), dtype=np.float32)))


KMAX = NUM.TC_BIAS_K[-1][0]
assert KMAX == 65536
M_PATHS = 256
A, B = synth(M_PATHS, KMAX)
H_TC = tc_model(A, B, keep_hist=True)                 # [4096, 256]
H_RN = fp32_rne(A, B)
REF = np.cumsum(A * B, axis=1)[:, G - 1::G].T          # 精确的运行和（FP64，误差 ~1e-16 量级）
E_TC = (H_TC - REF) / REF
E_RN = (H_RN - REF) / REF
NG = KMAX // G

# —— 论点：抹零只少不多 —— 每一条、每一组，欠的绝对量只增不减
assert np.all(np.diff(REF - H_TC, axis=0) >= 0) and np.all(REF - H_TC >= 0)
# —— 与实测核对：K ＝ 65,536 累加器模型本身就到实测量级
SIM_BIAS = {KMAX: toward_zero(H_TC[-1], REF[-1])}
SIM_RNE_65536 = toward_zero(H_RN[-1], REF[-1])
MEAS = dict(NUM.TC_BIAS_K)
assert 0.5 < -SIM_BIAS[KMAX] / MEAS[KMAX] < 2.0, SIM_BIAS
assert abs(SIM_RNE_65536) < -SIM_BIAS[KMAX] / 50, SIM_RNE_65536      # 就近舍入小两个数量级
# 短求和：累加器模型很小；加上「输出 RNE 舍到 BF16」才回到实测量级
SIM_BIAS_BF16OUT = {}
for _K in (256, 4096):                                 # 各 65,536 条点积（输出舍入噪声要压到 1e-5 以下），分块算
    _ys, _refs = [], []
    for _c in range(65536 * _K // (1 << 24)):
        _A, _B = synth((1 << 24) // _K, _K)
        _refs.append((_A * _B).sum(1))
        _ys.append(tc_model(_A, _B))
    _ref, _y = np.concatenate(_refs), np.concatenate(_ys)
    assert _ref.size == 65536
    SIM_BIAS[_K] = toward_zero(_y, _ref)
    SIM_BIAS_BF16OUT[_K] = toward_zero(bf16(_y.astype(np.float32)), _ref)
    assert -SIM_BIAS[_K] < MEAS[_K] / 4, (_K, SIM_BIAS[_K])
    assert 0.5 < -SIM_BIAS_BF16OUT[_K] / MEAS[_K] < 2.0, (_K, SIM_BIAS_BF16OUT[_K])
del _A, _B
# 文件头里写的两个数（3e-7 / 1.1e-5）也钉住，免得注释过期
assert 2e-7 < -SIM_BIAS[256] < 4.5e-7 and 0.8e-5 < -SIM_BIAS[4096] < 1.5e-5, SIM_BIAS

# —— 第二步那一条：筛子挑
_cross = (np.diff(np.sign(E_RN), axis=0) != 0).sum(0)
_okp = np.where((_cross >= np.median(_cross)) & (np.abs(E_RN[-1]) <= np.median(np.abs(E_RN[-1]))))[0]
PATH = int(_okp[0])
YMAX = 4e-4                                           # 主图纵轴到 −4e-4
ZOOM = 80                                              # 放大窗：纵轴 ±YMAX/ZOOM
assert np.all(E_TC[:, PATH] <= 0) and -E_TC[-1, PATH] < 0.95 * YMAX
assert np.abs(E_RN[:, PATH]).max() < 0.9 * YMAX / ZOOM
assert _cross[PATH] >= 20

# —— 第一步那一组：取这一条里真实的一组，挑 4 个乘积
SHOW_G = 1024                                          # 第 1025 组（已加了 16,384 个乘积）
_g0 = SHOW_G * G
ACC_V = H_TC[SHOW_G - 1, PATH]
E_ACC = int(flog2(np.array([ACC_V]))[0])
_a, _b = A[PATH, _g0:_g0 + G], B[PATH, _g0:_g0 + G]
_ep = flog2(_a) + flog2(_b)
assert _ep.max() < E_ACC, "这一组里累加器必须是指数最大的那个"
UNIT_W = E_ACC - KEEP                                  # 窗口最低位的权重 2^UNIT_W
ACC_INT = int(round(ACC_V / 2.0 ** (E_ACC - 23)))      # 24 位有效数字
assert 2 ** 23 <= ACC_INT < 2 ** 24


def prod_row(i):
    """第 i 个乘积：16 位整数尾数 P、最低位权重 w0；返回（位串从高到低, 最高位权重, 截掉部分 / 窗口单位）。"""
    P = int(round(_a[i] / 2.0 ** (flog2(_a[i:i + 1])[0] - 7))) * int(round(_b[i] / 2.0 ** (flog2(_b[i:i + 1])[0] - 7)))
    w0 = int(_ep[i]) - 14
    assert P * 2.0 ** w0 == _a[i] * _b[i]
    top = w0 + P.bit_length() - 1
    bits = [(P >> (top - w0 - j)) & 1 for j in range(top - w0 + 1)]
    cut_n = UNIT_W - w0                                # 有几位落在窗口外
    frac = (P & ((1 << cut_n) - 1)) / float(1 << cut_n) if cut_n > 0 else 0.0
    return bits, top, cut_n, frac


ROWS = []
for _i in range(G):
    _bits, _top, _cn, _fr = prod_row(_i)
    if _cn >= 2 and _fr not in (0.0, 0.5):
        ROWS.append((_i, _bits, _top, _cn, _fr))
_up = [r for r in ROWS if r[4] > 0.5]
_dn = [r for r in ROWS if r[4] < 0.5]
assert len(_up) >= 2 and len(_dn) >= 2
SHOW = sorted(_up[:2] + _dn[:2], key=lambda r: r[0])
COLS = max(E_ACC - r[2] + len(r[1]) for r in SHOW)     # 画到第几列
assert all(E_ACC - r[2] + len(r[1]) - 1 > KEEP for r in SHOW)


# ════════════════════════════════════════════════════════════════
# ② 画面
# ════════════════════════════════════════════════════════════════
def tx(s, size=28, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def dump_steps(name, marks, t):
    os.makedirs(STEPS_DIR, exist_ok=True)
    with open(os.path.join(STEPS_DIR, name + ".json"), "w") as fh:
        json.dump({"pauses": marks, "duration": round(t, 2)}, fh)


TITLE = "GPU 张量核：每 16 个一组，抹零"
CAPS = {
    "align": "一组：16 个乘积连同累加器，按最大的指数对齐",
    "s1": "第 1 步完成：窗口外的位，抹零次次少；就近舍入有多有少",
    "run": "很多组接着加（模拟，合成正数输入）",
    "s2": "第 2 步完成：抹零一路往下；就近舍入在 0 附近晃",
}
MORAL = "抹零只少不多：求和越长，偏得越多"
SIM_NOTE = "模拟，口径与实测曲线不同"
COL_RN = WHITE                                         # 就近舍入（对照）：中性白，不用蓝（蓝＝TPU）
CELL = 0.28
X0 = -5.45                                             # 第 0 列（累加器最高位）的左边
ROW_Y0, ROW_DY = 1.25, 0.72


def col_x(c):
    return X0 + c * CELL + CELL / 2


def bit_cells(bits, c0, y, col=GREY_B):            # GPU 的位：灰（全课 TPU 蓝、GPU 灰）
    g = VGroup()
    for j, b in enumerate(bits):
        r = Rectangle(width=CELL * 0.82, height=0.42, stroke_color=col, stroke_width=2.5,
                      fill_color=col, fill_opacity=0.85 if b else 0.0).move_to([col_x(c0 + j), y, 0])
        g.add(r)
    return g


def build_rows():
    """第 0 帧：每行尾数左对齐（还没按指数对齐）。返回 (labels, rows[list of VGroup], 目标起始列)。"""
    labels, rows, targets = VGroup(), [], []
    acc_bits = [(ACC_INT >> (23 - j)) & 1 for j in range(24)]
    labels.add(tx("累加器", 28, GREY_B).move_to([X0 - 0.85, ROW_Y0, 0]))
    rows.append(bit_cells(acc_bits, 0, ROW_Y0))
    targets.append(0)
    for k, (_i, bits, top, _cn, _fr) in enumerate(SHOW):
        y = ROW_Y0 - (k + 1) * ROW_DY
        labels.add(tx("乘积", 28, GREY_B).move_to([X0 - 0.85, y, 0]))
        rows.append(bit_cells(bits, 0, y))
        targets.append(E_ACC - top)
    dots = tx("……（一组 16 个）", 24, GREY).move_to([X0 + 1.6, ROW_Y0 - 5 * ROW_DY + 0.1, 0])
    return labels, rows, targets, dots


# 主图坐标
PX0, PX1, PY0, PY1 = -5.5, 1.5, 1.3, -2.5              # x：K 0 → 65,536；y：0 → −YMAX
TAB_X = 2.55


def px(k):
    return PX0 + (PX1 - PX0) * k / KMAX


def py(e):
    return PY0 + (PY1 - PY0) * (-e) / YMAX


IX0, IX1, IYC, IH = 2.7, 6.5, -1.75, 0.75             # 放大窗：右栏、就近舍入计数器下面；中线 IYC，半高 IH


def ipy(e):
    return IYC + IH * e / (YMAX / ZOOM)


def path_pts(series, kcur, fx, fy, nmax=400):
    n = max(2, int(round(kcur / KMAX * NG)))
    idx = np.unique(np.linspace(0, n - 1, min(n, nmax)).astype(int))
    pts = [[fx(0), fy(0.0), 0]] + [[fx((i + 1) * G), fy(series[i]), 0] for i in idx]
    return pts


def fmt_e(v):
    s = "%.1e" % v
    m, e = s.split("e")
    return "%s e%d" % (m.replace("-", "−"), int(e)) if v != 0 else "0"


class TruncVsRound(Scene):
    def construct(self):
        marks = []

        def hold(t=2.0):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        title = tx(TITLE, 38, WHITE, True).to_edge(UP, buff=0.35)
        cap = tx(" ", 28, GREY_B).next_to(title, DOWN, buff=0.22)
        labels, rows, targets, dots = build_rows()
        self.add(title, cap, labels, *rows)

        def say(s, color=GREY_B, extra=()):
            nonlocal cap
            n = tx(s, 28, color).next_to(title, DOWN, buff=0.22)
            self.play(FadeOut(cap), FadeIn(n), *extra, run_time=0.35)
            cap = n

        shown = VGroup()
        # ── 第 1 步：对齐 → 窗口 → 剪掉 → 两种舍法
        self.wait(0.6)
        say(CAPS["align"])
        self.play(*[r.animate.shift(RIGHT * CELL * t) for r, t in zip(rows, targets)], FadeIn(dots), run_time=1.4)
        shown.add(dots)
        xcut = X0 + (KEEP + 1) * CELL
        ytop, ybot = ROW_Y0 + 0.4, ROW_Y0 - 4 * ROW_DY - 0.3
        cut = DashedLine([xcut, ytop, 0], [xcut, ybot, 0], color=RED, stroke_width=4, dash_length=0.12)
        win = Line([X0 + CELL, ytop, 0], [xcut, ytop, 0])
        brace = Brace(win, UP, buff=0.05, color=GREY_B)
        blab = tx("最高位后 25 位", 24, GREY_B).next_to(brace, UP, buff=0.06)
        red_cells = VGroup()
        for k, (_i, bits, top, cn, _fr) in enumerate(SHOW):
            r = rows[k + 1]
            for cell in r[len(bits) - cn:]:
                red_cells.add(cell)
        self.play(Create(cut), FadeIn(brace), FadeIn(blab), run_time=0.7)
        self.play(*[c.animate.set_stroke(RED).set_fill(RED) for c in red_cells], run_time=0.5)
        shown.add(cut, brace, blab)
        # 剪掉：红格往下掉、淡出（向零）
        self.play(*[c.animate.shift(DOWN * 0.35).set_opacity(0) for c in red_cells], run_time=0.8)
        self.remove(red_cells)
        # 两列：抹零 / 就近舍入（对照）
        cx1, cx2 = col_x(COLS) + 0.75, col_x(COLS) + 2.05
        h1 = tx("抹零", 26, RED, True).move_to([cx1, ROW_Y0, 0])
        h2 = tx("就近舍入", 26, COL_RN, True).move_to([cx2, ROW_Y0 + 0.13, 0])
        h2b = tx("（对照）", 20, COL_RN).next_to(h2, DOWN, buff=0.04)
        tags1, tags2 = VGroup(), VGroup()
        for k, (_i, _bits, _top, _cn, fr) in enumerate(SHOW):
            y = ROW_Y0 - (k + 1) * ROW_DY
            tags1.add(tx("少", 28, RED).move_to([cx1, y, 0]))
            tags2.add(tx("多" if fr > 0.5 else "少", 28, GREEN if fr > 0.5 else RED).move_to([cx2, y, 0]))
        self.play(FadeIn(h1), FadeIn(tags1), run_time=0.5)
        self.play(FadeIn(h2), FadeIn(h2b), FadeIn(tags2), run_time=0.5)
        shown.add(h1, h2, h2b, tags1, tags2)
        say(CAPS["s1"])
        hold(2.0)

        # ── 第 2 步：很多组
        self.play(FadeOut(shown), FadeOut(labels), *[FadeOut(r) for r in rows], run_time=0.6)
        self.remove(shown, labels, *rows)
        shown = VGroup()
        say(CAPS["run"])
        ax_x = Line([PX0, PY0, 0], [PX1 + 0.15, PY0, 0], color=GREY, stroke_width=3)
        ax_y = Line([PX0, PY0 + 0.15, 0], [PX0, PY1 - 0.1, 0], color=GREY, stroke_width=3)
        yl = VGroup(tx("0", 24, GREY_B).next_to([PX0, py(0), 0], LEFT, buff=0.12),
                    tx("−2e-4", 24, GREY_B).next_to([PX0, py(-2e-4), 0], LEFT, buff=0.12),
                    tx("−4e-4", 24, GREY_B).next_to([PX0, py(-4e-4), 0], LEFT, buff=0.12))
        ticks = VGroup(*[Line([PX0 - 0.06, py(-v), 0], [PX0 + 0.06, py(-v), 0], color=GREY, stroke_width=3)
                         for v in (2e-4, 4e-4)])
        xl = VGroup(tx("加了多少个乘积 →", 24, GREY_B).next_to([(PX0 + PX1) / 2, PY0, 0], UP, buff=0.1),
                    tx("65,536", 24, GREY_B).next_to([px(KMAX), PY0, 0], UP, buff=0.1))
        ylab = tx("相对误差", 24, GREY_B).next_to(yl[2], DOWN, buff=0.12)
        box = Rectangle(width=IX1 - IX0 + 0.3, height=2 * IH + 0.3, stroke_color=COL_RN, stroke_width=2.5)
        box.move_to([(IX0 + IX1) / 2, IYC, 0])
        zero_i = DashedLine([IX0, IYC, 0], [IX1, IYC, 0], color=GREY, stroke_width=2, dash_length=0.08)
        zlab = tx("放大 80 倍", 22, COL_RN).next_to(box, DOWN, buff=0.08)
        zlab0 = tx("0", 20, GREY_B).next_to([IX0, IYC, 0], RIGHT, buff=0.0).shift(LEFT * 0.42)
        frame = VGroup(ax_x, ax_y, ticks, yl, xl, ylab, box, zero_i, zlab, zlab0)
        self.play(FadeIn(frame), run_time=0.6)
        shown.add(frame)

        kt = ValueTracker(0.0)

        def k_now():
            return kt.get_value()

        def mk_curve(series, fx, fy, col, w=5):
            def f():
                return VMobject(stroke_color=col, stroke_width=w).set_points_as_corners(
                    path_pts(series, max(k_now(), 1e-9), fx, fy))
            return always_redraw(f)

        c_tc = mk_curve(E_TC[:, PATH], px, py, RED)
        c_rn = mk_curve(E_RN[:, PATH], px, py, COL_RN, 4)
        c_rz = mk_curve(E_RN[:, PATH], lambda k: IX0 + (IX1 - IX0) * k / KMAX, ipy, COL_RN, 3)

        def val_at(series):
            i = int(round(k_now() / KMAX * NG)) - 1
            return 0.0 if i < 0 else float(series[i])

        def counter(series, name, col, y):
            def f():
                return VGroup(tx(name, 26, col, True), tx(fmt_e(val_at(series)), 30, col)).arrange(
                    DOWN, aligned_edge=LEFT, buff=0.1).move_to([TAB_X, y, 0], aligned_edge=LEFT)
            return always_redraw(f)

        n_tc = counter(E_TC[:, PATH], "抹零", RED, 1.0)
        n_rn = counter(E_RN[:, PATH], "就近舍入（对照）", COL_RN, -0.2)

        def kcount():
            k = int(round(k_now() / 256.0)) * 256
            return tx("加了 {:,} 个".format(k), 30, YELLOW).move_to([PX0 + 0.4, PY1 + 0.45, 0], aligned_edge=LEFT)
        n_k = always_redraw(kcount)
        self.add(c_tc, c_rn, c_rz, n_tc, n_rn, n_k)
        self.play(kt.animate.set_value(KMAX), run_time=7.0, rate_func=linear)
        shown.add(c_tc, c_rn, c_rz, n_tc, n_rn, n_k)
        simnote = tx(SIM_NOTE, 20, GREY_B).move_to([TAB_X, 1.78, 0], aligned_edge=LEFT)
        shown.add(simnote)
        say(CAPS["s2"], extra=(FadeIn(simnote),))           # 跟字幕同一拍淡入，不改停顿时刻
        hold(2.0)

        # ── 第 3 步：一句话
        moral = tx(MORAL, 40, YELLOW, True).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(moral, shift=UP * 0.2), run_time=0.7)
        shown.add(moral)
        hold(2.5)

        # ── 收尾复位：清干净，摆回第 0 帧
        self.play(FadeOut(shown), FadeOut(cap), run_time=0.6)
        self.remove(shown, cap, c_tc, c_rn, c_rz, n_tc, n_rn, n_k)
        labels0, rows0, _t, _d = build_rows()
        cap0 = tx(" ", 28, GREY_B).next_to(title, DOWN, buff=0.22)
        self.play(FadeIn(labels0), *[FadeIn(r) for r in rows0], run_time=0.6)
        self.add(cap0)
        self.wait(0.6)
        dump_steps("TruncVsRound", marks, self.renderer.time)


if __name__ == "__main__":
    print("PATH", PATH, "crossings", int(_cross[PATH]))
    print("SIM_BIAS (accumulator model, FP32 out):", {k: "%.2e" % v for k, v in sorted(SIM_BIAS.items())})
    print("SIM_BIAS_BF16OUT:", {k: "%.2e" % v for k, v in SIM_BIAS_BF16OUT.items()})
    print("SIM_RNE_65536 (FP32 seq RNE, 256 paths):", "%.2e" % SIM_RNE_65536)
    print("path end: tc %.2e  rne %.2e  rne max|.| %.2e" % (E_TC[-1, PATH], E_RN[-1, PATH], np.abs(E_RN[:, PATH]).max()))
    print("SHOW rows:", [(r[0], r[3], round(r[4], 3)) for r in SHOW], "COLS", COLS, "E_ACC", E_ACC)
