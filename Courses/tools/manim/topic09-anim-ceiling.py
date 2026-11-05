# -*- coding: utf-8 -*-
r"""专题九「GPU→TPU 训练精度对齐」：噪声上限 N 怎么来、被测怎么判（NoiseCeiling）。
挂在静态图 fig9-judge 下面。画面和本文件里都不写课件节号以外的内部信息。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间）
  N 不是一个现成的数，是一个过程的产物：先有 5 份重跑 → 两两配对、一对一对算出 10 根柱 → 从里面挑最高的那根 → 拿它去量被测。
  静态图（fig9-judge）只给终点（四个点落在哪个颜色里），画不出「N 是从一堆配对里挑出来的最大值」，
  也画不出「被测一出场，纵轴得缩一半才装得下」—— 3.45 倍的分量是在这次缩放里看出来的。

⛔ 数据从哪来
  · 10 对的真值：一次正式验收的结果文件（param_norm 带符号均值，第 2–1000 步全程窗口，分母取基线那一份）。
    结果文件只直接给出 N（10 对里的最大值，未舍入 1.237070e-7）和每一对的「留一比」——
    这一对的值 ÷ 其余 9 对里的最大值（PN_PAIR_LOO）。由此每一对的真值可以精确还原（本课推导）：
      · 除最大那对以外，其余 9 对的「其余 9 对最大值」都是 N，所以 |sᵢ| ＝ 留一比 × N；
      · 最大那对的留一比 ＝ N ÷ 第二大，所以第二大 ＝ N ÷ 1.6000 ＝ 7.73e-8，
        跟「1–4」那对按上一条算出的 0.6250 × N ＝ 7.73e-8 对上（断言钉住，这是还原方法的自检）。
  · 5 个点在横轴上的位置：同样由这 10 个真值反推（本课推导）。带符号均值近似可加（sᵢⱼ ≈ xᵢ − xⱼ），
    取第 1 份为原点、按 1–2…1–5 四对定位置，其余 6 对的距离与真值的残差平方和 < 1e-26（断言钉住）——
    所以「两点隔多远，柱就多高」在这组数上是成立的。整条轴左右翻转同样成立，朝向是任取的。
  · 被测 4.27e-7、N 1.24e-7、r 3.45：topic09_numbers.ACC 第一行（课件已有的公开数）；脚本另用结果文件的未舍入值
    4.268544e-7 ÷ 1.237070e-7 ＝ 3.4505 交叉核对。
  · 1/(n+1)：topic09_numbers.p_exceed（本课推导：对称性）。3 对 1/4、10 对 1/11、28 对 1/29。

⛔ 刻意没画
  · loss 和 grad_norm 两项、MARE 和最大单步两个统计：只演参数范数的带符号均值一项，其余同一个套路。
  · 这一次的被测比的是「精确轨迹」那份基线，而 N 用的是 ③ 那组（生产近似＋输出头 dgrad 换精确）GPU 重跑 5 份
    —— 这是一个有意的例外（课件 §9.3 讲），画面只写「TPU 对精确轨迹」加一行小字「尺子借自 ③ 那组 GPU 重跑（§9.3）」，不展开。
    （R6 评审第 1 条：原先写「GPU 生产程序重跑」，会被读成 ② 生产近似那组；尺子实际来自 ③。）
  · 画面上 5 个点的编号按左右顺序排（运行编号 1,2,5,4,3 → 画面 1–5）；编号本身没有含义，配对柱的标签跟着换。
  · 参照对共用运行、不是独立样本，实际超线概率比 1/(n+1) 略高：课件正文写了，画面不展开。
  · 横轴没有刻度：5 个点只表达相对位置。
  · 配色：GPU 重跑的点和配对柱灰、TPU 被测柱蓝、N 黄（全课 TPU＝蓝、GPU＝灰）。5 个点第 0 帧就在，第①步只是逐个闪一下。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic09-anim-ceiling.py NoiseCeiling WebPages/media/topic09-ceiling.mp4
   每一步落地的停顿时刻写进 steps/NoiseCeiling.json（课件挂 data-pauses）。
"""
import itertools
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Line, Dot, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, ORANGE,
                   UP, DOWN, LEFT, RIGHT, linear, interpolate_color)
from manim.utils.rate_functions import smooth

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic09_numbers as T  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── ① 数据：10 对的真值，从「N ＋ 留一比」精确还原 ─────────────────────────────
PN_N_RAW = 1.237070013957617e-07          # 结果文件：10 对里的最大值（未舍入）
PN_TEST_RAW = 4.2685443046109505e-07      # 结果文件：被测的带符号均值（未舍入）
PN_PAIR_LOO = {                           # 结果文件：这一对 ÷ 其余 9 对的最大值（第 i 份与第 j 份，1 起）
    (1, 2): 0.3841797290392483, (1, 3): 1.6000008380491806, (1, 4): 0.6249996726372103,
    (1, 5): 0.5677430273207673, (2, 3): 0.6158203036088835, (2, 4): 0.24081994066970203,
    (2, 5): 0.18356329854019784, (3, 4): 0.3750004202818857, (3, 5): 0.4322570697297758,
    (4, 5): 0.05725665681358128,
}
PAIRS = sorted(PN_PAIR_LOO)
assert len(PAIRS) == 10 == 5 * 4 // 2
assert sum(v > 1 for v in PN_PAIR_LOO.values()) == 1          # 只有最大那一对留一比 > 1
PAIR_VAL = {p: (PN_N_RAW if v > 1 else v * PN_N_RAW) for p, v in PN_PAIR_LOO.items()}
P_MAX = max(PAIR_VAL, key=PAIR_VAL.get)
_second = sorted(PAIR_VAL.values())[-2]
assert abs(PN_N_RAW / PN_PAIR_LOO[P_MAX] - _second) / _second < 1e-5, "还原自检：最大那对的留一比要等于 N ÷ 第二大"
# 舍入后对上课件的公开数
assert round(max(PAIR_VAL.values()), 9) == T.GPU_RERUN_PN == 1.24e-7
_name, X_TEST, N_PUB, R_PUB = T.ACC[0]
assert _name.startswith("param_norm") and (X_TEST, N_PUB, R_PUB) == (4.27e-7, 1.24e-7, 3.45)
assert round(PN_TEST_RAW, 9) == X_TEST and round(PN_TEST_RAW / PN_N_RAW, 2) == R_PUB

# 5 个点的位置：第 1 份为原点，1–k 四对定位，符号组合扫一遍，取残差最小的
_best = None
for sg in itertools.product((1, -1), repeat=4):
    xs = [0.0] + [sg[k - 2] * PAIR_VAL[(1, k)] for k in range(2, 6)]
    err = sum((abs(xs[i - 1] - xs[j - 1]) - PAIR_VAL[(i, j)]) ** 2 for i, j in PAIRS)
    if _best is None or err < _best[0]:
        _best = (err, xs)
POS_ERR, POS = _best
assert POS_ERR < 1e-26, POS_ERR                                 # 10 个距离全部对上：两点隔多远，柱就多高
assert min(POS) == 0.0                                          # 取的朝向里第 1 份在最左

# 画面编号按左右顺序重排（运行顺序 1,2,5,4,3 从左到右会让人困惑；编号本身没有含义）
_order = sorted(range(1, 6), key=lambda i: POS[i - 1])           # 从左到右的运行编号
assert _order == [1, 2, 5, 4, 3], _order
RUN2D = {run: d for d, run in enumerate(_order, start=1)}         # 运行编号 → 画面编号
POS = [POS[_order[d - 1] - 1] for d in range(1, 6)]              # 改成按画面编号索引
PAIR_VAL = {tuple(sorted((RUN2D[i], RUN2D[j]))): v for (i, j), v in PAIR_VAL.items()}
PAIRS = sorted(PAIR_VAL)
P_MAX = max(PAIR_VAL, key=PAIR_VAL.get)
assert P_MAX == (1, 5) and POS == sorted(POS)
assert all(abs(abs(POS[i - 1] - POS[j - 1]) - PAIR_VAL[(i, j)]) < 1e-13 for i, j in PAIRS)

# 1/(n+1)
PROBS = [(3, "1/4"), (10, "1/11"), (28, "1/29")]
assert [round(1 / T.p_exceed(n)) for n, _ in PROBS] == [4, 11, 29]

# ── ② 版面 ─────────────────────────────────────────────────────────────
AX_Y, AX_L, AX_R = 1.35, -6.3, 1.3              # 上：5 个点的横轴
DOT_X0 = -5.85
DOT_K = 6.75 / max(POS)                         # 场景单位 / 带符号均值
BASE_Y = -3.3                                   # 下：柱子的底线
BAR_X0, BAR_DX, BAR_W = -6.0, 0.66, 0.40
TEST_X = 1.0
BAND_L, BAND_R = -6.45, 1.65
S1 = 2.2 / PN_N_RAW                             # 前三步：N ＝ 2.2 单位高
S2 = 1.05 / PN_N_RAW                            # 第④步缩到 N ＝ 1.05，被测 3.45N 才装得下
TOP_Y = 0.72                                    # 红区上沿
assert BASE_Y + PN_TEST_RAW * S2 < TOP_Y < AX_Y - 0.4
COL_N, COL_TEST, COL_GPU = YELLOW, BLUE, GREY_B          # 全课：TPU 蓝、GPU 灰
ZONES = [(0, 1, GREEN), (1, 2, ORANGE), (2, None, RED)]

# ── ③ 时间轴（秒）：每一步落地后停 2 秒 ──────────────────────────────────────
PAUSE = 2.0
T1A = 0.6
T1B = T1A + 5 * 0.35 + 0.3                     # 5 个点逐个出现
T2A = T1B + PAUSE
PAIR_GAP, PAIR_RUN = 0.6, 0.85
T2B = T2A + PAIR_GAP * 9 + PAIR_RUN + 0.2
T3A = T2B + PAUSE
T3B = T3A + 1.5
T4A = T3B + PAUSE
T4B = T4A + 2.7
T5A = T4B + PAUSE
T5B = T5A + 1.7
T6A = T5B + PAUSE                              # 收尾：累积的东西淡出、第一句字幕淡回
T6B = T6A + 0.8
LANDS = [T1B, T2B, T3B, T4B, T5B]


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def prog(t, a, b, ease=True):
    p = clamp01((t - a) / (b - a))
    return smooth(p) if ease else p


def fade_out_all(t):
    """收尾淡出系数：累积的东西全部乘它。"""
    return 1.0 - prog(t, T6A, T6B, ease=False)


def sci(x):
    """课件口径的科学计数：1.24e-7（不补零）。"""
    m, e = ("%.2e" % x).split("e")
    return "%se%d" % (m, int(e))


assert sci(1.24e-7) == "1.24e-7" and sci(4.27e-7) == "4.27e-7"


def tx(s, size=24, color=WHITE):
    return Text(s, font_size=size, color=color)


def dot_x(i):
    return DOT_X0 + POS[i - 1] * DOT_K


def bar_x(k):
    return BAR_X0 + k * BAR_DX


def scale_at(t):
    p = prog(t, T4A, T4A + 1.0)
    return S1 + (S2 - S1) * p


def pair_window(k):
    a = T2A + k * PAIR_GAP
    return a, a + PAIR_RUN


CAPS = [
    (0.0, "① 第一关的尺子：GPU 重跑 5 份（一个点 ＝ 一份）",
     "③ 那组设置：生产近似＋输出头 dgrad 换精确"),
    (T2A, "② 两两配对，一共 10 对：每对算参数范数的带符号均值，取绝对值，竖成一根柱",
     "带符号均值 ＝ 第 2–1000 步逐步相对差 dₜ 的平均；两个点隔多远，柱就多高"),
    (T3A, "③ 取最大的那一根当噪声上限 N",
     "为什么取最大不取平均：被测若只是噪声，应落在已有样本的范围里"),
    (T4A, "④ 被测（TPU 对精确轨迹）出场：r ＝ 被测 ÷ N",
     "柱子比尺子高出一大截，纵轴得缩一半才装得下"),
    (T5A, "⑤ 样本越多，尺子越可信：纯噪声超过 n 个样本最大值的概率是 1/(n+1)", ""),
]


def cap_alpha(i, t):
    """第 i 句字幕在 t 时刻的不透明度：0.3 秒交叉淡入淡出；收尾段第一句淡回。"""
    lo = CAPS[i][0]
    hi = CAPS[i + 1][0] if i + 1 < len(CAPS) else T6A
    a = 1.0 if i == 0 else prog(t, lo - 0.15, lo + 0.15, ease=False)
    b = 1.0 - prog(t, hi - 0.15, hi + 0.15, ease=False)
    v = min(a, b)
    if i == 0:
        v = max(v, prog(t, T6A, T6B, ease=False))
    return clamp01(v)


class NoiseCeiling(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def clock():
            return tr.get_value()            # ⭐ 全片唯一读时钟的地方

        title = tx("噪声上限 N：拿 GPU 自己重跑当尺子", 34).to_edge(UP, buff=0.3)
        self.add(title)

        # ── 静态件（第 0 帧就在）：横轴、柱子底线
        axis = Line([AX_L, AX_Y, 0], [AX_R, AX_Y, 0], color=GREY, stroke_width=3)
        base = Line([BAND_L, BASE_Y, 0], [BAND_R, BASE_Y, 0], color=GREY, stroke_width=3)
        self.add(axis, base)

        # ── 背景三个区（第④步）
        def zones():
            t = clock()
            a = prog(t, T4A + 0.8, T4A + 1.4) * fade_out_all(t)
            s = scale_at(t)
            g = VGroup()
            for lo, hi, col in ZONES:
                y0 = BASE_Y + lo * PN_N_RAW * s
                y1 = TOP_Y if hi is None else BASE_Y + hi * PN_N_RAW * s
                g.add(Rectangle(width=BAND_R - BAND_L, height=max(y1 - y0, 1e-3), stroke_width=0,
                                fill_color=col, fill_opacity=0.22 * a)
                      .move_to([(BAND_L + BAND_R) / 2, (y0 + y1) / 2, 0]))
            return g
        self.add(always_redraw(zones))

        # ── 5 个点
        def dots():
            t = clock()
            g = VGroup()
            for i in range(1, 6):
                a0 = T1A + (i - 1) * 0.35                     # 第①步：逐个闪一下（点本身第 0 帧就在）
                pulse = prog(t, a0, a0 + 0.2) - prog(t, a0 + 0.2, a0 + 0.45)
                g.add(Dot([dot_x(i), AX_Y, 0], radius=0.11 * (1 + 0.45 * pulse),
                          color=interpolate_color(COL_GPU, WHITE, pulse), stroke_width=0))
            return g

        # ── 10 根柱：先在横轴上亮出两点之间那一段，再变形落成柱
        def bars():
            t = clock()
            s = scale_at(t)
            f = fade_out_all(t)
            g = VGroup()
            for k, p in enumerate(PAIRS):
                a, b = pair_window(k)
                if t < a:
                    continue
                i, j = p
                xa, xb = sorted((dot_x(i), dot_x(j)))
                h = PAIR_VAL[p] * s
                is_max = p == P_MAX
                if t < b:                            # 变形中
                    u = prog(t, a + 0.15, b)
                    seg_a = prog(t, a, a + 0.2)
                    g.add(Line([xa, AX_Y, 0], [xb, AX_Y, 0], color=YELLOW, stroke_width=7,
                               stroke_opacity=seg_a * (1 - u)))
                    w = (xb - xa) + (BAR_W - (xb - xa)) * u
                    hh = 0.07 + (h - 0.07) * u
                    cx = (xa + xb) / 2 + (bar_x(k) - (xa + xb) / 2) * u
                    cy = AX_Y + (BASE_Y + h / 2 - AX_Y) * u
                    col = interpolate_color(YELLOW, COL_GPU, u)
                    g.add(Rectangle(width=max(w, 1e-3), height=max(hh, 1e-3), stroke_width=0,
                                    fill_color=col, fill_opacity=0.9 * seg_a).move_to([cx, cy, 0]))
                else:
                    col = COL_GPU
                    if is_max:
                        col = interpolate_color(COL_GPU, COL_N, prog(t, T3A, T3A + 0.5))
                    g.add(Rectangle(width=BAR_W, height=max(h, 1e-3), stroke_width=0,
                                    fill_color=col, fill_opacity=0.9 * f)
                          .move_to([bar_x(k), BASE_Y + h / 2, 0]))
            return g

        # ── N 线（第③步拉出）
        def nline():
            t = clock()
            u = prog(t, T3A + 0.3, T3A + 1.1)
            f = fade_out_all(t)
            y = BASE_Y + PN_N_RAW * scale_at(t)
            x0 = bar_x(PAIRS.index(P_MAX)) - BAR_W / 2
            xl = x0 + (BAND_L - x0) * u
            xr = x0 + (BAND_R - x0) * u
            return Line([xl, y, 0], [max(xr, xl + 1e-3), y, 0], color=COL_N, stroke_width=4,
                        stroke_opacity=(1.0 if u > 0 else 0.0) * f)

        # ── 被测柱（第④步）
        def testbar():
            t = clock()
            u = prog(t, T4A + 1.4, T4A + 2.4)
            h = PN_TEST_RAW * scale_at(t) * u
            return Rectangle(width=BAR_W + 0.1, height=max(h, 1e-3), stroke_width=0, fill_color=COL_TEST,
                             fill_opacity=(0.92 if u > 0 else 0.0) * fade_out_all(t)
                             ).move_to([TEST_X, BASE_Y + h / 2, 0])

        # ── 第⑤步：三行概率条
        P_ROW_Y = [1.95, 1.4, 0.85]
        P_LBL_X, P_BAR_X0, P_K = 3.1, 3.65, 9.6

        def pbars():
            t = clock()
            g = VGroup()
            for r, (n, _) in enumerate(PROBS):
                u = prog(t, T5A + r * 0.4, T5A + r * 0.4 + 0.7)
                L = T.p_exceed(n) * P_K * u
                col = YELLOW if n == 10 else GREY_B
                g.add(Rectangle(width=max(L, 1e-3), height=0.3, stroke_width=0, fill_color=col,
                                fill_opacity=(0.85 if u > 0 else 0.0) * fade_out_all(t))
                      .move_to([P_BAR_X0 + L / 2, P_ROW_Y[r], 0]))
            return g

        for fn in (bars, testbar, nline, dots, pbars):
            self.add(always_redraw(fn))

        # ── 文字：建一次，updater 只改不透明度 / 位置
        texts = []                                   # (mobject, alpha_fn, pos_fn or None)

        def put(m, alpha_fn, pos_fn=None):
            texts.append((m, alpha_fn, pos_fn))

        for i, (_, s1, s2) in enumerate(CAPS):
            put(tx(s1, 24, WHITE).move_to([0, 2.78, 0]), lambda t, i=i: cap_alpha(i, t))
            if s2:
                put(tx(s2, 21, GREY_B).move_to([0, 2.32, 0]), lambda t, i=i: cap_alpha(i, t))

        for i in range(1, 6):
            a0 = T1A + (i - 1) * 0.35
            put(tx(str(i), 22, COL_GPU).move_to([dot_x(i), AX_Y + 0.32, 0]),
                lambda t: 1.0)

        for k, (i, j) in enumerate(PAIRS):
            _, b = pair_window(k)
            put(tx("%d–%d" % (i, j), 18, GREY_B).move_to([bar_x(k), BASE_Y - 0.25, 0]),
                lambda t, b=b: prog(t, b - 0.2, b + 0.2) * fade_out_all(t))

        kmax = PAIRS.index(P_MAX)
        lbl_n1 = tx("N ＝ 最大的那一对 ＝ " + sci(N_PUB), 22, COL_N)
        lbl_n2 = tx("N ＝ " + sci(N_PUB), 22, COL_N)
        for m in (lbl_n1, lbl_n2):
            m.set_x(-2.2)

        def n_pos(m):
            def f(t):
                return [m.get_x(), BASE_Y + PN_N_RAW * scale_at(t) + 0.24, 0]
            return f
        put(lbl_n1, lambda t: prog(t, T3A + 0.9, T3A + 1.3) * (1 - prog(t, T4A, T4A + 0.4)), n_pos(lbl_n1))
        put(lbl_n2, lambda t: prog(t, T4A, T4A + 0.4) * fade_out_all(t), n_pos(lbl_n2))

        lbl_x = tx("被测", 18, GREY_B).move_to([TEST_X, BASE_Y - 0.25, 0])
        put(lbl_x, lambda t: prog(t, T4A + 1.4, T4A + 1.8) * fade_out_all(t))
        lbl_xv = tx(sci(X_TEST), 20, COL_TEST).move_to([TEST_X, BASE_Y + PN_TEST_RAW * S2 + 0.25, 0])
        put(lbl_xv, lambda t: prog(t, T4A + 2.2, T4A + 2.6) * fade_out_all(t))

        zl = [("过（r ≤ 1）", GREEN, 0.5), ("待定（1–2）", ORANGE, 1.5), ("不过（r > 2）", RED, 2.35)]
        for s, col, at in zl:
            m = tx(s, 20, col)
            m.move_to([BAND_R + 0.25 + m.width / 2, BASE_Y + at * PN_N_RAW * S2, 0])
            put(m, lambda t: prog(t, T4A + 0.8, T4A + 1.4) * fade_out_all(t))
        rf = tx("r ＝ %s ÷ %s ＝ %.2f" % (sci(X_TEST), sci(N_PUB), R_PUB), 22, WHITE)
        rf.move_to([BAND_R + 0.25 + rf.width / 2, 0.15, 0])
        put(rf, lambda t: prog(t, T4A + 2.2, T4A + 2.6) * fade_out_all(t))
        rn = tx("尺子借自 ③ 那组 GPU 重跑（§9.3）", 18, GREY_B)
        rn.move_to([BAND_R + 0.25 + rn.width / 2, -0.3, 0])
        put(rn, lambda t: prog(t, T4A + 0.2, T4A + 0.6) * fade_out_all(t))

        for r, (n, frac) in enumerate(PROBS):
            a0 = T5A + r * 0.4
            put(tx("%d 对" % n, 20, YELLOW if n == 10 else GREY_B).move_to([P_LBL_X, P_ROW_Y[r], 0]),
                lambda t, a0=a0: prog(t, a0, a0 + 0.4) * fade_out_all(t))
            fm = tx(frac + ("（这一次）" if n == 10 else ""), 20, YELLOW if n == 10 else GREY_B)
            fm.move_to([P_BAR_X0 + T.p_exceed(n) * P_K + 0.15 + fm.width / 2, P_ROW_Y[r], 0])
            put(fm, lambda t, a0=a0: prog(t, a0 + 0.4, a0 + 0.8) * fade_out_all(t))

        def apply(m, af, pf):
            t = clock()
            m.set_opacity(af(t))
            if pf is not None:
                m.move_to(pf(t))

        for m, af, pf in texts:
            apply(m, af, pf)                         # ⛔ 先手动算一次：第 0 帧不能叠字
            m.add_updater(lambda m, af=af, pf=pf: apply(m, af, pf))
            self.add(m)

        # ── 走时钟：全部 linear；每一步落地记一个停顿点
        marks = []
        now = 0.0
        for land in LANDS:
            self.play(tr.animate.set_value(land), run_time=land - now, rate_func=linear)
            marks.append(round(self.renderer.time + 0.3, 2))
            self.play(tr.animate.set_value(land + PAUSE), run_time=PAUSE, rate_func=linear)
            now = land + PAUSE
        self.play(tr.animate.set_value(T6B), run_time=T6B - now, rate_func=linear)
        # 复位：T6B 时刻的画面就是第 0 帧（累积件全淡出、第一句字幕淡回）；把时钟拨回 0
        self.play(tr.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)
        self.play(tr.animate.set_value(0.0), run_time=0.4, rate_func=linear)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "NoiseCeiling.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
