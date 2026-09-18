# -*- coding: utf-8 -*-
r"""专题四 · 一路乘下去 ——&#160;`fig-vanish` 配的那段动画（候选排名第 3）

⭐⭐⭐ 第一版被我自己否了，值得记下来为什么：
  只画了**对数轴上那把扇子** ——&#160;而指数函数取完对数就是直线，
  于是画面是三条直线张开。⛔ 那跟它上面那张静态图**几乎一模一样**，
  动起来只多给了「谁先谁后」，不值一段视频。
  ⭐ 判据：**一段动画要活下来，得给出静态图给不了的东西。**
    「同样的内容播一遍」不算 ——&#160;那只是把读者的眼睛替他移了一次。

⭐⭐⭐ 这一版找到的增量，是**对数轴自己的代价**：
  **对数轴让你看得见十个数量级，可也正好让你感觉不到十个数量级有多大。**
  所以画两条轨，共用同一根横轴（层号）、同一根扫描线：
    上轨 ＝ 对数刻度：三条直线**才刚刚张开**
    下轨 ＝ 线性刻度：同一时刻，**蓝的早已顶出画框，红的已经贴在地上**
  ⭐ 落点：看上轨你以为「它们还差不多」，看下轨才知道**早就没法比了**。
    这个反差**只有把两条轨对齐播才成立** ——&#160;静态图并排画两幅，
    读者不会自己去对同一层。

⭐⭐ 数值落点跟静态图同一个：每层 ×0.8 和 ×1.2 **只差两成**，
  六十层后差十个数量级。指数放大的不是梯度，是「每层偏离 1 多少」那个偏差。

⛔ 四条房规（见 skill `manim-teaching-figures`）：
  ① 一个字都不放；② 静态图排在它上面；③ 首帧 ≡ 末帧；④ 数据当场算。

⭐ 复位用 `clock()` 倒着走 ——&#160;内容是累积的，直接清零只会把那一跳
  从接缝挪进片内。跟 `topic04-anim-memtime.py` 同一个手法。

📌 渲染：
    ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic04-anim-vanish.py Vanish WebPages/media/topic04-vanish.mp4
"""
import math

import numpy as np
from manim import (Scene, VGroup, Dot, Line, ValueTracker, always_redraw,
                   WHITE, linear)

RD_, BL_, GY_, GY2_, INK_ = "#d93025", "#4285f4", "#c3c7cb", "#80868b", "#202124"

L = 60                                   # 层数，跟静态图同一个口径
CASES = ((0.8, RD_), (1.0, GY2_), (1.2, BL_))

# ⭐ 全部当场乘出来：第 k 个点 ＝ 往回传了 k 层之后的相对幅度
RAW = {r: [r ** k for k in range(L)] for r, _ in CASES}
DEC = {r: [k * math.log10(r) for k in range(L)] for r, _ in CASES}

LO, HI = min(min(v) for v in DEC.values()), max(max(v) for v in DEC.values())
assert HI - LO > 9.0, "两头拉不开十个数量级，这一格就没有落点了（现在 %.1f）" % (HI - LO)

XL, XR = -6.2, 6.2                       # 右＝输出层（刚开始传），左＝输入层

# ── 上轨：对数刻度 ────────────────────────────────────────────────
A_MID, A_HALF = 1.95, 1.80
SYA = (A_HALF * 2) / (HI - LO)
A_Y0 = A_MID - (LO + HI) / 2 * SYA

# ── 下轨：线性刻度。VMAX 决定「多大算出框」──────────────────────
B_BOT, B_TOP = -3.55, -0.20
# ⛔ VMAX 是**构图参数也是论点参数**：它同时决定「多大算出框」和
#   「1.0 那条参考线坐多高」。第一版取 6.0，1.0 被压到离地面只有六分之一，
#   红线「从 1 掉到 0」那一段几乎贴着地面走，看不出它掉过。
#   ⭐ 收到 4.0：1.0 抬到四分之一高，红线的坠落看得见；
#     代价是蓝线出框提前到第 8 层 —— 反而更狠。
VMAX = 4.0
SYB = (B_TOP - B_BOT) / VMAX
GONE = 0.02                              # 小于这个就算「肉眼没了」（≈ 1080p 下 3 px）

# ⭐⭐⭐ 这段动画的论点必须被算出来、并且钉死，不能靠我说
K_OUT = next(k for k in range(L) if RAW[1.2][k] > VMAX)      # 蓝的第几层出框
K_GONE = next(k for k in range(L) if RAW[0.8][k] < GONE)     # 红的第几层看不见
K_MARK = max(K_OUT, K_GONE)
SPREAD = (DEC[1.2][K_MARK] - DEC[0.8][K_MARK]) * SYA / (A_HALF * 2)
assert K_MARK < L // 3, \
    "线性轴要在前三分之一就分出胜负，否则反差不够（现在第 %d 层）" % K_MARK
assert SPREAD < 0.35, \
    ("到第 %d 层，对数轴上要还挤在一起（张开 %.0f%%），"
     "『上轨看着差不多、下轨早没法比』才成立" % (K_MARK, SPREAD * 100))

T_SWEEP, T_HOLD, T_REW = 5.4, 1.1, 0.9
T_RESET = T_SWEEP + T_HOLD
T_END = T_RESET + T_REW


def px(k):
    return XR - (XR - XL) * k / (L - 1.0)


def pya(d):
    return A_Y0 + d * SYA


def pyb(v):
    return B_BOT + min(v, VMAX) * SYB


class Vanish(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        tt = ValueTracker(0.0)

        def clock():
            t = tt.get_value()
            if t <= T_SWEEP:
                return t
            if t <= T_RESET:
                return T_SWEEP
            return T_SWEEP * (1.0 - (t - T_RESET) / T_REW)

        def cur_k():
            return max(1, int(clock() / T_SWEEP * (L - 1)))

        # ── 底稿 ──────────────────────────────────────────────────
        base = VGroup()
        d = math.ceil(LO / 2.0) * 2                      # 上轨：每两个数量级一条
        while d <= HI:
            base.add(Line(np.array([XL - 0.25, pya(d), 0]),
                          np.array([XR + 0.25, pya(d), 0]),
                          stroke_color=GY_,
                          stroke_width=2.0 if abs(d) < 1e-9 else 1.0))
            d += 2
        # 下轨：一条地面、一条「原始幅度＝1」的参考线、一条**画框上沿**
        # ⭐ 上沿那条是这一轨的关键 ——&#160;蓝线越过它＝出框，
        #   没有这条线，「顶出去」就只是「画到头了」。
        for y, w in ((B_BOT, 2.0), (pyb(1.0), 1.4), (B_TOP, 2.0)):
            base.add(Line(np.array([XL - 0.25, y, 0]), np.array([XR + 0.25, y, 0]),
                          stroke_color=GY_, stroke_width=w))
        self.add(base)

        # ── 两条轨上的三条链，从右往左**被画出来** ─────────────────
        def track(r, col, fy, vals, clip):
            def mk():
                k, g = cur_k(), VGroup()
                for i in range(k):
                    y0, y1 = fy(vals[i]), fy(vals[i + 1])
                    # ⭐ 出框之后不再画 ——&#160;线停在上沿，读成「冲出去了」，
                    #   而不是「贴着顶边继续爬」（那会读成「到顶了就不动了」，
                    #   跟事实完全相反）。
                    if clip and vals[i] > VMAX:
                        break
                    g.add(Line(np.array([px(i), y0, 0]), np.array([px(i + 1), y1, 0]),
                               stroke_color=col, stroke_width=3.6))
                if not (clip and vals[k] > VMAX):
                    g.add(Dot(np.array([px(k), fy(vals[k]), 0]),
                              radius=0.095, color=col))
                return g
            return always_redraw(mk)

        for r, col in CASES:
            self.add(track(r, col, pya, DEC[r], False))
            self.add(track(r, col, pyb, RAW[r], True))

        # ⭐ 一根贯穿两轨的扫描线 ——&#160;它是「同一层」这件事的唯一证据。
        #   两轨的反差全靠它成立，所以它不是装饰。
        self.add(always_redraw(lambda: Line(
            np.array([px(cur_k()), B_BOT - 0.28, 0]),
            np.array([px(cur_k()), pya(HI) + 0.28, 0]),
            stroke_color=INK_, stroke_width=1.8)))

        self.play(tt.animate.set_value(T_END), run_time=T_END * 1.9,
                  rate_func=linear)
