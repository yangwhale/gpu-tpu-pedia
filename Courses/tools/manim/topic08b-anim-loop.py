# -*- coding: utf-8 -*-
r"""专题九 §9.6「编译时间：从 25 分钟到两分半」的动画 —— LoopUnroll。挂在静态图 fig-aot 下面。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间）
  · 要讲的是两件按时间发生的事：trace 时 Python 的 for 真的跑 N 遍、每遍往程序里记一份；编译时编译器把每一份都过一遍。
    静态图只能放两个终态（五块 vs 一块），看不出「是 trace 那一刻长出来的」和「编译时间跟着份数走」。

⛔ 数据从哪来（import 自 tools/topic08b_numbers.py，断言钉住）
  · 牛顿迭代 5 步：AOT 提速记录里的写法（Newton–Schulz 的 5 步迭代）。
  · N.AOT_STEPS 第 1、2 行：半层一格 361 → 277 秒（只改牛顿迭代那一刀）。只上字幕，不进几何。
⛔ 刻意没画 / 示意
  · 编译扫描的速度示意：每份循环体扫描时间相同，所以左边 5 份要 5 倍时间；真实编译时间对程序大小是超线性的，这里不画。
  · 不写开关名。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-loop.py LoopUnroll WebPages/media/topic08b-loop.mp4
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, RoundedRectangle, Line, Text, ValueTracker, always_redraw,
                   FadeIn, FadeOut, WHITE, GREY, GREY_B, BLUE, RED, GREEN, YELLOW, ORANGE,
                   UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

NSTEP = 5
S1 = [x for x in N.AOT_STEPS if x[0] == 1][0][2]
S2 = [x for x in N.AOT_STEPS if x[0] == 2][0][2]
assert (S1, S2) == (361, 277)

TITLE = "Python 循环 vs JAX 循环"
CODE_L = "for i in range(5):  x = step(x)"
CODE_R = "x = lax.fori_loop(0, 5, step, x)"
HEAD_L, HEAD_R = "Python 的 for", "JAX 的循环"
CAPS = [
    "trace：Python 的 for 真跑 5 遍，每遍记一份",
    "JAX 的循环：循环体只记一份，外面套一个循环",
    "编译：编译器把程序里的每一份都过一遍",
    "牛顿迭代改成 JAX 循环：半层一格 %d → %d 秒" % (S1, S2),
    "代价：跨迭代不能融合，舍入点会变，要上卡比",
]
for s in CAPS:
    assert sum(0.5 if ord(ch) < 128 else 1 for ch in s) <= 24, s
CNT_L = ["程序里：%d 份循环体" % k for k in range(NSTEP + 1)]
CNT_R = ["程序里：%d 份循环体" % k for k in (0, 1)]

XL, XR = -3.6, 3.6
BOX_TOP, BOX_H, BOX_W = 1.15, 3.3, 4.6
BLK_H = 0.5


def tx(s, size=24, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def block(x, i, col=BLUE):
    y = BOX_TOP - 0.38 - i * (BLK_H + 0.12)
    r = Rectangle(width=BOX_W - 0.8, height=BLK_H, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([x, y, 0])
    t = tx("step（几次矩阵乘）", 20, WHITE).move_to(r)
    return VGroup(r, t)


def dump_steps(name, marks, t):
    os.makedirs(STEPS_DIR, exist_ok=True)
    with open(os.path.join(STEPS_DIR, name + ".json"), "w") as fh:
        json.dump({"pauses": marks, "duration": round(t, 2)}, fh)


class LoopUnroll(Scene):
    def construct(self):
        marks = []
        title = tx(TITLE, 36, WHITE, True).to_edge(UP, buff=0.3)
        static = VGroup(
            tx(HEAD_L, 26, YELLOW, True).move_to([XL, 2.55, 0]),
            tx(HEAD_R, 26, YELLOW, True).move_to([XR, 2.55, 0]),
            tx(CODE_L, 40, GREY_B).scale(0.5).move_to([XL, 2.0, 0]),     # 按两倍排再缩一半：小字号下 Pango 会吞空格
            tx(CODE_R, 40, GREY_B).scale(0.5).move_to([XR, 2.0, 0]),
            Rectangle(width=BOX_W, height=BOX_H, stroke_color=GREY, stroke_width=3).move_to([XL, BOX_TOP - BOX_H / 2, 0]),
            Rectangle(width=BOX_W, height=BOX_H, stroke_color=GREY, stroke_width=3).move_to([XR, BOX_TOP - BOX_H / 2, 0]),
            tx("程序", 20, GREY).move_to([XL - BOX_W / 2 + 0.4, BOX_TOP + 0.2, 0]),
            tx("程序", 20, GREY).move_to([XR - BOX_W / 2 + 0.4, BOX_TOP + 0.2, 0]),
        )
        self.add(title, static)
        st = {"cap": tx(" ", 26, GREY_B).move_to([0, -2.85, 0]),
              "cl": tx(CNT_L[0], 22, BLUE).move_to([XL, -2.35, 0]),
              "cr": tx(CNT_R[0], 22, BLUE).move_to([XR, -2.35, 0])}
        self.add(st["cap"], st["cl"], st["cr"])

        def swap(key, new):
            old = st[key]
            st[key] = new
            return [FadeOut(old), FadeIn(new)]

        def say(s, color=GREY_B, extra=()):
            self.play(*swap("cap", tx(s, 26, color).move_to([0, -2.85, 0])), *extra, run_time=0.4)

        def hold(t):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        # ── 第一幕：trace
        self.wait(0.4)
        say(CAPS[0])
        left = []
        for i in range(NSTEP):
            b = block(XL, i)
            left.append(b)
            self.play(FadeIn(b, shift=DOWN * 0.2), *swap("cl", tx(CNT_L[i + 1], 22, BLUE).move_to([XL, -2.35, 0])), run_time=0.45)
        say(CAPS[1])
        rb = block(XR, 0)
        loop = tx("⟲ 跑 5 次", 22, YELLOW, True).next_to(rb, DOWN, buff=0.15)
        self.play(FadeIn(rb, shift=DOWN * 0.2), FadeIn(loop), *swap("cr", tx(CNT_R[1], 22, BLUE).move_to([XR, -2.35, 0])), run_time=0.6)
        hold(2.2)

        # ── 第二幕：编译扫描（每份同样快：左边要 5 倍时间）
        say(CAPS[2])
        tt = ValueTracker(0.0)
        y_top = BOX_TOP - 0.08
        y_l_end = BOX_TOP - 0.38 - (NSTEP - 1) * (BLK_H + 0.12) - BLK_H / 2 - 0.05
        y_r_end = BOX_TOP - 0.38 - BLK_H / 2 - 0.05

        def scan():
            t = tt.get_value()
            yl = y_top + (y_l_end - y_top) * t
            yr = y_top + (y_r_end - y_top) * min(1.0, t * NSTEP)
            g = [Line([XL - BOX_W / 2 + 0.1, yl, 0], [XL + BOX_W / 2 - 0.1, yl, 0], color=YELLOW, stroke_width=5)]
            if t * NSTEP < 1.0:
                g.append(Line([XR - BOX_W / 2 + 0.1, yr, 0], [XR + BOX_W / 2 - 0.1, yr, 0], color=YELLOW, stroke_width=5))
            return VGroup(*g)

        sweep = always_redraw(scan)
        self.add(sweep)
        done_r = tx("编完", 24, GREEN, True).move_to([XR + 1.6, BOX_TOP + 0.2, 0])
        self.play(tt.animate.set_value(1.0 / NSTEP), run_time=0.8, rate_func=linear)
        self.play(FadeIn(done_r), tt.animate.set_value(1.0), run_time=3.2, rate_func=linear)
        done_l = tx("编完（5 倍时间）", 24, RED, True).move_to([XL + 1.1, BOX_TOP + 0.2, 0])
        self.play(FadeIn(done_l), run_time=0.4)
        self.remove(sweep)
        hold(2.0)

        # ── 第三幕：实测数和代价
        say(CAPS[3], YELLOW)
        hold(2.4)
        say(CAPS[4])
        hold(2.4)

        # ── 收尾：回到第 0 帧
        self.play(*[FadeOut(b) for b in left], FadeOut(rb), FadeOut(loop), FadeOut(done_r), FadeOut(done_l),
                  *swap("cap", tx(" ", 26, GREY_B).move_to([0, -2.85, 0])),
                  *swap("cl", tx(CNT_L[0], 22, BLUE).move_to([XL, -2.35, 0])),
                  *swap("cr", tx(CNT_R[0], 22, BLUE).move_to([XR, -2.35, 0])), run_time=1.0)
        self.wait(0.5)
        dump_steps("LoopUnroll", marks, self.renderer.time)
