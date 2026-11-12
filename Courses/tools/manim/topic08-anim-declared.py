# -*- coding: utf-8 -*-
r"""专题八 · §9.7：代码说「这里舍一次」，编译器听了吗？（DeclaredElided）

渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08-anim-declared.py DeclaredElided /tmp/draft/topic08-declared.mp4 --draft
   构建手册/脚本/render.sh tools/manim/topic08-anim-declared.py DeclaredElided WebPages/media/topic08-declared.mp4
每一步落地的停顿时刻写进 tools/manim/steps/DeclaredElided.json（课件挂 data-pauses）。挂在 §9.7 静态图 fig8-elided 下面。

⭐ 为什么非动不可：论点是「同一个值走过同一串盒子，三种走法舍不舍不一样」——代码的本意（经过「转 BF16」就舍）、
   实际在一颗融合里（经过了也不舍）、写回内存交给下一颗（真舍）。静态图能摆出三张截面，摆不出「同一个值一路走、
   数字在哪一刻变、在哪一刻没变」；增量来自时间。
⭐ 数据从哪来：盒子的顺序是多路残差分支输入那一处（4 路加权和 → 转 BF16 → 转回 FP32 → 算均方 / 归一化输出），
   打包指令 0 条、补上强制舍入 256 条是最小复现的实测（topic08_numbers.EL_IHC_PACKS）。
   示意：值 0.72314453125 是随手取的 FP32 数；它的 BF16 由 topic08_numbers.bf16 当场算。
⭐ 刻意没画：kernel 里其余的运算（平方、求和、乘缩放）；4 路加权和本身怎么算。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, RoundedRectangle, DashedVMobject, Line, Arrow, Text, FadeIn, FadeOut, Transform,
                   Create, Indicate, WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, ORANGE, UP, DOWN, LEFT, RIGHT)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08_numbers as N   # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")
X = 0.72314453125
XB = N.bf16(X)
assert XB != X and abs(XB - X) / X < 2 ** -8
SX, SB = "%.8f" % X, "%.8f" % XB
P0, P1 = N.EL_IHC_PACKS
assert (P0, P1) == (0, 256)

Y_ROW = 0.9
XS = [-4.9, -1.9, 1.1, 4.3]
NAMES = ["4 路加权和\nFP32", "转 BF16", "转回 FP32", "算均方"]
COLS = [BLUE, ORANGE, BLUE, WHITE]


def tx(s, size=22, color=WHITE):
    return Text(s, font_size=size, color=color)


def box(i, label=None, col=None):
    r = RoundedRectangle(corner_radius=0.12, width=2.1, height=0.95, stroke_color=col or COLS[i], stroke_width=4)
    t = tx(label or NAMES[i], 20, col or COLS[i])
    return VGroup(r, t).move_to([XS[i], Y_ROW, 0])


def token(s, col=WHITE):
    r = RoundedRectangle(corner_radius=0.1, width=1.9, height=0.5, fill_color="#222222", fill_opacity=1,
                         stroke_color=col, stroke_width=3)
    return VGroup(r, tx(s, 19, col))


class DeclaredElided(Scene):
    def construct(self):
        marks = []
        title = tx("代码说「这里舍一次」，编译器听了吗？", 30).to_edge(UP)
        boxes = VGroup(*[box(i) for i in range(4)])
        arrows = VGroup(*[Arrow([XS[i] + 1.05, Y_ROW, 0], [XS[i + 1] - 1.05, Y_ROW, 0], buff=0.05, color=GREY,
                                stroke_width=3, max_tip_length_to_length_ratio=0.25) for i in range(3)])
        base = VGroup(title, boxes, arrows)
        self.add(base)
        self.wait(0.8)

        def cap(s, col=GREY_B):
            m = tx(s, 21, col).move_to([0, 2.45, 0])
            assert m.width < 13.6, (s, m.width)
            return m

        def mark():
            marks.append(round(self.renderer.time + 0.2, 2))

        y_tok = Y_ROW - 1.05

        # ── 第 1 步：代码的本意 ──
        c1 = cap("第 1 步　代码的本意：经过「转 BF16」就舍一次，后面用的是舍过的数")
        tok = token(SX).move_to([XS[0], y_tok, 0])
        self.play(FadeIn(c1), FadeIn(tok))
        self.play(tok.animate.move_to([XS[1], y_tok, 0]), run_time=1.0)
        snapped = token(SB, YELLOW).move_to([XS[1], y_tok, 0])
        self.play(Transform(tok, snapped), Indicate(boxes[1], color=YELLOW), run_time=0.8)
        self.play(tok.animate.move_to([XS[3], y_tok, 0]), run_time=1.4)
        mark()
        self.wait(2.2)

        # ── 第 2 步：实际，同一颗融合里 ──
        c2 = cap("第 2 步　实际：这一串在同一颗融合里，编译器把往返省掉，均方用的是没舍的数")
        fus = DashedVMobject(RoundedRectangle(corner_radius=0.2, width=12.2, height=2.9, stroke_color=GREY_B,
                                              stroke_width=2.5), num_dashes=70).move_to([-0.3, Y_ROW - 0.45, 0])
        flab = tx("同一颗融合", 18, GREY_B).move_to([4.6, Y_ROW - 1.6, 0])
        tok2 = token(SX).move_to([XS[0], y_tok, 0])
        self.play(FadeOut(c1), FadeOut(tok), FadeIn(c2), Create(fus), FadeIn(flab), FadeIn(tok2))
        self.play(tok2.animate.move_to([XS[1], y_tok, 0]), run_time=1.0)
        cross = VGroup(Line([XS[1] - 0.9, Y_ROW - 0.4, 0], [XS[1] + 0.9, Y_ROW + 0.4, 0], color=RED, stroke_width=6),
                       Line([XS[1] - 0.9, Y_ROW + 0.4, 0], [XS[1] + 0.9, Y_ROW - 0.4, 0], color=RED, stroke_width=6))
        self.play(Create(cross), run_time=0.6)
        self.play(tok2.animate.move_to([XS[3], y_tok, 0]), run_time=1.4)
        cnt = tx("后端指令里打包成 BF16 的：%d 条" % P0, 24, RED).move_to([0, -2.2, 0])
        self.play(FadeIn(cnt))
        mark()
        self.wait(2.4)

        # ── 第 3 步：写回内存交给下一颗，真舍 ──
        c3 = cap("第 3 步　同一个值还要写回内存、交给下一颗算归一化输出：这一路真舍了")
        mem = VGroup(RoundedRectangle(corner_radius=0.12, width=2.3, height=0.8, stroke_color=YELLOW, stroke_width=4),
                     tx("内存（BF16）", 19, YELLOW)).move_to([XS[2], -2.35, 0])
        nxt = VGroup(RoundedRectangle(corner_radius=0.12, width=3.1, height=0.8, stroke_color=WHITE, stroke_width=4),
                     tx("下一颗：归一化输出", 18)).move_to([XS[3] + 0.25, -2.35, 0])
        tok3 = token(SX).move_to([XS[2], y_tok, 0])
        self.play(FadeOut(c2), FadeOut(cnt), FadeIn(c3), FadeIn(mem), FadeIn(nxt), FadeIn(tok3))
        self.play(tok3.animate.move_to([XS[2], -1.55, 0]), run_time=0.9)
        self.play(Transform(tok3, token(SB, YELLOW).move_to([XS[2], -1.55, 0])), Indicate(mem, color=YELLOW), run_time=0.8)
        self.play(tok3.animate.move_to([XS[3] + 0.25, -1.55, 0]), run_time=1.0)
        diff = tx("两颗 kernel 用的不是同一个数", 22, YELLOW).move_to([-3.6, -2.35, 0])
        self.play(FadeIn(diff))
        mark()
        self.wait(2.6)

        # ── 第 4 步：强制舍入 ──
        c4 = cap("第 4 步　写成强制舍入（reduce_precision）：编译器省不掉，融合里也真舍")
        rp = box(1, "强制舍", GREEN)
        tok4 = token(SX).move_to([XS[0], y_tok, 0])
        self.play(FadeOut(c3), FadeOut(mem), FadeOut(nxt), FadeOut(tok3), FadeOut(diff), FadeOut(tok2), FadeOut(cross),
                  FadeIn(c4), Transform(boxes[1], rp), FadeIn(tok4))
        self.play(tok4.animate.move_to([XS[1], y_tok, 0]), run_time=1.0)
        self.play(Transform(tok4, token(SB, GREEN).move_to([XS[1], y_tok, 0])), Indicate(boxes[1], color=GREEN), run_time=0.8)
        self.play(tok4.animate.move_to([XS[3], y_tok, 0]), run_time=1.4)
        cnt4 = tx("打包成 BF16 的：%d 条" % P1, 24, GREEN).move_to([0, -2.2, 0])
        self.play(FadeIn(cnt4))
        mark()
        self.wait(3.0)

        # ── 收尾：回到第 0 帧 ──
        self.play(FadeOut(VGroup(c4, tok4, cnt4, fus, flab)), Transform(boxes[1], box(1)), run_time=0.8)
        self.wait(0.5)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "DeclaredElided.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
