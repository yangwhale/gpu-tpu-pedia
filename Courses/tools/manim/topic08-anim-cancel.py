# -*- coding: utf-8 -*-
r"""专题八 · §9.7：一台秤三个毛病，修好一个，读数反而偏了（CancelScale）。

渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08-anim-cancel.py CancelScale /tmp/draft/topic08-cancel.mp4 --draft
   构建手册/脚本/render.sh tools/manim/topic08-anim-cancel.py CancelScale WebPages/media/topic08-cancel.mp4
步点写进 tools/manim/steps/CancelScale.json。挂在 §9.7 静态图 fig8-cancel 下面。

⭐ 为什么非动不可：这一拍的戏剧性在「拨一个开关，指针往意料之外的方向跑」：只修甲（流梯度），指针几乎回零；
   再修乙（收拢处），指针反而往右跑回 6.9。静态图（fig8-cancel）把读数并排，看不出「修了反而偏」这一下的方向感；增量来自时间。
   只修乙那一格（12.5，单看分不开）不进动画，静态图里有。
⭐ 数据从哪来：四个读数（9.4 / 12.5 / 1.9 / 6.9，×1e-6）与 GPU 上拿掉收拢处梯度舍入的 −2.9，全部取自
   topic08_numbers 的 CX、EL_GPU_HEADGRAD（研究线单因素交叉结果文件，实测）。
   ⚠️ 推断：「三个毛病、乙丙抵掉、丙露出来」是研究线的推断（GPU 那一侧方向一致，只占一部分）；屏幕上标「推断」。
⭐ 刻意没画：每格的份数与散布（检验值在 fig-gatecmp、fig-twoway）；甲乙丙在 TPU 上各自的大小（没有单独测出）。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Line, Triangle, RoundedRectangle, Text, ValueTracker, always_redraw, FadeIn, FadeOut,
                   WHITE, GREY, GREY_B, GREEN, RED, YELLOW, ORANGE, BLUE, UP, DOWN, PI)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08_numbers as N   # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")
R = {k: v * 1e6 for k, v in N.CX.items()}
GPU_HC = N.EL_GPU_HEADGRAD["late"] * 1e6
assert ["%.1f" % R[k] for k in ("none", "head", "stream", "both")] == ["9.4", "12.5", "1.9", "6.9"] and "%.1f" % GPU_HC == "-2.9"

XL, XR = -5.6, 5.6
VMIN, VMAX = -4.0, 14.0
Y_TPU, Y_GPU = 0.2, -1.9


def xv(v):
    return XL + (v - VMIN) / (VMAX - VMIN) * (XR - XL)


def tx(s, size=22, color=WHITE):
    return Text(s, font_size=size, color=color)


def ruler(y, label, col):
    g = VGroup(Line([XL, y, 0], [XR, y, 0], color=GREY, stroke_width=3))
    for v in range(-4, 15, 2):
        g.add(Line([xv(v), y - 0.1, 0], [xv(v), y + 0.1, 0], color=GREY, stroke_width=2))
        g.add(tx(str(v), 15, GREY_B).move_to([xv(v), y - 0.32, 0]))
    g.add(Line([xv(0), y - 0.25, 0], [xv(0), y + 0.25, 0], color=WHITE, stroke_width=4))
    g.add(tx(label, 18, col).move_to([0, y - 0.7, 0]).align_to([XR, 0, 0], direction=[1, 0, 0]))
    return g


class CancelScale(Scene):
    def construct(self):
        marks = []
        title = tx("一台秤三个毛病：修好一个，读数反而偏了", 30).to_edge(UP)
        unit = tx("读数：第 551–1000 步主 loss 偏差，×1e-6", 17, GREY_B).move_to([0, 2.65, 0])
        r_tpu = ruler(Y_TPU, "TPU 对 GPU", ORANGE)
        r_gpu = ruler(Y_GPU, "GPU 自己（对 GPU 其余各份）", BLUE)
        sw_y = 1.75
        lab_a = tx("甲 流梯度（偏重）", 19, ORANGE).move_to([-4.6, sw_y, 0])
        lab_b = tx("乙 收拢处梯度（偏轻）", 19, BLUE).move_to([0.0, sw_y, 0])
        lab_c = tx("丙 还没找到（偏重）", 19, ORANGE).move_to([4.4, sw_y, 0])
        base = VGroup(title, unit, r_tpu, r_gpu, lab_a, lab_b, lab_c)
        self.add(base)

        st = {"a": False, "b": False}
        val = ValueTracker(R["none"])
        gv = ValueTracker(0.0)

        def state(key, x):
            ok = st[key]
            return tx("已修" if ok else "没修", 19, GREEN if ok else RED).move_to([x, sw_y - 0.42, 0])
        sa = always_redraw(lambda: state("a", -4.6))
        sb = always_redraw(lambda: state("b", 0.0))
        sc = tx("？", 22, ORANGE).move_to([4.4, sw_y - 0.42, 0])

        def pointer(tr, y, col):
            def draw():
                v = tr.get_value()
                t = Triangle(color=col, fill_color=col, fill_opacity=1).scale(0.16).rotate(PI).move_to([xv(v), y + 0.32, 0])
                return VGroup(t, tx("%+.1f" % v, 22, col).move_to([xv(v), y + 0.68, 0]))
            return draw
        pt = always_redraw(pointer(val, Y_TPU, ORANGE))
        self.add(sa, sb, sc, pt)
        self.wait(0.6)

        def cap(s, col=GREY_B):
            m = tx(s, 21, col).move_to([0, -3.3, 0])
            assert m.width < 13.6, (s, m.width)
            return m

        def mark():
            marks.append(round(self.renderer.time + 0.2, 2))

        steps = [
            ("第 1 步　都没修（修之前）：读数 %.1f，是三股相抵后的数" % R["none"], None, R["none"], GREY_B),
            ("第 2 步　只修甲：几乎回到零，乙和丙大致抵掉，看着像修好了", (True, False), R["stream"], GREY_B),
            ("第 3 步　再修乙：指针反而往右跑，高了 %s（p %s）—— 被乙遮住的丙露出来（推断）" % (N.CX_UP_T, N.TW_P_T["c_m58"]),
             (True, True), R["both"], YELLOW),
        ]
        cur = None
        for s_, newv, target, col in steps:
            c = cap(s_, col)
            self.play(*([FadeOut(cur)] if cur else []), FadeIn(c), run_time=0.5)
            if newv:
                st["a"], st["b"] = newv
            self.play(val.animate.set_value(target), run_time=1.4)
            cur = c
            mark()
            self.wait(2.4)

        c4 = cap("第 4 步　GPU 上只拿掉乙那次舍入：GPU 自己被压低 %.1f，方向一致，是那 %s 的一部分（推断）"
                 % (-GPU_HC, N.CX_UP_T), BLUE)
        pg = always_redraw(pointer(gv, Y_GPU, BLUE))
        self.play(FadeOut(cur), FadeIn(c4), FadeIn(pg), run_time=0.5)
        self.play(gv.animate.set_value(GPU_HC), run_time=1.4)
        mark()
        self.wait(3.0)

        # 收尾：回到第 0 帧
        st["a"], st["b"] = False, False
        self.play(FadeOut(c4), FadeOut(pg), val.animate.set_value(R["none"]), run_time=0.8)
        self.wait(0.5)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "CancelScale.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
