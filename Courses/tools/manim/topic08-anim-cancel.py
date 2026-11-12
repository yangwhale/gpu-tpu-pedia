# -*- coding: utf-8 -*-
r"""专题八 · §9.7：修好一个零件，读数反而离零更远（CancelScale）。

渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08-anim-cancel.py CancelScale /tmp/draft/topic08-cancel.mp4 --draft
   构建手册/脚本/render.sh tools/manim/topic08-anim-cancel.py CancelScale WebPages/media/topic08-cancel.mp4
步点写进 tools/manim/steps/CancelScale.json。挂在 §9.7 静态图 fig8-cancel 下面。

⭐ 为什么非动不可：这一拍的戏剧性在「拨一个开关，指针往意料之外的方向跑」：修好收拢处，指针离零更远；
   只修流梯度，指针几乎回零；两个都修，又停在中间。静态图（fig8-cancel）把四个读数并排，看不出「修了反而变坏」
   这一下的方向感；增量来自时间。
⭐ 数据从哪来：四个读数（9.4 / 12.5 / 1.9 / 6.9，×1e-6）与 GPU 上拿掉收拢处梯度舍入的 −2.9，全部取自
   topic08_numbers 的 CX、EL_GPU_HEADGRAD（研究线单因素交叉结果文件，实测）。
   ⚠️ 推断：「两个零件方向相反、读数是两者相抵」是研究线的推断（GPU 那一侧方向一致）；屏幕上标「推断」。
⭐ 刻意没画：每格的份数与散布（检验值在 fig-gatecmp、fig-twoway）；两个零件各自的大小（没有单独测过）。
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
        title = tx("修好一个零件，读数反而离零更远", 30).to_edge(UP)
        unit = tx("读数：第 551–1000 步主 loss 偏差，×1e-6", 17, GREY_B).move_to([0, 2.65, 0])
        r_tpu = ruler(Y_TPU, "TPU 对 GPU", ORANGE)
        r_gpu = ruler(Y_GPU, "GPU 自己（对 GPU 其余各份）", BLUE)
        sw_y = 1.75
        lab_a = tx("① 流梯度", 20, ORANGE).move_to([-3.6, sw_y, 0])
        lab_b = tx("② 收拢处梯度", 20, BLUE).move_to([2.2, sw_y, 0])
        base = VGroup(title, unit, r_tpu, r_gpu, lab_a, lab_b)
        self.add(base)

        st = {"a": False, "b": False}
        val = ValueTracker(R["none"])
        gv = ValueTracker(0.0)

        def state(key, x):
            ok = st[key]
            return tx("已补" if ok else "没补", 20, GREEN if ok else RED).move_to([x, sw_y, 0])
        sa = always_redraw(lambda: state("a", -1.9))
        sb = always_redraw(lambda: state("b", 4.2))

        def pointer(tr, y, col):
            def draw():
                v = tr.get_value()
                t = Triangle(color=col, fill_color=col, fill_opacity=1).scale(0.16).rotate(PI).move_to([xv(v), y + 0.32, 0])
                return VGroup(t, tx("%+.1f" % v, 22, col).move_to([xv(v), y + 0.68, 0]))
            return draw
        pt = always_redraw(pointer(val, Y_TPU, ORANGE))
        self.add(sa, sb, pt)
        self.wait(0.6)

        def cap(s, col=GREY_B):
            m = tx(s, 21, col).move_to([0, -3.3, 0])
            assert m.width < 13.6, (s, m.width)
            return m

        def mark():
            marks.append(round(self.renderer.time + 0.2, 2))

        steps = [
            ("第 1 步　两处都没补（修之前）：读数 %.1f" % R["none"], None, None, R["none"]),
            ("第 2 步　只补 ② 收拢处：指针反而往右跑，离零更远", "b", True, R["head"]),
            ("第 3 步　改成只补 ① 流梯度：几乎回到零，看着像修好了", "ab", (True, False), R["stream"]),
            ("第 4 步　两处都补（对齐配置）：停在中间，剩下这截推断在这一段以外", "b", True, R["both"]),
        ]
        cur = None
        for k, (s, key, newv, target) in enumerate(steps):
            c = cap(s, YELLOW if k == 1 else GREY_B)
            if key == "b":
                st["b"] = newv
            elif key == "ab":
                st["a"], st["b"] = newv
            self.play(*([FadeOut(cur)] if cur else []), FadeIn(c), run_time=0.5)
            self.play(val.animate.set_value(target), run_time=1.4)
            cur = c
            mark()
            self.wait(2.2)

        # 第 5 步：GPU 上反过来验
        c5 = cap("第 5 步　GPU 上拿掉 ② 那次梯度舍入：GPU 自己被往下压 %.1f，跟第 2 步多出的 %s 对得上（推断）"
                 % (-GPU_HC, N.CX_HEAD_UP_T), BLUE)
        pg = always_redraw(pointer(gv, Y_GPU, BLUE))
        self.play(FadeOut(cur), FadeIn(c5), FadeIn(pg), run_time=0.5)
        self.play(gv.animate.set_value(GPU_HC), run_time=1.4)
        mark()
        self.wait(3.0)

        # 收尾：回到第 0 帧
        st["a"], st["b"] = False, False
        self.play(FadeOut(c5), FadeOut(pg), val.animate.set_value(R["none"]), run_time=0.8)
        self.wait(0.5)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "CancelScale.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
