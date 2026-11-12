# -*- coding: utf-8 -*-
r"""专题八 · §9.8：先编译，再扫描（ScanSweep）。

渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08-anim-scan.py ScanSweep /tmp/draft/topic08-scan.mp4 --draft
   构建手册/脚本/render.sh tools/manim/topic08-anim-scan.py ScanSweep WebPages/media/topic08-scan.mp4
步点写进 tools/manim/steps/ScanSweep.json。挂在 §9.8 静态图 fig8-elided 下面（anim-declared 之后）。

⭐ 为什么非动不可：方法的要点是「一次扫过整个程序，同一个模式在哪儿出现就在哪儿亮」，再把亮起来的按位置归堆、
   计数一路涨到真实总数；最后拿 LLO 一颗颗抽查只含往返的 kernel。「扫过」「一路涨」「一颗颗盖章」都在时间里，
   静态图（fig8-elided 右块）只给得出终点。
⭐ 数据从哪来：各堆的终值（272 处、70 / 75 / 98 / 20 / 9、已补 55）、只含往返的 kernel 41 颗全 0 条、
   关省舍入负对照 35 颗无一为 0，全部取自 topic08_numbers 的 EL_* 块（研究线整步扫描结果文件，实测）。
   ⚠️ 示意：上面那条「程序」只画 60 格，哪几格亮、亮成什么颜色是随手摆的（真实程序 1666 颗 kernel、往返 272 处）；
     盖章的方块画 41 个和 35 个，数目是实测。
⭐ 刻意没画：按源码行归类的规则本身；混合 kernel 里的真舍（只在字幕里交代「另两类按规则推断」）。
"""
import json
import os
import random
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Square, Rectangle, RoundedRectangle, Text, ValueTracker, always_redraw,
                   FadeIn, FadeOut, WHITE, GREY, GREY_B, GREEN, RED, YELLOW, ORANGE, UP, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08_numbers as N   # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")
U = N.EL_USES
BUCKETS = [("① 分支输入梯度合流", U["branchin"], ORANGE), ("② 每微批 dW", U["wgrad"], ORANGE),
           ("③ 前向几处", N.EL_FWD, ORANGE), ("其余（优先级低）", N.EL_MINOR, GREY_B), ("索引器掩码（无害）", U["idx_mask"], GREY)]
assert sum(b[1] for b in BUCKETS) == N.EL_TOTAL == 272
K_ON, K_OFF = N.EL_K_TOTAL, N.EL_K_NOEXC
assert (K_ON, N.EL_K_ZERO, K_OFF, N.EL_K_NOEXC_ZERO) == (41, 41, 35, 0)
COVERED = N.EL_COVERED

NCELL = 60
random.seed(8)
KIND = []          # 每一格：None 普通 / 0..4 落进哪一堆 / "cov" 已补
for i in range(NCELL):
    r = random.random()
    KIND.append(None if r < 0.62 else ("cov" if r < 0.70 else random.choice([0, 0, 1, 1, 2, 2, 3, 4])))
assert sum(1 for k in KIND if k is not None) >= 15

X0, DX, Y_TAPE = -6.45, 0.22, 2.05
BX = [-5.0, -2.5, 0.0, 2.5, 5.0]
Y_B = 0.2


def tx(s, size=22, color=WHITE):
    return Text(s, font_size=size, color=color)


class ScanSweep(Scene):
    def construct(self):
        marks = []
        title = tx("先编译，再扫描：同一个模式，在哪儿出现就在哪儿亮", 28).to_edge(UP)
        tape = VGroup(*[Square(side_length=0.18, stroke_color=GREY, stroke_width=1.5, fill_opacity=0)
                        .move_to([X0 + i * DX, Y_TAPE, 0]) for i in range(NCELL)])
        tlab = tx("整个训练步编出来的程序（示意：60 格）", 16, GREY_B).move_to([0, Y_TAPE + 0.35, 0])
        self.add(title, tape, tlab)
        self.wait(0.8)

        def cap(s, col=GREY_B):
            m = tx(s, 20, col).move_to([0, -3.35, 0])
            assert m.width < 13.6, (s, m.width)
            return m

        def mark():
            marks.append(round(self.renderer.time + 0.2, 2))

        # ── 第 1 步：扫一遍 ──
        clk = ValueTracker(-1.0)
        heads = VGroup(*[tx(n, 17, c).move_to([BX[k], Y_B + 0.75, 0]) for k, (n, _v, c) in enumerate(BUCKETS)])
        c1 = cap("第 1 步　找「同一颗融合里转成 BF16、又转回 FP32」：一格一格扫，亮了就按位置归堆")

        def scanner():
            x = X0 + clk.get_value() * DX
            return Rectangle(width=0.26, height=0.4, stroke_color=YELLOW, stroke_width=3).move_to([x, Y_TAPE, 0])

        def lit():
            g = VGroup()
            for i, k in enumerate(KIND):
                if k is None or clk.get_value() < i:
                    continue
                col = GREEN if k == "cov" else BUCKETS[k][2]
                g.add(Square(side_length=0.18, stroke_width=0, fill_color=col, fill_opacity=0.9).move_to([X0 + i * DX, Y_TAPE, 0]))
            return g

        def counts():
            p = max(0.0, min(1.0, (clk.get_value() + 1) / NCELL))
            g = VGroup()
            for k, (_n, v, c) in enumerate(BUCKETS):
                n = int(round(v * p))
                g.add(tx("%d 处" % n, 26, c).move_to([BX[k], Y_B, 0]))
            g.add(tx("已补上，跳过：%d 处" % int(round(COVERED * p)), 18, GREEN).move_to([0, Y_B - 0.75, 0]))
            return g
        sc, lt, ct = always_redraw(scanner), always_redraw(lit), always_redraw(counts)
        self.play(FadeIn(c1), FadeIn(heads), FadeIn(sc), FadeIn(ct))
        self.add(lt)
        self.play(clk.animate.set_value(NCELL), run_time=6.0, rate_func=linear)
        tot = tx("一共 %d 处往返（按使用次数）；三根橙色是先查的范围" % N.EL_TOTAL, 22, YELLOW).move_to([0, Y_B - 1.35, 0])
        self.play(FadeIn(tot))
        mark()
        self.wait(2.6)

        # ── 第 2 步：LLO 抽查只含往返的 kernel ──
        c2 = cap("第 2 步　整颗只含往返的 kernel 拿 LLO 一颗颗数：打包成 BF16 的指令都是 0 条")
        self.play(FadeOut(c1), FadeOut(tot), FadeOut(sc), FadeIn(c2))
        sq_on = VGroup(*[Square(side_length=0.22, stroke_color=ORANGE, stroke_width=2) for _ in range(K_ON)])
        sq_on.arrange_in_grid(rows=3, buff=0.06).move_to([-2.9, -1.75, 0])
        lab_on = tx("省舍入开：只含往返的 %d 颗" % K_ON, 18, ORANGE).next_to(sq_on, UP, buff=0.12)
        self.play(FadeIn(sq_on), FadeIn(lab_on))
        zeros = VGroup(*[tx("0", 13, RED).move_to(s.get_center()) for s in sq_on])
        self.play(*[FadeIn(z) for z in zeros], run_time=1.6, lag_ratio=0.04)
        mark()
        self.wait(2.4)

        # ── 第 3 步：负对照 ──
        c3 = cap("第 3 步　负对照：关掉省舍入再编，同类 kernel 没有一颗是 0 条，「0 条」确实是省舍入造成的")
        sq_off = VGroup(*[Square(side_length=0.22, stroke_color=GREEN, stroke_width=2, fill_color=GREEN, fill_opacity=0.35)
                          for _ in range(K_OFF)])
        sq_off.arrange_in_grid(rows=3, buff=0.06).move_to([3.4, -1.75, 0])
        lab_off = tx("省舍入关：同类 %d 颗，0 条的 %d 颗" % (K_OFF, N.EL_K_NOEXC_ZERO), 18, GREEN).next_to(sq_off, UP, buff=0.12)
        self.play(FadeOut(c2), FadeIn(c3), FadeIn(sq_off), FadeIn(lab_off))
        mark()
        self.wait(3.0)

        # ── 收尾：回到第 0 帧 ──
        clk.set_value(-1.0)
        self.play(FadeOut(VGroup(c3, sq_on, zeros, lab_on, sq_off, lab_off, heads, ct, lt)), run_time=0.8)
        self.remove(ct, lt, sc)
        self.wait(0.5)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "ScanSweep.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
