# -*- coding: utf-8 -*-
r"""专题八 · 设备一到 8 个，编译器把一次矩阵乘拆成几段（WindowedAcc）。

渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08-anim-windowed.py WindowedAcc <草稿目录>/topic08-windowed.mp4 --draft
   构建手册/脚本/render.sh tools/manim/topic08-anim-windowed.py WindowedAcc WebPages/media/topic08-windowed.mp4
每一步落地的停顿时刻写进 tools/manim/steps/WindowedAcc.json（课件挂 data-pauses）。
挂在课件 §5.4 静态图 fig8-windowed 下面。

2026-11-13 改：原来画的是「4 片 4 轮、舍在轮与轮之间（推断）」的示意；第三次迁移的编译产物把循环体看清了，
   改成照实画：4 个设备先收齐再乘（舍 1 次）→ 8 个设备边传边乘（2 轮 × 两段 704 宽，舍 6 次）→ 打开投影 FP32 累加（舍 1 次）。

⭐ 为什么非动不可：论点是「设备一多，同一个求和被拆成几段，每写出一段、每加一次就多舍一次」——
   红块是随着权重片一片片到、一段段写出来才冒出来的，计数一格格往上涨；4 个设备那一行是权重整块先到、
   只乘一次。静态图 fig8-windowed 并排摆出了结果（1 个、6 个、1 个红块），画不出「边传边乘」本身：
   「片在路上、这边已经在乘」就是时间。

⭐ 数据从哪来（全部取自 topic08_numbers 的 M3_* 块；出处：对齐记录里第三次迁移的端到端说明与窗口化循环扫描结果）：
   · 循环体结构：8 片排 2 轮，每轮两次 704 宽的矩阵乘、输出 BF16，再加进 BF16 累加器（编译产物显示）；
     舍入次数 M3_TPU_ROUNDINGS ＝ 6 由 M3_ROUNDS、M3_MM_PER_ROUND 现算并断言；
   · 单算子 R：M3_PROJ_R 1.41 → M3_PROJ_R_AFTER 1.00（实测）；
   · 离线编译：BF16 累加的窗口化循环 4 个设备 0 个、8 和 16 个设备各 20 个（M3_SCAN，实测）。
   ⚠️ 示意：权重片从顶上「飞」下来只表示「这一段要用到哪几片」，不画环上谁传给谁；时间长短不按比例。

⭐ 刻意没画：环的两个方向；激活怎么切；反向（权重梯度一侧累加器本来是 FP32）；用编译参数整个关掉窗口化那条路
   （课件折叠里讲）；混元 3 的数（静态图右下角有）。
"""
import json
import math
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, RoundedRectangle, Line, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, ORANGE, PURPLE, UP, DOWN, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08_numbers as N   # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── 数（全部来自 topic08_numbers）────────────────────────────────────
R_BEFORE_T, R_AFTER_T = N.M3_PROJ_R_T, N.M3_PROJ_R_AFTER_T
assert (R_BEFORE_T, R_AFTER_T) == ("1.41", "1.00")
NR, NM = N.M3_ROUNDS, N.M3_MM_PER_ROUND
assert NR * (NM + 1) == N.M3_TPU_ROUNDINGS == 6 and N.M3_MM_K == 704 and N.M3_WIDTH == 2816
SCAN = {d: bf for chip, d, _l, bf in N.M3_SCAN if chip == "v5p"}
assert SCAN == {4: 0, 8: 20, 16: 20}
NE = NR * (NM + 1)                                   # 8 个设备那两行的格子数：每轮 NM 段矩阵乘 ＋ 1 次累加

# ── 时间轴（时钟 ＝ 场景时间）──────────────────────────────────────────
T_START = 0.8
FLY = 0.6                     # 权重片落下
EL = 0.8                      # 一格（一段矩阵乘或一次累加）
RD_GAP = 0.4                  # 两轮之间
LANE_DUR = NR * (NM + 1) * EL + (NR - 1) * RD_GAP + 0.6
SEGS = [("s1", 2.4, 2.2), ("s2", LANE_DUR, 2.4), ("s3", LANE_DUR, 2.2), ("s4", 1.4, 3.2)]
T = {}
_c = T_START
for _n, _d, _h in SEGS:
    T[_n] = _c
    T[_n + "_land"] = _c + _d
    _c += _d + _h
T_OUT = _c
FADE = 0.7
T_END = T_OUT + FADE + 0.2
PAUSES = [round(T[n + "_land"] + 0.2, 2) for n, _d, _h in SEGS]
assert 10 <= T_END + 0.5 <= 40, T_END
assert all(h >= 2.0 for _n, _d, h in SEGS)

# ── 版面 ──────────────────────────────────────────────────────────
XS = [-3.9 + 1.42 * e for e in range(NE)]   # 8 个设备那两行每一格的中心
BOX_W, BOX_H = 1.08, 0.62
RND_DX = 0.66                              # 舍块相对格中心
SH_Y = 2.2                                 # 顶上那排权重片
SH_X = [-4.6 + 0.62 * k for k in range(8)]
LANE_Y = {"a": 1.0, "b": -0.45, "c": -1.9}
LAB_X = -5.95
CNT_X = 5.75
assert XS[-1] + RND_DX + 0.15 < CNT_X - 0.7


def clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def ease(x):
    x = clip01(x)
    return x * x * (3 - 2 * x)


def win(t, a, b, f=0.3):
    if t < a or t >= b:
        return 0.0
    return min(clip01((t - a) / f), clip01((b - t) / f))


def tx(s, size=22, color=WHITE):
    return Text(s, font_size=size, color=color)


def gfade(t):
    return 1.0 - clip01((t - T_OUT) / FADE)


def t_cell(lane, e):
    """8 个设备那两行：第 e 格开始的时刻。"""
    r, j = divmod(e, NM + 1)
    return T["s2" if lane == "b" else "s3"] + r * ((NM + 1) * EL + RD_GAP) + j * EL


def is_acc(e):
    return e % (NM + 1) == NM


def cut_at(lane, e):
    """这一格出口舍不舍：生产原样每格都舍；FP32 累加只在最后一格舍。"""
    return lane == "b" or e == NE - 1


def t_cut(lane, e):
    return t_cell(lane, e) + (0.2 if is_acc(e) else FLY) + 0.2


# 4 个设备那一行
TA_GATHER, TA_MM, TA_CUT = T["s1"] + FLY + 0.1, T["s1"] + FLY + 0.7, T["s1"] + FLY + 1.3
A_GX = XS[0]
A_MM_L, A_MM_R = XS[1] - BOX_W / 2, XS[-1] + BOX_W / 2
assert sum(1 for e in range(NE) if cut_at("b", e)) == 6 and sum(1 for e in range(NE) if cut_at("c", e)) == 1


class WindowedAcc(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def clock():
            return tr.get_value()

        # ── 静态件 ──
        title = tx("设备一到 8 个，一次矩阵乘被拆成几段", 30).to_edge(UP)
        sh_lab = tx("8 片权重", 18, GREY_B).move_to([LAB_X, SH_Y, 0])
        shards = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.5, height=0.32, fill_color=ORANGE, fill_opacity=0.25,
                                           stroke_color=ORANGE, stroke_width=2).move_to([x, SH_Y, 0]) for x in SH_X])
        sh_txt = VGroup(*[tx(str(k + 1), 16, ORANGE).move_to([x, SH_Y, 0]) for k, x in enumerate(SH_X)])
        labs = VGroup()
        for lane, (a, b, col) in {"a": ("4 个设备", "先收齐再乘", GREY_B), "b": ("8 个设备", "生产原样", RED),
                                  "c": ("8 个设备", "FP32 累加", GREEN)}.items():
            labs.add(VGroup(tx(a, 20, WHITE), tx(b, 16, col)).arrange(DOWN, buff=0.08).move_to([LAB_X, LANE_Y[lane], 0]))
        assert max(m.get_right()[0] for m in labs) < XS[0] - BOX_W / 2 - 0.1
        self.add(title, sh_lab, shards, sh_txt, labs)

        def pair_x(e):                         # 第 e 格用到的两片权重
            r, j = divmod(e, NM + 1)
            k = r * NM * 2 + j * 2
            return SH_X[k], SH_X[k + 1]

        # ── 动态件 ──
        def draw():
            t = clock()
            gf = gfade(t)
            g = VGroup()
            # 4 个设备：8 片一起落进「收齐」框 → 一次长矩阵乘 → 最后舍一次
            ya = LANE_Y["a"]
            a0 = T["s1"]
            if a0 <= t < a0 + FLY + 0.15:
                p = ease((t - a0) / FLY)
                o = 1.0 - clip01((t - a0 - FLY) / 0.15)
                for x in SH_X:
                    g.add(Rectangle(width=0.36, height=0.24, fill_color=ORANGE, fill_opacity=0.9 * o, stroke_width=0
                                    ).move_to([x * (1 - p) + A_GX * p, (SH_Y - 0.3) * (1 - p) + ya * p, 0]))
            vg = clip01((t - TA_GATHER) / 0.2) * gf
            if vg > 0:
                g.add(RoundedRectangle(corner_radius=0.08, width=BOX_W, height=BOX_H, fill_color=ORANGE, fill_opacity=0.2 * vg,
                                       stroke_color=ORANGE, stroke_width=3, stroke_opacity=vg).move_to([A_GX, ya, 0]))
            vm = clip01((t - TA_MM) / 0.3) * gf
            if vm > 0:
                w = (A_MM_R - A_MM_L) * ease((t - TA_MM) / 0.3)
                g.add(RoundedRectangle(corner_radius=0.08, width=max(w, 0.05), height=BOX_H, fill_color=BLUE, fill_opacity=0.25 * vm,
                                       stroke_color=BLUE, stroke_width=3, stroke_opacity=vm).move_to([A_MM_L + w / 2, ya, 0]))
            vr = ease((t - TA_CUT) / 0.25) * gf
            if vr > 0:
                g.add(Rectangle(width=0.18, height=0.52, fill_color=RED, fill_opacity=0.35 * vr, stroke_color=RED,
                                stroke_width=3, stroke_opacity=vr).move_to([A_MM_R + 0.14, ya, 0]))
            # 8 个设备两行
            for lane in ("b", "c"):
                y = LANE_Y[lane]
                for e in range(NE):
                    c0 = t_cell(lane, e)
                    if not is_acc(e):
                        if c0 <= t < c0 + FLY + 0.15:
                            p = ease((t - c0) / FLY)
                            o = 1.0 - clip01((t - c0 - FLY) / 0.15)
                            for x in pair_x(e):
                                g.add(Rectangle(width=0.36, height=0.24, fill_color=ORANGE, fill_opacity=0.9 * o, stroke_width=0
                                                ).move_to([x * (1 - p) + XS[e] * p, (SH_Y - 0.3) * (1 - p) + y * p, 0]))
                        vb = clip01((t - c0 - FLY) / 0.15) * gf
                        col = BLUE
                    else:
                        vb = clip01((t - c0 - 0.2) / 0.2) * gf
                        col = PURPLE
                    if vb > 0:
                        g.add(RoundedRectangle(corner_radius=0.08, width=BOX_W, height=BOX_H, fill_color=col, fill_opacity=0.25 * vb,
                                               stroke_color=col, stroke_width=3, stroke_opacity=vb).move_to([XS[e], y, 0]))
                    tc = t_cut(lane, e)
                    vr = ease((t - tc) / 0.25) * gf
                    if vr > 0:
                        if cut_at(lane, e):
                            g.add(Rectangle(width=0.18, height=0.52, fill_color=RED, fill_opacity=0.35 * vr, stroke_color=RED,
                                            stroke_width=3, stroke_opacity=vr).move_to([XS[e] + RND_DX, y, 0]))
                        else:
                            g.add(Rectangle(width=0.18, height=0.52, fill_color=GREEN, fill_opacity=0.12 * vr, stroke_color=GREEN,
                                            stroke_width=2, stroke_opacity=0.7 * vr).move_to([XS[e] + RND_DX, y, 0]))
            return g
        dyn = always_redraw(draw)
        self.add(dyn)

        # 格里的字、轮次、计数（建一次，切透明度）
        END = T_OUT + FADE
        timed = []
        ya = LANE_Y["a"]
        timed.append((tx("收齐", 16, WHITE).move_to([A_GX, ya, 0]), TA_GATHER, END))
        timed.append((tx("一次矩阵乘，2816 宽，FP32 累加到底", 18, WHITE).move_to([(A_MM_L + A_MM_R) / 2, ya, 0]), TA_MM + 0.2, END))
        for lane in ("b", "c"):
            y = LANE_Y[lane]
            for e in range(NE):
                c0 = t_cell(lane, e)
                if is_acc(e):
                    m = tx("累加器", 16, WHITE).move_to([XS[e], y, 0])
                    timed.append((m, c0 + 0.2, END))
                else:
                    r, j = divmod(e, NM + 1)
                    k = r * NM * 2 + j * 2 + 1
                    m = tx("片 %d＋%d" % (k, k + 1), 15, WHITE).move_to([XS[e], y, 0])
                    timed.append((m, c0 + FLY, END))
                assert m.width < BOX_W - 0.08, m.width
            for r in range(NR):
                xm = (XS[r * (NM + 1)] + XS[r * (NM + 1) + NM]) / 2
                timed.append((tx("第 %d 轮" % (r + 1), 15, GREY_B).move_to([xm, y + 0.5, 0]), t_cell(lane, r * (NM + 1)), END))
        # 每行右边的计数：舍 n 次（随红块一格格涨）
        cnt_marks = []
        def add_count(lane, times, y, col_done):
            n = len(times)
            for i in range(n + 1):
                a = T["s1"] if i == 0 and lane == "a" else (times[i - 1] if i else (T["s2"] if lane == "b" else T["s3"]))
                b = times[i] if i < n else END
                col = (RED if lane == "b" and i > 1 else (GREEN if i == n else WHITE))
                if lane == "a":
                    col = GREEN if i == n else WHITE
                m = tx("舍 %d 次" % i, 20, col).move_to([CNT_X, y, 0])
                cnt_marks.append((m, a, b))
        add_count("a", [TA_CUT], ya, GREEN)
        add_count("b", [t_cut("b", e) for e in range(NE) if cut_at("b", e)], LANE_Y["b"], RED)
        add_count("c", [t_cut("c", e) for e in range(NE) if cut_at("c", e)], LANE_Y["c"], GREEN)

        # ── 第 4 步：实测 ──
        S1, S2, S3, S4 = T["s1"], T["s2"], T["s3"], T["s4"]
        fin1 = tx("单算子 R：%s → %s（实测）" % (R_BEFORE_T, R_AFTER_T), 22, WHITE).move_to([-3.2, -3.05, 0])
        fin2 = tx("离线编译：4 个设备 %d 个这种循环，8、16 个各 %d 个" % (SCAN[4], SCAN[8]), 22, WHITE).move_to([2.9, -3.05, 0])
        fin3 = tx("在 4 个设备上验精度，看不到这件事", 24, YELLOW).move_to([0, -3.6, 0])
        assert fin1.get_right()[0] < fin2.get_left()[0] - 0.2 and fin2.get_right()[0] < 7.0 and fin1.get_left()[0] > -7.0
        note = tx("结构取自编译产物：8 个设备上 q 低秩下投影的循环体；片从上面落下只表示用到哪几片", 16, GREY_B).move_to([0, -2.6, 0])
        assert note.width < 13.8

        caps = [
            (S1, "第 1 步　4 个设备：权重先整块收齐（切成 4 片），一次乘完，舍 1 次", GREY_B),
            (S2, "第 2 步　8 个设备：边传边乘，每段输出舍一次，加进 BF16 累加器再舍一次", GREY_B),
            (S3, "第 3 步　打开投影 FP32 累加：结构不变，中间全留 FP32，只在最后舍 1 次", GREY_B),
            (S4, "第 4 步　同一种切法，只是设备数变了", YELLOW),
        ]
        cap0 = tx("同一个往里投的投影，一次求和 2816 个乘积；红块 ＝ 一次舍到 BF16", 22, GREY_B).next_to(title, DOWN, buff=0.2)
        cap_items = []
        for k, (a, s_, col) in enumerate(caps):
            b = caps[k + 1][0] if k + 1 < len(caps) else END
            m = tx(s_, 22, col).next_to(title, DOWN, buff=0.2)
            assert m.width < 13.8, (s_, m.width)
            cap_items.append((m, a, b))
        assert cap0.width < 13.8

        timed_all = cap_items + timed + cnt_marks + [(note, S2, END), (fin1, S4 + 0.2, END), (fin2, S4 + 0.5, END), (fin3, S4 + 1.0, END)]

        def show(m, a, b):
            m.set_opacity(win(clock(), a, b, f=0.25 if b < END else FADE))
        for m, a, b in timed_all:
            show(m, a, b)
            m.add_updater(lambda mm, a=a, b=b: show(mm, a, b))
            self.add(m)

        def show0(m):
            t = clock()
            o = 1.0 - clip01((t - (S1 - 0.3)) / 0.3) if t < S1 else clip01((t - T_OUT) / FADE)
            m.set_opacity(o)
        show0(cap0)
        cap0.add_updater(show0)
        self.add(cap0)

        # ── 走时钟（必须 linear）──
        self.play(tr.animate.set_value(T_START), run_time=T_START, rate_func=linear)
        marks = []
        for n, _d, h in SEGS:
            self.play(tr.animate.set_value(T[n + "_land"]), run_time=T[n + "_land"] - tr.get_value(), rate_func=linear)
            marks.append(round(self.renderer.time + 0.2, 2))
            self.play(tr.animate.set_value(T[n + "_land"] + h), run_time=h, rate_func=linear)
        assert abs(tr.get_value() - T_OUT) < 1e-6
        assert all(abs(m - p) < 0.15 for m, p in zip(marks, PAUSES)), (marks, PAUSES)
        self.play(tr.animate.set_value(T_END), run_time=T_END - T_OUT, rate_func=linear)

        self.play(tr.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)
        self.wait(0.5)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "WindowedAcc.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
