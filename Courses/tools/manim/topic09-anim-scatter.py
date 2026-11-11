# -*- coding: utf-8 -*-
r"""专题九 · 案例二：不排序「读—加—舍—写」对排序「片上 FP32 累加、舍一次」（ScatterRound）。

渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic09-anim-scatter.py ScatterRound <草稿目录>/topic09-scatter.mp4 --draft
   构建手册/脚本/render.sh tools/manim/topic09-anim-scatter.py ScatterRound WebPages/media/topic09-scatter.mp4
每一步落地的停顿时刻写进 tools/manim/steps/ScatterRound.json（课件挂 data-pauses）。
挂在课件 §8.5 静态图 fig9-rmw 下面。

⭐ 为什么非动不可：论点是「同一个词每出现一次就多舍一次，舍入跟着下标流一次次攒起来」。静态图摆得出结果
   （红框几个），摆不出「舍入是随着下标流一个个经过才出现的」：不排序时同一个词的更新隔着别的词，每来一次都要
   把那一行读出、加上、舍回；排序后同一个词挨在一起，在片上缓冲里一口气加完、最后才舍一次 —— 增量来自时间。

⭐ 数据从哪来（第二次迁移 ＝ 混元 4 缩小版，TPU v5p 单算子；全部取自 topic09_numbers）：
   · 第 3 步：这一行在一次 4096 个下标的 scatter 里出现 ≥ 100 次的那组，离 FP64：不排序 C2_BYCOUNT 1.46e-2、
     排序 1.70e-3（实测；更新量是随机 BF16）；
   · 第 4 步：训练第 1 步，39 个高频词的行：纯数据并行 1.57e-2、FSDP 3.26e-3（实测，C2_GROUPS）。
   · 舍入次数：同一个词出现 k 次，不排序舍 k − 1 次（第一次加到 0 上不舍），排序舍 1 次 —— 按定义成立，由 OCC 现算并断言。
   ⚠️ 示意：下标流里 14 个位置、词 17 出现 6 次（真实高频词一次 scatter 里几十到几百次）；右边「累计舍入误差」条长按舍入次数画（5 对 1），
     只是示意，不是实测（屏上标「示意」）；token 号随手写。

⭐ 刻意没画：别的词那几行的更新（只跟着词 17 一行）；排序本身的耗时；excess precision 关掉的那一格（静态图中块有）。
"""
import json
import math
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, RoundedRectangle, Line, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, UP, DOWN, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic09_numbers as N   # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── 数（全部来自 topic09_numbers）────────────────────────────────────
K100_UNS, K100_SRT = N.C2_BYCOUNT["bf16_unsorted_excess_on"][3], N.C2_BYCOUNT["bf16_presorted_excess_on"][3]
HOT_DP, HOT_FSDP = N.C2_GROUPS[2][4], N.C2_GROUPS[2][3]
E = lambda v: ("%.2e" % v).replace("e-0", "e-")   # noqa: E731
assert (E(K100_UNS), E(K100_SRT), E(HOT_DP), E(HOT_FSDP)) == ("1.46e-2", "1.70e-3", "1.57e-2", "3.26e-3")

# ── 示意：下标流（不排序）与排好序的流；词 17 是要跟的那一行 ─────────────────
UNSORTED = ["17", "5", "902", "17", "33", "17", "5", "61", "17", "902", "17", "33", "61", "17"]
SORTED = sorted(UNSORTED, key=lambda s: (s != "17", int(s)))   # 示意：17 那一段挨在一起（真实排序按 token 号）
WORD = "17"
OCC = UNSORTED.count(WORD)
N_ROUND = {"uns": OCC - 1, "srt": 1}
assert OCC == 6 and N_ROUND == {"uns": 5, "srt": 1} and SORTED.count(WORD) == OCC

# ── 时间轴 ─────────────────────────────────────────────────────────
T_START = 0.8
STEP = 0.42                     # 下标流里每个位置经过的时间
LANE_DUR = len(UNSORTED) * STEP + 0.6
SEGS = [("s1", LANE_DUR, 2.2), ("s2", LANE_DUR, 2.2), ("s3", 1.4, 2.4), ("s4", 1.2, 3.0)]
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
assert 15 <= T_END + 0.5 <= 40, T_END
assert all(h >= 2.0 for _n, _d, h in SEGS)

# ── 版面 ──────────────────────────────────────────────────────────
LANE_Y = {"uns": 0.75, "srt": -1.45}
X0, DX = -4.6, 0.52              # 下标流的格子
CELL_X = 3.35                    # 词 17 那一行（不排序：BF16 表里那一格；排序：片上 FP32 缓冲）
BAR_X0, BAR_MAX = 4.2, 2.4
LAB_X = -6.05
assert X0 + DX * (len(UNSORTED) - 1) < CELL_X - 0.8 and CELL_X + 0.6 < BAR_X0 and BAR_X0 + BAR_MAX < 7.05


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


class ScatterRound(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def clock():
            return tr.get_value()

        seq = {"uns": UNSORTED, "srt": SORTED}
        seg = {"uns": "s1", "srt": "s2"}

        def arrive(lane, i):            # 第 i 个下标到达处理点的时刻
            return T[seg[lane]] + 0.3 + i * STEP

        def round_times(lane):
            ts = []
            idx = [i for i, s in enumerate(seq[lane]) if s == WORD]
            if lane == "uns":
                ts = [arrive(lane, i) + STEP * 0.6 for i in idx[1:]]          # 第一次加到 0 上不舍
            else:
                ts = [arrive(lane, idx[-1]) + STEP * 0.9]                     # 一段加完，写出时舍一次
            assert len(ts) == N_ROUND[lane]
            return ts

        def n_rounded(lane, t):
            return sum(ease((t - a) / 0.2) for a in round_times(lane))

        # ── 静态件 ──
        title = tx("同一个词的更新怎么加进词表那一行", 30).to_edge(UP)
        lab_a = VGroup(tx("不排序", 22, RED), tx("训练里纯数据并行", 16, GREY_B)).arrange(DOWN, buff=0.08
                                                                                ).move_to([LAB_X, LANE_Y["uns"], 0])
        lab_b = VGroup(tx("排序＋标记", 20, GREEN), tx("FSDP 那一侧", 16, GREY_B)).arrange(DOWN, buff=0.08
                                                                               ).move_to([LAB_X, LANE_Y["srt"], 0])
        cell_a = VGroup(tx("词 17 那一行", 16, GREY_B), tx("BF16 表", 16, GREY_B)).arrange(DOWN, buff=0.06
                                                                                ).move_to([CELL_X, LANE_Y["uns"] + 0.85, 0])
        cell_b = VGroup(tx("词 17 那一行", 16, GREY_B), tx("片上 FP32 缓冲", 16, GREEN)).arrange(DOWN, buff=0.06
                                                                                    ).move_to([CELL_X, LANE_Y["srt"] + 0.85, 0])
        lanes = VGroup(*[Line([X0 - 0.3, y - 0.36, 0], [CELL_X - 0.6, y - 0.36, 0], stroke_color=GREY, stroke_width=2)
                         for y in LANE_Y.values()])
        self.add(title, lab_a, lab_b, cell_a, cell_b, lanes)

        # 下标流：每个 token 号一格，经过时高亮；词 17 用蓝色
        def stream(lane):
            def draw():
                t = clock()
                y = LANE_Y[lane]
                g = VGroup()
                for i, s in enumerate(seq[lane]):
                    a = arrive(lane, i)
                    hot = a <= t < a + STEP
                    col = BLUE if s == WORD else GREY
                    g.add(RoundedRectangle(corner_radius=0.06, width=0.48, height=0.42, fill_color=col,
                                           fill_opacity=(0.55 if hot else 0.15), stroke_color=YELLOW if hot else col,
                                           stroke_width=3 if hot else 1.5).move_to([X0 + i * DX, y, 0]))
                return g
            return draw
        self.add(always_redraw(stream("uns")), always_redraw(stream("srt")))
        labels = VGroup()
        for lane in ("uns", "srt"):
            for i, s in enumerate(seq[lane]):
                labels.add(tx(s, 15, WHITE).move_to([X0 + i * DX, LANE_Y[lane], 0]))
        self.add(labels)

        # 词 17 那一格：不排序时每次「读—加—舍—写」，舍那一刻闪红；排序时缓冲变绿、最后舍一次
        def cell(lane):
            def draw():
                t = clock()
                y = LANE_Y[lane]
                flash = max([win(t, a - 0.05, a + 0.35, 0.1) for a in round_times(lane)] + [0.0])
                adding = any(arrive(lane, i) <= t < arrive(lane, i) + STEP for i, s in enumerate(seq[lane]) if s == WORD)
                base = GREEN if lane == "srt" else WHITE
                stroke = RED if flash > 0.05 else (BLUE if adding else base)
                fill_o = 0.2 + 0.5 * flash
                return RoundedRectangle(corner_radius=0.08, width=1.05, height=0.62,
                                        fill_color=RED if flash > 0.05 else base, fill_opacity=fill_o,
                                        stroke_color=stroke, stroke_width=4
                                        ).move_to([CELL_X, y, 0])
            return draw
        self.add(always_redraw(cell("uns")), always_redraw(cell("srt")))

        def counter(lane):
            def draw():
                t = clock()
                n = sum(1 for a in round_times(lane) if t >= a - 0.05)   # 跟闪红同一刻更新
                o = clip01((t - T[seg[lane]]) / 0.3) * gfade(t)
                m = tx("舍了 %d 次" % n, 18, RED if lane == "uns" else GREEN).move_to([CELL_X, LANE_Y[lane] - 0.62, 0])
                m.set_opacity(o)
                return m
            return draw
        self.add(always_redraw(counter("uns")), always_redraw(counter("srt")))

        # 右栏：第 1、2 步累计舍入误差（示意）；第 3、4 步换成实测
        S3, S4 = T["s3"], T["s4"]
        err_head = tx("累计舍入误差（示意）", 15, GREY_B).move_to([BAR_X0 + BAR_MAX / 2, LANE_Y["uns"] + 0.85, 0])

        def err_bars():
            t = clock()
            vis = 1.0 - clip01((t - S3) / 0.4)
            g = VGroup()
            for lane in ("uns", "srt"):
                L = BAR_MAX * n_rounded(lane, t) / N_ROUND["uns"]      # 示意：按舍入次数画（5 次对 1 次）
                if L > 1e-3 and vis > 0:
                    g.add(Rectangle(width=L, height=0.32, fill_color=RED if lane == "uns" else GREEN,
                                    fill_opacity=0.85 * vis, stroke_width=0).move_to([BAR_X0 + L / 2, LANE_Y[lane], 0]))
            return g
        self.add(always_redraw(err_bars))

        def meas_bars(a, vals, t0):
            def draw():
                t = clock()
                p = ease((t - t0) / 0.8)
                vis = (1.0 - clip01((t - (S4 if t0 < S4 else 1e9)) / 0.4)) * gfade(t)
                g = VGroup()
                if p <= 0 or vis <= 0:
                    return g
                for lane, v, col in zip(("uns", "srt"), vals, (RED, GREEN)):
                    L = max(0.06, BAR_MAX * v / max(vals)) * p
                    g.add(Rectangle(width=L, height=0.32, fill_color=col, fill_opacity=0.85 * vis, stroke_width=0
                                    ).move_to([BAR_X0 + L / 2, LANE_Y[lane], 0]))
                return g
            return draw
        self.add(always_redraw(meas_bars(None, (K100_UNS, K100_SRT), S3 + 0.2)))
        self.add(always_redraw(meas_bars(None, (HOT_DP, HOT_FSDP), S4 + 0.2)))
        head3 = tx("单算子·≥100 次的行（实测）", 15, GREY_B).move_to([BAR_X0 + BAR_MAX / 2, LANE_Y["uns"] + 0.85, 0])
        head4 = tx("训练第 1 步·高频行（实测）", 15, GREY_B).move_to([BAR_X0 + BAR_MAX / 2, LANE_Y["uns"] + 0.85, 0])
        v3a = tx("离 FP64 %s" % E(K100_UNS), 17, RED).move_to([BAR_X0 + BAR_MAX / 2, LANE_Y["uns"] - 0.5, 0])
        v3b = tx("离 FP64 %s" % E(K100_SRT), 17, GREEN).move_to([BAR_X0 + BAR_MAX / 2, LANE_Y["srt"] - 0.5, 0])
        v4a = tx("离 FP64 %s" % E(HOT_DP), 17, RED).move_to([BAR_X0 + BAR_MAX / 2, LANE_Y["uns"] - 0.5, 0])
        v4b = tx("离 FP64 %s" % E(HOT_FSDP), 17, GREEN).move_to([BAR_X0 + BAR_MAX / 2, LANE_Y["srt"] - 0.5, 0])
        assert max(m.width for m in (err_head, head3, head4)) < 7.1 - BAR_X0 + 0.6, [m.width for m in (err_head, head3, head4)]

        note = tx("示意：词 17 出现 6 次；真实高频词一次加几十到几百次；token 号随手写", 16, GREY_B).move_to([0, -2.75, 0])
        fin = tx("出现 k 次：不排序舍 k − 1 次，排序只舍 1 次", 22, YELLOW).move_to([0, -3.35, 0])

        S1, S2 = T["s1"], T["s2"]
        caps = [
            (S1, "第 1 步　不排序：同一个词的更新隔着别的词，每来一次都要读出、加上、舍回 BF16、写回", GREY_B),
            (S2, "第 2 步　排好序、带标记：同一个词挨在一起，在片上 FP32 缓冲里一口气加完，写出时舍一次", GREY_B),
            (S3, "第 3 步　单算子实测：出现 ≥ 100 次的行，不排序 %s，排序 %s" % (E(K100_UNS), E(K100_SRT)), GREY_B),
            (S4, "第 4 步　训练第 1 步：高频行纯数据并行 %s，FSDP %s" % (E(HOT_DP), E(HOT_FSDP)), YELLOW),
        ]
        END = T_OUT + FADE
        cap0 = tx("词嵌入的反向：每个 token 的梯度按 token 号加回词表那一行（scatter-add）", 22, GREY_B).next_to(title, DOWN, buff=0.22)
        cap_items = []
        for k, (a, s, col) in enumerate(caps):
            b = caps[k + 1][0] if k + 1 < len(caps) else END
            m = tx(s, 21, col).next_to(title, DOWN, buff=0.22)
            assert m.width < 13.6, (s, m.width)
            cap_items.append((m, a, b))
        assert cap0.width < 13.6

        timed_all = cap_items + [(err_head, S1, S3), (head3, S3 + 0.2, S4), (v3a, S3 + 0.6, S4), (v3b, S3 + 0.6, S4),
                                 (head4, S4 + 0.2, END), (v4a, S4 + 0.6, END), (v4b, S4 + 0.6, END),
                                 (note, S1 + 0.5, END), (fin, S2 + LANE_DUR - 0.8, END)]

        def show(m, a, b):
            m.set_opacity(win(clock(), a, b, f=0.3 if b < END else FADE))
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

        self.play(tr.animate.set_value(T_START), run_time=T_START, rate_func=linear)
        marks = []
        for n, _d, h in SEGS:
            self.play(tr.animate.set_value(T[n + "_land"]), run_time=T[n + "_land"] - tr.get_value(), rate_func=linear)
            marks.append(round(self.renderer.time + 0.2, 2))
            self.play(tr.animate.set_value(T[n + "_land"] + h), run_time=h, rate_func=linear)
        assert abs(tr.get_value() - T_OUT) < 1e-6
        assert all(abs(m - p) < 0.3 for m, p in zip(marks, PAUSES)), (marks, PAUSES)   # 草稿 15 帧会多漂一点；页面用实际记下的 marks
        self.play(tr.animate.set_value(T_END), run_time=T_END - T_OUT, rate_func=linear)
        self.play(tr.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)
        self.wait(0.5)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "ScatterRound.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
