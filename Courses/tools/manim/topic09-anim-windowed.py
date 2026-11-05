# -*- coding: utf-8 -*-
r"""专题九 · FSDP 下 collective matmul 的部分和舍入（WindowedAcc）。

渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic09-anim-windowed.py WindowedAcc <草稿目录>/topic09-windowed.mp4 --draft
   构建手册/脚本/render.sh tools/manim/topic09-anim-windowed.py WindowedAcc WebPages/media/topic09-windowed.mp4
每一步落地的停顿时刻写进 tools/manim/steps/WindowedAcc.json（课件挂 data-pauses）。
挂在课件 §5.4 静态图 fig9-windowed 下面。

⭐ 为什么非动不可：论点是「每多收一片、多算一轮，就多舍一次」—— 舍入次数跟着轮数长，误差一轮一轮攒起来；
   打开 FP32 累加以后，同样走四轮，前三轮一个红块都不出，只在最后舍一次。静态图把两行并排摆出了结果
   （4 个红块对 1 个红块），画不出「红块是随着部分和往下一轮传的那一刻才出现的、误差条是一格一格涨的」——
   「轮」本身就是时间，增量来自时间。

⭐ 数据从哪来（一个对齐实践：混元 3 截前 4 层，GPU B200×8 对 TPU v5p，同一份权重和数据；全部取自 topic09_numbers）：
   · qkv 投影输出对 GPU 的 rel_L2：QKV_VS_GPU_BEFORE 2.36e-3 → QKV_VS_GPU_AFTER 1.8e-4（实测）；
   · 对 FP64 的 R：R_QKV_BEFORE 1.235 → R_QKV_AFTER 1.000（实测）；
   · 1000 步参数范数那一项的 r（被测对精确轨迹，噪声取 GPU 生产程序重跑 10 对）：
     开关前 GOLDEN_R[2] ＝ 4.36e-7 ÷ 1.24e-7 ≈ 3.52，开关后 ACC_PN_R ＝ 3.45（实测，断言钉住两位小数）。
   · 舍入次数：默认 4 次（每轮出口一次，最后一轮的出口就是结果）、FP32 累加 1 次 —— 跟 fig9-windowed 同一口径，由 ROUNDS 现算并断言。
   ⚠️ 示意：4 片 4 轮（实际片数等于 FSDP 的路数）；右边「累计舍入误差」条的长度 ∝ √（已舍次数），
     只是「独立舍入一次次叠加」的示意，不是实测，跟第 3 步的实测 rel_L2 不成比例（屏上标「示意」）。
   ⚠️ 推断：「一轮之内照样 FP32 累加，舍的是轮与轮之间的部分和」——一手材料只写到循环累加器是 BF16，
     一轮之内的点积怎么累加是按矩阵单元的通常做法推的（屏上小字标「推断」）。

⭐ 刻意没画：环上权重片在设备之间怎么转（只画「这一轮收到第 k 片」）；激活怎么切；反向和输出层
   （开关只管前向这几处投影）；用编译参数整个关掉窗口化那条路（那样连 GPU 本来就有的一次舍入也省了，
   §5.4 出处里讲）；41 个循环各自的情况。
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
REL_BEFORE, REL_AFTER = N.QKV_VS_GPU_BEFORE, N.QKV_VS_GPU_AFTER
R_BEFORE, R_AFTER = N.R_QKV_BEFORE, N.R_QKV_AFTER
PN_R_BEFORE, PN_R_AFTER = N.GOLDEN_R[2], N.ACC_PN_R
REL_BEFORE_T, REL_AFTER_T = "%.2e" % REL_BEFORE, "%.2e" % REL_AFTER
PN_BEFORE_T, PN_AFTER_T = "%.2f" % PN_R_BEFORE, "%.2f" % PN_R_AFTER
assert (REL_BEFORE_T, REL_AFTER_T) == ("2.36e-03", "1.80e-04")
assert ("%.3f" % R_BEFORE, "%.3f" % R_AFTER) == ("1.235", "1.000")
assert (PN_BEFORE_T, PN_AFTER_T) == ("3.52", "3.45"), (PN_BEFORE_T, PN_AFTER_T)
assert REL_BEFORE / REL_AFTER > 10                     # 单层：差一个数量级以上
assert (PN_R_BEFORE - PN_R_AFTER) / PN_R_BEFORE < 0.03  # 1000 步：r 只少不到 3%，「几乎没动」
REL_SHORT = lambda v: "%.2e" % v                       # noqa: E731

# ── 示意：4 片 4 轮；每一轮出口舍不舍 ────────────────────────────────
K = 4
ROUNDS = {"bf16": [True] * K,                      # 默认：每轮的部分和以 BF16 传下去（最后一轮的出口就是结果）
          "fp32": [False] * (K - 1) + [True]}      # FP32 累加：只在最后舍一次
assert sum(ROUNDS["bf16"]) == 4 and sum(ROUNDS["fp32"]) == 1   # 跟 fig9-windowed「舍了 4 次 / 只舍 1 次」同一口径

# ── 时间轴（时钟 ＝ 场景时间）──────────────────────────────────────────
T_START = 0.8
RD = 1.1                       # 一轮：收片 0.45 → 部分积出现 → 部分和往下传 0.5
FLY, APPEAR, CARRY = 0.45, 0.15, 0.5
LANE_DUR = K * RD + 0.3
SEGS = [("s1", LANE_DUR, 2.0), ("s2", LANE_DUR, 2.0), ("s3", 1.6, 2.2), ("s4", 1.2, 3.0)]
T = {}
_c = T_START
for _n, _d, _h in SEGS:
    T[_n] = _c
    T[_n + "_land"] = _c + _d
    _c += _d + _h
T_OUT = _c                     # 收尾：累积件淡出、第一句字幕淡回
FADE = 0.7
T_END = T_OUT + FADE + 0.2
PAUSES = [round(T[n + "_land"] + 0.2, 2) for n, _d, _h in SEGS]
assert 10 <= T_END + 0.5 <= 40, T_END
assert all(h >= 2.0 for _n, _d, h in SEGS)            # 每一步落地停 ≥ 2 秒

# ── 版面 ──────────────────────────────────────────────────────────
XS = [-4.45 + 2.0 * k for k in range(K)]   # 每一轮那一格的中心
BOX_W, BOX_H = 1.12, 0.7
RND_DX, PLUS_DX = 0.8, 1.27               # 舍块、加号相对格中心的偏移
RES_X, RES_W = XS[-1] + 1.55, 0.9         # 结果框
DEV_Y = 1.85
LANE_Y = {"bf16": 0.45, "fp32": -1.35}
BAR_X0, BAR_MAX = 3.95, 2.75              # 右栏的条
LAB_X = -6.05
assert RES_X + RES_W / 2 < BAR_X0 - 0.3 and BAR_X0 + BAR_MAX < 7.0


def clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def ease(x):
    x = clip01(x)
    return x * x * (3 - 2 * x)


def win(t, a, b, f=0.3):
    """[a, b) 区间内可见，两端各 f 秒淡入淡出。"""
    if t < a or t >= b:
        return 0.0
    return min(clip01((t - a) / f), clip01((b - t) / f))


def tx(s, size=22, color=WHITE):
    return Text(s, font_size=size, color=color)


def gfade(t):
    """收尾淡出：所有累积件乘它，t ≥ T_OUT＋FADE 时全灭 ＝ 第 0 帧。"""
    return 1.0 - clip01((t - T_OUT) / FADE)


class WindowedAcc(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def clock():
            return tr.get_value()

        lane_seg = {"bf16": "s1", "fp32": "s2"}

        def r0(lane, k):                       # 这一行第 k 轮开始的时刻
            return T[lane_seg[lane]] + k * RD

        def n_rounded(lane, t):                # 到 t 为止这一行舍了几次（舍块在部分和经过时出现）
            n = 0.0
            for k, cut in enumerate(ROUNDS[lane]):
                if cut:
                    n += ease((t - (r0(lane, k) + FLY + APPEAR + CARRY * 0.45)) / 0.25)
            return n

        # ── 静态件（第 0 帧就在，全程不动）──
        title = tx("FSDP 下的 collective matmul：部分和在轮与轮之间舍到 BF16", 30).to_edge(UP)
        dev_lab = VGroup(tx("权重", 20, WHITE), tx("按收缩维切开", 16, GREY_B)).arrange(DOWN, buff=0.08
                                                                                  ).move_to([LAB_X, DEV_Y, 0])
        lab_a = VGroup(tx("默认", 22, RED), tx("BF16 累加器", 16, GREY_B)).arrange(DOWN, buff=0.08
                                                                              ).move_to([LAB_X, LANE_Y["bf16"], 0])
        lab_b = VGroup(tx("打开 FP32 累加", 18, GREEN), tx("FP32 累加器", 16, GREY_B)).arrange(DOWN, buff=0.08
                                                                                       ).move_to([LAB_X, LANE_Y["fp32"], 0])
        assert lab_b.get_right()[0] < XS[0] - BOX_W / 2 - 0.1, lab_b.get_right()
        lanes = VGroup(*[Line([XS[0] - BOX_W / 2, y, 0], [RES_X - RES_W / 2, y, 0], stroke_color=GREY, stroke_width=2)
                         for y in LANE_Y.values()])
        self.add(title, dev_lab, lab_a, lab_b, lanes)

        # 设备行：4 个设备各存一片（TPU 蓝）；这一轮在收的那片描黄
        dev_txt = VGroup()
        for k in range(K):
            dev_txt.add(tx("设备 %d · 第 %d 片" % (k + 1, k + 1), 15, BLUE).move_to([XS[k], DEV_Y, 0]))

        def receiving(t):
            for lane in ("bf16", "fp32"):
                for k in range(K):
                    if r0(lane, k) <= t < r0(lane, k) + FLY + APPEAR:
                        return k
            return -1

        def devices():
            t = clock()
            hot = receiving(t)
            g = VGroup()
            for k in range(K):
                g.add(RoundedRectangle(corner_radius=0.08, width=1.62, height=0.52, fill_color=BLUE, fill_opacity=0.18,
                                       stroke_color=YELLOW if k == hot else BLUE, stroke_width=4 if k == hot else 2
                                       ).move_to([XS[k], DEV_Y, 0]))
            return g
        dev_m = always_redraw(devices)
        self.add(dev_m, dev_txt)
        assert max(m.width for m in dev_txt) < 1.55

        # ── 两行的动态件 ──
        def lane_draw(lane):
            def draw():
                t = clock()
                y = LANE_Y[lane]
                gf = gfade(t)
                g = VGroup()
                for k in range(K):
                    a = r0(lane, k)
                    # 收片：一小块从设备 k 落到这一格
                    if a <= t < a + FLY + APPEAR:
                        p = ease((t - a) / FLY)
                        yy = (DEV_Y - 0.3) * (1 - p) + y * p
                        o = 1.0 - clip01((t - a - FLY) / APPEAR)
                        g.add(Rectangle(width=0.42, height=0.3, fill_color=BLUE, fill_opacity=0.9 * o,
                                        stroke_width=0).move_to([XS[k], yy, 0]))
                    # 部分积框
                    vb = clip01((t - a - FLY) / APPEAR) * gf
                    if vb > 0:
                        g.add(RoundedRectangle(corner_radius=0.08, width=BOX_W, height=BOX_H, fill_color=BLUE,
                                               fill_opacity=0.25 * vb, stroke_color=BLUE, stroke_width=3,
                                               stroke_opacity=vb).move_to([XS[k], y, 0]))
                    # 舍块
                    if ROUNDS[lane][k]:
                        vr = ease((t - (a + FLY + APPEAR + CARRY * 0.45)) / 0.25) * gf
                        if vr > 0:
                            g.add(Rectangle(width=0.2, height=0.56, fill_color=RED, fill_opacity=0.35 * vr,
                                            stroke_color=RED, stroke_width=3, stroke_opacity=vr
                                            ).move_to([XS[k] + RND_DX, y, 0]))
                    # 部分和往下一轮传（一颗小胶囊：BF16 白，FP32 绿）
                    c0 = a + FLY + APPEAR
                    if c0 <= t < c0 + CARRY:
                        p = (t - c0) / CARRY
                        x_from = XS[k] + BOX_W / 2
                        x_to = (XS[k + 1] - BOX_W / 2) if k + 1 < K else (RES_X - RES_W / 2)
                        x = x_from + (x_to - x_from) * p
                        col = WHITE if lane == "bf16" else GREEN
                        if lane == "bf16" and x > XS[k] + RND_DX:
                            col = RED
                        g.add(RoundedRectangle(corner_radius=0.07, width=0.3, height=0.18, fill_color=col,
                                               fill_opacity=1.0, stroke_width=0).move_to([x, y, 0]))
                # 结果框
                vres = clip01((t - (r0(lane, K - 1) + RD)) / 0.25) * gf
                if vres > 0:
                    col = RED if lane == "bf16" else GREEN
                    g.add(RoundedRectangle(corner_radius=0.08, width=RES_W, height=BOX_H, fill_color=col,
                                           fill_opacity=(0.25 if lane == "bf16" else 0.85) * vres, stroke_color=col,
                                           stroke_width=3, stroke_opacity=vres).move_to([RES_X, y, 0]))
                return g
            return draw
        lane_a_m = always_redraw(lane_draw("bf16"))
        lane_b_m = always_redraw(lane_draw("fp32"))
        self.add(lane_a_m, lane_b_m)

        # 格里的字、舍字、结果字（建一次，切透明度）
        timed = []
        for lane in ("bf16", "fp32"):
            y = LANE_Y[lane]
            for k in range(K):
                a = r0(lane, k) + FLY
                m = tx("部分积 %d" % (k + 1), 16, WHITE).move_to([XS[k], y, 0])
                assert m.width < BOX_W - 0.1, m.width
                timed.append((m, a, None))
                if k > 0:
                    timed.append((tx("＋", 22, WHITE).move_to([XS[k] - 2.0 + PLUS_DX, y, 0]), a, None))
                if ROUNDS[lane][k]:
                    timed.append((tx("舍", 16, RED).move_to([XS[k] + RND_DX, y - 0.5, 0]),
                                  r0(lane, k) + FLY + APPEAR + CARRY * 0.45, None))
            res_t = r0(lane, K - 1) + RD
            timed.append((tx("结果", 16, WHITE).move_to([RES_X, y, 0]), res_t, None))
            timed.append((tx("舍了 4 次" if lane == "bf16" else "只舍 1 次", 16, RED if lane == "bf16" else GREEN
                             ).move_to([RES_X, y - 0.55, 0]), res_t, None))
        assert sum(ROUNDS["bf16"]) == 4 and sum(ROUNDS["fp32"]) == 1   # 上面两句字跟 ROUNDS 一致

        # ── 右栏：第 1、2 步「累计舍入误差（示意）」；第 3 步换成实测 rel_L2 ──
        S3 = T["s3"]
        err_head = tx("累计舍入误差（示意）", 18, GREY_B).move_to([BAR_X0 + BAR_MAX / 2, DEV_Y, 0])

        def err_len(lane, t):
            return BAR_MAX * math.sqrt(n_rounded(lane, t)) / math.sqrt(K)

        def err_bars():
            t = clock()
            vis = 1.0 - clip01((t - S3) / 0.4)
            g = VGroup()
            for lane in ("bf16", "fp32"):
                L = err_len(lane, t)
                if L > 1e-3 and vis > 0:
                    g.add(Rectangle(width=L, height=0.32, fill_color=RED, fill_opacity=0.85 * vis, stroke_width=0
                                    ).move_to([BAR_X0 + L / 2, LANE_Y[lane], 0]))
            return g
        err_m = always_redraw(err_bars)
        self.add(err_m)

        cmp_head = tx("qkv 投影输出对 GPU（实测）", 18, GREY_B).move_to([BAR_X0 + BAR_MAX / 2, DEV_Y, 0])

        def cmp_bars():
            t = clock()
            p = ease((t - S3 - 0.4) / 0.8) * gfade(t)
            g = VGroup()
            if p <= 0:
                return g
            for lane, v, col in (("bf16", REL_BEFORE, RED), ("fp32", REL_AFTER, GREEN)):
                L = max(0.06, BAR_MAX * v / REL_BEFORE) * ease((t - S3 - 0.4) / 0.8)
                g.add(Rectangle(width=L, height=0.32, fill_color=col, fill_opacity=0.85 * p, stroke_width=0
                                ).move_to([BAR_X0 + L / 2, LANE_Y[lane], 0]))
            return g
        cmp_m = always_redraw(cmp_bars)
        self.add(cmp_m)
        cmp_a = tx("rel_L2 %s · R %.3f" % (REL_BEFORE_T, R_BEFORE), 17, RED).move_to(
            [BAR_X0 + BAR_MAX / 2, LANE_Y["bf16"] - 0.5, 0])
        cmp_b = tx("rel_L2 %s · R %.3f" % (REL_AFTER_T, R_AFTER), 17, GREEN).move_to(
            [BAR_X0 + BAR_MAX / 2, LANE_Y["fp32"] - 0.5, 0])
        assert max(cmp_a.width, cmp_b.width, cmp_head.width, err_head.width) < 7.1 - BAR_X0 + 0.2

        # ── 底部小字 ──
        note1 = tx("一轮之内照样 FP32 累加，舍的是轮与轮之间的部分和（推断）", 17, GREY_B).move_to([0, -2.55, 0])
        note2 = tx("示意：4 片 4 轮；实际片数等于 FSDP 的路数", 17, GREY_B).move_to([0, -2.9, 0])
        fin = tx("参数范数漂移的 r（对精确轨迹）：开关前 %s，开关后 %s" % (PN_BEFORE_T, PN_AFTER_T), 21, WHITE
                 ).move_to([0, -3.5, 0])

        # ── 字幕 ──
        S1, S2, S4 = T["s1"], T["s2"], T["s4"]
        caps = [
            (S1, "第 1 步　默认：一轮收一片、算一片的部分积；部分和传给下一轮前先舍到 BF16", GREY_B),
            (S2, "第 2 步　打开 FP32 累加：部分和一路用 FP32 传，只在最后舍一次", GREY_B),
            (S3, "第 3 步　qkv 投影对 GPU：rel_L2 %s → %s，R %.3f → %.3f" % (REL_BEFORE_T, REL_AFTER_T, R_BEFORE, R_AFTER),
             GREY_B),
            (S4, "单层这一处修好了；1000 步的参数范数漂移几乎没动（%s → %s）" % (PN_BEFORE_T, PN_AFTER_T), YELLOW),
        ]
        END = T_OUT + FADE              # 累积件都在这一刻淡完
        cap0 = tx("FSDP：权重按收缩维切成 4 片，每个设备存一片", 24, GREY_B).next_to(title, DOWN, buff=0.22)
        cap_items = []
        for k, (a, s, col) in enumerate(caps):
            b = caps[k + 1][0] if k + 1 < len(caps) else END
            m = tx(s, 24, col).next_to(title, DOWN, buff=0.22)
            assert m.width < 13.6, (s, m.width)
            cap_items.append((m, a, b))

        timed_all = cap_items + [(m, a, END if b is None else b) for m, a, b in timed] + [
            (err_head, S1, S3), (cmp_head, S3 + 0.3, END), (cmp_a, S3 + 1.0, END), (cmp_b, S3 + 1.0, END),
            (note1, S1 + 0.5, END), (note2, S1 + 0.5, END), (fin, S4 + 0.4, END),
        ]

        def show(m, a, b):
            m.set_opacity(win(clock(), a, b, f=0.3 if b < END else FADE))
        for m, a, b in timed_all:
            show(m, a, b)               # ⛔ 先手动算一次：updater 要等第一次 play 才跑
            m.add_updater(lambda mm, a=a, b=b: show(mm, a, b))
            self.add(m)

        def show0(m):                   # 第一句字幕：开头在，收尾淡回 —— 复位那一刻跟第 0 帧同一幕
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
            self.play(tr.animate.set_value(T[n + "_land"]), run_time=T[n + "_land"] - tr.get_value(),
                      rate_func=linear)
            marks.append(round(self.renderer.time + 0.2, 2))
            self.play(tr.animate.set_value(T[n + "_land"] + h), run_time=h, rate_func=linear)
        assert abs(tr.get_value() - T_OUT) < 1e-6
        assert all(abs(m - p) < 0.15 for m, p in zip(marks, PAUSES)), (marks, PAUSES)
        self.play(tr.animate.set_value(T_END), run_time=T_END - T_OUT, rate_func=linear)

        # ── 复位：时钟拨回 0（此刻累积件已全灭、第一句字幕已淡回）──
        self.play(tr.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)
        self.wait(0.5)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "WindowedAcc.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
