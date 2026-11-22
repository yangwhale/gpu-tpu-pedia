# -*- coding: utf-8 -*-
r"""专题九 §5.3「关键路径：注意力本来就压在一条链底下」—— CoverShrink。
放在静态图 fig-cover / fig-maxsum 下面；画面里不写课件节号。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  静态图并排画了「长注意力」和「短注意力」两组，画得出结果，画不出过程：
  注意力一点点缩短时，一层先跟着短（全兑现），短过橙色那条链以后，一层不再短、
  多省的部分变成红色的等。「省下多少」和「兑现多少」在拐点处分家 ——
  两个计数器一个继续涨、一个停住，这个分离只有在时间里才看得见。

⛔ 数据从哪来
  · 画面里的长度全是示意（不按比例），计数器用示意单位，不写毫秒。
  · 只有最后一句字幕是实测，取的是同一组「一重一轻」两颗芯片的 profile：注意力差 672 毫秒（N.ATTN_SAVED_MS）、等待多 541
    （N.HL_WAIT[1] − N.HL_WAIT[0] ＝ 1059 − 518）、一步只差 195（N.HL_STEP_DIFF）。148 是日志步时、另一个口径，不配在这里。都从
    tools/topic08b_numbers.py 取并断言。

⛔ 刻意没画
  · 一层里别的计算和通信段（只画注意力、其余计算、专家权重那条链）。
  · 最重的芯片本来就露在外面的那 518 毫秒等待（课件 fig-heavylight 已画）；这里第 0 帧注意力比链长、不等。
  · 毫秒刻度：长度是示意，硬配刻度会让人去量。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-cover.py CoverShrink WebPages/media/topic08b-cover.mp4
   拐点和实测那句的停顿写进 steps/CoverShrink.json（课件挂 data-pauses）。

⭐ 写法：时钟式。唯一的 ValueTracker ＝ 片内时间；注意力长度、字幕号都是 state(u) 的纯函数。
   复位只定义在 state 的倒放段（注意力长回去），末帧按定义就是第 0 帧。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Line, DashedLine, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, RED, YELLOW, ORANGE, UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── 实测（只用在最后一句字幕）──────────────────────────────────────
ATTN_MS = N.ATTN_SAVED_MS
WAIT_MS = N.HL_WAIT[1] - N.HL_WAIT[0]
STEP_MS = N.HL_STEP_DIFF
assert (ATTN_MS, WAIT_MS, STEP_MS) == (672, 541, 195)

# ── 几何（示意单位，不按比例）──────────────────────────────────────
X0 = -4.0              # 一层开始
C = 4.6                # 橙条（小工这一趟）长度，固定
A0, A1 = 7.0, 1.6      # 注意力：从明显比橙条长，缩到明显比橙条短
R = 0.9                # 师傅这一层的「其余计算」
NXT = 1.3              # 下一层露出的一截
Y_M, Y_S, H = 0.6, -0.8, 0.72     # 师傅 / 小工车道中心、条高
Y_SPAN = Y_M + H / 2 + 0.32         # 「一层」跨度线
assert A0 > C > A1


def layer_len(a):
    return max(a, C) + R


# 计数器（示意单位 ×10，整数）
SAVED_MAX = round((A0 - A1) * 10)
SHORT_MAX = round((A0 - C) * 10)
assert (SAVED_MAX, SHORT_MAX) == (54, 24)

# ── 时间表（秒）───────────────────────────────────────────────────
T_HOLD0 = 1.8                         # 第 0 帧：①
T_S1 = 4.6                            # A0 → C：②（全兑现）
T_K0 = T_HOLD0 + T_S1                 # 拐点
T_KH = 2.6                            # 拐点停顿：「从这里起……」
T_S2 = 4.6                            # C → A1：③（开始等）
T_E0 = T_K0 + T_KH                    # 第二段开始
T_E1 = T_E0 + T_S2                    # 缩完
T_H4 = 2.4                            # ④
T_H5 = 3.4                            # ⑤ 实测
T_RW0 = T_E1 + T_H4 + T_H5            # 倒放开始
T_RW = 1.4
T_TOTAL = T_RW0 + T_RW + 0.5
PAUSES = [round(T_K0 + 0.4, 2), round(T_E1 + T_H4 + 0.6, 2)]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def state(u):
    """片内时间 u → (注意力长度, 字幕号)。⭐ 复位只定义在这里。"""
    if u >= T_RW0:                                  # 倒放：注意力长回去，字幕回第一句
        p = clamp((u - T_RW0) / T_RW)
        return A1 + (A0 - A1) * p, 0
    if u < T_HOLD0:
        return A0, 0
    if u < T_K0:
        return A0 + (C - A0) * (u - T_HOLD0) / T_S1, 1
    if u < T_E0:
        return C, 2
    if u < T_E1:
        return C + (A1 - C) * (u - T_E0) / T_S2, 3
    if u < T_E1 + T_H4:
        return A1, 4
    return A1, 5


assert state(0.0) == (A0, 0) and state(T_TOTAL) == (A0, 0)   # 末帧 ≡ 首帧，按定义成立

CAPS = [
    ["师傅在算注意力，小工同时跑腿"],
    ["注意力变短：一层跟着变短"],
    ["从这里起，多省的都变成了等"],
    ["短过小工那趟腿：师傅开始等"],
    ["多省的计算，全变成了等"],
    ["实测两颗芯片：注意力差 %d 毫秒" % ATTN_MS, "等待多 %d，一步只差 %d 毫秒" % (WAIT_MS, STEP_MS)],
]
assert all(len(s) <= 22 for c in CAPS for s in c)
TITLE = "注意力一直在缩，一层却不跟着短"


def tx(s, size=24, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def bar(x0, x1, y, color, fill=0.75):
    w = max(x1 - x0, 1e-3)
    return Rectangle(width=w, height=H, stroke_width=0, fill_color=color, fill_opacity=fill if x1 - x0 > 1e-3 else 0).move_to([(x0 + x1) / 2, y, 0])


class CoverShrink(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def st():
            return state(tr.get_value())

        title = tx(TITLE, 32, WHITE, True).to_edge(UP, buff=0.35)
        self.add(title)

        # 车道标签、橙条（固定）、拐点虚线
        self.add(tx("师傅（主核）", 24, BLUE, True).move_to([-5.75, Y_M, 0]))
        self.add(tx("小工（小核）", 24, ORANGE, True).move_to([-5.75, Y_S, 0]))
        self.add(bar(X0, X0 + C, Y_S, ORANGE))
        self.add(tx("这一层专家权重收集＋梯度规约", 22, WHITE).move_to([X0 + C / 2, Y_S, 0]))
        self.add(DashedLine([X0 + C, Y_S - H / 2 - 0.1, 0], [X0 + C, Y_M + H / 2 + 0.1, 0], color=GREY, stroke_width=3))
        self.add(Line([X0, Y_M - H / 2 - 0.08, 0], [X0, Y_S + H / 2 + 0.08, 0], color=GREY, stroke_width=3))
        orig_end = X0 + layer_len(A0)
        self.add(Line([orig_end, Y_SPAN - 0.14, 0], [orig_end, Y_SPAN + 0.14, 0], color=GREY, stroke_width=3))
        self.add(tx("原来一层到这", 22, GREY).move_to([orig_end, Y_SPAN + 0.32, 0]))

        # 条内文字：建一次，按长度切透明度 / 挪位置
        lab_attn = tx("注意力", 24, WHITE, True)
        lab_rest = tx("其余", 22, WHITE)
        lab_wait = tx("停着等", 24, WHITE, True)
        lab_next = tx("下一层…", 22, GREY_B)
        lab_span = tx("一层", 22, YELLOW, True)

        def upd_labels(_m=None):
            a, _ = st()
            end = X0 + max(a, C)
            lab_attn.move_to([X0 + a / 2, Y_M, 0])
            lab_rest.move_to([end + R / 2, Y_M, 0])
            lab_next.move_to([end + R + NXT / 2 + 0.05, Y_M, 0])
            lab_wait.move_to([X0 + (a + C) / 2, Y_M, 0]).set_opacity(clamp((C - a - 0.9) / 0.4))
            lab_span.move_to([X0 + layer_len(a) / 2, Y_SPAN + 0.28, 0])

        labels = VGroup(lab_attn, lab_rest, lab_wait, lab_next, lab_span)

        def lanes():
            a, _ = st()
            end = X0 + max(a, C)
            g = VGroup(bar(X0, X0 + a, Y_M, BLUE),
                       bar(X0 + a, end, Y_M, RED) if a < C else VGroup(),
                       bar(end, end + R, Y_M, BLUE, 0.4),
                       Rectangle(width=NXT, height=H, stroke_color=GREY, stroke_width=2, fill_opacity=0)
                       .move_to([end + R + NXT / 2 + 0.05, Y_M, 0]),
                       Line([X0, Y_SPAN, 0], [end + R, Y_SPAN, 0], color=YELLOW, stroke_width=4))
            return g

        self.add(always_redraw(lanes))
        upd_labels()                                   # ⛔ 先手动算一次
        labels.add_updater(upd_labels)
        self.add(labels)

        # 计数器：标签、条（每帧现画）、数字（整数，Text 缓存）
        CY = [-2.05, -2.7]
        CX0 = -2.6
        self.add(tx("省下的计算", 24, BLUE, True).move_to([CX0 - 1.45, CY[0], 0]))
        self.add(tx("一层变短了", 24, YELLOW, True).move_to([CX0 - 1.45, CY[1], 0]))
        self.add(tx("（示意单位）", 22, GREY).move_to([CX0 - 1.45, -3.3, 0]))
        cache = {}

        def num(v, color):
            k = (v, color)
            if k not in cache:
                cache[k] = tx(str(v), 26, color, True)
            return cache[k].copy()

        def counters():
            a, _ = st()
            saved = A0 - a
            short = A0 - max(a, C)
            g = VGroup()
            for y, v, col in ((CY[0], saved, BLUE), (CY[1], short, YELLOW)):
                w = max(v, 1e-3)
                g.add(Rectangle(width=w, height=0.4, stroke_width=0, fill_color=col, fill_opacity=0.8 if v > 1e-3 else 0)
                      .move_to([CX0 + w / 2, y, 0]))
                g.add(num(round(v * 10), col).move_to([CX0 + w + 0.45, y, 0]))
            return g

        self.add(always_redraw(counters))

        # 字幕：建一次，按区间切透明度
        caps = []
        for k, lines in enumerate(CAPS):
            col = YELLOW if k in (2, 5) else GREY_B
            g = VGroup(*[tx(s, 26, col, k == 5) for s in lines]).arrange(DOWN, buff=0.12)
            g.move_to([0, 2.6, 0])
            caps.append(g)

        def show_cap(m, k):
            m.set_opacity(1.0 if st()[1] == k else 0.0)
        for k, m in enumerate(caps):
            show_cap(m, k)                              # ⛔ 先手动算一次
            m.add_updater(lambda m, k=k: show_cap(m, k))
            self.add(m)

        self.play(tr.animate.set_value(T_TOTAL), run_time=T_TOTAL, rate_func=linear)
        self.play(tr.animate.set_value(T_TOTAL), run_time=1 / 30, rate_func=linear)   # 强制刷一帧（末帧 ＝ 第 0 帧）
        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "CoverShrink.json"), "w") as fh:
            json.dump({"pauses": PAUSES, "duration": round(self.renderer.time, 2)}, fh)
