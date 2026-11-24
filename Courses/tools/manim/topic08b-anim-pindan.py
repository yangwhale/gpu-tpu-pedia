# -*- coding: utf-8 -*-
r"""专题九「稀疏注意力核」拼单那一小节的动画 —— PinDanGather。挂在静态图 fig-dsa-moe 下面；画面里不写课件节号。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · 「收拢」本身是一个动作：没被这一单挑过的 key 列消失，挑过的列往一起挤，16 列变成紧凑的 8 列。
    静态图只能左右并排两个终态，看的人要自己对「右边这 8 列分别是左边哪几列」；动起来每一列从哪来一眼可见。
  · 第三幕「小工收下一段时，师傅算这一段」是时间上的重叠，只有在时间里才看得见先后。

⛔ 数据从哪来
  · 玩具挑选和静态图 fig-dsa-moe 同一份（位置 12–15 的 4 个 query，各挑 4 个 key，含自己），并集 8 行、ρ ＝ 2 由数据当场算并断言。
  · 第三幕的段长、段数是示意（真实一段 1024 行，一层切 8 段）；不写毫秒。

⛔ 刻意没画
  · 不画 softmax、PV、反向；只画「收拢」和「收与算错开」。
  · 小工收的是 key 行（一行一行收），画成一段一段的紫块是示意。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-pindan.py PinDanGather WebPages/media/topic08b-pindan.mp4
   停顿点写进 steps/PinDanGather.json。

⭐ 写法：时钟式。唯一的 ValueTracker ＝ 片内时间；所有位置、透明度都是时间的纯函数；倒放段把画面送回第 0 帧。
"""
import json
import os

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Square, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, GREY_D, BLUE, RED, GREEN, YELLOW, PURPLE, UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR = os.path.join(HERE, "steps")

# ── 数据（同 fig-dsa-moe）──────────────────────────────────────────
TOY = {12: (0, 5, 11, 12), 13: (0, 5, 12, 13), 14: (0, 5, 13, 14), 15: (0, 9, 14, 15)}
QS = sorted(TOY)
UNION = sorted(set(k for v in TOY.values() for k in v))
assert UNION == [0, 5, 9, 11, 12, 13, 14, 15] and len(UNION) / 4 == 2.0
NK = 16
ACTIVE = [b for b in range(4) if any(b * 4 <= k < b * 4 + 4 for v in TOY.values() for k in v)]
assert ACTIVE == [0, 1, 2, 3]

# ── 几何 ─────────────────────────────────────────────────────────
CS = 0.44
GX0 = -4.2                                 # 第 0 列左边
GY0 = 1.55                                 # 第 0 行中心
def col_x(c):
    return GX0 + c * CS + CS / 2
def row_y(r):
    return GY0 - r * CS
def compact_x(j):
    return GX0 + j * CS + CS / 2

# ── 时间表（秒）─────────────────────────────────────────────────
T0 = 1.6                                   # 第 0 帧停
TB = 0.65                                  # 每块亮起
T1 = T0 + 4 * TB                           # 四块都亮
T2 = T1 + 1.8                              # 停：跟稠密一样
T3 = T2 + 0.6                              # 红框和不在单子里的列淡出
T4 = T3 + 1.8                              # 收拢
T5 = T4 + 2.6                              # 停：一单 ＝ 一个专家
NSEG = 4
TSEG = 0.8                                 # 每段
T6 = T5 + (NSEG + 1) * TSEG + 0.4          # 流水跑完
T7 = T6 + 2.0                              # 停
TRW = 1.6                                  # 倒放
T_TOTAL = T7 + TRW + 0.4
PAUSES = [round(T2 - 0.6, 2), round(T5 - 0.8, 2), round(T7 - 0.6, 2)]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(p):
    p = clamp(p)
    return p * p * (3 - 2 * p)


def phase(u):
    """片内时间 → (亮了几块, 红框透明度, 非并集列透明度, 收拢进度, 流水进度, 字幕号)。⭐ 复位只定义在这里。"""
    if u >= T7:                                            # 倒放：先散回原位，再把其余列淡回来
        p = clamp((u - T7) / TRW)
        return 0, 0.0, ease((p - 0.6) / 0.4), 1 - ease(p / 0.6), 0.0, 0
    if u < T0:
        return 0, 0.0, 1.0, 0.0, 0.0, 0
    if u < T1:
        return min(4, int((u - T0) / TB) + 1), 1.0, 1.0, 0.0, 0.0, 1
    if u < T2:
        return 4, 1.0, 1.0, 0.0, 0.0, 2
    if u < T3:
        p = (u - T2) / (T3 - T2)
        return 4, 1 - p, 1 - p, 0.0, 0.0, 3
    if u < T4:
        return 4, 0.0, 0.0, ease((u - T3) / (T4 - T3)), 0.0, 3
    if u < T5:
        return 4, 0.0, 0.0, 1.0, 0.0, 4
    if u < T6:
        return 4, 0.0, 0.0, 1.0, (u - T5) / (T6 - T5), 5
    return 4, 0.0, 0.0, 1.0, 1.0, 6


assert phase(0.0)[1:] == (0.0, 1.0, 0.0, 0.0, 0) and phase(T_TOTAL)[1:] == (0.0, 1.0, 0.0, 0.0, 0)

CAPS = ["一单 4 个 query，每人从 16 个 key 挑 4 个",
        "按块跳：块里有一个挑中，整块就得算",
        "4 块全要算，跟稠密一样",
        "拼一单：只留这一单挑过的 key，收拢成一摞",
        "一单是一个专家，收来的 key 行是它的 token",
        "小工收下一段时，师傅算这一段",
        "师傅只看见连续的大块，搬行交给小工"]
assert all(len(c) <= 32 for c in CAPS), [len(c) for c in CAPS]
TITLE = "拼单：先收拢，再整块乘"


def tx(s, size=24, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


class PinDanGather(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def ph():
            return phase(tr.get_value())

        self.add(tx(TITLE, 32, WHITE, True).to_edge(UP, buff=0.3))
        for r, q in enumerate(QS):
            self.add(tx("q%d" % q, 22, GREY_B, True).move_to([GX0 - 0.35, row_y(r), 0]))
        self.add(tx("key", 20, GREY).move_to([GX0 - 0.35, GY0 + 0.45, 0]))

        cache = {}

        def lab(s, col):
            k = (s, col)
            if k not in cache:
                cache[k] = tx(s, 18, col, True)
            return cache[k].copy()

        def grid():
            nb, fr, op, cp, _, _ = ph()
            g = VGroup()
            for c in range(NK):
                inu = c in UNION
                if inu:
                    x = col_x(c) + (compact_x(UNION.index(c)) - col_x(c)) * cp
                    a = 1.0
                else:
                    x = col_x(c)
                    a = op
                if a <= 0.01:
                    continue
                g.add(lab(str(c), GREEN if (inu and cp > 0.5) else GREY_B).move_to([x, GY0 + 0.45, 0]).set_opacity(a))
                for r, q in enumerate(QS):
                    sq = Square(side_length=CS - 0.06, stroke_width=1.5)
                    if c > q:
                        sq.set_fill(GREY_D, 0.9 * a).set_stroke(GREY_D, opacity=a)
                    elif c in TOY[q]:
                        sq.set_fill(BLUE, 0.95 * a).set_stroke(BLUE, opacity=a)
                    else:
                        sq.set_fill(WHITE, 0.0).set_stroke(GREY, opacity=a)
                    g.add(sq.move_to([x, row_y(r), 0]))
            for b in range(nb):
                if fr > 0.01:
                    g.add(Rectangle(width=4 * CS, height=4 * CS, stroke_color=RED, stroke_width=4, stroke_opacity=fr)
                          .move_to([GX0 + (4 * b + 2) * CS, GY0 - 1.5 * CS, 0]))
            if cp > 0.98:
                w = len(UNION) * CS
                g.add(Rectangle(width=w, height=4 * CS, stroke_color=GREEN, stroke_width=4).move_to([GX0 + w / 2, GY0 - 1.5 * CS, 0]))
            return g

        self.add(always_redraw(grid))

        # 计数与注解（按阶段切透明度）
        cnt = {k: tx("要整块算的块：%d / 4" % k, 26, RED, True).move_to([GX0 + 8 * CS, GY0 - 4 * CS - 0.25, 0]) for k in range(1, 5)}
        for k, m in cnt.items():
            m.add_updater(lambda m, k=k: m.set_opacity(1.0 if (ph()[0] == k and ph()[1] > 0.5) else 0.0))
            m.set_opacity(0.0)
            self.add(m)
        uni = VGroup(tx("并集 8 行", 28, GREEN, True), tx("ρ ＝ 8 ÷ 4 ＝ 2", 24, GREY_B),
                     tx("白格：没挑的，乘完用成员位掩掉", 20, GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        uni.move_to([1.6, 0.85, 0], aligned_edge=LEFT)
        uni.add_updater(lambda m: m.set_opacity(clamp((ph()[3] - 0.9) * 10)))
        uni.set_opacity(0.0)
        self.add(uni)

        # 第三幕：流水（两条车道）
        LX, SW = -2.6, 1.15
        YS, YT, HH = -1.25, -2.05, 0.5
        lane_lab = VGroup(tx("小工（小核）", 22, PURPLE, True).move_to([LX - 1.35, YS, 0]),
                          tx("师傅（主核）", 22, BLUE, True).move_to([LX - 1.35, YT, 0]))
        lane_lab.add_updater(lambda m: m.set_opacity(1.0 if ph()[4] > 0 else 0.0))
        lane_lab.set_opacity(0.0)
        self.add(lane_lab)

        def pipe():
            p = ph()[4]
            g = VGroup()
            if p <= 0:
                return g
            tnow = p * (NSEG + 1.5)
            for i in range(NSEG):
                for lane, start, col, name in ((YS, i, PURPLE, "收%d" % (i + 1)), (YT, i + 1, BLUE, "算%d" % (i + 1))):
                    f = clamp(tnow - start)
                    if f <= 0:
                        continue
                    x0 = LX + start * SW
                    g.add(Rectangle(width=max((SW - 0.08) * f, 1e-3), height=HH, stroke_width=0, fill_color=col, fill_opacity=0.85)
                          .move_to([x0 + (SW - 0.08) * f / 2, lane, 0], ))
                    if f > 0.6:
                        g.add(lab(name, WHITE).move_to([x0 + (SW - 0.08) / 2, lane, 0]))
            return g

        self.add(always_redraw(pipe))

        caps = []
        for k, s in enumerate(CAPS):
            m = tx(s, 26, YELLOW if k in (2, 4, 6) else GREY_B, k in (2, 4, 6)).move_to([0, -3.15, 0])
            m.add_updater(lambda m, k=k: m.set_opacity(1.0 if ph()[5] == k else 0.0))
            m.set_opacity(1.0 if k == 0 else 0.0)
            caps.append(m)
            self.add(m)

        self.play(tr.animate.set_value(T_TOTAL), run_time=T_TOTAL, rate_func=linear)
        self.play(tr.animate.set_value(T_TOTAL), run_time=1 / 30, rate_func=linear)   # 强制刷一帧（末帧 ＝ 第 0 帧）
        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "PinDanGather.json"), "w") as fh:
            json.dump({"pauses": PAUSES, "duration": round(self.renderer.time, 2)}, fh)
