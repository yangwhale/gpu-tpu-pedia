# -*- coding: utf-8 -*-
r"""专题九 §9.3「少重算一步；再砍一截，收集露出来」的动画 —— BracketCollapse。挂在静态图 fig-agwin 下面。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · 「括号」是一条异步通信从调度器放行（左竖线）到主核要用它（右竖线）之间那一段；里面夹的计算越多，通信越藏得住。
    加了融合核以后，括号里那几块计算没了，调度器把放行挪到紧挨着完成 —— 括号在时间里塌下去，
    师傅从「一直在算」变成「停着等」。静态图只能并排放两个终态，画不出「左竖线滑向右竖线」这个塌缩。

⛔ 数据从哪来（import 自 tools/topic08b_numbers.py，断言钉住；N 里没有的在本文件定义并注明出处）
  · N.IH_WIN_MS ＝ ((7.2, 9.0, 15.2), (0.4, 0.7, 0.7))：gate / up / down 三条专家权重收集括号里夹的主核计算（ms，各层中位），
    没开 → 开了。画面上括号宽度按这组数（同一个毫秒刻度 SC）。
  · EXPOSE ＝ (3.66, 3.81, 3.9)：开了以后主核停在三条收集完成点上的毫秒数 —— 课件 shot-agwin 图注同数；
    落在 N.IH_EXPOSE_EACH ＝ (3.6, 3.9) 之内（断言）。画面写 3.7 / 3.8 / 3.9 ms。
  · 每条收集在小工车道上的长度 L ＝ 开了以后括号里的计算 ＋ 露出 ＝ 4.06 / 4.51 / 4.6 ms（本课推导），
    即课件截图图注里「每条约 4 毫秒，两边一样长」；两幕用同一个 L（断言 3.9 < L < 4.7）。
  · N.IH_COMPUTE_SAVED ＝ 190、N.IH_CTL / N.IH_S ＝ 4.351 / 4.692、N.IH_PCT ＝ 7.8：只上计数，不进几何。
  · 整步等待的合计不上画面：它和每条 3.7–3.9 ms 一凑就能反推层数（公开课去标识）。

⛔ 刻意没画 / 示意
  · 三块蓝色「变短」的比例是示意（SHRINK ＝ 0.7）：整步主核少算约 190 ms 是各层合计，这一层少多少画面不给数。
    取 0.7 是为了让这一段两幕总长「不变短」（断言），跟「一步反而慢」同方向；收缩过程中不显示括号里的毫秒数，
    只在两个终态显示实测的 7.2 / 9.0 / 15.2 和 0.4 / 0.7 / 0.7。
  · 括号前后那几段蓝（开头、两条收集之间、结尾）长度示意；三条收集画成一条接一条，实际是否交叠不展开。
  · 不画延迟估计调大一倍那一格（342 只救回 63），不画融合核本身是什么结构，不写开关名、层数、芯片数。
  · 「调度器估的延迟和计算对不上」是课件里标了推断的原因，画面只写现象：「放行却排到紧挨着完成」。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-agwin.py BracketCollapse WebPages/media/topic08b-agwin.mp4
   两幕和结尾各停一下，停顿时刻写进 steps/BracketCollapse.json（课件挂 data-pauses）。

⭐ 写法：三个 ValueTracker —— u ＝ 第一幕的游标（0→1 画出来），s ＝ 计算变短（0→1），p ＝ 放行往后挪（0→1）。
   画面全是 (u, s, p) 的纯函数；收尾三个一起拨回 0（倒放），末帧按定义就是第 0 帧。字幕用 say() 换，结尾摆回空字幕。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Line, Text, ValueTracker, always_redraw,
                   FadeIn, FadeOut, WHITE, GREY, GREY_B, BLUE, RED, YELLOW, ORANGE,
                   UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ════════════════════════════════════════════════════════════════
# ① 数据
# ════════════════════════════════════════════════════════════════
assert N.IH_WIN_MS == ((7.2, 9.0, 15.2), (0.4, 0.7, 0.7))
assert N.IH_EXPOSE_EACH == (3.6, 3.9)
assert N.IH_COMPUTE_SAVED == 190
assert (N.IH_CTL, N.IH_S) == (4.351, 4.692) and "%.1f" % N.IH_PCT == "7.8"
W1, W2 = N.IH_WIN_MS
EXPOSE = (3.66, 3.81, 3.9)                     # 课件 shot-agwin 图注同数
assert all(N.IH_EXPOSE_EACH[0] <= e <= N.IH_EXPOSE_EACH[1] for e in EXPOSE)
L = tuple(w + e for w, e in zip(W2, EXPOSE))   # 每条收集在小工车道上的长度（ms），两幕一样
assert all(3.9 < x < 4.7 for x in L), L
assert all(x < w for x, w in zip(L, W1))      # 没开时：收集比括号短，藏得住
NB = 3
SHRINK = 0.7                                   # 示意：蓝色变短的比例
LEAD, GAP, TAIL = 2.0, 1.5, 2.0                # 示意：开头、两条之间、结尾的蓝（ms）
assert all(SHRINK * w > x for w, x in zip(W1, L))   # 只变短、还没挪放行时，收集仍藏得住（「挪」才是露出来的原因）
TOT1 = LEAD + sum(W1) + GAP * 2 + TAIL
TOT2 = LEAD + SHRINK * sum(W1) + sum(EXPOSE) + GAP * 2 + TAIL
assert TOT2 >= TOT1, (TOT1, TOT2)              # 这一段不变短（跟一步反而慢同方向）

TITLE = "省了计算，收集露出来"
ACT = ["没开融合核", "开了融合核"]
CAPS = [
    "放行以后，师傅还有 7 到 15 毫秒的活",
    "收集藏在底下",
    "计算少了，放行却排到紧挨着完成",
    "小工跑腿一样长，师傅每条停下来等将近 4 毫秒",
    "省了计算，先看原来被它盖住的通信露没露出来",
]
SAVED = "师傅少算约 %d 毫秒（整步）" % N.IH_COMPUTE_SAVED
COUNTERS = [
    "各层都这样，加起来比省的还多",
    "一步　%.3f → %.3f 秒（+%.1f%%）" % (N.IH_CTL, N.IH_S, N.IH_PCT),
]
LANE_TC, LANE_SC = "师傅", "小工"
REL, DONE = "放行", "完成"
NAMES = ("gate", "up", "down")
LAB1 = ["夹 %.1f ms" % w for w in W1]
LAB2 = ["夹 %.1f ms" % w for w in W2]
LABR = ["等 %.1f ms" % e for e in EXPOSE]
assert LABR == ["等 3.7 ms", "等 3.8 ms", "等 3.9 ms"]
LEGEND = "蓝＝师傅在算　橙＝小工跑腿　红＝师傅停着等"
for s in CAPS + [SAVED] + COUNTERS:            # 每句 ≤ 22 字（半角字符按半个字算）
    assert sum(0.5 if ord(ch) < 128 else 1 for ch in s) <= 22, s

# ── 几何
X0 = -5.0
SC = 10.3 / TOT2                               # 每毫秒多少单位
Y_TC, H_TC, Y_SC, H_SC = 1.05, 0.55, 0.25, 0.40
Y_REL = Y_TC + H_TC / 2 + 0.28
Y_NAME = Y_SC - H_SC / 2 - 0.3
Y_L1, Y_L2 = -0.75, -1.15


def lerp(a, b, m):
    return a + (b - a) * m


def geom(s, p):
    """返回每条收集的 (蓝块起点, 蓝块终点, 放行, 收集结束, 完成) 和整段终点，单位 ms（相对 X0）。"""
    t = LEAD
    out = []
    for i in range(NB):
        b = W1[i] * lerp(1.0, SHRINK, s)
        c0, c1 = t, t + b
        rel = lerp(c0, c1 - W2[i], p)
        ge = rel + L[i]
        done = max(c1, ge)
        out.append((c0, c1, rel, ge, done))
        t = done + (GAP if i < NB - 1 else 0.0)
    return out, t + TAIL


def X(ms):
    return X0 + ms * SC


def rect(a, b, y, h, col, op=0.9):
    if b - a < 1e-4:
        return None
    return Rectangle(width=b - a, height=h, stroke_width=0, fill_color=col, fill_opacity=op).move_to([(a + b) / 2, y, 0])


def tx(s, size=24, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def dump_steps(name, marks, t):
    os.makedirs(STEPS_DIR, exist_ok=True)
    with open(os.path.join(STEPS_DIR, name + ".json"), "w") as fh:
        json.dump({"pauses": marks, "duration": round(t, 2)}, fh)


# ════════════════════════════════════════════════════════════════
# ② 画面
# ════════════════════════════════════════════════════════════════
class BracketCollapse(Scene):
    def construct(self):
        marks = []
        u = ValueTracker(0.0)
        st = ValueTracker(0.0)
        pt = ValueTracker(0.0)

        def state():
            g, end = geom(st.get_value(), pt.get_value())
            c = X(lerp(0.0, end, u.get_value()))
            return g, end, c

        def scene_at():
            g, end, c = state()
            grp = []
            # 师傅：蓝＝算，红＝等
            grp.append(rect(X(0.0), min(X(end), c), Y_TC, H_TC - 0.08, BLUE, 0.85))   # 一整条蓝，红叠在上面（免得拼缝）
            for i in range(NB):
                c0, c1, rel, ge, done = g[i]
                grp.append(rect(X(c1), min(X(done), c), Y_TC, H_TC - 0.08, RED, 1.0))
            # 小工：三条收集（长度两幕一样）
            for i in range(NB):
                _c0, _c1, rel, ge, done = g[i]
                grp.append(rect(X(rel), min(X(ge), c), Y_SC, H_SC - 0.08, ORANGE, 0.9))
                for x in (rel, done):                       # 括号：放行、完成两条竖线
                    if c >= X(x) - 1e-6:
                        grp.append(Line([X(x), Y_SC - H_SC / 2 - 0.08, 0], [X(x), Y_TC + H_TC / 2 + 0.08, 0],
                                        color=WHITE, stroke_width=4))
            # 游标（只在第一幕走的时候看得见）
            if u.get_value() < 1.0 - 1e-6:
                grp.append(Line([c, Y_SC - H_SC / 2 - 0.12, 0], [c, Y_TC + H_TC / 2 + 0.12, 0], color=YELLOW, stroke_width=5))
            return VGroup(*[x for x in grp if x is not None])

        title = tx(TITLE, 36, WHITE, True).to_edge(UP, buff=0.3)
        static = VGroup(
            tx(LANE_TC, 24, BLUE).move_to([X0 - 0.75, Y_TC, 0]),
            tx(LANE_SC, 24, ORANGE).move_to([X0 - 0.75, Y_SC, 0]),
            tx(LEGEND, 22, GREY_B).move_to([0, -1.75, 0]),
        )
        body = always_redraw(scene_at)

        # 跟着几何走的字（建一次，只挪位置、切透明度）
        rel_t, done_t = tx(REL, 22, WHITE), tx(DONE, 22, WHITE)
        names = [tx(n, 22, GREY_B) for n in NAMES]
        l1 = [tx(s, 24, BLUE) for s in LAB1]
        l2 = [tx(s, 24, BLUE) for s in LAB2]
        lr = [tx(s, 24, RED, True) for s in LABR]

        def place(_m=None):
            g, end, c = state()
            s, p = st.get_value(), pt.get_value()
            c0, c1, rel, ge, done = g[0]
            rel_t.move_to([X(rel), Y_REL, 0]).set_opacity(1.0 if c >= X(rel) else 0.0)
            done_t.move_to([X(done), Y_REL, 0]).set_opacity(1.0 if c >= X(done) else 0.0)
            if done_t.get_left()[0] < rel_t.get_right()[0] + 0.08:      # 括号塌了：两个字分开摆
                rel_t.next_to(done_t, LEFT, buff=0.08)
            for i in range(NB):
                c0, c1, rel, ge, done = g[i]
                seen = 1.0 if c >= X(done) else 0.0
                names[i].move_to([X((c0 + done) / 2), Y_NAME, 0]).set_opacity(seen)
                l1[i].move_to([X((rel + done) / 2), Y_L1, 0]).set_opacity(seen * max(0.0, 1.0 - 3 * s))
                l2[i].move_to([X(rel) - l2[i].width / 2 + 0.25, Y_L1, 0]).set_opacity(max(0.0, (p - 0.7) / 0.3))
                lr[i].move_to([X((c1 + done) / 2), Y_L2, 0]).set_opacity(max(0.0, (p - 0.7) / 0.3))

        movers = VGroup(rel_t, done_t, *names, *l1, *l2, *lr)
        place()
        movers.add_updater(place)

        act = tx(ACT[0], 26, YELLOW, True).move_to([X0 - 0.75, 2.05, 0], aligned_edge=LEFT)
        cap = tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.2)
        self.add(title, static, body, movers, cap, act)

        def new_cap(s, color=GREY_B):
            return tx(s, 26, color).next_to(title, DOWN, buff=0.2)

        def say(s, color=GREY_B, extra=()):
            nonlocal cap
            n = new_cap(s, color)
            self.play(FadeOut(cap), FadeIn(n), *extra, run_time=0.35)
            cap = n

        def hold(t):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        def act_to(k):
            nonlocal act
            n = tx(ACT[k], 26, YELLOW, True).move_to([X0 - 0.75, 2.05, 0], aligned_edge=LEFT)
            out = (FadeOut(act), FadeIn(n))
            act = n
            return out

        # ── 第一幕：没开融合核，游标画出来
        self.wait(0.4)
        _g, end0 = geom(0.0, 0.0)
        u_mid = (_g[1][4]) / end0
        say(CAPS[0])
        self.play(u.animate.set_value(u_mid), run_time=2.4, rate_func=linear)
        say(CAPS[1])
        self.play(u.animate.set_value(1.0), run_time=1.9, rate_func=linear)
        hold(2.3)

        # ── 第二幕：开了融合核 —— 先变短，再挪放行
        saved = tx(SAVED, 24, YELLOW).move_to([0, -2.45, 0])
        say(CAPS[2], extra=act_to(1))
        self.play(st.animate.set_value(1.0), FadeIn(saved), run_time=1.6, rate_func=linear)
        self.play(pt.animate.set_value(1.0), run_time=2.6, rate_func=linear)
        counters = VGroup(*[tx(s, 24, YELLOW) for s in COUNTERS]).arrange(RIGHT, buff=0.9).move_to([0, -3.05, 0])
        say(CAPS[3], extra=(FadeIn(counters),))
        hold(2.8)
        say(CAPS[4], YELLOW)
        hold(2.2)

        # ── 收尾：倒放回第 0 帧
        self.play(FadeOut(counters), FadeOut(saved), FadeOut(cap), *act_to(0),
                  u.animate.set_value(0.0), st.animate.set_value(0.0), pt.animate.set_value(0.0),
                  run_time=1.0, rate_func=linear)
        self.remove(counters, saved, cap)
        self.add(new_cap(" "))
        self.play(u.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)   # 推一下 updater
        self.wait(0.5)
        dump_steps("BracketCollapse", marks, self.renderer.time)


if __name__ == "__main__":
    print("L", L, "TOT1 %.2f TOT2 %.2f SC %.3f" % (TOT1, TOT2, SC))
