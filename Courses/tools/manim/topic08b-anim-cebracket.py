# -*- coding: utf-8 -*-
r"""专题九 §9.5「回头再查那颗核：调度器凭什么把括号排空」的动画 —— EstimateBracket。挂在静态图 fig-cebracket 上面。
   2026-11-22 第二版：第三、四幕旁白改成机理（候选与搭档），加第五幕「门槛提到 3 ms，完成点回到用数据的地方」。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · 同一条专家权重收集的括号，经历「填满 → 塌掉 → 调准了还是塌 → 恢复」四个状态，只差一件事：融合核报不报耗时。
    静态图 fig-cebracket 只能把四个终态摞成四行；动起来才看得见「右竖线滑向左竖线、红色冒出来、再滑回去」，
    而且看得见触发它的那一下只是融合核头上那块小牌子换了字。

⛔ 数据从哪来（import 自 tools/topic08b_numbers.py，断言钉住）
  · N.CE_WIN[k][0][1]：拼好的 gate/up 那条收集，括号里调度器排进的计算（ms，调度器自己的估计）：
    对照 10.4，融合核报耗时 0.48，报的耗时按实测调准 0.55，不报耗时 11.9。画面上括号宽度按这组数（同一个毫秒刻度）。
  · N.CE_AG_EXPOSED ＝ (41.5, 290.5, 14)：上卡实测这条收集每步露出（ms）；N.CE[k][0]：上卡步时（s）。只上计数，不进几何。
  · N.CE_IHC_EST_MS ＝ (0.068, 0.178)：融合核四颗各报的耗时换算成时长 —— 牌子上写「报：约 0.1 ms」。

⛔ 刻意没画 / 示意
  · 小工车道上那条收集的长度 L ＝ 8.2 ms 取的是实测小核上的执行时间（N.CE_RATIO_ROWS 里那一行），只用来定「括号塌了以后师傅等多久」的示意长度；
    调度导出里的窗长（约 26 ms，放行到完成的距离）不画 —— 画面只讲「括号里排进了多少计算」这一件事。
  · 括号外那几段蓝（融合核后面、括号之后）长度示意；括号里的蓝画成一整段，不分哪几颗核。
  · 第五幕的括号 11.9 ms 取门槛扫描里「照实报 ＋ 门槛 3 ms ＋ 延迟估计 ×0.32」那格的调度器估计（64 片拓扑 11.91）；前一拍（门槛默认）只有客观判据预测；后一拍有上卡读数（同一台机器 3.907 → 3.868 s，−1.0%，逐位）。
  · 绿色的「矩阵乘」块是这批权重真正被用到的地方，位置示意；「完成」标签后面括号里写完成点钉在谁旁边（编译产物里的调度分组）：
    默认门槛下照实报，搭档仍是融合核；门槛 3 ms 时专家权重收集没有搭档，完成点按默认放到最晚、贴着矩阵乘。不写开关名、层数、芯片数。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-cebracket.py EstimateBracket WebPages/media/topic08b-cebracket.mp4
   每一幕落地停一下，停顿时刻写进 steps/EstimateBracket.json（课件挂 data-pauses）。

⭐ 写法：一个 ValueTracker m ＝ 括号里排进的计算（ms），画面全是 m 的纯函数；四幕就是把 m 拨到四个实测值。
   收尾把 m 拨回第一幕的值、字幕摆回空，末帧按定义就是第 0 帧。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, RoundedRectangle, Line, Text, ValueTracker, always_redraw,
                   FadeIn, FadeOut, WHITE, GREY_B, BLUE, RED, GREEN, YELLOW, ORANGE, PURPLE_B,
                   UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ════════════════════════════════════════════════════════════════
# ① 数据
# ════════════════════════════════════════════════════════════════
KEYS = ("ctl", "fk", "fkm", "fk0")
M = [N.CE_WIN[k][0][1] for k in KEYS] + [0.62, 11.9]  # 第五幕：照实报 ＋ 默认门槛 0.62 → 门槛 3 ms 11.91（64 片拓扑扫描里调度器的估计，取到一位小数）
assert M == [10.4, 0.48, 0.55, 11.9, 0.62, 11.9], M
L = [m for g, _n, _e, m in N.CE_RATIO_ROWS if g == "ag"][0]   # 8.2：小核上这条收集实际跑多久（示意长度用）
assert L == 8.2
assert M[0] > L and M[3] > L and M[1] < L and M[2] < L       # 填满的两幕藏得住，塌掉的两幕要等
assert N.CE_AG_EXPOSED == (41.5, 290.5, 14)
assert "%.3f" % N.CE["fk"][0] == "4.522" and "%.3f" % N.CE["fk0"][0] == "3.931" and "%.3f" % N.CE_CTL == "4.088"
assert N.CE_IHC_EST_MS == (0.068, 0.178)
HEAD = 1.4          # 示意：层首那一块（原写法的几块融合 ／ 融合核）
U0 = 10.4           # 示意：矩阵乘上游的计算至少这么多（括号塌了以后，它们被挪到等待之后、矩阵乘之前）
G = 2.4             # 示意：矩阵乘那一块
CTOT = 14.6         # 示意：这一段主核的计算合计（含矩阵乘）
assert CTOT >= max(M) + G

TITLE = "核报不报耗时，括号里排不排计算"
ACT = ["对照：没有融合核", "换上融合核：核报耗时", "把报的耗时按实测调准", "融合核：不报耗时", "所有核照实报：门槛默认", "所有核照实报：门槛 3 ms"]
TAG = [None, "报：约 0.1 ms", "报：按实测调准", "不报：按几乎 0 算", "报：照实　门槛：默认（极低）", "报：照实　门槛 3 ms：挡在外面"]
CAPS = [
    "调度器排的括号：放行到完成，中间填计算",
    "换上更快的融合核，它报了自己的耗时",
    "它成了候选、本轮最大：完成点钉在它旁边",
    "调准了还是候选、还是最大：照样钉在它旁边",
    "不报就进不了候选：完成点回到矩阵乘旁边",
    "全照实报、门槛不动：融合核又成搭档，又塌了",
    "门槛提到 3 ms：完成点回到用数据的地方",
    "耗时估计要么准，要么不填；以上卡为准",
]
INSIDE = ["括号里排进 %.1f ms" % M[0], "只排进 %.2f ms" % M[1], "只排进 %.2f ms" % M[2], "括号里排进 %.1f ms" % M[3],
          "只排进 %.2f ms" % M[4], "括号里排进 %.1f ms" % M[5]]
assert INSIDE == ["括号里排进 10.4 ms", "只排进 0.48 ms", "只排进 0.55 ms", "括号里排进 11.9 ms", "只排进 0.62 ms", "括号里排进 11.9 ms"]
DONE_LAB = ["完成", "完成（钉在融合核旁）", "完成（钉在矩阵乘旁）", "完成（没有搭档，贴着矩阵乘）"]
COUNT = [
    "这条收集每步露出 %.1f ms　一步 %.3f 秒" % (N.CE_AG_EXPOSED[0], N.CE_CTL),
    "这条收集每步露出 %.1f ms　一步 %.3f 秒（+%.1f%%）" % (N.CE_AG_EXPOSED[1], N.CE["fk"][0], N.CE_PCT["fk"]),
    "一步 %.3f 秒（+%.1f%%），一点没变" % (N.CE["fkm"][0], N.CE_PCT["fkm"]),
    "这条收集每步露出不到 %d ms　一步 %.3f 秒（−%.1f%%）" % (N.CE_AG_EXPOSED[2], N.CE["fk0"][0], -N.CE_PCT["fk0"]),
    "客观判据预测：整步比现在的推荐配置还差",
    "上卡 %.3f → %.3f 秒（−%.1f%%），逐位相同" % (N.CE_TH_ONCARD_CTL, N.CE_TH_ONCARD, -N.CE_TH_PCT),
]
assert COUNT[5] == "上卡 3.907 → 3.868 秒（−1.0%），逐位相同", COUNT[5]
assert N.CE_TH64[0] > N.CE_TH_V4[0] and N.CE_TH64[7] == min(N.CE_TH64)
assert COUNT[3].endswith("3.931 秒（−3.8%）"), COUNT[3]
LEGEND = "蓝＝师傅在算　紫＝融合核　绿＝用这批权重的矩阵乘　橙＝小工收权重　红＝停着等"
NOTE = "括号里的毫秒是调度器自己的估计；露出和步时是上卡实测；块长示意"
LANE_TC, LANE_SC = "师傅", "小工"
REL, DONE = "放行", "完成"
for s in CAPS:                                  # 每句 ≤ 22 字（半角字符按半个字算）
    assert sum(0.5 if ord(ch) < 128 else 1 for ch in s) <= 22, s

# ── 几何
X0 = -5.2
END_MAX = HEAD + L + (CTOT - min(M)) + 0.2
SC = 10.6 / END_MAX
Y_TC, H_TC, Y_SC, H_SC = 0.75, 0.55, -0.05, 0.40
Y_REL = Y_TC + H_TC / 2 + 0.30


def X(ms):
    return X0 + ms * SC


def geom(m):
    """括号里排进 m ms 计算时：(放行, 完成, 收集结束, 红色终点, 矩阵乘起点, 整段终点)，单位 ms。"""
    rel = HEAD
    done = HEAD + m
    ge = HEAD + L
    red_end = max(done, ge)
    g0 = red_end + (max(m, U0) - m)        # 矩阵乘前面还要先算完它的上游
    end = red_end + (CTOT - m)
    return rel, done, ge, red_end, g0, end


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
class EstimateBracket(Scene):
    def construct(self):
        marks = []
        mt = ValueTracker(M[0])
        kt = ValueTracker(0.0)        # 层首那一块：0 ＝ 原写法（蓝），1 ＝ 融合核（紫）

        def body():
            rel, done, ge, red_end, g0, end = geom(mt.get_value())
            k = kt.get_value()
            grp = []
            head_col = BLUE if k < 0.5 else PURPLE_B
            grp.append(rect(X(0), X(HEAD) - 0.03, Y_TC, H_TC - 0.08, head_col, 0.95))
            grp.append(rect(X(rel), X(done), Y_TC, H_TC - 0.08, BLUE, 0.85))
            grp.append(rect(X(done), X(red_end), Y_TC, H_TC - 0.08, RED, 1.0))
            grp.append(rect(X(red_end), X(end), Y_TC, H_TC - 0.08, BLUE, 0.85))
            grp.append(rect(X(g0) + 0.02, X(g0 + G) - 0.02, Y_TC, H_TC - 0.08, GREEN, 1.0))
            grp.append(rect(X(rel), X(ge), Y_SC, H_SC - 0.08, ORANGE, 0.9))
            for x in (rel, done):
                grp.append(Line([X(x), Y_SC - H_SC / 2 - 0.08, 0], [X(x), Y_TC + H_TC / 2 + 0.08, 0], color=WHITE, stroke_width=5))
            return VGroup(*[g for g in grp if g is not None])

        title = tx(TITLE, 36, WHITE, True).to_edge(UP, buff=0.3)
        static = VGroup(
            tx(LANE_TC, 24, BLUE).move_to([X0 - 0.75, Y_TC, 0]),
            tx(LANE_SC, 24, ORANGE).move_to([X0 - 0.75, Y_SC, 0]),
            tx(LEGEND, 21, GREY_B).move_to([0, -1.55, 0]),
            tx(NOTE, 20, GREY_B).move_to([0, -3.45, 0]),
        )
        rel_t = tx(REL, 22, WHITE)
        pin = ValueTracker(0.0)       # 「完成」后面括号里写钉在谁旁边：对应 DONE_LAB
        done_ts = [tx(s, 22, (WHITE, RED, GREEN, GREEN)[i]) for i, s in enumerate(DONE_LAB)]
        gm_t = tx("矩阵乘", 20, GREEN, True)

        def place(_m=None):
            rel, done, ge, red_end, g0, end = geom(mt.get_value())
            rel_t.move_to([X(rel), Y_REL, 0])
            k = int(round(pin.get_value()))
            for i, d in enumerate(done_ts):
                d.move_to([X(done), Y_REL, 0], aligned_edge=LEFT).shift(LEFT * 0.25)
                if d.get_left()[0] < rel_t.get_right()[0] + 0.1:     # 括号塌了：两个字分开摆
                    d.next_to(rel_t, RIGHT, buff=0.1)
                d.set_opacity(1.0 if i == k else 0.0)
            gm_t.move_to([X(g0 + G / 2), Y_TC - H_TC / 2 - 0.22, 0])

        movers = VGroup(rel_t, gm_t, *done_ts)
        place()
        movers.add_updater(place)
        self.add(title, static, always_redraw(body), movers)

        state = {"cap": tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.2),
                 "act": tx(ACT[0], 26, YELLOW, True).move_to([X0 - 1.25, 2.3, 0], aligned_edge=LEFT),
                 "ins": tx(INSIDE[0], 24, BLUE).move_to([X(HEAD + M[0] / 2), -0.75, 0]),
                 "cnt": tx(COUNT[0], 24, YELLOW).move_to([0, -2.35, 0]),
                 "tag": None}
        self.add(state["cap"], state["act"])

        def swap(key, new):
            old = state[key]
            state[key] = new
            out = []
            if old is not None:
                out.append(FadeOut(old))
            if new is not None:
                out.append(FadeIn(new))
            return out

        def say(s, color=GREY_B, extra=()):
            self.play(*swap("cap", tx(s, 26, color).next_to(title, DOWN, buff=0.2)), *extra, run_time=0.4)

        def act(i):
            return swap("act", tx(ACT[i], 26, YELLOW, True).move_to([X0 - 1.25, 2.3, 0], aligned_edge=LEFT))

        def tag(i):
            if TAG[i] is None:
                return swap("tag", None)
            col = GREEN if i in (3, 5) else RED
            box = RoundedRectangle(corner_radius=0.08, width=0.1, height=0.42, stroke_color=col, stroke_width=3, fill_opacity=0)
            t = tx(TAG[i], 22, col, True)
            box.stretch_to_fit_width(t.width + 0.3)
            g = VGroup(box, t)
            t.move_to(box)
            g.move_to([X(0), 1.8, 0], aligned_edge=LEFT)
            return swap("tag", g)

        def inside(i):
            m = M[i]
            col = BLUE if m > L else RED
            x = X(HEAD + m / 2) if m > L else X(HEAD + m) + 2.0
            return swap("ins", tx(INSIDE[i], 24, col, True).move_to([x, -0.75, 0]))

        def count(i):
            col = YELLOW if i in (0, 2, 5) else (RED if i in (1, 4) else GREEN)
            return swap("cnt", tx(COUNT[i], 24, col).move_to([0, -2.35, 0]))

        def hold(t):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        # ── 第一幕：对照
        self.wait(0.4)
        say(CAPS[0], extra=inside(0) + count(0))
        hold(2.6)

        # ── 第二幕：换上融合核、核报耗时 → 它成了候选，括号塌掉
        say(CAPS[1], extra=act(1) + tag(1) + swap("ins", None) + swap("cnt", None))
        self.play(kt.animate.set_value(1.0), run_time=0.6, rate_func=linear)
        self.wait(0.6)
        say(CAPS[2])
        self.play(mt.animate.set_value(M[1]), pin.animate.set_value(1.0), run_time=2.2, rate_func=linear)
        self.play(*(inside(1) + count(1)), run_time=0.4)
        hold(2.8)

        # ── 第三幕：按实测调准，还是候选、还是最大
        say(CAPS[3], extra=act(2) + tag(2) + count(2))
        self.play(mt.animate.set_value(M[2]), run_time=0.6, rate_func=linear)
        self.play(*inside(2), run_time=0.3)
        hold(2.6)

        # ── 第四幕：不报耗时 → 进不了候选，完成点回到矩阵乘旁边
        say(CAPS[4], extra=act(3) + tag(3) + swap("ins", None) + swap("cnt", None))
        self.play(mt.animate.set_value(M[3]), pin.animate.set_value(2.0), run_time=2.2, rate_func=linear)
        self.play(*(inside(3) + count(3)), run_time=0.4)
        hold(2.8)

        # ── 第五幕：所有核照实报。门槛不动又塌；门槛提到 3 ms，完成点回到用数据的地方
        say(CAPS[5], extra=act(4) + tag(4) + swap("ins", None) + swap("cnt", None))
        self.play(mt.animate.set_value(M[4]), pin.animate.set_value(1.0), run_time=2.0, rate_func=linear)
        self.play(*(inside(4) + count(4)), run_time=0.4)
        hold(2.4)
        say(CAPS[6], extra=act(5) + tag(5) + swap("ins", None) + swap("cnt", None))
        self.play(mt.animate.set_value(M[5]), pin.animate.set_value(3.0), run_time=2.2, rate_func=linear)
        self.play(*(inside(5) + count(5)), run_time=0.4)
        hold(2.8)
        say(CAPS[7], YELLOW)
        hold(2.4)

        # ── 收尾：拨回第一幕的第 0 帧
        self.play(*(swap("cap", tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.2)) + act(0) + tag(0) + swap("ins", None) + swap("cnt", None)),
                  mt.animate.set_value(M[0]), kt.animate.set_value(0.0), pin.animate.set_value(0.0), run_time=1.0, rate_func=linear)
        self.play(mt.animate.set_value(M[0]), run_time=1 / 30, rate_func=linear)   # 推一下 updater
        self.wait(0.5)
        dump_steps("EstimateBracket", marks, self.renderer.time)


if __name__ == "__main__":
    print("END_MAX %.2f SC %.3f" % (END_MAX, SC))
