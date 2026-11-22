# -*- coding: utf-8 -*-
r"""专题九 §6.4「缩不了的，就藏」的动画：两类活挤一条队 → 分两队 —— TwoQueues。
放在静态图 fig-twoqueue 下面；画面里不写课件节号。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  先来先做的队列里，「谁排在前面、后面的就得等」是一个随时间推进的排队过程：
  token 那趟腿是什么时候进队的、前面还压着几块专家权重收集、师傅走到要用它的那一刻它到没到 ——
  静态图 fig-twoqueue 画得出「一条挤满、一条闲着」和「分两队」两种排布，画不出等待从哪来、到哪去。
  游标一格一格往前走，红色在哪一刻长出来、分队以后哪一段红色消失，只有动起来才看得见。

⛔ 数据从哪来
  · 忙时全部 import 自 tools/topic08b_numbers.py 并断言：
      N.FWD_SC0_BUSY ＝ 18.57（小核 0，前向每层中位数）、N.FWD_TC_COMPUTE ＝ 18.65（主核计算）、
      N.FWD_SC1_BUSY ＝ 0.59（小核 1）；整步 N.CHAIN_SC_MS ＝ 1389、N.CHAIN_EXPOSED_MS ＝ 381；
      N.SPLIT_CTL / N.SPLIT_S ＝ 4.415 / 4.280（N.SPLIT_PCT ＝ 3.1）。
  · 小核 0 忙时的三个分项（专家权重收集 12.18、token 4.20、其它 2.18）不在 numbers 里，
    在本文件定义：课件 §6.4「出处与口径」正文同数。
  · 排队是当场真排的（下面 simulate()）：每条小工车道一个先来先做的队列，师傅按依赖等结果；
    两幕用同一组块、同一组进队时刻，只改「哪块进哪条队」。红色（师傅等）是模拟出来的，不是手摆的。
    专家权重收集按 3 块等分、token 收发按「发」「收」2 块等分，主核计算按 5 段拆 —— 这些切法和进队时刻是示意。

⛔ 刻意没画
  · 不画「同一条通信让两颗小核一起收」：课件另有结论，那样反而慢 1.2%（卡在链路带宽）。
    这里分队是让不同的两类活各排各的，画面里的每一块都只在一条车道上。
  · 改之后每层的忙时没有逐层量过（课件 §6.4 示意注）：第二幕不标忙时数字，块长按改之前的分项画。
  · 一层的等待长短是示意（模拟出来一层少等约一半），不代表一步快多少；一步快多少只引实测 4.415 → 4.280 秒。
  · 不画反向、不画层与层之间的预取细节、不写真实开关名。

🎨 配色：蓝＝师傅在算，红＝师傅停着等，橙＝专家权重收集，紫＝token 收发（跟静态图 fig-lanes / fig-twoqueue 的 token 紫 #9334e6 同色相，黑底上提亮一档），灰＝其它；黄＝时间游标和依赖线。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-queue.py TwoQueues WebPages/media/topic08b-queue.mp4
   两幕各自落地的停顿时刻写进 steps/TwoQueues.json（课件挂 data-pauses）。

⭐ 写法：时钟式。唯一的 ValueTracker ＝ 片内时间 u；画面全是 state(u) 的纯函数。
   复位只定义在 state 里：第二幕结束后淡回「第一幕、时间 0」那一帧 —— 末帧按定义就是第 0 帧。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, RoundedRectangle, Line, DashedLine, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, RED, YELLOW, ORANGE, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── 数据（当场取、当场断言）──────────────────────────────────────────
SC0_WEIGHT, SC0_TOKEN, SC0_OTHER = 12.18, 4.20, 2.18      # 课件 §6.4 出处正文同数（小核 0 忙时分项，ms）
assert abs(SC0_WEIGHT + SC0_TOKEN + SC0_OTHER - N.FWD_SC0_BUSY) < 0.02
assert (N.FWD_SC0_BUSY, N.FWD_TC_COMPUTE, N.FWD_SC1_BUSY) == (18.57, 18.65, 0.59)
assert (N.CHAIN_SC_MS, N.CHAIN_EXPOSED_MS) == (1389, 381)
assert (N.SPLIT_CTL, N.SPLIT_S) == (4.415, 4.280) and "%.1f" % N.SPLIT_PCT == "3.1"

W = SC0_WEIGHT / 3            # 专家权重收集：3 块
TK = SC0_TOKEN / 2            # token：发、收 2 块
# 块：id → (类别, 时长, 进队时刻；None ＝ 师傅算完 E3 才发)。进队时刻是示意。
BLOCKS = {
    "W1": ("w", W, 0.0), "W2": ("w", W, 0.0), "W3": ("w", W, 5.0),
    "T1": ("t", TK, 4.0), "O": ("o", SC0_OTHER, 4.0), "T2": ("t", TK, None),
    "S1": ("o", N.FWD_SC1_BUSY, 12.0),        # 小工乙原本那一丝活
}
SEQ = ["W1", "W2", "T1", "O", "W3", "S1", "T2"]   # 同一时刻进队的先后
# 师傅：(段名, 时长, 开算前要等哪块)。A ＝ 注意力等；E1–E3 用专家；F 要 token 收回来。
MASTER = [("A", 4.0, None), ("E1", 3.5, "T1"), ("E2", 3.5, "W2"), ("E3", 3.5, "W3"), ("F", None, "T2")]
MASTER[-1] = ("F", N.FWD_TC_COMPUTE - 4.0 - 3 * 3.5, "T2")
assert abs(sum(d for _, d, _ in MASTER) - N.FWD_TC_COMPUTE) < 1e-9
NEEDS_TOO = {"E1": "W1"}      # E1 也要 W1（两幕里都早到了，不画线）

LANE_BEFORE = {"W1": 0, "W2": 0, "W3": 0, "T1": 0, "O": 0, "T2": 0, "S1": 1}   # 0 ＝ 小工甲，1 ＝ 小工乙
LANE_AFTER = {"W1": 1, "W2": 1, "W3": 1, "T1": 0, "O": 0, "T2": 0, "S1": 1}    # token 钉小工甲，专家权重挪小工乙


def simulate(lane_of):
    """先来先做的排队 ＋ 师傅按依赖等。返回 (块 id → (车道, 进队, 开始, 结束), 师傅算段, 师傅等段, 依赖线)。"""
    issue = {b: BLOCKS[b][2] for b in BLOCKS}
    span = {}

    def run_lanes():                         # 只排进队时刻已知的块
        free = [0.0, 0.0]
        known = sorted([b for b in SEQ if issue[b] is not None], key=lambda b: (issue[b], SEQ.index(b)))
        out = {}
        for b in known:
            ln = lane_of[b]
            s = max(free[ln], issue[b])
            out[b] = (ln, issue[b], s, s + BLOCKS[b][1])
            free[ln] = s + BLOCKS[b][1]
        return out

    t, comp, wait, deps = 0.0, [], [], []
    for name, d, need in MASTER:
        if need is not None:
            span = run_lanes()
            ready = max(span[need][3], span[NEEDS_TOO[name]][3] if name in NEEDS_TOO else 0.0)
            if ready > t + 1e-9:
                wait.append((t, ready))
            t = max(t, ready)
            deps.append((need, t))
        comp.append((name, t, t + d))
        t += d
        if name == "E3":
            issue["T2"] = t                  # 师傅算完 E3 才发 token 回收
    span = run_lanes()
    return span, comp, wait, deps


SIM = [simulate(LANE_BEFORE), simulate(LANE_AFTER)]
WAIT_MS = [sum(b - a for a, b in s[2]) for s in SIM]
END = [s[1][-1][2] for s in SIM]
for k, (sp, _, _, _) in enumerate(SIM):      # 忙时守恒：活没少
    assert abs(sum(BLOCKS[b][1] for b in sp) - (N.FWD_SC0_BUSY + N.FWD_SC1_BUSY)) < 0.02
assert abs(sum(e - s for b, (ln, i, s, e) in SIM[0][0].items() if ln == 0) - N.FWD_SC0_BUSY) < 0.02
assert abs(sum(e - s for b, (ln, i, s, e) in SIM[0][0].items() if ln == 1) - N.FWD_SC1_BUSY) < 1e-9
# 论点：第一幕 token「发」排在专家权重后面，晚到 → 师傅等；第二幕红色明显变少
assert SIM[0][0]["T1"][2] > SIM[0][0]["T1"][1] + 1.0          # 进队后干等了一截
assert SIM[1][0]["T1"][2] == SIM[1][0]["T1"][1]               # 分队后一进队就开跑
assert WAIT_MS[1] < 0.6 * WAIT_MS[0], WAIT_MS
T_MAX = 27.5
assert max(END) < T_MAX

# ── 时间表（秒）───────────────────────────────────────────────────
U_CLK = 7.0                                   # 时钟 0 → T_MAX 毫秒
A1_0 = 1.0; A1_1 = A1_0 + U_CLK               # 第一幕走钟
X1_0 = 11.6; X1_1 = 12.2                      # 第一幕淡回时间 0
M_0, M_1 = 12.2, 13.0                         # 橙块从小工甲的队挪到小工乙的队
A2_0 = 13.0; A2_1 = A2_0 + U_CLK              # 第二幕走钟
X2_0 = 23.0; X2_1 = 23.6                      # 淡回第 0 帧
T_TOTAL = 24.0
U_WAIT1 = A1_0 + U_CLK * SIM[0][2][0][0] / T_MAX              # 第一幕师傅开始等的那一刻
# 字幕：(起, 止, 句号)
CAP_T = [(0, U_WAIT1, 0), (U_WAIT1, 6.2, 1), (6.2, 9.2, 2), (9.2, X1_0, 3),
         (X1_0, 15.2, 4), (15.2, 20.4, 5), (20.4, X2_0, 6), (X2_0, T_TOTAL + 1, 0)]
PAUSES = [11.3, 22.7]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def state(u):
    """片内时间 u → 画面状态。⭐ 复位只定义在这里。
    返回 dict：layers ＝ [(第几幕, 毫秒, 透明度)…]，move ＝ 橙块挪队进度或 None，lab ＝ 忙时标签透明度，cap ＝ 字幕号。"""
    cap = [c for a, b, c in CAP_T if a <= u < b][0]
    if u < X1_0:
        return dict(layers=[(0, T_MAX * clamp((u - A1_0) / U_CLK), 1.0)], move=None, lab=1.0, cap=cap)
    if u < X1_1:
        p = (u - X1_0) / (X1_1 - X1_0)
        return dict(layers=[(0, T_MAX, 1 - p), (0, 0.0, p)], move=None, lab=1 - p, cap=cap)
    if u < M_1:
        return dict(layers=[], move=clamp((u - M_0) / (M_1 - M_0)), lab=0.0, cap=cap)
    if u < X2_0:
        return dict(layers=[(1, T_MAX * clamp((u - A2_0) / U_CLK), 1.0)], move=None, lab=0.0, cap=cap)
    p = clamp((u - X2_0) / (X2_1 - X2_0))
    return dict(layers=[(1, T_MAX, 1 - p), (0, 0.0, p)] if p < 1 else [(0, 0.0, 1.0)], move=None, lab=p, cap=cap)


assert state(0.0) == state(T_TOTAL)          # 末帧 ≡ 首帧，按定义成立

# ── 画面常数 ───────────────────────────────────────────────────────
TOKEN_PURPLE = "#A35CF0"                      # 静态图 token 紫 #9334e6 在黑底上提亮一档
COL = {"w": ORANGE, "t": TOKEN_PURPLE, "o": GREY}
LANE_Y = [1.15, -0.45, -1.95]                 # 师傅、小工甲、小工乙
LH = 0.72                                     # 车道高
X0, X1 = -2.2, 6.85                          # 时间轴
Q_RIGHT, QW, QG = -2.45, 0.36, 0.08           # 队列：最靠右那块（下一个开跑）的右边缘、块宽、间隔


LAB_X = -5.8                                 # 车道名一栏的中心


def xm(ms):
    return X0 + (X1 - X0) * ms / T_MAX


def tx(s, size=22, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def bar(x0, x1, y, color, op=1.0, h=LH):
    w = max(x1 - x0, 1e-3)
    return Rectangle(width=w, height=h, stroke_color="#000000", stroke_width=3 if w > 0.06 else 0, stroke_opacity=op,
                     fill_color=color, fill_opacity=0.85 * op).move_to([(x0 + x1) / 2, y, 0])


def queue_pos(k, ln):
    """队里第 k 个（0 ＝ 下一个开跑）的中心。"""
    return [Q_RIGHT - QW / 2 - k * (QW + QG), LANE_Y[ln + 1], 0]


def queued(act, t):
    """时刻 t 还在队里等的块（按先来先做的顺序）。t ＝ 0 时，0 时刻进队的块也算在队里。"""
    sp = SIM[act][0]
    out = [[], []]
    for b in sorted(sp, key=lambda b: (sp[b][1], SEQ.index(b))):
        ln, i, s, e = sp[b]
        if i <= t and (t < s or (t == 0 and s == 0)):
            out[ln].append(b)
    return out


def qsq(b, pos, op, hot=False):
    return Rectangle(width=QW, height=LH * 0.62, stroke_color=YELLOW, stroke_width=5 if hot else 0, stroke_opacity=op,
                     fill_color=COL[BLOCKS[b][0]], fill_opacity=0.85 * op).move_to(pos)


def draw(act, t, op):
    sp, comp, wait, deps = SIM[act]
    g = VGroup()
    if op <= 1e-3:
        return g
    hot = [need for need, tm in deps for a, b in wait if abs(tm - b) < 1e-9 and a < t < b]   # 师傅此刻在等的那一块：描黄边
    for name, a, b in comp:                  # 师傅：蓝
        if a < t:
            g.add(bar(xm(a), xm(min(b, t)), LANE_Y[0], BLUE, op))
    for a, b in wait:                         # 师傅：红
        if a < t:
            g.add(bar(xm(a), xm(min(b, t)), LANE_Y[0], RED, op))
    for b, (ln, i, s, e) in sp.items():       # 小工：正在跑 / 跑完
        if s < t:
            g.add(bar(xm(s), xm(min(e, t)), LANE_Y[ln + 1], COL[BLOCKS[b][0]], op))
            if b in hot:
                g.add(Rectangle(width=max(xm(min(e, t)) - xm(s), 1e-3), height=LH, stroke_color=YELLOW, stroke_width=5,
                                stroke_opacity=op).move_to([(xm(s) + xm(min(e, t))) / 2, LANE_Y[ln + 1], 0]))
    for need, tm in deps:                     # 依赖线：那趟腿跑完 → 师傅接着算
        if tm <= t + 1e-9:
            ln, _, _, e = sp[need]
            g.add(DashedLine([xm(e), LANE_Y[ln + 1] + LH / 2, 0], [xm(tm), LANE_Y[0] - LH / 2, 0],
                             color=YELLOW, stroke_width=3, dash_length=0.08).set_stroke(opacity=op))
    for ln, lst in enumerate(queued(act, t)):  # 队列
        for k, b in enumerate(lst):
            g.add(qsq(b, queue_pos(k, ln), op, b in hot))
    if t > 0:                                 # 时间游标
        g.add(Line([xm(t), LANE_Y[2] - LH / 2 - 0.12, 0], [xm(t), LANE_Y[0] + LH / 2 + 0.12, 0],
                   color=YELLOW, stroke_width=4).set_stroke(opacity=op))
    return g


def draw_move(p):
    """橙块从小工甲的队挪到小工乙的队（时间 0 那一刻的队列，两幕之间插值）。"""
    q0, q1 = queued(0, 0.0), queued(1, 0.0)
    pos0 = {b: queue_pos(k, ln) for ln in (0, 1) for k, b in enumerate(q0[ln])}
    pos1 = {b: queue_pos(k, ln) for ln in (0, 1) for k, b in enumerate(q1[ln])}
    assert set(pos0) == set(pos1)
    s = p * p * (3 - 2 * p)
    return VGroup(*[qsq(b, [pos0[b][j] + (pos1[b][j] - pos0[b][j]) * s for j in range(3)], 1.0) for b in pos0])


CAPS = [
    ["两类活挤在小工甲一条队里，先来先做"],
    ["排在后面的那趟腿晚到：师傅停下来等"],
    ["小工乙几乎闲着", "像超市只开一个收银台，旁边的收银员闲着"],
    ["整步：这条链在小核上 %d ms，师傅露出来等 %d ms" % (N.CHAIN_SC_MS, N.CHAIN_EXPOSED_MS)],
    ["token 钉在小工甲，专家权重交给小工乙"],
    ["各排一队：活没少，等得少了"],
    ["一步 %.3f → %.3f 秒（−%.1f%%）" % (N.SPLIT_CTL, N.SPLIT_S, N.SPLIT_PCT)],
]
LANE_NAMES = ["师傅（主核）", "小工甲（小核 0）", "小工乙（小核 1）"]
BUSY = ["算 %.2f ms" % N.FWD_TC_COMPUTE, "忙 %.2f ms" % N.FWD_SC0_BUSY, "忙 %.2f ms" % N.FWD_SC1_BUSY]
LEGEND = [(BLUE, "师傅在算"), (RED, "师傅在等"), (ORANGE, "专家权重收集"), (TOKEN_PURPLE, "token 收发"), (GREY, "其它")]
FOOT = "一层前向；排布和等待长短是示意，忙时是实测中位数"


class TwoQueues(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def st():
            return state(tr.get_value())

        # 车道底色、名字、队列框、时间轴
        frame = VGroup()
        for k, y in enumerate(LANE_Y):
            frame.add(Rectangle(width=X1 - X0, height=LH, stroke_color=GREY, stroke_width=2).move_to([(X0 + X1) / 2, y, 0]))
            frame.add(tx(LANE_NAMES[k], 22, WHITE, True).move_to([LAB_X, y + 0.17, 0]))
        for ln in (0, 1):
            w = 4 * QW + 3 * QG + 0.12
            frame.add(RoundedRectangle(width=w, height=LH, corner_radius=0.08, stroke_color=GREY, stroke_width=2)
                      .move_to([Q_RIGHT - w / 2 + 0.06, LANE_Y[ln + 1], 0]))
        frame.add(tx("队列", 22, GREY_B).move_to([Q_RIGHT - (4 * QW + 3 * QG) / 2, LANE_Y[1] + LH / 2 + 0.24, 0]))
        frame.add(tx("时间 →", 22, GREY_B).move_to([X1 - 0.55, LANE_Y[0] + LH / 2 + 0.3, 0]))
        lx = -6.55
        for c, s in LEGEND:
            sq = Rectangle(width=0.3, height=0.3, stroke_width=0, fill_color=c, fill_opacity=0.85).move_to([lx + 0.15, -3.0, 0])
            lab = tx(s, 22, GREY_B)
            lab.next_to(sq, buff=0.12)
            frame.add(sq, lab)
            lx = lab.get_right()[0] + 0.45
        frame.add(tx(FOOT, 22, GREY).move_to([0, -3.6, 0]))
        self.add(frame)
        assert frame.width < 14.0, frame.width

        busy = VGroup(*[tx(BUSY[k], 22, GREY_B).move_to([LAB_X, LANE_Y[k] - 0.2, 0]) for k in range(3)])

        def upd_busy(m):
            m.set_opacity(st()["lab"])
        upd_busy(busy)                                  # ⛔ 先手动算一次
        busy.add_updater(upd_busy)
        self.add(busy)

        def dyn():
            s = st()
            g = VGroup(*[draw(a, t, op) for a, t, op in s["layers"]])
            if s["move"] is not None:
                g.add(draw_move(s["move"]))
            return g

        dyn_m = always_redraw(dyn)
        self.add(dyn_m)

        # 字幕：建一次，按区间切透明度
        caps = []
        for lines in CAPS:
            if len(lines) == 1:
                m = tx(lines[0], 28).move_to([0, 3.2, 0])
            else:
                m = VGroup(tx(lines[0], 28), tx(lines[1], 26, GREY_B)).arrange(direction=[0, -1, 0], buff=0.18).move_to([0, 2.95, 0])
            assert m.width < 13.5
            caps.append(m)
        caps[6].set_color(YELLOW)

        def show_cap(m, k):
            m.set_opacity(1.0 if st()["cap"] == k else 0.0)
        for k, m in enumerate(caps):
            show_cap(m, k)                              # ⛔ 先手动算一次
            m.add_updater(lambda m, k=k: show_cap(m, k))
            self.add(m)

        self.play(tr.animate.set_value(T_TOTAL), run_time=T_TOTAL, rate_func=linear)
        self.play(tr.animate.set_value(T_TOTAL), run_time=1 / 30, rate_func=linear)   # 强制刷一帧（末帧 ＝ 第 0 帧）
        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "TwoQueues.json"), "w") as fh:
            json.dump({"pauses": PAUSES, "duration": round(self.renderer.time, 2)}, fh)


if __name__ == "__main__":
    for k, (sp, comp, wait, deps) in enumerate(SIM):
        print("幕", k + 1, "红", round(WAIT_MS[k], 2), "层末", round(END[k], 2))
        for b in SEQ:
            print("  ", b, [round(v, 2) if isinstance(v, float) else v for v in sp[b]])
        print("   等", [(round(a, 2), round(b, 2)) for a, b in wait])
