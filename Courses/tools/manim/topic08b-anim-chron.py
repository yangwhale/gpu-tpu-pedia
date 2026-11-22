# -*- coding: utf-8 -*-
r"""专题九 · 开场编年史的动画：一站一站降下去，两次换底座「跳」回去 —— ChronDescent。
放在静态图 fig-chron（三段编年史柱图）下面；画面里不写课件节号。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  编年史本身就是一个「一站一站降下去」的过程：每一站砍掉一截、瓶颈换一类。
  两次换底座时同一配置量了两次，柱子「跳」回去 —— 跳回去不是变慢，是换了尺子。
  静态图把三段画成三块、各自一根纵轴，读者看得到每段的终点，看不到「降—跳—再降」这个节奏；
  这里把 30 根柱（28 站 ＋ 两个换底座后的起点）放进同一根时间轴，节奏只有放在时间里才看得出来。

⛔ 数据从哪来（全部 import 自 tools/topic08b_numbers.py 并断言）
  · 站 0–12：N.FRONT；站 13–16：N.HALF；站 17–27：N.LONG（站号、白话、步时、有效 MFU、瓶颈类）。
  · 衔接①：N.SW1 ＝ (2.731, 12.25, 5.476, 11.10)，同一份推荐配置，20 层底座 → 半层底座。
  · 衔接②：N.SW2 ＝ (5.311, 10.98, 5.553, 17.83)，站 13 那套配置，短文 → 长文。
    所以第二道虚线处柱子是从站 13 的高度（5.311，画一条点线引过来）跳到 5.553，不是从站 16 的 4.049 跳。
  · 终点：N.END_S ＝ 4.116、N.END_MFU ＝ 24.18（站 27，梯度四处精度写法改回参考实现，+0.1%；站 26 N.S26_S ／ N.S26_MFU ＝ 4.108 ／ 24.23；站 25 N.S25_S ／ N.S25_MFU ＝ 4.097 ／ 24.30；最快一站是站 24 N.S24_S ／ N.S24_MFU ＝ 4.094 ／ 24.32；更早：站 23 N.S23_S / N.S23_MFU ＝ 4.178 / 23.83，
    站 22 N.S22_S / N.S22_MFU ＝ 4.192 / 23.75，站 21 N.S21_S / N.S21_MFU ＝ 4.225 / 23.56）。
  · 配色照 tools/topic08b-fig-chron.py 的 KCOL：通信字节 红、小核队列 橙、排法调度与优化器 蓝、数据 灰；
    换底座后的起点是灰色虚线框。站 0 在静态图里也按「bytes」涂红，这里跟着。

⛔ 刻意没画
  · 不写芯片数；段名写「20 层底座」「半层底座·短文」「半层底座·长文」（层数来自公开模型卡）。
  · 一根纵轴是为了看节奏；跨段高低不能相除，画面在两道虚线处用字幕明说。
  · 「这一步 %」不上画面（各站优先用同作业对照，不等于相邻两柱相除）；白话超过 12 个字只显示前 12 个字。
  · 柱顶数字只常驻首末站和两个衔接点，其余站只在走到时显示（30 根柱的数字全挂会挤成一团）。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-chron.py ChronDescent WebPages/media/topic08b-chron.mp4
   两次换底座和终点的停顿时刻写进 steps/ChronDescent.json（课件挂 data-pauses）。

⭐ 写法：时钟式。唯一的进度 u（第 i 根柱占 [i, i+1)）；画面全是 u 的纯函数。
   复位：字幕淡出 → u 倒着走回 0（柱子倒放缩回去）→ 字幕淡回。末帧按定义就是第 0 帧。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Line, DashedLine, Dot, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, RED, YELLOW, ORANGE, GREEN, LEFT, RIGHT, UP, DOWN, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── 数据（当场取、当场断言）──────────────────────────────────────────
assert [r[0] for r in N.FRONT] == list(range(13))
assert [r[0] for r in N.HALF] == list(range(13, 17))
assert [r[0] for r in N.LONG] == list(range(17, 28))
assert N.SW1 == (2.731, 12.25, 5.476, 11.10) and N.SW1[0] == N.FRONT[-1][2] and N.SW1[1] == N.FRONT[-1][3]
assert N.SW1[2] == N.HALF_START
assert N.SW2 == (5.311, 10.98, 5.553, 17.83) and N.SW2[0] == N.HALF[0][2] and N.SW2[1] == N.HALF[0][3]
assert N.SW2[2] == N.LONG_START
assert (N.END_S, N.END_MFU) == (4.116, 24.18) == (N.LONG[-1][2], N.LONG[-1][3])
assert (N.S26_S, N.S26_MFU) == (4.108, 24.23) == (N.LONG[-2][2], N.LONG[-2][3])
assert (N.S25_S, N.S25_MFU) == (4.097, 24.30) == (N.LONG[-3][2], N.LONG[-3][3])
assert (N.S24_S, N.S24_MFU) == (4.094, 24.32) == (N.LONG[-4][2], N.LONG[-4][3])
assert (N.S23_S, N.S23_MFU) == (4.178, 23.83) == (N.LONG[-5][2], N.LONG[-5][3])
assert N.LONG[-1][5] == "num" and N.LONG[-2][5] == "num" and N.LONG[-3][5] == "num" and N.LONG[-4][5] == "bytes"
assert N.FRONT[0][2] == 12.743

SEG_NAME = ["20 层底座", "半层·短文", "半层·长文"]
KCOL = {"bytes": RED, "scq": ORANGE, "sched": BLUE, "data": GREY_B, "num": GREEN}
LEGEND = [(RED, "通信字节"), (ORANGE, "小核队列"), (BLUE, "排法与 kernel"), (GREY_B, "数据"), (GREEN, "精度修正")]

# 槽位：(标签, 步时, 有效 MFU, 段, 白话, 颜色类别, 起始高度来自哪个槽)
SLOTS = []
for s, n, t, m, _p, k in N.FRONT:
    SLOTS.append(dict(lab=str(s), t=t, m=m, seg=0, txt=n, k=k))
SLOTS.append(dict(lab="起", t=N.SW1[2], m=N.SW1[3], seg=1, txt=None, k="start", src=N.SW1[0]))
J1 = len(SLOTS) - 1
for s, n, t, m, _p, k in N.HALF:
    SLOTS.append(dict(lab=str(s), t=t, m=m, seg=1, txt=n, k=k))
SLOTS.append(dict(lab="起", t=N.SW2[2], m=N.SW2[3], seg=2, txt=None, k="start", src=N.SW2[0]))
J2 = len(SLOTS) - 1
I13 = J1 + 1                                             # 站 13 的槽位（衔接②的「旧尺子」从它引过来）
for s, n, t, m, _p, k in N.LONG:
    SLOTS.append(dict(lab=str(s), t=t, m=m, seg=2, txt=n, k=k))
NS = len(SLOTS)
assert NS == 30 and sum(1 for d in SLOTS if d["k"] != "start") == 28
assert SLOTS[I13]["lab"] == "13" and SLOTS[J1]["src"] == SLOTS[J1 - 1]["t"]
for i, d in enumerate(SLOTS):                          # 每根柱从哪个高度落下来（换底座的从旧尺子上的值升上去）
    d.setdefault("src", SLOTS[i - 1]["t"] if i else d["t"])
KEY = {0, J1 - 1, J1, J1 + 4, J2, NS - 1}                # 常驻柱顶数字：首站、站 12、两个起点、站 16、站 27
assert SLOTS[J1 + 4]["lab"] == "16" and SLOTS[NS - 1]["lab"] == "27"


def short(s, n=12):
    """白话只取前 n 个字：一个汉字算一个字，一个英文词／数字整体算一个字、不从中间切；空格不算。"""
    import re
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.]*| |.", s)
    out, c = "", 0
    for tk in toks:
        if c >= n:                                     # 截断：能退到最后一个分隔处（保留 ≥ 6 字）就退过去
            cut = max(out.rfind(x) for x in "＋、（，：")
            if cut > 0 and len(out[:cut].replace(" ", "")) >= 6:
                out = out[:cut]
            return out.rstrip(" ＋、（，：") + "…"
        out += tk
        c += tk != " "
    return out


TITLE = "优化编年史：28 站，一站一站降下去"
CAP_SWITCH = "换底座：同一配置量两次"
CAP_SWITCH2 = "换数据：同一配置量两次"
CAP_RULER = "跳回去不是变慢：换了尺子，前后不能相除"
CAP_END = ("28 站：先砍字节，再理排法，", "再修小核队列，最后对着长文一项项修")
for s in [TITLE, CAP_SWITCH, CAP_SWITCH2, CAP_RULER, *CAP_END]:
    assert len(s) <= 22, s

# ── 版面 ─────────────────────────────────────────────────────────
BASE, SC = -2.15, 0.33                                   # 0 秒的 y；每秒多高
XL, XR, GAP = -5.95, 6.45, 0.3                            # 两道虚线处各多留 GAP


def slot_x(i):
    w = (XR - XL - 2 * GAP) / NS
    return XL + w * (i + 0.5) + GAP * ((i >= J1) + (i >= J2))


BW = 0.34
X_DASH = [(slot_x(J1 - 1) + slot_x(J1)) / 2, (slot_x(J2 - 1) + slot_x(J2)) / 2]
Y_CAP1, Y_CAP2 = -3.05, -3.6


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def height(i, u):
    """第 i 根柱在进度 u 时的步时（None ＝ 还没出现）。"""
    if i == 0:
        return SLOTS[0]["t"] if u >= 0 else None
    if u < i:
        return None
    d = SLOTS[i]
    return d["src"] + (d["t"] - d["src"]) * ease((u - i) / 0.6)


def cur_slot(u):
    return max(0, min(NS - 1, int(u)))


def tx(s, size=26, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def dump_steps(name, marks, t):
    os.makedirs(STEPS_DIR, exist_ok=True)
    with open(os.path.join(STEPS_DIR, name + ".json"), "w") as fh:
        json.dump({"pauses": marks, "duration": round(t, 2)}, fh)


class ChronDescent(Scene):
    def construct(self):
        u = ValueTracker(0.0)          # 进度
        r = ValueTracker(0.0)          # 复位时字幕淡出的量（0 ＝ 正常）
        marks = []

        # ── 静态件 ──
        title = tx(TITLE, 34, WHITE, True).to_edge(UP, buff=0.3)
        leg = VGroup()
        for col, name in LEGEND:
            leg.add(VGroup(Rectangle(width=0.22, height=0.22, stroke_width=0, fill_color=col, fill_opacity=1),
                           tx(name, 22, GREY_B)).arrange(RIGHT, buff=0.1))
        leg.add(VGroup(Rectangle(width=0.22, height=0.22, stroke_color=GREY, stroke_width=2.5),
                       tx("切换后起点", 22, GREY_B)).arrange(RIGHT, buff=0.1))
        leg.arrange(RIGHT, buff=0.32).move_to([0, 2.88, 0]).to_edge(LEFT, buff=0.45)
        axis = Line([XL - 0.1, BASE, 0], [XR + 0.1, BASE, 0], color=GREY, stroke_width=3)
        yt = VGroup()
        for v in (4, 8, 12):
            yt.add(Line([XL - 0.18, BASE + v * SC, 0], [XL - 0.08, BASE + v * SC, 0], color=GREY, stroke_width=3))
            yt.add(tx("%d 秒" % v, 22, GREY_B).next_to([XL - 0.18, BASE + v * SC, 0], LEFT, buff=0.06))
        nums = VGroup(*[tx(d["lab"], 22, GREY if d["k"] == "start" else GREY_B).move_to([slot_x(i), BASE - 0.28, 0])
                        for i, d in enumerate(SLOTS)])
        dashes, dlabs = VGroup(), VGroup()
        for x, dl in zip(X_DASH, ("换底座", "换数据")):
            dashes.add(DashedLine([x, BASE - 0.1, 0], [x, 0.85, 0], color=GREY_B, stroke_width=3, dash_length=0.12))
            dlabs.add(tx(dl, 22, GREY_B).move_to([x, 1.08, 0]))   # 压低：别跟右上角计数器挤成一行
        seg_c = [(slot_x(4) + slot_x(J1 - 1)) / 2, (slot_x(J1) + slot_x(J2 - 1)) / 2, (slot_x(J2) + slot_x(NS - 1)) / 2]
        segl = VGroup(*[tx(SEG_NAME[k], 24, GREY_B, True).move_to([seg_c[k], 0.65, 0]) for k in range(3)])
        self.add(title, leg, axis, yt, nums, dashes, dlabs, segl)

        # ── 柱子（纯函数）──
        def bars():
            g = VGroup()
            uu = u.get_value()
            for i, d in enumerate(SLOTS):
                h = height(i, uu)
                if h is None:
                    continue
                if d["k"] == "start":
                    b = Rectangle(width=BW, height=h * SC, stroke_color=GREY_B, stroke_width=2.5, fill_opacity=0)
                else:
                    col = KCOL[d["k"]]
                    b = Rectangle(width=BW, height=h * SC, stroke_width=0, fill_color=col, fill_opacity=0.9)
                g.add(b.move_to([slot_x(i), BASE + h * SC / 2, 0]))
            # 衔接②：旧尺子上的值从站 13 引过来
            if uu >= J2:
                y = BASE + N.SW2[0] * SC
                g.add(DashedLine([slot_x(I13) + BW / 2, y, 0], [slot_x(J2) + BW / 2, y, 0],
                                 color=YELLOW, stroke_width=2.5, dash_length=0.07))
            i = cur_slot(uu)
            h = height(i, uu)
            g.add(Dot([slot_x(i), BASE + h * SC, 0], radius=0.07, color=YELLOW))
            return g
        self.add(always_redraw(bars))

        # ── 柱顶数字：建一次，挪位置、切透明度 ──
        vlabs = [tx("%.3f" % d["t"], 22, WHITE) for d in SLOTS]
        old2 = tx("%.3f" % N.SW2[0], 22, YELLOW)        # 衔接②的旧尺子读数，标在点线上方

        def place_vlab(m, i):
            uu, rr = u.get_value(), r.get_value()
            h = height(i, uu)
            if h is None:
                m.set_opacity(0)
                return
            m.move_to([slot_x(i), BASE + h * SC + 0.2, 0])
            now = cur_slot(uu) == i and uu < NS
            m.set_color(YELLOW if now else WHITE)
            m.set_opacity((1.0 if (now or i in KEY) else 0.0) * (1 - rr))

        for i, m in enumerate(vlabs):
            place_vlab(m, i)
            m.add_updater(lambda m, i=i: place_vlab(m, i))
            self.add(m)

        def place_old2(m):
            uu = u.get_value()
            m.move_to([(slot_x(I13 + 1) + slot_x(I13 + 2)) / 2, BASE + N.SW2[0] * SC + 0.2, 0])
            m.set_opacity((1.0 if uu >= J2 else 0.0) * (1 - r.get_value()))
        place_old2(old2)
        old2.add_updater(place_old2)
        self.add(old2)

        # ── 右上角计数器：每个槽位一组，建一次，切透明度 ──
        counters = []
        for d in SLOTS:
            g = VGroup(tx(SEG_NAME[d["seg"]], 24, GREY_B),
                       tx("步时 %s 秒" % ("%.3f" % d["t"]), 30, YELLOW, True),
                       tx("有效 MFU %.2f%%" % d["m"], 26, WHITE)).arrange(DOWN, aligned_edge=RIGHT, buff=0.1)
            g.move_to([6.55, 2.3, 0], aligned_edge=RIGHT).shift(DOWN * 0.15)
            counters.append(g)

        def show_counter(g, i):
            uu = u.get_value()
            g.set_opacity((1.0 if cur_slot(uu) == i else 0.0) * (1 - r.get_value()))

        # ── 底部两行字幕：建一次，按 u 区间切透明度 ──
        line1 = [tx("站 %s · %s" % (d["lab"], short(d["txt"])) if d["txt"] else (CAP_SWITCH2 if i == J2 else CAP_SWITCH), 26,
                    WHITE if d["txt"] else YELLOW).move_to([0, Y_CAP1, 0]) for i, d in enumerate(SLOTS)]
        ruler = tx(CAP_RULER, 26, YELLOW).move_to([0, Y_CAP2, 0])
        end1 = tx(CAP_END[0], 28, YELLOW, True).move_to([0, Y_CAP1, 0])
        end2 = tx(CAP_END[1], 28, YELLOW, True).move_to([0, Y_CAP2, 0])

        def show_line1(m, i):
            uu = u.get_value()
            m.set_opacity((1.0 if (cur_slot(uu) == i and uu < NS) else 0.0) * (1 - r.get_value()))

        def show_ruler(m):
            uu = u.get_value()
            on = any(j <= uu < j + 2 for j in (J1, J2))
            m.set_opacity((1.0 if on else 0.0) * (1 - r.get_value()))

        def show_end(m):
            m.set_opacity((1.0 if u.get_value() >= NS else 0.0) * (1 - r.get_value()))

        for i, g in enumerate(counters):
            show_counter(g, i)
            g.add_updater(lambda g, i=i: show_counter(g, i))
            self.add(g)
        for i, m in enumerate(line1):
            show_line1(m, i)
            m.add_updater(lambda m, i=i: show_line1(m, i))
            self.add(m)
        show_ruler(ruler)
        ruler.add_updater(show_ruler)
        self.add(ruler)
        for m in (end1, end2):
            show_end(m)
            m.add_updater(show_end)
            self.add(m)

        def go(v, t):
            self.play(u.animate.set_value(v), run_time=t, rate_func=linear)

        def hold(t):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.play(r.animate.set_value(0.0), run_time=t, rate_func=linear)   # 极短 play 推着 updater 走（值不变）

        # ── 时间线 ──
        go(1.0, 1.0)                                     # 站 0 先停一下
        for i in range(1, NS):
            if i in (J1, J2):
                go(i + 0.8, 1.0)                         # 升上去
                hold(1.5)                                # 跳的地方停住（下一步）
                go(i + 1.0, 0.3)
            else:
                go(i + 1.0, 0.5)
        go(NS + 0.01, 0.1)                               # 终点：换成结尾两行
        hold(2.6)

        # ── 复位：字幕淡出 → 倒放 → 字幕淡回第 0 帧 ──
        self.play(r.animate.set_value(1.0), run_time=0.35, rate_func=linear)
        go(0.0, 1.3)
        self.play(r.animate.set_value(0.0), run_time=0.35, rate_func=linear)
        self.play(u.animate.set_value(0.0), run_time=0.4, rate_func=linear)
        dump_steps("ChronDescent", marks, self.renderer.time)


if __name__ == "__main__":
    for i, d in enumerate(SLOTS):
        print(i, d["lab"], d["t"], d["m"], d["k"], d["src"], short(d["txt"]) if d["txt"] else "-")
