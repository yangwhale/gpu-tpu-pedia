# -*- coding: utf-8 -*-
r"""专题五 · 1.5 折叠段「TPU 的环面」：3D 环面上三条轴怎么同时用起来（AllGather 为例）。

⭐ 为什么非动不可：内容就是「随时间轮换」——&#160;同一颗芯片的 X／Y／Z 三条轴，
  每一步各被哪一块数据占着，下一步又换成谁。静态图（fig5-torus-3ring）能把三步的
  忙闲摊成一张甘特图，但画不出「在立方体里，三个方向的线同时亮」这个空间感。
  ⭐ 前后两幕对照：先演逐轴（每一步只亮一个方向，另外两个方向的线全闲着），
    再演轮换（三块数据各走一条轴，每一步三个方向全亮），底下两根用时条同一把尺子。

⛔ 数据当场算（stage_sizes／phase_load），并断言：
  · 4×4×4 切片，每颗芯片起手 1/64，逐轴收集后依次 4/64 → 16/64 → 64/64；
  · 逐轴每一步只有一条轴在搬，三步的量 3 : 12 : 48，合计 63（单位：1/64 份数据过一条轴）；
  · 轮换把数据切三块、路线 XYZ／YZX／ZXY，每一步三条轴各搬一块的同一阶段，
    每轴 1 : 4 : 16，合计 21 ——&#160;正好 1/3。
  ⚠️ 只按带宽算，不算延迟；XLA 实际怎么排这三块没核对过，这里讲的是原理。
    锚点：How to Scale Your Model 中文版 sharding 章「沿多个轴收集时，可用带宽成倍增加」。

⛔ 刻意没画：
  · 环回线 ——&#160;每条线首尾相接成环，画出来立方体会糊成一团，字幕里交代一句；
  · 每条线上的双向 ——&#160;两幕都一样按双向算，不影响 3 倍这个比；
  · ReduceScatter／AllReduce ——&#160;RS 是倒着走（大的一步在前），AR ＝ RS ＋ AG，正文一句话带过。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic05-anim-torus.py TorusRings WebPages/media/topic05-torus.mp4
"""
import json
import os
from fractions import Fraction as Fr

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体：不指定时拉丁字母会回落到衬线体、标点按繁中居中
from manim import (Scene, VGroup, Line, Dot, Rectangle, Text, ValueTracker, always_redraw, FadeIn, FadeOut,
                   Transform, WHITE, GREY, GREY_B, BLUE, ORANGE, GREEN, YELLOW, UP, DOWN, LEFT, RIGHT, linear, smooth)

L = 4                                   # 每一维 4 颗芯片
AX = "XYZ"
CH_COL = [BLUE, ORANGE, GREEN]          # 块 A／B／C，跟静态图同色
CH_NAME = "ABC"
SEQ = ["XYZ", "XYZ", "XYZ"]             # 逐轴：三块走同一条路，挤在同一条轴上
ROT = ["XYZ", "YZX", "ZXY"]             # 轮换：每一步三块各占一条轴


# ── 数据当场算 ─────────────────────────────────────────────
def stage_sizes(l=L):
    """每颗芯片手里的份额：起手 1/l³，每收完一条轴乘 l。"""
    return [Fr(l ** i, l ** 3) for i in range(4)]


def phase_load(routes, l=L):
    """每一步每条轴要搬多少（单位：整份数据）。一块占 1/len(routes)；沿一条长 l 的轴收集，
    要收进 (l−1) 倍于手头的量。返回 [{轴: 量}]×3。"""
    st = stage_sizes(l)
    out = []
    for p in range(3):
        d = {a: Fr(0) for a in AX}
        for r in routes:
            d[r[p]] += Fr(1, len(routes)) * st[p] * (l - 1)
        out.append(d)
    return out


def phase_time(routes):
    """一步的用时 ＝ 最忙那条轴的量（每条轴带宽一样）。"""
    return [max(d.values()) for d in phase_load(routes)]


assert stage_sizes() == [Fr(1, 64), Fr(4, 64), Fr(16, 64), Fr(1)]
assert all(sorted(r) == list(AX) for r in ROT) and all(len({r[p] for r in ROT}) == 3 for p in range(3))
T_SEQ = [t * 64 for t in phase_time(SEQ)]
T_ROT = [t * 64 for t in phase_time(ROT)]
assert T_SEQ == [3, 12, 48] and T_ROT == [1, 4, 16], (T_SEQ, T_ROT)
assert sum(T_SEQ) == 63 and sum(T_ROT) == 21 and sum(T_SEQ) == 3 * sum(T_ROT)
# 逐轴每一步只有一条轴在忙；轮换每一步三条全忙
assert all(sum(v > 0 for v in d.values()) == 1 for d in phase_load(SEQ))
assert all(sum(v > 0 for v in d.values()) == 3 for d in phase_load(ROT))
FRAC = ["1/64", "4/64", "16/64", "64/64"]

# ── 等轴测投影：X 往右下、Y 往左下、Z 往上 ─────────────────────
O = [-3.7, -0.5, 0]
EX, EY, EZ = (0.72, -0.33), (-0.72, -0.33), (0.0, 0.78)


def P(x, y, z):
    x, y, z = x - 1.5, y - 1.5, z - 1.5
    return [O[0] + x * EX[0] + y * EY[0], O[1] + x * EX[1] + y * EY[1] + z * EZ[1], 0]


def edges(axis):
    """沿 axis 方向的所有相邻连线（不含环回）。"""
    out = []
    for a in range(L):
        for b in range(L):
            for c in range(L - 1):
                if axis == "X":
                    out.append((P(c, a, b), P(c + 1, a, b)))
                elif axis == "Y":
                    out.append((P(a, c, b), P(a, c + 1, b)))
                else:
                    out.append((P(a, b, c), P(a, b, c + 1)))
    return out


def lattice(lit):
    """lit：{轴: 颜色}；不在里面的轴画成暗灰细线。"""
    g = VGroup()
    for a in AX:
        col = lit.get(a)
        for p, q in edges(a):
            g.add(Line(p, q, stroke_color=col or GREY, stroke_width=4 if col else 1.2,
                       stroke_opacity=1 if col else 0.45))
    for x in range(L):
        for y in range(L):
            for z in range(L):
                g.add(Dot(P(x, y, z), radius=0.045, color=GREY_B))
    return g


def axis_tags():
    g = VGroup()
    for a, (p, q) in zip(AX, [(P(3, 0, 0), P(3.9, 0, 0)), (P(0, 3, 0), P(0, 3.9, 0)), (P(0, 0, 3), P(0, 0, 3.75))]):
        g.add(Line(p, q, stroke_color=GREY_B, stroke_width=2))
        g.add(Text(a, font_size=26, color=GREY_B).move_to([q[0] + (0.2 if a == "X" else -0.2 if a == "Y" else 0),
                                                         q[1] + (0.18 if a == "Z" else 0), 0]))
    return g


# ── 右栏：三块数据的进度 ＋ 两根用时条 ───────────────────────
BX0, BW, BH = 3.0, 2.9, 0.32
ROW_Y = [1.55, 0.85, 0.15]
TX0, TSC = 1.25, 4.6 / 63               # 用时条：63 个单位 ＝ 4.6 宽
TY = [-1.65, -2.45]


def route_label(i, routes):
    return Text("块 %s  %s" % (CH_NAME[i], "→".join(routes[i])), font_size=22,
                color=CH_COL[i]).move_to([0.95, ROW_Y[i], 0], aligned_edge=LEFT)


def frac_label(i, k):
    return Text(FRAC[k], font_size=20, color=GREY_B).move_to([BX0 + BW + 0.15, ROW_Y[i], 0], aligned_edge=LEFT)


STEP_T, HOLD = 1.8, 2.0
STEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")
TITLE = "3D 环面上的 AllGather：三块数据轮换着走，三条轴一起忙"
OPEN_CAP = "4×4×4 切片，每颗芯片起手 1/64；每条线首尾相接成环（环回没画）"
ACTS = [("逐轴", SEQ, "先看逐轴：三块挤在同一条轴上，X 走完才走 Y、再走 Z"),
        ("轮换", ROT, "再看轮换：三块各挑一条轴起步，A 走 XYZ、B 走 YZX、C 走 ZXY")]


def step_cap(act, s, routes):
    if act == "逐轴":
        a = routes[0][s]
        idle = "、".join(x for x in AX if x != a)
        return "第 %d 步：只有 %s 方向在忙，%s 闲着" % (s + 1, a, idle)
    return "第 %d 步：" % (s + 1) + "，".join("%s 搬 %s" % (routes[i][s], CH_NAME[i]) for i in sorted(range(3), key=lambda i: routes[i][s]))


def done_cap(act, s):
    if act == "逐轴":
        return "第 %d 步完成：用时 %d（这一步每块都涨到 %s）" % (s + 1, T_SEQ[s], FRAC[s + 1])
    return "第 %d 步完成：用时 %d，三条轴全亮（每块都涨到 %s）" % (s + 1, T_ROT[s], FRAC[s + 1])


END_SEQ = "逐轴走完：用时 63，大部分时间只有一条轴在干活"
END_ROT = "轮换走完：用时 21 ＝ 逐轴的 1/3，三条轴的带宽全用上了"


def aria():
    """⭐ 无障碍描述从场景里的字串拼，屏幕上出现的每一句都在。"""
    s = ["3D 环面三轴轮换动画。左边是 4×4×4 的立方体格点，X 往右下、Y 往左下、Z 往上；",
         "右边是块 A 蓝、块 B 橙、块 C 绿三行进度条，下面两根用时条，逐轴和轮换用同一把尺子。",
         "标题：%s。字幕：%s。" % (TITLE, OPEN_CAP)]
    for act, routes, cap in ACTS:
        s.append(cap + "。")
        for k in range(3):
            s.append(step_cap(act, k, routes) + "，" + done_cap(act, k) + "。")
        s.append((END_SEQ if act == "逐轴" else END_ROT) + "。")
    s.append("最后画面复位。")
    return "".join(s)


class TorusRings(Scene):
    def setup(self):
        self.marks, self.cap = [], None

    def say(self, txt, color=None):
        new = Text(txt, font_size=24, color=color or GREY_B).next_to(self.title, DOWN, buff=0.18)
        self.play(*([FadeOut(self.cap)] if self.cap else []), FadeIn(new), run_time=0.4)
        self.cap = new

    def hold(self, t=HOLD):
        self.marks.append(round(self.renderer.time + 0.3, 2))
        self.wait(t)

    def construct(self):
        self.title = Text(TITLE, font_size=30, color=WHITE).to_edge(UP)
        fill = [ValueTracker(1 / 64) for _ in range(3)]
        tsec = [ValueTracker(0.0), ValueTracker(0.0)]

        def bars():
            g = VGroup()
            for i in range(3):
                g.add(Rectangle(width=BW, height=BH, stroke_color=GREY, stroke_width=2)
                      .move_to([BX0 + BW / 2, ROW_Y[i], 0]))
                w = max(BW * fill[i].get_value(), 0.02)
                g.add(Rectangle(width=w, height=BH - 0.04, stroke_width=0, fill_color=CH_COL[i], fill_opacity=0.9)
                      .move_to([BX0 + w / 2, ROW_Y[i], 0]))
            for j in range(2):
                w = max(TSC * tsec[j].get_value(), 0.001)
                g.add(Rectangle(width=w, height=0.36, stroke_width=0, fill_color=WHITE if j == 0 else YELLOW,
                                fill_opacity=0.85 if w > 0.01 else 0).move_to([TX0 + w / 2, TY[j], 0]))
            return g

        head = VGroup(Text("每颗芯片手里这一块收齐了多少", font_size=20, color=GREY).move_to([0.95, 2.15, 0], aligned_edge=LEFT),
                      Text("用时（同一把尺子，单位：1/64 份过一条轴）", font_size=20, color=GREY)
                      .move_to([0.2, -1.1, 0], aligned_edge=LEFT),
                      Text("逐轴", font_size=22, color=WHITE).move_to([0.2, TY[0], 0], aligned_edge=LEFT),
                      Text("轮换", font_size=22, color=YELLOW).move_to([0.2, TY[1], 0], aligned_edge=LEFT))
        routes = VGroup(*[route_label(i, SEQ) for i in range(3)])
        fracs = VGroup(*[frac_label(i, 0) for i in range(3)])
        cube = lattice({})
        # ⛔ 开场字幕直接摆上（不淡入）：收尾也是摆着这句停住，首尾才同帧
        self.cap = Text(OPEN_CAP, font_size=24, color=GREY_B).next_to(self.title, DOWN, buff=0.18)
        self.add(self.title, axis_tags(), cube, head, routes, fracs, always_redraw(bars), self.cap)
        self.wait(2.2)

        tnums = []
        for j, (act, rts, cap) in enumerate(ACTS):
            if j == 1:
                self.play(Transform(routes, VGroup(*[route_label(i, ROT) for i in range(3)])),
                          *[f.animate.set_value(1 / 64) for f in fill],
                          Transform(fracs, VGroup(*[frac_label(i, 0) for i in range(3)])), run_time=0.8)
            self.say(cap)
            self.wait(1.4)
            times = T_SEQ if act == "逐轴" else T_ROT
            for s in range(3):
                lit = {}
                for i in range(3):
                    lit[rts[i][s]] = WHITE if act == "逐轴" else CH_COL[i]
                self.say(step_cap(act, s, rts))
                new = lattice(lit)
                self.play(Transform(cube, new), run_time=0.5)
                self.play(*[f.animate.set_value(float(stage_sizes()[s + 1])) for f in fill],
                          tsec[j].animate.set_value(tsec[j].get_value() + times[s]),
                          Transform(fracs, VGroup(*[frac_label(i, s + 1) for i in range(3)])),
                          run_time=STEP_T, rate_func=linear)
                self.say(done_cap(act, s), YELLOW)
                self.hold()
                self.play(Transform(cube, lattice({})), run_time=0.4)
            tn = Text(str(sum(times)), font_size=22, color=WHITE if j == 0 else YELLOW).move_to(
                [TX0 + TSC * sum(times) + 0.12, TY[j], 0], aligned_edge=LEFT)
            tnums.append(tn)
            self.play(FadeIn(tn), run_time=0.3)
            self.say(END_SEQ if act == "逐轴" else END_ROT, GREEN if j == 1 else None)
            self.hold()

        # ⭐ 复位：清掉字幕和数字，进度、用时、路线全部拨回第 0 帧
        self.play(FadeOut(self.cap), *[FadeOut(t) for t in tnums],
                  *[f.animate.set_value(1 / 64) for f in fill], *[t.animate.set_value(0.0) for t in tsec],
                  Transform(routes, VGroup(*[route_label(i, SEQ) for i in range(3)])),
                  Transform(fracs, VGroup(*[frac_label(i, 0) for i in range(3)])), run_time=0.9, rate_func=smooth)
        self.remove(self.cap, *tnums)
        self.cap = None
        self.say(OPEN_CAP)
        self.wait(0.6)
        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "TorusRings.json"), "w") as fh:
            json.dump({"pauses": self.marks, "duration": round(self.renderer.time, 2)}, fh)


if __name__ == "__main__":
    print(aria())
