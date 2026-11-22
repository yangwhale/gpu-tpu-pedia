# -*- coding: utf-8 -*-
r"""专题九 §7.1「一道屏障，快 6.9%」的动画 —— BarrierHoist。挂在静态图 fig-barrier 下面。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · 论点是「一块优化器才用的数据被提前收进来，从前向之前一直占着显存，活过整个反向」——
    「占着」是一段跨越时间的占用：游标从左走到右，那块紫色在显存里一直不走，要看着它走过整个反向才有感觉。
  · 加屏障之后的变化也是时间轴上的挪动：那块收数据的动作从最前面滑到优化器段，
    反向那几条通信的括号被拉宽、师傅的红色等待缩短、整个反向变短。静态图只能并排放两个终态，画不出「挪」。

⛔ 数据从哪来（全部 import 自 tools/topic08b_numbers.py，断言钉住）
  · BAR_HOIST_GIB ＝ 7.125：没有屏障时被提前到前向之前、一直活到优化器段的那块数据（编译产物实测），画面写 7.1 GiB。
  · BAR_LIVE_GIB ＝ (48.26, 42.23)：反向循环之前产出、之后才用的数据合计，少 6.03 → 画面写「约 6 GiB」。
  · BAR_BRACKET ＝ (801, 1261)：反向循环里，每条小核通信「放行到完成」之间夹的计算合计。
    画面里三条括号里的蓝色宽度按这个比例放大（INNER2 / INNER1 ＝ 1261 / 801，断言）。
  · BAR_LAYER_CUT_PCT ＝ 11.8：反向每层短了约一成。画面里反向段的宽度按这个比例缩（W2 / W1 ＝ 1 − 0.118，断言）——
    缩掉的那一截正好等于三处红色等待缩掉的合计（W1 由这个条件反解出来）。
  · BAR_CTL_S / BAR_S ＝ 5.311 / 4.946，BAR_PCT ＝ 6.9。
  · BAR_SC1_DIFF_PCT ＝ 0.02：反向窗口里小核 1 的忙时前后只差 0.02%（只给比例）。画面里每条通信（橙条）两幕一样长。

⛔ 刻意没画
  · 三段的宽度、显存刻度都是示意：前向 / 优化器段宽度不按实测毫秒，显存曲线（前向攒、反向放的三角形）是示意形状，
    不代表峰值 —— 正常步峰值实际只少约 3 GiB（BAR_MEM_SAVED），画面不写峰值。
  · 「腾出约 6 GiB」说的是反向期间一直占着的数据合计（48.26 → 42.23），不是那块 7.1 GiB 整块：
    差的约 1.1 GiB 是别的数据，画面不展开，只写「约 6 GiB」。
  · 括号只画三条（实际有很多条；没夹计算的括号 18 → 1 条，这个数不上画面）；红色也不画成零（加屏障后仍有少量等待）。
  · 优化器段自己的通信、前向里的通信、两颗小核的分工都不画；不写屏障开关的名字。
  · 「余量紧所以调度器不敢拉长括号」是课件里标了「推断」的因果，画面只写现象：「显存紧：括号排得很短」。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-barrier.py BarrierHoist WebPages/media/topic08b-barrier.mp4
   两幕各停一下，停顿时刻写进 steps/BarrierHoist.json（课件挂 data-pauses）。

⭐ 写法：两个 ValueTracker —— u ＝ 游标（0→1 走完一步），m ＝ 幕（0 没屏障，1 加屏障）。
   画面全是 (u, m) 的纯函数；收尾把 u、m 一起拨回 0（倒放），末帧按定义就是第 0 帧。字幕用 say() 换，结尾摆回空字幕。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Polygon, Line, Text, ValueTracker, always_redraw,
                   FadeIn, FadeOut, Succession, Wait, WHITE, GREY, GREY_B, BLUE, RED, GREEN, YELLOW, ORANGE,
                   UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")
# 配色分工：橙只表示小工跑腿（通信）；显存里那块被提前收进来的数据用紫 —— 跟「谁在搬」分开，只表示「这块数据活着」
DATA_PURPLE = "#A35CF0"

# ════════════════════════════════════════════════════════════════
# ① 数据
# ════════════════════════════════════════════════════════════════
assert N.BAR_HOIST_GIB == 7.125
assert N.BAR_LIVE_GIB == (48.26, 42.23)
assert N.BAR_BRACKET == (801, 1261)
assert N.BAR_LAYER_CUT_PCT == 11.8
assert (N.BAR_CTL_S, N.BAR_S) == (5.311, 4.946)
assert N.BAR_SC1_DIFF_PCT == 0.02
HOIST_S = "%.1f" % N.BAR_HOIST_GIB                                   # 7.1
FREED = N.BAR_LIVE_GIB[0] - N.BAR_LIVE_GIB[1]
assert abs(FREED - 6.03) < 0.01
FREED_S = "%d" % round(FREED)                                        # 6
PCT_S = "%.1f" % N.BAR_PCT                                           # 6.9
assert PCT_S == "6.9" and HOIST_S == "7.1" and FREED_S == "6"

TITLE = "一道屏障，快 %s%%" % PCT_S
CAPS = [
    "提前收进来的 %s GiB，一路占到优化器才用" % HOIST_S,
    "反向显存紧：括号排得很短，师傅在等",
    "加一道屏障：这块挪回优化器段",
    "括号宽了，通信藏到计算底下",
    "活一点没少，排法变了",
]
COUNTERS = [
    "括号里夹的计算　%d → %d" % N.BAR_BRACKET,
    "反向每层　短了约一成",
    "一步　%.3f → %.3f 秒（−%s%%）" % (N.BAR_CTL_S, N.BAR_S, PCT_S),
    "小工在反向里的活　一点没少（差 %.2f%%）" % N.BAR_SC1_DIFF_PCT,
]
LANE_SC, LANE_TC, MEM_LAB, MEM_NOTE, LIMIT = "小工", "师傅", "显存", "（示意）", "显存上限"
SEGS = ["前向", "反向", "优化器"]
REL, DONE = "放行", "完成"
HOIST_LAB = "%s GiB" % HOIST_S
FREED_LAB = "反向腾出约 %s GiB" % FREED_S
for s in CAPS:                          # 每句 ≤ 22 字（半角字符按半个字算）
    assert sum(0.5 if ord(ch) < 128 else 1 for ch in s) <= 22, s

# ── 几何（示意宽度；反向段由数据反解）
X0 = -5.5
FW, OW = 2.2, 2.0                       # 前向、优化器段宽度（示意）
D = 0.75                                # 每条通信在小工车道上的长度：两幕一样（小工的活一点没少）
R1 = 0.30                               # 没屏障时每条括号后面师傅等的长度（示意）
INNER1 = D - R1                         # 括号里夹的计算宽度
INNER2 = INNER1 * N.BAR_BRACKET[1] / N.BAR_BRACKET[0]
R2 = D - INNER2
assert 0 < R2 < 0.2 * R1, R2            # 加屏障后红色缩小但不为零
NB = 3                                  # 画三条括号
W1 = NB * (R1 - R2) / (N.BAR_LAYER_CUT_PCT / 100)   # 反向段宽度：缩掉的 ＝ 红色缩掉的
C_BLUE = W1 - NB * R1
GAPS = [C_BLUE * f for f in (0.25, 0.30, 0.30, 0.15)]
assert abs((W1 - NB * (R1 - R2)) / W1 - (1 - N.BAR_LAYER_CUT_PCT / 100)) < 1e-9
assert all(g > D for g in GAPS[:3])     # 通信不重叠
XE1 = X0 + FW + W1 + OW
assert XE1 < 5.45, XE1

Y_SEG, Y_SC, H_SC, Y_TC, H_TC = 2.42, 1.82, 0.40, 1.18, 0.52
Y0, YL = -2.35, 0.20                    # 显存：零线、上限线
B0, H_OR = 0.25, 0.90                   # 底座、紫块（那块数据）高度（示意）
P = (YL - 0.08) - Y0 - H_OR             # 前向结束处底座高度：没屏障时顶刚好贴着上限
HW = 0.6                                # 小工车道上「收数据」那一块的宽度
USE_DX = 1.2                            # 优化器段里用掉那块的位置


def lerp(a, b, m):
    return a + (b - a) * m


def geom(m):
    r = lerp(R1, R2, m)
    xb = X0 + FW
    d = [xb + GAPS[0]]
    for i in (1, 2):
        d.append(d[-1] + r + GAPS[i])
    xo = d[-1] + r + GAPS[3]
    xe = xo + OW
    hx = lerp(X0, xo, m)
    return dict(r=r, xb=xb, d=d, xo=xo, xe=xe, hx=hx,
                os=hx + HW / 2, oe=xo + USE_DX)


def base_h(x, g):
    if x <= g["xb"]:
        return lerp(B0, P, (x - X0) / FW)
    if x <= g["xo"]:
        return lerp(P, B0, (x - g["xb"]) / (g["xo"] - g["xb"]))
    return B0


def band(a, b, low, high, col, op, g):
    """[a, b] 上 low(x)～high(x) 的填充多边形；a ≥ b 返回 None。"""
    if b - a < 1e-3:
        return None
    xs = sorted({a, b} | {x for x in (g["xb"], g["xo"]) if a < x < b})
    pts = [[x, high(x), 0] for x in xs] + [[x, low(x), 0] for x in reversed(xs)]
    return Polygon(*pts, stroke_width=0, fill_color=col, fill_opacity=op)


def rect(a, b, y, h, col, op=0.9):
    if b - a < 1e-3:
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
class BarrierHoist(Scene):
    def construct(self):
        marks = []
        u = ValueTracker(0.0)                       # 游标
        mt = ValueTracker(0.0)                      # 幕

        def cur(g):
            return lerp(X0, g["xe"], u.get_value())

        def scene_at():
            m = mt.get_value()
            g = geom(m)
            c = cur(g)
            grp = []
            # 车道外框与段界
            for y, h in ((Y_SC, H_SC), (Y_TC, H_TC)):
                grp.append(Rectangle(width=g["xe"] - X0, height=h, stroke_color=GREY, stroke_width=2)
                        .move_to([(X0 + g["xe"]) / 2, y, 0]))
            for x in (X0, g["xb"], g["xo"], g["xe"]):
                grp.append(Line([x, Y_TC - H_TC / 2, 0], [x, Y_SEG - 0.2, 0], color=GREY, stroke_width=3))
            # 师傅：蓝＝算，红＝等
            pieces = [(X0, g["d"][0], BLUE)]
            for i in range(NB):
                a = g["d"][i]
                nxt = g["d"][i + 1] if i + 1 < NB else g["xo"]
                pieces += [(a, a + g["r"], RED), (a + g["r"], nxt, BLUE)]
            pieces.append((g["xo"], g["xe"], BLUE))
            for a, b, col in pieces:
                grp.append(rect(a, min(b, c), Y_TC, H_TC - 0.08, col, 0.85))
            # 小工：收数据那一块 ＋ 反向里三条通信
            grp.append(rect(g["hx"], min(g["hx"] + HW, c), Y_SC, H_SC - 0.08, ORANGE, 0.95))
            for i in range(NB):
                e = g["d"][i] + g["r"]
                grp.append(rect(e - D, min(e, c), Y_SC, H_SC - 0.08, ORANGE, 0.7))
                for x in (e - D, g["d"][i]):          # 括号：放行、完成两条竖线
                    if c >= x:
                        grp.append(Line([x, Y_TC - H_TC / 2 - 0.08, 0], [x, Y_SC + H_SC / 2 + 0.08, 0],
                                     color=WHITE, stroke_width=4))
            # 显存
            grp.append(band(X0, min(g["xe"], c), lambda x: Y0, lambda x: Y0 + base_h(x, g), GREY, 0.55, g))
            grp.append(band(g["os"], min(g["oe"], c), lambda x: Y0 + base_h(x, g),
                         lambda x: Y0 + base_h(x, g) + H_OR, DATA_PURPLE, 0.85, g))
            # 游标
            grp.append(Line([c, Y0 - 0.05, 0], [c, Y_SEG - 0.2, 0], color=YELLOW, stroke_width=5))
            return VGroup(*[x for x in grp if x is not None])

        # 静态件
        title = tx(TITLE, 36, WHITE, True).to_edge(UP, buff=0.3)
        static = VGroup(
            tx(LANE_SC, 24, ORANGE).move_to([X0 - 0.7, Y_SC, 0]),
            tx(LANE_TC, 24, BLUE).move_to([X0 - 0.7, Y_TC, 0]),
            tx(MEM_LAB, 24, GREY_B).move_to([X0 - 0.7, -0.9, 0]),
            tx(MEM_NOTE, 22, GREY_B).move_to([X0 - 0.7, -1.3, 0]),
            Line([X0, Y0, 0], [XE1 + 0.1, Y0, 0], color=GREY, stroke_width=3),
            Line([X0, YL, 0], [XE1 + 0.1, YL, 0], color=RED, stroke_width=4),
            tx(LIMIT, 22, RED).next_to([XE1 + 0.1, YL, 0], RIGHT, buff=0.12),
        )
        body = always_redraw(scene_at)

        # 跟着几何走的字（建一次，只挪位置、切透明度）
        seg_t = [tx(s, 24, GREY_B) for s in SEGS]
        rel_t, done_t = tx(REL, 22, WHITE), tx(DONE, 22, WHITE)
        hoist_t = tx(HOIST_LAB, 22, WHITE, True)
        freed_t = tx(FREED_LAB, 22, GREEN)

        def place(_m=None):
            m = mt.get_value()
            g = geom(m)
            c = cur(g)
            for t_, (a, b) in zip(seg_t, ((X0, g["xb"]), (g["xb"], g["xo"]), (g["xo"], g["xe"]))):
                t_.move_to([(a + b) / 2, Y_SEG, 0])
            e0 = g["d"][0] + g["r"]
            rel_t.move_to([e0 - D, Y_TC - H_TC / 2 - 0.3, 0]).set_opacity(1.0 if c >= e0 - D else 0.0)
            done_t.move_to([g["d"][0], Y_TC - H_TC / 2 - 0.3, 0]).set_opacity(1.0 if c >= g["d"][0] else 0.0)
            # 「挪」是两幕之间的事：过渡时两组位置靠得很近，放行/完成两字分得开就留着
            if done_t.get_left()[0] < rel_t.get_right()[0] + 0.05:
                rel_t.next_to(done_t, LEFT, buff=0.1)
            x1 = X0 + 1.45
            p1 = [x1, Y0 + base_h(x1, g) + H_OR / 2, 0]
            xm = g["xo"] + (HW / 2 + USE_DX) / 2
            p2 = [xm, Y0 + B0 + H_OR + 0.25, 0]
            hoist_t.move_to([lerp(p1[0], p2[0], m), lerp(p1[1], p2[1], m), 0])
            need = lerp(X0 + 2.1, g["oe"], m)
            hoist_t.set_opacity(1.0 if c >= need - 1e-6 else 0.0)
            xf = g["xb"] + 1.6
            freed_t.move_to([xf, Y0 + base_h(xf, g) + H_OR / 2 + 0.1, 0]).set_opacity(min(1.0, max(0.0, (m - 0.7) / 0.3)) if c >= g["xo"] else 0.0)   # 紫块滑过去以后再出现

        movers = VGroup(*seg_t, rel_t, done_t, hoist_t, freed_t)
        place()
        movers.add_updater(place)

        cap = tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.2)
        self.add(title, static, body, movers, cap)

        def new_cap(s, color=GREY_B):
            return tx(s, 26, color).next_to(title, DOWN, buff=0.2)

        def say(s, color=GREY_B, extra=()):
            nonlocal cap
            n = new_cap(s, color)
            # 先淡出、空一下、再淡入：两句不叠影；总长仍 0.35 秒，停顿时刻不变
            self.play(Succession(FadeOut(cap, run_time=0.1), Wait(0.15), FadeIn(n, run_time=0.1)),
                      *[a.set_run_time(0.35) for a in extra])
            cap = n

        def hold(t):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        # ── 第一幕：没有屏障，游标走一步
        g0 = geom(0.0)
        span = g0["xe"] - X0
        u_b, u_o = (g0["xb"] - X0) / span, (g0["xo"] - X0) / span
        self.wait(0.5)
        # 字幕换句不跟游标同长慢慢交叉淡化（那样 25% 处两句各半透明叠在一起）：
        # 游标照走，字幕在这段开头「淡出 0.3 → 空 0.25 → 淡入 0.3」，其余时间静止
        def swap(old, new, total):
            return Succession(FadeOut(old, run_time=0.3), Wait(0.25), FadeIn(new, run_time=0.3), Wait(total - 0.85))
        n = new_cap(CAPS[0])
        self.play(u.animate(run_time=2.8, rate_func=linear).set_value(u_b), swap(cap, n, 2.8))
        cap = n
        n = new_cap(CAPS[1])
        self.play(u.animate(run_time=4.0, rate_func=linear).set_value(u_o), swap(cap, n, 4.0))
        cap = n
        self.play(u.animate.set_value(1.0), run_time=1.4, rate_func=linear)
        hold(2.0)

        # ── 第二幕：加一道屏障，同一块挪到优化器段
        say(CAPS[2], YELLOW)
        self.play(mt.animate.set_value(1.0), run_time=2.6, rate_func=linear)
        counters = VGroup(*[tx(s, 24, YELLOW) for s in COUNTERS]).arrange_in_grid(
            rows=2, cols=2, col_alignments="ll", buff=(0.9, 0.22)).to_edge(DOWN, buff=0.3)
        say(CAPS[3], extra=(FadeIn(counters),))
        hold(2.6)
        say(CAPS[4])
        hold(2.2)

        # ── 收尾：倒放回第 0 帧
        self.play(FadeOut(counters), FadeOut(cap), u.animate.set_value(0.0), mt.animate.set_value(0.0),
                  run_time=1.0, rate_func=linear)
        self.remove(counters, cap)
        self.add(new_cap(" "))
        self.play(u.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)   # 推一下 updater
        self.wait(0.5)
        dump_steps("BarrierHoist", marks, self.renderer.time)


if __name__ == "__main__":
    print("W1 %.3f  W2 %.3f  INNER %.3f→%.3f  R2 %.3f  XE1 %.2f  P %.2f" %
          (W1, W1 - NB * (R1 - R2), INNER1, INNER2, R2, XE1, P))
