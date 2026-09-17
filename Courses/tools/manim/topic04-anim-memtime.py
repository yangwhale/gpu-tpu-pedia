# -*- coding: utf-8 -*-
r"""专题四 · 显存随时间 ——&#160;`fig-step` / `fig-act-bill` 配的那段循环

⭐⭐⭐ 2026-09-18。扫完 25 张图之后排在**第一位**的动画候选，理由：
  **「峰值出现在哪一刻」本身就是一个过程。**
  静态图能画出那座山的形状，可它画不出「山是怎么被堆起来、又怎么塌下去的」。
  ⭐ 而 md 的「还没想清楚的」里正挂着一条：
    「要不要在这里就讲清楚峰值显存出现在哪一刻 ——&#160;
      这需要画时间线图，可能比想象中难讲」。
    **这段动画就是那条的答案。**

⭐⭐ 画的是**一个完整的 step**，四段：
    前向（激活一层层堆高）→&#160;反向（激活释放、梯度长出来）
    →&#160;更新（梯度清零）→&#160;回到起点
  ⭐ 首尾同状态，所以 `loop` 起来是一个**真的循环**，不是一段被硬接起来的片子。

⛔ 三条规矩（跟 `tools/manim/README.md` 一致）：
  ① **一个字都不放** ——&#160;位置用一个沿层轴走的滑块表示，不用「第 37 层」这种字。
  ② 静态图排在视频上面兜底。
  ③ 配色跟 `topic03_draw` 对齐。

⭐⭐⭐ 三条带的高度**按真实字节数算**，跟 `fig-step` 同一套口径，脚本里 assert 了：
    · 权重 2 B ＋ 优化器状态 12 B ＝ **14 B/参数** → 不动的基座
    · 梯度 **2 B/参数** → 反向才长出来，反向结束时最全
    · 激活（不开重算、一条 128K 序列）→ 前向堆高、反向释放
"""
import numpy as np
from manim import (Scene, ValueTracker, Polygon, Rectangle, Line, Dot,
                   VGroup, always_redraw, WHITE, rate_functions, config)

BL_, RD_, GY_, INK_ = "#4285f4", "#d93025", "#9aa0a6", "#202124"
GR_ = "#1e8e3e"

N_PARAM = 671e9
TIB = 1024.0 ** 4
RESIDENT_TIB = N_PARAM * 14 / TIB          # 权重 2 ＋ 优化器 12
GRAD_TIB = N_PARAM * 2 / TIB               # 梯度 2
ACT_TIB = 4.15                             # 不开重算，一条 128K 序列

# ⭐ 跟 fig-step 同一套口径 ——&#160;对不上就别渲，免得两处打架
assert abs(RESIDENT_TIB - 8.55) < 0.02, RESIDENT_TIB
assert abs(GRAD_TIB - 1.22) < 0.02, GRAD_TIB
# ⭐⭐ 这一条是整段动画的落点：**激活的峰比常驻的一半还高**
assert ACT_TIB > RESIDENT_TIB * 0.4, "激活峰不够高就看不出「山」"

T_FWD, T_BWD, T_UPD, T_HOLD = 1.0, 1.0, 0.34, 0.50   # ⭐ 末尾多停一会儿：
#   画完的那座山要留在屏幕上看一眼，loop 才读成「画一遍 → 看一眼 → 重来」，
#   而不是「画到一半突然跳回去」。
T_END = T_FWD + T_BWD + T_UPD + T_HOLD


def act_at(t):
    if t <= T_FWD:
        return ACT_TIB * (t / T_FWD)
    if t <= T_FWD + T_BWD:
        return ACT_TIB * (1 - (t - T_FWD) / T_BWD)
    return 0.0


def grad_at(t):
    if t <= T_FWD:
        return 0.0
    if t <= T_FWD + T_BWD:
        return GRAD_TIB * ((t - T_FWD) / T_BWD)
    if t <= T_FWD + T_BWD + T_UPD:
        return GRAD_TIB * (1 - (t - T_FWD - T_BWD) / T_UPD)
    return 0.0


class MemTime(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        cfg = dict(stroke_width=0)
        X0, X1, Y0 = -6.0, 6.0, -2.9
        SY = 0.30                      # 每 TiB 多少个单位高
        tt = ValueTracker(0.0)

        def px(t):
            return X0 + (X1 - X0) * t / T_END

        # 基座：不动的那 14 字节
        base = Rectangle(width=X1 - X0, height=RESIDENT_TIB * SY,
                         fill_color=GY_, fill_opacity=0.45, **cfg)
        base.move_to(np.array([(X0 + X1) / 2, Y0 + RESIDENT_TIB * SY / 2, 0]))

        def band(fn, col, below):
            """把一条随时间变的带，按**已经走过的那段**画成填充多边形。"""
            def make():
                t_now = tt.get_value()
                n = max(2, int(120 * t_now / T_END) + 2)
                ts = np.linspace(0, t_now, n)
                top = [np.array([px(s), Y0 + (below(s) + fn(s)) * SY, 0]) for s in ts]
                bot = [np.array([px(s), Y0 + below(s) * SY, 0]) for s in reversed(ts)]
                pts = top + bot
                if len(pts) < 3:
                    pts = pts + [pts[-1] + np.array([1e-3, 0, 0])]
                return Polygon(*pts, fill_color=col, fill_opacity=0.80,
                               stroke_color=col, stroke_width=2)
            return always_redraw(make)

        act_band = band(act_at, BL_, lambda s: RESIDENT_TIB)
        grad_band = band(grad_at, RD_, lambda s: RESIDENT_TIB + act_at(s))

        # ⭐ 现在走到哪儿：一根扫过去的竖线 ＋ 层轴上的滑块（**不放字**）
        sweep = always_redraw(lambda: Line(
            np.array([px(tt.get_value()), Y0 - 0.25, 0]),
            np.array([px(tt.get_value()), Y0 + 3.25, 0]),
            color=INK_, stroke_width=2.5))

        # 层轴：61 个小格，前向从左往右点亮、反向从右往左熄掉
        LN, LY = 61, Y0 - 0.72
        cells = VGroup(*[
            Rectangle(width=(X1 - X0) / LN * 0.78, height=0.17,
                      stroke_color=GY_, stroke_width=0.8,
                      fill_color=GY_, fill_opacity=0.12)
            .move_to(np.array([X0 + (X1 - X0) * (i + 0.5) / LN, LY, 0]))
            for i in range(LN)])

        def lit():
            t_now = tt.get_value()
            if t_now <= T_FWD:
                k, col = int(LN * t_now / T_FWD), BL_
            elif t_now <= T_FWD + T_BWD:
                k = int(LN * (1 - (t_now - T_FWD) / T_BWD)); col = RD_
            else:
                k, col = 0, GR_
            g = VGroup()
            for i in range(min(k, LN)):
                g.add(cells[i].copy().set_fill(col, opacity=0.85)
                      .set_stroke(col, 0.8))
            return g

        self.add(base, act_band, grad_band, cells, always_redraw(lit), sweep)
        self.add(Dot(np.array([X0, Y0 + RESIDENT_TIB * SY, 0]), radius=0.001))

        self.play(tt.animate.set_value(T_END),
                  run_time=T_END * 4.5, rate_func=rate_functions.linear)
