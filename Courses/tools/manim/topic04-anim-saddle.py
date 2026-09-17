# -*- coding: utf-8 -*-
r"""专题四 · 鞍点的 3D 曲面 ——&#160;`fig-saddle` Ⓐ 配的那段 6 秒动画

⭐⭐⭐ 2026-09-18 现场：「你看看三蓝一棕的绘图工具好不好用，
  要是好用的话你就多用用。」——&#160;实测下来结论分两半：

  ⛔ **manim 不能替换我们的静态图。** 它的 `--format` 只接受
    png / gif / mp4 / webm / mov ——&#160;**没有 SVG**。
    而我们那套图是内联 SVG，靠它换来的东西不只是好看：
    文字可选可搜、屏幕阅读器能读（每张都有 aria）、投屏无损缩放、
    进 git 能 diff，**而且全部 lint 都建立在「它是 SVG」上**
    （字号、撞车、图内节号、墨水体检）。换成 PNG 这些全丢。

  ⭐⭐ **但它在另一头无可替代：它本来就是拍动画的。**
    而 3B1B 的杀手锏从来就是「**动起来**」，不是静态构图。

⭐⭐⭐ 这一段是**增量最大**的那个用例：
  `fig-saddle` Ⓐ 只能画**两个剖面**，
  而「一个方向上翘、另一个方向下沉」这件事 ——&#160;**天生是三维的**。
  二维的图讲得出结论，讲不出那个形状。

⛔ 三条刻意的取舍：
  ① **一个字都不放。** 文字由页面上的图注和静态图承担 ——&#160;
     视频里的字既不能选中也不能被读屏，放进去等于把可访问性往回退。
  ② **配色跟 `fig-saddle` 对齐**：红 ＝ 往上（走不了），绿 ＝ 往下（能走出去）。
     ⭐ 两条剖面曲线是**真画在曲面上**的，不是贴上去的。
  ③ **整圈旋转，无缝循环** ——&#160;转满 360° 正好回到起点，
     所以页面里可以 `loop` 而不会看到跳帧。

📌 渲染（需要 `~/.venvs/manim`，见 tools/manim/README 那一节注释）：
    ~/.venvs/manim/bin/manim --format=mp4 -qh --media_dir /tmp/manim-out \
        tools/manim/topic04-anim-saddle.py Saddle
  产物拷到 `WebPages/media/topic04-saddle.mp4`。
"""
import numpy as np
from manim import (ThreeDScene, Surface, Dot3D, ParametricFunction, VGroup,
                   WHITE, DEGREES, config)

# ⭐ 跟 topic03_draw 的主色对齐（那边 RD / GR 就是这两个值）
RD_ = "#d93025"          # 往上 ——&#160;走不了
GR_ = "#1e8e3e"          # 往下 ——&#160;还能走
BL_ = "#4285f4"
INK_ = "#202124"

SPAN = 2.0               # 曲面在 u / v 上的范围
K = 0.35                 # z ＝ K(u² − v²)
TURN = 8.0               # 转满一圈用几秒（＝ 视频时长）


def z(u, v):
    return K * (u * u - v * v)


# ⭐ 这一条是整段动画的全部内容，写成 assert 钉住：
#   沿 u 走 loss **上升**，沿 v 走 loss **下降** ——&#160;这才叫鞍点。
assert z(SPAN, 0) > 0 > z(0, SPAN), "不是鞍面：两个方向得一升一降"
assert abs(z(0, 0)) < 1e-12, "鞍点本身的高度应当是 0"


class Saddle(ThreeDScene):
    def construct(self):
        self.camera.background_color = WHITE

        surf = Surface(
            lambda u, v: np.array([u, v, z(u, v)]),
            u_range=[-SPAN, SPAN], v_range=[-SPAN, SPAN],
            resolution=(28, 28), fill_opacity=0.62,
            checkerboard_colors=[BL_, "#a8c7fa"],
            stroke_width=0.35, stroke_color="#5f6368",
        )

        # ⭐ 两条**真画在曲面上**的剖面：红的往上翘，绿的往下沉
        up = ParametricFunction(
            lambda t: np.array([t, 0.0, z(t, 0.0) + 0.02]),
            t_range=[-SPAN, SPAN], color=RD_, stroke_width=7)
        down = ParametricFunction(
            lambda t: np.array([0.0, t, z(0.0, t) + 0.02]),
            t_range=[-SPAN, SPAN], color=GR_, stroke_width=7)

        ball = Dot3D(np.array([0.0, 0.0, 0.02]), color=INK_, radius=0.085)

        self.set_camera_orientation(phi=66 * DEGREES, theta=-55 * DEGREES, zoom=1.15)
        self.add(surf, VGroup(up, down), ball)
        # ⭐ 转满一整圈 → 首尾同帧 → 页面里 loop 起来看不出接缝
        self.begin_ambient_camera_rotation(rate=2 * np.pi / TURN)
        self.wait(TURN)
