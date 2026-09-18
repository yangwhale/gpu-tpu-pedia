# -*- coding: utf-8 -*-
r"""专题四 · 梯度下降的「降维阶梯」——&#160;一维 → 二维 → 不再画空间

⭐⭐⭐ 2026-09-18 现场：「你抄他的故事线、讲法、绘图代码，
  然后自己画自己的那种可以动的图 ——&#160;梯度下降肯定是有动的图。」

⭐⭐ 这一段的**骨架是从 3Blue1Brown 那一集的类名序列读出来的**（不是抄画面）。
  `_2017/nn/part2.py` 里 61 个 Scene，排下来正好是一条叙事线，其中最关键的一段：

      SingleVariableCostFunction   →  先在**一维**上讲通
      TwoVariableInputSpace        →  升到**二维**看形状
      CostSurface
      ConfusedAboutHighDimension   →  **主动承认**：再往上想不出来了
      NonSpatialGradientIntuition  →  于是**不再画空间**

  ⭐⭐⭐ 最后那一步是整段里最值得学的手法：**高维不画成空间，画成一列数** ——&#160;
    他用颜色表示每个分量的**符号**、长度表示**大小**。
    「想象三千亿维」是不可能的；「看一列条在动」是可能的。

⛔ 只学结构，画面全是我们自己的：
  · 他的 loss 是 MNIST 网络真跑出来的；我们的是**自己造的一条一维曲线**，
    造它的唯一要求是「有两个深浅不同的谷」——&#160;脚本自己扫出谷底位置。
  · 二维那一幕是**真的在算等高线和梯度**，不是画个示意的漩涡。
  · 第三幕那一列条，长度和颜色都来自**真实的梯度分量**。
  ⚠️ 3b1b/videos 的内容是 CC BY-NC-SA 4.0 ——&#160;
    所以**一行代码、一个画面都没有复制**，这里借的是「先一维、再二维、
    然后承认想不出来、最后换一种表示」这条思路。

⛔ 沿用本讲动画的四条规矩：
  ① 一个字都不放（文字交给页面图注）；
  ② 静态图排在它上面，视频加载不出来也不影响读懂；
  ③ 首尾同一帧，`loop` 不跳；
  ④ 数据全部当场算。

📌 渲染：
    ~/.venvs/manim/bin/manim --format=mp4 -qh --media_dir /tmp/manim-out \
        tools/manim/topic04-anim-descend.py Descend
    ffmpeg -y -i /tmp/manim-out/videos/topic04-anim-descend/1080p60/Descend.mp4 \
        -vf "scale=960:-2" -c:v libx264 -crf 30 -preset slow \
        -pix_fmt yuv420p -movflags +faststart -an \
        WebPages/media/topic04-descend.mp4
"""
import math

import numpy as np
from manim import (Scene, VGroup, Dot, Line, Rectangle, ParametricFunction,
                   ValueTracker, always_redraw, FadeIn, FadeOut, WHITE,
                   ORIGIN, UP, DOWN, LEFT, RIGHT, linear, config)

RD_, GR_, BL_, GY_, INK_ = "#d93025", "#1e8e3e", "#4285f4", "#c3c7cb", "#202124"

# ── 幕一：一条自己造的一维 loss，两个深浅不同的谷 ──────────────────
def f1(x):
    # ⛔ 第一版是 0.10x² + 0.95cos(1.15x) ——&#160;**偶函数**，
    #   于是两个谷必然左右对称、深浅一模一样，「落进不同的谷」就没戏了。
    #   ⭐ 断言当场挡下来了。判据：**断言挡住时先看它挡的是什么** ——&#160;
    #     这次挡的不是阈值太严，是我造的函数本身对称。加一个奇次项破对称。
    return 0.10 * x * x + 0.13 * x + 0.95 * math.cos(1.15 * x)


def g1(x):
    return 0.20 * x + 0.13 - 1.0925 * math.sin(1.15 * x)


# ⭐ 谷底让脚本自己扫，不写死
_MINS = [i / 400.0 for i in range(-2400, 2400)
         if g1(i / 400.0) < 0 < g1((i + 1) / 400.0)]
# ⭐ 两个谷就够讲「落进哪个取决于起点」——&#160;第一版要求三个，被这条断言挡下了。
#   ⛔ 放宽数量之后，**深浅拉不拉得开**交给下面那两条断言去卡，别一起放宽。
assert len(_MINS) >= 2, "这条曲线上至少要有两个谷，现在 %d" % len(_MINS)
DEEP = min(_MINS, key=f1)
SIDE = [m for m in _MINS if abs(m - DEEP) > 1.5]
assert SIDE, "得有一个不是最深的谷，不然「落在哪取决于起点」就没画面"
SHALLOW = min(SIDE, key=lambda m: abs(m - DEEP))
assert f1(SHALLOW) > f1(DEEP) + 0.2, "两个谷的深浅要拉开"


def roll1(x0, lr=0.28, steps=110):
    xs, x = [], x0
    for _ in range(steps):
        xs.append(x)
        x -= lr * g1(x)
    return xs


TRJ_A, TRJ_B = roll1(-4.6), roll1(4.9)
# ⭐ 这一幕的论点：同一条曲线、同一个规则，**落点不同**
assert abs(TRJ_A[-1] - TRJ_B[-1]) > 1.5, "两条要落进不同的谷，否则没戏"


# ── 幕二：二维，等高线 ＋ 真算梯度 ─────────────────────────────────
def f2(x, y):
    return 0.10 * x * x + 0.26 * y * y + 0.55 * math.cos(1.2 * x) * math.cos(1.1 * y)


def g2(x, y):
    return (0.20 * x - 0.66 * math.sin(1.2 * x) * math.cos(1.1 * y),
            0.52 * y - 0.605 * math.cos(1.2 * x) * math.sin(1.1 * y))


def roll2(p0, lr=0.30, steps=130):
    pts, (x, y) = [], p0
    for _ in range(steps):
        pts.append((x, y))
        gx, gy = g2(x, y)
        x -= lr * gx
        y -= lr * gy
    return pts


TRJ2 = roll2((-3.4, 2.5))
assert f2(*TRJ2[-1]) < f2(*TRJ2[0]) - 0.8, "二维那条得真的降下去"


# ── 幕三：不再画空间 ——&#160;一列分量条 ────────────────────────────
NDIM = 14
_rng = np.random.default_rng(7)
_W = _rng.standard_normal(NDIM) * 1.3
_A = np.abs(_rng.standard_normal(NDIM)) * 0.55 + 0.25   # 各分量的曲率


def wk(k):
    """第 k 步的参数向量 ——&#160;各分量按自己的曲率指数收缩。"""
    return _W * np.exp(-_A * k * 0.055)


assert np.abs(wk(60)).max() < np.abs(wk(0)).max() * 0.6, "那一列条得看得出在缩"
assert (_W > 0).any() and (_W < 0).any(), "得同时有正有负，颜色才有对比"

T1, T2, T3 = 4.6, 5.0, 5.2          # 三幕时长
TEND = T1 + T2 + T3 + 0.8           # 末尾留一点复位


class Descend(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        t = ValueTracker(0.0)

        # ═══ 幕一 ═══════════════════════════════════════════════
        XL, XR = -6.0, 6.0
        SX, SY, Y0 = 0.92, 0.52, -2.3

        def p1(x):
            return np.array([x * SX, Y0 + f1(x) * SY, 0])

        curve = ParametricFunction(p1, t_range=[XL, XR, 0.05],
                                   stroke_color=GY_, stroke_width=3)

        def ball(trj, col):
            def mk():
                s = t.get_value()
                if s > T1:
                    return Dot(radius=0.001, fill_opacity=0)
                k = min(len(trj) - 1, int(s / T1 * (len(trj) - 1)))
                return Dot(radius=0.11, color=col).move_to(p1(trj[k]))
            return always_redraw(mk)

        act1 = VGroup(curve)
        # ⭐ 留着引用 ——&#160;末尾复位时要按**同样的顺序**再 add 一遍来恢复 z 序，
        #   见文件末尾。`always_redraw` 的返回值不留引用就再也拿不到了。
        ball_a, ball_b = ball(TRJ_A, RD_), ball(TRJ_B, BL_)
        self.add(act1, ball_a, ball_b)

        # ═══ 幕二：等高线 ＋ 一条真轨迹 ═════════════════════════
        CS = 0.80

        def p2(x, y):
            return np.array([x * CS, y * CS, 0])

        rings = VGroup()
        for lv in (0.4, 1.0, 1.8, 2.8, 4.0, 5.4):
            for sgn in (1,):
                pts = []
                for a in range(0, 361, 4):
                    th = math.radians(a)
                    lo, hi = 0.05, 7.0
                    for _ in range(26):                # 二分找等值点
                        mid = (lo + hi) / 2
                        if f2(mid * math.cos(th), mid * math.sin(th)) < lv:
                            lo = mid
                        else:
                            hi = mid
                    r = (lo + hi) / 2
                    pts.append(p2(r * math.cos(th), r * math.sin(th)))
                ring = VGroup()
                for u, v in zip(pts, pts[1:]):
                    ring.add(Line(u, v, stroke_color=GY_, stroke_width=1.6))
                rings.add(ring)
        rings.set_opacity(0)
        self.add(rings)

        def path2():
            s = t.get_value()
            if not (T1 <= s < T1 + T2):
                return Dot(radius=0.001, fill_opacity=0)
            k = min(len(TRJ2) - 1, int((s - T1) / T2 * (len(TRJ2) - 1)))
            g = VGroup()
            for u, v in zip(TRJ2[:k + 1], TRJ2[1:k + 2]):
                g.add(Line(p2(*u), p2(*v), stroke_color=GR_, stroke_width=3.4))
            g.add(Dot(radius=0.11, color=GR_).move_to(p2(*TRJ2[k])))
            return g

        self.add(always_redraw(path2))

        # ═══ 幕三：一列分量条 ═══════════════════════════════════
        BW, BGAP = 0.30, 0.055

        def bars():
            s = t.get_value()
            if s < T1 + T2:
                return VGroup()
            k = int((s - T1 - T2) / T3 * 60)
            w = wk(min(k, 60))
            g = VGroup()
            y0 = (NDIM - 1) * (BW + BGAP) / 2
            for i, v in enumerate(w):
                ln = max(abs(v) * 1.5, 0.02)
                r = Rectangle(width=ln, height=BW,
                              stroke_width=0, fill_opacity=0.9,
                              fill_color=BL_ if v > 0 else RD_)
                r.move_to(np.array([ln / 2 * (1 if v > 0 else -1),
                                    y0 - i * (BW + BGAP), 0]))
                g.add(r)
            g.add(Line(np.array([0, y0 + 0.45, 0]),
                       np.array([0, y0 - NDIM * (BW + BGAP) - 0.1, 0]),
                       stroke_color=INK_, stroke_width=1.6))
            return g

        self.add(always_redraw(bars))

        # ── 幕与幕之间的切换：真的换画面，不是叠加 ──────────────
        self.play(t.animate.set_value(T1), run_time=T1, rate_func=linear)
        self.play(FadeOut(act1), rings.animate.set_opacity(1), run_time=0.5)
        self.play(t.animate.set_value(T1 + T2), run_time=T2 - 0.5,
                  rate_func=linear)
        self.play(rings.animate.set_opacity(0), run_time=0.5)
        self.play(t.animate.set_value(T1 + T2 + T3), run_time=T3 - 0.5,
                  rate_func=linear)
        # ── 复位：让最后一帧＝第一帧 ─────────────────────────────
        # ⛔ 第一版写成「先 FadeIn(act1)，再把 t 归零」——&#160;
        #   于是曲线淡回来的那 0.4 秒里，**第三幕那列条还挂在上面**，
        #   最后一帧是「曲线 ＋ 一列条」，跟第一帧对不上，loop 当场跳。
        # ⭐ 判据：**「首尾同一帧」要检查的是<最后一帧的全部内容>，
        #   不是「我有没有写复位这一步」** ——&#160;复位漏掉一个图层，它就不成立。
        # ⛔ 第二版写成「t.set_value(0) ＋ act1.set_opacity(1) ＋ wait(0.5)」——&#160;
        #   量出来**一点没变**：末帧还是那一列条。两个症状各对应一个 bug：
        #   ① 末帧**多**了条 ——&#160;`self.wait()` 没有驱动 `always_redraw` 的 updater，
        #      所以 t 归零了、画面还停在旧帧。**只有 play 一定会走一遍 updater。**
        #   ② 末帧**少**了曲线 ——&#160;`FadeOut(act1)` 把 act1 **移出了 scene**，
        #      `set_opacity(1)` 只改属性不改归属，得 `self.add` 加回来。
        # ⭐⭐ 判据：**首尾一致要量「差异像素占比」，不能量「平均像素差」** ——&#160;
        #   这张画面 98.5% 是白底，平均差被稀释到 1.26，看着像零，其实差着整整一幕。
        #   稀疏画面上任何按全图取平均的指标都没有判别力。
        # ⛔ 第三版写 `self.add(act1)`，量出来还剩 25.3% ——&#160;内容对了，
        #   但**末帧的球被曲线盖住了一半**。`Scene.add` 是追加到显示列表尾部
        #   ＝画在最上层；而首帧时球是在曲线之后 add 的、本来在曲线之上。
        # ⭐ 判据：**「把东西加回 scene」和「加回它原来的图层」是两件事。**
        # ⛔ 这个版本的 `Scene` 没有 `add_to_back`（当场 AttributeError 崩掉）。
        # ⭐⭐ 不用去找那个 API ——&#160;`Scene.add` 会**先把重复的摘掉、再追加**，
        #   所以**按最初那一行同样的顺序重新 add 一遍**就恢复了原本的 z 序。
        #   这么写还不依赖 manim 版本。
        t.set_value(0.0)
        act1.set_opacity(1)
        self.add(act1, ball_a, ball_b)
        # ⭐ 恒等动画：t 已经是 0，这一步只为**强制走一遍 updater**。
        #   这么写对「wait 到底驱不驱动 updater」这个我没验过的机制是鲁棒的 ——&#160;
        #   两种情况都能得到正确的末帧。
        self.play(t.animate.set_value(0.0), run_time=1 / 30)
        self.wait(0.5)
