# -*- coding: utf-8 -*-
r"""专题四 · 一路乘下去 ——&#160;`fig-vanish` 配的那段动画（候选排名第 3）

⭐⭐⭐ 第一版被我自己否了，值得记下来为什么：
  只画了**对数轴上那把扇子** ——&#160;而指数函数取完对数就是直线，
  于是画面是三条直线张开。⛔ 那跟它上面那张静态图**几乎一模一样**，
  动起来只多给了「谁先谁后」，不值一段视频。
  ⭐ 判据：**一段动画要活下来，得给出静态图给不了的东西。**
    「同样的内容播一遍」不算 ——&#160;那只是把读者的眼睛替他移了一次。

⭐⭐⭐ 这一版找到的增量，是**对数轴自己的代价**：
  **对数轴让你看得见十个数量级，可也正好让你感觉不到十个数量级有多大。**
  所以画两条轨，共用同一根横轴（层号）、同一根扫描线：
    上轨 ＝ 对数刻度：三条直线**才刚刚张开**
    下轨 ＝ 线性刻度：同一时刻，**蓝的早已顶出画框，红的已经贴在地上**
  ⭐ 落点：看上轨你以为「它们还差不多」，看下轨才知道**早就没法比了**。
    这个反差**只有把两条轨对齐播才成立** ——&#160;静态图并排画两幅，
    读者不会自己去对同一层。

⭐⭐ 数值落点跟静态图同一个：每层 ×0.8 和 ×1.2 **只差两成**，
  六十层后差十个数量级。指数放大的不是梯度，是「每层偏离 1 多少」那个偏差。

⛔ 房规看 skill `manim-teaching-figures` 的「房规」一节 ——&#160;**这里不抄第二份。**
  2026-09-18 松绑过一次（文字那条从「一个字都不放」改成「为讲解服务，不为装饰」），
  而旧版当时被抄进了好几个文件，改一次得满仓库找复述。
  ⭐ 判据：**同一条规矩只留一份正本，别处一律写指针** ——&#160;
    跟 `traps.md` §1.3「复位只能定义在一个地方」是同一条病。

⭐⭐ 动手前那一判（房规要求）：**这一格缺的是「解释」，不是「铺陈」。**
  具体理由，不是套话：
  · **不缺铺陈** ——&#160;决出胜负的三件事（蓝线顶出画框、红线贴地、扇子才张开三成）
    发生在扫描段的**前三分之一以内**，而且是**同时**发生的
    ——&#160;这条不是我说的，下面 `K_MARK < L // 3` 那条 assert 钉着。
    ⛔ 把它拉长到 20 秒，这三件事只会各自变慢，**反差一点都不会变强**，
    因为反差来自「同一时刻」，不来自「看得久」。
  · **缺解释** ——&#160;画面里读不出来的只有一件事：**两条轨的纵轴不是同一种刻度。**
    而整段的落点完全架在这个区别上，它不成立，上下两条轨就只是「两条曲线」。
  ⭐ 所以：只补**坐标系标识**，`T_SWEEP / T_HOLD / T_REW` 与 `loop` 一律不动。

⭐⭐⭐ 为什么坐标系标识不能丢给图注（这一条被质疑过，值得写下来）：
  图注是**看完之后**读的，而「此刻你看的是对数轴」是**观看当下**就要知道的
  ——&#160;它不是事后说明，它是读这张图的前提。
  ⭐ 一张图的刻度该标在图上，这跟图注说了什么无关。
  ⛔ 但也**仅限**这一类：图注里那几句结论（「第 8 层顶出画框」「张开三成」）
    一个字都不搬进画面 ——&#160;房规①「别把静态图已经说清的话再抄一遍」。

⛔⛔ 房规①的硬义务：**`aria-label` 要把画面里出现过的每一句话复述一遍。**
  画面里现在一共四处字：「对数刻度」「线性刻度」、`0.8^k`、`1.2^k`。
  ⚠️ 而 aria-label 在 `tools/topic04-build.py` 里，**不在本文件** ——&#160;
    动了这里的文字就必须同步动那边，否则读屏用户拿不到这四处。

⭐ 复位用 `clock()` 倒着走 ——&#160;内容是累积的，直接清零只会把那一跳
  从接缝挪进片内。跟 `topic04-anim-memtime.py` 同一个手法。
  ⭐ 那四处字是**静态**的（不挂 updater、不读 `clock()`），所以它们在首帧和末帧
    长得一模一样，「首帧 ≡ 末帧」这条不变量原样成立。

📌 渲染：
    ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic04-anim-vanish.py Vanish WebPages/media/topic04-vanish.mp4
"""
import math

import numpy as np
from manim import (Scene, VGroup, Dot, Line, Text, MathTex, ValueTracker,
                   always_redraw, WHITE, linear)

RD_, BL_, GY_, GY2_, INK_ = "#d93025", "#4285f4", "#c3c7cb", "#80868b", "#202124"

L = 60                                   # 层数，跟静态图同一个口径
CASES = ((0.8, RD_), (1.0, GY2_), (1.2, BL_))

# ⭐ 全部当场乘出来：第 k 个点 ＝ 往回传了 k 层之后的相对幅度
RAW = {r: [r ** k for k in range(L)] for r, _ in CASES}
DEC = {r: [k * math.log10(r) for k in range(L)] for r, _ in CASES}

LO, HI = min(min(v) for v in DEC.values()), max(max(v) for v in DEC.values())
assert HI - LO > 9.0, "两头拉不开十个数量级，这一格就没有落点了（现在 %.1f）" % (HI - LO)

XL, XR = -6.2, 6.2                       # 右＝输出层（刚开始传），左＝输入层

# ── 上轨：对数刻度 ────────────────────────────────────────────────
A_MID, A_HALF = 1.95, 1.80
SYA = (A_HALF * 2) / (HI - LO)
A_Y0 = A_MID - (LO + HI) / 2 * SYA

# ── 下轨：线性刻度。VMAX 决定「多大算出框」──────────────────────
B_BOT, B_TOP = -3.55, -0.20
# ⛔ VMAX 是**构图参数也是论点参数**：它同时决定「多大算出框」和
#   「1.0 那条参考线坐多高」。第一版取 6.0，1.0 被压到离地面只有六分之一，
#   红线「从 1 掉到 0」那一段几乎贴着地面走，看不出它掉过。
#   ⭐ 收到 4.0：1.0 抬到四分之一高，红线的坠落看得见；
#     代价是蓝线出框提前到第 8 层 —— 反而更狠。
VMAX = 4.0
SYB = (B_TOP - B_BOT) / VMAX
GONE = 0.02                              # 小于这个就算「肉眼没了」（≈ 1080p 下 3 px）

# ⭐⭐⭐ 这段动画的论点必须被算出来、并且钉死，不能靠我说
K_OUT = next(k for k in range(L) if RAW[1.2][k] > VMAX)      # 蓝的第几层出框
K_GONE = next(k for k in range(L) if RAW[0.8][k] < GONE)     # 红的第几层看不见
K_MARK = max(K_OUT, K_GONE)
SPREAD = (DEC[1.2][K_MARK] - DEC[0.8][K_MARK]) * SYA / (A_HALF * 2)
assert K_MARK < L // 3, \
    "线性轴要在前三分之一就分出胜负，否则反差不够（现在第 %d 层）" % K_MARK
assert SPREAD < 0.35, \
    ("到第 %d 层，对数轴上要还挤在一起（张开 %.0f%%），"
     "『上轨看着差不多、下轨早没法比』才成立" % (K_MARK, SPREAD * 100))

T_SWEEP, T_HOLD, T_REW = 5.4, 1.1, 0.9
T_RESET = T_SWEEP + T_HOLD
T_END = T_RESET + T_REW


def px(k):
    return XR - (XR - XL) * k / (L - 1.0)


def pya(d):
    return A_Y0 + d * SYA


def pyb(v):
    return B_BOT + min(v, VMAX) * SYB


def vat(vals, x):
    """曲线在横坐标 x 处的值 ——&#160;按**画出来的那条折线**线性插值，
    所以放字时算的遮挡跟画面里看到的是同一条线，不是另一条理想曲线。"""
    kf = min(max((XR - x) / (XR - XL) * (L - 1.0), 0.0), L - 1.0)
    k0 = int(kf)
    k1 = min(k0 + 1, L - 1)
    return vals[k0] + (vals[k1] - vals[k0]) * (kf - k0)


class Vanish(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        tt = ValueTracker(0.0)

        def clock():
            t = tt.get_value()
            if t <= T_SWEEP:
                return t
            if t <= T_RESET:
                return T_SWEEP
            return T_SWEEP * (1.0 - (t - T_RESET) / T_REW)

        def cur_k():
            return max(1, int(clock() / T_SWEEP * (L - 1)))

        # ── 底稿 ──────────────────────────────────────────────────
        base = VGroup()
        d = math.ceil(LO / 2.0) * 2                      # 上轨：每两个数量级一条
        while d <= HI:
            base.add(Line(np.array([XL - 0.25, pya(d), 0]),
                          np.array([XR + 0.25, pya(d), 0]),
                          stroke_color=GY_,
                          stroke_width=2.0 if abs(d) < 1e-9 else 1.0))
            d += 2
        # 下轨：一条地面、一条「原始幅度＝1」的参考线、一条**画框上沿**
        # ⭐ 上沿那条是这一轨的关键 ——&#160;蓝线越过它＝出框，
        #   没有这条线，「顶出去」就只是「画到头了」。
        for y, w in ((B_BOT, 2.0), (pyb(1.0), 1.4), (B_TOP, 2.0)):
            base.add(Line(np.array([XL - 0.25, y, 0]), np.array([XR + 0.25, y, 0]),
                          stroke_color=GY_, stroke_width=w))
        self.add(base)

        # ── 两条轨上的三条链，从右往左**被画出来** ─────────────────
        def track(r, col, fy, vals, clip):
            def mk():
                k, g = cur_k(), VGroup()
                for i in range(k):
                    y0, y1 = fy(vals[i]), fy(vals[i + 1])
                    # ⭐ 出框之后不再画 ——&#160;线停在上沿，读成「冲出去了」，
                    #   而不是「贴着顶边继续爬」（那会读成「到顶了就不动了」，
                    #   跟事实完全相反）。
                    if clip and vals[i] > VMAX:
                        break
                    g.add(Line(np.array([px(i), y0, 0]), np.array([px(i + 1), y1, 0]),
                               stroke_color=col, stroke_width=3.6))
                if not (clip and vals[k] > VMAX):
                    g.add(Dot(np.array([px(k), fy(vals[k]), 0]),
                              radius=0.095, color=col))
                return g
            return always_redraw(mk)

        for r, col in CASES:
            self.add(track(r, col, pya, DEC[r], False))
            self.add(track(r, col, pyb, RAW[r], True))

        # ⭐ 一根贯穿两轨的扫描线 ——&#160;它是「同一层」这件事的唯一证据。
        #   两轨的反差全靠它成立，所以它不是装饰。
        self.add(always_redraw(lambda: Line(
            np.array([px(cur_k()), B_BOT - 0.28, 0]),
            np.array([px(cur_k()), pya(HI) + 0.28, 0]),
            stroke_color=INK_, stroke_width=1.8)))

        # ── 坐标系标识 ───────────────────────────────────────────────
        # ⭐⭐ 画面里**只有**这四处字，全是「观看当下」才有用的东西：
        #   两条轨的纵轴是两种刻度，以及这两条链各自是谁。
        #   图注里那些结论（第 8 层、张开三成）一个字都不搬进来。
        # ⛔ 四个都是静态的：不挂 updater、不读 clock() ——&#160;所以首帧和末帧
        #   长得一模一样，「首帧 ≡ 末帧」这条不变量不受影响。
        PAD = 0.07
        MIN_GAP = 0.08

        def occupied(x0, x1, fy, vals_by_r, clip, n=96):
            """这段 x 区间里，**真画出来的**曲线占了哪些 y。"""
            ys = []
            for i in range(n + 1):
                x = x0 + (x1 - x0) * i / n
                for r, _ in CASES:
                    v = vat(vals_by_r[r], x)
                    # ⛔ 出框之后那一截根本没画（track() 里 break 掉了），
                    #   不能当成占位 ——&#160;否则下轨左半边会被一条不存在的线挡光。
                    if clip and v > VMAX:
                        continue
                    ys.append(fy(v))
            return ys

        def place(m, cx, y_want, y_lo, y_hi, obst, name):
            """想放在 y_want；被曲线挡住就在 [y_lo, y_hi] 里挪到最近的空位。

            ⭐ 房规④「参数也算数据」：**区间是我选的（意图），位置是搜出来的
              （安全），断言兜底（证据）** ——&#160;不是手调到「看着对」。
            ⛔ 挪不出来就当场失败，而不是悄悄把字叠在曲线上。
            """
            half = m.height / 2 + PAD
            assert y_hi - y_lo > 2 * half, \
                "「%s」的可放区间比字还窄（%.2f < %.2f）" % (
                    name, y_hi - y_lo, 2 * half)
            best, bd, top_gap = None, 1e9, -1e9
            for i in range(721):
                y = y_lo + half + (y_hi - y_lo - 2 * half) * i / 720.0
                gap = min(abs(y - yp) - half for yp in obst) if obst else 9.9
                top_gap = max(top_gap, gap)
                if gap >= MIN_GAP and abs(y - y_want) < bd:
                    best, bd = y, abs(y - y_want)
            assert best is not None, (
                "「%s」在 [%.2f, %.2f] 里放不下：最宽处离曲线只有 %.3f（要 %.2f）"
                "——&#160;⛔ 别把门槛调小，先看是不是构图挤了" % (
                    name, y_lo, y_hi, top_gap, MIN_GAP))
            m.move_to(np.array([cx, best, 0]))
            return m

        def tag(s, col, size, math=False):
            m = MathTex(s, color=col, font_size=size) if math else \
                Text(s, color=col, font_size=size)
            # ⭐ 垫一层白底：上轨每两个数量级有一条灰网格线，横穿整幅。
            #   不垫的话，要么字被线划掉，要么为了躲线把字顶到画框边上。
            m.add_background_rectangle(color=WHITE, opacity=1.0, buff=0.06)
            return m

        def up_obst(x0, x1):
            return occupied(x0, x1, pya, DEC, False)

        def lo_obst(x0, x1):
            return occupied(x0, x1, pyb, RAW, True)

        # ① 两条轨各自的刻度名 ——&#160;整段的反差就架在这一个区别上。
        # ⛔⛔ 横向要**让开扫描线的两个驻留位**，否则字的白底会把扫描线切一刀：
        #   · 右端 px(1) ＝ 首帧位置 ——&#160;而首帧正是循环接缝上被看最多的那一帧
        #   · 左端 XL ＝ 扫完之后 hold 那一秒多的停留位
        #   ⭐ 扫描线扫过中间时也会被挡一下，但那是一闪而过，不落在任何静止帧上。
        m = tag("对数刻度", INK_, 21)
        rx = px(1) - 0.16
        self.add(place(m, rx - m.width / 2, 3.45, pya(0.0) + 0.15, 3.92,
                       up_obst(rx - m.width, rx), "对数刻度"))
        m = tag("线性刻度", INK_, 21)
        lx = XL + 0.25
        self.add(place(m, lx + m.width / 2, -0.62, B_TOP - 1.55, B_TOP - 0.03,
                       lo_obst(lx, lx + m.width), "线性刻度"))

        # ② 两条链的记号 ——&#160;k ＝ 往回传了几层，跟 RAW / DEC 同一个 k。
        #   ⭐ 贴着各自那条线的**左端**放：那里两条线离得最远，不会认错主人；
        #     往内侧偏（蓝的往下、红的往上），避开画框边缘。
        CX = -5.5
        for r, col, inward in ((1.2, BL_, -0.44), (0.8, RD_, +0.44)):
            m = tag("%.1f^{k}" % r, col, 30, math=True)
            want = pya(vat(DEC[r], CX)) + inward
            self.add(place(m, CX, want, pya(LO) - 0.10, 3.92,
                           up_obst(CX - m.width / 2, CX + m.width / 2),
                           "%.1f^k" % r))

        self.play(tt.animate.set_value(T_END), run_time=T_END * 1.9,
                  rate_func=linear)
