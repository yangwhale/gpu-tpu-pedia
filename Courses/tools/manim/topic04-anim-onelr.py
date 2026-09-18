# -*- coding: utf-8 -*-
r"""专题四 · 一个学习率伺候不了所有参数 ——&#160;`fig-onelr` 配的动画（候选第 4）

⭐⭐⭐ 这一段刻意**不用前四段那个套路**（「两个东西同时跑，比谁快」）。
  静态图已经把 η 大和 η 小各画了一条轨迹，再让它们动一遍，
  增量只有「谁先谁后」——&#160;上一段 `vanish` 刚在这儿栽过。

⭐⭐⭐ 这一段的增量来自**第三个维度：学习率自己**。
  把 η 做成一根可以走的轴，从小连续扫到大，于是观众看到的不是两条轨迹，
  是**一条轨迹的形态在连续演变**：
      贴着谷底慢慢蹭 → 走得刚好 → 开始左右横跳 → 炸出画面
  ⛔ 静态图给不了这个 ——&#160;它只能挑两三个 η 各画一条，
    而**「中间发生了什么」「拐点在哪」正是这一节要讲的东西**。

⭐⭐ 落点跟静态图同一个，但这里是**看**出来的不是读出来的：
  **η 只要变动一成，就从收敛翻成发散。**
  ⛔ 这句话第一版我写成「能用的区间窄得吓人 ——&#160;从开始横跳到炸只有一条线宽」，
    **然后被自己的数打了脸**：从「开始横跳」到「炸」实测占标尺 **57%**，
    一点都不窄。我把两件事混成了一件。
  ⭐⭐ 判据：**叙事跟着数走，不是调参数去迁就叙事。**
    真正窄的是**门槛那一带** ——&#160;±5% 两条线在标尺上只隔 ~10%，
    而跨过去就是收敛与发散的分界。所以标尺上标的是那两条，不是横跳起点。

⭐ 还有一条静态图说不清的：**过了门槛不是「差一点」，是性质变了。**
  门槛以下每步都在缩，以上每步都在放大 ——&#160;滑块越线那一瞬间画面会翻脸。

⛔ 四条房规（见 skill `manim-teaching-figures`）：
  ① 一个字都不放；② 静态图排在它上面；③ 首帧 ≡ 末帧；④ 数据当场算。

⭐⭐ 复位这次不用造：**参数扫描天然首尾闭合** ——&#160;η 扫上去再扫回来
  就是一个三角波，末帧＝首帧是**波形自己保证的**，不需要 `clock()` 那套补丁。
  （`memtime` / `vanish` 那两段是累积式的，才必须另造复位段。）

📌 渲染：
    ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic04-anim-onelr.py OneLR WebPages/media/topic04-onelr.mp4
"""
import math

import numpy as np
from manim import (Scene, VGroup, Dot, Line, ValueTracker, always_redraw,
                   WHITE, linear)

RD_, BL_, GR_, GY_, GY2_, INK_ = ("#d93025", "#4285f4", "#1e8e3e",
                                  "#c3c7cb", "#80868b", "#202124")

# ── 谷的形状：跟静态图同一套口径 ──────────────────────────────────
A1, A2 = 25.0, 1.0                # A1 ＝ 陡方向，A2 ＝ 平方向
ETA_MAX = 2.0 / A1                # ⭐ 发散门槛，闭式：沿陡方向每步放大 |1 − ηA₁|
W0 = (1.5, 8.0)                   # 起点 (w1 陡, w2 平)
NSTEP = 26

# ⭐ 下端取得够小，平方向才「蹭」得出来（26 步只走掉三分之一）
ETA_LO, ETA_HI = ETA_MAX * 0.20, ETA_MAX * 1.16


def run(eta, n=NSTEP):
    out, w1, w2 = [W0], W0[0], W0[1]
    for _ in range(n):
        w1 -= eta * A1 * w1
        w2 -= eta * A2 * w2
        out.append((w1, w2))
    return out


# ⭐⭐⭐ 这段动画的三个论点，全部当场跑出来钉住 ——&#160;不是我写上去的
_lo = run(ETA_LO)
_hi = run(ETA_HI)
assert abs(_lo[-1][0]) < abs(W0[0]) * 0.5, "扫描下端必须是收敛的，不然没有对照"
assert abs(_hi[-1][0]) > abs(W0[0]) * 5, "扫描上端必须真的炸，不然「越线翻脸」是编的"
assert _lo[-1][1] > W0[1] * 0.5, \
    "⭐ 下端那条在**平方向**必须还没走到一半 ——&#160;「两头都不满意」的另一头"
assert ETA_LO < ETA_MAX < ETA_HI, "扫描区间必须跨过门槛，否则看不到翻脸那一下"

# ⭐ 颜色分三段用的分界（**不是**「窄」的证据，见文件头那条教训）：
#   η 超过 ETA_MAX/2 时 1−ηA₁ 变负，陡方向从「单调缩」变成「左右横跳」。
_ETA_OSC = ETA_MAX * 0.5

# ⭐⭐ 真正窄的那一带：门槛 ±5%。一边收敛、一边发散，而它们几乎贴在一起。
ETA_OK, ETA_BAD = ETA_MAX * 0.95, ETA_MAX * 1.05
BAND = (ETA_BAD - ETA_OK) / (ETA_HI - ETA_LO)
assert BAND < 0.15, "门槛那一带要在标尺上挤成窄窄一条（现在 %.0f%%）" % (BAND * 100)
assert abs(run(ETA_OK)[-1][0]) < W0[0] and abs(run(ETA_BAD)[-1][0]) > W0[0], \
    "⭐ ±5% 必须真的一个收敛一个发散 ——&#160;「差一成就翻脸」全靠这条"

# ── 画面 ──────────────────────────────────────────────────────────
SX, SY = 0.62, 0.62               # w2 → 横，w1 → 纵（平方向铺宽，陡方向在纵向跳）
CY = 0.75                         # 等高线区域的中心高度
BOX_X, BOX_Y = 6.0, 2.5           # 画到这儿就算飞出去了
RULE_Y = -2.85                    # η 标尺的高度
RX0, RX1 = -5.0, 5.0

T_UP = 5.6
T_END = T_UP * 2                  # ⭐ 三角波：上去再回来，首末天然同帧


def pt(w1, w2):
    return np.array([w2 * SX, CY + w1 * SY, 0])


def rule_x(eta):
    return RX0 + (RX1 - RX0) * (eta - ETA_LO) / (ETA_HI - ETA_LO)


class OneLR(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        tt = ValueTracker(0.0)

        def eta_now():
            t = tt.get_value()
            p = t / T_UP if t <= T_UP else (T_END - t) / T_UP   # 三角波
            return ETA_LO + (ETA_HI - ETA_LO) * max(0.0, min(1.0, p))

        # ── 底稿：几条真椭圆等高线 ＋ 一根 η 标尺 ────────────────
        base = VGroup()
        for c in (4.0, 16.0, 36.0, 64.0):
            pts = []
            for a in range(0, 361, 3):
                th = math.radians(a)
                # A1·w1² + A2·w2² = c 的参数式
                pts.append(pt(math.sqrt(c / A1) * math.sin(th),
                              math.sqrt(c / A2) * math.cos(th)))
            for u, v in zip(pts, pts[1:]):
                base.add(Line(u, v, stroke_color=GY_, stroke_width=1.2))
        base.add(Dot(pt(0, 0), radius=0.06, color=GY2_))        # 谷底
        base.add(Line(np.array([RX0 - 0.2, RULE_Y, 0]),
                      np.array([RX1 + 0.2, RULE_Y, 0]),
                      stroke_color=GY2_, stroke_width=2.2))
        # ⭐⭐ 门槛那条红线是全片唯一的「阈值」视觉 ——&#160;它是算出来的（2/A₁），
        #   不是我挑的位置。滑块越过它，上面的轨迹当场翻脸。
        base.add(Line(np.array([rule_x(ETA_MAX), RULE_Y - 0.30, 0]),
                      np.array([rule_x(ETA_MAX), RULE_Y + 0.30, 0]),
                      stroke_color=RD_, stroke_width=3.0))
        # ⭐⭐ 门槛 ±5% 两条细线：它们之间只有标尺的一成宽，
        #   而滑块从左边那条走到右边那条，上面的轨迹就从收敛变成发散。
        #   **「差一成就翻脸」这句话的全部画面就是这两条线有多近。**
        for e in (ETA_OK, ETA_BAD):
            base.add(Line(np.array([rule_x(e), RULE_Y - 0.19, 0]),
                          np.array([rule_x(e), RULE_Y + 0.19, 0]),
                          stroke_color=GY2_, stroke_width=1.6))
        self.add(base)

        # ── 当前 η 那条轨迹 ──────────────────────────────────────
        def traj():
            eta = eta_now()
            col = RD_ if eta > ETA_MAX else (BL_ if eta > _ETA_OSC else GR_)
            g, prev = VGroup(), None
            for w1, w2 in run(eta):
                if abs(w1) > BOX_Y / SY or abs(w2) > BOX_X / SX:
                    break                      # ⭐ 飞出画面就断掉，读成「炸了」
                q = pt(w1, w2)
                if prev is not None:
                    g.add(Line(prev, q, stroke_color=col, stroke_width=3.2))
                g.add(Dot(q, radius=0.055, color=col))
                prev = q
            return g

        self.add(always_redraw(traj))

        # ── 标尺上的滑块 ─────────────────────────────────────────
        def knob():
            eta = eta_now()
            col = RD_ if eta > ETA_MAX else (BL_ if eta > _ETA_OSC else GR_)
            return Dot(np.array([rule_x(eta), RULE_Y, 0]), radius=0.12, color=col)

        self.add(always_redraw(knob))

        # ⭐ 1.8 倍时长 20 秒，比同页其它几段都长一截；扫描本身够清楚，
        #   收到 1.3 倍（≈15 秒）跟它们对齐。
        self.play(tt.animate.set_value(T_END), run_time=T_END * 1.3,
                  rate_func=linear)
