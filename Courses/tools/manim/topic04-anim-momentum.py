# -*- coding: utf-8 -*-
r"""专题四 · 动量：把上一步的速度留下来 ——&#160;`fig-momentum` 配的动画（候选第 5）

⭐⭐⭐ 2026-09-18 现场，看完前六段之后的一句话：「为啥没有那么 fancy 的图？」
  ——&#160;答案不是能力，是**我自己定的房规**。原来那四条里有两条在压表现力：
    ⛔ 「一个字都不放」→ 砍掉了 3b1b 最强的手法：**公式和图形同步变形**
    ⛔ 「15 秒且首尾同帧」→ 一个概念只能铺 15 秒，还得留复位段
  ⭐ 他拍板松绑：「按最漂亮的风格来，一次搞一个，多轮迭代。」
  **这一段是松绑后的第一支。**

⭐⭐ 松绑后新增的三样（都在本机实测过才敢用）：
  ① `Text` 写中文 ——&#160;走 Pango，**不需要 CJK LaTeX 宏包**
  ② `MathTex` ＋ `TransformMatchingTex` ——&#160;公式项对项地变过去
  ③ `MovingCameraScene` ——&#160;镜头能推近看清「冲过浅坑」那一下
⛔ 没松的两条，仍然是硬规矩：
  ③ 静态图排在动画上面（视频加载不出来也要能读懂）
  ④ **数据全部当场算** ——&#160;这条永远不松

⛔⛔ 文字进画面之后多出一条**新义务**：
  视频里的字**选不中、读屏读不到、搜索搜不到**。
  ⭐ 所以 `aria-label` 必须把画面里出现过的每一句话都复述一遍 ——&#160;
    否则等于给用读屏的人留了一段空白。这条写进 skill 了。

⛔ 这一段**不循环**（23 秒、有起承转合）
  ⚠️ 这里原本写的是「26 秒」，实测 23.0 —— 又一个 traps §3.3：
    **注释里的数也是一种断言**。图注那边有 lint 盯着，docstring 这边没有，
    所以它一声不响地过期了整整一天。，页面上给 `controls` 不给 `loop`，
  `loop-baseline.json` 里标 `"loop": false` 让守卫跳过首尾检查。
  ⭐ 判据：**守卫要守的是「承诺」，不是「形状」。**

📌 渲染：
    ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic04-anim-momentum.py Momentum WebPages/media/topic04-momentum.mp4
"""
import math

import numpy as np
from manim import (MovingCameraScene, VGroup, Dot, Line, Arrow, MathTex, Text,
                   ParametricFunction, ValueTracker, always_redraw, Write,
                   FadeIn, FadeOut, Indicate, TransformMatchingTex,
                   UP, DOWN, ORIGIN, linear)
# ⭐⭐⭐ 2026-09-19：配色改成 **manim 自带的原生那套**（房规②b）——
# ⛔⛔ 但 `rate_func=linear` **不能一刀切删掉**，要看这个 play 在动什么：
#     · 动**一个物体**（`FadeIn` / `Indicate` / `Write` / `TransformMatchingTex` /
#       `camera.frame.animate.scale`）→ 走默认 `smooth`，匀速才是「机器感」的来源；
#     · 驱动**一个时钟**（`ValueTracker.animate.set_value`，后面挂 `always_redraw`）
#       → ⭐ **必须显式 `linear`**。`smooth` 会让每段 play 的两端速度归零，
#         于是**时间本身**在每个段界停顿 ——&#160;球走走停停，
#         而「冲过浅坑」那一下本来就是靠慢放铺出来的，时钟一非匀速当场毁掉。
#   ⭐ 判据：**`rate_func` 修饰的是「这段 play 的进度曲线」，
#     而当进度就是时间时，任何缓动都是在篡改物理。**
#   不写 `background_color`（默认 #000000 就是作者的用法），色板一律用命名色。
# ⛔ 导入别名不要用下划线开头的短名（`WHITE as _W` 那类会跟本文件已有变量撞名，
#   撞上之后拿到的是个数组，报「颜色不接受长度 N」）。这里用原名再做语义别名。
from manim import WHITE, BLUE, RED, GREY, GREY_B

INK_ = WHITE             # #FFFFFF ——&#160;公式与正文，黑底上改白
BL_ = BLUE               # #58C4DD ——&#160;带动量那颗球 ＋ 它的字幕
RD_ = RED                # #FC6255 ——&#160;「卡住 / 冲过了头」的警示
# ⛔ 地形曲线原来是 #c3c7cb（白底上的浅灰），黑底上它比球还亮，喧宾夺主。
#   ⭐ 现在 **曲线 GREY #888888 ＜ 灰球 GREY_B #BBBBBB**：
#     地形是背景、球才是主角，亮度顺序要跟这个层级一致。
GY_ = GREY               # #888888 ——&#160;地形曲线
# ⛔⛔ 灰球（无动量那颗）原来是 #80868b ——&#160;黑底上这块灰几乎沉进背景，
#   而「灰球卡在浅坑里」正是整段的对照主角，看不见就没有对照。
#   ⭐ 抬到 GREY_B #BBBBBB，抽帧确认过它在浅坑里清清楚楚。
GY2_ = GREY_B            # #BBBBBB ——&#160;无动量那颗球 ＋ 它的字幕


# ══════════════════════════════════════════════════════════════════
# 地形：一个**浅坑** ＋ 一个**深谷**。形状让脚本自己找，不手摆。
# ══════════════════════════════════════════════════════════════════
# ⛔ 第一版手写的是 `+0.16x` ——&#160;**地形写反了**：那一项让整条曲线往右上抬，
#   于是起点左边那个坑反而最深，球一进去就到底，「冲过去」无从谈起。
#   ⭐ 断言当场挡下（「先遇到的那个坑要明显浅」）。
# ⭐⭐ 判据：**参数也是数据，别手调** ——&#160;与其凑，不如把「要满足的条件」
#   写成筛子去搜。这组 (a, b, c, w, η, β) 是在 3×3×3×3×3×3 的网格里
#   按「深浅差 × 速度峰值比」最大挑出来的，454 个可行解里的头名。
def f(x):
    return 0.030 * x * x + 0.70 * math.cos(1.05 * x) - 0.26 * x


def g(x):
    return 0.060 * x - 0.735 * math.sin(1.05 * x) - 0.26


_MIN = [i / 500.0 for i in range(-3000, 3000)
        if g(i / 500.0) < 0 < g((i + 1) / 500.0)]
assert len(_MIN) >= 2, "地形上至少要有两个坑，现在 %d" % len(_MIN)

START = -5.6
# ⭐ 从起点往右数：第一个遇到的坑必须是**浅**的，后面那个才是深谷 ——
#   「冲过去」才有意义。这两条是这段动画的地形前提。
_right = [m for m in _MIN if m > START + 0.4]
SHALLOW, DEEP = _right[0], min(_right[1:], key=f)
assert f(SHALLOW) > f(DEEP) + 0.35, \
    "先遇到的那个坑要明显浅（浅 %.2f vs 深 %.2f）" % (f(SHALLOW), f(DEEP))
assert DEEP > SHALLOW, "深谷要在浅坑右边，球才是「冲过去」而不是「退回去」"

ETA, BETA, NSTEP = 0.30, 0.90, 120


def roll(beta):
    """beta=0 就是普通梯度下降；>0 是带动量的。返回 [(x, v), ...]。"""
    out, x, v = [], START, 0.0
    for _ in range(NSTEP):
        out.append((x, v))
        v = beta * v + g(x)
        x -= ETA * v
    out.append((x, v))
    return out


PLAIN, MOM = roll(0.0), roll(BETA)

# ⭐⭐⭐ 这段动画的全部论点，三条断言钉死 ——&#160;不是我写上去的
assert abs(PLAIN[-1][0] - SHALLOW) < 0.30, \
    "没有动量那颗必须**停在浅坑**（现在停在 %.2f，浅坑在 %.2f）" % (PLAIN[-1][0], SHALLOW)
assert abs(MOM[-1][0] - DEEP) < 0.40, \
    "带动量那颗必须**落进深谷**（现在停在 %.2f，深谷在 %.2f）" % (MOM[-1][0], DEEP)
_peak = max(abs(v) for _, v in MOM)
assert _peak > max(abs(v) for _, v in PLAIN) * 1.3, \
    "带动量那颗的速度峰值要明显更大，「攒起来的速度」才看得见"

# ⭐ 三个关键步号全部算出来 ——&#160;镜头和节奏都挂在它们身上，不是我挑的时间点。
# ⛔ 第一版按「跑满 120 步」线性播，结果**戏全在头 20 步**，
#   镜头推近的时候球早过去了。判据：**时长要按实际动作区间定，不按循环次数定。**
K_CROSS = next(i for i, (x, _) in enumerate(MOM) if x > SHALLOW + 0.25)
K_STUCK = next(i for i, (x, _) in enumerate(PLAIN) if abs(x - PLAIN[-1][0]) < 0.05)
K_LAND = next(i for i, (x, _) in enumerate(MOM) if abs(x - MOM[-1][0]) < 0.05)
# ⭐⭐ 带动量那颗**冲过了头**才荡回深谷 ——&#160;这是动量的真实副作用，
#   不是 bug，而且正好是下一节（阻尼 / Nesterov）的引子。钉住它。
K_OVER = max(range(K_LAND), key=lambda i: MOM[i][0])
assert MOM[K_OVER][0] > DEEP + 1.2, \
    "冲过头那一段要明显（最远 %.2f，深谷 %.2f）" % (MOM[K_OVER][0], DEEP)
# ⛔ 这里本来写的是 `K_CROSS < K_STUCK < K_LAND` ——&#160;**那是句废话**：
#   它拿**两条独立轨迹**的步号在比大小，而两幕是分开播的，
#   谁的第 19 步在前根本没有物理意义。
#   ⭐ 判据：**断言要断在同一个坐标系里的量上。** 换成这条轨迹自己的顺序。
assert K_CROSS < K_OVER < K_LAND, \
    "同一条轨迹上顺序必须是：冲过浅坑 → 冲到最远 → 荡回深谷"

# ── 画面 ──────────────────────────────────────────────────────────
SX, SY, Y0 = 0.98, 0.62, -1.15
XL, XR = -6.3, 6.3


def P(x):
    return np.array([x * SX, Y0 + f(x) * SY, 0])


class Momentum(MovingCameraScene):
    def construct(self):
        # ⛔ 不写 background_color —— 默认就是 #000000，这正是作者的用法
        curve = ParametricFunction(P, t_range=[XL, XR, 0.04],
                                   stroke_color=GY_, stroke_width=4)
        self.add(curve)

        def ball(traj, col, tr, arrow=False):
            def mk():
                k = min(len(traj) - 1, int(tr.get_value()))
                x, v = traj[k]
                g_ = VGroup(Dot(P(x), radius=0.13, color=col))
                if arrow and abs(v) > 0.05:
                    # ⭐ 速度箭头：长度 ∝ |v| ——&#160;「攒起来的速度」这件事
                    #   必须有个东西承载，否则「冲过去」看着像运气好。
                    # ⛔ 但长度**必须封顶**：镜头推近到 0.45 倍时画面等比放大 2.2 倍，
                    #   没封顶的箭头当场比球大五倍、还冲出画框。
                    #   ⭐ 判据：**任何「长度 ∝ 数据」的元素，在有镜头运动的片子里
                    #     都要设上限** ——&#160;它的视觉尺寸是「数据 × 镜头倍率」两项相乘。
                    ln = min(abs(v) * 0.55, 1.05) * (1 if v < 0 else -1)
                    g_.add(Arrow(P(x) + UP * 0.30,
                                 P(x) + UP * 0.30 + np.array([ln, 0, 0]),
                                 buff=0, stroke_width=5,
                                 max_tip_length_to_length_ratio=0.32, color=col))
                return g_
            return always_redraw(mk)

        # ═══ 幕一：没有动量 ——&#160;卡在第一个坑里 ════════════════
        t1 = ValueTracker(0.0)
        b1 = ball(PLAIN, GY2_, t1)
        self.add(b1)
        cap = Text("没有动量：停在第一个坑", font_size=30, color=GY2_).to_edge(UP)
        self.play(FadeIn(cap, shift=DOWN * 0.2), run_time=0.6)
        self.play(t1.animate.set_value(K_STUCK + 4), run_time=3.2, rate_func=linear)
        stuck = Dot(P(PLAIN[-1][0]), radius=0.13, color=GY2_)
        self.add(stuck)
        self.remove(b1)
        self.play(Indicate(stuck, color=RD_, scale_factor=1.6), run_time=0.9)

        # ═══ 幕二：带动量 ——&#160;冲过去 ═════════════════════════
        cap2 = Text("带动量：把上一步的速度留下来", font_size=30, color=BL_).to_edge(UP)
        # ⛔ 这里别用 TransformMatchingTex —— 它是给 MathTex 的（按 tex 子串配对），
        #   两段中文 Text 之间没有可配对的子串，白搭一层还容易出怪相。
        self.play(FadeOut(cap, shift=UP * 0.2), run_time=0.4)
        self.play(FadeIn(cap2, shift=DOWN * 0.2), run_time=0.5)

        t2 = ValueTracker(0.0)
        self.add(ball(MOM, BL_, t2, arrow=True))
        self.play(t2.animate.set_value(max(1, K_CROSS - 2)), run_time=1.0,
                  rate_func=linear)
        # ⭐ 镜头推近**那一下** ——&#160;冲过浅坑是整段的戏眼，而它只占三四步，
        #   不推近就一晃而过。推近＋慢放，把这三四步铺成两秒。
        self.play(self.camera.frame.animate.scale(0.45).move_to(P(SHALLOW) + UP * 0.30),
                  run_time=0.8)
        self.play(t2.animate.set_value(K_CROSS + 4), run_time=2.0, rate_func=linear)
        self.play(self.camera.frame.animate.scale(1 / 0.45).move_to(ORIGIN),
                  run_time=0.8)
        # ⭐⭐ 冲过头那一段：它会荡到深谷右边很远再回来。**别剪掉** ——
        #   那是动量真实的副作用，也是下一节要治的东西。
        self.play(t2.animate.set_value(K_OVER), run_time=1.6, rate_func=linear)
        over = Text("冲过了头", font_size=26, color=RD_).next_to(P(MOM[K_OVER][0]), UP, buff=0.5)
        self.play(FadeIn(over, shift=DOWN * 0.15), run_time=0.4)
        # ⛔ 这个标注**要早早撤掉**：它钉在最远那一点上，而球马上就荡回去了，
        #   跟着整段一起淡出的话，后一秒里它指着一个空地方。
        #   ⭐ 判据：**钉在位置上的标注，寿命不能超过它指的那个东西停留的时间。**
        self.play(FadeOut(over), run_time=0.35)
        self.play(t2.animate.set_value(K_LAND + 6), run_time=2.0, rate_func=linear)
        self.play(Indicate(Dot(P(MOM[-1][0]), radius=0.13, color=BL_),
                           color=BL_, scale_factor=1.7), run_time=0.9)

        # ═══ 幕三：公式 ——&#160;刚才看见的那件事，写成两行 ═══════
        self.play(FadeOut(cap2), run_time=0.4)
        e1 = MathTex(r"v_t", "=", r"\beta", r"v_{t-1}", "+", r"g_t",
                     color=INK_).scale(1.35).to_edge(UP, buff=0.75)
        self.play(Write(e1), run_time=1.3)
        self.play(Indicate(e1[2], color=BL_, scale_factor=1.6),
                  Indicate(e1[3], color=BL_, scale_factor=1.35), run_time=1.0)
        note = Text("上一步的速度，留下 β 那么多", font_size=26,
                    color=BL_).next_to(e1, DOWN, buff=0.35)
        self.play(FadeIn(note, shift=UP * 0.15), run_time=0.6)
        self.wait(0.8)
        e2 = MathTex(r"w_t", "=", r"w_{t-1}", "-", r"\eta", r"v_t",
                     color=INK_).scale(1.35).to_edge(UP, buff=0.75)
        self.play(FadeOut(note), TransformMatchingTex(e1, e2), run_time=1.3)
        note2 = Text("走的不是这一步的梯度，是攒下来的速度", font_size=26,
                     color=BL_).next_to(e2, DOWN, buff=0.35)
        self.play(FadeIn(note2, shift=UP * 0.15), run_time=0.6)
        self.wait(1.6)

        # ⭐⭐⭐ 2026-09-20 收尾复位 ——&#160;现场：「能放一个动图搞定的，就别放视频。」
        #   原来这一段是全讲唯一带播放器控件的：因为它 23 秒、首尾不一样，
        #   一循环就会突兀地跳回去，所以只好给它挂 controls。
        #   ⭐ 而首尾不一样是**可以修的**：第 0 帧只有那条灰色曲线（上面 self.add(curve)），
        #     所以结尾把除它以外的全部淡出，末帧就跟首帧一模一样。
        #   ⛔ 镜头此前已经复位回 ORIGIN／scale 1（见上面那次推拉），否则这里还得一并还原。
        #   判据：**要让一段动画能当动图用，先让它的最后一帧长得跟第一帧一样。**
        # ⛔⛔ 第一版只留了 curve ——&#160;拼接图当场打脸：**首帧还有一个灰球**
        #   （第 0 帧是 `self.add(curve)` ＋ `self.add(b1)`，而 b1 在 t1=0 处可见）。
        #   数字看着正常（22.9%，跟其他黑底循环同档），**是图暴露的**。
        #   ⭐ 判据：**首尾一致别只看那个百分比，一定要看拼接图** ——&#160;
        #     一个小球的有无，在整幅画的平均差异里几乎看不出来。
        # ⛔ 第二版又栽了一次：想靠 `t1.set_value(0)` ＋ `self.add(b1)` 把球复位，
        #   可 b1 在幕一结尾就被 `self.remove(b1)` 换成了静态的 stuck 点 ——
        #   末帧留下的是 stuck（停在浅谷），不是起点那颗。
        #   ⭐ 判据：**复位不要去唤醒一个已经被移走的 always_redraw，
        #     直接清干净、再摆一个跟第 0 帧同样的静态件** ——&#160;确定性比聪明重要。
        _keep = [m for m in self.mobjects if m is not curve]
        self.play(*[FadeOut(m) for m in _keep], run_time=0.9)
        self.remove(*_keep)
        self.add(Dot(P(PLAIN[0][0]), radius=0.13, color=GY2_))   # ＝ 第 0 帧那颗
        self.wait(0.6)
