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

═══ 2026-09-18 下午 · S5：房规松绑后回来加解释性文字 ═══════════════

⭐⭐⭐ **先判一句：这一段缺的是「解释」，不是「铺陈」。**
  理由：**画面本身一秒都不缺** ——&#160;山怎么堆、怎么塌、红带什么时候长出来，
  全都看得见；缺的是**这些颜色各自是什么、三段各自叫什么名字**，
  以及那个落点「峰值就在前向结束那一刻」**画面上一个字都没点破**。
  ⭐ 观众看到的是「几块颜色在动」——&#160;那是解释缺位，不是时间不够。
  ⛔ 所以 **`T_*` 一个都不改、时长不动、`loop` 不动**，只加字。
    判据（房规）：**「不够 fancy」的解药是信息密度，不是时长。**

⭐⭐ 字幕怎么挂：**幕间显式 `add` / `remove`**，不给 Text 挂每帧改 opacity 的
  updater ——&#160;那条在 `descend` 上实测把 -qh 从 28 it/s 拖到 0.47 it/s。
  判据：**幕与幕之间是离散事件，别做成每帧都要问一次的连续量。**
  ⭐ 代价是原来那一条 `self.play` 被切成五段；每段 `run_time` 仍然
    ＝ Δt × `SPEED`，rate_func 全是 linear，**所以画面速度逐帧不变**。
    ⛔⛔ 2026-09-19 这条**被误删过一次又加回来了** ——&#160;房规 ②b 那句
      「别显式传 linear」说的是 **play 在动一个物体**的时候；这里的 play
      动的是**时钟**（`ValueTracker` ＋ `always_redraw`），`smooth` 会让
      **时间本身**在五个段界各停一下。判据写在 construct 末尾那段注释里。
  ⭐ 末尾多一个 1/30 秒的恒等 `play`：把字幕拨回第一幕之后**强制走一遍
    updater**（`wait()` 不保证驱动 `always_redraw`），这样末帧才真的 ≡ 首帧。

⭐ 峰值标注的寿命：**前向结束时出现，回退段开始时撤掉**。
  这正好等于「那座山的顶停留在画面上的时间」——&#160;band 画的是**累积**历史，
  所以山顶从前向末尾一直挂到回退开始，标注没有一帧在指空地（traps §2.5）。

⛔ 沿用的规矩（①已于 2026-09-18 松绑，以 skill 的房规为准）：
  ① ~~一个字都不放~~ →&#160;**文字为讲解服务，不为装饰**；
     ⛔ 硬义务：页面 `<video>` 的 `aria-label` 必须把画面里出现过的
       **每一句话都复述一遍**（这一段的字幕见下面 `CAPS` / `SUBS` / 峰值标注）。
  ② 静态图排在视频上面兜底。
  ③ 短循环片 →&#160;首尾必须同帧。
  ④ 数据全部当场算。
  ⑤ ~~配色跟 `topic03_draw` 对齐~~ →&#160;**2026-09-19 改成 manim 原生深色那套**
     （房规 ②b）：黑底 ＋ BLUE/RED/GREEN/WHITE/GREY，默认 smooth，线宽 4。
     ⚠️ 这一段是七段里**面积最大**的，所以「白 → 黑」不是把颜色查表换一遍就完 ——
       白底上的「淡灰大面积」翻到黑底是**脏**，白底上的「细灰线」翻到黑底是**没有**。
       两处判断分别记在 `base` 和 `cells` 的注释里。

⭐⭐⭐ 三条带的高度**按真实字节数算**，跟 `fig-step` 同一套口径，脚本里 assert 了：
    · 权重 2 B ＋ 优化器状态 12 B ＝ **14 B/参数** → 不动的基座
    · 梯度 **2 B/参数** → 反向才长出来，反向结束时最全
    · 激活（不开重算、一条 128K 序列）→ 前向堆高、反向释放
  ⚠️ 跟 `fig-step` 的分组不同、口径相同：那张图把梯度并进「常驻」算 16 B/参数
    （因为它要讲「不随 batch 变的那一块」），这里把梯度单独拆出来
    （因为它**随时间变**，是这段动画的主角之一）。8.54 ＋ 1.22 ＝ **9.76**，
    跟它的 9.76 一个字节不差 ——&#160;这就是「口径一致」的外部锚点。

⚠️⚠️ 落点那句话的**适用范围**（`fig-step` §5 特意强调过）：
  「峰值在哪」必须先问「**哪一项**的峰」——&#160;激活的峰在前向末尾，
  梯度的峰在反向末尾，两个峰不在同一时刻。
  ⭐ 这里敢把**总量**的峰也说成「前向结束那一刻」，是因为
    **激活 4.15 TiB 比梯度 1.22 TiB 大得多**，反向段每放掉一份激活、
    只换回小得多的一份梯度，总量单调下降。
    ⛔ 这不是常识，是这组数字的性质 ——&#160;所以下面**当场扫一遍求 argmax** 验证。

📌 渲染：
    bash ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic04-anim-memtime.py MemTime \
        WebPages/media/topic04-memtime.mp4
"""
import numpy as np
from manim import (Scene, ValueTracker, Polygon, Rectangle, Line, DashedLine,
                   Dot, VGroup, Text, MathTex, always_redraw, RIGHT,
                   DOWN, LEFT, rate_functions, config)
# ⭐⭐⭐ 2026-09-19：配色改成 **manim 自带的原生那套**（skill 房规 ②b）。
#   原来是「白底 ＋ Google 配色」，三处主动覆盖了作者的默认，每处都覆盖成更差的：
#       背景  #000000  → 白        线宽  4 → 2 / 0.8        色板 → Google 那套
#   ⚠️ 房规里还有第三条「别显式传 rate_func=linear」——&#160;**这一条对本文件不适用**，
#     因为这里的 play 动的是时钟不是物体，见 construct 末尾。
#   ⛔ 别名不要用下划线开头的短名（`_W` 那类）——&#160;撞上已有变量之后
#     `stroke_color=` 会拿到一个数组，报「颜色不接受长度 14」。
#   ⭐ 这里沿用本文件原有的**后缀**下划线名（`BL_` / `RD_` / …），只换取值：
#     好处是**下面三百行一个字都不用动**，diff 就是「配色」这一件事。
from manim import (BLUE, RED, GREEN, WHITE, GREY, GREY_B, GREY_D)

BL_, RD_ = BLUE, RED                       # #58C4DD / #FC6255
GR_ = GREEN                                # #83C167
# ⛔ GY_ 在白底版身兼两职：**辅助线**和**大面积填充**。黑底上这两件事要分开 ——
#   同一个灰，够亮的线放大成一整块就是「脏」，够暗的块缩成一条线就是「没有」。
GY_ = GREY                                 # #888888 —— 线、描边（黑底要亮）
FILL_ = GREY_D                             # #444444 —— 大面积填充（黑底要暗）
INK_ = WHITE                               # 正文字
GY2_ = GREY_B                              # #BBBBBB —— 次级文字（说明行、轴注）

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

T_FWD, T_BWD, T_UPD, T_HOLD, T_REW = 1.0, 1.0, 0.34, 0.35, 0.55
# ⭐ T_HOLD：画完的那座山留在屏幕上看一眼。
# ⭐⭐ T_REW 是 2026-09-18 补的 —— 在此之前这支片子**根本没有复位段**：
#   首帧是一条空灰带、末帧是画满的三角，`loop` 每一轮硬跳一次（实测 18.2%）。
#   ⛔ 而它的内容是**累积**的，所以「末尾清零」只是把那一跳从接缝挪进片内。
#   ⭐ 改成**倒着走回去**：让时钟在最后这段里从 T_RESET 线性退回 0，
#     观众看到那座山退潮、扫描线滑回左边 —— 接缝读成
#     「画一遍 → 看一眼 → 退回去 → 重画」，而首末帧是**同一帧**。
T_RESET = T_FWD + T_BWD + T_UPD + T_HOLD      # 到这儿内容画完并停够了
T_END = T_RESET + T_REW
assert T_REW > 0, "没有回退段，首末帧对不上，loop 会跳"

SPEED = 4.5                                   # 时钟 1 个单位 ＝ 屏幕 4.5 秒


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


def total_at(t):
    """此刻显存里的总量 ——&#160;画面上那条上沿就是它。"""
    return RESIDENT_TIB + act_at(t) + grad_at(t)


# ⭐⭐⭐ 落点当场验一遍：**总量的峰确实落在前向结束那一刻**，不是我顺口说的。
#   ⛔ 别用「激活的峰在这儿」来代替 ——&#160;那是另一个命题（见上面 ⚠️⚠️）。
#   扫一遍求 argmax，落点必须就是 T_FWD。
# ⛔ 第一版只扫 `linspace(0, T_RESET, 2001)`，下一条断言当场挂了：
#   峰量成 12.6918 而不是 12.6937。⭐ 挡下来的不是「阈值太严」——&#160;
#   是**采样网格里根本没有 T_FWD 这个点**（2.69/2000 除不尽 1.0，最近的差 0.0007）。
#   总量是**分段线性**的，它的极值只可能落在折点上，所以折点必须在采样集合里。
_TS = np.union1d(np.linspace(0.0, T_RESET, 2001),
                 [T_FWD, T_FWD + T_BWD, T_FWD + T_BWD + T_UPD, T_RESET])
_TOT = np.array([total_at(s) for s in _TS])
PEAK_TIB = float(_TOT.max())
PEAK_T = float(_TS[int(_TOT.argmax())])
assert abs(PEAK_T - T_FWD) < 1e-2, "总量的峰没落在前向末尾，落点那句话就不成立：%r" % PEAK_T
assert abs(PEAK_TIB - (RESIDENT_TIB + ACT_TIB)) < 1e-6, PEAK_TIB
# ⭐ 峰之所以在这儿，机制是这一条：反向段每放掉一份激活只换回更小的一份梯度。
#   ⛔ 这条不成立（比如开了重算、激活缩到比梯度还小）的话，峰就跑到反向末尾去了。
assert ACT_TIB > GRAD_TIB, "激活不比梯度大的话，总量的峰会跑到反向末尾"
# ⭐ 反向末尾那个「梯度最全」的时刻，仍然明显低于峰 ——&#160;画面上那条峰值虚线要看得出差距
BWD_END_TIB = total_at(T_FWD + T_BWD)
assert PEAK_TIB > BWD_END_TIB + 2.0, (PEAK_TIB, BWD_END_TIB)

# ⭐ 复位只能定义在一个地方（traps §1.3）：所有绘制函数一律读 clock()。
#   ⛔ 这条断言兜住「某个绘制函数偷偷直接读 tracker」——&#160;拼接时不要写成
#     连续的字面量，否则它会数到自己头上。
_SRC = open(__file__, encoding="utf-8").read()
assert _SRC.count("tt.get_" + "value()") == 1, "有绘制函数绕过了 clock()"


class MemTime(Scene):
    def construct(self):
        # ⛔ 不写 self.camera.background_color —— 默认 #000000 就是作者的用法
        X0, X1, Y0 = -6.0, 6.0, -2.9
        SY = 0.30                      # 每 TiB 多少个单位高
        tt = ValueTracker(0.0)

        # ⛔ 这里必须除 T_RESET 不是 T_END —— 横轴要在内容画完时**正好铺满**。
        #   除 T_END 的话，回退段那 0.55 也会分走一截宽度，图就缩在左边了。
        def px(t):
            return X0 + (X1 - X0) * t / T_RESET

        def py(tib):
            return Y0 + tib * SY

        def clock():
            t = tt.get_value()
            if t <= T_RESET:
                return t
            return T_RESET * (1.0 - (t - T_RESET) / T_REW)

        # 基座：不动的那 14 字节
        # ⛔⛔ 本轮最大的一处设计判断。白底版是 `GY_(#9aa0a6) @ 0.45` ——&#160;
        #   在白底上是「一层淡淡的底」，**直接搬到黑底就是整屏下半部一大块脏灰**：
        #   它有 12 × 2.56 个单位，占画面约 24%，是全片面积最大的东西。
        # ⭐⭐ 判据：**它要表达的是「厚」，不是「亮」。**
        #   厚度已经由它的高度说完了（8.54 TiB × SY），填充只需要
        #   「这块地是有东西的」这一点点信号；真正该被看见的是它的**上沿** ——
        #   因为蓝色的山正是从那条线上长起来的。
        # ⭐ 所以改成**暗填充 ＋ 亮描边**：FILL_(#444444) 压到 0.55（合成 ≈ #252525，
        #   读作「有东西但不抢戏」），外圈给 GY2_(#BBBBBB) 线宽 4 把边界钉住。
        #   ⛔ 试过「只留描边不填充」——&#160;那样这块变成纯黑，跟画面外的黑连成一片，
        #     「常驻占掉一大块显存」这个体量感当场没了，而那是这一幕的前提。
        base = Rectangle(width=X1 - X0, height=RESIDENT_TIB * SY,
                         fill_color=FILL_, fill_opacity=0.55,
                         stroke_color=GY2_, stroke_width=4)
        base.move_to(np.array([(X0 + X1) / 2, Y0 + RESIDENT_TIB * SY / 2, 0]))

        def band(fn, col, below):
            """把一条随时间变的带，按**已经走过的那段**画成填充多边形。"""
            def make():
                t_now = clock()
                n = max(2, int(120 * t_now / T_END) + 2)
                ts = np.linspace(0, t_now, n)
                top = [np.array([px(s), Y0 + (below(s) + fn(s)) * SY, 0]) for s in ts]
                bot = [np.array([px(s), Y0 + below(s) * SY, 0]) for s in reversed(ts)]
                pts = top + bot
                if len(pts) < 3:
                    pts = pts + [pts[-1] + np.array([1e-3, 0, 0])]
                # ⭐ 线宽 2 → 4（房规：默认就是 4，别往下调）。
                #   黑底上这条描边还多一份活：它是山的**轮廓线**，
                #   填充压到 0.72 之后全靠它把形状咬住。
                return Polygon(*pts, fill_color=col, fill_opacity=0.72,
                               stroke_color=col, stroke_width=4)
            return always_redraw(make)

        act_band = band(act_at, BL_, lambda s: RESIDENT_TIB)
        grad_band = band(grad_at, RD_, lambda s: RESIDENT_TIB + act_at(s))

        # ⭐ 现在走到哪儿：一根扫过去的竖线 ＋ 层轴上的滑块
        # ⛔ 竖线原来的顶端写死在 Y0+3.25 ＝ 0.35，**比山顶（0.91）还矮** ——
        #   前向后半段它整个埋在蓝色里，看着像画漏了。
        #   ⭐ 改成从峰值高度算出来：永远比山顶高一点点，不多不少。
        SWEEP_TOP = py(PEAK_TIB) + 0.14
        sweep = always_redraw(lambda: Line(
            np.array([px(clock()), Y0 - 0.25, 0]),
            np.array([px(clock()), SWEEP_TOP, 0]),
            color=INK_, stroke_width=4))

        # 层轴：61 个小格，前向从左往右点亮、反向从右往左熄掉
        # ⛔ 白底版未点亮的格子是 `GY_ @ 0.12 ＋ 0.8 的描边`——&#160;搬到黑底，
        #   填充合成 #141414、描边 0.8 个单位约 1 px，**整排 61 格当场消失**，
        #   于是「点亮到第几层」失去了参照系：看得见亮的，看不见还剩多少没亮。
        # ⭐ 判据：**它是一把尺子，刻度必须先存在，才谈得上指到哪一格。**
        # ⭐⭐ 改成 FILL_(#444444) **实心不透明、干脆不要描边**。
        #   ① #444444 作为 21×23 px 的**色块**在黑底上清清楚楚 ——&#160;
        #      skill 里「GREY_D 在黑底上消失」说的是**细线**，面积不一样结论不一样；
        #   ② 去掉描边顺带绕开「线宽不低于 4」：一格才 21 px 宽，
        #      4 px 的边框会把它吃掉一大半，**这里正确的做法是没有线，不是细线**。
        #   格子之间本来就留了 22% 的缝（0.78 因子），不靠描边也分得开。
        LN, LY = 61, Y0 - 0.72
        cells = VGroup(*[
            Rectangle(width=(X1 - X0) / LN * 0.78, height=0.17,
                      stroke_width=0,
                      fill_color=FILL_, fill_opacity=1.0)
            .move_to(np.array([X0 + (X1 - X0) * (i + 0.5) / LN, LY, 0]))
            for i in range(LN)])

        def lit():
            t_now = clock()
            if t_now <= T_FWD:
                k, col = int(LN * t_now / T_FWD), BL_
            elif t_now <= T_FWD + T_BWD:
                k = int(LN * (1 - (t_now - T_FWD) / T_BWD)); col = RD_
            else:
                k, col = 0, GR_
            g = VGroup()
            for i in range(min(k, LN)):
                g.add(cells[i].copy().set_fill(col, opacity=1.0)
                      .set_stroke(col, 0))
            return g

# ⛔ 前向段 grad_at 恒为 0，`grad_band` 退化成一条线 ——&#160;但**零高度的
#   Polygon 照样会描边**，于是一条红边贴着蓝色山的斜边跑，看着像
#   「前向时梯度已经存在」。这是旧版就有的，而 2026-09-19 把带子描边
#   从 2 加到 4 之后它**响了一倍** —— 自己的改动放大了别人的旧缺陷，
#   那就归自己修。
# ⭐ 最小修法：把 grad 放到 act **底下**。前向段那条红边被蓝色描边盖住；
#   反向段两条带本来就上下分开，z 序无所谓；首末帧状态相同，不影响 loop。
        self.add(base, grad_band, act_band, cells, always_redraw(lit), sweep)
        self.add(Dot(np.array([X0, Y0 + RESIDENT_TIB * SY, 0]), radius=0.001))

        # ══════════════════════════════════════════════════════════════
        # 解释层 ——&#160;以下全是「一直在场」的字，不随幕变
        # ══════════════════════════════════════════════════════════════

        # ① 图例：三块颜色各自叫什么。
        #   ⭐ 放图例而不是就地贴标签 ——&#160;蓝和红是**长出来的**，
        #     贴在它们身上的标签在它们还没出现时会指着一块空地（traps §2.5）。
        #   ⛔ 图例的色块必须跟它指代的那块**长得一样** ——&#160;基座改成
        #     「暗填充 ＋ 亮描边」之后，这里的 chip 也得照改，否则图例上是一块灰、
        #     画面上是一个框，观众对不上号。描边按面积等比缩到 2（那块是 4）。
        legend = VGroup()
        for col, op, edge, name in ((FILL_, 0.55, GY2_, "常驻 · 权重＋优化器"),
                                    (BL_, 0.72, BL_, "激活 · 前向堆、反向放"),
                                    (RD_, 0.72, RD_, "梯度 · 反向才长出来")):
            chip = Rectangle(width=0.30, height=0.22,
                             stroke_color=edge, stroke_width=2,
                             fill_color=col, fill_opacity=op)
            legend.add(VGroup(chip, Text(name, font_size=21, color=INK_))
                       .arrange(RIGHT, buff=0.15))
        legend.arrange(RIGHT, buff=0.62).move_to(np.array([0, 2.26, 0]))

        # ② 灰带里的那笔账 ——&#160;它为什么这么厚、厚多少。
        #   ⭐ 写在带子**里面**：这块从第一帧到最后一帧都在，不会指空。
        resident_lab = VGroup(
            Text("常驻：一个 step 从头到尾都不动", font_size=25, color=INK_),
            Text("权重 2 ＋ 优化器状态 12 ＝ 14 字节/参数　→　%.2f TiB" % RESIDENT_TIB,
                 font_size=21, color=GY2_),
        ).arrange(DOWN, buff=0.16).move_to(np.array([0, -1.55, 0]))

        # ③ 两条轴各自是什么 ——&#160;横轴是**时间**不是层号（fig-act-bill 也强调过这条）
        axis_note = Text(
            "横轴 ＝ 一个 step 的时间　·　下面 61 格 ＝ 61 层，亮着的是此刻还挂着激活的层",
            font_size=20, color=GY2_).move_to(np.array([0, -3.26, 0]))

        self.add(legend, resident_lab, axis_note)

        # ══════════════════════════════════════════════════════════════
        # 幕：三段各自的名字 ＋ 一句为什么
        # ⛔ 用幕间 add/remove，**不给 Text 挂每帧改 opacity 的 updater**。
        # ══════════════════════════════════════════════════════════════
        CAPS = ("前向：激活一层层攒起来",
                "反向：激活逐层放掉，梯度一份份长出来",
                "更新完：梯度也清空，只剩常驻那一块",
                "下一个 step，从头再来一遍")
        SUBS = ("每过一层就多挂一份中间结果　——　一份都不能提前扔，反向还要用它",
                "红的在长，可它比蓝的小得多　——　所以总量从峰值那一刻起只降不升",
                "回到 %.2f TiB　——　下一个 step 从这条灰带上重新堆" % RESIDENT_TIB,
                "训练就是把这座山堆起来、再拆掉　——　重复几十万次")
        caps = [Text(s, font_size=30, color=INK_).move_to(np.array([0, 3.46, 0]))
                for s in CAPS]
        subs = [Text(s, font_size=23, color=GY2_).move_to(np.array([0, 2.90, 0]))
                for s in SUBS]

        # ④ 落点：**峰值就在前向结束那一刻**。
        #   ⭐ 两条虚线合起来才是论点：竖的说「是这一刻」，
        #     横的说「此后谁也没再碰到它」——&#160;只画竖的，观众看不出后面更低。
        peak_x, peak_y = px(T_FWD), py(PEAK_TIB)
        #   ⛔ 白底版这两条是 2.2 / 1.8 的细虚线 ——&#160;那是在白底上「不要太吵」的取舍。
        #     黑底上反过来：RD_(#FC6255) 本身够亮，但**虚线的每一段都短**，
        #     线宽不够时整条读成一串浮尘。⭐ 两条都提到 4（房规下限），
        #     并把 dash_length 从 0.10 放到 0.16 ——&#160;不然宽 4 长 0.10 的段
        #     接近正方形，看着是点阵不是虚线。
        peak_v = DashedLine(np.array([peak_x, py(RESIDENT_TIB), 0]),
                            np.array([peak_x, peak_y + 1.05, 0]),
                            color=RD_, stroke_width=4, dash_length=0.16)
        peak_h = DashedLine(np.array([X0, peak_y, 0]),
                            np.array([X1, peak_y, 0]),
                            color=RD_, stroke_width=4, dash_length=0.16)
        peak_txt = VGroup(
            Text("峰值就在这一刻　——　前向刚结束", font_size=26, color=RD_),
            MathTex(r"%.2f + %.2f = %.2f\;\mathrm{TiB}"
                    % (RESIDENT_TIB, ACT_TIB, PEAK_TIB), color=RD_).scale(0.72),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        # ⛔ 第一版放在 peak_y+1.28 ＝ 2.19，**正好压在图例那一行上**（图例在 2.26）。
        #   ⭐ 判据跟 descend 那次一样：**标注要贴着它指的那个东西，不是往空白处放**。
        #     峰顶 0.91、图例底边 2.10，这块字只能落在中间那一段。
        peak_txt.move_to(np.array([peak_x + 0.28 + peak_txt.width / 2,
                                   peak_y + 0.61, 0]))
        peak = VGroup(peak_h, peak_v, peak_txt)

        # ══════════════════════════════════════════════════════════════
        # 播：切成五段只为在幕间换字，run_time 仍是 Δt × SPEED，速度逐帧不变
        # ⛔⛔ 2026-09-19 这里**改错过一版，又改回来了**，值得留档：
        #   当时按房规 ②b「不要显式 rate_func=linear」把这五段的 linear 全删了。
        #   ⭐⭐ 判据错在哪：房规那条说的是**动一个物体**时别用匀速
        #     （匀速＝机器感）。可这里的 `play` 动的**不是物体，是时钟** ——
        #     `tt` 后面挂着一串 `always_redraw`，`rate_func` 作用在**时间轴**上。
        #   ⛔ `smooth` 两端速度归零，于是**时间本身**在每个幕界停顿一下：
        #     扫描线在四个段界各顿一次、段中反而狂飙（并行那位在 `anim-reverse`
        #     上实测段界位移近乎 0、段中峰值快 40 倍）。
        #   ⛔ 对这一段还额外坏一层：复位是 `clock()` **倒着走回去**，
        #     非匀速会让「退潮」的速度跟「涨潮」不对称，看着像两段不同的片子。
        # ⭐ 规矩写清楚，免得下次又删一遍：
        #     play 动**物体** → smooth（默认，别写 linear）
        #     play 动**时钟**（ValueTracker + always_redraw）→ **必须显式 linear**
        # ══════════════════════════════════════════════════════════════
        LIN = rate_functions.linear
        self.add(caps[0], subs[0])
        self.play(tt.animate.set_value(T_FWD),
                  run_time=T_FWD * SPEED, rate_func=LIN)

        # 前向结束 ——&#160;峰值标注在这一刻出现，和它指的那个山顶同时诞生
        self.remove(caps[0], subs[0])
        self.add(caps[1], subs[1], peak)
        self.play(tt.animate.set_value(T_FWD + T_BWD),
                  run_time=T_BWD * SPEED, rate_func=LIN)

        self.remove(caps[1], subs[1])
        self.add(caps[2], subs[2])
        self.play(tt.animate.set_value(T_RESET),
                  run_time=(T_UPD + T_HOLD) * SPEED, rate_func=LIN)

        # ⭐ 峰值标注在这里撤 ——&#160;山马上要退潮了，它不能比山顶活得久（traps §2.5）
        self.remove(caps[2], subs[2], peak)
        self.add(caps[3], subs[3])
        self.play(tt.animate.set_value(T_END),
                  run_time=T_REW * SPEED, rate_func=LIN)

        # ⭐⭐ 复位：此刻 clock() 已经是 0，图形部分自动等于首帧；
        #   **字幕也是首帧的一部分**，所以要一并拨回第一幕（traps §1.2）。
        #   ⭐ 末尾这个恒等 play 只为强制走一遍 updater ——&#160;`wait()` 不保证驱动
        #     `always_redraw`（traps §2.3），少了它末帧可能停在旧的一帧。
        self.remove(caps[3], subs[3])
        self.add(caps[0], subs[0])
        self.play(tt.animate.set_value(T_END), run_time=1 / 30, rate_func=LIN)
