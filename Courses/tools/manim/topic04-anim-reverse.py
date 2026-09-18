# -*- coding: utf-8 -*-
r"""专题四 · 正向模式 vs 反向模式 ——&#160;`fig-reverse` 配的那段 9 秒动画

⭐⭐⭐ 2026-09-18 现场：「我记得三蓝一棕讲的好，还有它画的动图超级好用，
  多借鉴一下。」——&#160;这句话推翻了我上一轮那个「多一段视频就多一份维护，
  所以先只做两段」的决定。按动画候选排名表，这是第 2 名。

⭐⭐⭐ 为什么这一格值得动起来（排名表的判据：增量只来自「时间」或「第三维」）：
  `fig-reverse` 要讲的是「**你得重复几遍**」——&#160;
  ⛔ 而**重复本身就是时间**。静态图只能画「一边 N 条线、一边 1 条线」，
    读者得自己把 N 条线换算成「跑 N 趟有多累」。
  ⭐ 动起来就不用换算了：下面那条早早跑完停在那儿全亮，
    上面那条还在一趟一趟地吭哧 ——&#160;**那份等待就是 N 倍的代价本身**。

⭐⭐ 这段动画在说的，正是静态图那句「乘法是可交换的，从哪头乘看上去都一样」
  之后的那半句：**区别不在乘什么，在你得重复几遍。**

⛔ 四条刻意的取舍（跟 saddle 那段同一套规矩）：
  ① **一个字都不放。** 文字由页面图注和上面的静态图承担 ——&#160;
     视频里的字既不能选中也不能被读屏。
     ⭐ 于是「谁是正向谁是反向」只能靠**波的方向**：上排从左往右，下排从右往左。
     「跑了几趟」只能靠右边那一摞方块：上排堆满，下排只有一块。
  ② **两排同时开跑。** 这是全部论点所在 ——&#160;
     ⛔ 分成两段先后播就什么都没了，因为对比的是**同一段时间里各自走到哪**。
  ③ **配色跟 `fig-reverse` 对齐**：红 ＝ 正向模式，绿 ＝ 反向模式。
  ④ 中间留一段「全亮保持」，末尾**把时钟拨回 0**，所以 `loop` 不跳帧。
     ⭐ 判据：**循环动画的第一帧和最后一帧必须长得一样**，
       否则「结束时全亮、开头全灭」每一轮都会闪一下。
     ⛔ 这里原本写的是「首尾都是全灭」——&#160;**那句话是错的，而且错了很久**：
       第一帧并不空，它有一个正在跑的亮节点、两个球、两排各一块计数。
       正因为把首帧想成「全灭」，复位才被写成「末尾全部关掉」，
       于是首末永远对不上（实测 32.7%）。
       ⭐⭐ 教训：**「复位」的目标不是「空」，是「第一帧」** ——&#160;
         这两个只有在第一帧真的是空的时候才重合。

📌 渲染（需要 `~/.venvs/manim`）：
    ~/.venvs/manim/bin/manim --format=mp4 -qh --media_dir /tmp/manim-out \
        tools/manim/topic04-anim-reverse.py Reverse
    ffmpeg -y -i /tmp/manim-out/videos/topic04-anim-reverse/1080p60/Reverse.mp4 \
        -vf "scale=960:-2" -c:v libx264 -crf 30 -preset slow \
        -pix_fmt yuv420p -movflags +faststart -an \
        WebPages/media/topic04-reverse.mp4
"""
from manim import (Scene, Circle, Square, Line, Dot, VGroup, ValueTracker,
                   always_redraw, WHITE, LEFT, RIGHT, UP, DOWN, ORIGIN,
                   linear, config)

RD_ = "#d93025"          # 正向模式
GR_ = "#1e8e3e"          # 反向模式
GY_ = "#9aa0a6"          # 还没拿到
INK_ = "#202124"

N = 8                    # 参数个数 ——&#160;正向模式要跑这么多趟
X0, X1 = -5.4, 0.2       # 参数点从哪到哪
LOSS_X = 1.5             # 末端那个「一个数」
TALLY_X = 2.9            # 右边计数方块从这儿开始堆
TALLY_W = 0.33
Y_FWD, Y_BWD = 1.35, -1.35

T_ROUND = 0.95                  # 正向：一个参数一趟
T_FWD = N * T_ROUND             # 7.6 ——&#160;跑满八趟
T_BWD = 1.5                     # 反向：一趟走完
T_HOLD = T_FWD + 0.8            # 8.4 ——&#160;全亮停一下
T_END = T_HOLD + 0.6            # 9.0 ——&#160;复位，好让循环无缝

# ⭐ 这段动画的全部论点就是这个比值 ——&#160;它必须大到"看得出在等"
assert T_FWD / T_BWD > 4.0, "正向跑完的时间没比反向长出量级感，对比就不成立"
assert T_END > T_HOLD, "结尾要留复位段，否则循环会闪"

PX = [X0 + (X1 - X0) * i / (N - 1.0) for i in range(N)]


def _lerp(a, b, s):
    return a + (b - a) * max(0.0, min(1.0, s))


class Reverse(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        tr = ValueTracker(0.0)

        # ⭐⭐ 2026-09-18：这支片子**首尾对不上**，量出来 32.7%。
        #   根因不是漏写复位 ——&#160;复位写了，但**是在每个绘制函数里各写一份
        #   `if t >= T_END - 0.6` 分支**，四份里漏了两份（两个球、两排计数块），
        #   于是末帧比首帧少两个球、少两块计数。
        # ⭐ 判据：**同一条不变量复制 N 份，N 越大越必然漏。**
        #   改成在**时间轴上做一次映射** ——&#160;复位段直接把时钟拨回 0，
        #   于是「末帧 ≡ 首帧」不是靠 N 个分支凑出来的，是按定义成立的。
        T_RESET = T_END - 0.6

        def clock():
            t = tr.get_value()
            return 0.0 if t >= T_RESET else t

        def rail(y):
            """两排共用的骨架 ——&#160;一条链 ＋ 末端一个方块。"""
            g = VGroup()
            g.add(Line([X0 - 0.35, y, 0], [LOSS_X - 0.32, y, 0],
                       stroke_color=GY_, stroke_width=3))
            g.add(Square(side_length=0.46, stroke_color=INK_, stroke_width=3,
                         fill_color=INK_, fill_opacity=0.10).move_to([LOSS_X, y, 0]))
            return g

        self.add(rail(Y_FWD), rail(Y_BWD))

        def node(i, y, col, st_fn):
            """一个参数点，**三态**：
              0 灰空心 ＝ 还没轮到　1 彩色空心 ＝ 这一趟正在为它跑　2 实心 ＝ 拿到了
            ⭐ 第 1 态是后加的：没有它，读者看不出「这一趟是为谁跑的」——&#160;
              而「一趟只能拿一个参数的梯度」正是正向模式贵在哪儿。
            """
            def mk():
                st = st_fn(clock(), i)
                c = Circle(radius=0.145,
                           stroke_color=GY_ if st == 0 else col,
                           stroke_width=3.2 if st != 1 else 4.6,
                           fill_color=col, fill_opacity=1.0 if st == 2 else 0.0)
                return c.move_to([PX[i], y, 0])
            return always_redraw(mk)

        # ── 上排：正向模式。第 k 趟只点亮第 k 个 ─────────────────────
        def fwd_lit(t, i):
            # ⭐ 不再自己判复位 —— clock() 已经把复位段拨回 t=0
            if t >= (i + 1) * T_ROUND:
                return 2                  # 这一趟跑完了，梯度拿到了
            if t >= i * T_ROUND:
                return 1                  # 正在为它跑
            return 0

        # ── 下排：反向模式。一趟走过去，沿途全点亮 ───────────────────
        def bwd_lit(t, i):
            if t >= T_BWD:
                return 2
            # 球从 loss 往左走，走过了就点亮 ——&#160;一趟之内全都拿到
            bx = _lerp(LOSS_X, PX[0], t / T_BWD)
            return 2 if PX[i] >= bx else 0

        for i in range(N):
            self.add(node(i, Y_FWD, RD_, fwd_lit))
            self.add(node(i, Y_BWD, GR_, bwd_lit))

        # ── 两个跑动的小球 ──────────────────────────────────────────
        def fwd_ball():
            t = clock()
            if t >= T_FWD:
                return Dot(radius=0.001, fill_opacity=0.0).move_to(ORIGIN)
            k = min(N - 1, int(t / T_ROUND))
            s = (t - k * T_ROUND) / T_ROUND
            return Dot(radius=0.125, color=RD_).move_to(
                [_lerp(PX[k], LOSS_X, s), Y_FWD, 0])

        def bwd_ball():
            t = clock()
            if t >= T_BWD:
                return Dot(radius=0.001, fill_opacity=0.0).move_to(ORIGIN)
            return Dot(radius=0.125, color=GR_).move_to(
                [_lerp(LOSS_X, PX[0], t / T_BWD), Y_BWD, 0])

        self.add(always_redraw(fwd_ball), always_redraw(bwd_ball))

        # ── 右边的计数：跑一趟堆一块。这是「重复几遍」的无字说法 ─────
        def tally(y, col, n_fn):
            def mk():
                t = clock()
                g = VGroup()
                for j in range(n_fn(t)):
                    g.add(Square(side_length=0.26, stroke_color=col,
                                 stroke_width=2, fill_color=col,
                                 fill_opacity=0.8)
                          .move_to([TALLY_X + j * TALLY_W, y, 0]))
                return g
            return always_redraw(mk)

        self.add(tally(Y_FWD, RD_, lambda t: min(N, int(t / T_ROUND) + 1)))
        self.add(tally(Y_BWD, GR_, lambda t: 1))

        self.play(tr.animate.set_value(T_END), run_time=T_END, rate_func=linear)
