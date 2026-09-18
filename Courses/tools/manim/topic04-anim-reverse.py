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

═══════════════════════════════════════════════════════════════════
S1（2026-09-18 下午）：房规松绑后回来加解释性文字
═══════════════════════════════════════════════════════════════════

⭐⭐⭐ 先判房规要求的那一句：**它缺的是「解释」，不是「铺陈」。**

  ⭐ 判据摆在时间轴上：这一格的**唯一论点是「两排同时跑，一排早早停下、
    另一排还在吭哧」**，而这件事在第 1.5 秒（反向跑完）就已经**演完了**，
    剩下 6 秒全是它在**持续成立**。动作没有被赶时间压掉，
    也没有第二个起承转合要交代 ——&#160;**铺陈是够的。**
  ⛔ 真正缺的是**命名**：画面里没有任何东西说
    「上排＝正向模式、下排＝反向模式」「那摞小方块是趟数」
    「右端那个方块是 loss」「点亮＝这个参数的梯度到手了」。
    观众得先自己把「波往哪边走」反推成模式名，才谈得上看懂结论 ——&#160;
    **这是解释缺位，不是时间不够。**
  ⭐⭐ 所以 **`T_END` 一秒不动（仍是 9.0），`loop` 也不动**，
    加的全是字：两行角色抬头 ＋ 两个公式 ＋ 三幕字幕 ＋ 两条灰色图例。
    判据（房规）：**「不够 fancy」的解药是信息密度，不是时长。**

⭐⭐ 两个公式是这次唯一**新增的知识**，静态图里没有：
    正向 `∂L/∂wᵢ`（一趟只到手一个分量） vs 反向 `∇L`（一趟整个梯度到手）。
  ⛔ 其余的话一律不抄静态图 ——&#160;「右边只有一个数」「从窄的那一头起步」
    「三千亿倍」都是静态图 Ⓑ Ⓓ 已经讲透的，画面里一个字都不重复。

⭐ 字幕用**幕间显式 `add` / `remove`** 切换，不给 `Text` 挂 `add_updater`：
  ⛔ 同讲的 descend 那支实测过，每帧给几十个字形重设 opacity，
    -qh 从 28 it/s 单调掉到 0.47 it/s。
  ⭐ 而且幕与幕之间本来就是**离散事件**，做成每帧要问一次的连续量是错的建模。

⭐⭐⭐ 复位与字幕的交接（这次的新讲究）：**幕三→幕一的切换点就放在 `T_RESET`**。
  于是最后那 0.6 秒里，几何已经被 `clock()` 拨回 t=0、字幕也回到第一幕 ——&#160;
  **末帧按定义等于首帧，不需要在结尾额外补一次复位动作**。
  ⛔ 反过来写（先播完再换字幕）就得在最后补一个恒等 `play` 去刷帧，
    还会把总时长顶出 9.0。

⛔ 三条刻意的取舍（其余跟 saddle 那段同一套规矩）：
  ① **两排同时开跑。** 这是全部论点所在 ——&#160;
     ⛔ 分成两段先后播就什么都没了，因为对比的是**同一段时间里各自走到哪**。
  ② **配色跟 `fig-reverse` 对齐**：红 ＝ 正向模式，绿 ＝ 反向模式。
  ③ 中间留一段「全亮保持」，末尾**把时钟拨回 0**，所以 `loop` 不跳帧。
     ⭐ 判据：**循环动画的第一帧和最后一帧必须长得一样**，
       否则「结束时全亮、开头全灭」每一轮都会闪一下。
     ⛔ 这里原本写的是「首尾都是全灭」——&#160;**那句话是错的，而且错了很久**：
       第一帧并不空，它有一个正在跑的亮节点、两个球、两排各一块计数。
       正因为把首帧想成「全灭」，复位才被写成「末尾全部关掉」，
       于是首末永远对不上（实测 32.7%）。
       ⭐⭐ 教训：**「复位」的目标不是「空」，是「第一帧」** ——&#160;
         这两个只有在第一帧真的是空的时候才重合。

⛔ 文字进画面之后多出一条**硬义务**（房规①）：
  视频里的字选不中、读屏读不到、搜索搜不到 ——&#160;
  所以页面上的 `aria-label` 必须把**画面里出现过的每一句话**复述一遍。
  本文件末尾的 `SCREEN_TEXT` 就是那份清单，**改了字幕要同步改它**。

📌 渲染（需要 `~/.venvs/manim`）：
    LOOP_BASELINE=$PWD/tools/manim/loop-baseline.json \
    ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic04-anim-reverse.py Reverse WebPages/media/topic04-reverse.mp4
"""
import numpy as np
from manim import (Scene, Circle, Square, Line, Dot, VGroup, ValueTracker,
                   always_redraw, MathTex, Text, WHITE, LEFT, RIGHT, UP,
                   ORIGIN, linear, config)

RD_ = "#d93025"          # 正向模式
GR_ = "#1e8e3e"          # 反向模式
GY_ = "#9aa0a6"          # 还没拿到
GY2_ = "#80868b"         # 图例 / 说明文字
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

# ── 复位点。字幕的最后一次切换也钉在这里，见 construct() 末尾 ────────
T_RESET = T_END - 0.6

# ⭐⭐ traps §1.3：复位只能定义在一个地方。这条断言就是那句话的执法者 ——&#160;
#   全文件只许出现一次 tracker 取值，其余绘制函数一律读 `clock()`。
# ⛔ needle 必须拆开拼，否则**这一行自己**也会被数进去，断言永远为 2。
_NEEDLE = "tr.get_" + "value()"
_SRC = open(__file__, encoding="utf-8").read()
assert _SRC.count(_NEEDLE) == 1, \
    "有绘制函数绕过了 clock()（数到 %d 处）——&#160;复位就不再是单点定义了" \
    % _SRC.count(_NEEDLE)


# ══════════════════════════════════════════════════════════════════
# 字幕 ——&#160;三幕，幕界**从动画自己的节拍算出来**，不是挑的时间点
# ══════════════════════════════════════════════════════════════════
# ⭐ 幕一→幕二：反向跑完、正向刚好跑完第 2 趟（此时对比第一次成立）
# ⭐ 幕二→幕三：正向跑到第 5 趟，剩下三趟留给「还在跑」那句话
T_A2 = T_BWD + 1 * T_ROUND
T_A3 = T_BWD + 4 * T_ROUND
assert 0 < T_A2 < T_A3 < T_RESET, \
    "幕界必须严格递增且落在复位点之前：%.2f / %.2f / %.2f" % (T_A2, T_A3, T_RESET)

_CN_NUM = "零一二三四五六七八九十"


def _cn(n):
    """把 1..10 写成中文。

    ⭐ 字幕里那个「八」必须跟 `N` 同源 ——&#160;traps §3.3：
      注释（以及画面上的字）里的数也是一种断言，手打的那个会一声不响地过期。
    """
    assert 1 <= n <= 10, "中文数词只覆盖 1..10，N=%d 超出" % n
    return _CN_NUM[n]


CAPS = ("同一条链，两个方向同时开跑",
        "下面这条从 loss 出发，一趟就跑完了",
        "上面这条还在跑 ——　%s个参数，%s趟" % (_cn(N), _cn(N)))

# ⭐ 可读性也当数据校：每幕至少给到 9 字/秒的余量。
#   ⛔ 万一过不了，**正确反应是把字幕改短或把幕界挪开，不是把 CPS 调大** ——&#160;
#     traps §3.1：断言挡住时先看它挡的是什么。
_CPS = 9.0
_ACT_SPAN = (T_A2 - 0.0, T_A3 - T_A2, T_RESET - T_A3)
for _i, (_txt, _dur) in enumerate(zip(CAPS, _ACT_SPAN)):
    assert _dur >= len(_txt) / _CPS, \
        "第 %d 幕字幕来不及读：%d 字 / %.2f 秒 ＝ %.1f 字每秒" \
        % (_i + 1, len(_txt), _dur, len(_txt) / _dur)

# ⭐⭐ 房规①：`aria-label` 要复述画面里出现过的每一句话。
#   这份清单就是给页面抄的**唯一来源** ——&#160;改了上面的字幕，这里自动跟着变。
#   ⛔ 公式念法要写口语，读屏念不出 LaTeX。
SCREEN_TEXT = (
    ("角色抬头（全程）", "正向模式　每趟只到手　∂L/∂wᵢ",
     "读作：正向模式，每趟只到手，偏 L 偏 w i"),
    ("角色抬头（全程）", "反向模式　一趟就到手　∇L",
     "读作：反向模式，一趟就到手，nabla L，也就是整个梯度"),
    ("链条末端方块内（全程，上下各一个）", "L", "读作：L，也就是 loss"),
    ("中间灰字（全程）", "%d 个参数" % N, ""),
    ("中间灰字（全程）", "走过的趟数", ""),
    ("字幕第 1 幕", CAPS[0], ""),
    ("字幕第 2 幕", CAPS[1], ""),
    ("字幕第 3 幕", CAPS[2], ""),
)


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
            # ⭐ S1：给末端那个方块一个名字。整段的枢纽就是「这一头是 L」——&#160;
            #   不写出来，「从 loss 出发」这句字幕就没有落点。
            #   ⛔ 但**只命名，不解释**：「这一头只有一个数」是静态图 Ⓑ 的台词，
            #     房规①说了别把静态图已经说清的话再抄一遍进画面。
            g.add(MathTex("L", color=INK_).scale(0.48).move_to([LOSS_X, y, 0]))
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

        # ── 右边的计数：跑一趟堆一块 ────────────────────────────────
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

        # ══════════════════════════════════════════════════════════
        # S1 新增的字 ——&#160;全部是**常驻**的，所以对首末同帧零风险
        # ══════════════════════════════════════════════════════════
        RAIL_L = X0 - 0.35                      # 两排骨架的左端，抬头跟它对齐

        def header(name, verb, tex, col, y):
            """一行角色抬头：`模式名 ＋ 一句话 ＋ 一个公式`。

            ⭐⭐ 这一行是这次升级的**主要增量**：
              画面本来只能靠「波往哪边走」暗示谁是谁，观众得先反推一次。
            ⭐ 而右边那个公式是静态图里没有的那条信息 ——&#160;
              正向一趟只到手**一个分量** `∂L/∂wᵢ`，反向一趟到手**整个** `∇L`。
              两个记号一长一短，长短差本身就在说话。
            """
            g = VGroup(Text(name, font_size=26, color=col),
                       Text(verb, font_size=20, color=GY2_),
                       MathTex(tex, color=col).scale(0.72))
            g.arrange(RIGHT, buff=0.26)
            g.set_y(y)
            g.align_to(np.array([RAIL_L, 0.0, 0.0]), LEFT)
            return g

        head_f = header("正向模式", "每趟只到手", r"\partial L / \partial w_i",
                        RD_, Y_FWD + 0.70)
        head_b = header("反向模式", "一趟就到手", r"\nabla L",
                        GR_, Y_BWD - 0.70)

        # ── 中间那条灰色图例带：两排共用，所以放在 y=0 ────────────────
        # ⭐ 判据（借 descend 那次的教训）：**图例要贴着它解释的那个东西。**
        #   「N 个参数」正对着两排圆圈之间，「走过的趟数」正对着两摞方块之间。
        leg_w = Text("%d 个参数" % N, font_size=20, color=GY2_)
        leg_w.move_to([(PX[0] + PX[-1]) / 2.0, 0.0, 0])
        leg_t = Text("走过的趟数", font_size=20, color=GY2_)
        leg_t.move_to([TALLY_X + (N - 1) * TALLY_W / 2.0, 0.0, 0])

        caps = [Text(txt, font_size=30, color=INK_).to_edge(UP, buff=0.42)
                for txt in CAPS]

        self.add(head_f, head_b, leg_w, leg_t, caps[0])

        # ⭐⭐ 版面体检：这些字是**算出来的位置**，所以让脚本自己验一遍有没有撞车。
        #   ⛔ 判据：手摆的坐标没人能复核 ——&#160;traps §3.2 对文字一样成立。
        def _bbox(m):
            return (m.get_left()[0], m.get_right()[0],
                    m.get_bottom()[1], m.get_top()[1])

        _boxes = [("正向抬头", head_f), ("反向抬头", head_b),
                  ("参数图例", leg_w), ("趟数图例", leg_t), ("字幕", caps[0])]
        for _i in range(len(_boxes)):
            for _j in range(_i + 1, len(_boxes)):
                (na, a), (nb, b) = _boxes[_i], _boxes[_j]
                x0, x1, y0, y1 = _bbox(a)
                u0, u1, v0, v1 = _bbox(b)
                assert x1 < u0 or u1 < x0 or y1 < v0 or v1 < y0, \
                    "「%s」和「%s」的包围盒撞上了" % (na, nb)
        for _n, _m in _boxes:
            assert _bbox(_m)[0] > -config.frame_width / 2 and \
                   _bbox(_m)[1] < config.frame_width / 2 and \
                   _bbox(_m)[2] > -config.frame_height / 2 and \
                   _bbox(_m)[3] < config.frame_height / 2, "「%s」出画框了" % _n
        # ⭐ 抬头不许压到它那一排的圆圈（圆半径 0.145）
        assert _bbox(head_f)[2] > Y_FWD + 0.145, "正向抬头压到上排圆圈了"
        assert _bbox(head_b)[3] < Y_BWD - 0.145, "反向抬头压到下排圆圈了"

        # ══════════════════════════════════════════════════════════
        # 播放 ——&#160;四段，总时长仍是 T_END（9.0 秒），一秒没加
        # ══════════════════════════════════════════════════════════
        # ⭐⭐⭐ 最后那次字幕切换钉在 **T_RESET**，不是钉在片尾：
        #   于是收尾那 0.6 秒里，几何已被 clock() 拨回 t=0、字幕也回到第一幕，
        #   **末帧按定义 ≡ 首帧**，不用在结尾补恒等 play 去刷帧。
        # ⛔ 顺带避开 traps §2.3 那个坑（wait 不一定驱动 updater）——&#160;
        #   这里最后一段本来就是 play，updater 一定会走。
        def _seg(to_t, drop, show):
            self.play(tr.animate.set_value(to_t),
                      run_time=to_t - _seg.at, rate_func=linear)
            _seg.at = to_t
            if drop is not None:
                self.remove(drop)
                # ⭐ traps §2.2：`Scene.add` 是追加到最上层。这里安全 ——&#160;
                #   字幕在画面顶端，跟任何元素都不重叠（上面那组断言保证了）。
                self.add(show)
        _seg.at = 0.0

        _seg(T_A2, caps[0], caps[1])
        _seg(T_A3, caps[1], caps[2])
        _seg(T_RESET, caps[2], caps[0])
        _seg(T_END, None, None)
