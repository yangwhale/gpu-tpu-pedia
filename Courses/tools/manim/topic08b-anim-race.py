# -*- coding: utf-8 -*-
r"""专题九 §10.1「先查自己：同一份程序，跑出两种结果」的动画 —— RaceTwoRuns。挂在静态图 fig-race 下面。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · 竞争的本质是「两件事谁先做完没有定死」：同一份程序、同一批数据，两次跑的先后不一样，结果就不一样。
    「先后」只有放在时间里才看得见 —— 两幕的车道、活的长短完全一样，差别只是第二次「别的设备的写」早到了一点，
    本机读到一半时这块内存里已经被写进新数据。静态图只能并排放两个终态，画不出「同样的活，只是某一次早到了一点」。
  · 第三幕「不是每次都出现」也是时间：一次一次地跑，圆点一个一个冒出来，大多数是绿的，偶尔一个橙。

⛔ 数据从哪来
  · 示意（不是实测）：这块内存里的 8 份数据、读写各用多久、写早到多少，全部是示意。
    两条车道刻意不叫「小工」：全课「小工」指小核，而这里的竞争是本机的读和别的设备 all-to-all 的写，
    修法恰恰是把 all-to-all 交给小核，叫「小工甲／乙」会让人以为两颗小核在打架。
    「一块刚空出来的内存，上一个主人还在做最后一次读，另一个设备的 all-to-all 已经往里写新数据」
    是课件 §10.1 折叠「最可能的竞争对」里静态分析找到的最可能的一对 —— 推断，没有在硬件上直接证实。
  · 实测：第三幕「12 次里 1 次不一样」＝ topic08b_numbers.RACE_DET 里「对照 ＋ 一个转置开关」那一行（12 次、1 次分叉，断言）。
    同一份编好的程序、同一批数据反复跑，跟第一次逐位比。橙点放在第几个是示意（实测没有记录是第几次）。
  · 第二幕本机读到「半新半旧」的那几份：由下面的读写时刻当场算（第 k 份读的时刻 ≥ 别的设备写完第 k 份的时刻 → 读到新数据），
    断言第一幕 8 份全旧、第二幕前 4 份旧后 4 份新。

⛔ 刻意没画
  · 不画 all-to-all 的另一端设备、不画「坏的是一整块专家、只有少数设备坏」等细节；不写开关名字。
  · 第三幕的 12 次跟前两幕不是同一组实验：前两幕是机制示意，第三幕是实测的「出现频率」。
  · 第一次跑的结果框写「和第一次逐位相同」（课件口径：同一份程序反复跑，跟第一次逐位比）。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-race.py RaceTwoRuns WebPages/media/topic08b-race.mp4
   不需要「下一步」停顿（两幕各自停 1.5 秒看结果），不写 steps json。

⭐ 写法：一个时钟 T 走完全程，画面全是 T 的纯函数；收尾 0.4 秒把内容淡回第 0 帧的样子，末帧按定义 ＝ 第 0 帧。
"""
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Square, Circle, Triangle, Line, Arrow, DashedVMobject, Text,
                   ValueTracker, always_redraw, interpolate_color,
                   WHITE, GREY, GREY_B, BLUE, RED, GREEN, YELLOW, ORANGE,
                   UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08b_numbers as N  # noqa: E402

# ════════════════════════════════════════════════════════════════
# ① 数据
# ════════════════════════════════════════════════════════════════
_row = [r for r in N.RACE_DET if r[0] == "对照 ＋ 一个转置开关"]
assert len(_row) == 1
_, RUNS, BAD = _row[0]
assert (RUNS, BAD) == (12, 1)          # 课件 §10.1：「单开就 12 次里 1 次分叉」
BAD_IDX = 7                            # 第几个是橙点：示意

# ── 一次跑的时间轴（秒，示意）：本机读 8 份，别的设备写 8 份（写得快）
NS = 8
R0, SL = 0.4, 0.375                    # 本机从 R0 开始读，每份 SL
WL = 0.10                              # 别的设备每份 WL
S1, S2 = 3.5, 1.5                      # 别的设备开始写：第 1 次 / 第 2 次（早到了一点）
AX = 4.4                               # 一次跑的轴长
assert R0 + NS * SL < S1 and S1 + NS * WL <= AX


def read_t(k):
    return R0 + (k + 0.5) * SL


def write_done(s, k):
    return s + (k + 1) * WL


def got_new(s):
    return [read_t(k) >= write_done(s, k) for k in range(NS)]


assert got_new(S1) == [False] * NS                     # 第 1 次：全是旧料
assert got_new(S2) == [False] * 4 + [True] * 4         # 第 2 次：半新半旧

# ── 视频时刻
P1 = 1.0                    # 第 1 次跑开始
RES1 = P1 + AX              # 第 1 次结果框
CLR = 7.0                   # 清场（6.7–7.0 淡出）
P2 = 7.5                    # 第 2 次跑开始
RES2 = P2 + AX
D0, DSTEP = 13.4, 0.25      # 第三幕：圆点逐个出现
TALLY = D0 + RUNS * DSTEP + 0.25
END0, END1, T_END = 20.0, 20.4, 20.7    # 收尾：END0–END1 淡回第 0 帧，再静止到 T_END
T_MIX = P2 + read_t(4)      # 第 2 次里本机第一次读到新数据的时刻
assert abs(T_MIX - 9.59) < 0.01

TITLE = "同一份程序，跑出两种结果"
CAPS = [
    ("同一份程序，同一批数据，跑两次", 0.0, P1),
    ("这一次：先读完，再写", P1, CLR),
    ("这一次：写早到了一点", CLR, 9.6),
    ("读到半新半旧：结果从这一步起不一样", 9.6, D0),
    ("竞争不报错，也不是每次都出现", D0, TALLY),
    ("一格只改一样，每格多跑几次，数坏了几次", TALLY, END0),
]
for s, _, _ in CAPS:                    # 每句 ≤ 22 字
    assert sum(0.5 if ord(ch) < 128 else 1 for ch in s) <= 22, s
RUN1_LAB, RUN2_LAB, SKETCH = "第 1 次跑", "第 2 次跑（同一份程序、同一批数据）", "（示意）"
LANE_A, LANE_B = "本机：还在读", "别的设备：写进来"
CELL_LAB, GOT_LAB = "同一块内存", "本机读到的"
RES1_TXT, RES2_TXT = "和第一次逐位相同 ✓", "从这一步起不一样"
EARLY = "早到了一点"
DOT_LAB = "每个点 ＝ 跑一次（实测）"
TALLY_TXT = "%d 次里 %d 次不一样" % (RUNS, BAD)

# ── 几何
XA, XB = -3.8, 0.5                  # 时间轴（整体右移 1.1：车道标签变长，左边要留地方）
Y_A, Y_B, H_L = 1.30, 0.20, 0.50    # 两条车道
CX0, CW = 2.55, 0.45                # 同一块内存：里面 8 份数据
Y_CELL, Y_GOT = 1.05, -0.45
Y_RES = -1.45
Y_DOT = -2.45


def xt(tau):
    return XA + (XB - XA) * tau / AX


def clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))


def win(T, a, b, f=0.15):
    """[a, b) 内可见，两端各 f 秒淡入淡出。"""
    if T < a or T >= b:
        return 0.0
    return min(clamp((T - a) / f), clamp((b - T) / f))


def run_state(T):
    """返回 (别的设备开始写的时刻, 轴上时刻 τ, 内容透明度)；不在跑时 s=None。"""
    if T < CLR:
        return S1, clamp(T - P1, 0, AX), (1.0 if T < CLR - 0.3 else clamp((CLR - T) / 0.3))
    if T < END0:
        return S2, clamp(T - P2, 0, AX), 1.0
    return S2, AX, clamp((END1 - T) / (END1 - END0))


def tx(s, size=24, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


# ════════════════════════════════════════════════════════════════
# ② 画面
# ════════════════════════════════════════════════════════════════
class RaceTwoRuns(Scene):
    def construct(self):
        clk = ValueTracker(0.0)

        def scene_at():
            T = clk.get_value()
            s, tau, ca = run_state(T)
            g = []
            # 车道框
            for y in (Y_A, Y_B):
                g.append(Rectangle(width=XB - XA, height=H_L, stroke_color=GREY, stroke_width=2)
                         .move_to([(XA + XB) / 2, y, 0]))
            if ca > 0:
                # 本机：读（蓝）
                ra, rb = R0, min(R0 + NS * SL, tau)
                if rb > ra:
                    g.append(Rectangle(width=xt(rb) - xt(ra), height=H_L - 0.1, stroke_width=0,
                                       fill_color=BLUE, fill_opacity=0.75 * ca).move_to([(xt(ra) + xt(rb)) / 2, Y_A, 0]))
                # 别的设备：写（橙＝all-to-all 写进来的新数据）
                wa, wb = s, min(s + NS * WL, tau)
                if wb > wa:
                    g.append(Rectangle(width=xt(wb) - xt(wa), height=H_L - 0.1, stroke_width=0,
                                       fill_color=ORANGE, fill_opacity=0.9 * ca).move_to([(xt(wa) + xt(wb)) / 2, Y_B, 0]))
                # 游标
                if 0 < tau < AX:
                    g.append(Line([xt(tau), Y_B - H_L / 2 - 0.1, 0], [xt(tau), Y_A + H_L / 2 + 0.1, 0],
                                  color=YELLOW, stroke_width=4))
            # 同一块内存：每份数据的颜色（蓝旧、橙新）
            for k in range(NS):
                fr = clamp((tau - (s + k * WL)) / WL) * ca
                x = CX0 + (k + 0.5) * CW
                g.append(Square(side_length=CW - 0.06, stroke_color=GREY_B, stroke_width=2,
                                fill_color=interpolate_color(BLUE, ORANGE, fr), fill_opacity=0.9).move_to([x, Y_CELL, 0]))
                # 本机读到的
                g.append(Square(side_length=CW - 0.06, stroke_color=GREY, stroke_width=2,
                                fill_opacity=0).move_to([x, Y_GOT, 0]))
                if ca > 0 and tau >= read_t(k):
                    new = read_t(k) >= write_done(s, k)
                    g.append(Square(side_length=CW - 0.06, stroke_width=0,
                                    fill_color=ORANGE if new else BLUE, fill_opacity=0.9 * ca).move_to([x, Y_GOT, 0]))
            if ca > 0:
                # 读指针（黄，格子下方）／写指针（橙，格子上方）
                if R0 <= tau < R0 + NS * SL:
                    k = int((tau - R0) / SL)
                    g.append(Triangle(color=YELLOW, fill_color=YELLOW, fill_opacity=ca, stroke_width=0)
                             .scale(0.12).move_to([CX0 + (k + 0.5) * CW, Y_CELL - CW / 2 - 0.16, 0]))
                if s <= tau < s + NS * WL:
                    k = int((tau - s) / WL)
                    g.append(Triangle(color=ORANGE, fill_color=ORANGE, fill_opacity=ca, stroke_width=0)
                             .scale(0.12).rotate(3.14159).move_to([CX0 + (k + 0.5) * CW, Y_CELL + CW / 2 + 0.16, 0]))
            return VGroup(*g)

        body = always_redraw(scene_at)

        # 静态件
        title = tx(TITLE, 34, WHITE, True).to_edge(UP, buff=0.3)
        static = VGroup(
            tx(LANE_A, 24, BLUE).next_to([XA, Y_A, 0], LEFT, buff=0.2),
            tx(LANE_B, 24, ORANGE).next_to([XA, Y_B, 0], LEFT, buff=0.2),
            tx(CELL_LAB, 24, GREY_B).move_to([CX0 + NS * CW / 2, Y_CELL + 0.72, 0]),
            tx(GOT_LAB, 22, GREY_B).next_to([CX0, Y_GOT, 0], LEFT, buff=0.2),
        )

        # 跟时钟切透明度的字（建一次）
        faders = []

        def fade(m, fn):
            def upd(mm):
                o = fn(clk.get_value())
                mm.set_opacity(o)
            m.add_updater(upd)
            upd(m)                     # ⛔ 先手动算一次：updater 等第一次 play 才跑
            faders.append(m)

        cap_y = title.get_bottom()[1] - 0.35
        for i, (s, a, b) in enumerate(CAPS):
            col = YELLOW if i in (2, 3) else GREY_B
            m = tx(s, 26, col).move_to([0, cap_y, 0])
            if i == 0:
                fade(m, lambda T, b=b: 1.0 if T < b - 0.15 else (win(T, -1, b) if T < END0 else clamp((T - (END1 - 0.2)) / 0.2)))
            else:
                fade(m, lambda T, a=a, b=b: win(T, a, b))

        y_run = 2.05
        r1 = tx(RUN1_LAB, 24, WHITE, True)
        r1.move_to([XA - 1.75 + r1.width / 2, y_run, 0])
        r2 = tx(RUN2_LAB, 24, WHITE, True)
        r2.move_to([XA - 1.75 + r2.width / 2, y_run, 0])
        sk = tx(SKETCH, 22, GREY)
        sk.move_to([CX0 + NS * CW - sk.width / 2, y_run, 0])
        fade(r1, lambda T: 1.0 if T < CLR - 0.15 else (win(T, -1, CLR) if T < END0 else clamp((T - (END1 - 0.2)) / 0.2)))
        fade(r2, lambda T: win(T, CLR, END0 + 0.2, 0.15))
        self.add(sk)

        def res_box(s, col):
            t = tx(s, 26, col, True)
            box = Rectangle(width=t.width + 0.5, height=t.height + 0.35, stroke_color=col, stroke_width=4,
                            fill_opacity=0)
            return VGroup(box, t).move_to([CX0 + NS * CW / 2, Y_RES, 0])
        fade(res_box(RES1_TXT, GREEN), lambda T: win(T, RES1, CLR - 0.15))
        fade(res_box(RES2_TXT, ORANGE), lambda T: win(T, RES2, END0 + 0.3, 0.3))

        # 「早到了一点」：第 1 次写的位置留一个虚框，箭头指向第 2 次
        ghost = DashedVMobject(Rectangle(width=xt(S1 + NS * WL) - xt(S1), height=H_L - 0.1, stroke_color=ORANGE,
                                         stroke_width=3).move_to([(xt(S1) + xt(S1 + NS * WL)) / 2, Y_B, 0]), num_dashes=16)
        arr = Arrow([xt(S1), Y_B - H_L / 2 - 0.22, 0], [xt(S2), Y_B - H_L / 2 - 0.22, 0], buff=0,
                    color=YELLOW, stroke_width=5, max_tip_length_to_length_ratio=0.12)
        early = tx(EARLY, 22, YELLOW).move_to([(xt(S1) + xt(S2)) / 2, Y_B - H_L / 2 - 0.55, 0])
        for m in (ghost, arr, early):
            fade(m, lambda T: win(T, CLR + 0.2, END0 + 1, 0.3) if T < END0 else clamp((END1 - T) / (END1 - END0)))

        # 第三幕：12 个圆点
        dots = []
        x_d0 = -(RUNS - 1) * 0.55 / 2
        for i in range(RUNS):
            c = Circle(radius=0.17, stroke_width=0, fill_color=ORANGE if i == BAD_IDX else GREEN, fill_opacity=1)
            c.move_to([x_d0 + i * 0.55, Y_DOT, 0])
            fade(c, lambda T, i=i: win(T, D0 + i * DSTEP, END0 + 0.3, 0.12) if T < END0 else clamp((END1 - T) / (END1 - END0)))
            dots.append(c)
        dl = tx(DOT_LAB, 22, GREY_B).next_to(dots[0], LEFT, buff=0.35)
        tl = tx(TALLY_TXT, 26, ORANGE, True).move_to([0, Y_DOT - 0.62, 0])
        fade(dl, lambda T: win(T, D0, END0 + 0.3) if T < END0 else clamp((END1 - T) / (END1 - END0)))
        fade(tl, lambda T: win(T, TALLY, END0 + 0.3) if T < END0 else clamp((END1 - T) / (END1 - END0)))

        self.add(title, static, body, *faders)

        self.play(clk.animate.set_value(T_END), run_time=T_END, rate_func=linear)
        self.wait(1 / 30)


if __name__ == "__main__":
    print(got_new(S1), got_new(S2), T_MIX)
