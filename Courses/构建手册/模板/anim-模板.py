# -*- coding: utf-8 -*-
r"""Manim 动画最小模板（时钟式）—— 复制成 tools/manim/topic0N-anim-<名>.py 再改。

草稿（几秒钟，迭代阶段一律用它）：
    构建手册/脚本/render.sh 构建手册/模板/anim-模板.py RingReduceScatter /tmp/anim-demo.mp4 --draft
然后看 /tmp/loopdiff-anim-demo.png（上＝首帧，下＝末帧），两帧应该是同一幕。

⭐ 文件头写三样（跟静态图一样）：
   · 为什么非动不可：增量来自「时间」—— 四张卡转三步，每一步每张卡都有一块多加进一个人。
     静态图能画出每一步的结果，画不出「所有线同时在用」。
   · 数据从哪来：每一步每张卡手里那块是几个人的和，由 owners() 现算，并断言最后一步是四人之和。
   · 刻意没画的：AllGather 那后半圈（另一支动画讲），以及延迟。

⭐ 写法：一个 ValueTracker 当时钟，画面全部写成「第 t 秒长什么样」的纯函数（always_redraw）。
   好处：每一帧都能由 t 算出来 —— 结尾把时钟拨回 0，首帧就回来了，loop 不跳。
   字幕也挂在时钟上，用 opacity 切换，不用 FadeIn／FadeOut 序列 —— 复位时字幕自动回到第一句。
"""
from manim import (Scene, VGroup, Rectangle, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, RED, GREEN, YELLOW, UP, DOWN, linear)

# ── 房规配色：manim 原生，别设 background_color（默认黑底就是 3Blue1Brown 的用法）──
CARD_COL = [BLUE, RED, GREEN, YELLOW]          # 四张卡各一色（黄在这里只是第四种卡色）
N = 4                                          # 卡数
STEP_T = 2.0                                   # 每一步多少秒
T_END = STEP_T * (N - 1)                       # 三步走完的时刻


# ── ① 数据当场算：第 s 步之后，卡 c 手里第 k 块是哪几张卡的和 ─────────────
def owners(c, k, s):
    """环形 ReduceScatter：第 j 步卡 c 把第 (c−j) 块发给右边，右边加到自己的同号块上。"""
    got = {cc: {kk: {cc} for kk in range(N)} for cc in range(N)}
    for j in range(s):
        new = {cc: {kk: set(v) for kk, v in got[cc].items()} for cc in range(N)}
        for cc in range(N):
            blk = (cc - j) % N
            new[(cc + 1) % N][blk] |= got[cc][blk]
        got = new
    return got[c][k]


# 最后一步：卡 c 的第 (c+1) 块是四张卡之和 —— 动画的论点必须是算出来成立的
assert all(owners(c, (c + 1) % N, N - 1) == set(range(N)) for c in range(N))


class RingReduceScatter(Scene):
    def construct(self):
        t = ValueTracker(0.0)                  # ② 唯一的时钟

        title = Text("环形 ReduceScatter：每人只跟右边的邻居说话", font_size=30).to_edge(UP)
        self.add(title)

        # ③ 画面 ＝ t 的纯函数。每块按「是几个人的和」画成几条色带
        def blocks():
            s = min(int(t.get_value() / STEP_T + 1e-6), N - 1)
            g = VGroup()
            for c in range(N):
                for k in range(N):
                    x, y = -2.4 + k * 1.6, 1.2 - c * 1.1
                    who = sorted(owners(c, k, s))
                    w = 1.3 / len(who)
                    for i, o in enumerate(who):
                        g.add(Rectangle(width=w, height=0.8, stroke_width=0,
                                        fill_color=CARD_COL[o], fill_opacity=0.85)
                              .move_to([x - 0.65 + w * (i + 0.5), y, 0]))
                    g.add(Rectangle(width=1.3, height=0.8, stroke_color=GREY,
                                    stroke_width=2).move_to([x, y, 0]))
            return g
        self.add(always_redraw(blocks))

        # ④ 字幕挂在时钟上，按区间切 opacity（建一次，不在 always_redraw 里每帧重排）
        caps = [(0.0, "开始：每张卡一整份，自己的颜色"),
                (STEP_T, "每一步：每人往右发一块，收到的加到同号块上"),
                (T_END, "三步之后：每人手里有一块是四个人的总和")]
        texts = [Text(s, font_size=24, color=GREY_B).next_to(title, DOWN, buff=0.25) for _, s in caps]

        def show_caption(m, i):
            lo = caps[i][0]
            hi = caps[i + 1][0] if i + 1 < len(caps) else 1e9
            m.set_opacity(1 if lo <= t.get_value() + 1e-6 < hi else 0)
        for i, m in enumerate(texts):
            show_caption(m, i)                 # ⛔ 先手动算一次：updater 要等第一次 play 才跑，
            m.add_updater(lambda m, i=i: show_caption(m, i))   # 不先算，第 0 帧三句字幕全叠在一起
            self.add(m)

        # ⑤ 走时钟：驱动 ValueTracker 的 play 必须 rate_func=linear，否则每个段界画面停一秒
        self.wait(0.8)
        for s in range(1, N):
            self.play(t.animate.set_value(s * STEP_T), run_time=1.0, rate_func=linear)
            self.wait(STEP_T - 1.0)
        self.wait(1.2)

        # ⑥ 复位：把时钟拨回 0 —— 画面和字幕都回到第 0 帧，首尾同帧是免费的
        self.play(t.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)
        self.wait(0.5)
