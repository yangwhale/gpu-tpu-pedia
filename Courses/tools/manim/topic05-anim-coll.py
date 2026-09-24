# -*- coding: utf-8 -*-
r"""专题五 · 第一节「先认识五种通信」的两支动画。

⭐ 为什么只做这两支（skill 第 0 步：增量只能来自「时间」或「第三维」）：
  · Ring ——&#160;环形传递**本身就是一步一步的过程**，静态图只能拍四张快照。
    而且 ReduceScatter 转三步、AllGather 再转三步，**放在同一段里一口气演完**，
    「AllReduce ＝ 前两个接起来」就不用再单独画一张了。
  · AllToAll ——&#160;每一块**飞去哪**是这个原语的全部内容；静态图只能画前后两张表。
    ⭐ 顺手白捡一个无缝循环：派发过去再原路送回来（MoE 每层正是这两次），
      第二次转置把画面还原成第 0 帧 ——&#160;**首尾同帧是这个原语自己的性质**。
  ⛔ 八个原语的总览（fig5-coll-1n / coll-nn）是**并列对比**，不做动画。

⭐ 画法跟静态图一一对应，读者不用重学：
  四张卡、一张一个颜色（蓝 / 橙 / 绿 / 紫）；每张卡四块；
  **加过的块画成竖条纹，条纹的颜色就是参与相加的那几张卡。**

⛔ 数据当场算：环形每一步谁发哪一块、收到后变成什么，都由
  `ring_rs_step` / `ring_ag_step` 按调度现算，并断言 RS 三步后卡 k 恰好握着第 k 块的完整总和、
  AG 三步后人人四块全满。块号约定跟 topic05-fig-coll.py 一致（第 s 步卡 k 发第 (k−s−1) mod 4 块）。

📌 渲染：~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic05-anim-coll.py Ring WebPages/media/topic05-ring.mp4
"""
from manim import (Scene, VGroup, Rectangle, Text, Arrow, CurvedArrow, FadeIn, FadeOut,
                   Indicate, AnimationGroup, WHITE, GREY, BLUE, GREEN, ORANGE, PURPLE_B,
                   UP, DOWN, LEFT, RIGHT, ORIGIN)

N = 4
COL = [BLUE, ORANGE, GREEN, PURPLE_B]
NAME = "ABCD"
CW, CH, GAPY = 1.05, 0.56, 0.18
XS = [-4.8, -1.6, 1.6, 4.8]
Y0 = 1.25                                      # 第 0 块的中心高度


def cy(j):
    return Y0 - j * (CH + GAPY)


def chunk(contrib, j, x, bold=False):
    """一块：contrib 是参与相加的卡号集合。条纹 ＝ 加过。"""
    g = VGroup()
    c = sorted(contrib)
    w = CW / len(c)
    for i, k in enumerate(c):
        g.add(Rectangle(width=w, height=CH, stroke_width=0, fill_color=COL[k],
                        fill_opacity=0.85).move_to([x - CW / 2 + w * (i + 0.5), cy(j), 0]))
    g.add(Rectangle(width=CW, height=CH, stroke_color=WHITE,
                    stroke_width=5 if bold else 1.5).move_to([x, cy(j), 0]))
    if len(c) == N:
        lab = "Σ%d" % j
    elif len(c) == 1:
        lab = "%s%d" % (NAME[c[0]], j)
    else:
        lab = ""
    if lab:
        g.add(Text(lab, font_size=24, color=WHITE, weight="BOLD").move_to([x, cy(j), 0]))
    return g


def cards(held, hot=()):
    g = VGroup()
    for k in range(N):
        for j in range(N):
            g.add(chunk(held[k][j], j, XS[k], (k, j) in hot))
    return g


def labels():
    return VGroup(*[Text("卡 %d" % k, font_size=30, color=COL[k]).move_to([XS[k], Y0 + 0.75, 0])
                    for k in range(N)])


def ring_arrows():
    g = VGroup()
    ybar = cy(1.5)
    for k in range(N - 1):
        g.add(Arrow([XS[k] + CW / 2 + 0.12, ybar, 0], [XS[k + 1] - CW / 2 - 0.12, ybar, 0],
                    color=GREY, stroke_width=4, buff=0))
    g.add(CurvedArrow([XS[3], cy(3) - CH / 2 - 0.15, 0], [XS[0], cy(3) - CH / 2 - 0.15, 0],
                      angle=-0.55, color=GREY, stroke_width=4))
    return g


def start_state():
    return [[{k} for _ in range(N)] for k in range(N)]


def ring_rs_step(held, s):
    sends = [(k, (k - s - 1) % N) for k in range(N)]
    new = [[set(c) for c in r] for r in held]
    for k, j in sends:
        new[(k + 1) % N][j] |= held[k][j]
    return new, sends


def ring_ag_step(held, s):
    # AllGather：卡 k 把它最近拿到的那块完整总和往右传（第 s 步发第 (k−s) mod 4 块），只替换不相加
    sends = [(k, (k - s) % N) for k in range(N)]
    new = [[set(c) for c in r] for r in held]
    for k, j in sends:
        assert held[k][j] == set(range(N)), (k, j, held[k][j])
        new[(k + 1) % N][j] = set(held[k][j])
    return new, sends


# 当场算一遍，断言调度对
_h = start_state()
for _s in range(N - 1):
    _h, _ = ring_rs_step(_h, _s)
for _k in range(N):
    assert _h[_k][_k] == set(range(N))
for _s in range(N - 1):
    _h, _ = ring_ag_step(_h, _s)
assert all(_h[k][j] == set(range(N)) for k in range(N) for j in range(N))


class Ring(Scene):
    def construct(self):
        held = start_state()
        labs, arr = labels(), ring_arrows()
        state = cards(held)
        self.add(labs, arr, state)
        cap = Text(" ", font_size=30).to_edge(UP)
        self.add(cap)
        self.wait(0.6)

        def swap_caption(txt, color=WHITE):
            nonlocal cap
            new = Text(txt, font_size=30, color=color).to_edge(UP)
            self.play(FadeOut(cap), FadeIn(new), run_time=0.4)
            cap = new

        def fly(sends, new_held, merge):
            nonlocal state, held
            movers = []
            for k, j in sends:
                src = chunk(held[k][j], j, XS[k])
                self.add(src)
                movers.append((src, (k + 1) % N, j))
            self.play(*[m.animate.move_to([XS[d], cy(j), 0]) for m, d, j in movers],
                      run_time=0.9)
            hot = {(d, j) for _, d, j in movers}
            new_state = cards(new_held, hot)
            self.remove(state, *[m for m, _, _ in movers])
            self.add(new_state)
            state, held = new_state, new_held
            self.wait(0.45)

        swap_caption("ReduceScatter：每人往右发一块，收到的加到自己那块上")
        for s in range(N - 1):
            nh, sends = ring_rs_step(held, s)
            fly(sends, nh, True)
        swap_caption("三步之后：每张卡恰好握着一块完整总和", GREEN)
        self.wait(0.8)

        swap_caption("AllGather：把总和接着往右传，只替换，不相加")
        for s in range(N - 1):
            nh, sends = ring_ag_step(held, s)
            fly(sends, nh, False)
        swap_caption("人人一份总和　＝　AllReduce ＝ ReduceScatter ＋ AllGather", GREEN)
        self.play(Indicate(state, color=WHITE, scale_factor=1.03), run_time=0.9)
        self.wait(0.8)

        # ⭐ 复位：清干净，再摆一份跟第 0 帧一模一样的静态件（skill 的通用收尾）
        self.play(FadeOut(state), FadeOut(cap), run_time=0.7)
        self.remove(state, cap)
        state = cards(start_state())
        self.play(FadeIn(state), run_time=0.7)
        self.add(Text(" ", font_size=30).to_edge(UP))
        self.wait(0.6)


class AllToAll(Scene):
    def construct(self):
        base = [[({k}, "%s%d" % (NAME[k], j)) for j in range(N)] for k in range(N)]

        def block(k_src, j_dst, x, y, bold=False):
            g = VGroup(Rectangle(width=CW, height=CH, stroke_width=0, fill_color=COL[k_src],
                                 fill_opacity=0.85).move_to([x, y, 0]),
                       Rectangle(width=CW, height=CH, stroke_color=WHITE,
                                 stroke_width=5 if bold else 1.5).move_to([x, y, 0]),
                       Text("%s%d" % (NAME[k_src], j_dst), font_size=24, color=WHITE,
                            weight="BOLD").move_to([x, y, 0]))
            return g

        labs = labels()
        self.add(labs)
        # blocks[(k, j)]：卡 k 手里、要去卡 j 的那一块；初始在 (卡 k, 第 j 行)
        blocks = {(k, j): block(k, j, XS[k], cy(j), k == j) for k in range(N) for j in range(N)}
        self.add(*blocks.values())
        cap = Text(" ", font_size=30).to_edge(UP)
        self.add(cap)
        self.wait(0.6)

        def swap_caption(txt, color=WHITE):
            nonlocal cap
            new = Text(txt, font_size=30, color=color).to_edge(UP)
            self.play(FadeOut(cap), FadeIn(new), run_time=0.4)
            cap = new

        swap_caption("派发：卡 k 的第 j 块 → 发给卡 j（粗框是自己留给自己的，不走网络）")
        # 转置：(k, j) 飞到 (卡 j, 第 k 行)
        self.play(*[b.animate.move_to([XS[j], cy(k), 0]) for (k, j), b in blocks.items()],
                  run_time=1.8)
        swap_caption("卡 j 收齐了四个人给它的那一份 —— 一张表转置了一次", GREEN)
        self.wait(1.2)
        swap_caption("专家算完，再转置一次送回去 —— MoE 每层两次 AllToAll")
        self.play(*[b.animate.move_to([XS[k], cy(j), 0]) for (k, j), b in blocks.items()],
                  run_time=1.8)
        self.wait(0.6)
        self.play(FadeOut(cap), run_time=0.4)
        self.remove(cap)
        self.add(Text(" ", font_size=30).to_edge(UP))
        self.wait(0.6)
