# -*- coding: utf-8 -*-
r"""专题五 · 第一节「先认识五种通信」的动画：环、AllToAll，以及七个原语各一支（见文件后半）。

⭐ 最早只做了环和 AllToAll 两支（下面是当时的理由；2026-09-24 改判，见文件后半「一个原语一支」）：
  · Ring ——&#160;环形传递**本身就是一步一步的过程**，静态图只能拍四张快照。
    而且 ReduceScatter 转三步、AllGather 再转三步，**放在同一段里一口气演完**，
    「AllReduce ＝ 前两个接起来」就不用再单独画一张了。
  · AllToAll ——&#160;每一块**飞去哪**是这个原语的全部内容；静态图只能画前后两张表。
    ⭐ 顺手白捡一个无缝循环：派发过去再原路送回来（MoE 每层正是这两次），
      第二次转置把画面还原成第 0 帧 ——&#160;**首尾同帧是这个原语自己的性质**。
  ⛔ 当时认为八个原语的总览是并列对比、不做动画 —— 这条后来被推翻：总览图照旧并列，每个原语另配一支。

⭐ 画法跟静态图基本一致，只有一处对调：**动画里一列是一张卡，静态图里一行是一张卡**（黑底横向摆放更好看；
  课件 1.1「先学会看图」专门写了一句提醒）。其余一致：
  四张卡、一张一个颜色（蓝 / 橙 / 绿 / 紫）；每张卡四块；
  **加过的块画成竖条纹，条纹的颜色就是参与相加的那几张卡。**

⛔ 数据当场算：环形每一步谁发哪一块、收到后变成什么，都由
  `ring_rs_step` / `ring_ag_step` 按调度现算，并断言 RS 三步后卡 k 恰好握着第 k 块的完整总和、
  AG 三步后人人四块全满。块号约定跟 topic05-fig-coll.py 一致（第 s 步卡 k 发第 (k−s−1) mod 4 块）。

⛔⛔ 2026-09-25 现场纠正（原话要点）：「你怎么 A 往右，然后剩下都往左。它应该是环形的，大家都往右发，
  到头了转一圈回来……第一步慢点，还没看明白呢，停一下再跳第二步。」
  病根两处：① 环里卡 3 → 卡 0 那一块是**横穿整个画面往左飞**的，看上去就是「别人都往左」；
  ② AllGather／ReduceScatter／AllReduce 三支用的是「块直接飞到目的地」的逻辑视图，一半往左一半往右。
  ⭐ 改成：人人对人人的四支（AllGather、ReduceScatter、AllReduce＝Ring、AllToAll）**全部按环一步一步走**：
    每一步人人同时往右发；卡 3 发出的那块**从右边出画面、沿卡片下面的车道绕回来、从左边进卡 0**。
    每一步飞 1.8 秒，落地后停 2 秒，字幕写「第 s 步完成：……」。
  ⭐ 每一步停住的时刻记进 `steps/<Scene>.json`，课件里的播放器读它：**每一步自动暂停，点「下一步」再走**。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic05-anim-coll.py Ring WebPages/media/topic05-ring.mp4
"""
import json
import os

from manim import (Scene, VGroup, VMobject, Rectangle, Text, Arrow, CurvedArrow, FadeIn, FadeOut,
                   Indicate, AnimationGroup, MoveAlongPath, DashedVMobject, Line, WHITE, GREY, BLUE, GREEN,
                   ORANGE, PURPLE_B, YELLOW, UP, DOWN, LEFT, RIGHT, ORIGIN, linear, smooth)

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


XR, XL = XS[-1] + CW / 2 + 0.55, XS[0] - CW / 2 - 0.55   # 绕回时出画面右边、进画面左边的位置
LANE = cy(N - 1) - CH / 2 - 0.55                          # 卡片下面那条绕回车道
YLAB = Y0 + 0.75


def ring_guide(xr=None, xl=None):
    """环的示意：卡名之间三根往右的箭头 ＋ 卡片下面一条虚线车道（卡 3 → 卡 0 绕回来）。"""
    xr, xl = xr or XR, xl or XL
    g = VGroup()
    for k in range(N - 1):
        g.add(Arrow([XS[k] + 0.55, YLAB, 0], [XS[k + 1] - 0.55, YLAB, 0],
                    color=GREY, stroke_width=3, buff=0, max_tip_length_to_length_ratio=0.12))
    lane = VMobject(stroke_color=GREY, stroke_width=2.5)
    lane.set_points_as_corners([[XS[-1] + 0.55, YLAB, 0], [xr, YLAB, 0], [xr, LANE, 0],
                                [xl, LANE, 0], [xl, YLAB, 0], [XS[0] - 0.55, YLAB, 0]])
    g.add(DashedVMobject(lane, num_dashes=70))
    g.add(Arrow([xl, YLAB, 0], [XS[0] - 0.5, YLAB, 0], color=GREY, stroke_width=3, buff=0,
                max_tip_length_to_length_ratio=0.5))
    g.add(Text("卡 3 → 卡 0：绕回来", font_size=20, color=GREY).move_to([0, LANE - 0.28, 0]))
    return g


def flight_path(k, j, d, r, x0=None, x1=None, xr=None, xl=None):
    """卡 k 第 j 块 → 卡 d 第 r 块。⭐ 只许往右走：d 在右边就直飞；
    d 在左边（到头了）就先往右出画面、沿车道绕到最左、再往右进卡 d。
    x0／x1 默认是两张卡的中线；AllToAll 用「寄出／收到」两列时另传。"""
    x0 = XS[k] if x0 is None else x0
    x1 = XS[d] if x1 is None else x1
    xr, xl = xr or XR, xl or XL
    p0, p1 = [x0, cy(j), 0], [x1, cy(r), 0]
    vm = VMobject()
    if d > k or (d == k and x1 >= x0):
        vm.set_points_as_corners([p0, p1])
    else:
        vm.set_points_as_corners([p0, [xr, cy(j), 0], [xr, LANE, 0], [xl, LANE, 0],
                                  [xl, cy(r), 0], p1])
    return vm


def start_state():
    return [[{k} for _ in range(N)] for k in range(N)]


def ring_rs_step(held, s):
    """环形 ReduceScatter 第 s 步：卡 k 把第 (k−s−1) mod 4 块发给右边，收的人加到自己那块上。"""
    sends = [(k, (k - s - 1) % N) for k in range(N)]
    new = [[set(c) for c in r] for r in held]
    for k, j in sends:
        new[(k + 1) % N][j] |= held[k][j]
    return new, sends


def ring_ag_step(held, s, need_full=False):
    """环形 AllGather 第 s 步：卡 k 把它上一步刚拿到的那块（第 (k−s) mod 4 块）发给右边，只替换不相加。"""
    sends = [(k, (k - s) % N) for k in range(N)]
    new = [[set(c) for c in r] for r in held]
    for k, j in sends:
        assert held[k][j], (k, j)
        if need_full:
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
    _h, _ = ring_ag_step(_h, _s, need_full=True)
assert all(_h[k][j] == set(range(N)) for k in range(N) for j in range(N))


FLY, HOLD = 1.8, 2.0            # 每一步飞 1.8 秒、落地停 2 秒（2026-09-25 现场：「别搞那么快，还没看明白呢」）
STEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")


class Stepper(Scene):
    """共用：标题、字幕、按环飞行、每一步的停顿打点。
    ⭐ `hold()` 每调一次就记一个时刻，渲染完写进 steps/<Scene>.json，课件播放器在这些时刻自动暂停。"""
    TITLE = ""

    def setup(self):
        self.marks, self.cap = [], None

    def say(self, txt, color=None):
        new = Text(txt, font_size=26, color=color or GREY_B_).next_to(self.title, DOWN, buff=0.18)
        self.play(*([FadeOut(self.cap)] if self.cap else []), FadeIn(new), run_time=0.4)
        self.cap = new

    def hold(self, t=HOLD):
        self.marks.append(round(self.renderer.time + 0.3, 2))
        self.wait(t)

    def dump(self):
        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, type(self).__name__ + ".json"), "w") as fh:
            json.dump({"pauses": self.marks, "duration": round(self.renderer.time, 2)}, fh)

    def opening(self, held, guide=True):
        self.title = Text(self.TITLE, font_size=30, color=WHITE).to_edge(UP)
        self.add(self.title, labels())
        if guide:
            self.add(ring_guide())
        self.state = grid(held)
        self.add(self.state)
        self.wait(0.8)

    def fly(self, held, flights, after, move=False):
        """flights：[(k, j, d, r)] 卡 k 第 j 块 → 卡 d 第 r 块，同时出发。move=True 时源头腾空。"""
        gone = {(k, j) for k, j, d, r in flights} if move else set()
        base = grid([[set() if (k, j) in gone else held[k][j] for j in range(N)] for k in range(N)])
        movers = [(chunk(held[k][j], j, XS[k]), flight_path(k, j, d, r)) for k, j, d, r in flights]
        self.remove(self.state)
        self.add(base, *[m for m, _ in movers])
        self.play(*[MoveAlongPath(m, pth, rate_func=smooth) for m, pth in movers], run_time=FLY)
        self.state = grid(after, {(d, r) for _, _, d, r in flights})
        self.remove(base, *[m for m, _ in movers])
        self.add(self.state)

    def closing(self, first):
        """⭐ 复位：清干净，摆回第 0 帧（skill 的通用收尾）。"""
        self.play(FadeOut(self.state), FadeOut(self.cap), run_time=0.6)
        self.remove(self.state, self.cap)
        self.cap = None
        self.state = grid(first)
        self.play(FadeIn(self.state), run_time=0.6)
        self.wait(0.6)
        self.dump()

    # ── 三段可复用的环 ────────────────────────────────────────
    def ring_rs(self, held):
        self.say("ReduceScatter：每一步人人同时往右发一块，收到的加到自己那块上")
        self.wait(1.0)
        for s in range(N - 1):
            self.say("第 %d 步（共 3 步）：0→1、1→2、2→3，卡 3 从右边绕回卡 0" % (s + 1))
            nh, sends = ring_rs_step(held, s)
            self.fly(held, [(k, j, (k + 1) % N, j) for k, j in sends], nh)
            held = nh
            self.say("第 %d 步完成：每张卡的粗框那块，又多加进了一个人" % (s + 1), YELLOW)
            self.hold()
        self.say("三步之后：每张卡恰好握着一块完整总和 Σ", GREEN)
        self.hold()
        only = [[set(range(N)) if j == k else set() for j in range(N)] for k in range(N)]
        new = grid(only)
        self.play(FadeOut(self.state), FadeIn(new), run_time=0.8)
        self.state = new
        self.say("其余几块是半路上的中间结果，不要了", GREY)
        self.hold(1.4)
        return only

    def ring_ag(self, held, what="总和"):
        self.say("AllGather：把%s接着往右传，只替换，不相加" % what)
        self.wait(1.0)
        for s in range(N - 1):
            self.say("第 1 步（共 3 步）：人人把自己那块%s传给右边，卡 3 绕回卡 0" % ("总和" if what == "总和" else "")
                     if s == 0 else
                     "第 %d 步（共 3 步）：人人把上一步刚收到的那块传给右边，卡 3 绕回卡 0" % (s + 1))
            nh, sends = ring_ag_step(held, s)
            self.fly(held, [(k, j, (k + 1) % N, j) for k, j in sends], nh)
            held = nh
            self.say("第 %d 步完成：每张卡又多了一块" % (s + 1), YELLOW)
            self.hold()
        return held


class Ring(Stepper):
    """环形 AllReduce ＝ 环形 ReduceScatter ＋ 环形 AllGather。课件 1.3 的 AllReduce 与 1.5 的环用同一支。"""
    TITLE = "AllReduce 全归约：连成一个环，人人只往右边的邻居发"

    def construct(self):
        first = start_state()
        self.opening(first)
        mid = self.ring_rs(first)
        self.ring_ag(mid)
        self.say("人人一份总和 ＝ AllReduce ＝ ReduceScatter ＋ AllGather", GREEN)
        self.play(Indicate(self.state, color=WHITE, scale_factor=1.03), run_time=0.9)
        self.hold()
        self.closing(first)


class AllReduce(Ring):
    pass


class ReduceScatter(Stepper):
    TITLE = "ReduceScatter　归约分散：人人一整份 → 各拿一块总和"

    def construct(self):
        first = start_state()
        self.opening(first)
        self.ring_rs(first)
        self.closing(first)


class AllGather(Stepper):
    TITLE = "AllGather　全收集：一人一块 → 人人一整份"

    def construct(self):
        first = [[{k} if j == k else set() for j in range(N)] for k in range(N)]
        self.opening(first)
        self.ring_ag(first, "自己那块")
        self.say("三步之后：人人一整份，只拼，不加", GREEN)
        self.hold()
        self.closing(first)


class AllToAll(Stepper):
    """AllToAll 的一般情形：派发时人人同时往所有方向「乱射」，专家算完再沿原路飞回（合并）。
    ⛔⛔ 2026-09-26 现场纠正：「转置只能说是 AllToAll 的一种特例……动图你还一步一步的往右发，
      跟环形通讯太像了。你得把 dispatch 和 combine 这种乱射之后又原路回来的感觉表现出来。」
      旧版按「第 s 步寄给右边第 s 个人」排成三步环，还只画等量（转置），两处都讲偏了。
    ⭐ 现在：每张卡上半「出发」8 个 token（颜色 ＝ 出自哪张卡，数字 ＝ 路由定的目的卡），
      下半「收到」。一次同时飞完（直线交叉），数目不等；专家算完描黄边；原路飞回原位。
    ⛔ 数目 C 是示意固定值，跟静态图 fig-a2a 同一组（topic05-fig-coll.py 的 A2A_C）。"""
    TITLE = "AllToAll　全交换：每人给每人寄一份，多少不一样"
    C = [[2, 3, 1, 2], [3, 1, 2, 2], [4, 2, 1, 1], [2, 2, 3, 1]]
    TS, TG = 0.5, 0.1

    def tok(self, k, d, done=False):
        g = VGroup(Rectangle(width=self.TS, height=self.TS, fill_color=COL[k], fill_opacity=0.95,
                             stroke_color=YELLOW if done else COL[k], stroke_width=5 if done else 1),
                   Text(str(d), font_size=22, color=WHITE, weight="BOLD"))
        g[1].move_to(g[0].get_center())
        return g

    def spot(self, card, area, idx):
        r, c = divmod(idx, 4)
        x = XS[card] + (c - 1.5) * (self.TS + self.TG)
        y = (1.15 - r * 0.6) if area == 0 else (-0.85 - r * 0.6)
        return [x, y, 0]

    def construct(self):
        import random as _r
        self.title = Text(self.TITLE, font_size=30, color=WHITE).to_edge(UP)
        frame = VGroup()
        for k in range(N):
            frame.add(Rectangle(width=2.75, height=4.3, stroke_color=GREY, stroke_width=2).move_to([XS[k], -0.5, 0]))
            frame.add(Text("卡 %d" % k, font_size=24, color=COL[k]).move_to([XS[k], 1.95, 0]))
            frame.add(Text("出发", font_size=18, color=GREY).move_to([XS[k] - 1.0, 1.55, 0]))
            frame.add(Text("收到", font_size=18, color=GREY).move_to([XS[k] - 1.0, -0.42, 0]))
        self.add(self.title, frame)
        src = {}                                   # (k, i) → (目的卡, 原位)
        for k in range(N):
            lst = [d for d in range(N) for _ in range(self.C[k][d])]
            _r.Random(10 + k).shuffle(lst)
            for i, d in enumerate(lst):
                src[(k, i)] = (d, self.spot(k, 0, i))
        dst, fill = {}, [0] * N                    # 收件那边按出发的卡排好
        for k in range(N):
            for i in range(8):
                d = src[(k, i)][0]
                dst[(k, i)] = self.spot(d, 1, fill[d])
                fill[d] += 1
        assert fill == [11, 8, 7, 6]
        toks = {key: self.tok(key[0], src[key][0]).move_to(src[key][1]) for key in src}
        self.add(*toks.values())
        self.wait(0.8)

        self.say("每个 token 已由路由定好去哪张卡（格子里的数字），每人要寄给每人的数目都不一样")
        self.hold()
        # ⭐ 轨迹线：派发时画出来、合并时同一批线再亮一次 —— 「原路回来」要靠看见同一条路
        trails = VGroup(*[Line(src[key][1], dst[key], color=COL[key[0]], stroke_width=2.5, stroke_opacity=0.55)
                          for key in src])
        self.say("派发：所有卡同时往所有方向寄，一次飞完，不排队、不接力")
        self.play(FadeIn(trails), run_time=0.5)
        self.play(*[m.animate.move_to(dst[key]) for key, m in toks.items()], run_time=2.2, rate_func=smooth)
        self.play(FadeOut(trails), run_time=0.4)
        self.say("收到的忙闲不均：卡 0 收了 11 个，卡 3 只收 6 个", YELLOW)
        self.hold()
        self.say("各卡上的专家算自己收到的（描黄边）；卡 0 最忙，大家都等它")
        done = {key: self.tok(key[0], src[key][0], True).move_to(dst[key]) for key in src}
        self.play(*[FadeOut(toks[key]) for key in src], *[FadeIn(done[key]) for key in src], run_time=0.9)
        toks = done
        self.hold()
        self.say("合并：又一次 AllToAll，沿原路飞回，回到出发的卡、原来的位置")
        self.play(FadeIn(trails), run_time=0.5)
        self.play(*[m.animate.move_to(src[key][1]) for key, m in toks.items()], run_time=2.2, rate_func=smooth)
        self.play(FadeOut(trails), run_time=0.4)
        self.remove(trails)
        self.say("MoE 每层两次：派发一次、合并一次；每份一样大时，才正好是一次转置", GREEN)
        self.hold()
        self.play(*[FadeOut(m) for m in toks.values()], FadeOut(self.cap), run_time=0.6)
        self.remove(*toks.values(), self.cap)
        self.cap = None
        fresh = [self.tok(key[0], src[key][0]).move_to(src[key][1]) for key in src]
        self.play(*[FadeIn(m) for m in fresh], run_time=0.6)
        self.wait(0.6)
        self.dump()


# ════════════════════════════════════════════════════════════════
# 一个原语一支（2026-09-24 现场：「那么多集合通信，是不是都应该做成会动的？」）
# ⭐ 判断改了：单看每个原语「谁的数据飞去了哪、到了是拼还是加」，这就是时间维度上的增量；
#   总览图负责并排对比，这几支负责把每一个动作演清楚，两者不冲突。
# ⭐ 逻辑视图，不是算法视图：块直接从发的人飞到收的人。真实的环形实现看 Ring 那一支。
# ════════════════════════════════════════════════════════════════
from manim import DashedVMobject

E = set()


def empty_chunk(j, x):
    return DashedVMobject(Rectangle(width=CW, height=CH, stroke_color=GREY, stroke_width=2)
                          .move_to([x, cy(j), 0]), num_dashes=18)


def cell(contrib, j, x, bold=False):
    return chunk(contrib, j, x, bold) if contrib else empty_chunk(j, x)


def grid(held, hot=()):
    return VGroup(*[cell(held[k][j], j, XS[k], (k, j) in hot) for k in range(N) for j in range(N)])


def full_all():
    return [[{k} for _ in range(N)] for k in range(N)]


def only_diag():
    return [[{k} if j == k else set() for j in range(N)] for k in range(N)]


def only_card0(rows):
    return [rows] + [[set() for _ in range(N)] for _ in range(N - 1)]


class Prim(Stepper):
    """一个原语：before → 若干段（字幕, 飞行清单, after）→ 复位到 before。
    飞行清单里每一项 (k, j, d, r, mode)：卡 k 第 j 块 → 卡 d 第 r 块；
    mode：copy 源留着 / move 源拿走 / add 源拿走、到了加起来。"""
    TITLE = ""

    def before(self):
        raise NotImplementedError

    def phases(self):
        raise NotImplementedError

    def construct(self):
        held = self.before()
        self.add(labels())
        state = grid(held)
        self.add(state)
        title = Text(self.TITLE, font_size=30, color=WHITE).to_edge(UP)
        self.add(title)
        self.wait(0.7)
        cap = None
        for txt, flights, after in self.phases():
            new = Text(txt, font_size=26, color=GREY_B_).next_to(title, DOWN, buff=0.18)
            self.play(*( [FadeOut(cap)] if cap else []), FadeIn(new), run_time=0.4)
            cap = new
            movers, stay = [], grid(held)
            gone = {(k, j) for k, j, d, r, m in flights if m in ("move", "add")}
            # 源头被拿走的块先换成空位
            src_view = [[set() if (k, j) in gone else held[k][j] for j in range(N)] for k in range(N)]
            base = grid(src_view)
            for k, j, d, r, m in flights:
                mv = chunk(held[k][j], j, XS[k])
                movers.append((mv, d, r))
            self.remove(state)
            self.add(base, *[m for m, _, _ in movers])
            self.play(*[m.animate.move_to([XS[d], cy(r), 0]) for m, d, r in movers], run_time=FLY)
            hot = {(d, r) for _, d, r in movers}
            state = grid(after, hot)
            self.remove(base, *[m for m, _, _ in movers])
            self.add(state)
            held = after
            self.hold()
        # ⭐ 复位：清干净，摆回第 0 帧
        self.play(FadeOut(state), FadeOut(cap), run_time=0.6)
        self.remove(state, cap)
        state = grid(self.before())
        self.play(FadeIn(state), run_time=0.6)
        self.wait(0.6)
        self.dump()


from manim import GREY_B as GREY_B_


class Broadcast(Prim):
    TITLE = "Broadcast　广播：一份 → 人人一份"

    def before(self):
        return only_card0([{0}] * N)

    def phases(self):
        fl = [(0, j, d, j, "copy") for d in range(1, N) for j in range(N)]
        return [("卡 0 把整份数据复制给每一个人", fl, full_all_of(0))]


def full_all_of(k0):
    return [[{k0} for _ in range(N)] for _ in range(N)]


class Scatter(Prim):
    TITLE = "Scatter　分发：一份拆开 → 一人一块"

    def before(self):
        return only_card0([{0}] * N)

    def phases(self):
        fl = [(0, j, j, j, "move") for j in range(1, N)]
        after = [[{0} if j == k else set() for j in range(N)] for k in range(N)]
        return [("卡 0 把第 j 块发给卡 j，自己只留第 0 块", fl, after)]


class Gather(Prim):
    TITLE = "Gather　收集：一人一块 → 拼成一份"

    def before(self):
        return only_diag()

    def phases(self):
        fl = [(k, k, 0, k, "move") for k in range(1, N)]
        return [("每人把自己那块交给卡 0，卡 0 按顺序拼起来", fl, only_card0([{k} for k in range(N)]))]


class Reduce(Prim):
    TITLE = "Reduce　归约：人人一份 → 加成一份"

    def before(self):
        return full_all()

    def phases(self):
        fl = [(k, j, 0, j, "add") for k in range(1, N) for j in range(N)]
        return [("每人把整份交给卡 0，卡 0 逐块相加", fl, only_card0([set(range(N))] * N))]


