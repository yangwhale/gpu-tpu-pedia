# -*- coding: utf-8 -*-
r"""专题五 · 第五节「切序列」：Ulysses 与 USP 的分步动画。

⭐ 2026-09-26 现场：「尤利西斯一直没搞懂，还有 USP 也没搞懂」→ 先画了静态图 fig-ulysses / fig-usp，
  又追问「高级的动图画了吗」。skill 第 0 步判过：增量来自时间 —— AllToAll 里「谁把哪一格发给谁」、
  USP 里「先机器内换、再机器间传」的先后，静态图只能画成三张快照。
⭐ 讲法跟静态图同一张「段 × 头」表：每一格 ＝ 一段 token 在一个头上的 Q、K、V。
  这里按卡摆：一列是一张卡，格子颜色 ＝ 哪一段，格子里写头号。
⭐ 多步过程照房规：每一步落地停住，停顿时刻写进 steps/<Scene>.json，课件播放器逐步暂停。

⛔ 刻意没画：因果掩码（之字形那张图管）；环里笔记的逐步流水（Ring Attention 那支动画管），
   USP 的环在这里只有 2 张卡，一次互换就是一整圈。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic05-anim-sp.py UlyssesSteps WebPages/media/topic05-ulysses.mp4
"""
import json
import os

from manim import (Scene, VGroup, Rectangle, Text, FadeIn, FadeOut, Indicate, WHITE, GREY, GREY_B,
                   BLUE, GREEN, ORANGE, PURPLE_B, YELLOW, RED, UP, DOWN, smooth)

FLY, HOLD = 1.8, 2.0
STEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")
SEG_COL = [BLUE, ORANGE, GREEN, PURPLE_B]          # 颜色 ＝ 第几段序列
CW, CH = 1.55, 0.5


def cell(s, h, w=CW, fs=18):
    g = VGroup(Rectangle(width=w, height=CH, stroke_width=0, fill_color=SEG_COL[s], fill_opacity=0.92),
               Text("段%d · 头%d" % (s, h), font_size=fs, color=WHITE, weight="BOLD"))
    g[1].move_to(g[0].get_center())
    return g


class SPStepper(Scene):
    TITLE = ""
    XS = []
    CELL_W, CELL_FS = CW, 18

    def setup(self):
        self.marks, self.cap = [], None

    def say(self, txt, color=None):
        new = Text(txt, font_size=24, color=color or GREY_B).next_to(self.title, DOWN, buff=0.18)
        self.play(*([FadeOut(self.cap)] if self.cap else []), FadeIn(new), run_time=0.4)
        self.cap = new

    def hold(self, t=HOLD):
        self.marks.append(round(self.renderer.time + 0.3, 2))
        self.wait(t)

    def dump(self):
        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, type(self).__name__ + ".json"), "w") as fh:
            json.dump({"pauses": self.marks, "duration": round(self.renderer.time, 2)}, fh)

    def pos(self, card, slot):
        return [self.XS[card], 0.95 - slot * 0.62, 0]

    def frames(self):
        g = VGroup()
        for c, x in enumerate(self.XS):
            g.add(Rectangle(width=CW + 0.3, height=2.75, stroke_color=GREY, stroke_width=2).move_to([x, 0.02, 0]))
            g.add(Text("卡 %d" % c, font_size=24, color=WHITE).move_to([x, 1.65, 0]))
        return g

    def place(self, where):
        """where：{(段, 头): (卡, 槽)} → 摆好全部格子，返回 {(段, 头): mobject}。"""
        cells = {}
        for (s, h), (c, k) in where.items():
            m = cell(s, h, self.CELL_W, self.CELL_FS).move_to(self.pos(c, k))
            cells[(s, h)] = m
        return cells

    def move(self, cells, where, keys=None, rt=FLY):
        keys = keys if keys is not None else list(where)
        self.play(*[cells[key].animate.move_to(self.pos(*where[key])) for key in keys], run_time=rt, rate_func=smooth)

    def box(self, keys, cells, color=YELLOW):
        return VGroup(*[Rectangle(width=self.CELL_W + 0.08, height=CH + 0.08, stroke_color=color, stroke_width=5)
                        .move_to(cells[k].get_center()) for k in keys])

    def closing(self, cells, first):
        """⭐ 复位：清干净，摆回第 0 帧。"""
        movers = list(cells.values())
        self.play(*[FadeOut(m) for m in movers], FadeOut(self.cap), run_time=0.6)
        self.remove(*movers, self.cap)
        self.cap = None
        fresh = self.place(first)
        self.play(*[FadeIn(m) for m in fresh.values()], run_time=0.6)
        self.wait(0.6)
        self.dump()


class UlyssesSteps(SPStepper):
    TITLE = "Ulysses：转置一下，每张卡拿一个头的全部段"
    XS = [-4.95, -1.65, 1.65, 4.95]
    CELL_W, CELL_FS = 1.25, 15
    # ⭐ 房规「落点被占就换布局」：每张卡分左右两列 —— 左「按段」、右「按头」。
    #   否则卡 0 发给卡 j 的那格，会落在卡 j 自己还没发走的格子上，叠住看不见（草稿里真叠了）。

    def pos(self, card, slot):
        col, row = divmod(slot, 4)
        return [self.XS[card] + (col - 0.5) * 1.38, 0.95 - row * 0.62, 0]

    def frames(self):
        g = VGroup()
        for c, x in enumerate(self.XS):
            g.add(Rectangle(width=3.0, height=2.75, stroke_color=GREY, stroke_width=2).move_to([x, 0.02, 0]))
            g.add(Text("卡 %d" % c, font_size=24, color=WHITE).move_to([x, 1.65, 0]))
            g.add(Text("按段", font_size=18, color=GREY_B).move_to([x - 0.69, -1.6, 0]))
            g.add(Text("按头", font_size=18, color=GREY_B).move_to([x + 0.69, -1.6, 0]))
        return g

    def construct(self):
        self.title = Text(self.TITLE, font_size=30, color=WHITE).to_edge(UP)
        by_seg = {(s, h): (s, h) for s in range(4) for h in range(4)}       # 卡 s 第 h 槽 ＝ 段 s 头 h
        by_head = {(s, h): (h, 4 + s) for s in range(4) for h in range(4)}  # 卡 h 右列第 s 格 ＝ 段 s 头 h
        self.add(self.title, self.frames())
        cells = self.place(by_seg)
        self.add(*cells.values())
        self.wait(0.8)

        self.say("进来时按段分：卡 k 拿第 k 段的全部头（颜色 ＝ 哪一段）")
        self.hold()
        hl = self.box([(s, 0) for s in range(4)], cells)
        self.say("可头 0 的注意力要的是：头 0 的全部段 —— 现在散在四张卡上", YELLOW)
        self.play(FadeIn(hl), run_time=0.4)
        self.hold()
        self.play(FadeOut(hl), run_time=0.3)
        self.remove(hl)

        self.say("AllToAll，先看卡 0：头 0 留在本卡，头 1、2、3 分别发给卡 1、2、3")
        self.move(cells, by_head, [(0, h) for h in range(4)])
        self.say("卡 0 那一行发完了：留 1 格，发 3 格", YELLOW)
        self.hold()
        self.say("其余三张卡同时照做：第 j 个头那一格，发给卡 j")
        self.move(cells, by_head, [(s, h) for s in range(1, 4) for h in range(4)])
        self.say("换完了：卡 j 拿头 j 的全部四段", YELLOW)
        self.hold()

        glow = VGroup(*[Rectangle(width=self.CELL_W + 0.2, height=2.5, stroke_color=GREEN, stroke_width=5)
                        .move_to([x + 0.69, 0.02, 0]) for x in self.XS])
        self.say("这一列要看的全在本卡：注意力本地算完，一次都不用问别人", GREEN)
        self.play(FadeIn(glow), run_time=0.4)
        self.play(*[Indicate(m, color=YELLOW, scale_factor=1.06) for m in cells.values()], run_time=1.0)
        self.hold()
        self.play(FadeOut(glow), run_time=0.3)
        self.remove(glow)

        self.say("算完再 AllToAll 一次，换回按段分，接着做逐 token 的运算")
        self.move(cells, by_seg)
        self.say("代价：每层前后各一次 AllToAll；卡数不能超过头数", GREEN)
        self.hold()
        self.closing(cells, by_seg)


class USPSteps(SPStepper):
    TITLE = "USP：机器里做 Ulysses，机器之间走环"
    XS = [-5.3, -2.9, 2.9, 5.3]

    def machines(self):
        g = VGroup()
        for m, (a, b) in enumerate(((0, 1), (2, 3))):
            cx = (self.XS[a] + self.XS[b]) / 2.0
            g.add(Rectangle(width=5.0, height=3.55, stroke_color=GREY_B, stroke_width=2).move_to([cx, 0.2, 0]))
            g.add(Text("机器%s" % ("一" if m == 0 else "二"), font_size=22, color=GREY_B).move_to([cx, -1.85, 0]))
        return g

    def construct(self):
        self.title = Text(self.TITLE, font_size=30, color=WHITE).to_edge(UP)
        by_seg = {(s, h): (s, h) for s in range(4) for h in range(4)}
        # 机器内 AllToAll 后：卡 2·(s//2)+(h//2) 拿「两段 × 两个头」，槽 ＝ (s%2)·2 ＋ (h%2)
        by_blk = {(s, h): (2 * (s // 2) + h // 2, (s % 2) * 2 + h % 2) for s in range(4) for h in range(4)}
        self.add(self.title, self.frames(), self.machines())
        cells = self.place(by_seg)
        self.add(*cells.values())
        self.wait(0.8)

        self.say("4 张卡、2 台机器；进来时照样按段分，卡 k 拿第 k 段的全部头")
        self.hold()
        self.say("第一步：只在机器里做 AllToAll（Ulysses 2）：卡 0、卡 1 互换头")
        self.move(cells, by_blk, [(s, h) for s in (0, 1) for h in range(4) if by_blk[(s, h)][0] != s])
        self.say("机器一换完：卡 0 拿段 0、1 的头 0、1，卡 1 拿段 0、1 的头 2、3", YELLOW)
        self.hold()
        self.say("机器二同时照做，拿后两段")
        self.move(cells, by_blk, [(s, h) for s in (2, 3) for h in range(4)])
        self.say("现在每张卡：两段 × 两个头 —— 头分开了，段还缺一半", YELLOW)
        self.hold()

        hl = self.box([(s, h) for s in (2, 3) for h in (0, 1)], cells, RED)
        self.say("卡 0 管头 0、1，可段 2、3 的笔记在机器二的卡 2 上", RED)
        self.play(FadeIn(hl), run_time=0.4)
        self.hold()
        self.play(FadeOut(hl), run_time=0.3)
        self.remove(hl)

        self.say("第二步：机器之间走环（Ring 2）：卡 0↔卡 2、卡 1↔卡 3 互传笔记")
        ghosts, trips = [], []
        for a, b in ((0, 2), (1, 3)):
            for src, dst in ((a, b), (b, a)):
                for key, (c, k) in by_blk.items():
                    if c != src:
                        continue
                    g = cell(*key).set_opacity(0.45).scale(0.8).move_to(self.pos(src, k))
                    ghosts.append(g)
                    trips.append(g.animate.move_to(self.pos(dst, k)))
        self.add(*ghosts)
        self.play(*trips, run_time=FLY, rate_func=smooth)
        self.play(*[FadeOut(g) for g in ghosts],
                  *[Indicate(m, color=YELLOW, scale_factor=1.06) for m in cells.values()], run_time=0.8)
        self.remove(*ghosts)
        self.say("传过去的是抄件（半透明）：各自算完自己两段 × 两头的注意力，抄件就扔", YELLOW)
        self.hold()

        self.say("算完，再在机器里 AllToAll 一次，换回按段分")
        self.move(cells, by_seg)
        self.say("总卡数 ＝ Ulysses 2 × 环 2；头数上限只管机器里那一维", GREEN)
        self.hold()
        self.closing(cells, by_seg)
