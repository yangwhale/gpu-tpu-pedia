# -*- coding: utf-8 -*-
r"""专题五 · 第三节起「五刀」的动画。

⭐ skill 第 0 步判过：每一支的增量都来自「时间」——
  · TPSplit —— 数据在两张卡上怎么流：列切、各自过激活、行切、最后一次 AllReduce。
    静态图能画出结构，画不出「中间那一段真的没有通信、通信只在最后一下」。
  · Pipeline —— 时间表是一格一格长出来的；先演 4 个 micro-batch 再演 8 个，
    灰色气泡从 3/4 缩到 3/8，这个「缩」只有放在时间里才看得见。

⛔ 数据当场算：流水线每一格的起止时刻由 `schedule()` 按 GPipe 调度现算
  （前向 i 在 stage s 的时刻 ＝ i＋s；反向从最后一段倒着排、按前向 2 倍长），
  气泡比例按 Narayanan 等 arXiv 2104.04473 §2.2.1 的 (p−1)/m 断言。
"""
from manim import (Scene, VGroup, Rectangle, Text, Arrow, FadeIn, FadeOut, Indicate,
                   WHITE, GREY, GREY_B, BLUE, GREEN, ORANGE, YELLOW, UP, DOWN, LEFT, RIGHT)

GREY_D_ = "#3a3a3a"


def cap_text(t, color=WHITE, size=28):
    return Text(t, font_size=size, color=color)


class TPSplit(Scene):
    def construct(self):
        title = cap_text("张量并行切一个 MLP：Y ＝ GeLU(X·W1)·W2", size=30).to_edge(UP)
        self.add(title)
        LANES = [(0.9, BLUE, "卡 0"), (-1.6, ORANGE, "卡 1")]

        def blk(w, h, col, lab, x, y, op=0.9):
            g = VGroup(Rectangle(width=w, height=h, stroke_width=1.5, stroke_color=WHITE,
                                 fill_color=col, fill_opacity=op),
                       Text(lab, font_size=24, color=WHITE, weight="BOLD"))
            g.move_to([x, y, 0])
            g[1].move_to(g[0].get_center())
            return g

        base = VGroup()
        for y, col, name in LANES:
            base.add(Text(name, font_size=26, color=col).move_to([-6.3, y, 0]))
            base.add(blk(1.1, 0.9, GREY, "X", -5.0, y))
            base.add(blk(0.6, 1.3, col, "W1", -3.3, y))
            base.add(blk(0.8, 0.6, col, "W2", 0.9, y))
        self.add(base)
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.5)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
            sub = n

        say("① W1 按列切：每张卡算出中间结果的一半")
        hs = VGroup(*[blk(1.1, 0.9, col, "H 半", -1.4, y) for y, col, _ in LANES])
        movers = VGroup(*[blk(1.1, 0.9, GREY, "X", -5.0, y) for y, _, _ in LANES])
        self.play(movers.animate.move_to([-1.4, (LANES[0][0] + LANES[1][0]) / 2, 0]).set_opacity(0),
                  FadeIn(hs), run_time=1.0)
        self.remove(movers)
        say("② 激活函数逐元素算：各算各的，这一段没有任何通信", GREEN)
        g_lab = VGroup(*[Text("GeLU", font_size=22, color=YELLOW).next_to(h, UP, buff=0.08) for h in hs])
        self.play(FadeIn(g_lab), *[Indicate(h, color=YELLOW, scale_factor=1.06) for h in hs], run_time=0.9)
        say("③ W2 按行切：每张卡只得到 Y 的一个部分和")
        ps = VGroup(*[blk(1.1, 0.9, col, "部分和", 3.0, y) for y, col, _ in LANES])
        self.play(*[h.copy().animate.move_to(p.get_center()).set_opacity(0) for h, p in zip(hs, ps)],
                  FadeIn(ps), run_time=1.0)
        say("④ AllReduce：两份部分和相加，两张卡都拿到完整的 Y", GREEN)
        ys = VGroup()
        for y, _, _ in LANES:
            g = VGroup(Rectangle(width=0.55, height=0.9, stroke_width=0, fill_color=BLUE, fill_opacity=0.9),
                       Rectangle(width=0.55, height=0.9, stroke_width=0, fill_color=ORANGE, fill_opacity=0.9))
            g[1].next_to(g[0], RIGHT, buff=0)
            box = Rectangle(width=1.1, height=0.9, stroke_color=WHITE, stroke_width=4)
            whole = VGroup(g, box, Text("Y", font_size=26, color=WHITE, weight="BOLD"))
            whole.move_to([5.3, y, 0])
            box.move_to(g.get_center())
            whole[2].move_to(g.get_center())
            ys.add(whole)
        cross = [ps[0].copy(), ps[1].copy(), ps[0].copy(), ps[1].copy()]
        targets = [ys[0], ys[0], ys[1], ys[1]]
        self.play(*[c.animate.move_to(t.get_center()).set_opacity(0) for c, t in zip(cross, targets)],
                  FadeIn(ys), run_time=1.2)
        self.remove(*cross)
        say("整个 MLP 只在最后通信一次　——　代价是每一层都有这一次")
        self.wait(1.2)
        keep = [hs, g_lab, ps, ys, sub]
        self.play(*[FadeOut(m) for m in keep], run_time=0.7)
        self.remove(*keep)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)


P = 4


def schedule(m):
    """返回 [(stage, t0, dur, kind, i)]，kind ∈ {F, B}，单位：前向一格。"""
    out = []
    for s in range(P):
        for i in range(m):
            out.append((s, i + s, 1, "F", i))
    t0 = m + P - 1
    for s in range(P):
        for i in range(m):
            out.append((s, t0 + (P - 1 - s) * 2 + i * 2, 2, "B", i))
    return out


def total_len(m):
    return (m + P - 1) + (2 * m + 2 * (P - 1))


for _m in (4, 8):
    busy = sum(d for s, t, d, k, i in schedule(_m) if s == 0)
    idle = total_len(_m) - busy
    # 每个 stage 的空闲 ＝ 3(p−1)（前向 p−1 格 ＋ 反向 2(p−1) 格）；相对理想计算时间 3m ＝ (p−1)/m
    assert idle / busy == (P - 1) / _m, (_m, idle, busy)


class Pipeline(Scene):
    def construct(self):
        title = cap_text("流水线并行：灰色是气泡，每个 stage 都在空等的时间", size=30).to_edge(UP)
        self.add(title)
        labels = VGroup(*[Text("stage %d" % s, font_size=24, color=WHITE).move_to([-6.1, 1.6 - s * 0.8, 0])
                          for s in range(P)])
        self.add(labels)
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.4)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
            sub = n

        def run(m):
            L = total_len(m)
            cw = min(11.0 / L, 0.34)
            x0 = -5.4
            lanes = VGroup(*[Rectangle(width=L * cw, height=0.55, stroke_width=0, fill_color=GREY_D_,
                                       fill_opacity=1).move_to([x0 + L * cw / 2, 1.6 - s * 0.8, 0])
                             for s in range(P)])
            self.play(FadeIn(lanes), run_time=0.4)
            items = sorted(schedule(m), key=lambda e: e[1])
            by_t = {}
            for e in items:
                by_t.setdefault(e[1], []).append(e)
            drawn = VGroup()
            for t in range(L):
                news = []
                for s, t0, d, k, i in by_t.get(t, []):
                    r = Rectangle(width=d * cw - 0.03, height=0.5, stroke_width=0,
                                  fill_color=BLUE if k == "F" else GREEN, fill_opacity=0.95)
                    r.move_to([x0 + (t0 + d / 2) * cw, 1.6 - s * 0.8, 0])
                    news.append(r)
                if news:
                    self.play(*[FadeIn(r) for r in news], run_time=0.11)
                    drawn.add(*news)
                else:
                    self.wait(0.11)
            return lanes, drawn

        say("先用 4 个 micro-batch：气泡 ÷ 理想计算时间 ＝ (4−1) ÷ 4 ＝ 3/4")
        l1, d1 = run(4)
        self.wait(1.0)
        self.play(FadeOut(l1), FadeOut(d1), run_time=0.5)
        self.remove(l1, d1)
        say("换成 8 个 micro-batch：＝ (4−1) ÷ 8 ＝ 3/8，灰色明显缩了", GREEN)
        l2, d2 = run(8)
        self.wait(1.2)
        say("气泡只能摊薄、不能消灭：micro-batch 越多越省，可每张卡要攒的激活也越多")
        self.wait(1.4)
        self.play(FadeOut(l2), FadeOut(d2), FadeOut(sub), run_time=0.6)
        self.remove(l2, d2, sub)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)


# ════════════════════════════════════════════════════════════════
# 第四节：专家并行 —— token 飞去专家那里，负载由数据决定
# ════════════════════════════════════════════════════════════════
NCARD = 4
XS_EP = [-4.8, -1.6, 1.6, 4.8]
COL_EP = [BLUE, ORANGE, GREEN, "#9A72AC"]
# 16 个 token 各挑一个专家（真实 V3 挑 8 个；这里取 1 个方便看）。故意让 E0 成为热门。
ROUTE = [0, 3, 0, 5, 0, 2, 7, 0, 1, 0, 6, 0, 4, 3, 0, 2]
LOAD = [ROUTE.count(e) for e in range(8)]
assert max(LOAD) >= 2.5 * (len(ROUTE) / 8), LOAD          # 热门专家至少是平均负载的 2.5 倍
assert all(0 <= e < 8 for e in ROUTE)


class ExpertParallel(Scene):
    def construct(self):
        from manim import Circle, RED
        title = cap_text("专家并行：token 飞到专家那里，算完再飞回来", size=30).to_edge(UP)
        self.add(title)
        cards = VGroup()
        for c in range(NCARD):
            cards.add(Text("卡 %d" % c, font_size=26, color=COL_EP[c]).move_to([XS_EP[c], 1.95, 0]))
            for k in range(2):
                e = 2 * c + k
                box = Rectangle(width=1.2, height=0.6, stroke_color=WHITE, stroke_width=2).move_to(
                    [XS_EP[c] - 0.7 + 1.4 * k, -2.6, 0])
                cards.add(box, Text("专家 %d" % e, font_size=20, color=WHITE).move_to(box.get_center()))
        self.add(cards)

        def home(t):
            c, i = divmod(t, 4)
            return [XS_EP[c] - 0.9 + 0.6 * i, 1.35, 0]
        toks = [Circle(radius=0.2, stroke_width=0, fill_color=COL_EP[t // 4], fill_opacity=1).move_to(home(t))
                for t in range(16)]
        self.add(*toks)
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.5)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
            sub = n

        say("每个 token 由路由挑一个专家（真实的 V3 每个 token 挑 8 个）")
        self.wait(0.6)
        say("派发（AllToAll）：token 飞到专家所在的卡，在专家门口排队")
        slot = [0] * 8
        targets = []
        for t, e in enumerate(ROUTE):
            c, k = divmod(e, 2)
            x = XS_EP[c] - 0.7 + 1.4 * k
            targets.append([x, -2.0 + 0.42 * slot[e], 0])
            slot[e] += 1
        self.play(*[tk.animate.move_to(tg) for tk, tg in zip(toks, targets)], run_time=1.6)
        hot = max(range(8), key=lambda e: LOAD[e])
        others = [LOAD[e] for e in range(8) if e != hot]
        say("专家 %d 排了 %d 个，别的专家只有 %d 到 %d 个：它算完之前，大家都得等"
            % (hot, LOAD[hot], min(others), max(others)), RED)
        self.play(*[Indicate(toks[t], color=RED, scale_factor=1.25) for t, e in enumerate(ROUTE) if e == hot],
                  run_time=1.0)
        self.wait(0.8)
        say("合并（AllToAll）：算完再送回原来的卡")
        self.play(*[tk.animate.move_to(home(t)) for t, tk in enumerate(toks)], run_time=1.6)
        say("发给谁由数据决定，负载天生不均　——　这是专家并行独有的病", GREEN)
        self.wait(1.4)
        self.play(FadeOut(sub), run_time=0.4)
        self.remove(sub)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)
