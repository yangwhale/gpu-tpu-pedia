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

⛔ 刻意没画（2026-09-25 补）：ExpertParallel 每个 token 只挑 1 个专家（V3 实际挑 8 个），为了看得清；
   RingAttention 演的是不带因果掩码的情形（因果掩码下负载不均，第五节另讲之字形）；
   PDDisagg 与 MeshMap 的格数、快慢 1:9 都是示意，不是实测时序。
"""
from manim import (Scene, VGroup, Rectangle, Text, Arrow, FadeIn, FadeOut, Indicate,
                   WHITE, GREY, GREY_B, BLUE, GREEN, ORANGE, YELLOW, RED, UP, DOWN, LEFT, RIGHT)

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

        # ⭐ 2026-09-25 L3 试讲：跟重画后的 fig-tp-mlp 对齐 —— W1 画成宽矩阵、竖切一刀（只亮自己那半），
        #   W2 画成高矩阵、横切一刀；最后的 Y 是灰色的完整副本（相加，不是拼接），中间标 AllReduce。
        from manim import DashedLine

        def half_mat(w, h, col, x, y, vertical, first, lab):
            outline = Rectangle(width=w, height=h, stroke_width=1.5, stroke_color=GREY_B).move_to([x, y, 0])
            if vertical:
                part = Rectangle(width=w / 2, height=h, stroke_width=0, fill_color=col, fill_opacity=0.9)
                part.move_to([x - w / 4 if first else x + w / 4, y, 0])
                cut = DashedLine([x, y - h / 2 - 0.12, 0], [x, y + h / 2 + 0.12, 0], color=WHITE, stroke_width=2)
            else:
                part = Rectangle(width=w, height=h / 2, stroke_width=0, fill_color=col, fill_opacity=0.9)
                part.move_to([x, y + h / 4 if first else y - h / 4, 0])
                cut = DashedLine([x - w / 2 - 0.12, y, 0], [x + w / 2 + 0.12, y, 0], color=WHITE, stroke_width=2)
            t = Text(lab, font_size=20, color=WHITE, weight="BOLD").move_to(part.get_center())
            return VGroup(outline, part, cut, t)

        base = VGroup()
        for k, (y, col, name) in enumerate(LANES):
            base.add(Text(name, font_size=26, color=col).move_to([-6.3, y, 0]))
            base.add(blk(1.1, 0.9, GREY, "X", -5.0, y))
            base.add(half_mat(1.8, 0.6, col, -3.1, y, True, k == 0, "W1"))
            base.add(half_mat(0.55, 1.4, col, 0.9, y, False, k == 0, "W2"))
        base.add(Text("竖切一刀", font_size=18, color=GREY_B).move_to([-3.1, 1.75, 0]),
                 Text("横切一刀", font_size=18, color=GREY_B).move_to([0.9, 1.95, 0]))
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
        ys = VGroup(*[blk(1.1, 0.9, GREY, "Y", 5.5, y) for y, _, _ in LANES])
        ar = VGroup(Rectangle(width=1.0, height=2.9, stroke_color=GREEN, stroke_width=3),
                    Text("AllReduce", font_size=18, color=GREEN)).move_to([4.25, (LANES[0][0] + LANES[1][0]) / 2, 0])
        ar[1].rotate(1.5708).move_to(ar[0].get_center())
        self.play(FadeIn(ar), run_time=0.3)
        cross = [ps[0].copy(), ps[1].copy(), ps[0].copy(), ps[1].copy()]
        targets = [ys[0], ys[0], ys[1], ys[1]]
        self.play(*[c.animate.move_to(t.get_center()).set_opacity(0) for c, t in zip(cross, targets)],
                  FadeIn(ys), run_time=1.2)
        self.remove(*cross)
        say("整个 MLP 只在最后通信一次　——　代价是每一层都有这一次")
        self.wait(1.2)
        keep = [hs, g_lab, ps, ys, ar, sub]
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

        # ⭐ 2026-09-25 逐图审：原来两次的格宽都按屏宽摊，8 份时灰色的**绝对长度**没变（只是占比变了），
        #   图注却说「缩一半」。改成总 batch 不变：8 份时每份的格子窄一半，灰色才真的短一半。
        CW4 = 0.34

        def run(m):
            L = total_len(m)
            cw = CW4 * 4 / m
            x0 = -5.4
            lanes = VGroup(*[Rectangle(width=L * cw, height=0.55, stroke_width=0, fill_color="#6b6b6b",
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
        say("总量不变、切成 8 份：气泡 ＝ (4−1) ÷ 8 ＝ 3/8，灰色真的短了一半", GREEN)
        l2, d2 = run(8)
        self.wait(1.2)
        say("气泡只能摊薄、不能消灭：份数越多越省，可每份太小，卡就吃不饱")
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


# ⭐ 增量来自时间（2026-09-25 对照构建手册补写）：负载不均是「看着堆起来」的 ——
#   token 一个个飞到专家门口，热门专家那一队越排越长，别人早算完在等它。静态图只能画一张结果分布，
#   画不出「它算完之前大家都得等」。
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
        say("发给谁由数据决定，负载天生不均　——　这是专家并行最重的病", GREEN)
        self.wait(1.4)
        self.play(FadeOut(sub), run_time=0.4)
        self.remove(sub)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)


# ════════════════════════════════════════════════════════════════
# 第五节：Ring Attention（训练切激活） 与 DCP（推理切 KV）
# ════════════════════════════════════════════════════════════════
NR = 4
COL_R = [BLUE, ORANGE, GREEN, "#9A72AC"]


# ⭐ 增量来自时间（2026-09-25 补写）：KV 沿环一步一步传，每张卡那一行注意力一格一格填满；
#   要点「传下一块的时候正在算这一块」是两件事同时发生，只有放在时间里才看得见重叠。

def ring_kv_at(card, step):
    """第 step 步，卡 card 手上是第几段 KV：每步把手上的 KV 传给下一张卡。"""
    return (card - step) % NR


# 四步走完，每张卡把 4 段 KV 各见一次
for _c in range(NR):
    assert sorted(ring_kv_at(_c, s) for s in range(NR)) == list(range(NR))


class RingAttention(Scene):
    def construct(self):
        title = cap_text("Ring Attention：Q 不动，KV 沿环传", size=30).to_edge(UP)
        self.add(title)
        rows = VGroup()
        for c in range(NR):
            y = 1.3 - c * 1.0
            rows.add(Text("卡 %d" % c, font_size=24, color=COL_R[c]).move_to([-6.2, y, 0]))
            q = VGroup(Rectangle(width=0.9, height=0.6, stroke_width=0, fill_color=COL_R[c], fill_opacity=0.9),
                       Text("Q%d" % c, font_size=22, color=WHITE, weight="BOLD"))
            q.move_to([-5.0, y, 0])
            q[1].move_to(q[0].get_center())
            rows.add(q)
        self.add(rows)
        grid = VGroup()
        for i in range(NR):
            for j in range(NR):
                grid.add(Rectangle(width=0.62, height=0.62, stroke_color=GREY, stroke_width=1.5)
                         .move_to([2.6 + j * 0.7, 1.3 - i * 1.0, 0]))
        self.add(grid)
        hdr = VGroup(*[Text("KV%d" % j, font_size=18, color=GREY_B).move_to([2.6 + j * 0.7, 1.95, 0]) for j in range(NR)])
        self.add(hdr, Text("注意力块（行 ＝ 哪张卡的 Q，列 ＝ 哪段 KV）", font_size=20, color=GREY_B).move_to([3.65, -2.55, 0]))

        def kv_blk(j, x, y):
            g = VGroup(Rectangle(width=0.9, height=0.6, stroke_color=WHITE, stroke_width=1.5, fill_color=GREY_D_,
                                 fill_opacity=1), Text("KV%d" % j, font_size=22, color=WHITE))
            g.move_to([x, y, 0])
            g[1].move_to(g[0].get_center())
            return g
        kvs = [kv_blk(c, -3.6, 1.3 - c * 1.0) for c in range(NR)]
        self.add(*kvs)
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.5)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
            sub = n

        filled = VGroup()
        say("每张卡固定一段 Q；每一步算手上这对（Q, KV），同时把 KV 传给下一张")
        for s in range(NR):
            news = []
            for c in range(NR):
                j = ring_kv_at(c, s)
                r = Rectangle(width=0.58, height=0.58, stroke_width=0, fill_color=COL_R[c], fill_opacity=0.9)
                r.move_to([2.6 + j * 0.7, 1.3 - c * 1.0, 0])
                news.append(r)
            self.play(*[FadeIn(r) for r in news], run_time=0.5)
            filled.add(*news)
            if s < NR - 1:
                # KV 往下一张卡传（卡 3 绕回卡 0）
                self.play(*[kvs[j].animate.move_to([-3.6, 1.3 - ((c + 1) % NR) * 1.0, 0])
                            for c in range(NR) for j in [ring_kv_at(c, s)]], run_time=0.7)
        say("转完一圈：每张卡都跟所有 KV 算过了，自己那一行填满", GREEN)
        self.wait(1.0)
        say("关键：传下一块的时候正在算这一块，通信藏在计算后面")
        self.wait(1.4)
        # 复位：KV 回到起点（第 NR−1 步后，卡 c 手上是 KV_(c+1)），清掉填色
        self.play(FadeOut(filled), FadeOut(sub), *[kvs[j].animate.move_to([-3.6, 1.3 - j * 1.0, 0]) for j in range(NR)],
                  run_time=0.8)
        self.remove(filled, sub)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)


# ⭐ 增量来自时间（2026-09-25 补写）：decode 本身就是一个 token 一个 token 往外出，
#   KV 落到哪张卡按 token 号轮转 —— 「轮流存」这个动作只能演出来。开头先演一幕「只开 TP」当对照。
class DecodeCP(Scene):
    def construct(self):
        from manim import Dot, Circle
        title = cap_text("DCP：decode 时 KV 按 token 轮流存到各张卡", size=30).to_edge(UP)
        self.add(title)
        XS4 = [-4.5, -1.5, 1.5, 4.5]
        heads = VGroup(*[Text("卡 %d" % c, font_size=26, color=COL_R[c]).move_to([XS4[c], 1.9, 0]) for c in range(NR)])
        self.add(heads)
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.5)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
            sub = n

        NT = 12
        # ⭐ 2026-09-25 L11：补「之前」的对照帧 —— 只开 TP 时每张卡都存全部 12 个 token 的 KV，
        #   「容量是原来的 4 倍」才有基准。
        say("只开 TP：笔记切不开，每张卡都存全部 12 个 token 的 KV", RED)
        before = VGroup(*[VGroup(Rectangle(width=1.4, height=1.6, stroke_width=0, fill_color=RED, fill_opacity=0.55),
                                 Text("12 个 token", font_size=18, color=WHITE)).move_to([XS4[c], 0.9, 0])
                          for c in range(NR)])
        for g in before:
            g[1].move_to(g[0].get_center())
        self.play(FadeIn(before), run_time=0.5)
        self.wait(1.0)
        self.play(FadeOut(before), run_time=0.4)
        self.remove(before)
        say("DCP：每生成一个 token，它的 KV 存到第 (token 号 mod 4) 张卡上")
        cells = VGroup()
        for t in range(NT):
            c = t % NR
            r = Rectangle(width=1.4, height=0.34, stroke_width=0, fill_color=COL_R[c], fill_opacity=0.85)
            r.move_to([XS4[c], 1.35 - (t // NR) * 0.42, 0])
            lab = Text("token %d" % t, font_size=16, color=WHITE).move_to(r.get_center())
            g = VGroup(r, lab)
            self.play(FadeIn(g), run_time=0.16)
            cells.add(g)
        say("12 个 token，每张卡只存 3 个的 KV：同样的卡，能装 4 倍的笔记", GREEN)
        self.wait(0.8)
        say("算注意力：新 token 的 Q 发给所有卡，各自在自己那份 KV 上算")
        q = Circle(radius=0.22, stroke_width=0, fill_color=YELLOW, fill_opacity=1).move_to([0, -1.3, 0])
        qlab = Text("新 Q", font_size=20, color=YELLOW).next_to(q, DOWN, buff=0.1)
        self.play(FadeIn(q), FadeIn(qlab), run_time=0.3)
        qs = [q.copy() for _ in range(NR)]
        self.play(*[qc.animate.move_to([XS4[c], 0.0, 0]) for c, qc in enumerate(qs)], run_time=0.8)
        say("四份部分结果合并成一份：每层多几次通信，换回 4 倍的 KV 空间")
        self.play(*[qc.animate.move_to([0, -1.3, 0]) for qc in qs], run_time=0.8)
        self.wait(1.2)
        self.play(FadeOut(cells), FadeOut(q), FadeOut(qlab), FadeOut(sub), *[FadeOut(qc) for qc in qs], run_time=0.6)
        self.remove(cells, q, qlab, sub, *qs)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)


# ── PD 分离 ─────────────────────────────────────────────────────────────
# ⭐ 增量来自时间：放在一起时，长 prefill 落下来的那几步里 decode 格子「长不出来」；
#   拆开后 decode 那条线一格一格照常长，prefill 在另一条线上并行跑完、再把 KV 递过去。
#   格数是示意（prefill 占 PD_LEN 步），不是实测时序；字幕里的字数按下面现算。
PD_STEPS, PD_AT, PD_LEN = 16, 5, 5
PD_TOGETHER = PD_STEPS - PD_LEN          # 放一起：prefill 那几步没人出字
PD_APART = PD_STEPS                      # 拆开：decode 每一步都出字
assert (PD_TOGETHER, PD_APART) == (11, 16)


class PDDisagg(Scene):
    def construct(self):
        title = cap_text("PD 分离：prefill 和 decode 拆到两批机器上", size=30).to_edge(UP)
        self.add(title)
        X0, SW = -4.4, 0.6
        Y1, Y2, Y3 = 1.2, -0.7, -2.1
        labs = VGroup(Text("放在一起", font_size=24, color=WHITE).move_to([-5.9, Y1, 0]),
                      Text("prefill 机器", font_size=22, color=ORANGE).move_to([-5.9, Y2, 0]),
                      Text("decode 机器", font_size=22, color=GREEN).move_to([-5.9, Y3, 0]))
        self.add(labs)
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.5)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
            sub = n

        def tick(step, y, col=GREEN):
            return Rectangle(width=SW - 0.08, height=0.5, stroke_width=0, fill_color=col,
                             fill_opacity=0.9).move_to([X0 + step * SW, y, 0])

        def pblock(step, y):
            w = PD_LEN * SW - 0.08
            r = Rectangle(width=w, height=0.5, stroke_width=0, fill_color=ORANGE, fill_opacity=0.95)
            r.move_to([X0 + step * SW + (PD_LEN - 1) * SW / 2, y, 0])
            return VGroup(r, Text("新请求的 prefill", font_size=18, color=WHITE, weight="BOLD").move_to(r.get_center()))

        shown = VGroup()
        say("放在一起：大家一步一步 decode，每格出一个字")
        s = 0
        while s < PD_STEPS:
            if s == PD_AT:
                pb = pblock(s, Y1)
                self.play(FadeIn(pb), run_time=0.9)
                warn = Text("这 %d 步没人出字" % PD_LEN, font_size=20, color=RED).next_to(pb, DOWN, buff=0.12)
                self.play(FadeIn(warn), run_time=0.3)
                shown.add(pb, warn)
                s += PD_LEN
                continue
            t = tick(s, Y1)
            self.play(FadeIn(t), run_time=0.12)
            shown.add(t)
            s += 1
        say("拆开：prefill 在自己的机器上跑，decode 那边一步不停")
        kv = None
        for s in range(PD_STEPS):
            anims = [FadeIn(tick(s, Y3))]
            if s == PD_AT:
                pb2 = pblock(s, Y2)
                anims.append(FadeIn(pb2))
                shown.add(pb2)
            self.play(*anims, run_time=0.14)
            shown.add(anims[0].mobject)
            if s == PD_AT + PD_LEN - 1:
                end = pb2[0].get_right()
                kv = Arrow([end[0], Y2 - 0.3, 0], [end[0] + 0.5, Y3 + 0.3, 0], buff=0, color=BLUE,
                           stroke_width=5, max_tip_length_to_length_ratio=0.3)
                kvl = Text("KV 传过去", font_size=18, color=BLUE).next_to(kv, RIGHT, buff=0.1)
                self.play(FadeIn(kv), FadeIn(kvl), run_time=0.3)
                shown.add(kv, kvl)
        say("放在一起：decode 被截走 %d 步；拆开：一格不断（多用了一批 prefill 机器）" % PD_LEN, GREEN)
        self.wait(1.4)
        say("代价：多一趟 KV 传输；按我们 v7x 那套的带宽估算约 100 毫秒")
        self.wait(1.4)
        self.play(FadeOut(shown), FadeOut(sub), run_time=0.6)
        self.remove(shown, sub)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)


# ── 摆到机器上 ─────────────────────────────────────────────────────────
# ⭐ 增量来自时间：同样 8 张卡、同样 TP4 × DP2，只是 TP 组摆的位置不同。
#   摆法一 TP 的来回都在机器里的快线上，慢线只走一次 DP；摆法二 TP 每一轮都要过慢线，
#   计时器一格一格跳，差距是「看着它慢」出来的。
# ⛔ 示意模型（不是实测）：一步 8 轮 TP 通信、1 次 DP 通信；快线 1 格、慢线 9 格
#   （9 ＝ GB300 NVLink 1.8 TB/s ÷ 每 GPU 网卡 200 GB/s，见 topic05-fig-map.py）。
MAP_TP, MAP_DP, MAP_FAST, MAP_SLOW = 8, 1, 1, 9
MAP_A = MAP_TP * MAP_FAST + MAP_DP * MAP_SLOW
MAP_B = MAP_TP * MAP_SLOW + MAP_DP * MAP_FAST
assert (MAP_A, MAP_B) == (17, 73) and round(MAP_B / MAP_A, 1) == 4.3


class MeshMap(Scene):
    """⭐⭐ 2026-09-25 按「一步一步来、停一下再跳」重做（原话见 topic05-anim-coll.py 头注）：
      原来格数 ∝ 动画时长，摆法一 8 轮 TP 一共只演了 0.8 秒，根本看不见；消息点全是白的，看不出谁传给谁。
      现在：每个 TP 组的环画成同色箭头，过慢线的那几段标红；第 1 轮慢放并停住，
      后 7 轮快进（计数照加）；DP 那一次单独停住。停顿时刻写 steps/MeshMap.json，课件播放器据此自动暂停。"""
    def construct(self):
        import json, os
        from manim import Dot, RoundedRectangle, DashedLine, RED, CurvedArrow, ArcBetweenPoints, MoveAlongPath, PI
        marks = []

        def hold(t=2.0):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)
        title = cap_text("同样 8 张卡、TP4 × DP2：TP 组摆在哪，差好几倍", size=30).to_edge(UP)
        self.add(title)
        NX = [-3.3, 3.3]
        base = VGroup()
        for n, x in enumerate(NX):
            base.add(RoundedRectangle(width=4.2, height=3.0, corner_radius=0.2, stroke_color=GREY_B,
                                      stroke_width=2).move_to([x, -0.4, 0]))
            base.add(Text("机器 %d" % n, font_size=22, color=GREY_B).move_to([x, 1.85, 0]))
        link = DashedLine([-1.2, -0.4, 0], [1.2, -0.4, 0], color=GREY_B, stroke_width=3)
        base.add(link, Text("慢线：机器之间", font_size=18, color=GREY_B).move_to([0, -0.05, 0]))
        self.add(base)
        POS = []
        for n, x in enumerate(NX):
            for r in range(2):
                for c in range(2):
                    POS.append([x - 0.9 + c * 1.8, -0.4 + 0.7 - r * 1.4, 0])
        cards = VGroup(*[Rectangle(width=1.1, height=0.8, stroke_width=0, fill_color=GREY_D_,
                                   fill_opacity=1).move_to(POS[i]) for i in range(8)])
        self.add(cards)
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.5)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
            sub = n

        clock = [None]
        kept = []

        def show_clock(v, x, lab, col=WHITE):
            t = Text("%s：这一步用了 %d 格" % (lab, v), font_size=24, color=col).move_to([x, -2.6, 0])
            if clock[0] is not None:
                self.remove(clock[0])
            self.add(t)
            clock[0] = t

        def machine(i):
            return 0 if i < 4 else 1

        def is_return(grp, k):
            """环的最后一段（第 4 张 → 第 1 张）跨机器时，走机器外面的弧线绕回来，不横穿画面。"""
            a, b = grp[k], grp[(k + 1) % 4]
            return k == 3 and machine(a) != machine(b)

        def arc_ends(grp, gi, k):
            a, b = grp[k], grp[(k + 1) % 4]
            dy = 0.42 if gi == 0 else -0.42
            return [POS[a][0], POS[a][1] + dy, 0], [POS[b][0], POS[b][1] + dy, 0], (PI / 4 if gi == 0 else -PI / 4)

        def ring(groups):
            g = VGroup()
            for gi, (grp, col) in enumerate(zip(groups, (BLUE, ORANGE))):
                for k in range(4):
                    a, b = grp[k], grp[(k + 1) % 4]
                    cross = machine(a) != machine(b)
                    if is_return(grp, k):
                        p0, p1, ang = arc_ends(grp, gi, k)
                        g.add(CurvedArrow(p0, p1, angle=ang, color=RED, stroke_width=5))
                    else:
                        g.add(Arrow(POS[a], POS[b], buff=0.5, color=RED if cross else col,
                                    stroke_width=5 if cross else 3, max_tip_length_to_length_ratio=0.15))
            return g

        def tp_round(groups, cross_tp, run_time):
            dots, anims = [], []
            for gi, (grp, col) in enumerate(zip(groups, (BLUE, ORANGE))):
                for k in range(4):
                    a, b = grp[k], grp[(k + 1) % 4]
                    d = Dot(POS[a], radius=0.12, color=RED if machine(a) != machine(b) else WHITE)
                    dots.append(d)
                    if is_return(grp, k):
                        p0, p1, ang = arc_ends(grp, gi, k)
                        d.move_to(p0)
                        anims.append(MoveAlongPath(d, ArcBetweenPoints(p0, p1, angle=ang)))
                    else:
                        anims.append(d.animate.move_to(POS[b]))
            self.add(*dots)
            self.play(*anims, run_time=run_time)
            self.remove(*dots)
            return MAP_SLOW if cross_tp else MAP_FAST

        def dp(groups, cross_tp, x, lab, used):
            cost = MAP_FAST if cross_tp else MAP_SLOW
            say("最后 1 次 DP：两个组里对应的卡交换梯度，%s：%d 格" % ("这一对在同一台机器里" if cross_tp else "要过慢线", cost))
            dots = [Dot(POS[groups[0][k]], radius=0.12, color=GREEN) for k in range(4)] + \
                   [Dot(POS[groups[1][k]], radius=0.12, color=GREEN) for k in range(4)]
            self.add(*dots)
            self.play(*[d.animate.move_to(POS[groups[1][k]]) for k, d in enumerate(dots[:4])],
                      *[d.animate.move_to(POS[groups[0][k]]) for k, d in enumerate(dots[4:])], run_time=1.4)
            self.remove(*dots)
            used += cost
            show_clock(used, x, lab, GREEN if used == MAP_A else RED)
            hold()
            return used

        def run(groups, cross_tp, x, lab, first_msg):
            arrows = ring(groups)
            self.play(*[cards[i].animate.set_fill(col, opacity=0.9)
                        for g, col in zip(groups, (BLUE, ORANGE)) for i in g], FadeIn(arrows), run_time=0.6)
            show_clock(0, x, lab)
            say(first_msg)
            used = tp_round(groups, cross_tp, 1.6)
            show_clock(used, x, lab)
            say("第 1 轮完成：%d 格" % used, YELLOW)
            hold()
            say("后面 7 轮一模一样，快进")
            for _ in range(MAP_TP - 1):
                used += tp_round(groups, cross_tp, 0.35)
                show_clock(used, x, lab)
            used = dp(groups, cross_tp, x, lab, used)
            kept.append(clock[0])
            clock[0] = None
            self.play(FadeOut(arrows), run_time=0.4)
            return used

        say("摆法一：一个 TP 组就在一台机器里")
        GA = [[0, 1, 3, 2], [4, 5, 7, 6]]
        a = run(GA, False, -3.3, "摆法一", "第 1 轮 TP：每组沿自己的环传一格，全在机器里，快线 1 格")
        say("摆法二：TP 组横跨两台机器（红箭头是过慢线的那几段）")
        GB = [[0, 1, 4, 5], [2, 3, 6, 7]]      # 1→4 贴着缝过去，5→0 从机器上方绕回（下排从下方）
        self.wait(0.6)
        b = run(GB, True, 3.3, "摆法二", "第 1 轮 TP：环上有两段过慢线，这一轮要等最慢的那段：9 格")
        assert (a, b) == (MAP_A, MAP_B)
        say("%d 格对 %d 格：同样的卡，慢 %.1f 倍（示意）" % (MAP_A, MAP_B, MAP_B / MAP_A), GREEN)
        hold(2.2)
        self.play(FadeOut(sub), *[FadeOut(k) for k in kept],
                  *[c.animate.set_fill(GREY_D_, opacity=1) for c in cards], run_time=0.6)
        self.remove(sub, *kept)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "MeshMap.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)


# ── FSDP 一步 ───────────────────────────────────────────────────────────
# ⭐ 增量来自时间：静态图能列出「AG、AG、RS」三个格子，列不出「借来 → 用 → 还掉 → 反向再借」
#   这个节奏。每张卡始终只长期拿着每层 1/4（自己那一段），整层只在用的那一刻出现。
# ⛔ 次数现算：前向每层 1 次 AllGather，反向每层 1 次 AllGather ＋ 1 次 ReduceScatter。
FS_L = 3
FS_COMM = FS_L * 1 + FS_L * 2
assert FS_COMM == 3 * FS_L == 9


class FSDPStep(Scene):
    def construct(self):
        title = cap_text("FSDP：每层用之前借回来，用完就还", size=30).to_edge(UP)
        self.add(title)
        XC = [-4.8, -1.6, 1.6, 4.8]
        YL = [1.1, 0.0, -1.1]
        SEGW = 0.62

        def seg_pos(card, k, i):
            return [XC[card] - 0.93 + k * SEGW, YL[i], 0]

        base = VGroup()
        for c in range(NR):
            base.add(Text("卡 %d" % c, font_size=24, color=COL_R[c]).move_to([XC[c], 2.0, 0]))
            for i in range(FS_L):
                base.add(Rectangle(width=SEGW * 4 + 0.08, height=0.62, stroke_color=GREY_B,
                                   stroke_width=1.2).move_to([XC[c], YL[i], 0]))
        for i in range(FS_L):
            base.add(Text("第 %d 层" % (i + 1), font_size=18, color=GREY_B).move_to([-6.6, YL[i], 0]))
        own = VGroup(*[Rectangle(width=SEGW - 0.06, height=0.5, stroke_width=0, fill_color=COL_R[c],
                                 fill_opacity=0.9).move_to(seg_pos(c, c, i))
                       for c in range(NR) for i in range(FS_L)])
        self.add(base, own)
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.5)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.3)
            sub = n

        count = [0, None]

        def bump():
            count[0] += 1
            t = Text("通信次数：%d" % count[0], font_size=24, color=WHITE).move_to([0, -2.4, 0])
            if count[1] is not None:
                self.remove(count[1])
            self.add(t)
            count[1] = t

        def gather(i):
            cps, anims = [], []
            for c in range(NR):
                for k in range(NR):
                    if k == c:
                        continue
                    r = Rectangle(width=SEGW - 0.06, height=0.5, stroke_width=0, fill_color=COL_R[k],
                                  fill_opacity=0.9).move_to(seg_pos(k, k, i))
                    cps.append(r)
                    anims.append(r.animate.move_to(seg_pos(c, k, i)))
            self.add(*cps)
            self.play(*anims, run_time=0.55)
            bump()
            return cps

        def use(i, col):
            glow = VGroup(*[Rectangle(width=SEGW * 4 + 0.08, height=0.62, stroke_color=col,
                                      stroke_width=4).move_to([XC[c], YL[i], 0]) for c in range(NR)])
            self.play(FadeIn(glow), run_time=0.2)
            self.play(FadeOut(glow), run_time=0.2)

        def drop(cps):
            self.play(*[FadeOut(r) for r in cps], run_time=0.25)
            self.remove(*cps)

        def scatter(i):
            dots, anims = [], []
            for c in range(NR):
                for k in range(NR):
                    if k == c:
                        continue
                    d = Rectangle(width=0.16, height=0.16, stroke_width=0, fill_color=YELLOW,
                                  fill_opacity=1).move_to(seg_pos(c, k, i))
                    dots.append(d)
                    anims.append(d.animate.move_to(seg_pos(k, k, i)))
            self.add(*dots)
            self.play(*anims, run_time=0.55)
            self.play(*[FadeOut(d) for d in dots], run_time=0.2)
            self.remove(*dots)
            bump()

        say("前向：每层先 AllGather 拼回整层，算完只留自己那一段")
        for i in range(FS_L):
            cps = gather(i)
            use(i, BLUE)
            drop(cps)
        say("反向：扔掉的权重要再拼一次；算出的梯度 ReduceScatter 给各自的主人")
        for i in reversed(range(FS_L)):
            cps = gather(i)
            use(i, GREEN)
            drop(cps)
            scatter(i)
        assert count[0] == FS_COMM
        say("每层三次：前向拼一次，反向再拼一次、散一次", GREEN)
        self.wait(1.6)
        self.play(FadeOut(sub), FadeOut(count[1]), run_time=0.5)
        self.remove(sub, count[1])
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)


# ── Ulysses ─────────────────────────────────────────────────────────────
# ⭐ 2026-09-25 R9 麻瓜：「只说来回换，没说换完有什么好处」。增量来自时间：
#   一次 AllToAll 把「每卡一段序列、全部头」换成「每卡全部序列、一个头」，
#   本卡就能把这个头的注意力算完，再换回去。颜色 ＝ 哪一段序列，格子里写的是第几个头。
class Ulysses(Scene):
    def construct(self):
        title = cap_text("Ulysses：用两次 AllToAll，在「按序列切」和「按头切」之间换", size=28).to_edge(UP)
        self.add(title)
        XS4 = [-4.5, -1.5, 1.5, 4.5]
        heads = VGroup(*[Text("卡 %d" % c, font_size=24, color=WHITE).move_to([XS4[c], 1.9, 0]) for c in range(NR)])
        self.add(heads)

        def pos(card, idx):
            return [XS4[card], 1.2 - idx * 0.72, 0]

        cells = {}
        for k in range(NR):              # 序列第 k 段
            for j in range(NR):          # 第 j 个头
                r = Rectangle(width=1.9, height=0.56, stroke_width=0, fill_color=COL_R[k], fill_opacity=0.9)
                t = Text("第 %d 段 · 头 %d" % (k, j), font_size=17, color=WHITE)
                g = VGroup(r, t).move_to(pos(k, j))
                t.move_to(r.get_center())
                cells[(k, j)] = g
        self.add(*cells.values())
        sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
        self.add(sub)
        self.wait(0.5)

        def say(t, color=GREY_B):
            nonlocal sub
            n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
            self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
            sub = n

        say("开始：每张卡拿一段序列，这一段的全部头都在（颜色 ＝ 哪一段）")
        self.wait(1.2)
        say("第一次 AllToAll：第 j 个头的那一格，送到卡 j")
        self.play(*[cells[(k, j)].animate.move_to(pos(j, k)) for k in range(NR) for j in range(NR)], run_time=1.4)
        say("现在每张卡：全部序列、一个头 —— 这个头的注意力在本卡就能算完", GREEN)
        glow = VGroup(*[Rectangle(width=2.1, height=3.0, stroke_color=YELLOW, stroke_width=4).move_to([XS4[c], 0.12, 0])
                        for c in range(NR)])
        self.play(FadeIn(glow), run_time=0.3)
        self.wait(1.0)
        self.play(FadeOut(glow), run_time=0.3)
        say("第二次 AllToAll：算完再换回按序列切，接着往下走")
        self.play(*[cells[(k, j)].animate.move_to(pos(k, j)) for k in range(NR) for j in range(NR)], run_time=1.4)
        say("代价：每层两次 AllToAll；卡数不能超过头数", GREEN)
        self.wait(1.4)
        self.play(FadeOut(sub), run_time=0.4)
        self.remove(sub)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)
