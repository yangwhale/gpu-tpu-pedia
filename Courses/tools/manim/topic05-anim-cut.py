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


# ════════════════════════════════════════════════════════════════
# 第五节：Ring Attention（训练切激活） 与 DCP（推理切 KV）
# ════════════════════════════════════════════════════════════════
NR = 4
COL_R = [BLUE, ORANGE, GREEN, "#9A72AC"]


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
        say("每生成一个 token，它的 KV 存到第 (token 号 mod 4) 张卡上")
        cells = VGroup()
        for t in range(NT):
            c = t % NR
            r = Rectangle(width=1.4, height=0.34, stroke_width=0, fill_color=COL_R[c], fill_opacity=0.85)
            r.move_to([XS4[c], 1.35 - (t // NR) * 0.42, 0])
            lab = Text("token %d" % t, font_size=16, color=WHITE).move_to(r.get_center())
            g = VGroup(r, lab)
            self.play(FadeIn(g), run_time=0.16)
            cells.add(g)
        say("12 个 token，每张卡只存 3 个的 KV：容量是原来的 4 倍", GREEN)
        self.wait(0.8)
        say("算注意力：新 token 的 Q 发给所有卡，各自在自己那份 KV 上算")
        q = Circle(radius=0.22, stroke_width=0, fill_color=YELLOW, fill_opacity=1).move_to([0, -1.3, 0])
        qlab = Text("新 Q", font_size=20, color=YELLOW).next_to(q, DOWN, buff=0.1)
        self.play(FadeIn(q), FadeIn(qlab), run_time=0.3)
        qs = [q.copy() for _ in range(NR)]
        self.play(*[qc.animate.move_to([XS4[c], 0.0, 0]) for c, qc in enumerate(qs)], run_time=0.8)
        say("四份部分结果带着 LSE 合并成一份：多一次合并通信，换回 4 倍的 KV 空间")
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
        say("同样 %d 步：放在一起出 %d 个字，拆开出 %d 个（示意）" % (PD_STEPS, PD_TOGETHER, PD_APART), GREEN)
        self.wait(1.4)
        say("代价：多一趟 KV 传输；我们在 TPU v7x 上实测约 100 毫秒")
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
    def construct(self):
        from manim import Dot, Line, RoundedRectangle, DashedLine
        title = cap_text("同样 8 张卡、TP4 × DP2：TP 组摆在哪，差好几倍", size=30).to_edge(UP)
        self.add(title)
        NX = [-3.3, 3.3]
        base = VGroup()
        for n, x in enumerate(NX):
            base.add(RoundedRectangle(width=4.2, height=3.0, corner_radius=0.2, stroke_color=GREY_B,
                                      stroke_width=2).move_to([x, -0.4, 0]))
            base.add(Text("机器 %d" % n, font_size=22, color=GREY_B).move_to([x, 1.4, 0]))
        link = DashedLine([-1.2, -0.4, 0], [1.2, -0.4, 0], color=GREY_B, stroke_width=3)
        base.add(link, Text("慢线：机器之间", font_size=18, color=GREY_B).move_to([0, -0.05, 0]))
        self.add(base)
        # 每台机器 2 × 2 张卡：卡号 0–3 在机器 0，4–7 在机器 1
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

        TICK = 0.05
        clock = [None]
        kept = []

        def show_clock(v, x, lab, col=WHITE):
            t = Text("%s：这一步用了 %d 格" % (lab, v), font_size=24, color=col).move_to([x, -2.6, 0])
            if clock[0] is not None:
                self.remove(clock[0])
            self.add(t)
            clock[0] = t

        def run(groups, cross_tp, x, lab):
            # groups：两个 TP 组各自的 4 张卡；一轮 TP ＝ 组内沿环传一格
            used = 0
            show_clock(0, x, lab)
            for _ in range(MAP_TP):
                dots, anims = [], []
                for g, col in zip(groups, (BLUE, ORANGE)):
                    for k in range(4):
                        a, b = POS[g[k]], POS[g[(k + 1) % 4]]
                        d = Dot(a, radius=0.11, color=WHITE)
                        dots.append(d)
                        anims.append(d.animate.move_to(b))
                cost = MAP_SLOW if cross_tp else MAP_FAST
                self.add(*dots)
                self.play(*anims, run_time=cost * TICK * 2)
                self.remove(*dots)
                used += cost
                show_clock(used, x, lab)
            # DP：两个组里对应的卡把梯度对一下
            cost = MAP_FAST if cross_tp else MAP_SLOW
            dots = [Dot(POS[groups[0][k]], radius=0.11, color=GREEN) for k in range(4)]
            self.add(*dots)
            self.play(*[d.animate.move_to(POS[groups[1][k]]) for k, d in enumerate(dots)], run_time=cost * TICK * 2)
            self.remove(*dots)
            used += cost
            show_clock(used, x, lab, GREEN if used == MAP_A else RED)
            kept.append(clock[0])
            clock[0] = None
            return used

        def paint(groups):
            self.play(*[cards[i].animate.set_fill(col, opacity=0.9)
                        for g, col in zip(groups, (BLUE, ORANGE)) for i in g], run_time=0.4)

        say("摆法一：一个 TP 组就在一台机器里，DP 才过慢线")
        GA = [[0, 1, 3, 2], [4, 5, 7, 6]]
        paint(GA)
        a = run(GA, False, -3.3, "摆法一")
        self.wait(0.8)
        say("摆法二：TP 组横跨两台机器，每一轮都要过慢线")
        GB = [[0, 1, 5, 4], [2, 3, 7, 6]]
        paint(GB)
        b = run(GB, True, 3.3, "摆法二")
        assert (a, b) == (MAP_A, MAP_B)
        say("%d 格对 %d 格：同样的卡，慢 %.1f 倍（示意）" % (MAP_A, MAP_B, MAP_B / MAP_A), GREEN)
        self.wait(1.6)
        self.play(FadeOut(sub), *[FadeOut(k) for k in kept],
                  *[c.animate.set_fill(GREY_D_, opacity=1) for c in cards], run_time=0.6)
        self.remove(sub, *kept)
        self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))
        self.wait(0.6)


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
