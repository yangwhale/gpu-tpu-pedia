# -*- coding: utf-8 -*-
r"""专题六「推理」的三支动画：连续批处理（第四节）、投机解码一步（第十一节）、分页 KV（第三节）。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · ContBatch —— 静态甘特图（fig6-gantt）画得出结果，画不出「同一时刻两边各走到哪」。
    两边计时器一起跳：上面的槽位一格格变灰、空等最长的那个，下面的槽位一空就有人补进来，
    最后同样 15 步做完的请求数是「看着拉开」的。
  · SpecStep —— 投机解码一步的内容就是一串先后：小模型一个个猜 → 大模型一趟算完 → 从左往右比 →
    第一个错处截断、换成大模型的字。静态图（fig6-spec-step）只能画结果格子，画不出「从左往右比、
    碰到第一个错就停」这个顺序，而「猜对几个就出几个」正是这个顺序的直接结果。
  · PagedKV —— 分页的要点是「按需」：写满一块才去领下一块、请求一结束块立刻还回、新人马上用上。
    静态图（fig6-paged）只有一个时刻的页表快照，「领」和「还」两个动作都在时间里。

⛔ 数据从哪来（当场算，断言钉住）
  · ContBatch：4 个槽位、16 个请求，长度 REQ 是**示意**（前 8 个与 fig6-gantt 的 3/8/2/5/6/2/7/4 相同，
    后 8 个是续上的编排）。两种调度由 static_sched / cont_sched 现模拟；看 15 步（＝静态批第二批整批返回的那一刻，
    两种「做完」的口径在这一刻一致）：静态批做完 8 个、空转 23 格；连续批做完 14 个、空转 0 格。
  · SpecStep：大模型的「贪心续写」用一句公开古诗当示意（TARGET），草稿每步猜的 4 个字写死在 DRAFTS；
    接受个数由 verify() 从左往右比出来，断言第 1 步收下 2 个、出 3 个；第 2 步收下 4 个、出 5 个（含白送 1 个）。
    温度为 0 时「一样才收下」就是真实规则；温度不为 0 时是 min(1, p/q) 的拒绝采样（正文 11.1 有，画面不画）。
  · PagedKV：每块 4 个 token、16 块显存，请求的 prompt 长度与结束时刻是示意（REQS），
    分配器「从空闲池领编号最小的块」由 paged_sim 现模拟。参数是按条件筛出来的（见 REQS 上方注释），
    断言：请求 4 进来时领到的正好是请求 2 还回的那 2 块；每个请求的块都不连续；浪费只在每个请求最后一块里。

⛔ 刻意没画
  · ContBatch：新请求进来那一步的 prefill 有多重（4.2 节 chunked prefill 那张图的事）；请求到达时间（假设一开始全在排队）；
    每一步的时长随批里人数变化（这里每步等长）。
  · SpecStep：拒绝采样的概率细节；树形草稿；草稿模型和大模型的真实耗时比（底下的用时条只表达「猜很便宜、验是搬一趟」）。
  · PagedKV：共享前缀与写时复制（静态图 fig6-paged 已画）；连续预留不画第二套，只在字幕里对照一句。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py ContBatch WebPages/media/topic06-contbatch.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py SpecStep  WebPages/media/topic06-spec.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py PagedKV   WebPages/media/topic06-paged.mp4
   SpecStep 每一步落地的停顿时刻写进 steps/SpecStep.json（课件挂 data-pauses）。
"""
import json
import math
import os

from manim import (Scene, VGroup, Rectangle, Line, Text, ValueTracker, FadeIn, FadeOut, Indicate,
                   WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, ORANGE, PURPLE_B,
                   UP, DOWN, LEFT, RIGHT, linear)

STEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")


def tx(s, size=26, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


# ════════════════════════════════════════════════════════════════
# ① ContBatch：静态批 vs 连续批
# ════════════════════════════════════════════════════════════════
SLOTS = 4
REQ = [3, 8, 2, 5, 6, 2, 7, 4, 3, 5, 2, 6, 4, 3, 2, 5]     # ⚠️ 示意：每个请求要出几个字（步数）
NAMES = "ABCDEFGHIJKLMNOP"
WIN = 15                                                     # 看前 15 步


def static_sched():
    """静态批：4 个一批，最长的做完整批才返回、下一批才进来。返回 grid[t][slot] = (r, k) 或 None，及每个请求的完成时刻。"""
    grid, fin, t = [], {}, 0
    for b in range(0, len(REQ), SLOTS):
        batch = list(range(b, min(b + SLOTS, len(REQ))))
        L = max(REQ[r] for r in batch)
        for k in range(L):
            grid.append([(r, k) if k < REQ[r] else None for r in batch] + [None] * (SLOTS - len(batch)))
        for r in batch:
            fin[r] = t + REQ[r]
        t += L
    return grid, fin


def cont_sched():
    """连续批（Orca 式逐步调度）：每一步开始前，空槽立刻塞进排队的下一个。"""
    q, slots, left, grid, fin, t = list(range(len(REQ))), [None] * SLOTS, {}, [], {}, 0
    while q or any(s is not None for s in slots):
        for i in range(SLOTS):
            if slots[i] is None and q:
                r = q.pop(0)
                slots[i], left[r] = r, REQ[r]
        grid.append([(s, REQ[s] - left[s]) if s is not None else None for s in slots])
        t += 1
        for i in range(SLOTS):
            r = slots[i]
            if r is not None:
                left[r] -= 1
                if left[r] == 0:
                    fin[r], slots[i] = t, None
    return grid, fin


G_ST, FIN_ST = static_sched()
G_CB, FIN_CB = cont_sched()
DONE_ST = sum(1 for r in FIN_ST if FIN_ST[r] <= WIN)
DONE_CB = sum(1 for r in FIN_CB if FIN_CB[r] <= WIN)
IDLE_ST = sum(c is None for row in G_ST[:WIN] for c in row)
IDLE_CB = sum(c is None for row in G_CB[:WIN] for c in row)
assert REQ[:8] == [3, 8, 2, 5, 6, 2, 7, 4]                   # 跟 fig6-gantt 同一组开头
assert len(G_ST) >= WIN and len(G_CB) >= WIN
assert max(REQ[0:4]) + max(REQ[4:8]) == WIN      # 第一批 8 步、第二批 7 步：15 步恰是第二批整批返回的时刻
assert DONE_ST == 8 and DONE_CB == 14, (DONE_ST, DONE_CB)
assert IDLE_ST == 23 and IDLE_CB == 0, (IDLE_ST, IDLE_CB)


def _color_requests():
    """A–D 沿用 fig6-gantt 的配色；之后贪心挑色，保证同一槽位里首尾相接的两个请求不同色（两种调度都查）。
    ⛔ 不能整套照搬 fig6-gantt 的前 8 色：它只排过静态批，连续批里 D 后面紧接 H，两个都是紫（断言当场挡下）。"""
    pal = [BLUE, ORANGE, GREEN, PURPLE_B]
    col = {0: BLUE, 1: ORANGE, 2: GREEN, 3: PURPLE_B}
    nb = {r: set() for r in range(len(REQ))}
    for g in (G_ST, G_CB):
        for t in range(1, len(g)):
            for s in range(SLOTS):
                a, b = g[t - 1][s], g[t][s]
                if a and b and a[0] != b[0]:
                    nb[a[0]].add(b[0])
                    nb[b[0]].add(a[0])
    for r in range(4, len(REQ)):
        used = {col[x] for x in nb[r] if x in col}
        col[r] = next(c for c in pal[r % 4:] + pal[:r % 4] if c not in used)   # 从轮换色起挑，免得全是蓝
    for r in nb:
        for x in nb[r]:
            assert col[r] != col[x], (NAMES[r], NAMES[x])
    return col


RCOL = _color_requests()


class ContBatch(Scene):
    def construct(self):
        CW, CH = 0.64, 0.42
        X0 = -5.0                                   # 第 0 列左边
        YS = [1.5, 1.0, 0.5, 0.0]
        YC = [-1.2, -1.7, -2.2, -2.7]
        clock = ValueTracker(0.0)

        title = tx("同样 4 个槽位、同一串请求：两种排法", 30).to_edge(UP)
        self.add(title)
        heads = VGroup(tx("静态批：凑满一批，等最长的做完", 24, GREY_B).move_to([0, 2.1, 0]),
                       tx("连续批：每一步都重新排，谁做完谁让位", 24, GREY_B).move_to([0, -0.6, 0]))
        heads[0].align_to([X0, 0, 0], LEFT)
        heads[1].align_to([X0, 0, 0], LEFT)
        slots_lab = VGroup()
        for ys in (YS, YC):
            for s, y in enumerate(ys):
                slots_lab.add(tx("槽 %d" % (s + 1), 20, GREY_B).move_to([X0 - 0.55, y, 0]))
        self.add(heads, slots_lab)

        # 格子建一次，updater 只改透明度
        cells = []                                    # (rect, letter or None, col_index, base_opacity)
        for grid, ys in ((G_ST, YS), (G_CB, YC)):
            for c in range(WIN):
                for s in range(SLOTS):
                    cell = grid[c][s]
                    x = X0 + (c + 0.5) * CW
                    if cell is None:
                        r = Rectangle(width=CW - 0.06, height=CH, stroke_width=0, fill_color=GREY)
                        r.move_to([x, ys[s], 0])
                        cells.append((r, None, c, 0.45))
                    else:
                        rq, k = cell
                        r = Rectangle(width=CW - 0.06, height=CH, stroke_width=0, fill_color=RCOL[rq])
                        r.move_to([x, ys[s], 0])
                        lab = tx(NAMES[rq], 22, WHITE, True).move_to(r.get_center()) if k == 0 else None
                        cells.append((r, lab, c, 0.9))
        grid_g = VGroup(*[r for r, _, _, _ in cells])
        lab_g = VGroup(*[l for _, l, _, _ in cells if l is not None])

        def upd_cells(_m):
            t = clock.get_value()
            for r, lab, c, op in cells:
                a = clip01(t - c)
                r.set_fill(opacity=op * a)
                if lab is not None:
                    lab.set_opacity(a)
        grid_g.add_updater(upd_cells)
        self.add(grid_g, lab_g)

        # 光标：当前走到哪一步（两段，只跨格子区）
        def cursor():
            x = X0 + clock.get_value() * CW
            return VGroup(Line([x, YS[0] + 0.3, 0], [x, YS[-1] - 0.3, 0], color=YELLOW, stroke_width=4),
                          Line([x, YC[0] + 0.3, 0], [x, YC[-1] - 0.3, 0], color=YELLOW, stroke_width=4))
        from manim import always_redraw
        cur = always_redraw(cursor)
        self.add(cur)

        # 计数器与时钟：每个取值一份 Text，切透明度
        XR = 5.85
        cnt_st = [tx("做完 %d 个" % k, 30, WHITE, True).move_to([XR, (YS[0] + YS[-1]) / 2, 0]) for k in range(len(REQ) + 1)]
        cnt_cb = [tx("做完 %d 个" % k, 30, GREEN, True).move_to([XR, (YC[0] + YC[-1]) / 2, 0]) for k in range(len(REQ) + 1)]
        ticks = [tx("已走 %d 步" % n, 26, YELLOW).move_to([X0 + WIN * CW / 2, -3.4, 0]) for n in range(WIN + 1)]
        note = tx("请求长度是示意", 20, GREY).move_to([XR, -3.4, 0])
        self.add(note)

        def done(fin, t):
            return sum(1 for r in fin if fin[r] <= t + 1e-9)

        def upd_counts(_m):
            t = clock.get_value()
            a, b, n = done(FIN_ST, t), done(FIN_CB, t), int(math.floor(t + 1e-9))
            for k, m in enumerate(cnt_st):
                m.set_opacity(1 if k == a else 0)
            for k, m in enumerate(cnt_cb):
                m.set_opacity(1 if k == b else 0)
            for k, m in enumerate(ticks):
                m.set_opacity(1 if k == n else 0)
        counts = VGroup(*cnt_st, *cnt_cb, *ticks)
        upd_counts(counts)
        counts.add_updater(upd_counts)
        self.add(counts)

        # 字幕：挂在时钟上，按区间切
        CAPS = [
            (lambda t: t < 1e-6, "16 个请求排队，长短不一；每一列是一步 decode，每一步都要搬一趟权重", GREY_B),
            (lambda t: 1e-6 <= t < 8, "上面：短的早做完了，槽位也得空等最长的那个（灰格＝这一趟白搬）", GREY_B),
            (lambda t: 8 <= t < WIN - 1e-6, "下面：谁做完，下一步就换排队的下一个进来，槽位一格不空", GREY_B),
            (lambda t: t >= WIN - 1e-6, "同样 %d 步：静态批做完 %d 个（空转 %d 格），连续批做完 %d 个（空转 %d 格）" % (WIN, DONE_ST, IDLE_ST, DONE_CB, IDLE_CB), YELLOW),
        ]
        caps = [tx(s, 24, c).next_to(title, DOWN, buff=0.18) for _, s, c in CAPS]

        def upd_caps(_m):
            t = clock.get_value()
            for (f, _, _), m in zip(CAPS, caps):
                m.set_opacity(1 if f(t) else 0)
        cap_g = VGroup(*caps)
        upd_caps(cap_g)                               # ⛔ 先手动算一次第 0 帧
        upd_cells(grid_g)
        cap_g.add_updater(upd_caps)
        self.add(cap_g)

        self.wait(2.0)
        self.play(clock.animate.set_value(WIN), run_time=WIN * 0.8, rate_func=linear)
        self.wait(3.4)
        self.play(clock.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)   # 复位：拨回时钟
        self.wait(0.5)


# ════════════════════════════════════════════════════════════════
# ② SpecStep：投机解码一步、再一步
# ════════════════════════════════════════════════════════════════
TARGET = "床前明月光疑是地上霜举头望明月"        # ⚠️ 示意：当作大模型的贪心续写
PREFIX0 = 4                                       # 一开始已写下「床前明月」
K = 4
DRAFTS = ["光疑似地", "地上霜举"]                 # ⚠️ 示意：小模型两步各猜的 4 个字


def big_model(prefix):
    """大模型在这个前缀后面会写的字（示意）。前缀已经偏离 TARGET 时返回 None：那个位置建立在错字上，作废。"""
    if TARGET.startswith(prefix) and len(prefix) < len(TARGET):
        return TARGET[len(prefix)]
    return None


def verify(prefix, draft):
    """大模型一趟算出 K+1 个位置自己的字；从左往右比，第一个不一样就停。返回 (每个位置大模型的字, 收下几个, 这一步出的字)。"""
    own = [big_model(prefix + draft[:i]) for i in range(len(draft) + 1)]
    acc = 0
    while acc < len(draft) and own[acc] == draft[acc]:
        acc += 1
    out = draft[:acc] + own[acc]                  # 第一个错处（或全对时的第 K+1 个位置）用大模型自己的字
    return own, acc, out


_p = TARGET[:PREFIX0]
SPEC = []
for _d in DRAFTS:
    _own, _acc, _out = verify(_p, _d)
    SPEC.append((len(_p), _d, _own, _acc, _out))
    _p += _out
assert [s[3] for s in SPEC] == [2, 4] and [len(s[4]) for s in SPEC] == [3, 5]
assert _p == TARGET[:PREFIX0 + 8]                 # 两步共出 8 个字，和大模型自己一个个写的完全一样
assert all(len(s[4]) >= 1 for s in SPEC)


class SpecStep(Scene):
    def construct(self):
        marks = []
        PX0, PP = -4.4, 0.8                       # 第 0 个字的中心 x、字距
        YP, YD, YT = 1.75, 0.55, -0.75            # 已写下 ／ 小模型猜 ／ 大模型验 三行
        BOX = 0.66

        def px(i):
            return PX0 + i * PP

        def box(ch, i, y, col, op=0.85):
            g = VGroup(Rectangle(width=BOX, height=BOX, stroke_color=col, stroke_width=3,
                                 fill_color=col, fill_opacity=op),
                       tx(ch, 32, WHITE, True))
            g.move_to([px(i), y, 0])
            g[1].move_to(g[0].get_center())
            return g

        title = tx("投机解码：小模型先猜 4 个，大模型一趟验完", 30).to_edge(UP)
        rowlab = VGroup(tx("已写下", 24, GREY_B).move_to([-6.1, YP, 0]),
                        tx("小模型猜", 24, PURPLE_B).move_to([-6.1, YD, 0]),
                        tx("大模型验", 24, BLUE).move_to([-6.1, YT, 0]))
        prefix0 = VGroup(*[box(TARGET[i], i, YP, GREY, 0.5) for i in range(PREFIX0)])
        YS_ = -2.35                                # 用时条
        strip_lab = tx("用时", 22, GREY_B).move_to([-6.1, YS_, 0])
        legend = tx("紫＝小模型猜一个（便宜）　灰＝大模型搬一趟权重（贵）", 20, GREY_B).move_to([0, -3.05, 0])
        cap0 = tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18)
        self.add(title, rowlab, prefix0, strip_lab, legend, cap0)
        self.cap = cap0
        shown = VGroup()                           # ⭐ 中途加进来的东西都收进这一组，收尾一并清掉
        prefix_extra = VGroup()
        sx = [PX0 - BOX / 2]                       # 用时条走到哪

        def say(s, color=GREY_B):
            n = tx(s, 26, color).next_to(title, DOWN, buff=0.18)
            self.play(FadeOut(self.cap), FadeIn(n), run_time=0.35)
            self.cap = n

        def hold(t=2.0):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        def seg(w, col, op):
            r = Rectangle(width=w - 0.04, height=0.36, stroke_width=0, fill_color=col, fill_opacity=op)
            r.move_to([sx[0] + w / 2, YS_, 0])
            sx[0] += w
            return r

        self.wait(0.6)
        for step, (p0, draft, own, acc, out) in enumerate(SPEC, start=1):
            shown_step = VGroup()
            # ① 小模型一个个猜
            say("第 %d 步：小模型一个接一个猜 4 个字，每个都很快" % step)
            drafts = VGroup()
            for j in range(K):
                b = box(draft[j], p0 + j, YD, PURPLE_B, 0.85).scale(0.92)
                s = seg(0.32, PURPLE_B, 0.9)
                self.play(FadeIn(b, shift=0.2 * DOWN), FadeIn(s), run_time=0.3)
                drafts.add(b)
                shown.add(s)
            # ② 大模型一趟同时算 K+1 个位置
            say("大模型搬一趟权重，同时算出这 5 个位置它自己会写的字", BLUE)
            owns = VGroup(*[box(ch if ch else "?", p0 + j, YT, BLUE if ch else GREY, 0.55)
                            for j, ch in enumerate(own)])
            frame = Rectangle(width=PP * (K + 1) + 0.12, height=BOX + 0.3, stroke_color=BLUE, stroke_width=3)
            frame.move_to([(px(p0) + px(p0 + K)) / 2, YT, 0])
            ftag = tx("一趟", 22, BLUE).next_to(frame, RIGHT, buff=0.15)
            s = seg(1.9, GREY, 0.75)
            self.play(FadeIn(frame), FadeIn(ftag), FadeIn(owns), FadeIn(s), run_time=1.0)
            shown.add(s)
            # ③ 从左往右比
            say("从左往右逐个比：一样就收下")
            for j in range(min(acc + 1, K)):
                ok = j < acc
                col = GREEN if ok else RED
                self.play(drafts[j][0].animate.set_fill(col, 0.9).set_stroke(col),
                          Indicate(owns[j], color=YELLOW, scale_factor=1.08), run_time=0.45)
            if acc < K:
                say("第 %d 个不一样：换成大模型自己的字，后面的全丢掉" % (acc + 1), RED)
                dims = [drafts[j] for j in range(acc + 1, K)] + [owns[j] for j in range(acc + 1, K + 1)]
                qnote = tx("？＝前面那个字已经错了，这个位置算了也作废", 20, GREY_B)
                qnote.move_to([px(p0 + acc + 1) + 0.6, YT - 0.75, 0])
                self.play(*[m.animate.set_opacity(0.18) for m in dims], FadeIn(qnote),
                          owns[acc][0].animate.set_fill(BLUE, 0.95).set_stroke(YELLOW, 5), run_time=0.6)
                shown_step.add(qnote)
            else:
                say("4 个全对；第 5 个位置大模型本来就算了，白送 1 个", GREEN)
                self.play(owns[K][0].animate.set_fill(BLUE, 0.95).set_stroke(YELLOW, 5), run_time=0.5)
            # ④ 收下的飞上去
            movers, landed = [], VGroup()
            for j in range(len(out)):
                src = drafts[j] if j < acc else owns[j]
                col = GREEN if j < acc else BLUE
                dst = box(out[j], p0 + j, YP, col, 0.85)
                movers.append(src.copy())
                landed.add(dst)
            self.play(*[m.animate.move_to(d.get_center()) for m, d in zip(movers, landed)], run_time=0.8)
            self.remove(*movers)
            self.add(landed)
            prefix_extra.add(landed)
            if acc < K:
                say("第 %d 步完成：猜对 %d 个 ＋ 大模型补 1 个 ＝ 出 %d 个字" % (step, acc, len(out)), GREEN)
            else:
                say("第 %d 步完成：猜对 %d 个 ＋ 白送 1 个 ＝ 出 %d 个字" % (step, acc, len(out)), GREEN)
            hold(2.0)
            self.play(FadeOut(drafts), FadeOut(owns), FadeOut(frame), FadeOut(ftag), FadeOut(shown_step), run_time=0.4)
            self.remove(drafts, owns, frame, ftag, shown_step)

        say("猜对几个就出几个，再加大模型自己的 1 个：每步至少出 1 个", YELLOW)
        self.wait(2.2)
        self.play(FadeOut(shown), FadeOut(prefix_extra), FadeOut(self.cap), run_time=0.6)
        self.remove(shown, prefix_extra, self.cap)
        self.add(tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18))    # ＝ 第 0 帧那个空字幕
        self.wait(0.6)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "SpecStep.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)


# ════════════════════════════════════════════════════════════════
# ③ PagedKV：按块领、用完即还
# ════════════════════════════════════════════════════════════════
PAGE, NBLK = 4, 16
# ⚠️ 示意：(名字, 第几步进来, prompt 长度, 最后一次写入在第几步 / None＝一直在跑)。
#   这组数是用筛子搜出来的（条件：请求 4 进来时领到的正好是请求 2 还回的块、中间没人抢；
#   四个请求的块都不连续；显存最多用到 NBLK−2 块；请求 4 后来还会再长出一块），挑步数最少的一组。
REQS = [("1", 1, 6, None), ("2", 1, 3, 5), ("3", 1, 6, None), ("4", 7, 5, None)]
TP = 12                                           # 一共演 12 步
RCOLP = {"1": BLUE, "2": ORANGE, "3": GREEN, "4": PURPLE_B}


def paged_sim():
    """逐步模拟：每步每个在跑的请求写 1 个 token（进来那一步一次写下整段 prompt）；
    写到块边界才从空闲池领编号最小的一块；请求结束的下一步把块全部还回。
    返回 tokens[(blk, slot)] = [(owner, s_on, s_off)]、alloc = [(owner, k, blk, s_on, s_off)]、每步的空闲块数、末态长度。"""
    free, table, length, alive = list(range(NBLK)), {n: [] for n, *_ in REQS}, {}, set()
    toks, alloc, nfree = [], [], [NBLK]
    released = {}
    for s in range(1, TP + 1):
        for n, a, p, e in REQS:
            if e is not None and s == e + 1 and n in alive:
                released[n] = (list(table[n]), s)
                free = sorted(free + table[n])
                alive.discard(n)
        for n, a, p, e in REQS:
            if s == a:
                alive.add(n)
                length[n] = 0
                add = p
            else:
                add = 1 if n in alive else 0
            for _ in range(add):
                L = length[n]
                if L % PAGE == 0:
                    table[n].append(free.pop(0))
                    alloc.append([n, L // PAGE, table[n][-1], s, None])
                toks.append([table[n][L // PAGE], L % PAGE, n, s, None])
                length[n] = L + 1
        nfree.append(len(free))
    for n, (blks, s) in released.items():
        for x in toks:
            if x[2] == n:
                x[4] = s
        for x in alloc:
            if x[0] == n:
                x[4] = s
    return table, toks, alloc, nfree, length, released


TABLE, TOKS, ALLOC, NFREE, LEN_END, RELEASED = paged_sim()
_alive_end = [n for n, a, p, e in REQS if e is None]
WASTE = sum(len(TABLE[n]) * PAGE - LEN_END[n] for n in _alive_end)
assert TABLE["4"][:2] == RELEASED["2"][0] == [2, 5], (TABLE, RELEASED)
assert all(any(b - a != 1 for a, b in zip(TABLE[n], TABLE[n][1:])) for n in TABLE)    # 每个请求的块都不连续
assert max(NBLK - f for f in NFREE) <= NBLK - 2
assert all(0 <= len(TABLE[n]) * PAGE - LEN_END[n] < PAGE for n in _alive_end)        # 浪费只在最后一块
assert WASTE == 8 and NFREE[-1] == 3, (WASTE, NFREE)
E2, A4 = REQS[1][3], REQS[3][1]
assert A4 == E2 + 2                               # 中间空出一步：先看见「还回」，再看见「用上」


class PagedKV(Scene):
    def construct(self):
        clock = ValueTracker(0.0)
        SW, SH, BP, RP = 0.4, 0.46, 2.0, 1.05       # token 格宽高、块间距、行距
        MX0, MY0 = -6.5, 1.4

        def blk_xy(b):
            return MX0 + (b % 4) * BP, MY0 - (b // 4) * RP

        title = tx("分页 KV：写满一块才领下一块，用完立刻还", 30).to_edge(UP)
        self.add(title)

        outlines, blabs = [], VGroup()
        for b in range(NBLK):
            x, y = blk_xy(b)
            o = Rectangle(width=PAGE * SW + 0.12, height=SH + 0.14, stroke_color=GREY, stroke_width=2.5)
            o.move_to([x + PAGE * SW / 2, y, 0])
            outlines.append(o)
            blabs.add(tx("块 %d" % b, 18, GREY_B).move_to([x + PAGE * SW / 2, y + 0.44, 0]))
        out_g = VGroup(*outlines)
        self.add(blabs)

        slot_rect = {}
        for b in range(NBLK):
            x, y = blk_xy(b)
            for i in range(PAGE):
                r = Rectangle(width=SW - 0.06, height=SH, stroke_width=0, fill_color=GREY, fill_opacity=0)
                r.move_to([x + (i + 0.5) * SW, y, 0])
                slot_rect[(b, i)] = r
        tok_by_slot = {}
        for b, i, n, s_on, s_off in TOKS:
            tok_by_slot.setdefault((b, i), []).append((n, s_on, s_off))
        own_by_blk = {}
        for n, k, b, s_on, s_off in ALLOC:
            own_by_blk.setdefault(b, []).append((n, s_on, s_off))

        def life(t, s_on, s_off):
            """进来那一步淡入（s_on−1 → s_on）；被还回那一步淡出（s_off−1 → s_off−0.5）。"""
            a = clip01(t - (s_on - 1))
            if s_off is not None:
                a = min(a, clip01((s_off - 0.5 - t) / 0.5))
            return a

        def upd_mem(_m):
            t = clock.get_value()
            for key, r in slot_rect.items():
                best, col = 0.0, GREY
                for n, s_on, s_off in tok_by_slot.get(key, []):
                    a = life(t, s_on, s_off)
                    if a > best:
                        best, col = a, RCOLP[n]
                r.set_fill(col, opacity=0.9 * best)
            for b, o in enumerate(outlines):
                col, w = GREY, 2.5
                for n, s_on, s_off in own_by_blk.get(b, []):
                    if life(t, s_on, s_off) > 0.5:
                        col, w = RCOLP[n], 4
                o.set_stroke(col, width=w)
        mem = VGroup(out_g, *slot_rect.values())
        upd_mem(mem)
        mem.add_updater(upd_mem)
        self.add(mem)

        # 页表
        TX0 = 2.45
        TY = [1.3, 0.4, -0.5, -1.4]
        thead = tx("页表：第几块 → 住在哪", 22, GREY_B).move_to([4.4, 1.82, 0])
        self.add(thead)
        rowlabs = {}
        entries = []
        for idx, (n, a, p, e) in enumerate(REQS):
            rl = tx("请求 %s" % n, 24, RCOLP[n], True).move_to([TX0, TY[idx], 0])
            rowlabs[n] = (rl, a, e)
        for n, k, b, s_on, s_off in ALLOC:
            idx = [r[0] for r in REQS].index(n)
            g = VGroup(Rectangle(width=0.52, height=0.46, stroke_color=RCOLP[n], stroke_width=3, fill_opacity=0),
                       tx(str(b), 22, WHITE, True))
            g.move_to([TX0 + 1.0 + k * 0.62, TY[idx], 0])
            g[1].move_to(g[0].get_center())
            entries.append((g, s_on, s_off))
        ended = tx("已结束：块已还回", 20, GREY).move_to([TX0 + 2.15, TY[1], 0])
        e_s = RELEASED["2"][1]

        def upd_table(_m):
            t = clock.get_value()
            for g, s_on, s_off in entries:
                a = life(t, s_on, s_off)
                g[0].set_stroke(opacity=a)            # ⛔ 不用 set_opacity：它连 fill 一起设，框会被填成白色
                g[1].set_opacity(a)
            for n, (rl, a, e) in rowlabs.items():
                rl.set_opacity(1.0 if t > a - 1 + 1e-6 else 0.3)
            ended.set_opacity(clip01((t - (e_s - 0.5)) / 0.5))
        tab = VGroup(*[r[0] for r in rowlabs.values()], *[g for g, _, _ in entries], ended)
        upd_table(tab)
        tab.add_updater(upd_table)
        self.add(tab)

        # 空闲块计数 ＋ 步数
        frees = [tx("空闲块：%d" % k, 26, WHITE).move_to([-4.6, -2.75, 0]) for k in range(NBLK + 1)]
        steps = [tx("第 %d 步" % s, 26, YELLOW).move_to([-1.2, -2.75, 0]) for s in range(TP + 1)]

        def upd_cnt(_m):
            t = clock.get_value()
            s = int(math.floor(t + 1e-9))
            # 空闲块数跟着块轮廓变色的时刻切（life > 0.5），跟画面一致
            used = sum(1 for b in range(NBLK)
                       if any(life(t, so, sf) > 0.5 for _, so, sf in own_by_blk.get(b, [])))
            for k, m in enumerate(frees):
                m.set_opacity(1 if k == NBLK - used else 0)
            for k, m in enumerate(steps):
                m.set_opacity(1 if k == s else 0)
        cnt = VGroup(*frees, *steps)
        upd_cnt(cnt)
        cnt.add_updater(upd_cnt)
        self.add(cnt)

        CAPS = [
            (lambda t: t < 1e-6, "KV 显存切成固定大小的块，每块装 4 个 token（示意）"),
            (lambda t: 1e-6 <= t < 3.2, "三个请求进来、边生成边长：写满一块，才去空闲池领一块新的"),
            (lambda t: 3.2 <= t < E2 + 0.2, "领到哪块算哪块：一个请求的块东一块西一块，页表记住先后"),
            (lambda t: E2 + 0.2 <= t < A4 - 0.8, "请求 2 结束：它的 2 块立刻还回空闲池"),
            (lambda t: A4 - 0.8 <= t < TP - 1e-6, "新请求 4 进来，马上用上刚还回来的那 2 块"),
            (lambda t: t >= TP - 1e-6, "浪费只剩每个请求最后一块的空尾巴（此刻共 %d 格）\n"
                                       "对照：按最大长度连续预留，没写到的位置也一直占着" % WASTE),
        ]
        def cap_mob(s, col):
            """多行字幕拆成几个 Text 竖排：Text 自带的换行对中文行距不对，两行会叠在一起。"""
            g = VGroup(*[tx(line, 24, col) for line in s.split("\n")]).arrange(DOWN, buff=0.12)
            return g.next_to(title, DOWN, buff=0.15)
        caps = [cap_mob(s, YELLOW if i == len(CAPS) - 1 else GREY_B) for i, (_, s) in enumerate(CAPS)]
        legend = tx("每种颜色一个请求；灰框＝空闲块", 20, GREY).move_to([4.4, -2.75, 0])
        self.add(legend)

        def upd_caps(_m):
            t = clock.get_value()
            for (f, _), m in zip(CAPS, caps):
                m.set_opacity(1 if f(t) else 0)
        cap_g = VGroup(*caps)
        upd_caps(cap_g)
        cap_g.add_updater(upd_caps)
        self.add(cap_g)

        self.wait(2.0)
        self.play(clock.animate.set_value(TP), run_time=TP * 1.0, rate_func=linear)
        self.wait(4.0)
        self.play(clock.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)
        self.wait(0.5)
