# -*- coding: utf-8 -*-
r"""专题六「推理」的七支动画：连续批处理（第三节）、投机解码一步（第十一节）、分页 KV（第四节）、PD 分离（第八节）、KV 当存储（第四节）、一步里装了什么（第三节）、KV 一页页怎么搬上芯片（第五节）。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · ContBatch —— 静态甘特图（fig6-gantt）画得出结果，画不出「同一时刻两边各走到哪」。
    两边计时器一起跳：上面的槽位一格格变灰、空等最长的那个，下面的槽位一空就有人补进来，
    最后同样 15 步做完的请求数是「看着拉开」的。
  · SpecStep —— 投机解码一步的内容就是一串先后：小模型一个个猜 → 大模型一趟算完 → 从左往右比 →
    第一个错处截断、换成大模型的字。静态图（fig6-spec-step）只能画结果格子，画不出「从左往右比、
    碰到第一个错就停」这个顺序，而「猜对几个就出几个」正是这个顺序的直接结果。
  · PagedKV —— 分页的要点是「按需」：写满一块才去领下一块、请求一结束块立刻还回、新人马上用上。
    静态图（fig6-paged）只有一个时刻的页表快照，「领」和「还」两个动作都在时间里。
  · PDSplit —— 卡顿是时间现象：同一时刻上面所有人停笔、下面照常出字。静态图（fig5-pd）画得出结果，画不出「同一把时钟下两边各走到哪」。
  · KVStore —— 前缀复用加分层下放的内容是两串动作：查找「从上往下一层层找、找到就停」，淘汰「满了把最冷的挤下一层、
    下一层也满就接着往下推」。静态图（fig6-disk-cache、fig6-offload）只给命中前后的首字对比和盈亏带宽，
    画不出「找了几层」和「块在层间搬来搬去」，而这两个动作正是「KV 当存储」的意思。
  · StepMix —— 「新人怎么接上」是跨步的事：E 的 prompt 第 1 步切一块、第 2 步装完当步出首字、第 3 步已经是一格 decode。
    静态图（fig6-stepmix）把四行摆在一起，看得出每行装了什么，看不出格子从哪来、人怎么从门口挪到车上。
  · KVLoad —— 「边算边搬」本身就是两件事在同一时刻发生：一个槽在算、另一个槽在装下一块，序列交界也不停。
    静态图（fig6-s5-kvload）只能摆出某一刻的两个槽和一条时间轴，看不出页怎么按页表一页页飞上片、槽怎么轮换。

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
  · KVStore：三层容量、初始摆放、四个请求的前缀是示意（KV_CAP / KV_INIT / KV_REQS），查找与 LRU 淘汰由 kv_sim 现模拟；
    断言四个请求依次是 HBM 命中、主机内存命中、SSD 命中、全不中，最后一个把淘汰一路推到 SSD。首字条里只有
    「SSD 拉回 : 重算 ＝ 0.5 : 13」有出处（DeepSeek 磁盘缓存，同 fig6-disk-cache），HBM／主机内存两条长短是示意。

  · KVLoad：每页 4 个 token、每块 2 页、3 条序列和页号是示意（KVL_*），与 topic06-fig-decode.py 同一组数；
    规则照 RPA 论文（arXiv 2604.15464 §3.3、算法 1）：按页表取页拼成片上一块、双缓冲、只搬有效长度、同组查询头共用一块 K/V。
  · StepMix：预算 10 格、车上 4 人、门口 2 人都是示意（MIX_B / MIX_RUN0 / MIX_Q0），规则照 vLLM V1 文档
    （先排全部 decode，预算有剩再排 prefill，装不下就切），由 mix_sim 现模拟；与 topic06-fig-batch.py 同一组数、同一组断言。

⛔ 刻意没画
  · ContBatch：新请求进来那一步的 prefill 有多重（4.2 节 chunked prefill 那张图的事）；请求到达时间（假设一开始全在排队）；
    每一步的时长随批里人数变化（这里每步等长）。
  · SpecStep：拒绝采样的概率细节；树形草稿；草稿模型和大模型的真实耗时比（底下的用时条只表达「猜很便宜、验是搬一趟」）。
  · PagedKV：共享前缀与写时复制（静态图 fig6-paged 已画）；连续预留不画第二套，只在字幕里对照一句。
  · PDSplit：KV 缓存池命中那一拍（画面已两组十行，再加就挤）；chunked prefill（4.2 节的事，混部也能缓解卡顿）；传 KV 的真实耗时（8.2 节按带宽估）。
  · KVStore：部分命中（只命中前缀的前一段，剩下的接着算）；远端存储那一层（Mooncake／LMCache，3.6 正文）；
    盈亏带宽（搬回来比重算慢就不该存，fig6-offload 的事）；块大小与真实容量。

  · KVLoad：本步新 KV 异步写回页池（正文交代）；两个 TensorCore 怎么分序列（论文没写）；每块耗时随 token 数变化（这里等长）。
  · StepMix：KV 池满了的情况（3.4、3.6）；每步耗时随格数变化；SGLang 默认不混批（正文交代）；PD 分离的 KV 交接（第八节）。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py ContBatch WebPages/media/topic06-contbatch.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py SpecStep  WebPages/media/topic06-spec.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py PagedKV   WebPages/media/topic06-paged.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py PDSplit   WebPages/media/topic06-pdsplit.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py KVStore   WebPages/media/topic06-kvstore.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py StepMix   WebPages/media/topic06-stepmix.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve.py KVLoad    WebPages/media/topic06-kvload.mp4
   SpecStep、StepMix、KVLoad 每一步落地的停顿时刻写进 steps/<Scene>.json（课件挂 data-pauses）。
"""
import json
import math
import os

from manim import (Scene, VGroup, Rectangle, Line, Text, ValueTracker, FadeIn, FadeOut, Indicate,
                   WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, ORANGE, PURPLE_B, TEAL,
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


# ════════════════════════════════════════════════════════════════
# ④ PDSplit：混部时 prefill 把 decode 卡住；拆开以后字距均匀
# ════════════════════════════════════════════════════════════════
# ⚠️ 全是示意（不标毫秒）：时间以「一步 decode」为一格，只表相对长短。
#   混部：一台机器，第 0–3 步三条请求在 decode；长 prompt 的 prefill 占住 [PF0, PF1)；之后新请求加进批。
#   分离：prefill 机算同一段 [PF0, PF1)，算完传 KV（XFER 格，短于 prefill，对应正文「只占一成上下」的量级关系），
#         decode 机上 A/B/C 每一步都出字，新请求在 KV 到达后的下一步加入。
PD_T = 14                                        # 一共演 14 格
PF0, PF1 = 4, 9                                  # 长 prompt 的 prefill 占的时间段
XFER = 0.5                                       # 传 KV 的那一小段
PD_OLD = ["A", "B", "C"]


def pd_sched():
    """逐步排：返回 {模式: {请求: [出字的步号 k，表示 [k, k+1) 这一格]}} 及机器忙的段。
    混部：机器在 [PF0, PF1) 只算 prefill，这几步没人出字；prefill 一次算完整段 KV，并给出新请求的第一个字，之后它跟大家一起一步一个字。
    分离：decode 机每步都出字；新请求在 KV 到达（PF1 + XFER）之后的第一个整步加入。"""
    mix = {n: [k for k in range(PD_T) if not (PF0 <= k < PF1)] for n in PD_OLD}
    mix["new"] = list(range(PF1, PD_T))
    split = {n: list(range(PD_T)) for n in PD_OLD}
    join = int(math.ceil(PF1 + XFER))
    split["new"] = list(range(join, PD_T))
    return {"mix": mix, "split": split}, join


PD, PD_JOIN = pd_sched()
_gaps_mix = [b - a for a, b in zip(PD["mix"]["A"], PD["mix"]["A"][1:])]
_gaps_split = [b - a for a, b in zip(PD["split"]["A"], PD["split"]["A"][1:])]
assert max(_gaps_mix) == PF1 - PF0 + 1 and set(_gaps_split) == {1}      # 混部有一处字距拉长到 6 格；分离处处 1 格
assert XFER < (PF1 - PF0) / 5 + 1e-9                                     # 传 KV 比 prefill 短得多（示意）
assert PD["split"]["new"][0] == PF1 + 1 and PD["mix"]["new"][0] == PF1


class PDSplit(Scene):
    def construct(self):
        clock = ValueTracker(0.0)
        U, CH = 0.66, 0.3                          # 一格宽、格高
        X0 = -4.4                                  # 第 0 格左边
        XL = -5.65                                 # 行标签中心
        # 上半（混部）与下半（分离）的行 y
        YM = {"bus": 1.66, "new": 1.27, "A": 0.9, "B": 0.54, "C": 0.18}
        YS = {"pf": -0.9, "new": -1.32, "A": -1.7, "B": -2.06, "C": -2.42}

        def xc(k):
            return X0 + (k + 0.5) * U

        title = tx("PD 分离：prefill 挪走，decode 的字不再被卡住", 30).to_edge(UP)
        self.add(title)
        heads = VGroup(tx("混部：prefill 和 decode 挤在一台机器上", 22, GREY_B).move_to([0, 2.1, 0]),
                       tx("PD 分离：prefill 机只算 prefill，decode 机只出字", 22, GREY_B).move_to([0, -0.47, 0]))
        for h in heads:
            h.align_to([-6.6, 0, 0], LEFT)
        labs = VGroup(
            tx("整机在算", 20, GREY_B).move_to([XL, YM["bus"], 0]),
            tx("新请求", 20, GREEN).move_to([XL, YM["new"], 0]),
            *[tx("请求 " + n, 20, GREEN).move_to([XL, YM[n], 0]) for n in PD_OLD],
            tx("prefill 机", 20, ORANGE).move_to([XL, YS["pf"], 0]),
            tx("decode 机", 18, GREY_B).rotate(math.pi / 2).move_to([XL - 0.95, (YS["new"] + YS["C"]) / 2, 0]),
            tx("新请求", 20, GREEN).move_to([XL, YS["new"], 0]),
            *[tx("请求 " + n, 20, GREEN).move_to([XL, YS[n], 0]) for n in PD_OLD],
        )
        self.add(heads, labs)

        # 所有随时间出现的格子：(mobject, 出现时刻 t0)。不透明度 = clip01(t − t0)，在 [t0, t0+1) 里淡入
        items = []

        def cell(k, y, col, h=CH):
            r = Rectangle(width=U - 0.1, height=h, stroke_width=0, fill_color=col, fill_opacity=0)
            r.move_to([xc(k), y, 0])
            items.append((r, k, 0.9))

        for mode, Y in (("mix", YM), ("split", YS)):
            for n, ks in PD[mode].items():
                for k in ks:
                    cell(k, Y[n], GREEN)               # 全课约定：decode 绿、prefill 橙、KV 紫（新请求靠行区分，不另配色）
        for k in range(PD_T):                      # 上半「整机在算」：decode 的步是细绿条
            if not (PF0 <= k < PF1):
                cell(k, YM["bus"], GREEN, 0.14)
        for Y, key in ((YM, "bus"), (YS, "pf")):   # prefill 块：逐格淡入、格与格无缝，长成一整块
            for k in range(PF0, PF1):
                r = Rectangle(width=U + 0.005, height=0.36, stroke_width=0, fill_color=ORANGE, fill_opacity=0)
                r.move_to([xc(k), Y[key], 0])
                items.append((r, k, 0.85))
        cells = VGroup(*[m for m, _, _ in items])

        # 一次性出现的标注：(mobject, 淡入起点, 是否 stroke 类)
        pf_lab_m = tx("长 prompt 的 prefill", 18, WHITE, True).move_to([(xc(PF0) + xc(PF1 - 1)) / 2, YM["bus"], 0])
        pf_lab_s = tx("长 prompt 的 prefill", 18, WHITE, True).move_to([(xc(PF0) + xc(PF1 - 1)) / 2, YS["pf"], 0])
        gap = Rectangle(width=(PF1 - PF0) * U, height=YM["A"] - YM["C"] + 0.5,
                        stroke_color=RED, stroke_width=4, fill_color=RED, fill_opacity=0)
        gap.move_to([X0 + (PF0 + PF1) / 2 * U, (YM["A"] + YM["C"]) / 2, 0])
        late = VGroup(*[tx(s, 22, RED, True) for s in ("下一个字", "被推迟")]).arrange(DOWN, buff=0.1)
        late.move_to([5.85, (YM["A"] + YM["C"]) / 2, 0])
        even = VGroup(*[tx(s, 22, GREEN, True) for s in ("字距", "始终均匀")]).arrange(DOWN, buff=0.1)
        even.move_to([5.85, (YS["A"] + YS["C"]) / 2, 0])
        from manim import Arrow
        x_from, x_to = X0 + PF1 * U, X0 + PD_JOIN * U
        arrow = Arrow([x_from, YS["pf"] - 0.18, 0], [x_to + 0.05, YS["new"] + 0.2, 0], buff=0,
                      color=PURPLE_B, stroke_width=5, max_tip_length_to_length_ratio=0.35)
        kv_lab = tx("传 KV", 22, PURPLE_B, True).next_to(arrow, RIGHT, buff=0.12).shift(0.12 * UP)
        notes = [(pf_lab_m, PF0 + 0.6), (pf_lab_s, PF0 + 0.6), (late, PF0 + 0.8), (arrow, PF1), (kv_lab, PF1),
                 (even, PF1 + 1.2)]
        GAP_T = PF0 + 0.8

        def upd(_m):
            t = clock.get_value()
            for m, t0, op in items:
                m.set_fill(opacity=op * clip01(t - t0))
            for m, t0 in notes:
                m.set_opacity(clip01((t - t0) / 0.5))
            a = clip01((t - GAP_T) / 0.5)
            gap.set_stroke(opacity=a)              # ⛔ 不用 set_opacity：它连 fill 一起设成实心红
            gap.set_fill(opacity=0.14 * a)
        layer = VGroup(cells, gap, *[m for m, _ in notes])
        upd(layer)
        layer.add_updater(upd)
        self.add(layer)

        from manim import always_redraw

        def cursor():
            x = X0 + clock.get_value() * U
            return VGroup(Line([x, YM["bus"] + 0.25, 0], [x, YM["C"] - 0.25, 0], color=YELLOW, stroke_width=3),
                          Line([x, YS["pf"] + 0.25, 0], [x, YS["C"] - 0.25, 0], color=YELLOW, stroke_width=3))
        cur = always_redraw(cursor)
        self.add(cur)

        note = tx("一格＝一步 decode 的时间，只表相对长短（示意）", 20, GREY).move_to([0, -3.25, 0])
        self.add(note)

        CAPS = [
            (lambda t: t < PF0, "三条请求在 decode：每一步，每条请求出一个字，格子等间距", GREY_B),
            (lambda t: PF0 <= t < PF1, "来了一个长 prompt：prefill 一次算完整段的 KV。混部时整台机器被它占住", GREY_B),
            (lambda t: PF1 <= t < PD_JOIN + 1, "分离时它在 prefill 机上算，算完把 KV 传给 decode 机（传的是 KV，不是权重）", GREY_B),
            (lambda t: PD_JOIN + 1 <= t < PD_T - 1e-6, "新请求带着 prefill 给的第一个字进 decode 批，之后一步一个字", GREY_B),
            (lambda t: t >= PD_T - 1e-6, "拆开后 decode 的字距始终均匀；代价是多一批机器、每个请求多传一趟 KV", YELLOW),
        ]
        caps = [tx(s, 24, c).next_to(title, DOWN, buff=0.18) for _, s, c in CAPS]

        def upd_caps(_m):
            t = clock.get_value()
            for (f, _, _), m in zip(CAPS, caps):
                m.set_opacity(1 if f(t) else 0)
        cap_g = VGroup(*caps)
        upd_caps(cap_g)
        cap_g.add_updater(upd_caps)
        self.add(cap_g)

        self.wait(2.0)
        self.play(clock.animate.set_value(PD_T), run_time=PD_T * 1.1, rate_func=linear)
        self.wait(4.0)
        self.play(clock.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)    # 复位：拨回时钟
        self.wait(0.5)


# ════════════════════════════════════════════════════════════════
# ⑤ KVStore：KV 当存储 —— 先在 HBM 找，再往下找，都没有才重算；满了冷的往下放
# ════════════════════════════════════════════════════════════════
# ⚠️ 示意：三层容量、初始摆放、四个请求的前缀都是编排出来的，只为四个请求依次落在「HBM 命中 → 主机内存 →
#   SSD → 哪层都没有」四种情形，且最后一个请求能把淘汰一路推到 SSD。每层内部从左到右＝最近用过 → 最久没用（LRU）。
#   首字条只有「SSD 拉回」对「重算」的比例有出处：DeepSeek 磁盘缓存 128K prompt 首字 13 s → 0.5 s（同 fig6-disk-cache），
#   HBM／主机内存两条的长短是示意。
KV_TIERS = ["HBM", "主机内存", "SSD"]
KV_CAP = [3, 5, 7]
KV_INIT = [list("ABC"), list("DEFG"), list("HIJK")]
KV_REQS = ["A", "D", "H", "L"]
KV_MISS_S, KV_HIT_S = 13.0, 0.5                  # 同 topic06-fig-kv.py 的 T_MISS / T_HIT


def kv_sim():
    """逐个请求：找前缀在哪层（从上往下）→ 拿出来放到 HBM 最前面（没找到就是重算出一块新的）→
    每层超出容量就把最冷的那块挪到下一层最前面，一路往下。
    返回 [(请求, 命中层 or None, [插入后的状态, 每次挪完的状态...], [(挪的块, 从哪层, 到哪层)...])]。"""
    tiers = [list(t) for t in KV_INIT]
    out = []
    for p in KV_REQS:
        hit = next((i for i, t in enumerate(tiers) if p in t), None)
        if hit is not None:
            tiers[hit].remove(p)
        tiers[0].insert(0, p)
        states, moves = [[list(t) for t in tiers]], []
        for i in range(len(tiers)):
            if len(tiers[i]) > KV_CAP[i]:
                assert i + 1 < len(tiers), "SSD 也满了：示意里不该发生"
                x = tiers[i].pop()
                tiers[i + 1].insert(0, x)
                moves.append((x, i, i + 1))
                states.append([list(t) for t in tiers])
        out.append((p, hit, states, moves))
    return out


KV_RUN = kv_sim()
assert [r[1] for r in KV_RUN] == [0, 1, 2, None]                       # 四种情形各一个
assert [len(r[3]) for r in KV_RUN] == [0, 1, 1, 2]                     # 命中 HBM 不挪；最后一个一路挪到 SSD
assert KV_RUN[-1][3][-1][1:] == (1, 2)
assert all(len(t) <= c for t, c in zip(KV_RUN[-1][2][-1], KV_CAP))
assert KV_MISS_S / KV_HIT_S == 26


class KVStore(Scene):
    def construct(self):
        SX0, PITCH, BW, BH = -2.6, 1.0, 0.8, 0.56      # 第 0 格左边、格距、块宽高
        TY = [1.7, 0.45, -0.8]                          # 三层的 y
        XL = -4.0                                       # 层名中心
        XR = -6.0                                       # 请求卡中心
        BY = [-2.0, -2.42, -2.84, -3.26]                # 首字条四行
        SEC = 0.45                                      # 首字条：1 秒 ＝ 0.45 个单位（只对 SSD 与重算两条有意义）
        BAR_W = [0.1, 0.17, KV_HIT_S * SEC, KV_MISS_S * SEC]   # ⚠️ 前两条是示意
        BAR_C = [GREEN, PURPLE_B, PURPLE_B, ORANGE]      # 搬回的是 KV → 紫；重算是 prefill → 橙
        BAR_T = ["HBM 里就有", "从主机内存拉回", "从 SSD 拉回", "重算 prefill"]

        def slot_xy(tier, k):
            return [SX0 + (k + 0.5) * PITCH, TY[tier], 0]

        title = tx("KV 当存储：先找，找到就搬回，都没有才重算", 30).to_edge(UP)
        self.add(title)
        tier_g = VGroup()
        outlines = []
        SUBS = ["小、最快", "大一些、慢一些", "最大、最慢"]
        for i, (nm, cap) in enumerate(zip(KV_TIERS, KV_CAP)):
            o = Rectangle(width=cap * PITCH + 0.1, height=BH + 0.3, stroke_color=GREY, stroke_width=3)
            o.move_to([SX0 + cap * PITCH / 2, TY[i], 0])
            outlines.append(o)
            tier_g.add(o, tx(nm, 24, WHITE, True).move_to([XL, TY[i] + 0.15, 0]),
                       tx(SUBS[i], 18, GREY_B).move_to([XL, TY[i] - 0.22, 0]))
        lru = tx("每层从左到右：最近用过 → 最久没用", 18, GREY).move_to([4.75, TY[0], 0])
        strip_head = tx("首字要等多久", 20, GREY_B).move_to([XL, BY[0] + 0.42, 0])
        foot = tx("块数、层的大小都是示意；首字条只有 SSD 与重算两条按 DeepSeek 实测比例画", 18, GREY).move_to([0, -3.72, 0])
        self.add(tier_g, lru, strip_head, foot)

        def make_block(name, col=PURPLE_B):            # 全课约定：KV 紫
            g = VGroup(Rectangle(width=BW, height=BH, stroke_color=col, stroke_width=3, fill_color=col, fill_opacity=0.55),
                       tx(name, 26, WHITE, True))
            g[1].move_to(g[0].get_center())
            return g

        def place_init():
            d = {}
            for i, t in enumerate(KV_INIT):
                for k, n in enumerate(t):
                    d[n] = make_block(n).move_to(slot_xy(i, k))
            return d

        blocks = place_init()
        self.add(*blocks.values())
        INTRO = "每块是一段算好的前缀 KV；新请求先看自己的前缀存在哪一层"
        cap0 = tx(INTRO, 24, GREY_B).next_to(title, DOWN, buff=0.18)
        self.add(cap0)
        self.cap = cap0
        shown = VGroup()                                 # ⭐ 中途加进来的东西都收进这一组，收尾一并清掉

        def say(s, color=GREY_B):
            n = tx(s, 24, color).next_to(title, DOWN, buff=0.18)
            self.play(FadeOut(self.cap), FadeIn(n), run_time=0.35)
            self.cap = n

        def layout(state):
            return [blocks[n].animate.move_to(slot_xy(i, k)) for i, t in enumerate(state) for k, n in enumerate(t)]

        self.wait(1.8)
        card = None
        for r, (p, hit, states, moves) in enumerate(KV_RUN):
            new_card = VGroup(tx("请求 %d" % (r + 1), 24, YELLOW, True), tx("前缀 %s" % p, 24, WHITE)).arrange(DOWN, buff=0.12)
            new_card.move_to([XR, TY[0], 0])
            frame_c = Rectangle(width=1.5, height=1.05, stroke_color=YELLOW, stroke_width=3).move_to(new_card.get_center())
            new_card.add(frame_c)
            self.play(*([FadeOut(card)] if card else []), FadeIn(new_card, shift=0.2 * RIGHT), run_time=0.3)
            if card:
                self.remove(card)
            card = new_card
            # ① 从上往下逐层找
            marks = VGroup()
            last = hit if hit is not None else len(KV_TIERS) - 1
            for i in range(last + 1):
                ok = (i == hit)
                probe = outlines[i].copy().set_stroke(YELLOW, width=5)
                m = tx("命中" if ok else "没有", 22, GREEN if ok else RED, True).next_to(outlines[i], RIGHT, buff=0.2)
                anims = [FadeIn(probe), FadeIn(m)]
                if ok:
                    anims.append(blocks[p][0].animate.set_stroke(YELLOW, width=6))
                self.play(*anims, run_time=0.35)
                self.play(FadeOut(probe), run_time=0.15)
                self.remove(probe)
                marks.add(m)
            # ② 命中哪层决定首字等多久
            if hit == 0:
                say("前缀 %s 就在 HBM：直接用，接着往后算" % p, GREEN)
            elif hit == 1:
                say("前缀 %s 在主机内存：拉回 HBM，比重算 prefill 快" % p, PURPLE_B)
            elif hit == 2:
                say("前缀 %s 在 SSD：拉回来也比重算快（DeepSeek：首字 13 s → 0.5 s）" % p, PURPLE_B)
            else:
                say("前缀 %s 哪层都没有：只能重算 prefill" % p, ORANGE)
            bar = Rectangle(width=BAR_W[r], height=0.28, stroke_width=0, fill_color=BAR_C[r], fill_opacity=0.9)
            bar.move_to([SX0 + BAR_W[r] / 2, BY[r], 0])
            blab = VGroup(tx("请求 %d" % (r + 1), 20, GREY_B).move_to([XL, BY[r], 0]),
                          tx(BAR_T[r], 20, BAR_C[r]).next_to(bar, RIGHT, buff=0.15))
            if hit is None:
                # 重算条从左往右慢慢长出来：慢，是这一条的全部意思
                blocks[p] = make_block(p, ORANGE).move_to([XR, TY[0] - 1.25, 0])
                bar.stretch_to_fit_width(0.02).align_to([SX0, 0, 0], LEFT)
                self.add(bar)
                self.play(FadeIn(blab[0]), run_time=0.25)
                self.play(bar.animate.stretch_to_fit_width(BAR_W[r]).align_to([SX0, 0, 0], LEFT),
                          FadeIn(blocks[p]), run_time=1.8, rate_func=linear)
                self.play(FadeIn(blab[1]), run_time=0.3)
            else:
                self.play(FadeIn(bar), FadeIn(blab), run_time=0.45)
            shown.add(bar, blab)
            # ③ 放进 HBM 最前面（其余往右挪一格；超出容量的那块露在框外）
            self.play(*layout(states[0]), FadeOut(marks), run_time=0.6)     # 「命中／没有」先收掉：露在框外的那块会落在它的位置上
            self.remove(marks)
            blocks[p][0].set_stroke(ORANGE if hit is None else PURPLE_B, width=3)
            # ④ 满了：最冷的一块往下放，一层一层推
            for j, (x, a, b) in enumerate(moves):
                if j == 0 and len(moves) == 1:
                    say("HBM 满了：最久没用的 %s 往下放到%s" % (x, KV_TIERS[b]))
                elif j == 0:                             # 一路推到底的那次，一句字幕说完两跳
                    say("HBM 满了，%s 往下放；主机内存也满了，%s 再推到 SSD" % (x, moves[1][0]))
                self.play(Indicate(blocks[x], color=RED, scale_factor=1.12), run_time=0.35)
                self.play(*layout(states[j + 1]), run_time=0.6)
            self.wait(0.8)

        say("先找、再搬、最后才算；显存满了，冷的 KV 一层层往下放", YELLOW)
        self.wait(1.9)
        # 收尾复位：清干净，再摆一个跟第 0 帧一模一样的静态件
        everything = VGroup(shown, card, self.cap, *blocks.values())
        self.play(FadeOut(everything), run_time=0.6)
        self.remove(everything, shown, card, self.cap, *blocks.values())
        fresh = place_init()
        self.play(*[FadeIn(b) for b in fresh.values()], FadeIn(tx(INTRO, 24, GREY_B).next_to(title, DOWN, buff=0.18)),
                  run_time=0.5)                  # ＝ 第 0 帧：初始摆放 ＋ 第一句字幕
        self.wait(0.6)


# ════════════════════════════════════════════════════════════════
# ⑥ StepMix：一步里装了什么（第三节 3.3）
# ════════════════════════════════════════════════════════════════
MIX_B = 10                                                   # ⚠️ 示意：每步 token 预算（真实 8K–16K）
MIX_RUN0 = [("A", 40, 9), ("B", 12, 9), ("C", 25, 9), ("D", 7, 1)]   # 名字, 已有 KV, 还要出几个字
MIX_Q0 = [("E", 13, 6), ("F", 4, 5)]                         # 名字, prompt 长度, 一共要出几个字
MIX_T = 4
MIXCOL = {"A": BLUE, "B": GREEN, "C": ORANGE, "D": GREY_B, "E": PURPLE_B, "F": TEAL}


def mix_sim():
    """与 topic06-fig-batch.py 的 mix_sim 同一套规则（vLLM V1：先排全部 decode，预算有剩再切 prefill），
    这里多记每步开始时车上／门口的状态，给画面用。"""
    run = [list(r) for r in MIX_RUN0]
    q = [[n, p, 0, o] for n, p, o in MIX_Q0]
    steps = []
    for _ in range(MIX_T):
        st = dict(run=[(r[0], r[2]) for r in run], queue=[(e[0], e[1]) for e in q])
        cells, out, ctx, budget = [], {}, {}, MIX_B
        for r in run:
            cells.append((r[0], "d", 1))
            budget -= 1
            r[1] += 1
            r[2] -= 1
            ctx[r[0]], out[r[0]] = r[1], "tok"
        joined = []
        for e in q:
            if budget == 0:
                break
            n = min(budget, e[1])
            budget -= n
            e[1] -= n
            e[2] += n
            cells.append((e[0], "p", n))
            ctx[e[0]] = e[2]
            out[e[0]] = "first" if e[1] == 0 else "none"
            if e[1] == 0:
                joined.append(e)
        left = [r[0] for r in run if r[2] == 0]
        run = [r for r in run if r[2] > 0]
        for e in joined:
            q.remove(e)
            run.append([e[0], e[2], e[3] - 1])
        st.update(cells=cells, out=out, ctx=ctx, free=budget, left=left,
                  joined=[e[0] for e in joined], wait=[e[0] for e in q],
                  run_after=[(r[0], r[2]) for r in run], queue_after=[(e[0], e[1]) for e in q])
        steps.append(st)
    return steps


MIX = mix_sim()
assert MIX[0]["cells"][-1] == ("E", "p", 6) and MIX[0]["out"]["E"] == "none" and MIX[0]["left"] == ["D"]
assert MIX[1]["cells"][-1] == ("E", "p", 7) and MIX[1]["out"]["E"] == "first" and MIX[1]["wait"] == ["F"]
assert MIX[2]["cells"][-1] == ("F", "p", 4) and MIX[2]["out"]["F"] == "first" and MIX[2]["free"] == 2
assert all(k == "d" for _, k, _ in MIX[3]["cells"]) and MIX[3]["free"] == 5
assert MIX[0]["ctx"] == {"A": 41, "B": 13, "C": 26, "D": 8, "E": 6} and MIX[1]["ctx"]["E"] == 13
MIX_SAY = [
    ("剩 6 格切给 E：13 个 prompt 先算 6 个", "第 1 步完成：D 出完最后一字下车；E 这步不出字"),
    ("剩 7 格：E 剩下的 7 个正好装完；F 这步排不上", "第 2 步完成：E 最后一块当步出首字，下一步直接 decode"),
    ("剩 6 格：F 的 4 个一块装完，空着 2 格", "第 3 步完成：F 也当步出首字；没人在门口等着上车"),
    ("没人排队：空着 5 格。预算是上限，不必塞满", "第 4 步完成：5 人各出 1 字，车上始终没断"),
]


class StepMix(Scene):
    def construct(self):
        marks = []
        CW, CX0, YS = 0.82, -3.6, -0.25              # 预算格宽、第 0 格左边、这一步那条的 y
        YR, YQ, YW, YA, YO = 1.85, 0.85, -1.05, -1.72, -2.6   # 车上 ／ 排队 ／ 乘权重 ／ 注意力 ／ 出字
        LX = -5.3

        def cx(k):
            return CX0 + (k + 0.5) * CW

        def chip(name, s):
            col = MIXCOL[name]
            g = VGroup(Rectangle(width=1.62, height=0.6, stroke_color=col, stroke_width=3,
                                 fill_color=col, fill_opacity=0.25),
                       tx("%s  %s" % (name, s), 19, WHITE))
            g[1].move_to(g[0].get_center())
            return g

        def chips(run, queue):
            d = {}
            for i, (n, left) in enumerate(run):
                d[n] = chip(n, "还差 %d 字" % left).move_to([CX0 + 0.81 + i * 1.78, YR, 0])
            for i, (n, p) in enumerate(queue):
                d[n] = chip(n, "prompt %d" % p).move_to([CX0 + 0.81 + i * 1.78, YQ, 0])
            return d

        def cell(name, kind, k):
            col = MIXCOL[name]
            g = VGroup(Rectangle(width=CW - 0.08, height=0.62, stroke_color=col, stroke_width=3,
                                 fill_color=col, fill_opacity=0.9 if kind == "d" else 0.3),
                       tx(name, 24, WHITE, True))
            g[1].move_to(g[0].get_center())
            return g.move_to([cx(k), YS, 0])

        title = tx("一步里装了什么：新人怎么接上", 30).to_edge(UP)
        labs = VGroup(*[tx(s, 22, GREY_B).move_to([LX, y, 0]) for s, y in
                        (("在车上", YR), ("排队", YQ), ("这一步", YS), ("乘权重", YW), ("注意力", YA), ("出字", YO))])
        slots = VGroup(*[Rectangle(width=CW - 0.08, height=0.62, stroke_color=GREY, stroke_width=1.5)
                         .set_stroke(opacity=0.6).move_to([cx(k), YS, 0]) for k in range(MIX_B)])
        legend = VGroup(tx("实心＝decode：每人 1 格、出 1 字　　浅色＝prompt 块：最后一块那步出首字", 18, GREY_B).move_to([0, -3.3, 0]),
                        tx("每步预算 10 格、每格 1 个 token（真实部署是 8K 到 16K）；人数和长度都是示意", 18, GREY).move_to([0, -3.68, 0]))
        cap0 = tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18)
        st0 = chips(MIX[0]["run"], MIX[0]["queue"])
        self.add(title, labs, slots, legend, cap0, *st0.values())
        self.cap = cap0
        cur = dict(st0)

        def say(s, color=GREY_B, after=0.5):
            n = tx(s, 26, color).next_to(title, DOWN, buff=0.18)
            self.play(FadeOut(self.cap), FadeIn(n), run_time=0.35)
            self.cap = n
            if after:
                self.wait(after)

        def hold(t=2.0):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        self.wait(0.6)
        for s, st in enumerate(MIX):
            step_g = VGroup()
            # ① decode 先占位：车上每人飞下来 1 格
            say("第 %d 步：车上 %d 人，各占 1 格 decode" % (s + 1, len(st["run"])))
            k, groups, flies = 0, [], []
            for name, kind, n in st["cells"]:
                g0 = k
                for _ in range(n):
                    c = cell(name, kind, k)
                    tgt = c.get_center()
                    c.move_to(cur[name].get_center())
                    flies.append((kind, c, tgt))
                    k += 1
                groups.append((name, kind, g0, k))
            used = k
            dec = [(c, t) for kd, c, t in flies if kd == "d"]
            self.play(*[c.animate.move_to(t) for c, t in dec], run_time=0.7)
            step_g.add(*[c for c, _ in dec])
            # ② 剩下的切给 prompt
            say(MIX_SAY[s][0], PURPLE_B if st["free"] < MIX_B - len(st["run"]) else GREY_B)
            pre = [(c, t) for kd, c, t in flies if kd == "p"]
            if pre:
                anims = [c.animate.move_to(t) for c, t in pre]
                for n, p in st["queue"]:                 # 门口那张卡上的剩余数跟着减
                    took = sum(m for nm, kd, m in st["cells"] if nm == n)
                    if took and p - took > 0:
                        nc = chip(n, "prompt %d" % (p - took)).move_to(cur[n].get_center())
                        anims += [FadeOut(cur[n]), FadeIn(nc)]
                        cur[n] = nc
                self.play(*anims, run_time=0.8)
                step_g.add(*[c for c, _ in pre])
            # ③ 整条一起乘权重；注意力各读各的
            say("整条 %d 格拼成一个矩阵，一起乘权重" % used, BLUE, after=0.2)
            bar = VGroup(Rectangle(width=used * CW - 0.08, height=0.4, stroke_width=0, fill_color=BLUE, fill_opacity=0.5),
                         tx("矩阵乘：不管谁是谁", 20, WHITE))
            bar.move_to([CX0 + used * CW / 2, YW, 0])
            bar[1].move_to(bar[0].get_center())
            self.play(FadeIn(bar), run_time=0.5)
            say("只有注意力分开算：各读各的 KV", after=0.2)
            attn = VGroup()
            for name, kind, a, b in groups:
                col = MIXCOL[name]
                ln = Line([CX0 + a * CW + 0.08, YA + 0.15, 0], [CX0 + b * CW - 0.08, YA + 0.15, 0], color=col, stroke_width=5)
                attn.add(ln, tx("读 %d" % st["ctx"][name], 18, col).move_to([CX0 + (a + b) * CW / 2, YA - 0.14, 0]))
            self.play(FadeIn(attn), run_time=0.5)
            outs = VGroup()
            for name, kind, a, b in groups:
                o = st["out"][name]
                t, col = {"tok": ("出字", GREEN), "first": ("首字", YELLOW), "none": ("不出字", GREY)}[o]
                outs.add(tx(t, 20 if o != "none" else 18, col, o == "first").move_to([CX0 + (a + b) * CW / 2, YO, 0]))
            self.play(FadeIn(outs), run_time=0.5)
            step_g.add(bar, attn, outs)
            # ④ 下车、上车、剩余数更新
            new = chips(st["run_after"], st["queue_after"])
            anims = []
            for n, c in cur.items():
                if n in st["joined"]:
                    anims.append(c.animate.move_to(new[n].get_center()))
                else:
                    anims.append(FadeOut(c))
            self.play(*anims, run_time=0.6)
            self.remove(*cur.values())
            self.play(*[FadeIn(c) for c in new.values()], run_time=0.3)
            cur = new
            say(MIX_SAY[s][1], GREEN, after=0)
            hold(2.0)
            self.play(FadeOut(step_g), run_time=0.4)
            self.remove(step_g)

        say("一步 ＝ 一趟前向：decode 先占位，剩下的格子切给 prompt", YELLOW, after=0)
        self.wait(2.2)
        # 收尾复位：清干净，再摆一套跟第 0 帧一模一样的
        self.play(FadeOut(VGroup(*cur.values())), FadeOut(self.cap), run_time=0.5)
        self.remove(*cur.values(), self.cap)
        fresh = chips(MIX[0]["run"], MIX[0]["queue"])
        self.play(*[FadeIn(c) for c in fresh.values()], run_time=0.5)
        self.add(tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18))    # ＝ 第 0 帧那个空字幕
        self.wait(0.6)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "StepMix.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)


# ════════════════════════════════════════════════════════════════
# ⑦ KVLoad：KV 一页页怎么搬上芯片（第五节）
# ════════════════════════════════════════════════════════════════
# ⚠️ 示意：与 topic06-fig-decode.py 的 fig6-s5-kvload 同一组数（每页 4 个 token、每块 2 页、3 条序列）
KVL_PAGE, KVL_BLK, KVL_POOL, KVL_G = 4, 2, 16, 4
KVL_SEQS = [("A", 15, [6, 1, 11, 4]), ("B", 10, [9, 14, 2]), ("C", 3, [7])]
KVL_COL = {"A": BLUE, "B": ORANGE, "C": GREEN}


def kvl_blocks():
    out = []
    for name, n, pages in KVL_SEQS:
        for b in range(0, len(pages), KVL_BLK):
            pg = pages[b:b + KVL_BLK]
            out.append((name, b // KVL_BLK + 1, pg, min(n, (b + len(pg)) * KVL_PAGE) - b * KVL_PAGE))
    return out


KVL_B = kvl_blocks()
assert [(b[0], b[1], b[3]) for b in KVL_B] == [("A", 1, 8), ("A", 2, 7), ("B", 1, 8), ("B", 2, 2), ("C", 1, 3)]
KVL_SAY = ["先按 A 的页表取两页，拼成一块搬上片",
           "算 A 块 1；同时 A 块 2 已经在路上",
           "算 A 块 2；A 还没算完，B 块 1 已经在路上",
           "算 B 块 1；B 块 2 只有 2 个 token，就只搬 2 个",
           "算 B 块 2；同时取 C：只有 3 个 token",
           "算 C 块 1；没有下一块了"]
assert len(KVL_SAY) == len(KVL_B) + 1


class KVLoad(Scene):
    def construct(self):
        marks = []
        PW, PH_ = 0.95, 0.42
        where = {p: (name, i + 1, min(KVL_PAGE, n - i * KVL_PAGE)) for name, n, ps in KVL_SEQS for i, p in enumerate(ps)}

        def ppos(k):
            return [-6.3 + (k % 4) * 1.05, 1.5 - (k // 4) * 0.52, 0]

        def page(name, i, ntok):
            col = KVL_COL[name]
            box = Rectangle(width=PW, height=PH_, stroke_color=col, stroke_width=2.5)
            fill = Rectangle(width=PW * ntok / KVL_PAGE, height=PH_, stroke_width=0, fill_color=col, fill_opacity=0.35)
            fill.align_to(box, LEFT)
            return VGroup(box, fill, tx("%s 第%d页" % (name, i), 17, WHITE))

        title = tx("KV 一页页怎么搬上芯片", 30).to_edge(UP)
        heads = VGroup(tx("显存：KV 页池", 22, GREY_B).move_to([-4.73, 2.12, 0]),
                       tx("片上：两个槽", 22, PURPLE_B).move_to([-0.4, 2.12, 0]),
                       tx("一块喂一组查询头", 22, GREEN).move_to([3.9, 2.12, 0]))
        pool = VGroup()
        for k in range(KVL_POOL):
            if k in where:
                pool.add(page(*where[k]).move_to(ppos(k)))
            else:
                pool.add(VGroup(Rectangle(width=PW, height=PH_, stroke_color=GREY, stroke_width=1.2).set_stroke(opacity=0.5),
                                tx("空", 16, GREY)).move_to(ppos(k)))
        tables = VGroup(*[tx("%s 的页表：%s" % (nm, " → ".join("格%d" % (p + 1) for p in ps)), 18, KVL_COL[nm])
                          for nm, n, ps in KVL_SEQS])
        for j, t in enumerate(tables):
            t.move_to([-6.8, -0.62 - j * 0.38, 0], aligned_edge=LEFT)
        SY = [1.05, -0.3]
        slots = VGroup(*[Rectangle(width=3.3, height=0.95, stroke_color=GREY_B, stroke_width=2).move_to([-0.4, y, 0]) for y in SY])
        slab = VGroup(*[tx("槽 %d" % (i + 1), 20, GREY_B).move_to([-1.75, y, 0]) for i, y in enumerate(SY)])

        def chip_pos(si, q):
            return [-0.85 + q * 1.05, SY[si], 0]

        def status(si, s, col):
            return tx(s, 18, col).move_to([-0.4, SY[si] + 0.62, 0])

        st = [status(0, "空着", GREY), status(1, "空着", GREY)]
        rows = VGroup(*[VGroup(Rectangle(width=1.5, height=0.4, stroke_color=WHITE, stroke_width=1.5),
                               tx("查询头 %d" % (h + 1), 17, WHITE)).move_to([2.5, 1.5 - h * 0.5, 0]) for h in range(KVL_G)])
        kvbox = Rectangle(width=2.2, height=1.9, stroke_color=GREY, stroke_width=2).move_to([4.9, 0.75, 0])
        kvlab0 = tx("等第一块", 18, GREY).move_to(kvbox.get_center())
        rnote = tx("4 个头摞成 4 行，共用这一块 K、V", 18, GREY_B).move_to([3.9, -0.55, 0])
        TX0, TW = -4.5, 1.85
        tl_lab = VGroup(tx("搬", 20, GREY_B).move_to([-6.0, -2.0, 0]), tx("算", 20, GREY_B).move_to([-6.0, -2.68, 0]))
        tl_slot = VGroup(*[Rectangle(width=TW - 0.1, height=0.5, stroke_color=GREY, stroke_width=1.2).set_stroke(opacity=0.5)
                           .move_to([TX0 + c * TW, y, 0]) for c in range(len(KVL_B) + 1) for y in (-2.0, -2.68)])
        legend = tx("示意：每页 4 个 token、每块 2 页；浅色多长＝页里写了几个 token", 18, GREY).move_to([0, -3.55, 0])
        cap0 = tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18)
        self.add(title, heads, pool, tables, slots, slab, *st, rows, kvbox, kvlab0, rnote, tl_lab, tl_slot, legend, cap0)
        self.cap = cap0
        kvlab = kvlab0

        def say(s, color=GREY_B, after=0.4):
            n = tx(s, 26, color).next_to(title, DOWN, buff=0.18)
            self.play(FadeOut(self.cap), FadeIn(n), run_time=0.35)
            self.cap = n
            if after:
                self.wait(after)

        def hold(t=1.8):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        def tcell(c, r, blk):
            name, bi, pg, ntok = blk
            col = KVL_COL[name]
            g = VGroup(Rectangle(width=TW - 0.1, height=0.5, stroke_color=col, stroke_width=2.5,
                                 fill_color=col, fill_opacity=0.3 if r == 0 else 0.9),
                       tx("%s 块%d · %d 个" % (name, bi, ntok) if r == 0 else "%s 块%d" % (name, bi), 18, WHITE))
            g[1].move_to(g[0].get_center())
            return g.move_to([TX0 + c * TW, -2.0 if r == 0 else -2.68, 0])

        inslot = [None, None]
        timeline = VGroup()
        self.wait(0.6)
        for t in range(len(KVL_B) + 1):
            say(KVL_SAY[t], after=0.2)
            anims = []
            # 算：上一块（在 (t-1)%2 号槽里）
            if t >= 1:
                ci = (t - 1) % 2
                name, bi, pg, ntok = KVL_B[t - 1]
                col = KVL_COL[name]
                ns = status(ci, "在算：%s 块%d" % (name, bi), col)
                nl = tx("%s 块%d 的 K、V" % (name, bi), 20, col, True).move_to(kvbox.get_center())
                tc = tcell(t, 1, KVL_B[t - 1])
                timeline.add(tc)
                anims += [FadeOut(st[ci]), FadeIn(ns), FadeOut(kvlab), FadeIn(nl), kvbox.animate.set_stroke(color=col),
                          FadeIn(tc), slots[ci].animate.set_stroke(color=col, width=3.5)]
                st[ci], kvlab = ns, nl
            # 搬：第 t 块进 t%2 号槽，和「算」同时进行
            if t < len(KVL_B):
                si = t % 2
                name, bi, pg, ntok = KVL_B[t]
                col = KVL_COL[name]
                flies = VGroup()
                for q, p in enumerate(pg):
                    c = page(*where[p]).move_to(ppos(p))
                    flies.add(c)
                    anims.append(c.animate.move_to(chip_pos(si, q)))
                inslot[si] = flies
                ns = status(si, "在装下一块：%s 块%d" % (name, bi), GREY_B)
                tc = tcell(t, 0, KVL_B[t])
                timeline.add(tc)
                anims += [FadeOut(st[si]), FadeIn(ns), FadeIn(tc), Indicate(tables[[s[0] for s in KVL_SEQS].index(name)], scale_factor=1.06)]
                st[si] = ns
            self.play(*anims, run_time=1.1)
            if t >= 1:
                self.play(*[Indicate(r, color=KVL_COL[KVL_B[t - 1][0]], scale_factor=1.08) for r in rows], run_time=0.6)
            hold()
            if t >= 1:                                   # 算完腾槽
                ci = (t - 1) % 2
                ns = status(ci, "空着", GREY)
                self.play(FadeOut(inslot[ci]), FadeOut(st[ci]), FadeIn(ns), slots[ci].animate.set_stroke(color=GREY_B, width=2), run_time=0.4)
                self.remove(inslot[ci])
                st[ci], inslot[ci] = ns, None
        say("每个字节只搬一次，算的时候下一块已在路上", YELLOW, after=0)
        self.wait(2.2)
        # 收尾复位：回到第 0 帧
        self.play(FadeOut(timeline), FadeOut(kvlab), FadeOut(self.cap), kvbox.animate.set_stroke(color=GREY), run_time=0.5)
        self.remove(timeline, kvlab, self.cap)
        self.play(FadeIn(kvlab0), run_time=0.3)
        self.add(tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18))
        self.wait(0.6)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "KVLoad.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
