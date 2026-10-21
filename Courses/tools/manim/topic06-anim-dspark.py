# -*- coding: utf-8 -*-
r"""专题六「推理」讲 DSpark 的两支动画：顺序小头怎么把并行草稿串起来（DSparkDraft）、调度器怎么按把握和负载分验证名额（DSparkSched）。
两档课件共用；画面和本文件里都不写课件节号。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · DSparkDraft —— 「并行」和「顺序」本身就是时间上的区别：并行是两格同一刻各自抽签，顺序是前一格落定了、后一格才改分数再抽。
    静态图（fig6-spec-three）画得出结果「of problem」和「of course」，画不出「同一刻」对「先后」——而毛病正出在这个先后上。
  · DSparkSched —— 调度器是一个贪心过程：按存活概率从高到低一个个放进验证、每放一个看一眼吞吐、吞吐一降就停。
    「停在哪」取决于走到哪，静态图只能给终点；动画里同一批草稿走两遍（闲时、忙时），看着停的位置往前挪。

⛔ 数据从哪来
  · DSparkDraft：of course／no problem／of problem 的例子出自 DSpark 论文（arXiv 2607.05147）§3.1，「选了 of 就抬高 course」也是论文原话。
    两格的分数 55%／45%、顺序头之后的 95%／5% 都是示意；拼串概率 2 × 0.55 × 0.45 ≈ 50% 是本课推导（按这组示意数）。
    骨干 5 层、顺序头是低秩查表（r＝256）：论文 §3.1、§4.1。整轮只慢 0.2–1.3%（batch 128、草稿长度 4–16）：§4.3.2。
    接受长度比 DFlash 高 16–18%（Qwen3 4B／8B／14B 分别 16.3／18.4／18.3%）：§4.2。计时条长短是示意。
  · DSparkSched：算法照论文 Algorithm 1（§3.2.2）在脚本里现跑：存活概率＝把握连乘；全批候选按存活概率降序；
    逐个放进验证，Θ ＝ 预期收下的 token 数 × 每秒步数，Θ 不再涨就停（early stop）。断言钉住两遍各停在第几个。
    三个请求的把握、「一步耗时 ＝ 固定开销 ＋ 每个 token 1 份」这条示意吞吐表（闲时固定开销 100、忙时 4）都是示意，
    只为让同一批草稿在两种负载下停在不同位置。每请求草稿 5 个照线上 DSpark-5（§5.4）。

⛔ 刻意没画
  · 线上版的两处改动（§5.2）：真机吞吐表是台阶状的，所以去掉了 early stop、改成全局搜；为配合零开销调度，验多少个（top-K 的 K）
    用两步之前的把握来估。动画只演论文算法 1 的原样，改动写在正文。
  · 置信度头的校准（STS）、RNN 版顺序头、拒绝采样的概率细节、验完之后大模型补的那 1 个字。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic06-anim-dspark.py DSparkDraft WebPages/media/topic06-dspark-draft.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-dspark.py DSparkSched WebPages/media/topic06-dspark-sched.mp4
   两支每一步落地的停顿时刻写进 steps/<Scene>.json（课件挂 data-pauses）。
"""
import json
import os

from manim import (Scene, VGroup, Rectangle, Line, Dot, Text, FadeIn, FadeOut, Indicate, Transform,
                   WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, PURPLE,
                   UP, DOWN, LEFT, RIGHT, linear)

STEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")


def tx(s, size=26, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def dump_steps(name, marks, t):
    os.makedirs(STEPS_DIR, exist_ok=True)
    with open(os.path.join(STEPS_DIR, name + ".json"), "w") as fh:
        json.dump({"pauses": marks, "duration": round(t, 2)}, fh)


class _Base(Scene):
    def setup_caps(self, title):
        self.title_m = title
        self.cap = tx(" ", 25, GREY_B).next_to(title, DOWN, buff=0.18)
        self.add(self.cap)
        self.marks = []

    def say(self, s, color=GREY_B):
        n = tx(s, 25, color).next_to(self.title_m, DOWN, buff=0.18)
        self.play(FadeOut(self.cap), FadeIn(n), run_time=0.35)
        self.cap = n

    def hold(self, t=2.0):
        self.marks.append(round(self.renderer.time + 0.3, 2))
        self.wait(t)


# ════════════════════════════════════════════════════════════════
# ① DSparkDraft：并行各抽各的 → 顺序小头按前一格改分数
# ════════════════════════════════════════════════════════════════
P1 = {"of": 0.55, "no": 0.45}                  # 示意：第 1 格两种开头
P2_PAR = {"course": 0.55, "problem": 0.45}     # 并行：第 2 格看不到第 1 格，只能按两种说法平摊
P2_SEQ = {"course": 0.95, "problem": 0.05}     # 顺序头：第 1 格定了 of 之后（示意）
COLLIDE = P1["of"] * P2_PAR["problem"] + P1["no"] * P2_PAR["course"]
assert round(COLLIDE, 3) == 0.495               # 拼串（of problem 或 no course）约一半 —— 本课推导
assert abs(sum(P2_SEQ.values()) - 1) < 1e-9 and P2_SEQ["course"] > P2_PAR["course"]
GAIN_VS_DFLASH = (16.3, 18.4, 18.3)            # §4.2，Qwen3-4B／8B／14B
assert (round(min(GAIN_VS_DFLASH)), round(max(GAIN_VS_DFLASH))) == (16, 18)


class DSparkDraft(_Base):
    def construct(self):
        title = tx("DSpark 第一刀：并行打底，再按前一格改下一格", 30).to_edge(UP)
        XA, XS = -4.6, [-1.6, 2.6]                     # 上一轮确认的字、两个草稿格
        YT, YB = 1.05, -0.75                           # 字格行、分数条区
        BW, BH = 2.3, 0.72

        def slot(i, word=None, col=GREY_B):
            g = VGroup(Rectangle(width=BW, height=BH, stroke_color=col, stroke_width=3,
                                 fill_color=col, fill_opacity=0.18 if word else 0.0))
            if word:
                g.add(tx(word, 30, WHITE, True))
            return g.move_to([XS[i], YT, 0])

        anchor = VGroup(Rectangle(width=1.7, height=BH, stroke_color=GREEN, stroke_width=3, fill_color=GREEN, fill_opacity=0.35),
                        tx("Yes,", 30, WHITE, True)).move_to([XA, YT, 0])
        a_lab = tx("上一轮确认的字", 18, GREEN).next_to(anchor, DOWN, buff=0.15)
        empties = VGroup(slot(0), slot(1))
        s_labs = VGroup(*[tx("草稿第 %d 格" % (i + 1), 18, GREY_B).next_to(empties[i], UP, buff=0.12) for i in range(2)])
        legend = tx("例子出自 DSpark 论文；分数、条长都是示意", 16, GREY).move_to([0, -3.65, 0])
        self.add(title, anchor, a_lab, empties, s_labs, legend)
        self.setup_caps(title)
        shown = VGroup()

        def bars(i, dist, col):
            """一格下面的候选分数条：词、条、百分比。"""
            g = VGroup()
            for k, (w, p) in enumerate(dist.items()):
                y = YB - k * 0.55
                g.add(VGroup(tx(w, 22, WHITE).move_to([XS[i] - 1.55, y, 0]),
                             Rectangle(width=max(1.9 * p, 0.04), height=0.3, stroke_width=0, fill_color=col, fill_opacity=0.85)
                             .move_to([XS[i] - 0.65 + 1.9 * p / 2, y, 0]),
                             tx("%d%%" % round(p * 100), 18, GREY_B).move_to([XS[i] - 0.65 + 1.9 * p + 0.38, y, 0])))
            return g

        self.wait(0.6)
        # ── 第 1 步：DFlash 式并行
        self.say("并行（DFlash）：一趟前向，两格同时出分数")
        b1, b2 = bars(0, P1, PURPLE), bars(1, P2_PAR, PURPLE)
        self.play(FadeIn(b1), FadeIn(b2), run_time=0.6)
        shown.add(b1, b2)
        self.say("第 2 格看不见第 1 格选了啥，只能两种说法各押一半")
        self.play(Indicate(b2, color=YELLOW, scale_factor=1.04), run_time=0.7)
        self.wait(0.5)
        self.say("两格各自抽签：第 1 格抽到 of，第 2 格抽到 problem")
        f1, f2 = slot(0, "of", GREEN), slot(1, "problem", RED)
        self.play(Indicate(b1[0], color=GREEN), Indicate(b2[1], color=RED), run_time=0.6)
        self.play(Transform(empties[0], f1), Transform(empties[1], f2), run_time=0.5)
        bad = tx("✗ 拼成「of problem」：两种说法串了（按这组数，约一半概率）", 20, RED, True).move_to([0.5, -2.45, 0])
        self.play(FadeIn(bad), run_time=0.4)
        shown.add(bad)
        self.say("第 1 步完成：一趟出整块很快，但越往块尾越容易拼串", GREEN)
        self.hold(2.2)
        # ── 第 2 步：顺序小头
        self.say("DSpark：骨干还是这一趟；再加一个顺序小头，从左往右走")
        self.play(FadeOut(bad), FadeOut(b2), Transform(empties[1], slot(1)), run_time=0.5)
        self.remove(bad, b2)
        shown.remove(bad, b2)
        self.say("第 1 格定了 of：小头按 of 给第 2 格加一笔分")
        arrow = Line([XS[0] + BW / 2 + 0.05, YT, 0], [XS[1] - BW / 2 - 0.05, YT, 0], color=GREEN, stroke_width=5).add_tip(tip_length=0.22)
        arrow_lab = tx("前一格的字 → 下一格的偏置", 17, GREEN).next_to(arrow, DOWN, buff=0.14)
        self.play(FadeIn(arrow), FadeIn(arrow_lab), run_time=0.5)
        shown.add(arrow, arrow_lab)
        b2p = bars(1, P2_PAR, PURPLE)
        self.add(b2p)
        b2s = bars(1, P2_SEQ, GREEN)
        self.play(Transform(b2p, b2s), run_time=1.1)
        shown.add(b2p)
        self.say("course 抬到 95%，problem 压到 5%；再抽")
        self.play(Indicate(b2p[0], color=GREEN), run_time=0.6)
        self.play(Transform(empties[1], slot(1, "course", GREEN)), run_time=0.5)
        good = tx("✓ 「of course」", 22, GREEN, True).move_to([0.5, -2.45, 0])
        self.play(FadeIn(good), run_time=0.4)
        shown.add(good)
        self.say("第 2 步完成：块里的字前后接得上了", GREEN)
        self.hold(2.2)
        # ── 第 3 步：代价
        self.say("代价：贵的骨干只跑一趟；小头每格只是查一次表、乘一个小矩阵")
        self.play(FadeOut(b1), FadeOut(b2p), FadeOut(good), run_time=0.4)
        self.remove(b1, b2p, good)
        shown.remove(b1, b2p, good)
        X0, Y1, Y2 = -4.0, -1.05, -1.75
        lab1 = tx("骨干 5 层", 20, PURPLE, True).move_to([X0 - 1.15, Y1, 0])
        lab2 = tx("顺序小头", 20, GREEN, True).move_to([X0 - 1.15, Y2, 0])
        bb = Rectangle(width=7.0, height=0.34, stroke_width=0, fill_color=PURPLE, fill_opacity=0.85)
        bb.move_to([X0 + 3.5, Y1, 0])
        ticks = VGroup(*[Rectangle(width=0.16, height=0.34, stroke_width=0, fill_color=GREEN, fill_opacity=0.9)
                         .move_to([X0 + 7.0 + 0.12 + k * 0.22, Y2, 0]) for k in range(2)])
        self.play(FadeIn(lab1), FadeIn(lab2), run_time=0.3)
        self.play(FadeIn(bb, shift=RIGHT * 0.3), run_time=0.8, rate_func=linear)
        self.play(FadeIn(ticks[0]), run_time=0.2)
        self.play(FadeIn(ticks[1]), run_time=0.2)
        cost = tx("batch 128：草稿 4 到 16 个，整轮只比 DFlash 慢 0.2–1.3%", 20, GREY_B).move_to([0.3, -2.45, 0])
        self.play(FadeIn(cost), run_time=0.4)
        shown.add(lab1, lab2, bb, ticks, cost)
        self.say("第 3 步完成：接受长度比 DFlash 高 16–18%（Qwen3 4B–14B）", GREEN)
        self.hold(2.4)
        self.say("一趟出整块的速度留着，块里的前后照应补回来", YELLOW)
        self.wait(2.0)
        # ── 收尾复位
        self.play(FadeOut(shown), FadeOut(self.cap), Transform(empties[0], slot(0)), Transform(empties[1], slot(1)),
                  run_time=0.6)
        self.remove(shown, self.cap)
        self.add(tx(" ", 25, GREY_B).next_to(title, DOWN, buff=0.18))
        self.wait(0.6)
        dump_steps("DSparkDraft", self.marks, self.renderer.time)


# ════════════════════════════════════════════════════════════════
# ② DSparkSched：按存活概率排队，吞吐不涨就停
# ════════════════════════════════════════════════════════════════
REQS = [("请求 1 · 写代码", [0.95, 0.95, 0.90, 0.90, 0.85]),     # 示意把握
        ("请求 2 · 闲聊", [0.70, 0.60, 0.50, 0.50, 0.40]),
        ("请求 3 · 解题", [0.90, 0.85, 0.70, 0.60, 0.50])]
GAMMA = 5                                       # 线上 DSpark-5
FIX_LIGHT, FIX_HEAVY = 100, 4                   # 示意吞吐表：一步耗时 ＝ 固定开销 ＋ 每个 token 1 份


def survival(c):
    out, p = [], 1.0
    for x in c:
        p *= x
        out.append(p)
    return out


SURV = [survival(c) for _, c in REQS]
CANDS = sorted(((SURV[r][j], r, j) for r in range(len(REQS)) for j in range(GAMMA)), key=lambda t: (-t[0], t[1], t[2]))


def schedule(fix):
    """论文 Algorithm 1：τ、B 从「每请求 1 个必出的字」起步；逐个放进候选，Θ ＝ τ × SPS(B) 不再涨就停。
    返回 (被收下的候选列表, 每一步的 Θ／Θ0, 停下时试的那个候选或 None)。"""
    R = len(REQS)
    sps = lambda b: 1.0 / (fix + b)
    B, tau = R, float(R)
    best = tau * sps(B)
    th0, picked, trace = best, [], [1.0]
    for a, r, j in CANDS:
        B += 1
        tau += a
        th = tau * sps(B)
        trace.append(th / th0)
        if th > best:
            best = th
            picked.append((r, j))
        else:
            return picked, trace, (r, j)
    return picked, trace, None


PICK_L, TR_L, STOP_L = schedule(FIX_LIGHT)
PICK_H, TR_H, STOP_H = schedule(FIX_HEAVY)
PER_H = [sum(1 for r, _ in PICK_H if r == k) for k in range(len(REQS))]
assert len(PICK_L) == 14 and STOP_L == (1, 4)           # 闲时：只有闲聊最后那格（存活 4%）没进
assert len(PICK_H) == 7 and PER_H == [4, 1, 2]          # 忙时：代码 4 个、闲聊 1 个、解题 2 个
# 贪心顺序天然守住前缀：同一请求里第 j 格不会比第 j-1 格先进
for pick in (PICK_L, PICK_H):
    for r in range(len(REQS)):
        js = [j for rr, j in pick if rr == r]
        assert js == list(range(len(js)))


class DSparkSched(_Base):
    def construct(self):
        title = tx("DSpark 第二刀：按把握排队，按负载定验几个", 30).to_edge(UP)
        X0, DX, YS = -3.7, 1.0, [1.15, 0.05, -1.05]
        CW, CH = 0.86, 0.86

        def cpos(r, j):
            return [X0 + j * DX, YS[r], 0]

        cells = VGroup(*[Rectangle(width=CW, height=CH, stroke_color=GREY_B, stroke_width=2).move_to(cpos(r, j))
                         for r in range(3) for j in range(GAMMA)])
        rlabs = VGroup(*[tx(REQS[r][0], 19, GREY_B).move_to([X0 - 1.85, YS[r], 0]) for r in range(3)])
        clabs = VGroup(*[tx("第 %d 格" % (j + 1), 16, GREY).move_to([X0 + j * DX, YS[0] + 0.68, 0]) for j in range(GAMMA)])
        # 右边：Θ 轨迹
        PX0, PX1, PY0, PY1 = 2.55, 6.55, -1.55, 1.45
        ax = VGroup(Line([PX0, PY0, 0], [PX1, PY0, 0], color=GREY, stroke_width=2),
                    Line([PX0, PY0, 0], [PX0, PY1, 0], color=GREY, stroke_width=2))
        ax_x = tx("已放进验证的草稿数", 16, GREY).move_to([(PX0 + PX1) / 2, PY0 - 0.3, 0])
        ax_y = tx("预期吞吐", 16, GREY).move_to([PX0, PY1 + 0.25, 0])
        legend = tx("算法照 DSpark 论文 Algorithm 1 现算；把握和吞吐表是示意", 16, GREY).move_to([0, -3.65, 0])
        self.add(title, cells, rlabs, clabs, ax, ax_x, ax_y, legend)
        self.setup_caps(title)
        shown = VGroup()

        def num(r, j, s, col=WHITE):
            return tx(s, 20, col, True).move_to(cpos(r, j))

        self.wait(0.6)
        # ── 第 1 步：把握 → 存活概率
        self.say("每个请求猜 5 个，小头给每格估一个「把握」")
        cs = VGroup(*[num(r, j, "%.2f" % REQS[r][1][j], GREY_B) for r in range(3) for j in range(GAMMA)])
        cdef = tx("把握＝前面几格都收下时，这一格也被收下的概率", 17, GREY_B).move_to([X0 + 2 * DX - 0.9, YS[2] - 0.85, 0])
        self.play(FadeIn(cs), FadeIn(cdef), run_time=0.6)
        shown.add(cdef)
        self.wait(1.2)
        self.say("把握从左往右连乘＝活到这一格的概率，越往后越低")
        fills = VGroup()
        for r in range(3):
            for j in range(GAMMA):
                a = SURV[r][j]
                h = CH * a
                fills.add(Rectangle(width=CW - 0.06, height=max(h, 0.02), stroke_width=0, fill_color=BLUE, fill_opacity=0.45)
                          .move_to([X0 + j * DX, YS[r] - CH / 2 + h / 2, 0]))
        ss = VGroup(*[num(r, j, "%d%%" % round(SURV[r][j] * 100)) for r in range(3) for j in range(GAMMA)])
        self.play(FadeIn(fills), Transform(cs, ss), run_time=0.9)
        shown.add(fills, cs)
        self.say("第 1 步完成：闲聊那行掉得最快，写代码那行最稳", GREEN)
        self.hold(2.0)

        def run(picks, trace, stop, fix, tag, col):
            ymax = max(trace) * 1.08
            n = len(trace)

            def pt(k, v):
                return [PX0 + 0.15 + k * (PX1 - PX0 - 0.3) / (len(CANDS)), PY0 + (v / ymax) * (PY1 - PY0 - 0.15), 0]

            note = tx("一步耗时 ＝ 固定 %d ＋ 每个 token 1" % fix, 18, col).move_to([(PX0 + PX1) / 2, PY1 + 0.7, 0])
            self.play(FadeIn(note), run_time=0.3)
            marks_g = VGroup(note)
            prev = Dot(pt(0, trace[0]), radius=0.06, color=col)
            self.play(FadeIn(prev), run_time=0.2)
            marks_g.add(prev)
            for k, (a, r, j) in enumerate(CANDS[:n - 1]):
                box = Rectangle(width=CW, height=CH, stroke_color=col, stroke_width=5).move_to(cpos(r, j))
                d = Dot(pt(k + 1, trace[k + 1]), radius=0.06, color=col)
                seg = Line(prev.get_center(), d.get_center(), color=col, stroke_width=3)
                if (r, j) == stop:
                    box.set_stroke(RED)
                    d.set_color(RED)
                    seg.set_color(RED)
                self.play(FadeIn(box), FadeIn(seg), FadeIn(d), run_time=0.28 if fix == FIX_LIGHT else 0.45, rate_func=linear)
                marks_g.add(box, seg, d)
                prev = d
            if stop is not None:
                x = tx("✗ 红框那格放进来吞吐反降，停", 18, RED, True).move_to([X0 + 2 * DX - 0.9, YS[2] - 1.35, 0])
                self.play(FadeIn(x), run_time=0.3)
                marks_g.add(x)
            return marks_g

        # ── 第 2 步：闲时
        self.say("闲时：全批候选按存活概率从高到低，一个个放进验证")
        g = run(PICK_L, TR_L, STOP_L, FIX_LIGHT, "闲", GREEN)
        self.say("第 2 步完成：多验几乎不拖慢这一步，放进 %d 个，只剩 4%% 那格没要" % len(PICK_L), GREEN)
        self.hold(2.2)
        self.play(FadeOut(g), run_time=0.4)
        self.remove(g)
        # ── 第 3 步：忙时
        self.say("忙时：同一批草稿，但每多验 1 个，整步都更慢")
        g = run(PICK_H, TR_H, STOP_H, FIX_HEAVY, "忙", YELLOW)
        self.say("第 3 步完成：只放进 %d 个 —— 代码 %d 个、闲聊 %d 个、解题 %d 个" % ((len(PICK_H),) + tuple(PER_H)), GREEN)
        self.hold(2.4)
        self.say("名额按把握排，总数看机器忙不忙；没把握的字不占名额", YELLOW)
        self.wait(2.2)
        # ── 收尾复位
        self.play(FadeOut(g), FadeOut(shown), FadeOut(self.cap), run_time=0.6)
        self.remove(g, shown, self.cap)
        self.add(tx(" ", 25, GREY_B).next_to(title, DOWN, buff=0.18))
        self.wait(0.6)
        dump_steps("DSparkSched", self.marks, self.renderer.time)
