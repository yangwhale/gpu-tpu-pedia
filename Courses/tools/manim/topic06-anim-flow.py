# -*- coding: utf-8 -*-
r"""专题六「推理」两支讲「搬」的动画：decode 每一步搬一趟权重（DecodeTrip）、长上下文切序列时谁搬家（WhoMoves）。
两档课件共用；画面和本文件里都不写课件节号。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · DecodeTrip —— 「decode 卡在搬」是一个「看着它慢」的现象：同一把时钟下，搬的条一格格往前爬，算的条几乎不动。
    静态条形图（fig6-b1）给得出 5 毫秒对 0.016 毫秒的结果，给不出两件只在时间里才有的事：
    ① 权重一层层从显存搬上来、用完就扔，下一个字又得从头搬一趟；② 行数从 1 涨到 313，搬的条不变、算的条追上来。
  · WhoMoves —— 「谁搬家」就是一串动作：PCP 里笔记沿环一站站传、每张卡边收边合；DCP 里一个提问发到每张卡、
    各算一段、最后在终点合一次。静态图（fig6-pcp-dcp）画得出两种切法的表，画不出「谁在动、往哪动、在哪合」。

⛔ 数据从哪来（当场算，断言钉住）
  · DecodeTrip：DeepSeek-V3 每个字约 37 GB 激活权重（FP8，每参数 1 字节）；TPU v7 显存 7.37 TB/s、FP8 4,614 TFLOP/s
    （bf16 2,307 的两倍）。搬一趟 37e9 ÷ 7.37e12 ≈ 5.0 ms；一行算 2 × 37e9 ÷ 4.614e15 ≈ 0.016 ms；
    0.016 ÷ 5 ≈ 0.32%。313 ＝ 2,307 ÷ 7.37（bf16 屋脊）＝ FP8 屋脊 626 ÷ 每参数 2 次运算。
    「313 行时算和搬差不多一样长」＝ 313 × 0.016 ≈ 5 ms，是本课推导（同一组规格数，只是换个说法）。
    条长按真实比例画：5 ms 占 8 个单位，0.016 ms 只有 0.026 个单位（画面上一两个像素）——「细到看不见」就是要点。
    权重画成 8 格是示意（V3 有 61 层）。
  · WhoMoves：4 张卡、4 段、示意。PCP 照 Ring Attention：卡 i 拿第 i 段提问，笔记段沿环往右传，3 步后每段笔记到过每张卡；
    因果掩码下卡 i 只用第 0…i 段笔记（三角外的段路过不用算）—— ring_sim 现模拟、断言每张卡最后合了 i+1 段。
    DCP 照 vLLM：笔记按字轮流存（卡 i 存第 i、i+4、i+8… 个字），提问发到每张卡，各交出分子和分母，终点加总后只除一次。

⛔ 刻意没画
  · DecodeTrip：KV、激活值和通信（只算权重的理想下限，正文交代）；屋脊折线本身（fig6-ai 的事）；MoE 只挑部分专家。
  · WhoMoves：Ring 的负载均衡排布（zigzag 等，卡 0 那张很闲）；KV 头和 Q 头的个数（「谁小搬谁」留给正文）；
    算例数字（「先别除」的数字算例在 fig6-softmax-merge）。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic06-anim-flow.py DecodeTrip WebPages/media/topic06-trip.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-flow.py WhoMoves   WebPages/media/topic06-whomoves.mp4
   两支每一步落地的停顿时刻写进 steps/<Scene>.json（课件挂 data-pauses）。
"""
import json
import os

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体：不指定时拉丁字母会回落到衬线体、标点按繁中居中
from manim import (Scene, VGroup, Rectangle, Line, Text, ValueTracker, FadeIn, FadeOut, Indicate,
                   WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, ORANGE,
                   UP, DOWN, LEFT, RIGHT, linear, always_redraw)

STEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps")


def tx(s, size=26, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def dump_steps(name, marks, t):
    os.makedirs(STEPS_DIR, exist_ok=True)
    with open(os.path.join(STEPS_DIR, name + ".json"), "w") as fh:
        json.dump({"pauses": marks, "duration": round(t, 2)}, fh)


# ════════════════════════════════════════════════════════════════
# ① DecodeTrip：每出一个字，权重从显存搬一趟
# ════════════════════════════════════════════════════════════════
V3_ACTIVE_B = 37e9          # 每个字要读的激活权重，FP8 每参数 1 字节
V7_BW = 7.37e12             # B/s
V7_BF16 = 2307e12           # FLOP/s
V7_FP8 = 2 * V7_BF16
T_MOVE = V3_ACTIVE_B / V7_BW * 1e3              # ms
T_ROW = 2 * V3_ACTIVE_B / V7_FP8 * 1e3          # ms，一行
RIDGE = V7_BF16 / V7_BW
assert round(T_MOVE, 1) == 5.0 and round(T_ROW, 3) == 0.016
assert round(T_ROW / T_MOVE * 100, 2) == 0.32
assert round(RIDGE) == 313 and round(V7_FP8 / V7_BW / 2) == 313
assert abs(313 * T_ROW - T_MOVE) / T_MOVE < 0.01                # 313 行：算 ≈ 搬（本课推导）
N_BLK = 8                    # 示意：权重画成 8 格
BAR_W = 8.0                  # 5 ms 占的长度
U_MS = BAR_W / T_MOVE


class DecodeTrip(Scene):
    def construct(self):
        marks = []
        title = tx("每出一个字，权重都要从显存搬一趟", 30).to_edge(UP)
        WX, BX, Y0 = -4.75, 2.55, -0.05
        wh = Rectangle(width=3.0, height=3.9, stroke_color=BLUE, stroke_width=3).move_to([WX, Y0, 0])
        wh_lab = tx("显存（仓库）", 22, BLUE).next_to(wh, UP, buff=0.12)
        wb = Rectangle(width=4.9, height=3.9, stroke_color=ORANGE, stroke_width=3).move_to([BX, Y0, 0])
        wb_lab = tx("算的地方（工作台）", 22, ORANGE).next_to(wb, UP, buff=0.12)
        wh_note = tx("37 GB 权重", 20, GREY_B).move_to([WX, Y0 - 1.68, 0])

        def blk_pos(k):
            return [WX, Y0 + 1.3 - k * 0.36, 0]

        def make_blocks():
            return VGroup(*[Rectangle(width=2.3, height=0.28, stroke_width=0, fill_color=BLUE, fill_opacity=0.8)
                            .move_to(blk_pos(k)) for k in range(N_BLK)])

        blocks0 = make_blocks()
        LAND = [BX - 1.25, Y0 + 0.2, 0]            # 权重格落在工作台左半
        ROWX = BX + 1.25                            # 行画在工作台右半
        row1 = Rectangle(width=1.6, height=0.16, stroke_width=0, fill_color=YELLOW, fill_opacity=0.95).move_to([ROWX, Y0 + 0.2, 0])
        row1_lab = tx("1 个人 ＝ 1 行", 20, YELLOW).next_to(row1, DOWN, buff=0.18)

        # 两根计时条：按真实比例（5 ms ＝ BAR_W）
        BY_M, BY_C, BX0 = -2.62, -3.12, -4.3
        lab_m = tx("搬", 22, BLUE, True).move_to([BX0 - 0.45, BY_M, 0])
        lab_c = tx("算", 22, ORANGE, True).move_to([BX0 - 0.45, BY_C, 0])
        legend = tx("只算权重的理想下限：DeepSeek-V3、一颗 TPU v7、FP8；条长按真实比例，权重画成 8 格是示意", 16, GREY).move_to([0, -3.65, 0])
        mv = ValueTracker(0.0)                      # 这一趟搬到几成
        rows = ValueTracker(1.0)                    # 一趟喂几行

        def bar(w, y, col):
            r = Rectangle(width=max(w, 0.035), height=0.3, stroke_width=0, fill_color=col, fill_opacity=0.9)
            r.move_to([BX0 + max(w, 0.035) / 2, y, 0])
            if w <= 0:
                r.set_fill(opacity=0)
            return r

        bar_m = always_redraw(lambda: bar(BAR_W * mv.get_value(), BY_M, BLUE))
        bar_c = always_redraw(lambda: bar(U_MS * T_ROW * rows.get_value() * mv.get_value(), BY_C, ORANGE))
        cap0 = tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18)
        self.add(title, wh, wh_lab, wb, wb_lab, wh_note, blocks0, lab_m, lab_c, legend, bar_m, bar_c, cap0)
        self.cap = cap0
        shown = VGroup()

        def say(s, color=GREY_B):
            n = tx(s, 26, color).next_to(title, DOWN, buff=0.18)
            self.play(FadeOut(self.cap), FadeIn(n), run_time=0.35)
            self.cap = n

        def hold(t=2.0):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        def trip(blocks, rowmob, per):
            """权重一格格搬上工作台、跟行乘一下、用完就扔；搬的条跟着爬。"""
            for k, b in enumerate(blocks):
                # ⛔ rate_func 只给时钟（mv）：写在 play 上会盖掉 Indicate 的 there_and_back，行会停在白色
                self.play(b.animate.move_to(LAND), mv.animate(rate_func=linear).set_value((k + 0.8) / N_BLK),
                          run_time=per * 0.65)
                self.play(Indicate(rowmob, color=WHITE, scale_factor=1.04), b.animate.set_fill(opacity=0.0),
                          mv.animate(rate_func=linear).set_value((k + 1) / N_BLK), run_time=per * 0.35)
                self.remove(b)

        self.wait(0.6)
        # ── 第 1 步：一个人
        say("一个人：每层权重从显存搬上来，只跟这 1 行乘一下")
        self.play(FadeIn(row1), FadeIn(row1_lab), run_time=0.4)
        shown.add(row1, row1_lab)
        self.remove(blocks0)
        b1 = make_blocks()
        self.add(b1)
        trip(b1, row1, 0.5)
        tm = tx("约 5 毫秒", 20, BLUE, True).move_to([BX0 + BAR_W + 0.65, BY_M, 0])
        tc = tx("0.016 毫秒：算力只用上 0.32%", 20, ORANGE, True).next_to([BX0 + 0.05, BY_C, 0], RIGHT, buff=0.15)
        out1 = VGroup(Rectangle(width=1.5, height=0.5, stroke_color=GREEN, stroke_width=3, fill_color=GREEN, fill_opacity=0.25),
                      tx("出 1 个字", 20, GREEN, True)).move_to([ROWX, Y0 + 1.3, 0])
        self.play(FadeIn(tm), FadeIn(tc), FadeIn(out1), run_time=0.5)
        shown.add(tm, tc, out1)
        say("第 1 步完成：搬约 5 毫秒，算 0.016 毫秒，出 1 个字", GREEN)
        hold(2.2)
        # ── 第 2 步：下一个字，再搬一趟
        say("下一个字：工作台放不下 37 GB，权重得从头再搬一趟")
        self.play(FadeOut(tm), FadeOut(tc), mv.animate.set_value(0.0), run_time=0.4, rate_func=linear)
        self.remove(tm, tc)
        shown.remove(tm, tc)
        b2 = make_blocks()
        self.play(FadeIn(b2), run_time=0.3)
        trip(b2, row1, 0.22)
        out2 = VGroup(Rectangle(width=1.5, height=0.5, stroke_color=GREEN, stroke_width=3, fill_color=GREEN, fill_opacity=0.25),
                      tx("出 1 个字", 20, GREEN, True)).move_to([ROWX - 1.7, Y0 + 1.3, 0])
        self.play(FadeIn(out2), FadeIn(tm), run_time=0.45)
        shown.add(out2, tm)
        say("第 2 步完成：又是一整趟、又是约 5 毫秒，还是只出 1 个字", GREEN)
        hold(2.0)
        # ── 第 3 步：一趟喂 313 行
        say("一趟喂 313 行：搬的还是这一车，算一点点追上来")
        rowsblk = Rectangle(width=1.6, height=2.0, stroke_width=0, fill_color=YELLOW, fill_opacity=0.7).move_to([ROWX, Y0 + 0.05, 0])
        rows_lab = tx("313 行", 22, YELLOW, True).next_to(rowsblk, DOWN, buff=0.12)
        self.play(FadeOut(out1), FadeOut(out2), FadeOut(tm), FadeOut(row1_lab),
                  row1.animate.become(rowsblk), FadeIn(rows_lab),
                  mv.animate.set_value(0.0), run_time=0.6, rate_func=linear)
        self.remove(out1, out2, tm, row1_lab)
        shown.remove(out1, out2, tm, row1_lab, row1)
        shown.add(rows_lab)
        rows.set_value(313)
        b3 = make_blocks()
        self.play(FadeIn(b3), run_time=0.3)
        trip(b3, row1, 0.5)
        tm3 = tx("约 5 毫秒", 20, BLUE, True).move_to([BX0 + BAR_W + 0.65, BY_M, 0])
        tc3 = tx("313 × 0.016 ≈ 5 毫秒", 20, ORANGE, True).move_to([BX0 + BAR_W + 1.45, BY_C, 0])
        self.play(FadeIn(tm3), FadeIn(tc3), run_time=0.4)
        shown.add(tm3, tc3)
        say("第 3 步完成：算和搬一样长 —— 再多就卡在算，prefill 一趟几千行就在这一边", GREEN)
        hold(2.4)
        say("decode 卡在搬，prefill 卡在算", YELLOW)
        self.wait(2.0)
        # ── 收尾复位：回到第 0 帧
        self.play(FadeOut(shown), FadeOut(row1), FadeOut(self.cap), mv.animate.set_value(0.0),
                  run_time=0.6, rate_func=linear)
        self.remove(shown, row1, self.cap)
        rows.set_value(1.0)
        self.play(FadeIn(blocks0), run_time=0.4)
        self.add(tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18))
        self.wait(0.6)
        dump_steps("DecodeTrip", marks, self.renderer.time)


# ════════════════════════════════════════════════════════════════
# ② WhoMoves：PCP 搬笔记、DCP 搬提问
# ════════════════════════════════════════════════════════════════
NC = 4                       # 示意：4 张卡、4 段


def ring_sim():
    """Ring：卡 i 起手拿第 i 段笔记；每一步全体往右传一站（卡 NC-1 绕回卡 0）。
    返回 arrivals[s][i] ＝ 第 s 步后卡 i 手上是第几段，used[i] ＝ 卡 i 合过的段（因果：只用 ≤ i 的段）。"""
    hold = list(range(NC))
    used = {i: [i] for i in range(NC)}
    arrivals = []
    for _ in range(NC - 1):
        hold = [hold[(i - 1) % NC] for i in range(NC)]
        arrivals.append(list(hold))
        for i, j in enumerate(hold):
            if j <= i:
                used[i].append(j)
    return arrivals, used


RING, RING_USED = ring_sim()
assert all(sorted(RING_USED[i]) == list(range(i + 1)) for i in range(NC))      # 卡 i 最后合了第 0…i 段
assert all(sorted(col) == list(range(NC)) for col in zip(*([list(range(NC))] + RING)))  # 每段笔记到过每张卡
DCP_TOK = [[t for t in range(12) if t % NC == i] for i in range(NC)]           # 按字轮流存
assert DCP_TOK[1][:3] == [1, 5, 9]


class WhoMoves(Scene):
    def construct(self):
        marks = []
        title = tx("谁搬家：PCP 搬笔记，DCP 搬提问", 30).to_edge(UP)
        XS = [-4.8, -1.6, 1.6, 4.8]
        CY, CW, CH = 0.2, 2.8, 2.9
        cards = VGroup(*[Rectangle(width=CW, height=CH, stroke_color=GREY_B, stroke_width=2.5).move_to([x, CY, 0]) for x in XS])
        clabs = VGroup(*[tx("卡 %d" % i, 20, GREY_B).move_to([XS[i], CY + CH / 2 - 0.25, 0]) for i in range(NC)])
        legend = tx("4 张卡、4 段都是示意", 16, GREY).move_to([0, -3.65, 0])
        cap0 = tx(" ", 26, GREY_B).next_to(title, DOWN, buff=0.18)
        self.add(title, cards, clabs, legend, cap0)
        self.cap = cap0
        shown = VGroup()

        def say(s, color=GREY_B):
            n = tx(s, 25, color).next_to(title, DOWN, buff=0.18)
            self.play(FadeOut(self.cap), FadeIn(n), run_time=0.35)
            self.cap = n

        def hold(t=2.0):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        # ── PCP ─────────────────────────────────────────────
        YQ, YSL, YK, LANE = CY + 0.72, CY - 0.12, CY - 0.95, -1.75
        say("PCP（prefill）：几千个提问按行分给 4 张卡，每张卡先拿自己那段笔记")
        qs = VGroup(*[VGroup(Rectangle(width=2.2, height=0.42, stroke_width=0, fill_color=YELLOW, fill_opacity=0.75),
                             tx("提问 第 %d 段" % i, 18, WHITE, True)).move_to([XS[i], YQ, 0]) for i in range(NC)])
        slots = []
        for i in range(NC):
            row = VGroup()
            for j in range(NC):
                sq = Rectangle(width=0.46, height=0.4, stroke_width=2,
                               stroke_color=GREEN if j <= i else GREY, fill_color=GREEN, fill_opacity=0.0)
                if j > i:
                    sq.set_stroke(opacity=0.35)
                g = VGroup(sq, tx(str(j), 16, GREY_B if j <= i else GREY)).move_to([XS[i] - 0.84 + j * 0.56, YSL, 0])
                if j > i:
                    g[1].set_opacity(0.4)
                row.add(g)
            slots.append(row)
        slots = VGroup(*slots)
        slot_lab = VGroup(*[tx("要合的笔记段", 15, GREY_B).move_to([XS[i], YSL - 0.38, 0]) for i in range(NC)])

        def kv(j, i):
            return VGroup(Rectangle(width=1.5, height=0.42, stroke_width=0, fill_color=BLUE, fill_opacity=0.85),
                          tx("笔记 第 %d 段" % j, 17, WHITE, True)).move_to([XS[i], YK, 0])

        held = [kv(i, i) for i in range(NC)]
        hid = list(range(NC))
        self.play(FadeIn(qs), FadeIn(slots), FadeIn(slot_lab), *[FadeIn(h) for h in held], run_time=0.7)
        shown.add(qs, slots, slot_lab)
        # 起手：各合自己那段
        self.play(*[slots[i][i][0].animate.set_fill(GREEN, 0.75) for i in range(NC)], run_time=0.5)
        tri = tx("灰框＝三角外：靠前的提问看不到后面的字，路过不用算", 18, GREY_B).move_to([0, -2.55, 0])
        self.play(FadeIn(tri), run_time=0.3)
        shown.add(tri)
        hold(1.6)
        for s, arr in enumerate(RING, start=1):
            say("第 %d 站：笔记段全体往右传一站，最右边那段绕回卡 0" % s, BLUE)
            # 前三段直接往右平移一张卡；最右那段绕回：先往右出卡，沿下方车道绕到最左，再往右进卡 0（构建手册 10 · 房规 7）
            moves = [held[i].animate.move_to([XS[i + 1], YK, 0]) for i in range(NC - 1)]
            last = held[NC - 1]
            self.play(*moves, last.animate.move_to([XS[NC - 1] + 1.45, LANE, 0]), run_time=0.9)
            self.play(last.animate.move_to([XS[0] - 1.45, LANE, 0]), run_time=0.6)
            self.play(last.animate.move_to([XS[0], YK, 0]), run_time=0.5)
            held = [held[NC - 1]] + held[:NC - 1]
            hid = [hid[NC - 1]] + hid[:NC - 1]
            assert hid == arr
            fills, dims = [], []
            for i, j in enumerate(arr):
                if j <= i:
                    fills.append(slots[i][j][0].animate.set_fill(GREEN, 0.75))
                    fills.append(Indicate(held[i], color=GREEN, scale_factor=1.06))
                else:
                    dims.append(held[i].animate.set_opacity(0.3))
            self.play(*fills, *dims, run_time=0.6)
            if s < len(RING):
                say("第 %d 站完成：用得上的边收边合，三角外的路过" % s, GREEN)
                hold(1.5)
                self.play(*[h.animate.set_opacity(1.0) for h in held], run_time=0.2)
        say("PCP 完成：提问不动，笔记绕环搬过来；每张卡边收边合，不用跨卡拼", GREEN)
        hold(2.4)
        self.play(FadeOut(shown), *[FadeOut(h) for h in held], run_time=0.5)
        self.remove(shown, *held)
        shown = VGroup()
        # ── DCP ─────────────────────────────────────────────
        say("DCP（decode）：每步只剩 1 个提问；笔记按字轮流存在 4 张卡上，不动")
        notes = VGroup()
        for i in range(NC):
            stripes = VGroup(*[Rectangle(width=0.16, height=0.9, stroke_width=0, fill_color=BLUE, fill_opacity=0.85)
                               .move_to([XS[i] - 1.0 + k * 0.25, CY - 0.2, 0]) for k in range(9)])
            lab = tx("第 %s… 个字" % "、".join(str(t) for t in DCP_TOK[i][:3]), 15, BLUE).move_to([XS[i], CY - 0.88, 0])
            notes.add(VGroup(stripes, lab))
        q = VGroup(Rectangle(width=1.5, height=0.4, stroke_width=0, fill_color=YELLOW, fill_opacity=0.9),
                   tx("1 个提问", 18, WHITE, True)).move_to([0, CY + CH / 2 + 0.3, 0])
        self.play(FadeIn(notes), FadeIn(q), run_time=0.6)
        shown.add(notes)
        hold(1.4)
        say("第 1 步：这一个提问发到每张卡，各算自己那一段", YELLOW)
        qcs = [q.copy() for _ in range(NC)]
        self.play(*[c.animate.move_to([XS[i], CY + 0.65, 0]).scale(0.85) for i, c in enumerate(qcs)], run_time=1.0)
        self.play(*[Indicate(notes[i][0], color=YELLOW, scale_factor=1.04) for i in range(NC)], run_time=0.6)
        parts = VGroup(*[VGroup(Rectangle(width=1.9, height=0.4, stroke_width=0, fill_color=ORANGE, fill_opacity=0.85),
                                tx("分子、分母", 17, WHITE, True)).move_to([XS[i], CY - 1.25 - 0.05, 0]) for i in range(NC)])
        self.play(FadeIn(parts), run_time=0.4)
        shown.add(*qcs)
        say("第 1 步完成：每张卡交出自己那段的分子和分母，先别除", ORANGE)
        hold(1.8)
        say("第 2 步：在终点拼一次 —— 分子加总、分母加总，只除一次", ORANGE)
        merge = VGroup(Rectangle(width=6.6, height=0.5, stroke_color=GREEN, stroke_width=3, fill_color=GREEN, fill_opacity=0.2),
                       tx("分子加总 ÷ 分母加总 ＝ 这个提问的答案", 19, GREEN, True)).move_to([0, -2.55, 0])
        self.play(*[p.animate.move_to([(i - 1.5) * 1.2, -2.55, 0]).scale(0.55).set_opacity(0.0) for i, p in enumerate(parts)],
                  FadeIn(merge), run_time=1.0)
        self.remove(parts)
        shown.add(merge)
        say("第 2 步完成：笔记没动，提问搬了家，答案在终点拼", GREEN)
        hold(2.0)
        say("谁小搬谁：提问多就搬笔记（PCP），只剩 1 个提问就搬提问（DCP）", YELLOW)
        self.wait(2.4)
        # ── 收尾复位
        self.play(FadeOut(shown), FadeOut(q), FadeOut(self.cap), run_time=0.6)
        self.remove(shown, q, self.cap)
        self.add(tx(" ", 25, GREY_B).next_to(title, DOWN, buff=0.18))
        self.wait(0.6)
        dump_steps("WhoMoves", marks, self.renderer.time)
