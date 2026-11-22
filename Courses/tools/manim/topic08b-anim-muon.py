# -*- coding: utf-8 -*-
r"""专题九 §9.2「优化器里白搬的字节」的动画：一块权重在优化器里一步搬几趟、每趟多重（MuonShip）。
挂在静态图 fig-muon 下面。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  · 论点是「趟数」和「每趟多重」：按设计一块只搬两趟细的 BF16；编出来的程序多了两趟粗的 FP32
    （主权重被搬过去、加完再搬回来）。静态图把三条箭头并排摆着，读者看到的是「三条线」，
    看不出「先过去、在那边加、再回来」的先后，也数不出计数器一趟一趟涨到 5 份。
    趟数和粗细只有在时间里一趟一趟走，才数得清。

⛔ 数据从哪来（全部取自 topic08b_numbers，断言钉住）
  · N.MU_MOM_U ＝ 1 份（一块的 BF16 动量）、N.MU_PARAM_U ＝ 2 份（FP32 主权重，两倍）；公开课只给份数。
  · 按设计一块 N.MU_LEAF_DESIGN ＝ 2 份（动量过去 ＋ BF16 结果回来）；
    编出来一块 N.MU_LEAF_BEFORE ＝ 5 份（动量过去 ＋ 主权重过去 ＋ 新主权重回来）。
  · N.MU_LEAVES ＝ 3 块，每块都是设计的 2.5 倍。
  · 正式修法：N.MU_CTL → N.MU_S ＝ 4.673 → 4.377 秒，N.MU_PCT ＝ 6.3%。
  · 线宽与字节成正比（1 份 ↔ 8 px），粗细是数据，不是装饰。

⛔ 刻意没画
  · 小核上的耗时（20.5 / 41.0 ms）、主核等了多少 —— 那是出处折叠里的数，这里只数字节。
  · 另一类按头切开的权重用「全收再切」搬的那批 —— 另一条路，画进来会混。
  · 局部修法（站 19 那一版）—— 只画正式修法，跟静态图「修好以后」同一种。
  · 不写模型、规模、开关名。正交化只画成一个方框里的字，不画迭代本身。
  · 配色跟静态图 fig-muon 对齐：BF16 动量蓝、BF16 结果绿、白搬的 FP32 红；FP32 主权重停在原处时用紫。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-muon.py MuonShip WebPages/media/topic08b-muon.mp4
   三幕各自落地的停顿时刻写进 steps/MuonShip.json（课件挂 data-pauses）。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, RoundedRectangle, Line, Polygon, Text,
                   FadeIn, FadeOut, Create, Indicate, Transform,
                   WHITE, GREY, GREY_B, BLUE, RED, GREEN, YELLOW, PURPLE_B,
                   UP, DOWN, LEFT, RIGHT)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ════════════════════════════════════════════════════════════════
# ① 数据：全部从 N 取，断言钉住
# ════════════════════════════════════════════════════════════════
MOM, PAR = N.MU_MOM_U, N.MU_PARAM_U        # 以一块的 BF16 动量为 1 份（公开课不给绝对 GiB）
assert (MOM, PAR) == (1, 2)
assert N.MU_LEAVES == 3
DESIGN = [MOM, MOM]                       # 按设计：动量过去、BF16 结果回来
BEFORE = [MOM, PAR, PAR]                  # 编出来：动量过去、主权重过去、新主权重回来
assert sum(DESIGN) == N.MU_LEAF_DESIGN == 2 and sum(BEFORE) == N.MU_LEAF_BEFORE == 5
assert (N.MU_CTL, N.MU_S) == (4.673, 4.377) and abs(N.MU_PCT - 6.3) < 0.05


def gib(v):
    """份数：整数。"""
    return "%d" % v


PX_PER_GIB = 8 / MOM                      # 线宽 ∝ 字节：1 份 ↔ 8 px，2 份 ↔ 16 px

# ════════════════════════════════════════════════════════════════
# ② 字（唯一来源；aria-label 照这里复述）
# ════════════════════════════════════════════════════════════════
TITLE = "优化器里白搬的字节：一块权重、一步"
LBOX = ("参数原来的切法", "每个设备一片")
RBOX = ("正交化要的切法", "每个设备拿整块矩阵")
LBL_MOM, LBL_PAR = "BF16 动量", "FP32 主权重"
ORTHO = "正交化（迭代五轮）"
ADD = "加到主权重上"
CAP = {
    "a1": ("按设计：只搬两趟 BF16", None),
    "a2a": ("编出来：主权重也被搬了过去", None),
    "a2b": ("在那边加完，再整块搬回来", None),
    "a2c": ("一块搬了 %s 份，是设计的 %.1f 倍" % (gib(sum(BEFORE)), N.MU_RATIO),
            "好比把整件家具抬进调漆间刷完再抬回来"),
    "a3": ("结果钉回参数自己的切法，正交化一位没动",
           "一步 %.3f → %.3f 秒（−%.1f%%）" % (N.MU_CTL, N.MU_S, N.MU_PCT)),
}
def zi(s):
    """按「字」数：汉字和全角标点算 1，ASCII（数字、字母、空格）算半个。"""
    return sum(0.5 if ord(c) < 128 else 1 for c in s)


for _a, _b in CAP.values():
    assert zi(_a) <= 22 and (_b is None or zi(_b) <= 22), (_a, _b)

COL_MOM, COL_RES, COL_PAR, COL_WASTE = BLUE, GREEN, PURPLE_B, RED

# ════════════════════════════════════════════════════════════════
# ③ 版面
# ════════════════════════════════════════════════════════════════
LX, RX = -4.75, 4.75                      # 两个框的中心 x
BOX_W, BOX_H, BOX_CY = 3.5, 3.3, -0.55
EDGE_L, EDGE_R = LX + BOX_W / 2, RX - BOX_W / 2      # 通道两端
LANE = {"out1": 0.5, "out2": -0.45, "back": -1.4}   # 三条车道
CAP_Y1, CAP_Y2 = 2.85, 2.33
PKT_W = 1.5
ADD_TOP_Y = BOX_CY + BOX_H / 2 - 0.32          # 框里顶上一行字（正交化）
ADD_BOT_Y = BOX_CY - BOX_H / 2 + 0.25          # 框里底下一行字（加到主权重上）


def tx(s, size=26, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def bar(color, v, w=PKT_W):
    """一包数据：高度 ∝ 字节。"""
    return Rectangle(width=w, height=0.11 * v / MOM * 1.6, stroke_width=0,
                     fill_color=color, fill_opacity=1.0)


def build_static():
    """第 0 帧上一直在的东西（标题、两个框、框里的两样、计数器的 0）。"""
    title = tx(TITLE, 34, WHITE, True).to_edge(UP, buff=0.3)
    g = VGroup(title)
    for cx, (h, s) in ((LX, LBOX), (RX, RBOX)):
        box = RoundedRectangle(width=BOX_W, height=BOX_H, corner_radius=0.15, stroke_color=GREY_B, stroke_width=3)
        box.move_to([cx, BOX_CY, 0])
        hd = tx(h, 28, WHITE, True).next_to(box, UP, buff=0.45)
        sb = tx("（%s）" % s, 22, GREY_B).next_to(box, UP, buff=0.08)
        g.add(box, hd, sb)
    mom = bar(COL_MOM, MOM).move_to([LX, LANE["out1"], 0])
    par = bar(COL_PAR, PAR).move_to([LX, LANE["out2"], 0])
    lm = tx(LBL_MOM, 22, COL_MOM).next_to(mom, DOWN, buff=0.1)
    lp = tx(LBL_PAR, 22, COL_PAR).next_to(par, DOWN, buff=0.1)
    g.add(mom, par, lm, lp)
    return g, mom, par


def counter_mobs(total, thin, thick, col=YELLOW):
    a = tx("本块这一步搬了 %s 份（BF16 动量 ＝ 1 份）" % gib(total), 30, col, True).move_to([0, -2.72, 0])
    b = tx("细箭头 %d 趟 · 红色粗箭头 %d 趟" % (thin, thick), 24, GREY_B).move_to([0, -3.3, 0])
    return VGroup(a, b)


def arrow(x0, x1, y, v, color):
    """线宽 ∝ 字节的箭头：一根线 ＋ 一个三角形尖（尖分开画，好让线先长出来、尖后落地）。"""
    w = v * PX_PER_GIB
    s = 1 if x1 > x0 else -1
    tip_l, tip_h = 0.16 + 0.012 * w, 0.12 + 0.012 * w
    line = Line([x0, y, 0], [x1 - s * tip_l * 0.9, y, 0], stroke_color=color, stroke_width=w)
    tip = Polygon([x1, y, 0], [x1 - s * tip_l, y + tip_h, 0], [x1 - s * tip_l, y - tip_h, 0],
                  stroke_width=0, fill_color=color, fill_opacity=1.0)
    lab = tx("%s 份" % gib(v), 24, color, True).move_to([(x0 + x1) / 2, y + 0.3, 0])
    return line, tip, lab


def dump_steps(name, marks, t):
    os.makedirs(STEPS_DIR, exist_ok=True)
    with open(os.path.join(STEPS_DIR, name + ".json"), "w") as fh:
        json.dump({"pauses": marks, "duration": round(t, 2)}, fh)


class MuonShip(Scene):
    def construct(self):
        marks = []

        def hold(t):
            marks.append(round(self.renderer.time + 0.3, 2))
            self.wait(t)

        static, mom0, par0 = build_static()
        cnt = counter_mobs(0, 0, 0)
        cap = VGroup()
        self.add(static, cnt)

        def say(key, extra=(), rt=0.35):
            nonlocal cap
            a, b = CAP[key]
            n = VGroup(tx(a, 28, WHITE).move_to([0, CAP_Y1, 0]))
            if b:
                n.add(tx(b, 26, GREY_B).move_to([0, CAP_Y2, 0]))
            self.play(FadeOut(cap), FadeIn(n), *extra, run_time=rt)
            cap = n

        def count_to(total, thin, thick, col=YELLOW):
            """计数器换数：返回一个动画，跟搬运同一拍播（包在路上，数就在涨）。"""
            return Transform(cnt, counter_mobs(total, thin, thick, col))

        def ship(pkt, x1, y, v, color, rt, count):
            """一包从现在的位置沿车道 y 搬到 x1：线跟着长、计数器跟着换，落地后尖出现。"""
            x0 = EDGE_L + 0.1 if x1 > 0 else EDGE_R - 0.1
            xe = EDGE_R - 0.1 if x1 > 0 else EDGE_L + 0.1
            line, tip, lab = arrow(x0, xe, y, v, color)
            self.play(pkt.animate.move_to([x1, y, 0]), Create(line), FadeIn(lab), count_to(*count), run_time=rt)
            self.add(tip)
            return VGroup(line, tip, lab)

        def ortho_txt():
            return tx(ORTHO, 22, YELLOW).move_to([RX, ADD_TOP_Y, 0])

        def run_design(rt):
            """按设计 / 修好：动量过去 → 正交化 → BF16 结果回来 → 左边加到主权重上。返回本幕新加的东西。"""
            shown = VGroup()
            p = mom0.copy()
            self.add(p)
            shown.add(p, ship(p, RX, LANE["out1"], MOM, COL_MOM, rt, (MOM, 1, 0)))
            o = ortho_txt()
            self.play(FadeIn(o), p.animate.set_fill(COL_RES), run_time=0.45)
            shown.add(o)
            self.play(p.animate.move_to([RX, LANE["back"], 0]), run_time=0.3)
            shown.add(ship(p, LX, LANE["back"], MOM, COL_RES, rt, (sum(DESIGN), 2, 0)))
            ad = tx(ADD, 22, COL_RES, True).move_to([LX, ADD_BOT_Y, 0])
            self.play(p.animate.move_to(par0.get_center()).set_opacity(0), FadeIn(ad),
                      Indicate(par0, color=COL_RES, scale_factor=1.08), run_time=0.6)
            self.remove(p)
            shown.add(ad)
            return shown

        def clear(shown, rt=0.4):
            self.play(FadeOut(shown), count_to(0, 0, 0), run_time=rt)
            self.remove(shown)

        # ── 第一幕：按设计
        self.wait(0.3)
        say("a1")
        shown = run_design(1.0)
        hold(1.6)

        # ── 第二幕：编出来的程序
        clear(shown)
        say("a2a")
        shown = VGroup()
        pm, pp = mom0.copy(), par0.copy()
        self.add(pm, pp)
        shown.add(pm, ship(pm, RX, LANE["out1"], MOM, COL_MOM, 0.9, (MOM, 1, 0)))
        self.play(pp.animate.set_fill(COL_WASTE), run_time=0.25)
        shown.add(pp, ship(pp, RX, LANE["out2"], PAR, COL_WASTE, 1.2, (MOM + PAR, 1, 1)))
        o = ortho_txt()
        self.play(FadeIn(o), pm.animate.set_fill(COL_RES), run_time=0.45)
        shown.add(o)
        ad = tx(ADD, 22, COL_WASTE, True).move_to([RX, ADD_BOT_Y, 0])
        say("a2b", extra=(pm.animate.move_to(pp.get_center()).set_opacity(0), FadeIn(ad),
                          Indicate(pp, color=YELLOW, scale_factor=1.08)), rt=0.7)
        self.remove(pm)
        shown.add(ad)
        self.play(pp.animate.move_to([RX, LANE["back"], 0]), run_time=0.3)
        shown.add(ship(pp, LX, LANE["back"], PAR, COL_WASTE, 1.2, (sum(BEFORE), 1, 2, RED)))
        self.play(pp.animate.move_to(par0.get_center()).set_opacity(0),
                  Indicate(par0, color=COL_WASTE, scale_factor=1.08), run_time=0.5)
        self.remove(pp)
        say("a2c", extra=(Indicate(cnt[0], color=RED, scale_factor=1.1),), rt=0.6)
        hold(2.4)

        # ── 第三幕：修好
        clear(shown)
        say("a3")
        shown = run_design(0.7)
        hold(2.2)

        # ── 收尾复位：清干净，摆回第 0 帧（计数器回 0、字幕清空）
        self.play(FadeOut(shown), FadeOut(cap), count_to(0, 0, 0), run_time=0.6)
        self.remove(shown, cap, cnt)
        self.add(counter_mobs(0, 0, 0))
        self.wait(0.6)
        dump_steps("MuonShip", marks, self.renderer.time)


if __name__ == "__main__":
    print(TITLE)
    for k, v in CAP.items():
        print(k, v)
