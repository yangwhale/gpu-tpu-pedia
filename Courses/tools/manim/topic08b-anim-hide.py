# -*- coding: utf-8 -*-
r"""专题九 §1.3「师傅和小工：主核、小核，和藏住」—— HideExpose。
放在静态图 fig-hide 下面；画面里不写课件节号。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  「藏住 / 露出」说的是两条车道在时间上的相对位置：小工跑腿的同时，师傅手上有没有活。
  师傅走到要用结果的那一刻，腿跑完了就是藏住（不花步时），没跑完师傅就停下等（露出，红色）。
  静态图 fig-hide 把三条通信各自对齐到起点画成结果；只有看着时间游标走过去，
  才看得出同样是小工跑一趟腿，有时花时间、有时不花 —— 「走到那一刻才判」只能放在时间里。

⛔ 数据从哪来
  · 第一幕「泡茶」：华罗庚《统筹方法》原例（讲义 §1.3 正文同数）：洗水壶 1 分钟、烧开水 15 分钟，
    水烧着的时候洗壶、洗杯、拿茶叶共 4 分钟 → 16 分钟喝上茶；坐等水开再去洗杯拿茶叶 → 20 分钟，原文叫「窝了工」。
    条只用两条示意（烧水、洗杯拿茶叶），分钟数只写总数 16 / 20。
  · 第二幕全部 import 自 tools/topic08b_numbers.py 并断言：
    N.HIDE 三条通信（名字, 开始 ms, 时长 ms, 主核开始等的时刻 或 None）：
      专家梯度 reduce-scatter 0.0 开始 16.2 ms，主核 6.1 要用 → 露出 10.1；
      token 收发 all-gather 33.7 开始 1.6 ms，全藏住（主核没有停在它上面）；
      专家权重 all-gather 37.6 开始 8.1 ms，主核 40.1 要用 → 露出 5.6。
    露出量用 N.TC_WAITS 核对（同一时刻开始、同一时刻结束）。
    时间轴是 N.LAYER_MS ＝ 79.6 ms 这一层反向的前 47 ms；这段里主核只停了上面两次（N.TC_WAITS 里下一次从 47.8 开始），
    所以主核车道其余部分画蓝（在算）是实测，不是示意。

⚠️ 顺序：游标按真实时间从左往右走，所以三条的出场顺序是「露出 10.1 → 藏住 → 露出 5.6」，
   不是按露出多少排的（那样就得给每条各开一根时间轴，「同一个游标走过去」就没了）。
   token 收发主核没有停在它上面，profile 里没有「要用」的时刻，所以它的判定放在它跑完那一刻：
   「跑完了，师傅还在算」。不编一个要用时刻。

⛔ 刻意没画
  · 这一层里别的通信（小集合、数据重排、稠密层权重 all-gather、另一颗小核）—— fig-lanes 已画，这里只看三条。
  · 47 ms 以后的那半层底座。
  · 泡茶里洗水壶那 1 分钟没画成条（只画两条，起点空一格）；泡茶的分钟数不跟毫秒放在同一幕。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic08b-anim-hide.py HideExpose WebPages/media/topic08b-hide.mp4
   每次判定处的停顿写进 steps/HideExpose.json（课件挂 data-pauses）。

⭐ 写法：时钟式。唯一的 ValueTracker ＝ 片内时间；两幕的透明度、茶钟（分钟）、层钟（毫秒）、字幕号都是 state(u) 的纯函数。
   复位只定义在 state 的最后一段：第二幕淡出、第一幕空车道淡回，末帧按定义就是第 0 帧。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Line, DashedLine, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, RED, YELLOW, ORANGE, UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import topic08b_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── 第一幕：泡茶（华罗庚《统筹方法》原例；讲义 §1.3 正文同数）─────────────
WASH_KETTLE, BOIL, CUPS_TEA = 1, 15, 4
TEA_GOOD = WASH_KETTLE + BOIL                    # 洗杯拿茶叶塞进烧水的 15 分钟里
TEA_BAD = WASH_KETTLE + BOIL + CUPS_TEA          # 坐等水开，开了再洗杯拿茶叶
assert (TEA_GOOD, TEA_BAD) == (16, 20) and CUPS_TEA < BOIL
# 每条 (车道 0=师傅 1=小工, 开始分钟, 结束分钟, 颜色, 字)
ROW1 = [(1, WASH_KETTLE, TEA_GOOD, ORANGE, "烧水"),
        (0, WASH_KETTLE, WASH_KETTLE + CUPS_TEA, BLUE, "洗杯拿茶叶")]
ROW2 = [(1, WASH_KETTLE, TEA_GOOD, ORANGE, "烧水"),
        (0, WASH_KETTLE, TEA_GOOD, RED, "干等水开 · 窝了工"),
        (0, TEA_GOOD, TEA_BAD, BLUE, "洗杯拿茶叶")]

# ── 第二幕：实测三条通信（全从 N 取）────────────────────────────────
HIDE = {}
for name, s, d, w in N.HIDE:
    key = "egrad" if "梯度" in name else ("ew" if "权重" in name else "tok")
    HIDE[key] = (s, d, w)
assert set(HIDE) == {"egrad", "ew", "tok"}
EXPOSE = {}
for k, (s, d, w) in HIDE.items():
    if w is None:
        EXPOSE[k] = 0.0
        assert not any(s <= ws < s + d for ws, _wd, _kk in N.TC_WAITS)        # 它跑的时候主核一次也没停
    else:
        e = round(s + d - w, 1)
        assert any(abs(ws - w) < 0.05 and abs(wd - e) < 0.05 for ws, wd, _kk in N.TC_WAITS), k   # 跟 TC_WAITS 对上
        EXPOSE[k] = e
assert (HIDE["egrad"], HIDE["tok"], HIDE["ew"]) == ((0.0, 16.2, 6.1), (33.7, 1.6, None), (37.6, 8.1, 40.1))
assert (EXPOSE["egrad"], EXPOSE["ew"], EXPOSE["tok"]) == (10.1, 5.6, 0.0)
MS_END = 47.0
assert MS_END < N.LAYER_MS
WAITS = [(ws, wd) for ws, wd, _k in N.TC_WAITS if ws < MS_END]
assert WAITS == [(6.1, 10.1), (40.1, 5.6)] and all(ws + wd <= MS_END for ws, wd in WAITS)   # 窗口里只有这两次停
TOK_END = HIDE["tok"][0] + HIDE["tok"][1]
assert abs(TOK_END - 35.3) < 1e-9

# ── 时间表（秒）───────────────────────────────────────────────────
T_R1 = 0.8; GROW = 2.4
T_R1E = T_R1 + GROW            # 第一行长完（16 分钟）
T_R2 = T_R1E + 1.8
T_R2E = T_R2 + GROW            # 第二行长完（20 分钟）
T_X = T_R2E + 1.4              # 换幕
T_XE = T_X + 0.6
# 层钟：(片内开始秒, 毫秒起, 毫秒止, 走多久)；段与段之间的空隙就是停顿
J1, J2, J3 = HIDE["egrad"][2], TOK_END, HIDE["ew"][2]
HOLD = 1.5
SEGS = []
_t = T_XE + 0.3
for m0, m1, dur in [(0.0, J1, 1.2), (J1, J2, 3.4), (J2, J3, 0.8), (J3, MS_END, 1.1)]:
    SEGS.append((_t, m0, m1, dur))
    _t += dur + HOLD
T_JUDGE = [SEGS[i][0] + SEGS[i][3] for i in range(3)]         # 三次判定落地
T_FIN = SEGS[3][0] + SEGS[3][3]                                  # 层钟走完，最后一句
T_RST = T_FIN + 2.0
T_RSTE = T_RST + 0.7
T_TOTAL = T_RSTE + 0.4
PAUSES = [round(x + 0.3, 2) for x in (T_R1E, T_R2E, *T_JUDGE, T_FIN)]
assert 18 <= T_TOTAL <= 25, T_TOTAL

CAPS = [
    "华罗庚《统筹方法》：泡一壶茶",
    "水自己在烧：小工自己在跑腿",
    "师傅手上有活：这趟腿不花时间，叫藏住",
    "师傅手上没活，只能干等：叫露出",
    "换成实测：反向一层里的三条通信",
    "第 %.1f ms 师傅走到完成点，腿还没跑完：露出" % J1,
    "这趟跑完时师傅还在算：藏住",
    "第 %.1f ms 走到完成点，还差一截：露出" % J3,
    "只有露出的那一截，算进步时",
]
assert all(len(c.replace(" ", "")) <= 22 for c in CAPS)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def state(u):
    """片内时间 u → (第一幕透明度, 第二幕透明度, 第一行分钟, 第二行分钟, 层钟毫秒, 字幕号)。⭐ 复位只定义在这里。"""
    if u >= T_RST:                                   # 复位：第二幕淡出、第一幕空车道淡回、字幕回第一句
        p = clamp((u - T_RST) / (T_RSTE - T_RST))
        return p, 1.0 - p, 0.0, 0.0, MS_END, 0 if p >= 0.5 else 8
    m1 = TEA_GOOD * clamp((u - T_R1) / GROW)
    m2 = TEA_BAD * clamp((u - T_R2) / GROW)
    a1 = 1.0 - clamp((u - T_X) / (T_XE - T_X))
    a2 = 1.0 - a1
    ms = 0.0
    for t0, s0, s1, dur in SEGS:
        if u >= t0:
            ms = s0 + (s1 - s0) * clamp((u - t0) / dur)
    if u < T_R1:
        cap = 0
    elif u < T_R1E:
        cap = 1
    elif u < T_R2:
        cap = 2
    elif u < T_X:
        cap = 3
    elif u < T_JUDGE[0]:
        cap = 4
    elif u < T_JUDGE[1]:
        cap = 5
    elif u < T_JUDGE[2]:
        cap = 6
    elif u < T_FIN:
        cap = 7
    else:
        cap = 8
    return a1, a2, m1, m2, ms, cap


def _vis(u):          # 末帧和首帧只比「画面上看得见的」：第二幕透明度 0 时层钟是几都看不见
    a1, a2, m1, m2, _ms, cap = state(u)
    return a1, a2, m1, m2, cap


assert _vis(0.0) == _vis(T_TOTAL) == (1.0, 0.0, 0.0, 0.0, 0)   # 末帧 ≡ 首帧，按定义成立

# ── 画面常数 ───────────────────────────────────────────────────────
FS = 22                                   # 画面最小字号
# 第一幕
TX0, TSC = -3.6, 0.45                     # 分钟 → x
assert TX0 + TEA_BAD * TSC < 5.5


def tx_(m):
    return TX0 + m * TSC


TROW_Y = [(1.45, 0.85), (-0.75, -1.35)]  # 两行 × (师傅, 小工) 车道中心
TBH = 0.46
# 第二幕
LX0, LSC = -4.0, 9.6 / MS_END             # 毫秒 → x


def lx(ms):
    return LX0 + ms * LSC


Y_M, Y_S, LBH = 0.75, -0.75, 0.5          # 师傅 / 小工车道中心、条高
COMM = [HIDE["egrad"], HIDE["tok"], HIDE["ew"]]


def tx(s, size=FS, color=WHITE, bold=False):
    return Text(s, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")


def remember(m):
    for sm in m.family_members_with_points():
        sm._base_op = (sm.get_stroke_opacity(), sm.get_fill_opacity())
    return m


def fade_to(m, a):
    for sm in m.family_members_with_points():
        so, fo = sm._base_op
        sm.set_stroke(opacity=so * a)
        sm.set_fill(opacity=fo * a)


def bar(x0, x1, y, h, color, fill=0.45):
    w = max(x1 - x0, 0.001)
    return Rectangle(width=w, height=h, stroke_color=color, stroke_width=3 if x1 - x0 > 0.02 else 0,
                     fill_color=color, fill_opacity=fill).move_to([x0 + w / 2, y, 0])


class HideExpose(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def st():
            return state(tr.get_value())

        title = tx("一趟腿花不花时间：看师傅那时候手上有没有活", 30, WHITE, True).to_edge(UP, buff=0.3)
        self.add(title)

        # ── 第一幕静态件：车道名、车道线、条里的字、终点标 ────────────────
        act1 = VGroup()
        for r, (ym, ys) in enumerate(TROW_Y):
            act1.add(tx(["统筹着做", "坐等水开"][r], 24, YELLOW, True).move_to([-6.0, (ym + ys) / 2, 0]))
            for lab, y in (("师傅", ym), ("小工", ys)):
                act1.add(tx(lab, FS, GREY_B).next_to([TX0 - 0.15, y, 0], LEFT, buff=0.0))
                act1.add(Line([TX0, y - TBH / 2 - 0.02, 0], [tx_(TEA_BAD), y - TBH / 2 - 0.02, 0], color=GREY, stroke_width=2))
        bar_labs = []                                   # (行, 结束分钟, 文字)：条长过文字右缘才显示
        for r, row in enumerate((ROW1, ROW2)):
            for lane, m0, m1, col, s in row:
                y = TROW_Y[r][lane]
                t = remember(tx(s, FS, WHITE).move_to([(tx_(m0) + tx_(m1)) / 2, y, 0]))
                bar_labs.append((r, (t.get_right()[0] - TX0) / TSC, t))
        end_labs = []
        for r, (mins, s) in enumerate(((TEA_GOOD, "%d 分钟喝上茶" % TEA_GOOD), (TEA_BAD, "%d 分钟" % TEA_BAD))):
            ym, ys = TROW_Y[r]
            g = VGroup(Line([tx_(mins), ys - TBH / 2 - 0.08, 0], [tx_(mins), ym + TBH / 2 + 0.08, 0], color=YELLOW, stroke_width=4),
                       tx(s, 24, YELLOW, True).move_to([tx_(mins), ym + TBH / 2 + 0.3, 0]))
            end_labs.append((r, mins, remember(g)))
        remember(act1)

        # ── 第二幕静态件 ──────────────────────────────────────────
        act2 = VGroup()
        for lab, y in (("师傅（主核）", Y_M), ("小工（小核）", Y_S)):
            act2.add(tx(lab, FS, GREY_B).next_to([LX0 - 0.2, y, 0], LEFT, buff=0.0))
            act2.add(Line([LX0, y - LBH / 2 - 0.02, 0], [lx(MS_END), y - LBH / 2 - 0.02, 0], color=GREY, stroke_width=2))
        y_ax = -1.95
        act2.add(Line([LX0, y_ax, 0], [lx(MS_END), y_ax, 0], color=GREY, stroke_width=2))
        for t in range(0, 41, 10):
            act2.add(Line([lx(t), y_ax - 0.07, 0], [lx(t), y_ax + 0.07, 0], color=GREY, stroke_width=2))
            act2.add(tx(str(t), FS, GREY).move_to([lx(t), y_ax - 0.3, 0]))
        act2.add(tx("反向一层里的时间（ms，实测）", FS, GREY).next_to([lx(MS_END), y_ax - 0.72, 0], LEFT, buff=0.0))
        # 每趟腿的全长：灰框（先看得见它要跑多久）
        for s, d, _w in COMM:
            act2.add(Rectangle(width=d * LSC, height=LBH, stroke_color=GREY, stroke_width=2, fill_opacity=0)
                     .move_to([lx(s + d / 2), Y_S, 0]))
        # 名字
        act2.add(tx("专家梯度 reduce-scatter %.1f" % HIDE["egrad"][1], FS, ORANGE).next_to([lx(0), Y_S - LBH / 2 - 0.12, 0], DOWN + RIGHT, buff=0.0))
        act2.add(tx("token 收发 %.1f" % HIDE["tok"][1], FS, ORANGE).next_to([lx(HIDE["tok"][0]) - 0.08, Y_S + LBH / 2 + 0.28, 0], LEFT, buff=0.0))   # 右缘停在条左边，判定时游标不压字
        act2.add(tx("专家权重 all-gather %.1f" % HIDE["ew"][1], FS, ORANGE).next_to([lx(HIDE["ew"][0] + HIDE["ew"][1]), Y_S - LBH / 2 - 0.12, 0], DOWN + LEFT, buff=0.0))
        # 「要用」：主核要用结果的时刻（只有 profile 里有的两条）
        for w in (J1, J3):
            act2.add(DashedLine([lx(w), Y_M + LBH / 2 + 0.05, 0], [lx(w), Y_S + LBH / 2 + 0.02, 0], color=WHITE, stroke_width=2, dash_length=0.08))
            act2.add(tx("完成点", FS, WHITE).next_to([lx(w), Y_M - LBH / 2 - 0.05, 0], DOWN + RIGHT, buff=0.06))
        # token 收发：profile 里主核没停在它上面，没有「要用」时刻 —— 判定放在它跑完那一刻
        tok_mark = remember(VGroup(
            DashedLine([lx(J2), Y_S + LBH / 2 + 0.02, 0], [lx(J2), Y_M + LBH / 2 + 0.45, 0], color=BLUE, stroke_width=2, dash_length=0.08),
            tx("跑完", FS, BLUE).next_to([lx(J2), Y_M - LBH / 2 - 0.05, 0], DOWN + LEFT, buff=0.06)))
        remember(act2)
        # 三次判定的标签
        judges = [remember(tx("停着等 %.1f ms" % EXPOSE["egrad"], 24, RED, True).next_to([lx(J1), Y_M + LBH / 2 + 0.12, 0], UP + RIGHT, buff=0.0)),
                  remember(VGroup(tx("藏住 · 不花步时", 24, BLUE, True).move_to([lx(J2) - 0.4, Y_M + LBH / 2 + 0.75, 0]), tok_mark)),
                  remember(tx("停着等 %.1f ms" % EXPOSE["ew"], 24, RED, True).move_to([lx(J3 + EXPOSE["ew"] / 2), Y_M + LBH / 2 + 0.3, 0]))]

        def upd_static(_m=None):
            a1, a2, m1, m2, ms, _cap = st()
            fade_to(act1, a1)
            for r, mend, t in bar_labs:
                fade_to(t, a1 * (1.0 if (m1, m2)[r] >= mend else 0.0))
            for r, mins, g in end_labs:
                fade_to(g, a1 * (1.0 if (m1, m2)[r] >= mins - 1e-6 else 0.0))
            fade_to(act2, a2)
            for k, jm in enumerate((J1, J2, J3)):
                fade_to(judges[k], a2 * (1.0 if ms >= jm - 1e-6 else 0.0))

        static_all = VGroup(act1, act2, *[t for _r, _m, t in bar_labs], *[g for _r, _m, g in end_labs], *judges)
        upd_static()                                   # ⛔ 先手动算一次
        static_all.add_updater(upd_static)
        self.add(static_all)

        # ── 动态：长出来的条、游标 ──────────────────────────────────
        def dyn():
            a1, a2, m1, m2, ms, _cap = st()
            g = VGroup()
            if a1 > 0:
                for r, (row, m) in enumerate(((ROW1, m1), (ROW2, m2))):
                    for lane, b0, b1, col, _s in row:
                        if m > b0:
                            g.add(bar(tx_(b0), tx_(min(m, b1)), TROW_Y[r][lane], TBH, col).set_opacity(a1 * 0.45)
                                  .set_stroke(opacity=a1))
            if a2 > 0:
                # 小工：三趟腿，游标走过的部分填橙
                for s, d, _w in COMM:
                    if ms > s:
                        g.add(bar(lx(s), lx(min(ms, s + d)), Y_S, LBH, ORANGE).set_fill(opacity=a2 * 0.55).set_stroke(opacity=a2))
                # 师傅：走过的部分蓝，停着等的部分红
                cuts = [0.0]
                for ws, wd in WAITS:
                    cuts += [ws, ws + wd]
                cuts.append(MS_END)
                for i in range(len(cuts) - 1):
                    c0, c1 = cuts[i], min(cuts[i + 1], ms)
                    if c1 > c0:
                        col = RED if i % 2 == 1 else BLUE
                        g.add(bar(lx(c0), lx(c1), Y_M, LBH, col).set_fill(opacity=a2 * 0.55).set_stroke(opacity=a2))
                x = lx(ms)
                g.add(Line([x, Y_S - LBH / 2 - 0.1, 0], [x, Y_M + LBH / 2 + 0.1, 0], color=YELLOW, stroke_width=5).set_stroke(opacity=a2))
            return g

        dyn_m = always_redraw(dyn)
        self.add(dyn_m)

        # ── 字幕：建一次，按区间切透明度 ─────────────────────────────
        caps = [tx(c, 24, GREY_B).next_to(title, DOWN, buff=0.22) for c in CAPS]
        caps[8].set_color(YELLOW)

        def show_cap(m, k):
            m.set_opacity(1.0 if st()[5] == k else 0.0)
        for k, m in enumerate(caps):
            show_cap(m, k)                              # ⛔ 先手动算一次
            m.add_updater(lambda m, k=k: show_cap(m, k))
            self.add(m)

        # ── 走时钟：一整段 linear，停顿由 state 的时间表决定 ─────────────
        self.play(tr.animate.set_value(T_TOTAL), run_time=T_TOTAL, rate_func=linear)
        self.play(tr.animate.set_value(T_TOTAL), run_time=1 / 30, rate_func=linear)   # 强制刷一帧（末帧 ＝ 第 0 帧）
        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "HideExpose.json"), "w") as fh:
            json.dump({"pauses": PAUSES, "duration": round(self.renderer.time, 2)}, fh)
