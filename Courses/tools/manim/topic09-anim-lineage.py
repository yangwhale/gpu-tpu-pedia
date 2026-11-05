# -*- coding: utf-8 -*-
r"""专题九「GPU→TPU 训练精度对齐」：最终 golden 是一步步改出来的（GoldenLineage）。
挂在第九节 §9.1 静态图 fig9-golden 下面。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间）
  「一截截缩」：每给 golden 加一样东西，同一根柱子（同一个 TPU 被测、同一把尺子 N）当场缩一截，
  缩掉的那截不消失，原地换成一块标了名字的「已查清」色块 —— 柱子总长不变，红色那段越来越短。
  静态图 fig9-golden 只能并排放三根各自独立的柱子，读者得自己把三根对齐、自己做减法，
  而且看不出「缩掉的部分去哪了」；拆账图 fig9-decomp 有分段，但没有「是哪一步改动把这一段揭出来的」。
  把「改一样东西 → 同一根柱子缩一截 → 缩掉的那截有了名字」串成因果，只能放在时间里。
  （「红色那段」现为蓝色：2026-11-05 按 R6 评审第 6 条改配色，见下方「配色」。）
  ⚠️ 这条理由是成立但偏弱的：一张「同一根柱子、分段着色」的静态堆叠条也能表达终态；
     动画多出来的只是「每一段是哪一步揭出来的」这个先后。

⛔ 数据从哪来
  · 三个被测值（param_norm 带符号均值，被测 ＝ v5p 生产配置、FP32 累加开关打开前那一版，1000 步，实测）：
      对 ② 生产近似 2.93e-6、对 ③ 只修一处 1.25e-6、对 ④ 精确轨迹 4.36e-7。
    出处：tools/topic09-fig-golden.py 的 STEPS（课件静态图同一份数）；本脚本里重抄一份并断言比值对上。
  · 尺子 N ＝ topic09_numbers.GPU_RERUN_PN ＝ 1.24e-7（③ 那组 GPU 重跑 5 份、两两 10 对的最大值；
    R6 评审第 1 条：原先写「GPU 生产程序重跑」，会被读成 ② 那组）。
  · 比值 23.6 / 10.1 / 3.52 当场算（2.93e-6 ÷ 1.24e-7 等），断言跟静态图的断言一致。
  · 当前版（开关打开后）对 ④：topic09_numbers.ACC[0] 的 r ＝ 3.45。
  · 色块名字：「GPU 输出头 dgrad 截断」「GPU 其余 BF16 误差」＝ topic09_numbers.DECOMP 前两项的名字。
    ⛔ 色块长度是这里的「带符号均值 ÷ N」之差（13.5N、6.6N，本课推导），不是 DECOMP 的 59%／29%
       （那是第 1000 步参数范数差的拆账，另一个量）—— 所以画面上不标百分比。

⛔ 刻意没画
  · 起点 ① 没有柱子：它不重算，跟生产配置不一样，不拿来验收（静态图同口径）。
  · loss、grad_norm 两项：只演参数范数一项。
  · 色块里没有误差棒：两次缩各是一次改动前后的差，画面只表达「这截是谁的」。
  · 不提「第一关／第二关」：关卡名留到 fig9-twolayer 再出（跟静态图 R3 的约定一致），画面只说「还不过」。
    例外：③ 卡片下一行小字「第一关的尺子取自这组重跑」（R6 评审第 1 条，跟静态图 fig9-golden 的 ③ 卡片同一句）。

🎨 配色（2026-11-05 按 R6 评审第 6 条改成跟静态图 fig9-decomp 同一套）：
  · 柱子三段：dgrad 截断＝红、GPU 其余 BF16 误差＝橙、TPU 一侧剩下的（还没查清的）＝蓝。
  · 「超过 2×N ＝ 不过」不再用整根红柱表达，改成柱尾数字旁一个红色「不过」标签（红在全课 ＝ 偏差源／不过，
    这里两个用法同色但都指「有问题」，不翻转语义）。
  · 卡片：① 暗灰、② 亮灰、③ 橙、④ 紫 —— 跟静态图 fig9-golden 的卡片同色；紫只用在 ④ 精确轨迹卡片（及「对 ④」）。
    ⚠️ ③ 卡片的橙跟「GPU 其余」色块的橙同色（静态图 fig9-golden 与 fig9-decomp 本身就是这样），
       ③ 一步揭出来的是红色那块；色块下都写了名字。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic09-anim-lineage.py GoldenLineage WebPages/media/topic09-lineage.mp4
   每一步落地的停顿时刻写进 steps/GoldenLineage.json（课件挂 data-pauses）。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, RoundedRectangle, Line, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, RED, ORANGE, PURPLE_B,
                   UP, RIGHT, linear, interpolate_color)
from manim.utils.rate_functions import smooth

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic09_numbers as T  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── ① 数据 ────────────────────────────────────────────────────────────
NOISE = T.GPU_RERUN_PN                                   # N ＝ 1.24e-7
PN_VS = {2: 2.93e-6, 3: 1.25e-6, 4: 4.36e-7}             # 被测对第 k 份 golden（实测，同 topic09-fig-golden.py STEPS）
R = {k: v / NOISE for k, v in PN_VS.items()}
assert abs(R[2] - 23.6) < 0.1 and abs(R[3] - 10.1) < 0.1 and abs(R[4] - 3.52) < 0.01, R
assert R[2] > R[3] > R[4] > 2                            # 一路缩，但最后仍超 2×N
R_CUR = T.ACC[0][3]                                      # 当前版对 ④：3.45
assert T.ACC[0][0].startswith("param_norm") and R_CUR == 3.45
CHUNK3, CHUNK4 = R[2] - R[3], R[3] - R[4]                # 两次缩掉的长度（单位 N，本课推导）
assert abs(CHUNK3 - 13.5) < 0.1 and abs(CHUNK4 - 6.6) < 0.1, (CHUNK3, CHUNK4)
NAME3, NAME4 = T.DECOMP[0][0], T.DECOMP[1][0]
assert (NAME3, NAME4) == ("输出头 dgrad 截断", "GPU 其余 BF16 误差")

# ── ② 配色（全课约定；黑底上「更亮 ＝ 更强」，所以起点用暗灰、生产 golden 用亮灰）──
C_START, C_PROD, C_FIX, C_EXACT = GREY, GREY_B, ORANGE, PURPLE_B
C_FAIL, C_TPU = RED, BLUE
C_DGRAD, C_OTHER = RED, ORANGE           # 跟 fig9-decomp 同色：dgrad 截断红、GPU 其余橙、TPU 剩下的蓝
C_REST = C_TPU
CHIP_COL = [C_START, C_PROD, C_FIX, C_EXACT]
STEP_COL = {2: C_PROD, 3: C_FIX, 4: C_EXACT}

# ── ③ 版面 ────────────────────────────────────────────────────────────
X0 = -4.9                       # 柱子起点
K = 0.41                        # 场景单位 / N
BAR_Y, BAR_H = -0.85, 0.64
assert X0 + R[2] * K < 5.6      # 最长那根 ＋ 右边的数要装得下
X2N = X0 + 2 * K
CHIP_Y, CHIP_W, CHIP_H = 1.3, 3.15, 1.05
CHIP_X = [-5.1, -1.7, 1.7, 5.1]

# ── ④ 时间轴（秒）：每一步落地后停 2 秒 ─────────────────────────────────
PAUSE = 2.0
T1A, T1B = 0.5, 1.5
T2A = T1B + PAUSE
T2B = T2A + 2.4
T3A = T2B + PAUSE
T3B = T3A + 2.5
T4A = T3B + PAUSE
T4B = T4A + 2.5
T5A = T4B + PAUSE
T5B = T5A + 1.0
T6A = T5B + PAUSE + 0.5          # 收尾：累积件淡出、第一句字幕淡回
T6B = T6A + 0.8
LANDS = [T1B, T2B, T3B, T4B, T5B]
STEP_A = {1: T1A, 2: T2A, 3: T3A, 4: T4A}


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def prog(t, a, b, ease=True):
    p = clamp01((t - a) / (b - a))
    return smooth(p) if ease else p


def fade_out_all(t):
    return 1.0 - prog(t, T6A, T6B, ease=False)


def red_len(t):
    """蓝段（还没查清、算在 TPU 一侧的）长度（单位 N）：② 长出来，③④ 各缩一截。"""
    v = R[2] * prog(t, T2A + 0.4, T2A + 2.0)
    v -= CHUNK3 * prog(t, T3A + 0.4, T3A + 2.1)
    v -= CHUNK4 * prog(t, T4A + 0.4, T4A + 2.1)
    return max(v, 0.0)


def tx(s, size=24, color=WHITE):
    return Text(s, font_size=size, color=color)


CAPS = [
    (0.0, "① 起点：框架默认配置（BF16、不重算、TF32 开）", "不拿来验收，只当归因参照"),
    (T2A, "② 生产近似：＋整层重算 ＋路由回放",
     "同一个 TPU 被测对它：参数范数差 %.1f × N，远超 2×N" % R[2]),
    (T3A, "③ 只修一处：＋输出头 dgrad 换精确",
     "同一根柱子缩到 %.1f × N；缩掉的那截查清了，是 GPU 输出头 dgrad 截断" % R[3]),
    (T4A, "④ 精确轨迹：＋BF16 权重、全程 FP32、真关 TF32",
     "再缩到 %.1f × N；缩掉的那截是 GPU 其余的 BF16 误差。仍超过 2×N，不过" % R[4]),
    (T5A, "每往精确值靠一步，缩掉的都是 GPU golden 自己的偏差",
     "剩下那截才是 TPU 一侧的，还在查"),
]


def cap_alpha(i, t):
    lo = CAPS[i][0]
    hi = CAPS[i + 1][0] if i + 1 < len(CAPS) else T6A
    # R18：原来新旧字幕在 lo±0.15 交叉淡变，同一位置叠字 → 改成旧的在 [hi−0.3, hi] 先淡出、新的在 [lo, lo+0.3] 再淡入
    a = 1.0 if i == 0 else prog(t, lo, lo + 0.3, ease=False)
    b = 1.0 - prog(t, hi - 0.3, hi, ease=False)
    v = min(a, b)
    if i == 0:
        v = max(v, prog(t, T6A, T6B, ease=False))
    return clamp01(v)


def lit(i, t):
    """第 i 张卡（1 起）亮起的程度。"""
    a = STEP_A[i]
    return prog(t, a, a + 0.5) * fade_out_all(t)


def current(i, t):
    """第 i 张卡是不是「正在说的那一份」（填一层淡色）。"""
    a = STEP_A[i]
    b = STEP_A[i + 1] if i < 4 else T5A
    return min(prog(t, a, a + 0.5), 1 - prog(t, b, b + 0.5)) * fade_out_all(t)


CHIPS = [("① 起点", "框架默认"), ("② 生产近似", "＋重算 ＋路由回放"),
         ("③ 只修一处", "＋dgrad 换精确"), ("④ 精确轨迹", "＋BF16 权重、全程 FP32")]


class GoldenLineage(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def clock():
            return tr.get_value()            # ⭐ 全片唯一读时钟的地方

        title = tx("最终 golden 是一步步改出来的：每一步只加一样东西", 32).to_edge(UP, buff=0.3)
        self.add(title)

        # ── 静态件（第 0 帧就在）：柱子起点线、2×N 竖线、底注
        base = Line([X0, BAR_Y - 0.5, 0], [X0, BAR_Y + 0.5, 0], color=GREY, stroke_width=3)
        line2n = Line([X2N, BAR_Y - 0.48, 0], [X2N, BAR_Y + 0.48, 0], color=WHITE, stroke_width=4)
        lbl2n = tx("2×N", 20, WHITE).move_to([X2N, BAR_Y - 0.7, 0])
        tpu = tx("TPU 被测", 22, C_TPU)
        tpu.move_to([X0 - 0.2 - tpu.width / 2, BAR_Y + 0.17, 0])
        foot = tx("柱长 ＝ 同一个 TPU 被测的参数范数带符号均值 ÷ N；N ＝ ③ 那组 GPU 重跑 5 份两两 10 对的最大值",
                  18, GREY_B).move_to([0, -3.45, 0])
        self.add(base, tpu, foot)

        # ── 四张卡
        def chips():
            t = clock()
            g = VGroup()
            for i in range(1, 5):
                col = CHIP_COL[i - 1]
                u = lit(i, t)
                c = current(i, t)
                g.add(RoundedRectangle(corner_radius=0.12, width=CHIP_W, height=CHIP_H,
                                       stroke_color=interpolate_color(GREY, col, u) if i > 1 else col,
                                       stroke_width=2 + 2 * c, stroke_opacity=0.35 + 0.65 * u,
                                       fill_color=col, fill_opacity=0.16 * c)
                      .move_to([CHIP_X[i - 1], CHIP_Y, 0]))
                if i < 4:
                    xa = CHIP_X[i - 1] + CHIP_W / 2 + 0.03
                    g.add(Line([xa, CHIP_Y, 0], [xa + 0.19, CHIP_Y, 0], color=GREY, stroke_width=3))
            return g

        # ── 柱子：蓝段（还没查清）＋ 两块「已查清」色块（红 ＝ dgrad 截断、橙 ＝ GPU 其余）
        def bar():
            t = clock()
            f = fade_out_all(t)
            g = VGroup()
            L = red_len(t)
            if L > 1e-4:
                hl = prog(t, T5A, T5A + 0.6)            # ⑤：剩下的蓝段描一圈白
                g.add(Rectangle(width=L * K, height=BAR_H, stroke_color=WHITE, stroke_width=5,
                                stroke_opacity=hl * f, fill_color=C_REST,
                                fill_opacity=0.95 * f).move_to([X0 + L * K / 2, BAR_Y, 0]))
            for k, lo, hi, col, a in ((3, R[3], R[2], C_DGRAD, T3A), (4, R[4], R[3], C_OTHER, T4A)):
                u = prog(t, a + 0.4, a + 2.1)
                if u <= 0:
                    continue
                left = max(L, lo)                       # 色块从蓝段末端一直铺到上一次的末端
                w = (hi - left) * K
                if w <= 1e-4:
                    continue
                fc = interpolate_color(C_REST, col, u)  # 刚缩出来时还是蓝，慢慢变成名字的颜色
                g.add(Rectangle(width=w, height=BAR_H, stroke_color=col, stroke_width=3,
                                stroke_opacity=u * f, fill_color=fc, fill_opacity=(0.95 - 0.45 * u) * f)
                      .move_to([X0 + (left + hi) * K / 2, BAR_Y, 0]))
            return g

        self.add(always_redraw(chips), always_redraw(bar))
        self.add(line2n, lbl2n)                        # 2×N 线压在柱子上面

        # ── 文字：建一次，updater 只改不透明度
        texts = []

        def put(m, alpha_fn):
            texts.append((m, alpha_fn))

        for i, (_, s1, s2) in enumerate(CAPS):
            put(tx(s1, 26, WHITE).move_to([0, 2.72, 0]), lambda t, i=i: cap_alpha(i, t))
            put(tx(s2, 21, GREY_B).move_to([0, 2.26, 0]), lambda t, i=i: cap_alpha(i, t))

        for i, (nm, add) in enumerate(CHIPS, start=1):
            col = CHIP_COL[i - 1]
            put(tx(nm, 24, col).move_to([CHIP_X[i - 1], CHIP_Y + 0.2, 0]),
                lambda t, i=i: 0.35 + 0.65 * lit(i, t))
            put(tx(add, 18, GREY_B).move_to([CHIP_X[i - 1], CHIP_Y - 0.22, 0]),
                lambda t, i=i: 0.35 + 0.65 * lit(i, t))

        # ③ 卡片下一行小字：第一关的尺子取自这组重跑（R6 评审第 1 条）
        put(tx("第一关的尺子取自这组重跑", 16, C_FIX).move_to([CHIP_X[2], CHIP_Y - CHIP_H / 2 - 0.22, 0]),
            lambda t: 0.35 + 0.65 * lit(3, t))

        # 起点那一步：轨道上一句灰字，代替柱子
        put(tx("（① 不重算，跟生产配置不一样：没有柱子）", 20, GREY).move_to([X0 + 4.6, BAR_Y, 0]),
            lambda t: prog(t, T1A, T1A + 0.5) * (1 - prog(t, T2A, T2A + 0.4)))

        # 左边第二行：此刻对的是哪一份
        for k in (2, 3, 4):
            m = tx("对 %s" % "①②③④"[k - 1], 22, STEP_COL[k])
            m.move_to([X0 - 0.2 - m.width / 2, BAR_Y - 0.2, 0])
            hi = STEP_A[k + 1] if k < 4 else None
            put(m, lambda t, k=k, hi=hi: prog(t, STEP_A[k], STEP_A[k] + 0.4)
                * (1.0 if hi is None else 1 - prog(t, hi, hi + 0.4)) * fade_out_all(t))

        # 柱尾（蓝段末端）的数 ＋ 红色「不过」标签：落地后淡入、下一次缩之前淡出
        for k in (2, 3, 4):
            num = tx("%.1f × N" % R[k], 26, C_REST)
            bad = Text("不过", font_size=24, color=C_FAIL, weight="BOLD")   # 只用字，不加框：set_opacity 会把空心框填实
            m = VGroup(num, bad).arrange(RIGHT, buff=0.14).move_to([X0 + R[k] * K, BAR_Y + 0.55, 0])
            assert R[k] > 2                             # 三次都超过 2×N：都挂「不过」
            if k == 2:
                m.set_x(X0 + R[k] * K - m.width / 2)    # 最长那根：右对齐到末端，别出框
            if k == 4:
                m.set_x(X2N + 0.12 + m.width / 2)       # 最短那根：别压到 2×N 竖线
            a_in = STEP_A[k] + 1.9
            hi = STEP_A[k + 1] if k < 4 else None
            put(m, lambda t, a_in=a_in, hi=hi: prog(t, a_in, a_in + 0.4)
                * (1.0 if hi is None else 1 - prog(t, hi + 0.1, hi + 0.4)) * fade_out_all(t))

        # 两块色块的名字（两行）
        for lo, hi, col, a, name in ((R[3], R[2], C_DGRAD, T3A, "GPU " + NAME3), (R[4], R[3], C_OTHER, T4A, NAME4)):
            cx = X0 + (lo + hi) * K / 2
            m1 = tx(name, 20, col).move_to([cx, BAR_Y - 0.62, 0])
            m2 = tx("（已查清）", 20, col).move_to([cx, BAR_Y - 0.97, 0])
            assert m1.width < (hi - lo) * K + 0.1, (name, m1.width, (hi - lo) * K)
            for m in (m1, m2):
                put(m, lambda t, a=a: prog(t, a + 1.8, a + 2.3) * fade_out_all(t))

        note = tx("被测是 FP32 累加开关打开前那一版 TPU；当前版对 ④ 是 %.2f × N" % R_CUR, 18, GREY)
        note.move_to([0, -2.55, 0])
        put(note, lambda t: prog(t, T4A + 2.0, T4A + 2.5) * fade_out_all(t))

        rest = tx("剩下这截：TPU 一侧，还在查", 22, C_TPU)
        rest.move_to([X0 + rest.width / 2 - 0.1, BAR_Y + 1.05, 0])
        put(rest, lambda t: prog(t, T5A, T5A + 0.6) * fade_out_all(t))

        for m, _ in texts:
            assert m.width < 13.6, (m.text if hasattr(m, "text") else m, m.width)

        def apply(m, af):
            m.set_opacity(af(clock()))

        for m, af in texts:
            apply(m, af)                         # ⛔ 先手动算一次：第 0 帧不能叠字
            m.add_updater(lambda m, af=af: apply(m, af))
            self.add(m)

        # ── 走时钟：全部 linear；每一步落地记一个停顿点
        marks = []
        now = 0.0
        for land in LANDS:
            self.play(tr.animate.set_value(land), run_time=land - now, rate_func=linear)
            marks.append(round(self.renderer.time + 0.3, 2))
            self.play(tr.animate.set_value(land + PAUSE), run_time=PAUSE, rate_func=linear)
            now = land + PAUSE
        self.play(tr.animate.set_value(T6B), run_time=T6B - now, rate_func=linear)
        # 复位：T6B 时刻的画面就是第 0 帧（累积件全淡出、第一句字幕淡回）；把时钟拨回 0
        self.play(tr.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)
        self.play(tr.animate.set_value(0.0), run_time=0.4, rate_func=linear)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "GoldenLineage.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
