# -*- coding: utf-8 -*-
r"""专题九 · MoE 路由的近平局翻转（RouteFlip）。

渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic09-anim-flip.py RouteFlip /tmp/draft/topic09-flip.mp4 --draft
   构建手册/脚本/render.sh tools/manim/topic09-anim-flip.py RouteFlip WebPages/media/topic09-flip.mp4
每一步落地的停顿时刻写进 tools/manim/steps/RouteFlip.json（课件挂 data-pauses）。

⭐ 为什么非动不可：论点是一个因果链「输入差一丝 → 分数挪一丝 → 第 8、9 名在某一刻交叉 → 选中的专家换人
   → 这个 token 从一个专家搬到另一个」。静态图能画出前后两个名次表，画不出「挪动小到主图上看不见、
   放大以后才看见两根条交叉、交叉那一刻选中的颜色跟着跳」—— 增量来自时间。

⭐ 数据从哪来（一个对齐实践：混元 3 截前 4 层，GPU B200×8 对 TPU v5p，同一份权重和数据）：
   · 路由的输入两边差 rel_L2 3.4e-3 – 3.9e-3（第 1–3 层，逐层隔离回放：每层输入取 GPU 的，看 TPU 算出的
     MLP 前归一化输出对 GPU），几乎全部来自注意力输出 —— 正好是 BF16 半格 2⁻⁸ ≈ 3.9e-3 的量级（断言）。
   · 翻转 token 的第 8、9 名分差（参照一侧的「sigmoid 分数＋偏置」）中位数 2.7e-5 – 6.3e-5，
     全部 token 的中位数 3.1e-4 – 8.0e-4。
   · 路由 GEMM 喂完全相同的输入：3 层 × 4096 个 token 翻转 0（logits 两边仍差 rel_L2 3.8e-6 – 4.0e-6）。
   · 两边全程 FP32 的前向（同一状态、同一批）：第 1 层 4096 个 token 里仍有 1 个换了专家，
     它的第 8、9 名只差 1.6e-6。（只用第 1 层：第 2、3 层的翻转是 TPU 那份「FP32」参照里
     路由专家其实按 BF16 算的下游，不能当「FP32 也挡不住」的证据。）
   · 第 1 步梯度：差最大的张量是第 3 层专家 0 的权重梯度，rel_L2 0.102；比到的 50 个张量合起来 1.28e-3。
     这个专家在一批 4096 个 token 里分到 139 个，换出 3 个、换进 1 个；估算 √(k/n) = √(4/139) ≈ 0.17（断言），
     与实测 0.10 同量级。GPU 自己重跑第 1 步前向确定、路由不变，所以 GPU 重跑的噪声对这类差是 0。
   · 排名用「分数＋偏置」，选中后的加权用不带偏置的分数（归一后乘缩放）。
   ⚠️ 示意：10 根条的高度、每个分数被挪动的量（δ）是造的；只有「初始分差取在翻转 token 分差中位数的范围里、
     δ 足以翻转近平局却翻不动典型分差」这两条用断言钉住。

🎨 配色（2026-11-05 按全课规矩改）：条的颜色表示「谁算的分数」—— GPU 那份灰、挪到 TPU 那份变蓝（跟着分数挪动同步渐变）；
   入选前 8 名与否只用填充深浅（实 / 淡）和虚线区分。原先用「蓝 ＝ 入选」，会被读成 TPU。

⭐ 刻意没画：192 个专家（只画前 10 名）；top-8 里另外 7 个专家各自的输出；路由权重怎么归一；
   反向里梯度怎么回传；第 2、3 层的翻转比例；「GPU 自己重跑也会翻」—— 依据里只有「第 1 步 GPU 重跑路由不变」，
   没有 GPU 自己翻转的证据，所以不画。
"""
import json
import math
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, Rectangle, Line, DashedLine, Circle, Text, ValueTracker, always_redraw,
                   WHITE, GREY, GREY_B, BLUE, GREEN, RED, YELLOW, UP, DOWN, LEFT, RIGHT, linear,
                   interpolate_color)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import topic09_numbers as N   # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── 新常数（实测；出处见报告，待并入 topic09_numbers）────────────────────
ROUTE_IN_RELL2 = (3.4e-3, 3.9e-3)          # 路由输入（MLP 前归一化输出）两边 rel_L2，第 1–3 层
FLIP_MARGIN_P50 = (2.7e-5, 6.3e-5)         # 翻转 token 的第 k 与 k+1 名分差中位数
ALL_MARGIN_P50 = (3.1e-4, 8.0e-4)          # 全部 token 的分差中位数
SAME_IN_LOGITS = (3.8e-6, 4.0e-6)          # 喂相同输入时两边路由 logits 的 rel_L2
SAME_IN_LAYERS, TOKENS = 3, 4096           # 喂相同输入：3 层 × 4096 个 token，翻转 0
FP32_FLIPS_L1, FP32_MARGIN = 1, 1.6e-6     # 两边全程 FP32：第 1 层翻 1 个 token，分差 1.6e-6
EXPERTS, TOPK = 192, 8
G_EXPERT, G_NET, G_NET_TENSORS = 0.102, 1.28e-3, 50   # 第 1 步梯度 rel_L2：最差专家 / 比到的 50 个张量合起来
E_TOKENS, E_OUT, E_IN = 139, 3, 1          # 第 3 层专家 0：一批 4096 个 token 里分到 139 个，换出 3、换进 1

BF16_HALF = N.BF16_GAP1 / 2
assert ROUTE_IN_RELL2[0] < ROUTE_IN_RELL2[1] <= BF16_HALF + 1e-4      # 输入差 ＝ BF16 半格量级
assert FP32_FLIPS_L1 * TOPK <= TOPK * TOKENS and TOPK * TOKENS == 32768   # 「1 / 32768 个槽位」＝ 4096 个 token 里 1 个
EST = math.sqrt((E_OUT + E_IN) / E_TOKENS)
assert abs(EST - 0.17) < 0.005, EST                                 # 估算 √(k/n) ≈ 0.17
assert 0.5 < G_EXPERT / EST < 2                                     # 与实测同量级
RATIO = G_EXPERT / G_NET
assert abs(RATIO - 80) < 1, RATIO                                   # 约 80 倍
assert FP32_MARGIN < FLIP_MARGIN_P50[0]                             # FP32 下翻的那一个，比 BF16 下的典型翻转还近

# ── 示意数据：这个 token 前 10 名的「分数＋偏置」（GPU 一侧），以及 TPU 一侧挪动的量 ──────
GAP0 = 3.0e-5                                                       # 第 8、9 名初始分差
assert FLIP_MARGIN_P50[0] <= GAP0 <= FLIP_MARGIN_P50[1]
S8 = 0.3953
S_GPU = [0.612, 0.574, 0.541, 0.508, 0.470, 0.447, 0.421, S8, S8 - GAP0, 0.352]
IA, IB = 7, 8                                                       # 专家 A（GPU 第 8 名）、专家 B（GPU 第 9 名）
DELTA = [1.8e-5, -3.1e-5, 2.2e-5, -1.2e-5, 3.4e-5, -2.0e-5, 0.9e-5, -2.5e-5, 2.0e-5, -1.6e-5]
S_TPU = [a + d for a, d in zip(S_GPU, DELTA)]


def ranks(s):
    return sorted(range(len(s)), key=lambda i: -s[i])


R_GPU, R_TPU = ranks(S_GPU), ranks(S_TPU)
assert set(R_GPU[:TOPK]) ^ set(R_TPU[:TOPK]) == {IA, IB}           # top-8 只换了 A ↔ B
assert R_GPU[:TOPK - 1] == R_TPU[:TOPK - 1] and R_GPU[TOPK:][1:] == R_TPU[TOPK:][1:]   # 其余名次不动
GAP1 = S_TPU[IB] - S_TPU[IA]
assert abs(GAP1 - 1.5e-5) < 1e-9
assert max(DELTA) - min(DELTA) < ALL_MARGIN_P50[0]                  # 同样的挪动翻不动一个典型分差的 token

# ── 时间轴（时钟 ＝ 场景时间）──────────────────────────────────────────
T_START = 0.6
SEGS = [("s1", 1.8), ("s2", 2.4), ("s3a", 1.6), ("s3b", 1.6), ("s4", 1.4), ("s5", 1.2)]
HOLD = 3.0
T = {}
_c = T_START
for _n, _d in SEGS:
    T[_n] = _c
    T[_n + "_land"] = _c + _d
    _c += _d + HOLD
T_END = _c
PAUSES = [round(T[n + "_land"] + 0.3, 2) for n, _ in SEGS]
assert 10 <= T_END + 0.6 <= 40, T_END

# ── 版面 ──────────────────────────────────────────────────────────
BASE_Y, HSCALE = -2.15, 6.2          # 主图基线；分数 → 高度
BAR_W, BAR_DX, BAR_X0 = 0.38, 0.54, -6.15
CUT_X = BAR_X0 + BAR_DX * (TOPK - 0.5)
RX = 3.55                            # 右半边面板的中线
ZX, ZY, ZW, ZH = RX, -1.05, 3.6, 2.5 # 放大框
Z_LO, Z_HI = S8 - 1.25e-4, S8 + 0.75e-4
ZOOM = (ZH / (Z_HI - Z_LO)) / HSCALE
assert ZOOM > 1000


def clip01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def ease(x):
    x = clip01(x)
    return x * x * (3 - 2 * x)


def win(t, a, b, f=0.3):
    """[a, b) 区间内可见，两端各 f 秒淡入淡出。"""
    if t < a or t >= b:
        return 0.0
    return min(clip01((t - a) / f), clip01((b - t) / f))


def tx(s, size=22, color=WHITE):
    return Text(s, font_size=size, color=color)


class RouteFlip(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def clock():
            return tr.get_value()

        def grow(t):
            return ease((t - T["s1"]) / 1.0)

        def shift(t):                   # 0 → 1：GPU 分数 → TPU 分数
            return ease((t - T["s2"] - 0.5) / 1.3)

        def swap(t):                    # 名次对调后，主图上两根条换位置
            return ease((t - T["s2"] - 1.8) / 0.6)

        def side_col(t):                # 全课配色：GPU 灰、TPU 蓝 —— 条的颜色跟着「这是谁算的分数」走
            return interpolate_color(GREY_B, BLUE, clip01(shift(t)))

        def scores(t):
            p = shift(t)
            return [a + p * d for a, d in zip(S_GPU, DELTA)]

        # ── 静态件 ──
        title = tx("MoE 路由的近平局翻转：差一丝，就换一个专家", 30).to_edge(UP)
        axis = Line([BAR_X0 - 0.4, BASE_Y, 0], [BAR_X0 + BAR_DX * 9.6, BASE_Y, 0], stroke_color=GREY)
        self.add(title, axis)

        # ── 主图：前 10 名的条 ──
        def bars():
            t = clock()
            g, s, p = grow(t), scores(t), swap(t)
            r_now = ranks(s)
            sel = set(r_now[:TOPK])
            out = VGroup()
            if g <= 0:                  # 第 0 帧：还没长出来，不留细线
                return out
            for i in range(10):
                x = BAR_X0 + BAR_DX * ((1 - p) * R_GPU.index(i) + p * R_TPU.index(i))
                h = max(1e-3, s[i] * HSCALE * g)
                on = i in sel
                col = side_col(t)                   # GPU 灰 → TPU 蓝；入选与否只看填充深浅
                r = Rectangle(width=BAR_W, height=h, fill_color=col, fill_opacity=0.85 if on else 0.25,
                              stroke_color=YELLOW if i in (IA, IB) else col,
                              stroke_width=4 if i in (IA, IB) else 2)
                r.move_to([x, BASE_Y + h / 2, 0])
                out.add(r)
            return out
        bars_m = always_redraw(bars)
        self.add(bars_m)

        # 主图的字（建一次，按时间切透明度）
        def follow_label(m, i, dy):
            def upd(mm):
                t = clock()
                p = swap(t)
                x = BAR_X0 + BAR_DX * ((1 - p) * R_GPU.index(i) + p * R_TPU.index(i))
                mm.move_to([x, BASE_Y + dy, 0])
                mm.set_opacity(clip01((t - T["s1"] - 0.6) / 0.4) if t >= T["s1"] else 0)
            upd(m)
            m.add_updater(upd)
            return m

        lab_a = follow_label(tx("A", 22, YELLOW), IA, -0.3)
        lab_b = follow_label(tx("B", 22, YELLOW), IB, -0.3)
        ranks_txt = VGroup(*[tx(str(k + 1), 18, GREY_B).move_to([BAR_X0 + BAR_DX * k, BASE_Y - 0.62, 0])
                             for k in range(10)])
        cut = DashedLine([CUT_X, BASE_Y - 0.15, 0], [CUT_X, BASE_Y + 4.1, 0], stroke_color=GREY_B)
        cut_l = tx("前 8 名入选｜落选", 18, GREY_B).move_to([CUT_X, BASE_Y + 4.33, 0])
        chart_head = tx("这个 token 的前 10 名（共 192 个专家）", 20, GREY_B).move_to([-3.35, BASE_Y - 1.05, 0])
        side_gpu = tx("GPU 算出的分数＋偏置", 22, WHITE).move_to([-4.75, 1.95, 0])
        side_tpu = tx("TPU 算出的分数＋偏置", 22, WHITE).move_to([-4.75, 1.95, 0])
        note_bias = tx("排名用「分数＋偏置」；选中后加权用不带偏置的分数 —— 偏置只管选谁", 18, GREY_B
                       ).move_to([0, -3.75, 0])

        # ── 右半边：② 输入差 ＋ 放大框 ──
        in1 = tx("路由的输入（注意力之后）两边差 rel_L2", 20, GREY_B).move_to([RX, 2.05, 0])
        in2 = tx("3.4e-3 – 3.9e-3 ≈ BF16 舍入量级", 22, RED).move_to([RX, 1.62, 0])
        zframe = Rectangle(width=ZW, height=ZH, stroke_color=GREY_B, stroke_width=2).move_to([ZX, ZY, 0])
        zhead = tx("放大看两根条的顶端（约 %d 倍）" % (round(ZOOM, -2)), 18, GREY_B
                   ).next_to(zframe, UP, buff=0.12)

        def zbars():
            t = clock()
            s = scores(t)
            sel = set(ranks(s)[:TOPK])
            vis = win(t, T["s1"] + 0.9, T["s3a"] + 0.3)
            out = VGroup()
            for j, i in enumerate((IA, IB)):
                y_top = ZY - ZH / 2 + (s[i] - Z_LO) / (Z_HI - Z_LO) * ZH
                h = max(1e-3, y_top - (ZY - ZH / 2))
                on = i in sel
                col = side_col(t)
                r = Rectangle(width=0.9, height=h, fill_color=col, fill_opacity=(0.85 if on else 0.25) * vis,
                              stroke_color=YELLOW, stroke_width=4, stroke_opacity=vis)
                r.move_to([ZX - 0.75 + 1.5 * j, ZY - ZH / 2 + h / 2, 0])
                out.add(r)
            return out
        zbars_m = always_redraw(zbars)
        self.add(zbars_m)
        za = tx("A", 22, YELLOW).move_to([ZX - 0.75, ZY - ZH / 2 - 0.25, 0])
        zb = tx("B", 22, YELLOW).move_to([ZX + 0.75, ZY - ZH / 2 - 0.25, 0])
        zgap0 = tx("B 只比 A 低 3e-5", 20, WHITE).move_to([ZX, ZY + ZH / 2 - 0.3, 0])
        zgap1 = tx("A 反比 B 低 1.5e-5：A 落选，B 入选", 20, YELLOW).move_to([ZX, ZY + ZH / 2 - 0.3, 0])
        zfoot1 = tx("翻转 token 的第 8、9 名分差：中位数 2.7e-5 – 6.3e-5", 18, GREY_B
                    ).move_to([RX, ZY - ZH / 2 - 0.65, 0])
        zfoot2 = tx("全部 token：中位数 3.1e-4 – 8.0e-4", 18, GREY_B).move_to([RX, ZY - ZH / 2 - 1.0, 0])

        # ── ③a 两个专家框，token 从 A 搬到 B ──
        EY = 0.3
        boxA = Rectangle(width=2.2, height=1.6, stroke_color=GREY_B, stroke_width=3).move_to([RX - 1.55, EY, 0])
        boxB = Rectangle(width=2.2, height=1.6, stroke_color=GREY_B, stroke_width=3).move_to([RX + 1.55, EY, 0])
        nA = tx("专家 A", 24, YELLOW).move_to(boxA.get_top() + 0.3 * DOWN + 0.0 * UP).shift(0.0 * UP)
        nB = tx("专家 B", 24, YELLOW).move_to(boxB.get_top() + 0.3 * DOWN)
        nA.move_to([RX - 1.55, EY + 1.1, 0])
        nB.move_to([RX + 1.55, EY + 1.1, 0])
        e1 = tx("GPU：这个 token 的这一份由 A 算", 20, GREY_B).move_to([RX, EY - 1.3, 0])
        e2 = tx("TPU：整块换成 B 算 —— 不是差一丝，是换了一个人", 20, WHITE).move_to([RX, EY - 1.7, 0])

        def token_dot():
            t = clock()
            vis = win(t, T["s3a"], T["s3b"])
            p = ease((t - T["s3a"] - 0.3) / 1.2)
            x = (RX - 1.55) * (1 - p) + (RX + 1.55) * p
            y = EY - 0.15 + 0.9 * math.sin(math.pi * p)
            c = Circle(radius=0.17, fill_color=YELLOW, fill_opacity=vis, stroke_width=0).move_to([x, y, 0])
            return c
        dot_m = always_redraw(token_dot)
        self.add(dot_m)

        # ── ③b 少 token 的专家：梯度的一大块 ──
        b1 = tx("第 3 层专家 0：这一批 4096 个 token 里只分到 139 个", 20, GREY_B).move_to([RX, 2.05, 0])
        b2 = tx("TPU 上换出 3 个、换进 1 个", 22, WHITE).move_to([RX, 1.62, 0])
        b3 = tx("第 1 步梯度对 GPU 的 rel_L2", 20, GREY_B).move_to([RX, 0.95, 0])
        GB_X0, GB_LEN = RX - 2.6, 4.6
        gl1 = tx("这个专家的权重梯度  0.102", 20, YELLOW).move_to([RX, 0.45, 0])
        gl2 = tx("比到的 50 个张量合起来  1.28e-3（约差 %d 倍）" % round(RATIO), 20, GREY_B).move_to([RX, -0.55, 0])
        b4 = tx("估算 √(换掉的 / 分到的) ＝ √(4/139) ≈ %.2f，同量级" % EST, 18, GREY_B).move_to([RX, -1.35, 0])
        b5 = tx("GPU 自己重跑：这些张量的差几乎是 0", 18, GREY_B).move_to([RX, -1.75, 0])

        def gbars():
            t = clock()
            vis = win(t, T["s3b"], T["s4"])
            p = ease((t - T["s3b"] - 0.3) / 1.0)
            out = VGroup()
            for v, y, col in ((G_EXPERT, 0.05, YELLOW), (G_NET, -0.95, GREY_B)):
                w = max(1e-3, GB_LEN * v / G_EXPERT * p)
                out.add(Rectangle(width=w, height=0.26, fill_color=col, fill_opacity=0.85 * vis,
                                  stroke_width=0).move_to([GB_X0 + w / 2, y, 0]))
            return out
        gbars_m = always_redraw(gbars)
        self.add(gbars_m)

        # ── ④ 对照 ──
        c1 = tx("两边喂完全相同的路由输入", 22, WHITE).move_to([RX, 1.85, 0])
        c2 = tx("3 层 × 4096 个 token：翻转 0", 24, GREEN).move_to([RX, 1.35, 0])
        c3 = tx("（logits 两边仍差 4e-6，但一个都没翻）", 18, GREY_B).move_to([RX, 0.95, 0])
        c4 = tx("两边全程 FP32（路由也是）", 22, WHITE).move_to([RX, 0.05, 0])
        c5 = tx("第 1 层 4096 个 token 里照样翻 1 个", 24, RED).move_to([RX, -0.45, 0])
        c6 = tx("它的第 8、9 名只差 1.6e-6", 18, GREY_B).move_to([RX, -0.85, 0])
        c7 = tx("根子在输入差，不在路由器自己的精度", 22, YELLOW).move_to([RX, -1.75, 0])

        # ── ⑤ 收 ──
        f1 = tx("GPU 自己重跑第 1 步：前向确定，路由一个不变", 20, GREY_B).move_to([RX, 1.2, 0])
        f2 = tx("拿 GPU 重跑当噪声尺，", 22, WHITE).move_to([RX, 0.55, 0])
        f3 = tx("路由翻转在这把尺上是 0 宽度", 22, WHITE).move_to([RX, 0.1, 0])
        f4 = tx("0.102 要按专家拆开看，不是 TPU 算错了梯度", 20, YELLOW).move_to([RX, -0.65, 0])

        # ── 字幕 ──
        caps = [
            (T["s1"], "① 一个 token 给 192 个专家打分，取前 8 名：第 8、9 名几乎一样高", GREY_B),
            (T["s2"], "② 上游注意力两边差一点点 → 分数挪一丝 → 第 8、9 名对调", GREY_B),
            (T["s3a"], "③ 这个 token 的那一份输出，整块换了一个专家来算", GREY_B),
            (T["s3b"], "③ 分到 token 少的专家，进出一个 token 就是它梯度的一大块", GREY_B),
            (T["s4"], "④ 对照：同样的输入不翻；全换 FP32 照样翻", GREY_B),
            (T["s5"], "近平局 ＋ 一丝输入差 ＝ 选了另一个专家", YELLOW),
        ]
        END = T_END + 10
        cap_items = []
        for k, (a, s, col) in enumerate(caps):
            b = caps[k + 1][0] if k + 1 < len(caps) else END
            cap_items.append((tx(s, 26, col).next_to(title, DOWN, buff=0.22), a, b))

        S1, S2, S3A, S3B, S4, S5 = (T[k] for k in ("s1", "s2", "s3a", "s3b", "s4", "s5"))
        timed = cap_items + [
            (ranks_txt, S1 + 0.6, END), (cut, S1 + 0.8, END), (cut_l, S1 + 0.8, END), (chart_head, S1 + 0.6, END),
            (side_gpu, S1 + 0.6, S2 + 1.2), (side_tpu, S2 + 1.2, END),
            (note_bias, S1 + 1.2, S3A),
            (in1, S2, S3A), (in2, S2, S3A),
            (zframe, S1 + 0.9, S3A), (zhead, S1 + 0.9, S3A), (za, S1 + 0.9, S3A), (zb, S1 + 0.9, S3A),
            (zgap0, S1 + 1.2, S2 + 1.3), (zgap1, S2 + 1.5, S3A),
            (zfoot1, S2 + 2.0, S3A), (zfoot2, S2 + 2.0, S3A),
            (boxA, S3A, S3B), (boxB, S3A, S3B), (nA, S3A, S3B), (nB, S3A, S3B),
            (e1, S3A, S3B), (e2, S3A + 1.2, S3B),
            (b1, S3B, S4), (b2, S3B + 0.3, S4), (b3, S3B + 0.5, S4), (gl1, S3B + 0.8, S4), (gl2, S3B + 0.8, S4),
            (b4, S3B + 1.2, S4), (b5, S3B + 1.2, S4),
            (c1, S4, S5), (c2, S4, S5), (c3, S4, S5), (c4, S4 + 0.5, S5), (c5, S4 + 0.5, S5), (c6, S4 + 0.5, S5),
            (c7, S4 + 1.0, S5),
            (f1, S5, END), (f2, S5 + 0.4, END), (f3, S5 + 0.4, END), (f4, S5 + 0.8, END),
        ]

        def show(m, a, b):
            o = win(clock(), a, b)
            if isinstance(m, (Text, VGroup)):
                m.set_opacity(o)
            else:                        # 线框：只动描边，别把填充也设上（黑底会被填白）
                m.set_stroke(opacity=o)
        for m, a, b in timed:
            show(m, a, b)               # ⛔ 先手动算一次：updater 要等第一次 play 才跑
            m.add_updater(lambda mm, a=a, b=b: show(mm, a, b))
            self.add(m)
        self.add(lab_a, lab_b)

        # ── 走时钟（必须 linear）──
        self.play(tr.animate.set_value(T_START), run_time=T_START, rate_func=linear)
        marks = []
        for n, _d in SEGS:
            self.play(tr.animate.set_value(T[n + "_land"]), run_time=T[n + "_land"] - tr.get_value(),
                      rate_func=linear)
            marks.append(round(self.renderer.time + 0.3, 2))
            nxt = T[n + "_land"] + HOLD
            self.play(tr.animate.set_value(nxt), run_time=HOLD, rate_func=linear)
        assert abs(tr.get_value() - T_END) < 1e-6
        assert all(abs(m - p) < 0.15 for m, p in zip(marks, PAUSES)), (marks, PAUSES)   # 草稿 15 fps 每个 play 取整到帧，会漂几十毫秒

        # ── 复位：时钟拨回 0，画面和字幕都回到第 0 帧 ──
        self.play(tr.animate.set_value(0.0), run_time=1 / 30, rate_func=linear)
        self.wait(0.5)

        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "RouteFlip.json"), "w") as fh:
            json.dump({"pauses": marks, "duration": round(self.renderer.time, 2)}, fh)
