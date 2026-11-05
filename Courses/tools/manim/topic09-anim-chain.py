# -*- coding: utf-8 -*-
r"""专题九「GPU→TPU 训练精度对齐」§8.6 串起来：五环一条链 —— DriftChain。
放在静态图 fig9-chain 下面；画面里不写课件节号。

⭐ 为什么非动不可（构建手册 10 · 第一节：增量只能来自时间或第三维）
  偏移是「随训练步数一点点长出来」的：①② 每一步都只差一点、而且每一步都一样，
  ③④⑤ 是这一点点按步数积起来的结果。静态图 fig9-chain 画得出因果链，
  画不出「同一个小差、一步一步积成看得见的漂移」—— 这个「积」只有放在时间里才看得见。

⛔ 数据从哪来（全部 import 自 tools/topic09_numbers.py，不另抄）
  · ① HEAD_BF16_S1 / HEAD_BF16_S100：GPU 输出层 dgrad 沿 w̄ 的相对误差，第 1 / 100 步 −6.51e-4 / −6.10e-4（实测）。
  · ② SWAP 第三行：同一状态（GPU 第 99 步）、同一批数据，GPU 生产程序沿漂移方向每步少走 0.36%（实测，全体参数）。
    「大头在第 0 层注意力输出投影」出自 fig9-chain 的出处栏（实测，换权重重算）。
  · ③ LSE_PROBE：固定探针 lse 差（TPU − GPU），第 25/50/100/300 步四个实测点；灰点 ＝ 第三列 GPU 两次之间。
    第 0 步两边是同一份权重，差记为 0（本课推导，不是测出来的点）。
  · ④ ROWS_300 第一行：没出现过的词，第 300 步输出层平方和 TPU − GPU ＋17.1（实测）。
    这一环原先（fig9-chain 初版）标「推断」，2026-11-05 改为实测：第 300 步同一步测量，没出现过的词那几行
    两侧 |g| 均方根几乎一样（TPU/GPU 0.990），92% 的元素 √v̂ < ε，Σw·u 的差 −4994、其中 m̂ 那份占 −5450。
    这几个数暂在本脚本里定义（RARE_*），待并入 topic09_numbers.py。
  · ⑤ topic09_series.json 的 pnorm_diff_tpu_minus_gpu_a / pnorm_diff_gpu_b_minus_gpu_a：参数范数差逐步实测值（第 1–300 步）；
    「按组拆：输出层占 87%」＝ GROUP_HEAD_300 / GROUP_TOTAL_300 ＝ 3.83e-3 / 4.39e-3（本课推导，按实测拆账算）。
    loss 偏低「一起出现、一起消失」是干预实测；它为什么跟着变，机制未验证。

⚠️ 中间帧是插值示意：
  · ③ 只有 4 个实测点，点与点之间连直线（时钟走到哪，线就画到哪）；数字只在走过实测点时更新，不显示插出来的数。
  · ④ 只有第 300 步一个实测值，柱子从 0 线性长到 ＋17.1 是示意；数字只在第 300 步出现。
  · ⑤ 是逐步实测，不插值。
  · ①② 是「每一步」的量，不随步数累积，所以不画增长。

⛔ 刻意没画
  · 1000 步全程（只走到第 300 步：③④⑤ 三环在第 300 步都有实测值，再往后 lse 只有训练批口径、不是同一个探针）。
  · 第五环里 loss 的逐步曲线（第 300 步以前 loss 差还淹在噪声里，画了只是一条抖线）。
  · w̄ 的几何（§8.1 的静态图已画）、双向干预的柱子（§8.2 已画）。

🎨 配色（2026-11-05 按全课规矩改）：TPU − GPU 的差一律蓝（③ 折线、④ 柱、⑤ 曲线和它们的数字），GPU 两次之间灰；
   ① 偏差来源红、② 中性橙、⑤ 盒子中性白；「实测」标签灰、探针刻度白 —— 紫留给精确轨迹、绿留给「通过」，这里都不出现。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic09-anim-chain.py DriftChain WebPages/media/topic09-chain.mp4
   每一步落地的停顿时刻写进 steps/DriftChain.json（课件挂 data-pauses）。

⭐ 写法：时钟式。唯一的 ValueTracker ＝ 片内时间；所有东西（五环亮灭、步数、字幕）都是 state(u) 的纯函数。
   复位定义在一个地方（state 里的倒放段）：步数倒着走回 0、五环一起暗回去 —— 末帧按定义就是第 0 帧。
"""
import json
import os
import sys

import manim as _mn; _mn.Text.set_default(font="Noto Sans CJK SC"); _mn.MarkupText.set_default(font="Noto Sans CJK SC")   # 统一字体
from manim import (Scene, VGroup, VMobject, Rectangle, RoundedRectangle, Line, Dot, Triangle, Text, ValueTracker,
                   always_redraw, WHITE, GREY, GREY_B, BLUE, RED, YELLOW, ORANGE,
                   UP, DOWN, LEFT, RIGHT, linear)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import topic09_numbers as N  # noqa: E402

STEPS_DIR = os.path.join(HERE, "steps")

# ── 数据（当场取、当场断言）──────────────────────────────────────────
HEAD1, HEAD100 = abs(N.HEAD_BF16_S1) * 100, abs(N.HEAD_BF16_S100) * 100     # → 百分数
assert "%.3f" % HEAD1 == "0.065" and "%.3f" % HEAD100 == "0.061"
assert N.HEAD_BF16_S1 < 0 and N.HEAD_BF16_S100 < 0                           # 少推，不是多推
SWAP_GPU = [v for lab, v, _ in N.SWAP if lab == "GPU 生产程序"][0]
assert SWAP_GPU < 0 and "%.2f" % abs(SWAP_GPU) == "0.36"
S_END = 300
LSE_PTS = [(0, 0.0, 0.0)] + list(N.LSE_PROBE)                                # (步, TPU−GPU, GPU 两次之间)
assert [p[0] for p in LSE_PTS] == [0, 25, 50, 100, 300]
assert all(LSE_PTS[i][1] > LSE_PTS[i + 1][1] for i in range(4))             # 单调往下长：论点是「积起来」
assert all(abs(g) < abs(v) for _, v, g in N.LSE_PROBE[1:])                   # 第 50 步起 GPU 自己小得多
RARE_LAB, RARE_300 = N.ROWS_300[0]
# ④ 改实测的依据（第 300 步，没出现过的词那几行；实测；待并入 topic09_numbers.py）
RARE_G_RMS_RATIO = 0.990        # 两侧 |g| 均方根之比 TPU/GPU
RARE_SUBEPS_SHARE = 0.92        # √v̂ < ε 的元素占比
# 画面写 √v 不写 √v̂：Pango 把组合音标 ̂ 排得跟 v 分家；第 300 步 β2=0.95 的偏差校正 1−0.95³⁰⁰ ≈ 1，v̂ 与 v 相同
assert 1 - N.BETA2 ** 300 > 0.99999
RARE_WU_DIFF, RARE_WU_MHAT = -4994, -5450   # Σw·u 的差，其中 m̂ 那一份
assert abs(RARE_G_RMS_RATIO - 1) < 0.02                       # 梯度大小两边几乎一样 → 差不在梯度大小
assert abs(RARE_WU_MHAT) > abs(RARE_WU_DIFF) * 0.9           # 差几乎全由 m̂（更新方向）那份带来
assert RARE_LAB == "没出现过" and RARE_300 == 17.1
SER = N.series()
PN = [0.0] + SER["pnorm_diff_tpu_minus_gpu_a"][:S_END]                      # 下标 ＝ 步号；第 0 步记 0
PN_G = [0.0] + SER["pnorm_diff_gpu_b_minus_gpu_a"][:S_END]
assert len(PN) == S_END + 1
PN_END = PN[S_END]
assert "%.1e" % PN_END == "4.6e-03"                                         # 与 fig9-drift 的「第 300 步 ＋4.6e-3」一致
HEAD_SHARE = N.GROUP_HEAD_300 / N.GROUP_TOTAL_300
assert round(HEAD_SHARE * 100) == 87
assert max(abs(v) for v in PN_G) < 0.15 * PN_END                            # 灰线贴着零

# ── 时间表（秒）───────────────────────────────────────────────────
T_START = 0.8          # 第 0 帧停一下
RAMP, HOLD = 0.6, 2.0  # 每环亮起 0.6 秒、停 2 秒
L = [T_START + i * (RAMP + HOLD) for i in range(5)]   # 第 i 环开始亮
T_CLK0 = L[4] + RAMP + HOLD                           # 时钟开始走
T_CLK = 10.0                                          # 0 → 300 步
T_CLK1 = T_CLK0 + T_CLK
T_END_HOLD = 2.6
T_RW0 = T_CLK1 + T_END_HOLD                           # 倒放开始
T_RW = 1.6
T_TOTAL = T_RW0 + T_RW + 0.6
PAUSES = [round(L[i] + RAMP + 0.3, 2) for i in range(5)] + [round(T_CLK1 + 0.3, 2)]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def state(u):
    """片内时间 u → (五环亮度, 步数, 字幕号)。⭐ 复位只定义在这里。"""
    if u >= T_RW0:                                    # 倒放：步数退回 0，五环一起暗回去，字幕回到第一句
        p = clamp((u - T_RW0) / T_RW)
        return [1.0 - p] * 5, S_END * (1.0 - p), 0
    lit = [clamp((u - L[i]) / RAMP) for i in range(5)]
    s = S_END * clamp((u - T_CLK0) / T_CLK)
    if u < L[0]:
        cap = 0
    elif u < T_CLK0:
        cap = 1 + max(i for i in range(5) if u >= L[i])
    elif u < T_CLK1:
        cap = 6
    else:
        cap = 7
    return lit, s, cap


assert state(0.0) == ([0.0] * 5, 0.0, 0) and state(T_TOTAL) == ([0.0] * 5, 0.0, 0)   # 末帧 ≡ 首帧，按定义成立


def lse_at(s):
    """③：实测点之间连直线（插值示意）。"""
    for (s0, v0, _), (s1, v1, _) in zip(LSE_PTS, LSE_PTS[1:]):
        if s <= s1:
            return v0 + (v1 - v0) * (s - s0) / (s1 - s0)
    return LSE_PTS[-1][1]


# ── 画面常数 ───────────────────────────────────────────────────────
# 全课配色：TPU＝蓝、GPU＝灰、精确轨迹＝紫、红＝偏差来源、绿＝通过。
# ① 是偏差来源（红）；② 中性橙；③④ 画的是 TPU − GPU 的差（蓝）；⑤ 盒子中性白，里面的 TPU − GPU 曲线蓝。
# 这支片子里没有精确轨迹、也没有「通过」，所以不用紫、不用绿。
COLS = [RED, ORANGE, BLUE, BLUE, WHITE]
COL_DIFF = BLUE                          # TPU − GPU 的差
COL_PROBE = WHITE                        # 时钟上探针实测的那几步
XC = [-5.6 + 2.8 * i for i in range(5)]
BW, BH, BY = 2.55, 0.95, 2.05            # 环的盒子
TITLES = ["① GPU 输出层 dgrad", "② 主干权重", "③ logits 整体水平", "④ 冷门词那几行", "⑤ 看到的现象"]
TAGS = ["实测"] * 5
CH_TOP, CH_BOT, CH_W = 0.75, -1.15, 2.2  # ③⑤ 小图的竖向范围、宽度
CLK_Y, CLK_X0, CLK_X1 = -2.75, -5.2, 5.6


def tx(s, size=18, color=WHITE, bold=False):
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


def sx(i, s):            # 小图横坐标：步数 0..300 线性
    return XC[i] - CH_W / 2 + CH_W * s / S_END


def poly(pts, color, width=4):
    m = VMobject(stroke_color=color, stroke_width=width)
    if len(pts) >= 2:
        m.set_points_as_corners([[x, y, 0] for x, y in pts])
    else:
        m.set_points_as_corners([[0, 0, 0], [0, 0, 0]]).set_stroke(opacity=0)
    return m


CAPS = [
    "五环一条链：每一环是下一环的原因；底下的时钟是训练步数",
    "① GPU 输出层 dgrad：12 万长的求和朝零截掉一点，每一步少推 0.065%",
    "② 主干权重沿这个方向每一步少走一点",
    "③ 所有 logits 一起偏一截（lse 差）；softmax 看不见它",
    "④ 冷门词那几行在 Adam 的 ε 区（%d%% 的元素 √v < ε），压多压少跟着 logit 水平走" % round(RARE_SUBEPS_SHARE * 100),
    "⑤ 参数范数多涨；loss 偏低一起出现、一起消失（机制未验证）",
    "时钟走起来：①② 每一步都一样，③④⑤ 一步步积起来",
    "第 300 步：每步只差一点、同一个方向，积成看得见的漂移",
]


class DriftChain(Scene):
    def construct(self):
        tr = ValueTracker(0.0)

        def st():
            return state(tr.get_value())

        title = tx("五环一条链：每一步少推一点，几百步积出来", 30).to_edge(UP, buff=0.3)
        self.add(title)

        # ── 五环：暗版常驻，亮版按亮度叠上去 ──────────────────────────
        dim_grp, lit_grp, panels = [], [], []
        for i in range(5):
            box = RoundedRectangle(width=BW, height=BH, corner_radius=0.12, stroke_color=GREY, stroke_width=3).move_to([XC[i], BY, 0])
            dim_grp.append(remember(VGroup(box, tx(TITLES[i], 17, GREY, True).move_to([XC[i], BY + 0.17, 0]))))
            tc = GREY_B                          # 「实测」标签：中性灰（绿在全课是「通过」）
            g = VGroup(RoundedRectangle(width=BW, height=BH, corner_radius=0.12, stroke_color=COLS[i], stroke_width=4,
                                        fill_color=COLS[i], fill_opacity=0.12).move_to([XC[i], BY, 0]),
                       tx(TITLES[i], 17, COLS[i], True).move_to([XC[i], BY + 0.17, 0]))
            tag = tx(TAGS[i], 15, tc, True).move_to([XC[i], BY - 0.22, 0])
            g.add(RoundedRectangle(width=tag.width + 0.22, height=0.32, corner_radius=0.1, stroke_color=tc, stroke_width=2)
                  .move_to(tag), tag)
            lit_grp.append(remember(g))
        arrows = VGroup(*[Line([XC[i] + BW / 2 + 0.03, BY, 0], [XC[i + 1] - BW / 2 - 0.03, BY, 0], color=GREY, stroke_width=3)
                          .add_tip(tip_length=0.13, tip_width=0.13) for i in range(4)])
        self.add(arrows)

        # 静态面板（只改透明度）
        def col(i, rows, y0=1.2, gap=0.36):
            g = VGroup()
            for k, (s, size, c, b) in enumerate(rows):
                g.add(tx(s, size, c, b).move_to([XC[i], y0 - k * gap if not isinstance(y0, list) else y0[k], 0]))
            return g

        panels.append(col(0, [("每一步少推", 17, GREY_B, False), ("%.3f%%" % HEAD1, 32, RED, True),
                              ("沿 w̄（第 1 步实测）", 15, GREY_B, False), ("第 100 步：%.3f%%" % HEAD100, 15, GREY_B, False),
                              ("→ 每一步都差这么多", 15, WHITE, False)], y0=[1.2, 0.7, 0.2, -0.15, -0.55]))
        panels.append(col(1, [("每一步少走", 17, GREY_B, False), ("%.2f%%" % abs(SWAP_GPU), 32, ORANGE, True),
                              ("沿漂移方向（同一状态实测）", 15, GREY_B, False), ("大头：第 0 层", 15, GREY_B, False),
                              ("注意力的输出投影", 15, GREY_B, False)], y0=[1.2, 0.7, 0.2, -0.15, -0.5]))
        # ③ 小图框架
        p3 = VGroup(tx("lse 差（TPU − GPU）", 15, GREY_B).move_to([XC[2], 1.2, 0]),
                    Line([sx(2, 0), CH_TOP, 0], [sx(2, S_END), CH_TOP, 0], color=GREY, stroke_width=2),
                    tx("0", 13, GREY).next_to([sx(2, 0), CH_TOP, 0], LEFT, buff=0.06),
                    tx("−0.2", 13, GREY).next_to([sx(2, 0), self_y3(-0.2), 0], LEFT, buff=0.06),
                    Line([sx(2, 0), self_y3(-0.2), 0], [sx(2, S_END), self_y3(-0.2), 0], color=GREY, stroke_width=1).set_stroke(opacity=0.5))
        panels.append(p3)
        # ④ 柱子框架
        bar_x, bar_w = XC[3], 0.7
        p4 = VGroup(tx("没出现过的词那几行", 15, GREY_B).move_to([XC[3], 1.2, 0]),
                    tx("平方和多涨（TPU − GPU）", 13, GREY_B).move_to([XC[3], 0.9, 0]),
                    Line([bar_x - 0.7, CH_BOT, 0], [bar_x + 0.7, CH_BOT, 0], color=GREY, stroke_width=2),
                    tx("logit 越高，压得越多", 13, COL_DIFF).move_to([XC[3], -1.85, 0]))
        panels.append(p4)
        # ⑤ 小图框架
        p5 = VGroup(tx("参数范数差（TPU − GPU）", 15, GREY_B).move_to([XC[4], 1.2, 0]),
                    Line([sx(4, 0), y5(0), 0], [sx(4, S_END), y5(0), 0], color=GREY, stroke_width=2),
                    tx("0", 13, GREY).next_to([sx(4, 0), y5(0), 0], LEFT, buff=0.06))
        panels.append(p5)
        for p in panels:
            remember(p)

        # 只在走过实测点时出现的数字
        lse_labs = [remember(tx("第 %d 步 %s" % (s, ("%.4f" % v).rstrip("0").replace("-", "−")), 15, COL_DIFF, True)
                             .move_to([XC[2], -1.5, 0])) for s, v, _ in N.LSE_PROBE]
        rare_lab = remember(tx("＋%.1f（第 300 步实测）" % RARE_300, 15, COL_DIFF, True).move_to([XC[3], -1.5, 0]))
        pn_lab = remember(VGroup(tx("＋%.1e（第 300 步）" % PN_END, 15, COL_DIFF, True).move_to([XC[4], -1.5, 0]),
                                 tx("按组拆：输出层占 %d%%" % round(HEAD_SHARE * 100), 13, GREY_B).move_to([XC[4], -1.85, 0])))
        for p in panels:
            remember(p)

        def upd_static(_m=None):
            lit, s, _ = st()
            for i in range(5):
                fade_to(dim_grp[i], 1.0 - lit[i])         # 暗版让位，免得两层字叠影
                fade_to(lit_grp[i], lit[i])
                fade_to(panels[i], lit[i])
            for k, (ps, _, _) in enumerate(N.LSE_PROBE):
                nxt = N.LSE_PROBE[k + 1][0] if k + 1 < len(N.LSE_PROBE) else 1e9
                fade_to(lse_labs[k], lit[2] * (1.0 if ps <= s + 1e-6 < nxt else 0.0))
            fade_to(rare_lab, lit[3] * (1.0 if s >= S_END - 1e-6 else 0.0))
            fade_to(pn_lab, lit[4] * (1.0 if s >= S_END - 1e-6 else 0.0))

        static_all = VGroup(*dim_grp, *lit_grp, *panels, *lse_labs, rare_lab, pn_lab)
        upd_static()                                   # ⛔ 先手动算一次
        static_all.add_updater(upd_static)
        self.add(static_all)

        # ── 动态：③ 折线、④ 柱、⑤ 曲线（每帧由步数现算）─────────────
        def dyn():
            lit, s, _ = st()
            g = VGroup()
            # ③
            pts = [(sx(2, p[0]), self_y3(p[1])) for p in LSE_PTS if p[0] <= s]
            pts.append((sx(2, s), self_y3(lse_at(s))))
            g.add(poly(pts, COL_DIFF, 4).set_stroke(opacity=lit[2]))
            for ps, v, gg in N.LSE_PROBE:
                if ps <= s + 1e-6:
                    g.add(Dot([sx(2, ps), self_y3(v), 0], radius=0.05, color=COL_DIFF).set_opacity(lit[2]))
                    g.add(Dot([sx(2, ps), self_y3(gg), 0], radius=0.04, color=GREY).set_opacity(lit[2]))
            g.add(Dot([sx(2, s), self_y3(lse_at(s)), 0], radius=0.06, color=WHITE).set_opacity(lit[2]))
            # ④ 柱子：线性插值示意
            h = max((CH_TOP - 0.1 - CH_BOT) * clamp(s / S_END), 0.001)
            g.add(Rectangle(width=bar_w, height=h, stroke_width=0, fill_color=COL_DIFF, fill_opacity=0.75 * lit[3])
                  .move_to([bar_x, CH_BOT + h / 2, 0]))
            # ⑤ 逐步实测
            n = int(s + 1e-6)
            g.add(poly([(sx(4, k), y5(PN_G[k])) for k in range(n + 1)], GREY, 2).set_stroke(opacity=lit[4]))
            g.add(poly([(sx(4, k), y5(PN[k])) for k in range(n + 1)], COL_DIFF, 3).set_stroke(opacity=lit[4]))
            # 步数时钟
            x = CLK_X0 + (CLK_X1 - CLK_X0) * s / S_END
            g.add(Triangle(color=YELLOW, fill_color=YELLOW, fill_opacity=1, stroke_width=0).scale(0.11).rotate(3.14159)
                  .move_to([x, CLK_Y + 0.17, 0]))
            g.add(Line([CLK_X0, CLK_Y, 0], [x, CLK_Y, 0], color=YELLOW, stroke_width=6))
            return g

        self.add(always_redraw(dyn))

        # 时钟：轴、刻度（静态），步数字（只有 301 种，按步换）
        clk = VGroup(tx("训练步", 17, GREY_B).move_to([CLK_X0 - 0.75, CLK_Y, 0]),
                     Line([CLK_X0, CLK_Y, 0], [CLK_X1, CLK_Y, 0], color=GREY, stroke_width=3))
        for s in (0, 25, 50, 100, 150, 200, 250, 300):
            x = CLK_X0 + (CLK_X1 - CLK_X0) * s / S_END
            probe = s in (25, 50, 100, 300)
            clk.add(Line([x, CLK_Y - 0.09, 0], [x, CLK_Y + 0.09, 0], color=COL_PROBE if probe else GREY, stroke_width=3))
            clk.add(tx(str(s), 14, COL_PROBE if probe else GREY).move_to([x, CLK_Y - 0.28, 0]))
        self.add(clk)
        step_lab = always_redraw(lambda: tx("第 %d 步" % int(st()[1] + 1e-6), 17, YELLOW, True)
                                 .move_to([CLK_X0 + (CLK_X1 - CLK_X0) * st()[1] / S_END, CLK_Y + 0.45, 0]))
        self.add(step_lab)
        foot = tx("灰色 ＝ GPU 两次之间；刻度上白色 ＝ 探针实测的那几步；③ 实测点之间连直线，④ 柱子中间是插值示意，⑤ 是逐步实测", 14, GREY).move_to([0, -3.62, 0])
        self.add(foot)

        # ── 字幕：建一次，按区间切透明度 ─────────────────────────────
        caps = [tx(c, 21, GREY_B).next_to(title, DOWN, buff=0.18) for c in CAPS]
        caps[7].set_color(YELLOW)

        def show_cap(m, k):
            m.set_opacity(1.0 if st()[2] == k else 0.0)
        for k, m in enumerate(caps):
            show_cap(m, k)                              # ⛔ 先手动算一次
            m.add_updater(lambda m, k=k: show_cap(m, k))
            self.add(m)

        # ── 走时钟：一整段 linear，停顿由 state 的时间表决定 ─────────────
        self.play(tr.animate.set_value(T_TOTAL), run_time=T_TOTAL, rate_func=linear)
        self.play(tr.animate.set_value(T_TOTAL), run_time=1 / 30, rate_func=linear)   # 强制刷一帧（末帧 ＝ 第 0 帧）
        os.makedirs(STEPS_DIR, exist_ok=True)
        with open(os.path.join(STEPS_DIR, "DriftChain.json"), "w") as fh:
            json.dump({"pauses": PAUSES, "duration": round(self.renderer.time, 2)}, fh)


def self_y3(v):          # ③ 纵轴：0 在顶，往下是负；−0.25 到底
    return CH_TOP + (CH_BOT - CH_TOP) * (-v) / 0.25


def y5(v):               # ⑤ 纵轴：−0.5e-3 到 5e-3
    lo, hi = -0.5e-3, 5.0e-3
    return CH_BOT + (CH_TOP - CH_BOT) * (v - lo) / (hi - lo)


assert self_y3(LSE_PTS[-1][1]) > CH_BOT and y5(max(PN)) < CH_TOP and y5(min(PN + PN_G)) > CH_BOT   # 不出框
