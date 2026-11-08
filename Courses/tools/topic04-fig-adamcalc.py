# -*- coding: utf-8 -*-
r"""专题四 · §3.3b「Adam 的两笔账，拿几个数当场算一遍」

2026-11-08 补。这一节原来把 Adam 讲成了一串话：「二阶矩是这个参数自己的尺子」
「除完之后大家在同一个刻度上」「梯度正负乱跳，比值变小，自动缩步」。
每一句都对，可**一个数都没算** —— 听的人点头，却说不出「梯度恒为 2 时到底走多远」。
这张图只做一件事：把那几句话换成能心算的数。

三格，一格一个问题：
  Ⓐ 梯度差一万倍，步子差多少？ —— 一样大。恒为 0.02 ／ 2 ／ 200，m̂ ÷ √v̂ 都是 1。
  Ⓑ 那梯度里还剩什么信息？ —— 方向，以及方向有多一致。
     恒为 ＋10 → 每步净走 1 个学习率；带方向的噪声 → 约 0.2；＋10／−10 交替 → 净走 0。
  Ⓒ 每步都按学习率走，到了谷底不会来回蹦吗？ —— 谷底梯度变成噪声，m 自己缩小；
     真正停稳靠学习率调度。

承重的数全部是**真跑 Adam 跑出来的**（Algorithm 1 原式，含偏差修正），不是手填：
  · Adam 更新式与默认值 β₁＝0.9、β₂＝0.999、ε＝1e-8：arXiv 1412.6980 Algorithm 1（已读原文）
  · ＋10／−10 交替时 |m̂| 的稳态有闭式：(1−β₁)/(1＋β₁)×10 —— 脚本拿它对一遍
  · Ⓒ 用 β₂＝0.95（大模型预训练常用值，见 §3.6 出处），一维二次谷 ＋ 高斯噪声，
    结论（远处≈1、谷底≈0.2、调度后才停稳）在**多个随机种子上取平均**再断言，不靠一个幸运种子

刻意没画的：Ⓒ 是一维玩具，不是真实 loss 曲面；它只说明「谷底的梯度是噪声」这一件事。
"""
import math
import random

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400
B1, B2, EPS = 0.9, 0.999, 1e-8            # Adam 原文默认值（Algorithm 1）


def adam(gs, b1=B1, b2=B2, eps=EPS):
    """逐步返回 (m̂, √v̂, m̂/(√v̂+ε))。原文 Algorithm 1 逐行照抄。"""
    m = v = 0.0
    out = []
    for t, g in enumerate(gs, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        out.append((mh, math.sqrt(vh), mh / (math.sqrt(vh) + eps)))
    return out


# ══════════ Ⓐ 恒定梯度：大小不同，步子一样 ══════════════════════════
CONST = (0.02, 2.0, 200.0)
A_RES = {g: adam([g] * 100)[-1] for g in CONST}
for g, (mh, sv, r) in A_RES.items():
    assert abs(mh - g) < 1e-9 * g and abs(sv - g) < 1e-9 * g, (g, mh, sv)
    assert abs(r - 1) < 1e-5, "恒定梯度下必须每步正好走 1 个学习率：g=%g 给了 %r" % (g, r)
assert CONST[-1] / CONST[0] == 1e4          # 图上写「差一万倍」

# ══════════ Ⓑ 方向有多一致 → 步子多大 ════════════════════════════
N_LONG = 4000
random.seed(7)
NOISY = [2.0 + 10.0 * random.gauss(0, 1) for _ in range(N_LONG)]   # 均值 2、噪声 10
STRIPS = (
    ("一直是 ＋10", [10.0] * N_LONG, GR),
    ("均值 ＋2，噪声 ±10", NOISY, OR),
    ("＋10、−10 交替", [10.0 * (-1) ** t for t in range(N_LONG)], RD),
)


def net_per_step(gs):
    r = adam(gs)[200:]                       # 去掉开头，取稳态
    return sum(x[2] for x in r) / len(r), sum(abs(x[2]) for x in r) / len(r)


B_NET = [net_per_step(gs) for _, gs, _ in STRIPS]
assert abs(B_NET[0][0] - 1) < 1e-6
assert 0.12 < B_NET[1][0] < 0.30, B_NET[1]           # 带方向的噪声：约 0.2
assert abs(B_NET[2][0]) < 0.01, B_NET[2]               # 交替：净走 0
ALT_STEP = (1 - B1) / (1 + B1)                         # 交替时每步幅度的闭式
assert abs(B_NET[2][1] - ALT_STEP) < 1e-3, (B_NET[2][1], ALT_STEP)
_alt = adam(STRIPS[2][1][:200])[-1]
assert abs(_alt[1] - 10) < 1e-6 and abs(abs(_alt[0]) - ALT_STEP * 10) < 1e-3   # √v̂＝10，|m̂|≈0.53

# ══════════ Ⓒ 到了谷底 ═══════════════════════════════════════════
VB2 = 0.95
T_V, LR, SIG, W0 = 600, 0.1, 2.0, 10.0


def sched_const(t):
    return LR


def sched_cos(t):
    return LR * 0.5 * (1 + math.cos(math.pi * t / T_V))


def valley(sched, seed):
    """loss ＝ ½w²，梯度 ＝ w ＋ 噪声。返回每步的 w 和 m̂/√v̂。"""
    rnd = random.Random(seed)
    w, m, v, ws, rs = W0, 0.0, 0.0, [], []
    for t in range(1, T_V + 1):
        g = w + SIG * rnd.gauss(0, 1)
        m = B1 * m + (1 - B1) * g
        v = VB2 * v + (1 - VB2) * g * g
        r = (m / (1 - B1 ** t)) / (math.sqrt(v / (1 - VB2 ** t)) + EPS)
        w -= sched(t) * r
        ws.append(w)
        rs.append(r)
    return ws, rs


def _mean(xs):
    return sum(xs) / len(xs)


SEEDS = range(1, 21)
_far, _near, _hc, _hs, _mc, _ms = [], [], [], [], [], []
for s in SEEDS:
    wc, rc = valley(sched_const, s)
    ws_, _ = valley(sched_cos, s)
    _far.append(_mean([abs(x) for x in rc[5:60]]))
    _near.append(_mean([abs(x) for x in rc[300:]]))
    _hc.append(_mean([abs(x) for x in wc[500:]]))
    _hs.append(_mean([abs(x) for x in ws_[500:]]))
    _mc.append(_mean([abs(wc[i] - wc[i - 1]) for i in range(500, T_V)]))
    _ms.append(_mean([abs(ws_[i] - ws_[i - 1]) for i in range(500, T_V)]))
FAR, NEAR = _mean(_far), _mean(_near)
HOVER_C, HOVER_S = _mean(_hc), _mean(_hs)
MOVE_C, MOVE_S = _mean(_mc), _mean(_ms)      # 最后 100 步，平均每步挪多少
assert FAR > 0.85, "离谷底远时，信号压过噪声，每步应接近 1 个学习率：%.2f" % FAR
assert 0.1 < NEAR < 0.3, "到了谷底梯度是噪声，m 应缩到约 0.2：%.2f" % NEAR
assert HOVER_S < HOVER_C / 2, "学习率降下来之后离谷底更近：%.3f vs %.3f" % (HOVER_S, HOVER_C)
assert MOVE_S < MOVE_C / 20, "学习率降下来之后才停得住：%.5f vs %.5f" % (MOVE_S, MOVE_C)

SEED_SHOW = 1
W_C, R_C = valley(sched_const, SEED_SHOW)
W_S, _ = valley(sched_cos, SEED_SHOW)


def smooth(xs, k=15):
    out = []
    for i in range(len(xs)):
        seg = xs[max(0, i - k):i + 1]
        out.append(_mean([abs(x) for x in seg]))
    return out


R_SMOOTH = smooth(R_C)


def _curve(f, xs, ys, x0, y0, w, h, ymin, ymax, col, sw=2.4):
    n = len(xs)
    pts = []
    for i in range(n):
        px = x0 + w * (xs[i] - xs[0]) / (xs[-1] - xs[0])
        yv = min(max(ys[i], ymin), ymax)
        pts.append((px, y0 - h * (yv - ymin) / (ymax - ymin)))
    f.path(pts, col, sw, arrow=False)


def main():
    f = Fig(W, "Adam 的两笔账，拿几个数当场算一遍。"
               "第一格：梯度恒为零点零二、二、二百，三行的梯度差一万倍，"
               "可一阶动量 m 和二阶动量开根号都等于梯度本身，相除都是一，"
               "所以三行每步都正好走一个学习率，三支步长箭头一样长。"
               "第二格：三条梯度序列，一直是正十的，每步净走一个学习率；"
               "均值正二、噪声正负十的，每步净走约零点二个学习率；"
               "正十负十交替的，m 正负相消只剩零点五左右，而根号 v 是十，"
               "每步只走零点零五个学习率，而且一来一回，净走为零。"
               "第三格：一维的谷，梯度等于位置加噪声。离谷底远时每步接近一个学习率，"
               "一路直走过去；到了谷底梯度只剩噪声，m 自己缩小，每步只剩约零点二个学习率；"
               "学习率不变时它在谷底附近小幅晃，学习率按余弦降下来之后才停稳。")

    y0 = f.header(
        "Adam 的两笔账　——　<tspan font-weight=\"700\">拿几个数当场算一遍</tspan>",
        "m ＝ 梯度的滑动平均（记方向）　·　v ＝ 梯度平方的滑动平均（记大小）　·　"
        "<tspan font-weight=\"700\">每步走 ＝ 学习率 × m ÷ √v</tspan>"
        "（m、v 都已做偏差修正；ε 只防除零，下面忽略）",
        [(BL, "梯度"), (GR, "m ÷ √v ＝ 走几个学习率")])

    # ══════════ Ⓐ ══════════════════════════════════════════════════
    PH = 300
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 梯度差<tspan font-weight=\"700\">一万倍</tspan>，"
                 "步子<tspan font-weight=\"700\">一样大</tspan>", BL,
                 sub="左边梯度画在对数轴上；右边箭头长度 ＝ 这一步走几个学习率")
    AX0, AXW = 150, 420                       # 对数轴 0.01 … 1000
    LMIN, LMAX = -2.0, 3.0

    def lx(g):
        return AX0 + AXW * (math.log10(g) - LMIN) / (LMAX - LMIN)

    f.t(AX0, py + 32, "梯度（对数轴）", GY, size=12.5)
    f.t(640, py + 32, "m", GY, True, 13, "middle")
    f.t(740, py + 32, "v", GY, True, 13, "middle")
    f.t(840, py + 32, "√v", GY, True, 13, "middle")
    f.t(1100, py + 32, "这一步走多远", GY, True, 13, "middle")
    for k, g in enumerate(CONST):
        yy = py + 76 + k * 62
        mh, sv, r = A_RES[g]
        lab = ("%g" % g)
        f.t(AX0 - 16, yy + 5, "恒为 " + lab, BL, True, 14, "end")
        f.box(AX0, yy - 11, lx(g) - AX0, 22, "#e8f0fe", BL, 3)
        f.t(640, yy + 5, "%g" % mh, INK, size=14, anchor="middle", mono=True)
        f.t(740, yy + 5, "%g" % (sv * sv), INK, size=14, anchor="middle", mono=True)
        f.t(840, yy + 5, "%g" % sv, INK, size=14, anchor="middle", mono=True)
        f.t(905, yy + 5, "→ %g÷%g ＝ %d" % (mh, sv, round(r)), GR, True, 13.5)
        f.line(1040, yy, 1040 + 150 * r, yy, GR, 3.2)
        f.t(1200, yy + 5, "1 个学习率", GR, True, 14)
    for d in range(int(LMIN), int(LMAX) + 1):
        xx = lx(10 ** d)
        f.line(xx, py + 254, xx, py + 248, GY2, 1, arrow=False)
        f.t(xx, py + 268, "%g" % (10 ** d), GY2, size=11, anchor="middle", mono=True)
    f.line(AX0, py + 251, AX0 + AXW, py + 251, GY2, 1, arrow=False)
    f.t(1100, py + 262,
        "大小被 √v 除掉了 ——&#160;<tspan font-weight=\"700\">"
        "一个学习率，就能同时管住这三个参数</tspan>", INK, size=14, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ ══════════════════════════════════════════════════
    PH2 = 350
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 大小没了，剩下的是<tspan font-weight=\"700\">方向有多一致</tspan>", OR,
                  sub="每条画前 40 步的梯度（柱）；右边是稳态下每步净走几个学习率")
    SX0, SW, NSHOW = 260, 560, 40
    for k, (lab, gs, col) in enumerate(STRIPS):
        cy = py2 + 76 + k * 82
        f.t(SX0 - 18, cy + 5, lab, col, True, 14, "end")
        f.line(SX0, cy, SX0 + SW, cy, GY2, 1, arrow=False)
        bw = SW / NSHOW
        for i in range(NSHOW):
            g = max(min(gs[i], 22.0), -22.0)
            h = 30 * g / 22.0
            f.box(SX0 + i * bw + 2, cy - max(h, 0), bw - 4, abs(h), "#e8f0fe", BL, 1)
        net, mag = B_NET[k]
        f.line(960, cy, 960 + 220 * max(net, 0) + 0.01, cy, col, 4,
               arrow=net > 0.05)
        f.line(960, cy - 26, 960, cy + 26, GY2, 1, arrow=False)
        f.t(1200, cy + 5, "净走 %.2f" % abs(net) if net > 0.05 else "净走 0", col, True, 15)
    f.t(960, py2 + 44, "0", GY2, size=11, anchor="middle", mono=True)
    f.t(1180, py2 + 44, "1 个学习率", GY2, size=11, anchor="middle")
    f.line(1180, py2 + 50, 1180, py2 + 270, GY2, 1, dash="3 4", arrow=False)
    f.t(SX0 + SW + 20, py2 + 76 + 2 * 82 + 30,
        "m 正负相消 ≈ ±%.2f，√v ＝ 10 → 每步 ±%.2f，一来一回" % (ALT_STEP * 10, ALT_STEP),
        RD, size=12.5)
    f.t(700, py2 + 316,
        "<tspan font-weight=\"700\">走多远 ≈ 学习率 × 方向有多确定</tspan>"
        "　——　梯度一致就走满，拿不准就自动缩步", INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ ══════════════════════════════════════════════════
    PH3 = 376
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ 每步都走一个学习率，到了谷底<tspan font-weight=\"700\">不会来回蹦</tspan>吗？", GR,
                  sub="一维的谷：梯度 ＝ 位置 ＋ 噪声。左：位置随步数；右：每步走几个学习率")
    # 左上：整条路（看「直着走过去」）；左下：谷底放大（看「晃」和「停稳」）
    OX, OY, AW, AH = 90, py3 + 290, 560, 220
    xs = list(range(1, T_V + 1))
    TY, TH = py3 + 130, 90                          # 上图底边、高度：w 从 -1 到 10
    f.line(OX, TY - TH * 1 / 11, OX + AW, TY - TH * 1 / 11, GY2, 1, arrow=False)
    f.line(OX, TY, OX, TY - TH, GY2, 1, arrow=False)
    f.t(OX - 8, TY - TH * 1 / 11 + 4, "谷底", GY2, size=11, anchor="end")
    f.t(OX - 8, TY - TH + 8, "+%g" % W0, GY2, size=11, anchor="end", mono=True)
    _curve(f, xs, W_C, OX, TY, AW, TH, -1, W0, GY2, 2.2)
    f.t(OX + 130, TY - TH + 22, "远处：每步约 1 个学习率，直着走过去", INK, size=12.5)
    ZLO, ZR = 200, 1.5
    zx = xs[ZLO - 1:]
    BY, BH = OY, 100
    f.t(OX, OY - AH + 112, "谷底放大（第 %d 步以后，纵轴放大约 %d 倍）" % (ZLO, round((BH / (2 * ZR)) / (TH / 11))),
        GY, size=11.5)
    BY, BH = OY, 100
    f.line(OX, BY - BH / 2, OX + AW, BY - BH / 2, GY2, 1, dash="3 4", arrow=False)
    f.line(OX, BY, OX, BY - BH, GY2, 1, arrow=False)
    f.t(OX - 8, BY - BH / 2 + 4, "谷底", GY2, size=11, anchor="end")
    _curve(f, zx, W_C[ZLO - 1:], OX, BY, AW, BH, -ZR, ZR, GY2, 2.0)
    _curve(f, zx, W_S[ZLO - 1:], OX, BY, AW, BH, -ZR, ZR, GR, 2.6)
    f.t(OX + AW, BY + 18, "第 %d 步" % T_V, GY2, size=11, anchor="end")
    f.t(OX + 6, BY + 18, "第 %d 步" % ZLO, GY2, size=11)
    f.t(OX + AW + 8, BY - BH + 14, "学习率不变：", GY, True, 12.5)
    f.t(OX + AW + 8, BY - BH + 32, "一直小幅晃", GY, True, 12.5)
    f.t(OX + AW + 8, BY - BH / 2 + 22, "学习率按余弦", GR, True, 12.5)
    f.t(OX + AW + 8, BY - BH / 2 + 40, "降下来：停稳", GR, True, 12.5)
    # 右：|m/√v|
    RX, RW = 800, 520
    f.line(RX, OY, RX + RW, OY, GY2, 1, arrow=False)
    f.line(RX, OY, RX, OY - AH, GY2, 1, arrow=False)
    for lv in (1.0, NEAR):
        yy = OY - AH * lv / 1.2
        f.line(RX, yy, RX + RW, yy, GY2, 1, dash="3 4", arrow=False)
    f.t(RX - 8, OY - AH / 1.2 + 4, "1", GY2, size=11, anchor="end", mono=True)
    f.t(RX - 8, OY - AH * NEAR / 1.2 + 4, "%.1f" % NEAR, GY2, size=11, anchor="end", mono=True)
    f.t(RX - 8, OY + 4, "0", GY2, size=11, anchor="end", mono=True)
    f.t(RX + RW, OY + 18, "第 %d 步" % T_V, GY2, size=11, anchor="end")
    _curve(f, xs, R_SMOOTH, RX, OY, RW, AH, 0, 1.2, OR, 2.4)
    f.t(RX + 140, OY - AH / 1.2 - 10, "信号压过噪声 ≈ %.2f" % FAR, OR, True, 12.5)
    f.t(RX + RW, OY - AH * 0.62,
        "谷底只剩噪声，m 自己缩小 ≈ %.2f" % NEAR, OR, True, 12.5, "end")
    f.t(700, py3 + 342,
        "Adam 自己会在谷底缩步，<tspan font-weight=\"700\">但总步长归学习率管</tspan>"
        "　——　最后停稳，靠的是学习率调度", INK, size=15, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "Adam 故意不看梯度有多大，只看方向有多稳",
                ("不同参数的梯度能差几个数量级，「步长 ＝ 学习率 × 梯度」一个学习率顾不过来；"
                 "<tspan font-weight=\"700\">除以 √v 等于给每个参数单独归一化</tspan>，"
                 "于是一个学习率管得住所有参数。",
                 "代价是梯度的大小这条信息被扔了，剩下方向和一致程度 ——&#160;"
                 "<tspan font-weight=\"700\">真实训练的梯度大半是噪声，所以典型的一步是零点几个学习率，不是 1</tspan>。"))

    yb = f.src(yb + 16,
               "更新式逐行照 Adam 原文 Algorithm 1（arXiv 1412.6980，已读原文）："
               "m ← β₁m ＋ (1−β₁)g，v ← β₂v ＋ (1−β₂)g²，m̂ ＝ m/(1−β₁ᵗ)，v̂ ＝ v/(1−β₂ᵗ)，"
               "θ ← θ − α·m̂/(√v̂ ＋ ε)；原文给的默认值 α＝0.001、β₁＝0.9、β₂＝0.999、ε＝10⁻⁸。"
               "Ⓐ Ⓑ 用这组默认值。",
               "本课推导：＋10／−10 交替时，m 的稳态在 ±(1−β₁)/(1＋β₁)×10 ≈ ±%.2f 之间跳，"
               "√v̂ 恒为 10，所以每步幅度是 %.3f 个学习率 ——&#160;脚本里拿这个闭式对了一遍真跑的结果。"
               % (ALT_STEP * 10, ALT_STEP),
               "Ⓑ 中间那条是「均值 2、标准差 10 的高斯噪声」，净走 %.2f 是 %d 步稳态的平均；"
               "Ⓒ 用 β₂＝0.95、学习率 0.1、噪声标准差 2 的一维二次谷，"
               "远处 %.2f、谷底 %.2f 两个数是 20 个随机种子的平均，图上画的是其中一个种子。"
               "它是玩具，只说明「谷底的梯度是噪声」这一件事。"
               % (B_NET[1][0], N_LONG - 200, FAR, NEAR),
               "谷底那个「约 0.2」跟本讲 Adam update RMS ≈ 0.2 那条（苏剑林的推导，"
               "纯噪声梯度下 √[(1−β₁)/(1＋β₁)] ≈ 0.23）是同一件事：这里的「每步走几个学习率」取的是绝对值平均，"
               "口径略不同，量级一致。")
    f.save("fig4-adamcalc.svg", yb + 14)


if __name__ == "__main__":
    main()
