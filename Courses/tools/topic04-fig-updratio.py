# -*- coding: utf-8 -*-
r"""专题四 · §3.5「那一步该迈多大」——&#160;一条能用眼睛对齐的参考线

⭐⭐⭐ 2026-09-18 新画（R19）。素材库那条（**Ⓒ 级**，agent 只读了 notebook 源码）
  说的是：把 `‖更新量‖ ⁄ ‖参数‖` 画成对数曲线，再在 **1e−3** 处画一条黑线。
  ⛔ 按这一批的规矩，落地前**自核了 CS231n 原文**，原话是：

    "A rough heuristic is that this ratio should be somewhere around **1e-3**.
     If it is lower than this then the learning rate might be too low.
     If it is higher then the learning rate is likely too high."

  ⭐⭐ 原文还有一句**转述里没有、但很要紧**的：
    **"Note: _updates_, not the raw gradients"** ——&#160;看的是**更新量**，不是梯度。
    （对 Adam 这两个差得远：它的更新量基本是 η，跟梯度大小几乎无关。）

⭐⭐⭐ **这一格最值得记的，是它跑出来的结果先把我要画的论点驳了。**
  我本来想画「太小／刚好／太大」三条，指望「太大 → loss 更差」。
  ⛔ 可真跑完发现：**太大那条的末 loss 反而最低**（0.37 vs 0.68）。
  ⭐ 顺着查下去才看清代价在别处：它的**权重范数涨到四倍出头**、
    **loss 抖动大了近一个量级**，而且**学习率再大几倍就直接 NaN**。
    ⛔ 具体倍数**不写死在这段注释里** ——&#160;图上那几个数是脚本当场算的。
    ⭐ 判据：**注释里的数也是一种断言**，写错了一样误导下一个人。
      （这一版初稿就写错过一次：把手测另一个学习率时的「20 倍」抄了进来，
       而脚本扫出来的学习率下实际是 8.6 倍。）
  ⭐⭐⭐ 于是论点被改成了真话：
    **这个比值不是「loss 最低」的指标，是「你离悬崖多远」的指标。**
    ——&#160;判据：**诊断指标不是优化目标。**

⚠️ 这是个**玩具问题**（随机 X → 随机 Y，没有验证集），
  所以「loss 更低」本来就不代表学得更好 ——&#160;它在记忆噪声。
  ⛔ 图上必须把这句说出来，否则就是选择性呈现。

📌 出处：CS231n《Neural Networks Part 3》"Ratio of weights:updates"
  ——&#160;上面引号里为原文原话。1e−3 是该文给的 rough heuristic。
"""
import math

import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400
STEPS = 400
TARGET = 1e-3                      # CS231n 给的那条经验线

# ⭐ 固定种子 ——&#160;同一份源码每次构建必须画出同一张图
_rng = np.random.default_rng(0)
_X = _rng.standard_normal((64, 8))
_Y = _rng.standard_normal((64, 4))


def _init():
    r = np.random.default_rng(1)
    return (r.standard_normal((8, 16)) / np.sqrt(8),
            r.standard_normal((16, 4)) / np.sqrt(16))


def run(lr, steps=STEPS):
    """一个两层 tanh MLP，纯 numpy 手写前向反向。返回 (比值, loss, ‖W1‖)。"""
    W1, W2 = _init()
    rs, ls, nw = [], [], []
    for _ in range(steps):
        H = np.tanh(_X @ W1)
        E = H @ W2 - _Y
        ls.append(float((E ** 2).mean()))
        nw.append(float(np.linalg.norm(W1)))
        dW2 = H.T @ E / len(_X)
        dH = (E @ W2.T) * (1 - H ** 2)
        dW1 = _X.T @ dH / len(_X)
        u1 = -lr * dW1              # ⭐ 是「更新量」不是梯度 ——&#160;原文特意强调的
        rs.append(float(np.linalg.norm(u1) / np.linalg.norm(W1)))
        W1 = W1 + u1
        W2 = W2 - lr * dW2
    return np.array(rs), np.array(ls), np.array(nw)


def _median_ratio(lr):
    return float(np.median(run(lr)[0][50:]))


# ⭐⭐ 「刚好」那个学习率是**扫出来的**，不是我挑的：让稳态比值最贴 1e−3
_best = None
for _e in np.arange(-3.0, 1.001, 0.02):
    _lr = float(10.0 ** _e)
    _m = _median_ratio(_lr)
    if not math.isfinite(_m) or _m <= 0:
        continue
    _d = abs(math.log10(_m) - math.log10(TARGET))
    if _best is None or _d < _best[0]:
        _best = (_d, _lr)
LR_OK = _best[1]
SPREAD = 30.0
LR_LO, LR_HI = LR_OK / SPREAD, LR_OK * SPREAD

R_LO, L_LO_, N_LO = run(LR_LO)
R_OK, L_OK, N_OK = run(LR_OK)
R_HI, L_HI_, N_HI = run(LR_HI)

M_LO, M_OK, M_HI = (float(np.median(r[50:])) for r in (R_LO, R_OK, R_HI))
assert M_LO < TARGET / 3 < TARGET * 3 < M_HI, \
    "三条要分别落在黑线的下方／线上／上方，现在 %.1e / %.1e / %.1e" % (M_LO, M_OK, M_HI)
assert abs(math.log10(M_OK / TARGET)) < 0.35, \
    "「刚好」那条没贴住 1e−3，实际 %.2e" % M_OK

# ⭐⭐⭐ 「太大」的代价不在 loss，在这两处 ——&#160;都是跑出来的
GROWTH = N_HI[-1] / N_HI[0]
GROWTH_OK = N_OK[-1] / N_OK[0]
JIT_HI = float(np.abs(np.diff(L_HI_[-100:])).mean())
JIT_OK = float(np.abs(np.diff(L_OK[-100:])).mean())
assert GROWTH > 3 * GROWTH_OK, "「太大」那条的权重膨胀要明显得多"
# ⛔ 阈值取「论点成立的最低要求」，不是「刚好让它过」。实测约 8.6 倍 ——&#160;
#   第一版写 >10 挂了，⭐ 而正确的反应是**先打印实际值**，不是把 10 改成 8。
#   这里定 5：低于 5 倍就谈不上「明显」了。
assert JIT_HI > 5 * JIT_OK, \
    "「太大」那条的 loss 抖动要明显更大，现在只有 %.1f 倍" % (JIT_HI / JIT_OK)
# ⛔ 而它的末 loss 反而更低 ——&#160;这一条**必须**在图上说出来，不能藏
assert L_HI_[-1] < L_OK[-1], "跑出来就是这样：太大那条 loss 反而低。别把它藏掉"

# ⭐ 悬崖在哪儿：二分找出「再大多少倍就 NaN」
_lo, _hi = LR_HI, LR_HI * 64
assert math.isfinite(run(_lo, 200)[1][-1]) and not math.isfinite(run(_hi, 200)[1][-1])
for _ in range(24):
    _mid = math.sqrt(_lo * _hi)
    if math.isfinite(run(_mid, 200)[1][-1]):
        _lo = _mid
    else:
        _hi = _mid
CLIFF = _hi / LR_HI
assert CLIFF < 12, "悬崖离得太远就说不上「在边上」了，现在 %.1f 倍" % CLIFF

YLO, YHI = -5.0, 0.0               # log10 纵轴（比值）


def main():
    f = Fig(W, "上半张是三条对数曲线，纵轴是每一步的更新量除以参数本身的大小，"
               "横轴是训练步数，中间横着一条黑色的参考线画在千分之一处。"
               "学习率太小那条一直贴在黑线下方，刚好那条骑在黑线上，"
               "太大那条远在黑线上方。"
               "下半张是同样三条设置下第一层权重范数的变化：太小和刚好那两条基本平着，"
               "太大那条一路涨到四倍。"
               "结论是这个比值不是用来让 loss 最低的，是用来看你离发散还有多远的")

    y0 = f.header(
        "「学习率设对了吗」　——　<tspan font-weight=\"700\">"
        "有一条能用眼睛对齐的线</tspan>",
        "⭐ 量的是<tspan font-weight=\"700\">更新量 ÷ 参数本身</tspan>"
        "　——　<tspan font-weight=\"700\">是更新量，不是梯度</tspan>（原文特意强调的）",
        [(GY2, "太小"), (GR, "刚好"), (RD, "太大"), (INK, "黑线 ＝ 1e−3")])

    # ══════════ Ⓐ 三条比值曲线 ＋ 那条黑线 ═══════════════════════════
    PH = 420
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 三个学习率，同一个网络　——　"
                 "<tspan font-weight=\"700\">看它们各自落在黑线的哪一侧</tspan>", INK,
                 sub="⭐ 纵轴对数；黑线是 CS231n 给的经验值 "
                     "<tspan font-weight=\"700\">1e−3</tspan>"
                     "　——　<tspan font-weight=\"700\">一条粗略的经验线，不是定律</tspan>")

    GX0, GX1 = 130, 1240
    GT, GB = py + 62, py + 312

    def gx(t):
        return GX0 + (GX1 - GX0) * t / float(STEPS - 1)

    def gy(v):
        r = (math.log10(max(v, 1e-9)) - YLO) / (YHI - YLO)
        return GB - (GB - GT) * r

    f.line(GX0, GB, GX1 + 16, GB, GY2, 1.2, arrow=False)
    f.line(GX0, GB, GX0, GT - 8, GY2, 1.2, arrow=False)
    for e in range(-5, 1):
        yy = gy(10.0 ** e)
        f.line(GX0 - 5, yy, GX1, yy, "#edeff1", 0.9, arrow=False)
        f.t(GX0 - 9, yy + 4, "10%s" % ("⁰" if e == 0 else "⁻%d" % -e),
            GY2, size=11, anchor="end")
    f.t(GX1 + 22, GB + 16, "步数 →", GY2, size=11.5)

    # ⭐ 那条黑线 ——&#160;这一格的主角
    yt = gy(TARGET)
    f.line(GX0, yt, GX1 + 8, yt, INK, 2.4, arrow=False)
    f.t(GX0 + 10, yt - 9, "1e−3　CS231n 的经验线", INK, True, 13)

    for seq, col, tag, lr in ((R_LO, GY2, "太小", LR_LO),
                              (R_OK, GR, "刚好", LR_OK),
                              (R_HI, RD, "太大", LR_HI)):
        d = "M %.1f %.1f" % (gx(0), gy(seq[0]))
        for t in range(1, STEPS):
            d += " L %.1f %.1f" % (gx(t), gy(seq[t]))
        f.path(d, col, 2.2, arrow=False)
        f.t(gx(STEPS - 1) + 14, gy(seq[-1]) + 5,
            "%s　η＝%.3g" % (tag, lr), col, True, 12.5)

    f.t(700, py + 356,
        "⭐⭐ 原文的读法就三句："
        "<tspan font-weight=\"700\">贴着线 ＝ 大致合适</tspan>；"
        "<tspan font-weight=\"700\">低太多 ＝ 学习率可能偏小</tspan>；"
        "<tspan font-weight=\"700\">高太多 ＝ 学习率可能偏大</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(700, py + 388,
        "⭐ 它的好处是<tspan font-weight=\"700\">跟模型大小无关</tspan>"
        "　——　分子分母同量纲，约掉了。换个模型这条线还在原地。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 「太大」的代价，不在 loss 上 ═══════════════════════
    PH2 = 372
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐⭐ 可「太大」那条的 loss <tspan font-weight=\"700\">反而更低</tspan>"
                  "　——　代价在别处", RD,
                  sub="⛔ 跑出来就是这样，<tspan font-weight=\"700\">不藏</tspan>"
                      "　——　所以这条比值<tspan font-weight=\"700\">不是「让 loss 最低」的指标</tspan>")

    NT, NB = py2 + 62, py2 + 250
    _nmax = max(float(N_HI.max()), float(N_OK.max())) * 1.06

    def ny(v):
        return NB - (NB - NT) * v / _nmax

    f.line(GX0, NB, GX1 + 16, NB, GY2, 1.2, arrow=False)
    f.line(GX0, NB, GX0, NT - 8, GY2, 1.2, arrow=False)
    for v in (0, 5, 10, 15):
        if v * 1.0 <= _nmax:
            f.line(GX0 - 5, ny(v), GX1, ny(v), "#edeff1", 0.9, arrow=False)
            f.t(GX0 - 9, ny(v) + 4, "%d" % v, GY2, size=11, anchor="end")
    f.t(GX0, NT - 18, "第一层权重的范数 ‖W‖", GY2, size=12)

    # ⛔ 在这一格里「太小」和「刚好」两条**几乎重合**（3.63 vs 4.12）——&#160;
    #   硬把标签错开是在掩饰。⭐ 判据：**两条线真的重合时，就合并标注并说出来**，
    #   而不是拉开标签假装它们分得开。
    for seq, col, tag in ((N_LO, GY2, None), (N_OK, GR, "刚好／太小"), (N_HI, RD, "太大")):
        d = "M %.1f %.1f" % (gx(0), ny(seq[0]))
        for t in range(1, STEPS):
            d += " L %.1f %.1f" % (gx(t), ny(seq[t]))
        f.path(d, col, 2.2, arrow=False)
        if tag:
            f.t(gx(STEPS - 1) + 14, ny(seq[-1]) + 5, tag, col, True, 12.5)
            if "／" in tag:
                f.t(gx(STEPS - 1) + 14, ny(seq[-1]) + 24,
                    "两条几乎重合", GY2, size=11.5)

    f.t(700, py2 + 288,
        "⭐⭐ 太大那条把权重<tspan font-weight=\"700\">撑到 %.1f 倍</tspan>，"
        "loss 的逐步抖动大了<tspan font-weight=\"700\">约 %d 倍</tspan>"
        "　——　而学习率<tspan font-weight=\"700\">再乘 %.1f 就直接 NaN</tspan>。"
        % (GROWTH, int(round(JIT_HI / JIT_OK)), CLIFF),
        INK, size=14.5, anchor="middle")
    f.t(700, py2 + 326,
        "⭐⭐⭐ 判据：<tspan font-weight=\"700\">"
        "这个比值不是「loss 最低」的指标，是「你离悬崖多远」的指标。</tspan>"
        "　——　<tspan font-weight=\"700\">诊断指标不是优化目标。</tspan>",
        RD, size=15, anchor="middle")
    f._pan = None

    yb = f.band(py2 + PH2 + 20, "warn",
                "⚠️ 这是个玩具问题 ——　所以「loss 更低」在这儿本来就不算数",
                ("⛔ 数据是<tspan font-weight=\"700\">随机 X → 随机 Y</tspan>，"
                 "没有验证集 ——&#160;网络在<tspan font-weight=\"700\">记忆噪声</tspan>，"
                 "谁记得快谁 loss 低。"
                 "⭐ 真实训练里，撑大权重 ＋ 抖动加剧通常换来的是"
                 "<tspan font-weight=\"700\">更差</tspan>的泛化。",
                 "⭐⭐ 但这一格想讲的那条<tspan font-weight=\"700\">不依赖这个问题的好坏</tspan>："
                 "那条比值线只回答「你的步子相对参数本身是不是一个合理的量级」，"
                 "<tspan font-weight=\"700\">它是个体温计，不是治疗方案</tspan>。"))

    yb = f.src(yb + 16,
               "📌 CS231n《Neural Networks Part 3》\"Ratio of weights:updates\"："
               "「A rough heuristic is that this ratio should be somewhere around 1e-3.」"
               "⭐ 原文还特意强调 <tspan font-weight=\"700\">"
               "\"Note: updates, not the raw gradients\"</tspan>。",
               "⚠️ 曲线是我们自己跑的：两层 tanh MLP（8→16→4），"
               "纯 numpy 手写前向反向，固定随机种子，跑 %d 步。"
               "「刚好」那个学习率是**扫出来的**（让稳态比值最贴 1e−3），不是挑的；"
               "「再乘 %.1f 就 NaN」是<tspan font-weight=\"700\">二分搜出来的</tspan>。"
               % (STEPS, CLIFF))

    f.save("fig4-updratio.svg", yb + 14)


if __name__ == "__main__":
    main()
