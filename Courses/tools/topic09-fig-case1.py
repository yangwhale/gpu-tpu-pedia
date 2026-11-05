# -*- coding: utf-8 -*-
r"""专题九 · 第七节「案例（上）」的三张图。

fig9-drift  回答：loss 偏移之外，哪个每步都记的量也在慢慢长？—— 参数范数：TPU 从约第 100 步起近似线性地多涨。
            数据：topic09_series.json 的真实逐步值（TPU − GPU 第一份；GPU 第二份 − 第一份当噪声）。
fig9-rows   回答：多涨的那一截在哪？—— 几乎全在输出层，而且是数据里没出现过的那些词的行。
            数字：第 300 步分组与按词频分桶的平方和差（实测，见 topic09_numbers）。
fig9-eps    回答：为什么偏偏是冷门词的行这么敏感？—— 它们的梯度比 Adam 的 ε 还小，在那儿 Adam 不再「归一化」。
            现算：稳态下 m̂ ≈ g、v̂ ≈ g²，更新 u ＝ g ÷ (|g| ＋ ε)；弹性 (du/u)÷(dg/g) ＝ ε ÷ (|g| ＋ ε)（本课推导）。
            ⛔ 刻意没画：真实的 m̂、v̂ 不是稳态，这条曲线只讲「在 ε 附近 Adam 怎么变」，不拿来算具体的更新量。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU
import topic09_numbers as N

W = 1400
S = N.series()
PT = S["pnorm_diff_tpu_minus_gpu_a"]
PG = S["pnorm_diff_gpu_b_minus_gpu_a"]
P0, P1 = S["pnorm_gpu_a_first_last"]
assert abs((P1 - P0) - N.PN_GPU_GROWTH) < 0.001, P1 - P0
assert abs(PT[999] - N.PN_TPU_EXTRA) < 0.0005, PT[999]
assert abs(max(abs(x) for x in PG) - 6.6e-4) < 0.05e-4
EXTRA_SHARE = PT[999] / (P1 - P0)
assert abs(EXTRA_SHARE - 0.17) < 0.01


def fig_drift():
    f = Fig(W, "横轴第 1 到 1000 步，纵轴是参数范数的差。灰色是第二次 GPU 减第一次，一直贴着零，最多差万分之几。"
               "蓝色是 TPU 减 GPU：前 100 步看不出，之后近似一条斜线往上走，到第 1000 步多出 0.021。"
               "GPU 自己 1000 步一共涨了 0.127，TPU 多涨了约 17%。")
    y0 = f.header("跟 loss 一起漂起来的，还有参数范数　——　<tspan font-weight=\"700\">从第 100 步起，TPU 一直比 GPU 涨得快一点</tspan>",
                  "一个点 ＝ 一步的参数范数之差（全部参数平方和开根号，约 3042）；灰 ＝ GPU 自己重跑的差，当噪声",
                  [(GY2, "GPU 第二次 − 第一次（噪声）"), (BL, "TPU − GPU 第一次")])
    PH = 330
    py = f.panel(0, y0, W, PH, "参数范数，1000 步（实测）", BL)
    X0, XW, YB, YH = 110, 1240, py + 270, 230
    lo, hi = -0.002, 0.022

    def X(st):
        return X0 + (st - 1) / 999.0 * XW

    def Y(v):
        return YB - (v - lo) / (hi - lo) * YH
    f.line(X0, Y(0), X0 + XW, Y(0), GY2, 1.0, arrow=False)
    for v in (0.0, 0.01, 0.02):
        f.t(X0 - 10, Y(v) + 4, "%+.2f" % v if v else "0", GY, size=12, anchor="end")
    for st in (1, 100, 500, 1000):
        f.t(X(st), YB + 24, "第 %d 步" % st, GY, size=12, anchor="middle")
    f.path([(X(k + 1), Y(v)) for k, v in enumerate(PG)], GY2, 1.4, arrow=False)
    f.path([(X(k + 1), Y(v)) for k, v in enumerate(PT)], BL, 2.4, arrow=False)
    f.t(X(1000) - 6, Y(PT[999]) - 12, "第 1000 步 ＋%.3f" % PT[999], BL, True, 15, "end")
    f.t(X(560), Y(0) + 22, "GPU 自己：最多差 %.1e，贴着零" % max(abs(x) for x in PG), GY, True, 13, "middle")
    f.line(X(100), py + 20, X(100), YB, OR, 1.4, dash="5 5", arrow=False)
    f.t(X(100) + 8, py + 34, "约第 100 步：loss 偏移也是这时拉开的", OR, True, 13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "偏移是慢慢长出来的 → 嫌疑从「某个前向算子」收窄到「每一步的更新」", [
        "GPU 自己 1000 步一共涨 0.127，TPU 多涨了 0.021（约 17%）；第 1 步的那点差只是记录分辨率。",
        "代价：这只是线索，不是结论 —— 「更新规则有差别」还要一项项验，先验最便宜的。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：GPU 第一份运行参数范数从 3042.377 涨到 3042.503；TPU − GPU 第 100 / 300 / 1000 步 ＋3.9e-4 / ＋4.6e-3 / ＋2.1e-2；GPU 两次之间最多差 6.6e-4（记录分辨率的 4 格）。",
               "📌 两边记录的格距不同：GPU 侧是 FP32 平方和开根号，平方和（约 925 万）一格是 1，折成参数范数约 1 ÷ (2 × 3042) ≈ 1.6e-4，GPU 之间的差都是它的整数倍；TPU 侧记的就是 FP32 的参数范数，3042 附近一格 2.4e-4。第 1 步的 ＋1.7e-4 不到 TPU 一格，不是差异。")
    f.save("fig9-drift.svg", yb + 14)


# 左右换算（本课推导）：‖θ‖ ＝ √S → Δ‖θ‖ ≈ ΔS ÷ (2‖θ‖)。ΔS 用案例记录里的输出层平方和差合计 ROWS_300_TOTAL（＋23.2）；
# 四根柱是各桶四舍五入后的值，加起来 23.2，跟记录的合计一致（R11：100 次以上那桶是 ＋0.47，不是 0）。
ROWS_BARS_SUM = sum(v for _, v in N.ROWS_300)
ROWS_CONV = N.ROWS_300_TOTAL / (2 * P0)
ROWS_BARS_CONV = ROWS_BARS_SUM / (2 * P0)
assert abs(ROWS_BARS_SUM - 23.2) < 1e-9 and abs(P0 - 3042.4) < 0.1
assert abs(ROWS_CONV / N.GROUP_HEAD_300 - 1) < 0.01, ROWS_CONV          # 3.81e-3 对 3.83e-3
assert abs(ROWS_BARS_CONV / N.GROUP_HEAD_300 - 1) < 0.04, ROWS_BARS_CONV  # 3.73e-3，差约 3%


def fig_rows():
    f = Fig(W, "左：第 300 步参数范数的差拆到每一组参数，输出层贡献了 4.39e-3 里的 3.83e-3，其余各组都在噪声量级。"
               "右：输出层的平方和差再按词在前 300 步数据里出现的次数分桶：没出现过的词 +17.1，出现 1 到 10 次的 +3.1，11 到 100 次的 +2.5，100 次以上的 +0.5；GPU 两次之间每桶不超过 0.21。"
               "两边能换算：右边合计 ＋23.2 除以（2 × 参数范数 3042）约 3.81e-3，跟左边输出层的 3.83e-3 对得上。")
    y0 = f.header("漂移几乎全在输出层，而且全在冷门词那几行",
                  ("左：柱长 ＝ 这一组贡献的参数范数差；右：柱长 ＝ 输出层平方和多出多少。换算：右 ÷（2 × 参数范数 %d）≈ 左（合计 %.1f → %.2e，对 %.2e）" % (round(P0), N.ROWS_300_TOTAL, ROWS_CONV, N.GROUP_HEAD_300)).replace("e-0", "e-"),
                  [(BL, "TPU − GPU"), (GY2, "GPU 两次之间（噪声）")])
    PH = 344      # R3：右格底注原来贴着面板下框
    py = f.panel(0, y0, 560, PH, "按参数分组，第 300 步（实测）", BL)
    rows = [("输出层", N.GROUP_HEAD_300), ("其余全部", N.GROUP_TOTAL_300 - N.GROUP_HEAD_300)]
    for i, (lab, v) in enumerate(rows):
        y = py + 60 + i * 110
        f.t(24, y + 24, lab, INK, True, 16)
        L = v / N.GROUP_TOTAL_300 * 330
        f.box(150, y, L, 40, BL, BL, 4)
        f.t(150 + L + 10, y + 26, "%.2e" % v, BL, True, 14)
    f.t(24, py + 290, "合计 %.2e，其余全部 ＝ 合计 − 输出层；GPU 两次之间输出层只差 6e-6" % N.GROUP_TOTAL_300, GY, size=13)
    f._pan = None
    px = 580
    py2 = f.panel(px, y0, W - px, PH, "输出层按词频分桶，第 300 步平方和差（实测）", BL)
    for i, (lab, v) in enumerate(N.ROWS_300):
        y = py2 + 36 + i * 68
        f.t(px + 24, y + 24, lab, INK, True, 15)
        L = max(v / 17.1 * 500, 3)
        # R3：「约 0」那根原来用了噪声灰，会被读成噪声 → 跟其它桶一样用 TPU − GPU 的蓝
        f.box(px + 170, y + 4, L, 34, BL, BL, 4)
        f.box(px + 170, y + 40, N.ROWS_GPU_PAIR_MAX / 17.1 * 500, 6, GY2, GY2, 2)
        f.t(px + 170 + L + 10, y + 28, "＋%.1f" % v, BL, True, 14)
    f.t(px + 24, py2 + 300, "没出现过的词一共 %s 行；灰色细条 ＝ GPU 两次之间每桶的差，都不超过 %.2f" % (format(N.RARE_ROWS, ","), N.ROWS_GPU_PAIR_MAX), GY, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "TPU 上冷门词的行被压得少：从初值算起 GPU 缩了 66.3，TPU 只缩了 49.2", [
        "权重衰减两边一样，都贡献 −31.8；差全在 Adam 更新造成的那部分收缩：GPU −34.5，TPU 只有 −17.3。",
        "代价：知道了「在哪」，还不知道「为什么」—— 冷门行的梯度大小两边一样，差在方向。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：分组参数平方和、逐行平方和都按 FP32 主权重算，两边组名一致；第 100 / 200 / 300 步输出层贡献 ＋4.1e-4 / ＋1.85e-3 / ＋3.83e-3，逐步变大。",
               ("📌 换算（本课推导）：参数范数 ＝ √S（S ＝ 全部参数平方和），S 多出 ΔS 时参数范数多出约 ΔS ÷ (2 × 参数范数)。输出层平方和差合计 ＋%.1f ÷（2 × %d）≈ %.2e，跟左边输出层的 %.2e 差约 %.1f%%。"
               % (N.ROWS_300_TOTAL, round(P0), ROWS_CONV, N.GROUP_HEAD_300, abs(ROWS_CONV / N.GROUP_HEAD_300 - 1) * 100)).replace("e-0", "e-"),
               "📌 实测：没出现过的词那些行，√v̂ 均值两边只差 0–4%，Adam 更新的大小只差 1%–3%；Σ w·u（更新在这一行权重方向上的分量，加起来）从第 25 步起就分开。")
    f.save("fig9-rows.svg", yb + 14)


GS = [10 ** (-11 + 0.05 * i) for i in range(121)]          # 1e-11 … 1e-5
U = [g / (g + N.EPS) for g in GS]
ELAST = {g: N.EPS / (g + N.EPS) for g in (N.RARE_G_LO, 1e-6)}
assert abs(ELAST[N.RARE_G_LO] - 0.714) < 0.001 and abs(ELAST[1e-6] - 0.0099) < 0.0001


def fig_eps():
    f = Fig(W, "横轴是一个参数的梯度大小，对数刻度；纵轴是 Adam 给它的更新量，按学习率归一化。"
               "梯度远大于 ε 时，更新被拉平到 1 附近，梯度差 1%，更新只差 0.01%；"
               "梯度和 ε 差不多或更小时，曲线变成斜坡，更新几乎跟梯度成正比，梯度差 1%，更新差约 0.7%。"
               "冷门词那些行的梯度在 4e-9 到 1e-8，正好落在斜坡上。")
    y0 = f.header("Adam 的 ε 区：梯度太小，Adam 就不再「拉平」　——　<tspan font-weight=\"700\">一点偏差原样漏过去</tspan>",
                  "横轴 ＝ 一个参数的梯度大小（对数）；纵轴 ＝ Adam 这一步的更新量（÷ 学习率）；稳态下更新 ≈ g ÷ (|g| ＋ ε)，ε ＝ 1e-8",
                  [(INK, "Adam 的更新"), (RD, "冷门词的行"), (GR, "常见词的行")])
    PH = 380
    py = f.panel(0, y0, W, PH, "更新量 ≈ g ÷ (|g| ＋ ε)（本课推导，稳态近似）", GY)
    X0, XW, YB, YH = 120, 1150, py + 300, 250

    def X(g):
        return X0 + (math.log10(g) + 11) / 6.0 * XW

    def Y(u):
        return YB - u * YH
    f.line(X0, YB, X0 + XW, YB, GY2, 1.2)
    f.line(X0, YB, X0, YB - YH - 10, GY2, 1.2)
    for e in range(-11, -4):
        f.t(X(10 ** e), YB + 22, "1e%d" % e, GY, size=12, anchor="middle")
    for u in (0, 0.5, 1.0):
        f.t(X0 - 10, Y(u) + 4, "%g" % u, GY, size=12, anchor="end")
    f.box(X(N.RARE_G_LO), Y(1.0) - 6, X(N.RARE_G_HI) - X(N.RARE_G_LO), YH + 6, "#fce8e6", "#fce8e6", 0)
    f.box(X(3e-7), Y(1.0) - 6, X(3e-6) - X(3e-7), YH + 6, "#e6f4ea", "#e6f4ea", 0)
    f.path([(X(g), Y(u)) for g, u in zip(GS, U)], INK, 3, arrow=False)
    f.line(X(N.EPS), YB, X(N.EPS), Y(1.0) - 10, GY2, 1.4, dash="5 5", arrow=False)
    f.t(X(N.EPS) + 6, Y(1.0) - 14, "ε ＝ 1e-8", GY, True, 13)
    f.t(X(N.RARE_G_LO) - 10, Y(0.9), "冷门词的行", RD, True, 15, "end")
    f.t(X(N.RARE_G_LO) - 10, Y(0.9) + 22, "梯度差 1%% → 更新差 %.1f%%" % ELAST[N.RARE_G_LO], RD, size=13, anchor="end")
    f.t(X(3e-6) + 10, Y(0.6), "常见词的行", GR, True, 15)
    f.t(X(3e-6) + 10, Y(0.6) + 22, "梯度差 1%% → 更新差 %.2f%%" % ELAST[1e-6], GR, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "常见词的行，Adam 把每一步拉成差不多大，梯度里的小偏差被拉平；冷门词的行不会", [
        "这些行每一步只收到 softmax 一点点「往下压」的梯度，比 ε 还小：在这里更新几乎跟梯度成正比，梯度的方向一变，更新就跟着变，Adam 不会把它拉平。",
        "所以同样一点梯度上的差，常见词的行几乎不受影响，冷门词的行原样收下。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导：Adam 更新 u ＝ m̂ ÷ (√v̂ ＋ ε)。梯度稳定时 m̂ ≈ g、√v̂ ≈ |g|，u ≈ g ÷ (|g| ＋ ε)；弹性 (du/u) ÷ (dg/g) ＝ ε ÷ (|g| ＋ ε)：g ＝ 4e-9 时 0.71，g ＝ 1e-6 时 0.0099。",
               "📌 实测：没出现过的词那些行，第 200–300 步 √v̂ 的均值在 4e-9 到 1e-8（第 1 步梯度中位数 5.2e-8，12%% 小于 ε）；第 300 步有 %.0f%% 的元素 √v̂ ＜ ε，中位数 %s；" % (N.RARE_EPS_FRAC * 100, ("%.1e" % N.RARE_VHAT_MED).replace("e-0", "e-")) + "β₁ ＝ 0.9、β₂ ＝ 0.95、ε ＝ 1e-8；没有出现 FP32 下溢或非正规数。",
               "注：绿色那一段「常见词的行」只是示意它们的梯度远大于 ε，具体位置没有画实测值。")
    f.save("fig9-eps.svg", yb + 14)


fig_drift()
fig_rows()
fig_eps()
