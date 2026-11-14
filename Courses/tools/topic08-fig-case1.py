# -*- coding: utf-8 -*-
r"""专题八 · 第七节「案例（上）」的三张图。

fig8-drift  回答：loss 偏移之外，哪个每步都记的量也在慢慢长？—— 参数范数：TPU 从约第 100 步起近似线性地多涨。
            数据：topic08_series.json 的真实逐步值（TPU − GPU 第一份；GPU 第二份 − 第一份当噪声）。
fig8-rows   回答：多涨的那一截在哪？—— 几乎全在输出层，而且是数据里没出现过的那些词的行。
            数字：第 300 步分组与按词频分桶的平方和差（实测，见 topic08_numbers）。
fig8-eps    回答：为什么偏偏是冷门词的行这么敏感？—— 它们的梯度比 Adam 的 ε 还小，在那儿 Adam 不再「归一化」。
            现算：稳态下 m̂ ≈ g、v̂ ≈ g²，更新 u ＝ g ÷ (|g| ＋ ε)；弹性 (du/u)÷(dg/g) ＝ ε ÷ (|g| ＋ ε)（本课推导）。
            ⛔ 刻意没画：真实的 m̂、v̂ 不是稳态，这条曲线只讲「在 ε 附近 Adam 怎么变」，不拿来算具体的更新量。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU
Fig.BAND_KEEP_LINES = 0   # 锤炼阶段（2026-11-14）：图上落点带只留一行粗体结论，其余进图下折叠
import topic08_numbers as N

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


LT = S["loss_rel_tpu_vs_gpu_a"]
LG = [S["loss_rel_gpu_b_vs_gpu_a"], S["loss_rel_gpu_c_vs_gpu_a"]]
PGS = [S["pnorm_diff_gpu_b_minus_gpu_a"], S["pnorm_diff_gpu_c_minus_gpu_a"]]
LT_TAIL = sum(LT[300:1000]) / 700.0
LG_TAIL = [sum(g[300:1000]) / 700.0 for g in LG]
assert abs(LT_TAIL - (-1.66e-4)) < 0.01e-4, LT_TAIL                 # 第 301–1000 步，同 fig-dt
assert abs(LG_TAIL[0] - (-3.78e-5)) < 0.02e-5 and abs(LG_TAIL[1] - 2.5e-7) < 0.2e-7, LG_TAIL   # 同 GPU_RUNS10_TAIL 第 2、3 份
assert abs(PT[99] - 3.9e-4) < 0.1e-4 and abs(PT[299] - 4.6e-3) < 0.1e-3                       # 第 100 ／ 300 步
PG_MAX = max(abs(x) for g in PGS for x in g)
assert abs(PG_MAX - 9.9e-4) < 0.1e-4, PG_MAX                                                # 两份 GPU 里最大 9.9e-4（记录分辨率 6 格）


def _roll(x, w=50):
    return [sum(x[k - w + 1:k + 1]) / w for k in range(w - 1, len(x))]


def fig_drift():
    f = Fig(W, "上下两格共用横轴第 1 到 1000 步。上格：主 loss 的逐步相对差，TPU 对 GPU 基线（蓝），另两份 GPU 重跑对同一基线（灰），细线是逐步值，粗线是 50 步滑动平均。"
               "灰色粗线围着零转；蓝色粗线从第 100 步前后往下离开，第 300 步以后平均约 −1.7e-4。"
               "下格：参数范数的逐步绝对差，同样两条灰线贴着零；蓝线从第 100 步前后往上，一路涨到第 1000 步的 ＋0.021。")
    y0 = f.header("loss 往下、参数范数往上：两条线都从第 100 步前后离开噪声",
                  "同一条时间轴；对照都是同一份 GPU 基线；灰 ＝ 另两份 GPU 重跑对同一基线，当噪声",
                  [(GY2, "GPU 重跑 − 基线（细：逐步；粗：50 步平均）"), (BL, "TPU − 基线")])
    X0, XW = 120, 1230

    def X(st):
        return X0 + (st - 1) / 999.0 * XW
    # 上格：loss
    PH1 = 300
    py = f.panel(0, y0, W, PH1, "主 loss 的逐步相对差 dₜ ＝（被测 − 基线）÷ 基线", BL)
    lo, hi = -6e-4, 4e-4
    YT, YH = py + 18, PH1 - 70

    def Y1(v):
        return YT + (hi - v) / (hi - lo) * YH
    f.line(X0, Y1(0), X0 + XW, Y1(0), INK, 1.2, arrow=False)
    for v in (-4e-4, -2e-4, 2e-4):
        f.t(X0 - 10, Y1(v) + 4, "%+.0f×10⁻⁴" % (v * 1e4), GY, size=12, anchor="end")
    f.t(X0 - 10, Y1(0) + 4, "0", INK, True, 12, "end")
    for g in LG:
        f.p.append('<polyline points="%s" fill="none" stroke="#9aa0a6" stroke-width="0.7" stroke-opacity="0.45"/>'
                   % " ".join("%.1f,%.1f" % (X(k + 1), Y1(max(lo, min(hi, v)))) for k, v in enumerate(g)))
    f.p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="0.8" stroke-opacity="0.45"/>'
               % (" ".join("%.1f,%.1f" % (X(k + 1), Y1(max(lo, min(hi, v)))) for k, v in enumerate(LT)), BL))
    for g in LG:
        f.p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>'
                   % (" ".join("%.1f,%.1f" % (X(k + 50), Y1(v)) for k, v in enumerate(_roll(g))), GY))
    f.p.append('<polyline points="%s" fill="none" stroke="#0b57d0" stroke-width="3.4"/>'
               % " ".join("%.1f,%.1f" % (X(k + 50), Y1(v)) for k, v in enumerate(_roll(LT))))
    f.line(X(301), Y1(LT_TAIL), X(1000), Y1(LT_TAIL), INK, 1.4, dash="6 4", arrow=False)
    _lab = "第 301–1000 步平均 %s（两份 GPU：%s、%s）" % (N.tx(LT_TAIL), N.tx(LG_TAIL[0]), N.tx(LG_TAIL[1]))
    f.box(X(1000) - 470, Y1(-5.2e-4) - 17, 470, 24, "#fff", INK, 5)
    f.line(X(1000) - 235, Y1(-5.2e-4) - 17, X(1000) - 235, Y1(LT_TAIL), INK, 1.0, arrow=False)
    f.t(X(1000) - 8, Y1(-5.2e-4), _lab, INK, True, 13.5, "end")
    f._pan = None
    # 下格：参数范数
    PH2 = 300
    py2 = f.panel(0, y0 + PH1 + 14, W, PH2, "参数范数的逐步差（全部 FP32 主权重平方和开根号，约 %d；绝对差）" % round(P0), BL)
    lo2, hi2 = -0.003, 0.023
    YT2, YH2 = py2 + 18, PH2 - 76

    def Y2(v):
        return YT2 + (hi2 - v) / (hi2 - lo2) * YH2
    f.line(X0, Y2(0), X0 + XW, Y2(0), INK, 1.2, arrow=False)
    for v in (0.01, 0.02):
        f.t(X0 - 10, Y2(v) + 4, "%+.2f" % v, GY, size=12, anchor="end")
    f.t(X0 - 10, Y2(0) + 4, "0", INK, True, 12, "end")
    for g in PGS:
        f.path([(X(k + 1), Y2(v)) for k, v in enumerate(g)], GY, 1.6, arrow=False)
    f.path([(X(k + 1), Y2(v)) for k, v in enumerate(PT)], BL, 3.0, arrow=False)
    f.t(X(1000) - 6, Y2(PT[999]) - 12, "第 1000 步 ＋%.3f（GPU 自己一千步涨 %.3f）" % (PT[999], N.PN_GPU_GROWTH), BL, True, 15, "end")
    f.t(X(620), Y2(0) + 24, "两份 GPU 重跑：最多差 %.1e，贴着零" % PG_MAX, GY, True, 13, "middle")
    for st in (1, 100, 300, 500, 1000):
        f.t(X(st), py2 + PH2 - 32, "第 %d 步" % st, GY, size=12, anchor={1: "start", 1000: "end"}.get(st, "middle"))
    f._pan = None
    f.line(X(100), y0 + 34, X(100), py2 + PH2 - 44, OR, 1.6, dash="5 5", arrow=False)
    f.t(X(100) + 8, y0 + 50, "约第 100 步：两条一起离开噪声", OR, True, 13)
    yb = f.band(py2 + PH2 + 20, "ok", "偏移是一步步攒出来的：loss 往下、参数范数往上，同一时间离开噪声", [
        "第 1 步两边的更新量没有差别；loss 三百步以后走平，参数范数一路还在涨。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：混元 3 截前 4 层，三份默认设置的 GPU 运行（一份当基线）和 TPU 第一次 1000 步运行；逐步原值来自对齐记录的逐步曲线文件。主 loss 第 301–1000 步带符号均值：TPU %s，另两份 GPU %s、%s。参数范数：GPU 基线从 %.3f 涨到 %.3f；TPU − 基线第 100 ／ 300 ／ 1000 步 ＋%.1e ／ ＋%.1e ／ ＋%.1e。"
               % (N.tx(LT_TAIL), N.tx(LG_TAIL[0]), N.tx(LG_TAIL[1]), P0, P1, PT[99], PT[299], PT[999]),
               "📌 两边记录的格距不同：GPU 侧是 FP32 平方和开根号，平方和（约 925 万）一格是 1，折成参数范数约 1 ÷ (2 × 3042) ≈ 1.6e-4，GPU 之间的差都是它的整数倍；TPU 侧记的就是 FP32 的参数范数，3042 附近一格 2.4e-4。",
               "⚠️ 粗线是 50 步滑动平均，只为看方向；判定见第三节。")
    f.save("fig8-drift.svg", yb + 14)


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
    rows = [("输出层", N.GROUP_HEAD_300), ("合计减输出层", N.GROUP_TOTAL_300 - N.GROUP_HEAD_300)]
    for i, (lab, v) in enumerate(rows):
        y = py + 60 + i * 110
        f.t(24, y + 24, lab, INK, True, 16)
        L = v / N.GROUP_TOTAL_300 * 330
        f.box(150, y, L, 40, BL, BL, 4)
        f.t(150 + L + 10, y + 26, ("%.2e" % v).replace("e-0", "e-"), BL, True, 14)
    f.t(24, py + 290, "合计 %.2e；下面那一截大半是按组线性拆分的残差，不是别的参数组的贡献；GPU 两次之间输出层只差 6e-6" % N.GROUP_TOTAL_300, GY, size=13)
    # 第二轮：输出层贡献随步数（只有第 100／200／300 步三个实测点）
    _pts = N.GROUP_HEAD_STEPS
    _sx = lambda st: 300 + (st - 100) / 200.0 * 200
    _sy = lambda v: py + 258 - v / 4e-3 * 36
    f.t(290, py + 212, "输出层贡献随步数（3 个实测点）", GY, True, 12.5)
    f.path([(_sx(a_), _sy(b_)) for a_, b_ in _pts], BL, 2, arrow=False)
    for a_, b_ in _pts:
        f.box(_sx(a_) - 4, _sy(b_) - 4, 8, 8, BL, BL, 4)
        f.t(_sx(a_), py + 272, "第 %d 步" % a_, GY, size=11.5, anchor="middle")
    f.t(_sx(300) + 8, _sy(_pts[-1][1]) + 18, N.tx(_pts[-1][1], "%.2e"), BL, True, 11.5, "end")
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
               "📌 实测：分组参数平方和、逐行平方和都按 FP32 主权重算，两边组名一致；第 100 / 200 / 300 步输出层贡献 %s，逐步变大。" % " / ".join(N.tx(v, "%.2e", plus=True) for _s, v in N.GROUP_HEAD_STEPS),
               ("📌 换算（本课推导）：参数范数 ＝ √S（S ＝ 全部参数平方和），S 多出 ΔS 时参数范数多出约 ΔS ÷ (2 × 参数范数)。输出层平方和差合计 ＋%.1f ÷（2 × %d）≈ %.2e，跟左边输出层的 %.2e 差约 %.1f%%。"
               % (N.ROWS_300_TOTAL, round(P0), ROWS_CONV, N.GROUP_HEAD_300, abs(ROWS_CONV / N.GROUP_HEAD_300 - 1) * 100)).replace("e-0", "e-"),
               "📌 实测：没出现过的词那些行，√v̂ 均值两边只差 0–4%，Adam 更新的大小只差 1%–3%；Σ w·u（更新在这一行权重方向上的分量，加起来）从第 25 步起就分开。")
    f.save("fig8-rows.svg", yb + 14)


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
    PH = 450
    py = f.panel(0, y0, W, PH, "更新量 ≈ g ÷ (|g| ＋ ε)（本课推导，稳态近似）；下面那条是本例冷门行的实测", GY)
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
    # 实测层：第 300 步没出现过的词那些行的 √v̂（只有两个统计量：中位数、ε 以下的比例；不是完整分布）
    sy = YB + 44
    lft = N.RARE_EPS_FRAC
    f.box(X0, sy, X(N.EPS) - X0, 22, RD, RD, 3)
    f.box(X(N.EPS), sy, (X(N.EPS) - X0) * (1 - lft) / lft * 0.35, 22, "#f6aea9", "#f6aea9", 3)
    f.t(X0 + 10, sy + 16, "第 300 步，冷门行 √v̂：%.0f%% 的元素在 ε 左边（实测）" % (lft * 100), "#fff", True, 13)
    mx = X(N.RARE_VHAT_MED)
    f.line(mx, sy - 6, mx, sy + 30, INK, 2.2, arrow=False)
    f.t(mx, sy + 46, "中位数 %s，约为 ε 的四分之一" % ("%.1e" % N.RARE_VHAT_MED).replace("e-0", "e-"), INK, True, 13, "middle")
    f.t(X(N.EPS) + (X(N.EPS) - X0) * (1 - lft) / lft * 0.35 + 8, sy + 16, "其余 %.0f%%" % ((1 - lft) * 100), GY, size=12)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "常见词的行，Adam 把每一步拉成差不多大，梯度里的小偏差被拉平；冷门词的行不会", [
        "这些行每一步只收到 softmax 一点点「往下压」的梯度，比 ε 还小：在这里更新几乎跟梯度成正比，梯度的方向一变，更新就跟着变，Adam 不会把它拉平。",
        "所以同样一点梯度上的差，常见词的行几乎不受影响，冷门词的行原样收下。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导：Adam 更新 u ＝ m̂ ÷ (√v̂ ＋ ε)。梯度稳定时 m̂ ≈ g、√v̂ ≈ |g|，u ≈ g ÷ (|g| ＋ ε)；弹性 (du/u) ÷ (dg/g) ＝ ε ÷ (|g| ＋ ε)：g ＝ 4e-9 时 0.71，g ＝ 1e-6 时 0.0099。",
               "📌 实测：没出现过的词那些行，第 200–300 步 √v̂ 的均值在 4e-9 到 1e-8（第 1 步梯度中位数 5.2e-8，12%% 小于 ε）；第 300 步有 %.0f%% 的元素 √v̂ ＜ ε，中位数 %s；" % (N.RARE_EPS_FRAC * 100, ("%.1e" % N.RARE_VHAT_MED).replace("e-0", "e-")) + "β₁ ＝ 0.9、β₂ ＝ 0.95、ε ＝ 1e-8；没有出现 FP32 下溢或非正规数。",
               "注：绿色那一段「常见词的行」只是示意它们的梯度远大于 ε，具体位置没有画实测值。最下面那条实测只有三样是量到的：ε 分界、ε 以下的比例 92%、中位数；条的两端长度是示意，不是分布。")
    f.save("fig8-eps.svg", yb + 14)


fig_drift()
fig_rows()
fig_eps()
