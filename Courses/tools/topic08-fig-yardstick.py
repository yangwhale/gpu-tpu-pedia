# -*- coding: utf-8 -*-
r"""专题八 · 第三节补的两张「尺子」图（第二次迁移的新材料）。

fig8-loo   回答：r 落在 1–2 之间，算不算出事？—— 先看纯噪声自己能到多少。
           GPU 同设置的几份重跑，轮流挑一份当被测、一份当基线，尺子用其余几份重算，判法跟 TPU 被测一模一样；
           一次比较要看 8 项检查（7 条曲线，主 loss 两个窗口）× 3 个统计 ＝ 24 个数，取最差的 —— 纯噪声也常常「略超」。
           第三行：8 份尺子时被推过 2 倍的六个 TPU 改动，补到 16 份后全部回到 1–2 之间。
           数据：topic08_tpuself.json（导出自第二次迁移的对齐记录：TPU 自比的 GPU 留一法、长跑灰区表），全部实测。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE
Fig.BAND_KEEP_LINES = 0   # 锤炼阶段（2026-11-14）：图上落点带只留一行粗体结论，其余进图下折叠
import topic08_numbers as N

W = 1400
T = N.tpuself()
L300, L1K = T["loo_300_n8"], T["loo_1000_n16"]
GZ = T["grayzone_1000"]
# 锚点：跟对齐记录的汇总数对上（实测）
assert L300["n_combos"] == 56 and len(L300["worst_r"]) == 56
assert L1K["n_combos"] == 240 and len(L1K["worst_r"]) == 240
assert abs(L300["max"] - 2.47) < 0.005 and abs(L1K["max"] - 1.82) < 0.005
assert abs(L300["frac_gt_1"] - 0.59) < 0.005 and abs(L300["frac_gt_2"] - 0.11) < 0.005
assert abs(L1K["frac_gt_1"] - 0.22) < 0.005 and L1K["frac_gt_2"] == 0
assert len(GZ) == 6 and min(g["r1000_n8"] for g in GZ) == 2.00 and max(g["r1000_n8"] for g in GZ) == 2.75
assert abs(min(g["r1000_n16"] for g in GZ) - 1.19) < 1e-9 and abs(max(g["r1000_n16"] for g in GZ) - 1.64) < 1e-9   # 2026-11-10 分母统一为参照后的值
LOO8_1000_MAX = 2.00        # 1000 步只有 8 份时的留一法最大值（对齐记录「GPU 重跑的留一法 r」表，已被 16 份取代）
YEL = "#f9ab00"


def fig_loo():
    f = Fig(W, "一根横轴是 r，0 到 1 绿色判过，1 到 2 黄色判待定，大于 2 红色判不过。"
               "第一行：GPU 同设置 8 份、300 步，轮流一份当被测、一份当基线、其余当尺子，一共 56 种，一个点一种；"
               "59% 超过 1，其中 11% 超过 2，最大 2.47。第二行：1000 步、16 份，240 种，22% 在黄区，没有进红区的，最大 1.82。"
               "第三行：六个 TPU 改动跑 1000 步，用 8 份的尺子时 r 在 2.00 到 2.75，压线或越过当时纯噪声的最大值 2.00；"
               "尺子补到 16 份后落到 1.19 到 1.64，都在纯噪声最大值 1.82 以内。")
    y0 = f.header("纯噪声互相比，也常常超线",
                  "一个点 ＝ GPU 重跑里挑一份当被测、一份当基线、其余当尺子；判法跟 TPU 一样：几十个 r 取最差的（300 步 24 个，1000 步 27 个）",
                  [(GY2, "GPU 对 GPU（纯噪声）"), (BL, "TPU 改动：空心 ＝ 8 份尺子，实心 ＝ 16 份尺子")])
    X0, XW, RMAX = 250, 1110, 3.0

    def X(r):
        return X0 + min(r, RMAX) / RMAX * XW

    rows = [("300 步 · GPU 8 份", "56 种轮换", L300, 120),
            ("1000 步 · GPU 16 份", "240 种轮换", L1K, 220)]
    PH = sum(h for *_x, h in rows) + 150 + 46 + 100
    py = f.panel(0, y0, W, PH, "同一把判法，先拿纯噪声试一遍（实测）", GY)
    ytop, ybot = py + 12, py + PH - 100
    for a, b, fill in [(0, 1, "#e6f4ea"), (1, 2, "#fef7e0"), (2, RMAX, "#fce8e6")]:
        f.box(X(a), ytop, X(b) - X(a), ybot - ytop, fill, "none", 0)
    for r in (0, 1, 2, 3):
        f.line(X(r), ytop, X(r), ybot, GY2 if r else LINE, 1.0, arrow=False)
        f.t(X(r), ybot + 22, "%d" % r, GY, size=14, anchor="middle")
    f.t(X(3) - 4, ybot + 46, "r ＝ 被测的差 ÷ 尺子 N", GY, size=13, anchor="end")
    for a, b, lab, col in [(0, 1, "过", GR), (1, 2, "待定", YEL), (2, RMAX, "不过", RD)]:
        f.t((X(a) + X(b)) / 2, ytop + 20, lab, col, True, 15, "middle")

    y = ytop + 34
    DOT, STEP = 2.5, 5.6          # 点半径、蜂群里点与点的间距（像素）
    for name, sub, d, h in rows:
        yc = y + h / 2.0
        f.t(24, yc - 4, name, INK, True, 14)
        f.t(24, yc + 18, sub, GY, size=13)
        # 蜂群：按 x 分箱，一箱里的点上下交替往外叠（确定性，不用随机抖动）
        bins = {}
        for v in d["worst_r"]:
            bins.setdefault(int(X(v) // STEP), []).append(v)
        tallest = max(len(v) for v in bins.values())
        assert (tallest // 2 + 1) * STEP < h / 2.0, "蜂群叠太高：%d 个点一箱" % tallest
        for k, vs in bins.items():
            for j, v in enumerate(sorted(vs)):
                off = ((j + 1) // 2) * STEP * (1 if j % 2 else -1)
                f.box(X(v) - DOT, yc + off - DOT, 2 * DOT, 2 * DOT, GY2, GY2, DOT)
        mx = d["max"]
        f.line(X(mx), yc - h / 2.0 + 6, X(mx), yc + h / 2.0 - 6, INK, 1.4, dash="4 3", arrow=False)
        f.t(X(mx) + 8, yc - h / 2.0 + 22, "纯噪声最大 %.2f" % mx, INK, True, 14)
        f.t(X(mx) + 8, yc - h / 2.0 + 42, ("%d%% 过 1，%d%% 过 2" % (round(d["frac_gt_1"] * 100), round(d["frac_gt_2"] * 100))) if d["frac_gt_2"] else ("%d%% 过 1，没有过 2 的" % round(d["frac_gt_1"] * 100)), GY, size=13)
        y += h

    # 第三行：六个 TPU 改动，8 份尺子 → 16 份尺子
    h3 = 150
    f.t(24, y + h3 / 2.0 - 4, "1000 步 · TPU 六个改动", BL, True, 14)
    f.t(24, y + h3 / 2.0 + 18, "8 份时压线或越过 2 的", GY, size=13)
    f.t(24, y + h3 / 2.0 + 38, "TPU 没重跑，只换了尺子", BL, size=13)
    f.line(X(LOO8_1000_MAX), y + 6, X(LOO8_1000_MAX), y + h3 - 6, GY, 1.2, dash="2 3", arrow=False)
    f.t(X(LOO8_1000_MAX) + 6, y + h3 - 10, "8 份时纯噪声最大 2.00", GY, size=13)
    f.line(X(L1K["max"]), y + 6, X(L1K["max"]), y + h3 - 6, INK, 1.4, dash="4 3", arrow=False)
    for i, g in enumerate(sorted(GZ, key=lambda g: -g["r1000_n8"])):
        yy = y + 18 + i * 20
        x8, x16 = X(g["r1000_n8"]), X(g["r1000_n16"])
        f.box(x8 - 5, yy - 5, 10, 10, "#fff", BL, 5, sw=1.6)
        f.line(x8 - 8, yy, x16 + 9, yy, BL, 1.4)
        f.box(x16 - 5, yy - 5, 10, 10, BL, BL, 5)
    f.t(X(2.75), y + h3 - 10, "8 份尺子：2.00–2.75", BL, size=13, anchor="middle")
    f.t(X(1.18) - 10, y + h3 - 10, "16 份尺子：1.19–1.64", BL, True, 13, "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "r 在 1–2 之间，先别急着判错：纯噪声自己就常常到这里", [
        "300 步 8 份：近一半落进待定、一成进不过；1000 步 16 份：两成落进待定，最高 1.82。份数越多，纯噪声越难超线。",
        "代价：份数少，尺子就窄（8 份时六个 TPU 改动压线或越过 2）；跟重跑分不开也不等于没改精度。这把「纯噪声最大值」是事后才量的。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸模型）：GPU 同设置 300 步 8 份（56 种轮换，每种尺子 21 对）、1000 步 16 份（240 种，每种 105 对）；"
               "每种轮换：一份当被测、一份当基线，尺子用其余各份两两重算，检查跟 TPU 自比相同（主 loss 全程与后段、MTP、索引器蒸馏、梯度范数、参数范数、路由偏置均值与标准差；1000 步多一条 MTP 后段）。"
               "300 步 59% 的最差 r 过 1（其中 11% 过 2，落在 1–2 之间的是 48%），最大 2.47；1000 步 16 份 22% 过 1、没有过 2 的，最大 1.82；1000 步只有 8 份时最大 2.00、64% 过 1。",
               "📌 第三行：TPU 自比 1000 步对 TPU golden；八份尺子的上限比十六份的窄 1.0–2.1 倍，多数检查里放大最多的是带符号均值（路由偏置均值 2.07 倍、梯度范数 1.84 倍）。"
               "六个改动：少开投影 FP32 累加 2.75 → 1.64、关矩阵乘之间的融合 2.50 → 1.49、SwiGLU 少舍一次 2.34 → 1.37、每层多记几个统计量 2.33 → 1.40、"
               "16 设备纯数据并行 ＋ FP32 词嵌入梯度 2.07 → 1.30、TPU 运行库降一版 2.00 → 1.19（16 份一列是 2026-11-10 曲线分母统一为参照以后的值；8 份一列是统一之前判的）；其中两个只换顺序，两个动了精度。",
               "⚠️ 口径：「GPU 轮换（留一法）最大值」是看到 TPU 结果以后才补的解读，只用来读表，不改判档；对齐记录把它事先登记了，再拿后来的新改动和新尺寸检验。")
    f.save("fig8-loo.svg", yb + 14)


import math  # noqa: E402
import random  # noqa: E402

# ── fig8-integrator：参数范数为什么灵（本课推导 ＋ 示意模拟）──────────────────────────
#   每一步给一个随机增量（噪声，标准差 1）；被测另外每步多一个固定的小偏 b。累加起来：
#   噪声的累加按 √t 长（正负抵消），偏差按 t 长（同一个方向）。两条线在 b·t ＝ 2√t，即 t* ＝ (2/b)² 处相交。
#   参数范数正是这样一个累加器：‖θ‖ 每步的变化 ≈ 这一步更新沿 θ 方向的分量（本课推导，学习率小时成立）。
INT_B, INT_T, INT_WALKS, INT_SEED = 0.1, 1000, 12, 13
INT_TSTAR = (2 / INT_B) ** 2
assert INT_TSTAR == 400


def _walk(rng, b):
    s, out = 0.0, [0.0]
    for _ in range(INT_T):
        s += rng.gauss(0, 1) + b
        out.append(s)
    return out


def fig_integrator():
    rng = random.Random(INT_SEED)
    noise = [_walk(rng, 0.0) for _ in range(INT_WALKS)]
    drift = _walk(rng, INT_B)
    # 示意要讲得通：被测那条在 t* 之后确实冲出噪声带、之前大多在带里（固定种子下核过）
    band = lambda t: 2 * math.sqrt(t)
    assert drift[INT_T] > band(INT_T) * 1.5
    assert sum(1 for t in range(1, 200) if abs(drift[t]) <= band(t)) > 190
    first_out = next(t for t in range(1, INT_T + 1) if all(drift[u] > band(u) for u in range(t, INT_T + 1)))
    assert 350 < first_out < 600, first_out
    f = Fig(W, "两块。左块：横轴是训练步数 0 到 1000，纵轴是累加起来的差。十二条灰色细线是只有随机噪声的累加，"
               "像醉汉走路，散开的宽度按步数的平方根长，灰色阴影是正负两倍平方根的噪声带。虚线 400 步是平均意义上偏差追上带边的地方，单条蓝线早一点晚一点出带都正常。"
               "蓝线每步多一个只有噪声十分之一大的固定偏差，前几百步埋在带里，四五百步以后冲出去，越走越远。"
               "右块：两支一样长的箭头指向不同方向，长度就是参数范数，所以范数分不出它们；"
               "下面两根条，一组多涨一点、一组少涨一点，平方和加起来不变。")
    y0 = f.header("参数范数是个累加器：噪声按根号长，偏差按步数长",
                  "左：示意模拟（固定种子），每步一个随机增量；蓝线每步另多一个固定的小偏，只有噪声的十分之一。右：范数只看长度，看不见方向",
                  [(GY2, "只有噪声"), (BL, "噪声 ＋ 每步一点固定偏差"), ("#e8eaed", "噪声带 ±2√t")])
    PH = 400
    LW = 860
    py = f.panel(0, y0, LW, PH, "Ⓐ 为什么灵：每步一点点，几百步冲出噪声带", BL)
    X0, XW, YM, YH = 70, LW - 120, py + 170, 140
    lim = max(max(abs(v) for v in w) for w in noise + [drift])

    def X(t):
        return X0 + t / float(INT_T) * XW

    def Y(v):
        return YM - v / lim * YH
    up = [(X(t), Y(band(t))) for t in range(0, INT_T + 1, 10)]
    dn = [(X(t), Y(-band(t))) for t in range(INT_T, -1, -10)]
    f.poly(up + dn, "#f1f3f4", "none")
    f.line(X0, YM, X0 + XW, YM, LINE, 1.0, arrow=False)
    for w in noise:
        f.path([(X(t), Y(w[t])) for t in range(0, INT_T + 1, 5)], GY2, 0.9, arrow=False)
    f.path([(X(t), Y(drift[t])) for t in range(0, INT_T + 1, 5)], BL, 2.4, arrow=False)
    f.line(X(INT_TSTAR), py + 18, X(INT_TSTAR), YM + YH + 6, INK, 1.2, dash="4 3", arrow=False)
    f.t(X(INT_TSTAR) + 8, py + 34, "0.1·t ＝ 2√t：平均在 400 步追上带边", INK, True, 13)
    f.t(X(INT_T) - 4, Y(drift[INT_T]) - 12, "偏差：∝ t", BL, True, 14, "end")
    f.t(X(INT_T) - 4, Y(band(INT_T)) + 26, "噪声：∝ √t", GY, True, 14, "end")
    for t in (0, 250, 500, 750, 1000):
        f.t(X(t), YM + YH + 28, "%d" % t, GY, size=13, anchor="middle")
    f.t(X0 + XW, YM + YH + 50, "训练步", GY, size=13, anchor="end")
    f._pan = None
    RX = LW + 24
    py2 = f.panel(RX, y0, W - RX, PH, "Ⓑ 盲区：只看长度，不看方向", RD)
    cx, cy, R = RX + 80, py2 + 175, 120
    f.path("M %d %d A %d %d 0 0 0 %d %d" % (cx + R, cy, R, R, cx, cy - R), LINE, 1.2, dash="3 3", arrow=False)
    for ang, col, lab in [(18, GY, "GPU"), (62, BL, "TPU")]:
        a = math.radians(ang)
        f.line(cx, cy, cx + R * math.cos(a), cy - R * math.sin(a), col, 2.4)
        f.t(cx + (R + 14) * math.cos(a), cy - (R + 14) * math.sin(a) + 5, lab, col, True, 14)
    f.box(cx - 4, cy - 4, 8, 8, INK, INK, 4)
    f.t(cx + R + 40, cy - 118, "一样长：范数相同", INK, True, 14)
    f.t(cx + R + 40, cy - 96, "指向不同：模型已经不一样", GY, size=13)
    by = py2 + 250
    for i, (lab, d) in enumerate([("A 组", +1), ("B 组", -1)]):
        y = by + i * 34
        f.t(RX + 24, y + 15, lab, INK, True, 13)
        if d > 0:
            f.box(RX + 80, y, 160, 22, "#e8eaed", "none", 3)
            f.box(RX + 240, y, 30, 22, OR, "none", 3)
        else:
            f.box(RX + 80, y, 130, 22, "#e8eaed", "none", 3)
            f.box(RX + 210, y, 30, 22, "none", BL, 3, sw=1.4, dash="3 2")
        f.t(RX + 290, y + 16, "多涨一点" if d > 0 else "少涨一点", GY, size=13)
    f.t(RX + 24, by + 92, "平方和加起来不变：总范数什么都看不出", RD, True, 13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "随机的差每一步正负抵消，固定方向的差一步步攒起来：累加器会把小偏差放大成看得见的漂移", [
        "混元 3 那次就是参数范数最先冲出去：定案前四项判定里它 3.45 倍，其余三项都在 2 倍以内。",
        "代价：范数只是一个长度。方向变了、两组反着偏，它都看不见，所以还要配第 1 步逐张量对 FP64。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导 ＋ 示意模拟：增量取标准正态（固定种子），蓝线每步多 0.1；设各步的差相互独立，噪声累加的标准差是 √t，偏差累加是 0.1·t（真实训练会把扰动放大，独立只是近似），"
               "两者在 0.1·t ＝ 2√t 处相交，t* ＝ 400。真实训练里 b 多大、噪声多宽各不相同，t* 只示意形状。"
               "参数范数是累加器：‖θ‖² 每步变 2θ·Δθ ＋ ‖Δθ‖²，学习率小时 ‖θ‖ 的变化约等于 Δθ 沿 θ 方向的分量。",
               "📌 实测锚点：混元 3 定案前第一关参数范数 r 3.45、loss 全程 1.92、尾段 1.29、梯度范数 1.14（判定图）。")
    f.save("fig8-integrator.svg", yb + 14)


# ── fig8-twoaxis：第 1 步逐张量，两把参照（实测）──────────────────────────────────
#   横：对 TPU golden 的相对差 ÷ GPU 第 1 步三份的尺子（＞ 1 ＝ 超出噪声）；
#   纵：被测离 CPU FP64 ÷ max(golden 离 FP64, GPU 三份中位数离 FP64)（＞ 1.2 ＝ 丢了精度；对齐记录 setup.md「定责」）。
S1 = T["step1_scatter"]
assert [a["label"] for a in S1] == ["v5p 换 v7x", "FSDP 改成纯数据并行", "关掉编译器的省舍入"]
assert all(a["n_tensors"] == 205 and a["n_small_skipped"] == 45 and len(a["points"]) == 160 for a in S1)
assert max(y for _x, y, _e in S1[0]["points"]) <= 1.2                      # 换 v7x：一个都没变远
_emb = [[(x, y) for x, y, e in a["points"] if e][0] for a in S1]
assert abs(_emb[1][1] - 3.62) < 0.005 and abs(_emb[2][1] - 26.13) < 0.005  # 词嵌入梯度：3.6 倍、26 倍（实验矩阵：纯数据并行、关省舍入两组）


def fig_twoaxis():
    f = Fig(W, "三块散点，每块是 TPU 上的一个改动。一个点是一个参数张量第 1 步的梯度，共 160 个。"
               "横轴是它跟 TPU 基线差多少，除以 GPU 重跑的差，大于 1 就超出噪声；纵轴是它离 64 位真值的距离，除以原来离真值的距离，大于 1.2 就是丢了精度。"
               "左块换到 v7x：148 个点超出噪声，可全部贴着纵轴 1，一个都没变远，是换了顺序。"
               "中块改成纯数据并行：只有词嵌入梯度一个红点冲到 R 3.62。右块关掉编译器的省舍入：词嵌入梯度冲到 R 26.1（离 FP64 是 golden 的 27 倍），另有一个路由门控权重 1.20 倍刚过线。")
    y0 = f.header("第 1 步逐张量：一把尺子量差多少，另一把量离真值多远",
                  "一个点 ＝ 一个参数张量的梯度　横：对 TPU 基线的差 ÷ GPU 重跑的差（＞ 1 超出噪声）　纵：离 CPU FP64 的距离 ÷ 原 TPU 基线和 GPU 里离得远的那个（＞ 1.2 丢了精度）",
                  [("#e6f4ea", "在噪声里"), ("#eceff3", "换了顺序：差变大，离真值没变远"), ("#fce8e6", "丢了精度"), (RD, "词嵌入梯度")])
    PW, GAP, PH = 448, 28, 400
    XL, XH, YL, YH = 0.1, 1e4, 0.15, 40.0

    for k, a in enumerate(S1):
        px = k * (PW + GAP)
        py = f.panel(px, y0, PW, PH, a["label"], BL if k == 0 else RD)
        ax0, ax1, ay0, ay1 = px + 58, px + PW - 18, py + 20, py + PH - 92

        def X(v):
            return ax0 + (math.log10(max(min(v, XH), XL)) - math.log10(XL)) / (math.log10(XH) - math.log10(XL)) * (ax1 - ax0)

        def Y(v):
            return ay1 - (math.log10(max(min(v, YH), YL)) - math.log10(YL)) / (math.log10(YH) - math.log10(YL)) * (ay1 - ay0)
        f.box(ax0, ay0, X(1) - ax0, ay1 - ay0, "#e6f4ea", "#e6f4ea", 0)
        f.box(X(1), Y(1.2), ax1 - X(1), ay1 - Y(1.2), "#eceff3", "#eceff3", 0)
        f.box(ax0, ay0, ax1 - ax0, Y(1.2) - ay0, "#fce8e6", "#fce8e6", 0)
        f.line(X(1), ay0, X(1), ay1, GY2, 1.0, arrow=False)
        f.line(ax0, Y(1.2), ax1, Y(1.2), RD, 1.0, dash="4 3", arrow=False)
        f.line(ax0, Y(1), ax1, Y(1), LINE, 0.8, arrow=False)
        for v in (0.1, 1, 10, 100, 1000, 10000):
            f.t(X(v), ay1 + 20, ("%g" % v) if v < 1000 else ("1e%d" % round(math.log10(v))), GY, size=13, anchor="middle")
        for v, lab in ((0.2, "0.2"), (1, "1"), (1.2, "1.2"), (5, "5"), (30, "30")):
            f.t(ax0 - 8, Y(v) + 5, lab, RD if v == 1.2 else GY, size=13, anchor="end")
        f.t((ax0 + ax1) / 2, ay1 + 42, "对基线的差 ÷ GPU 噪声", GY, size=13, anchor="middle")
        nover = sum(1 for x, _y, _e in a["points"] if x > 1)
        nfar = sum(1 for x, y, _e in a["points"] if y > 1.2)
        for x, y, e in a["points"]:
            if not e:
                f.box(X(x) - 2.6, Y(y) - 2.6, 5.2, 5.2, BL, BL, 2.6)
        for x, y, e in a["points"]:
            if e:
                f.box(X(x) - 6, Y(y) - 6, 12, 12, RD, INK, 6)
                f.t(X(x) - 12 if y > 1.2 else X(x) + 10, Y(y) + 5 if y > 1.2 else Y(y) - 8,
                    ("词嵌入梯度 %.1f 倍" % y) if y > 1.2 else "词嵌入", RD, True, 13, "end" if y > 1.2 else None)
        f.t(ax1 - 8, ay1 - 30, "超出噪声 %d／160" % nover, INK, True, 13, "end")
        f.t(ax1 - 8, ay1 - 10, ("离真值变远：%d" % nfar) if nfar else "离真值变远：0 → 换了顺序", RD if nfar else GR, True, 13, "end")
        f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "超出噪声不等于出错：离真值没变远，就是换了顺序；变远了，才是丢了精度", [
        "换 v7x，九成张量超出 GPU 的噪声，可一个都没离真值更远；纯数据并行、关省舍入，飞得远的都是词嵌入梯度 —— 查哪里，一眼就知道。",
        "代价：要一份 CPU 上 FP64 算的第 1 步梯度。多路残差混合的 45 个小参数本身病态，单列处理，没画。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸模型，TPU v5p 自比，运行库 0.0.46，对同一个 TPU golden）：第 1 步 205 个参数张量的训练梯度，其中多路残差混合的 45 个小参数（缩放、偏置、混合权重）已知病态、对齐记录单列已接受，不画；画 160 个。",
               "📌 横：被测对 golden 的 rel_L2 ÷ GPU 第 1 步三份两两的最大 rel_L2（这个张量自己的尺子）。纵：被测离 CPU FP64 的 rel_L2 ÷ max(golden 离 FP64, GPU 三份离 FP64 的中位数)；对齐记录的定责规则是 ≤ 1.2 记「换了顺序」，超过记「丢了精度」。",
               "📌 词嵌入梯度离 FP64：golden 4.00e-3、GPU 三份中位数 4.17e-3；纯数据并行 1.51e-2（R 3.62；对 golden 3.77 倍）；关 excess precision 1.09e-1（R 26.1；对 golden 27 倍）。关省舍入那一块另有第 4 层路由门控权重 1.20 倍，按规则也记丢了精度、定不了责。横轴大到上千的点，是 GPU 三份完全逐位相同的张量：尺子是 0，按一个很小的下限（2⁻²⁰）算。")
    f.save("fig8-twoaxis.svg", yb + 14)


# ── fig8-rpipe：照着算一遍 —— 一条序列、两个窗口、三种 reduce；从 K 份运行到一个 r ─────────────
#   Ⓐ 混元 3 TPU 对 GPU 第一份的逐步 loss 相对差（topic08_series.json，实测）：窗口只决定求平均的范围。
#   Ⓑ 第二次迁移 TPU 自比「FSDP 改纯数据并行」那一臂前 300 步（对齐记录的结果文件与《尺子》入门 1.7 节的算例表，实测；分母一律参照）。
_SER = N.series()["loss_rel_tpu_vs_gpu_a"]
_FULL, _TAIL = _SER[1:1000], _SER[300:1000]
def _st(x):
    k = max(range(len(x)), key=lambda i: abs(x[i]))
    return sum(x) / len(x), sum(abs(v) for v in x) / len(x), abs(x[k]), k
SM_F, MA_F, MX_F, KF = _st(_FULL)
SM_T, MA_T, MX_T, KT = _st(_TAIL)
assert abs(SM_T - (-1.66e-4)) < 0.01e-4 and abs(SM_F - (-1.30e-4)) < 0.01e-4 and KF + 2 == 380 and KT + 301 == 380
#   每格 (被测, N, r)：r 取结果文件里用未舍入原值算的比值（用表里舍入后的数现除，末位可能差 1）
DP8 = [("主 loss 全程", (5.58e-6, 3.01e-6, 1.85), (8.84e-6, 6.72e-6, 1.31), (3.70e-5, 4.53e-5, 0.82)),
       ("主 loss 后段", (9.22e-6, 4.54e-6, 2.03), (1.11e-5, 8.46e-6, 1.32), (3.70e-5, 4.53e-5, 0.82)),
       ("梯度范数", (2.60e-4, 9.92e-5, 2.62), (5.23e-4, 3.85e-4, 1.36), (1.84e-3, 2.24e-3, 0.82))]
assert all(abs(a / b - r) < 0.015 for row in DP8 for a, b, r in row[1:])
assert [max(r for _a, _b, r in row[1:]) for row in DP8] == [1.85, 2.03, 2.62]


def _e(v):
    m, e = ("%.2e" % v).split("e")
    return "%se%d" % (m, int(e))


def fig_rpipe():
    f = Fig(W, "两块。上块：一条逐步相对差的序列，第 1 步单独拿出来叫 d0，第 2 到 1000 步是全程窗口、第 301 到 1000 步是后段窗口；"
               "两个窗口读的是同一条序列，各自算出带符号均值、MARE、最大单步三个数：全程 −1.30e-4、1.53e-4、6.65e-4（第 380 步），后段 −1.66e-4、1.79e-4、同一步。"
               "下块：从 GPU 重跑 8 份两两配成 28 对，每对按同样的窗口和规则算三个统计、各取最大得到 N；被测的三个统计各除以 N 得三个 r，一项检查取最大，一次比较再取最大。"
               "算例是纯数据并行那一臂：主 loss 全程 1.85、后段 2.03、梯度范数 2.62，整次比较 2.62。")
    y0 = f.header("照着算一遍：一条序列 → 三个数 → 除以 N → 取最差",
                  "上：窗口只决定拿哪一段求平均，原始数据只有一份；下：r 怎么算出来、在两级上怎么取最差（算例数字都是实测）",
                  [(BL, "被测对参照的 dₜ"), (GY2, "GPU 重跑"), (OR, "取最大")])
    # Ⓐ
    PH1 = 300
    py = f.panel(0, y0, W, PH1, "Ⓐ 一条序列，两个窗口，三种 reduce（混元 3，TPU 对 GPU 第一份，逐步 loss 相对差）", BL)
    X0, XW, YM, YH = 110, 820, py + 120, 70
    lim = 7e-4

    def X(t):
        return X0 + (t - 1) / 999.0 * XW

    def Y(v):
        return YM - v / lim * YH
    f.line(X0, YM, X0 + XW, YM, LINE, 1.0, arrow=False)
    for t in range(2, 1001):
        f.line(X(t), YM, X(t), Y(_SER[t - 1]), "#a8c7fa", 0.6, arrow=False)
    f.box(X(1) - 4, Y(_SER[0]) - 4, 8, 8, INK, INK, 4)
    f.t(X(1) - 8, YM - 14, "d₀", INK, True, 13, "end")
    for (a, b, lab, yy, col) in [(2, 1000, "全程窗口：第 2–1000 步，999 个数", YM + YH + 20, BL), (301, 1000, "后段窗口：第 301–1000 步，700 个数", YM + YH + 46, OR)]:
        f.line(X(a), yy, X(b), yy, col, 2.2, arrow=False)
        f.line(X(a), yy - 6, X(a), yy + 6, col, 2.2, arrow=False)
        f.line(X(b), yy - 6, X(b), yy + 6, col, 2.2, arrow=False)
        f.t(X(a) + 6, yy - 6, lab, col, True, 13)
    f.t(X(380) + 10, Y(_SER[379]) + 4, "← 第 380 步", INK, size=12.5)
    TX = X0 + XW + 40
    f.t(TX, py + 40, "每个窗口算出三个数（不是滑动平均）", INK, True, 14)
    rows = [("", "全程", "后段"), ("带符号均值", _e(SM_F), _e(SM_T)), ("MARE", _e(MA_F), _e(MA_T)), ("最大单步", _e(MX_F), _e(MX_T))]
    for i, (a, b, c) in enumerate(rows):
        y = py + 76 + i * 30
        f.t(TX, y, a, GY if i == 0 else INK, i > 0, 13.5)
        f.t(TX + 150, y, b, BL if i else GY, i > 0, 13.5)
        f.t(TX + 270, y, c, OR if i else GY, i > 0, 13.5)
    f.t(TX, py + 214, "第 1 步单独报：d₀ ＝ ＋2.5e-6，不进窗口", GY, size=13)
    f.t(TX, py + 238, "两个窗口读的是同一条序列", GY, size=13)
    f._pan = None
    # Ⓑ
    y1 = y0 + PH1 + 20
    PH2 = 330
    py2 = f.panel(0, y1, W, PH2, "Ⓑ 从 K 份运行到一个 r（第二次迁移，TPU 自比，FSDP 改纯数据并行，前 300 步）", OR)
    bx = [20, 300, 600, 900, 1160]
    top = py2 + 24
    lab = [("GPU 重跑 8 份", "同一设置"), ("两两配对", "C(8,2) ＝ 28 对"), ("每对算三个统计", "同窗口、同分母、去第 1 步"), ("各取 28 对里最大", "→ 三个 N"), ("剔除逐位相同的对", "不足 3 对最多待定")]
    for i, (a, b) in enumerate(lab):
        f.box(bx[i], top, 230, 56, "#fff", GY2, 8, sw=1.3, dash="4 3" if i == 4 else None)
        f.t(bx[i] + 115, top + 24, a, INK, True, 13.5, "middle")
        f.t(bx[i] + 115, top + 44, b, GY, size=12.5, anchor="middle")
        if i < 3:
            f.line(bx[i] + 232, top + 28, bx[i + 1] - 4, top + 28, GY, 1.4)
    yt = top + 96
    f.t(20, yt, "被测的三个统计 ÷ N", INK, True, 14)
    f.t(330, yt, "带符号均值", GY, size=13, anchor="middle")
    f.t(560, yt, "MARE", GY, size=13, anchor="middle")
    f.t(790, yt, "最大单步", GY, size=13, anchor="middle")
    f.t(990, yt, "一项检查取最大", OR, True, 13, "middle")
    for i, (name, a, b, c) in enumerate(DP8):
        y = yt + 34 + i * 40
        f.t(20, y, name, INK, True, 13.5)
        rs = []
        for j, (num, den, r) in enumerate((a, b, c)):
            rs.append(r)
            f.t(330 + j * 230, y, "%s ÷ %s ＝ %.2f" % (_e(num), _e(den), r), RD if r > 1 else INK, r > 1, 13, "middle")
        f.t(990, y, "%.2f" % max(rs), RD if max(rs) > 1 else INK, True, 14, "middle")
    f.t(20, yt + 34 + 3 * 40, "其余 5 项都 ≤ 1　·　后段 ＝ 第 101–300 步（200 个 dₜ）", GY, size=13)
    f.box(1100, yt + 10, 270, 120, "#fff", OR, 8, sw=1.6)
    f.t(1235, yt + 44, "一次比较取最大", OR, True, 14, "middle")
    f.t(1235, yt + 84, "2.62", RD, True, 26, "middle", big=True)
    f.t(1235, yt + 112, "> 2：超出尺子", RD, size=13, anchor="middle")
    f._pan = None
    yb = f.band(y1 + PH2 + 20, "ok", "照着这四步就能自己算：dₜ → 窗口里三个统计 → 各除以 N → 检查、比较两级各取最差", [
        "三处领头的都是带符号均值，MARE 和最大单步只略高于噪声：这是系统偏差的样子（loss 持续偏高、梯度范数持续偏低）。",
        "代价：N 是有限几对里的最大值，份数少会偏窄；一次比较看的数多，纯噪声也常落进 1–2。",
    ])
    yb = f.src(yb + 10,
               "📌 Ⓐ 实测：混元 3 截 4 层，TPU v5p 第一次 1000 步运行对 GPU 第一份，逐步 loss 相对差 dₜ ＝（TPU − GPU）÷ GPU；全程（第 2–1000 步）带符号均值 −1.30e-4、MARE 1.53e-4、最大单步 6.65e-4（第 380 步）；"
               "后段（第 301–1000 步）−1.66e-4、1.79e-4、同一步；第 1 步 ＋2.5e-6 单独报。",
               "📌 Ⓑ 实测：第二次迁移小尺寸模型，TPU 自比「FSDP 8 改纯数据并行 8」对 TPU golden，前 300 步；N 是 GPU 同设置 8 份两两 28 对的上限；分母一律用参照（梯度范数也是，对齐记录 2026-11-10 起统一）。"
               "主 loss 全程 ＋5.58e-6 ／ 3.01e-6、8.84e-6 ／ 6.72e-6、3.70e-5（第 273 步）／ 4.53e-5；后段 ＋9.22e-6 ／ 4.54e-6；梯度范数 −2.60e-4 ／ 9.92e-5、5.23e-4 ／ 3.85e-4、1.84e-3（第 28 步）／ 2.24e-3。带符号均值判定时取绝对值，正负号另看方向。",
               "⚠️ 两块数据来自两个模型：上块是混元 3（为了用一条完整的 1000 步序列），下块是第二次迁移（为了有一份完整公开的算例表）。")
    f.save("fig8-rpipe.svg", yb + 14)


# ── fig8-step1：第 1 步逐张量怎么定责（第二次迁移 TPU 自比，词嵌入梯度两次；实测）──────────────────
#   出处：对齐记录 TPU 自比两臂结果文件的 step1_rows 与《尺子》入门 2.2–2.5 节。
EMB_ROWS = [("纯数据并行", 1.51e-2, 1.46e-2, 5.89, 3.62), ("修复后（FP32 累加）", 3.79e-3, 3.37e-3, 1.36, 0.91)]
EMB_REF, EMB_GPU, EMB_SCALE = 4.00e-3, 4.17e-3, 2.48e-3
for _n, t64, d, k, R in EMB_ROWS:
    assert abs(d / EMB_SCALE - k) < 0.02 and abs(t64 / max(EMB_REF, EMB_GPU) - R) < 0.01


def fig_step1():
    f = Fig(W, "两块。上块是一条流程：第 1 步的梯度从一阶动量倒推出来；跟基线算相对 L2，除以这个张量自己的尺子（GPU 三份两两的最大差，不低于 2 的负 20 次方），不超过 1 就在噪声内；"
               "超过的，再看它离 64 位真值的距离，除以基线和 GPU 中位数里离得远的那个，不超过 1.2 记换了顺序，超过记丢了精度。"
               "两行算例都是词嵌入梯度：纯数据并行时超出尺子 5.89 倍、离真值远 3.62 倍，丢了精度；修好以后仍超出尺子 1.36 倍，可离真值只有 0.91 倍，换了顺序。"
               "下块是两类尺子怎么配合的四格：曲线和第 1 步都过，只是换了顺序；曲线不过、第 1 步全定责，多半是第 1 步之后的事；曲线过、第 1 步有丢精度的，是曲线没灵敏到；两样都不过，按 bug 查。")
    y0 = f.header("第 1 步逐张量：先量差多少，再请 FP64 判是谁的",
                  "上：一个张量走一遍定责流程（两行算例是同一个张量在两次改动下的读数）；下：两类尺子各管什么，四种组合怎么读",
                  [(BL, "第 1 步逐张量"), (GR, "换了顺序"), (RD, "丢了精度")])
    PH1 = 350
    py = f.panel(0, y0, W, PH1, "Ⓐ 定责流程（第二次迁移，TPU 自比，词嵌入梯度）", BL)
    steps = [("① 拿到梯度", "一阶动量倒推：", "m₁ ÷ (1 − β₁) ÷ 裁剪系数"),
             ("② 跟基线比", "d ＝ ‖被测 − 基线‖ ÷ ‖基线‖", "全部元素，FP64 里累加"),
             ("③ 除以尺子", "尺子 ＝ GPU 三份两两最大差", "不低于 2⁻²⁰；≤ 1 在噪声内"),
             ("④ 对 FP64 定责", "R ＝ 离 FP64 ÷ max(基线, GPU 中位数)", "≤ 1.2 换了顺序；＞ 1.2 丢了精度")]
    xs = [16, 360, 700, 1040]
    for i, (a, b, c) in enumerate(steps):
        f.box(xs[i], py + 20, 330 if i < 3 else 344, 74, "#fff", BL, 8, sw=1.3)
        f.t(xs[i] + 14, py + 44, a, BL, True, 14)
        f.t(xs[i] + 14, py + 66, b, INK, size=13)
        f.t(xs[i] + 14, py + 86, c, GY, size=12.5)
        if i < 3:
            f.line(xs[i] + 332, py + 57, xs[i + 1] - 4, py + 57, GY, 1.4)
    f.t(16, py + 130, "算例：词嵌入梯度（基线离 FP64 4.00e-3，GPU 三份中位数 4.17e-3，尺子 2.48e-3）", INK, True, 13.5)
    hdr = ["", "离 FP64", "对基线 d", "d ÷ 尺子", "R", "记作"]
    cx = [16, 330, 520, 710, 900, 1060]
    for j, hname in enumerate(hdr):
        f.t(cx[j], py + 166, hname, GY, size=13)
    for i, (n, t64, d, k, R) in enumerate(EMB_ROWS):
        y = py + 200 + i * 40
        far = R > 1.2
        f.t(cx[0], y, n, INK, True, 13.5)
        f.t(cx[1], y, _e(t64), INK, size=13.5)
        f.t(cx[2], y, _e(d), INK, size=13.5)
        f.t(cx[3], y, "%.2f（超出尺子）" % k, RD if k > 1 else INK, k > 1, 13.5)
        f.t(cx[4], y, "%.2f" % R, RD if far else GR, True, 13.5)
        f.t(cx[5], y, "丢了精度" if far else "换了顺序", RD if far else GR, True, 13.5)
    f.t(16, py + 300, "修好以后仍超出尺子，可离真值比基线和 GPU 都近：超出尺子不等于有问题，所以第 ③ 步之后一定要走第 ④ 步。", GY, size=13)
    f._pan = None
    y1 = y0 + PH1 + 20
    PH2 = 250
    py2 = f.panel(0, y1, W, PH2, "Ⓑ 两类尺子怎么配合", OR)
    f.t(250, py2 + 34, "第 1 步：没有丢精度的张量", GR, True, 14, "middle")
    f.t(600, py2 + 34, "第 1 步：有丢精度的张量", RD, True, 14, "middle")
    cells = [("曲线：分不开", "只是换了顺序，不是 bug", "多是曲线没灵敏到那一处", GR, OR),
             ("曲线：超出", "多半是第 1 步之后的事", "按 bug 查", OR, RD)]
    for i, (rowlab, c1, c2, col1, col2) in enumerate(cells):
        y = py2 + 56 + i * 82
        f.t(16, y + 40, rowlab, INK, True, 13.5)
        f.box(120 + 40, y, 260, 66, "#fff", col1, 8, sw=1.6)
        f.t(290, y + 38, c1, col1, True, 13.5, "middle")
        f.box(470, y, 260, 66, "#fff", col2, 8, sw=1.6)
        f.t(600, y + 38, c2, col2, True, 13.5, "middle")
    notes = ["左上：换切分、换 v7x、升运行库（基准配置下），大多落在这一格。",
             "左下：比如只作用在权重一侧的开关，第 1 步权重还在 BF16 格点上，测不到。",
             "右上：比如关小矩阵乘改写：300 步曲线 r 0.94，第 1 步一个归一化权重离 FP64 远 1.40 倍。",
             "右下：纯数据并行、关省舍入：曲线 2.62、13.3，第 1 步词嵌入梯度 R 3.62、R 26.1（运行库 0.0.46）。"]
    for i, n in enumerate(notes):
        f.t(770, py2 + 74 + i * 40, n, GY, size=13)
    f._pan = None
    yb = f.band(y1 + PH2 + 20, "ok", "曲线说有没有差、差多大、怎么随时间攒；第 1 步说差在谁身上、是换了顺序还是丢了精度", [
        "两样都过，才说改动只是换了求和顺序；有一样不过就要查，四格各指一个方向。",
        "代价：要一份 CPU 上 FP64 算的第 1 步梯度，而且裁判一律用 GPU golden 那一侧的代码，不能拿被测自己的输出当真值。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，TPU v5p 自比，运行库 0.0.46，对 TPU golden）：词嵌入梯度离 FP64 的 rel_L2，golden 4.00e-3、GPU 三份中位数 4.17e-3；纯数据并行 1.51e-2，对 golden 的 d 1.46e-2，尺子 2.48e-3，d ÷ 尺子 5.89，R 3.62；"
               "修复后 3.79e-3、d 3.37e-3、1.36、R 0.91。205 个张量的尺子中位数 2.68e-3。",
               "📌 梯度：TPU 两侧都从第一动量倒推（Adam 参数 m ÷ (1 − β₁)，Muon 参数 buf ÷ (1 − β)），再除以第 1 步的裁剪系数得到裁剪前的梯度；倒推本身带进几个 2⁻²⁴ 量级的相对舍入，所以尺子不低于 2⁻²⁰（GPU 三份逐位相同的张量尺子是 0，205 个里 10 个）。",
               "⚠️ 门槛 1.2 的来历（推断）：只差加法顺序时 R ≈ 1；多一次同量级的独立舍入，误差方差约翻倍，R ≈ √2 ≈ 1.41；1.2 取在两者之间。FP64 参照是用参照一侧（GPU）的同一套模型代码，在 CPU 上把第 1 步整网改成 FP64 重算的。"
               "GPU↔TPU 对齐时 R 的分母只用 GPU 三份中位数，另有「两边离 FP64 都小于一次 BF16 舍入（2⁻⁸）就按下限判过」一条。")
    f.save("fig8-step1.svg", yb + 14)


fig_loo()
fig_integrator()
fig_twoaxis()
fig_rpipe()
fig_step1()
