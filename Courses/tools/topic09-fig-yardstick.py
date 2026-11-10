# -*- coding: utf-8 -*-
r"""专题九 · 第三节补的两张「尺子」图（第二次迁移的新材料）。

fig9-loo   回答：r 落在 1–2 之间，算不算出事？—— 先看纯噪声自己能到多少。
           GPU 同设置的几份重跑，轮流挑一份当被测、一份当基线，尺子用其余几份重算，判法跟 TPU 被测一模一样；
           一次比较要看 8 条曲线 × 3 个统计 ＝ 24 个数，取最差的 —— 纯噪声也常常「略超」。
           第三行：8 份尺子时被推过 2 倍的六个 TPU 改动，补到 16 份后全部回到 1–2 之间。
           数据：topic09_tpuself.json（导出自第二次迁移的对齐记录：TPU 自比的 GPU 留一法、长跑灰区表），全部实测。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE
import topic09_numbers as N

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
assert abs(min(g["r1000_n16"] for g in GZ) - 1.18) < 1e-9 and abs(max(g["r1000_n16"] for g in GZ) - 1.62) < 1e-9
LOO8_1000_MAX = 2.00        # 1000 步只有 8 份时的留一法最大值（对齐记录「GPU 重跑的留一法 r」表，已被 16 份取代）
YEL = "#f9ab00"


def fig_loo():
    f = Fig(W, "一根横轴是 r，0 到 1 绿色判过，1 到 2 黄色判待定，大于 2 红色判不过。"
               "第一行：GPU 同设置 8 份、300 步，轮流一份当被测、一份当基线、其余当尺子，一共 56 种，一个点一种；"
               "59% 超过 1，其中 11% 超过 2，最大 2.47。第二行：1000 步、16 份，240 种，22% 在黄区，没有进红区的，最大 1.82。"
               "第三行：六个 TPU 改动跑 1000 步，用 8 份的尺子时 r 在 2.00 到 2.75，越过了当时纯噪声的最大值 2.00；"
               "尺子补到 16 份后落到 1.18 到 1.62，都在纯噪声最大值 1.82 以内。")
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
    f.t(X(1.18) - 10, y + h3 - 10, "16 份尺子：1.18–1.62", BL, True, 13, "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "r 在 1–2 之间，先别急着判错：纯噪声自己就常常到这里", [
        "300 步 8 份：近一半落进待定、一成进不过；1000 步 16 份：两成落进待定，最高 1.82。份数越多，纯噪声越难超线。",
        "代价：份数少，尺子就窄（8 份时六个 TPU 改动压线或越过 2）；跟重跑分不开也不等于没改精度。这把「纯噪声最大值」是事后才量的。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸模型）：GPU 同设置 300 步 8 份（56 种轮换，每种尺子 21 对）、1000 步 16 份（240 种，每种 105 对）；"
               "每种轮换：一份当被测、一份当基线，尺子用其余各份两两重算，检查跟 TPU 自比相同（主 loss 全程与后段、MTP、索引器蒸馏、梯度范数、参数范数、路由偏置均值与标准差；1000 步多一条 MTP 后段）。"
               "300 步 59% 的最差 r 过 1（其中 11% 过 2，落在 1–2 之间的是 48%），最大 2.47；1000 步 16 份 22% 过 1、没有过 2 的，最大 1.82；1000 步只有 8 份时最大 2.00、64% 过 1。",
               "📌 第三行：TPU 自比 1000 步对 TPU golden；八份尺子的上限比十六份的窄 1.0–2.1 倍，多数检查里放大最多的是带符号均值（路由偏置均值 2.07 倍、梯度范数 1.70 倍）。"
               "六个改动：少开投影 FP32 累加 2.75 → 1.62、关矩阵乘之间的融合 2.50 → 1.47、SwiGLU 少舍一次 2.34 → 1.38、每层多记几个统计量 2.33 → 1.37、"
               "16 设备纯数据并行 ＋ FP32 词嵌入梯度 2.07 → 1.30、TPU 运行库降一版 2.00 → 1.18；其中两个只换顺序，两个动了精度。",
               "⚠️ 口径：「GPU 留一法最大值」是看到 TPU 结果以后才补的解读，只用来读表，不改判档；对齐记录把它事先登记了，再拿后来的新改动和新尺寸检验。")
    f.save("fig9-loo.svg", yb + 14)


import math  # noqa: E402
import random  # noqa: E402

# ── fig9-integrator：参数范数为什么灵（本课推导 ＋ 示意模拟）──────────────────────────
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
    f.save("fig9-integrator.svg", yb + 14)


# ── fig9-twoaxis：第 1 步逐张量，两把参照（实测）──────────────────────────────────
#   横：对 TPU golden 的相对差 ÷ GPU 第 1 步三份的刻度（＞ 1 ＝ 超出噪声）；
#   纵：被测离 CPU FP64 ÷ max(golden 离 FP64, GPU 三份中位数离 FP64)（＞ 1.2 ＝ 丢了精度；对齐记录 setup.md「定责」）。
S1 = T["step1_scatter"]
assert [a["label"] for a in S1] == ["v5p 换 v7x", "FSDP 改成纯数据并行", "关掉编译器的省舍入"]
assert all(a["n_tensors"] == 205 and a["n_small_skipped"] == 45 and len(a["points"]) == 160 for a in S1)
assert max(y for _x, y, _e in S1[0]["points"]) <= 1.2                      # 换 v7x：一个都没变远
_emb = [[(x, y) for x, y, e in a["points"] if e][0] for a in S1]
assert abs(_emb[1][1] - 3.62) < 0.005 and abs(_emb[2][1] - 26.13) < 0.005  # 词嵌入梯度：3.6 倍、26 倍（MATRIX B1、C 表）


def fig_twoaxis():
    f = Fig(W, "三块散点，每块是 TPU 上的一个改动。一个点是一个参数张量第 1 步的梯度，共 160 个。"
               "横轴是它跟 TPU 基线差多少，除以 GPU 重跑的差，大于 1 就超出噪声；纵轴是它离 64 位真值的距离，除以原来离真值的距离，大于 1.2 就是丢了精度。"
               "左块换到 v7x：148 个点超出噪声，可全部贴着纵轴 1，一个都没变远，是换了顺序。"
               "中块改成纯数据并行：只有词嵌入梯度一个红点冲到 3.6 倍。右块关掉编译器的省舍入：词嵌入梯度冲到 26 倍，另有一个路由门控权重 1.20 倍刚过线。")
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
               "📌 实测（第二次迁移，小尺寸模型，TPU v5p 自比，对同一个 TPU golden）：第 1 步 205 个参数张量的训练梯度，其中多路残差混合的 45 个小参数（缩放、偏置、混合权重）已知病态、对齐记录单列已接受，不画；画 160 个。",
               "📌 横：被测对 golden 的 rel_L2 ÷ GPU 第 1 步三份两两的最大 rel_L2（这个张量自己的刻度）。纵：被测离 CPU FP64 的 rel_L2 ÷ max(golden 离 FP64, GPU 三份离 FP64 的中位数)；对齐记录的定责规则是 ≤ 1.2 记「换了顺序」，超过记「丢了精度」。",
               "📌 词嵌入梯度离 FP64：golden 4.00e-3、GPU 三份中位数 4.17e-3；纯数据并行 1.51e-2（3.6 倍）；关 excess precision 1.09e-1（26 倍）。关省舍入那一块另有第 4 层路由门控权重 1.20 倍，按规则也记丢了精度、定不了责。横轴大到上千的点，是 GPU 三份完全逐位相同的张量：刻度是 0，按一个很小的下限（2⁻²⁰）算。")
    f.save("fig9-twoaxis.svg", yb + 14)


fig_loo()
fig_integrator()
fig_twoaxis()
