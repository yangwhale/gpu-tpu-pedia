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
    f = Fig(W, "一根横轴是 r，0 到 1 绿色判过，1 到 2 黄色判略超，大于 2 红色。"
               "第一行：GPU 同设置 8 份、300 步，轮流一份当被测、一份当基线、其余当尺子，一共 56 种，一个点一种；"
               "59% 落进黄区，11% 进红区，最大 2.47。第二行：1000 步、16 份，240 种，22% 在黄区，没有进红区的，最大 1.82。"
               "第三行：六个 TPU 改动跑 1000 步，用 8 份的尺子时 r 在 2.00 到 2.75，越过了当时纯噪声的最大值 2.00；"
               "尺子补到 16 份后落到 1.18 到 1.62，都在纯噪声最大值 1.82 以内。")
    y0 = f.header("纯噪声互相比，也常常「略超」",
                  "一个点 ＝ GPU 同设置重跑里挑一份当被测、一份当基线，尺子用其余各份重算；判法跟 TPU 一模一样：8 条曲线 × 3 个统计 ＝ 24 个 r，取最差的",
                  [(GY2, "GPU 对 GPU（纯噪声）"), (BL, "TPU 改动：8 份尺子 → 16 份尺子")])
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
    for a, b, lab, col in [(0, 1, "过", GR), (1, 2, "略超", YEL), (2, RMAX, "不过", RD)]:
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
    yb = f.band(py + PH + 20, "ok", "r 在 1–2 之间，先别急着查：纯噪声自己就常常到这里", [
        "一次比较看 24 个数，挑最差的那个；每个数超过自己尺子的机会不大，24 个里总有一个超的机会就大了。",
        "代价：份数少，尺子就窄。8 份时六个 TPU 改动压线或越过 2，补到 16 份全回来 —— 这把「纯噪声最大值」是事后才量的。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸模型）：GPU 同设置 300 步 8 份（56 种轮换，每种尺子 21 对）、1000 步 16 份（240 种，每种 105 对）；"
               "每种轮换：一份当被测、一份当基线，尺子用其余各份两两重算，检查跟 TPU 自比相同（主 loss 全程与后段、MTP、索引器蒸馏、梯度范数、参数范数、路由偏置均值与标准差）。"
               "300 步 59% 的最差 r 过 1、11% 过 2，最大 2.47；1000 步 16 份 22% 过 1、没有过 2 的，最大 1.82；1000 步只有 8 份时最大 2.00、64% 过 1。",
               "📌 第三行：TPU 自比 1000 步对 TPU golden；八份尺子的上限比十六份的窄 1.0–2.1 倍，放大最多的是带符号均值（梯度范数 1.70 倍）。"
               "六个改动：少开投影 FP32 累加 2.75 → 1.62、关矩阵乘之间的融合 2.50 → 1.47、SwiGLU 少舍一次 2.34 → 1.38、每层多记几个统计量 2.33 → 1.37、"
               "16 设备纯数据并行 ＋ FP32 词嵌入梯度 2.07 → 1.30、TPU 运行库降一版 2.00 → 1.18；其中两个只换顺序，两个动了精度。",
               "⚠️ 口径：「GPU 留一法最大值」是看到 TPU 结果以后才补的解读，只用来读表，不改判档；对齐记录把它事先登记了，再拿后来的新改动和新尺寸检验。")
    f.save("fig9-loo.svg", yb + 14)


fig_loo()
