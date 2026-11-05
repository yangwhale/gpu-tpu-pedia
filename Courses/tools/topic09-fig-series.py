# -*- coding: utf-8 -*-
r"""专题九 · 第三节「逐步序列」的三张图。

fig9-dt     回答：1000 个逐步相对差 dₜ，怎么压成三个数？—— 带符号均值（偏）、MARE（抖）、最大单步（尖）。
            数据：topic09_series.json 的真实逐步值（TPU 对第一份 GPU、第二份 GPU 对第一份 GPU），点都是真点。
fig9-judge  回答：一个数怎么判过不过？—— r ＝ 被测 ÷ 噪声上限 N；≤ 1 过，> 2 不过，中间待定。
            数据：v5p 当前配置对精确轨迹的第一关验收（topic09_numbers.ACC，实测）。
fig9-pairs  回答：参照对为什么要多？—— 纯噪声超过 n 个样本最大值的概率是 1/(n+1)。
            数字：1/(n+1) 现算；GPU 对 GPU 演练里待定条数（实测）。
"""
import statistics

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, wpx, MONO_K
import topic09_numbers as N

W = 1400
S = N.series()
D_TPU = S["loss_rel_tpu_vs_gpu_a"]          # 下标 0 ＝ 第 1 步
D_GPU = S["loss_rel_gpu_b_vs_gpu_a"]
TAIL = slice(300, 1000)                     # 第 301–1000 步
SM_TPU, SM_GPU = N.mean(D_TPU[TAIL]), N.mean(D_GPU[TAIL])
MARE_TPU = N.mean([abs(x) for x in D_TPU[TAIL]])
SD_TPU, SD_GPU = statistics.pstdev(D_TPU[TAIL]), statistics.pstdev(D_GPU[TAIL])
IMAX = max(range(300, 1000), key=lambda k: abs(D_TPU[k]))
LOW_SHARE = sum(1 for x in D_TPU[TAIL] if x < 0) / 700.0
assert abs(SM_TPU - (-1.66e-4)) < 0.01e-4, SM_TPU            # 案例里的 −1.66e-4
assert abs(SM_GPU - (-3.8e-5)) < 0.1e-5, SM_GPU
assert abs(SD_TPU - SD_GPU) < 0.02e-4                         # 抖得一样大
assert abs(LOW_SHARE - 0.90) < 0.01
assert IMAX + 1 == 380


def fig_dt():
    f = Fig(W, "两排散点，横轴是第 1 到 1000 步，纵轴是这一步 loss 的相对差。"
               "上排是第二次 GPU 运行对第一次：点围着零上下抖，第 301 到 1000 步平均只偏负十万分之 3.8。"
               "下排是 TPU 对第一次 GPU：抖动的宽度和上排一样，但整团点往下沉，平均偏负万分之 1.66，九成的步更低。"
               "最大单步在第 380 步。")
    y0 = f.header("1000 个点压成三个数：偏多少、抖多大、最尖的一下",
                  "一个点 ＝ 一步的相对差 dₜ ＝（被测 − 基线）÷ 基线；阴影 ＝ 第 301–1000 步（尾段窗口）；黑线 ＝ 窗口里的带符号均值",
                  [(GY2, "GPU 第二次 对 第一次（噪声）"), (BL, "TPU 对 GPU 第一次")])
    PW, PH = W, 250
    X0, XW = 110, 1250
    lo, hi = -7e-4, 7e-4

    for r, (title, data, col, sm) in enumerate([
            ("GPU 自己跟自己：点围着零抖", D_GPU, GY2, SM_GPU),
            ("TPU 对 GPU：抖得一样宽，整团往下沉", D_TPU, BL, SM_TPU)]):
        py = f.panel(0, y0 + r * (PH + 16), PW, PH, title, GY if r == 0 else BL)
        YM = py + (PH - 70) / 2.0

        def Y(v):
            return YM - v / hi * (PH - 90) / 2.0

        def X(step):
            return X0 + (step - 1) / 999.0 * XW
        f.box(X(301), py + 6, X(1000) - X(301), PH - 74, "#f1f3f4", "#f1f3f4", 0)
        f.line(X0, Y(0), X0 + XW, Y(0), GY2, 1.0, arrow=False)
        for v in (-6e-4, -3e-4, 3e-4, 6e-4):
            f.t(X0 - 10, Y(v) + 4, "%+.0f×10⁻⁴" % (v * 1e4), GY, size=12, anchor="end")
        for k, v in enumerate(data):
            f.p.append('<circle cx="%.1f" cy="%.1f" r="1.6" fill="%s" fill-opacity="0.55"/>' % (X(k + 1), Y(v), col))
        f.line(X(301), Y(sm), X(1000), Y(sm), INK, 3, arrow=False)
        # R3：标签原来压在散点上 → 挪到窗口里点最稀的下沿，垫一块白底再写
        lab = "带符号均值 %s" % ("%.2e" % sm).replace("e-0", "e-")
        lw = int(wpx(lab, 14) * MONO_K) + 16
        ly = Y(lo) + 4
        f.box(X(1000) - lw, ly - 18, lw, 26, "#fff", INK, 5)
        f.line(X(1000) - lw / 2.0, ly - 18, X(1000) - lw / 2.0, Y(sm) + 3, INK, 1.0, arrow=False)
        f.t(X(1000) - 8, ly, lab, INK, True, 14, "end")
        if r == 1:
            f.box(X(IMAX + 1) - 9, Y(D_TPU[IMAX]) - 9, 18, 18, "none", OR, 9, sw=2)
            f.t(X(IMAX + 1) + 14, Y(D_TPU[IMAX]) + 5, "最大单步：第 %d 步 %.1e" % (IMAX + 1, D_TPU[IMAX]), OR, True, 13)
            f.t(X0 + XW, py + 22, "第 301–1000 步：九成的步 TPU 更低；逐步标准差 %.2e，跟上排 %.2e 一样" % (SD_TPU, SD_GPU), GY, size=13, anchor="end")
        for st in (1, 300, 1000):
            f.t(X(st), py + PH - 46, "第 %d 步" % st, GY, size=12, anchor="middle")
        f._pan = None
    yb = f.band(y0 + 2 * (PH + 16) + 4, "ok", "带符号均值专抓「一直往一边偏」：随机的抖动正负抵消，剩下的就是偏移", [
        "MARE（|dₜ| 的平均）看整体抖多大，最大单步拦住突然跳一下；三个都判，最差的那个决定结论。",
        "代价：窗口和分母要事先写死 —— 同一次运行，换个窗口结论可能相反（前 100 步几乎看不出偏）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：两条 1000 步运行（同权重、同数据、同顺序），第 301–1000 步 TPU 对 GPU 第一份的带符号均值 −1.66e-4，GPU 第二份对第一份 −3.8e-5；两条的逐步标准差都是 1.33e-4，一阶自相关接近 0。",
               "📌 首步差 d₁ 单独报，不进窗口：GPU 重跑第 1 步通常逐位相同，当不了尺子。")
    f.save("fig9-dt.svg", yb + 14)


def fig_judge():
    f = Fig(W, "一根数轴，刻度是被测值除以噪声上限 N 的倍数 r。0 到 1 是绿色，判通过；1 到 2 是黄色，判待定；大于 2 是红色，判不通过。"
               "四个点是 v5p 当前配置对精确轨迹的第一关验收：梯度范数 1.14 倍、loss 尾段 1.29 倍、loss 全程 1.92 倍都在黄色区，"
               "参数范数 3.45 倍落在红色区。")
    y0 = f.header("判过不过：被测值 ÷ 噪声上限 N",
                  "N ＝ 噪声参照里每一对运行算出来的同一个量，取最大的那个；r ≤ 1 就像多跑了一次噪声，r > 2 远超噪声能解释的范围",
                  [(GR, "通过"), ("#f9ab00", "待定（加参照，或者去定责）"), (RD, "不通过")])
    PH = 270
    py = f.panel(0, y0, W, PH, "例子：这一讲最后的验收结果（v5p 当前配置，实测；比的是什么，后面讲）", BL)
    X0, XW, RMAX = 90, 1220, 4.0

    def X(r):
        return X0 + r / RMAX * XW
    yl = py + 150
    for a, b, col, fill, lab in [(0, 1, GR, "#e6f4ea", "过"), (1, 2, "#f9ab00", "#fef7e0", "待定"), (2, RMAX, RD, "#fce8e6", "不过")]:
        f.box(X(a), yl - 26, X(b) - X(a), 52, fill, "none", 0)
        f.t((X(a) + X(b)) / 2, yl + 50, lab, col, True, 16, "middle")
    for r in range(0, 5):
        f.line(X(r), yl - 30, X(r), yl + 30, GY2, 1.2, arrow=False)
        f.t(X(r), yl + 78, "%d×N" % r, GY, size=13, anchor="middle")
    for k, (name, x, n, r) in enumerate(N.ACC):
        col = GR if r <= 1 else ("#f9ab00" if r <= 2 else RD)
        f.box(X(r) - 8, yl - 8, 16, 16, col, INK, 8)
        f.t(X(r), yl - 40 - (k % 3) * 24, "%s %.2f" % (name.replace("带符号均值", "").replace("（", "").replace("）", "").strip(), r), INK, True, 13, "middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每一项用三个统计各算一个 r，最差的决定这一项；图上画的就是最差的那个", [
        "这一次：参数范数 3.45 倍，不通过；loss 两个窗口、梯度范数在 1–2 倍之间，待定。",
        "代价：2 倍这条线是约定的余量，不是统计推导出来的；待定不能直接签，要先加参照重判。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：被测是 v5p 当前配置 1000 步，基线是精确轨迹的一份（它是什么，后面讲 golden 时细说），N 取 GPU 生产程序重跑 5 份两两 10 对的最大值。",
               "📌 param_norm 带符号均值 4.27e-7、N 1.24e-7；loss 全程 8.62e-5 / 4.49e-5，尾段 7.23e-5 / 5.62e-5；grad_norm 2.32e-3 / 2.03e-3。")
    f.save("fig9-judge.svg", yb + 14)


NS = [3, 10, 15, 28, 45]
PS = [N.p_exceed(n) for n in NS]
assert abs(PS[0] - 0.25) < 1e-12 and abs(PS[3] - 1 / 29.0) < 1e-12


def fig_pairs():
    f = Fig(W, "左边柱子：一个纯粹的噪声，超过 n 对参照里最大值的概率是 1 除以 n 加 1。3 对时是四分之一，10 对 9%，28 对 3.4%，45 对 2.2%。"
               "右边是一次 GPU 对 GPU 的演练，本来应该全部通过：3 对参照时 46 条检查里 8 条判待定，扩到 28 对只剩 1 条。")
    y0 = f.header("参照对太少，尺子就太短　——　<tspan font-weight=\"700\">3 对时，纯噪声也有四分之一会「超线」</tspan>",
                  "左：柱高 ＝ 一个纯噪声的被测超过 n 对参照最大值的概率，1/(n+1)；右：一次本该全过的 GPU 对 GPU 演练，一格一条检查，橙色 ＝ 判成待定",
                  [(GY2, "推导（纯噪声）"), (OR, "实测：待定"), (GR, "实测：通过")])
    PH = 330
    py = f.panel(0, y0, 760, PH, "纯噪声超线的概率（本课推导）", GY)
    BX, BW, BH = 70, 110, 220
    yb0 = py + 270
    for i, (n, p) in enumerate(zip(NS, PS)):
        x = BX + i * (BW + 26)
        h = p / 0.25 * BH
        f.box(x, yb0 - h, BW, h, GY2, GY2, 4)
        f.t(x + BW / 2, yb0 - h - 10, "%.1f%%" % (p * 100), INK, True, 15, "middle")
        f.t(x + BW / 2, yb0 + 24, "%d 对" % n, GY, size=14, anchor="middle")
    f._pan = None
    px = 790
    py2 = f.panel(px, y0, W - px, PH, "GPU 对 GPU 演练：本该全部通过（实测）", OR)
    rows = [(3, "46 条检查，8 条待定", 46, 8), (15, "重判那 8 条＋梯度范数 1 条：剩 3 条待定", 9, 3), (28, "重判全部 46 条：只剩 1 条", 46, 1)]
    for i, (n, lab, tot, bad) in enumerate(rows):
        y = py2 + 30 + i * 92
        f.t(px + 24, y + 18, "%d 对参照" % n, INK, True, 15)
        f.t(px + 130, y + 18, lab, GY, size=13)
        for j in range(tot):
            col = OR if j < bad else "#e6f4ea"
            f.box(px + 24 + j * 12, y + 32, 10, 22, col, OR if j < bad else GR, 2)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "正式签字前，每种设置至少跑 6–8 份运行，凑出 15–28 对参照", [
        "混元 3：GPU 默认设置那一组跑了 10 份、45 对；最后验收用的那一组（设置不同，不能混用）只有 10 对，是缺口。",
        "代价：参照对共用运行，不是独立样本，实际超线的机会比这个算法略高；最大单步这种极值统计最吃样本数。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导：被测若只是噪声，它和 n 个参照是同分布的 n＋1 个样本，每一个当最大值的机会相等，所以超过其余 n 个的概率是 1/(n＋1)。",
               "📌 实测：GPU 对 GPU 演练（46 条检查，各级各种量加起来），3 份运行两两 3 对时有 8 条待定；扩到 6 份 15 对，重判那 8 条外加梯度范数一条，共 9 条，剩 3 条；8 份 28 对重判全部 46 条只剩 1 条（loss 最大单步，1.03 倍）。")
    f.save("fig9-pairs.svg", yb + 14)


fig_dt()
fig_judge()
fig_pairs()
