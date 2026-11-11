# -*- coding: utf-8 -*-
r"""专题八 · 第三节「逐步序列」的三张图。

fig8-dt     回答：1000 个逐步相对差 dₜ，怎么压成三个数？—— 带符号均值（偏）、MARE（抖）、最大单步（尖）。
            数据：topic08_series.json 的真实逐步值（TPU 对第一份 GPU、第二份 GPU 对第一份 GPU），点都是真点。
fig8-judge  回答：一个数怎么判过不过？—— r ＝ 被测 ÷ 噪声上限 N；≤ 1 过，> 2 不过，中间待定。
            数据：v5p 当前配置对精确轨迹的第一关验收（topic08_numbers.ACC，实测）。
fig8-pairs  回答：参照对为什么要多？—— 纯噪声超过 n 个样本最大值的概率是 1/(n+1)。
            数字：1/(n+1) 现算；GPU 对 GPU 演练里待定条数（实测）。
fig8-baseline 回答：只对一份基线比，基线自己偏了怎么办？—— 把同设置 10 份 GPU 运行都画出来，基线正好在上沿；换成 10 份均值重算，TPU 照样偏。
            数据：10 份各自对第一份的第 301–1000 步 loss 带符号均值（topic08_numbers.GPU_RUNS10_TAIL，实测）；TPU 三种跑法（TPU_ARMS，实测）；扣均值是本课推导。
            ⛔ 刻意没画纵轴：灰点竖着错开只为不重叠，纵向没有意义。
"""
import statistics

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, wpx, MONO_K
import topic08_numbers as N

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
               "📌 实测：两条 1000 步运行（同权重、同数据、同顺序），第 301–1000 步 TPU 对 GPU 第一份的带符号均值 −1.66e-4，GPU 第二份对第一份 −3.8e-5；两条的逐步标准差是 1.33e-4 和 1.34e-4，一阶自相关接近 0。",
               "📌 首步差 d₀ 单独报，不进窗口：GPU 重跑第 1 步通常逐位相同，当不了尺子。")
    f.save("fig8-dt.svg", yb + 14)


def fig_judge():
    f = Fig(W, "一根数轴，数轴上标的是被测值除以噪声上限 N 的倍数 r。0 到 1 是绿色，判通过；1 到 2 是黄色，判待定；大于 2 是红色，判不通过。"
               "四个红黄点是定案前的一次判定（旧基线：精确轨迹）：梯度范数 1.14 倍、loss 尾段 1.29 倍、loss 全程 1.92 倍都在黄色区，"
               "参数范数 3.45 倍落在红色区。绿区里那一簇是定案后对最终 golden 的同四项：0.50 到 0.85 倍，全过。")
    y0 = f.header("判过不过：被测值 ÷ 噪声上限 N",
                  "N ＝ 噪声参照里每一对运行算出来的同一个量，取最大的那个；r ≤ 1 就像多跑了一次噪声，r > 2 远超噪声能解释的范围",
                  [(GR, "通过"), ("#f9ab00", "待定（加参照，或者去定责）"), (RD, "不通过")])
    PH = 270
    py = f.panel(0, y0, W, PH, "例子：v5p 当前配置的两次判定（实测）—— 红黄点：定案前，对旧基线精确轨迹、尺子借 ③ 那组；绿点：定案后，对最终 golden、尺子 45 对（都在讲 golden 时细说）", BL)
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
    fin = [N.FINAL[0][k] for k in (1, 2, 3, 4)]
    for v in fin:
        f.box(X(v) - 7, yl - 7, 14, 14, "#fff", GR, 7, sw=2.5)
    f.t((X(min(fin)) + X(max(fin))) / 2, yl - 40, "定案后四项 %.2f–%.2f" % (min(fin), max(fin)), GR, True, 13, "middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每一项用三个统计各算一个 r，最差的决定这一项；图上画的就是最差的那个", [
        "定案前那一次：参数范数 3.45 倍，不通过；loss 两个窗口、梯度范数在 1–2 倍之间，待定。换成最终 golden 和它自己的尺子以后，四项都在绿区。",
        "代价：2 倍这条线是约定的余量，不是统计推导出来的；待定不能直接签，要先加参照重判。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（定案前那次）：被测是 v5p 当前配置 1000 步，基线是精确轨迹的一份（它是什么，后面讲 golden 时细说），N 取 ③ 那组 GPU 重跑（生产近似＋输出头 dgrad 换成 FP32）5 份两两 10 对的最大值。",
               "📌 param_norm 带符号均值 4.27e-7、N 1.24e-7；loss 全程 8.62e-5 / 4.49e-5，尾段 7.23e-5 / 5.62e-5；grad_norm 2.32e-3 / 2.03e-3。",
               "📌 定案后：基线换成最终 golden 的一份、N 取它自己 10 份两两 45 对，loss 全程 0.80、尾段 0.80、梯度范数 0.85、参数范数 0.50。N 取参照各对里绝对值最大的那个；r ＝ |被测| ÷ N。参数范数的相对差只有 4e-7 也判不过，因为 r 不跟「看上去大不大」比，跟这个量自己的噪声比。")
    f.save("fig8-judge.svg", yb + 14)


NS = [3, 10, 15, 28, 45]
PS = [N.p_exceed(n) for n in NS]
assert abs(PS[0] - 0.25) < 1e-12 and abs(PS[3] - 1 / 29.0) < 1e-12


def fig_pairs():
    f = Fig(W, "左边柱子：一个纯粹的噪声，超过 n 对参照里最大值的概率是 1 除以 n 加 1。3 对时是四分之一，10 对 9%，28 对 3.4%，45 对 2.2%。"
               "右边是第一阶段 100 步的 GPU 对 GPU 演练，本来应该全部通过：3 对参照时 46 条检查里 8 条判待定，扩到 GPU 对 GPU 28 对只剩 1 条。")
    y0 = f.header("参照对太少，尺子就太短　——　<tspan font-weight=\"700\">3 对时，纯噪声也有四分之一会「超线」</tspan>",
                  "左：柱高 ＝ 一个纯噪声的被测超过 n 对参照最大值的概率，1/(n+1)；右：第一阶段 100 步的 GPU 对 GPU 演练（本该全过），一格一条检查，橙色 ＝ 判成待定",
                  [(GY2, "推导（纯噪声）"), (OR, "实测：待定"), (GR, "实测：通过")])
    PH = 330
    py = f.panel(0, y0, 760, PH, "纯噪声超线的概率（本课推导）", GY)
    BX, BW, BH = 70, 110, 220
    yb0 = py + 270
    for i, (n, p) in enumerate(zip(NS, PS)):
        x = BX + i * (BW + 26)
        h = p / 0.25 * BH
        f.box(x, yb0 - h, BW, h, GY2, GY2, 4)
        f.t(x + BW / 2, yb0 - h - 10, ("%.2f%%" if n == 15 else "%.1f%%") % (p * 100), INK, True, 15, "middle")   # R18：1/16 ＝ 6.25%，不写成 6.2%
        f.t(x + BW / 2, yb0 + 24, "%d 对" % n, GY, size=14, anchor="middle")
    f._pan = None
    px = 790
    py2 = f.panel(px, y0, W - px, PH, "第一阶段 100 步的 GPU 对 GPU 演练：本该全过（实测）", OR)
    rows = [(3, "46 条检查，8 条待定", 46, 8), (15, "重判那 8 条＋梯度范数 1 条：剩 3 条待定", 9, 3), (28, "重判全部 46 条：只剩 1 条", 46, 1)]
    for i, (n, lab, tot, bad) in enumerate(rows):
        y = py2 + 30 + i * 92
        f.t(px + 24, y + 18, "GPU 对 GPU %d 对" % n, INK, True, 15)
        f.t(px + 190, y + 18, lab, GY, size=13)
        for j in range(tot):
            col = OR if j < bad else "#e6f4ea"
            f.box(px + 24 + j * 12, y + 32, 10, 22, col, OR if j < bad else GR, 2)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "正式签字前，每种设置至少跑 6–8 份运行，凑出 15–28 对参照", [
        "n 份运行两两配对 ＝ n(n−1)/2 对：3 份 3 对、6 份 15 对、8 份 28 对、10 份 45 对。",
        "混元 3：GPU 默认设置那一组跑了 10 份、45 对；定案前验收用的那一组（设置不同，不能混用）只有 10 对；最终 golden 补到 10 份 45 对。",
        "代价：参照对共用运行，不是独立样本，实际超线的机会比这个算法略高；最大单步这种极值统计最吃样本数。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导：被测若只是噪声，它和 n 个参照是同分布的 n＋1 个样本，每一个当最大值的机会相等，所以超过其余 n 个的概率是 1/(n＋1)。",
               "📌 实测：第一阶段 GPU 对 GPU 演练（每份 100 步，46 条检查，各级各种量加起来），3 份运行两两 3 对时有 8 条待定；扩到 6 份 15 对，重判那 8 条外加梯度范数一条，共 9 条，剩 3 条；8 份 GPU 对 GPU 28 对重判全部 46 条只剩 1 条（loss 最大单步，1.03 倍）。")
    f.save("fig8-pairs.svg", yb + 14)


def _m(v, fmt="%.2e"):
    """−1.66e-4 这种写法：真负号、指数去掉前导 0。"""
    return (fmt % v).replace("e-0", "e-").replace("e+0", "e+").replace("-", "−")


def fig_baseline():
    G = N.GPU_RUNS10_TAIL
    MU = N.GPU_RUNS10_TAIL_MEAN
    SD = statistics.stdev(G)
    ZS = [(t - MU) / SD for t in N.TPU_ARMS]
    assert abs(min(ZS) - (-5.3)) < 0.05 and abs(max(ZS) - (-2.9)) < 0.05, ZS     # 案例里的「偏出 2.9–5.3 个标准差」
    f = Fig(W, "一根横轴，是第 301 到 1000 步 loss 的带符号均值，以第一份 GPU 运行为基线。"
               "上面一排 10 个灰点是同一设置的 10 份 GPU 运行，从负万分之 0.65 到正百万分之 2.4，被圈出来的是基线自己，等于 0，它在 10 份里排第三高，几乎在最右沿；10 份的均值是负十万分之 2.5。"
               "下面三行蓝点是同一份代码在 TPU 上的三种跑法：负万分之 1.66、1.35、1.03。每个蓝点画两段：到基线的灰线，和到 10 份均值的蓝线。"
               "扣掉均值后是负万分之 1.41、1.10、0.78，仍偏出 2.9 到 5.3 个标准差。")
    y0 = f.header("基线只是 10 份里的一份，还贴在右沿　——　<tspan font-weight=\"700\">量到 10 份的中心，TPU 照样偏</tspan>",
                  "横轴 ＝ 尾段 loss 的带符号均值（对基线那一份，越往左越低）；灰点 ＝ 同设置的 GPU 运行，圈出的是基线自己；细灰线 ＝ 到基线，粗蓝线 ＝ 到 10 份均值",
                  [(GY2, "GPU 同设置 10 份"), (BL, "TPU 三种跑法"), (GY, "10 份的均值（虚线）")])
    PH = 470
    py = f.panel(0, y0, W, PH, "第 301–1000 步 loss 带符号均值（实测）", BL)
    X0, XW, LO, HI = 230, 1110, -2.0e-4, 0.3e-4

    def X(v):
        return X0 + (v - LO) / (HI - LO) * XW
    yg = py + 92                         # 灰点那一排的基准线
    ytop, ybot = py + 40, py + PH - 56
    # 刻度与竖格线
    for k in range(-4, 1):
        v = k * 0.5e-4
        f.line(X(v), ytop, X(v), ybot, LINE, 1.0, arrow=False)
        f.t(X(v), ybot + 22, "0" if k == 0 else _m(v, "%.1e"), GY, size=13, anchor="middle")
    f.line(X(0), ytop, X(0), ybot, GY, 1.6, arrow=False)
    # 灰点：按横坐标逐个放，挨得近就往上摞一层（竖直错开只为不重叠）
    placed = []
    for v in sorted(G):
        lvl = 0
        while any(abs(X(v) - x) < 17 and l == lvl for x, l in placed):
            lvl += 1
        placed.append((X(v), lvl))
        cy = yg - lvl * 17
        f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (X(v), cy, GY2))
        if v == 0.0:
            ring = (X(v), cy)
    f.p.append('<circle cx="%.1f" cy="%.1f" r="13" fill="none" stroke="%s" stroke-width="2"/>' % (ring[0], ring[1], INK))
    f.t(ring[0] - 18, ring[1] - 22, "基线自己（＝ 0），10 份里排第 %d 高" % N.GPU_RUNS10_BASE_RANK, INK, True, 14, "end")
    f.line(X(min(G)), yg + 22, X(max(G)), yg + 22, GY2, 3, arrow=False)
    f.t(X(min(G)) - 10, yg + 27, "10 份的范围 %s 到 +%s" % (_m(min(G), "%.1e"), _m(max(G), "%.1e")), GY, size=13, anchor="end")
    # 10 份均值
    f.line(X(MU), yg + 34, X(MU), ybot, GY, 2, dash="6 5", arrow=False)
    f.t(X(MU) - 8, yg + 52, "10 份均值 %s" % _m(MU), GY, True, 14, "end")
    # TPU 三种跑法
    labs = ["原样", "不重算", "改片上内存上限"]
    for i, (lab, v) in enumerate(zip(labs, N.TPU_ARMS)):
        y = py + 200 + i * 76
        f.t(24, y + 5, lab, BL, True, 15)
        f.line(X(v), y - 14, X(0), y - 14, GY2, 1.4, arrow=False)
        f.t((X(v) + X(0)) / 2.0, y - 20, "到基线 %s" % _m(v), GY, size=13, anchor="middle")
        f.line(X(v), y + 12, X(MU), y + 12, BL, 5, arrow=False)
        f.t((X(v) + X(MU)) / 2.0, y + 34, "到均值 %s" % _m(v - MU), BL, True, 14, "middle")
        f.p.append('<circle cx="%.1f" cy="%.1f" r="8" fill="%s"/>' % (X(v), y, BL))
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "N 取最大值，兜住了「基线贴边」；被测值被带偏多少，拿 10 份的均值重算一遍就看得出", [
        "这一次：换成均值，三种跑法各缩回约 %s，仍偏出 %.1f–%.1f 个标准差 —— 偏差是真的。" % (_m(-MU, "%.1e"), -max(ZS), -min(ZS)),
        "代价：换哪一份当基线，被测值就跟着挪，最多挪灰点的整个宽度（这一次约 %s）。" % _m(max(G) - min(G), "%.1e"),
    ])
    yb = f.src(yb + 10,
               "📌 实测：同一设置（默认设置、不开分布式优化器）的 GPU 运行 10 份，含基线那一份；每份对基线算第 301–1000 步 loss 逐步相对差的平均。10 份的均值 %s、样本标准差 %s。比基线高的两份只高 %s 和 %s。"
               % (_m(MU), _m(SD), _m(sorted(G)[-2]), _m(max(G))),
               "📌 实测：TPU 三种跑法是同一份代码（原样、不重算、改编译器的片上内存上限），对同一份基线 %s ／ %s ／ %s。"
               % tuple(_m(v) for v in N.TPU_ARMS),
               "⚠️ 本课推导：扣掉均值 ＝ 被测值 − 10 份均值；标准差倍数 ＝ 扣掉均值后 ÷ 样本标准差。灰点竖着错开只为不重叠，没有纵轴。")
    f.save("fig8-baseline.svg", yb + 14)


fig_dt()
fig_judge()
fig_pairs()
fig_baseline()
