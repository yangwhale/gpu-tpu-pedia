# -*- coding: utf-8 -*-
r"""专题八 · 第八节「案例（下）」的五张图。

fig8-gauge     回答：输出头流回隐状态的梯度里，哪个方向能让「所有 logits 一起升降」？它每步少一点，会积出什么？
               数字：探针测到的 lse（logits 的 log-sum-exp）差随步数（实测）；每步少推 6.5e-4 来自输出头回放（实测）。
fig8-head      回答：那股力少在哪一段？—— GPU 输出层 dgrad 的 BF16 张量核矩阵乘；求和越长偏得越多。
fig8-intervene 回答：怎么证明是这一股力？—— 两边都去掉它，差回到噪声；GPU 反着加上它，复现 TPU 的整套特征。
fig8-swap      回答：是每一步都算得不一样，还是状态不同带出来的反馈？—— 把同一个状态搬到两边，各算一步。
fig8-exact     回答：到底谁离精确值近？—— GPU 上跑一条「BF16 权重、全程 FP32 计算」的精确轨迹，它站在 TPU 一边；可它的长求和也偏（11-06）。
fig8-layer0    回答：冷门行的方向从哪来？—— 残差流固定探针：第 0 层输出最先沿隐状态方向 ĥ ＝ γ⊙û 偏低、往后逐层放大；同一份输入只换第 0 层权重，确认是权重长歪，大头在注意力输出投影。
               数字：L0_PROBE（每 25 步，逐点实测）、L0_SWAP_*（第 0 层权重互换，实测）。
               ⛔ 刻意没画最后归一化输出那条线：它和第 3 层几乎重合，多一条只添乱。
数字全部来自 topic08_numbers（实测）。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU, wpx, MONO_K
Fig.BAND_KEEP_LINES = 0   # 锤炼阶段（2026-11-14）：图上落点带只留一行粗体结论，其余进图下折叠
import topic08_numbers as N

W = 1400

LSE = N.LSE_PROBE


def fig_gauge():
    f = Fig(W, "左：输出层每个词有一行权重，画成一束箭头，它们有一个共同的平均方向，粗箭头。"
               "隐状态 h 沿这个平均方向挪一点，所有词的 logits 就一起升高或降低一点；反向时，梯度沿这个方向的那一份，就是把所有 logits 一起往下压的力。"
               "右：TPU 对 GPU 的 lse 差随步数往下走：第 25 步 −0.0016，第 50 步 −0.016，第 100 步 −0.078，第 300 步 −0.218；GPU 两次之间小得多。")
    y0 = f.header("有一个方向，能让所有词的 logits 一起升降　——　<tspan font-weight=\"700\">那股力每步少一点，几百步就积出来了</tspan>",
                  "左：细箭头 ＝ 每个词那一行权重，粗箭头 ＝ 平均方向 w̄，红箭头 ＝ h 沿 w̄ 的一小步；右：横轴 ＝ 训练步（对数），往下 ＝ TPU 的 lse 更低",
                  [(INK, "平均方向 w̄"), (GY, "隐状态 h"), (RD, "h 沿 w̄ 的一小步"), (BL, "TPU − GPU"), (GY2, "GPU 两次之间")])
    PH = 360
    py = f.panel(0, y0, 600, PH, "输出头：logit ＝ h · 这个词的那一行", GY)
    ox, oy = 120, py + 215
    import random
    rnd = random.Random(7)
    for k in range(22):
        a = math.radians(38 + rnd.uniform(-26, 26))
        L = rnd.uniform(150, 230)
        f.line(ox, oy, ox + L * math.cos(a), oy - L * math.sin(a), "#bdc1c6", 1.2, arrow=False)
    a0 = math.radians(38)
    f.line(ox, oy, ox + 260 * math.cos(a0), oy - 260 * math.sin(a0), INK, 5)
    f.t(ox + 268 * math.cos(a0), oy - 268 * math.sin(a0) - 6, "w̄", INK, True, 20, big=True)
    hx, hy = ox + 150, oy + 24
    f.line(ox, oy, hx, hy, GY, 4)
    f.t(hx + 8, hy + 22, "隐状态 h", GY, True, 15)
    dx, dy = 70 * math.cos(a0), -70 * math.sin(a0)
    f.line(hx, hy, hx + dx, hy + dy, RD, 4)
    f.t(hx + dx + 8, hy + dy - 4, "h 沿 w̄ 挪一小步", RD, True, 14)
    f.t(24, py + 312, "每个词的 logit ＝ h·这一行；各行都带着 w̄ 那一份 → 所有 logits 一起升降", INK, True, 14)
    f._pan = None
    px = 620
    py2 = f.panel(px, y0, W - px, PH, "lse（logits 整体水平）差随步数往下走（实测，固定一条序列的探针）", BL)
    GX0, GW, GY0, GH = px + 110, 600, py2 + 270, 220

    def X(st):
        return GX0 + (math.log10(st) - math.log10(20)) / (math.log10(400) - math.log10(20)) * GW

    YT = GY0 - GH                      # 顶部是 0，往下是负：logits 水平「往下走」

    def Y(v):
        return YT + (-v) / 0.24 * GH
    f.line(GX0, YT, GX0 + GW, YT, GY2, 1.2, arrow=False)
    f.line(GX0, YT, GX0, GY0, GY2, 1.2, arrow=False)
    for v in (0, -0.1, -0.2):
        f.t(GX0 - 10, Y(v) + 4, "%g" % v, GY, size=12, anchor="end")
    for st in (25, 50, 100, 300):
        f.t(X(st), YT - 10, "第 %d 步" % st, GY, size=12, anchor="middle")
    f.path([(X(s), Y(v)) for s, v, _ in LSE], BL, 2.6, arrow=False)
    # R18：灰点（GPU 两次之间）先画，TPU 蓝点最后画 —— 第 25 步两点几乎重合，蓝点不能被盖住
    for s, v, g in LSE:
        if g is not None:
            f.box(X(s) - 5, Y(-abs(g)) - 5, 10, 10, GY2, GY2, 5)
    for i, (s, v, g) in enumerate(LSE):
        f.box(X(s) - 6, Y(v) - 6, 12, 12, BL, BL, 6)
        # R3：「−0.078」原来写在点的右下，被下一段线穿过 → 一律写在左下（第一个点贴着纵轴，写右下）
        if i == 0:
            f.t(X(s) + 10, Y(v) + 20, "%g" % v, BL, True, 13)
        else:
            f.t(X(s) - 10, Y(v) + 20, "%g" % v, BL, True, 13, "end")
    f.t(X(300) - 10, Y(-0.021) + 20, "GPU 两次之间", GY, True, 13, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "这股力每一步只差 6.5e-4（相对；哪一边少推，后面揭晓），同一个方向、每一步都差，几百步积成 logits 整体水平的差", [
        "所有 logits 一起升降，softmax 的概率不变；可冷门词那几行压多压少，跟的正是原始 logit 的整体水平 —— 整体水平低一点，那几行就少压一点。",
        "代价：第 1 步只差 1e-4 量级，单看一步、对 GPU 噪声比，完全看不出来。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：两边在同一条固定序列上每 25 步跑一次前向（单独编译，不扰动训练）；TPU 对第一份 GPU 的 lse 均值差第 25 / 50 / 100 / 300 步 −0.0016 / −0.016 / −0.078 / −0.218，GPU 两次之间 −0.0001 / ＋0.0013 / −0.0062 / −0.021。",
               "📌 实测：第 1000 步的训练批上，TPU 的 logits 在整个词表上整体低 0.27、lse 低约 0.33，GPU 两次之间 0.0001。",
               "⚠️ 本课推导：dlogits 各项加起来是 0，可各行在 w̄ 上的投影长短不一，乘上各行以后沿 w̄ 并不抵消 —— 所以 dh 沿 w̄ 那一份不为零。",
               "注：左图的箭头是示意：真实的输出层有 120,832 行、4096 维。")
    f.save("fig8-gauge.svg", yb + 14)


def fig_head():
    f = Fig(W, "左：输出头反向沿「所有 logits 一起动」方向的相对误差，对数刻度，第 1 步。GPU 的交叉熵反向只有 2.7e-6；GPU 输出层 dgrad 的 BF16 矩阵乘 6.51e-4，是大头；"
               "TPU 整个头反向 2.4e-5；GPU 名义 FP32、实际 TF32 的那一版 8.08e-4，比 BF16 还差；真关 TF32 的 FP32 在第 1 步只有 1.7e-6，可它也是一次累加到底，训练到中途会长大。"
               "右：拿真实的输出头张量，把词表行随机打乱后取 K 行，K 从 16,384 到 120,832（整张词表），误差从 6.9e-5 涨到 6.5e-4；横纵都是对数刻度。")
    y0 = f.header("少掉的那一点，在 GPU 输出层 dgrad 那一次长求和里",
                  "柱长 ＝ 沿「所有 logits 一起动」方向的相对误差（对数刻度，一格 ＝ 十倍，越短越准），参照 FP64；右：横轴 K ＝ 词表行随机打乱后取的行数，横纵都是对数刻度",
                  [(RD, "GPU 张量核"), (GY, "名义 FP32（实为 TF32，先不看）"), (BL, "TPU"), (GR, "真 FP32"), (GY2, "交叉熵反向")])
    PH = 400
    py = f.panel(0, y0, 820, PH, "第 1 步，拿训练里实际用到的张量重算（实测）", OR)
    rows = [("GPU 交叉熵反向", "求 dlogits 那一段", N.CE_BWD_S1, GY2),
            ("GPU 输出层 dgrad，BF16", "dh ＝ dlogits · W，沿词表求和", abs(N.HEAD_BF16_S1), RD),
            ("TPU 整个输出头反向", "MXU：BF16 乘、FP32 累加", abs(N.HEAD_TPU_S1), BL),
            ("GPU dgrad 名义 FP32（先不看，见第十节）", "没关 TF32 —— 一度被当成「精确」", abs(N.HEAD_TF32_S1), GY),
            ("GPU dgrad FP32，真关 TF32", "第 1 步几乎不偏；训练中途会长大", abs(N.HEAD_EXACT_S1), GR)]
    bx, bw, emin, emax = 330, 380, 1e-6, 1e-3
    # ⭐ R3：对数柱的原点藏着（1e-6），柱长看着差 2 倍其实差 27 倍 → 每个十倍一根竖格线、顶上标数
    for e in range(-6, -2):
        xg = bx + (e + 6) / 3.0 * bw
        f.line(xg, py + 24, xg, py + 26 + 5 * 72 - 30, LINE, 1.0, arrow=False)
        f.t(xg, py + 18, "1e%d" % e, GY, size=12, anchor="middle")
    for i, (lab, sub, v, c) in enumerate(rows):
        y = py + 26 + i * 72
        f.t(24, y + 18, lab, INK, True, 14)
        f.t(24, y + 40, sub, GY, size=12)
        L = max((math.log10(v) - math.log10(emin)) / (math.log10(emax) - math.log10(emin)) * bw, 4)
        f.box(bx, y + 4, L, 28, c, c, 4)
        f.t(bx + L + 8, y + 24, "%.1e" % v, c, True, 14)
    f._pan = None
    px = 840
    py2 = f.panel(px, y0, W - px, PH, "求和越长，偏得越多（实测，GPU BF16）", RD)
    GX0, GW, GY0, GH = px + 90, 420, py2 + 300, 240

    def X(k):
        return GX0 + (math.log10(k) - 4) / (math.log10(150000) - 4) * GW

    def Y(e):
        return GY0 - (math.log10(e) - math.log10(2e-5)) / (math.log10(1e-3) - math.log10(2e-5)) * GH
    f.line(GX0, GY0, GX0 + GW, GY0, GY2, 1.2, arrow=False)
    f.line(GX0, GY0, GX0, GY0 - GH, GY2, 1.2, arrow=False)
    for e in (1e-4, 1e-3):
        f.t(GX0 - 8, Y(e) + 4, "%.0e" % e, GY, size=12, anchor="end")
    pts = [(X(k), Y(e)) for k, e in N.HEAD_BIAS_K]
    f.path(pts, RD, 2.6, arrow=False)
    for (k, e), (x, y) in zip(N.HEAD_BIAS_K, pts):
        f.box(x - 6, y - 6, 12, 12, RD, RD, 6)
        f.t(x, GY0 + 22, format(k, ","), GY, size=12, anchor="middle")
        f.t(x - 10, y - 10, "%.1e" % e, RD, True, 13, "end")
    f.t(GX0 + GW, GY0 + 46, "K ＝ 词表行随机打乱后取的行数（对数刻度）", GY, size=12, anchor="end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "交叉熵没问题，TPU 这一步准得多；偏的是 GPU 那一次 12 万长的 BF16 矩阵乘 —— 朝零截掉了一点", [
        "同样朝零；增长的形状跟合成数据不同 —— 这里 K 从 65,536 到 120,832 涨了三倍多，比线性还快；第 1、100 步都在 −6e-4 量级。",
        "代价：要先证明量的就是训练里实际用到的那几个张量（这里核对过，和抓下来的逐位相同）；精确参照还得真关 TF32。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：GPU 在线性层的前向、反向里记下输出层实际用到的 h、logits、权重和交给反向的 dlogits，与 dump 逐位相同；离线拆成「交叉熵反向」「矩阵乘」两段各对 FP64。第 100 步 GPU −6.10e-4，TPU −7.4e-5。",
               "📌 实测：右图把真实输出头张量的词表行打乱后取前 K 行，沿该方向的误差 −6.9e-5 / −2.0e-4 / −6.5e-4；朝零的整体尺度偏差 1.9e-5 / 7.3e-5 / 2.3e-4。")
    f.save("fig8-head.svg", yb + 14)


def fig_intervene():
    f = Fig(W, "两组柱子，左边是第 301 到 1000 步 loss 的平均偏移，右边是第 1000 步参数范数的差。"
               "第一根：原样，TPU 对 GPU，loss −1.66e-4、参数范数 +0.021。"
               "第二根：两边都把那个方向的梯度去掉，再比：loss 尾段平均 −3.4e-5，接近噪声 —— 但这是前半 +4.9e-5、后半 −9.6e-5 正负抵消出来的，后半仍剩约六成；参数范数 +0.0061，还剩约三成。"
               "第三根：GPU 人为往这个方向多推一把，对不加的 GPU 比：loss −1.26e-4、参数范数 +0.0205，几乎复现了 TPU。"
               "第三组：第 1000 步训练批上的 lse 差，原样约 −0.33，两边都去掉之后 −0.007，同设置 GPU 两次之间 −0.012；加力那组只跑过 300 步的 lse（−0.168，TPU 原样同步 −0.218），1000 步没有这一项。")
    y0 = f.header("去掉它：logits 水平差归零，loss 接近噪声，参数范数剩三成；加上它：差就回来了",
                  "柱长 ＝ 差有多大（零点 ＝ GPU 第一次运行）；灰色带 ＝ GPU 自己重跑的差（噪声），带边标了数",
                  [(BL, "原样：TPU 对 GPU"), (GR, "两边都去掉那个方向"), (OR, "GPU 人为多推一把，对 GPU 默认")])
    PH = 330
    groups = [("loss 平均偏移（第 301–1000 步，相对）", [(N.TPU_LOSS_TAIL, BL), (N.REMOVE_LOSS, GR), (N.FORCE_LOSS, OR)], 1.9e-4, 2.5e-5, "%.2e", "−2.5e-5"),
              ("参数范数差（第 1000 步，绝对）", [(N.TPU_PN_1000, BL), (N.REMOVE_PN, GR), (N.FORCE_PN, OR)], 0.025, 4.9e-4, "%+.4f", "＋4.9e-4"),
              ("lse 差（第 1000 步训练批）", [(N.ORIG_LSE, BL), (N.REMOVE_LSE, GR), (None, OR)], 0.37, 0.012, "%+.3f", "−0.012")]
    for gi, (title, bars, vmax, noise, fmt, nlab) in enumerate(groups):
        px = gi * 472
        py = f.panel(px, y0, 456, PH, title, BL)
        bx, bw = px + 24, 330
        f.box(bx, py + 30, max(noise / vmax * bw, 3), 3 * 78, "#dadce0", "#dadce0", 0)
        for i, (v, c) in enumerate(bars):
            y = py + 40 + i * 78
            if v is None:
                f.t(bx + 6, y + 26, "1000 步没有这一项（300 步时 −0.168）", GY, size=13)
                continue
            L = max(abs(v) / vmax * bw, 3)
            f.box(bx, y, L, 40, c, c, 4)
            f.t(bx + L + 8, y + 26, fmt % v, c, True, 14)
        f.t(bx, py + 30 + 3 * 78 + 26, "灰色带 ＝ GPU 重跑的差 %s" % nlab, GY, size=12.5)
        f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "loss 偏移、参数范数漂移、logits 整体偏低，是同一股力的三个后果", [
        "两边都去掉之后，loss 尾段平均几乎为零是正负抵消：前半（第 301–600 步）＋%s、后半（第 601–1000 步）%s，后半仍剩约六成。" % (N.sci(N.REMOVE_LOSS_H1), N.sci(N.REMOVE_LOSS_H2).replace("-", "−")),
        "代价：这两个开关改的是数学本身，只能拿来证明「是这股力」，不能放进 golden。参数范数还剩约三成没去掉：去掉的只是沿这个方向的那一份，剩下这截第九节的新账没有单独对上。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（1000 步）：两边都去掉 dh 沿 w̄ 的全部分量后，TPU 对 GPU：loss −3.4e-5，param_norm ＋6.1e-3；同设置 GPU 两次之间 loss −2.5e-5、param_norm ＋4.9e-4。",
               "📌 实测（1000 步，lse）：两边都去掉以后第 1000 步 lse 差 −0.007（原样约 −0.33），同设置 GPU 两次之间 −0.012。",
               "📌 实测：人为沿这个方向加一股力 —— 把 GPU 自己一份本来接近零的小分量（每个 token 的 dlogits 加起来，再乘 w̄）变成原来的 −8 倍，跑 1000 步，对默认 GPU：loss −1.26e-4，param_norm ＋2.05e-2；再跑一份 −1.27e-4、＋2.01e-2。TPU 原样对 GPU：−1.66e-4、＋2.1e-2。GPU 人为多推一把，连 logits 下移、冷门行范数偏大一起复现。这股力的大小没有跟 6.5e-4 那个缺口对过，只看结果能复现。")
    f.save("fig8-intervene.svg", yb + 14)


def fig_swap():
    f = Fig(W, "把 GPU 第 99 步的整套状态原样搬到 TPU 上，两边用同一批数据各算一步梯度，跟同一个精确参照（GPU 上全程 FP32 算的那一份）比，"
               "看沿漂移方向比精确值多走还是少走了多少。TPU 生产程序 0.00%，几乎正好；GPU 只把输出头 dgrad 换成 FP32 的，少走 0.10%；GPU 生产程序少走 0.36%。"
               "细线是 4 个状态、12 个 batch 的范围，结论都一样。")
    y0 = f.header("同一个起点，各算一步：GPU 每一步都少走一点　——　<tspan font-weight=\"700\">TPU 几乎正好</tspan>",
                  "柱长 ＝ 这一步沿「TPU − GPU 权重差」方向，比精确值少走了漂移本身的百分之几（往左 ＝ 少走）；细线 ＝ 12 个 batch 的范围",
                  [(BL, "TPU"), (GR, "GPU，只换FP32 dgrad"), (RD, "GPU")])
    PH = 300
    py = f.panel(0, y0, W, PH, "同一状态、同一批数据（实测，全体参数）", BL)
    X0, SC = 900, 1600                      # 0 点在 X0；每 1% 1600 px → 0.5% ＝ 800 px
    cols = {"TPU 生产程序": BL, "GPU 只换FP32 dgrad": GR, "GPU 生产程序": RD}
    f.line(X0, py + 20, X0, py + 260, INK, 1.6, arrow=False)
    f.t(X0 + 8, py + 16, "精确值", INK, True, 13)
    for i, (lab, v, (lo, hi)) in enumerate(N.SWAP):
        y = py + 40 + i * 74
        c = cols[lab]
        f.t(24, y + 26, lab, c, True, 16)
        L = abs(v) / 100.0 * SC * 100
        if v < 0:
            f.box(X0 - L, y, L, 40, c, c, 4)
        else:
            f.box(X0 - 3, y, 6, 40, c, c, 2)
        f.line(X0 + lo * SC, y + 20, X0 + hi * SC, y + 20, INK, 2, arrow=False)
        f.line(X0 + lo * SC, y + 10, X0 + lo * SC, y + 30, INK, 2, arrow=False)
        f.line(X0 + hi * SC, y + 10, X0 + hi * SC, y + 30, INK, 2, arrow=False)
        f.t(X0 + max(hi, 0) * SC + 16, y + 27, ("少走 %.2f%%" % -v) if v < 0 else "%.2f%%（几乎正好）" % v, c, True, 15)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "同一个状态、同一批数据，GPU 算出来的梯度每一步都偏一点 —— 偏差在单步计算里就有", [
        "大头是那一次 dgrad：换成 FP32 的，GPU 从少走 0.36% 降到 0.10%；TPU 生产程序每个 batch 都在精确值的 ±0.1% 里。",
        "代价：要把一侧的整套 FP32 主权重原样导进另一侧（逐个张量核对指纹），还要先确认那个「精确参照」自己真是精确的。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：1193 个张量导入 TPU 后逐个核对 sha256（按 GPU 框架布局数，q/k/v、gate/up 已各合成一个，所以少于按开源命名数的 1781 个）。方向 d ＝ 第 100 步 TPU 与 GPU 主权重之差；梯度差经 Adam 逐元素缩放、乘学习率和裁剪系数后投到 d 上，按 |d|² 归一。",
               "📌 实测：θ₉₉ 上 TPU 0.00%、GPU FP32 dgrad −0.10%、GPU −0.36%；另外三个状态（GPU θ₂₉₉、TPU θ₉₉、TPU θ₂₉₉）、共 12 个 batch：TPU −0.04% 到 ＋0.10%，GPU FP32 dgrad −0.05% 到 −0.22%，GPU −0.31% 到 −0.47%。",
               "📌 参照用 GPU 上的全程 FP32（权重先舍成 BF16，跟生产用的同一份）：TPU 自己那份「全程 FP32」程序核对后发现路由专家实际按 BF16 算（对 FP64 3.2e-3），不能当参照。")
    f.save("fig8-swap.svg", yb + 14)


EX1, EX2 = N.EXACT_LOSS_TAIL
EXACT_ROWS = [   # (标题, [(值, 颜色, 标签, 标签抬高)], 满刻度, GPU 散布半宽, 格式)
    ("第 975 步参数范数差", [(N.EXACT_POS[0][1], GR, "FP32 dgrad", 16), (N.EXACT_POS[1][1], PU, "精确轨迹", 36), (N.EXACT_POS[2][1], BL, "TPU", 16)], 0.022, N.EXACT_REF, "%+.4f"),
    ("第 975 步 lse 差", [(-0.175, GR, "FP32 dgrad", 16), (-0.262, PU, "精确轨迹", 36), (-0.286, BL, "TPU", 16)], 0.32, 0.013, "%+.3f"),
    # ⭐ 精确轨迹的 loss 有两份（−1.53e-4、−1.16e-4），TPU 的 −1.27e-4 夹在中间 → 画成一段紫色区间
    ("第 301–1000 步 loss 平均偏移（相对）", [(-7.3e-5, GR, "FP32 dgrad", 16), (EX2, PU, "精确轨迹（权重扰动 1e-6）", 36), (N.EXACT_TPU_LOSS_TAIL, BL, "TPU", 16), (EX1, PU, "精确轨迹", 36)], 1.7e-4, 9e-6, "%.2e"),
]


def fig_exact():
    f = Fig(W, "三行，都是对四次默认 GPU 运行的平均值比，零点就是 GPU。"
               "参数范数：GPU 只把 dgrad 换成 FP32 的，走到 +0.0125；精确轨迹走到 +0.0182；TPU 在 +0.0199。"
               "lse：FP32 dgrad 约 −0.175，精确轨迹 −0.262，TPU −0.286。loss：FP32 dgrad 约 −7.3e-5；精确轨迹有两份，−1.16e-4 和 −1.53e-4，TPU 的 −1.27e-4 夹在两份中间。"
               "灰色是 GPU 四次运行自己的散布。")
    y0 = f.header("精确轨迹站在 TPU 一边　——　<tspan font-weight=\"700\">可它自己也还差一步才到真精确</tspan>",
                  "零点 ＝ 4 次默认 GPU 运行的平均（不重算配置）；往右 ＝ 离 GPU 越远；精确轨迹 ＝ BF16 权重、全程 FP32（真关 TF32），可它的输出头长求和是 FP32 一次累加，中后期自己会偏，改成分段才到真精确",
                  [(GR, "GPU 只换FP32 dgrad"), (PU, "精确轨迹"), (BL, "TPU"), (GY2, "GPU 自己的散布（灰底）")])
    PH = 110
    y = y0
    for title, pts, vmax, ref, fmt in EXACT_ROWS:
        # R3：限定语「对 4 份默认 GPU 运行的均值；这个配置不重算」原来三个面板各写一遍 → 挪进副标题，只写一次
        py = f.panel(0, y, W, PH, title, BL)
        X0, XW = 90, 1220

        def X(v):
            return X0 + abs(v) / vmax * XW
        yl = py + 48
        f.line(X0, yl, X0 + XW, yl, GY2, 1.2, arrow=False)
        for q in (0.5, 1.0):
            f.line(X0 + q * XW, yl - 5, X0 + q * XW, yl + 5, GY2, 1.2, arrow=False)
            f.t(X0 + q * XW, yl + 24, fmt % (pts[-1][0] / abs(pts[-1][0]) * q * vmax), GY, size=11, anchor="middle")
        f.box(X0 - ref / vmax * XW, yl - 12, 2 * ref / vmax * XW, 24, "#dadce0", "#bdc1c6", 0)   # R18：原 #f1f3f4 在面板上看不见
        f.t(X0, yl + 32, "0（GPU 均值）", GY, True, 13, "middle")
        pu = [v for v, c, _, _ in pts if c == PU]
        if len(pu) == 2:
            f.box(X(min(pu, key=abs)), yl - 4, X(max(pu, key=abs)) - X(min(pu, key=abs)), 8, PU, PU, 4)
        for v, c, lab, up in pts:
            f.box(X(v) - 9, yl - 9, 18, 18, c, c if c != PU or len(pu) < 2 else "#fff", 9, sw=2)
            f.t(X(v), yl - up, "%s %s" % (lab, fmt % v), c, True, 13, "middle")
        f._pan = None
        y = py + PH - 30 + 14
    yb = f.band(y + 6, "ok", "精确轨迹走了 TPU 九成的路：参数范数 91%，lse 九成以上；loss 上 TPU 落在精确轨迹两份之间", [
        "离精确值远的首先是 GPU 的 BF16 golden。可这条精确轨迹的输出头 dgrad 仍是 cuBLAS FP32 一次累加整个词表，自己也偏：改成分段累加，漂移再大到它的 1.45 倍，TPU 生产是 1.10 倍（第 1000 步）。",
        "代价：精确轨迹慢（全程 FP32），而且是确定性的 —— 它自己量不出噪声，下一节要借别的尺子。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：精确轨迹 ＝ 主权重 FP32，每步前向前舍到 BF16，前向反向全部 FP32、关 TF32、不用融合注意力；第 2 / 3 步 loss 11.23569 / 10.58785，TPU 11.23569 / 10.58784，GPU BF16 11.23543 / 10.58775。",
               "📌 实测：第 975 步对 4 份默认 GPU 运行均值：param_norm 精确轨迹 ＋1.82e-2、TPU ＋1.99e-2、FP32 dgrad ＋1.23e-2 – ＋1.28e-2（图中取中点），4 份 GPU 自己 ±4.5e-4；lse −0.262 / −0.286 / −0.16 – −0.19，GPU ±0.013；301–1000 步 loss 精确轨迹两份 −1.53e-4、−1.16e-4，TPU −1.27e-4，FP32 dgrad −4.3e-5 – −1.02e-4，GPU ±9e-6。",
               "注：lse 与 loss 两行里「FP32 dgrad」取两份运行的中点，画的是位置示意。")
    f.save("fig8-exact.svg", yb + 14)


CHAIN = [   # R3：卡片文字砍到一行；边框统一中性灰
    ("① GPU 输出层 dgrad", "每步少推约 6.5e-4"),
    ("② 主干权重", "第 0 层注意力输出投影最先歪"),
    ("③ logits 的整体水平", "整体偏一截，softmax 看不见"),
    ("④ 冷门词那几行", "在 ε 区：logit 越高压得越多"),
    ("⑤ 看到的现象", "TPU 参数范数显大、loss 偏低"),
]


def fig_chain():
    f = Fig(W, "一条五环的链。第一环：GPU 输出层 dgrad 沿整张词表做长求和，沿 w̄ 每步少推约 6.5e-4，这个量是实测；归因成张量核长求和朝零截掉，是推断。"
               "第二环：主干权重沿这个方向每步少走一点，第 0 层注意力的输出投影最先长歪。"
               "第三环：隐状态在 w̄ 上的分量不同，所有 logits 一起偏一截，softmax 看不见。"
               "第四环：冷门词那几行在 ε 区，更新跟着梯度走，logit 水平越高被压得越多。"
               "第五环：GPU 的冷门行缩得多，TPU 的参数范数显得大，loss 偏低也一起出现。五环的量都有直接测量。")
    y0 = f.header("从一次求和到参数范数：五环一条链",
                  "一格 ＝ 一环，从左往右一环推一环；右上角的小标签 ＝ 这一环有直接测量；① 的原因另标",
                  [(GR, "实测"), (OR, "推断")])
    PH = 150
    py = f.panel(0, y0, W, PH, "TPU 在这一环上偏得小得多、GPU 每步少推一点 —— 两边一比，看上去像是 TPU 偏了", BL)
    BW, GAP = 252, 27
    for i, (title, line) in enumerate(CHAIN):
        x = 8 + i * (BW + GAP)
        f.box(x, py + 24, BW, 100, "#fff", GY2, 10, sw=1.5)
        f.t(x + 14, py + 60, title, INK, True, 15)
        f.box(x + BW - 54, py + 40, 44, 22, "none", GR, 11)
        f.t(x + BW - 32, py + 56, "实测", GR, True, 12, "middle")
        if i == 0:      # R6：① 的量是实测，「朝零截掉」这个原因是推断 —— 分两行各标
            f.t(x + 14, py + 90, "量：" + line, INK, size=13)
            f.t(x + BW - 14, py + 90, "实测", GR, True, 12, "end")
            f.t(x + 14, py + 112, "因：截断＋吞小项", INK, size=13)
            f.t(x + BW - 14, py + 112, "推断", OR, True, 12, "end")
        else:
            f.t(x + 14, py + 98, line, INK, size=13, w=BW - 24)
        if i < len(CHAIN) - 1:
            f.line(x + BW + 3, py + 74, x + BW + GAP - 3, py + 74, GY2, 2)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "证据：两边都去掉第一环那个方向的力，logit 水平的差归零、loss 接近噪声、参数范数剩三成；GPU 人为加上它，整套现象一起出现", [
        "方向：GPU 少推 → GPU 主干权重沿那个方向少走 → GPU 的 logits 整体偏高 → GPU 冷门行压得更多 → GPU 参数范数小、loss 高；两边一比，看上去成了 TPU 偏。",
        "第四环在第 300 步同一步上量过：两边冷门行梯度的大小几乎一样（0.990），差在方向；九成多的元素落在 ε 区，Σ w·u 之差的大头来自一阶动量。",
        "loss 为什么也跟着变（softmax 看不见整体平移），机制没有验证；实测只是它跟着这股力一起出现、一起消失。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 推断：① 的量（沿 w̄ 每步少推约 6.5e-4）是实测；把它归因成「长求和偏向一边（张量核截断＋吞小项）」是推断。",
               "📌 实测：GPU dgrad 沿 w̄ 第 1 / 100 步 −6.5e-4 / −6.1e-4；第 0 层输出在各层里最先分开、往后逐层放大（第 50 步起转为偏低），换权重重算确认是权重长歪、大头在注意力输出投影；第 1000 步训练批上 TPU 的 logits 整体低 0.27。",
               "📌 实测（第四环，第 300 步同一步，没出现过的词那些行）：两边 |g| 的均方根之比 %.3f；Σ w·g GPU 8.1e-5、TPU 1.2e-6；%.0f%% 的元素 √v̂ ＜ ε（中位数 %s），这些元素的更新 ≈ m̂ ÷ ε；Σ w·u 两边差 −%d，其中 m̂ 那一份 −%d。"
               % (N.RARE_RMS_RATIO, N.RARE_EPS_FRAC * 100, ("%.1e" % N.RARE_VHAT_MED).replace("e-0", "e-"), -N.SUMWU_DIFF, -N.SUMWU_MHAT))
    f.save("fig8-chain.svg", yb + 14)


def _r2(v):
    """两位有效数字、四舍五入（不走二进制的「银行家」舍入）：−0.00535 → −5.4e-3。"""
    from decimal import Decimal, ROUND_HALF_UP
    d = Decimal(repr(v))
    e = d.adjusted()
    q = (d.scaleb(-e)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return ("−%se%d" % (-q, e)) if v < 0 else "+%se%d" % (q, e)


def _m3(v):
    """三位有效数字（照 N.sci）：−1.94e-3、+7.5e-6。"""
    t = N.sci(abs(v))
    return ("−" if v < 0 else "+") + t


def fig_layer0():
    ST = N.L0_PROBE_STEPS
    P = N.L0_PROBE
    i100, i300 = ST.index(100), ST.index(300)
    f = Fig(W, "左：横轴是训练步，纵轴是每一层输出在 token 上的均值，沿隐状态方向 h-hat（等于 gamma 乘 u-hat）的投影，TPU 对 GPU 的相对差，往下是 TPU 更低。"
               "embedding 的输出两边一样，贴着零；第 0 层（稠密层）从第 50 步起偏低，第 100 步负千分之 1.8，第 300 步负千分之 5.4；第 1、2、3 层一层比一层低，第 3 层到负百分之 1.4 左右。"
               "灰线是 GPU 重跑 A 对基线那份，第 0 层的同一个量。"
               "右：拿同一份第 0 层输入在 CPU 上重算，只换第 0 层权重。第 100 步换成 TPU 的权重，输出沿 h-hat 低负千分之 1.94，跟探针对得上；换成另外两份 GPU 的权重只有正百万分之 7.5 和负万分之 2.4。"
               "只换一组时，注意力输出投影一组就占负千分之 1.33，是大头。")
    y0 = f.header("冷门行的方向从哪来：第 0 层最先分开，往后逐层放大　——　<tspan font-weight=\"700\">是第 0 层的权重长歪了</tspan>",
                  "左：纵轴 ＝ 各层输出（token 均值）沿隐状态方向 ĥ ＝ γ⊙û 的投影，TPU 对 GPU 的相对差，往下 ＝ TPU 更低；右：同一份输入只换第 0 层权重重算，柱长 ＝ 沿 ĥ 低了多少",
                  [(BL, "TPU 对 GPU：第 0 层"), ("#8ab4f8", "第 1–3 层"), ("#c6dafc", "embedding"), (GY2, "GPU 两次之间（第 0 层）"), (RD, "大头")])
    PH = 440
    PW = 840
    py = f.panel(0, y0, PW, PH, "残差流固定探针：每 25 步一个点（实测）", BL)
    GX0, GW, YT, GH = 100, 640, py + 40, PH - 100
    VHI, VLO = 1.5e-3, -1.65e-2

    def X(st):
        return GX0 + st / 1000.0 * GW

    def Y(v):
        return YT + (VHI - v) / (VHI - VLO) * GH
    for v in (0, -5e-3, -1e-2, -1.5e-2):
        f.line(GX0, Y(v), GX0 + GW, Y(v), GY2 if v == 0 else LINE, 1.2 if v == 0 else 1.0, arrow=False)
        f.t(GX0 - 10, Y(v) + 4, "0" if v == 0 else _r2(v).replace(".0e", "e"), GY, size=12, anchor="end")
    for st in (100, 300, 500, 1000):
        f.t(X(st), YT + GH + 22, "第 %d 步" % st, GY, size=12, anchor="middle")
    f.path([(X(s), Y(v)) for s, v in zip(ST, N.L0_PROBE_GPU_PAIR)], GY2, 1.4, arrow=False)
    f.path([(X(s), Y(v)) for s, v in zip(ST, P["embedding"])], "#c6dafc", 2.4, arrow=False)
    for k in (3, 2, 1):
        f.path([(X(s), Y(v)) for s, v in zip(ST, P["第 %d 层" % k])], "#8ab4f8", 1.8, arrow=False)
    f.path([(X(s), Y(v)) for s, v in zip(ST, P["第 0 层"])], BL, 3.2, arrow=False)
    # 右端标签（按末值排开，至少隔 17px）
    ends = sorted([(Y(P[k][-1]), k) for k in ("第 0 层", "第 1 层", "第 2 层", "第 3 层")])
    last = -1e9
    for yv, k in ends:
        ly = max(yv + 5, last + 17)
        f.t(X(ST[-1]) + 10, ly, k, BL if k == "第 0 层" else "#4285f4", k == "第 0 层", 13)
        last = ly
    f.t(X(ST[-1]) + 10, Y(0) - 8, "embedding", "#4285f4", size=12)
    # 两个读数：标签放在第 0 层上方的空白处（0 线和第 0 层之间），细线连回实测点
    for i, (lx, lv) in ((i100, (380, -1.6e-3)), (i300, (380, -3.3e-3))):
        v = P["第 0 层"][i]
        lab = "第 %d 步 %s" % (ST[i], _r2(v))
        lw = int(wpx(lab, 14) * MONO_K) + 16
        f.line(X(ST[i]), Y(v), X(lx), Y(lv) - 5, BL, 1.0, arrow=False)
        f.box(X(lx), Y(lv) - 18, lw, 26, "#fff", BL, 5)
        f.t(X(lx) + 8, Y(lv), lab, BL, True, 14)
        f.box(X(ST[i]) - 6, Y(v) - 6, 12, 12, BL, BL, 6)
    f.t(X(60), Y(0) - 12, "第 50 步起转为偏低", BL, size=13)
    f._pan = None
    px = PW + 20
    py2 = f.panel(px, y0, W - px, PH, "第 0 层权重互换：第 100 步（实测）", GR)
    Z, SC = px + 24, 260 / 2.0e-3

    def bar(y, lab, v, col):
        f.t(Z + 8, y, lab, INK, False, 13)
        L = max(abs(v) * SC, 1.5) if v < 0 else 1.5
        f.box(Z, y + 8, L, 20, col, col, 3)
        f.t(Z + L + 8, y + 24, _m3(v), col, True, 13)
    f.t(Z, py2 + 24, "整层换掉　·　柱往右 ＝ 输出沿 ĥ 更低", GY, size=13)
    f.line(Z, py2 + 34, Z, py2 + 190, GY2, 1.2, arrow=False)
    sw = N.L0_SWAP_100
    for j, (lab, v, col) in enumerate([("换成 TPU 的第 0 层权重", sw[0], BL),
                                         ("换成 GPU 重跑 A 的", sw[1], GY2),
                                         ("换成 GPU 重跑 B 的", sw[2], GY2)]):
        bar(py2 + 52 + j * 46, lab, v, col)
    xp = Z + abs(P["第 0 层"][i100]) * SC
    # R18：标注原来写在虚线下端，挤进「换成 GPU 重跑 A 的」那一行 → 挪到蓝柱上方、虚线右侧
    f.line(xp, py2 + 38, xp, py2 + 86, BL, 1.6, dash="5 4", arrow=False)
    f.t(xp + 6, py2 + 48, "探针同一步 %s" % _r2(P["第 0 层"][i100]), BL, size=12)
    y2 = py2 + 210
    f.t(Z, y2, "只换一组（换成 TPU 的）", GY, size=13)
    f.line(Z, y2 + 10, Z, y2 + 152, GY2, 1.2, arrow=False)
    for j, (lab, v) in enumerate(N.L0_SWAP_ONE_100):
        bar(y2 + 28 + j * 42, lab, v, RD if j == 0 else "#8ab4f8")
    f.t(Z, y2 + 170, "归一化的 scale、q／k 归一化：都在 %s 以下" % ("%.0e" % N.L0_SWAP_SMALL).replace("e-0", "e-"), GY, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "偏低从第 0 层的权重来：同一份输入，换上 TPU 那一份第 0 层权重，就复现了探针量到的偏低", [
        "换上 GPU 另两份的权重差得小很多；只换一组时，注意力输出投影一组就占了约七成。",
        "没回答的：权重为什么往这个方向长 —— 第 1 步两边权重相同时，第 0 层沿这个方向的梯度看不出偏小，要回头看输出头流回来的那股力。",
    ])
    yb = f.src(yb + 10,
               "📌 ĥ 是什么：最后隐状态均值的方向，按最后归一化的 scale 加权；它跟前面输出层的 w̄（输出层各行权重的平均方向）不是同一个方向，一个在隐状态这边、一个由输出层权重定。实测（探针）：两边在同一条固定序列上每 25 步另跑一次前向（单独编译，不扰动训练），各层输出在 token 上取均值向量，投到 ĥ ＝ γ⊙û（γ：基线那份 GPU 运行该步最后归一化的 scale；û：它该步最后归一化输出均值的单位方向）。最后归一化输出 ＝ γ⊙x/rms(x)，所以这个投影就是这一层对「最后隐状态均值沿 û 那一份」的贡献。画的是（TPU − 基线 GPU）÷ |基线 GPU|；灰线是第二份 GPU 对基线的同一个量。",
               "📌 实测：第 300 步第 0–3 层 %s ／ %s ／ %s ／ %s；embedding 输出第 1–975 步 TPU 最大 2.3e-4，GPU 两次之间最大 1.5e-4。第 1 步第 0 层已差 %s（GPU 两次之间约 −6e-9），是前向数值差，第 50 步起转负并随步数增长。"
               % (tuple(_r2(P["第 %d 层" % k][i300]) for k in range(3)) + (_m3(P["第 3 层"][i300]), _r2(P["第 0 层"][0]))),
               "📌 实测（权重互换）：取基线 GPU 运行该步第 0 层的输入（一个 microbatch），在 CPU 上 FP32 重算第 0 层，只换权重；数值 ＝（换上的权重 − 基线自己的权重）算出的输出均值沿 ĥ 的投影之差 ÷ |基线自己的投影|。第 100 步 %s（第二、三份 GPU %s ／ %s）；第 50 步 %s（%s ／ %s），同步的探针是 %s，对得上。"
               % (_m3(N.L0_SWAP_100[0]), _r2(N.L0_SWAP_100[1]), _r2(N.L0_SWAP_100[2]), _r2(N.L0_SWAP_50[0]), _r2(N.L0_SWAP_50[1]), _r2(N.L0_SWAP_50[2]), _r2(P["第 0 层"][ST.index(50)])),
               "⚠️ 推断（案例记录里的，没有单独验证）：第 50 → 100 步，步数翻倍、差涨约 9 倍，不像每步一点固定偏差在线性累积，像有反馈在放大 —— logit 水平一旦走低，梯度跟着变，再往同一方向推。",
               "注：第 0 层是稠密层，第 1–3 层是 MoE 层；最后归一化输出那条线跟第 3 层几乎重合，没画。")
    f.save("fig8-layer0.svg", yb + 14)


def fig_onepass():
    """fig8-onepass 回答：只把这次求和换成 FP32，为什么单步好了、一千步还差一大截？分块为什么就够？
    左上：逐步参数范数（生产近似配置，对 GPU 生产 5 份均值）：只换 FP32（5 份均值）、分块修好的 golden（基线那份）、TPU 生产、真精确（虚线）。
    左下：同几条对 TPU 生产的差，看谁离开零、谁贴着零。右：同一次矩阵乘离线换写法（ABL_*）。
    数据：topic08_series.json 的 pnorm_rc_*（实测逐步值）；ABL_*（离线消融，实测）。"""
    S = N.series()
    T, Xf, Cg, Ex, SP = (S[k] for k in ("pnorm_rc_tpu_prod", "pnorm_rc_gpu_dgrad_fp32_mean5", "pnorm_rc_gpu_chunk_golden", "pnorm_rc_gpu_true_exact", "pnorm_rc_gpu_spread"))
    assert abs(T[999] - N.PN_TPU_PROD) < 1e-5 and abs(Cg[999] - N.PN_PRODCHUNK) < 1e-5 and abs(Ex[999] - N.PN_TRUE) < 1e-5
    assert abs(Xf[999] - 1.24e-2) < 0.005e-2
    f = Fig(W, "左上：参数范数对 GPU 生产 5 份均值的差，第 1 到 1000 步。橙线：GPU 生产只把输出头那次求和换成 FP32 一次加到底（5 份均值），第 1000 步只到 +0.0124；绿线：只把那次求和改成分块（最终 golden 的基线那份），+0.0207；蓝线 TPU 生产 +0.0209，绿线和蓝线几乎重合；紫色虚线是真精确 +0.0274。"
               "左下：同几条减去 TPU 生产：橙线一路往下离开，绿线贴着零。右：同一次矩阵乘离线换写法，张量核一次加到底一直在 1e-4 量级；FP32 一次加到底第 1 步只有 1.7e-6，第 100 步起又到 2.6e-4；分块一直在 1e-7 到 1e-6。")
    y0 = f.header("只换 FP32：单步好了，一千步只走到六成　——　<tspan font-weight=\"700\">改成分块，跟 TPU 重合</tspan>",
                  "左 ＝ 训练轨迹（生产近似配置；参数范数绝对差，对 GPU 生产 5 份均值）；右 ＝ 同一次矩阵乘离线换写法，沿「所有 logits 一起升降」方向对 FP64 的相对偏差",
                  [(OR, "只换 FP32 一次加到底"), (GR, "改成分块（最终 golden：升 FP32 再分块）"), (BL, "TPU 生产"), (PU, "真精确（虚线）"), (GY2, "GPU 生产 5 份的散布")])
    LW = 860
    X0, XW = 86, LW - 236

    def Xs(st):
        return X0 + (st - 1) / 999.0 * XW
    # 左上
    PH1 = 330
    py = f.panel(0, y0, LW, PH1, "参数范数的差（实测逐步）", BL)
    hi = 0.03
    YB, YH = py + PH1 - 40, PH1 - 80

    def Y(v):
        return YB - v / hi * YH
    up = " ".join("%.1f,%.1f" % (Xs(k + 1), Y(e)) for k, e in enumerate(SP))
    dn = " ".join("%.1f,%.1f" % (Xs(k + 1), Y(-e)) for k, e in reversed(list(enumerate(SP))))
    f.p.append('<polygon points="%s %s" fill="#e8eaed" stroke="none"/>' % (up, dn))
    f.line(X0, Y(0), X0 + XW, Y(0), INK, 1.2, arrow=False)
    for v in (0.01, 0.02, 0.03):
        f.t(X0 - 8, Y(v) + 4, "%+.2f" % v, GY, size=12, anchor="end")
    f.t(X0 - 8, Y(0) + 4, "0", INK, True, 12, "end")
    f.p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.6" stroke-dasharray="6 4"/>'
               % (" ".join("%.1f,%.1f" % (Xs(k + 1), Y(v)) for k, v in enumerate(Ex)), PU))
    for data, col, w in ((Xf, OR, 2.4), (T, BL, 2.6), (Cg, GR, 1.8)):
        f.path([(Xs(k + 1), Y(v)) for k, v in enumerate(data)], col, w, arrow=False)
    f.t(Xs(1000) + 6, Y(Ex[999]) + 4, "＋%.4f 真精确" % Ex[999], PU, True, 12.5)
    f.t(Xs(1000) + 6, Y(T[999]) - 4, "＋%.4f TPU" % T[999], BL, True, 12.5)
    f.t(Xs(1000) + 6, Y(Cg[999]) + 12, "＋%.4f 分块" % Cg[999], GR, True, 12.5)
    f.t(Xs(1000) + 6, Y(Xf[999]) + 4, "＋%.4f 只换 FP32" % Xf[999], OR, True, 12.5)
    f.t(X0 + 8, py + 26, "GPU 生产 ＝ 0（灰底是它 5 份的散布）", GY, size=12.5)
    f._pan = None
    # 左下
    PH2 = 240
    py2 = f.panel(0, y0 + PH1 + 12, LW, PH2, "减去 TPU 生产（实测逐步；粗线 50 步平均）", BL)
    lo2, hi2 = -0.010, 0.002
    YT2, YH2 = py2 + 14, PH2 - 72

    def Y2(v):
        return YT2 + (hi2 - v) / (hi2 - lo2) * YH2
    up = " ".join("%.1f,%.1f" % (Xs(k + 1), Y2(e)) for k, e in enumerate(SP))
    dn = " ".join("%.1f,%.1f" % (Xs(k + 1), Y2(-e)) for k, e in reversed(list(enumerate(SP))))
    f.p.append('<polygon points="%s %s" fill="#e8eaed" stroke="none"/>' % (up, dn))
    f.line(X0, Y2(0), X0 + XW, Y2(0), INK, 1.2, arrow=False)
    for v in (-0.008, -0.004):
        f.t(X0 - 8, Y2(v) + 4, "%+.3f" % v, GY, size=12, anchor="end")
    f.t(X0 - 8, Y2(0) + 4, "0", INK, True, 12, "end")
    w = 50
    for data, col in ((Xf, OR), (Cg, GR)):
        d = [a - b for a, b in zip(data, T)]
        f.p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="0.8" stroke-opacity="0.5"/>'
                   % (" ".join("%.1f,%.1f" % (Xs(k + 1), Y2(v)) for k, v in enumerate(d)), col))
        r = [sum(d[k - w + 1:k + 1]) / w for k in range(w - 1, len(d))]
        f.p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="3"/>'
                   % (" ".join("%.1f,%.1f" % (Xs(k + w), Y2(v)) for k, v in enumerate(r)), col))
    dX, dC = Xf[999] - T[999], Cg[999] - T[999]
    assert abs(dC) < 0.0005 and dX < -0.008
    f.t(Xs(1000) + 6, Y2(dX) + 4, "%+.4f" % dX, OR, True, 12.5)
    f.t(Xs(1000) + 6, Y2(dC) - 4, "%+.4f" % dC, GR, True, 12.5)
    for st in (1, 200, 500, 1000):
        f.t(Xs(st), py2 + PH2 - 32, "第 %d 步" % st, GY, size=12, anchor={1: "start", 1000: "end"}.get(st, "middle"))
    f._pan = None
    # 右：离线三个时间点
    RX, RW = LW + 20, W - LW - 20
    PHR = PH1 + 12 + PH2
    ry = f.panel(RX, y0, RW, PHR, "同一次矩阵乘，离线换写法（实测）", INK)
    rows = [("张量核一次加到底", N.ABL_BF16_ONESHOT, RD), ("FP32 一次加到底", N.ABL_FP32_ONESHOT, OR), ("张量核分块", N.ABL_BF16_CHUNK, GR)]
    bx0, bw = RX + 84, RW - 170

    def BX(v):
        return bx0 + (math.log10(max(abs(v), 1e-9)) + 9) / 6.5 * bw
    for e in (-9, -7, -5, -3):
        f.t(BX(10 ** e), ry + PHR - 44, "1e%d" % e, GY, size=11, anchor="middle")
        f.line(BX(10 ** e), ry + 40, BX(10 ** e), ry + PHR - 58, LINE, 1, arrow=False)
    for i, (nm, vals, col) in enumerate(rows):
        yy = ry + 50 + i * 150
        f.t(RX + 12, yy + 14, nm, col, True, 13.5)
        for j, (st, v) in enumerate(zip((1, 100, 300), vals)):
            yb_ = yy + 24 + j * 30
            f.box(bx0, yb_ + 4, BX(v) - bx0, 20, col, col, 3)
            f.t(bx0 - 6, yb_ + 19, "第 %d 步" % st, GY, size=11, anchor="end")
            f.t(BX(v) + 6, yb_ + 19, N.tx(v, "%.1e"), col, True, 12)
    f._pan = None
    yb = f.band(y0 + PHR + 14, "ok", "只换 FP32 治好了截断（第 1 步 1.7e-6），一千步却只走到 TPU 的六成；改成分块两样都治，跟 TPU 重合", [
        "真精确比 TPU 还高一截：两边生产程序都把流进输出头反向的梯度存成 BF16，这一截两边一样，不算谁的差（第九节拆账）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（左）：混元 3 截前 4 层，生产近似配置（整层重算＋路由回放）；逐步参数范数对 GPU 生产 5 份均值。橙线 ＝ GPU 生产只把输出头那次求和换成 FP32 一次加到底（真关 TF32），5 份均值；绿线 ＝ GPU 生产只把那次求和改成分块（升 FP32 再分块；最终 golden 的基线那份；乘法留在张量核只分块的写法一千步跟它分不开；10 份第 1000 步之间最多差 1.15e-3）；蓝线 ＝ TPU 当前对齐配置；紫色虚线 ＝ 真精确（BF16 权重、全程 FP32、那次求和分块）。灰底 ＝ GPU 生产 5 份各自对均值的最大偏离。",
               "📌 实测（右）：训练里实际用到的那几个张量（第 1、100、300 步），一张 B200 上换写法重算这次矩阵乘，沿 w̄ 对 FP64 的相对偏差。")
    f.save("fig8-onepass.svg", yb + 14)


fig_gauge()
fig_onepass()
fig_chain()
fig_head()
fig_intervene()
fig_swap()
fig_exact()
fig_layer0()
