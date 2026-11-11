# -*- coding: utf-8 -*-
r"""专题八 · 第六节「观测」的一张图。

fig8-hooks  回答：为什么 GPU 上随手加个钩子抓数没事，TPU 上不行？TPU 上的第 1 步证据怎么拿？
            GPU：每个 kernel 的输出本来就写回显存，钩子读的是已经落地的数（实测：确定性设置下开不开钩子逐步逐位相同）。
            TPU：整张图一起编译，多一个输出就改了融合，舍入位置跟着变（实测：v5p 上加钩子的程序和生产程序不逐位）。
            三条干净的路：步与步之间读状态 ＋ 从 Adam 一阶动量反推梯度；单独编译的隔离回放；单独编译的固定探针。
            反推公式 g ＝ (m_k − β₁·m_{k−1}) ÷ (1 − β₁) ÷ 裁剪系数 是本课推导（由 Adam 一阶动量的定义倒过来写）。
fig8-diverge  回答：两边分叉以后为什么只比标量曲线？—— 连 FP32 轨迹从第 2 步起也离两条 BF16 轨迹 30–100 倍远，只有第 1 步（同一份权重）能当裁判。
              数字是第二次迁移真实权重截层的实测（主 loss 的相对差）。
fig8-ongrid   回答：从 BF16 检查点续训时，为什么只管权重一侧精度的开关，第 1 步前后逐位相同？—— 主权重的初值全在 BF16 格点上、激活也是 BF16，
              单遍 BF16 乘 ＋ FP32 累加已经精确；第一次更新以后主权重离开格点，开关才开始起作用。随机 FP32 初始化起训时不成立。
              数字全部是本脚本现跑的玩具实验（线性模型 ＋ Adam），不是任何真实模型的数。
              刻意没画：两条 loss 曲线本身 —— 玩具里开关带来的差只有千分之一量级，两条线叠成一条，看不出东西；
              改画「同一份权重，开关开、关算出的输出差」，它就是开关能起作用的那部分，第 1 步恰好是 0。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
import topic08_numbers as N

W = 1400

# 反推公式自检：随便造一组数，正着算一步 m，再倒回来
_g, _m0, _c = 0.37, -0.12, N.CLIP1
_m1 = N.BETA1 * _m0 + (1 - N.BETA1) * (_g * _c)
assert abs((_m1 - N.BETA1 * _m0) / (1 - N.BETA1) / _c - _g) < 1e-12


def fig_hooks():
    f = Fig(W, "左格 GPU：一个个算子按顺序跑，每个 kernel 的输出本来就写回显存，钩子读的是已经落地的数，数不变。"
               "中格 TPU：整张计算图一起编译，编译器会把相邻的算子融成一块、省掉中间的舍入，红色小块是一次舍到 BF16。生产程序四个算子融成一块，只在出口舍 1 次；"
               "加一个输出，融合在钩子处被切成两块，切口多舍一次，共 2 次，舍入位置跟着变，量出来的已经不是生产那份数。"
               "右格：TPU 上三条不扰动的路。一，步与步之间读状态：主权重、两个动量，再从一阶动量倒推出这一步的梯度；二，隔离回放：拿 GPU 的输入单独编译一层或一个算子来跑；三，固定探针：每隔几步在一条固定的序列上单独跑一次前向。")
    y0 = f.header("在 TPU 上「看一眼」会改掉你要看的东西　——　<tspan font-weight=\"700\">证据要从程序外面拿</tspan>",
                  "方块 ＝ 算子；虚框 ＝ 编译器融成的一块；红色小块 ＝ 一次舍到 BF16；放大镜 ＝ 抓数的钩子",
                  [(GY, "GPU 的算子"), (BL, "TPU 的算子"), (RD, "一次舍到 BF16"), (GR, "钩子不改数值"), (OR, "！钩子改了数值")])
    PH = 400

    def lens(cx, cy, col):
        """放大镜 ＝ 钩子（两格同一个画法）。"""
        f.p.append('<circle cx="%d" cy="%d" r="16" fill="#fff" stroke="%s" stroke-width="3"/>' % (cx, cy, col))
        f.line(cx + 12, cy + 12, cx + 26, cy + 28, col, 3, arrow=False)
    # GPU：算子是 GPU 的深灰，钩子绿（不改数值）
    py = f.panel(0, y0, 420, PH, "GPU：钩子读的是已落地的数", GY)
    for i, lab in enumerate(["矩阵乘", "归一化", "激活", "矩阵乘"]):
        x = 30 + i * 92
        f.box(x, py + 90, 78, 46, LINE2, GY, 6)
        f.t(x + 39, py + 118, lab, GY, True, 13, "middle")
        if i < 3:
            f.line(x + 78, py + 113, x + 92, py + 113, GY2, 1.4)
    lens(174, py + 186, GR)
    f.line(174, py + 168, 174, py + 140, GR, 1.4)
    # 「只是读一下」没说为什么不扰动 → 写明读的是本来就落地的数
    f.t(24, py + 238, "每个 kernel 的输出本来就写回显存，", GR, True, 14)
    f.t(24, py + 260, "钩子读的是已经落地的数", GR, True, 14)
    f.t(24, py + 300, "开不开钩子：逐步逐位相同", GR, True, 15)
    f.t(24, py + 326, "（确定性设置下实测）", GY, size=13)
    f._pan = None
    # TPU：算子一律 TPU 蓝；红色小块 ＝ 一次舍到 BF16（照 fig8-excess 的画法）；「钩子改了数值」另用橙色 ！ 标，不涂红
    px = 440
    py = f.panel(px, y0, 420, PH, "TPU：多一个输出，融合就变了", BL)
    OW = 66

    def cut(x, y):
        """一次舍到 BF16：红色小块 ＋「舍」字（跟 fig8-excess 一样）。"""
        f.box(x, y + 3, 12, 38, "#fce8e6", RD, 3)
        f.t(x + 6, y + 62, "舍", RD, True, 12, "middle")

    def bang(cx, cy):
        """钩子改了数值：橙色圆 ＋ 感叹号。"""
        f.p.append('<circle cx="%d" cy="%d" r="11" fill="%s"/>' % (cx, cy, OR))
        f.t(cx, cy + 5, "!", "#fff", True, 15, "middle")
    f.t(px + 24, py + 50, "生产程序：舍 1 次", INK, True, 14)
    f.box(px + 24, py + 64, 348, 60, "none", GY2, 10, dash="6 5")
    for i, lab in enumerate(["矩阵乘", "归一化", "激活", "矩阵乘"]):
        xx = px + 32 + i * 86
        f.box(xx, py + 72, OW, 44, "#e8f0fe", BL, 6)
        f.t(xx + OW / 2, py + 99, lab, BL, True, 13, "middle")
    cut(px + 380, py + 72)
    f.t(px + 24, py + 168, "加了钩子的程序：舍 2 次", INK, True, 14)
    f.box(px + 24, py + 182, 162, 60, "none", GY2, 10, dash="6 5")
    f.box(px + 210, py + 182, 162, 60, "none", GY2, 10, dash="6 5")
    for i, lab in enumerate(["矩阵乘", "归一化", "激活", "矩阵乘"]):
        xx = px + 32 + i * 80 + (26 if i >= 2 else 0)
        f.box(xx, py + 190, OW, 44, "#e8f0fe", BL, 6)
        f.t(xx + OW / 2, py + 217, lab, BL, True, 13, "middle")
    cut(px + 192, py + 190)
    cut(px + 380, py + 190)
    bang(px + 372, py + 182)
    # 切口处的放大镜 ＝ 钩子；改了数值 → 橙色
    f.line(px + 198, py + 262, px + 198, py + 272, OR, 1.4, arrow=False)
    lens(px + 198, py + 290, OR)
    f.t(px + 236, py + 290, "多一个输出：融合切开", OR, True, 13)
    f.t(px + 236, py + 310, "切口多舍一次", OR, True, 13)
    bang(px + 36, py + 342)
    f.t(px + 54, py + 347, "和生产程序不再逐位（v5p 实测）", OR, True, 15)
    f._pan = None
    # 三条路：每条只留一行（细节在正文三小节）
    px = 880
    py = f.panel(px, y0, W - px, PH, "TPU 上三条不扰动的路", GR)
    items = [("① 步与步之间读状态", "主权重、两个动量；梯度从动量倒推"),
             ("② 隔离回放", "拿 GPU 的输入，单独编译一层来跑"),
             ("③ 固定探针", "每 25 步单独编译跑一次前向")]
    y = py + 44
    for t, ln in items:
        f.t(px + 20, y, t, GR, True, 15)
        f.t(px + 36, y + 24, ln, INK, size=13, w=W - px - 50)
        y += 62
        if t.startswith("①"):
            f.t(px + 36, y - 10, "g ＝ (mₖ − 0.9·mₖ₋₁) ÷ 0.1 ÷ 裁剪系数", PU, True, 14, mono=True)
            y += 34
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "规矩：生产程序本身就是测量对象，证据从它外面拿", [
        "倒推出来的梯度在 GPU 上对过实际抓到的梯度，相对差 2e-7 到 4e-7 —— 跟直接抓一样准，而且一位都不扰动。",
        "代价：每一步都加钩子的「观测版」程序要先当扰动臂过标定，v5p 上没过，只能当诊断值，不能当正式证据。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导：Adam 一阶动量 mₖ ＝ β₁·mₖ₋₁ ＋ (1 − β₁)·gₖ（gₖ 是裁剪之后的梯度），倒过来 gₖ ＝ (mₖ − β₁·mₖ₋₁) ÷ (1 − β₁)；再除以裁剪系数得到裁剪之前的梯度。β₁ ＝ 0.9。",
               "📌 实测：GPU 上反推梯度对抓到的梯度 rel_L2 2e-7 – 4e-7；TPU 上固定探针单独编译，训练轨迹和不开探针时 1000 步逐位相同。",
               "📌 实测：v5p 上加抓数输出的训练程序和生产程序不逐位；XLA 默认在融合内部跳过中间的 BF16 舍入（excess precision），融合一变，舍入位置就变。",
               "注：中格四个算子、融合切成几块、舍几次都是示意（跟 fig8-excess 同一种画法）；真实的融合怎么切由编译器决定。")
    f.save("fig8-hooks.svg", yb + 14)


def fig_diverge():
    import math as _m
    f = Fig(W, "纵轴是相对差的大小，对数刻度；横轴是三个时段：第 1 步、第 2 到 10 步、第 11 到 100 步。"
               "红点是 TPU 对 GPU 的差，都偏同一边：第 1 步 −8.5e-5，之后 −6.8e-5、−4.8e-5。紫色一带是一条 FP32 轨迹离两条 BF16 轨迹的距离：从第 2 步起就是 2e-3 到 1e-2，是两边之差的 30 到 100 倍；这条轨迹只跑了 30 步。"
               "第 1 步两边用同一份权重，两个小圈是各离 FP32 多远，还能比；第 2 步起就比不了了。")
    y0 = f.header("连权重都不舍的 FP32 轨迹，第 2 步起就走开了：分叉以后只比标量曲线",
                  "纵轴 ＝ 主 loss 相对差的大小（对数）；红 ＝ TPU 对 GPU；紫带 ＝ FP32 轨迹离两条 BF16 轨迹；第 1 步两个小圈 ＝ 各离 FP32 多远",
                  [(RD, "TPU 对 GPU"), (PU, "FP32 轨迹离 BF16 轨迹"), (BL, "TPU 离 FP32"), (GY, "GPU 离 FP32")])
    PH = 340
    py = f.panel(0, y0, W, PH, "真实权重截层，主 loss（实测）", PU)
    X = [320, 720, 1120]
    Y0, Y1 = py + PH - 60, py + 30
    def Y(v):
        return Y0 - (_m.log10(v) - _m.log10(1e-5)) / (_m.log10(3e-2) - _m.log10(1e-5)) * (Y0 - Y1)
    for e in (1e-5, 1e-4, 1e-3, 1e-2):
        f.line(160, Y(e), W - 60, Y(e), LINE2, 1, arrow=False)
        f.t(150, Y(e) + 4, "%.0e" % e, GY, size=12, anchor="end")
    for x, lab in zip(X, ["第 1 步（同一份权重）", "第 2–10 步", "第 11–100 步"]):
        f.t(x, Y0 + 30, lab, INK, True, 14, "middle")
    f.box(X[1] - 150, Y(1e-2), (X[1] + X[2]) / 2 - (X[1] - 150), Y(2e-3) - Y(1e-2), "#f3e8fd", PU, 4)
    f.t(X[1] - 140, Y(1e-2) + 18, "FP32 轨迹离两条 BF16 轨迹 %s" % N.DIV_FP32_AWAY, PU, True, 13)
    f.t(X[1] - 140, Y(1e-2) + 38, "（只跑到第 %d 步）" % N.DIV_FP32_STEPS, PU, size=12)
    for x, v in zip(X, [N.LADDER_REAL_STEP1, -N.DIV_STEP2_10, -N.DIV_STEP11_100]):
        f.p.append('<circle cx="%d" cy="%.1f" r="8" fill="%s"/>' % (x, Y(abs(v)), RD))
        f.t(x + 14, Y(abs(v)) + 5, N.tx(v, "%.1e"), RD, True, 13)
    for dx, v, col, lab in [(-60, abs(N.DIV_STEP1[0]), BL, "TPU"), (-100, N.DIV_STEP1[1], GY, "GPU")]:
        f.p.append('<circle cx="%d" cy="%.1f" r="6" fill="none" stroke="%s" stroke-width="2"/>' % (X[0] + dx, Y(v), col))
        f.t(X[0] + dx - 10, Y(v) + 5, lab, col, True, 12, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "逐元素、逐张量只在第 0、1 步比；第 2 步起两边各走各的，只比 loss、范数这类标量曲线", [
        "第 1 步两边吃同一份权重，还能拿 FP32 比谁更近；第 2 步起这条 FP32 轨迹离两边几十倍，比不出谁更精确。",
        "代价：它连权重都不舍成 BF16，第 1 步以后练的就是另一个模型；第 1 步两边用的是同一份检查点，所以只有第 1 步能拿它比。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，真实权重截层，同一批数据上的 TPU FP32 轨迹 %d 步）：TPU 对 GPU 8 份均值的带符号相对差，第 1 步 %s、第 2–10 步约 −6.8e-5、第 11–100 步约 −4.8e-5，同号、逐步衰减；第 1 步各离 FP32：TPU −2.0e-5、GPU ＋6.5e-5。"
               % (N.DIV_FP32_STEPS, N.LADDER_REAL_STEP1_T),
               "⚠️ FP32 参照自己离 FP64 多远没有测，第 1 步的比较只当诊断。纵轴画的是大小，红点三个差都是负的。")
    f.save("fig8-diverge.svg", yb + 14)


# ── fig8-ongrid 的玩具实验：数全是这里现跑的 ───────────────────────────
import numpy as _np


def _bf16(x):
    """FP32 → BF16（就近舍入、平局取偶）→ 再放回 FP32。只动位，不依赖别的库。"""
    u = _np.asarray(x, _np.float32).view(_np.uint32).astype(_np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(_np.uint32).view(_np.float32)


# 舍入自检：1 附近 BF16 的格距是 2⁻⁷；正好在两格中间时取末位为偶的那一格
assert _bf16(_np.float32(1 + 2 ** -8)) == 1.0
assert _bf16(_np.float32(1 + 3 * 2 ** -8)) == _np.float32(1 + 2 ** -6)
# 两个 BF16 相乘：有效位各 8 位，乘积至多 16 位，FP32（24 位）装得下 → FP32 里的乘积就是精确值
_r = _bf16(_np.random.default_rng(0).standard_normal(200000).astype(_np.float32))
assert _np.array_equal((_r[:100000] * _r[100000:]).astype(_np.float64),
                       _r[:100000].astype(_np.float64) * _r[100000:].astype(_np.float64))

OG_K, OG_M, OG_B, OG_STEPS, OG_LR, OG_STD = 256, 32, 64, 60, 1e-5, 0.02


def _og_run(W0, on):
    """玩具线性模型 y ＝ x·W，MSE，Adam（FP32 主权重）。激活 x 是 BF16。
    on＝False：权重先舍成 BF16 再乘（单遍 BF16）；on＝True：权重按 FP32 乘（高精度）。只有这一处不同。"""
    rng = _np.random.default_rng(1)
    Wt = (rng.standard_normal((OG_K, OG_M)) * OG_STD).astype(_np.float32)
    W, m, v = W0.copy(), _np.zeros_like(W0), _np.zeros_like(W0)
    b1, b2, eps = _np.float32(0.9), _np.float32(0.999), _np.float32(1e-8)
    L, Ws = [], [W.copy()]
    for t in range(1, OG_STEPS + 1):
        X = _bf16(rng.standard_normal((OG_B, OG_K)).astype(_np.float32))
        T = (X @ Wt + _np.float32(0.01) * rng.standard_normal((OG_B, OG_M)).astype(_np.float32)).astype(_np.float32)
        E = X @ (W if on else _bf16(W)) - T
        L.append(_np.mean(E * E, dtype=_np.float32))
        G = (X.T @ E) * _np.float32(2.0 / (OG_B * OG_M))
        m = b1 * m + (1 - b1) * G
        v = b2 * v + (1 - b2) * G * G
        W = (W - _np.float32(OG_LR) * (m / (1 - b1 ** t)) / (_np.sqrt(v / (1 - b2 ** t)) + eps)).astype(_np.float32)
        Ws.append(W.copy())
    return _np.array(L), Ws


_W_rand = (_np.random.default_rng(7).standard_normal((OG_K, OG_M)) * OG_STD).astype(_np.float32)
_W_ckpt = _bf16(_W_rand)                      # 「BF16 检查点」：同一份随机数先舍到 BF16
_L_off, _Ws_off = _og_run(_W_ckpt, False)
_L_on, _Ws_on = _og_run(_W_ckpt, True)
# 从 BF16 检查点起训：第 1 步 loss、第 1 次更新后的整份权重，开关前后逐位相同；第 2 步起不同
assert _L_off[0] == _L_on[0] and _np.array_equal(_Ws_off[1], _Ws_on[1])
assert _L_off[1] != _L_on[1]
# 随机 FP32 初始化起训：第 1 步就不同
_Lr_off, _ = _og_run(_W_rand, False)
_Lr_on, _Ws_ron = _og_run(_W_rand, True)
assert _Lr_off[0] != _Lr_on[0]

_Xp = _bf16(_np.random.default_rng(99).standard_normal((2048, OG_K)).astype(_np.float32))


def _og_gap(W):
    """同一份权重、同一批 BF16 输入：开关开、关算出的输出的相对差（L2）。"""
    a = (_Xp @ W).astype(_np.float64)
    b = (_Xp @ _bf16(W)).astype(_np.float64)
    return float(_np.linalg.norm(a - b) / _np.linalg.norm(a))


def _og_off(W):
    return float(_np.mean(W != _bf16(W)))


def _og_dist(W):
    """离最近 BF16 格点的平均距离，以格距为单位（均匀乱撒时是 0.25）。"""
    W = W.astype(_np.float64)
    ulp = 2.0 ** (_np.floor(_np.log2(_np.abs(W))) - 7)
    return float(_np.mean(_np.abs(W - _bf16(W.astype(_np.float32))) / ulp))


OG_GAP = [_og_gap(W) for W in _Ws_on[:OG_STEPS]]       # OG_GAP[k-1] ＝ 第 k 步前向用的那份权重
OG_GAP_RAND = [_og_gap(W) for W in _Ws_ron[:OG_STEPS]]
OG_OFF = [_og_off(W) for W in _Ws_on]
OG_DIST = [_og_dist(W) for W in _Ws_on]
assert OG_GAP[0] == 0.0 and OG_OFF[0] == 0.0 and OG_DIST[0] == 0.0
assert OG_OFF[1] > 0.99                               # 第一次更新以后几乎全部离开格点
assert 0 < OG_GAP[1] < OG_GAP[20]                     # 离开了，但一开始离得近，差还小
assert 0.22 < OG_DIST[30] < 0.28                      # 约 30 步后像均匀乱撒一样（0.25）
assert OG_GAP_RAND[0] > 0 and abs(OG_GAP_RAND[0] / OG_GAP[-1] - 1) < 0.2   # 随机初始化第 1 步就在平台上
OG_PLATEAU = sum(OG_GAP[30:]) / len(OG_GAP[30:])


def fig_ongrid():
    import math as _m
    f = Fig(W, "左格是一段数轴，竖线是 BF16 能精确表示的格点。从 BF16 检查点起训，第 0 步的 FP32 主权重全落在格点上；"
               "第 1 次更新以后全部离开格点，但离得很近，平均 %.2f 格；第 30 次更新以后散到格点之间，平均 %.2f 格。"
               "右格横轴是训练步，纵轴是同一份权重开关开、关算出的输出相对差，对数刻度。从检查点起训，第 1 步恰好是 0，逐位相同；"
               "第 2 步起是 %.1e，几十步后长到约 %.1e。随机 FP32 初始化起训，第 1 步就已经是 %.1e。数都是玩具实验现跑的。"
               % (OG_DIST[1], OG_DIST[30], OG_GAP[1], OG_PLATEAU, OG_GAP_RAND[0]))
    y0 = f.header("第 1 步开关前后逐位相同，不等于开关没用　——　<tspan font-weight=\"700\">权重还在 BF16 格点上</tspan>",
                  "只管权重一侧精度的开关（例：权重按 FP32 乘，还是先舍成 BF16 再乘）；左：主权重在不在格点上；右：同一份权重开关开、关差多少",
                  [(GY, "BF16 格点"), (BL, "FP32 主权重（从 BF16 检查点起训）"), (OR, "随机 FP32 初始化起训")])
    PH = 410
    # ── 左格：数轴 ────────────────────────────────────────────────
    LW = 640
    py = f.panel(0, y0, LW, PH, "Ⓐ 主权重：第 0 步全在格点上，更新以后散开", BL)
    sp = 2.0 ** -13                                   # [2⁻⁶, 2⁻⁵) 这一段 BF16 的格距
    g0 = 2.0 ** -6 + 24 * sp
    NG = 4
    W0f = _Ws_on[0].ravel().astype(_np.float64)
    pick = []
    for j in range(NG + 1):                           # 每个格点挑三个落在它上面的元素
        idx = _np.nonzero(W0f == g0 + j * sp)[0][:3]
        pick += list(idx)
    assert len(pick) == 3 * (NG + 1), len(pick)
    XL, XR = 230, LW - 50
    def X(v):
        return XL + (v - g0) / (NG * sp) * (XR - XL)
    rows = [(0, "第 0 步", "检查点"), (1, "第 1 次更新后", None), (30, "第 30 次更新后", None)]
    for r, (k, lab, sub_) in enumerate(rows):
        yy = py + 70 + r * 96
        f.line(XL - 10, yy, XR + 10, yy, LINE2, 1, arrow=False)
        for j in range(NG + 1):
            f.line(X(g0 + j * sp), yy - 22, X(g0 + j * sp), yy + 22, GY2, 1.4, arrow=False)
        f.t(16, yy - 4, lab, INK, True, 13)
        f.t(16, yy + 14, "全部平均离格点 %.2f 格" % OG_DIST[k] if k else "全部在格点上", BL if k else GY, size=12)
        Wk = _Ws_on[k].ravel().astype(_np.float64)
        seen = {}
        for i in pick:
            x = X(Wk[i])
            if not (XL - 12 <= x <= XR + 12):
                continue
            n = seen.get(round(x), 0)
            seen[round(x)] = n + 1
            f.p.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" fill-opacity="0.85"/>'
                       % (x, yy - 14 + (n % 3) * 14, BL))
    f.t(XL - 20, py + PH - 52, "一格 ＝ 2⁻¹³（[2⁻⁶, 2⁻⁵) 这一段 BF16 的格距）", GY, size=12)
    f._pan = None
    # ── 右格：开关开、关的差 ──────────────────────────────────────
    RX = LW + 20
    py2 = f.panel(RX, y0, W - RX, PH, "Ⓑ 同一份权重，开关开、关算出的输出差", BL)
    PX0, PX1 = RX + 90, W - 40
    YT, YB = py2 + 30, py2 + PH - 124
    lo, hi = 1e-5, 1e-2
    def Y(v):
        return YB - (_m.log10(v) - _m.log10(lo)) / (_m.log10(hi) - _m.log10(lo)) * (YB - YT)
    def XS(k):
        return PX0 + (k - 1) / (OG_STEPS - 1) * (PX1 - PX0)
    for e in (1e-5, 1e-4, 1e-3, 1e-2):
        f.line(PX0, Y(e), PX1, Y(e), LINE2, 1, arrow=False)
        f.t(PX0 - 10, Y(e) + 4, "%.0e" % e, GY, size=12, anchor="end")
    Z = YB + 34                                         # 「＝ 0」那一行画在对数轴下面
    f.line(PX0, Z, PX1, Z, LINE2, 1, dash="4 4", arrow=False)
    f.t(PX0 - 10, Z + 4, "0", GY, size=12, anchor="end")
    for k in (1, 10, 20, 30, 40, 50, 60):
        f.t(XS(k), Z + 26, "%d" % k, GY, size=12, anchor="middle")
    f.t(PX1, Z + 44, "训练步", GY, size=12, anchor="end")
    f.path([(XS(k), Y(OG_GAP_RAND[k - 1])) for k in range(1, OG_STEPS + 1)], OR, 2, dash="6 4", arrow=False)
    f.path([(XS(k), Y(OG_GAP[k - 1])) for k in range(2, OG_STEPS + 1)], BL, 2.4, arrow=False)
    f.line(XS(1), Z, XS(2), Y(OG_GAP[1]), BL, 1.2, dash="2 3", arrow=False)
    f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="#fff" stroke="%s" stroke-width="2.4"/>' % (XS(1), Z, BL))
    f.t(XS(1) + 14, Z - 8, "第 1 步 ＝ 0：逐位相同", BL, True, 13)
    f.p.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>' % (XS(1), Y(OG_GAP_RAND[0]), OR))
    f.t(XS(1) + 12, Y(OG_GAP_RAND[0]) - 12, "随机初始化：第 1 步就是 %s" % N.tx(OG_GAP_RAND[0], "%.1e"), OR, True, 13)
    f.t(XS(2) + 10, Y(OG_GAP[1]) + 20, "第 2 步 %s" % N.tx(OG_GAP[1], "%.1e"), BL, True, 13)
    f.t(XS(40), Y(OG_PLATEAU) + 24, "几十步后约 %s" % N.tx(OG_PLATEAU, "%.1e"), BL, True, 13, "middle")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "权重一侧的开关，拿训练若干步以后的权重去测，或者看多步曲线；别拿第 1 步「逐位相同」当没用的证据", [
        "第 1 步：权重在格点上、激活是 BF16，两个 BF16 相乘精确，单遍 BF16 一位都没截掉，高精度多算的那几遍乘的全是 0。",
        "随机 FP32 初始化起训时不成立：第 0 步的权重就不在格点上，第 1 步就看得出差（橙虚线）。",
    ])
    yb = f.src(yb + 10,
               "📌 本图脚本现跑的玩具实验：线性模型 %d×%d，批 %d，MSE，Adam（lr %.0e，FP32 主权重），输入舍成 BF16；「BF16 检查点」＝ 一份标准差 %.2f 的随机权重先舍到 BF16。"
               "开关关 ＝ 权重先舍到 BF16 再乘；开关开 ＝ 权重按 FP32 乘；其余逐行相同。两次训练第 1 步的 loss 和第 1 次更新后的整份权重逐位相同（脚本里 assert），第 2 步起 loss 不同。"
               % (OG_K, OG_M, OG_B, OG_LR, OG_STD),
               "📌 第 1 次更新后离开格点的元素占 %.1f%%；右格纵轴 ＝ 同一份权重、同一批 2048 条 BF16 输入，开关开、关两种输出之差的 L2 范数除以输出的 L2 范数。"
               % (100 * OG_OFF[1]),
               "⚠️ 本课推导：BF16 有效位 8 位，两个相乘至多 16 位，FP32 有 24 位，乘积不舍入（脚本里对 10 万对随机数 assert 过）。"
               "高精度档把 FP32 拆成几份 BF16 分别乘再加，格点上的数拆出来只有第一份非零，其余几份乘出来全是 0；"
               "多遍和单遍逐位相同还要求那一份的累加顺序不变（推断）。",
               "注：前提是权重以 FP32 进矩阵乘、开关只改权重这一侧；要是框架在进矩阵乘之前就把权重转成了 BF16，这类开关从头到尾都不起作用。"
               "左格挑了 %d 个第 0 步正好落在这 5 个格点上的元素，每个格点三个；数轴位置是真实值，第 30 次更新后走出这一段的不画。玩具的差是多大没有意义，只看第 1 步是不是 0。" % (len(pick)))
    f.save("fig8-ongrid.svg", yb + 14)


fig_hooks()
fig_diverge()
fig_ongrid()
