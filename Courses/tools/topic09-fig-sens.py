# -*- coding: utf-8 -*-
r"""专题九 · 第十节「尺子能抓多小」（第二次迁移的新材料）。

fig9-sens  回答：这把尺子能抓到多小的 bug？抓不到什么？
           Ⓐ 故意埋 bug、扫大小（TPU 自比 300 步，对 golden）：学习率乘 (1 ＋ ε)；MTP loss 的权重乘 (1 ＋ ε)（只动一部分梯度）。
              r 跟 ε 成正比（带符号的系统偏差）；ε 小到 3e-5 以下，淹在轨迹的混沌放大里，跟重跑分不开。
           Ⓑ 尺子量不出来的：几个动了精度的改动，跑到 1000 步、用 GPU 16 份当刻度，r 仍在纯噪声最大值以内。
           数据：对齐记录 TPU 自比一节的 MATRIX.md「S 灵敏度」「L 长跑」、README 脾气表（实测）。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE
import topic09_numbers as N

W = 1400
LR = [(1e-5, 1.21), (3e-5, 1.06), (1e-4, 4.11), (1e-3, 46.25), (1e-2, 461.42), (1e-1, 4505.22)]
MTP_MAIN = [(1e-3, 0.88), (1e-2, 6.18), (1e-1, 48.0)]          # 训练效果：主 loss 的 r
MTP_REC = [(1e-3, 50.12), (1e-2, 501.85), (1e-1, 5010.57)]    # 记下的 MTP loss 本身的 r
SLOPE = 4.1e4                                                  # 学习率：r ≈ 4.1e4 × ε（MATRIX S 原文）
for e, r in LR[2:]:
    assert abs(r / (SLOPE * e) - 1) < 0.15, (e, r)   # 「约等于」：1e-3 时 46.3 ÷ 41 ＝ 1.13
QUIET = [("少开「投影照 GPU 的 FP32 累加」", 1.64), ("SwiGLU 少舍一次", 1.37), ("索引器打分不用真 FP32", 1.06),
         ("指数等函数回到 TPU 默认近似", 1.07), ("Muon 正交化改 FP32", 0.97)]   # 分母统一为参照后的值
assert all(r < N.LOO16_1K_MAX for _n, r in QUIET)


def fig_sens():
    f = Fig(W, "两块。左块是对数对数图：横轴是故意埋的 bug 有多大，纵轴是 300 步曲线的 r。蓝线是学习率偏大，"
               "1e-4 时 r 4.1，之后每大十倍 r 也大十倍；3e-5 和 1e-5 时 r 只有 1.06 和 1.21，跟纯噪声分不开。"
               "橙色实线是 MTP loss 的权重偏大：对主 loss 的影响要到 1e-2 才显出来（r 6.2），可记下的 MTP loss 本身 1e-3 时 r 就是 50。"
               "右块五根横条：五个动了精度的改动跑满 1000 步，r 1.64、1.37、1.06、1.07、0.97，都没超过纯噪声最大值 1.82。")
    y0 = f.header("尺子能抓多小：学习率偏 1e-4 抓得到，偏 3e-5 就分不开",
                  "左：故意埋 bug、扫大小（300 步，对数对数）；右：尺子量不出来的几个精度改动（1000 步，GPU 16 份当刻度）",
                  [(BL, "学习率 × (1 ＋ ε)"), (OR, "MTP loss 权重 × (1 ＋ ε)：主 loss"), ("#f6c08d", "同上：记下的 MTP loss"), ("#e8eaed", "纯噪声能到的范围")])
    PH = 430
    LW = 780
    py = f.panel(0, y0, LW, PH, "Ⓐ 故意埋 bug：r 跟 bug 的大小成正比，太小就淹进噪声", BL)
    ax0, ax1, ay0, ay1 = 90, LW - 40, py + 20, py + PH - 80
    XL, XH, YL, YH = 5e-6, 2e-1, 0.5, 1e4

    def X(v):
        return ax0 + (math.log10(v) - math.log10(XL)) / (math.log10(XH) - math.log10(XL)) * (ax1 - ax0)

    def Y(v):
        return ay1 - (math.log10(v) - math.log10(YL)) / (math.log10(YH) - math.log10(YL)) * (ay1 - ay0)
    f.box(ax0, Y(N.LOO8_300_MAX), ax1 - ax0, ay1 - Y(N.LOO8_300_MAX), "#e8eaed", "#e8eaed", 0)
    f.t(ax1 - 8, Y(N.LOO8_300_MAX) + 18, "纯噪声能到 %s（300 步）" % N.LOO8_300_MAX_T, GY, True, 13, "end")
    for v in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        f.line(X(v), ay0, X(v), ay1, LINE, 0.8, arrow=False)
        f.t(X(v), ay1 + 20, "1e%d" % round(math.log10(v)), GY, size=13, anchor="middle")
    for v in (1, 10, 100, 1000, 10000):
        f.line(ax0, Y(v), ax1, Y(v), LINE, 0.8, arrow=False)
        f.t(ax0 - 8, Y(v) + 5, "%g" % v, GY, size=13, anchor="end")
    f.t((ax0 + ax1) / 2, ay1 + 44, "埋进去的 bug 有多大：ε", GY, size=13, anchor="middle")
    f.t(ax0 - 76, ay0 + 4, "r", GY, True, 14)
    # 理想正比线（学习率）
    f.line(X(5e-5), Y(SLOPE * 5e-5), X(2e-1), Y(min(SLOPE * 2e-1, YH)), BL, 1.0, dash="3 3", arrow=False)
    f.path([(X(e), Y(r)) for e, r in MTP_REC], "#f6c08d", 2.0, dash="5 4", arrow=False)
    f.path([(X(e), Y(r)) for e, r in MTP_MAIN], OR, 2.2, arrow=False)
    f.path([(X(e), Y(r)) for e, r in LR], BL, 2.6, arrow=False)
    for e, r in MTP_REC:
        f.box(X(e) - 4, Y(r) - 4, 8, 8, "#fff", "#f6c08d", 4, sw=1.8)
    for e, r in MTP_MAIN:
        f.box(X(e) - 5, Y(r) - 5, 10, 10, OR, OR, 5)
    for e, r in LR:
        f.box(X(e) - 5.5, Y(r) - 5.5, 11, 11, BL, INK, 5.5)
    f.t(X(1e-4) + 10, Y(4.11) + 18, "1e-4：r 4.1，抓得到", BL, True, 13)
    f.t(X(1e-5) - 10, Y(1.21) - 22, "3e-5 以下：分不开", GY, True, 13)
    f.t(X(1e-3) - 12, Y(50.12) - 14, "记下的 MTP loss：跟蓝线几乎重合", "#c26401", True, 13, "end")
    f.t(X(1e-3) + 10, Y(0.88) + 20, "训练效果要到 1e-2 才显", OR, True, 13)
    f.t(X(2e-2) + 6, Y(SLOPE * 2e-2) + 40, "r ≈ 4.1e4 × ε", BL, size=13)
    f._pan = None
    RX = LW + 24
    py2 = f.panel(RX, y0, W - RX, PH, "Ⓑ 动了精度，尺子却量不出来（1000 步）", RD)
    bx0, bx1 = RX + 24, W - 60

    def BX(r):
        return bx0 + r / 2.2 * (bx1 - bx0)
    for i, (name, r) in enumerate(QUIET):
        y = py2 + 30 + i * 62
        f.t(bx0, y + 14, name, INK, True, 13.5)
        f.box(bx0, y + 24, BX(r) - bx0, 20, BL, BL, 3)
        f.t(BX(r) + 8, y + 40, "%.2f" % r, BL, True, 13)
    f.line(BX(N.LOO16_1K_MAX), py2 + 22, BX(N.LOO16_1K_MAX), py2 + PH - 60, INK, 1.4, dash="4 3", arrow=False)
    f.t(BX(N.LOO16_1K_MAX) - 6, py2 + PH - 44, "纯噪声最大 %s" % N.LOO16_1K_MAX_T, INK, True, 13, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "全局的系统偏差，相对大小 1e-4 就抓得到；再小，或者只动一小块，曲线就看不出来", [
        "所以记下的每一项 loss 都要逐项比，别只看总 loss：只动一部分梯度的 bug，训练效果藏得住，记下的那一项藏不住。",
        "代价：右边五个动了精度的改动，1000 步曲线分不开，第 1 步对 FP64 也没抓到 —— 分不开不等于没影响，要看累积效果得几个一起关。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，TPU v5p 自比，300 步对 golden，r 用 GPU 8 份两两 28 对的上限）：学习率 × (1 ＋ ε)，ε ＝ 1e-5 ／ 3e-5 ／ 1e-4 ／ 1e-3 ／ 1e-2 ／ 1e-1，"
               "曲线最差 r 1.21 ／ 1.06 ／ 4.11 ／ 46.3 ／ 461 ／ 4505，ε ≥ 3e-5 时领头的是主 loss 带符号均值（1e-5 时是路由偏置标准差）；ε ≥ 1e-4 时 r ≈ 4.1e4 × ε。"
               "MTP loss 权重 × (1 ＋ ε)，ε ＝ 1e-3 ／ 1e-2 ／ 1e-1：主 loss r 0.88 ／ 6.18 ／ 48.0，记下的 MTP loss 本身 50 ／ 502 ／ 5011。另：loss 多乘 1.01（漏除常数），梯度范数 r 94.5、主 loss 5.23（分母统一为参照后；统一前 93.9）。",
               "📌 右块：对 golden 的 1000 步比较，刻度 GPU 1000 步 16 份两两 120 对，纯噪声（GPU 留一法）最大 1.82；五个都是「动了精度」的改动，300 步时 r 0.91–1.06。",
               "⚠️ 「抓得到的下限在 3e-5 到 1e-4 之间」只对学习率这类全局偏差、这个模型、300 步成立。")
    f.save("fig9-sens.svg", yb + 14)


fig_sens()
