# -*- coding: utf-8 -*-
r"""专题九 · 第九节补的「第二次迁移成绩单」（第二次迁移的新材料）。

fig9-mig2  回答：同一套方法换一个模型再走一遍，结果怎样？
          Ⓐ 小尺寸模型 1000 步 9 项（2026-11-11 起的现行比较：关 base2、尺子 GPU 16 份）：过 7、待定 2（主 loss 全程 1.55、后段 1.52），
             八份 TPU 主 loss 共同偏高约 4.3e-6，在查。
          Ⓑ GPU 自己换机器数、换切法（1 台 → 2 台、4 台，DP／EP／PP）：对单机尺子最大 1.76 倍，没有超出；学习率大 10% 的正对照 r 4149。
          Ⓒ 真实宽度三层：第 1 步 TPU 偏低定到 base2 常数舍成 BF16（logits ×1.0018），只关这个开关就到 −1.24e-6；1000 步回放开／关 base2 九项对比；后段偏低是轨迹差。
          数据（Ⓒ 是中间结果，其余实测）：对齐记录 小尺寸一节（1000 步现行比较的 8 份、16 份两份结果文件）、
                GPU 扩展一节（各运行对单机的结果文件）、真实宽度一节（汇总页、第 1 步对 FP64、前 164 步中间判档），全部实测。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU

W = 1400
MINI = [("主 loss（全程）", 1.55), ("主 loss（后段）", 1.52), ("MTP loss（全程）", 1.00), ("MTP loss（后段）", 1.00),
        ("索引器蒸馏 loss", 0.77), ("梯度范数", 0.81), ("参数范数", 0.55), ("路由偏置均值", 0.20), ("路由偏置标准差", 0.77)]
#   现行比较（2026-11-11 起）：对齐配置关 base2，TPU 运行对 GPU 基线，尺子 GPU 1000 步 16 份两两 120 对；每格三个统计里最差的 r（小尺寸对齐报告「训练曲线」表）
assert sum(1 for _n, a in MINI if a <= 1) == 7 and max(a for _n, a in MINI) == 1.55
GPUSCALE = [("1 台，原设置重跑", 0.93), ("1 台，EP 8", 0.95), ("2 台 DP16＋EP4（8 份）", [1.76, 0.90, 1.59, 0.95, 0.97, 1.17, 1.00, 0.81]),
            ("2 台 DP16＋EP16", 0.89), ("2 台 PP2", 1.25), ("4 台 PP2", 1.50)]
GPU_POS = 4149.37                     # 正对照：同 2 台设置、学习率大 10%
RW3_GPU_A, RW3_TPU, RW3_TPU_OFF = 5.76e-5, -9.87e-5, -1.24e-6   # 第 1 步主 loss 对 CPU FP64：GPU、TPU 开 base2、TPU 只关 base2（真实宽度一节、根源一览第 24 行）
assert abs(abs(RW3_TPU) / RW3_GPU_A - 1.71) < 0.01
RW3_1K = {"开 base2": (2, 4, 3), "关 base2": (3, 6, 0)}       # 1000 步回放 9 项：过／待定／不过
assert all(sum(v) == 9 for v in RW3_1K.values())
KAPPA_T = "1.0018"


def fig_mig2():
    f = Fig(W, "三块。左块九根横条：第二次迁移的小尺寸模型 1000 步九项检查，尺子是 GPU 16 份；七项在 1 以内，主 loss 全程 1.55、后段 1.52 待定，八份 TPU 的主 loss 都偏高约 4.3e-6，还在查。"
               "中块：GPU 自己换机器数和切法，每个点一次运行对单机尺子的最大倍数，最大 1.76，没有超过 2；学习率大 10% 的正对照是 4149 倍。"
               "右块：真实宽度三层，第 1 步主 loss 对 64 位真值 GPU 偏高 5.8e-5、TPU 偏低 9.9e-5；定到注意力的 base2 常数舍成 BF16、logits 整体乘 1.0018，只关这个开关 TPU 到 −1.2e-6。"
               "1000 步回放九项：开着过 2、待定 4、不过 3；关掉过 3、待定 6、不过 0。")
    y0 = f.header("第二次迁移：同一套方法再走一遍",
                  "左：小尺寸 1000 步成绩（尺子 GPU 16 份）；中：GPU 自己扩机器、换切法；右：真实宽度三层，第 1 步的差定了责，1000 步跑完",
                  [(BL, "过"), (OR, "待定"), (RD, "不过"), (GY2, "GPU 运行"), (GR, "关 base2 以后")])
    PH = 470
    AW, BW = 520, 430
    # Ⓐ
    py = f.panel(0, y0, AW, PH, "Ⓐ 小尺寸，1000 步九项（现行配置，实测）", BL)
    lx, x0, x1 = 18, 190, AW - 40

    def X(r):
        return x0 + r / 1.7 * (x1 - x0)
    f.box(X(1), py + 14, X(1.7) - X(1), 9 * 40 + 6, "#fef7e0", "#fef7e0", 0)
    f.line(X(1), py + 10, X(1), py + 14 + 9 * 40 + 10, GY, 1.2, dash="4 3", arrow=False)
    for i, (name, a) in enumerate(MINI):
        y = py + 20 + i * 40
        f.t(x0 - 10, y + 18, name, INK, a > 1, 13, "end")
        f.box(x0, y + 8, X(min(a, 1.7)) - x0, 18, OR if a > 1 else BL, OR if a > 1 else BL, 2)
        f.t(X(min(a, 1.7)) + 6, y + 22, "%.2f" % a, OR if a > 1 else GY, a > 1, 12.5)
    f.t(X(1), py + 14 + 9 * 40 + 30, "1", GY, size=13, anchor="middle")
    f.t(lx, py + PH - 70, "过 7、待定 2（主 loss 两个窗口）", INK, True, 13.5)
    f.t(lx, py + PH - 48, "八份 TPU 主 loss 都偏高约 4.3e-6，还在查", RD, size=12.5)
    f._pan = None
    # Ⓑ
    BX0 = AW + 22
    py2 = f.panel(BX0, y0, BW, PH, "Ⓑ GPU 换机器数、换切法（实测）", GY)
    gx0, gx1 = BX0 + 190, BX0 + BW - 24

    def GX(r):
        return gx0 + r / 2.2 * (gx1 - gx0)
    f.box(GX(0), py2 + 14, GX(1) - GX(0), 6 * 46, "#e6f4ea", "#e6f4ea", 0)
    f.box(GX(1), py2 + 14, GX(2) - GX(1), 6 * 46, "#fef7e0", "#fef7e0", 0)
    for v in (0, 1, 2):
        f.t(GX(v), py2 + 14 + 6 * 46 + 20, "%d" % v, GY, size=13, anchor="middle")
    for i, (name, v) in enumerate(GPUSCALE):
        y = py2 + 14 + i * 46 + 23
        f.t(gx0 - 8, y + 5, name, INK, False, 12.5, "end")
        for k, u in enumerate(v if isinstance(v, list) else [v]):
            f.box(GX(u) - 5, y - 5 + (k % 2) * 4 - 2, 10, 10, GY2, INK, 5)
    f.t(BX0 + 18, py2 + 14 + 6 * 46 + 52, "学习率大 10%% 的正对照：%d 倍（出了图）" % round(GPU_POS), RD, True, 13)
    f.t(BX0 + 18, py2 + 14 + 6 * 46 + 76, "存档续跑：跟不断开逐位相同", GR, True, 13)
    f.t(BX0 + 18, py2 + 14 + 6 * 46 + 98, "（两台机器，上万个状态张量逐个比）", GY, size=12.5)
    f._pan = None
    # Ⓒ
    CX0 = AW + BW + 44
    py3 = f.panel(CX0, y0, W - CX0, PH, "Ⓒ 真实宽度三层（实测）", OR)
    cx0, cx1 = CX0 + 30, W - 40
    lo, hi = -1.2e-4, 0.9e-4

    def CX(v):
        return cx0 + (v - lo) / (hi - lo) * (cx1 - cx0)
    f.t(CX0 + 16, py3 + 24, "第 1 步主 loss 对 CPU FP64 的相对差", INK, True, 13)
    yl = py3 + 76
    f.line(cx0, yl, cx1, yl, LINE, 1.2, arrow=False)
    f.line(CX(0), yl - 18, CX(0), yl + 18, INK, 1.4, arrow=False)
    f.t(CX(0), yl + 36, "0 ＝ 真值", GY, size=12.5, anchor="middle")
    f.box(CX(RW3_GPU_A) - 6, yl - 6, 12, 12, GY2, INK, 6)
    f.t(CX(RW3_GPU_A), yl - 16, "GPU", GY, True, 12.5, "middle")
    f.box(CX(RW3_TPU) - 6, yl - 6, 12, 12, RD, INK, 6)
    f.t(CX(RW3_TPU), yl - 16, "TPU", RD, True, 12.5, "middle")
    f.line(CX(RW3_TPU) + 8, yl + 14, CX(RW3_TPU_OFF) - 8, yl + 14, GR, 1.6)
    f.box(CX(RW3_TPU_OFF) - 6, yl + 8, 12, 12, GR, INK, 6)
    f.t(CX0 + 16, yl + 66, "TPU 偏低，定到注意力的 base2：", INK, True, 13)
    f.t(CX0 + 16, yl + 88, "常数 log₂e 舍成 BF16，logits 整体 ×%s" % KAPPA_T, INK, size=12.5)
    f.t(CX0 + 16, yl + 110, "只关这个开关：%.2e → %.2e" % (RW3_TPU, RW3_TPU_OFF), GR, True, 12.5)
    f.t(CX0 + 16, yl + 150, "1000 步回放九项（过／待定／不过）", INK, True, 13)
    for i, (k, (p_, q_, r_)) in enumerate(RW3_1K.items()):
        y = yl + 180 + i * 30
        f.t(CX0 + 16, y, k, INK, True, 12.5)
        for j, (n_, col) in enumerate(((p_, BL), (q_, OR), (r_, RD))):
            for m in range(n_):
                f.box(CX0 + 100 + (sum((p_, q_, r_)[:j]) + m) * 22, y - 12, 18, 16, col, col, 3)
    f.t(CX0 + 16, yl + 260, "关了以后不过清零；剩下的后段偏低，", GY, size=12.5)
    f.t(CX0 + 16, yl + 282, "在同一份权重上核过：是轨迹差，", GY, size=12.5)
    f.t(CX0 + 16, yl + 304, "不是前向差，来源还在查", GY, size=12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "换一个模型，方法照样走得通：小尺寸曲线大半对上，真实宽度第 1 步的差查到了根", [
        "还不能签字：小尺寸主 loss 两个窗口待定（八份 TPU 共同偏高，在查）、第 1 步有 3 个新冒出来的小参数要定责；真实宽度后段偏低是轨迹差，来源还在查。",
        "代价：小尺寸这一代的判法和尺子都是 2026-11-11 才换上的（关 base2、尺子 16 份）；上一代那两项已接受的结论不能套到这一代。",
    ])
    yb = f.src(yb + 10,
               "📌 Ⓐ 小尺寸（约 4.3 亿参数，保留新模型全部结构、缩宽度和层数，随机初始化）TPU v5p 8 设备 ↔ B200×8，1000 步，现行比较：对齐配置关 base2，被测对 GPU 基线一份，尺子 GPU 16 份两两 120 对；每格是三个统计里最差的 r。"
               "主 loss 1.55（带符号均值 TPU ＋7.21e-6，上限 4.65e-6）、后段 1.52、MTP 1.00、后段 1.00、索引器蒸馏 0.77、梯度范数 0.81、参数范数 0.55、路由偏置均值 0.20、标准差 0.77。"
               "开、关 base2 各 4 份重复，主 loss 都偏高、两组交叠：这份偏高跟 base2 无关，平均 ＋4.3e-6（GPU 每份对其余 15 份在 −2.63e-6 到 ＋2.34e-6），原因还在查。上一代（开 base2、尺子 8 份）是梯度范数 1.23、路由偏置标准差 1.14 待定，已接受。",
               "📌 Ⓑ 对单机 GPU golden 的前 300 步，尺子是单机 8 份两两 28 对；8 项检查取最差倍数。2 台 DP16＋EP4 跑了 8 份（最大 1.76，参数范数）；多机自己的尺子在主 loss、梯度范数、参数范数上是单机的 1.0–1.3 倍；两组对单机基线的窗口带符号均值没有系统偏移（Welch |t| ≤ 1.3）。"
               "正对照：同 2 台设置、学习率大 10%，8 项里 7 项超出，最大 4149 倍。存档续跑：确定性模式下「跑 20 步」对「跑 10 步、存、续跑 10 步」，2 台 PP2＋DP8＋EP8 的 8760 个状态张量、真实宽度 2 台的 14512 个状态张量都逐位相同。",
               "📌 Ⓒ 真实宽度第 0–2 层 ＋ MTP（3.21e10 参数），GPU 2 台跑 1000 步当 golden，TPU v5p 16 片原样回放：第 1 步主 loss 对 CPU FP64 GPU ＋5.76e-5、TPU −9.87e-5；"
               "注意力内核里 q 是 BF16，常数 log₂e 跟着舍成 1.4453125，logits（不含 sink）整体乘 1.0018；只关 base2 这一个开关，TPU −1.24e-6（CPU FP64 只加这个系数的移动跟实测相符）。"
               "1000 步回放九项：开 base2 过 2／待定 4／不过 3，关 base2 过 3／待定 6／不过 0。后段主 loss 偏低：同一份权重上 32 批对比，前向差不到一个标准误，后段的偏低全是轨迹差。"
               "索引器蒸馏那一项 GPU 自己偏高，约六成来自 GPU 把索引器的旋转位置编码放在 BF16 里算。所有尺寸的对齐配置因此都关了 base2。")
    f.save("fig9-mig2.svg", yb + 14)


fig_mig2()
