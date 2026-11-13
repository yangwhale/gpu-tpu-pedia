# -*- coding: utf-8 -*-
r"""专题八 · 第九节补的「第二次迁移成绩单」（第二次迁移的新材料）。

fig8-mig2  回答：同一套方法换一个模型再走一遍，结果怎样？
          Ⓐ 小尺寸模型 1000 步 9 项（2026-11-11 起的现行比较：关 base2、尺子 GPU 16 份）：过 7、待定 2（主 loss 全程 1.55、后段 1.52），
             TPU 开、关 base2 各 4 份共八份，主 loss 全部偏高、平均约 4.3e-6，在查。
          Ⓑ GPU 自己换机器数、换切法（1 台 → 2 台、4 台，DP／EP／PP）：对单机尺子最大 1.76 倍，没有超出；学习率大 10% 的正对照 r 4149。
          Ⓒ 真实宽度三层：第 1 步 TPU 偏低定到 base2 常数舍成 BF16（logits ×1.0018），只关这个开关就到 −1.24e-6；1000 步回放开／关 base2 九项对比；后段偏低是轨迹差。
          数据（全部实测）：对齐记录 小尺寸一节（1000 步现行比较的 8 份、16 份两份结果文件）、
                GPU 扩展一节（各运行对单机的结果文件）、真实宽度一节（汇总页、第 1 步对 FP64、前 164 步中间判档），全部实测。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU
Fig.BAND_KEEP_LINES = 0   # 锤炼阶段（2026-11-14）：图上落点带只留一行粗体结论，其余进图下折叠
import topic08_numbers as N
MI_SD = N.MI_SD
FC_SD_ = N.FC_SD

W = 1400
MINI = [("主 loss（全程）", 0.98), ("主 loss（后段）", 0.97), ("MTP loss（全程）", 0.81), ("MTP loss（后段）", 0.80),
        ("索引器蒸馏 loss", 0.75), ("梯度范数", 0.79), ("参数范数", 0.14), ("路由偏置均值", 1.31), ("路由偏置标准差", 0.85)]
#   现行比较（2026-11-11 起 C 口径）：TPU 第一份对 GPU 基线，尺子 GPU 1000 步 16 份两两 120 对；每格三个统计里最差的 r（小尺寸现行比较的结果文件，逐项复算）
assert sum(1 for _n, a in MINI if a <= 1) == N.C_NOW["pass"] and max(a for _n, a in MINI) == 1.31 and MINI[0][1] == N.C_NOW["loss_r"]
GPUSCALE = [("1 台，原设置重跑", 0.93), ("1 台，EP 8", 0.95), ("2 台 DP16＋EP4（8 份）", [1.76, 0.90, 1.59, 0.95, 0.97, 1.17, 1.00, 0.81]),
            ("2 台 DP16＋EP16", 0.89), ("2 台 PP2", 1.25), ("4 台 PP2", 1.50)]
GPU_POS = 4149.37                     # 正对照：同 2 台设置、学习率大 10%
RWIDE_GPU_A, RWIDE_TPU, RWIDE_TPU_OFF = 5.76e-5, -9.87e-5, -1.24e-6   # 第 1 步主 loss 对 CPU FP64：GPU、TPU 开 base2、TPU 只关 base2（真实宽度一节、根源一览第 24 行）
assert abs(abs(RWIDE_TPU) / RWIDE_GPU_A - 1.71) < 0.01
RWIDE_1K = {"开 base2": (2, 4, 3), "关 base2": (3, 6, 0)}       # 1000 步回放 9 项：过／待定／不过
assert all(sum(v) == 9 for v in RWIDE_1K.values())
KAPPA_T = "1.0018"


def fig_mig2():
    f = Fig(W, "三块。左块九根横条：第二次迁移的小尺寸模型 1000 步九项检查，尺子是 GPU 16 份；八项在 1 以内，主 loss 0.98，刚好过线；路由偏置均值 1.31 待定，同配置另五份在 0.36 到 0.80。"
               "中块：GPU 自己换机器数和切法，每个点一次运行对单机尺子的最大倍数，最大 1.76，没有超过 2；学习率大 10% 的正对照是 4149 倍。"
               "右块：真实宽度三层，第 1 步主 loss 对 64 位真值 GPU 偏高 5.8e-5、TPU 偏低 9.9e-5；定到注意力的 base2 常数舍成 BF16、logits 整体乘 1.0018，只关这个开关 TPU 到 −1.2e-6。"
               "1000 步回放九项：开着过 2、待定 4、不过 3；关掉过 3、待定 6、不过 0。")
    y0 = f.header("第二次迁移：同一套方法再走一遍",
                  "左：小尺寸 1000 步成绩（尺子 GPU 16 份）；中：GPU 自己扩机器、换切法；右：真实宽度三层，第 1 步的差定了责，1000 步跑完",
                  [(BL, "过"), (OR, "待定"), (RD, "不过"), (GY2, "GPU 运行"), (GR, "关 base2 以后")])
    PH = 470
    AW, BW = 520, 430
    # Ⓐ
    py = f.panel(0, y0, AW, PH, "Ⓐ 小尺寸，1000 步九项（现行对齐配置，实测）", BL)
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
    f.t(lx, py + PH - 70, "过 8、待定 1（路由偏置均值；另五份 0.36–0.80）", INK, True, 13.5)
    f.t(lx, py + PH - 48, "主 loss 0.98 刚好过线；后段跟不修还分不开", OR, size=12.5)
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
    f.box(CX(RWIDE_GPU_A) - 6, yl - 6, 12, 12, GY2, INK, 6)
    f.t(CX(RWIDE_GPU_A), yl - 16, "GPU", GY, True, 12.5, "middle")
    f.box(CX(RWIDE_TPU) - 6, yl - 6, 12, 12, RD, INK, 6)
    f.t(CX(RWIDE_TPU), yl - 16, "TPU", RD, True, 12.5, "middle")
    f.line(CX(RWIDE_TPU) + 8, yl + 14, CX(RWIDE_TPU_OFF) - 8, yl + 14, GR, 1.6)
    f.box(CX(RWIDE_TPU_OFF) - 6, yl + 8, 12, 12, GR, INK, 6)
    f.t(CX0 + 16, yl + 66, "TPU 偏低，定到注意力的 base2：", INK, True, 13)
    f.t(CX0 + 16, yl + 88, "常数 log₂e 舍成 BF16，logits 整体 ×%s" % KAPPA_T, INK, size=12.5)
    f.t(CX0 + 16, yl + 110, "只关这个开关：%.2e → %.2e" % (RWIDE_TPU, RWIDE_TPU_OFF), GR, True, 12.5)
    f.t(CX0 + 16, yl + 150, "1000 步回放九项（过／待定／不过）", INK, True, 13)
    for i, (k, (p_, q_, r_)) in enumerate(RWIDE_1K.items()):
        y = yl + 180 + i * 30
        f.t(CX0 + 16, y, k, INK, True, 12.5)
        for j, (n_, col) in enumerate(((p_, BL), (q_, OR), (r_, RD))):
            for m in range(n_):
                f.box(CX0 + 100 + (sum((p_, q_, r_)[:j]) + m) * 22, y - 12, 18, 16, col, col, 3)
    f.t(CX0 + 16, yl + 260, "关了以后不过清零；剩下的后段偏低，", GY, size=12.5)
    f.t(CX0 + 16, yl + 282, "在同一份权重上核过：是轨迹差，", GY, size=12.5)
    f.t(CX0 + 16, yl + 304, "不是前向差，来源还在查", GY, size=12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "换一个模型，方法照样走得通：小尺寸九项八项对上，真实宽度第 1 步的差查到了根", [
        "还不能签字：小尺寸新配置下第 1 步还有 1 个小参数待定、主 loss 刚好过线、后段跟不修分不开；真实宽度后段偏低是轨迹差，来源还在查。",
        "代价：小尺寸这一代的配置、判法和尺子都是 2026-11-11 才换上的（关 base2、多路残差一带照 GPU 舍、尺子 16 份）；上一代已接受的结论不能套到这一代。",
    ])
    yb = f.src(yb + 10,
               "📌 Ⓐ 小尺寸（约 4.3 亿参数，保留新模型全部结构、缩宽度和层数，随机初始化）TPU v5p 8 设备 ↔ B200×8，1000 步，现行比较：对齐配置关 base2、多路残差一带的舍入照 GPU 写法，被测是事先定的第一份，尺子 GPU 16 份两两 120 对；每格是三个统计里最差的 r。"
               "主 loss 0.98（最差是平均绝对相对误差）、后段 0.97、MTP 0.81、后段 0.80、索引器蒸馏 0.75、梯度范数 0.79、参数范数 0.14、路由偏置均值 1.31、标准差 0.85；同配置 6 份都判，没有不过；另一份主 loss、MTP 略超（1.14、1.07），一份梯度范数略超（1.04）。"
               "上一代（现行对齐配置之前）主 loss 1.55、后段 1.52 待定：TPU 8 份主 loss 都偏高、平均 ＋4.3e-6，主因后来定到流梯度的舍法（§9.7）。",
               "📌 Ⓑ 对单机 GPU golden 的前 300 步，尺子是单机 8 份两两 28 对；8 项检查取最差倍数。2 台 DP16＋EP4 跑了 8 份（最大 1.76，参数范数）；多机自己的尺子在主 loss、梯度范数、参数范数上是单机的 1.0–1.3 倍；两组对单机基线的窗口带符号均值没有系统偏移（Welch |t| ≤ 1.3）。"
               "正对照：同 2 台设置、学习率大 10%，8 项里 7 项超出，最大 4149 倍。存档续跑：确定性模式下「跑 20 步」对「跑 10 步、存、续跑 10 步」，2 台 PP2＋DP8＋EP8 的 8760 个状态张量、真实宽度 2 台的 14512 个状态张量都逐位相同。",
               "📌 Ⓒ 真实宽度第 0–2 层 ＋ MTP（3.21e10 参数），GPU 2 台跑 1000 步当 golden，TPU v5p 16 片原样回放：第 1 步主 loss 对 CPU FP64 GPU（基线那份）＋5.76e-5（4 份 ＋5.1e-5 到 ＋5.8e-5）、TPU −9.87e-5；"
               "注意力内核里 q 是 BF16，常数 log₂e 跟着舍成 1.4453125，logits（不含 sink）整体乘 1.0018；只关 base2 这一个开关，TPU −1.24e-6（CPU FP64 只加这个系数的移动跟实测相符）。"
               "1000 步回放九项：开 base2 过 2／待定 4／不过 3，关 base2 过 3／待定 6／不过 0。后段主 loss 偏低：同一份权重上 32 批对比，前向差不到一个标准误，后段的偏低全是轨迹差。"
               "索引器蒸馏那一项 GPU 自己偏高，约六成来自 GPU 把索引器的旋转位置编码放在 BF16 里算。所有尺寸的对齐配置因此都关了 base2。")
    f.save("fig8-mig2.svg", yb + 14)


def fig_minidrift():
    """fig8-minidrift 回答：小尺寸 TPU 主 loss 那截共同偏高，是从哪一步开始的、是不是前向的差？
    上：主 loss，TPU 8 份对 GPU 16 份逐步均值的相对差，25 步一块（柱）；灰带 ＝ ± GPU 留一法标准差。
    下：同一批运行的 MTP loss，同样画法（反着走）。数据：topic08_numbers 的 MD（实测）。"""
    MD = N.MD
    f = Fig(W, "上下两条。上：小尺寸 1000 步，TPU 8 份主 loss 对 GPU 16 份逐步均值的相对差，每 25 步一根柱，灰带是 GPU 自己的散布。"
               "第 1 步 −2.0e-6；第 51 到 100 步一个小鼓包；第 100 到 550 步在 0 附近；第 550 步起一路往上，到最后一块 ＋1.8e-5。"
               "下：同一批运行的 MTP loss，第 100 步起反着往下，最后 −3.2e-5。前段鼓包定到多路残差收拢那次少舍的 BF16；后段主因后来定到流梯度的舍法。")
    y0 = f.header("小尺寸主 loss 那截偏高：几乎全在第 550 步以后，越往后越大",
                  "TPU 8 份对 GPU 16 份逐步均值的相对差，25 步一块；灰带 ＝ ± GPU 留一法标准差（实测）",
                  [(OR, "8 份同号偏高的块"), (BL, "8 份同号偏低的块"), (GY2, "其余块"), (LINE, "± GPU 散布")])
    X0, XW = 120, W - 170

    def X(step):
        return X0 + (step - 1) / 999.0 * XW

    def strip(y, h, key, vmax, title, col):
        py = f.panel(0, y, W, h, title, col)
        mid = py + (h - 30) / 2.0
        k = (h - 30) / 2.0 - 16

        def Y(v):
            return mid - v / vmax * k
        for b in MD[key]["blocks"]:
            a, c = (int(x) for x in b[0].split("-"))
            sd = b[2]
            f.box(X(a), Y(sd), X(c) - X(a), Y(-sd) - Y(sd), "#eceff1", "#eceff1", 0)
        f.line(X0, mid, X0 + XW, mid, GY2, 1, arrow=False)
        tk = 1e-5 if vmax < 3e-5 else 2e-5
        for v in (tk, -tk):
            f.t(X0 - 8, Y(v) + 4, ("%+.0e" % v).replace("e-0", "e-"), GY, size=11.5, anchor="end")
        f.t(X0 - 8, mid + 4, "0", GY, size=11.5, anchor="end")
        for b in MD[key]["blocks"]:
            a, c = (int(x) for x in b[0].split("-"))
            v, n = b[1], b[3]
            fill = OR if n == 8 and v > 0 else (BL if n == 0 and v < 0 else "#bdc1c6")
            yv = Y(v)
            f.box(X(a) + 2, min(yv, mid), X(c) - X(a) - 4, abs(yv - mid), fill, fill, 1)
        return py, Y, mid
    PH = 250
    py, Y, mid = strip(y0, PH, "主loss", 2.0e-5, "主 loss：TPU 减 GPU（相对差）", RD)
    for st in (1, 250, 500, 750, 1000):
        f.t(X(st), py + PH - 36, "第 %d 步" % st, GY, size=12, anchor="middle")
    f.t(X(1) + 6, Y(N.MD_STEP1) + 22, "第 1 步 %s" % N.MD_STEP1_T, INK, True, 12.5)
    f.t(X(60), Y(5.2e-6) - 10, "小鼓包：已定位（见正文）", GY, True, 12.5)
    f.t(X(260), Y(-6e-6), "第 100–550 步在 0 附近，正负相抵", GY, True, 12.5)
    f.t(X(560), Y(1.75e-5) - 4, "第 550 步起一路往上，最后 25 步平均 %s" % N.MD_LAST_T, OR, True, 13)
    f._pan = None
    y1 = py + PH - 30 + 14
    PH2 = 210
    py2, Y2, mid2 = strip(y1, PH2, "MTP", 5.0e-5, "同一批运行的 MTP loss：反着走（纵轴刻度是上图的 2.5 倍）", BL)
    f.t(X(470), Y2(3.4e-5), "第 751–1000 步 %s，8 份全部偏低（MTP 的尺子宽 4 倍，判档能过）" % N.MD_MTP_LATE_T, BL, True, 12.5)
    f._pan = None
    yb = f.band(py2 + PH2 - 30 + 20, "warn", "是训练轨迹慢慢偏开，不是前向的差", [
        "同一份权重上两边前向几乎没差（不到后段的一成）；下面一处处排除。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，TPU v5p 8 设备 ↔ B200×8，1000 步，现行对齐配置之前那一代）：TPU 8 份 ＝ 开、关 base2 各 4 份（其中 6 份只改执行方式）；每一步取 8 份对 GPU 16 份逐步均值的相对差再平均，25 步一块；灰带是 GPU 16 份轮流当被测、对其余 15 份均值的标准差。"
               "窗口值：第 2–1000 步 ＋4.3e-6，第 751–1000 步 %s，MTP 同窗口 %s。隔离实验（第 551–1000 步，TPU 对同配置 GPU 4 份均值，每种 TPU 2 份）：%s。" % (N.MD_LATE_T, N.MD_MTP_LATE_T, N.MD_ISO_T),
               "📌 前段鼓包：多路残差收拢成一路的地方，GPU 前向舍一次 BF16、反向传回的梯度也舍一次；TPU 省掉了梯度那次（最小复现的后端指令里核过，打包 0 条）。加一个强制舍入的开关（前向、梯度都舍），第 51–100 步回到 GPU 附近，第 751–1000 步不变；只在前向舍、梯度原样传回，鼓包又回来。",
               "📌 同一份权重上的前向：6 个状态、19 批，TPU 减 GPU 写法 %s；偏差跟步数、参数范数同步（第 501 步起相关 %.2f），跟梯度范数无关。关掉 MTP、KD 后幅度约为对照的六成，是它们放大了还是样本少，在查。" % (N.MD_SAME_STATE_T, N.MD_CORR[0]),
               "📌 后段（第 551–1000 步，对 GPU 16 份均值）：修之前 4 份均值 %s，补流梯度舍法（流的梯度逐个舍、分支输出也舍）5 份均值 ＋1.9e-6；" % N.FC_MEAN_T["修之前"].replace("+", "＋") + "把它拆成两半各跑 1 份：只补梯度累加 ＋0.6e-6、只补分支输出 ＋3.0e-6；两边都去掉多路残差，TPU 2 份 ＋1.9e-6、＋0.7e-6。延长到 2000 步：尾部在第 1000–1250 步到顶（＋2.9e-5）后回落到 GPU 两份的散布里。",
               "⚠️ 柱的颜色只看 8 份的符号是否一致，不是判档；那一代判档按第三节的三个统计和 GPU 16 份的尺子是主 loss 1.55、1.52 待定，现行对齐配置下主 loss 过。")
    f.save("fig8-minidrift.svg", yb + 14)


def fig_ihcroot():
    """fig8-ihcroot 回答：小尺寸主 loss 后段为什么偏高？先排除法缩范围：两边写法差在多路残差的流梯度怎么加。
    数据：topic08_numbers 的 MI、MD_ISO（实测；「旧 GPU 口径」＝ 按现在这份 GPU 参照的写法）。"""
    MI = N.MI
    f = Fig(W, "两块。左：多路残差的流在反向里的样子。流在一层里有三处用到，回来三份梯度。GPU：三份各舍 BF16，再一个一个加，每加一次舍一次；"
               "TPU：三份在 FP32 里加完，最后舍一次（推断）。"
               "右：第 551 到 1000 步主 loss 的 TPU 减 GPU。关掉 MTP、关掉索引器蒸馏、两个都关，都还偏高 0.9e-5 到 1.2e-5；对照 1.48e-5；"
               "两边都去掉多路残差，1.9e-6 和 0.7e-6，落进 GPU 的散布 2.1e-6。所以范围缩到多路残差这一段。")
    y0 = f.header("后段偏高：排除了一圈，范围缩到多路残差这一段",
                  "第二次迁移，小尺寸，v5p，现行对齐配置之前那一代；对比基准是 GPU 现行写法",
                  [(OR, "TPU 偏高"), (GR, "两边都去掉多路残差"), (GY2, "排除的候选")])
    PH = 360
    LW = 620
    py = f.panel(0, y0, LW, PH, "Ⓐ 嫌疑人：流的三份梯度怎么加（下一步验证）", BL)
    rows = [("GPU", BL, [("三份各舍 BF16", BL, False), ("逐个相加", BL, False), ("每加一次舍一次", BL, False)], "CPU 上用 GPU 同一段代码逐位核过"),
            ("TPU", OR, [("三份留 FP32", OR, True), ("FP32 里加完", OR, True), ("最后舍一次", OR, True)], "推断：省舍入下中间那几次被合掉（虚线）")]
    for i, (lab, col, steps, note) in enumerate(rows):
        y = py + 40 + i * 130
        f.t(20, y + 30, lab, col, True, 16)
        x = 80
        for j, (t, c, dash) in enumerate(steps):
            w = 160 if len(t) > 6 else 120
            f.box(x, y + 8, w, 40, "#fff", c, 8, sw=1.6, dash="5 4" if dash else None)
            f.t(x + w / 2, y + 33, t, c, True, 13.5, "middle")
            if j < len(steps) - 1:
                f.line(x + w, y + 28, x + w + 26, y + 28, GY, 1.4)
            x += w + 30
        f.t(80, y + 78, note, col, True, 13.5)
    f.t(20, py + PH - 76, "每次舍入的差都很小，1000 步攒下来成了那截偏高。", INK, True, 13.5)
    f.t(20, py + PH - 50, "是不是它，靠后面两张图两头各改一处来验。", GY, size=12.5)
    f._pan = None
    RX = LW + 20
    py2 = f.panel(RX, y0, W - RX, PH, "Ⓑ 第 551–1000 步主 loss，TPU 减 GPU", OR)
    bars = [("关掉 MTP", N.MD_ISO["关MTP"], GY2, None), ("关掉索引器蒸馏", N.MD_ISO["关KD"], GY2, None), ("两个都关", N.MD_ISO["都关"], GY2, None),
            ("对照（1 份）", MI["对照"]["tpu_mean"], OR, None),
            ("两边都去掉多路残差（2 份）", MI["两边都去掉多路残差"]["tpu_mean"], GR, MI["两边都去掉多路残差"]["tpu_each"])]
    BX0, BX1, vmax = RX + 270, W - 110, 1.6e-5

    def BX(v):
        return BX0 + max(v, 0) / vmax * (BX1 - BX0)
    f.box(BX0, py2 + 14, BX(MI_SD) - BX0, 5 * 40 + 10, "#eceff1", "#eceff1", 0)
    f.t(BX(MI_SD) + 4, py2 + 5 * 40 + 40, "GPU 自己的散布 " + ("%.1e" % MI_SD).replace("e-0", "e-"), GY, size=12)
    for i, (lab, v, col, each) in enumerate(bars):
        y = py2 + 22 + i * 40
        f.t(RX + 14, y + 19, lab, col if col != GY2 else GY, True, 13)
        f.box(BX0, y + 4, max(BX(v) - BX0, 3), 22, col, col, 4)
        if each:
            for e in each:
                f.box(BX(e) - 3, y + 12, 6, 6, "#fff", INK, 3)
        xe = max([BX(v)] + [BX(e) for e in (each or [])])
        txt = ("%.2e" % v if lab.startswith("对照") else "%.1e" % v).replace("e-0", "e-")
        f.t(xe + 10, y + 20, "＋" + txt, col if col != GY2 else GY, True, 13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "warn", "关掉 MTP、蒸馏都还偏；两边都换成普通残差就不飘 —— 问题在多路残差这一段", [
        "排除法只能缩小范围，定不了机制；后面两张图在这一段里两头各改一处。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，TPU v5p，1000 步）：第 551–1000 步主 loss 对 GPU 均值的带符号均值。对照是一份运行（后两张图的修之前四份含它），对 GPU 16 份均值；去多路残差那组对它自己的 GPU 3 份；关 MTP、关蒸馏那三行对同配置的 GPU 4 份均值。"
               "GPU 侧索引器旋转位置编码改 FP32 也一样飘；TPU 自比里换运行库、换 v7x、换并行方式等 35 个 1000 步运行，第 751–1000 步主 loss 全部偏高。",
               "📌 GPU 那边的写法在 CPU 上用 GPU 同一段代码逐位核过（三份 BF16 梯度按分组相加、每加一次舍一次）；TPU 这几处是否被省掉是推断，没看后端指令；GPU 一边照这个推断改写就长出同样的偏（后两张图）。",
               "⚠️ 标准答案按现在这份 GPU 参照的写法；对齐记录一度凭单边几份运行写成定案、又撤回，后来每格补到 4–5 份、两头各改一处做两样本检验才定下。")
    f.save("fig8-ihcroot.svg", yb + 14)


def fig_fixcmp():
    """fig8-fixcmp 回答：把 GPU 的流梯度舍法补到 TPU 上，那截偏高回不回来？三列同一纵轴：修之前 ／ 补流梯度舍法 ／ 对齐配置（C 口径）。
    数据：topic08_numbers 的 FC（各运行逐步记录，私有导出；实测）。"""
    FC = N.FC
    f = Fig(W, "两列，纵轴刻度相同：主 loss 的 TPU 减 GPU（对 GPU 16 份逐步均值的相对差，25 步滑动平均），灰带是 GPU 每份对其余 15 份的同样偏差的范围。"
               "修之前四份：前 550 步在 0 上下，之后一路往上，四份尾巴都在。"
               "补上 GPU 的流梯度舍法五份：均值降到 ＋1.9e-6，两份仍超尾巴的阈值，三份贴着 0。")
    y0 = f.header("把 GPU 的流梯度舍法补上，尾巴降下来：均值 ＋1.9e-6（5 份里 2 份仍有尾巴）",
                  "第二次迁移，小尺寸，v5p，1000 步；两列同一纵轴；每条线一份运行",
                  [(RD, "修之前"), (GR, "补流梯度舍法"), (GY2, "GPU 自己的散布")])
    cols = [("修之前", "修之前（4 份）", RD), ("补流梯度舍法", "补流梯度舍法（5 份）", GR)]
    PW, GAP, PH = 689, 22, 430
    vmin, vmax = -2.0e-5, 4.0e-5
    band = FC["band"]
    for ci, (key, title, col) in enumerate(cols):
        x0 = ci * (PW + GAP)
        py = f.panel(x0, y0, PW, PH, title, col)
        X0, X1 = x0 + 64, x0 + PW - 16
        YT, YB = py + 24, py + PH - 134

        def X(st):
            return X0 + (st - 1) / 999.0 * (X1 - X0)

        def Y(v):
            return YT + (vmax - v) / (vmax - vmin) * (YB - YT)
        up = [(X(i + 1), Y(min(b, vmax))) for i, b in enumerate(band) if 12 <= i <= 987 and i % 5 == 0]
        dn = [(X(i + 1), Y(max(-b, vmin))) for i, b in enumerate(band) if 12 <= i <= 987 and i % 5 == 0]
        f.poly("M " + " L ".join("%.1f %.1f" % p for p in up + dn[::-1]) + " Z", "#e8eaed", "none")
        f.line(X0, Y(0), X1, Y(0), GY2, 1, arrow=False)
        for v in (-2e-5, -1e-5, 1e-5, 2e-5, 3e-5, 4e-5):
            f.line(X0, Y(v), X1, Y(v), LINE, 0.6, arrow=False)
            if ci == 0:
                f.t(X0 - 6, Y(v) + 4, ("%+.0e" % v).replace("e-0", "e-"), GY, size=11.5, anchor="end")
        if ci == 0:
            f.t(X0 - 6, Y(0) + 4, "0", GY, size=11.5, anchor="end")
        f.box(X(551), YT, X(1000) - X(551), YB - YT, "none", LINE, 0, dash="3 3")
        for st in (1, 500, 1000):
            f.t(X(st), YB + 18, "%d" % st, GY, size=11.5, anchor="middle")
        runs = FC["cols"][key]
        for r in runs:
            pts = [(X(i + 1), Y(min(max(v, vmin), vmax))) for i, v in enumerate(r["ma"]) if 12 <= i <= 987 and i % 4 == 0]
            f.path(pts, col, 1.6, arrow=False)
        f.t(x0 + 16, YB + 46, "第 551–1000 步平均（每份）：", INK, True, 13)
        ws = sorted((r["w551"] for r in runs), reverse=True)
        f.t(x0 + 16, YB + 68, "、".join(("%+.1e" % w).replace("e-0", "e-") for w in ws), col, True, 12.5, w=PW - 30)
        f.t(x0 + 16, YB + 92, "均值 %s；尾巴还在：%d 份 ／ %d 份" % (N.FC_MEAN_T[key], N.FC_TAIL[key], len(runs)), col, True, 13.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "补上 GPU 的流梯度舍法，尾巴降下来", [
        "均值从 %s 降到 %s（两样本检验 p %s）；只补一边还不算数，GPU 一边反过来改，见下一张图。" % (N.FC_MEAN_T["修之前"], N.FC_MEAN_T["补流梯度舍法"], N.TW_P_T["tpu"]),
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，TPU v5p 8 设备，1000 步）：主 loss 对 GPU 16 份逐步均值的相对差，25 步滑动平均（两端各留半窗不画）；灰带是 GPU 16 份每份对其余 15 份均值的相对差（同样平滑）的逐步最大绝对值，只作背景。"
               "「尾巴还在」按对齐记录的判据：第 751–1000 步均值超过 %s。p 是对修之前那一格的置换检验。" % ("%.0e" % N.TAIL_TH).replace("e-0", "e-"),
               "📌 两列都开着「多路残差分支输入和更新后的流强制舍 BF16」那个开关（修之前就开着）。修之前四份里三份只改执行方式（片上内存上限、调度），同一配置。「补流梯度舍法」再让流的几份梯度先各舍 BF16、照 GPU 的顺序逐个相加（分支输出也舍 BF16）。"
               "只补流梯度时，第 51–100 步的小鼓包还在（%s，修之前 %s）。" % (N.FC_BUMP_T["补流梯度舍法"], N.FC_BUMP_T["对照"]),
               "⚠️ 都是 mini 尺寸；延长到 2000 步时，修之前的尾巴在第 1000–1250 步到顶后回落到 GPU 散布里。")
    f.save("fig8-fixcmp.svg", yb + 14)


def fig_twoway():
    """fig8-twoway 回答：凭什么说主因就是流梯度的舍法？—— 两头各改一处：TPU 改成 GPU 的写法，偏高消失；GPU 改成 TPU 的写法，GPU 自己偏高。
    数据：topic08_numbers 的 TW（对齐记录三份单因素交叉结果文件；实测）。"""
    TW = N.TW
    f = Fig(W, "两块，纵轴相同：每个点是一份 1000 步运行第 551 到 1000 步主 loss 的带符号均值，横线是这一格的均值。"
               "左块 TPU 对 GPU 16 份均值：修之前四份都在 ＋6e-6 以上；补上 GPU 的流梯度舍法，五份降到均值 ＋1.9e-6。"
               "右块 GPU 对其余 15 份：GPU 自己 16 份散在 0 上下；GPU 把流梯度换成 TPU 的舍法，四份全部为正、均值 ＋3.9e-6；只换收拢处整段写法，四份散在 0 上下。")
    y0 = f.header("两头各改一处：TPU 换成 GPU 的舍法，偏高降下来；GPU 换成 TPU 的舍法，自己偏高",
                  "第二次迁移，小尺寸，1000 步，第 551–1000 步主 loss 带符号均值；每点一份运行，横线是均值",
                  [(RD, "TPU 原写法"), (GR, "TPU 改成 GPU 写法"), (GY2, "GPU 原写法"), (OR, "GPU 改成 TPU 写法")])
    PH, GAP = 470, 22
    PWL = (W - GAP) // 2
    vmin, vmax = -1.0e-5, 1.6e-5
    left = [("对照", "修之前", RD), ("补流梯度舍法", "补流梯度舍法", GR)]
    right = [("GPU 自己", "GPU 自己", GY2), ("GPU 换流梯度舍法", "换流梯度舍法", OR), ("GPU 换收拢处舍法", "换收拢处整段写法", OR)]
    for pi, (title, colr, cells, side) in enumerate((("Ⓐ TPU 一边：对 GPU 16 份均值", RD, left, "tpu"),
                                                    ("Ⓑ GPU 一边：每份对其余 15 份均值", OR, right, "gpu"))):
        x0 = pi * (PWL + GAP)
        py = f.panel(x0, y0, PWL, PH, title, colr)
        YT, YB = py + 30, py + PH - 120
        X0, X1 = x0 + 70, x0 + PWL - 20

        def Y(v):
            return YT + (vmax - v) / (vmax - vmin) * (YB - YT)
        for v in (-1e-5, -5e-6, 5e-6, 1e-5, 1.5e-5):
            f.line(X0, Y(v), X1, Y(v), LINE, 0.6, arrow=False)
            f.t(X0 - 6, Y(v) + 4, ("%+.1e" % v).replace("e-0", "e-").replace(".0e", "e"), GY, size=11.5, anchor="end")
        f.line(X0, Y(0), X1, Y(0), GY, 1.2, arrow=False)
        f.t(X0 - 6, Y(0) + 4, "0", GY, size=11.5, anchor="end")
        f.box(X0, Y(N.FC_SD), X1 - X0, Y(-N.FC_SD) - Y(N.FC_SD), "#eceff1", "none", 0)
        cw = (X1 - X0) / float(len(cells))
        for k, (key, lab, col) in enumerate(cells):   # 数据键沿用导出文件里的名字
            c = TW[side][key]
            cx = X0 + cw * (k + 0.5)
            vals = c["w551"]
            n = len(vals)
            for m, v in enumerate(vals):
                dx = (m - (n - 1) / 2.0) * (min(10, 120.0 / n))
                f.box(cx + dx - 5, Y(v) - 5, 10, 10, col, "#fff", 5, sw=1)
            f.line(cx - 46, Y(c["mean"]), cx + 46, Y(c["mean"]), INK, 2.4, arrow=False)
            f.t(cx, YB + 24, lab, col if col != GY2 else GY, True, 13.5, "middle")
            f.t(cx, YB + 46, "%d 份，均值 %s" % (n, ("%+.1e" % c["mean"]).replace("e-0", "e-") if abs(c["mean"]) > 1e-9 else "0"), INK, False, 12.5, "middle")
        if side == "tpu":
            a, b = X0 + cw * 0.5, X0 + cw * 1.5
            f.t((a + b) / 2, YT + 8, "差 %s，p %s" % (("%+.1e" % (TW["tpu"]["补流梯度舍法"]["mean"] - TW["tpu"]["对照"]["mean"])).replace("e-0", "e-"), N.TW_P_T["tpu"]), GR, True, 13.5, "middle")
        else:
            f.t(X0 + cw * 1.5, YT + 8, "4 份全为正，p %s" % N.TW_P_T["gpu"], OR, True, 13.5, "middle")
            f.t(X0 + cw * 2.5, YT + 8, "p %s，没变" % N.TW_P_T["head"], GY, True, 13, "middle")
        f.t(x0 + 16, py + PH - 40, "灰带：GPU 留一法标准差 ± %s" % ("%.1e" % N.FC_SD).replace("e-0", "e-"), GY, size=12)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "两头对上，才说得上主因是它：流的几份梯度怎么相加、在哪儿舍", [
        "只在 TPU 一边补，补好了也可能是两处差别凑巧抵消（上一轮就是这样撤回的）；GPU 一边反着改、长出同样方向的偏，两个方向对上，才说得上是它。",
        "GPU 每份梯度先舍 BF16 再逐个相加，每加一次舍一次；TPU 推断是在 FP32 里加完才舍一次。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，TPU v5p 8 设备 ↔ B200×8，1000 步）：每格 4–5 份（GPU 自己那格 16 份），第 551–1000 步主 loss 带符号均值；p 是对同一块里第一格做的置换检验（左块对修之前，右块对 GPU 自己）。",
               "📌 GPU 那边的写法在 CPU 上用 GPU 同一段代码逐位核过；GPU 的三个诊断开关都不开时，CPU 上 191 个梯度逐位不变。GPU 只换收拢处整段写法，后段不变，可第 51–100 步冒出 %s（p %s）—— 正是 TPU 不补收拢处舍入时的前段鼓包，又一处两头对上。" % (N.TW_HEAD51_T, N.TW_HEAD51_P_T),
               "⚠️ 都是 mini 尺寸、每格份数不多；这里的结论是「主因」。")
    f.save("fig8-twoway.svg", yb + 14)


def fig_gatecmp():
    """fig8-gatecmp 回答：对齐配置为什么没把后段修掉？—— 收拢处那次舍入一开，再补流梯度舍法（全套或任一半）就测不出作用；
    判档主 loss 只是刚好过线。数据：topic08_numbers 的 MG、R6（对齐记录单因素交叉与 6 份重复结果文件；实测）。"""
    MG, R6 = N.MG, N.R6
    f = Fig(W, "两块。左：每个点是一份 1000 步运行第 551 到 1000 步主 loss 的偏差，横线是均值。只开收拢处的舍入四份均值 ＋1.3e-5；"
               "在它上面再补流梯度舍法，全套四份 ＋6.7e-6、只补梯度一半 ＋7.9e-6、只补分支输出一半 ＋1.1e-5，对只开收拢处的 p 是 0.14、0.14、0.54，都测不出作用。"
               "右：主 loss 的判档倍数，灰点是 GPU 自己 15 份轮流当被测，最高 0.97；紫点是对齐配置 6 份，3 份高过 GPU 的全部 15 份，最高 1.14。")
    y0 = f.header("收拢处一开，流梯度那处就测不出作用；对齐配置第一份主 loss 0.98 刚好过线",
                  "第二次迁移，小尺寸，v5p，1000 步；每点一份运行；机制还在查",
                  [(OR, "只开收拢处舍入"), (PU, "再补流梯度舍法"), (GY2, "GPU 自己")])
    PH, GAP = 440, 22
    LW = 820
    py = f.panel(0, y0, LW, PH, "Ⓐ 收拢处开着时，再补流梯度舍法（第 551–1000 步主 loss 偏差）", OR)
    YT, YB = py + 34, py + PH - 110
    X0, X1 = 80, LW - 20
    vmin, vmax = -5e-6, 2.0e-5

    def Y(v):
        return YT + (vmax - v) / (vmax - vmin) * (YB - YT)
    for v in (-5e-6, 5e-6, 1e-5, 1.5e-5, 2e-5):
        f.line(X0, Y(v), X1, Y(v), LINE, 0.6, arrow=False)
        f.t(X0 - 6, Y(v) + 4, ("%+.1e" % v).replace("e-0", "e-").replace(".0e", "e"), GY, size=11.5, anchor="end")
    f.line(X0, Y(0), X1, Y(0), GY, 1.2, arrow=False)
    f.t(X0 - 6, Y(0) + 4, "0", GY, size=11.5, anchor="end")
    f.box(X0, Y(N.FC_SD), X1 - X0, Y(-N.FC_SD) - Y(N.FC_SD), "#eceff1", "none", 0)
    cells = [("M52", "只开收拢处", OR), ("C", "＋全套", PU), ("C_stream", "＋只补梯度一半", PU), ("C_branch", "＋只补分支输出一半", PU)]
    cw = (X1 - X0) / 4.0
    for k, (key, lab, col) in enumerate(cells):   # 数据键沿用导出文件里的名字
        c = MG[key]
        cx = X0 + cw * (k + 0.5)
        for m, v in enumerate(c["w551"]):
            f.box(cx + (m - 1.5) * 12 - 5, Y(v) - 5, 10, 10, col, "#fff", 5, sw=1)
        f.line(cx - 46, Y(c["mean"]), cx + 46, Y(c["mean"]), INK, 2.4, arrow=False)
        f.t(cx, YB + 24, lab, col, True, 13.5, "middle")
        f.t(cx, YB + 46, "4 份，均值 %s" % N.MG_MEAN_T[key], INK, False, 12.5, "middle")
        if key != "M52":
            f.t(cx, YT - 8, "p %s" % N.MG_P_T[key], GY, True, 13, "middle")
    f.t(16, py + PH - 30, "p 是对「只开收拢处」那一格的置换检验；灰带：GPU 留一法标准差 ± %s" % ("%.1e" % N.FC_SD).replace("e-0", "e-"), GY, size=12)
    f._pan = None
    RX = LW + GAP
    py2 = f.panel(RX, y0, W - RX, PH, "Ⓑ 主 loss 判档倍数：对齐配置 6 份 vs GPU 自己 15 份", PU)
    BX0, BX1 = RX + 40, W - 30
    lo, hi = 0.7, 1.25

    def BX(r):
        return BX0 + (r - lo) / (hi - lo) * (BX1 - BX0)
    yg, yc = py2 + 120, py2 + 230
    f.box(BX(1.0), py2 + 60, BX(hi) - BX(1.0), 230, "#fef7e0", "none", 0)
    f.line(BX(1.0), py2 + 56, BX(1.0), py2 + 300, GY, 1.2, dash="4 3", arrow=False)
    f.line(BX(max(R6["gpu_loo"])), py2 + 56, BX(max(R6["gpu_loo"])), py2 + 300, GY2, 1.2, dash="2 3", arrow=False)
    f.t(BX(1.0), py2 + 318, "1（过线）", GY, size=12, anchor="middle")
    f.t(BX(max(R6["gpu_loo"])) - 4, py2 + 50, "GPU 最高 %s" % N.R6_GPU_MAX_T, GY, size=12, anchor="end")
    for r in (0.8, 0.9, 1.1, 1.2):
        f.t(BX(r), py2 + 318, "%.1f" % r, GY, size=11.5, anchor="middle")
    f.t(RX + 16, yg - 22, "GPU 自己（15 份）", GY, True, 13)
    for m, r in enumerate(sorted(R6["gpu_loo"])):
        f.box(BX(r) - 5, yg - 5 + (m % 3) * 8, 10, 10, GY2, "#fff", 5, sw=1)
    f.t(RX + 16, yc - 22, "对齐配置（6 份）", PU, True, 13)
    for m, r in enumerate(sorted(R6["c"])):
        f.box(BX(r) - 6, yc - 6 + (m % 2) * 10, 12, 12, PU, "#fff", 6, sw=1)
    f.t(RX + 16, py2 + PH - 56, "6 份里 %d 份高过 GPU 的全部 15 份：" % N.R6_ABOVE, INK, True, 13)
    f.t(RX + 16, py2 + PH - 32, "过线是过了，可后段那截偏还在。", INK, False, 13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "warn", "收拢处那次舍入开着时，再补流梯度舍法测不出作用", [
        "唯一明显变好的是只补流梯度、不开收拢处；三处都照 GPU 写法的对齐配置，后段跟不修分不开。为什么，下一张图。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，TPU v5p 8 设备，1000 步）：左块每格 4 份，第 551–1000 步主 loss 对 GPU 16 份逐步均值的带符号均值；三格对修之前（只开前向那个强制舍入）也都分不开。"
               "「只补梯度一半」「只补分支输出一半」是把流梯度舍法拆成两半各自单开。",
               "📌 右块：现行检查项目录、GPU 16 份两两 120 对的尺子；GPU 那 15 份是每份对基线那份的同样判档（留一法）。对齐配置 6 份没有不过，现行那份 0.98。",
               "⚠️ 都是 mini 尺寸、每格份数不多；「测不出作用」是这几份运行下的结论，不等于证明没有作用。")
    f.save("fig8-gatecmp.svg", yb + 14)


fig_mig2()
fig_minidrift()
fig_ihcroot()
fig_fixcmp()
fig_twoway()
fig_gatecmp()
