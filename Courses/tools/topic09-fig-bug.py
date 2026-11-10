# -*- coding: utf-8 -*-
r"""专题九 · 第十节「日常回归」补的图（第二次迁移的新材料）。

fig9-bug  回答：改完代码怎么最快知道变没变？工具真能抓到 bug 吗？
          Ⓐ 改完先只编译、比训练步编译后程序的指纹：一样就逐位相同，不用跑；不一样再跑 300 步看档。
          Ⓑ 一个真 bug：数据并行 ＋ 梯度累积 ＋ 词嵌入梯度 FP32 累加三样同时成立时，编译器把这份梯度的跨卡求和挪出了累积循环，
             循环里又把 0 号以外各卡的累加器清零，结果每张卡（0 号除外）只留下最后一个微批。
          数据：对齐记录 TPU 自比一节的 MATRIX.md「D 等价改写」「已修」「B1b」（实测）；机制来自编译后程序里循环体的那条 select（实测）。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE
import topic09_numbers as N

W = 1400
SAME = ["给中间值起名字", "拆出一个函数", "求平均写成求和再除"]
DIFF_BIT = ["乘法换个顺序", "除以 n 换成乘 1/n", "梯度裁剪换写法", "多记一个监控量"]
DIFF = [("x·x 代替 square(x)", "1.09"), ("einsum 换广播乘", "1.10")]
BUG = {"norm": 0.51, "rows_zero": 2829, "rows": 10911, "far": 130, "r": 611, "fix_r": 0.98}
assert abs(BUG["rows_zero"] / BUG["rows"] - 0.26) < 0.005


def fig_bug():
    f = Fig(W, "两块。左块：改完代码先只编译，比训练步编译后程序的指纹。起名字、拆函数、求平均写成求和再除，"
               "编出来的程序一模一样，结果一定逐位相同，不用跑；程序变了再跑：乘法换顺序、除以 n 换成乘 1/n、换梯度裁剪写法、多记一个监控量，程序变了结果仍逐位；"
               "x 乘 x 代替平方、einsum 换广播乘，结果不逐位，r 在 1 附近。"
               "右块：八张卡、每张卡两个微批的梯度格子。正常时十六格都加进来；编译器把跨卡求和挪出循环、又在循环里把 0 号以外的卡清零以后，"
               "1 到 7 号卡的第一个微批全丢了：词嵌入梯度只剩 0.51 倍，四分之一的行整行为零，r 611。修好以后 r 0.98。")
    y0 = f.header("改完先比指纹；工具抓到的一个真 bug",
                  "左：程序一样就一定逐位，省掉一次训练；右：一个只在「数据并行 ＋ 梯度累积 ＋ 词嵌入梯度 FP32 累加」同时成立时出现的编译器 bug（第二次迁移，实测）",
                  [(GR, "逐位相同"), (BL, "程序变了：跑 300 步看档"), (RD, "被清零的梯度")])
    PH = 420
    LW = 640
    py = f.panel(0, y0, LW, PH, "Ⓐ 改完先只编译，比程序指纹", GR)
    f.box(24, py + 20, 160, 46, "#fff", INK, 8, sw=1.4)
    f.t(104, py + 49, "改了一处代码", INK, True, 14, "middle")
    f.line(184, py + 43, 230, py + 43, GY, 1.6)
    f.box(234, py + 20, 180, 46, "#fff", INK, 8, sw=1.4)
    f.t(324, py + 49, "只编译，不训练", INK, True, 14, "middle")
    f.line(414, py + 43, 456, py + 43, GY, 1.6)
    f.box(460, py + 20, 156, 46, "#fff", INK, 8, sw=1.4)
    f.t(538, py + 49, "比程序指纹", INK, True, 14, "middle")
    # 两条出路
    f.line(538, py + 66, 160, py + 104, GR, 1.6)
    f.line(538, py + 66, 470, py + 104, BL, 1.6)
    f.box(24, py + 108, 290, 250, "#e6f4ea", GR, 8)
    f.t(40, py + 134, "一样 → 一定逐位相同，不用跑", GR, True, 14)
    for i, s in enumerate(SAME):
        f.t(44, py + 168 + i * 30, "· " + s, INK, size=13.5)
    f.box(330, py + 108, 286, 250, "#e8f0fe", BL, 8)
    f.t(346, py + 134, "不一样 → 跑 300 步看档", BL, True, 14)
    f.t(350, py + 162, "结果仍逐位：", GR, True, 13)
    for i, s in enumerate(DIFF_BIT):
        f.t(350, py + 186 + i * 24, "· " + s, INK, size=13.5)
    f.t(350, py + 290, "不逐位：", BL, True, 13)
    for i, (s, r) in enumerate(DIFF):
        f.t(350, py + 314 + i * 24, "· " + s, INK, size=13.5)
        f.t(600, py + 314 + i * 24, "r " + r, BL, True, 13.5, "end")
    f._pan = None
    RX = LW + 24
    py2 = f.panel(RX, y0, W - RX, PH, "Ⓑ 编译器把跨卡求和挪出了累积循环", RD)
    gx, gy, cw, ch = RX + 110, py2 + 54, 50, 36
    f.t(RX + 24, gy + 18, "微批 1", INK, True, 13)
    f.t(RX + 24, gy + ch + 10 + 18, "微批 2", INK, True, 13)
    for d in range(8):
        f.t(gx + d * (cw + 6) + cw / 2, gy - 10, "卡%d" % d, GY, size=13, anchor="middle")
        for m in range(2):
            lost = (m == 0 and d > 0)
            x, y = gx + d * (cw + 6), gy + m * (ch + 10)
            if lost:
                f.box(x, y, cw, ch, "#fce8e6", RD, 4, sw=1.4, dash="4 3")
                f.line(x + 8, y + 8, x + cw - 8, y + ch - 8, RD, 1.6, arrow=False)
                f.line(x + cw - 8, y + 8, x + 8, y + ch - 8, RD, 1.6, arrow=False)
            else:
                f.box(x, y, cw, ch, "#e8f0fe", BL, 4)
    ax = gx + 8 * (cw + 6) + 4
    f.line(ax, gy + ch + 10 + ch / 2, ax + 26, gy + ch + 10 + ch / 2, RD, 1.6)
    f.t(ax + 30, gy + ch + 10 + ch / 2 - 4, "跨卡求和", RD, True, 12.5)
    f.t(ax + 30, gy + ch + 10 + ch / 2 + 14, "只在这里一次", RD, size=12.5)
    ty = gy + 2 * ch + 44
    f.t(RX + 24, ty, "循环里每一圈都对「卡号 ≠ 0」的卡把累加器清零，求和挪到了循环外面：", INK, size=13.5)
    f.t(RX + 24, ty + 24, "1–7 号卡只剩最后一个微批，0 号卡完整。", INK, size=13.5)
    rows = [("词嵌入梯度范数", "%.2f 倍" % BUG["norm"]), ("有梯度的行里整行为零", "%d／%d（%d%%）" % (BUG["rows_zero"], BUG["rows"], round(100.0 * BUG["rows_zero"] / BUG["rows"]))),
            ("离 FP64", "远 %d 倍" % BUG["far"]), ("300 步曲线 r", "%d" % BUG["r"])]
    for i, (k, v) in enumerate(rows):
        y = ty + 62 + i * 28
        f.t(RX + 40, y, k, GY, size=13.5)
        f.t(RX + 400, y, v, RD, True, 14)
    f.t(RX + 24, ty + 62 + 4 * 28 + 10, "修法：每个微批的梯度累加前加一道屏障，挡住这次挪动 → r %.2f；golden 重跑逐位不变" % BUG["fix_r"], GR, True, 13.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "便宜的先做：先比程序指纹，再跑 300 步；尺子和 FP64 一起看，连编译器的 bug 都能抓出来", [
        "那个 bug 的读数一点不含糊：梯度少一半、四分之一的行整行为零、r 611。少了任何一样条件（不累积、不开开关、改成 FSDP）都不丢梯度。",
        "代价：指纹只能证明「没变」；变了以后是好是坏，仍要跑曲线、对 FP64。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，TPU v5p 自比）：指纹 ＝ 训练步优化后程序去掉元数据和源码位置表后的哈希。起名字（重算名）、拆辅助函数、mean 换 sum/n，程序相同、结果逐位；"
               "乘法换序、除以 n 换乘 1/n、换梯度裁剪写法、多记一个逐头监控量、reduce-scatter 交给 SparseCore，程序不同、结果仍逐位（按各臂结果文件的 program_equal；对齐记录 README「其他发现」把前三项写成了程序相同，以结果文件为准）；"
               "x·x 代替 lax.square、einsum 换广播乘，程序不同、结果不逐位，300 步 r 1.09 ／ 1.10（x/sqrt 代替 x·rsqrt 同理，0.98）。",
               "📌 bug：纯数据并行 8、每设备微批 1 × 累积 2、词嵌入梯度 FP32 累加三样同时成立。编译后的循环体里有一条 select(卡号, 0, 累加器)，这份 FP32 梯度只有 1 次跨卡求和（BF16 版每个微批一次，共 2 次）。"
               "第 1 步词嵌入梯度范数是正确值的 0.51 倍，正确值里有梯度的 10911 行中 2829 行整行为零，离 FP64 远 130 倍；raw 梯度范数 6.19（正确 6.36）。"
               "修法：三者同时成立时每个微批的梯度累加前加 optimization_barrier；修后纯数据并行 ＋ FP32 累加词嵌入梯度离 FP64 3.79e-3、梯度范数 6.3596（FP64 6.3594）、300 步 r 0.98；golden 用修后代码重跑逐位相同、程序相同。",
               "⚠️ 卡数画 8、微批画 2 是这次的实际设置；「求和挪到循环外、循环里清零」是从编译后程序读出来的。")
    f.save("fig9-bug.svg", yb + 14)


fig_bug()
