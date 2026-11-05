# -*- coding: utf-8 -*-
r"""专题九 · 第五节「浮点与硬件」的四张图。

fig9-formats   回答：BF16、TF32、FP32 差在哪？两个 BF16 相乘为什么不丢精度、舍入发生在哪？
               数字：位宽来自 IEEE 754 / 各家格式定义；「乘积在 FP32 里精确」是本课推导（8 位 × 8 位有效数字 ＝ 至多 16 位 ≤ 24 位）。
fig9-truncate  回答：GPU 张量核在长求和里为什么系统地偏小？—— 每 16 个乘积对齐到最大指数，低位直接截掉（向零），像收银台每单都抹零。
               数字：实测的 B200 张量核模型（16 个一组、保留 25 位、向零截断）；合成正数输入的偏差随求和长度 K 变化（实测）。
               ⛔ 刻意没画：K ＝ 120,832 时合成输入的偏差反而降到 6.4e-5（推断是 cuBLAS 换了算法），写在出处里。
fig9-tf32      回答：「我关了 TF32」为什么不算数？—— 开关读回来的值和实际跑的不一样。
fig9-windowed  回答：TPU 上 FSDP 的矩阵乘为什么多舍了几次？—— 窗口化的 collective matmul 把每一轮的分片部分积在 BF16 里累加。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU
import topic09_numbers as N

W = 1400
FMT = {n: (e, m) for n, e, m in N.FORMATS}
assert FMT["FP32"] == (8, 23) and FMT["BF16"] == (8, 7) and FMT["TF32"] == (8, 10)
SIG_BF16 = FMT["BF16"][1] + 1                       # 有效数字位数（含隐含的 1）
SIG_FP32 = FMT["FP32"][1] + 1
assert 2 * SIG_BF16 <= SIG_FP32                     # 16 ≤ 24：BF16 × BF16 的乘积在 FP32 里放得下


def bits(f, x, y, n, col, fill, bw=30, bh=34, label=None, dash=None):
    for i in range(n):
        f.box(x + i * bw, y, bw - 3, bh, fill, col, 3, dash=dash)
    if label:
        f.t(x + n * bw / 2.0, y + bh + 20, label, col, True, 13, "middle")
    return x + n * bw


def fig_formats():
    f = Fig(W, "上半：三种格式的位。FP32 是 1 位符号、8 位指数、23 位尾数；TF32 也是 8 位指数，尾数只用 10 位；BF16 是 8 位指数、7 位尾数，一共 16 位。"
               "三者指数位一样多，所以能表示的范围一样大，差的是尾数，也就是精度。"
               "下半：两个 BF16 相乘，有效数字 8 位乘 8 位，最多 16 位，放进 FP32 的 24 位里还空 8 位，乘积是精确的；"
               "舍入发生在加法：一个大数加一个小数，要先把小数往右对齐，挤出 24 位窗口的那几位就丢了。")
    y0 = f.header("指数一样宽、尾数不一样长　——　<tspan font-weight=\"700\">乘是精确的，加才舍入</tspan>",
                  "一格 ＝ 一位；红 ＝ 符号，蓝 ＝ 指数（决定范围），绿 ＝ 尾数（决定精度）",
                  [(RD, "符号"), (BL, "指数"), (GR, "尾数"), (GY2, "这一位不参与运算")])
    PH = 250
    py = f.panel(0, y0, W, PH, "三种格式", BL)
    X0 = 130
    rows = [("FP32", 23, 0, "1 附近一格 2⁻²³ ≈ 1.2e-7"), ("TF32", 10, 13, "1 附近一格 2⁻¹⁰ ≈ 9.8e-4"), ("BF16", 7, 0, "1 附近一格 2⁻⁷ ≈ 7.8e-3")]
    for i, (name, mant, pad, note) in enumerate(rows):
        y = py + 26 + i * 70
        f.t(24, y + 24, name, INK, True, 18, big=True)
        x = bits(f, X0, y, 1, RD, "#fce8e6")
        x = bits(f, x, y, 8, BL, "#e8f0fe")
        x = bits(f, x, y, mant, GR, "#e6f4ea")
        if pad:
            bits(f, x, y, pad, GY2, "#fff", dash="3 3")
        f.t(X0 + 32 * 30 + 16, y + 24, note, GY, size=13)
    f._pan = None
    py2 = f.panel(0, y0 + PH + 16, W, 260, "BF16 × BF16 → FP32：乘积放得下；加法要对齐，挤出去的位就丢了", GR)
    # 乘法
    y = py2 + 30
    f.t(24, y + 22, "乘", INK, True, 18, big=True)
    xa = bits(f, 80, y, SIG_BF16, GR, "#e6f4ea", bw=22, bh=26, label="8 位有效数字")
    f.t(xa + 14, y + 20, "×", INK, True, 18, big=True)
    xb = bits(f, xa + 40, y, SIG_BF16, GR, "#e6f4ea", bw=22, bh=26, label="8 位")
    f.t(xb + 14, y + 20, "＝", INK, True, 18, big=True)
    xc = bits(f, xb + 40, y, 2 * SIG_BF16, GR, GR, bw=22, bh=26)
    bits(f, xc, y, SIG_FP32 - 2 * SIG_BF16, GY2, "#fff", bw=22, bh=26, dash="3 3")
    f.t(xb + 40 + SIG_FP32 * 11, y + 46, "16 位放进 FP32 的 24 位，还空 8 位：精确", GR, True, 13, "middle")
    # 加法
    y = py2 + 140
    f.t(24, y + 22, "加", INK, True, 18, big=True)
    bits(f, 80, y - 20, SIG_FP32, BL, "#e8f0fe", bw=22, bh=24)
    f.t(80 + SIG_FP32 * 22 + 14, y - 2, "大数（24 位）", BL, True, 13)
    SH = 10
    bits(f, 80 + SH * 22, y + 14, SIG_FP32 - SH, OR, "#fef7e0", bw=22, bh=24)
    bits(f, 80 + SIG_FP32 * 22, y + 14, SH, RD, "#fce8e6", bw=22, bh=24, dash="3 3")
    f.line(80 + SIG_FP32 * 22 - 2, y - 30, 80 + SIG_FP32 * 22 - 2, y + 48, RD, 2, dash="5 4", arrow=False)
    f.t(80 + (SIG_FP32 + SH) * 22 + 14, y + 32, "小数往右对齐：挤出 24 位窗口的 %d 位丢掉" % SH, RD, True, 13)
    f._pan = None
    yb = f.band(py2 + 280, "ok", "两个硬件都是「BF16 乘、FP32 累加」，差别全在累加那一步怎么舍", [
        "乘积本身精确，所以讲精度要盯着「很多乘积加起来」：加得越长，舍入的机会越多。",
    ])
    yb = f.src(yb + 10,
               "📌 位宽：FP32 1/8/23（IEEE 754 binary32）；BF16 1/8/7；TF32 1/8/10（NVIDIA 张量核用的 19 位格式，按 32 位存放）。",
               "⚠️ 本课推导：有效数字含隐含的那一位，BF16 是 8 位、FP32 是 24 位，8 × 8 位的乘积至多 16 位 ≤ 24 位，所以只要不溢出就精确。")
    f.save("fig9-formats.svg", yb + 14)


def fig_truncate():
    f = Fig(W, "左边：GPU 张量核累加的模型。每 16 个乘积连同累加器一起，按其中最大的指数对齐；对齐后超出 25 位窗口的低位直接截掉，向零，不是四舍五入。"
               "就像收银台每一单都抹零，只少不多。右边：用全是正数的合成输入，求和长度 K 从 256 到 65,536，"
               "朝零的相对偏差从 2.2e-5 涨到 3.1e-4；不走张量核的 FP32 不到 5e-8。")
    y0 = f.header("GPU 张量核：每 16 个一组，低位抹掉　——　<tspan font-weight=\"700\">只少不多，加得越长少得越多</tspan>",
                  "左：一行 ＝ 一个乘积（对齐后）；虚线右边 ＝ 窗口外的低位，被截掉。右：横轴 ＝ 一次求和加多少个乘积（对数），纵轴 ＝ 结果朝零偏了多少（对数）",
                  [(BL, "留下的位"), (RD, "截掉的位（向零）"), (OR, "实测：BF16 张量核"), (GR, "实测：FP32 不走张量核")])
    PH = 400
    py = f.panel(0, y0, 640, PH, "一组 16 个乘积 ＋ 累加器，按最大指数对齐（实测的 B200 模型）", BL)
    X0, BW, BH = 40, 13, 14
    WIN = 25
    shifts = [0, 3, 1, 6, 2, 9, 4, 12, 5, 2, 7, 15, 3, 8, 1, 11, 4]   # 示意：每个乘积相对最大指数右移几位
    assert len(shifts) == N.TC_GROUP + 1
    for i, s in enumerate(shifts):
        y = py + 24 + i * 20
        lab = "累加器" if i == 0 else "乘积 %d" % i
        f.t(X0 + 52, y + 12, lab, GY, size=11, anchor="end")
        L = 24
        for b in range(L):
            pos = s + b
            if pos < WIN:
                f.box(X0 + 60 + pos * BW, y, BW - 2, BH, "#e8f0fe", BL, 2)
            else:
                f.box(X0 + 60 + pos * BW, y, BW - 2, BH, "#fce8e6", RD, 2)
    xw = X0 + 60 + WIN * BW - 1
    f.line(xw, py + 14, xw, py + 24 + 17 * 20, RD, 2, dash="6 4", arrow=False)
    f.t(xw - 6, py + 24 + 17 * 20 + 4, "25 位窗口", RD, True, 13, "end")
    f._pan = None
    px = 670
    py2 = f.panel(px, y0, W - px, PH, "偏差随求和长度长大（实测，合成正数输入）", OR)
    GX0, GY0, GW, GH = px + 90, py2 + 320, 580, 280
    KMIN, KMAX, EMIN, EMAX = 2 ** 7, 2 ** 18, 1e-8, 1e-3

    def GXf(k):
        return GX0 + (math.log10(k) - math.log10(KMIN)) / (math.log10(KMAX) - math.log10(KMIN)) * GW

    def GYf(e):
        return GY0 - (math.log10(e) - math.log10(EMIN)) / (math.log10(EMAX) - math.log10(EMIN)) * GH
    f.line(GX0, GY0, GX0 + GW, GY0, GY2, 1.2)
    f.line(GX0, GY0, GX0, GY0 - GH, GY2, 1.2)
    for e in (1e-8, 1e-6, 1e-4):
        f.t(GX0 - 8, GYf(e) + 4, "%.0e" % e, GY, size=12, anchor="end")
    for k in (256, 4096, 65536):
        f.t(GXf(k), GY0 + 20, "K ＝ %s" % format(k, ","), GY, size=12, anchor="middle")
    pts = [(GXf(k), GYf(e)) for k, e in N.TC_BIAS_K]
    f.path(pts, OR, 2.5, arrow=False)
    for (k, e), (x, y) in zip(N.TC_BIAS_K, pts):
        f.box(x - 7, y - 7, 14, 14, OR, OR, 7)
        f.t(x - 10, y - 12, "%.1e" % e, OR, True, 13, "end")
    f.line(GX0, GYf(N.FP32_BIAS_MAX), GX0 + GW, GYf(N.FP32_BIAS_MAX), GR, 2.5, dash="7 5", arrow=False)
    f.t(GX0 + GW, GYf(N.FP32_BIAS_MAX) - 10, "FP32 不走张量核：≤ 5e-8", GR, True, 13, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "四舍五入有多有少，长了能抵消；抹零每次都少，长了就积成一个朝零的系统偏差", [
        "正数、负数输入都一样朝零偏，随 K 增大：加到六万多个，就偏到万分之三。",
        "代价：短的求和看不出来；单算子对 GPU 自己比也看不出来 —— 只有拿精确值当参照才量得到。",
    ])
    yb = f.src(yb + 10,
               "📌 实测的 B200 张量核模型（一个经验模型，不是厂商规格）：每 16 个乘积连同累加器对齐到最大指数，低于 2^(emax−25) 的位向零截掉，写回 FP32 也向零舍入。「25 位」是拟合出来的窗口宽度，跟 FP32 的 24 位有效数字不是一回事。它在路由打分（K ＝ 4096）上逐位复现了三层全部 4096 × 192 个元素；把参数改成 8 个一组、24 位或就近舍入，命中率都掉到 1% 以下。",
               "📌 实测：合成负数输入 −9.5e-6 / −6.9e-5 / −3.4e-4，同样朝零；TF32 小约一个量级；FP32 不走张量核 ≤ 5e-8。K ＝ 120,832 时合成输入的偏差反而降到 6.4e-5（推断：cuBLAS 对这个形状换了算法），所以图上没有画这一点。",
               "注：左图每个乘积右移几位是示意，不是实测数据。")
    f.save("fig9-truncate.svg", yb + 14)


def fig_tf32():
    f = Fig(W, "FP32 矩阵乘对 FP64 的相对误差，对数刻度。训练容器默认开着 TF32。只把 torch 的两个 TF32 开关关掉，读回来是 False，"
               "但 Transformer Engine 的 FP32 矩阵乘照样走 TF32，误差 2.94e-4 不变；设环境变量 NVIDIA_TF32_OVERRIDE=0 后两条路都是真 FP32，1.15e-6，"
               "而 torch 的开关读回来反而是 True。判据：每次启动实测一遍，1e-5 以下才算真关了。")
    y0 = f.header("「我关了 TF32」不算数　——　<tspan font-weight=\"700\">开关读回来的值，和实际跑的不一样</tspan>",
                  "柱长 ＝ FP32 矩阵乘对 FP64 的误差（对数刻度，越短越准）；左边小字 ＝ 那一刻 torch 的开关读回来是什么",
                  [(RD, "实际走 TF32"), (GR, "实际是真 FP32"), (GY2, "判据线（虚线）")])
    PH = 330
    py = f.panel(0, y0, W, PH, "GPU 训练容器里的 FP32 矩阵乘（实测）", BL)
    bx, bw = 420, 800
    emin, emax = 1e-7, 1e-3
    rows = [("只关 torch 的两个开关", "读回 allow_tf32 ＝ False；Transformer Engine 那条路", N.TF32_TE_ON, RD),
            ("设 NVIDIA_TF32_OVERRIDE=0", "读回 allow_tf32 ＝ True（！）；两条路都关了", N.TF32_ALL_OFF, GR)]

    def LX(v):
        return bx + (math.log10(v) - math.log10(emin)) / (math.log10(emax) - math.log10(emin)) * bw
    for i, (lab, sub, v, c) in enumerate(rows):
        y = py + 44 + i * 110
        f.t(24, y + 20, lab, INK, True, 15)
        f.t(24, y + 44, sub, GY, size=13)
        f.box(bx, y + 4, LX(v) - bx, 32, c, c, 4)
        f.t(LX(v) + 10, y + 27, "%.2e" % v, c, True, 15)
    xc = LX(N.TF32_PROBE_OK)
    f.line(xc, py + 20, xc, py + 270, GY2, 2, dash="6 5", arrow=False)
    f.t(xc + 8, py + 280, "判据：启动时实测 ≤ 1e-5 才算真关", GY, True, 14)
    f._pan = None
    yb = f.band(y0 + PH + 20, "bad", "开关的读回值不能当证据；拿一次实测当证据", [
        "只有环境变量能连 Transformer Engine 一起关掉，关掉以后 torch 的开关读回来还是 True。",
        "代价：用没关干净的「FP32」当精确参照，量出来的偏差会比被测还大，结论就反了。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：官方训练容器设了 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1，allow_tf32 两个开关默认 True。只关 torch 开关，torch 自己的 FP32 矩阵乘变成 IEEE FP32，Transformer Engine 的 FP32 GEMM 仍是 2.94e-4；设 NVIDIA_TF32_OVERRIDE=0 后两者都是 1.15e-6。",
               "📌 TPU 没有 TF32；FP32 操作数按编译时的矩阵乘精度档处理，默认档是先舍成 BF16 单遍乘、FP32 累加。")
    f.save("fig9-tf32.svg", yb + 14)


def fig_windowed():
    f = Fig(W, "上一行：TPU 上 FSDP 把权重切开分到各个设备上（示意画 4 片），编译器把这个矩阵乘做成窗口化的 collective matmul：一边轮流收下一片权重，一边算当前这片的部分积。"
               "默认每一轮的部分积以 BF16 输出、在 BF16 累加器里相加，4 轮舍 4 次；打开 FP32 累加的开关后，累加器是 FP32，最后只舍一次，跟 GPU 一样。"
               "右边：qkv 投影输出对 GPU 的相对差从 2.36e-3 降到 1.8e-4。")
    y0 = f.header("TPU 上 FSDP 的矩阵乘：分片部分积在 BF16 里累加　——　<tspan font-weight=\"700\">比 GPU 多舍了几次</tspan>",
                  "一格 ＝ 一轮：收下一片权重的同时，算当前这片的部分积（这种「边收边算」就叫 collective matmul）；红色小块 ＝ 一次舍到 BF16",
                  [(BL, "部分积"), (RD, "舍到 BF16"), (GR, "FP32 累加器，只在最后舍一次")])
    PH = 300
    py = f.panel(0, y0, 920, PH, "示意：权重切成 4 片、4 轮（实际 FSDP 是 8 路）", BL)
    for r, (lab, fp32) in enumerate([("默认：BF16 累加器", False), ("打开 FP32 累加", True)]):
        y = py + 40 + r * 120
        f.t(24, y + 26, lab, GR if fp32 else RD, True, 15)
        x = 210
        for k in range(4):
            f.box(x, y, 110, 46, "#e8f0fe", BL, 6)
            f.t(x + 55, y + 28, "部分积 %d" % (k + 1), BL, True, 14, "middle")
            x += 110
            f.t(x + 14, y + 30, "＋", INK, True, 18, "middle", big=True)
            if not fp32:
                f.box(x + 28, y + 6, 14, 34, "#fce8e6", RD, 3)
                f.t(x + 35, y + 64, "舍", RD, True, 12, "middle")
            x += 56
        f.box(x - 10, y, 70, 46, GR if fp32 else "#fce8e6", GR if fp32 else RD, 6)
        f.t(x + 25, y + 28, "结果", "#fff" if fp32 else RD, True, 14, "middle")
        if fp32:
            f.t(x + 25, y + 66, "只舍 1 次", GR, True, 12, "middle")
        else:
            f.t(x + 25, y + 66, "舍了 4 次", RD, True, 12, "middle")
    f._pan = None
    px = 940
    py2 = f.panel(px, y0, W - px, PH, "qkv 投影输出对 GPU（实测）", GR)
    for i, (lab, v, c) in enumerate([("默认", N.QKV_VS_GPU_BEFORE, RD), ("打开 FP32 累加", N.QKV_VS_GPU_AFTER, GR)]):
        y = py2 + 44 + i * 96
        f.t(px + 20, y, lab, INK, True, 14)
        L = v / N.QKV_VS_GPU_BEFORE * 380
        f.box(px + 20, y + 12, L, 30, c, c, 4)
        f.t(px + 20 + L + 8 if L < 300 else px + 30, y + 34, "rel_L2 %.2e" % v, c if L < 300 else "#fff", True, 14)
    f.t(px + 20, py2 + 240, "FP64 的 R：%.3f → %.3f" % (N.R_QKV_BEFORE, N.R_QKV_AFTER), INK, True, 15)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "这一处是 TPU 自己多舍的：打开开关，前向就跟 GPU 一样只舍一次", [
        "收缩维被 FSDP 切开的投影（注意力的 q／k／v、稠密 MLP 的输入投影）都走这条路，一份 v5p 配置里有 41 个这样的循环。",
        "代价：它只修了前向的那几次舍入；对后面案例里那条参数范数漂移几乎没有作用（关掉窗口化跑 1000 步只少 3%）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：v5p 配置优化后的 HLO 里有 41 个窗口化的点积循环，q 投影的循环累加器是 BF16。打开 FP32 累加（矩阵乘的输出类型设成 FP32，最后舍一次）后，qkv 输出对 GPU 的 rel_L2 2.36e-3 → 1.8e-4。",
               "📌 另一条路是用编译参数整个关掉窗口化：那样连 GPU 本来就有的一次舍入也省掉了，MLP 前归一化的 R 到了 1.64，反而对不齐 —— 多舍少舍都不行，所以用开关不用参数。",
               "注：左图画的是 4 片 4 轮的示意，实际片数等于 FSDP 的路数。")
    f.save("fig9-windowed.svg", yb + 14)


fig_formats()
fig_truncate()
fig_tf32()
fig_windowed()
