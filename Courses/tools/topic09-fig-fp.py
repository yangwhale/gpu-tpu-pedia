# -*- coding: utf-8 -*-
r"""专题九 · 第五节「浮点与硬件」的四张图。

fig9-formats   回答：BF16、TF32、FP32 差在哪？两个 BF16 相乘为什么不丢精度、舍入发生在哪？
               数字：位宽来自 IEEE 754 / 各家格式定义；「乘积在 FP32 里精确」是本课推导（8 位 × 8 位有效数字 ＝ 至多 16 位 ≤ 24 位）。
fig9-truncate  回答：GPU 张量核在长求和里为什么系统地偏小？—— 每 16 个乘积对齐到最大指数，低位直接截掉（向零），像收银台每单都抹零。
               数字：实测的 B200 张量核模型（16 个一组、保留 25 位、向零截断）；合成正数输入的偏差随求和长度 K 变化（实测）。
               ⛔ 刻意没画：K ＝ 120,832 时合成输入的偏差反而降到 6.4e-5（推断是 cuBLAS 换了算法），写在出处里。
fig9-tf32      回答：「我关了 TF32」为什么不算数？—— 开关读回来的值和实际跑的不一样。
fig9-windowed  回答：TPU 上 FSDP 的矩阵乘为什么多舍了几次？—— 窗口化的 collective matmul 把每一轮的分片部分积在 BF16 里累加。
fig9-excess    回答：excess precision 改的是什么？—— 舍在哪儿：融合块里中间那几次「舍到 BF16」被删掉，只在块出口舍；关掉它，舍入回到每个算子出口。
               示意图：算子个数、融合怎么切都是示意，不带实测数；实测（关掉以后单层更远、1000 步 loss 偏移小一半）只进 📌。
               ⛔ 刻意没画「哪一行更准」：这张图只讲舍入位置。关掉不是更准，画个准确度刻度就讲反了。
fig9-boundary  回答：对不齐通常出在哪？—— 一个值在 16 位和 32 位之间进出的地方。每个算子两边各写一行五格 dtype 路径，
               五列各放一个真实例子，不一样的那一格画红框（2026-11-07 第二次迁移的教训，例子取自两次迁移）。
               ⛔ 刻意没画误差大小：这张图是一张检查表，量级在出处里；第 4 行的效果在第 1 步比 GPU 噪声小几十倍，画柱子会让人以为它不重要。
fig9-elide     回答：编译器「更精确」为什么也会对不齐？—— GPU 一定会做的 FP32→BF16→FP32 往返，TPU 上被省掉，1000 步 loss 次次偏高。
               数字全是实测（五次运行、GPU 噪声上限、强制舍入后）；「被编译器省掉」是推断（开关效果支持，没看编译产物）。
fig9-trans     回答：TPU 默认的 exp、log 有多粗？高精度的软件实现怎么算？—— 默认几十到上百格，GPU 个位数；区间缩减 ＋ 多项式 ＋ 牛顿修正。
               右格的算法是教科书做法（示意），不是哪一家库的源码。
fig9-captain   回答：一个小标量对不上，怎么分清是病态还是 bug？—— 船上称船长：船 7100 吨、船长 110 公斤，秤的一格 0.47 公斤 ＝ 船长的 0.43%。
               比方的数按 lse 1 单位 ↔ 1000 吨换算（本课推导），比例与实测一致：f32 一格 ＝ KL 的 4.3e-3。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU, wpx, MONO_K
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
               "下半：两个 BF16 相乘，有效数字 8 位乘 8 位（存下的 7 位尾数，前面再加一位不存的 1），最多 16 位，放进 FP32 的 24 位里还空 8 位，乘积是精确的；"
               "舍入发生在加法：一个大数加一个小数，要先把小数往右对齐，挤出 24 位窗口的那几位要舍掉 —— 普通 FP32 加法是就近舍入，可能进一位；张量核怎么舍见下一张图。虚线白格是不用的位，浅灰底深灰框是加法里的大数。")
    y0 = f.header("指数一样宽、尾数不一样长　——　<tspan font-weight=\"700\">乘是精确的，加才舍入</tspan>",
                  "一格 ＝ 一位；红 ＝ 符号，蓝 ＝ 指数（决定范围），绿 ＝ 尾数（决定精度）",
                  [(RD, "符号"), (BL, "指数"), (GR, "尾数")])
    # R4 麻瓜第 2 条：原图例「灰 ＝ 这一位不参与运算」跟加法行的灰色大数（它参与运算）打架，
    #   而 TF32 不用的位画的其实是虚线白格。→ 图例拆成两项，手画在同一行（共用 header 只会画实心色块）：
    #   虚线白格 ＝ 不用的位；浅灰底 ＋ 深灰框 ＝ 加法里的大数（只示位数，不分字段）。
    ly = y0 - 28
    lx = sum(21 + wpx(lab, 14) * MONO_K + 26 for lab in ("符号", "指数", "尾数"))
    for kind, lab in (("dash", "虚线白格 ＝ 不用的位（TF32 那 13 位、乘积空出的 8 位）"),
                      ("big", "浅灰底深灰框 ＝ 加法里的大数（只示位数）")):
        if kind == "dash":
            f.box(lx, ly - 11, 14, 14, "#fff", GY2, 3, dash="3 3")
        else:
            f.box(lx, ly - 11, 14, 14, LINE2, GY, 3, sw=1.5)
        f.t(lx + 21, ly, lab, GY, size=14)
        lx += 21 + wpx(lab, 14) * MONO_K + 26
    assert lx < W, lx
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
    py2 = f.panel(0, y0 + PH + 16, W, 260, "BF16 × BF16 → FP32：乘积放得下；加法要对齐，挤出去的位要舍掉", GR)
    # 乘法
    y = py2 + 30
    f.t(24, y + 22, "乘", INK, True, 18, big=True)
    xa = bits(f, 80, y, SIG_BF16, GR, "#e6f4ea", bw=22, bh=26, label="8 位有效数字")
    f.t(xa + 14, y + 20, "×", INK, True, 18, big=True)
    xb = bits(f, xa + 40, y, SIG_BF16, GR, "#e6f4ea", bw=22, bh=26, label="8 位")
    # R4 麻瓜第 1 条：上半 BF16 只画 7 个尾数格，这里是 8 格 —— 多的那格是不存的隐含位「1」
    for x1 in (80, xa + 40):
        f.box(x1, y, 19, 26, "#fff", GR, 3)
        f.t(x1 + 9.5, y + 18, "1", GR, True, 13, "middle")
    f.t(80, y + 72, "BF16 只存 7 位尾数，前面还有一位不存的 1（白格），所以有效数字是 8 位", GY, size=13)
    f.t(xb + 14, y + 20, "＝", INK, True, 18, big=True)
    xc = bits(f, xb + 40, y, 2 * SIG_BF16, GR, GR, bw=22, bh=26)
    bits(f, xc, y, SIG_FP32 - 2 * SIG_BF16, GY2, "#fff", bw=22, bh=26, dash="3 3")
    f.t(xb + 40 + SIG_FP32 * 11, y + 46, "16 位放进 FP32 的 24 位，还空 8 位：精确", GR, True, 13, "middle")
    # 加法
    y = py2 + 140
    f.t(24, y + 22, "加", INK, True, 18, big=True)
    for i in range(SIG_FP32):                                         # R3：中性灰，别借上半「指数」的蓝；R4：深灰框，跟虚线「不用的位」分开
        f.box(80 + i * 22, y - 20, 19, 24, LINE2, GY, 3, sw=1.5)
    f.t(80 + SIG_FP32 * 22 + 14, y - 2, "大数（24 位）", GY, True, 13)
    SH = 10
    bits(f, 80 + SH * 22, y + 14, SIG_FP32 - SH, OR, "#fef7e0", bw=22, bh=24)
    bits(f, 80 + SIG_FP32 * 22, y + 14, SH, RD, "#fce8e6", bw=22, bh=24, dash="3 3")
    f.line(80 + SIG_FP32 * 22 - 2, y - 30, 80 + SIG_FP32 * 22 - 2, y + 48, RD, 2, dash="5 4", arrow=False)
    # 原「挤出的 10 位丢掉」会被读成普通 FP32 加法也是截断 → 写明普通加法是就近舍入，张量核的截断留给下一张图
    xr = 80 + (SIG_FP32 + SH) * 22 + 14
    f.t(xr, y + 22, "小数往右对齐：挤出 24 位窗口的 %d 位要舍掉" % SH, RD, True, 13)
    f.t(xr, y + 42, "普通 FP32 加法：就近舍入（可能进一位）", GY, size=13)
    f.t(xr, y + 60, "张量核的做法见下一张图", GY, size=13)
    f._pan = None
    yb = f.band(py2 + 280, "ok", "两个硬件都是「BF16 乘、FP32 累加」，差别全在累加那一步怎么舍", [
        "乘积本身精确，所以讲精度要盯着「很多乘积加起来」：加得越长，舍入的机会越多。",
    ])
    yb = f.src(yb + 10,
               "📌 位宽：FP32 1/8/23（IEEE 754 binary32）；BF16 1/8/7；TF32 1/8/10（NVIDIA 张量核用的 19 位格式，按 32 位存放）。",
               "⚠️ 本课推导：有效数字含隐含的那一位，BF16 是 8 位、FP32 是 24 位，8 × 8 位的乘积至多 16 位 ≤ 24 位，所以只要不溢出就精确。")
    f.save("fig9-formats.svg", yb + 14)


def fig_truncate():
    f = Fig(W, "左边：GPU 张量核累加的模型。每 16 个乘积连同累加器一起，按其中最大的指数对齐；对齐后最高位往后 25 位以外的低位直接截掉，向零，不是四舍五入。"
               "就像收银台每一单都抹零，只少不多。右边：用全是正数的合成输入，求和长度 K 从 256 到 65,536，"
               "朝零的相对偏差从 2.2e-5 涨到 3.1e-4；不走张量核的 FP32 不到 5e-8。")
    y0 = f.header("GPU 张量核：每 16 个一组，低位抹掉　——　<tspan font-weight=\"700\">只少不多，加得越长少得越多</tspan>",
                  "左：一行 ＝ 一个乘积（对齐后）；虚线右边 ＝ 窗口外的低位，被截掉。右：横轴 ＝ 一次求和加多少个乘积（对数），纵轴 ＝ 结果朝零偏了多少（对数）",
                  [(GY2, "留下的位"), (RD, "截掉的位（向零）"), (GY, "实测：GPU BF16 张量核"), (GR, "实测：FP32 不走张量核")])
    PH = 400
    py = f.panel(0, y0, 640, PH, "一组 16 个乘积 ＋ 累加器，按最大指数对齐（实测的 B200 模型）", BL)
    X0, BW, BH = 40, 13, 14
    WIN = N.TC_KEEP + 1          # 保留「最高位 ＋ 其后 25 位」＝ 26 列（跟仿真代码的定义一致；原来把最高位算在 25 里，少画一列）
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
                f.box(X0 + 60 + pos * BW, y, BW - 2, BH, LINE2, GY2, 2)
            else:
                f.box(X0 + 60 + pos * BW, y, BW - 2, BH, "#fce8e6", RD, 2)
    xw = X0 + 60 + WIN * BW - 1
    f.line(xw, py + 4, xw, py + 24 + 17 * 20, RD, 2, dash="6 4", arrow=False)
    # R3：原来写在窗口线底端，贴着面板下沿被切掉一半 → 挪到顶端、累加器那一行上方
    f.t(xw - 6, py + 16, "最高位后 25 位", RD, True, 13, "end")
    f._pan = None
    px = 670
    py2 = f.panel(px, y0, W - px, PH, "偏差随求和长度长大（实测，合成正数输入，输出写回 BF16）", GY)
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
    f.path(pts, GY, 2.5, arrow=False)
    for i, ((k, e), (x, y)) in enumerate(zip(N.TC_BIAS_K, pts)):
        f.box(x - 7, y - 7, 14, 14, GY, GY, 7)
        # R3：第一个点贴着纵轴，标签往左写会横跨纵轴 → 第一个放到点的右下方
        if i == 0:
            f.t(x + 12, y + 20, "%.1e" % e, GY, True, 13)
        else:
            f.t(x - 10, y - 12, "%.1e" % e, GY, True, 13, "end")
        # 原先压在 K ＝ 65,536 点下面的三行「动画 3.5e-4 口径不同」挪进 📌（图里只留 highlight）
    f.line(GX0, GYf(N.FP32_BIAS_MAX), GX0 + GW, GYf(N.FP32_BIAS_MAX), GR, 2.5, dash="7 5", arrow=False)
    f.t(GX0 + GW, GYf(N.FP32_BIAS_MAX) - 10, "FP32 不走张量核：≤ 5e-8", GR, True, 13, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "四舍五入有多有少，长了能抵消；抹零每次都少，长了就积成一个朝零的系统偏差", [
        "正数、负数输入都一样朝零偏，随 K 增大：加到六万多个，就偏到万分之三。",
        "口径：这条实测含输出舍成 BF16；短求和那几点主要是输出舍入（推断），长求和才是截断主导。",
    ])
    yb = f.src(yb + 10,
               "📌 实测的 B200 张量核模型（一个经验模型，不是厂商规格）：每 16 个乘积连同累加器对齐到最大指数，低于 2^(emax−25) 的位向零截掉，写回 FP32 也向零舍入。「25 位」是拟合出来的窗口宽度，跟 FP32 的 24 位有效数字不是一回事。它在路由打分（K ＝ 4096）上逐位复现了三层全部 4096 × 192 个元素；把参数改成 8 个一组、24 位或就近舍入，命中率都掉到 1% 以下。",
               "📌 实测：合成负数输入 −9.5e-6 / −6.9e-5 / −3.4e-4，同样朝零；TF32 小约一个量级；FP32 不走张量核 ≤ 5e-8。K ＝ 120,832 时合成输入的偏差反而降到 6.4e-5（推断：cuBLAS 对这个形状换了算法），所以图上没有画这一点。",
               "⚠️ 推断：量这条曲线时输出写回的是 BF16。只算截断模型，K ＝ 256 约 3e-7、K ＝ 4096 约 1.1e-5；加上输出舍成 BF16，才回到实测的量级 —— 所以短求和那几点主要是输出舍入，长求和才是截断主导。",
               "注：左图每个乘积右移几位是示意，不是实测数据。",
               "注：动画终点的 3.5e-4 是同一张量核模型的模拟（单条点积，只有累加器截断，不含输出舍到 BF16、不含 cuBLAS 分块），跟右图实测 3.1e-4 口径不同；两者差多少、差在哪，没有单独拆过。另外动画纵轴带正负号（朝零偏画成负的），这张右图画的是偏了多少，取的绝对值。")
    f.save("fig9-truncate.svg", yb + 14)


def fig_tf32():
    f = Fig(W, "FP32 矩阵乘对 FP64 的相对误差，对数刻度。训练容器默认开着 TF32。只把 torch 的两个 TF32 开关关掉，读回来是 False，"
               "但 Transformer Engine 的 FP32 矩阵乘照样走 TF32，误差 2.94e-4 不变；设环境变量 NVIDIA_TF32_OVERRIDE=0 后两条路都是真 FP32，1.15e-6，"
               "而 torch 的开关读回来反而是 True。判据：每次启动实测一遍，1e-5 以下才算真关了。")
    y0 = f.header("「我关了 TF32」不算数　——　<tspan font-weight=\"700\">开关读回来的值，和实际跑的不一样</tspan>",
                  "柱长 ＝ FP32 矩阵乘对 FP64 的误差（对数刻度，一格 ＝ 十倍，越短越准）；左边小字 ＝ 那一刻 torch 的开关读回来是什么",
                  [(RD, "实际走 TF32"), (GR, "实际是真 FP32"), (GY2, "判据线（虚线）")])
    PH = 330
    py = f.panel(0, y0, W, PH, "GPU 训练容器里的 FP32 矩阵乘（实测）", BL)
    bx, bw = 420, 800
    emin, emax = 1e-7, 1e-3
    rows = [("只关 torch 的两个开关", "读回 allow_tf32 ＝ False；Transformer Engine 那条路", N.TF32_TE_ON, RD),
            ("设 NVIDIA_TF32_OVERRIDE=0", "读回 allow_tf32 ＝ True（！）；两条路都关了", N.TF32_ALL_OFF, GR)]

    def LX(v):
        return bx + (math.log10(v) - math.log10(emin)) / (math.log10(emax) - math.log10(emin)) * bw
    # R3：图注说「一格十倍」，原来图上没有格 → 每个十倍画一根竖格线并标数
    for e in range(-7, -2):
        xg = LX(10.0 ** e)
        f.line(xg, py + 26, xg, py + 250, LINE, 1.0, arrow=False)
        f.t(xg, py + 20, "1e%d" % e, GY, size=12, anchor="middle")
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
        "每一轮里面照样是 FP32 累加；舍到 BF16 的是轮与轮之间传下去的部分和（推断）。",
        "代价：这个开关只管前向那几处投影；反向和输出层不受它影响。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：v5p 配置优化后的 HLO 里有 41 个窗口化的点积循环，q 投影的循环累加器是 BF16。打开 FP32 累加（矩阵乘的输出类型设成 FP32，最后舍一次）后，qkv 输出对 GPU 的 rel_L2 2.36e-3 → 1.8e-4。",
               "📌 另一条路是用编译参数整个关掉窗口化：那样连 GPU 本来就有的一次舍入也省掉了，MLP 前归一化的 R 到了 1.64，反而对不齐 —— 多舍少舍都不行，所以用开关不用参数。",
               "⚠️ 推断：一手材料只写到「每一轮的分片部分积以 BF16 输出、在 BF16 累加器里相加」（q 投影的循环累加器是 BF16）；一轮之内那次点积照常在 FP32 里累加、出口舍成 BF16，是按 TPU 矩阵单元的通常做法推的，没有单独看过那一轮的 HLO。所以左图「舍」画在轮与轮之间。",
               "注：左图画的是 4 片 4 轮的示意，实际片数等于 FSDP 的路数。")
    f.save("fig9-windowed.svg", yb + 14)


def fig_excess():
    f = Fig(W, "四行，每行同样四个算子从左往右串起来，红色小块是一次「舍到 BF16」。"
               "第一行 GPU：一个算子一个 kernel，每个算子出口舍一次，共 4 次。"
               "第二行 TPU 默认：XLA 把相邻算子融成块（示意：前两个一块、后两个一块），块里一路用 FP32，只在块出口舍，共 2 次，舍在哪由融合怎么切决定。"
               "第三行关掉 excess precision：块还在，但块里每个算子出口照样舍，共 4 次，位置跟第一行一样。"
               "第四行开关不动，只在一处显式写「舍到 BF16」：那一处一定舍，编译器不会删。"
               "关掉只是让舍入位置确定，不是更准。")
    y0 = f.header("excess precision 改的是「舍在哪儿」　——　<tspan font-weight=\"700\">关掉它，位置定了，不等于更准</tspan>",
                  "从左往右是一串算子；红色小块 ＝ 一次舍到 BF16，数一数每行有几块、落在哪；虚线框 ＝ 编译器融合成的一块（块里不落地）",
                  [(GY, "GPU 上的算子"), (BL, "TPU 上的算子"), (RD, "舍到 BF16"), (PU, "显式写的那一次")])
    RH, NOP = 104, 4
    PH = 40 + 4 * RH
    py = f.panel(0, y0, W, PH, "示意：同样四个算子，四种舍法", BL)
    X0, OW, GAP = 330, 118, 74
    rows = [  # (标题, 小字, 颜色, 融合块, 哪些出口舍, 显式舍的位置, 右侧两行)
        ("GPU：逐个算子", "一个算子一个 kernel，输出写回 BF16", GY, [], [0, 1, 2, 3], None,
         ("舍 4 次", "每个算子出口一次")),
        ("TPU 默认", "excess precision 开：块里不舍", BL, [(0, 1), (2, 3)], [1, 3], None,
         ("舍 2 次", "落在融合块边界上")),
        ("关掉 excess precision", "块还在，块里照写的舍", BL, [(0, 1), (2, 3)], [0, 1, 2, 3], None,
         ("舍 4 次", "位置跟 GPU 那一行一样")),
        ("只在一处显式写", "开关不动，写「舍到 BF16」", BL, [(0, 1), (2, 3)], [1, 3], 0,
         ("舍 3 次", "写了的那处一定舍")),
    ]

    def ox(k):
        return X0 + k * (OW + GAP)
    for r, (lab, sub, col, fus, cuts, extra, (cnt, where)) in enumerate(rows):
        y = py + 26 + r * RH
        f.t(24, y + 30, lab, col if col != GY else INK, True, 15)
        f.t(24, y + 54, sub, GY, size=13)
        for a, b in fus:
            f.box(ox(a) - 12, y + 4, ox(b) + OW + 14 - (ox(a) - 12), 58, "none", BL, 10, sw=1.6, dash="7 5")
        for k in range(NOP):
            fill = "#e8f0fe" if col == BL else "#fff"
            f.box(ox(k), y + 11, OW, 46, fill, col, 6)
            f.t(ox(k) + OW / 2.0, y + 40, "算子 %d" % (k + 1), col, True, 14, "middle")
            xa = ox(k) + OW
            f.line(xa + 4, y + 34, xa + GAP - 6, y + 34, GY2, 1.6, arrow=(k < NOP - 1))
            if k in cuts or k == extra:
                c = PU if k == extra else RD
                f.box(xa + 22, y + 15, 16, 38, "#f3e8fd" if k == extra else "#fce8e6", c, 3, sw=2 if k == extra else 1)
                f.t(xa + 30, y + 80, "你写的" if k == extra else "舍", c, True, 12, "middle")
        f.t(1130, y + 30, cnt, RD, True, 16)
        f.t(1130, y + 54, where, GY, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "开着时少舍几次，离精确值更近，可舍入落在哪由融合怎么切说了算；关掉换来的是位置确定", [
        "要跟 GPU 在某一处舍得一样，就在那一处写「舍到 BF16」（JAX 的 lax.reduce_precision），别整个关掉。",
        "代价：关掉不会更准，跟 GPU 也未必更近 —— GPU 自己也有在 FP32 里算完只舍一次的融合算子（比如归一化）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（混元 3 · v5p，只加关掉 excess precision 这一个编译参数）：单层回放里每个张量对 GPU 的 rel_L2 都变大，约大 10%%–60%%，例如 qkv 输出 %s → %s；跑 1000 步，第 301–1000 步 loss 偏移 %s → %s，小了一半；参数范数的漂移一点没少。"
               % (N.sci(N.QKV_VS_GPU_BEFORE), N.sci(N.QKV_VS_GPU_NOEXC), N.sci(N.NOEXC_LOSS_TAIL_ON), N.sci(N.NOEXC_LOSS_TAIL)),
               "📌 实测：MoE 层 MLP 前的 RMSNorm，代码里写的是舍两次，开着时 XLA 把中间那次省掉，跟 GPU（FP32 算完舍一次的融合 kernel）逐位相同 %s；关掉以后只剩 %s。这一处恰好是「开着更像 GPU」。"
               % (N.RMSNORM_BITWISE_EXC, N.RMSNORM_BITWISE_NOEXC),
               "注：四个算子、融合切成两块、显式舍写在第一个算子出口，都是示意；真实的融合怎么切由编译器决定。GPU 那一行画的是「每个算子单独一个 kernel」的常见情形。")
    f.save("fig9-excess.svg", yb + 14)


def fig_boundary():
    f = Fig(W, "一张检查表，画成五条路径。横向五格是一个算子的五个环节：输入、乘法（矩阵乘精度）、累加、输出存储、舍在哪；"
               "每一格里线的高低表示这个值在几位里：高 ＝ 32 位（FP32），低 ＝ 16 位（BF16）。蓝线是 TPU，灰色虚线是 GPU；"
               "两条线分开的那一格涂红，就是要对上的边界。五个例子正好每列一个。"
               "第一行，路由打分乘哪份权重：TPU 原来乘 FP32 主权重，GPU 乘舍成 BF16 的权重。"
               "第二行，名义 FP32 的输出头：TPU 默认精度下 FP32 操作数只算一遍 BF16，GPU 是真 FP32。"
               "第三行，FSDP 下的投影：TPU 部分和在轮与轮之间舍成 BF16，GPU 在 FP32 里加完。"
               "第四行，注意力打分 q 乘 k：GPU 参照自己写的实现先把结果存成 BF16 再进 softmax，TPU 和生产的融合内核都留在 FP32。"
               "第五行，RMSNorm：TPU 默认归一化结果和缩放各舍一次，GPU 算完只舍一次。"
               "两条线全程重合的算子只剩加法顺序的差，那是噪声；分开一格，就是一个要对上的边界。")
    y0 = f.header("每个算子两边各画一条精度路径，分开的那一格就是要对上的边界",
                  "前四格：线高 ＝ 值在 32 位、线低 ＝ 16 位；最后一格数舍了几次；蓝线蓝字 ＝ TPU，灰虚线黑字 ＝ GPU；红底 ＝ 两边分开的那一格",
                  [(BL, "TPU"), (GY, "GPU"), (RD, "两边不一样")])
    COLS = ["输入", "乘法 ／ 矩阵乘精度", "累加", "输出 ／ 存储", "舍在哪"]
    X0, CW, CG, RH = 300, 186, 8, 96
    # (算子, 出处, 哪一列不一样, TPU 五格位宽, GPU 五格位宽, 共同的小字（不一样那格写两边各自的）)
    FIXSIDE = ["改 TPU", "改 TPU", "改 TPU", "改 GPU", "改 TPU"]
    rows = [
        ("路由打分：乘哪份权重", "第二次迁移", 0, [32, 32, 32, 32, 32], [16, 32, 32, 32, 32],
         [("FP32 主权重", "BF16 权重"), "FP32 里乘", "FP32", "FP32 打分", "—"]),
        ("名义 FP32 的输出头", "第二次迁移", 1, [32, 16, 32, 32, 32], [32, 32, 32, 32, 32],
         ["FP32", ("默认：单遍 BF16", "真 FP32（TF32 关）"), "FP32", "FP32", "—"]),
        ("FSDP 切开收缩维的投影", "混元 3", 2, [16, 16, 16, 16, 16], [16, 16, 32, 16, 16],
         ["BF16", "BF16 × BF16", ("轮与轮之间舍 BF16", "FP32 里加完"), "BF16", "随累加那格"]),
        ("注意力打分 q·k", "第二次迁移", 3, [16, 16, 32, 32, 32], [16, 16, 32, 16, 32],
         ["BF16", "BF16 × BF16", "FP32", ("留 FP32", "先存 BF16"), "softmax 在 FP32"]),
        ("RMSNorm", "混元 3", 4, [16, 32, 32, 16, 16], [16, 32, 32, 16, 32],
         ["BF16", "FP32 里算", "FP32", "BF16", ("TPU 舍 2 次", "GPU 舍 1 次")]),
    ]
    PH = 70 + len(rows) * RH
    py = f.panel(0, y0, W, PH, "五个例子，五列各一个", BL)
    for j, c in enumerate(COLS):
        f.t(X0 + j * (CW + CG) + CW / 2.0, py + 24, c, INK, True, 14, "middle")
    f.t(X0 + 5 * (CW + CG) + 6, py + 24, "改哪边", INK, True, 14)
    y = py + 36
    for (op, where, bad, tpu, gpu, labs) in rows:
        f.t(20, y + 36, op, INK, True, 15)
        f.t(20, y + 58, where, GY, size=12)
        hi, lo = y + 30, y + 58
        for j in range(5):
            x = X0 + j * (CW + CG)
            f.box(x, y + 4, CW, RH - 14, "#fce8e6" if j == bad else "#f8f9fa", RD if j == bad else LINE2, 4, sw=1.6 if j == bad else 1)
            if j == bad and j == 4:      # 「舍在哪」：数刻痕，不用线高（线高在这一列没有意义）
                ta, ga = labs[j]
                for k, (lab, n, col) in enumerate([(ta, 2, BL), (ga, 1, INK)]):
                    yy = y + 22 + k * 34
                    for m in range(n):
                        f.box(x + 14 + m * 12, yy - 12, 6, 16, RD, RD, 1)
                    f.t(x + 48, yy + 1, lab, col, True, 12)
            elif j == bad:
                ta, ga = labs[j]
                f.t(x + CW / 2.0, hi - 9 if tpu[j] == 32 else lo + 22, ta, BL, True, 12.5, "middle")
                f.t(x + CW / 2.0, hi - 9 if gpu[j] == 32 else lo + 22, ga, INK, True, 12.5, "middle")
            else:
                f.t(x + CW / 2.0, y + RH - 16, labs[j], GY, size=12, anchor="middle")
        # 两条阶梯线：TPU 实线，GPU 虚线、往下错 4 px，重合时也看得见两条
        for vals, col, dash, off in [(tpu, BL, None, 0), (gpu, GY, "6 4", 5)]:
            pts = []
            for j, v in enumerate(vals[:4]):
                x = X0 + j * (CW + CG)
                yy = (hi if v == 32 else lo) + off
                pts.append((x + 10, yy)); pts.append((x + CW - 10, yy))
            for (xa, ya), (xb, yb_) in zip(pts, pts[1:]):      # 逐段画：体检按线段数「形状」，一条 path 只算一笔
                f.line(xa, ya, xb, yb_, col, 2.6 if col == BL else 2.0, dash=dash, arrow=False)
        f.t(X0 + 5 * (CW + CG) + 6, y + 46, FIXSIDE[rows.index((op, where, bad, tpu, gpu, labs))], RD if FIXSIDE[rows.index((op, where, bad, tpu, gpu, labs))] == "改 GPU" else GY, True, 13)
        y += RH
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "两条线重合、格里的字也一样，才只剩加法顺序的差（噪声）；分开一格，就是一个要对上的边界", [
        "看代码找对不齐，就是把每个算子两边各写一行五格，逐格比；有差的格子再去找开关，没有开关的记一行再定谁改。",
        "代价：GPU 重跑量不出这种差 —— 重跑之间舍入位置一模一样，尺子在这里宽度是 0；要靠逐格读代码，再拿 FP64 定责。",
    ])
    yb = f.src(yb + 10,
               "📌 出处：第 2、4 行是第二次迁移（一个更新的、带稀疏注意力的 MoE 模型）读代码查出的；第 3、5 行是混元 3。第 1 行是混元 3 时期就有的开关，第二次迁移才需要打开：TPU 把路由打分设成最高精度以后才出现这个差；混元 3 有同一个开关，但那时默认精度本来就先舍 BF16，开不开逐位相同。第 4 行是参照一侧自己写的不融合实现，生产的融合内核和 TPU 内核都留 FP32，所以改的是 GPU。",
               "📌 第 4 行只改这一处，第 1 步 loss %s、梯度范数 %s，比 GPU 自己的噪声（%s ／ %s）小几十倍 —— 它是按写法定的 bug，不是按大小抓到的。"
               % (N.ATTN_FIX_LOSS_T, N.ATTN_FIX_GN_T, N.ATTN_NOISE_LOSS_T, N.ATTN_NOISE_GN_T),
               "📌 同在「乘法」这一列的还有一处：TPU 的 HIGHEST 精度是多遍 BF16 拼出来的近似 FP32，GPU 是逐元素 FP32 乘加；一处多项加权和改成逐项 FP32 乘加以后，第 1 步对 FP32 参照 R ＞ 1.2 的张量 %s（共 %d 个）。"
               % (N.MIX_OVER_T, N.MIX_TENSORS),
               "注：线高只分 16 位、32 位两档，HIGHEST 这类「多遍 BF16 拼近似 FP32」也画在 32 位，所以乘法那一格要比字、不只比高度（HIGHEST 对逐元素 FP32 乘加照样差末位）；「舍在哪」一列不画线，红色刻痕一道 ＝ 舍一次到 16 位，只在它本身是根子时画（第 5 行：TPU 默认归一化结果、缩放各舍一次，GPU 算完只舍一次）。第 3 行累加那一格：一轮之内照常 FP32 累加，舍的是轮与轮之间的部分和（§5.4 的推断）。")
    f.save("fig9-boundary.svg", yb + 14)


def fig_captain():
    f = Fig(W, "左格：船上称船长。先称船加船长，再称空船，两次相减得船长的体重。船 %d 吨，船长 %d 公斤，秤能读到 %s 公斤一格，约七位有效数字，已经很准；"
               "可这一格落到船长身上，就是他体重的 %s。下面一行对回模型：船是 log-sum-exp，约 %.1f；船长是蒸馏损失 KL，每行约 %s；秤的一格是 f32 在 %.1f 处的一格，约 %s，等于 KL 的 %s。"
               "右格：病态还是 bug，问三件事。它的输入两边离参照一样近吗？R %s，一样近。由它出发的梯度一样近吗？R %s，一样近。"
               "它是不是两个大数相减剩下的小数、差折成大数的格在一两格以内？TPU 的 lse 平均差 %s 格。三条都是，就是病态在放大。"
               "第四问：被放大的那一格是谁的？同样的 f32 计算换到 CPU（PyTorch）上只差 −0.004 格，TPU 差 %s 格，所以那一格来自 TPU 默认的 exp 和 log。"
               "换成高精度的 exp、log 以后，lse 的偏差 %s，300 步那一项的 r %s（同一代还开了另一个开关，没拆开）。"
               % (N.SHIP_T, N.CAPTAIN_KG, N.TICK_KG_T, N.TICK_PCT_T, N.KD_LSE, N.sci(N.KD_KL), N.KD_LSE, N.KD_ULP_T, N.KD_ULP_OF_KL_T,
                  N.KD_IN_R, N.KD_GRAD_R, N.KD_LSE_TPU_ULP_T, N.KD_LSE_TPU_ULP_T, N.KD_FIX_LSE_T, N.KD_FIX_R300_T))
    y0 = f.header("小数是两个大数相减时，f32 差一格就是千分之几　——　<tspan font-weight=\"700\">病态是放大器，不是根因</tspan>",
                  "左边：在一杆称船的秤上称船长；右边：前三问认出放大器，第四问找被放大的那一格是谁的",
                  [(GY, "大数（船、lse）"), (OR, "小数（船长、KL）"), (RD, "秤的一格")])
    PH = 470
    LW = 800
    py = f.panel(0, y0, LW, PH, "船上称船长（比方；比例跟实测一样）", OR)
    # 两次称重：两根几乎一样长的灰条，船长那一截细到看不见
    BX, BL_ = 30, 480
    for i, (lab, val, cap) in enumerate([("船 ＋ 船长", "%s 公斤" % format(int(N.SHIP_T * 1000 + N.CAPTAIN_KG), ","), True),
                                         ("空船", "%s 公斤" % format(int(N.SHIP_T * 1000), ","), False)]):
        y = py + 30 + i * 56
        f.t(BX, y + 22, lab, INK, True, 14)
        f.box(BX + 110, y + 6, BL_, 26, "#e8eaed", GY, 4)
        if cap:
            f.box(BX + 110 + BL_, y + 6, 2, 26, OR, OR, 0)
        f.t(BX + 110 + BL_ + 14, y + 25, val, INK, True, 13)
    f.t(BX, py + 162, "两次相减 ＝ %d 公斤：船长只有船的约 1/%s，两根条看不出差别" % (N.CAPTAIN_KG, format(int(round(N.KD_RATIO, -3)), ",")), GY, size=13)
    # 放大镜：船长那一截，按秤的一格画刻度
    ZY, ZX, ZL = py + 196, 30, 740
    f.t(ZX, ZY, "放大到船长那一截：一格 %s 公斤" % N.TICK_KG_T, INK, True, 14)
    f.box(ZX, ZY + 14, ZL, 34, "#fef7e0", OR, 4)
    f.t(ZX + ZL / 2.0, ZY + 37, "船长 %d 公斤" % N.CAPTAIN_KG, OR, True, 14, "middle")
    tick = ZL * N.TICK_KG / N.CAPTAIN_KG
    f.box(ZX + ZL - tick, ZY + 10, tick, 42, RD, RD, 0)
    f.line(ZX + ZL - tick / 2.0, ZY + 56, ZX + ZL - 150, ZY + 86, RD, 1.2, arrow=False)
    f.t(ZX + ZL - 156, ZY + 92, "差一格 ＝ 船长体重的 %s" % N.TICK_PCT_T, RD, True, 14, "end")
    # 对回模型
    my = py + 330
    f.line(ZX, my - 16, ZX + ZL, my - 16, LINE, 1, arrow=False)
    for k, (a, b) in enumerate([("船", "log-sum-exp ≈ %.1f" % N.KD_LSE), ("船长", "辅助损失 KL ≈ %s" % N.sci(N.KD_KL)),
                                ("秤的一格", "f32 在 lse 处一格 ≈ %s" % N.KD_ULP_T)]):
        x = ZX + k * 250
        f.t(x, my + 8, a, OR if k == 1 else (RD if k == 2 else GY), True, 14)
        f.t(x, my + 32, b, INK, size=13)
    f.t(ZX, my + 64, "一格 ＝ KL 的 %s；实测偏低 %s（一行对 FP64），同一量级" % (N.KD_ULP_OF_KL_T, N.KD_DIFF_PCT_T), RD, True, 13)
    f._pan = None
    px = LW + 20
    py2 = f.panel(px, y0, W - px, PH, "病态还是 bug：问四件事（实测）", GR)
    qs = [("① 它的输入，两边离参照一样近？", "三组输入：R %s" % N.KD_IN_R),
          ("② 由它出发的梯度，两边一样近？", "20 个张量：R %s" % N.KD_GRAD_R),
          ("③ 大数相减剩小数、差在一两格内？", "lse 对 FP64：平均 %s 格" % N.KD_LSE_TPU_ULP_T),
          ("④ 那一格是谁的？同样的计算换 CPU 算", "CPU（PyTorch）−0.004 格，TPU %s 格" % N.KD_LSE_TPU_ULP_T)]
    for i, (q, a) in enumerate(qs):
        y = py2 + 22 + i * 76
        f.t(px + 18, y + 20, q, INK, True, 14)
        f.t(px + 40, y + 46, a, RD if i == 3 else GR, True, 13)
        f.t(px + 18, y + 46, "→" if i == 3 else "✓", RD if i == 3 else GR, True, 15)
        bx, U = px + 330, 160          # 右侧小图：① ② 画 GPU 距离（灰，＝ 1）与 TPU 距离（蓝，区间）、1.2 门槛虚线；③ 画 ±2 格的刻度
        if i < 2:
            lo, hi = [float(v) for v in (N.KD_IN_R if i == 0 else N.KD_GRAD_R).split("–")]
            f.box(bx, y + 6, U, 9, "#dadce0", GY, 2)
            f.box(bx, y + 22, U * lo, 9, BL, BL, 2)
            f.box(bx + U * lo, y + 22, U * (hi - lo), 9, "#aecbfa", BL, 2)
            f.line(bx + U * N.R_GATE, y, bx + U * N.R_GATE, y + 38, GR, 1.4, dash="4 3", arrow=False)
            f.t(bx + U * N.R_GATE + 4, y + 50, "1.2", GR, size=11)
            f.t(bx - 6, y + 14, "GPU", GY, size=11, anchor="end")
            f.t(bx - 6, y + 31, "TPU", BL, size=11, anchor="end")
        elif i == 2:
            f.line(bx, y + 20, bx + U * 1.3, y + 20, GY2, 1.2, arrow=False)
            for k in range(5):
                xx = bx + k * U * 1.3 / 4
                f.line(xx, y + 14, xx, y + 26, GY2, 1.2, arrow=False)
                f.t(xx, y + 42, str(k - 2), GY, size=11, anchor="middle")
            xm = bx + (2 + N.KD_LSE_TPU_ULP) * U * 1.3 / 4
            f.box(xm - 5, y + 13, 10, 14, RD, RD, 5)
            f.t(bx + U * 1.3 / 2, y + 58, "单位：f32 的格", GY, size=11, anchor="middle")
        else:                           # ④：两根条，CPU 几乎为零、TPU 一格
            for k, (lab, v, col) in enumerate([("CPU", N.KD_LSE_CPU_ULP, GY), ("TPU", N.KD_LSE_TPU_ULP, BL)]):
                f.t(bx - 6, y + 14 + k * 17, lab, col, size=11, anchor="end")
                f.box(bx, y + 6 + k * 17, max(2, U * abs(v)), 9, col, col, 2)
    f.box(px + 16, py2 + 330, W - px - 32, 80, "#e6f4ea", GR, 6)
    f.t(px + 30, py2 + 356, "①–③ 是：病态在放大；④ 指向 TPU 的 exp、log", GR, True, 14)
    f.t(px + 30, py2 + 384, "换高精度写法：lse %s；300 步 r %s" % (N.KD_FIX_LSE_T, N.KD_FIX_R300_T), GR, True, 13)
    f.t(px + 18, py2 + 434, "①② 有一条不是 → 回到五格表找边界", GY, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "病态把一格以内的差放大成千分之一；放大器认出来以后，还要问被放大的那一格是谁的", [
        "这一次那一格是 TPU 默认的 exp、log（偏低、只有几十格的精度）；换成高精度写法，偏差就没了。",
        "代价：要是那一格是两边都躲不开的末位（CPU 上也偏），才写「不过，已定责」，并附上这几问的证据。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，第 1 步，稀疏注意力里的一项辅助蒸馏损失）：同一行 TPU 图内对同一输入的 FP64 重算 %s（对 GPU、全部运行合起来 %s）；TPU 图内 lse 对 FP64 平均 %s 格（只有 %s 的行偏正），CPU 上 PyTorch 的 f32 只差 %s 格（CPU 上的 JAX f32 算同一行 KD 是 ＋2.9e-4，不是同一种实现）；输入与梯度见右格。lse 与同一行 KD 的前后两端设置不同；300 步 r 那一对同一代还开了一个优化器开关，没有拆开。"
               % (N.KD_DIFF_T, N.KD_VS_GPU_T, N.KD_LSE_TPU_ULP_T, N.KD_POS_ROWS, ("%.3f" % N.KD_LSE_CPU_ULP).replace("-", "−")),
               "⚠️ 本课推导：比方按 lse 1 单位 ↔ 1000 吨换算，船 %d 吨、船长 %d 公斤、一格 %s 公斤，比例跟实测的「一格 ＝ KL 的 %s」一致。⚠️ 照学生那一侧算，lse 低 0.95 格该让 KL 低约 4.1e-3，实测是 1.2e-3：两者只对得上量级，一手材料没有逐项对上（推断：教师那一侧的 lse 也偏、抵掉一部分，未核实）。"
               % (N.SHIP_T, N.CAPTAIN_KG, N.TICK_KG_T, N.KD_ULP_OF_KL_T),
               "注：秤读到 0.47 公斤，对 7100 吨约是七位有效数字，正是 f32 的水平（约 7 位十进制）。这杆秤就是 f32 格式本身，不是前面那把噪声尺子。")
    f.save("fig9-captain.svg", yb + 14)


def fig_elide():
    f = Fig(W, "一条横轴：1000 步主 loss 的带符号均值，TPU 对 GPU 8 份的平均，单位 1e-6。灰色一段是 GPU 自己两两最大的那一对，正负 %s。"
               "第一行：一处加权和照 GPU 改写法之前，TPU 在 ＋1.1，落在灰段里。第二行：改了以后五次 TPU 运行，其中三次只改执行参数，全在右边，%s，同一个方向。"
               "五次里四次出了灰段。第三行：在两处强制写一次「舍到 BF16」以后，约 ＋3.5（推算），回到灰段里。"
               % (N.ELIDE_GPU_MAX_T, N.ELIDE_RANGE_T))
    y0 = f.header("编译器省掉一次舍入，1000 步 loss 五次同号偏高、四次出了噪声",
                  "横轴 ＝ 1000 步主 loss 的带符号均值（TPU 对 GPU 8 份平均，×1e-6）；灰段 ＝ GPU 自己两两最大的差；一个点 ＝ 一次 TPU 运行",
                  [(GY, "GPU 自己的噪声"), (BL, "TPU 运行"), (GR, "强制舍入以后（推算）")])
    PH = 300
    py = f.panel(0, y0, W, PH, "同一 TPU，三版代码，同一把尺子", BL)
    X0, X1, LO, HI = 360, 1320, -5.0, 10.0
    def X(v):
        return X0 + (v * 1e6 - LO) / (HI - LO) * (X1 - X0)
    yt, yb_ = py + 20, py + PH - 70
    f.box(X(-N.ELIDE_GPU_MAX), yt, X(N.ELIDE_GPU_MAX) - X(-N.ELIDE_GPU_MAX), yb_ - yt, "#e8eaed", "none", 0)
    f.line(X(0), yt, X(0), yb_, GY2, 1, dash="3 3", arrow=False)
    for v in range(-4, 11, 2):
        f.line(X(v * 1e-6), yb_, X(v * 1e-6), yb_ + 6, GY2, 1, arrow=False)
        f.t(X(v * 1e-6), yb_ + 22, ("＋%d" % v) if v > 0 else str(v).replace("-", "−"), GY, size=12, anchor="middle")
    rows = [("改写法之前", [N.ELIDE_BEFORE], BL, True),
            ("改写法以后（5 次）", list(N.ELIDE_RUNS), BL, True),
            ("两处强制舍入以后（推算）", [N.ELIDE_AFTER], GR, False)]
    for i, (lab, vals, col, filled) in enumerate(rows):
        y = py + 70 + i * 66
        f.t(24, y + 5, lab, INK, True, 15)
        for v in vals:
            f.p.append('<circle cx="%.1f" cy="%d" r="8" fill="%s" stroke="%s" stroke-width="2"/>' % (X(v), y, col if filled else "#fff", col))
    f.t(X(N.ELIDE_GPU_MAX) + 8, yt + 18, "GPU 自己最大 ±%s" % N.ELIDE_GPU_MAX_T, GY, size=12)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "GPU 每次都把中间值存成 BF16，TPU 上这次往返被编译器拿掉：值更精确，跟 GPU 反而对不上", [
        "要和 GPU 舍在同一处，就在那一处显式写「舍到 BF16」（JAX 的 lax.reduce_precision），编译器不会删。",
        "代价：显式舍入是为了对齐，生产里关掉同样准（推断）；剩下约 ＋3.5e-6 还没查清。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移）：五次 TPU 运行对 GPU 8 份均值 %s，GPU 两两 28 对的带符号均值最大 %s；强制舍入后判定 r %s（对 GPU 基线，带符号均值），对 GPU 均值约 ＋3.5e-6（推算）。"
               % (N.ELIDE_RANGE_T, N.ELIDE_GPU_MAX_T, N.ELIDE_R_T),
               "⚠️ 推断：「往返被编译器省掉」由开关的效果推出，没有去看编译后的程序。五次里有三次只改执行参数（注意力分块、专家并行、MoE 实现），同号偏高，说明不是轨迹偶然分开。")
    f.save("fig9-elide.svg", yb + 14)


def fig_trans():
    f = Fig(W, "左格：四个超越函数，横条长度是对正确舍入的最大误差，按格（ulp）计，对数刻度。"
               "TPU 默认：%s。TPU 高精度写法：%s。GPU：exp 2、log 1、sigmoid 和 silu 约 4。"
               "右格：高精度的 exp 怎么用软件算：先把 x 拆成 k 个 ln2 加一个小余数 s，余数上用多项式算 e 的 s 次方，再乘 2 的 k 次方，只改指数位、精确；"
               "求倒数用牛顿修正，每修一次误差平方一次。" % (N.TRANS_T, N.TRANS_HI_T))
    y0 = f.header("TPU 默认的 exp、log 差几十格，GPU 只差个位数",
                  "左：条长 ＝ 对正确舍入的最大误差（格，对数刻度）；右：高精度的软件 exp 和倒数，分几步算",
                  [(BL, "TPU 默认"), (GR, "TPU 高精度写法"), (GY, "GPU")])
    PH = 420
    LW = 760
    py = f.panel(0, y0, LW, PH, "最大误差，按格计（TPU 实测；GPU 按文档，sigmoid、silu 为推断）", BL)
    X0, X1 = 150, LW - 60
    import math as _m
    def X(u):
        return X0 + _m.log10(max(u, 1)) / _m.log10(200) * (X1 - X0)
    for u in (1, 2, 5, 10, 20, 50, 100, 200):
        f.line(X(u), py + 30, X(u), py + PH - 72, LINE2, 1, arrow=False)
        f.t(X(u), py + PH - 54, str(u), GY, size=12, anchor="middle")
    for i, (fn, (d, h, g)) in enumerate(N.TRANS.items()):
        y = py + 30 + i * 76
        f.t(24, y + 26, fn, INK, True, 16)
        for k, (v, col) in enumerate([(d, BL), (h, GR), (g, GY)]):
            yy = y + k * 20
            f.box(X0, yy, max(4, X(v) - X0), 14, col, col, 2)
            f.t(X(v) + 6 if X(v) > X0 + 4 else X0 + 10, yy + 12, str(v), col, True, 12)
    f._pan = None
    px = LW + 20
    py2 = f.panel(px, y0, W - px, PH, "高精度的软件实现：分三步（示意）", GR)
    steps = [("① 区间缩减", "x ＝ k·ln2 ＋ s，|s| ≤ ln2 ÷ 2", "把大范围的 x 压到一小段"),
             ("② 多项式", "eˢ ≈ 1 ＋ s ＋ s²/2 ＋ … ＋ s⁶/720", "s 小，几项就到 FP32 末位"),
             ("③ 乘 2ᵏ", "只改指数位，精确", "eˣ ＝ 2ᵏ · eˢ"),
             ("倒数：牛顿修正", "y ← y·(2 − a·y)", "每修一次，误差平方一次")]
    for i, (a, b, c) in enumerate(steps):
        y = py2 + 30 + i * 86
        f.t(px + 20, y + 18, a, GR if i < 3 else PU, True, 15)
        f.t(px + 40, y + 44, b, INK, True, 14)
        f.t(px + 40, y + 66, c, GY, size=12)
        if i < 2:
            f.line(px + 26, y + 52, px + 26, y + 92, GY2, 1.2)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "默认的 exp 不只是粗，还系统偏低：真实输入上只有 %s 正好舍对，平均偏 %s 格" % (N.EXP_CR_FRAC, N.EXP_SIGNED_ULP_T), [
        "偏低的 exp 进了 log-sum-exp，再被病态放大，就是上一节那千分之一。",
        "代价：单算子对上了 GPU 的精度，整网未必跟着对上 —— sigmoid、silu 换成精确写法后整网反而偏得多，根因还在查；改完要整网再验。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（TPU v5p，FP32，对 FP64 正确舍入数格）：默认 %s；高精度写法 %s。GPU：%s。TPU 默认的上限跟 JAX 文档给这一代芯片写的一致。"
               % (N.TRANS_T, N.TRANS_HI_T, N.TRANS_GPU_NOTE),
               "⚠️ 右格是教科书做法（区间缩减、多项式、牛顿迭代），只讲思路；各家库的系数、项数和缩减方式不同。sigmoid、silu 的高精度写法用高精度的 exp 搭：e ＝ exp(−|x|)，再算 1 ÷ (1 ＋ e)，倒数做一次牛顿修正。")
    f.save("fig9-trans.svg", yb + 14)


fig_formats()
fig_truncate()
fig_tf32()
fig_windowed()
fig_excess()
fig_boundary()
fig_captain()
fig_elide()
fig_trans()
