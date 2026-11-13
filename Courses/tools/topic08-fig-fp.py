# -*- coding: utf-8 -*-
r"""专题八 · 第五节「浮点与硬件」的四张图。

fig8-formats   回答：BF16、TF32、FP32 差在哪？两个 BF16 相乘为什么不丢精度、舍入发生在哪？
               数字：位宽来自 IEEE 754 / 各家格式定义；「乘积在 FP32 里精确」是本课推导（8 位 × 8 位有效数字 ＝ 至多 16 位 ≤ 24 位）。
fig8-truncate  回答：GPU 张量核在长求和里为什么系统地偏小？—— 每 16 个乘积对齐到最大指数，低位直接截掉（向零），像收银台每单都抹零。
               数字：实测的 B200 张量核模型（16 个一组、保留 25 位、向零截断）；合成正数输入的偏差随求和长度 K 变化（实测）。
               ⛔ 刻意没画：K ＝ 120,832 时合成输入的偏差反而降到 6.4e-5（推断是 cuBLAS 换了算法），写在出处里。
fig8-tf32      回答：「我关了 TF32」为什么不算数？—— 开关读回来的值和实际跑的不一样。
fig8-inout     回答：FSDP 切同一维，为什么有的投影多舍、有的不多舍？—— 切在求和维上每片只给一份部分和、要加；切在输出维上只拼。
               数字：第三次迁移单算子回放 R 1.41 ／ 1.00（实测）；切哪一维取自编译产物里的权重片形状。
fig8-windowed  回答：为什么 8 个设备才多舍、多舍几次？—— 编译器把一次矩阵乘拆成边传边乘，部分和在 BF16 累加器里加，一次求和舍 6 次；4 个设备不拆。
               数字：循环体结构是编译产物显示；4／8／16 个设备的循环个数是离线编译实测。2026-11-13 起替代原来「4 片 4 轮、舍在轮与轮之间（推断）」的示意。
fig8-lmhead    回答：输出头默认精度档为什么前 300 步差出 7 倍、第 1 步却看不见？—— 默认档把 FP32 输入舍成 BF16，舍的是训练后离开格点的主权重。
               数字：单独回放一次矩阵乘的误差、曲线与步时（实测，单份）；左格格点是示意。
fig8-excess    回答：excess precision 改的是什么？—— 舍在哪儿：融合块里中间那几次「舍到 BF16」被删掉，只在块出口舍；关掉它，舍入回到每个算子出口。
               示意图：算子个数、融合怎么切都是示意，不带实测数；实测（关掉以后单层更远、1000 步 loss 偏移小一半）只进 📌。
               ⛔ 刻意没画「哪一行更准」：这张图只讲舍入位置。关掉不是更准，画个准确度刻度就讲反了。
fig8-normscale 回答：同一行「转 BF16 再转回」，为什么 4K 舍、32K 不舍？为什么几十步以后才显？—— 删不删跟着切法变；scale 初值在格点上，要等 FP32 原值偏开。
               数字：扫描、二分、scale 偏离都实测；外推虚线与「为什么切法一变就删」是推断。
fig8-boundary  回答：对不齐通常出在哪？—— 一个值在 16 位和 32 位之间进出的地方。每个算子两边各写一行五格 dtype 路径，
               五列各放一个真实例子，不一样的那一格画红框（2026-11-07 第二次迁移的教训，例子取自两次迁移）。
               ⛔ 刻意没画误差大小：这张图是一张检查表，量级在出处里；第 4 行的效果在第 1 步比 GPU 噪声小几十倍，画柱子会让人以为它不重要。
fig8-elide     回答：编译器「更精确」为什么也会对不齐？—— GPU 一定会做的 FP32→BF16→FP32 往返，TPU 上被省掉，1000 步 loss 次次偏高。
               数字全是实测（五次运行、GPU 噪声上限、强制舍入后）；「后端只省一半」在最小复现里看到，长跑偏高归到它是推断。
fig8-trans     回答：TPU 默认的 exp、log 有多粗？高精度的软件实现怎么算？—— 默认几十到上百格，GPU 个位数；区间缩减 ＋ 多项式 ＋ 牛顿修正。
               右格的算法是教科书做法（示意），不是哪一家库的源码。
fig8-ulp2      回答：精确超越函数的开关在两代芯片上管不管用？—— v5p 管用（几十格 → 个位数），v7x 配 0.0.46 不起作用。数字全是实测。
fig8-captain   回答：一个小标量对不上，怎么分清是病态还是 bug？—— 船上称船长：船 7100 吨、船长 110 公斤，秤的一格 0.47 公斤 ＝ 船长的 0.43%。
               比方的数按 lse 1 单位 ↔ 1000 吨换算（本课推导），比例与实测一致：f32 一格 ＝ KL 的 4.3e-3。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU, wpx, MONO_K
import topic08_numbers as N

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
    f.save("fig8-formats.svg", yb + 14)


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
    f.save("fig8-truncate.svg", yb + 14)


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
    f.save("fig8-tf32.svg", yb + 14)


def fig_inout():
    f = Fig(W, "左：往里投的矩阵（例如 q 的低秩下投影、MLP 的 gate 和 up），模型宽度是输入维，也就是矩阵乘里被加起来的求和维。"
               "FSDP 把它切成 8 片，每片权重只能算出整个输出的一份部分和，8 份必须加起来。"
               "右：往外投的矩阵（注意力输出投影 o_proj、MLP 的 down），模型宽度是输出维，每片权重算出输出里完整的几列，8 段只是拼起来，没有加法。"
               "下：第三次迁移单算子回放（8 个设备），往里投的 7 种投影 TPU 离 FP64 是 GPU 的 %.2f 倍，往外投的 3 种是 %.2f 倍。" % (N.M3_PROJ_R, N.M3_OUTPROJ_R))
    y0 = f.header("同一刀，落在求和维上要加，落在输出维上只拼",
                  "y ＝ x·W。FSDP 把 W 沿模型宽度切成 8 片、每个设备存一片；模型宽度在左边是求和维，在右边是输出维",
                  [(OR, "一片权重"), (RD, "要加起来"), (GR, "只拼起来")])
    PH, PW = 312, 690
    for side in (0, 1):
        px = side * (W - PW)
        py = f.panel(px, y0, PW, PH, ("往里投：模型宽度是求和维（q_a、gate、up……）" if side == 0 else
                                      "往外投：模型宽度是输出维（o_proj、down）"), RD if side == 0 else GR)
        # 激活 x：每个设备上是完整的
        xx, xy = px + 24, py + 62
        xw = 250 if side == 0 else 170
        f.t(xx, xy - 12, "激活 x（每个设备上完整）", INK, True, 13)
        f.box(xx, xy, xw, 40, "#e8f0fe", BL, 4)
        if side == 0:
            for k in range(1, 8):
                f.line(xx + xw * k / 8.0, xy + 2, xx + xw * k / 8.0, xy + 38, BL, 1, dash="3,3", arrow=False)
            f.t(xx + xw / 2.0, xy + 60, "模型宽度：按片对上权重", BL, True, 13, "middle")
        else:
            f.t(xx + xw / 2.0, xy + 60, "另一维，不切", BL, True, 13, "middle")
        f.t(xx + xw + 22, xy + 28, "×", INK, True, 22, "middle", big=True)
        wx = xx + xw + 44
        if side == 0:                                   # 按行切 8 片（行 ＝ 求和维）
            ww, wh = 70, 176
            for k in range(8):
                f.box(wx, xy - 18 + k * wh / 8.0, ww, wh / 8.0 - 3, "#fdecdc", OR, 2)
            f.t(wx + ww / 2.0, xy - 26, "权重：按行切 8 片", OR, True, 13, "middle")
            ox = wx + ww + 40
            f.line(wx + ww + 8, xy + 60, ox - 6, xy + 60, GY2, 1.4)
            for k in range(3):                          # 8 份同样大小的部分和，叠着画
                f.box(ox + k * 10, xy + 10 + k * 14, 150, 52, "#fff", RD, 4)
            f.t(ox + 95, xy + 72, "8 份部分和", RD, True, 14, "middle")
            f.t(ox + 95, xy + 110, "形状都是整个 y", GY, size=12, anchor="middle")
            f.t(px + 24, py + PH - 88, "y ＝ x₁W₁ ＋ x₂W₂ ＋ … ＋ x₈W₈", INK, True, 15, mono=True)
            f.t(px + 24, py + PH - 56, "8 份必须加起来：在什么精度里加，决定多出多少误差", RD, True, 14)
        else:                                           # 按列切 8 片（列 ＝ 输出维）
            ww, wh = 200, 96
            for k in range(8):
                f.box(wx + k * ww / 8.0, xy - 18, ww / 8.0 - 3, wh, "#fdecdc", OR, 2)
            f.t(wx + ww / 2.0, xy - 26, "权重：按列切 8 片", OR, True, 13, "middle")
            ox = wx + ww + 30
            f.line(wx + ww + 6, xy + 30, ox - 4, xy + 30, GY2, 1.4)
            for k in range(8):
                f.box(ox + k * 17, xy + 4, 15, 52, "#e6f4ea", GR, 2)
            f.t(ox + 68, xy + 80, "8 段拼起来", GR, True, 14, "middle")
            f.t(px + 24, py + PH - 88, "y ＝ [ x·W₁ ｜ x·W₂ ｜ … ｜ x·W₈ ]", INK, True, 15, mono=True)
            f.t(px + 24, py + PH - 56, "每个输出元素仍是一次完整的求和、只舍一次，跟 GPU 一样", GR, True, 14)
        f._pan = None
    # 下：哪些投影中招（实测）
    sy = y0 + PH + 18
    SH = 108
    py = f.panel(0, sy, W, SH, "第三次迁移，单算子回放（8 个设备）：TPU 离 FP64 ÷ GPU 离 FP64（R）", INK, tag="实测")
    x = 20
    for name in N.M3_HIT:
        w = wpx(name, 13) * MONO_K + 22
        f.box(x, py + 16, w, 30, "#fce8e6", RD, 5)
        f.t(x + w / 2.0, py + 36, name, RD, True, 13, "middle")
        x += w + 8
    f.t(20, py + 68, "往里投的 7 种：R %s（≈ √2，像多舍了一次）" % N.M3_PROJ_R_T, RD, True, 14)
    x2 = 980
    for name in N.M3_MISS:
        w = wpx(name, 13) * MONO_K + 22
        f.box(x2, py + 16, w, 30, "#e6f4ea", GR, 5)
        f.t(x2 + w / 2.0, py + 36, name, GR, True, 13, "middle")
        x2 += w + 8
    assert x < 980 - 10 and x2 < W - 6, (x, x2)
    f.t(980, py + 68, "往外投的 3 种：R %.2f" % N.M3_OUTPROJ_R, GR, True, 14)
    f._pan = None
    yb = f.band(sy + SH + 20, "ok", "判据：FSDP 切的那一维，是不是这次矩阵乘的求和维", [
        "是：每片只给出一份部分和，要加；不是：每片给出完整的几列，只拼。中招的正好是前一种。",
        "求和本来就要做；不拆开时它在一次矩阵乘里用 FP32 加完，只舍一次。问题出在编译器把这一次拆成了几次（下一张图）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：对齐记录里第三次迁移的单算子回放（8 个设备）：往里投的那批投影 TPU 对 FP64 %.2e、GPU %.2e，R %s（结果文件原值 1.41–1.42）；o_proj、down 是 %.2f。"
               % (N.M3_PROJ_TPU, N.M3_PROJ_GPU, N.M3_PROJ_R_T, N.M3_OUTPROJ_R),
               "📌 每个设备拿到哪一维的哪一片，取自 8 个设备编译产物里的权重片形状：往里投的一片是模型宽度的 1/8（2816 ÷ 8 ＝ 352）乘整个输出维；o_proj 的一片是整个输入维乘模型宽度的 1/8。",
               "📌 q_b、kv_b 的输入不是模型宽度，是 MLA 的低秩中间维，这一维也被 FSDP 切了，同样是切在求和维上。路由打分那次矩阵乘也切在求和维上，但生产本来就让它输出 FP32，不中招。路由专家走另一条路（权重收齐后在本卡做分组矩阵乘），不受影响。",
               "注：左右两边的激活、权重、输出画的是形状关系，不按比例。")
    f.save("fig8-inout.svg", yb + 14)


def fig_windowed():
    f = Fig(W, "左边三行是同一个往里投的投影（第三次迁移的 q 低秩下投影，求和维 2816）在三种情况下怎么算完。"
               "第一行，4 个设备：编译器先把 8 片权重收齐，再做一次 2816 宽的矩阵乘，FP32 累加到底，最后舍一次，跟 GPU 一样。"
               "第二行，8 个设备、生产原样：编译器改成边传边乘，8 片排成 2 轮，每轮两次 704 宽的矩阵乘，每次输出舍成 BF16，再加进 BF16 累加器又舍一次，一共舍 6 次。"
               "第三行，8 个设备、打开投影 FP32 累加：结构不变，中间全部留在 FP32，只在最后舍一次。"
               "右边：离线编译训练步，数部分和累加器是 BF16 的窗口化循环：4 个设备 0 个，8 个、16 个设备各 20 个（v7x 16 个设备也是 20 个），打开开关后全部变成 FP32。")
    y0 = f.header("设备一到 8 个，编译器把一次矩阵乘拆成几段　——　<tspan font-weight=\"700\">一次求和舍 6 次</tspan>",
                  "同一个投影、同一种 FSDP 切法；变的只是编译器选「先收齐再乘」还是「边传边乘」。红块 ＝ 一次舍到 BF16",
                  [(BL, "矩阵乘"), (PU, "累加器"), (RD, "舍到 BF16"), (GR, "FP32，最后只舍一次")])
    PH, LW = 430, 930
    py = f.panel(0, y0, LW, PH, "q 低秩下投影的一次前向（编译产物显示）", BL)
    X0, BW, RW, GAP = 196, 76, 16, 10
    STEP = BW + RW + GAP
    rows = [("4 个设备", "先收齐，再乘", "a"), ("8 个设备", "生产原样：边传边乘", "b"), ("8 个设备", "打开投影 FP32 累加", "c")]
    for r, (lab, sub, kind) in enumerate(rows):
        y = py + 56 + r * 124
        f.t(20, y + 22, lab, INK, True, 16)
        f.t(20, y + 44, sub, {"a": GY, "b": RD, "c": GR}[kind], True, 13)
        if kind == "a":
            f.box(X0, y, 150, 50, "#fdecdc", OR, 5)
            f.t(X0 + 75, y + 22, "收齐 8 片权重", OR, True, 13, "middle")
            f.t(X0 + 75, y + 40, "（整块 all-gather）", GY, size=11, anchor="middle")
            f.line(X0 + 154, y + 25, X0 + 182, y + 25, GY2, 1.3)
            f.box(X0 + 186, y, 4 * STEP + 20, 50, "#e8f0fe", BL, 5)
            f.t(X0 + 186 + (4 * STEP + 20) / 2.0, y + 22, "一次矩阵乘，2816 宽", BL, True, 14, "middle")
            f.t(X0 + 186 + (4 * STEP + 20) / 2.0, y + 40, "FP32 累加到底", GY, size=11, anchor="middle")
            xr = X0 + 186 + 4 * STEP + 26
            f.box(xr, y + 6, RW, 38, "#fce8e6", RD, 3)
            f.t(xr + RW / 2.0, y + 66, "①", RD, True, 13, "middle")
            nr = 1
        else:
            n = 0
            x = X0
            for rnd in range(N.M3_ROUNDS):
                f.t(x + (3 * STEP - GAP) / 2.0, y - 10, "第 %d 轮（片 %d–%d）" % (rnd + 1, 4 * rnd + 1, 4 * rnd + 4), GY, True, 12, "middle")
                for j in range(N.M3_MM_PER_ROUND + 1):
                    acc = (j == N.M3_MM_PER_ROUND)
                    if acc:
                        f.box(x, y, BW, 50, "#f3e8fd", PU, 5)
                        f.t(x + BW / 2.0, y + 22, "累加器", PU, True, 13, "middle")
                        f.t(x + BW / 2.0, y + 40, "＋＝ 两份", GY, size=11, anchor="middle")
                    else:
                        a = 4 * rnd + 2 * j + 1
                        f.box(x, y, BW, 50, "#e8f0fe", BL, 5)
                        f.t(x + BW / 2.0, y + 22, "片 %d＋%d" % (a, a + 1), BL, True, 13, "middle")
                        f.t(x + BW / 2.0, y + 40, "704 宽", GY, size=11, anchor="middle")
                    last = (rnd == N.M3_ROUNDS - 1 and acc)
                    if kind == "b" or last:
                        n += 1
                        f.box(x + BW + 2, y + 6, RW, 38, "#fce8e6", RD, 3)
                        f.t(x + BW + 2 + RW / 2.0, y + 66, "①②③④⑤⑥"[n - 1] if kind == "b" else "①", RD, True, 13, "middle")
                    elif kind == "c":
                        f.box(x + BW + 2, y + 6, RW, 38, "#e6f4ea", GR, 3)
                    x += STEP
            nr = n
            xr = x - GAP - RW
        f.t(LW - 20, y + 30, "舍 %d 次" % nr, RD if nr > 1 else GR, True, 16, "end")
    assert X0 + 6 * STEP < LW - 90
    f._pan = None
    px = LW + 20
    py2 = f.panel(px, y0, W - px, PH, "离线编译训练步：累加器是 BF16 的循环", PU, tag="实测")
    BX0, BMAX = px + 150, 230
    for i, (chip, dev, loops, bf) in enumerate(N.M3_SCAN):
        y = py2 + 40 + i * 62
        f.t(px + 18, y + 18, "%s · %d 个设备" % (chip, dev), INK, True, 14)
        L = BMAX * bf / 20.0
        if bf:
            f.box(BX0, y + 2, L, 26, RD, RD, 3)
        f.t(BX0 + L + 8 if bf else BX0, y + 21, str(bf) + ("" if bf else "　（不拆）"), RD if bf else GY, True, 15)
    y = py2 + 40 + 4 * 62
    f.t(px + 18, y + 18, "打开开关后", INK, True, 14)
    f.t(BX0, y + 21, "0　（全部变 FP32）", GR, True, 15)
    f.t(px + 18, py2 + PH - 80, "单算子 R：%s → %s" % (N.M3_PROJ_R_T, N.M3_PROJ_R_AFTER_T), INK, True, 15)
    f.t(px + 18, py2 + PH - 52, "混元 3 的 qkv：%.3f → %.3f" % (N.R_QKV_BEFORE, N.R_QKV_AFTER), GY, True, 13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "在 4 个设备的小切片上验精度，这件事完全看不到", [
        "切法没变：4 个、8 个设备上权重都按 FSDP 切满。变的只是编译器拆不拆；拆了，每个往里投的投影都多舍几次。",
        "矩阵单元里的 FP32 累加只管一次矩阵乘之内；拆成几段，每段的结果都按程序声明的 BF16 写出来，累加器也是 BF16。",
        "修法是让这批矩阵乘声明 FP32 输出：累加器跟着变成 FP32，循环结束只舍一次。它不改编译器，只改交给编译器的输出类型。",
    ])
    yb = f.src(yb + 10,
               "📌 编译产物显示（第三次迁移，8 个设备，运行库 0.0.46）：q 低秩下投影的循环体里，8 片权重沿两个方向的环传过来，排成 2 轮；每轮两次 704（＝ 2 × 352）宽的矩阵乘，输出是 BF16；和 BF16 累加器在一颗融合里转成 FP32 相加，再舍回 BF16。v7x 上这一步直接在 BF16 里加，舍入次数相同。",
               "📌 实验确认：同一份生产配置在 CPU 上为不同拓扑离线编译训练步。v5p 4 个设备没有窗口化循环，权重整块收齐；8 个、16 个设备各 64 个窗口化循环，其中带部分和相加、累加器 BF16 的 20 个（正好是上一张图里中招的那几种投影的前向和重算），其余是按输出维写各自的列，或权重梯度一侧（本来就是 FP32）。打开投影 FP32 累加后，这 20 个全部变成 FP32。",
               "📌 大小不是门槛：8 个设备上，连最小的路由打分矩阵也拆了；4 个设备上，连最大的 gate、up 也整块收。拆不拆看的是切法和设备数。为什么设备少不拆是推断：编译器按快慢选，设备少时环上首尾那几次传递的固定开销占比大，拆了省不下时间。更大的规模没有编过，按「会拆」准备。",
               "📌 混元 3（v5p 8 个设备）是同一个机制：qkv 投影 R %.3f → %.3f，配置里有 41 个这样的循环。" % (N.R_QKV_BEFORE, N.R_QKV_AFTER),
               "⚠️ 推断：舍了 6 次 R 却只有约 √2，是因为前几次舍的是小的部分和（量级约为最终结果的一半），误差也小；再加上两边共有的误差（输入本来就是 BF16）。")
    f.save("fig8-windowed.svg", yb + 14)


def fig_lmhead():
    L = N.M3_LMH
    f = Fig(W, "左：输出头那次矩阵乘的输入里，激活本来就是 BF16，权重是 FP32 主权重。TPU 默认精度档把 FP32 输入舍成 BF16、只算一遍。"
               "第 0 步主权重就在 BF16 的格点上，舍了不变；训练一步以后主权重离开格点，默认档把它舍回最近的格点，差出一截。"
               "右：单独回放这一次矩阵乘，logits 对 FP64 的相对误差：权重在格点上默认档 %.1e；离格以后默认档 %.1e 到 %.1e，"
               "high（3 遍）%.1e，highest（6 遍）%.1e，主机上真 FP32 %.1e。下：前 300 步主 loss 的 r，默认 %.2f、high %.2f、highest %.2f；步时 high ＋5.7%%，highest ＋12.3%%。"
               % (L[("on", "default")][0], L[("off1", "default")][0], L[("off10", "default")][0], L[("off10", "high")][0],
                  L[("off10", "highest")][0], N.M3_LMH_HOSTFP32[1], N.M3_LMH_LOSS4K["default"][1], N.M3_LMH_LOSS4K["high"][1], N.M3_LMH_LOSS4K["highest"][1]))
    y0 = f.header("默认精度档舍的不是激活，是离开格点的主权重　——　<tspan font-weight=\"700\">第 1 步看不见，第 2 步起差几千倍</tspan>",
                  "第三次迁移的输出头：logits ＝ x · W，x 是 BF16，W 是 FP32 主权重；两边升成 FP32 交给矩阵单元，精度档没单独设",
                  [(GR, "主权重（FP32）"), (RD, "默认档舍回格点"), (GY, "BF16 格点")])
    PH, LW = 330, 600
    py = f.panel(0, y0, LW, PH, "为什么第 1 步看不出来（示意）", INK)
    X0, X1 = 150, LW - 40
    ticks = [X0 + (X1 - X0) * k / 4.0 for k in range(5)]
    rows = [("第 0 步", "主权重在格点上", ticks[2], "舍了不变"),
            ("训练 1 步后", "主权重离开格点", ticks[2] + 0.42 * (ticks[3] - ticks[2]), "默认档舍回格点，差出这一截"),
            ("用 high", "拆成两段 BF16 算 3 遍", ticks[2] + 0.42 * (ticks[3] - ticks[2]), "剩一点点")]
    for i, (a, b, xw, note) in enumerate(rows):
        y = py + 56 + i * 90
        f.t(20, y + 4, a, INK, True, 15)
        f.t(20, y + 26, b, GY, size=12)
        f.line(X0 - 10, y, X1 + 10, y, LINE2, 1.2, arrow=False)
        for x in ticks:
            f.line(x, y - 9, x, y + 9, GY, 1.4, arrow=False)
        f.box(xw - 7, y - 7, 14, 14, GR, GR, 7)
        if i == 1:
            f.line(xw - 9, y + 20, ticks[2] + 4, y + 20, RD, 2.2)
            f.t((xw + ticks[2]) / 2.0, y + 44, note, RD, True, 12, "middle")
        elif i == 2:
            f.box(xw - 22, y + 14, 14, 6, RD, RD, 2)
            f.t(xw + 4, y + 44, note, GR, True, 12, "middle")
        else:
            f.t(xw, y + 40, note, GY, True, 12, "middle")
    f._pan = None
    px = LW + 20
    py2 = f.panel(px, y0, W - px, PH, "logits 对 FP64 的相对误差（单独回放这一次矩阵乘，实测）", RD, tag="对数刻度")
    BX0, BX1 = px + 250, W - 70
    lo, hi = -7.3, -2.3

    def X(v):
        return BX0 + (math.log10(v) - lo) / (hi - lo) * (BX1 - BX0)
    for e in range(-7, -2):
        f.line(X(10 ** e), py2 + 16, X(10 ** e), py2 + PH - 56, LINE2, 1, arrow=False)
        f.t(X(10 ** e), py2 + PH - 38, "1e%d" % e, GY, size=12, anchor="middle")
    bars = [("默认 · 格点上", L[("on", "default")][0], GY), ("默认 · 离格 1 步", L[("off1", "default")][0], RD),
            ("默认 · 离格 10 步", L[("off10", "default")][0], RD), ("high · 离格 10 步", L[("off10", "high")][0], OR),
            ("highest · 离格 10 步", L[("off10", "highest")][0], GR), ("主机真 FP32 · 离格", N.M3_LMH_HOSTFP32[1], GY)]
    for i, (lab, v, c) in enumerate(bars):
        y = py2 + 24 + i * 38
        f.t(px + 18, y + 17, lab, INK, True, 13)
        f.box(BX0, y + 4, max(3, X(v) - BX0), 18, c, c, 3)
        f.t(X(v) + 8, y + 18, "%.1e" % v, c, True, 13)
    f._pan = None
    sy = y0 + PH + 18
    SH = 120
    py3 = f.panel(0, sy, W, SH, "三档怎么选（单份运行；前 300 步主 loss 的 r ／ 步时）", INK)
    cards = [("默认（1 遍，生产原样）", "%.2f　不过" % N.M3_LMH_LOSS4K["default"][1], "步时基准", RD),
             ("high（3 遍）", "%.2f" % N.M3_LMH_LOSS4K["high"][1], "＋%.1f%%" % (N.M3_LMH_COST["high"] * 100), GR),
             ("highest（6 遍）", "%.2f" % N.M3_LMH_LOSS4K["highest"][1], "＋%.1f%%" % (N.M3_LMH_COST["highest"] * 100), BL)]
    cw = (W - 60) / 3.0
    for i, (a, b, c, col) in enumerate(cards):
        x = 20 + i * (cw + 10)
        f.box(x, py3 + 14, cw, 70, "#fff", col, 6)
        f.t(x + 16, py3 + 38, a, col, True, 14)
        f.t(x + 16, py3 + 68, "前 300 步 " + b + "　·　" + c, INK, True, 14)
    f._pan = None
    yb = f.band(sy + SH + 20, "ok", "推荐 high：曲线效果跟 highest 一样，代价约一半", [
        "第 1 步主权重还在格点上，三档前向一样；第 2 步起才差开，所以第 1 步逐张量看不出它（超 2N 的张量 91 → 91）。",
        "这一次矩阵乘的输出是整个词表，是整网最宽的一次，多算几遍就贵：high 多 2 遍，highest 多 5 遍。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：对齐记录第三次迁移，单独回放输出头这一次矩阵乘，输入取 GPU 第 1 步，对 CPU FP64。离格 1 步、10 步是把权重换成训练 1 步、10 步后的值。对 x 的梯度跟前向一样大（默认档离格 %.1e），乘的是同一个 W；对 W 的梯度只有一处小误差（默认档把 FP32 的 logits 梯度舍成 BF16，%.1e）。" % (L[("off10", "default")][1], L[("off10", "default")][2]),
               "📌 「约 5000 倍」：默认档离格 10 步 %.1e ÷ 主机真 FP32 离格 %.1e ≈ 5100（本课复算，对齐记录同口径）。" % (L[("off10", "default")][0], N.M3_LMH_HOSTFP32[1]),
               "📌 曲线（4K，单份运行）：1000 步主 loss 默认 %.2f（待定）、high %.2f、highest %.2f；32K 上 high %.2f、highest %.2f。前 300 步 high %.2f、highest %.2f 只差在单份运行的噪声里，不按小数点排先后。步时 %.3f ／ %.3f ／ %.3f s。"
               % (N.M3_LMH_LOSS4K["default"][0], N.M3_LMH_LOSS4K["high"][0], N.M3_LMH_LOSS4K["highest"][0], N.M3_LMH_LOSS32K["high"][0], N.M3_LMH_LOSS32K["highest"][0],
                  N.M3_LMH_LOSS4K["high"][1], N.M3_LMH_LOSS4K["highest"][1], N.M3_LMH_STEP4K["default"], N.M3_LMH_STEP4K["high"], N.M3_LMH_STEP4K["highest"]),
               "📌 第 1 步的梯度其实变了：默认档对 highest，116 个张量相对差中位数 %.1e，比各张量对 FP64 的误差（约 1e-2）小两个量级，所以超 2N 的个数不动。" % N.M3_LMH_STEP1_MED,
               "注：左格的格点间距和权重离格多远是示意。")
    f.save("fig8-lmhead.svg", yb + 14)


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
    f.save("fig8-excess.svg", yb + 14)


def fig_normscale():
    f = Fig(W, "左：RMSNorm 乘的那组缩放系数 scale，初值 1.0。GPU 乘的是舍成 BF16 的副本，头一两百步一直是 1.0；TPU 在 32K 的程序里这次舍入被编译器删了，乘的是 FP32 原值，"
               "每步偏开约 6.4e-6，第 50、100、150 步偏离中位数 3.2e-4、6.5e-4、9.5e-4；第 200 步起才开始有元素跨过 BF16 的舍入边界。"
               "右：32K 上在第 2 级之上逐个加开关，主 loss 偏差的平均（单位是 GPU 8 份的份间标准差 σ）：对照约 −24σ，加投影 FP32 累加、精确超越函数、精确 Adam 都不动，"
               "四处舍入一起回到 −0.6σ；再拆，只加 RMSNorm scale 舍入就回到 −0.2σ。")
    y0 = f.header("同一行代码，4K 的程序里舍，32K 的程序里被删　——　<tspan font-weight=\"700\">编译器删不删，跟着切法变</tspan>",
                  "第三次迁移：RMSNorm 先把 FP32 的 scale 舍成 BF16 再乘（照 GPU），用「转 BF16 再转回」写。离线编译：4K 程序 8 个 norm 被删 0 个，32K 程序 8 个全删",
                  [(GY, "GPU：乘 BF16 的 scale"), (BL, "TPU 32K：乘 FP32 的 scale"), (RD, "主 loss 偏差")])
    PH, LW = 360, 640
    py = f.panel(0, y0, LW, PH, "为什么到几十步以后才显出来：scale 离 1.0 多远", INK, tag="实测 ＋ 外推虚线")
    X0, X1, Y0, Y1 = 90, LW - 30, py + PH - 70, py + 40

    def XY(step, d):
        return X0 + step / 250.0 * (X1 - X0), Y0 - d / 1.8e-3 * (Y0 - Y1)
    for st in (0, 50, 100, 150, 200, 250):
        x, _ = XY(st, 0)
        f.line(x, Y0, x, Y0 + 6, GY, 1, arrow=False)
        f.t(x, Y0 + 22, str(st), GY, size=12, anchor="middle")
    f.t((X0 + X1) / 2.0, Y0 + 42, "训练步", GY, size=12, anchor="middle")
    for d in (0, 5e-4, 1e-3, 1.5e-3):
        _, y = XY(0, d)
        f.line(X0, y, X1, y, LINE2, 1, arrow=False)
        f.t(X0 - 8, y + 4, "0" if d == 0 else "%.1e" % d, GY, size=11, anchor="end")
    xa, ya = XY(0, 0)
    xb, yb_ = XY(250, 0)
    f.line(xa, ya - 2, xb, yb_ - 2, GY, 3.2, arrow=False)
    f.t(XY(120, 0)[0], ya - 10, "GPU：BF16 副本一直是 1.0", GY, True, 13, "middle")
    pts = [(0, 0.0)] + list(N.M3_SCALE_TRACK)
    for (s1, d1), (s2, d2) in zip(pts, pts[1:]):
        x1, y1 = XY(s1, d1)
        x2, y2 = XY(s2, d2)
        f.line(x1, y1, x2, y2, BL, 2.6, arrow=False)
    for st, d in N.M3_SCALE_TRACK:
        x, y = XY(st, d)
        f.box(x - 5, y - 5, 10, 10, BL, BL, 5)
        f.t(x - 8, y - 10, "%.1e" % d, BL, True, 12, "end")
    x1, y1 = XY(150, N.M3_SCALE_TRACK[-1][1])
    x2, y2 = XY(250, N.M3_SCALE_TRACK[-1][1] + 100 * N.M3_SCALE_SLOPE)
    f.line(x1, y1, x2, y2, BL, 1.6, dash="5,4", arrow=False)
    f.t(XY(4, 0)[0], XY(0, 1.45e-3)[1], "TPU：乘 FP32 原值，每步偏开约 6.4e-6", BL, True, 13)
    xc, _ = XY(200, 0)
    f.line(xc, Y1 + 30, xc, Y0, OR, 1.4, dash="3,3", arrow=False)
    f.t(xc + 6, XY(0, 3.5e-4)[1], "第 200 步起", OR, True, 12)
    f.t(xc + 6, XY(0, 3.5e-4)[1] + 18, "GPU 才有元素跳格", OR, True, 12)
    f._pan = None
    px = LW + 20
    py2 = f.panel(px, y0, W - px, PH, "二分：32K，第 2 级之上逐个加开关（主 loss 偏差，σ）", RD, tag="实测")
    BX0, BX1 = px + 250, W - 40
    lo, hi = -34.0, 4.0

    def X(v):
        return BX0 + (v - lo) / (hi - lo) * (BX1 - BX0)
    f.box(X(-3), py2 + 10, X(3) - X(-3), PH - 76, "#e6f4ea", "none", 0)
    f.t(X(0), py2 + PH - 52, "±3σ", GR, True, 12, "middle")
    for v in (-30, -20, -10, 0):
        f.line(X(v), py2 + 10, X(v), py2 + PH - 66, LINE2, 1, arrow=False)
        f.t(X(v), py2 + PH - 36, "%d" % v if v else "0", GY, size=12, anchor="middle")
    rows = [("第 51–300 步", None)] + N.M3_BISECT_300 + [("第 51–120 步（120 步运行）", None)] + N.M3_BISECT_120
    y = py2 + 14
    for lab, v in rows:
        if v is None:
            f.t(px + 16, y + 14, lab, GY, True, 12)
            y += 22
            continue
        f.t(px + 26, y + 14, lab, INK, True, 12.5)
        c = RD if v < -3 else GR
        f.box(min(X(v), X(0)), y + 2, abs(X(v) - X(0)) or 3, 16, c, c, 2)
        vt = ("%.1f" % v).replace("-", "−")
        vt = {"另外三处之一": "≤ " + vt}.get(lab, vt)
        if (lab, v) == N.M3_BISECT_120[0]:
            vt = "约 −30"
        f.t(X(v) + 8 if v < -3 else X(0) + 8, y + 15, vt, "#fff" if v < -3 else GR, True, 12)
        y += 26
    assert y < py2 + PH - 60, y
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "在一种切法上验过「这个开关不用开」，换了切法要重验", [
        "删不删看这对转换落不落在同一颗融合里，而那取决于切法：FSDP 8 时 scale 要先在 8 个设备间收齐，BF16 值真被写出来；FSDP 2 时整个往返落在本卡一颗融合里（推断）。",
        "第 1 步看不见：scale 初值 1.0 在 BF16 格点上，两边相同；这个开关的反向是恒等，不碰梯度。要等 FP32 原值偏开、GPU 那边还停在 1.0 才显出来。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：对齐记录第三次迁移，32K（16 个设备，上下文并行 8 × FSDP 2）。偏差从第 20 步起系统性偏低，第 100 步约 %dσ，第 500 步前后翻正；MTP 同向约 −40σ。离线编译扫描：32K 程序里 RMSNorm 乘之前那次舍入 8 个 norm 全部被删、保住 0 个；4K 程序没有被删的记录。" % N.M3_DRIFT_PEAK,
               "📌 二分取的是阶段②第 2 级之上加开关的诊断运行；第 51–120 步那组是 120 步的短跑，对照约 −30σ，另外三处单加都还在 −30.8σ 以下。scale 偏离每 50 步读一次（16 个 norm scale 的中位数）；舍成 BF16 后不等于 1.0 的元素比例第 200 步中位数 2.9%%、最多 30%%，第 250 步 7%% ／ 55%%。",
               "📌 4K 负对照：只开 RMSNorm scale 舍入，300 步除墙钟外 13800 个逐步量跟全关逐位相同，第 1 步梯度文件逐字节相同 —— 4K 程序里这次舍入本来就在。修法步时 32K %.3f → %.3f s（各一份）。" % N.M3_NORM_STEP,
               "⚠️ 推断：为什么切法一变就删（没有查到具体的融合决定）；虚线是按每步 6.4e-6 外推。")
    f.save("fig8-normscale.svg", yb + 14)


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
               "📌 乘法那一格要看编译后的样子：一处 HIGHEST 写的多路残差四路加权和，编译器已改写成 FP32 逐元素乘加、不上矩阵单元，跟 GPU 只差加法顺序（TPU 配对相加、GPU 依次相加），归顺序那一格；改成依次相加后第 1 步该处梯度对 CPU FP64 R ＞ 1.2 的张量 %s，后一代配置没复现。"
               % N.MIX_OVER_T,
               "注：线高只分 16 位、32 位两档，HIGHEST 这类「多遍 BF16 拼近似 FP32」也画在 32 位，所以乘法那一格要比字、不只比高度；HIGHEST 是不是多遍 BF16，看编译后它还是不是矩阵乘；「舍在哪」一列不画线，红色刻痕一道 ＝ 舍一次到 16 位，只在它本身是根子时画（第 5 行：TPU 默认归一化结果、缩放各舍一次，GPU 算完只舍一次）。第 3 行累加那一格：一轮之内照常 FP32 累加，舍的是轮与轮之间的部分和（§5.4 的推断）。")
    f.save("fig8-boundary.svg", yb + 14)


def fig_ulp2():
    f = Fig(W, "四个超越函数，各四根柱子：v5p 默认、v5p 开精确超越函数、v7x 默认、v7x 开精确超越函数，柱高是逐元素对 FP64 正确舍入值的最大误差（格），对数刻度。"
               "exp 105、1、64、64；log 62、3、2、2；sigmoid 83、3、30、30；silu 86、4、29、30。v5p 开了以后降到个位数；v7x 开不开一样。")
    y0 = f.header("精确超越函数的开关：v5p 上管用，v7x（运行库 0.0.46）上不起作用",
                  "第三次迁移，同一份输入、同一种算法；柱高 ＝ 最大误差（格），对数刻度",
                  [(RD, "v5p 默认"), (GR, "v5p 开"), (OR, "v7x 默认"), (BL, "v7x 开")])
    PH = 330
    py = f.panel(0, y0, W, PH, "逐元素最大误差（实测）", INK)
    Y0, Y1 = py + PH - 60, py + 30

    def Y(u):
        return Y0 - math.log10(max(u, 1)) / math.log10(200) * (Y0 - Y1)
    for u in (1, 10, 100):
        f.line(80, Y(u), W - 30, Y(u), LINE2, 1, arrow=False)
        f.t(70, Y(u) + 4, str(u), GY, size=12, anchor="end")
    cols = [RD, GR, OR, BL]
    gw = (W - 140) / 4.0
    for i, (fn, vals) in enumerate(N.M3_ULP.items()):
        gx = 110 + i * gw
        for k, v in enumerate(vals):
            x = gx + k * 62
            h = max(3, Y0 - Y(v))
            f.box(x, Y0 - h, 46, h, cols[k], cols[k], 3)
            f.t(x + 23, Y0 - h - 8, str(v), cols[k], True, 13, "middle")
        f.t(gx + 2 * 62 - 8, Y0 + 30, fn, INK, True, 15, "middle")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "换芯片要重验开关：同一个开关，在一代芯片上管用，在另一代上可能是空操作", [
        "v5p 上开了以后，整段挪到向量单元用多项式算（编译产物显示），步时 ＋0.8%。只给 JAX 设 highest 不够：sigmoid、silu 不动。",
        "v7x 默认就比 v5p 准（log 2 格），但 exp 最大仍差 64 格；开关开不开，编出的指令一样。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：对齐记录第三次迁移，逐元素对 FP64 正确舍入值。v5p 只给 JAX 设 precision=highest：exp %d、log %d、sigmoid %d、silu %d。v7x 默认的平均误差：exp 11.8、log 0.65、sigmoid 2.4、silu 2.3 格。"
               % tuple(N.M3_ULP_JAXONLY[k] for k in ("exp", "log", "sigmoid", "silu")),
               "📌 单算子回放（第三次迁移 v5p，生产原样）：post 门 8.68 倍、路由权重 4.36、主交叉熵 2.97、MTP 交叉熵 3.97；开了以后 0.96、1.15、1.14、1.25。曲线上只有 MTP 一项看得出（1.06 → 0.99）。",
               "📌 编译产物显示：v5p 默认的 exp、log 交给硬件的超越函数单元；精确档改到向量单元上用多项式，每个向量寄存器的指令 3 条 → 43–45 条。v7x 开不开编出的指令相同。")
    f.save("fig8-ulp2.svg", yb + 14)


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
    f.save("fig8-captain.svg", yb + 14)


def fig_elide():
    f = Fig(W, "一条横轴：1000 步主 loss 的带符号均值，TPU 对 GPU 8 份的平均，单位 1e-6。灰色一段是 GPU 自己两两最大的那一对，正负 %s。"
               "第一行：一处四路加权和照 GPU 改加法顺序之前，TPU 在 ＋1.1，落在灰段里。第二行：改了以后五次 TPU 运行，其中三次只改执行参数，全在右边，%s，同一个方向。"
               "五次里四次出了灰段。第三行：在两处强制写一次「舍到 BF16」以后，约 ＋3.5，回到灰段里。"
               % (N.ELIDE_GPU_MAX_T, N.ELIDE_RANGE_T))
    y0 = f.header("编译器省掉一次舍入，1000 步 loss 五次同号偏高、四次出了噪声",
                  "横轴 ＝ 1000 步主 loss 的带符号均值（TPU 对 GPU 8 份平均，×1e-6）；灰段 ＝ GPU 自己两两最大的差；一个点 ＝ 一次 TPU 运行",
                  [(GY, "GPU 自己的噪声"), (BL, "TPU 运行"), (GR, "强制舍入以后")])
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
    rows = [("改顺序之前", [N.ELIDE_BEFORE], BL, True),
            ("改顺序以后（5 次）", list(N.ELIDE_RUNS), BL, True),
            ("两处强制舍入以后", [N.ELIDE_AFTER], GR, False)]
    for i, (lab, vals, col, filled) in enumerate(rows):
        y = py + 70 + i * 66
        f.t(24, y + 5, lab, INK, True, 15)
        for v in vals:
            f.p.append('<circle cx="%.1f" cy="%d" r="8" fill="%s" stroke="%s" stroke-width="2"/>' % (X(v), y, col if filled else "#fff", col))
    f.t(X(N.ELIDE_GPU_MAX) + 8, yt + 18, "GPU 自己最大 ±%s" % N.ELIDE_GPU_MAX_T, GY, size=12)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "GPU 每次都把中间值存成 BF16；TPU 上这个值后面两处用，归一化输出那处舍了，算均方那处没舍：值更精确，跟 GPU 反而对不上", [
        "要和 GPU 舍在同一处，就在那一处显式写「舍到 BF16」（JAX 的 lax.reduce_precision），编译器不会删。",
        "代价：显式舍入是为了对齐，生产里关掉同样准（推断）；剩下约 ＋3.5e-6 还没查清。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移）：五次 TPU 运行对 GPU 8 份均值 %s，GPU 两两 28 对的带符号均值最大 %s；强制舍入后判定 r %s（对 GPU 基线，带符号均值），对 GPU 均值约 ＋3.5e-6（实测）。TPU 开、关 base2 各 4 份共八份，主 loss 全部比 GPU 偏高、平均约 4.3e-6（第 1 步还没有）；同口径（每份 GPU 对其余各份）GPU 只在 ±2.6e-6 以内，灰段是两两最大差，口径更宽。"
               % (N.ELIDE_RANGE_T, N.ELIDE_GPU_MAX_T, N.ELIDE_R_T),
               "📌 编译产物：优化后的 HLO 里「转 BF16 再转回」都还在，差别在后端 —— 算均方那颗 kernel 用的是没舍的 FP32，归一化输出那颗舍了（在一个最小复现里数过后端指令；长跑偏高归到这一处是推断）。五次里有三次只改执行参数（注意力分块、专家并行、MoE 实现），同号偏高，说明不是轨迹偶然分开。")
    f.save("fig8-elide.svg", yb + 14)


def fig_trans():
    f = Fig(W, "左格：四个超越函数，横条长度是对正确舍入的最大误差，按格（ulp）计，对数刻度。"
               "TPU 默认：%s。TPU 高精度写法：%s。GPU：exp 2、log 1、sigmoid 和 silu 最大 3.4（实测）。"
               "右格：高精度的 exp 怎么用软件算：先把 x 拆成 k 个 ln2 加一个小余数 s，余数上用多项式算 e 的 s 次方，再乘 2 的 k 次方，只改指数位、精确；"
               "求倒数用牛顿修正，每修一次误差平方一次。" % (N.TRANS_T, N.TRANS_HI_T))
    y0 = f.header("TPU 默认的 exp、log 差几十格，GPU 只差个位数",
                  "左：条长 ＝ 对正确舍入的最大误差（格，对数刻度）；右：高精度的软件 exp 和倒数，分几步算",
                  [(BL, "TPU 默认"), (GR, "TPU 高精度写法"), (GY, "GPU")])
    PH = 420
    LW = 760
    py = f.panel(0, y0, LW, PH, "最大误差，按格计（实测）", BL)
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
        "代价：单算子对上了 GPU 的精度，整网未必跟着对上 —— sigmoid、silu 换成精确写法后整网先是偏得更多，查到一处是反向公式跟 GPU 不同；改完要整网再验。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（TPU v5p，FP32，对 FP64 正确舍入数格）：默认 %s；高精度写法 %s。GPU：%s。TPU 默认的上限跟 JAX 文档给这一代芯片写的一致。"
               % (N.TRANS_T, N.TRANS_HI_T, N.TRANS_GPU_NOTE),
               "⚠️ 右格是教科书做法（区间缩减、多项式、牛顿迭代），只讲思路；各家库的系数、项数和缩减方式不同。sigmoid、silu 的高精度写法用高精度的 exp 搭：e ＝ exp(−|x|)，再算 1 ÷ (1 ＋ e)，倒数做一次牛顿修正。")
    f.save("fig8-trans.svg", yb + 14)


fig_formats()
fig_truncate()
fig_tf32()
fig_inout()
fig_windowed()
fig_lmhead()
fig_excess()
fig_normscale()
fig_boundary()
fig_captain()
fig_elide()
fig_trans()
fig_ulp2()
