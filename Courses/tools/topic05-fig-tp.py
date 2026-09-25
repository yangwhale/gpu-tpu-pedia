# -*- coding: utf-8 -*-
r"""专题五 · 第三节「第二刀：切权重」的三张静态图。

⭐ 这一节承重的是一条推导（⚠️ 推导，不是某篇论文的原话，页面上也标成推导）：
   「每在网络上搬一个字节，能换来多少 FLOPs」——&#160;
     · FSDP：一步搬 ≈ 6P 字节（两次 AllGather 拼 bf16 权重 ＋ 一次 ReduceScatter 分 bf16 梯度），
       算 6PT FLOPs（T ＝ 每张卡这一步的 token 数，稠密近似）→ 每字节 ＝ **T**，跟模型多大无关。
     · TP：一层前向 2 次、反向 2 次 AllReduce，每次每卡发 ≈ 4Th 字节（bf16 激活），共 16Th；
       一层算 72h²T／n（稠密层 12h² 参数，前向 2、反向 4 倍）→ 每字节 ＝ **4.5h／n**，跟 batch 无关。
   硬件那一边：v7 每芯片 2,307 TFLOP/s（bf16）÷ 每卡发出方向 600 GB/s ≈ **3,845 FLOPs/字节**。
   ⛔⛔ 2026-09-25 专家评审抓到的错：原来用的是 1,200 GB/s ——&#160;那是 6 条链路 × 200 GB/s
     **收发两个方向加起来**的数，而分子 6Ψ、16Th 都是「每卡发出」的量，只能跟发出方向比（一半，600）。
     旧门槛 1,922 大了一倍。只用一根轴时分母再除以 3。

⛔ 所有数字现算并断言。

⛔ 刻意没画：fig-intensity 只画最乐观的带宽线（三根轴都用满、首尾成环），真实门槛更高，正文 3.4 说明；
   fig-pp-bubble 画的是 GPipe 调度（先全部前向再全部反向），不是 1F1B —— 两者气泡一样大，GPipe 更好看懂。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE
RD_ = RD

W = 1400
C_V7 = 2307e12                  # v7 每芯片 bf16 FLOP/s
B_V7 = 0.6e12                   # v7 每芯片 ICI 发出方向（6 条链路 × 100 GB/s；双向合计才是 1,200）
RIDGE = C_V7 / B_V7
H_V3 = 7168
assert abs(RIDGE - 3845) < 1, RIDGE


def tp_intensity(h, n):
    return 4.5 * h / n


N_TP_MAX = 4.5 * H_V3 / RIDGE
assert 8 < N_TP_MAX < 8.5, N_TP_MAX         # ≈ 8.4：TP 8 路只是勉强在线上


def fig_intensity():
    f = Fig(W, "每在网络上搬一个字节能换来多少次计算。横轴是每张卡这一步分到的 token 数。"
               "FSDP 那条线斜着往上走：它每字节换来的计算正好等于 token 数，batch 越大越划算。"
               "TP 那条线是平的，只取决于隐藏维除以 TP 度数，跟 batch 无关。"
               "中间那条水平虚线是 TPU v7 的硬件线，约三千八百四十五：每秒能算的次数除以每秒能搬的字节。"
               "线下面就是被通信拖住。FSDP 在 token 数少于三千八百四十五时掉到线下；"
               "V3 的 TP 开到 8 路时，按最乐观的线才勉强在线上，开到 32 路时在线下")
    y0 = f.header("两把刀，两种账　——　<tspan font-weight=\"700\">FSDP 看 batch，TP 看隐藏维</tspan>",
                  "纵轴：每在网络上搬 1 字节，换来多少 FLOPs（⚠️ 推导，稠密层近似）。"
                  "低于硬件线 ＝ 算得没有搬得快，被通信拖住",
                  [(BL, "FSDP：＝ 每卡 token 数 T"), (OR, "TP：＝ 4.5 × 隐藏维 ÷ TP 度数"), (RD, "v7 硬件线 ≈ 3,845")])
    PX, PY, PW, PH = 150, y0 + 20, 980, 380
    TMAX, IMAX = 8192, 8192
    # ⭐ 2026-09-25 逐图审：红线以下涂浅红，接回砌墙的比方（师傅＝卡，小工＝网络）
    _yr = PY + PH - PH * RIDGE / IMAX
    f.poly([(PX, _yr), (PX + PW, _yr), (PX + PW, PY + PH), (PX, PY + PH)], "#fce8e6")   # box() 会把大块底色压平，用 poly
    f.t(PX + PW * 0.64, PY + PH - PH * 2500 / IMAX, "线下：师傅（卡）在等小工（网络）", RD, True, 15)
    f.t(PX + 20, PY + 30, "线上：师傅忙得过来", GR, True, 15)
    f.box(PX, PY, PW, PH, "none", LINE, 6)

    def X(t):
        return PX + PW * t / TMAX

    def Y(i):
        return PY + PH - PH * min(i, IMAX) / IMAX
    for t in (0, 2048, 4096, 6144, 8192):
        f.t(X(t), PY + PH + 22, "{:,}".format(t), GY, size=12.5, anchor="middle")
    f.t(PX + PW / 2, PY + PH + 46, "每张卡这一步的 token 数 T", GY, True, 13.5, anchor="middle")
    for i in (0, 2048, 4096, 6144, 8192):
        f.t(PX - 10, Y(i) + 5, "{:,}".format(i), GY, size=12.5, anchor="end")
    f.path("M%d,%d L%d,%d" % (X(RIDGE), Y(0), X(RIDGE), Y(RIDGE)), RD, 1.6, dash="4,4", arrow=False)
    f.t(X(RIDGE) - 8, Y(420), "← T ＜ 3,845：FSDP 被拖住", RD, True, 13, "end")
    f.path("M%d,%d L%d,%d" % (X(0), Y(0), X(TMAX), Y(TMAX)), BL, 3, arrow=False)
    f.t(X(6600), Y(6600) - 14, "FSDP", BL, True, 15)
    for n, lab in ((8, "TP 8 路"), (32, "TP 32 路")):
        yi = tp_intensity(H_V3, n)
        f.path("M%d,%d L%d,%d" % (X(0), Y(yi), X(TMAX), Y(yi)), OR, 2.5,
               dash="7,4" if n == 32 else None, arrow=False)
        f.t(X(TMAX) + 10, Y(yi) + 5, "%s ≈ %s" % (lab, "{:,.0f}".format(yi)), OR, True, 13.5)
    f.path("M%d,%d L%d,%d" % (X(0), Y(RIDGE), X(TMAX), Y(RIDGE)), RD, 2, dash="4,4", arrow=False)
    f.t(X(TMAX) + 10, Y(RIDGE) + 20, "v7 硬件线 ≈ 3,845", RD, True, 13.5)
    yb = f.band(PY + PH + 70, "ok", "batch 小就换 TP，batch 大就用 FSDP", [
        "FSDP 每字节换来的计算 ＝ 每卡 token 数：在 v7 上每卡少于约 3,845 个 token，就搬得比算得慢。"
        "　<tspan font-weight=\"700\">加卡又不想加 batch，FSDP 迟早掉到线下。</tspan>",
        "TP 的账跟 batch 无关，只看隐藏维 ÷ TP 度数：V3 的隐藏维 7,168，"
        "TP 8 路 ≈ 4,032 按最乐观的线才勉强在线上，32 路 ≈ 1,008 就掉下去了　——　<tspan font-weight=\"700\">TP 有一个跟 batch 无关的上限</tspan>。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 推导，非论文原话：FSDP 一步搬 ≈ 6Ψ 字节（2 次 AG 拼 bf16 权重 ＋ 1 次 RS 分 bf16 梯度）、算 6ΨT FLOPs；"
               "TP 一层 4 次 AllReduce 各发 ≈ 4Th 字节、算 72h²T／n。都按稠密层、通信与计算完全重叠算。",
               "📌 v7：每芯片 bf16 2,307 TFLOP/s；ICI 1,200 GB/s 是 6 条链路收发合计，每卡发出方向按 600 GB/s 算（Inferact TPU megakernel 博客规格表、wiki ici-dcn，"
               "来源为 Google TPU7x 文档；6 条链路的拆分与发出方向 600 是推导）。只用一根轴时硬件线约高 3 倍。V3 隐藏维 7,168 取自 config.json。")
    f.save("fig5-intensity.svg", yb + 14)


def fig_tp_mlp():
    f = Fig(W, "张量并行怎么切一个 MLP。输入 X 每张卡都有一整份。第一块权重 W1 按列切成两半，"
               "每张卡算出中间结果的一半，激活函数可以各自算，不用通信。第二块权重 W2 按行切，"
               "每张卡算出的是完整输出的一部分和，最后做一次 AllReduce 加起来，每张卡拿到完整的 Y。"
               "关键是两次矩阵乘之间不需要任何通信：先列切、再行切，正好让一次 AllReduce 放在最后")
    y0 = f.header("TP 切 MLP　——　<tspan font-weight=\"700\">先按列切，再按行切，中间一次通信都不用</tspan>",
                  "两张卡的例子。灰 ＝ 两张卡都有的完整副本，蓝 ／ 橙 ＝ 各自那一半；虚线是切口。蓝的只跟蓝的相乘，所以中间不用找对方要东西",
                  [(BL, "卡 0 的那一半"), (OR, "卡 1 的那一半"), (GY2, "完整副本")])
    # ⭐ 2026-09-25 逐图审后重画：原来 W1、W2 只是两个色块，看不出往哪个方向切。现在按真实形状画、切口画成虚线。
    PH = 365
    py = f.panel(0, y0, W, PH, "Y ＝ GeLU(X · W1) · W2", BL, sub="Megatron-LM 的切法")
    CY = py + 175                        # 各矩阵的垂直中线

    def half(x, y, w, h, vertical, lab, sub):
        """一块矩阵，按列（vertical=True）或按行切成蓝／橙两半，中间一条虚线是切口。"""
        if vertical:
            f.box(x, y, w / 2, h, BL, BL, 3)
            f.box(x + w / 2, y, w / 2, h, OR, OR, 3)
            f.path("M%d,%d L%d,%d" % (x + w / 2, y - 12, x + w / 2, y + h + 12), INK, 2, dash="5,4", arrow=False)
        else:
            f.box(x, y, w, h / 2, BL, BL, 3)
            f.box(x, y + h / 2, w, h / 2, OR, OR, 3)
            f.path("M%d,%d L%d,%d" % (x - 12, y + h / 2, x + w + 12, y + h / 2), INK, 2, dash="5,4", arrow=False)
        f.t(x + w / 2, y - 22, lab, INK, True, 15, "middle")
        f.t(x + w / 2, y + h + 30, sub, GY, size=13, anchor="middle")

    f.box(50, CY - 60, 70, 120, GY2, GY2, 3)
    f.t(85, CY + 6, "X", "#ffffff", True, 18, "middle")
    f.t(85, CY - 82, "输入 X", INK, True, 15, "middle")
    f.t(85, CY + 90, "每张卡都有整份", GY, size=13, anchor="middle")
    f.t(150, CY + 8, "×", INK, True, 22, "middle")
    half(180, CY - 35, 240, 70, True, "W1：竖着切一刀", "左半在卡 0，右半在卡 1")
    f.t(450, CY + 8, "＝", INK, True, 22, "middle")
    half(480, CY - 60, 240, 120, True, "中间结果也是左右两半", "GeLU 逐个元素算，各算各的")
    f.t(750, CY + 8, "×", INK, True, 22, "middle")
    half(780, CY - 110, 60, 220, False, "W2：横着切一刀", "")
    f.t(810, CY + 140, "上半在卡 0，下半在卡 1", GY, size=13, anchor="middle")
    f.t(870, CY + 8, "＝", INK, True, 22, "middle")
    f.box(900, CY - 75, 70, 60, BL, BL, 3)
    f.t(935, CY - 40, "部分和", "#ffffff", True, 13, "middle")
    f.t(935, CY + 8, "＋", INK, True, 20, "middle")
    f.box(900, CY + 15, 70, 60, OR, OR, 3)
    f.t(935, CY + 50, "部分和", "#ffffff", True, 13, "middle")
    f.box(1000, CY - 32, 150, 64, "none", GR, 8, sw=2)
    f.t(1075, CY - 4, "AllReduce", GR, True, 16, "middle")
    f.t(1075, CY + 18, "唯一一次通信", GR, size=13, anchor="middle")
    f.line(972, CY - 45, 998, CY - 12, GY2, 1.6)
    f.line(972, CY + 45, 998, CY + 12, GY2, 1.6)
    f.line(1152, CY, 1200, CY, GY2, 1.6)
    f.box(1210, CY - 60, 70, 120, GY2, GY2, 3)
    f.t(1245, CY + 6, "Y", "#ffffff", True, 18, "middle")
    f.t(1245, CY + 90, "每张卡拿到整份", GY, size=13, anchor="middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "切法的全部巧思，就是把通信挤到最后一次", [
        "第一块按列切、第二块按行切：中间结果各留一半、各自过激活函数，<tspan font-weight=\"700\">两次矩阵乘之间零通信</tspan>。",
        "attention 同理：按头切，每张卡算自己那几个头，出口处一次 AllReduce。"
        "　于是一层前向 2 次、反向 2 次 AllReduce，<tspan font-weight=\"700\">每层都有，频率极高</tspan>。",
    ])
    yb = f.src(yb + 10, "📌 出处：Shoeybi 等，Megatron-LM，arXiv 1909.08053 sec. 3（MLP 与 self-attention 的切法、f／g 两个通信算子）。")
    f.save("fig5-tp-mlp.svg", yb + 14)


def gelu(x):
    """GeLU 精确式 0.5·x·(1＋erf(x／√2))，Megatron 用的也是它（或 tanh 近似，差在小数点后三位）。"""
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


# 反例的三个数：同一个元素在两张卡上各有一份部分和
P0, P1 = 2.0, -3.0
G_TRUE = gelu(P0 + P1)                  # 先加再过 GeLU：正确答案
G_WRONG = gelu(P0) + gelu(P1)           # 各过各的再加：错的
assert abs(G_TRUE - (-0.1587)) < 1e-3 and abs(G_WRONG - 1.9504) < 1e-3, (G_TRUE, G_WRONG)


def fig_tp_order():
    """⭐ 2026-09-25 现场讲课补的「为什么」图：fig-tp-mlp 只画了「先列后行」，没画「反过来为什么不行」。
    承重的只有一件事：GeLU 不是线性的，部分和不能先各自过激活再相加。数字用 gelu() 现算并断言。
    ⭐ 同日逐图审（⛔ 写字板）后重画：左边把 GeLU 曲线真画出来、三个点标在曲线上；右边两条路并排。"""
    f = Fig(W, "为什么 Megatron 一定要先竖着切第一块权重。左边是 GeLU 曲线：先横着切的话，同一个元素在两张卡上各有一份部分和，"
               "卡 0 是 2、卡 1 是负 3，真值是负 1。各过各的 GeLU 再相加得 1.950，先加再过 GeLU 得负 0.159，差得很远。"
               "右边两条路：先横着切，过 GeLU 之前必须先做一次 AllReduce；先竖着切，每个元素整个在一张卡上，直接过 GeLU，零通信")
    y0 = f.header("为什么非得先竖着切　——　<tspan font-weight=\"700\">GeLU 是弯的，部分和不能各过各的</tspan>",
                  "同一个 MLP：Y ＝ GeLU(X · W1) · W2。只看中间结果里的某一个元素",
                  [(BL, "卡 0 手里的"), (OR, "卡 1 手里的"), (RD, "错"), (GR, "对")])
    PH = 380
    LW = 820
    py = f.panel(0, y0, LW, PH, "先横着切，每张卡只有一份部分和：2 和 −3，真值 −1", RD)
    X0, X1, U0, U1 = 70, 780, -3.5, 2.5
    YT, YB, G0, G1 = py + 40, py + 300, -0.4, 2.6

    def px(u):
        return X0 + (u - U0) / (U1 - U0) * (X1 - X0)

    def py_(g):
        return YB - (g - G0) / (G1 - G0) * (YB - YT)

    f.line(X0, py_(0), X1, py_(0), GY2, 1.2, arrow=False)
    f.line(px(0), YT, px(0), YB, GY2, 1.2, arrow=False)
    f.t(X1, py_(0) + 18, "输入", GY, size=13, anchor="end")
    f.t(px(0) + 8, YT + 4, "GeLU 之后", GY, size=13)
    pts = [(U0 + i * (U1 - U0) / 120) for i in range(121)]
    f.path("M " + " L ".join("%.1f %.1f" % (px(u), py_(gelu(u))) for u in pts), INK, 2.4, arrow=False)

    def dot(u, col, lab, dx, dy, anchor=None):
        x, y = px(u), py_(gelu(u))
        f.line(x, py_(0), x, y, col, 1.2, dash="4,3", arrow=False)
        f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (x, y, col))
        f.t(x + dx, y + dy, lab, col, True, 15, anchor)

    dot(P0, BL, "卡 0：GeLU(2) ＝ %.3f" % gelu(P0), -16, -12, "end")
    dot(P1, OR, "卡 1：GeLU(−3) ＝ %.3f" % gelu(P1), 0, -18, "middle")
    dot(P0 + P1, GR, "真值：GeLU(−1) ＝ %.3f" % G_TRUE, 0, 34, "middle")
    # 各过各的再加：落在 1.950 那条水平线上
    yw = py_(G_WRONG)
    f.line(X0, yw, px(0), yw, RD, 1.6, dash="6,4", arrow=False)
    f.t(X0 + 6, yw - 10, "各过各的再加：%.3f ＋ (%.3f) ＝ %.3f　✗" % (gelu(P0), gelu(P1), G_WRONG), RD, True, 15)
    f.t(24, py + PH - 44, "差了十几倍，符号都反了：这个元素必须先加齐，才能过 GeLU", RD, True, 15)
    f._pan = None

    RX = LW + 40
    RWID = W - RX
    py2 = f.panel(RX, y0, RWID, PH, "两种切法，这个元素走的路", INK)

    def chip(x, y, col, txt, w=64):
        f.box(x, y, w, 40, col, col, 5)
        f.t(x + w / 2, y + 26, txt, "#ffffff", True, 15, "middle")

    def gelu_box(x, y):
        f.box(x, y, 64, 40, "none", INK, 6, sw=1.6)
        f.t(x + 32, y + 26, "GeLU", INK, True, 14, "middle")

    # 横切在先
    ry = py2 + 50
    f.t(RX + 20, ry, "横切在先", RD, True, 15)
    chip(RX + 20, ry + 18, BL, "2")
    chip(RX + 20, ry + 64, OR, "−3")
    f.line(RX + 88, ry + 60, RX + 110, ry + 60, GY2, 1.6)
    f.box(RX + 114, ry + 38, 108, 44, "none", RD, 8, sw=2)
    f.t(RX + 168, ry + 66, "AllReduce", RD, True, 14, "middle")
    f.line(RX + 226, ry + 60, RX + 246, ry + 60, GY2, 1.6)
    chip(RX + 250, ry + 40, GR, "−1", 54)
    f.line(RX + 308, ry + 60, RX + 326, ry + 60, GY2, 1.6)
    gelu_box(RX + 330, ry + 40)
    f.line(RX + 398, ry + 60, RX + 418, ry + 60, GY2, 1.6)
    f.t(RX + 424, ry + 66, "%.3f" % G_TRUE, GR, True, 15)
    f.t(RX + 20, ry + 128, "多一次通信，每个 MLP 都要付", RD, size=14)
    # 竖切在先
    gy = py2 + 212
    f.t(RX + 20, gy, "竖切在先（Megatron）", GR, True, 15)
    chip(RX + 20, gy + 18, BL, "−1")
    f.box(RX + 20, gy + 64, 64, 40, "none", GY2, 5, dash="5,4")
    f.t(RX + 52, gy + 89, "没有", GY, size=13, anchor="middle")
    f.line(RX + 88, gy + 38, RX + 326, gy + 38, GY2, 1.6)
    f.t(RX + 207, gy + 30, "不用问别人", GR, size=13, anchor="middle")
    gelu_box(RX + 330, gy + 18)
    f.line(RX + 398, gy + 38, RX + 418, gy + 38, GY2, 1.6)
    f.t(RX + 424, gy + 44, "%.3f" % G_TRUE, GR, True, 15)
    f.t(RX + 20, gy + 120, "整个元素只在卡 0 上，零通信", GR, size=14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "所以一层 4 次 AllReduce：前向 2 次、反向 2 次", [
        "前向：MLP 出口一次、attention 出口一次（Megatron 叫它 g）。",
        "反向倒过来：入口处 X 的梯度是两张卡各算一半、要加起来，又各一次（叫 f）。",
    ])
    yb = f.src(yb + 10,
               "📌 Shoeybi 等，Megatron-LM，arXiv 1909.08053 sec. 3：GeLU 非线性，按行切第一块需要在 GeLU 前同步；f 前向恒等、反向 AllReduce，g 反之。",
               "⚠️ 本课反例：2 与 −3 是随手取的数，GeLU 按精确式现算。")
    f.save("fig5-tp-order.svg", yb + 14)


def fig_wide_deep():
    """⭐ 2026-09-25 夜 · 蒸馏 R4：TP 与 PP 的对照记忆点「一个切宽、一个切深」（Hugging Face《Ultra-Scale Playbook》
    的 hidden vs depth 说法）。⛔ 刻意不说「横切／竖切」：各家对横纵的用法相反，本课「竖着切／横着切」已经用来说列切／行切。"""
    f = Fig(W, "同一个 8 层的模型，两种切法。左边切宽：每一层都切成四竖条，四张卡各拿一条，所以每一层算完都要对一次账，做一次 AllReduce，只能坐快线。"
               "右边切深：前两层给卡 0、再两层给卡 1，依此类推，卡和卡之间只在段的交界递一次半成品，能跨慢线；代价是有人要等，就是气泡")
    y0 = f.header("TP 切宽，PP 切深　——　<tspan font-weight=\"700\">一个每层都要对账，一个只在交界递一次</tspan>",
                  "同一个 8 层的模型、4 张卡。颜色 ＝ 这一块归哪张卡",
                  [(BL, "卡 0"), (OR, "卡 1"), (GR, "卡 2"), (PU, "卡 3"), (RD, "要通信的地方")])
    PH = 440
    HW = 680
    COLS = [BL, OR, GR, PU]
    LH, LG = 34, 8
    # 左：TP
    py = f.panel(0, y0, HW, PH, "切宽（TP）：每一层都切成四条", BL)
    X0, LW = 60, 320
    for l in range(8):
        yy = py + 40 + l * (LH + LG)
        for k in range(4):
            f.box(X0 + k * LW / 4, yy, LW / 4 - 3, LH, COLS[k], COLS[k], 3)
        f.t(X0 - 10, yy + 23, "层 %d" % (l + 1), GY, size=12.5, anchor="end")
        f.box(X0 + LW + 14, yy + 8, 18, 18, RD_, RD_, 9)
    f.t(X0 + LW + 44, py + 60, "每一层都要", RD_, True, 14)
    f.t(X0 + LW + 44, py + 82, "四张卡对账", RD_, True, 14)
    f.t(X0 + LW + 44, py + 104, "（attention、MLP 各一次 AllReduce）", RD_, size=13)
    f.t(24, py + PH - 30, "每层都要通信 → 只能坐最快的那圈线", BL, True, 14)
    f._pan = None
    # 右：PP
    px = HW + 40
    py2 = f.panel(px, y0, HW, PH, "切深（PP）：两层一段，一张卡一段", OR)
    X1 = px + 60
    for l in range(8):
        yy = py2 + 40 + l * (LH + LG)
        k = l // 2
        f.box(X1, yy, LW, LH, COLS[k], COLS[k], 3)
        f.t(X1 - 10, yy + 23, "层 %d" % (l + 1), GY, size=12.5, anchor="end")
        if l % 2 == 1 and l < 7:
            f.box(X1 + LW + 14, yy + LH + LG / 2 - 9, 18, 18, RD_, RD_, 9)
    f.t(X1 + LW + 44, py2 + 104, "只在段的交界", RD_, True, 14)
    f.t(X1 + LW + 44, py2 + 126, "递一次半成品", RD_, True, 14)
    f.t(X1 + LW + 44, py2 + 148, "（一对一收发）", RD_, size=13)
    f.t(px + 24, py2 + PH - 30, "只 3 处通信 → 能跨慢线；代价是有人闲着等", OR, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "一个切宽、一个切深，各付各的代价", [
        "切宽：每张卡每层只算四分之一，可每一层都得跟另外三张卡对账，所以 TP 出不了一台机器。",
        "切深：卡和卡之间几乎不说话，所以 PP 敢横跨很多台机器；代价是流水线灌满、排空时有人在等。",
    ])
    f.save("fig5-wide-deep.svg", yb + 14)


def bubble_ratio(p, m):
    """Narayanan 等 arXiv 2104.04473 sec. 2.2.1 的口径：气泡时间 ÷ 理想计算时间 ＝ (p−1)/m。
    GPipe 与 1F1B 一样大（1F1B 只省激活显存）。"""
    return (p - 1) / m


def bubble_share(p, m):
    """同一件事换成「占整步时长」的口径：(p−1)/(m+p−1)。"""
    return (p - 1) / (m + p - 1)


def fig_pp():
    P, M = 4, 8
    f = Fig(W, "流水线并行的时间表。四个 stage 从上到下，横轴是时间。每个小方块是一个 micro-batch 在这一段的前向或反向。"
               "一开始只有第一段在干活，后面几段在等；最后只剩第一段在做反向，后面几段已经做完在等。"
               "那两个三角形的空白就是气泡。micro-batch 越多，气泡占的比例越小。"
               "四段八个 micro-batch 时，气泡是理想计算时间的八分之三，占整步约百分之二十七")
    y0 = f.header("PP 的代价：气泡　——　<tspan font-weight=\"700\">开头等人灌满，结尾等人排空</tspan>",
                  "4 个 stage、8 个 micro-batch，GPipe 式时间表（先全部前向、再全部反向）。反向按前向的 2 倍长画",
                  [(BL, "前向"), (GR, "反向"), ("#cfd8dc", "气泡：这一段在空等")])
    CW = 30
    PH = 30 + P * 46 + 60
    py = f.panel(0, y0, W, PH, "一步里每个 stage 在干什么", BL)
    X0 = 130
    for s in range(P):
        yy = py + 24 + s * 46
        f.t(20, yy + 24, "stage %d" % s, INK, True, 14)
        # 气泡：先把整条时间轴涂成浅灰，再在上面盖前向／反向块
        total_t = (M + P - 1) + (M * 2 + (P - 1) * 2)
        f.box(X0, yy + 6, total_t * CW, 30, "#eceff1", "none", 3)
        # 前向：micro-batch i 在 stage s 的时刻 = i + s
        for i in range(M):
            x = X0 + (i + s) * CW
            f.box(x + 1, yy + 6, CW - 2, 30, BL, BL, 3)
            f.t(x + CW / 2, yy + 26, str(i), "#ffffff", True, 11.5, "middle")
        # 反向：从最后一段开始，时刻 = (M+P-1) + (M-1-i)*2 + (P-1-s)*2
        t0 = M + P - 1
        for i in range(M):
            x = X0 + (t0 + (P - 1 - s) * 2 + i * 2) * CW
            f.box(x + 1, yy + 6, 2 * CW - 2, 30, GR, GR, 3)
            f.t(x + CW, yy + 26, str(i), "#ffffff", True, 11.5, "middle")
    total = (M + P - 1) + (M * 2 + (P - 1) * 2)
    f.line(X0, py + 24 + P * 46 + 8, X0 + total * CW, py + 24 + P * 46 + 8, GY2, 1.4)
    f.t(X0 + total * CW, py + 24 + P * 46 + 28, "时间 →", GY, size=13, anchor="end")
    b, sh = bubble_ratio(P, M), bubble_share(P, M)
    assert abs(b - 3 / 8) < 1e-9 and abs(sh - 3 / 11) < 1e-9
    f.t(X0, py + 24 + P * 46 + 30,
        "气泡 ÷ 理想计算时间 ＝ (p−1) ÷ m ＝ 3 ÷ 8 ≈ %.0f%%（占整步约 %.0f%%）" % (b * 100, sh * 100), RD, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "气泡是纯损失，只能摊薄，不能消灭", [
        "stage 越多、micro-batch 越少，气泡越大：<tspan font-weight=\"700\">(p−1) ÷ m</tspan>。"
        "　所以 PP 要配足够多的 micro-batch，而 micro-batch 多了，每张卡要攒的激活也多。",
        "后来的调度都在跟这块空白较劲：交错式（VPP）把每段再切细，Zero Bubble 拿权重梯度去填缝，"
        "DualPipe 两头同时灌。<tspan font-weight=\"700\">代价都是更复杂的调度和更多的点对点通信</tspan>。",
    ])
    yb = f.src(yb + 10, "📌 气泡占比：Narayanan 等，arXiv 2104.04473 sec. 2.2（GPipe／1F1B 的 bubble time fraction）；"
                        "交错式把气泡再除以每卡的虚拟段数 v。Zero Bubble：arXiv 2401.10241。DualPipe：github.com/deepseek-ai/DualPipe。")
    f.save("fig5-pp-bubble.svg", yb + 14)


fig_intensity()
fig_tp_mlp()
fig_tp_order()
fig_wide_deep()
fig_pp()
