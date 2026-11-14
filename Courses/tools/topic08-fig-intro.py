# -*- coding: utf-8 -*-
r"""专题八 · 第一节「什么叫对齐」的三张图。

fig8-assoc   回答：为什么不能要求 TPU 和 GPU 一位不差？
             数字现算：1 ＋ 2⁻⁸ ＋ 2⁻⁸ 在 BF16 里按两种顺序加，用 topic08_numbers.bf16（就近舍入、平局取偶）真算。
             刻意没画：FP32 的同类例子（道理一样，格子更密，画不出来）。
fig8-ruler   回答：差多少才算「不一样」？—— 拿 GPU 自己重跑的差当尺子。
             数字：10 次 GPU 重跑的范围与 TPU 三种跑法，见 topic08_numbers（实测）。
fig8-funnel  回答：从哪儿查起？—— L0 到 L4 五级，验收从下往上，排查从上往下。
             刻意没画：每级的具体检查项（在正文表里）。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0   # 锤炼阶段（2026-11-14）：图上落点带只留一行粗体结论，其余进图下折叠
import topic08_numbers as N

W = 1400

# ── fig8-assoc 的数：真算一遍 ──────────────────────────────────────
A, B, C = 1.0, 2.0 ** -8, 2.0 ** -8
AB = N.bf16(A + B)
LEFT = N.bf16(AB + C)
BC = N.bf16(B + C)
RIGHT = N.bf16(A + BC)
assert AB == 1.0 and LEFT == 1.0, (AB, LEFT)            # 每次都正好落在两格正中间，平局取偶回到 1
assert BC == 2.0 ** -7 and RIGHT == 1.0078125, (BC, RIGHT)
ULP1 = 2.0 ** -7                                        # BF16 在 1 附近的格距


def fig_assoc():
    f = Fig(W, "同一个式子 1 加 2 的负 8 次方再加 2 的负 8 次方，在 BF16 里按两种顺序算。"
               "左边先加前两个：1 加 0.0039 正好落在两格正中间，按平局取偶舍回 1，再加一次还是舍回 1，结果是 1。"
               "右边先把两个小数加起来得到 0.0078，正好是一格，1 加它落在格点上，结果是 1.0078。"
               "两种顺序差了一整格。")
    y0 = f.header("加法换个顺序，末位就不一样　——　<tspan font-weight=\"700\">所以「一位不差」做不到</tspan>",
                  "竖线 ＝ BF16 在 1 及以上能表示的数（一格 ＝ 2⁻⁷ ≈ 0.0078；1 以下格距减半）；圆点 ＝ 精确结果，落在两格之间就要舍到最近的那一格",
                  [(PU, "精确值"), (INK, "舍入后实际存下的数")])
    PH = 300
    for k, (title, col, xs, res) in enumerate([
            ("先算 (1 ＋ 0.0039)，再 ＋ 0.0039", GY, [(A + B, AB), (AB + C, LEFT)], LEFT),
            ("先算 (0.0039 ＋ 0.0039)，再 ＋ 1", GY, [(A + BC, RIGHT)], RIGHT)]):
        px = k * 710
        py = f.panel(px, y0, 690, PH, title, col)
        # 数轴：1 到 1 + 3 格（1 以下格距减半成 2⁻⁸，不画，免得画错）
        X0, SC = px + 70, 540 / (3 * ULP1)
        yl = py + 130

        def X(v):
            return X0 + (v - 1.0) * SC
        f.line(X0 - 20, yl, X0 + 560, yl, GY2, 1.2, arrow=False)
        for i in range(4):
            v = 1.0 + i * ULP1
            f.line(X(v), yl - 16, X(v), yl + 16, INK, 1.6, arrow=False)
            f.t(X(v), yl + 40, "%.4f" % v, GY, size=13, anchor="middle")
        for j, (exact, stored) in enumerate(xs):
            yy = yl - 50 - j * 34
            f.box(X(exact) - 7, yy - 7, 14, 14, PU, PU, 7)
            if exact != stored:
                f.path([(X(exact), yy + 8), (X(stored) + 4, yl - 10)], INK, 1.6)
                f.t(X(exact) + 12, yy + 5, "第 %d 次加：正好在两格中间 → 舍回 %.4f" % (j + 1, stored), INK, size=13)
            else:
                f.t(X(exact) + 12, yy + 5, "正好落在格上，不用舍", INK, size=13)
        f.box(X(res) - 8, yl - 8, 16, 16, INK, INK, 8)
        f.t(px + 24, py + PH - 40, "结果：%.7f" % res, INK, True, 17)
        f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "差的不是谁算错了，是加法顺序不同", [
        "GPU 和 TPU 把一次矩阵乘切成不同的块、按不同的顺序累加，末位必然不同；几百步训练下来，两条轨迹必然分开。",
        "所以能要求的只有一条：TPU 和 GPU 的差，不超过 GPU 自己跟自己的差。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导：BF16 有 7 位尾数，1 附近的格距 2⁻⁷；1 ＋ 2⁻⁸ 正好落在 1 和 1 ＋ 2⁻⁷ 中间，就近舍入平局取偶，舍回 1。",
               "📌 脚本里用 struct 把 float32 截到高 16 位并按平局取偶进位，真算两遍，结果写死在断言里。")
    f.save("fig8-assoc.svg", yb + 14)


# ── fig8-ruler ────────────────────────────────────────────────────
ARM_SPREAD = max(N.TPU_ARMS) - min(N.TPU_ARMS)          # 6.3e-5
RULER_SPREAD = N.GPU_RUNS10_HI - N.GPU_RUNS10_LO        # 6.7e-5
assert 0.5 < ARM_SPREAD / RULER_SPREAD < 1.5, (ARM_SPREAD, RULER_SPREAD)
def fig_ruler():
    f = Fig(W, "横轴是第 301 到 1000 步 loss 的平均相对偏差，以第一份 GPU 运行为准。"
               "灰色一段是 GPU 用同样设置再跑 10 次落下的范围，从负万分之 0.65 到正百万分之 2.4。"
               "TPU 三种跑法落在负万分之 1.03 到负万分之 1.66，全在尺子外面，最远偏出 5.3 个标准差。")
    y0 = f.header("尺子：不问 TPU 跟 GPU 差多少，问它比 GPU 自己跟自己差得多不多",
                  "横轴 ＝ 第 301–1000 步 loss 平均偏了多少（相对，以第一份 GPU 运行为准）；灰色一段 ＝ 同样设置的 GPU 运行一共 10 次落下的范围（含基线那一份）",
                  [(GY2, "GPU 自己重跑（噪声）"), (BL, "TPU 三种跑法")])
    PH = 250
    py = f.panel(0, y0, W, PH, "混元 3 截 4 层，1000 步（实测）", BL)
    LO, HI = -2.0e-4, 0.5e-4
    X0, XW = 90, 1220

    def X(v):
        return X0 + (v - LO) / (HI - LO) * XW
    yl = py + 140
    f.line(X0, yl, X0 + XW, yl, GY2, 1.3, arrow=False)
    for i in range(6):
        v = LO + i * 0.5e-4
        f.line(X(v), yl - 6, X(v), yl + 6, GY2, 1.2, arrow=False)
        f.t(X(v), yl + 48, "0" if abs(v) < 1e-12 else "%+.1f×10⁻⁴" % (v * 1e4), GY, size=13, anchor="middle")
    # 0 在哪、为什么灰段不以 0 为中心 —— 先把 0 是谁标出来（贴右沿的原因在 fig8-baseline）
    f.t(X(0), yl + 68, "基线 ＝ 第一份 GPU", INK, True, 13, "middle")
    f.box(X(N.GPU_RUNS10_LO), yl - 22, X(N.GPU_RUNS10_HI) - X(N.GPU_RUNS10_LO), 44, LINE2, GY2, 6)
    f.t((X(N.GPU_RUNS10_LO) + X(N.GPU_RUNS10_HI)) / 2, yl - 34, "10 次 GPU 重跑都在这儿", GY, True, 14, "middle")
    xm = X(N.GPU_RUNS10_TAIL_MEAN)        # 标准差从均值量，不从基线 0 量
    f.line(xm, yl - 22, xm, yl + 22, INK, 1.6, dash="4 3", arrow=False)
    f.t(xm + 5, yl + 5, "均值 %s" % N.tx(N.GPU_RUNS10_TAIL_MEAN, "%.2e"), INK, size=12)
    for k, v in enumerate(N.TPU_ARMS):
        f.box(X(v) - 8, yl - 8, 16, 16, BL, BL, 8)
        z = (v - N.GPU_RUNS10_TAIL_MEAN) / N._sd10
        f.t(X(v), yl - 26 - (k % 2) * 22, "%s · 偏 %.1f 个标准差" % ({0: "原样", 1: "不开重算", 2: "改片上内存上限"}.get(k, ""), abs(z)), BL, size=13, anchor="middle")
    f.t(X(-1.35e-4), py + 44, "TPU：三种跑法都落在尺子外，方向一致", BL, True, 15, "middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "尺子一立，「1e-4 出头」就有了大小", [
        "TPU 的偏移离 GPU 重跑的均值 2.9 到 5.3 个标准差，而且三种跑法都朝同一边：不是噪声，值得查。",
        "标准差在这里只用来描述有多远；正式的判法后面讲。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：10 次同设置 GPU 运行（含最早跑的 3 份）的带符号均值 %s ± %s（均值 ± 样本标准差），范围 −6.5e-5 至 +2.4e-6。" % (N.tx(N.GPU_RUNS10_TAIL_MEAN, "%.2e"), N.tx(N._sd10, "%.2e")),
               "📌 实测：TPU 原样 −1.66e-4、不开重算 −1.35e-4、改编译器片上内存上限 −1.03e-4；标准差个数 ＝ (TPU − GPU 10 份均值) ÷ GPU 样本标准差（本课推导），从图上那条虚线量，不从基线 0 量。",
               "📌 实测：第一份 GPU 运行几乎贴着 10 次的最上沿（排第三高）；扣掉 10 次的均值以后，TPU 三种跑法仍偏 −0.8e-4 到 −1.4e-4。",
               "📌 实测：三个蓝点彼此也散开约 %.0fe-5，跟灰段（约 %.0fe-5）差不多宽：是整团挪了，不是哪一种跑法单独跑偏。" % (ARM_SPREAD * 1e5, RULER_SPREAD * 1e5),
               "📌 实测：最左是原样跑：第 301–1000 步平均 −1.66e-4，开头说的「约 1.7e-4」就是它。",
               "注：落在尺子外 ≠ 判不过；正式判法见判定那张图（r ＝ 被测 ÷ 噪声上限 N）。")
    f.save("fig8-ruler.svg", yb + 14)


# ── fig8-funnel ───────────────────────────────────────────────────
LEVELS = [
    ("L4", "多步", "整条训练轨迹", BL),
    ("L3", "整网第 1 步", "整张网络，只看第 1 步", BL),
    ("L2", "单层", "一层一层隔离开", BL),
    ("L1", "单个算子", "最小的计算单元", BL),
    ("L0", "地基", "喂进去的东西", RD),
]


def fig_funnel():
    f = Fig(W, "五级检查，画成一座塔。最底下 L0 地基最宽：数据、权重、配置必须一位不差。"
               "往上依次是 L1 单个算子、L2 单层、L3 整网第一步、L4 多步。"
               "左边箭头向上：验收从下往上，低一级不过，上一级的结论不可信。"
               "右边箭头向下：出了问题从上往下，一级一级缩小范围。")
    y0 = f.header("五级检查：验收从下往上，排查从上往下",
                  "塔越往下越宽 ＝ 越基础、越要先过；左边箭头是验收的方向，右边箭头是排查的方向",
                  [(RD, "必须一位不差"), (BL, "比噪声，或比 FP64 真值")])
    TH, GAP = 70, 10
    top = y0 + 20
    CX = W / 2
    for i, (lv, name, what, col) in enumerate(LEVELS):
        w = 520 + i * 150
        y = top + i * (TH + GAP)
        f.box(CX - w / 2, y, w, TH, "#fff", col, 10, sw=2)
        f.box(CX - w / 2, y, 8, TH, col, col, 3)
        f.t(CX - w / 2 + 30, y + 30, lv, col, True, 20, big=True)
        f.t(CX - w / 2 + 84, y + 30, name, INK, True, 16)
        f.t(CX - w / 2 + 84, y + 54, what, GY, size=14)
    bot = top + 5 * (TH + GAP) - GAP
    f.line(70, bot, 70, top + 6, GR, 3)
    f.t(110, bot + 30, "验收：从下往上", GR, True, 15, "middle")
    f.t(110, bot + 52, "低一级不过，上一级不可信", GY, size=13, anchor="middle")
    f.line(W - 70, top, W - 70, bot - 6, RD, 3)
    f.t(W - 110, bot + 30, "排查：从上往下", RD, True, 15, "middle")
    f.t(W - 110, bot + 52, "L4 发现，逐级缩小范围", GY, size=13, anchor="middle")
    yb = f.band(bot + 80, "ok", "地基不平，楼上量到的差就说不清是算出来的还是喂出来的", [
        "L0 的要求是一位不差；L1–L4 才谈「差多少算正常」。",
        "代价：每一级都要能单独拿到证据 —— 在 TPU 上怎么拿，后面专门讲。",
    ])
    f.save("fig8-funnel.svg", yb + 14)


fig_assoc()
fig_ruler()
fig_funnel()
