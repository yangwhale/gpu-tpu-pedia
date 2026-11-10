# -*- coding: utf-8 -*-
r"""专题九 · 第十节的「TPU 脾气」图（第二次迁移的新材料）。

fig9-temper  回答：在 TPU 上改了一样东西，结果会不会变？变了算不算 bug？
             同一个小尺寸模型、同一份权重和数据，每次只改一样东西跑 300 步，跟一次确认过的 TPU 运行（golden）比，r 用 GPU 重跑当刻度。
             一行一类改动；左边一格数逐位相同的，右边按 300 步曲线最差 r 画点（对数轴）；
             蓝点 ＝ 跟 GPU 重跑分不开；红点 ＝ 分得开；红圈 ＝ 曲线分不开、第 1 步有张量离 FP64 更远。
             灵敏度那一类（故意埋 bug）不在这张图，单独一张（fig9-sens）。
             数据：topic09_tpuself.json 的 arms（导出自对齐记录 TPU 自比一节的 README.md 的脾气表，实测）。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE
import topic09_numbers as N

W = 1400
T = N.tpuself()
ARMS = [a for a in T["arms"] if not a["cat"].startswith("灵敏度")]
ROWS = [("校准", "校准：重跑、换机器、换集群"), ("并行与切分", "并行与切分"), ("上下文并行", "上下文并行"),
        ("编译开关", "编译开关"), ("数学上等价的代码改写", "数学上等价的改写"),
        ("会影响精度的真实改动", "动到精度的改动"), ("软件版本", "TPU 运行库版本"), ("硬件（v5p ↔ v7x）", "换硬件 v5p ↔ v7x")]
assert sorted({a["cat"] for a in ARMS}) == sorted(c for c, _ in ROWS)
assert len(ARMS) == 89 and len(T["arms"]) == 100


def kind(a):
    if a["tier"] == "逐位相同":
        return "bit"
    if a["gpu_cmp"] == "可区分":
        return "far" if a["r300"] > N.LOO8_300_MAX else "step1"
    return "noise"


COUNT = {c: {"bit": 0, "noise": 0, "far": 0, "step1": 0} for c, _ in ROWS}
for a in ARMS:
    COUNT[a["cat"]][kind(a)] += 1
# 跟对齐记录 README 的汇总表对上（逐位 ／ 分不开 ／ 分得开）
assert (COUNT["校准"]["bit"], COUNT["并行与切分"]["bit"], COUNT["编译开关"]["bit"]) == (6, 2, 9)
assert COUNT["并行与切分"]["noise"] == 18 and COUNT["编译开关"]["noise"] == 9 and COUNT["数学上等价的代码改写"]["noise"] == 5
assert COUNT["数学上等价的代码改写"]["far"] + COUNT["数学上等价的代码改写"]["step1"] == 0
TOTAL = {k: sum(c[k] for c in COUNT.values()) for k in ("bit", "noise", "far", "step1")}
assert TOTAL == {"bit": 29, "noise": 45, "far": 10, "step1": 5}, TOTAL   # README 汇总表：逐位 29、分不开 45、分得开 15

# 图上点名的几个（标签写在点旁边），按 change 原文找
#   值：(文字, dx, dy, anchor)；位置手摆，避开同一行别的点和标签
LABEL = {"FSDP 改成纯数据并行": ("纯数据并行 2.6（16 卡 2.7）", 8, -12, None),
         "纯数据并行 + 词嵌入梯度用 FP32 累加（修复前，有 bug）": ("FP32 累加撞上编译器 bug 611", -10, -10, "end"),
         "关掉编译器的省舍入": ("关省舍入 13", 9, -8, None), "输出头矩阵乘降回默认精度": ("输出头降回默认精度 5.9", 9, -9, None),
         "RMSNorm 的 ε 改小": ("RMSNorm 的 ε 改小 343", -10, -10, "end"), "Adam 的 ε 改大": ("Adam 的 ε 改大 412", -10, 20, "end"),
         "路由偏置改零均值更新": ("偏置换更新公式 3.0", -6, 22, None), "v7x 上关掉编译器的省舍入": ("关省舍入 3.4", 9, -8, None),
         "纯数据并行、不做梯度累积": ("不累积 3.7", 9, 18, None)}


def fig_temper():
    f = Fig(W, "八行，每行一类改动。最左一格是逐位相同的个数；右边每个点是一个改动 300 步曲线最差的 r，对数轴。"
               "校准六个全部逐位相同。并行与切分、编译开关、等价改写、运行库版本、换硬件，大多数点挤在 1 附近的蓝色，跟 GPU 重跑分不开。"
               "红点只有十个，几乎都跟词嵌入梯度或改公式有关：纯数据并行 2.6、关省舍入 13、FP32 累加撞上编译器 bug 611、输出头降回默认精度 5.9、"
               "RMSNorm 的 ε 改小 343、Adam 的 ε 改大 412。另有五个红圈，曲线分不开，是第 1 步对 FP64 抓出来的。")
    y0 = f.header("TPU 的脾气：改一样东西，结果变不变",
                  "一行一类改动，一个点一个改动（第二次迁移，300 步，对 TPU 自己的 golden；r 用 GPU 重跑当刻度）。底色：绿 ≤ 1，黄 到纯噪声最高，红 超过",
                  [(GY2, "逐位相同（左格数个数）"), (BL, "变了，跟 GPU 重跑分不开"), (RD, "分得开")])
    LBW, BITW = 250, 150
    X0, X1 = LBW + BITW + 30, W - 40
    RL, RH = 0.7, 1000.0
    RH_ROW = 54
    PH = len(ROWS) * RH_ROW + 80
    py = f.panel(0, y0, W, PH, "89 个改动（实测）", GY)

    def X(r):
        return X0 + (math.log10(max(min(r, RH), RL)) - math.log10(RL)) / (math.log10(RH) - math.log10(RL)) * (X1 - X0)
    ytop, ybot = py + 10, py + 10 + len(ROWS) * RH_ROW
    f.box(X(RL), ytop, X(1) - X(RL), ybot - ytop, "#e6f4ea", "#e6f4ea", 0)
    f.box(X(1), ytop, X(N.LOO8_300_MAX) - X(1), ybot - ytop, "#fef7e0", "#fef7e0", 0)
    f.box(X(N.LOO8_300_MAX), ytop, X(RH) - X(N.LOO8_300_MAX), ybot - ytop, "#fce8e6", "#fce8e6", 0)
    for v in (1, 2, 10, 100, 1000):
        f.line(X(v), ytop, X(v), ybot, LINE if v != 1 else GY2, 0.8, arrow=False)
        f.t(X(v), ybot + 20, "%g" % v, GY, size=13, anchor="middle")
    f.line(X(N.LOO8_300_MAX), ytop - 4, X(N.LOO8_300_MAX), ybot, INK, 1.4, dash="4 3", arrow=False)
    f.t(X(N.LOO8_300_MAX) + 6, ytop + 16, "纯噪声最高 %s" % N.LOO8_300_MAX_T, INK, True, 13)
    f.box(X(30) - 6, ybot - 26, 12, 12, "#fff", RD, 6, sw=2)
    f.t(X(30) + 12, ybot - 16, "空心：曲线分不开，第 1 步离 FP64 更远", RD, size=13)
    f.t(X(RH), ybot + 40, "300 步曲线最差 r（对数）", GY, size=13, anchor="end")
    f.t(LBW + BITW / 2 + 10, ytop - 0 + 2, "", GY)
    for i, (cat, lab) in enumerate(ROWS):
        yc = ytop + i * RH_ROW + RH_ROW / 2.0
        if i:
            f.line(0 + 14, ytop + i * RH_ROW, W - 14, ytop + i * RH_ROW, LINE, 0.6, arrow=False)
        c = COUNT[cat]
        n = c["bit"] + c["noise"] + c["far"] + c["step1"]
        f.t(20, yc + 5, lab, INK, True, 14)
        f.t(LBW - 10, yc + 5, "%d 个" % n, GY, size=13, anchor="end")
        for j in range(c["bit"]):
            f.box(LBW + 10 + (j % 9) * 15, yc - 12 + (j // 9) * 15, 11, 11, GY2, GY2, 2)
        if c["bit"] == 0:
            f.t(LBW + 10, yc + 5, "—", GY, size=13)
        pts = [a for a in ARMS if a["cat"] == cat and kind(a) != "bit"]
        used = {}
        for a in sorted(pts, key=lambda a: a["r300"]):
            x = X(a["r300"])
            slot = int(x // 9)
            k = used.get(slot, 0)
            used[slot] = k + 1
            yy = yc + ((k + 1) // 2) * 9 * (1 if k % 2 else -1)
            kd = kind(a)
            if kd == "noise":
                f.box(x - 4, yy - 4, 8, 8, BL, BL, 4)
            elif kd == "far":
                f.box(x - 5.5, yy - 5.5, 11, 11, RD, INK, 5.5)
            else:
                f.box(x - 5.5, yy - 5.5, 11, 11, "#fff", RD, 5.5, sw=2)
            if a["change"] in LABEL:
                txt, dx, dy, anc = LABEL[a["change"]]
                f.t(x + dx, yy + dy, txt, RD, True, 12.5, anc)
    f.t(LBW + 10, ytop - 0 + 0, "", GY)
    f.t(LBW + 4, ybot + 20, "← 逐位相同", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "不逐位不等于有 bug：换切分、换编译开关、等价改写、升运行库、换 v7x，几乎都跟重跑分不开", [
        "曲线上分得开的只有两类：在 BF16 里多加了几次的词嵌入梯度（纯数据并行、关省舍入），和故意改了精度或公式的（输出头降回单遍 BF16、两个 ε、偏置更新公式）；另有五个靠第 1 步对 FP64 抓到。",
        "代价：这把尺子也有量不出来的东西 —— 有几个动了精度的改动跑到 1000 步也分不开，所以第 1 步要对 FP64 再看一遍（红圈就是这么抓到的）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸模型，TPU v5p 8 设备，FSDP 8、每设备微批 1 × 累积 2；golden 换一组机器、换集群重跑都逐位相同）：每个改动从 golden 的命令出发只改一样，跑 300 步，"
               "对 golden 比 8 条曲线（主 loss 全程与后段、MTP、索引器蒸馏、梯度范数、参数范数、路由偏置均值与标准差）的三个统计，r 的分母是 GPU 同设置 8 份两两 28 对的上限；"
               "第 1 步 205 个梯度张量另对 CPU FP64 定责（离 FP64 比 golden 与 GPU 里远的那个还远 1.2 倍以上，记丢了精度）。",
               "📌 点的颜色按对齐记录脾气表的「和 GPU 重跑比」一列：曲线最差 r 不超过 GPU 8 份留一法最大值 2.47、且第 1 步没有离 FP64 更远的张量，记「分不开」；否则「分得开」。"
               "曲线 ≤ 2.47 而第 1 步抓到的画红圈：关小矩阵乘改写（一个已知病态的小归一化张量）、RMSNorm 多舍一次、多路残差混合不强制舍 BF16、加上下文并行、v7x 上改纯数据并行。"
               "一共 100 行，去掉故意埋 bug 的 11 行（另一张图）；另有 32 个改动跑了 1000 步，见正文。",
               "⚠️ 同一行里竖着错开只为不重叠；「上下文并行」那一行的逐位相同是它自己重跑的一次。")
    f.save("fig9-temper.svg", yb + 14)


fig_temper()
