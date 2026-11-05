# -*- coding: utf-8 -*-
r"""专题九 · 第二节「地基 L0」的两张图。

fig9-replay  回答：怎么保证两边吃的是同一批数据？—— GPU 训练时把每一步每张卡吃的东西原样存下、记指纹，TPU 按同样顺序回放。
             数字：1000 步 × 8 卡 × 2 个 microbatch ＝ 16,000 个单元（实测全部逐位相同）；
             换成 TPU 自己的数据加载器、不重排：内容一样，只有 2,000 个单元落到同号设备（实测）。
             ⛔ 刻意没画：原生加载器到底怎么交错取行 —— 只画结果比例，不画我们没核过的映射。
fig9-wd      回答：配置文件能直接照抄吗？—— 权重衰减写 0.1，每一步真正用的是 0.033。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE
import topic09_numbers as N

W = 1400
COLS = [BL, OR, GR, RD, "#9334e6", "#00838f", "#b06000", GY]     # 8 张卡 8 个颜色
assert len(COLS) == N.RANKS
SAME_SHARE = N.NATIVE_SAME_RANK / float(N.DATA_UNITS)
assert abs(SAME_SHARE - 1 / 8.0) < 1e-12


def fig_replay():
    f = Fig(W, "上面一排是 GPU 的 8 张卡，每张卡这一步吃两小份数据 m0 和 m1，一共 16 个单元，每个单元记一个指纹。"
               "中间一排是 TPU 的 8 个设备按同样顺序回放：设备 d 的第 m 份就是 GPU 第 d 张卡的第 m 份，16 个单元逐个比指纹，1000 步共 16,000 个全部相同。"
               "下面一条：如果改用 TPU 自己的数据加载器、不重排，内容虽然一样，只有八分之一的单元落到同号设备上。")
    y0 = f.header("数据不重读，原样回放　——　<tspan font-weight=\"700\">每一步、每张卡、每一小份都对指纹</tspan>",
                  "一格 ＝ 一张卡这一步吃的一小份数据（microbatch）；颜色 ＝ 它在 GPU 上属于哪张卡",
                  [(GR, "指纹相同"), (RD, "落到了别的设备")])
    PH = 380
    py = f.panel(0, y0, W, PH, "一步里的 16 个单元（8 张卡 × 2 小份）", BL)
    CW, CH, GAP = 54, 40, 26
    X0 = 200

    def row(y, label, sub, mark=None):
        f.t(24, y + 26, label, INK, True, 15)
        f.t(24, y + 48, sub, GY, size=13)
        for d in range(N.RANKS):
            x = X0 + d * (2 * CW + GAP + 4)
            for m in range(N.MICRO):
                f.box(x + m * (CW + 4), y + 10, CW, CH, COLS[d], COLS[d], 5)
                f.t(x + m * (CW + 4) + CW / 2, y + 36, "%d·m%d" % (d, m), "#fff", True, 13, "middle")
            if mark:
                f.t(x + CW + 2, y + CH + 36, mark, GR, True, 16, "middle")
    row(py + 20, "GPU 训练时", "存下每一格 ＋ 指纹")
    for d in range(N.RANKS):
        x = X0 + d * (2 * CW + GAP + 4) + CW + 2
        f.line(x, py + 76, x, py + 116, GY2, 1.4)
    row(py + 120, "TPU 回放", "设备 d 拿 GPU 第 d 卡", mark="✓")
    # 原生加载器：比例条
    by = py + 260
    f.t(24, by + 22, "TPU 原生加载器", INK, True, 15)
    f.t(24, by + 44, "不重排：内容同，位置错", GY, size=13)
    BW = 1100
    f.box(X0, by + 6, BW * SAME_SHARE, 34, GR, GR, 4)
    f.box(X0 + BW * SAME_SHARE, by + 6, BW * (1 - SAME_SHARE), 34, "#fce8e6", RD, 4)
    f.t(X0 + 10, by + 70, "2,000 / 16,000 落到同号设备", GR, True, 14)
    f.t(X0 + BW, by + 70, "其余 14,000 个落到别的设备 → 加一步重排，才全部逐位", RD, True, 14, "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "数据这一级必须一位不差：1000 步 × 8 卡 × 2 小份 ＝ 16,000 个单元，逐个比", [
        "两边吃的东西完全一样，后面量出来的差才是「算」的差，不是「喂」的差。",
        "代价：GPU 那边要多存一份每步的输入；TPU 训练时不走自己的数据管线（生产管线另外单独验）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：每个单元记 tokens、labels、loss_mask、position_ids 各一个 sha256，另比每步非 padding 的 token 总数。",
               "📌 实测：TPU 原生加载器读同一份数据，索引表逐字节相同、样本流逐位相同；按原样布局只有 2,000 / 16,000 个单元落到同号设备（两边梯度累积切分的方式不同），重排后全部逐位。")
    f.save("fig9-replay.svg", yb + 14)


def fig_wd():
    f = Fig(W, "权重衰减的配置文件里写 0.1，灰色虚线。可是训练框架的学习率调度器每一步都用自己的起止值把它覆盖掉，"
               "每一步真正用的是 0.033，绿色实线。TPU 那边要按 0.033 配；照抄 0.1，每一步就多衰减三倍。")
    y0 = f.header("配置文件写的，不一定是实际用的　——　<tspan font-weight=\"700\">权重衰减：写着 0.1，用的是 0.033</tspan>",
                  "横轴 ＝ 训练步；纵轴 ＝ 这一步优化器实际用的权重衰减系数",
                  [(GY, "配置里写的"), (GR, "每一步实际用的")])
    PH = 300
    py = f.panel(0, y0, W, PH, "同一个超参，两处写法", GR)
    X0, XW, YB, YH = 120, 1180, py + 250, 200

    def Y(v):
        return YB - v / 0.12 * YH
    f.line(X0, YB, X0 + XW, YB, GY2, 1.2)
    f.line(X0, YB, X0, py + 40, GY2, 1.2)
    for v in (0.0, 0.033, 0.1):
        f.t(X0 - 12, Y(v) + 5, "%g" % v, GY, size=13, anchor="end")
    f.line(X0, Y(N.WD_CONFIG), X0 + XW, Y(N.WD_CONFIG), GY2, 1.6, dash="8 6", arrow=False)
    f.t(X0 + 20, Y(N.WD_CONFIG) - 12, "配置文件：weight_decay = 0.1（只是摆设）", GY, True, 14)
    f.line(X0, Y(N.WD_EFFECTIVE), X0 + XW, Y(N.WD_EFFECTIVE), GR, 4, arrow=False)
    f.t(X0 + 20, Y(N.WD_EFFECTIVE) + 28, "调度器每一步覆盖成 start = end = 0.033 → 实际生效", GR, True, 14)
    for k in range(1, 12):
        x = X0 + k * XW / 12.0
        f.line(x, Y(N.WD_CONFIG) + 6, x, Y(N.WD_EFFECTIVE) - 6, GR, 1.2)
    f.t(X0 + XW - 10, Y(N.WD_CONFIG) - 12, "TPU 要是照抄 0.1：每一步多衰减 %.0f 倍" % (N.WD_CONFIG / N.WD_EFFECTIVE), RD, True, 14, "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "对照表要写「实际生效的值」和证据，不写「配置里的值」", [
        "每个影响数值的设置一行：GPU 怎么配、TPU 怎么配、是否等价、证据从哪来（运行时读出的元数据，不是 yaml）。",
        "代价：要去读框架真正跑的那条代码路径；只读配置，一个超参能对错三倍还不报错。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：GPU 侧配置 weight_decay: 0.1；调度器 start_weight_decay = end_weight_decay = 0.033、constant、override_opt_param_scheduler=True，运行元数据记录的就是 0.033。TPU 侧配 adam_weight_decay: 0.033。",
               "⚠️ 本课推导：0.1 ÷ 0.033 ≈ 3。")
    f.save("fig9-wd.svg", yb + 14)


fig_replay()
fig_wd()
