# -*- coding: utf-8 -*-
r"""专题九 · 第六节「观测」的一张图。

fig9-hooks  回答：为什么 GPU 上随手加个钩子抓数没事，TPU 上不行？TPU 上的第 1 步证据怎么拿？
            GPU：每个 kernel 的输出本来就写回显存，钩子读的是已经落地的数（实测：确定性设置下开不开钩子逐步逐位相同）。
            TPU：整张图一起编译，多一个输出就改了融合，舍入位置跟着变（实测：v5p 上加钩子的程序和生产程序不逐位）。
            三条干净的路：步与步之间读状态 ＋ 从 Adam 一阶动量反推梯度；单独编译的隔离回放；单独编译的固定探针。
            反推公式 g ＝ (m_k − β₁·m_{k−1}) ÷ (1 − β₁) ÷ 裁剪系数 是本课推导（由 Adam 一阶动量的定义倒过来写）。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
import topic09_numbers as N

W = 1400

# 反推公式自检：随便造一组数，正着算一步 m，再倒回来
_g, _m0, _c = 0.37, -0.12, N.CLIP1
_m1 = N.BETA1 * _m0 + (1 - N.BETA1) * (_g * _c)
assert abs((_m1 - N.BETA1 * _m0) / (1 - N.BETA1) / _c - _g) < 1e-12


def fig_hooks():
    f = Fig(W, "左格 GPU：一个个算子按顺序跑，每个 kernel 的输出本来就写回显存，钩子读的是已经落地的数，数不变。"
               "中格 TPU：整张计算图一起编译，编译器会把相邻的算子融成一块、省掉中间的舍入，红色小块是一次舍到 BF16。生产程序四个算子融成一块，只在出口舍 1 次；"
               "加一个输出，融合在钩子处被切成两块，切口多舍一次，共 2 次，舍入位置跟着变，量出来的已经不是生产那份数。"
               "右格：TPU 上三条不扰动的路。一，步与步之间读状态：主权重、两个动量，再从一阶动量倒推出这一步的梯度；二，隔离回放：拿 GPU 的输入单独编译一层或一个算子来跑；三，固定探针：每隔几步在一条固定的序列上单独跑一次前向。")
    y0 = f.header("在 TPU 上「看一眼」会改掉你要看的东西　——　<tspan font-weight=\"700\">证据要从程序外面拿</tspan>",
                  "方块 ＝ 算子；虚框 ＝ 编译器融成的一块；红色小块 ＝ 一次舍到 BF16；放大镜 ＝ 抓数的钩子",
                  [(GY, "GPU 的算子"), (BL, "TPU 的算子"), (RD, "一次舍到 BF16"), (GR, "钩子不改数值"), (OR, "！钩子改了数值")])
    PH = 400

    def lens(cx, cy, col):
        """放大镜 ＝ 钩子（两格同一个画法）。"""
        f.p.append('<circle cx="%d" cy="%d" r="16" fill="#fff" stroke="%s" stroke-width="3"/>' % (cx, cy, col))
        f.line(cx + 12, cy + 12, cx + 26, cy + 28, col, 3, arrow=False)
    # GPU：算子是 GPU 的深灰，钩子绿（不改数值）
    py = f.panel(0, y0, 420, PH, "GPU：钩子读的是已落地的数", GY)
    for i, lab in enumerate(["矩阵乘", "归一化", "激活", "矩阵乘"]):
        x = 30 + i * 92
        f.box(x, py + 90, 78, 46, LINE2, GY, 6)
        f.t(x + 39, py + 118, lab, GY, True, 13, "middle")
        if i < 3:
            f.line(x + 78, py + 113, x + 92, py + 113, GY2, 1.4)
    lens(174, py + 186, GR)
    f.line(174, py + 168, 174, py + 140, GR, 1.4)
    # 「只是读一下」没说为什么不扰动 → 写明读的是本来就落地的数
    f.t(24, py + 238, "每个 kernel 的输出本来就写回显存，", GR, True, 14)
    f.t(24, py + 260, "钩子读的是已经落地的数", GR, True, 14)
    f.t(24, py + 300, "开不开钩子：逐步逐位相同", GR, True, 15)
    f.t(24, py + 326, "（确定性设置下实测）", GY, size=13)
    f._pan = None
    # TPU：算子一律 TPU 蓝；红色小块 ＝ 一次舍到 BF16（照 fig9-excess 的画法）；「钩子改了数值」另用橙色 ！ 标，不涂红
    px = 440
    py = f.panel(px, y0, 420, PH, "TPU：多一个输出，融合就变了", BL)
    OW = 66

    def cut(x, y):
        """一次舍到 BF16：红色小块 ＋「舍」字（跟 fig9-excess 一样）。"""
        f.box(x, y + 3, 12, 38, "#fce8e6", RD, 3)
        f.t(x + 6, y + 62, "舍", RD, True, 12, "middle")

    def bang(cx, cy):
        """钩子改了数值：橙色圆 ＋ 感叹号。"""
        f.p.append('<circle cx="%d" cy="%d" r="11" fill="%s"/>' % (cx, cy, OR))
        f.t(cx, cy + 5, "!", "#fff", True, 15, "middle")
    f.t(px + 24, py + 50, "生产程序：舍 1 次", INK, True, 14)
    f.box(px + 24, py + 64, 348, 60, "none", GY2, 10, dash="6 5")
    for i, lab in enumerate(["矩阵乘", "归一化", "激活", "矩阵乘"]):
        xx = px + 32 + i * 86
        f.box(xx, py + 72, OW, 44, "#e8f0fe", BL, 6)
        f.t(xx + OW / 2, py + 99, lab, BL, True, 13, "middle")
    cut(px + 380, py + 72)
    f.t(px + 24, py + 168, "加了钩子的程序：舍 2 次", INK, True, 14)
    f.box(px + 24, py + 182, 162, 60, "none", GY2, 10, dash="6 5")
    f.box(px + 210, py + 182, 162, 60, "none", GY2, 10, dash="6 5")
    for i, lab in enumerate(["矩阵乘", "归一化", "激活", "矩阵乘"]):
        xx = px + 32 + i * 80 + (26 if i >= 2 else 0)
        f.box(xx, py + 190, OW, 44, "#e8f0fe", BL, 6)
        f.t(xx + OW / 2, py + 217, lab, BL, True, 13, "middle")
    cut(px + 192, py + 190)
    cut(px + 380, py + 190)
    bang(px + 372, py + 182)
    # 切口处的放大镜 ＝ 钩子；改了数值 → 橙色
    f.line(px + 198, py + 262, px + 198, py + 272, OR, 1.4, arrow=False)
    lens(px + 198, py + 290, OR)
    f.t(px + 236, py + 290, "多一个输出：融合切开", OR, True, 13)
    f.t(px + 236, py + 310, "切口多舍一次", OR, True, 13)
    bang(px + 36, py + 342)
    f.t(px + 54, py + 347, "和生产程序不再逐位（v5p 实测）", OR, True, 15)
    f._pan = None
    # 三条路：每条只留一行（细节在正文三小节）
    px = 880
    py = f.panel(px, y0, W - px, PH, "TPU 上三条不扰动的路", GR)
    items = [("① 步与步之间读状态", "主权重、两个动量；梯度从动量倒推"),
             ("② 隔离回放", "拿 GPU 的输入，单独编译一层来跑"),
             ("③ 固定探针", "每 25 步单独编译跑一次前向")]
    y = py + 44
    for t, ln in items:
        f.t(px + 20, y, t, GR, True, 15)
        f.t(px + 36, y + 24, ln, INK, size=13, w=W - px - 50)
        y += 62
        if t.startswith("①"):
            f.t(px + 36, y - 10, "g ＝ (mₖ − 0.9·mₖ₋₁) ÷ 0.1 ÷ 裁剪系数", PU, True, 14, mono=True)
            y += 34
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "规矩：生产程序本身就是测量对象，证据从它外面拿", [
        "倒推出来的梯度在 GPU 上对过实际抓到的梯度，相对差 2e-7 到 4e-7 —— 跟直接抓一样准，而且一位都不扰动。",
        "代价：每一步都加钩子的「观测版」程序要先当扰动臂过标定，v5p 上没过，只能当诊断值，不能当正式证据。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导：Adam 一阶动量 mₖ ＝ β₁·mₖ₋₁ ＋ (1 − β₁)·gₖ（gₖ 是裁剪之后的梯度），倒过来 gₖ ＝ (mₖ − β₁·mₖ₋₁) ÷ (1 − β₁)；再除以裁剪系数得到裁剪之前的梯度。β₁ ＝ 0.9。",
               "📌 实测：GPU 上反推梯度对抓到的梯度 rel_L2 2e-7 – 4e-7；TPU 上固定探针单独编译，训练轨迹和不开探针时 1000 步逐位相同。",
               "📌 实测：v5p 上加抓数输出的训练程序和生产程序不逐位；XLA 默认在融合内部跳过中间的 BF16 舍入（excess precision），融合一变，舍入位置就变。",
               "注：中格四个算子、融合切成几块、舍几次都是示意（跟 fig9-excess 同一种画法）；真实的融合怎么切由编译器决定。")
    f.save("fig9-hooks.svg", yb + 14)


fig_hooks()
