# -*- coding: utf-8 -*-
r"""专题三 · §五「旋钮①：让每一份更小」的图（2026-09-12 加）。

⭐ 两件事，一张图：

  左 · **四种存法摆在同一个形状下** ——&nbsp;题眼是那个反直觉：
       **MQA 比 MLA 还小**。所以这一支比的从来不是「谁存得少」，
       是「同样一份字节换回多少能力」。

  右 · **为什么 RoPE 必须单独走一路** ——&nbsp;讲义写着这是
       「理解 MLA 的关键一步，也是最容易讲糊的一步」。
       它本质是**一个代数重写成不成立**的问题，
       ⭐ 而「成不成立」画出来比说出来清楚得多。

📌 数字与 §二 那张图、以及课前第一题同源，脚本里当场算并断言。
"""
from topic03_draw import (Fig, wpx, _sz,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
GiB = 2 ** 30


def main():
    L, T, B = 61, 131072, 2
    def tot(per): return per * B * L * T / GiB
    VAR = [
        ("MHA", 2 * 128 * 128, "每个头各存一份 K/V", GY, "128 份"),
        ("GQA-8", 2 * 8 * 128, "分成 8 组，组内共用", OR, "8 份"),
        ("MLA", 512 + 64, "压成 512 维隐向量 ＋ 64 维 RoPE", GR, "不按头存"),
        ("MQA", 2 * 1 * 128, "所有头共用同一份", RD, "1 份"),
    ]
    assert abs(tot(VAR[0][1]) - 488) < 1 and abs(tot(VAR[2][1]) - 8.58) < .05
    assert tot(VAR[3][1]) < tot(VAR[2][1])          # ⭐ 题眼：MQA 比 MLA 还小

    f = Fig(W, "旋钮一：让每一份更小。四种存法在同一形状下的对照 —— "
               "MQA 比 MLA 存得还少，所以这一支比的不是谁存得少；"
               "右边讲为什么 MLA 的 RoPE 必须单独走一路")
    f.marks = set()
    y = f.header(
        '旋钮 ① 让每一份更小 ——&#160;'
        '<tspan font-weight="700">但比的从来不是「谁存得最少」</tspan>',
        "同一个形状（V3：61 层 · 128 头 · 每头 128 维 · 128K · bf16），换四种存法")

    # ⭐⭐⭐ 2026-09-14 重画 ①。审图原话：「左半张是一张**六列表格**，
    #   最后那列的柱子自己承认『后三根细到几乎看不见』。」
    # ⭐ 真正的题眼是**二维**的：MQA 比 MLA 还小，**但更差**。
    #   一维表格画不出「又小又差」——&nbsp;所以换成散点：
    #   横轴「一份占多少地方」，纵轴「换回多少能力」，四个点一摆自己就说话了。
    # ⛔⛔ 纵轴必须诚实：**四家没有同一份可比的实测**。
    #   唯一同基准的一对是 MQA 论文表 3（MHA 29.9 / MQA 30.2 / 真单头 31.2）；
    #   GQA 与 MLA 各自的论文只声称「接近／不弱于 MHA」，不是同一张表。
    #   所以纵轴画成**两档定性**，并在图上写死这一句 ——
    #   画成连续刻度等于编出一份不存在的实测。
    CW = W
    PH = 420

    # ══════════ ① 四种存法：一张散点 ══════════════════════════
    ay = f.panel(0, y, CW, PH, "四种存法 ——　把它们摆到一张图上", BL,
                 sub="横轴：一份占多少地方（对数）　·　纵轴：换回多少能力（⚠️ 定性）")

    X0, X1 = 210, 940
    YT, YB = ay + 40, ay + 250
    import math as _m
    def sx(g):                     # 1 ～ 1000 GiB，三个数量级
        return X0 + (X1 - X0) * _m.log10(max(g, 1.0)) / 3.0
    # 两档定性分区
    f.spot(X0 - 40, YT, X1 - X0 + 120, 96, "#e6f4ea")
    f.spot(X0 - 40, YT + 120, X1 - X0 + 120, 90, "#fce8e6")
    f.t(X0 - 30, YT + 26, "跟 MHA 基本打平", GR, True, 18)
    f.t(X0 - 30, YT + 146, "明显更差", RD, True, 18)
    f.line(X0 - 40, YB + 6, X1 + 80, YB + 6, LINE, 1.4, arrow=False)
    for g, lab in ((3.0, "3"), (10, "10"), (30, "30"), (100, "100"),
                   (488, "488")):
        f.line(sx(g), YB + 6, sx(g), YB + 12, GY2, 1.2, arrow=False)
        f.t(sx(g), YB + 32, lab, GY2, size=15, anchor="middle")
    f.t(X1 + 20, YB + 32, "GiB ——　越右边越占地方", GY2, size=16)

    PTS = [("MHA", 488.0, 1, GY, "每个头各存一份"),
           ("GQA-8", 30.50, 1, OR, "8 组，组内共用"),
           ("MLA", 8.58, 1, GR, "压成 512 ＋ 64"),
           ("MQA", 3.81, 0, RD, "所有头共用一份")]
    for nm, g, hi, col, desc in PTS:
        cy = YT + (48 if hi else 166)
        f.box(sx(g) - 9, cy - 9, 18, 18, col, col, 9)
        f.t(sx(g), cy - 22, nm, col, True, 21, "middle")
        f.t(sx(g), cy + 34, "%.2f GiB" % g if g < 100 else "%.0f GiB" % g,
            col, True, 17, "middle")
        f.t(sx(g), cy + 56, desc, GY2, size=15, anchor="middle")

    # 题眼：从 MQA 指到 MLA 的那一段
    # ⛔ 题眼这行别跟 MQA 的数值标签抢同一条基线 —— 压到分区底部去
    f.line(sx(3.81), YT + 132, sx(8.58), YT + 78, RD, 2.0)
    f.t(sx(30.5), YT + 196, "⭐⭐ 题眼：MQA 比 MLA 还小 2.25 倍，"
        "<tspan font-weight=\"700\">却更差</tspan>", RD, True, 19)

    f.box(1076, ay + 30, 308, 232, "#fff", INK, 10)
    f.t(1096, ay + 62, "⭐ 所以这一支比的", INK, True, 20)
    f.t(1096, ay + 88, "不是「谁存得最少」", INK, True, 20)
    f.t(1096, ay + 124, "MQA 早在 2019 年就把", GY, size=17, w=272)
    f.t(1096, ay + 148, "体积压到头了。", GY, size=17, w=272)
    f.t(1096, ay + 184, "⭐ 要比的是：同样一份", BL, True, 18, w=272)
    f.t(1096, ay + 208, "字节，换回多少能力。", BL, True, 18, w=272)
    f.t(1096, ay + 244, "（这正是第五节的线）", GY2, size=15)

    f.t(16, ay + 300, "⚠️ <tspan font-weight=\"700\">纵轴是定性的</tspan>"
        " ——　四家<tspan font-weight=\"700\">没有同一份可比的实测</tspan>。"
        "唯一同基准的一对是 MQA 论文表 3："
        "MHA 29.9 ／ MQA 30.2 ／ 真单头 31.2（困惑度，越低越好）；",
        GY, size=17, w=1368)
    f.t(16, ay + 326, "GQA 与 MLA 各自的论文只声称「接近／不弱于 MHA」，"
        "不是同一张表 ——　所以这里只画两档，不画连续刻度。", GY, size=17,
        w=1368)
    f.t(16, ay + 362, "📐 横轴那几个数：61 层 · 128K · bf16 · batch 1，"
        "一个用户一份 ——　MHA 488 GiB ／ GQA-8 30.50 ／ MLA 8.58 ／ MQA 3.81。",
        GY2, size=16, w=1368)

    # ══════════ 右：RoPE 为什么必须单独走一路 ══════════
    # ⭐⭐⭐ 2026-09-14 重画：原来是两行代数。现在先给一个**寄快递**的画面 ——
    #   代数留在下面当佐证，但**看懂靠的是上面那张图**。
    y = y + PH + 18
    RX = 0
    PH = 512
    by = f.panel(RX, y, W, PH, "为什么 MLA 的 RoPE 必须单独走一路", PU,
                 sub="先看一个寄快递的画面")

    ey = by + 10
    # ① 没有 RoPE：仓库存压缩包，收件人自己拆
    f.t(RX + 16, ey + 22, "① 没有 RoPE：仓库只存压缩包", GR, True, _sz(15))
    f.box(RX + 16, ey + 34, 120, 52, "#e6f4ea", GR, 8)
    f.t(RX + 76, ey + 66, "压缩包", GR, True, _sz(14), "middle")
    f.line(RX + 144, ey + 60, RX + 188, ey + 60, GY2, 1.6)
    f.box(RX + 196, ey + 34, 150, 52, "#fff", GR, 8)
    f.t(RX + 271, ey + 56, "收件人自己拆", GY, True, _sz(13), "middle")
    f.t(RX + 271, ey + 78, "（吸收进 q 那一侧）", GY2, size=_sz(14), anchor="middle")
    f.t(RX + 366, ey + 66, "✅ 仓库省地方", GR, True, _sz(14))
    ey += 104

    # ② 有 RoPE：中间多一道必须在仓库做的工序
    f.t(RX + 16, ey + 22, "② 插进 RoPE：中间多了一道仓库必须做的工序", RD, True,
        _sz(15))
    f.box(RX + 16, ey + 34, 120, 52, "#fce8e6", RD, 8)
    f.t(RX + 76, ey + 66, "压缩包", RD, True, _sz(14), "middle")
    f.line(RX + 144, ey + 60, RX + 176, ey + 60, RD, 1.6)
    f.box(RX + 182, ey + 34, 116, 52, "#fff", RD, 8)
    f.t(RX + 240, ey + 58, "按位置转一下", RD, True, _sz(13), "middle")
    f.t(RX + 240, ey + 80, "（那个 R）", GY2, size=_sz(14), anchor="middle")
    f.line(RX + 306, ey + 60, RX + 338, ey + 60, RD, 1.6)
    f.box(RX + 344, ey + 34, 150, 52, "#fff", RD, 8)
    f.t(RX + 419, ey + 66, "只好先拆开再存", RD, True, _sz(13), "middle")
    f.t(RX + 16, ey + 108, "⛔ 这道工序卡在中间，"
        "<tspan font-weight=\"700\">拆包这件事就挪不走了</tspan>", GY, size=_sz(16))
    ey += 130

    # ③ 解法：把要贴标签的那一小部分单拿出来
    f.box(RX + 16, ey, CW - 32, 96, "#f3e8fd", PU, 8)
    f.t(RX + 32, ey + 26, "③ 解法：把「必须在仓库做工序」的那一小部分单拿出来",
        PU, True, _sz(13.5))
    f.t(RX + 32, ey + 50,
        '<tspan font-weight="700">512 维</tspan>不带位置 → 照旧只存压缩包、拆包挪给收件人',
        INK, size=_sz(15))
    f.t(RX + 32, ey + 72,
        '<tspan font-weight="700">64 维</tspan>专扛 RoPE → 老老实实拆开存　'
        '＝　一共 <tspan font-weight="700">576</tspan>', INK, size=_sz(15))
    ey += 108

    # 代数留作佐证，放在最后、字小一号
    f.t(RX + 16, ey + 18, "📌 代数上就是这两行：", GY2, size=_sz(14.5))
    f.box(RX + 16, ey + 26, CW - 32, 28, "#f8f9fa", LINE, 6)
    f.t(RX + 32, ey + 45, "qᵀ (W_UK c) ＝ (W_UKᵀ q)ᵀ c", GR, bold=True,
        size=_sz(16), mono=True)
    f.t(RX + 300, ey + 45, "✅ 挪得走", GR, size=_sz(14.5))
    f.box(RX + 16, ey + 60, CW - 32, 28, "#f8f9fa", LINE, 6)
    f.t(RX + 32, ey + 79, "qᵀ R (W_UK c)", RD, bold=True, size=_sz(16),
        mono=True)
    f.t(RX + 300, ey + 79, "⛔ R 夹在中间，分不开", RD, size=_sz(14.5))

    yy = y + PH + 18
    yy = f.band(yy, "info", "⭐ 把 5.3 当成一个套路记住，别当成 MLA 的实现细节", [
        '<tspan font-weight="700">「为了保住某个代数变换，把功能拆成两路」</tspan>'
        '——&#160;这个动作后面还会换面貌出现：'
        '<tspan font-weight="700">V4 的部分 RoPE、K3 的 NoPE</tspan>。',
        '⛔ <tspan font-weight="700">MLA 的代价是用计算换显存</tspan>：多了一对降维／升维矩阵乘。'
        '而且<tspan font-weight="700">它省的是推理时的 KV cache，不是训练时的激活</tspan>'
        '——&#160;训练前向里 K/V 会被解压出来算。<tspan font-weight="700">这是个非常常见的误解。</tspan>',
        '⭐ <tspan font-weight="700">Gated MLA</tspan>（K3）在 MLA 输出端加一个全秩门控 —— '
        '⛔ <tspan font-weight="700">它一个字节的 KV 都不省</tspan>，'
        '它让模型能学会「这一层这个位置，注意力的输出干脆不要」。'])

    yy = f.src(yy + 16,
               "四个数由公式当场算出并断言（脚本内）；MLA 超参出自 V3 论文 §4.2："
               "n_h=128, d_h=128, d_c=512, d_h^R=64, 61 层",
               "MQA：Shazeer arXiv 1911.02150　GQA：Ainslie 等 arXiv 2305.13245　"
               "MLA：DeepSeek-V2/V3 arXiv 2412.19437　Gated MLA：Kimi K3 技术报告 §2.1.2")
    f.save("fig3-knob1.svg", yy + 6)


main()
