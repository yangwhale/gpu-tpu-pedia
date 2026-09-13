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

    CW, GAP = 640, 58
    RX = CW + GAP
    PH = 512

    # ══════════ 左：四种存法 ══════════
    ay = f.panel(0, y, CW, PH, "四种存法，同一个形状", BL,
                 sub="每 token 每层要留几个数 →&#160;128K 一个用户共多少")
    f.t(16, ay + 4, "存法", GY, bold=True, size=14)
    f.t(112, ay + 4, "KV 头", GY, bold=True, size=14)
    f.t(196, ay + 4, "每 token 每层", GY, bold=True, size=14)
    f.t(330, ay + 4, "128K 单用户", GY, bold=True, size=14)
    f.t(430, ay + 4, "相对 MHA", GY, bold=True, size=14)
    mx = VAR[0][1]
    BX, BARW = 496, 62          # 真实线性刻度：488 GiB 占满 62px
    ZX, ZOOMW = 566, 62         # 放大镜：以 GQA-8 的 30.50 GiB 为满格
    f.t(BX, ay + 4, "真实比例", GY, bold=True, size=14)
    f.t(ZX, ay + 4, "放大 16×", GY2, bold=True, size=14)
    for i, (nm, per, desc, col, heads) in enumerate(VAR):
        yy = ay + 26 + i * 46
        f.t(16, yy + 4, nm, col, bold=True, size=_sz(16))
        f.t(112, yy + 4, heads, GY2, size=14)
        f.t(196, yy + 4, format(per, ","), INK, bold=True, size=14, mono=True)
        f.t(330, yy + 4, "%.2f GiB" % tot(per) if tot(per) < 100
            else "%.0f GiB" % tot(per), col, bold=True, size=_sz(15), mono=True)
        f.t(430, yy + 4, "—" if i == 0 else "省 %.0f×" % (mx / per),
            GY2, size=14)
        f.t(16, yy + 22, desc, GY2, size=14)
        bw = max(1.0, BARW * per / float(mx))
        f.box(BX, yy - 5, bw, 12, col, col, 2)
        if per != mx:                       # 放大镜：后三根用自己的基准再画一遍
            zw = max(2.0, ZOOMW * per / float(VAR[1][1]))
            f.box(ZX, yy - 5, zw, 12, "#fff", col, 2, 1.2)

    f.t(16, ay + PH - 158,
        "⭐ <tspan font-weight=\"700\">左列是真实线性比例</tspan> ——&#160;"
        "后三根细到几乎看不见，", GY2, size=14, w=CW - 32)
    f.t(16, ay + PH - 138,
        "这正是要的画面（MQA 是 MHA 的 0.78%）。", GY2, size=14, w=CW - 32)
    f.t(16, ay + PH - 118,
        "右列换了基准（满格 ＝ GQA-8 的 30.50 GiB）。", GY2, size=14, w=CW - 32)
    f.t(16, ay + PH - 74,
        '⭐⭐ <tspan font-weight="700">题眼在这儿：MQA 只要 3.81 GiB，'
        '比 MLA 的 8.58 还小 2.25 倍。</tspan>', RD, size=_sz(16))
    f.t(16, ay + PH - 54,
        'MQA 2019 年就把体积压到头了 ——&#160;'
        '<tspan font-weight="700">代价是质量掉得厉害</tspan>。', GY, size=_sz(15))
    f.t(16, ay + PH - 32,
        '⭐ 所以这一支比的<tspan font-weight="700">不是「谁存得最少」，'
        '是「同样一份字节，换回多少能力」</tspan>。', BL, size=_sz(16))

    # ══════════ 右：RoPE 为什么必须单独走一路 ══════════
    # ⭐⭐⭐ 2026-09-14 重画：原来是两行代数。现在先给一个**寄快递**的画面 ——
    #   代数留在下面当佐证，但**看懂靠的是上面那张图**。
    by = f.panel(RX, y, CW, PH, "为什么 MLA 的 RoPE 必须单独走一路", PU,
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
