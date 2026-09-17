# -*- coding: utf-8 -*-
r"""专题四 · §4.2「ZeRO 的三级」——&#160;四行条，一眼看完

⭐⭐⭐ 2026-09-17 新画。现场原话把 ZeRO 定成了**正课内容**而不是一条出处：
  「其实是训练的主线、很重要的一个论文」。而这一节在图上**一格都没有**，
  只有一个三条的列表。⛔ 一个被点名为「主线」的东西，不该只有文字。

⭐⭐ 这张图的取舍：**画「每张卡还剩多少」，不画「怎么切」。**
  ⛔ 讲 ZeRO 最常见的画法是三张分片示意图（谁拿哪一块）——&#160;
    那个画面解释的是**实现**，而这一讲从头到尾在讲**账**。
  ⭐ 而账的画面只有一个：**同一根条，被削掉三次。**
    削完还剩多少，直接决定「一张卡装不装得下」——&#160;这才是读者要的那个数。

⭐⭐⭐ Ⓐ 的四个数**全部由公式算出来，不是抄的**，而且跟论文 Figure 1 的
  四个数字对得上（7.5B 模型、64 路数据并行：120 / 31.4 / 16.6 / 1.9 GB）。
  ⭐ 这是这张图最值钱的一点：**判据是我们自己推的，推出来的数跟人家印的一样。**
  ⛔ 判据（跟 §2.3 那张排序表同一条）：**能算的就别抄。**
    抄来的数字对不上时你不知道是谁错了；算出来的数字对不上，当场就报错。

📌 出处：Rajbhandari 等，arXiv 1910.02054。
  · 每参数 16 字节（2Ψ ＋ 2Ψ ＋ 12Ψ）与 Figure 1 的四个数 —— §3
  · 通信量：基线 DP 与 Pos / Pos+g 都是 2Ψ；Pos+g+p 是 3Ψ，即 1.5 倍 —— §7.2
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

PSI = 7.5e9        # 论文 Figure 1 用的模型规模
ND = 64            # 论文 Figure 1 用的数据并行路数

# (名称, 权重每参数字节, 梯度每参数字节, 优化器每参数字节)　——　切过的除以 ND
LEVELS = (
    ("基线　纯数据并行", "每张卡都存一整份", 2.0, 2.0, 12.0, GY2, "#f1f3f4",
     "2Ψ", "—"),
    ("ZeRO-1　切优化器状态", "P<tspan baseline-shift=\"-22%\" font-size=\"9\">os</tspan>",
     2.0, 2.0, 12.0 / ND, BL, "#e8f0fe", "2Ψ", "跟基线一样"),
    ("ZeRO-2　再切梯度", "P<tspan baseline-shift=\"-22%\" font-size=\"9\">os+g</tspan>",
     2.0, 2.0 / ND, 12.0 / ND, GR, "#e6f4ea", "2Ψ", "跟基线一样"),
    ("ZeRO-3　最后才切权重", "P<tspan baseline-shift=\"-22%\" font-size=\"9\">os+g+p</tspan>"
     "　≈ FSDP",
     2.0 / ND, 2.0 / ND, 12.0 / ND, OR, "#fef7e0", "3Ψ", "⛔ 1.5 倍"),
)

# ⭐ 跟论文 Figure 1 对账 —— 对不上就别构建。
_GB = [(w + g + o) * PSI / 1e9 for _, _, w, g, o, _, _, _, _ in LEVELS]
for _got, _want in zip(_GB, (120.0, 31.4, 16.6, 1.9)):
    assert abs(_got - _want) < 0.06, \
        "跟论文 Figure 1 对不上：算出 %.2f GB，论文印的是 %.1f GB" % (_got, _want)


def main():
    f = Fig(W, "ZeRO 的三级画成四根横条。基线那根最长，"
               "每张卡都要存一整份的权重、梯度和优化器状态；"
               "往下每一级削掉一块，优化器状态先切、梯度次之、权重最后。"
               "右边标着每张卡实际还剩多少显存，"
               "以及这一级的通信量是基线的几倍")

    y0 = f.header(
        "ZeRO 的三级　——　<tspan font-weight=\"700\">"
        "同一根条，被削掉三次</tspan>",
        "⭐ 顺序不用背：<tspan font-weight=\"700\">谁最大、谁最少被用到，就先切谁</tspan>",
        [(PU, "权重"), (BL, "梯度"), (RD, "优化器状态")])

    # ══════════ Ⓐ 四行条 ═════════════════════════════════════════════
    PH = 396
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 横条的长度 ＝ "
                 "<tspan font-weight=\"700\">每张卡真正要装下的量</tspan>", RD,
                 sub="⭐ 按 <tspan font-weight=\"700\">75 亿参数、64 路数据并行</tspan>"
                     "算 ——&#160;论文 Figure 1 用的就是这组")

    LX, LW = 360, 700                 # 条的起点与「基线 ＝ 满格」的宽度
    PXB = LW / 16.0                   # 每字节多少像素 —— ⭐ 四行共用这一个常数
    ROWH, BH = 82, 40

    for i, (name, tag, wb, gb, ob, col, fill, vol, volnote) in enumerate(LEVELS):
        ry = py + 52 + i * ROWH
        f.t(24, ry + 20, name, col, True, 15)
        f.t(24, ry + 42, tag, GY2, size=12)

        x = LX
        for val, c, lab in ((wb, PU, "权重"), (gb, BL, "梯度"), (ob, RD, "优化器状态")):
            bw = val * PXB
            f.box(x, ry, max(bw, 1.6), BH, c, c, 3)
            if bw >= 56:                       # 太窄就不往里塞字
                f.t(x + bw / 2, ry + 26, lab, "#fff", True, 12, "middle")
            x += bw + 2

        # 削掉的那部分：画成虚线的空框，让「削掉了多少」看得见
        if x < LX + LW:
            f.box(x + 2, ry, LX + LW - x - 2, BH, "none", GY2, 3, dash="4 4")
            if i:
                f.t((x + LX + LW) / 2, ry + 26, "切走了", GY2, size=11.5, anchor="middle")

        gbv = (wb + gb + ob) * PSI / 1e9
        f.t(LX + LW + 24, ry + 18, "%.1f GB" % gbv, col, True, 16)
        f.t(LX + LW + 24, ry + 38, "每张卡", GY2, size=11.5)
        f.t(LX + LW + 178, ry + 18, vol, INK if i < 3 else OR, True, 15)
        f.t(LX + LW + 178, ry + 38, volnote, GY2 if i < 3 else OR, size=11.5)

    f.t(LX + LW + 24, py + 34, "还要装多少", GY, True, 12)
    f.t(LX + LW + 178, py + 34, "通信量", GY, True, 12)
    f.t(LX, py + 34, "⭐ 三段的长度按<tspan font-weight=\"700\">同一个换算常数</tspan>画，"
        "所以长短可以直接比", GY2, size=12)

    f.t(24, py + 368, "⭐⭐⭐ 这四个数<tspan font-weight=\"700\">不是抄的，是算的</tspan>"
        "　——　脚本里带 assert 跟论文 Figure 1 对账，对不上就不让构建。"
        "<tspan font-weight=\"700\">能算的就别抄。</tspan>", INK, size=13.5)
    f._pan = None

    # ══════════ Ⓑ 为什么是这个顺序 ══════════════════════════════════
    PH2 = 366
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐ 顺序不是历史巧合　——　"
                  "<tspan font-weight=\"700\">是「多大」和「多久用一次」排出来的</tspan>", BL,
                  sub="⛔ 所以这三级<tspan font-weight=\"700\">不能换顺序</tspan>"
                      "，也不用背")

    # ⛔⛔ 2026-09-18 R11 重画。原来是**三张并排的卡片**，每张列
    #   「占多少字节 / 多久用一次 / 所以第几个切」。
    #   ⭐ 可这一格的论证本来就是**二维的**：一个轴是「多大」，
    #     另一个轴是「多久用一次」——&#160;三样东西落在这个平面上的**位置**，
    #     自己就把顺序排出来了。卡片把二维压成了三段文字，白扔掉一个维度。
    # ⭐⭐ 判据：**当一件事是「按两个指标排序」时，它就该是一张二维图，
    #   而不是三张卡** ——&#160;卡片只能告诉你结论，位置能让你自己看出结论。
    ITEMS = (
        (RD, "优化器状态", 12.0, 1.0, "每步只用一次", "更新那一瞬间", "通信不变"),
        (BL, "梯度", 2.0, 1.0, "反向结束汇总一次", "在那之前可以散着放", "通信不变"),
        (PU, "权重", 2.0, 122.0, "每层前向都要", "反向还要再来一次", "⛔ 通信 ×1.5"),
    )
    # ⭐ 权重那个 122 ＝ 61 层 × 前向后向各一次 ——&#160;不是随手写的
    assert ITEMS[2][3] == 2 * 61, "权重每步的取用次数应当是 61 层 × 2"
    # ⭐⭐ 这一格的全部论证：**按「先大后忙」排出来的顺序，正好就是 ZeRO 的三级**
    _order = sorted(range(3), key=lambda i: (ITEMS[i][3], -ITEMS[i][2]))
    assert _order == [0, 1, 2], "先切又大又闲的 ——　排出来必须正好是 ZeRO 的顺序"

    OX, OY = 330, py2 + 252
    AW, AH = 690, 178
    f.line(OX, OY, OX + AW + 40, OY, GY2, 1.2, arrow=False)
    f.line(OX, OY, OX, OY - AH - 26, GY2, 1.2, arrow=False)
    f.t(OX + AW + 46, OY + 5, "每步用几次 →", GY2, size=12.5)
    f.t(OX + 6, OY - AH - 32, "↑ 每参数几字节", GY2, size=12.5)

    def sx(n):
        return OX + AW * math.log10(n) / math.log10(200.0)

    def sy(b):
        return OY - AH * b / 14.0

    for n, lab in ((1, "1 次"), (10, "10"), (100, "100")):
        f.line(sx(n), OY - 5, sx(n), OY + 5, GY2, 1, arrow=False)
        f.t(sx(n), OY + 24, lab, GY2, size=12, anchor="middle")
    for bb in (2, 12):
        f.line(OX - 5, sy(bb), OX + 5, sy(bb), GY2, 1, arrow=False)
        f.t(OX - 14, sy(bb) - 12, "%d B" % bb, GY2, size=12, anchor="end")

    pts = []
    for col, name, byt, freq, when, when2, comm in ITEMS:
        x, yy = sx(freq), sy(byt)
        r = 9 + 3.4 * math.sqrt(byt)          # ⭐ 泡泡大小也按字节数，重复编码「多大」
        f.box(x - r, yy - r, 2 * r, 2 * r, "#fff", col, int(r), sw=2.4)
        pts.append((x, yy, col, name, when, comm))

    # ⭐⭐⭐ 切的顺序 ＝ 沿着这条折线走：先往下（按大小），再往右（按频率）
    for k in range(len(pts) - 1):
        x0, y0, _, _, _, _ = pts[k]
        x1, y1, _, _, _, _ = pts[k + 1]
        f.line(x0, y0 + 26, x1, y1 - 26, GY2, 1.6, dash="5 3")

    # ⛔ 第一版三个泡泡的注释都上下摆，结果全撞上了轴标题和落点句。
    #   ⭐ 判据：**散点图的标签，各自朝「自己这一侧最空」的方向放** ——&#160;
    #     统一朝上或统一朝下，一定会撞到轴。
    PLACE = (("right", 0), ("left", 0), ("up", -76))
    for k, (x, yy, col, name, when, comm) in enumerate(pts):
        side, dy = PLACE[k]
        if side == "up":
            ax, base, anc = x, yy + dy, "middle"
            rows = ((name_ := "%d｜%s" % (k + 1, name), 0), (when, 22), (comm, 42))
        else:
            sgn = 1 if side == "right" else -1
            ax, base, anc = x + sgn * 34, yy - 16, ("start" if sgn > 0 else "end")
            rows = (("%d｜%s" % (k + 1, name), 0), (when, 22), (comm, 42))
        for txt, off in rows:
            bold = txt.startswith("⛔") or off == 0
            f.t(ax, base + off, txt,
                col if (off == 0 or txt.startswith("⛔")) else GY,
                bold, 15.5 if off == 0 else 12, anc)

    f.box(OX + AW - 240, OY - AH - 18, 248, 46, "#fef7e0", OR, 6)
    f.t(OX + AW - 116, OY - AH + 12, "⛔ 这一角是空的：又大又忙的东西",
        OR, True, 13, "middle")

    f.t(700, py2 + 322,
        "⭐⭐⭐ 顺序不用背 ——&#160;<tspan font-weight=\"700\">"
        "沿着「先大后忙」这条线走一遍，排出来的就是 ZeRO 的三级。</tspan>"
        "⛔ 而这条判据出了 ZeRO 还能用："
        "<tspan font-weight=\"700\">先动又大又闲的那一项，最后才碰又小又忙的。</tspan>",
        INK, size=14.5, anchor="middle")
    f._pan = None

    yb = f.band(py2 + PH2 + 20, "ok",
                "顺带把「基线的 2Ψ」拆开看 ——　它解释了为什么切优化器状态是白捡的",
                ("⭐ 纯数据并行每步那笔 all-reduce，其实是"
                 "<tspan font-weight=\"700\">两步</tspan>："
                 "先 reduce-scatter（Ψ）把梯度归约并散开，"
                 "再 all-gather（Ψ）把结果收回来 ——&#160;合起来 2Ψ。",
                 "⭐⭐⭐ 而 ZeRO-1 要的<tspan font-weight=\"700\">正好是「散开」那个中间状态</tspan>："
                 "每张卡只更新自己那一份优化器状态。"
                 "<tspan font-weight=\"700\">它没有多要一次通信，它只是没把中间结果扔掉。</tspan>",
                 "⛔ 而 ZeRO-3 就不一样了：权重每一层都要用，"
                 "前向一次 all-gather、反向再一次，加上梯度那次 reduce-scatter，"
                 "总共 3Ψ ——&#160;<tspan font-weight=\"700\">1.5 倍，这是它要付的价。</tspan>"))

    yb = f.src(yb + 16,
               "📌 arXiv 1910.02054 §3（16 字节口径与 Figure 1 的四个数）、"
               "§7.2（通信量 2Ψ / 2Ψ / 3Ψ）。",
               "⭐ 图上四个 GB 数由 75 亿参数 × 每参数字节数算出，"
               "脚本内 assert 与论文 Figure 1 对账（120 / 31.4 / 16.6 / 1.9 GB）。")

    f.save("fig4-zero.svg", yb + 14)


if __name__ == "__main__":
    main()
