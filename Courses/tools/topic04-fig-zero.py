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
    PH2 = 258
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐ 顺序不是历史巧合　——　"
                  "<tspan font-weight=\"700\">是「多大」和「多久用一次」排出来的</tspan>", BL,
                  sub="⛔ 所以这三级<tspan font-weight=\"700\">不能换顺序</tspan>"
                      "，也不用背")

    WHY = (
        (RD, "#fce8e6", "优化器状态", "12 字节", "每一步只用<tspan font-weight=\"700\">一次</tspan>",
         "就在更新那一瞬间", "⭐ 最大 ＋ 最少用 →　先切它，几乎不加通信"),
        (BL, "#e8f0fe", "梯度", "2 字节", "反向<tspan font-weight=\"700\">结束时</tspan>汇总一次",
         "在那之前可以散着放", "⭐ 次之 —— 通信量还是跟基线一样"),
        (PU, "#f3e8fd", "权重", "2 字节", "<tspan font-weight=\"700\">每一层</tspan>前向都要用",
         "还有反向时再要一次", "⛔ 最后才切 —— 通信涨到 1.5 倍"),
    )
    for i, (col, fill, what, size_, freq, freq2, note) in enumerate(WHY):
        x = 40 + i * 442
        f.box(x, py2 + 36, 418, 196, fill, col, 8)
        f.t(x + 209, py2 + 70, what, col, True, 19, "middle")
        f.t(x + 209, py2 + 100, "占 " + size_ + " / 参数", INK, True, 14, "middle")
        f.t(x + 209, py2 + 136, freq, INK, True, 14.5, "middle")
        f.t(x + 209, py2 + 158, freq2, GY, size=12.5, anchor="middle")
        f.t(x + 209, py2 + 202, note, col, size=12.5, anchor="middle")
        if i:
            f.t(x - 14, py2 + 134, "→", GY2, True, 20, "middle")
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
