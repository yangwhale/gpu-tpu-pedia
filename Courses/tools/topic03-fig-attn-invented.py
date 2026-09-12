# -*- coding: utf-8 -*-
r"""专题三 · §一「注意力是怎么被发明出来的」（2026-09-13 夜间 · R7）。

⭐⭐ 三步，**每一步都在修上一步的一个具体毛病** ——&nbsp;
   注意力不是谁一次想出来的，是被三次「这儿不对」推出来的。

  ① **2014 Bahdanau** ——&nbsp;毛病是 seq2seq 要把整句话压进**一个固定长度向量**。
     他的第一个念头**不是「注意力」，是「对齐」**（机器翻译里本来就有词对词对应）。
     ⭐ 真正的关键一步是那个「**软**」字：硬切片不可导，
     softmax 加权平均可导 ——&nbsp;于是对齐能**跟翻译模型一起学**。
     ⭐⭐ 论文自己的说法很漂亮：上下文向量 = **在所有可能的对齐上取期望**。

  ② **2017 第一刀：把打分函数换成点积** ——&nbsp;
     Bahdanau 的打分是个小前馈网络（加性注意力）。
     ⭐⭐ 原文明说：两者理论复杂度相仿，但点积**快得多、省内存**，
     因为它**能用高度优化的矩阵乘实现**。
     ——&nbsp;**选点积不是因为它更准，是因为它能变成矩阵乘。**
     这是这门课「硬件反过来决定公式」的第一个例子，而且是论文原话。
     √d 的由来也在同一节的脚注里：分量独立、均值 0 方差 1 → 点积方差 = d_k。

  ③ **2017 第二刀：既然能直连，那还要 RNN 干什么** ——&nbsp;
     关联任意两个位置的操作数：ConvS2S 线性、ByteNet 对数、**自注意力常数**。
     ⭐⭐ 但论文同一句话里就承认了代价：**加权平均降低了有效分辨率**，
     「an effect we counteract with Multi-Head Attention」——&nbsp;
     **多头是用来补偿这个代价的**，不是「多个视角」这种营销词。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "注意力是怎么被发明出来的：2014 年 Bahdanau 为了修固定长度向量"
               "这个瓶颈提出软对齐，2017 年把打分换成点积（为了能用矩阵乘），"
               "再去掉循环，多头用来补偿加权平均造成的分辨率损失")
    f.marks = set()
    y0 = f.header(
        "注意力是怎么被发明出来的　——　三步，每一步都在修上一步的一个具体毛病",
        "⛔ 它<tspan font-weight=\"700\">不是谁一次想出来的</tspan>，"
        "是被三次「这儿不对」推出来的",
        [(BL, "2014 · 修瓶颈"), (GR, "2017 · 修打分函数"),
         (PU, "2017 · 去掉循环"), (OR, "被硬件推着走的地方")])

    ph = 436

    # ══ ① 2014 Bahdanau ═════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 2014　毛病：一个向量装不下", BL,
                 sub="Bahdanau / Cho / Bengio")

    yy = py + 26
    # 小示意：一整句 → 一个小方块 → 解码
    f.box(x + 24, yy, 150, 34, "#fff", GY, 6)
    f.t(x + 99, yy + 22, "一整句源文", GY, True, 12, "middle")
    f.line(x + 180, yy + 17, x + 216, yy + 17, GY2, 1.4)
    f.box(x + 222, yy + 4, 54, 26, "#fff", RD, 6)
    f.t(x + 249, yy + 21, "一个向量", RD, True, 11, "middle")
    f.line(x + 282, yy + 17, x + 318, yy + 17, GY2, 1.4)
    f.box(x + 324, yy, 92, 34, "#fff", GY, 6)
    f.t(x + 370, yy + 22, "解码", GY, True, 12, "middle")
    yy += 48
    f.t(x + 24, yy, "⛔ 句子越长越崩 —— 这是当时实测出来的", RD, size=11.5,
        w=pw - 48)
    yy += 26

    f.box(x + 22, yy, pw - 44, 74, "#fff", BL, 8)
    f.box(x + 22, yy, 4, 74, BL, BL, 2)
    f.box(x + 24, yy, 3, 74, "#fff", "#fff", 0)
    f.t(x + 40, yy + 24, "他的第一个念头不是「注意力」", BL, True, 12.5)
    f.t(x + 40, yy + 46, "是<tspan font-weight=\"700\">对齐</tspan> —— 机器翻译里本来就有的", GY, size=11.5)
    f.t(x + 40, yy + 65, "「这个译词对应原文哪几个词」", GY2, size=11)
    yy += 88

    f.box(x + 22, yy, pw - 44, 96, "#fff", GR, 8)
    f.box(x + 22, yy, 4, 96, GR, GR, 2)
    f.box(x + 24, yy, 3, 96, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐ 关键的一步是那个「软」字", GR, True, 13)
    f.t(x + 40, yy + 48, "硬切一段出来 → <tspan font-weight=\"700\">不可导</tspan>，学不了", GY, size=11.5)
    f.t(x + 40, yy + 68, "softmax 加权平均 → <tspan font-weight=\"700\">可导</tspan>", GY, size=11.5)
    f.t(x + 40, yy + 88, "于是对齐能跟翻译模型<tspan font-weight=\"700\">一起训</tspan>", GR, size=11.5)
    yy += 108

    f.t(x + 22, yy, "⭐⭐ 论文自己的说法，值得记住：", INK, True, 12.5)
    f.t(x + 22, yy + 22, "上下文向量 ＝ <tspan font-weight=\"700\">在所有可能的对齐上取期望</tspan>", INK,
        True, 12.5, w=pw - 44)
    fits(yy + 30, y0, ph, "①")

    # ══ ② 2017 第一刀：点积 ═════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 2017 第一刀　换打分函数", GR,
                 sub="加性 → 点积")

    yy = py + 26
    for who, how, col in [
        ("Bahdanau 的打分", "一个<tspan font-weight=\"700\">小前馈网络</tspan>（加性注意力）", GY),
        ("Transformer 的打分", "<tspan font-weight=\"700\">一个点积</tspan> q · k", GR),
    ]:
        f.box(x + 22, yy, pw - 44, 52, "#fff", col, 8)
        f.t(x + 38, yy + 22, who, col, True, 12)
        f.t(x + 38, yy + 41, how, GY, size=11.5, w=pw - 76)
        yy += 60

    yy += 4
    f.box(x + 22, yy, pw - 44, 100, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 100, OR, OR, 2)
    f.box(x + 24, yy, 3, 100, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐⭐ 换的理由，原文明说了", OR, True, 13)
    f.t(x + 40, yy + 48, "两者<tspan font-weight=\"700\">理论复杂度相仿</tspan>", GY, size=11.5)
    f.t(x + 40, yy + 68, "但点积<tspan font-weight=\"700\">快得多、省内存</tspan> —— 因为它", GY, size=11.5)
    f.t(x + 40, yy + 88, "<tspan font-weight=\"700\">能用高度优化的矩阵乘实现</tspan>", OR, True, 12.5)
    yy += 114

    f.t(x + 22, yy, "⭐ 那个 √d 也在同一节的脚注里", INK, True, 12.5)
    yy += 22
    f.box(x + 22, yy, pw - 44, 96, "#fff", LINE, 8)
    f.t(x + 38, yy + 24, "假设各分量独立、均值 0、方差 1", GY, size=11.5)
    f.t(x + 38, yy + 45, "→ 点积 q·k 的方差就是 <tspan font-weight=\"700\">d_k</tspan>", GY, size=11.5)
    f.t(x + 38, yy + 66, "d 一大，softmax 被推进饱和区，", GY, size=11.5)
    f.t(x + 38, yy + 86, "梯度小到学不动 → 所以除以 √d_k", INK, True, 12)
    fits(yy + 96, y0, ph, "②")

    # ══ ③ 2017 第二刀：去循环 ＋ 多头 ════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 2017 第二刀　那还要 RNN 干什么", PU,
                 sub="顺带交代多头为什么存在")

    yy = py + 24
    f.t(x + 22, yy, "关联任意两个位置，要几步？", INK, True, 12.5)
    yy += 18
    for who, cost, col in [("ConvS2S", "线性", GY2), ("ByteNet", "对数", GY2),
                           ("自注意力", "<tspan font-weight=\"700\">常数</tspan>", PU)]:
        f.box(x + 22, yy, pw - 44, 40, "#fff", LINE if col == GY2 else PU, 8)
        f.t(x + 38, yy + 25, who, col if col != GY2 else GY, True, 12)
        f.t(x + pw - 38, yy + 25, cost, col if col != GY2 else GY, True, 12,
            "end")
        yy += 46

    yy += 6
    f.box(x + 22, yy, pw - 44, 74, "#fff", PU, 8)
    f.box(x + 22, yy, 4, 74, PU, PU, 2)
    f.box(x + 24, yy, 3, 74, "#fff", "#fff", 0)
    f.t(x + 40, yy + 24, "于是那句标题就顺理成章了：", PU, True, 12.5)
    f.t(x + 40, yy + 46, "既然任意两个位置能直连，", GY, size=11.5)
    f.t(x + 40, yy + 65, "<tspan font-weight=\"700\">那还要循环干什么</tspan>", PU, True, 12.5)
    yy += 88

    f.box(x + 22, yy, pw - 44, 118, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 118, OR, OR, 2)
    f.box(x + 24, yy, 3, 118, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐⭐ 多头为什么存在 —— 论文给了因果", OR, True, 13)
    f.t(x + 40, yy + 49, "同一句话里它就承认了代价：", GY, size=11.5)
    f.t(x + 40, yy + 69, "<tspan font-weight=\"700\">加权平均降低了有效分辨率</tspan>", GY, True, 12)
    f.t(x + 40, yy + 91, "多头就是用来<tspan font-weight=\"700\">抵消这个代价</tspan>的", OR, True, 12.5)
    f.t(x + 40, yy + 111, "原话：单头时「averaging inhibits this」", GY2,
        size=11)
    fits(yy + 118, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 这一张要带走的：注意力是被三次「这儿不对」推出来的", [
        "① 一个向量装不下 → <tspan font-weight=\"700\">软对齐</tspan>；"
        "② 小网络打分太慢 → <tspan font-weight=\"700\">点积</tspan>；"
        "③ 既然能直连 → <tspan font-weight=\"700\">扔掉循环</tspan>，"
        "而扔掉之后分辨率变糙 → <tspan font-weight=\"700\">多头补回来</tspan>。",
        "⛔ 所以别把它讲成「有人灵光一闪设计了注意力」——&#160;"
        "<tspan font-weight=\"700\">每一步都能指出它在修哪一个具体毛病</tspan>，"
        "这才是可以学的部分。",
    ])

    yy = f.band(yy + 14, "ok", "⭐ 硬件反过来决定公式 —— 这门课的主线，第一次出现就在这儿", [
        "点积胜出<tspan font-weight=\"700\">不是因为它更准</tspan>（论文说两者理论复杂度相仿，"
        "小 d 下表现也相似），是因为它<tspan font-weight=\"700\">能写成矩阵乘</tspan>。",
        "⭐ 记住这条，后面看 head_dim=128 撞 MXU、看「按块选不按 token 选」、"
        "看「DSA 必须搭 MQA 模式」，都是同一件事的不同面："
        "<tspan font-weight=\"700\">能被硬件喜欢的公式，才活得下来。</tspan>",
    ])

    yy = f.band(yy + 14, "warn", "两个容易讲歪的地方", [
        "⚠️ <tspan font-weight=\"700\">「多头 ＝ 多个视角」是个营销式说法。</tspan>"
        "论文给的因果是反的：先有「加权平均把分辨率弄糙了」这个代价，"
        "多头是<tspan font-weight=\"700\">用来抵消它的补丁</tspan>。",
        "⚠️ <tspan font-weight=\"700\">「固定长度向量是瓶颈」是 Bahdanau 说的</tspan>，"
        "不是 seq2seq 原文说的 ——&#160;原文只是那么做，没说它是瓶颈。",
    ])

    yy = f.src(yy + 16,
               "① 出自 Bahdanau 等 arXiv 1409.0473（ICLR 2015）摘要与 §3："
               "fixed-length vector 是 bottleneck、(soft-)search、jointly trained、"
               "「expected annotation over possible alignments」",
               "②③ 出自 Attention Is All You Need（Vaswani 等 arXiv 1706.03762）"
               "原文 §2 与 §3.2.1–3.2.2 及其脚注 4：矩阵乘那句、方差 d_k 的推导、"
               "「reduced effective resolution due to averaging … counteract with Multi-Head」")
    f.save("fig3-attn-invented.svg", yy + 6)


main()
