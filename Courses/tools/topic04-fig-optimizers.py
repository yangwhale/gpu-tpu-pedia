# -*- coding: utf-8 -*-
r"""专题四 · §3「优化器谱系：每一代往每个参数身上挂了多少字节」

⭐⭐⭐ 2026-09-16 新画。现场原话：
  「你这个不得把主流优化器先捋一下？像什么 Adam、AdamW，还有 Muon，
    还有一个最开始的叫 G 什么来着？」——&#160;那个是 **SGD**。
  （顺带：现场说的「GDN」是 Gated DeltaNet，上一讲的**架构**，不是优化器。）

⭐⭐ 这张图的取舍只有一条：**不按「算法怎么算」排，按「每个参数要挂几个字节」排。**
  ⛔ 讲优化器的图通常画的是更新公式或者收敛曲线 ——&#160;
    那两样在这一讲里都**不承重**。这一讲的主线是一张账单，
    而优化器在账单上就是**一行的乘数**。
  ⭐ 判据：**同一批对象，按不同的量排，就是不同的图。**
    选哪个量，取决于这一讲要它回答什么。

⭐⭐⭐ 于是这张图自己说出了一件别的排法说不出来的事：
  **Adam 是一个峰，而 Muon 是往回走的那一步。**
  三十年里状态量一路加上去（0 →&#160;1 →&#160;2 份），
  2024 年有人把 v 那一份**整个拿掉了**，靠对更新矩阵做正交化补回来。

⛔⛔ 刻意没画的：
  ① **收敛曲线 / 谁比谁快。** 本课没有这些优化器的对照实测，画就是编。
     图上只标结构（挂几份状态）与各自论文自己写的定位。
  ② **更新公式。** 一写公式，「状态几份」这个唯一要看的量就被淹没了。
  ③ 横轴**不是线性年份**，只表示先后 ——&#160;图上写明了。

📌 出处（全部公开）：
  · SGD / 动量：Polyak 1964（heavy ball）；Nesterov 1983（NAG）
  · AdaGrad：Duchi, Hazan, Singer, JMLR 2011（无 arXiv）
  · RMSProp：Hinton 2012 Coursera 第 6 讲 ——&#160;**从未正式发表**
  · Adam：Kingma & Ba, **arXiv 1412.6980**（ICLR 2015）
  · AdamW：Loshchilov & Hutter, **arXiv 1711.05101**（ICLR 2019）
  · Adafactor：Shazeer & Stern, **arXiv 1804.04235**
  · Lion：Chen et al., **arXiv 2302.06675**（Symbolic Discovery）
  · Muon：Keller Jordan 等，2024，作者自己的 writeup
    kellerjordan.github.io/posts/muon —— 名字即缩写
    **MomentUm Orthogonalized by Newton-Schulz**
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

# ⭐ 混合精度下每参数挂的五样东西（字节）。改图先改这里，别在文字里手写。
W_BF16, G_BF16, MASTER_FP32, M_FP32, V_FP32 = 2, 2, 4, 4, 4
ADAMW_TOTAL = W_BF16 + G_BF16 + MASTER_FP32 + M_FP32 + V_FP32
MUON_TOTAL = ADAMW_TOTAL - V_FP32          # 只是少了 v 那一份
FP32_SHARE = MASTER_FP32 + M_FP32 + V_FP32
assert ADAMW_TOTAL == 16 and MUON_TOTAL == 12 and FP32_SHARE == 12


def main():
    f = Fig(W, "优化器谱系按「每个参数挂几份状态」排：SGD 零份，加动量一份，"
               "AdaGrad 和 RMSProp 各一份，Adam 与 AdamW 两份，"
               "而 2024 年的 Muon 只留一份动量、对更新矩阵做正交化，"
               "把二阶矩那一整份去掉了")

    y0 = f.header(
        "优化器谱系 ——　<tspan font-weight=\"700\">"
        "每一代往每个参数身上挂了几份状态</tspan>",
        "⛔ 这张图<tspan font-weight=\"700\">不按算法怎么算排，按账单排</tspan>"
        "　·　⚠️ 横轴只表示先后，<tspan font-weight=\"700\">不是线性年份</tspan>",
        [(GY2, "0 份"), (BL, "1 份"), (RD, "2 份"), (GR, "往回走")])

    # ══════════ Ⓐ 谱系：横轴先后，纵轴状态份数 ═════════════════════
    PH = 470
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 三十年只在加同一样东西 ——　"
                 "<tspan font-weight=\"700\">直到 2024 年有人把它拿掉一份</tspan>", BL,
                 sub="⭐ 「份数」指的是<tspan font-weight=\"700\">每个参数要额外存几个"
                     "跟它一样大的张量</tspan>")

    AX = py + 356
    f.line(90, AX, 1330, AX, GY2, 2.0)
    for k, (fr, lab) in enumerate(((0.0, "0 份"), (0.34, "1 份"), (0.68, "2 份"))):
        y = AX - 20 - fr * 250
        f.line(90, y, 1330, y, LINE2, 0.9, dash="4 5", arrow=False)
        f.t(74, y + 5, lab, GY2, True, 13, "end")

    # (x, 名字, 年, 状态份数 frac, 色, 一句话)
    PTS = (
        (170, "SGD", "", 0.0, GY2, "只按梯度走一步"),
        (360, "＋ 动量", "1964", 0.34, BL, "记住上一步往哪走"),
        (560, "AdaGrad", "2011", 0.34, BL, "每个参数一个学习率"),
        (760, "RMSProp", "2012", 0.34, BL, "把累加换成滑动平均"),
        (960, "Adam", "2014", 0.68, RD, "动量 ＋ 自适应，两份都要"),
        (1150, "AdamW", "2017", 0.68, RD, "修 weight decay 那个 bug"),
        (1310, "Muon", "2024", 0.34, GR, "去掉 v，改为正交化"),
    )
    for x, name, yr, fr, col, one in PTS:
        y = AX - 20 - fr * 250
        f.box(x - 6, y - 6, 12, 12, col, col, 6)
        f.t(x, y - 20, name, col, True, 17, "middle")
        if yr:
            f.t(x, y - 40, yr, GY2, size=12.5, anchor="middle")
        f.line(x, y + 8, x, AX - 6, LINE2, 1.0, dash="3 3", arrow=False)
        f.t(x, AX + 24, one, GY, size=12.5, anchor="middle")

    f.t(700, AX + 66, "⛔ 横轴只表示先后，不是线性年份", GY2, size=13,
        anchor="middle")

    # ⭐ 那一步「往回走」——&#160;它是全图唯一要人记住的形状
    # ⛔ 这个框原来在 py+28..124，**底边离 AdamW 的年份标签只剩 2px**
    #   （那个标签在 py+126）——&#160;渲染出来「2017」被框边压掉一半。
    #   ⭐ 抬到 py+10..98，留出 28px。判据：**标注框的下沿要按它下面那行文字的
    #     实际 y 算，不能按「看着没挨着」估** ——&#160;跟 fig3-gun 那条箭头同一个病。
    f.box(1040, py + 10, 304, 88, "#e6f4ea", GR, 8)
    f.t(1064, py + 38, "⭐⭐ Muon 是往回走的那一步", GR, True, 17)
    f.t(1064, py + 62, "三十年一路加，2024 年", GY, size=14)
    f.t(1064, py + 84, "有人把二阶矩那一份整个拿掉了", GY, size=14)
    f._pan = None

    yy = f.band(py + PH + 22, "warn", "AdamW 不是新算法 ——　它是在修一个 bug", [
        "⭐ 论文摘要原话：<tspan font-weight=\"700\">L2 正则和 weight decay 对普通 SGD 是等价的，"
        "但对 Adam 这种自适应方法<tspan text-decoration=\"underline\">不等价</tspan></tspan>。"
        "⛔ 因为 L2 那一项是<tspan font-weight=\"700\">混在梯度里</tspan>进去的，"
        "再被二阶矩的分母一除 ——&#160;"
        "<tspan font-weight=\"700\">梯度大的参数，它的 weight decay 被稀释掉了</tspan>。",
        "⭐⭐ 所以 AdamW 的修法是把 weight decay <tspan font-weight=\"700\">"
        "从梯度里解耦出来</tspan>，更新参数时单独减一下。"
        "<tspan font-weight=\"700\">那个 W 就是 decoupled 的意思。</tspan>",
    ])

    # ══════════ Ⓑ 那 16 字节到底是哪五样 ═════════════════════════
    PH2 = 404
    py2 = f.panel(0, yy + 26, W, PH2,
                  "Ⓑ 所以「每参数 16 字节」是哪五样 ——　"
                  "<tspan font-weight=\"700\">其中 12 字节是 fp32 的那三份</tspan>", RD,
                  sub="⭐ 混合精度训练的常规配置")

    ITEMS = (
        ("bf16 权重", W_BF16, GY2, "#f1f3f4", "前向反向都用它"),
        ("bf16 梯度", G_BF16, GY2, "#f1f3f4", "反向算出来的"),
        ("fp32 主权重", MASTER_FP32, RD, "#fce8e6", "真身，用来累加"),
        ("fp32 动量 m", M_FP32, RD, "#fce8e6", "Adam 的第一份"),
        ("fp32 二阶矩 v", V_FP32, RD, "#fce8e6", "⭐ Muon 去掉的就是它"),
    )
    BX, BW = 92, 1216
    x = BX
    for nm, nb, col, fill, note in ITEMS:
        w = BW * nb / float(ADAMW_TOTAL)
        f.box(x, py2 + 40, w, 72, fill, col, 8)
        f.t(x + w / 2.0, py2 + 72, nm, col, True, 15.5, "middle")
        f.t(x + w / 2.0, py2 + 96, "%d B" % nb, col, True, 17, "middle")
        f.t(x + w / 2.0, py2 + 134, note, GY, size=12.5, anchor="middle")
        x += w
    f.t(BX + BW + 12, py2 + 82, "＝ %d B" % ADAMW_TOTAL, INK, True, 19)

    # ── 分组括号 ─────────────────────────────────────────────────
    # ⭐⭐⭐ 2026-09-17 加。这根条原来是五段并排，读者看到的是「五样东西」，
    #   而这一讲真正要他记住的是**一刀两段**：
    #     2 B 推理也要 ｜ 14 B 训练才要。
    #   ⛔ 那句话本来写在下面的正文里 ——&#160;**而条就在眼前，字却要另外去读**。
    #   ⭐ 判据：**一根条上如果有一刀是全讲的主轴，那一刀就该画在条上，不是写在条下。**
    def bracket(x0, x1, y, col, label, sub_=None):
        """向下开口的分组括号。"""
        f.line(x0, y, x0, y + 9, col, 1.4, arrow=False)
        f.line(x0, y + 9, x1, y + 9, col, 1.4, arrow=False)
        f.line(x1, y, x1, y + 9, col, 1.4, arrow=False)
        f.t((x0 + x1) / 2.0, y + 30, label, col, True, 14.5, "middle")
        if sub_:
            f.t((x0 + x1) / 2.0, y + 52, sub_, GY, size=12.5, anchor="middle")

    _cut = BX + BW * W_BF16 / float(ADAMW_TOTAL)
    bracket(BX, _cut, py2 + 152, GY2, "%d B　推理也要" % W_BF16,
            "就是你下载到的那份权重")
    bracket(_cut, BX + BW, py2 + 152, RD,
            "%d B　<tspan font-weight=\"700\">训练才要</tspan>" % (ADAMW_TOTAL - W_BF16),
            "⭐ 这一讲讲的，全是这一段")

    f.t(92, py2 + 254, "⭐ 而这一讲那句「最大的一块是优化器状态」，"
                       "落到数上就是这么来的：", INK, True, 17)
    f.t(92, py2 + 282, "<tspan font-weight=\"700\">权重只占 2 B，"
                       "优化器那边（主权重 ＋ m ＋ v）占 %d B</tspan>"
                       " ——　<tspan font-weight=\"700\">六倍</tspan>。"
                       % FP32_SHARE, GY, size=15.5)
    f.t(92, py2 + 316, "⛔ 而换成 Muon：二阶矩那一份没了 ——　"
                       "<tspan font-weight=\"700\">%d B 变成 %d B</tspan>，少四分之一。"
                       % (ADAMW_TOTAL, MUON_TOTAL), GR, True, 16)
    f.t(92, py2 + 346, "⚠️ 但它<tspan font-weight=\"700\">只管二维参数</tspan> ——　"
                       "embedding 和输出头仍然走 AdamW，所以整模型省不到四分之一。",
        GY, size=14.5)
    f._pan = None

    yy = f.band(py2 + PH2 + 22, "ok", "Muon 拿什么换来的这一份", [
        "⭐ 名字就是做法：<tspan font-weight=\"700\">MomentUm Orthogonalized by "
        "Newton-Schulz</tspan> ——&#160;先按普通的 SGD 加动量算出更新量，"
        "再对这个更新<tspan font-weight=\"700\">矩阵</tspan>做几轮 Newton-Schulz 迭代，"
        "把它<tspan font-weight=\"700\">近似正交化</tspan>"
        "（等价于换成离它最近的半正交矩阵，也就是把奇异值都拉到 1）。",
        "⭐⭐ <tspan font-weight=\"700\">Adam 是逐元素缩放步长，Muon 是整个矩阵一起做谱归一化</tspan>"
        " ——&#160;不让某几个方向独大。"
        "⛔ 代价：作者自己写明 <tspan font-weight=\"700\">Muon 每一步的墙钟时间比 AdamW 慢</tspan>，"
        "它赚的是<tspan font-weight=\"700\">样本效率</tspan>（同样的 loss 用更少步数）。",
    ], fold=True)

    yy = f.src(yy + 24,
               "Adam：Kingma & Ba，<tspan font-weight=\"700\">arXiv 1412.6980</tspan>"
               "（ICLR 2015）；AdamW：Loshchilov & Hutter，"
               "<tspan font-weight=\"700\">arXiv 1711.05101</tspan>（ICLR 2019）——&#160;"
               "Ⓐ 那条带子里的说法是它摘要的转述",
               "Muon：Keller Jordan 等 2024，作者 writeup "
               "kellerjordan.github.io/posts/muon。"
               "「只用于 2D 参数」「标量 / 向量 / 输入输出层仍用 AdamW」"
               "「per-step wallclock 比 AdamW 慢」<tspan font-weight=\"700\">三条都是原文</tspan>",
               "动量：Polyak 1964（heavy ball）／Nesterov 1983；"
               "AdaGrad：Duchi, Hazan, Singer, JMLR 2011（无 arXiv）；"
               "RMSProp：Hinton 2012 Coursera 第 6 讲 ——&#160;"
               "<tspan font-weight=\"700\">从未正式发表</tspan>，这一条本身值得一提",
               "另有三条省状态的岔路本图没画："
               "Adafactor（<tspan font-weight=\"700\">arXiv 1804.04235</tspan>，把 v 分解成"
               "一行加一列）、Lion（<tspan font-weight=\"700\">arXiv 2302.06675</tspan>，"
               "只用梯度符号、单动量）、8-bit Adam（状态量化，不改算法）",
               "⛔ <tspan font-weight=\"700\">本图不含任何收敛速度的对照</tspan> ——&#160;"
               "本课没有这些优化器的对照实测，画曲线就是编。图上只有结构与各自论文的定位")
    f.save("fig4-optimizers.svg", yy + 6)


main()
