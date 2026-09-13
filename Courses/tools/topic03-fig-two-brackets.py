# -*- coding: utf-8 -*-
r"""专题三 · §7.4 前置：**同一张矩阵里，两种括号**

⭐⭐⭐ 2026-09-14 夜间 R30 新画。这张图补的是本课一个**只在图注里说过一句**的空白：
   fig3-assoc 的图注写了「因果 mask 挡住结合律，所以真实实现是分块：
   块内按左边算，块间才用右边」—— 但**从来没画出来过**。
   而 §7.4 现有的 fig3-chunkwise 讲的是「排队办事」的比喻（并行度 1→C），
   **不是矩阵**。于是「为什么分块之后括号又能换回来」这一步，全课是空的。

⛔ 这不是 fig3-chunkwise 的替代，是它的**前一格**：
   这张回答「为什么可以分块」，那张回答「分块之后硬件上怎么排」。

═══ 调研结论（2026-09-14，六个来源逐一核过原图 / 原文）═══
现存四张图各有一样别人没有的东西，但**没有一张同时做到下面三件**：
  ① GLA Fig 3（arXiv 2312.06635v6，TikZ 源码 figures/second.tex）——
     唯一把「这一格能不能上 tensor core」编进图例的。
     配色原值：inter-chunk 灰 RGB(220,220,220)、inter-sub-chunk 橙(250,230,200)、
     intra-sub-chunk 粉(247,206,205)、causal mask 白。**一个公式都没有。**
  ② Mamba-2 Fig 5（arXiv 2405.21060）—— 唯一把「矩阵分块」和「RNN 数据流」
     用同一套颜色锁在一张图里的。但用的是 SSD 记号 C^T A B，中文读者要多翻一层。
  ③ snowchord《Linear Attention, Visualized》—— 唯一让**块的颜色 ＝ 那条路径
     主角张量的颜色**（块间紫＝S，块内蓝＝Q），且 mask 用**纹理不用颜色**。
  ④ rudrite research FIG. B —— 唯一用**一根竖条**（而不是公式）表示「低秩」。

⭐ 所以本图的空位是这三样，全网没人合在一起画：
   **(a) 两个括号的式子直接锚在各自的块上**（GLA 无公式，Mamba-2 是 SSD 记号）；
   **(b) 先画「不分块为什么不行」，再画分块**（现存全都直接给结论）；
   **(c) 用「有没有画出格子」本身当编码** —— 对角块画成一格一格（真的物化了
        一个 C×C 矩阵），块间画成一整块加一根竖条（**根本没物化**，压成了状态）。
   (c) 是本图自己的，四个来源都没这么干。

═══ 逐字引文（均为原文核对，不是转述）═══
- Albert Gu / Tri Dao 博客 goombalab.github.io/blog/2024/mamba2-part2-theory：
  "The issue is: once the L mask is incorporated into , we can no longer
   directly apply matrix associativity! This is the problem that the original
   Linear Attention paper addresses."
  ⭐ 同一段还有一句**必须一起引、否则会讲过头**：
  "Although it is commonly believed that incorporating attention masks L
   prevents matrix multiplication reordering, it turns out to still be
   compatible. In particular, associativity of matrix multiplication is a
   special case of tensor contraction reduction orders; although the former no
   longer applies, the latter can integrate the attention mask L."
  → 即：**被挡住的是「结合律」这个特例，不是「重排」本身。** 图里别写死成
    「mask 之后就不能重排了」。
- snowchord.com/blog/linear-attention-visualized（本图的落点句，逐字）：
  "The local block keeps a C×C causal mask because tokens in the chunk have
   different prefix lengths. The history blocks need no mask because every
   token in chunk r may read every token in an earlier chunk."
  "Associativity compresses the dense history: Σ_{s<r} A_{r,s} V_s
   = Q_r K_{<r}^T V_{<r} = Q_r S_r"
- Lightning Attention-2（arXiv 2401.04658，lightning2.tex）：
  "The intra- and inter-block operations are segregated, with intra-blocks
   employing the left product and inter-blocks utilizing the right product."
  ⭐ **left product / right product 这对词最适合直译进中文课件** ——
    比 intra/inter 更贴「换括号」这条本课自己的暗线。
- GLA 正文（同上 tex）：
  "The chunkwise parallel form, which interpolates between the parallel and
   recurrent forms with an extra ``parameter'' C, makes it possible to more
   easily make the above tradeoffs for fine-grained optimization."
  "the intra-chunk computations in GLA cannot leverage half-precision matmuls
   (and thus tensor cores) due to log space computations"
  ⭐ 后一句是**只对 GLA 成立的口径**，不能推广成「块内一律上不了 tensor core」。

⛔ 图里每一个数都是脚本当场算的（带断言），一个都不是引来的。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2)

W_ = 1400

L1 = 12          # 示意序列长度
C = 4            # 块大小
NB = L1 // C     # 块数
assert L1 % C == 0 and NB == 3


def causal_cells(n):
    """n×n 因果矩阵里真正有效（下三角含对角）的格子数。"""
    return n * (n + 1) // 2


def diag_cells(n, c):
    """分块之后，**仍然要一格一格老实算**的格子数 ＝ 每个对角块内的下三角。"""
    assert n % c == 0
    return (n // c) * causal_cells(c)


# ── 账：这三组数支撑面板 ③，全部当场算 ──────────────────────────
TOT_1 = causal_cells(L1)                 # 12 → 78
DIA_1 = diag_cells(L1, C)                # 3 块 × 10 → 30
OFF_1 = TOT_1 - DIA_1                    # 48
L2 = L1 * 2                              # 24
TOT_2 = causal_cells(L2)                 # 300
DIA_2 = diag_cells(L2, C)                # 6 块 × 10 → 60
OFF_2 = TOT_2 - DIA_2                    # 240
# 严格下三角的整块数（每一块都被一次 Q_r S_r 顶掉）
NOFF_BLK = NB * (NB - 1) // 2            # 3
assert (TOT_1, DIA_1, OFF_1) == (78, 30, 48)
assert (TOT_2, DIA_2, OFF_2) == (300, 60, 240)
assert NOFF_BLK * C * C == OFF_1         # 3 块 × 16 格 ＝ 48，对得上
assert DIA_2 == DIA_1 * 2, "句长翻倍，要老实算的那部分只能线性涨"
_R_TOT = TOT_2 / float(TOT_1)            # 3.846…
_R_DIA = DIA_2 / float(DIA_1)            # 2.0
assert abs(_R_DIA - 2.0) < 1e-9 and 3.8 < _R_TOT < 3.9
# 一般式自检：对角块总格数 ＝ L·(C+1)/2，与 L 成正比、与 C 成正比
assert DIA_1 == L1 * (C + 1) // 2 and DIA_2 == L2 * (C + 1) // 2

CS = 26          # 单元格边长
MSK = "#eceff1"  # 被 mask 的底色（配纹理用，⭐ mask 靠纹理不靠颜色）
BLU = "#d4e4fd"  # 块内 ＝ Q 的蓝（装置借自 snowchord：块色 ＝ 该路径主角的色）
PUR = "#e7dbec"  # 块间 ＝ S 的紫


def main():
    f = Fig(W_, "同一张因果注意力矩阵里的两种括号："
                "不分块时逐元素乘那张下三角挡住结合律，右括号插不进去；"
                "切成块之后，对角块仍用左括号一格一格算，"
                "而块间那些块里根本没有 mask，结合律在那儿从来没被挡住过，"
                "于是可以压成一个与句长无关的状态")
    f.marks = set()
    y0 = f.header(
        "同一张矩阵里，两种括号",
        "为什么<tspan font-weight=\"700\">非得分块</tspan> ——&#160;"
        "被 mask 挡住的其实只有<tspan font-weight=\"700\">对角块</tspan>",
        [(RD, "左括号 · 二次型"), (PU, "右括号 · 状态"), (GY2, "被 mask")])

    def hatch(x, y, s, col="#b0bec5"):
        """45° 细斜纹 ——&#160;表示「这格被 mask 了」。
        ⭐ 纹理而不是颜色：读者一眼知道那不是一个值，是被删掉了。"""
        k = 3
        for i in range(1, k + 1):
            d = s * i / float(k + 1)
            f.line(x, y + d, x + d, y, col, 0.8, arrow=False)
            f.line(x + d, y + s, x + s, y + d, col, 0.8, arrow=False)

    def cell(x, y, fill):
        f.box(x, y, CS - 2, CS - 2, fill, "#fff", 2, 1)

    # ══════════ ① 不分块：括号被焊死 ══════════════════════════════
    PH1 = 396
    top = f.panel(0, y0, W_, PH1, "① 不分块的时候，括号是焊死的", RD,
                  sub="同一个乘法的两种算法，右边那种这里用不了")

    MX, MY = 64, top + 64
    f.t(MX, MY - 34, "A = (Q Kᵀ) ⊙ M　　12 × 12", INK, True, 16)
    f.t(MX, MY - 14, "行 ＝ 第几个 token 在问　·　列 ＝ 它读到谁", GY, False, 13)
    for i in range(L1):
        for j in range(L1):
            x, y = MX + j * CS, MY + i * CS
            if j <= i:
                cell(x, y, BLU)
            else:
                cell(x, y, MSK)
                hatch(x, y, CS - 2)
    f.box(MX - 7, MY - 7, L1 * CS + 12, L1 * CS + 12, "none", RD, 7, 2)

    TX = MX + L1 * CS + 46
    f.t(TX, MY - 14, "左括号：先造 A，再乘 V", RD, True, 16)
    f.lines(TX, MY + 8, 430, [
        "<tspan font-weight=\"700\">(Q Kᵀ) ⊙ M</tspan> 这一步"
        "必须把整张 12×12 摆出来 ——",
        "句子长一倍，这张表就大四倍。<tspan font-weight=\"700\">这就是二次</tspan>。",
    ], size=15, lh=25)

    f.t(TX, MY + 84, "右括号：先把 Kᵀ V 攒成一个状态", PU, True, 16)
    f.lines(TX, MY + 106, 430, [
        "<tspan font-weight=\"700\">Q (Kᵀ V)</tspan> 中间那块只有 d × d，",
        "<tspan font-weight=\"700\">跟句子多长完全无关</tspan> ——&#160;这就是线性。",
    ], size=15, lh=25)

    # 红色阻断标记：⊙ M 卡在中间
    BY = MY + 178
    f.box(TX, BY, 430, 92, "#fff", RD, 8, 1.6)
    f.t(TX + 16, BY + 30, "⛔ 但这里插不进去", RD, True, 16)
    f.lines(TX + 16, BY + 52, 398, [
        "<tspan font-weight=\"700\">⊙ M 夹在 Q Kᵀ 和 V 中间</tspan> ——&#160;",
        "括号一挪，逐元素乘就无处安放。",
    ], size=15, lh=25)

    SX = TX + 486
    f.t(SX, MY - 14, "被挡住的到底是什么", INK, True, 16)
    f.lines(SX, MY + 8, W_ - SX - 60, [
        "Mamba-2 作者自己的说法（逐字）：",
        "「once the L mask is incorporated ...,",
        "we can no longer directly apply matrix",
        "associativity!」",
        "",
        "⚠️ <tspan font-weight=\"700\">但同一段紧接着还有一句，"
        "必须一起讲</tspan>：",
        "被挡住的是<tspan font-weight=\"700\">「结合律」这个特例</tspan>，",
        "不是「重排」本身 ——&#160;换成更一般的",
        "张量缩并顺序，mask 是能被吸收进去的。",
        "",
        "⭐ 这正是分块能成立的理由，下一格就是。",
    ], size=15, lh=25)

    # ══════════ ② 分块：两种括号同框 ══════════════════════════════
    y1 = y0 + PH1 + 20
    PH2 = 440
    top = f.panel(0, y1, W_, PH2, "② 切成块之后，同一张矩阵里出现两种括号", PU,
                  sub="⭐ 注意「有没有画出格子」本身就是编码")

    MX, MY = 64, top + 72
    f.t(MX, MY - 42, "同一张 12 × 12，只是切成 3 × 3 个块（C = 4）",
        INK, True, 16)
    f.t(MX, MY - 22,
        "画成一格一格 ＝ 真的摆出来了　·　画成一整块 ＝ 压成状态",
        GY, False, 13)

    for br in range(NB):
        for bc in range(NB):
            bx, by = MX + bc * C * CS, MY + br * C * CS
            if bc == br:
                # 对角块：一格一格画出来 —— 它真的物化了一个 C×C 矩阵
                for i in range(C):
                    for j in range(C):
                        x, y = bx + j * CS, by + i * CS
                        if j <= i:
                            cell(x, y, BLU)
                        else:
                            cell(x, y, MSK)
                            hatch(x, y, CS - 2)
                f.box(bx - 4, by - 4, C * CS + 6, C * CS + 6, "none", BL, 6, 2)
            elif bc < br:
                # 块间：一整块 ＋ 一根竖条（竖条 ＝ 秩被压到 d，装置借自 rudrite）
                f.box(bx, by, C * CS - 2, C * CS - 2, PUR, PU, 6, 1.6)
                f.box(bx + C * CS / 2.0 - 5, by + 10, 8, C * CS - 22,
                      "#c9a8d8", "none", 4)
            else:
                # 未来：只留虚线空框
                f.box(bx, by, C * CS - 2, C * CS - 2, "none", "#cfd8dc", 6,
                      1.2, dash="4 4")

    f.box(MX - 8, MY - 8, L1 * CS + 14, L1 * CS + 14, "none", LINE2, 8, 1.4)

    # 右侧：状态串珠（借 Songlin Yang 讲座 p47 的 state passing）
    # ⛔ 状态画在**块行之间的边界**上，而且只有 NB-1 个 ——
    #   chunk 1 没有历史块，它本来就不消费状态。早先每行画一个是错的。
    SBX = MX + L1 * CS + 34
    for k in range(NB - 1):
        cy = MY + (k + 1) * C * CS - 17
        f.box(SBX, cy, 62, 34, "#fff", PU, 6, 1.6)
        f.t(SBX + 31, cy + 23, "S%d" % (k + 1), PU, True, 15, "middle")
        if k:
            f.line(SBX + 31, cy - C * CS + 20, SBX + 31, cy - 4, PU, 1.6)
    f.t(SBX + 31, MY + L1 * CS + 4, "前面攒下的状态", PU, True, 14, "middle")
    f.t(SBX + 31, MY + L1 * CS + 26, "只有 %d 个" % (NB - 1), GY, False, 13,
        "middle")

    AX = SBX + 104
    f.t(AX, MY - 14, "对角块　→　左括号", BL, True, 16)
    f.lines(AX, MY + 8, 408, [
        "<tspan font-weight=\"700\">(Q_r K_rᵀ ⊙ M) V_r</tspan>　"
        "——&#160;块内老实算",
        "块里的 token 前缀长度不同，"
        "<tspan font-weight=\"700\">mask 必须留着</tspan>。",
        "代价只有 C × C，跟整句多长无关。",
    ], size=15, lh=25)

    f.t(AX, MY + 112, "块间　→　右括号", PU, True, 16)
    f.lines(AX, MY + 134, 408, [
        "<tspan font-weight=\"700\">Q_r S_r</tspan>　——&#160;一次乘法顶掉一整块",
        "<tspan font-weight=\"700\">这些块里根本没有 mask</tspan>：chunk r 的",
        "每个 token 都能读前面 chunk 的每个 token。",
        "没有 mask，结合律在这儿<tspan font-weight=\"700\">从来没被挡过</tspan>。",
    ], size=15, lh=25)

    f.box(AX, MY + 244, 408, 96, "#fff", GR, 8, 1.6)
    f.t(AX + 16, MY + 274, "⭐ 所以那根竖条是什么意思", GR, True, 16)
    f.lines(AX + 16, MY + 296, 376, [
        "整块 16 个格的内容，被压进一个 d × d 的 S ——",
        "<tspan font-weight=\"700\">它从来没被摆出来过</tspan>。",
    ], size=15, lh=25)

    # ══════════ ③ 账：翻倍之后谁涨了 ══════════════════════════════
    y2 = y1 + PH2 + 20
    PH3 = 286
    top = f.panel(0, y2, W_, PH3, "③ 把句子拉长一倍，涨的是哪一部分", OR,
                  sub="每个数都是这张图自己数出来的")

    def acct(x, n, tot, dia, off, nb):
        f.box(x, top + 18, 620, 240, "#fff", LINE, 8, 1.2)
        f.t(x + 22, top + 50, "句长 %d　（C 仍然是 %d，共 %d 块）"
            % (n, C, nb), INK, True, 16)
        f.lines(x + 22, top + 76, 576, [
            "因果矩阵里有效的格子　　　　　"
            "<tspan font-weight=\"700\">%d 格</tspan>" % tot,
            "其中落在对角块里、<tspan font-weight=\"700\">要老实算</tspan>的　"
            "<tspan font-weight=\"700\" fill=\"%s\">%d 格</tspan>" % (BL, dia),
            "其余被<tspan font-weight=\"700\">状态顶掉</tspan>的　　　　　　"
            "<tspan font-weight=\"700\" fill=\"%s\">%d 格</tspan>" % (PU, off),
        ], size=15, lh=30)
        f.t(x + 22, top + 196, "占比：老实算的只有 %d%%"
            % round(100.0 * dia / tot), GY, False, 15)
        f.t(x + 22, top + 226,
            "——&#160;剩下 %d%% 全部收进 %d 个 S 里"
            % (round(100.0 * off / tot), nb - 1), GY, False, 15)

    acct(40, L1, TOT_1, DIA_1, OFF_1, NB)
    acct(740, L2, TOT_2, DIA_2, OFF_2, L2 // C)

    yy = y2 + PH3 + 22
    yy = f.band(yy, "info", "这张图的落点：被 mask 挡住的只有对角块", [
        "句长翻倍，<tspan font-weight=\"700\">整张表的有效格子涨了 %.2f 倍</tspan>"
        "（%d → %d）——&#160;这就是二次；"
        "而<tspan font-weight=\"700\">真正要一格一格算的只涨了 %.0f 倍</tspan>"
        "（%d → %d），<tspan font-weight=\"700\">正好线性</tspan>。"
        % (_R_TOT, TOT_1, TOT_2, _R_DIA, DIA_1, DIA_2),
        "原因不是「块小所以算得动」，是"
        "<tspan font-weight=\"700\">块间那些块里根本没有 mask</tspan> ——&#160;"
        "结合律在那儿<tspan font-weight=\"700\">从来没被挡住过</tspan>，"
        "所以它们能被压成一个与句长无关的状态。",
    ])
    yy = f.band(yy + 12, "warn", "三处别讲过头", [
        "① 被 mask 挡住的是<tspan font-weight=\"700\">「结合律」这个特例</tspan>，"
        "不是「重排」本身 ——&#160;Mamba-2 作者明确写了：换成更一般的张量缩并顺序，"
        "mask 是能被吸收进去的。图里画的是<tspan font-weight=\"700\">通常实现</tspan>"
        "走的那条路，不是数学上的唯一解。",
        "② 对角块那 %d%% <tspan font-weight=\"700\">不会随句长消失</tspan>，"
        "它只是从平方变成线性。想再压，只能调小 C ——&#160;"
        "而 C 被片上内存顶死，那是下一张图（§7.4）的事。"
        % round(100.0 * DIA_1 / TOT_1),
        "③ GLA 论文说它的块内算不了半精度 matmul、上不了 tensor core，"
        "<tspan font-weight=\"700\">那是 GLA 用 log space 才有的口径</tspan>，"
        "别推广成「块内一律上不了 tensor core」。",
    ])

    yy = f.src(yy + 14,
               "📌 「结合律被挡住／但张量缩并仍可吸收 mask」"
               "出自 Mamba-2 作者博客 goombalab.github.io/blog/2024/"
               "mamba2-part2-theory，逐字核对。",
               "📌 「块间不需要 mask，因为 chunk r 的每个 token 都能读"
               "前面 chunk 的每个 token」出自 snowchord.com/blog/"
               "linear-attention-visualized，逐字核对。",
               "📌 「块内用左乘法、块间用右乘法」出自 Lightning Attention-2"
               "（arXiv 2401.04658）原文 intra-blocks employing the left "
               "product and inter-blocks utilizing the right product。",
               "📌 画法上借了四处：块色＝主角张量色、mask 用纹理不用颜色"
               "（snowchord）；块间一根竖条表示低秩（rudrite research）；"
               "右侧状态串珠（Songlin Yang 讲座 slides）；两级分块与 tensor "
               "core 口径（GLA, arXiv 2312.06635 Figure 3）。"
               "⛔ 图里的格子数全部是本脚本当场数的，不是引来的。")

    f.save("fig3-two-brackets.svg", yy + 10)


if __name__ == "__main__":
    main()
