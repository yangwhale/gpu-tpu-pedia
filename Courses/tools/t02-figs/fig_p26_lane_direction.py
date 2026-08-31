# -*- coding: utf-8 -*-
"""图 P-26 —— 同一个向量寄存器，上下走很便宜，横着走很贵。

**为什么补这张图。** 3.2 刚加了一大段讲 XLU 的文字，但这段话的全部重点是
**方向** —— 「纵向」和「横向」在同一张 8×128 的表上是两条完全不同的路。
方向这种东西，文字讲三遍不如画一遍：读者脑子里得先有那张座位表，
后面「softmax 撞在最贵的方向上」才落得下去。原来那段没图，等于让每个人
自己在脑内建模，建错了后面全错。

**这张图只回答一个问题：为什么同样是「把一排数加起来」，换个方向就贵了。**
所以中间那张座位表是主角，两个方向的路径是它的两个分支 —— 不是三张并列的图。
绿色那一列和红色那一行是同一张表上的两笔，必须画在同一个网格里，
分开画就又变成「两件事」了。

**⚠️ 证据分层要画出来。**「一次 shuffle 约一个周期」「XLU 慢且贵」「2 个 XLU」
「跨 lane 至少要过一趟 VMEM／XLU／SMEM」都是公开资料的说法。

**底带那条红带不能省，但注意它现在讲的是另一件事。** 早先的版本在那里写
「列与列之间没有直连通路」，把「贵」归因于缺线 —— <b>那是想当然，已经收回</b>。
真正的说法是「贵在出一趟寄存器堆的往返」，这条有出处。
红带留下来是为了<b>把这次收回本身写在图上</b>，顺带交代仍然没出处的那一件：
<b>XLU 的内部电路长什么样</b>。
⚠️ 改这段文字时别把收回改没了 —— P-28/P-29/P-30 三张图都建立在
「贵在往返、不在缺线」这个口径上。
"""
from common import Fig, para, BL, GN, RD, YL, PU, TL, INK, SUB, GREY, FILL

W = 1400

# ── 第一带：座位表 ────────────────────────────────────────
MAP_Y, MAP_H = 84, 252
GX, GY = 176, 152           # 网格左上角
CW, CH = 30, 17             # 一格的大小
NCOL = 16                   # 13 条真 lane + 一个省略 + 最后两条
ELL = 13                    # 省略号落在第几格
HOT_COL, HOT_ROW = 4, 3     # 高亮的那一列 / 那一行

# ── 第二带：两个方向 ──────────────────────────────────────
DIR_Y, DIR_H = MAP_Y + MAP_H + 24, 262
LX, RX, CWID = 20, 708, 672

# ── 第三、四带 ────────────────────────────────────────────
WHY_Y, WHY_H = DIR_Y + DIR_H + 26, 100
N2_Y, N2_H = WHY_Y + WHY_H + 22, 104
BND_Y, BND_H = N2_Y + N2_H + 22, 92
H = BND_Y + BND_H + 20


def build():
    f = Fig(W, H,
            "TPU 向量寄存器 8×128 的方向不对称：沿 sublane 轴归约三步就完，"
            "沿 lane 轴归约必须经过 XLU，而公开资料称它慢且贵")
    f.title("同一张 8 × 128 的座位表　—— "
            "<tspan font-weight=\"700\">上下走三步就完，横着走要另找一个单元</tspan>")
    f.legend([(GN, "纵向 · 沿 sublane 轴（8）"),
              (RD, "横向 · 跨 lane（128）"),
              (YL, "神经网络最常见的两个归约，恰好都在横向")])
    _seatmap(f)
    _vertical(f)
    _horizontal(f)
    _why(f)
    _n2(f)
    _bound(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
# 一、座位表 —— 整张图的地基
# ══════════════════════════════════════════════════════════════════════
# ⛔ 绿列和红行必须画在**同一个网格**里。第一版把它们拆成左右两张小网格，
#    渲染出来读者会问「这是两个寄存器吗」—— 而这张图的全部意思恰恰是
#    「同一个寄存器，同一批数，只是方向不同」。
def _seatmap(f):
    f.rect(20, MAP_Y, 1360, MAP_H, "#fff", SUB, 1.2, 10)
    f.t(38, MAP_Y + 26, "一个向量寄存器就是这么一张表：8 行（sublane）× 128 列（lane）", "sec")

    # lane 表头
    f.t(GX + NCOL * CW / 2, GY - 26, "128 条 lane　→", "lbl", SUB, "middle")
    for c in range(NCOL):
        lab = "⋯" if c == ELL else str(c if c < ELL else 126 + (c - ELL - 1))
        f.t(GX + c * CW + CW / 2, GY - 8, lab, "xxs",
            RD if c == HOT_COL else GREY, "middle")

    # 8 行 sublane
    f.t(GX - 14, GY - 26, "8 个", "xxs", SUB, "end")
    f.t(GX - 14, GY - 12, "sublane", "xxs", SUB, "end")
    for r in range(8):
        y = GY + r * CH
        f.t(GX - 14, y + 12, str(r), "xxs", GREY, "end")
        for c in range(NCOL):
            x = GX + c * CW
            if c == HOT_COL:
                fill, st = FILL[GN], GN
            elif r == HOT_ROW:
                fill, st = FILL[RD], RD
            else:
                fill, st = "#fff", "#e4e6e8"
            f.rect(x, y, CW - 2, CH - 2, fill, st, 1.0, 2)
    gb = GY + 8 * CH

    # 「列是真硬件」—— 在网格底下钉一排 ALU 小块，让「一列 = 一套硬件」看得见
    for c in range(NCOL):
        x = GX + c * CW
        f.rect(x, gb + 8, CW - 2, 12, FILL[GN] if c == HOT_COL else "#f1f3f4",
               GN if c == HOT_COL else "#dadce0", 1.0, 2)
    f.t(GX - 14, gb + 18, "ALU", "xxs", GREY, "end")
    f.t(GX, gb + 38, "每一列底下是一整套自己的 ALU（v4 论文：每 lane 16 个）", "xxs", SUB)

    # 右半：这张表怎么读
    tx = GX + NCOL * CW + 46
    tw = 1380 - tx - 18
    y = para(f, tx, GY - 12, tw,
             "<b>⭐ 列不是画出来的，是真的硬件。</b>"
             "同一列上下那 8 个数，落在<b>同一套硬件</b>里；"
             "隔壁那一列，是<b>另一套硬件</b>。", "sm", 20)
    y += 10
    for c, sw, txt in [
            (GN, "绿色这一列",
             "沿它<b>上下</b>求和 —— 数就在自己这套硬件里，不用出门。"),
            (RD, "红色这一行",
             "沿它<b>横着</b>求和 —— 要跨 128 套互不相通的硬件。")]:
        f.rect(tx, y - 11, 12, 12, FILL[c], c, 1.2, 2)
        f.t(tx + 20, y, sw, "lbl", c)
        y = para(f, tx + 108, y, tw - 108, txt, "xs", 17) + 8
    para(f, tx, y + 6, tw,
         "<b>于是「上下」和「左右」根本不是一回事。</b>下面两格分别走一遍。",
         "xs", 17, fill=INK)


# ══════════════════════════════════════════════════════════════════════
# 二、纵向 —— 三步搞定
# ══════════════════════════════════════════════════════════════════════
# 四列 × 8 格：每过一步，参与的格子减半，剩下的染绿。
# 这个「一半一半地塌下去」的形状本身就是 log₂8 = 3，比写一行字管用。
STEPS = [("原始", 8), ("错 4 位加一次", 4), ("错 2 位加一次", 2), ("错 1 位加一次", 1)]
SC_W, SC_H, SC_GAP = 34, 15, 112


def _vertical(f):
    f.rect(LX, DIR_Y, CWID, DIR_H, FILL[GN], GN, 1.4, 10)
    f.t(LX + 18, DIR_Y + 27, "纵向　·　沿 sublane 轴（8 个）", "sec", GN)
    f.t(LX + 18, DIR_Y + 46, "硬件自带一条 shuffle 能沿这个轴滚一格，公开资料说约一个周期",
        "xs", SUB)

    y0 = DIR_Y + 66
    for i, (lab, alive) in enumerate(STEPS):
        x = LX + 46 + i * (SC_W + SC_GAP)
        for r in range(8):
            on = r < alive
            f.rect(x, y0 + r * SC_H, SC_W, SC_H - 2,
                   FILL[GN] if on else "#fff", GN if on else "#dadce0",
                   1.2 if on else 0.8, 2)
        f.t(x + SC_W / 2, y0 + 8 * SC_H + 16, lab, "xxs",
            INK if i else GREY, "middle")
        if i:
            ax = x - SC_GAP + SC_W
            f.line(ax + 14, y0 + 56, x - 14, y0 + 56, GN, 1.6, "aG")
    # 结果
    rx = LX + 46 + 3 * (SC_W + SC_GAP) + SC_W + 40
    f.t(rx, y0 + 52, "= Σ", "numb", GN)
    f.t(rx, y0 + 74, "三步", "lbl", GN)

    para(f, LX + 18, DIR_Y + DIR_H - 44, CWID - 36,
         "<b>8 → 4 → 2 → 1，三步，全程在寄存器里就地完成</b> —— "
         "数一次都没有离开自己那套硬件。这就是「便宜」的全部含义。",
         "sm", 19, fill=INK)


# ══════════════════════════════════════════════════════════════════════
# 三、横向 —— 得出门，而门很窄
# ══════════════════════════════════════════════════════════════════════
def _horizontal(f):
    f.rect(RX, DIR_Y, CWID, DIR_H, FILL[RD], RD, 1.4, 10)
    f.t(RX + 18, DIR_Y + 27, "横向　·　跨 lane（128 条）", "sec", RD)
    f.t(RX + 18, DIR_Y + 46, "没有这条 shuffle —— 数据要横着走，只能交出去", "xs", SUB)

    # 一排 lane
    lx, ly, lw = RX + 26, DIR_Y + 68, 22
    n = 14
    for c in range(n):
        f.rect(lx + c * (lw + 3), ly, lw, 20, FILL[RD], RD, 1.0, 2)
    f.t(lx + n * (lw + 3) + 10, ly + 14, "…… 128 条", "xxs", SUB)

    # 交给 XLU
    xy = ly + 52
    f.path("M%d %d C%d %d %d %d %d %d" % (lx + 40, ly + 24, lx + 40, xy - 6,
                                          lx + 150, xy - 6, lx + 150, xy - 2),
           RD, 1.6, marker="aR")
    f.path("M%d %d C%d %d %d %d %d %d" % (lx + 250, ly + 24, lx + 250, xy - 6,
                                          lx + 170, xy - 6, lx + 170, xy - 2),
           RD, 1.6, marker="aR")
    f.rect(lx, xy, 210, 46, "#fff", TL, 1.8, 8)
    f.t(lx + 12, xy + 20, "XLU　跨 lane 单元", "box", TL)
    f.t(lx + 12, xy + 37, "转置 · 跨 lane 归约 · shuffle", "xxs", SUB)

    # 常常还要绕 VMEM
    f.line(lx + 210, xy + 23, lx + 262, xy + 23, GREY, 1.4, "aK", "4 3")
    f.rect(lx + 266, xy + 6, 96, 34, "#fff", GREY, 1.2, 6)
    f.t(lx + 314, xy + 27, "VMEM", "lbl", SUB, "middle")
    f.t(lx + 266, xy + 56, "往往还要绕这一趟 —— 不是在寄存器里就地完成", "xxs", SUB)

    # 公开资料的原话
    f.rect(lx + 382, xy - 2, 236, 50, "#fff", RD, 1.4, 8)
    para(f, lx + 394, xy + 20, 212,
         "公开资料对它的形容：<r>「慢，而且贵」</r>", "xs", 17)

    # 供需比 —— 这才是「窄」的量化
    sy = DIR_Y + DIR_H - 82
    f.rect(RX + 18, sy, CWID - 36, 44, "#fff", RD, 1.2, 8)
    para(f, RX + 32, sy + 20, CWID - 64,
         "一个标量核管着：一个 VPU（<b>几千个 ALU</b>）· 多个 MXU · "
         "<r>2 个 XLU</r> · 多个 DMA 引擎", "xs", 17)
    f.t(RX + 32, sy + 38, "需求侧几千，供给侧个位数 —— 这就是那间传达室有多窄",
        "xxs", SUB)
    f.t(RX + 18, DIR_Y + DIR_H - 16,
        "「2 个」出自公开资料，但那句话没标代次 —— 只当「不止一个」用", "xxs", GREY)


# ══════════════════════════════════════════════════════════════════════
# 四、为什么这件事对大模型格外要命
# ══════════════════════════════════════════════════════════════════════
def _why(f):
    f.rect(20, WHY_Y, 1360, WHY_H, FILL[YL], YL, 1.6, 10)
    f.t(38, WHY_Y + 28,
        "⭐ 麻烦的是：神经网络最常做的那两个归约，方向恰好都是横的", "sec")
    y = para(f, 38, WHY_Y + 54, 1324,
             "张量的<b>最后一维默认铺在 lane 方向</b>上。而 <code>softmax</code> "
             "沿最后一维求和，<code>RMSNorm</code> 也沿最后一维求和 —— "
             "<b>两个最常见的算子，都撞在上面那条最贵的路上。</b>", "xs", 19)
    para(f, 38, y + 4, 1324,
         "这就是为什么在 TPU 上，attention 里 softmax 那一段"
         "<b>经常比你按 FLOP 算出来的贵得多</b>：<r>贵的不是算，是把数横着挪。</r>",
         "sm", 19, fill=INK)


# ══════════════════════════════════════════════════════════════════════
# 「为什么 8 便宜、128 贵」这个问题一定会被追到底，而答案是一个数。
# 没有这一条，前面所有「贵」都只是断言。
# ══════════════════════════════════════════════════════════════════════
def _n2(f):
    f.rect(20, N2_Y, 1360, N2_H, FILL[TL], TL, 1.6, 10)
    f.t(38, N2_Y + 28, "⭐ 那为什么 8 那个方向便宜、128 那个方向就得专门做个单元？"
                       "—— 差的是一个平方", "sec", TL)
    y = para(f, 38, N2_Y + 54, 1324,
             "要让一组数<b>任意互换位置</b>，需要的开关数按 <b>N²</b> 涨。"
             "sublane 方向 <b>8 见方 ＝ 64</b>；lane 方向 <b>128 见方 ＝ 16,384</b> "
             "—— <r>差 256 倍。</r>", "xs", 19)
    para(f, 38, y + 4, 1324,
         "所以 8 那个网络<b>小到可以摊进每一条 lane 里</b>，一个周期就转完；"
         "128 那个<b>只能全核共享一两个</b>，还得排队。"
         "<g>⚠️ N² 是数字设计通则，不是 TPU 的公开规格 —— 这里用它解释数量级。</g>",
         "xs", 19)


# ══════════════════════════════════════════════════════════════════════
# 五、边界
# ══════════════════════════════════════════════════════════════════════
def _bound(f):
    f.rect(20, BND_Y, 1360, BND_H, FILL[RD], RD, 1.4, 10)
    f.t(38, BND_Y + 28, "⚠️ 一个说法要收回：贵不是因为「没有连线」", "sec")
    y = para(f, 38, BND_Y + 54, 1324,
             "这张图早先的版本写着「列与列之间没有直连通路」——那是想当然，已经删掉。"
             "公开资料的说法是：跨 lane <b>至少要过一趟 VMEM／XLU／SMEM</b> ——"
             "<r>贵在「出一趟寄存器堆」这个往返，不在缺一根线。</r>", "xs", 19)
    para(f, 38, y + 2, 1324,
         "下一张图把这个往返画出来。<b>XLU 的内部电路仍然没有出处</b> ——"
         "公开资料只说它是个独立单元、而且慢，没说它里面长什么样。", "xs", 19)


if __name__ == "__main__":
    import io
    io.open("out/fig_p26_lane_direction.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p26_lane_direction")
