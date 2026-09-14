# -*- coding: utf-8 -*-
r"""专题三 · §七「旋钮③ 立身之本：softmax 一拿掉，括号就能挪」

⭐⭐⭐ 2026-09-13 新画。调研全网之后发现的**我们最大的一个缺口**：
   「去掉 softmax，乘法就可以重新结合」是旋钮③ 的**立身之本**，
   而本课原来**一张图都没有** ——&nbsp;只有一句话，和一个结果
   （输出形状 BKHH，S 不见了）。⛔ 结果不等于动作：
   学生看得到 S 消失了，看不到**是哪一步让它消失的**。

📌 装置偷自 Google Research 的 Performer 博客（2020-10
   research.google/blog/rethinking-attention-with-performers）：
   **括号画成彩色虚线框，矩阵按真实比例画成方块。**
   那个 S×S 的大方块「没被建出来」是**看出来的**，不是读出来的。
   ⭐ 判据：这是全场 10 秒测试效率最高的一个装置 ——&nbsp;
   它不需要任何前置知识，大方块贵、小方块便宜、换个括号大方块就没了。

⚠️ 一条必须写在图上的口径（否则会教出一个太好的印象）：
   **因果 mask 会把这个重排挡住** ——&nbsp;逐元素乘那张下三角，
   恰恰就是挡住你用结合律的东西。所以真实实现是 chunkwise：
   块内老老实实按左边算，块间才用右边。见 §7.4。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    S, D = 8, 3          # 画面比例：序列 8 格、头维 3 格（示意，标注写真实量级）
    U = 30               # 一格多少像素

    f = Fig(W, "softmax 一拿掉，括号就能挪：左边先算 Q 乘 K 转置，中间被迫造出一个"
               "序列长 × 序列长的大方块；右边先算 K 转置乘 V，中间那块只有头维 ×"
               "头维，跟句子多长完全无关")
    f.marks = set()
    y0 = f.header(
        "旋钮③ 的立身之本 ——　<tspan font-weight=\"700\">同一个乘法，"
        "只是把括号挪了个位置</tspan>",
        "⭐ 矩阵按真实比例画。<tspan font-weight=\"700\">那个大方块有没有被造出来，"
        "是看出来的，不是读出来的。</tspan>",
        [(RD, "必须造出来的大方块"), (GR, "跟句子长短无关的小方块")])

    # ⛔ 各个矩阵高矮差很多（句长 8 格 vs 头维 3 格）——&nbsp;必须**按中线对齐**，
    #   否则读者会以为高度不一样代表别的意思。mat() 收的是**中线 y**不是顶边。
    def mat(x, cy, cols, rows, fill, col, lab, sub=None):
        w_, h_ = cols * U, rows * U
        y = cy - h_ / 2.0
        f.box(x, y, w_, h_, fill, col, 4, 1.6)
        for c in range(1, cols):
            f.line(x + c * U, y, x + c * U, y + h_, LINE2, 0.8, arrow=False)
        for r in range(1, rows):
            f.line(x, y + r * U, x + w_, y + r * U, LINE2, 0.8, arrow=False)
        f.t(x + w_ / 2.0, y - 12, lab, col, True, 18, "middle")
        if sub:
            f.t(x + w_ / 2.0, y + h_ + 22, sub, GY2, size=15, anchor="middle")
        return x + w_

    def times(x, cy):
        f.t(x + 13, cy + 9, "×", GY, True, 26, "middle")
        return x + 26

    def bracket(x0, x1, cy, half, col, tag):
        """括号 ＝ 一个彩色虚线框。⭐ 这是整张图的装置本身。"""
        f.box(x0 - 14, cy - half - 38, x1 - x0 + 28, 2 * half + 72,
              "none", col, 10, 1.8, "7,5")
        # ⛔ 标签原来放在虚线框**上面**，压住了面板标题栏。
        #   ⭐ 放进框里（左上角，紧贴虚线内侧）——&nbsp;而且这样更像真的括号注解。
        # ⛔ 又撞了：标签放框内左上角，正好压住第一个矩阵的名字（Q）。
        #   ⭐ 挪到框的**左下角外侧** —— 那儿是整张图唯一确定空着的地方。
        f.t(x0 - 14, cy + half + 56, tag, col, True, 18)

    HS, HD = S * U / 2.0, D * U / 2.0

    # ══════════════════════════════════════════════════════════════
    # ⭐⭐ 2026-09-14 补密度。原来两块面板右边各空着 476 / 626 px
    #   （占画布 34% / 45%），而全图只说了「大方块贵、小方块便宜」，
    #   **贵多少一个字都没有** ——&#160;真实量级只躲在最下面那行小字里。
    # ⭐ 判据：**一张图如果只给定性，读者记住的就只有定性。**
    #   「看出来的」是这张图的优点，但看出来之后得有个数接住，
    #   否则下课就只剩一句「右边那个比较小」。
    #
    # ⛔⛔ 这两个数的口径必须盯死，差一点就复活一句已经判过假的话：
    #   左边那张 S×S 的表，**FlashAttention 之后并不整个落在显存里**
    #   （分块算，算完就扔）。所以这里只能说「有多少个格子要算」，
    #   ⛔ 不能说「要占多少显存」。贵在要算的次数，不在显存。
    #   （同一条口径 R52 刚在 fig3-mha-qkv 上纠过，并已进退役清单。）
    # ⭐ 算式全部当场算，不手写 ——&#160;手写的数会跟 S/D 的改动脱钩。
    S_REAL, D_REAL = 128 * 1024, 128
    N_BIG, N_SMALL = S_REAL ** 2, D_REAL ** 2
    assert N_BIG == 17179869184 and N_SMALL == 16384
    RATIO = N_BIG // N_SMALL
    assert RATIO == 1048576                      # 2²⁰，正好一百万出头

    def sidecard(px_, py_, col, tint, title, big, unit, note, foot):
        # ⛔ 卡片高度、正文行数、落点行的 y 三者是**联动**的 ——&#160;
        #   第一版把落点写死在 +216，而四行正文正好排到 +212，两行叠在一起。
        # ⭐ 改成**落点跟着正文走**：正文排完再往下 28px，卡片高度也由此算出来。
        #   这样以后加减一行正文都不会再撞。
        # ⛔ 底框必须**先画** ——&#160;有填充色的 box 画在文字后面会把文字整片盖掉
        #   （这套基元里已经因为「跟背景同色的填充也是一次覆盖」栽过一回）。
        f.box(1000, py_, 344, 180 + len(note) * 22, tint, col, 10)
        f.t(1024, py_ + 34, title, col, True, 17)
        f.t(1024, py_ + 82, big, col, True, 34)
        f.t(1024, py_ + 108, unit, GY2, size=14)
        yy_ = py_ + 146
        for ln in note:
            f.t(1024, yy_, ln, GY, size=15, w=300)
            yy_ += 22
        f.t(1024, yy_ + 14, foot, col, True, 15, w=300)

    # ══════════ 上：softmax 那条路 ══════════════════════════════
    PH = 444
    py = f.panel(0, y0, W, PH,
                 "① softmax 在的时候 ——　必须先把那个大方块造出来", RD,
                 sub="分母要对所有位置求和，所以乘法顺序被锁死")
    cy = py + 168
    x = 70
    x0 = x
    x = mat(x, cy, D, S, "#e8f0fe", BL, "Q", "句长 × 头维")
    x = times(x + 16, cy) + 16
    x = mat(x, cy, S, D, "#fef7e0", OR, "Kᵀ", "头维 × 句长")
    bracket(x0, x, cy, HS, RD, "① 先算这一对")
    f.t(x + 40, cy + 9, "→", RD, True, 28, "middle"); x += 78
    x = mat(x, cy, S, S, "#fce8e6", RD, "Q·Kᵀ", "句长 × 句长")
    f.t(x - S * U / 2.0, cy + HS + 46, "⛔ 句子翻倍，它翻四倍", RD, True, 17,
        "middle")
    x = times(x + 16, cy) + 16
    mat(x, cy, D, S, "#e6f4ea", GR, "V", "句长 × 头维")
    sidecard(1000, py + 40, RD, "#fce8e6",
             "这张表到底多大（S ＝ 128K）",
             "171.8 亿", "个数 ＝ S² ＝ %s（每层、每个头）" % format(N_BIG, ","),
             ["⛔ 它不是「要占这么多显存」——　",
              "FlashAttention 之后分块算、算完就扔。",
              "⭐ 但每一个格子还是都要算一遍。",
              "省掉的是显存，不是算力。"],
             "⛔ 句子翻倍 → 这个数翻四倍")
    f.t(70, py + PH - 34, "⛔ softmax 的分母要对<tspan font-weight=\"700\">"
        "所有位置</tspan>求和 ——　所以这张表躲不开，必须先整个算出来。",
        RD, True, 17, w=900)

    # ══════════ 下：拿掉 softmax ════════════════════════════════
    y1 = y0 + PH + 18
    # ⛔⛔ 2026-09-14：底部那三样原来写成 py2 + PH2 - n，而 py2 ＝ y1 + 30、
    #   PH2 却是相对 y1 量的 ——&#160;**两个基准混用**，于是不管 PH2 调多大，
    #   最后一行永远比面板下沿低 4px（调 PH2 根本没用，我先白调了一次）。
    # ⭐ 判据：**同一块区域里的坐标必须共用一个基准。** 面板高度是相对面板顶
    #   量的，那么面板底部的东西也要相对面板顶量，不能混进 panel() 返回的
    #   那个「内容起点」。改成统一用 y1。
    PH2 = 500
    py2 = f.panel(0, y1, W, PH2,
                  "② 把 softmax 拿掉 ——　括号一挪，大方块根本没被造出来", GR,
                  sub="同一个乘法，同一个结果")
    cy = py2 + 168
    x = 70
    x = mat(x, cy, D, S, "#e8f0fe", BL, "Q", "句长 × 头维")
    x = times(x + 16, cy) + 16
    x0 = x
    x = mat(x, cy, S, D, "#fef7e0", OR, "Kᵀ", "头维 × 句长")
    x = times(x + 16, cy) + 16
    x = mat(x, cy, D, S, "#e6f4ea", GR, "V", "句长 × 头维")
    bracket(x0, x, cy, HS, GR, "① 改成先算这一对")
    f.t(x + 40, cy + 9, "→", GR, True, 28, "middle"); x += 78
    mat(x, cy, D, D, "#e6f4ea", GR, "KᵀV", "头维 × 头维")
    f.t(x + HD, cy + HD + 46, "⭐ 句子再长，它还是这么大", GR, True, 17,
        "middle")
    sidecard(1000, py2 + 40, GR, "#e6f4ea",
             "这块到底多大（D ＝ 128）",
             "16,384", "个数 ＝ D²　　⭐ 跟 S 没有任何关系",
             ["⭐ 句长从 2K 涨到 128K，",
              "这个数一个都没变 ——　它压根不认识 S。",
              "⭐⭐ 跟上面那张表差 %s 倍" % format(RATIO, ","),
              "（2²⁰，正好一百万出头）。"],
             "⭐ 这就是旋钮③ 的全部本钱")
    f.box(70, y1 + PH2 - 100, 1260, 66, "#e6f4ea", GR, 10)
    f.t(92, y1 + PH2 - 70, "⭐⭐ 消失的那个 S，就是这个没被造出来的大方块。",
        GR, True, 17)
    f.t(92, y1 + PH2 - 44, "这也正是为什么它的状态形状是 "
        "<tspan font-weight=\"700\">BKHH</tspan> ——　"
        "里面根本没有句长这一维。", GY, size=17)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y1 + PH2 + 20
    yy = f.band(yy, "warn", "⛔ 一条不说就会教出太好印象的口径：因果 mask 把这个重排挡住了", [
        # ⛔ 这里原来写了一条 .replace("**", …) 链 ——&nbsp;正是 memory 里
        #   「一组映射只要值域和定义域有交集就不能顺序执行」那条。直接写 tspan。
        "上面那一挪，成立的前提是<tspan font-weight=\"700\">每个位置都能看见所有位置"
        "</tspan>。可自回归下有一张下三角的 mask ——&#160;"
        "<tspan font-weight=\"700\">逐元素乘那张 mask，恰恰就是挡住你用结合律的那个东西</tspan>。",
        "所以真实实现是<tspan font-weight=\"700\">折中</tspan>："
        "把序列切成块，<tspan font-weight=\"700\">块内按左边那条路老老实实算</tspan>"
        "（mask 只在块内起作用），<tspan font-weight=\"700\">块间才用右边那条路传一个状态</tspan>。",
        "⭐ 所以后面那个「排队办事、只传一张交接单」"
        "不是工程妥协 ——&#160;它是<tspan font-weight=\"700\">「结合律」在有 mask "
        "的情况下的正确推广形态</tspan>。",
    ])
    yy = f.src(yy + 16,
               "装置偷自 Google Research《Rethinking Attention with Performers》"
               "（2020-10）：括号画成彩色虚线框、矩阵按真实比例画",
               "「mask 才是挡住结合律的那个东西」出自 Hailey Schoelkopf "
               "《Linear Attention Fundamentals》；"
               "「结合律是张量收缩顺序的特例」出自 Mamba-2 (SSD) 博客 Part II",
               "⚠️ 图里 8×3 的格数是<tspan font-weight=\"700\">示意</tspan>；真实量级是句长 128K、头维 128")
    f.save("fig3-assoc.svg", yy + 6)


main()
