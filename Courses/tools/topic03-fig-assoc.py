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
        f.t(x0 + 4, cy - half - 12, tag, col, True, 18)

    HS, HD = S * U / 2.0, D * U / 2.0

    # ══════════ 上：softmax 那条路 ══════════════════════════════
    PH = 396
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
    f.t(70, py + PH - 34, "⛔ softmax 的分母要对<tspan font-weight=\"700\">"
        "所有位置</tspan>求和 ——　所以这张表躲不开，必须先整个算出来。",
        RD, True, 19, w=1260)

    # ══════════ 下：拿掉 softmax ════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 404
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
    f.box(70, py2 + PH2 - 82, 1260, 66, "#e6f4ea", GR, 10)
    f.t(92, py2 + PH2 - 52, "⭐⭐ 消失的那个 S，就是这个没被造出来的大方块。",
        GR, True, 22)
    f.t(92, py2 + PH2 - 26, "这也正是为什么它的状态形状是 "
        "<tspan font-weight=\"700\">BKHH</tspan> ——　"
        "里面根本没有句长这一维。", GY, size=18)

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
        "⭐ 所以 <a href=\"#s七\">§7.4</a> 那个「排队办事、只传一张交接单」"
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
