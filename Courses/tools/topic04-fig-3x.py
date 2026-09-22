# -*- coding: utf-8 -*-
r"""专题四 · §1.3「三倍算力是数出来的，不是估出来的」

⭐⭐⭐ 2026-09-16 新画。现场那一问是「反向计算到底是怎么回事」——&#160;
  这张图只回答其中最实的一小块：**一层里到底发生了几次矩阵乘。**

⭐⭐ 取舍只有一条：**画「几次乘法」，不画链式法则。**
  ⛔ 链式法则的图（一串 ∂ 套 ∂）看起来很有学问，但它回答不了
    「为什么是 3 倍」——&#160;而这一讲要的恰恰就是那个 3。
  ⭐ 判据：**一张图只该回答它所在那一节要回答的问题。**

⭐ 全图的钥匙在 Ⓑ：**反向之所以是两次而不是一次，
  是因为它要同时回答两个不同的问题** ——&#160;
  「我这块权重该怎么改」和「上游该收到什么」。
  两个问题，两次乘法。不是「反向比较慢」这种含糊说法。

⛔⛔ 刻意没画的：
  ① **偏置、norm、激活函数。** 它们的反向不是矩阵乘，在这笔账里是噪音。
  ② **具体的转置写法。** 写成 Wᵀ / Xᵀ 只会让人盯着记号，
     而这一节要的是「用到谁」，不是「怎么摆」。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

FWD, BWD = 1, 2                      # 每层：前向 1 次矩阵乘，反向 2 次
TOTAL = FWD + BWD
TOTAL_REMAT = TOTAL + FWD            # 开了全量重算，再多跑一遍前向
assert TOTAL == 3 and TOTAL_REMAT == 4


def main():
    f = Fig(W, "一层里前向只做一次矩阵乘，反向要做两次："
               "一次算这块权重自己的梯度，一次算该往下游传的责任 δ。"
               "一加二等于三，这就是训练比推理贵三倍的全部来源。"
               "如果再开全量重算，还要多跑一遍前向，变成四倍")

    y0 = f.header(
        "三倍算力是<tspan font-weight=\"700\">数出来的</tspan>"
        "　——　前向一次矩阵乘，反向两次",
        "⭐ 这一格不讲链式法则，只数<tspan font-weight=\"700\">乘法做了几次</tspan>",
        [(BL, "前向 1 次"), (RD, "反向 2 次"), (GR, "合计 3 次")])

    # ══════════ Ⓐ 一层里的三次矩阵乘 ═════════════════════════════
    PH = 620
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 把一层摊开 ——　<tspan font-weight=\"700\">"
                 "总共只有三次矩阵乘，一次向前、两次向后</tspan>", BL,
                 sub="⚠️ 只数矩阵乘：norm / 激活函数 / 偏置在这笔账里可以忽略")

    # ⛔⛔ 2026-09-18 R10 重画。原来这一格是**三张卡片**，
    #   每张卡里画个白框、框里写「一次矩阵乘」——&#160;
    #   ⭐ 一格在**数矩阵乘**，却一个矩阵都没画，字写着「一次矩阵乘」。
    #   这正是那条判据要抓的东西：**把文字删掉，这一格什么都不剩。**
    # ⭐⭐ 现在改成画**三次真的矩阵乘**，而且三个形状**按真实维度成比例**。
    #   于是两件事一眼就看出来了：
    #     ① 三次乘法长得不一样，可**每次的乘加次数完全相同**（都是 T·d·d）
    #        ——&#160;「1 ＋ 2 ＝ 3」不再是算术，是**三块等面积的积**。
    #     ② 第 ② 次那个操作数是**前向存下来的输入** ——&#160;
    #        用虚线框 ＋ 一根指回上一行的箭头画出来。
    #        **激活为什么扔不掉，这根箭头就是全部答案。**
    T_PX, D_PX = 118.0, 78.0          # token 维 / 特征维，各占多少像素

    # ⭐ 三次乘法的形状（行 × 列，单位就是上面那两个尺度）
    #   FLOPs 都 ∝ 行 × 内维 × 列 ——&#160;脚本当场验它们真的相等
    MULS = (
        (BL, "① 前向", "算这一层的输出",
         ("输入 X", T_PX, D_PX, False), ("权重 W", D_PX, D_PX, False),
         ("输出 Y", T_PX, D_PX, False), "→ 交给下一层"),
        (RD, "② 反向 · 权重梯度", "算「我这块权重该怎么改」",
         ("输入 Xᵀ", D_PX, T_PX, True), ("上游责任 dY", T_PX, D_PX, False),
         ("权重梯度 dW", D_PX, D_PX, False), "→ 交给优化器"),
        (RD, "③ 反向 · 传给下游", "算「前一层该收到什么」",
         ("上游责任 dY", T_PX, D_PX, False), ("权重 Wᵀ", D_PX, D_PX, False),
         ("新的责任 dX", T_PX, D_PX, False), "→ 交给前一层"),
    )
    # ⭐ 三次的乘加次数必须真的相等 ——&#160;这是「1＋2＝3」成立的全部前提
    _flops = [a[1] * a[2] * b_[2] for _, _, _, a, b_, _c, _n in MULS]
    assert len(set(round(v) for v in _flops)) == 1, \
        "三次乘法的乘加次数必须相等，现在是 %s" % _flops

    LX, MX0, ROW = 40, 300, 168
    for i, (col, tag, what, A, B, C, out) in enumerate(MULS):
        cy = py + 78 + i * ROW
        f.t(LX, cy + 6, tag, col, True, 17)
        f.t(LX, cy + 30, what, GY, size=12.5)

        x = MX0
        for k, (name, h, w, borrowed) in enumerate((A, B, C)):
            top = cy + 18 - h / 2.0
            # ⭐ 真的画一个矩形，而且**高宽按维度成比例**
            f.box(x, top, w, h, "#fff" if not borrowed else "#fef7e0",
                  OR if borrowed else col, 4,
                  sw=2.2 if borrowed else 1.4, dash="5 3" if borrowed else None)
            # 里面拉几道网格线，让它看起来像个矩阵而不是个方块
            for g in range(1, 4):
                f.line(x, top + h * g / 4.0, x + w, top + h * g / 4.0,
                       GY2, 0.6, arrow=False)
                f.line(x + w * g / 4.0, top, x + w * g / 4.0, top + h,
                       GY2, 0.6, arrow=False)
            f.t(x + w / 2.0, top + h + 18, name,
                OR if borrowed else col, True, 12.5, "middle")
            x += w
            if k < 2:
                f.t(x + 22, cy + 24, "×" if k == 0 else "＝", GY2, True, 20, "middle")
                x += 44
        f.t(x + 24, cy + 24, out, col, size=13)

        if i == 1:      # ⭐⭐⭐ 那根把激活账单钉死的箭头
            # ⛔⛔ 2026-09-19：这根箭头原来画在 MX0 + T_PX/2 ——&#160;那是**②行**那个框的中心。
            #   可①行的「输入 X」是竖着的（宽 D_PX），中心在 MX0 + D_PX/2 ——&#160;
            #   **两行的框宽根本不一样**，于是箭头斜着指进框里、还压着「输入 X」那行字。
            #   而且它上端 cy−ROW+62 比①行框底（cy−ROW+77）还高，下端 cy−48 又悬在
            #   ②行框顶（cy−21）上方 27px ——&#160;**两头都没接上**。
            # ⭐ 改成走左侧的折线，直接连两个框的**左边缘**：宽度不同也永远对得上。
            #   判据：**连接两个元素的线，端点要算自那两个元素本身，不要各自写死坐标。**
            a_top = cy + 18                      # ②行 Xᵀ 框的竖直中心
            a_bot = cy - ROW + 18                # ①行 X 框的竖直中心
            gx = MX0 - 30                        # 左侧让出来的走线
            f.line(MX0 - 2, a_top, gx, a_top, OR, 2.0, arrow=False)
            f.line(gx, a_top, gx, a_bot, OR, 2.0, arrow=False)
            f.line(gx, a_bot, MX0 - 4, a_bot, OR, 2.0)      # 箭头指回①行那个框
            # ⛔ 注解原来落在 cy−34，而②行 dY 框顶在 cy−41 ——&#160;**字直接压进框里**。
            #   ⭐ 挪到两行之间那条空带（①行图注底 cy−73 与②行框顶 cy−41 之间）。
            f.t(MX0 + T_PX + 30, cy - 56,
                "⭐ 这一块<tspan font-weight=\"700\">不是新算的</tspan>"
                "　——　是 ① 里那个输入<tspan font-weight=\"700\">被存下来了</tspan>",
                OR, True, 13.5)

    f.t(700, py + 78 + 3 * ROW - 24,
        "⭐⭐⭐ 三块积的<tspan font-weight=\"700\">面积一样大</tspan>"
        "　——　所以 <tspan font-weight=\"700\">1 ＋ 2 ＝ 3</tspan> 不是个比喻，"
        "是<tspan font-weight=\"700\">数出来的</tspan>。"
        "⛔ 而 ② 那块虚线的，就是<tspan font-weight=\"700\">激活扔不掉的全部原因</tspan>。",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 岔路：一进两出 ═════════════════════════════════
    # ⭐⭐⭐ 2026-09-17 重画。原来这一格是**两张并排的问题卡**：
    #   「问题一：我这块权重该怎么改」「问题二：我前面那层该收到什么」。
    #   话是对的，但它是**读**出来的 ——&#160;两张卡并排，
    #   读者看到的是「有两件事」，看不到「为什么正好是两件、一件都省不掉」。
    # ⭐ 改成岔路之后，那个「为什么」变成了图形本身：
    #   **一条线进来，分成两支** ——&#160;
    #     · 上面那支乘完就<b>到头了</b>（交给优化器，没有出口箭头）
    #     · 下面那支乘完<b>接着往左走</b>（喂给前一层）
    #   两支**乘的东西不一样**，所以合并不了；
    #   一支断了链条就断，一支没了这一层就白算 ——&#160;**一个都省不掉。**
    # ⛔ 判据：**「为什么是 N 个」这种问题，要用图形结构回答，不要用并列的卡片。**
    #   并列只表达「有 N 个」，结构才表达「为什么是 N 个」。
    PH2 = 356
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ ⭐⭐⭐ 全图的钥匙：<tspan font-weight=\"700\">"
                  "一条线进来，分成两支</tspan>", RD,
                  sub="⛔ 不是「反向比较慢」这种含糊说法 ——&#160;"
                      "<tspan font-weight=\"700\">是两支，而且一支都省不掉</tspan>")

    JX, JY = 470, py2 + 168           # 分岔点
    UY, DY = py2 + 92, py2 + 244      # 上支 / 下支

    # 进来的那条
    f.box(52, JY - 34, 300, 68, "#fce8e6", RD, 8)
    f.t(202, JY - 6, "上游传来的责任 δ", RD, True, 16, "middle")
    f.t(202, JY + 18, "（就这一个东西）", GY2, size=12.5, anchor="middle")
    f.line(352, JY, JX - 14, JY, RD, 2.0, arrow=False)
    f.box(JX - 9, JY - 9, 18, 18, RD, RD, 9)

    # 岔开的两支
    f.line(JX, JY, JX + 70, UY, RD, 1.8)
    f.line(JX, JY, JX + 70, DY, RD, 1.8)

    def branch(y, col, mul, out, dest, end, note):
        f.box(JX + 78, y - 34, 330, 68, "#fff", col, 8, sw=1.6)
        f.t(JX + 243, y - 8, "× " + mul, INK, True, 15, "middle")
        f.t(JX + 243, y + 18, "一次矩阵乘", col, True, 13.5, "middle")
        f.line(JX + 408, y, JX + 470, y, col, 1.6)
        f.box(JX + 478, y - 30, 250, 60, "#f1f3f4", col, 8)
        f.t(JX + 603, y + 5, out, col, True, 16, "middle")
        f.t(JX + 748, y - 6, dest, INK, True, 14.5)
        f.t(JX + 748, y + 16, end, col, True, 13)
        f.t(JX + 243, y + 50, note, GY, size=12.5, anchor="middle")

    branch(UY, PU, "前向存下来的<tspan font-weight=\"700\">输入</tspan>",
           "这块权重的梯度", "交给优化器",
           "⛔ 到此为止", "⭐ 这是这一步真正要的东西　·　到这儿这一支就不往前了")
    branch(DY, BL, "这一层的<tspan font-weight=\"700\">权重</tspan>",
           "新的责任 δ", "喂给前一层",
           "⭐ 接着往左走", "⭐ 链条靠它　·　少了它，再往前就断了")

    f.t(700, py2 + 322, "⭐⭐⭐ <tspan font-weight=\"700\">两支乘的东西不一样</tspan>"
        "（一支乘输入、一支乘权重）——　<tspan font-weight=\"700\">所以合并不了</tspan>；"
        "一支断了链条就断，一支没了这一层就白算。<tspan font-weight=\"700\">"
        "一个都省不掉，这就是那个 2。</tspan>",
        INK, size=14, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 加起来 ═════════════════════════════════════════
    PH3 = 250
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3,
                  "Ⓒ 于是这笔账就封口了", GR)

    BARS = ((FWD, BL, "#e8f0fe", "推理　只有前向"),
            (TOTAL, GR, "#e6f4ea", "训练　前向 ＋ 反向"),
            (TOTAL_REMAT, OR, "#fef7e0", "训练 ＋ 全量重算"))
    UNIT = 210
    for i, (n, col, fill, nm) in enumerate(BARS):
        y = py3 + 36 + i * 54
        f.t(300, y + 26, nm, INK, True, 15.5, "end")
        for k in range(n):
            f.box(320 + k * (UNIT + 10), y, UNIT, 38, fill, col, 6)
            f.t(320 + k * (UNIT + 10) + UNIT / 2.0, y + 25,
                "一遍" if k == 0 or i == 2 and k == 3 else "一遍",
                col, True, 15, "middle")
        f.t(320 + n * (UNIT + 10) + 6, y + 26, "＝ %d×" % n, col, True, 19)
    f.t(320, py3 + 208, "⭐ 「一遍」＝ 一次走完整个网络的矩阵乘量。"
                        "<tspan font-weight=\"700\">推理只买一遍，训练要买三遍</tspan>",
        GY, size=14.5)
    f._pan = None

    yy = f.band(py3 + PH3 + 22, "ok", "这张图顺带回答了另外两个常见疑问", [
        "❓ <tspan font-weight=\"700\">「为什么训练比推理贵这么多」</tspan> ——&#160;"
        "算力上就是这个 3 倍；⛔ 但真正拉开差距的不是它，"
        "而是<tspan font-weight=\"700\">显存里那一整条从头挂到尾的激活</tspan>（下一张图）。",
        "❓ <tspan font-weight=\"700\">「反向能不能只算一次」</tspan> ——&#160;"
        "能，如果你<tspan font-weight=\"700\">不打算继续往前传</tspan>（比如只微调最后一层）。"
        "⭐ 那种情况下第 ③ 次确实可以省掉。"
        "⛔ <tspan font-weight=\"700\">但别把它安到冻结层头上</tspan> ——&#160;LoRA 那种每层都挂 adapter 的，第 ③ 笔一笔都省不掉；它省的是第 ② 笔（权重梯度）。见 3.8。",
    ], keep=True)

    yy = f.src(yy + 24,
               "⚠️ 「3 倍」是<tspan font-weight=\"700\">矩阵乘口径</tspan>的常用近似："
               "只数 matmul，忽略 norm / 激活函数 / 偏置 / 通信。"
               "真实 step 里这些占比不大，但<tspan font-weight=\"700\">不是零</tspan>",
               "⛔ 图上第 ② 步那句「用前向存下来的输入」是<tspan font-weight=\"700\">"
               "整个专题的枢纽</tspan> ——&#160;激活扔不掉、以及下一节那笔重算交易，"
               "全都挂在这一句上")
    f.save("fig4-3x.svg", yy + 6)


main()
