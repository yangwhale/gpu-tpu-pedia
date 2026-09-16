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
               "一次算这块权重自己的梯度，一次算该往下游传的敏感度。"
               "一加二等于三，这就是训练比推理贵三倍的全部来源。"
               "如果再开全量重算，还要多跑一遍前向，变成四倍")

    y0 = f.header(
        "三倍算力是<tspan font-weight=\"700\">数出来的</tspan>"
        "　——　前向一次矩阵乘，反向两次",
        "⭐ 这一格不讲链式法则，只数<tspan font-weight=\"700\">乘法做了几次</tspan>",
        [(BL, "前向 1 次"), (RD, "反向 2 次"), (GR, "合计 3 次")])

    # ══════════ Ⓐ 一层里的三次矩阵乘 ═════════════════════════════
    PH = 356
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 把一层摊开 ——　<tspan font-weight=\"700\">"
                 "总共只有三次矩阵乘，一次向前、两次向后</tspan>", BL,
                 sub="⚠️ 只数矩阵乘：norm / 激活 / 偏置在这笔账里可以忽略")

    STEPS = (
        (BL, "#e8f0fe", "① 前向", "算这一层的输出",
         "用：这一层的输入 × 这一层的权重", "→ 输出，交给下一层"),
        (RD, "#fce8e6", "② 反向 · 权重梯度", "算「我这块权重该怎么改」",
         "用：上游传来的敏感度 × 前向存下来的输入", "→ 这块权重的梯度"),
        (RD, "#fce8e6", "③ 反向 · 传给下游", "算「上游该收到什么」",
         "用：上游传来的敏感度 × 这一层的权重", "→ 敏感度，交给前一层"),
    )
    for i, (col, fill, tag, what, use, out) in enumerate(STEPS):
        x = 48 + i * 440
        f.box(x, py + 34, 416, 254, fill, col, 8)
        f.box(x, py + 34, 416, 4, col, col, 2)
        f.t(x + 208, py + 70, tag, col, True, 19, "middle")
        f.t(x + 208, py + 98, what, INK, True, 15.5, "middle")
        f.box(x + 20, py + 120, 376, 62, "#fff", col, 6)
        f.t(x + 208, py + 148, "一次矩阵乘", col, True, 16, "middle")
        f.t(x + 208, py + 212, use, GY, size=13, anchor="middle")
        f.t(x + 208, py + 250, out, col, True, 14, "middle")
        if i:
            f.t(x - 12, py + 162, "＋", GY2, True, 22, "middle")
    f._pan = None

    # ══════════ Ⓑ 为什么反向是两次，不是一次 ═════════════════════
    PH2 = 246
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ ⭐ 全图的钥匙：<tspan font-weight=\"700\">"
                  "反向之所以是两次，因为它要回答两个不同的问题</tspan>", RD,
                  sub="⛔ 不是「反向比较慢」这种含糊说法 ——　"
                      "<tspan font-weight=\"700\">是两个问题，所以两次</tspan>")

    QS = (
        ("问题一", "我这块权重该怎么改？", "→ 这是这一步真正要的东西",
         "⭐ 答案留下来，交给优化器"),
        ("问题二", "我前面那一层该收到什么？", "→ 这是让链条能继续往回走",
         "⭐ 答案传下去，这一层就算完了"),
    )
    for i, (tag, q, why, note) in enumerate(QS):
        x = 64 + i * 652
        f.box(x, py2 + 34, 620, 154, "#fff", RD, 8)
        f.t(x + 24, py2 + 68, tag, RD, True, 17)
        f.t(x + 24, py2 + 102, q, INK, True, 19)
        f.t(x + 24, py2 + 134, why, GY, size=14)
        f.t(x + 24, py2 + 166, note, RD, True, 14)
    f.t(700, py2 + 214, "⛔ 两个问题都要用到「上游传来的那个敏感度」，"
                        "但<tspan font-weight=\"700\">另一边乘的东西不一样</tspan>"
                        " ——　所以省不掉其中任何一次", GY, size=14.5, anchor="middle")
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
        "⭐ 那种情况下第 ③ 次确实可以省掉 ——&#160;"
        "<tspan font-weight=\"700\">冻结层为什么便宜，原因就在这儿。</tspan>",
    ], keep=True)

    yy = f.src(yy + 24,
               "⚠️ 「3 倍」是<tspan font-weight=\"700\">矩阵乘口径</tspan>的常用近似："
               "只数 matmul，忽略 norm / 激活 / 偏置 / 通信。"
               "真实 step 里这些占比不大，但<tspan font-weight=\"700\">不是零</tspan>",
               "⛔ 图上第 ② 步那句「用前向存下来的输入」是<tspan font-weight=\"700\">"
               "整个专题的枢纽</tspan> ——&#160;激活扔不掉、以及下一节那笔重算交易，"
               "全都挂在这一句上")
    f.save("fig4-3x.svg", yy + 6)


main()
