# -*- coding: utf-8 -*-
r"""专题四 · §1.5「为什么是从后往前」——&#160;两个方向摆在一起看

⭐⭐⭐ 2026-09-17 新画。这一格是整个第一节**最深的一条**，
  而它原来是**纯文字**：两个 li，一段落点带，一张图都没有。
  ⛔ 现场那句原话是「这么多年了都没有人真正的把它讲明白」——&#160;
    而讲明白它的关键，恰恰是一张**能让人自己数一遍**的图。

⭐⭐ 这张图的取舍：**画「种子插在哪一头」，不画链式法则的公式。**
  ⛔ 绝大多数教程在这里画的是一串 ∂y/∂x 相乘 ——&#160;
    公式是对的，但它解释不了「为什么反过来走就便宜了」，
    因为**乘法是可交换的，从哪头乘看上去都一样**。
  ⭐ 真正的区别不在乘什么，在**你得重复几遍**：
    · 正向模式：扰动的种子插在**输入端**，有几个参数就要插几次
    · 反向模式：种子插在**输出端**，而输出只有一个数 ——&#160;插一次就完了
  这件事画出来就是：**一边要画 N 条线，一边只画一条。**

⭐ Ⓑ 那条判据能出这一讲：**从窄的那一头起步。**
  它对「多输入单输出」成立，反过来就该反过来 ——&#160;
  所以图上必须把反例也画上，否则读者会把它记成「反向传播永远更好」。

⚠️ 这张图里不放任何实测数字。三千亿是 §1.1 已经立过的模型规模，
  其余全是结构，没有量。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

OPS = ("嵌入", "第 1 层", "第 2 层", "…", "最后一层", "loss")


def main():
    f = Fig(W, "把正向模式和反向模式画在同一条计算链上。"
               "正向模式的扰动种子插在输入端，有多少个参数就要重跑多少遍；"
               "反向模式的种子插在输出端，而输出只有一个数，所以一遍就够，"
               "而且一路往回走的时候每个参数的梯度顺手就拿到了。"
               "判据是从窄的那一头起步 —— 如果反过来是少输入多输出，"
               "那就该用正向模式")

    y0 = f.header(
        "为什么是<tspan font-weight=\"700\">从后往前</tspan>　——　"
        "<tspan font-weight=\"700\">因为那一头只有一个数</tspan>",
        "⛔ 这里的关键<tspan font-weight=\"700\">不是链式法则怎么乘</tspan>"
        "（乘法从哪头开始都一样）——&#160;是<tspan font-weight=\"700\">你得重复几遍</tspan>",
        [(OR, "正向：插在输入端"), (GR, "反向：插在输出端")])

    # ── 一条链的画法，两排共用 ────────────────────────────────────
    CX0, CW, CGAP = 232, 148, 26
    def chain(cy, col, dim=False):
        xs = []
        for i, name in enumerate(OPS):
            x = CX0 + i * (CW + CGAP)
            last = (i == len(OPS) - 1)
            f.box(x, cy, CW, 52, "#fff", RD if last else LINE, 8,
                  sw=1.6 if last else 1)
            f.t(x + CW / 2, cy + 32, name, RD if last else (GY2 if dim else INK),
                bold=last, size=14 if last else 13, anchor="middle")
            xs.append(x)
        return xs

    # ══════════ Ⓐ 正向模式 ═════════════════════════════════════════
    PH = 296
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 正向模式　——　"
                 "<tspan font-weight=\"700\">种子插在左边，有几个参数就要插几次</tspan>", OR,
                 sub="⭐ 每插一次，就要把<tspan font-weight=\"700\">整条链从头走到尾</tspan>一遍")

    cy = py + 78
    xs = chain(cy, OR)

    # 左边三个种子 —— 用「三遍」代表 N 遍
    for k, (dy, lab) in enumerate(((-44, "第 1 遍"), (0, "第 2 遍"), (44, "第 3 遍"))):
        yy = cy + 26 + dy
        f.box(66, yy - 15, 96, 30, "#fef7e0", OR, 6)
        f.t(114, yy + 5, lab, OR, True, 12.5, "middle")
        f.line(164, yy, xs[0] - 6, cy + 26, OR, 1.2, arrow=True)
    f.t(114, cy + 96, "⋮", OR, True, 22, "middle")
    f.t(114, cy + 126, "共 <tspan font-weight=\"700\">3000 亿</tspan> 遍",
        OR, True, 13.5, "middle")

    # 链上的方向箭头
    for i in range(len(OPS) - 1):
        f.line(xs[i] + CW + 3, cy + 26, xs[i + 1] - 3, cy + 26, OR, 1.5)

    f.t(CX0, cy + 92, "⭐ 每一遍走完，只拿到<tspan font-weight=\"700\">一个</tspan>"
        "参数的梯度 ——　因为你这一遍只扰动了它一个", GY, size=13)
    f.t(CX0, cy + 118, "⛔ 所以它的代价跟<tspan font-weight=\"700\">参数个数</tspan>"
        "成正比。三千亿个参数，就是三千亿遍整条链。", RD, size=13)
    # ⚠️ 诚实补一句：种子画在最左端只是为了画面整齐。
    #   参数是分布在每一层的，真要插是插在它所在的那一层。
    #   ⭐ 但这不影响这张图要说的事 ——「一个参数一遍」这一条不变。
    f.t(CX0, cy + 144, "⚠️ 画成从最左边插只是为了整齐 ——　"
        "参数分布在每一层，种子就插在它所在的那一层。"
        "<tspan font-weight=\"700\">不变的是「一个参数一遍」。</tspan>",
        GY2, size=12)
    f._pan = None

    # ══════════ Ⓑ 反向模式 ═════════════════════════════════════════
    PH2 = 268
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 反向模式　——　"
                  "<tspan font-weight=\"700\">种子插在右边，而右边只有一个数</tspan>", GR,
                  sub="⭐⭐⭐ <tspan font-weight=\"700\">所以只插一次，"
                      "而且回来的路上每个参数的梯度顺手就到手了</tspan>")

    cy2 = py2 + 78
    xs2 = chain(cy2, GR)

    # 右边一个种子
    sx = xs2[-1] + CW + 18
    f.box(sx, cy2 + 11, 96, 30, "#e6f4ea", GR, 6)
    f.t(sx + 48, cy2 + 31, "种子 ＝ 1", GR, True, 12.5, "middle")
    f.t(sx + 48, cy2 + 62, "<tspan font-weight=\"700\">只有这一个</tspan>",
        GR, size=12.5, anchor="middle")

    # 链上的方向箭头 —— 反着走
    for i in range(len(OPS) - 1, 0, -1):
        f.line(xs2[i] - 3, cy2 + 26, xs2[i - 1] + CW + 3, cy2 + 26, GR, 1.5)

    # 每个算子底下挂一个「梯度到手」
    for i in range(len(OPS) - 1):
        f.t(xs2[i] + CW / 2, cy2 + 76, "✓", GR, True, 15, "middle")
        f.t(xs2[i] + CW / 2, cy2 + 96, "梯度到手", GR, size=11.5, anchor="middle")

    f.t(CX0, cy2 + 132, "⭐⭐⭐ 走到最左边的时候，"
        "<tspan font-weight=\"700\">三千亿个梯度已经全部在手里了</tspan>"
        "　——　而你只走了<tspan font-weight=\"700\">一遍</tspan>。", INK, size=14)
    f.t(CX0, cy2 + 158, "⭐ 一路上传的始终是<tspan font-weight=\"700\">一个东西</tspan>"
        "（那个种子沿途被改写），"
        "所以那些巨大的中间雅可比<tspan font-weight=\"700\">从来不用真的算出来</tspan>。",
        GY, size=13)
    f._pan = None

    # ══════════ Ⓒ 判据 ＋ 反例 ══════════════════════════════════════
    PH3 = 222
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐ 所以规则只有一条，而且它<tspan font-weight=\"700\">"
                  "跟神经网络没关系</tspan>", PU,
                  sub="⛔ 别把它记成「反向传播永远更好」——&#160;"
                      "<tspan font-weight=\"700\">换个形状它就反过来</tspan>")

    f.t(700, py3 + 62, "<tspan font-weight=\"700\">从窄的那一头起步。</tspan>",
        PU, True, 24, "middle")

    CASES = (
        (GR, "#e6f4ea", "三千亿个参数 →　1 个 loss",
         "输入多、输出少", "<tspan font-weight=\"700\">从后往前</tspan>（就是反向传播）",
         "⭐ 这就是我们的情况"),
        (OR, "#fef7e0", "10 个参数 →　100 万维输出",
         "输入少、输出多", "<tspan font-weight=\"700\">从前往后</tspan>",
         "⛔ 这时候反向传播反而亏"),
    )
    for i, (col, fill, shape, why, how, note) in enumerate(CASES):
        x = 96 + i * 628
        f.box(x, py3 + 84, 580, 118, fill, col, 8)
        f.t(x + 290, py3 + 112, shape, INK, True, 16, "middle")
        f.t(x + 290, py3 + 138, why, GY, size=13, anchor="middle")
        f.t(x + 290, py3 + 166, "→　" + how, col, True, 15, "middle")
        f.t(x + 290, py3 + 192, note, col, size=12.5, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "判据：看到任何一个「要求一大堆偏导」的问题，先数输入多还是输出多",
                ("⭐ 它决定的不是「算得对不对」——&#160;"
                 "<tspan font-weight=\"700\">两个方向算出来的结果一模一样</tspan>，"
                 "决定的是<tspan font-weight=\"700\">你要把整条链走多少遍</tspan>。",
                 "⛔ 而这正是<tspan font-weight=\"700\">推理里没有的那一半</tspan>："
                 "推理只有一遍前向，压根不存在「往回走」这个动作，"
                 "也就不存在「沿途把梯度收下来」这件事。"))

    yb = f.src(yb + 16,
               "⭐ 这一格是结构，不是数据 ——　图上唯一的量是「三千亿」，"
               "那是模型规模，前面已经立过。",
               "📌 「正向模式 / 反向模式」是自动微分的标准术语；"
               "反向传播是反向模式用在神经网络上的那个特例。")

    f.save("fig4-reverse.svg", yb + 14)


if __name__ == "__main__":
    main()
