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

    # ══════════ Ⓒ 反向是「同一张网倒着走」 ═══════════════════════
    # ⭐⭐⭐ 2026-09-17 新增。取自**李宏毅**（台大）那套讲法：
    #   他在投影片上把 backward pass 画成**另一张 neural network** ——
    #   同样的拓扑、边反过来、权重转置，而每个「神经元」的动作
    #   从「套一个非线性」换成了 **multiply a constant**（原文用词），
    #   那个常数是 σ′(z)，**在前向就被定死了**。
    # ⭐⭐ 为什么这一格值得单画：前面 Ⓐ Ⓑ 回答的是「往哪边走、走几遍」，
    #   而这一格回答的是**「反向那一遍长什么样」** ——&#160;
    #   答案是「跟前向一模一样，只是三处换了」。
    #   ⛔ 这比「链式法则一层层往回乘」好懂得多，因为它给了一个**形状**。
    # ⭐⭐⭐ 而第三行那个「常数在前向就定下来了」，
    #   是这一讲「激活扔不掉」的**第三次**出现（前两次：1.6 的闭环、小电路的乘法门）
    #   ——&#160;三条路殊途同归，这正是它该被反复说的理由。
    # 📌 出处：李宏毅《Backpropagation》课程投影片（台大）——&#160;
    #   投影片上写着 "new type of neuron / multiply a constant" 与 (W^{l+1})ᵀ。
    #   ⛔ 图是我们自己重画的。
    PH_M = 432
    pym = f.panel(0, py2 + PH2 + 20, W, PH_M,
                  "Ⓒ ⭐⭐⭐ 换个看法：<tspan font-weight=\"700\">"
                  "反向那一遍，其实就是同一张网倒着走</tspan>", BL,
                  sub="⭐ 不是「另一套算法」——&#160;"
                      "<tspan font-weight=\"700\">同样的形状，只有三处换了</tspan>")

    # ⭐⭐⭐ 2026-09-18 重画。旧版是三行对照表（走的方向 / 每条边乘的 /
    #   每个节点做的）——&#160;ink 全 0，把字删掉什么都不剩。
    #   ⛔ 而这一格的命题是「**同一张网倒着走**」——&#160;
    #     它本来就是一张**网络图**，写成三行表格等于把形状描述成文字、
    #     再让读者翻译回形状。
    #   ⭐⭐⭐ 判据：**「两样东西的骨架一样、只有几处不同」这种话，
    #     必须画成两张骨架真的一样的图** ——&#160;
    #     「一样」是看出来的，列成表反而把它拆散了。
    #   ⭐ 又一次用上 Olah 那招（见素材库 🅒）：同一张底图画两遍，只移动高亮。

    # 一张 2–3–2 的小网。两边**用同一组坐标**，这是「同一张网」的全部意思
    LAYERS = ((2, 120), (3, 100), (2, 120))      # (节点数, 首节点相对 y)
    GAPY, DX = 66, 132

    def _nodes(x0, base_y):
        out = []
        for li, (n, y0_) in enumerate(LAYERS):
            out.append([(x0 + li * DX, base_y + y0_ + k * GAPY) for k in range(n)])
        return out

    NY = pym + 40
    FWD = _nodes(196, NY)
    BWD = _nodes(846, NY)
    # ⭐ 这条 assert 就是这一格的命题本身：两张网的形状必须逐点相同
    assert [[(x - 196, y) for x, y in L] for L in FWD] == \
           [[(x - 846, y) for x, y in L] for L in BWD], \
        "两边骨架对不上 —— 那就讲不成「同一张网」了"

    def draw_net(NS, back):
        """back=False 前向，True 反向。**除了下面这三处，两边一模一样。**"""
        ecol = GR if back else GY2
        for li in range(len(NS) - 1):
            for (x1, y1) in NS[li]:
                for (x2, y2) in NS[li + 1]:
                    # ① 方向：反向就是把每条边的箭头掉个头
                    a, b = ((x2, y2), (x1, y1)) if back else ((x1, y1), (x2, y2))
                    f.line(a[0] + (14 if back else 14), a[1],
                           b[0] - (14 if back else 14), b[1],
                           ecol, 1.0, arrow=False)
            mx = (NS[li][0][0] + NS[li + 1][0][0]) / 2.0
            ax0, ax1 = (mx + 22, mx - 22) if back else (mx - 22, mx + 22)
            f.line(ax0, NY + 264, ax1, NY + 264, ecol, 2.4)
            # ② 边上乘的：同一个 W，反向是它的转置
            f.t(mx, NY + 252, "Wᵀ" if back else "W", ecol, True, 14, "middle")

        for li, layer in enumerate(NS):
            for (x, y) in layer:
                if back:
                    # ③ 节点做的：从「套一个非线性」换成「乘一个常数」——
                    #    画成放大器那个三角，跟圆形的视觉差别一眼可见
                    # ⛔ 尖端必须朝**左** ——&#160;放大器那个三角的尖，
                    #   指的就是信号往哪儿流，而这一遍是往回走的。
                    #   ⭐ 第一版画成朝右了。方向这东西，
                    #     在「我知道它该往哪走」的脑子里最容易翻面。
                    f.path("M %.1f %.1f L %.1f %.1f L %.1f %.1f Z"
                           % (x + 12, y - 14, x + 12, y + 14, x - 15, y),
                           RD, 2.0, arrow=False, fill="#fce8e6")
                else:
                    f.box(x - 15, y - 15, 30, 30, "#fff", BL, 15, sw=2)
                    # 圆里那道小 S 就是激活函数本人
                    f.path("M %.1f %.1f C %.1f %.1f %.1f %.1f %.1f %.1f"
                           % (x - 7, y + 5, x - 2, y + 5, x + 2, y - 5, x + 7, y - 5),
                           BL, 1.6, arrow=False)

    draw_net(FWD, False)
    draw_net(BWD, True)

    f.t(196 + DX, NY + 66, "前向", BL, True, 17, "middle")
    f.t(846 + DX, NY + 66, "反向", RD, True, 17, "middle")

    # 中间那句话：骨架是一样的
    f.t(700, NY + 168, "同一张网", GY, True, 16, "middle")
    f.t(700, NY + 196, "只有三处换了", GY2, size=13, anchor="middle")
    f.line(700, NY + 120, 700, NY + 150, GY2, 1.1, dash="4 4", arrow=False)
    f.line(700, NY + 214, 700, NY + 250, GY2, 1.1, dash="4 4", arrow=False)

    # ⭐⭐⭐ 这一格真正的落点：那个常数是**从前向那边拿的**。
    #   ⭐ 标签就贴在它说的那个三角形旁边 ——&#160;
    #     判据：**横跨半张图的箭头既压别人的位置，又让读者去追它指向哪儿。**
    _bx, _by = BWD[2][0]
    f.line(_bx + 18, _by, _bx + 58, _by, OR, 1.2, dash="3 3", arrow=False)
    f.box(_bx + 58, _by - 32, 208, 62, "#fef7e0", OR, 6)
    f.t(_bx + 162, _by - 10, "这个常数是 σ′(z)", OR, True, 13.5, "middle")
    f.t(_bx + 162, _by + 14, "<tspan font-weight=\"700\">前向那一遍算好的</tspan>",
        INK, size=13, anchor="middle")

    # 三处不同，编号标在各自发生的地方
    for i, (_s, col, tx, ty) in enumerate((
            ("① 每条边的箭头掉了个头", GR, 700, NY + 306),
            ("② 同一个 W，转置过来　——　没有新参数", GR, 196 + DX, NY + 306),
            ("③ 节点从「套非线性」换成「乘一个常数」", RD, 846 + DX, NY + 306))):
        f.t(tx, ty, _s, col, True, 13, "middle")

    f.t(700, NY + 352, "⭐⭐⭐ 落点在第 ③ 处："
        "<tspan font-weight=\"700\">那个常数在前向就定下来了</tspan>"
        "　——　所以前向算出来的东西<tspan font-weight=\"700\">必须留在场上</tspan>，"
        "反向才有东西可乘。", INK, size=14.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓓ 判据 ＋ 反例 ══════════════════════════════════════
    PH3 = 222
    py3 = f.panel(0, pym + PH_M + 20, W, PH3,
                  "Ⓓ ⭐⭐ 所以规则只有一条，而且它<tspan font-weight=\"700\">"
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
