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
  ② **链式法则的推导。** 这一节要的是「用到谁」，不是「怎么证」。

⛔⛔ 2026-09-23 现场：「这一段完全看不懂 ——&#160;为什么转置啊？
  然后什么前向反向之类的，这是啥东西？中间那个第二步向第一步的箭头又是怎么回事？」
  三条全中，而且三条是**同一个毛病的三个出口**：
  ⭐ 这张图原来的定位是「数乘法做了几次」，所以它**只画了算式，没画意思**。
    可算式里偏偏立着两个 ᵀ ——&#160;一个没解释过的记号，
    它不会让人「先跳过」，它会让人**停在那儿，后面全不看了**。
  ⛔ 判据：**图上任何一个没被解释过的记号，都是一道闸门，不是一个细节。**
    你以为读者会绕过去，其实他停在那里了。
  ⭐ 所以补了一整格 Ⓑ 专讲转置：先说它是什么（行列对调，一个数都没变），
    再说为什么非它不可（结果形状一旦定死，转置是唯一摆得上的方式）。
  ⛔ 另外两处是实打实的错：
    · 那根「输入是存下来的」箭头**画反了** ——&#160;箭头指着 ①，
      可流向是 ① 存下来 →&#160;② 取用。图里别的箭头全是数据流向，
      **只有它是「引用指针」的读法**，于是它被读成了「② 产生了 ①」。
    · 权重 W 原来画成**正方形**（两边都是 D_PX）。V3 里没有这样的方阵，
      而且这一格正想说明「形状决定了怎么摆」——&#160;画成方的就什么也说明不了。
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
    # ⛔ 2026-09-23：原来只有一个 D_PX，于是 W 被画成了**正方形**。
    #   V3 里没有两边都一样的权重矩阵，而这一格恰恰要说「形状决定了怎么摆」——&#160;
    #   画成方的，转置看上去就成了「什么都没变」。拆成进、出两个尺度。
    T_PX = 120.0                      # token 维（一条序列有多少个位置）
    DIN_PX, DOUT_PX = 84.0, 56.0      # 进来的宽度 d_in / 出去的宽度 d_out

    # ⭐ 三次乘法的形状（行 × 列，单位就是上面那两个尺度）
    #   FLOPs 都 ∝ 行 × 内维 × 列 ——&#160;脚本当场验它们真的相等
    MULS = (
        (BL, "① 前向", "算这一层的输出",
         ("输入 X", T_PX, DIN_PX, False), ("权重 W", DIN_PX, DOUT_PX, False),
         ("输出 Y", T_PX, DOUT_PX, False), "→ 交给下一层"),
        (RD, "② 反向 · 权重梯度", "算「我这块权重该怎么改」",
         ("输入 Xᵀ", DIN_PX, T_PX, True), ("上游责任 dY", T_PX, DOUT_PX, False),
         ("权重梯度 dW", DIN_PX, DOUT_PX, False), "→ 交给优化器"),
        (RD, "③ 反向 · 传给下游", "算「前一层该收到什么」",
         ("上游责任 dY", T_PX, DOUT_PX, False), ("权重 Wᵀ", DOUT_PX, DIN_PX, False),
         ("新的责任 dX", T_PX, DIN_PX, False), "→ 交给前一层"),
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
            # ⛔⛔ 2026-09-23：这根线原来**箭头指着 ①**（"这块引用自上一行"）。
            #   可这张图里别的箭头全是**数据流向**（→ 交给下一层 / → 交给优化器），
            #   只有它是「引用指针」的读法 ——&#160;于是现场直接读成了「② 产生了 ①」。
            #   ⭐ 判据：**同一张图里只能有一种箭头语义。**
            #     混进第二种，读者不会去猜是哪一种，他会按最常见那种读。
            #   改成顺着流向走：① 前向存下来 →&#160;② 反向取用。
            a_use = cy + 18                      # ②行 Xᵀ 框的竖直中心（取用端）
            a_src = cy - ROW + 18                # ①行 X 框的竖直中心（存下来那端）
            gx = MX0 - 30                        # 左侧让出来的走线
            f.line(MX0 - 2, a_src, gx, a_src, OR, 2.0, arrow=False)
            f.line(gx, a_src, gx, a_use, OR, 2.0, arrow=False)
            f.line(gx, a_use, MX0 - 4, a_use, OR, 2.0)      # 箭头落在②行那个框上
            # ⛔ 注解原来落在 cy−34，而②行 dY 框顶在 cy−41 ——&#160;**字直接压进框里**。
            #   ⭐ 挪到两行之间那条空带（①行图注底 cy−73 与②行框顶 cy−41 之间）。
            f.t(MX0 + T_PX + 30, cy - 56,
                "⭐ 这一块<tspan font-weight=\"700\">不是新算的</tspan>"
                "　——　是 ① 里那个输入<tspan font-weight=\"700\">被存下来了</tspan>",
                OR, True, 13.5)

    # ⛔ 2026-09-23：原来这句写「三块积的面积一样大」。可画面上三块积的
    #   面积**并不一样**（Y 是 T×d_out、dW 是 d_in×d_out、dX 是 T×d_in）——&#160;
    #   一样的是**乘加次数**，三次都是 T × d_in × d_out。
    #   ⭐ 判据：**别用一个读者能在画面上当场证伪的说法去撑结论。**
    #     他证伪的不是那句话，是整张图。
    f.t(700, py + 78 + 3 * ROW - 40,
        "⭐⭐⭐ 三次乘法的<tspan font-weight=\"700\">乘加次数一模一样</tspan>"
        "（都是「位置数 × 进来的宽度 × 出去的宽度」）"
        "　——　所以 <tspan font-weight=\"700\">1 ＋ 2 ＝ 3</tspan> 不是个比喻，"
        "是<tspan font-weight=\"700\">数出来的</tspan>。",
        INK, size=15, anchor="middle")
    f.t(700, py + 78 + 3 * ROW - 14,
        "⛔ ② 那块虚线的，就是<tspan font-weight=\"700\">激活扔不掉的全部原因</tspan>。"
        "　·　⭐ 那两个 <tspan font-weight=\"700\">ᵀ</tspan> 是怎么回事？"
        "<tspan font-weight=\"700\">下面 Ⓑ 一整格专讲它。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 那两个「ᵀ」到底是怎么回事 ═══════════════════════
    # ⭐⭐⭐ 2026-09-23 新加。现场原话：「为什么转置啊？」
    #   ⛔ 讲法上的取舍：**不讲链式法则，也不讲「梯度的矩阵形式」。**
    #     那两条都是「怎么证」，而现场问的是「这是啥」。
    #   ⭐ 分两层答，顺序不能倒：
    #     ① 它是什么 ——&#160;行列对调，**一个数都没变**。先把「它很高深」这个
    #        印象拆掉，后面那层才听得进去。
    #     ② 为什么非它不可 ——&#160;矩阵乘要求「左边的列数 ＝ 右边的行数」，
    #        而结果的形状是被定死的（dW 得跟 W 同形、dX 得跟 X 同形），
    #        于是**转置是唯一摆得上的方式**。它不是一步额外运算。
    PHT = 522
    pyt = f.panel(0, py + PH + 22, W, PHT,
                  "Ⓑ 那两个「ᵀ」是怎么回事 ——　<tspan font-weight=\"700\">"
                  "转置不是多算了一步，是同一张表换个方向读</tspan>", OR,
                  sub="⭐ 分两层：先说它<tspan font-weight=\"700\">是什么</tspan>，"
                      "再说<tspan font-weight=\"700\">为什么非它不可</tspan>")

    # ── 左半：转置是什么（一个数都没动）──────────────────────────
    f.t(48, pyt + 46, "① 它是什么", OR, True, 16)
    f.t(48, pyt + 70, "行变列、列变行　——　一个数都没动", GY, size=13)

    CELL = 34
    MA = ((1, 2, 3), (4, 5, 6))
    ax, ay = 64, pyt + 92
    for r in range(2):
        for c in range(3):
            f.box(ax + c * CELL, ay + r * CELL, CELL, CELL,
                  "#fef7e0" if r == 0 else "#fff", GY2, 3)
            f.t(ax + c * CELL + CELL / 2, ay + r * CELL + CELL / 2 + 6,
                str(MA[r][c]), INK, True, 14, "middle")
    f.t(ax + 1.5 * CELL, ay + 2 * CELL + 24, "A　（2 行 3 列）",
        INK, True, 13, "middle")

    f.line(ax + 3 * CELL + 18, ay + CELL, ax + 3 * CELL + 62, ay + CELL, OR, 2.2)
    f.t(ax + 3 * CELL + 40, ay + CELL - 14, "转置", OR, True, 12.5, "middle")

    bx = ax + 3 * CELL + 80
    for r in range(3):
        for c in range(2):
            f.box(bx + c * CELL, ay + r * CELL, CELL, CELL,
                  "#fef7e0" if c == 0 else "#fff", GY2, 3)
            f.t(bx + c * CELL + CELL / 2, ay + r * CELL + CELL / 2 + 6,
                str(MA[c][r]), INK, True, 14, "middle")
    f.t(bx + CELL, ay + 3 * CELL + 24, "Aᵀ　（3 行 2 列）", INK, True, 13, "middle")

    f.t(48, ay + 3 * CELL + 68,
        "⭐ 黄色那<tspan font-weight=\"700\">一行</tspan>，转完变成了黄色那"
        "<tspan font-weight=\"700\">一列</tspan>。", OR, True, 13.5)
    f.t(48, ay + 3 * CELL + 92,
        "数一个没少、值一个没变　——　变的只是<tspan font-weight=\"700\">"
        "「从哪个方向读」</tspan>。", GY, size=13)
    f.t(48, ay + 3 * CELL + 118,
        "⛔ 所以它<tspan font-weight=\"700\">不花算力</tspan>，"
        "真实实现里常常连搬都不搬。", GY2, size=12.5)

    # ── 右半：为什么非它不可（形状逼出来的）──────────────────────
    sx = 470
    f.t(sx, pyt + 46, "② 为什么非它不可", OR, True, 16)
    f.t(sx, pyt + 70,
        "矩阵乘有一条死规矩：<tspan font-weight=\"700\">"
        "左边那个的列数，必须等于右边那个的行数</tspan>", GY, size=13)

    CHAIN = (
        (BL, "①", "X ［T × d_in］", "W ［d_in × d_out］", "Y ［T × d_out］",
         "d_in", "原本的摆法：把输入按权重混一遍"),
        (RD, "②", "Xᵀ ［d_in × T］", "dY ［T × d_out］", "dW ［d_in × d_out］",
         "T（位置）", "要得到一块<tspan font-weight=\"700\">跟 W 同形</tspan>的结果 ——　"
         "只有这一种摆法"),
        (RD, "③", "dY ［T × d_out］", "Wᵀ ［d_out × d_in］", "dX ［T × d_in］",
         "d_out", "要得到一块<tspan font-weight=\"700\">跟 X 同形</tspan>的结果 ——　"
         "只有这一种摆法"),
    )
    for i, (col, tag, a_, b_, c_, kill, why) in enumerate(CHAIN):
        yy = pyt + 112 + i * 92
        f.t(sx, yy, tag, col, True, 16)
        f.t(sx + 30, yy, "%s　×　%s　＝　%s" % (a_, b_, c_), INK, True, 14.5)
        f.t(sx + 30, yy + 26,
            "中间对上的那一维是 <tspan font-weight=\"700\" fill=\"%s\">%s</tspan>"
            "　——　乘完它就<tspan font-weight=\"700\">被吃掉了</tspan>" % (OR, kill),
            GY, size=12.5)
        f.t(sx + 30, yy + 48, why, col, size=12.5)

    f.box(40, pyt + 400, 1320, 98, "#fff8e1", OR, 8)
    f.t(64, pyt + 430,
        "⭐ 所以转置<tspan font-weight=\"700\">不是一步额外的运算</tspan>："
        "你要的结果形状一旦定死 ——　dW 得跟 W 同形、dX 得跟 X 同形 ——　"
        "<tspan font-weight=\"700\">转置就是唯一摆得上的方式</tspan>。", INK, size=14.5)
    f.t(64, pyt + 458,
        "⭐⭐ 换成人话：前向问的是「<tspan font-weight=\"700\">输入怎么混成输出</tspan>」；"
        "③ 问的是反过来那个问题 ——　「<tspan font-weight=\"700\">"
        "输出的责任怎么分回输入</tspan>」。", INK, size=14.5)
    f.t(64, pyt + 484,
        "<tspan font-weight=\"700\">同一张 W，换个方向读，写出来就是 Wᵀ。</tspan>"
        "　·　而 ② 里 Xᵀ 的作用是<tspan font-weight=\"700\">把「位置」摆到中间去让它被吃掉</tspan>"
        "　——　这就是 1.2d 说的那笔「结账」。", INK, size=14.5)
    f._pan = None

    # ══════════ Ⓒ 岔路：一进两出 ═════════════════════════════════
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
    py2 = f.panel(0, pyt + PHT + 22, W, PH2,
                  "Ⓒ ⭐⭐⭐ 全图的钥匙：<tspan font-weight=\"700\">"
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

    # ══════════ Ⓓ 加起来 ═════════════════════════════════════════
    PH3 = 250
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3,
                  "Ⓓ 于是这笔账就封口了", GR)

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
