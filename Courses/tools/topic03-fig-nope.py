# -*- coding: utf-8 -*-
r"""专题三 · §8.3「NoPE」—— 不是绕过那个挡路的东西，是让它根本不用存在

⭐⭐⭐ 2026-09-13 夜间 R17 新画。这是 fig3-absorb（§5.4b）的**续集**，
   两张图必须连着读：
   · §5.4b 说：MLA 本来可以「吸收」，但 RoPE 往中间塞了个 R，**挡住了**。
     于是只好拆出 64 维一路专门扛位置。
   · 这一张说：**混合架构之后，那个 R 干脆没了。**

⭐⭐ 而且不是「找到了绕过它的技巧」，是**换人干了**：
   线性层（KDA）本身就是按顺序递推的 —— 衰减和门控天生带时序。
   于是「谁负责位置」这件事整个转交出去，全注意力层只管检索。

📌 原话核过（Kimi Linear，arXiv 2510.26692 §2.x）：
   · “we apply NoPE to all full attention (MLA) layers. This design
     **delegates the entire responsibility for encoding positional information
     and recency bias … to the KDA layers**.”
   · “KDA is thus established as the **primary position-aware operator**”
   · “**GDN serves an analogue role to RoPE**”
   · “NoPE **enables their conversion to the highly-efficient pure Multi-Query
     Attention (MQA) during inference**”

⛔ 一个必须守住的边界（§8.3 正文里刚修过，别让图再错一遍）：
   **KV 降 75% 不是 NoPE 的功劳，是 3:1 配比的。** NoPE 省的是那 64 维。

⭐ 这张图真正要教的不是 NoPE，是**约束之间是有连接的**：
   改一个看起来完全无关的架构选择（层怎么配），
   可以让另一个约束（位置编码挡住吸收）**整个消失**。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2, wpx)

W = 1400
DC, DR = 512, 64                  # V3 的形状：576 ＝ 512（可吸收）＋ 64（扛 RoPE）


def main():
    tot = DC + DR
    assert tot == 576
    cut = 100.0 * DR / tot        # 去掉那 64 维，再省多少

    f = Fig(W, "NoPE：不是绕过挡路的那个 R，是让它根本不用存在")
    yy = f.header(
        "NoPE ——&#160;不是绕过挡路的那个东西，是让它<tspan font-weight=\"700\">"
        "根本不用存在</tspan>",
        "讲 MLA 时留了个疙瘩：它本来可以把上投影「吸收」掉，"
        "可 RoPE 往中间塞了个跟位置有关的旋转 <tspan font-weight=\"700\">R</tspan>，"
        "把这条路挡死了，只好拆出 64 维一路专门扛它。"
        "——&#160;<tspan font-weight=\"700\">混合架构之后，那个 R 干脆没了。</tspan>",
        legend=[(RD, "挡路的 R"), (GR, "线性层：天生带顺序"),
                (BL, "全注意力层：只管检索")])

    # ══ ① 单一架构：每层都得自己知道位置 ═══════════════════════════
    PH1 = 330
    top = f.panel(0, yy, W, PH1,
                  "① 今天大多数模型：<tspan font-weight=\"700\">每一层都得自己"
                  "知道先后</tspan>，所以每一层的 K 上都得带个 R",
                  RD, tag="MLA 留下的那个疙瘩")
    # ⛔ 这里原来是 top+48，于是上方那条虚线弧和它的标签（ry−56 / ry−62）
    #   被画到**面板标题栏上面**去了 —— 越界自检查的是下沿，**管不了上沿**。
    #   ⭐ 判据：往上画的东西，护栏一条都没有，只能靠看。
    ry = top + 104
    f.icon("note", 40, ry, 44, 50, BL, "#e8f0fe")
    f.t(62, ry + 68, "q", BL, size=15, anchor="middle")
    f.line(92, ry + 26, 132, ry + 26, GY2, 1.6)
    f.box(140, ry - 2, 96, 54, "#fce8e6", RD, 8)
    f.t(188, ry + 22, "R", RD, bold=True, size=20, anchor="middle")
    f.t(188, ry + 42, "按位置转", RD, size=14, anchor="middle")
    f.line(244, ry + 26, 284, ry + 26, GY2, 1.6)
    f.box(292, ry - 2, 110, 54, "#f3e8fd", PU, 8)
    f.t(347, ry + 30, "W_UK", PU, bold=True, size=17, anchor="middle")
    f.line(410, ry + 26, 450, ry + 26, GY2, 1.6)
    f.icon("box", 458, ry, 46, 50, GR, "#e6f4ea")
    f.t(481, ry + 68, "c", GR, size=15, anchor="middle")

    # 「想把紫块挪过去，被 R 挡住」
    f.p.append('<path d="M 347 %d Q 270 %d 188 %d" fill="none" stroke="%s" '
               'stroke-width="2" stroke-dasharray="5 5"/>'
               % (ry - 12, ry - 56, ry - 12, GY2))
    f.t(268, ry - 62, "想把它挪到 q 那边 ——", GY, size=15, anchor="middle")
    f.t(188, ry + 94, "⛔ 被这个 R 挡住", RD, bold=True, size=17,
        anchor="middle")
    f.t(188, ry + 116, "夹在中间的东西挪不出去", RD, size=14, anchor="middle")

    f.t(560, ry + 4, "于是 MLA 只好拆成两路：", INK, bold=True, size=17)
    f.box(560, ry + 22, 300, 40, "#e6f4ea", GR, 6)
    f.t(710, ry + 47, "512 维 · 可以吸收", GR, bold=True, size=16,
        anchor="middle")
    f.box(872, ry + 22, 150, 40, "#fce8e6", RD, 6)
    f.t(947, ry + 47, "64 维 · 扛 R", RD, bold=True, size=16, anchor="middle")
    f.t(1042, ry + 47, "＝ %d" % tot, INK, bold=True, size=20)
    f.t(560, ry + 92,
        "⭐ 那 64 维<tspan font-weight=\"700\">不是为了存信息</tspan>，"
        "是为了<tspan font-weight=\"700\">给 R 找个不挡路的地方待着</tspan>。",
        GY, size=16)
    f.t(30, top + PH1 - 46,
        "🏠 生活版：办公室里<tspan font-weight=\"700\">每个人都自己戴表对时</tspan>"
        " ——&#160;人人都要，人人都得带着。", INK, size=17)

    # ══ ② 混合之后：这活换人干了 ═══════════════════════════════════
    yy = top + PH1 + 26
    PH2 = 320
    top = f.panel(0, yy, W, PH2,
                  "② 混合架构之后：「谁负责位置」这件事<tspan "
                  "font-weight=\"700\">换人干了</tspan>", GR,
                  tag="Kimi Linear 原话：delegates the entire responsibility")
    ry = top + 56
    # 一条流水线：3 个线性层 + 1 个全注意力层
    seq = [("线性层", GR, "#e6f4ea"), ("线性层", GR, "#e6f4ea"),
           ("线性层", GR, "#e6f4ea"), ("全注意力", BL, "#e8f0fe")]
    for i, (nm, col, tint) in enumerate(seq):
        x = 40 + i * 168
        f.box(x, ry, 140, 74, tint, col, 8)
        f.t(x + 70, ry + 32, nm, col, bold=True, size=17, anchor="middle")
        f.t(x + 70, ry + 56,
            "天生带顺序" if col == GR else "只管检索", GY, size=14,
            anchor="middle")
        if i < 3:
            f.line(x + 146, ry + 37, x + 162, ry + 37, GY2, 1.4)
    f.t(40, ry - 22, "一个循环单元（3 : 1）——&#160;整个模型就是它重复 23 次",
        GY, bold=True, size=16, cls="svglbl")
    f.t(350, ry + 98, "…… 这样的单元再重复 22 次", GY2, size=15,
        anchor="middle")

    f.t(760, ry - 4, "⭐ 线性层<tspan font-weight=\"700\">本来就是一步一步"
        "往下递推的</tspan>：", INK, size=17)
    f.t(760, ry + 22, "它的衰减和门控，<tspan font-weight=\"700\">本身就在编码"
        "「谁先谁后」</tspan>。", GY, size=16)
    f.t(760, ry + 48, "⭐⭐ 所以位置这件事，已经<tspan font-weight=\"700\">"
        "有人干了</tspan>。", GR, bold=True, size=17)
    f.t(760, ry + 78, "→ 全注意力层<tspan font-weight=\"700\">就不用再编一遍"
        "</tspan>。", INK, size=16)
    f.t(760, ry + 104, "→ <tspan font-weight=\"700\">那个 R 没有理由存在了。"
        "</tspan>", RD, bold=True, size=17)

    f.t(30, top + PH2 - 72,
        "📌 论文原话：<tspan font-weight=\"700\">「delegates the entire "
        "responsibility for encoding positional information and recency bias "
        "… to the KDA layers」</tspan>", GY, size=15)
    f.t(30, top + PH2 - 46,
        "🏠 生活版：<tspan font-weight=\"700\">流水线本身就是按顺序走的</tspan>"
        " ——&#160;你在第几站是自明的，<tspan font-weight=\"700\">表可以不戴了"
        "</tspan>。", INK, size=17)

    # ══ ③ 一拿掉，三样东西一起消失 ═══════════════════════════════════
    yy = top + PH2 + 26
    # ⭐⭐ 2026-09-14 R38b：这一格原来是三张并排卡片 ——&#160;
    #   标题写着「三样麻烦**一起**消失」「它们本来看着互不相干」，
    #   可**「一起」这件事图上没有任何东西承载**，读者只能选择相信。
    # ⭐⭐ 判据（可以直接拿去查别的图）：
    #    **面板标题里出现连接词（一起 / 同时 / 因此 / 所以），
    #      图上就必须有一个对应的图形连接物。** 没有就是在用文字冒充结构。
    # ⭐ 于是改成「一个根 → 三条分支」：左边一个被划掉的 R，
    #   一条竖脊分出三根箭头。三样东西挂在同一个根上，一眼就看见。
    # ⚠️ 顺带从三列改一列 ——&#160;正文宽了，2-3 行压成 1 行，
    #   所以净增高只有 20px（268 → 298），不是靠加高换来的。
    # ⭐⭐⭐ 2026-09-14 R49 **再改一次**。R38b 那版（左边一个打叉的 R，
    #   一条竖脊扇出三个文字框）解决了「同一个根」，但没解决「同时」——
    #   ⛔ **三个文字框在图的外面。** 它们跟画面里任何一个几何位置都没关系，
    #     读者只能靠语言把它们连回 R 身上，于是仍然是「逐条相信你三次」。
    #
    # ⭐⭐ 这一版换成**双胞胎**：左右两张同构的图，只有该变的地方变了。
    #   为什么是它 ——&#160;**缺席是画不出来的**。单画一张「R 不在」的流程图，
    #   读者看到的只是一张正常的图，他不会自发去想「这儿本来该有个东西」。
    #   配一个只差这几处的孪生图，缺席才有了坐标：视线在左图找到那个块，
    #   平移到右图同一位置，发现是空的 ——&#160;**减法是读者自己做的**。
    #   而「同时」也终于有了承载物：三个洞出现在同一张图的三个坐标上。
    #
    # ⛔⛔ 实现上最要紧的一条：**两栏不是画两遍，是同一个函数调两次。**
    #   手抄第二遍必然走样，而「只有这里变了」这个信号**经不起任何无关差异**——
    #   框大小、字号、箭头弧度，差一点点就把信号稀释掉了。
    #   所以下面 `col()` 只有一个 `ghost` 开关，两栏的结构差异在代码里**不存在**。
    #
    # ⭐ 四条画法规则（都不是我拍的，是照着做对了的人抄的）：
    #   ① **空槽宽度不收，两栏等宽。** 本能会想把右栏压短来表达「路变短了」——
    #      不行：一收拢，后面的东西全位移，读者看到的是两张不同的图。
    #      ⭐ **空出来的那块地方本身就是信息量**，而且比「变短了」更准确。
    #   ② **幽灵不打叉。** 叉是「被否定」，幽灵是「不在了」，是两个语义。
    #      而且一打叉，你就替读者把减法做完了，装置的全部价值就在于让他自己发现。
    #   ③ **右栏一点强调色都不要有**（尤其不要绿色）。红＝正在被拿掉，
    #      灰虚线＝已经不在了。右栏空着就已经把话说完，再上个绿色等于凭空
    #      多一个「绿是什么意思」要读者判断。
    #   ④ 那根**穿过幽灵框的直箭头**是全图最重要的一笔 ——&#160;
    #      它是唯一「左栏没有、右栏有」的新形状，回答的正是
    #      「东西不在了之后，那条路到底变成什么样」。
    PH3 = 514
    top = f.panel(0, yy, W, PH3,
                  "③ 那三样麻烦<tspan font-weight=\"700\">本来就长在 R 身上"
                  "</tspan> ——&#160;所以它一走，三个位置<tspan "
                  "font-weight=\"700\">同时空了</tspan>", BL,
                  tag="三条都是原文说的")

    # ── 局部坐标系：两栏只差一个 x 平移，同名元素 y 完全相同 ──────
    COLW, COLGAP = 640, 50
    COLX = (30, 30 + COLW + COLGAP)
    assert COLX[1] + COLW <= W - 30, COLX
    LQ, LR, LU, LCC = 0, 84, 216, 366          # q / R / W_UK / c 的局部 x
    WQ, WR, WU, WC = 40, 88, 100, 42
    assert LCC + WC < COLW, "一行放不下就别硬塞"
    FY = top + 52                               # 流程那一行的顶
    FH = 48
    SPINE = LR + WR / 2.0                       # 竖脊：从 R 底下垂下来
    CX0, CHH, CGAP = 150, 34, 12                # 三个徽章
    CY0 = FY + FH + 34
    CW = COLW - CX0 - 6
    # 那条 576 的小条，画在徽章②里 ——&#160;⛔ 两栏必须像素级同起点同长度
    BX, BW512, BW64 = 250, 156, 24
    assert CX0 + BX + BW512 + BW64 < COLW, "徽章②里那条超出栏宽"
    GH = "#9aa0a6"                              # 幽灵灰：比正文淡，但还读得出
    PHANTOM = "14 4 3 4"                        # 制图里的 phantom line：长-短-短

    TROUBLE = [("吸收被挡住", RD), ("多出 %d 维" % DR, RD), ("外推要重标定", RD)]

    def col(ox, ghost):
        """画一栏。⛔ 两栏共用这一个函数 ——&#160;结构差异在代码里不存在。"""
        ink = GH if ghost else INK
        sub = GH if ghost else GY
        # ── 标题：⚠️ 句式必须同构，不然标题本身就成了一处「差异」，
        #    会跟真正的差异抢注意力。
        f.t(ox, FY - 20,
            "R 没了 ——&#160;q 直接吸进去" if ghost else "R 还在 ——&#160;q 吸不进去",
            GY if ghost else RD, bold=True, size=17, cls="svglbl")

        # q（两栏一模一样）
        f.icon("note", ox + LQ, FY, WQ, FH, BL, "#e8f0fe")
        f.t(ox + LQ + WQ / 2, FY + FH + 18, "q", BL, size=14, anchor="middle")

        if ghost:
            # ④ 那一笔：一根不断的粗黑直箭头，**从幽灵框正中穿过去**，不绕开。
            #    左栏是两段带转折的，这里是一根可以一眼看完的直线。
            f.line(ox + LQ + WQ + 4, FY + FH / 2, ox + LU - 4, FY + FH / 2,
                   INK, 2.6)
            # R 的幽灵：同尺寸、无填充、phantom 笔触、标签保留但降灰
            f.box(ox + LR, FY, WR, FH, "none", GH, 8, 1.3, dash=PHANTOM)
            f.t(ox + LR + WR / 2, FY + 30, "R", GH, bold=True, size=20,
                anchor="middle")
        else:
            f.line(ox + LQ + WQ + 4, FY + FH / 2, ox + LR - 4, FY + FH / 2,
                   GY2, 1.5)
            f.box(ox + LR, FY, WR, FH, "#fce8e6", RD, 8)
            f.t(ox + LR + WR / 2, FY + 22, "R", RD, bold=True, size=19,
                anchor="middle")
            f.t(ox + LR + WR / 2, FY + 40, "按位置转", RD, size=12,
                anchor="middle")
            f.line(ox + LR + WR + 4, FY + FH / 2, ox + LU - 4, FY + FH / 2,
                   GY2, 1.5)

        # ⛔ 这里原来把右栏的 W_UK 画淡了一档。**错的** ——&#160;W_UK 根本没变。
        #   一旦它跟左栏不一样，读者就会去想「这个是不是也弱化了」，
        #   而那正是差值自检该抓的东西：**没变的东西必须逐像素一样**，
        #   任何无关差异都在稀释「只有这里变了」这个信号。
        f.box(ox + LU, FY, WU, FH, "#f3e8fd", PU, 8)
        f.t(ox + LU + WU / 2, FY + 30, "W_UK", PU, bold=True, size=16,
            anchor="middle")
        f.line(ox + LU + WU + 4, FY + FH / 2, ox + LCC - 4, FY + FH / 2,
               GY2, 1.5)
        f.icon("box", ox + LCC, FY, WC, FH, GR, "#e6f4ea")
        f.t(ox + LCC + WC / 2, FY + FH + 18, "c", GR, size=14, anchor="middle")

        # ── 竖脊：三个徽章是从 R 身上垂下来的，不是飘在旁边的 ──
        cys = [CY0 + i * (CHH + CGAP) + CHH / 2.0 for i in range(3)]
        f.line(ox + SPINE, FY + FH, ox + SPINE, cys[-1],
               GH if ghost else RD, 1.3, dash=PHANTOM if ghost else None,
               arrow=False)
        for i, cy in enumerate(cys):
            f.line(ox + SPINE, cy, ox + CX0 - 6, cy,
                   GH if ghost else RD, 1.3,
                   dash=PHANTOM if ghost else None, arrow=False)
            lab, lc = TROUBLE[i]
            if ghost:
                # 空槽：**宽度一分不收**，跟左栏的徽章逐像素等大
                f.box(ox + CX0, cy - CHH / 2, CW, CHH, "none", GH, 8, 1.3,
                      dash=PHANTOM)
            else:
                f.box(ox + CX0, cy - CHH / 2, CW, CHH, "#fce8e6", RD, 8, 1.2)
            f.badge(ox + CX0 + 18, cy, i + 1, GH if ghost else lc)
            if not ghost:
                f.t(ox + CX0 + 38, cy + 6, lab, lc, bold=True, size=15)
            # 徽章②里那条 576：⛔ 512 段两栏**同起点、同长度**，
            #   右栏只把 64 那一格换成同笔触的空槽 ——&#160;绝不许拉长填满。
            #   ⭐ 于是「64 没了」和「R 没了」在画面上成了两个笔触相同的洞，
            #     一个字不用读就知道这俩是同一件事的两个面。
            if i == 1:
                f.box(ox + CX0 + BX, cy - 7, BW512, 14, "#e6f4ea", GR, 3)
                f.t(ox + CX0 + BX + BW512 / 2, cy + 5, "%d" % DC,
                    GR, bold=True, size=12, anchor="middle")
                if ghost:
                    f.box(ox + CX0 + BX + BW512 + 3, cy - 7, BW64, 14,
                          "none", GH, 3, 1.2, dash=PHANTOM)
                else:
                    f.box(ox + CX0 + BX + BW512 + 3, cy - 7, BW64, 14,
                          "#fce8e6", RD, 3)
                    f.t(ox + CX0 + BX + BW512 + 3 + BW64 / 2, cy + 5,
                        "%d" % DR, RD, bold=True, size=11, anchor="middle")
                f.t(ox + CX0 + BX + BW512 + BW64 + 12, cy + 5,
                    "＝ %d" % tot if not ghost else "＝ %d" % DC,
                    ink, bold=True, size=13)
        return cys

    cysA = col(COLX[0], False)
    cysC = col(COLX[1], True)
    # ⛔ 两栏的 y 必须逐个相等 ——&#160;差一个像素，读者就得先做一遍图匹配，
    #   而匹配没做完，比较根本不会开始。
    assert cysA == cysC, (cysA, cysC)

    # ⛔ 这里原来在两栏中间放了个大「→」。去掉了：它跟流程行同一个 y，
    #   紧挨着右栏的 q，**读起来像流程的一部分**而不是「前后对照」。
    #   ⭐ 两个标题（R 还在 / R 没了）已经把顺序说清了，多这一笔只会加噪声。

    # ── 三个洞各自变成了什么 ────────────────────────────────────
    from topic03_draw import wrap_rich
    ROWS = [
        ("吸收完全生效",
         "没有 R 夹在中间，上投影可以整个吸进 q 那一侧。"
         "<tspan font-weight=\"700\">推理时 MLA 直接退化成纯 MQA。</tspan>", GR),
        ("那 %d 维没了" % DR,
         "%d ＝ %d ＋ %d 里的 %d 整个消失 ——&#160;"
         "<tspan font-weight=\"700\">在这一步之上再省 %.1f%%</tspan>。"
         % (tot, DC, DR, DR, cut), BL),
        ("不用再调外推",
         "没有位置编码，就<tspan font-weight=\"700\">没有外推要重标定</tspan>。"
         "长上下文扩展里最烦人的一块调参，直接不存在了。", PU),
    ]
    # ⚠️ 这里的间距是**量出来的**不是拍的：+22 时小标题贴着左栏徽章③，
    #   而落点那句压在第③行的徽章上。两处都要留 ≥30px。
    RY0, RH = cysC[-1] + CHH / 2 + 40, 42
    f.t(30, RY0 - 8, "右边那三个空槽，分别变成了：", INK, bold=True, size=16)
    for i, (ttl, body, colr) in enumerate(ROWS):
        by = RY0 + 6 + i * RH
        f.badge(46, by + 14, i + 1, colr)
        f.t(70, by + 20, ttl, colr, bold=True, size=17, cls="svglbl")
        rows_ = wrap_rich(body, W - 250, 16 * 1.12)
        # ⛔ 一行放不下就是布局没算对 ——&#160;这一格的收益全来自「一行一条」
        assert len(rows_) == 1, (ttl, len(rows_))
        f.t(210, by + 20, rows_[0], GY, size=16)
    # ⛔ `top` 是**内容起点**（panel 返回的是 yy+30），所以面板下沿是
    #   top + PH3 - 30。这里原来写 -22，等于越过下沿 8px ——&#160;
    #   加高 PH3 治不了，因为这一行跟着 PH3 一起往下走。别人的面板用 -46。
    f.t(30, top + PH3 - 52,
        "⭐⭐ <tspan font-weight=\"700\">注意这不是「找到了绕过 R 的技巧」"
        "</tspan> ——&#160;是让别人替它把活干了，"
        "于是 <tspan font-weight=\"700\">R 根本不用存在</tspan>。"
        "<tspan font-weight=\"700\">被绕过</tspan>和"
        "<tspan font-weight=\"700\">不存在</tspan>，是两件事。", INK, size=17)

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH3 + 30
    yy = f.band(yy, "info", "这张图真正想教的不是 NoPE，是约束之间有连接", [
        "一个看起来<tspan font-weight=\"700\">纯粹是效率考虑</tspan>的选择"
        "（层怎么配比），解开了一个看起来<tspan font-weight=\"700\">完全无关"
        "</tspan>的约束（位置编码挡住吸收）。",
        "⭐ 本讲这样的连接已经出现过好几次："
        "MLA 那次是「为了保住一个代数变换，把功能拆成两路」；"
        "这里是它的反面 ——&#160;<tspan font-weight=\"700\">"
        "为了不再需要那个变换，干脆换个人来提供它的前提</tspan>。",
        "⭐⭐ 所以看到一个约束的时候，除了问「怎么绕过」，"
        "还要问一句：<tspan font-weight=\"700\">它凭什么在那儿？"
        "有没有别人能把它那份活接走？</tspan>",
    ])
    yy = f.band(yy + 14, "bad", "⛔ 一条很容易记错的账，这里说死", [
        "「Kimi Linear 把 KV 降了 75%」<tspan font-weight=\"700\">"
        "不是 NoPE 的功劳</tspan> ——&#160;那 75% 来自 "
        "<tspan font-weight=\"700\">3:1 的配比</tspan>"
        "（四层里只有一层是全注意力），跟加不加位置编码<tspan "
        "font-weight=\"700\">没关系</tspan>。",
        "⭐ NoPE 省的是③里那一条：<tspan font-weight=\"700\">%d 维里的那 %d 维"
        "</tspan>。两笔要分开记。" % (tot, DR),
    ], fold=True)
    yy = f.src(yy + 16,
               "四句原话均出自 Kimi Linear（arXiv 2510.26692）："
               "「we apply NoPE to all full attention (MLA) layers」·"
               "「delegates the entire responsibility for encoding positional "
               "information and recency bias … to the KDA layers」·"
               "「KDA is thus established as the primary position-aware "
               "operator」·「NoPE enables their conversion to the "
               "highly-efficient pure Multi-Query Attention (MQA) during "
               "inference」",
               "⚠️ 图里 %d ＝ %d ＋ %d 用的是 <tspan font-weight=\"700\">"
               "DeepSeek-V3 的形状</tspan>（当尺子用，机制一样）；"
               "Kimi Linear 自己那套 MLA 超参本课<tspan font-weight=\"700\">"
               "没有核过</tspan>，别把这三个数安到它头上"
               % (tot, DC, DR),
               "⭐ K3 是照搬 Kimi Linear 这套做法，不是它先做的 ——&#160;"
               "K3 自己写的是「follows the hybrid design of Kimi Linear」")
    f.save("fig3-nope.svg", yy + 6)


main()
