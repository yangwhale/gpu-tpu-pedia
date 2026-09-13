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
        "上一节（§5.4b）留了个疙瘩：MLA 本来可以把上投影「吸收」掉，"
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
                  RD, tag="§5.4b 那个疙瘩")
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
    PH3 = 268
    top = f.panel(0, yy, W, PH3,
                  "③ R 一拿掉，<tspan font-weight=\"700\">三样麻烦一起消失"
                  "</tspan> ——&#160;而它们本来看着互不相干", BL,
                  tag="三条都是原文说的")
    for i, (ttl, body, col) in enumerate([
        ("吸收完全生效",
         "没有 R 夹在中间，上投影可以整个吸进 q 那一侧。"
         "<tspan font-weight=\"700\">推理时 MLA 直接退化成纯 MQA。</tspan>", GR),
        ("那 64 维没了",
         "%d ＝ %d ＋ %d 里的 %d 整个消失 ——&#160;"
         "<tspan font-weight=\"700\">在这一步之上再省 %.1f%%</tspan>。"
         % (tot, DC, DR, DR, cut), BL),
        ("不用再调外推",
         "没有位置编码，就<tspan font-weight=\"700\">没有外推要重标定</tspan>。"
         "长上下文扩展里最烦人的一块调参，直接不存在了。", PU),
    ]):
        bx = 30 + i * 452
        f.box(bx, top + 34, 426, 132, "none", LINE, 9)
        f.badge(bx + 16, top + 50, i + 1, col)
        f.t(bx + 62, top + 72, ttl, col, bold=True, size=18, cls="svglbl")
        yy2 = top + 106
        from topic03_draw import wrap_rich
        for r in wrap_rich(body, 394, 16 * 1.12):
            f.t(bx + 16, yy2, r, GY, size=16)
            yy2 += 24
    f.t(30, top + PH3 - 44,
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
        "§5.4b「为了保住一个代数变换，把功能拆成两路」；"
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
    ])
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
