# -*- coding: utf-8 -*-
r"""专题三 · §六「NSA 的三条路是被什么逼出来的」（2026-09-13 夜间 20 轮 · R4）。

⭐⭐ 这一张的结构跟别的图不一样，是**故意的**：
   它先讲**坑**，再讲**路**。因为 NSA 论文自己就是这么写的 ——
   §2 整节叫「重新审视稀疏注意力方法」，把前人踩的坑一条条拆开，
   §3 的三条分支**每一条都对着一个坑**。

  ① **事后稀疏的四个坑**（全部出自 NSA §2，不是我们归纳的）
     · 省了计算没省时间：解码期稀疏，prefill 期却要先算注意力图、建索引
     · 离散操作断梯度：k-means、SimHash 这类选择不可导，学不到「该怎么挑」
     · 按 token 选 → 访存不连续 → 用不了 FlashAttention → 掉回低利用率
     · ⭐⭐ 对 GQA 致命的一条：每个头独立选，**同组各头选择的并集**
       才是真正要搬的内存量 ——&nbsp;**计算稀疏了，访存没稀疏。**

  ② **三条路，一条对一个坑**：压缩（粗看）＋ 选择（细看，**按块**）
     ＋ 滑窗（近处），用一个**学出来的门**加权合起来。

  ③ **native 到底 native 在哪**：事后稀疏让模型偏离预训练轨迹。
     ⚠️ 这里有一处**口径打架**，必须如实呈现：H2O 说「95% 稀疏、5% 够用」，
     NSA 引的 Chen 等 2024 说「top 20% 只覆盖 70% 的注意力分数」。
     ⭐ 方向一致、程度不一致 ——&nbsp;**别把任何一个当普适常数。**
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "NSA 的三条路是被什么逼出来的：先看事后稀疏的四个坑，"
               "再看三条分支怎么一条对一个坑，最后是 native 的含义")
    f.marks = set()
    y0 = f.header(
        "NSA　——　三条路不是设计出来的，是被四个坑逼出来的",
        "⭐ 这张图<tspan font-weight=\"700\">先讲坑再讲路</tspan>，"
        "因为论文自己就是这么写的：§2 拆坑，§3 每条分支对着一个坑",
        [(RD, "事后稀疏踩的坑"), (GR, "NSA 的对策"),
         (OR, "硬件逼出来的"), (BL, "学出来的，不是写死的")])

    ph = 452

    # ══ ① 四个坑 ════════════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 事后稀疏的四个坑", RD,
                 sub="NSA §2 自己列的")

    yy = py + 22
    PITS = [
        ("省了计算，没省时间",
         "解码期稀疏，prefill 期却要先算注意力图、建索引",
         "H2O 这一类"),
        ("离散操作把梯度断了",
         "k-means、SimHash 这种选择不可导 —— <tspan font-weight=\"700\">学不到「该怎么挑」</tspan>",
         "ClusterKV / MagicPIG"),
        ("按 token 选 → 访存不连续",
         "从 KV cache 里捞一个个散落的 token，<tspan font-weight=\"700\">FlashAttention 用不上</tspan>",
         "HashAttention"),
        ("在 GQA 上「稀疏」会失效",
         "每个头各选各的，真正要搬的是<tspan font-weight=\"700\">同组各头的并集</tspan>",
         "Quest"),
    ]
    for i, (title, body, who) in enumerate(PITS):
        h = 84 if i == 3 else 76
        col = OR if i >= 2 else RD
        f.box(x + 22, yy, pw - 44, h, "#fff", col, 8)
        f.box(x + 22, yy, 4, h, col, col, 2)
        f.box(x + 24, yy, 3, h, "#fff", "#fff", 0)
        f.t(x + 40, yy + 24, "坑%d　%s" % (i + 1, title), col, True, 12.5)
        f.t(x + 40, yy + 46, body, GY, size=11.5, w=pw - 80)
        f.t(x + 40, yy + 66, "例：" + who, GY2, size=11)
        if i == 3:
            f.t(x + 40, yy + 78, "⭐⭐ 计算稀疏了，访存没稀疏", OR, True, 11.5)
        yy += h + 10
    fits(yy, y0, ph, "① 四个坑")

    # ══ ② 三条路 ════════════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 三条路，一条对一个坑", GR,
                 sub="并行跑，门控加权合起来")

    yy = py + 22
    ROADS = [
        ("压缩　粗看一遍全局", "连续块用一个<tspan font-weight=\"700\">可学的 MLP</tspan>压成一个 key",
         "带块内位置编码", GR),
        ("选择　细看要紧的几块", "⭐ <tspan font-weight=\"700\">按块选，不按 token 选</tspan> —— 坑③逼的",
         "组内共享选择 —— 坑④逼的", GR),
        ("滑窗　近处永远保留", "最近的一段，原样给",
         "局部信息不该靠「挑」来保证", GR),
    ]
    for title, body, note, col in ROADS:
        f.box(x + 22, yy, pw - 44, 82, "#fff", col, 8)
        f.box(x + 22, yy, 4, 82, col, col, 2)
        f.box(x + 24, yy, 3, 82, "#fff", "#fff", 0)
        f.t(x + 40, yy + 24, title, col, True, 12.5)
        f.t(x + 40, yy + 47, body, GY, size=11.5, w=pw - 80)
        f.t(x + 40, yy + 68, note, GY2, size=11, w=pw - 80)
        yy += 92

    yy += 4
    f.box(x + 22, yy, pw - 44, 86, "#fff", BL, 8)
    f.box(x + 22, yy, 4, 86, BL, BL, 2)
    f.box(x + 24, yy, 3, 86, "#fff", "#fff", 0)
    f.t(x + 40, yy + 24, "怎么合：o ＝ Σ g<tspan baseline-shift=\"sub\" "
        "font-size=\"8\">c</tspan> · Attn(q, 第 c 条路)", BL, True, 12.5)
    f.t(x + 40, yy + 46, "门 g 由一个 MLP ＋ sigmoid 算出来", GY, size=11.5)
    f.t(x + 40, yy + 67, "⭐ 配比是<tspan font-weight=\"700\">学出来的</tspan>，不是写死的三分之一", BL,
        size=11.5)
    fits(yy + 86, y0, ph, "② 三条路")

    # ══ ③ native 在哪 ═══════════════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ native 到底 native 在哪", PU,
                 sub="不是「训的时候也开着」这么简单")

    yy = py + 24
    f.t(x + 22, yy, "论文的论点只有一句：", GY, size=12)
    yy += 22
    f.box(x + 22, yy, pw - 44, 58, "#fff", PU, 8)
    f.box(x + 22, yy, 4, 58, PU, PU, 2)
    f.box(x + 24, yy, 3, 58, "#fff", "#fff", 0)
    f.t(x + 40, yy + 24, "事后加稀疏，等于<tspan font-weight=\"700\">把模型推离它的预训练轨迹</tspan>",
        PU, True, 12.5)
    f.t(x + 40, yy + 45, "那些靠全局检索的头，是最先被剪坏的", GY, size=11.5)
    yy += 74

    f.t(x + 22, yy, "⚠️ 这里有一处口径打架 —— 如实讲", OR, True, 13,
        cls="svglbl")
    yy += 24
    for who, claim, col in [
        ("H2O 2023", "注意力矩阵 95% 以上稀疏，<tspan font-weight=\"700\">5% 的 KV 就够</tspan>", GR),
        ("Chen 等 2024（NSA 引）", "<tspan font-weight=\"700\">top 20% 只覆盖 70%</tspan> 的注意力分数", RD),
    ]:
        f.box(x + 22, yy, pw - 44, 52, "#fff", col, 8)
        f.t(x + 38, yy + 22, who, col, True, 12)
        f.t(x + 38, yy + 41, claim, GY, size=11.5, w=pw - 76)
        yy += 60

    f.t(x + 22, yy + 4, "⭐ 方向一致（确实稀疏），程度打架。", INK, True, 12.5)
    f.t(x + 22, yy + 25, "⛔ 别把任何一个当普适常数 —— 稀疏度随", GY,
        size=11.5)
    f.t(x + 22, yy + 44, "模型、层、上下文长度变。", GY, size=11.5)
    yy += 62

    f.box(x + 22, yy, pw - 44, 72, "#fff", LINE, 8)
    f.t(x + 38, yy + 23, "NSA 的实证：27B backbone，260B token 预训练",
        GY, size=11.5)
    f.t(x + 38, yy + 44, "通用 / 长文 / 推理三类上<tspan font-weight=\"700\">持平或超过全注意力</tspan>",
        GR, True, 12)
    f.t(x + 38, yy + 63, "64K 长度下，解码、前向、反向三段都更快", GY2,
        size=11)
    fits(yy + 72, y0, ph, "③ native")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 这一张最值钱的一句：计算稀疏 ≠ 访存稀疏", [
        "坑③和坑④是<tspan font-weight=\"700\">同一件事的两个面</tspan>："
        "你在纸上少算了 90% 的格子，但只要那些格子<tspan font-weight=\"700\">"
        "散落在显存各处</tspan>，或者<tspan font-weight=\"700\">同组的头各挑各的</tspan>，"
        "要搬的字节一点没少。",
        "⭐ 所以 NSA 的选择粒度是<tspan font-weight=\"700\">块</tspan>、"
        "选择范围是<tspan font-weight=\"700\">组内共享</tspan> ——&#160;"
        "这两个设计<tspan font-weight=\"700\">都不是为了精度，是为了访存</tspan>。",
        "⛔ 看任何一篇讲稀疏的文章，先问一句："
        "<tspan font-weight=\"700\">它省的是 FLOPs 还是字节？</tspan>"
        "省 FLOPs 很容易，省字节才算数。",
    ])

    yy = f.band(yy + 14, "ok", "暗线第四次出现 —— 这次连论文标题都挑明了", [
        "NSA 的 N 就是 <tspan font-weight=\"700\">Natively trainable</tspan>。"
        "同一个对立在这一讲已经出现四次："
        "Eigen Attention 对 MLA、GQA 对 MLA、H2O 对 DSA、"
        "<tspan font-weight=\"700\">这一堆事后方法对 NSA</tspan>。",
        "⭐ 到这里可以把它当结论讲了："
        "<tspan font-weight=\"700\">凡是要改注意力形状的改动，事后做都便宜，"
        "但天花板明显更低；要拿到上限，就得把约束写进训练。</tspan>",
    ])

    yy = f.src(yy + 16,
               "四个坑与三条路均出自 NSA 原论文 Yuan 等 arXiv 2502.11089 "
               "§2.1–2.2 与 §3.2–3.3（ACL 2025）；例子里的方法名也是原文点的",
               "「top 20% 只覆盖 70%」是 NSA 转引 Chen 等 2024；"
               "「95% 稀疏」出自 H2O arXiv 2306.14048 ——&#160;两者口径不同，图中如实并列")
    f.save("fig3-nsa-why.svg", yy + 6)


main()
