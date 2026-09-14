# -*- coding: utf-8 -*-
r"""专题三 · §5.x「砍头这一支是怎么想出来的」

⭐⭐⭐ 2026-09-13 **整张重画**，换成一个办公室里的画面：**查通讯录。**

  · **一个 query 头 ＝ 一个要查东西的人**（他有自己关心的角度）
  · **一份 K/V ＝ 一本通讯录**（要占抽屉，要搬来搬去）

  ① **MHA ＝ 八个人，各带一本** ——&nbsp;抽屉塞满了。
  ② **MQA ＝ 八个人，共用一本** ——&nbsp;抽屉空了，
     ⭐ 而且**八个人还在，只是共用一本**。
  ③ **真单头 ＝ 只派一个人去查** ——&nbsp;抽屉一样空，
     ⛔ **但效果明显更差**（困惑度 31.2 vs 30.2）。

  ⭐⭐ 于是那句判据就不用解释了：
  **贵的是通讯录（K/V），不是查的人（query 头）。
  砍通讯录，别砍人。**

📌 数字取自 MQA 原论文 Shazeer arXiv 1911.02150 **表 3**，脚本里断言。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    MHA, MQA, H1 = 29.9, 30.2, 31.2
    SMALL = [("h=2, d_k=64", 31.1), ("h=4, d_k=32", 31.0), ("h=8, d_k=16", 30.9)]
    assert H1 > MQA and abs((H1 - MQA) - 1.0) < 1e-9

    f = Fig(W, "砍头这一支怎么想出来的：一个 query 头就是一个要查东西的人，"
               "一份 KV 就是一本通讯录；八个人各带一本、八个人共用一本、"
               "只派一个人去查 —— 后两种抽屉一样空，但只派一个人明显更差")
    f.marks = set()
    y0 = f.header(
        "砍头这一支 ——　贵的是通讯录，不是查的人",
        "<tspan font-weight=\"700\">一个 query 头</tspan> ＝ 一个要查东西的人；"
        "<tspan font-weight=\"700\">一份 K/V</tspan> ＝ 一本占抽屉的通讯录",
        [(GY, "各带一本"), (GR, "共用一本"), (RD, "只派一个人")])

    # ══════════ ① 三种办法 ══════════════════════════════════════
    PH = 380
    py = f.panel(0, y0, W, PH, "① 三种办法 ——　后两种占的抽屉一样多",
                 GR, sub="同一篇论文的消融表")

    ay = py + 22
    CASES = [
        (GY, "① 八个人，各带一本", 8, 8, MHA, "抽屉塞满", "MHA"),
        (GR, "② 八个人，共用一本", 8, 1, MQA, "抽屉空了", "MQA"),
        (RD, "③ 只派一个人去查", 1, 1, H1, "抽屉一样空", "真单头 h=1"),
    ]
    for i, (col, title, nq, nkv, ppl, drawer, tag) in enumerate(CASES):
        bx = 56 + i * 442
        f.box(bx, ay + 24, 400, 268, "#fff", col, 10)
        f.t(bx + 22, ay + 62, title, col, True, 22)
        # 人
        for k in range(nq):
            f.box(bx + 22 + (k % 8) * 46, ay + 80, 36, 40, "#f1f3f4", GY2, 5)
            f.t(bx + 40 + (k % 8) * 46, ay + 106, "人", GY, True, 15, "middle")
        # 通讯录
        f.t(bx + 22, ay + 152, "通讯录：", GY2, size=15)
        for k in range(nkv):
            f.box(bx + 100 + k * 38, ay + 132, 30, 38, col, "none", 4)
        f.t(bx + 22, ay + 196, drawer, col, True, 19)
        f.t(bx + 22, ay + 232, "困惑度", GY2, size=15)
        f.t(bx + 100, ay + 238, "%.1f" % ppl, col, True, 34)
        f.t(bx + 22, ay + 272, tag, GY2, size=14)

    f.t(56, ay + 318, "⭐⭐ 后两种<tspan font-weight=\"700\">占的抽屉一模一样</tspan>"
        "（都只有一本），差别只在「还有几个人在查」——&#160;"
        "就这一点，困惑度差了整整 <tspan font-weight=\"700\">%.1f</tspan>。"
        % (H1 - MQA), INK, True, 17)

    # ══════════ ② 那 GQA 是怎么回事 ═════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 296
    py2 = f.panel(0, y1, W, PH2, "② GQA ——　分组，每组一本", BL,
                  sub="它的两个发明点，都不是「折中一下」")

    by = py2 + 22
    for g in range(4):
        gx = 56 + g * 178
        f.box(gx, by + 26, 160, 120, "#e8f0fe", BL, 8)
        f.t(gx + 80, by + 54, "第 %d 组" % (g + 1), BL, True, 17, "middle")
        for k in range(2):
            f.box(gx + 22 + k * 44, by + 66, 34, 34, "#fff", GY2, 5)
            f.t(gx + 39 + k * 44, by + 90, "人", GY, True, 14, "middle")
        f.box(gx + 112, by + 66, 26, 34, BL, "none", 4)
        f.t(gx + 80, by + 130, "共用一本", GY2, size=14, anchor="middle")

    f.box(788, by + 26, 572, 120, "#fff", GR, 10)
    f.t(812, by + 62, "发明点一：不用从头重训", GR, True, 21)
    f.t(812, by + 96, "把已有的那几本<tspan font-weight=\"700\">平均</tspan>成一本，", GY,
        size=17)
    f.t(812, by + 128, "再用 <tspan font-weight=\"700\">5%</tspan> 的原始预训练算力续一下", GY,
        size=17)

    f.box(56, by + 170, 1304, 100, "#e8f0fe", BL, 10)
    f.t(80, by + 208, "发明点二：⭐ 真正的动机是「模型越大，头越多」", BL, True, 17)
    f.t(80, by + 244, "全组共用一本，在小模型上还行；模型一大，"
        "<tspan font-weight=\"700\">削减力度就失控了</tspan> ——&#160;"
        "分组让「几个人共用一本」这个比例，跟着模型规模走。", GY, size=17)

    # ══════════ ③ 2019 年那张单子 ══════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 386
    py3 = f.panel(0, y2, W, PH3, "③ 顺带一件事：今天三个旋钮里的两个，2019 年那篇论文里就已经是垂直的两根轴",
                  PU, sub="MQA 论文 §2.4，原文列的")

    # ⭐⭐⭐ 2026-09-14 R49b/R50 重画。原来这一格是**一张竖排列表 ＋ 旁边一个框**。
    #
    # ⛔ 病灶只有一句话：**这一格的核心词是「正交」，而图上没有任何东西是正交的。**
    #   「本文走一条正交的路」印在右边那个框里，读者只能把它当一句修辞收下。
    #   （同型判据，可以拿去查别的图：**正文里出现一个几何词 ——&#160;正交 / 一起 /
    #     两端 / 夹在中间 ——&#160;图上就必须有对应的几何，否则是用文字冒充结构。**
    #     R38b 那条是「一起」，这条是「正交」。）
    #
    # ⭐ 而这张图本来就握着最好的素材：单子上每一条落的都是**旋钮②**，
    #   MQA 走的是**旋钮①**。两个旋钮在本课里本来就是两节（§五／§六）。
    #   所以直接画成**两根垂直的轴**：单子全在横轴上，MQA 拐上纵轴。
    #   「正交」这个词从此有了图形对应物 ——&#160;而且是原文自己的用词，不是我安的。
    ey = py3 + 22
    OX, OYA = 380, ey + 236          # 原点
    AXL, AXU = 470, 152              # 横轴长 / 纵轴高

    f.t(40, ey + 20, "Shazeer 先摆了一张单子：想让解码每步少搬点东西，可以 ——",
        GY, True, 17)

    # ── 两根轴。⛔ 粗细不是装饰：横轴是**单子上那些条**落的地方，
    #   纵轴是**这篇论文自己走的那条**，所以纵轴加粗。
    f.line(OX, OYA, OX + AXL, OYA, PU, 2.0)
    f.line(OX, OYA, OX, OYA - AXU, PU, 3.4)
    f.t(OX + AXL, OYA + 26, "旋钮②　每个 query 看多少（§六）",
        PU, True, 16, anchor="end")
    f.t(OX - 6, OYA - AXU - 24, "旋钮①　每个 token 存多少（§五）", PU, True, 16)
    f.t(OX - 6, OYA + 26, "MHA", GY, True, 15, anchor="middle")
    f.t(OX - 6, OYA + 44, "各带一本、全都看", GY2, size=13, anchor="middle")

    # ⭐⭐ 直角记号 ——&#160;全世界通用的那个小方角。「正交」两个字就贴在它旁边，
    #   这样它指的是**这个角**，而不是右边框里那句话。
    RA = 20
    f.line(OX + RA, OYA, OX + RA, OYA - RA, PU, 1.4, arrow=False)
    f.line(OX, OYA - RA, OX + RA, OYA - RA, PU, 1.4, arrow=False)
    f.t(OX + RA + 9, OYA - 6, "正交", PU, True, 14)

    # ── 横轴上：单子里那两条，后来各得了一个名字 ───────────────
    # ⛔ fr 不能再往左：纵轴那两条的说明文字是**向右伸**的，横轴这两条的说明文字
    #   是**向上伸**的，两者会在左上角撞成一团（第一版 0.42 就撞了）。
    #   两根轴共用一个左上象限，这是所有二维散点教学图的固定病。
    XS = [(0.55, "只看一个局部邻域", "后来叫滑窗"),
          (0.92, "压缩历史位置的个数", "后来叫稀疏 / 压缩")]
    for fr, name, later in XS:
        x = OX + AXL * fr
        f.box(x - 7, OYA - 7, 14, 14, PU, PU, 7)
        f.line(x, OYA - 14, x, OYA - 44, GY2, 1.2, dash="4 3", arrow=False)
        f.t(x, OYA - 52, name, GY, True, 15, anchor="middle")
        f.t(x, OYA - 34, later, GY2, size=13, anchor="middle")

    # ── 纵轴上：这篇论文自己走的那条 ───────────────────────────
    YS = [(0.46, "八个人共用一本", "GQA（2023）"),
          (0.92, "只派一个人去查", "MQA ——　本文")]
    for fr, name, later in YS:
        y = OYA - AXU * fr
        f.box(OX - 7, y - 7, 14, 14, PU, PU, 7)
        f.line(OX + 14, y, OX + 54, y, GY2, 1.2, dash="4 3", arrow=False)
        f.t(OX + 62, y - 2, name, GY, True, 15)
        f.t(OX + 62, y + 16, later, GY2, size=13)

    # ── 左边那条**连轴都不是**的：故意画在平面外面，灰的 ─────────
    f.box(40, ey + 112, 288, 96, "#fff", LINE, 8, dash="4 4")
    f.t(58, ey + 142, "还有一条：限制序列长度", GY2, True, 16)
    f.t(58, ey + 168, "——　干脆不让 S 变长。", GY2, size=14)
    f.t(58, ey + 192, "⚠️ 它连轴都不是，所以在平面外。", GY2, size=14)

    f.box(900, ey + 26, 470, 186, "#f3e8fd", PU, 10)
    f.t(924, ey + 62, "他说：本文走一条<tspan font-weight=\"700\">正交</tspan>的路",
        PU, True, 20)
    f.t(924, ey + 94, "——　去掉通讯录的「份数」这一维", PU, True, 19)
    f.t(924, ey + 120, "⭐ 而「看多少个位置」一个没动 ——　这就是正交。", GY, size=15)
    f.t(924, ey + 158, "⭐⭐ 这门课的骨架不是我们事后归纳的，", PU, True, 17)
    f.t(924, ey + 184, "是当事人自己列的 ——　连「正交」都是原话。", PU, True, 17)

    f.t(40, OYA + 74,
        "⭐ 顺带交代第三个旋钮：<tspan font-weight=\"700\">它不在这张平面上</tspan> ——&#160;"
        "换机制（§七）不是在这两根轴上挪，是<tspan font-weight=\"700\">换掉整个坐标系</tspan>。"
        "所以 2019 年那张单子上没有它，也不该有。", GY2, size=15)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 带走一句：贵的是通讯录，不是查的人", [
        "<tspan font-weight=\"700\">砍通讯录（K/V）省的是真金白银</tspan>；"
        "<tspan font-weight=\"700\">砍查的人（query 头）省不了多少，却很伤</tspan>。"
        "同样一本通讯录，八个人查和一个人查，困惑度差 %.1f。" % (H1 - MQA),
        "⭐ 这也正好解释了后面那件反直觉的事："
        "<tspan font-weight=\"700\">MQA 存得比 MLA 还少，效果却更差</tspan> ——&#160;"
        "它砍到的不只是通讯录，还砍掉了「从几个角度去查」这件事。",
    ])

    yy = f.band(yy + 14, "warn", "口径：那张表还有一组对照，别漏", [
        "同一张表里还有一组：把总维度摊给更多头（缓存都只有一本）——&#160;"
        + "、".join("%s → %.1f" % (a, b) for a, b in SMALL) + "。"
        "⭐ 都比真单头好，但<tspan font-weight=\"700\">都不如 MQA 的 %.1f</tspan>。"
        % MQA,
        "⚠️ 表 3 里各行的 <tspan font-weight=\"700\">d_ff 并不相同</tspan>"
        "（8192 / 9088 / 9984）——&#160;那是为了<tspan font-weight=\"700\">对齐总参数量</tspan>；"
        "看困惑度差之前先知道这一点。",
    ])

    yy = f.band(yy + 14, "ok", "暗线第二次出现：事后压 vs 从头按压缩训", [
        "GQA 是<tspan font-weight=\"700\">事后</tspan>的典范 ——&#160;"
        "把已有的那几本通讯录<tspan font-weight=\"700\">平均</tspan>成一本，"
        "再用 5% 的原始预训练算力续一下就能用，<tspan font-weight=\"700\">不用从头重训</tspan>。",
        "MLA 是<tspan font-weight=\"700\">从头</tspan>的典范 ——&#160;"
        "要重训，但换来的是 56.9× 而不是 16×。"
        "⭐ <tspan font-weight=\"700\">这两条路一直并存到今天</tspan>，§六 还会再遇见两次。",
    ])

    yy = f.src(yy + 16,
               "①② 出自 MQA 原论文 Shazeer arXiv 1911.02150（§2.4 与表 3 ——&#160;"
               "⚠️ 这篇一共只有 3 张表，Billion-Word LM 基准的 dev 困惑度）；"
               "「正交」是原文用词",
               "③ 出自 GQA 原论文 Ainslie 等 arXiv 2305.13245 §2.1–2.2："
               "mean pooling、α=5% 续训、GQA-1=MQA / GQA-H=MHA",
               "⚠️ 「通讯录 / 查的人」是<tspan font-weight=\"700\">本课的比喻</tspan>")
    f.save("fig3-mqa-why.svg", yy + 6)


main()
