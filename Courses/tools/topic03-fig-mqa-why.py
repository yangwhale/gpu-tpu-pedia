# -*- coding: utf-8 -*-
r"""专题三 · §5.x「砍头这一支是怎么想出来的」

⭐⭐⭐ 2026-09-14 **整张重画**，换成一个办公室里的画面：**查通讯录。**

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
    PH = 360
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
        % (H1 - MQA), INK, True, 21)

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
    f.t(80, by + 208, "发明点二：⭐ 真正的动机是「模型越大，头越多」", BL, True, 21)
    f.t(80, by + 244, "全组共用一本，在小模型上还行；模型一大，"
        "<tspan font-weight=\"700\">削减力度就失控了</tspan> ——&#160;"
        "分组让「几个人共用一本」这个比例，跟着模型规模走。", GY, size=18)

    # ══════════ ③ 2019 年那张单子 ══════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 262
    py3 = f.panel(0, y2, W, PH3, "③ 顺带一件事：今天三个旋钮里的两个，2019 年那张单子上就有",
                  PU, sub="MQA 论文 §2.4，原文列的")

    ey = py3 + 22
    f.t(56, ey + 24, "Shazeer 先摆了一张单子：想让解码每步少搬点东西，可以 ——",
        GY, size=18)
    OPTS = [
        (0, "① 限制序列长度", "就不让它变长", "—"),
        (0, "② 减少「被看到」的位置数", "而这一条又分两种：", ""),
        (1, "· 只看一个局部邻域", "后来叫滑窗", "旋钮②"),
        (1, "· 压缩历史位置的个数", "后来叫稀疏 / 压缩", "旋钮②"),
    ]
    oy = ey + 40
    for lvl, name, later, knob in OPTS:
        ix = 56 + lvl * 28
        f.box(ix, oy, 700 - lvl * 28, 40, "#fff", LINE, 6)
        f.t(ix + 16, oy + 26, name, GY, True, 17)
        f.t(ix + 260, oy + 26, later, GY2, size=15)
        if knob and knob != "—":
            f.t(740, oy + 26, knob, PU, True, 17)
        oy += 46

    f.box(800, ey + 40, 560, 184, "#f3e8fd", PU, 10)
    f.t(824, ey + 80, "然后他说：本文走一条<tspan font-weight=\"700\">正交</tspan>的路",
        PU, True, 20)
    f.t(824, ey + 116, "——　去掉通讯录的「份数」这一维", PU, True, 22)
    f.t(824, ey + 148, "查的人一个不动", GY, size=17)
    f.t(824, ey + 190, "⭐⭐ 这门课的骨架不是我们事后归纳的，", PU, True, 19)
    f.t(824, ey + 218, "是当事人自己列的。", PU, True, 19)

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
