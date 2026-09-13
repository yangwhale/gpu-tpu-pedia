# -*- coding: utf-8 -*-
r"""专题三 · §6.3b「NSA 的三条路是被什么逼出来的」

⭐⭐⭐ 2026-09-14 **整张重画**，换成一个人人都用过的画面：
   **读一本很厚的书，你会怎么读？**

   ① **三条路 ＝ 读厚书的三种办法**（画出来，不用解释）
      · **翻目录** ——&nbsp;每章压成一行，先粗看全书 →&nbsp;压缩分支
      · **挑几章精读** ——&nbsp;⭐ 注意是**整章整章地挑**，不是东一句西一句 →&nbsp;选择分支
      · **手边这几页** ——&nbsp;刚读过的上下文 →&nbsp;滑窗分支
      三条同时用，**用一个学出来的门决定各占多少**。
   ② **为什么非得整章整章地挑** ——&nbsp;这是全图最该记住的一格：
      东一句西一句地抽，**书要来回翻**（访存不连续）；整章拿，**一次就搬走**。
      ⭐⭐ 而且 GQA 下更狠：一组里每个头各挑各的，
      真正要搬的是**所有头挑的并集** ——&nbsp;**算是省了，搬没省。**
   ③ **四个坑** ——&nbsp;出自 NSA 论文 §2，逐条点了名。
      ⚠️ 「三条分支一条对一个坑」是**本课的读法**，论文没做这个映射。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    f = Fig(W, "NSA 的三条路就是读一本厚书的三种办法：翻目录粗看、整章整章地挑着精读、"
               "看手边这几页；为什么非得整章整章地挑，因为东一句西一句要来回翻书，"
               "而且 GQA 下真正要搬的是同组各头选择的并集")
    f.marks = set()
    y0 = f.header(
        "NSA 的三条路 ——　其实就是你读一本厚书的三种办法",
        "三条<tspan font-weight=\"700\">同时用</tspan>，"
        "用一个<tspan font-weight=\"700\">学出来的门</tspan>决定各占多少",
        [(BL, "翻目录"), (GR, "挑章精读"), (OR, "手边这几页"), (RD, "前人踩的坑")])

    # ══════════ ① 读厚书的三种办法 ══════════════════════════════
    PH = 308
    py = f.panel(0, y0, W, PH, "① 一本很厚的书摆在你面前 ——　你会怎么读",
                 GR, sub="这就是那三条分支")

    ay = py + 22
    WAYS = [
        (BL, "翻目录", "每章压成一行", "全书都扫到了，但粗",
         "＝ 压缩分支"),
        (GR, "挑几章精读", "⭐ 整章整章地挑", "挑中的看得很细",
         "＝ 选择分支"),
        (OR, "手边这几页", "刚翻过的上下文", "最近的一定看",
         "＝ 滑窗分支"),
    ]
    for i, (col, name, how, what, who) in enumerate(WAYS):
        bx = 56 + i * 342
        f.box(bx, ay + 20, 310, 196, "#fff", col, 10)
        f.t(bx + 24, ay + 60, name, col, True, 26)
        # 每格配一个小画面：一本书的 12 章
        for c in range(12):
            cx = bx + 26 + (c % 6) * 46
            cyy = ay + 84 + (c // 6) * 40
            if i == 0:                      # 翻目录：每章都碰一下，但很薄
                f.box(cx, cyy + 12, 40, 12, col, "none", 2)
            elif i == 1:                    # 挑章精读：整章亮，只亮两章
                on = c in (2, 3, 9)
                f.box(cx, cyy, 40, 32, col if on else BG2,
                      "none" if on else LINE2, 3)
            else:                           # 手边这几页：只亮最后几章
                on = c >= 9
                f.box(cx, cyy, 40, 32, col if on else BG2,
                      "none" if on else LINE2, 3)
        f.t(bx + 24, ay + 186, how, col, True, 18)
        f.t(bx + 24, ay + 210, what, GY2, size=15)
        f.t(bx + 24, ay + 240, who, GY, True, 17)

    f.box(1084, ay + 20, 276, 196, "#f3e8fd", PU, 10)
    f.t(1108, ay + 60, "三条同时用", PU, True, 24)
    f.t(1108, ay + 100, "一个学出来的门", GY, size=17)
    f.t(1108, ay + 130, "决定这一步", GY, size=17)
    f.t(1108, ay + 160, "各占多少", GY, size=17)
    f.t(1108, ay + 198, "⭐ 不是三选一", PU, True, 19)

    # ══════════ ② 为什么非得整章整章地挑 ════════════════════════
    y1 = y0 + PH + 18
    PH2 = 342
    py2 = f.panel(0, y1, W, PH2,
                  "② 全图最该记住的一格：为什么非得「整章整章」地挑",
                  RD, sub="东一句西一句，书要来回翻")

    by = py2 + 20
    f.t(56, by + 20, "东一句西一句地抽", RD, True, 22)
    for c in range(24):
        on = c in (1, 5, 6, 12, 17, 21)
        f.box(56 + c * 24, by + 36, 18, 52, RD if on else BG2,
              "none" if on else LINE2, 2)
    f.t(56, by + 116, "⛔ 要翻 6 次书 ——　每次只拿一句", RD, True, 19)
    f.t(56, by + 144, "落到硬件上就是：访存不连续，FlashAttention 用不上", GY,
        size=17)

    f.t(720, by + 20, "整章整章地拿", GR, True, 22)
    for c in range(24):
        on = 4 <= c < 10 or 16 <= c < 22
        f.box(720 + c * 24, by + 36, 18, 52, GR if on else BG2,
              "none" if on else LINE2, 2)
    f.t(720, by + 116, "✅ 只翻 2 次 ——　每次整块搬走", GR, True, 19)
    f.t(720, by + 144, "同样多的内容，搬运次数差好几倍", GY, size=17)

    gy_ = by + 178
    f.box(56, gy_, 1304, 112, "#fce8e6", RD, 10)
    f.t(80, gy_ + 40, "⭐⭐ 而在 GQA 上，这件事更狠", RD, True, 22)
    f.t(80, gy_ + 76, "一组里<tspan font-weight=\"700\">每个头各挑各的</tspan>，"
        "可它们共用同一份 KV ——&#160;真正要搬的是"
        "<tspan font-weight=\"700\">所有头挑中的并集</tspan>。", GY, size=18)
    f.t(80, gy_ + 104, "→　<tspan font-weight=\"700\">算是省了，搬没省。</tspan>"
        "这正是 NSA 点名 Quest 的那一条。", RD, True, 18)

    # ══════════ ③ 四个坑 ════════════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 244
    py3 = f.panel(0, y2, W, PH3, "③ 这些讲究是从哪来的 ——　前人踩过的四个坑",
                  OR, sub="NSA 论文 §2 逐条点了名")

    ey = py3 + 18
    PITS = [
        ("省了算，没省时间", ["解码时稀疏，可 prefill", "还得先把注意力图算出来"],
         "H2O 这一类"),
        ("挑的动作不可导", ["k-means、SimHash 这种挑法", "学不到「该怎么挑」"],
         "ClusterKV / MagicPIG"),
        ("按 token 挑 → 来回翻书", ["散落各处的 token，", "FlashAttention 用不上"],
         "HashAttention"),
        ("GQA 上要搬并集", ["每个头各挑各的，", "真正搬的是它们的并集"], "Quest"),
    ]
    for i, (t1, t2, who) in enumerate(PITS):
        bx = 56 + i * 332
        f.box(bx, ey + 20, 300, 172, "#fff", OR, 10)
        f.t(bx + 20, ey + 56, "坑 %d" % (i + 1), GY2, True, 15)
        f.t(bx + 20, ey + 86, t1, OR, True, 19, w=264)
        for k, ln in enumerate(t2):
            f.t(bx + 20, ey + 118 + k * 24, ln, GY, size=15, w=264)
        f.t(bx + 20, ey + 180, "例：" + who, GY2, size=14)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐ 这一张真正的教益：稀疏不是「少算」，是「少搬」", [
        "四个坑里有<tspan font-weight=\"700\">三个都跟「搬」有关</tspan> ——&#160;"
        "省了计算没省时间、访存不连续、搬的是并集。"
        "<tspan font-weight=\"700\">只有「挑的动作不可导」那条是算法问题。</tspan>",
        "⭐ 所以看一篇讲稀疏的文章，先问一句："
        "<tspan font-weight=\"700\">它省的是 FLOPs，还是字节？</tspan>"
        "省 FLOPs 谁都会 ——&#160;在纸上少算 90% 的格子而已；"
        "可只要那些格子散落在显存各处，<tspan font-weight=\"700\">要搬的字节一点没少</tspan>。",
    ])

    yy = f.band(yy + 14, "warn", "两条口径", [
        "⚠️ <tspan font-weight=\"700\">「四个坑」是论文 §2 自己列的</tspan>"
        "（逐条点了名，连例子都是原文的）；"
        "⭐ 但<tspan font-weight=\"700\">「三条分支一条对一个坑」是本课的读法</tspan> ——&#160;"
        "论文没有做这个一一映射。",
        "⚠️ 稀疏到底能稀疏到什么程度，"
        "<tspan font-weight=\"700\">公开口径本身就打架</tspan>：H2O 说「95% 稀疏、5% 够用」，"
        "NSA 引的 Chen 等 2024 说「前 20% 只覆盖 70% 的注意力分数」。"
        "⭐ 方向一致、程度差一倍多 ——&#160;<tspan font-weight=\"700\">别把任何一个当普适常数。</tspan>",
    ])

    yy = f.band(yy + 14, "ok", "暗线第四次出现：事后压 vs 从头按压缩训", [
        "前面三次是 Eigen Attention 对 MLA、GQA 对 MLA、H2O 对 DSA。"
        "<tspan font-weight=\"700\">这是第四次，而且它就写在名字里</tspan> ——&#160;"
        "NSA 的 N 就是 <tspan font-weight=\"700\">Natively trainable</tspan>。",
        "⭐ 事后稀疏是「拿一个按『每个都看』训出来的模型，临时叫它少看」——&#160;"
        "它<tspan font-weight=\"700\">偏离了自己的预训练轨迹</tspan>；"
        "native 是<tspan font-weight=\"700\">从第一天就按「我会少看」训</tspan>。"
        "<tspan font-weight=\"700\">四个分支，同一条暗线。</tspan>",
    ])

    yy = f.src(yy + 16,
               "四个坑与三条分支出自 NSA（Yuan 等 arXiv 2502.11089）§2 与 §3；"
               "27B backbone / 260B token 亦出自该文",
               "⚠️ 「读厚书 / 翻目录 / 整章拿」是<tspan font-weight=\"700\">本课的比喻</tspan>；"
               "论文那侧的说法是 compression / selection / sliding window 三分支"
               "加一个 learned gate")
    f.save("fig3-nsa-why.svg", yy + 6)


main()
