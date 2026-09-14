# -*- coding: utf-8 -*-
r"""专题三 · §1.3b「为什么会记混」—— 一个 128 维的脑袋装得下多少件互不相干的事

⭐⭐⭐ 2026-09-13 夜间 R14 新画。派出去的六个调研 agent 一致指出：
   **这是全领域的视觉空白。** 每一篇讲注意力的文章都会写「每个头 128 维」，
   没有一篇回答读者立刻会想的那个问题 ——
   「128 个数，凭什么记得住上万个 token 的区别？」

⭐ 答案分两层，而**第二层才是真正有用的那层**：
   ① 要求「完全不撞」（两两正交）—— 128 维里最多 128 根，这是秩论证，铁的。
   ② 放宽成「差不多不撞」—— 容量按指数涨（Johnson–Lindenstrauss 引理）。
      本课当场算：128 维里随机丢 1 万根，最挤的那两根仍有 60.5°。
   ⭐⭐ 于是「记混」不是玄学，是**第二层的定价**：每两个之间都有一点点像。

⛔ 而这一点点像，单看可以忽略 —— cos=0.2 时一个干扰项只有正主的万分之一点二。
   **但「可以忽略」不能乘以一万。** 一万个加起来 = 正主的 1.17 倍，
   正主只剩 46%。⭐ 这就是长上下文「越读越糊」的机制本身。

📌 这张图串起本讲三处，都是学生问过而当时只能糊弄过去的：
   · 为什么每个头是 128 维不是 8 维（8 维里丢 1000 根，最挤的一对几乎重合）
   · 为什么 MLA 压到 512 维不死 —— 它要的从来不是「完全正交」，是「差不多」
   · 为什么上下文一长就记混 —— N 变大是**两头夹击**：干扰更多，而且最挤的更挤

⚠️ 归属要诚实：
   · 「差不多正交能装指数多个」是 Johnson–Lindenstrauss 引理的推论，不是本课的
   · 「叠加（superposition）」这个说法与它的机制解释出自 Anthropic
     Elhage 等《Toy Models of Superposition》（arXiv 2209.10652，2022）
   · ⭐ 但图里所有数字都是**本课当场算的**（numpy，脚本里断言），不是抄来的
"""
import math

import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2, wpx)

W = 1400
D = 128                      # 一个头的维度 —— 本讲主线用的就是这个数


def max_abscos(n, d, seed):
    """在 d 维里随机丢 n 根单位杆，返回最挤的那一对的 |cos|。

    ⛔ 不要直接算 n×n 的 Gram —— n=10000 时那是 4 亿个数。分块。
    """
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, d)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    best = 0.0
    for i in range(0, n, 2000):
        G = np.abs(V[i:i + 2000] @ V.T)
        for r in range(G.shape[0]):
            G[r, i + r] = 0.0                 # 自己跟自己不算
        best = max(best, float(G.max()))
    return best


def main():
    # ══ 先把所有数算出来，算不出来就别画 ═══════════════════════════
    # ① 严格正交的上限 = 维度数（秩论证，当场验一遍）
    rng = np.random.default_rng(11)
    assert np.linalg.matrix_rank(rng.standard_normal((D + 1, D))) == D

    # ② 放宽到「差不多」之后的实测夹角
    NS = (D, 1000, 10000)
    COS = [max_abscos(n, D, 7 + i) for i, n in enumerate(NS)]
    ANG = [math.degrees(math.acos(c)) for c in COS]
    # ⛔ 断言两件事，缺一条这张图就不成立：
    #   · 挤是挤了，但挤得**非常慢**（N 涨 78 倍，夹角掉不到 12°）
    assert ANG[0] - ANG[2] < 12, ANG
    #   · 而且确实是单调变挤的（不是噪声）
    assert ANG[0] > ANG[1] > ANG[2] > 55, ANG
    # ③ 对照：8 维里丢 1000 根会挤成什么样（解释「为什么不是 8 维」）
    C8 = max_abscos(1000, 8, 99)
    A8 = math.degrees(math.acos(C8))
    assert A8 < 20, A8                         # 几乎重合

    # ④ 串扰：q 正好命中 key_A，N 个干扰项与它 cos=c，logit = √d·cos
    #    ⚠️ 这个 √d 不是凑的 —— q、k 各分量 O(1) 时内积 ≈ d·cos，
    #      注意力再除以 √d，剩下的正好是 √d·cos。推导链就这一行。
    CC = 0.2
    one = math.exp(math.sqrt(D) * (CC - 1.0))
    share = [100.0 / (1.0 + n * one) for n in (100, 1000, 10000)]
    assert one < 2e-4 and share[0] > 98 and share[2] < 50, (one, share)

    f = Fig(W, "为什么会记混：128 维里能塞下多少个彼此不像的方向")
    yy = f.header(
        "为什么会记混 ——&#160;一个 128 维的脑袋，装得下多少件「互不相干」的事",
        "每篇文章都写「每个头 128 维」，然后就过去了。可 128 个数凭什么记得住"
        "上万个 token 的区别？这张图把它算出来：严格不撞只能装 128 个，"
        "放宽成「差不多不撞」就能装上万个 ——&#160;而「记混」正是这笔放宽的价钱。",
        legend=[(GR, "完全不撞（两两垂直）"), (BL, "差不多不撞"),
                (RD, "串扰＝记混")])

    # ══ ① 完全不撞的方向，一间屋子里只有这么多 ═════════════════════
    PH1 = 334
    top = f.panel(0, yy, W, PH1,
                  "① 先问上限：要求「谁跟谁都不沾边」，一间屋子里能立几根杆？",
                  GR, tag="秩论证 · 当场验")
    by = top + 214                             # 三个场景共用的地面高度

    def scene(ox, title, note, ncol):
        f.t(ox, top + 34, title, ncol, bold=True, size=17, cls="svglbl")
        f.t(ox, top + 58, note, GY, size=15)

    # 场景 A：一张桌面 —— 2 维
    scene(40, "一张桌面（2 维）", "横一根、竖一根，就满了", GR)
    f.box(30, by - 96, 300, 96, BG2, LINE2, 8)
    f.line(90, by - 16, 290, by - 16, GR, 2.6)
    f.line(90, by - 16, 90, by - 88, GR, 2.6)
    f.t(300, by + 4, "2 根", GR, bold=True, size=17, anchor="end")
    f.line(340, by - 96, 340, by, GY2, 1, dash="3 5", arrow=False)

    # 场景 B：房间的墙角 —— 3 维
    scene(400, "房间的墙角（3 维）", "再加一根「往里」，三根，也满了", GR)
    f.box(390, by - 96, 300, 96, BG2, LINE2, 8)
    f.line(470, by - 16, 640, by - 16, GR, 2.6)
    f.line(470, by - 16, 470, by - 88, GR, 2.6)
    f.line(470, by - 16, 408, by - 62, GR, 2.6)      # 斜着那根＝「往里」
    f.t(660, by + 4, "3 根", GR, bold=True, size=17, anchor="end")
    f.line(700, by - 96, 700, by, GY2, 1, dash="3 5", arrow=False)

    # 场景 C：128 维 —— 画不出来，但数学替我们数过了
    scene(760, "一个注意力头（128 维）", "画不出来 ——&#160;但这个上限不用画也知道",
          GR)
    f.box(750, by - 96, 620, 96, "none", LINE2, 8)
    for i in range(13):                        # 一把扇形的杆子，示意「很多根」
        a = math.radians(8 + i * 13)
        f.line(800, by - 20, 800 + 74 * math.cos(a), by - 20 - 74 * math.sin(a),
               GR, 1.8, arrow=False)
    f.t(906, by - 60, "…", GY2, size=22)
    f.t(940, by - 62, "最多 128 根。", INK, bold=True, size=17)
    f.t(940, by - 38,
        "第 129 根一定能被前面那些<tspan font-weight=\"700\">拼出来</tspan>"
        " ——&#160;它不是新东西。", GY, size=15)
    f.t(940, by - 14,
        "（脚本当场验：128 维里取 129 个向量，秩只有 128）", GY2, size=14)
    f.t(30, top + PH1 - 48,
        "⭐ 所以「互不相干的方向」这种<tspan font-weight=\"700\">奢侈品</tspan>，"
        "一个头只买得起 128 件。"
        "如果模型真按这个标准办事，它一辈子也只认得 128 个概念。", INK, size=16)

    # ══ ② 放宽一点点，容量就爆炸 ═══════════════════════════════════
    yy = top + PH1 + 26
    PH2 = 396
    top = f.panel(0, yy, W, PH2,
                  "② 那就别那么讲究：只要「差不多不撞」就行 ——&#160;容量立刻爆炸",
                  BL, tag="本课当场算 · numpy")
    f.t(30, top + 32,
        "把标准从「正好 90°」放宽到「大概 90° 就行」，能塞进去多少根？"
        "下面三个表盘是<tspan font-weight=\"700\">真的随机丢进 128 维再量出来的</tspan>"
        " ——&#160;量的是<tspan font-weight=\"700\">最挤的那一对</tspan>。", GY,
        size=16)

    for i, n in enumerate(NS):
        cx, cy = 250 + i * 330, top + 250
        R = 118
        f.box(cx - 150, top + 62, 300, 236, "none", LINE2, 9)
        # 90° 参照（虚线）＋ 水平基准杆
        f.line(cx, cy, cx, cy - R - 8, GY2, 1.2, dash="4 5", arrow=False)
        f.t(cx + 6, cy - R - 14, "90°（完全不撞）", GY2, size=14)
        f.line(cx, cy, cx + R, cy, BL, 2.6, arrow=False)
        a = math.radians(ANG[i])
        f.line(cx, cy, cx + R * math.cos(a), cy - R * math.sin(a), BL, 2.6,
               arrow=False)
        # 夹角弧
        f.p.append('<path d="M %.1f %.1f A 44 44 0 0 0 %.1f %.1f" fill="none" '
                   'stroke="%s" stroke-width="1.6"/>'
                   % (cx + 44, cy, cx + 44 * math.cos(a), cy - 44 * math.sin(a),
                      OR))
        f.t(cx + 58, cy - 26, "%.1f°" % ANG[i], OR, bold=True, size=19,
            cls="svglbl")
        f.t(cx, top + 88, "丢 %s 根进去" % ("{:,}".format(n)), INK, bold=True,
            size=17, anchor="middle", cls="svglbl")
        f.t(cx, cy + 34, "最挤的一对仍差 %.0f° 才重合" % ANG[i], GY, size=14,
            anchor="middle")

    # ⛔ 这两句原先挤在一行，右边整段被裁掉（SVG 文字溢出**不报错也没滚动条**）。
    #   ⭐ 而且原文顺手写了个「不到 12°」，跟下一句 8 维那个 12° 撞成同一个数 ——
    #     两个毫不相干的量印成同一个数字，是比溢出更坏的错。现在两个都当场算。
    f.t(30, top + PH2 - 78,
        "⭐ 读这三个表盘只读一件事：<tspan font-weight=\"700\">"
        "根数涨了 %.0f 倍（128 → 1 万），最挤的那一对才挤了 %.0f°。</tspan>"
        "　容量几乎是白捡的。" % (NS[2] / float(NS[0]), ANG[0] - ANG[2]),
        INK, size=16)
    f.t(30, top + PH2 - 52,
        "⛔ 反过来，<tspan font-weight=\"700\">维度一小就立刻崩</tspan>："
        "同样丢 1000 根，8 维里最挤的一对只剩 %.0f° ——&#160;那两根基本就是同一根，"
        "模型分不出它们。<tspan font-weight=\"700\">这就是「为什么不用 8 维」。"
        "</tspan>" % A8, RD, size=16)

    # ══ ③ 这笔放宽的价钱：一万个「可以忽略」 ════════════════════════
    yy = top + PH2 + 26
    PH3 = 366
    top = f.panel(0, yy, W, PH3,
                  "③ 便宜没白占：「差不多」就是「有点像」，"
                  "而有点像会漏票 ——&#160;这就是记混", RD,
                  tag="cos=0.2 · logit=√d·cos")

    # 左半：单个干扰小到画不出来
    f.t(40, top + 34, "先看一个干扰项有多小", INK, bold=True, size=17,
        cls="svglbl")
    bx, bb = 70, top + 228
    f.box(bx, bb - 168, 76, 168, "#e6f4ea", GR, 6)
    f.t(bx + 38, bb - 178, "正主", GR, bold=True, size=16, anchor="middle")
    f.line(bx, bb + 1, bx + 350, bb + 1, LINE, 1.2, arrow=False)
    f.box(bx + 150, bb - 3, 76, 3, RD, RD, 1)
    f.t(bx + 188, bb - 12, "一个干扰项", RD, bold=True, size=16,
        anchor="middle")
    f.t(bx + 188, bb + 24, "细到画不出来", RD, size=15, anchor="middle")
    f.t(bx + 188, bb + 46, "＝ 正主的 %.4f%%" % (one * 100), GY, size=15,
        anchor="middle")
    f.t(40, top + PH3 - 74,
        "⭐ 单看它，<tspan font-weight=\"700\">完全可以忽略。</tspan>", INK,
        size=16)
    f.t(40, top + PH3 - 48,
        "⛔ 但「可以忽略」<tspan font-weight=\"700\">不能乘以一万。</tspan>", RD,
        size=16)

    # 右半：数量一上来，正主就被淹了
    f.line(470, top + 34, 470, top + PH3 - 24, LINE2, 1.2, arrow=False)
    f.t(510, top + 34, "再看数量一上来会怎么样", INK, bold=True, size=17,
        cls="svglbl")
    BW = 560
    for i, n in enumerate((100, 1000, 10000)):
        ry = top + 76 + i * 74
        f.t(510, ry + 26, "%s 个干扰" % "{:,}".format(n), GY, bold=True,
            size=16)
        f.box(640, ry, BW, 38, "none", LINE, 6)
        g = BW * share[i] / 100.0
        f.box(640, ry, g, 38, "#e6f4ea", GR, 6)
        col = GR if share[i] > 80 else (OR if share[i] > 60 else RD)
        f.t(640 + BW + 12, ry + 26, "正主 %.0f%%" % share[i], col, bold=True,
            size=17)
        if g > 96:
            f.t(640 + g / 2.0, ry + 25, "正主拿到的", GR, size=14,
                anchor="middle")
        if BW - g > 120:
            f.t(640 + g + (BW - g) / 2.0, ry + 25, "漏给干扰的", RD, size=14,
                anchor="middle")

    f.t(510, top + PH3 - 78,
        "⭐ 生活里的同一件事：<tspan font-weight=\"700\">"
        "一个人在台下小声嘀咕，你听不见；一万个人同时小声嘀咕，"
        "台上的人就喊不过了。</tspan>", INK, size=16)
    f.t(510, top + PH3 - 52,
        "⛔ 而长上下文是<tspan font-weight=\"700\">两头夹击</tspan>："
        "嘀咕的人变多（N 涨），而且最吵的那个还离得更近了（②里的夹角在变小）。",
        RD, size=16)

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH3 + 30
    yy = f.band(yy, "info", "这张图一次回答了本讲三个「为什么」", [
        "<tspan font-weight=\"700\">为什么每个头是 128 维，不是 8 维？</tspan>"
        "——&#160;因为容量不是线性的。8 维里丢 1000 根，最挤的一对只剩 %.0f°"
        "（几乎是同一根）；128 维里丢 1 万根还有 %.0f°。" % (A8, ANG[2]),
        "<tspan font-weight=\"700\">为什么 MLA 压到 512 维还能用？</tspan>"
        "——&#160;因为它要的从来不是「完全正交」。"
        "这正是「白送的那段 vs 赌的那段」里<tspan font-weight=\"700\">赌的那段</tspan>"
        "赌的东西：赌压完之后大家<tspan font-weight=\"700\">还够不像</tspan>。",
        "<tspan font-weight=\"700\">为什么上下文越长越容易记混？</tspan>"
        "——&#160;③ 那三根条就是答案。⛔ 注意这不是「模型偷懒」，"
        "是 softmax 把所有人的票加起来等于 1 的<tspan font-weight=\"700\">直接后果</tspan>。",
    ])
    yy = f.band(yy + 14, "ok", "顺手接住后面讲稀疏那一节：稀疏注意力到底在省什么", [
        "如果记混是「干扰项太多」造成的，那<tspan font-weight=\"700\">"
        "少看几个反而可能更准</tspan> ——&#160;这不是省钱的副作用，是它自己的好处。",
        "⭐ DSA 在 128K 里只挑 2048 个（1.5625%），NSA 分三路挑。"
        "本讲前面把它们讲成「为了省」，③ 这三根条说明"
        "<tspan font-weight=\"700\">它们同时也在降噪</tspan>。",
    ])
    yy = f.src(yy + 16,
               "⭐ 图里所有数字都是本课当场算的（numpy；随机单位向量，d=128；"
               "夹角量的是最挤的那一对），脚本里带断言 ——&#160;不是引来的",
               "⚠️ 但<tspan font-weight=\"700\">想法不是本课的</tspan>："
               "「差不多正交能装指数多个」是 Johnson–Lindenstrauss 引理的推论；"
               "「叠加（superposition）」这套解释出自 Anthropic Elhage 等"
               "《Toy Models of Superposition》（arXiv 2209.10652）",
               "⚠️ 串扰那格的 logit ＝ √d·cos 是<tspan font-weight=\"700\">"
               "推导来的不是测来的</tspan>：q、k 各分量 O(1) 时内积 ≈ d·cos，"
               "注意力再除以 √d，剩下 √d·cos。真实模型里 q、k 的模长会被"
               "训练调整，所以这一格读<tspan font-weight=\"700\">趋势</tspan>，"
               "别读绝对值")
    f.save("fig3-capacity.svg", yy + 6)


main()
