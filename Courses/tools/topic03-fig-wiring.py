# -*- coding: utf-8 -*-
r"""专题三 · §五 开篇「四种存法，先只看接线」

⭐⭐⭐ 2026-09-14 R59 新画。这一格在本课里是**真的空着**，而且空得很典型：

   §五 手上已经有三张讲这四种存法的图，可它们**都不是入口**——
     · fig3-knob1      是**散点**：横轴一份占多少地方、纵轴换回多少能力。
                       ⛔ 它默认你已经知道「一份」指的是什么。
     · fig3-copy-matrix 是**矩阵**：把 GQA 的「复制」一格一格画出来。
                       ⛔ 它默认你已经知道谁在跟谁共用。
     · fig3-absorb      是**代数搬家**：W^UK 从缓存那侧挪到 q 那侧。
                       ⛔ 它默认你已经接受了 MLA 存的不是 K/V。

   ⭐ 三张图都从「第二步」起讲。**第一步是一张接线图**：
     几个头、几份 K/V、谁连谁、最后**哪几个盒子要跟着这段对话一直留着**。

⭐⭐ 所以这张图有一条硬规矩：**零公式、零数字。**
   不写维度、不写 GiB、不写头数，连「GQA-2」的那个 2 都不写 ——
   ⛔ 一出现数字，读者就会去算，而这一格要的是**先看懂形状**。
   要算的账在下一张（fig3-knob1）里，那里每个数都核过。

📌 判断「省在哪」只看最下面那排行李箱：
   **算力四家几乎一样，差别全在「跨 token 得留下来多少」。**
   ⭐ MLA 那一格的题眼是：箱子里装的**根本不是 K/V**，
     是一份能在用的时候现场展开成 K/V 的压缩件 —— 展开出来的那几个
     画成虚线，因为它们**用完就扔，不进箱子**。

⚠️ 本图不作任何效果好坏的判断（谁掉点、谁不掉点）——&nbsp;那是 fig3-knob1
   纵轴的事，而那根轴本课已经明确画成两档定性、不画连续刻度。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400
NH = 4                     # 画四个头 —— 少了看不出分组，多了挤
PW = 338                   # 每格宽
GAP = 12


def main():
    f = Fig(W, "四种 KV 存法的接线图：MHA 每头一份、GQA 几个头合用一份、"
               "MQA 全部头合用一份、MLA 只存一份压缩件用时现场展开。"
               "零公式零数字，只看谁连谁、以及最后哪几个盒子要留下来")
    f.marks = set()
    y0 = f.header(
        "四种存法 ——　<tspan font-weight=\"700\">先只看接线，一个数都不看"
        "</tspan>",
        "⭐ <tspan font-weight=\"700\">四家的算力几乎一样，差别全在「这段对话要一直留多少」"
        "</tspan>　——　最下面那排行李箱就是各自的留存量。"
        "<tspan font-weight=\"700\">省多少、值不值</tspan>，下一张才算。",
        [(GY, "Q 头"), (INK, "实线盒子＝要进箱子"),
         (GY2, "虚线盒子＝用完就扔"), (PU, "MLA 的压缩件")])

    PH = 424
    ay = f.panel(0, y0, W, PH,
                 "同一个模型、同样多的头，只换「K/V 怎么存」这一件事", INK,
                 sub="每一格从上往下读：一排头 → 它们各自去问谁要 K/V "
                     "→ 最后哪几个盒子进箱子")

    # ── 一格的公共画法 ────────────────────────────────────────────
    #   qy  头那一排的圆心
    #   ky  K/V 盒子那一排的顶边
    #   by  行李箱那一排的顶边
    # ⛔ QY 不能再往上 —— 小标题（ay+62）的下沿离 Q 圆的顶边只剩 13px，
    #   MLA 那格「谁也不直接存 K/V」最长，再高一点就从圆底下穿过去。
    QY, KY, BY = ay + 94, ay + 158, ay + 276
    QR = 15

    def head_row(x0, n=NH):
        """一排 query 头。⛔ 四格必须完全一致 —— 变的只能是下面的连线。"""
        step = 62
        x = x0 + (PW - (n - 1) * step) / 2.0
        pts = []
        for i in range(n):
            cx = x + i * step
            f.p.append('<circle cx="%.1f" cy="%.1f" r="%d" fill="#fff" '
                       'stroke="%s" stroke-width="1.8"/>' % (cx, QY, QR, GY2))
            f.t(cx, QY + 5, "Q", GY, True, 14, "middle")
            pts.append(cx)
        return pts

    def kv_row(x0, n, col, dash=None, label="K V"):
        """一排 K/V 盒子。dash 不为空 ＝ 现场展开、用完就扔。"""
        bw, bh, step = 46, 40, 62
        x = x0 + (PW - (n - 1) * step - bw) / 2.0
        pts = []
        for i in range(n):
            bx = x + i * step
            f.box(bx, KY, bw, bh, "#fff", col, 7, 1.8, dash=dash)
            f.t(bx + bw / 2.0, KY + 26, label, col, True, 15, "middle")
            pts.append(bx + bw / 2.0)
        return pts

    def wire(qs, ks, col, dash=None):
        """把每个头连到它该去的那个盒子。⭐ 这些线就是整张图的全部内容。"""
        for i, qx in enumerate(qs):
            kx = ks[i * len(ks) // len(qs)]
            f.line(qx, QY + QR + 2, kx, KY - 3, col, 1.5, dash=dash,
                   arrow=False)

    def trunk(x0, n, col):
        """行李箱：跨 token 真正要留下来的东西。⭐ 题眼全在这一排。"""
        bw, step = 40, 52
        x = x0 + (PW - (n - 1) * step - bw) / 2.0
        for i in range(n):
            f.icon("box", x + i * step, BY, bw, 44, col, "#fff")
        return x

    # ── 四格 ──────────────────────────────────────────────────────
    #   ⛔ 顺序按「共用得越来越狠」排：各存各的 → 分组共用 → 全体共用
    #     → 干脆不存 K/V。⭐ MLA 摆最后，因为它不在那条连续谱上。
    CASE = [
        ("MHA", GY2, NH, "每个头各问各的",
         "谁也不共用 ——　这就是基准线，没省。"),
        ("GQA", OR, 2, "几个头合用一份",
         "头还是那么多，⭐ 箱子里的盒子变少了。"),
        ("MQA", RD, 1, "所有头合用同一份",
         "共用到头了，⛔ 也最容易掉点。"),
        ("MLA", PU, None, "谁也不直接存 K/V",
         "⭐ 箱子里那个<tspan font-weight=\"700\">不是 K/V</tspan>。"),
    ]
    for i, (nm, col, nk, sub, note) in enumerate(CASE):
        x0 = 14 + i * (PW + GAP)
        f.box(x0, ay + 6, PW, 386, "#fff", col, 10)
        f.box(x0, ay + 6, PW, 5, col, col, 3)
        f.box(x0, ay + 4, PW, 5, "#fff", "#fff", 0)
        f.t(x0 + 18, ay + 40, nm, col, True, 23)
        f.t(x0 + 18, ay + 62, sub, GY, size=15)

        qs = head_row(x0)
        if nm == "MLA":
            # ⭐ MLA 这一格是唯一多一层的：头 → 一份压缩件 → 现场展开的 K/V。
            #   ⛔ 展开出来的那排画虚线，且**不进箱子** —— 这一笔是全图的题眼，
            #     少了它，MLA 看起来就只是「另一种 MQA」。
            cx = x0 + PW / 2.0
            f.box(cx - 54, KY - 2, 108, 44, "#f6f0fd", col, 8, 1.8)
            f.t(cx, KY + 25, "压缩件", col, True, 16, "middle")
            for qx in qs:
                f.line(qx, QY + QR + 2, cx, KY - 5, col, 1.5, arrow=False)
            # 现场展开：虚线盒子摆在压缩件和箱子之间，靠右侧一小排
            ex = x0 + 26
            for k in range(NH):
                bx = ex + k * 72
                f.box(bx, KY + 58, 56, 30, "#fff", GY2, 6, 1.4, dash="4 4")
                f.t(bx + 28, KY + 78, "K V", GY2, size=13, anchor="middle")
                f.line(cx, KY + 44, bx + 28, KY + 56, GY2, 1.1, dash="3 3",
                       arrow=False)
            f.t(x0 + PW / 2.0, KY + 102, "用的时候现场展开，用完就扔", GY2,
                size=14, anchor="middle")
            trunk(x0, 1, col)
        else:
            ks = kv_row(x0, nk, col)
            wire(qs, ks, col)
            trunk(x0, nk, col)

        f.t(x0 + PW / 2.0, BY + 62, "跨 token 要留下来的", GY2, size=14,
            anchor="middle")
        f.t(x0 + 18, BY + 96, note, GY, size=15, w=PW - 36)

    # ── 落点 ──────────────────────────────────────────────────────
    yy = y0 + PH + 22
    yy = f.band(yy, "info", "⭐⭐ 这一排箱子，就是后面所有账的那个「一份」", [
        "四家<tspan font-weight=\"700\">该做的乘加几乎一样多</tspan> ——&#160;"
        "省下来的不是算力，是<tspan font-weight=\"700\">"
        "「这段对话从头到尾得一直背着的东西」</tspan>。",
        "⭐ MLA 那一格请多看两眼：箱子里只有一个，"
        "<tspan font-weight=\"700\">而且它不是 K/V</tspan>。"
        "每个头真正要用的 K/V 是从它<tspan font-weight=\"700\">现场展开</tspan>"
        "出来的 ——&#160;那几个虚线盒子用完就扔，不占箱子。",
        "⛔ 所以 MLA 不是「共用得更狠的 MQA」。"
        "<tspan font-weight=\"700\">前三家在同一条谱上（各存各的 → 分组 → 全体），"
        "MLA 换了一条路。</tspan>",
    ])
    yy = f.band(yy + 14, "warn", "这张图故意不回答的两件事", [
        "<tspan font-weight=\"700\">① 省多少。</tspan>"
        "看 <a href=\"#fig-knob1\">下一张散点图</a> ——&#160;"
        "那里横轴是「一份占多少地方」，每个数本课都当场算过。"
        "⚠️ 那张图有个反直觉的题眼：<tspan font-weight=\"700\">"
        "MQA 比 MLA 存得还少</tspan>。",
        # ⛔ 这里**不能**写 href="#s5-4"：§五 的标题是合并的
        #   「5.1 ＋ 5.2 ＋ 5.3 ＋ 5.4」，anchorize 把四个号全指到同一个
        #   id `s5-1`，`s5-4` 这个 id 根本不存在。⭐ 指图的 fid 最稳。
        "<tspan font-weight=\"700\">② 凭什么能这么省、以及谁掉点。</tspan>"
        "「共用」那一步到底是什么，看 <a href=\"#fig-copy-matrix\">那张把复制"
        "矩阵一格一格画出来的图</a>；MLA 展开那一步为什么不白费力气，"
        "看 <a href=\"#fig-absorb\">「吸收」那张图</a>。",
    ])
    yy = f.src(yy + 16,
               "⭐ 这张图<tspan font-weight=\"700\">不含任何数</tspan>，"
               "所以也没有需要核的口径 ——&#160;它画的是四种做法的"
               "<tspan font-weight=\"700\">接线关系</tspan>，"
               "四家各自的出处（MQA arXiv 1911.02150、GQA arXiv 2305.13245、"
               "MLA DeepSeek-V2 arXiv 2405.04434）记在后面那几张有数的图上",
               "⚠️ 头的个数画成 4 只是为了一眼能数完，"
               "<tspan font-weight=\"700\">跟任何真实模型的头数都无关</tspan>；"
               "GQA 画成两组同理 ——&#160;真实分组数看讲四种存法的那张散点图",
               "⛔ 图里不区分 K 和 V 各自的份数（它们在这四家里都是同进同出），"
               "一个盒子代表「一个头位置上的 K 和 V 合起来的那一份」")
    f.save("fig3-wiring.svg", yy + 6)


main()
