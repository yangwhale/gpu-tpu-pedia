# -*- coding: utf-8 -*-
r"""专题三 · §2「为什么是现在」—— 同一个模型，两个场景，主角换人了

⭐⭐⭐ 2026-09-13 夜间 R20 新画。装置偷自知乎 姜富春《彻底理解 MLA》
   （zhuanlan.zhihu.com/p/16730036197）：**同一个模型、只把 batch 和上下文拧一下，
   瓶颈就从「参数」换成了「KV cache」。**

⭐ 为什么这个装置值钱：它回答的是一个**时间线上的怪事** ——
   这个形状 2017 年就造出来了，2019 年 MQA 那篇就指着它说是问题，
   可真正全行业动手改，是 2024 年以后。**中间那几年，技术一个字没变。**
   变的是**工作负载**。这张图把「变的是什么」画出来。

⛔ 画法上唯一的关键：**两根柱子里「权重」那一段必须像素级相同**
   ——&nbsp;同色、同高、同位置。读者要看到的是「什么都没改，主角却换了人」。
   如果两段高度稍有不同，这张图就废了。

📌 所有数都是本课前面已经核过的，这里只做除法：
   · 权重 625 GiB（671B 原生 FP8，1 B/参数）
   · MLA 的 KV：61 层 · 128K · bf16 · 一个用户 ＝ 8.58 GiB
   · 反事实对照：同一个模型假如用 MHA ＝ 488 GiB（同口径）
   ⚠️ 488 / 8.58 ＝ 56.9，正好对上 §五 那个 4.571 × 12.4 —— 互为交叉验证。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2, wpx, wrap_rich)

W = 1400
WEIGHT = 625.0        # GiB，671B 原生 FP8
KV128K = 8.58         # GiB，MLA · 61 层 · 128K · bf16 · 一个用户
KVMHA = 488.0         # GiB，同口径的反事实：假如它用 MHA
DEV = 94.74           # GiB / device，TPU v7


def main():
    A = KV128K * (4 / 128.0) * 1          # 一个人 · 4K
    B = KV128K * 1.0 * 64                 # 64 个人 · 128K
    # ⛔ 这个等式是这张图和 §五 之间的交叉验证，对不上就说明有一处口径错了。
    assert abs(KVMHA / KV128K - 56.9) < 0.1, KVMHA / KV128K
    nA = int(-(-(WEIGHT + A) // DEV))
    nB = int(-(-(WEIGHT + B) // DEV))
    assert nB > nA

    f = Fig(W, "同一个模型，两个场景：瓶颈从参数换成了 KV cache")
    yy = f.header(
        "为什么是现在 ——&#160;同一个模型，两个场景，<tspan font-weight=\"700\">"
        "主角换人了</tspan>",
        "这个形状 2017 年就造出来了，2019 年就有人指着它说是问题，"
        "可全行业真正动手改是 2024 年以后。"
        "<tspan font-weight=\"700\">中间那几年，技术一个字没变。</tspan>"
        "——&#160;这张图画的是那个变了的东西。",
        legend=[(GY2, "模型权重（两边完全一样）"), (RD, "KV cache"),
                (BL, "要几张卡")])

    # ══ ① 两根柱子 ════════════════════════════════════════════════
    PH1 = 470
    top = f.panel(0, yy, W, PH1,
                  "① 同一个模型（DeepSeek-V3）。"
                  "<tspan font-weight=\"700\">权重那一段一个字节没动。</tspan>",
                  RD, tag="61 层 · bf16 · MLA")
    BASE = top + 348                       # 地面
    SCALE = 300.0 / (WEIGHT + B)           # 最高那根柱占 300px
    BW = 190
    for i, (ttl, sub_, kv, nd) in enumerate([
        ("一个人，短对话", "上下文 4K", A, nA),
        ("64 个人，长文档", "上下文 128K", B, nB),
    ]):
        cx = 180 + i * 330
        hw = WEIGHT * SCALE
        hk = kv * SCALE
        # ⭐ 权重那一段：两根柱子里必须完全一样 —— 同一个表达式算出来的
        f.box(cx, BASE - hw, BW, hw, "#f1f3f4", GY2, 6)
        f.t(cx + BW / 2, BASE - hw / 2 - 6, "权重", GY, bold=True, size=18,
            anchor="middle")
        f.t(cx + BW / 2, BASE - hw / 2 + 18, "%.0f GiB" % WEIGHT, GY, size=16,
            anchor="middle")
        if hk >= 14:
            f.box(cx, BASE - hw - hk, BW, hk, "#fce8e6", RD, 6)
            f.t(cx + BW / 2, BASE - hw - hk / 2 - 6, "KV cache", RD, bold=True,
                size=18, anchor="middle")
            f.t(cx + BW / 2, BASE - hw - hk / 2 + 18, "%.0f GiB" % kv, RD,
                size=16, anchor="middle")
        else:
            f.box(cx, BASE - hw - 3, BW, 3, RD, RD, 1)
            f.t(cx + BW / 2, BASE - hw - 14,
                "KV cache　%.2f GiB（细到画不出来）" % kv, RD, bold=True,
                size=16, anchor="middle")
        f.line(cx, BASE + 1, cx + BW, BASE + 1, LINE, 1.2, arrow=False)
        f.t(cx + BW / 2, BASE + 30, ttl, INK, bold=True, size=19,
            anchor="middle", cls="svglbl")
        f.t(cx + BW / 2, BASE + 54, sub_, GY2, size=15, anchor="middle")
        f.t(cx + BW / 2, BASE + 82, "要 %d 张卡" % nd, BL, bold=True, size=18,
            anchor="middle")

    # 右边：三个数
    RX = 780
    f.box(RX, top + 28, 596, 300, "none", LINE, 9)
    f.t(RX + 20, top + 60, "把这两根柱子读成三个数", INK, bold=True, size=19,
        cls="svglbl")
    for j, (lab, val, col) in enumerate([
        ("权重", "%.0f GiB  →  %.0f GiB　（一个字节没动）"
         % (WEIGHT, WEIGHT), GY),
        ("KV cache", "%.2f GiB  →  %.0f GiB　（涨了 %.0f 倍）"
         % (A, B, B / A), RD),
        ("KV 占总量", "%.2f%%  →  %.1f%%" % (100 * A / (WEIGHT + A),
                                             100 * B / (WEIGHT + B)), RD),
        ("要几张卡", "%d  →  %d　（翻了将近一倍）" % (nA, nB), BL),
    ]):
        yj = top + 100 + j * 46
        f.t(RX + 20, yj, lab, col, bold=True, size=17)
        f.t(RX + 150, yj, val, INK if j else GY, size=17, mono=True)
    f.t(RX + 20, top + 300,
        "⭐⭐ <tspan font-weight=\"700\">什么都没改。</tspan>"
        "是<tspan font-weight=\"700\">工作负载</tspan>变了，不是技术变了。",
        INK, size=17)

    # ══ ② 而这已经是 MLA 之后的数字 ════════════════════════════════
    yy = top + PH1 + 26
    PH2 = 400   # ⚠️ R43 从 262 涨上来：这一格从两张文字卡片换成了四行尺子
    top = f.panel(0, yy, W, PH2,
                  "② 别忘了：上面那根红柱子<tspan font-weight=\"700\">已经是 MLA "
                  "压过之后</tspan>的 ——&#160;如果它用 MHA 呢", BL,
                  tag="同口径的反事实")
    # ⭐⭐⭐ 2026-09-14 R43 重画。原来这一格是**两张写着数字的卡片**，
    #   而它要说的那句话是「这已经不是『要几张卡』，是『做不了』」——
    #   **「做不了」是个程度，程度得画出来才有分量，写出来只是一个形容词。**
    #
    # ⭐⭐ 装置：**四个场景摆到同一根「要几张卡」的尺子上，让第四根自己冲出图外。**
    #   ⭐ 这跟面板① 是同一个手法的另一头 —— 面板① 用「0.27 GiB，细到画不出来」
    #     表示小，这一格用「337 张卡，大到画不下」表示大。
    #     **同一张图，一头细到画不出来，一头大到画不下。**
    #
    # ⭐⭐⭐ 而摆上尺子之后，掉出来一句本课以前没说过的话：
    #   **MHA 伺候一个人要 12 张卡，MLA 伺候 64 个人要 13 张卡。**
    #   差一张卡，服务的人数差 64 倍 —— 这比「56.9 倍」具体得多，
    #   而且它是这张图自己的四个数做除法得出来的，没引进任何新口径。
    SCEN = [
        ("MLA", "一个人", 1, KV128K, GR),
        ("MHA", "一个人", 1, KVMHA, RD),
        ("MLA", "64 个人", 64, KV128K, GR),
        ("MHA", "64 个人", 64, KVMHA, RD),
    ]
    # ⚠️ 这四个数跟面板① 的 nA/nB **不是同一套口径**：nA 是「4K 上下文」，
    #   这一格四行全是 128K。所以 cards[0]＝7 ≠ nA＝7 只是巧合，别拿来互相断言。
    cards = [int(-(-(WEIGHT + kv * n) // DEV)) for _, _, n, kv, _ in SCEN]
    assert cards[2] == nB, (cards[2], nB)      # 这一对才是同口径的（MLA · 64 人 · 128K）
    # ⭐ 下面两条断言就是上面那句话的依据，改任何一个常数它们会先炸
    assert cards == [7, 12, 13, 337], cards
    assert cards[2] - cards[1] == 1, cards     # 差一张卡
    AX0, AXW, AXMAX = 300, 1040.0, 20          # 尺子只画到 20 张卡
    PXC = AXW / AXMAX
    over = cards[3] - AXMAX
    wides = over * PXC / W                     # 画不下的部分折合几个图宽
    assert 11 < wides < 12.5, wides            # ＝ 317 张卡 × 52 px ÷ 1400

    f.t(30, top + 52,
        "把四个场景摆到<tspan font-weight=\"700\">同一根尺子</tspan>上 ——&#160;"
        "横轴就是「要几张卡」（权重 %.0f GiB ＋ KV，按 TPU v7 每 device "
        "%.2f GiB 向上取整）。" % (WEIGHT, DEV), GY, size=16)
    for i in range(0, AXMAX + 1, 4):
        x = AX0 + i * PXC
        f.line(x, top + 74, x, top + 296, LINE2, 1, arrow=False)
        f.t(x, top + 70, "%d" % i, GY2, size=13, anchor="middle")
    f.t(AX0 - 10, top + 70, "张卡", GY2, size=13, anchor="end")

    ROWY = []
    for i, (kind, who, n, kv, col) in enumerate(SCEN):
        y = top + 86 + i * 54
        ROWY.append(y)
        f.t(AX0 - 16, y + 27, "%s ·　%s" % (kind, who), col, True, 17, "end")
        c = cards[i]
        if c <= AXMAX:
            f.box(AX0, y, c * PXC, 38, "#fce8e6" if col is RD else "#e6f4ea",
                  col, 5, 1.6)
            f.t(AX0 + c * PXC + 14, y + 26, "%d 张卡" % c, col, True, 19)
        else:
            # ⛔ 冲出图外那一根：**不要缩放坐标去把它塞进来** ——
            #   一缩放，前三根就全挤成一团，而「塞不下」这件事恰恰是论点。
            # ⭐ 箭头尖故意顶到**图框上**（W-8），不是停在刻度尽头 ——
            #   停在 20 那一格看起来像「正好装满」，顶到框上才是「装不下」。
            f.box(AX0, y, AXW + 18, 38, "#fce8e6", RD, 5, 1.6)
            f.poly([(AX0 + AXW + 18, y), (W - 8, y + 19),
                    (AX0 + AXW + 18, y + 38)], "#fce8e6", RD, 1.6)
            f.t(AX0 + 20, y + 26,
                "%d 张卡 ——&#160;<tspan font-weight=\"700\">"
                "按这把尺子还要再往右画 %.1f 个图宽</tspan>" % (cards[3], wides),
                RD, True, 17)
    # ⭐ 把「差一张卡」这件事用一个括号括起来 ——&#160;光靠两行数字读者不会自己去减。
    # ⚠️ 括号竖边落在 bx1+92 而不是紧贴条尾：第 2、3 行的「12 张卡」「13 张卡」
    #   标注就挂在条尾右边，实测各自伸到 998 / 1052，贴着画会压上去。
    # ⭐ 竖边的位置是**从标注实际有多宽算出来的**，不是拍脑袋加个偏移 ——
    #   「13 张卡」比「12 张卡」宽不了多少，但只要贴着画就会被横线穿过去。
    lab_end = max(AX0 + cards[r] * PXC + 14 + wpx("%d 张卡" % cards[r], 19)
                  for r in (1, 2))
    vx = lab_end + 26
    f.line(vx, ROWY[1] + 19, vx, ROWY[2] + 19, PU, 1.6, arrow=False)
    for r in (1, 2):
        f.line(lab_end + 8, ROWY[r] + 19, vx, ROWY[r] + 19, PU, 1.4,
               arrow=False)
    NOTE = "⭐ 差一张卡，人数差 64 倍"
    assert vx + 14 + wpx(NOTE, 18) < W - 20, vx + 14 + wpx(NOTE, 18)
    f.t(vx + 14, ROWY[1] + 52,
        "⭐ 差<tspan font-weight=\"700\">一张卡</tspan>，"
        "人数差 <tspan font-weight=\"700\">64 倍</tspan>", PU, True, 18)

    # ⭐⭐ 这句是摆上尺子之后**掉出来的**，本课以前没说过 ——&#160;
    #   它比「56.9 倍」具体，因为它说的是同一台机器上的两件事。
    f.t(30, top + 328,
        "⭐⭐ 把第 2、3 行叠在一起读："
        "<tspan font-weight=\"700\">MHA 伺候一个人要 12 张卡，"
        "MLA 伺候 64 个人要 13 张卡。</tspan>"
        "同样一台机器 ——&#160;一个换六十四个。", INK, size=17)

    f.t(30, top + PH2 - 44,
        "⭐⭐ 所以 MLA 那个 <tspan font-weight=\"700\">56.9×</tspan>"
        "（＝ %.0f ÷ %.2f）不是一次「优化」——&#160;"
        "<tspan font-weight=\"700\">它是把这件事从「做不了」变成「做得了」。</tspan>"
        % (KVMHA, KV128K), INK, size=17)

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH2 + 30
    yy = f.band(yy, "info", "这张图解释了一条时间线上的怪事", [
        "<tspan font-weight=\"700\">2017</tspan> 年这个形状就造出来了；"
        "<tspan font-weight=\"700\">2019</tspan> 年 MQA 那篇论文的摘要里"
        "就写着「incremental inference is often slow, due to the "
        "memory-bandwidth cost of repeatedly loading the large keys and "
        "values tensors」——&#160;<tspan font-weight=\"700\">问题早就被指出来了</tspan>。",
        "⛔ 可全行业真正动手改，是 <tspan font-weight=\"700\">2024</tspan> 年以后。"
        "中间那几年，<tspan font-weight=\"700\">技术一个字没变</tspan>。",
        "⭐⭐ 变的是这张图右边那根柱子：<tspan font-weight=\"700\">"
        "上下文从几千涨到十几万，同时服务的人从一个涨到几十个</tspan>。"
        "两个都在乘，而它们乘的是<tspan font-weight=\"700\">同一项</tspan>。",
    ])
    yy = f.band(yy + 14, "ok", "顺手给一条提问顺序，本讲后面一直在用", [
        "<tspan font-weight=\"700\">问「省了多少」之前，先问「省的是哪一样」。</tspan>"
        "权重是<tspan font-weight=\"700\">所有人共享一份</tspan>，"
        "KV cache 是<tspan font-weight=\"700\">每人一份</tspan> ——&#160;"
        "它们根本不是同一类开销。",
        "⭐ 所以 KV cache 不是「显存里的一项」，它直接决定"
        "<tspan font-weight=\"700\">你能同时服务多少人</tspan>。"
        "<tspan font-weight=\"700\">这一条到专题六会变成 batch size 的硬上限。</tspan>",
    ])
    yy = f.src(yy + 16,
               "装置偷自知乎 姜富春《deepseek 技术解读(1)－彻底理解 MLA》"
               "（zhuanlan.zhihu.com/p/16730036197）——&#160;"
               "他用 Qwen-72B 做的这个对照，本图换成本讲一直在用的 V3 口径重算",
               "⚠️ 数全部来自本讲前面已核过的三个：权重 625 GiB（671B 原生 FP8）·"
               "MLA 的 KV 8.58 GiB（61 层 · 128K · bf16 · 一个用户）·"
               "反事实 MHA 488 GiB（同口径）。"
               "<tspan font-weight=\"700\">488 ÷ 8.58 ＝ 56.9</tspan>，"
               "正好对上 MLA 那个 4.571 × 12.4 ——&#160;两条路算出同一个数，互为交叉验证",
               "📌 「要几张卡」按 <tspan font-weight=\"700\">TPU v7 每 device "
               "94.74 GiB</tspan> 算（这个数本课的 AOT 工具链里核过："
               "编译器自己报的 95.38G − 94.74G ＝ 656.93M，只有按 1024 才成立）。"
               "⚠️ 换别的硬件只是这两根柱子的刻度变，"
               "<tspan font-weight=\"700\">结论不变</tspan>；"
               "而且这笔账<tspan font-weight=\"700\">只算了装得下装不下，没算带宽</tspan>")
    f.save("fig3-flip.svg", yy + 6)


main()
